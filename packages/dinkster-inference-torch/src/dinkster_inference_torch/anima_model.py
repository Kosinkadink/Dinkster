"""Native torch Anima diffusion transformer.

Transcribed from comfy/ldm/anima/model.py @ 82f839f5e737d8bfce480872b
a05e5a430f2526f: a Cosmos Predict2 backbone whose cross-attention
context comes from a six-block LLM adapter that maps Qwen3-0.6B hidden
states onto T5 vocabulary positions. The adapter's target and model
widths are equal in the supported profile, so the reference's identity
in_proj has no module here, and attention masks (unused by the Anima
pipeline) are not ported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference import ANIMA_CONFIG, AnimaConfig

from .attention import AttentionKernel, select_attention
from .cosmos_predict2 import CosmosPredict2Geometry, CosmosPredict2Model
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations

_DEFAULT_DIT_ATTENTION = select_attention("flux").kernel
_DEFAULT_ADAPTER_ATTENTION = select_attention("qwen").kernel
_NORM_EPS = 1e-6
_ADAPTER_MLP_RATIO = 4
_ADAPTER_ROPE_THETA = 10000.0
# Anima trains its backbone rope with 4x spatial extrapolation.
_ANIMA_ROPE_H_EXTRAPOLATION = 4.0
_ANIMA_ROPE_W_EXTRAPOLATION = 4.0
_ANIMA_ROPE_T_EXTRAPOLATION = 1.0
# Adapter output rows are zero-padded up to the T5-XXL context length.
_T5_CONTEXT_ROWS = 512


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _adapter_rotary_embedding(
    length: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-style interleaved-halves rotary table as ([1, length, head_dim],) cos and sin."""
    inverse_frequencies = 1.0 / (
        _ADAPTER_ROPE_THETA
        ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim)
    )
    positions = torch.arange(length, dtype=torch.float32, device=device)
    frequencies = torch.outer(positions, inverse_frequencies)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        angles.cos().to(dtype).unsqueeze(0),
        angles.sin().to(dtype).unsqueeze(0),
    )


def _apply_adapter_rope(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cosine, sine = rotary
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return x * cosine + _rotate_half(x) * sine


class AnimaAdapterAttention(torch.nn.Module):
    """One adapter attention projection with rope on queries and keys."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        head_dim: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = operations.linear(query_dim, inner, bias=False)
        self.q_norm = operations.rms_norm(head_dim, eps=_NORM_EPS)
        self.k_proj = operations.linear(context_dim, inner, bias=False)
        self.k_norm = operations.rms_norm(head_dim, eps=_NORM_EPS)
        self.v_proj = operations.linear(context_dim, inner, bias=False)
        self.o_proj = operations.linear(inner, query_dim, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        query_rotary: tuple[torch.Tensor, torch.Tensor],
        key_rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, length = x.shape[0], x.shape[1]
        query_shape = (batch, -1, self.heads, self.head_dim)
        kv_shape = (context.shape[0], -1, self.heads, self.head_dim)
        query = self.q_norm(self.q_proj(x).view(query_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value = self.v_proj(context).view(kv_shape).transpose(1, 2)
        query = _apply_adapter_rope(query, query_rotary)
        key = _apply_adapter_rope(key, key_rotary)
        output = self._attention_kernel(
            query,
            key,
            value,
            mask=None,
            causal=False,
            scale=None,
            enable_gqa=False,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.o_proj(output)


class AnimaAdapterBlock(torch.nn.Module):
    """Pre-norm self-attention, cross-attention, and GELU MLP with plain residuals."""

    def __init__(
        self,
        source_dim: int,
        width: int,
        heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        head_dim = width // heads
        self.norm_self_attn = operations.rms_norm(width, eps=_NORM_EPS)
        self.self_attn = AnimaAdapterAttention(
            width,
            width,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm_cross_attn = operations.rms_norm(width, eps=_NORM_EPS)
        self.cross_attn = AnimaAdapterAttention(
            width,
            source_dim,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm_mlp = operations.rms_norm(width, eps=_NORM_EPS)
        self.mlp = torch.nn.Sequential(
            operations.linear(width, _ADAPTER_MLP_RATIO * width),
            torch.nn.GELU(),
            operations.linear(_ADAPTER_MLP_RATIO * width, width),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        target_rotary: tuple[torch.Tensor, torch.Tensor],
        context_rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normalized = self.norm_self_attn(x)
        x = x + self.self_attn(normalized, normalized, target_rotary, target_rotary)
        normalized = self.norm_cross_attn(x)
        x = x + self.cross_attn(normalized, context, target_rotary, context_rotary)
        return x + self.mlp(self.norm_mlp(x))


class AnimaLLMAdapter(torch.nn.Module):
    """Maps text-encoder hidden states onto embedded T5 token positions."""

    def __init__(
        self,
        config: AnimaConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ADAPTER_ATTENTION,
    ) -> None:
        super().__init__()
        width = config.adapter_width
        if width != config.adapter_heads * config.adapter_head_dim:
            raise ValueError("adapter width must equal adapter heads times head dimension")
        self.head_dim = config.adapter_head_dim
        self.source_width = config.adapter_source_width
        self.embed = operations.embedding(config.adapter_vocabulary, width)
        self.blocks = torch.nn.ModuleList(
            AnimaAdapterBlock(
                config.adapter_source_width,
                width,
                config.adapter_heads,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.adapter_blocks)
        )
        self.out_proj = operations.linear(width, width)
        self.norm = operations.rms_norm(width, eps=_NORM_EPS)

    def forward(
        self, source_hidden_states: torch.Tensor, target_input_ids: torch.Tensor
    ) -> torch.Tensor:
        context = source_hidden_states
        if context.ndim != 3 or context.shape[2] != self.source_width:
            raise ValueError(f"source hidden states must be [B, rows, {self.source_width}]")
        if not context.is_floating_point():
            raise ValueError("source hidden states must use a floating-point dtype")
        if target_input_ids.ndim != 2 or target_input_ids.is_floating_point():
            raise ValueError("target input ids must be integer [B, rows]")
        if target_input_ids.shape[0] != context.shape[0]:
            raise ValueError("target input ids must match the source batch")
        if target_input_ids.device != context.device:
            raise ValueError("target input ids must be on the source device")
        x = self.embed(target_input_ids).to(context.dtype)
        target_rotary = _adapter_rotary_embedding(x.shape[1], self.head_dim, x.device, x.dtype)
        context_rotary = _adapter_rotary_embedding(
            context.shape[1], self.head_dim, x.device, x.dtype
        )
        prefetch = make_prefetch_queue(self.blocks)
        try:
            for block in self.blocks:
                prefetch_queue_pop(prefetch, block)
                x = block(x, context, target_rotary, context_rotary)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        return self.norm(self.out_proj(x))


class AnimaModel(CosmosPredict2Model):
    """Unregistered Anima source: the Predict2 backbone plus its LLM adapter.

    Subclassing keeps the backbone's state-dict keys at the top level,
    matching the checkpoint layout where only ``llm_adapter.*`` is added.
    """

    def __init__(
        self,
        config: AnimaConfig = ANIMA_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_DIT_ATTENTION,
        adapter_attention_kernel: AttentionKernel = _DEFAULT_ADAPTER_ATTENTION,
    ) -> None:
        if config.adapter_width != config.context_width:
            raise ValueError("adapter output width must equal the cross-attention context width")
        if config.hidden_width != config.attention_heads * config.attention_head_dim:
            raise ValueError("hidden width must equal attention heads times head dimension")
        patch_temporal, patch_height, patch_width = config.patch
        if patch_height != patch_width:
            raise ValueError("spatial patch dimensions must be square")
        patch_volume = patch_temporal * patch_height * patch_width
        if config.patchified_input_channels != (config.latent_channels + 1) * patch_volume:
            raise ValueError(
                "patchified input channels must cover the latent plus padding-mask"
                " channels times the patch volume"
            )
        geometry = CosmosPredict2Geometry(
            in_channels=config.latent_channels,
            out_channels=config.output_latent_channels,
            patch_spatial=patch_height,
            patch_temporal=patch_temporal,
            model_channels=config.hidden_width,
            num_blocks=config.blocks,
            num_heads=config.attention_heads,
            crossattn_emb_channels=config.context_width,
            adaln_lora_dim=config.adaln_lora_dim,
            rope_h_extrapolation_ratio=_ANIMA_ROPE_H_EXTRAPOLATION,
            rope_w_extrapolation_ratio=_ANIMA_ROPE_W_EXTRAPOLATION,
            rope_t_extrapolation_ratio=_ANIMA_ROPE_T_EXTRAPOLATION,
        )
        super().__init__(geometry, operations=operations, attention_kernel=attention_kernel)
        self.config = config
        self.llm_adapter = AnimaLLMAdapter(
            config,
            operations=operations,
            attention_kernel=adapter_attention_kernel,
        )

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        t5xxl_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.llm_adapter(text_embeds, text_ids)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        if out.shape[1] < _T5_CONTEXT_ROWS:
            out = F.pad(out, (0, 0, 0, _T5_CONTEXT_ROWS - out.shape[1]))
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        t5xxl_ids: torch.Tensor | None = None,
        t5xxl_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights)
        return super().forward(x, timesteps, context)


__all__ = [
    "AnimaAdapterAttention",
    "AnimaAdapterBlock",
    "AnimaLLMAdapter",
    "AnimaModel",
]
