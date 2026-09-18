"""Native Gemma 2/3/4 text towers and LTX-2 text-embedding projections.

The Gemma 3 path mirrors ComfyUI's text-only ``Llama2_`` tower at
b78cec87: sqrt(hidden)-scaled embeddings, additive RMSNorm weights,
four-norm residual blocks, and the repeating five-local/one-global
RoPE pattern. Gemma 2 uses global grouped-query attention throughout.
Gemma 4 uses its distinct global attention width, K-as-V global layers,
value normalization, layer scalar, and non-additive RMSNorm weights.
State keys match each ``model.*`` subtree after assembly strips that
source prefix; vision and multimodal projector siblings are not built.

The model returns the full hidden-state stack the LTX projections
consume: the embedding output plus every layer output, with the last
entry final-normed, stacked on a new dimension after the batch
(matching the reference ``intermediate_output="all"`` collection with
``final_layer_norm_intermediate=False``).

``LtxGemmaTextEncoder`` owns the behavior outside transformer math:
the prompt policy (template off by default, exactly like the LTXAV
conditioning nodes at the pin), trimming the stack to the attended
suffix of the left-padded row, and each projection's normalization -
the single projection range-normalizes over tokens and hidden then
folds the stack into the width for one bias-free Linear, and the dual
projection RMS-normalizes over hidden and feeds scaled video and audio
Linears whose outputs concatenate. The reference optionally casts the
stack to bfloat16 on devices where it prefers bf16; this port computes
in the model's dtype and returns float32. Image tokens are out of
scope: the encoder takes text only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from dinkster_inference import Conditioning
from dinkster_inference.gemma_text import (
    GEMMA3_LTX_12B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    LTX_TEXT_STACK_FEATURES,
    GemmaTextConfig,
    LtxGemmaPromptTokens,
    tokenize_ltx_gemma_prompt,
)

from .attention import AttentionKernel, select_attention
from .gemma_tokenizer import GemmaJsonTokenizer, GemmaSentencePieceTokenizer
from .ltx_connector import LtxTextConnectors
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import (
    INITLESS,
    Operations,
    ResidencyRouted,
    bound_compute_device,
    bound_compute_dtype,
    materialized_rms_norm_weight,
)
from .quant_linear import Int8Linear

# Gemma is a Llama-family text tower; it shares the qwen attention role
# rather than widening the fixed role set.
_DEFAULT_GEMMA_ATTENTION = select_attention("qwen").kernel


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE in the reference's operation order."""
    if type(frequencies) is torch.Tensor:

        def rotate(hidden: torch.Tensor) -> torch.Tensor:
            shape = hidden.shape
            pairs = hidden.reshape(*shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
            output = frequencies[..., 0] * pairs[..., 0] + frequencies[..., 1] * pairs[..., 1]
            return output.movedim(-1, -2).reshape(shape).type_as(hidden)

        return rotate(query), rotate(key)
    cosine, sine, negative_sine = frequencies
    q = query * cosine
    half = q.shape[-1] // 2
    q[..., :half].addcmul_(query[..., half:], negative_sine)
    q[..., half:].addcmul_(query[..., :half], sine)
    k = key * cosine
    k[..., :half].addcmul_(key[..., half:], negative_sine)
    k[..., half:].addcmul_(key[..., :half], sine)
    return q.to(query.dtype), k.to(key.dtype)


def _rope(
    head_dim: int,
    length: int,
    theta: float,
    scale: float,
    *,
    device: torch.device,
    rotary_fraction: float = 1.0,
    dtype: torch.dtype | None = None,
    precompute_inverse_on_cpu: bool = False,
    matrix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    rotary_pairs = int(rotary_fraction * head_dim // 2)
    inverse_device = None if precompute_inverse_on_cpu else device
    numerator = torch.arange(0, 2 * rotary_pairs, 2, device=inverse_device).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    if rotary_pairs < head_dim // 2:
        inverse = torch.cat(
            (inverse, torch.zeros(head_dim // 2 - rotary_pairs, device=inverse_device))
        )
    inverse = inverse / scale
    if precompute_inverse_on_cpu:
        inverse = inverse.to(device=device)
    positions = torch.arange(length, device=device).float()
    frequencies = inverse[:, None] @ positions[None, :]
    frequencies = frequencies.transpose(0, 1)
    if matrix:
        cosine = frequencies.cos()
        sine = frequencies.sin()
        return (
            torch.stack(
                (torch.stack((cosine, -sine), dim=-1), torch.stack((sine, cosine), dim=-1)),
                dim=-2,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(dtype=dtype)
        )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().unsqueeze(0).unsqueeze(0)
    sine = embedding.sin().unsqueeze(0).unsqueeze(0)
    if dtype is not None:
        cosine = cosine.to(dtype)
        sine = sine.to(dtype)
    half = head_dim // 2
    return cosine, sine[..., :half], -sine[..., half:]


def _rms_norm(module: torch.nn.RMSNorm, hidden: torch.Tensor, *, add_weight: bool) -> torch.Tensor:
    """Gemma RMSNorm over an operations-managed module."""
    if torch.is_grad_enabled():
        weight = module.weight + 1.0 if add_weight else module.weight
        weight = weight.to(hidden)
        return F.rms_norm(hidden, module.weight.shape, weight, module.eps)
    with materialized_rms_norm_weight(module) as weight:
        if add_weight:
            weight = weight.to(module.weight.dtype) + 1.0
        weight = weight.to(hidden)
        return F.rms_norm(hidden, weight.shape, weight, module.eps)


class GemmaAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: GemmaTextConfig,
        sliding: bool,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_GEMMA_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.sliding = sliding
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.num_key_value_heads
            if sliding
            else config.num_global_key_value_heads or config.num_key_value_heads
        )
        self.head_dim = config.head_dim if sliding else config.global_head_dim or config.head_dim
        self.enable_gqa = (
            config.architecture == "gemma2_lumina_2b" and self.num_heads != self.num_kv_heads
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        hidden = config.hidden_size
        query = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.q_proj = operations.linear(hidden, query, bias=False)
        self.k_proj = operations.linear(hidden, kv, bias=False)
        self.v_proj = (
            None
            if config.global_k_eq_v and not sliding
            else operations.linear(hidden, kv, bias=False)
        )
        self.o_proj = operations.linear(query, hidden, bias=False)
        self.q_norm = (
            operations.rms_norm(self.head_dim, eps=config.rms_norm_eps)
            if config.architecture != "gemma2_lumina_2b"
            else None
        )
        self.k_norm = (
            operations.rms_norm(self.head_dim, eps=config.rms_norm_eps)
            if config.architecture != "gemma2_lumina_2b"
            else None
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = (
            self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        value = (
            self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
            if self.v_proj is not None
            else key
        )
        if self.q_norm is not None:
            query = _rms_norm(self.q_norm, query, add_weight=self.config.rms_norm_add)
        if self.k_norm is not None:
            key = _rms_norm(self.k_norm, key, add_weight=self.config.rms_norm_add)
        if self.config.value_rms_norm:
            value = F.rms_norm(value, (self.head_dim,), eps=self.config.rms_norm_eps)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_rope(query, key, frequencies)
        enable_gqa = self.enable_gqa
        if self.num_heads != self.num_kv_heads and not enable_gqa:
            if self.config.architecture == "gemma4_ltx_12b" and (
                not self.sliding or length < self.config.sliding_window
            ):
                enable_gqa = True
            else:
                groups = self.num_heads // self.num_kv_heads
                key = key.repeat_interleave(groups, dim=1)
                value = value.repeat_interleave(groups, dim=1)
        output = self._attention_kernel(
            query,
            key,
            value,
            mask=mask,
            causal=False,
            scale=self.config.attention_scale,
            enable_gqa=enable_gqa,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class GemmaMlp(torch.nn.Module):
    def __init__(self, config: GemmaTextConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.gate_proj = operations.linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = operations.linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = operations.linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(hidden), approximate="tanh") * self.up_proj(hidden)
        )


class GemmaBlock(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: GemmaTextConfig,
        index: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_GEMMA_ATTENTION,
    ) -> None:
        super().__init__()
        self.sliding = config.sliding_pattern[index % len(config.sliding_pattern)]
        self.self_attn = GemmaAttention(
            config,
            self.sliding,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = GemmaMlp(config, operations=operations)
        self.input_layernorm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = operations.rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = operations.rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = operations.rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rms_norm_add = config.rms_norm_add
        if config.layer_scalar:
            self.register_buffer("layer_scalar", torch.empty(1))

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        sliding_mask: torch.Tensor | None,
        frequencies_global: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
        frequencies_local: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if self.sliding:
            frequencies = frequencies_local
            if sliding_mask is not None:
                mask = sliding_mask if mask is None else mask + sliding_mask
        else:
            frequencies = frequencies_global
        residual = hidden
        hidden = _rms_norm(self.input_layernorm, hidden, add_weight=self.rms_norm_add)
        hidden = self.self_attn(hidden, mask, frequencies)
        hidden = _rms_norm(self.post_attention_layernorm, hidden, add_weight=self.rms_norm_add)
        hidden = residual + hidden
        residual = hidden
        hidden = _rms_norm(self.pre_feedforward_layernorm, hidden, add_weight=self.rms_norm_add)
        hidden = self.mlp(hidden)
        hidden = _rms_norm(self.post_feedforward_layernorm, hidden, add_weight=self.rms_norm_add)
        hidden = residual + hidden
        if hasattr(self, "layer_scalar"):
            stored = self.get_buffer("layer_scalar")
            binding = self._offloaded_residency()
            if binding is None:
                scalar = stored.to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden * scalar
            else:
                with binding.lease() as lease:
                    hidden = hidden * lease.get("layer_scalar", dtype=hidden.dtype)
        return hidden


class GemmaTextModel(torch.nn.Module):
    """The exact model-scoped state layout shared with the detector."""

    def __init__(
        self,
        config: GemmaTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_GEMMA_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = operations.embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            GemmaBlock(config, index, operations=operations, attention_kernel=attention_kernel)
            for index in range(config.num_hidden_layers)
        )
        self.norm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the [batch, layers + 1, tokens, hidden] state stack."""
        if ids.ndim != 2:
            raise ValueError(f"Gemma ids must be [batch x tokens], got {tuple(ids.shape)}")
        if attention_mask is not None and attention_mask.shape != ids.shape:
            raise ValueError(
                "Gemma attention mask must match ids, got"
                f" {tuple(attention_mask.shape)} vs {tuple(ids.shape)}"
            )
        hidden = (self.embed_tokens(ids) if embeds is None else embeds).float()
        if hidden.shape != (*ids.shape, self.config.hidden_size):
            raise ValueError("Gemma preembedded input must match ids and hidden size")
        hidden = hidden * (self.config.hidden_size**0.5)
        length = ids.shape[1]
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(hidden.dtype).reshape(
                attention_mask.shape[0], 1, 1, length
            ).expand(attention_mask.shape[0], 1, length, length)
            mask_fill = torch.finfo(hidden.dtype).min
            if self.config.architecture == "gemma3_ltx_12b":
                mask_fill /= 4
            mask = mask.masked_fill(mask.to(torch.bool), mask_fill)
        if length > 1:
            mask_fill = torch.finfo(hidden.dtype).min
            if self.config.architecture == "gemma3_ltx_12b":
                mask_fill /= 4
            causal = torch.full(
                (length, length),
                mask_fill,
                dtype=hidden.dtype,
                device=hidden.device,
            ).triu_(1)
            mask = causal if mask is None else mask + causal
        sliding_mask = None
        if length > self.config.sliding_window:
            sliding_mask = torch.full(
                (length, length),
                torch.finfo(hidden.dtype).min,
                dtype=hidden.dtype,
                device=hidden.device,
            ).tril_(-self.config.sliding_window)
        frequencies_global = _rope(
            self.config.global_head_dim or self.config.head_dim,
            length,
            self.config.rope_theta_global,
            self.config.rope_scale_global,
            device=hidden.device,
            rotary_fraction=self.config.global_partial_rotary_factor,
            dtype=(hidden.dtype if self.config.architecture == "gemma4_ltx_12b" else None),
            precompute_inverse_on_cpu=self.config.architecture == "gemma4_ltx_12b",
            matrix=self.config.architecture == "gemma4_ltx_12b",
        )
        frequencies_local = _rope(
            self.config.head_dim,
            length,
            self.config.rope_theta_local,
            self.config.rope_scale_local,
            device=hidden.device,
            dtype=(hidden.dtype if self.config.architecture == "gemma4_ltx_12b" else None),
            precompute_inverse_on_cpu=self.config.architecture == "gemma4_ltx_12b",
            matrix=self.config.architecture == "gemma4_ltx_12b",
        )
        stack: list[torch.Tensor] = []
        prefetch = make_prefetch_queue(self.layers)
        try:
            for layer in self.layers:
                prefetch_queue_pop(prefetch, layer)
                stack.append(hidden.unsqueeze(1))
                hidden = layer(hidden, mask, sliding_mask, frequencies_global, frequencies_local)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        hidden = _rms_norm(self.norm, hidden, add_weight=self.config.rms_norm_add)
        stack.append(hidden.unsqueeze(1))
        return torch.cat(stack, dim=1)


class LtxDualTextProjection(torch.nn.Module):
    """The dual video/audio aggregate projection over the state stack."""

    def __init__(
        self,
        in_features: int = LTX_TEXT_STACK_FEATURES,
        video_features: int = 4096,
        audio_features: int = 2048,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.audio_aggregate_embed = operations.linear(in_features, audio_features, bias=True)
        self.video_aggregate_embed = operations.linear(in_features, video_features, bias=True)

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        source_dim = stack.shape[-1]
        folded = stack.movedim(1, -1)
        folded = (folded * torch.rsqrt(torch.mean(folded**2, dim=2, keepdim=True) + 1e-6)).flatten(
            start_dim=2
        )
        video = self.video_aggregate_embed(
            folded * math.sqrt(self.video_aggregate_embed.out_features / source_dim)
        )
        audio = self.audio_aggregate_embed(
            folded * math.sqrt(self.audio_aggregate_embed.out_features / source_dim)
        )
        return torch.cat((video, audio), dim=-1)


class LtxGemmaTextEncoder:
    """Raw prompt to LTX-2 projected Gemma conditioning, with no weighting.

    ``projection`` is either the single bias-free Linear (folded stack
    to 3840) or :class:`LtxDualTextProjection` (video 4096 + audio
    2048 concatenated to 6144); the single path's range normalization
    lives here because the reference keeps that module a bare Linear.
    ``connectors``, when given, refine the single projection's output
    into concatenated video and audio embeddings (7680), exactly as the
    reference applies them for combined checkpoints; the reference
    never consults its connectors on the dual path, so pairing them
    with the dual projection is refused instead of silently ignoring
    loaded weights.
    """

    def __init__(
        self,
        model: GemmaTextModel,
        projection: torch.nn.Linear | LtxDualTextProjection,
        tokenizer: GemmaSentencePieceTokenizer | GemmaJsonTokenizer,
        connectors: LtxTextConnectors | None = None,
    ) -> None:
        if model.config.architecture not in ("gemma3_ltx_12b", "gemma4_ltx_12b"):
            raise ValueError(
                "LtxGemmaTextEncoder requires the gemma3_ltx_12b or gemma4_ltx_12b profile, got"
                f" {model.config.architecture!r}"
            )
        if model.config.architecture == "gemma4_ltx_12b" and not isinstance(
            projection, LtxDualTextProjection
        ):
            raise ValueError("LTX-2.5 Gemma 4 text requires the matching dual projection")
        if connectors is not None and isinstance(projection, LtxDualTextProjection):
            raise ValueError(
                "text-embedding connectors apply only after the single projection;"
                " a dual-projection checkpoint cannot carry them"
            )
        self.model = model
        self.projection = projection
        self.connectors = connectors
        self._tokenizer = tokenizer

    def encode(self, text: str, *, apply_template: bool = False) -> Conditioning[torch.Tensor]:
        tokens = tokenize_ltx_gemma_prompt(
            text,
            encode=self._tokenizer.encode,
            apply_template=apply_template,
            config=(
                GEMMA4_LTX_12B_CONFIG
                if self.model.config.architecture == "gemma4_ltx_12b"
                else GEMMA3_LTX_12B_CONFIG
            ),
        )
        return self.encode_tokens(tokens)

    def _projection_dtype(self) -> torch.dtype:
        linear = (
            self.projection.video_aggregate_embed
            if isinstance(self.projection, LtxDualTextProjection)
            else self.projection
        )
        if isinstance(linear, Int8Linear):
            return linear.compute_dtype
        return bound_compute_dtype(linear) or linear.weight.dtype

    def encode_tokens(self, tokens: LtxGemmaPromptTokens) -> Conditioning[torch.Tensor]:
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(tokens.ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.tensor(tokens.attention_mask, dtype=torch.long, device=device).unsqueeze(
            0
        )
        stack = self.model(ids, attention).float()
        attended = sum(tokens.attention_mask)
        stack = stack[:, :, stack.shape[2] - attended :]
        # The reference casts the attended stack to the text compute dtype
        # before the fold, normalization, and projection, and floats only
        # the final output; a float32 projection makes this a no-op.
        stack = stack.to(self._projection_dtype())
        if isinstance(self.projection, LtxDualTextProjection):
            output = self.projection(stack)
        else:
            folded = stack.movedim(1, -1)
            folded = (
                8.0
                * (folded - folded.mean(dim=(1, 2), keepdim=True))
                / (
                    folded.amax(dim=(1, 2), keepdim=True)
                    - folded.amin(dim=(1, 2), keepdim=True)
                    + 1e-6
                )
            )
            folded = folded.reshape(folded.shape[0], folded.shape[1], -1)
            output = self.projection(folded)
            if self.connectors is not None:
                output = self.connectors(output)
        return Conditioning(output.float(), None)


__all__ = [
    "GemmaAttention",
    "GemmaBlock",
    "GemmaMlp",
    "GemmaTextModel",
    "LtxDualTextProjection",
    "LtxGemmaTextEncoder",
]
