"""Unregistered base Qwen Image diffusion transformer source.

This ports ``comfy/ldm/qwen_image/model.py`` from ComfyUI
2a68ce33b4c9ea6ee4283e618a74560cefb32694. Construction consumes the
torch-free exact base profile and preserves its checkpoint state names.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as F
from dinkster_inference.qwen_image import QWEN_IMAGE_CONFIG, QwenImageConfig

from .attention import AttentionKernel, select_attention
from .flux import EmbedND, apply_rope
from .model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from .operations import INITLESS, Operations

_DEFAULT_QWEN_IMAGE_ATTENTION = select_attention("qwen").kernel


class QwenImageBlockPatch(Protocol):
    """One ordered image-token transform applied after every Qwen block."""

    def __call__(self, image: torch.Tensor, block_index: int) -> torch.Tensor: ...


__all__ = [
    "QwenImage",
    "QwenImageAttention",
    "QwenImageBlockPatch",
    "QwenImageFeedForward",
    "QwenImageLastLayer",
    "QwenImageTimestepEmbeddings",
    "QwenImageTransformerBlock",
    "qwen_image_timestep_embedding",
]


def qwen_image_timestep_embedding(timesteps: torch.Tensor) -> torch.Tensor:
    """The exact 256-channel, scale-1000 Qwen Image time projection."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have rank 1, got rank {timesteps.ndim}")
    half = 128
    exponent = (
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    angles = timesteps[:, None].float() * torch.exp(exponent)[None]
    angles = 1000 * angles
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class _TimestepEmbedding(torch.nn.Module):
    def __init__(self, hidden: int, *, operations: Operations) -> None:
        super().__init__()
        self.linear_1 = operations.linear(256, hidden)
        self.act = torch.nn.SiLU()
        self.linear_2 = operations.linear(hidden, hidden)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class QwenImageTimestepEmbeddings(torch.nn.Module):
    def __init__(
        self,
        hidden: int,
        *,
        use_additional_t_cond: bool = False,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.timestep_embedder = _TimestepEmbedding(hidden, operations=operations)
        self.addition_t_embedding = (
            operations.embedding(2, hidden) if use_additional_t_cond else None
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden: torch.Tensor,
        additional_t_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = qwen_image_timestep_embedding(timestep).to(dtype=hidden.dtype)
        embedding = self.timestep_embedder(projected)
        if self.addition_t_embedding is None:
            if additional_t_cond is not None:
                raise ValueError("additional timestep condition requires Qwen Image Layered")
            return embedding
        if additional_t_cond is None:
            additional_t_cond = torch.zeros(
                (embedding.shape[0],), dtype=torch.long, device=embedding.device
            )
        if (
            additional_t_cond.shape != (embedding.shape[0],)
            or additional_t_cond.dtype != torch.long
            or additional_t_cond.device != embedding.device
            or bool(torch.any((additional_t_cond < 0) | (additional_t_cond > 1)))
        ):
            raise ValueError(
                "additional timestep condition must be int64 [batch] with values 0 or 1"
            )
        return embedding + self.addition_t_embedding(additional_t_cond).to(embedding.dtype)


class _GELU(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, operations: Operations) -> None:
        super().__init__()
        self.proj = operations.linear(dim_in, dim_out)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden), approximate="tanh")


class QwenImageFeedForward(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.net = torch.nn.ModuleList(
            (
                _GELU(dim, dim * 4, operations=operations),
                torch.nn.Identity(),
                operations.linear(dim * 4, dim),
            )
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden = module(hidden)
        return hidden


class QwenImageAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_IMAGE_ATTENTION,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        if inner != dim:
            raise ValueError(f"heads * head_dim must equal dim, got {heads} * {head_dim} != {dim}")
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.norm_q = operations.rms_norm(head_dim, eps=1e-6)
        self.norm_k = operations.rms_norm(head_dim, eps=1e-6)
        self.norm_added_q = operations.rms_norm(head_dim, eps=1e-6)
        self.norm_added_k = operations.rms_norm(head_dim, eps=1e-6)
        self.to_q = operations.linear(dim, inner, bias=True)
        self.to_k = operations.linear(dim, inner, bias=True)
        self.to_v = operations.linear(dim, inner, bias=True)
        self.add_q_proj = operations.linear(dim, inner, bias=True)
        self.add_k_proj = operations.linear(dim, inner, bias=True)
        self.add_v_proj = operations.linear(dim, inner, bias=True)
        self.to_out = torch.nn.ModuleList((operations.linear(inner, dim), torch.nn.Identity()))
        self.to_add_out = operations.linear(inner, dim)

    def _project(
        self,
        hidden: torch.Tensor,
        query: torch.nn.Linear,
        key: torch.nn.Linear,
        value: torch.nn.Linear,
        query_norm: torch.nn.RMSNorm,
        key_norm: torch.nn.RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        q = query_norm(query(hidden).view(shape).transpose(1, 2).contiguous())
        k = key_norm(key(hidden).view(shape).transpose(1, 2).contiguous())
        v = value(hidden).view(shape).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_q, img_k, img_v = self._project(
            image, self.to_q, self.to_k, self.to_v, self.norm_q, self.norm_k
        )
        txt_q, txt_k, txt_v = self._project(
            text,
            self.add_q_proj,
            self.add_k_proj,
            self.add_v_proj,
            self.norm_added_q,
            self.norm_added_k,
        )
        text_tokens = text.shape[1]
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        q, k = apply_rope(q, k, frequencies)
        attended = self._attention_kernel(q, k, v, mask=mask, causal=False)
        attended = attended.transpose(1, 2).reshape(image.shape[0], -1, image.shape[2])
        text_out = self.to_add_out(attended[:, :text_tokens])
        image_out = attended[:, text_tokens:]
        for module in self.to_out:
            image_out = module(image_out)
        return image_out, text_out


class QwenImageTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_IMAGE_ATTENTION,
    ) -> None:
        super().__init__()
        self.img_mod = torch.nn.Sequential(torch.nn.SiLU(), operations.linear(dim, 6 * dim))
        self.img_norm1 = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)
        self.img_norm2 = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)
        self.img_mlp = QwenImageFeedForward(dim, operations=operations)
        self.txt_mod = torch.nn.Sequential(torch.nn.SiLU(), operations.linear(dim, 6 * dim))
        self.txt_norm1 = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)
        self.txt_norm2 = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)
        self.txt_mlp = QwenImageFeedForward(dim, operations=operations)
        self.attn = QwenImageAttention(
            dim,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )

    @staticmethod
    def _modulate(
        hidden: torch.Tensor,
        parameters: torch.Tensor,
        timestep_zero_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        shift, scale, gate = parameters.chunk(3, dim=-1)
        if timestep_zero_index is not None:
            batch = shift.shape[0] // 2
            regular = torch.addcmul(
                shift[:batch, None],
                hidden[:, :timestep_zero_index],
                1 + scale[:batch, None],
            )
            zero = torch.addcmul(
                shift[batch:, None],
                hidden[:, timestep_zero_index:],
                1 + scale[batch:, None],
            )
            return torch.cat((regular, zero), dim=1), (
                gate[:batch, None],
                gate[batch:, None],
            )
        return torch.addcmul(shift[:, None], hidden, 1 + scale[:, None]), gate[:, None]

    @staticmethod
    def _apply_gate(
        update: torch.Tensor,
        hidden: torch.Tensor,
        gate: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        timestep_zero_index: int | None,
    ) -> torch.Tensor:
        if timestep_zero_index is None:
            assert isinstance(gate, torch.Tensor)
            return torch.addcmul(hidden, gate, update)
        assert isinstance(gate, tuple)
        return hidden + torch.cat(
            (
                update[:, :timestep_zero_index] * gate[0],
                update[:, timestep_zero_index:] * gate[1],
            ),
            dim=1,
        )

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        frequencies: torch.Tensor,
        mask: torch.Tensor | None,
        timestep_zero_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        if timestep_zero_index is not None:
            temb = temb.chunk(2, dim=0)[0]
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
        img_norm, img_gate = self._modulate(self.img_norm1(image), img_mod1, timestep_zero_index)
        txt_norm, txt_gate = self._modulate(self.txt_norm1(text), txt_mod1)
        assert isinstance(txt_gate, torch.Tensor)
        del img_mod1, txt_mod1
        img_attn, txt_attn = self.attn(img_norm, txt_norm, mask, frequencies)
        del img_norm, txt_norm
        image = self._apply_gate(img_attn, image, img_gate, timestep_zero_index)
        text = text + txt_gate * txt_attn
        del img_attn, txt_attn, img_gate, txt_gate
        img_norm, img_gate = self._modulate(self.img_norm2(image), img_mod2, timestep_zero_index)
        image = self._apply_gate(self.img_mlp(img_norm), image, img_gate, timestep_zero_index)
        txt_norm, txt_gate = self._modulate(self.txt_norm2(text), txt_mod2)
        assert isinstance(txt_gate, torch.Tensor)
        text = torch.addcmul(text, txt_gate, self.txt_mlp(txt_norm))
        return text, image


class QwenImageLastLayer(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = operations.linear(dim, dim * 2)
        self.norm = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return torch.addcmul(shift[:, None], self.norm(hidden), (1 + scale)[:, None])


class QwenImage(torch.nn.Module):
    """Base Qwen Image DiT source without assembly or runtime registration."""

    def __init__(
        self,
        config: QwenImageConfig = QWEN_IMAGE_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_IMAGE_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        if config.patch != (2, 2):
            raise ValueError(f"Qwen Image requires a 2x2 spatial patch, got {config.patch}")
        if sum(config.rope_axes) != config.attention_head_dim:
            raise ValueError("Qwen Image RoPE axes must sum to attention_head_dim")
        if config.patchified_input_channels != config.output_latent_channels * 4:
            raise ValueError("Qwen Image patchified channels must equal latent channels * 4")
        if config.default_ref_method not in ("index", "index_timestep_zero", "negative_index"):
            raise ValueError("unsupported Qwen Image reference packing method")
        self.pe_embedder = EmbedND(config.attention_head_dim, 10000, config.rope_axes)
        self.time_text_embed = QwenImageTimestepEmbeddings(
            config.hidden_width,
            use_additional_t_cond=config.use_additional_t_cond,
            operations=operations,
        )
        self.txt_norm = operations.rms_norm(config.text_width, eps=1e-6)
        self.img_in = operations.linear(config.patchified_input_channels, config.hidden_width)
        self.txt_in = operations.linear(config.text_width, config.hidden_width)
        self.transformer_blocks = torch.nn.ModuleList(
            QwenImageTransformerBlock(
                config.hidden_width,
                config.attention_heads,
                config.attention_head_dim,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.transformer_blocks)
        )
        self.norm_out = QwenImageLastLayer(config.hidden_width, operations=operations)
        self.proj_out = operations.linear(config.hidden_width, config.output_latent_channels * 4)
        if config.default_ref_method == "index_timestep_zero":
            self.register_buffer("__index_timestep_zero__", torch.tensor([]))

    def _validate_inputs(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None,
        ref_latents: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, ...]:
        if x.ndim != 5:
            raise ValueError(f"Qwen Image latent must have rank 5 [B,C,T,H,W], got rank {x.ndim}")
        batch, channels, temporal, height, width = x.shape
        if channels != self.config.output_latent_channels:
            raise ValueError(
                "Qwen Image latent channels must be"
                f" {self.config.output_latent_channels}, got {channels}"
            )
        if min(batch, temporal, height, width) < 1:
            raise ValueError("Qwen Image latent extents must be positive")
        if not x.is_floating_point():
            raise ValueError("Qwen Image latent must use a floating-point dtype")
        if (
            timesteps.shape != (batch,)
            or not timesteps.is_floating_point()
            or timesteps.device != x.device
        ):
            raise ValueError(
                f"timesteps must be floating [batch] = ({batch},) on the latent device"
            )
        if (
            context.ndim != 3
            or context.shape[0] != batch
            or context.shape[2] != self.config.text_width
        ):
            raise ValueError(
                f"context must be [batch,tokens,{self.config.text_width}], got"
                f" {tuple(context.shape)}"
            )
        if not context.is_floating_point() or context.device != x.device:
            raise ValueError("context must be floating and on the latent device")
        if attention_mask is not None:
            if attention_mask.shape != context.shape[:2]:
                raise ValueError(
                    "attention_mask must match [batch,text_tokens], got"
                    f" {tuple(attention_mask.shape)} vs {tuple(context.shape[:2])}"
                )
            if attention_mask.device != x.device or attention_mask.is_complex():
                raise ValueError("attention_mask must be real and on the latent device")
        if ref_latents is None:
            return ()
        if not isinstance(ref_latents, (tuple, list)):
            raise ValueError("ref_latents must be a sequence of tensors")
        refs = tuple(ref_latents)
        for index, ref in enumerate(refs):
            if ref.ndim != 5:
                raise ValueError(f"reference {index} must have rank 5, got rank {ref.ndim}")
            if ref.shape[0] != batch or ref.shape[1] != channels:
                raise ValueError(
                    f"reference {index} must have batch/channels ({batch},{channels}), got"
                    f" {tuple(ref.shape[:2])}"
                )
            if min(ref.shape[2:]) < 1:
                raise ValueError(f"reference {index} extents must be positive")
            if not ref.is_floating_point() or ref.dtype != x.dtype or ref.device != x.device:
                raise ValueError(f"reference {index} must match latent dtype and device")
        return refs

    def pack_image(
        self,
        x: torch.Tensor,
        *,
        index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int, int]]:
        batch, channels, temporal, height, width = x.shape
        pad_h = (-height) % 2
        pad_w = (-width) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="circular")
        padded_shape = (batch, channels, temporal, x.shape[-2], x.shape[-1])
        _, _, _, padded_h, padded_w = padded_shape
        h_len = padded_h // 2
        w_len = padded_w // 2
        packed = (
            x.view(batch, channels, temporal, h_len, 2, w_len, 2)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(batch, temporal * h_len * w_len, channels * 4)
        )
        ids = torch.zeros((temporal, h_len, w_len, 3), device=x.device)
        if temporal > 1:
            ids[..., 0] = (
                ids[..., 0]
                + torch.linspace(0, temporal - 1, steps=temporal, device=x.device, dtype=x.dtype)[
                    :, None, None
                ]
            )
        else:
            ids[..., 0] = ids[..., 0] + index
        ids[..., 1] = (
            ids[..., 1]
            + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[
                None, :, None
            ]
            - (h_len // 2)
        )
        ids[..., 2] = (
            ids[..., 2]
            + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[
                None, None, :
            ]
            - (w_len // 2)
        )
        return packed, ids.reshape(1, -1, 3).expand(batch, -1, -1), padded_shape

    def _joint_mask(
        self,
        attention_mask: torch.Tensor | None,
        image_tokens: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.is_floating_point():
            text_mask = attention_mask.to(dtype=dtype)
        else:
            text_mask = (attention_mask.to(dtype=dtype) - 1) * torch.finfo(dtype).max
        image_mask = torch.zeros((text_mask.shape[0], image_tokens), dtype=dtype, device=device)
        return torch.cat((text_mask, image_mask), dim=1)[:, None, None]

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ref_latents: Sequence[torch.Tensor] | None = None,
        additional_t_cond: torch.Tensor | None = None,
        control_residuals: Sequence[torch.Tensor | None] | None = None,
        block_patches: Sequence[QwenImageBlockPatch] = (),
    ) -> torch.Tensor:
        refs = self._validate_inputs(x, timesteps, context, attention_mask, ref_latents)
        if control_residuals is not None and len(control_residuals) != len(self.transformer_blocks):
            raise ValueError("Qwen Image control residuals must match the transformer block count")
        batch, _, _, height, width = x.shape
        image, image_ids, padded_shape = self.pack_image(x)
        target_tokens = image.shape[1]
        reference_index = 0
        for ref in refs:
            reference_index += -1 if self.config.default_ref_method == "negative_index" else 1
            index = reference_index
            ref_image, ref_ids, _ = self.pack_image(ref, index=index)
            image = torch.cat((image, ref_image), dim=1)
            image_ids = torch.cat((image_ids, ref_ids), dim=1)
        timestep_zero_index = None
        if refs and self.config.default_ref_method == "index_timestep_zero":
            timesteps = torch.cat((timesteps, torch.zeros_like(timesteps)), dim=0)
            timestep_zero_index = target_tokens
        h_tokens = padded_shape[-2] // 2
        w_tokens = padded_shape[-1] // 2
        text_start = round(max(w_tokens // 2, h_tokens // 2))
        text_ids = (
            torch.arange(
                text_start,
                text_start + context.shape[1],
                device=x.device,
            )
            .reshape(1, -1, 1)
            .expand(batch, -1, 3)
        )
        ids = torch.cat((text_ids, image_ids), dim=1)
        frequencies = self.pe_embedder(ids).to(dtype=x.dtype).contiguous()
        del ids, text_ids, image_ids

        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(context))
        temb = self.time_text_embed(timesteps, image, additional_t_cond)
        mask = self._joint_mask(
            attention_mask,
            image.shape[1],
            dtype=image.dtype,
            device=image.device,
        )
        prefetch = make_prefetch_queue(self.transformer_blocks)
        try:
            for index, block in enumerate(self.transformer_blocks):
                prefetch_queue_pop(prefetch, block)
                text, image = block(image, text, temb, frequencies, mask, timestep_zero_index)
                for patch in block_patches:
                    image = patch(image, index)
                residual = None if control_residuals is None else control_residuals[index]
                if residual is not None:
                    if (
                        residual.ndim != 3
                        or residual.shape[0] != image.shape[0]
                        or residual.shape[1] > image.shape[1]
                        or residual.shape[2] != image.shape[2]
                        or residual.device != image.device
                        or residual.dtype != image.dtype
                    ):
                        raise ValueError(
                            "Qwen Image control residual must match the image-token prefix"
                        )
                    controlled_tokens = residual.shape[1]
                    image = torch.cat(
                        (
                            image[:, :controlled_tokens] + residual,
                            image[:, controlled_tokens:],
                        ),
                        dim=1,
                    )
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        if timestep_zero_index is not None:
            temb = temb.chunk(2, dim=0)[0]
        image = self.proj_out(self.norm_out(image, temb))[:, :target_tokens]
        _, channels, temporal, padded_h, padded_w = padded_shape
        return (
            image.view(batch, temporal, padded_h // 2, padded_w // 2, channels, 2, 2)
            .permute(0, 4, 1, 2, 5, 3, 6)
            .reshape(padded_shape)[:, :, :, :height, :width]
        )
