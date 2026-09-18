"""Native Wan 2.1 HuMo diffusion transformer."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import WAN21_HUMO_17B, Wan21Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .operations import INITLESS, Operations
from .ops import cast_weight
from .wan21_model import (
    Wan21Model,
    WanAttentionBlock,
    WanCrossAttention,
    _repeat_time_rows,  # pyright: ignore[reportPrivateUsage]
    sinusoidal_embedding_1d,
)

_DEFAULT_ATTENTION = select_attention("flux").kernel


class _Wan21HumoCrossAttention(WanCrossAttention):
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence = x.shape[:2]
        heads, width = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        if context.shape[1] % 16:
            raise ValueError("HuMo audio context must contain 16 tokens per video frame")
        grouped_frames = batch * (context.shape[1] // 16)
        if sequence % (context.shape[1] // 16):
            raise ValueError("HuMo video rows must divide evenly across audio frames")
        q = q.reshape(grouped_frames, -1, heads, width).transpose(1, 2)
        k = k.reshape(grouped_frames, 16, heads, width).transpose(1, 2)
        v = v.reshape(grouped_frames, 16, heads, width).transpose(1, 2)
        attended = self._attention_kernel(q, k, v)
        return self.o(attended.transpose(1, 2).reshape(batch, sequence, heads * width))


class _Wan21HumoAudioAttention(torch.nn.Module):
    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.audio_cross_attn = _Wan21HumoCrossAttention(
            config.hidden_size,
            config.num_heads,
            qk_norm=config.qk_norm,
            eps=config.eps,
            operations=operations,
            attention_kernel=attention_kernel,
            kv_dim=1536,
        )
        self.norm1_audio = operations.layer_norm(config.hidden_size, eps=config.eps)

    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return x + self.audio_cross_attn(self.norm1_audio(x), audio)


class _Wan21HumoAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.audio_cross_attn_wrapper = _Wan21HumoAudioAttention(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def _forward_humo(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        audio: torch.Tensor,
        modulation_state: torch.Tensor,
    ) -> torch.Tensor:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        x = x.contiguous()
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x),
            1 + _repeat_time_rows(modulation[1], x),
        )
        update = self.self_attn(normalized, freqs)
        x = torch.addcmul(x, update, _repeat_time_rows(modulation[2], x))
        x = x + self.cross_attn(self.norm3(x), context)
        x = self.audio_cross_attn_wrapper(x, audio)
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[3], x),
            self.norm2(x),
            1 + _repeat_time_rows(modulation[4], x),
        )
        return torch.addcmul(
            x,
            self.ffn(normalized),
            _repeat_time_rows(modulation[5], x),
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_humo(x, time, freqs, context, audio, modulation)
        with binding.lease() as lease:
            return self._forward_humo(
                x,
                time,
                freqs,
                context,
                audio,
                lease.get("modulation", dtype=x.dtype),
            )


class _Layer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


class _Wan21HumoAudioProjection(torch.nn.Module):
    def __init__(self, *, operations: Operations) -> None:
        super().__init__()
        self.audio_proj_glob_1 = _Layer(operations.linear(51200, 512))
        self.audio_proj_glob_2 = _Layer(operations.linear(512, 512))
        self.audio_proj_glob_3 = _Layer(operations.linear(512, 24576))
        self.audio_proj_glob_norm = _Layer(operations.layer_norm(1536))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, frames = audio.shape[:2]
        audio = audio.reshape(batch * frames, 51200)
        audio = torch.relu(self.audio_proj_glob_1(audio))
        audio = torch.relu(self.audio_proj_glob_2(audio))
        audio = self.audio_proj_glob_3(audio).reshape(batch * frames, 16, 1536)
        return self.audio_proj_glob_norm(audio).reshape(batch, frames * 16, 1536)


class Wan21HumoModel(Wan21Model):
    """Exact HuMo 17B consumer for windowed Whisper and reference latents."""

    def __init__(
        self,
        config: Wan21Config = WAN21_HUMO_17B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config.model_variant != "humo":
            raise ValueError("Wan21HumoModel requires a HuMo configuration")
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
            _block_type=_Wan21HumoAttentionBlock,
        )
        self.audio_proj = _Wan21HumoAudioProjection(operations=operations)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        audio_embed: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(
            x,
            timesteps,
            context,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if (
            type(audio_embed) is not torch.Tensor
            or not audio_embed.is_floating_point()
            or audio_embed.layout != torch.strided
        ):
            raise TypeError("HuMo audio must be an exact strided floating torch.Tensor")
        if tuple(audio_embed.shape) != (x.shape[0], x.shape[2], 8, 5, 1280):
            raise ValueError("HuMo audio must have shape [batch,target_frames,8,5,1280]")
        if (
            type(reference_latent) is not torch.Tensor
            or not reference_latent.is_floating_point()
            or reference_latent.layout != torch.strided
        ):
            raise TypeError("HuMo reference must be an exact strided floating torch.Tensor")
        if (
            reference_latent.ndim != 5
            or reference_latent.shape[0] != x.shape[0]
            or reference_latent.shape[1] != self.config.in_channels
            or reference_latent.shape[2] <= 0
            or reference_latent.shape[3:] != x.shape[3:]
        ):
            raise ValueError("HuMo reference must match target batch and spatial geometry")

        original_shape = x.shape[2:]
        pad_h = (-x.shape[3]) % self.config.patch_size[1]
        pad_w = (-x.shape[4]) % self.config.patch_size[2]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="circular")
            reference_latent = F.pad(
                reference_latent,
                (0, pad_w, 0, pad_h, 0, 0),
                mode="circular",
            )
        target = self.patch_embedding(x.float()).to(x.dtype)
        reference = self.patch_embedding(reference_latent.float()).to(x.dtype)
        target_grid = (target.shape[2], target.shape[3], target.shape[4])
        reference_grid = (reference.shape[2], reference.shape[3], reference.shape[4])
        freqs = torch.cat(
            (
                self._rope(target_grid, target),
                self._rope(reference_grid, target, time_start=target_grid[0]),
            ),
            dim=1,
        )
        target_rows = target.flatten(2).shape[2]
        hidden = torch.cat(
            (target.flatten(2).transpose(1, 2), reference.flatten(2).transpose(1, 2)),
            dim=1,
        )
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps.flatten()).to(hidden.dtype)
        ).reshape(timesteps.shape[0], -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        context = self.text_embedding(context)
        reference_audio = audio_embed.new_zeros(
            audio_embed.shape[0],
            reference_grid[0],
            *audio_embed.shape[2:],
        )
        audio = self.audio_proj(torch.cat((audio_embed, reference_audio), dim=1))
        with attention_kernel_context(
            self._attention_kernel,
            hidden.numel(),
            device=hidden.device,
        ):
            for block in self.blocks:
                hidden = cast("_Wan21HumoAttentionBlock", block)(
                    hidden,
                    projected_time,
                    freqs,
                    context,
                    audio,
                )
        hidden = self.head(hidden, time)[:, :target_rows]
        batch = hidden.shape[0]
        patch_t, patch_h, patch_w = self.config.patch_size
        hidden = hidden.view(
            batch,
            *target_grid,
            patch_t,
            patch_h,
            patch_w,
            self.config.out_channels,
        )
        hidden = torch.einsum("bthwpqrc->bctphqwr", hidden)
        hidden = hidden.reshape(
            batch,
            self.config.out_channels,
            target_grid[0] * patch_t,
            target_grid[1] * patch_h,
            target_grid[2] * patch_w,
        )
        return hidden[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


__all__ = ["Wan21HumoModel"]
