"""Native Wan 2.2 WanDancer diffusion transformer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from dinkster_inference.wan21 import WAN22_WANDANCER_14B, Wan21Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .flux import EmbedND, apply_rope
from .operations import INITLESS, CastOperations, Operations
from .wan21_model import (
    Wan21Model,
    WanCrossAttention,
    WanHead,
    WanImageEmbedding,
    sinusoidal_embedding_1d,
)

_AUDIO_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
_DEFAULT_ATTENTION = select_attention("flux").kernel


class _MusicSelfAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.q_proj = operations.linear(256, 256)
        self.k_proj = operations.linear(256, 256)
        self.v_proj = operations.linear(256, 256)
        self.out_proj = operations.linear(256, 256)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        batch, rows = x.shape[:2]
        q = self.q_proj(x).view(batch, rows, 4, 64)
        k = self.k_proj(x).view(batch, rows, 4, 64)
        q, k = apply_rope(q, k, freqs)
        v = self.v_proj(x).view(batch, rows, 4, 64)
        attended = self._attention_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(attended.transpose(1, 2).reshape(batch, rows, 256))


class _MusicEncoderLayer(torch.nn.Module):
    def __init__(self, *, operations: Operations, attention_kernel: AttentionKernel) -> None:
        super().__init__()
        self.self_attn = _MusicSelfAttention(
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.linear1 = operations.linear(256, 1024)
        self.linear2 = operations.linear(1024, 256)
        self.norm1 = operations.layer_norm(256)
        self.norm2 = operations.layer_norm(256)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), freqs)
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class _MusicInjector(torch.nn.Module):
    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.injected_block_id = {
            block_id: index for index, block_id in enumerate(_AUDIO_INJECT_LAYERS)
        }
        self.injector = torch.nn.ModuleList(
            WanCrossAttention(
                config.hidden_size,
                config.num_heads,
                qk_norm=True,
                eps=config.eps,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in _AUDIO_INJECT_LAYERS
        )
        self.injector_pre_norm_feat = torch.nn.ModuleList(
            operations.layer_norm(config.hidden_size, eps=config.eps, elementwise_affine=False)
            for _ in _AUDIO_INJECT_LAYERS
        )
        self.injector_pre_norm_vec = torch.nn.ModuleList(
            operations.layer_norm(config.hidden_size, eps=config.eps, elementwise_affine=False)
            for _ in _AUDIO_INJECT_LAYERS
        )

    def forward(
        self,
        x: torch.Tensor,
        block_id: int,
        audio: torch.Tensor,
        target_offset: int,
        target_rows: int,
        scale: float,
    ) -> torch.Tensor:
        injector_id = self.injected_block_id.get(block_id)
        if injector_id is None or scale == 0.0:
            return x
        frames = audio.shape[1]
        if target_rows % frames:
            raise ValueError("WanDancer video token rows must divide evenly across music frames")
        target_end = target_offset + target_rows
        target = (
            x[:, target_offset:target_end]
            .unflatten(1, (frames, target_rows // frames))
            .flatten(0, 1)
        )
        normalized = self.injector_pre_norm_feat[injector_id](target)
        context = audio.flatten(0, 1).unsqueeze(1)
        residual = self.injector[injector_id](normalized, context)
        residual = residual.unflatten(0, (x.shape[0], frames)).flatten(1, 2)
        return torch.cat(
            (
                x[:, :target_offset],
                x[:, target_offset:target_end] + residual * scale,
                x[:, target_end:],
            ),
            dim=1,
        )


class Wan22DancerModel(Wan21Model):
    """Strict WanDancer model with FPS-selected patch and output heads."""

    def __init__(
        self,
        config: Wan21Config = WAN22_WANDANCER_14B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config.model_variant != "wandancer":
            raise ValueError("Wan22DancerModel requires a WanDancer configuration")
        super().__init__(config, operations=operations, attention_kernel=attention_kernel)
        self.patch_embedding_global = CastOperations(torch.float32).conv3d(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.img_emb_refimage = WanImageEmbedding(
            config.hidden_size,
            flf_pos_embed_token_number=None,
            operations=operations,
        )
        self.head_global = WanHead(config, operations=operations)
        self.music_projection = operations.linear(35, 256)
        self.music_encoder = torch.nn.ModuleList(
            _MusicEncoderLayer(operations=operations, attention_kernel=attention_kernel)
            for _ in range(2)
        )
        self.music_rope_embedder = EmbedND(dim=64, theta=10000, axes_dim=(64,))
        self.music_injector = _MusicInjector(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def _dancer_rope(
        self,
        grid: tuple[int, int, int],
        reference: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        time, height, width = grid
        ids = torch.zeros((time, height, width, 3), device=reference.device, dtype=reference.dtype)
        if int(fps + 0.5) == 30:
            time_positions = torch.linspace(0, time - 1, time, device=ids.device, dtype=ids.dtype)
        else:
            time_scale = 30.0 / fps
            time_positions = torch.arange(time, device=ids.device, dtype=ids.dtype) * time_scale
            time_positions[-1] = int(time_scale * time + 0.5) - 1
        ids[..., 0] += time_positions[:, None, None]
        ids[..., 1] += torch.arange(height, device=ids.device, dtype=ids.dtype)[None, :, None]
        ids[..., 2] += torch.arange(width, device=ids.device, dtype=ids.dtype)[None, None, :]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        reference_vision: torch.Tensor | None = None,
        audio_embed: torch.Tensor | None = None,
        fps: float = 30.0,
        audio_inject_scale: float = 1.0,
    ) -> torch.Tensor:
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("WanDancer fps must be finite and positive")
        if type(audio_inject_scale) is not float or not math.isfinite(audio_inject_scale):
            raise ValueError("WanDancer audio_inject_scale must be a finite float")
        self._validate(
            x,
            timesteps,
            context,
            vision,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        batch = x.shape[0]
        if reference_vision is not None and (
            type(reference_vision) is not torch.Tensor
            or not reference_vision.is_floating_point()
            or reference_vision.layout != torch.strided
            or reference_vision.ndim != 3
            or reference_vision.shape[0] != batch
            or reference_vision.shape[1] == 0
            or reference_vision.shape[2] != 1280
        ):
            raise ValueError("WanDancer reference_vision must be [batch,rows,1280]")
        if audio_embed is not None:
            if (
                type(audio_embed) is not torch.Tensor
                or not audio_embed.is_floating_point()
                or audio_embed.layout != torch.strided
                or audio_embed.ndim != 3
                or audio_embed.shape[2] != 35
                or audio_embed.shape[0] < 1
                or batch % audio_embed.shape[0]
            ):
                raise ValueError("WanDancer audio must be [batch,rows,35] and divide model batch")
            if audio_embed.shape[0] != batch:
                audio_embed = audio_embed.repeat(batch // audio_embed.shape[0], 1, 1)

        original_shape = x.shape[2:]
        pad_h = (-x.shape[3]) % 2
        pad_w = (-x.shape[4]) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="circular")
        global_branch = int(fps + 0.5) != 30
        embedding = self.patch_embedding_global if global_branch else self.patch_embedding
        projected = embedding(x.float()).to(x.dtype)
        grid = projected.shape[2:]
        latent_frames = grid[0]
        hidden = projected.flatten(2).transpose(1, 2)
        target_rows = hidden.shape[1]
        freqs = self._dancer_rope(
            grid,
            hidden,
            float(fps),
        )
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps.flatten()).to(hidden.dtype)
        ).reshape(batch, -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        text = self.text_embedding(context)
        image_rows = 0
        if vision is not None:
            assert self.img_emb is not None
            image = self.img_emb(vision)
            image_rows += image.shape[1]
            text = torch.cat((image, text), dim=1)
        if reference_vision is not None:
            reference_image = self.img_emb_refimage(reference_vision)
            image_rows += reference_image.shape[1]
            text = torch.cat((reference_image, text), dim=1)

        audio: torch.Tensor | None = None
        if audio_embed is not None:
            music_feature = self.music_projection(audio_embed)
            ids = torch.arange(
                music_feature.shape[1], device=music_feature.device, dtype=music_feature.dtype
            )[None, :, None]
            music_freqs = self.music_rope_embedder(ids).movedim(1, 2)
            for layer in self.music_encoder:
                music_feature = layer(music_feature, music_freqs)
            audio = F.interpolate(
                music_feature.unsqueeze(1),
                size=(latent_frames * 8, self.config.hidden_size),
                mode="bilinear",
            ).squeeze(1)

        with attention_kernel_context(
            self._attention_kernel,
            hidden.numel(),
            device=hidden.device,
        ):
            for index, block in enumerate(self.blocks):
                hidden = block(hidden, projected_time, freqs, text, image_rows)
                if audio is not None:
                    hidden = self.music_injector(
                        hidden,
                        index,
                        audio,
                        0,
                        target_rows,
                        audio_inject_scale,
                    )
        head = self.head_global if global_branch else self.head
        hidden = head(hidden, time)
        patch_t, patch_h, patch_w = self.config.patch_size
        hidden = hidden.view(
            batch,
            *grid,
            patch_t,
            patch_h,
            patch_w,
            self.config.out_channels,
        )
        hidden = torch.einsum("bthwpqrc->bctphqwr", hidden)
        output = hidden.reshape(
            batch,
            self.config.out_channels,
            grid[0] * patch_t,
            grid[1] * patch_h,
            grid[2] * patch_w,
        )
        return output[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


__all__ = ["Wan22DancerModel"]
