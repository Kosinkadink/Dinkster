"""Native Wan 2.2 sound-to-video diffusion transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference import WAN22_S2V_14B, Wan21Config

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight
from .wan21_model import Wan21Model, WanCrossAttention, sinusoidal_embedding_1d

_AUDIO_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)
_DEFAULT_ATTENTION = select_attention("flux").kernel


class _CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.conv = operations.conv1d(in_channels, out_channels, 3, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (2, 0), mode="replicate"))


class _S2VMotionEncoder(ResidencyRouted, torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        quarter = dim // 4
        self.conv1_local = _CausalConv1d(1024, dim, stride=1, operations=operations)
        self.conv1_global = _CausalConv1d(1024, quarter, stride=1, operations=operations)
        self.norm1 = operations.layer_norm(quarter, eps=1e-6, elementwise_affine=False)
        self.act = torch.nn.SiLU()
        self.conv2 = _CausalConv1d(quarter, dim // 2, stride=2, operations=operations)
        self.conv3 = _CausalConv1d(dim // 2, dim, stride=2, operations=operations)
        self.final_linear = operations.linear(dim, dim)
        self.norm2 = operations.layer_norm(dim // 2, eps=1e-6, elementwise_affine=False)
        self.norm3 = operations.layer_norm(dim, eps=1e-6, elementwise_affine=False)
        self.padding_tokens = torch.nn.Parameter(torch.empty(1, 1, 1, dim))

    def _branch(self, x: torch.Tensor, first: _CausalConv1d) -> torch.Tensor:
        x = self.act(self.norm1(first(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm2(self.conv2(x).transpose(1, 2))).transpose(1, 2)
        return self.act(self.norm3(self.conv3(x).transpose(1, 2)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        channels_first = x.transpose(1, 2)
        local = self.conv1_local(channels_first)
        local = local.unflatten(1, (4, local.shape[1] // 4)).flatten(0, 1)
        local = self.act(self.norm1(local.transpose(1, 2))).transpose(1, 2)
        local = self.act(self.norm2(self.conv2(local).transpose(1, 2))).transpose(1, 2)
        local = self.act(self.norm3(self.conv3(local).transpose(1, 2)))
        local = local.unflatten(0, (batch, 4))
        local = local.permute(0, 2, 1, 3)

        binding = self._offloaded_residency()
        if binding is None:
            padding = cast_weight(
                self.padding_tokens,
                device=local.device,
                dtype=local.dtype,
            )
        else:
            with binding.lease() as lease:
                padding = lease.get("padding_tokens", dtype=local.dtype)
        padding = padding.expand(batch, local.shape[1], -1, -1)
        local = torch.cat((local, padding), dim=2)

        global_rows = self._branch(channels_first, self.conv1_global)
        global_rows = self.final_linear(global_rows).unsqueeze(2)
        return global_rows, local


class _S2VAudioEncoder(ResidencyRouted, torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.encoder = _S2VMotionEncoder(dim, operations=operations)
        self.weights = torch.nn.Parameter(torch.empty(1, 25, 1, 1))
        self.act = torch.nn.SiLU()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            weights = cast_weight(self.weights, device=features.device, dtype=features.dtype)
        else:
            with binding.lease() as lease:
                weights = lease.get("weights", dtype=features.dtype)
        weights = self.act(weights)
        weighted = ((features * weights) / weights.sum(dim=1, keepdim=True)).sum(dim=1)
        return self.encoder(weighted.transpose(1, 2))


class _S2VAdaLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = operations.linear(dim, dim * 2)
        self.norm = operations.layer_norm(dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(self.silu(embedding)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _S2VAudioInjector(torch.nn.Module):
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
            operations.layer_norm(
                config.hidden_size,
                eps=config.eps,
                elementwise_affine=False,
            )
            for _ in _AUDIO_INJECT_LAYERS
        )
        self.injector_pre_norm_vec = torch.nn.ModuleList(
            operations.layer_norm(
                config.hidden_size,
                eps=config.eps,
                elementwise_affine=False,
            )
            for _ in _AUDIO_INJECT_LAYERS
        )
        self.injector_adain_layers = torch.nn.ModuleList(
            _S2VAdaLayerNorm(config.hidden_size, operations=operations)
            for _ in _AUDIO_INJECT_LAYERS
        )

    def forward(
        self,
        x: torch.Tensor,
        block_id: int,
        audio: torch.Tensor,
        global_audio: torch.Tensor,
        target_rows: int,
    ) -> torch.Tensor:
        injector_id = self.injected_block_id.get(block_id)
        if injector_id is None:
            return x
        frames = audio.shape[1]
        target = x[:, :target_rows].unflatten(1, (frames, target_rows // frames))
        target = target.flatten(0, 1)
        global_embedding = global_audio.flatten(0, 1)[:, 0]
        normalized = self.injector_adain_layers[injector_id](target, global_embedding)
        residual = self.injector[injector_id](
            normalized,
            audio.flatten(0, 1),
        ).unflatten(0, (x.shape[0], frames))
        residual = residual.flatten(1, 2)
        return torch.cat((x[:, :target_rows] + residual, x[:, target_rows:]), dim=1)


class _S2VFramePacker(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.proj = operations.conv3d(16, dim, (1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = operations.conv3d(16, dim, (2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = operations.conv3d(16, dim, (4, 8, 8), stride=(4, 8, 8))

    def forward(
        self,
        motion: torch.Tensor,
        model: Wan22S2VModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if motion.shape[2] < 19:
            motion = F.pad(motion, (0, 0, 0, 0, 19 - motion.shape[2], 0))
        else:
            motion = motion[:, :, -19:]
        far, middle, recent = motion.split((16, 2, 1), dim=2)
        recent = self.proj(recent).flatten(2).transpose(1, 2)
        middle_projected = self.proj_2x(middle)
        far_projected = self.proj_4x(far)
        packed = torch.cat(
            (
                recent,
                middle_projected.flatten(2).transpose(1, 2),
                far_projected.flatten(2).transpose(1, 2),
            ),
            dim=1,
        )
        height, width = motion.shape[-2:]
        freqs = torch.cat(
            (
                model.rope_encode(1, height, width, motion, t_start=-1),
                model.rope_encode(
                    1,
                    height,
                    width,
                    motion,
                    t_start=-3,
                    steps_h=middle_projected.shape[-2],
                    steps_w=middle_projected.shape[-1],
                ),
                model.rope_encode(
                    4,
                    height,
                    width,
                    motion,
                    t_start=-19,
                    steps_h=far_projected.shape[-2],
                    steps_w=far_projected.shape[-1],
                ),
            ),
            dim=1,
        )
        return packed, freqs


class Wan22S2VModel(Wan21Model):
    """Wan 2.2 S2V backbone with audio, image, motion, and video controls."""

    def __init__(
        self,
        config: Wan21Config = WAN22_S2V_14B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config.model_variant != "s2v":
            raise ValueError("Wan22S2VModel requires an S2V configuration")
        super().__init__(config, operations=operations, attention_kernel=attention_kernel)
        self.trainable_cond_mask = operations.embedding(3, config.hidden_size)
        self.casual_audio_encoder = _S2VAudioEncoder(config.hidden_size, operations=operations)
        self.cond_encoder = operations.conv3d(
            16,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.audio_injector = _S2VAudioInjector(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.frame_packer = _S2VFramePacker(config.hidden_size, operations=operations)

    def rope_encode(
        self,
        time: int,
        height: int,
        width: int,
        reference: torch.Tensor,
        *,
        t_start: int = 0,
        steps_h: int | None = None,
        steps_w: int | None = None,
    ) -> torch.Tensor:
        steps_h = height // 2 if steps_h is None else steps_h
        steps_w = width // 2 if steps_w is None else steps_w
        device = reference.device
        dtype = reference.dtype
        ids = torch.zeros((time, steps_h, steps_w, 3), device=device, dtype=dtype)
        ids[..., 0] += torch.linspace(
            t_start,
            t_start + time - 1,
            time,
            device=device,
            dtype=dtype,
        )[:, None, None]
        ids[..., 1] += torch.linspace(
            0,
            height // 2 - 1,
            steps_h,
            device=device,
            dtype=dtype,
        )[None, :, None]
        ids[..., 2] += torch.linspace(
            0,
            width // 2 - 1,
            steps_w,
            device=device,
            dtype=dtype,
        )[None, None, :]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        audio_embed: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
        control_video: torch.Tensor | None = None,
        reference_motion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vision is not None:
            raise ValueError("Wan 2.2 S2V does not consume vision rows")
        for name, value in (("x", x), ("timesteps", timesteps), ("context", context)):
            if type(value) is not torch.Tensor or not value.is_floating_point():
                raise TypeError(f"{name} must be an exact floating torch.Tensor")
        if x.ndim != 5 or x.shape[1] != 16 or any(size <= 0 for size in x.shape):
            raise ValueError("S2V x must be non-empty [batch,16,time,height,width]")
        batch, _, target_frames, height, width = x.shape
        if timesteps.shape not in ((batch,), (batch, target_frames)):
            raise ValueError("S2V timesteps must be scalar per batch or one value per target frame")
        if (
            context.ndim != 3
            or context.shape[0] != batch
            or context.shape[2] != self.config.text_dim
        ):
            raise ValueError(f"S2V context must be [batch,text_rows,{self.config.text_dim}]")
        for name, value in (
            ("reference_latent", reference_latent),
            ("control_video", control_video),
            ("reference_motion", reference_motion),
        ):
            if value is not None and (
                type(value) is not torch.Tensor
                or value.ndim != 5
                or value.shape[0] != batch
                or value.shape[1] != 16
                or value.shape[-2:] != (height, width)
            ):
                raise ValueError(f"S2V {name} must match x batch, channels, and spatial geometry")
        if control_video is not None and control_video.shape[2] != target_frames:
            raise ValueError("S2V control_video must match the target temporal geometry")
        if reference_latent is not None and reference_latent.shape[2] != 1:
            raise ValueError("S2V reference_latent must contain exactly one latent frame")
        if audio_embed is not None and (
            type(audio_embed) is not torch.Tensor
            or audio_embed.shape[:3] != (batch, 25, 1024)
            or audio_embed.ndim != 4
            or audio_embed.shape[3] < target_frames * 4
        ):
            raise ValueError(
                "S2V audio_embed must be [batch,25,1024,samples] with four samples per frame"
            )

        original_shape = x.shape[2:]
        pad_h = (-height) % 2
        pad_w = (-width) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
            if control_video is not None:
                control_video = F.pad(control_video, (0, pad_w, 0, pad_h), mode="circular")
            if reference_latent is not None:
                reference_latent = F.pad(reference_latent, (0, pad_w, 0, pad_h), mode="circular")
            if reference_motion is not None:
                reference_motion = F.pad(reference_motion, (0, pad_w, 0, pad_h), mode="circular")
        height, width = x.shape[-2:]
        target = self.patch_embedding(x.float()).to(x.dtype)
        if control_video is not None:
            target = target + self.cond_encoder(control_video)
        grid = target.shape[2:]
        target = target.flatten(2).transpose(1, 2)
        target_rows = target.shape[1]
        cond_mask = self.trainable_cond_mask(torch.arange(3, device=x.device, dtype=torch.long)).to(
            x.dtype
        )
        hidden = target + cond_mask[0]
        freqs = self.rope_encode(target_frames, height, width, target)
        time_rows = (
            timesteps[:, None].expand(-1, target_frames) if timesteps.ndim == 1 else timesteps
        )
        if reference_latent is not None:
            reference = self.patch_embedding(reference_latent.float()).to(x.dtype)
            hidden = torch.cat(
                (hidden, reference.flatten(2).transpose(1, 2) + cond_mask[1]),
                dim=1,
            )
            freqs = torch.cat(
                (
                    freqs,
                    self.rope_encode(
                        1,
                        height,
                        width,
                        hidden,
                        t_start=max(30, target_frames + 9),
                    ),
                ),
                dim=1,
            )
            time_rows = torch.cat((time_rows, time_rows.new_zeros((batch, 1))), dim=1)
        if reference_motion is not None:
            motion, motion_freqs = self.frame_packer(reference_motion, self)
            hidden = torch.cat(
                (
                    hidden,
                    motion + cond_mask[2],
                ),
                dim=1,
            )
            freqs = torch.cat((freqs, motion_freqs), dim=1)
            time_rows = torch.cat(
                (torch.repeat_interleave(time_rows, 2, dim=1), time_rows.new_zeros((batch, 3))),
                dim=1,
            )

        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, time_rows.flatten()).to(hidden.dtype)
        ).reshape(batch, -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        context = self.text_embedding(context)
        audio_global: torch.Tensor | None = None
        audio_local: torch.Tensor | None = None
        if audio_embed is not None:
            audio_global, audio_local = self.casual_audio_encoder(
                audio_embed[..., : target_frames * 4]
            )
        with attention_kernel_context(self._attention_kernel, hidden.numel(), device=hidden.device):
            for index, block in enumerate(self.blocks):
                hidden = block(hidden, projected_time, freqs, context, None)
                if audio_local is not None:
                    assert audio_global is not None
                    hidden = self.audio_injector(
                        hidden,
                        index,
                        audio_local,
                        audio_global,
                        target_rows,
                    )
        hidden = self.head(hidden, time)[:, :target_rows]
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


__all__ = ["Wan22S2VModel"]
