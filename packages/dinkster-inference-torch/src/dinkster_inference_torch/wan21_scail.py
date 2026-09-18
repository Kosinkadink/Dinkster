"""Native Wan 2.1 SCAIL and SCAIL2 diffusion transformers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference import WAN21_SCAIL_14B, Wan21Config

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, CastOperations, Operations
from .wan21_model import Wan21Model, sinusoidal_embedding_1d

_DEFAULT_ATTENTION = select_attention("flux").kernel


class WanScailModel(Wan21Model):
    """SCAIL backbone with separate reference, pose, and identity-mask streams."""

    def __init__(
        self,
        config: Wan21Config = WAN21_SCAIL_14B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config.model_variant not in ("scail", "scail2"):
            raise ValueError("WanScailModel requires a SCAIL model configuration")
        super().__init__(config, operations=operations, attention_kernel=attention_kernel)
        self.patch_embedding_pose = CastOperations(torch.float32).conv3d(
            20,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.patch_embedding_mask = (
            CastOperations(torch.float32).conv3d(
                28,
                config.hidden_size,
                config.patch_size,
                stride=config.patch_size,
            )
            if config.model_variant == "scail2"
            else None
        )

    @staticmethod
    def _pad(value: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
        pad_t = (-value.shape[2]) % patch_size[0]
        pad_h = (-value.shape[3]) % patch_size[1]
        pad_w = (-value.shape[4]) % patch_size[2]
        if pad_t or pad_h or pad_w:
            return F.pad(value, (0, pad_w, 0, pad_h, 0, pad_t), mode="circular")
        return value

    def _scail_rope(
        self,
        target_grid: tuple[int, int, int],
        reference_frames: int,
        pose_grid: tuple[int, int, int] | None,
        pose_spatial_scale: tuple[float, float] | None,
        x: torch.Tensor,
        *,
        replacement: bool,
    ) -> torch.Tensor:
        target_time, height, width = target_grid

        def encode(
            shape: tuple[int, int, int],
            *,
            time_start: float = 0.0,
            height_start: float = 0.0,
            width_start: float = 0.0,
            height_scale: float = 1.0,
            width_scale: float = 1.0,
        ) -> torch.Tensor:
            time, rows, columns = shape
            ids = torch.empty((time, rows, columns, 3), device=x.device, dtype=x.dtype)
            ids[..., 0] = torch.linspace(
                time_start,
                time_start + time - 1,
                time,
                device=x.device,
                dtype=x.dtype,
            )[:, None, None]
            ids[..., 1] = torch.linspace(
                height_start,
                height_start + (rows - 1) * height_scale,
                rows,
                device=x.device,
                dtype=x.dtype,
            )[None, :, None]
            ids[..., 2] = torch.linspace(
                width_start,
                width_start + (columns - 1) * width_scale,
                columns,
                device=x.device,
                dtype=x.dtype,
            )[None, None, :]
            return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

        if replacement:
            video_start = float(max(reference_frames - 1, 0))
            parts: list[torch.Tensor] = []
            if reference_frames:
                parts.append(
                    encode(
                        (reference_frames, height, width),
                        height_start=120.0,
                    )
                )
            parts.append(encode(target_grid, time_start=video_start))
        else:
            video_start = float(reference_frames)
            parts = [encode((reference_frames + target_time, height, width))]

        if pose_grid is not None:
            assert pose_spatial_scale is not None
            height_scale, width_scale = pose_spatial_scale
            parts.append(
                encode(
                    pose_grid,
                    time_start=video_start,
                    height_start=(height_scale - 1.0) / 2.0,
                    width_start=120.0 + (width_scale - 1.0) / 2.0,
                    height_scale=height_scale,
                    width_scale=width_scale,
                )
            )
        return torch.cat(parts, dim=1)

    def _validate_scail(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None,
        reference_latent: torch.Tensor | None,
        pose_latents: torch.Tensor | None,
        reference_mask: torch.Tensor | None,
        driving_mask: torch.Tensor | None,
        replacement: bool,
    ) -> None:
        super()._validate(
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
        if tuple(timesteps.shape) != (x.shape[0],):
            raise ValueError("Wan SCAIL timesteps must have one value per batch item")
        if type(replacement) is not bool:
            raise TypeError("replacement must be an exact bool")

        def validate_latent(
            name: str,
            value: torch.Tensor | None,
            channels: int,
        ) -> None:
            if value is None:
                return
            if (
                type(value) is not torch.Tensor
                or not value.is_floating_point()
                or value.layout != torch.strided
            ):
                raise TypeError(f"{name} must be an exact strided floating torch.Tensor")
            if (
                value.ndim != 5
                or value.shape[0] != x.shape[0]
                or value.shape[1] != channels
                or any(size == 0 for size in value.shape[2:])
            ):
                raise ValueError(f"{name} must have shape [batch,{channels},time,height,width]")

        validate_latent("reference_latent", reference_latent, 20)
        validate_latent("pose_latents", pose_latents, 20)
        validate_latent("reference_mask", reference_mask, 28)
        validate_latent("driving_mask", driving_mask, 28)
        if reference_latent is not None and reference_latent.shape[3:] != x.shape[3:]:
            raise ValueError("reference_latent must match the target spatial shape")
        if reference_mask is not None:
            if self.patch_embedding_mask is None:
                raise ValueError("identity masks are supported only by SCAIL2")
            reference_frames = 0 if reference_latent is None else reference_latent.shape[2]
            if reference_mask.shape[2:] != (reference_frames + x.shape[2], *x.shape[3:]):
                raise ValueError(
                    "reference_mask must match the combined reference and target shape"
                )
        if driving_mask is not None:
            if self.patch_embedding_mask is None:
                raise ValueError("identity masks are supported only by SCAIL2")
            if pose_latents is None or driving_mask.shape[2:] != pose_latents.shape[2:]:
                raise ValueError("driving_mask must match pose_latents")

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        reference_latent: torch.Tensor | None = None,
        pose_latents: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        driving_mask: torch.Tensor | None = None,
        replacement: bool = False,
    ) -> torch.Tensor:
        self._validate_scail(
            x,
            timesteps,
            context,
            vision,
            reference_latent,
            pose_latents,
            reference_mask,
            driving_mask,
            replacement,
        )
        original_shape = x.shape[2:]
        x = self._pad(x, self.config.patch_size)
        reference_frames = 0
        if reference_latent is not None:
            reference_latent = self._pad(reference_latent, self.config.patch_size)
            reference_frames = reference_latent.shape[2] // self.config.patch_size[0]
            x = torch.cat((reference_latent, x), dim=2)
        x = self.patch_embedding(x.float()).to(x.dtype)
        if reference_mask is not None:
            assert self.patch_embedding_mask is not None
            reference_mask = self._pad(reference_mask, self.config.patch_size)
            x = x + self.patch_embedding_mask(reference_mask.float()).to(x.dtype)
        combined_grid = (x.shape[2], x.shape[3], x.shape[4])
        target_grid = (combined_grid[0] - reference_frames, combined_grid[1], combined_grid[2])

        pose_grid: tuple[int, int, int] | None = None
        pose_spatial_scale: tuple[float, float] | None = None
        pose_rows = 0
        if pose_latents is not None:
            pose_latents = self._pad(pose_latents, self.config.patch_size)
            pose_spatial_scale = (
                original_shape[1] / pose_latents.shape[3],
                original_shape[2] / pose_latents.shape[4],
            )
            pose = self.patch_embedding_pose(pose_latents.float()).to(x.dtype)
            if driving_mask is not None:
                assert self.patch_embedding_mask is not None
                driving_mask = self._pad(driving_mask, self.config.patch_size)
                pose = pose + self.patch_embedding_mask(driving_mask.float()).to(x.dtype)
            pose_grid = (pose.shape[2], pose.shape[3], pose.shape[4])
            pose = pose.flatten(2).transpose(1, 2)
            pose_rows = pose.shape[1]
        else:
            pose = None

        freqs = self._scail_rope(
            target_grid,
            reference_frames,
            pose_grid,
            pose_spatial_scale,
            x,
            replacement=replacement,
        )
        x = x.flatten(2).transpose(1, 2)
        if pose is not None:
            x = torch.cat((x, pose), dim=1)

        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps).to(x.dtype)
        ).reshape(timesteps.shape[0], 1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        context = self.text_embedding(context)
        image_rows: int | None = None
        if vision is not None:
            assert self.img_emb is not None
            projected_vision = self.img_emb(vision)
            image_rows = projected_vision.shape[1]
            context = torch.cat((projected_vision, context), dim=1)

        for block in self.blocks:
            x = block(x, projected_time, freqs, context, image_rows)
        x = self.head(x, time)
        if pose_rows:
            x = x[:, :-pose_rows]
        reference_rows = reference_frames * combined_grid[1] * combined_grid[2]
        if reference_rows:
            x = x[:, reference_rows:]

        batch = x.shape[0]
        patch_t, patch_h, patch_w = self.config.patch_size
        x = x.view(batch, *target_grid, patch_t, patch_h, patch_w, self.config.out_channels)
        x = torch.einsum("bthwpqrc->bctphqwr", x)
        x = x.reshape(
            batch,
            self.config.out_channels,
            target_grid[0] * patch_t,
            target_grid[1] * patch_h,
            target_grid[2] * patch_w,
        )
        return x[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


__all__ = ["WanScailModel"]
