"""LTX-2.x learned spatial latent upscaler."""

from __future__ import annotations

import torch
from dinkster_inference import LTXLatentUpsamplerConfig

from .operations import INITLESS, Operations


class PixelShuffle2D(torch.nn.Module):
    """Two-dimensional pixel shuffle with the reference channel ordering."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, packed_channels, height, width = value.shape
        factor = self.factor
        channels = packed_channels // (factor * factor)
        return (
            value.reshape(batch, channels, factor, factor, height, width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, channels, height * factor, width * factor)
        )


class LTXLatentUpsamplerResBlock(torch.nn.Module):
    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv1 = operations.conv3d(channels, channels, 3, padding=1)
        self.norm1 = operations.group_norm(channels, num_groups=32, eps=1e-5)
        self.conv2 = operations.conv3d(channels, channels, 3, padding=1)
        self.norm2 = operations.group_norm(channels, num_groups=32, eps=1e-5)
        self.activation = torch.nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)


class LTXLatentUpsampler(torch.nn.Module):
    """The official x2 spatial latent upscaler, without einops."""

    def __init__(
        self,
        config: LTXLatentUpsamplerConfig,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if (
            config.dims != 3
            or not config.spatial_upsample
            or config.temporal_upsample
            or config.spatial_scale != 2.0
            or config.rational_resampler
        ):
            raise ValueError("LTX latent upscaler requires the x2 spatial 3D profile")
        channels = config.mid_channels
        self.config = config
        self.initial_conv = operations.conv3d(config.in_channels, channels, 3, padding=1)
        self.initial_norm = operations.group_norm(channels, num_groups=32, eps=1e-5)
        self.initial_activation = torch.nn.SiLU()
        self.res_blocks = torch.nn.ModuleList(
            LTXLatentUpsamplerResBlock(channels, operations=operations)
            for _ in range(config.num_blocks_per_stage)
        )
        self.upsampler = torch.nn.Sequential(
            operations.conv2d(channels, 4 * channels, 3, padding=1),
            PixelShuffle2D(2),
        )
        self.post_upsample_res_blocks = torch.nn.ModuleList(
            LTXLatentUpsamplerResBlock(channels, operations=operations)
            for _ in range(config.num_blocks_per_stage)
        )
        self.final_conv = operations.conv3d(channels, config.in_channels, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _channels, frames, _height, _width = latent.shape
        value = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            value = block(value)
        channels, height, width = value.shape[1], value.shape[3], value.shape[4]
        value = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        value = self.upsampler(value)
        height, width = value.shape[2], value.shape[3]
        value = value.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        for block in self.post_upsample_res_blocks:
            value = block(value)
        return self.final_conv(value)


__all__ = ["LTXLatentUpsampler", "LTXLatentUpsamplerResBlock", "PixelShuffle2D"]
