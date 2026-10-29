"""Native Pixal3D Neighborhood Attention Filtering feature upsampler."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from dinkster_inference import PIXAL3D_NAF, NAFConfig

from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight


def _axis_neighborhood_indices(
    length: int,
    source_length: int,
    kernel: int,
    dilation: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    residues = positions.remainder(dilation)
    positions_in_class = positions.div(dilation, rounding_mode="floor")
    class_sizes = (length - 1 - residues).div(dilation, rounding_mode="floor") + 1
    if bool((class_sizes < kernel).any()):
        raise ValueError("NAF neighborhood is larger than a dilation residue class")
    radius = kernel // 2
    starts = torch.minimum(
        (positions_in_class - radius).clamp_min(0),
        class_sizes - kernel,
    )
    offsets = torch.arange(kernel, device=device)
    high_indices = residues[:, None] + (starts[:, None] + offsets) * dilation
    return ((2 * high_indices + 1) * source_length).div(2 * length, rounding_mode="floor")


def neighborhood_attention_2d(
    query: torch.Tensor,
    key_lr: torch.Tensor,
    value_lr: torch.Tensor,
    *,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    scale: float,
    tile: int = 128,
    value_chunk: int = 64,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tiled pure-torch NAF attention without materializing high-resolution K/V."""
    batch, height, width, heads, query_channels = query.shape
    value_channels = value_lr.shape[-1]
    kernel_h, kernel_w = kernel_size
    dilation_h, dilation_w = dilation
    out = (
        torch.empty(
            (batch, heads, value_channels, height, width),
            device=query.device,
            dtype=query.dtype,
        )
        if output is None
        else output
    )
    tile_h = min(tile, height)
    tile_w = min(tile, width)
    chunk = min(value_chunk, value_channels)
    windows = kernel_h * kernel_w
    key_channels = key_lr.permute(0, 3, 4, 1, 2)
    value_channels_lr = value_lr.permute(0, 3, 4, 1, 2)
    height_indices = _axis_neighborhood_indices(
        height,
        key_lr.shape[1],
        kernel_h,
        dilation_h,
        device=query.device,
    )
    width_indices = _axis_neighborhood_indices(
        width,
        key_lr.shape[2],
        kernel_w,
        dilation_w,
        device=query.device,
    )

    for height_start in range(0, height, tile_h):
        for width_start in range(0, width, tile_w):
            height_end = min(height_start + tile_h, height)
            width_end = min(width_start + tile_w, width)
            actual_h = height_end - height_start
            actual_w = width_end - width_start
            row_indices = height_indices[height_start:height_end, None, :, None]
            column_indices = width_indices[None, width_start:width_end, None, :]
            key_windows = key_channels[..., row_indices, column_indices]
            key_windows = key_windows.permute(0, 1, 3, 4, 5, 6, 2).reshape(
                batch,
                heads,
                actual_h * actual_w,
                windows,
                query_channels,
            )
            query_tile = (
                query[:, height_start:height_end, width_start:width_end]
                .permute(0, 3, 1, 2, 4)
                .reshape(batch, heads, actual_h * actual_w, 1, query_channels)
            )
            attention = (torch.matmul(query_tile, key_windows.transpose(-1, -2)) * scale).softmax(
                dim=-1
            )

            for channel_start in range(0, value_channels, chunk):
                channel_end = min(channel_start + chunk, value_channels)
                value_windows = value_channels_lr[
                    :, :, channel_start:channel_end, row_indices, column_indices
                ]
                value_windows = value_windows.permute(0, 1, 3, 4, 5, 6, 2).reshape(
                    batch,
                    heads,
                    actual_h * actual_w,
                    windows,
                    channel_end - channel_start,
                )
                attended = torch.matmul(attention, value_windows).squeeze(-2)
                attended = attended.view(
                    batch,
                    heads,
                    actual_h,
                    actual_w,
                    channel_end - channel_start,
                ).permute(0, 1, 4, 2, 3)
                out[
                    :,
                    :,
                    channel_start:channel_end,
                    height_start:height_end,
                    width_start:width_end,
                ] = attended
    return out


class NAFCrossAttention(torch.nn.Module):
    def __init__(self, config: NAFConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kernel_size = (config.kernel_size, config.kernel_size)
        self.scale = (config.channels // config.attention_heads) ** -0.5

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        return (
            value.view(batch, self.heads, channels // self.heads, height, width)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        high_h, high_w = query.shape[-2:]
        low_h, low_w = key.shape[-2:]
        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key).to(query.dtype)
        value_heads = self._split_heads(value).to(query.dtype)
        output_heads = (
            None
            if output is None
            else output.view(
                query.shape[0], self.heads, value.shape[1] // self.heads, high_h, high_w
            )
        )
        attended = neighborhood_attention_2d(
            query_heads,
            key_heads,
            value_heads,
            kernel_size=self.kernel_size,
            dilation=(high_h // low_h, high_w // low_w),
            scale=self.scale,
            output=output_heads,
        )
        return attended.view(query.shape[0], -1, high_h, high_w)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class NAFRope(ResidencyRouted, torch.nn.Module):
    def __init__(self, config: NAFConfig) -> None:
        super().__init__()
        self.heads = config.rope_heads
        head_dim = config.channels // config.rope_heads
        self.periods = torch.nn.Parameter(torch.empty(head_dim // 4), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        head_dim = channels // self.heads
        reshaped = value.view(batch, self.heads, head_dim, height, width)
        reshaped = reshaped.permute(0, 1, 3, 4, 2).reshape(
            batch, self.heads, height * width, head_dim
        )
        binding = self._offloaded_residency()
        if binding is None:
            periods = cast_weight(self.periods, dtype=torch.float32, device=value.device)
        else:
            with binding.lease() as lease:
                periods = lease.get("periods", dtype=torch.float32)
        coords_h = torch.arange(0.5, height, device=value.device, dtype=torch.float32) / height
        coords_w = torch.arange(0.5, width, device=value.device, dtype=torch.float32) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1).mul(2.0).sub(1.0)
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cosine = angles.cos().to(value.dtype)
        sine = angles.sin().to(value.dtype)
        reshaped = reshaped * cosine + _rotate_half(reshaped) * sine
        return (
            reshaped.view(batch, self.heads, height, width, head_dim)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch, channels, height, width)
        )


class NAFEncoderBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel: int, *, operations: Operations) -> None:
        super().__init__()
        self.norm1 = operations.group_norm(channels, num_groups=8, eps=1e-5)
        self.conv1 = operations.conv2d(channels, channels, kernel, padding=kernel // 2)
        self.conv1.padding_mode = "reflect"
        self.norm2 = operations.group_norm(channels, num_groups=8, eps=1e-5)
        self.conv2 = operations.conv2d(channels, channels, kernel, padding=kernel // 2)
        self.conv2.padding_mode = "reflect"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(F.silu(self.norm1(value)))
        return self.conv2(F.silu(self.norm2(value)))


class NAFEncoderBranch(torch.nn.Sequential):
    def __init__(
        self,
        channels: int,
        kernel: int,
        layers: int,
        *,
        operations: Operations,
    ) -> None:
        initial = operations.conv2d(3, channels, kernel, padding=kernel // 2)
        initial.padding_mode = "reflect"
        modules: list[torch.nn.Module] = [initial]
        modules.extend(
            NAFEncoderBlock(channels, kernel, operations=operations) for _ in range(layers)
        )
        super().__init__(*modules)


class NAFImageEncoder(torch.nn.Module):
    def __init__(self, config: NAFConfig, *, operations: Operations) -> None:
        super().__init__()
        half = config.channels // 2
        self.encoder = NAFEncoderBranch(half, 1, config.image_layers, operations=operations)
        self.sem_encoder = NAFEncoderBranch(half, 3, config.image_layers, operations=operations)
        self.rope = NAFRope(config)

    def forward(self, image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        output_h, output_w = output_size
        if image.shape[-2] > 4 * output_h or image.shape[-1] > 4 * output_w:
            image = F.interpolate(
                image,
                size=(min(image.shape[-2], 4 * output_h), min(image.shape[-1], 4 * output_w)),
                mode="bilinear",
                align_corners=False,
            )
        encoded = torch.cat((self.encoder(image), self.sem_encoder(image)), dim=1)
        return self.rope(F.adaptive_avg_pool2d(encoded, output_size))


class NAF(torch.nn.Module):
    def __init__(
        self,
        config: NAFConfig = PIXAL3D_NAF,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = NAFImageEncoder(config, operations=operations)
        self.upsampler = NAFCrossAttention(config)

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.image_encoder(image, output_size)
        key = F.adaptive_avg_pool2d(query, (features.shape[-2], features.shape[-1]))
        return self.upsampler(query, key, features, output=output)


__all__ = ["NAF", "NAFCrossAttention", "neighborhood_attention_2d"]
