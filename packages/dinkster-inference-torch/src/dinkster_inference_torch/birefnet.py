"""BiRefNet architecture used by ComfyUI's background-removal model.

The module structure and forward math match ComfyUI commit
c67885b14556cf3e4e061862925282d403d09862. The model itself is MIT-licensed;
this adaptation uses native torch operations and component residency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import deform_conv2d  # pyright: ignore[reportMissingTypeStubs]

from .operations import INITLESS, Operations, ResidencyRouted, materialized_conv2d_parameters

_CONTEXT_CHANNELS = (384, 768, 1536)


def _window_partition(value: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = value.shape
    value = value.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return value.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    value = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return value.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        output = out_features or in_features
        hidden = hidden_features or in_features
        self.fc1 = operations.linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = operations.linear(hidden, output)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))


class WindowAttention(ResidencyRouted, nn.Module):
    relative_position_index: torch.Tensor

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        entries = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.empty(entries, num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = operations.linear(dim, dim * 3, bias=qkv_bias)
        self.proj = operations.linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def _relative_bias(self) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self.relative_position_bias_table[self.relative_position_index.long().view(-1)]
        with binding.lease() as lease:
            table = lease.get(
                "relative_position_bias_table", dtype=self.relative_position_bias_table.dtype
            )
            index = lease.get("relative_position_index", dtype=self.relative_position_index.dtype)
            return table[index.long().view(-1)]

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, values = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_bias = self._relative_bias().view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_bias = relative_bias.permute(2, 0, 1).contiguous().to(attention)
        attention = attention + relative_bias.unsqueeze(0)

        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(
                batch // windows,
                windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.softmax(attention)
        value = (attention @ values).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(value)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = operations.layer_norm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            operations=operations,
        )
        self.norm2 = operations.layer_norm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), operations=operations)
        self.height = 0
        self.width = 0

    def forward(self, value: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        batch, _, channels = value.shape
        height, width = self.height, self.width
        shortcut = value
        value = self.norm1(value).view(batch, height, width, channels)

        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        value = F.pad(value, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = value.shape[1:3]

        if self.shift_size > 0:
            shifted = torch.roll(
                value,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
            attention_mask: torch.Tensor | None = mask_matrix
        else:
            shifted = value
            attention_mask = None

        windows = _window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attended = self.attn(windows, mask=attention_mask)
        attended = attended.view(-1, self.window_size, self.window_size, channels)
        shifted = _window_reverse(
            attended,
            self.window_size,
            padded_height,
            padded_width,
        )

        if self.shift_size > 0:
            value = torch.roll(
                shifted,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            value = shifted
        if pad_right > 0 or pad_bottom > 0:
            value = value[:, :height, :width, :].contiguous()

        value = value.view(batch, height * width, channels)
        value = shortcut + value
        return value + self.mlp(self.norm2(value))


class PatchMerging(nn.Module):
    def __init__(self, dim: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.reduction = operations.linear(4 * dim, 2 * dim, bias=False)
        self.norm = operations.layer_norm(4 * dim)

    def forward(self, value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = value.shape
        value = value.view(batch, height, width, channels)
        if height % 2 == 1 or width % 2 == 1:
            value = F.pad(value, (0, 0, 0, width % 2, 0, height % 2))

        parts = (
            value[:, 0::2, 0::2, :],
            value[:, 1::2, 0::2, :],
            value[:, 0::2, 1::2, :],
            value[:, 1::2, 1::2, :],
        )
        value = torch.cat(parts, -1).view(batch, -1, 4 * channels)
        return self.reduction(self.norm(value))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        downsample: type[PatchMerging] | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    operations=operations,
                )
                for index in range(depth)
            ]
        )
        self.downsample = None if downsample is None else downsample(dim, operations=operations)

    def forward(
        self,
        value: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        padded_height = int(np.ceil(height / self.window_size)) * self.window_size
        padded_width = int(np.ceil(width / self.window_size)) * self.window_size
        image_mask = torch.zeros((1, padded_height, padded_width, 1), device=value.device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = count
                count += 1

        mask_windows = _window_partition(image_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)

        for block in self.blocks:
            assert isinstance(block, SwinTransformerBlock)
            block.height, block.width = height, width
            value = block(value, attention_mask)
        if self.downsample is None:
            return value, height, width, value, height, width
        down = self.downsample(value, height, width)
        return value, height, width, down, (height + 1) // 2, (width + 1) // 2


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        *,
        normalize: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.proj = operations.conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = operations.layer_norm(embed_dim) if normalize else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, _, height, width = value.size()
        if width % self.patch_size[1] != 0:
            value = F.pad(value, (0, self.patch_size[1] - width % self.patch_size[1]))
        if height % self.patch_size[0] != 0:
            value = F.pad(value, (0, 0, 0, self.patch_size[0] - height % self.patch_size[0]))
        value = self.proj(value)
        if self.norm is not None:
            output_height, output_width = value.shape[2:]
            value = self.norm(value.flatten(2).transpose(1, 2))
            value = value.transpose(1, 2).view(
                -1,
                self.embed_dim,
                output_height,
                output_width,
            )
        return value


class SwinTransformer(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.out_indices = tuple(out_indices)
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            normalize=patch_norm,
            operations=operations,
        )
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2**index),
                    depth=depths[index],
                    num_heads=num_heads[index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    downsample=PatchMerging if index < self.num_layers - 1 else None,
                    operations=operations,
                )
            )
        self.num_features = [int(embed_dim * 2**index) for index in range(self.num_layers)]
        for index in self.out_indices:
            self.add_module(f"norm{index}", operations.layer_norm(self.num_features[index]))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = self.patch_embed(value)
        height, width = value.shape[2:]
        outputs: list[torch.Tensor] = []
        value = value.flatten(2).transpose(1, 2)
        for index, layer in enumerate(self.layers):
            assert isinstance(layer, BasicLayer)
            layer_output, output_height, output_width, value, height, width = layer(
                value,
                height,
                width,
            )
            if index in self.out_indices:
                norm = getattr(self, f"norm{index}")
                layer_output = norm(layer_output)
                output = layer_output.view(
                    -1,
                    output_height,
                    output_width,
                    self.num_features[index],
                )
                outputs.append(output.permute(0, 3, 1, 2).contiguous())
        return tuple(outputs)


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int = 1,
        *,
        bias: bool = False,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        kernel = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = (padding, padding)
        self.offset_conv = operations.conv2d(
            in_channels,
            2 * kernel[0] * kernel[1],
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
            bias=True,
        )
        self.modulator_conv = operations.conv2d(
            in_channels,
            kernel[0] * kernel[1],
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
            bias=True,
        )
        self.regular_conv = operations.conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(value)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(value))
        with materialized_conv2d_parameters(self.regular_conv) as (weight, _bias):
            return deform_conv2d(
                input=value,
                offset=offset,
                weight=weight,
                bias=None,
                padding=self.padding,
                mask=modulator,
                stride=self.stride,
            )


class BasicDecBlk(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        inter_channels: int = 64,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        inter_channels = 64
        self.conv_in = operations.conv2d(in_channels, inter_channels, 3, stride=1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels, operations=operations)
        self.conv_out = operations.conv2d(inter_channels, out_channels, 3, stride=1, padding=1)
        self.bn_in = operations.batch_norm2d(inter_channels)
        self.bn_out = operations.batch_norm2d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.relu_in(self.bn_in(self.conv_in(value)))
        return self.bn_out(self.conv_out(self.dec_att(value)))


class BasicLatBlk(nn.Module):
    def __init__(
        self, in_channels: int = 64, out_channels: int = 64, *, operations: Operations = INITLESS
    ) -> None:
        super().__init__()
        self.conv = operations.conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class _ASPPModuleDeformable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        planes: int,
        kernel_size: int,
        padding: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            operations=operations,
        )
        self.bn = operations.batch_norm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(value)))


class ASPPDeformable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        parallel_block_sizes: Sequence[int] = (1, 3, 7),
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        output_channels = in_channels if out_channels is None else out_channels
        intermediate_channels = 256
        self.aspp1 = _ASPPModuleDeformable(
            in_channels, intermediate_channels, 1, 0, operations=operations
        )
        self.aspp_deforms = nn.ModuleList(
            [
                _ASPPModuleDeformable(
                    in_channels,
                    intermediate_channels,
                    size,
                    size // 2,
                    operations=operations,
                )
                for size in parallel_block_sizes
            ]
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            operations.conv2d(in_channels, intermediate_channels, 1, stride=1, bias=False),
            operations.batch_norm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        branches = 2 + len(self.aspp_deforms)
        self.conv1 = operations.conv2d(
            intermediate_channels * branches,
            output_channels,
            1,
            bias=False,
        )
        self.bn1 = operations.batch_norm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.aspp1(value)
        deformed = [branch(value) for branch in self.aspp_deforms]
        pooled = self.global_avg_pool(value)
        pooled = F.interpolate(pooled, size=first.shape[2:], mode="bilinear", align_corners=True)
        value = torch.cat((first, *deformed, pooled), dim=1)
        return self.relu(self.bn1(self.conv1(value)))


class BiRefNet(nn.Module):
    def __init__(self, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.bb = SwinTransformer(
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            window_size=12,
            operations=operations,
        )
        channels = [3072, 1536, 768, 384]
        self.squeeze_module = nn.Sequential(
            BasicDecBlk(channels[0] + sum(_CONTEXT_CHANNELS), channels[0], operations=operations)
        )
        self.decoder = Decoder(channels, operations=operations)

    def forward_enc(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second, third, fourth = self.bb(value)
        _, _, height, width = value.shape
        half = F.interpolate(
            value,
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=True,
        )
        half_first, half_second, half_third, half_fourth = self.bb(half)
        first = torch.cat(
            (
                first,
                F.interpolate(
                    half_first,
                    size=first.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            dim=1,
        )
        second = torch.cat(
            (
                second,
                F.interpolate(
                    half_second,
                    size=second.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            dim=1,
        )
        third = torch.cat(
            (
                third,
                F.interpolate(
                    half_third,
                    size=third.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            dim=1,
        )
        fourth = torch.cat(
            (
                fourth,
                F.interpolate(
                    half_fourth,
                    size=fourth.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            dim=1,
        )
        context = (
            F.interpolate(first, size=fourth.shape[2:], mode="bilinear", align_corners=True),
            F.interpolate(second, size=fourth.shape[2:], mode="bilinear", align_corners=True),
            F.interpolate(third, size=fourth.shape[2:], mode="bilinear", align_corners=True),
        )
        fourth = torch.cat((*context, fourth), dim=1)
        return first, second, third, fourth

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        first, second, third, fourth = self.forward_enc(pixel_values)
        fourth = self.squeeze_module(fourth)
        return self.decoder((pixel_values, first, second, third, fourth))


class Decoder(nn.Module):
    def __init__(self, channels: Sequence[int], *, operations: Operations = INITLESS) -> None:
        super().__init__()
        decoder_block = BasicDecBlk
        lateral_block = BasicLatBlk
        simple_convs = SimpleConvs
        self.split = True
        intermediate_channels = 64

        self.ipt_blk5 = simple_convs(
            2**10 * 3, channels[0] // 8, intermediate_channels, operations=operations
        )
        self.ipt_blk4 = simple_convs(
            2**8 * 3, channels[0] // 8, intermediate_channels, operations=operations
        )
        self.ipt_blk3 = simple_convs(
            2**6 * 3, channels[1] // 8, intermediate_channels, operations=operations
        )
        self.ipt_blk2 = simple_convs(
            2**4 * 3, channels[2] // 8, intermediate_channels, operations=operations
        )
        self.ipt_blk1 = simple_convs(
            3, channels[3] // 8, intermediate_channels, operations=operations
        )

        self.decoder_block4 = decoder_block(
            channels[0] + channels[0] // 8,
            channels[1],
            operations=operations,
        )
        self.decoder_block3 = decoder_block(
            channels[1] + channels[0] // 8,
            channels[2],
            operations=operations,
        )
        self.decoder_block2 = decoder_block(
            channels[2] + channels[1] // 8,
            channels[3],
            operations=operations,
        )
        self.decoder_block1 = decoder_block(
            channels[3] + channels[2] // 8,
            channels[3] // 2,
            operations=operations,
        )
        self.conv_out1 = nn.Sequential(
            operations.conv2d(channels[3] // 2 + channels[3] // 8, 1, 1, stride=1, padding=0)
        )
        self.lateral_block4 = lateral_block(channels[1], channels[1], operations=operations)
        self.lateral_block3 = lateral_block(channels[2], channels[2], operations=operations)
        self.lateral_block2 = lateral_block(channels[3], channels[3], operations=operations)

        self.conv_ms_spvn_4 = operations.conv2d(channels[1], 1, 1, stride=1, padding=0)
        self.conv_ms_spvn_3 = operations.conv2d(channels[2], 1, 1, stride=1, padding=0)
        self.conv_ms_spvn_2 = operations.conv2d(channels[3], 1, 1, stride=1, padding=0)

        guidance_channels = 16
        self.gdt_convs_4 = nn.Sequential(
            operations.conv2d(channels[0] // 2, guidance_channels, 3, stride=1, padding=1),
            operations.batch_norm2d(guidance_channels),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_3 = nn.Sequential(
            operations.conv2d(channels[1] // 2, guidance_channels, 3, stride=1, padding=1),
            operations.batch_norm2d(guidance_channels),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_2 = nn.Sequential(
            operations.conv2d(channels[2] // 2, guidance_channels, 3, stride=1, padding=1),
            operations.batch_norm2d(guidance_channels),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_pred_2 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )
        self.gdt_convs_pred_3 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )
        self.gdt_convs_pred_4 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )
        self.gdt_convs_attn_2 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )
        self.gdt_convs_attn_3 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )
        self.gdt_convs_attn_4 = nn.Sequential(
            operations.conv2d(guidance_channels, 1, 1, stride=1, padding=0)
        )

    @staticmethod
    def get_patches_batch(value: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        size_height, size_width = patch.shape[2:]
        batches: list[torch.Tensor] = []
        for index in range(value.shape[0]):
            columns = torch.split(value[index], split_size_or_sections=size_width, dim=-1)
            patches: list[torch.Tensor] = []
            for column in columns:
                patches.extend(
                    part.unsqueeze(0)
                    for part in torch.split(
                        column,
                        split_size_or_sections=size_height,
                        dim=-2,
                    )
                )
            batches.append(torch.cat(patches, dim=1))
        return torch.cat(batches, dim=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        value, first, second, third, fourth = features

        patches = self.get_patches_batch(value, fourth)
        fourth = torch.cat(
            (
                fourth,
                self.ipt_blk5(
                    F.interpolate(
                        patches,
                        size=fourth.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                ),
            ),
            dim=1,
        )
        decoded_fourth = self.decoder_block4(fourth)
        guidance_fourth = self.gdt_convs_4(decoded_fourth)
        decoded_fourth = decoded_fourth * self.gdt_convs_attn_4(guidance_fourth).sigmoid()
        decoded_third = F.interpolate(
            decoded_fourth,
            size=third.shape[2:],
            mode="bilinear",
            align_corners=True,
        ) + self.lateral_block4(third)

        patches = self.get_patches_batch(value, decoded_third)
        decoded_third = torch.cat(
            (
                decoded_third,
                self.ipt_blk4(
                    F.interpolate(
                        patches,
                        size=third.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                ),
            ),
            dim=1,
        )
        decoded_third = self.decoder_block3(decoded_third)
        guidance_third = self.gdt_convs_3(decoded_third)
        decoded_third = decoded_third * self.gdt_convs_attn_3(guidance_third).sigmoid()
        decoded_second = F.interpolate(
            decoded_third,
            size=second.shape[2:],
            mode="bilinear",
            align_corners=True,
        ) + self.lateral_block3(second)

        patches = self.get_patches_batch(value, decoded_second)
        decoded_second = torch.cat(
            (
                decoded_second,
                self.ipt_blk3(
                    F.interpolate(
                        patches,
                        size=second.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                ),
            ),
            dim=1,
        )
        decoded_second = self.decoder_block2(decoded_second)
        guidance_second = self.gdt_convs_2(decoded_second)
        decoded_second = decoded_second * self.gdt_convs_attn_2(guidance_second).sigmoid()
        decoded_first = F.interpolate(
            decoded_second,
            size=first.shape[2:],
            mode="bilinear",
            align_corners=True,
        ) + self.lateral_block2(first)

        patches = self.get_patches_batch(value, decoded_first)
        decoded_first = torch.cat(
            (
                decoded_first,
                self.ipt_blk2(
                    F.interpolate(
                        patches,
                        size=first.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                ),
            ),
            dim=1,
        )
        decoded_first = self.decoder_block1(decoded_first)
        decoded_first = F.interpolate(
            decoded_first,
            size=value.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        patches = self.get_patches_batch(value, decoded_first)
        decoded_first = torch.cat(
            (
                decoded_first,
                self.ipt_blk1(
                    F.interpolate(
                        patches,
                        size=value.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                ),
            ),
            dim=1,
        )
        return self.conv_out1(decoded_first)


class SimpleConvs(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int = 64,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.conv1 = operations.conv2d(in_channels, inter_channels, 3, stride=1, padding=1)
        self.conv_out = operations.conv2d(inter_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(value))


__all__ = ["BiRefNet"]
