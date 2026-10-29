"""Wan 2.2 Animate motion and face modules."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn.functional as F

from .attention import AttentionKernel
from .operations import Operations, ResidencyRouted
from .ops import cast_weight


class _DirectState(ResidencyRouted):
    @contextmanager
    def _materialized(
        self,
        reference: torch.Tensor,
        *names: str,
    ) -> Generator[dict[str, torch.Tensor]]:
        module = cast("torch.nn.Module", self)
        binding = self._offloaded_residency()
        if binding is None:
            yield {
                name: cast_weight(
                    cast("torch.Tensor", getattr(module, name)),
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for name in names
            }
            return
        with binding.lease() as lease:
            yield {name: lease.get(name, dtype=reference.dtype) for name in names}


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.conv = operations.conv1d(input_channels, output_channels, 3, stride=stride)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(value, (2, 0), mode="replicate"))


class FaceEncoder(_DirectState, torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.num_heads = 4
        self.conv1_local = CausalConv1d(512, 1024 * self.num_heads, operations=operations)
        self.norm1 = operations.layer_norm(1024, eps=1e-6, elementwise_affine=False)
        self.act = torch.nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, stride=2, operations=operations)
        self.norm2 = operations.layer_norm(1024, eps=1e-6, elementwise_affine=False)
        self.conv3 = CausalConv1d(1024, 1024, stride=2, operations=operations)
        self.norm3 = operations.layer_norm(1024, eps=1e-6, elementwise_affine=False)
        self.out_proj = operations.linear(1024, hidden_size)
        self.padding_tokens = torch.nn.Parameter(torch.empty(1, 1, 1, hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames = value.shape[:2]
        value = self.conv1_local(value.transpose(1, 2))
        value = value.view(batch, self.num_heads, 1024, frames).permute(0, 1, 3, 2)
        value = self.act(self.norm1(value.flatten(0, 1))).transpose(1, 2)
        value = self.conv2(value).transpose(1, 2)
        value = self.act(self.norm2(value)).transpose(1, 2)
        value = self.conv3(value).transpose(1, 2)
        value = self.out_proj(self.act(self.norm3(value)))
        value = value.unflatten(0, (batch, self.num_heads)).permute(0, 2, 1, 3)
        with self._materialized(value, "padding_tokens") as state:
            padding = state["padding_tokens"].expand(batch, value.shape[1], -1, -1)
            return torch.cat((value, padding), dim=2)


class FaceBlock(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.linear1_kv = operations.linear(hidden_size, hidden_size * 2)
        self.linear1_q = operations.linear(hidden_size, hidden_size)
        self.linear2 = operations.linear(hidden_size, hidden_size)
        self.q_norm = operations.rms_norm(self.head_dim, eps=1e-6)
        self.k_norm = operations.rms_norm(self.head_dim, eps=1e-6)
        self.pre_norm_feat = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_norm_motion = operations.layer_norm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, value: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        batch, frames, motion_rows, _ = motion.shape
        if value.shape[0] != batch or value.shape[1] % frames:
            raise ValueError("Wan Animate motion rows must divide the patch-token sequence")
        spatial_rows = value.shape[1] // frames
        kv = self.linear1_kv(self.pre_norm_motion(motion))
        kv = kv.view(batch, frames, motion_rows, 2, self.num_heads, self.head_dim)
        key, val = kv.unbind(3)
        query = self.linear1_q(self.pre_norm_feat(value)).view(
            batch, frames, spatial_rows, self.num_heads, self.head_dim
        )
        query = self.q_norm(query).to(val)
        key = self.k_norm(key).to(val)
        query = query.flatten(0, 1).transpose(1, 2)
        key = key.flatten(0, 1).transpose(1, 2)
        val = val.flatten(0, 1).transpose(1, 2)
        attended = self._attention_kernel(query, key, val)
        attended = attended.transpose(1, 2).reshape(batch, frames * spatial_rows, -1)
        return self.linear2(attended)


class FaceAdapter(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.fuser_blocks = torch.nn.ModuleList(
            FaceBlock(
                hidden_size,
                num_heads,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(num_layers)
        )


def _upfirdn2d(value: torch.Tensor, kernel: torch.Tensor, pad: tuple[int, int]) -> torch.Tensor:
    _, channels, height, width = value.shape
    pad_start, pad_end = pad
    value = F.pad(value, (pad_start, pad_end, pad_start, pad_end))
    value = value.reshape(-1, 1, height + pad_start + pad_end, width + pad_start + pad_end)
    weight = torch.flip(kernel, (0, 1)).view(1, 1, *kernel.shape)
    value = F.conv2d(value, weight)
    value = value.reshape(
        -1,
        channels,
        height + pad_start + pad_end - kernel.shape[0] + 1,
        width + pad_start + pad_end - kernel.shape[1] + 1,
    )
    return value


class Blur(_DirectState, torch.nn.Module):
    def __init__(self, pad: tuple[int, int]) -> None:
        super().__init__()
        self.register_buffer("kernel", torch.empty(4, 4))
        self.pad = pad

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with self._materialized(value, "kernel") as state:
            return _upfirdn2d(value, state["kernel"], self.pad)


class FusedLeakyReLU(_DirectState, torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with self._materialized(value, "bias") as state:
            return F.leaky_relu(value + state["bias"], 0.2) * math.sqrt(2)


class EqualConv2d(_DirectState, torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter | None = (
            torch.nn.Parameter(torch.empty(output_channels)) if bias else None
        )
        self.scale = 1 / math.sqrt(input_channels * kernel_size**2)
        self.stride = stride
        self.padding = padding

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        names = ("weight",) if self.bias is None else ("weight", "bias")
        with self._materialized(value, *names) as state:
            return F.conv2d(
                value,
                state["weight"] * self.scale,
                state.get("bias"),
                stride=self.stride,
                padding=self.padding,
            )


class EqualLinear(_DirectState, torch.nn.Module):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_features, input_features))
        self.bias = torch.nn.Parameter(torch.empty(output_features))
        self.scale = 1 / math.sqrt(input_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with self._materialized(value, "weight", "bias") as state:
            return F.linear(value, state["weight"] * self.scale, state["bias"])


class ConvLayer(torch.nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        downsample: bool = False,
        bias: bool = True,
        activate: bool = True,
    ) -> None:
        layers: list[torch.nn.Module] = []
        if downsample:
            factor = 2
            padding = (4 - factor) + (kernel_size - 1)
            layers.append(Blur(((padding + 1) // 2, padding // 2)))
            stride, conv_padding = 2, 0
        else:
            stride, conv_padding = 1, kernel_size // 2
        layers.append(
            EqualConv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=conv_padding,
                bias=bias and not activate,
            )
        )
        if activate:
            layers.append(FusedLeakyReLU(output_channels))
        super().__init__(*layers)


class ResBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(input_channels, input_channels, 3)
        self.conv2 = ConvLayer(input_channels, output_channels, 3, downsample=True)
        self.skip = ConvLayer(
            input_channels,
            output_channels,
            1,
            downsample=True,
            activate=False,
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (self.conv2(self.conv1(value)) + self.skip(value)) / math.sqrt(2)


class EncoderApp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 512),
            (512, 512),
            (512, 512),
        )
        self.convs = torch.nn.ModuleList(
            [ConvLayer(3, 32, 1)]
            + [
                ResBlock(input_channels, output_channels)
                for input_channels, output_channels in channels
            ]
            + [EqualConv2d(512, 512, 4, bias=False)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for convolution in self.convs:
            value = convolution(value)
        return value.squeeze(-1).squeeze(-1)


class MotionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net_app = EncoderApp()
        self.fc = torch.nn.Sequential(
            *(EqualLinear(512, 512) for _ in range(4)),
            EqualLinear(512, 20),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net_app(value))


class Direction(_DirectState, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(512, 20))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with self._materialized(value, "weight") as state:
            stabilized = state["weight"] + 1e-8 * torch.eye(
                512, 20, device=value.device, dtype=value.dtype
            )
            direction, _ = torch.linalg.qr(stabilized.float())
            return torch.sum(value.unsqueeze(-1) * direction.T.to(value.dtype), dim=1)


class Synthesis(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direction = Direction()


class AnimateMotionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = MotionEncoder()
        self.dec = Synthesis()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dec.direction(self.enc(value))


__all__ = ["AnimateMotionEncoder", "FaceAdapter", "FaceEncoder"]
