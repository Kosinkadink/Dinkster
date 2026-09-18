"""Native decode-only MiniMax Music 3 DAV audio codec."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from dinkster_inference import (
    MINIMAX_MUSIC3_DAV_CONFIG,
    MINIMAX_MUSIC3_DAV_DESCRIPTOR,
    MiniMaxMusic3DavConfig,
)

from .codecs import CodecPlugin
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight


class _DecodeOnlyEncoder:
    def encode(self, content: torch.Tensor) -> torch.Tensor:
        del content
        raise ValueError("MiniMax Music 3 DAV is decode-only")


class MiniMaxMusic3Snake(ResidencyRouted, torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            alpha = cast_weight(self.alpha, device=hidden.device, dtype=hidden.dtype)
            return hidden + torch.sin(alpha * hidden).square() * (alpha + 1e-9).reciprocal()
        with binding.lease() as lease:
            alpha = lease.get("alpha", dtype=hidden.dtype)
            return hidden + torch.sin(alpha * hidden).square() * (alpha + 1e-9).reciprocal()


class _WeightNormalizedConv1d(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty(out_channels, 1, 1))
        self.weight_v = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.dilation = dilation
        self.padding = padding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            weight_g = cast_weight(self.weight_g, device=hidden.device, dtype=hidden.dtype)
            weight_v = cast_weight(self.weight_v, device=hidden.device, dtype=hidden.dtype)
            bias = cast_weight(self.bias, device=hidden.device, dtype=hidden.dtype)
            weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                weight_v, weight_g, 0
            )
            return F.conv1d(hidden, weight, bias, padding=self.padding, dilation=self.dilation)
        with binding.lease() as lease:
            weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                lease.get("weight_v", dtype=hidden.dtype),
                lease.get("weight_g", dtype=hidden.dtype),
                0,
            )
            return F.conv1d(
                hidden,
                weight,
                lease.get("bias", dtype=hidden.dtype),
                padding=self.padding,
                dilation=self.dilation,
            )


class _WeightNormalizedConvTranspose1d(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty(in_channels, 1, 1))
        self.weight_v = torch.nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            weight_g = cast_weight(self.weight_g, device=hidden.device, dtype=hidden.dtype)
            weight_v = cast_weight(self.weight_v, device=hidden.device, dtype=hidden.dtype)
            bias = cast_weight(self.bias, device=hidden.device, dtype=hidden.dtype)
            weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                weight_v, weight_g, 0
            )
            return F.conv_transpose1d(
                hidden, weight, bias, stride=self.stride, padding=self.padding
            )
        with binding.lease() as lease:
            weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                lease.get("weight_v", dtype=hidden.dtype),
                lease.get("weight_g", dtype=hidden.dtype),
                0,
            )
            return F.conv_transpose1d(
                hidden,
                weight,
                lease.get("bias", dtype=hidden.dtype),
                stride=self.stride,
                padding=self.padding,
            )


class MiniMaxMusic3DavResidual(torch.nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            MiniMaxMusic3Snake(channels),
            _WeightNormalizedConv1d(
                channels,
                channels,
                7,
                dilation=dilation,
                padding=3 * dilation,
            ),
            MiniMaxMusic3Snake(channels),
            _WeightNormalizedConv1d(channels, channels, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden)
        if residual.shape[-1] != hidden.shape[-1]:
            padding = (hidden.shape[-1] - residual.shape[-1]) // 2
            hidden = hidden[..., padding : hidden.shape[-1] - padding]
        return hidden + residual


class MiniMaxMusic3DavDecoderBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            MiniMaxMusic3Snake(input_channels),
            _WeightNormalizedConvTranspose1d(
                input_channels,
                output_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            MiniMaxMusic3DavResidual(output_channels, 1),
            MiniMaxMusic3DavResidual(output_channels, 3),
            MiniMaxMusic3DavResidual(output_channels, 9),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.block(hidden)


class MiniMaxMusic3DavDecoder(torch.nn.Module):
    def __init__(self, config: MiniMaxMusic3DavConfig) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [
            _WeightNormalizedConv1d(
                config.hidden_channels,
                config.decoder_channels,
                7,
                padding=3,
            )
        ]
        output_channels = config.decoder_channels
        for index, stride in enumerate(config.strides):
            input_channels = config.decoder_channels // (2**index)
            output_channels = config.decoder_channels // (2 ** (index + 1))
            layers.append(MiniMaxMusic3DavDecoderBlock(input_channels, output_channels, stride))
        layers.extend(
            (
                MiniMaxMusic3Snake(output_channels),
                _WeightNormalizedConv1d(output_channels, 1, 7, padding=3),
                torch.nn.Tanh(),
            )
        )
        self.model = torch.nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model(hidden)


class MiniMaxMusic3Dav(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3DavConfig = MINIMAX_MUSIC3_DAV_CONFIG,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        self.dec_in_proj = operations.conv1d(config.latent_channels // 2, config.hidden_channels, 1)
        self.decoder = MiniMaxMusic3DavDecoder(config)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _channels, frames = latent.shape
        folded = latent.reshape(batch * 2, self.config.latent_channels // 2, frames)
        waveform = self.decoder(self.dec_in_proj(folded))
        return waveform.reshape(batch, 2, -1)

    forward = decode


def minimax_music3_dav_codec(module: MiniMaxMusic3Dav) -> CodecPlugin:
    if type(module) is not MiniMaxMusic3Dav:
        raise TypeError("MiniMax Music 3 DAV codec requires an exact native module")
    return CodecPlugin(
        descriptor=MINIMAX_MUSIC3_DAV_DESCRIPTOR,
        encoder=_DecodeOnlyEncoder(),
        decoder=module,
        compute_dtype=torch.float32,
    )


__all__ = [
    "MiniMaxMusic3Dav",
    "minimax_music3_dav_codec",
]
