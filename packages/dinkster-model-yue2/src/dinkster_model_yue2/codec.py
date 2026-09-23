"""YuE2 48 kHz AudioOobleck codec."""

from __future__ import annotations

import math
from typing import Any

import torch
from dinkster_inference_torch.operations import INITLESS, Operations


def _weight_norm(module: torch.nn.Module) -> torch.nn.Module:
    return torch.nn.utils.parametrizations.weight_norm(module)


class SnakeBeta(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(channels))
        self.beta = torch.nn.Parameter(torch.empty(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().view(1, -1, 1).to(value)
        beta = self.beta.exp().view(1, -1, 1).to(value)
        return value + torch.sin(value * alpha).square() / (beta + 1e-9)


def _conv1d(operations: Operations, *args: Any, **kwargs: Any) -> torch.nn.Module:
    return _weight_norm(operations.conv1d(*args, **kwargs))


def _conv_transpose1d(operations: Operations, *args: Any, **kwargs: Any) -> torch.nn.Module:
    return _weight_norm(operations.conv_transpose1d(*args, **kwargs))


class ResidualUnit(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        padding = dilation * 3
        self.layers = torch.nn.Sequential(
            SnakeBeta(channels),
            _conv1d(
                operations,
                channels,
                channels,
                7,
                dilation=dilation,
                padding=padding,
            ),
            SnakeBeta(channels),
            _conv1d(operations, channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            *(ResidualUnit(in_channels, rate, operations=operations) for rate in (1, 3, 9)),
            SnakeBeta(in_channels),
            _conv1d(
                operations,
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            SnakeBeta(in_channels),
            _conv_transpose1d(
                operations,
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            *(ResidualUnit(out_channels, rate, operations=operations) for rate in (1, 3, 9)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class OobleckEncoder(torch.nn.Module):
    def __init__(self, *, operations: Operations) -> None:
        super().__init__()
        channels = 64
        multipliers = (1, 1, 2, 4, 8, 16, 32)
        strides = (2, 2, 4, 4, 5, 6)
        layers: list[torch.nn.Module] = [_conv1d(operations, 2, channels, 7, padding=3)]
        for index, stride in enumerate(strides):
            layers.append(
                EncoderBlock(
                    channels * multipliers[index],
                    channels * multipliers[index + 1],
                    stride,
                    operations=operations,
                )
            )
        layers.extend(
            (
                SnakeBeta(channels * multipliers[-1]),
                _conv1d(operations, channels * multipliers[-1], 128, 3, padding=1),
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class OobleckDecoder(torch.nn.Module):
    def __init__(self, *, operations: Operations) -> None:
        super().__init__()
        channels = 64
        multipliers = (1, 1, 2, 4, 8, 16, 32)
        strides = (2, 2, 4, 4, 5, 6)
        layers: list[torch.nn.Module] = [
            _conv1d(operations, 64, channels * multipliers[-1], 7, padding=3)
        ]
        for index in range(len(strides), 0, -1):
            layers.append(
                DecoderBlock(
                    channels * multipliers[index],
                    channels * multipliers[index - 1],
                    strides[index - 1],
                    operations=operations,
                )
            )
        layers.extend(
            (
                SnakeBeta(channels),
                _conv1d(operations, channels, 2, 7, padding=3, bias=False),
                torch.nn.Identity(),
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class AudioOobleckVAE(torch.nn.Module):
    sample_rate = 48_000

    def __init__(self, _config: object = None, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.encoder = OobleckEncoder(operations=operations)
        self.decoder = OobleckDecoder(operations=operations)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        mean, _scale = self.encoder(value).chunk(2, dim=1)
        return mean

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


__all__ = ["AudioOobleckVAE"]
