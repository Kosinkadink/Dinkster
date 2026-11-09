"""ESRGAN-family super-resolution architectures.

Both modules reproduce spandrel 0.4.2 (commit 724cca38) exactly, including
its state-dict key layout, so checkpoints load with ``strict=True``:

- ``RRDBNet`` is the old-arch ESRGAN network (flattened ``model.N`` keys).
  New-arch checkpoints are converted to this layout by the loader.
- ``SRVGGNetCompact`` is the RealESRGAN Compact network (``body.N`` keys).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _conv3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def _lrelu() -> nn.LeakyReLU:
    return nn.LeakyReLU(negative_slope=0.2, inplace=True)


class ResidualDenseBlock(nn.Module):
    """Five-conv residual dense block; ``plus`` adds the ESRGAN+ 1x1 paths."""

    def __init__(self, filters: int, growth: int, *, plus: bool) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(filters, growth, kernel_size=1, bias=False) if plus else None
        self.conv1 = nn.Sequential(_conv3(filters, growth), _lrelu())
        self.conv2 = nn.Sequential(_conv3(filters + growth, growth), _lrelu())
        self.conv3 = nn.Sequential(_conv3(filters + 2 * growth, growth), _lrelu())
        self.conv4 = nn.Sequential(_conv3(filters + 3 * growth, growth), _lrelu())
        self.conv5 = nn.Sequential(_conv3(filters + 4 * growth, filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1 is not None:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1 is not None:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, filters: int, growth: int, *, plus: bool) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock(filters, growth, plus=plus)
        self.RDB2 = ResidualDenseBlock(filters, growth, plus=plus)
        self.RDB3 = ResidualDenseBlock(filters, growth, plus=plus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ShortcutBlock(nn.Module):
    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_width, 0, pad_height), mode="reflect")


class RRDBNet(nn.Module):
    """Old-arch ESRGAN with power-of-two ``scale``. ``shuffle_factor`` is the
    RealESRGAN pixel-unshuffle wrapper: the network consumes unshuffled
    ``in_channels`` and its effective scale is ``scale // shuffle_factor``."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        filters: int,
        blocks: int,
        scale: int,
        growth: int = 32,
        plus: bool = False,
        shuffle_factor: int | None = None,
    ) -> None:
        super().__init__()
        if scale < 1 or scale & (scale - 1):
            raise ValueError(f"scale must be a power of two, got {scale}")
        self.scale = scale
        self.shuffle_factor = shuffle_factor
        trunk = nn.Sequential(
            *(RRDB(filters, growth, plus=plus) for _ in range(blocks)),
            _conv3(filters, filters),
        )
        layers: list[nn.Module] = [_conv3(in_channels, filters), ShortcutBlock(trunk)]
        for _ in range(int(math.log2(scale))):
            layers.extend(
                (
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    _conv3(filters, filters),
                    _lrelu(),
                )
            )
        layers.extend((_conv3(filters, filters), _lrelu(), _conv3(filters, out_channels)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shuffle_factor:
            true_scale = self.scale // self.shuffle_factor
            height, width = x.shape[-2:]
            x = _pad_to_multiple(x, self.shuffle_factor)
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
            x = self.model(x)
            return x[:, :, : height * true_scale, : width * true_scale]
        return self.model(x)


class SRVGGNetCompact(nn.Module):
    """RealESRGAN Compact: PReLU conv body, pixel shuffle, nearest-base residual."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        filters: int,
        convs: int,
        scale: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        body: list[nn.Module] = [_conv3(in_channels, filters), nn.PReLU(num_parameters=filters)]
        for _ in range(convs):
            body.extend((_conv3(filters, filters), nn.PReLU(num_parameters=filters)))
        body.append(_conv3(filters, out_channels * scale * scale))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        return out + base


__all__ = ["RRDB", "RRDBNet", "ResidualDenseBlock", "SRVGGNetCompact", "ShortcutBlock"]
