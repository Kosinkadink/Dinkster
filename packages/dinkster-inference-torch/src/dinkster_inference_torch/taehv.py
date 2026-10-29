"""Native TAEHV decoder matching comfy/taehv/taehv.py @ 783545f6.

Only the decoder half of the Wan variants is implemented (TGrow strides
1, 2, 2; three x2 spatial upsamples; ReLU activations), in the reference's
parallel mode: every MemBlock sees the previous timestep's input at that
block (zeros for the first), so a decoded clip is a pure function of the
latent window it was given.
"""

from __future__ import annotations

import torch
from dinkster_inference.taehv import TAEHVConfig

from .operations import INITLESS, Operations
from .taesd import Clamp


class TAEHVMemBlock(torch.nn.Module):
    """Reference MemBlock: the input concatenated channel-wise with the
    previous timestep's input, through three 3x3 convs with a residual
    skip (identity here - all decoder MemBlocks keep their width)."""

    def __init__(self, channels: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            operations.conv2d(channels * 2, channels, 3, padding=1),
            torch.nn.ReLU(),
            operations.conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(),
            operations.conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.conv(torch.cat([x, past], 1)) + x)


class TAEHVTGrow(torch.nn.Module):
    """Reference TGrow: a 1x1 bias-free conv widening each timestep to
    ``stride`` consecutive output timesteps."""

    def __init__(self, channels: int, stride: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.stride = stride
        self.conv = operations.conv2d(channels, channels * stride, 1)
        self.conv.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, channels, height, width = x.shape
        return self.conv(x).reshape(-1, channels, height, width)


def _conv(inc: int, out: int, operations: Operations, *, bias: bool = True) -> torch.nn.Conv2d:
    layer = operations.conv2d(inc, out, 3, padding=1)
    if not bias:
        layer.bias = None
    return layer


class TAEHVDecoder(torch.nn.Sequential):
    """Positional-key TAEHV decoder for the Wan light preview TAEs."""

    def __init__(self, config: TAEHVConfig, *, operations: Operations = INITLESS) -> None:
        widths = (256, 128, 64)
        strides = (1, 2, 2)
        layers: list[torch.nn.Module] = [
            Clamp(),
            _conv(config.latent_channels, widths[0], operations),
            torch.nn.ReLU(),
        ]
        for width, next_width, stride in zip(widths, (*widths[1:], 64), strides, strict=True):
            layers += [
                *(TAEHVMemBlock(width, operations=operations) for _ in range(3)),
                torch.nn.Upsample(scale_factor=2),
                TAEHVTGrow(width, stride, operations=operations),
                _conv(width, next_width, operations, bias=False),
            ]
        layers += [
            torch.nn.ReLU(),
            _conv(64, 3 * config.patch_size**2, operations),
        ]
        super().__init__(*layers)
        self.config = config

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent clip [B, C, T, H, W] to content frames
        [B, 4T - 3, 3, H * upscale, W * upscale] in the light decoders'
        [0, 1] value convention (the reference's latent_format=None,
        do_scale=False path)."""
        config = self.config
        if latent.ndim != 5 or latent.shape[1] != config.latent_channels:
            raise ValueError(
                f"TAEHV decode requires [B,{config.latent_channels},T,H,W] latent,"
                f" got {tuple(latent.shape)}"
            )
        batch, channels, frames, height, width = latent.shape
        x = latent.movedim(1, 2).reshape(batch * frames, channels, height, width)
        current = frames
        for module in self:
            if isinstance(module, TAEHVMemBlock):
                grouped = x.reshape(batch, current, *x.shape[1:])
                past = torch.nn.functional.pad(grouped, (0, 0, 0, 0, 0, 0, 1, 0))
                x = module(x, past[:, :current].reshape(x.shape))
            elif isinstance(module, TAEHVTGrow):
                x = module(x)
                current *= module.stride
            else:
                x = module(x)
        if config.patch_size > 1:
            x = torch.nn.functional.pixel_shuffle(x, config.patch_size)
        x = x.reshape(batch, current, *x.shape[1:])
        return x[:, config.frames_to_trim :]


__all__ = [
    "TAEHVDecoder",
    "TAEHVMemBlock",
    "TAEHVTGrow",
]
