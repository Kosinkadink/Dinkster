"""Torch-free Wan 2.1 causal video VAE configuration and state layout.

The architecture and state names follow ``comfy/ldm/wan/vae.py`` at
ComfyUI commit ``b78cec87``. The layout is suitable for strict header-only
assembly planning without constructing a torch module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

__all__ = [
    "WAN21_FLOW_RVS_VAE_CONFIG",
    "WAN21_VAE_CONFIG",
    "Wan21VAEConfig",
    "wan21_vae_layout",
]


@dataclass(frozen=True, slots=True)
class Wan21VAEConfig:
    dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    image_channels: int = 3
    conv_out_channels: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.z_dim <= 0 or self.num_res_blocks <= 0:
            raise ValueError("dim, z_dim, and num_res_blocks must be positive")
        if len(self.dim_mult) < 2 or any(multiplier <= 0 for multiplier in self.dim_mult):
            raise ValueError("dim_mult must contain at least two positive entries")
        if len(self.temporal_downsample) != len(self.dim_mult) - 1:
            raise ValueError("temporal_downsample must have one entry per resample level")
        if sum(self.temporal_downsample) != 2:
            raise ValueError("Wan 2.1 requires exactly two temporal downsample levels")
        if (self.image_channels, self.conv_out_channels) not in ((3, 3), (3, 1)):
            raise ValueError("Wan 2.1 content channels must be RGB-to-RGB or RGB-to-mask")
        if not 0.0 <= self.dropout < 1.0 or not math.isfinite(self.dropout):
            raise ValueError("dropout must be finite in [0,1)")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.attn_scales):
            raise ValueError("attention scales must be finite and positive")

    @property
    def spatial_ratio(self) -> int:
        return 2 ** (len(self.dim_mult) - 1)


WAN21_VAE_CONFIG = Wan21VAEConfig()
WAN21_FLOW_RVS_VAE_CONFIG = Wan21VAEConfig(conv_out_channels=1)


def _affine(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    weight: tuple[int, ...],
) -> None:
    layout[f"{prefix}.weight"] = weight
    layout[f"{prefix}.bias"] = (weight[0],)


def _causal_conv(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int, int],
) -> None:
    _affine(layout, prefix, (out_channels, in_channels, *kernel))


def _conv2d(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> None:
    _affine(layout, prefix, (out_channels, in_channels, 3, 3))


def _residual(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> None:
    layout[f"{prefix}.residual.0.gamma"] = (in_channels, 1, 1, 1)
    _causal_conv(layout, f"{prefix}.residual.2", in_channels, out_channels, (3, 3, 3))
    layout[f"{prefix}.residual.3.gamma"] = (out_channels, 1, 1, 1)
    _causal_conv(layout, f"{prefix}.residual.6", out_channels, out_channels, (3, 3, 3))
    if in_channels != out_channels:
        _causal_conv(layout, f"{prefix}.shortcut", in_channels, out_channels, (1, 1, 1))


def _attention(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
) -> None:
    layout[f"{prefix}.norm.gamma"] = (channels, 1, 1)
    _affine(layout, f"{prefix}.to_qkv", (channels * 3, channels, 1, 1))
    _affine(layout, f"{prefix}.proj", (channels, channels, 1, 1))


def _downsample(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
    *,
    temporal: bool,
) -> None:
    _conv2d(layout, f"{prefix}.resample.1", channels, channels)
    if temporal:
        _causal_conv(layout, f"{prefix}.time_conv", channels, channels, (3, 1, 1))


def _upsample(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
    *,
    temporal: bool,
) -> None:
    _conv2d(layout, f"{prefix}.resample.1", channels, channels // 2)
    if temporal:
        _causal_conv(layout, f"{prefix}.time_conv", channels, channels * 2, (3, 1, 1))


def _middle(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
) -> None:
    _residual(layout, f"{prefix}.0", channels, channels)
    _attention(layout, f"{prefix}.1", channels)
    _residual(layout, f"{prefix}.2", channels, channels)


def wan21_vae_layout(
    config: Wan21VAEConfig = WAN21_VAE_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """Return the exact model-key to tensor-shape contract."""

    if not isinstance(cast("object", config), Wan21VAEConfig):
        raise TypeError("config must be Wan21VAEConfig")
    layout: dict[str, tuple[int, ...]] = {}

    encoder_dims = [config.dim * multiplier for multiplier in (1, *config.dim_mult)]
    _causal_conv(
        layout,
        "encoder.conv1",
        config.image_channels,
        encoder_dims[0],
        (3, 3, 3),
    )
    scale = 1.0
    layer_index = 0
    encoder_out = encoder_dims[0]
    for level, (in_channels, encoder_out) in enumerate(
        zip(encoder_dims[:-1], encoder_dims[1:], strict=True)
    ):
        for _ in range(config.num_res_blocks):
            prefix = f"encoder.downsamples.{layer_index}"
            _residual(layout, prefix, in_channels, encoder_out)
            layer_index += 1
            if scale in config.attn_scales:
                _attention(layout, f"encoder.downsamples.{layer_index}", encoder_out)
                layer_index += 1
            in_channels = encoder_out
        if level != len(config.dim_mult) - 1:
            _downsample(
                layout,
                f"encoder.downsamples.{layer_index}",
                encoder_out,
                temporal=config.temporal_downsample[level],
            )
            layer_index += 1
            scale /= 2.0
    _middle(layout, "encoder.middle", encoder_out)
    layout["encoder.head.0.gamma"] = (encoder_out, 1, 1, 1)
    _causal_conv(
        layout,
        "encoder.head.2",
        encoder_out,
        config.z_dim * 2,
        (3, 3, 3),
    )
    _causal_conv(layout, "conv1", config.z_dim * 2, config.z_dim * 2, (1, 1, 1))
    _causal_conv(layout, "conv2", config.z_dim, config.z_dim, (1, 1, 1))

    decoder_dims = [
        config.dim * multiplier for multiplier in (config.dim_mult[-1], *reversed(config.dim_mult))
    ]
    _causal_conv(layout, "decoder.conv1", config.z_dim, decoder_dims[0], (3, 3, 3))
    _middle(layout, "decoder.middle", decoder_dims[0])
    scale = 1.0 / 2 ** (len(config.dim_mult) - 2)
    layer_index = 0
    decoder_out = decoder_dims[-1]
    reversed_temporal = tuple(reversed(config.temporal_downsample))
    for level, (in_channels, decoder_out) in enumerate(
        zip(decoder_dims[:-1], decoder_dims[1:], strict=True)
    ):
        if level in (1, 2, 3):
            in_channels //= 2
        for _ in range(config.num_res_blocks + 1):
            prefix = f"decoder.upsamples.{layer_index}"
            _residual(layout, prefix, in_channels, decoder_out)
            layer_index += 1
            if scale in config.attn_scales:
                _attention(layout, f"decoder.upsamples.{layer_index}", decoder_out)
                layer_index += 1
            in_channels = decoder_out
        if level != len(config.dim_mult) - 1:
            _upsample(
                layout,
                f"decoder.upsamples.{layer_index}",
                decoder_out,
                temporal=reversed_temporal[level],
            )
            layer_index += 1
            scale *= 2.0
    layout["decoder.head.0.gamma"] = (decoder_out, 1, 1, 1)
    _causal_conv(
        layout,
        "decoder.head.2",
        decoder_out,
        config.conv_out_channels,
        (3, 3, 3),
    )
    return layout
