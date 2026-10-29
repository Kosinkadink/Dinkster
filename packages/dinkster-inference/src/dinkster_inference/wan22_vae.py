"""Torch-free Wan 2.2 causal video VAE configuration and state layout.

The architecture and state names follow ``comfy/ldm/wan/vae2_2.py`` at
ComfyUI commit ``b78cec87``. Its production topology follows the loader config
in ``comfy/sd.py`` at that commit. The official checkpoint has 196 float16
tensors, 48 latent channels, 16x spatial downscale, and 4x causal temporal
downscale.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .weights import TensorGeometry

__all__ = [
    "WAN22_VAE_CONFIG",
    "Wan22VAEConfig",
    "Wan22VAEHeaderError",
    "validate_wan22_vae_header",
    "wan22_vae_layout",
]


class Wan22VAEHeaderError(ValueError):
    """A checkpoint header is not the official Wan 2.2 VAE layout."""


@dataclass(frozen=True, slots=True)
class Wan22VAEConfig:
    """Wan 2.2 topology with adjustable widths for reduced execution tests."""

    dim: int = 160
    decoder_dim: int = 256
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    image_channels: int = 3
    conv_out_channels: int = 3
    patch_size: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        widths = (self.dim, self.decoder_dim, self.z_dim, self.num_res_blocks)
        if any(type(value) is not int or value <= 0 for value in widths):
            raise ValueError("dim, decoder_dim, z_dim, and num_res_blocks must be positive ints")
        if self.dim_mult != (1, 2, 4, 4):
            raise ValueError("Wan 2.2 requires the (1,2,4,4) channel multiplier topology")
        if self.temporal_downsample != (False, True, True):
            raise ValueError(
                "Wan 2.2 requires temporal downsampling at the final two resample levels"
            )
        if self.attn_scales:
            raise ValueError("Wan 2.2 has no level attention blocks")
        if self.image_channels != 3 or self.conv_out_channels != 3:
            raise ValueError("Wan 2.2 content must have exactly three RGB channels")
        if self.patch_size != 2:
            raise ValueError("Wan 2.2 requires 2x2 spatial patching")
        if not 0.0 <= self.dropout < 1.0 or not math.isfinite(self.dropout):
            raise ValueError("dropout must be finite in [0,1)")

    @property
    def spatial_ratio(self) -> int:
        return self.patch_size * 2 ** (len(self.dim_mult) - 1)

    @property
    def temporal_ratio(self) -> int:
        return 2 ** sum(self.temporal_downsample)


WAN22_VAE_CONFIG = Wan22VAEConfig()


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


def _middle(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
) -> None:
    _residual(layout, f"{prefix}.0", channels, channels)
    _attention(layout, f"{prefix}.1", channels)
    _residual(layout, f"{prefix}.2", channels, channels)


def _down_block(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
    config: Wan22VAEConfig,
    *,
    level: int,
) -> None:
    for block in range(config.num_res_blocks):
        _residual(layout, f"{prefix}.downsamples.{block}", in_channels, out_channels)
        in_channels = out_channels
    if level != len(config.dim_mult) - 1:
        resample = f"{prefix}.downsamples.{config.num_res_blocks}"
        _conv2d(layout, f"{resample}.resample.1", out_channels, out_channels)
        if config.temporal_downsample[level]:
            _causal_conv(layout, f"{resample}.time_conv", out_channels, out_channels, (3, 1, 1))


def _up_block(
    layout: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
    config: Wan22VAEConfig,
    *,
    level: int,
) -> None:
    for block in range(config.num_res_blocks + 1):
        _residual(layout, f"{prefix}.upsamples.{block}", in_channels, out_channels)
        in_channels = out_channels
    if level != len(config.dim_mult) - 1:
        resample = f"{prefix}.upsamples.{config.num_res_blocks + 1}"
        _conv2d(layout, f"{resample}.resample.1", out_channels, out_channels)
        if tuple(reversed(config.temporal_downsample))[level]:
            _causal_conv(layout, f"{resample}.time_conv", out_channels, out_channels * 2, (3, 1, 1))


def wan22_vae_layout(
    config: Wan22VAEConfig = WAN22_VAE_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """Return the exact model-key to tensor-shape contract."""

    if not isinstance(cast("object", config), Wan22VAEConfig):
        raise TypeError("config must be Wan22VAEConfig")
    layout: dict[str, tuple[int, ...]] = {}

    patch_channels = config.image_channels * config.patch_size**2
    encoder_dims = [config.dim * multiplier for multiplier in (1, *config.dim_mult)]
    _causal_conv(layout, "encoder.conv1", patch_channels, encoder_dims[0], (3, 3, 3))
    encoder_out = encoder_dims[0]
    for level, (in_channels, encoder_out) in enumerate(
        zip(encoder_dims[:-1], encoder_dims[1:], strict=True)
    ):
        _down_block(
            layout,
            f"encoder.downsamples.{level}",
            in_channels,
            encoder_out,
            config,
            level=level,
        )
    _middle(layout, "encoder.middle", encoder_out)
    layout["encoder.head.0.gamma"] = (encoder_out, 1, 1, 1)
    _causal_conv(layout, "encoder.head.2", encoder_out, config.z_dim * 2, (3, 3, 3))
    _causal_conv(layout, "conv1", config.z_dim * 2, config.z_dim * 2, (1, 1, 1))
    _causal_conv(layout, "conv2", config.z_dim, config.z_dim, (1, 1, 1))

    decoder_dims = [
        config.decoder_dim * multiplier
        for multiplier in (config.dim_mult[-1], *reversed(config.dim_mult))
    ]
    _causal_conv(layout, "decoder.conv1", config.z_dim, decoder_dims[0], (3, 3, 3))
    _middle(layout, "decoder.middle", decoder_dims[0])
    decoder_out = decoder_dims[0]
    for level, (in_channels, decoder_out) in enumerate(
        zip(decoder_dims[:-1], decoder_dims[1:], strict=True)
    ):
        _up_block(
            layout,
            f"decoder.upsamples.{level}",
            in_channels,
            decoder_out,
            config,
            level=level,
        )
    layout["decoder.head.0.gamma"] = (decoder_out, 1, 1, 1)
    _causal_conv(layout, "decoder.head.2", decoder_out, patch_channels, (3, 3, 3))
    return layout


def validate_wan22_vae_header(
    geometries: Mapping[str, TensorGeometry],
) -> Wan22VAEConfig:
    """Validate the exact official Wan 2.2 VAE geometry and floating storage."""

    expected = wan22_vae_layout()
    actual_keys = set(geometries)
    missing = tuple(sorted(set(expected) - actual_keys))
    extra = tuple(sorted(actual_keys - set(expected)))
    if missing:
        raise Wan22VAEHeaderError("missing required Wan 2.2 VAE keys: " + ", ".join(missing[:3]))
    if extra:
        raise Wan22VAEHeaderError("foreign Wan 2.2 VAE keys: " + ", ".join(extra[:3]))
    for key, shape in expected.items():
        geometry = geometries[key]
        if geometry.shape != shape:
            raise Wan22VAEHeaderError(
                f"geometry mismatch for {key}: got {geometry.shape}, expected {shape}"
            )
        if geometry.dtype.kind != "float":
            raise Wan22VAEHeaderError(
                f"storage dtype mismatch for {key}: got {geometry.dtype.name}, expected floating"
            )
    return WAN22_VAE_CONFIG
