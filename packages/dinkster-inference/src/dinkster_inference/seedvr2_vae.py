"""Exact torch-free SeedVR2 causal video VAE description."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT16, FLOAT32
from .latents import LatentDescriptor
from .weights import TensorGeometry


class SeedVR2VAEHeaderError(ValueError):
    """A header is not the supported SeedVR2 VAE layout."""


@dataclass(frozen=True, slots=True)
class SeedVR2VAEConfig:
    channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    latent_channels: int = 16
    image_channels: int = 3
    temporal_scale: int = 4
    spatial_scale: int = 8
    scaling_factor: float = 0.9152
    shifting_factor: float = 0.0

    def __post_init__(self) -> None:
        if tuple(getattr(self, field) for field in self.__dataclass_fields__) != (
            (128, 256, 512, 512),
            2,
            16,
            3,
            4,
            8,
            0.9152,
            0.0,
        ):
            raise ValueError("SeedVR2VAEConfig only represents the published VAE")


SEEDVR2_VAE_CONFIG = SeedVR2VAEConfig()
SEEDVR2_VAE_DETECTOR_KEY = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"
SEEDVR2_CODEC = CodecDescriptor(
    id="dinkster.seedvr2_vae",
    display_name="SeedVR2 causal video VAE",
    kind="video",
    latent=LatentDescriptor(
        channels=16,
        dimensions=3,
        spatial_downscale=8,
        temporal_downscale=4,
        temporal_causal=True,
    ),
    supported_dtypes=frozenset((BFLOAT16, FLOAT16, FLOAT32)),
    content_channels=3,
    supports_tiling=True,
    tiling=CodecTiling(
        decode_tile=(9999, 32, 32),
        decode_overlap=(0, 8, 8),
        encode_tile=(9999, 512, 512),
        encode_overlap=(0, 64, 64),
    ),
)


def _affine(out: dict[str, tuple[int, ...]], key: str, shape: tuple[int, ...]) -> None:
    out[f"{key}.weight"] = shape
    out[f"{key}.bias"] = (shape[0],)


def _resnet(out: dict[str, tuple[int, ...]], root: str, inp: int, width: int) -> None:
    for norm, channels in (("norm1", inp), ("norm2", width)):
        _affine(out, f"{root}.{norm}", (channels,))
    _affine(out, f"{root}.conv1", (width, inp, 3, 3, 3))
    _affine(out, f"{root}.conv2", (width, width, 3, 3, 3))
    if inp != width:
        _affine(out, f"{root}.conv_shortcut", (width, inp, 1, 1, 1))


def _mid(out: dict[str, tuple[int, ...]], root: str, width: int) -> None:
    _resnet(out, f"{root}.resnets.0", width, width)
    _affine(out, f"{root}.attentions.0.group_norm", (width,))
    for key in ("to_q", "to_k", "to_v", "to_out.0"):
        _affine(out, f"{root}.attentions.0.{key}", (width, width))
    _resnet(out, f"{root}.resnets.1", width, width)


def seedvr2_vae_layout() -> Mapping[str, tuple[int, ...]]:
    config = SEEDVR2_VAE_CONFIG
    out: dict[str, tuple[int, ...]] = {}
    _affine(out, "encoder.conv_in", (128, 3, 3, 3, 3))
    inp = 128
    for level, width in enumerate(config.channels):
        for block in range(2):
            _resnet(out, f"encoder.down_blocks.{level}.resnets.{block}", inp, width)
            inp = width
        if level < 3:
            temporal = 3 if level >= 1 else 1
            _affine(
                out,
                f"encoder.down_blocks.{level}.downsamplers.0.conv",
                (width, width, temporal, 3, 3),
            )
    _mid(out, "encoder.mid_block", 512)
    _affine(out, "encoder.conv_norm_out", (512,))
    _affine(out, "encoder.conv_out", (32, 512, 3, 3, 3))
    _affine(out, "decoder.conv_in", (512, 16, 3, 3, 3))
    _mid(out, "decoder.mid_block", 512)
    inp = 512
    for level, width in enumerate(reversed(config.channels)):
        for block in range(3):
            _resnet(out, f"decoder.up_blocks.{level}.resnets.{block}", inp, width)
            inp = width
        if level < 3:
            ratio = 8 if level < 2 else 4
            _affine(
                out,
                f"decoder.up_blocks.{level}.upsamplers.0.upscale_conv",
                (width * ratio, width, 1, 1, 1),
            )
            _affine(out, f"decoder.up_blocks.{level}.upsamplers.0.conv", (width, width, 3, 3, 3))
    _affine(out, "decoder.conv_norm_out", (128,))
    _affine(out, "decoder.conv_out", (3, 128, 3, 3, 3))
    if len(out) != 250:
        raise AssertionError(f"SeedVR2 VAE layout has {len(out)} keys, expected 250")
    return MappingProxyType(out)


def detect_seedvr2_vae_config(geometries: Mapping[str, TensorGeometry]) -> SeedVR2VAEConfig:
    if SEEDVR2_VAE_DETECTOR_KEY not in geometries:
        raise SeedVR2VAEHeaderError(f"missing detector key {SEEDVR2_VAE_DETECTOR_KEY}")
    expected = seedvr2_vae_layout()
    if set(geometries) != set(expected):
        raise SeedVR2VAEHeaderError("SeedVR2 VAE key set does not match the published layout")
    for key, shape in expected.items():
        if geometries[key].shape != shape:
            raise SeedVR2VAEHeaderError(f"geometry mismatch for {key}")
    return SEEDVR2_VAE_CONFIG


__all__ = [
    "SEEDVR2_CODEC",
    "SEEDVR2_VAE_CONFIG",
    "SEEDVR2_VAE_DETECTOR_KEY",
    "SeedVR2VAEConfig",
    "SeedVR2VAEHeaderError",
    "detect_seedvr2_vae_config",
    "seedvr2_vae_layout",
]
