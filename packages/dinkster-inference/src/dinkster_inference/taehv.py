"""Torch-free TAEHV video-TAE decoder contracts.

TAEHV (madebyollin/taehv, vendored as comfy/taehv/taehv.py @ 783545f6) is
the tiny causal video autoencoder family behind Wan's light preview
decoders (``lighttaew2_1``, ``lighttaew2_2``). Only the decoder half is
contracted here - Dinkster uses it for sampling previews, never for output
decoding. The two Wan variants are geometry-distinguishable (16 vs 48
latent channels, 3 vs 12 output channels), so detection is pure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .weights import TensorGeometry


class TAEHVDetectError(ValueError):
    """An artifact does not hold one complete supported TAEHV decoder."""


@dataclass(frozen=True)
class TAEHVConfig:
    """One supported TAEHV decoder geometry.

    ``patch_size`` is the pixel-shuffle factor of the final layer: the
    decoder's three x2 upsamples times ``patch_size`` give the total
    spatial upscale. The temporal upscale is fixed at 4 (TGrow strides
    1, 2, 2) and the first ``temporal_upscale - 1`` decoded frames are
    warm-up output that the reference trims."""

    latent_channels: int
    patch_size: int
    hidden_channels: int = 256

    def __post_init__(self) -> None:
        if (self.latent_channels, self.patch_size) not in ((16, 1), (48, 2)):
            raise ValueError(
                "unsupported TAEHV decoder geometry: "
                f"{self.latent_channels} latent channels, patch {self.patch_size}"
            )
        if self.hidden_channels != 256:
            raise ValueError("only the native 256-wide TAEHV architecture is supported")

    @property
    def spatial_upscale(self) -> int:
        return 8 * self.patch_size

    @property
    def temporal_upscale(self) -> int:
        return 4

    @property
    def frames_to_trim(self) -> int:
        return self.temporal_upscale - 1


def _conv(
    layout: dict[str, tuple[int, ...]],
    key: str,
    out: int,
    inc: int,
    *,
    bias: bool = True,
    kernel: int = 3,
) -> None:
    layout[f"{key}.weight"] = (out, inc, kernel, kernel)
    if bias:
        layout[f"{key}.bias"] = (out,)


def _mem_block(layout: dict[str, tuple[int, ...]], key: str, inc: int, out: int) -> None:
    _conv(layout, f"{key}.conv.0", out, inc * 2)
    _conv(layout, f"{key}.conv.2", out, out)
    _conv(layout, f"{key}.conv.4", out, out)


def taehv_decoder_layout(config: TAEHVConfig) -> dict[str, tuple[int, ...]]:
    """Exact decoder state-dict listing of comfy/taehv/taehv.py @ 783545f6
    for the Wan variants (TGrow strides 1, 2, 2; positional keys with the
    ``decoder.`` prefix stripped)."""
    layout: dict[str, tuple[int, ...]] = {}
    _conv(layout, "1", 256, config.latent_channels)
    for index in (3, 4, 5):
        _mem_block(layout, str(index), 256, 256)
    _conv(layout, "7.conv", 256, 256, bias=False, kernel=1)  # TGrow stride 1
    _conv(layout, "8", 128, 256, bias=False)
    for index in (9, 10, 11):
        _mem_block(layout, str(index), 128, 128)
    _conv(layout, "13.conv", 256, 128, bias=False, kernel=1)  # TGrow stride 2
    _conv(layout, "14", 64, 128, bias=False)
    for index in (15, 16, 17):
        _mem_block(layout, str(index), 64, 64)
    _conv(layout, "19.conv", 128, 64, bias=False, kernel=1)  # TGrow stride 2
    _conv(layout, "20", 64, 64, bias=False)
    _conv(layout, "22", 3 * config.patch_size**2, 64)
    return layout


def detect_taehv_decoder_config(geometries: Mapping[str, TensorGeometry]) -> TAEHVConfig:
    """Strictly detect one complete TAEHV decoder in an artifact header.

    Accepts the ``decoder.``-prefixed half of a combined artifact (any
    ``encoder.`` half is ignored) or a bare decoder-only artifact."""
    stripped = {
        (key[len("decoder.") :] if key.startswith("decoder.") else key): geometry
        for key, geometry in geometries.items()
        if not key.startswith("encoder.")
    }
    for config in (TAEHVConfig(16, 1), TAEHVConfig(48, 2)):
        expected = taehv_decoder_layout(config)
        if set(stripped) == set(expected) and all(
            stripped[key].shape == shape for key, shape in expected.items()
        ):
            return config
    raise TAEHVDetectError(
        "TAEHV decoder artifact is missing, mixed, truncated, or has wrong tensor geometry"
    )


__all__ = [
    "TAEHVConfig",
    "TAEHVDetectError",
    "detect_taehv_decoder_config",
    "taehv_decoder_layout",
]
