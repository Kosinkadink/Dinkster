"""Torch-free TAESD/TAESDXL artifact contracts.

The two families intentionally have identical tensor geometry. Family is a
role supplied by SD assembly, not guessed from indistinguishable bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .autoencoder_kl import KLMemoryEstimator
from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT32
from .latents import LatentDescriptor
from .weights import TensorGeometry

TAESDFamily = Literal["sd15", "sdxl"]
TAESDRole = Literal["encoder", "decoder"]


class TAESDDetectError(ValueError):
    """An artifact is not one complete supported TAESD half."""


@dataclass(frozen=True)
class TAESDConfig:
    family: TAESDFamily
    role: TAESDRole
    latent_channels: int = 4
    hidden_channels: int = 64
    spatial_downscale: int = 8
    vae_scale: float | None = None
    vae_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in ("sd15", "sdxl"):
            raise ValueError(f"unsupported TAESD family {self.family!r}")
        if self.role not in ("encoder", "decoder"):
            raise ValueError(f"unsupported TAESD role {self.role!r}")
        expected_scale = 0.18215 if self.family == "sd15" else 0.13025
        if self.vae_scale is None:
            object.__setattr__(self, "vae_scale", expected_scale)
        elif self.vae_scale != expected_scale:
            raise ValueError(
                f"{self.family} TAESD vae_scale must be {expected_scale}, got {self.vae_scale}"
            )
        if self.vae_shift != 0.0:
            raise ValueError(f"TAESD vae_shift must be 0.0, got {self.vae_shift}")
        if self.latent_channels != 4:
            raise ValueError("TAESD latent width must be 4")
        if self.hidden_channels != 64 or self.spatial_downscale != 8:
            raise ValueError("only the native 64-wide x8 TAESD architecture is supported")


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


def _block(layout: dict[str, tuple[int, ...]], key: str, inc: int, out: int) -> None:
    _conv(layout, f"{key}.conv.0", out, inc)
    _conv(layout, f"{key}.conv.2", out, out)
    _conv(layout, f"{key}.conv.4", out, out)
    if inc != out:
        _conv(layout, f"{key}.skip", out, inc, bias=False, kernel=1)


def taesd_layout(role: TAESDRole) -> dict[str, tuple[int, ...]]:
    """Exact state-dict listing of comfy/taesd/taesd.py @ f4b99bc."""
    layout: dict[str, tuple[int, ...]] = {}
    if role == "encoder":
        _conv(layout, "0", 64, 3)
        _block(layout, "1", 64, 64)
        for start in (2, 6, 10):
            _conv(layout, str(start), 64, 64, bias=False)
            for index in range(start + 1, start + 4):
                _block(layout, str(index), 64, 64)
        _conv(layout, "14", 4, 64)
    else:
        _conv(layout, "1", 64, 4)
        for index in (3, 4, 5):
            _block(layout, str(index), 64, 64)
        _conv(layout, "7", 64, 64, bias=False)
        for index in (8, 9, 10):
            _block(layout, str(index), 64, 64)
        _conv(layout, "12", 64, 64, bias=False)
        for index in (13, 14, 15):
            _block(layout, str(index), 64, 64)
        _conv(layout, "17", 64, 64, bias=False)
        _block(layout, "18", 64, 64)
        _conv(layout, "19", 3, 64)
    return layout


def detect_taesd_config(
    geometries: Mapping[str, TensorGeometry],
    *,
    family: TAESDFamily,
    role: TAESDRole | None = None,
) -> TAESDConfig:
    """Strictly detect one independent encoder or decoder header."""
    prefixes: Mapping[TAESDRole, str] = {
        "encoder": "taesd_encoder.",
        "decoder": "taesd_decoder.",
    }
    candidates: list[tuple[TAESDRole, dict[str, TensorGeometry]]] = []
    for candidate, prefix in prefixes.items():
        # A combined VAE artifact contains both prefixed halves. A bare
        # official madebyollin artifact contains exactly one unprefixed half.
        stripped = {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in geometries.items()
            if k.startswith(prefix) or not any(k.startswith(other) for other in prefixes.values())
        }
        expected = taesd_layout(candidate)
        if set(stripped) == set(expected) and all(
            stripped[k].shape == shape for k, shape in expected.items()
        ):
            candidates.append((candidate, stripped))
    if role is not None:
        candidates = [item for item in candidates if item[0] == role]
    if len(candidates) != 1:
        wanted = role or "encoder or decoder"
        raise TAESDDetectError(
            f"TAESD {wanted} artifact is missing, mixed, truncated, or has wrong tensor geometry"
        )
    return TAESDConfig(family=family, role=candidates[0][0])


def taesd_descriptor(family: TAESDFamily) -> CodecDescriptor:
    return CodecDescriptor(
        id=f"dinkster.taesd.{family}",
        display_name="TAESD" if family == "sd15" else "TAESDXL",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
        tiling=CodecTiling(
            decode_tile=(64, 64),
            decode_overlap=(16, 16),
            encode_tile=(512, 512),
            encode_overlap=(64, 64),
        ),
    )


class TAESDMemoryEstimator(KLMemoryEstimator):
    """Pinned ComfyUI TAESD memory policy.

    At f4b99bc the TAESD branch keeps the VAE wrapper's default KL
    estimates. This named subtype records that grounding while sharing the
    exact formulas and optional AMD ratio with :class:`KLMemoryEstimator`.
    """


__all__ = [
    "TAESDConfig",
    "TAESDDetectError",
    "TAESDMemoryEstimator",
    "detect_taesd_config",
    "taesd_descriptor",
    "taesd_layout",
]
