"""Codecs (VAE and kin) as plugins, not a signature switch.

ComfyUI's VAE class is one object that detects ~30 codec families by
key signatures and mutates its own layout/memory/tiling fields per
branch (comfy/sd.py VAE @ b78cec87). Here each codec is a descriptor
plus encoder/decoder implementations; a TAESD-style tiny codec and a
video VAE register the same way. Tiled traversal is planned here
(torch-free, from what the descriptor declares) and executed by
dinkster-inference-torch codecs.py; OOM fallback policy stays with the
executing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from .devices import DType
from .latents import LatentDescriptor
from .patches import SizedTensor
from .tiling import (
    CausalScale,
    LinearScale,
    Scale,
    TilePlan,
    TilePlanError,
    plan_tiles,
)
from .weights import TensorGeometry

T = TypeVar("T", bound=SizedTensor)

CodecKind = Literal["image", "video", "audio", "volume"]


class CodecMemoryEstimator(Protocol):
    """Bytes needed to encode/decode content of a given geometry - the
    typed home for the per-family formulas embedded in comfy/sd.py VAE
    memory_used_* lambdas @ b78cec87."""

    def encode_bytes(self, content: TensorGeometry) -> int: ...

    def decode_bytes(self, latent: TensorGeometry) -> int: ...


class CodecEncoder(Protocol[T]):
    def encode(self, content: T) -> T: ...


class CodecDecoder(Protocol[T]):
    def decode(self, latent: T) -> T: ...


@dataclass(frozen=True)
class CodecTiling:
    """Tiling defaults a codec declares; the executing layer plans
    against them (explicit sizes at call time override). Decode
    entries are in LATENT units and encode entries in CONTENT units,
    one per content dimension - the same convention as the
    reference's decode_tiled_*/encode_tiled_* defaults (comfy/sd.py
    @ b78cec87: 2D decode tile 64 latents / overlap 16, 2D encode
    tile 512 pixels / overlap 64)."""

    decode_tile: tuple[int, ...]
    decode_overlap: tuple[int, ...]
    encode_tile: tuple[int, ...]
    encode_overlap: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("decode_tile", "encode_tile"):
            if any(v < 1 for v in getattr(self, name)):
                raise ValueError(f"{name} entries must be >= 1")
        for name in ("decode_overlap", "encode_overlap"):
            if any(v < 0 for v in getattr(self, name)):
                raise ValueError(f"{name} entries must be >= 0")


@dataclass(frozen=True)
class CodecDescriptor:
    """One registered codec: identity, latent space, content kind,
    content channel count (decoded space: 3 for RGB image/video
    codecs, waveform channels for audio), and the dtypes it can run
    in. ``supports_tiling`` tells the executor whether an
    overlap/feather tile plan is a legal fallback; ``tiling`` carries
    the codec's default tile geometry when it is."""

    id: str
    display_name: str
    kind: CodecKind
    latent: LatentDescriptor
    supported_dtypes: frozenset[DType]
    content_channels: int = 3
    supports_tiling: bool = True
    tiling: CodecTiling | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.content_channels < 1:
            raise ValueError(f"content_channels must be >= 1, got {self.content_channels}")
        if self.tiling is None:
            return
        if not self.supports_tiling:
            raise ValueError(
                f"codec {self.id}: tiling defaults declared but supports_tiling is False"
            )
        dims = self.latent.dimensions
        for name in (
            "decode_tile",
            "decode_overlap",
            "encode_tile",
            "encode_overlap",
        ):
            entries = getattr(self.tiling, name)
            if len(entries) != dims:
                raise ValueError(
                    f"codec {self.id}: tiling.{name} has {len(entries)}"
                    f" entries for {dims} content dimensions"
                )


def latent_scales(latent: LatentDescriptor) -> tuple[Scale, ...]:
    """Per-content-dimension scale rules between latent and content
    space. 3D content is (time, height, width); the time axis is
    :class:`CausalScale` when the descriptor declares causal temporal
    compression, otherwise linear. The 1D (audio) axis uses
    ``spatial_downscale`` - the descriptor's single content-axis
    ratio, as in the reference's 1D paths."""
    spatial = LinearScale(latent.spatial_downscale)
    if latent.dimensions == 1:
        return (spatial,)
    if latent.dimensions == 2:
        return (spatial, spatial)
    temporal: Scale
    if latent.temporal_causal:
        temporal = CausalScale(latent.temporal_downscale)
    else:
        temporal = LinearScale(latent.temporal_downscale)
    return (temporal, spatial, spatial)


def _codec_plan(
    descriptor: CodecDescriptor,
    shape: tuple[int, ...],
    tile: tuple[int, ...] | None,
    overlap: tuple[int, ...] | None,
    *,
    downscale: bool,
    direction: str,
) -> TilePlan:
    if not descriptor.supports_tiling:
        raise TilePlanError(f"codec {descriptor.id} does not support tiled {direction}")
    if tile is None or overlap is None:
        tiling = descriptor.tiling
        if tiling is None:
            raise TilePlanError(
                f"codec {descriptor.id} declares no tiling defaults;"
                f" pass tile and overlap explicitly"
            )
        if downscale:
            default_tile, default_overlap = (
                tiling.encode_tile,
                tiling.encode_overlap,
            )
        else:
            default_tile, default_overlap = (
                tiling.decode_tile,
                tiling.decode_overlap,
            )
        if tile is None:
            tile = default_tile
        if overlap is None:
            overlap = default_overlap
    return plan_tiles(
        shape,
        tile,
        overlap=overlap,
        scale=latent_scales(descriptor.latent),
        downscale=downscale,
    )


def plan_codec_decode(
    descriptor: CodecDescriptor,
    latent_shape: tuple[int, ...],
    *,
    tile: tuple[int, ...] | None = None,
    overlap: tuple[int, ...] | None = None,
) -> TilePlan:
    """Plan a tiled decode of latent content ``latent_shape`` (content
    dims only, latent units). Defaults come from the descriptor's
    :class:`CodecTiling`; refuses with :class:`TilePlanError` when the
    codec forbids tiling or declares no defaults and none are given."""
    return _codec_plan(
        descriptor,
        latent_shape,
        tile,
        overlap,
        downscale=False,
        direction="decode",
    )


def plan_codec_encode(
    descriptor: CodecDescriptor,
    content_shape: tuple[int, ...],
    *,
    tile: tuple[int, ...] | None = None,
    overlap: tuple[int, ...] | None = None,
) -> TilePlan:
    """Plan a tiled encode of content ``content_shape`` (content dims
    only, content units). The plan divides by the scale rules (the
    reference's ``downscale=True`` encode direction)."""
    return _codec_plan(
        descriptor,
        content_shape,
        tile,
        overlap,
        downscale=True,
        direction="encode",
    )


__all__ = [
    "CodecDecoder",
    "CodecDescriptor",
    "CodecEncoder",
    "CodecKind",
    "CodecMemoryEstimator",
    "CodecTiling",
    "latent_scales",
    "plan_codec_decode",
    "plan_codec_encode",
]
