"""Latent formats as declarative descriptors.

ComfyUI's comfy/latent_formats.py (@ b78cec87) is 35 classes that are
almost entirely constants - channels, scale/shift, preview matrices -
with occasional embedded reshape logic. The constants ARE the format;
here they are one frozen descriptor.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Generic, Literal, TypeVar

LatentDimensions = Literal[1, 2, 3]
T = TypeVar("T")
U = TypeVar("U")


def _validate_role(role: object) -> str:
    if type(role) is not str or not role:
        raise ValueError("latent stream roles must be nonempty strings")
    return role


@dataclass(frozen=True, slots=True)
class LatentStream(Generic[T]):
    """One payload identified by a stable semantic role."""

    role: str
    payload: T

    def __post_init__(self) -> None:
        _validate_role(self.role)
        if self.payload is None:
            raise ValueError("latent stream payload must not be None")


@dataclass(frozen=True, eq=False, slots=True)
class MultiStreamLatent(Generic[T]):
    """A nonempty, ordered collection of uniquely role-labeled payloads."""

    streams: tuple[LatentStream[T], ...]
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if type(self.streams) is not tuple or not self.streams:
            raise ValueError("multi-stream latent must contain at least one stream")
        if any(type(stream) is not LatentStream for stream in self.streams):
            raise TypeError("streams must contain exact LatentStream values")
        roles = self.roles
        if len(roles) != len(set(roles)):
            raise ValueError("latent stream roles must be unique")

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, T]]) -> MultiStreamLatent[T]:
        return cls(tuple(LatentStream(role, payload) for role, payload in pairs))

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(stream.role for stream in self.streams)

    def by_role(self, role: str) -> T:
        _validate_role(role)
        for stream in self.streams:
            if stream.role == role:
                return stream.payload
        raise KeyError(role)

    def map(self, transform: Callable[[T], U]) -> MultiStreamLatent[U]:
        return MultiStreamLatent(
            tuple(LatentStream(stream.role, transform(stream.payload)) for stream in self.streams)
        )

    def same_topology(self, other: object) -> bool:
        return type(other) is MultiStreamLatent and self.roles == other.roles

    def replace(self, role: str, payload: T) -> MultiStreamLatent[T]:
        if role not in self.roles:
            raise KeyError(role)
        return MultiStreamLatent(
            tuple(
                LatentStream(stream.role, payload if stream.role == role else stream.payload)
                for stream in self.streams
            )
        )


@dataclass(frozen=True, slots=True)
class LatentPackStreamLayout:
    """One stream's immutable location in a flattened latent pack."""

    role: str
    shape: tuple[int, ...]
    elements: int
    offset: int

    def __post_init__(self) -> None:
        _validate_role(self.role)
        if len(self.shape) < 2 or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("latent stream shapes require positive batch and content dimensions")
        expected = math.prod(self.shape[1:])
        if type(self.elements) is not int or self.elements != expected:
            raise ValueError("latent stream element count does not match its shape")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("latent stream offset must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class LatentPackLayout:
    """Immutable structural metadata needed to restore a flattened pack."""

    streams: tuple[LatentPackStreamLayout, ...]

    def __post_init__(self) -> None:
        if type(self.streams) is not tuple or not self.streams:
            raise ValueError("latent pack layout must contain at least one stream")
        if any(type(stream) is not LatentPackStreamLayout for stream in self.streams):
            raise TypeError("layout streams must be exact LatentPackStreamLayout values")
        roles = tuple(stream.role for stream in self.streams)
        if len(roles) != len(set(roles)):
            raise ValueError("latent pack layout roles must be unique")
        batch = self.streams[0].shape[0]
        offset = 0
        for stream in self.streams:
            if stream.shape[0] != batch:
                raise ValueError("latent pack stream batch sizes must match")
            if stream.offset != offset:
                raise ValueError("latent pack stream offsets must be contiguous and ordered")
            offset += stream.elements

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(stream.role for stream in self.streams)

    def by_role(self, role: str) -> LatentPackStreamLayout:
        _validate_role(role)
        for stream in self.streams:
            if stream.role == role:
                return stream
        raise KeyError(role)

    @property
    def batch_size(self) -> int:
        return self.streams[0].shape[0]

    @property
    def packed_elements(self) -> int:
        return sum(stream.elements for stream in self.streams)

    @property
    def packed_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, 1, self.packed_elements)


@dataclass(frozen=True)
class LatentDescriptor:
    """Everything a graph/engine needs to know about a latent space
    without executing a model.

    ``dimensions`` counts content axes beyond batch/channel: 1 = audio,
    2 = image, 3 = video/volume. ``rgb_factors`` (one triple per
    channel) and ``rgb_bias`` drive cheap previews; ``scale_factor`` /
    ``shift_factor`` define the default affine process_in/out
    (``(x - shift) * scale`` on the way in).

    ``temporal_causal`` marks 3D spaces whose first frame is not
    compressed: ``t`` latent frames decode to
    ``t * temporal_downscale - (temporal_downscale - 1)`` content
    frames (the reference's lambda downscale_ratio video VAEs,
    comfy/sd.py @ b78cec87); non-causal spaces scale time linearly.

    ``content_fps`` is the frame rate the family's decoded content plays
    at (16 for Wan); previews derive their display rate from it. None
    for non-temporal spaces or when the family fixes no rate.
    """

    channels: int
    dimensions: LatentDimensions = 2
    scale_factor: float = 1.0
    shift_factor: float = 0.0
    spatial_downscale: int = 8
    temporal_downscale: int = 1
    temporal_causal: bool = False
    content_fps: float | None = None
    rgb_factors: tuple[tuple[float, float, float], ...] | None = None
    rgb_bias: tuple[float, float, float] | None = None
    taesd_decoder: str | None = None

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.spatial_downscale <= 0 or self.temporal_downscale <= 0:
            raise ValueError("downscale ratios must be positive")
        if self.temporal_causal and self.dimensions != 3:
            raise ValueError(
                f"temporal_causal requires 3 content dimensions, got {self.dimensions}"
            )
        if self.content_fps is not None and self.content_fps <= 0:
            raise ValueError(f"content_fps must be positive, got {self.content_fps}")
        if self.rgb_factors is not None and len(self.rgb_factors) != self.channels:
            raise ValueError(
                f"rgb_factors must have one triple per channel "
                f"({self.channels}), got {len(self.rgb_factors)}"
            )


@dataclass(frozen=True)
class MultiStreamLatentDescriptor:
    """An ordered description of two or more named latent streams."""

    streams: tuple[tuple[str, LatentDescriptor], ...]

    def __post_init__(self) -> None:
        if type(self.streams) is not tuple:
            raise TypeError("multi-stream latent descriptor streams must be an exact tuple")
        if len(self.streams) < 2:
            raise ValueError("multi-stream latent descriptor requires at least two streams")
        names: list[str] = []
        for stream in self.streams:
            if type(stream) is not tuple or len(stream) != 2:
                raise TypeError("descriptor streams must be exact (name, descriptor) tuples")
            name, descriptor = stream
            if type(name) is not str:
                raise TypeError("descriptor stream names must be exact strings")
            if not name:
                raise ValueError("descriptor stream names must be nonempty")
            if type(descriptor) is not LatentDescriptor:
                raise TypeError("descriptor streams must contain exact LatentDescriptor values")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("descriptor stream names must be unique")


__all__ = [
    "LatentPackLayout",
    "LatentPackStreamLayout",
    "LatentDescriptor",
    "LatentDimensions",
    "LatentStream",
    "MultiStreamLatent",
    "MultiStreamLatentDescriptor",
]
