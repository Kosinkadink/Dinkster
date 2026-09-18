"""Torch-free sparse latent, sparse volume, and textured mesh contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from .patches import SizedTensor
from .sampling import ArithTensor

T = TypeVar("T", bound=SizedTensor, covariant=True)


class _SparseFeatureTensor(ArithTensor, SizedTensor, Protocol):
    pass


A = TypeVar("A", bound=_SparseFeatureTensor)

_SUPPORT_ID = re.compile(r"sha256:[0-9a-f]{64}")


def _shape(value: SizedTensor, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must expose an integer shape") from error
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError(f"{name} shape must contain nonnegative exact integers")
    return shape


def _vector3(value: object, name: str, *, positive: bool = False) -> tuple[float, float, float]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple of three finite floats")
    items = cast("tuple[object, ...]", value)
    if len(items) != 3 or any(type(item) is not float or not math.isfinite(item) for item in items):
        raise TypeError(f"{name} must be an exact tuple of three finite floats")
    result = cast("tuple[float, float, float]", items)
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True, eq=False, slots=True)
class DenseVoxelGrid(Generic[T]):
    """One batched dense voxel field with explicit channel semantics."""

    values: T
    channels: tuple[str, ...]
    frame: str
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        value_shape = _shape(self.values, "dense voxel values")
        if len(value_shape) != 5 or value_shape[0] < 1 or value_shape[1] < 1:
            raise ValueError("dense voxel values must have shape (batch, channels, x, y, z)")
        if any(size < 1 for size in value_shape[2:]) or len(set(value_shape[2:])) != 1:
            raise ValueError("dense voxel spatial dimensions must be nonempty and cubic")
        if (
            type(self.channels) is not tuple
            or not self.channels
            or any(type(channel) is not str or not channel for channel in self.channels)
            or len(set(self.channels)) != len(self.channels)
            or len(self.channels) != value_shape[1]
        ):
            raise ValueError("dense voxel channels must describe every unique feature channel")
        if self.frame not in ("y_up", "z_up"):
            raise ValueError("dense voxel frame must be y_up or z_up")

    @property
    def resolution(self) -> int:
        return tuple(self.values.shape)[-1]

    @property
    def data(self) -> T:
        """Single-channel voxel data for generic Comfy-compatible consumers."""
        if len(self.channels) != 1:
            raise ValueError("dense voxel data requires exactly one channel")
        return cast("T", cast("Any", self.values)[:, 0])

    @property
    def voxel_colors(self) -> None:
        return None


@dataclass(frozen=True, eq=False, slots=True)
class SparseSupport(Generic[T]):
    """Authenticated coordinates and contiguous per-batch row layout."""

    coordinates: T
    batch_counts: tuple[int, ...]
    resolution: int
    origin: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    support_id: str
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        coordinate_shape = _shape(self.coordinates, "sparse coordinates")
        if len(coordinate_shape) != 2 or coordinate_shape[1] != 4:
            raise ValueError("sparse coordinates must have shape (points, 4)")
        if (
            type(self.batch_counts) is not tuple
            or not self.batch_counts
            or any(type(count) is not int or count < 1 for count in self.batch_counts)
        ):
            raise ValueError("sparse batch counts must be a nonempty tuple of positive ints")
        if sum(self.batch_counts) != coordinate_shape[0]:
            raise ValueError("sparse batch counts must account for every coordinate row")
        if type(self.resolution) is not int or self.resolution < 1:
            raise ValueError("sparse resolution must be a positive exact int")
        _vector3(self.origin, "sparse origin")
        _vector3(self.voxel_size, "sparse voxel size", positive=True)
        if type(self.support_id) is not str or _SUPPORT_ID.fullmatch(self.support_id) is None:
            raise ValueError("sparse support id must be a canonical sha256 digest")

    @property
    def point_count(self) -> int:
        return sum(self.batch_counts)

    @property
    def batch_size(self) -> int:
        return len(self.batch_counts)

    @property
    def batch_slices(self) -> tuple[slice, ...]:
        slices: list[slice] = []
        offset = 0
        for count in self.batch_counts:
            slices.append(slice(offset, offset + count))
            offset += count
        return tuple(slices)

    def same_support(self, other: object) -> bool:
        if type(other) is not SparseSupport:
            return False
        peer = cast("SparseSupport[SizedTensor]", other)
        return (
            self.support_id == peer.support_id
            and self.batch_counts == peer.batch_counts
            and self.resolution == peer.resolution
            and self.origin == peer.origin
            and self.voxel_size == peer.voxel_size
            and tuple(self.coordinates.shape) == tuple(peer.coordinates.shape)
        )


@dataclass(frozen=True, eq=False, slots=True)
class SparseLatent(Generic[A]):
    """Sampled feature rows bound to one immutable sparse support."""

    support: SparseSupport[SizedTensor]
    features: A
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if type(self.support) is not SparseSupport:
            raise TypeError("sparse latent support must be an exact SparseSupport")
        feature_shape = _shape(self.features, "sparse latent features")
        if len(feature_shape) != 2 or feature_shape[1] < 1:
            raise ValueError("sparse latent features must have shape (points, channels)")
        if feature_shape[0] != self.support.point_count:
            raise ValueError("sparse latent features must have one row per support coordinate")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.features.shape)

    def replace(self, features: A) -> SparseLatent[A]:
        return SparseLatent(self.support, features)

    def _operand(self, other: object) -> A | float:
        if type(other) is SparseLatent:
            peer = cast("SparseLatent[A]", other)
            if not self.support.same_support(peer.support):
                raise ValueError("sparse latent arithmetic requires identical support")
            if self.shape != peer.shape:
                raise ValueError("sparse latent arithmetic requires identical feature shapes")
            return peer.features
        if type(other) in (int, float) and not isinstance(other, bool):
            return float(cast("int | float", other))
        raise TypeError("sparse latent arithmetic requires a scalar or exact SparseLatent")

    def __add__(self, other: SparseLatent[A] | float) -> SparseLatent[A]:
        return self.replace(self.features + self._operand(other))

    def __sub__(self, other: SparseLatent[A] | float) -> SparseLatent[A]:
        return self.replace(self.features - self._operand(other))

    def __mul__(self, other: SparseLatent[A] | float) -> SparseLatent[A]:
        return self.replace(self.features * self._operand(other))

    def __truediv__(self, other: SparseLatent[A] | float) -> SparseLatent[A]:
        features = cast("_SparseFeatureTensor", self.features)
        return self.replace(cast("A", features / cast("Any", self._operand(other))))


@dataclass(frozen=True, eq=False, slots=True)
class SparseVolume(Generic[T]):
    """Sparse voxel rows with explicit channel semantics."""

    support: SparseSupport[T]
    features: T
    channels: tuple[str, ...]
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if type(self.support) is not SparseSupport:
            raise TypeError("sparse volume support must be an exact SparseSupport")
        feature_shape = _shape(self.features, "sparse volume features")
        if len(feature_shape) != 2 or feature_shape[0] != self.support.point_count:
            raise ValueError("sparse volume features must have shape (support points, channels)")
        if (
            type(self.channels) is not tuple
            or not self.channels
            or any(type(channel) is not str or not channel for channel in self.channels)
            or len(set(self.channels)) != len(self.channels)
        ):
            raise ValueError("sparse volume channels must be nonempty, unique strings")
        if feature_shape[1] != len(self.channels):
            raise ValueError("sparse volume channel names must describe every feature column")

    @property
    def data(self) -> T:
        """Sparse coordinates for generic Comfy-compatible voxel consumers."""
        return self.support.coordinates

    @property
    def voxel_colors(self) -> T:
        return self.features

    @property
    def resolution(self) -> int:
        return self.support.resolution


@dataclass(frozen=True, eq=False, slots=True)
class SparseSubdivisionGuides(Generic[T]):
    """Ordered sparse subdivision decisions for a matching decoder."""

    levels: tuple[SparseVolume[T], ...]
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if (
            type(self.levels) is not tuple
            or not self.levels
            or any(type(level) is not SparseVolume for level in self.levels)
        ):
            raise TypeError("sparse subdivision guides must be a nonempty tuple of volumes")
        if any(level.channels != SUBDIVISION_CHANNELS for level in self.levels):
            raise ValueError(
                "sparse subdivision guide volumes must describe the eight octree children"
            )


SUBDIVISION_CHANNELS = tuple(
    f"subdivision.x{child % 2}.y{child // 2 % 2}.z{child // 4 % 2}" for child in range(8)
)


PBR_CHANNELS = (
    "base_color.r",
    "base_color.g",
    "base_color.b",
    "metallic",
    "roughness",
    "alpha",
)


@dataclass(frozen=True, eq=False, slots=True)
class TriangleMesh(Generic[T]):
    """One indexed triangle mesh in an explicit coordinate frame."""

    vertices: T
    faces: T
    frame: str
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        vertex_shape = _shape(self.vertices, "mesh vertices")
        if len(vertex_shape) != 2 or vertex_shape[1] != 3:
            raise ValueError("mesh vertices must have shape (vertices, 3)")
        face_shape = _shape(self.faces, "mesh faces")
        if len(face_shape) != 2 or face_shape[1] != 3:
            raise ValueError("mesh faces must have shape (triangles, 3)")
        if self.frame not in ("y_up", "z_up"):
            raise ValueError("mesh frame must be y_up or z_up")


@dataclass(eq=False, slots=True)
class TriangleMeshBatch(Generic[T]):
    """Padded triangle-mesh batch with optional per-item lengths and material data."""

    vertices: T
    faces: T
    uvs: T | None = None
    vertex_colors: T | None = None
    texture: T | None = None
    metallic_roughness: T | None = None
    vertex_counts: T | None = None
    face_counts: T | None = None
    unlit: bool = False
    normals: T | None = None
    tangents: T | None = None
    normal_map: T | None = None
    occlusion_in_mr: bool = False
    material: object | None = None
    emissive: T | None = None
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        vertex_shape = _shape(self.vertices, "mesh batch vertices")
        face_shape = _shape(self.faces, "mesh batch faces")
        if len(vertex_shape) != 3 or vertex_shape[0] < 1 or vertex_shape[2] != 3:
            raise ValueError("mesh batch vertices must have shape (batch, vertices, 3)")
        if len(face_shape) != 3 or face_shape[0] != vertex_shape[0] or face_shape[2] != 3:
            raise ValueError("mesh batch faces must have shape (batch, triangles, 3)")
        if (self.vertex_counts is None) != (self.face_counts is None):
            raise ValueError("mesh batch vertex and face counts must be provided together")


@dataclass(frozen=True, eq=False, slots=True)
class TexturedMesh(Generic[T]):
    """One indexed triangle mesh with a sparse PBR voxel field."""

    vertices: T
    faces: T
    pbr: SparseVolume[T]
    frame: str
    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        vertex_shape = _shape(self.vertices, "mesh vertices")
        if len(vertex_shape) != 2 or vertex_shape[1] != 3:
            raise ValueError("mesh vertices must have shape (vertices, 3)")
        face_shape = _shape(self.faces, "mesh faces")
        if len(face_shape) != 2 or face_shape[1] != 3:
            raise ValueError("mesh faces must have shape (triangles, 3)")
        if type(self.pbr) is not SparseVolume:
            raise TypeError("mesh PBR data must be an exact SparseVolume")
        if self.frame not in ("y_up", "z_up"):
            raise ValueError("mesh frame must be y_up or z_up")


__all__ = [
    "DenseVoxelGrid",
    "PBR_CHANNELS",
    "SparseLatent",
    "SparseSubdivisionGuides",
    "SparseSupport",
    "SparseVolume",
    "SUBDIVISION_CHANNELS",
    "TexturedMesh",
    "TriangleMesh",
    "TriangleMeshBatch",
]
