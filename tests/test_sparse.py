from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from dinkster_inference.sparse import (
    PBR_CHANNELS,
    SUBDIVISION_CHANNELS,
    SparseLatent,
    SparseSubdivisionGuides,
    SparseSupport,
    SparseVolume,
    TexturedMesh,
    TriangleMeshBatch,
)


@dataclass(frozen=True)
class Tensor:
    values: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.values), len(self.values[0]) if self.values else 0)

    def _apply(self, other: Tensor | float, operation: str) -> Tensor:
        if type(other) is Tensor:
            peer = other.values
        else:
            scalar = cast(float, other)
            peer = tuple((scalar,) * self.shape[1] for _ in self.values)

        def apply(a: float, b: float) -> float:
            if operation == "add":
                return a + b
            if operation == "sub":
                return a - b
            if operation == "mul":
                return a * b
            return a / b

        return Tensor(
            tuple(
                tuple(apply(a, b) for a, b in zip(left, right, strict=True))
                for left, right in zip(self.values, peer, strict=True)
            )
        )

    def __add__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "add")

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "sub")

    def __mul__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "mul")

    def __truediv__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "div")


def support(digit: str = "1") -> SparseSupport[Tensor]:
    return SparseSupport(
        Tensor(((0, 1, 2, 3), (0, 4, 5, 6))),
        (2,),
        64,
        (-0.5, -0.5, -0.5),
        (1.0 / 64, 1.0 / 64, 1.0 / 64),
        "sha256:" + digit * 64,
    )


def test_sparse_support_requires_complete_contiguous_layout_authority() -> None:
    value = support()
    assert value.point_count == 2
    assert value.batch_size == 1
    assert value.batch_slices == (slice(0, 2),)
    with pytest.raises(ValueError, match="account for every coordinate"):
        SparseSupport(
            value.coordinates,
            (1,),
            64,
            value.origin,
            value.voxel_size,
            value.support_id,
        )
    with pytest.raises(ValueError, match="positive ints"):
        SparseSupport(
            value.coordinates,
            (2, 0),
            64,
            value.origin,
            value.voxel_size,
            value.support_id,
        )


def test_sparse_latent_arithmetic_preserves_and_authenticates_support() -> None:
    first = SparseLatent(support(), Tensor(((1.0, 2.0), (3.0, 4.0))))
    second = SparseLatent(first.support, Tensor(((5.0, 6.0), (7.0, 8.0))))
    assert (first + second).features.values == ((6.0, 8.0), (10.0, 12.0))
    rehydrated = SparseLatent(support(), second.features)
    assert (first + rehydrated).features.values == ((6.0, 8.0), (10.0, 12.0))
    assert (first * 2.0).support is first.support
    assert (first / 2.0).features.values == ((0.5, 1.0), (1.5, 2.0))
    with pytest.raises(ValueError, match="identical support"):
        _ = first + SparseLatent(support("2"), second.features)


def test_sparse_latent_requires_exact_feature_row_correspondence() -> None:
    with pytest.raises(ValueError, match="one row per support coordinate"):
        SparseLatent(support(), Tensor(((1.0, 2.0),)))


def test_textured_mesh_retains_sparse_geometry_and_channel_semantics() -> None:
    pbr = SparseVolume(
        support(),
        Tensor(((0.1,) * 6, (0.2,) * 6)),
        PBR_CHANNELS,
    )
    mesh = TexturedMesh(
        Tensor(((0.0, 0.0, 0.0),)),
        Tensor(((0.0, 0.0, 0.0),)),
        pbr,
        "z_up",
    )
    assert mesh.pbr.support.resolution == 64
    assert mesh.pbr.support.origin == (-0.5, -0.5, -0.5)
    assert mesh.pbr.support.voxel_size == (1.0 / 64, 1.0 / 64, 1.0 / 64)
    generic = TexturedMesh(
        mesh.vertices,
        mesh.faces,
        SparseVolume(support(), Tensor(((0.1,), (0.2,))), ("density",)),
        "z_up",
    )
    assert generic.pbr.channels == ("density",)
    with pytest.raises(ValueError, match="shape \\(vertices, 3\\)"):
        TexturedMesh(Tensor(((0.0, 0.0),)), mesh.faces, pbr, "z_up")


def test_subdivision_guides_name_all_octree_child_columns() -> None:
    level = SparseVolume(
        support(),
        Tensor(((0.0,) * 8, (1.0,) * 8)),
        SUBDIVISION_CHANNELS,
    )
    guides = SparseSubdivisionGuides((level,))
    assert guides.levels[0].channels[0] == "subdivision.x0.y0.z0"
    assert guides.levels[0].channels[-1] == "subdivision.x1.y1.z1"


def test_triangle_mesh_batch_exposes_generic_mutable_mesh_surface() -> None:
    @dataclass(frozen=True)
    class BatchedTensor:
        shape: tuple[int, ...]

    vertices = BatchedTensor((1, 1, 3))
    faces = BatchedTensor((1, 1, 3))
    batch = TriangleMeshBatch(vertices, faces)

    assert batch.vertices is vertices
    assert batch.faces is faces
    assert batch.uvs is None
    batch.material = {"double_sided": True}
    assert batch.material == {"double_sided": True}
