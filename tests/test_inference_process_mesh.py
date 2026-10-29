from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference import (
    PlacementMap,
    ProcessMesh,
    ProcessMeshCoordinate,
    ProcessMeshError,
)


def test_process_mesh_uses_row_major_logical_ranks_and_coordinates() -> None:
    mesh = ProcessMesh(2, 2, 2, 2)

    assert mesh.ranks == tuple(range(16))
    assert mesh.coordinates(0) == ProcessMeshCoordinate(0, 0, 0, 0)
    assert mesh.coordinates(7) == ProcessMeshCoordinate(0, 1, 1, 1)
    assert mesh.coordinates(10) == ProcessMeshCoordinate(1, 0, 1, 0)
    assert mesh.coordinates(15) == ProcessMeshCoordinate(1, 1, 1, 1)
    for rank in mesh.ranks:
        assert mesh.rank_of(mesh.coordinates(rank)) == rank


def test_process_mesh_derives_all_six_subgroup_families() -> None:
    mesh = ProcessMesh(2, 2, 2, 2)

    assert mesh.guidance_groups() == (
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
    )
    assert mesh.tp_groups() == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (8, 12),
        (9, 13),
        (10, 14),
        (11, 15),
    )
    assert mesh.ulysses_groups() == (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
        (12, 14),
        (13, 15),
    )
    assert mesh.ring_groups() == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
    )
    assert mesh.sequence_groups() == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    )
    assert mesh.model_parallel_groups() == (
        tuple(range(8)),
        tuple(range(8, 16)),
    )


def test_degree_one_axes_remain_in_identity_but_create_no_singleton_groups() -> None:
    mesh = ProcessMesh(1, 1, 2, 2)

    assert mesh.guidance_groups() == ()
    assert mesh.tp_groups() == ()
    assert mesh.ulysses_groups() == ((0, 2), (1, 3))
    assert mesh.ring_groups() == ((0, 1), (2, 3))
    assert mesh.sequence_groups() == ((0, 1, 2, 3),)
    assert mesh.model_parallel_groups() == ((0, 1, 2, 3),)
    assert mesh.identity_facts() == (
        "process_mesh_layout=guidance-tp-sp_ulysses-sp_ring.v1",
        "process_mesh_axes=cfg1xtp1xu2xr2",
        "process_mesh_world_size=4",
        f"process_mesh_subgroup_digest={mesh.subgroup_digest}",
    )
    expected = hashlib.sha256(":".join(mesh.identity_facts()).encode()).hexdigest()
    assert mesh.digest == expected

    degenerate = ProcessMesh(1, 1, 1, 1)
    assert degenerate.identity_facts()[1] == "process_mesh_axes=cfg1xtp1xu1xr1"
    assert len(degenerate.subgroup_digest) == 64
    assert all(
        groups == ()
        for groups in (
            degenerate.guidance_groups(),
            degenerate.tp_groups(),
            degenerate.ulysses_groups(),
            degenerate.ring_groups(),
            degenerate.sequence_groups(),
            degenerate.model_parallel_groups(),
        )
    )


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1"))
@pytest.mark.parametrize("axis", ("guidance", "tp", "sp_ulysses", "sp_ring"))
def test_process_mesh_rejects_invalid_axis_degrees(axis: str, value: object) -> None:
    values: dict[str, object] = {"guidance": 1, "tp": 1, "sp_ulysses": 1, "sp_ring": 1}
    values[axis] = value

    with pytest.raises(ProcessMeshError, match="axis degree"):
        ProcessMesh(**values)  # type: ignore[arg-type]


def test_process_mesh_rejects_invalid_rank_and_coordinate_records() -> None:
    mesh = ProcessMesh(2, 2, 1, 1)

    for rank in (True, -1, 4):
        with pytest.raises(ProcessMeshError, match="rank"):
            mesh.coordinates(rank)
    with pytest.raises(ProcessMeshError, match="exact int"):
        ProcessMeshCoordinate(0, True, 0, 0)
    with pytest.raises(ProcessMeshError, match="exact ProcessMeshCoordinate"):
        mesh.rank_of((0, 0, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(ProcessMeshError, match="coordinate"):
        mesh.rank_of(ProcessMeshCoordinate(2, 0, 0, 0))


def test_process_mesh_coordinate_rejects_negative_direct_construction() -> None:
    with pytest.raises(ProcessMeshError, match=">= 0"):
        ProcessMeshCoordinate(-1, -1, -1, -1)


def test_subgroup_digest_distinguishes_axis_shapes_with_equal_world_sizes() -> None:
    assert ProcessMesh(1, 1, 2, 2).subgroup_digest != ProcessMesh(1, 1, 1, 4).subgroup_digest


def test_placement_map_identity_and_nonidentity_facts() -> None:
    mesh = ProcessMesh(1, 1, 1, 4)
    identity = PlacementMap.identity(mesh)
    placement = PlacementMap(mesh, (2, 0, 3, 1))

    assert identity.identity_facts() == ()
    assert placement.identity_facts() == ("process_mesh_rank_order=2,0,3,1",)
    assert placement.mesh_digest == mesh.digest
    assert placement.apply((0, 1, 3)) == (2, 0, 1)
    assert placement.logical_rank(3) == 2
    assert placement.physical_rank(2) == 3
    with pytest.raises(FrozenInstanceError):
        placement.physical_ranks = (0, 1, 2, 3)  # type: ignore[misc]


@pytest.mark.parametrize(
    "build",
    (
        lambda: PlacementMap("bad", (0,)),  # type: ignore[arg-type]
        lambda: PlacementMap(ProcessMesh(1, 1, 1, 1), ()),
        lambda: PlacementMap(ProcessMesh(1, 1, 1, 1), [0]),  # type: ignore[arg-type]
        lambda: PlacementMap(ProcessMesh(1, 1, 1, 1), (True,)),
        lambda: PlacementMap(ProcessMesh(1, 1, 1, 2), (0, 0)),
        lambda: PlacementMap(ProcessMesh(1, 1, 1, 1), (1,)),
    ),
)
def test_placement_map_rejects_invalid_construction(build: Callable[[], object]) -> None:
    with pytest.raises(ProcessMeshError):
        build()


def test_placement_map_rejects_wrong_rank_count_for_mesh() -> None:
    with pytest.raises(ProcessMeshError, match="length"):
        PlacementMap(ProcessMesh(1, 1, 1, 2), (0,))


def test_placement_map_rejects_invalid_lookup_inputs() -> None:
    placement = PlacementMap(ProcessMesh(1, 1, 1, 2), (1, 0))

    for rank in (True, -1, 2):
        with pytest.raises(ProcessMeshError):
            placement.physical_rank(rank)
        with pytest.raises(ProcessMeshError):
            placement.logical_rank(rank)
    with pytest.raises(ProcessMeshError, match="tuple"):
        placement.apply([0])  # type: ignore[arg-type]
