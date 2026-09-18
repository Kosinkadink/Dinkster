from __future__ import annotations

import pytest
from dinkster_inference import UspMesh, UspMeshCoordinate, UspMeshError


def test_usp_mesh_keeps_logical_rank_and_coordinate_api() -> None:
    mesh = UspMesh(2, 2, 2)

    assert mesh.ranks == tuple(range(8))
    assert mesh.world_size == 8
    for rank in mesh.ranks:
        assert mesh.rank_of(mesh.coordinates(rank)) == rank
    assert mesh.coordinates(7) == UspMeshCoordinate(1, 1, 1)


def test_usp_mesh_groups_are_logical_and_elide_degree_one_axes() -> None:
    mesh = UspMesh.build(guidance=1, ulysses=2, ring=2)

    assert mesh.ring_groups() == ((0, 1), (2, 3))
    assert mesh.ulysses_groups() == ((0, 2), (1, 3))
    assert mesh.guidance_groups() == ()
    assert mesh.sequence_groups() == ((0, 1, 2, 3),)


def test_usp_mesh_identity_uses_full_normalized_process_mesh_form() -> None:
    mesh = UspMesh(1, 2, 2)

    assert mesh.identity_facts() == (
        "process_mesh_layout=guidance-tp-sp_ulysses-sp_ring.v1",
        "process_mesh_axes=cfg1xtp1xu2xr2",
        "process_mesh_world_size=4",
        f"process_mesh_subgroup_digest={mesh.process_mesh.subgroup_digest}",
    )
    assert mesh.digest == mesh.process_mesh.digest


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1"))
@pytest.mark.parametrize("axis", ("guidance", "ulysses", "ring"))
def test_usp_mesh_rejects_invalid_axis_sizes(axis: str, value: object) -> None:
    axes: dict[str, object] = {"guidance": 1, "ulysses": 1, "ring": 1}
    axes[axis] = value

    with pytest.raises(UspMeshError, match="axis size"):
        UspMesh(**axes)  # type: ignore[arg-type]


def test_usp_mesh_rejects_invalid_rank_and_coordinate_inputs() -> None:
    mesh = UspMesh(2, 2, 1)

    for rank in (True, -1, 4):
        with pytest.raises(UspMeshError, match="rank"):
            mesh.coordinates(rank)
    with pytest.raises(UspMeshError, match="exact UspMeshCoordinate"):
        mesh.rank_of((0, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(UspMeshError, match="coordinate"):
        mesh.rank_of(UspMeshCoordinate(2, 0, 0))
    with pytest.raises(UspMeshError, match="exact int"):
        UspMeshCoordinate(True, 0, 0)


def test_usp_mesh_constructor_no_longer_accepts_a_rank_permutation() -> None:
    with pytest.raises(TypeError):
        UspMesh(1, 1, 3, (2, 1, 0))  # type: ignore[call-arg]
