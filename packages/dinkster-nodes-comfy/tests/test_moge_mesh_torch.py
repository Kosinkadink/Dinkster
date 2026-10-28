"""Point-map triangulation, frame conversion, batch selection and textures."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.moge_mesh import MoGePointMapToMesh
from dinkster_values import decode_latent, encode_latent


def geometry(perspective: bool, size: int = 3) -> dict[str, Any]:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size + 2), indexing="ij")
    # Spherical Z is negative here while the supplied radial depth is positive.
    points = torch.stack((x * 2 + 1, y * 3 + 2, torch.full_like(x, -7)), -1).float()
    packet: dict[str, Any] = {
        "points": torch.stack((points + 100, points)),
        "depth": torch.full((2, size, size + 2), 9.0),
        "image": torch.arange(2 * size * (size + 2) * 3).reshape(2, size, size + 2, 3).float(),
    }
    if perspective:
        packet["intrinsics"] = torch.eye(3).repeat(2, 1, 1)
    return packet


@pytest.mark.parametrize("perspective", (False, True))
@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("texture", (False, True))
def test_selected_geometry_mesh_frames_uvs_and_texture(
    perspective: bool, portable: bool, texture: bool
) -> None:
    packet = geometry(perspective)
    before = {k: v.clone() for k, v in packet.items()}
    value = decode_latent(encode_latent(packet), require_samples=False) if portable else packet
    mesh: Any = MoGePointMapToMesh.execute(
        moge_geometry=value,
        batch_index=1,
        decimation=2,
        discontinuity_threshold=0,
        texture=texture,
    )["MESH"]
    # Stride two selects a 2x3 grid; its two quads yield four triangles.
    vertices = torch.tensor(
        [[1, 2, -7], [5, 2, -7], [9, 2, -7], [1, 8, -7], [5, 8, -7], [9, 8, -7]],
        dtype=torch.float32,
    )
    expected = vertices * torch.tensor([1, -1, -1]) if perspective else vertices[:, [1, 2, 0]]
    faces = torch.tensor([[0, 1, 4], [1, 2, 5], [0, 4, 3], [1, 5, 4]])
    if perspective:
        faces = faces[:, [0, 2, 1]]
    torch.testing.assert_close(mesh.vertices, expected[None])
    torch.testing.assert_close(mesh.faces, faces[None])
    torch.testing.assert_close(
        mesh.uvs,
        torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]]),
    )
    if texture:
        torch.testing.assert_close(mesh.texture, before["image"][1:2])
    else:
        assert mesh.texture is None
    for key in before:
        torch.testing.assert_close(packet[key], before[key])


def test_spherical_radial_depth_prevents_false_discontinuity() -> None:
    packet = geometry(False)
    packet["points"][1, ..., 2] = -torch.arange(1.0, 6.0)
    # Negative spherical Z would fail the relative-depth edge check; radial
    # depth remains constant and permits every quad on the spherical mesh.
    mesh: Any = MoGePointMapToMesh.execute(
        moge_geometry=packet,
        batch_index=1,
        discontinuity_threshold=0.04,
    )["MESH"]
    assert mesh.vertices.shape == (1, 15, 3)
    assert mesh.faces.shape == (1, 16, 3)
    del packet["depth"]
    with pytest.raises(ValueError, match="MoGe produced an empty mesh"):
        MoGePointMapToMesh.execute(moge_geometry=packet, batch_index=1)


def test_missing_depth_uses_point_depth_without_texture_image() -> None:
    packet = geometry(True)
    packet["points"][..., 2] = 3
    del packet["depth"], packet["image"]
    mesh: Any = MoGePointMapToMesh.execute(moge_geometry=packet, texture=False)["MESH"]
    assert mesh.faces.shape == (1, 16, 3) and mesh.texture is None


def test_geometry_errors_and_empty_mesh() -> None:
    with pytest.raises(ValueError, match="moge_geometry has no points output"):
        MoGePointMapToMesh.execute(moge_geometry={})
    with pytest.raises(
        ValueError, match="batch_index 2 out of range; moge_geometry has batch size 2"
    ):
        MoGePointMapToMesh.execute(moge_geometry=geometry(True), batch_index=2)
    packet = geometry(True)
    packet["points"][:] = float("inf")
    with pytest.raises(ValueError, match="MoGe produced an empty mesh"):
        MoGePointMapToMesh.execute(moge_geometry=packet)
