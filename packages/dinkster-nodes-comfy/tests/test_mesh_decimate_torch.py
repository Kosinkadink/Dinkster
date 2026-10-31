"""Mesh simplification keeps per-item limits, colors and source values intact."""

from typing import Any

import pytest
import torch
from dinkster_api.v1 import TriangleMeshBatch, materialize_mesh
from dinkster_nodes_comfy.mesh_decimate import DecimateMesh


def source() -> Any:
    size = 8
    vertices = torch.tensor(
        [[x * 0.7, y * 1.2, 0.1 * ((x * y) % 3)] for y in range(size) for x in range(size)],
        dtype=torch.float32,
    )
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend([[a, a + 1, a + size], [a + 1, a + size + 1, a + size]])
    return TriangleMeshBatch(
        vertices=vertices[None],
        faces=torch.tensor([faces]),
        vertex_colors=torch.tensor([0.2, 0.7, 0.4, 0.9]).expand(1, len(vertices), 4).clone(),
        normals=torch.ones(1, len(vertices), 3),
        uvs=torch.ones(1, len(vertices), 2),
        tangents=torch.ones(1, len(vertices), 4),
        texture=torch.full((1, 2, 3, 3), 0.3),
        material={"doubleSided": True},
        unlit=True,
    )


@pytest.mark.parametrize("target", [0, 98, 99])
def test_disabled_and_satisfied_target_preserve_geometry(target: int) -> None:
    original = source()
    result = materialize_mesh(
        DecimateMesh.execute(mesh=original, target_face_count=target, placement_mode="midpoint")[
            "mesh"
        ]
    )
    torch.testing.assert_close(result.vertices, original.vertices, rtol=0, atol=0)
    torch.testing.assert_close(result.faces, original.faces, rtol=0, atol=0)
    torch.testing.assert_close(result.vertex_colors, original.vertex_colors, rtol=0, atol=0)
    assert result.face_counts.tolist() == [98]
    assert result.uvs is None and result.normals is None and result.tangents is None
    assert original.normals is not None and original.uvs is not None


@pytest.mark.parametrize("mode", ["midpoint", "qem"])
def test_reduction_preserves_colors_materials_and_input(mode: str) -> None:
    original = source()
    saved_vertices = original.vertices.clone()
    saved_faces = original.faces.clone()
    outputs = [
        materialize_mesh(
            DecimateMesh.execute(mesh=original, target_face_count=60, placement_mode=mode)["mesh"]
        )
        for _ in range(2)
    ]
    first, second = outputs
    assert first.face_counts.tolist() == [60]
    assert first.vertex_counts[0] < original.vertices.shape[1]
    assert torch.isfinite(first.vertices).all()
    assert first.faces.min() >= 0 and first.faces.max() < first.vertex_counts[0]
    torch.testing.assert_close(first.vertices, second.vertices, rtol=0, atol=0)
    torch.testing.assert_close(first.faces, second.faces, rtol=0, atol=0)
    expected_colors = torch.tensor([0.2, 0.7, 0.4, 0.9]).expand_as(first.vertex_colors)
    torch.testing.assert_close(first.vertex_colors, expected_colors)
    torch.testing.assert_close(first.texture, original.texture, rtol=0, atol=0)
    assert first.material == {"doubleSided": True} and first.unlit
    assert first.vertices.dtype == original.vertices.dtype
    assert first.faces.dtype == original.faces.dtype
    assert first.vertices.device == original.vertices.device
    assert first.uvs is None and first.normals is None and first.tangents is None
    torch.testing.assert_close(original.vertices, saved_vertices, rtol=0, atol=0)
    torch.testing.assert_close(original.faces, saved_faces, rtol=0, atol=0)
