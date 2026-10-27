"""CPU geometry consumers preserve complete portable mesh and voxel values."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    TriangleMeshBatch,
    TypeRegistry,
    Voxel,
    materialize_mesh,
    materialize_voxel,
    register_inference_types,
)
from dinkster_nodes_comfy.mesh import GetMeshInfo, RotateMesh


def mesh() -> TriangleMeshBatch[Any]:
    return TriangleMeshBatch(
        torch.tensor([[[1.0, 2.0, 3.0], [-2.0, 5.0, 7.0]]], dtype=torch.float64),
        torch.tensor([[[0, 1, 0]]]),
        normals=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        tangents=torch.tensor([[[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=torch.float64),
        texture=torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        material={"roughnessFactor": 0.3, "baseColorFactor": [0.2, 0.4, 0.6, 1.0]},
        unlit=True,
        occlusion_in_mr=True,
    )


@pytest.mark.parametrize("portable", [False, True])
def test_rotation_preserves_materials_normals_tangent_handedness_and_source(portable: bool) -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(source)) if portable else source
    result = RotateMesh.execute(
        mesh=value,
        mode="euler_xyz",
        **{"mode.angle_x": 0.0, "mode.angle_y": 0.0, "mode.angle_z": 90.0},
    )["mesh"]
    assert isinstance(result, TriangleMeshBatch)
    expected = torch.tensor([[[-2.0, 1.0, 3.0], [-5.0, -2.0, 7.0]]], dtype=torch.float64)
    torch.testing.assert_close(result.vertices, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        result.normals,
        torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(result.tangents[..., 3], source.tangents[..., 3], rtol=0, atol=0)
    torch.testing.assert_close(source.vertices, mesh().vertices, rtol=0, atol=0)
    assert result.material == source.material
    assert result.unlit and result.occlusion_in_mr
    torch.testing.assert_close(result.texture, source.texture, rtol=0, atol=0)
    quaternion = RotateMesh.execute(
        mesh=value,
        mode="quaternion",
        **{"mode.qw": 3.0, "mode.qx": 0.0, "mode.qy": 0.0, "mode.qz": 3.0},
    )["mesh"]
    assert isinstance(quaternion, TriangleMeshBatch)
    torch.testing.assert_close(quaternion.vertices, expected, rtol=0, atol=4e-15)
    assert materialize_mesh(source) is source
    assert (
        RotateMesh.execute(
            mesh=source,
            mode="euler_xyz",
            **dict.fromkeys(("mode.angle_x", "mode.angle_y", "mode.angle_z"), 0.0),
        )["mesh"]
        is source
    )
    with pytest.raises(ValueError, match="zero magnitude"):
        RotateMesh.execute(
            mesh=source,
            mode="quaternion",
            **dict.fromkeys(("mode.qw", "mode.qx", "mode.qy", "mode.qz"), 0.0),
        )


def test_materialization_retains_every_mesh_field_and_voxel_layout() -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    restored = materialize_mesh(spec.decode(spec.encode(source)))
    for field in fields(source):
        left, right = getattr(restored, field.name), getattr(source, field.name)
        if isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    voxel = Voxel(
        torch.tensor([[1, 3, 7], [9, 5, 2]], dtype=torch.int32),
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        resolution=64,
    )
    assert materialize_voxel(voxel) is voxel
    spec = registry.spec(VOXEL_TYPE)
    restored_voxel = materialize_voxel(spec.decode(spec.encode(voxel)))
    torch.testing.assert_close(restored_voxel.data, voxel.data, rtol=0, atol=0)
    torch.testing.assert_close(restored_voxel.voxel_colors, voxel.voxel_colors, rtol=0, atol=0)
    assert restored_voxel.resolution == 64


def test_mesh_info_uses_real_counts_not_padding() -> None:
    value = TriangleMeshBatch(
        torch.zeros(2, 10000, 3),
        torch.zeros(2, 9, 3, dtype=torch.int64),
        vertex_counts=torch.tensor([10000, 2]),
        face_counts=torch.tensor([9, 1]),
        texture=torch.zeros(2, 3, 5, 3),
    )
    result = GetMeshInfo.execute(mesh=value)
    assert result["MESH"] is value
    assert result["STRING"] == (
        "Batch:      2 meshes\nVertices:   10,002 (10.0K) total\nFaces:      10 total\n"
        "  [0]      10,000 verts  \u00b7           9 faces\n"
        "  [1]           2 verts  \u00b7           1 faces\nAttributes: texture 3\u00d75"
    )
    assert math.isfinite(value.vertices.sum().item())
