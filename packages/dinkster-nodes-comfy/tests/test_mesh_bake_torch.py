"""Observable ambient occlusion, controls and counted mesh batches."""

from __future__ import annotations

import pytest
import torch
from dinkster_api.v1 import TriangleMeshBatch
from dinkster_nodes_comfy.mesh_bake import BakeAmbientOcclusion


def plane(z: float = 0.0, extent: float = 1.0) -> TriangleMeshBatch:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [
                [
                    [-extent, -extent, z],
                    [extent, -extent, z],
                    [extent, extent, z],
                    [-extent, extent, z],
                ]
            ]
        ),
        faces=torch.tensor([[[0, 1, 2], [0, 2, 3]]]),
        uvs=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]),
    )


def bake(low: TriangleMeshBatch, high: TriangleMeshBatch, **overrides: object) -> torch.Tensor:
    parameters = dict(resolution=64, samples=16, max_distance=0.5, strength=1.0, bias=0.001)
    parameters.update(overrides)
    torch.manual_seed(749)
    return BakeAmbientOcclusion.execute(low_poly=low, high_poly=high, **parameters)["IMAGE"]


def test_ambient_occlusion_distance_strength_and_gutters() -> None:
    low = plane()
    high = plane(0.25, 4.0)
    # Keep the original surface in the high mesh so projection cannot jump to the roof.
    high.vertices = torch.cat([low.vertices, high.vertices], dim=1)
    high.faces = torch.cat([low.faces, high.faces + 4], dim=1)
    high.uvs = None
    open_surface = bake(low, low)
    torch.testing.assert_close(open_surface, torch.ones_like(open_surface), rtol=0, atol=0)
    occluded = bake(low, high)
    assert occluded.mean() < 0.15
    short = bake(low, high, max_distance=0.01)
    torch.testing.assert_close(short, torch.ones_like(short), rtol=0, atol=0)
    half = bake(low, high, strength=0.5)
    torch.testing.assert_close(half, (occluded + 1.0) / 2.0, rtol=0, atol=0)
    low.uvs = low.uvs * 0.4 + 0.3
    gutters = bake(low, high)
    assert gutters[0, 0, 0].mean() < 0.15
    assert gutters.shape == (1, 64, 64, 3) and gutters.device.type == "cpu"


def test_ambient_occlusion_ragged_empty_and_shared_high_mesh() -> None:
    low = plane()
    low.vertices = low.vertices.repeat(2, 1, 1)
    low.faces = low.faces.repeat(2, 1, 1)
    low.uvs = low.uvs.repeat(2, 1, 1)
    low.vertex_counts = torch.tensor([4, 3])
    low.face_counts = torch.tensor([2, 0])
    result = bake(low, plane())
    assert result.shape == (2, 64, 64, 3)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    low.uvs = None
    with pytest.raises(ValueError, match="low_poly has no UVs"):
        bake(low, plane())
