"""Native mesh operation behavior without ComfyUI imports."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from dinkster_inference import (
    PBR_CHANNELS,
    DenseVoxelGrid,
    SparseSupport,
    SparseVolume,
    TriangleMeshBatch,
)
from dinkster_inference_torch.mesh_operations import (
    apply_texture_to_mesh,
    bake_ambient_occlusion,
    bake_normal_map_from_mesh,
    bake_texture_from_voxel,
    decimate_mesh,
    paint_mesh,
    remesh_mesh,
    render_uv_atlas,
    smooth_mesh_normals,
    unwrap_mesh,
    voxel_grid_to_mesh,
)


def asymmetric_mesh(*, uvs: bool = True) -> TriangleMeshBatch[torch.Tensor]:
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.4]]])
    faces = torch.tensor([[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]], dtype=torch.int64)
    uv_values = torch.tensor([[[0.0, 0.0], [0.9, 0.1], [0.2, 0.8], [0.7, 0.6]]])
    return TriangleMeshBatch(vertices=vertices, faces=faces, uvs=uv_values if uvs else None)


def sparse_colors() -> SparseVolume[torch.Tensor]:
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 3, 2, 1]], dtype=torch.int64)
    support = SparseSupport(
        coordinates=coordinates,
        batch_counts=(2,),
        resolution=4,
        origin=(-0.5, -0.5, -0.5),
        voxel_size=(0.25, 0.25, 0.25),
        support_id="sha256:" + "1" * 64,
    )
    features = torch.tensor([[1.0, 0.25, 0.0, 0.2, 0.7, 1.0], [0.0, 0.5, 1.0, 0.8, 0.1, 1.0]])
    return SparseVolume(support=support, features=features, channels=PBR_CHANNELS)


def test_voxel_to_mesh_uses_strict_threshold_and_axis_flip() -> None:
    values = torch.zeros(1, 1, 4, 4, 4)
    values[0, 0, 0, 1, 2] = 0.6
    values[0, 0, 1, 2, 3] = 0.61
    voxel = DenseVoxelGrid(values=values, channels=("density",), frame="y_up")
    result = voxel_grid_to_mesh(voxel, algorithm="basic", threshold=0.6)
    assert result.vertices.shape == (1, 24, 3)
    assert result.faces.shape == (1, 12, 3)
    assert torch.equal(result.vertices.amin(dim=1), torch.tensor([[0.5, 0.0, -0.5]]))
    assert torch.equal(result.vertices.amax(dim=1), torch.tensor([[1.0, 0.5, 0.0]]))


def test_decimate_mesh_reduces_asymmetric_tetrahedron() -> None:
    result = decimate_mesh(asymmetric_mesh(), target_face_count=2)
    assert result.face_counts is not None
    assert result.vertex_counts is not None
    assert int(result.face_counts[0]) <= 2
    assert int(result.vertex_counts[0]) < 4
    assert result.uvs is None


def test_remesh_mesh_preserves_asymmetric_bounds() -> None:
    result = remesh_mesh(
        asymmetric_mesh(uvs=False),
        resolution=8,
        sign_mode="sdf",
        qef=True,
        drop_small_components=0.0,
    )
    assert result.vertex_counts is not None
    count = int(result.vertex_counts[0])
    assert count > 4
    vertices = result.vertices[0, :count]
    assert vertices[:, 0].max() > vertices[:, 2].max()
    assert result.faces.dtype == torch.int32


def test_remesh_mesh_polls_cancellation_before_geometry_work() -> None:
    with pytest.raises(RuntimeError, match="remesh_mesh: cancelled"):
        remesh_mesh(asymmetric_mesh(uvs=False), resolution=8, cancelled=lambda: True)


def test_smooth_mesh_normals_uses_area_weighted_asymmetric_geometry() -> None:
    result = smooth_mesh_normals(asymmetric_mesh(), crease_angle=180.0)
    assert result.normals is not None
    assert torch.allclose(result.normals.norm(dim=-1), torch.ones(1, 4), atol=1e-6)
    assert not torch.allclose(result.normals[0, 0], result.normals[0, 1])
    assert result.uvs is not None


def test_unwrap_mesh_duplicates_seams_and_keeps_uvs_in_unit_square() -> None:
    result = unwrap_mesh(asymmetric_mesh(uvs=False), segmenter="adaptive", resolution=32)
    assert result.uvs is not None
    count = result.vertices.shape[1]
    assert count >= 4
    assert torch.all((result.uvs[0, :count] >= 0.0) & (result.uvs[0, :count] <= 1.0))
    ranges = result.uvs[0, :count].amax(dim=0) - result.uvs[0, :count].amin(dim=0)
    assert ranges[0] != ranges[1]


def test_paint_mesh_uses_nearest_voxel_and_linearizes_rgb() -> None:
    mesh = asymmetric_mesh(uvs=False)
    result = paint_mesh(mesh, sparse_colors())
    assert result.vertex_colors is not None
    expected_first = torch.tensor([0.0, 0.5**2.2, 1.0], device=result.vertex_colors.device)
    assert torch.allclose(result.vertex_colors[0, 0], expected_first, atol=1e-6)
    assert result.vertex_colors.shape[-1] == 3


def test_bake_texture_from_voxel_separates_asymmetric_pbr_channels() -> None:
    base, metallic, roughness = bake_texture_from_voxel(
        asymmetric_mesh(), sparse_colors(), texture_size=8
    )
    assert base.shape == metallic.shape == roughness.shape == (1, 8, 8, 3)
    covered = base.sum(dim=-1) > 0
    assert bool(covered.any())
    assert not torch.equal(metallic[covered], roughness[covered])


def test_bake_normal_map_from_mesh_has_flat_and_detailed_texels() -> None:
    low = asymmetric_mesh()
    high = asymmetric_mesh()
    high.vertices[0, 3, 2] += 0.15
    result = bake_normal_map_from_mesh(low, high, resolution=8, cage_distance=0.5)
    assert result.shape == (1, 8, 8, 3)
    flat = torch.tensor([0.5, 0.5, 1.0])
    assert bool(torch.any(torch.linalg.vector_norm(result[0] - flat, dim=-1) > 1e-3))


def test_bake_ambient_occlusion_returns_bounded_grayscale() -> None:
    result = bake_ambient_occlusion(
        asymmetric_mesh(), asymmetric_mesh(), resolution=8, samples=4, max_distance=0.7
    )
    assert result.shape == (1, 8, 8, 3)
    assert torch.all((result >= 0.0) & (result <= 1.0))
    assert torch.equal(result[..., 0], result[..., 1])
    assert torch.equal(result[..., 1], result[..., 2])


def test_render_uv_atlas_draws_each_asymmetric_batch_item() -> None:
    first = asymmetric_mesh()
    second = asymmetric_mesh()
    assert first.uvs is not None
    assert second.uvs is not None
    second.uvs[0, 3] = torch.tensor([0.1, 0.2])
    mesh = TriangleMeshBatch(
        vertices=torch.cat([first.vertices, second.vertices]),
        faces=torch.cat([first.faces, second.faces]),
        uvs=torch.cat([first.uvs, second.uvs]),
    )
    result = render_uv_atlas(mesh, resolution=16)
    assert result.shape == (2, 16, 16, 3)
    background = torch.tensor([0.13, 0.13, 0.13])
    assert torch.any(torch.all(result == background, dim=-1))
    assert torch.any(torch.any(result != background, dim=-1))
    assert not torch.equal(result[0], result[1])


def test_apply_texture_normalizes_uvs_and_packs_orm_channels() -> None:
    mesh = asymmetric_mesh()
    assert mesh.uvs is not None
    mesh.uvs = mesh.uvs * torch.tensor([2.0, 3.0]) + torch.tensor([-0.5, 0.2])
    base = torch.tensor([[[[-1.0, 0.25, 2.0]]]])
    metallic = torch.full((1, 2, 1, 3), 0.2)
    roughness = torch.full((1, 1, 2, 3), 0.7)
    result = apply_texture_to_mesh(mesh, base, metallic=metallic, roughness=roughness)
    assert result.texture is not None
    assert torch.equal(result.texture, torch.tensor([[[[0.0, 0.25, 1.0]]]]))
    assert result.metallic_roughness is not None
    assert result.metallic_roughness.shape == (1, 2, 2, 3)
    assert torch.allclose(result.metallic_roughness[0, 0, 0], torch.tensor([1.0, 0.7, 0.2]))
    assert result.uvs is not None
    assert torch.equal(result.uvs.amin(dim=1), torch.zeros(1, 2))
    assert torch.allclose(result.uvs.amax(dim=1), torch.tensor([[0.75, 1.0]]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vertex_counts", torch.tensor([5]), "vertex count 5"),
        ("face_counts", torch.tensor([5]), "face count 5"),
        ("faces", torch.tensor([[[0, 1, 7]]]), "face indices"),
        ("uvs", torch.zeros(1, 4, 3), "uvs must have shape"),
        ("vertex_colors", torch.zeros(1, 4, 2), "vertex_colors must have shape"),
        ("normals", torch.zeros(1, 4, 4), "normals must have shape"),
        ("tangents", torch.zeros(1, 4, 3), "tangents must have shape"),
    ],
)
def test_mesh_validation_fails_before_operation(
    field: str, value: torch.Tensor, message: str
) -> None:
    mesh = asymmetric_mesh()
    if field == "vertex_counts":
        mesh.vertex_counts = value
        mesh.face_counts = torch.tensor([4])
    elif field == "face_counts":
        mesh.vertex_counts = torch.tensor([4])
        mesh.face_counts = value
    else:
        setattr(mesh, field, value)
    with pytest.raises(ValueError, match=message):
        smooth_mesh_normals(mesh)


def test_module_imports_and_executes_with_comfyui_imports_blocked() -> None:
    script = r"""
import importlib.abc
import sys

class BlockComfy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'comfy', 'comfy_extras', 'comfy_api', 'folder_paths', 'nodes'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('blocked import: ' + fullname)

sys.meta_path.insert(0, BlockComfy())
import torch
from dinkster_inference import TriangleMeshBatch
from dinkster_inference_torch.mesh_operations import apply_texture_to_mesh
mesh = TriangleMeshBatch(
    vertices=torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]),
    faces=torch.tensor([[[0, 1, 2]]]),
    uvs=torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]),
)
result = apply_texture_to_mesh(mesh, torch.ones(1, 1, 1, 3))
assert result.texture.shape == (1, 1, 1, 3)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
