"""Exact CPU utility parity, including execution with upstream imports forbidden."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import TriangleMeshBatch
from dinkster_inference_torch.image_crop import crop_images_to_masks
from dinkster_inference_torch.mesh import mesh_info, mesh_item_to_glb_bytes

vectors = importlib.import_module("tools.gen_crop_mesh_goldens")
GOLDEN = json.loads((Path(__file__).parent / "goldens/crop_mesh_25dfc16f.json").read_text())


@pytest.mark.parametrize("name,kwargs", vectors.crop_cases().items())
def test_crop_source_parity(name: str, kwargs: dict[str, Any]) -> None:
    expected = GOLDEN["crops"][name]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            crop_images_to_masks(**kwargs)
        return
    before = {
        key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)
    }
    result = crop_images_to_masks(**kwargs)
    assert result.device.type == "cpu"
    assert vectors.digest(result) == expected
    for key, value in before.items():
        assert torch.equal(kwargs[key], value)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        item
        for item in vectors.mesh_cases().items()
        if item[0] not in {"negative_count_slice", "oversized_count_and_uvs"}
    ],
)
def test_mesh_source_parity(name: str, kwargs: dict[str, Any]) -> None:
    mesh = TriangleMeshBatch[torch.Tensor](**kwargs)
    before = {
        key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)
    }
    expected = GOLDEN["meshes"][name]
    assert mesh_info(mesh) == expected["info"]
    for index, item in enumerate(expected["items"]):
        if "error" in item:
            with pytest.raises(ValueError, match=re.escape(item["error"])):
                mesh_item_to_glb_bytes(mesh, index)
        else:
            assert vectors.digest(mesh_item_to_glb_bytes(mesh, index)) == item
    for key, value in before.items():
        assert torch.equal(getattr(mesh, key), value)


def test_glb_layout_and_material_contract() -> None:
    import struct

    mesh = TriangleMeshBatch[torch.Tensor](**vectors.mesh_cases()["material"])
    glb = mesh_item_to_glb_bytes(mesh, 0)
    assert glb is not None
    assert struct.unpack_from("<4sII", glb) == (b"glTF", 2, len(glb))
    size, chunk = struct.unpack_from("<II", glb, 12)
    assert chunk == 0x4E4F534A and size % 4 == 0
    document = json.loads(glb[20 : 20 + size])
    binary_size, binary_chunk = struct.unpack_from("<II", glb, 20 + size)
    assert binary_chunk == 0x004E4942
    assert binary_size == document["buffers"][0]["byteLength"]
    assert binary_size % 4 == 0
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= binary_size
    material = document["materials"][0]
    assert (
        material["occlusionTexture"]["index"]
        == (material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"])
    )
    assert material["normalTexture"]["scale"] == 0.5
    assert material["emissiveFactor"] == [1.0, 0.2, 0.3]
    assert document["extensionsUsed"] == ["KHR_materials_emissive_strength"]


@pytest.mark.parametrize(
    "name,error",
    [
        ("negative_count_slice", "vertex count must be in [0, 4], got -1"),
        ("oversized_count_and_uvs", "vertex count must be in [0, 4], got 6"),
    ],
)
def test_glb_rejects_counts_outside_padded_storage(name: str, error: str) -> None:
    mesh = TriangleMeshBatch[torch.Tensor](**vectors.mesh_cases()[name])
    with pytest.raises(ValueError, match=re.escape(f"save_glb: {error}")):
        mesh_item_to_glb_bytes(mesh, 0)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("vertex_counts", torch.tensor([-1]), "vertex count must be in [0, 4], got -1"),
        ("vertex_counts", torch.tensor([5]), "vertex count must be in [0, 4], got 5"),
        ("face_counts", torch.tensor([-1]), "face count must be in [0, 2], got -1"),
        ("face_counts", torch.tensor([3]), "face count must be in [0, 2], got 3"),
        ("uvs", torch.zeros(1, 4, 1), "uvs must be (N, 2) with N=4"),
        ("vertex_colors", torch.zeros(1, 4, 2), "vertex_colors must be (N, 3 or 4) with N=4"),
        ("normals", torch.zeros(1, 4, 2), "normals must be (N, 3) with N=4"),
        ("tangents", torch.zeros(1, 4, 3), "tangents must be (N, 4) with N=4"),
    ],
)
def test_glb_rejects_invalid_counts_and_attribute_widths(
    field: str, value: torch.Tensor, error: str
) -> None:
    kwargs = vectors.mesh_cases()["bare"] | {
        "vertex_counts": torch.tensor([4]),
        "face_counts": torch.tensor([2]),
        field: value,
    }
    mesh = TriangleMeshBatch[torch.Tensor](**kwargs)
    with pytest.raises(ValueError, match=re.escape(f"save_glb: {error}")):
        mesh_item_to_glb_bytes(mesh, 0)


def test_cold_native_execution_without_upstream() -> None:
    script = """
import importlib.abc
import sys
class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'comfy', 'comfy_extras', 'comfy_api', 'nodes',
                                     'folder_paths', 'server'}:
            raise AssertionError('upstream import: ' + fullname)
sys.meta_path.insert(0, NoUpstream())
from dinkster_compat_comfy.native_arm import (
    GenerationImageCropToMask, GenerationGetMeshInfo, GenerationMeshToModel3D,
)
import torch
from dinkster_inference import TriangleMeshBatch
image = torch.ones(1, 8, 8, 4)
result = GenerationImageCropToMask.execute(images=image, masks=torch.ones(1, 8, 8),
                                          width=8, height=8)
assert result['images'].shape == (1, 8, 8, 3)
mesh = TriangleMeshBatch(vertices=torch.zeros(1, 3, 3), faces=torch.tensor([[[0, 1, 2]]]))
info = GenerationGetMeshInfo.execute(mesh=mesh)
assert info['mesh'] is mesh
assert info['info'] == 'Vertices:   3\\nFaces:      1\\nAttributes: none'
model = GenerationMeshToModel3D.execute(mesh=mesh)['model']
assert model['format'] == 'glb' and model['bytes'].startswith(b'glTF')
mesh.faces = torch.empty(1, 0, 3, dtype=torch.int64)
try:
    GenerationMeshToModel3D.execute(mesh=mesh)
except ValueError as error:
    assert str(error) == 'mesh is empty'
else:
    raise AssertionError('empty mesh accepted')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_export_uses_first_item_and_info_preserves_native_identity() -> None:
    from dinkster_compat_comfy.native_arm import GenerationGetMeshInfo, GenerationMeshToModel3D

    mesh = TriangleMeshBatch[torch.Tensor](**vectors.mesh_cases()["batch"])
    assert GenerationGetMeshInfo.execute(mesh=mesh)["mesh"] is mesh
    model: Any = GenerationMeshToModel3D.execute(mesh=mesh)["model"]
    assert model == {"format": "glb", "bytes": mesh_item_to_glb_bytes(mesh, 0)}
