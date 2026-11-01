"""Geometry transport preserves complete runtime fields without importing Torch."""

from __future__ import annotations

import struct
import subprocess
import sys
from dataclasses import fields

import pytest
from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    TriangleMeshBatch,
    TypeRegistry,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor, encode_latent


@pytest.mark.parametrize("type_id", ("DA3_MODEL", "FACE_DETECTION_MODEL", "SAM3D_BODY_MODEL"))
def test_family_models_retain_their_complete_holding_process_object(type_id: str) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(type_id)
    wrapper = {"model": object(), "patchers": {"short": object(), "full": object()}}
    assert spec.decode(spec.encode(wrapper)) is wrapper
    register_inference_types(registry)
    assert registry.spec(type_id) is spec

    authority = TypeRegistry()
    previous = authority.register(type_id, encode=lambda _: b"first", decode=lambda _: wrapper)
    register_inference_types(authority)
    assert authority.spec(type_id) is previous
    assert authority.spec(type_id).encode(wrapper) == b"first"


def tensor(shape: tuple[int, ...], dtype: str = "float32") -> EncodedLatentTensor:
    count = 1
    for size in shape:
        count *= size
    fmt = "q" if dtype == "int64" else "f"
    return EncodedLatentTensor(dtype, shape, struct.pack("<" + fmt * count, *range(count)))


def test_mesh_roundtrip_preserves_all_fields_and_material_shapes() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    register_inference_types(registry)
    assert registry.spec(MESH_TYPE) is spec
    mesh = TriangleMeshBatch(
        tensor((2, 4, 3)),
        tensor((2, 3, 3), "int64"),
        uvs=tensor((2, 4, 2)),
        vertex_colors=tensor((2, 4, 4)),
        texture=tensor((2, 3, 5, 3)),
        metallic_roughness=tensor((2, 3, 5, 3)),
        vertex_counts=tensor((2,), "int64"),
        face_counts=tensor((2,), "int64"),
        unlit=True,
        normals=tensor((2, 4, 3)),
        tangents=tensor((2, 4, 4)),
        normal_map=tensor((2, 3, 5, 3)),
        occlusion_in_mr=True,
        material={
            "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
            "metallicFactor": 0.7,
            "roughnessFactor": 0.3,
            "doubleSided": True,
            "extras": {"tuple": (None, "OPAQUE"), "tensor": tensor((2,))},
        },
        emissive=tensor((2, 3, 5, 3)),
    )
    encoded = spec.encode(mesh)
    restored = spec.decode(encoded)
    assert type(restored) is TriangleMeshBatch
    for field in fields(mesh):
        assert getattr(restored, field.name) == getattr(mesh, field.name), field.name
    assert spec.encode(restored) == encoded
    assert spec.fingerprint is not None
    assert spec.fingerprint(mesh) == spec.fingerprint(restored)
    mesh.material = {"unsupported": object()}
    with pytest.raises(TypeError, match="unsupported"):
        spec.encode(mesh)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (0, 3), (5, 3)])
def test_voxel_roundtrip_accepts_dense_sparse_and_empty_payloads(shape: tuple[int, ...]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(VOXEL_TYPE)
    value = {
        "data": tensor(shape),
        "voxel_colors": tensor((5, 3)),
        "resolution": (32, 64, 128),
        "source_metadata": {"layout": "source"},
    }
    restored = spec.decode(spec.encode(value))
    assert restored == value
    plain = {"data": tensor(shape), "voxel_colors": None, "resolution": None}
    restored = spec.decode(spec.encode(plain))
    assert restored == plain


def test_geometry_transport_is_cold_and_rejects_invalid_mesh_layouts() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import MESH_TYPE, VOXEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor
from dinkster_api.v1 import TriangleMeshBatch
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_workers import load_manifest
runtimes = {node.schema().node_type: node.schema() for node in COMFY_NODES}
names = {'RotateMesh', 'GetMeshInfo', 'VAEDecodeHunyuan3D', 'VoxelToMeshBasic', 'VoxelToMesh'}
assert names <= runtimes.keys()
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert names <= set(manifest.namespaces)
assert runtimes['VoxelToMeshBasic'].deprecation is not None
assert runtimes['VoxelToMeshBasic'].search_visibility == 'deprecated'
r = TypeRegistry()
register_inference_types(r)
m = TriangleMeshBatch(EncodedLatentTensor('float32', (1, 0, 3), b''),
                      EncodedLatentTensor('int64', (1, 0, 3), b''))
s = r.spec(MESH_TYPE)
assert type(s.decode(s.encode(m))) is TriangleMeshBatch
s = r.spec(VOXEL_TYPE)
v = {'data': m.vertices, 'voxel_colors': None, 'resolution': None}
assert s.decode(s.encode(v)) == v
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    registry = TypeRegistry()
    register_inference_types(registry)
    mesh = TriangleMeshBatch(tensor((1, 2, 3)), tensor((1, 1, 3), "int64"))
    values = {field.name: getattr(mesh, field.name) for field in fields(mesh)}
    values["vertex_counts"] = tensor((1,), "int64")
    spec = registry.spec(MESH_TYPE)
    with pytest.raises(ValueError, match="provided together"):
        spec.decode(encode_latent({"samples": values}))
    values["vertex_counts"] = None
    values["vertices"] = tensor((2, 3))
    with pytest.raises(ValueError, match="batch, vertices, 3"):
        spec.decode(encode_latent({"samples": values}))
    values["vertices"] = tensor((1, 2, 3))
    values["unlit"] = 1
    with pytest.raises(ValueError, match="booleans"):
        spec.decode(encode_latent({"samples": values}))
