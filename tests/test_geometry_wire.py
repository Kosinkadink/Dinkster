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
    Voxel,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor, encode_latent


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
    value = Voxel(tensor(shape), voxel_colors=tensor((5, 3)), resolution=(32, 64, 128))
    restored = spec.decode(spec.encode(value))
    assert type(restored) is Voxel
    assert restored.data == value.data
    assert restored.voxel_colors == value.voxel_colors
    assert restored.resolution == (32, 64, 128)
    plain = Voxel(tensor(shape))
    restored = spec.decode(spec.encode(plain))
    assert isinstance(restored, Voxel)
    assert restored.voxel_colors is None and restored.resolution is None


def test_geometry_transport_is_cold_and_rejects_invalid_mesh_layouts() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import MESH_TYPE, VOXEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor
from dinkster_api.v1 import TriangleMeshBatch, Voxel
r = TypeRegistry()
register_inference_types(r)
m = TriangleMeshBatch(EncodedLatentTensor('float32', (1, 0, 3), b''),
                      EncodedLatentTensor('int64', (1, 0, 3), b''))
for t, v in ((MESH_TYPE, m), (VOXEL_TYPE, Voxel(m.vertices))):
    s = r.spec(t)
    assert type(s.decode(s.encode(v))) is type(v)
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
