"""Geometry transport preserves complete runtime fields without importing Torch."""

from __future__ import annotations

import asyncio
import os
import struct
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import EncodedLatentTensor, encode_latent, register_core_types

from dinkster.compose import PackSpec, ServingComposer


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


@pytest.mark.parametrize(
    "type_id", ("DA3_GEOMETRY", "FACE_LANDMARKS", "MHR_POSE_DATA", "KIMODO_POSE_DATA")
)
def test_vision_packets_preserve_native_containers_and_optional_fields(type_id: str) -> None:
    import numpy as np
    from dinkster_api.v1 import initialize_inference_process, materialize_vision_packet

    initialize_inference_process()
    import torch

    array = np.array([[2.25, -5.5, 11.0], [17.5, 23.0, -31.0]], dtype=np.float32)[:, ::2]
    live = torch.tensor([[True, False], [False, True]])
    face = {
        "bbox_xyxy": np.array([-2, 3, 71, 97], dtype=np.float32),
        "blendshapes": {"smile": 0.125, "frown": 0.75},
        "landmarks_xy": array,
        "landmarks_3d": array.copy(),
        "transformation_matrix": np.arange(16, dtype=np.float64).reshape(4, 4),
        "presence": np.float32(0.375),
        "score": 0.625,
    }
    if type_id == "DA3_GEOMETRY":
        packet: Any = {
            "depth": torch.arange(12, dtype=torch.float64).reshape(2, 2, 3),
            "image": torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3),
            "mode": "multiview",
            "intrinsics": torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
            "extrinsics": torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4),
            "sky": None,
        }
    elif type_id == "FACE_LANDMARKS":
        packet = {
            "frames": [[face], [], [dict(face, score=0.25), face]],
            "image_size": (113, 197),
            "connection_sets": {"outline": frozenset({(9, 2), (3, 7)}), "empty": frozenset()},
        }
    else:
        person: Any = {
            key: array.copy()
            for key in (
                "pred_vertices",
                "pred_keypoints_3d",
                "pred_keypoints_2d",
                "pred_cam_t",
                "pred_pose_raw",
                "global_rot",
                "body_pose_params",
                "hand_pose_params",
                "scale_params",
                "shape_params",
                "expr_params",
                "pred_joint_coords",
                "pred_global_rots",
                "mhr_model_params",
                "lhand_bbox",
                "rhand_bbox",
            )
        }
        person.update(
            mask=live, focal_length=np.array(123.5, dtype=np.float32), pred_face_keypoints_3d=None
        )
        packet = {
            "frames": [[person], [], [dict(person, mask=None)]],
            "image_size": (113, 197),
            "faces": np.array([[65537, 2, 4294967295]], dtype=np.uint32),
            "canonical_colors": {"positions": array, "head_mask": np.array([True, False])},
            "hand_vert_mask": np.array([False, True]),
        }
        if type_id == "KIMODO_POSE_DATA":
            packet["_skeleton_override"] = {
                "parents": np.array([-1, 0], dtype=np.int32),
                "bind_global_m": np.arange(16, dtype=np.float32).reshape(2, 8),
                "lbs_compact_joints": np.array([[65535, 32768]], dtype=np.uint16),
                "lbs_compact_weights": np.array([[0.125, 0.875]], dtype=np.float32),
                "lbs_compact_max_inf": 2,
                "rest_verts_m": array,
                "faces": packet["faces"],
                "per_frame_y_down": False,
                "openpose18_joint_indices": np.array([[1, -1], [0, 1]], dtype=np.int32),
                "openpose18_joint_weights": np.array([0.25, 0.75], dtype=np.float32),
            }
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(type_id)
    register_inference_types(registry)
    assert registry.spec(type_id) is spec
    encoded = spec.encode(packet)
    cold = spec.decode(encoded)
    assert spec.encode(cold) == encoded
    restored = materialize_vision_packet(cold)

    def equal(actual: Any, expected: Any) -> None:
        assert type(actual) is type(expected)
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
        elif isinstance(expected, torch.Tensor):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert torch.equal(actual, expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                equal(left, right)
        else:
            assert actual == expected

    equal(restored, packet)
    assert materialize_vision_packet(packet) is packet
    assert spec.fingerprint is not None
    assert spec.fingerprint(restored) == spec.fingerprint(packet)
    assert spec.fingerprint({**packet, "extra": ("mapping", {"$": "tensor"})}) != spec.fingerprint(
        packet
    )
    authority = TypeRegistry()
    first = authority.register(type_id, encode=lambda _: b"authority", decode=lambda _: packet)
    register_inference_types(authority)
    assert authority.spec(type_id) is first


def test_vision_packet_cold_boundary_refuses_pickle_and_bad_records() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['numpy'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor, encode_latent
r = TypeRegistry()
register_inference_types(r)
face = ('mapping', {'frames': ('list', []), 'image_size': ('tuple', [113,197]),
                    'connection_sets': ('mapping', {'edge': ('topology', [(9,2)])}),
                    'array': ('numpy', EncodedLatentTensor(
                        'uint16', (2,), b'\\xff\\xff\\x00\\x80'))})
s = r.spec('FACE_LANDMARKS')
wire = encode_latent({'samples':face})
decoded = s.decode(wire)
assert decoded['connection_sets']['edge'] == frozenset({(9,2)})
assert s.encode(decoded) == wire
for name in ('torch', 'numpy', 'dinkster_inference'):
    assert sys.modules[name] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    registry = TypeRegistry()
    register_inference_types(registry)
    for type_id in ("DA3_GEOMETRY", "FACE_LANDMARKS", "MHR_POSE_DATA", "KIMODO_POSE_DATA"):
        spec = registry.spec(type_id)
        for bad in (
            b"cos\nsystem\n(S'false'\ntR.",
            encode_latent({"samples": ("numpy", "not a tensor")}),
            encode_latent({"samples": ("topology", [(True, 2)])}),
            encode_latent({"samples": ("topology", [(1, 2), (1, 2)])}),
            encode_latent({"samples": ("mapping", {})}),
        ):
            with pytest.raises((TypeError, ValueError)):
                spec.decode(bad)


class VisionPacketProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision-source.make",
            outputs=(OutputSpec("packet", TypeExpr.concrete("KIMODO_POSE_DATA")),),
        )

    @classmethod
    def execute(cls) -> Any:
        import numpy as np
        from dinkster_api.v1 import initialize_inference_process

        initialize_inference_process()
        import torch

        return cls.outputs(
            packet={
                "frames": [
                    [
                        {
                            "mask": torch.tensor([[False, True]]),
                            "pred_cam_t": np.array([-7.25, 3.5, 17.0], dtype=np.float32),
                        }
                    ],
                    [],
                ],
                "image_size": (113, 197),
                "faces": np.array([[2, 65537, 9]], dtype=np.uint32),
                "_skeleton_override": {
                    "lbs_compact_joints": np.array([[65535, 32768]], dtype=np.uint16)
                },
                "topology": frozenset({(9, 2), (3, 7)}),
            }
        )


class VisionPacketConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision-consumer.read",
            inputs=(InputSpec("packet", TypeExpr.concrete("KIMODO_POSE_DATA")),),
            outputs=(OutputSpec("packet", TypeExpr.concrete("KIMODO_POSE_DATA")),),
        )

    @classmethod
    def execute(cls, *, packet: object) -> Any:
        import numpy as np
        from dinkster_api.v1 import materialize_vision_packet

        value = materialize_vision_packet(packet)
        assert value["image_size"] == (113, 197)
        assert value["frames"][1] == []
        assert value["frames"][0][0]["mask"].tolist() == [[False, True]]
        camera = value["frames"][0][0]["pred_cam_t"]
        assert type(camera) is np.ndarray and camera.tolist() == [-7.25, 3.5, 17.0]
        assert value["faces"].dtype == np.uint32
        assert value["faces"].tolist() == [[2, 65537, 9]]
        assert value["_skeleton_override"]["lbs_compact_joints"].tolist() == [[65535, 32768]]
        assert value["topology"] == frozenset({(9, 2), (3, 7)})
        return cls.outputs(packet=value)


def test_vision_packet_crosses_workers_and_persistent_cache(tmp_path: Path) -> None:
    manifests: list[Path] = []
    for name, node in (("source", "VisionPacketProducer"), ("consumer", "VisionPacketConsumer")):
        module = f"vision_packet_{name}"
        (tmp_path / f"{module}.py").write_text(
            f"from test_geometry_wire import {node}, register_inference_types\nNODES=({node},)\n",
            encoding="ascii",
        )
        manifest = tmp_path / f"{name}.toml"
        manifest.write_text(
            f'[pack]\nname="vision-{name}"\npublisher="dinkster"\n'
            f'namespaces=["vision-{name}"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests.append(manifest)

    async def scenario() -> None:
        graph = Graph(
            {
                "source": GraphNode("vision-source.make"),
                "consumer": GraphNode("vision-consumer.read", {"packet": Link("source", "packet")}),
            }
        )
        for replay in (False, True):
            registry = TypeRegistry()
            register_core_types(registry)
            register_inference_types(registry)
            composer = ServingComposer(
                registry=registry,
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                pack_scratch_root=tmp_path / "scratch",
                worker_env={
                    "DINKSTER_ACCELERATOR": "cpu",
                    "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                },
            )
            try:
                for manifest in manifests:
                    await composer.add_pack(
                        PackSpec(
                            manifest=manifest,
                            in_process=False,
                            worker_group=manifest.stem,
                            group_manifests=(manifest,),
                        )
                    )
                result = await composer.composition.make_engine(lambda _: None).run(
                    graph, ["consumer"]
                )
                assert set(result.cached if replay else result.executed) == {"source", "consumer"}
                packet = result.outputs["consumer"]["packet"].resolve()
                VisionPacketConsumer.execute(packet=packet)
            finally:
                await composer.close()

    asyncio.run(scenario())
