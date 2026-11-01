"""Geometry transport preserves complete runtime fields without importing Torch."""

from __future__ import annotations

import asyncio
import os
import struct
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dinkster_api.v1 import (
    MESH_TYPE,
    SHAPE_SUBDIVIDES_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    materialize_shape_subdivides,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import EncodedLatentTensor, encode_latent, register_core_types
from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

from dinkster.compose import PackSpec, ServingComposer


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
from dinkster_api.v1 import (
    MESH_TYPE, VOXEL_TYPE, SHAPE_SUBDIVIDES_TYPE, TypeRegistry, register_inference_types,
)
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
s = r.spec(SHAPE_SUBDIVIDES_TYPE)
stages = [{'feats': EncodedLatentTensor('float32', (0, 2), b''),
           'coords': EncodedLatentTensor('int64', (0, 4), b''),
           'shape': (3, 2), 'scale': ((1, 3), (2, 7), (9, 11))}]
assert s.decode(s.encode(stages)) == stages
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


def test_shape_subdivides_roundtrip_and_materialization_preserve_sparse_state() -> None:
    import torch

    initialize_inference_process()
    from dinkster_inference.ldm.trellis2.vae import SparseTensor

    stages = [
        SparseTensor(
            feats=torch.tensor([[1.25, -2.5], [7.0, 0.125]], dtype=torch.float16),
            coords=torch.tensor([[0, 4, 3, 2], [1, 7, 8, 9]], dtype=torch.int32),
            shape=torch.Size((5, 2)),
            scale=(Fraction(1, 3), Fraction(5, 7), Fraction(9, 11)),
            spatial_cache={"not_serializable": object()},
        ),
        SparseTensor(
            feats=torch.empty((0, 3), dtype=torch.bool),
            coords=torch.empty((0, 4), dtype=torch.int64),
            shape=torch.Size((2, 3)),
            scale=(Fraction(2, 5), Fraction(3, 8), Fraction(4, 9)),
        ),
    ]
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SHAPE_SUBDIVIDES_TYPE)
    register_inference_types(registry)
    assert registry.spec(SHAPE_SUBDIVIDES_TYPE) is spec
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(registry.wrap(SHAPE_SUBDIVIDES_TYPE, stages), blobs, [])
    restored, _ = codec.decode(wire, blobs, [])
    decoded = restored.resolve()
    assert spec.encode(decoded) == spec.encode(stages)
    assert spec.fingerprint is not None
    assert spec.fingerprint(decoded) == spec.fingerprint(stages)
    assert spec.fingerprint(stages[::-1]) != spec.fingerprint(stages)
    for source in (stages, decoded):
        materialized = materialize_shape_subdivides(source)
        assert len(materialized) == 2
        for expected, actual in zip(stages, materialized, strict=True):
            assert type(actual) is SparseTensor
            assert actual.shape == expected.shape
            assert type(actual.shape) is torch.Size
            assert actual._scale == expected._scale
            assert all(type(component) is Fraction for component in actual._scale)
            assert actual._spatial_cache == {}
            assert torch.equal(actual.feats, expected.feats)
            assert torch.equal(actual.coords, expected.coords)
            assert actual.feats.dtype == expected.feats.dtype
            assert actual.coords.dtype == expected.coords.dtype
            if source is stages:
                assert actual.feats is expected.feats
                assert actual.coords is expected.coords
    prior = TypeRegistry()
    authority = prior.register(SHAPE_SUBDIVIDES_TYPE)
    register_inference_types(prior)
    assert prior.spec(SHAPE_SUBDIVIDES_TYPE) is authority


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": None},
        {"shape": (1, 7)},
        {"shape": (True, 2)},
        {"scale": ((1, 0), (1, 1), (1, 1))},
        {"scale": (0.5, 1, 1)},
        {"coords": tensor((1, 3), "int64")},
        {"coords": tensor((1, 4))},
        {"feats": tensor((2,))},
    ],
)
def test_shape_subdivides_rejects_malformed_structural_stages(changes: dict[str, Any]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    stage = {
        "feats": tensor((1, 2)),
        "coords": tensor((1, 4), "int64"),
        "shape": (3, 2),
        "scale": ((1, 3), (5, 7), (9, 11)),
    }
    spec = registry.spec(SHAPE_SUBDIVIDES_TYPE)
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError, match="SHAPE_SUBDIVIDES"):
        spec.validate_encoded(encode_latent({"samples": [{**stage, **changes}]}), {})


def test_shape_subdivides_refuses_pickle_before_reconstruction(tmp_path: Path) -> None:
    import pickle

    marker = tmp_path / "marker.txt"
    marker.write_text("unchanged")

    class Payload:
        def __reduce__(self) -> Any:
            return marker.write_text, ("modified",)

    registry = TypeRegistry()
    register_inference_types(registry)
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    value = [
        SimpleNamespace(
            feats=tensor((0, 2)), coords=tensor((0, 4), "int64"), shape=(1, 2), _scale=(1, 1, 1)
        )
    ]
    wire, _ = codec.encode(registry.wrap(SHAPE_SUBDIVIDES_TYPE, value), blobs, [])
    payload = wire["payload"]
    assert isinstance(payload, dict)
    blobs[payload["blob"]] = b"pickle:" + pickle.dumps(Payload())
    with pytest.raises(EncodedPayloadValidationError, match="framing"):
        codec.decode(wire, blobs, [])
    assert marker.read_text() == "unchanged"


class ShapeSubdividesProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="shape-source.make",
            outputs=(OutputSpec("stages", TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            stages=[
                SimpleNamespace(
                    feats=tensor((1, 2)),
                    coords=tensor((1, 4), "int64"),
                    shape=(3, 2),
                    _scale=(Fraction(1, 3), Fraction(5, 7), Fraction(9, 11)),
                    _spatial_cache={"not_serializable": object()},
                )
            ]
        )


class ShapeSubdividesConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="shape-consumer.read",
            inputs=(InputSpec("stages", TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE)),),
            outputs=(OutputSpec("stages", TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE)),),
        )

    @classmethod
    def execute(cls, *, stages: object) -> Mapping[str, object]:
        restored = materialize_shape_subdivides(stages)
        assert len(restored) == 1
        stage = restored[0]
        assert stage.shape == (3, 2)
        assert stage._scale == (Fraction(1, 3), Fraction(5, 7), Fraction(9, 11))
        assert stage._spatial_cache == {}
        assert stage.feats.tolist() == [[0.0, 1.0]]
        assert stage.coords.tolist() == [[0, 1, 2, 3]]
        return cls.outputs(stages=restored)


@pytest.mark.parametrize("in_process", (True, False))
def test_shape_subdivides_survives_ordinary_hosts_and_cache_restart(
    tmp_path: Path, in_process: bool
) -> None:
    manifests: list[Path] = []
    for name, node in (
        ("source", "ShapeSubdividesProducer"),
        ("consumer", "ShapeSubdividesConsumer"),
    ):
        module = f"shape_{name}"
        (tmp_path / f"{module}.py").write_text(
            f"from test_geometry_wire import {node}, register_inference_types\nNODES=({node},)\n",
            encoding="ascii",
        )
        manifest = tmp_path / f"{name}.toml"
        manifest.write_text(
            f'[pack]\nname="shape-{name}"\npublisher="dinkster"\n'
            f'namespaces=["shape-{name}"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests.append(manifest)

    async def scenario() -> None:
        graph = Graph(
            {
                "source": GraphNode("shape-source.make"),
                "consumer": GraphNode("shape-consumer.read", {"stages": Link("source", "stages")}),
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
                            in_process=in_process,
                            worker_group=None if in_process else manifest.stem,
                            group_manifests=() if in_process else (manifest,),
                        )
                    )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["consumer"]
                )
                assert set(result.cached if replay else result.executed) == {"source", "consumer"}
                stages = result.outputs["consumer"]["stages"].resolve()
                spec = registry.spec(SHAPE_SUBDIVIDES_TYPE)
                assert spec.encode(stages) == spec.encode(
                    ShapeSubdividesProducer.execute()["stages"]
                )
            finally:
                await composer.close()

    asyncio.run(scenario())
