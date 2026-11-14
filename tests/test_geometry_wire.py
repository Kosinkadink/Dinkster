"""Geometry transport preserves complete runtime fields without importing Torch."""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dinkster_api.v1 import (
    SHAPE_SUBDIVIDES_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    materialize_shape_subdivides,
    register_inference_types,
    register_vision_packet_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import EncodedLatentTensor, register_core_types
from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario


def tensor(shape: tuple[int, ...], dtype: str = "float32") -> EncodedLatentTensor:
    count = 1
    for size in shape:
        count *= size
    fmt = "q" if dtype == "int64" else "f"
    return EncodedLatentTensor(dtype, shape, struct.pack("<" + fmt * count, *range(count)))


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

    run_scenario(scenario())


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
    register_vision_packet_types(registry)
    spec = registry.spec(type_id)
    register_vision_packet_types(registry)
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
    register_vision_packet_types(authority)
    assert authority.spec(type_id) is first


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

    run_scenario(scenario())
