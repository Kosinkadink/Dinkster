"""Cold portable LOAD3D_CAMERA values retain the complete public camera dictionary."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import LATENT_TYPE, EncodedLatentTensor, RenditionUnavailable, encode_latent

from dinkster.compose import PackSpec, ServingComposer

PERSPECTIVE = {
    "position": {"x": 1, "y": 2.5, "z": -3},
    "target": {"x": 0.0, "y": 1, "z": 2},
    "zoom": 1.25,
    "cameraType": "perspective",
    "fov": 45,
    "quaternion": {"x": 0, "y": 0.5, "z": 0, "w": 1},
    "useCustomUp": True,
    "aspect": 16 / 9,
    "near": 0.01,
    "far": 1000,
    "extension": {"label": "source", "nested": (1, 2)},
}
ORTHOGRAPHIC = {
    "position": {"x": -4.0, "y": 3, "z": 8},
    "target": {"x": 0, "y": 0, "z": 0},
    "zoom": 2,
    "cameraType": "orthographic",
    "frustum": {"left": -2, "right": 2.0, "top": 3, "bottom": -3.0},
}
MODEL_INFO = [
    {
        "position": {"x": -3, "y": 2.5, "z": 8},
        "quaternion": {"x": 0, "y": 0.5, "z": 0, "w": 1},
        "scale": {"x": 2, "y": 0, "z": -1.5},
        "extension": {"label": "first"},
    },
    {
        "position": {"x": 7, "y": -4, "z": 1},
        "quaternion": {"x": 0.5, "y": 0, "z": 0, "w": 0.5},
        "scale": {"x": 1, "y": 3, "z": 4},
    },
]


@pytest.mark.parametrize(
    ("type_id", "value", "expected"),
    (
        (
            LOAD3D_CAMERA_TYPE,
            PERSPECTIVE,
            {**PERSPECTIVE, "extension": {"label": "source", "nested": [1, 2]}},
        ),
        (LOAD3D_CAMERA_TYPE, ORTHOGRAPHIC, ORTHOGRAPHIC),
        (LOAD3D_MODEL_INFO_TYPE, MODEL_INFO, MODEL_INFO),
        (LOAD3D_MODEL_INFO_TYPE, [], []),
    ),
)
def test_camera_metadata_json_renditions(type_id: str, value: object, expected: object) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    register_inference_types(registry)
    wrapped = registry.wrap(type_id, value)
    rendered = registry.render(wrapped, "json")
    assert rendered.kind == "json"
    assert rendered.mime == "application/json"
    assert json.loads(rendered.data) == expected
    assert registry.render(wrapped).data == rendered.data


def test_camera_metadata_json_refuses_tensor_extensions_without_changing_portability() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    extension = EncodedLatentTensor("float32", (1,), b"\x00\x00\x80?")
    for type_id, value in (
        (LOAD3D_CAMERA_TYPE, {**PERSPECTIVE, "extension": extension}),
        (LOAD3D_MODEL_INFO_TYPE, [{**MODEL_INFO[0], "extension": extension}]),
    ):
        spec = registry.spec(type_id)
        assert spec.decode(spec.encode(value)) == value
        wrapped = registry.wrap(type_id, value)
        with pytest.raises(RenditionUnavailable):
            registry.render(wrapped, "json")


@pytest.mark.parametrize("value", ([], MODEL_INFO))
def test_load3d_model_info_preserves_order_and_transform_values(value: list[object]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOAD3D_MODEL_INFO_TYPE)
    register_inference_types(registry)
    assert registry.spec(LOAD3D_MODEL_INFO_TYPE) is spec
    payload = spec.encode(value)
    assert spec.validate_encoded is not None
    spec.validate_encoded(payload, {})
    assert spec.decode(payload) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(spec.decode(payload)) == spec.fingerprint(value)
    if value:
        assert spec.fingerprint(list(reversed(value))) != spec.fingerprint(value)
    latent_fingerprint = registry.spec(LATENT_TYPE).fingerprint
    assert latent_fingerprint is not None
    assert spec.fingerprint(value) != latent_fingerprint({"samples": value})


@pytest.mark.parametrize(
    "value",
    (
        {},
        [None],
        [{}],
        [{**MODEL_INFO[0], "position": {"x": 0, "y": 1, "z": True}}],
        [{**MODEL_INFO[0], "quaternion": {"x": 0, "y": 0, "z": 0}}],
        [{**MODEL_INFO[0], "scale": [1, 2, 3]}],
    ),
)
def test_load3d_model_info_rejects_malformed_transforms(value: object) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOAD3D_MODEL_INFO_TYPE)
    assert spec.validate_encoded is not None
    payload = encode_latent({"samples": value})
    with pytest.raises(ValueError):
        spec.encode(value)
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


@pytest.mark.parametrize("value", (PERSPECTIVE, ORTHOGRAPHIC))
def test_load3d_camera_roundtrip_preserves_complete_structure(value: dict[str, object]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOAD3D_CAMERA_TYPE)
    register_inference_types(registry)
    assert registry.spec(LOAD3D_CAMERA_TYPE) is spec
    assert spec.validate_encoded is not None
    payload = spec.encode(value)
    spec.validate_encoded(payload, {})
    assert spec.decode(payload) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(spec.decode(payload)) == spec.fingerprint(value)
    assert spec.fingerprint(dict(reversed(tuple(value.items())))) == spec.fingerprint(value)
    latent_fingerprint = registry.spec(LATENT_TYPE).fingerprint
    assert latent_fingerprint is not None
    assert latent_fingerprint({"samples": value}) != spec.fingerprint(value)


@pytest.mark.parametrize(
    "value",
    (
        {},
        {**PERSPECTIVE, "position": (1, 2, 3)},
        {**PERSPECTIVE, "target": {"x": 0, "y": 1, "z": True}},
        {**PERSPECTIVE, "zoom": True},
        {**PERSPECTIVE, "cameraType": 1},
        {**PERSPECTIVE, "quaternion": {"x": 0, "y": 0, "z": 0}},
        {**PERSPECTIVE, "useCustomUp": 1},
        {**PERSPECTIVE, "frustum": {"left": 0, "right": 1, "top": 1, "bottom": False}},
    ),
)
def test_load3d_camera_rejects_malformed_payloads(value: dict[str, object]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOAD3D_CAMERA_TYPE)
    assert spec.validate_encoded is not None
    payload = encode_latent({"samples": value})
    with pytest.raises(ValueError):
        spec.encode(value)
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


def test_load3d_camera_public_registration_is_cold_and_shared() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import (\n"
            "    LOAD3D_CAMERA_TYPE, TypeRegistry, register_inference_types,\n"
            ")\n"
            "from dinkster_values.builtin_types import LOAD3D_CAMERA_TYPE as canonical\n"
            "assert LOAD3D_CAMERA_TYPE is canonical\n"
            "registry = TypeRegistry()\n"
            "register_inference_types(registry)\n"
            "register_inference_types(registry)\n"
            "value = {'position': {'x': 0, 'y': 1, 'z': 2},\n"
            "         'target': {'x': 0, 'y': 0, 'z': 0},\n"
            "         'zoom': 1, 'cameraType': 'perspective'}\n"
            "spec = registry.spec(canonical)\n"
            "payload = spec.encode(value)\n"
            "spec.validate_encoded(payload, {})\n"
            "assert spec.decode(payload) == value\n"
            "from dinkster_api.v1 import LOAD3D_MODEL_INFO_TYPE\n"
            "from dinkster_values.builtin_types import LOAD3D_MODEL_INFO_TYPE as model_type\n"
            "assert LOAD3D_MODEL_INFO_TYPE is model_type\n"
            "models = registry.spec(model_type)\n"
            "assert models.decode(models.encode([])) == []\n"
            "assert not any(\n"
            "    name == 'torch' or name.startswith('torch.')\n"
            "    or name == 'dinkster_inference'\n"
            "    or name.startswith('dinkster_inference.')\n"
            "    for name in sys.modules\n"
            ")\n",
        ],
        check=True,
    )


class CameraSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="load3d-camera-fixture.source",
            outputs=(
                OutputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                OutputSpec("model_info", TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(camera=PERSPECTIVE, model_info=MODEL_INFO)


class CameraConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="load3d-camera-fixture.consumer",
            inputs=(
                InputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                InputSpec("model_info", TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, camera: object, model_info: object) -> Mapping[str, object]:
        assert camera == PERSPECTIVE
        assert model_info == MODEL_INFO
        return cls.outputs(result="camera")


class CameraTransportConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="load3d-camera-transport.consumer",
            inputs=(
                InputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                InputSpec("model_info", TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE)),
            ),
            outputs=(
                OutputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                OutputSpec("model_info", TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, camera: object, model_info: object) -> Mapping[str, object]:
        assert camera == PERSPECTIVE
        assert model_info == MODEL_INFO
        return cls.outputs(camera=camera, model_info=model_info)


@pytest.mark.parametrize("in_process", (True, False))
def test_load3d_camera_co_located_host_modes_run_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1] / "packages" / "dinkster-nodes-comfy"
        (tmp_path / "load3d_camera_fixture.py").write_text(
            "from test_load3d_camera_codec import CameraConsumer, CameraSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (CameraSource, CameraConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="load3d-camera-fixture"\npublisher="dinkster"\n'
            'namespaces=["load3d-camera-fixture"]\n[pack.entry]\n'
            'nodes="load3d_camera_fixture:NODES"\ntypes="load3d_camera_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "load3d-camera-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("load3d-camera-fixture.source"),
                    "consumer": GraphNode(
                        "load3d-camera-fixture.consumer",
                        {
                            "camera": Link("source", "camera"),
                            "model_info": Link("source", "model_info"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["result"].resolve() == "camera"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_load3d_camera_crosses_distinct_isolated_workers_with_cached_replay(tmp_path: Path) -> None:
    async def scenario() -> None:
        test_directory = Path(__file__).parent
        modules = {
            "load3d_camera_source": ("CameraSource", "load3d-camera-fixture"),
            "load3d_camera_consumer": ("CameraTransportConsumer", "load3d-camera-transport"),
        }
        manifests: dict[str, Path] = {}
        for module, (node, namespace) in modules.items():
            (tmp_path / f"{module}.py").write_text(
                f"from test_load3d_camera_codec import {node}\n"
                "from dinkster_api.v1 import register_inference_types\n"
                f"NODES = ({node},)\n",
                encoding="ascii",
            )
            manifest = tmp_path / f"{module}-pack.toml"
            manifest.write_text(
                f'[pack]\nname="{module}"\npublisher="dinkster"\n'
                f'namespaces=["{namespace}"]\n[pack.entry]\n'
                f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
                encoding="ascii",
            )
            manifests[module] = manifest
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(test_directory))),
            },
        )
        try:
            await composer.add_pack(
                PackSpec(
                    manifest=manifests["load3d_camera_source"],
                    worker_group="load3d-camera-source",
                    group_manifests=(manifests["load3d_camera_source"],),
                )
            )
            await composer.add_pack(
                PackSpec(
                    manifest=manifests["load3d_camera_consumer"],
                    worker_group="load3d-camera-consumer",
                    group_manifests=(manifests["load3d_camera_consumer"],),
                )
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("load3d-camera-fixture.source"),
                    "consumer": GraphNode(
                        "load3d-camera-transport.consumer",
                        {
                            "camera": Link("source", "camera"),
                            "model_info": Link("source", "model_info"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["consumer"])
            assert first.outputs["consumer"]["camera"].resolve() == PERSPECTIVE
            assert first.outputs["consumer"]["model_info"].resolve() == MODEL_INFO
            assert set(first.executed) == {"source", "consumer"}
            replay = await engine.run(graph, ["consumer"])
            assert replay.outputs["consumer"]["camera"].resolve() == PERSPECTIVE
            assert replay.outputs["consumer"]["model_info"].resolve() == MODEL_INFO
            assert set(replay.cached) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
