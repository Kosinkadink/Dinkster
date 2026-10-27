"""Cold portable LOAD3D_CAMERA values retain the complete public camera dictionary."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    LOAD3D_CAMERA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import LATENT_TYPE, encode_latent

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
            outputs=(OutputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(camera=PERSPECTIVE)


class CameraConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="load3d-camera-fixture.consumer",
            inputs=(InputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, camera: object) -> Mapping[str, object]:
        assert camera == PERSPECTIVE
        return cls.outputs(result="camera")


@pytest.mark.parametrize("in_process", (True, False))
def test_load3d_camera_crosses_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
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
                        "load3d-camera-fixture.consumer", {"camera": Link("source", "camera")}
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
