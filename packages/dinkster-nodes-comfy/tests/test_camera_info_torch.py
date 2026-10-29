"""Camera geometry and portable ordinary-host transport on CPU."""

from __future__ import annotations

import asyncio
import copy
import math
import os
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    CAMERA_INFO_STATE_TYPE,
    CORE_BOOLEAN,
    LOAD3D_CAMERA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.camera_info import CreateCameraInfo

from dinkster.compose import PackSpec, ServingComposer


def _execute(mode: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    values = dict(
        target_x=1.0,
        target_y=-2.0,
        target_z=3.0,
        roll=0.0,
        fov=47.0,
        zoom=1.7,
        camera_type="orthographic",
    )
    values.update(overrides)
    return CreateCameraInfo.execute(mode=mode, **values)["camera_info"]


def test_orbit_translation_and_roll_preserve_source_inputs() -> None:
    mode = {"mode": "orbit", "yaw": 90.0, "pitch": 30.0, "distance": 6.0}
    original = copy.deepcopy(mode)
    state = {"position": {"x": 90, "y": 80, "z": 70}, "other": [1, 2]}
    old_state = copy.deepcopy(state)
    camera = _execute(mode, roll=37.0, camera_info_state=state)
    assert mode == original and state == old_state
    assert camera["position"] == pytest.approx({"x": 1 + 3 * math.sqrt(3), "y": 1, "z": 3})
    assert camera["target"] == {"x": 1, "y": -2, "z": 3}
    assert camera["fov"] == 47 and camera["zoom"] == 1.7
    assert camera["cameraType"] == "orthographic" and camera["useCustomUp"] is True
    # Independent camera axes for yaw=90, pitch=30, then local-Z roll.
    q = camera["quaternion"]
    half_roll = math.radians(37) / 2
    w = math.cos(math.pi / 12) / math.sqrt(2)
    x = -math.sin(math.pi / 12) / math.sqrt(2)
    y = math.cos(math.pi / 12) / math.sqrt(2)
    z = math.sin(math.pi / 12) / math.sqrt(2)
    assert q == pytest.approx(
        {
            "w": w * math.cos(half_roll) - z * math.sin(half_roll),
            "x": x * math.cos(half_roll) + y * math.sin(half_roll),
            "y": y * math.cos(half_roll) - x * math.sin(half_roll),
            "z": z * math.cos(half_roll) + w * math.sin(half_roll),
        },
        abs=1e-7,
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    (
        ((1, -2, 7), {"w": 1, "x": 0, "y": 0, "z": 0}),
        ((1, -2, -1), {"w": 0, "x": 0, "y": 1, "z": 0}),
        ((1, 2, 3), {"w": 0, "x": 0, "y": math.sqrt(0.5), "z": math.sqrt(0.5)}),
        ((1, -6, 3), {"w": math.sqrt(0.5), "x": math.sqrt(0.5), "y": 0, "z": 0}),
    ),
)
def test_look_at_half_turn_and_vertical_fallback(
    position: tuple[int, ...], expected: dict[str, float]
) -> None:
    camera = _execute(
        dict(
            mode="look_at",
            **dict(zip(("position_x", "position_y", "position_z"), position, strict=True)),
        )
    )
    assert camera["quaternion"] == pytest.approx(expected, abs=1e-7)
    assert "useCustomUp" not in camera


def test_quaternion_normalization_ignores_target_roll_and_state() -> None:
    mode = dict(
        mode="quaternion",
        position_x=2.0,
        position_y=-4.0,
        position_z=7.0,
        quat_x=0.0,
        quat_y=2.0,
        quat_z=0.0,
        quat_w=2.0,
    )
    camera = _execute(
        mode,
        target_x=900,
        target_y=-800,
        target_z=700,
        roll=123,
        camera_info_state={"unconsumed": True},
    )
    assert camera["position"] == {"x": 2, "y": -4, "z": 7}
    assert camera["target"] == pytest.approx({"x": 1, "y": -4, "z": 7}, abs=1e-6)
    assert camera["quaternion"] == pytest.approx(
        {"x": 0, "y": math.sqrt(0.5), "z": 0, "w": math.sqrt(0.5)}, abs=1e-7
    )
    assert camera["useCustomUp"] is True
    mode.update(quat_y=0, quat_w=0)
    zero = _execute(mode)
    assert zero["quaternion"] == {"w": 0, "x": 0, "y": 0, "z": 0}
    assert zero["target"] == {"x": 2, "y": -4, "z": 6}


class CameraStateSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "camera-info-fixture.state",
            outputs=(OutputSpec("state", TypeExpr.concrete(CAMERA_INFO_STATE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(state={"position": {"x": 88, "y": -99, "z": 77}, "view": [1, 3]})


class CameraInfoInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "camera-info-fixture.inspect",
            inputs=(InputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, camera: dict[str, Any]) -> Any:
        assert camera["position"] == {"x": 2, "y": -4, "z": 7}
        assert camera["target"] == {"x": 1, "y": -2, "z": 3}
        assert camera["cameraType"] == "orthographic"
        assert camera["fov"] == 47 and camera["zoom"] == 1.7
        assert camera["useCustomUp"] is True
        assert sum(value**2 for value in camera["quaternion"].values()) == pytest.approx(
            1, abs=2e-7
        )
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_camera_info_ordinary_production_manifest_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "camera_info_fixture.py").write_text(
            "from test_camera_info_torch import CameraStateSource, CameraInfoInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CameraStateSource, CameraInfoInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="camera-info-fixture"\npublisher="dinkster"\n'
            'namespaces=["camera-info-fixture"]\n[pack.entry]\n'
            'nodes="camera_info_fixture:NODES"\n'
            'types="camera_info_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(Path.cwd() / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "camera-info-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "state": GraphNode("camera-info-fixture.state"),
                    "camera": GraphNode(
                        "CreateCameraInfo",
                        {
                            "mode.position_x": 2.0,
                            "mode.position_y": -4.0,
                            "mode.position_z": 7.0,
                            "target_x": 1.0,
                            "target_y": -2.0,
                            "target_z": 3.0,
                            "roll": 37.0,
                            "fov": 47.0,
                            "zoom": 1.7,
                            "camera_type": "orthographic",
                            "camera_info_state": Link("state", "state"),
                        },
                        slot_variants={"mode": "look_at"},
                    ),
                    "inspect": GraphNode(
                        "camera-info-fixture.inspect", {"camera": Link("camera", "camera_info")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
