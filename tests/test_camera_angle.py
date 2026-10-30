"""Camera-angle geometry, bucket boundaries and ordinary host transport."""

from __future__ import annotations

import asyncio
import math
import os
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
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.camera_angle import CameraAngle

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("horizontal", "vertical", "zoom", "description"),
    (
        (0, -16, 1.999, "front view low-angle shot wide shot"),
        (22, -15, 2, "front view eye-level shot medium shot"),
        (23, 14, 5.999, "front-right quarter view eye-level shot medium shot"),
        (67, 15, 6, "front-right quarter view elevated shot close-up"),
        (68, 44, 10, "right side view elevated shot close-up"),
        (113, 45, 7, "back-right quarter view high-angle shot close-up"),
        (180, 0, 5, "back view eye-level shot medium shot"),
        (225, 0, 5, "back-left quarter view eye-level shot medium shot"),
        (270, 0, 5, "left side view eye-level shot medium shot"),
        (337, 0, 5, "front-left quarter view eye-level shot medium shot"),
        (338, 0, 5, "front view eye-level shot medium shot"),
        (360, 0, 5, "front view eye-level shot medium shot"),
    ),
)
def test_camera_shot_boundaries(
    horizontal: int, vertical: int, zoom: float, description: str
) -> None:
    result = CameraAngle.execute(horizontal_angle=horizontal, vertical_angle=vertical, zoom=zoom)
    assert result["prompt"] == description


def test_camera_geometry_and_clamps_ignore_presentation_values() -> None:
    result = CameraAngle.execute(
        horizontal_angle=90, vertical_angle=30, zoom=4, image=object(), view="presentation"
    )
    camera = result["camera_info"]
    assert isinstance(camera, dict)
    assert camera["position"] == pytest.approx({"x": 3 * math.sqrt(3), "y": 3, "z": 0})
    assert camera["target"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert camera["zoom"] == 1.35 and camera["fov"] == 35
    assert camera["cameraType"] == "perspective"
    assert CameraAngle.execute(horizontal_angle=-2, vertical_angle=-60, zoom=-1) == (
        CameraAngle.execute(horizontal_angle=0, vertical_angle=-30, zoom=0)
    )
    assert CameraAngle.execute(horizontal_angle=700, vertical_angle=90, zoom=20) == (
        CameraAngle.execute(horizontal_angle=360, vertical_angle=60, zoom=10)
    )


class CameraAngleInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "camera-angle-fixture.inspect",
            inputs=(
                InputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                InputSpec("prompt", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, camera: dict[str, object], prompt: str) -> Mapping[str, object]:
        assert camera["position"] == pytest.approx({"x": 3 * math.sqrt(3), "y": 3, "z": 0})
        assert camera["target"] == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert camera["zoom"] == 1.35 and camera["fov"] == 35
        assert camera["cameraType"] == "perspective"
        assert prompt == "right side view elevated shot medium shot"
        return cls.outputs(summary="camera geometry and prompt transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_camera_angle_native_manifest_executes_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        )
        (tmp_path / "camera_angle_fixture.py").write_text(
            "from test_camera_angle import CameraAngleInspect\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (CameraAngleInspect,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="camera-angle-fixture"\npublisher="dinkster"\n'
            'namespaces=["camera-angle-fixture"]\n[pack.entry]\n'
            'nodes="camera_angle_fixture:NODES"\n'
            'types="camera_angle_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "camera-angle-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "camera": GraphNode(
                        "CameraAngle", {"horizontal_angle": 90, "vertical_angle": 30, "zoom": 4.0}
                    ),
                    "inspect": GraphNode(
                        "camera-angle-fixture.inspect",
                        {
                            "camera": Link("camera", "camera_info"),
                            "prompt": Link("camera", "prompt"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == (
                    "camera geometry and prompt transported"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
