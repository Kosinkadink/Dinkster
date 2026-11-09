"""Observable pose pixels, input preservation and ordinary portable transport."""

import asyncio
import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    POSE_KEYPOINT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sdpose import SDPoseDrawKeypoints

from dinkster.compose import PackSpec, ServingComposer

OPTIONS = dict(
    draw_body=False,
    draw_hands=False,
    draw_face=False,
    draw_feet=False,
    stick_width=4,
    face_point_size=1,
    score_threshold=0.5,
    draw_head=False,
)


def frames() -> list[dict[str, Any]]:
    person = {
        name: [-20.0, -20.0, 0.0] * count
        for name, count in (
            ("pose_keypoints_2d", 18),
            ("face_keypoints_2d", 70),
            ("hand_right_keypoints_2d", 21),
            ("hand_left_keypoints_2d", 21),
            ("foot_keypoints_2d", 6),
        )
    }
    person["face_keypoints_2d"][:6] = [11.75, 9.5, 0.5, 27, 20, 0.49]
    person["face_keypoints_2d"][-6:] = [33, 29, 1, 35, 30, 1]
    return [
        {"canvas_width": 48, "canvas_height": 40, "people": [person]},
        {"canvas_width": 17, "canvas_height": 23, "people": []},
    ]


def expected_face() -> torch.Tensor:
    expected = torch.zeros(2, 40, 48, 3)
    for x, y in ((11, 9), (10, 9), (12, 9), (11, 8), (11, 10)):
        expected[0, y, x] = 1
    return expected


def test_face_threshold_eye_drop_first_canvas_and_source_preservation() -> None:
    keypoints = frames()
    original = copy.deepcopy(keypoints)
    result = SDPoseDrawKeypoints.execute(keypoints=keypoints, **{**OPTIONS, "draw_face": True})[
        "IMAGE"
    ]
    torch.testing.assert_close(result, expected_face(), rtol=0, atol=0)
    assert keypoints == original
    assert result.dtype == torch.float32 and result.device.type == "cpu"


@pytest.mark.parametrize(
    ("field", "toggle", "index", "rgb"),
    (
        ("pose_keypoints_2d", "draw_body", 2, (255, 170, 0)),
        ("pose_keypoints_2d", "draw_head", 0, (255, 0, 0)),
        ("foot_keypoints_2d", "draw_feet", 0, (255, 0, 0)),
        ("hand_right_keypoints_2d", "draw_hands", 0, (0, 0, 255)),
        ("hand_left_keypoints_2d", "draw_hands", 0, (0, 0, 255)),
    ),
)
def test_independent_toggles_use_correct_group_colors(
    field: str, toggle: str, index: int, rgb: tuple[int, int, int]
) -> None:
    keypoints = frames()[:1]
    person = keypoints[0]["people"][0]
    person["face_keypoints_2d"] = [-20, -20, 0] * 70
    person[field][index * 3 : index * 3 + 3] = [15.5, 19.25, 0.5]
    result = SDPoseDrawKeypoints.execute(keypoints=keypoints, **{**OPTIONS, toggle: True})["IMAGE"]
    expected = torch.zeros(1, 40, 48, 3)
    for y in range(15, 24):
        for x in range(11, 20):
            if (x - 15) ** 2 + (y - 19) ** 2 <= 16:
                expected[0, y, x] = torch.tensor(rgb) / 255
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_body_limb_fills_between_markers_and_respects_width() -> None:
    keypoints = frames()[:1]
    person = keypoints[0]["people"][0]
    person["pose_keypoints_2d"][3:9] = [10, 20, 1, 30, 20, 1]
    image = SDPoseDrawKeypoints.execute(
        keypoints=keypoints, **{**OPTIONS, "draw_body": True, "stick_width": 2}
    )["IMAGE"]
    assert image[0, 20, 20].tolist() == [1, 0, 0]
    assert image[0, 24, 20].tolist() == [0, 0, 0]
    torch.testing.assert_close(image[0, 20, 10], torch.tensor([255, 85, 0]) / 255, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 20, 30], torch.tensor([255, 170, 0]) / 255, rtol=0, atol=0)


def test_empty_input_and_disabled_groups_are_black() -> None:
    empty = SDPoseDrawKeypoints.execute(keypoints=[], **OPTIONS)["IMAGE"]
    assert empty.shape == (1, 64, 64, 3) and empty.dtype == torch.float32
    assert torch.count_nonzero(empty) == 0
    disabled = SDPoseDrawKeypoints.execute(keypoints=frames(), **OPTIONS)["IMAGE"]
    assert disabled.shape == (2, 40, 48, 3) and torch.count_nonzero(disabled) == 0


class PoseSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-drawing-fixture.source",
            outputs=(OutputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(keypoints=frames())


class PoseObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-drawing-fixture.observer",
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any, keypoints: Any) -> Mapping[str, object]:
        torch.testing.assert_close(materialize_image(image), expected_face(), rtol=0, atol=0)
        assert keypoints == frames()
        return cls.outputs(result="pose pixels, frame order and source preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_draw_through_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "pose_drawing_fixture.py").write_text(
            "from test_sdpose_drawing_torch import PoseSource, PoseObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (PoseSource, PoseObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="sdpose-drawing-fixture"\npublisher="dinkster"\n'
            'namespaces=["sdpose-drawing-fixture"]\n[pack.entry]\n'
            'nodes="pose_drawing_fixture:NODES"\n'
            'types="pose_drawing_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else manifest.stem,
                        group_manifests=() if in_process else (manifest,),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sdpose-drawing-fixture.source"),
                    "draw": GraphNode(
                        "SDPoseDrawKeypoints",
                        {**OPTIONS, "draw_face": True, "keypoints": Link("source", "keypoints")},
                    ),
                    "observer": GraphNode(
                        "sdpose-drawing-fixture.observer",
                        {"image": Link("draw", "IMAGE"), "keypoints": Link("source", "keypoints")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observer"])
                assert result.outputs["observer"]["result"].resolve() == (
                    "pose pixels, frame order and source preserved"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
