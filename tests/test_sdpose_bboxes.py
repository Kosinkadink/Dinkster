"""Public face-box geometry and Torch-free native discovery."""

import asyncio
import copy
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_STRING,
    POSE_KEYPOINT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sdpose import SDPoseFaceBBoxes

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("scale", "square", "expected"),
    (
        (1.0, False, {"x": 8, "y": 8, "width": 16, "height": 4}),
        (4.0, False, {"x": 0, "y": 5, "width": 32, "height": 8}),
        (4.0, True, {"x": 0, "y": 0, "width": 32, "height": 32}),
    ),
)
def test_face_bbox_area_scale_and_asymmetric_vertical_expansion(
    scale: float, square: bool, expected: dict[str, int]
) -> None:
    keypoints = [
        {
            "canvas_width": 64,
            "canvas_height": 32,
            "people": [{"face_keypoints_2d": [8, 8, -2, 24, 12, 0]}],
        }
    ]
    before = copy.deepcopy(keypoints)
    assert SDPoseFaceBBoxes.execute(keypoints=keypoints, scale=scale, force_square=square) == {
        "bboxes": [[expected]]
    }
    assert keypoints == before


def test_face_boxes_keep_frame_person_order_and_reanchor_square_at_edge() -> None:
    keypoints = [
        {
            "canvas_width": 64,
            "canvas_height": 32,
            "people": [
                {},
                {"face_keypoints_2d": None},
                {"face_keypoints_2d": []},
                {"face_keypoints_2d": [48, 24, 0, 60, 28, 0]},
                {"face_keypoints_2d": [4, 2, 1, 8, 14, 1]},
                {"face_keypoints_2d": [5, 9, 1, 5, 17, 1]},
            ],
        },
        {"canvas_width": 32, "canvas_height": 64, "people": []},
    ]
    assert SDPoseFaceBBoxes.execute(keypoints=keypoints, scale=1, force_square=True) == {
        "bboxes": [
            [
                {"x": 48, "y": 20, "width": 12, "height": 12},
                {"x": 0, "y": 2, "width": 12, "height": 12},
            ],
            [],
        ]
    }
    assert SDPoseFaceBBoxes.execute(keypoints=[], scale=1, force_square=False) == {"bboxes": []}


def test_sdpose_bbox_schema_and_execution_are_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_api.v1 import BOUNDING_BOX_TYPE, POSE_KEYPOINT_TYPE, TypeExpr\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.sdpose import SDPoseFaceBBoxes\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "schema = SDPoseFaceBBoxes.define_schema()\n"
            "assert SDPoseFaceBBoxes in COMFY_NODES and schema.node_type in manifest.namespaces\n"
            "assert [p.id for p in schema.inputs] == ['keypoints', 'scale', 'force_square']\n"
            "assert schema.inputs[0].type == TypeExpr.concrete(POSE_KEYPOINT_TYPE)\n"
            "assert schema.outputs[0].type == TypeExpr.concrete(BOUNDING_BOX_TYPE)\n"
            "assert schema.inputs[1].default == 1.5 and schema.inputs[2].default is True\n"
            "assert SDPoseFaceBBoxes.execute(keypoints=[], scale=1.5, force_square=True) "
            "== {'bboxes': []}\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


FRAMES = [
    {
        "canvas_width": 64,
        "canvas_height": 32,
        "people": [
            {"face_keypoints_2d": [8, 8, -2, 24, 12, 0], "label": "first"},
            {"face_keypoints_2d": [48, 24, 1, 60, 28, 1]},
        ],
    },
    {"canvas_width": 32, "canvas_height": 64, "people": []},
]


class FaceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-fixture.source",
            outputs=(OutputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(keypoints=copy.deepcopy(FRAMES))


class FaceObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-fixture.observer",
            inputs=(
                InputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),
                InputSpec("boxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, keypoints: object, boxes: object) -> Mapping[str, object]:
        assert keypoints == FRAMES
        assert boxes == [
            [
                {"x": 8, "y": 8, "width": 16, "height": 4},
                {"x": 48, "y": 24, "width": 12, "height": 4},
            ],
            [],
        ]
        return cls.outputs(result="face boxes and source keypoints preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_face_boxes_cross_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path("packages/dinkster-nodes-comfy/dinkster-pack.toml").resolve()
        module = "sdpose_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_sdpose_bboxes import FaceSource, FaceObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (FaceSource, FaceObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="sdpose-fixture"\npublisher="dinkster"\n'
            'namespaces=["sdpose-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            },
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
                    "source": GraphNode("sdpose-fixture.source"),
                    "boxes": GraphNode(
                        "SDPoseFaceBBoxes",
                        {
                            "keypoints": Link("source", "keypoints"),
                            "scale": 1.0,
                            "force_square": False,
                        },
                    ),
                    "observer": GraphNode(
                        "sdpose-fixture.observer",
                        {
                            "keypoints": Link("source", "keypoints"),
                            "boxes": Link("boxes", "bboxes"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observer"])
                assert result.outputs["observer"]["result"].resolve() == (
                    "face boxes and source keypoints preserved"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
