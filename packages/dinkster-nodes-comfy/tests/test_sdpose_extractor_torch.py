"""Controlled SDPose heads expose crop geometry and public frame transport."""

import asyncio
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MODEL_TYPE,
    POSE_KEYPOINT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sdpose_extract import SDPoseKeypointExtractor

from dinkster.compose import PackSpec, ServingComposer


class ControlledHead:
    heatmap_size = (2, 3)
    shoulder_score = 0.3

    def __call__(self, features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        assert features is not None and features.shape[1] == 640
        assert torch.all(features == 1)
        kp = np.column_stack((np.arange(133) / 16 + 2, np.arange(133) / 32 + 3)).astype(np.float32)
        kp[0] = [-1, 9]
        sc = np.full(133, 0.75, dtype=np.float32)
        sc[5], sc[6] = self.shoulder_score, 0.875
        return np.tile(kp, (len(features), 1, 1)), np.tile(sc, (len(features), 1))


class ControlledModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(heatmap_head=ControlledHead()))
        self.model_options = {"transformer_options": {"preserved": "source"}}

    def clone(self) -> "ControlledModel":
        clone = ControlledModel()
        clone.model_options = copy.deepcopy(self.model_options)
        return clone


class ControlledVAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[1:] == (12, 8, 3) and image.dtype == torch.float32
        self.encoded.append(image.clone())
        return image.permute(0, 3, 1, 2)


def controlled_sample(model: ControlledModel, **options: Any) -> None:
    assert options["steps"] == 1 and options["cfg"] == 1.0
    assert options["sampler_name"] == "euler" and options["scheduler"] == "simple"
    assert options["disable_noise"] and options["disable_pbar"]
    assert torch.count_nonzero(options["noise"]) == 0
    latent = options["latent_image"]
    patch = model.model_options["transformer_options"]["patches"]["output_block_patch"][0]
    patch(torch.zeros(len(latent), 320, 1, 1), None, {})
    features = torch.ones(len(latent), 640, 1, 1)
    patch(features, None, {})
    features.zero_()


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_inference_process()
    monkeypatch.setattr(
        importlib.import_module("dinkster_inference.sample"), "sample", controlled_sample
    )


def assert_person(person: dict[str, Any], sx: float, sy: float, x: int = 0, y: int = 0) -> None:
    body = person["pose_keypoints_2d"]
    assert body[:3] == [-1, -1, 0.75]
    # OpenPose right shoulder is original MMPose index 6, not index 2.
    expected_shoulder = torch.tensor([2 + 6 / 16, 3 + 6 / 32])
    expected_shoulder /= torch.tensor([sx, sy])
    expected_shoulder += torch.tensor([x, y])
    assert body[6:9] == [*expected_shoulder.tolist(), 0.875]
    assert body[5] == 0  # Neck confidence is zero when either shoulder is exactly 0.3.
    assert [
        len(person[k])
        for k in (
            "pose_keypoints_2d",
            "foot_keypoints_2d",
            "face_keypoints_2d",
            "hand_right_keypoints_2d",
            "hand_left_keypoints_2d",
        )
    ] == [54, 18, 210, 63, 63]
    assert person["face_keypoints_2d"][-6:] == body[42:48]
    right = person["hand_right_keypoints_2d"]
    expected_hand = torch.tensor([2 + 91 / 16, 3 + 91 / 32])
    expected_hand /= torch.tensor([sx, sy])
    expected_hand += torch.tensor([x, y])
    assert right[:2] == expected_hand.tolist()


@pytest.mark.usefixtures("runtime")
def test_batched_extraction_layout_resize_and_source_preservation() -> None:
    model, vae = ControlledModel(), ControlledVAE()
    image = torch.arange(3 * 24 * 32 * 3, dtype=torch.float32).reshape(3, 24, 32, 3) / 10000
    before = image.clone()
    frames = SDPoseKeypointExtractor.execute(model=model, vae=vae, image=image, batch_size=2)[
        "keypoints"
    ]
    assert [len(v) for v in vae.encoded] == [2, 1]
    expected = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size=(12, 8), mode="area")
    torch.testing.assert_close(torch.cat(vae.encoded).permute(0, 3, 1, 2), expected, rtol=0, atol=0)
    assert len(frames) == 3
    for frame in frames:
        assert frame["canvas_width"] == 32 and frame["canvas_height"] == 24
        assert len(frame["people"]) == 1
        assert_person(frame["people"][0], 0.25, 0.5)
    torch.testing.assert_close(image, before, rtol=0, atol=0)
    assert model.model_options == {"transformer_options": {"preserved": "source"}}


@pytest.mark.usefixtures("runtime")
def test_neck_uses_minimum_confidence_above_strict_boundary() -> None:
    model = ControlledModel()
    model.model.diffusion_model.heatmap_head.shoulder_score = 0.3125
    frames = SDPoseKeypointExtractor.execute(
        model=model, vae=ControlledVAE(), image=torch.zeros(1, 24, 32, 3), batch_size=1
    )["keypoints"]
    body = frames[0]["people"][0]["pose_keypoints_2d"]
    assert body[3:6] == [9.375, 6.34375, 0.3125]


@pytest.mark.parametrize(
    "boxes,counts",
    [
        (
            [
                [
                    {"x": -2, "y": 3, "width": 10, "height": 8},
                    {"x": 20, "y": 4, "width": 0, "height": 2},
                ]
            ],
            [1, 1],
        ),
        ([[{"x": 5, "y": 4, "width": 2, "height": 3}], []], [1, 1]),
        ([[{"x": 30, "y": 4, "width": -2, "height": 2}]], [0, 0]),
        ([], [1, 1]),
        ({"x": 5, "y": 4, "width": 2, "height": 3}, [1, 1]),
        (
            [
                [
                    {"x": -2, "y": 3, "width": 10, "height": 8},
                    {"x": 5, "y": 4, "width": 2, "height": 3},
                ]
            ],
            [2, 2],
        ),
    ],
)
@pytest.mark.usefixtures("runtime")
def test_crop_clamping_last_frame_reuse_and_fallback(boxes: Any, counts: list[int]) -> None:
    original = copy.deepcopy(boxes)
    vae = ControlledVAE()
    image = torch.linspace(0, 1, 2 * 16 * 24 * 3).reshape(2, 16, 24, 3)
    frames = SDPoseKeypointExtractor.execute(
        model=ControlledModel(), vae=vae, image=image, batch_size=16, bboxes=boxes
    )["keypoints"]
    assert [len(f["people"]) for f in frames] == counts
    assert boxes == original
    if counts == [0, 0]:
        assert vae.encoded == []
    elif isinstance(boxes, list) and boxes and boxes[0][0]["x"] == -2:
        for f in frames:
            assert_person(f["people"][0], 1, 1.5, 0, 3)
            if counts == [2, 2]:
                assert_person(f["people"][1], 4, 4, 5, 4)
        expected = torch.nn.functional.interpolate(
            image[:1, 3:11, :8].permute(0, 3, 1, 2),
            size=(12, 8),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(vae.encoded[0].permute(0, 3, 1, 2), expected, rtol=0, atol=0)
    elif not boxes:
        for f in frames:
            assert_person(f["people"][0], 8 / 24, 12 / 16)
    else:
        assert_person(frames[0]["people"][0], 4, 4, 5, 4)
        if isinstance(boxes, list):
            assert_person(frames[1]["people"][0], 8 / 24, 12 / 16)
        else:
            assert_person(frames[1]["people"][0], 4, 4, 5, 4)


@pytest.mark.usefixtures("runtime")
def test_missing_heatmap_head_retains_error() -> None:
    model = ControlledModel()
    model.model.diffusion_model = object()
    with pytest.raises(ValueError, match="does not have a heatmap_head"):
        SDPoseKeypointExtractor.execute(
            model=model, vae=ControlledVAE(), image=torch.zeros(1, 4, 8, 3), batch_size=1
        )


class ExtractionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-extract-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        importlib.import_module("dinkster_inference.sample").sample = controlled_sample
        return cls.outputs(
            model=ControlledModel(), vae=ControlledVAE(), image=torch.zeros(2, 24, 32, 3)
        )


class ExtractionObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sdpose-extract-fixture.observer",
            inputs=(InputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, keypoints: Any) -> dict[str, object]:
        assert len(keypoints) == 2
        for f in keypoints:
            assert f["canvas_width"] == 32 and f["canvas_height"] == 24
            assert_person(f["people"][0], 0.25, 0.5)
        return cls.outputs(result="ordered OpenPose frames")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_grouped_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    sampler = importlib.import_module("dinkster_inference.sample")
    original = sampler.sample

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "extract_fixture.py").write_text(
            "from test_sdpose_extractor_torch import ExtractionSource, ExtractionObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(ExtractionSource,ExtractionObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="sdpose-extract-fixture"\npublisher="dinkster"\n'
            'namespaces=["sdpose-extract-fixture"]\n[pack.entry]\nnodes="extract_fixture:NODES"\n'
            'types="extract_fixture:register_inference_types"\n',
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
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="sdpose-extract",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sdpose-extract-fixture.source"),
                    "extract": GraphNode(
                        "SDPoseKeypointExtractor",
                        {
                            "model": Link("source", "model"),
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                            "batch_size": 1,
                        },
                    ),
                    "observe": GraphNode(
                        "sdpose-extract-fixture.observer",
                        {"keypoints": Link("extract", "keypoints")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observe"])
                assert result.outputs["observe"]["result"].resolve() == "ordered OpenPose frames"
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        sampler.sample = original
