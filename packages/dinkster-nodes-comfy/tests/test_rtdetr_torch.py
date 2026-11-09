"""Controlled CPU outputs for native RT-DETR nodes."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.rtdetr import DrawBBoxes, RTDETR_detect

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("class_name", "limit", "labels"), [("all", 2, ["car", "person"]), ("person", 1, ["person"])]
)
def test_detector_filters_strictly_and_orders_each_frame(
    monkeypatch: pytest.MonkeyPatch, class_name: str, limit: int, labels: list[str]
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    batches = []

    def predict(images: torch.Tensor, size: tuple[int, int]) -> list[dict[str, torch.Tensor]]:
        assert size == (7, 3)
        assert images.shape[1:] == (3, 640, 640)
        batches.append(images.shape[0])
        return [
            {
                "boxes": torch.tensor([[2, 1, 6, 3], [0, 0, 2, 2], [1, 2, 5, 5], [3, 4, 8, 9]]),
                "labels": torch.tensor([0, 0, 2, 0]),
                "scores": torch.tensor([0.75, 0.5, 0.875, 0.625]),
            }
            for _ in images
        ]

    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=predict))
    source = torch.linspace(0, 1, 33 * 3 * 7 * 3).reshape(33, 3, 7, 3)
    original = source.clone()
    result = RTDETR_detect.execute(
        model=model, image=source, threshold=0.5, class_name=class_name, max_detections=limit
    )["bboxes"]
    assert batches == [32, 1]
    assert len(result) == 33
    for frame in result:
        assert [box["label"] for box in frame] == labels
        person = frame[-1]
        assert person == {
            "x": 2.0,
            "y": 1.0,
            "width": 4.0,
            "height": 2.0,
            "label": "person",
            "score": 0.75,
        }
        if class_name == "all":
            assert frame[0] == {
                "x": 1.0,
                "y": 2.0,
                "width": 4.0,
                "height": 3.0,
                "label": "car",
                "score": 0.875,
            }
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_draw_broadcasts_boxes_preserves_source_and_draws_high_scores_last() -> None:
    source = torch.full((2, 17, 23, 3), 0.5)
    boxes = [
        {"x": 2, "y": 3, "width": 12, "height": 9, "label": "not a COCO label", "score": 0.9},
        {"x": 2, "y": 3, "width": 12, "height": 9, "label": "bicycle", "score": 0.25},
    ]
    output = DrawBBoxes.execute(image=source, bboxes=boxes)["out_image"]
    assert output.shape == source.shape and output.dtype == torch.float32
    torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)
    torch.testing.assert_close(output[0, 3, 2], torch.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0)
    torch.testing.assert_close(output[0, 0, 0], torch.full((3,), 127 / 255), rtol=0, atol=0)
    assert torch.all(source == 0.5)
    assert boxes[0]["label"] == "not a COCO label"


def test_draw_pads_missing_frames_and_creates_canvas_from_first_frame() -> None:
    source = torch.ones((2, 13, 19, 3))
    box = {"x": 2, "y": 1, "width": 9, "height": 7}
    padded = DrawBBoxes.execute(image=source, bboxes=[[box], [], [box]])["out_image"]
    torch.testing.assert_close(padded[1], source[1], rtol=0, atol=0)
    canvas = DrawBBoxes.execute(bboxes=[[box], [{"x": 0, "y": 0, "width": 50, "height": 60}]])[
        "out_image"
    ]
    assert canvas.shape == (1, 8, 11, 3)
    blank = DrawBBoxes.execute(bboxes=[])["out_image"]
    assert blank.shape == (1, 640, 640, 3) and torch.count_nonzero(blank) == 0


class ControlledDetector:
    def __init__(self) -> None:
        self.model = SimpleNamespace(diffusion_model=self.predict)
        self.batches: list[int] = []

    def predict(self, images: torch.Tensor, size: tuple[int, int]) -> list[dict[str, torch.Tensor]]:
        assert size == (23, 17) and images.shape[1:] == (3, 640, 640)
        self.batches.append(images.shape[0])
        return [
            {
                "boxes": torch.tensor([[2, 3, 14, 12], [2, 3, 14, 12], [1, 1, 5, 6]]),
                "labels": torch.tensor([1, 0, 2]),
                "scores": torch.tensor([0.875, 0.75, 0.5]),
            }
            for _ in images
        ]


class DetectionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "rtdetr-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_model_gpu = lambda model: None
        return cls.outputs(
            model=ControlledDetector(), image=torch.full((2, 17, 23, 3), 0.5).numpy()
        )


class DetectionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "rtdetr-fixture.inspect",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("source", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("bboxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("canvas", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, model: ControlledDetector, source: Any, bboxes: Any, image: Any, canvas: Any
    ) -> dict[str, object]:
        source = materialize_image(source)
        image = materialize_image(image)
        canvas = materialize_image(canvas)
        assert model.batches == [2]
        expected = [
            {"x": 2.0, "y": 3.0, "width": 12.0, "height": 9.0, "label": "bicycle", "score": 0.875},
            {"x": 2.0, "y": 3.0, "width": 12.0, "height": 9.0, "label": "person", "score": 0.75},
        ]
        assert bboxes == [expected, expected]
        assert source.shape == image.shape == (2, 17, 23, 3)
        assert torch.all(source == 0.5)
        assert canvas.shape == (1, 12, 14, 3)
        torch.testing.assert_close(image[0], image[1], rtol=0, atol=0)
        torch.testing.assert_close(
            image[0, 3, 2], torch.tensor([0.0, 200 / 255, 0.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(canvas[0, 3, 2], image[0, 3, 2], rtol=0, atol=0)
        torch.testing.assert_close(image[0, 0, 0], torch.full((3,), 127 / 255), rtol=0, atol=0)
        return cls.outputs(summary="ordered boxes and rendered pixels")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_detects_and_draws_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    original_load = management.load_model_gpu

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"rtdetr_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_rtdetr_torch import DetectionSource, DetectionInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (DetectionSource, DetectionInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "rtdetr-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["rtdetr-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
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
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="rtdetr-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("rtdetr-fixture.source"),
                    "detect": GraphNode(
                        "RTDETR_detect",
                        {
                            "model": Link("source", "model"),
                            "image": Link("source", "image"),
                            "threshold": 0.5,
                            "class_name": "all",
                            "max_detections": 2,
                        },
                    ),
                    "draw": GraphNode(
                        "DrawBBoxes",
                        {"image": Link("source", "image"), "bboxes": Link("detect", "bboxes")},
                    ),
                    "canvas": GraphNode("DrawBBoxes", {"bboxes": Link("detect", "bboxes")}),
                    "inspect": GraphNode(
                        "rtdetr-fixture.inspect",
                        {
                            "model": Link("source", "model"),
                            "source": Link("source", "image"),
                            "bboxes": Link("detect", "bboxes"),
                            "image": Link("draw", "out_image"),
                            "canvas": Link("canvas", "out_image"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                output = (
                    (await engine.run(graph, ["inspect"])).outputs["inspect"]["summary"].resolve()
                )
                assert output == "ordered boxes and rendered pixels"
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.load_model_gpu = original_load
