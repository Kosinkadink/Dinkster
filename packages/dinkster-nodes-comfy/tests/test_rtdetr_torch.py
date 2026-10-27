"""Controlled CPU outputs for native RT-DETR nodes."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.rtdetr import DrawBBoxes, RTDETR_detect


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
