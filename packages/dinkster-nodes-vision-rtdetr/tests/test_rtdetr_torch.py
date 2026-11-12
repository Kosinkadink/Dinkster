"""Controlled CPU outputs for native RT-DETR nodes."""

from __future__ import annotations

import torch
from dinkster_vision_rtdetr.rtdetr import DrawBBoxes


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
