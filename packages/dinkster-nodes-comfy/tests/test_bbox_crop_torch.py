"""Observable crop geometry, padding and empty-input behavior on CPU."""

from __future__ import annotations

import pytest
import torch
from dinkster_nodes_comfy.bbox_crop import CropByBBoxes


def test_crop_unions_boxes_broadcasts_last_frame_and_preserves_source() -> None:
    image = torch.arange(3 * 6 * 8 * 4, dtype=torch.float32).reshape(3, 6, 8, 4)
    original = image.clone()
    boxes = [[{"x": 1, "y": 2, "width": 2, "height": 2}, {"x": 4, "y": 1, "width": 2, "height": 2}]]
    output = CropByBBoxes.execute(
        image=image.numpy(), bboxes=boxes, output_width=5, output_height=3, padding=0
    )["IMAGE"]
    torch.testing.assert_close(output, original[:, 1:4, 1:6], rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_crop_pad_keeps_aspect_and_centers_black_border() -> None:
    image = torch.tensor([[[[2.0], [4.0], [6.0], [8.0]], [[12.0], [14.0], [16.0], [18.0]]]])
    output = CropByBBoxes.execute(
        image=image,
        bboxes={"x": 0, "y": 0, "width": 4, "height": 2},
        output_width=4,
        output_height=6,
        padding=0,
        keep_aspect="pad",
    )["IMAGE"]
    expected = torch.zeros(1, 6, 4, 1)
    expected[:, 2:4] = image
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_crop_padding_clamps_and_skips_empty_frames() -> None:
    image = torch.arange(2 * 4 * 5 * 3, dtype=torch.float64).reshape(2, 4, 5, 3)
    output = CropByBBoxes.execute(
        image=image,
        bboxes=[[], [{"x": 0, "y": 1, "width": 2, "height": 1}]],
        output_width=3,
        output_height=3,
        padding=1,
    )["IMAGE"]
    torch.testing.assert_close(output, image[1:2, :3, :3], rtol=0, atol=0)
    assert output.dtype == torch.float64


@pytest.mark.parametrize("boxes", ([], [[]]))
def test_empty_crop_preserves_tensor_identity(boxes: list) -> None:
    image = torch.ones(2, 3, 7, 3)
    assert (
        CropByBBoxes.execute(image=image, bboxes=boxes, output_width=8, output_height=5, padding=0)[
            "IMAGE"
        ]
        is image
    )


def test_degenerate_box_uses_reference_fallback_and_tiny_black_result() -> None:
    box = {"x": 9, "y": 9, "width": 0, "height": 0}
    image = torch.arange(10 * 20, dtype=torch.float32).reshape(1, 10, 20, 1)
    output = CropByBBoxes.execute(
        image=image, bboxes=box, output_width=3, output_height=3, padding=0
    )["IMAGE"]
    torch.testing.assert_close(output, image[:, 1:4, 8:11], rtol=0, atol=0)
    tiny = CropByBBoxes.execute(
        image=torch.ones(1, 2, 2, 4), bboxes=box, output_width=5, output_height=3, padding=0
    )["IMAGE"]
    assert tiny.shape == (1, 3, 5, 4) and torch.count_nonzero(tiny) == 0
