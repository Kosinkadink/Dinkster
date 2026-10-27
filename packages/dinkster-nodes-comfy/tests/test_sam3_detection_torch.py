"""CPU prompt routing, score boundaries and mask geometry for SAM3 detection."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.sam3_detection import SAM3_Detect


@pytest.fixture(autouse=True)
def cpu_management(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(management, "intermediate_device", lambda: torch.device("cpu"))


def model(sam: Any) -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(get_dtype=lambda: torch.float32, diffusion_model=sam)
    )


@pytest.mark.parametrize("individual", (False, True))
def test_no_prompts_returns_empty_detections_and_correct_mask_batch(individual: bool) -> None:
    image = torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4) / 120
    original = image.clone()
    result = SAM3_Detect.execute(
        model=model(None), image=image.numpy(), individual_masks=individual
    )
    assert result["bboxes"] == [[], []]
    assert result["masks"].shape == ((0, 3, 5) if individual else (2, 3, 5))
    assert torch.count_nonzero(result["masks"]) == 0
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("multiple", (False, True))
@pytest.mark.parametrize("individual", (False, True))
def test_text_filters_strictly_orders_limits_each_prompt_and_preserves_source(
    multiple: bool, individual: bool
) -> None:
    class Detector:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, frame: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.calls += 1
            assert frame.shape == (1, 3, 1008, 1008)
            assert kwargs["orig_size"] == (3, 5) and kwargs["threshold"] == 0.5
            assert kwargs["text_embeddings"].shape == (1, 3, 2)
            torch.testing.assert_close(kwargs["text_mask"], torch.ones(1, 3, dtype=torch.int64))
            return {
                "boxes": torch.tensor([[[0, 0, 2, 1], [1, 0, 4, 2], [2, 1, 5, 3], [0, 1, 1, 3]]]),
                "scores": torch.tensor([[0.0, 1.0, 2.0, -1.0]]),
                "masks": torch.ones(1, 4, 2, 2),
            }

    sam = Detector()
    embedding = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    metadata = (
        {
            "sam3_multi_cond": [
                {"cond": embedding, "attention_mask": None, "max_detections": 2},
                {"cond": embedding + 5, "attention_mask": None, "max_detections": 1},
            ]
        }
        if multiple
        else {}
    )
    image = torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4) / 120
    original = image.clone()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=image,
        conditioning=[[embedding, metadata]],
        refine_iterations=0,
        individual_masks=individual,
    )
    assert sam.calls == (4 if multiple else 2)
    expected_boxes = [(2.0, 1.0, 3.0, 2.0)]
    if multiple:
        expected_boxes += [(1.0, 0.0, 3.0, 2.0), (2.0, 1.0, 3.0, 2.0)]
    for boxes in result["bboxes"]:
        assert [(b["x"], b["y"], b["width"], b["height"]) for b in boxes] == expected_boxes
        assert boxes[0]["score"] == float(torch.tensor(2.0).sigmoid())
    expected_count = 6 if multiple and individual else 2
    assert result["masks"].shape == (expected_count, 3, 5)
    assert torch.all(result["masks"] == 1)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("iterations", (0, 3))
def test_points_scale_pixel_coordinates_and_keep_positive_negative_labels(iterations: int) -> None:
    class Segmenter:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def forward_segment(self, frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls.append(kwargs)
            if "point_inputs" in kwargs:
                points = kwargs["point_inputs"]
                torch.testing.assert_close(
                    points["point_coords"], torch.tensor([[[201.6, 672.0], [806.4, 336.0]]])
                )
                assert points["point_labels"].tolist() == [[1, 0]]
                assert points["point_labels"].dtype == torch.int32
            else:
                assert kwargs["mask_inputs"].shape == (1, 1, 3, 5)
            return torch.tensor([[[[-1.0, -1.0, 1.0, 1.0, 1.0]]]]).expand(1, 1, 3, 5)

    sam = Segmenter()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=torch.zeros(1, 3, 5, 3),
        positive_coords='[{"x": 1, "y": 2}]',
        negative_coords='[{"x": 4, "y": 1}]',
        refine_iterations=iterations,
    )
    assert len(sam.calls) == max(1, iterations)
    assert result["bboxes"] == [[]]
    torch.testing.assert_close(
        result["masks"], torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0]]]).expand(1, 3, 5)
    )


def test_per_frame_boxes_pad_last_frame_and_empty_frames_skip_segmentation() -> None:
    corners: list[torch.Tensor] = []

    def segment(frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        corners.append(kwargs["box_inputs"])
        return torch.ones(1, 1, 3, 5)

    result = SAM3_Detect.execute(
        model=model(SimpleNamespace(forward_segment=segment)),
        image=torch.zeros(3, 3, 5, 3),
        bboxes=[[], [{"x": 1, "y": 1, "width": 3, "height": 2}]],
        refine_iterations=0,
    )
    assert len(corners) == 2
    expected = torch.tensor([[[201.6, 336.0], [806.4, 1008.0]]])
    for box in corners:
        torch.testing.assert_close(box, expected)
    assert torch.count_nonzero(result["masks"][0]) == 0
    assert torch.all(result["masks"][1:] == 1)


@pytest.mark.parametrize("iterations", (1, 2))
def test_text_refinement_crops_original_pixels_and_unions_coarse_masks(iterations: int) -> None:
    class Refiner:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, frame: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            coarse = torch.full((1, 1, 6, 8), -1.0)
            coarse[0, 0, 5, 7] = 1
            return {
                "boxes": torch.tensor([[[2.0, 1.0, 6.0, 4.0]]]),
                "scores": torch.tensor([[2.0]]),
                "masks": coarse,
            }

        def forward_segment(self, frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls += 1
            assert frame.shape == (1, 3, 1008, 1008)
            assert torch.all(frame[:, 0] == 0.25) and torch.all(frame[:, 1] == 0.75)
            assert kwargs["mask_inputs"].shape == (1, 1, 1008, 1008)
            return torch.ones(1, 1, 2, 3)

    image = torch.ones(1, 6, 8, 3)
    image[..., 0], image[..., 1] = 0.25, 0.75
    sam = Refiner()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=image,
        conditioning=[[torch.ones(1, 2, 3), {}]],
        refine_iterations=iterations,
    )
    expected = torch.zeros(1, 6, 8)
    expected[:, 0:4, 1:6] = 1
    expected[:, 5, 7] = 1
    torch.testing.assert_close(result["masks"], expected, rtol=0, atol=0)
    assert sam.calls == iterations
