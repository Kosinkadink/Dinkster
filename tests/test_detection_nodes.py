from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from dinkster_nodes_foundation import ListElement
from dinkster_nodes_image import (
    Detection,
    DetectionFilter,
    DetectionInfo,
    DetectionSort,
    DetectionsToMasks,
    MakeDetection,
    Region,
)


def _detection(
    label: str = "object",
    score: float = 1.0,
    region: Region | None = None,
    mask: np.ndarray | None = None,
) -> Detection:
    return Detection(
        label=label,
        score=score,
        region=Region(0, 0, 2, 2) if region is None else region,
        mask=mask,
    )


def test_make_detection_without_mask() -> None:
    result = MakeDetection.execute(region=Region(1, 2, 3, 4), label="cat", score=0.5)
    assert result == {"detection": Detection("cat", 0.5, Region(1, 2, 3, 4))}


def test_make_detection_accepts_a_single_mask() -> None:
    mask = np.zeros((1, 3, 4), dtype=np.float32)
    mask[0, 1, 2] = 1.0
    result = MakeDetection.execute(region=Region(0, 0, 4, 3), mask=mask)
    detection = cast("Detection", result["detection"])
    assert detection.mask is not None
    assert detection.mask.shape == (3, 4)
    assert detection.mask[1, 2] == 1.0
    with pytest.raises(ValueError, match="single mask"):
        MakeDetection.execute(region=Region(0, 0, 4, 3), mask=np.zeros((2, 3, 4), np.float32))


def test_detection_info_reports_fields() -> None:
    detection = _detection("cat", 0.5, Region(1, 2, 3, 4))
    assert DetectionInfo.execute(detection=detection) == {
        "region": Region(1, 2, 3, 4),
        "label": "cat",
        "score": 0.5,
        "area": 12.0,
        "has_mask": False,
    }
    masked = _detection(mask=np.zeros((2, 2), np.float32))
    assert DetectionInfo.execute(detection=masked)["has_mask"] is True


def test_detection_filter_applies_all_bounds_in_order() -> None:
    detections = [
        _detection("cat", 0.9, Region(0, 0, 2, 2)),
        _detection("dog", 0.8, Region(0, 0, 4, 4)),
        _detection("cat", 0.3, Region(0, 0, 6, 6)),
        _detection("bird", 0.95, Region(0, 0, 10, 10)),
    ]
    result = DetectionFilter.execute(detections=detections)
    assert result == {"detections": detections, "count": 4}
    by_label = DetectionFilter.execute(detections=detections, labels="cat, dog")
    assert by_label["count"] == 3
    by_score = DetectionFilter.execute(detections=detections, min_score=0.5)
    assert cast("list[Detection]", by_score["detections"]) == [
        detections[0],
        detections[1],
        detections[3],
    ]
    by_area = DetectionFilter.execute(detections=detections, min_area=10.0, max_area=50.0)
    assert cast("list[Detection]", by_area["detections"]) == [detections[1], detections[2]]
    truncated = DetectionFilter.execute(detections=detections, max_results=2)
    assert cast("list[Detection]", truncated["detections"]) == detections[:2]
    with pytest.raises(TypeError, match=r"detections\[0\] must be a detection"):
        DetectionFilter.execute(detections=["not a detection"])


def test_detection_sort_is_stable_per_key() -> None:
    first = _detection("b", 0.5, Region(4, 0, 2, 2))
    second = _detection("a", 0.9, Region(0, 4, 4, 4))
    third = _detection("c", 0.9, Region(2, 2, 1, 1))
    detections = [first, second, third]
    by_score = DetectionSort.execute(detections=detections)
    assert cast("list[Detection]", by_score["detections"]) == [second, third, first]
    by_area = DetectionSort.execute(detections=detections, key="area", descending=False)
    assert cast("list[Detection]", by_area["detections"]) == [third, first, second]
    by_left = DetectionSort.execute(detections=detections, key="left", descending=False)
    assert cast("list[Detection]", by_left["detections"]) == [second, third, first]
    by_top = DetectionSort.execute(detections=detections, key="top", descending=False)
    assert cast("list[Detection]", by_top["detections"]) == [first, third, second]
    by_label = DetectionSort.execute(detections=detections, key="label", descending=False)
    assert cast("list[Detection]", by_label["detections"]) == [second, first, third]
    with pytest.raises(ValueError, match="unknown detection sort key"):
        DetectionSort.execute(detections=detections, key="bogus")


def test_detections_to_masks_rasterizes_boxes() -> None:
    detections = [
        _detection(region=Region(1, 1, 2, 1)),
        _detection(region=Region(2.4, 0.6, 1.0, 1.0)),
    ]
    result = DetectionsToMasks.execute(detections=detections, width=5, height=4)
    masks = cast("list[np.ndarray]", result["masks"])
    combined = cast("np.ndarray", result["combined"])
    assert len(masks) == 2
    expected_first = np.zeros((1, 4, 5), np.float32)
    expected_first[0, 1:2, 1:3] = 1.0
    np.testing.assert_array_equal(masks[0], expected_first)
    expected_second = np.zeros((1, 4, 5), np.float32)
    expected_second[0, 0:2, 2:4] = 1.0
    np.testing.assert_array_equal(masks[1], expected_second)
    np.testing.assert_array_equal(combined, np.maximum(expected_first, expected_second))


def test_detections_to_masks_clips_boxes_whose_extent_overflows() -> None:
    result = DetectionsToMasks.execute(
        detections=[
            _detection(region=Region(1e308, 0, 1e308, 1)),
            _detection(region=Region(0, 1e308, 1, 1e308)),
            _detection(region=Region(1, 1, 1e308, 1e308)),
            _detection(region=Region(10**308, 0, 10**308, 1)),
            _detection(region=Region(0, 10**308, 1, 10**308)),
            _detection(region=Region(0, 0, 10**308, 10**308)),
        ],
        width=4,
        height=4,
    )
    masks = cast("list[np.ndarray]", result["masks"])
    empty = np.zeros((1, 4, 4), np.float32)
    np.testing.assert_array_equal(masks[0], empty)
    np.testing.assert_array_equal(masks[1], empty)
    expected = np.zeros((1, 4, 4), np.float32)
    expected[0, 1:4, 1:4] = 1.0
    np.testing.assert_array_equal(masks[2], expected)
    np.testing.assert_array_equal(masks[3], empty)
    np.testing.assert_array_equal(masks[4], empty)
    np.testing.assert_array_equal(masks[5], np.ones((1, 4, 4), np.float32))


def test_detections_to_masks_leaves_zero_area_boxes_empty() -> None:
    result = DetectionsToMasks.execute(
        detections=[
            _detection(region=Region(1.2, 1.2, 0, 0)),
            _detection(region=Region(1.2, 1.2, 0, 2)),
            _detection(region=Region(1.2, 1.2, 2, 0)),
        ],
        width=4,
        height=4,
    )
    empty = np.zeros((1, 4, 4), np.float32)
    for mask in cast("list[np.ndarray]", result["masks"]):
        np.testing.assert_array_equal(mask, empty)
    np.testing.assert_array_equal(cast("np.ndarray", result["combined"]), empty)


def test_detections_to_masks_clips_out_of_frame_boxes() -> None:
    result = DetectionsToMasks.execute(
        detections=[
            _detection(region=Region(-2, -2, 3, 3)),
            _detection(region=Region(10, 10, 5, 5)),
        ],
        width=4,
        height=4,
    )
    masks = cast("list[np.ndarray]", result["masks"])
    expected = np.zeros((1, 4, 4), np.float32)
    expected[0, 0:1, 0:1] = 1.0
    np.testing.assert_array_equal(masks[0], expected)
    np.testing.assert_array_equal(masks[1], np.zeros((1, 4, 4), np.float32))


def test_detections_to_masks_uses_carried_masks_and_checks_shape() -> None:
    mask = np.zeros((4, 5), np.float32)
    mask[2, 3] = 0.5
    result = DetectionsToMasks.execute(
        detections=[_detection(region=Region(3, 2, 1, 1), mask=mask)], width=5, height=4
    )
    np.testing.assert_array_equal(
        cast("list[np.ndarray]", result["masks"])[0], mask.reshape(1, 4, 5)
    )
    with pytest.raises(ValueError, match="does not match"):
        DetectionsToMasks.execute(
            detections=[_detection(region=Region(0, 0, 1, 1), mask=mask)], width=6, height=4
        )


def test_detections_to_masks_dilation_grows_and_erodes() -> None:
    detections = [_detection(region=Region(2, 2, 1, 1))]
    grown = DetectionsToMasks.execute(detections=detections, width=5, height=5, dilation=1)
    grown_mask = cast("list[np.ndarray]", grown["masks"])[0]
    expected = np.zeros((1, 5, 5), np.float32)
    expected[0, 2, 1:4] = 1.0
    expected[0, 1:4, 2] = 1.0
    np.testing.assert_array_equal(grown_mask, expected)
    eroded = DetectionsToMasks.execute(
        detections=[_detection(region=Region(1, 1, 3, 3))], width=5, height=5, dilation=-1
    )
    eroded_mask = cast("list[np.ndarray]", eroded["masks"])[0]
    expected_eroded = np.zeros((1, 5, 5), np.float32)
    expected_eroded[0, 2, 2] = 1.0
    np.testing.assert_array_equal(eroded_mask, expected_eroded)


def test_detections_to_masks_handles_an_empty_list() -> None:
    result = DetectionsToMasks.execute(detections=[], width=3, height=2)
    assert result["masks"] == []
    np.testing.assert_array_equal(
        cast("np.ndarray", result["combined"]), np.zeros((1, 2, 3), np.float32)
    )


def test_list_element_selects_a_detection() -> None:
    detections = [_detection("cat"), _detection("dog")]
    filtered = DetectionFilter.execute(detections=detections, labels="dog")
    selected = ListElement.execute(list=cast("list[object]", filtered["detections"]), index=0)
    assert cast("Detection", selected["item"]).label == "dog"
