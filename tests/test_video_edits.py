"""VIDEO metadata edits never open the source or materialize frames."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest
from dinkster_values.video_edits import (
    crop_rectangle,
    effective_video_facts,
    trim_window,
)


def _clip(*edits: object, **probe: object) -> dict[str, object]:
    return {
        "source": object(),
        "probe": {
            "width": 1920,
            "height": 1080,
            "rotation": 0,
            "duration": Fraction(10),
            "fps": Fraction(24),
            "frame_count": 240,
            "frame_count_kind": "header",
            **probe,
        },
        "edits": list(edits),
    }


@pytest.mark.parametrize(
    ("start", "duration", "expected"),
    [
        (0, 0, (0, 10)),
        (2, 0, (2, 8)),
        (-2, 0, (8, 2)),
        (-20, 3, (0, 3)),
        (8, 20, (8, 2)),
        (1.25, 3.5, (Fraction(5, 4), Fraction(7, 2))),
    ],
)
def test_trim_window(start: object, duration: object, expected: tuple[object, object]) -> None:
    assert trim_window(Fraction(10), start, duration) == expected


@pytest.mark.parametrize("start", [10, 11, 100])
def test_empty_trim(start: int) -> None:
    with pytest.raises(ValueError, match="no frames"):
        trim_window(Fraction(10), start, 0)


@pytest.mark.parametrize(
    ("rect", "expected"),
    [
        ({}, (0, 0, 1920, 1080)),
        ({"width": 0, "height": 12}, (0, 0, 1920, 1080)),
        ({"width": 12, "height": -1}, (0, 0, 1920, 1080)),
        ({"x": 101, "y": 41, "width": 1281, "height": 721}, (100, 40, 1280, 720)),
        ({"x": -20, "y": -10, "width": 200, "height": 100}, (0, 0, 200, 100)),
        ({"x": 1919, "y": 1079, "width": 200, "height": 100}, (1918, 1078, 2, 2)),
        ({"x": 2000, "y": 1100, "width": 200, "height": 100}, (1918, 1078, 2, 2)),
        ({"width": 1, "height": 10}, (0, 0, 1920, 1080)),
    ],
)
def test_crop_pixels(rect: dict[str, object], expected: tuple[int, int, int, int]) -> None:
    assert crop_rectangle(1920, 1080, rect) == expected


def test_composed_trims_intersect_current_clip_without_reading_source() -> None:
    clip = _clip(
        {"trim": {"start_time": 2, "duration": 3}},
        {"trim": {"start_time": 1, "duration": 0}},
    )
    assert effective_video_facts(clip)["duration"] == 2
    assert cast("dict[str, object]", clip["probe"])["duration"] == 10
    clip = _clip(
        {"trim": {"start_time": 2, "duration": 3}},
        {"trim": {"start_time": -1, "duration": 0}},
    )
    assert effective_video_facts(clip)["duration"] == 1
