"""VIDEO metadata edits never open the source or materialize frames."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest
from dinkster_values.video_edits import (
    crop_rectangle,
    effective_video_facts,
    scale_geometry,
    seconds,
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


def test_strict_duration() -> None:
    assert trim_window(Fraction(10), 2, 8, True) == (2, 8)
    with pytest.raises(ValueError, match="exceeds available"):
        trim_window(Fraction(10), 2, 9, True)


def test_fractional_seconds_do_not_accumulate_float_errors() -> None:
    assert seconds(0.1, "time") + seconds(0.2, "time") == Fraction(3, 10)
    assert seconds(Fraction(1001, 30000), "time") == Fraction(1001, 30000)


@pytest.mark.parametrize("value", [True, "1", None, float("inf"), float("nan")])
def test_invalid_seconds(value: object) -> None:
    with pytest.raises(ValueError, match="finite seconds"):
        seconds(value, "time")


def test_unknown_duration_stays_unknown() -> None:
    assert trim_window(None, 1, 3) == (1, None)
    assert effective_video_facts(_clip(duration=None))["duration"] is None
    with pytest.raises(ValueError, match="known duration"):
        trim_window(None, -1, 0)
    with pytest.raises(ValueError, match="known duration"):
        trim_window(None, 0, 3, True)


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


def test_full_odd_frame_crop_is_noop() -> None:
    assert crop_rectangle(1919, 1079, {"width": 1919, "height": 1079}) == (0, 0, 1919, 1079)


@pytest.mark.parametrize("fit", ["stretch", "crop", "pad"])
def test_identity_scale(fit: str) -> None:
    assert scale_geometry(1920, 1080, {"width": 1920, "height": 1080, "fit": fit}) == (
        1920,
        1080,
        0,
        0,
        1920,
        1080,
    )


def test_scale_geometry_uses_even_rounding() -> None:
    assert scale_geometry(1920, 1080, {"width": 1000, "height": 1000, "fit": "crop"}) == (
        1778,
        1000,
        389,
        0,
        1000,
        1000,
    )
    assert scale_geometry(1920, 1080, {"width": 1000, "height": 1000, "fit": "pad"}) == (
        1000,
        562,
        0,
        219,
        1000,
        1000,
    )


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


def test_rotation_precedes_crop_and_scale_order_matters() -> None:
    crop = {"crop": {"x": 100, "y": 40, "width": 600, "height": 800}}
    scale = {"scale": {"width": 300, "height": 200}}
    assert effective_video_facts(_clip(rotation=90))["width"] == 1080
    a = effective_video_facts(_clip(crop, scale, rotation=90))
    b = effective_video_facts(_clip(scale, crop, rotation=90))
    assert (a["width"], a["height"]) == (300, 200)
    assert (b["width"], b["height"]) == (200, 160)


def test_concat_effective_geometry_and_duration() -> None:
    crop = {"crop": {"width": 960, "height": 1080}}
    a = _clip(crop, {"concat": [_clip(crop)]})
    facts = effective_video_facts(a)
    assert (facts["width"], facts["height"], facts["duration"]) == (960, 1080, 20)
    assert facts["frame_count"] == 480
    assert facts["frame_count_kind"] == "header"
    with pytest.raises(ValueError, match="matching effective dimensions"):
        effective_video_facts(_clip(crop, {"concat": [_clip()]}))


def test_trim_after_concat_and_unknown_duration() -> None:
    clip = _clip({"concat": [_clip()]}, {"trim": {"start_time": 8, "duration": 4}})
    assert effective_video_facts(clip)["duration"] == 4
    assert effective_video_facts(clip)["frame_count_kind"] == "estimated"
    assert effective_video_facts(_clip({"concat": [_clip(duration=None)]}))["duration"] is None


def test_ambiguous_unknown_and_recursive_edits_fail() -> None:
    for edit in [{"trim": {}, "crop": {}}, {"trim": {}, "unknown": 1}, {"concat": []}]:
        with pytest.raises(ValueError):
            effective_video_facts(_clip(edit))
    clip = _clip()
    cast("list[object]", clip["edits"]).append({"concat": [clip]})
    with pytest.raises(ValueError, match="nesting"):
        effective_video_facts(clip)
