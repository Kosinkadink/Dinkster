from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference import (
    MINIMAX_H3_AUDIO_MASK_MAPPING,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    CodecTemporalMapping,
    LatentMaskError,
    LatentMaskMapping,
    parse_frame_ranges,
    parse_time_ranges,
)


def test_minimax_h3_mask_mappings_declare_exact_content_geometry() -> None:
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.role == "video"
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.temporal.content_frames_per_latent == (
        1,
        4,
        4,
        4,
        4,
    )
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.content_rate_hz == 24
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.spatial_downscale == 16
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.content_ranges(7) == (
        (0, 1),
        (1, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (17, 18),
        (18, 22),
    )
    assert MINIMAX_H3_VIDEO_MASK_MAPPING.duration_seconds(7) == 22 / 24

    assert MINIMAX_H3_AUDIO_MASK_MAPPING.role == "audio"
    assert MINIMAX_H3_AUDIO_MASK_MAPPING.temporal.content_frames_per_latent == (1,)
    assert MINIMAX_H3_AUDIO_MASK_MAPPING.content_rate_hz == 40
    assert MINIMAX_H3_AUDIO_MASK_MAPPING.spatial_downscale is None
    assert MINIMAX_H3_AUDIO_MASK_MAPPING.duration_seconds(80) == 2.0


def test_latent_mask_mapping_is_frozen_and_validated() -> None:
    mapping = LatentMaskMapping("video", CodecTemporalMapping((1, 2)), 24.0, 8)
    with pytest.raises(FrozenInstanceError):
        mapping.role = "audio"  # type: ignore[misc]
    with pytest.raises(LatentMaskError, match="nonempty"):
        LatentMaskMapping("", CodecTemporalMapping((1,)), 24.0)
    with pytest.raises(LatentMaskError, match="finite and positive"):
        LatentMaskMapping("video", CodecTemporalMapping((1,)), float("inf"))
    with pytest.raises(LatentMaskError, match="positive integer"):
        LatentMaskMapping("video", CodecTemporalMapping((1,)), 24.0, 0)


def test_frame_ranges_support_indices_python_slices_negatives_and_end() -> None:
    assert parse_frame_ranges("0, 2:8:2\n-2, -1:end", 10) == (0, 2, 4, 6, 8, 9)
    assert parse_frame_ranges("::-3", 8) == (1, 4, 7)
    assert parse_frame_ranges("", 8) == ()


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("8", "outside"),
        ("end", "invalid frame index"),
        ("end:4", "only as a slice stop"),
        ("1:4:0", "must not be zero"),
        ("1:2:3:4", "invalid frame slice"),
    ),
)
def test_frame_ranges_refuse_invalid_or_unbounded_entries(value: str, message: str) -> None:
    with pytest.raises(LatentMaskError, match=message):
        parse_frame_ranges(value, 8)


def test_time_ranges_support_open_end_relative_and_clamped_bounds() -> None:
    assert parse_time_ranges("0:0.5, 1:end\n-0.25:", 2.0) == (
        (0.0, 0.5),
        (1.0, 2.0),
        (1.75, 2.0),
    )
    assert parse_time_ranges("-4:4", 2.0) == ((0.0, 2.0),)
    assert parse_time_ranges("", 2.0) == ()


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("0.5", "must be start:stop"),
        ("end:1", "only as a time-range stop"),
        ("1:0", "selects no time"),
        ("nan:1", "must be finite"),
    ),
)
def test_time_ranges_refuse_invalid_entries(value: str, message: str) -> None:
    with pytest.raises(LatentMaskError, match=message):
        parse_time_ranges(value, 2.0)
