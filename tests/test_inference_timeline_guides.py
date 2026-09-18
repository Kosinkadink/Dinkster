from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference import (
    CodecTemporalMapping,
    MultiStreamLatent,
    TimelineGuide,
    TimelineGuideError,
    resolve_timeline_frame_index,
)


def test_codec_temporal_mapping_repeats_groups_and_maps_timeline_positions() -> None:
    mapping = CodecTemporalMapping((1, 4, 4, 4, 4), 5, 3)

    assert tuple(mapping.content_extent(size) for size in (1, 2, 5, 7, 12)) == (
        1,
        5,
        17,
        22,
        39,
    )
    assert mapping.timeline_position(6) == 10.0
    assert mapping.timeline_position(7) == (5.0 / 3.0) * 7


def test_timeline_guide_resolves_end_relative_anchor_and_validates_target_fit() -> None:
    latent = MultiStreamLatent.from_pairs((("video", object()), ("audio", object())))
    guide = TimelineGuide(resolve_timeline_frame_index(-5, 22), 5, latent)

    assert guide.frame_index == 17
    assert guide.latent.roles == ("video", "audio")
    guide.validate_for_target(22)
    with pytest.raises(TimelineGuideError, match="do not fit"):
        TimelineGuide(18, 5, latent).validate_for_target(22)
    with pytest.raises(TimelineGuideError, match="outside"):
        resolve_timeline_frame_index(-23, 22)
    with pytest.raises(FrozenInstanceError):
        guide.frame_index = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "mapping",
    (
        CodecTemporalMapping((1,)),
        CodecTemporalMapping((1, 4), 5, 3),
    ),
)
def test_codec_temporal_mapping_requires_positive_extents(mapping: CodecTemporalMapping) -> None:
    with pytest.raises(TimelineGuideError, match="positive integer"):
        mapping.content_extent(0)
