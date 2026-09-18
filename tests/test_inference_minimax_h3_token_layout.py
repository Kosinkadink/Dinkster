"""MiniMax H3 torch-free packed token geometry tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference import (
    MiniMaxH3DiTPayloadKind,
    MiniMaxH3GuideTokenGeometry,
    MiniMaxH3KeyframeRole,
    MiniMaxH3ReferenceTokenGeometry,
    MiniMaxH3TokenLayoutError,
    MiniMaxH3TokenLayoutPlan,
    MiniMaxH3VideoLatentGeometry,
    TokenGridTransform,
    map_transforms,
    plan_minimax_h3_token_layout,
)


def test_t2va_layout_declares_odd_video_padding_and_target_transforms() -> None:
    plan = plan_minimax_h3_token_layout(
        text_tokens=7,
        target_video=MiniMaxH3VideoLatentGeometry(2, 3, 5),
        target_audio_temporal=8,
        sequence_shards=4,
    )

    assert tuple(
        (
            segment.identity,
            segment.modality,
            segment.role,
            segment.start,
            segment.stop,
            segment.grid,
        )
        for segment in plan.layout.segments
    ) == (
        ("text", "text", "context", 0, 7, (7,)),
        ("target-audio", "audio", "target", 7, 23, (2, 8)),
        ("target-video", "video", "target", 23, 35, (2, 2, 3)),
    )
    assert plan.layout.valid_rows == 35
    assert plan.layout.padded_rows == 1
    assert plan.layout.total_rows == 36
    assert plan.partition.sequence_length == 35
    assert plan.partition.chunk == 9
    assert tuple(
        (shard.start, shard.stop, shard.padded_rows) for shard in plan.partition.shards
    ) == ((0, 9, 0), (9, 18, 0), (18, 27, 0), (27, 35, 1))

    transforms = map_transforms(plan.layout, plan.transforms)
    assert set(transforms) == {"video", "audio"}
    assert transforms["video"] == TokenGridTransform(
        "minimax-h3.video-replicate-pad-2x2-amax.v1",
        "video",
        "target-video",
        (2, 3, 5),
        256,
    )
    assert transforms["audio"] == TokenGridTransform(
        "minimax-h3.audio-feature-amax.v1",
        "audio",
        "target-audio",
        (32, 2, 8),
        256,
    )


def test_fl2va_layout_matches_private_packed_layout_order() -> None:
    plan = plan_minimax_h3_token_layout(
        text_tokens=4,
        target_video=MiniMaxH3VideoLatentGeometry(7, 4, 6),
        target_audio_temporal=37,
        keyframes=(MiniMaxH3KeyframeRole.FIRST, MiniMaxH3KeyframeRole.LAST),
    )

    assert tuple(
        (segment.identity, segment.start, segment.stop, segment.grid)
        for segment in plan.layout.segments
    ) == (
        ("text", 0, 4, (4,)),
        ("keyframe-first", 4, 10, (1, 2, 3)),
        ("keyframe-last", 10, 16, (1, 2, 3)),
        ("target-audio", 16, 90, (2, 37)),
        ("target-video", 90, 132, (7, 2, 3)),
    )


def test_timeline_guides_declare_video_and_audio_rows_before_references() -> None:
    plan = plan_minimax_h3_token_layout(
        text_tokens=4,
        target_video=MiniMaxH3VideoLatentGeometry(7, 3, 5),
        target_audio_temporal=37,
        guides=(
            MiniMaxH3GuideTokenGeometry(
                6,
                5,
                MiniMaxH3VideoLatentGeometry(2, 3, 5),
                10,
            ),
        ),
        references=(
            MiniMaxH3ReferenceTokenGeometry(
                MiniMaxH3DiTPayloadKind.IMAGE,
                MiniMaxH3VideoLatentGeometry(1, 3, 5),
            ),
        ),
    )

    assert tuple(
        (segment.identity, segment.modality, segment.role, segment.start, segment.stop)
        for segment in plan.layout.segments
    ) == (
        ("text", "text", "context", 0, 4),
        ("guide-1-video", "video", "condition", 4, 16),
        ("guide-1-audio", "audio", "condition", 16, 36),
        ("reference-1-video", "video", "reference", 36, 42),
        ("target-audio", "audio", "target", 42, 116),
        ("target-video", "video", "target", 116, 158),
    )


def test_ref2va_layout_preserves_reference_order_and_audio_before_video() -> None:
    references = (
        MiniMaxH3ReferenceTokenGeometry(
            MiniMaxH3DiTPayloadKind.IMAGE,
            MiniMaxH3VideoLatentGeometry(1, 3, 3),
        ),
        MiniMaxH3ReferenceTokenGeometry(MiniMaxH3DiTPayloadKind.AUDIO, audio_temporal=2),
        MiniMaxH3ReferenceTokenGeometry(
            MiniMaxH3DiTPayloadKind.VIDEO,
            MiniMaxH3VideoLatentGeometry(3, 3, 5),
            3,
        ),
    )
    plan = plan_minimax_h3_token_layout(
        text_tokens=5,
        target_video=MiniMaxH3VideoLatentGeometry(2, 4, 4),
        target_audio_temporal=8,
        references=references,
    )

    assert tuple(
        (segment.identity, segment.start, segment.stop, segment.grid)
        for segment in plan.layout.segments
    ) == (
        ("text", 0, 5, (5,)),
        ("reference-1-video", 5, 9, (1, 2, 2)),
        ("reference-2-audio", 9, 13, (2, 2)),
        ("reference-3-audio", 13, 19, (2, 3)),
        ("reference-3-video", 19, 37, (3, 2, 3)),
        ("target-audio", 37, 53, (2, 8)),
        ("target-video", 53, 61, (2, 2, 2)),
    )


@pytest.mark.parametrize(
    ("temporal", "height", "width", "message"),
    (
        (0, 4, 4, "video temporal"),
        (2, 0, 4, "video height"),
        (2, 4, 0, "video width"),
        (True, 4, 4, "video temporal"),
        (2, 4.0, 4, "video height"),
    ),
)
def test_video_geometry_refuses_non_positive_exact_dimensions(
    temporal: object, height: object, width: object, message: str
) -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError, match=message):
        MiniMaxH3VideoLatentGeometry(temporal, height, width)  # type: ignore[arg-type]


def test_single_frame_target_matches_ordinary_latent_adaptation() -> None:
    plan = plan_minimax_h3_token_layout(
        text_tokens=1,
        target_video=MiniMaxH3VideoLatentGeometry(1, 2, 2),
        target_audio_temporal=2,
    )

    assert tuple(
        (segment.identity, segment.start, segment.stop, segment.grid)
        for segment in plan.layout.segments
    ) == (
        ("text", 0, 1, (1,)),
        ("target-audio", 1, 5, (2, 2)),
        ("target-video", 5, 6, (1, 1, 1)),
    )
    assert plan.transforms[0].source_geometry == (1, 2, 2)
    assert plan.transforms[1].source_geometry == (32, 2, 2)


def test_target_audio_must_match_decoded_video_duration() -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError, match="target video frame count"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=MiniMaxH3VideoLatentGeometry(7, 2, 2),
            target_audio_temporal=36,
        )


def test_timeline_guide_video_must_match_the_target_spatial_geometry() -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError, match="spatial geometry"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=MiniMaxH3VideoLatentGeometry(2, 3, 5),
            target_audio_temporal=8,
            guides=(
                MiniMaxH3GuideTokenGeometry(
                    0,
                    1,
                    MiniMaxH3VideoLatentGeometry(1, 3, 4),
                ),
            ),
        )


@pytest.mark.parametrize(
    "guides",
    (
        (
            MiniMaxH3GuideTokenGeometry(0, 5, MiniMaxH3VideoLatentGeometry(2, 3, 5)),
            MiniMaxH3GuideTokenGeometry(4, 5, MiniMaxH3VideoLatentGeometry(2, 3, 5)),
        ),
        (
            MiniMaxH3GuideTokenGeometry(0, 1, audio_temporal=6),
            MiniMaxH3GuideTokenGeometry(3, 1, MiniMaxH3VideoLatentGeometry(1, 3, 5)),
        ),
    ),
)
def test_timeline_guides_refuse_overlapping_video_and_audio_ranges(
    guides: tuple[MiniMaxH3GuideTokenGeometry, ...],
) -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError, match="guide 2 overlaps guide 1"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=MiniMaxH3VideoLatentGeometry(7, 3, 5),
            target_audio_temporal=37,
            guides=guides,
        )


@pytest.mark.parametrize(
    "reference",
    (
        lambda: MiniMaxH3ReferenceTokenGeometry(
            MiniMaxH3DiTPayloadKind.IMAGE,
            MiniMaxH3VideoLatentGeometry(2, 2, 2),
        ),
        lambda: MiniMaxH3ReferenceTokenGeometry(
            MiniMaxH3DiTPayloadKind.IMAGE,
            MiniMaxH3VideoLatentGeometry(1, 2, 2),
            1,
        ),
        lambda: MiniMaxH3ReferenceTokenGeometry(MiniMaxH3DiTPayloadKind.AUDIO),
        lambda: MiniMaxH3ReferenceTokenGeometry(
            MiniMaxH3DiTPayloadKind.AUDIO,
            MiniMaxH3VideoLatentGeometry(1, 2, 2),
            1,
        ),
        lambda: MiniMaxH3ReferenceTokenGeometry(MiniMaxH3DiTPayloadKind.VIDEO),
    ),
)
def test_reference_geometry_refuses_invalid_kind_shapes(reference: object) -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError):
        reference()  # type: ignore[operator]


def test_keyframes_are_unique_and_mutually_exclusive_with_references() -> None:
    target = MiniMaxH3VideoLatentGeometry(2, 2, 2)
    with pytest.raises(MiniMaxH3TokenLayoutError, match="unique"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=target,
            target_audio_temporal=8,
            keyframes=(MiniMaxH3KeyframeRole.FIRST, MiniMaxH3KeyframeRole.FIRST),
        )
    with pytest.raises(MiniMaxH3TokenLayoutError, match="mutually exclusive"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=target,
            target_audio_temporal=8,
            keyframes=(MiniMaxH3KeyframeRole.FIRST,),
            references=(
                MiniMaxH3ReferenceTokenGeometry(
                    MiniMaxH3DiTPayloadKind.IMAGE,
                    MiniMaxH3VideoLatentGeometry(1, 2, 2),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"text_tokens": 0}, "text tokens"),
        ({"text_tokens": True}, "text tokens"),
        ({"target_audio_temporal": 8.0}, "target audio temporal"),
        ({"sequence_shards": 0}, "sequence shards"),
        ({"keyframes": [MiniMaxH3KeyframeRole.FIRST]}, "keyframes"),
        ({"guides": []}, "guides"),
        ({"references": []}, "references"),
    ),
)
def test_planner_refuses_invalid_boundary_types(changes: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "text_tokens": 1,
        "target_video": MiniMaxH3VideoLatentGeometry(2, 2, 2),
        "target_audio_temporal": 8,
    }
    values.update(changes)
    with pytest.raises(MiniMaxH3TokenLayoutError, match=message):
        plan_minimax_h3_token_layout(**values)  # type: ignore[arg-type]


def test_partition_infeasibility_is_a_family_layout_error() -> None:
    with pytest.raises(MiniMaxH3TokenLayoutError, match="shard_count"):
        plan_minimax_h3_token_layout(
            text_tokens=1,
            target_video=MiniMaxH3VideoLatentGeometry(2, 2, 2),
            target_audio_temporal=8,
            sequence_shards=20,
        )


def test_plan_is_immutable_and_factory_owned() -> None:
    plan = plan_minimax_h3_token_layout(
        text_tokens=1,
        target_video=MiniMaxH3VideoLatentGeometry(2, 2, 2),
        target_audio_temporal=8,
    )
    with pytest.raises(FrozenInstanceError):
        plan.transforms = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="produced only by the family planner"):
        MiniMaxH3TokenLayoutPlan(plan.layout, plan.transforms, plan.partition)


def test_layout_and_transform_digests_are_deterministic_and_geometry_sensitive() -> None:
    first = plan_minimax_h3_token_layout(
        text_tokens=2,
        target_video=MiniMaxH3VideoLatentGeometry(2, 3, 4),
        target_audio_temporal=8,
    )
    second = plan_minimax_h3_token_layout(
        text_tokens=2,
        target_video=MiniMaxH3VideoLatentGeometry(2, 3, 4),
        target_audio_temporal=8,
    )
    changed = plan_minimax_h3_token_layout(
        text_tokens=3,
        target_video=MiniMaxH3VideoLatentGeometry(2, 3, 4),
        target_audio_temporal=8,
    )

    assert first == second
    assert first.layout.digest == second.layout.digest
    assert tuple(transform.digest for transform in first.transforms) == tuple(
        transform.digest for transform in second.transforms
    )
    assert first.layout.digest != changed.layout.digest
