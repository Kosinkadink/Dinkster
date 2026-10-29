"""Torch-free MiniMax H3 packed-sequence geometry declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .minimax_h3 import (
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_VIDEO_TEMPORAL_MAPPING,
    MiniMaxH3DiTPayloadKind,
    MiniMaxH3KeyframeRole,
)
from .sequence_partition import (
    SequencePartition,
    SequencePartitionError,
    plan_sequence_partition,
)
from .token_layout import (
    ModelTokenLayout,
    ModelTokenSegment,
    TokenGridTransform,
)

__all__ = [
    "MiniMaxH3GuideTokenGeometry",
    "MiniMaxH3ReferenceTokenGeometry",
    "MiniMaxH3TokenLayoutError",
    "MiniMaxH3TokenLayoutPlan",
    "MiniMaxH3VideoLatentGeometry",
    "plan_minimax_h3_token_layout",
    "validate_minimax_h3_guide_timeline",
]

_VIDEO_TRANSFORM = "minimax-h3.video-replicate-pad-2x2-amax.v1"
_AUDIO_TRANSFORM = "minimax-h3.audio-feature-amax.v1"
_QUANTIZATION_LEVELS = 256


class MiniMaxH3TokenLayoutError(ValueError):
    pass


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise MiniMaxH3TokenLayoutError(f"{name} must be an exact int >= 1")


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoLatentGeometry:
    """Unpadded H3 video latent geometry in temporal, height, width order."""

    temporal: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name, value in (
            ("video temporal", self.temporal),
            ("video height", self.height),
            ("video width", self.width),
        ):
            _positive_int(name, value)

    @property
    def token_grid(self) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = MINIMAX_H3_CONFIG.patch
        return (
            -(-self.temporal // patch_t),
            -(-self.height // patch_h),
            -(-self.width // patch_w),
        )


@dataclass(frozen=True, slots=True)
class MiniMaxH3ReferenceTokenGeometry:
    """Declared latent geometry for one ordered H3 reference block."""

    kind: MiniMaxH3DiTPayloadKind
    video: MiniMaxH3VideoLatentGeometry | None = None
    audio_temporal: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not MiniMaxH3DiTPayloadKind:
            raise MiniMaxH3TokenLayoutError("reference kind must be exact MiniMaxH3DiTPayloadKind")
        if self.video is not None and type(self.video) is not MiniMaxH3VideoLatentGeometry:
            raise MiniMaxH3TokenLayoutError(
                "reference video must be exact MiniMaxH3VideoLatentGeometry"
            )
        if self.audio_temporal is not None:
            _positive_int("reference audio temporal", self.audio_temporal)
        if self.kind is MiniMaxH3DiTPayloadKind.IMAGE:
            if self.video is None or self.video.temporal != 1 or self.audio_temporal is not None:
                raise MiniMaxH3TokenLayoutError(
                    "image references require one video latent frame and no audio"
                )
        elif self.kind is MiniMaxH3DiTPayloadKind.AUDIO:
            if self.video is not None or self.audio_temporal is None:
                raise MiniMaxH3TokenLayoutError("audio references require only audio geometry")
        elif self.video is None:
            raise MiniMaxH3TokenLayoutError("video references require video geometry")


@dataclass(frozen=True, slots=True)
class MiniMaxH3GuideTokenGeometry:
    """Declared latent geometry for one target-relative timeline guide."""

    frame_index: int
    frame_count: int
    video: MiniMaxH3VideoLatentGeometry | None = None
    audio_temporal: int | None = None

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise MiniMaxH3TokenLayoutError("guide frame index must be an exact int >= 0")
        _positive_int("guide frame count", self.frame_count)
        if self.video is not None and type(self.video) is not MiniMaxH3VideoLatentGeometry:
            raise MiniMaxH3TokenLayoutError(
                "guide video must be exact MiniMaxH3VideoLatentGeometry"
            )
        if self.audio_temporal is not None:
            _positive_int("guide audio temporal", self.audio_temporal)
        if self.video is None and self.audio_temporal is None:
            raise MiniMaxH3TokenLayoutError("guides require video and/or audio geometry")
        if (
            self.video is not None
            and MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(self.video.temporal)
            != self.frame_count
        ):
            raise MiniMaxH3TokenLayoutError(
                "guide frame count must match its video latent temporal geometry"
            )


def validate_minimax_h3_guide_timeline(
    guides: tuple[MiniMaxH3GuideTokenGeometry, ...],
) -> None:
    """Refuse guides whose video or audio conditions occupy the same timeline range."""

    if type(guides) is not tuple or any(
        type(guide) is not MiniMaxH3GuideTokenGeometry for guide in guides
    ):
        raise MiniMaxH3TokenLayoutError(
            "guides must be a tuple of exact MiniMaxH3GuideTokenGeometry values"
        )
    occupied: list[tuple[int, int, int]] = []
    numerator = MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_numerator
    denominator = MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_denominator
    for ordinal, guide in enumerate(guides, start=1):
        start = numerator * guide.frame_index
        stops: list[int] = []
        if guide.video is not None:
            stops.append(numerator * (guide.frame_index + guide.frame_count))
        if guide.audio_temporal is not None:
            stops.append(start + denominator * guide.audio_temporal)
        stop = max(stops)
        for other_ordinal, other_start, other_stop in occupied:
            if start < other_stop and other_start < stop:
                raise MiniMaxH3TokenLayoutError(
                    f"guide {ordinal} overlaps guide {other_ordinal} on the target timeline"
                )
        occupied.append((ordinal, start, stop))


@dataclass(frozen=True, slots=True, init=False)
class MiniMaxH3TokenLayoutPlan:
    """One canonical H3 global layout, transforms, and sequence partition."""

    layout: ModelTokenLayout
    transforms: tuple[TokenGridTransform, ...]
    partition: SequencePartition

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("MiniMax H3 token layout plans are produced only by the family planner")


def _target_frame_count(video: MiniMaxH3VideoLatentGeometry) -> int:
    return MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(video.temporal)


def _segment(
    identity: str,
    modality: str,
    role: str,
    offset: int,
    grid: tuple[int, ...],
) -> ModelTokenSegment:
    rows = 1
    for size in grid:
        rows *= size
    return ModelTokenSegment(identity, modality, role, offset, offset + rows, grid)


def plan_minimax_h3_token_layout(
    *,
    text_tokens: int,
    target_video: MiniMaxH3VideoLatentGeometry,
    target_audio_temporal: int,
    keyframes: tuple[MiniMaxH3KeyframeRole, ...] = (),
    guides: tuple[MiniMaxH3GuideTokenGeometry, ...] = (),
    references: tuple[MiniMaxH3ReferenceTokenGeometry, ...] = (),
    sequence_shards: int = 1,
) -> MiniMaxH3TokenLayoutPlan:
    """Declare the exact H3 packed row order from semantic latent geometry."""

    _positive_int("text tokens", text_tokens)
    if type(target_video) is not MiniMaxH3VideoLatentGeometry:
        raise MiniMaxH3TokenLayoutError("target video must be exact MiniMaxH3VideoLatentGeometry")
    _positive_int("target audio temporal", target_audio_temporal)
    _positive_int("sequence shards", sequence_shards)
    if type(keyframes) is not tuple or any(
        type(keyframe) is not MiniMaxH3KeyframeRole for keyframe in keyframes
    ):
        raise MiniMaxH3TokenLayoutError(
            "keyframes must be a tuple of exact MiniMaxH3KeyframeRole values"
        )
    if len(keyframes) > 2 or len(set(keyframes)) != len(keyframes):
        raise MiniMaxH3TokenLayoutError("keyframe roles must be unique with at most two entries")
    validate_minimax_h3_guide_timeline(guides)
    if type(references) is not tuple or any(
        type(reference) is not MiniMaxH3ReferenceTokenGeometry for reference in references
    ):
        raise MiniMaxH3TokenLayoutError(
            "references must be a tuple of exact MiniMaxH3ReferenceTokenGeometry values"
        )
    if keyframes and references:
        raise MiniMaxH3TokenLayoutError("keyframes and references are mutually exclusive")

    frame_count = _target_frame_count(target_video)
    expected_audio = round(
        frame_count / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )
    if target_audio_temporal != expected_audio:
        raise MiniMaxH3TokenLayoutError(
            "target audio temporal length must match the target video frame count"
        )
    for guide in guides:
        if guide.frame_index + guide.frame_count > frame_count:
            raise MiniMaxH3TokenLayoutError("guide frame range exceeds the target timeline")
        if guide.video is not None and (
            guide.video.height != target_video.height or guide.video.width != target_video.width
        ):
            raise MiniMaxH3TokenLayoutError("guide video spatial geometry must match the target")
        if guide.audio_temporal is not None:
            remaining_audio = int(
                target_audio_temporal
                - MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(guide.frame_index)
            )
            if guide.audio_temporal > remaining_audio:
                raise MiniMaxH3TokenLayoutError("guide audio exceeds the target timeline")

    segments: list[ModelTokenSegment] = []
    offset = 0

    def append(identity: str, modality: str, role: str, grid: tuple[int, ...]) -> None:
        nonlocal offset
        segment = _segment(identity, modality, role, offset, grid)
        segments.append(segment)
        offset = segment.stop

    append("text", "text", "context", (text_tokens,))
    target_frame_grid = (1, *target_video.token_grid[1:])
    for role in keyframes:
        append(f"keyframe-{role.value}", "video", "condition", target_frame_grid)
    for ordinal, guide in enumerate(guides, start=1):
        if guide.video is not None:
            append(
                f"guide-{ordinal}-video",
                "video",
                "condition",
                (guide.video.token_grid[0], *target_video.token_grid[1:]),
            )
        if guide.audio_temporal is not None:
            append(
                f"guide-{ordinal}-audio",
                "audio",
                "condition",
                (MINIMAX_H3_CONFIG.audio_content_channels, guide.audio_temporal),
            )
    for ordinal, reference in enumerate(references, start=1):
        if reference.audio_temporal is not None:
            append(
                f"reference-{ordinal}-audio",
                "audio",
                "reference",
                (MINIMAX_H3_CONFIG.audio_content_channels, reference.audio_temporal),
            )
        if reference.video is not None:
            append(
                f"reference-{ordinal}-video",
                "video",
                "reference",
                reference.video.token_grid,
            )
    append(
        "target-audio",
        "audio",
        "target",
        (MINIMAX_H3_CONFIG.audio_content_channels, target_audio_temporal),
    )
    append("target-video", "video", "target", target_video.token_grid)

    try:
        partition = plan_sequence_partition(offset, sequence_shards)
    except SequencePartitionError as error:
        raise MiniMaxH3TokenLayoutError(str(error)) from None
    layout = ModelTokenLayout(tuple(segments), partition.padded_length - offset)
    transforms = (
        TokenGridTransform(
            _VIDEO_TRANSFORM,
            "video",
            "target-video",
            (target_video.temporal, target_video.height, target_video.width),
            _QUANTIZATION_LEVELS,
        ),
        TokenGridTransform(
            _AUDIO_TRANSFORM,
            "audio",
            "target-audio",
            (
                MINIMAX_H3_CONFIG.audio_latent_channels,
                MINIMAX_H3_CONFIG.audio_content_channels,
                target_audio_temporal,
            ),
            _QUANTIZATION_LEVELS,
        ),
    )
    plan = object.__new__(MiniMaxH3TokenLayoutPlan)
    object.__setattr__(plan, "layout", layout)
    object.__setattr__(plan, "transforms", transforms)
    object.__setattr__(plan, "partition", partition)
    return plan
