"""MiniMax H3 declarations for generic persistent sequence sharding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .sequence_sharding import (
    PackedSequenceFacts,
    SequenceAttentionKernelFactory,
    SequenceGather,
    SequenceSharding,
)

MiniMaxH3PackedSegmentKind = Literal[
    "text",
    "condition",
    "condition_audio",
    "reference_audio",
    "reference_video",
    "audio",
    "video",
]

_SEGMENT_KINDS = frozenset(
    (
        "text",
        "condition",
        "condition_audio",
        "reference_audio",
        "reference_video",
        "audio",
        "video",
    )
)

MINIMAX_H3_PACKED_SEQUENCE_LAYOUT = "minimax-h3-packed-sequence.v1"


@dataclass(frozen=True, slots=True)
class MiniMaxH3PackedSequenceFacts(PackedSequenceFacts):
    """Immutable boundaries for one packed MiniMax H3 DiT sequence."""

    segments: tuple[tuple[int, int, MiniMaxH3PackedSegmentKind], ...]
    layout_identity: str = field(default=MINIMAX_H3_PACKED_SEQUENCE_LAYOUT, init=False)

    def __post_init__(self) -> None:
        PackedSequenceFacts.__post_init__(self)
        for _start, _stop, kind in self.segments:
            if kind not in _SEGMENT_KINDS:
                raise ValueError(f"unknown segment kind {kind!r}")

    @property
    def conditioning_prefix_length(self) -> int | None:
        video = tuple((start, stop) for start, stop, kind in self.segments if kind == "video")
        if len(video) != 1 or video[0][1] != self.sequence_length:
            return None
        return video[0][0]


MiniMaxH3AttentionKernelFactory = SequenceAttentionKernelFactory[MiniMaxH3PackedSequenceFacts]
MiniMaxH3SequenceGather = SequenceGather
MiniMaxH3SequenceSharding = SequenceSharding


__all__ = [
    "MINIMAX_H3_PACKED_SEQUENCE_LAYOUT",
    "MiniMaxH3AttentionKernelFactory",
    "MiniMaxH3PackedSequenceFacts",
    "MiniMaxH3SequenceGather",
    "MiniMaxH3SequenceSharding",
]
