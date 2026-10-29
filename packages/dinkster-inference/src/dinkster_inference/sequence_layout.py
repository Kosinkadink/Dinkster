"""Torch-free attention layout metadata for one sequence shard."""

from __future__ import annotations

from dataclasses import dataclass

from .sequence_partition import SequenceShard

__all__ = ["SequenceLayout", "SequenceLayoutError"]


class SequenceLayoutError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SequenceLayout:
    """Bind one padded sequence shard to its attention semantics."""

    global_sequence_length: int
    shard: SequenceShard
    original_token_order_reference: str
    head_count: int
    head_start: int
    head_stop: int
    rope_position_offset: int
    mesh_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("global_sequence_length", self.global_sequence_length),
            ("head_count", self.head_count),
            ("head_start", self.head_start),
            ("head_stop", self.head_stop),
            ("rope_position_offset", self.rope_position_offset),
        ):
            if type(value) is not int:
                raise SequenceLayoutError(f"{name} must be an exact int")
        if self.global_sequence_length < 1:
            raise SequenceLayoutError("global_sequence_length must be >= 1")
        if type(self.shard) is not SequenceShard:
            raise SequenceLayoutError("shard must be an exact SequenceShard")
        if self.shard.stop <= self.shard.start:
            raise SequenceLayoutError("shard must contain at least one valid row")
        if self.shard.stop > self.global_sequence_length:
            raise SequenceLayoutError("shard stop must not exceed global_sequence_length")
        if self.shard.stop < self.global_sequence_length and self.shard.padded_rows:
            raise SequenceLayoutError("only the global tail shard may contain padding")
        if (
            type(self.original_token_order_reference) is not str
            or not self.original_token_order_reference
        ):
            raise SequenceLayoutError("original_token_order_reference must be a non-empty str")
        if self.head_count < 1:
            raise SequenceLayoutError("head_count must be >= 1")
        if not 0 <= self.head_start < self.head_stop <= self.head_count:
            raise SequenceLayoutError(
                "head ownership must satisfy 0 <= head_start < head_stop <= head_count"
            )
        if self.rope_position_offset != self.shard.start:
            raise SequenceLayoutError("rope_position_offset must equal the shard start")
        if type(self.mesh_identity) is not str or not self.mesh_identity:
            raise SequenceLayoutError("mesh_identity must be a non-empty str")
