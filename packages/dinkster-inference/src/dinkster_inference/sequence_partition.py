"""Torch-free contiguous partition planning for packed-sequence sharding."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SequencePartition",
    "SequencePartitionError",
    "SequenceShard",
    "plan_sequence_partition",
    "translate_segments",
]


class SequencePartitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SequenceShard:
    index: int
    start: int
    stop: int
    padded_rows: int

    def __post_init__(self) -> None:
        for name, value in (
            ("index", self.index),
            ("start", self.start),
            ("stop", self.stop),
            ("padded_rows", self.padded_rows),
        ):
            if type(value) is not int:
                raise SequencePartitionError(f"{name} must be an exact int")
        if self.index < 0:
            raise SequencePartitionError("shard index must be non-negative")
        if self.start < 0:
            raise SequencePartitionError("shard start must be non-negative")
        if self.stop < self.start:
            raise SequencePartitionError("shard stop must not precede start")
        if self.padded_rows < 0:
            raise SequencePartitionError("padded_rows must be non-negative")

    @property
    def valid_rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class SequencePartition:
    sequence_length: int
    shard_count: int
    chunk: int
    shards: tuple[SequenceShard, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("sequence_length", self.sequence_length),
            ("shard_count", self.shard_count),
            ("chunk", self.chunk),
        ):
            if type(value) is not int:
                raise SequencePartitionError(f"{name} must be an exact int")
        if self.sequence_length < 1:
            raise SequencePartitionError("sequence_length must be >= 1")
        if self.shard_count < 1:
            raise SequencePartitionError("shard_count must be >= 1")
        if type(self.shards) is not tuple or any(
            type(shard) is not SequenceShard for shard in self.shards
        ):
            raise SequencePartitionError("shards must be a tuple of exact SequenceShard values")
        if self.shard_count != len(self.shards):
            raise SequencePartitionError("shard_count must equal the number of shards")
        previous_stop = 0
        valid_rows = 0
        padded_span = 0
        for index, shard in enumerate(self.shards):
            if shard.index != index:
                raise SequencePartitionError("shard indices must be consecutive and ordered")
            if shard.start != previous_stop:
                raise SequencePartitionError("shard spans must be contiguous from zero")
            if shard.valid_rows < 1:
                raise SequencePartitionError("every shard must own at least one valid row")
            if shard.valid_rows + shard.padded_rows != self.chunk:
                raise SequencePartitionError("each shard must fill one uniform chunk")
            if index < self.shard_count - 1 and shard.padded_rows != 0:
                raise SequencePartitionError("only the tail shard may contain padding")
            previous_stop = shard.stop
            valid_rows += shard.valid_rows
            padded_span += shard.valid_rows + shard.padded_rows
        if valid_rows != self.sequence_length:
            raise SequencePartitionError("shard valid rows must equal sequence_length")
        if padded_span != self.shard_count * self.chunk:
            raise SequencePartitionError("total padded span must equal shard_count * chunk")

    @property
    def padded_length(self) -> int:
        return self.chunk * self.shard_count


def plan_sequence_partition(sequence_length: int, shard_count: int) -> SequencePartition:
    """Plan equal-size chunks where every shard owns at least one valid row.

    Equal-chunk padding refuses lengths where the computed chunk would leave
    an empty trailing shard.
    """

    if type(sequence_length) is not int or sequence_length < 1:
        raise SequencePartitionError("sequence_length must be an exact int >= 1")
    if type(shard_count) is not int or shard_count < 1:
        raise SequencePartitionError("shard_count must be an exact int >= 1")
    if shard_count > sequence_length:
        raise SequencePartitionError("shard_count must not exceed sequence_length")
    chunk = -(-sequence_length // shard_count)
    if (shard_count - 1) * chunk >= sequence_length:
        raise SequencePartitionError(
            "sequence is too short for the shard count under equal-chunk padding"
        )
    shards = tuple(
        SequenceShard(
            index,
            index * chunk,
            min((index + 1) * chunk, sequence_length),
            chunk - (min((index + 1) * chunk, sequence_length) - index * chunk),
        )
        for index in range(shard_count)
    )
    return SequencePartition(sequence_length, shard_count, chunk, shards)


def translate_segments(
    segments: tuple[tuple[int, int, int], ...], start: int, stop: int
) -> tuple[tuple[int, int, int], ...]:
    """Intersect global H3 modulation segments with a shard and make them local."""

    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise SequencePartitionError("start and stop must be exact ints with 0 <= start < stop")
    if type(segments) is not tuple:
        raise SequencePartitionError("segments must be a tuple of triples")
    previous_stop = 0
    translated: list[tuple[int, int, int]] = []
    for segment in segments:
        if type(segment) is not tuple or len(segment) != 3:
            raise SequencePartitionError("each segment must be a tuple of three exact ints")
        segment_start, segment_stop, row = segment
        if any(type(value) is not int for value in segment):
            raise SequencePartitionError("each segment must contain three exact ints")
        if not 0 <= segment_start < segment_stop or row < 0:
            raise SequencePartitionError("segments require 0 <= start < stop and row >= 0")
        if segment_start < previous_stop:
            raise SequencePartitionError("segments must be ascending and non-overlapping")
        previous_stop = segment_stop
        intersection_start = max(segment_start, start)
        intersection_stop = min(segment_stop, stop)
        if intersection_start < intersection_stop:
            translated.append((intersection_start - start, intersection_stop - start, row))
    return tuple(translated)
