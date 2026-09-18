from __future__ import annotations

import pytest
from dinkster_inference import (
    SequencePartition,
    SequencePartitionError,
    SequenceShard,
    plan_sequence_partition,
    translate_segments,
)


@pytest.mark.parametrize(
    ("sequence_length", "shard_count", "chunk", "expected"),
    (
        (10, 2, 5, ((0, 0, 5, 0), (1, 5, 10, 0))),
        (10, 3, 4, ((0, 0, 4, 0), (1, 4, 8, 0), (2, 8, 10, 2))),
        (4, 4, 1, ((0, 0, 1, 0), (1, 1, 2, 0), (2, 2, 3, 0), (3, 3, 4, 0))),
        (37296, 2, 18648, ((0, 0, 18648, 0), (1, 18648, 37296, 0))),
        (
            37296,
            4,
            9324,
            (
                (0, 0, 9324, 0),
                (1, 9324, 18648, 0),
                (2, 18648, 27972, 0),
                (3, 27972, 37296, 0),
            ),
        ),
        (7, 1, 7, ((0, 0, 7, 0),)),
    ),
)
def test_plan_sequence_partition_exact_shards(
    sequence_length: int,
    shard_count: int,
    chunk: int,
    expected: tuple[tuple[int, int, int, int], ...],
) -> None:
    partition = plan_sequence_partition(sequence_length, shard_count)

    assert partition.sequence_length == sequence_length
    assert partition.shard_count == shard_count
    assert partition.chunk == chunk
    assert partition.padded_length == chunk * shard_count
    assert (
        tuple(
            (shard.index, shard.start, shard.stop, shard.padded_rows) for shard in partition.shards
        )
        == expected
    )
    assert tuple(shard.valid_rows for shard in partition.shards) == tuple(
        stop - start for _, start, stop, _ in expected
    )


@pytest.mark.parametrize(
    ("index", "start", "stop", "padded_rows", "message"),
    (
        (-1, 0, 1, 0, "index"),
        (0, -1, 1, 0, "start"),
        (0, 2, 1, 0, "stop"),
        (0, 0, 1, -1, "padded_rows"),
    ),
)
def test_sequence_shard_rejects_invalid_boundaries(
    index: int, start: int, stop: int, padded_rows: int, message: str
) -> None:
    with pytest.raises(SequencePartitionError, match=message):
        SequenceShard(index, start, stop, padded_rows)


@pytest.mark.parametrize(
    ("sequence_length", "shard_count", "chunk", "shards", "message"),
    (
        (0, 1, 1, (SequenceShard(0, 0, 1, 0),), "sequence_length"),
        (1, 0, 1, (), "shard_count"),
        (4, 3, 2, (SequenceShard(0, 0, 2, 0), SequenceShard(1, 2, 4, 0)), "shard_count"),
        (
            4,
            2,
            2,
            (SequenceShard(0, 0, 2, 0), SequenceShard(1, 3, 5, 0)),
            "contiguous",
        ),
        (
            4,
            2,
            2,
            (SequenceShard(1, 0, 2, 0), SequenceShard(0, 2, 4, 0)),
            "indices",
        ),
        (
            3,
            2,
            2,
            (SequenceShard(0, 0, 1, 1), SequenceShard(1, 1, 3, 0)),
            "tail shard",
        ),
        (
            3,
            2,
            3,
            (SequenceShard(0, 0, 2, 0), SequenceShard(1, 2, 3, 0)),
            "uniform chunk",
        ),
        (
            2,
            2,
            1,
            (SequenceShard(0, 0, 1, 0), SequenceShard(1, 1, 1, 1)),
            "at least one valid row",
        ),
        (
            5,
            2,
            2,
            (SequenceShard(0, 0, 2, 0), SequenceShard(1, 2, 4, 0)),
            "valid rows",
        ),
    ),
)
def test_sequence_partition_rejects_inconsistent_shards(
    sequence_length: int,
    shard_count: int,
    chunk: int,
    shards: tuple[SequenceShard, ...],
    message: str,
) -> None:
    with pytest.raises(SequencePartitionError, match=message):
        SequencePartition(sequence_length, shard_count, chunk, shards)


def test_plan_sequence_partition_allows_padding_larger_than_valid_tail() -> None:
    partition = plan_sequence_partition(13, 4)

    assert tuple((shard.start, shard.stop, shard.padded_rows) for shard in partition.shards) == (
        (0, 4, 0),
        (4, 8, 0),
        (8, 12, 0),
        (12, 13, 3),
    )
    assert tuple(
        row for shard in partition.shards for row in range(shard.start, shard.stop)
    ) == tuple(range(13))


@pytest.mark.parametrize(
    ("sequence_length", "shard_count", "message"),
    (
        (0, 1, "sequence_length"),
        (1, 0, "shard_count"),
        (2, 3, "must not exceed"),
        (5, 4, "too short"),
        (True, 1, "sequence_length"),
        (1.5, 1, "sequence_length"),
        (1, True, "shard_count"),
        (1, 1.5, "shard_count"),
    ),
)
def test_plan_sequence_partition_rejects_invalid_inputs(
    sequence_length: object, shard_count: object, message: str
) -> None:
    with pytest.raises(SequencePartitionError, match=message):
        plan_sequence_partition(
            sequence_length,  # type: ignore[arg-type]
            shard_count,  # type: ignore[arg-type]
        )


def test_translate_segments_is_identity_over_full_range() -> None:
    segments = ((0, 3, 0), (3, 8, 1), (10, 12, 2))

    assert translate_segments(segments, 0, 12) == segments


def test_translate_segments_splits_boundary_and_preserves_rows() -> None:
    segments = ((2, 8, 4),)

    left = translate_segments(segments, 0, 5)
    right = translate_segments(segments, 5, 10)

    assert left == ((2, 5, 4),)
    assert right == ((0, 3, 4),)
    assert sum(stop - start for start, stop, _ in (*left, *right)) == 6


def test_translate_segments_drops_empty_intersections() -> None:
    assert translate_segments(((0, 2, 0), (8, 10, 1)), 3, 7) == ()


@pytest.mark.parametrize(
    "segments",
    (
        ((0, 4, 0), (3, 5, 1)),
        ((4, 6, 0), (1, 3, 1)),
        ((0, 0, 0),),
        ((-1, 2, 0),),
        ((0, 2, -1),),
        ((0, 2),),
        ((0, 2, True),),
    ),
)
def test_translate_segments_rejects_invalid_segments(segments: object) -> None:
    with pytest.raises(SequencePartitionError):
        translate_segments(segments, 0, 10)  # type: ignore[arg-type]


@pytest.mark.parametrize(("start", "stop"), ((0, 0), (3, 2), (-1, 2), (True, 2), (0, 2.5)))
def test_translate_segments_rejects_invalid_shard_bounds(start: object, stop: object) -> None:
    with pytest.raises(SequencePartitionError, match="start and stop"):
        translate_segments((), start, stop)  # type: ignore[arg-type]


def test_translated_segments_reconstruct_original_coverage() -> None:
    segments = ((0, 5, 0), (5, 13, 1), (13, 20, 2))
    partition = plan_sequence_partition(20, 3)
    reconstructed: list[tuple[int, int]] = []

    for shard in partition.shards:
        for start, stop, row in translate_segments(segments, shard.start, shard.stop):
            reconstructed.extend(
                (global_row, row) for global_row in range(start + shard.start, stop + shard.start)
            )

    expected = [
        (global_row, row) for start, stop, row in segments for global_row in range(start, stop)
    ]
    assert reconstructed == expected
