from __future__ import annotations

import pytest
from dinkster_inference import (
    ModelTokenLayout,
    ModelTokenSegment,
    RowValue,
    TokenGridTransform,
    TokenLayoutError,
    TokenRowSpan,
    TokenRowTable,
    localize_row_spans,
    map_transforms,
    plan_sequence_partition,
    validate_row_spans,
)


def _av_layout(padded_rows: int = 0) -> ModelTokenLayout:
    return ModelTokenLayout(
        (
            ModelTokenSegment("video-0", "video", "latent", 0, 24, (2, 3, 4)),
            ModelTokenSegment("audio-0", "audio", "latent", 24, 30, (6,)),
        ),
        padded_rows,
    )


def test_layout_properties_and_lookup() -> None:
    layout = _av_layout(padded_rows=2)

    assert layout.valid_rows == 30
    assert layout.total_rows == 32
    assert layout.modalities == ("video", "audio")
    assert layout.by_identity("audio-0").rows == 6
    assert layout.segments[0].rows == 24
    with pytest.raises(KeyError):
        layout.by_identity("missing")


@pytest.mark.parametrize(
    ("identity", "modality", "role", "start", "stop", "grid", "message"),
    (
        ("", "video", "latent", 0, 4, (4,), "identity"),
        ("seg", " video", "latent", 0, 4, (4,), "modality"),
        ("seg", "video", "", 0, 4, (4,), "role"),
        ("seg|0", "video", "latent", 0, 4, (4,), "must not contain"),
        ("seg", "vi\tdeo", "latent", 0, 4, (4,), "must not contain"),
        ("seg", "video", "latent", -1, 4, (5,), "non-negative"),
        ("seg", "video", "latent", 4, 4, (1,), "half-open"),
        ("seg", "video", "latent", 0, 4, (), "non-empty"),
        ("seg", "video", "latent", 0, 4, (0, 4), ">= 1"),
        ("seg", "video", "latent", 0, 4, (3,), "flatten"),
    ),
)
def test_segment_rejects_invalid_declarations(
    identity: str,
    modality: str,
    role: str,
    start: int,
    stop: int,
    grid: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(TokenLayoutError, match=message):
        ModelTokenSegment(identity, modality, role, start, stop, grid)


def test_layout_rejects_gaps_duplicates_and_bad_padding() -> None:
    video = ModelTokenSegment("video-0", "video", "latent", 0, 24, (2, 3, 4))
    with pytest.raises(TokenLayoutError, match="contiguous"):
        ModelTokenLayout((video, ModelTokenSegment("audio-0", "audio", "latent", 25, 31, (6,))), 0)
    with pytest.raises(TokenLayoutError, match="contiguous"):
        ModelTokenLayout((ModelTokenSegment("late", "video", "latent", 4, 8, (4,)),), 0)
    with pytest.raises(TokenLayoutError, match="unique"):
        ModelTokenLayout((video, ModelTokenSegment("video-0", "audio", "latent", 24, 30, (6,))), 0)
    with pytest.raises(TokenLayoutError, match="padded_rows"):
        ModelTokenLayout((video,), -1)
    with pytest.raises(TokenLayoutError, match="at least one segment"):
        ModelTokenLayout((), 0)


def test_layout_digest_is_stable_and_fact_sensitive() -> None:
    assert _av_layout().digest == _av_layout().digest
    assert _av_layout().digest != _av_layout(padded_rows=2).digest
    reordered_grid = ModelTokenLayout(
        (
            ModelTokenSegment("video-0", "video", "latent", 0, 24, (3, 2, 4)),
            ModelTokenSegment("audio-0", "audio", "latent", 24, 30, (6,)),
        ),
        0,
    )
    assert reordered_grid.digest != _av_layout().digest


def test_map_transforms_binds_at_most_one_per_modality() -> None:
    layout = _av_layout()
    video = TokenGridTransform("video-2x2-maxpool.v1", "video", "video-0", (2, 6, 8), 256)
    audio = TokenGridTransform("audio-feature-maxpool.v1", "audio", "audio-0", (2, 6), 256)

    bound = map_transforms(layout, (video, audio))
    assert bound == {"video": video, "audio": audio}

    assert map_transforms(layout, (video,)) == {"video": video}
    assert map_transforms(layout, ()) == {}

    with pytest.raises(TokenLayoutError, match="duplicate"):
        map_transforms(layout, (video, video, audio))
    with pytest.raises(TokenLayoutError, match="not declared by the layout"):
        map_transforms(
            layout,
            (video, audio, TokenGridTransform("t.v1", "text", "video-0", (4,), 1)),
        )
    with pytest.raises(TokenLayoutError, match="unknown segment"):
        map_transforms(
            layout,
            (TokenGridTransform("t.v1", "video", "missing", (4,), 256), audio),
        )
    with pytest.raises(TokenLayoutError, match="match its bound segment"):
        map_transforms(
            layout,
            (TokenGridTransform("t.v1", "video", "audio-0", (4,), 256), audio),
        )


def test_mixed_global_layout_requires_only_target_av_transforms() -> None:
    layout = ModelTokenLayout(
        (
            ModelTokenSegment("text-0", "text", "prompt", 0, 8, (8,)),
            ModelTokenSegment("keyframe-0", "video", "condition", 8, 14, (1, 2, 3)),
            ModelTokenSegment("reference-video-0", "video", "reference", 14, 26, (2, 2, 3)),
            ModelTokenSegment("reference-audio-0", "audio", "reference", 26, 30, (2, 2)),
            ModelTokenSegment("target-video", "video", "target", 30, 54, (2, 3, 4)),
            ModelTokenSegment("target-audio", "audio", "target", 54, 60, (2, 3)),
        ),
        2,
    )
    video = TokenGridTransform("video-2x2-maxpool.v1", "video", "target-video", (2, 6, 8), 256)
    audio = TokenGridTransform("audio-feature-maxpool.v1", "audio", "target-audio", (8, 6), 256)

    bound = map_transforms(layout, (video, audio))

    assert bound == {"video": video, "audio": audio}
    assert bound["video"].segment_identity == "target-video"
    assert bound["audio"].segment_identity == "target-audio"
    assert layout.modalities == ("text", "video", "audio")
    assert layout.valid_rows == 60
    assert layout.total_rows == 62


def test_transform_digest_binds_every_fact() -> None:
    base = TokenGridTransform("video-2x2-maxpool.v1", "video", "video-0", (2, 6, 8), 256)
    assert (
        base.digest
        == TokenGridTransform("video-2x2-maxpool.v1", "video", "video-0", (2, 6, 8), 256).digest
    )
    for other in (
        TokenGridTransform("video-2x2-maxpool.v2", "video", "video-0", (2, 6, 8), 256),
        TokenGridTransform("video-2x2-maxpool.v1", "video", "video-0", (2, 6, 8), 128),
        TokenGridTransform("video-2x2-maxpool.v1", "video", "video-0", (2, 48), 256),
    ):
        assert other.digest != base.digest


def test_row_table_validation_and_shard_slicing() -> None:
    layout = _av_layout()
    table = TokenRowTable(tuple(float(row) for row in range(30)))
    table.validate_for(layout)

    partition = plan_sequence_partition(layout.valid_rows, 4)
    gathered: list[RowValue] = []
    for shard in partition.shards:
        piece = table.shard_slice(shard.start, shard.stop)
        assert len(piece) == shard.valid_rows
        gathered.extend(piece)
    assert tuple(gathered) == table.values

    with pytest.raises(TokenLayoutError, match="valid semantic rows"):
        TokenRowTable((0.0,)).validate_for(layout)
    with pytest.raises(TokenLayoutError, match="within the table"):
        table.shard_slice(24, 31)
    with pytest.raises(TokenLayoutError, match="0 <= start < stop"):
        table.shard_slice(5, 5)


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ((), "at least one value"),
        ((1, 2.0), "homogeneous"),
        ((True, False), "non-empty tuples"),
        (("a",), "non-empty tuples"),
        (((1, 2), (3,)), "homogeneous"),
        (((1, 2), (3.0, 4.0)), "homogeneous"),
        ((((1,), 2),), "homogeneous"),
        (((),), "non-empty tuples"),
    ),
)
def test_row_table_rejects_invalid_values(values: tuple[object, ...], message: str) -> None:
    with pytest.raises(TokenLayoutError, match=message):
        TokenRowTable(values)  # type: ignore[arg-type]


def test_row_table_carries_trailing_value_dimensions() -> None:
    table = TokenRowTable(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))

    assert table.rows == 3
    assert table.row_shape == (2,)
    assert table.shard_slice(1, 3) == ((3.0, 4.0), (5.0, 6.0))

    nested = TokenRowTable((((1, 2), (3, 4)), ((5, 6), (7, 8))))
    assert nested.rows == 2
    assert nested.row_shape == (2, 2)

    assert TokenRowTable((7, 8, 9)).row_shape == ()


def test_validate_row_spans_refuses_padded_tail_semantics() -> None:
    layout = ModelTokenLayout((ModelTokenSegment("seg", "video", "target", 0, 2, (2,)),), 2)

    validate_row_spans(layout, (TokenRowSpan(0, 2, 7),))
    validate_row_spans(layout, ())

    with pytest.raises(TokenLayoutError, match="padded tail"):
        validate_row_spans(layout, (TokenRowSpan(2, 4, 7),))
    with pytest.raises(TokenLayoutError, match="padded tail"):
        validate_row_spans(layout, (TokenRowSpan(0, 3, 7),))
    with pytest.raises(TokenLayoutError, match="ascending"):
        validate_row_spans(layout, (TokenRowSpan(1, 2, 0), TokenRowSpan(0, 1, 1)))
    with pytest.raises(TokenLayoutError, match="exact ModelTokenLayout"):
        validate_row_spans(object(), (TokenRowSpan(0, 1, 0),))  # type: ignore[arg-type]
    with pytest.raises(TokenLayoutError, match="exact TokenRowSpan"):
        validate_row_spans(layout, ((0, 1, 0),))  # type: ignore[arg-type]


def test_unquantized_transform_is_representable_and_identity_distinct() -> None:
    quantized = TokenGridTransform("t.v1", "video", "video-0", (2, 6, 8), 256)
    exact = TokenGridTransform("t.v1", "video", "video-0", (2, 6, 8), None)

    assert exact.quantization_levels is None
    assert exact.digest != quantized.digest
    assert exact.digest == TokenGridTransform("t.v1", "video", "video-0", (2, 6, 8), None).digest

    with pytest.raises(TokenLayoutError, match="None or an exact int"):
        TokenGridTransform("t.v1", "video", "video-0", (2, 6, 8), 0)
    with pytest.raises(TokenLayoutError, match="None or an exact int"):
        TokenGridTransform("t.v1", "video", "video-0", (2, 6, 8), 2.0)  # type: ignore[arg-type]


def test_layout_boundaries_require_exact_layout_values() -> None:
    with pytest.raises(TokenLayoutError, match="exact ModelTokenLayout"):
        map_transforms(object(), ())  # type: ignore[arg-type]
    with pytest.raises(TokenLayoutError, match="exact ModelTokenLayout"):
        TokenRowTable((1,)).validate_for(object())  # type: ignore[arg-type]


def test_row_spans_localize_by_intersection_and_offset() -> None:
    spans = (TokenRowSpan(0, 10, 3), TokenRowSpan(10, 24, 7), TokenRowSpan(24, 30, 1))

    local = localize_row_spans(spans, 8, 16)
    assert tuple(span.as_triple for span in local) == ((0, 2, 3), (2, 8, 7))

    tail = localize_row_spans(spans, 24, 30)
    assert tuple(span.as_triple for span in tail) == ((0, 6, 1),)

    assert localize_row_spans((TokenRowSpan(0, 4, 0),), 8, 16) == ()

    with pytest.raises(TokenLayoutError, match="ascending"):
        localize_row_spans((TokenRowSpan(4, 8, 0), TokenRowSpan(0, 4, 1)), 0, 8)
    with pytest.raises(TokenLayoutError, match="exact TokenRowSpan"):
        localize_row_spans(((0, 4, 0),), 0, 8)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "stop", "row", "message"),
    ((-1, 4, 0, "0 <= start < stop"), (4, 4, 0, "0 <= start < stop"), (0, 4, -1, "non-negative")),
)
def test_row_span_rejects_invalid_boundaries(start: int, stop: int, row: int, message: str) -> None:
    with pytest.raises(TokenLayoutError, match=message):
        TokenRowSpan(start, stop, row)
