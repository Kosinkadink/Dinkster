from __future__ import annotations

import pytest
from dinkster_inference import SequenceLayout, SequenceLayoutError, SequenceShard


def _layout(**changes: object) -> SequenceLayout:
    values: dict[str, object] = {
        "global_sequence_length": 13,
        "shard": SequenceShard(3, 12, 13, 3),
        "original_token_order_reference": "packed-token-order:example",
        "head_count": 8,
        "head_start": 4,
        "head_stop": 8,
        "rope_position_offset": 12,
        "mesh_identity": "mesh-generation:example",
    }
    values.update(changes)
    return SequenceLayout(**values)  # type: ignore[arg-type]


def test_sequence_layout_records_attention_semantics() -> None:
    layout = _layout()

    assert layout.global_sequence_length == 13
    assert layout.shard.valid_rows == 1
    assert layout.shard.padded_rows == 3
    assert (layout.head_start, layout.head_stop) == (4, 8)
    assert layout.rope_position_offset == layout.shard.start


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"global_sequence_length": True}, "exact int"),
        ({"global_sequence_length": 0}, "global_sequence_length"),
        ({"shard": SequenceShard(0, 0, 14, 0)}, "shard stop"),
        (
            {"shard": SequenceShard(0, 0, 0, 0), "rope_position_offset": 0},
            "at least one valid row",
        ),
        ({"shard": SequenceShard(0, 0, 4, 1), "rope_position_offset": 0}, "global tail"),
        ({"original_token_order_reference": ""}, "original_token_order_reference"),
        ({"head_count": 0}, "head_count"),
        ({"head_start": 8}, "head ownership"),
        ({"head_stop": 9}, "head ownership"),
        ({"rope_position_offset": 11}, "rope_position_offset"),
        ({"mesh_identity": ""}, "mesh_identity"),
    ),
)
def test_sequence_layout_rejects_inconsistent_metadata(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(SequenceLayoutError, match=message):
        _layout(**changes)
