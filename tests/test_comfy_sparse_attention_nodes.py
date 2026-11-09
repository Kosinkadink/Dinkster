"""Cold wire contract for native sparse-attention workflow inputs."""

from typing import Any, cast

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_nodes_comfy.sparse_attention import BlockSparseAttention


def test_sparse_attention_wire_keeps_method_before_schedule_and_single_line_blocks() -> None:
    wire = schema_to_wire(BlockSparseAttention.schema())
    inputs = [
        item for item in cast(list[dict[str, Any]], wire["interface"]) if item["role"] != "output"
    ]
    assert [item["id"] for item in inputs] == [
        "model",
        "selection",
        "start_percent",
        "end_percent",
        "dense_blocks",
        "min_tokens",
        "extra_tokens",
        "sink_conditioning",
        "verbose",
    ]
    assert inputs[1]["role"] == "dynamicCombo"
    assert inputs[4]["widget"] == {"type": "STRING", "multiline": False}
    assert schema_to_wire(schema_from_wire(wire)) == wire
