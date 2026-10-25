"""Torch-free schema contracts for native context-window nodes."""

from dinkster_api.v1 import CORE_STRING, NumberWidget, TypeExpr, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.context_windows import CONTEXT_WINDOW_NODES


def test_context_window_nodes_are_native_pack_schemas() -> None:
    expected = (
        (
            "ContextWindowsManual",
            (
                "model",
                "context_length",
                "context_overlap",
                "context_schedule",
                "context_stride",
                "closed_loop",
                "fuse_method",
                "dim",
                "freenoise",
                "cond_retain_index_list",
                "split_conds_to_windows",
                "latent_retain_index_list",
                "causal_window_fix",
            ),
        ),
        (
            "WanContextWindowsManual",
            (
                "model",
                "context_length",
                "context_overlap",
                "context_schedule",
                "context_stride",
                "closed_loop",
                "fuse_method",
                "freenoise",
                "retain_first_frame",
                "split_conds_to_windows",
            ),
        ),
        (
            "LTXVContextWindows",
            (
                "model",
                "context_length",
                "context_overlap",
                "context_schedule",
                "context_stride",
                "closed_loop",
                "fuse_method",
                "freenoise",
                "retain_first_frame",
                "split_conds_to_windows",
            ),
        ),
    )
    for node, (node_type, inputs) in zip(CONTEXT_WINDOW_NODES, expected, strict=True):
        schema = node.schema()
        assert schema.node_type == node_type
        assert tuple(input.id for input in schema.inputs) == inputs
        assert schema.outputs[0].id == "MODEL"
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert node in COMFY_NODES


def test_context_window_manual_preserves_retain_index_strings() -> None:
    schema = CONTEXT_WINDOW_NODES[0].schema()
    retain_inputs = schema.inputs[9], schema.inputs[11]
    assert all(
        item.type == TypeExpr.concrete(CORE_STRING) and item.default == "" for item in retain_inputs
    )
    assert schema.inputs[0].required
    assert schema.inputs[1].widget == NumberWidget(min=1)
