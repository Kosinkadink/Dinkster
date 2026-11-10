"""Torch-free schema contracts for native context-window nodes."""

from dinkster_api.v1 import CORE_STRING, NumberWidget, TypeExpr, schema_from_wire, schema_to_wire
from dinkster_ltx import NODES as LTX_NODES
from dinkster_ltx.context_windows import LTXVContextWindows
from dinkster_models import NODES as MODEL_NODES
from dinkster_models.context_windows import ContextWindowsManual
from dinkster_wan import NODES as WAN_NODES
from dinkster_wan.context_windows import WanContextWindowsManual


def test_context_window_nodes_are_native_pack_schemas() -> None:
    expected = (
        (
            ContextWindowsManual,
            MODEL_NODES,
            "dinkster.models.context_windows_manual",
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
            WanContextWindowsManual,
            WAN_NODES,
            "dinkster.wan.wan_context_windows_manual",
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
            LTXVContextWindows,
            LTX_NODES,
            "dinkster.ltx.ltxv_context_windows",
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
    for node, registered, node_type, inputs in expected:
        schema = node.schema()
        assert schema.node_type == node_type
        assert tuple(input.id for input in schema.inputs) == inputs
        assert schema.outputs[0].id == "MODEL"
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert node in registered
    assert tuple(
        node.schema().category
        for node in (ContextWindowsManual, WanContextWindowsManual, LTXVContextWindows)
    ) == (
        "model/patch",
        "model/patch/wan",
        "model/patch/ltxv",
    )
    assert WanContextWindowsManual.schema().inputs[1].widget == NumberWidget(
        min=1, max=16384, step=4
    )
    assert LTXVContextWindows.schema().inputs[1].widget == NumberWidget(min=1, max=16384, step=8)


def test_context_window_manual_preserves_retain_index_strings() -> None:
    schema = ContextWindowsManual.schema()
    retain_inputs = schema.inputs[9], schema.inputs[11]
    assert all(
        item.type == TypeExpr.concrete(CORE_STRING) and item.default == "" for item in retain_inputs
    )
    assert schema.inputs[0].required
    assert schema.inputs[1].widget == NumberWidget(min=1)
