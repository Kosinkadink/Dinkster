"""Public contracts for native FreeU model patch nodes."""

from __future__ import annotations

from dinkster_api.v1 import CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.freeu import FreeU, FreeU_V2


def test_freeu_nodes_expose_pinned_model_patch_schemas() -> None:
    assert FreeU in COMFY_NODES
    assert FreeU_V2 in COMFY_NODES
    for node, defaults in ((FreeU, (1.1, 1.2, 0.9, 0.2)), (FreeU_V2, (1.3, 1.4, 0.9, 0.2))):
        schema = node.define_schema()
        assert schema.node_type == node.__name__
        assert schema.category == "model/patch/unet"
        assert [input.id for input in schema.inputs] == ["model", "b1", "b2", "s1", "s2"]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert [input.type for input in schema.inputs[1:]] == [TypeExpr.concrete(CORE_FLOAT)] * 4
        assert [input.default for input in schema.inputs[1:]] == list(defaults)
        assert all(input.advanced for input in schema.inputs[1:])
        assert all(
            input.widget == NumberWidget(min=0.0, max=10.0, step=0.01)
            for input in schema.inputs[1:]
        )
        assert [(output.id, output.type) for output in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
