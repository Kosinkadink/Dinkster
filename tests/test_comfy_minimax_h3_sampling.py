"""Public MiniMax H3 sampling schema without the inference runtime."""

from __future__ import annotations

from dinkster_api.v1 import CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.minimax_h3_sampling import MiniMaxH3SigmaShift


def test_minimax_h3_sigma_shift_public_schema() -> None:
    schema = MiniMaxH3SigmaShift.schema()
    assert MiniMaxH3SigmaShift in COMFY_NODES
    assert schema.node_type == "MiniMaxH3SigmaShift"
    assert schema.display_name == "ModelSamplingMiniMaxH3"
    assert schema.category == "model/patch/minimax"
    assert tuple(i.id for i in schema.inputs) == ("model", "shift_video", "shift_audio")
    assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert all(i.required for i in schema.inputs)
    assert tuple(i.default for i in schema.inputs[1:]) == (12.0, 3.0)
    for item in schema.inputs[1:]:
        assert item.type == TypeExpr.concrete(CORE_FLOAT)
        assert item.widget == NumberWidget(min=0.01, max=100.0, step=0.01)
    assert tuple((o.id, o.type) for o in schema.outputs) == (
        ("MODEL", TypeExpr.concrete(MODEL_TYPE)),
    )
