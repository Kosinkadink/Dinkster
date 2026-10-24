"""Public interfaces for rotary geometry and differential denoise masks."""

from dinkster_api.v1 import CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.rope_and_mask import DifferentialDiffusion, ScaleROPE


def test_rope_and_mask_schemas_match_core_interfaces() -> None:
    assert ScaleROPE in COMFY_NODES
    assert DifferentialDiffusion in COMFY_NODES
    rope = ScaleROPE.define_schema()
    assert rope.node_type == "ScaleROPE"
    assert rope.category == "model/patch"
    assert [item.id for item in rope.inputs] == [
        "model",
        "scale_x",
        "shift_x",
        "scale_y",
        "shift_y",
        "scale_t",
        "shift_t",
    ]
    assert [item.default for item in rope.inputs[1:]] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    for index, item in enumerate(rope.inputs[1:]):
        assert item.advanced
        assert item.type == TypeExpr.concrete(CORE_FLOAT)
        assert item.widget == (
            NumberWidget(min=-256.0, max=256.0, step=0.1)
            if index % 2
            else NumberWidget(min=0.0, max=100.0, step=0.1)
        )
    mask = DifferentialDiffusion.define_schema()
    assert mask.display_name == "Differential Diffusion"
    assert mask.category == "experimental"
    assert [item.id for item in mask.inputs] == ["model", "strength"]
    strength = mask.inputs[1]
    assert strength.default == 1.0
    assert not strength.required
    assert strength.widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    for schema in (rope, mask):
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
