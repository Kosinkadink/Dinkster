"""Public schema for native Anima LLLite model patching."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy.anima_lllite import AnimaLLLiteApply


def test_anima_lllite_preserves_input_order_bounds_and_optional_mask() -> None:
    schema = AnimaLLLiteApply.schema()
    assert schema.node_type == "AnimaLLLiteApply"
    assert schema.display_name == "Apply Anima LLLite"
    assert schema.category == "model/patch/anima"
    assert [item.id for item in schema.inputs] == [
        "model",
        "model_patch",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "mask",
    ]
    assert [item.required for item in schema.inputs] == [True] * 6 + [False]
    assert schema.inputs[3].widget == NumberWidget(min=-10.0, max=10.0, step=0.01)
    assert [item.default for item in schema.inputs[3:6]] == [1.0, 0.0, 1.0]
    assert schema.outputs[0].id == "MODEL"
    assert schema_to_wire(schema_from_wire(schema_to_wire(schema))) == schema_to_wire(schema)
