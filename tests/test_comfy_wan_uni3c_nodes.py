"""Cold public contract for Wan Uni3C render-video guidance."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.wan_uni3c import WanUni3CControlnetApply


def test_wan_uni3c_schema_and_cold_registration() -> None:
    assert WanUni3CControlnetApply in COMFY_NODES
    schema = WanUni3CControlnetApply.schema()
    assert schema.category == "model/patch/wan"
    assert [item.id for item in schema.inputs] == [
        "model",
        "model_patch",
        "vae",
        "render_video",
        "strength",
        "start_percent",
        "end_percent",
    ]
    assert all(item.required for item in schema.inputs)
    assert schema.inputs[4].widget == NumberWidget(min=-10.0, max=10.0, step=0.01)
    assert [item.default for item in schema.inputs[4:]] == [1.0, 0.0, 1.0]
    assert schema.outputs[0].id == "MODEL"
    assert schema_from_wire(schema_to_wire(schema)) == schema
