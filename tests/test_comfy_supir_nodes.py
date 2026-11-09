"""Cold schema for native SUPIR model patching."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.supir import SUPIRApply


def test_supir_schema_preserves_guidance_controls_and_registration() -> None:
    schema = SUPIRApply.schema()
    assert SUPIRApply in COMFY_NODES
    assert schema.node_type == "SUPIRApply"
    assert schema.display_name == "Apply SUPIR Patch"
    assert schema.category == "model/patch/supir"
    assert [item.id for item in schema.inputs] == [
        "model",
        "model_patch",
        "vae",
        "image",
        "strength_start",
        "strength_end",
        "restore_cfg",
        "restore_cfg_s_tmin",
    ]
    assert [item.default for item in schema.inputs[4:]] == [1.0, 1.0, 4.0, 0.05]
    assert [item.advanced for item in schema.inputs] == [False] * 6 + [True] * 2
    assert schema.inputs[6].widget == NumberWidget(min=0.0, max=20.0, step=0.1)
    assert schema_to_wire(schema_from_wire(schema_to_wire(schema))) == schema_to_wire(schema)
