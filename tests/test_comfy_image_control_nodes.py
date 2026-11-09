"""Cold public schemas for image-control model patches."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.image_control_patches import IMAGE_CONTROL_PATCH_NODES


def test_image_control_public_contracts() -> None:
    expected = (
        (
            "model",
            "model_patch",
            "vae",
            "image",
            "strength",
            "mask",
            "start_percent",
            "end_percent",
        ),
        (
            "model",
            "model_patch",
            "vae",
            "strength",
            "image",
            "inpaint_image",
            "mask",
            "start_percent",
            "end_percent",
        ),
        ("model", "model_patch", "clip_vision_output"),
    )
    for node, names in zip(IMAGE_CONTROL_PATCH_NODES, expected, strict=True):
        assert node in COMFY_NODES
        schema = node.schema()
        assert tuple(item.id for item in schema.inputs) == names
        assert schema.outputs[0].id == "MODEL"
        assert schema_from_wire(schema_to_wire(schema)) == schema
        if "strength" in names:
            inputs = {item.id: item for item in schema.inputs}
            assert inputs["strength"].widget == NumberWidget(min=-10.0, max=10.0, step=0.01)
            for name, default in (("start_percent", 0.0), ("end_percent", 1.0)):
                assert not inputs[name].required
                assert inputs[name].default == default
                assert inputs[name].advanced
                assert inputs[name].widget == NumberWidget(min=0.0, max=1.0, step=0.001)
