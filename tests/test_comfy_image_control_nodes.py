"""Cold public schemas for image-control model patches."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_flux import NODES as FLUX_NODES
from dinkster_flux.image_control_patches import USOStyleReference
from dinkster_misc_models import NODES as MISC_NODES
from dinkster_misc_models.image_control_patches import ZImageFunControlnet
from dinkster_qwen import NODES as QWEN_NODES
from dinkster_qwen.image_control_patches import QwenImageDiffsynthControlnet


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
    for (node, registered, node_type), names in zip(
        (
            (
                QwenImageDiffsynthControlnet,
                QWEN_NODES,
                "dinkster.qwen.qwen_image_diffsynth_controlnet",
            ),
            (ZImageFunControlnet, MISC_NODES, "dinkster.misc.models.z_image_fun_controlnet"),
            (USOStyleReference, FLUX_NODES, "dinkster.flux.uso_style_reference"),
        ),
        expected,
        strict=True,
    ):
        assert node in registered
        schema = node.schema()
        assert schema.node_type == node_type
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
