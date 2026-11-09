"""Public custom-sampling contracts remain safe without Torch."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.custom_sampling import CUSTOM_SAMPLING_NODES


def test_custom_sampling_schema_contracts() -> None:
    expected = (
        (
            "model",
            "add_noise",
            "noise_seed",
            "cfg",
            "positive",
            "negative",
            "sampler",
            "sigmas",
            "latent_image",
        ),
        ("noise", "guider", "sampler", "sigmas", "latent_image"),
    )
    for node, names in zip(CUSTOM_SAMPLING_NODES, expected, strict=True):
        assert node in COMFY_NODES
        schema = node.schema()
        assert tuple(item.id for item in schema.inputs) == names
        assert tuple(item.id for item in schema.outputs) == ("output", "denoised_output")
        assert schema_from_wire(schema_to_wire(schema)) == schema
    inputs = CUSTOM_SAMPLING_NODES[0].schema().inputs
    assert inputs[1].default is True and inputs[1].advanced
    assert inputs[2].widget == NumberWidget(
        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
    )
    assert inputs[3].default == 8.0
    assert inputs[3].widget == NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01)
