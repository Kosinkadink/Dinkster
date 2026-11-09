"""Public custom-sampling contracts remain safe without Torch."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_sampling import NODES
from dinkster_sampling.custom_sampling import SamplerCustom, SamplerCustomAdvanced


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
    for node, node_type, names in zip(
        (SamplerCustom, SamplerCustomAdvanced),
        ("dinkster.sampling.sampler_custom", "dinkster.sampling.sampler_custom_advanced"),
        expected,
        strict=True,
    ):
        assert node in NODES
        schema = node.schema()
        assert schema.node_type == node_type
        assert tuple(item.id for item in schema.inputs) == names
        assert tuple(item.id for item in schema.outputs) == ("output", "denoised_output")
        assert schema_from_wire(schema_to_wire(schema)) == schema
    inputs = SamplerCustom.schema().inputs
    assert inputs[1].default is True and inputs[1].advanced
    assert inputs[2].widget == NumberWidget(
        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
    )
    assert inputs[3].default == 8.0
    assert inputs[3].widget == NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01)
