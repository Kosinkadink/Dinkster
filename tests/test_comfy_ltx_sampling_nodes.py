"""Torch-free native LTXV sampling schema contracts."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_sampling import LTX_SAMPLING_NODES


def test_native_ltx_sampling_schemas() -> None:
    expected = (
        ("ModelSamplingLTXV", ("model", "max_shift", "base_shift", "latent"), "MODEL"),
        (
            "LTXVScheduler",
            ("steps", "max_shift", "base_shift", "stretch", "terminal", "latent"),
            "SIGMAS",
        ),
    )
    for node, (name, inputs, output) in zip(LTX_SAMPLING_NODES, expected, strict=True):
        schema = node.schema()
        assert schema.node_type == name
        assert tuple(item.id for item in schema.inputs) == inputs
        assert schema.outputs[0].type.types == (output,)
        assert not schema.inputs[-1].required
        assert all(item.required for item in schema.inputs[:-1])
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert node in COMFY_NODES
    scheduler = LTX_SAMPLING_NODES[1].schema()
    assert scheduler.inputs[3].advanced and scheduler.inputs[4].advanced
    widget = scheduler.inputs[4].widget
    assert isinstance(widget, NumberWidget)
    assert widget.min == 0.0
    assert widget.max == 0.99
