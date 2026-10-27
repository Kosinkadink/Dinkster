"""Morphology and color conversion preserve ComfyUI's ordered interfaces."""

from dinkster_api.v1 import ComboOption, ComboWidget
from dinkster_nodes_comfy.morphology import MORPHOLOGY_NODES


def test_morphology_nodes_preserve_image_ports_and_operation_choices() -> None:
    expected = (
        (
            "Morphology",
            "Apply Morphology",
            "image/filters",
            ("image", "operation", "kernel_size"),
            ("IMAGE",),
        ),
        ("ImageRGBToYUV", "Image RGB to YUV", "image/color", ("image",), ("Y", "U", "V")),
        ("ImageYUVToRGB", "Image YUV to RGB", "image/color", ("Y", "U", "V"), ("IMAGE",)),
    )
    for node, (node_type, label, category, inputs, outputs) in zip(
        MORPHOLOGY_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            label,
            category,
        )
        assert tuple(port.id for port in schema.inputs) == inputs
        assert tuple(port.id for port in schema.outputs) == outputs
        assert all(port.type.types == ("IMAGE",) for port in schema.outputs)
    schema = MORPHOLOGY_NODES[0].schema()
    widget = schema.inputs[1].widget
    assert isinstance(widget, ComboWidget)
    assert tuple(
        option.value if isinstance(option, ComboOption) else option for option in widget.options
    ) == (
        "erode",
        "dilate",
        "open",
        "close",
        "gradient",
        "bottom_hat",
        "top_hat",
    )
    assert schema.inputs[2].default == 3
