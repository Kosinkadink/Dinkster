"""Torch-free public interfaces for ComfyUI compositing nodes."""

from dinkster_api.v1 import ComboWidget, schema_to_wire
from dinkster_nodes_comfy.compositing import (
    COMPOSITING_NODES,
    PorterDuffImageComposite,
    PorterDuffMode,
)


def test_compositing_schemas_match_the_fixed_reference_contract() -> None:
    expected = (
        (
            "PorterDuffImageComposite",
            "Porter-Duff Image Composite",
            ("source", "source_alpha", "destination", "destination_alpha", "mode"),
            ("IMAGE", "MASK"),
            ("alpha composite", "blend modes", "layer blend", "transparency blend"),
        ),
        (
            "SplitImageWithAlpha",
            "Split Image with Alpha",
            ("image",),
            ("IMAGE", "MASK"),
            ("extract alpha", "separate transparency", "remove alpha"),
        ),
        (
            "JoinImageWithAlpha",
            "Join Image with Alpha",
            ("image", "alpha"),
            ("IMAGE",),
            ("add transparency", "apply alpha", "composite alpha", "RGBA"),
        ),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            tuple(input_spec.id for input_spec in node.schema().inputs),
            tuple(output_spec.id for output_spec in node.schema().outputs),
            node.schema().search_terms,
        )
        for node in COMPOSITING_NODES
    )
    assert actual == expected
    schema = PorterDuffImageComposite.schema()
    assert schema.inputs[-1].default == "DST"
    assert isinstance(schema.inputs[-1].widget, ComboWidget)
    assert tuple(
        option if isinstance(option, str) else option.value
        for option in schema.inputs[-1].widget.options
    ) == tuple(mode.name for mode in PorterDuffMode)
    for node in COMPOSITING_NODES:
        assert schema_to_wire(node.schema())
