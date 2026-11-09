"""Torch-free public interfaces for ComfyUI compositing nodes."""

from dinkster_api.v1 import ComboOption, ComboWidget, schema_to_wire
from dinkster_image import NODES
from dinkster_image.ported.compositing import (
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    PorterDuffMode,
    SplitImageWithAlpha,
)


def test_compositing_schemas_match_the_fixed_reference_contract() -> None:
    nodes = (PorterDuffImageComposite, SplitImageWithAlpha, JoinImageWithAlpha)
    expected = (
        (
            "dinkster.image.porter_duff_image_composite",
            "Porter-Duff Image Composite",
            ("source", "source_alpha", "destination", "destination_alpha", "mode"),
            ("IMAGE", "MASK"),
            (
                "alpha composite",
                "blend modes",
                "layer blend",
                "transparency blend",
                "PorterDuffImageComposite",
            ),
        ),
        (
            "dinkster.image.split_image_with_alpha",
            "Split Image with Alpha",
            ("image",),
            ("IMAGE", "MASK"),
            ("extract alpha", "separate transparency", "remove alpha", "SplitImageWithAlpha"),
        ),
        (
            "dinkster.image.join_image_with_alpha",
            "Join Image with Alpha",
            ("image", "alpha"),
            ("IMAGE",),
            ("add transparency", "apply alpha", "composite alpha", "RGBA", "JoinImageWithAlpha"),
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
        for node in nodes
    )
    assert actual == expected
    schema = PorterDuffImageComposite.schema()
    assert schema.inputs[-1].default == "DST"
    assert isinstance(schema.inputs[-1].widget, ComboWidget)
    assert tuple(
        option.value if isinstance(option, ComboOption) else option
        for option in schema.inputs[-1].widget.options
    ) == tuple(mode.name for mode in PorterDuffMode)
    for node in nodes:
        assert node in NODES
        assert schema_to_wire(node.schema())
