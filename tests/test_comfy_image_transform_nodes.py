"""Core image transform schemas preserve the ComfyUI workflow contract."""

from dinkster_image import NODES
from dinkster_image.ported.image_transform import (
    ImageAddNoise,
    ImageCrop,
    ImageFlip,
    ImageFromBatch,
    ImageRotate,
    ImageScaleToMaxDimension,
    ImageStitch,
    PadImageNode,
    RepeatImageBatch,
    ResizeAndPadImage,
)
from dinkster_schema import schema_from_wire, schema_to_wire, validate_replacement_references
from dinkster_schema.replace import rule_to_wire

IMAGE_TRANSFORM_NODES = (
    ImageCrop,
    RepeatImageBatch,
    ImageFromBatch,
    ImageAddNoise,
    ImageStitch,
    ResizeAndPadImage,
    ImageRotate,
    ImageFlip,
    ImageScaleToMaxDimension,
    PadImageNode,
)


def test_image_transform_nodes_have_the_exact_core_interfaces() -> None:
    expected = (
        (
            "dinkster.image.image_crop",
            "Crop Image (DEPRECATED)",
            "image/transform",
            ("image", "width", "height", "x", "y"),
        ),
        (
            "dinkster.image.repeat_image_batch",
            "Repeat Image Batch",
            "image/batch",
            ("image", "amount"),
        ),
        (
            "dinkster.image.image_from_batch",
            "Get Image from Batch",
            "image/batch",
            ("image", "batch_index", "length"),
        ),
        (
            "dinkster.image.image_add_noise",
            "Add Noise to Image",
            "image/filters",
            ("image", "seed", "strength"),
        ),
        (
            "dinkster.image.image_stitch",
            "Stitch Images",
            "image/transform",
            ("image1", "direction", "match_image_size", "spacing_width", "spacing_color", "image2"),
        ),
        (
            "dinkster.image.resize_and_pad_image",
            "Resize And Pad Image",
            "image/transform",
            ("image", "target_width", "target_height", "padding_color", "interpolation"),
        ),
        ("dinkster.image.image_rotate", "Rotate Image", "image/transform", ("image", "rotation")),
        ("dinkster.image.image_flip", "Flip Image", "image/transform", ("image", "flip_method")),
        (
            "dinkster.image.image_scale_to_max_dimension",
            "Scale Image to Max Dimension",
            "image/upscaling",
            ("image", "upscale_method", "largest_size"),
        ),
        # PadImageNode schema from F bc5f880d; padding choices are dynamic, not scalar inputs.
        ("dinkster.image.pad_image_node", "Pad Image", "image/transform", ("image",)),
    )
    assert all(node in NODES for node in IMAGE_TRANSFORM_NODES)
    assert len(IMAGE_TRANSFORM_NODES) == len(expected)
    for node, (node_type, display_name, category, inputs) in zip(
        IMAGE_TRANSFORM_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            display_name,
            category,
        )
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert tuple(output_spec.type.types for output_spec in schema.outputs) == (("IMAGE",),)


def test_deprecated_longer_edge_resize_publishes_its_successor_mapping() -> None:
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_TRANSFORM_NODES}
    schema = schemas["dinkster.image.image_scale_to_max_dimension"]
    wire = schema_to_wire(schema)
    assert [
        rule_to_wire(rule)
        for rule in schema.replacements
        if rule.from_type == "ResizeImagesByLongerEdge"
    ] == [
        {
            "from": "ResizeImagesByLongerEdge",
            "cases": [
                {
                    "to": "dinkster.image.image_scale_to_max_dimension",
                    "inputs": {
                        "image": {"kind": "copy", "input": "images"},
                        "largest_size": {"kind": "copy", "input": "longer_edge"},
                        "upscale_method": {"kind": "constant", "value": "lanczos"},
                    },
                    "outputs": {"IMAGE": "images"},
                }
            ],
        }
    ]
    assert schema_to_wire(schema_from_wire(wire)) == wire
    assert "ResizeImagesByLongerEdge" not in schemas
    assert validate_replacement_references(schemas) == ()
