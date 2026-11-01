"""Core image transform schemas preserve the ComfyUI workflow contract."""

from dinkster_nodes_comfy.image_transform import IMAGE_TRANSFORM_NODES


def test_image_transform_nodes_have_the_exact_core_interfaces() -> None:
    expected = (
        (
            "ImageCrop",
            "Crop Image (DEPRECATED)",
            "image/transform",
            ("image", "width", "height", "x", "y"),
        ),
        ("RepeatImageBatch", "Repeat Image Batch", "image/batch", ("image", "amount")),
        (
            "ImageFromBatch",
            "Get Image from Batch",
            "image/batch",
            ("image", "batch_index", "length"),
        ),
        ("ImageAddNoise", "Add Noise to Image", "image/filters", ("image", "seed", "strength")),
        (
            "ImageStitch",
            "Stitch Images",
            "image/transform",
            ("image1", "direction", "match_image_size", "spacing_width", "spacing_color", "image2"),
        ),
        (
            "ResizeAndPadImage",
            "Resize And Pad Image",
            "image/transform",
            ("image", "target_width", "target_height", "padding_color", "interpolation"),
        ),
        ("ImageRotate", "Rotate Image", "image/transform", ("image", "rotation")),
        ("ImageFlip", "Flip Image", "image/transform", ("image", "flip_method")),
        (
            "ImageScaleToMaxDimension",
            "Scale Image to Max Dimension",
            "image/upscaling",
            ("image", "upscale_method", "largest_size"),
        ),
        # PadImageNode schema from F bc5f880d; padding choices are dynamic, not scalar inputs.
        ("PadImageNode", "Pad Image", "image/transform", ("image",)),
    )
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
