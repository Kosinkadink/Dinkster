"""Core mask node schemas preserve the ComfyUI workflow contract."""

from dinkster_nodes_comfy.mask import MASK_NODES


def test_mask_nodes_have_the_exact_core_interfaces() -> None:
    expected = (
        (
            "LatentCompositeMasked",
            "Latent Composite Masked",
            "model/latent",
            ("destination", "source", "x", "y", "resize_source", "mask"),
            "LATENT",
        ),
        (
            "ImageCompositeMasked",
            "Image Composite Masked",
            "image/compositing",
            ("destination", "source", "x", "y", "resize_source", "mask"),
            "IMAGE",
        ),
        ("MaskToImage", "Convert Mask to Image", "image/mask", ("mask",), "IMAGE"),
        ("ImageToMask", "Convert Image to Mask", "image/mask", ("image", "channel"), "MASK"),
        (
            "ImageColorToMask",
            "Convert Image Color to Mask",
            "image/mask",
            ("image", "color"),
            "MASK",
        ),
        ("SolidMask", "Create Solid Mask", "image/mask", ("value", "width", "height"), "MASK"),
        ("InvertMask", "Invert Mask", "image/mask", ("mask",), "MASK"),
        ("CropMask", "Crop Mask", "image/mask", ("mask", "x", "y", "width", "height"), "MASK"),
        (
            "MaskComposite",
            "Combine Masks",
            "image/mask",
            ("destination", "source", "x", "y", "operation"),
            "MASK",
        ),
        (
            "FeatherMask",
            "Feather Mask",
            "image/mask",
            ("mask", "left", "top", "right", "bottom"),
            "MASK",
        ),
        ("GrowMask", "Grow Mask", "image/mask", ("mask", "expand", "tapered_corners"), "MASK"),
        ("ThresholdMask", "Threshold Mask", "image/mask", ("mask", "value"), "MASK"),
    )
    assert len(MASK_NODES) == len(expected)
    for node, (node_type, display_name, category, inputs, output) in zip(
        MASK_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            display_name,
            category,
        )
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert tuple(output_spec.type.types for output_spec in schema.outputs) == ((output,),)
