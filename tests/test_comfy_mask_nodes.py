"""Core mask node schemas preserve the ComfyUI workflow contract."""

from dinkster_image import NODES as IMAGE_NODES
from dinkster_image.ported.mask import (
    CropMask,
    FeatherMask,
    GrowMask,
    ImageColorToMask,
    ImageCompositeMasked,
    ImageToMask,
    InvertMask,
    MaskComposite,
    MaskToImage,
    SolidMask,
    ThresholdMask,
)
from dinkster_latent import NODES as LATENT_NODES
from dinkster_latent.mask import LatentCompositeMasked


def test_mask_nodes_have_the_exact_core_interfaces() -> None:
    nodes = (
        LatentCompositeMasked,
        ImageCompositeMasked,
        MaskToImage,
        ImageToMask,
        ImageColorToMask,
        SolidMask,
        InvertMask,
        CropMask,
        MaskComposite,
        FeatherMask,
        GrowMask,
        ThresholdMask,
    )
    expected = (
        (
            "dinkster.latent.latent_composite_masked",
            "Latent Composite Masked",
            "model/latent",
            ("destination", "source", "x", "y", "resize_source", "mask"),
            "LATENT",
        ),
        (
            "dinkster.image.image_composite_masked",
            "Image Composite Masked",
            "image/compositing",
            ("destination", "source", "x", "y", "resize_source", "mask"),
            "IMAGE",
        ),
        ("dinkster.image.mask_to_image", "Convert Mask to Image", "image/mask", ("mask",), "IMAGE"),
        (
            "dinkster.image.image_to_mask",
            "Convert Image to Mask",
            "image/mask",
            ("image", "channel"),
            "MASK",
        ),
        (
            "dinkster.image.image_color_to_mask",
            "Convert Image Color to Mask",
            "image/mask",
            ("image", "color"),
            "MASK",
        ),
        (
            "dinkster.image.solid_mask",
            "Create Solid Mask",
            "image/mask",
            ("value", "width", "height"),
            "MASK",
        ),
        ("dinkster.image.invert_mask", "Invert Mask", "image/mask", ("mask",), "MASK"),
        (
            "dinkster.image.crop_mask",
            "Crop Mask",
            "image/mask",
            ("mask", "x", "y", "width", "height"),
            "MASK",
        ),
        (
            "dinkster.image.mask_composite",
            "Combine Masks",
            "image/mask",
            ("destination", "source", "x", "y", "operation"),
            "MASK",
        ),
        (
            "dinkster.image.feather_mask",
            "Feather Mask",
            "image/mask",
            ("mask", "left", "top", "right", "bottom"),
            "MASK",
        ),
        (
            "dinkster.image.grow_mask",
            "Grow Mask",
            "image/mask",
            ("mask", "expand", "tapered_corners"),
            "MASK",
        ),
        (
            "dinkster.image.threshold_mask",
            "Threshold Mask",
            "image/mask",
            ("mask", "value"),
            "MASK",
        ),
    )
    assert len(nodes) == len(expected)
    for node, (node_type, display_name, category, inputs, output) in zip(
        nodes, expected, strict=True
    ):
        assert node in (LATENT_NODES if node is LatentCompositeMasked else IMAGE_NODES)
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            display_name,
            category,
        )
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert tuple(output_spec.type.types for output_spec in schema.outputs) == ((output,),)
