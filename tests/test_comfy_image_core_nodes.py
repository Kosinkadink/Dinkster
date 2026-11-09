"""Public core image interfaces and alpha-preserving inversion."""

import numpy as np
from dinkster_nodes_comfy.image_core import IMAGE_CORE_NODES, ImageInvert, ImageScale


def test_core_image_interfaces_use_exact_ids_and_ordered_inputs() -> None:
    expected = {
        "ImageScale": ("image", "upscale_method", "width", "height", "crop"),
        "ImageScaleBy": ("image", "upscale_method", "scale_by"),
        "ImageInvert": ("image",),
        "ImageBatch": ("image1", "image2"),
        "EmptyImage": ("width", "height", "batch_size", "color"),
        "ImagePadForOutpaint": ("image", "left", "top", "right", "bottom", "feathering"),
    }
    assert {
        node.schema().node_type: tuple(port.id for port in node.schema().inputs)
        for node in IMAGE_CORE_NODES
    } == expected
    for node in IMAGE_CORE_NODES:
        schema = node.schema()
        assert schema.outputs[0].type.types == ("IMAGE",)
        if schema.node_type == "ImagePadForOutpaint":
            assert schema.outputs[1].type.types == ("MASK",)


def test_inversion_preserves_alpha_and_source_pixels() -> None:
    source = np.array([[[[0.0, 0.25, 0.75, 0.125], [1.0, 0.5, -0.5, 0.875]]]], dtype=np.float32)
    original = source.copy()
    result = ImageInvert.execute(image=source)["IMAGE"]
    np.testing.assert_array_equal(
        result, np.array([[[[1.0, 0.75, 0.25, 0.125], [0.0, 0.5, 1.5, 0.875]]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(source, original)
    rgb = ImageInvert.execute(image=source[..., :3])["IMAGE"]
    np.testing.assert_array_equal(
        rgb, np.array([[[[1.0, 0.75, 0.25], [0.0, 0.5, 1.5]]]], dtype=np.float32)
    )


def test_zero_dimension_resize_returns_the_original_value() -> None:
    image = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
    assert (
        ImageScale.execute(
            image=image, upscale_method="bilinear", width=0, height=0, crop="disabled"
        )["IMAGE"]
        is image
    )
