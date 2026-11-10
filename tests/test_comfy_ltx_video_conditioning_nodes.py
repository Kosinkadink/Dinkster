"""Torch-free schemas for native LTXV video conditioning nodes."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_ltx import NODES
from dinkster_ltx.ltx_video_conditioning import (
    LTXVConditioning,
    LTXVCropGuides,
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
)


def test_ltx_video_conditioning_schemas() -> None:
    nodes = (LTXVImgToVideo, LTXVImgToVideoInplace, LTXVConditioning, LTXVCropGuides)
    expected = (
        (
            "dinkster.ltx.ltxv_img_to_video",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("vae", VAE_TYPE),
                ("image", IMAGE_TYPE),
                ("width", CORE_INT),
                ("height", CORE_INT),
                ("length", CORE_INT),
                ("batch_size", CORE_INT),
                ("strength", CORE_FLOAT),
            ),
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("latent", LATENT_TYPE),
            ),
        ),
        (
            "dinkster.ltx.ltxv_img_to_video_inplace",
            (
                ("vae", VAE_TYPE),
                ("image", IMAGE_TYPE),
                ("latent", LATENT_TYPE),
                ("strength", CORE_FLOAT),
                ("bypass", CORE_BOOLEAN),
            ),
            (("latent", LATENT_TYPE),),
        ),
        (
            "dinkster.ltx.ltxv_conditioning",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("frame_rate", CORE_FLOAT),
            ),
            (("positive", CONDITIONING_TYPE), ("negative", CONDITIONING_TYPE)),
        ),
        (
            "dinkster.ltx.ltxv_crop_guides",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("latent", LATENT_TYPE),
            ),
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("latent", LATENT_TYPE),
            ),
        ),
    )
    assert len(nodes) == len(expected)
    for node, (name, inputs, outputs) in zip(nodes, expected, strict=True):
        schema = node.schema()
        assert schema.node_type == name
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == outputs
        assert all(item.required for item in schema.inputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert NODES.count(node) == 1
