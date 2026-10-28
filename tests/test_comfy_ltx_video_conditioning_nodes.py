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
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_video_conditioning import LTX_VIDEO_CONDITIONING_NODES


def test_ltx_video_conditioning_schemas() -> None:
    expected = (
        (
            "LTXVImgToVideo",
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
            "LTXVImgToVideoInplace",
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
            "LTXVConditioning",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("frame_rate", CORE_FLOAT),
            ),
            (("positive", CONDITIONING_TYPE), ("negative", CONDITIONING_TYPE)),
        ),
        (
            "LTXVCropGuides",
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
    assert len(LTX_VIDEO_CONDITIONING_NODES) == len(expected)
    for node, (name, inputs, outputs) in zip(LTX_VIDEO_CONDITIONING_NODES, expected, strict=True):
        schema = node.schema()
        assert node.__name__ == schema.node_type == name
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == outputs
        assert all(item.required for item in schema.inputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert node in COMFY_NODES
