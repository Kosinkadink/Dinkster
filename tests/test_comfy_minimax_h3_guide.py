"""Torch-free MiniMax H3 guide schema and discovery."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.minimax_h3_latent import MiniMaxH3AddGuide


def test_minimax_h3_add_guide_schema_and_cold_discovery() -> None:
    schema = MiniMaxH3AddGuide.schema()
    assert (schema.node_type, schema.display_name, schema.category) == (
        "MiniMaxH3AddGuide",
        "Add Guide for MiniMax H3",
        "model/conditioning/minimax",
    )
    assert schema.inputs == (
        InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        InputSpec(
            "vae",
            TypeExpr.concrete(VAE_TYPE),
            required=False,
            doc="Video VAE, needed when an image is connected.",
        ),
        InputSpec(
            "audio_vae",
            TypeExpr.concrete(VAE_TYPE),
            required=False,
            doc="Audio VAE, needed when an audio is connected.",
        ),
        InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
        InputSpec(
            "image",
            TypeExpr.concrete(IMAGE_TYPE),
            required=False,
            doc=(
                "Image or video frames to anchor. Multi-frame batches are anchored as a clip and "
                "cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. "
                "Batches shorter than 5 frames use only the first image."
            ),
        ),
        InputSpec(
            "audio",
            TypeExpr.concrete(AUDIO_TYPE),
            required=False,
            doc=(
                "Soundtrack to anchor starting at the same frame index, cropped to the video's "
                "remaining duration."
            ),
        ),
        InputSpec(
            "frame_idx",
            TypeExpr.concrete(CORE_INT),
            default=0,
            widget=NumberWidget(min=-9999, max=9999),
            doc=(
                "Frame index to anchor the image or the clip's first frame at. Negative values are "
                "counted from the end of the video."
            ),
        ),
    )
    assert schema.outputs == (OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),)
    assert schema_from_wire(schema_to_wire(schema)) == schema
