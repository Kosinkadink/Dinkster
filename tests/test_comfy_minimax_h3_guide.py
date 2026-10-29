"""Torch-free MiniMax H3 guide schema and discovery."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
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
from dinkster_nodes_comfy.minimax_h3_reference import MiniMaxH3ReferenceToVideo


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


def test_minimax_h3_reference_members_are_optional_and_roundtrip() -> None:
    schema = MiniMaxH3ReferenceToVideo.schema()
    assert schema.inputs[0] == InputSpec("clip", TypeExpr.concrete(CLIP_TYPE))
    assert schema.inputs[0].required

    expected_families = (
        ("ref_images", "ref_image_", IMAGE_TYPE, 9),
        ("ref_videos", "ref_video_", IMAGE_TYPE, 3),
        ("ref_video_audios", "ref_video_audio_", AUDIO_TYPE, 3),
        ("ref_audios", "ref_audio_", AUDIO_TYPE, 3),
    )
    assert len(schema.input_families) == len(expected_families)
    assert tuple(family.id for family in schema.input_families) == tuple(
        family[0] for family in expected_families
    )
    for family, (family_id, prefix, kind, limit) in zip(
        schema.input_families, expected_families, strict=True
    ):
        assert (family.id, family.member_prefix, family.min_members, family.max_members) == (
            family_id,
            prefix,
            0,
            limit,
        )
        assert not family.required
        assert family.template == (
            InputSpec(prefix.rstrip("_"), TypeExpr.concrete(kind), required=False),
        )

    assert schema_from_wire(schema_to_wire(schema)) == schema
