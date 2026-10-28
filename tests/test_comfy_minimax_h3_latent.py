"""Torch-free MiniMax H3 AV latent schema and discovery."""

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.minimax_h3_latent import EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo


def test_minimax_h3_empty_av_schema_and_cold_discovery() -> None:
    schema = EmptyMiniMaxH3LatentAV.schema()
    integer = TypeExpr.concrete(CORE_INT)
    assert schema.node_type == "EmptyMiniMaxH3LatentAV"
    assert schema.display_name == "Empty MiniMax H3 AV Latent"
    assert schema.category == "model/latent/minimax"
    assert schema.inputs[:2] == (
        InputSpec("width", integer, default=1344, widget=NumberWidget(min=32, max=16384, step=32)),
        InputSpec("height", integer, default=768, widget=NumberWidget(min=32, max=16384, step=32)),
    )
    assert schema.inputs[2].id == "length"
    assert schema.inputs[2].default == 124
    assert schema.inputs[2].widget == NumberWidget(min=5, max=3600, step=17)
    assert schema.outputs == (OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),)
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_minimax_h3_image_schema_and_cold_discovery() -> None:
    schema = MiniMaxH3ImageToVideo.schema()
    assert schema.node_type == "MiniMaxH3ImageToVideo"
    assert schema.display_name == "MiniMax H3 Image to Video"
    assert schema.category == "model/conditioning/minimax"
    assert schema.inputs == (
        InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
        InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
        InputSpec(
            "prompt",
            TypeExpr.concrete(CORE_STRING),
            widget=StringWidget(multiline=True, dynamic_prompts=True),
        ),
        *EmptyMiniMaxH3LatentAV.schema().inputs,
        InputSpec("first_frame", TypeExpr.concrete(IMAGE_TYPE), required=False),
        InputSpec("last_frame", TypeExpr.concrete(IMAGE_TYPE), required=False),
    )
    assert schema.outputs == (
        OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert MiniMaxH3ImageToVideo in COMFY_NODES
