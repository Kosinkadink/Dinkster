"""Core text encoding and CLIP layer selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
)


class CLIPTextEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPTextEncode",
            display_name="CLIP Text Encode (Prompt)",
            category="model/conditioning",
            search_terms=(
                "text",
                "prompt",
                "text prompt",
                "positive prompt",
                "negative prompt",
                "encode text",
                "text encoder",
                "encode prompt",
            ),
            description=(
                "Encodes a text prompt using a CLIP model into an embedding that can be "
                "used to guide the diffusion model towards generating specific images."
            ),
            inputs=(
                InputSpec(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    doc="The text to be encoded.",
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "clip",
                    TypeExpr.concrete(CLIP_TYPE),
                    doc="The CLIP model used for encoding the text.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "CONDITIONING",
                    TypeExpr.concrete(CONDITIONING_TYPE),
                    doc=(
                        "A conditioning containing the embedded text used to guide "
                        "the diffusion model towards generating specific images."
                    ),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, clip: Any, text: str) -> Mapping[str, object]:
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not "
                "contain a valid clip or text encoder model."
            )
        tokens = clip.tokenize(text)
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))


class CLIPSetLastLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPSetLastLayer",
            display_name="CLIP Set Last Layer",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "stop_at_clip_layer",
                    TypeExpr.concrete(CORE_INT),
                    default=-1,
                    widget=NumberWidget(min=-24, max=-1, step=1),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls, *, clip: Any, stop_at_clip_layer: int) -> Mapping[str, object]:
        selected = clip.clone()
        selected.clip_layer(stop_at_clip_layer)
        return cls.outputs(CLIP=selected)


class CLIPVisionEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPVisionEncode",
            display_name="CLIP Vision Encode",
            category="model/conditioning",
            inputs=(
                InputSpec("clip_vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "crop",
                    TypeExpr.concrete(CORE_COMBO),
                    default="center",
                    widget=ComboWidget(("center", "none")),
                ),
            ),
            outputs=(OutputSpec("CLIP_VISION_OUTPUT", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, clip_vision: Any, image: Any, crop: str) -> Mapping[str, object]:
        return cls.outputs(
            CLIP_VISION_OUTPUT=clip_vision.encode_image(
                materialize_image(image), crop=crop == "center"
            )
        )


CLIP_NODES = (CLIPTextEncode, CLIPSetLastLayer, CLIPVisionEncode)
