"""Native Hunyuan image and image-to-video text conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_clip_vision_output,
)

_PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\n"
    "Describe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, "
    "physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


class CLIPTextEncodeHunyuanDiT(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPTextEncodeHunyuanDiT",
            display_name="CLIP Text Encode (Hunyuan Image)",
            category="model/conditioning/hunyuan image",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "bert",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "mt5xl",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls, *, clip: Any, bert: str, mt5xl: str) -> Mapping[str, object]:
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))


class TextEncodeHunyuanVideo_ImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeHunyuanVideo_ImageToVideo",
            category="model/conditioning/hunyuan video",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
                InputSpec(
                    "prompt",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "image_interleave",
                    TypeExpr.concrete(CORE_INT),
                    default=2,
                    widget=NumberWidget(min=1, max=512),
                    doc=(
                        "How much the image influences things vs the text prompt. "
                        "Higher number means more influence from the text prompt."
                    ),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, clip: Any, clip_vision_output: object, prompt: str, image_interleave: int = 2
    ) -> Mapping[str, object]:
        vision = materialize_clip_vision_output(clip_vision_output)
        tokens = clip.tokenize(
            prompt,
            llama_template=_PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
            image_embeds=vision.mm_projected,
            image_interleave=image_interleave,
        )
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))


HUNYUAN_TEXT_NODES = (CLIPTextEncodeHunyuanDiT, TextEncodeHunyuanVideo_ImageToVideo)
