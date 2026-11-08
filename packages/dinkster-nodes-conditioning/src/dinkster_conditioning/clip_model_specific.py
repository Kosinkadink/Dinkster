"""Model-specific text encoding and tokenizer options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
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
    first_party_schema,
)


class CLIPTextEncodeControlnet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return first_party_schema(
            NodeSchema(
                node_type="CLIPTextEncodeControlnet",
                display_name="CLIP Text Encode (Controlnet)",
                category="model/conditioning",
                inputs=(
                    InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                    InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                    InputSpec(
                        "text",
                        TypeExpr.concrete(CORE_STRING),
                        widget=StringWidget(multiline=True, dynamic_prompts=True),
                    ),
                ),
                outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
            )
        )

    @classmethod
    def execute(cls, *, clip: Any, conditioning: Any, text: str) -> Mapping[str, object]:
        cond, pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
        result: list[list[Any]] = []
        for embedding, metadata in conditioning:
            options = metadata.copy()
            options["cross_attn_controlnet"] = cond
            options["pooled_output_controlnet"] = pooled
            result.append([embedding, options])
        return cls.outputs(CONDITIONING=result)


class T5TokenizerOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return first_party_schema(
            NodeSchema(
                node_type="T5TokenizerOptions",
                display_name="T5 Tokenizer Options",
                category="model/conditioning",
                inputs=(
                    InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                    *(
                        InputSpec(
                            name,
                            TypeExpr.concrete(CORE_INT),
                            default=0,
                            widget=NumberWidget(min=0, max=10000, step=1),
                        )
                        for name in ("min_padding", "min_length")
                    ),
                ),
                outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
            )
        )

    @classmethod
    def execute(cls, *, clip: Any, min_padding: int, min_length: int) -> Mapping[str, object]:
        selected = clip.clone()
        for name in ("t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"):
            selected.set_tokenizer_option(f"{name}_min_padding", min_padding)
            selected.set_tokenizer_option(f"{name}_min_length", min_length)
        return cls.outputs(CLIP=selected)
