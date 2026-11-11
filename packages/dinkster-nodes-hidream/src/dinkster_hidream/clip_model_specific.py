"""Model-specific text encoding and tokenizer options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    with_identity_replacement,
)


class CLIPTextEncodeHiDream(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.hidream.clip_text_encode_hi_dream",
            display_name="CLIP Text Encode (HiDream)",
            category="model/conditioning/hidream",
            search_terms=("hidream prompt",),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_STRING),
                        widget=StringWidget(multiline=True, dynamic_prompts=True),
                    )
                    for name in ("clip_l", "clip_g", "t5xxl", "llama")
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )
        return with_identity_replacement(schema, "CLIPTextEncodeHiDream")

    @classmethod
    def execute(
        cls, *, clip: Any, clip_l: str, clip_g: str, t5xxl: str, llama: str
    ) -> Mapping[str, object]:
        tokens = clip.tokenize(clip_g)
        tokens["l"] = clip.tokenize(clip_l)["l"]
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        tokens["llama"] = clip.tokenize(llama)["llama"]
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))
