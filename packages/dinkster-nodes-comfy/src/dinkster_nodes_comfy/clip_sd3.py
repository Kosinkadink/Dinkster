"""SD3 triple-encoder text conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_STRING,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

_CLIP = TypeExpr.concrete(CLIP_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


class CLIPTextEncodeSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPTextEncodeSD3",
            display_name="CLIP Text Encode (SD3)",
            category="model/conditioning/stable diffusion",
            search_terms=("sd3 prompt",),
            inputs=(
                InputSpec("clip", _CLIP),
                *(
                    InputSpec(
                        name,
                        _STRING,
                        widget=StringWidget(multiline=True, dynamic_prompts=True),
                    )
                    for name in ("clip_l", "clip_g", "t5xxl")
                ),
                InputSpec(
                    "empty_padding",
                    _COMBO,
                    widget=ComboWidget(("none", "empty_prompt")),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        clip_l: str,
        clip_g: str,
        t5xxl: str,
        empty_padding: str,
    ) -> Mapping[str, object]:
        no_padding = empty_padding == "none"
        tokens = clip.tokenize(clip_g)
        if not clip_g and no_padding:
            tokens["g"] = []

        if not clip_l and no_padding:
            tokens["l"] = []
        else:
            tokens["l"] = clip.tokenize(clip_l)["l"]

        if not t5xxl and no_padding:
            tokens["t5xxl"] = []
        else:
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))
