"""Model-specific text encoding and tokenizer options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    first_party_schema,
)


class CLIPTextEncodePixArtAlpha(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="CLIPTextEncodePixArtAlpha",
            display_name="CLIP Text Encode (PixArt Alpha)",
            category="model/conditioning/pixart",
            search_terms=("pixart prompt",),
            description=(
                "Encodes text and sets the resolution conditioning for PixArt Alpha. "
                "Does not apply to PixArt Sigma."
            ),
            inputs=(
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                InputSpec(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, clip: Any, width: int, height: int, text: str) -> Mapping[str, object]:
        tokens = clip.tokenize(text)
        return cls.outputs(
            CONDITIONING=clip.encode_from_tokens_scheduled(
                tokens, add_dict={"width": width, "height": height}
            )
        )


class CLIPTextEncodeLumina2(Node):
    SYSTEM_PROMPT = {
        "superior": "You are an assistant designed to generate superior images with the superior "
        "degree of image-text alignment based on textual prompts or user prompts.",
        "alignment": "You are an assistant designed to generate high-quality images with the "
        "highest degree of image-text alignment based on textual prompts.",
    }

    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="CLIPTextEncodeLumina2",
            display_name="CLIP Text Encode (Lumina 2)",
            category="model/conditioning/lumina",
            search_terms=("lumina prompt",),
            description=(
                "Encodes a system prompt and a user prompt using a CLIP model into an embedding "
                "that can be used to guide the diffusion model towards generating specific images."
            ),
            inputs=(
                InputSpec(
                    "system_prompt",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(tuple(ComboOption(key) for key in cls.SYSTEM_PROMPT)),
                ),
                InputSpec(
                    "user_prompt",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, clip: Any, user_prompt: str, system_prompt: str) -> Mapping[str, object]:
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not "
                "contain a valid clip or text encoder model."
            )
        prompt = f"{cls.SYSTEM_PROMPT[system_prompt]} <Prompt Start> {user_prompt}"
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))


class CLIPTextEncodeKandinsky5(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="CLIPTextEncodeKandinsky5",
            display_name="CLIP Text Encode (Kandinsky 5)",
            category="model/conditioning/kandinsky",
            search_terms=("kandinsky prompt",),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_STRING),
                        widget=StringWidget(multiline=True, dynamic_prompts=True),
                    )
                    for name in ("clip_l", "qwen25_7b")
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, clip: Any, clip_l: str, qwen25_7b: str) -> Mapping[str, object]:
        tokens = clip.tokenize(clip_l)
        tokens["qwen25_7b"] = clip.tokenize(qwen25_7b)["qwen25_7b"]
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))
