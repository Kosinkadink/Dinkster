"""Core CLIP nodes preserve scheduled outputs and isolate layer selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_conditioning.clip import (
    CLIPTextEncode,
)
from dinkster_flux.flux_conditioning import (
    FluxKontextMultiReferenceLatentMethod,
)
from dinkster_hidream.clip_model_specific import CLIPTextEncodeHiDream
from dinkster_misc_models.clip_model_specific import (
    CLIPTextEncodeLumina2,
    CLIPTextEncodePixArtAlpha,
)


class TextEncoder:
    def __init__(self, layer: int = -1) -> None:
        self.layer = layer

    def clone(self) -> TextEncoder:
        return TextEncoder(self.layer)

    def clip_layer(self, layer: int) -> None:
        self.layer = layer

    def tokenize(self, text: str) -> dict[str, str]:
        return {"literal": text}

    def encode_from_tokens_scheduled(self, tokens: dict[str, str]) -> list[list[object]]:
        return [
            [tokens["literal"], {"layer": self.layer, "start_percent": 0.125}],
            [tokens["literal"] + " next", {"layer": self.layer, "start_percent": 0.75}],
        ]


def test_missing_clip_reports_checkpoint_encoder_error() -> None:
    with pytest.raises(RuntimeError, match="checkpoint does not contain a valid clip"):
        CLIPTextEncode.execute(clip=None, text="prompt")


class ModelTextEncoder:
    def __init__(self) -> None:
        self.options: dict[str, int] = {"unrelated": 17}

    def clone(self) -> ModelTextEncoder:
        result = ModelTextEncoder()
        result.options = self.options.copy()
        return result

    def set_tokenizer_option(self, name: str, value: int) -> None:
        self.options[name] = value

    def tokenize(self, text: str) -> dict[str, str]:
        return {key: f"{key}:{text}" for key in ("l", "g", "t5xxl", "llama", "extra")}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, str], add_dict: dict[str, Any] | None = None
    ) -> list[list[object]]:
        return [[tokens, {"start_percent": 0.25, **(add_dict or {})}]]


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("offset", "offset"),
        ("index", "index"),
        ("uxo/uno", "uxo"),
        ("uso/uno", "uxo"),
        ("index_timestep_zero", "index_timestep_zero"),
        ("index_timestep_zero_isolated", "index_timestep_zero_isolated"),
        ("index_timestep_zero_text_start", "index_timestep_zero_text_start"),
    ),
)
def test_flux_reference_method_preserves_source(method: str, expected: str) -> None:
    source = [["embedding", {"guidance": 2.5, "reference_latents_method": "prior"}]]
    result = FluxKontextMultiReferenceLatentMethod.execute(
        conditioning=source, reference_latents_method=method
    )["CONDITIONING"]
    assert result == [["embedding", {"guidance": 2.5, "reference_latents_method": expected}]]
    assert source == [["embedding", {"guidance": 2.5, "reference_latents_method": "prior"}]]


def test_model_specific_prompts_and_resolution() -> None:
    encoder = ModelTextEncoder()
    pixart = CLIPTextEncodePixArtAlpha.execute(
        clip=encoder, text="literal\n{one|two}", width=832, height=1216
    )["CONDITIONING"]
    assert pixart == [
        [
            encoder.tokenize("literal\n{one|two}"),
            {"start_percent": 0.25, "width": 832, "height": 1216},
        ]
    ]
    hidream: Any = CLIPTextEncodeHiDream.execute(
        clip=encoder, clip_l="left", clip_g="global", t5xxl="long", llama="instruct"
    )["CONDITIONING"]
    assert hidream == [
        [
            {
                "l": "l:left",
                "g": "g:global",
                "t5xxl": "t5xxl:long",
                "llama": "llama:instruct",
                "extra": "extra:global",
            },
            {"start_percent": 0.25},
        ]
    ]
    for key, system in (
        (
            "superior",
            "You are an assistant designed to generate superior images with the superior "
            "degree of image-text alignment based on textual prompts or user prompts.",
        ),
        (
            "alignment",
            "You are an assistant designed to generate high-quality images with the "
            "highest degree of image-text alignment based on textual prompts.",
        ),
    ):
        result = CLIPTextEncodeLumina2.execute(
            clip=encoder, user_prompt="  user\n", system_prompt=key
        )["CONDITIONING"]
        assert result == [
            [encoder.tokenize(f"{system} <Prompt Start>   user\n"), {"start_percent": 0.25}]
        ]
    with pytest.raises(RuntimeError, match="checkpoint does not contain a valid clip"):
        CLIPTextEncodeLumina2.execute(clip=None, user_prompt="", system_prompt="alignment")


class EncoderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="clip-fixture.source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(CLIP=TextEncoder())


class ConditioningText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="clip-fixture.text",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: list[list[Any]]) -> Mapping[str, object]:
        return cls.outputs(text=repr(conditioning))


CLIP_FIXTURES = (EncoderSource, ConditioningText)
