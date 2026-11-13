"""SDXL prompt streams and conditioning metadata through the public nodes."""

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
from dinkster_sd.clip_sdxl import CLIPTextEncodeSDXL


class SDXLEncoder:
    def tokenize(self, text: str) -> dict[str, list[str]]:
        chunks = text.split("|")
        return {"g": [f"g:{chunk}" for chunk in chunks], "l": [f"l:{c}" for c in chunks]}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, list[str]], *, add_dict: dict[str, object]
    ) -> list[list[object]]:
        return [
            [tokens, {**add_dict, "start_percent": 0.125}],
            [tokens, {**add_dict, "start_percent": 0.75}],
        ]


_SIZE = {
    "width": 768,
    "height": 1280,
    "crop_w": 13,
    "crop_h": 27,
    "target_width": 1536,
    "target_height": 896,
}


@pytest.mark.parametrize(
    ("text_g", "text_l", "global_tokens", "local_tokens"),
    (
        ("red", "blue", ["g:red"], ["l:blue"]),
        ("red|green|white", "blue", ["g:red", "g:green", "g:white"], ["l:blue", "l:", "l:"]),
        ("red", "blue|black|gold", ["g:red", "g:", "g:"], ["l:blue", "l:black", "l:gold"]),
        ("", "blue|black", ["g:", "g:"], ["l:blue", "l:black"]),
    ),
)
def test_sdxl_pads_only_the_shorter_prompt_stream(
    text_g: str, text_l: str, global_tokens: list[str], local_tokens: list[str]
) -> None:
    result = CLIPTextEncodeSDXL.execute(clip=SDXLEncoder(), text_g=text_g, text_l=text_l, **_SIZE)[
        "CONDITIONING"
    ]
    assert result == [
        [{"g": global_tokens, "l": local_tokens}, {**_SIZE, "start_percent": 0.125}],
        [{"g": global_tokens, "l": local_tokens}, {**_SIZE, "start_percent": 0.75}],
    ]


class SDXLSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-fixture.source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(CLIP=SDXLEncoder())


class SDXLInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> Mapping[str, object]:
        return cls.outputs(text=repr(conditioning))


SDXL_FIXTURES = (SDXLSource, SDXLInspect)
