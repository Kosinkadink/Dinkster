"""Model text encoders preserve tensor conditioning through ordinary pack hosts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
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
from dinkster_conditioning.clip_model_specific import CLIPTextEncodeControlnet


class TensorEncoder:
    def __init__(self) -> None:
        self.options: dict[str, int] = {}

    def clone(self) -> TensorEncoder:
        result = TensorEncoder()
        result.options = self.options.copy()
        return result

    def set_tokenizer_option(self, name: str, value: int) -> None:
        self.options[name] = value

    def tokenize(self, text: str) -> dict[str, int]:
        return {key: len(text) + offset for offset, key in enumerate(("l", "g", "t5xxl", "llama"))}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, int], add_dict: dict[str, int] | None = None
    ) -> list[list[Any]]:
        tensor = torch.tensor([[list(tokens.values())]], dtype=torch.float32)
        return [
            [tensor, {"start_percent": 0.25, **self.options, **(add_dict or {})}],
            [tensor + 1, {"start_percent": 0.75}],
        ]

    def encode_from_tokens(
        self, tokens: dict[str, int], return_pooled: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert return_pooled
        return torch.tensor([tokens["l"]], dtype=torch.float32), torch.tensor(
            [tokens["g"]], dtype=torch.float32
        )


def test_controlnet_preserves_embeddings_and_metadata() -> None:
    embedding = torch.tensor([[[3.0, -2.0]]])
    source = [
        [embedding, {"tag": "first", "pooled_output_controlnet": "old"}],
        [embedding + 2, {"tag": "second", "start_percent": 0.7}],
    ]
    result = CLIPTextEncodeControlnet.execute(
        clip=TensorEncoder(), conditioning=source, text="abc"
    )["CONDITIONING"]
    for index, (value, options) in enumerate(result):
        assert value is source[index][0]
        assert options is not source[index][1]
        assert options["tag"] == ("first", "second")[index]
        torch.testing.assert_close(
            options["cross_attn_controlnet"], torch.tensor([3.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            options["pooled_output_controlnet"], torch.tensor([4.0]), rtol=0, atol=0
        )
    assert source[0][1] == {"tag": "first", "pooled_output_controlnet": "old"}
    assert source[1][1] == {"tag": "second", "start_percent": 0.7}


class EncoderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="encoder-fixture.source",
            outputs=(OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=TensorEncoder())


class InspectConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="encoder-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> dict[str, object]:
        def plain(value: Any) -> Any:
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value.tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        return cls.outputs(text=json.dumps(plain(conditioning), sort_keys=True))


FIXTURES = (EncoderSource, InspectConditioning)
