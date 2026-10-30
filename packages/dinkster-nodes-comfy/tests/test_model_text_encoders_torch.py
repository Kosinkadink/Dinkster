"""Model text encoders preserve tensor conditioning through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.clip_model_specific import CLIPTextEncodeControlnet

from dinkster.compose import PackSpec, ServingComposer


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


@pytest.mark.parametrize("in_process", (True,))
def test_model_encoders_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        test_dir = str(Path(__file__).resolve().parent)
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((test_dir, os.environ.get("PYTHONPATH", "")))}
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "encoder_fixture.py").write_text(
                "from test_model_text_encoders_torch import FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, *FIXTURES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["encoder-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "encoder_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "encoder_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "encoder_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            clip = Link("source", "clip")
            nodes = {
                "source": GraphNode("encoder-fixture.source"),
                "options": GraphNode(
                    "T5TokenizerOptions", {"clip": clip, "min_padding": 7, "min_length": 43}
                ),
                "pixart": GraphNode(
                    "CLIPTextEncodePixArtAlpha",
                    {"clip": Link("options", "CLIP"), "text": "abc", "width": 832, "height": 1216},
                ),
                "lumina": GraphNode(
                    "CLIPTextEncodeLumina2",
                    {"clip": clip, "user_prompt": "x", "system_prompt": "alignment"},
                ),
                "hidream": GraphNode(
                    "CLIPTextEncodeHiDream",
                    {
                        "clip": clip,
                        "clip_l": "a",
                        "clip_g": "bbb",
                        "t5xxl": "ccccc",
                        "llama": "ddddddd",
                    },
                ),
                "controlnet": GraphNode(
                    "CLIPTextEncodeControlnet",
                    {"clip": clip, "conditioning": Link("pixart", "CONDITIONING"), "text": "other"},
                ),
            }
            for name in ("pixart", "lumina", "hidream", "controlnet"):
                nodes[f"{name}-inspect"] = GraphNode(
                    "encoder-fixture.inspect", {"conditioning": Link(name, "CONDITIONING")}
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(
                    Graph(nodes=nodes),
                    [f"{name}-inspect" for name in ("pixart", "lumina", "hidream", "controlnet")],
                )
                values = {
                    name: json.loads(result.outputs[f"{name}-inspect"]["text"].resolve())
                    for name in ("pixart", "lumina", "hidream", "controlnet")
                }
                assert values["pixart"][0][0] == [[[3.0, 4.0, 5.0, 6.0]]]
                assert values["pixart"][1] == [[[[4.0, 5.0, 6.0, 7.0]]], {"start_percent": 0.75}]
                assert values["pixart"][0][1] == {
                    "start_percent": 0.25,
                    "width": 832,
                    "height": 1216,
                    **{
                        f"{key}_{option}": value
                        for key in ("t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl")
                        for option, value in (("min_padding", 7), ("min_length", 43))
                    },
                }
                assert values["hidream"][0] == [[[[1.0, 4.0, 7.0, 10.0]]], {"start_percent": 0.25}]
                prompt = (
                    "You are an assistant designed to generate high-quality images with the "
                    "highest degree of image-text alignment based on textual prompts. "
                    "<Prompt Start> x"
                )
                assert values["lumina"][0] == [
                    [[[float(len(prompt) + i) for i in range(4)]]],
                    {"start_percent": 0.25},
                ]
                assert values["controlnet"][0][0] == values["pixart"][0][0]
                assert values["controlnet"][0][1] == {
                    **values["pixart"][0][1],
                    "cross_attn_controlnet": [5.0],
                    "pooled_output_controlnet": [6.0],
                }
                assert values["controlnet"][1][1] == {
                    "start_percent": 0.75,
                    "cross_attn_controlnet": [5.0],
                    "pooled_output_controlnet": [6.0],
                }
        finally:
            await composer.close()

    asyncio.run(scenario())
