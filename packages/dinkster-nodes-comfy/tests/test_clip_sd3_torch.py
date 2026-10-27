"""SD3 triple-encoder conditioning through ordinary CPU pack hosts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

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
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.clip_sd3 import CLIPTextEncodeSD3

from dinkster.compose import PackSpec, ServingComposer


class TripleEncoder:
    def tokenize(self, text: str) -> dict[str, list[list[int]]]:
        return {
            key: [[offset + ord(character)] for character in text] or [[offset]]
            for key, offset in (("g", 1000), ("l", 2000), ("t5xxl", 3000), ("extra", 4000))
        }

    def encode_from_tokens_scheduled(self, tokens: dict[str, list[list[int]]]) -> list[list[Any]]:
        token_streams = {key: value.copy() for key, value in tokens.items()}
        tensor = torch.tensor([[sum(value[0] for value in stream) for stream in tokens.values()]])
        return [
            [tensor, {"start_percent": 0.25, "token_streams": token_streams}],
            [tensor + 1, {"start_percent": 0.75, "token_streams": token_streams}],
        ]


def _tokens(
    *, clip_l: str, clip_g: str, t5xxl: str, empty_padding: str
) -> dict[str, list[list[int]]]:
    encoder = TripleEncoder()
    result = CLIPTextEncodeSD3.execute(
        clip=encoder,
        clip_l=clip_l,
        clip_g=clip_g,
        t5xxl=t5xxl,
        empty_padding=empty_padding,
    )["CONDITIONING"][0][1]["token_streams"]
    return result


@pytest.mark.parametrize(
    ("clip_l", "clip_g", "expected_g", "expected_l"),
    (
        ("a", "wxyz", [[1119], [1120], [1121], [1122]], [[2097], [2000], [2000], [2000]]),
        ("wxyz", "a", [[1097], [1000], [1000], [1000]], [[2119], [2120], [2121], [2122]]),
    ),
)
def test_sd3_aligns_asymmetric_clip_streams_in_both_directions(
    clip_l: str, clip_g: str, expected_g: list[list[int]], expected_l: list[list[int]]
) -> None:
    tokens = _tokens(clip_l=clip_l, clip_g=clip_g, t5xxl="t5", empty_padding="none")
    assert tokens["g"] == expected_g
    assert tokens["l"] == expected_l
    assert tokens["t5xxl"] == [[3116], [3053]]
    assert tokens["extra"] == ([[4119], [4120], [4121], [4122]] if clip_g == "wxyz" else [[4097]])


@pytest.mark.parametrize(
    ("empty_padding", "expected"),
    (
        ("none", {"g": [], "l": [], "t5xxl": [], "extra": [[4000]]}),
        ("empty_prompt", {"g": [[1000]], "l": [[2000]], "t5xxl": [[3000]], "extra": [[4000]]}),
    ),
)
def test_sd3_empty_padding_selects_empty_stream_or_empty_prompt(
    empty_padding: str, expected: dict[str, list[list[int]]]
) -> None:
    assert _tokens(clip_l="", clip_g="", t5xxl="", empty_padding=empty_padding) == expected


def test_sd3_schema_and_scheduled_output_match_the_fixed_reference() -> None:
    schema = CLIPTextEncodeSD3.define_schema()
    assert CLIPTextEncodeSD3 in COMFY_NODES
    assert schema.node_type == "CLIPTextEncodeSD3"
    assert schema.display_name == "CLIP Text Encode (SD3)"
    assert schema.category == "model/conditioning/stable diffusion"
    assert schema.search_terms == ("sd3 prompt",)
    assert [input.id for input in schema.inputs] == [
        "clip",
        "clip_l",
        "clip_g",
        "t5xxl",
        "empty_padding",
    ]
    assert schema.inputs[4].advanced
    assert schema.inputs[4].widget.options == ("none", "empty_prompt")
    scheduled = CLIPTextEncodeSD3.execute(
        clip=TripleEncoder(),
        clip_l="l",
        clip_g="gg",
        t5xxl="t",
        empty_padding="none",
    )["CONDITIONING"]
    assert scheduled[0][1]["start_percent"] == 0.25
    assert scheduled[1][1]["start_percent"] == 0.75


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sd3-text-fixture.source",
            outputs=(OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=TripleEncoder())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sd3-text-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> dict[str, object]:
        value = [[tensor.tolist(), metadata] for tensor, metadata in conditioning]
        return cls.outputs(summary=json.dumps(value, sort_keys=True))


@pytest.mark.parametrize("in_process", (True,))
def test_sd3_conditioning_transports_through_both_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(source_root),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            }
        )
        try:
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "sd3_text_fixture.py").write_text(
                "from test_clip_sd3_torch import Inspect, Source\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (Source, Inspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "sd3-text-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["sd3-text-fixture"]\n\n[pack.entry]\n'
                'nodes = "sd3_text_fixture:NODES"\n'
                'types = "sd3_text_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "sd3-text-fixture",
                        group_manifests=() if in_process else (native_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sd3-text-fixture.source"),
                    "encode": GraphNode(
                        "CLIPTextEncodeSD3",
                        {
                            "clip": Link("source", "clip"),
                            "clip_l": "a",
                            "clip_g": "bc",
                            "t5xxl": "def",
                            "empty_padding": "none",
                        },
                    ),
                    "inspect": GraphNode(
                        "sd3-text-fixture.inspect",
                        {"conditioning": Link("encode", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                values = json.loads(result.outputs["inspect"]["summary"].resolve())
                assert values[0][1]["start_percent"] == 0.25
                assert values[1][1]["start_percent"] == 0.75
                assert values[0][1]["token_streams"] == {
                    "g": [[1098], [1099]],
                    "l": [[2097], [2000]],
                    "t5xxl": [[3100], [3101], [3102]],
                    "extra": [[4098], [4099]],
                }
                assert values[1][1]["token_streams"] == values[0][1]["token_streams"]
        finally:
            await composer.close()

    asyncio.run(scenario())
