"""Kandinsky 5 prompt-stream replacement through ordinary pack hosts."""

from __future__ import annotations

import asyncio
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
from dinkster_nodes_comfy.clip_model_specific import CLIPTextEncodeKandinsky5

from dinkster.compose import PackSpec, ServingComposer


class KandinskyClip:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> dict[str, object]:
        self.calls.append(text)
        return {
            "clip_l": ["clip-l", text],
            "qwen25_7b": ["qwen", text],
            "other": {"prompt": text},
        }

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        return [
            [
                torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                {"tokens": tokens, "source": "first"},
            ],
            [torch.tensor([[3.0]], dtype=torch.float32), {"tokens": tokens, "source": "second"}],
        ]


def test_kandinsky_text_overwrites_only_qwen_stream() -> None:
    clip = KandinskyClip()
    result = CLIPTextEncodeKandinsky5.execute(
        clip=clip, clip_l="wide river", qwen25_7b="night city"
    )["CONDITIONING"]
    assert clip.calls == ["wide river", "night city"]
    for tensor, metadata in result:
        assert metadata["tokens"] == {
            "clip_l": ["clip-l", "wide river"],
            "qwen25_7b": ["qwen", "night city"],
            "other": {"prompt": "wide river"},
        }
        assert metadata["source"] in {"first", "second"}
        assert tensor.dtype in {torch.float64, torch.float32}
    assert torch.equal(result[0][0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert torch.equal(result[1][0], torch.tensor([[3.0]], dtype=torch.float32))


class KandinskySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="kandinsky-text-fixture.source",
            outputs=(OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=KandinskyClip())


class KandinskyInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="kandinsky-text-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> dict[str, object]:
        assert [entry[1]["source"] for entry in conditioning] == ["first", "second"]
        assert [entry[1]["tokens"]["clip_l"] for entry in conditioning] == [
            ["clip-l", "wide river"],
            ["clip-l", "wide river"],
        ]
        assert [entry[1]["tokens"]["qwen25_7b"] for entry in conditioning] == [
            ["qwen", "night city"],
            ["qwen", "night city"],
        ]
        assert [entry[1]["tokens"]["other"] for entry in conditioning] == [
            {"prompt": "wide river"},
            {"prompt": "wide river"},
        ]
        assert torch.equal(conditioning[0][0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))
        assert torch.equal(conditioning[1][0], torch.tensor([[3.0]], dtype=torch.float32))
        return cls.outputs(result="kandinsky prompt streams exact")


@pytest.mark.parametrize("in_process", (True,))
def test_kandinsky_text_runs_twice_in_both_manifest_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(package / "src"))),
            }
        )
        try:
            (tmp_path / "kandinsky_text_fixture.py").write_text(
                "from test_kandinsky_text_torch import KandinskyInspect, KandinskySource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, KandinskySource, KandinskyInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (package / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["kandinsky-text-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "kandinsky_text_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "kandinsky_text_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "kandinsky_text_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("kandinsky-text-fixture.source"),
                    "text": GraphNode(
                        "CLIPTextEncodeKandinsky5",
                        {
                            "clip": Link("source", "clip"),
                            "clip_l": "wide river",
                            "qwen25_7b": "night city",
                        },
                    ),
                    "inspect": GraphNode(
                        "kandinsky-text-fixture.inspect",
                        {"conditioning": Link("text", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["result"].resolve()
                    == "kandinsky prompt streams exact"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
