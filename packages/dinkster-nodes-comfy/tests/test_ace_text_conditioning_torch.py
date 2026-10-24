"""CPU ACE Step text conditioning and ordinary-manifest execution."""

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
from dinkster_nodes_comfy.ace_text_conditioning import (
    TextEncodeAceStepAudio,
    TextEncodeAceStepAudio15,
)

from dinkster.compose import PackSpec, ServingComposer


class AsymmetricClip:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_conditioning: list[list[Any]] = []

    def tokenize(self, tags: str, **options: object) -> dict[str, object]:
        self.calls.append((tags, options))
        return {"tags": tags, **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        self.last_conditioning = [
            [torch.tensor([[1.0, 3.0]], dtype=torch.float64), {"source": "first", **tokens}],
            [torch.tensor([[2.0]], dtype=torch.float32), {"source": "second", **tokens}],
        ]
        return self.last_conditioning


def test_ace_text_encoders_delegate_exact_options_and_copy_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_nodes_comfy.ace_text_conditioning.initialize_inference_process", lambda: None
    )
    ace = AsymmetricClip()
    result = TextEncodeAceStepAudio.execute(
        clip=ace, tags="soft drums", lyrics="first\nsecond", lyrics_strength=3.25
    )["CONDITIONING"]
    assert ace.calls == [("soft drums", {"lyrics": "first\nsecond"})]
    assert [entry[1]["lyrics_strength"] for entry in result] == [3.25, 3.25]
    assert [entry[1]["source"] for entry in result] == ["first", "second"]
    assert all(
        entry[1] is not original[1]
        for entry, original in zip(result, ace.last_conditioning, strict=True)
    )
    assert all(
        entry[0] is original[0]
        for entry, original in zip(result, ace.last_conditioning, strict=True)
    )

    ace15 = AsymmetricClip()
    result15 = TextEncodeAceStepAudio15.execute(
        clip=ace15,
        tags="bright bass",
        lyrics="la",
        seed=0xFFFFFFFFFFFFFFFF,
        bpm=171,
        duration=93.7,
        timesignature="6",
        language="yue",
        keyscale="Bb minor",
        generate_audio_codes=False,
        cfg_scale=7.4,
        temperature=1.23,
        top_p=0.43,
        top_k=19,
        min_p=0.017,
    )["CONDITIONING"]
    assert ace15.calls == [
        (
            "bright bass",
            {
                "lyrics": "la",
                "bpm": 171,
                "duration": 93.7,
                "timesignature": 6,
                "language": "yue",
                "keyscale": "Bb minor",
                "seed": 0xFFFFFFFFFFFFFFFF,
                "generate_audio_codes": False,
                "cfg_scale": 7.4,
                "temperature": 1.23,
                "top_p": 0.43,
                "top_k": 19,
                "min_p": 0.017,
            },
        )
    ]
    assert [entry[1]["source"] for entry in result15] == ["first", "second"]
    assert result15[0][0].dtype is torch.float64 and result15[1][0].dtype is torch.float32
    assert all(
        entry[0] is original[0]
        for entry, original in zip(result15, ace15.last_conditioning, strict=True)
    )


_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


class AceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ace-text-fixture.source",
            outputs=(OutputSpec("clip", _CLIP),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=AsymmetricClip())


class AceInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ace-text-fixture.inspect",
            inputs=(
                InputSpec("ace", _CONDITIONING),
                InputSpec("ace15", _CONDITIONING),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, ace: Any, ace15: Any) -> dict[str, object]:
        assert [entry[1]["lyrics_strength"] for entry in ace] == [2.5, 2.5]
        assert [entry[1]["source"] for entry in ace] == ["first", "second"]
        assert [entry[1]["tags"] for entry in ace] == ["soft", "soft"]
        assert [entry[1]["lyrics"] for entry in ace] == ["words", "words"]
        assert torch.equal(ace[0][0], torch.tensor([[1.0, 3.0]], dtype=torch.float64))
        assert torch.equal(ace[1][0], torch.tensor([[2.0]], dtype=torch.float32))
        assert ace[0][0].dtype is torch.float64 and ace[1][0].dtype is torch.float32
        assert [entry[1]["tags"] for entry in ace15] == ["bright", "bright"]
        assert [entry[1]["lyrics"] for entry in ace15] == ["la", "la"]
        assert [entry[1]["bpm"] for entry in ace15] == [171, 171]
        assert [entry[1]["duration"] for entry in ace15] == [93.7, 93.7]
        assert torch.equal(ace15[0][0], torch.tensor([[1.0, 3.0]], dtype=torch.float64))
        assert torch.equal(ace15[1][0], torch.tensor([[2.0]], dtype=torch.float32))
        assert ace15[0][1]["timesignature"] == 6
        assert ace15[0][1]["seed"] == 0xFFFFFFFFFFFFFFFF
        assert ace15[0][1]["language"] == "yue"
        assert ace15[0][1]["keyscale"] == "Bb minor"
        assert ace15[0][1]["generate_audio_codes"] is False
        assert ace15[0][1]["cfg_scale"] == 7.4
        assert ace15[0][1]["temperature"] == 1.23
        assert ace15[0][1]["top_p"] == 0.43
        assert ace15[0][1]["top_k"] == 19
        assert ace15[0][1]["min_p"] == 0.017
        return cls.outputs(result="ace conditioning exact")


@pytest.mark.parametrize("in_process", (True, False))
def test_ace_text_conditioning_executes_twice_in_both_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
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
            (tmp_path / "ace_text_fixture.py").write_text(
                "from test_ace_text_conditioning_torch import AceInspect, AceSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, AceSource, AceInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (package / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["ace-text-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "ace_text_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "ace_text_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("ace-text-fixture.source"),
                    "ace": GraphNode(
                        "TextEncodeAceStepAudio",
                        {
                            "clip": Link("source", "clip"),
                            "tags": "soft",
                            "lyrics": "words",
                            "lyrics_strength": 2.5,
                        },
                    ),
                    "ace15": GraphNode(
                        "TextEncodeAceStepAudio1.5",
                        {
                            "clip": Link("source", "clip"),
                            "tags": "bright",
                            "lyrics": "la",
                            "seed": 0xFFFFFFFFFFFFFFFF,
                            "bpm": 171,
                            "duration": 93.7,
                            "timesignature": "6",
                            "language": "yue",
                            "keyscale": "Bb minor",
                            "generate_audio_codes": False,
                            "cfg_scale": 7.4,
                            "temperature": 1.23,
                            "top_p": 0.43,
                            "top_k": 19,
                            "min_p": 0.017,
                        },
                    ),
                    "inspect": GraphNode(
                        "ace-text-fixture.inspect",
                        {
                            "ace": Link("ace", "CONDITIONING"),
                            "ace15": Link("ace15", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "ace conditioning exact"
        finally:
            await composer.close()

    asyncio.run(scenario())
