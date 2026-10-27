"""YuE2 graph defaults and conditioning through ordinary native pack hosts.

Expected-value provenance: ComfyUI F bc5f880dbdd43e426077f42ccec53bb577ffee9f,
comfy_extras/nodes_yue2.py, YuE2GenerateABC and YuE2GenerateMusic; frame rate
from comfy/text_encoders/yue2.py at that same immutable revision:
FRAMES_PER_SECOND = 25. No expected value is a recorded model prediction.

ABC inputs: controlled Clip below, style="piano", lyrics="verse", seed=5,
mode="melody"; omitted controls use schema defaults max_abc_tokens=8192,
temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005, penalty_window=100.
Music inputs: same Clip/style/lyrics, seed=7, mode="full", max_duration=2.25,
abc="X:5" or omitted (schema default ""), cfg_scale=0.375 or omitted (schema
default 1.0); other schema defaults are temperature=1.0, top_p=0.95, top_k=100,
repetition_penalty=1.2. execute uses max(1, round(max_duration * FRAMES_PER_SECOND)),
sets mode="off" when not abc.strip(), and returns the first yue2_frames / fps.
The hand-chosen fixture returns "X:5", ones(1, 2), and yue2_frames=75; therefore
the hand-derived duration is 75 / 25 = 3.0 and token budget max(1, round(2.25 * 25))
is 56. Summary strings below are hand-derived from the fixture's own formatting.
"""

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
    CORE_FLOAT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class Clip:
    def tokenize(self, style: str, **options: Any) -> dict[str, Any]:
        return {"style": style, **options}

    def generate(self, tokens: dict[str, Any], **options: Any) -> str:
        assert tokens["style"] == "piano"
        assert tokens["lyrics"] == "verse"
        assert tokens["cot"] == "melody"
        assert options["seed"] == 5
        return "X:5"

    def decode(self, generated: str) -> str:
        return generated

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[Any]:
        # Hand-chosen frame count; exact rate and duration derivation are documented above.
        return [[torch.ones(1, 2), {"yue2_frames": 75, "observed": dict(tokens)}]]


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="yue2-host-fixture.source",
            outputs=(OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=Clip())


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="yue2-host-fixture.consumer",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("seconds", TypeExpr.concrete(CORE_FLOAT)),
                InputSpec("expected_cfg", TypeExpr.concrete(CORE_FLOAT)),
                InputSpec("expected_abc", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Any, seconds: float, expected_cfg: float, expected_abc: str
    ) -> dict[str, object]:
        assert seconds == 3.0  # Hand-derived 75 / 25; immutable sources and inputs above.
        assert torch.equal(conditioning[0][0], torch.ones(1, 2))
        observed = conditioning[0][1]["observed"]
        # YuE2GenerateMusic: exact inputs, source revision and expressions above.
        assert observed["max_tokens"] == 56
        assert observed["cot"] == ("full" if expected_abc else "off")
        assert observed["abc"] == expected_abc
        assert observed["cfg_scale"] == expected_cfg
        assert observed["style"] == "piano"
        assert observed["lyrics"] == "verse"
        assert observed["seed"] == 7
        return cls.outputs(summary=f"{expected_abc}:{expected_cfg}")


HOST_NODES = (Source, Consumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_yue2_cfg_defaults_and_explicit_values_in_native_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        (tmp_path / "yue2_host_entry.py").write_text(
            "from test_yue2_hosts_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="yue2-host-fixture"\npublisher="dinkster"\n'
            'namespaces=["yue2-host-fixture"]\n[pack.entry]\n'
            'nodes="yue2_host_entry:HOST_NODES"\n'
            'types="yue2_host_entry:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        paths = (tmp_path, Path(__file__).parent, checkout / "src")
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (*map(str, paths), *map(str, sorted(checkout.glob("packages/*/src"))))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "yue2-host-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            nodes = {
                "source": GraphNode("yue2-host-fixture.source"),
                "abc": GraphNode(
                    "YuE2GenerateABC",
                    {
                        "clip": Link("source", "clip"),
                        "style": "piano",
                        "lyrics": "verse",
                        "seed": 5,
                        "mode": "melody",
                    },
                ),
            }
            targets = []
            for abc in ("score", "empty"):
                for cfg in ("default", "explicit"):
                    name = f"{abc}-{cfg}"
                    nodes[name] = GraphNode(
                        "YuE2GenerateMusic",
                        {
                            "clip": Link("source", "clip"),
                            "style": "piano",
                            "lyrics": "verse",
                            "seed": 7,
                            "mode": "full",
                            "max_duration": 2.25,
                            **({"abc": Link("abc", "abc")} if abc == "score" else {}),
                            **({"cfg_scale": 0.375} if cfg == "explicit" else {}),
                        },
                    )
                    # YuE2GenerateMusic.define_schema at the revision above:
                    # cfg_scale default=1.0, optional=True, even for empty ABC.
                    nodes[f"check-{name}"] = GraphNode(
                        "yue2-host-fixture.consumer",
                        {
                            "conditioning": Link(name, "CONDITIONING"),
                            "seconds": Link(name, "seconds"),
                            "expected_cfg": 0.375 if cfg == "explicit" else 1.0,
                            "expected_abc": "X:5" if abc == "score" else "",
                        },
                    )
                    targets.append(f"check-{name}")
            graph = Graph(nodes=nodes)
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, targets)
                for abc in ("score", "empty"):
                    for cfg in ("default", "explicit"):
                        # Hand-derived fixture string; not a model prediction.
                        expected = f"{'X:5' if abc == 'score' else ''}:"
                        expected += str(0.375 if cfg == "explicit" else 1.0)
                        assert result.outputs[f"check-{abc}-{cfg}"]["summary"].resolve() == expected
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
