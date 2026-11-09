"""Native textbox positions with real CPU CLIP and GLIGEN runtime values."""

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
    GLIGEN_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.gligen_conditioning import GLIGENTextBoxApply

from dinkster.compose import PackSpec, ServingComposer


class _Tokenizer:
    def tokenize_with_weights(self, text: str, return_word_ids: bool, **kwargs: Any) -> str:
        assert return_word_ids is False
        return text


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, Any] = {}

    def reset_clip_options(self) -> None:
        self.options = {}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        self.options.update(options)

    def encode_token_weights(self, tokens: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.options["projected_pooled"] is False
        assert self.options["execution_device"] == torch.device("cpu")
        assert tokens == "first textbox\nsecond line"
        return torch.tensor([[999.0]]), torch.tensor([[3.0, -11.0, 0.5]])


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    clip = CLIP(no_init=True)
    clip.cond_stage_model = _TextEncoder()
    clip.patcher = ModelPatcher(
        clip.cond_stage_model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    clip.tokenizer = _Tokenizer()
    clip.tokenizer_options = {}
    clip.layer_idx = None
    clip.load_model = lambda tokens: clip.patcher
    return clip


def _gligen() -> Any:
    initialize_inference_process()
    from dinkster_inference.gligen import Gligen
    from dinkster_inference.model_patcher import CoreModelPatcher

    return CoreModelPatcher(
        Gligen([], torch.nn.Identity(), key_dim=3), torch.device("cpu"), torch.device("cpu")
    )


def _conditioning() -> list[list[Any]]:
    return [
        [
            torch.tensor([[-5.0, 13.0]]),
            {
                "source": "prior",
                "strength": 0.3,
                "gligen": ("position", _gligen(), [(torch.tensor([[17.0]]), 5, 9, 1, 2)]),
            },
        ],
        [torch.tensor([[7.0, -2.0]]), {"source": "fresh", "strength": 0.8}],
    ]


def _apply(conditioning: Any, clip: Any, model: Any, **sizes: int) -> Any:
    return GLIGENTextBoxApply.execute(
        conditioning_to=conditioning,
        clip=clip,
        gligen_textbox_model=model,
        text="first textbox\nsecond line",
        **sizes,
    )["CONDITIONING"]


def test_gligen_textbox_uses_unprojected_pool_and_copies_each_metadata_record() -> None:
    source, clip, model = _conditioning(), _clip(), _gligen()
    result = _apply(source, clip, model, width=81, height=63, x=17, y=39)
    assert result is not source and len(result) == 2
    assert len(source[0][1]["gligen"][2]) == 1 and "gligen" not in source[1][1]
    assert source[0][1]["gligen"][1] is not model
    for index, (embedding, values) in enumerate(result):
        assert embedding is source[index][0]
        assert values is not source[index][1]
        assert values["source"] == source[index][1]["source"]
        assert values["strength"] == source[index][1]["strength"]
        kind, bound_model, positions = values["gligen"]
        assert kind == "position" and bound_model is model
        pooled, *box = positions[-1]
        assert box == [7, 10, 4, 2]
        assert torch.equal(pooled, torch.tensor([[3.0, -11.0, 0.5]]))
        assert len(positions) == (2 if index == 0 else 1)
    assert result[0][1]["gligen"][2][0] is source[0][1]["gligen"][2][0]
    assert _apply([], clip, model, width=8, height=8, x=0, y=0) == []


class GligenTextboxSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="textbox-fixture.source",
            outputs=(
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("model", TypeExpr.concrete(GLIGEN_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(conditioning=_conditioning(), clip=_clip(), model=_gligen())


class GligenTextboxInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="textbox-fixture.inspect",
            inputs=(
                InputSpec("source", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("result", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("model", TypeExpr.concrete(GLIGEN_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, result: Any, model: Any) -> dict[str, object]:
        from dinkster_inference.gligen import Gligen
        from dinkster_inference.model_patcher import CoreModelPatcher

        assert type(model) is CoreModelPatcher and type(model.model) is Gligen
        assert len(source[0][1]["gligen"][2]) == 1 and "gligen" not in source[1][1]
        for index, (embedding, metadata) in enumerate(result):
            assert torch.equal(embedding, source[index][0])
            assert metadata["source"] == source[index][1]["source"]
            assert metadata["strength"] == 0.25
            position, owner, boxes = metadata["gligen"]
            assert position == "position" and owner is model
            assert len(boxes) == (3 if index == 0 else 2)
            for box, expected in zip(boxes[-2:], ((7, 10, 4, 2), (2, 1, 1, 0)), strict=True):
                assert box[1:] == expected
                assert torch.equal(box[0], torch.tensor([[3.0, -11.0, 0.5]]))
        return cls.outputs(summary="textbox positions preserved")


TEXTBOX_FIXTURES = (GligenTextboxSource, GligenTextboxInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_gligen_textbox_crosses_both_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        native = root / "dinkster-pack.toml"
        fixture = tmp_path / "textbox-pack.toml"
        fixture.write_text(
            '[pack]\nname="textbox-fixture"\npublisher="dinkster"\n'
            'namespaces=["textbox-fixture"]\n[pack.entry]\n'
            'nodes="textbox_fixture:TEXTBOX_FIXTURES"\n'
            'types="textbox_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        (tmp_path / "textbox_fixture.py").write_text(
            "from test_gligen_conditioning_torch import TEXTBOX_FIXTURES\n"
            "from dinkster_nodes_comfy import register_inference_types\n",
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(root / "tests"), str(root / "src"))
                ),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="textbox-fixture", group_manifests=manifests
                    )
                )
            common = {
                "clip": Link("source", "clip"),
                "gligen_textbox_model": Link("source", "model"),
                "text": "first textbox\nsecond line",
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("textbox-fixture.source"),
                    "first": GraphNode(
                        "GLIGENTextBoxApply",
                        {
                            **common,
                            "conditioning_to": Link("source", "conditioning"),
                            "width": 81,
                            "height": 63,
                            "x": 17,
                            "y": 39,
                        },
                    ),
                    "second": GraphNode(
                        "GLIGENTextBoxApply",
                        {
                            **common,
                            "conditioning_to": Link("first", "CONDITIONING"),
                            "width": 8,
                            "height": 16,
                            "x": 0,
                            "y": 8,
                        },
                    ),
                    "strength": GraphNode(
                        "ConditioningSetAreaStrength",
                        {"conditioning": Link("second", "CONDITIONING"), "strength": 0.25},
                    ),
                    "inspect": GraphNode(
                        "textbox-fixture.inspect",
                        {
                            "source": Link("source", "conditioning"),
                            "result": Link("strength", "CONDITIONING"),
                            "model": Link("source", "model"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve() == "textbox positions preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
