"""CPU duration prediction with real patchers and caption/head seams."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_duration import LTXVDurationPredictor

from dinkster.compose import PackSpec, ServingComposer


class CaptionModel(torch.nn.Module):
    cross_attention_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    @property
    def diffusion_model(self) -> CaptionModel:
        return self

    def get_dtype_inference(self) -> torch.dtype:
        return torch.float32

    def preprocess_text_embeds(self, context: torch.Tensor, unprocessed: bool) -> torch.Tensor:
        assert context.shape == (1, 3, 6)
        assert context.dtype is torch.float32
        return context + (10 if unprocessed else 0)


def inputs(seconds: float = 3.1) -> tuple[Any, list[Any], Any]:
    initialize_inference_process()
    from dinkster_inference.ldm.lightricks.duration_head import DurationHead
    from dinkster_inference.model_patcher import CoreModelPatcher, ModelPatcher

    head = DurationHead(4, 2, 4, 1, 1, 4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.mlp_out.bias.fill_(math.log(seconds))
    cpu = torch.device("cpu")
    return (
        ModelPatcher(CaptionModel(), cpu, cpu),
        [
            [
                torch.arange(36, dtype=torch.float64).reshape(2, 3, 6),
                {"unprocessed_ltxav_embeds": True},
            ],
            [torch.zeros(1, 1, 1), {}],
        ],
        CoreModelPatcher(head, cpu, cpu),
    )


@pytest.mark.parametrize(
    ("seconds", "minimum", "maximum", "frames"),
    ((3.1, 1.0, 20.0, 25), (0.2, 1.0, 20.0, 17), (8.0, 1.0, 2.0, 17)),
)
def test_real_duration_head_preserves_raw_seconds_and_frame_limits(
    seconds: float,
    minimum: float,
    maximum: float,
    frames: int,
) -> None:
    model, positive, head = inputs(seconds)
    before = positive[0][0].clone()
    result = LTXVDurationPredictor.execute(
        model=model,
        positive=positive,
        duration_head=head,
        frame_rate=10.0,
        min_seconds=minimum,
        max_seconds=maximum,
    )
    assert result["num_frames"] == frames
    assert result["seconds"] == pytest.approx(seconds, rel=1e-6)
    assert torch.equal(positive[0][0], before)


@pytest.mark.parametrize("unprocessed", (False, True))
def test_caption_split_and_flag_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    unprocessed: bool,
) -> None:
    model, positive, head = inputs()
    if not unprocessed:
        positive[0][1] = {}

    def predict(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        expected = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
        if unprocessed:
            expected = expected + 10
        assert torch.equal(video, expected[..., :4])
        assert torch.equal(audio, expected[..., 4:])
        assert not torch.is_grad_enabled()
        return torch.tensor([4.2])

    monkeypatch.setattr(head.model, "forward", predict)
    result = LTXVDurationPredictor.execute(
        model=model,
        positive=positive,
        duration_head=head,
        frame_rate=10.0,
        min_seconds=1.0,
        max_seconds=20.0,
    )
    assert result["num_frames"] == 41
    assert result["seconds"] == pytest.approx(4.2)
    head.model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="not an LTX duration head"):
        LTXVDurationPredictor.execute(
            model=model,
            positive=positive,
            duration_head=head,
            frame_rate=10.0,
            min_seconds=1.0,
            max_seconds=20.0,
        )


class DurationSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="duration-fixture.source",
            outputs=tuple(
                OutputSpec(n, TypeExpr.concrete(t))
                for n, t in (
                    ("model", MODEL_TYPE),
                    ("positive", CONDITIONING_TYPE),
                    ("head", MODEL_PATCH_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        model, positive, head = inputs()
        return cls.outputs(model=model, positive=positive, head=head)


@pytest.mark.parametrize("in_process", (True,))
def test_duration_prediction_crosses_ordinary_hosts_twice(
    tmp_path: Path,
    in_process: bool,
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "duration_fixture.py").write_text(
            "from test_ltx_duration_torch import DurationSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (DurationSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="duration-fixture"\npublisher="dinkster"\n'
            'namespaces=["duration-fixture"]\n[pack.entry]\n'
            'nodes="duration_fixture:NODES"\ntypes="duration_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(package / "src"),
                    )
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "duration-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("duration-fixture.source"),
                    "duration": GraphNode(
                        "LTXVDurationPredictor",
                        {
                            "model": Link("source", "model"),
                            "positive": Link("source", "positive"),
                            "duration_head": Link("source", "head"),
                            "frame_rate": 10.0,
                            "min_seconds": 1.0,
                            "max_seconds": 20.0,
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["duration"])
                assert result.outputs["duration"]["num_frames"].resolve() == 25
                assert result.outputs["duration"]["seconds"].resolve() == pytest.approx(3.1)
        finally:
            await composer.close()

    asyncio.run(scenario())
