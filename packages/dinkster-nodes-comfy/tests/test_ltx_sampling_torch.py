"""CPU behavior and ordinary-host contracts for native LTXV sampling."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    MODEL_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_sampling import LTXVScheduler, ModelSamplingLTXV

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(model_config=SimpleNamespace(sampling_settings={}))
        self.sampling = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.model = self.model
        result.sampling = self.sampling
        return result

    def add_object_patch(self, name: str, value: Any) -> None:
        assert name == "model_sampling"
        self.sampling = value


@pytest.mark.parametrize("latent_shape", (None, (2, 7, 4, 16, 16), (3, 5, 3, 8, 11)))
def test_model_sampling_clones_and_uses_spatial_temporal_token_count(
    latent_shape: tuple | None,
) -> None:
    source = Patcher()
    latent = None if latent_shape is None else {"samples": torch.full(latent_shape, -3.0)}
    result = ModelSamplingLTXV.execute(
        model=source, max_shift=3.17, base_shift=0.63, latent=latent
    )["MODEL"]
    tokens = 4096 if latent_shape is None else math.prod(latent_shape[2:])
    expected_shift = (tokens - 1024) * (3.17 - 0.63) / 3072 + 0.63
    assert result is not source
    assert source.sampling is None
    assert result.model is source.model
    assert result.sampling.shift == pytest.approx(expected_shift, abs=1e-15)
    t = torch.tensor([0.13, 0.61], dtype=torch.float64)
    expected = math.exp(expected_shift) / (math.exp(expected_shift) + 1 / t - 1)
    torch.testing.assert_close(result.sampling.sigma(t), expected, rtol=1e-14, atol=1e-14)
    image = torch.tensor([[[[2.0, -7.0]]], [[[5.0, 11.0]]]])
    prediction = torch.tensor([[[[-3.0, 4.0]]], [[[8.0, -2.0]]]])
    sigma = torch.tensor([0.25, 0.75])
    assert torch.equal(
        result.sampling.calculate_denoised(sigma, prediction, image),
        image - prediction * sigma[:, None, None, None],
    )
    if latent is not None:
        assert torch.equal(latent["samples"], torch.full(latent_shape, -3.0))


@pytest.mark.parametrize("stretch,terminal", ((False, 0.1), (True, 0.0), (True, 0.37)))
def test_scheduler_matches_independent_scalar_curve_and_preserves_latent(
    stretch: bool, terminal: float
) -> None:
    latent = {"samples": torch.full((2, 9, 3, 8, 11), 7.0, dtype=torch.float64)}
    result = LTXVScheduler.execute(
        steps=7, max_shift=2.91, base_shift=0.43, stretch=stretch, terminal=terminal, latent=latent
    )["SIGMAS"]
    shift = (264 - 1024) * (2.91 - 0.43) / 3072 + 0.43
    expected = [math.exp(shift) / (math.exp(shift) + 1 / (1 - i / 7) - 1) for i in range(7)]
    if stretch:
        expected = [1 - (1 - value) * (1 - terminal) / (1 - expected[-1]) for value in expected]
    torch.testing.assert_close(result, torch.tensor([*expected, 0.0]), rtol=2e-6, atol=2e-7)
    assert result.dtype is torch.get_default_dtype()
    assert result[-1] == 0
    assert torch.equal(latent["samples"], torch.full((2, 9, 3, 8, 11), 7.0, dtype=torch.float64))


def test_scheduler_default_tokens_and_single_step_source_boundary() -> None:
    result = LTXVScheduler.execute(
        steps=4, max_shift=1.7, base_shift=0.4, stretch=False, terminal=0.2
    )["SIGMAS"]
    expected = [math.exp(1.7) / (math.exp(1.7) + 1 / t - 1) for t in (1.0, 0.75, 0.5, 0.25)]
    torch.testing.assert_close(result, torch.tensor([*expected, 0.0]))
    boundary = LTXVScheduler.execute(
        steps=1, max_shift=0.0, base_shift=0.0, stretch=True, terminal=0.2
    )["SIGMAS"]
    assert torch.isnan(boundary[0]) and boundary[1] == 0


class SamplingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-sampling-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=Patcher(),
            latent={"samples": torch.full((2, 9, 3, 8, 11), 7.0, dtype=torch.float64)},
        )


class SamplingConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-sampling-fixture.consumer",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("sigmas", TypeExpr.concrete(SIGMAS_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, sigmas: object) -> dict[str, object]:
        values = materialize_latent(sigmas)
        shift = (264 - 1024) * (2.91 - 0.43) / 3072 + 0.43
        assert model.sampling.shift == pytest.approx(shift, abs=1e-15)
        expected = [math.exp(shift) / (math.exp(shift) + 1 / (1 - i / 7) - 1) for i in range(7)]
        expected = [1 - (1 - value) * 0.63 / (1 - expected[-1]) for value in expected]
        torch.testing.assert_close(values, torch.tensor([*expected, 0.0]), rtol=2e-6, atol=2e-7)
        assert values.dtype is torch.float32 and values.device.type == "cpu"
        return cls.outputs(summary="ltx-sampling")


@pytest.mark.parametrize("in_process", (True,))
def test_sampling_nodes_compose_twice_in_both_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            (tmp_path / "ltx_sampling_fixture.py").write_text(
                "from test_ltx_sampling_torch import SamplingSource, SamplingConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (SamplingSource, SamplingConsumer)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "ltx-sampling-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["ltx-sampling-fixture"]\n'
                '[pack.entry]\nnodes = "ltx_sampling_fixture:NODES"\n'
                'types = "ltx_sampling_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for path in (manifest, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=path,
                        in_process=in_process,
                        worker_group=None if in_process else "ltx-sampling-fixture",
                        group_manifests=() if in_process else (manifest, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-sampling-fixture.source"),
                    "patch": GraphNode(
                        "ModelSamplingLTXV",
                        {
                            "model": Link("source", "model"),
                            "latent": Link("source", "latent"),
                            "max_shift": 2.91,
                            "base_shift": 0.43,
                        },
                    ),
                    "schedule": GraphNode(
                        "LTXVScheduler",
                        {
                            "latent": Link("source", "latent"),
                            "steps": 7,
                            "max_shift": 2.91,
                            "base_shift": 0.43,
                            "stretch": True,
                            "terminal": 0.37,
                        },
                    ),
                    "consumer": GraphNode(
                        "ltx-sampling-fixture.consumer",
                        {"model": Link("patch", "MODEL"), "sigmas": Link("schedule", "SIGMAS")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "ltx-sampling"
        finally:
            await composer.close()

    asyncio.run(scenario())
