"""CPU noise-guidance callbacks and ordinary-manifest model execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.noise_guidance import EpsilonScaling, FreSca, TemporalScoreRescaling

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self) -> None:
        self.pre: Any = None
        self.post: Any = None

    def clone(self) -> Patcher:
        return Patcher()

    def set_model_sampler_pre_cfg_function(self, callback: Any) -> None:
        self.pre = callback

    def set_model_sampler_post_cfg_function(self, callback: Any) -> None:
        self.post = callback


def test_noise_scaling_and_score_rescaling_preserve_source_and_compute_expected_values() -> None:
    source = Patcher()
    x = torch.tensor([2.0, 4.0, -3.0])
    denoised = torch.tensor([1.0, -2.0, 5.0])
    epsilon = EpsilonScaling.execute(model=source, scaling_factor=0.5)["MODEL"]
    assert torch.equal(
        epsilon.post({"input": x, "denoised": denoised}), torch.tensor([0.0, -8.0, 13.0])
    )
    rescale = TemporalScoreRescaling.execute(model=source, tsr_k=0.5, tsr_sigma=2.0)["MODEL"]
    args = {
        "input": x,
        "denoised": denoised,
        "sigma": torch.tensor(2.0),
        "model": SimpleNamespace(
            current_patcher=SimpleNamespace(get_model_object=lambda _: object())
        ),
    }
    # Ordinary epsilon sampling has alpha=1, SNR=1/4 and variance=4, hence ratio=2/3.
    torch.testing.assert_close(rescale.post(args), torch.tensor([4 / 3, 0.0, 7 / 3]))
    bypass = TemporalScoreRescaling.execute(model=source, tsr_k=1.0, tsr_sigma=2.0)["MODEL"]
    assert bypass.post({"sigma": torch.tensor(2.0), "denoised": denoised}) is denoised
    assert rescale.post({"sigma": torch.tensor(0.0), "denoised": denoised}) is denoised
    assert source.pre is None and source.post is None


@pytest.mark.parametrize("flow", (False, True))
def test_score_rescaling_uses_each_samples_sigma_and_preserves_zero_samples(flow: bool) -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference.model_sampling import CONST

    source = Patcher()
    patched = TemporalScoreRescaling.execute(model=source, tsr_k=0.5, tsr_sigma=2.0)["MODEL"]
    sampling = CONST() if flow else object()
    model = SimpleNamespace(current_patcher=SimpleNamespace(get_model_object=lambda _: sampling))
    x = torch.arange(60, dtype=torch.float64).reshape(2, 3, 2, 5) / 7 - 2
    denoised = torch.flip(x, dims=(-1,)) / 3 + 1
    x_before, denoised_before = x.clone(), denoised.clone()
    cases = ((0.25, 0.5), (0.0, 0.5), (0.25, 0.0), (0.0, 0.0), (1.0, 0.5))
    for sigmas in cases:
        sigma = torch.tensor(sigmas, dtype=torch.float64)
        expected = denoised.clone()
        for index, value in enumerate(sigmas):
            if value == 0 or (flow and value == 1):
                continue
            alpha = 1 - value if flow else 1.0
            snr = (alpha / value) ** 2
            ratio = (snr * 4 + 1) / (snr * 8 + 1)
            expected[index] = x[index] / alpha * (1 - ratio) + denoised[index] * ratio
        args = {"input": x, "denoised": denoised, "sigma": sigma, "model": model}
        result = patched.post(args)
        torch.testing.assert_close(result, expected)
        for index, value in enumerate(sigmas):
            if value == 0 or (flow and value == 1):
                assert torch.equal(result[index], denoised[index])
        assert torch.equal(sigma, torch.tensor(sigmas, dtype=torch.float64))
        bypass = TemporalScoreRescaling.execute(model=source, tsr_k=1.0, tsr_sigma=2.0)["MODEL"]
        assert bypass.post({"sigma": sigma, "denoised": denoised}) is denoised
    assert torch.equal(x, x_before)
    assert torch.equal(denoised, denoised_before)
    assert source.post is None


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_frequency_guidance_separates_dc_and_high_frequency_and_preserves_tail(
    dtype: torch.dtype,
) -> None:
    source = Patcher()
    patched = FreSca.execute(model=source, scale_low=2.0, scale_high=3.0, freq_cutoff=1)["MODEL"]
    checker = torch.tensor([[1, -1, 1, -1], [-1, 1, -1, 1]] * 2, dtype=dtype)
    cond = (4 + checker).reshape(1, 1, 4, 4)
    uncond = torch.ones_like(cond)
    tail = torch.zeros_like(cond)
    before = cond.clone()
    args = {"conds_out": [cond, uncond, tail], "conds": [object(), object(), object()]}
    result = patched.pre(args)
    assert torch.equal(result[0], (7 + 3 * checker).reshape_as(cond))
    assert result[0].dtype == dtype
    assert result[1] is uncond and result[2] is tail
    assert torch.equal(cond, before)
    for conds, outputs in (([None, object()], [cond, uncond]), ([object()], [cond])):
        assert patched.pre({"conds_out": outputs, "conds": conds}) is outputs
    assert source.pre is None


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="noise-guidance-fixture.model",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class Result(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="noise-guidance-fixture.result",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(MODEL_TYPE))
                for name in ("epsilon", "score", "frequency", "source")
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls, *, epsilon: Patcher, score: Patcher, frequency: Patcher, source: Patcher
    ) -> Mapping[str, object]:
        denoised = torch.tensor([3.0, -2.0])
        assert torch.equal(
            epsilon.post({"input": torch.tensor([1.0, 4.0]), "denoised": denoised}),
            torch.tensor([5.0, -8.0]),
        )
        assert score.post({"sigma": torch.tensor(0.0), "denoised": denoised}) is denoised
        x = torch.tensor([[[[-2.0, 5.0, 8.0]]], [[[3.0, -1.0, 7.0]]]])
        predictions = torch.tensor([[[[2.0, -4.0, 1.0]]], [[[-2.0, 5.0, 3.0]]]])
        result = score.post(
            {
                "input": x,
                "denoised": predictions,
                "sigma": torch.tensor([0.0, 0.5]),
                "model": SimpleNamespace(
                    current_patcher=SimpleNamespace(get_model_object=lambda _: object())
                ),
            }
        )
        assert torch.equal(result[0], predictions[0])
        torch.testing.assert_close(result[1], x[1] * (16 / 33) + predictions[1] * (17 / 33))
        inputs = [torch.full((1, 1, 4, 4), 4.0), torch.ones(1, 1, 4, 4)]
        outputs = frequency.pre({"conds_out": inputs, "conds": [object(), object()]})
        assert torch.equal(outputs[0], torch.full((1, 1, 4, 4), 7.0))
        assert source.pre is None and source.post is None
        return cls.outputs(value=7.0)


FIXTURE_NODES = (ModelSource, Result)


@pytest.mark.parametrize("in_process", (True,))
def test_noise_guidance_runs_twice_in_both_ordinary_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            (tmp_path / "noise_guidance_fixture.py").write_text(
                "from test_noise_guidance_torch import FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["noise-guidance-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "noise_guidance_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "noise_guidance_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("noise-guidance-fixture.model", {}),
                    "epsilon": GraphNode(
                        "Epsilon Scaling", {"model": Link("source", "MODEL"), "scaling_factor": 0.5}
                    ),
                    "score": GraphNode(
                        "TemporalScoreRescaling",
                        {"model": Link("source", "MODEL"), "tsr_k": 0.5, "tsr_sigma": 2.0},
                    ),
                    "frequency": GraphNode(
                        "FreSca",
                        {
                            "model": Link("source", "MODEL"),
                            "scale_low": 2.0,
                            "scale_high": 3.0,
                            "freq_cutoff": 1,
                        },
                    ),
                    "result": GraphNode(
                        "noise-guidance-fixture.result",
                        {
                            name: Link(name, "MODEL")
                            for name in ("epsilon", "score", "frequency", "source")
                        },
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["result"])
                assert result.outputs["result"]["value"].resolve() == 7.0
        finally:
            await composer.close()

    asyncio.run(scenario())
