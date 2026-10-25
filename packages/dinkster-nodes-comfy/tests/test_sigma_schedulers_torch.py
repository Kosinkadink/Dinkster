"""CPU behavior for native sigma scheduler nodes against pinned inference."""

from __future__ import annotations

import asyncio
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MODEL_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sigma_schedulers import (
    BasicScheduler,
    BetaSamplingScheduler,
    ExponentialScheduler,
    KarrasScheduler,
    LaplaceScheduler,
    PolyexponentialScheduler,
    SDTurboScheduler,
    VPScheduler,
)

from dinkster.compose import PackSpec, ServingComposer

SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


def _summary(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return f"{tensor.dtype};{tuple(tensor.shape)};{hashlib.sha256(raw).hexdigest()}"


class SigmaSummary(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="scheduler-sigmas.summary",
            inputs=(InputSpec("sigmas", SIGMAS),),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, sigmas: Any) -> dict[str, object]:
        return cls.outputs(summary=_summary(materialize_latent(sigmas)))


class ModelSampling:
    sigmas = torch.tensor((0.0, 0.125, 0.5, 1.0, 2.0, 4.0, 8.0), dtype=torch.float32)

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.to(torch.float32) / 1000


class Model:
    def get_model_object(self, name: str) -> ModelSampling:
        assert name == "model_sampling"
        return ModelSampling()


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="scheduler-sigmas.model",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(MODEL=Model())


def _enable_cpu_runtime() -> None:
    initialize_inference_process()


def test_parameterized_schedulers_match_pinned_runtime_outputs() -> None:
    _enable_cpu_runtime()
    cases: tuple[tuple[type[Any], dict[str, float | int], tuple[float, ...]], ...] = (
        (
            KarrasScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 3.2},
            (3.7000005, 1.6964283, 0.60366714, 0.13000005, 0.0),
        ),
        (
            ExponentialScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13},
            (3.6999998, 1.2118502, 0.3969138, 0.13000001, 0.0),
        ),
        (
            PolyexponentialScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 1.7},
            (3.7000003, 0.69804287, 0.21807629, 0.13000001, 0.0),
        ),
        (
            LaplaceScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "mu": 0.25, "beta": 0.8},
            (3.7, 1.7759936, 0.92833745, 0.13),
        ),
        (
            VPScheduler,
            {"steps": 4, "beta_d": 13.2, "beta_min": 0.37, "eps_s": 0.017},
            (32.607094, 4.9330225, 1.2199332, 0.09072536, 0.0),
        ),
    )
    for scheduler, inputs, expected in cases:
        sigmas = scheduler.execute(**inputs)["SIGMAS"]
        assert sigmas.dtype == torch.float32
        torch.testing.assert_close(sigmas, torch.tensor(expected, dtype=torch.float32))


def test_model_schedulers_match_denoise_boundaries_and_turbo_timesteps() -> None:
    _enable_cpu_runtime()
    model = Model()
    empty = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=0.0)["SIGMAS"]
    assert empty.dtype == torch.float32
    assert empty.numel() == 0
    full = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=1.0)["SIGMAS"]
    assert full.dtype == torch.float32
    torch.testing.assert_close(full, torch.tensor((8.0, 2.0, 0.5, 0.0)))
    fractional = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=0.6)[
        "SIGMAS"
    ]
    assert fractional.dtype == torch.float32
    torch.testing.assert_close(fractional, torch.tensor((2.0, 0.5, 0.125, 0.0)))
    beta = BetaSamplingScheduler.execute(model=model, steps=4, alpha=0.7, beta=1.3)["SIGMAS"]
    assert beta.dtype == torch.float32
    torch.testing.assert_close(beta, torch.tensor((8.0, 1.0, 0.5, 0.125, 0.0)))
    turbo = SDTurboScheduler.execute(model=model, steps=4, denoise=0.6)["SIGMAS"]
    torch.testing.assert_close(turbo, torch.tensor((0.599, 0.499, 0.399, 0.299, 0.0)))
    assert turbo.dtype == torch.float32


@pytest.mark.parametrize("in_process", (True, False))
def test_all_scheduler_sigmas_cross_native_and_consumer_pack_hosts(
    in_process: bool, tmp_path: Path
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
            module = f"scheduler_sigma_fixture_{int(in_process)}"
            (tmp_path / f"{module}.py").write_text(
                "from test_sigma_schedulers_torch import ModelSource, SigmaSummary\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ModelSource, SigmaSummary)\n",
                encoding="ascii",
            )
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                native_manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["scheduler-sigmas",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            cases = {
                "basic": (
                    "BasicScheduler",
                    {"scheduler": "simple", "steps": 3, "denoise": 0.6},
                    (2.0, 0.5, 0.125, 0.0),
                ),
                "karras": (
                    "KarrasScheduler",
                    {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 3.2},
                    (3.7000005, 1.6964283, 0.60366714, 0.13000005, 0.0),
                ),
                "exponential": (
                    "ExponentialScheduler",
                    {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13},
                    # MKL exp dispatch differs by one ULP across CPUs. Preserve
                    # exact transport against the reference equation on this host.
                    (*torch.linspace(math.log(3.7), math.log(0.13), 4).exp().tolist(), 0.0),
                ),
                "polyexponential": (
                    "PolyexponentialScheduler",
                    {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 1.7},
                    (3.7000003, 0.69804287, 0.21807629, 0.13000001, 0.0),
                ),
                "laplace": (
                    "LaplaceScheduler",
                    {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "mu": 0.25, "beta": 0.8},
                    (3.7, 1.7759936, 0.92833745, 0.13),
                ),
                "vp": (
                    "VPScheduler",
                    {"steps": 4, "beta_d": 13.2, "beta_min": 0.37, "eps_s": 0.017},
                    (32.607094, 4.9330225, 1.2199332, 0.09072536, 0.0),
                ),
                "beta": (
                    "BetaSamplingScheduler",
                    {"steps": 4, "alpha": 0.7, "beta": 1.3},
                    (8.0, 1.0, 0.5, 0.125, 0.0),
                ),
                "turbo": (
                    "SDTurboScheduler",
                    {"steps": 4, "denoise": 0.6},
                    (0.599, 0.499, 0.399, 0.299, 0.0),
                ),
            }
            nodes: dict[str, GraphNode] = {"model": GraphNode("scheduler-sigmas.model")}
            expected: dict[str, str] = {}
            for name, (node_type, inputs, values) in cases.items():
                scheduler_inputs = dict(inputs)
                if node_type in {"BasicScheduler", "BetaSamplingScheduler", "SDTurboScheduler"}:
                    scheduler_inputs["model"] = Link("model", "MODEL")
                nodes[name] = GraphNode(node_type, scheduler_inputs)
                nodes[f"{name}-consumer"] = GraphNode(
                    "scheduler-sigmas.summary", {"sigmas": Link(name, "SIGMAS")}
                )
                expected[f"{name}-consumer"] = _summary(torch.tensor(values, dtype=torch.float32))
            graph = Graph(nodes=nodes)
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, list(expected))
                for node, summary in expected.items():
                    assert result.outputs[node]["summary"].resolve() == summary
        finally:
            await composer.close()

    asyncio.run(scenario())
