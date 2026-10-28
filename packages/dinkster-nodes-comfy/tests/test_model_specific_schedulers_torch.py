"""CPU execution and ordinary-manifest transport for model-specific schedules."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_specific_schedulers import (
    AYS_NOISE_LEVELS,
    GITS_NOISE_LEVELS,
    OPTIMAL_STEPS_NOISE_LEVELS,
    AlignYourStepsScheduler,
    GITSScheduler,
    Ideogram4Scheduler,
    OptimalStepsScheduler,
)

from dinkster.compose import PackSpec, ServingComposer


def _sigmas(result: Mapping[str, object]) -> torch.Tensor:
    return materialize_latent(result["SIGMAS"])


def _assert_sigmas(result: Mapping[str, object], expected: list[float]) -> None:
    torch.testing.assert_close(
        _sigmas(result), torch.tensor(expected, dtype=torch.float32), rtol=0.0, atol=1e-7
    )


def test_model_specific_schedulers_match_table_interpolation_and_denoise_boundaries() -> None:
    ays_table = _sigmas(AlignYourStepsScheduler.execute(model_type="SD1", steps=10, denoise=0.5))
    torch.testing.assert_close(
        ays_table,
        torch.tensor([1.3943805, 0.9642584, 0.6523686, 0.3977456, 0.1515233, 0.0]),
    )
    _assert_sigmas(
        GITSScheduler.execute(coeff=0.8, steps=3, denoise=1.0),
        [14.61464119, 11.54541874, 6.77309084, 0.0],
    )
    _assert_sigmas(
        GITSScheduler.execute(coeff=1.2, steps=21, denoise=0.5),
        [
            0.45593986,
            0.38061461,
            0.32555375,
            0.28130382,
            0.24309608,
            0.21883933,
            0.19313487,
            0.16521354,
            0.13353667,
            0.09272503,
            0.0,
        ],
    )
    _assert_sigmas(
        OptimalStepsScheduler.execute(model_type="FLUX", steps=5, denoise=1.0),
        [0.9968, 0.9819, 0.966, 0.9158, 0.5512, 0.0],
    )
    assert (
        _sigmas(OptimalStepsScheduler.execute(model_type="FLUX", steps=20, denoise=0.0)).numel()
        == 0
    )


def test_ideogram4_schedule_has_exact_aspect_shift_and_spread_values() -> None:
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=1.75),
        [0.99987662, 0.76501203, 0.5, 0.234988, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=1024, height=512, mu=0.0, std=1.75),
        [0.99987662, 0.82155681, 0.58578646, 0.30284566, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=1.0, std=1.75),
        [0.99987662, 0.89847165, 0.7310586, 0.45503256, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=0.5),
        [0.99987662, 0.58352113, 0.5, 0.41647887, 0.0],
    )


def test_scheduler_execution_never_mutates_source_tables() -> None:
    before = copy.deepcopy((AYS_NOISE_LEVELS, GITS_NOISE_LEVELS, OPTIMAL_STEPS_NOISE_LEVELS))
    AlignYourStepsScheduler.execute(model_type="SVD", steps=4, denoise=1.0)
    GITSScheduler.execute(coeff=0.8, steps=3, denoise=1.0)
    GITSScheduler.execute(coeff=1.2, steps=21, denoise=0.5)
    OptimalStepsScheduler.execute(model_type="FLUX", steps=5, denoise=1.0)
    assert (AYS_NOISE_LEVELS, GITS_NOISE_LEVELS, OPTIMAL_STEPS_NOISE_LEVELS) == before


class SigmaConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        sigmas = TypeExpr.concrete(SIGMAS_TYPE)
        return NodeSchema(
            node_type="scheduler-fixture.consumer",
            inputs=tuple(
                InputSpec(name, sigmas) for name in ("ays", "gits", "optimal", "ideogram")
            ),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(CORE_STRING))
                for name in ("ays", "gits", "optimal", "ideogram")
            ),
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> Mapping[str, object]:
        tensors = [
            materialize_latent(kwargs[name]) for name in ("ays", "gits", "optimal", "ideogram")
        ]
        assert all(tensor.dtype == torch.float32 for tensor in tensors)
        return cls.outputs(
            **{
                name: json.dumps(tensor.tolist())
                for name, tensor in zip(
                    ("ays", "gits", "optimal", "ideogram"), tensors, strict=True
                )
            }
        )


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_executes_four_scheduler_graph_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = f"model_specific_scheduler_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from test_model_specific_schedulers_torch import SigmaConsumer\n"
                "NODES = (*COMFY_NODES, SigmaConsumer)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["scheduler-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "ays": GraphNode(
                        "AlignYourStepsScheduler", {"model_type": "SVD", "steps": 4, "denoise": 1.0}
                    ),
                    "gits": GraphNode("GITSScheduler", {"coeff": 0.8, "steps": 3, "denoise": 1.0}),
                    "optimal": GraphNode(
                        "OptimalStepsScheduler", {"model_type": "FLUX", "steps": 5, "denoise": 1.0}
                    ),
                    "ideogram": GraphNode(
                        "Ideogram4Scheduler",
                        {"steps": 4, "width": 1024, "height": 512, "mu": 0.5, "std": 0.5},
                    ),
                    "consumer": GraphNode(
                        "scheduler-fixture.consumer",
                        {
                            name: Link(name, "SIGMAS")
                            for name in ("ays", "gits", "optimal", "ideogram")
                        },
                    ),
                }
            )
            expected_values = {
                "ays": [700.0, 11.25711441040039, 1.7890000343322754, 0.26404356956481934, 0.0],
                "gits": [14.614641189575195, 11.545418739318848, 6.773090839385986, 0.0],
                "optimal": [
                    0.9968000054359436,
                    0.9818999767303467,
                    0.9660000205039978,
                    0.9157999753952026,
                    0.5511999726295471,
                    0.0,
                ],
                "ideogram": [
                    0.9998766183853149,
                    0.7656337022781372,
                    0.6998478770256042,
                    0.6246491074562073,
                    0.0,
                ],
            }
            expected = {name: json.dumps(values) for name, values in expected_values.items()}
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert {
                    name: result.outputs["consumer"][name].resolve()
                    for name in ("ays", "gits", "optimal", "ideogram")
                } == expected
                if attempt:
                    assert set(result.cached) == {"ays", "gits", "optimal", "ideogram", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
