"""CPU execution and ordinary-manifest transport for model-specific schedules."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
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
    AlignYourStepsScheduler,
    GITSScheduler,
    Ideogram4Scheduler,
    OptimalStepsScheduler,
)

from dinkster.compose import PackSpec, ServingComposer


def _sigmas(result: Mapping[str, object]) -> torch.Tensor:
    return materialize_latent(result["SIGMAS"])


def test_model_specific_schedulers_preserve_table_interpolation_denoise_and_aspect_behavior() -> (
    None
):
    ays_table = _sigmas(AlignYourStepsScheduler.execute(model_type="SD1", steps=10, denoise=0.5))
    torch.testing.assert_close(
        ays_table,
        torch.tensor([1.3943805, 0.9642584, 0.6523686, 0.3977456, 0.1515233, 0.0]),
    )
    gits_table = _sigmas(GITSScheduler.execute(coeff=0.8, steps=2, denoise=1.0))
    torch.testing.assert_close(gits_table, torch.tensor([14.614641, 7.4900165, 0.0]))
    gits_interpolated = _sigmas(GITSScheduler.execute(coeff=1.2, steps=21, denoise=1.0))
    assert len(gits_interpolated) == 22
    assert gits_interpolated[-1] == 0
    assert not torch.equal(gits_interpolated[:-1], gits_interpolated.new_zeros(21))
    optimal = _sigmas(OptimalStepsScheduler.execute(model_type="FLUX", steps=20, denoise=0.0))
    assert optimal.numel() == 0
    square = _sigmas(Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=1.75))
    wide = _sigmas(Ideogram4Scheduler.execute(steps=4, width=1024, height=512, mu=0.0, std=1.75))
    shifted = _sigmas(Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=1.0, std=1.75))
    spread = _sigmas(Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=0.5))
    assert square[-1] == wide[-1] == shifted[-1] == spread[-1] == 0
    assert not torch.equal(square, wide)
    assert not torch.equal(square, shifted)
    assert not torch.equal(square, spread)


class SigmaConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        sigmas = TypeExpr.concrete(SIGMAS_TYPE)
        return NodeSchema(
            node_type="scheduler-fixture.consumer",
            inputs=tuple(
                InputSpec(name, sigmas) for name in ("ays", "gits", "optimal", "ideogram")
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> Mapping[str, object]:
        tensors = [
            materialize_latent(kwargs[name]) for name in ("ays", "gits", "optimal", "ideogram")
        ]
        assert all(tensor.dtype == torch.float32 for tensor in tensors)
        assert len({tuple(tensor.tolist()) for tensor in tensors}) == 4
        return cls.outputs(value=float(sum(tensor.sum().item() for tensor in tensors)))


@pytest.mark.parametrize("in_process", (True, False))
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
            engine = composer.composition.make_engine(lambda _event: None)
            first = (await engine.run(graph, ["consumer"])).outputs["consumer"]["value"].resolve()
            second = (await engine.run(graph, ["consumer"])).outputs["consumer"]["value"].resolve()
            assert first == second
        finally:
            await composer.close()

    asyncio.run(scenario())
