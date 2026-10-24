"""CPU callback behavior and ordinary-manifest model transport."""

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
from dinkster_nodes_comfy.rope_and_mask import DifferentialDiffusion

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self, rope: tuple[float, ...] = (), mask: Any = None) -> None:
        self.rope = rope
        self.mask = mask

    def clone(self) -> Patcher:
        return Patcher(self.rope, self.mask)

    def set_model_rope_options(self, *options: float) -> None:
        self.rope = options

    def set_model_denoise_mask_function(self, callback: Any) -> None:
        self.mask = callback


class Sampling:
    sigma_min = 1.0

    def timestep(self, sigma: torch.Tensor | float) -> torch.Tensor | float:
        return sigma**2


def options(last: float) -> dict[str, Any]:
    return {
        "model": SimpleNamespace(inner_model=SimpleNamespace(model_sampling=Sampling())),
        "sigmas": torch.tensor([5.0, last]),
    }


@pytest.mark.parametrize("strength", (0.0, 0.4, 1.0))
def test_differential_threshold_uses_sampling_timesteps_and_clamped_end(strength: float) -> None:
    source = Patcher()
    patched = DifferentialDiffusion.execute(model=source, strength=strength)["MODEL"]
    mask = torch.tensor([[0.0, 0.2], [0.25, 0.75]], dtype=torch.float64)
    original = mask.clone()
    for last, binary in (
        (0.5, torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)),
        (2.0, torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)),
    ):
        expected = 0.4 * binary + 0.6 * mask if strength == 0.4 else binary
        actual = patched.mask(torch.tensor([3.0]), mask, options(last))
        assert torch.equal(actual, expected)
        assert actual.dtype == mask.dtype
    assert torch.equal(mask, original)
    assert patched is not source
    assert source.mask is None


def test_differential_includes_threshold_and_defaults_to_binary_mask() -> None:
    patched = DifferentialDiffusion.execute(model=Patcher())["MODEL"]
    mask = torch.tensor([[0.0, 0.499], [0.5, 1.0]])
    linear = SimpleNamespace(sigma_min=1.0, timestep=lambda sigma: sigma)
    extra = {
        "model": SimpleNamespace(inner_model=SimpleNamespace(model_sampling=linear)),
        "sigmas": torch.tensor([5.0, 0.0]),
    }
    assert torch.equal(
        patched.mask(torch.tensor([3.0]), mask, extra), torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    )


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rope-fixture.model",
            inputs=(),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class Result(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rope-fixture.result",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("source", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher, source: Patcher) -> Mapping[str, object]:
        assert model.rope == (2.0, -3.0, 4.0, -5.0, 6.0, -7.0)
        assert source.rope == () and source.mask is None
        result = model.mask(
            torch.tensor([3.0]), torch.tensor([[0.0, 0.125], [0.25, 0.75]]), options(2.0)
        )
        assert torch.equal(result, torch.tensor([[0.0, 0.0625], [0.625, 0.875]]))
        return cls.outputs(value=1.0)


FIXTURE_NODES = (ModelSource, Result)


@pytest.mark.parametrize("in_process", (True, False))
def test_rope_and_mask_models_execute_twice_in_both_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            (tmp_path / "rope_fixture.py").write_text(
                "from test_rope_and_mask_torch import FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["rope-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "rope_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "rope_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("rope-fixture.model", {}),
                    "rope": GraphNode(
                        "ScaleROPE",
                        {
                            "model": Link("source", "MODEL"),
                            "scale_x": 2.0,
                            "shift_x": -3.0,
                            "scale_y": 4.0,
                            "shift_y": -5.0,
                            "scale_t": 6.0,
                            "shift_t": -7.0,
                        },
                    ),
                    "mask": GraphNode(
                        "DifferentialDiffusion", {"model": Link("rope", "MODEL"), "strength": 0.5}
                    ),
                    "result": GraphNode(
                        "rope-fixture.result",
                        {"model": Link("mask", "MODEL"), "source": Link("source", "MODEL")},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["result"])
                assert result.outputs["result"]["value"].resolve() == 1.0
        finally:
            await composer.close()

    asyncio.run(scenario())
