"""CPU contracts for HiDream-O1 shifted-grid prediction blending."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
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
from dinkster_nodes_comfy.hidream_patches import HiDreamO1PatchSeamSmoothing

from dinkster.compose import PackSpec, ServingComposer


class Sampling:
    multiplier = 3.0

    def percent_to_sigma(self, percent: float) -> float:
        return (1.0 - percent) * 10.0


class Model:
    def __init__(self) -> None:
        self.wrappers: dict[tuple[str, str], Any] = {}

    def clone(self) -> Model:
        result = Model()
        result.wrappers = self.wrappers.copy()
        return result

    def get_model_object(self, name: str) -> Sampling:
        assert name == "model_sampling"
        return Sampling()

    def add_wrapper_with_key(self, point: str, name: str, wrapper: Any) -> None:
        self.wrappers[point, name] = wrapper


def test_shifted_grid_average_preserves_edges_and_scales_strength() -> None:
    source = Model()
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=source,
        start_percent=0.2,
        end_percent=0.8,
        pattern="single_shift",
        passes="2",
        blend="average",
        strength=0.625,
    )["MODEL"]
    assert isinstance(patched, Model)
    assert patched is not source and source.wrappers == {}
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    bias = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 283

    def predict(value: torch.Tensor, sigma: torch.Tensor, *, offset: float) -> torch.Tensor:
        return value.square() + bias + offset

    sigma = torch.tensor([15.0])
    natural = predict(image, sigma, offset=0.75)
    shifted = image.square() + torch.roll(bias, (-16, -16), (-2, -1)) + 0.75
    expected_center = natural * 0.375 + ((natural + shifted) / 2) * 0.625
    result = wrapper(predict, image, sigma, offset=0.75)
    torch.testing.assert_close(
        result[..., 36:60, 36:92], expected_center[..., 36:60, 36:92], rtol=0, atol=0
    )
    assert torch.equal(result[..., :33, :], natural[..., :33, :])
    assert torch.equal(result[..., -33:, :], natural[..., -33:, :])
    assert torch.equal(result[..., :, :33], natural[..., :, :33])
    assert torch.equal(result[..., :, -33:], natural[..., :, -33:])


@pytest.mark.parametrize("pattern", ("single_shift", "symmetric"))
@pytest.mark.parametrize("blend", ("average", "window", "median"))
def test_ramped_smoothing_window_boundaries_and_pass_counts(pattern: str, blend: str) -> None:
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=Model(),
        start_percent=0.2,
        end_percent=0.8,
        pattern=pattern,
        passes="ramp_2_4_8",
        blend=blend,
        strength=1.0,
    )["MODEL"]
    assert isinstance(patched, Model)
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    calls: list[torch.Tensor] = []

    def predict(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return value.square()

    for timestep, level in (
        (24.1, 0),
        (24.0, 2),
        (18.1, 2),
        (18.0, 4),
        (12.1, 4),
        (12.0, 8),
        (6.0, 8),
        (5.9, 0),
    ):
        calls.clear()
        actual = wrapper(predict, image, torch.tensor([timestep]))
        expected_calls = max(level, 1) + (pattern == "symmetric" and level > 0)
        assert len(calls) == expected_calls
        torch.testing.assert_close(actual, image.square(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "strength,start,end", ((0.0, 0.2, 0.8), (-0.5, 0.2, 0.8), (0.5, 0.7, 0.7), (0.5, 0.8, 0.2))
)
def test_disabled_smoothing_preserves_model_identity(
    strength: float, start: float, end: float
) -> None:
    model = Model()
    assert (
        HiDreamO1PatchSeamSmoothing.execute(
            model=model,
            start_percent=start,
            end_percent=end,
            pattern="single_shift",
            passes="2",
            blend="average",
            strength=strength,
        )["MODEL"]
        is model
    )


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.source", outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=Model())


class ModelInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> dict[str, object]:
        wrapper = next(iter(model.wrappers.values()))
        x = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
        bias = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 283

        def predict(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return value.square() + bias

        sigma = torch.tensor([15.0])
        natural = predict(x, sigma)
        shifted = x.square() + torch.roll(bias, (-16, -16), (-2, -1))
        expected = natural * 0.375 + ((natural + shifted) / 2) * 0.625
        result = wrapper(predict, x, sigma)
        torch.testing.assert_close(
            result[..., 36:60, 36:92], expected[..., 36:60, 36:92], rtol=0, atol=0
        )
        assert torch.equal(result[..., :33, :], natural[..., :33, :])
        assert torch.equal(wrapper(predict, x, torch.tensor([24.1])), natural)
        return cls.outputs(value=float(result.sum()))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_hidream_smoothing_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"hidream_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_hidream_patches_torch import ModelSource, ModelInspect\n"
            "from dinkster_inference_wire import register_inference_types\n"
            "NODES = (ModelSource, ModelInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "hidream-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["hidream-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(__file__).parent),
                        os.environ.get("PYTHONPATH", ""),
                    )
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="hidream-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hidream-fixture.source"),
                    "patch": GraphNode(
                        "HiDreamO1PatchSeamSmoothing",
                        {
                            "model": Link("source", "model"),
                            "start_percent": 0.2,
                            "end_percent": 0.8,
                            "pattern": "single_shift",
                            "passes": "2",
                            "blend": "average",
                            "strength": 0.625,
                        },
                    ),
                    "inspect": GraphNode(
                        "hidream-fixture.inspect", {"model": Link("patch", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            results = [
                (await engine.run(graph, ["inspect"])).outputs["inspect"]["value"].resolve()
                for _ in range(2)
            ]
            assert results[0] == results[1] and results[0] > 0
        finally:
            await composer.close()

    asyncio.run(scenario())
