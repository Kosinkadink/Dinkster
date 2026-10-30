"""CPU contracts for native SenseNova sampling options."""

from __future__ import annotations

import asyncio
import math
import os
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
from dinkster_nodes_comfy.sensenova_options import SenseNovaSamplingOptions

from dinkster.compose import PackSpec, ServingComposer

_MODEL = TypeExpr.concrete(MODEL_TYPE)


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(
                sampling_settings={"shift": 0.75, "noise_scale": 1.75, "multiplier": 17}
            )
        )
        self.sampling: Any = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.model = self.model
        result.sampling = self.sampling
        return result

    def add_object_patch(self, name: str, value: Any) -> None:
        assert name == "model_sampling"
        self.sampling = value


@pytest.mark.parametrize("shift", (1.0, 0.25, 3.5))
def test_sensenova_options_preserves_source_and_configured_noise_scale(shift: float) -> None:
    original = Patcher()
    patched = SenseNovaSamplingOptions.execute(model=original, shift=shift)["MODEL"]
    assert patched is not original
    assert original.sampling is None
    assert original.model.model_config.sampling_settings == {
        "shift": 0.75,
        "noise_scale": 1.75,
        "multiplier": 17,
    }
    sampling = patched.sampling
    base = torch.tensor([0.0, 0.17, 0.63, 1.0])
    expected = shift * base / (1.0 + (shift - 1.0) * base)
    torch.testing.assert_close(sampling.sigma((1.0 - base) * 1000), expected)
    assert len(sampling.sigmas) == 1001
    assert sampling.noise_scale == 1.75
    assert sampling.percent_to_sigma(0.0) == 1.0
    assert sampling.percent_to_sigma(1.0) == 0.0
    assert sampling.percent_to_sigma(0.4) == pytest.approx(shift * 0.6 / (1 + (shift - 1) * 0.6))
    noise = torch.full((1, 2, 33, 65), 2.0)
    latent = torch.full_like(noise, -3.0)
    scale = math.sqrt(2 * 3 / 64) * 1.75
    torch.testing.assert_close(
        sampling.noise_scaling(torch.tensor([0.4]), noise, latent),
        0.4 * scale * noise + 0.6 * latent,
    )


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sensenova-fixture.source", outputs=(OutputSpec("model", _MODEL),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=Patcher())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sensenova-fixture.inspect",
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("sigma", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> dict[str, object]:
        assert model.sampling.noise_scale == 1.75
        return cls.outputs(sigma=model.sampling.percent_to_sigma(0.4))


@pytest.mark.parametrize("in_process", (True,))
def test_sensenova_options_runs_twice_through_both_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
            }
        )
        try:
            native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "sensenova_fixture.py").write_text(
                "from test_sensenova_options_torch import Source, Inspect\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (Source, Inspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "sensenova-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["sensenova-fixture"]\n\n[pack.entry]\n'
                'nodes = "sensenova_fixture:NODES"\n'
                'types = "sensenova_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "sensenova-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sensenova-fixture.source"),
                    "patch": GraphNode(
                        "SenseNovaSamplingOptions",
                        {"model": Link("source", "model"), "shift": 3.5},
                    ),
                    "inspect": GraphNode(
                        "sensenova-fixture.inspect", {"model": Link("patch", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["sigma"].resolve() == pytest.approx(0.84)
        finally:
            await composer.close()

    asyncio.run(scenario())
