"""In-memory LoRA application with real patchers and ordinary native hosts."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    LORA_MODEL_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.lora_model_loader import LoraModelLoader

from dinkster.compose import PackSpec, ServingComposer


class _RawModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.linear = torch.nn.Linear(3, 2, bias=False)
        self.diffusion_model.linear.weight.data.copy_(
            torch.tensor([[1.0, -2.0, 4.0], [3.0, 5.0, -1.0]])
        )
        self.model_config = type("Config", (), {"unet_config": {}})()


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_RawModel(), torch.device("cpu"), torch.device("cpu"))


def _lora() -> dict[str, torch.Tensor]:
    return {
        "lora_unet_linear.lora_down.weight": torch.tensor([[1.0, 2.0, -1.0]]),
        "lora_unet_linear.lora_up.weight": torch.tensor([[2.0], [-3.0]]),
    }


def _assert_applied(model: Any, strength: float, bypass: bool) -> None:
    if bypass:
        model.inject_model()
        try:
            output = model.model.diffusion_model.linear(torch.tensor([[2.0, -1.0, 0.5]]))
            expected = [[5.5, 1.25]] if strength == 0.5 else [[7.0, -1.0]]
            torch.testing.assert_close(output, torch.tensor(expected), rtol=0, atol=0)
        finally:
            model.eject_model()
    else:
        from dinkster_inference.lora import calculate_weight

        weight = calculate_weight(
            model.patches["diffusion_model.linear.weight"],
            model.model.diffusion_model.linear.weight.detach().clone(),
            "diffusion_model.linear.weight",
        )
        expected = (
            [[2.0, 0.0, 3.0], [1.5, 2.0, 0.5]]
            if strength == 0.5
            else [[-1.0, -6.0, 6.0], [6.0, 11.0, -4.0]]
        )
        torch.testing.assert_close(weight, torch.tensor(expected), rtol=0, atol=0)


@pytest.mark.parametrize("bypass", (False, True))
@pytest.mark.parametrize("strength", (0.5, -1.0))
def test_lora_model_loader_applies_signed_in_memory_weights_without_mutating_sources(
    strength: float, bypass: bool
) -> None:
    model, lora = _model(), _lora()
    original_weights = model.model.diffusion_model.linear.weight.detach().clone()
    original_lora = {key: value.clone() for key, value in lora.items()}
    result = LoraModelLoader.execute(model=model, lora=lora, strength_model=strength, bypass=bypass)
    assert result["model"] is not model
    _assert_applied(result["model"], strength, bypass)
    assert not model.patches and not model.injections
    torch.testing.assert_close(
        model.model.diffusion_model.linear.weight, original_weights, rtol=0, atol=0
    )
    assert lora.keys() == original_lora.keys()
    for key, value in lora.items():
        torch.testing.assert_close(value, original_lora[key], rtol=0, atol=0)


def test_lora_model_loader_zero_strength_needs_no_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_nodes_comfy.lora_model_loader as module

    def unavailable() -> None:
        raise AssertionError("zero strength must not initialize inference")

    monkeypatch.setattr(module, "initialize_inference_process", unavailable)
    model = object()
    for bypass in (False, True):
        assert LoraModelLoader.execute(model=model, lora={}, strength_model=0, bypass=bypass) == {
            "model": model
        }


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("lora", TypeExpr.concrete(LORA_MODEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_model(), lora=_lora())


class ModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-fixture.consume",
            inputs=(
                InputSpec("normal", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("bypass", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("zero", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, normal: Any, bypass: Any, zero: Any) -> dict[str, object]:
        _assert_applied(normal, 0.5, False)
        _assert_applied(bypass, -1.0, True)
        assert not zero.patches and not zero.injections
        torch.testing.assert_close(
            zero.model.diffusion_model.linear.weight,
            torch.tensor([[1.0, -2.0, 4.0], [3.0, 5.0, -1.0]]),
            rtol=0,
            atol=0,
        )
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_lora_model_loader_ordinary_hosts_consume_resident_outputs_and_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        (tmp_path / "lora_model_fixture.py").write_text(
            "from test_lora_model_loader_torch import ModelSource, ModelConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ModelSource, ModelConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="lora-model-fixture"\npublisher="dinkster"\n'
            'namespaces=["lora-model-fixture"]\n[pack.entry]\n'
            'nodes="lora_model_fixture:NODES"\ntypes="lora_model_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (
            tmp_path,
            Path(__file__).parent,
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
            *(Path(path) for path in sys.path if path and Path(path).is_dir()),
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest, worker_group="lora-model", group_manifests=(native, fixture)
                    )
                )
            nodes = {"source": GraphNode("lora-model-fixture.source")}
            for name, strength, bypass in (
                ("normal", 0.5, False),
                ("bypass", -1.0, True),
                ("zero", 0, False),
            ):
                nodes[name] = GraphNode(
                    "LoraModelLoader",
                    {
                        "model": Link("source", "model"),
                        "lora": Link("source", "lora"),
                        "strength_model": strength,
                        "bypass": bypass,
                    },
                )
            nodes["consume"] = GraphNode(
                "lora-model-fixture.consume",
                {name: Link(name, "model") for name in ("normal", "bypass", "zero")},
            )
            graph = Graph(nodes=nodes)
            engine = composer.composition.make_engine(lambda event: None)
            for run in range(2):
                result = await engine.run(graph, ["consume"])
                assert result.outputs["consume"]["valid"].resolve() is True
                if run == 1:
                    assert set(result.cached) == set(nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
