"""CPU contracts for native Torch compilation and wanBlockSwap."""

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
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.compile import TorchCompileModel, wanBlockSwap


def _model_patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    class DiffusionModel(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * 3.0 + 2.0

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diffusion_model = DiffusionModel()

    return ModelPatcher(Model(), torch.device("cpu"), torch.device("cpu"))


def test_torch_compile_model_clones_compiles_and_restores_the_original_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _model_patcher()
    source.model_options["preserved"] = {"value": 17}
    source_options = source.model_options.copy()
    compiled = torch.nn.Identity()
    calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []

    def compile_seam(model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        calls.append((model, kwargs))
        return compiled

    monkeypatch.setattr(torch, "compile", compile_seam)
    patched = TorchCompileModel.execute(model=source, backend="cudagraphs")["MODEL"]

    assert patched is not source
    assert source.model_options == source_options
    assert calls == [
        (
            patched.model.diffusion_model,
            {
                "backend": "cudagraphs",
                "options": {"guard_filter_fn": calls[0][1]["options"]["guard_filter_fn"]},
                "mode": None,
                "fullgraph": False,
                "dynamic": None,
            },
        )
    ]
    kwargs = patched.model_options["torch_compile_kwargs"]
    assert kwargs["backend"] == "cudagraphs"
    assert kwargs["mode"] is None and kwargs["fullgraph"] is False and kwargs["dynamic"] is None
    guard_filter = kwargs["options"]["guard_filter_fn"]
    assert guard_filter(
        (SimpleNamespace(name="transformer_options.x"), SimpleNamespace(name="other"))
    ) == [False, True]

    wrapper = patched.wrappers["apply_model"]["torch.compile"][-1]
    original = patched.model.diffusion_model

    value = torch.tensor([[-3.0, 5.0]])
    from dinkster_inference.patcher_extension import WrapperExecutor

    execute = WrapperExecutor.new_class_executor(
        lambda current: patched.model.diffusion_model(current), patched.model, [wrapper]
    )
    assert torch.equal(execute.execute(value), value)
    assert torch.equal(execute.execute(value), value)
    assert patched.model.diffusion_model is original

    def fail(_value: torch.Tensor) -> torch.Tensor:
        assert patched.model.diffusion_model is compiled
        raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        WrapperExecutor.new_class_executor(fail, patched.model, [wrapper]).execute(value)
    assert patched.model.diffusion_model is original


def test_torch_compile_model_replaces_only_its_own_wrapper_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _model_patcher()
    source.add_wrapper_with_key("apply_model", "other", object())
    source.add_wrapper_with_key("apply_model", "torch.compile", object())
    monkeypatch.setattr(torch, "compile", lambda model, **_kwargs: model)

    patched = TorchCompileModel.execute(model=source, backend="inductor")["MODEL"]

    assert patched.wrappers["apply_model"]["other"] == source.wrappers["apply_model"]["other"]
    assert len(patched.wrappers["apply_model"]["torch.compile"]) == 1


def test_wan_block_swap_is_a_deprecated_identity_node() -> None:
    model = object()
    schema = wanBlockSwap.schema()

    assert schema.node_type == "wanBlockSwap"
    assert schema.deprecation is not None
    assert schema.search_visibility == "deprecated"
    assert wanBlockSwap.execute(model=model)["MODEL"] is model


class CompileModelSource(Node):
    original_compile: Any = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="compile-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        cls.original_compile = torch.compile

        def compile_seam(_model: torch.nn.Module, **_kwargs: object) -> torch.nn.Module:
            try:
                return torch.nn.Identity()
            finally:
                torch.compile = cls.original_compile
                cls.original_compile = None

        torch.compile = compile_seam
        return cls.outputs(MODEL=_model_patcher())


class CompileModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="compile-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        from dinkster_inference.patcher_extension import WrapperExecutor

        try:
            value = torch.tensor([[-3.0, 5.0]])
            wrapper = model.wrappers["apply_model"]["torch.compile"][-1]
            original = model.model.diffusion_model
            executor = WrapperExecutor.new_class_executor(
                lambda current: model.model.diffusion_model(current), model.model, [wrapper]
            )
            result = executor.execute(value)
            assert torch.equal(result, value)
            assert model.model.diffusion_model is original
            assert model.model_options["torch_compile_kwargs"]["backend"] == "inductor"
            return cls.outputs(value=float(result.sum()))
        finally:
            if CompileModelSource.original_compile is not None:
                torch.compile = CompileModelSource.original_compile
                CompileModelSource.original_compile = None


COMPILE_FIXTURE_NODES = (CompileModelSource, CompileModelConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_compile_and_noop_nodes_twice(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        original_compile = torch.compile
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = "compile_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_compile_torch import COMPILE_FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = COMPILE_FIXTURE_NODES\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "compile-fixture-dinkster-pack.toml"
            fixture_manifest.write_text(
                "[pack]\n"
                'name = "compile-fixture"\n'
                'publisher = "dinkster"\n'
                'namespaces = ["compile-fixture"]\n'
                "\n[pack.entry]\n"
                f'nodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            specs = (
                (
                    PackSpec(manifest=manifest, in_process=True),
                    PackSpec(manifest=fixture_manifest, in_process=True),
                )
                if in_process
                else (
                    PackSpec(
                        manifest=manifest,
                        worker_group="compile-fixture",
                        group_manifests=(manifest, fixture_manifest),
                    ),
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="compile-fixture",
                        group_manifests=(manifest, fixture_manifest),
                    ),
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
            graph = Graph(
                nodes={
                    "source": GraphNode("compile-fixture.source"),
                    "compile": GraphNode(
                        "TorchCompileModel",
                        {"model": Link("source", "MODEL"), "backend": "inductor"},
                    ),
                    "noop": GraphNode("wanBlockSwap", {"model": Link("compile", "MODEL")}),
                    "consumer": GraphNode(
                        "compile-fixture.consumer", {"model": Link("noop", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["consumer"])
            assert first.outputs["consumer"]["value"].resolve() == 2.0
            second = await engine.run(graph, ["consumer"])
            assert second.outputs["consumer"]["value"].resolve() == 2.0
            assert set(second.cached) == set(graph.nodes)
        finally:
            torch.compile = original_compile
            CompileModelSource.original_compile = None
            await composer.close()

    asyncio.run(scenario())
