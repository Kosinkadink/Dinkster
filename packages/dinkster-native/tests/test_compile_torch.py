"""CPU contracts for native Torch compilation and wanBlockSwap."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_native.compile import TorchCompileModel, wanBlockSwap


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

    class Executor:
        class_obj = patched.model

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return self.class_obj.diffusion_model(value)

    executor = Executor()
    value = torch.tensor([[-3.0, 5.0]])
    assert torch.equal(wrapper(executor, value), value)
    assert torch.equal(wrapper(executor, value), value)
    assert executor.class_obj.diffusion_model is original

    class FailingExecutor(Executor):
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            assert self.class_obj.diffusion_model is compiled
            raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        wrapper(FailingExecutor(), value)
    assert executor.class_obj.diffusion_model is original


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

    assert schema.node_type == "comfy.wanBlockSwap"
    assert schema.deprecation is not None
    assert schema.search_visibility == "deprecated"
    assert wanBlockSwap.execute(model=model)["MODEL"] is model
