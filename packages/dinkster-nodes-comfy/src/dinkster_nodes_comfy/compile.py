"""Native Torch compilation and legacy Wan block-swap compatibility nodes."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    MODEL_TYPE,
    ComboWidget,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_COMPILE_KEY = "torch.compile"
_COMPILE_KWARGS_KEY = "torch_compile_kwargs"
_MODEL = TypeExpr.concrete(MODEL_TYPE)


def _skip_transformer_options_guards(guard_entries: Any) -> list[bool]:
    return ["transformer_options" not in entry.name for entry in guard_entries]


def _apply_compiled_diffusion_model(compiled: Any) -> Callable[..., Any]:
    def wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
        original = executor.class_obj.diffusion_model
        executor.class_obj.diffusion_model = compiled
        try:
            return executor(*args, **kwargs)
        finally:
            executor.class_obj.diffusion_model = original

    return wrapper


class TorchCompileModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TorchCompileModel",
            category="experimental",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "backend",
                    TypeExpr.concrete(CORE_COMBO),
                    advanced=True,
                    widget=ComboWidget(("inductor", "cudagraphs")),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, backend: str) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        patcher_extension = importlib.import_module("dinkster_inference.patcher_extension")
        patched = model.clone(disable_dynamic=True)
        compile_kwargs = {
            "backend": backend,
            "options": {"guard_filter_fn": _skip_transformer_options_guards},
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
        }
        compiled = torch.compile(patched.get_model_object("diffusion_model"), **compile_kwargs)
        patched.remove_wrappers_with_key(patcher_extension.WrappersMP.APPLY_MODEL, _COMPILE_KEY)
        patched.add_wrapper_with_key(
            patcher_extension.WrappersMP.APPLY_MODEL,
            _COMPILE_KEY,
            _apply_compiled_diffusion_model(compiled),
        )
        patched.model_options[_COMPILE_KWARGS_KEY] = compile_kwargs
        return cls.outputs(MODEL=patched)


class wanBlockSwap(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wanBlockSwap",
            description=(
                "Intercept wanBlockSwap custom node that causes major instability "
                "and make it no-op."
            ),
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL),),
            deprecation=Deprecation(
                message="wanBlockSwap is a no-op because it conflicts with model memory management."
            ),
            search_visibility="deprecated",
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        return cls.outputs(MODEL=model)


COMPILE_NODES = (TorchCompileModel, wanBlockSwap)
