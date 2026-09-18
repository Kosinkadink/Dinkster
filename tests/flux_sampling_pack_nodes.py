"""Worker fixture executing the native Flux patch over a weight-free handle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_compat_comfy import register_resident_type
from dinkster_compat_comfy.native_arm import GenerationModelSamplingFlux
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_FLOAT, TypeRegistry

MODEL = TypeExpr.concrete("dinkster.model")
FLOAT = TypeExpr.concrete(CORE_FLOAT)


def register_types(registry: TypeRegistry) -> None:
    register_resident_type(registry, "dinkster.model")


class LoadModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.flux_model",
            display_name="Test Flux Model",
            category="test",
            inputs=(),
            outputs=(OutputSpec("model", MODEL),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        from test_native_arm import _handle, _native_arm, _runtime

        runtime = _runtime()
        runtime.with_sampling_space = lambda _space: runtime
        return cls.outputs(model=_handle(_native_arm(), runtime))


class ReadShift(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.flux_shift",
            display_name="Read Flux Shift",
            category="test",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("shift", FLOAT),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        model.require_active()
        return cls.outputs(shift=model.sampling_space.shift)


NODES = (LoadModel, GenerationModelSamplingFlux, ReadShift)
