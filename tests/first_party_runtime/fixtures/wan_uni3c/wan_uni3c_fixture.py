"""Independent public types for ordinary-host Uni3C patch execution."""

from typing import Any

from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    register_inference_types,
)
from test_wan_uni3c_torch import Base, Control, Vae, inspect, patcher, video

_PORTS = (
    ("model", MODEL_TYPE),
    ("weights", MODEL_PATCH_TYPE),
    ("vae", VAE_TYPE),
    ("video", IMAGE_TYPE),
)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-uni3c-fixture.source",
            inputs=(),
            outputs=tuple(OutputSpec(name, TypeExpr.concrete(kind)) for name, kind in _PORTS),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=patcher(Base()), weights=patcher(Control()), vae=Vae(), video=video()
        )


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-uni3c-fixture.inspect",
            inputs=tuple(InputSpec(name, TypeExpr.concrete(kind)) for name, kind in _PORTS)
            + (InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, weights: Any, vae: Any, video: Any, patched: Any
    ) -> dict[str, object]:
        inspect(model, patched, weights, vae, materialize_image(video))
        return cls.outputs(summary="Uni3C residuals and source video preserved")


FIXTURE_NODES = (Source, Inspect)
__all__ = ["FIXTURE_NODES", "register_inference_types"]
