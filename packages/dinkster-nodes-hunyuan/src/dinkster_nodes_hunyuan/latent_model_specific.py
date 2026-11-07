from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    initialize_inference_process,
)


def _integer(
    name: str, default: int, maximum: int, minimum: int = 1, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_INT),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _schema(
    name: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    display: str | None = None,
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=name,
        display_name=display or name,
        category=category,
        description=description,
        inputs=inputs,
        outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
    )


def _runtime() -> tuple[Any, Any]:
    initialize_inference_process()
    return importlib.import_module("torch"), importlib.import_module(
        "dinkster_inference.model_management"
    )


class EmptyLatentHunyuan3Dv2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "EmptyLatentHunyuan3Dv2",
            "model/latent/hunyuan 3d",
            (_integer("resolution", 3072, 8192), _integer("batch_size", 1, 4096)),
            "Empty Hunyuan 3D v2 Latent",
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, resolution: int, batch_size: int) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros([batch_size, 64, resolution], device=management.intermediate_device())
        return cls.outputs(LATENT={"samples": samples, "type": "hunyuan3dv2"})
