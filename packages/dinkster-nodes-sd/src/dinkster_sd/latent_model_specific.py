"""Empty latent layouts for image and 3D model families."""

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


def _image_inputs() -> tuple[InputSpec, ...]:
    return (
        _integer("width", 1024, 16384, 16, 16),
        _integer("height", 1024, 16384, 16, 16),
        _integer("batch_size", 1, 4096),
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


class EmptySD3LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema("EmptySD3LatentImage", "model/latent/stable diffusion", _image_inputs())
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(LATENT={"samples": samples, "downscale_ratio_spacial": 8})
