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
    initialize_inference_process,
    with_identity_replacement,
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


class EmptyFlux2LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "dinkster.flux.empty_flux2_latent_image",
            "model/latent/flux",
            _image_inputs(),
            "Empty Flux 2 Latent",
        )
        return with_identity_replacement(schema, "EmptyFlux2LatentImage")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 128, height // 16, width // 16], device=management.intermediate_device()
        )
        return cls.outputs(LATENT={"samples": samples})
