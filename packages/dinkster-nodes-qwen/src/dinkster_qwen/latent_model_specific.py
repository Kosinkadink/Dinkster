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


class EmptyQwenImageLayeredLatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "dinkster.qwen.empty_qwen_image_layered_latent_image",
            "model/latent/qwen",
            (
                _integer("width", 640, 16384, 16, 16),
                _integer("height", 640, 16384, 16, 16),
                _integer("layers", 3, 16384, 0, 1),
                _integer("batch_size", 1, 4096),
            ),
            "Empty Qwen Image Layered Latent",
        )
        return with_identity_replacement(schema, "EmptyQwenImageLayeredLatentImage")

    @classmethod
    def execute(
        cls, *, width: int, height: int, layers: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 16, layers + 1, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        return cls.outputs(LATENT={"samples": samples})
