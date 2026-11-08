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


class EmptyHiDreamO1LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "EmptyHiDreamO1LatentImage",
            "model/latent/hidream",
            (
                _integer("width", 2048, 4096, 64, 32),
                _integer("height", 2048, 4096, 64, 32),
                _integer("batch_size", 1, 64),
            ),
            "Empty HiDream-O1 Latent Image",
            "Empty pixel-space latent for HiDream-O1-Image. The model was "
            "trained at ~4 megapixels; lower resolutions go off-distribution "
            "and quality regresses noticeably. Trained resolutions: "
            "2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, "
            "2496x1664, 1664x2496, 3104x1312, 1312x3104, 2304x1792, 1792x2304.",
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            (batch_size, 3, height, width), device=management.intermediate_device()
        )
        return cls.outputs(LATENT={"samples": samples})
