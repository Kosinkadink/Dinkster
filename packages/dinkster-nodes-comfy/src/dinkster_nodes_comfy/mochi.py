"""Mochi video latent creation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

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
)


class EmptyMochiLatentVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="EmptyMochiLatentVideo",
            category="model/latent/mochi",
            inputs=(
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=848,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=480,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=25,
                    widget=NumberWidget(min=7, max=16384, step=6),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        return cls.outputs(
            LATENT={
                "samples": torch.zeros(
                    [batch_size, 12, ((length - 1) // 6) + 1, height // 8, width // 8],
                    device=management.intermediate_device(),
                )
            }
        )
