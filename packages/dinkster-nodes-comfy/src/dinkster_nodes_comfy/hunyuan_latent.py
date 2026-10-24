"""Hunyuan image latent constructors and refiner conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


class EmptyHunyuanImageLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="EmptyHunyuanImageLatent",
            display_name="Empty Hunyuan Image Latent",
            category="model/latent/hunyuan image",
            inputs=(
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=2048,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=2048,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        return cls.outputs(
            LATENT={
                "samples": torch.zeros(
                    [batch_size, 64, height // 32, width // 32],
                    device=management.intermediate_device(),
                )
            }
        )


class HunyuanRefinerLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HunyuanRefinerLatent",
            display_name="Hunyuan Latent Refiner",
            category="model/conditioning/hunyuan video",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
                InputSpec(
                    "noise_augmentation",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        latent: object,
        noise_augmentation: float,
    ) -> Mapping[str, object]:
        source = materialize_latent(latent)
        samples = source["samples"]
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        values = {"concat_latent_image": samples, "noise_augmentation": noise_augmentation}
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
            latent={
                **source,
                "samples": torch.zeros(
                    [samples.shape[0], 32, *samples.shape[-3:]],
                    device=management.intermediate_device(),
                ),
            },
        )


HUNYUAN_LATENT_NODES = (EmptyHunyuanImageLatent, HunyuanRefinerLatent)
