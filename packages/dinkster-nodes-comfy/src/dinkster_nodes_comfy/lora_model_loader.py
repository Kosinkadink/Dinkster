"""Apply an in-memory training LoRA through the pinned inference runtime."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    LORA_MODEL_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)


class LoraModelLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoraModelLoader",
            display_name="Load LoRA Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model",
                    TypeExpr.concrete(MODEL_TYPE),
                    doc="The diffusion model the LoRA will be applied to.",
                ),
                InputSpec(
                    "lora",
                    TypeExpr.concrete(LORA_MODEL_TYPE),
                    doc="The LoRA model to apply to the diffusion model.",
                ),
                InputSpec(
                    "strength_model",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-100.0, max=100.0),
                    doc="How strongly to modify the diffusion model. This value can be negative.",
                ),
                InputSpec(
                    "bypass",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=False,
                    doc=(
                        "When enabled, applies LoRA in bypass mode without modifying base model "
                        "weights. Useful for training and when model weights are offloaded."
                    ),
                ),
            ),
            outputs=(
                OutputSpec(
                    "model", TypeExpr.concrete(MODEL_TYPE), doc="The modified diffusion model."
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, lora: Mapping[str, Any], strength_model: float, bypass: bool = False
    ) -> Mapping[str, object]:
        if strength_model == 0:
            return cls.outputs(model=model)
        initialize_inference_process()
        lora = materialize_latent(lora)
        sd = importlib.import_module("dinkster_inference.sd")
        if bypass:
            model_lora, _ = sd.load_bypass_lora_for_models(model, None, lora, strength_model, 0)
        else:
            model_lora, _ = sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return cls.outputs(model=model_lora)
