"""Anima LLLite conditioning through the inference model patcher."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class AnimaLLLiteApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AnimaLLLiteApply",
            display_name="Apply Anima LLLite",
            category="model/patch/anima",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE), required=False),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        model_patch: Any,
        image: object,
        strength: float,
        start_percent: float,
        end_percent: float,
        mask: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        lllite = importlib.import_module("dinkster_inference.ldm.anima.lllite")
        pixels = materialize_image(image)[..., :3]
        if model_patch.model.cond_in_channels == 4:
            mask = torch.zeros_like(pixels[..., 0]) if mask is None else materialize_image(mask)
        else:
            mask = None
        sampling = model.get_model_object("model_sampling")
        patch = lllite.AnimaLLLitePatch(
            model_patch,
            pixels,
            mask,
            strength,
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        result = model.clone()
        result.set_model_post_input_patch(patch)
        result.set_model_attn1_patch(
            lllite.AnimaLLLiteAttentionPatch(
                patch, {"q": "self_attn_q_proj", "k": "self_attn_k_proj", "v": "self_attn_v_proj"}
            )
        )
        result.set_model_attn2_patch(
            lllite.AnimaLLLiteAttentionPatch(patch, {"q": "cross_attn_q_proj"})
        )
        result.set_model_patch(lllite.AnimaLLLiteMLPPatch(patch), "mlp_patch")
        return cls.outputs(MODEL=result)
