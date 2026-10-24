"""Perpendicular negative guidance nodes backed by the pinned sampler."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    GUIDER_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_GUIDER = TypeExpr.concrete(GUIDER_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


def _inference_modules() -> tuple[Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("dinkster_inference.samplers"),
        importlib.import_module("dinkster_inference.sampler_helpers"),
        importlib.import_module("node_helpers"),
    )


def _perp_neg(
    noise_pred_pos: Any,
    noise_pred_neg: Any,
    noise_pred_empty: Any,
    neg_scale: float,
    cond_scale: float,
) -> Any:
    torch = importlib.import_module("torch")
    positive = noise_pred_pos - noise_pred_empty
    negative = noise_pred_neg - noise_pred_empty
    perpendicular = (
        negative - (torch.mul(negative, positive).sum() / torch.norm(positive).pow(2)) * positive
    )
    return noise_pred_empty + cond_scale * (positive - perpendicular * neg_scale)


class PerpNeg(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PerpNeg",
            display_name="Perp-Neg (Deprecated by Perp-Neg Guider)",
            category="experimental",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("empty_conditioning", _CONDITIONING),
                InputSpec(
                    "neg_scale",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, empty_conditioning: Any, neg_scale: float
    ) -> Mapping[str, object]:
        _, sampler_helpers, _ = _inference_modules()
        patched = model.clone()
        empty_cond = sampler_helpers.convert_cond(empty_conditioning)

        def cfg_function(args: dict[str, Any]) -> Any:
            samplers, _, _ = _inference_modules()
            x = args["input"]
            empty_processed = samplers.encode_model_conds(
                args["model"].extra_conds, empty_cond, x, x.device, "negative"
            )
            (noise_pred_empty,) = samplers.calc_cond_batch(
                args["model"], [empty_processed], x, args["sigma"], args["model_options"]
            )
            return x - _perp_neg(
                args["cond_denoised"],
                args["uncond_denoised"],
                noise_pred_empty,
                neg_scale,
                args["cond_scale"],
            )

        patched.set_model_sampler_cfg_function(cfg_function)
        return cls.outputs(MODEL=patched)


class PerpNegGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PerpNegGuider",
            display_name="Perp-Neg Guider",
            category="experimental",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("empty_conditioning", _CONDITIONING),
                InputSpec(
                    "cfg",
                    _FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                InputSpec(
                    "neg_scale",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("GUIDER", _GUIDER),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Any,
        negative: Any,
        empty_conditioning: Any,
        cfg: float,
        neg_scale: float,
    ) -> Mapping[str, object]:
        samplers, _, node_helpers = _inference_modules()

        class GuiderPerpNeg(samplers.CFGGuider):
            def set_conds(self, positive: Any, negative: Any, empty: Any) -> None:
                empty = node_helpers.conditioning_set_values(empty, {"prompt_type": "negative"})
                self.inner_set_conds(
                    {"positive": positive, "negative": negative, "empty_negative_prompt": empty}
                )

            def set_cfg(self, cfg: float, neg_scale: float) -> None:
                self.cfg = cfg
                self.neg_scale = neg_scale

            def predict_noise(
                self,
                x: Any,
                timestep: Any,
                model_options: dict[str, Any] | None = None,
                seed: Any = None,
            ) -> Any:
                if model_options is None:
                    model_options = {}
                positive_cond = self.conds.get("positive")
                negative_cond = self.conds.get("negative")
                empty_cond = self.conds.get("empty_negative_prompt")
                if not model_options.get("disable_cfg1_optimization", False) and math.isclose(
                    self.neg_scale, 0.0
                ):
                    negative_cond = None
                    if math.isclose(self.cfg, 1.0):
                        empty_cond = None
                conds = [positive_cond, negative_cond, empty_cond]
                output = samplers.calc_cond_batch(
                    self.inner_model, conds, x, timestep, model_options
                )
                for callback in model_options.get("sampler_pre_cfg_function", []):
                    output = callback(
                        {
                            "conds": conds,
                            "conds_out": output,
                            "cond_scale": self.cfg,
                            "timestep": timestep,
                            "input": x,
                            "sigma": timestep,
                            "model": self.inner_model,
                            "model_options": model_options,
                        }
                    )
                positive_noise, negative_noise, empty_noise = output
                result = _perp_neg(
                    positive_noise, negative_noise, empty_noise, self.neg_scale, self.cfg
                )
                for callback in model_options.get("sampler_post_cfg_function", []):
                    result = callback(
                        {
                            "denoised": result,
                            "cond": positive_cond,
                            "uncond": negative_cond,
                            "cond_scale": self.cfg,
                            "model": self.inner_model,
                            "uncond_denoised": negative_noise,
                            "cond_denoised": positive_noise,
                            "sigma": timestep,
                            "model_options": model_options,
                            "input": x,
                            "empty_cond": empty_cond,
                            "empty_cond_denoised": empty_noise,
                        }
                    )
                return result

        guider = GuiderPerpNeg(model)
        guider.set_conds(positive, negative, empty_conditioning)
        guider.set_cfg(cfg, neg_scale)
        return cls.outputs(GUIDER=guider)


PERPNEG_NODES = (PerpNeg, PerpNegGuider)
