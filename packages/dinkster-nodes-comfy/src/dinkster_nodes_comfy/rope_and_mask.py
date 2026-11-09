"""ROPE geometry and timestep-dependent denoise-mask model patches."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)


class ScaleROPE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ScaleROPE",
            category="model/patch",
            description="Scale and shift the ROPE of the model.",
            inputs=(
                InputSpec("model", _MODEL),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_FLOAT),
                        default=0.0 if name.startswith("shift") else 1.0,
                        advanced=True,
                        widget=NumberWidget(
                            min=-256.0 if name.startswith("shift") else 0.0,
                            max=256.0 if name.startswith("shift") else 100.0,
                            step=0.1,
                        ),
                    )
                    for name in ("scale_x", "shift_x", "scale_y", "shift_y", "scale_t", "shift_t")
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        scale_x: float,
        shift_x: float,
        scale_y: float,
        shift_y: float,
        scale_t: float,
        shift_t: float,
    ) -> Mapping[str, object]:
        patched = model.clone()
        patched.set_model_rope_options(scale_x, shift_x, scale_y, shift_y, scale_t, shift_t)
        return cls.outputs(MODEL=patched)


class DifferentialDiffusion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DifferentialDiffusion",
            display_name="Differential Diffusion",
            category="experimental",
            search_terms=("inpaint gradient", "variable denoise strength"),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    required=False,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, strength: float = 1.0) -> Mapping[str, object]:
        patched = model.clone()
        patched.set_model_denoise_mask_function(partial(cls.forward, strength=strength))
        return cls.outputs(MODEL=patched)

    @classmethod
    def forward(
        cls, sigma: Any, denoise_mask: Any, extra_options: Mapping[str, Any], strength: float
    ) -> Any:
        sampling = extra_options["model"].inner_model.model_sampling
        step_sigmas = extra_options["sigmas"]
        sigma_to = sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        ts_from = sampling.timestep(step_sigmas[0])
        ts_to = sampling.timestep(sigma_to)
        current_ts = sampling.timestep(sigma[0])
        threshold = (current_ts - ts_to) / (ts_from - ts_to)
        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)
        if strength and strength < 1:
            return strength * binary_mask + (1 - strength) * denoise_mask
        return binary_mask


ROPE_AND_MASK_NODES = (ScaleROPE, DifferentialDiffusion)
