"""Native LTXV token-dependent model sampling and sigma schedules."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    MODEL_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_SHIFT_INPUTS = (
    InputSpec(
        "max_shift",
        TypeExpr.concrete(CORE_FLOAT),
        default=2.05,
        widget=NumberWidget(min=0.0, max=100.0, step=0.01),
    ),
    InputSpec(
        "base_shift",
        TypeExpr.concrete(CORE_FLOAT),
        default=0.95,
        widget=NumberWidget(min=0.0, max=100.0, step=0.01),
    ),
)


def _shift(max_shift: float, base_shift: float, latent: object | None) -> float:
    tokens = 4096 if latent is None else math.prod(materialize_latent(latent)["samples"].shape[2:])
    slope = (max_shift - base_shift) / (4096 - 1024)
    return tokens * slope + (base_shift - slope * 1024)


class ModelSamplingLTXV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ModelSamplingLTXV",
            category="model/patch/ltxv",
            inputs=(
                InputSpec("model", _MODEL),
                *_SHIFT_INPUTS,
                InputSpec("latent", _LATENT, required=False),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, max_shift: float, base_shift: float, latent: object | None = None
    ) -> Mapping[str, object]:
        initialize_inference_process()
        modules = importlib.import_module("dinkster_inference.model_sampling")
        result = model.clone()
        sampling_class = type(
            "ModelSamplingAdvanced", (modules.ModelSamplingFlux, modules.CONST), {}
        )
        sampling = sampling_class(model.model.model_config)
        sampling.set_parameters(shift=_shift(max_shift, base_shift, latent))
        result.add_object_patch("model_sampling", sampling)
        return cls.outputs(MODEL=result)


class LTXVScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec(
                    "steps",
                    TypeExpr.concrete(CORE_INT),
                    default=20,
                    widget=NumberWidget(min=1, max=10000),
                ),
                *_SHIFT_INPUTS,
                InputSpec(
                    "stretch",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    advanced=True,
                    doc="Stretch the sigmas to be in the range [terminal, 1].",
                ),
                InputSpec(
                    "terminal",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=0.99, step=0.01),
                    advanced=True,
                    doc="The terminal value of the sigmas after stretching.",
                ),
                InputSpec("latent", _LATENT, required=False),
            ),
            outputs=(OutputSpec("SIGMAS", TypeExpr.concrete(SIGMAS_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        max_shift: float,
        base_shift: float,
        stretch: bool,
        terminal: float,
        latent: object | None = None,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        shift = _shift(max_shift, base_shift, latent)
        sigmas = torch.where(
            sigmas != 0,
            math.exp(shift) / (math.exp(shift) + (1 / sigmas - 1) ** 1),
            0,
        )
        if stretch:
            non_zero = sigmas != 0
            one_minus = 1.0 - sigmas[non_zero]
            scale = one_minus[-1] / (1.0 - terminal)
            sigmas[non_zero] = 1.0 - one_minus / scale
        return cls.outputs(SIGMAS=sigmas)


LTX_SAMPLING_NODES = (ModelSamplingLTXV, LTXVScheduler)
