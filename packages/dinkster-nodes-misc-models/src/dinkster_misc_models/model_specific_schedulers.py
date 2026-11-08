"""Model-specific scheduler tables and their native SIGMAS nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
)

_LOGSNR_MIN = -15.0

_LOGSNR_MAX = 18.0


def _logit_normal_schedule(u: Any, mean: float, std: float) -> Any:
    torch = importlib.import_module("torch")
    u = torch.as_tensor(u, dtype=torch.float64)
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    return t.clamp(t_min, t_max)


def _ideogram4_sigmas(num_steps: int, width: int, height: int, mu: float, std: float) -> Any:
    torch = importlib.import_module("torch")
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
    sigmas = (1.0 - _logit_normal_schedule(u, mean, std)).flip(0)
    sigmas[-1] = 0.0
    return sigmas.to(torch.float32)


class Ideogram4Scheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="Ideogram4Scheduler",
            display_name="Ideogram 4 Scheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec(
                    "steps",
                    TypeExpr.concrete(CORE_INT),
                    default=20,
                    widget=NumberWidget(min=1, max=200),
                ),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=256, max=8192, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=256, max=8192, step=16),
                ),
                InputSpec(
                    "mu",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.05),
                ),
                InputSpec(
                    "std",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.75,
                    widget=NumberWidget(min=0.1, max=5.0, step=0.05),
                ),
            ),
            outputs=(OutputSpec("SIGMAS", TypeExpr.concrete(SIGMAS_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(
        cls, *, steps: int, width: int, height: int, mu: float, std: float
    ) -> Mapping[str, object]:
        return cls.outputs(SIGMAS=_ideogram4_sigmas(steps, width, height, mu, std))
