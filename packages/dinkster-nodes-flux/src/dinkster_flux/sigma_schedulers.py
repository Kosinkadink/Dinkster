"""Native ComfyUI sigma scheduler nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    with_identity_replacement,
)

_INT = TypeExpr.concrete(CORE_INT)

_SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)

_CATEGORY = "model/sampling/schedulers"


def _integer(name: str, default: int, minimum: int, maximum: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum),
    )


class Flux2Scheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.flux2_scheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 4096),
                InputSpec(
                    "width", _INT, default=1024, widget=NumberWidget(min=16, max=16384, step=1)
                ),
                InputSpec(
                    "height", _INT, default=1024, widget=NumberWidget(min=16, max=16384, step=1)
                ),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )
        return with_identity_replacement(schema, "Flux2Scheduler")

    @classmethod
    def execute(cls, *, steps: int, width: int, height: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        sequence_length = round(width * height / (16 * 16))
        long_schedule = 0.00016927 * sequence_length + 0.45666666
        if sequence_length > 4300:
            mu = long_schedule
        else:
            short_schedule = 8.73809524e-05 * sequence_length + 1.89833333
            slope = (long_schedule - short_schedule) / 190.0
            mu = slope * steps + (long_schedule - 200.0 * slope)
        timesteps = torch.linspace(1, 0, steps + 1)
        exponential = math.exp(float(mu))
        return cls.outputs(SIGMAS=exponential / (exponential + (1 / timesteps - 1) ** 1.0))
