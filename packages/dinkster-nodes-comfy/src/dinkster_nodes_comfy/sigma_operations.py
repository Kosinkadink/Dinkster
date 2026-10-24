"""Native ComfyUI sigma schedule transformation nodes."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    MODEL_TYPE,
    SIGMAS_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_latent,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_CATEGORY = "model/sampling/sigmas"


def _sigmas(value: object) -> Any:
    return materialize_latent(value)


class SplitSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplitSigmas",
            category=_CATEGORY,
            inputs=(
                InputSpec("sigmas", _SIGMAS),
                InputSpec("step", _INT, default=0, widget=NumberWidget(min=0, max=10_000, step=1)),
            ),
            outputs=(OutputSpec("high_sigmas", _SIGMAS), OutputSpec("low_sigmas", _SIGMAS)),
        )

    @classmethod
    def execute(cls, *, sigmas: object, step: int) -> Mapping[str, object]:
        materialized: Any = _sigmas(sigmas)
        return cls.outputs(high_sigmas=materialized[: step + 1], low_sigmas=materialized[step:])


class SplitSigmasDenoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplitSigmasDenoise",
            category=_CATEGORY,
            inputs=(
                InputSpec("sigmas", _SIGMAS),
                InputSpec(
                    "denoise", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
            ),
            outputs=(OutputSpec("high_sigmas", _SIGMAS), OutputSpec("low_sigmas", _SIGMAS)),
        )

    @classmethod
    def execute(cls, *, sigmas: object, denoise: float) -> Mapping[str, object]:
        materialized: Any = _sigmas(sigmas)
        total_steps = round(max(materialized.shape[-1] - 1, 0) * denoise)
        return cls.outputs(
            high_sigmas=materialized[:-total_steps], low_sigmas=materialized[-(total_steps + 1) :]
        )


class FlipSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FlipSigmas",
            category=_CATEGORY,
            inputs=(InputSpec("sigmas", _SIGMAS),),
            outputs=(OutputSpec("sigmas", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, sigmas: object) -> Mapping[str, object]:
        materialized: Any = _sigmas(sigmas)
        if len(materialized) == 0:
            return cls.outputs(sigmas=materialized)
        result: Any = materialized.flip(0)
        if result[0] == 0:
            result[0] = 0.0001
        return cls.outputs(sigmas=result)


class SetFirstSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SetFirstSigma",
            category=_CATEGORY,
            inputs=(
                InputSpec("sigmas", _SIGMAS),
                InputSpec(
                    "sigma",
                    _FLOAT,
                    default=136.0,
                    widget=NumberWidget(min=0.0, max=20_000.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("sigmas", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, sigmas: object, sigma: float) -> Mapping[str, object]:
        result = _sigmas(sigmas).clone()
        result[0] = sigma
        return cls.outputs(sigmas=result)


class ExtendIntermediateSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ExtendIntermediateSigmas",
            category=_CATEGORY,
            search_terms=("interpolate sigmas",),
            inputs=(
                InputSpec("sigmas", _SIGMAS),
                InputSpec("steps", _INT, default=2, widget=NumberWidget(min=1, max=100, step=1)),
                InputSpec(
                    "start_at_sigma",
                    _FLOAT,
                    default=-1.0,
                    widget=NumberWidget(min=-1.0, max=20_000.0, step=0.01),
                ),
                InputSpec(
                    "end_at_sigma",
                    _FLOAT,
                    default=12.0,
                    widget=NumberWidget(min=0.0, max=20_000.0, step=0.01),
                ),
                InputSpec(
                    "spacing",
                    _COMBO,
                    widget=ComboWidget(
                        tuple(ComboOption(option) for option in ("linear", "cosine", "sine"))
                    ),
                ),
            ),
            outputs=(OutputSpec("sigmas", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls,
        *,
        sigmas: object,
        steps: int,
        start_at_sigma: float,
        end_at_sigma: float,
        spacing: str,
    ) -> Mapping[str, object]:
        materialized: Any = _sigmas(sigmas)
        torch: Any = importlib.import_module("torch")
        if start_at_sigma < 0:
            start_at_sigma = float("inf")
        positions: Any = torch.linspace(0, 1, steps + 1, device=materialized.device)[1:-1]
        if spacing == "linear":
            spacing_values: Any = positions
        elif spacing == "cosine":
            spacing_values = torch.sin(positions * math.pi / 2)
        elif spacing == "sine":
            spacing_values = 1 - torch.cos(positions * math.pi / 2)
        else:
            raise KeyError(spacing)
        extended: list[Any] = []
        for index in range(len(materialized) - 1):
            current, following = materialized[index], materialized[index + 1]
            extended.append(current)
            if end_at_sigma <= current <= start_at_sigma:
                extended.extend((spacing_values * (following - current) + current).tolist())
        if len(materialized) > 0:
            extended.append(materialized[-1])
        return cls.outputs(sigmas=torch.FloatTensor(extended))


class SamplingPercentToSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SamplingPercentToSigma",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "sampling_percent",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.0001),
                ),
                InputSpec(
                    "return_actual_sigma",
                    _BOOLEAN,
                    default=False,
                    doc=(
                        "Return the actual sigma value instead of the value used for interval "
                        "checks.\n"
                        "This only affects results at 0.0 and 1.0."
                    ),
                ),
            ),
            outputs=(OutputSpec("sigma_value", _FLOAT),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, sampling_percent: float, return_actual_sigma: bool
    ) -> Mapping[str, object]:
        sampling = model.get_model_object("model_sampling")
        sigma = sampling.percent_to_sigma(sampling_percent)
        if return_actual_sigma:
            if sampling_percent == 0.0:
                sigma = sampling.sigma_max.item()
            elif sampling_percent == 1.0:
                sigma = sampling.sigma_min.item()
        return cls.outputs(sigma_value=sigma)


class ManualSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ManualSigmas",
            category=_CATEGORY,
            search_terms=("custom noise schedule", "define sigmas"),
            inputs=(
                InputSpec(
                    "sigmas", _STRING, default="1, 0.5", widget=StringWidget(multiline=False)
                ),
            ),
            outputs=(OutputSpec("sigmas", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, sigmas: str) -> Mapping[str, object]:
        torch: Any = importlib.import_module("torch")
        return cls.outputs(
            sigmas=torch.FloatTensor(
                [float(value) for value in re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)]
            )
        )


SIGMA_OPERATION_NODES = (
    SplitSigmas,
    SplitSigmasDenoise,
    FlipSigmas,
    SetFirstSigma,
    ExtendIntermediateSigmas,
    SamplingPercentToSigma,
    ManualSigmas,
)
