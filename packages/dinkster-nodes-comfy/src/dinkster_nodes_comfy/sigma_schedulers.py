"""Native ComfyUI sigma scheduler nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    SIGMAS_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
_CATEGORY = "model/sampling/schedulers"
_SCHEDULERS = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)


def _integer(name: str, default: int, minimum: int, maximum: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum),
    )


def _float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    advanced: bool = True,
) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


class BasicScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BasicScheduler",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scheduler",
                    _COMBO,
                    widget=ComboWidget(tuple(ComboOption(value) for value in _SCHEDULERS)),
                ),
                _integer("steps", 20, 1, 10_000),
                _float("denoise", 1.0, 0.0, 1.0, 0.01, advanced=False),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, scheduler: str, steps: int, denoise: float
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        if denoise <= 0.0:
            return cls.outputs(SIGMAS=torch.FloatTensor([]))
        samplers = importlib.import_module("dinkster_inference.samplers")
        total_steps = int(steps / denoise) if denoise < 1.0 else steps
        sigmas = samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        return cls.outputs(SIGMAS=sigmas[-(steps + 1) :])


class KarrasScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="KarrasScheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 10_000),
                _float("sigma_max", 14.614642, 0.0, 5000.0, 0.01),
                _float("sigma_min", 0.0291675, 0.0, 5000.0, 0.01),
                _float("rho", 7.0, 0.0, 100.0, 0.01),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls, *, steps: int, sigma_max: float, sigma_min: float, rho: float
    ) -> Mapping[str, object]:
        sampling = importlib.import_module("dinkster_inference.k_diffusion.sampling")
        return cls.outputs(SIGMAS=sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho))


class ExponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ExponentialScheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 10_000),
                _float("sigma_max", 14.614642, 0.0, 5000.0, 0.01),
                _float("sigma_min", 0.0291675, 0.0, 5000.0, 0.01),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, steps: int, sigma_max: float, sigma_min: float) -> Mapping[str, object]:
        sampling = importlib.import_module("dinkster_inference.k_diffusion.sampling")
        return cls.outputs(SIGMAS=sampling.get_sigmas_exponential(steps, sigma_min, sigma_max))


class PolyexponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PolyexponentialScheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 10_000),
                _float("sigma_max", 14.614642, 0.0, 5000.0, 0.01),
                _float("sigma_min", 0.0291675, 0.0, 5000.0, 0.01),
                _float("rho", 1.0, 0.0, 100.0, 0.01),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls, *, steps: int, sigma_max: float, sigma_min: float, rho: float
    ) -> Mapping[str, object]:
        sampling = importlib.import_module("dinkster_inference.k_diffusion.sampling")
        return cls.outputs(
            SIGMAS=sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho=rho)
        )


class LaplaceScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LaplaceScheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 10_000),
                _float("sigma_max", 14.614642, 0.0, 5000.0, 0.01),
                _float("sigma_min", 0.0291675, 0.0, 5000.0, 0.01),
                _float("mu", 0.0, -10.0, 10.0, 0.1),
                _float("beta", 0.5, 0.0, 10.0, 0.1),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls, *, steps: int, sigma_max: float, sigma_min: float, mu: float, beta: float
    ) -> Mapping[str, object]:
        sampling = importlib.import_module("dinkster_inference.k_diffusion.sampling")
        return cls.outputs(
            SIGMAS=sampling.get_sigmas_laplace(steps, sigma_min, sigma_max, mu=mu, beta=beta)
        )


class VPScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VPScheduler",
            category=_CATEGORY,
            inputs=(
                _integer("steps", 20, 1, 10_000),
                _float("beta_d", 19.9, 0.0, 5000.0, 0.01),
                _float("beta_min", 0.1, 0.0, 5000.0, 0.01),
                _float("eps_s", 0.001, 0.0, 1.0, 0.0001),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(
        cls, *, steps: int, beta_d: float, beta_min: float, eps_s: float
    ) -> Mapping[str, object]:
        sampling = importlib.import_module("dinkster_inference.k_diffusion.sampling")
        return cls.outputs(
            SIGMAS=sampling.get_sigmas_vp(steps, beta_d=beta_d, beta_min=beta_min, eps_s=eps_s)
        )


class BetaSamplingScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BetaSamplingScheduler",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                _integer("steps", 20, 1, 10_000),
                _float("alpha", 0.6, 0.0, 50.0, 0.01),
                _float("beta", 0.6, 0.0, 50.0, 0.01),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, model: Any, steps: int, alpha: float, beta: float) -> Mapping[str, object]:
        samplers = importlib.import_module("dinkster_inference.samplers")
        return cls.outputs(
            SIGMAS=samplers.beta_scheduler(
                model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta
            )
        )


class SDTurboScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SDTurboScheduler",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                _integer("steps", 1, 1, 10),
                _float("denoise", 1.0, 0.0, 1.0, 0.01, advanced=False),
            ),
            outputs=(OutputSpec("SIGMAS", _SIGMAS),),
        )

    @classmethod
    def execute(cls, *, model: Any, steps: int, denoise: float) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step : start_step + steps]
        sigmas = model.get_model_object("model_sampling").sigma(timesteps)
        return cls.outputs(SIGMAS=torch.cat((sigmas, sigmas.new_zeros((1,)))))


SIGMA_SCHEDULER_NODES = (
    BasicScheduler,
    KarrasScheduler,
    ExponentialScheduler,
    PolyexponentialScheduler,
    LaplaceScheduler,
    VPScheduler,
    BetaSamplingScheduler,
    SDTurboScheduler,
)
