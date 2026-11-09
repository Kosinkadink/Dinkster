"""Native ComfyUI sampler factory nodes backed by dinkster_inference."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    SAMPLER_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_SAMPLER = TypeExpr.concrete(SAMPLER_TYPE)
_CATEGORY = "model/sampling/samplers"

# This is the fixed ComfyUI SAMPLER_NAMES order at bc5f880dbdd43e426077f42ccec53bb577ffee9f.
SAMPLER_NAMES = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "cfgpp_ud10_ab",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)


def _runtime_samplers() -> Any:
    initialize_inference_process()
    samplers = importlib.import_module("dinkster_inference.samplers")
    if tuple(samplers.SAMPLER_NAMES) != SAMPLER_NAMES:
        raise RuntimeError(
            "pinned dinkster_inference sampler names differ from the native node schema"
        )
    return samplers


def _sampler_output(sampler: Any) -> Mapping[str, object]:
    return {"SAMPLER": sampler}


def _float(
    name: str,
    default: float,
    maximum: float,
    *,
    minimum: float = 0.0,
    step: float = 0.01,
    advanced: bool = False,
    doc: str = "",
    round_: bool | float | None = False,
) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=advanced,
        doc=doc,
        widget=NumberWidget(min=minimum, max=maximum, step=step, round=round_),
    )


def _schema(
    node_type: str,
    inputs: tuple[InputSpec, ...],
    *,
    display_name: str = "",
    search_terms: tuple[str, ...] = (),
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name or node_type,
        category=_CATEGORY,
        inputs=inputs,
        outputs=(OutputSpec("SAMPLER", _SAMPLER),),
        search_terms=search_terms,
        description=description,
    )


class KSamplerSelect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KSamplerSelect",
            (InputSpec("sampler_name", _COMBO, widget=ComboWidget(options=SAMPLER_NAMES)),),
        )

    @classmethod
    def execute(cls, *, sampler_name: str) -> Mapping[str, object]:
        return _sampler_output(_runtime_samplers().sampler_object(sampler_name))


class SamplerEulerAncestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerEulerAncestral",
            (
                _float("eta", 1.0, 100.0, advanced=True),
                _float("s_noise", 1.0, 100.0, advanced=True),
            ),
        )

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return _sampler_output(
            _runtime_samplers().ksampler("euler_ancestral", {"eta": eta, "s_noise": s_noise})
        )


class SamplerEulerAncestralCFGPP(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerEulerAncestralCFGPP",
            (_float("eta", 1.0, 1.0), _float("s_noise", 1.0, 10.0)),
            display_name="SamplerEulerAncestralCFG++",
        )

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return _sampler_output(
            _runtime_samplers().ksampler("euler_ancestral_cfg_pp", {"eta": eta, "s_noise": s_noise})
        )


class SamplerLMS(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerLMS",
            (
                InputSpec(
                    "order", _INT, default=4, advanced=True, widget=NumberWidget(min=1, max=100)
                ),
            ),
        )

    @classmethod
    def execute(cls, *, order: int) -> Mapping[str, object]:
        return _sampler_output(_runtime_samplers().ksampler("lms", {"order": order}))


class SamplerDPMPP_3M_SDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerDPMPP_3M_SDE",
            (
                _float("eta", 1.0, 100.0, advanced=True),
                _float("s_noise", 1.0, 100.0, advanced=True),
                InputSpec(
                    "noise_device",
                    _COMBO,
                    advanced=True,
                    widget=ComboWidget(options=("gpu", "cpu")),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, eta: float, s_noise: float, noise_device: str) -> Mapping[str, object]:
        name = "dpmpp_3m_sde" if noise_device == "cpu" else "dpmpp_3m_sde_gpu"
        return _sampler_output(_runtime_samplers().ksampler(name, {"eta": eta, "s_noise": s_noise}))


class SamplerDPMPP_2M_SDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerDPMPP_2M_SDE",
            (
                InputSpec("solver_type", _COMBO, widget=ComboWidget(options=("midpoint", "heun"))),
                _float("eta", 1.0, 100.0, advanced=True),
                _float("s_noise", 1.0, 100.0, advanced=True),
                InputSpec(
                    "noise_device",
                    _COMBO,
                    advanced=True,
                    widget=ComboWidget(options=("gpu", "cpu")),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, solver_type: str, eta: float, s_noise: float, noise_device: str
    ) -> Mapping[str, object]:
        name = "dpmpp_2m_sde" if noise_device == "cpu" else "dpmpp_2m_sde_gpu"
        return _sampler_output(
            _runtime_samplers().ksampler(
                name, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type}
            )
        )


class SamplerDPMPP_SDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerDPMPP_SDE",
            (
                _float("eta", 1.0, 100.0, advanced=True),
                _float("s_noise", 1.0, 100.0, advanced=True),
                _float("r", 0.5, 100.0, advanced=True),
                InputSpec(
                    "noise_device",
                    _COMBO,
                    advanced=True,
                    widget=ComboWidget(options=("gpu", "cpu")),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, eta: float, s_noise: float, r: float, noise_device: str
    ) -> Mapping[str, object]:
        name = "dpmpp_sde" if noise_device == "cpu" else "dpmpp_sde_gpu"
        return _sampler_output(
            _runtime_samplers().ksampler(name, {"eta": eta, "s_noise": s_noise, "r": r})
        )


class SamplerDPMPP_2S_Ancestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerDPMPP_2S_Ancestral",
            (_float("eta", 1.0, 100.0), _float("s_noise", 1.0, 100.0)),
        )

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return _sampler_output(
            _runtime_samplers().ksampler("dpmpp_2s_ancestral", {"eta": eta, "s_noise": s_noise})
        )


class SamplerDPMAdaptative(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerDPMAdaptative",
            (
                InputSpec(
                    "order", _INT, default=3, advanced=True, widget=NumberWidget(min=2, max=3)
                ),
                _float("rtol", 0.05, 100.0, advanced=True),
                _float("atol", 0.0078, 100.0, advanced=True),
                _float("h_init", 0.05, 100.0, advanced=True),
                _float("pcoeff", 0.0, 100.0, advanced=True),
                _float("icoeff", 1.0, 100.0, advanced=True),
                _float("dcoeff", 0.0, 100.0, advanced=True),
                _float("accept_safety", 0.81, 100.0, advanced=True),
                _float("eta", 0.0, 100.0, advanced=True),
                _float("s_noise", 1.0, 100.0, advanced=True),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        order: int,
        rtol: float,
        atol: float,
        h_init: float,
        pcoeff: float,
        icoeff: float,
        dcoeff: float,
        accept_safety: float,
        eta: float,
        s_noise: float,
    ) -> Mapping[str, object]:
        return _sampler_output(
            _runtime_samplers().ksampler(
                "dpm_adaptive",
                {
                    "order": order,
                    "rtol": rtol,
                    "atol": atol,
                    "h_init": h_init,
                    "pcoeff": pcoeff,
                    "icoeff": icoeff,
                    "dcoeff": dcoeff,
                    "accept_safety": accept_safety,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            )
        )


class SamplerER_SDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerER_SDE",
            (
                InputSpec(
                    "solver_type",
                    _COMBO,
                    widget=ComboWidget(options=("ER-SDE", "Reverse-time SDE", "ODE")),
                ),
                InputSpec(
                    "max_stage", _INT, default=3, advanced=True, widget=NumberWidget(min=1, max=3)
                ),
                _float(
                    "eta",
                    1.0,
                    10.0,
                    advanced=True,
                    doc=(
                        "Stochastic strength of SDEs.\n"
                        "When eta=0, they reduce to deterministic ODE.\n"
                        "Large eta may cause invalid outputs.\n"
                        "If this occurs, try decreasing this value."
                    ),
                ),
                _float("s_noise", 1.0, 100.0, advanced=True),
            ),
        )

    @classmethod
    def execute(
        cls, *, solver_type: str, max_stage: int, eta: float, s_noise: float
    ) -> Mapping[str, object]:
        initialize_inference_process()

        def er_sde_noise_scaler(x: Any) -> Any:
            return x * ((x**0.3).exp() + 10.0) ** eta

        def reverse_time_sde_noise_scaler(x: Any) -> Any:
            return x ** (eta + 1)

        def ode_noise_scaler(x: Any) -> Any:
            return x

        solver_scalers = {
            "ER-SDE": er_sde_noise_scaler,
            "Reverse-time SDE": reverse_time_sde_noise_scaler,
            "ODE": ode_noise_scaler,
        }
        if solver_type == "ODE" or eta == 0:
            s_noise = 0.0
            solver_type = "ODE"
        return _sampler_output(
            _runtime_samplers().ksampler(
                "er_sde",
                {
                    "s_noise": s_noise,
                    "noise_scaler": solver_scalers[solver_type],
                    "max_stage": max_stage,
                },
            )
        )


class SamplerSASolver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerSASolver",
            (
                InputSpec("model", _MODEL),
                _float("eta", 1.0, 10.0, advanced=True),
                _float("sde_start_percent", 0.2, 1.0, advanced=True, step=0.001, round_=None),
                _float("sde_end_percent", 0.8, 1.0, advanced=True, step=0.001, round_=None),
                _float("s_noise", 1.0, 100.0, advanced=True),
                InputSpec(
                    "predictor_order",
                    _INT,
                    default=3,
                    advanced=True,
                    widget=NumberWidget(min=1, max=6),
                ),
                InputSpec(
                    "corrector_order",
                    _INT,
                    default=4,
                    advanced=True,
                    widget=NumberWidget(min=0, max=6),
                ),
                InputSpec("use_pece", _BOOLEAN, advanced=True),
                InputSpec("simple_order_2", _BOOLEAN, advanced=True),
            ),
            search_terms=("sde",),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        eta: float,
        sde_start_percent: float,
        sde_end_percent: float,
        s_noise: float,
        predictor_order: int,
        corrector_order: int,
        use_pece: bool,
        simple_order_2: bool,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        sa_solver = importlib.import_module("dinkster_inference.k_diffusion.sa_solver")
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=eta)
        return _sampler_output(
            _runtime_samplers().ksampler(
                "sa_solver",
                {
                    "tau_func": tau_func,
                    "s_noise": s_noise,
                    "predictor_order": predictor_order,
                    "corrector_order": corrector_order,
                    "use_pece": use_pece,
                    "simple_order_2": simple_order_2,
                },
            )
        )


class SamplerSEEDS2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerSEEDS2",
            (
                InputSpec("solver_type", _COMBO, widget=ComboWidget(options=("phi_1", "phi_2"))),
                _float("eta", 1.0, 100.0, advanced=True, doc="Stochastic strength"),
                _float("s_noise", 1.0, 100.0, advanced=True, doc="SDE noise multiplier"),
                _float(
                    "r",
                    0.5,
                    1.0,
                    minimum=0.01,
                    advanced=True,
                    doc="Relative step size for the intermediate stage (c2 node)",
                ),
            ),
            search_terms=("sde", "exp heun"),
            description=(
                "This sampler node can represent multiple samplers:\n\n"
                "seeds_2\n"
                "- default setting\n\n"
                "exp_heun_2_x0\n"
                "- solver_type=phi_2, r=1.0, eta=0.0\n\n"
                "exp_heun_2_x0_sde\n"
                "- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0"
            ),
        )

    @classmethod
    def execute(
        cls, *, solver_type: str, eta: float, s_noise: float, r: float
    ) -> Mapping[str, object]:
        return _sampler_output(
            _runtime_samplers().ksampler(
                "seeds_2", {"eta": eta, "s_noise": s_noise, "r": r, "solver_type": solver_type}
            )
        )


SAMPLER_FACTORY_NODES = (
    KSamplerSelect,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerDPMPP_3M_SDE,
    SamplerDPMPP_2M_SDE,
    SamplerDPMPP_SDE,
    SamplerDPMPP_2S_Ancestral,
    SamplerDPMAdaptative,
    SamplerER_SDE,
    SamplerSASolver,
    SamplerSEEDS2,
)
