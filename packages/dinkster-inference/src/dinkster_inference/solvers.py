"""Backend-agnostic sampler descriptors and option schemas."""

from __future__ import annotations

from typing import Any

from .registry import Registry
from .sampling import (
    BuiltinSamplerSelection,
    NoiseKind,
    OptionKind,
    OptionSpec,
    OptionValue,
    SamplerDescriptor,
    catalog_descriptor_rebuild,
    catalog_descriptor_snapshot,
    resolve_options,
)


def _descriptor(
    name: str,
    display_name: str,
    *,
    namespace: str = "dinkster",
    options: tuple[OptionSpec, ...] = (),
    noise: NoiseKind = NoiseKind.NONE,
    discard_penultimate: bool = False,
    requires_snr_offset: bool = False,
    needs_uncond: bool = False,
    random_inpaint_noise: bool = False,
    aliases: tuple[str, ...] | None = None,
    supports_step_begin: bool = True,
) -> SamplerDescriptor[Any]:
    return SamplerDescriptor(
        id=f"{namespace}.{name}",
        display_name=display_name,
        make=None,
        options=options,
        noise=noise,
        aliases=(name,) if aliases is None else aliases,
        discard_penultimate=discard_penultimate,
        requires_snr_offset=requires_snr_offset,
        needs_uncond=needs_uncond,
        random_inpaint_noise=random_inpaint_noise,
        supports_step_begin=supports_step_begin,
    )


_ANCESTRAL_OPTIONS = (
    OptionSpec(
        "eta",
        OptionKind.FLOAT,
        1.0,
        minimum=0.0,
        maximum=1.0,
        doc="ancestral noise fraction",
    ),
    OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, doc="injected noise multiplier"),
)
_SDE_OPTIONS = (
    OptionSpec("eta", OptionKind.FLOAT, 1.0, minimum=0.0, doc="SDE noise fraction"),
    OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, doc="injected noise multiplier"),
)
_CHURN_OPTIONS = (
    OptionSpec("s_churn", OptionKind.FLOAT, 0.0, minimum=0.0, doc="Karras churn amount"),
    OptionSpec("s_tmin", OptionKind.FLOAT, 0.0, minimum=0.0, doc="churn only at sigma >= s_tmin"),
    OptionSpec(
        "s_tmax",
        OptionKind.FLOAT,
        float("inf"),
        minimum=0.0,
        doc="churn only at sigma <= s_tmax",
    ),
    OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, doc="churn noise multiplier"),
)

DINKSTER_AR_VIDEO = _descriptor(
    "ar_video",
    "AR Video",
    options=(OptionSpec("num_frame_per_block", OptionKind.INT, 1, minimum=1, maximum=64),),
    supports_step_begin=False,
)
DINKSTER_EULER = _descriptor("euler", "Euler", options=_CHURN_OPTIONS, noise=NoiseKind.GAUSSIAN)
DINKSTER_EULER_CFG_PP = _descriptor("euler_cfg_pp", "Euler CFG++", needs_uncond=True)
DINKSTER_EULER_ANCESTRAL = _descriptor(
    "euler_ancestral", "Euler ancestral", options=_ANCESTRAL_OPTIONS, noise=NoiseKind.GAUSSIAN
)
DINKSTER_EULER_ANCESTRAL_CFG_PP = _descriptor(
    "euler_ancestral_cfg_pp",
    "Euler ancestral CFG++",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    needs_uncond=True,
)
DINKSTER_HEUN = _descriptor("heun", "Heun", options=_CHURN_OPTIONS, noise=NoiseKind.GAUSSIAN)
DINKSTER_HEUNPP2 = _descriptor(
    "heunpp2", "Heun++ 2", options=_CHURN_OPTIONS, noise=NoiseKind.GAUSSIAN
)
_PHI_OPTION = (OptionSpec("solver_type", OptionKind.CHOICE, "phi_2", choices=("phi_1", "phi_2")),)
DINKSTER_EXP_HEUN_2_X0 = _descriptor(
    "exp_heun_2_x0", "Exp Heun 2 x0", options=_PHI_OPTION, requires_snr_offset=True
)
DINKSTER_EXP_HEUN_2_X0_SDE = _descriptor(
    "exp_heun_2_x0_sde",
    "Exp Heun 2 x0 SDE",
    options=(*_ANCESTRAL_OPTIONS, *_PHI_OPTION),
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_LMS = _descriptor(
    "lms", "LMS", options=(OptionSpec("order", OptionKind.INT, 4, minimum=1, maximum=100),)
)
DINKSTER_DPM_FAST = _descriptor(
    "dpm_fast",
    "DPM fast",
    options=(
        OptionSpec("eta", OptionKind.FLOAT, 0.0, minimum=0.0, maximum=1.0),
        OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0),
    ),
    noise=NoiseKind.GAUSSIAN,
    supports_step_begin=False,
)
DINKSTER_DPM_ADAPTIVE = _descriptor(
    "dpm_adaptive",
    "DPM adaptive",
    options=(
        OptionSpec("order", OptionKind.INT, 3, minimum=2, maximum=3),
        OptionSpec("rtol", OptionKind.FLOAT, 0.05, minimum=0.0),
        OptionSpec("atol", OptionKind.FLOAT, 0.0078, minimum=0.0),
        OptionSpec("h_init", OptionKind.FLOAT, 0.05, minimum=0.0),
        OptionSpec("pcoeff", OptionKind.FLOAT, 0.0),
        OptionSpec("icoeff", OptionKind.FLOAT, 1.0),
        OptionSpec("dcoeff", OptionKind.FLOAT, 0.0),
        OptionSpec("accept_safety", OptionKind.FLOAT, 0.81, minimum=0.0),
        OptionSpec("eta", OptionKind.FLOAT, 0.0, minimum=0.0, maximum=100.0),
        OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0),
    ),
    noise=NoiseKind.GAUSSIAN,
    supports_step_begin=False,
)
DINKSTER_DPM_2 = _descriptor(
    "dpm_2",
    "DPM-2",
    options=_CHURN_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    discard_penultimate=True,
)
DINKSTER_DPM_2_ANCESTRAL = _descriptor(
    "dpm_2_ancestral",
    "DPM-2 ancestral",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    discard_penultimate=True,
)
DINKSTER_DPMPP_2S_ANCESTRAL = _descriptor(
    "dpmpp_2s_ancestral",
    "DPM++ 2S ancestral",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
)
DINKSTER_DPMPP_2S_ANCESTRAL_CFG_PP = _descriptor(
    "dpmpp_2s_ancestral_cfg_pp",
    "DPM++ 2S ancestral CFG++",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    needs_uncond=True,
)
_DPM_SDE_OPTIONS = (
    *_SDE_OPTIONS,
    OptionSpec(
        "r",
        OptionKind.FLOAT,
        0.5,
        minimum=1e-6,
        maximum=100.0,
        doc="midpoint ratio; zero is refused because the solver divides by r",
    ),
)
DINKSTER_DPMPP_SDE = _descriptor(
    "dpmpp_sde",
    "DPM++ SDE",
    options=_DPM_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_SDE_GPU = _descriptor(
    "dpmpp_sde_gpu",
    "DPM++ SDE GPU",
    options=_DPM_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN_GPU,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_2M = _descriptor("dpmpp_2m", "DPM++ 2M")
DINKSTER_DPMPP_2M_CFG_PP = _descriptor("dpmpp_2m_cfg_pp", "DPM++ 2M CFG++", needs_uncond=True)
_TWO_M_SDE_OPTIONS = (
    *_SDE_OPTIONS,
    OptionSpec(
        "solver_type",
        OptionKind.CHOICE,
        "midpoint",
        choices=("midpoint", "heun"),
        doc="second-order correction style",
    ),
)
DINKSTER_DPMPP_2M_SDE = _descriptor(
    "dpmpp_2m_sde",
    "DPM++ 2M SDE",
    options=_TWO_M_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_2M_SDE_GPU = _descriptor(
    "dpmpp_2m_sde_gpu",
    "DPM++ 2M SDE GPU",
    options=_TWO_M_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN_GPU,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_2M_SDE_HEUN = _descriptor(
    "dpmpp_2m_sde_heun",
    "DPM++ 2M SDE Heun",
    options=_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_2M_SDE_HEUN_GPU = _descriptor(
    "dpmpp_2m_sde_heun_gpu",
    "DPM++ 2M SDE Heun GPU",
    options=_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN_GPU,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_3M_SDE = _descriptor(
    "dpmpp_3m_sde",
    "DPM++ 3M SDE",
    options=_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN,
    requires_snr_offset=True,
)
DINKSTER_DPMPP_3M_SDE_GPU = _descriptor(
    "dpmpp_3m_sde_gpu",
    "DPM++ 3M SDE GPU",
    options=_SDE_OPTIONS,
    noise=NoiseKind.BROWNIAN_GPU,
    requires_snr_offset=True,
)
DINKSTER_DDPM = _descriptor("ddpm", "DDPM", noise=NoiseKind.GAUSSIAN)
DINKSTER_LCM = _descriptor(
    "lcm",
    "LCM",
    options=(
        OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, doc="initial renoise multiplier"),
        OptionSpec(
            "s_noise_end",
            OptionKind.OPTIONAL_FLOAT,
            None,
            minimum=0.0,
            doc="final renoise multiplier",
        ),
        OptionSpec(
            "noise_clip_std",
            OptionKind.FLOAT,
            0.0,
            minimum=0.0,
            doc="noise clamp in standard deviations",
        ),
    ),
    noise=NoiseKind.GAUSSIAN,
)
_MAX_ORDER_OPTION = (OptionSpec("max_order", OptionKind.INT, 4, minimum=1, maximum=4),)
DINKSTER_IPNDM = _descriptor("ipndm", "iPNDM", options=_MAX_ORDER_OPTION)
DINKSTER_IPNDM_V = _descriptor("ipndm_v", "iPNDM V", options=_MAX_ORDER_OPTION)
DINKSTER_DEIS = _descriptor(
    "deis",
    "DEIS",
    options=(
        OptionSpec("max_order", OptionKind.INT, 3, minimum=1, maximum=4),
        OptionSpec("deis_mode", OptionKind.CHOICE, "tab", choices=("tab",)),
    ),
)
_GE_OPTION = (OptionSpec("ge_gamma", OptionKind.FLOAT, 2.0),)
DINKSTER_GRADIENT_ESTIMATION = _descriptor(
    "gradient_estimation", "Gradient estimation", options=_GE_OPTION
)
DINKSTER_GRADIENT_ESTIMATION_CFG_PP = _descriptor(
    "gradient_estimation_cfg_pp",
    "Gradient estimation CFG++",
    options=_GE_OPTION,
    needs_uncond=True,
)
DINKSTER_ER_SDE = _descriptor(
    "er_sde",
    "ER-SDE",
    options=(
        OptionSpec(
            "solver_type",
            OptionKind.CHOICE,
            "ER-SDE",
            choices=("ER-SDE", "Reverse-time SDE", "ODE"),
        ),
        OptionSpec("max_stage", OptionKind.INT, 3, minimum=1, maximum=3),
        OptionSpec("eta", OptionKind.FLOAT, 1.0, minimum=0.0, maximum=100.0),
        OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, maximum=100.0),
    ),
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_SEEDS_2 = _descriptor(
    "seeds_2",
    "SEEDS 2",
    options=(
        *_ANCESTRAL_OPTIONS,
        OptionSpec("r", OptionKind.FLOAT, 0.5, minimum=0.0, maximum=1.0),
        OptionSpec("solver_type", OptionKind.CHOICE, "phi_1", choices=("phi_1", "phi_2")),
    ),
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_SEEDS_3 = _descriptor(
    "seeds_3",
    "SEEDS 3",
    options=(
        *_ANCESTRAL_OPTIONS,
        OptionSpec("r_1", OptionKind.FLOAT, 1 / 3, minimum=0.0, maximum=1.0),
        OptionSpec("r_2", OptionKind.FLOAT, 2 / 3, minimum=0.0, maximum=1.0),
    ),
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
_SA_OPTIONS = (
    OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0),
    OptionSpec("predictor_order", OptionKind.INT, 3, minimum=1, maximum=4),
    OptionSpec("corrector_order", OptionKind.INT, 4, minimum=1, maximum=4),
    OptionSpec("simple_order_2", OptionKind.BOOL, False),
)
DINKSTER_SA_SOLVER = _descriptor(
    "sa_solver",
    "SA-Solver",
    options=_SA_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_SA_SOLVER_PECE = _descriptor(
    "sa_solver_pece",
    "SA-Solver PECE",
    options=_SA_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_CONFIGURED_SA_SOLVER = _descriptor(
    "configured_sa_solver",
    "Configured SA-Solver",
    options=(
        OptionSpec("eta", OptionKind.FLOAT, 1.0, minimum=0.0, maximum=10.0),
        OptionSpec("sde_start_sigma", OptionKind.FLOAT, 1.0, minimum=0.0),
        OptionSpec("sde_end_sigma", OptionKind.FLOAT, 0.0, minimum=0.0),
        OptionSpec("s_noise", OptionKind.FLOAT, 1.0, minimum=0.0, maximum=100.0),
        OptionSpec("predictor_order", OptionKind.INT, 3, minimum=1, maximum=6),
        OptionSpec("corrector_order", OptionKind.INT, 4, minimum=0, maximum=6),
        OptionSpec("use_pece", OptionKind.BOOL, False),
        OptionSpec("simple_order_2", OptionKind.BOOL, False),
    ),
    noise=NoiseKind.GAUSSIAN,
    requires_snr_offset=True,
)
DINKSTER_RES_MULTISTEP = _descriptor("res_multistep", "Res multistep")
DINKSTER_RES_MULTISTEP_CFG_PP = _descriptor(
    "res_multistep_cfg_pp", "Res multistep CFG++", needs_uncond=True
)
DINKSTER_RES_MULTISTEP_ANCESTRAL = _descriptor(
    "res_multistep_ancestral",
    "Res multistep ancestral",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
)
DINKSTER_RES_MULTISTEP_ANCESTRAL_CFG_PP = _descriptor(
    "res_multistep_ancestral_cfg_pp",
    "Res multistep ancestral CFG++",
    options=_ANCESTRAL_OPTIONS,
    noise=NoiseKind.GAUSSIAN,
    needs_uncond=True,
)
DINKSTER_DDIM = _descriptor("ddim", "DDIM", random_inpaint_noise=True)
DINKSTER_UNI_PC = _descriptor("uni_pc", "UniPC", discard_penultimate=True)
DINKSTER_UNI_PC_BH2 = _descriptor("uni_pc_bh2", "UniPC BH2", discard_penultimate=True)


def _res4lyf(
    name: str,
    display_name: str,
    *,
    noise: bool = False,
    aliases: tuple[str, ...] | None = None,
    options: tuple[OptionSpec, ...] = (),
) -> SamplerDescriptor[Any]:
    return _descriptor(
        name,
        display_name,
        namespace="res4lyf",
        aliases=aliases,
        options=options,
        noise=NoiseKind.RES4LYF_GAUSSIAN if noise else NoiseKind.NONE,
    )


RES4LYF_RES_2M = _res4lyf("res_2m", "RES 2M", noise=True)
RES4LYF_RES_3M = _res4lyf("res_3m", "RES 3M", noise=True)
RES4LYF_RES_2S = _res4lyf("res_2s", "RES 2S", noise=True)
RES4LYF_RES_3S = _res4lyf("res_3s", "RES 3S", noise=True)
RES4LYF_RES_5S = _res4lyf("res_5s", "RES 5S", noise=True)
RES4LYF_RES_6S = _res4lyf("res_6s", "RES 6S", noise=True)
RES4LYF_RES_2M_ODE = _res4lyf("res_2m_ode", "RES 2M ODE", aliases=("res_2m_ode", "rk"))
RES4LYF_RES_3M_ODE = _res4lyf("res_3m_ode", "RES 3M ODE")
RES4LYF_RES_2S_ODE = _res4lyf("res_2s_ode", "RES 2S ODE")
RES4LYF_RES_3S_ODE = _res4lyf("res_3s_ode", "RES 3S ODE")
RES4LYF_RES_5S_ODE = _res4lyf("res_5s_ode", "RES 5S ODE")
RES4LYF_RES_6S_ODE = _res4lyf("res_6s_ode", "RES 6S ODE")
RES4LYF_DEIS_2M = _res4lyf("deis_2m", "DEIS 2M", noise=True)
RES4LYF_DEIS_3M = _res4lyf("deis_3m", "DEIS 3M", noise=True)
RES4LYF_DEIS_2M_ODE = _res4lyf("deis_2m_ode", "DEIS 2M ODE")
RES4LYF_DEIS_3M_ODE = _res4lyf("deis_3m_ode", "DEIS 3M ODE")
RES4LYF_RK_BETA = _res4lyf(
    "rk_beta",
    "RK beta",
    noise=True,
    options=(
        OptionSpec(
            "rk_type",
            OptionKind.CHOICE,
            "res_2m",
            choices=(
                "res_2m",
                "res_3m",
                "res_2s",
                "res_3s",
                "res_5s",
                "res_6s",
                "deis_2m",
                "deis_3m",
            ),
            doc="Runge-Kutta method family",
        ),
        OptionSpec(
            "eta",
            OptionKind.FLOAT,
            0.5,
            minimum=0.0,
            maximum=0.99,
            doc="step SDE noise fraction (the reference NaNs at eta >= 1 on EPS models)",
        ),
        OptionSpec(
            "eta_substep",
            OptionKind.FLOAT,
            0.5,
            minimum=0.0,
            maximum=0.99,
            doc="substep SDE noise fraction",
        ),
    ),
)

_BUILTINS = (
    DINKSTER_AR_VIDEO,
    DINKSTER_EULER,
    DINKSTER_EULER_CFG_PP,
    DINKSTER_EULER_ANCESTRAL,
    DINKSTER_EULER_ANCESTRAL_CFG_PP,
    DINKSTER_HEUN,
    DINKSTER_HEUNPP2,
    DINKSTER_EXP_HEUN_2_X0,
    DINKSTER_EXP_HEUN_2_X0_SDE,
    DINKSTER_DPM_2,
    DINKSTER_DPM_2_ANCESTRAL,
    DINKSTER_LMS,
    DINKSTER_DPM_FAST,
    DINKSTER_DPM_ADAPTIVE,
    DINKSTER_DPMPP_2S_ANCESTRAL,
    DINKSTER_DPMPP_2S_ANCESTRAL_CFG_PP,
    DINKSTER_DPMPP_SDE,
    DINKSTER_DPMPP_SDE_GPU,
    DINKSTER_DPMPP_2M,
    DINKSTER_DPMPP_2M_CFG_PP,
    DINKSTER_DPMPP_2M_SDE,
    DINKSTER_DPMPP_2M_SDE_GPU,
    DINKSTER_DPMPP_2M_SDE_HEUN,
    DINKSTER_DPMPP_2M_SDE_HEUN_GPU,
    DINKSTER_DPMPP_3M_SDE,
    DINKSTER_DPMPP_3M_SDE_GPU,
    DINKSTER_DDPM,
    DINKSTER_LCM,
    DINKSTER_IPNDM,
    DINKSTER_IPNDM_V,
    DINKSTER_DEIS,
    DINKSTER_RES_MULTISTEP,
    DINKSTER_RES_MULTISTEP_CFG_PP,
    DINKSTER_RES_MULTISTEP_ANCESTRAL,
    DINKSTER_RES_MULTISTEP_ANCESTRAL_CFG_PP,
    DINKSTER_GRADIENT_ESTIMATION,
    DINKSTER_GRADIENT_ESTIMATION_CFG_PP,
    DINKSTER_ER_SDE,
    DINKSTER_SEEDS_2,
    DINKSTER_SEEDS_3,
    DINKSTER_SA_SOLVER,
    DINKSTER_SA_SOLVER_PECE,
    DINKSTER_CONFIGURED_SA_SOLVER,
    DINKSTER_DDIM,
    DINKSTER_UNI_PC,
    DINKSTER_UNI_PC_BH2,
    RES4LYF_RES_2M,
    RES4LYF_RES_3M,
    RES4LYF_RES_2S,
    RES4LYF_RES_3S,
    RES4LYF_RES_5S,
    RES4LYF_RES_6S,
    RES4LYF_RES_2M_ODE,
    RES4LYF_RES_3M_ODE,
    RES4LYF_RES_2S_ODE,
    RES4LYF_RES_3S_ODE,
    RES4LYF_RES_5S_ODE,
    RES4LYF_RES_6S_ODE,
    RES4LYF_DEIS_2M,
    RES4LYF_DEIS_3M,
    RES4LYF_DEIS_2M_ODE,
    RES4LYF_DEIS_3M_ODE,
    RES4LYF_RK_BETA,
)

_CANONICAL_SAMPLER_FIELDS = tuple(
    catalog_descriptor_snapshot(descriptor) for descriptor in _BUILTINS
)


def builtin_samplers() -> tuple[SamplerDescriptor[Any], ...]:
    return tuple(
        SamplerDescriptor(**catalog_descriptor_rebuild(field_values))
        for field_values in _CANONICAL_SAMPLER_FIELDS
    )


def builtin_sampler_registry() -> Registry[SamplerDescriptor[Any]]:
    registry: Registry[SamplerDescriptor[Any]] = Registry()
    for descriptor in builtin_samplers():
        registry.register(descriptor)
    return registry


def select_builtin_sampler(sampler_id: str, **overrides: object) -> BuiltinSamplerSelection:
    descriptor = builtin_sampler_registry().get(sampler_id)
    if descriptor is None:
        raise ValueError(f"unknown built-in sampler {sampler_id!r}")
    resolved: dict[str, OptionValue] = resolve_options(descriptor.options, overrides)
    return BuiltinSamplerSelection(
        descriptor.id,
        tuple((spec.name, resolved[spec.name]) for spec in descriptor.options),
    )


__all__ = [
    "DINKSTER_AR_VIDEO",
    "DINKSTER_CONFIGURED_SA_SOLVER",
    "DINKSTER_DDIM",
    "DINKSTER_DDPM",
    "DINKSTER_DEIS",
    "DINKSTER_DPM_2",
    "DINKSTER_DPM_2_ANCESTRAL",
    "DINKSTER_DPMPP_2M",
    "DINKSTER_DPMPP_2M_CFG_PP",
    "DINKSTER_DPMPP_2M_SDE",
    "DINKSTER_DPMPP_2M_SDE_GPU",
    "DINKSTER_DPMPP_2M_SDE_HEUN",
    "DINKSTER_DPMPP_2M_SDE_HEUN_GPU",
    "DINKSTER_DPMPP_2S_ANCESTRAL",
    "DINKSTER_DPMPP_2S_ANCESTRAL_CFG_PP",
    "DINKSTER_DPMPP_3M_SDE",
    "DINKSTER_DPMPP_3M_SDE_GPU",
    "DINKSTER_DPMPP_SDE",
    "DINKSTER_DPMPP_SDE_GPU",
    "DINKSTER_DPM_ADAPTIVE",
    "DINKSTER_DPM_FAST",
    "DINKSTER_ER_SDE",
    "DINKSTER_EULER",
    "DINKSTER_EULER_ANCESTRAL",
    "DINKSTER_EULER_ANCESTRAL_CFG_PP",
    "DINKSTER_EULER_CFG_PP",
    "DINKSTER_EXP_HEUN_2_X0",
    "DINKSTER_EXP_HEUN_2_X0_SDE",
    "DINKSTER_GRADIENT_ESTIMATION",
    "DINKSTER_GRADIENT_ESTIMATION_CFG_PP",
    "DINKSTER_HEUN",
    "DINKSTER_HEUNPP2",
    "DINKSTER_IPNDM",
    "DINKSTER_IPNDM_V",
    "DINKSTER_LCM",
    "DINKSTER_LMS",
    "DINKSTER_RES_MULTISTEP",
    "DINKSTER_RES_MULTISTEP_ANCESTRAL",
    "DINKSTER_RES_MULTISTEP_ANCESTRAL_CFG_PP",
    "DINKSTER_RES_MULTISTEP_CFG_PP",
    "DINKSTER_SA_SOLVER",
    "DINKSTER_SA_SOLVER_PECE",
    "DINKSTER_SEEDS_2",
    "DINKSTER_SEEDS_3",
    "DINKSTER_UNI_PC",
    "DINKSTER_UNI_PC_BH2",
    "RES4LYF_DEIS_2M",
    "RES4LYF_DEIS_2M_ODE",
    "RES4LYF_DEIS_3M",
    "RES4LYF_DEIS_3M_ODE",
    "RES4LYF_RES_2M",
    "RES4LYF_RES_2M_ODE",
    "RES4LYF_RES_2S",
    "RES4LYF_RES_2S_ODE",
    "RES4LYF_RES_3M",
    "RES4LYF_RES_3M_ODE",
    "RES4LYF_RES_3S",
    "RES4LYF_RES_3S_ODE",
    "RES4LYF_RES_5S",
    "RES4LYF_RES_5S_ODE",
    "RES4LYF_RES_6S",
    "RES4LYF_RES_6S_ODE",
    "RES4LYF_RK_BETA",
    "builtin_sampler_registry",
    "builtin_samplers",
    "select_builtin_sampler",
]
