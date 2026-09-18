"""Generate sampling-math goldens from the ComfyUI reference checkout.

Runs the REFERENCE implementations (comfy.samplers schedules,
comfy.model_sampling spaces and prediction mixins @ the audited
baseline) and writes tests/goldens/sampling_goldens.json. Dinkster's
native stage-3 ports are pinned against these values - the oracle is
the reference code itself, never a re-derivation.

Usage (needs a torch+scipy interpreter; the workspace root venv is
deliberately torch-free):

    COMFYUI_REFERENCE=/path/to/ComfyUI-at-b78cec87 \
    DINKSTER_CURRENT_COMFY_GIT=/path/to/ComfyUI-git-with-current-pin \
    PYTHONPATH=/path/to/ComfyUI-at-b78cec87 \
        /path/to/torch-venv/bin/python tools/gen_sampling_goldens.py

The default authority remains b78cec87 for every pre-existing entry. Current
node contracts are generated in clean archives of their pinned commits from
DINKSTER_CURRENT_COMFY_GIT and carry that second pin on every entry.
The current-pin repository may be checked out at another revision; the
generator reads the named commit object, never its working tree.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import comfy.options  # noqa: E402
import torch  # noqa: E402

# comfy.model_management probes CUDA at import unless --cpu is parsed.
# CUDA-less non-darwin interpreters (the +cpu tuples) need the flag on
# argv, including in the archived-commit re-execution of this script,
# which starts with a bare argv.
if sys.platform != "darwin" and not torch.cuda.is_available():
    sys.argv.append("--cpu")
comfy.options.enable_args_parsing()
# The reference import reaches torchvision even though sampling goldens
# never use image operators. Some CPU validation wheels omit this schema.
if importlib.util.find_spec("torchvision") is None:
    try:
        _torchvision_stub = torch.library.Library("torchvision", "DEF")
        _torchvision_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        _torchvision_stub.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except RuntimeError:
        pass
import comfy.k_diffusion.sampling as kds  # noqa: E402
import comfy.model_sampling as ms  # noqa: E402
import comfy.samplers as samplers  # noqa: E402
import scipy  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

CURRENT_ER_SDE_MODE = os.environ.get("DINKSTER_GENERATE_CURRENT_ER_SDE") == "1"
CURRENT_SA_SOLVER_MODE = os.environ.get("DINKSTER_GENERATE_CURRENT_SA_SOLVER") == "1"
if CURRENT_ER_SDE_MODE:
    from comfy_extras.nodes_custom_sampler import SamplerER_SDE  # noqa: E402
if CURRENT_SA_SOLVER_MODE:
    from comfy_extras.nodes_custom_sampler import SamplerSASolver  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = platform_golden_path(REPO / "tests" / "goldens" / "sampling_goldens.json", torch.__version__)
CURRENT_SA_SOLVER_OUT = REPO / "tests" / "goldens" / "configured_sa_solver_1d48d9cf.json"
STEPS = (4, 12, 20)
SCHEDULES = tuple(samplers.SCHEDULER_HANDLERS)
BASELINE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
CURRENT_ER_SDE_COMMIT = "76135e557da1ec7dcb270160f01e597565e3e003"
CURRENT_SA_SOLVER_COMMIT = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"


def floats(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def space_goldens(space: object) -> dict[str, object]:
    out: dict[str, object] = {
        "sigma_min": float(space.sigma_min),
        "sigma_max": float(space.sigma_max),
    }
    sigmas = getattr(space, "sigmas", None)
    if sigmas is not None:
        table = floats(sigmas)
        out["table_len"] = len(table)
        out["table_head"] = table[:3]
        out["table_tail"] = table[-3:]
        out["table_mid"] = table[len(table) // 2]
    ts = [0.0, 0.5, 250.7, 500.0, 999.0]
    if isinstance(space, ms.ModelSamplingFlux):
        ts = [0.001, 0.25, 0.5, 0.75, 1.0]
    elif isinstance(space, ms.ModelSamplingContinuousEDM):
        # EDM timesteps are 0.25 * ln(sigma): probe within the domain.
        ts = [-1.5, -0.5, 0.0, 0.5, 1.19]
    out["sigma_of_t"] = {str(t): float(space.sigma(torch.tensor(t))) for t in ts}
    probe_sigmas = [
        float(space.sigma_min) * 1.5,
        (float(space.sigma_min) + float(space.sigma_max)) / 7.0,
        float(space.sigma_max) * 0.5,
    ]
    out["timestep_of_sigma"] = {
        repr(s): float(space.timestep(torch.tensor(s))) for s in probe_sigmas
    }
    out["percent_to_sigma"] = {
        str(p): float(space.percent_to_sigma(p)) for p in (0.0, 0.1, 0.5, 0.9, 1.0)
    }
    return out


def schedule_goldens(space: object) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for name in SCHEDULES:
        per_steps: dict[str, list[float]] = {}
        for steps in STEPS:
            per_steps[str(steps)] = floats(samplers.calculate_sigmas(space, name, steps))
        out[name] = per_steps
    return out


def custom_sigma_query_goldens(spaces: dict[str, Any]) -> dict[str, object]:
    beta: dict[str, list[float]] = {}
    percent_to_sigma: dict[str, dict[str, float]] = {}
    for family_id, space in spaces.items():
        beta[family_id] = floats(samplers.beta_scheduler(space, 13, alpha=2.0, beta=5.0))
        percent_to_sigma[family_id] = {}
        for percent in (0.0, 0.37, 1.0):
            for return_actual_sigma in (False, True):
                output = space.percent_to_sigma(percent)
                if return_actual_sigma:
                    if percent == 0.0:
                        output = space.sigma_max.item()
                    elif percent == 1.0:
                        output = space.sigma_min.item()
                key = f"{percent},{return_actual_sigma}"
                percent_to_sigma[family_id][key] = float(output)

    sd_turbo_space = spaces["dinkster.sd15"]
    start_step = 10 - int(10 * 0.65)
    timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step : start_step + 4]
    sd_turbo = torch.cat([sd_turbo_space.sigma(timesteps), torch.zeros(1, dtype=torch.float32)])
    return {
        "beta": beta,
        "beta_options": {"steps": 13, "alpha": 2.0, "beta": 5.0},
        "percent_to_sigma": percent_to_sigma,
        "sd_turbo": floats(sd_turbo),
        "sd_turbo_options": {"steps": 4, "denoise": 0.65},
    }


def prediction_goldens() -> dict[str, object]:
    x = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float64)
    model_output = torch.tensor([0.125, 0.75, -0.5, 1.5], dtype=torch.float64)
    noise = torch.tensor([1.0, -0.5, 0.0, 2.0], dtype=torch.float64)
    latent = torch.tensor([0.1, 0.2, -0.3, 0.4], dtype=torch.float64)
    sigma = torch.tensor(0.7, dtype=torch.float64)

    out: dict[str, object] = {
        "x": floats(x),
        "model_output": floats(model_output),
        "noise": floats(noise),
        "latent": floats(latent),
        "sigma": float(sigma),
    }
    cases = {
        "EPS": (ms.EPS, 1.0),
        "V_PREDICTION": (ms.V_PREDICTION, 1.0),
        "EDM": (ms.EDM, 0.5),
        "CONST": (ms.CONST, 1.0),
        "IMG_TO_IMG_FLOW": (ms.IMG_TO_IMG_FLOW, 1.0),
        "X0": (ms.X0, 1.0),
    }
    for name, (cls, sigma_data) in cases.items():
        obj = cls()
        obj.sigma_data = sigma_data
        entry: dict[str, object] = {
            "sigma_data": sigma_data,
            "calculate_input": floats(obj.calculate_input(sigma, x)),
            "calculate_denoised": floats(obj.calculate_denoised(sigma, model_output, x)),
            "noise_scaling": floats(obj.noise_scaling(sigma, noise.clone(), latent.clone())),
            "noise_scaling_max_denoise": floats(
                obj.noise_scaling(sigma, noise.clone(), latent.clone(), max_denoise=True)
            ),
            "inverse_noise_scaling": floats(obj.inverse_noise_scaling(sigma, latent.clone())),
        }
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# Solver goldens (stage 3b): run the reference sample_* functions with a
# deterministic mock denoiser and injected counter-based noise, in float64.
# The Dinkster solver tests replicate the exact same denoiser/noise formulas
# over plain floats and pin the final latent. Cases are chosen so the
# reference and the port consume noise draws in lockstep (see the
# draw-count note on each stochastic case).

SOLVER_X0 = [0.5, -1.0, 2.0, 0.25]
SOLVER_SIGMAS_EPS = [14.614642, 6.0, 2.5, 1.0, 0.4, 0.0291675, 0.0]
SOLVER_SIGMAS_FLOW = [0.98, 0.75, 0.5, 0.25, 0.1, 0.0]
SOLVER_SIGMAS_FLOW_SNR = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]


def solver_mock_denoised(x: torch.Tensor, s: float) -> torch.Tensor:
    """The shared mock denoiser: nonlinear in x so coefficient errors
    cannot cancel, pure float64 elementwise so the torch-free twin in
    tests/test_inference_solvers.py computes bit-comparable values."""
    return x * (1.0 / (1.0 + s)) + (x * x) * (0.05 / (1.0 + s))


def solver_mock_noise(k: int, dims: int) -> list[float]:
    """Deterministic pseudo-noise for draw index ``k``."""
    return [math.sin(1.3 * k + 0.7 * j + 0.1) for j in range(dims)]


def solver_mock_uncond(x: torch.Tensor, s: float) -> torch.Tensor:
    """The mock UNCONDITIONED prediction for the CFG++ cases: a second
    nonlinear formula, deliberately distinct from solver_mock_denoised
    so the combined/uncond channels can never be confused."""
    return x * (0.8 / (1.2 + s)) - (x * x) * (0.03 / (1.0 + s))


class _MockModel:
    """The wrapped-model stand-in the reference solvers probe: callable
    denoiser plus the model_sampling attribute chains sample_* actually
    reads (inner_model.inner_model.model_sampling for the RF isinstance
    dispatch, inner_model.model_patcher.get_model_object for noise_scale
    and lcm's noise_scaling)."""

    def __init__(self, model_sampling: object) -> None:
        self.inner_model = SimpleNamespace(
            inner_model=SimpleNamespace(model_sampling=model_sampling),
            model_patcher=SimpleNamespace(get_model_object=lambda name: model_sampling),
        )

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return solver_mock_denoised(x, float(sigma.reshape(-1)[0]))


class _MockCfgModel(_MockModel):
    """The CFG++ solvers capture the uncond through the post-CFG hook
    chain, which the real path runs inside sampling_function
    (comfy/samplers.py @ 947c2749) - a layer this mock replaces. So the
    mock plays that layer's part: combined output from
    solver_mock_denoised, uncond from solver_mock_uncond, and every
    sampler_post_cfg_function applied over the same args keys the
    reference passes (the cfg_pp hooks read only uncond_denoised and
    return args["denoised"])."""

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        s = float(sigma.reshape(-1)[0])
        denoised = solver_mock_denoised(x, s)
        uncond_denoised = solver_mock_uncond(x, s)
        model_options = kwargs.get("model_options") or {}
        assert isinstance(model_options, dict)
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": denoised,
                "uncond_denoised": uncond_denoised,
                "cond_denoised": denoised,
                "input": x,
                "sigma": sigma,
                "model_options": model_options,
            }
            denoised = fn(args)
        return denoised


def solver_goldens() -> dict[str, object]:
    # SA-Solver reads percent_to_sigma from model_sampling to construct
    # its default stochastic interval. Use the complete reference model
    # sampling classes for every mock space rather than prediction-only
    # mixins; their prediction math is identical for all existing cases.
    class _DiscreteEps(ms.ModelSamplingDiscrete, ms.EPS):
        pass

    class _FlowConst(ms.ModelSamplingDiscreteFlow, ms.CONST):
        pass

    eps_ms = ms.EPS()
    eps_ms.percent_to_sigma = _DiscreteEps().percent_to_sigma
    const_ms = ms.CONST()
    const_ms.percent_to_sigma = _FlowConst().percent_to_sigma
    const_scaled_ms = ms.CONST()
    const_scaled_ms.noise_scale = 1.25
    const_scaled_ms.percent_to_sigma = _FlowConst().percent_to_sigma

    # flow_snr: a flow schedule starting at exactly sigma 1.0 - the case
    # the SDE solvers clamp via offset_first_sigma_for_snr (percent_to_
    # sigma probing needs a full flow model_sampling, not the bare CONST
    # mixin) and dpmpp_2s_ancestral special-cases (sigmas[i] == 1.0).
    flow_snr_ms = _FlowConst()

    spaces = {
        "eps": (eps_ms, SOLVER_SIGMAS_EPS),
        "flow": (const_ms, SOLVER_SIGMAS_FLOW),
        "flow_scaled": (const_scaled_ms, SOLVER_SIGMAS_FLOW),
        "flow_snr": (flow_snr_ms, SOLVER_SIGMAS_FLOW_SNR),
    }

    def sample_dpm_fast_wrapper(
        model: object, x: torch.Tensor, sigmas: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        sigma_min = sigmas[-2] if sigmas[-1] == 0 else sigmas[-1]
        return kds.sample_dpm_fast(model, x, sigma_min, sigmas[0], len(sigmas) - 1, **kwargs)

    def sample_dpm_adaptive_wrapper(
        model: object, x: torch.Tensor, sigmas: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        sigma_min = sigmas[-2] if sigmas[-1] == 0 else sigmas[-1]
        return kds.sample_dpm_adaptive(model, x, sigma_min, sigmas[0], **kwargs)

    # (case, sample fn, space key, kwargs, noise mode)
    # Noise modes: "none" - deterministic, no noise path runs;
    # "injected" - the counter-based sampler goes in as noise_sampler;
    # "randn_patch" - euler/heun/dpm_2 churn noise hardcodes
    # torch.randn_like (no noise_sampler parameter exists), so the
    # counter-based draw is patched in its place, keeping the
    # surrounding reference math untouched.
    # Stochastic cases keep eta > 0 (or all-zero noise coefficients) so
    # the port's skip-zero-coefficient deviation cannot desynchronize
    # the shared draw counter.
    cases: list[tuple[str, object, str, dict[str, object], str]] = [
        ("euler_eps", kds.sample_euler, "eps", {}, "none"),
        (
            "euler_churn_eps",
            kds.sample_euler,
            "eps",
            {"s_churn": 0.5, "s_tmin": 0.05, "s_tmax": 8.0, "s_noise": 0.9},
            "randn_patch",
        ),
        ("heun_eps", kds.sample_heun, "eps", {}, "none"),
        ("heun_churn_eps", kds.sample_heun, "eps", {"s_churn": 0.4}, "randn_patch"),
        ("heunpp2_eps", kds.sample_heunpp2, "eps", {}, "randn_patch"),
        ("heunpp2_churn_eps", kds.sample_heunpp2, "eps", {"s_churn": 0.4}, "randn_patch"),
        ("dpm_2_eps", kds.sample_dpm_2, "eps", {}, "none"),
        ("dpm_2_churn_eps", kds.sample_dpm_2, "eps", {"s_churn": 0.3}, "randn_patch"),
        ("lms_eps", kds.sample_lms, "eps", {}, "none"),
        ("lms_eps_order2", kds.sample_lms, "eps", {"order": 2}, "none"),
        ("lms_eps_order6", kds.sample_lms, "eps", {"order": 6}, "none"),
        ("dpm_fast_eps", sample_dpm_fast_wrapper, "eps", {}, "injected"),
        ("dpm_fast_flow", sample_dpm_fast_wrapper, "flow", {}, "injected"),
        ("dpm_adaptive_eps", sample_dpm_adaptive_wrapper, "eps", {}, "injected"),
        (
            "dpm_adaptive_eps_eta2",
            sample_dpm_adaptive_wrapper,
            "eps",
            {"eta": 2.0, "atol": 2.1},
            "injected",
        ),
        (
            "dpm_adaptive_flow_order2_options",
            sample_dpm_adaptive_wrapper,
            "flow",
            {
                "order": 2,
                "rtol": 0.03,
                "atol": 0.005,
                "h_init": 0.08,
                "pcoeff": 0.1,
                "icoeff": 0.9,
                "dcoeff": 0.05,
                "accept_safety": 0.85,
                "eta": 0.4,
                "s_noise": 0.7,
            },
            "injected",
        ),
        ("euler_ancestral_eps", kds.sample_euler_ancestral, "eps", {}, "injected"),
        (
            "euler_ancestral_eps_eta0.4",
            kds.sample_euler_ancestral,
            "eps",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        (
            "euler_ancestral_eps_eta0",
            kds.sample_euler_ancestral,
            "eps",
            {"eta": 0.0},
            "injected",
        ),
        ("euler_ancestral_flow", kds.sample_euler_ancestral, "flow", {}, "injected"),
        (
            "euler_ancestral_flow_scaled",
            kds.sample_euler_ancestral,
            "flow_scaled",
            {},
            "injected",
        ),
        ("dpm_2_ancestral_eps", kds.sample_dpm_2_ancestral, "eps", {}, "injected"),
        ("dpm_2_ancestral_flow", kds.sample_dpm_2_ancestral, "flow", {}, "injected"),
        ("dpmpp_2s_ancestral_eps", kds.sample_dpmpp_2s_ancestral, "eps", {}, "injected"),
        ("dpmpp_2s_ancestral_flow", kds.sample_dpmpp_2s_ancestral, "flow", {}, "injected"),
        ("dpmpp_sde_eps", kds.sample_dpmpp_sde, "eps", {}, "injected"),
        ("dpmpp_sde_eps_r0.33", kds.sample_dpmpp_sde, "eps", {"r": 1.0 / 3.0}, "injected"),
        ("dpmpp_sde_eps_r2", kds.sample_dpmpp_sde, "eps", {"r": 2.0}, "injected"),
        ("dpmpp_sde_flow", kds.sample_dpmpp_sde, "flow", {}, "injected"),
        ("dpmpp_2m_eps", kds.sample_dpmpp_2m, "eps", {}, "none"),
        ("dpmpp_2m_sde_eps", kds.sample_dpmpp_2m_sde, "eps", {}, "injected"),
        (
            "dpmpp_2m_sde_eps_heun",
            kds.sample_dpmpp_2m_sde,
            "eps",
            {"solver_type": "heun"},
            "injected",
        ),
        (
            "dpmpp_2m_sde_eps_eta0.5",
            kds.sample_dpmpp_2m_sde,
            "eps",
            {"eta": 0.5},
            "injected",
        ),
        ("dpmpp_2m_sde_eps_eta0", kds.sample_dpmpp_2m_sde, "eps", {"eta": 0.0}, "injected"),
        ("dpmpp_2m_sde_flow", kds.sample_dpmpp_2m_sde, "flow", {}, "injected"),
        ("dpmpp_3m_sde_eps", kds.sample_dpmpp_3m_sde, "eps", {}, "injected"),
        ("dpmpp_3m_sde_flow", kds.sample_dpmpp_3m_sde, "flow", {}, "injected"),
        ("ddpm_eps", kds.sample_ddpm, "eps", {}, "injected"),
        ("lcm_eps", kds.sample_lcm, "eps", {}, "injected"),
        ("lcm_eps_snoise0.7", kds.sample_lcm, "eps", {"s_noise": 0.7}, "injected"),
        ("lcm_flow_scaled", kds.sample_lcm, "flow_scaled", {}, "injected"),
        (
            "lcm_eps_noise_range",
            kds.sample_lcm,
            "eps",
            {"s_noise": 0.5, "s_noise_end": 1.3},
            "injected",
        ),
        ("lcm_eps_noise_clip", kds.sample_lcm, "eps", {"noise_clip_std": 0.7}, "injected"),
        ("ipndm_eps", kds.sample_ipndm, "eps", {}, "none"),
        ("ipndm_eps_order2", kds.sample_ipndm, "eps", {"max_order": 2}, "none"),
        ("ipndm_v_eps", kds.sample_ipndm_v, "eps", {}, "none"),
        ("ipndm_v_flow", kds.sample_ipndm_v, "flow", {}, "none"),
        ("deis_eps", kds.sample_deis, "eps", {}, "none"),
        ("deis_flow", kds.sample_deis, "flow", {}, "none"),
        ("deis_eps_order2", kds.sample_deis, "eps", {"max_order": 2}, "none"),
        ("gradient_estimation_eps", kds.sample_gradient_estimation, "eps", {}, "none"),
        (
            "gradient_estimation_flow_gamma1.5",
            kds.sample_gradient_estimation,
            "flow",
            {"ge_gamma": 1.5},
            "none",
        ),
        (
            "gradient_estimation_cfg_pp_eps",
            kds.sample_gradient_estimation_cfg_pp,
            "eps",
            {},
            "none",
        ),
        ("er_sde_eps", kds.sample_er_sde, "eps", {}, "injected"),
        ("er_sde_flow", kds.sample_er_sde, "flow", {}, "injected"),
        ("er_sde_flow_scaled", kds.sample_er_sde, "flow_scaled", {}, "injected"),
        (
            "er_sde_eps_stage2_snoise0.6",
            kds.sample_er_sde,
            "eps",
            {"max_stage": 2, "s_noise": 0.6},
            "injected",
        ),
        ("exp_heun_2_x0_eps_phi2", kds.sample_exp_heun_2_x0, "eps", {}, "none"),
        (
            "exp_heun_2_x0_flow_phi1",
            kds.sample_exp_heun_2_x0,
            "flow",
            {"solver_type": "phi_1"},
            "none",
        ),
        ("exp_heun_2_x0_sde_eps", kds.sample_exp_heun_2_x0_sde, "eps", {}, "injected"),
        (
            "exp_heun_2_x0_sde_flow_eta0.4_phi1",
            kds.sample_exp_heun_2_x0_sde,
            "flow",
            {"eta": 0.4, "solver_type": "phi_1"},
            "injected",
        ),
        ("seeds_2_eps_phi1", kds.sample_seeds_2, "eps", {}, "injected"),
        (
            "seeds_2_eps_phi2_eta0",
            kds.sample_seeds_2,
            "eps",
            {"eta": 0.0, "solver_type": "phi_2"},
            "none",
        ),
        (
            "seeds_2_flow_phi2_eta0.4",
            kds.sample_seeds_2,
            "flow",
            {"eta": 0.4, "solver_type": "phi_2"},
            "injected",
        ),
        ("seeds_3_eps", kds.sample_seeds_3, "eps", {}, "injected"),
        ("seeds_3_flow_eta0.4", kds.sample_seeds_3, "flow", {"eta": 0.4}, "injected"),
        ("sa_solver_eps", kds.sample_sa_solver, "eps", {}, "injected"),
        ("sa_solver_flow", kds.sample_sa_solver, "flow", {}, "injected"),
        (
            "sa_solver_eps_options",
            kds.sample_sa_solver,
            "eps",
            {"s_noise": 0.6, "predictor_order": 2, "corrector_order": 3},
            "injected",
        ),
        (
            "sa_solver_pece_eps_simple",
            kds.sample_sa_solver_pece,
            "eps",
            {"predictor_order": 2, "corrector_order": 2, "simple_order_2": True},
            "injected",
        ),
        ("uni_pc_eps", samplers.uni_pc.sample_unipc, "eps", {}, "none"),
        ("uni_pc_flow", samplers.uni_pc.sample_unipc, "flow", {}, "none"),
        ("uni_pc_bh2_eps", samplers.uni_pc.sample_unipc_bh2, "eps", {}, "none"),
        ("uni_pc_bh2_flow", samplers.uni_pc.sample_unipc_bh2, "flow", {}, "none"),
        # gamma saturation: s_churn large enough that gamma caps at
        # sqrt(2) - 1 for every in-range step.
        (
            "euler_churn_cap_eps",
            kds.sample_euler,
            "eps",
            {"s_churn": 100.0, "s_tmin": 0.05, "s_tmax": 20.0},
            "randn_patch",
        ),
        # sigmas[i] == 1.0 special case in the 2S ancestral RF variant.
        (
            "dpmpp_2s_ancestral_flow1",
            kds.sample_dpmpp_2s_ancestral,
            "flow_snr",
            {},
            "injected",
        ),
        # flow schedules starting at exactly 1.0: the reference offsets
        # the first sigma internally (offset_first_sigma_for_snr); the
        # port test applies the ported helper before solving.
        ("dpmpp_sde_flow1", kds.sample_dpmpp_sde, "flow_snr", {}, "injected"),
        ("dpmpp_2m_sde_flow1", kds.sample_dpmpp_2m_sde, "flow_snr", {}, "injected"),
        ("dpmpp_3m_sde_flow1", kds.sample_dpmpp_3m_sde, "flow_snr", {}, "injected"),
        # heun solver_type on the flow/alpha_t coefficient path.
        (
            "dpmpp_2m_sde_flow_heun",
            kds.sample_dpmpp_2m_sde,
            "flow",
            {"solver_type": "heun"},
            "injected",
        ),
        # eta=0 for the remaining SDE families (reference guards its
        # draws with eta > 0, so counts stay in sync).
        ("dpmpp_sde_eps_eta0", kds.sample_dpmpp_sde, "eps", {"eta": 0.0}, "injected"),
        ("dpmpp_3m_sde_eps_eta0", kds.sample_dpmpp_3m_sde, "eps", {"eta": 0.0}, "injected"),
        # RF ancestral variants with nondefault eta/s_noise on the
        # noise_scale-scaled flow space.
        (
            "dpm_2_ancestral_flow_scaled_eta0.4",
            kds.sample_dpm_2_ancestral,
            "flow_scaled",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        (
            "dpmpp_2s_ancestral_flow_scaled_eta0.4",
            kds.sample_dpmpp_2s_ancestral,
            "flow_scaled",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        # -- CFG++ solver family (post-CFG uncond capture) --------------
        # These run against _MockCfgModel, which supplies a distinct
        # uncond formula through the reference post_cfg_function hook
        # chain the solvers install. euler_cfg_pp and dpmpp_2m_cfg_pp
        # are deterministic; the ancestral pair draws through the shared
        # counter. euler[_ancestral]_cfg_pp branches on CONST inside
        # sigma_to_half_log_snr, so both eps and flow spaces matter;
        # flow_scaled exercises the noise_scale multiplier on s_noise.
        ("euler_cfg_pp_eps", kds.sample_euler_cfg_pp, "eps", {}, "none"),
        ("euler_cfg_pp_flow", kds.sample_euler_cfg_pp, "flow", {}, "none"),
        (
            "euler_ancestral_cfg_pp_eps",
            kds.sample_euler_ancestral_cfg_pp,
            "eps",
            {},
            "injected",
        ),
        (
            "euler_ancestral_cfg_pp_eps_eta0.4",
            kds.sample_euler_ancestral_cfg_pp,
            "eps",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        (
            "euler_ancestral_cfg_pp_flow",
            kds.sample_euler_ancestral_cfg_pp,
            "flow",
            {},
            "injected",
        ),
        (
            "euler_ancestral_cfg_pp_flow_scaled",
            kds.sample_euler_ancestral_cfg_pp,
            "flow_scaled",
            {},
            "injected",
        ),
        (
            "dpmpp_2s_ancestral_cfg_pp_eps",
            kds.sample_dpmpp_2s_ancestral_cfg_pp,
            "eps",
            {},
            "injected",
        ),
        (
            "dpmpp_2s_ancestral_cfg_pp_flow_scaled",
            kds.sample_dpmpp_2s_ancestral_cfg_pp,
            "flow_scaled",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        ("dpmpp_2m_cfg_pp_eps", kds.sample_dpmpp_2m_cfg_pp, "eps", {}, "none"),
        ("dpmpp_2m_cfg_pp_flow", kds.sample_dpmpp_2m_cfg_pp, "flow", {}, "none"),
        # -- res_multistep family ---------------------------------------
        # The deterministic wrappers still construct and call the
        # reference default noise sampler, but sigma_up is zero at eta
        # 0 so those draws are output-inert; "none" leaves that exact
        # reference behavior in place. All schedules have enough
        # intervals to execute the second-order history branch.
        ("res_multistep_eps", kds.sample_res_multistep, "eps", {}, "none"),
        ("res_multistep_flow", kds.sample_res_multistep, "flow", {}, "none"),
        (
            "res_multistep_cfg_pp_eps",
            kds.sample_res_multistep_cfg_pp,
            "eps",
            {},
            "none",
        ),
        (
            "res_multistep_cfg_pp_flow",
            kds.sample_res_multistep_cfg_pp,
            "flow",
            {},
            "none",
        ),
        (
            "res_multistep_ancestral_eps",
            kds.sample_res_multistep_ancestral,
            "eps",
            {},
            "injected",
        ),
        (
            "res_multistep_ancestral_eps_eta0.4",
            kds.sample_res_multistep_ancestral,
            "eps",
            {"eta": 0.4, "s_noise": 0.8},
            "injected",
        ),
        (
            "res_multistep_ancestral_flow",
            kds.sample_res_multistep_ancestral,
            "flow",
            {},
            "injected",
        ),
        (
            "res_multistep_ancestral_flow_scaled",
            kds.sample_res_multistep_ancestral,
            "flow_scaled",
            {},
            "injected",
        ),
        (
            "res_multistep_ancestral_cfg_pp_eps",
            kds.sample_res_multistep_ancestral_cfg_pp,
            "eps",
            {},
            "injected",
        ),
        (
            "res_multistep_ancestral_cfg_pp_flow_scaled",
            kds.sample_res_multistep_ancestral_cfg_pp,
            "flow_scaled",
            {},
            "injected",
        ),
    ]

    dims = len(SOLVER_X0)
    out: dict[str, object] = {
        "x0": list(SOLVER_X0),
        "sigmas": {
            "eps": list(SOLVER_SIGMAS_EPS),
            "flow": list(SOLVER_SIGMAS_FLOW),
            "flow_scaled": list(SOLVER_SIGMAS_FLOW),
            "flow_snr": list(SOLVER_SIGMAS_FLOW_SNR),
        },
        "noise_scale": {
            "eps": 1.0,
            "flow": 1.0,
            "flow_scaled": 1.25,
            "flow_snr": 1.0,
        },
    }
    results: dict[str, object] = {}
    for name, fn, space_key, kwargs, noise_mode in cases:
        model_sampling, sigma_list = spaces[space_key]
        model_cls = _MockCfgModel if "cfg_pp" in name else _MockModel
        model = model_cls(model_sampling)
        # UniPC's reference builds its tiny coefficient systems in the
        # ambient default dtype, then tensordots them with model values.
        # Its production contract is float32; feeding a synthetic float64
        # latent raises a mixed-dtype error before any solver step.
        dtype = torch.float32 if name.startswith("uni_pc") else torch.float64
        x = torch.tensor(SOLVER_X0, dtype=dtype)
        sigmas = torch.tensor(sigma_list, dtype=dtype)
        counter = {"k": 0}

        def draw(
            *_args: object,
            _counter: dict[str, int] = counter,
            _dtype: torch.dtype = dtype,
            **_kwargs: object,
        ) -> torch.Tensor:
            values = solver_mock_noise(_counter["k"], dims)
            _counter["k"] += 1
            return torch.tensor(values, dtype=_dtype)

        saved_randn_like = torch.randn_like
        if noise_mode == "randn_patch":
            # euler/heun/dpm_2 have no noise_sampler parameter; their
            # churn noise hardcodes torch.randn_like (unseedable). Route
            # it through the same shared counter the port's injected
            # NoiseSampler uses, keeping the surrounding reference math
            # untouched.
            torch.randn_like = draw
        try:
            final = fn(
                model,
                x,
                sigmas,
                extra_args={},
                disable=True,
                **({"noise_sampler": draw} if noise_mode == "injected" else {}),
                **kwargs,
            )
        finally:
            torch.randn_like = saved_randn_like
        results[name] = {
            "space": space_key,
            "options": {k: v for k, v in kwargs.items()},
            "noise_mode": noise_mode,
            "noise_draws": counter["k"],
            "final": floats(final),
        }
    out["cases"] = results
    return out


def current_er_sde_goldens() -> dict[str, object]:
    """Execute SamplerER_SDE and its bound sample_er_sde at the current pin."""
    if not CURRENT_ER_SDE_MODE:
        raise RuntimeError("current ER-SDE generation must run in its isolated reference process")

    class _DiscreteEps(ms.ModelSamplingDiscrete, ms.EPS):
        pass

    model_sampling = ms.EPS()
    model_sampling.percent_to_sigma = _DiscreteEps().percent_to_sigma
    model = _MockModel(model_sampling)
    results: dict[str, object] = {}
    solver_names = {
        "ER-SDE": "er_sde",
        "Reverse-time SDE": "reverse_time_sde",
        "ODE": "ode",
    }
    cases = [
        (solver_type, solver_name, eta, max_stage)
        for solver_type, solver_name in solver_names.items()
        for eta in (0.0, 0.5, 1.0, 2.0)
        for max_stage in (1, 2, 3)
    ]
    cases.append(("ER-SDE", "er_sde", 11.0, 3))
    for solver_type, solver_name, eta, max_stage in cases:
        sampler = SamplerER_SDE.execute(solver_type, max_stage, eta, 1.0)[0]
        if sampler.sampler_function is not kds.sample_er_sde:
            raise RuntimeError("SamplerER_SDE did not bind sample_er_sde")
        x = torch.tensor(SOLVER_X0, dtype=torch.float64)
        sigmas = torch.tensor(SOLVER_SIGMAS_EPS, dtype=torch.float64)
        counter = {"k": 0}

        def draw(
            *_args: object,
            _counter: dict[str, int] = counter,
            **_kwargs: object,
        ) -> torch.Tensor:
            values = solver_mock_noise(_counter["k"], len(SOLVER_X0))
            _counter["k"] += 1
            return torch.tensor(values, dtype=torch.float64)

        final = sampler.sampler_function(
            model,
            x,
            sigmas,
            extra_args={},
            disable=True,
            noise_sampler=draw,
            **sampler.extra_options,
        )
        name = f"er_sde_current_{solver_name}_eta{eta:g}_stage{max_stage}"
        results[name] = {
            "reference_commit": CURRENT_ER_SDE_COMMIT,
            "space": "eps",
            "options": {
                "solver_type": solver_type,
                "max_stage": max_stage,
                "eta": eta,
                "s_noise": 1.0,
            },
            "noise_mode": "injected",
            "noise_draws": counter["k"],
            "final": floats(final),
        }
    return results


def generate_current_er_sde_from_commit(comfy_git: Path) -> dict[str, object]:
    """Run the current matrix from the immutable commit object in a clean archive."""
    subprocess.run(
        ["git", "-C", str(comfy_git), "cat-file", "-e", f"{CURRENT_ER_SDE_COMMIT}^{{commit}}"],
        check=True,
    )
    archive = subprocess.run(
        ["git", "-C", str(comfy_git), "archive", CURRENT_ER_SDE_COMMIT],
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="dinkster-er-sde-reference-") as directory:
        reference = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(reference, filter="data")
        env = os.environ.copy()
        env["DINKSTER_GENERATE_CURRENT_ER_SDE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(reference),
                *(entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry),
            )
        )
        generated = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    return json.loads(generated.stdout)


def current_sa_solver_golden() -> dict[str, object]:
    """Execute SamplerSASolver and its configured solver at the current pin."""
    if not CURRENT_SA_SOLVER_MODE:
        raise RuntimeError("current SA-Solver generation must run in its isolated process")

    class _DiscreteEps(ms.ModelSamplingDiscrete, ms.EPS):
        pass

    model_sampling = _DiscreteEps()

    class _NodeModel:
        @staticmethod
        def get_model_object(name: str) -> object:
            if name != "model_sampling":
                raise KeyError(name)
            return model_sampling

    node_options = {
        "eta": 0.65,
        "sde_start_percent": 0.17,
        "sde_end_percent": 0.73,
        "s_noise": 0.6,
        "predictor_order": 5,
        "corrector_order": 2,
        "use_pece": True,
        "simple_order_2": True,
    }
    sampler = SamplerSASolver.execute(_NodeModel(), **node_options)[0]
    if sampler.sampler_function is not kds.sample_sa_solver:
        raise RuntimeError("SamplerSASolver did not bind sample_sa_solver")
    x = torch.tensor(SOLVER_X0, dtype=torch.float64)
    sigmas = torch.tensor(SOLVER_SIGMAS_EPS, dtype=torch.float64)
    model = _MockModel(model_sampling)
    counter = {"k": 0}

    def draw(*_args: object, **_kwargs: object) -> torch.Tensor:
        values = solver_mock_noise(counter["k"], len(SOLVER_X0))
        counter["k"] += 1
        return torch.tensor(values, dtype=torch.float64)

    final = sampler.sampler_function(
        model,
        x,
        sigmas,
        extra_args={},
        disable=True,
        noise_sampler=draw,
        **sampler.extra_options,
    )
    return {
        "reference_commit": CURRENT_SA_SOLVER_COMMIT,
        "space": "eps",
        "reference_node_options": node_options,
        "options": {
            "eta": node_options["eta"],
            "sde_start_sigma": float(
                model_sampling.percent_to_sigma(node_options["sde_start_percent"])
            ),
            "sde_end_sigma": float(
                model_sampling.percent_to_sigma(node_options["sde_end_percent"])
            ),
            "s_noise": node_options["s_noise"],
            "predictor_order": node_options["predictor_order"],
            "corrector_order": node_options["corrector_order"],
            "use_pece": node_options["use_pece"],
            "simple_order_2": node_options["simple_order_2"],
        },
        "noise_mode": "injected",
        "noise_draws": counter["k"],
        "final": floats(final),
    }


def generate_current_sa_solver_from_commit(comfy_git: Path) -> dict[str, object]:
    """Run the configured SA-Solver case from its immutable commit object."""
    subprocess.run(
        ["git", "-C", str(comfy_git), "cat-file", "-e", f"{CURRENT_SA_SOLVER_COMMIT}^{{commit}}"],
        check=True,
    )
    archive = subprocess.run(
        ["git", "-C", str(comfy_git), "archive", CURRENT_SA_SOLVER_COMMIT],
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="dinkster-sa-solver-reference-") as directory:
        reference = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(reference, filter="data")
        env = os.environ.copy()
        env["DINKSTER_GENERATE_CURRENT_SA_SOLVER"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(reference),
                *(entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry),
            )
        )
        generated = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    return json.loads(generated.stdout)


def beta_ppf_goldens() -> dict[str, float]:
    points = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.999)
    out: dict[str, float] = {}
    for a, b in (
        (0.4, 0.4),
        (0.5, 0.5),
        (0.6, 0.6),
        (1.0, 1.0),
        (2.0, 5.0),
        (0.5, 3.0),
    ):
        for p in points:
            out[f"{a},{b},{p}"] = float(scipy.stats.beta.ppf(p, a, b))
    return out


def main() -> None:
    if CURRENT_ER_SDE_MODE:
        print(json.dumps(current_er_sde_goldens()))
        return
    if CURRENT_SA_SOLVER_MODE:
        print(json.dumps(current_sa_solver_golden()))
        return

    comfy_root = Path(os.environ.get("COMFYUI_REFERENCE", REPO.parent / "ComfyUI")).resolve()
    baseline = subprocess.run(
        ["git", "-C", str(comfy_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(comfy_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"refusing to generate goldens: {comfy_root} has uncommitted "
            f"changes, so the recorded commit would not describe the code "
            f"that actually ran:\n{dirty}"
        )
    if baseline != BASELINE_COMMIT:
        raise SystemExit(
            f"refusing to generate baseline goldens: expected {BASELINE_COMMIT}, got {baseline}"
        )

    current_comfy_git = Path(
        os.environ.get("DINKSTER_CURRENT_COMFY_GIT", str(comfy_root))
    ).resolve()
    current_er_sde_cases = generate_current_er_sde_from_commit(current_comfy_git)
    current_sa_solver_case = generate_current_sa_solver_from_commit(current_comfy_git)

    discrete = ms.ModelSamplingDiscrete()  # SD15/SDXL defaults
    flow_sd3 = ms.ModelSamplingDiscreteFlow()
    flow_sd3.set_parameters(shift=3.0)
    flux = ms.ModelSamplingFlux()  # shift=1.15
    flow_schnell = ms.ModelSamplingDiscreteFlow()
    flow_schnell.set_parameters(shift=1.0, multiplier=1.0)
    edm = ms.ModelSamplingContinuousEDM()  # 0.002..120
    custom_query_spaces = {
        "dinkster.sd15": discrete,
        "dinkster.sdxl": discrete,
        "dinkster.sdxl_refiner": discrete,
        "dinkster.flux_dev": flux,
        "dinkster.flux_schnell": flow_schnell,
        "dinkster.sdxl_continuous_edm": edm,
    }

    goldens = {
        "_meta": {
            "reference_commit": baseline,
            "reference_overrides": {
                "solvers.cases.er_sde_current_*": CURRENT_ER_SDE_COMMIT,
            },
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "generator": "tools/gen_sampling_goldens.py",
            **tuple_provenance(torch.__version__),
        },
        "spaces": {
            "discrete_sd": space_goldens(discrete),
            "flow_shift3": space_goldens(flow_sd3),
            "flux_shift1.15": space_goldens(flux),
            "continuous_edm": space_goldens(edm),
        },
        "schedules": {
            "discrete_sd": schedule_goldens(discrete),
            "flow_shift3": schedule_goldens(flow_sd3),
        },
        "custom_sigma_queries": custom_sigma_query_goldens(custom_query_spaces),
        "predictions": prediction_goldens(),
        "solvers": solver_goldens(),
        "beta_ppf": beta_ppf_goldens(),
        "chroma_radiance_beta": floats(
            samplers.beta_scheduler(flow_schnell, 30, alpha=0.4, beta=0.4)
        ),
        "shift_fns": {
            "time_snr_shift": {
                f"{a},{t}": ms.time_snr_shift(a, t) for a in (1.0, 3.0) for t in (0.1, 0.5, 0.9)
            },
            "flux_time_shift": {
                f"{mu},{t}": ms.flux_time_shift(mu, 1.0, t)
                for mu in (0.5, 1.15)
                for t in (0.1, 0.5, 0.9)
            },
        },
    }

    solver_cases = goldens["solvers"]["cases"]
    assert isinstance(solver_cases, dict)
    old_default = solver_cases["er_sde_eps"]
    current_default = current_er_sde_cases["er_sde_current_er_sde_eta1_stage3"]
    assert isinstance(old_default, dict) and isinstance(current_default, dict)
    for field in ("space", "noise_mode", "noise_draws", "final"):
        if old_default[field] != current_default[field]:
            raise SystemExit(f"current ER-SDE eta=1 default moved existing field {field!r}")
    solver_cases.update(current_er_sde_cases)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(goldens, indent=1) + "\n")
    CURRENT_SA_SOLVER_OUT.write_text(json.dumps(current_sa_solver_case, indent=1) + "\n")
    print(f"wrote {OUT} (reference {baseline[:8]}, {len(SCHEDULES)} schedules)")


if __name__ == "__main__":
    sys.exit(main())
