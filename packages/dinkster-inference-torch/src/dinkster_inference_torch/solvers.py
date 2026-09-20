"""Torch reference-kernel solver bindings and overrides.

The portable solver bindings use Python floats. Executed tensor math can be
bit-visible, however: ComfyUI keeps
Euler, DPM++ 2M, DPM++ 2M SDE, DPM++ SDE, ER-SDE, and res_multistep arithmetic on the latent
device as float32 tensors, while UniPC keeps its coefficients, linear solves,
and updates there. This module rebinds those executed seams without changing
the portable solver contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from dinkster_inference import (
    CallableEvidence,
    Denoiser,
    NoiseSampler,
    OptionValue,
    Registry,
    SamplerDescriptor,
    SamplerInfo,
    SolverFn,
    SolverStateEvent,
    StepCallback,
    StepEvent,
    builtin_samplers,
    is_flow_parameterization,
)
from dinkster_inference.sampling import StepBeginSolverFn

from . import _portable_solvers


def _emit(
    on_step: StepCallback | None,
    info: SamplerInfo,
    step: int,
    total: int,
    sigma: float,
    current: torch.Tensor,
    denoised: torch.Tensor,
) -> None:
    if info.on_state is not None:
        info.on_state(
            SolverStateEvent[object](
                step=step,
                total=total,
                sigma=sigma,
                phase="pre_update",
                current=current,
                denoised=denoised,
            )
        )
    if on_step is not None:
        on_step(StepEvent(step=step, total=total, sigma=sigma))


def euler(
    *,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = math.inf,
    s_noise: float = 1.0,
) -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_euler`` with device-float32 scalar kernels."""

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        count = len(sigmas) - 1
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma = schedule[index]
            if s_churn > 0:
                gamma = (
                    min(s_churn / count, math.sqrt(2.0) - 1.0) if s_tmin <= sigma <= s_tmax else 0.0
                )
                sigma_hat = sigma * (gamma + 1.0)
            else:
                gamma = 0.0
                sigma_hat = sigma
            if gamma > 0:
                if noise is None:
                    raise ValueError(
                        "euler with s_churn > 0 requires a noise sampler; "
                        "construct one matching the descriptor's NoiseKind and pass it as noise="
                    )
                eps = noise(sigmas[index], sigmas[index + 1]) * s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            denoised = denoiser(x, float(sigma_hat))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            d = (x - denoised) / sigma_hat
            sigma_next = schedule[index + 1]
            x = x + d * (sigma_next - sigma_hat)
        return x

    return solve


def euler_ancestral(
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_euler_ancestral`` on device-float32 kernels."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        count = len(sigmas) - 1
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        flow = is_flow_parameterization(info.parameterization)
        effective_s_noise = s_noise * info.noise_scale
        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma = schedule[index]
            sigma_next = schedule[index + 1]
            denoised = denoiser(x, float(sigma))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            if flow:
                if sigma_next == 0:
                    x = denoised
                    continue
                downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
                sigma_down = sigma_next * downstep_ratio
                alpha_next = 1.0 - sigma_next
                alpha_down = 1.0 - sigma_down
                renoise_coeff = (
                    sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
                ) ** 0.5
                ratio = sigma_down / sigma
                x = ratio * x + (1.0 - ratio) * denoised
                if eta > 0:
                    x = (alpha_next / alpha_down) * x
                    if effective_s_noise != 0 and renoise_coeff != 0:
                        if noise is None:
                            raise ValueError(
                                "euler_ancestral with nonzero noise requires a noise sampler; "
                                "construct one matching the descriptor's NoiseKind and pass it "
                                "as noise="
                            )
                        x = (
                            x
                            + noise(sigmas[index], sigmas[index + 1])
                            * effective_s_noise
                            * renoise_coeff
                        )
                continue

            if not eta:
                sigma_down: torch.Tensor | float = sigma_next
                sigma_up: torch.Tensor | float = 0.0
            else:
                sigma_up = min(
                    sigma_next,
                    eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) ** 0.5,
                )
                sigma_down = (sigma_next**2 - sigma_up**2) ** 0.5
            if sigma_down == 0:
                x = denoised
                continue
            derivative = (x - denoised) / sigma
            x = x + derivative * (sigma_down - sigma)
            if effective_s_noise != 0 and sigma_up != 0:
                if noise is None:
                    raise ValueError(
                        "euler_ancestral with nonzero noise requires a noise sampler; "
                        "construct one matching the descriptor's NoiseKind and pass it as noise="
                    )
                x = x + noise(sigmas[index], sigmas[index + 1]) * effective_s_noise * sigma_up
        return x

    return solve


def dpmpp_2m() -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_dpmpp_2m`` on device-float32 kernels."""

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        del noise
        count = len(sigmas) - 1
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        old_denoised: torch.Tensor | None = None
        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma = schedule[index]
            denoised = denoiser(x, float(sigma))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            t = sigma.log().neg()
            t_next = schedule[index + 1].log().neg()
            h = t_next - t
            if old_denoised is None or schedule[index + 1] == 0:
                x = ((-t_next).exp() / (-t).exp()) * x - (-h).expm1() * denoised
            else:
                h_last = t - schedule[index - 1].log().neg()
                r = h_last / h
                denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                x = ((-t_next).exp() / (-t).exp()) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
        return x

    return solve


def _dpmpp_2m_sde(
    *, eta: float, s_noise: float, solver_type: str
) -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_dpmpp_2m_sde`` with device-float32 scalar kernels."""
    if solver_type not in ("heun", "midpoint"):
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        if len(sigmas) <= 1:
            return x
        flow = is_flow_parameterization(info.parameterization)
        if flow and sigmas[0] >= 1.0:
            raise ValueError("flow SDE sampling requires an SNR-offset sigma schedule")
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        scaled_noise = s_noise * info.noise_scale
        old_denoised: torch.Tensor | None = None
        h_last: torch.Tensor | None = None
        count = len(sigmas) - 1
        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma, sigma_next = schedule[index], schedule[index + 1]
            denoised = denoiser(x, float(sigma))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            if sigma_next == 0:
                x = denoised
            else:
                lambda_s = sigma.logit().neg() if flow else sigma.log().neg()
                lambda_t = sigma_next.logit().neg() if flow else sigma_next.log().neg()
                h = lambda_t - lambda_s
                h_eta = h * (eta + 1)
                alpha_t = sigma_next * lambda_t.exp()
                x = (
                    sigma_next / sigma * (-h * eta).exp() * x
                    + alpha_t * (-h_eta).expm1().neg() * denoised
                )
                if old_denoised is not None:
                    assert h_last is not None
                    r = h_last / h
                    if solver_type == "heun":
                        x = x + (
                            alpha_t
                            * ((-h_eta).expm1().neg() / (-h_eta) + 1)
                            * (1 / r)
                            * (denoised - old_denoised)
                        )
                    else:
                        x = x + (0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r)) * (
                            denoised - old_denoised
                        )
                if eta > 0 and scaled_noise > 0:
                    if noise is None:
                        raise ValueError("dpmpp_2m_sde requires a noise sampler")
                    x = (
                        x
                        + noise(sigmas[index], sigmas[index + 1])
                        * sigma_next
                        * (-2 * h * eta).expm1().neg().sqrt()
                        * scaled_noise
                    )
                h_last = h
            old_denoised = denoised
        return x

    return solve


def res_multistep() -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_res_multistep`` with device-float32 scalar kernels."""

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        del noise
        count = len(sigmas) - 1
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        old_sigma_down: torch.Tensor | None = None
        old_denoised: torch.Tensor | None = None
        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma = schedule[index]
            denoised = denoiser(x, float(sigma))
            sigma_down = schedule[index + 1]
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            if sigma_down == 0 or old_denoised is None:
                derivative = (x - denoised) / sigma
                x = x + derivative * (sigma_down - sigma)
            else:
                assert old_sigma_down is not None
                t = sigma.log().neg()
                t_old = old_sigma_down.log().neg()
                t_next = sigma_down.log().neg()
                t_prev = schedule[index - 1].log().neg()
                h = t_next - t
                c2 = (t_prev - t_old) / h
                phi1 = torch.expm1(-h) / -h
                phi2 = (phi1 - 1.0) / -h
                b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
                b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
                x = (-h).exp() * x + h * (b1 * denoised + b2 * old_denoised)
            old_denoised = denoised
            old_sigma_down = sigma_down
        return x

    return solve


def _dpmpp_sde(
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
) -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_dpmpp_sde`` with device-float32 scalar kernels."""
    if r <= 0.0:
        raise ValueError("dpmpp_sde r must be > 0")

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        if len(sigmas) <= 1:
            return x
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        count = len(sigmas) - 1
        flow = is_flow_parameterization(info.parameterization)
        scaled_noise = s_noise * info.noise_scale
        stochastic = eta > 0.0 and scaled_noise > 0.0
        fac = 1.0 / (2.0 * r)

        def lambda_fn(sigma: torch.Tensor) -> torch.Tensor:
            return sigma.logit().neg() if flow else sigma.log().neg()

        def sigma_fn(value: torch.Tensor) -> torch.Tensor:
            return value.neg().sigmoid() if flow else value.neg().exp()

        def ancestral_step(
            sigma_from: torch.Tensor, sigma_to: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor | float]:
            if not eta:
                return sigma_to, 0.0
            sigma_up = min(
                sigma_to,
                eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
            )
            return (sigma_to**2 - sigma_up**2) ** 0.5, sigma_up

        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma_s = schedule[index]
            sigma_t = schedule[index + 1]
            denoised = denoiser(x, float(sigma_s))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            if sigma_t == 0:
                x = denoised
                continue

            lambda_s = lambda_fn(sigma_s)
            lambda_t = lambda_fn(sigma_t)
            h = lambda_t - lambda_s
            lambda_s_1 = lambda_s + r * h
            sigma_s_1 = sigma_fn(lambda_s_1)
            alpha_s = sigma_s * lambda_s.exp()
            alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
            alpha_t = sigma_t * lambda_t.exp()

            sigma_down, sigma_up = ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp())
            lambda_s_1_down = sigma_down.log().neg()
            h_down = lambda_s_1_down - lambda_s
            x_2 = (alpha_s_1 / alpha_s) * (-h_down).exp() * x - alpha_s_1 * (
                -h_down
            ).expm1() * denoised
            if stochastic:
                if noise is None:
                    raise ValueError(
                        "dpmpp_sde with nonzero noise requires a noise sampler; "
                        "construct one matching the descriptor's NoiseKind and pass it as noise="
                    )
                x_2 = (
                    x_2
                    + alpha_s_1 * noise(float(sigma_s), float(sigma_s_1)) * scaled_noise * sigma_up
                )
            denoised_2 = denoiser(x_2, float(sigma_s_1))

            sigma_down, sigma_up = ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp())
            lambda_t_down = sigma_down.log().neg()
            h_down = lambda_t_down - lambda_s
            denoised_d = (1.0 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * (-h_down).exp() * x - alpha_t * (-h_down).expm1() * denoised_d
            if stochastic:
                assert noise is not None
                x = x + alpha_t * noise(float(sigma_s), float(sigma_t)) * scaled_noise * sigma_up
        return x

    return solve


def er_sde(
    *,
    s_noise: float = 1.0,
    noise_scaler: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_stage: int = 3,
    solver_type: str = "ER-SDE",
    eta: float = 1.0,
) -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI ``sample_er_sde`` with device-float32 scalar kernels."""
    if not 1 <= max_stage <= 3:
        raise ValueError(f"er_sde max_stage must be in [1, 3], got {max_stage}")
    if solver_type not in {"ER-SDE", "Reverse-time SDE", "ODE"}:
        raise ValueError("er_sde solver_type must be 'ER-SDE', 'Reverse-time SDE', or 'ODE'")
    if not math.isfinite(eta) or not 0 <= eta <= 100:
        raise ValueError(f"er_sde eta must be in [0, 100], got {eta}")
    if noise_scaler is not None and (solver_type != "ER-SDE" or eta != 1.0):
        raise ValueError(
            "er_sde noise_scaler cannot be combined with a non-default solver_type or eta"
        )
    if solver_type == "ODE" or eta == 0:
        solver_type = "ODE"
        s_noise = 0.0

    def default_er_sde_scale(value: torch.Tensor) -> torch.Tensor:
        return value * ((value**0.3).exp() + 10.0)

    def eta_er_sde_scale(value: torch.Tensor) -> torch.Tensor:
        return value * ((value**0.3).exp() + 10.0) ** eta

    def reverse_time_scale(value: torch.Tensor) -> torch.Tensor:
        return value ** (eta + 1.0)

    def ode_scale(value: torch.Tensor) -> torch.Tensor:
        return value

    if noise_scaler is not None:
        scale = noise_scaler
    elif solver_type == "ER-SDE":
        scale = default_er_sde_scale if eta == 1.0 else eta_er_sde_scale
    elif solver_type == "Reverse-time SDE":
        scale = reverse_time_scale
    else:
        scale = ode_scale

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        if len(sigmas) <= 1:
            return x
        schedule = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        flow = is_flow_parameterization(info.parameterization)
        half_log_snrs = schedule.logit().neg() if flow else schedule.log().neg()
        er_lambdas = half_log_snrs.neg().exp()
        point_indices = torch.arange(0, 200.0, dtype=torch.float32, device=x.device)
        scaled_noise = s_noise * info.noise_scale
        old_denoised: torch.Tensor | None = None
        old_denoised_d: torch.Tensor | None = None
        count = len(sigmas) - 1

        for index in range(count):
            if on_step_begin is not None:
                on_step_begin(index)
            sigma_s = schedule[index]
            sigma_t = schedule[index + 1]
            denoised = denoiser(x, float(sigma_s))
            _emit(on_step, info, index, count, sigmas[index], x, denoised)
            stage_used = min(max_stage, index + 1)
            if sigma_t == 0:
                x = denoised
            else:
                er_lambda_s = er_lambdas[index]
                er_lambda_t = er_lambdas[index + 1]
                alpha_s = sigma_s / er_lambda_s
                alpha_t = sigma_t / er_lambda_t
                r_alpha = alpha_t / alpha_s
                r = scale(er_lambda_t) / scale(er_lambda_s)
                x = r_alpha * r * x + alpha_t * (1 - r) * denoised

                if stage_used >= 2:
                    assert old_denoised is not None
                    dt = er_lambda_t - er_lambda_s
                    lambda_step_size = -dt / 200.0
                    lambda_pos = er_lambda_t + point_indices * lambda_step_size
                    scaled_pos = scale(lambda_pos)
                    s = torch.sum(1 / scaled_pos) * lambda_step_size
                    denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[index - 1])
                    x = x + alpha_t * (dt + s * scale(er_lambda_t)) * denoised_d

                    if stage_used >= 3:
                        assert old_denoised_d is not None
                        s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                        denoised_u = (denoised_d - old_denoised_d) / (
                            (er_lambda_s - er_lambdas[index - 2]) / 2
                        )
                        x = x + alpha_t * ((dt**2) / 2 + s_u * scale(er_lambda_t)) * denoised_u
                    old_denoised_d = denoised_d

                if scaled_noise > 0:
                    if noise is None:
                        raise ValueError(
                            "er_sde with nonzero noise requires a noise sampler; "
                            "construct one matching the descriptor's NoiseKind and pass it "
                            "as noise="
                        )
                    x = x + alpha_t * noise(float(sigma_s), float(sigma_t)) * scaled_noise * (
                        er_lambda_t**2 - er_lambda_s**2 * r**2
                    ).sqrt().nan_to_num(nan=0.0)
            old_denoised = denoised
        return x

    return solve


class _SigmaConvert:
    """UniPC's sigma-space VP schedule (uni_pc.py:821-838 @ f4b99bc)."""

    @staticmethod
    def marginal_log_mean_coeff(sigma: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.marginal_log_mean_coeff(sigma))

    def marginal_std(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(sigma)))

    def marginal_lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = self.marginal_log_mean_coeff(sigma)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def _expand_dims(value: torch.Tensor, dims: int) -> torch.Tensor:
    """UniPC expand_dims (uni_pc.py:808-818 @ f4b99bc)."""
    return value[(...,) + (None,) * (dims - 1)]


_UniPCTrace = Callable[[int, torch.Tensor], None]


def _uni_pc(variant: str, *, trace: _UniPCTrace | None = None) -> StepBeginSolverFn[torch.Tensor]:
    """Torch-exact ComfyUI UniPC BH1/BH2 (uni_pc.py:579-697, 700-873)."""

    def solve(
        denoiser: Denoiser[torch.Tensor],
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[torch.Tensor] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        del noise
        if len(sigmas) <= 1:
            return x

        solver_name = "uni_pc" if variant == "bh1" else f"uni_pc_{variant}"
        normalized = [
            0.001 if index == len(sigmas) - 1 and sigma == 0.0 else float(sigma)
            for index, sigma in enumerate(sigmas)
        ]
        if any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in normalized):
            raise ValueError(
                f"{solver_name} degenerate schedule: sigmas must be finite and positive"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{solver_name} degenerate schedule: repeated or invalid sigmas")

        # sample_unipc @ f4b99bc, uni_pc.py:846-869: keep the schedule on
        # the latent device as float32, replace only the terminal zero,
        # normalize the initial noise, and use the lower-order-final BH path.
        timesteps = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
        if timesteps[-1] == 0:
            timesteps = timesteps[:]
            timesteps[-1] = 0.001
        executed_sigmas = timesteps.tolist()
        if any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in executed_sigmas):
            raise ValueError(
                f"{solver_name} degenerate schedule after float32 conversion: "
                "sigmas must be finite and positive"
            )
        if len(set(executed_sigmas)) != len(executed_sigmas):
            raise ValueError(
                f"{solver_name} degenerate schedule after float32 conversion: "
                "repeated or invalid sigmas"
            )
        schedule = _SigmaConvert()
        x = x / torch.sqrt(1.0 + timesteps[0] ** 2.0)
        order = min(3, len(timesteps) - 2)

        # model_wrapper + predict_eps_sigma + UniPC.data_prediction_fn
        # @ f4b99bc, uni_pc.py:282-349, 390-410, 840-843. Do not collapse
        # these algebraically: each float32 tensor operation is bit-visible.
        def model_fn(value: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t = t.expand(value.shape[0])
            sigma = t.view(t.shape[:1] + (1,) * (value.ndim - 1))
            model_input = value * ((sigma**2 + 1.0) ** 0.5)
            denoised = denoiser(model_input, float(t.reshape(-1)[0]))
            eps = (model_input - denoised) / sigma
            alpha_t = schedule.marginal_alpha(t)
            sigma_t = schedule.marginal_std(t)
            return (value - _expand_dims(sigma_t, value.dim()) * eps) / _expand_dims(
                alpha_t, value.dim()
            )

        model_prev_list: list[torch.Tensor] = []
        t_prev_list: list[torch.Tensor] = []
        callback_index = 0

        def emit_before_update() -> None:
            nonlocal callback_index
            source_t = t_prev_list[-1]
            source_sigma = float(source_t.reshape(-1)[0])
            source_model = model_prev_list[-1]
            current = x * torch.sqrt(1.0 + source_t[0] ** 2.0)
            if info.on_state is not None:
                source_model = denoiser(current, source_sigma)
            _emit(
                on_step,
                info,
                callback_index,
                steps,
                source_sigma,
                current,
                source_model,
            )
            callback_index += 1

        # multistep_uni_pc_bh_update @ f4b99bc, uni_pc.py:579-697.
        def update(
            value: torch.Tensor,
            t: torch.Tensor,
            step_order: int,
            *,
            use_corrector: bool,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            dims = value.dim()
            t_prev_0 = t_prev_list[-1]
            lambda_prev_0 = schedule.marginal_lambda(t_prev_0)
            lambda_t = schedule.marginal_lambda(t)
            model_prev_0 = model_prev_list[-1]
            sigma_prev_0 = schedule.marginal_std(t_prev_0)
            sigma_t = schedule.marginal_std(t)
            log_alpha_t = schedule.marginal_log_mean_coeff(t)
            alpha_t = torch.exp(log_alpha_t)
            h = lambda_t - lambda_prev_0

            rks: list[torch.Tensor | float] = []
            diffs: list[torch.Tensor] = []
            for index in range(1, step_order):
                t_prev_i = t_prev_list[-(index + 1)]
                model_prev_i = model_prev_list[-(index + 1)]
                lambda_prev_i = schedule.marginal_lambda(t_prev_i)
                rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
                rks.append(rk)
                diffs.append((model_prev_i - model_prev_0) / rk)
            rks.append(1.0)
            rks_tensor = torch.tensor(rks, device=value.device)

            hh = -h[0]
            h_phi_1 = torch.expm1(hh)
            h_phi_k = h_phi_1 / hh - 1
            factorial_i = 1
            b: list[torch.Tensor] = []
            rows: list[torch.Tensor] = []
            if variant == "bh1":
                b_h = hh
            elif variant == "bh2":
                b_h = torch.expm1(hh)
            else:
                raise ValueError(f"unknown UniPC variant {variant!r}")
            for index in range(1, step_order + 1):
                rows.append(torch.pow(rks_tensor, index - 1))
                b.append(h_phi_k * factorial_i / b_h)
                factorial_i *= index + 1
                h_phi_k = h_phi_k / hh - 1 / factorial_i
            matrix = torch.stack(rows)
            rhs = torch.tensor(b, device=value.device)

            diff_tensor = torch.stack(diffs, dim=1) if diffs else None
            use_predictor = diff_tensor is not None
            rhos_p: torch.Tensor | None = None
            if diff_tensor is not None:
                if step_order == 2:
                    rhos_p = torch.tensor([0.5], device=rhs.device)
                else:
                    rhos_p = torch.linalg.solve(matrix[:-1, :-1], rhs[:-1])
            rhos_c: torch.Tensor | None = None
            if use_corrector:
                if step_order == 1:
                    rhos_c = torch.tensor([0.5], device=rhs.device)
                else:
                    rhos_c = torch.linalg.solve(matrix, rhs)

            base = (
                _expand_dims(sigma_t / sigma_prev_0, dims) * value
                - _expand_dims(alpha_t * h_phi_1, dims) * model_prev_0
            )
            candidate = base
            if use_predictor:
                assert diff_tensor is not None and rhos_p is not None
                pred_res = torch.tensordot(
                    diff_tensor,
                    rhos_p,
                    dims=([1], [0]),  # pyright: ignore[reportArgumentType]
                )
                candidate = base - _expand_dims(alpha_t * b_h, dims) * pred_res
            if not use_corrector:
                return candidate, None
            assert rhos_c is not None
            model_t = model_fn(candidate, t)
            if diff_tensor is not None:
                corr_res: torch.Tensor | int = torch.tensordot(
                    diff_tensor,
                    rhos_c[:-1],
                    dims=([1], [0]),  # pyright: ignore[reportArgumentType]
                )
            else:
                corr_res = 0
            d1_t = model_t - model_prev_0
            corrected = base - _expand_dims(alpha_t * b_h, dims) * (corr_res + rhos_c[-1] * d1_t)
            return corrected, model_t

        # UniPC.sample(method="multistep", lower_order_final=True)
        # @ f4b99bc, uni_pc.py:700-754. The final iteration performs one
        # extra update at terminal 0.001 without evaluating the model there.
        steps = len(timesteps) - 1
        for step_index in range(steps):
            if on_step_begin is not None:
                on_step_begin(step_index)
            if step_index == 0:
                vec_t = timesteps[0].expand(x.shape[0])
                model_prev_list = [model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                if steps == 1:
                    emit_before_update()
            elif step_index < order:
                emit_before_update()
                vec_t = timesteps[step_index].expand(x.shape[0])
                x, model_x = update(x, vec_t, step_index, use_corrector=True)
                if model_x is None:
                    model_x = model_fn(x, vec_t)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                extra_final_step = 1 if step_index == steps - 1 else 0
                for step in range(step_index, step_index + 1 + extra_final_step):
                    emit_before_update()
                    vec_t = timesteps[step].expand(x.shape[0])
                    step_order = min(order, steps + 1 - step)
                    x, model_x = update(
                        x,
                        vec_t,
                        step_order,
                        use_corrector=step < steps,
                    )
                    for index in range(order - 1):
                        t_prev_list[index] = t_prev_list[index + 1]
                        model_prev_list[index] = model_prev_list[index + 1]
                    t_prev_list[-1] = vec_t
                    if step < steps:
                        if model_x is None:
                            model_x = model_fn(x, vec_t)
                        model_prev_list[-1] = model_x
            if trace is not None:
                trace(step_index, x)

        if callback_index != steps:
            raise RuntimeError("UniPC callback count does not match its schedule")

        x /= schedule.marginal_alpha(timesteps[-1])
        if sigmas[-1] == 0.0 and is_flow_parameterization(info.parameterization):
            # sample_unipc's terminal replacement mutates ComfyUI's caller schedule,
            # so its outer flow inverse scaling divides by the replacement too.
            x /= 1.0 - timesteps[-1]
        return x

    return solve


def uni_pc() -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI UniPC BH1 on reference float32 tensor kernels."""
    return _uni_pc("bh1")


def uni_pc_bh2() -> StepBeginSolverFn[torch.Tensor]:
    """ComfyUI UniPC BH2 on reference float32 tensor kernels."""
    return _uni_pc("bh2")


def _float_option(options: Mapping[str, OptionValue], name: str) -> float:
    value = options[name]
    assert isinstance(value, float), f"option {name!r} resolved to non-float {value!r}"
    return value


def _make_euler(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    return euler(
        s_churn=_float_option(options, "s_churn"),
        s_tmin=_float_option(options, "s_tmin"),
        s_tmax=_float_option(options, "s_tmax"),
        s_noise=_float_option(options, "s_noise"),
    )


def _make_euler_ancestral(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    return euler_ancestral(
        eta=_float_option(options, "eta"),
        s_noise=_float_option(options, "s_noise"),
    )


def _make_dpmpp_2m(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    del options
    return dpmpp_2m()


def _make_dpmpp_2m_sde(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    solver_type = options["solver_type"]
    assert isinstance(solver_type, str)
    return _dpmpp_2m_sde(
        eta=_float_option(options, "eta"),
        s_noise=_float_option(options, "s_noise"),
        solver_type=solver_type,
    )


def _make_dpmpp_2m_sde_heun(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    return _dpmpp_2m_sde(
        eta=_float_option(options, "eta"),
        s_noise=_float_option(options, "s_noise"),
        solver_type="heun",
    )


def _make_ddim(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    del options
    return euler()


def _make_res_multistep(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    del options
    return res_multistep()


def _make_dpmpp_sde(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    return _dpmpp_sde(
        eta=_float_option(options, "eta"),
        s_noise=_float_option(options, "s_noise"),
        r=_float_option(options, "r"),
    )


def _make_er_sde(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    solver_type = options["solver_type"]
    max_stage = options["max_stage"]
    assert isinstance(solver_type, str)
    assert isinstance(max_stage, int) and not isinstance(max_stage, bool)
    return er_sde(
        solver_type=solver_type,
        max_stage=max_stage,
        eta=_float_option(options, "eta"),
        s_noise=_float_option(options, "s_noise"),
    )


def _make_uni_pc(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    del options
    return uni_pc()


def _make_uni_pc_bh2(options: Mapping[str, OptionValue]) -> SolverFn[torch.Tensor]:
    del options
    return uni_pc_bh2()


# Behavior evidence for the float-sensitive overrides, captured at
# import before any caller code runs: the distributed receipt gate
# trusts registries built here, and a function object's __code__ and
# defaults are reassignable, so each construction re-verifies them.
_TORCH_MAKE_EVIDENCE: dict[str, CallableEvidence] = {
    "dinkster.euler": CallableEvidence.capture(_make_euler),
    "dinkster.euler_ancestral": CallableEvidence.capture(_make_euler_ancestral),
    "dinkster.dpmpp_2m": CallableEvidence.capture(_make_dpmpp_2m),
    "dinkster.dpmpp_2m_sde": CallableEvidence.capture(_make_dpmpp_2m_sde),
    "dinkster.dpmpp_2m_sde_gpu": CallableEvidence.capture(_make_dpmpp_2m_sde),
    "dinkster.dpmpp_2m_sde_heun": CallableEvidence.capture(_make_dpmpp_2m_sde_heun),
    "dinkster.dpmpp_2m_sde_heun_gpu": CallableEvidence.capture(_make_dpmpp_2m_sde_heun),
    "dinkster.ddim": CallableEvidence.capture(_make_ddim),
    "dinkster.res_multistep": CallableEvidence.capture(_make_res_multistep),
    "dinkster.dpmpp_sde": CallableEvidence.capture(_make_dpmpp_sde),
    "dinkster.dpmpp_sde_gpu": CallableEvidence.capture(_make_dpmpp_sde),
    "dinkster.er_sde": CallableEvidence.capture(_make_er_sde),
    "dinkster.uni_pc": CallableEvidence.capture(_make_uni_pc),
    "dinkster.uni_pc_bh2": CallableEvidence.capture(_make_uni_pc_bh2),
}


def torch_sampler_registry(
    descriptors: Iterable[SamplerDescriptor[Any]] | None = None,
) -> Registry[SamplerDescriptor[Any]]:
    """A sampler registry bound to torch execution factories."""
    registry: Registry[SamplerDescriptor[Any]] = Registry()
    portable_factories = {
        descriptor.id: descriptor.make for descriptor in _portable_solvers.builtin_samplers()
    }
    for descriptor in builtin_samplers() if descriptors is None else descriptors:
        if descriptor.make is None and "build" not in vars(descriptor):
            portable_factory = portable_factories.get(descriptor.id)
            if portable_factory is not None:
                descriptor = replace(descriptor, make=portable_factory)
            evidence = _TORCH_MAKE_EVIDENCE.get(descriptor.id)
            if evidence is not None:
                descriptor = replace(descriptor, make=evidence.resolve())
        registry.register(descriptor)
    return registry


__all__ = [
    "dpmpp_2m",
    "er_sde",
    "euler",
    "euler_ancestral",
    "res_multistep",
    "torch_sampler_registry",
    "uni_pc",
    "uni_pc_bh2",
]
