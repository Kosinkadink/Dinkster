"""Float32-exact sigma schedules: reference kernels for brownian replay.

dinkster_inference.schedules ports every schedule as pure float64 math,
pinned within 3e-6 of the reference (tests/test_inference_sampling_math
REL) - close enough for value math, and exactly wrong for brownian
trees: a BrownianTreeNoise query time shifted by even one float32 ulp
moves the tree's 1e-6 grid, and brownian roughness turns a time shift
of delta into an O(sqrt(delta)) change in the draw, decorrelating the
noise stream. The SD flip-parity harness caught dpmpp_2m_sde over
karras drifting to 3e-2 from exactly this (test_sd_pipeline.py
sd15_sde_karras).

The reference computes the float-sensitive schedules with float32
TENSOR kernels (torch.linspace ramps, float32 scalar mul/add, float32
pow/exp/tan - comfy/samplers.py + comfy/k_diffusion/sampling.py
@ b78cec87) whose rounding pure Python cannot reproduce: torch's
vectorized float32 transcendentals are not correctly rounded, and
torch.linspace's fill order is size-dependent. This module executes
the same ops on the same kernels instead - bit-identical sigmas by
construction, the same same-torch-same-arch reproducibility contract
as every other executing port in this package.

Overridden here, each with executed evidence (test_sd_pipeline.py /
test_pipeline.py / test_sampling_execution.py): simple, karras,
exponential, normal, sgm_uniform, linear_quadratic, kl_optimal.
Simple selects entries from the reference's float32 table because a
one-ulp flow sigma difference can move BF16 denoiser values. The
remaining table-index schedules (ddim_uniform, beta) select reference
float32 flow and EDM tables. The schedules rebound here drift far past one ulp
on pure kernels (karras reached 3.4e-2 end-to-end), which is what
forces the ports.

normal and sgm_uniform route through the model-sampling object's
timestep()/sigma() conversions, and calculate_sigmas @ b78cec87 hands
karras/exponential/kl_optimal float(sigma_min)/float(sigma_max) read
from the float32 sigma table - so bit-exactness needs the SPACE
surface on reference kernels too. _reference_surface realizes the
four wired space kinds (DiscreteSigmas -> ModelSamplingDiscrete's
float32 log-sigma interpolation, FlowSigmas ->
ModelSamplingDiscreteFlow, FluxFlowSigmas -> ModelSamplingFlux, each
rebuilding the reference's float32 sigma table, plus ContinuousEDMSigmas).
Unknown custom space kinds fall back to the pure port's float64
surface: value-close, not brownian-safe - port the kind here WITH
executed golden coverage before an SDE sampler meets it (ROADMAP:
Native inference).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import torch
from dinkster_inference import (
    SIGMA_PERCENT_ZERO,
    CallableEvidence,
    ContinuousEDMSigmas,
    DiscreteSigmas,
    FlowSigmas,
    FluxFlowSigmas,
    Registry,
    SchedulerDescriptor,
    SigmaSpace,
    builtin_schedulers,
)
from dinkster_inference import (
    beta_schedule as _pure_beta_schedule,
)
from dinkster_inference import (
    ddim_uniform_schedule as _pure_ddim_uniform_schedule,
)
from dinkster_inference import (
    normal_schedule as _pure_normal_schedule,
)
from dinkster_inference import (
    sgm_uniform_schedule as _pure_sgm_uniform_schedule,
)
from dinkster_inference import (
    simple_schedule as _pure_simple_schedule,
)

__all__ = [
    "beta_schedule",
    "continuous_edm_percent_to_sigma",
    "custom_beta_sigmas",
    "custom_percent_to_sigma",
    "ddim_uniform_schedule",
    "discrete_percent_to_sigma",
    "exponential_schedule",
    "ideogram4_sigmas",
    "karras_schedule",
    "kl_optimal_schedule",
    "linear_quadratic_schedule",
    "normal_schedule",
    "sd_turbo_sigmas",
    "sgm_uniform_schedule",
    "simple_schedule",
    "scheduler_on_device",
    "torch_scheduler_registry",
]


def _require_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")


def ideogram4_sigmas(
    steps: int,
    width: int,
    height: int,
    mu: float,
    std: float,
) -> torch.Tensor:
    """Ideogram 4's float64 logit-normal quantiles as float32 sigmas."""

    _require_steps(steps)
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    quantiles = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    timestep = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(quantiles))
    minimum = 1.0 / (1.0 + math.exp(0.5 * 18.0))
    maximum = 1.0 / (1.0 + math.exp(0.5 * -15.0))
    sigmas = (1.0 - timestep.clamp(minimum, maximum)).flip(0)
    sigmas[-1] = 0.0
    return sigmas.float()


# --------------------------------------------------------------------------
# Reference model-sampling surfaces: the float32 tensor realizations of
# the wired SigmaSpace kinds, exactly as the reference constructs them
# (comfy/model_sampling.py @ b78cec87). Rebuilding the table per call
# is microseconds against a sampling run and keeps this module
# stateless.


class _DiscreteRef:
    """ModelSamplingDiscrete @ b78cec87: float32 sigma/log-sigma
    buffers (log taken in float64 BEFORE the float32 cast, per
    set_sigmas), nearest-log-sigma timestep, log-space interpolated
    sigma."""

    def __init__(
        self,
        entries: tuple[float, ...],
        device: torch.device | str | None = None,
    ) -> None:
        sigmas = torch.tensor(entries, dtype=torch.float64)
        self.sigmas = sigmas.float().to(device=device)
        self.log_sigmas = sigmas.log().float().to(device=device)

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """comfy/model_sampling.py time_snr_shift @ b78cec87 (the alpha
    == 1.0 identity branch included: it skips the float32 kernels
    entirely in the reference too)."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class _FlowRef:
    """ModelSamplingDiscreteFlow @ b78cec87: float32 table from
    time_snr_shift over (arange/timesteps) * multiplier, timestep =
    sigma * multiplier."""

    def __init__(
        self,
        shift: float,
        multiplier: float,
        timesteps: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.shift = shift
        self.multiplier = multiplier
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier).to(
            device=device
        )

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.multiplier

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return _time_snr_shift(self.shift, timestep / self.multiplier)


class _FluxRef:
    """ModelSamplingFlux @ b78cec87: float32 table from
    flux_time_shift over arange/timesteps, timestep = sigma."""

    def __init__(
        self,
        shift: float,
        timesteps: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.shift = shift
        self.sigmas = self.sigma(torch.arange(1, timesteps + 1, 1) / timesteps).to(device=device)

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        # flux_time_shift @ b78cec87 with sigma pinned to 1.0 (the
        # only value ModelSamplingFlux passes).
        e_mu = math.exp(self.shift)
        return e_mu / (e_mu + (1 / timestep - 1) ** 1.0)


class _ContinuousEDMRef:
    """ModelSamplingContinuousEDM @ b78cec87 on float32 kernels."""

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        timesteps: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.sigmas = (
            torch.linspace(math.log(sigma_min), math.log(sigma_max), timesteps)
            .exp()
            .to(device=device)
        )

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * sigma.log()

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return (timestep / 0.25).exp()


@dataclass(frozen=True)
class _ExactTableView:
    """A non-validating SigmaSpace view over an exact reference table."""

    inner: SigmaSpace
    exact_table: tuple[float, ...]

    @property
    def sigma_min(self) -> float:
        return self.exact_table[0]

    @property
    def sigma_max(self) -> float:
        return self.exact_table[-1]

    @property
    def table(self) -> tuple[float, ...]:
        return self.exact_table

    def sigma(self, timestep: float) -> float:
        return self.inner.sigma(timestep)

    def timestep(self, sigma: float) -> float:
        return self.inner.timestep(sigma)

    def percent_to_sigma(self, percent: float) -> float:
        if isinstance(self.inner, DiscreteSigmas):
            return discrete_percent_to_sigma(self.inner, percent)
        if isinstance(self.inner, ContinuousEDMSigmas):
            return continuous_edm_percent_to_sigma(self.inner, percent)
        return self.inner.percent_to_sigma(percent)


def _reference_surface(
    space: SigmaSpace,
    device: torch.device | str | None = None,
) -> _DiscreteRef | _FlowRef | _FluxRef | _ContinuousEDMRef | None:
    if isinstance(space, DiscreteSigmas):
        return _DiscreteRef(space.entries, device)
    if isinstance(space, FlowSigmas):
        return _FlowRef(space.shift, space.multiplier, space.timesteps, device)
    if isinstance(space, FluxFlowSigmas):
        return _FluxRef(space.shift, space.timesteps, device)
    if isinstance(space, ContinuousEDMSigmas):
        return _ContinuousEDMRef(space.min_sigma, space.max_sigma, space.timesteps, device)
    return None


def _table_space(
    space: SigmaSpace,
    device: torch.device | str | None = None,
) -> SigmaSpace:
    """Use exact float32 reference tables for index-based schedulers."""
    return _exact_table_space(space, device)


def _exact_table_space(
    space: SigmaSpace,
    device: torch.device | str | None = None,
) -> SigmaSpace:
    """View a known space through its reference float32 sigma table."""
    ref = _reference_surface(space, device)
    if ref is None:
        return space
    return _ExactTableView(space, tuple(float(sigma) for sigma in ref.sigmas))


def continuous_edm_percent_to_sigma(
    space: ContinuousEDMSigmas,
    percent: float,
) -> float:
    """ModelSamplingContinuousEDM.percent_to_sigma on reference endpoints."""
    if percent <= 0.0:
        return 999999999.9
    if percent >= 1.0:
        return 0.0
    ref = _ContinuousEDMRef(space.min_sigma, space.max_sigma, space.timesteps)
    log_sigma_min = math.log(float(ref.sigma_min))
    return math.exp(
        (math.log(float(ref.sigma_max)) - log_sigma_min) * (1.0 - percent) + log_sigma_min
    )


def discrete_percent_to_sigma(
    space: DiscreteSigmas,
    percent: float,
) -> float:
    """ModelSamplingDiscrete.percent_to_sigma on the reference's float32 kernel."""
    if percent <= 0.0:
        return SIGMA_PERCENT_ZERO
    if percent >= 1.0:
        return 0.0
    timestep = torch.tensor((1.0 - percent) * (len(space.entries) - 1))
    return float(_DiscreteRef(space.entries).sigma(timestep))


def _endpoints(
    space: SigmaSpace,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    """calculate_sigmas @ b78cec87 hands the use_ms=False handlers
    float(model_sampling.sigma_min) / float(model_sampling.sigma_max)
    - the float32 table endpoints as python floats. Unknown space
    kinds fall back to the pure float64 surface."""
    ref = _reference_surface(space, device)
    if ref is None:
        return space.sigma_min, space.sigma_max
    return float(ref.sigma_min), float(ref.sigma_max)


# --------------------------------------------------------------------------
# Schedules.


def simple_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    ref = _reference_surface(space, device)
    if ref is None:
        return _pure_simple_schedule(steps, space)
    _require_steps(steps)
    table = ref.sigmas
    stride = len(table) / steps
    return (
        *(float(table[-(1 + int(index * stride))]) for index in range(steps)),
        0.0,
    )


def ddim_uniform_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    return _pure_ddim_uniform_schedule(steps, _table_space(space, device))


def beta_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    alpha: float = 0.6,
    beta: float = 0.6,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    return _pure_beta_schedule(steps, _table_space(space, device), alpha=alpha, beta=beta)


def beta57_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """Beta spacing with alpha=0.5/beta=0.7 (the RES4LYF "beta57"
    scheduler) selecting from the space's exact reference float32
    sigma table. Unlike dinkster.beta this reads through
    _exact_table_space even for discrete spaces: beta57 ships paired
    with brownian SDE goldens, and the brownian tree decorrelates on
    one-float32-ulp sigma differences, so the returned sigmas must be
    the reference's float32 table reads bit-for-bit. bong_tangent
    needs no override here: the reference computes it with scalar
    libm calls (no tensor kernels) and materializes through the same
    float64 -> float32 rounding the pure port applies, so the values
    agree bit-for-bit on any host by construction."""
    return beta_schedule(
        steps,
        _exact_table_space(space, device),
        alpha=0.5,
        beta=0.7,
        device=device,
    )


def custom_beta_sigmas(
    space: SigmaSpace,
    steps: int,
    alpha: float,
    beta: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """BetaSamplingScheduler sigmas over the space's exact float32 table."""
    return beta_schedule(
        steps,
        space,
        alpha=alpha,
        beta=beta,
        device=device,
    )


def custom_percent_to_sigma(
    space: SigmaSpace,
    percent_to_sigma: Callable[[float], float],
    percent: float,
    *,
    return_actual_sigma: bool,
) -> float:
    """One percent-to-sigma query, optionally clamped to the reference
    endpoints at the exact 0.0 / 1.0 percents (comfy/samplers.py
    percent_to_sigma with return_actual_sigma @ b78cec87)."""
    sigma = percent_to_sigma(percent)
    if return_actual_sigma:
        sigma_min, sigma_max = _endpoints(space)
        if percent == 0.0:
            sigma = sigma_max
        elif percent == 1.0:
            sigma = sigma_min
    return sigma


def sd_turbo_sigmas(
    space: SigmaSpace,
    steps: int,
    denoise: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """SDTurboScheduler on ModelSamplingDiscrete's reference kernel."""
    if not isinstance(space, DiscreteSigmas):
        raise ValueError("SD Turbo sigmas require a discrete sigma space")
    if not 1 <= steps <= 10:
        raise ValueError(f"SD Turbo steps must be in [1, 10], got {steps}")
    if not math.isfinite(denoise) or not 0.0 <= denoise <= 1.0:
        raise ValueError(f"SD Turbo denoise must be finite and in [0.0, 1.0], got {denoise}")
    start_step = 10 - int(10 * denoise)
    timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step : start_step + steps]
    sigmas = _DiscreteRef(space.entries, device).sigma(timesteps)
    return (*(float(sigma) for sigma in sigmas), 0.0)


def karras_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    rho: float = 7.0,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """Karras et al. (2022) schedule on the reference's own kernels
    (get_sigmas_karras @ b78cec87): float32 table endpoints via
    calculate_sigmas' float() reads, the float32 linspace ramp, the
    float32 scalar mul/add, the float32 pow."""
    _require_steps(steps)
    sigma_min, sigma_max = _endpoints(space, device)
    ramp = torch.linspace(0, 1, steps, dtype=torch.float32)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return (*(float(sigma) for sigma in sigmas), 0.0)


def exponential_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """Uniform in log-sigma on the reference's kernels
    (get_sigmas_exponential @ b78cec87): float64 math.log endpoints,
    float32 linspace, float32 exp."""
    _require_steps(steps)
    sigma_min, sigma_max = _endpoints(space, device)
    sigmas = torch.linspace(
        math.log(sigma_max),
        math.log(sigma_min),
        steps,
        dtype=torch.float32,
    ).exp()
    return (*(float(sigma) for sigma in sigmas), 0.0)


def normal_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """Uniform in timestep on the reference's kernels (comfy/
    samplers.py normal_scheduler @ b78cec87): float32 linspace over
    the space's timestep endpoints, per-point float32 sigma
    conversion."""
    return _normal(steps, space, sgm=False, device=device)


def sgm_uniform_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """SGM variant: one extra point, last dropped (normal_scheduler
    with sgm=True @ b78cec87)."""
    return _normal(steps, space, sgm=True, device=device)


def _normal(
    steps: int,
    space: SigmaSpace,
    *,
    sgm: bool,
    device: torch.device | str | None,
) -> tuple[float, ...]:
    _require_steps(steps)
    ref = _reference_surface(space, device)
    if ref is None:
        pure = _pure_sgm_uniform_schedule if sgm else _pure_normal_schedule
        return pure(steps, space)
    start = ref.timestep(ref.sigma_max)
    end = ref.timestep(ref.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(ref.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = [float(ref.sigma(timesteps[x])) for x in range(len(timesteps))]
    if append_zero:
        sigs.append(0.0)
    return tuple(sigs)


def linear_quadratic_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """Mochi's linear-then-quadratic schedule on the reference's
    kernels (comfy/samplers.py linear_quadratic_schedule @ b78cec87):
    the schedule list is pure float64 in the reference too; the
    float32 rounding happens at torch.FloatTensor(schedule) *
    sigma_max, replicated here."""
    _require_steps(steps)
    if steps == 1:
        schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        if not 0 < linear_steps < steps:
            raise ValueError("linear_steps must be in (0, steps)")
        linear = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
            quadratic_steps**2
        )
        const = quadratic_coef * (linear_steps**2)
        quadratic = [
            quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, steps)
        ]
        schedule = [1.0 - x for x in (*linear, *quadratic, 1.0)]
    ref = _reference_surface(space, device)
    sigma_max = (
        ref.sigma_max.cpu()
        if ref is not None
        else torch.tensor(space.sigma_max, dtype=torch.float32)
    )
    sigmas = torch.tensor(schedule, dtype=torch.float32) * sigma_max
    return tuple(float(sigma) for sigma in sigmas)


def kl_optimal_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, ...]:
    """KL-optimal spacing on the reference's kernels (comfy/
    samplers.py kl_optimal_scheduler @ b78cec87): float64 math.atan
    endpoints, float32 arange ramp, float32 mul/add, float32 tan.

    Requires steps >= 2 like the pure port: the reference divides
    arange(steps) by steps - 1, so steps == 1 yields NaN there."""
    if steps < 2:
        raise ValueError(
            "kl_optimal needs steps >= 2 (the reference formula divides "
            "by steps - 1 and produces NaN for a single step)"
        )
    sigma_min, sigma_max = _endpoints(space, device)
    adj_idxs = torch.arange(steps, dtype=torch.float).div_(steps - 1)
    sigmas = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return (*(float(sigma) for sigma in sigmas), 0.0)


_TORCH_SCHEDULES = {
    "dinkster.karras": karras_schedule,
    "dinkster.exponential": exponential_schedule,
    "dinkster.normal": normal_schedule,
    "dinkster.sgm_uniform": sgm_uniform_schedule,
    "dinkster.linear_quadratic": linear_quadratic_schedule,
    "dinkster.kl_optimal": kl_optimal_schedule,
}

_EDM_TABLE_SCHEDULES = {
    "dinkster.simple": simple_schedule,
    "dinkster.ddim_uniform": ddim_uniform_schedule,
    "dinkster.beta": beta_schedule,
    "res4lyf.beta57": beta57_schedule,
}

# Behavior evidence for the overrides, captured at import before any
# caller code runs: the distributed receipt gate trusts registries built
# here, and a function object's __code__ and defaults are reassignable,
# so each construction re-verifies them.
_TORCH_SCHEDULE_EVIDENCE: dict[str, CallableEvidence] = {
    schedule_id: CallableEvidence.capture(schedule_fn)
    for schedule_id, schedule_fn in (*_TORCH_SCHEDULES.items(), *_EDM_TABLE_SCHEDULES.items())
}


def scheduler_on_device(
    scheduler: SchedulerDescriptor,
    device: torch.device | str | None,
) -> SchedulerDescriptor:
    """Bind a builtin reference-kernel scheduler to its model device."""
    evidence = _TORCH_SCHEDULE_EVIDENCE.get(scheduler.id)
    if evidence is None:
        return scheduler
    schedule = evidence.resolve()
    if scheduler.make_sigmas is not schedule:
        return scheduler
    return replace(scheduler, make_sigmas=partial(schedule, device=device))


def torch_scheduler_registry() -> Registry[SchedulerDescriptor]:
    """builtin_scheduler_registry with the float32-sensitive schedules
    rebound to this module's reference-kernel implementations - the
    registry the executing runtimes default to."""
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in builtin_schedulers():
        evidence = _TORCH_SCHEDULE_EVIDENCE.get(descriptor.id)
        if evidence is not None:
            descriptor = replace(descriptor, make_sigmas=evidence.resolve())
        registry.register(descriptor)
    return registry
