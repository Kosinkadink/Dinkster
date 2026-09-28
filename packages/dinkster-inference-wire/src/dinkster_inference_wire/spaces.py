"""Sigma spaces: how a model family's noise levels map to timesteps.

The typed port of ComfyUI's model_sampling *schedule* classes
(comfy/model_sampling.py ModelSamplingDiscrete / DiscreteFlow / Flux /
ContinuousEDM @ b78cec87) - the half of model_sampling that answers
"what sigma is timestep t?" and "what timestep is sigma s?". The
executing backend owns the *prediction* half (what the model output
means); ComfyUI fuses both by dynamic multiple inheritance, which the
plan explicitly rejects.

Everything here is pure float math - no torch, no tensors. Values are
float64 where the reference stores float32 tables; conformance tests
pin agreement against reference goldens within float32 tolerance.

``percent_to_sigma`` generalizes the reference's hardcoded ``* 999.0``
to ``* (len(table) - 1)``; identical for the universal 1000-entry case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

#: percent_to_sigma(0.0) for non-flow spaces - the reference's "before
#: everything" sentinel (comfy/model_sampling.py @ b78cec87).
SIGMA_PERCENT_ZERO = 999999999.9


def time_snr_shift(alpha: float, t: float) -> float:
    """SD3-style flow shift (comfy/model_sampling.py time_snr_shift
    @ b78cec87)."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    """Flux exponential flow shift (comfy/model_sampling.py
    flux_time_shift @ b78cec87). ``t`` must be nonzero."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class SigmaSpace(Protocol):
    """What schedules and solvers may ask about a model's sigma line.

    Replaces stage 1's placeholder ScheduleContext: schedules need the
    sigma<->timestep conversions, not just the endpoints. ``table`` is
    the ascending discrete sigma table when the space has one (index 0
    = sigma_min, matching ModelSamplingDiscrete.sigmas @ b78cec87);
    None for spaces that are purely functional.
    """

    @property
    def sigma_min(self) -> float: ...

    @property
    def sigma_max(self) -> float: ...

    @property
    def table(self) -> tuple[float, ...] | None: ...

    def sigma(self, timestep: float) -> float:
        """The sigma at (possibly fractional) ``timestep``."""
        ...

    def timestep(self, sigma: float) -> float:
        """The reference model's timestep transform of ``sigma``.

        NOT necessarily the inverse of ``sigma()``: shifted flow
        spaces (FlowSigmas with shift != 1, FluxFlowSigmas) return
        the unshifted ``sigma * 1000`` exactly as the reference
        (ModelSamplingDiscreteFlow.timestep @ b78cec87) - fidelity
        to that mapping is the contract, invertibility is not."""
        ...

    def percent_to_sigma(self, percent: float) -> float:
        """Sigma bound for a "first/last N% of sampling" range."""
        ...


def linear_beta_sigmas(
    *,
    linear_start: float = 0.00085,
    linear_end: float = 0.012,
    timesteps: int = 1000,
    zsnr: bool = False,
) -> tuple[float, ...]:
    """The discrete sigma table of a linear-beta DDPM schedule.

    Ports make_beta_schedule("linear") + ModelSamplingDiscrete
    _register_schedule (comfy/ldm/modules/diffusionmodules/util.py,
    comfy/model_sampling.py @ b78cec87): betas are squared linspace in
    sqrt-space, sigmas are sqrt((1 - cumprod(alpha)) / cumprod(alpha)).
    Defaults are the SD15/SDXL settings.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be >= 1")
    start = math.sqrt(linear_start)
    end = math.sqrt(linear_end)
    alphas_cumprod: list[float] = []
    running = 1.0
    for i in range(timesteps):
        frac = i / (timesteps - 1) if timesteps > 1 else 0.0
        beta = (start + (end - start) * frac) ** 2
        running *= 1.0 - beta
        alphas_cumprod.append(running)
    sigmas = tuple(math.sqrt((1.0 - ac) / ac) for ac in alphas_cumprod)
    if zsnr:
        sigmas = _rescale_zero_terminal_snr_sigmas(sigmas)
    return sigmas


def _rescale_zero_terminal_snr_sigmas(
    sigmas: tuple[float, ...],
) -> tuple[float, ...]:
    """Zero-terminal-SNR rescale (comfy/model_sampling.py
    rescale_zero_terminal_snr_sigmas @ b78cec87)."""
    alphas_bar_sqrt = [math.sqrt(1.0 / (s * s + 1.0)) for s in sigmas]
    first = alphas_bar_sqrt[0]
    last = alphas_bar_sqrt[-1]
    scale = first / (first - last)
    alphas_bar = [((a - last) * scale) ** 2 for a in alphas_bar_sqrt]
    alphas_bar[-1] = 4.8973451890853435e-08
    return tuple(math.sqrt((1.0 - ab) / ab) for ab in alphas_bar)


@dataclass(frozen=True)
class DiscreteSigmas:
    """A table-backed sigma space (ModelSamplingDiscrete @ b78cec87).

    ``sigma()`` interpolates linearly in log-sigma between adjacent
    table entries; ``timestep()`` returns the index whose log-sigma is
    nearest (both exactly as the reference).
    """

    entries: tuple[float, ...]
    _log: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.entries) < 2:
            raise ValueError("a discrete sigma table needs at least 2 entries")
        if any(s <= 0 for s in self.entries):
            raise ValueError("sigma table entries must be positive")
        if any(a >= b for a, b in zip(self.entries, self.entries[1:], strict=False)):
            raise ValueError("sigma table must be strictly ascending")
        object.__setattr__(self, "_log", tuple(math.log(s) for s in self.entries))

    @classmethod
    def linear_beta(
        cls,
        *,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        timesteps: int = 1000,
        zsnr: bool = False,
    ) -> DiscreteSigmas:
        return cls(
            linear_beta_sigmas(
                linear_start=linear_start,
                linear_end=linear_end,
                timesteps=timesteps,
                zsnr=zsnr,
            )
        )

    @property
    def sigma_min(self) -> float:
        return self.entries[0]

    @property
    def sigma_max(self) -> float:
        return self.entries[-1]

    @property
    def table(self) -> tuple[float, ...] | None:
        return self.entries

    def sigma(self, timestep: float) -> float:
        t = min(max(timestep, 0.0), len(self.entries) - 1)
        low = math.floor(t)
        high = math.ceil(t)
        w = t - low
        return math.exp((1.0 - w) * self._log[low] + w * self._log[high])

    def timestep(self, sigma: float) -> float:
        log_sigma = math.log(sigma)
        return float(
            min(
                range(len(self._log)),
                key=lambda i: abs(log_sigma - self._log[i]),
            )
        )

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return SIGMA_PERCENT_ZERO
        if percent >= 1.0:
            return 0.0
        return self.sigma((1.0 - percent) * (len(self.entries) - 1))


@dataclass(frozen=True)
class FlowSigmas:
    """SD3-style discrete flow space (ModelSamplingDiscreteFlow
    @ b78cec87): sigma(t) = time_snr_shift(shift, t / multiplier)."""

    shift: float = 1.0
    multiplier: float = 1000.0
    timesteps: int = 1000

    def __post_init__(self) -> None:
        if self.timesteps < 1:
            raise ValueError("timesteps must be >= 1")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")

    @property
    def sigma_min(self) -> float:
        return self.sigma(self.multiplier / self.timesteps)

    @property
    def sigma_max(self) -> float:
        return self.sigma(self.multiplier)

    @property
    def table(self) -> tuple[float, ...] | None:
        return tuple(
            self.sigma((i / self.timesteps) * self.multiplier) for i in range(1, self.timesteps + 1)
        )

    def sigma(self, timestep: float) -> float:
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def timestep(self, sigma: float) -> float:
        return sigma * self.multiplier

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


@dataclass(frozen=True)
class FluxFlowSigmas:
    """Flux exponential flow space (ModelSamplingFlux @ b78cec87):
    sigma(t) = flux_time_shift(shift, 1.0, t), t in (0, 1]."""

    shift: float = 1.15
    timesteps: int = 10000

    def __post_init__(self) -> None:
        if self.timesteps < 1:
            raise ValueError("timesteps must be >= 1")

    @property
    def sigma_min(self) -> float:
        return self.sigma(1.0 / self.timesteps)

    @property
    def sigma_max(self) -> float:
        return self.sigma(1.0)

    @property
    def table(self) -> tuple[float, ...] | None:
        return tuple(self.sigma(i / self.timesteps) for i in range(1, self.timesteps + 1))

    def sigma(self, timestep: float) -> float:
        return flux_time_shift(self.shift, 1.0, timestep)

    def timestep(self, sigma: float) -> float:
        return sigma

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


@dataclass(frozen=True)
class ContinuousEDMSigmas:
    """Continuous EDM space (ModelSamplingContinuousEDM @ b78cec87):
    timestep = 0.25 * ln(sigma). The reference materializes a
    1000-entry log-spaced table "for compatibility with some
    schedulers"; so does this."""

    min_sigma: float = 0.002
    max_sigma: float = 120.0
    timesteps: int = 1000

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_sigma) or not math.isfinite(self.max_sigma):
            raise ValueError("sigma bounds must be finite")
        if self.min_sigma <= 0 or self.max_sigma <= self.min_sigma:
            raise ValueError("need 0 < min_sigma < max_sigma")
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")

    @property
    def sigma_min(self) -> float:
        return self.min_sigma

    @property
    def sigma_max(self) -> float:
        return self.max_sigma

    @property
    def table(self) -> tuple[float, ...] | None:
        lo = math.log(self.min_sigma)
        hi = math.log(self.max_sigma)
        n = self.timesteps
        return tuple(math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n))

    def sigma(self, timestep: float) -> float:
        return math.exp(timestep / 0.25)

    def timestep(self, sigma: float) -> float:
        return 0.25 * math.log(sigma)

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return SIGMA_PERCENT_ZERO
        if percent >= 1.0:
            return 0.0
        lo = math.log(self.min_sigma)
        hi = math.log(self.max_sigma)
        return math.exp((hi - lo) * (1.0 - percent) + lo)


__all__ = [
    "ContinuousEDMSigmas",
    "DiscreteSigmas",
    "FlowSigmas",
    "FluxFlowSigmas",
    "SIGMA_PERCENT_ZERO",
    "SigmaSpace",
    "flux_time_shift",
    "linear_beta_sigmas",
    "time_snr_shift",
]
