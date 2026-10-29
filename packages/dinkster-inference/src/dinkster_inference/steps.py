"""Sigma selection for one sampling run: the KSampler step logic.

Ports comfy/samplers.py KSampler.calculate_sigmas / KSampler.set_steps
and Sampler.max_denoise @ b78cec87 as pure functions over registered
scheduler descriptors: the denoise-fraction tail trim (img2img
strength), the discard-penultimate-sigma correction some solvers need
(declared as SamplerDescriptor.discard_penultimate instead of the
reference's hardcoded name set), and the "is this a full denoise from
pure noise" predicate that noise_scaling consumes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .sampling import SchedulerDescriptor
from .spaces import SigmaSpace

_FULL_DENOISE = 0.9999
"""Above this, KSampler.set_steps treats denoise as 1.0 (@ b78cec87)."""


def sampling_sigmas(
    scheduler: SchedulerDescriptor,
    space: SigmaSpace,
    steps: int,
    *,
    denoise: float | None = None,
    discard_penultimate: bool = False,
) -> tuple[float, ...]:
    """The sigmas one sampling run walks (KSampler.calculate_sigmas +
    set_steps @ b78cec87).

    ``denoise`` is the img2img strength: None or > 0.9999 keeps the
    full schedule, <= 0 yields no steps at all (the run returns the
    latent untouched), anything between schedules ``int(steps /
    denoise)`` steps and keeps only the last ``steps + 1`` sigmas.
    ``discard_penultimate`` is the correction declared by solvers like
    dpm_2: one extra step is scheduled and the penultimate sigma
    dropped (KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS @ b78cec87).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if denoise is not None and denoise <= 0.0:
        return ()
    if denoise is None or denoise > _FULL_DENOISE:
        return _schedule(scheduler, space, steps, discard_penultimate)
    total = int(steps / denoise)
    sigmas = _schedule(scheduler, space, total, discard_penultimate)
    return sigmas[-(steps + 1) :]


def _schedule(
    scheduler: SchedulerDescriptor,
    space: SigmaSpace,
    steps: int,
    discard_penultimate: bool,
) -> tuple[float, ...]:
    if discard_penultimate:
        sigmas = tuple(scheduler.make_sigmas(steps + 1, space))
        return (*sigmas[:-2], sigmas[-1])
    return tuple(scheduler.make_sigmas(steps, space))


def max_denoise(sigma_max: float, sigmas: Sequence[float]) -> bool:
    """Whether the run starts at (or above) the model's maximum noise
    level, i.e. denoises from pure noise (Sampler.max_denoise
    @ b78cec87). Feeds noise_scaling's ``max_denoise`` flag."""
    if not sigmas:
        raise ValueError("sigmas must be nonempty")
    sigma = float(sigmas[0])
    return math.isclose(sigma_max, sigma, rel_tol=1e-05) or sigma > sigma_max


__all__ = [
    "max_denoise",
    "sampling_sigmas",
]
