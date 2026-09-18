"""Sigma schedules: pure functions from (steps, space) to sigmas.

Ports of ComfyUI's nine registered schedules (comfy/samplers.py
SCHEDULER_HANDLERS + comfy/k_diffusion/sampling.py get_sigmas_karras /
get_sigmas_exponential @ b78cec87), against the SigmaSpace protocol
instead of a model_sampling object. Each returns descending sigmas
ending at 0.0 and is deterministic pure float math.

Float64 here vs float32 tensor kernels in the reference: values agree
within ~3e-6 relative, NOT bitwise - fine for value math, but
brownian-tree noise streams need bit-exact reference sigmas, so the
executing runtimes override karras, exponential, normal, sgm_uniform,
linear_quadratic, and kl_optimal with torch reference-kernel ports
(dinkster_inference_torch.schedules); the table-index schedules (simple,
ddim_uniform, beta) select entries from the space's sigma table and
stay within ~1 float32 ulp of the reference here (the reference reads
a float32 table, this reads the float64 SigmaSpace table), which is
inside the brownian tree's 1e-6 cache grid.

The beta schedule needs the beta-distribution quantile function; the
reference reaches for scipy.stats.beta.ppf, which is the regularized
incomplete beta inverse. A private inverse is implemented here
(Lentz continued fraction + bisection) so dinkster-inference stays free
of scipy; goldens pin it against scipy's values on the schedule's
parameter domain. It is deliberately NOT exported: the bisection
terminates on absolute x-width, which is exact enough for selecting
indices in ~1000-entry sigma tables but not a general scipy
replacement (tail-concentrated distributions lose precision).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Any

from .registry import Registry
from .sampling import (
    SchedulerDescriptor,
    catalog_descriptor_rebuild,
    catalog_descriptor_snapshot,
)
from .spaces import SigmaSpace

_ZERO_TOL = 0.00001  # reference abs_tol for "this sigma is zero"


def _require_table(space: SigmaSpace, schedule: str) -> tuple[float, ...]:
    table = space.table
    if table is None:
        raise ValueError(
            f"the {schedule} schedule needs a discrete sigma table, "
            "which this sigma space does not have"
        )
    if len(table) < 2:
        raise ValueError(
            f"the {schedule} schedule needs a sigma table with at least 2 entries, got {len(table)}"
        )
    return table


def _require_steps(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")


def normal_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """Uniform in timestep (comfy/samplers.py normal_scheduler
    @ b78cec87)."""
    return _normal(steps, space, sgm=False)


def sgm_uniform_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """SGM variant: one extra point, last dropped (normal_scheduler
    with sgm=True @ b78cec87)."""
    return _normal(steps, space, sgm=True)


def _normal(steps: int, space: SigmaSpace, *, sgm: bool) -> tuple[float, ...]:
    _require_steps(steps)
    start = space.timestep(space.sigma_max)
    end = space.timestep(space.sigma_min)

    append_zero = True
    if sgm:
        points = steps + 1
        timesteps = [start + (end - start) * i / (points - 1) for i in range(points)][:-1]
    else:
        if math.isclose(space.sigma(end), 0.0, abs_tol=_ZERO_TOL):
            steps += 1
            append_zero = False
        timesteps = (
            [start + (end - start) * i / (steps - 1) for i in range(steps)]
            if steps > 1
            else [start]
        )

    sigs = [space.sigma(t) for t in timesteps]
    if append_zero:
        sigs.append(0.0)
    return tuple(sigs)


def simple_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """Evenly strided walk down the discrete table (comfy/samplers.py
    simple_scheduler @ b78cec87)."""
    _require_steps(steps)
    table = _require_table(space, "simple")
    ss = len(table) / steps
    sigs = [table[-(1 + int(x * ss))] for x in range(steps)]
    sigs.append(0.0)
    return tuple(sigs)


def ddim_uniform_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """DDIM-style uniform stride from index 1 (comfy/samplers.py
    ddim_scheduler @ b78cec87)."""
    _require_steps(steps)
    table = _require_table(space, "ddim_uniform")
    if math.isclose(table[1], 0.0, abs_tol=_ZERO_TOL):
        steps += 1
        sigs: list[float] = []
    else:
        sigs = [0.0]
    ss = max(len(table) // steps, 1)
    x = 1
    while x < len(table):
        sigs.append(table[x])
        x += ss
    return tuple(reversed(sigs))


def karras_schedule(steps: int, space: SigmaSpace, *, rho: float = 7.0) -> tuple[float, ...]:
    """Karras et al. (2022) noise schedule (comfy/k_diffusion/
    sampling.py get_sigmas_karras @ b78cec87).

    Float64 here vs float32 tensor kernels in the reference: values
    agree within ~3e-6 relative, NOT bitwise. That is fine for value
    math but decorrelates brownian-tree noise streams, whose query
    times must match the reference exactly - SDE replay uses the
    torch-exact override (dinkster_inference_torch.schedules) instead.
    """
    _require_steps(steps)
    min_inv_rho = space.sigma_min ** (1.0 / rho)
    max_inv_rho = space.sigma_max ** (1.0 / rho)
    sigs = [
        (max_inv_rho + (i / (steps - 1) if steps > 1 else 0.0) * (min_inv_rho - max_inv_rho)) ** rho
        for i in range(steps)
    ]
    sigs.append(0.0)
    return tuple(sigs)


def exponential_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """Uniform in log-sigma (comfy/k_diffusion/sampling.py
    get_sigmas_exponential @ b78cec87)."""
    _require_steps(steps)
    lo = math.log(space.sigma_min)
    hi = math.log(space.sigma_max)
    sigs = [
        math.exp(hi + (lo - hi) * (i / (steps - 1) if steps > 1 else 0.0)) for i in range(steps)
    ]
    sigs.append(0.0)
    return tuple(sigs)


def beta_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> tuple[float, ...]:
    """Beta-distribution timestep spacing, arXiv:2407.12173
    (comfy/samplers.py beta_scheduler @ b78cec87)."""
    _require_steps(steps)
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    table = _require_table(space, "beta")
    total_timesteps = len(table) - 1
    sigs: list[float] = []
    last_t = -1
    for i in range(steps):
        p = 1.0 - i / steps
        t = _round_half_even(_beta_quantile(p, alpha, beta) * total_timesteps)
        if t != last_t:
            sigs.append(table[int(t)])
        last_t = t
    sigs.append(0.0)
    return tuple(sigs)


def beta57_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """Beta spacing with alpha=0.5, beta=0.7, the "beta57" scheduler
    RES4LYF registers globally (RES4LYF __init__.py @ 26036f64:
    partial(comfy.samplers.beta_scheduler, alpha=0.5, beta=0.7)).

    Pure table-index selection like the beta schedule, so this port
    is exact for existing spaces without a torch reference kernel."""
    return beta_schedule(steps, space, alpha=0.5, beta=0.7)


def _f32(value: float) -> float:
    """Round to the nearest float32 (the reference materializes bong
    tangent sigmas with torch.tensor's float64 -> float32 conversion,
    which is this same round-to-nearest-even)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bong_tangent_segment(
    steps: int, slope: float, pivot: int, start: float, end: float
) -> list[float]:
    """One arctangent easing segment (RES4LYF sigmas.py
    get_bong_tangent_sigmas @ 26036f64), expression order preserved."""
    smax = ((2 / math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / math.pi) * math.atan(-slope * ((steps - 1) - pivot)) + 1) / 2
    srange = smax - smin
    sscale = start - end
    return [
        ((((2 / math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale
        + end
        for x in range(steps)
    ]


def bong_tangent_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """Two-stage arctangent easing from 1.0 through 0.5 to 0.0, the
    "bong_tangent" scheduler RES4LYF registers globally (RES4LYF
    sigmas.py bong_tangent_scheduler @ 26036f64, reference defaults).

    The reference receives model_sampling but never reads it, so the
    sigmas are the same 1.0..0.0 tangent shape on every model family;
    this port ignores the space the same way. The reference computes
    the values with Python libm math and materializes them as a
    float32 tensor, so each value here is quantized through float32 -
    bit-identical sigmas without a torch kernel.

    Requires steps >= 2: at steps == 1 the reference's first stage
    degenerates to a single point and its range normalization divides
    by zero; this port refuses loudly instead."""
    del space
    if steps < 2:
        raise ValueError(
            "bong_tangent needs steps >= 2 (the reference normalizes each "
            "tangent stage by its value range, which is zero for the "
            "single-point stage a one-step schedule produces)"
        )
    start, middle, end = 1.0, 0.5, 0.0
    pivot_1, pivot_2 = 0.6, 0.6
    slope_1, slope_2 = 0.2, 0.2

    steps += 2
    midpoint = int((steps * pivot_1 + steps * pivot_2) / 2)
    pivot_1_idx = int(steps * pivot_1)
    pivot_2_idx = int(steps * pivot_2)
    slope_1 = slope_1 / (steps / 40)
    slope_2 = slope_2 / (steps / 40)
    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len

    tan_sigmas_1 = _bong_tangent_segment(stage_1_len, slope_1, pivot_1_idx, start, middle)
    tan_sigmas_2 = _bong_tangent_segment(
        stage_2_len, slope_2, pivot_2_idx - stage_1_len, middle, end
    )
    return tuple(_f32(sigma) for sigma in (*tan_sigmas_1[:-1], *tan_sigmas_2))


def linear_quadratic_schedule(
    steps: int,
    space: SigmaSpace,
    *,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> tuple[float, ...]:
    """Mochi's linear-then-quadratic schedule (comfy/samplers.py
    linear_quadratic_schedule @ b78cec87, from genmoai/models)."""
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
    return tuple(x * space.sigma_max for x in schedule)


def kl_optimal_schedule(steps: int, space: SigmaSpace) -> tuple[float, ...]:
    """KL-optimal spacing, arXiv:2404.04292 (comfy/samplers.py
    kl_optimal_scheduler @ b78cec87, from A1111 PR 15608).

    Requires steps >= 2: the reference divides arange(steps) by
    steps - 1, so steps == 1 yields NaN there; this port refuses it
    loudly instead of inventing a usable one-step schedule."""
    if steps < 2:
        raise ValueError(
            "kl_optimal needs steps >= 2 (the reference formula divides "
            "by steps - 1 and produces NaN for a single step)"
        )
    atan_min = math.atan(space.sigma_min)
    atan_max = math.atan(space.sigma_max)
    sigs = [
        math.tan((i / (steps - 1)) * atan_min + (1.0 - i / (steps - 1)) * atan_max)
        for i in range(steps)
    ]
    sigs.append(0.0)
    return tuple(sigs)


def offset_first_sigma_for_snr(
    sigmas: Sequence[float],
    space: SigmaSpace,
    *,
    flow: bool,
    percent_offset: float = 1e-4,
) -> tuple[float, ...]:
    """Nudge a leading sigma of 1.0 down so flow-model logSNR is finite
    (offset_first_sigma_for_snr @ b78cec87: logit(1) is +inf, so SDE
    solvers on flow/CONST models cannot start exactly at sigma 1).

    The reference applies this inside each SDE solver via model_sampling
    probing; here schedule preparation owns it - call it on the prepared
    schedule before a Brownian/SDE solver when the model is flow. It is
    the identity for non-flow models and for schedules already below 1.
    """
    if len(sigmas) <= 1 or not flow or sigmas[0] < 1.0:
        return tuple(sigmas)
    return (space.percent_to_sigma(percent_offset), *sigmas[1:])


# --------------------------------------------------------------------------
# Scheduler catalog: descriptors with provenance for the nine core
# schedules, replacing comfy/samplers.py SCHEDULER_HANDLERS/
# SCHEDULER_NAMES (@ b78cec87), plus the two schedulers RES4LYF
# registers globally (@ 26036f64). Aliases carry the legacy ComfyUI
# vocabulary so ported workflows resolve; extra parameters (karras rho,
# beta alpha/beta, linear_quadratic threshold_noise) keep their
# reference defaults here - non-default use calls the schedule function
# directly.

DINKSTER_SIMPLE = SchedulerDescriptor(
    id="dinkster.simple",
    display_name="Simple",
    make_sigmas=simple_schedule,
    aliases=("simple",),
)

DINKSTER_SGM_UNIFORM = SchedulerDescriptor(
    id="dinkster.sgm_uniform",
    display_name="SGM uniform",
    make_sigmas=sgm_uniform_schedule,
    aliases=("sgm_uniform",),
)

DINKSTER_KARRAS = SchedulerDescriptor(
    id="dinkster.karras",
    display_name="Karras",
    make_sigmas=karras_schedule,
    aliases=("karras",),
)

DINKSTER_EXPONENTIAL = SchedulerDescriptor(
    id="dinkster.exponential",
    display_name="Exponential",
    make_sigmas=exponential_schedule,
    aliases=("exponential",),
)

DINKSTER_DDIM_UNIFORM = SchedulerDescriptor(
    id="dinkster.ddim_uniform",
    display_name="DDIM uniform",
    make_sigmas=ddim_uniform_schedule,
    aliases=("ddim_uniform",),
)

DINKSTER_BETA = SchedulerDescriptor(
    id="dinkster.beta",
    display_name="Beta",
    make_sigmas=beta_schedule,
    aliases=("beta",),
)

DINKSTER_NORMAL = SchedulerDescriptor(
    id="dinkster.normal",
    display_name="Normal",
    make_sigmas=normal_schedule,
    aliases=("normal",),
)

DINKSTER_LINEAR_QUADRATIC = SchedulerDescriptor(
    id="dinkster.linear_quadratic",
    display_name="Linear quadratic",
    make_sigmas=linear_quadratic_schedule,
    aliases=("linear_quadratic",),
)

DINKSTER_KL_OPTIMAL = SchedulerDescriptor(
    id="dinkster.kl_optimal",
    display_name="KL optimal",
    make_sigmas=kl_optimal_schedule,
    aliases=("kl_optimal",),
)

# The two schedulers RES4LYF registers into SCHEDULER_NAMES globally
# (RES4LYF __init__.py @ 26036f64), in its registration order. The
# aliases carry the exact names ComfyUI workflows serialize.

RES4LYF_BONG_TANGENT = SchedulerDescriptor(
    id="res4lyf.bong_tangent",
    display_name="Bong tangent",
    make_sigmas=bong_tangent_schedule,
    aliases=("bong_tangent",),
)

RES4LYF_BETA57 = SchedulerDescriptor(
    id="res4lyf.beta57",
    display_name="Beta57",
    make_sigmas=beta57_schedule,
    aliases=("beta57",),
)


# Field values of every catalog descriptor, captured at import before
# any caller code runs. The descriptor constants above are published
# API and therefore caller-reachable: a frozen dataclass without slots
# still accepts object.__setattr__, so a shared instance can be
# mutated. The snapshot is deep - primitives by exact type and schedule
# functions as behavior evidence - so builtin_schedulers() neither hands
# back a poisoned constant nor trusts a mutated schedule function.
_CANONICAL_SCHEDULER_FIELDS: tuple[tuple[tuple[str, Any], ...], ...] = tuple(
    catalog_descriptor_snapshot(descriptor)
    for descriptor in (
        DINKSTER_SIMPLE,
        DINKSTER_SGM_UNIFORM,
        DINKSTER_KARRAS,
        DINKSTER_EXPONENTIAL,
        DINKSTER_DDIM_UNIFORM,
        DINKSTER_BETA,
        DINKSTER_NORMAL,
        DINKSTER_LINEAR_QUADRATIC,
        DINKSTER_KL_OPTIMAL,
        RES4LYF_BONG_TANGENT,
        RES4LYF_BETA57,
    )
)


def builtin_schedulers() -> tuple[SchedulerDescriptor, ...]:
    """The ported schedule catalog, in SCHEDULER_HANDLERS order
    (@ b78cec87).

    Constructed fresh on every call from field values captured at
    import: distributed receipt admission trusts this catalog, so it
    must never hand back the caller-reachable descriptor constants."""
    return tuple(
        SchedulerDescriptor(**catalog_descriptor_rebuild(field_values))
        for field_values in _CANONICAL_SCHEDULER_FIELDS
    )


def builtin_scheduler_registry() -> Registry[SchedulerDescriptor]:
    """A fresh registry preloaded with the ported schedule catalog."""
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in builtin_schedulers():
        registry.register(descriptor)
    return registry


def _round_half_even(x: float) -> float:
    """numpy.rint semantics: round half to even (the reference rounds
    beta quantiles with numpy.rint @ b78cec87)."""
    floor = math.floor(x)
    diff = x - floor
    if diff > 0.5:
        return floor + 1.0
    if diff < 0.5:
        return float(floor)
    return float(floor) if floor % 2 == 0 else floor + 1.0


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b), the beta distribution CDF - Lentz's continued
    fraction (Numerical Recipes betai/betacf), accurate to ~1e-14."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b) / a
    return 1.0 - front * _beta_cf(1.0 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _beta_quantile(p: float, a: float, b: float) -> float:
    """The beta distribution quantile used for schedule table selection.

    Closed forms preserve the reference's operation order at symmetric
    midpoint ties. Other inputs use bisection on the CDF above, which is
    not a general scipy.stats.beta.ppf replacement for concentrated tails.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if a == 1.0 and b == 1.0:
        return p
    if a == 0.5 and b == 0.5:
        if p == 0.5:
            # Pinned SciPy 1.18.1 returns two ULPs below the exact midpoint.
            return math.nextafter(math.nextafter(0.5, 0.0), 0.0)
        value = math.sin(p * math.pi / 2.0)
        return value * value
    if a == b and a < 1.0 and p == 0.5:
        return (1.0 - a) / (2.0 - a - b)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _regularized_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-16:
            break
    return (lo + hi) / 2.0


__all__ = [
    "DINKSTER_BETA",
    "DINKSTER_DDIM_UNIFORM",
    "DINKSTER_EXPONENTIAL",
    "DINKSTER_KARRAS",
    "DINKSTER_KL_OPTIMAL",
    "DINKSTER_LINEAR_QUADRATIC",
    "DINKSTER_NORMAL",
    "DINKSTER_SGM_UNIFORM",
    "DINKSTER_SIMPLE",
    "beta_schedule",
    "builtin_scheduler_registry",
    "builtin_schedulers",
    "ddim_uniform_schedule",
    "exponential_schedule",
    "karras_schedule",
    "kl_optimal_schedule",
    "linear_quadratic_schedule",
    "normal_schedule",
    "offset_first_sigma_for_snr",
    "sgm_uniform_schedule",
    "simple_schedule",
]
