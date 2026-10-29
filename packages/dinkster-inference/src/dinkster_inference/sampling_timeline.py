"""Invocation-local realized sampling timelines.

Declarations are resolved against the exact sigma sequence executed by the
solver. One immutable row is installed at each outer step boundary so every
consumer in that step observes the same attention plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Literal

from .control_gain import ExecutedStepAnchors, ExecutedTimeline
from .recipe import canonical_float

AttentionModifier = Literal[
    "skip_softmax",
    "sol_conditioning_exact_kv",
]
AttentionPlanProvider = Literal["sdpa", "dinkster_kitchen_int8", "sage", "sol"]
AttentionScheduleProvider = Literal["dinkster_kitchen_int8", "sage", "sol"]
CurveInterpolation = Literal["linear", "monotone_cubic"]

_SCHEDULE_DOMAIN = "dinkster.sampling-timeline.schedule.v1"
_TIMELINE_DOMAIN = "dinkster.sampling-timeline.realized.v1"
_PROVIDERS = frozenset(("dinkster_kitchen_int8", "sage", "sol"))
_SOL_CONDITIONING_MODIFIERS = frozenset(("sol_conditioning_exact_kv",))
_MODIFIERS = frozenset(("skip_softmax",)) | _SOL_CONDITIONING_MODIFIERS


def _validate_digest(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{name} must be a lowercase sha256 hex string")


def _canonical_digest(domain: str, facts: object) -> str:
    payload = json.dumps(
        {"domain": domain, "facts": facts},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplingParameterCurve:
    """Inference-owned immutable snapshot of a ``dinkster.curve`` value."""

    points: tuple[tuple[float, float], ...]
    interpolation: CurveInterpolation = "linear"
    _tangents: tuple[tuple[float, float], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.points) is not tuple or not 1 <= len(self.points) <= 4096:
            raise TypeError("sampling parameter curve must contain 1 to 4096 points")
        normalized: list[tuple[float, float]] = []
        previous: float | None = None
        for index, point in enumerate(self.points):
            if type(point) is not tuple or len(point) != 2:
                raise TypeError(f"sampling parameter curve point {index} must be a pair")
            position, value = point
            if (
                type(position) is not float
                or not math.isfinite(position)
                or type(value) is not float
                or not math.isfinite(value)
            ):
                raise TypeError("sampling parameter curve points must be finite exact floats")
            if previous is not None and position <= previous:
                raise ValueError("sampling parameter curve positions must be strictly increasing")
            normalized.append((position, value))
            previous = position
        if self.interpolation not in ("linear", "monotone_cubic"):
            raise ValueError("sampling parameter curve interpolation is unsupported")
        object.__setattr__(self, "points", tuple(normalized))
        object.__setattr__(
            self,
            "_tangents",
            self._monotone_tangents() if self.interpolation == "monotone_cubic" else (),
        )

    def _monotone_tangents(self) -> tuple[tuple[float, float], ...]:
        if len(self.points) == 1:
            return ()
        with localcontext() as context:
            context.prec = 50
            points = tuple(
                (Decimal.from_float(position), Decimal.from_float(value))
                for position, value in self.points
            )
            secants = [
                (right[1] - left[1]) / (right[0] - left[0])
                for left, right in zip(points, points[1:], strict=False)
            ]
            slopes = [secants[0]]
            slopes.extend(
                Decimal(0) if left * right <= 0 else (left + right) / 2
                for left, right in zip(secants, secants[1:], strict=False)
            )
            slopes.append(secants[-1])
            for index, secant in enumerate(secants):
                if secant == 0:
                    slopes[index] = slopes[index + 1] = Decimal(0)
                    continue
                alpha = slopes[index] / secant
                beta = slopes[index + 1] / secant
                magnitude = alpha * alpha + beta * beta
                if magnitude > 9:
                    scale = Decimal(3) / magnitude.sqrt()
                    slopes[index] = scale * alpha * secant
                    slopes[index + 1] = scale * beta * secant
            return tuple(
                (float(slopes[index] / secant), float(slopes[index + 1] / secant))
                if secant != 0
                else (0.0, 0.0)
                for index, secant in enumerate(secants)
            )

    def evaluate(self, position: float) -> float:
        if type(position) is not float or not math.isfinite(position):
            raise TypeError("sampling parameter curve position must be a finite exact float")
        if position <= self.points[0][0]:
            return self.points[0][1]
        if position >= self.points[-1][0]:
            return self.points[-1][1]
        right = bisect_left(self.points, position, key=lambda point: point[0])
        right_position, right_value = self.points[right]
        left_position, left_value = self.points[right - 1]
        span = right_position - left_position
        if math.isfinite(span):
            amount = (position - left_position) / span
        else:
            scale = max(abs(left_position), abs(right_position), abs(position), 1.0)
            amount = (position / scale - left_position / scale) / (
                right_position / scale - left_position / scale
            )
        if self.interpolation == "monotone_cubic":
            amount2 = amount * amount
            amount3 = amount2 * amount
            left_tangent, right_tangent = self._tangents[right - 1]
            value_scale = max(abs(left_value), abs(right_value))
            if value_scale == 0:
                return 0.0
            scaled_left = left_value / value_scale
            scaled_right = right_value / value_scale
            scaled_delta = scaled_right - scaled_left
            value = (
                (2 * amount3 - 3 * amount2 + 1) * scaled_left
                + (amount3 - 2 * amount2 + amount) * left_tangent * scaled_delta
                + (-2 * amount3 + 3 * amount2) * scaled_right
                + (amount3 - amount2) * right_tangent * scaled_delta
            ) * value_scale
        else:
            value = (1.0 - amount) * left_value + amount * right_value
        if not math.isfinite(value):
            raise ValueError("sampling parameter curve produced a non-finite value")
        return value

    @property
    def facts(self) -> dict[str, object]:
        return {
            "interpolation": self.interpolation,
            "points": [
                [canonical_float(position), canonical_float(value)]
                for position, value in self.points
            ],
        }


@dataclass(frozen=True, slots=True)
class AttentionModifierSchedule:
    """One discrete piecewise-constant attention modifier window."""

    modifier: AttentionModifier
    start_percent: float
    end_percent: float

    def __post_init__(self) -> None:
        if self.modifier not in _MODIFIERS:
            raise ValueError("sampling timeline attention modifier is unsupported")
        if (
            type(self.start_percent) is not float
            or not math.isfinite(self.start_percent)
            or type(self.end_percent) is not float
            or not math.isfinite(self.end_percent)
            or not 0.0 <= self.start_percent < self.end_percent <= 1.0
        ):
            raise ValueError("attention modifier requires 0 <= start_percent < end_percent <= 1")

    @property
    def facts(self) -> dict[str, str]:
        return {
            "modifier": self.modifier,
            "startPercent": canonical_float(self.start_percent),
            "endPercent": canonical_float(self.end_percent),
        }


@dataclass(frozen=True, slots=True)
class SamplingTimelineSchedule:
    """A discrete provider window with optional categorical modifier windows."""

    approximate_provider: AttentionScheduleProvider
    start_percent: float
    end_percent: float
    sol_tau: SamplingParameterCurve | None = None
    attention_modifiers: tuple[AttentionModifierSchedule, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.approximate_provider not in _PROVIDERS:
            raise ValueError("sampling timeline approximate provider is unsupported")
        if (
            type(self.start_percent) is not float
            or not math.isfinite(self.start_percent)
            or type(self.end_percent) is not float
            or not math.isfinite(self.end_percent)
            or not 0.0 <= self.start_percent < self.end_percent <= 1.0
        ):
            raise ValueError("sampling timeline requires 0 <= start_percent < end_percent <= 1")
        if self.sol_tau is not None:
            if self.approximate_provider != "sol":
                raise ValueError("only the Sol provider accepts a numeric parameter curve")
            if type(self.sol_tau) is not SamplingParameterCurve:
                raise TypeError("Sol tau must be an exact SamplingParameterCurve")
        if type(self.attention_modifiers) is not tuple or any(
            type(modifier) is not AttentionModifierSchedule for modifier in self.attention_modifiers
        ):
            raise TypeError("attention modifiers must be an exact AttentionModifierSchedule tuple")
        modifier_names = tuple(modifier.modifier for modifier in self.attention_modifiers)
        if len(modifier_names) != len(set(modifier_names)):
            raise ValueError("attention modifier schedules must have unique names")
        conditioning_modifiers = _SOL_CONDITIONING_MODIFIERS.intersection(modifier_names)
        if conditioning_modifiers and self.approximate_provider != "sol":
            raise ValueError("conditioning sink modifiers require the Sol provider")
        if any(
            modifier.modifier in _SOL_CONDITIONING_MODIFIERS
            and (
                modifier.start_percent < self.start_percent
                or modifier.end_percent > self.end_percent
            )
            for modifier in self.attention_modifiers
        ):
            raise ValueError("conditioning sink modifiers must stay inside the Sol provider window")
        modifiers = tuple(sorted(self.attention_modifiers, key=lambda modifier: modifier.modifier))
        object.__setattr__(self, "attention_modifiers", modifiers)
        facts: dict[str, object] = {
            "approximateProvider": self.approximate_provider,
            "startPercent": canonical_float(self.start_percent),
            "endPercent": canonical_float(self.end_percent),
            "solTau": None if self.sol_tau is None else self.sol_tau.facts,
        }
        if modifiers:
            facts["attentionModifiers"] = [modifier.facts for modifier in modifiers]
        object.__setattr__(self, "digest", _canonical_digest(_SCHEDULE_DOMAIN, facts))


@dataclass(frozen=True, slots=True)
class AttentionPlan:
    """One immutable provider, modifier, and numeric-parameter decision."""

    provider: AttentionPlanProvider
    modifiers: tuple[AttentionModifier, ...]
    parameters: MappingProxyType[str, float]

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS | {"sdpa"}:
            raise ValueError("attention plan provider is unsupported")
        if type(self.modifiers) is not tuple or any(
            modifier not in _MODIFIERS for modifier in self.modifiers
        ):
            raise TypeError("attention plan modifiers must be a recognized modifier tuple")
        if len(self.modifiers) != len(set(self.modifiers)):
            raise ValueError("attention plan modifiers must be unique")
        conditioning_modifiers = _SOL_CONDITIONING_MODIFIERS.intersection(self.modifiers)
        if conditioning_modifiers and self.provider != "sol":
            raise ValueError("realized conditioning sink modifiers require the Sol provider")
        if type(self.parameters) is not MappingProxyType:
            raise TypeError("realized attention parameters must be immutable")
        parameters = dict(self.parameters)
        for name, value in parameters.items():
            if type(name) is not str or type(value) is not float or not math.isfinite(value):
                raise TypeError("realized attention parameters must be finite named floats")
        if self.provider != "sol" and parameters:
            raise ValueError("only realized Sol rows may carry attention parameters")
        if self.provider == "sol" and set(parameters) != {"sol.tau"}:
            raise ValueError("realized Sol rows require exactly the sol.tau parameter")
        object.__setattr__(self, "modifiers", tuple(sorted(self.modifiers)))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, slots=True)
class RealizedSamplingRow:
    """One immutable outer-step decision shared by all sampling features."""

    anchors: ExecutedStepAnchors
    attention_plan: AttentionPlan
    schedule_digest: str

    def __post_init__(self) -> None:
        if type(self.anchors) is not ExecutedStepAnchors:
            raise TypeError("realized sampling row requires exact executed anchors")
        if type(self.attention_plan) is not AttentionPlan:
            raise TypeError("realized sampling row requires an exact AttentionPlan")
        _validate_digest(self.schedule_digest, "realized row schedule digest")


@dataclass(frozen=True, slots=True)
class RealizedSamplingTimeline:
    schedule_digest: str
    executed: ExecutedTimeline
    rows: tuple[RealizedSamplingRow, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_digest(self.schedule_digest, "realized timeline schedule digest")
        if type(self.executed) is not ExecutedTimeline:
            raise TypeError("realized sampling timeline requires an exact ExecutedTimeline")
        if type(self.rows) is not tuple or len(self.rows) != len(self.executed.steps):
            raise TypeError("realized sampling rows must cover the exact executed timeline")
        if any(row.schedule_digest != self.schedule_digest for row in self.rows):
            raise ValueError("realized sampling rows must share the timeline schedule digest")
        if any(
            row.anchors != step for row, step in zip(self.rows, self.executed.steps, strict=True)
        ):
            raise ValueError("realized sampling row anchors must match the executed timeline")
        facts: list[dict[str, object]] = []
        for row in self.rows:
            row_facts: dict[str, object] = {
                "step": row.anchors.step_index,
                "sigma": canonical_float(row.anchors.sigma),
                "progress": canonical_float(row.anchors.progress),
                "attentionProvider": row.attention_plan.provider,
                "attentionParameters": {
                    name: canonical_float(value)
                    for name, value in row.attention_plan.parameters.items()
                },
            }
            if row.attention_plan.modifiers:
                row_facts["attentionModifiers"] = list(row.attention_plan.modifiers)
            facts.append(row_facts)
        object.__setattr__(
            self,
            "digest",
            _canonical_digest(
                _TIMELINE_DOMAIN,
                {"scheduleDigest": self.schedule_digest, "rows": facts},
            ),
        )


def executed_sampling_timeline(sigmas: tuple[float, ...]) -> ExecutedTimeline:
    """Build the canonical outer-step anchors for an executed sigma sequence."""
    if type(sigmas) is not tuple or len(sigmas) < 2:
        raise ValueError("an executed sampling timeline requires at least one outer step")
    steps = len(sigmas) - 1
    return ExecutedTimeline(
        tuple(
            ExecutedStepAnchors(
                index,
                float(sigma),
                index / (steps - 1) if steps > 1 else 0.0,
            )
            for index, sigma in enumerate(sigmas[:-1])
        )
    )


def realize_sampling_timeline(
    schedule: SamplingTimelineSchedule, sigmas: tuple[float, ...]
) -> RealizedSamplingTimeline:
    if type(schedule) is not SamplingTimelineSchedule:
        raise TypeError("sampling timeline schedule must be exact")
    executed = executed_sampling_timeline(sigmas)
    rows: list[RealizedSamplingRow] = []
    for anchors in executed.steps:
        active = schedule.start_percent <= anchors.progress <= schedule.end_percent
        provider = schedule.approximate_provider if active else "sdpa"
        parameters: dict[str, float] = {}
        if provider == "sol":
            tau = 1.0 if schedule.sol_tau is None else schedule.sol_tau.evaluate(anchors.progress)
            if tau <= 0.0:
                raise ValueError("realized Sol tau must be greater than zero")
            parameters["sol.tau"] = tau
        modifiers: tuple[AttentionModifier, ...] = tuple(
            modifier.modifier
            for modifier in schedule.attention_modifiers
            if modifier.start_percent <= anchors.progress <= modifier.end_percent
        )
        rows.append(
            RealizedSamplingRow(
                anchors,
                AttentionPlan(provider, modifiers, MappingProxyType(parameters)),
                schedule.digest,
            )
        )
    return RealizedSamplingTimeline(schedule.digest, executed, tuple(rows))


_active_timeline: ContextVar[RealizedSamplingTimeline | None] = ContextVar(
    "dinkster_realized_sampling_timeline", default=None
)
_active_row: ContextVar[RealizedSamplingRow | None] = ContextVar(
    "dinkster_realized_sampling_row", default=None
)


def current_realized_sampling_timeline() -> RealizedSamplingTimeline | None:
    return _active_timeline.get()


def current_realized_sampling_row() -> RealizedSamplingRow | None:
    return _active_row.get()


def require_realized_sampling_step(step_index: int, sigma: float, progress: float) -> None:
    """Bind another scheduled feature to the engine-installed row when present."""
    row = current_realized_sampling_row()
    if row is None:
        return
    if (row.anchors.step_index, row.anchors.sigma, row.anchors.progress) != (
        step_index,
        sigma,
        progress,
    ):
        raise RuntimeError("scheduled feature row diverged from the realized sampling timeline")


@contextmanager
def use_realized_sampling_timeline(timeline: RealizedSamplingTimeline):
    """Install a timeline and return its only allowed row activation callback."""
    if type(timeline) is not RealizedSamplingTimeline:
        raise TypeError("realized sampling timeline must be exact")
    timeline_token = _active_timeline.set(timeline)
    row_token = _active_row.set(None)

    def activate(step_index: int) -> None:
        if _active_timeline.get() is not timeline:
            raise RuntimeError("realized sampling row activation escaped its timeline scope")
        if type(step_index) is not int or not 0 <= step_index < len(timeline.rows):
            raise IndexError("realized sampling step index is out of range")
        _active_row.set(timeline.rows[step_index])

    try:
        yield activate
    finally:
        _active_row.reset(row_token)
        _active_timeline.reset(timeline_token)


__all__ = [
    "AttentionModifier",
    "AttentionModifierSchedule",
    "AttentionPlan",
    "AttentionPlanProvider",
    "AttentionScheduleProvider",
    "RealizedSamplingRow",
    "RealizedSamplingTimeline",
    "SamplingParameterCurve",
    "SamplingTimelineSchedule",
    "current_realized_sampling_row",
    "current_realized_sampling_timeline",
    "executed_sampling_timeline",
    "realize_sampling_timeline",
    "require_realized_sampling_step",
    "use_realized_sampling_timeline",
]
