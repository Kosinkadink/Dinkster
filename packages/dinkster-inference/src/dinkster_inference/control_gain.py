"""Torch-free declared control gain schedules and exact realization.

A ``ContributionGain`` declares one timeline gain schedule (constant,
canonical keyframe curve, or direct table over the exact executed
timeline), one global gain, site and guidance-lane gain maps keyed by
opaque resolved identifiers, and an ordered effect-mask digest tuple
(empty for the all-one field). Site-selector resolution and lane
vocabulary checks are owned by the compilation boundary that holds
those vocabularies; keys here are validated identifiers only.

``realize_gain_table`` runs after the exact executed sigma table is
fixed and produces one ``RealizedGainTable`` row per executed step:
the three canonical timeline anchors, the analytic segment, any
assigned keyframe, the resolved timeline/global/site/lane scalars,
and the ordered effect-mask digests. Keyframe hold rows are assigned
by the ``monotone-nearest.v1`` profile: each keyframe receives at
most one contiguous run of distinct rows, runs preserve declared
order (state monotonicity), and the assignment maximizes keyframes
with at least one row, then assigned occurrences, then minimizes
total absolute anchor distance in the declared coordinate, breaking
remaining ties by the lexicographically earliest (keyframe
declaration index, step_index) assignment.

Scalar gain values interpolate between normalized keyframe states;
effect-mask tuples are piecewise-held per anchor segment. A keyframe
replacement map must cover exactly the declared site or lane key set,
so per-key interpolation is total. ``zero.v1`` endpoints zero the
timeline factor and hold the nearest keyframe's remaining normalized
state. Sigma-coordinate anchors must strictly descend, matching
execution order; progress and step_index anchors strictly ascend.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from .recipe import canonical_float

__all__ = [
    "MONOTONE_NEAREST_ASSIGNMENT",
    "ConstantGainCurve",
    "ContributionGain",
    "DirectGainTableCurve",
    "ExecutedStepAnchors",
    "ExecutedTimeline",
    "GainEndpoint",
    "GainInterpolation",
    "GainKeyframe",
    "GainOmission",
    "GainRefusal",
    "GainRefusalCode",
    "KeyframeGainCurve",
    "KeyframeHoldPolicy",
    "RealizedGainRow",
    "RealizedGainTable",
    "TimelineCoordinate",
    "TimelineGainCurve",
    "contribution_gain_slot_facts",
    "realize_gain_table",
]

_SCHEDULE_DOMAIN = "dinkster.control-gain.schedule.v1"
_TABLE_DOMAIN = "dinkster.control-gain.realized-table.v1"

MONOTONE_NEAREST_ASSIGNMENT = "monotone-nearest.v1"


class GainRefusalCode(StrEnum):
    INVALID_GAIN_SCHEDULE = "invalid_gain_schedule"
    UNSUPPORTED_GAIN_PROFILE = "unsupported_gain_profile"
    UNSATISFIED_KEYFRAME_MINIMUM = "unsatisfied_keyframe_minimum"


class GainRefusal(ValueError):
    def __init__(self, code: GainRefusalCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class TimelineCoordinate(StrEnum):
    PROGRESS = "progress"
    SIGMA = "sigma"
    STEP_INDEX = "step_index"


class GainInterpolation(StrEnum):
    HOLD = "hold.v1"
    LINEAR = "linear.v1"
    SMOOTHSTEP = "smoothstep.v1"


class GainEndpoint(StrEnum):
    CLAMP = "clamp.v1"
    ZERO = "zero.v1"


class GainOmission(StrEnum):
    INHERIT = "inherit"
    RESET = "reset"


class KeyframeHoldPolicy(StrEnum):
    BEST_EFFORT = "best_effort"
    HARD = "hard"


def _refuse(detail: str) -> GainRefusal:
    return GainRefusal(GainRefusalCode.INVALID_GAIN_SCHEDULE, detail)


def _require_finite_float(name: str, value: float) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise _refuse(f"{name} must be a finite exact float")


def _require_profile(name: str, value: object, kind: type[StrEnum]) -> None:
    if type(value) is not kind:
        raise GainRefusal(
            GainRefusalCode.UNSUPPORTED_GAIN_PROFILE,
            f"{name} must be an exact {kind.__name__} value",
        )


def _require_identifier(name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise _refuse(f"{name} must be a non-empty string")
    if "\n" in value or "=" in value:
        raise _refuse(f"{name} must not contain newlines or '='")


def _is_sha256_hex(digest: object) -> bool:
    return (
        type(digest) is str
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def _require_mask_digests(name: str, digests: tuple[str, ...]) -> None:
    if type(digests) is not tuple:
        raise _refuse(f"{name} must be a tuple of sha256 digests")
    for index, digest in enumerate(digests):
        if not _is_sha256_hex(digest):
            raise _refuse(f"{name}[{index}] must be a lowercase sha256 hex digest")


def _require_gain_map(name: str, pairs: tuple[tuple[str, float], ...]) -> None:
    if type(pairs) is not tuple:
        raise _refuse(f"{name} must be a tuple of (key, gain) pairs")
    previous: str | None = None
    for entry in pairs:
        if type(entry) is not tuple or len(entry) != 2:
            raise _refuse(f"{name} entries must be (key, gain) pairs")
        key, gain = entry
        _require_identifier(f"{name} key", key)
        _require_finite_float(f"{name}[{key!r}]", gain)
        if previous is not None and key <= previous:
            raise _refuse(f"{name} keys must be unique and strictly ascending")
        previous = key


_DECIMAL_CHUNK_DIGITS = 100
_DECIMAL_CHUNK_BASE = 10**_DECIMAL_CHUNK_DIGITS


def _decimal_string(value: int) -> str:
    """Exact decimal text for a nonnegative int of any magnitude.

    Assembled from bounded chunks so it never trips the interpreter's
    int-to-str digit limit (which can be as low as 640 digits),
    keeping canonical serialization total over the accepted integer
    domain regardless of the process-global limit setting.
    """
    if value < _DECIMAL_CHUNK_BASE:
        return str(value)
    chunks: list[int] = []
    remaining = value
    while remaining:
        remaining, chunk = divmod(remaining, _DECIMAL_CHUNK_BASE)
        chunks.append(chunk)
    leading = str(chunks[-1])
    rest = "".join(str(chunk).zfill(_DECIMAL_CHUNK_DIGITS) for chunk in reversed(chunks[:-1]))
    return leading + rest


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(preimage: str) -> str:
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _gain_map_facts(pairs: tuple[tuple[str, float], ...]) -> list[list[str]]:
    return [[key, canonical_float(gain)] for key, gain in pairs]


@dataclass(frozen=True, slots=True)
class ExecutedStepAnchors:
    """The three engine-owned timeline anchors of one executed step."""

    step_index: int
    sigma: float
    progress: float

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise TypeError("step_index must be an exact int >= 0")
        if type(self.sigma) is not float or not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise TypeError("sigma must be a finite exact float >= 0")
        if type(self.progress) is not float or not (0.0 <= self.progress <= 1.0):
            raise TypeError("progress must be a finite exact float in [0, 1]")


@dataclass(frozen=True, slots=True)
class ExecutedTimeline:
    """The exact executed step sequence fixed by run.sigma_schedule."""

    steps: tuple[ExecutedStepAnchors, ...]

    def __post_init__(self) -> None:
        if type(self.steps) is not tuple or not self.steps:
            raise TypeError("steps must be a non-empty tuple of ExecutedStepAnchors")
        for position, step in enumerate(self.steps):
            if type(step) is not ExecutedStepAnchors:
                raise TypeError("steps must contain only exact ExecutedStepAnchors values")
            if step.step_index != position:
                raise TypeError("step_index values must be dense and ascending from 0")
        progresses = [step.progress for step in self.steps]
        if any(
            later <= earlier for earlier, later in zip(progresses, progresses[1:], strict=False)
        ):
            raise TypeError("progress must be strictly increasing across executed steps")


@dataclass(frozen=True, slots=True)
class ConstantGainCurve:
    timeline_gain: float

    def __post_init__(self) -> None:
        _require_finite_float("timeline_gain", self.timeline_gain)


@dataclass(frozen=True, slots=True)
class DirectGainTableCurve:
    """A timeline gain per executed step, matched exactly at realization."""

    timeline_gains: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.timeline_gains) is not tuple or not self.timeline_gains:
            raise _refuse("timeline_gains must be a non-empty tuple of finite floats")
        for index, gain in enumerate(self.timeline_gains):
            _require_finite_float(f"timeline_gains[{index}]", gain)


@dataclass(frozen=True, slots=True)
class GainKeyframe:
    """One declared keyframe: anchor, gain, optional state replacements.

    ``site_gains``, ``lane_gains``, and ``effect_mask_digests`` are
    full replacements of the corresponding declared value when
    provided; ``None`` means omitted, resolved by the schedule's
    declared omission behavior. A replacement gain map must cover
    exactly the declared key set.
    """

    keyframe_id: str
    anchor: float | int
    timeline_gain: float
    site_gains: tuple[tuple[str, float], ...] | None = None
    lane_gains: tuple[tuple[str, float], ...] | None = None
    effect_mask_digests: tuple[str, ...] | None = None
    minimum_realized_steps: int = 1

    def __post_init__(self) -> None:
        _require_identifier("keyframe_id", self.keyframe_id)
        _require_finite_float("timeline_gain", self.timeline_gain)
        if self.site_gains is not None:
            _require_gain_map("site_gains", self.site_gains)
        if self.lane_gains is not None:
            _require_gain_map("lane_gains", self.lane_gains)
        if self.effect_mask_digests is not None:
            _require_mask_digests("effect_mask_digests", self.effect_mask_digests)
        if type(self.minimum_realized_steps) is not int or self.minimum_realized_steps < 1:
            raise _refuse("minimum_realized_steps must be an exact int >= 1")


@dataclass(frozen=True, slots=True)
class KeyframeGainCurve:
    """A canonical keyframe curve in exactly one timeline coordinate."""

    coordinate: TimelineCoordinate
    keyframes: tuple[GainKeyframe, ...]
    interpolation: GainInterpolation
    endpoint: GainEndpoint
    omission: GainOmission
    hold_policy: KeyframeHoldPolicy

    def __post_init__(self) -> None:
        _require_profile("coordinate", self.coordinate, TimelineCoordinate)
        _require_profile("interpolation", self.interpolation, GainInterpolation)
        _require_profile("endpoint", self.endpoint, GainEndpoint)
        _require_profile("omission", self.omission, GainOmission)
        _require_profile("hold_policy", self.hold_policy, KeyframeHoldPolicy)
        if type(self.keyframes) is not tuple or not self.keyframes:
            raise _refuse("keyframes must be a non-empty tuple of GainKeyframe values")
        seen_ids: set[str] = set()
        for keyframe in self.keyframes:
            if type(keyframe) is not GainKeyframe:
                raise _refuse("keyframes must contain only exact GainKeyframe values")
            if keyframe.keyframe_id in seen_ids:
                raise _refuse(f"duplicate keyframe_id {keyframe.keyframe_id!r}")
            seen_ids.add(keyframe.keyframe_id)
            if self.coordinate is TimelineCoordinate.STEP_INDEX:
                if type(keyframe.anchor) is not int or keyframe.anchor < 0:
                    raise _refuse("step_index anchors must be exact ints >= 0")
            else:
                if type(keyframe.anchor) is not float:
                    raise _refuse(f"{self.coordinate.value} anchors must be exact floats")
                _require_finite_float("anchor", keyframe.anchor)
                if self.coordinate is TimelineCoordinate.PROGRESS and not (
                    0.0 <= keyframe.anchor <= 1.0
                ):
                    raise _refuse("progress anchors must lie in [0, 1]")
                if self.coordinate is TimelineCoordinate.SIGMA and not (keyframe.anchor >= 0.0):
                    raise _refuse("sigma anchors must be >= 0")
        anchors = [keyframe.anchor for keyframe in self.keyframes]
        if self.coordinate is TimelineCoordinate.SIGMA:
            ordered = all(
                later < earlier for earlier, later in zip(anchors, anchors[1:], strict=False)
            )
        else:
            ordered = all(
                later > earlier for earlier, later in zip(anchors, anchors[1:], strict=False)
            )
        if not ordered:
            raise _refuse(
                "anchors must be strictly ordered in execution order"
                " (sigma descending, progress and step_index ascending)"
            )


TimelineGainCurve = ConstantGainCurve | DirectGainTableCurve | KeyframeGainCurve


def _curve_facts(curve: TimelineGainCurve) -> dict[str, object]:
    if type(curve) is ConstantGainCurve:
        return {"kind": "constant", "timeline_gain": canonical_float(curve.timeline_gain)}
    if type(curve) is DirectGainTableCurve:
        return {
            "kind": "table",
            "timeline_gains": [canonical_float(gain) for gain in curve.timeline_gains],
        }
    if type(curve) is not KeyframeGainCurve:
        raise _refuse("curve must be an exact TimelineGainCurve value")
    return {
        "kind": "keyframes",
        "coordinate": curve.coordinate.value,
        "interpolation": curve.interpolation.value,
        "endpoint": curve.endpoint.value,
        "omission": curve.omission.value,
        "assignment": MONOTONE_NEAREST_ASSIGNMENT,
        "hold_policy": curve.hold_policy.value,
        "keyframes": [
            {
                "id": keyframe.keyframe_id,
                # Validation fixes anchor types per coordinate: exact
                # int for step_index, exact float otherwise.
                "anchor": (
                    _decimal_string(keyframe.anchor)
                    if type(keyframe.anchor) is int
                    else canonical_float(float(keyframe.anchor))
                ),
                "timeline_gain": canonical_float(keyframe.timeline_gain),
                "site_gains": (
                    None if keyframe.site_gains is None else _gain_map_facts(keyframe.site_gains)
                ),
                "lane_gains": (
                    None if keyframe.lane_gains is None else _gain_map_facts(keyframe.lane_gains)
                ),
                "effect_mask_digests": (
                    None
                    if keyframe.effect_mask_digests is None
                    else list(keyframe.effect_mask_digests)
                ),
                "minimum_realized_steps": _decimal_string(keyframe.minimum_realized_steps),
            }
            for keyframe in curve.keyframes
        ],
    }


_CURVE_TYPES = (ConstantGainCurve, DirectGainTableCurve, KeyframeGainCurve)


@dataclass(frozen=True, slots=True)
class ContributionGain:
    """The immutable declared gain data of one control contribution."""

    curve: TimelineGainCurve
    global_gain: float
    site_gains: tuple[tuple[str, float], ...] = ()
    lane_gains: tuple[tuple[str, float], ...] = ()
    effect_mask_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.curve) not in _CURVE_TYPES:
            raise _refuse("curve must be an exact timeline gain curve value")
        _require_finite_float("global_gain", self.global_gain)
        _require_gain_map("site_gains", self.site_gains)
        _require_gain_map("lane_gains", self.lane_gains)
        _require_mask_digests("effect_mask_digests", self.effect_mask_digests)
        if type(self.curve) is KeyframeGainCurve:
            site_keys = tuple(key for key, _ in self.site_gains)
            lane_keys = tuple(key for key, _ in self.lane_gains)
            for keyframe in self.curve.keyframes:
                if keyframe.site_gains is not None:
                    if tuple(key for key, _ in keyframe.site_gains) != site_keys:
                        raise _refuse(
                            f"keyframe {keyframe.keyframe_id!r} site_gains must cover"
                            " exactly the declared site key set"
                        )
                if keyframe.lane_gains is not None:
                    if tuple(key for key, _ in keyframe.lane_gains) != lane_keys:
                        raise _refuse(
                            f"keyframe {keyframe.keyframe_id!r} lane_gains must cover"
                            " exactly the declared lane key set"
                        )

    @property
    def canonical_preimage(self) -> str:
        facts = {
            "curve": _curve_facts(self.curve),
            "global_gain": canonical_float(self.global_gain),
            "site_gains": _gain_map_facts(self.site_gains),
            "lane_gains": _gain_map_facts(self.lane_gains),
            "effect_mask_digests": list(self.effect_mask_digests),
        }
        return _canonical_json((_SCHEDULE_DOMAIN, facts))

    @property
    def digest(self) -> str:
        return _digest(self.canonical_preimage)


@dataclass(frozen=True, slots=True)
class RealizedGainRow:
    """The resolved gain state of one executed step."""

    step_index: int
    sigma: float
    progress: float
    segment: str
    keyframe_id: str | None
    timeline_gain: float
    global_gain: float
    site_gains: tuple[tuple[str, float], ...]
    lane_gains: tuple[tuple[str, float], ...]
    effect_mask_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealizedGainTable:
    """One row per executed step plus keyframe accounting; build via
    :func:`realize_gain_table` only."""

    schedule_digest: str
    assignment_profile: str | None
    hold_policy: str | None
    rows: tuple[RealizedGainRow, ...]
    keyframe_counts: tuple[tuple[str, int, int], ...]
    canonical_preimage: str
    digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("RealizedGainTable values are created by realize_gain_table")


@dataclass(frozen=True, slots=True)
class _KeyframeState:
    timeline_gain: float
    site_gains: tuple[tuple[str, float], ...]
    lane_gains: tuple[tuple[str, float], ...]
    effect_mask_digests: tuple[str, ...]


def _normalized_states(gain: ContributionGain, curve: KeyframeGainCurve) -> list[_KeyframeState]:
    base = _KeyframeState(0.0, gain.site_gains, gain.lane_gains, gain.effect_mask_digests)
    states: list[_KeyframeState] = []
    previous = base
    for keyframe in curve.keyframes:
        fallback = previous if curve.omission is GainOmission.INHERIT else base
        state = _KeyframeState(
            keyframe.timeline_gain,
            keyframe.site_gains if keyframe.site_gains is not None else fallback.site_gains,
            keyframe.lane_gains if keyframe.lane_gains is not None else fallback.lane_gains,
            keyframe.effect_mask_digests
            if keyframe.effect_mask_digests is not None
            else fallback.effect_mask_digests,
        )
        states.append(state)
        previous = state
    return states


def _row_coordinate(step: ExecutedStepAnchors, coordinate: TimelineCoordinate) -> float | int:
    """The step's coordinate value; step_index stays an exact int."""
    if coordinate is TimelineCoordinate.STEP_INDEX:
        return step.step_index
    if coordinate is TimelineCoordinate.SIGMA:
        return step.sigma
    return step.progress


def _execution_position(value: float | int, coordinate: TimelineCoordinate) -> float | int:
    """Map a coordinate value to a scale increasing in execution order."""
    return -value if coordinate is TimelineCoordinate.SIGMA else value


def _interpolate(
    left: _KeyframeState,
    right: _KeyframeState,
    fraction: float,
    interpolation: GainInterpolation,
) -> _KeyframeState:
    if interpolation is GainInterpolation.HOLD:
        return left
    if interpolation is GainInterpolation.SMOOTHSTEP:
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)

    def scalar(a: float, b: float) -> float:
        # a*(1-f) + b*f never overflows for finite a, b and f in [0, 1]
        # (the exact convex combination is bounded by max(|a|, |b|)),
        # unlike a + (b - a)*f whose b - a can overflow.
        result = a * (1.0 - fraction) + b * fraction
        if not math.isfinite(result):
            raise _refuse("interpolated gain is non-finite")
        return result

    return _KeyframeState(
        scalar(left.timeline_gain, right.timeline_gain),
        tuple(
            (key, scalar(a, b))
            for (key, a), (_, b) in zip(left.site_gains, right.site_gains, strict=True)
        ),
        tuple(
            (key, scalar(a, b))
            for (key, a), (_, b) in zip(left.lane_gains, right.lane_gains, strict=True)
        ),
        left.effect_mask_digests,
    )


_Value = tuple[int, int, Fraction]

_ZERO_VALUE: _Value = (0, 0, Fraction(0))


def _add_values(left: _Value, right: _Value) -> _Value:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _assign_hold_rows(
    curve: KeyframeGainCurve,
    row_positions: list[float | int],
    anchor_positions: list[float | int],
) -> list[tuple[int, int]]:
    """monotone-nearest.v1: per keyframe (start, length), length 0 = none.

    Runs are contiguous, disjoint, and ordered by keyframe declaration
    index, which is the profile's state-monotonicity invariant. The
    value triple maximizes (keyframes with a row, occurrences assigned,
    negated total absolute distance) lexicographically; reconstruction
    then prefers assigning over skipping, earlier starts, and longer
    runs, which realizes the lexicographically earliest
    (keyframe declaration index, step_index) assignment among ties.

    Distances are exact rationals: float coordinates and exact-int
    step indices convert to Fraction losslessly, so sums never
    overflow or round and the value comparisons decide the profile's
    exact mathematical objective.
    Reconstruction re-evaluates candidate values and matches them by
    exact equality against the stored table.
    """
    step_count = len(row_positions)
    keyframes = curve.keyframes
    keyframe_count = len(keyframes)
    row_fractions = [Fraction(position) for position in row_positions]
    anchor_fractions = [Fraction(position) for position in anchor_positions]

    def run_distance(keyframe_index: int, start: int, length: int) -> Fraction:
        anchor = anchor_fractions[keyframe_index]
        distance = Fraction(0)
        for row in range(start, start + length):
            distance += abs(row_fractions[row] - anchor)
        return distance

    # suffix[k][s]: best value using keyframes k.. over rows s.. only.
    # An assignment feasible from s+1 is feasible from s, so each entry
    # folds in suffix[k][s+1]; runs are only enumerated at their exact
    # start.
    suffix: list[list[_Value]] = [
        [_ZERO_VALUE] * (step_count + 1) for _ in range(keyframe_count + 1)
    ]
    for keyframe_index in range(keyframe_count - 1, -1, -1):
        requested = keyframes[keyframe_index].minimum_realized_steps
        anchor = anchor_fractions[keyframe_index]
        current = suffix[keyframe_index]
        following = suffix[keyframe_index + 1]
        for start in range(step_count, -1, -1):
            best = following[start]
            if start < step_count:
                if current[start + 1] > best:
                    best = current[start + 1]
                distance = Fraction(0)
                max_length = min(requested, step_count - start)
                for length in range(1, max_length + 1):
                    distance += abs(row_fractions[start + length - 1] - anchor)
                    candidate = _add_values((1, length, -distance), following[start + length])
                    if candidate > best:
                        best = candidate
            current[start] = best

    assignments: list[tuple[int, int]] = []
    cursor = 0
    remaining = suffix[0][0]
    for keyframe_index in range(keyframe_count):
        requested = keyframes[keyframe_index].minimum_realized_steps
        chosen: tuple[int, int] = (0, 0)
        found = False
        for start in range(cursor, step_count):
            max_length = min(requested, step_count - start)
            for length in range(max_length, 0, -1):
                candidate = _add_values(
                    (1, length, -run_distance(keyframe_index, start, length)),
                    suffix[keyframe_index + 1][start + length],
                )
                if candidate == remaining:
                    chosen = (start, length)
                    found = True
                    break
            if found:
                break
        if found:
            start, length = chosen
            assignments.append((start, length))
            remaining = suffix[keyframe_index + 1][start + length]
            cursor = start + length
        else:
            assignments.append((cursor, 0))
            remaining = suffix[keyframe_index + 1][cursor]
    return assignments


def _freeze_table(
    schedule_digest: str,
    assignment_profile: str | None,
    hold_policy: str | None,
    rows: tuple[RealizedGainRow, ...],
    keyframe_counts: tuple[tuple[str, int, int], ...],
) -> RealizedGainTable:
    row_facts = [
        {
            "step_index": row.step_index,
            "sigma": canonical_float(row.sigma),
            "progress": canonical_float(row.progress),
            "segment": row.segment,
            "keyframe_id": row.keyframe_id,
            "timeline_gain": canonical_float(row.timeline_gain),
            "global_gain": canonical_float(row.global_gain),
            "site_gains": _gain_map_facts(row.site_gains),
            "lane_gains": _gain_map_facts(row.lane_gains),
            "effect_mask_digests": list(row.effect_mask_digests),
        }
        for row in rows
    ]
    preimage = _canonical_json(
        (
            _TABLE_DOMAIN,
            schedule_digest,
            assignment_profile,
            hold_policy,
            row_facts,
            [
                # requested is schedule-declared and unbounded; realized
                # is a row count bounded by the dense executed timeline.
                [keyframe_id, _decimal_string(requested), realized]
                for keyframe_id, requested, realized in keyframe_counts
            ],
        )
    )
    table = object.__new__(RealizedGainTable)
    object.__setattr__(table, "schedule_digest", schedule_digest)
    object.__setattr__(table, "assignment_profile", assignment_profile)
    object.__setattr__(table, "hold_policy", hold_policy)
    object.__setattr__(table, "rows", rows)
    object.__setattr__(table, "keyframe_counts", keyframe_counts)
    object.__setattr__(table, "canonical_preimage", preimage)
    object.__setattr__(table, "digest", _digest(preimage))
    return table


def _realize_keyframes(
    gain: ContributionGain, curve: KeyframeGainCurve, timeline: ExecutedTimeline
) -> RealizedGainTable:
    states = _normalized_states(gain, curve)
    anchor_positions = [
        _execution_position(keyframe.anchor, curve.coordinate) for keyframe in curve.keyframes
    ]
    row_positions = [
        _execution_position(_row_coordinate(step, curve.coordinate), curve.coordinate)
        for step in timeline.steps
    ]
    assignments = _assign_hold_rows(curve, row_positions, anchor_positions)

    held_by_row: dict[int, int] = {}
    for keyframe_index, (start, length) in enumerate(assignments):
        for row in range(start, start + length):
            held_by_row[row] = keyframe_index

    rows: list[RealizedGainRow] = []
    for step in timeline.steps:
        row_index = step.step_index
        if row_index in held_by_row:
            keyframe_index = held_by_row[row_index]
            state = states[keyframe_index]
            segment = f"hold:{curve.keyframes[keyframe_index].keyframe_id}"
            keyframe_id: str | None = curve.keyframes[keyframe_index].keyframe_id
        else:
            keyframe_id = None
            position = row_positions[row_index]
            if position < anchor_positions[0]:
                if curve.endpoint is GainEndpoint.ZERO:
                    state = _KeyframeState(
                        0.0,
                        states[0].site_gains,
                        states[0].lane_gains,
                        states[0].effect_mask_digests,
                    )
                else:
                    state = states[0]
                segment = f"endpoint.before:{curve.endpoint.value}"
            elif position >= anchor_positions[-1]:
                if position == anchor_positions[-1]:
                    state = states[-1]
                    segment = f"anchor:{curve.keyframes[-1].keyframe_id}"
                elif curve.endpoint is GainEndpoint.ZERO:
                    state = _KeyframeState(
                        0.0,
                        states[-1].site_gains,
                        states[-1].lane_gains,
                        states[-1].effect_mask_digests,
                    )
                    segment = f"endpoint.after:{curve.endpoint.value}"
                else:
                    state = states[-1]
                    segment = f"endpoint.after:{curve.endpoint.value}"
            else:
                left = 0
                while anchor_positions[left + 1] <= position:
                    left += 1
                span = anchor_positions[left + 1] - anchor_positions[left]
                fraction = (position - anchor_positions[left]) / span
                state = _interpolate(states[left], states[left + 1], fraction, curve.interpolation)
                segment = f"segment[{left}]:{curve.interpolation.value}"
        rows.append(
            RealizedGainRow(
                step.step_index,
                step.sigma,
                step.progress,
                segment,
                keyframe_id,
                state.timeline_gain,
                gain.global_gain,
                state.site_gains,
                state.lane_gains,
                state.effect_mask_digests,
            )
        )

    keyframe_counts = tuple(
        (keyframe.keyframe_id, keyframe.minimum_realized_steps, assignments[index][1])
        for index, keyframe in enumerate(curve.keyframes)
    )
    if curve.hold_policy is KeyframeHoldPolicy.HARD:
        for keyframe_id_value, requested, realized in keyframe_counts:
            if realized < requested:
                raise GainRefusal(
                    GainRefusalCode.UNSATISFIED_KEYFRAME_MINIMUM,
                    f"keyframe {keyframe_id_value!r} realized {realized} of"
                    f" {_decimal_string(requested)} requested rows",
                )
    return _freeze_table(
        gain.digest,
        MONOTONE_NEAREST_ASSIGNMENT,
        curve.hold_policy.value,
        tuple(rows),
        keyframe_counts,
    )


def realize_gain_table(gain: ContributionGain, timeline: ExecutedTimeline) -> RealizedGainTable:
    """Realize the declared schedule over the exact executed timeline."""
    if type(gain) is not ContributionGain:
        raise _refuse("gain must be an exact ContributionGain value")
    if type(timeline) is not ExecutedTimeline:
        raise _refuse("timeline must be an exact ExecutedTimeline value")
    curve = gain.curve
    if type(curve) is ConstantGainCurve:
        rows = tuple(
            RealizedGainRow(
                step.step_index,
                step.sigma,
                step.progress,
                "constant",
                None,
                curve.timeline_gain,
                gain.global_gain,
                gain.site_gains,
                gain.lane_gains,
                gain.effect_mask_digests,
            )
            for step in timeline.steps
        )
        return _freeze_table(gain.digest, None, None, rows, ())
    if type(curve) is DirectGainTableCurve:
        if len(curve.timeline_gains) != len(timeline.steps):
            raise _refuse(
                f"direct table declares {len(curve.timeline_gains)} rows for"
                f" {len(timeline.steps)} executed steps; tables are never"
                " clipped, padded, or resampled"
            )
        rows = tuple(
            RealizedGainRow(
                step.step_index,
                step.sigma,
                step.progress,
                "table",
                None,
                curve.timeline_gains[step.step_index],
                gain.global_gain,
                gain.site_gains,
                gain.lane_gains,
                gain.effect_mask_digests,
            )
            for step in timeline.steps
        )
        return _freeze_table(gain.digest, None, None, rows, ())
    if type(curve) is not KeyframeGainCurve:
        raise _refuse("gain.curve must be an exact TimelineGainCurve value")
    return _realize_keyframes(gain, curve, timeline)


def contribution_gain_slot_facts(
    gain: ContributionGain, table: RealizedGainTable
) -> tuple[str, ...]:
    """The intervention-plan slot fact fragment for one contribution's
    gain schedule and realization."""
    if type(gain) is not ContributionGain:
        raise _refuse("gain must be an exact ContributionGain value")
    if type(table) is not RealizedGainTable:
        raise _refuse("table must be an exact RealizedGainTable value")
    if table.schedule_digest != gain.digest:
        raise _refuse("table was not realized from this declared gain schedule")
    facts = [
        f"gain.schedule={gain.digest}",
        f"gain.realized_table={table.digest}",
        f"gain.global={canonical_float(gain.global_gain)}",
    ]
    if type(gain.curve) is KeyframeGainCurve:
        curve = gain.curve
        facts.extend(
            (
                f"gain.coordinate={curve.coordinate.value}",
                f"gain.interpolation={curve.interpolation.value}",
                f"gain.endpoint={curve.endpoint.value}",
                f"gain.omission={curve.omission.value}",
                f"gain.assignment={MONOTONE_NEAREST_ASSIGNMENT}",
                f"gain.hold_policy={curve.hold_policy.value}",
            )
        )
    facts.extend(f"gain.site[{key}]={canonical_float(value)}" for key, value in gain.site_gains)
    facts.extend(f"gain.lane[{key}]={canonical_float(value)}" for key, value in gain.lane_gains)
    facts.extend(
        f"gain.effect_mask[{index}]={digest}"
        for index, digest in enumerate(gain.effect_mask_digests)
    )
    return tuple(facts)
