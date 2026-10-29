"""Torch-free planning for temporal context windows during sampling.

A context-windows plan splits a long temporal axis into overlapping
index windows so the model is applied per window and the per-window
outputs are fuse-averaged back into the full-length result beneath the
guidance combine. The window index generation per schedule, the
relative-fuse bias, the sigma-to-step match, and the FreeNoise shuffle
plan are a parity port of ComfyUI's context scheduling
(comfy/context_windows.py @ b78cec87). Fuse weights, merge traversal,
coverage validation, and digest identity come from the shared
windowed-evaluation construct in :mod:`.window_plan`, whose weight
kinds encode the same reference semantics: :func:`context_window_plan`
compiles one step's windows into a ``CompositeWindowPlan``. Tensor
slicing and accumulation live with the torch execution layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .window_plan import (
    CompositeWindowPlan,
    LayerWindow,
    MediaAxis,
    MergeDeclaration,
    WindowIndexList,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    compile_window_plan,
)

__all__ = [
    "TEMPORAL_WINDOW_AXIS",
    "ContextFuseMethod",
    "ContextWindowSchedule",
    "ContextWindowsError",
    "ContextWindowsSpec",
    "context_window_plan",
    "freenoise_plan",
    "plan_windows",
    "relative_bias",
    "step_index_for_sigma",
]

TEMPORAL_WINDOW_AXIS = "temporal"


class ContextWindowsError(ValueError):
    pass


class ContextWindowSchedule(StrEnum):
    """How window index lists are laid out over the temporal axis.

    Values are the reference's schedule identifiers."""

    STATIC_STANDARD = "standard_static"
    UNIFORM_STANDARD = "standard_uniform"
    UNIFORM_LOOPED = "looped_uniform"
    BATCHED = "batched"


class ContextFuseMethod(StrEnum):
    """How overlapping window outputs are combined.

    Values are the reference's fuse identifiers. RELATIVE blends windows
    sequentially by per-index bias (:func:`relative_bias`) instead of
    weight accumulation, so :func:`context_window_plan` refuses it."""

    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"


@dataclass(frozen=True, slots=True)
class ContextWindowsSpec:
    """One validated context-windows configuration.

    ``length``/``overlap``/``stride`` are in temporal-axis units of the
    latent being windowed. ``dim`` is the latent axis windows slice.
    ``causal_anchor`` prepends the frame before each non-initial window
    to the model input and strips it from the output, keeping causal
    models continuous across window edges. ``freenoise`` requests the
    initial-noise shuffle described by :func:`freenoise_plan`.

    The retain lists carry the reference's retain_index_list semantics:
    retained index ``r`` overwrites local position ``r`` of every
    window's model input (after any causal anchor is prepended) with
    index ``r`` of the full tensor, keeping reference content visible in
    every window. ``latent_retain_indices`` applies to the latent inside
    the windowed evaluation; ``cond_retain_indices`` is declared for
    family runtimes that window frame-aligned conditioning and does not
    touch the latent path. Both default empty, which changes nothing."""

    schedule: ContextWindowSchedule
    fuse_method: ContextFuseMethod
    length: int
    overlap: int
    stride: int = 1
    closed_loop: bool = False
    dim: int = 0
    freenoise: bool = False
    causal_anchor: bool = False
    cond_retain_indices: tuple[int, ...] = ()
    latent_retain_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schedule) is not ContextWindowSchedule:
            raise ContextWindowsError("schedule must be an exact ContextWindowSchedule")
        if type(self.fuse_method) is not ContextFuseMethod:
            raise ContextWindowsError("fuse_method must be an exact ContextFuseMethod")
        for name in ("length", "overlap", "stride", "dim"):
            if type(getattr(self, name)) is not int:
                raise ContextWindowsError(f"{name} must be an int")
        for name in ("closed_loop", "freenoise", "causal_anchor"):
            if type(getattr(self, name)) is not bool:
                raise ContextWindowsError(f"{name} must be a bool")
        if self.length < 1:
            raise ContextWindowsError("length must be at least 1")
        if not 0 <= self.overlap < self.length:
            raise ContextWindowsError("overlap must be nonnegative and less than length")
        if self.stride < 1:
            raise ContextWindowsError("stride must be at least 1")
        if self.dim < 0:
            raise ContextWindowsError("dim must be nonnegative")
        if self.fuse_method is ContextFuseMethod.OVERLAP_LINEAR and self.overlap < 1:
            raise ContextWindowsError("overlap-linear fusing requires overlap of at least 1")
        for name, value in (
            ("cond_retain_indices", self.cond_retain_indices),
            ("latent_retain_indices", self.latent_retain_indices),
        ):
            if type(value) is not tuple:
                raise ContextWindowsError(f"{name} must be a tuple")
            previous = -1
            for index in value:
                if type(index) is not int:
                    raise ContextWindowsError(f"{name} entries must be ints")
                if index <= previous:
                    raise ContextWindowsError(
                        f"{name} entries must be nonnegative and strictly increasing"
                    )
                previous = index
            if value and value[-1] >= self.length:
                raise ContextWindowsError(f"{name} entries must be less than length")


def _ordered_halving(value: int) -> float:
    """The reference's deterministic per-step jitter: the 64-bit binary
    reversal of ``value`` as a fraction of 2**64."""

    reversed_bits = f"{value:064b}"[::-1]
    return int(reversed_bits, 2) / (1 << 64)


def _uniform_raw_windows(
    spec: ContextWindowsSpec, num_frames: int, step: int, *, looped_end: bool
) -> list[list[int]]:
    windows: list[list[int]] = []
    stride_limit = int(math.ceil(math.log2(num_frames / spec.length))) + 1
    context_stride = min(spec.stride, stride_limit)
    for exponent in range(context_stride):
        context_step = 1 << exponent
        pad = int(round(num_frames * _ordered_halving(step)))
        end_pad = 0 if looped_end and spec.closed_loop else -spec.overlap
        for start in range(
            int(_ordered_halving(step) * context_step) + pad,
            num_frames + pad + end_pad,
            spec.length * context_step - spec.overlap,
        ):
            windows.append(
                [
                    index % num_frames
                    for index in range(start, start + spec.length * context_step, context_step)
                ]
            )
    return windows


def _does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    previous = -1
    for position, value in enumerate(window):
        value %= num_frames
        if value < previous:
            return True, position
        previous = value
    return False, -1


def _shift_window_to_start(window: list[int], num_frames: int) -> None:
    start = window[0]
    for position in range(len(window)):
        window[position] = (window[position] - start + num_frames) % num_frames


def _shift_window_to_end(window: list[int], num_frames: int) -> None:
    _shift_window_to_start(window, num_frames)
    end_delta = num_frames - window[-1] - 1
    for position in range(len(window)):
        window[position] += end_delta


def _uniform_looped_windows(
    spec: ContextWindowsSpec, num_frames: int, step: int
) -> list[list[int]]:
    if num_frames < spec.length:
        return [list(range(num_frames))]
    return _uniform_raw_windows(spec, num_frames, step, looped_end=True)


def _uniform_standard_windows(
    spec: ContextWindowsSpec, num_frames: int, step: int
) -> list[list[int]]:
    if num_frames <= spec.length:
        return [list(range(num_frames))]
    windows = _uniform_raw_windows(spec, num_frames, step, looped_end=False)
    # Shift any window that wraps past the end back onto the tail, insert
    # a replacement window where the shift left its first wrapped index
    # uncovered, and drop exact duplicates.
    delete_positions: list[int] = []
    position = 0
    while position < len(windows):
        rolls, roll_position = _does_window_roll_over(windows[position], num_frames)
        if rolls:
            roll_value = windows[position][roll_position]
            _shift_window_to_end(windows[position], num_frames)
            if roll_value not in windows[(position + 1) % len(windows)]:
                windows.insert(position + 1, list(range(roll_value, roll_value + spec.length)))
        for earlier in range(position):
            if windows[position] == windows[earlier]:
                delete_positions.append(position)
                break
        position += 1
    for stale in reversed(delete_positions):
        windows.pop(stale)
    return windows


def _static_standard_windows(spec: ContextWindowsSpec, num_frames: int) -> list[list[int]]:
    if num_frames <= spec.length:
        return [list(range(num_frames))]
    windows: list[list[int]] = []
    delta = spec.length - spec.overlap
    for start in range(0, num_frames, delta):
        ending = start + spec.length
        if ending >= num_frames:
            final_start = start - (ending - num_frames)
            windows.append(list(range(final_start, final_start + spec.length)))
            break
        windows.append(list(range(start, start + spec.length)))
    return windows


def _batched_windows(spec: ContextWindowsSpec, num_frames: int) -> list[list[int]]:
    if num_frames <= spec.length:
        return [list(range(num_frames))]
    return [
        list(range(start, min(start + spec.length, num_frames)))
        for start in range(0, num_frames, spec.length)
    ]


def plan_windows(
    spec: ContextWindowsSpec, num_frames: int, step: int
) -> tuple[tuple[int, ...], ...]:
    """Window index lists over ``num_frames`` for solver step ``step``.

    Uniform schedules jitter with the step; static schedules ignore it.
    Every frame index must be covered by at least one window - an
    uncovered index would silently fuse to garbage, so it refuses
    instead."""

    if type(num_frames) is not int or num_frames < 1:
        raise ContextWindowsError("num_frames must be a positive int")
    if type(step) is not int or step < 0:
        raise ContextWindowsError("step must be a nonnegative int")
    if spec.schedule is ContextWindowSchedule.UNIFORM_LOOPED:
        windows = _uniform_looped_windows(spec, num_frames, step)
    elif spec.schedule is ContextWindowSchedule.UNIFORM_STANDARD:
        windows = _uniform_standard_windows(spec, num_frames, step)
    elif spec.schedule is ContextWindowSchedule.STATIC_STANDARD:
        windows = _static_standard_windows(spec, num_frames)
    else:
        windows = _batched_windows(spec, num_frames)
    covered: set[int] = set()
    for window in windows:
        covered.update(window)
    if len(covered) != num_frames:
        missing = sorted(set(range(num_frames)) - covered)
        raise ContextWindowsError(
            f"context windows leave frame indices uncovered: {missing[:8]}"
            f" (schedule {spec.schedule}, length {spec.length}, overlap"
            f" {spec.overlap}, stride {spec.stride}, {num_frames} frames)"
        )
    return tuple(tuple(window) for window in windows)


_FUSE_WEIGHT_KINDS = {
    ContextFuseMethod.FLAT: WindowWeightKind.FLAT,
    ContextFuseMethod.PYRAMID: WindowWeightKind.PYRAMID,
    ContextFuseMethod.OVERLAP_LINEAR: WindowWeightKind.OVERLAP_LINEAR,
}


def context_window_plan(
    spec: ContextWindowsSpec, num_frames: int, step: int
) -> CompositeWindowPlan:
    """Compile one step's windows into a composite windowed-evaluation plan.

    The step's windows become a single-axis (:data:`TEMPORAL_WINDOW_AXIS`)
    window-plan layer whose weight profile carries the spec's fuse
    method, so per-window weights, merge traversal, and totality
    validation come from the shared construct. RELATIVE fusing has no
    accumulate-normalize representation and refuses here; its callers
    blend windows sequentially from :func:`plan_windows` and
    :func:`relative_bias` instead."""

    weight_kind = _FUSE_WEIGHT_KINDS.get(spec.fuse_method)
    if weight_kind is None:
        raise ContextWindowsError(
            "relative fusing blends by per-index bias and has no composite window plan"
        )
    windows = plan_windows(spec, num_frames, step)
    overlap = 0
    if spec.fuse_method is ContextFuseMethod.OVERLAP_LINEAR:
        # Windows shorter than the configured overlap only occur where the
        # boundary conditions suppress the ramps, so clamping keeps them
        # admissible without changing any weight.
        overlap = min(spec.overlap, min(len(window) for window in windows))
    layer = WindowPlanLayer(
        axes=(TEMPORAL_WINDOW_AXIS,),
        windows=tuple(LayerWindow((WindowIndexList(window),)) for window in windows),
        weight_profiles=(WindowWeightProfile(weight_kind, overlap),),
        merge=MergeDeclaration(),
    )
    return compile_window_plan(
        axes=(MediaAxis(TEMPORAL_WINDOW_AXIS, num_frames),),
        layers=(layer,),
    )


def relative_bias(index: int, first: int, last: int) -> float:
    """Relative-fuse influence of ``index`` within a window spanning
    ``first``..``last``: highest at the window center, floored at 1e-2."""

    center = (first + last) / 2
    half_span = (last - first + 1e-2) / 2
    return max(1e-2, 1.0 - abs(index - center) / half_span)


def step_index_for_sigma(sigmas: tuple[float, ...], sigma: float) -> int | None:
    """The schedule position of ``sigma``, or None for an off-schedule
    value (a multi-stage solver substep, which keeps the prior step's
    windows)."""

    for index, candidate in enumerate(sigmas):
        if abs(candidate - sigma) <= 1e-8 + 1e-4 * abs(sigma):
            return index
    return None


def freenoise_plan(num_frames: int, length: int, overlap: int) -> tuple[tuple[int, int, int], ...]:
    """FreeNoise shuffle spans as ``(source_start, target_start, count)``.

    Each span replaces ``count`` frames of initial noise starting at
    ``target_start`` with a seeded permutation of the ``count`` frames
    starting at ``source_start``, so windows past the first reuse
    shuffled noise instead of drawing fresh statistics."""

    if type(num_frames) is not int or num_frames < 1:
        raise ContextWindowsError("num_frames must be a positive int")
    if type(length) is not int or length < 1:
        raise ContextWindowsError("length must be a positive int")
    if type(overlap) is not int or not 0 <= overlap < length:
        raise ContextWindowsError("overlap must be nonnegative and less than length")
    spans: list[tuple[int, int, int]] = []
    delta = length - overlap
    for start in range(0, num_frames - length, delta):
        place = start + length
        count = min(delta, num_frames - place)
        if count <= 0:
            break
        spans.append((start, place, count))
    return tuple(spans)
