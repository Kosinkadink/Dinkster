"""Torch-free plans for single-job replicated-rank execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

from .guidance import GuidanceEvaluationPlan, GuidancePredictionSource
from .patches import SizedTensor

T = TypeVar("T", bound=SizedTensor)


class MultiGPUPlanError(ValueError):
    pass


@dataclass(frozen=True)
class RankWeights:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.values) is not tuple
            or len(self.values) < 2
            or any(
                type(value) not in (int, float) or value <= 0 or not math.isfinite(value)
                for value in self.values
            )
        ):
            raise MultiGPUPlanError("at least two positive rank weights are required")
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


def _weighted_counts(count: int, weights: RankWeights) -> tuple[int, ...]:
    if count < 0:
        raise MultiGPUPlanError("work count cannot be negative")
    total = sum(weights.values)
    exact = tuple(count * weight / total for weight in weights.values)
    result = [int(value) for value in exact]
    for rank in sorted(range(len(result)), key=lambda i: (-(exact[i] - result[i]), i))[
        : count - sum(result)
    ]:
        result[rank] += 1
    return tuple(result)


@dataclass(frozen=True)
class GuidanceLaneAssignment:
    rank: int
    lane_id: str
    plan_index: int
    source: GuidancePredictionSource


def plan_guidance_lanes(
    plan: GuidanceEvaluationPlan[T], weights: RankWeights
) -> tuple[GuidanceLaneAssignment, ...]:
    model = tuple(
        (index, lane) for index, lane in enumerate(plan.lanes) if lane.conditioning is not None
    )
    counts = _weighted_counts(len(model), weights)
    ranks = tuple(rank for rank, count in enumerate(counts) for _ in range(count))
    assigned = {index: rank for rank, (index, _lane) in zip(ranks, model, strict=True)}
    return tuple(
        GuidanceLaneAssignment(
            assigned.get(index, 0),
            lane.id,
            index,
            GuidancePredictionSource.MODEL
            if lane.conditioning is not None
            else GuidancePredictionSource.SYNTHETIC_ZERO,
        )
        for index, lane in enumerate(plan.lanes)
    )


@dataclass(frozen=True)
class WindowUnitAssignment:
    """One joint window bound to the rank that evaluates it."""

    rank: int
    window_index: int


def plan_window_units(window_count: int, weights: RankWeights) -> tuple[WindowUnitAssignment, ...]:
    """Assign joint-window indices to ranks deterministically.

    The result is one assignment per window index, in ascending window
    order, with contiguous runs per rank sized by the shared
    weighted-count rule. Every rank derives the identical exact
    partition of the window set (no overlap, no gap) from only the
    window count and the group weights, so no negotiation traffic is
    ever needed.
    """

    if type(window_count) is not int or window_count < 1:
        raise MultiGPUPlanError("window count must be an exact int of at least one")
    counts = _weighted_counts(window_count, weights)
    ranks = tuple(rank for rank, count in enumerate(counts) for _ in range(count))
    return tuple(WindowUnitAssignment(rank, index) for index, rank in enumerate(ranks))
