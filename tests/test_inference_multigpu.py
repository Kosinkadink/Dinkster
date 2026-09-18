from __future__ import annotations

import pytest
from dinkster_inference import (
    Conditioning,
    GuidanceCondition,
    GuidanceEvaluationPlan,
    GuidancePredictionSource,
    GuidanceRole,
    MultiGPUPlanError,
    RankWeights,
    WindowUnitAssignment,
    plan_guidance_lanes,
    plan_window_units,
)


class _Tensor:
    shape = (1,)


def test_weighted_guidance_plan_preserves_canonical_lane_order() -> None:
    conditioning = Conditioning(_Tensor())
    plan = GuidanceEvaluationPlan(
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, conditioning),
            GuidanceCondition("zero", GuidanceRole.CONDITIONAL, None),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, conditioning),
        ),
        "positive",
        "negative",
    )
    assignments = plan_guidance_lanes(plan, RankWeights((1, 2, 1)))

    assert tuple(item.lane_id for item in assignments) == ("positive", "zero", "negative")
    assert tuple(item.rank for item in assignments) == (0, 0, 1)
    assert assignments[1].source is GuidancePredictionSource.SYNTHETIC_ZERO


def test_rank_weights_require_two_positive_finite_values() -> None:
    for values in ((1.0,), (1.0, 0.0), (1.0, float("inf"))):
        with pytest.raises(MultiGPUPlanError, match="positive rank weights"):
            RankWeights(values)


def test_window_units_exactly_partition_the_window_set() -> None:
    for world_size in (2, 4):
        weights = RankWeights(tuple(1.0 for _ in range(world_size)))
        for window_count in (1, 2, 3, 4, 5, 7, 8):
            assignments = plan_window_units(window_count, weights)

            assert tuple(item.window_index for item in assignments) == tuple(range(window_count))
            assert all(0 <= item.rank < world_size for item in assignments)
            ranks = tuple(item.rank for item in assignments)
            assert ranks == tuple(sorted(ranks))


def test_window_units_are_deterministic_and_balanced() -> None:
    weights = RankWeights((1.0, 1.0))

    assert plan_window_units(3, weights) == plan_window_units(3, weights)
    assert plan_window_units(3, weights) == (
        WindowUnitAssignment(0, 0),
        WindowUnitAssignment(0, 1),
        WindowUnitAssignment(1, 2),
    )
    four_way = plan_window_units(6, RankWeights((1.0, 1.0, 1.0, 1.0)))
    counts = [sum(1 for item in four_way if item.rank == rank) for rank in range(4)]
    assert counts == [2, 2, 1, 1]


def test_window_units_require_a_positive_exact_window_count() -> None:
    weights = RankWeights((1.0, 1.0))
    for count in (0, -1, True, 2.0):
        with pytest.raises(MultiGPUPlanError, match="exact int"):
            plan_window_units(count, weights)  # type: ignore[arg-type]
