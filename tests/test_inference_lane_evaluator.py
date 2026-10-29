from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
from dinkster_inference import (
    LaneAssignment,
    LaneEvaluationCancelled,
    LaneItem,
    evaluate_lanes,
    plan_lanes,
)


@pytest.mark.parametrize("lane_count", [True, False, 0, -1])
def test_plan_lanes_rejects_invalid_lane_count(lane_count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        plan_lanes(("value",), cast(int, lane_count))


def test_plan_lanes_snapshots_balanced_contiguous_assignments() -> None:
    values = list(range(11))

    assignments = plan_lanes(values, 4)
    values.clear()

    assert tuple(len(assignment.items) for assignment in assignments) == (3, 3, 3, 2)
    assert tuple(assignment.index for assignment in assignments) == (0, 1, 2, 3)
    assert tuple(item.index for lane in assignments for item in lane.items) == tuple(range(11))
    assert tuple(item.value for lane in assignments for item in lane.items) == tuple(range(11))
    assert plan_lanes((), 4) == ()


def test_plan_lanes_never_creates_empty_assignments() -> None:
    assignments = plan_lanes(("a", "b"), 8)

    assert tuple(len(assignment.items) for assignment in assignments) == (1, 1)


def test_lane_values_are_frozen_and_strictly_validated() -> None:
    item = LaneItem(0, "a")
    assignment = LaneAssignment(0, (item,))

    with pytest.raises(FrozenInstanceError):
        cast(Any, item).index = 1
    with pytest.raises(FrozenInstanceError):
        cast(Any, assignment).index = 1
    with pytest.raises(TypeError):
        LaneItem(cast(int, True), "a")
    with pytest.raises(ValueError):
        LaneItem(-1, "a")
    with pytest.raises(TypeError):
        LaneAssignment(0, cast(tuple[LaneItem[str], ...], [item]))
    with pytest.raises(ValueError):
        LaneAssignment(0, ())
    with pytest.raises(ValueError):
        LaneAssignment(0, (LaneItem(0, "a"), LaneItem(2, "b")))


def test_evaluate_lanes_restores_global_order_and_releases_each_started_lane_once() -> None:
    assignments = plan_lanes(tuple(range(11)), 4)
    started: list[int] = []
    released: list[int] = []

    def evaluate(assignment: LaneAssignment[int]) -> tuple[str, ...]:
        started.append(assignment.index)
        return tuple(f"result-{item.index}" for item in assignment.items)

    result = evaluate_lanes(assignments, evaluate, release=released.append)

    assert result == tuple(f"result-{index}" for index in range(11))
    assert started == [0, 1, 2, 3]
    assert released == [0, 1, 2, 3]


def test_evaluate_lanes_snapshots_the_assignment_sequence() -> None:
    assignments = list(plan_lanes(("a", "b", "c"), 3))
    started: list[int] = []
    released: list[int] = []

    def evaluate(assignment: LaneAssignment[str]) -> tuple[str, ...]:
        started.append(assignment.index)
        assignments.clear()
        return tuple(item.value for item in assignment.items)

    assert evaluate_lanes(assignments, evaluate, release=released.append) == ("a", "b", "c")
    assert started == [0, 1, 2]
    assert released == [0, 1, 2]


@pytest.mark.parametrize(
    "assignments",
    [
        (LaneAssignment(1, (LaneItem(0, "a"),)),),
        (
            LaneAssignment(0, (LaneItem(0, "a"),)),
            LaneAssignment(0, (LaneItem(1, "b"),)),
        ),
        (
            LaneAssignment(0, (LaneItem(1, "a"),)),
            LaneAssignment(1, (LaneItem(2, "b"),)),
        ),
        (
            LaneAssignment(0, (LaneItem(0, "a"), LaneItem(1, "b"))),
            LaneAssignment(1, (LaneItem(1, "c"),)),
        ),
    ],
)
def test_evaluate_lanes_rejects_malformed_plans_before_callbacks(
    assignments: tuple[LaneAssignment[str], ...],
) -> None:
    called = False

    def unexpected(*_args: object) -> Any:
        nonlocal called
        called = True
        raise AssertionError("callback must not be called")

    with pytest.raises(ValueError):
        evaluate_lanes(assignments, unexpected, release=unexpected, cancelled=unexpected)
    assert called is False


def test_evaluate_lanes_rejects_foreign_and_corrupted_plans_before_callbacks() -> None:
    foreign = cast(tuple[LaneAssignment[str], ...], ("foreign",))
    with pytest.raises(TypeError):
        evaluate_lanes(foreign, lambda _assignment: (), release=lambda _index: None)

    item = LaneItem(0, "a")
    assignment = LaneAssignment(0, (item,))
    object.__setattr__(item, "index", True)
    with pytest.raises(TypeError):
        evaluate_lanes((assignment,), lambda _assignment: (), release=lambda _index: None)


@pytest.mark.parametrize("result", [[], "a", iter(("a",))])
def test_evaluate_lanes_rejects_foreign_results_and_releases(result: object) -> None:
    released: list[int] = []

    with pytest.raises(TypeError):
        evaluate_lanes(
            plan_lanes(("a",), 1),
            lambda _assignment: cast(tuple[str, ...], result),
            release=released.append,
        )
    assert released == [0]


@pytest.mark.parametrize("result", [(), ("a", "b")])
def test_evaluate_lanes_requires_exact_result_cardinality(result: tuple[str, ...]) -> None:
    released: list[int] = []

    with pytest.raises(ValueError):
        evaluate_lanes(
            plan_lanes(("a",), 1),
            lambda _assignment: result,
            release=released.append,
        )
    assert released == [0]


def test_evaluate_lanes_failure_releases_failed_lane_and_stops() -> None:
    started: list[int] = []
    released: list[int] = []

    def evaluate(assignment: LaneAssignment[str]) -> tuple[str, ...]:
        started.append(assignment.index)
        if assignment.index == 1:
            raise LookupError("evaluation failed")
        return tuple(item.value for item in assignment.items)

    with pytest.raises(LookupError, match="evaluation failed"):
        evaluate_lanes(plan_lanes(("a", "b", "c"), 3), evaluate, release=released.append)
    assert started == [0, 1]
    assert released == [0, 1]


def test_evaluate_lanes_cancellation_stops_before_next_lane() -> None:
    started: list[int] = []
    released: list[int] = []
    cancellation = iter((False, True))

    with pytest.raises(LaneEvaluationCancelled, match="before lane 1"):
        evaluate_lanes(
            plan_lanes(("a", "b", "c"), 3),
            lambda assignment: (
                started.append(assignment.index) or tuple(item.value for item in assignment.items)
            ),
            release=released.append,
            cancelled=lambda: next(cancellation),
        )
    assert started == [0]
    assert released == [0]


def test_evaluate_lanes_cancellation_before_first_lane_starts_nothing() -> None:
    started: list[int] = []
    released: list[int] = []

    with pytest.raises(LaneEvaluationCancelled, match="before lane 0"):
        evaluate_lanes(
            plan_lanes(("a",), 1),
            lambda assignment: started.append(assignment.index) or ("a",),
            release=released.append,
            cancelled=lambda: True,
        )
    assert started == []
    assert released == []


@pytest.mark.parametrize("invalid", [0, 1, None, "false"])
def test_evaluate_lanes_rejects_invalid_cancellation_values(invalid: object) -> None:
    with pytest.raises(TypeError):
        evaluate_lanes(
            plan_lanes(("a",), 1),
            lambda _assignment: ("a",),
            release=lambda _index: None,
            cancelled=lambda: cast(bool, invalid),
        )


def test_evaluate_lanes_rejects_invalid_cancellation_after_completed_lane() -> None:
    started: list[int] = []
    released: list[int] = []
    cancellation = iter((False, cast(bool, 0)))

    with pytest.raises(TypeError):
        evaluate_lanes(
            plan_lanes(("a", "b"), 2),
            lambda assignment: started.append(assignment.index) or ("result",),
            release=released.append,
            cancelled=lambda: next(cancellation),
        )
    assert started == [0]
    assert released == [0]


def test_evaluate_lanes_preserves_primary_failure_when_release_also_fails() -> None:
    def evaluate(_assignment: LaneAssignment[str]) -> tuple[str, ...]:
        raise LookupError("evaluation failed")

    def release(_index: int) -> None:
        raise RuntimeError("release failed")

    with pytest.raises(LookupError, match="evaluation failed") as raised:
        evaluate_lanes(plan_lanes(("a",), 1), evaluate, release=release)
    assert raised.value.__notes__ == ["lane 0 release failed: RuntimeError('release failed')"]


@pytest.mark.parametrize(
    ("result", "expected_error"),
    [
        (["result"], TypeError),
        ((), ValueError),
    ],
)
def test_evaluate_lanes_preserves_result_error_when_release_also_fails(
    result: object,
    expected_error: type[Exception],
) -> None:
    started: list[int] = []

    def evaluate(assignment: LaneAssignment[str]) -> tuple[str, ...]:
        started.append(assignment.index)
        return cast(tuple[str, ...], result)

    def release(_index: int) -> None:
        raise RuntimeError("release failed")

    with pytest.raises(expected_error) as raised:
        evaluate_lanes(plan_lanes(("a", "b"), 2), evaluate, release=release)
    assert started == [0]
    assert raised.value.__notes__ == ["lane 0 release failed: RuntimeError('release failed')"]


def test_evaluate_lanes_release_failure_stops_before_later_lanes() -> None:
    started: list[int] = []

    def release(_index: int) -> None:
        raise RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        evaluate_lanes(
            plan_lanes(("a", "b"), 2),
            lambda assignment: started.append(assignment.index) or ("result",),
            release=release,
        )
    assert started == [0]


def test_evaluate_lanes_empty_plan_calls_nothing() -> None:
    def unexpected(*_args: object) -> Any:
        raise AssertionError("callback must not be called")

    assert evaluate_lanes((), unexpected, release=unexpected, cancelled=unexpected) == ()
