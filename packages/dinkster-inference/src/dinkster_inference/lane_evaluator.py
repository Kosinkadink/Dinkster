"""Deterministic CPU lane planning and sequential evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

T = TypeVar("T", covariant=True)
U = TypeVar("U")


def _validate_index(index: int, name: str) -> None:
    if type(index) is not int:
        raise TypeError(f"{name} must be an int")
    if index < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class LaneItem(Generic[T]):
    """An input value paired with its original global index."""

    index: int
    value: T

    def __post_init__(self) -> None:
        _validate_index(self.index, "item index")


@dataclass(frozen=True)
class LaneAssignment(Generic[T]):
    """A nonempty contiguous run of indexed items assigned to one lane."""

    index: int
    items: tuple[LaneItem[T], ...]

    def __post_init__(self) -> None:
        _validate_index(self.index, "lane index")
        raw_items = cast(object, self.items)
        if type(raw_items) is not tuple:
            raise TypeError("lane items must be a tuple")
        items = cast(tuple[object, ...], raw_items)
        if not items:
            raise ValueError("lane items must not be empty")
        if any(not isinstance(item, LaneItem) for item in items):
            raise TypeError("lane items must contain only LaneItem values")
        lane_items = cast(tuple[LaneItem[T], ...], items)
        first_index = lane_items[0].index
        if tuple(item.index for item in lane_items) != tuple(
            range(first_index, first_index + len(lane_items))
        ):
            raise ValueError("lane item indices must be contiguous")


class LaneEvaluationCancelled(RuntimeError):
    """Raised when cancellation is observed before a lane starts."""


def plan_lanes(items: Sequence[T], lane_count: int) -> tuple[LaneAssignment[T], ...]:
    """Snapshot items into balanced, contiguous, nonempty lane assignments."""

    if type(lane_count) is not int:
        raise TypeError("lane_count must be an int")
    if lane_count <= 0:
        raise ValueError("lane_count must be positive")

    snapshot = tuple(items)
    if not snapshot:
        return ()

    effective_lanes = min(lane_count, len(snapshot))
    base_size, remainder = divmod(len(snapshot), effective_lanes)
    assignments: list[LaneAssignment[T]] = []
    offset = 0
    for lane_index in range(effective_lanes):
        lane_size = base_size + (lane_index < remainder)
        lane_items = tuple(
            LaneItem(index=item_index, value=snapshot[item_index])
            for item_index in range(offset, offset + lane_size)
        )
        assignments.append(LaneAssignment(index=lane_index, items=lane_items))
        offset += lane_size
    return tuple(assignments)


def _validate_plan(assignments: tuple[object, ...]) -> int:
    expected_item_index = 0
    for expected_lane_index, assignment in enumerate(assignments):
        if not isinstance(assignment, LaneAssignment):
            raise TypeError("assignments must contain only LaneAssignment values")
        assignment = cast(LaneAssignment[object], assignment)
        _validate_index(assignment.index, "lane index")
        if assignment.index != expected_lane_index:
            raise ValueError("lane indices must be contiguous and unique")
        raw_items = cast(object, assignment.items)
        if type(raw_items) is not tuple:
            raise TypeError("every lane must contain an item tuple")
        items = cast(tuple[object, ...], raw_items)
        if not items:
            raise ValueError("every lane must contain a nonempty item tuple")
        for item in items:
            if not isinstance(item, LaneItem):
                raise TypeError("lane items must contain only LaneItem values")
            _validate_index(item.index, "item index")
            if item.index != expected_item_index:
                raise ValueError("global item indices must be contiguous and unique")
            expected_item_index += 1
    return expected_item_index


def evaluate_lanes(
    assignments: Sequence[LaneAssignment[T]],
    evaluate: Callable[[LaneAssignment[T]], tuple[U, ...]],
    *,
    release: Callable[[int], None],
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[U, ...]:
    """Evaluate a validated lane plan sequentially and restore global result order."""

    raw_plan = tuple(cast(Sequence[object], assignments))
    item_count = _validate_plan(raw_plan)
    plan = cast(tuple[LaneAssignment[T], ...], raw_plan)
    if not plan:
        return ()

    missing = object()
    results: list[U | object] = [missing] * item_count
    for assignment in plan:
        cancellation_result = cancelled()
        if type(cancellation_result) is not bool:
            raise TypeError("cancelled must return a bool")
        if cancellation_result:
            raise LaneEvaluationCancelled(
                f"lane evaluation cancelled before lane {assignment.index}"
            )

        evaluation_error: BaseException | None = None
        try:
            lane_results = evaluate(assignment)
            if type(lane_results) is not tuple:
                raise TypeError("lane evaluator must return a tuple")
            if len(lane_results) != len(assignment.items):
                raise ValueError("lane evaluator must return exactly one result per item")
            for item, result in zip(assignment.items, lane_results, strict=True):
                results[item.index] = result
        except BaseException as error:
            evaluation_error = error
            raise
        finally:
            try:
                release(assignment.index)
            except BaseException as release_error:
                if evaluation_error is None:
                    raise
                evaluation_error.add_note(
                    f"lane {assignment.index} release failed: {release_error!r}"
                )

    return cast(tuple[U, ...], tuple(results))
