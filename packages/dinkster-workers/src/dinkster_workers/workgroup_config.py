"""Explicit parent-side configuration for MultiDevice workgroup lanes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from dinkster_protocol import (
    MAX_WORKGROUP_REPLICAS,
    WORKGROUP_CAPABILITY,
    DeviceResourceId,
    ReplicaBinding,
    ReplicaId,
    ReplicaRecipeId,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupMessage,
    WorkUnitDefinition,
    WorkUnitId,
    negotiate_workgroup_capabilities,
)

from .workgroup import ReplicaEndpoint

__all__ = ["WorkGroupLaneCandidate", "build_workgroup_configuration"]

_Send = Callable[[WorkGroupMessage], Awaitable[None]]
_Receive = Callable[[], Awaitable[WorkGroupMessage]]


@dataclass(frozen=True, slots=True)
class WorkGroupLaneCandidate:
    """One explicit parent-namespace lane and its advertised transport."""

    replica: ReplicaId
    worker: WorkerInstanceId
    device: DeviceResourceId
    recipe: ReplicaRecipeId
    unit: WorkUnitId
    slot: SemanticSlot
    capabilities: frozenset[str]
    send: _Send
    receive: _Receive

    def __post_init__(self) -> None:
        exact = (
            ("replica", self.replica, ReplicaId),
            ("worker", self.worker, WorkerInstanceId),
            ("device", self.device, DeviceResourceId),
            ("recipe", self.recipe, ReplicaRecipeId),
            ("unit", self.unit, WorkUnitId),
            ("slot", self.slot, SemanticSlot),
        )
        for name, value, expected in exact:
            if type(value) is not expected:
                raise ValueError(f"candidate {name} has the wrong type")
        if type(self.capabilities) is not frozenset or any(
            type(value) is not str for value in self.capabilities
        ):
            raise ValueError("candidate capabilities must be a frozenset of strings")
        if WORKGROUP_CAPABILITY not in negotiate_workgroup_capabilities(self.capabilities):
            raise ValueError(f"candidate capabilities do not negotiate {WORKGROUP_CAPABILITY}")
        for name in ("send", "receive"):
            if not callable(getattr(self, name)):
                raise ValueError(f"candidate {name} must be callable")


def build_workgroup_configuration(
    group: WorkGroupId,
    attempt: WorkGroupAttempt,
    candidates: Sequence[WorkGroupLaneCandidate],
) -> tuple[WorkGroupDefinition, tuple[ReplicaEndpoint, ...]]:
    """Validate and canonically bind a complete explicit lane set."""

    if type(group) is not WorkGroupId or type(attempt) is not WorkGroupAttempt:
        raise ValueError("configuration requires exact group and attempt identities")
    if not isinstance(candidates, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("candidates must be a bounded nonempty tuple")
    if not 1 <= len(candidates) <= MAX_WORKGROUP_REPLICAS:
        raise ValueError("candidates must be a bounded nonempty tuple")
    try:
        candidate_tuple = tuple(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be a bounded nonempty tuple") from exc
    if len(candidate_tuple) != len(candidates):
        raise ValueError("candidates must be a bounded nonempty tuple")
    for index, candidate in enumerate(candidate_tuple):
        if type(candidate) is not WorkGroupLaneCandidate:
            raise ValueError(f"candidate {index} has the wrong type")

    definition, ordered = _plan_workgroup_definition(group, attempt, candidate_tuple)
    by_replica = {candidate.replica: candidate for candidate in ordered}
    endpoints = tuple(
        ReplicaEndpoint.bind(
            definition,
            member.replica,
            send=by_replica[member.replica].send,
            receive=by_replica[member.replica].receive,
        )
        for member in definition.members
    )
    return definition, endpoints


class _LaneIdentity(Protocol):
    @property
    def replica(self) -> ReplicaId: ...

    @property
    def worker(self) -> WorkerInstanceId: ...

    @property
    def device(self) -> DeviceResourceId: ...

    @property
    def recipe(self) -> ReplicaRecipeId: ...

    @property
    def unit(self) -> WorkUnitId: ...

    @property
    def slot(self) -> SemanticSlot: ...


_LaneT = TypeVar("_LaneT", bound=_LaneIdentity)


def _plan_workgroup_definition(
    group: WorkGroupId,
    attempt: WorkGroupAttempt,
    lanes: tuple[_LaneT, ...],
) -> tuple[WorkGroupDefinition, tuple[_LaneT, ...]]:
    ordered = tuple(sorted(lanes, key=_candidate_key))
    reasons = _duplicate_reasons(ordered)
    if reasons:
        raise ValueError("; ".join(reasons))

    definition = WorkGroupDefinition(
        group=group,
        attempt=attempt,
        members=tuple(
            ReplicaBinding(
                replica=candidate.replica,
                worker=candidate.worker,
                device=candidate.device,
                recipe=candidate.recipe,
            )
            for candidate in ordered
        ),
        units=tuple(
            WorkUnitDefinition(
                unit=candidate.unit,
                replica=candidate.replica,
                slot=candidate.slot,
            )
            for candidate in ordered
        ),
    )
    return definition, ordered


def _candidate_key(candidate: _LaneIdentity) -> tuple[str, ...]:
    return (
        candidate.replica.value,
        candidate.worker.value,
        candidate.device.value,
        candidate.recipe.value,
        candidate.unit.value,
        candidate.slot.value,
    )


def _duplicate_reasons(
    candidates: tuple[_LaneIdentity, ...],
) -> tuple[str, ...]:
    facts: tuple[tuple[str, Callable[[_LaneIdentity], str]], ...] = (
        ("replica", lambda candidate: candidate.replica.value),
        ("worker", lambda candidate: candidate.worker.value),
        ("device", lambda candidate: candidate.device.value),
        ("unit", lambda candidate: candidate.unit.value),
        (
            "replica/slot",
            lambda candidate: f"{candidate.replica.value}/{candidate.slot.value}",
        ),
    )
    reasons: list[str] = []
    for name, fact in facts:
        values = tuple(fact(candidate) for candidate in candidates)
        duplicates = {value for value, count in Counter(values).items() if count > 1}
        reasons.extend(f"duplicate {name} {value}" for value in sorted(duplicates))
    return tuple(reasons)
