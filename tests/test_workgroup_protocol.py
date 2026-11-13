from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    AbortWorkGroup,
    BeginAdmission,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaBinding,
    ReplicaId,
    ReplicaReady,
    ReplicaRecipeId,
    ReplicaRefused,
    RequestCancellation,
    RunWorkUnit,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupLifecycle,
    WorkGroupPrepared,
    WorkGroupProtocolError,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
    reduce_workgroup,
    workgroup_message_from_wire,
    workgroup_message_to_wire,
)


def recipe() -> ReplicaRecipeId:
    return ReplicaRecipeId("sha256:" + "a" * 64)


def definition(attempt: int = 1) -> WorkGroupDefinition:
    return WorkGroupDefinition(
        WorkGroupId("group-1"),
        WorkGroupAttempt(attempt),
        (
            ReplicaBinding(
                ReplicaId("replica-b"),
                WorkerInstanceId("worker-b"),
                DeviceResourceId("device-b"),
                recipe(),
            ),
            ReplicaBinding(
                ReplicaId("replica-a"),
                WorkerInstanceId("worker-a"),
                DeviceResourceId("device-a"),
                recipe(),
            ),
        ),
        (
            WorkUnitDefinition(WorkUnitId("unit-b"), ReplicaId("replica-b"), SemanticSlot.SINGLE),
            WorkUnitDefinition(WorkUnitId("unit-a"), ReplicaId("replica-a"), SemanticSlot.SINGLE),
        ),
    )


def child_kwargs(replica: str = "a", attempt: int = 1) -> dict[str, Any]:
    return {
        "worker": WorkerInstanceId(f"worker-{replica}"),
        "replica": ReplicaId(f"replica-{replica}"),
        "group": WorkGroupId("group-1"),
        "attempt": WorkGroupAttempt(attempt),
        "device": DeviceResourceId(f"device-{replica}"),
    }


def unit_kwargs(replica: str = "a", attempt: int = 1) -> dict[str, Any]:
    return {
        **child_kwargs(replica, attempt),
        "unit": WorkUnitId(f"unit-{replica}"),
        "slot": SemanticSlot.SINGLE,
    }


def messages() -> tuple[object, ...]:
    child = child_kwargs()
    unit = unit_kwargs()
    return (
        PrepareReplica(**child, recipe=recipe()),
        ReplicaReady(**child),
        ReplicaRefused(**child, reason="no capacity"),
        BeginWorkGroup(**child),
        WorkGroupPrepared(**child),
        WorkGroupRefused(**child, reason="no admission"),
        CommitWorkGroup(**child),
        AbortWorkGroup(**child),
        RunWorkUnit(**unit),
        WorkUnitProgress(**unit, completed=1, total=2),
        WorkUnitResult(**unit),
        WorkUnitFailed(**unit, reason="unit failed"),
        CancelWorkGroup(**child, reason="cancel"),
        WorkGroupCancelled(**child),
        ReleaseWorkGroup(**child),
        WorkGroupReleased(**child),
    )


@pytest.mark.parametrize("message", messages())
def test_all_closed_messages_round_trip(message: object) -> None:
    wire = workgroup_message_to_wire(message)  # type: ignore[arg-type]
    assert wire["version"] == 1
    assert wire["capability"] == WORKGROUP_CAPABILITY
    assert workgroup_message_from_wire(wire) == message
    with pytest.raises(FrozenInstanceError):
        message.group = WorkGroupId("other")  # type: ignore[attr-defined]


def test_cancellation_during_admission_cancels_every_contacted_member() -> None:
    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    state, commands = reduce_workgroup(state, RequestCancellation("stop"))
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    state, _ = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("a")))
    assert state.prepared == (ReplicaId("replica-a"),)


def test_fresh_attempt_fences_every_prior_attempt_response() -> None:
    prior = WorkGroupPrepared(**child_kwargs("a", attempt=1))
    fresh_definition = definition(attempt=1)
    fresh_definition = replace(fresh_definition, attempt=fresh_definition.attempt.next())
    state, _ = reduce_workgroup(WorkGroupLifecycle(fresh_definition), BeginAdmission())
    with pytest.raises(WorkGroupProtocolError, match="stale"):
        reduce_workgroup(state, prior)
