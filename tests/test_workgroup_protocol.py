from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from typing import Any, cast

import pytest
from dinkster_protocol import (
    MAX_REASON_BYTES,
    MAX_WORKGROUP_REPLICAS,
    MAX_WORKGROUP_UNITS,
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    WORKGROUP_VERSION,
    AbortWorkGroup,
    BeginAdmission,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    DispatchWork,
    GatherFailed,
    GatherSucceeded,
    InvocationResult,
    NodeError,
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
    SettleFailure,
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
    WorkGroupState,
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
    negotiate_result_capabilities,
    negotiate_workgroup_capabilities,
    reduce_workgroup,
    workgroup_message_from_wire,
    workgroup_message_to_wire,
)
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import PROTOCOL_VERSION, ValueCodec, encode_result


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


def prepared_lifecycle() -> WorkGroupLifecycle:
    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    for replica in ("b", "a"):
        state, _ = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs(replica)))
    assert state.state is WorkGroupState.DISPATCHING
    return state


def running_lifecycle() -> WorkGroupLifecycle:
    state, _ = reduce_workgroup(prepared_lifecycle(), DispatchWork())
    return state


def test_constants_capabilities_and_identity_bounds_are_strict() -> None:
    assert WORKGROUP_CAPABILITY == "dinkster.multidevice-workgroup.v1"
    assert WORKGROUP_DATA_PLANE_CAPABILITY == "dinkster.single-job-workgroup.v2"
    assert WORKGROUP_VERSION == 1
    assert negotiate_workgroup_capabilities([]) == frozenset()
    assert negotiate_workgroup_capabilities([WORKGROUP_CAPABILITY, "unknown"]) == frozenset(
        {WORKGROUP_CAPABILITY}
    )
    assert negotiate_workgroup_capabilities(
        [WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY]
    ) == frozenset({WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY})
    for invalid in (None, WORKGROUP_CAPABILITY, (WORKGROUP_CAPABILITY,), [1]):
        with pytest.raises(WorkGroupProtocolError):
            negotiate_workgroup_capabilities(invalid)
    with pytest.raises(WorkGroupProtocolError, match="unique"):
        negotiate_workgroup_capabilities([WORKGROUP_CAPABILITY, WORKGROUP_CAPABILITY])
    for kind in (ReplicaId, WorkerInstanceId, WorkGroupId, WorkUnitId, DeviceResourceId):
        assert kind("a")
        assert kind("a" * 128)
        for invalid in ("", "A", "-a", "a-", "a/b", "a" * 129, 1):
            with pytest.raises(WorkGroupProtocolError):
                kind(invalid)  # type: ignore[arg-type]
    for invalid in (True, 0, -1, 1.0):
        with pytest.raises(WorkGroupProtocolError):
            WorkGroupAttempt(invalid)  # type: ignore[arg-type]
    assert WorkGroupAttempt(1).next() == WorkGroupAttempt(2)
    for invalid in ("sha256:" + "A" * 64, "sha256:" + "a" * 63, "a" * 64):
        with pytest.raises(WorkGroupProtocolError):
            ReplicaRecipeId(invalid)
    assert tuple(SemanticSlot) == (SemanticSlot.SINGLE,)


def test_definitions_are_frozen_closed_bounded_and_canonical() -> None:
    d = definition()
    assert tuple(member.replica.value for member in d.members) == ("replica-a", "replica-b")
    assert tuple(unit.unit.value for unit in d.units) == ("unit-a", "unit-b")
    with pytest.raises(FrozenInstanceError):
        d.group = WorkGroupId("other")  # type: ignore[misc]
    with pytest.raises(WorkGroupProtocolError, match="tuple"):
        replace(d, members=list(d.members))  # type: ignore[arg-type]
    with pytest.raises(WorkGroupProtocolError, match="unknown"):
        replace(
            d,
            units=(
                WorkUnitDefinition(
                    WorkUnitId("unit-z"), ReplicaId("replica-z"), SemanticSlot.SINGLE
                ),
            ),
        )
    with pytest.raises(WorkGroupProtocolError, match="duplicate"):
        replace(d, members=(d.members[0], d.members[0]))
    with pytest.raises(WorkGroupProtocolError, match="duplicate"):
        replace(d, units=(d.units[0], d.units[0]))
    with pytest.raises(WorkGroupProtocolError, match="bounded"):
        replace(d, members=())
    with pytest.raises(WorkGroupProtocolError, match="bounded"):
        replace(d, members=(d.members[0],) * (MAX_WORKGROUP_REPLICAS + 1))
    with pytest.raises(WorkGroupProtocolError, match="bounded"):
        replace(d, units=())
    with pytest.raises(WorkGroupProtocolError, match="bounded"):
        replace(d, units=(d.units[0],) * (MAX_WORKGROUP_UNITS + 1))


@pytest.mark.parametrize("message", messages())
def test_all_closed_messages_round_trip(message: object) -> None:
    wire = workgroup_message_to_wire(message)  # type: ignore[arg-type]
    assert wire["version"] == 1
    assert wire["capability"] == WORKGROUP_CAPABILITY
    assert workgroup_message_from_wire(wire) == message
    with pytest.raises(FrozenInstanceError):
        message.group = WorkGroupId("other")  # type: ignore[attr-defined]


def test_all_closed_messages_have_frozen_type_names_and_field_sets() -> None:
    expected = (
        ("prepareReplica", {"recipe"}),
        ("replicaReady", set()),
        ("replicaRefused", {"reason"}),
        ("beginWorkGroup", set()),
        ("workGroupPrepared", set()),
        ("workGroupRefused", {"reason"}),
        ("commitWorkGroup", set()),
        ("abortWorkGroup", set()),
        ("runWorkUnit", {"unit", "slot"}),
        ("workUnitProgress", {"unit", "slot", "completed", "total"}),
        ("workUnitResult", {"unit", "slot"}),
        ("workUnitFailed", {"unit", "slot", "reason"}),
        ("cancelWorkGroup", {"reason"}),
        ("workGroupCancelled", set()),
        ("releaseWorkGroup", set()),
        ("workGroupReleased", set()),
    )
    correlation = {
        "type",
        "version",
        "capability",
        "workerInstance",
        "replica",
        "group",
        "attempt",
        "device",
    }
    for message, (type_name, extra) in zip(messages(), expected, strict=True):
        wire = workgroup_message_to_wire(message)  # type: ignore[arg-type]
        assert wire["type"] == type_name
        assert set(wire) == correlation | extra


def test_wire_rejects_unknown_missing_mutable_and_malformed_shapes() -> None:
    wire = workgroup_message_to_wire(WorkUnitProgress(**unit_kwargs(), completed=1, total=2))
    mutations: list[object] = [
        list(wire.items()),
        {**wire, "extra": 1},
        {key: value for key, value in wire.items() if key != "unit"},
        {**wire, "type": "WorkUnitProgress"},
        {**wire, "version": True},
        {**wire, "version": 2},
        {**wire, "capability": "unknown"},
        {**wire, "capability": SemanticSlot.SINGLE},
        {**wire, "attempt": True},
        {**wire, "replica": "Replica-A"},
        {**wire, "slot": "positive"},
        {**wire, "completed": True},
        {**wire, "completed": 3},
    ]
    for value in mutations:
        with pytest.raises(WorkGroupProtocolError):
            workgroup_message_from_wire(value)
    with pytest.raises(WorkGroupProtocolError):
        WorkUnitProgress(**unit_kwargs(), completed=True, total=1)  # type: ignore[arg-type]
    with pytest.raises(WorkGroupProtocolError):
        WorkUnitResult(**{**unit_kwargs(), "worker": "worker-a"})  # type: ignore[arg-type]
    for reason in ("", "x" * (MAX_REASON_BYTES + 1), "\ud800"):
        with pytest.raises(WorkGroupProtocolError):
            WorkUnitFailed(**unit_kwargs(), reason=reason)


def test_admission_preparation_dispatch_progress_result_and_success() -> None:
    state, commands = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    assert state.state is WorkGroupState.ADMITTING
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    state, commands = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("b")))
    assert commands == ()
    state, commands = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("a")))
    assert state.state is WorkGroupState.DISPATCHING
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    state, commands = reduce_workgroup(state, DispatchWork())
    assert state.state is WorkGroupState.RUNNING
    assert [cast(RunWorkUnit, command).unit.value for command in commands] == [
        "unit-a",
        "unit-b",
    ]
    state, commands = reduce_workgroup(
        state, WorkUnitProgress(**unit_kwargs("b"), completed=1, total=2)
    )
    assert commands == ()
    state, _ = reduce_workgroup(state, WorkUnitProgress(**unit_kwargs("b"), completed=2, total=2))
    state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("b")))
    state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("a")))
    assert state.state is WorkGroupState.GATHERING
    assert [unit.value for unit in state.completed] == ["unit-a", "unit-b"]
    state, commands = reduce_workgroup(state, GatherSucceeded())
    assert state.state is WorkGroupState.SUCCEEDED
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    for replica in ("b", "a"):
        state, _ = reduce_workgroup(state, WorkGroupReleased(**child_kwargs(replica)))
    assert [replica.value for replica in state.released] == ["replica-a", "replica-b"]


def test_refusal_rolls_back_every_member_without_dispatch() -> None:
    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    state, _ = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("b")))
    state, commands = reduce_workgroup(
        state, WorkGroupRefused(**child_kwargs("a"), reason="refused")
    )
    assert state.state is WorkGroupState.REFUSED
    assert state.refusals == ((ReplicaId("replica-a"), "refused"),)
    assert [type(command) for command in commands] == [AbortWorkGroup, AbortWorkGroup]
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]


def test_late_preparation_after_refusal_is_retained_without_forward_commands() -> None:
    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    state, _ = reduce_workgroup(state, WorkGroupRefused(**child_kwargs("b"), reason="refused"))
    state, commands = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("a")))
    assert state.state is WorkGroupState.REFUSED
    assert state.prepared == (ReplicaId("replica-a"),)
    assert commands == ()


@pytest.mark.parametrize("before_dispatch", [True, False])
def test_cancellation_before_and_after_dispatch_is_complete(before_dispatch: bool) -> None:
    state = prepared_lifecycle()
    if not before_dispatch:
        state, _ = reduce_workgroup(state, DispatchWork())
    state, commands = reduce_workgroup(state, RequestCancellation("stop"))
    assert state.state is WorkGroupState.CANCELLING
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    state, _ = reduce_workgroup(state, WorkGroupCancelled(**child_kwargs("b")))
    state, commands = reduce_workgroup(state, WorkGroupCancelled(**child_kwargs("a")))
    assert state.state is WorkGroupState.CANCELLED
    assert [type(command) for command in commands] == [ReleaseWorkGroup, ReleaseWorkGroup]


def test_cancellation_during_admission_cancels_every_contacted_member() -> None:
    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    state, commands = reduce_workgroup(state, RequestCancellation("stop"))
    assert [command.replica.value for command in commands] == ["replica-a", "replica-b"]
    state, _ = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs("a")))
    assert state.prepared == (ReplicaId("replica-a"),)


@pytest.mark.parametrize("gather_failure", [False, True])
def test_failure_cleanup_requires_cancel_and_release_before_settlement(
    gather_failure: bool,
) -> None:
    state = running_lifecycle()
    if gather_failure:
        state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("a")))
        state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("b")))
        failure: object = GatherFailed("gather failed")
    else:
        failure = WorkUnitFailed(**unit_kwargs("a"), reason="unit failed")
    state, commands = reduce_workgroup(state, failure)  # type: ignore[arg-type]
    assert state.state is WorkGroupState.FAILING
    assert state.failures == (() if gather_failure else ((WorkUnitId("unit-a"), "unit failed"),))
    assert [type(command) for command in commands] == [CancelWorkGroup, CancelWorkGroup]
    with pytest.raises(WorkGroupProtocolError, match="incomplete"):
        reduce_workgroup(state, SettleFailure())
    for replica in ("b", "a"):
        state, commands = reduce_workgroup(state, WorkGroupCancelled(**child_kwargs(replica)))
    assert [type(command) for command in commands] == [ReleaseWorkGroup, ReleaseWorkGroup]
    for replica in ("a", "b"):
        state, _ = reduce_workgroup(state, WorkGroupReleased(**child_kwargs(replica)))
    state, commands = reduce_workgroup(state, SettleFailure())
    assert state.state is WorkGroupState.FAILED
    assert commands == ()


def test_failure_rejects_release_until_cancellation_emits_cleanup_once() -> None:
    state, _ = reduce_workgroup(
        running_lifecycle(), WorkUnitFailed(**unit_kwargs("a"), reason="unit failed")
    )
    original = state
    with pytest.raises(WorkGroupProtocolError):
        reduce_workgroup(state, WorkGroupReleased(**child_kwargs("a")))
    assert state == original
    state, commands = reduce_workgroup(state, WorkGroupCancelled(**child_kwargs("a")))
    assert commands == ()
    state, commands = reduce_workgroup(state, WorkGroupCancelled(**child_kwargs("b")))
    assert [(type(command), command.replica.value) for command in commands] == [
        (ReleaseWorkGroup, "replica-a"),
        (ReleaseWorkGroup, "replica-b"),
    ]
    for replica in ("a", "b"):
        state, commands = reduce_workgroup(state, WorkGroupReleased(**child_kwargs(replica)))
        assert commands == ()
    with pytest.raises(WorkGroupProtocolError):
        reduce_workgroup(state, WorkGroupReleased(**child_kwargs("a")))


def test_progress_duplicates_regressions_total_changes_and_completed_units_reject() -> None:
    state = running_lifecycle()
    state, _ = reduce_workgroup(state, WorkUnitProgress(**unit_kwargs("a"), completed=1, total=3))
    for completed, total in ((1, 3), (0, 3), (2, 4)):
        with pytest.raises(WorkGroupProtocolError):
            reduce_workgroup(
                state,
                WorkUnitProgress(**unit_kwargs("a"), completed=completed, total=total),
            )
    state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("a")))
    with pytest.raises(WorkGroupProtocolError):
        reduce_workgroup(state, WorkUnitProgress(**unit_kwargs("a"), completed=2, total=3))


def test_response_permutations_retain_canonical_facts_and_commands() -> None:
    snapshots = []
    for order in permutations(("a", "b")):
        state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
        command_types = []
        for replica in order:
            state, commands = reduce_workgroup(state, WorkGroupPrepared(**child_kwargs(replica)))
            command_types.append(tuple((type(c).__name__, c.replica.value) for c in commands))
        state, commands = reduce_workgroup(state, DispatchWork())
        for replica in reversed(order):
            state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs(replica)))
        snapshots.append(
            (
                state.prepared,
                state.completed,
                tuple(cast(RunWorkUnit, c).unit for c in commands),
                command_types[-1],
            )
        )
    assert snapshots[0][:3] == snapshots[1][:3]
    assert snapshots[0][3] == snapshots[1][3]


def test_stale_future_foreign_duplicate_illegal_and_terminal_events_reject_immutably() -> None:
    state = running_lifecycle()
    original = state
    bad_events = (
        WorkUnitResult(**unit_kwargs("a", attempt=2)),
        WorkUnitResult(**{**unit_kwargs("a"), "worker": WorkerInstanceId("worker-z")}),
        WorkUnitResult(**{**unit_kwargs("a"), "device": DeviceResourceId("device-z")}),
        WorkUnitResult(**{**unit_kwargs("a"), "group": WorkGroupId("group-z")}),
        WorkUnitResult(**{**unit_kwargs("a"), "unit": WorkUnitId("unit-z")}),
    )
    for event in bad_events:
        with pytest.raises(WorkGroupProtocolError):
            reduce_workgroup(state, event)
        assert state == original
    with pytest.raises(WorkGroupProtocolError):
        WorkUnitResult(**{**unit_kwargs("a"), "slot": "single"})  # type: ignore[arg-type]
    state, _ = reduce_workgroup(state, WorkUnitResult(**unit_kwargs("a")))
    with pytest.raises(WorkGroupProtocolError, match="duplicate"):
        reduce_workgroup(state, WorkUnitResult(**unit_kwargs("a")))
    with pytest.raises(WorkGroupProtocolError):
        reduce_workgroup(state, DispatchWork())
    terminal, _ = reduce_workgroup(prepared_lifecycle(), RequestCancellation("stop"))
    for replica in ("a", "b"):
        terminal, _ = reduce_workgroup(terminal, WorkGroupCancelled(**child_kwargs(replica)))
    with pytest.raises(WorkGroupProtocolError):
        reduce_workgroup(terminal, RequestCancellation("again"))


def test_fresh_attempt_fences_every_prior_attempt_response() -> None:
    prior = WorkGroupPrepared(**child_kwargs("a", attempt=1))
    fresh_definition = definition(attempt=1)
    fresh_definition = replace(fresh_definition, attempt=fresh_definition.attempt.next())
    state, _ = reduce_workgroup(WorkGroupLifecycle(fresh_definition), BeginAdmission())
    with pytest.raises(WorkGroupProtocolError, match="stale"):
        reduce_workgroup(state, prior)


def test_lifecycle_rejects_mutable_foreign_duplicate_and_incoherent_facts() -> None:
    d = definition()
    with pytest.raises(WorkGroupProtocolError, match="tuples"):
        WorkGroupLifecycle(d, prepared=[])  # type: ignore[arg-type]
    with pytest.raises(WorkGroupProtocolError, match="canonical"):
        WorkGroupLifecycle(d, prepared=(ReplicaId("replica-a"), ReplicaId("replica-a")))
    with pytest.raises(WorkGroupProtocolError, match="dispatched"):
        WorkGroupLifecycle(d, completed=(WorkUnitId("unit-a"),))
    for state in (WorkGroupState.RUNNING, WorkGroupState.GATHERING, WorkGroupState.SUCCEEDED):
        with pytest.raises(WorkGroupProtocolError):
            WorkGroupLifecycle(d, state=state)
    with pytest.raises(WorkGroupProtocolError, match="cleanup"):
        WorkGroupLifecycle(
            d,
            state=WorkGroupState.FAILED,
            prepared=(ReplicaId("replica-a"), ReplicaId("replica-b")),
            dispatched=(WorkUnitId("unit-a"), WorkUnitId("unit-b")),
        )
    with pytest.raises(FrozenInstanceError):
        WorkGroupLifecycle(d).state = WorkGroupState.RUNNING  # type: ignore[misc]


def test_lifecycle_rejects_every_constructible_unreachable_shape() -> None:
    d = definition()
    replicas = (ReplicaId("replica-a"), ReplicaId("replica-b"))
    units = (WorkUnitId("unit-a"), WorkUnitId("unit-b"))
    hostile = (
        lambda: WorkGroupLifecycle(d, state=WorkGroupState.CANCELLING, dispatched=(units[0],)),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.CANCELLING,
            prepared=(replicas[0],),
            dispatched=units,
        ),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.CANCELLED,
            dispatched=units,
            cancelled=replicas,
        ),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.DISPATCHING,
            prepared=replicas,
            dispatched=units,
        ),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.RUNNING,
            prepared=replicas,
            dispatched=units,
            cancelled=(replicas[0],),
        ),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.RUNNING,
            prepared=replicas,
            dispatched=units,
            released=(replicas[0],),
        ),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.FAILING,
            prepared=replicas,
            dispatched=units,
            released=(replicas[0],),
        ),
        lambda: WorkGroupLifecycle(d, state=WorkGroupState.ADMITTING, prepared=replicas),
        lambda: WorkGroupLifecycle(
            d,
            state=WorkGroupState.RUNNING,
            prepared=replicas,
            dispatched=units,
            completed=units,
        ),
        lambda: WorkGroupLifecycle(d, state=WorkGroupState.CANCELLING, cancelled=replicas),
    )
    for construct in hostile:
        with pytest.raises(WorkGroupProtocolError):
            construct()


def test_cancelled_rejects_late_prepared_progress_and_result_without_mutation() -> None:
    admission, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    admission, _ = reduce_workgroup(admission, RequestCancellation("stop"))
    for replica in ("a", "b"):
        admission, _ = reduce_workgroup(admission, WorkGroupCancelled(**child_kwargs(replica)))
    running, _ = reduce_workgroup(running_lifecycle(), RequestCancellation("stop"))
    for replica in ("a", "b"):
        running, _ = reduce_workgroup(running, WorkGroupCancelled(**child_kwargs(replica)))
    cases = (
        (admission, WorkGroupPrepared(**child_kwargs("a"))),
        (running, WorkUnitProgress(**unit_kwargs("a"), completed=1, total=2)),
        (running, WorkUnitResult(**unit_kwargs("a"))),
    )
    for lifecycle, event in cases:
        original = lifecycle
        with pytest.raises(WorkGroupProtocolError):
            reduce_workgroup(lifecycle, event)
        assert lifecycle == original


def test_reducer_rejects_foreign_subclasses() -> None:
    class ForeignPrepared(WorkGroupPrepared):
        pass

    class ForeignBegin(BeginAdmission):
        pass

    state, _ = reduce_workgroup(WorkGroupLifecycle(definition()), BeginAdmission())
    with pytest.raises(WorkGroupProtocolError, match="exact"):
        reduce_workgroup(state, ForeignPrepared(**child_kwargs("a")))
    with pytest.raises(WorkGroupProtocolError, match="exact"):
        reduce_workgroup(WorkGroupLifecycle(definition()), ForeignBegin())


def test_boundary_protocol_version_preserves_legacy_error_results() -> None:
    assert PROTOCOL_VERSION == 9
    header, blobs, segments = encode_result(
        ValueCodec(TypeRegistry(), use_shm=False),
        InvocationResult(error=NodeError("node-1", "Node", "failed")),
        "inv-1",
        1.5,
    )
    assert header == {
        "type": "result",
        "invocationId": "inv-1",
        "executeMs": 1.5,
        "error": {
            "nodeId": "node-1",
            "nodeType": "Node",
            "message": "failed",
            "traceback": "",
            "hints": [],
        },
    }
    assert blobs == []
    assert segments == []
    assert negotiate_result_capabilities([]) == frozenset()
    assert negotiate_result_capabilities(["dinkster.invocation-result-algebra.v1"]) == frozenset(
        {"dinkster.invocation-result-algebra.v1"}
    )
