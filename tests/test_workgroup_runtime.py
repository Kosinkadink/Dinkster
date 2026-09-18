from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, replace
from functools import wraps
from typing import Any

import pytest
from dinkster_memory import ReservationRequest
from dinkster_protocol import (
    AbortWorkGroup,
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
    RunWorkUnit,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupLifecycle,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkGroupState,
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
)
from dinkster_workers import (
    ReplicaEndpoint,
    WorkGroupCoordinator,
    WorkGroupPreparationRefused,
    WorkGroupRuntimeError,
)


def async_test(
    function: Callable[[], Coroutine[Any, Any, None]],
) -> Callable[[], None]:
    @wraps(function)
    def run() -> None:
        asyncio.run(function())

    return run


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


def child_kwargs(replica: str, attempt: int = 1) -> dict[str, Any]:
    return {
        "worker": WorkerInstanceId(f"worker-{replica}"),
        "replica": ReplicaId(f"replica-{replica}"),
        "group": WorkGroupId("group-1"),
        "attempt": WorkGroupAttempt(attempt),
        "device": DeviceResourceId(f"device-{replica}"),
    }


def unit_kwargs(replica: str, attempt: int = 1) -> dict[str, Any]:
    return {
        **child_kwargs(replica, attempt),
        "unit": WorkUnitId(f"unit-{replica}"),
        "slot": SemanticSlot.SINGLE,
    }


class RecordingReservations:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.requests: tuple[ReservationRequest, ...] = ()

    @asynccontextmanager
    async def reserve(self, requests: Sequence[ReservationRequest]) -> AsyncIterator[None]:
        self.requests = tuple(requests)
        self.events.append("reserve-enter")
        try:
            yield
        finally:
            self.events.append("reserve-exit")


class FakeReplica:
    def __init__(
        self,
        d: WorkGroupDefinition,
        replica: str,
        replies: tuple[WorkGroupMessage | BaseException, ...],
        events: list[str],
    ) -> None:
        self.replica = replica
        self.replies: asyncio.Queue[WorkGroupMessage | BaseException] = asyncio.Queue()
        for reply in replies:
            self.replies.put_nowait(reply)
        self.sent: list[WorkGroupMessage] = []
        self.events = events
        self.command_events: dict[type[object], asyncio.Event] = {
            kind: asyncio.Event()
            for kind in (
                PrepareReplica,
                BeginWorkGroup,
                RunWorkUnit,
                CancelWorkGroup,
                ReleaseWorkGroup,
            )
        }
        self.endpoint = ReplicaEndpoint.bind(
            d,
            ReplicaId(f"replica-{replica}"),
            send=self.send,
            receive=self.receive,
        )

    async def send(self, message: WorkGroupMessage) -> None:
        self.sent.append(message)
        self.events.append(f"send-{message.TYPE}-{self.replica}")
        event = self.command_events.get(type(message))
        if event is not None:
            event.set()

    async def receive(self) -> WorkGroupMessage:
        reply = await self.replies.get()
        trigger = {
            ReplicaReady: PrepareReplica,
            ReplicaRefused: PrepareReplica,
            WorkGroupPrepared: BeginWorkGroup,
            WorkGroupRefused: BeginWorkGroup,
            WorkUnitProgress: RunWorkUnit,
            WorkUnitResult: RunWorkUnit,
            WorkUnitFailed: RunWorkUnit,
            WorkGroupCancelled: CancelWorkGroup,
            WorkGroupReleased: ReleaseWorkGroup,
        }.get(type(reply), RunWorkUnit)
        await self.command_events[trigger].wait()
        if isinstance(reply, BaseException):
            self.events.append(f"recv-error-{self.replica}")
            raise reply
        self.events.append(f"recv-{reply.TYPE}-{self.replica}")
        return reply


def success_replies(replica: str) -> tuple[WorkGroupMessage, ...]:
    return (
        ReplicaReady(**child_kwargs(replica)),
        WorkGroupPrepared(**child_kwargs(replica)),
        WorkUnitProgress(**unit_kwargs(replica), completed=1, total=1),
        WorkUnitResult(**unit_kwargs(replica)),
        WorkGroupReleased(**child_kwargs(replica)),
    )


@async_test
async def test_success_is_canonical_reserved_and_released_exactly_once() -> None:
    d = definition()
    events: list[str] = []
    reservations = RecordingReservations(events)
    a = FakeReplica(d, "a", success_replies("a"), events)
    b = FakeReplica(d, "b", success_replies("b"), events)

    lifecycle = await WorkGroupCoordinator(reservations).execute(
        d,
        (b.endpoint, a.endpoint),
        (ReservationRequest("vram:device-a", 10),),
    )

    assert lifecycle.state is WorkGroupState.SUCCEEDED
    assert lifecycle.released == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    assert reservations.requests == (ReservationRequest("vram:device-a", 10),)
    sends = [event for event in events if event.startswith("send-")]
    assert sends[:2] == ["send-prepareReplica-a", "send-prepareReplica-b"]
    assert [type(message) for message in a.sent] == [
        PrepareReplica,
        BeginWorkGroup,
        CommitWorkGroup,
        RunWorkUnit,
        ReleaseWorkGroup,
    ]
    assert [type(message) for message in b.sent] == [
        PrepareReplica,
        BeginWorkGroup,
        CommitWorkGroup,
        RunWorkUnit,
        ReleaseWorkGroup,
    ]
    assert events[0] == "reserve-enter"
    assert events[-1] == "reserve-exit"


@async_test
async def test_prepare_refusal_drains_and_rolls_back_every_contacted_member() -> None:
    d = definition()
    events: list[str] = []
    reservations = RecordingReservations(events)
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaRefused(**child_kwargs("a"), reason="no capacity"),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (ReplicaReady(**child_kwargs("b")), WorkGroupReleased(**child_kwargs("b"))),
        events,
    )

    with pytest.raises(WorkGroupPreparationRefused, match="preparation refused") as caught:
        await WorkGroupCoordinator(reservations).execute(d, (a.endpoint, b.endpoint))

    assert caught.value.lifecycle.state is WorkGroupState.DEFINED
    assert caught.value.refusals == (ReplicaRefused(**child_kwargs("a"), reason="no capacity"),)
    for fake in (a, b):
        assert [type(message) for message in fake.sent] == [
            PrepareReplica,
            AbortWorkGroup,
            ReleaseWorkGroup,
        ]
    assert events[-1] == "reserve-exit"


@async_test
async def test_partial_admission_refusal_rolls_back_without_dispatch() -> None:
    d = definition()
    events: list[str] = []
    failure_notifications: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupRefused(**child_kwargs("a"), reason="busy"),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d,
        (a.endpoint, b.endpoint),
        on_failure=lambda: failure_notifications.append("failed"),
    )

    assert lifecycle.state is WorkGroupState.REFUSED
    assert lifecycle.refusals == ((ReplicaId("replica-a"), "busy"),)
    assert failure_notifications == []
    assert not any(
        isinstance(message, (CommitWorkGroup, RunWorkUnit)) for message in (*a.sent, *b.sent)
    )
    for fake in (a, b):
        assert sum(isinstance(message, AbortWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1


@async_test
async def test_failure_drains_progress_then_cancels_releases_and_settles() -> None:
    d = definition()
    events: list[str] = []
    failure_notifications: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupPrepared(**child_kwargs("a")),
            WorkUnitFailed(**unit_kwargs("a"), reason="failed"),
            WorkGroupCancelled(**child_kwargs("a")),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkUnitProgress(**unit_kwargs("b"), completed=1, total=2),
            WorkUnitResult(**unit_kwargs("b")),
            WorkGroupCancelled(**child_kwargs("b")),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d,
        (a.endpoint, b.endpoint),
        on_failure=lambda: failure_notifications.append("failed"),
    )

    assert lifecycle.state is WorkGroupState.FAILED
    assert lifecycle.failures == ((WorkUnitId("unit-a"), "failed"),)
    assert failure_notifications == ["failed"]
    assert lifecycle.cancelled == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    assert lifecycle.released == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1


@async_test
async def test_concurrent_failures_drain_then_cancel_release_and_settle() -> None:
    d = definition()
    events: list[str] = []

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkUnitFailed(**unit_kwargs(replica), reason=f"{replica} failed"),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = FakeReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d, (a.endpoint, b.endpoint)
    )

    assert lifecycle.state is WorkGroupState.FAILED
    assert lifecycle.failures == (
        (WorkUnitId("unit-a"), "a failed"),
        (WorkUnitId("unit-b"), "b failed"),
    )
    assert lifecycle.cancelled == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    assert lifecycle.released == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1
    assert events[-1] == "reserve-exit"


@async_test
async def test_cancel_ack_may_overtake_a_late_work_unit_failure() -> None:
    d = definition()
    events: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupPrepared(**child_kwargs("a")),
            WorkUnitFailed(**unit_kwargs("a"), reason="a failed"),
            WorkGroupCancelled(**child_kwargs("a")),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkGroupCancelled(**child_kwargs("b")),
            WorkUnitFailed(**unit_kwargs("b"), reason="b failed"),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d, (a.endpoint, b.endpoint)
    )

    assert lifecycle.state is WorkGroupState.FAILED
    assert lifecycle.failures == (
        (WorkUnitId("unit-a"), "a failed"),
        (WorkUnitId("unit-b"), "b failed"),
    )
    assert lifecycle.cancelled == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    assert lifecycle.released == (ReplicaId("replica-a"), ReplicaId("replica-b"))
    assert "recv-workUnitFailed-b" in events
    assert events[-1] == "reserve-exit"


@async_test
async def test_gather_failure_uses_reducer_cleanup_without_payload_math() -> None:
    d = definition()
    events: list[str] = []

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkUnitResult(**unit_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = FakeReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)

    async def gather_failed(_: object) -> str:
        return "gather failed"

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d,
        (a.endpoint, b.endpoint),
        gather=gather_failed,
    )

    assert lifecycle.state is WorkGroupState.FAILED
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1


@async_test
async def test_empty_gather_failure_reason_is_rejected_and_cleaned_up() -> None:
    d = definition()
    events: list[str] = []

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkUnitResult(**unit_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = FakeReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)

    async def gather_failed(_: object) -> str:
        return ""

    with pytest.raises(WorkGroupRuntimeError, match="workgroup runtime failed") as caught:
        await WorkGroupCoordinator(RecordingReservations(events)).execute(
            d,
            (a.endpoint, b.endpoint),
            gather=gather_failed,
        )

    assert caught.value.lifecycle.state is WorkGroupState.CANCELLED
    assert events[-1] == "reserve-exit"
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1


@async_test
async def test_gather_runtime_error_still_cleans_before_reservation_exit() -> None:
    d = definition()
    events: list[str] = []

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkUnitResult(**unit_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = FakeReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)

    async def gather_error(lifecycle: WorkGroupLifecycle) -> None:
        raise WorkGroupRuntimeError("gather callback failed", lifecycle)

    with pytest.raises(WorkGroupRuntimeError, match="gather callback failed") as caught:
        await WorkGroupCoordinator(RecordingReservations(events)).execute(
            d,
            (a.endpoint, b.endpoint),
            gather=gather_error,
        )

    assert caught.value.lifecycle.state is WorkGroupState.CANCELLED
    assert caught.value.lifecycle.released == (
        ReplicaId("replica-a"),
        ReplicaId("replica-b"),
    )
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1
    assert events[-1] == "reserve-exit"


@async_test
async def test_caller_cancellation_finishes_cleanup_before_releasing_reservation() -> None:
    d = definition()
    events: list[str] = []
    running = asyncio.Event()

    class BlockingReplica(FakeReplica):
        async def send(self, message: WorkGroupMessage) -> None:
            await super().send(message)
            if isinstance(message, RunWorkUnit):
                running.set()

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = BlockingReplica(d, "a", replies("a"), events)
    b = BlockingReplica(d, "b", replies("b"), events)
    task = asyncio.create_task(
        WorkGroupCoordinator(RecordingReservations(events)).execute(d, (a.endpoint, b.endpoint))
    )
    await running.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events[-1] == "reserve-exit"
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1
        assert events.index(f"recv-workGroupCancelled-{fake.replica}") < events.index(
            f"send-releaseWorkGroup-{fake.replica}"
        )


@async_test
async def test_admission_failure_records_prepared_reply_observed_during_cleanup() -> None:
    d = definition()
    events: list[str] = []

    class LatePreparedReplica(FakeReplica):
        async def receive(self) -> WorkGroupMessage:
            reply = await super().receive()
            if isinstance(reply, WorkGroupPrepared):
                await self.command_events[CancelWorkGroup].wait()
            return reply

    a = LatePreparedReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupPrepared(**child_kwargs("a")),
            WorkGroupCancelled(**child_kwargs("a")),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b", attempt=2)),
        ),
        events,
    )

    with pytest.raises(WorkGroupRuntimeError, match="endpoint reply") as caught:
        await WorkGroupCoordinator(RecordingReservations(events), cleanup_timeout=0.05).execute(
            d, (a.endpoint, b.endpoint)
        )

    assert caught.value.lifecycle.prepared == (ReplicaId("replica-a"),)
    assert sum(isinstance(message, CancelWorkGroup) for message in a.sent) == 1
    assert sum(isinstance(message, ReleaseWorkGroup) for message in a.sent) == 1
    assert events[-1] == "reserve-exit"


@async_test
async def test_cancellation_during_ambiguous_run_send_is_not_retried_or_retyped() -> None:
    d = definition()
    events: list[str] = []
    send_started = asyncio.Event()

    class BlockingSendReplica(FakeReplica):
        async def send(self, message: WorkGroupMessage) -> None:
            await super().send(message)
            if isinstance(message, RunWorkUnit):
                send_started.set()
                await asyncio.Future()

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = BlockingSendReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)
    task = asyncio.create_task(
        WorkGroupCoordinator(RecordingReservations(events)).execute(d, (a.endpoint, b.endpoint))
    )
    await send_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sum(isinstance(message, RunWorkUnit) for message in a.sent) == 1
    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1
    assert events[-1] == "reserve-exit"


@async_test
async def test_blocked_cleanup_send_cannot_starve_reachable_replica() -> None:
    d = definition()
    events: list[str] = []
    running = asyncio.Event()

    class BlockingCleanupReplica(FakeReplica):
        async def send(self, message: WorkGroupMessage) -> None:
            await super().send(message)
            if isinstance(message, RunWorkUnit):
                running.set()
            if isinstance(message, CancelWorkGroup):
                await asyncio.Future()

    def replies(replica: str) -> tuple[WorkGroupMessage, ...]:
        return (
            ReplicaReady(**child_kwargs(replica)),
            WorkGroupPrepared(**child_kwargs(replica)),
            WorkGroupCancelled(**child_kwargs(replica)),
            WorkGroupReleased(**child_kwargs(replica)),
        )

    a = BlockingCleanupReplica(d, "a", replies("a"), events)
    b = FakeReplica(d, "b", replies("b"), events)
    # The blocked replica always consumes its whole split of the cleanup
    # budget, so the budget must be generous enough that scheduling stalls
    # on a loaded CI runner cannot eat the reachable replica's share.
    task = asyncio.create_task(
        WorkGroupCoordinator(RecordingReservations(events), cleanup_timeout=2.0).execute(
            d, (a.endpoint, b.endpoint)
        )
    )
    await running.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10.0)

    assert sum(isinstance(message, CancelWorkGroup) for message in b.sent) == 1
    assert sum(isinstance(message, ReleaseWorkGroup) for message in b.sent) == 1
    assert events[-1] == "reserve-exit"


@async_test
async def test_post_release_eof_cannot_preempt_another_release_barrier() -> None:
    d = definition()
    events: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (*success_replies("a"), EOFError("closed after release")),
        events,
    )
    b = FakeReplica(d, "b", success_replies("b"), events)

    lifecycle = await WorkGroupCoordinator(RecordingReservations(events)).execute(
        d, (a.endpoint, b.endpoint)
    )

    assert lifecycle.state is WorkGroupState.SUCCEEDED
    assert lifecycle.released == (ReplicaId("replica-a"), ReplicaId("replica-b"))


@async_test
async def test_foreign_and_stale_replies_are_fenced_before_acceptance() -> None:
    d = definition()
    events: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (ReplicaReady(**child_kwargs("a", attempt=2)),),
        events,
    )
    b = FakeReplica(d, "b", (ReplicaReady(**child_kwargs("b")),), events)

    with pytest.raises(WorkGroupRuntimeError, match="endpoint reply"):
        await WorkGroupCoordinator(RecordingReservations(events), cleanup_timeout=0.01).execute(
            d, (a.endpoint, b.endpoint)
        )


def test_endpoint_is_immutable_attempt_bound_and_validates_before_send() -> None:
    d = definition()
    events: list[str] = []
    fake = FakeReplica(d, "a", (), events)
    with pytest.raises(FrozenInstanceError):
        fake.endpoint.binding = d.members[1]  # type: ignore[misc]
    fresh = replace(d, attempt=d.attempt.next())

    async def exercise() -> None:
        with pytest.raises(ValueError, match="bound"):
            await fake.endpoint.send(
                PrepareReplica(**child_kwargs("a", attempt=2), recipe=recipe())
            )

    asyncio.run(exercise())
    assert fake.sent == []
    with pytest.raises(ValueError, match="definition"):
        WorkGroupCoordinator(RecordingReservations(events)).validate(fresh, (fake.endpoint,))


@async_test
async def test_duplicate_progress_is_rejected_then_cleaned_up_once() -> None:
    d = definition()
    events: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupPrepared(**child_kwargs("a")),
            WorkUnitProgress(**unit_kwargs("a"), completed=1, total=2),
            WorkUnitProgress(**unit_kwargs("a"), completed=1, total=2),
            WorkGroupCancelled(**child_kwargs("a")),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkGroupCancelled(**child_kwargs("b")),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    with pytest.raises(WorkGroupRuntimeError, match="runtime failed"):
        await WorkGroupCoordinator(RecordingReservations(events)).execute(
            d, (a.endpoint, b.endpoint)
        )

    for fake in (a, b):
        assert sum(isinstance(message, CancelWorkGroup) for message in fake.sent) == 1
        assert sum(isinstance(message, ReleaseWorkGroup) for message in fake.sent) == 1


@async_test
async def test_retry_attempt_rejects_prior_attempt_reply() -> None:
    d = definition(attempt=2)
    events: list[str] = []
    a = FakeReplica(d, "a", (ReplicaReady(**child_kwargs("a", attempt=1)),), events)
    b = FakeReplica(d, "b", (ReplicaReady(**child_kwargs("b", attempt=2)),), events)

    with pytest.raises(WorkGroupRuntimeError, match="endpoint reply"):
        await WorkGroupCoordinator(RecordingReservations(events), cleanup_timeout=0.01).execute(
            d, (a.endpoint, b.endpoint)
        )


@async_test
async def test_admission_refusal_release_failure_still_releases_peer() -> None:
    d = definition()
    events: list[str] = []

    class ReleaseFailingReplica(FakeReplica):
        async def send(self, message: WorkGroupMessage) -> None:
            await super().send(message)
            if isinstance(message, ReleaseWorkGroup):
                raise ConnectionError("release transport lost")

    a = ReleaseFailingReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupRefused(**child_kwargs("a"), reason="busy"),
            WorkGroupReleased(**child_kwargs("a")),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    with pytest.raises(WorkGroupRuntimeError, match="transport lost"):
        await WorkGroupCoordinator(RecordingReservations(events)).execute(
            d, (a.endpoint, b.endpoint)
        )

    assert sum(isinstance(message, ReleaseWorkGroup) for message in b.sent) == 1
    assert "recv-workGroupReleased-b" in events
    assert events[-1] == "reserve-exit"


@async_test
async def test_transport_loss_releases_parent_lease_without_fabricating_terminal_state() -> None:
    d = definition()
    events: list[str] = []
    a = FakeReplica(
        d,
        "a",
        (
            ReplicaReady(**child_kwargs("a")),
            WorkGroupPrepared(**child_kwargs("a")),
            ConnectionError("lost"),
        ),
        events,
    )
    b = FakeReplica(
        d,
        "b",
        (
            ReplicaReady(**child_kwargs("b")),
            WorkGroupPrepared(**child_kwargs("b")),
            WorkGroupCancelled(**child_kwargs("b")),
            WorkGroupReleased(**child_kwargs("b")),
        ),
        events,
    )

    def failure_started() -> None:
        events.append("failure-started")
        raise LookupError("notification failed")

    with pytest.raises(WorkGroupRuntimeError, match="transport lost") as caught:
        await WorkGroupCoordinator(RecordingReservations(events), cleanup_timeout=0.05).execute(
            d,
            (a.endpoint, b.endpoint),
            on_failure=failure_started,
        )

    assert caught.value.lifecycle.state is not WorkGroupState.FAILED
    assert events[-1] == "reserve-exit"
    assert events.index("failure-started") < events.index("send-cancelWorkGroup-b")
    assert sum(isinstance(message, CancelWorkGroup) for message in b.sent) == 1
    assert sum(isinstance(message, ReleaseWorkGroup) for message in b.sent) == 1
    assert events.index("recv-workGroupCancelled-b") < events.index("send-releaseWorkGroup-b")
