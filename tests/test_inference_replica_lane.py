from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from dinkster_inference.replica_lane import (
    ReplicaAdmissionRefusal,
    ReplicaLaneError,
    ReplicaLaneRuntime,
    ReplicaLaneState,
    ReplicaPreparationRefusal,
    ReplicaUnitFailure,
)
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
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkGroupState,
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitId,
    WorkUnitResult,
)
from dinkster_workers import ReplicaEndpoint, WorkGroupCoordinator, WorkGroupPreparationRefused


def _definition(attempt: int = 1) -> WorkGroupDefinition:
    recipe = ReplicaRecipeId("sha256:" + "a" * 64)
    return WorkGroupDefinition(
        WorkGroupId("group"),
        WorkGroupAttempt(attempt),
        tuple(
            ReplicaBinding(
                ReplicaId(f"replica-{name}"),
                WorkerInstanceId(f"worker-{name}"),
                DeviceResourceId(f"device-{name}"),
                recipe,
            )
            for name in ("a", "b")
        ),
        tuple(
            WorkUnitDefinition(
                WorkUnitId(f"unit-{name}"), ReplicaId(f"replica-{name}"), SemanticSlot.SINGLE
            )
            for name in ("a", "b")
        ),
    )


def _commands(d: WorkGroupDefinition, name: str) -> dict[str, WorkGroupMessage]:
    binding = next(member for member in d.members if member.replica == ReplicaId(f"replica-{name}"))
    return {
        "prepare": PrepareReplica(
            binding.worker, binding.replica, d.group, d.attempt, binding.device, binding.recipe
        ),
        "begin": BeginWorkGroup(
            binding.worker, binding.replica, d.group, d.attempt, binding.device
        ),
        "commit": CommitWorkGroup(
            binding.worker, binding.replica, d.group, d.attempt, binding.device
        ),
        "run": RunWorkUnit(
            binding.worker,
            binding.replica,
            d.group,
            d.attempt,
            binding.device,
            WorkUnitId(f"unit-{name}"),
            SemanticSlot.SINGLE,
        ),
        "cancel": CancelWorkGroup(
            binding.worker, binding.replica, d.group, d.attempt, binding.device, "stop"
        ),
        "abort": AbortWorkGroup(
            binding.worker, binding.replica, d.group, d.attempt, binding.device
        ),
        "release": ReleaseWorkGroup(
            binding.worker, binding.replica, d.group, d.attempt, binding.device
        ),
    }


class RecordingHost:
    def __init__(
        self,
        *,
        preparation: ReplicaPreparationRefusal | None = None,
        admission: ReplicaAdmissionRefusal | None = None,
        unit: ReplicaUnitFailure | None = None,
    ) -> None:
        self.preparation = preparation
        self.admission = admission
        self.unit = unit
        self.calls: list[str] = []
        self.raise_on: str | None = None

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.raise_on == name:
            raise RuntimeError(f"{name} failed")

    def prepare(self) -> ReplicaPreparationRefusal | None:
        self._call("prepare")
        return self.preparation

    def admit(self) -> ReplicaAdmissionRefusal | None:
        self._call("admit")
        return self.admission

    def activate(self) -> None:
        self._call("activate")

    def run_unit(self, unit: WorkUnitDefinition) -> ReplicaUnitFailure | None:
        self._call(f"run:{unit.unit.value}")
        return self.unit

    def abort(self) -> None:
        self._call("abort")

    def cancel(self, reason: str) -> None:
        self._call(f"cancel:{reason}")

    def release(self) -> None:
        self._call("release")


def _advance(runtime: ReplicaLaneRuntime, commands: dict[str, WorkGroupMessage]) -> None:
    assert type(runtime.handle(commands["prepare"])[0]) is ReplicaReady
    assert type(runtime.handle(commands["begin"])[0]) is WorkGroupPrepared
    assert runtime.handle(commands["commit"]) == ()


def test_constructor_and_command_fences_precede_host_calls() -> None:
    d = _definition()
    host = RecordingHost()
    with pytest.raises(ReplicaLaneError, match="absent"):
        ReplicaLaneRuntime(d, ReplicaId("foreign"), host)
    assert host.calls == []

    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    prepare = _commands(d, "a")["prepare"]
    mutations = (
        replace(prepare, worker=WorkerInstanceId("foreign")),
        replace(prepare, replica=ReplicaId("replica-b")),
        replace(prepare, group=WorkGroupId("foreign")),
        replace(prepare, attempt=WorkGroupAttempt(2)),
        replace(prepare, device=DeviceResourceId("foreign")),
        replace(prepare, recipe=ReplicaRecipeId("sha256:" + "b" * 64)),
        ReplicaReady(
            worker=WorkerInstanceId("worker-a"),
            replica=ReplicaId("replica-a"),
            group=d.group,
            attempt=d.attempt,
            device=DeviceResourceId("device-a"),
        ),
    )
    for message in mutations:
        with pytest.raises(ReplicaLaneError):
            runtime.handle(message)
    assert host.calls == []


def test_legal_lifecycle_unit_result_release_and_absorbing_state() -> None:
    d = _definition()
    host = RecordingHost()
    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    commands = _commands(d, "a")
    _advance(runtime, commands)
    result = runtime.handle(commands["run"])
    assert type(result[0]) is WorkUnitResult
    released = runtime.handle(commands["release"])
    assert type(released[0]) is WorkGroupReleased
    assert runtime.state is ReplicaLaneState.RELEASED
    assert host.calls == ["prepare", "admit", "activate", "run:unit-a", "release"]
    for command in commands.values():
        with pytest.raises(ReplicaLaneError):
            runtime.handle(command)
    assert host.calls.count("release") == 1


def test_refusal_failure_cancel_and_abort_are_exact() -> None:
    d = _definition()
    commands = _commands(d, "a")

    prepare_host = RecordingHost(preparation=ReplicaPreparationRefusal("no memory"))
    prepare_runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), prepare_host)
    assert type(prepare_runtime.handle(commands["prepare"])[0]) is ReplicaRefused
    assert prepare_runtime.handle(commands["abort"]) == ()
    assert type(prepare_runtime.handle(commands["release"])[0]) is WorkGroupReleased
    assert prepare_host.calls == ["prepare", "abort", "release"]

    admission_host = RecordingHost(admission=ReplicaAdmissionRefusal("not admitted"))
    admission_runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), admission_host)
    admission_runtime.handle(commands["prepare"])
    assert type(admission_runtime.handle(commands["begin"])[0]) is WorkGroupRefused
    assert type(admission_runtime.handle(commands["cancel"])[0]) is WorkGroupCancelled
    assert type(admission_runtime.handle(commands["release"])[0]) is WorkGroupReleased

    failed_host = RecordingHost(unit=ReplicaUnitFailure("unit failed"))
    failed_runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), failed_host)
    _advance(failed_runtime, commands)
    assert type(failed_runtime.handle(commands["run"])[0]) is WorkUnitFailed
    with pytest.raises(ReplicaLaneError, match="duplicate"):
        failed_runtime.handle(commands["run"])
    assert type(failed_runtime.handle(commands["cancel"])[0]) is WorkGroupCancelled
    assert type(failed_runtime.handle(commands["release"])[0]) is WorkGroupReleased


def test_illegal_order_unit_and_slot_refuse_before_host_calls() -> None:
    d = _definition()
    host = RecordingHost()
    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    commands = _commands(d, "a")
    for name in ("begin", "commit", "run", "cancel", "abort", "release"):
        with pytest.raises(ReplicaLaneError):
            runtime.handle(commands[name])
    foreign_unit = replace(commands["run"], unit=WorkUnitId("unit-b"))
    foreign_slot = replace(commands["run"], replica=ReplicaId("replica-b"))
    for command in (foreign_unit, foreign_slot):
        with pytest.raises(ReplicaLaneError):
            runtime.handle(command)
    assert host.calls == []


def test_unexpected_error_propagates_and_release_is_still_exactly_once() -> None:
    d = _definition()
    host = RecordingHost()
    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    commands = _commands(d, "a")
    _advance(runtime, commands)
    host.raise_on = "run:unit-a"
    with pytest.raises(RuntimeError, match="run:unit-a failed"):
        runtime.handle(commands["run"])
    host.raise_on = None
    runtime.handle(commands["cancel"])
    runtime.handle(commands["release"])
    assert host.calls[-2:] == ["cancel:stop", "release"]
    assert host.calls.count("release") == 1


@pytest.mark.parametrize(
    ("command", "failure_call"),
    (("prepare", "prepare"), ("begin", "admit"), ("commit", "activate"), ("run", "run:unit-a")),
)
def test_host_failure_command_retry_is_fenced_before_a_second_call(
    command: str, failure_call: str
) -> None:
    d = _definition()
    host = RecordingHost()
    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    commands = _commands(d, "a")
    if command != "prepare":
        runtime.handle(commands["prepare"])
    if command not in ("prepare", "begin"):
        runtime.handle(commands["begin"])
    if command == "run":
        runtime.handle(commands["commit"])
    host.raise_on = failure_call
    with pytest.raises(RuntimeError, match="failed"):
        runtime.handle(commands[command])
    before = tuple(host.calls)
    with pytest.raises(ReplicaLaneError, match="duplicate"):
        runtime.handle(commands[command])
    assert tuple(host.calls) == before


def test_release_failure_is_terminal_and_fresh_attempt_needs_fresh_runtime() -> None:
    d = _definition()
    commands = _commands(d, "a")
    host = RecordingHost()
    runtime = ReplicaLaneRuntime(d, ReplicaId("replica-a"), host)
    runtime.handle(commands["prepare"])
    host.raise_on = "release"
    with pytest.raises(RuntimeError, match="release failed"):
        runtime.handle(commands["release"])
    assert runtime.state is ReplicaLaneState.RELEASED
    with pytest.raises(ReplicaLaneError):
        runtime.handle(commands["release"])
    assert host.calls.count("release") == 1

    retry = _definition(2)
    retry_runtime = ReplicaLaneRuntime(retry, ReplicaId("replica-a"), RecordingHost())
    with pytest.raises(ReplicaLaneError, match="stale"):
        retry_runtime.handle(commands["prepare"])
    assert type(retry_runtime.handle(_commands(retry, "a")["prepare"])[0]) is ReplicaReady


class _Reservations:
    @asynccontextmanager
    async def reserve(self, requests: Sequence[ReservationRequest]) -> AsyncIterator[None]:
        del requests
        yield


class _RuntimeEndpoint:
    def __init__(
        self,
        definition: WorkGroupDefinition,
        name: str,
        host: RecordingHost,
        *,
        delay_run: bool = False,
    ) -> None:
        replica = ReplicaId(f"replica-{name}")
        self.host = host
        self.runtime = ReplicaLaneRuntime(definition, replica, host)
        self.replies: asyncio.Queue[WorkGroupMessage] = asyncio.Queue()
        self.run_seen = asyncio.Event()
        self.delay_run = delay_run
        self.endpoint = ReplicaEndpoint.bind(
            definition,
            replica,
            send=self.send,
            receive=self.replies.get,
        )

    async def send(self, message: WorkGroupMessage) -> None:
        if type(message) is RunWorkUnit and self.delay_run:
            self.run_seen.set()
            return
        for reply in self.runtime.handle(message):
            self.replies.put_nowait(reply)


def test_real_coordinator_composition_success_refusal_and_unit_failure() -> None:
    async def run() -> None:
        d = _definition()
        success = tuple(_RuntimeEndpoint(d, name, RecordingHost()) for name in ("a", "b"))
        lifecycle = await WorkGroupCoordinator(_Reservations()).execute(
            d, tuple(item.endpoint for item in success)
        )
        assert lifecycle.state is WorkGroupState.SUCCEEDED
        assert all(item.runtime.state is ReplicaLaneState.RELEASED for item in success)

        refused = (
            _RuntimeEndpoint(d, "a", RecordingHost(preparation=ReplicaPreparationRefusal("no"))),
            _RuntimeEndpoint(d, "b", RecordingHost()),
        )
        with pytest.raises(WorkGroupPreparationRefused):
            await WorkGroupCoordinator(_Reservations()).execute(
                d, tuple(item.endpoint for item in refused)
            )
        assert all(item.runtime.state is ReplicaLaneState.RELEASED for item in refused)

        failed = (
            _RuntimeEndpoint(d, "a", RecordingHost(unit=ReplicaUnitFailure("bad"))),
            _RuntimeEndpoint(d, "b", RecordingHost()),
        )
        lifecycle = await WorkGroupCoordinator(_Reservations()).execute(
            d, tuple(item.endpoint for item in failed)
        )
        assert lifecycle.state is WorkGroupState.FAILED
        assert all(item.runtime.state is ReplicaLaneState.RELEASED for item in failed)

    asyncio.run(run())


def test_real_coordinator_cancellation_cancels_then_releases() -> None:
    async def run() -> None:
        d = _definition()
        replicas = tuple(
            _RuntimeEndpoint(d, name, RecordingHost(), delay_run=True) for name in ("a", "b")
        )
        task = asyncio.create_task(
            WorkGroupCoordinator(_Reservations()).execute(
                d, tuple(item.endpoint for item in replicas)
            )
        )
        await asyncio.gather(*(item.run_seen.wait() for item in replicas))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(item.runtime.state is ReplicaLaneState.RELEASED for item in replicas)
        for item in replicas:
            assert item.host.calls[-2:] == ["cancel:runtime cleanup", "release"]

    asyncio.run(run())
