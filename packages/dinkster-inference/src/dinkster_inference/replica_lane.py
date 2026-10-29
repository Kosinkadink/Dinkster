"""Strict replica-side execution for one immutable workgroup attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypedDict, runtime_checkable

from dinkster_protocol import (
    MAX_REASON_BYTES,
    AbortWorkGroup,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaId,
    ReplicaReady,
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
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitResult,
)


class _ReplyIdentity(TypedDict):
    worker: WorkerInstanceId
    replica: ReplicaId
    group: WorkGroupId
    attempt: WorkGroupAttempt
    device: DeviceResourceId


class ReplicaLaneError(ValueError):
    """A command violates the bound replica attempt or its lifecycle."""


def _validate_reason(reason: object) -> None:
    if type(reason) is not str or not reason:
        raise ReplicaLaneError("reason must be a bounded nonempty UTF-8 string")
    try:
        encoded = reason.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReplicaLaneError("reason must be a bounded nonempty UTF-8 string") from exc
    if len(encoded) > MAX_REASON_BYTES:
        raise ReplicaLaneError("reason must be a bounded nonempty UTF-8 string")


@dataclass(frozen=True, slots=True)
class ReplicaPreparationRefusal:
    reason: str

    def __post_init__(self) -> None:
        _validate_reason(self.reason)


@dataclass(frozen=True, slots=True)
class ReplicaAdmissionRefusal:
    reason: str

    def __post_init__(self) -> None:
        _validate_reason(self.reason)


@dataclass(frozen=True, slots=True)
class ReplicaUnitFailure:
    reason: str

    def __post_init__(self) -> None:
        _validate_reason(self.reason)


class ReplicaLaneState(StrEnum):
    IDLE = "IDLE"
    READY = "READY"
    REFUSED = "REFUSED"
    ADMITTED = "ADMITTED"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"
    RELEASED = "RELEASED"


@runtime_checkable
class ReplicaLaneHost(Protocol):
    """Synchronous lane-local effects driven by the closed protocol lifecycle.

    A method that raises must settle its own partial effects because that command
    is fenced against retry and preparation failure cannot advance to release.
    """

    def prepare(self) -> ReplicaPreparationRefusal | None: ...

    def admit(self) -> ReplicaAdmissionRefusal | None: ...

    def activate(self) -> None: ...

    def run_unit(self, unit: WorkUnitDefinition) -> ReplicaUnitFailure | None: ...

    def abort(self) -> None: ...

    def cancel(self, reason: str) -> None: ...

    def release(self) -> None: ...


_COMMAND_TYPES = (
    PrepareReplica,
    BeginWorkGroup,
    CommitWorkGroup,
    AbortWorkGroup,
    RunWorkUnit,
    CancelWorkGroup,
    ReleaseWorkGroup,
)


class ReplicaLaneRuntime:
    """Consume commands for one exact definition member and attempt."""

    def __init__(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
        host: object,
    ) -> None:
        if type(definition) is not WorkGroupDefinition or type(replica) is not ReplicaId:
            raise ReplicaLaneError("runtime requires exact definition and replica identities")
        binding = next((member for member in definition.members if member.replica == replica), None)
        if binding is None:
            raise ReplicaLaneError("runtime replica is absent from the definition")
        if not isinstance(host, ReplicaLaneHost):
            raise ReplicaLaneError("host does not satisfy ReplicaLaneHost")
        self.definition = definition
        self.binding = binding
        self.host = host
        self.state = ReplicaLaneState.IDLE
        self._completed: set[object] = set()
        self._prepare_called = False
        self._admit_called = False
        self._activate_called = False
        self._release_called = False

    def _fence(self, message: WorkGroupMessage) -> WorkUnitDefinition | None:
        if type(message) not in _COMMAND_TYPES:
            raise ReplicaLaneError("runtime requires an exact command type")
        binding = self.binding
        if (
            message.worker,
            message.replica,
            message.group,
            message.attempt,
            message.device,
        ) != (
            binding.worker,
            binding.replica,
            self.definition.group,
            self.definition.attempt,
            binding.device,
        ):
            raise ReplicaLaneError("stale, future, or foreign command correlation")
        if type(message) is PrepareReplica and message.recipe != binding.recipe:
            raise ReplicaLaneError("prepare command has a foreign replica recipe")
        if type(message) is not RunWorkUnit:
            return None
        unit = next(
            (candidate for candidate in self.definition.units if candidate.unit == message.unit),
            None,
        )
        if unit is None or (message.replica, message.slot) != (unit.replica, unit.slot):
            raise ReplicaLaneError("run command has a foreign unit or semantic slot")
        if unit.replica != binding.replica or unit.slot is not SemanticSlot.SINGLE:
            raise ReplicaLaneError("runtime accepts only its exact SINGLE-slot unit")
        return unit

    def handle(self, message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
        unit = self._fence(message)
        binding = self.binding
        common: _ReplyIdentity = {
            "worker": binding.worker,
            "replica": binding.replica,
            "group": self.definition.group,
            "attempt": self.definition.attempt,
            "device": binding.device,
        }
        if self.state is ReplicaLaneState.RELEASED:
            raise ReplicaLaneError("released runtime is absorbing")

        if type(message) is PrepareReplica:
            if self.state is not ReplicaLaneState.IDLE or self._prepare_called:
                raise ReplicaLaneError("prepare is illegal or duplicate")
            self._prepare_called = True
            outcome = self.host.prepare()
            if outcome is None:
                self.state = ReplicaLaneState.READY
                return (ReplicaReady(**common),)
            if type(outcome) is not ReplicaPreparationRefusal:
                raise ReplicaLaneError("prepare returned an invalid outcome")
            self.state = ReplicaLaneState.REFUSED
            return (ReplicaRefused(**common, reason=outcome.reason),)

        if type(message) is BeginWorkGroup:
            if self.state is not ReplicaLaneState.READY or self._admit_called:
                raise ReplicaLaneError("begin is illegal or duplicate")
            self._admit_called = True
            outcome = self.host.admit()
            if outcome is None:
                self.state = ReplicaLaneState.ADMITTED
                return (WorkGroupPrepared(**common),)
            if type(outcome) is not ReplicaAdmissionRefusal:
                raise ReplicaLaneError("admit returned an invalid outcome")
            self.state = ReplicaLaneState.REFUSED
            return (WorkGroupRefused(**common, reason=outcome.reason),)

        if type(message) is CommitWorkGroup:
            if self.state is not ReplicaLaneState.ADMITTED or self._activate_called:
                raise ReplicaLaneError("commit is illegal or duplicate")
            self._activate_called = True
            self.host.activate()
            self.state = ReplicaLaneState.ACTIVE
            return ()

        if type(message) is RunWorkUnit:
            if self.state is not ReplicaLaneState.ACTIVE or unit is None:
                raise ReplicaLaneError("run is not active")
            if unit.unit in self._completed:
                raise ReplicaLaneError("run unit is duplicate")
            self._completed.add(unit.unit)
            outcome = self.host.run_unit(unit)
            if outcome is not None and type(outcome) is not ReplicaUnitFailure:
                raise ReplicaLaneError("run_unit returned an invalid outcome")
            if outcome is None:
                return (WorkUnitResult(**common, unit=unit.unit, slot=unit.slot),)
            return (
                WorkUnitFailed(
                    **common,
                    unit=unit.unit,
                    slot=unit.slot,
                    reason=outcome.reason,
                ),
            )

        if type(message) is AbortWorkGroup:
            if self.state not in (
                ReplicaLaneState.READY,
                ReplicaLaneState.REFUSED,
                ReplicaLaneState.ADMITTED,
            ):
                raise ReplicaLaneError("abort is illegal or duplicate")
            self.state = ReplicaLaneState.ABORTED
            self.host.abort()
            return ()

        if type(message) is CancelWorkGroup:
            if self.state not in (
                ReplicaLaneState.READY,
                ReplicaLaneState.REFUSED,
                ReplicaLaneState.ADMITTED,
                ReplicaLaneState.ACTIVE,
            ):
                raise ReplicaLaneError("cancel is illegal or duplicate")
            self.state = ReplicaLaneState.CANCELLED
            self.host.cancel(message.reason)
            return (WorkGroupCancelled(**common),)

        if type(message) is ReleaseWorkGroup:
            if self.state not in (
                ReplicaLaneState.READY,
                ReplicaLaneState.REFUSED,
                ReplicaLaneState.ADMITTED,
                ReplicaLaneState.ACTIVE,
                ReplicaLaneState.CANCELLED,
                ReplicaLaneState.ABORTED,
            ):
                raise ReplicaLaneError("release is illegal or duplicate")
            if self._release_called:
                raise ReplicaLaneError("release is duplicate")
            self._release_called = True
            self.state = ReplicaLaneState.RELEASED
            self.host.release()
            return (WorkGroupReleased(**common),)

        raise ReplicaLaneError("unsupported command")


__all__ = [
    "ReplicaAdmissionRefusal",
    "ReplicaLaneError",
    "ReplicaLaneHost",
    "ReplicaLaneRuntime",
    "ReplicaLaneState",
    "ReplicaPreparationRefusal",
    "ReplicaUnitFailure",
    "WorkUnitDefinition",
]
