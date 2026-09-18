"""Inert, transport-independent MultiDevice workgroup protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias, cast

__all__ = [
    "WORKGROUP_CAPABILITY",
    "WORKGROUP_DATA_PLANE_CAPABILITY",
    "WORKGROUP_VERSION",
    "MAX_WORKGROUP_REPLICAS",
    "MAX_WORKGROUP_UNITS",
    "MAX_REASON_BYTES",
    "WorkGroupProtocolError",
    "ReplicaId",
    "WorkerInstanceId",
    "WorkGroupId",
    "WorkGroupAttempt",
    "WorkUnitId",
    "DeviceResourceId",
    "ReplicaRecipeId",
    "SemanticSlot",
    "ReplicaBinding",
    "WorkUnitDefinition",
    "WorkGroupDefinition",
    "PrepareReplica",
    "ReplicaReady",
    "ReplicaRefused",
    "BeginWorkGroup",
    "WorkGroupPrepared",
    "WorkGroupRefused",
    "CommitWorkGroup",
    "AbortWorkGroup",
    "RunWorkUnit",
    "WorkUnitProgress",
    "WorkUnitResult",
    "WorkUnitFailed",
    "CancelWorkGroup",
    "WorkGroupCancelled",
    "ReleaseWorkGroup",
    "WorkGroupReleased",
    "WorkGroupMessage",
    "workgroup_message_to_wire",
    "workgroup_message_from_wire",
    "negotiate_workgroup_capabilities",
    "WorkGroupState",
    "WorkGroupLifecycle",
    "BeginAdmission",
    "DispatchWork",
    "GatherSucceeded",
    "GatherFailed",
    "RequestCancellation",
    "SettleFailure",
    "WorkGroupEvent",
    "reduce_workgroup",
]

WORKGROUP_CAPABILITY = "dinkster.multidevice-workgroup.v1"
WORKGROUP_DATA_PLANE_CAPABILITY = "dinkster.single-job-workgroup.v2"
WORKGROUP_VERSION = 1
MAX_WORKGROUP_REPLICAS = 256
MAX_WORKGROUP_UNITS = 4096
MAX_REASON_BYTES = 4096


class WorkGroupProtocolError(ValueError):
    """A closed-protocol validation or state-transition error."""


_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")


@dataclass(frozen=True)
class _Id:
    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or _ID.fullmatch(self.value) is None:
            raise WorkGroupProtocolError(f"invalid {type(self).__name__}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ReplicaId(_Id):
    pass


@dataclass(frozen=True)
class WorkerInstanceId(_Id):
    pass


@dataclass(frozen=True)
class WorkGroupId(_Id):
    pass


@dataclass(frozen=True)
class WorkUnitId(_Id):
    pass


@dataclass(frozen=True)
class DeviceResourceId(_Id):
    pass


@dataclass(frozen=True)
class ReplicaRecipeId:
    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", self.value) is None:
            raise WorkGroupProtocolError("invalid ReplicaRecipeId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class WorkGroupAttempt:
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value < 1:
            raise WorkGroupProtocolError("attempt must be a positive integer")

    def next(self) -> WorkGroupAttempt:
        return WorkGroupAttempt(self.value + 1)


class SemanticSlot(StrEnum):
    SINGLE = "single"


def negotiate_workgroup_capabilities(capabilities: object) -> frozenset[str]:
    if type(capabilities) not in (list, set, frozenset):
        raise WorkGroupProtocolError("capabilities must be a list, set, or frozenset")
    values = cast("list[object] | set[object] | frozenset[object]", capabilities)
    if any(type(v) is not str for v in values) or len(values) != len(set(values)):
        raise WorkGroupProtocolError("capabilities must contain unique strings")
    return frozenset(
        {WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY}
        & set(cast("list[str] | set[str]", values))
    )


@dataclass(frozen=True)
class ReplicaBinding:
    replica: ReplicaId
    worker: WorkerInstanceId
    device: DeviceResourceId
    recipe: ReplicaRecipeId

    def __post_init__(self) -> None:
        _exact_fields(self)


@dataclass(frozen=True)
class WorkUnitDefinition:
    unit: WorkUnitId
    replica: ReplicaId
    slot: SemanticSlot

    def __post_init__(self) -> None:
        _exact_fields(self)


@dataclass(frozen=True)
class WorkGroupDefinition:
    group: WorkGroupId
    attempt: WorkGroupAttempt
    members: tuple[ReplicaBinding, ...]
    units: tuple[WorkUnitDefinition, ...]

    def __post_init__(self) -> None:
        _exact_fields(self, skip=("members", "units"))
        if type(self.members) is not tuple or not 1 <= len(self.members) <= MAX_WORKGROUP_REPLICAS:
            raise WorkGroupProtocolError("members must be a bounded nonempty tuple")
        if type(self.units) is not tuple or not 1 <= len(self.units) <= MAX_WORKGROUP_UNITS:
            raise WorkGroupProtocolError("units must be a bounded nonempty tuple")
        if any(type(x) is not ReplicaBinding for x in self.members) or any(
            type(x) is not WorkUnitDefinition for x in self.units
        ):
            raise WorkGroupProtocolError("definition children must have exact protocol types")
        members = tuple(sorted(self.members, key=lambda x: (x.replica.value, x.device.value)))
        units = tuple(
            sorted(self.units, key=lambda x: (x.slot.value, x.unit.value, x.replica.value))
        )
        for attr in ("replica", "worker", "device"):
            if len({getattr(x, attr) for x in members}) != len(members):
                raise WorkGroupProtocolError(f"duplicate member {attr}")
        if len({x.unit for x in units}) != len(units) or len(
            {(x.replica, x.slot) for x in units}
        ) != len(units):
            raise WorkGroupProtocolError("duplicate unit or semantic slot")
        replicas = {x.replica for x in members}
        if any(x.replica not in replicas for x in units):
            raise WorkGroupProtocolError("unit refers to an unknown replica")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "units", units)


def _reason(value: object) -> None:
    if type(value) is not str or not value:
        raise WorkGroupProtocolError("reason must be a bounded nonempty UTF-8 string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WorkGroupProtocolError("reason must be a bounded nonempty UTF-8 string") from exc
    if len(encoded) > MAX_REASON_BYTES:
        raise WorkGroupProtocolError("reason must be a bounded nonempty UTF-8 string")


def _exact_fields(value: object, *, skip: tuple[str, ...] = ()) -> None:
    hints = value.__class__.__annotations__
    for name, expected in hints.items():
        if name in skip or name == "TYPE":
            continue
        actual = getattr(value, name)
        if isinstance(expected, str):
            expected = globals().get(expected, expected)
        if isinstance(expected, type) and type(actual) is not expected:
            raise WorkGroupProtocolError(f"{name} has the wrong type")


@dataclass(frozen=True)
class _Message:
    TYPE: ClassVar[str]

    def __post_init__(self) -> None:
        if isinstance(self, _Child):
            expected = (
                (self.worker, WorkerInstanceId),
                (self.replica, ReplicaId),
                (self.group, WorkGroupId),
                (self.attempt, WorkGroupAttempt),
                (self.device, DeviceResourceId),
            )
            if any(type(value) is not kind for value, kind in expected):
                raise WorkGroupProtocolError("message correlation has the wrong type")
        if isinstance(self, _Unit) and (
            type(self.unit) is not WorkUnitId or type(self.slot) is not SemanticSlot
        ):
            raise WorkGroupProtocolError("message unit correlation has the wrong type")
        _exact_fields(self)


@dataclass(frozen=True)
class _Child(_Message):
    worker: WorkerInstanceId
    replica: ReplicaId
    group: WorkGroupId
    attempt: WorkGroupAttempt
    device: DeviceResourceId


@dataclass(frozen=True)
class _Unit(_Child):
    unit: WorkUnitId
    slot: SemanticSlot


@dataclass(frozen=True)
class PrepareReplica(_Child):
    TYPE: ClassVar[str] = "prepareReplica"
    recipe: ReplicaRecipeId


@dataclass(frozen=True)
class ReplicaReady(_Child):
    TYPE: ClassVar[str] = "replicaReady"


@dataclass(frozen=True)
class ReplicaRefused(_Child):
    TYPE: ClassVar[str] = "replicaRefused"
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _reason(self.reason)


@dataclass(frozen=True)
class BeginWorkGroup(_Child):
    TYPE: ClassVar[str] = "beginWorkGroup"


@dataclass(frozen=True)
class WorkGroupPrepared(_Child):
    TYPE: ClassVar[str] = "workGroupPrepared"


@dataclass(frozen=True)
class WorkGroupRefused(_Child):
    TYPE: ClassVar[str] = "workGroupRefused"
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _reason(self.reason)


@dataclass(frozen=True)
class CommitWorkGroup(_Child):
    TYPE: ClassVar[str] = "commitWorkGroup"


@dataclass(frozen=True)
class AbortWorkGroup(_Child):
    TYPE: ClassVar[str] = "abortWorkGroup"


@dataclass(frozen=True)
class RunWorkUnit(_Unit):
    TYPE: ClassVar[str] = "runWorkUnit"


@dataclass(frozen=True)
class WorkUnitProgress(_Unit):
    TYPE: ClassVar[str] = "workUnitProgress"
    completed: int
    total: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.completed) is not int
            or type(self.total) is not int
            or self.total < 1
            or not 0 <= self.completed <= self.total
        ):
            raise WorkGroupProtocolError("invalid progress")


@dataclass(frozen=True)
class WorkUnitResult(_Unit):
    TYPE: ClassVar[str] = "workUnitResult"


@dataclass(frozen=True)
class WorkUnitFailed(_Unit):
    TYPE: ClassVar[str] = "workUnitFailed"
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _reason(self.reason)


@dataclass(frozen=True)
class CancelWorkGroup(_Child):
    TYPE: ClassVar[str] = "cancelWorkGroup"
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _reason(self.reason)


@dataclass(frozen=True)
class WorkGroupCancelled(_Child):
    TYPE: ClassVar[str] = "workGroupCancelled"


@dataclass(frozen=True)
class ReleaseWorkGroup(_Child):
    TYPE: ClassVar[str] = "releaseWorkGroup"


@dataclass(frozen=True)
class WorkGroupReleased(_Child):
    TYPE: ClassVar[str] = "workGroupReleased"


WorkGroupMessage: TypeAlias = (
    PrepareReplica
    | ReplicaReady
    | ReplicaRefused
    | BeginWorkGroup
    | WorkGroupPrepared
    | WorkGroupRefused
    | CommitWorkGroup
    | AbortWorkGroup
    | RunWorkUnit
    | WorkUnitProgress
    | WorkUnitResult
    | WorkUnitFailed
    | CancelWorkGroup
    | WorkGroupCancelled
    | ReleaseWorkGroup
    | WorkGroupReleased
)
_MESSAGES = {
    c.TYPE: c
    for c in (
        PrepareReplica,
        ReplicaReady,
        ReplicaRefused,
        BeginWorkGroup,
        WorkGroupPrepared,
        WorkGroupRefused,
        CommitWorkGroup,
        AbortWorkGroup,
        RunWorkUnit,
        WorkUnitProgress,
        WorkUnitResult,
        WorkUnitFailed,
        CancelWorkGroup,
        WorkGroupCancelled,
        ReleaseWorkGroup,
        WorkGroupReleased,
    )
}
_WIRE_NAMES = {
    "worker": "workerInstance",
    "replica": "replica",
    "group": "group",
    "attempt": "attempt",
    "device": "device",
    "recipe": "recipe",
    "unit": "unit",
    "slot": "slot",
    "reason": "reason",
    "completed": "completed",
    "total": "total",
}
_WRAPPERS = {
    "worker": WorkerInstanceId,
    "replica": ReplicaId,
    "group": WorkGroupId,
    "attempt": WorkGroupAttempt,
    "device": DeviceResourceId,
    "recipe": ReplicaRecipeId,
    "unit": WorkUnitId,
    "slot": SemanticSlot,
}


def workgroup_message_to_wire(message: WorkGroupMessage) -> dict[str, object]:
    if type(message) not in _MESSAGES.values():
        raise WorkGroupProtocolError("unknown message")
    out: dict[str, object] = {
        "type": message.TYPE,
        "version": WORKGROUP_VERSION,
        "capability": WORKGROUP_CAPABILITY,
    }
    for f in fields(message):
        value = getattr(message, f.name)
        out[_WIRE_NAMES[f.name]] = (
            value.value
            if isinstance(value, (_Id, ReplicaRecipeId, WorkGroupAttempt, SemanticSlot))
            else value
        )
    return out


def workgroup_message_from_wire(wire: object) -> WorkGroupMessage:
    if type(wire) is not dict:
        raise WorkGroupProtocolError("message must be an ordinary dict")
    raw = cast("dict[object, object]", wire)
    if (
        type(raw.get("version")) is not int
        or raw.get("version") != WORKGROUP_VERSION
        or type(raw.get("capability")) is not str
        or raw.get("capability") != WORKGROUP_CAPABILITY
        or type(raw.get("type")) is not str
    ):
        raise WorkGroupProtocolError("wrong capability, version, or type")
    cls = _MESSAGES.get(cast("str", raw["type"]))
    if cls is None:
        raise WorkGroupProtocolError("unknown message type")
    expected = {"type", "version", "capability"} | {_WIRE_NAMES[f.name] for f in fields(cls)}
    if set(raw) != expected or any(type(k) is not str for k in raw):
        raise WorkGroupProtocolError("message has non-exact fields")
    kwargs: dict[str, object] = {}
    for f in fields(cls):
        value = raw[_WIRE_NAMES[f.name]]
        wrapper = _WRAPPERS.get(f.name)
        if wrapper is not None:
            expected_wire_type = int if wrapper is WorkGroupAttempt else str
            if type(value) is not expected_wire_type:
                raise WorkGroupProtocolError(f"invalid {f.name}")
            try:
                value = wrapper(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise WorkGroupProtocolError(f"invalid {f.name}") from exc
        kwargs[f.name] = value
    try:
        constructor = cast("type[Any]", cls)
        return cast("WorkGroupMessage", constructor(**kwargs))
    except (TypeError, ValueError) as exc:
        raise WorkGroupProtocolError("invalid message") from exc


class WorkGroupState(StrEnum):
    DEFINED = "DEFINED"
    ADMITTING = "ADMITTING"
    DISPATCHING = "DISPATCHING"
    RUNNING = "RUNNING"
    GATHERING = "GATHERING"
    SUCCEEDED = "SUCCEEDED"
    REFUSED = "REFUSED"
    FAILING = "FAILING"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class WorkGroupLifecycle:
    definition: WorkGroupDefinition
    state: WorkGroupState = WorkGroupState.DEFINED
    prepared: tuple[ReplicaId, ...] = ()
    dispatched: tuple[WorkUnitId, ...] = ()
    completed: tuple[WorkUnitId, ...] = ()
    cancelled: tuple[ReplicaId, ...] = ()
    progress: tuple[tuple[WorkUnitId, int, int], ...] = ()
    released: tuple[ReplicaId, ...] = ()
    failures: tuple[tuple[WorkUnitId, str], ...] = ()
    refusals: tuple[tuple[ReplicaId, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.definition) is not WorkGroupDefinition
            or type(self.state) is not WorkGroupState
        ):
            raise WorkGroupProtocolError("invalid lifecycle")
        for name in (
            "prepared",
            "dispatched",
            "completed",
            "cancelled",
            "progress",
            "released",
            "failures",
            "refusals",
        ):
            if type(getattr(self, name)) is not tuple:
                raise WorkGroupProtocolError("lifecycle facts must be tuples")
        replica_ids = {member.replica for member in self.definition.members}
        unit_ids = {unit.unit for unit in self.definition.units}
        for name in ("prepared", "cancelled", "released"):
            values = getattr(self, name)
            if (
                any(type(value) is not ReplicaId for value in values)
                or len(values) != len(set(values))
                or tuple(sorted(values, key=lambda value: value.value)) != values
                or not set(values) <= replica_ids
            ):
                raise WorkGroupProtocolError(f"invalid canonical {name} facts")
        for name in ("dispatched", "completed"):
            values = getattr(self, name)
            if (
                any(type(value) is not WorkUnitId for value in values)
                or len(values) != len(set(values))
                or tuple(sorted(values, key=lambda value: value.value)) != values
                or not set(values) <= unit_ids
            ):
                raise WorkGroupProtocolError(f"invalid canonical {name} facts")
        progress_units: list[WorkUnitId] = []
        for item in self.progress:
            if type(item) is not tuple or len(item) != 3:
                raise WorkGroupProtocolError("invalid progress facts")
            unit, completed, total = item
            if (
                type(unit) is not WorkUnitId
                or unit not in unit_ids
                or type(completed) is not int
                or type(total) is not int
                or total < 1
                or not 0 <= completed <= total
            ):
                raise WorkGroupProtocolError("invalid progress facts")
            progress_units.append(unit)
        if len(progress_units) != len(set(progress_units)) or tuple(
            sorted(progress_units, key=lambda value: value.value)
        ) != tuple(progress_units):
            raise WorkGroupProtocolError("progress facts must be uniquely sorted")
        failure_units: list[WorkUnitId] = []
        for item in self.failures:
            if type(item) is not tuple or len(item) != 2:
                raise WorkGroupProtocolError("invalid failure facts")
            unit, reason = item
            if type(unit) is not WorkUnitId or unit not in unit_ids:
                raise WorkGroupProtocolError("invalid failure facts")
            _reason(reason)
            failure_units.append(unit)
        if len(failure_units) != len(set(failure_units)) or tuple(
            sorted(failure_units, key=lambda value: value.value)
        ) != tuple(failure_units):
            raise WorkGroupProtocolError("failure facts must be uniquely sorted")
        refusal_replicas: list[ReplicaId] = []
        for item in self.refusals:
            if type(item) is not tuple or len(item) != 2:
                raise WorkGroupProtocolError("invalid refusal facts")
            replica, reason = item
            if type(replica) is not ReplicaId or replica not in replica_ids:
                raise WorkGroupProtocolError("invalid refusal facts")
            _reason(reason)
            refusal_replicas.append(replica)
        if len(refusal_replicas) != len(set(refusal_replicas)) or tuple(
            sorted(refusal_replicas, key=lambda value: value.value)
        ) != tuple(refusal_replicas):
            raise WorkGroupProtocolError("refusal facts must be uniquely sorted")
        if not set(self.completed) <= set(self.dispatched):
            raise WorkGroupProtocolError("completed units must have been dispatched")
        if not {unit for unit, _, _ in self.progress} <= set(self.dispatched):
            raise WorkGroupProtocolError("progress units must have been dispatched")
        all_replicas = tuple(member.replica for member in self.definition.members)
        all_units = tuple(sorted(unit_ids, key=lambda value: value.value))
        if self.dispatched not in ((), all_units):
            raise WorkGroupProtocolError("dispatched facts must be empty or complete")
        if self.dispatched and self.prepared != all_replicas:
            raise WorkGroupProtocolError("dispatched facts require every member prepared")
        if self.state is WorkGroupState.DEFINED and any(
            (
                self.prepared,
                self.dispatched,
                self.completed,
                self.cancelled,
                self.progress,
                self.released,
                self.failures,
                self.refusals,
            )
        ):
            raise WorkGroupProtocolError("defined lifecycle cannot contain runtime facts")
        if self.state is WorkGroupState.ADMITTING and any(
            (
                self.dispatched,
                self.completed,
                self.cancelled,
                self.progress,
                self.released,
                self.failures,
                self.refusals,
            )
        ):
            raise WorkGroupProtocolError("admitting lifecycle contains later-phase facts")
        if self.state is WorkGroupState.ADMITTING and self.prepared == all_replicas:
            raise WorkGroupProtocolError("admitting lifecycle cannot be fully prepared")
        if self.state is WorkGroupState.DISPATCHING and self.dispatched:
            raise WorkGroupProtocolError("dispatching lifecycle cannot contain dispatched facts")
        if (
            self.state
            in (
                WorkGroupState.DISPATCHING,
                WorkGroupState.RUNNING,
                WorkGroupState.GATHERING,
                WorkGroupState.SUCCEEDED,
                WorkGroupState.FAILING,
                WorkGroupState.FAILED,
            )
            and self.prepared != all_replicas
        ):
            raise WorkGroupProtocolError("post-admission lifecycle requires every member prepared")
        if (
            self.state
            in (
                WorkGroupState.RUNNING,
                WorkGroupState.GATHERING,
                WorkGroupState.SUCCEEDED,
                WorkGroupState.FAILING,
                WorkGroupState.FAILED,
            )
            and self.dispatched != all_units
        ):
            raise WorkGroupProtocolError("post-dispatch lifecycle requires every unit dispatched")
        if self.state in (WorkGroupState.GATHERING, WorkGroupState.SUCCEEDED) and (
            self.completed != all_units
        ):
            raise WorkGroupProtocolError("gather lifecycle requires every unit completed")
        if self.state is WorkGroupState.RUNNING and self.completed == all_units:
            raise WorkGroupProtocolError("running lifecycle cannot be fully completed")
        if self.state is WorkGroupState.REFUSED and any(
            (
                self.dispatched,
                self.completed,
                self.cancelled,
                self.progress,
                self.released,
                self.failures,
            )
        ):
            raise WorkGroupProtocolError("refused lifecycle contains forward-progress facts")
        if self.failures and self.state not in (WorkGroupState.FAILING, WorkGroupState.FAILED):
            raise WorkGroupProtocolError("failure facts are illegal in this state")
        if self.refusals and self.state is not WorkGroupState.REFUSED:
            raise WorkGroupProtocolError("refusal facts are illegal in this state")
        if self.state is WorkGroupState.CANCELLED and self.cancelled != all_replicas:
            raise WorkGroupProtocolError("cancelled lifecycle requires every member cancelled")
        if self.cancelled and self.state not in (
            WorkGroupState.CANCELLING,
            WorkGroupState.CANCELLED,
            WorkGroupState.FAILING,
            WorkGroupState.FAILED,
        ):
            raise WorkGroupProtocolError("cancelled facts are illegal in this state")
        if self.released and self.state not in (
            WorkGroupState.SUCCEEDED,
            WorkGroupState.CANCELLED,
            WorkGroupState.FAILING,
            WorkGroupState.FAILED,
        ):
            raise WorkGroupProtocolError("released facts are illegal in this state")
        if (
            self.state is WorkGroupState.FAILING
            and self.released
            and self.cancelled != all_replicas
        ):
            raise WorkGroupProtocolError("failure releases require complete cancellation")
        if self.state is WorkGroupState.CANCELLING and self.cancelled == all_replicas:
            raise WorkGroupProtocolError("cancelling lifecycle cannot be fully cancelled")
        if self.state is WorkGroupState.FAILED and (
            self.cancelled != all_replicas or self.released != all_replicas
        ):
            raise WorkGroupProtocolError("failed lifecycle requires complete cleanup")


@dataclass(frozen=True)
class BeginAdmission:
    pass


@dataclass(frozen=True)
class DispatchWork:
    pass


@dataclass(frozen=True)
class GatherSucceeded:
    pass


@dataclass(frozen=True)
class GatherFailed:
    reason: str

    def __post_init__(self) -> None:
        _reason(self.reason)


@dataclass(frozen=True)
class RequestCancellation:
    reason: str

    def __post_init__(self) -> None:
        _reason(self.reason)


@dataclass(frozen=True)
class SettleFailure:
    pass


WorkGroupEvent: TypeAlias = (
    BeginAdmission
    | DispatchWork
    | GatherSucceeded
    | GatherFailed
    | RequestCancellation
    | SettleFailure
    | WorkGroupPrepared
    | WorkGroupRefused
    | WorkUnitProgress
    | WorkUnitResult
    | WorkUnitFailed
    | WorkGroupCancelled
    | WorkGroupReleased
)
_WORKGROUP_EVENT_TYPES = (
    BeginAdmission,
    DispatchWork,
    GatherSucceeded,
    GatherFailed,
    RequestCancellation,
    SettleFailure,
    WorkGroupPrepared,
    WorkGroupRefused,
    WorkUnitProgress,
    WorkUnitResult,
    WorkUnitFailed,
    WorkGroupCancelled,
    WorkGroupReleased,
)


def _child(
    binding: ReplicaBinding, cls: type[_Child], d: WorkGroupDefinition, **extra: object
) -> WorkGroupMessage:
    constructor = cast("type[Any]", cls)
    return cast(
        "WorkGroupMessage",
        constructor(
            worker=binding.worker,
            replica=binding.replica,
            group=d.group,
            attempt=d.attempt,
            device=binding.device,
            **extra,
        ),
    )


def _binding(d: WorkGroupDefinition, replica: ReplicaId) -> ReplicaBinding:
    binding = next((m for m in d.members if m.replica == replica), None)
    if binding is None:
        raise WorkGroupProtocolError("unknown replica")
    return binding


def _fence(lifecycle: WorkGroupLifecycle, event: _Child) -> ReplicaBinding:
    d = lifecycle.definition
    b = _binding(d, event.replica)
    if (event.group, event.attempt, event.worker, event.device) != (
        d.group,
        d.attempt,
        b.worker,
        b.device,
    ):
        raise WorkGroupProtocolError("stale, future, or foreign correlation")
    if isinstance(event, _Unit):
        u = next((u for u in d.units if u.unit == event.unit), None)
        if u is None or (u.replica, u.slot) != (event.replica, event.slot):
            raise WorkGroupProtocolError("foreign unit correlation")
    return b


def reduce_workgroup(
    lifecycle: WorkGroupLifecycle, event: WorkGroupEvent
) -> tuple[WorkGroupLifecycle, tuple[WorkGroupMessage, ...]]:
    if type(lifecycle) is not WorkGroupLifecycle or type(event) not in _WORKGROUP_EVENT_TYPES:
        raise WorkGroupProtocolError("reducer requires exact lifecycle and event types")
    d = lifecycle.definition
    if isinstance(event, _Child):
        _fence(lifecycle, event)
    if type(event) is BeginAdmission:
        if lifecycle.state is not WorkGroupState.DEFINED:
            raise WorkGroupProtocolError("admission already begun")
        return replace(lifecycle, state=WorkGroupState.ADMITTING), tuple(
            _child(m, BeginWorkGroup, d) for m in d.members
        )
    if isinstance(event, WorkGroupPrepared):
        if (
            lifecycle.state
            not in (
                WorkGroupState.ADMITTING,
                WorkGroupState.REFUSED,
                WorkGroupState.CANCELLING,
            )
            or event.replica in lifecycle.prepared
        ):
            raise WorkGroupProtocolError("illegal or duplicate preparation")
        prepared = tuple(sorted((*lifecycle.prepared, event.replica), key=lambda x: x.value))
        if lifecycle.state is not WorkGroupState.ADMITTING:
            return replace(lifecycle, prepared=prepared), ()
        if len(prepared) == len(d.members):
            return replace(lifecycle, state=WorkGroupState.DISPATCHING, prepared=prepared), tuple(
                _child(m, CommitWorkGroup, d) for m in d.members
            )
        return replace(lifecycle, prepared=prepared), ()
    if isinstance(event, WorkGroupRefused):
        if lifecycle.state not in (WorkGroupState.ADMITTING, WorkGroupState.REFUSED) or any(
            replica == event.replica for replica, _ in lifecycle.refusals
        ):
            raise WorkGroupProtocolError("illegal refusal")
        refusals = tuple(
            sorted(
                (*lifecycle.refusals, (event.replica, event.reason)),
                key=lambda item: item[0].value,
            )
        )
        commands = (
            tuple(_child(m, AbortWorkGroup, d) for m in d.members)
            if lifecycle.state is WorkGroupState.ADMITTING
            else ()
        )
        return replace(lifecycle, state=WorkGroupState.REFUSED, refusals=refusals), commands
    if type(event) is DispatchWork:
        if lifecycle.state is not WorkGroupState.DISPATCHING:
            raise WorkGroupProtocolError("dispatch is illegal")
        dispatch_commands: list[WorkGroupMessage] = []
        for u in d.units:
            b = _binding(d, u.replica)
            dispatch_commands.append(_child(b, RunWorkUnit, d, unit=u.unit, slot=u.slot))
        return replace(
            lifecycle, state=WorkGroupState.RUNNING, dispatched=tuple(u.unit for u in d.units)
        ), tuple(dispatch_commands)
    if isinstance(event, WorkUnitProgress):
        if (
            lifecycle.state
            not in (
                WorkGroupState.RUNNING,
                WorkGroupState.CANCELLING,
            )
            or event.unit in lifecycle.completed
        ):
            raise WorkGroupProtocolError("progress is not active")
        p = {u: (c, t) for u, c, t in lifecycle.progress}
        old = p.get(event.unit)
        if old is not None and (event.total != old[1] or event.completed <= old[0]):
            raise WorkGroupProtocolError("progress regressed, duplicated, or changed total")
        p[event.unit] = (event.completed, event.total)
        return replace(
            lifecycle, progress=tuple((u, *p[u]) for u in sorted(p, key=lambda x: x.value))
        ), ()
    if isinstance(event, WorkUnitResult):
        if (
            lifecycle.state
            not in (
                WorkGroupState.RUNNING,
                WorkGroupState.CANCELLING,
            )
            or event.unit in lifecycle.completed
        ):
            raise WorkGroupProtocolError("illegal or duplicate result")
        done = tuple(sorted((*lifecycle.completed, event.unit), key=lambda x: x.value))
        state = (
            WorkGroupState.GATHERING
            if lifecycle.state is WorkGroupState.RUNNING and len(done) == len(d.units)
            else lifecycle.state
        )
        return replace(lifecycle, state=state, completed=done), ()
    if isinstance(event, WorkUnitFailed):
        if lifecycle.state not in (WorkGroupState.RUNNING, WorkGroupState.FAILING) or any(
            unit == event.unit for unit, _ in lifecycle.failures
        ):
            raise WorkGroupProtocolError("illegal failure")
        failures = tuple(
            sorted(
                (*lifecycle.failures, (event.unit, event.reason)),
                key=lambda item: item[0].value,
            )
        )
        if lifecycle.state is WorkGroupState.FAILING:
            return replace(lifecycle, failures=failures), ()
        return replace(lifecycle, state=WorkGroupState.FAILING, failures=failures), tuple(
            _child(m, CancelWorkGroup, d, reason=event.reason)
            for m in d.members
            if m.replica in lifecycle.prepared
        )
    if isinstance(event, GatherFailed):
        if lifecycle.state is not WorkGroupState.GATHERING:
            raise WorkGroupProtocolError("illegal failure")
        return replace(lifecycle, state=WorkGroupState.FAILING), tuple(
            _child(m, CancelWorkGroup, d, reason=event.reason)
            for m in d.members
            if m.replica in lifecycle.prepared
        )
    if type(event) is GatherSucceeded:
        if lifecycle.state is not WorkGroupState.GATHERING:
            raise WorkGroupProtocolError("illegal gather success")
        return replace(lifecycle, state=WorkGroupState.SUCCEEDED), tuple(
            _child(m, ReleaseWorkGroup, d) for m in d.members
        )
    if isinstance(event, RequestCancellation):
        if lifecycle.state not in (
            WorkGroupState.ADMITTING,
            WorkGroupState.DISPATCHING,
            WorkGroupState.RUNNING,
            WorkGroupState.GATHERING,
        ):
            raise WorkGroupProtocolError("cancellation is illegal")
        return replace(lifecycle, state=WorkGroupState.CANCELLING), tuple(
            _child(m, CancelWorkGroup, d, reason=event.reason) for m in d.members
        )
    if isinstance(event, WorkGroupCancelled):
        if (
            lifecycle.state not in (WorkGroupState.CANCELLING, WorkGroupState.FAILING)
            or event.replica in lifecycle.cancelled
        ):
            raise WorkGroupProtocolError("illegal or duplicate cancellation")
        cancelled = tuple(sorted((*lifecycle.cancelled, event.replica), key=lambda x: x.value))
        applicable = {member.replica for member in d.members}
        commands: tuple[WorkGroupMessage, ...] = ()
        state = lifecycle.state
        if set(cancelled) == applicable:
            commands = tuple(
                _child(m, ReleaseWorkGroup, d) for m in d.members if m.replica in applicable
            )
            if state is WorkGroupState.CANCELLING:
                state = WorkGroupState.CANCELLED
        return replace(lifecycle, state=state, cancelled=cancelled), commands
    if isinstance(event, WorkGroupReleased):
        if (
            lifecycle.state
            not in (WorkGroupState.SUCCEEDED, WorkGroupState.CANCELLED, WorkGroupState.FAILING)
            or event.replica in lifecycle.released
            or (
                lifecycle.state is WorkGroupState.FAILING
                and len(lifecycle.cancelled) != len(d.members)
            )
        ):
            raise WorkGroupProtocolError("illegal or duplicate release")
        return replace(
            lifecycle,
            released=tuple(sorted((*lifecycle.released, event.replica), key=lambda x: x.value)),
        ), ()
    if type(event) is SettleFailure:
        if (
            lifecycle.state is not WorkGroupState.FAILING
            or set(lifecycle.cancelled) != set(lifecycle.prepared)
            or set(lifecycle.released) != set(lifecycle.prepared)
        ):
            raise WorkGroupProtocolError("failure cleanup is incomplete")
        return replace(lifecycle, state=WorkGroupState.FAILED), ()
    raise WorkGroupProtocolError("unsupported event")
