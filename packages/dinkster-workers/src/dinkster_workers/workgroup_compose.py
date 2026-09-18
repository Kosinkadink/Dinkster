"""Production composition of explicit lanes through authenticated workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from dinkster_protocol import (
    MAX_WORKGROUP_REPLICAS,
    WORKGROUP_CAPABILITY,
    DeviceResourceId,
    ReplicaId,
    ReplicaRecipeId,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupDefinition,
    WorkGroupId,
    WorkUnitId,
    negotiate_workgroup_capabilities,
)

from .workgroup import ReplicaEndpoint
from .workgroup_config import _plan_workgroup_definition  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "WorkGroupBoundaryWorker",
    "WorkGroupWorkerLane",
    "compose_workgroup_configuration",
]


@runtime_checkable
class WorkGroupBoundaryWorker(Protocol):
    @property
    def instance_token(self) -> str | None: ...

    @property
    def workgroup_capabilities(self) -> frozenset[str]: ...

    def bind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> ReplicaEndpoint: ...

    def unbind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkGroupWorkerLane:
    replica: ReplicaId
    worker: WorkerInstanceId
    device: DeviceResourceId
    recipe: ReplicaRecipeId
    unit: WorkUnitId
    slot: SemanticSlot
    boundary: WorkGroupBoundaryWorker

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
                raise ValueError(f"lane {name} has the wrong type")
        if not isinstance(cast("object", self.boundary), WorkGroupBoundaryWorker):
            raise ValueError("lane boundary must implement WorkGroupBoundaryWorker")


def compose_workgroup_configuration(
    group: WorkGroupId,
    attempt: WorkGroupAttempt,
    lanes: Sequence[WorkGroupWorkerLane],
) -> tuple[WorkGroupDefinition, tuple[ReplicaEndpoint, ...]]:
    """Validate every lane, plan canonically, then bind real endpoints."""

    if type(group) is not WorkGroupId or type(attempt) is not WorkGroupAttempt:
        raise ValueError("composition requires exact group and attempt identities")
    if not isinstance(lanes, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("lanes must be a bounded nonempty tuple")
    if not 1 <= len(lanes) <= MAX_WORKGROUP_REPLICAS:
        raise ValueError("lanes must be a bounded nonempty tuple")
    try:
        lane_tuple = tuple(lanes)
    except TypeError as exc:
        raise ValueError("lanes must be a bounded nonempty tuple") from exc
    if len(lane_tuple) != len(lanes):
        raise ValueError("lanes must be a bounded nonempty tuple")
    for index, lane in enumerate(lane_tuple):
        if type(lane) is not WorkGroupWorkerLane:
            raise ValueError(f"lane {index} has the wrong type")

    for index, lane in enumerate(lane_tuple):
        token = lane.boundary.instance_token
        if type(token) is not str or token != lane.worker.value:
            raise ValueError(f"lane {index} worker differs from the authenticated session")
        capabilities = lane.boundary.workgroup_capabilities
        if type(capabilities) is not frozenset or any(
            type(value) is not str for value in capabilities
        ):
            raise ValueError(f"lane {index} capabilities must be a frozenset of strings")
        if WORKGROUP_CAPABILITY not in negotiate_workgroup_capabilities(capabilities):
            raise ValueError(f"lane {index} capabilities do not negotiate {WORKGROUP_CAPABILITY}")

    definition, ordered = _plan_workgroup_definition(group, attempt, lane_tuple)
    by_replica = {lane.replica: lane for lane in ordered}
    bound: list[tuple[WorkGroupBoundaryWorker, ReplicaId]] = []
    endpoints: list[ReplicaEndpoint] = []
    try:
        for member in definition.members:
            boundary = by_replica[member.replica].boundary
            endpoint = boundary.bind_workgroup_endpoint(definition, member.replica)
            bound.append((boundary, member.replica))
            if (
                type(endpoint) is not ReplicaEndpoint
                or endpoint.definition != definition
                or endpoint.binding != member
            ):
                raise ValueError("boundary returned an endpoint with a foreign binding")
            endpoints.append(endpoint)
    except BaseException:
        for boundary, replica in reversed(bound):
            try:
                boundary.unbind_workgroup_endpoint(definition, replica)
            except BaseException:
                pass
        raise
    return definition, tuple(endpoints)
