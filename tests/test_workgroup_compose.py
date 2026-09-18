from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from typing import cast, overload

import pytest
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
    WorkGroupMessage,
    WorkUnitId,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import (
    InProcessWorker,
    IsolatedWorker,
    RemoteWorker,
    WorkGroupBoundaryWorker,
    WorkGroupLaneCandidate,
    WorkGroupWorkerLane,
    build_workgroup_configuration,
    compose_workgroup_configuration,
)
from dinkster_workers.boundary import ValueCodec
from dinkster_workers.isolated import GroupIsolatedWorker, GroupMemberWorker
from dinkster_workers.session import BoundarySession
from dinkster_workers.workgroup import ReplicaEndpoint
from dinkster_workers.workgroup_session import (
    WORKGROUP_HELLO_FIELD,
    WorkGroupSession,
    WorkGroupTransportClosed,
)
from test_isolated import TESTS_DIR, core_registry, write_workgroup_manifest


class Boundary:
    def __init__(
        self,
        token: str,
        events: list[str] | None = None,
        *,
        capabilities: frozenset[str] = frozenset({WORKGROUP_CAPABILITY}),
        bind_error: BaseException | None = None,
        unbind_error: BaseException | None = None,
    ) -> None:
        self.instance_token = token
        self.workgroup_capabilities = capabilities
        self.events = events if events is not None else []
        self.bind_error = bind_error
        self.unbind_error = unbind_error

        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        self.transport = WorkGroupSession(send)
        self.transport.negotiate({WORKGROUP_HELLO_FIELD: list(capabilities)})

    def bind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> ReplicaEndpoint:
        self.events.append(f"bind:{replica.value}")
        if self.bind_error is not None:
            raise self.bind_error
        return self.transport.bind(
            definition,
            replica,
            worker_instance=self.instance_token,
        )

    def unbind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> None:
        self.events.append(f"unbind:{replica.value}")
        self.transport.unbind(definition, replica)
        if self.unbind_error is not None:
            raise self.unbind_error


def lane(name: str, boundary: WorkGroupBoundaryWorker | None = None) -> WorkGroupWorkerLane:
    return WorkGroupWorkerLane(
        replica=ReplicaId(f"replica-{name}"),
        worker=WorkerInstanceId(f"worker-{name}"),
        device=DeviceResourceId(f"device-{name}"),
        recipe=ReplicaRecipeId("sha256:" + hashlib.sha256(name.encode()).hexdigest()),
        unit=WorkUnitId(f"unit-{name}"),
        slot=SemanticSlot.SINGLE,
        boundary=boundary or Boundary(f"worker-{name}"),
    )


def compose(
    lanes: Sequence[WorkGroupWorkerLane],
) -> tuple[WorkGroupDefinition, tuple[ReplicaEndpoint, ...]]:
    return compose_workgroup_configuration(WorkGroupId("group-a"), WorkGroupAttempt(3), lanes)


def test_worker_lane_is_exact_frozen_and_collection_is_bounded() -> None:
    value = lane("a")
    with pytest.raises(FrozenInstanceError):
        value.worker = WorkerInstanceId("other")  # type: ignore[misc]
    for field, wrong in (
        ("replica", "replica-a"),
        ("worker", "worker-a"),
        ("device", "device-a"),
        ("recipe", "sha256:" + "a" * 64),
        ("unit", "unit-a"),
        ("slot", "single"),
    ):
        with pytest.raises(ValueError, match=rf"lane {field} has the wrong type"):
            replace(value, **{field: wrong})
    with pytest.raises(ValueError, match="WorkGroupBoundaryWorker"):
        replace(value, boundary=object())

    consumed = False

    def generate() -> Iterator[WorkGroupWorkerLane]:
        nonlocal consumed
        consumed = True
        yield value

    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        compose(generate())  # type: ignore[arg-type]
    assert not consumed
    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        compose(())
    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        compose(tuple(lane(str(index)) for index in range(MAX_WORKGROUP_REPLICAS + 1)))

    class Oversized(Sequence[WorkGroupWorkerLane]):
        def __len__(self) -> int:
            return MAX_WORKGROUP_REPLICAS + 1

        @overload
        def __getitem__(self, index: int) -> WorkGroupWorkerLane: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[WorkGroupWorkerLane]: ...

        def __getitem__(
            self, index: int | slice
        ) -> WorkGroupWorkerLane | Sequence[WorkGroupWorkerLane]:
            raise AssertionError(f"oversized sequence was consumed at {index}")

    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        compose(Oversized())


def test_composition_permutations_equal_the_legacy_builder() -> None:
    lanes = (lane("c"), lane("a"), lane("b"))

    async def send(_message: WorkGroupMessage) -> None:
        return None

    async def receive() -> WorkGroupMessage:
        raise RuntimeError("no reply")

    candidates = tuple(
        WorkGroupLaneCandidate(
            replica=value.replica,
            worker=value.worker,
            device=value.device,
            recipe=value.recipe,
            unit=value.unit,
            slot=value.slot,
            capabilities=frozenset({WORKGROUP_CAPABILITY}),
            send=send,
            receive=receive,
        )
        for value in lanes
    )
    legacy, legacy_endpoints = build_workgroup_configuration(
        WorkGroupId("group-a"), WorkGroupAttempt(3), candidates
    )

    for order in permutations(("c", "a", "b")):
        definition, endpoints = compose(tuple(lane(name) for name in order))
        assert definition == legacy
        assert tuple(endpoint.binding for endpoint in endpoints) == tuple(
            endpoint.binding for endpoint in legacy_endpoints
        )


def test_every_identity_and_capability_is_validated_before_any_bind() -> None:
    events: list[str] = []
    good = lane("a", Boundary("worker-a", events))
    bad_token = lane("b", Boundary("other-worker", events))
    with pytest.raises(ValueError, match="authenticated session"):
        compose((good, bad_token))
    assert events == []

    bad_capability = lane("b", Boundary("worker-b", events, capabilities=frozenset()))
    with pytest.raises(ValueError, match="do not negotiate"):
        compose((good, bad_capability))
    assert events == []

    duplicate = replace(lane("b", Boundary("worker-a", events)), worker=good.worker)
    with pytest.raises(ValueError, match="duplicate worker worker-a"):
        compose((duplicate, good))
    assert events == []


def test_partial_bind_rolls_back_in_reverse_and_preserves_primary_error() -> None:
    events: list[str] = []
    first = lane("a", Boundary("worker-a", events, unbind_error=RuntimeError("cleanup")))
    second = lane("b", Boundary("worker-b", events))
    third = lane("c", Boundary("worker-c", events, bind_error=LookupError("primary")))

    with pytest.raises(LookupError, match="primary"):
        compose((third, first, second))
    assert events == [
        "bind:replica-a",
        "bind:replica-b",
        "bind:replica-c",
        "unbind:replica-b",
        "unbind:replica-a",
    ]


def test_partial_bind_rolls_back_pristine_registration_after_transport_failure() -> None:
    first_boundary = Boundary("worker-a")

    class FailedTransportBoundary(Boundary):
        def bind_workgroup_endpoint(
            self, definition: WorkGroupDefinition, replica: ReplicaId
        ) -> ReplicaEndpoint:
            del definition, replica
            first_boundary.transport.fail(ConnectionError("lost"))
            raise WorkGroupTransportClosed("second transport closed")

    with pytest.raises(WorkGroupTransportClosed, match="second transport closed"):
        compose((lane("a", first_boundary), lane("b", FailedTransportBoundary("worker-b"))))
    assert first_boundary.transport._channels == {}  # noqa: SLF001


def test_foreign_endpoint_return_is_refused_and_rolled_back() -> None:
    class ForeignEndpointBoundary(Boundary):
        def bind_workgroup_endpoint(
            self, definition: WorkGroupDefinition, replica: ReplicaId
        ) -> ReplicaEndpoint:
            super().bind_workgroup_endpoint(definition, replica)
            return object()  # type: ignore[return-value]

    boundary = ForeignEndpointBoundary("worker-a")
    with pytest.raises(ValueError, match="foreign binding"):
        compose((lane("a", boundary),))
    assert boundary.transport._channels == {}  # noqa: SLF001


def negotiated_session(worker: str) -> BoundarySession:
    values = TypeRegistry()
    register_core_types(values)
    session = BoundarySession(
        values,
        role="test",
        pack=worker,
        codec=ValueCodec(values),
    )
    session._schemas = {}  # noqa: SLF001
    session._alive = True  # noqa: SLF001
    session._instance_token = worker  # noqa: SLF001
    session._workgroup.negotiate(  # noqa: SLF001
        {WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]}
    )
    return session


def test_real_worker_facades_compose_while_in_process_stays_outside_protocol() -> None:
    isolated = object.__new__(IsolatedWorker)
    isolated._session = negotiated_session("worker-a")  # noqa: SLF001
    grouped = GroupMemberWorker(
        cast("GroupIsolatedWorker", SimpleNamespace()),
        negotiated_session("worker-b"),
    )
    remote = object.__new__(RemoteWorker)
    remote._session = negotiated_session("worker-c")  # noqa: SLF001

    assert isinstance(isolated, WorkGroupBoundaryWorker)
    assert isinstance(grouped, WorkGroupBoundaryWorker)
    assert isinstance(remote, WorkGroupBoundaryWorker)
    definition, endpoints = compose((lane("c", remote), lane("a", isolated), lane("b", grouped)))
    assert tuple(endpoint.binding for endpoint in endpoints) == definition.members

    values = TypeRegistry()
    register_core_types(values)
    assert not isinstance(InProcessWorker({}, values), WorkGroupBoundaryWorker)


def test_launched_isolated_and_group_member_compose_real_endpoints(tmp_path: Path) -> None:
    async def scenario() -> None:
        isolated_dir = tmp_path / "isolated"
        grouped_dir = tmp_path / "grouped"
        isolated_dir.mkdir()
        grouped_dir.mkdir()
        isolated = IsolatedWorker(
            write_workgroup_manifest(isolated_dir),
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        group = GroupIsolatedWorker(
            "composition-test",
            (write_workgroup_manifest(grouped_dir),),
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await isolated.start()
        await group.start()
        try:
            member = group.members["isopack"]
            assert isolated.instance_token is not None
            assert member.instance_token is not None
            lanes = (
                replace(
                    lane("a", isolated),
                    worker=WorkerInstanceId(isolated.instance_token),
                ),
                replace(
                    lane("b", member),
                    worker=WorkerInstanceId(member.instance_token),
                ),
            )
            definition, endpoints = compose(lanes)
            assert tuple(endpoint.binding for endpoint in endpoints) == definition.members
            isolated.unbind_workgroup_endpoint(definition, lanes[0].replica)
            member.unbind_workgroup_endpoint(definition, lanes[1].replica)
        finally:
            await group.close()
            await isolated.close()

    asyncio.run(scenario())
