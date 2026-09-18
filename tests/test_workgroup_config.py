from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from typing import Any

import pytest
from dinkster_protocol import (
    MAX_WORKGROUP_REPLICAS,
    WORKGROUP_CAPABILITY,
    DeviceResourceId,
    PrepareReplica,
    ReplicaId,
    ReplicaReady,
    ReplicaRecipeId,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupMessage,
    WorkUnitId,
)
from dinkster_workers import ReplicaEndpoint, WorkGroupLaneCandidate, build_workgroup_configuration


async def _send(message: WorkGroupMessage) -> None:
    del message


async def _receive() -> WorkGroupMessage:
    raise RuntimeError("no reply")


def _candidate(name: str, **changes: object) -> WorkGroupLaneCandidate:
    values: dict[str, object] = {
        "replica": ReplicaId(f"replica-{name}"),
        "worker": WorkerInstanceId(f"worker-{name}"),
        "device": DeviceResourceId(f"parent-device-{name}"),
        "recipe": ReplicaRecipeId("sha256:" + name * 64),
        "unit": WorkUnitId(f"unit-{name}"),
        "slot": SemanticSlot.SINGLE,
        "capabilities": frozenset({WORKGROUP_CAPABILITY}),
        "send": _send,
        "receive": _receive,
    }
    values.update(changes)
    return WorkGroupLaneCandidate(**values)  # type: ignore[arg-type]


def _build(
    candidates: tuple[WorkGroupLaneCandidate, ...], attempt: int = 3
) -> tuple[WorkGroupDefinition, tuple[ReplicaEndpoint, ...]]:
    return build_workgroup_configuration(
        WorkGroupId("group-1"), WorkGroupAttempt(attempt), candidates
    )


def test_permutations_build_one_canonical_definition_and_endpoint_order() -> None:
    candidates = (_candidate("c"), _candidate("a"), _candidate("b"))
    observed = []

    for order in permutations(candidates):
        definition, endpoints = _build(order)
        observed.append((definition, tuple(endpoint.binding for endpoint in endpoints)))
        assert tuple(member.replica.value for member in definition.members) == (
            "replica-a",
            "replica-b",
            "replica-c",
        )
        assert tuple(endpoint.binding for endpoint in endpoints) == definition.members

    assert all(result == observed[0] for result in observed)


def test_candidate_is_frozen_and_requires_exact_public_identity_types() -> None:
    candidate = _candidate("a")
    with pytest.raises(FrozenInstanceError):
        candidate.device = DeviceResourceId("other")  # type: ignore[misc]

    wrong_fields = {
        "replica": "replica-a",
        "worker": "worker-a",
        "device": "parent-device-a",
        "recipe": "sha256:" + "a" * 64,
        "unit": "unit-a",
        "slot": "single",
    }
    for field, value in wrong_fields.items():
        with pytest.raises(ValueError, match=rf"candidate {field} has the wrong type"):
            replace(candidate, **{field: value})


def test_capability_and_transport_validation_is_closed_and_stable() -> None:
    candidate = _candidate("a")
    for capabilities in (set(), {WORKGROUP_CAPABILITY}, (WORKGROUP_CAPABILITY,), frozenset({1})):
        with pytest.raises(
            ValueError, match="candidate capabilities must be a frozenset of strings"
        ):
            replace(candidate, capabilities=capabilities)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="do not negotiate dinkster.multidevice-workgroup.v1"):
        replace(candidate, capabilities=frozenset({"other"}))
    assert replace(
        candidate,
        capabilities=frozenset({WORKGROUP_CAPABILITY, "unrelated.capability"}),
    )
    for field in ("send", "receive"):
        with pytest.raises(ValueError, match=rf"candidate {field} must be callable"):
            replace(candidate, **{field: None})


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("replica", ReplicaId("replica-a"), "duplicate replica replica-a"),
        ("worker", WorkerInstanceId("worker-a"), "duplicate worker worker-a"),
        ("device", DeviceResourceId("parent-device-a"), "duplicate device parent-device-a"),
        ("unit", WorkUnitId("unit-a"), "duplicate unit unit-a"),
        ("slot", SemanticSlot.SINGLE, "duplicate replica/slot replica-a/single"),
    ),
)
def test_duplicate_facts_refuse_before_any_endpoint_is_bound(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, reason: str
) -> None:
    a = _candidate("a")
    changes: dict[str, object] = {field: value}
    if field == "slot":
        changes["replica"] = ReplicaId("replica-a")
    b = replace(_candidate("b"), **changes)
    binds: list[object] = []
    original = ReplicaEndpoint.bind

    def record(*args: object, **kwargs: object) -> object:
        binds.append((args, kwargs))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ReplicaEndpoint, "bind", record)
    with pytest.raises(ValueError, match=reason):
        _build((b, a))
    assert binds == []


def test_group_attempt_cardinality_and_candidate_shapes_fail_closed() -> None:
    candidate = _candidate("a")
    with pytest.raises(ValueError, match="exact group and attempt identities"):
        build_workgroup_configuration("group-1", WorkGroupAttempt(1), (candidate,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact group and attempt identities"):
        build_workgroup_configuration(WorkGroupId("group-1"), 1, (candidate,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        _build(())
    with pytest.raises(ValueError, match="candidate 1 has the wrong type"):
        _build((candidate, object()))  # type: ignore[arg-type]


def test_candidate_collection_refuses_unbounded_shapes_and_over_limit_before_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate("a")
    consumed = False

    def generate() -> Iterator[WorkGroupLaneCandidate]:
        nonlocal consumed
        consumed = True
        yield candidate

    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        build_workgroup_configuration(
            WorkGroupId("group-1"),
            WorkGroupAttempt(1),
            generate(),  # type: ignore[arg-type]
        )
    assert not consumed

    oversized = tuple(
        replace(
            candidate,
            replica=ReplicaId(f"replica-{index}"),
            worker=WorkerInstanceId(f"worker-{index}"),
            device=DeviceResourceId(f"parent-device-{index}"),
            unit=WorkUnitId(f"unit-{index}"),
        )
        for index in range(MAX_WORKGROUP_REPLICAS + 1)
    )
    binds: list[object] = []

    def record(*args: object, **kwargs: object) -> object:
        binds.append((args, kwargs))
        raise AssertionError("oversized configuration must not bind an endpoint")

    monkeypatch.setattr(ReplicaEndpoint, "bind", record)
    with pytest.raises(ValueError, match="bounded nonempty tuple"):
        _build(oversized)
    assert binds == []


def test_parent_device_namespace_and_candidate_transports_are_preserved_exactly() -> None:
    sent: list[WorkGroupMessage] = []
    replies: asyncio.Queue[WorkGroupMessage] = asyncio.Queue()

    async def send(message: WorkGroupMessage) -> None:
        sent.append(message)

    async def receive() -> WorkGroupMessage:
        return await replies.get()

    candidate = _candidate(
        "a",
        device=DeviceResourceId("parent-node-7-device-2"),
        send=send,
        receive=receive,
    )
    definition, (endpoint,) = _build((candidate,))
    binding = definition.members[0]
    command = PrepareReplica(
        worker=binding.worker,
        replica=binding.replica,
        group=definition.group,
        attempt=definition.attempt,
        device=binding.device,
        recipe=binding.recipe,
    )
    reply = ReplicaReady(
        worker=binding.worker,
        replica=binding.replica,
        group=definition.group,
        attempt=definition.attempt,
        device=binding.device,
    )

    async def exercise() -> None:
        await endpoint.send(command)
        replies.put_nowait(reply)
        assert await endpoint.receive() == reply

    asyncio.run(exercise())
    assert binding.device == DeviceResourceId("parent-node-7-device-2")
    assert sent == [command]


def test_returned_endpoint_refuses_foreign_attempt_binding_and_transport_messages() -> None:
    definition, (endpoint,) = _build((_candidate("a"),))
    binding = endpoint.binding
    common: dict[str, Any] = {
        "worker": binding.worker,
        "replica": binding.replica,
        "group": definition.group,
        "attempt": definition.attempt,
        "device": binding.device,
    }
    foreign_attempt = PrepareReplica(
        **{**common, "attempt": WorkGroupAttempt(99)}, recipe=binding.recipe
    )
    foreign_binding = PrepareReplica(
        **{**common, "worker": WorkerInstanceId("foreign-worker")}, recipe=binding.recipe
    )

    async def exercise() -> None:
        with pytest.raises(ValueError, match="not bound to this group attempt"):
            await endpoint.send(foreign_attempt)
        with pytest.raises(ValueError, match="not bound to this group attempt"):
            await endpoint.send(foreign_binding)
        with pytest.raises(RuntimeError, match="no reply"):
            await endpoint.receive()

    asyncio.run(exercise())
