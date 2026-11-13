from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypedDict

import pytest
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    CancelWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReplicaBinding,
    ReplicaId,
    ReplicaReady,
    ReplicaRecipeId,
    RunWorkUnit,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupMessage,
    WorkUnitDefinition,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import host as host_module
from dinkster_workers.boundary import PROTOCOL_VERSION, ValueCodec, write_frame
from dinkster_workers.in_process import InProcessWorker
from dinkster_workers.session import BoundarySession
from dinkster_workers.workgroup_session import (
    WORKGROUP_FRAME_TYPE,
    WORKGROUP_HELLO_FIELD,
    WorkGroupTransportClosed,
)

from tests.async_scenarios import run_scenario


def registry() -> TypeRegistry:
    value = TypeRegistry()
    register_core_types(value)
    return value


def definition(worker: str, *, attempt: int = 1, group: str = "group-a") -> WorkGroupDefinition:
    replica = ReplicaId("replica-a")
    return WorkGroupDefinition(
        WorkGroupId(group),
        WorkGroupAttempt(attempt),
        (
            ReplicaBinding(
                replica,
                WorkerInstanceId(worker),
                DeviceResourceId("device-a"),
                ReplicaRecipeId("sha256:" + "a" * 64),
            ),
        ),
        (WorkUnitDefinition(WorkUnitId("unit-a"), replica, SemanticSlot.SINGLE),),
    )


class ChildFields(TypedDict):
    worker: WorkerInstanceId
    replica: ReplicaId
    group: WorkGroupId
    attempt: WorkGroupAttempt
    device: DeviceResourceId


def child_fields(worker: str, *, attempt: int = 1, group: str = "group-a") -> ChildFields:
    return {
        "worker": WorkerInstanceId(worker),
        "replica": ReplicaId("replica-a"),
        "group": WorkGroupId(group),
        "attempt": WorkGroupAttempt(attempt),
        "device": DeviceResourceId("device-a"),
    }


async def open_host(
    handler: Callable[[WorkGroupMessage], Awaitable[tuple[WorkGroupMessage, ...]]] | None,
) -> tuple[BoundarySession, dict[str, object], asyncio.Server, asyncio.Task[None]]:
    values = registry()
    worker = InProcessWorker({}, values)
    accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        accepted.set_result(task)
        await host_module.serve_connection(
            reader,
            writer,
            pack_name="test",
            worker=worker,
            schemas={},
            planner=None,
            consumers={},
            codec=ValueCodec(values),
            workgroup_handler=handler,
        )

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(address[0], address[1])
    host_task = await accepted
    session = BoundarySession(
        values,
        role="test worker",
        pack="test",
        codec=ValueCodec(values),
    )
    hello = await session.begin(reader, writer, timeout=2)
    return session, hello, server, host_task


async def close_host(
    session: BoundarySession, server: asyncio.Server, host_task: asyncio.Task[None]
) -> None:
    if session.alive:
        await session.send({"type": "shutdown"}, [])
    await session.close()
    await host_task
    server.close()
    await server.wait_closed()


def test_handler_capability_is_authenticated_present_or_absent() -> None:
    async def scenario() -> None:
        async def handler(_message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
            return ()

        present, present_hello, present_server, present_task = await open_host(handler)
        absent, absent_hello, absent_server, absent_task = await open_host(None)
        try:
            assert present.workgroup_capabilities == frozenset({WORKGROUP_CAPABILITY})
            assert absent.workgroup_capabilities == frozenset()
            assert WORKGROUP_HELLO_FIELD == "workgroupCapabilities"
            assert present_hello[WORKGROUP_HELLO_FIELD] == [WORKGROUP_CAPABILITY]
            assert {k: v for k, v in present_hello.items() if k != WORKGROUP_HELLO_FIELD} == (
                absent_hello
            )
        finally:
            await close_host(present, present_server, present_task)
            await close_host(absent, absent_server, absent_task)

        values = registry()
        rejected: asyncio.Future[ValueError] = asyncio.get_running_loop().create_future()

        async def reject_reserved_hello(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await host_module.serve_connection(
                    reader,
                    writer,
                    pack_name="test",
                    worker=InProcessWorker({}, values),
                    schemas={},
                    planner=None,
                    consumers={},
                    codec=ValueCodec(values),
                    hello_extra={WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]},
                )
            except ValueError as exc:
                rejected.set_result(exc)
                writer.close()

        reserved_server = await asyncio.start_server(reject_reserved_hello, "127.0.0.1", 0)
        address = reserved_server.sockets[0].getsockname()
        _reader, writer = await asyncio.open_connection(address[0], address[1])
        assert WORKGROUP_HELLO_FIELD in str(await asyncio.wait_for(rejected, 2))
        writer.close()
        await writer.wait_closed()
        reserved_server.close()
        await reserved_server.wait_closed()

    run_scenario(scenario())


def test_parent_endpoint_round_trip_preserves_reply_order() -> None:
    async def scenario() -> None:
        async def handler(message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
            worker = message.worker.value
            if type(message) is PrepareReplica:
                return (ReplicaReady(**child_fields(worker)),)
            if type(message) is RunWorkUnit:
                unit = {
                    **child_fields(worker),
                    "unit": WorkUnitId("unit-a"),
                    "slot": SemanticSlot.SINGLE,
                }
                return (
                    WorkUnitProgress(**unit, completed=1, total=2),
                    WorkUnitProgress(**unit, completed=2, total=2),
                    WorkUnitResult(**unit),
                )
            raise AssertionError(type(message))

        session, _hello, server, host_task = await open_host(handler)
        try:
            assert session.instance_token is not None
            endpoint = session.bind_workgroup_endpoint(
                definition(session.instance_token), ReplicaId("replica-a")
            )
            await endpoint.send(
                PrepareReplica(
                    **child_fields(session.instance_token),
                    recipe=ReplicaRecipeId("sha256:" + "a" * 64),
                )
            )
            assert type(await endpoint.receive()) is ReplicaReady
            await endpoint.send(
                RunWorkUnit(
                    **child_fields(session.instance_token),
                    unit=WorkUnitId("unit-a"),
                    slot=SemanticSlot.SINGLE,
                )
            )
            replies = [await endpoint.receive() for _ in range(3)]
            assert [type(reply) for reply in replies] == [
                WorkUnitProgress,
                WorkUnitProgress,
                WorkUnitResult,
            ]
            assert [reply.completed for reply in replies[:2]] == [1, 2]  # type: ignore[attr-defined]
        finally:
            await close_host(session, server, host_task)

    run_scenario(scenario())


def test_cancel_command_reaches_handler_while_run_is_pending() -> None:
    async def scenario() -> None:
        cancel_seen = asyncio.Event()

        async def handler(message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
            worker = message.worker.value
            if type(message) is RunWorkUnit:
                await cancel_seen.wait()
                return (
                    WorkUnitResult(
                        **child_fields(worker),
                        unit=WorkUnitId("unit-a"),
                        slot=SemanticSlot.SINGLE,
                    ),
                )
            if type(message) is CancelWorkGroup:
                cancel_seen.set()
                return (WorkGroupCancelled(**child_fields(worker)),)
            raise AssertionError(type(message))

        session, _hello, server, host_task = await open_host(handler)
        try:
            assert session.instance_token is not None
            endpoint = session.bind_workgroup_endpoint(
                definition(session.instance_token), ReplicaId("replica-a")
            )
            await endpoint.send(
                RunWorkUnit(
                    **child_fields(session.instance_token),
                    unit=WorkUnitId("unit-a"),
                    slot=SemanticSlot.SINGLE,
                )
            )
            await endpoint.send(
                CancelWorkGroup(**child_fields(session.instance_token), reason="cancel")
            )
            assert type(await asyncio.wait_for(endpoint.receive(), 2)) is WorkGroupCancelled
            assert type(await asyncio.wait_for(endpoint.receive(), 2)) is WorkUnitResult
        finally:
            await close_host(session, server, host_task)

    run_scenario(scenario())


def test_protocol_corruption_closes_boundary_and_fails_pending_receive() -> None:
    async def scenario() -> None:
        accepted: asyncio.Future[asyncio.StreamWriter] = asyncio.get_running_loop().create_future()
        receiving = asyncio.Event()

        async def corrupt(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            accepted.set_result(writer)
            await write_frame(
                writer,
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "pack": "test",
                    "schemas": {},
                    "workerInstance": "worker-a",
                    WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY],
                },
                [],
            )
            await asyncio.wait_for(receiving.wait(), 2)
            await write_frame(
                writer,
                {"type": WORKGROUP_FRAME_TYPE, "message": {"type": "unknown"}},
                [],
            )

        server = await asyncio.start_server(corrupt, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        peer_writer = await accepted
        values = registry()
        session = BoundarySession(values, role="test", pack="test", codec=ValueCodec(values))
        try:
            await session.begin(reader, writer, timeout=2)
            endpoint = session.bind_workgroup_endpoint(
                definition("worker-a"), ReplicaId("replica-a")
            )

            async def receive() -> None:
                receiving.set()
                await endpoint.receive()

            with pytest.raises(WorkGroupTransportClosed):
                await asyncio.wait_for(receive(), 2)
            assert not session.alive
        finally:
            await session.close()
            peer_writer.close()
            await peer_writer.wait_closed()
            server.close()
            await server.wait_closed()

    run_scenario(scenario())
