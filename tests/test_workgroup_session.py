from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypedDict

import pytest
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    CancelWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReleaseWorkGroup,
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
    WorkGroupReleased,
    WorkUnitDefinition,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import host as host_module
from dinkster_workers.boundary import BoundaryError, ValueCodec, write_frame
from dinkster_workers.in_process import InProcessWorker
from dinkster_workers.session import BoundarySession
from dinkster_workers.workgroup_session import (
    WORKGROUP_FRAME_TYPE,
    WORKGROUP_HELLO_FIELD,
    WorkGroupSession,
    WorkGroupTransportClosed,
    workgroup_frame,
    workgroup_message_from_frame,
)


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


def test_closed_envelope_is_exact_and_blob_free() -> None:
    message = ReplicaReady(**child_fields("worker-a"))
    frame = workgroup_frame(message)
    assert frame == {
        "type": WORKGROUP_FRAME_TYPE,
        "message": {
            "type": "replicaReady",
            "version": 1,
            "capability": WORKGROUP_CAPABILITY,
            "workerInstance": "worker-a",
            "replica": "replica-a",
            "group": "group-a",
            "attempt": 1,
            "device": "device-a",
        },
    }
    assert workgroup_message_from_frame(frame, ()) == message
    for malformed, blobs in (
        ({**frame, "extra": True}, ()),
        ({"type": WORKGROUP_FRAME_TYPE}, ()),
        (frame, (b"blob",)),
        ({"type": WORKGROUP_FRAME_TYPE, "message": {"type": "unknown"}}, ()),
    ):
        with pytest.raises(BoundaryError):
            workgroup_message_from_frame(malformed, blobs)


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_endpoint_owns_direction_and_correlation_refusal() -> None:
    async def scenario() -> None:
        async def foreign_correlation(
            message: WorkGroupMessage,
        ) -> tuple[WorkGroupMessage, ...]:
            return (ReplicaReady(**child_fields(message.worker.value, attempt=2)),)

        session, _hello, server, host_task = await open_host(foreign_correlation)
        try:
            assert session.instance_token is not None
            endpoint = session.bind_workgroup_endpoint(
                definition(session.instance_token), ReplicaId("replica-a")
            )
            command = PrepareReplica(
                **child_fields(session.instance_token),
                recipe=ReplicaRecipeId("sha256:" + "a" * 64),
            )
            await endpoint.send(command)
            with pytest.raises(WorkGroupTransportClosed):
                await endpoint.receive()
            assert not session.alive

            sent: list[dict[str, object]] = []

            async def send(header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
                sent.append(dict(header))

            transport = WorkGroupSession(send)
            transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
            direct = transport.bind(
                definition("worker-a"),
                ReplicaId("replica-a"),
                worker_instance="worker-a",
            )
            command = PrepareReplica(
                **child_fields("worker-a"),
                recipe=ReplicaRecipeId("sha256:" + "a" * 64),
            )
            transport.accept(workgroup_frame(command), ())
            with pytest.raises(ValueError, match="illegal type"):
                await direct.receive()
        finally:
            await close_host(session, server, host_task)

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_multiple_bound_endpoints_route_replies_by_exact_correlation() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint_a = transport.bind(
            definition("worker-a", group="group-a"),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        endpoint_b = transport.bind(
            definition("worker-a", group="group-b"),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        receive_a = asyncio.create_task(endpoint_a.receive())
        receive_b = asyncio.create_task(endpoint_b.receive())
        await asyncio.sleep(0)
        reply_b = ReplicaReady(**child_fields("worker-a", group="group-b"))
        reply_a = ReplicaReady(**child_fields("worker-a", group="group-a"))
        transport.accept(workgroup_frame(reply_b), ())
        transport.accept(workgroup_frame(reply_a), ())
        assert await receive_a == reply_a
        assert await receive_b == reply_b

    asyncio.run(scenario())


def test_released_attempts_retire_only_after_consumption_and_stay_bounded() -> None:
    async def scenario() -> None:
        sent: list[dict[str, object]] = []

        async def send(header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            sent.append(dict(header))

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a", attempt=1),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        released = WorkGroupReleased(**child_fields("worker-a", attempt=1))
        transport.accept(workgroup_frame(released), ())
        assert len(transport._channels) == 1  # noqa: SLF001

        invalid = definition("worker-b", attempt=2)
        with pytest.raises(ValueError, match="authenticated session"):
            transport.bind(invalid, ReplicaId("replica-a"), worker_instance="worker-a")
        assert len(transport._channels) == 1  # noqa: SLF001
        assert await endpoint.receive() == released

        channels = transport._channels  # noqa: SLF001
        with pytest.raises(ValueError, match="authenticated session"):
            transport.bind(invalid, ReplicaId("replica-a"), worker_instance="worker-a")
        assert transport._channels is channels  # noqa: SLF001
        with pytest.raises(BoundaryError, match="retired group attempt"):
            transport.accept(
                workgroup_frame(ReplicaReady(**child_fields("worker-a", attempt=1))), ()
            )

        old_endpoint = endpoint
        for attempt in range(2, 34):
            endpoint = transport.bind(
                definition("worker-a", attempt=attempt),
                ReplicaId("replica-a"),
                worker_instance="worker-a",
            )
            assert len(transport._channels) == 1  # noqa: SLF001
            with pytest.raises(WorkGroupTransportClosed):
                await old_endpoint.receive()
            with pytest.raises(WorkGroupTransportClosed):
                await old_endpoint.send(
                    ReleaseWorkGroup(**child_fields("worker-a", attempt=attempt - 1))
                )
            with pytest.raises(BoundaryError, match="retired group attempt"):
                transport.accept(
                    workgroup_frame(
                        WorkGroupReleased(**child_fields("worker-a", attempt=attempt - 1))
                    ),
                    (),
                )
            released = WorkGroupReleased(**child_fields("worker-a", attempt=attempt))
            transport.accept(workgroup_frame(released), ())
            assert await endpoint.receive() == released
            old_endpoint = endpoint

        assert sent == []

    asyncio.run(scenario())


def test_release_consumption_wakes_every_concurrent_receiver() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a"), ReplicaId("replica-a"), worker_instance="worker-a"
        )
        pending = [asyncio.create_task(endpoint.receive()) for _ in range(2)]
        await asyncio.sleep(0)
        released = WorkGroupReleased(**child_fields("worker-a"))
        transport.accept(workgroup_frame(released), ())

        results = await asyncio.gather(*pending, return_exceptions=True)
        assert results.count(released) == 1
        failures = [result for result in results if isinstance(result, BaseException)]
        assert len(failures) == 1
        assert type(failures[0]) is WorkGroupTransportClosed

    asyncio.run(scenario())


def test_release_retires_after_all_earlier_accepted_replies_drain() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a"), ReplicaId("replica-a"), worker_instance="worker-a"
        )
        ready = ReplicaReady(**child_fields("worker-a"))
        released = WorkGroupReleased(**child_fields("worker-a"))
        transport.accept(workgroup_frame(ready), ())
        transport.accept(workgroup_frame(released), ())

        assert await endpoint.receive() == ready
        await endpoint.send(ReleaseWorkGroup(**child_fields("worker-a")))
        assert await endpoint.receive() == released
        with pytest.raises(WorkGroupTransportClosed):
            await endpoint.receive()
        with pytest.raises(WorkGroupTransportClosed):
            await endpoint.send(ReleaseWorkGroup(**child_fields("worker-a")))

    asyncio.run(scenario())


def test_queued_release_rejects_late_and_duplicate_replies_before_consumption() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a"), ReplicaId("replica-a"), worker_instance="worker-a"
        )
        released = WorkGroupReleased(**child_fields("worker-a"))
        transport.accept(workgroup_frame(released), ())

        for late in (ReplicaReady(**child_fields("worker-a")), released):
            with pytest.raises(BoundaryError, match="retired group attempt"):
                transport.accept(workgroup_frame(late), ())

        assert await endpoint.receive() == released
        with pytest.raises(WorkGroupTransportClosed):
            await endpoint.receive()

    asyncio.run(scenario())


def test_future_attempt_reply_refuses_without_settling_current_attempt() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a", attempt=1),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )

        for future in (
            ReplicaReady(**child_fields("worker-a", attempt=2)),
            WorkGroupReleased(**child_fields("worker-a", attempt=2)),
        ):
            with pytest.raises(BoundaryError, match="retired group attempt"):
                transport.accept(workgroup_frame(future), ())

        ready = ReplicaReady(**child_fields("worker-a", attempt=1))
        transport.accept(workgroup_frame(ready), ())
        assert await endpoint.receive() == ready

    asyncio.run(scenario())


def test_transport_loss_close_and_cancellation_wake_without_retained_tasks() -> None:
    async def scenario() -> None:
        sent: list[dict[str, object]] = []

        async def send(header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            sent.append(dict(header))

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a"),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        cancelled = asyncio.create_task(endpoint.receive())
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert all(  # noqa: SLF001 - task ownership proof
            not channel.waiters for channel in transport._channels.values()
        )

        retained = ReplicaReady(**child_fields("worker-a"))
        raced = asyncio.create_task(endpoint.receive())
        await asyncio.sleep(0)
        transport.accept(workgroup_frame(retained), ())
        raced.cancel()
        with pytest.raises(asyncio.CancelledError):
            await raced
        assert await endpoint.receive() == retained

        pending = [asyncio.create_task(endpoint.receive()) for _ in range(2)]
        await asyncio.sleep(0)
        transport.fail(ConnectionError("lost"))
        for task in pending:
            with pytest.raises(WorkGroupTransportClosed):
                await task
        assert all(  # noqa: SLF001
            not channel.waiters and not channel.messages for channel in transport._channels.values()
        )
        assert sent == []

    asyncio.run(scenario())


def test_accepted_reply_drains_before_transport_failure() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        transport = WorkGroupSession(send)
        transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
        endpoint = transport.bind(
            definition("worker-a"),
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        pending = asyncio.create_task(endpoint.receive())
        await asyncio.sleep(0)
        reply = ReplicaReady(**child_fields("worker-a"))
        transport.accept(workgroup_frame(reply), ())
        transport.fail(ConnectionError("EOF after reply"))

        assert await pending == reply
        with pytest.raises(WorkGroupTransportClosed):
            await endpoint.receive()

    asyncio.run(scenario())


def test_pristine_unbind_allows_same_attempt_rebind_and_works_after_failure() -> None:
    async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
        return None

    transport = WorkGroupSession(send)
    transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
    current = definition("worker-a")
    old = transport.bind(current, ReplicaId("replica-a"), worker_instance="worker-a")
    transport.unbind(current, ReplicaId("replica-a"))
    rebound = transport.bind(current, ReplicaId("replica-a"), worker_instance="worker-a")

    async def old_endpoint_is_closed() -> None:
        with pytest.raises(WorkGroupTransportClosed):
            await old.receive()

    asyncio.run(old_endpoint_is_closed())
    transport.fail(ConnectionError("lost"))
    transport.unbind(current, ReplicaId("replica-a"))
    assert rebound.binding == current.members[0]


def test_unbind_requires_the_exact_bound_definition() -> None:
    async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
        return None

    transport = WorkGroupSession(send)
    transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
    current = definition("worker-a")
    transport.bind(current, ReplicaId("replica-a"), worker_instance="worker-a")
    foreign = WorkGroupDefinition(
        current.group,
        current.attempt,
        current.members,
        (
            WorkUnitDefinition(
                WorkUnitId("unit-other"), ReplicaId("replica-a"), SemanticSlot.SINGLE
            ),
        ),
    )

    with pytest.raises(ValueError, match="different workgroup definition"):
        transport.unbind(foreign, ReplicaId("replica-a"))
    transport.unbind(current, ReplicaId("replica-a"))


def test_unbind_refuses_active_channels_and_releases_settled_channels() -> None:
    async def scenario() -> None:
        async def send(_header: Mapping[str, object], _blobs: Sequence[bytes]) -> None:
            return None

        def bound(attempt: int) -> tuple[WorkGroupSession, WorkGroupDefinition, object]:
            transport = WorkGroupSession(send)
            transport.negotiate({WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY]})
            current = definition("worker-a", attempt=attempt)
            endpoint = transport.bind(current, ReplicaId("replica-a"), worker_instance="worker-a")
            return transport, current, endpoint

        sent, sent_definition, sent_endpoint = bound(1)
        await sent_endpoint.send(  # type: ignore[union-attr]
            PrepareReplica(
                **child_fields("worker-a", attempt=1),
                recipe=ReplicaRecipeId("sha256:" + "a" * 64),
            )
        )
        with pytest.raises(RuntimeError, match="still active"):
            sent.unbind(sent_definition, ReplicaId("replica-a"))

        queued, queued_definition, _ = bound(2)
        queued.accept(workgroup_frame(ReplicaReady(**child_fields("worker-a", attempt=2))), ())
        with pytest.raises(RuntimeError, match="still active"):
            queued.unbind(queued_definition, ReplicaId("replica-a"))

        waiting, waiting_definition, waiting_endpoint = bound(3)
        waiter = asyncio.create_task(waiting_endpoint.receive())  # type: ignore[union-attr]
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="still active"):
            waiting.unbind(waiting_definition, ReplicaId("replica-a"))
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        terminal, terminal_definition, terminal_endpoint = bound(4)
        released = WorkGroupReleased(**child_fields("worker-a", attempt=4))
        terminal.accept(workgroup_frame(released), ())
        with pytest.raises(RuntimeError, match="still active"):
            terminal.unbind(terminal_definition, ReplicaId("replica-a"))
        assert await terminal_endpoint.receive() == released  # type: ignore[union-attr]
        terminal.unbind(terminal_definition, ReplicaId("replica-a"))
        rebound = terminal.bind(
            terminal_definition,
            ReplicaId("replica-a"),
            worker_instance="worker-a",
        )
        assert rebound.binding == terminal_definition.members[0]

    asyncio.run(scenario())


def test_protocol_corruption_closes_boundary_and_fails_pending_receive() -> None:
    async def scenario() -> None:
        accepted: asyncio.Future[asyncio.StreamWriter] = asyncio.get_running_loop().create_future()

        async def corrupt(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            accepted.set_result(writer)
            await write_frame(
                writer,
                {
                    "type": "hello",
                    "pack": "test",
                    "schemas": {},
                    "workerInstance": "worker-a",
                    WORKGROUP_HELLO_FIELD: [WORKGROUP_CAPABILITY],
                },
                [],
            )
            await asyncio.sleep(0.05)
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
            with pytest.raises(WorkGroupTransportClosed):
                await asyncio.wait_for(endpoint.receive(), 2)
            assert not session.alive
        finally:
            await session.close()
            peer_writer.close()
            await peer_writer.wait_closed()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
