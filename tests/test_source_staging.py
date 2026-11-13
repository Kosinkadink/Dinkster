"""Per-invocation source staging lifetime through the worker protocol."""

from __future__ import annotations

import asyncio
import struct
import threading
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from dinkster_assets import AssetRef, digest_bytes
from dinkster_protocol import Invocation, InvocationResult, NodeError
from dinkster_schema import NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import current_execution_context
from dinkster_workers.boundary import ValueCodec, encode_invocation, read_frame, write_frame
from dinkster_workers.host import serve_connection

from tests.async_scenarios import run_scenario


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")
    )


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        del digest
        return self.path


def _ref(path: Path, *, name: str = "untrusted.exe") -> AssetRef:
    data = path.read_bytes()
    return AssetRef(
        digest_bytes(data),
        name,
        999999,
        "text/html",
        "../../ambient",
        _Resolver(path),
    )


_HOST_SCHEMA = NodeSchema(
    node_type="test.stage",
    inputs=(),
    outputs=(OutputSpec("out", TypeExpr.concrete("STRING")),),
)


class _HostSession:
    def __init__(self, events: list[str], *, close_error: bool) -> None:
        self.events = events
        self.close_error = close_error

    def materialize(self, asset: AssetRef, kind: str, category: str) -> str:
        del asset, kind, category
        self.events.append("materialize")
        return ".dinkster-source/session/digest.png"

    def close(self) -> None:
        self.events.append("close")
        if self.close_error:
            raise RuntimeError("close boom")


class _HostProvider:
    def __init__(self, events: list[str], *, close_error: bool = False) -> None:
        self.events = events
        self.close_error = close_error

    def sweep(self) -> None:
        self.events.append("sweep")

    def open(self, invocation_id: str, authorities: Sequence[object]) -> _HostSession:
        del invocation_id, authorities
        self.events.append("open")
        return _HostSession(self.events, close_error=self.close_error)


class _HostWorker:
    attention_capabilities = None
    attention_route_token = None
    registered_types: tuple[str, ...] = ()

    def __init__(self, registry: TypeRegistry, ref: AssetRef, *, node_error: bool) -> None:
        self.registry = registry
        self.ref = ref
        self.node_error = node_error

    async def invoke(self, invocation: Invocation, on_event: Any = None) -> InvocationResult:
        del on_event
        context = current_execution_context()
        assert context is not None and context.materialize_source is not None
        context.materialize_source(self.ref, "media/image", "input")
        if self.node_error:
            return InvocationResult(
                error=NodeError(invocation.node_id, invocation.node_type, "node boom")
            )
        return InvocationResult(outputs={"out": self.registry.wrap("STRING", "ok")})


async def _host_result(
    ref: AssetRef, provider: _HostProvider, *, node_error: bool = False
) -> dict[str, Any]:
    registry = TypeRegistry()
    register_core_types(registry)
    accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        accepted.set_result(task)
        await serve_connection(
            reader,
            writer,
            pack_name="test",
            worker=_HostWorker(registry, ref, node_error=node_error),  # type: ignore[arg-type]
            schemas={"test.stage": _HOST_SCHEMA},
            planner=None,
            consumers={},
            codec=ValueCodec(registry),
            source_staging=provider,  # type: ignore[arg-type]
        )

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(address[0], address[1])
    host_task = await accepted
    hello = await read_frame(reader)
    assert hello is not None and hello[0]["type"] == "hello"
    invocation = Invocation("inv", "node", "test.stage", {}, _HOST_SCHEMA)
    header, blobs, segments, _ = encode_invocation(ValueCodec(registry), invocation)
    assert not segments
    await write_frame(writer, header, blobs)
    frame = await read_frame(reader)
    assert frame is not None
    await write_frame(writer, {"type": "shutdown"}, [])
    await host_task
    server.close()
    await server.wait_closed()
    return frame[0]


def test_host_closes_before_success_and_turns_cleanup_failure_into_node_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)

    events: list[str] = []
    result = run_scenario(_host_result(ref, _HostProvider(events)))
    assert "outputs" in result and "error" not in result
    assert events == ["sweep", "open", "materialize", "close"]

    events = []
    result = run_scenario(_host_result(ref, _HostProvider(events, close_error=True)))
    assert result["error"]["message"].startswith("source staging cleanup failed")
    assert events == ["sweep", "open", "materialize", "close"]


class _BlockingSession(_HostSession):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events, close_error=False)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def materialize(self, asset: AssetRef, kind: str, category: str) -> str:
        self.events.append("materialize")
        self.entered.set()
        assert self.release.wait(5)
        return super().materialize(asset, kind, category)

    def close(self) -> None:
        super().close()
        self.closed.set()


class _BlockingProvider(_HostProvider):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.session = _BlockingSession(events)

    def open(self, invocation_id: str, authorities: Sequence[object]) -> _BlockingSession:
        del invocation_id, authorities
        self.events.append("open")
        return self.session


class _LateMaterializingWorker(_HostWorker):
    def __init__(self, registry: TypeRegistry, ref: AssetRef, closed: threading.Event) -> None:
        super().__init__(registry, ref, node_error=False)
        self.closed = closed
        self.late_refused = threading.Event()

    async def invoke(self, invocation: Invocation, on_event: Any = None) -> InvocationResult:
        del on_event

        def execute() -> InvocationResult:
            context = current_execution_context()
            assert context is not None and context.materialize_source is not None
            context.materialize_source(self.ref, "media/image", "input")
            assert self.closed.wait(5)
            try:
                context.materialize_source(self.ref, "media/image", "input")
            except RuntimeError as exc:
                assert "closing or closed" in str(exc)
                self.late_refused.set()
            return InvocationResult(outputs={"out": self.registry.wrap("STRING", "ok")})

        return await asyncio.to_thread(execute)


@pytest.mark.parametrize("terminal", ["cancel", "disconnect"])
def test_host_terminal_paths_close_and_refuse_late_materialization(
    tmp_path: Path, terminal: str
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source.png"
        source.write_bytes(_png())
        ref = _ref(source)
        events: list[str] = []
        provider = _BlockingProvider(events)
        registry = TypeRegistry()
        register_core_types(registry)
        worker = _LateMaterializingWorker(registry, ref, provider.session.closed)
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await serve_connection(
                reader,
                writer,
                pack_name="test",
                worker=worker,  # type: ignore[arg-type]
                schemas={"test.stage": _HOST_SCHEMA},
                planner=None,
                consumers={},
                codec=ValueCodec(registry),
                source_staging=provider,  # type: ignore[arg-type]
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        hello = await read_frame(reader)
        assert hello is not None and hello[0]["type"] == "hello"
        invocation = Invocation("inv", "node", "test.stage", {}, _HOST_SCHEMA)
        header, blobs, segments, _ = encode_invocation(ValueCodec(registry), invocation)
        assert not segments
        await write_frame(writer, header, blobs)
        assert await asyncio.to_thread(provider.session.entered.wait, 5)
        if terminal == "cancel":
            await write_frame(writer, {"type": "cancel", "invocationId": "inv"}, [])
        else:
            writer.close()
            await writer.wait_closed()
        provider.session.release.set()
        assert await asyncio.to_thread(provider.session.closed.wait, 5)
        assert await asyncio.to_thread(worker.late_refused.wait, 5)
        if terminal == "cancel":
            await write_frame(writer, {"type": "shutdown"}, [])
        await host_task
        server.close()
        await server.wait_closed()
        assert events == ["sweep", "open", "materialize", "materialize", "close"]

    run_scenario(scenario())


class _DoubleCancelWorker(_HostWorker):
    def __init__(self, registry: TypeRegistry, ref: AssetRef) -> None:
        super().__init__(registry, ref, node_error=False)
        self.entered = threading.Event()
        self.allow_late = threading.Event()
        self.late_refused = threading.Event()

    async def invoke(self, invocation: Invocation, on_event: Any = None) -> InvocationResult:
        del on_event

        def execute() -> InvocationResult:
            context = current_execution_context()
            assert context is not None and context.materialize_source is not None
            context.materialize_source(self.ref, "media/image", "input")
            self.entered.set()
            assert self.allow_late.wait(5)
            try:
                context.materialize_source(self.ref, "media/image", "input")
            except RuntimeError as exc:
                assert "closing or closed" in str(exc)
                self.late_refused.set()
            return InvocationResult(outputs={"out": self.registry.wrap("STRING", "ok")})

        return await asyncio.to_thread(execute)
