"""Per-invocation Comfy source staging authority and lifetime."""

from __future__ import annotations

import asyncio
import os
import struct
import threading
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dinkster_compat_comfy.source_staging as source_staging_module
import pytest
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_compat_comfy.source_staging import (
    ComfySourceStagingProvider,
    SourceStagingError,
)
from dinkster_memory import ReservationRequest
from dinkster_protocol import Invocation, InvocationResult, MediaSourceAuthority, NodeError
from dinkster_schema import NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import current_execution_context
from dinkster_workers.boundary import ValueCodec, encode_invocation, read_frame, write_frame
from dinkster_workers.host import serve_connection

from tests.platform_support import symlink_or_skip

requires_confined_source_staging = pytest.mark.skipif(
    os.name != "posix", reason="source staging requires POSIX descriptor APIs"
)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")
    )


class _FolderPaths:
    def __init__(self, root: Path) -> None:
        self.roots = {name: root / name for name in ("input", "output", "temp")}
        for path in self.roots.values():
            path.mkdir(parents=True)

    def get_input_directory(self) -> str:
        return str(self.roots["input"])

    def get_output_directory(self) -> str:
        return str(self.roots["output"])

    def get_temp_directory(self) -> str:
        return str(self.roots["temp"])

    def get_annotated_filepath(self, name: str) -> str:
        category = "input"
        if name.endswith(" [output]"):
            category, name = "output", name[:-9]
        elif name.endswith(" [temp]"):
            category, name = "temp", name[:-7]
        return str(Path(os.path.abspath(self.roots[category] / name)))


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


def _authority(ref: AssetRef, size: int) -> MediaSourceAuthority:
    return MediaSourceAuthority(
        ref.digest,
        "media/image",
        "image/png",
        "png",
        size,
    )


_HOST_SCHEMA = NodeSchema(
    node_type="test.stage",
    inputs=(),
    outputs=(OutputSpec("out", TypeExpr.concrete("core.string")),),
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
        return InvocationResult(outputs={"out": self.registry.wrap("core.string", "ok")})


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
    result = asyncio.run(_host_result(ref, _HostProvider(events)))
    assert "outputs" in result and "error" not in result
    assert events == ["sweep", "open", "materialize", "close"]

    events = []
    result = asyncio.run(_host_result(ref, _HostProvider(events, close_error=True)))
    assert result["error"]["message"].startswith("source staging cleanup failed")
    assert events == ["sweep", "open", "materialize", "close"]


def test_host_keeps_node_error_primary_when_cleanup_also_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    events: list[str] = []
    result = asyncio.run(
        _host_result(
            _ref(source),
            _HostProvider(events, close_error=True),
            node_error=True,
        )
    )
    assert result["error"]["message"] == "node boom"
    assert "Source staging cleanup also failed" in result["error"]["traceback"]
    assert events == ["sweep", "open", "materialize", "close"]


def test_host_reserve_denial_creates_no_staging_residue(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "source.png"
        source.write_bytes(_png())
        ref = _ref(source)
        events: list[str] = []
        provider = _HostProvider(events)
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
                worker=_HostWorker(registry, ref, node_error=False),  # type: ignore[arg-type]
                schemas={"test.stage": _HOST_SCHEMA},
                planner=lambda invocation: (ReservationRequest("ram", 1),),
                consumers={},
                codec=ValueCodec(registry),
                source_staging=provider,  # type: ignore[arg-type]
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        assert (await read_frame(reader))[0]["type"] == "hello"  # type: ignore[index]
        invocation = Invocation("inv", "node", "test.stage", {}, _HOST_SCHEMA)
        header, blobs, segments, _ = encode_invocation(ValueCodec(registry), invocation)
        assert not segments
        await write_frame(writer, header, blobs)
        reserve = await read_frame(reader)
        assert reserve is not None and reserve[0]["type"] == "memoryReserve"
        await write_frame(
            writer,
            {"type": "memoryDeny", "requestId": "inv", "message": "denied"},
            [],
        )
        result = await read_frame(reader)
        assert result is not None and "memory admission failed" in result[0]["error"]["message"]
        release = await read_frame(reader)
        assert release is not None and release[0]["type"] == "memoryRelease"
        await write_frame(writer, {"type": "shutdown"}, [])
        await host_task
        server.close()
        await server.wait_closed()
        assert events == ["sweep"]

    asyncio.run(scenario())


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
            return InvocationResult(outputs={"out": self.registry.wrap("core.string", "ok")})

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

    asyncio.run(scenario())


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
            return InvocationResult(outputs={"out": self.registry.wrap("core.string", "ok")})

        return await asyncio.to_thread(execute)


def test_cancel_then_disconnect_refuses_late_materialization_before_cleanup_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source.png"
        source.write_bytes(_png())
        ref = _ref(source)
        events: list[str] = []
        provider = _BlockingProvider(events)
        registry = TypeRegistry()
        register_core_types(registry)
        worker = _DoubleCancelWorker(registry, ref)
        cleanup_queued = asyncio.Event()
        cleanup_can_start = asyncio.Event()
        original_to_thread = asyncio.to_thread

        async def controlled_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
            if getattr(function, "__name__", "") == "close_staging":
                cleanup_queued.set()
                await cleanup_can_start.wait()
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
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
        try:
            assert (await read_frame(reader))[0]["type"] == "hello"  # type: ignore[index]
            invocation = Invocation("inv", "node", "test.stage", {}, _HOST_SCHEMA)
            header, blobs, segments, _ = encode_invocation(ValueCodec(registry), invocation)
            assert not segments
            await write_frame(writer, header, blobs)
            assert await original_to_thread(provider.session.entered.wait, 5)
            await write_frame(writer, {"type": "cancel", "invocationId": "inv"}, [])
            await cleanup_queued.wait()
            writer.close()
            await writer.wait_closed()
            assert not host_task.done()
            provider.session.release.set()
            assert await original_to_thread(worker.entered.wait, 5)
            worker.allow_late.set()
            assert await original_to_thread(worker.late_refused.wait, 5)
            assert not provider.session.closed.is_set()
            assert not host_task.done()
            cleanup_can_start.set()
            assert await original_to_thread(provider.session.closed.wait, 5)
        finally:
            provider.session.release.set()
            worker.allow_late.set()
            cleanup_can_start.set()
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(host_task, 5)
            server.close()
            await server.wait_closed()
        assert events == ["sweep", "open", "materialize", "materialize", "close"]

    asyncio.run(scenario())


@requires_confined_source_staging
def test_authorized_vault_bytes_stage_all_categories_and_cleanup(tmp_path: Path) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    vault = AssetVault(tmp_path / "vault")
    data = _png()
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        path = writer.commit()
    ref = _ref(path)
    provider = ComfySourceStagingProvider(folder_paths)
    session = provider.open("raw/user/invocation", (_authority(ref, len(data)),))
    names = {
        category: session.materialize(ref, "media/image", category)
        for category in ("input", "output", "temp")
    }
    assert names["input"].startswith(".dinkster-source/")
    assert names["output"].endswith(" [output]")
    assert names["temp"].endswith(" [temp]")
    assert "raw" not in "".join(names.values())
    assert session.materialize(ref, "media/image", "input") == names["input"]
    for category, name in names.items():
        plain = name.removesuffix(f" [{category}]")
        assert (folder_paths.roots[category] / plain).read_bytes() == data
    session.close()
    assert not any(
        any(root.joinpath(".dinkster-source").glob("*")) for root in folder_paths.roots.values()
    )


@requires_confined_source_staging
def test_mount_requires_fresh_bytes_and_vault_requires_authority(tmp_path: Path) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    mounted = folder_paths.roots["input"] / "operator.bin"
    mounted.write_bytes(_png())
    provider = ComfySourceStagingProvider(folder_paths)
    session = provider.open("inv", ())
    assert session.materialize(_ref(mounted), "media/image", "input").endswith(".png")
    session.close()

    outside = tmp_path / "vault.png"
    outside.write_bytes(_png())
    session = provider.open("inv", ())
    with pytest.raises(SourceStagingError, match="not operator-mounted"):
        session.materialize(_ref(outside), "media/image", "input")
    session.close()


@requires_confined_source_staging
def test_provider_is_lazy_and_rejects_symlinked_category_roots(tmp_path: Path) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    provider = ComfySourceStagingProvider(folder_paths)
    session = provider.open("inv", ())
    assert not any(
        root.joinpath(".dinkster-source").exists() for root in folder_paths.roots.values()
    )
    session.close()

    real_input = folder_paths.roots["input"]
    moved_input = real_input.with_name("real-input")
    real_input.rename(moved_input)
    symlink_or_skip(real_input, moved_input, target_is_directory=True)
    with pytest.raises(SourceStagingError, match="category roots must be real non-symlink"):
        ComfySourceStagingProvider(folder_paths)


def test_unsupported_platform_is_dormant_until_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    if os.name == "posix":
        monkeypatch.setattr(source_staging_module, "_supports_confined_staging", lambda: False)
    provider = ComfySourceStagingProvider(folder_paths)
    provider.sweep()
    session = provider.open("inv", ())
    with pytest.raises(SourceStagingError, match="unavailable on this platform"):
        session.materialize(object(), "media/image", "input")
    session.close()
    assert not any(
        root.joinpath(".dinkster-source").exists() for root in folder_paths.roots.values()
    )


@requires_confined_source_staging
def test_symlinked_comfy_base_preserves_lexical_round_trip(tmp_path: Path) -> None:
    real = tmp_path / "real-comfy"
    real.mkdir()
    linked = tmp_path / "linked-comfy"
    symlink_or_skip(linked, real, target_is_directory=True)
    folder_paths = _FolderPaths(linked)
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)
    session = ComfySourceStagingProvider(folder_paths).open("inv", (_authority(ref, len(_png())),))
    name = session.materialize(ref, "media/image", "input")
    assert Path(folder_paths.get_annotated_filepath(name)).is_relative_to(linked / "input")
    session.close()


@requires_confined_source_staging
def test_authority_and_materialization_refuse_mismatch_and_unsafe_targets(
    tmp_path: Path,
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)
    provider = ComfySourceStagingProvider(folder_paths)
    bad = MediaSourceAuthority(ref.digest, "media/video", "video/mp4", "mp4", len(_png()))
    session = provider.open("inv", (bad,))
    with pytest.raises(SourceStagingError, match="requested kind|carried authority"):
        session.materialize(ref, "media/image", "input")
    session.close()

    session = provider.open("inv", (_authority(ref, len(_png())),))
    with pytest.raises(SourceStagingError, match="requested media kind"):
        session.materialize(ref, "image", "input")
    with pytest.raises(SourceStagingError, match="source category"):
        session.materialize(ref, "media/image", "../input")
    with pytest.raises(SourceStagingError, match="only AssetRef"):
        session.materialize(object(), "media/image", "input")  # type: ignore[arg-type]
    session.close()

    session = provider.open("inv", (_authority(ref, 1024 * 1024 * 1024 + 1),))
    with pytest.raises(SourceStagingError, match="exceeds the bounded staging limit"):
        session.materialize(ref, "media/image", "input")
    session.close()


@requires_confined_source_staging
def test_sweep_preserves_active_and_unknown_entries(
    tmp_path: Path,
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)
    provider = ComfySourceStagingProvider(folder_paths)
    session = provider.open("inv", (_authority(ref, len(_png())),))
    session.materialize(ref, "media/image", "input")
    staging = folder_paths.roots["input"] / ".dinkster-source"
    live = next(path for path in staging.iterdir() if path.name[0] != ".")

    unknown = staging / ("1" * 32)
    unknown.mkdir()
    (unknown / ".owner").write_bytes(b"")
    (unknown / "unknown").write_bytes(b"preserve")
    symlink = staging / ("2" * 32)
    symlink_or_skip(symlink, unknown, target_is_directory=True)
    dead = staging / ("3" * 32)
    dead.mkdir()
    (dead / ".owner").write_bytes(b"")
    (dead / ("a" * 64 + ".png")).write_bytes(_png())

    provider.sweep()
    assert live.exists()
    assert unknown.exists()
    assert symlink.is_symlink()
    assert not dead.exists()
    session.close()


@requires_confined_source_staging
def test_sweep_reclaims_strict_abandoned_claims_and_removal_tombstones(
    tmp_path: Path,
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    staging = folder_paths.roots["input"] / ".dinkster-source"
    staging.mkdir()
    live_claim = staging / f".claim-{os.getpid()}-{'1' * 32}-{'2' * 16}"
    live_claim.mkdir()
    dead_claim = staging / f".claim-99999999-{'3' * 32}-{'4' * 16}"
    dead_claim.mkdir()
    live_removal = staging / f".remove-{os.getpid()}-{'5' * 32}"
    live_removal.mkdir()
    dead_removal = staging / f".remove-99999999-{'6' * 32}"
    dead_removal.mkdir()
    unknown_claim = staging / ".claim-not-ours"
    unknown_claim.mkdir()
    oversized_claim = staging / f".claim-{'9' * 21}-{'7' * 32}-{'8' * 16}"
    oversized_claim.mkdir()
    leading_zero_claim = staging / f".claim-0001-{'9' * 32}-{'a' * 16}"
    leading_zero_claim.mkdir()
    unicode_claim = staging / f".claim-\u0661-{'b' * 32}-{'c' * 16}"
    unicode_claim.mkdir()

    ComfySourceStagingProvider(folder_paths).sweep()

    assert live_claim.exists()
    assert not dead_claim.exists()
    assert live_removal.exists()
    assert not dead_removal.exists()
    assert unknown_claim.exists()
    assert oversized_claim.exists()
    assert leading_zero_claim.exists()
    assert unicode_claim.exists()


@requires_confined_source_staging
def test_live_cleanup_tombstone_is_preserved_by_concurrent_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)
    provider = ComfySourceStagingProvider(folder_paths)
    session = provider.open("inv", (_authority(ref, len(_png())),))
    session.materialize(ref, "media/image", "input")
    original_stat = os.stat
    original_kill = os.kill
    checked_pids: list[int] = []
    swept = False

    def record_liveness(pid: int, signal: int) -> None:
        checked_pids.append(pid)
        original_kill(pid, signal)

    def sweep_during_removal(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal swept
        if isinstance(path, str) and path.startswith(".remove-") and not swept:
            swept = True
            provider.sweep()
        return original_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "kill", record_liveness)
    monkeypatch.setattr(os, "stat", sweep_during_removal)
    session.close()

    assert swept
    assert os.getpid() in checked_pids


@requires_confined_source_staging
def test_cleanup_refuses_rebound_session_without_removing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    ref = _ref(source)
    session = ComfySourceStagingProvider(folder_paths).open("inv", (_authority(ref, len(_png())),))
    name = session.materialize(ref, "media/image", "input")
    session_path = folder_paths.roots["input"] / ".dinkster-source" / Path(name).parts[1]
    displaced = session_path.with_name("displaced")
    session_path.rename(displaced)
    session_path.mkdir()
    (session_path / "foreign").write_bytes(b"preserve")
    original_stat = os.stat
    rebound = False

    def rebind_during_removal(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal rebound
        if isinstance(path, str) and path.startswith(".remove-") and not rebound:
            rebound = True
            session_path.mkdir()
            (session_path / "newer").write_bytes(b"also preserve")
        return original_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", rebind_during_removal)

    with pytest.raises(SourceStagingError, match="rebound during cleanup"):
        session.close()

    assert (session_path / "newer").read_bytes() == b"also preserve"
    tombstones = tuple(session_path.parent.glob(".remove-*"))
    assert len(tombstones) == 1
    assert (tombstones[0] / "foreign").read_bytes() == b"preserve"
    assert displaced.exists()


@requires_confined_source_staging
def test_sweep_preserves_replacements_when_stale_session_rebinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    provider = ComfySourceStagingProvider(folder_paths)
    staging = folder_paths.roots["input"] / ".dinkster-source"
    staging.mkdir()
    session_path = staging / ("1" * 32)
    session_path.mkdir()
    (session_path / ".owner").write_bytes(b"")
    original = session_path.with_name("original")
    original_rename = os.rename
    rebound = False

    def rebind_during_removal(
        source: object, target: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal rebound
        if (
            source == session_path.name
            and isinstance(target, str)
            and target.startswith(".remove-")
            and not rebound
        ):
            rebound = True
            original_rename(source, original.name, *args, **kwargs)  # type: ignore[arg-type]
            session_path.mkdir()
            (session_path / ".owner").write_bytes(b"replacement")
        original_rename(source, target, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", rebind_during_removal)
    provider.sweep()

    tombstones = tuple(staging.glob(".remove-*"))
    assert len(tombstones) == 1
    assert (tombstones[0] / ".owner").read_bytes() == b"replacement"
    assert (original / ".owner").is_file()


@requires_confined_source_staging
def test_sweep_preserves_stat_open_rebind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    provider = ComfySourceStagingProvider(folder_paths)
    staging = folder_paths.roots["input"] / ".dinkster-source"
    staging.mkdir()
    session_path = staging / ("1" * 32)
    session_path.mkdir()
    (session_path / ".owner").write_bytes(b"original")
    displaced = staging / "displaced"
    original_open = os.open
    rebound = False

    def rebind_before_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal rebound
        if path == session_path.name and not rebound:
            rebound = True
            session_path.rename(displaced)
            session_path.mkdir()
            (session_path / ".owner").write_bytes(b"replacement")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", rebind_before_open)
    provider.sweep()

    assert (displaced / ".owner").read_bytes() == b"original"
    assert (session_path / ".owner").read_bytes() == b"replacement"


@requires_confined_source_staging
def test_staging_root_symlink_refuses(tmp_path: Path) -> None:
    folder_paths = _FolderPaths(tmp_path / "comfy")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    symlink_or_skip(
        folder_paths.roots["input"] / ".dinkster-source", elsewhere, target_is_directory=True
    )
    provider = ComfySourceStagingProvider(folder_paths)
    with pytest.raises(SourceStagingError, match="must not be a symlink"):
        provider.sweep()
