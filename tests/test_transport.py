"""Portable boundary endpoints (DESIGN 3.3): the framing is transport-
neutral, so an isolated worker must behave identically over a unix socket
and over authenticated loopback TCP - the Windows path, exercised here on
every platform so it cannot rot until someone boots Windows."""

from __future__ import annotations

import asyncio
import glob
import os
import tempfile
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode
from dinkster_workers import (
    BoundaryListener,
    IsolatedWorker,
    TransportChoice,
    TransportError,
    connect_endpoint,
    unix_endpoints_supported,
)
from dinkster_workers.transport import TOKEN_ENV
from test_isolated import DEV_MANIFEST, core_registry, image_graph


def test_endpoint_parsing_rejects_garbage() -> None:
    async def scenario() -> None:
        for endpoint in ("unix:", "tcp:no-port", "tcp:127.0.0.1:notaport", "carrier-pigeon:x"):
            with pytest.raises(TransportError):
                await connect_endpoint(endpoint)

    asyncio.run(scenario())


def test_tcp_listener_rejects_wrong_token_and_accepts_right_one() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            listener = await BoundaryListener.create(Path(tmpdir), transport="tcp")
            try:
                assert listener.endpoint.startswith("tcp:127.0.0.1:")
                token = listener.child_env[TOKEN_ENV]
                host, port = "127.0.0.1", int(listener.endpoint.rsplit(":", 1)[1])

                # Wrong token: the connection is dropped, nothing resolves.
                _, bad_writer = await asyncio.open_connection(host, port)
                bad_writer.write(b"f" * len(token))
                await bad_writer.drain()
                done, _ = await asyncio.wait([listener.connected], timeout=0.2)
                assert not done
                bad_writer.close()

                # Right token: the future resolves with a usable stream.
                _, good_writer = await asyncio.open_connection(host, port)
                good_writer.write(token.encode("ascii"))
                await good_writer.drain()
                reader, writer = await asyncio.wait_for(listener.connected, 5.0)
                good_writer.write(b"ping")
                await good_writer.drain()
                assert await reader.readexactly(4) == b"ping"
                writer.close()
                good_writer.close()
            finally:
                await listener.close()

    asyncio.run(scenario())


def test_missing_token_fails_fast_on_child_side() -> None:
    async def scenario() -> None:
        assert os.environ.get(TOKEN_ENV) is None
        with pytest.raises(TransportError, match=TOKEN_ENV):
            await connect_endpoint("tcp:127.0.0.1:1")

    asyncio.run(scenario())


def test_unix_listener_where_supported() -> None:
    if not unix_endpoints_supported():
        pytest.skip("no unix endpoints on this platform")

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            listener = await BoundaryListener.create(Path(tmpdir), transport="auto")
            try:
                assert listener.endpoint.startswith("unix:")
                assert listener.child_env == {}
                _, client_writer = await connect_endpoint(listener.endpoint)
                reader, writer = await asyncio.wait_for(listener.connected, 5.0)
                client_writer.write(b"ping")
                await client_writer.drain()
                assert await reader.readexactly(4) == b"ping"
                writer.close()
                client_writer.close()
            finally:
                await listener.close()

    asyncio.run(scenario())


def test_isolated_worker_over_tcp_matches_unix(tmp_path: Path) -> None:
    """The whole M2 flow - handshake, invocation, shm handoff, ack - over
    the TCP fallback. Fingerprints must match the unix-transport run
    (hazard H4: location - and transport - independence)."""
    existing_segments = set(glob.glob("/dev/shm/dinkster*"))

    async def run_with(transport: TransportChoice) -> dict[str, str]:
        registry = core_registry()
        worker = IsolatedWorker(
            DEV_MANIFEST,
            registry,
            transport=transport,
            shm_threshold=64,  # force the shm path for image payloads
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(image_graph(), ["s", "b"])
            return {
                output_id: value.fingerprint for output_id, value in result.outputs["s"].items()
            }
        finally:
            await worker.close()

    async def scenario() -> None:
        over_tcp = await run_with("tcp")
        if unix_endpoints_supported():
            assert over_tcp == await run_with("unix")

    asyncio.run(scenario())
    if os.path.isdir("/dev/shm"):
        assert not set(glob.glob("/dev/shm/dinkster*")).difference(existing_segments)


def test_cancel_mid_flight_leaves_no_segments() -> None:
    """Held-until-ack segment handles (the Windows-safe lifetime) must not
    leak when an invocation is cancelled after its shm inputs were sent.
    The large string input rides shm (threshold 64); whether the cancel
    lands before or after the child's ack, the sender's release paths must
    leave nothing behind."""
    existing_segments = set(glob.glob("/dev/shm/dinkster*"))

    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry, shm_threshold=64)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "w": GraphNode("dev.util.delay", {"value": "x" * 1_000_000, "seconds": 30.0})
                }
            )
            task = asyncio.create_task(engine.run(graph, ["w"]))
            await asyncio.sleep(0.5)  # let the invocation cross the boundary
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await worker.close()

    asyncio.run(scenario())
    if os.path.isdir("/dev/shm"):
        assert not set(glob.glob("/dev/shm/dinkster*")).difference(existing_segments)
