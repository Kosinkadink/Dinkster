"""Portable boundary endpoints (DESIGN 3.3): the framing is transport-
neutral, so an isolated worker must behave identically over a unix socket
and over authenticated loopback TCP - the Windows path, exercised here on
every platform so it cannot rot until someone boots Windows."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import Invocation
from dinkster_workers import (
    BoundaryListener,
    IsolatedWorker,
    TransportChoice,
    unix_endpoints_supported,
)
from dinkster_workers._pipe_transport import bridge_process, pipe_command
from dinkster_workers.boundary import TransferStat
from dinkster_workers.interpreter import module_command
from dinkster_workers.launch import LaunchSpec
from dinkster_workers.transport import TOKEN_ENV
from test_isolated import DEV_MANIFEST, core_registry, image_graph

from tests.async_scenarios import run_scenario


class PipeLauncher:
    supports_shared_memory = False

    def __init__(self) -> None:
        self.bridges: list[asyncio.Task[None]] = []

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        command, endpoints = pipe_command(spec.command)
        process = await asyncio.create_subprocess_exec(
            *module_command(command),
            env={**os.environ, **spec.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        self.bridges.append(asyncio.create_task(bridge_process(process, endpoints, spec.env)))
        return process

    async def close(self) -> None:
        for bridge in self.bridges:
            bridge.cancel()
        await asyncio.gather(*self.bridges, return_exceptions=True)


@pytest.mark.parametrize("grouped", [False, True])
def test_pipe_worker_large_values_and_cancellation(
    tmp_path: Path, shm_segments: set[str], grouped: bool
) -> None:
    from dinkster_workers import GroupIsolatedWorker
    from test_isolated import TESTS_DIR, write_iso_manifest

    async def scenario() -> None:
        launcher = PipeLauncher()
        registry = core_registry()
        if grouped:
            manifest = write_iso_manifest(tmp_path)
            (tmp_path / "isopack_nodes.py").write_text(
                (TESTS_DIR / "isopack_nodes.py").read_text()
                + '\nprint("pack stdout must not corrupt pipe frames", flush=True)\n'
                + 'assert os.environ["HTTPS_PROXY"] == os.environ["DINKSTER_EGRESS_PROXY"]\n'
                + 'assert os.environ["HTTP_PROXY"] == os.environ["DINKSTER_EGRESS_PROXY"]\n'
            )
            owner = GroupIsolatedWorker(
                "pipe-test",
                [manifest, DEV_MANIFEST],
                registry,
                extra_env={
                    "DINKSTER_EGRESS_PROXY": "http://dinkster:unit-test@127.0.0.1:12345",
                    "HTTP_PROXY": "http://127.0.0.1:12345",
                    "HTTPS_PROXY": "http://127.0.0.1:12345",
                },
                launcher=launcher,
                transport="tcp",
                shm_threshold=1,
            )
            worker = owner.members["dinkster-nodes-dev"]
        else:
            owner = IsolatedWorker(
                DEV_MANIFEST, registry, launcher=launcher, transport="tcp", shm_threshold=1
            )
            worker = owner
        await owner.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            value = "asymmetric-0123456789" * 50_000
            graph = Graph(
                nodes={"echo": GraphNode("dev.util.delay", {"value": value, "seconds": 0.0})}
            )
            result = await engine.run(graph, ["echo"])
            assert next(iter(result.outputs["echo"].values())).resolve() == value
            slow = Graph(
                nodes={"slow": GraphNode("dev.util.delay", {"value": value, "seconds": 30.0})}
            )
            pending = asyncio.create_task(engine.run(slow, ["slow"]))
            if isinstance(owner, GroupIsolatedWorker):
                sibling = owner.members["isopack"]
                other = Engine(
                    schemas=dict(sibling.schemas),
                    registry=registry,
                    worker=sibling,
                    cache=MemoryLRUCache(),
                )
                second = Graph(
                    nodes={
                        "sibling": GraphNode("iso.sleepy", {"value": value[::-1], "seconds": 0.0})
                    }
                )
                result = await other.run(second, ["sibling"])
                assert result.outputs["sibling"]["value"].resolve() == value[::-1]
            else:
                await asyncio.sleep(0.2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            uncached = Graph(
                nodes={
                    "echo": GraphNode("dev.util.delay", {"value": value + "after", "seconds": 0.0})
                }
            )
            again = await engine.run(uncached, ["echo"])
            assert again.outputs["echo"]["value"].resolve() == value + "after"
        finally:
            await owner.close()
            await launcher.close()

    run_scenario(scenario(), timeout=60)
    assert not shm_segments


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

    run_scenario(scenario())


def test_isolated_worker_over_tcp_matches_unix(tmp_path: Path, shm_segments: set[str]) -> None:
    """The whole M2 flow - handshake, invocation, shm handoff, ack - over
    the TCP fallback. Fingerprints must match the unix-transport run
    (hazard H4: location - and transport - independence)."""

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

    run_scenario(scenario())
    if os.path.isdir("/dev/shm"):
        assert shm_segments
        assert not [name for name in shm_segments if (Path("/dev/shm") / name).exists()]


def test_cancel_mid_flight_leaves_no_segments(
    monkeypatch: pytest.MonkeyPatch, shm_segments: set[str]
) -> None:
    """Held-until-ack segment handles (the Windows-safe lifetime) must not
    leak when an invocation is cancelled after its shm inputs were sent.
    The large string input rides shm (threshold 64); whether the cancel
    lands before or after the child's ack, the sender's release paths must
    leave nothing behind."""

    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry, shm_threshold=64)
        await worker.start()
        try:
            sent = asyncio.Event()
            send_invocation = worker._session._send_invocation

            async def send(invocation: Invocation) -> dict[str, TransferStat]:
                stats = await send_invocation(invocation)
                sent.set()
                return stats

            monkeypatch.setattr(worker._session, "_send_invocation", send)
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
            await asyncio.wait_for(sent.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await worker.close()

    run_scenario(scenario())
    if os.path.isdir("/dev/shm"):
        assert shm_segments
        assert not [name for name in shm_segments if (Path("/dev/shm") / name).exists()]
