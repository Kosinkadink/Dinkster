"""Daemon session leases. The one-conversation slot is held by a TTL lease:
a second engine is refused with a frame naming the current holder; a holder
silent past the TTL is evicted (a live engine heartbeats while idle, so
only a crashed or partitioned one goes silent) and can never wedge the
daemon; and the engine applies the same clock to the daemon, declaring a
silent one dead instead of waiting on TCP."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest
from dinkster_workers.boundary import PROTOCOL_VERSION, read_frame, write_frame
from dinkster_workers.service import ServiceError, run_service
from test_isolated import DEV_MANIFEST, core_registry, image_graph
from test_relay import eventually
from test_remote import TOKEN, make_engine, remote_worker, start_service, stop_service

_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
_SIGCONT = getattr(signal, "SIGCONT", signal.SIGTERM)


def test_lease_ttl_below_the_heartbeat_floor_is_refused() -> None:
    """The engine heartbeats at clamp(ttl/3, 1s, 15s): a positive TTL under
    the minimum would evict every healthy idle session, so the service
    refuses it up front. Only 0 (disabled) or >= the minimum is valid."""
    for bad in (0.5, -1.0, float("nan"), float("inf")):
        with pytest.raises(ServiceError, match="lease-ttl"):
            asyncio.run(run_service("127.0.0.1", 0, "unused", b"x" * 16, lease_ttl=bad))


def test_resume_grace_must_be_finite_and_non_negative() -> None:
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ServiceError, match="resume-grace"):
            asyncio.run(run_service("127.0.0.1", 0, "unused", b"x" * 16, resume_grace=bad))


async def zombie_client(
    host: str, port: int
) -> tuple[
    asyncio.StreamReader,
    asyncio.StreamWriter,
    dict[str, object],
    dict[str, object],
]:
    """An authenticated client that completes the hello and then goes silent
    without ever disconnecting: what a crashed engine's half-open connection
    looks like from the daemon's side."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(TOKEN.encode("utf-8"))
    await writer.drain()
    await write_frame(
        writer,
        {
            "type": "clientHello",
            "protocol": PROTOCOL_VERSION,
            "payloadTransports": ["inline"],
            "engine": {
                "label": "zombie@nowhere:pid0",
                "instanceId": "zombie-engine-instance",
            },
        },
        [],
    )
    frame = await asyncio.wait_for(read_frame(reader), 30.0)
    assert frame is not None and frame[0].get("type") == "resumeAccepted"
    admission = frame[0]
    frame = await asyncio.wait_for(read_frame(reader), 30.0)
    assert frame is not None and frame[0].get("type") == "hello"
    return reader, writer, admission, frame[0]


def test_engine_identity_alone_cannot_supersede_a_live_transport(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        first_writer: asyncio.StreamWriter | None = None
        second_writer: asyncio.StreamWriter | None = None
        try:
            first_reader, first_writer, _admission, _hello = await zombie_client(host, port)
            second_reader, second_writer = await asyncio.open_connection(host, port)
            second_writer.write(TOKEN.encode("utf-8"))
            await second_writer.drain()
            await write_frame(
                second_writer,
                {
                    "type": "clientHello",
                    "protocol": PROTOCOL_VERSION,
                    "payloadTransports": ["inline"],
                    "engine": {
                        "label": "zombie@nowhere:pid0",
                        "instanceId": "zombie-engine-instance",
                    },
                },
                [],
            )
            refusal = await asyncio.wait_for(read_frame(second_reader), 5.0)
            assert refusal is not None
            assert refusal[0]["type"] == "error"
            assert "daemon process identity" in str(refusal[0]["message"])

            await write_frame(first_writer, {"type": "heartbeat"}, [])
            async with asyncio.timeout(5.0):
                while True:
                    heartbeat = await read_frame(first_reader)
                    assert heartbeat is not None
                    if heartbeat[0]["type"] == "heartbeatAck":
                        break
        finally:
            if second_writer is not None:
                second_writer.close()
            if first_writer is not None:
                first_writer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_refusal_names_the_holder_and_heartbeats_keep_an_idle_lease(tmp_path: Path) -> None:
    """A second engine's refusal names WHO holds the slot, and an idle
    holder far past the TTL keeps its lease because its heartbeats renew
    it - idleness is not staleness."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, "--lease-ttl", "3")
        try:
            registry = core_registry()
            first = remote_worker(host, port, registry)
            await first.start()
            try:
                # Well past the TTL with no traffic but heartbeats.
                await asyncio.sleep(7.0)
                second = remote_worker(host, port, core_registry())
                with pytest.raises(RuntimeError, match="busy.*leased to engine box1@"):
                    await second.start()
                await second.close()
                # The idle-but-heartbeating holder still owns a working slot.
                engine = make_engine(registry, first)
                result = await engine.run(image_graph(), ["s"])
                assert result.outputs["s"]
            finally:
                await first.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_silent_holder_is_evicted_and_the_next_engine_attaches(tmp_path: Path) -> None:
    """A holder that stops sending anything (no disconnect, no heartbeat -
    a half-open connection) is evicted within roughly the TTL, and the next
    engine attaches and executes without any operator action."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, "--lease-ttl", "3")
        try:
            _reader, zombie_writer, _admission, hello = await zombie_client(host, port)
            assert hello.get("leaseTtl") == 3.0
            # The slot really is held, and the refusal is attributed.
            blocked = remote_worker(host, port, core_registry())
            with pytest.raises(RuntimeError, match="busy.*zombie@nowhere"):
                await blocked.start()
            await blocked.close()
            # The zombie never sends again: the watchdog evicts it and the
            # slot opens without anyone disconnecting. Poll rather than
            # assume the eviction moment.
            third = remote_worker(host, port, core_registry())
            deadline = asyncio.get_running_loop().time() + 20.0
            while True:
                try:
                    await third.start()
                    break
                except RuntimeError as exc:
                    if "busy" not in str(exc):
                        raise
                    if asyncio.get_running_loop().time() > deadline:
                        raise
                    await third.close()
                    await asyncio.sleep(0.2)
                    third = remote_worker(host, port, core_registry())
            try:
                engine = make_engine(core_registry(), third)
                result = await engine.run(image_graph(), ["s"])
                assert result.outputs["s"]
            finally:
                await third.close()
            zombie_writer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX signals")
def test_engine_treats_a_silent_daemon_as_dead(tmp_path: Path) -> None:
    """The lease clock is bilateral: a daemon that stops answering (stopped
    process, not a closed socket) is declared dead by the engine's keepalive
    within the advertised TTL - alive goes False instead of hanging on TCP."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, "--lease-ttl", "3")
        try:
            worker = remote_worker(host, port, core_registry())
            await worker.start()
            try:
                assert worker.alive
                os.kill(proc.pid, _SIGSTOP)
                try:
                    assert await eventually(lambda: not worker.alive, timeout=15.0)
                finally:
                    os.kill(proc.pid, _SIGCONT)
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
