"""Inherited binary pipes, multiplexed for worker groups; pack output uses stderr."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import struct
import sys
import threading
from collections.abc import Mapping, Sequence

from ._windows_loop import unix_socketpair
from .transport import TOKEN_ENV, TransportError

_HEADER = struct.Struct("!II")
_CHUNK = 64 * 1024
_child_sockets: dict[int, socket.socket] = {}


def take_pipe_socket(index: int) -> socket.socket:
    try:
        return _child_sockets.pop(index)
    except KeyError as exc:
        raise TransportError(f"pipe endpoint {index} is unavailable or already connected") from exc


def prepare_child_pipes(count: int) -> None:
    """Reserve stdio before logging or importing a pack, and start bounded pumps."""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(0, os.O_BINARY)
        msvcrt.setmode(1, os.O_BINARY)
    source, destination = os.dup(0), os.dup(1)
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
    finally:
        os.close(null)
    peers: list[socket.socket] = []
    try:
        for index in range(count):
            near, far = unix_socketpair()
            _child_sockets[index] = near
            peers.append(far)
    except BaseException:
        for sock in (*_child_sockets.values(), *peers):
            sock.close()
        _child_sockets.clear()
        os.close(source)
        os.close(destination)
        raise
    lock = threading.Lock()

    def send(index: int, sock: socket.socket) -> None:
        try:
            while True:
                chunk = sock.recv(_CHUNK)
                frame = memoryview(_HEADER.pack(index, len(chunk)) + chunk)
                with lock:
                    while frame:
                        written = os.write(destination, frame)
                        if written == 0:
                            raise BrokenPipeError("worker pipe closed")
                        frame = frame[written:]
                if not chunk:
                    break
        except OSError:
            # Closing the host's pipe disconnects every member, not just this one.
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def read_exactly(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = os.read(source, size - len(data))
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    def receive() -> None:
        try:
            while True:
                index, size = _HEADER.unpack(read_exactly(_HEADER.size))
                if index >= count or size > _CHUNK:
                    raise ValueError("invalid worker pipe frame")
                if size:
                    peers[index].sendall(read_exactly(size))
                else:
                    peers[index].shutdown(socket.SHUT_WR)
        except (EOFError, OSError, ValueError):
            for sock in peers:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
        finally:
            os.close(source)

    for index, sock in enumerate(peers):
        threading.Thread(target=send, args=(index, sock), daemon=True).start()
    threading.Thread(target=receive, daemon=True).start()


async def bridge_process(
    process: asyncio.subprocess.Process,
    endpoints: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    """Keep engine listeners host-side; no TCP connection enters the container."""
    assert process.stdin is not None and process.stdout is not None
    outgoing = process.stdin
    incoming = process.stdout
    writers: list[asyncio.StreamWriter] = []
    tasks: list[asyncio.Task[None]] = []

    async def send(index: int, reader: asyncio.StreamReader) -> None:
        while True:
            chunk = await reader.read(_CHUNK)
            outgoing.write(_HEADER.pack(index, len(chunk)) + chunk)
            await outgoing.drain()
            if not chunk:
                return

    async def receive() -> None:
        while True:
            index, size = _HEADER.unpack(await incoming.readexactly(_HEADER.size))
            if index >= len(writers) or size > _CHUNK:
                raise TransportError("invalid contained worker pipe frame")
            writer = writers[index]
            if size:
                writer.write(await incoming.readexactly(size))
                await writer.drain()
            else:
                writer.close()

    try:
        for index, endpoint in enumerate(endpoints):
            kind, _, address = endpoint.partition(":")
            if kind == "tcp":
                host, _, port = address.rpartition(":")
                reader, writer = await asyncio.open_connection(host, int(port))
                writer.write(environment[TOKEN_ENV].encode("ascii"))
                await writer.drain()
            elif kind == "unix" and sys.platform != "win32":
                reader, writer = await asyncio.open_unix_connection(address)
            else:
                raise TransportError(f"cannot bridge endpoint {endpoint!r}")
            writers.append(writer)
            tasks.append(asyncio.create_task(send(index, reader)))
        tasks.append(asyncio.create_task(receive()))
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        outgoing.close()
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
