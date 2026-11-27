"""Namespace-local HTTP proxy bridge; the host retains all egress authority."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress

from .egress import EGRESS_PROXY_ENV, MAX_PROXY_CLIENTS, relay_streams


async def _run(command: list[str]) -> int:
    if sys.platform != "linux":
        raise RuntimeError("the namespace proxy bridge requires Linux")
    endpoint = os.environ[EGRESS_PROXY_ENV]
    clients: set[asyncio.Task[None]] = set()

    async def bridge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is None or len(clients) >= MAX_PROXY_CLIENTS:
            writer.close()
            return
        clients.add(task)
        upstream: asyncio.StreamWriter | None = None
        try:
            remote, upstream = await asyncio.open_unix_connection(endpoint)
            await relay_streams(reader, writer, remote, upstream)
        except OSError:
            pass
        finally:
            for stream in (upstream, writer):
                if stream is not None:
                    stream.close()
                    with suppress(OSError):
                        await stream.wait_closed()
            clients.discard(task)

    # This loopback is inside the unshared network namespace, not host loopback.
    server = await asyncio.start_server(bridge, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
            environment.pop(name)
    environment.update(https_proxy=f"http://127.0.0.1:{port}", no_proxy="")
    try:
        async with server:
            process = await asyncio.create_subprocess_exec(*command, env=environment)
            return await process.wait()
    finally:
        server.close()
        await server.wait_closed()
        pending = tuple(clients)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__" and sys.platform == "linux":
    status = asyncio.run(_run(sys.argv[1:]))
    if status < 0:
        if -status != signal.SIGKILL:
            signal.signal(-status, signal.SIG_DFL)
        os.kill(os.getpid(), -status)
        signal.pause()
    sys.exit(status)
