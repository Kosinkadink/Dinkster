"""Portable boundary endpoints: how parent and child processes find each other.

Framing (boundary.py) is transport-neutral bytes-in-order; this module owns
endpoint creation, which is the only platform-specific part of the boundary:

- ``unix``: a socket file inside the parent's private tmpdir. First choice
  wherever asyncio supports unix servers (Linux, macOS). Access control is
  the directory mode - nothing else on the machine can connect.
- ``tcp``: loopback TCP with a one-time secret. The fallback for Windows,
  where asyncio's proactor loop has no unix-socket support. Loopback TCP is
  reachable by every local user, so the child must present the secret as its
  first bytes before any frames flow. The secret travels through the child's
  environment (never argv, which is world-readable process metadata).

Endpoints are serialized as ``unix:<path>`` or ``tcp:<host>:<port>`` so the
child can be told where to connect with a single string.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import secrets
import socket
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

TOKEN_ENV = "DINKSTER_BOUNDARY_TOKEN"
_TOKEN_BYTES = 32
_TOKEN_HEX_LEN = _TOKEN_BYTES * 2

TransportChoice = Literal["auto", "unix", "tcp"]


class TransportError(Exception):
    """An endpoint could not be created, parsed, or connected."""


def unix_endpoints_supported() -> bool:
    """Whether this platform can host unix-socket boundary endpoints.

    asyncio only implements unix servers on the selector loops, which
    Windows' default proactor loop is not - so Windows always gets tcp,
    even on builds where the OS itself has AF_UNIX.
    """
    return sys.platform != "win32" and hasattr(socket, "AF_UNIX")


class BoundaryListener:
    """The parent side of a boundary endpoint.

    Create it, launch the child with ``endpoint`` (argv-safe) and
    ``child_env`` (secrets), then await ``connected`` for the first
    authenticated (reader, writer) pair. Unauthenticated tcp connections
    are dropped without resolving the future.
    """

    def __init__(
        self,
        server: asyncio.Server,
        endpoint: str,
        child_env: Mapping[str, str],
        connected: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
    ) -> None:
        self._server = server
        self.endpoint = endpoint
        self.child_env = dict(child_env)
        self.connected = connected

    @classmethod
    async def create(
        cls,
        tmpdir: Path,
        *,
        transport: TransportChoice = "auto",
        token: str | None = None,
    ) -> BoundaryListener:
        if transport == "auto":
            transport = "unix" if unix_endpoints_supported() else "tcp"
        if transport == "unix" and not unix_endpoints_supported():
            raise TransportError("unix endpoints are not supported on this platform")
        loop = asyncio.get_running_loop()
        connected: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
            loop.create_future()
        )
        if transport == "unix":
            if sys.platform == "win32":
                # Statically unreachable when analyzed for Windows; asyncio's
                # unix-server APIs do not exist in Windows type stubs.
                raise TransportError("unix endpoints are not supported on this platform")

            def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                if connected.done():
                    writer.close()
                else:
                    connected.set_result((reader, writer))

            socket_path = str(tmpdir / "worker.sock")
            server = await asyncio.start_unix_server(on_connect, path=socket_path)
            return cls(server, f"unix:{socket_path}", {}, connected)

        token = token or secrets.token_hex(_TOKEN_BYTES)

        async def on_tcp_connect(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                presented = await asyncio.wait_for(reader.readexactly(_TOKEN_HEX_LEN), timeout=10.0)
            except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
                writer.close()
                return
            if connected.done() or not hmac.compare_digest(presented, token.encode("ascii")):
                writer.close()
                return
            connected.set_result((reader, writer))

        server = await asyncio.start_server(on_tcp_connect, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        return cls(server, f"tcp:127.0.0.1:{port}", {TOKEN_ENV: token}, connected)

    async def close(self) -> None:
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception:  # noqa: BLE001 - closing must not mask the original error
            pass


async def connect_endpoint(
    endpoint: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """The child side: connect to a serialized endpoint and authenticate.

    For tcp endpoints the secret is read from ``DINKSTER_BOUNDARY_TOKEN`` and
    sent before anything else; the parent reads exactly that many bytes, so
    frames start cleanly after it.
    """
    kind, _, rest = endpoint.partition(":")
    if kind == "unix":
        if not rest:
            raise TransportError(f"malformed unix endpoint: {endpoint!r}")
        if sys.platform == "win32":
            raise TransportError("unix endpoints are not supported on this platform")
        return await asyncio.open_unix_connection(rest)
    if kind == "tcp":
        host, _, port_text = rest.rpartition(":")
        if not host or not port_text.isdigit():
            raise TransportError(f"malformed tcp endpoint: {endpoint!r}")
        token = os.environ.get(TOKEN_ENV, "")
        if len(token) != _TOKEN_HEX_LEN:
            raise TransportError(
                f"tcp endpoint requires a {_TOKEN_HEX_LEN}-char token in ${TOKEN_ENV}"
            )
        reader, writer = await asyncio.open_connection(host, int(port_text))
        writer.write(token.encode("ascii"))
        await writer.drain()
        return reader, writer
    raise TransportError(f"unknown endpoint kind: {endpoint!r}")
