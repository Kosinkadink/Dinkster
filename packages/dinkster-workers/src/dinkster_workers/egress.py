"""Per-worker HTTPS egress proxy for network-isolated pack sandboxes."""

from __future__ import annotations

import asyncio
import base64
import hmac
import ipaddress
import json
import os
import socket
import stat
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from dinkster_store.egress import (
    EgressProxyError as EgressProxyError,
)
from dinkster_store.egress import (
    canonical_host as _canonical_host,
)
from dinkster_store.egress import (
    egress_origin_from_url as egress_origin_from_url,
)
from dinkster_store.egress import (
    is_public_unicast as _is_public_unicast,
)
from dinkster_store.egress import (
    normalize_egress_origin as normalize_egress_origin,
)
from dinkster_store.egress import (
    origin as _origin,
)

EGRESS_PROXY_ENV = "DINKSTER_EGRESS_PROXY"

_MAX_HEADER_BYTES = 16 * 1024
MAX_PROXY_CLIENTS = 64
_MAX_RESOLVE_BODY_BYTES = 64 * 1024
_REQUEST_TIMEOUT_SECONDS = 10.0
_UPSTREAM_TIMEOUT_SECONDS = 10.0


class _EgressDenied(EgressProxyError):
    pass


class _ProxyAuthenticationRequired(EgressProxyError):
    pass


def _parse_authority(value: str) -> tuple[str, int, str]:
    if not value or any(char in value for char in "/?#@"):
        raise EgressProxyError("invalid proxy authority")
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError as exc:
        raise EgressProxyError("invalid proxy authority") from exc
    if not parsed.hostname or port is None or not 1 <= port <= 65535:
        raise EgressProxyError("proxy authority requires an explicit valid port")
    host = _canonical_host(parsed.hostname)
    return host, port, _origin(host, port)


async def _resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        async with asyncio.timeout(_UPSTREAM_TIMEOUT_SECONDS):
            infos = await asyncio.get_running_loop().getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
    except (OSError, TimeoutError) as exc:
        raise EgressProxyError("destination DNS resolution failed") from exc
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise EgressProxyError("destination DNS returned no addresses")
    parsed: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise EgressProxyError("destination DNS returned an invalid address") from exc
        if not _is_public_unicast(address):
            raise _EgressDenied("destination resolved to a non-public address")
        parsed.append(address)
    return tuple(address.compressed for address in parsed)


async def _connect(
    addresses: tuple[str, ...], port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    failure: OSError | None = None
    try:
        async with asyncio.timeout(_UPSTREAM_TIMEOUT_SECONDS):
            for value in addresses:
                address = ipaddress.ip_address(value)
                family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
                try:
                    return await asyncio.open_connection(value, port, family=family)
                except OSError as exc:
                    failure = exc
    except TimeoutError as exc:
        raise EgressProxyError("destination connection timed out") from exc
    raise EgressProxyError("destination connection failed") from failure


async def _write_response(
    writer: asyncio.StreamWriter,
    status_code: int,
    reason: str,
    body: bytes = b"",
    *,
    content_type: str = "text/plain",
) -> None:
    headers = [
        f"HTTP/1.1 {status_code} {reason}\r\n",
        f"Content-Length: {len(body)}\r\n",
        "Connection: close\r\n",
    ]
    if body:
        headers.append(f"Content-Type: {content_type}\r\n")
    writer.write("".join(headers).encode("ascii") + b"\r\n" + body)
    await writer.drain()


async def _try_write_response(
    writer: asyncio.StreamWriter,
    status_code: int,
    reason: str,
    body: bytes = b"",
    *,
    content_type: str = "text/plain",
) -> None:
    with suppress(OSError):
        await _write_response(
            writer,
            status_code,
            reason,
            body,
            content_type=content_type,
        )


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(64 * 1024):
        writer.write(data)
        await writer.drain()


async def relay_streams(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    remote_reader: asyncio.StreamReader,
    remote_writer: asyncio.StreamWriter,
) -> None:
    tasks = {
        asyncio.create_task(_pump(client_reader, remote_writer)),
        asyncio.create_task(_pump(remote_reader, client_writer)),
    }
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class EgressProxy:
    """An HTTP CONNECT proxy scoped to one worker's HTTPS origin allowlist."""

    def __init__(
        self,
        socket_path: Path | None,
        origins: tuple[str, ...],
        on_denied: Callable[[str], None] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self.origins = frozenset(normalize_egress_origin(origin) for origin in origins)
        self._on_denied = on_denied
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.Task[None]] = set()
        self._closing = False
        self.url: str | None = None
        self._authorization: str | None = None

    @property
    def socket_path(self) -> Path:
        if self._socket_path is None:
            raise EgressProxyError("TCP proxy has no Unix socket path")
        return self._socket_path

    @classmethod
    async def start_tcp(
        cls,
        origins: tuple[str, ...],
        token: str,
        *,
        on_denied: Callable[[str], None] | None = None,
    ) -> EgressProxy:
        if len(token) < 32 or not token.isalnum() or not token.isascii():
            raise EgressProxyError("TCP egress proxy requires a private random token")
        proxy = cls(None, origins, on_denied)
        proxy._authorization = "Basic " + base64.b64encode(f"dinkster:{token}".encode()).decode()
        proxy._server = await asyncio.start_server(
            proxy._handle, host="127.0.0.1", port=0, limit=_MAX_HEADER_BYTES
        )
        port = proxy._server.sockets[0].getsockname()[1]
        proxy.url = f"http://dinkster:{token}@127.0.0.1:{port}"
        return proxy

    @classmethod
    async def start(
        cls,
        socket_path: Path,
        origins: tuple[str, ...],
        *,
        on_denied: Callable[[str], None] | None = None,
    ) -> EgressProxy:
        if sys.platform == "win32":
            raise EgressProxyError("egress proxy requires Unix sockets")
        proxy = cls(socket_path, origins, on_denied)
        try:
            metadata = socket_path.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(metadata.st_mode):
                raise EgressProxyError(f"egress socket path already exists: {socket_path}")
            socket_path.unlink()
        proxy._server = await asyncio.start_unix_server(
            proxy._handle,
            path=socket_path,
            limit=_MAX_HEADER_BYTES,
        )
        try:
            os.chmod(socket_path, 0o600)
        except BaseException:
            proxy._closing = True
            proxy._server.close()
            await proxy._server.wait_closed()
            proxy._server = None
            with suppress(FileNotFoundError):
                socket_path.unlink()
            raise
        return proxy

    async def _request_head(self, reader: asyncio.StreamReader) -> tuple[str, str, str]:
        try:
            async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
                raw = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError) as exc:
            raise EgressProxyError("invalid proxy request headers") from exc
        try:
            lines = raw.decode("ascii").split("\r\n")
            method, target, version = lines[0].split(" ")
        except (UnicodeDecodeError, ValueError) as exc:
            raise EgressProxyError("invalid proxy request line") from exc
        if version not in ("HTTP/1.0", "HTTP/1.1"):
            raise EgressProxyError("unsupported proxy HTTP version")
        authorization: list[str] = []
        for line in lines[1:]:
            if not line:
                continue
            if ":" not in line:
                raise EgressProxyError("invalid proxy request header")
            name, value = line.split(":", 1)
            if name.lower() == "proxy-authorization":
                authorization.append(value.strip())
        if self._authorization is not None and (
            len(authorization) != 1
            or not hmac.compare_digest(authorization[0], self._authorization)
        ):
            raise _ProxyAuthenticationRequired("proxy authentication required")
        return method, target, version

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if self._closing or task is None or len(self._clients) >= MAX_PROXY_CLIENTS:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return
        self._clients.add(task)
        remote_writer: asyncio.StreamWriter | None = None
        origin: str | None = None
        try:
            method, target, _version = await self._request_head(reader)
            host, port, origin = _parse_authority(target)
            if origin not in self.origins:
                raise _EgressDenied("destination origin is not allowed")
            addresses = await _resolve_public_addresses(host, port)
            if method == "RESOLVE":
                body = json.dumps({"addresses": addresses}, separators=(",", ":")).encode("utf-8")
                if len(body) > _MAX_RESOLVE_BODY_BYTES:
                    raise EgressProxyError("resolver response is too large")
                await _try_write_response(
                    writer,
                    200,
                    "OK",
                    body,
                    content_type="application/json",
                )
            elif method == "CONNECT":
                remote_reader, remote_writer = await _connect(addresses, port)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                await relay_streams(reader, writer, remote_reader, remote_writer)
            else:
                await _try_write_response(writer, 405, "Method Not Allowed")
        except _ProxyAuthenticationRequired:
            await _try_write_response(writer, 407, "Proxy Authentication Required")
        except _EgressDenied:
            if origin is not None and self._on_denied is not None:
                self._on_denied(origin)
            await _try_write_response(writer, 403, "Forbidden")
        except (EgressProxyError, OSError):
            await _try_write_response(writer, 502, "Bad Gateway")
        finally:
            if remote_writer is not None:
                remote_writer.close()
                with suppress(Exception):
                    await remote_writer.wait_closed()
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            self._clients.discard(task)

    async def close(self) -> None:
        self._closing = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        clients = tuple(self._clients)
        for task in clients:
            task.cancel()
        await asyncio.gather(*clients, return_exceptions=True)
        if self.url is None:
            with suppress(FileNotFoundError):
                self.socket_path.unlink()
