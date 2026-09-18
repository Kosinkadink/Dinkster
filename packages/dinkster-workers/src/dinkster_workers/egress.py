"""Per-worker HTTPS egress proxy for network-isolated pack sandboxes."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import stat
import sys
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

EGRESS_PROXY_ENV = "DINKSTER_EGRESS_PROXY"

_MAX_HEADER_BYTES = 16 * 1024
_MAX_CLIENTS = 64
_MAX_RESOLVE_BODY_BYTES = 64 * 1024
_REQUEST_TIMEOUT_SECONDS = 10.0
_UPSTREAM_TIMEOUT_SECONDS = 10.0


class EgressProxyError(ValueError):
    """An egress origin or proxy request is invalid."""


class _EgressDenied(EgressProxyError):
    pass


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def _canonical_host(host: str) -> str:
    host = host.rstrip(".")
    if not host or "%" in host:
        raise EgressProxyError("egress origin requires a valid host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            canonical = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise EgressProxyError("egress origin requires a valid host") from exc
        labels = canonical.split(".")
        if len(canonical) > 253 or any(
            not label
            or len(label) > 63
            or label[0] == "-"
            or label[-1] == "-"
            or any(not (char.isascii() and (char.isalnum() or char == "-")) for char in label)
            for label in labels
        ):
            raise EgressProxyError("egress origin requires a valid host") from None
        return canonical
    if not _is_public_unicast(address):
        raise EgressProxyError("egress origin cannot name a non-public address")
    return address.compressed


def _origin(host: str, port: int) -> str:
    rendered = f"[{host}]" if ":" in host else host
    return f"https://{rendered}" if port == 443 else f"https://{rendered}:{port}"


def egress_origin_from_url(value: str) -> str:
    """Return the canonical HTTPS origin named by a URL."""
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise EgressProxyError(f"invalid egress origin {value!r}") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise EgressProxyError("egress origin must use https:// and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise EgressProxyError("egress origin cannot include credentials")
    if parsed.netloc.rsplit("@", 1)[-1].endswith(":"):
        raise EgressProxyError("egress origin requires a valid port")
    if port is not None and not 1 <= port <= 65535:
        raise EgressProxyError("egress origin requires a valid port")
    return _origin(_canonical_host(parsed.hostname), port if port is not None else 443)


def normalize_egress_origin(value: str) -> str:
    """Return one canonical HTTPS origin suitable for exact allowlisting."""
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:
        raise EgressProxyError(f"invalid egress origin {value!r}") from exc
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise EgressProxyError("egress origin cannot include a path, query, or fragment")
    return egress_origin_from_url(candidate)


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


async def _relay(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    remote_reader: asyncio.StreamReader,
    remote_writer: asyncio.StreamWriter,
) -> None:
    tasks = {
        asyncio.create_task(_pump(client_reader, remote_writer)),
        asyncio.create_task(_pump(remote_reader, client_writer)),
    }
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class EgressProxy:
    """A Unix-socket HTTP CONNECT proxy scoped to one worker allowlist."""

    def __init__(self, socket_path: Path, origins: tuple[str, ...]) -> None:
        self.socket_path = socket_path
        self.origins = frozenset(normalize_egress_origin(origin) for origin in origins)
        if not self.origins:
            raise EgressProxyError("egress proxy requires at least one allowed origin")
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.Task[None]] = set()
        self._closing = False

    @classmethod
    async def start(cls, socket_path: Path, origins: tuple[str, ...]) -> EgressProxy:
        if sys.platform == "win32":
            raise EgressProxyError("egress proxy requires Unix sockets")
        proxy = cls(socket_path, origins)
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
        for line in lines[1:]:
            if line and ":" not in line:
                raise EgressProxyError("invalid proxy request header")
        return method, target, version

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if self._closing or task is None or len(self._clients) >= _MAX_CLIENTS:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return
        self._clients.add(task)
        remote_writer: asyncio.StreamWriter | None = None
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
                await _relay(reader, writer, remote_reader, remote_writer)
            else:
                await _try_write_response(writer, 405, "Method Not Allowed")
        except _EgressDenied:
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
        with suppress(FileNotFoundError):
            self.socket_path.unlink()
