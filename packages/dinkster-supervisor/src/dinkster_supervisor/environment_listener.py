"""Persisted public listener policy for managed project launches."""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .environment_state import EnvironmentError, read_document


def read_response_compression(value: object) -> bool:
    if not isinstance(value, bool):
        raise EnvironmentError("responseCompression must be a boolean")
    return value


@dataclass(frozen=True)
class PublicListener:
    addresses: tuple[str, ...] = ("127.0.0.1",)
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    certificate: str | None = None
    key: str | None = None

    @classmethod
    def read(cls, value: object) -> PublicListener:
        if not isinstance(value, dict):
            raise EnvironmentError("publicListener must be an object")
        fields = cast("dict[str, object]", value)
        if set(fields) - {"addresses", "allowedHosts", "allowedOrigins", "certificate", "key"}:
            raise EnvironmentError("unknown publicListener setting")
        lists: list[tuple[str, ...]] = []
        for name, default in (
            ("addresses", ["127.0.0.1"]),
            ("allowedHosts", []),
            ("allowedOrigins", []),
        ):
            entries = fields.get(name, default)
            if not isinstance(entries, list) or any(
                not isinstance(item, str) or not item for item in cast("list[object]", entries)
            ):
                raise EnvironmentError(
                    f"publicListener {name} must be an array of nonempty strings"
                )
            lists.append(tuple(cast("list[str]", entries)))
        addresses, hosts, origins = lists
        if not addresses or len(set(addresses)) != len(addresses):
            raise EnvironmentError("publicListener addresses must be nonempty and distinct")
        for address in addresses:
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError as exc:
                raise EnvironmentError(
                    f"bind address must be a numeric IPv4 or IPv6 address: {address}"
                ) from exc
            if parsed.is_unspecified or parsed.is_multicast or str(parsed) != address:
                raise EnvironmentError(
                    f"bind address must be a canonical, specific interface address: {address}"
                )
        for host in hosts:
            try:
                valid_host = str(ipaddress.ip_address(host)) == host
            except ValueError:
                valid_host = len(host) <= 253 and all(
                    re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                    for label in host.split(".")
                )
            if not valid_host:
                raise EnvironmentError(
                    "allowed Hosts must be canonical hostnames or IP literals without ports"
                )
        for origin in origins:
            try:
                parsed_origin = urlsplit(origin)
                valid_port = parsed_origin.port
            except ValueError as exc:
                raise EnvironmentError(f"invalid allowed origin: {origin}") from exc
            if (
                parsed_origin.scheme not in {"http", "https"}
                or not parsed_origin.hostname
                or parsed_origin.username is not None
                or parsed_origin.password is not None
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
                or any(character.isspace() for character in origin)
                or valid_port == 0
            ):
                raise EnvironmentError(f"allowed origin must be an exact HTTP(S) origin: {origin}")
        certificate, key = fields.get("certificate"), fields.get("key")
        if (certificate is None) != (key is None):
            raise EnvironmentError("HTTPS requires both certificate and key paths")
        for name, path in (("certificate", certificate), ("key", key)):
            if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
                raise EnvironmentError(f"HTTPS {name} must be an absolute file path")
        return cls(
            addresses, hosts, origins, cast("str | None", certificate), cast("str | None", key)
        )

    @classmethod
    def project(cls, root: Path) -> PublicListener:
        path = root / "environment.json"
        return cls.read(read_document(path).get("publicListener", {})) if path.is_file() else cls()

    def document(self) -> dict[str, object]:
        return {
            "addresses": list(self.addresses),
            "allowedHosts": list(self.allowed_hosts),
            "allowedOrigins": list(self.allowed_origins),
            **({"certificate": self.certificate, "key": self.key} if self.certificate else {}),
        }

    def url(self, port: int) -> str:
        address = self.addresses[0]
        host = f"[{address}]" if ":" in address else address
        return f"{'https' if self.certificate else 'http'}://{host}:{port}"

    def server_context(self) -> ssl.SSLContext | None:
        if self.certificate is None:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(self.certificate, self.key, password="")
        except (OSError, ssl.SSLError) as exc:
            raise EnvironmentError(f"cannot load HTTPS certificate/key: {exc}") from exc
        return context

    def bind(self, port: int) -> list[socket.socket]:
        listeners: list[socket.socket] = []
        for address in self.addresses:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            try:
                listeners.append(socket.create_server((address, port), family=family))
                port = listeners[0].getsockname()[1]
            except OSError as exc:
                for listener in listeners:
                    listener.close()
                raise EnvironmentError(
                    f"cannot bind public listener {address}:{port}: {exc}"
                ) from exc
        return listeners
