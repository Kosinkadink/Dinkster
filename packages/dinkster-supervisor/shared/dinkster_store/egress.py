"""Exact HTTPS origin validation shared by Home and worker egress enforcement."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


class EgressProxyError(ValueError):
    """An egress origin or proxy request is invalid."""


def is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def canonical_host(host: str) -> str:
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
    if not is_public_unicast(address):
        raise EgressProxyError("egress origin cannot name a non-public address")
    return address.compressed


def origin(host: str, port: int) -> str:
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
    return origin(canonical_host(parsed.hostname), port if port is not None else 443)


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
