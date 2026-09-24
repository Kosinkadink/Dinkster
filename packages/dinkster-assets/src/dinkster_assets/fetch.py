"""Digest-verified fetch: bytes move by identity (DESIGN 3.12).

``fetch_asset`` turns a digest plus candidate URLs into a verified local
file in an AssetVault. Every byte streams through the vault's verifying
writer, so a lying mirror, a truncated download, or a tampering middlebox
produces nothing on disk and the next candidate is tried - the digest the
caller already knows is the only authority. stdlib ``urllib`` on purpose:
fetch runs wherever Dinkster proper runs and needs no event loop, so a
worker's synchronous resolve path can use it directly.

``ChainResolver`` composes resolvers lookup-order-first (library, vault,
then fetching), and ``FetchingResolver`` is the last link: on miss it asks
a sources callback (a ProvenanceStore, a peer's /assets endpoint, or both)
where the bytes might live and pulls them into the vault. After one
success the vault itself answers - each identity crosses the network at
most once per machine.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from http.client import HTTPMessage
from pathlib import Path
from typing import Protocol, cast

from dinkster_values import MEBIBYTE

from .identity import AssetError, require_digest
from .model import AssetResolution, AssetResolver, RecordedAssetResolver
from .vault import AssetVault

FETCH_CHUNK_SIZE = 8 * MEBIBYTE
DEFAULT_FETCH_TIMEOUT = 60.0
MAX_PUBLIC_REDIRECTS = 10

SourcesFor = Callable[[str], Sequence[str]]
"""digest -> candidate URLs, best-first. Empty means unknown identity."""


class FetchAborted(Exception):
    """A caller-requested abort stopped an in-progress fetch.

    Deliberately NOT an AssetError: fetch_asset swallows AssetError per
    candidate URL (a failed lead), while an abort must stop the whole
    attempt and unwind the in-progress writer (rollback to nothing)."""


class PublicFetchDenied(AssetError):
    """A transfer cannot prove a credential-free globally routed HTTPS chain."""


_PUBLIC_HTTPS_EVIDENCE = object()


@dataclass(frozen=True)
class _PublicHTTPSFetchEvidence:
    token: object
    path: Path
    digest: str
    listed_url: str
    final_url: str
    size_bytes: int


@dataclass(frozen=True)
class FetchResult:
    """Verified bytes plus the winning HTTP chain endpoints."""

    path: Path
    digest: str
    listed_url: str = ""
    final_url: str = ""
    size_bytes: int = 0
    _public_https_evidence: _PublicHTTPSFetchEvidence | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def public_https_verified(self) -> bool:
        evidence = self._public_https_evidence
        return evidence is not None and (
            evidence.token is _PUBLIC_HTTPS_EVIDENCE
            and evidence.path == self.path
            and evidence.digest == self.digest
            and evidence.listed_url == self.listed_url
            and evidence.final_url == self.final_url
            and evidence.size_bytes == self.size_bytes
        )


class HTTPSResponse(Protocol):
    status: int
    headers: HTTPMessage

    def read(self, amount: int | None = None) -> bytes: ...


def _globally_routable_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PublicFetchDenied(f"public HTTPS DNS lookup failed for {host!r}") from exc
    addresses = tuple(dict.fromkeys(row[4][0] for row in rows if isinstance(row[4][0], str)))
    if not addresses:
        raise PublicFetchDenied(f"public HTTPS DNS lookup returned no addresses for {host!r}")
    try:
        parsed = tuple(ipaddress.ip_address(address.split("%", 1)[0]) for address in addresses)
    except ValueError as exc:
        raise PublicFetchDenied(
            f"public HTTPS DNS returned an invalid address for {host!r}"
        ) from exc
    if any(
        "%" in address or not address_ip.is_global
        for address, address_ip in zip(addresses, parsed, strict=True)
    ):
        raise PublicFetchDenied(f"public HTTPS DNS returned a non-global address for {host!r}")
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
        super().__init__(
            host,
            port,
            timeout=timeout,
            context=context,
        )
        self._address = address
        self._ssl_context = context
        self.peer_verified = False

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        try:
            wrapped = self._ssl_context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise
        try:
            peer = ipaddress.ip_address(wrapped.getpeername()[0].split("%", 1)[0])
            expected = ipaddress.ip_address(self._address.split("%", 1)[0])
            if peer != expected or not peer.is_global:
                raise PublicFetchDenied("public HTTPS connection reached an unapproved peer")
        except BaseException:
            wrapped.close()
            raise
        self.sock = wrapped
        self.peer_verified = True


@contextlib.contextmanager
def _open_pinned_https(
    url: str,
    addresses: Sequence[str],
    headers: Mapping[str, str],
    timeout: float,
) -> Generator[HTTPSResponse]:
    parsed = urllib.parse.urlsplit(url)
    assert parsed.hostname is not None
    port = parsed.port or 443
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    failures: list[BaseException] = []
    for address in addresses:
        connection = _PinnedHTTPSConnection(parsed.hostname, port, address, timeout)
        try:
            connection.request("GET", path, headers=dict(headers))
            response = connection.getresponse()
            if not connection.peer_verified:
                raise PublicFetchDenied("public HTTPS connection reached an unapproved peer")
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            failures.append(exc)
            connection.close()
            continue
        except BaseException:
            connection.close()
            raise
        try:
            yield cast("HTTPSResponse", response)
        finally:
            response.close()
            connection.close()
        return
    if failures:
        raise PublicFetchDenied("public HTTPS connection failed") from failures[-1]
    raise PublicFetchDenied("public HTTPS connection has no approved destination")


def _public_https_url(url: str) -> urllib.parse.SplitResult:
    if (
        not url
        or url != url.strip()
        or any(character.isspace() or ord(character) < 32 for character in url)
    ):
        raise PublicFetchDenied("public HTTPS URL must be trimmed and contain no whitespace")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise PublicFetchDenied("public HTTPS URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise PublicFetchDenied(
            "public acquisition requires HTTPS without userinfo, query, or fragment"
        )
    return parsed


def _credential_free_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forbidden = {"authorization", "cookie", "proxy-authorization"}
    if any(name.casefold() in forbidden for name in headers):
        raise PublicFetchDenied("public acquisition request contains credentials or cookies")
    return dict(headers)


def _content_length(headers: HTTPMessage) -> int | None:
    values = headers.get_all("Content-Length", [])
    if not values:
        return None
    if len(values) != 1:
        raise PublicFetchDenied("public HTTPS response has ambiguous Content-Length headers")
    raw = values[0]
    try:
        size = int(raw)
    except ValueError as exc:
        raise PublicFetchDenied("public HTTPS response has an invalid Content-Length") from exc
    if size < 0:
        raise PublicFetchDenied("public HTTPS response has an invalid Content-Length")
    return size


def fetch_public_asset(
    digest: str,
    listed_url: str,
    expected_size: int,
    vault: AssetVault,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    headers: Mapping[str, str] | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> FetchResult:
    """Fetch through a pinned, credential-free public HTTPS redirect chain."""
    require_digest(digest)
    if type(expected_size) is not int or expected_size < 0:
        raise AssetError("public acquisition requires a non-negative expected size")
    if headers:
        _credential_free_headers(headers)
        raise PublicFetchDenied("public acquisition does not permit custom request headers")
    request_headers = _credential_free_headers({"User-Agent": "dinkster-public-fetch"})
    current = listed_url
    seen: set[str] = set()
    for _redirect in range(MAX_PUBLIC_REDIRECTS + 1):
        if should_abort is not None and should_abort():
            raise FetchAborted(f"fetch of {digest} aborted")
        parsed = _public_https_url(current)
        if current in seen:
            raise PublicFetchDenied("public HTTPS redirect loop")
        seen.add(current)
        addresses = _globally_routable_addresses(parsed.hostname or "", parsed.port or 443)
        if not addresses:
            raise PublicFetchDenied("public HTTPS DNS returned no addresses")
        try:
            parsed_addresses = tuple(
                ipaddress.ip_address(address.split("%", 1)[0]) for address in addresses
            )
        except ValueError as exc:
            raise PublicFetchDenied("public HTTPS DNS returned an invalid address") from exc
        if any(
            "%" in address or not address_ip.is_global
            for address, address_ip in zip(addresses, parsed_addresses, strict=True)
        ):
            raise PublicFetchDenied("public HTTPS DNS returned a non-global address")
        with _open_pinned_https(current, addresses, request_headers, timeout) as response:
            if any(
                response.headers.get(name) is not None for name in ("Set-Cookie", "Set-Cookie2")
            ):
                raise PublicFetchDenied("public HTTPS chain attempted to set a cookie")
            if response.status in (301, 302, 303, 307, 308):
                locations = response.headers.get_all("Location", [])
                if len(locations) != 1 or not locations[0]:
                    raise PublicFetchDenied("public HTTPS redirect has no Location")
                current = urllib.parse.urljoin(current, locations[0])
                _public_https_url(current)
                continue
            if not 200 <= response.status < 300:
                raise PublicFetchDenied(f"public HTTPS response returned status {response.status}")
            declared_size = _content_length(response.headers)
            if declared_size is not None and declared_size != expected_size:
                raise PublicFetchDenied("public HTTPS response size does not match its declaration")
            with vault.writer(digest) as writer:
                received = 0
                while chunk := response.read(FETCH_CHUNK_SIZE):
                    if should_abort is not None and should_abort():
                        raise FetchAborted(f"fetch of {digest} aborted")
                    received += len(chunk)
                    if received > expected_size:
                        raise PublicFetchDenied("public HTTPS response exceeds its declared size")
                    writer.write(chunk)
                    if on_progress is not None:
                        on_progress(received)
                if received != expected_size:
                    raise PublicFetchDenied(
                        "public HTTPS response size does not match its declaration"
                    )
                path = writer.commit()
            return FetchResult(
                path,
                digest,
                listed_url,
                current,
                received,
                _public_https_evidence=_PublicHTTPSFetchEvidence(
                    _PUBLIC_HTTPS_EVIDENCE,
                    path,
                    digest,
                    listed_url,
                    current,
                    received,
                ),
            )
    raise PublicFetchDenied("public HTTPS redirect limit exceeded")


def fetch_asset(
    digest: str,
    urls: Sequence[str],
    vault: AssetVault,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    headers_for: Callable[[str], Mapping[str, str]] | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path | None:
    """Fetch one identity from the first URL that yields verifying bytes.

    Returns the vault path, or None if every candidate failed (network,
    HTTP error, digest mismatch) - failures are leads that did not pan
    out, never exceptions, matching the conservative posture of every
    other distribution surface. Already-held identities return without
    touching the network.

    ``headers_for`` supplies extra request headers per URL (an engine
    peer endpoint's bearer credential; never logged, never in the URL).
    ``should_abort`` is polled between chunks and before each candidate;
    when it trips, the in-progress writer rolls back and FetchAborted
    escapes - the one non-failure exit. ``on_progress`` observes the
    running byte count of the current candidate."""
    require_digest(digest)
    held = vault.resolve(digest)
    if held is not None:
        return held
    for url in urls:
        if should_abort is not None and should_abort():
            raise FetchAborted(f"fetch of {digest} aborted")
        if not url.startswith(("http://", "https://")):
            continue  # a file:// or ftp:// lead is not a fetchable source
        try:
            headers = {"User-Agent": "dinkster-fetch"}
            if headers_for is not None:
                headers.update(headers_for(url))
            request = urllib.request.Request(url, headers=headers)
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310 - scheme checked above
                vault.writer(digest) as writer,
            ):
                received = 0
                while chunk := response.read(FETCH_CHUNK_SIZE):
                    if should_abort is not None and should_abort():
                        raise FetchAborted(f"fetch of {digest} aborted")
                    writer.write(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received)
                return writer.commit()
        except (
            OSError,
            urllib.error.URLError,
            # A transfer the peer cut short (http.client.IncompleteRead)
            # is a lead that did not pan out, like every other failure.
            http.client.HTTPException,
            AssetError,
            TimeoutError,
        ):
            continue
    return None


class ChainResolver:
    """AssetResolver over resolvers tried in order - first hit wins.
    Compose fastest/cheapest first: model library, vault, then anything
    that touches the network."""

    def __init__(self, *resolvers: AssetResolver) -> None:
        if not resolvers:
            raise AssetError("ChainResolver requires at least one resolver")
        self._resolvers = resolvers

    def resolve(self, digest: str) -> Path | None:
        for resolver in self._resolvers:
            with contextlib.suppress(AssetError):
                path = resolver.resolve(digest)
                if path is not None:
                    return path
        return None

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        for resolver in self._resolvers:
            with contextlib.suppress(AssetError):
                if isinstance(resolver, RecordedAssetResolver):
                    resolution = resolver.resolve_asset(digest)
                    if resolution is not None:
                        return resolution
                else:
                    path = resolver.resolve(digest)
                    if path is not None:
                        return AssetResolution(path)
        return None


class FetchingResolver:
    """The chain's last link: a miss becomes a provenance-guided fetch.

    ``sources_for`` supplies candidate URLs per digest (a ProvenanceStore's
    ``sources``, a peer endpoint's ``/assets/{digest}``, or a merge). The
    resolve is synchronous and can block for the duration of a download -
    by design, since materialization happens where a node actually reads
    bytes; place this resolver behind local ones so it only runs on true
    misses."""

    def __init__(
        self,
        vault: AssetVault,
        sources_for: SourcesFor,
        *,
        timeout: float = DEFAULT_FETCH_TIMEOUT,
    ) -> None:
        self._vault = vault
        self._sources_for = sources_for
        self._timeout = timeout

    def resolve(self, digest: str) -> Path | None:
        return fetch_asset(digest, self._sources_for(digest), self._vault, timeout=self._timeout)

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        path = self.resolve(digest)
        return None if path is None else self._vault.resolve_asset(digest)
