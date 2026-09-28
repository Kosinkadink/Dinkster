"""Persistent subscriptions to local and hosted resolver indexes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from http.client import HTTPConnection, HTTPException, HTTPMessage, HTTPResponse, HTTPSConnection
from pathlib import Path
from typing import IO, cast
from urllib.parse import quote, unquote_plus, urlsplit, urlunsplit

from .component_manifest import AssetComponentManifest
from .identity import AssetError
from .json_metadata import thaw_json
from .p2p_global import (
    ProviderArtifactP2PV1,
    ProviderLocationV1,
    ProviderP2PEnumerationV1,
    ProviderP2PSnapshotV1,
    ProviderP2PTombstoneV1,
    provider_declarations,
)
from .p2p_grants import P2P_REMOTE_GRANT_MAX_SECONDS, PublicSwarmDeclarationV1
from .provenance import ProvenanceRecord, ProvenanceStore
from .public_acquisition import PublicAcquisitionSourceV1
from .resolver_index import (
    RESOLVER_INDEX_MAX_BYTES,
    RESOLVER_INDEX_MAX_ENTRIES,
    RESOLVER_INDEX_MAX_STRING,
    ResolverIndex,
    ResolverIndexError,
    decode_resolver_index_document,
    parse_resolver_index,
    require_region,
    resolver_index_from_wire,
)

RESOLVER_REVALIDATE_SECONDS = 6 * 60 * 60
RESOLVER_SUBSCRIPTIONS_MAX = 64
RESOLVER_SUBSCRIPTIONS_MAX_BYTES = RESOLVER_INDEX_MAX_BYTES * (RESOLVER_SUBSCRIPTIONS_MAX + 1)
RESOLVER_FETCH_TIMEOUT = 15.0
RESOLVER_FETCH_MAX_PAGES = 256

_SOURCE_PREFIX = "resolver-index:"
_LOCAL_HTTP_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_REFRESH_WORKERS = 8
# Python cannot cancel OS DNS calls. Timed-out transports retain their slot
# until they exit, bounding abandoned work across all subscription stores.
_FETCH_SLOTS = threading.BoundedSemaphore(_REFRESH_WORKERS)


class ResolverSubscriptionError(AssetError):
    """A resolver-index subscription could not be loaded or persisted."""


@dataclass(frozen=True)
class ResolverSuggestion:
    digest: str
    name: str
    source_id: str
    source: str
    index_name: str = ""
    kind: str = ""
    size: int = -1
    component_manifest: AssetComponentManifest | None = None


@dataclass(frozen=True)
class ResolverSubscription:
    id: str
    source: str
    source_type: str
    index: ResolverIndex
    etag: str = ""
    checked_at: float = 0.0
    error: str = ""
    refreshed_at: float = 0.0
    trusted_for_p2p: bool = False
    license_authoritative: bool = False
    p2p_tombstones: tuple[tuple[str, float], ...] = ()
    complete_snapshot: bool = False

    def descriptor(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "sourceType": self.source_type,
            "name": self.index.name,
            "description": self.index.description,
            "homepage": self.index.homepage,
            "updated": self.index.updated,
            "entryCount": len(self.index.entries),
            "etag": self.etag,
            "checkedAt": self.checked_at,
            "refreshedAt": self.refreshed_at,
            "error": self.error,
            "trustedForP2P": self.trusted_for_p2p,
            "licenseAuthoritative": self.license_authoritative,
        }

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "sourceType": self.source_type,
            "completeSnapshot": self.complete_snapshot,
            "etag": self.etag,
            "checkedAt": self.checked_at,
            "refreshedAt": self.refreshed_at,
            "error": self.error,
            "trustedForP2P": self.trusted_for_p2p,
            "licenseAuthoritative": self.license_authoritative,
            "p2pTombstones": [
                {"digest": digest, "observedAt": observed_at}
                for digest, observed_at in self.p2p_tombstones
            ],
            "index": self.index.to_wire(),
        }


def _subscription_url(value: str) -> str:
    if (
        value != value.strip()
        or len(value) > RESOLVER_INDEX_MAX_STRING
        or any(character.isspace() for character in value)
    ):
        raise ResolverSubscriptionError(
            "resolver index URL must be trimmed, bounded, and contain no raw whitespace"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ResolverSubscriptionError(f"invalid resolver index URL: {value!r}") from exc
    local_http = parsed.scheme == "http" and parsed.hostname in _LOCAL_HTTP_HOSTS
    if (
        (parsed.scheme != "https" and not local_http)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ResolverSubscriptionError(
            "resolver index URLs must use HTTPS (loopback HTTP is allowed) "
            "and must not contain credentials or a fragment"
        )
    return value


@dataclass(frozen=True)
class _OfficialResolverBootstrap:
    source: str
    provider_id: str
    subscription_id: str

    def __post_init__(self) -> None:
        try:
            _subscription_url(self.source)
        except (ValueError, ResolverSubscriptionError) as exc:
            raise ResolverSubscriptionError(
                "official resolver bootstrap requires a valid URL"
            ) from exc
        if not self.source.isascii():
            raise ResolverSubscriptionError(
                "official resolver URL must be ASCII; percent-encode non-ASCII paths and queries"
            )
        if (
            not self.provider_id
            or self.provider_id != self.provider_id.strip()
            or len(self.provider_id) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in self.provider_id)
        ):
            raise ResolverSubscriptionError(
                "official resolver provider ID must be a trimmed string "
                "of 1-512 characters without controls"
            )
        if len(self.subscription_id) != 32 or any(
            char not in "0123456789abcdef" for char in self.subscription_id
        ):
            raise ResolverSubscriptionError(
                "official resolver bootstrap has an invalid subscription ID"
            )

    def to_wire(self) -> dict[str, str]:
        return {
            "source": self.source,
            "providerId": self.provider_id,
            "subscriptionId": self.subscription_id,
        }

    @classmethod
    def from_wire(cls, value: object) -> _OfficialResolverBootstrap:
        if not isinstance(value, Mapping):
            raise ResolverSubscriptionError("official resolver bootstrap must be an object")
        wire = cast("Mapping[str, object]", value)
        if set(wire) != {"source", "providerId", "subscriptionId"} or not all(
            isinstance(item, str) for item in wire.values()
        ):
            raise ResolverSubscriptionError("official resolver bootstrap has an invalid shape")
        return cls(
            cast("str", wire["source"]),
            cast("str", wire["providerId"]),
            cast("str", wire["subscriptionId"]),
        )


def _canonical_source(value: str) -> tuple[str, str]:
    if (
        not value
        or value != value.strip()
        or len(value) > RESOLVER_INDEX_MAX_STRING
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ResolverSubscriptionError(
            "resolver index source must be a trimmed string of at most "
            f"{RESOLVER_INDEX_MAX_STRING} characters without controls"
        )
    windows_path = (
        len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}
    )
    parsed = urlsplit(value)
    if parsed.scheme and not windows_path:
        return _subscription_url(value), "url"
    return str(Path(value).expanduser().resolve()), "local"


def _read_bounded(handle: IO[bytes]) -> bytes:
    data = handle.read(RESOLVER_INDEX_MAX_BYTES + 1)
    if len(data) > RESOLVER_INDEX_MAX_BYTES:
        raise ResolverIndexError(f"resolver index exceeds {RESOLVER_INDEX_MAX_BYTES} bytes")
    return data


def _read_local(path: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            return _read_bounded(handle)
    except OSError as exc:
        raise ResolverSubscriptionError(f"cannot read resolver index {path!r}: {exc}") from exc


def _index_revision(index: ResolverIndex) -> str:
    try:
        canonical = json.dumps(
            index.to_wire(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ResolverSubscriptionError("resolver snapshot must contain valid Unicode") from exc
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _etag(value: str | None) -> str:
    if value is None:
        return ""
    if len(value) > 1024 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ResolverSubscriptionError("resolver index response has an invalid ETag")
    return value


def _timestamp(value: object, error: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolverSubscriptionError(error)
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ResolverSubscriptionError(error) from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ResolverSubscriptionError(error)
    return normalized


class _ResolverRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        _subscription_url(newurl)
        if urlsplit(req.full_url).scheme == "https" and urlsplit(newurl).scheme != "https":
            raise ResolverSubscriptionError("resolver index redirect cannot downgrade HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ResolverDeadlineHandler(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    def __init__(self, deadline: float) -> None:
        # HTTPSHandler.__init__ eagerly loads system roots even for HTTP.
        # HTTPSConnection creates its verified context only when HTTPS is used.
        urllib.request.AbstractHTTPHandler.__init__(self)
        self._deadline = deadline
        self._timers: list[threading.Timer] = []

    def _watch_socket(self, connection: HTTPConnection) -> None:
        sock = connection.sock
        assert sock is not None
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            connection.close()
            raise ResolverSubscriptionError("resolver index refresh exceeded its time limit")
        sock.settimeout(remaining)

        def interrupt() -> None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

        # Socket timeouts alone allow a slow trickle of headers or chunk framing
        # to extend the refresh indefinitely. Interrupt the actual connection.
        timer = threading.Timer(max(0.0, self._deadline - time.monotonic()), interrupt)
        timer.daemon = True
        self._timers.append(timer)
        timer.start()

    def http_open(self, req: urllib.request.Request) -> HTTPResponse:
        handler = self

        class Connection(HTTPConnection):
            def connect(self) -> None:
                super().connect()
                handler._watch_socket(self)

        return self.do_open(Connection, req)

    def https_open(self, req: urllib.request.Request) -> HTTPResponse:
        handler = self

        class Connection(HTTPSConnection):
            def connect(self) -> None:
                super().connect()
                handler._watch_socket(self)

        return self.do_open(Connection, req)

    def close(self) -> None:
        for timer in self._timers:
            timer.cancel()
            timer.join()


def _fetch_url(
    url: str,
    etag: str,
    *,
    timeout: float,
) -> tuple[bytes | None, str]:
    deadline = time.monotonic() + timeout
    slots = _FETCH_SLOTS
    if not slots.acquire(blocking=False):
        raise ResolverSubscriptionError("resolver index fetch capacity exhausted")
    result: Future[tuple[bytes | None, str]] = Future()

    def fetch() -> None:
        try:
            result.set_result(_fetch_url_until_deadline(url, etag, deadline=deadline))
        except BaseException as exc:
            result.set_exception(exc)
        finally:
            slots.release()

    # A daemon, rather than an executor context, lets the caller release the
    # store lock by the deadline even during DNS, address attempts or TLS. Only
    # the waiting caller can publish; late results never touch store state.
    worker = threading.Thread(target=fetch, name="resolver-fetch", daemon=True)
    try:
        worker.start()
    except BaseException:
        slots.release()
        raise
    try:
        fetched = result.result(timeout=max(0.0, deadline - time.monotonic()))
    except TimeoutError as exc:
        raise ResolverSubscriptionError("resolver index refresh exceeded its time limit") from exc
    if time.monotonic() >= deadline:
        raise ResolverSubscriptionError("resolver index refresh exceeded its time limit")
    return fetched


def _fetch_url_until_deadline(
    url: str,
    etag: str,
    *,
    deadline: float,
) -> tuple[bytes | None, str]:
    parsed_url = urlsplit(url)
    # Preserve every non-cursor query byte, including duplicate keys and escaping.
    query = "&".join(
        part
        for part in parsed_url.query.split("&")
        if unquote_plus(part.partition("=")[0]) != "cursor"
    )
    first_url = urlunsplit(parsed_url._replace(query=query))
    page_url = first_url
    total_bytes = 0
    seen_cursors: set[str] = set()
    digests: set[str] = set()
    entries: list[dict[str, object]] = []
    metadata: dict[str, object] | None = None
    first_etag = ""
    for page_number in range(RESOLVER_FETCH_MAX_PAGES):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResolverSubscriptionError("resolver index refresh exceeded its time limit")
        data, page_etag = _fetch_page(
            _subscription_url(page_url),
            etag if page_number == 0 else "",
            timeout=remaining,
            deadline=deadline,
            max_bytes=RESOLVER_INDEX_MAX_BYTES - total_bytes,
        )
        if time.monotonic() >= deadline:
            raise ResolverSubscriptionError("resolver index refresh exceeded its time limit")
        if data is None:
            if page_number != 0 or not etag:
                raise ResolverSubscriptionError(
                    "resolver index returned not-modified without a cached complete document"
                )
            return None, page_etag
        total_bytes += len(data)
        wire = dict(decode_resolver_index_document(data))
        cursor = wire.pop("nextCursor", None)
        if cursor is not None and (not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048):
            raise ResolverIndexError("resolver nextCursor must be a string of 1-2048 characters")
        protocol = wire.get("providerProtocol")
        if "providerProtocol" in wire and (type(protocol) is not int or protocol != 1):
            raise ResolverIndexError("unsupported resolver providerProtocol")
        page = resolver_index_from_wire(wire)
        page_metadata = {
            key: wire.get(key)
            for key in (
                "providerProtocol",
                "dinksterResolver",
                "name",
                "description",
                "homepage",
                "updated",
            )
        }
        if metadata is None:
            metadata = page_metadata
            first_etag = page_etag
        elif metadata != page_metadata:
            raise ResolverIndexError("resolver pagination metadata changed between pages")
        if len(entries) + len(page.entries) > RESOLVER_INDEX_MAX_ENTRIES:
            raise ResolverIndexError(
                f"resolver index accepts at most {RESOLVER_INDEX_MAX_ENTRIES} entries"
            )
        for entry in page.entries:
            if entry.digest in digests:
                raise ResolverIndexError(f"resolver pagination duplicates digest {entry.digest}")
            digests.add(entry.digest)
            entries.append(entry.to_wire())
        if cursor is None:
            assembled = page.to_wire()
            assembled["entries"] = entries
            result = json.dumps(assembled, separators=(",", ":")).encode("utf-8")
            if len(result) > RESOLVER_INDEX_MAX_BYTES:
                raise ResolverIndexError(f"resolver index exceeds {RESOLVER_INDEX_MAX_BYTES} bytes")
            if time.monotonic() >= deadline:
                raise ResolverSubscriptionError("resolver index refresh exceeded its time limit")
            # A page validator cannot attest that the rest of the catalog is
            # unchanged. Multi-page caches must always refetch every page.
            return result, first_etag if page_number == 0 else ""
        assert isinstance(cursor, str)
        if cursor in seen_cursors:
            raise ResolverIndexError("resolver pagination cursor cycle")
        seen_cursors.add(cursor)
        try:
            encoded_cursor = quote(cursor, safe="")
        except UnicodeEncodeError as exc:
            raise ResolverIndexError("resolver nextCursor must be valid UTF-8") from exc
        page_url = urlunsplit(
            parsed_url._replace(query=query + ("&" if query else "") + "cursor=" + encoded_cursor)
        )
    raise ResolverIndexError(f"resolver pagination exceeds {RESOLVER_FETCH_MAX_PAGES} pages")


def _fetch_page(
    url: str,
    etag: str,
    *,
    timeout: float,
    deadline: float,
    max_bytes: int,
) -> tuple[bytes | None, str]:
    headers = {"User-Agent": "dinkster-resolver-index"}
    if etag:
        headers["If-None-Match"] = etag
    request = urllib.request.Request(url, headers=headers)
    deadline_handler = _ResolverDeadlineHandler(deadline)
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _ResolverRedirectHandler(), deadline_handler
        )
        response = opener.open(request, timeout=timeout)
        with response:
            _subscription_url(response.geturl())
            if response.status != 200:
                raise ResolverSubscriptionError(
                    f"resolver index request requires HTTP 200, got HTTP {response.status}"
                )
            content_length = response.headers.get("Content-Length")
            declared_length: int | None = None
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise ResolverSubscriptionError(
                        "resolver index response has an invalid Content-Length"
                    ) from exc
                if declared_length < 0:
                    raise ResolverSubscriptionError(
                        "resolver index response has an invalid Content-Length"
                    )
                if declared_length > max_bytes:
                    raise ResolverIndexError(
                        f"resolver index exceeds {RESOLVER_INDEX_MAX_BYTES} bytes"
                    )
            data = bytearray()
            while True:
                if time.monotonic() >= deadline:
                    raise ResolverSubscriptionError(
                        "resolver index refresh exceeded its time limit"
                    )
                chunk = response.read1(min(64 * 1024, max_bytes - len(data) + 1))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ResolverIndexError(
                        f"resolver index exceeds {RESOLVER_INDEX_MAX_BYTES} bytes"
                    )
            if declared_length is not None and len(data) != declared_length:
                raise ResolverSubscriptionError("resolver index response body is incomplete")
            return bytes(data), _etag(response.headers.get("ETag"))
    except urllib.error.HTTPError as exc:
        with exc:
            if exc.code == 304:
                response_etag = _etag(exc.headers.get("ETag"))
                return None, response_etag or etag
        raise ResolverSubscriptionError(
            f"resolver index request failed with HTTP {exc.code}: {url}"
        ) from exc
    except (OSError, urllib.error.URLError, HTTPException, TimeoutError) as exc:
        raise ResolverSubscriptionError(f"resolver index request failed: {url}: {exc}") from exc
    finally:
        deadline_handler.close()


def _subscription_from_wire(wire: Mapping[str, object]) -> ResolverSubscription:
    expected = {
        "id",
        "source",
        "sourceType",
        "completeSnapshot",
        "etag",
        "checkedAt",
        "refreshedAt",
        "error",
        "trustedForP2P",
        "licenseAuthoritative",
        "p2pTombstones",
        "index",
    }
    unknown = set(wire) - expected
    if unknown:
        raise ResolverSubscriptionError(
            f"persisted resolver subscription has unknown fields: {sorted(unknown)}"
        )
    subscription_id = wire.get("id")
    source = wire.get("source")
    source_type = wire.get("sourceType")
    etag = wire.get("etag", "")
    checked_at = wire.get("checkedAt", 0.0)
    error = wire.get("error", "")
    refreshed_at = wire.get("refreshedAt", checked_at if not error else 0.0)
    trusted_for_p2p = wire.get("trustedForP2P", False)
    license_authoritative = wire.get("licenseAuthoritative", False)
    complete_snapshot = wire.get("completeSnapshot", False)
    tombstones = wire.get("p2pTombstones", [])
    index = wire.get("index")
    if (
        not isinstance(subscription_id, str)
        or len(subscription_id) != 32
        or any(character not in "0123456789abcdef" for character in subscription_id)
    ):
        raise ResolverSubscriptionError("persisted resolver subscription has an invalid id")
    if not isinstance(source, str) or not source:
        raise ResolverSubscriptionError("persisted resolver subscription has an invalid source")
    if source_type not in {"local", "url"}:
        raise ResolverSubscriptionError("persisted resolver subscription has an invalid sourceType")
    canonical_source, canonical_type = _canonical_source(source)
    if canonical_source != source or canonical_type != source_type:
        raise ResolverSubscriptionError("persisted resolver subscription source is not canonical")
    if not isinstance(etag, str) or not isinstance(error, str):
        raise ResolverSubscriptionError("persisted resolver subscription status must be strings")
    if type(trusted_for_p2p) is not bool or type(license_authoritative) is not bool:
        raise ResolverSubscriptionError("persisted resolver P2P trust flags must be booleans")
    if type(complete_snapshot) is not bool:
        raise ResolverSubscriptionError("persisted resolver completeSnapshot must be a boolean")
    if not isinstance(tombstones, list):
        raise ResolverSubscriptionError("persisted resolver P2P tombstones must be a bounded list")
    tombstone_rows = cast("list[object]", tombstones)
    if len(tombstone_rows) > RESOLVER_INDEX_MAX_ENTRIES:
        raise ResolverSubscriptionError("persisted resolver P2P tombstones must be a bounded list")
    parsed_tombstones: list[tuple[str, float]] = []
    for position, row in enumerate(tombstone_rows):
        if not isinstance(row, Mapping):
            raise ResolverSubscriptionError(
                f"persisted resolver P2P tombstone {position} has an invalid shape"
            )
        row_wire = cast("Mapping[object, object]", row)
        if set(row_wire) != {"digest", "observedAt"} or any(
            not isinstance(key, str) for key in row_wire
        ):
            raise ResolverSubscriptionError(
                f"persisted resolver P2P tombstone {position} has an invalid shape"
            )
        try:
            tombstone = ProviderP2PTombstoneV1.from_wire(cast("Mapping[str, object]", row_wire))
        except AssetError as exc:
            raise ResolverSubscriptionError(
                f"persisted resolver P2P tombstone {position} is invalid: {exc}"
            ) from exc
        parsed_tombstones.append((tombstone.digest, tombstone.observed_at))
    if len({digest for digest, _ in parsed_tombstones}) != len(parsed_tombstones):
        raise ResolverSubscriptionError("persisted resolver P2P tombstones contain duplicates")
    normalized_checked_at = _timestamp(
        checked_at,
        "persisted resolver subscription checkedAt is invalid",
    )
    normalized_refreshed_at = _timestamp(
        refreshed_at,
        "persisted resolver subscription refreshedAt is invalid",
    )
    if normalized_refreshed_at > normalized_checked_at:
        raise ResolverSubscriptionError(
            "persisted resolver subscription refreshedAt cannot follow checkedAt"
        )
    if not isinstance(index, Mapping):
        raise ResolverSubscriptionError("persisted resolver subscription index must be an object")
    return ResolverSubscription(
        id=subscription_id,
        source=source,
        source_type=cast("str", source_type),
        index=resolver_index_from_wire(cast("Mapping[str, object]", index)),
        etag=_etag(etag),
        checked_at=normalized_checked_at,
        refreshed_at=normalized_refreshed_at,
        error=error,
        trusted_for_p2p=trusted_for_p2p,
        license_authoritative=license_authoritative,
        p2p_tombstones=tuple(parsed_tombstones),
        complete_snapshot=complete_snapshot,
    )


class ResolverSubscriptionStore:
    """One durable subscription list feeding a removable provenance layer."""

    def __init__(
        self,
        path: Path | str,
        provenance: ProvenanceStore,
        *,
        region: str = "",
        clock: Callable[[], float] = time.time,
        fetch_timeout: float = RESOLVER_FETCH_TIMEOUT,
        revalidate_seconds: float = RESOLVER_REVALIDATE_SECONDS,
    ) -> None:
        if region:
            require_region(region)
        if not math.isfinite(fetch_timeout) or fetch_timeout <= 0:
            raise ValueError("resolver fetch timeout must be finite and positive")
        if not math.isfinite(revalidate_seconds) or revalidate_seconds < 0:
            raise ValueError("resolver revalidation interval must be finite and non-negative")
        self._path = Path(path)
        self._provenance = provenance
        self._region = region
        self._clock = clock
        self._fetch_timeout = fetch_timeout
        self._revalidate_seconds = revalidate_seconds
        self._lock = threading.RLock()
        self._official_bootstrap: _OfficialResolverBootstrap | None = None
        self._subscriptions = self._load()
        self._filename_index: dict[str, tuple[ResolverSuggestion, ...]] = {}
        self._p2p_snapshots_cache: tuple[ProviderP2PSnapshotV1, ...] | None = None
        self._p2p_declarations_cache: tuple[PublicSwarmDeclarationV1, ...] | None = None
        self._p2p_declaration_batches: tuple[
            tuple[float, float, tuple[PublicSwarmDeclarationV1, ...]], ...
        ] = ()
        self._p2p_active_batch_key: tuple[int, ...] | None = None
        self._p2p_active_declarations_cache: tuple[PublicSwarmDeclarationV1, ...] = ()
        self._ensure_p2p_cache()
        active_sources = {
            self._source_name(subscription.id) for subscription in self._subscriptions
        }
        for source_name in self._provenance.source_names():
            if source_name.startswith(_SOURCE_PREFIX) and source_name not in active_sources:
                self._provenance.remove_source(source_name)
        for subscription in self._subscriptions:
            self._apply(subscription)
        self._rebuild_filename_index()

    @staticmethod
    def _source_name(subscription_id: str) -> str:
        return _SOURCE_PREFIX + subscription_id

    def _load(self) -> list[ResolverSubscription]:
        try:
            with self._path.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size > RESOLVER_SUBSCRIPTIONS_MAX_BYTES:
                    raise ResolverSubscriptionError(
                        "resolver subscription file exceeds "
                        f"{RESOLVER_SUBSCRIPTIONS_MAX_BYTES} bytes"
                    )
                data = handle.read(RESOLVER_SUBSCRIPTIONS_MAX_BYTES + 1)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ResolverSubscriptionError(
                f"cannot load resolver subscriptions from {self._path}: {exc}"
            ) from exc
        if len(data) > RESOLVER_SUBSCRIPTIONS_MAX_BYTES:
            raise ResolverSubscriptionError(
                f"resolver subscription file exceeds {RESOLVER_SUBSCRIPTIONS_MAX_BYTES} bytes"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResolverSubscriptionError(
                "resolver subscription file must be valid UTF-8"
            ) from exc

        def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ResolverSubscriptionError(
                        f"duplicate resolver subscription JSON field {key!r}"
                    )
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise ResolverSubscriptionError(f"invalid resolver subscription JSON number {value}")

        try:
            loaded: object = json.loads(
                text,
                object_pairs_hook=object_pairs,
                parse_constant=reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ResolverSubscriptionError(
                "invalid resolver subscription JSON at "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        except ValueError as exc:
            raise ResolverSubscriptionError(f"invalid resolver subscription JSON: {exc}") from exc
        except RecursionError as exc:
            raise ResolverSubscriptionError(
                "resolver subscription JSON is nested too deeply"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise ResolverSubscriptionError("unsupported resolver subscription file format")
        loaded_wire = cast("Mapping[str, object]", loaded)
        unknown = set(loaded_wire) - {
            "dinksterResolverSubscriptions",
            "subscriptions",
            "officialBootstrap",
        }
        if unknown:
            raise ResolverSubscriptionError(
                f"resolver subscription file has unknown fields: {sorted(unknown)}"
            )
        if loaded_wire.get("dinksterResolverSubscriptions") != 1:
            raise ResolverSubscriptionError("unsupported resolver subscription file format")
        if "officialBootstrap" in loaded_wire:
            self._official_bootstrap = _OfficialResolverBootstrap.from_wire(
                loaded_wire["officialBootstrap"]
            )
        rows = loaded_wire.get("subscriptions")
        if not isinstance(rows, list):
            raise ResolverSubscriptionError("resolver subscription file requires a list")
        rows_wire = cast("list[object]", rows)
        if len(rows_wire) > RESOLVER_SUBSCRIPTIONS_MAX:
            raise ResolverSubscriptionError(
                f"resolver subscription file exceeds {RESOLVER_SUBSCRIPTIONS_MAX} subscriptions"
            )
        subscriptions: list[ResolverSubscription] = []
        ids: set[str] = set()
        sources: set[str] = set()
        for position, row in enumerate(rows_wire):
            if not isinstance(row, Mapping):
                raise ResolverSubscriptionError(f"subscriptions[{position}] must be an object")
            subscription = _subscription_from_wire(cast("Mapping[str, object]", row))
            if subscription.id in ids or subscription.source in sources:
                raise ResolverSubscriptionError(
                    "resolver subscription ids and sources must be unique"
                )
            ids.add(subscription.id)
            sources.add(subscription.source)
            self._check_official_identity(subscription)
            subscriptions.append(subscription)
        return subscriptions

    def _save(self) -> None:
        try:
            serialized = json.dumps(
                {
                    "dinksterResolverSubscriptions": 1,
                    "subscriptions": [
                        subscription.to_wire() for subscription in self._subscriptions
                    ],
                    **(
                        {"officialBootstrap": self._official_bootstrap.to_wire()}
                        if self._official_bootstrap is not None
                        else {}
                    ),
                },
                indent=1,
                allow_nan=False,
            )
            encoded = serialized.encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ResolverSubscriptionError(
                f"cannot serialize resolver subscriptions: {exc}"
            ) from exc
        if len(encoded) > RESOLVER_SUBSCRIPTIONS_MAX_BYTES:
            raise ResolverSubscriptionError(
                f"resolver subscription file exceeds {RESOLVER_SUBSCRIPTIONS_MAX_BYTES} bytes"
            )
        # Validate derived authority before replacing durable state. A failed
        # build or write leaves the previous caches intact for caller rollback.
        cache = self._build_p2p_cache()
        tmp = self._path.with_name(self._path.name + f".tmp-{os.getpid()}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(encoded)
            os.replace(tmp, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        (
            self._p2p_snapshots_cache,
            self._p2p_declarations_cache,
            self._p2p_declaration_batches,
        ) = cache
        self._p2p_active_batch_key = None
        self._p2p_active_declarations_cache = ()

    def _records(self, subscription: ResolverSubscription) -> tuple[ProvenanceRecord, ...]:
        records: list[ProvenanceRecord] = []
        for entry in subscription.index.entries:
            metadata: dict[str, object] = {
                "name": entry.name,
                "resolverIndex": subscription.source,
            }
            if entry.gated:
                metadata["gated"] = entry.gated
            if entry.kind:
                metadata["kind"] = entry.kind
            if entry.size >= 0:
                metadata["size"] = entry.size
            if entry.family:
                metadata["family"] = entry.family
            if entry.variant:
                metadata["variant"] = thaw_json(entry.variant)
            if entry.component_manifest is not None:
                metadata["components"] = entry.component_manifest.to_wire()
            records.append(
                ProvenanceRecord(
                    digest=entry.digest,
                    sources=entry.urls_for(self._region),
                    license=entry.license,
                    note=entry.notes,
                    metadata=metadata,
                )
            )
        return tuple(records)

    def _apply(self, subscription: ResolverSubscription) -> None:
        self._provenance.replace_source(
            self._source_name(subscription.id), self._records(subscription)
        )

    @staticmethod
    def _basename(value: str) -> str:
        return value.replace("\\", "/").rsplit("/", 1)[-1]

    def _rebuild_filename_index(self) -> None:
        indexed: dict[str, list[ResolverSuggestion]] = {}
        for subscription in self._subscriptions:
            for entry in subscription.index.entries:
                basename = self._basename(entry.name)
                if not basename:
                    continue
                indexed.setdefault(basename, []).append(
                    ResolverSuggestion(
                        digest=entry.digest,
                        name=entry.name,
                        source_id=subscription.id,
                        source=subscription.source,
                        index_name=subscription.index.name,
                        kind=entry.kind,
                        size=entry.size,
                        component_manifest=entry.component_manifest,
                    )
                )
        self._filename_index = {name: tuple(suggestions) for name, suggestions in indexed.items()}

    def subscriptions(self) -> tuple[ResolverSubscription, ...]:
        with self._lock:
            return tuple(self._subscriptions)

    def public_sources(self, digest: str | None = None) -> tuple[PublicAcquisitionSourceV1, ...]:
        """Receipt-eligible declarations from each currently subscribed index."""
        sources: list[PublicAcquisitionSourceV1] = []
        with self._lock:
            for subscription in self._subscriptions:
                revision = _index_revision(subscription.index)
                for entry in subscription.index.entries:
                    if (digest is not None and entry.digest != digest) or entry.size < 0:
                        continue
                    try:
                        sources.append(
                            PublicAcquisitionSourceV1(
                                digest=entry.digest,
                                size_bytes=entry.size,
                                source_type="declarative-resolver",
                                source_id=subscription.id,
                                source_revision=revision,
                                listed_urls=entry.urls_for(self._region),
                            )
                        )
                    except AssetError:
                        continue
        return tuple(sources)

    @staticmethod
    def _safe_p2p_name(name: str) -> bool:
        lowered = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        return lowered.endswith((".safetensors", ".gguf"))

    @staticmethod
    def _provider_locations(urls: Sequence[str], *, gated: bool) -> tuple[ProviderLocationV1, ...]:
        locations: list[ProviderLocationV1] = []
        for url in urls:
            try:
                locations.append(
                    ProviderLocationV1(url, eligible=not gated, credential_free=not gated)
                )
            except AssetError:
                continue
        return tuple(locations)

    def _provider_snapshot(
        self, subscription: ResolverSubscription
    ) -> ProviderP2PSnapshotV1 | None:
        self._check_official_identity(subscription)
        index = subscription.index
        # Legacy caches may have discarded nextCursor. Their HTTP entries remain
        # usable, but only a new complete refresh can establish P2P authority.
        if (
            not subscription.complete_snapshot
            or not subscription.trusted_for_p2p
            or not index.name
            or not index.updated
        ):
            return None
        # Snapshot revisions cover full content independently of HTTP validators.
        source_revision = f"{index.updated}|{_index_revision(index)}"
        artifacts: list[ProviderArtifactP2PV1] = []
        enumeration: list[ProviderP2PEnumerationV1] = []
        expires_at = subscription.refreshed_at + P2P_REMOTE_GRANT_MAX_SECONDS
        for entry in sorted(index.entries, key=lambda row: row.digest):
            if entry.size <= 0 or entry.p2p is None:
                continue
            try:
                source_id = (
                    "resolver:"
                    + hashlib.sha256(f"{index.name}\0{entry.digest}".encode()).hexdigest()
                )
                grant_id = hashlib.sha256(
                    f"resolver-p2p-v1\0{index.name}\0{source_revision}\0{entry.digest}".encode()
                ).hexdigest()
                artifacts.append(
                    ProviderArtifactP2PV1(
                        source_id=source_id,
                        digest=entry.digest,
                        size_bytes=entry.size,
                        descriptor=entry.p2p,
                        license=entry.license,
                        format_safe=self._safe_p2p_name(entry.name),
                        locations=self._provider_locations(
                            entry.urls_for(self._region), gated=entry.gated
                        ),
                    )
                )
                enumeration.append(
                    ProviderP2PEnumerationV1(
                        grant_id=grant_id,
                        digest=entry.digest,
                        size_bytes=entry.size,
                        descriptor=entry.p2p,
                        expires_at=expires_at,
                    )
                )
            except AssetError:
                continue
        try:
            return ProviderP2PSnapshotV1(
                provider_id=index.name,
                source_revision=source_revision,
                refreshed_at=subscription.refreshed_at,
                artifacts=tuple(artifacts),
                p2p_artifacts=tuple(enumeration),
                p2p_trackers=(),
                tombstones=tuple(
                    ProviderP2PTombstoneV1(digest, observed_at)
                    for digest, observed_at in subscription.p2p_tombstones
                ),
            )
        except AssetError:
            return None

    def provider_p2p_snapshots(self) -> tuple[ProviderP2PSnapshotV1, ...]:
        """Current complete resolver refreshes explicitly trusted for internet P2P."""
        with self._lock:
            self._ensure_p2p_cache()
            assert self._p2p_snapshots_cache is not None
            return self._p2p_snapshots_cache

    def _ensure_p2p_cache(self) -> None:
        if self._p2p_snapshots_cache is not None:
            return
        (
            self._p2p_snapshots_cache,
            self._p2p_declarations_cache,
            self._p2p_declaration_batches,
        ) = self._build_p2p_cache()

    def _build_p2p_cache(
        self,
    ) -> tuple[
        tuple[ProviderP2PSnapshotV1, ...],
        tuple[PublicSwarmDeclarationV1, ...],
        tuple[tuple[float, float, tuple[PublicSwarmDeclarationV1, ...]], ...],
    ]:
        snapshots = tuple(
            snapshot
            for subscription in self._subscriptions
            if (snapshot := self._provider_snapshot(subscription)) is not None
        )
        declarations: list[PublicSwarmDeclarationV1] = []
        batches: list[tuple[float, float, tuple[PublicSwarmDeclarationV1, ...]]] = []
        for snapshot in snapshots:
            snapshot_declarations = tuple(
                decision.declaration
                for decision in provider_declarations(
                    snapshot,
                    trusted_provider_ids=frozenset({snapshot.provider_id}),
                    now=snapshot.refreshed_at,
                )
                if decision.declaration is not None
            )
            grouped: dict[tuple[float, float], list[PublicSwarmDeclarationV1]] = {}
            for declaration in snapshot_declarations:
                grouped.setdefault((declaration.refreshed_at, declaration.expires_at), []).append(
                    declaration
                )
            batches.extend(
                (active_from, expires_at, tuple(rows))
                for (active_from, expires_at), rows in grouped.items()
            )
            declarations.extend(snapshot_declarations)
        return snapshots, tuple(declarations), tuple(batches)

    def public_swarm_declarations(
        self, digest: str | None = None
    ) -> tuple[PublicSwarmDeclarationV1, ...]:
        """Current declarations from resolver subscriptions explicitly trusted for P2P."""
        with self._lock:
            self._ensure_p2p_cache()
            assert self._p2p_declarations_cache is not None
            now = _timestamp(self._clock(), "resolver clock returned an invalid timestamp")
            active_batch_key = tuple(
                index
                for index, (active_from, expires_at, _) in enumerate(self._p2p_declaration_batches)
                if active_from <= now < expires_at
            )
            if digest is None:
                if len(active_batch_key) == len(self._p2p_declaration_batches):
                    return self._p2p_declarations_cache
                if not active_batch_key:
                    return ()
                if active_batch_key == self._p2p_active_batch_key:
                    return self._p2p_active_declarations_cache
                declarations = tuple(
                    declaration
                    for index in active_batch_key
                    for declaration in self._p2p_declaration_batches[index][2]
                )
                self._p2p_active_batch_key = active_batch_key
                self._p2p_active_declarations_cache = declarations
                return declarations
            return tuple(
                declaration
                for index in active_batch_key
                for declaration in self._p2p_declaration_batches[index][2]
                if declaration.digest == digest
            )

    def suggest(self, name: str) -> tuple[ResolverSuggestion, ...]:
        basename = self._basename(name)
        if not basename:
            return ()
        with self._lock:
            suggestions = self._filename_index.get(basename, ())
            seen: set[str] = set()
            result: list[ResolverSuggestion] = []
            for suggestion in suggestions:
                if suggestion.digest not in seen:
                    seen.add(suggestion.digest)
                    result.append(suggestion)
            return tuple(result)

    def _new_subscription(self, source: str) -> ResolverSubscription:
        canonical_source, source_type = _canonical_source(source)
        if len(self._subscriptions) >= RESOLVER_SUBSCRIPTIONS_MAX:
            raise ResolverSubscriptionError(
                f"at most {RESOLVER_SUBSCRIPTIONS_MAX} resolver indexes may be subscribed"
            )
        if any(row.source == canonical_source for row in self._subscriptions):
            raise ResolverSubscriptionError(
                f"resolver index is already subscribed: {canonical_source}"
            )
        etag = ""
        if source_type == "url":
            data, etag = _fetch_url(canonical_source, "", timeout=self._fetch_timeout)
            if data is None:
                raise ResolverSubscriptionError(
                    "resolver index returned not-modified without a cached document"
                )
        else:
            data = _read_local(canonical_source)
        checked_at = _timestamp(self._clock(), "resolver clock returned an invalid timestamp")
        return ResolverSubscription(
            id=uuid.uuid4().hex,
            source=canonical_source,
            source_type=source_type,
            index=parse_resolver_index(data),
            etag=etag,
            checked_at=checked_at,
            refreshed_at=checked_at,
            complete_snapshot=True,
        )

    def _add_subscription(self, subscription: ResolverSubscription) -> ResolverSubscription:
        self._apply(subscription)
        self._subscriptions.append(subscription)
        try:
            self._save()
        except Exception:
            self._subscriptions.pop()
            self._provenance.remove_source(self._source_name(subscription.id))
            raise
        self._rebuild_filename_index()
        return subscription

    def subscribe(self, source: str) -> ResolverSubscription:
        with self._lock:
            return self._add_subscription(self._new_subscription(source))

    @staticmethod
    def _require_official_identity(
        subscription: ResolverSubscription,
        bootstrap: _OfficialResolverBootstrap | None,
    ) -> None:
        if bootstrap is not None and subscription.id == bootstrap.subscription_id:
            if (
                subscription.source != bootstrap.source
                or subscription.index.name != bootstrap.provider_id
            ):
                raise ResolverSubscriptionError(
                    "official resolver provider identity does not match its bootstrap"
                )

    def _check_official_identity(self, subscription: ResolverSubscription) -> None:
        self._require_official_identity(subscription, self._official_bootstrap)

    def bootstrap_official(
        self, source: str | None, provider_id: str | None
    ) -> ResolverSubscription | None:
        """Trust the configured official provider once, retaining later user choices."""
        if not source or not provider_id:
            raise ResolverSubscriptionError(
                "official resolver URL and provider ID must both be configured"
            )
        configured = _OfficialResolverBootstrap(source, provider_id, uuid.uuid4().hex)
        with self._lock:
            previous = self._official_bootstrap
            if previous is not None:
                if (previous.source, previous.provider_id) != (source, provider_id):
                    raise ResolverSubscriptionError(
                        "official resolver configuration differs from its recorded bootstrap; "
                        "saved choice preserved"
                    )
                # A missing recorded subscription is an explicit unsubscribe,
                # not permission to recreate it on the next startup.
                return next(
                    (row for row in self._subscriptions if row.id == previous.subscription_id),
                    None,
                )
            existing = next((row for row in self._subscriptions if row.source == source), None)
            subscription = existing or self._new_subscription(source)
            if subscription.index.name != provider_id:
                raise ResolverSubscriptionError(
                    "official resolver export name does not match configured provider ID"
                )
            if existing is None:
                subscription = replace(
                    subscription, trusted_for_p2p=True, license_authoritative=True
                )
                if self._provider_snapshot(subscription) is None:
                    raise ResolverSubscriptionError(
                        "official resolver export lacks complete provider snapshot metadata"
                    )
            self._official_bootstrap = replace(configured, subscription_id=subscription.id)
            try:
                if existing is None:
                    return self._add_subscription(subscription)
                # A pre-existing subscription already embodies a user's trust
                # choices. Pin its identity without elevating either flag.
                self._save()
            except Exception:
                self._official_bootstrap = previous
                raise
            return subscription

    def set_p2p_trust(
        self,
        subscription_id: str,
        *,
        trusted_for_p2p: bool,
        license_authoritative: bool,
    ) -> ResolverSubscription:
        if type(trusted_for_p2p) is not bool or type(license_authoritative) is not bool:
            raise ResolverSubscriptionError("resolver P2P trust flags must be booleans")
        with self._lock:
            position = next(
                (
                    position
                    for position, subscription in enumerate(self._subscriptions)
                    if subscription.id == subscription_id
                ),
                None,
            )
            if position is None:
                raise ResolverSubscriptionError(
                    f"unknown resolver index subscription: {subscription_id}"
                )
            previous = self._subscriptions[position]
            updated = replace(
                previous,
                trusted_for_p2p=trusted_for_p2p,
                license_authoritative=license_authoritative,
            )
            self._subscriptions[position] = updated
            try:
                self._save()
            except Exception:
                self._subscriptions[position] = previous
                raise
            return updated

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            position = next(
                (
                    position
                    for position, subscription in enumerate(self._subscriptions)
                    if subscription.id == subscription_id
                ),
                None,
            )
            if position is None:
                return False
            subscription = self._subscriptions[position]
            self._provenance.remove_source(self._source_name(subscription.id))
            self._subscriptions.pop(position)
            try:
                self._save()
            except Exception:
                self._subscriptions.insert(position, subscription)
                self._apply(subscription)
                raise
            self._rebuild_filename_index()
            return True

    def _refresh_one(
        self,
        subscription: ResolverSubscription,
        *,
        now: float,
    ) -> tuple[ResolverSubscription, bool]:
        if (
            subscription.source_type == "url"
            and 0 <= now - subscription.checked_at < self._revalidate_seconds
        ):
            return subscription, False
        try:
            if subscription.source_type == "url":
                data, etag = _fetch_url(
                    subscription.source,
                    subscription.etag if subscription.complete_snapshot else "",
                    timeout=self._fetch_timeout,
                )
                if data is None and not subscription.complete_snapshot:
                    raise ResolverSubscriptionError(
                        "resolver index returned not-modified without a cached complete document"
                    )
                index = subscription.index if data is None else parse_resolver_index(data)
                tombstones = (
                    subscription.p2p_tombstones
                    if data is None
                    else tuple(
                        (digest, now)
                        for digest in sorted(
                            {
                                entry.digest
                                for entry in subscription.index.entries
                                if entry.p2p is not None
                            }
                            - {entry.digest for entry in index.entries if entry.p2p is not None}
                        )
                    )
                )
                refreshed = replace(
                    subscription,
                    index=index,
                    etag=etag,
                    checked_at=now,
                    refreshed_at=now,
                    error="",
                    p2p_tombstones=tombstones,
                    complete_snapshot=True,
                )
            else:
                index = parse_resolver_index(_read_local(subscription.source))
                refreshed = replace(
                    subscription,
                    index=index,
                    checked_at=now,
                    refreshed_at=now,
                    error="",
                    complete_snapshot=True,
                    p2p_tombstones=tuple(
                        (digest, now)
                        for digest in sorted(
                            {
                                entry.digest
                                for entry in subscription.index.entries
                                if entry.p2p is not None
                            }
                            - {entry.digest for entry in index.entries if entry.p2p is not None}
                        )
                    ),
                )
        except (ResolverIndexError, ResolverSubscriptionError) as exc:
            refreshed = replace(
                subscription,
                checked_at=now,
                error=str(exc),
            )
        return refreshed, True

    def refresh(self, subscription_id: str | None = None) -> tuple[dict[str, object], ...]:
        with self._lock:
            if subscription_id is not None and not any(
                subscription.id == subscription_id for subscription in self._subscriptions
            ):
                raise ResolverSubscriptionError(
                    f"unknown resolver index subscription: {subscription_id}"
                )
            now = _timestamp(
                self._clock(),
                "resolver clock returned an invalid timestamp",
            )
            targets = tuple(
                subscription
                for subscription in self._subscriptions
                if subscription_id is None or subscription.id == subscription_id
            )
            official_bootstrap = self._official_bootstrap
            if not targets:
                return ()

            def refresh_target(
                subscription: ResolverSubscription,
            ) -> tuple[ResolverSubscription, ResolverSubscription, bool]:
                refreshed, did_attempt = self._refresh_one(subscription, now=now)
                try:
                    self._require_official_identity(refreshed, official_bootstrap)
                except ResolverSubscriptionError as exc:
                    refreshed = replace(subscription, checked_at=now, error=str(exc))
                return subscription, refreshed, did_attempt

        if len(targets) == 1:
            refreshed_targets = [refresh_target(targets[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(_REFRESH_WORKERS, len(targets))) as executor:
                refreshed_targets = list(executor.map(refresh_target, targets))

        with self._lock:
            current_by_id = {subscription.id: subscription for subscription in self._subscriptions}
            for original in targets:
                if current_by_id.get(original.id) is not original:
                    raise ResolverSubscriptionError(
                        f"resolver index subscription changed during refresh: {original.id}"
                    )
            if self._official_bootstrap != official_bootstrap:
                raise ResolverSubscriptionError(
                    "official resolver bootstrap changed during refresh"
                )
            attempted = [
                refreshed.descriptor()
                for _, refreshed, did_attempt in refreshed_targets
                if did_attempt
            ]
            if not attempted:
                return ()

            subscriptions_before = list(self._subscriptions)
            filename_index_before = self._filename_index
            p2p_cache_before = (
                self._p2p_snapshots_cache,
                self._p2p_declarations_cache,
                self._p2p_declaration_batches,
                self._p2p_active_batch_key,
                self._p2p_active_declarations_cache,
            )
            positions = {
                subscription.id: position
                for position, subscription in enumerate(self._subscriptions)
            }
            applied: list[ResolverSubscription] = []
            try:
                for original, refreshed, did_attempt in refreshed_targets:
                    if not did_attempt:
                        continue
                    applied.append(original)
                    self._apply(refreshed)
                    self._subscriptions[positions[original.id]] = refreshed
                self._save()
                self._rebuild_filename_index()
            except Exception:
                self._subscriptions = subscriptions_before
                try:
                    # Concrete writers keep a failing override from stranding candidate state.
                    for previous in reversed(applied):
                        ProvenanceStore.replace_source(
                            self._provenance,
                            self._source_name(previous.id),
                            self._records(previous),
                        )
                    with contextlib.suppress(Exception):
                        ResolverSubscriptionStore._save(self)
                finally:
                    self._filename_index = filename_index_before
                    (
                        self._p2p_snapshots_cache,
                        self._p2p_declarations_cache,
                        self._p2p_declaration_batches,
                        self._p2p_active_batch_key,
                        self._p2p_active_declarations_cache,
                    ) = p2p_cache_before
                raise
            return tuple(attempted)
