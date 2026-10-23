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
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from http.client import HTTPConnection, HTTPException, HTTPMessage, HTTPResponse, HTTPSConnection
from pathlib import Path
from typing import IO, cast
from urllib.parse import quote, unquote_plus, urlsplit, urlunsplit

from .component_manifest import AssetComponentManifest
from .identity import AssetError
from .json_metadata import thaw_json
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
    # Persisted rows may contain fields from capabilities this host no longer uses.
    subscription_id = wire.get("id")
    source = wire.get("source")
    source_type = wire.get("sourceType")
    etag = wire.get("etag", "")
    checked_at = wire.get("checkedAt", 0.0)
    error = wire.get("error", "")
    refreshed_at = wire.get("refreshedAt", checked_at if not error else 0.0)
    complete_snapshot = wire.get("completeSnapshot", False)
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
    if type(complete_snapshot) is not bool:
        raise ResolverSubscriptionError("persisted resolver completeSnapshot must be a boolean")
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
        tmp = self._path.with_name(self._path.name + f".tmp-{os.getpid()}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(encoded)
            os.replace(tmp, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

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
            self._official_bootstrap = replace(configured, subscription_id=subscription.id)
            try:
                if existing is None:
                    return self._add_subscription(subscription)
                self._save()
            except Exception:
                self._official_bootstrap = previous
                raise
            return subscription

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
                refreshed = replace(
                    subscription,
                    index=index,
                    etag=etag,
                    checked_at=now,
                    refreshed_at=now,
                    error="",
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
                raise
            return tuple(attempted)
