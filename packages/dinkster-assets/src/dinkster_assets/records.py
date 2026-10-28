"""Scoped library records: the mutable half of workflow persistence.

The vault holds immutable bytes keyed by digest; a library RECORD is the
mutable name pointing at them - display name, labels, folder placement,
timestamps - owned by exactly one SCOPE. "Saving a new version" is upload
new bytes + repoint the record's digest: records are cheap pointers,
digests are immutable history.

Scoping is structural from the first slice (the ComfyUI --multi-user
lesson: bolting tenancy on later means a migration tool nobody ships).
Every operation takes an explicit non-empty scope; single-user mode is
the one reserved scope DEFAULT_SCOPE ("local"), never "scope absent".
Bytes deduplicate globally by digest, but catalog membership is scoped:
a record lookup with the wrong scope is a miss, not a permission error -
knowing a digest or record id grants nothing.

Storage is one SQLite file (stdlib, transactional, atomic) - the first
mutable multi-row store in the codebase, where the accrete-only JSON
stores (provenance, pins) would race on concurrent update. Writes take a
process-wide lock; readers ride WAL. Optimistic concurrency: every
record carries a revision, updates must present it, and a stale revision
raises instead of silently last-writer-wins - two tabs editing one
record is the normal case, not the weird one.

Listing is query-first and cursor-paged (the frontend collection
contract): the cursor binds the query that produced it, so a changed
query invalidates the cursor instead of silently paging wrong results.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import sqlalchemy as sa
from alembic import command
from sqlalchemy.pool import StaticPool

from .catalog import validate_virtual_path
from .identity import AssetError, digest_bytes, digest_file, require_digest
from .kind import require_asset_kind
from .media import MediaClassification, classify_media, classify_media_file
from .model import AssetRef
from .schema import migration_config

DEFAULT_SCOPE = "local"

_NAME_LIMIT = 512
_LABEL_LIMIT = 128
_LABEL_COUNT_LIMIT = 32
# Alembic's command API installs process-global environment proxies.
_migration_lock = threading.Lock()


class StaleRevision(AssetError):
    """The record changed since the caller read it: retry with the
    current revision (optimistic concurrency, never last-writer-wins)."""


# The _clean_* validators take ``object`` on purpose: they are the trust
# boundary for anything reaching the store, so the isinstance checks are
# load-bearing even though typed callers already pass str.


def _clean_scope(scope: object) -> str:
    if not isinstance(scope, str) or not scope.strip():
        raise AssetError("scope must be a non-empty string")
    return scope.strip()


def _clean_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise AssetError("name must be a non-empty string")
    if len(name) > _NAME_LIMIT:
        raise AssetError(f"name exceeds {_NAME_LIMIT} characters")
    return name.strip()


def _clean_media_type(media_type: object) -> str:
    if not isinstance(media_type, str) or not media_type.strip():
        raise AssetError("mediaType must be a non-empty string")
    return media_type.strip()


def _clean_byte_size(byte_size: object) -> int:
    if isinstance(byte_size, bool) or not isinstance(byte_size, int):
        raise AssetError("media grant byte size must be an integer")
    if byte_size <= 0:
        raise AssetError("media grant byte size must be positive")
    return byte_size


def _clean_labels(labels: Sequence[object]) -> tuple[str, ...]:
    if len(labels) > _LABEL_COUNT_LIMIT:
        raise AssetError(f"at most {_LABEL_COUNT_LIMIT} labels per record")
    cleaned: dict[str, None] = {}
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise AssetError("labels must be non-empty strings")
        if len(label) > _LABEL_LIMIT:
            raise AssetError(f"label exceeds {_LABEL_LIMIT} characters")
        cleaned[label.strip()] = None
    return tuple(cleaned)


def _clean_folder(folder: object) -> str:
    if not isinstance(folder, str):
        raise AssetError("folder must be a string")
    return validate_virtual_path(folder) if folder else ""


@dataclass(frozen=True)
class LibraryRecord:
    """One scoped, mutable pointer at immutable bytes."""

    id: str
    scope: str
    name: str
    digest: str
    media_type: str
    labels: tuple[str, ...] = ()
    folder: str = ""
    created: float = 0.0
    modified: float = 0.0
    revision: int = 1

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "id": self.id,
            "scope": self.scope,
            "name": self.name,
            "digest": self.digest,
            "mediaType": self.media_type,
            "labels": list(self.labels),
            "created": self.created,
            "modified": self.modified,
            "revision": self.revision,
        }
        if self.folder:
            wire["folder"] = self.folder
        return wire


@dataclass(frozen=True)
class MediaGrant:
    """Immutable authority for one classified digest in exactly one scope."""

    scope: str
    digest: str
    kind: str
    media_type: str
    extension: str
    byte_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _clean_scope(self.scope))
        object.__setattr__(self, "digest", require_digest(self.digest))
        if (self.kind, self.media_type, self.extension) != (
            "data/latent",
            "application/x-comfy-latent",
            "latent",
        ):
            MediaClassification(self.kind, self.media_type, self.extension)
        object.__setattr__(self, "byte_size", _clean_byte_size(self.byte_size))


@dataclass(frozen=True)
class AssetObservation:
    """Scoped ingest history, not authority to resolve its observed location."""

    id: str
    scope: str
    asset: AssetRef
    original_name: str
    kind: str
    origin: Mapping[str, str]
    project: str
    created: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", MappingProxyType(dict(self.origin)))

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "asset": self.asset.to_wire(),
            "originalName": self.original_name,
            "kind": self.kind,
            "origin": dict(self.origin),
            "project": self.project,
            "created": self.created,
        }


class LibraryStore:
    """SQLite-backed scoped record store. Thread-safe: handlers call it
    via asyncio.to_thread, so one connection crosses threads behind a
    lock (sqlite serializes anyway; the lock keeps transactions whole)."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = sa.create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(str(path), check_same_thread=False),
            poolclass=StaticPool,
        )
        try:
            with _migration_lock, self._engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA journal_mode=WAL")
                connection.commit()
                config = migration_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
                connection.commit()
        except BaseException:
            self._engine.dispose()
            raise
        self._database = self._engine.raw_connection()
        self._conn = cast(sqlite3.Connection, self._database.driver_connection)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            latest = self._conn.execute("SELECT MAX(modified) FROM records").fetchone()[0]
            self._last_modified = float(latest) if latest is not None else 0.0

    def _modified_now(self) -> float:
        latest = self._conn.execute("SELECT MAX(modified) FROM records").fetchone()[0]
        if latest is not None:
            self._last_modified = max(self._last_modified, float(latest))
        now = time.time()
        if now <= self._last_modified:
            now = math.nextafter(self._last_modified, math.inf)
        self._last_modified = now
        return now

    def close(self) -> None:
        with self._lock:
            self._database.close()
            self._engine.dispose()

    def record_asset(
        self,
        scope: str,
        asset: AssetRef,
        *,
        original_name: str,
        kind: str,
        origin: Mapping[str, str],
        project: str = "default",
        observation_id: str | None = None,
        created: float | None = None,
    ) -> AssetObservation:
        scope = _clean_scope(scope)
        if not original_name.strip():
            raise AssetError("asset name must be a non-empty string")
        original_name = original_name.strip()
        require_asset_kind(kind)
        if any(
            not name or name in {".", ".."} or "/" in name or "\\" in name
            for name in (original_name, asset.name)
        ):
            raise AssetError("asset names must be basenames")
        if type(asset.size) is not int or asset.size < 0:
            raise AssetError("asset byte size must be a nonnegative integer")
        if asset.virtual_path:
            validate_virtual_path(asset.virtual_path)
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", project) is None:
            raise AssetError("project must be a lowercase project id")
        details = dict(origin)
        fields = {
            "upload": set[str](),
            "download": {"url"},
            "transfer": {"machine"},
            "produced": {"job", "node"},
            "unknown": set[str](),
        }.get(details.get("kind", ""))
        if (
            fields is None
            or set(details) != fields | {"kind"}
            or any(not value.strip() for value in details.values())
        ):
            raise AssetError("invalid asset origin")
        if details["kind"] == "download":
            url = urlsplit(details["url"])
            if url.scheme not in {"https", "http"} or not url.hostname:
                raise AssetError("download origin requires an HTTP URL")
            details["url"] = urlunsplit(
                (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")
            )
        if observation_id is not None and re.fullmatch(r"[0-9a-f]{32}", observation_id) is None:
            raise AssetError("invalid observation id")
        if created is not None and (not math.isfinite(created) or created < 0):
            raise AssetError("invalid observation time")
        observation = AssetObservation(
            observation_id or uuid.uuid4().hex,
            scope,
            asset,
            original_name,
            kind,
            details,
            project,
            time.time() if created is None else created,
        )
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM asset_observations WHERE id = ?", (observation.id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["scope"] != scope
                    or existing["digest"] != asset.digest
                    or existing["original_name"] != original_name
                    or existing["stored_name"] != asset.name
                    or existing["kind"] != kind
                    or existing["media_type"] != asset.media_type
                    or existing["byte_size"] != asset.size
                    or existing["virtual_path"] != asset.virtual_path
                    or json.loads(existing["origin"]) != details
                    or existing["project"] != project
                    or existing["created"] != observation.created
                ):
                    raise AssetError("observation id already describes another publication")
                return observation
            self._conn.execute(
                "INSERT INTO asset_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    observation.id,
                    scope,
                    asset.digest,
                    original_name,
                    asset.name,
                    kind,
                    asset.media_type,
                    asset.size,
                    asset.virtual_path,
                    json.dumps(details),
                    project,
                    observation.created,
                ),
            )
        return observation

    def query_assets(
        self,
        scope: str,
        *,
        project: str = "default",
        digest: str = "",
        text: str = "",
        kind: str = "",
        origin: str = "",
        since: float | None = None,
        until: float | None = None,
        limit: int = 50,
        after: tuple[float, str] | None = None,
    ) -> list[tuple[AssetObservation, ...]]:
        """Page content identities while retaining every matching project's alias/origin."""
        scope = _clean_scope(scope)
        clauses = ["scope = ?", "project = ?"]
        parameters: list[object] = [scope, project]
        if digest:
            clauses.append("digest = ?")
            parameters.append(require_digest(digest))
        matching = [
            "matched.scope = asset_observations.scope",
            "matched.project = asset_observations.project",
            "matched.digest = asset_observations.digest",
        ]
        if text:
            matching.append(
                "(instr(lower(matched.original_name), lower(?)) > 0 OR "
                "instr(lower(matched.stored_name), lower(?)) > 0 OR "
                "EXISTS (SELECT 1 FROM json_each(matched.origin) "
                "WHERE instr(lower(json_each.value), lower(?)) > 0))"
            )
            parameters.extend((text, text, text))
        if kind:
            matching.append("matched.kind = ?")
            parameters.append(require_asset_kind(kind))
        if origin:
            matching.append("json_extract(matched.origin, '$.kind') = ?")
            parameters.append(origin)
        if since is not None:
            matching.append("matched.created >= ?")
            parameters.append(since)
        if until is not None:
            matching.append("matched.created <= ?")
            parameters.append(until)
        clauses.append(
            "EXISTS (SELECT 1 FROM asset_observations matched WHERE " + " AND ".join(matching) + ")"
        )
        having = ""
        if after is not None:
            having = " HAVING (MAX(created) < ? OR (MAX(created) = ? AND digest < ?))"
            parameters.extend((after[0], after[0], after[1]))
        parameters.append(max(1, limit))
        with self._lock, self._conn:
            self._conn.execute("BEGIN")
            selected = self._conn.execute(
                "SELECT digest, MAX(created) AS latest FROM asset_observations WHERE "
                + " AND ".join(clauses)
                + " GROUP BY digest"
                + having
                + " ORDER BY latest DESC, digest DESC LIMIT ?",
                parameters,
            ).fetchall()
            if not selected:
                return []
            digests = [row["digest"] for row in selected]
            rows = self._conn.execute(
                "SELECT * FROM asset_observations WHERE scope = ? AND project = ? AND digest IN ("
                + ",".join("?" for _ in digests)
                + ") ORDER BY created, id",
                [scope, project, *digests],
            ).fetchall()
        grouped: dict[str, list[AssetObservation]] = {digest: [] for digest in digests}
        for row in rows:
            asset = AssetRef(
                row["digest"],
                row["stored_name"],
                row["byte_size"],
                row["media_type"],
                row["virtual_path"],
            )
            grouped[asset.digest].append(
                AssetObservation(
                    row["id"],
                    row["scope"],
                    asset,
                    row["original_name"],
                    row["kind"],
                    json.loads(row["origin"]),
                    row["project"],
                    row["created"],
                )
            )
        return [tuple(grouped[digest]) for digest in digests]

    def put_dependencies(self, digest: str, dependencies: Sequence[Mapping[str, object]]) -> bool:
        """Persist one immutable, content-derived dependency manifest."""
        digest = require_digest(digest)
        manifest = json.dumps(list(dependencies), separators=(",", ":"), sort_keys=True)
        with self._lock, self._conn:
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO asset_dependencies VALUES (?, ?)", (digest, manifest)
            )
            row = self._conn.execute(
                "SELECT manifest FROM asset_dependencies WHERE digest = ?", (digest,)
            ).fetchone()
        assert row is not None
        if row["manifest"] != manifest:
            raise AssetError("dependency manifest conflicts with immutable stored manifest")
        return inserted.rowcount == 1

    def get_dependencies(self, digest: str) -> list[dict[str, object]] | None:
        digest = require_digest(digest)
        with self._lock:
            row = self._conn.execute(
                "SELECT manifest FROM asset_dependencies WHERE digest = ?", (digest,)
            ).fetchone()
        return None if row is None else json.loads(row["manifest"])

    def put_derivation(self, cache_key: str, manifest: Mapping[str, object]) -> bool:
        """Persist an immutable content-derived result under its complete cache key."""
        cache_key = require_digest(cache_key)
        encoded = json.dumps(dict(manifest), separators=(",", ":"), sort_keys=True)
        with self._lock, self._conn:
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO asset_derivations VALUES (?, ?)", (cache_key, encoded)
            )
            row = self._conn.execute(
                "SELECT manifest FROM asset_derivations WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        assert row is not None
        if row["manifest"] != encoded:
            raise AssetError("derivation conflicts with immutable stored manifest")
        return inserted.rowcount == 1

    def get_derivation(self, cache_key: str) -> dict[str, object] | None:
        cache_key = require_digest(cache_key)
        with self._lock:
            row = self._conn.execute(
                "SELECT manifest FROM asset_derivations WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return None if row is None else json.loads(row["manifest"])

    def create(
        self,
        scope: str,
        name: str,
        digest: str,
        media_type: str,
        *,
        labels: Sequence[str] = (),
        folder: str = "",
    ) -> LibraryRecord:
        scope = _clean_scope(scope)
        name = _clean_name(name)
        digest = require_digest(digest)
        media_type = _clean_media_type(media_type)
        clean_labels = _clean_labels(labels)
        folder = _clean_folder(folder)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            now = self._modified_now()
            record = LibraryRecord(
                id=uuid.uuid4().hex,
                scope=scope,
                name=name,
                digest=digest,
                media_type=media_type,
                labels=clean_labels,
                folder=folder,
                created=now,
                modified=now,
                revision=1,
            )
            self._conn.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.scope,
                    record.name,
                    record.digest,
                    record.media_type,
                    json.dumps(list(record.labels)),
                    record.folder,
                    record.created,
                    record.modified,
                    record.revision,
                ),
            )
        return record

    def get(self, scope: str, record_id: str) -> LibraryRecord | None:
        """Scoped lookup: a wrong scope is a miss, never a hint that the
        id exists elsewhere."""
        scope = _clean_scope(scope)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM records WHERE id = ? AND scope = ?",
                (record_id, scope),
            ).fetchone()
        return _from_row(row) if row is not None else None

    def grant_media(self, grant: MediaGrant, data: bytes) -> MediaGrant:
        """Persist exact scoped media authority once.

        Every persisted fact is re-derived from the supplied bytes. Repeating
        identical authority is idempotent. The primary key never updates or
        repoints: any differing fact for that key refuses.
        """
        return self.grant_media_result(grant, data)[0]

    def grant_media_result(self, grant: MediaGrant, data: bytes | Path) -> tuple[MediaGrant, bool]:
        """Persist exact authority and atomically report whether it was new."""
        if isinstance(data, Path):
            classification = classify_media_file(data)
            digest = digest_file(data)
            byte_size = data.stat().st_size
        else:
            classification = classify_media(data)
            digest = digest_bytes(data)
            byte_size = len(data)
        derived = MediaGrant(
            scope=grant.scope,
            digest=digest,
            kind=classification.kind,
            media_type=classification.media_type,
            extension=classification.extension,
            byte_size=byte_size,
        )
        if derived != grant:
            raise AssetError("media grant facts do not match supplied bytes")
        with self._lock, self._conn:
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO media_grants VALUES (?, ?, ?, ?, ?, ?)",
                (
                    grant.scope,
                    grant.digest,
                    grant.kind,
                    grant.media_type,
                    grant.extension,
                    grant.byte_size,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM media_grants WHERE scope = ? AND digest = ? AND kind = ?",
                (grant.scope, grant.digest, grant.kind),
            ).fetchone()
        assert row is not None
        stored = _media_grant_from_row(row)
        if stored != grant:
            raise AssetError(
                "media grant conflict: immutable authority already exists "
                "for this scope, digest, and kind"
            )
        return stored, inserted.rowcount == 1

    def grant_latent_result(self, grant: MediaGrant) -> tuple[MediaGrant, bool]:
        """Persist strict-parser-derived latent authority."""
        if (
            grant.kind != "data/latent"
            or grant.media_type != "application/x-comfy-latent"
            or grant.extension != "latent"
            or grant.byte_size <= 0
        ):
            raise AssetError("invalid latent grant")
        require_digest(grant.digest)
        with self._lock, self._conn:
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO media_grants VALUES (?, ?, ?, ?, ?, ?)",
                (
                    grant.scope,
                    grant.digest,
                    grant.kind,
                    grant.media_type,
                    grant.extension,
                    grant.byte_size,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM media_grants WHERE scope = ? AND digest = ? AND kind = ?",
                (grant.scope, grant.digest, grant.kind),
            ).fetchone()
        assert row is not None
        stored = _media_grant_from_row(row)
        if stored != grant:
            raise AssetError("latent grant conflicts with immutable scoped authority")
        return stored, inserted.rowcount == 1

    def media_grant(self, scope: str, digest: str, kind: str) -> MediaGrant | None:
        """Exact scoped lookup; a wrong scope is a miss and digest alone is no key."""
        scope = _clean_scope(scope)
        digest = require_digest(digest)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM media_grants WHERE scope = ? AND digest = ? AND kind = ?",
                (scope, digest, kind),
            ).fetchone()
        return _media_grant_from_row(row) if row is not None else None

    def update(
        self,
        scope: str,
        record_id: str,
        revision: int,
        *,
        name: str | None = None,
        digest: str | None = None,
        media_type: str | None = None,
        labels: Sequence[str] | None = None,
        folder: str | None = None,
    ) -> LibraryRecord | None:
        """Repoint/rename one record. None fields keep their value. Returns
        None for an unknown (or out-of-scope) record; raises StaleRevision
        when the presented revision is not current."""
        scope = _clean_scope(scope)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM records WHERE id = ? AND scope = ?",
                (record_id, scope),
            ).fetchone()
            if row is None:
                return None
            current = _from_row(row)
            if revision != current.revision:
                raise StaleRevision(
                    f"record {record_id} is at revision {current.revision}, not {revision}"
                )
            updated = LibraryRecord(
                id=current.id,
                scope=current.scope,
                name=_clean_name(name) if name is not None else current.name,
                digest=require_digest(digest) if digest is not None else current.digest,
                media_type=(
                    _clean_media_type(media_type) if media_type is not None else current.media_type
                ),
                labels=(_clean_labels(labels) if labels is not None else current.labels),
                folder=_clean_folder(folder) if folder is not None else current.folder,
                created=current.created,
                modified=self._modified_now(),
                revision=current.revision + 1,
            )
            self._conn.execute(
                "UPDATE records SET name = ?, digest = ?, media_type = ?,"
                " labels = ?, folder = ?, modified = ?, revision = ?"
                " WHERE id = ?",
                (
                    updated.name,
                    updated.digest,
                    updated.media_type,
                    json.dumps(list(updated.labels)),
                    updated.folder,
                    updated.modified,
                    updated.revision,
                    updated.id,
                ),
            )
        return updated

    def delete(self, scope: str, record_id: str) -> bool:
        """Remove the record - never the bytes (records are cheap pointers;
        vault GC of unreferenced blobs is a separate, later concern)."""
        scope = _clean_scope(scope)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM records WHERE id = ? AND scope = ?",
                (record_id, scope),
            )
        return cursor.rowcount > 0

    def query(
        self,
        scope: str,
        *,
        text: str = "",
        label: str = "",
        limit: int = 50,
        after: tuple[float, str] | None = None,
    ) -> list[LibraryRecord]:
        """Newest-modified first, keyset-paged: ``after`` is the (modified,
        id) of the previous page's last record. Fetch limit+1 to learn
        whether a next page exists."""
        scope = _clean_scope(scope)
        clauses = ["scope = ?"]
        params: list[object] = [scope]
        if text:
            clauses.append("instr(lower(name), lower(?)) > 0")
            params.append(text)
        if label:
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(records.labels) WHERE json_each.value = ?)"
            )
            params.append(label)
        if after is not None:
            clauses.append("(modified < ? OR (modified = ? AND id < ?))")
            params.extend((after[0], after[0], after[1]))
        sql = (
            "SELECT * FROM records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY modified DESC, id DESC LIMIT ?"
        )
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]


def _from_row(row: sqlite3.Row) -> LibraryRecord:
    return LibraryRecord(
        id=row["id"],
        scope=row["scope"],
        name=row["name"],
        digest=row["digest"],
        media_type=row["media_type"],
        labels=tuple(json.loads(row["labels"])),
        folder=row["folder"],
        created=row["created"],
        modified=row["modified"],
        revision=row["revision"],
    )


def _media_grant_from_row(row: sqlite3.Row) -> MediaGrant:
    return MediaGrant(
        scope=row["scope"],
        digest=row["digest"],
        kind=row["kind"],
        media_type=row["media_type"],
        extension=row["extension"],
        byte_size=row["byte_size"],
    )
