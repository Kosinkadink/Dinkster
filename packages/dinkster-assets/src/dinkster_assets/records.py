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
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .catalog import validate_virtual_path
from .identity import AssetError, digest_bytes, digest_file, require_digest
from .media import MediaClassification, classify_media, classify_media_file

DEFAULT_SCOPE = "local"

_NAME_LIMIT = 512
_LABEL_LIMIT = 128
_LABEL_COUNT_LIMIT = 32


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


class LibraryStore:
    """SQLite-backed scoped record store. Thread-safe: handlers call it
    via asyncio.to_thread, so one connection crosses threads behind a
    lock (sqlite serializes anyway; the lock keeps transactions whole)."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    name TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    created REAL NOT NULL,
                    modified REAL NOT NULL,
                    revision INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS records_scope_modified"
                " ON records(scope, modified DESC, id DESC)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media_grants (
                    scope TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    PRIMARY KEY (scope, digest, kind)
                )
                """
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS asset_dependencies ("
                "digest TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS asset_derivations ("
                "cache_key TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
            )
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
            self._conn.close()

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
