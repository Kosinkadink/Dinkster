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
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
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
from .vault import AssetVault

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


@dataclass(frozen=True)
class AssetRetention:
    """An explicit owner of asset identities, not read authority or a catalog entry."""

    scope: str
    owner_kind: str
    owner_id: str
    revision: int
    digests: tuple[str, ...]
    complete: bool = False
    source_token: str | None = None

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "scope": self.scope,
            "ownerKind": self.owner_kind,
            "ownerId": self.owner_id,
            "revision": self.revision,
            "digests": list(self.digests),
            "complete": self.complete,
        }
        if self.source_token is not None:
            wire["sourceToken"] = self.source_token
        return wire


@dataclass(frozen=True)
class ResultAssetOffer:
    """An immutable identity snapshot, independent of cache and job retirement."""

    token: str
    scope: str
    job_ref: str
    attempt: int
    node_id: str
    output_id: str
    publication: int
    digests: tuple[str, ...]
    complete: bool
    artifact_json: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "token": self.token,
            "scope": self.scope,
            "jobRef": self.job_ref,
            "attemptId": self.attempt,
            "nodeId": self.node_id,
            **(
                {"outputId": self.output_id}
                if self.artifact_json is None
                else {"artifact": json.loads(self.artifact_json)}
            ),
            "publication": self.publication,
            "digests": list(self.digests),
            "complete": self.complete,
        }


def _retention_key(scope: object, owner_kind: object, owner_id: object) -> tuple[str, str, str]:
    scope = _clean_scope(scope)
    if owner_kind not in ("document", "history", "shown-result", "preview"):
        raise AssetError("invalid asset retention owner kind")
    if (
        not isinstance(owner_id, str)
        or not owner_id
        or owner_id != owner_id.strip()
        or len(owner_id) > _NAME_LIMIT
    ):
        raise AssetError("asset retention owner id must be non-empty and at most 512 characters")
    return scope, cast(str, owner_kind), owner_id


def _retention_digests(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(cast(list[object], raw)) > 4096:
        raise AssetError("retained digests must be a list of at most 4096 identities")
    return _canonical_digests(cast(list[object], raw))


def _canonical_digests(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise AssetError("asset references must be a list")
    digests: set[str] = set()
    for digest in cast(list[object], raw):
        if not isinstance(digest, str):
            raise AssetError("retained digest must be a canonical asset identity")
        digests.add(require_digest(digest))
    return tuple(sorted(digests))


def _retention_from_row(row: sqlite3.Row) -> AssetRetention:
    key = _retention_key(row["scope"], row["owner_kind"], row["owner_id"])
    if type(row["revision"]) is not int or row["revision"] < 1:
        raise AssetError("asset retention revision is invalid")
    if type(row["complete"]) is not int or row["complete"] not in (0, 1):
        raise AssetError("asset retention completeness is invalid")
    try:
        digests = _retention_digests(json.loads(row["digests"]))
    except (TypeError, ValueError) as exc:
        raise AssetError("asset retention state is unreadable") from exc
    token = row["source_token"]
    if token is not None:
        _result_offer_token(token)
    return AssetRetention(*key, row["revision"], digests, bool(row["complete"]), token)


def _result_offer_token(token: object) -> str:
    if not isinstance(token, str) or re.fullmatch("[0-9a-f]{32}", token) is None:
        raise AssetError("invalid result offer token")
    return token


def _artifact_offer_json(
    artifact: Mapping[str, object], node_id: str, roots: tuple[str, ...]
) -> str:
    fields = {"nodeId", "digest", "name", "size", "mediaType", "virtualPath", "index"}
    if set(artifact) not in (fields, fields | {"preview"}):
        raise AssetError("artifact offer descriptor fields are invalid")
    index = artifact["index"]
    if artifact["nodeId"] != node_id or type(index) is not int or index < 0:
        raise AssetError("artifact offer source identity is invalid")
    descriptors = [artifact]
    if "preview" in artifact:
        raw = artifact["preview"]
        if not isinstance(raw, Mapping):
            raise AssetError("artifact offer preview descriptor is invalid")
        preview = cast(Mapping[str, object], raw)
        if set(preview) != fields - {"nodeId", "index"}:
            raise AssetError("artifact offer preview descriptor is invalid")
        descriptors.append(preview)
    digests: list[str] = []
    for descriptor in descriptors:
        if any(
            type(descriptor[name]) is not str or not descriptor[name]
            for name in ("name", "mediaType", "virtualPath")
        ) or (type(descriptor["size"]) is not int or descriptor["size"] < 0):
            raise AssetError("artifact offer file descriptor is invalid")
        ref = AssetRef.from_wire(descriptor)
        if (
            descriptor is not artifact
            and ref.virtual_path != f"previews/{ref.digest[7:]}/{ref.name}"
        ):
            raise AssetError("artifact offer preview identity is invalid")
        digests.append(ref.digest)
    if _canonical_digests(digests) != roots:
        raise AssetError("artifact offer identities do not match its descriptor")
    return json.dumps(dict(artifact), sort_keys=True, separators=(",", ":"))


def _result_offer_from_row(row: sqlite3.Row) -> ResultAssetOffer:
    token = _result_offer_token(row["token"])
    for name in ("job_ref", "node_id"):
        if not isinstance(row[name], str) or not row[name]:
            raise AssetError("result offer source identity is invalid")
    for name in ("attempt", "publication"):
        if type(row[name]) is not int or row[name] < 1:
            raise AssetError("result offer source version is invalid")
    for name in ("complete", "claimed"):
        if type(row[name]) is not int or row[name] not in (0, 1):
            raise AssetError("result offer ownership state is invalid")
    try:
        digests = _canonical_digests(json.loads(row["digests"]))
        artifact_json = row["artifact"]
        if artifact_json is not None:
            artifact = json.loads(artifact_json)
            if not isinstance(artifact, dict) or row["output_id"] != "":
                raise AssetError("artifact offer source identity is invalid")
            artifact_json = _artifact_offer_json(
                cast(dict[str, object], artifact), row["node_id"], digests
            )
        elif not isinstance(row["output_id"], str) or not row["output_id"]:
            raise AssetError("result offer output identity is invalid")
    except (TypeError, ValueError) as exc:
        raise AssetError("result offer identities are unreadable") from exc
    return ResultAssetOffer(
        token,
        _clean_scope(row["scope"]),
        row["job_ref"],
        row["attempt"],
        row["node_id"],
        row["output_id"],
        row["publication"],
        digests,
        bool(row["complete"]),
        artifact_json,
    )


class LibraryStore:
    """SQLite-backed scoped record store. Thread-safe: handlers call it
    via asyncio.to_thread, so one connection crosses threads behind a
    lock (sqlite serializes anyway; the lock keeps transactions whole)."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        if not read_only:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        database = Path(path).resolve().as_uri() + "?mode=ro" if read_only else str(path)
        self._engine = sa.create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(database, uri=read_only, check_same_thread=False),
            poolclass=StaticPool,
        )
        try:
            if not read_only:
                with _migration_lock, self._engine.connect() as connection:
                    connection.exec_driver_sql("PRAGMA journal_mode=WAL")
                    connection.commit()
                    config = migration_config()
                    config.attributes["connection"] = connection
                    command.upgrade(config, "head")
                    connection.commit()
            self._database = self._engine.raw_connection()
            self._conn = cast(sqlite3.Connection, self._database.driver_connection)
            self._conn.row_factory = sqlite3.Row
            self._lock = threading.Lock()
            with self._lock, self._conn:
                latest = self._conn.execute("SELECT MAX(modified) FROM records").fetchone()[0]
                self._last_modified = float(latest) if latest is not None else 0.0
        except BaseException:
            self._engine.dispose()
            raise

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

    def cleanup_limit(self) -> int | None:
        """Read the global size policy; null means unlimited, not unknown."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT singleton, limit_bytes FROM asset_cleanup_policy"
            ).fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise AssetError("asset cleanup policy is unavailable")
        limit = rows[0][1]
        if limit is not None and (type(limit) is not int or not 0 < limit <= 2**53 - 1):
            raise AssetError("asset cleanup policy is invalid")
        return limit

    def set_cleanup_limit(self, limit: int | None) -> None:
        """Changing the size policy never grants collection authority."""
        if limit is not None and (type(limit) is not int or not 0 < limit <= 2**53 - 1):
            raise AssetError("limitBytes must be a positive safe integer or null")
        with self._lock, self._conn:
            changed = self._conn.execute(
                "UPDATE asset_cleanup_policy SET limit_bytes = ? WHERE singleton = 1", (limit,)
            ).rowcount
            if changed != 1:
                raise AssetError("asset cleanup policy is unavailable")

    def retain_assets(
        self,
        scope: str,
        owner_kind: str,
        owner_id: str,
        digests: list[str],
        *,
        revision: int,
        complete: bool = False,
    ) -> AssetRetention:
        """Replace an owner's manifest atomically; revision zero creates a new owner.

        Missing bytes are allowed: reopening a document must retain its missing
        identities too, without granting permission to fetch or read them.
        """
        key = _retention_key(scope, owner_kind, owner_id)
        cleaned = _retention_digests(digests)
        if type(revision) is not int or revision < 0:
            raise AssetError("asset retention revision must be a nonnegative integer")
        if type(complete) is not bool:
            raise AssetError("asset retention completeness must be a boolean")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            retained = self._retain_assets_locked(key, cleaned, revision, complete)
        return retained

    def _retain_assets_locked(
        self,
        key: tuple[str, str, str],
        digests: tuple[str, ...],
        revision: int,
        complete: bool,
        source_token: str | None = None,
    ) -> AssetRetention:
        row = self._conn.execute(
            "SELECT * FROM asset_retentions WHERE scope = ? AND owner_kind = ? AND owner_id = ?",
            key,
        ).fetchone()
        current = _retention_from_row(row) if row is not None else None
        if revision != (current.revision if current is not None else 0):
            raise StaleRevision("retained asset values changed")
        retained = AssetRetention(*key, revision + 1, digests, complete, source_token)
        self._conn.execute(
            "INSERT INTO asset_retentions "
            "(scope, owner_kind, owner_id, revision, digests, complete, source_token) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(scope, owner_kind, owner_id) DO UPDATE SET "
            "revision = excluded.revision, digests = excluded.digests, "
            "complete = excluded.complete, source_token = excluded.source_token",
            (*key, retained.revision, json.dumps(digests), int(complete), source_token),
        )
        return retained

    def offer_result_assets(
        self,
        scope: str,
        job_ref: str,
        attempt: int,
        node_id: str,
        output_id: str,
        publication: int,
        digests: list[str],
        *,
        complete: bool,
        artifact: Mapping[str, object] | None = None,
    ) -> ResultAssetOffer:
        """Persist the source's roots before exposing its token to a consumer."""
        scope = _clean_scope(scope)
        if not job_ref or not node_id or (not output_id if artifact is None else output_id != ""):
            raise AssetError("result offer source identity is invalid")
        if any(type(version) is not int or version < 1 for version in (attempt, publication)):
            raise AssetError("result offer source version is invalid")
        if type(complete) is not bool:
            raise AssetError("result offer completeness must be a boolean")
        roots = _canonical_digests(digests)
        artifact_json = (
            _artifact_offer_json(artifact, node_id, roots) if artifact is not None else None
        )
        offer = ResultAssetOffer(
            uuid.uuid4().hex,
            scope,
            job_ref,
            attempt,
            node_id,
            output_id,
            publication,
            roots,
            complete and len(roots) <= 4096,
            artifact_json,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO asset_result_offers "
                "(token, scope, job_ref, attempt, node_id, output_id, "
                "publication, digests, complete, artifact) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    offer.token,
                    scope,
                    job_ref,
                    attempt,
                    node_id,
                    output_id,
                    publication,
                    json.dumps(roots),
                    int(offer.complete),
                    artifact_json,
                ),
            )
        return offer

    def result_asset_offer(self, scope: str, token: str) -> ResultAssetOffer | None:
        scope, token = _clean_scope(scope), _result_offer_token(token)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM asset_result_offers WHERE scope = ? AND token = ?",
                (scope, token),
            ).fetchone()
        return _result_offer_from_row(row) if row is not None else None

    def record_result_rendition(
        self, scope: str, token: str, cache_key: str, document_digest: str, digests: list[str]
    ) -> bool:
        """Bind immutable derived roots without changing the source's own census.

        The caller holds publication ownership until this transaction settles.
        A retained token alone cannot revive a source whose last owner released it.
        """
        scope, token = _clean_scope(scope), _result_offer_token(token)
        document_digest = require_digest(document_digest)
        roots = _canonical_digests(digests)
        if not cache_key:
            raise AssetError("rendition cache identity is invalid")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM asset_result_offers WHERE scope = ? AND token = ?",
                (scope, token),
            ).fetchone()
            if row is None:
                return False
            offer = _result_offer_from_row(row)
            if row["claimed"]:
                owners = [
                    _retention_from_row(owner)
                    for owner in self._conn.execute(
                        "SELECT * FROM asset_retentions WHERE scope = ? AND source_token = ?",
                        (scope, token),
                    )
                ]
                if not any(
                    owner.digests == offer.digests and owner.complete == offer.complete
                    for owner in owners
                ):
                    return False
            existing = self._conn.execute(
                "SELECT document_digest, digests FROM asset_result_renditions "
                "WHERE scope = ? AND source_token = ? AND cache_key = ?",
                (scope, token, cache_key),
            ).fetchone()
            if existing is not None:
                if (
                    existing["document_digest"] != document_digest
                    or _canonical_digests(json.loads(existing["digests"])) != roots
                ):
                    raise StaleRevision("result rendition is already sealed")
            else:
                self._conn.execute(
                    "INSERT INTO asset_result_renditions "
                    "(scope, source_token, cache_key, document_digest, digests) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (scope, token, cache_key, document_digest, json.dumps(roots)),
                )
        return True

    def publish_terminal_result(
        self,
        scope: str,
        job_ref: str,
        attempt: int,
        tokens: Sequence[str],
        *,
        state: str,
        identity: Mapping[str, object],
        artifacts_complete: bool,
    ) -> dict[str, object]:
        """Seal exact publications, never select a latest offer by output identity."""
        scope = _clean_scope(scope)
        if not job_ref or type(attempt) is not int or attempt < 1:
            raise AssetError("terminal result identity is invalid")
        if state not in {"completed", "failed", "cancelled"}:
            raise AssetError("terminal result state is invalid")
        if type(artifacts_complete) is not bool:
            raise AssetError("terminal artifact completeness must be a boolean")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            outputs: dict[str, dict[str, dict[str, object]]] = {}
            artifacts: list[dict[str, object]] = []
            complete = state == "completed" and artifacts_complete
            for token in tokens:
                row = self._conn.execute(
                    "SELECT * FROM asset_result_offers WHERE scope = ? AND token = ?",
                    (scope, _result_offer_token(token)),
                ).fetchone()
                if row is None:
                    raise AssetError("terminal result offer is unavailable")
                offer = _result_offer_from_row(row)
                if offer.job_ref != job_ref or offer.attempt != attempt:
                    raise AssetError("terminal result offer belongs to a different execution")
                if offer.artifact_json is not None:
                    if not artifacts_complete or json.loads(offer.artifact_json)["index"] != len(
                        artifacts
                    ):
                        raise AssetError("terminal artifact publication is ambiguous")
                    artifacts.append(offer.to_wire())
                else:
                    node = outputs.setdefault(offer.node_id, {})
                    if offer.output_id in node:
                        raise AssetError("terminal result publication is ambiguous")
                    node[offer.output_id] = offer.to_wire()
                complete = complete and offer.complete
            result: dict[str, object] = {
                "scope": scope,
                "jobRef": job_ref,
                "attemptId": attempt,
                "state": state,
                "identity": dict(identity),
                "resultOffers": outputs,
                "artifacts": artifacts if artifacts_complete else None,
                "complete": complete,
            }
            manifest = json.dumps(result, sort_keys=True, separators=(",", ":"))
            existing = self._conn.execute(
                "SELECT manifest FROM asset_terminal_results WHERE scope = ? AND job_ref = ?",
                (scope, job_ref),
            ).fetchone()
            if existing is not None:
                if existing["manifest"] != manifest:
                    raise StaleRevision("terminal result publication is already sealed")
            else:
                self._conn.execute(
                    "INSERT INTO asset_terminal_results (scope, job_ref, manifest) "
                    "VALUES (?, ?, ?)",
                    (scope, job_ref, manifest),
                )
        return result

    def terminal_result(self, scope: str, job_ref: str) -> dict[str, object] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT manifest FROM asset_terminal_results WHERE scope = ? AND job_ref = ?",
                (_clean_scope(scope), job_ref),
            ).fetchone()
        if row is None:
            return None
        try:
            raw = json.loads(row["manifest"])
        except (TypeError, ValueError) as exc:
            raise AssetError("terminal result publication is unreadable") from exc
        if not isinstance(raw, dict):
            raise AssetError("terminal result publication is invalid")
        result = cast(dict[str, object], raw)
        if (result.get("scope"), result.get("jobRef")) != (
            scope,
            job_ref,
        ):
            raise AssetError("terminal result publication identity is invalid")
        return result

    def claim_result_assets(
        self,
        scope: str,
        token: str,
        owner_kind: str,
        owner_id: str,
    ) -> AssetRetention | None:
        """Transfer offer protection to a fresh generation in one write transaction.

        A repeated or lost request must reconcile the durable source binding,
        not replace a generation that may already have different consumers.
        """
        key = _retention_key(scope, owner_kind, owner_id)
        if owner_kind not in ("shown-result", "preview"):
            raise AssetError("result offers require a shown-result or preview owner")
        token = _result_offer_token(token)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM asset_result_offers WHERE scope = ? AND token = ?",
                (key[0], token),
            ).fetchone()
            if row is None:
                return None
            offer = _result_offer_from_row(row)
            if row["claimed"]:
                owners = [
                    _retention_from_row(owner)
                    for owner in self._conn.execute(
                        "SELECT * FROM asset_retentions WHERE scope = ? AND source_token = ?",
                        (key[0], token),
                    )
                ]
                # A surviving token is not proof that its former roots still exist.
                if not any(
                    owner.digests == offer.digests and owner.complete == offer.complete
                    for owner in owners
                ):
                    return None
            roots = _retention_digests(list(offer.digests))
            retained = self._retain_assets_locked(key, roots, 0, offer.complete, token)
            self._conn.execute(
                "UPDATE asset_result_offers SET claimed = 1 WHERE token = ?", (token,)
            )
        return retained

    def retained_assets(self, scope: str, owner_kind: str, owner_id: str) -> AssetRetention | None:
        key = _retention_key(scope, owner_kind, owner_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM asset_retentions "
                "WHERE scope = ? AND owner_kind = ? AND owner_id = ?",
                key,
            ).fetchone()
        return None if row is None else _retention_from_row(row)

    def release_assets(
        self, scope: str, owner_kind: str, owner_id: str, *, revision: int
    ) -> AssetRetention | None:
        key = _retention_key(scope, owner_kind, owner_id)
        if type(revision) is not int or revision < 1:
            raise AssetError("asset retention revision must be a positive integer")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM asset_retentions "
                "WHERE scope = ? AND owner_kind = ? AND owner_id = ?",
                key,
            ).fetchone()
            if row is None:
                return None
            if _retention_from_row(row).revision != revision:
                raise StaleRevision("retained asset values changed")
            retained = AssetRetention(*key, revision + 1, (), True)
            # Retain the revision even when empty so reusing an owner cannot
            # let a stale tab release its newer values.
            self._conn.execute(
                "UPDATE asset_retentions "
                "SET revision = ?, digests = '[]', complete = 1, source_token = NULL "
                "WHERE scope = ? AND owner_kind = ? AND owner_id = ?",
                (retained.revision, *key),
            )
        return retained

    @contextmanager
    def hold_asset_references(
        self, additional_roots: Sequence[str] = ()
    ) -> Generator[frozenset[str]]:
        """Inspect catalog and explicit roots in every scope behind their write barrier.

        This excludes other SQLite writers, not cache or history publishers.
        Missing associations for opaque documents cannot establish emptiness.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute("SELECT * FROM asset_retentions").fetchall()
                retained = [_retention_from_row(row) for row in rows]
                if any(not owner.complete for owner in retained):
                    raise AssetError("retained owner manifest is incomplete")
                roots = set(digest for owner in retained for digest in owner.digests)
                active_sources = {
                    (owner.scope, owner.source_token)
                    for owner in retained
                    if owner.source_token is not None
                }
                for row in self._conn.execute("SELECT * FROM asset_result_offers"):
                    offer = _result_offer_from_row(row)
                    if row["claimed"]:
                        continue
                    if not offer.complete:
                        raise AssetError("result offer manifest is incomplete")
                    roots.update(offer.digests)
                    active_sources.add((offer.scope, offer.token))
                for row in self._conn.execute("SELECT * FROM asset_result_renditions"):
                    if (row["scope"], row["source_token"]) in active_sources:
                        try:
                            roots.update(_canonical_digests(json.loads(row["digests"])))
                        except (TypeError, ValueError) as exc:
                            raise AssetError("result rendition references are unavailable") from exc
                roots.update(require_digest(digest) for digest in additional_roots)
                for table in ("records", "media_grants", "asset_observations"):
                    for (digest,) in self._conn.execute(f"SELECT digest FROM {table}"):
                        if not isinstance(digest, str):
                            raise AssetError("catalog reference identity is invalid")
                        roots.add(require_digest(digest))
                for (references,) in self._conn.execute(
                    "SELECT asset_digests FROM asset_derivations"
                ):
                    try:
                        roots.update(_canonical_digests(json.loads(references)))
                    except (TypeError, ValueError) as exc:
                        raise AssetError("derivation asset references are unavailable") from exc
                manifests = {
                    row["digest"]: row["manifest"]
                    for row in self._conn.execute("SELECT digest, manifest FROM asset_dependencies")
                }
                pending = list(roots)
                while pending:
                    digest = pending.pop()
                    if digest not in manifests:
                        continue
                    try:
                        dependencies = json.loads(manifests[digest])
                    except (TypeError, ValueError) as exc:
                        raise AssetError("asset dependency state is unreadable") from exc
                    if not isinstance(dependencies, list):
                        raise AssetError("asset dependencies must be a list")
                    for dependency in cast(list[object], dependencies):
                        if not isinstance(dependency, dict) or not isinstance(
                            child := cast(dict[str, object], dependency).get("digest"), str
                        ):
                            raise AssetError("asset dependency identity is unavailable")
                        require_digest(child)
                        if child not in roots:
                            roots.add(child)
                            pending.append(child)
                yield frozenset(roots)
            except sqlite3.Error as exc:
                raise AssetError("retained asset state is unavailable") from exc
            finally:
                self._conn.rollback()

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

    def backfill_asset_observations(self, vault: AssetVault) -> None:
        """Recover known scoped membership, never assign anonymous bytes to a scope."""
        with self._lock:
            grants = self._conn.execute(
                "SELECT * FROM media_grants WHERE NOT EXISTS "
                "(SELECT 1 FROM asset_observations WHERE "
                "asset_observations.scope = media_grants.scope AND "
                "asset_observations.digest = media_grants.digest)"
            ).fetchall()
            records = self._conn.execute(
                "SELECT * FROM records WHERE NOT EXISTS "
                "(SELECT 1 FROM asset_observations WHERE "
                "asset_observations.scope = records.scope AND "
                "asset_observations.digest = records.digest)"
            ).fetchall()
        candidates: list[tuple[str, AssetRef, str, str, float | None]] = []
        for row in grants:
            grant = _media_grant_from_row(row)
            name = grant.digest.split(":", 1)[1]
            candidates.append(
                (
                    grant.scope,
                    AssetRef(grant.digest, name, grant.byte_size, grant.media_type),
                    grant.kind,
                    f"media:{grant.scope}:{grant.digest}:{grant.kind}",
                    None,
                )
            )
        for row in records:
            name = row["name"]
            if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
                name = row["digest"].split(":", 1)[1]
            candidates.append(
                (
                    row["scope"],
                    AssetRef(row["digest"], name, 0, row["media_type"]),
                    "document/workflow"
                    if row["media_type"] == "application/x-dinkster-workflow+json"
                    else "document/file",
                    f"record:{row['id']}:{row['digest']}",
                    row["created"],
                )
            )
        for scope, ref, kind, identity, created in candidates:
            try:
                with AssetRef(
                    ref.digest, ref.name, ref.size, ref.media_type, resolver=vault
                ).open() as handle:
                    size = handle.seek(0, 2)
                if identity.startswith("media:") and size != ref.size:
                    continue
                observation_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
                with self._lock:
                    if self._conn.execute(
                        "SELECT 1 FROM asset_observations WHERE id = ?", (observation_id,)
                    ).fetchone():
                        continue
                self.record_asset(
                    scope,
                    AssetRef(ref.digest, ref.name, size, ref.media_type),
                    original_name=ref.name,
                    kind=kind,
                    origin={"kind": "unknown"},
                    observation_id=observation_id,
                    created=created,
                )
            except (AssetError, OSError):
                continue

    def unrecorded_vault_assets(
        self,
        vault: AssetVault,
        *,
        text: str = "",
        limit: int = 50,
        after: str | None = None,
    ) -> list[AssetRef]:
        """Operator inventory only: no scoped membership or fabricated ingest history."""
        assets: list[AssetRef] = []
        root = vault.root.resolve()
        with self._lock:
            known = {
                row["digest"]
                for row in self._conn.execute(
                    "SELECT digest FROM records UNION SELECT digest FROM media_grants "
                    "UNION SELECT digest FROM asset_observations"
                ).fetchall()
            }
        for digest in sorted(set(vault.digests()), reverse=True):
            if (
                digest in known
                or (after is not None and digest >= after)
                or text.lower() not in digest
            ):
                continue
            ref = AssetRef(digest, digest.split(":", 1)[1], 0, resolver=vault)
            try:
                path = vault.resolve(digest)
                if path is None or not path.resolve().is_relative_to(root):
                    continue
                with ref.open() as handle:
                    size = os.fstat(handle.fileno()).st_size
            except (AssetError, OSError):
                continue
            assets.append(AssetRef(digest, ref.name, size))
            if len(assets) >= max(1, limit):
                break
        return assets

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

    def put_derivation(
        self, cache_key: str, manifest: Mapping[str, object], *, references: list[str]
    ) -> bool:
        """Persist immutable provenance and its producer-declared external identities atomically."""
        cache_key = require_digest(cache_key)
        encoded = json.dumps(dict(manifest), separators=(",", ":"), sort_keys=True)
        cleaned = _canonical_digests(references)
        with self._lock, self._conn:
            inserted = self._conn.execute(
                "INSERT OR IGNORE INTO asset_derivations (cache_key, manifest, asset_digests) "
                "VALUES (?, ?, ?)",
                (cache_key, encoded, json.dumps(cleaned)),
            )
            row = self._conn.execute(
                "SELECT manifest, asset_digests FROM asset_derivations WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            assert row is not None
            if row["manifest"] != encoded:
                raise AssetError("derivation conflicts with immutable stored manifest")
            if row["asset_digests"] is None:
                self._conn.execute(
                    "UPDATE asset_derivations SET asset_digests = ? WHERE cache_key = ?",
                    (json.dumps(cleaned), cache_key),
                )
            else:
                try:
                    stored = _canonical_digests(json.loads(row["asset_digests"]))
                except (TypeError, ValueError) as exc:
                    raise AssetError("derivation asset references are unavailable") from exc
                if stored != cleaned:
                    raise AssetError("derivation conflicts with immutable asset references")
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
