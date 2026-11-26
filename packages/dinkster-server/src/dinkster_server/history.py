"""Persistent execution history: terminal jobs, on disk, scope-keyed
(DESIGN roadmap: "Source-document references on jobs + persistent
history" + the identity/scoping constraint).

The queue's in-memory history is a polling grace window (a client that
missed a job_state event can still fetch the result for a while); THIS
is the durable record. Every job that reaches a terminal state is
written down with its stable per-run identity (runId), its job identity
(clientId/jobId), timings, terminal state, error payload, run summary
counts, and - the point of the exercise - the execution-opaque
``sourceDocument`` digest naming the exact workflow asset that produced
it, with ``workflowName``, the name that workflow had when the run was
submitted (omitted when none was sent, as for runs recorded before it).
Since environment stamps live INSIDE documents, persisting the document
(library) plus this link answers "what exact document and
environment produced this run" with no second channel.

Output values live in the cache and are served by the peek surface with
its own retention and refusal vocabulary. History retains only their
external asset identities, not payloads, so clearing caches does not erase
the assets referenced by a durable run record. List and detail responses expose
an optional ``thumbnail`` asset reference for image/video output, preferring
an existing saved preview; runs without media omit it.

The store is also the queue's durability hook (QueuePersistence): every
accepted job gets a row in the ``accepted`` table before the submission
is acknowledged, updated when it starts and retired after immutable
terminal history is committed. Legacy stores share one transaction;
separate accepted storage commits history before retiring engine work.
Rows left behind by a process that died mid-queue are swept at the next startup by
recover_interrupted() into history records with state ``interrupted`` -
durable visibility, zero automatic re-execution: an interrupted job runs
again only when a human resubmits it (user ruling on issue #556).

Scoping is structural: every job is stamped with its selected scope,
principal, and principal kind at acceptance, and terminal history preserves
those facts. History wire records expose them as ``principalId`` and
``principalKind``.

Legacy storage uses WAL and a process-wide write lock; sync methods are
crossed via asyncio.to_thread. Helper-backed stores keep only accepted work
in an engine-owned file. Legacy callers can separate it or keep the shared
file. Terminal rows are immutable once written; re-recording a runId leaves
the first row unchanged.

Surface (present only when create_app got a HistoryStore):

- GET /api/history?scope=&clientId=&sourceDocument=&state=&limit=&cursor=
                                        query-first, newest-finished
                                        first; the cursor binds the query
                                        (400 on mismatch)
- DELETE /api/history?scope=&clientId=&sourceDocument=&state=&before=
                                        bulk clear, same filters as the
                                        list plus finished-before; no
                                        filters clears the whole scope;
                                        -> {"deleted": n}
- GET /api/history/{runId}?scope=       one record; wrong scope is 404
- DELETE /api/history/{runId}?scope=    remove one record (404 on miss/
                                        wrong scope); asset bytes are
                                        never touched
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, LibraryStore, require_digest
from dinkster_assets.value import value_asset_references
from dinkster_library import LibraryClient, LibraryError

from .auth import LOCAL_PRINCIPAL, principal_for, resolve_scope
from .paging import decode_cursor, encode_cursor
from .queue import TERMINAL_STATES, Job

__all__ = [
    "HISTORY_KEY",
    "HistoryRecord",
    "HistoryStore",
    "add_history_routes",
    "record_from_job",
]


@dataclass(frozen=True)
class HistoryRecord:
    """One terminal run, as durably recorded."""

    run_id: str
    scope: str
    client_id: str
    job_id: str
    state: str  # terminal JobState, or "interrupted" for crash leftovers
    priority: int
    submitted_at: float
    started_at: float | None
    finished_at: float
    principal_id: str = "local"
    principal_kind: str = "human"
    attempt: int = 1
    source_document: str = ""  # "" = unset (omitted on the wire, never null)
    workflow_name: str = field(default="", kw_only=True)
    """The workflow's name when the run was submitted; "" = none was recorded."""
    error: dict[str, Any] | None = None
    executed: int = 0
    cached: int = 0
    skipped: int = 0
    node_receipts: tuple[dict[str, str], ...] = ()
    thumbnail: AssetRef | None = field(default=None, kw_only=True)
    asset_digests: tuple[str, ...] | None = field(default=(), kw_only=True)
    """External identities only; None means a legacy row has unknown references."""
    automatic: tuple[dict[str, object], ...] = field(default=(), kw_only=True)
    """Each automatic field the run used, on its wire: the value and whether it
    was worked out or was the default."""

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "jobRef": self.run_id,  # canonical name; runId is the legacy alias
            "runId": self.run_id,
            "scope": self.scope,
            "principalId": self.principal_id,
            "principalKind": self.principal_kind,
            "clientId": self.client_id,
            "jobId": self.job_id,
            "state": self.state,
            "priority": self.priority,
            "attemptId": self.attempt,
            "submittedAt": self.submitted_at,
            "finishedAt": self.finished_at,
            "executed": self.executed,
            "cached": self.cached,
            "skipped": self.skipped,
            "nodeReceipts": [dict(receipt) for receipt in self.node_receipts],
        }
        if self.started_at is not None:
            wire["startedAt"] = self.started_at
        if self.source_document:
            wire["sourceDocument"] = self.source_document
        if self.workflow_name:
            wire["workflowName"] = self.workflow_name
        if self.error is not None:
            wire["error"] = self.error
        if self.automatic:
            wire["automaticValues"] = [dict(item) for item in self.automatic]
        if self.thumbnail is not None:
            wire["thumbnail"] = self.thumbnail.to_wire()
        return wire


def record_from_job(job: Job) -> HistoryRecord:
    """The durable facts of a terminal job. Counts summarize the run
    (full outputs stay in the cache/peek system, never in history)."""
    if job.state not in TERMINAL_STATES:
        raise ValueError(f"job {job.key} is {job.state}, not terminal")
    references: dict[str, AssetRef] = {}
    thumbnail: AssetRef | None = None
    if job.result is not None:
        for outputs in job.result.outputs.values():
            references.update(value_asset_references(outputs))
        for artifact in job.result.artifacts:
            for item in (artifact.preview, artifact):
                if item is None:
                    continue
                ref = AssetRef(
                    item.digest, item.name, item.size, item.media_type, item.virtual_path
                )
                references[ref.digest] = ref
                if thumbnail is None and ref.media_type.startswith(("image/", "video/")):
                    thumbnail = ref
        if thumbnail is None:
            thumbnail = next(
                (
                    ref
                    for ref in references.values()
                    if ref.media_type.startswith(("image/", "video/"))
                ),
                None,
            )
    node_receipts: list[dict[str, str]] = []
    for node_id in sorted(job.node_receipts):
        receipt = dict(job.node_receipts[node_id])
        if receipt["disposition"] == "running":
            receipt["disposition"] = "interrupted"
        node_receipts.append(receipt)
    return HistoryRecord(
        run_id=job.run_id,
        scope=job.scope,
        principal_id=job.principal_id,
        principal_kind=job.principal_kind,
        client_id=job.key.client_id,
        job_id=job.key.job_id,
        state=job.state,
        priority=job.priority,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at if job.finished_at is not None else 0.0,
        attempt=job.attempt,
        source_document=job.source_document,
        workflow_name=job.workflow_name,
        error=job.error,
        executed=len(job.result.executed) if job.result is not None else 0,
        cached=len(job.result.cached) if job.result is not None else 0,
        skipped=len(job.result.skipped) if job.result is not None else 0,
        node_receipts=tuple(node_receipts),
        automatic=tuple(record.to_wire() for record in job.automatic),
        thumbnail=thumbnail,
        asset_digests=tuple(sorted(references)),
    )


def _asset_digests(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise AssetError("history asset references are unavailable")
    digests: set[str] = set()
    for digest in cast(list[object], raw):
        if not isinstance(digest, str):
            raise AssetError("history asset reference is invalid")
        digests.add(require_digest(digest))
    return tuple(sorted(digests))


def _published_asset_digests(
    library: LibraryStore | None, scope: str, run_id: str, attempt: int
) -> tuple[str, ...] | None:
    if library is None:
        return None
    try:
        result = library.terminal_result(scope, run_id)
        if (
            result is None
            or result.get("complete") is not True
            or result.get("attemptId") != attempt
        ):
            return None
        outputs = cast(dict[str, dict[str, dict[str, object]]], result["resultOffers"])
        artifacts = cast(list[dict[str, object]], result["artifacts"])
        offers = [offer for node in outputs.values() for offer in node.values()] + artifacts
        roots: set[str] = set()
        for offer in offers:
            if (
                offer.get("complete") is not True
                or offer.get("scope") != scope
                or offer.get("jobRef") != run_id
                or offer.get("attemptId") != attempt
            ):
                return None
            roots.update(_asset_digests(offer["digests"]))
        return tuple(sorted(roots))
    except (AssetError, KeyError, TypeError, AttributeError):
        return None


class HistoryStore:
    """Scoped terminal history and optionally separate engine accepted work.

    Store operations share one lock; callers cross via asyncio.to_thread.
    """

    def __init__(
        self, path: Path | str | LibraryClient, *, accepted_path: Path | None = None
    ) -> None:
        self._helper = path if isinstance(path, LibraryClient) else None
        self._lock = threading.Lock()
        self._invalidate_asset_references: Callable[[], None] | None = None
        if self._helper is not None:
            if "history" not in cast("list[str]", self._helper.hello["capabilities"]):
                raise ValueError("library helper does not support history")
            if accepted_path is None:
                raise ValueError("helper history requires engine-owned accepted storage")
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(accepted_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._accepted_conn = self._conn
            self._create_accepted()
            return
        assert not isinstance(path, LibraryClient)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._accepted_conn = self._conn
        if accepted_path is not None and accepted_path.resolve() != Path(path).resolve():
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            self._accepted_conn = sqlite3.connect(str(accepted_path), check_same_thread=False)
            self._accepted_conn.row_factory = sqlite3.Row
            self._accepted_conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    run_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    principal TEXT NOT NULL DEFAULT 'local',
                    principal_kind TEXT NOT NULL DEFAULT 'human',
                    client_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    submitted REAL NOT NULL,
                    started REAL,
                    finished REAL NOT NULL,
                    source_document TEXT NOT NULL,
                    error TEXT,
                    executed INTEGER NOT NULL,
                    cached INTEGER NOT NULL,
                    skipped INTEGER NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    node_receipts TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(history)")}
            if "attempt" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1"
                )
            if "principal" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN principal TEXT NOT NULL DEFAULT 'local'"
                )
            if "principal_kind" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN principal_kind TEXT NOT NULL DEFAULT 'human'"
                )
            if "node_receipts" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN node_receipts TEXT NOT NULL DEFAULT '[]'"
                )
            if "automatic" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN automatic TEXT NOT NULL DEFAULT '[]'"
                )
            if "workflow_name" not in columns:
                self._conn.execute(
                    "ALTER TABLE history ADD COLUMN workflow_name TEXT NOT NULL DEFAULT ''"
                )
            if "asset_digests" not in columns:
                # Old rows did not record output identities. NULL blocks
                # collection rather than claiming those outputs had no assets.
                self._conn.execute("ALTER TABLE history ADD COLUMN asset_digests TEXT")
            if "thumbnail" not in columns:
                self._conn.execute("ALTER TABLE history ADD COLUMN thumbnail TEXT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS history_scope_finished"
                " ON history(scope, finished DESC, run_id DESC)"
            )
        self._create_accepted()

    def _create_accepted(self) -> None:
        with self._lock, self._accepted_conn:
            # Only this engine's accepted rows are swept on its next startup.
            self._accepted_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accepted (
                    run_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    principal_kind TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    submitted REAL NOT NULL,
                    started REAL,
                    fingerprint TEXT NOT NULL,
                    source_document TEXT NOT NULL,
                    execution_identity TEXT NOT NULL,
                    attempt INTEGER NOT NULL
                )
                """
            )
            columns = {
                row["name"] for row in self._accepted_conn.execute("PRAGMA table_info(accepted)")
            }
            if "workflow_name" not in columns:
                self._accepted_conn.execute(
                    "ALTER TABLE accepted ADD COLUMN workflow_name TEXT NOT NULL DEFAULT ''"
                )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            finally:
                if self._accepted_conn is not self._conn:
                    self._accepted_conn.close()

    def set_asset_reference_invalidator(
        self, library: LibraryClient, invalidate: Callable[[], None]
    ) -> None:
        """Only the helper owning terminal history may publish this engine's accepted roots."""
        with self._lock:
            if self._helper is not library:
                raise ValueError("engine references require history on the same helper")
            if self._invalidate_asset_references not in (None, invalidate):
                raise ValueError("history already has an asset reference publisher")
            self._invalidate_asset_references = invalidate

    @contextmanager
    def hold_engine_references(self) -> Generator[tuple[frozenset[str], bool]]:
        """Hold local accepted work; pending jobs may still introduce scratch references."""
        if self._helper is None or self._invalidate_asset_references is None:
            raise AssetError("engine references require a bound helper history publisher")
        with self._lock:
            try:
                rows = self._accepted_conn.execute(
                    "SELECT source_document FROM accepted"
                ).fetchall()
                roots = frozenset(require_digest(row[0]) for row in rows if row[0])
                yield roots, not rows
            except sqlite3.Error as exc:
                raise AssetError("accepted asset reference state is unavailable") from exc

    @contextmanager
    def hold_asset_references(self) -> Generator[frozenset[str]]:
        """Hold every scope's accepted and terminal workflow roots behind their write barrier.

        Terminal target identities are durable; opaque document contents and
        live preview values still have other owners.
        """
        if self._helper is not None:
            raise AssetError("helper-backed history requires helper-owned collection")
        if self._accepted_conn is not self._conn:
            raise AssetError("separate engine accepted storage requires helper-owned collection")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute("SELECT source_document FROM history").fetchall()
                rows += self._accepted_conn.execute(
                    "SELECT source_document FROM accepted"
                ).fetchall()
                roots: set[str] = set()
                for (digest,) in rows:
                    if not isinstance(digest, str):
                        raise AssetError("history asset reference is invalid")
                    if digest:
                        roots.add(require_digest(digest))
                for (manifest,) in self._conn.execute("SELECT asset_digests FROM history"):
                    try:
                        roots.update(_asset_digests(json.loads(manifest)))
                    except (TypeError, ValueError) as exc:
                        raise AssetError("history asset references are unavailable") from exc
                yield frozenset(roots)
            except sqlite3.Error as exc:
                raise AssetError("history asset reference state is unavailable") from exc
            finally:
                self._conn.rollback()

    def put(self, record: HistoryRecord) -> None:
        """Idempotent by run_id: recording the same terminal job twice
        leaves the immutable first row in place (a run reaches terminal
        exactly once, but the write path must tolerate retries)."""
        if not record.scope.strip():
            raise ValueError("scope must be a non-empty string")
        with self._lock, self._conn:
            self._put_locked(record)

    def _put_locked(self, record: HistoryRecord) -> bool:
        """The one history INSERT, shared by put, job_terminal, and
        recover_interrupted; the caller holds the lock and a transaction.
        Returns False when an immutable row already claimed the run_id."""
        if self._helper is not None:
            document = record.to_wire()
            document["assetDigests"] = (
                list(_asset_digests(list(record.asset_digests)))
                if record.asset_digests is not None
                else None
            )
            return cast(
                bool,
                self._helper.call("history.append", scope=record.scope, record=document)["created"],
            )
        cursor = self._conn.execute(
            "INSERT INTO history"
            " (run_id, scope, principal, principal_kind, client_id, job_id, state,"
            " priority, submitted,"
            " started, finished, source_document, error, executed, cached,"
            " skipped, attempt, node_receipts, asset_digests, thumbnail, automatic,"
            " workflow_name)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id) DO NOTHING",
            (
                record.run_id,
                record.scope,
                record.principal_id,
                record.principal_kind,
                record.client_id,
                record.job_id,
                record.state,
                record.priority,
                record.submitted_at,
                record.started_at,
                record.finished_at,
                record.source_document,
                json.dumps(record.error) if record.error is not None else None,
                record.executed,
                record.cached,
                record.skipped,
                record.attempt,
                json.dumps(record.node_receipts, separators=(",", ":")),
                json.dumps(_asset_digests(list(record.asset_digests)))
                if record.asset_digests is not None
                else None,
                json.dumps(record.thumbnail.to_wire()) if record.thumbnail is not None else None,
                json.dumps(record.automatic, separators=(",", ":")),
                record.workflow_name,
            ),
        )
        return cursor.rowcount > 0

    # -- QueuePersistence (the queue's durability hook) -------------------
    # These are called synchronously on the event loop by JobQueue - see
    # QueuePersistence in queue.py for why inline (write-before-acknowledge
    # and the atomic terminal move need strict ordering).

    def job_accepted(self, job: Job) -> None:
        """Record an accepted job before the queue acknowledges it. A raise
        here refuses the submission."""
        execution_identity = (
            job.execution.extension_snapshot_digest if job.execution is not None else ""
        )
        with self._lock, self._accepted_conn:
            if self._invalidate_asset_references is not None:
                self._invalidate_asset_references()
            self._accepted_conn.execute(
                "INSERT INTO accepted"
                " (run_id, scope, principal, principal_kind, client_id, job_id,"
                " state, priority, submitted, started, fingerprint,"
                " source_document, execution_identity, attempt, workflow_name)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.run_id,
                    job.scope,
                    job.principal_id,
                    job.principal_kind,
                    job.key.client_id,
                    job.key.job_id,
                    "queued",
                    job.priority,
                    job.submitted_at,
                    None,
                    job.fingerprint,
                    job.source_document,
                    execution_identity,
                    job.attempt,
                    job.workflow_name,
                ),
            )

    def job_started(self, job: Job) -> None:
        """Mark an accepted job as running so a crash can distinguish
        never-started from was-running."""
        with self._lock, self._accepted_conn:
            self._accepted_conn.execute(
                "UPDATE accepted SET state = 'running', started = ? WHERE run_id = ?",
                (job.started_at, job.run_id),
            )

    def job_terminal(self, job: Job) -> None:
        """Commit immutable history before retiring the accepted row.

        Shared-file stores use one transaction. Separate engine state commits
        history first, so a crash between commits safely retries the same run.
        """
        record = record_from_job(job)
        with self._lock, self._accepted_conn, self._conn:
            if self._accepted_conn is not self._conn:
                self._accepted_conn.execute("BEGIN IMMEDIATE")
            self._put_locked(record)
            self._accepted_conn.execute("DELETE FROM accepted WHERE run_id = ?", (record.run_id,))

    def recover_interrupted(
        self, *, library: LibraryStore | None = None
    ) -> tuple[HistoryRecord, ...]:
        """Sweep accepted rows a dead process left behind into terminal
        history records with state ``interrupted``. NOTHING is re-executed:
        both never-started and was-running jobs become visible history that
        a human must explicitly resubmit (user ruling on issue #556).
        Idempotent - existing history rows are never rewritten, and a second
        call finds an empty table."""
        now = time.time()
        records: list[HistoryRecord] = []
        with self._lock, self._accepted_conn, self._conn:
            if self._accepted_conn is not self._conn:
                self._accepted_conn.execute("BEGIN IMMEDIATE")
            rows = self._accepted_conn.execute(
                "SELECT * FROM accepted ORDER BY submitted"
            ).fetchall()
            for row in rows:
                was_running = row["state"] == "running"
                phase = "running" if was_running else "queued"
                message = (
                    "the server exited while this job was running"
                    if was_running
                    else "the server exited before this job started"
                ) + "; it was not re-run - resubmit it to run it"
                error: dict[str, Any] = {
                    "kind": "interrupted",
                    "phase": phase,
                    "message": message,
                    "fingerprint": row["fingerprint"],
                }
                if row["execution_identity"]:
                    error["extensionSnapshotDigest"] = row["execution_identity"]
                record = HistoryRecord(
                    run_id=row["run_id"],
                    scope=row["scope"],
                    principal_id=row["principal"],
                    principal_kind=row["principal_kind"],
                    client_id=row["client_id"],
                    job_id=row["job_id"],
                    state="interrupted",
                    priority=row["priority"],
                    submitted_at=row["submitted"],
                    started_at=row["started"],
                    finished_at=now,
                    attempt=row["attempt"],
                    source_document=row["source_document"],
                    workflow_name=row["workflow_name"],
                    error=error,
                    # A crash may follow sealed result publication but precede
                    # the history commit. Only that exact complete manifest
                    # can replace unknown references with durable identities.
                    asset_digests=_published_asset_digests(
                        library, row["scope"], row["run_id"], row["attempt"]
                    ),
                )
                if self._put_locked(record):
                    records.append(record)
            self._accepted_conn.execute("DELETE FROM accepted")
        return tuple(records)

    def get(self, scope: str, run_id: str) -> HistoryRecord | None:
        """Scoped lookup: a wrong scope is a miss, never a hint that the
        run exists elsewhere."""
        with self._lock:
            if self._helper is not None:
                document = self._helper.call("history.get", scope=scope, id=run_id)["record"]
                return _from_document(cast("dict[str, object]", document)) if document else None
            row = self._conn.execute(
                "SELECT * FROM history WHERE run_id = ? AND scope = ?",
                (run_id, scope),
            ).fetchone()
        return _from_row(row) if row is not None else None

    def delete(self, scope: str, run_id: str) -> bool:
        """Remove one record - never the asset bytes its sourceDocument
        names (same delete-keeps-bytes stance as the library; vault GC of
        unreferenced digests is a separate, later concern)."""
        with self._lock, self._conn:
            if self._helper is not None:
                return cast(
                    bool, self._helper.call("history.delete", scope=scope, id=run_id)["deleted"]
                )
            cursor = self._conn.execute(
                "DELETE FROM history WHERE run_id = ? AND scope = ?",
                (run_id, scope),
            )
        return cursor.rowcount > 0

    def delete_where(
        self,
        scope: str,
        *,
        client_id: str = "",
        source_document: str = "",
        state: str = "",
        before: float | None = None,
    ) -> int:
        """Bulk clear: the same exact-match filters as query, plus
        ``before`` (finished strictly earlier than) for retention-style
        pruning. No filters clears the whole scope - the explicit scope is
        what makes that a deliberate act. Returns the number removed."""
        if not scope.strip():
            raise ValueError("scope must be a non-empty string")
        if self._helper is not None:
            with self._lock:
                return cast(
                    int,
                    self._helper.call(
                        "history.clear",
                        scope=scope,
                        clientId=client_id,
                        sourceDocument=source_document,
                        state=state,
                        before=before,
                    )["deleted"],
                )
        clauses, params = _filter_clauses(
            scope,
            client_id=client_id,
            source_document=source_document,
            state=state,
        )
        if before is not None:
            clauses.append("finished < ?")
            params.append(before)
        sql = "DELETE FROM history WHERE " + " AND ".join(clauses)
        with self._lock, self._conn:
            cursor = self._conn.execute(sql, params)
        return cursor.rowcount

    def query(
        self,
        scope: str,
        *,
        client_id: str = "",
        source_document: str = "",
        state: str = "",
        limit: int = 50,
        after: tuple[float, str] | None = None,
    ) -> list[HistoryRecord]:
        """Newest-finished first, keyset-paged: ``after`` is the
        (finished, run_id) of the previous page's last record. Filters
        are exact matches; fetch limit+1 to learn whether a next page
        exists."""
        if self._helper is not None:
            arguments: dict[str, object] = {
                "scope": scope,
                "clientId": client_id,
                "sourceDocument": source_document,
                "state": state,
            }
            if after is not None:
                arguments["after"] = list(after)
            selected: list[HistoryRecord] = []
            with self._lock:
                while len(selected) < max(1, limit):
                    page = self._helper.call(
                        "history.list", **arguments, limit=min(200, max(1, limit) - len(selected))
                    )
                    selected.extend(
                        _from_document(document)
                        for document in cast("list[dict[str, object]]", page["items"])
                    )
                    if page["cursor"] is None:
                        break
                    arguments.pop("after", None)
                    arguments["cursor"] = page["cursor"]
            return selected
        clauses, params = _filter_clauses(
            scope,
            client_id=client_id,
            source_document=source_document,
            state=state,
        )
        if after is not None:
            clauses.append("(finished < ? OR (finished = ? AND run_id < ?))")
            params.extend((after[0], after[0], after[1]))
        sql = (
            "SELECT * FROM history WHERE "
            + " AND ".join(clauses)
            + " ORDER BY finished DESC, run_id DESC LIMIT ?"
        )
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]


def _filter_clauses(
    scope: str,
    *,
    client_id: str,
    source_document: str,
    state: str,
) -> tuple[list[str], list[object]]:
    """The one definition of history's exact-match filters, shared by
    query and delete_where so browse and clear can never drift apart."""
    clauses = ["scope = ?"]
    params: list[object] = [scope]
    if client_id:
        clauses.append("client_id = ?")
        params.append(client_id)
    if source_document:
        clauses.append("source_document = ?")
        params.append(source_document)
    if state:
        clauses.append("state = ?")
        params.append(state)
    return clauses, params


def _from_document(document: dict[str, object]) -> HistoryRecord:
    return HistoryRecord(
        run_id=cast(str, document["runId"]),
        scope=cast(str, document["scope"]),
        principal_id=cast(str, document["principalId"]),
        principal_kind=cast(str, document["principalKind"]),
        client_id=cast(str, document["clientId"]),
        job_id=cast(str, document["jobId"]),
        state=cast(str, document["state"]),
        priority=cast(int, document["priority"]),
        submitted_at=cast(float, document["submittedAt"]),
        started_at=cast("float | None", document.get("startedAt")),
        finished_at=cast(float, document["finishedAt"]),
        attempt=cast(int, document["attemptId"]),
        source_document=cast(str, document.get("sourceDocument", "")),
        workflow_name=cast(str, document.get("workflowName", "")),
        error=cast("dict[str, Any] | None", document.get("error")),
        executed=cast(int, document["executed"]),
        cached=cast(int, document["cached"]),
        skipped=cast(int, document["skipped"]),
        node_receipts=tuple(cast("list[dict[str, str]]", document["nodeReceipts"])),
        thumbnail=AssetRef.from_wire(cast("dict[str, object]", document["thumbnail"]))
        if "thumbnail" in document
        else None,
        automatic=tuple(cast("list[dict[str, object]]", document.get("automaticValues", []))),
        asset_digests=_asset_digests(document["assetDigests"])
        if document["assetDigests"] is not None
        else None,
    )


def _from_row(row: sqlite3.Row) -> HistoryRecord:
    error_raw = row["error"]
    return HistoryRecord(
        run_id=row["run_id"],
        scope=row["scope"],
        principal_id=row["principal"],
        principal_kind=row["principal_kind"],
        client_id=row["client_id"],
        job_id=row["job_id"],
        state=row["state"],
        priority=row["priority"],
        submitted_at=row["submitted"],
        started_at=row["started"],
        finished_at=row["finished"],
        attempt=row["attempt"],
        source_document=row["source_document"],
        workflow_name=row["workflow_name"],
        error=json.loads(error_raw) if error_raw is not None else None,
        executed=row["executed"],
        cached=row["cached"],
        skipped=row["skipped"],
        node_receipts=tuple(json.loads(row["node_receipts"])),
        thumbnail=AssetRef.from_wire(json.loads(row["thumbnail"]))
        if row["thumbnail"] is not None
        else None,
        automatic=tuple(json.loads(row["automatic"])),
        asset_digests=_asset_digests(json.loads(row["asset_digests"]))
        if row["asset_digests"] is not None
        else None,
    )


# -- HTTP surface --------------------------------------------------------


HISTORY_KEY: web.AppKey[HistoryStore] = web.AppKey("history_store")


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _history_scope(request: web.Request) -> str:
    principal = principal_for(request)
    value = request.query.get("scope")
    if principal.local:
        if value is None or not value.strip():
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "scope is required (single-user: 'local')"}),
                content_type="application/json",
            )
        return value.strip()
    if value is None:
        return resolve_scope(principal, "history:read", None)
    if not value or value != value.strip() or any(ch.isspace() for ch in value):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "scope must be a non-empty whitespace-free string"}),
            content_type="application/json",
        )
    return value


def _history_scope_allowed(request: web.Request, scope: str) -> bool:
    principal = principal_for(request)
    return principal is LOCAL_PRINCIPAL or principal.allows_in(scope, "history:read")


async def handle_history_list(request: web.Request) -> web.Response:
    store = request.app[HISTORY_KEY]
    scope = _history_scope(request)
    client_id = request.query.get("clientId", "")
    source_document = request.query.get("sourceDocument", "")
    state = request.query.get("state", "")
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        return _json_error(400, "limit must be an integer")
    bound = {
        "s": scope,
        "c": client_id,
        "d": source_document,
        "t": state,
    }
    after = None
    cursor = request.query.get("cursor")
    if cursor:
        after = decode_cursor(cursor, bound)
    records = (
        await asyncio.to_thread(
            lambda: store.query(
                scope,
                client_id=client_id,
                source_document=source_document,
                state=state,
                limit=limit + 1,
                after=after,
            )
        )
        if _history_scope_allowed(request, scope)
        else []
    )
    wire: dict[str, object] = {"records": [record.to_wire() for record in records[:limit]]}
    if len(records) > limit:
        last = records[limit - 1]
        wire["cursor"] = encode_cursor(bound, (last.finished_at, last.run_id))
    return web.json_response(wire)


async def handle_history_get(request: web.Request) -> web.Response:
    store = request.app[HISTORY_KEY]
    scope = _history_scope(request)
    record = (
        await asyncio.to_thread(store.get, scope, request.match_info["run_id"])
        if _history_scope_allowed(request, scope)
        else None
    )
    if record is None:
        return _json_error(404, "no such run")
    return web.json_response(record.to_wire())


async def handle_history_delete(request: web.Request) -> web.Response:
    """Remove one record; never the workflow asset its sourceDocument
    names. Wrong scope stays a plain 404 - deletion leaks no more about
    other scopes than lookup does."""
    store = request.app[HISTORY_KEY]
    scope = _history_scope(request)
    deleted = (
        await asyncio.to_thread(store.delete, scope, request.match_info["run_id"])
        if _history_scope_allowed(request, scope)
        else False
    )
    if not deleted:
        return _json_error(404, "no such run")
    return web.Response(status=204)


async def handle_history_clear(request: web.Request) -> web.Response:
    """Bulk clear with exactly the list endpoint's filters plus before=
    (finished earlier than the given epoch timestamp). No filters clears
    the scope entirely - the required explicit scope makes that a
    deliberate act. Returns how many rows went."""
    store = request.app[HISTORY_KEY]
    scope = _history_scope(request)
    before_raw = request.query.get("before")
    before = None
    if before_raw is not None:
        try:
            before = float(before_raw)
        except ValueError:
            return _json_error(400, "before must be an epoch timestamp")
    deleted = (
        await asyncio.to_thread(
            lambda: store.delete_where(
                scope,
                client_id=request.query.get("clientId", ""),
                source_document=request.query.get("sourceDocument", ""),
                state=request.query.get("state", ""),
                before=before,
            )
        )
        if _history_scope_allowed(request, scope)
        else 0
    )
    return web.json_response({"deleted": deleted})


@web.middleware
async def _history_errors(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    try:
        return await handler(request)
    except LibraryError as error:
        if request.path != "/api/history" and not request.path.startswith("/api/history/"):
            raise
        return _json_error(
            400 if error.code == "invalid" else 503,
            str(error) if error.code == "invalid" else "history store unavailable",
        )


def add_history_routes(app: web.Application, store: HistoryStore) -> None:
    """Routes only - the store's lifecycle belongs to whoever records into
    it (create_app closes it AFTER the queue drains: shutdown cancellations
    are terminal transitions, and their writes must land in an open store)."""
    app[HISTORY_KEY] = store
    app.middlewares.append(_history_errors)
    app.router.add_get("/api/history", handle_history_list)
    app.router.add_delete("/api/history", handle_history_clear)
    app.router.add_get("/api/history/{run_id}", handle_history_get)
    app.router.add_delete("/api/history/{run_id}", handle_history_delete)
