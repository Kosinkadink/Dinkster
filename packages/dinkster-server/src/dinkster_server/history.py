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
it. Since environment stamps live INSIDE documents, persisting the
document (library) plus this link answers "what exact document and
environment produced this run" with no second channel.

What is deliberately NOT here: output values. Values live in the cache
and are served by the peek surface with its own retention and refusal
vocabulary; history rows are facts about runs, small enough to keep
forever.

The store is also the queue's durability hook (QueuePersistence): every
accepted job gets a row in the ``accepted`` table before the submission
is acknowledged, updated when it starts and atomically retired into
``history`` when it reaches a terminal state. Rows left behind by a
process that died mid-queue are swept at the next startup by
recover_interrupted() into history records with state ``interrupted`` -
durable visibility, zero automatic re-execution: an interrupted job runs
again only when a human resubmits it (user ruling on issue #556).

Scoping is structural: every job is stamped with its selected scope,
principal, and principal kind at acceptance, and terminal history preserves
those facts. History wire records expose them as ``principalId`` and
``principalKind``.

Storage mirrors LibraryStore: one SQLite file, WAL, a process-wide
write lock, sync methods crossed via asyncio.to_thread. Rows are
immutable once written (a terminal job never changes); re-recording a
runId leaves the first row unchanged.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web

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
    error: dict[str, Any] | None = None
    executed: int = 0
    cached: int = 0
    skipped: int = 0
    node_receipts: tuple[dict[str, str], ...] = ()

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
        if self.error is not None:
            wire["error"] = self.error
        return wire


def record_from_job(job: Job) -> HistoryRecord:
    """The durable facts of a terminal job. Counts summarize the run
    (full outputs stay in the cache/peek system, never in history)."""
    if job.state not in TERMINAL_STATES:
        raise ValueError(f"job {job.key} is {job.state}, not terminal")
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
        error=job.error,
        executed=len(job.result.executed) if job.result is not None else 0,
        cached=len(job.result.cached) if job.result is not None else 0,
        skipped=len(job.result.skipped) if job.result is not None else 0,
        node_receipts=tuple(node_receipts),
    )


class HistoryStore:
    """SQLite-backed scoped run history. Same threading contract as
    LibraryStore: one connection behind a lock, callers cross via
    asyncio.to_thread."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
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
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS history_scope_finished"
                " ON history(scope, finished DESC, run_id DESC)"
            )
            # Accepted-but-not-terminal jobs (the queue's durability hook).
            # A row exists exactly while the queue holds the job; leftovers
            # mean the process died and recover_interrupted() sweeps them.
            self._conn.execute(
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

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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
        cursor = self._conn.execute(
            "INSERT INTO history"
            " (run_id, scope, principal, principal_kind, client_id, job_id, state,"
            " priority, submitted,"
            " started, finished, source_document, error, executed, cached,"
            " skipped, attempt, node_receipts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO accepted"
                " (run_id, scope, principal, principal_kind, client_id, job_id,"
                " state, priority, submitted, started, fingerprint,"
                " source_document, execution_identity, attempt)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )

    def job_started(self, job: Job) -> None:
        """Mark an accepted job as running so a crash can distinguish
        never-started from was-running."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE accepted SET state = 'running', started = ? WHERE run_id = ?",
                (job.started_at, job.run_id),
            )

    def job_terminal(self, job: Job) -> None:
        """One transaction: the terminal history row lands and the accepted
        row goes, atomically - no crash window where the job is both
        durable history and a sweepable leftover."""
        record = record_from_job(job)
        with self._lock, self._conn:
            self._put_locked(record)
            self._conn.execute("DELETE FROM accepted WHERE run_id = ?", (record.run_id,))

    def recover_interrupted(self) -> tuple[HistoryRecord, ...]:
        """Sweep accepted rows a dead process left behind into terminal
        history records with state ``interrupted``. NOTHING is re-executed:
        both never-started and was-running jobs become visible history that
        a human must explicitly resubmit (user ruling on issue #556).
        Idempotent - existing history rows are never rewritten, and a second
        call finds an empty table."""
        now = time.time()
        records: list[HistoryRecord] = []
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT * FROM accepted ORDER BY submitted").fetchall()
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
                    error=error,
                )
                if self._put_locked(record):
                    records.append(record)
            self._conn.execute("DELETE FROM accepted")
        return tuple(records)

    def get(self, scope: str, run_id: str) -> HistoryRecord | None:
        """Scoped lookup: a wrong scope is a miss, never a hint that the
        run exists elsewhere."""
        with self._lock:
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
        error=json.loads(error_raw) if error_raw is not None else None,
        executed=row["executed"],
        cached=row["cached"],
        skipped=row["skipped"],
        node_receipts=tuple(json.loads(row["node_receipts"])),
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


def add_history_routes(app: web.Application, store: HistoryStore) -> None:
    """Routes only - the store's lifecycle belongs to whoever records into
    it (create_app closes it AFTER the queue drains: shutdown cancellations
    are terminal transitions, and their writes must land in an open store)."""
    app[HISTORY_KEY] = store
    app.router.add_get("/api/history", handle_history_list)
    app.router.add_delete("/api/history", handle_history_clear)
    app.router.add_get("/api/history/{run_id}", handle_history_get)
    app.router.add_delete("/api/history/{run_id}", handle_history_delete)
