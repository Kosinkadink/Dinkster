"""Durable execution-run logs: the journal-backed reliable copy of the
live job event stream.

Each run's published wire events land in an ``execution-run/<runId>``
stream of a JournalStore owned by this module. The live socket stays
droppable chatter; this journal is what the frontend backfills from after
a reconnect or reload (GET /api/runs/{runId}/journal). Payloads are the
exact published wire dicts - post-redaction, carrying the per-job ``seq``
the frontend orders and dedupes by - so replayed and live events are the
same contract; the journal's own seq is only the replay cursor.

Durability classes follow the run's needs, not the training ledger's:
lifecycle events, job transitions, and warning logs are durable while the
run lives; info logs and progress are coalescible and pruned when the run
reaches a terminal state (the last progress record per node is kept).
Whole streams then expire via retention: only the newest ``keep_runs``
finished runs are kept, and no finished run outlives ``keep_days``. Runs
still executing are never expired.

Node execution errors have no records of their own here: they ride the
job error report and the ``node_failed``/``job_state`` events exactly as
published, the same single substrate the node badge derives from.

Appends are batched - one journal transaction per flush tick, not one
per log line. The journal is a record, not execution machinery: a failed
write is logged and dropped, never fatal to the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from aiohttp import web
from dinkster_schema import LOG_EVENT, PROGRESS_EVENT

from .auth import LOCAL_PRINCIPAL, principal_for, resolve_scope
from .events import BINARY_BLOB_KEY
from .journal import JournalPage, JournalStore, JournalTransaction
from .queue import TERMINAL_STATES

__all__ = [
    "EXECUTION_JOURNAL_KEY",
    "EXECUTION_RUN_STREAM_PREFIX",
    "ExecutionJournal",
    "add_execution_journal_routes",
    "execution_run_stream",
]

_log = logging.getLogger(__name__)

EXECUTION_RUN_STREAM_PREFIX = "execution-run/"

MAX_PENDING_RECORDS = 8192
"""Buffered-append ceiling: past it, coalescible telemetry is shed (the
live socket already delivered it) so a stalled flush cannot grow the
buffer unboundedly. Durable records still queue past the cap - the soft
bound bends before a fact is lost, mirroring the event hub."""

# Wire types journaled as durable lifecycle facts. node_event is handled
# separately (its "log"/"progress" sub-events classify individually);
# everything else on the wire (previews, memory samples, arbitrary node
# reports) is transient chatter the journal does not keep.
_LIFECYCLE_TYPES = frozenset(
    {
        "run_started",
        "node_started",
        "node_cached",
        "cache_miss",
        "node_finished",
        "node_failed",
        "node_skipped",
        "region_expanded",
        "region_finished",
        "run_finished",
    }
)


def execution_run_stream(run_id: str) -> str:
    return EXECUTION_RUN_STREAM_PREFIX + run_id


@dataclass(frozen=True)
class _Append:
    run_id: str
    scope: str
    name: str
    payload: Mapping[str, object]
    durable: bool
    timestamp: float


@dataclass(frozen=True)
class _Finalize:
    """Terminal marker: after it flushes, the run's coalescible telemetry
    is pruned (keeping the last progress per node) and retention runs."""

    run_id: str
    keepers: tuple[Mapping[str, object], ...]
    timestamp: float


class ExecutionJournal:
    """Batched append pipeline, post-run pruning, retention, and replay
    reads for execution-run streams. Owns its JournalStore (and with it
    the SQLite file): ``close`` lands what is buffered and closes it.

    ``observe`` is called synchronously on the event loop for every
    published job wire event and never blocks; a background task flushes
    the buffer every ``flush_interval`` seconds in one transaction.
    """

    def __init__(
        self,
        store: JournalStore,
        *,
        keep_runs: int = 200,
        keep_days: float = 30.0,
        flush_interval: float = 0.1,
    ) -> None:
        if keep_runs < 0:
            raise ValueError("keep_runs must be non-negative")
        if keep_days < 0:
            raise ValueError("keep_days must be non-negative")
        self._store = store
        self._keep_runs = keep_runs
        self._keep_days = keep_days
        self._flush_interval = flush_interval
        self._pending: list[_Append | _Finalize] = []
        self._last_progress: dict[str, dict[str, Mapping[str, object]]] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._closed = False

        # The run index rides the same SQLite file as the streams (the
        # training pattern: family tables move atomically with journal
        # appends). It is what scopes replay reads and orders retention.
        def bootstrap(txn: JournalTransaction) -> None:
            txn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_runs (
                    run_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    started REAL NOT NULL,
                    finished REAL
                )
                """
            )
            # A run left unfinished by a previous process is abandoned,
            # not live: stamp it finished at its last persisted event so
            # retention bounds it like any other finished run. (No run of
            # THIS process exists yet, so every unfinished row qualifies.)
            txn.execute(
                "UPDATE execution_runs SET finished = COALESCE("
                " (SELECT MAX(ts) FROM journal WHERE stream = ? || run_id), started)"
                " WHERE finished IS NULL",
                (EXECUTION_RUN_STREAM_PREFIX,),
            )
            self._expire(txn, now=time.time())

        self._store.transact(bootstrap)

    # -- ingest (event loop) -------------------------------------------

    def observe(self, wire: Mapping[str, object], *, scope: str) -> None:
        """Classify one published wire event and buffer it for the run's
        stream. Events the journal does not keep are ignored."""
        if self._closed:
            return
        if BINARY_BLOB_KEY in wire:
            return
        run_id = wire.get("jobRef") if "jobRef" in wire else wire.get("runId")
        if not isinstance(run_id, str) or not run_id:
            return
        kind = wire.get("type")
        terminal = False
        if kind == "job_state":
            name = "job_state"
            durable = True
            terminal = wire.get("state") in TERMINAL_STATES
        elif kind == "node_event":
            event = wire.get("event")
            if event == LOG_EVENT:
                data = wire.get("data")
                level: object = None
                if isinstance(data, Mapping):
                    level = cast("Mapping[str, object]", data).get("level")
                name = LOG_EVENT
                durable = level == "warning"
            elif event == PROGRESS_EVENT:
                name = PROGRESS_EVENT
                durable = False
                node_id = wire.get("nodeId")
                if isinstance(node_id, str):
                    self._last_progress.setdefault(run_id, {})[node_id] = dict(wire)
            else:
                return
        elif kind in _LIFECYCLE_TYPES:
            name = str(kind)
            durable = True
        else:
            return
        if not durable and len(self._pending) >= MAX_PENDING_RECORDS:
            return
        self._pending.append(
            _Append(run_id, scope, name, dict(wire), durable, timestamp=time.time())
        )
        if terminal:
            keepers = tuple(self._last_progress.pop(run_id, {}).values())
            self._pending.append(_Finalize(run_id, keepers, timestamp=time.time()))

    # -- flush pipeline -------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        """Stop the flush loop, land what is buffered (the queue's close
        just emitted terminal transitions), and close the store. Waits
        for an in-flight write instead of abandoning it: nothing may
        still hold the store when it closes."""
        self._closed = True
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        await self.flush()
        await asyncio.to_thread(self._store.close)

    async def _run(self) -> None:
        while not self._closed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), self._flush_interval)
            await self.flush()

    async def flush(self) -> None:
        """Land everything buffered now, in one transaction."""
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        try:
            await asyncio.to_thread(self._write, batch)
        except Exception:
            _log.exception("execution journal write failed; %d records dropped", len(batch))

    def _write(self, batch: list[_Append | _Finalize]) -> None:
        def apply(txn: JournalTransaction) -> None:
            registered: set[str] = set()
            for item in batch:
                if isinstance(item, _Append):
                    if item.run_id not in registered:
                        txn.execute(
                            "INSERT INTO execution_runs (run_id, scope, started)"
                            " VALUES (?, ?, ?) ON CONFLICT(run_id) DO NOTHING",
                            (item.run_id, item.scope, item.timestamp),
                        )
                        registered.add(item.run_id)
                    self._append(
                        txn,
                        item.run_id,
                        item.name,
                        item.payload,
                        durable=item.durable,
                        timestamp=item.timestamp,
                    )
                else:
                    stream = execution_run_stream(item.run_id)
                    for keeper in item.keepers:
                        self._append(
                            txn,
                            item.run_id,
                            PROGRESS_EVENT,
                            keeper,
                            durable=True,
                            timestamp=item.timestamp,
                        )
                    txn.prune_coalescible(stream, txn.latest_seq(stream))
                    txn.execute(
                        "UPDATE execution_runs SET finished = ? WHERE run_id = ?",
                        (item.timestamp, item.run_id),
                    )
                    self._expire(txn, now=item.timestamp)

        self._store.transact(apply)

    def _append(
        self,
        txn: JournalTransaction,
        run_id: str,
        name: str,
        payload: Mapping[str, object],
        *,
        durable: bool,
        timestamp: float,
    ) -> None:
        # Encoding failures (oversized or non-JSON payloads) skip the one
        # record: canonical encoding raises before any SQL runs for it, so
        # the rest of the batch is unaffected.
        try:
            txn.append(
                execution_run_stream(run_id),
                name,
                payload,
                durable=durable,
                timestamp=timestamp,
            )
        except ValueError as error:
            _log.warning("execution journal record dropped (%s/%s): %s", run_id, name, error)

    def _expire(self, txn: JournalTransaction, *, now: float) -> None:
        expired: set[str] = set()
        cutoff = now - self._keep_days * 86400.0
        rows = txn.execute(
            "SELECT run_id FROM execution_runs WHERE finished IS NOT NULL AND finished < ?",
            (cutoff,),
        ).fetchall()
        expired.update(row["run_id"] for row in rows)
        rows = txn.execute(
            "SELECT run_id FROM execution_runs WHERE finished IS NOT NULL"
            " ORDER BY finished DESC, run_id DESC LIMIT -1 OFFSET ?",
            (self._keep_runs,),
        ).fetchall()
        expired.update(row["run_id"] for row in rows)
        for run_id in expired:
            txn.delete_stream(execution_run_stream(run_id))
            txn.execute("DELETE FROM execution_runs WHERE run_id = ?", (run_id,))

    # -- reads ----------------------------------------------------------

    def scope_of(self, run_id: str) -> str | None:
        """The scope the run was journaled under, or None for a run the
        journal has never seen (including not-yet-flushed ones)."""

        def run(txn: JournalTransaction) -> str | None:
            row = txn.execute(
                "SELECT scope FROM execution_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return str(row["scope"]) if row is not None else None

        return self._store.transact(run)

    def read(self, run_id: str, *, after: int = 0, limit: int = 1000) -> JournalPage:
        """Replay the run's stream (see JournalStore.read)."""
        return self._store.read(execution_run_stream(run_id), after=after, limit=limit)

    def read_scoped(
        self, run_id: str, scope: str, *, after: int = 0, limit: int = 1000
    ) -> JournalPage | None:
        """One replay page, or None for a run the journal does not hold
        under exactly this scope. The scope check and the read share one
        transaction, so retention deleting the run between them cannot
        yield a 200-with-empty-journal instead of a miss."""

        def run(txn: JournalTransaction) -> JournalPage | None:
            row = txn.execute(
                "SELECT scope FROM execution_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None or str(row["scope"]) != scope:
                return None
            return txn.read(execution_run_stream(run_id), after=after, limit=limit)

        return self._store.transact(run)


# -- HTTP surface --------------------------------------------------------


EXECUTION_JOURNAL_KEY: web.AppKey[ExecutionJournal] = web.AppKey("execution_journal")


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _run_scope(request: web.Request) -> str:
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
        return resolve_scope(principal, "jobs:read", None)
    if not value or value != value.strip() or any(ch.isspace() for ch in value):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "scope must be a non-empty whitespace-free string"}),
            content_type="application/json",
        )
    return value


def _run_scope_allowed(request: web.Request, scope: str) -> bool:
    principal = principal_for(request)
    return principal is LOCAL_PRINCIPAL or principal.allows_in(scope, "jobs:read")


async def handle_run_journal(request: web.Request) -> web.Response:
    """GET /api/runs/{run_id}/journal?scope=&after=&limit= - one replay
    page of the run's journaled wire events, oldest first. Mirrors the
    training replay shape; a wrong scope is indistinguishable from a
    miss."""
    journal = request.app[EXECUTION_JOURNAL_KEY]
    scope = _run_scope(request)
    run_id = request.match_info["run_id"]
    try:
        after = int(request.query.get("after", "0"))
    except ValueError:
        return _json_error(400, "after must be an integer")
    if after < 0:
        return _json_error(400, "after must be non-negative")
    try:
        limit = min(1000, max(1, int(request.query.get("limit", "500"))))
    except ValueError:
        return _json_error(400, "limit must be an integer")
    page = (
        await asyncio.to_thread(journal.read_scoped, run_id, scope, after=after, limit=limit)
        if _run_scope_allowed(request, scope)
        else None
    )
    if page is None:
        return _json_error(404, "no such run")
    return web.json_response(
        {
            "records": [record.to_wire() for record in page.records],
            "latestSeq": page.latest_seq,
            "coalescedBelow": page.coalesced_below,
        }
    )


def add_execution_journal_routes(app: web.Application, journal: ExecutionJournal) -> None:
    """Routes only - the journal's lifecycle belongs to create_app's
    cleanup, which closes it after the queue drains."""
    app[EXECUTION_JOURNAL_KEY] = journal
    app.router.add_get("/api/runs/{run_id}/journal", handle_run_journal)
