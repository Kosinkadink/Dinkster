"""Durable per-stream journal: the host-owned append/replay substrate for
correctness-critical event streams (docs/training-design.md section 9.2).

The live `node_event` channel is droppable chatter; facts that must never
drop live here first and are mirrored to live events, not the other way
around. Training sessions are the first stream family; server-extension
managed jobs are expected to adopt this same substrate, envelope, and
sequence semantics rather than ship a private journal.

The store is deliberately stream-agnostic: a stream id is an opaque
caller-namespaced key (training uses `training-session/<sessionId>` via
dinkster_protocol.training.training_session_stream), a record is a named
JSON payload with a durability class, and `seq` is assigned here, at
append time, monotonically per stream. Typed event unions live in the
protocol layer; this module owns only durability, ordering, replay, and
retention.

Retention within a live stream can remove only coalescible telemetry
rows. Durable rows are immutable while their stream exists, so a replay
from any `after` cursor always contains every durable fact; the
per-stream `coalescedBelow` watermark tells consumers how far telemetry
pruning has gone. A missing-durable-fact gap therefore cannot be
produced through this API. Separately, `delete_stream` removes a stream
as a whole unit - the retention story for stream families whose facts
expire with the stream (execution-run logs). A family whose durable
facts must outlive everything (training sessions) must never call it.

Storage mirrors HistoryStore/LibraryStore: one SQLite file, WAL, a
process-wide write lock, sync methods crossed via asyncio.to_thread.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from dinkster_protocol.payload import canonical_json_payload

__all__ = [
    "MAX_JOURNAL_PAYLOAD_BYTES",
    "JournalPage",
    "JournalRecord",
    "JournalStore",
    "JournalTransaction",
]

_T = TypeVar("_T")


MAX_JOURNAL_PAYLOAD_BYTES = 256 * 1024
"""Substrate-level payload ceiling. Stream families enforce tighter limits
(training events cap at MAX_TRAINING_EVENT_DATA_BYTES); this guard keeps a
misbehaving appender from turning the journal into a blob store."""


@dataclass(frozen=True)
class JournalRecord:
    """One appended fact, immutable once written."""

    stream_id: str
    seq: int
    timestamp: float
    name: str
    durable: bool
    payload: Mapping[str, object]

    def to_wire(self) -> dict[str, object]:
        return {
            "streamId": self.stream_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "name": self.name,
            "durable": self.durable,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class JournalPage:
    """One replay page. `latest_seq` is 0 for an empty stream. Telemetry
    records with seq <= `coalesced_below` may have been pruned; durable
    records are always complete."""

    records: tuple[JournalRecord, ...]
    latest_seq: int
    coalesced_below: int


class JournalStore:
    """SQLite-backed per-stream journal. Same threading contract as
    HistoryStore/LibraryStore: one connection behind a lock, callers cross
    via asyncio.to_thread."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal (
                    stream TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    name TEXT NOT NULL,
                    durable INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (stream, seq)
                )
                """
            )
            # last_seq is the authoritative sequence counter: deriving the
            # next seq from MAX(seq) would reuse numbers after retention
            # prunes the newest coalescible rows.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_streams (
                    stream TEXT PRIMARY KEY,
                    last_seq INTEGER NOT NULL DEFAULT 0,
                    coalesced_below INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append(
        self,
        stream_id: str,
        name: str,
        payload: Mapping[str, object],
        *,
        durable: bool,
        timestamp: float | None = None,
    ) -> JournalRecord:
        """Assign the stream's next sequence number and write the record in
        one transaction. The returned record carries the assigned seq."""
        return self.transact(
            lambda txn: txn.append(stream_id, name, payload, durable=durable, timestamp=timestamp)
        )

    def transact(self, fn: Callable[[JournalTransaction], _T]) -> _T:
        """Run ``fn`` inside one durable journal transaction.

        This is the host-owned append/transaction API from training-design.md
        section 9.2: a stream-family owner that must move its own ledger state
        and append journal facts atomically (the training session supervisor's
        claim/commit protocol) runs both inside one call. The callable receives
        a :class:`JournalTransaction`; if it raises, every write it made rolls
        back. The callable runs under the store's write lock, so it must not
        block on IO beyond its own SQLite statements.

        The transaction is opened eagerly (BEGIN IMMEDIATE) so the callable's
        READS are already inside it: a decision made on what was read cannot
        be invalidated by another connection writing between the read and the
        write, even when several stores share the file.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(JournalTransaction(self._conn))
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()
            return result

    def read(self, stream_id: str, *, after: int = 0, limit: int = 1000) -> JournalPage:
        """Records with seq strictly greater than `after`, oldest first."""
        if after < 0:
            raise ValueError("after must be non-negative")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM journal WHERE stream = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (stream_id, after, max(1, limit)),
            ).fetchall()
            latest = self._latest_seq_locked(stream_id)
            floor = self._coalesced_below_locked(stream_id)
        return JournalPage(
            records=tuple(_from_row(row) for row in rows),
            latest_seq=latest,
            coalesced_below=floor,
        )

    def latest_seq(self, stream_id: str) -> int:
        with self._lock:
            return self._latest_seq_locked(stream_id)

    def prune_coalescible(self, stream_id: str, up_to: int) -> int:
        """Retention for telemetry: delete coalescible records with
        seq <= up_to and advance the stream's coalesced watermark. Durable
        records are never touched. Returns the number removed."""
        return self.transact(lambda txn: txn.prune_coalescible(stream_id, up_to))

    def delete_stream(self, stream_id: str) -> int:
        """Remove the stream as a whole unit (see module docstring).
        Returns the number of records removed."""
        return self.transact(lambda txn: txn.delete_stream(stream_id))

    def _latest_seq_locked(self, stream_id: str) -> int:
        return _latest_seq(self._conn, stream_id)

    def _coalesced_below_locked(self, stream_id: str) -> int:
        return _coalesced_below(self._conn, stream_id)


class JournalTransaction:
    """Write handle passed to :meth:`JournalStore.transact` callables.

    Valid only for the duration of that call: the store's write lock and
    SQLite transaction are already held, so methods here perform no locking
    of their own. Stream-family owners use ``execute`` for their own tables
    (which must live in the same SQLite file) and ``append`` for journal
    facts, and both commit or roll back together.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(
        self,
        stream_id: str,
        name: str,
        payload: Mapping[str, object],
        *,
        durable: bool,
        timestamp: float | None = None,
    ) -> JournalRecord:
        """Assign the stream's next sequence number and write the record.
        The returned record carries the assigned seq."""
        if not stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        if not name.strip():
            raise ValueError("name must be a non-empty string")
        if type(durable) is not bool:
            raise ValueError("durable must be a bool: it decides what retention may prune")
        encoded = canonical_json_payload(
            payload, max_bytes=MAX_JOURNAL_PAYLOAD_BYTES, description="payload"
        )
        written_at = time.time() if timestamp is None else timestamp
        if type(written_at) is not float or not math.isfinite(written_at) or written_at < 0.0:
            raise ValueError("timestamp must be a finite non-negative float")
        self._conn.execute(
            "INSERT INTO journal_streams (stream, last_seq) VALUES (?, 1)"
            " ON CONFLICT(stream) DO UPDATE SET last_seq = last_seq + 1",
            (stream_id,),
        )
        seq = _latest_seq(self._conn, stream_id)
        self._conn.execute(
            "INSERT INTO journal (stream, seq, ts, name, durable, payload)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (stream_id, seq, written_at, name, 1 if durable else 0, encoded),
        )
        return JournalRecord(
            stream_id=stream_id,
            seq=seq,
            timestamp=written_at,
            name=name,
            durable=durable,
            # Decode the persisted canonical bytes so the returned record
            # shares no structure with the caller's mutable mapping.
            payload=cast("Mapping[str, object]", json.loads(encoded)),
        )

    def latest_seq(self, stream_id: str) -> int:
        return _latest_seq(self._conn, stream_id)

    def read(self, stream_id: str, *, after: int = 0, limit: int = 1000) -> JournalPage:
        """Records with seq strictly greater than `after`, oldest first
        (see :meth:`JournalStore.read`), inside this transaction - for
        stream-family owners whose read must be atomic with their own
        table lookups (for example a scope check against retention)."""
        if after < 0:
            raise ValueError("after must be non-negative")
        rows = self._conn.execute(
            "SELECT * FROM journal WHERE stream = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
            (stream_id, after, max(1, limit)),
        ).fetchall()
        return JournalPage(
            records=tuple(_from_row(row) for row in rows),
            latest_seq=_latest_seq(self._conn, stream_id),
            coalesced_below=_coalesced_below(self._conn, stream_id),
        )

    def prune_coalescible(self, stream_id: str, up_to: int) -> int:
        """Retention for telemetry: delete coalescible records with
        seq <= up_to and advance the stream's coalesced watermark. Durable
        records are never touched. Returns the number removed."""
        if up_to < 0:
            raise ValueError("up_to must be non-negative")
        watermark = min(up_to, _latest_seq(self._conn, stream_id))
        cursor = self._conn.execute(
            "DELETE FROM journal WHERE stream = ? AND seq <= ? AND durable = 0",
            (stream_id, watermark),
        )
        if watermark > _coalesced_below(self._conn, stream_id):
            self._conn.execute(
                "INSERT INTO journal_streams (stream, coalesced_below) VALUES (?, ?)"
                " ON CONFLICT(stream) DO UPDATE SET coalesced_below = excluded.coalesced_below",
                (stream_id, watermark),
            )
        return cursor.rowcount

    def delete_stream(self, stream_id: str) -> int:
        """Remove every record of the stream and its counter row (see the
        module docstring: whole-stream retention for families whose facts
        expire with the stream). Reusing a deleted stream id restarts its
        seq at 1. Returns the number of records removed."""
        cursor = self._conn.execute("DELETE FROM journal WHERE stream = ?", (stream_id,))
        self._conn.execute("DELETE FROM journal_streams WHERE stream = ?", (stream_id,))
        return cursor.rowcount

    def execute(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Cursor:
        """Run a statement against the shared SQLite file inside this
        transaction. For stream-family tables that must move atomically
        with journal appends."""
        return self._conn.execute(sql, params)


def _latest_seq(conn: sqlite3.Connection, stream_id: str) -> int:
    row = conn.execute(
        "SELECT last_seq FROM journal_streams WHERE stream = ?",
        (stream_id,),
    ).fetchone()
    return int(row["last_seq"]) if row is not None else 0


def _coalesced_below(conn: sqlite3.Connection, stream_id: str) -> int:
    row = conn.execute(
        "SELECT coalesced_below FROM journal_streams WHERE stream = ?",
        (stream_id,),
    ).fetchone()
    return int(row["coalesced_below"]) if row is not None else 0


def _from_row(row: sqlite3.Row) -> JournalRecord:
    return JournalRecord(
        stream_id=row["stream"],
        seq=row["seq"],
        timestamp=row["ts"],
        name=row["name"],
        durable=bool(row["durable"]),
        payload=cast("Mapping[str, object]", json.loads(row["payload"])),
    )
