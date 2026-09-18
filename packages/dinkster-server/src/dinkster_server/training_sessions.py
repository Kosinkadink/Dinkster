"""Training session supervisor: the durable claim/commit ledger for
training advances (docs/training-design.md sections 3.2, 8.2, 9.1-9.3).

A training session is a lineage of committed checkpoints advanced by
exactly-once operations. The graph layer's ``AdvanceTraining`` node is not
idempotent, so exactly-once lives HERE, in a durable claim protocol keyed
by (session, operation id, input checkpoint) and guarded by a fence epoch:

- ``claim_advance`` records the intent to run one advance. A committed
  operation idempotently returns its unique output; a concurrent claim of
  the same operation under the current fence is a retryable
  ``AdvanceInProgress``; a stale started operation under an OLDER fence is
  taken over (resumed) by the current fence holder.
- ``commit_advance`` publishes the output checkpoint and moves the session's
  committed head. Commits under a superseded fence are refused, the step
  cursor strictly increases, and the checkpoint's covered journal watermark
  can never regress or run ahead of the stream.
- ``abort_advance`` is how a fence holder supersedes a stale claim so a new
  operation id can be claimed.

Every state change appends its durable fact to the session's journal stream
in the SAME SQLite transaction that moves the ledger tables (the host-owned
append/transaction API, design section 9.2), so the ledger and the journal
can never disagree. All state lives in the journal's SQLite file: one
durable training substrate, one fsync domain.

``session_attempt`` on appended events currently mirrors the fence epoch:
the fence holder IS the session's supervising incarnation, and a separate
attempt counter would carry no extra information until supervision and
fencing can change hands independently.

HTTP surface (present only when create_app got a TrainingSessionStore);
reads only - mutation crosses the worker boundary, never this API:

- GET /api/training/sessions/{sessionId}?scope=              record + handle
- GET /api/training/sessions/{sessionId}/events?after=&limit= journal replay
- GET /api/training/sessions/{sessionId}/checkpoints          committed lineage

Scoping mirrors history: sessions are scope-stamped at creation, a wrong
scope is a plain 404, and token principals need the ``training:read``
capability.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass

from aiohttp import web
from dinkster_protocol.training import (
    TrainingEventName,
    TrainingJournalEvent,
    TrainingSessionHandle,
    is_durable_training_event,
    is_training_checkpoint_digest,
    is_training_operation_id,
    is_training_session_id,
    training_session_stream,
)

from .auth import LOCAL_PRINCIPAL, principal_for, resolve_scope
from .journal import JournalPage, JournalStore, JournalTransaction

__all__ = [
    "TRAINING_SESSIONS_KEY",
    "AdvanceInProgress",
    "StaleTrainingFence",
    "TrainingCheckpointRecord",
    "TrainingLineageConflict",
    "TrainingOperationRecord",
    "TrainingSessionError",
    "TrainingSessionRecord",
    "TrainingSessionStore",
    "UnknownTrainingSession",
    "add_training_routes",
]


class TrainingSessionError(Exception):
    """Base for supervisor refusals. Every subclass is a durable-ledger
    verdict, not a transport error: retrying the identical request yields
    the identical verdict unless the ledger moved."""


class UnknownTrainingSession(TrainingSessionError):
    """The session id names no known session."""


class StaleTrainingFence(TrainingSessionError):
    """The caller's fence epoch has been superseded. The holder of the
    current epoch owns the session; a fenced-out worker must stop, never
    retry with the same epoch."""


class AdvanceInProgress(TrainingSessionError):
    """Retryable: this operation is already claimed under the current
    fence. The claimant is (or recently was) running it; retry after it
    commits, aborts, or the fence changes hands."""


class TrainingLineageConflict(TrainingSessionError):
    """The request contradicts the durable lineage ledger: wrong input
    checkpoint, conflicting facts for an existing identity, a regressing
    cursor or watermark, or an operation in a state that cannot satisfy
    the request."""


@dataclass(frozen=True)
class TrainingSessionRecord:
    """One session's supervisor-owned state: identity facts fixed at
    creation plus the committed head the fence holder advances."""

    session_id: str
    scope: str
    principal_id: str
    created_at: float
    fence_epoch: int
    config_digest: str
    extension_snapshot_digest: str
    committed_manifest_digest: str
    committed_step_cursor: int
    committed_journal_seq: int
    state: str  # "active" | "completed"

    def handle(self) -> TrainingSessionHandle:
        """The graph-boundary handle for the committed head. Its
        ``journal_seq`` is the head checkpoint's covered watermark, exactly
        as commit recorded it."""
        return TrainingSessionHandle(
            session_id=self.session_id,
            checkpoint_manifest_digest=self.committed_manifest_digest,
            step_cursor=self.committed_step_cursor,
            config_digest=self.config_digest,
            session_extension_snapshot_digest=self.extension_snapshot_digest,
            journal_seq=self.committed_journal_seq,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "sessionId": self.session_id,
            "scope": self.scope,
            "principalId": self.principal_id,
            "createdAt": self.created_at,
            "fenceEpoch": self.fence_epoch,
            "configDigest": self.config_digest,
            "sessionExtensionSnapshotDigest": self.extension_snapshot_digest,
            "committedManifestDigest": self.committed_manifest_digest,
            "committedStepCursor": self.committed_step_cursor,
            "committedJournalSeq": self.committed_journal_seq,
            "state": self.state,
        }


@dataclass(frozen=True)
class TrainingCheckpointRecord:
    """One committed checkpoint in the session's lineage. The initial
    checkpoint has an empty parent and operation id."""

    session_id: str
    manifest_digest: str
    parent_digest: str
    step_cursor: int
    journal_seq: int
    operation_id: str
    committed_at: float

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "sessionId": self.session_id,
            "manifestDigest": self.manifest_digest,
            "stepCursor": self.step_cursor,
            "journalSeq": self.journal_seq,
            "committedAt": self.committed_at,
        }
        if self.parent_digest:
            wire["parentDigest"] = self.parent_digest
        if self.operation_id:
            wire["operationId"] = self.operation_id
        return wire


@dataclass(frozen=True)
class TrainingOperationRecord:
    """One advance operation's ledger row: the claim, its fence, and (once
    resolved) its unique outcome."""

    session_id: str
    operation_id: str
    input_manifest_digest: str
    status: str  # "started" | "committed" | "aborted"
    fence_epoch: int
    recovery_checkpoint_digest: str
    output_manifest_digest: str
    output_step_cursor: int
    output_journal_seq: int
    claimed_at: float
    resolved_at: float | None
    paused: bool = False
    """True while a started operation is durably stopped at a safe point;
    only such an operation may be continued by a same-fence re-claim."""

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "sessionId": self.session_id,
            "operationId": self.operation_id,
            "inputManifestDigest": self.input_manifest_digest,
            "status": self.status,
            "fenceEpoch": self.fence_epoch,
            "claimedAt": self.claimed_at,
        }
        if self.paused:
            wire["paused"] = True
        if self.recovery_checkpoint_digest:
            wire["recoveryCheckpointDigest"] = self.recovery_checkpoint_digest
        if self.status == "committed":
            wire["outputManifestDigest"] = self.output_manifest_digest
            wire["outputStepCursor"] = self.output_step_cursor
            wire["outputJournalSeq"] = self.output_journal_seq
        if self.resolved_at is not None:
            wire["resolvedAt"] = self.resolved_at
        return wire


def _session_from_row(row: sqlite3.Row) -> TrainingSessionRecord:
    return TrainingSessionRecord(
        session_id=row["session_id"],
        scope=row["scope"],
        principal_id=row["principal"],
        created_at=row["created_at"],
        fence_epoch=row["fence_epoch"],
        config_digest=row["config_digest"],
        extension_snapshot_digest=row["extension_snapshot_digest"],
        committed_manifest_digest=row["committed_manifest_digest"],
        committed_step_cursor=row["committed_step_cursor"],
        committed_journal_seq=row["committed_journal_seq"],
        state=row["state"],
    )


def _checkpoint_from_row(row: sqlite3.Row) -> TrainingCheckpointRecord:
    return TrainingCheckpointRecord(
        session_id=row["session_id"],
        manifest_digest=row["manifest_digest"],
        parent_digest=row["parent_digest"],
        step_cursor=row["step_cursor"],
        journal_seq=row["journal_seq"],
        operation_id=row["operation_id"],
        committed_at=row["committed_at"],
    )


def _operation_from_row(row: sqlite3.Row) -> TrainingOperationRecord:
    return TrainingOperationRecord(
        session_id=row["session_id"],
        operation_id=row["operation_id"],
        input_manifest_digest=row["input_manifest_digest"],
        status=row["status"],
        fence_epoch=row["fence_epoch"],
        recovery_checkpoint_digest=row["recovery_checkpoint_digest"],
        output_manifest_digest=row["output_manifest_digest"],
        output_step_cursor=row["output_step_cursor"],
        output_journal_seq=row["output_journal_seq"],
        claimed_at=row["claimed_at"],
        resolved_at=row["resolved_at"],
        paused=bool(row["paused"]),
    )


def _written_at(timestamp: float | None) -> float:
    return time.time() if timestamp is None else timestamp


def _append_event(
    txn: JournalTransaction,
    *,
    session_id: str,
    name: TrainingEventName,
    timestamp: float,
    fence_epoch: int,
    advance_id: str = "",
    phase: str = "",
    data: Mapping[str, object] | None = None,
) -> int:
    """Append one supervisor fact to the session's stream inside the open
    transaction and return its assigned seq. The TrainingJournalEvent
    constructor is the validation gate: nothing lands in the journal that
    replay could refuse to decode."""
    event = TrainingJournalEvent(
        session_id=session_id,
        name=name,
        timestamp=timestamp,
        advance_id=advance_id,
        phase=phase,
        session_attempt=fence_epoch,
        fence_epoch=fence_epoch,
        data=data if data is not None else {},
    )
    record = txn.append(
        training_session_stream(session_id),
        event.name.value,
        event.to_wire(),
        durable=is_durable_training_event(event.name),
        timestamp=event.timestamp,
    )
    return record.seq


class TrainingSessionStore:
    """The durable session/checkpoint/operation ledger, living in the SAME
    SQLite file as the journal it appends to (one transaction moves both).

    The store takes ownership of the JournalStore it is given: ``close``
    closes it. Same threading contract as the journal itself - sync methods,
    callers cross via asyncio.to_thread."""

    def __init__(self, journal: JournalStore) -> None:
        self._journal = journal
        journal.transact(self._create_tables)

    def close(self) -> None:
        self._journal.close()

    @staticmethod
    def _create_tables(txn: JournalTransaction) -> None:
        txn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                principal TEXT NOT NULL,
                created_at REAL NOT NULL,
                fence_epoch INTEGER NOT NULL,
                config_digest TEXT NOT NULL,
                extension_snapshot_digest TEXT NOT NULL,
                committed_manifest_digest TEXT NOT NULL,
                committed_step_cursor INTEGER NOT NULL,
                committed_journal_seq INTEGER NOT NULL,
                state TEXT NOT NULL
            )
            """
        )
        txn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_checkpoints (
                session_id TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                parent_digest TEXT NOT NULL,
                step_cursor INTEGER NOT NULL,
                journal_seq INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                committed_at REAL NOT NULL,
                PRIMARY KEY (session_id, manifest_digest)
            )
            """
        )
        txn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_operations (
                session_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                input_manifest_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                fence_epoch INTEGER NOT NULL,
                recovery_checkpoint_digest TEXT NOT NULL,
                output_manifest_digest TEXT NOT NULL,
                output_step_cursor INTEGER NOT NULL,
                output_journal_seq INTEGER NOT NULL,
                claimed_at REAL NOT NULL,
                resolved_at REAL,
                paused INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (session_id, operation_id)
            )
            """
        )
        columns = {row["name"] for row in txn.execute("PRAGMA table_info(training_operations)")}
        if "paused" not in columns:
            txn.execute(
                "ALTER TABLE training_operations ADD COLUMN paused INTEGER NOT NULL DEFAULT 0"
            )

    # -- internal row access (inside an open transaction) -----------------

    @staticmethod
    def _session_row(txn: JournalTransaction, session_id: str) -> sqlite3.Row | None:
        return txn.execute(
            "SELECT * FROM training_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    @staticmethod
    def _require_session(txn: JournalTransaction, session_id: str) -> sqlite3.Row:
        row = TrainingSessionStore._session_row(txn, session_id)
        if row is None:
            raise UnknownTrainingSession(f"no training session {session_id!r}")
        return row

    @staticmethod
    def _operation_row(
        txn: JournalTransaction, session_id: str, operation_id: str
    ) -> sqlite3.Row | None:
        return txn.execute(
            "SELECT * FROM training_operations WHERE session_id = ? AND operation_id = ?",
            (session_id, operation_id),
        ).fetchone()

    @staticmethod
    def _check_fence(row: sqlite3.Row, fence_epoch: int) -> None:
        if type(fence_epoch) is not int or fence_epoch < 1:
            raise ValueError("fence_epoch must be an int >= 1")
        if fence_epoch != row["fence_epoch"]:
            raise StaleTrainingFence(
                f"fence epoch {fence_epoch} is not the session's current epoch {row['fence_epoch']}"
            )

    # -- mutations ---------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        *,
        scope: str,
        principal_id: str = "local",
        config_digest: str,
        extension_snapshot_digest: str,
        initial_manifest_digest: str,
        timestamp: float | None = None,
    ) -> TrainingSessionRecord:
        """Record a new session and its initial checkpoint. Idempotent for
        an identical repeat (same id, same identity facts); the same id
        with different facts is a lineage conflict."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not scope.strip() or scope != scope.strip():
            raise ValueError("scope must be a non-empty trimmed string")
        if not principal_id.strip():
            raise ValueError("principal_id must be a non-empty string")
        for label, digest in (
            ("config_digest", config_digest),
            ("extension_snapshot_digest", extension_snapshot_digest),
            ("initial_manifest_digest", initial_manifest_digest),
        ):
            if not is_training_checkpoint_digest(digest):
                raise ValueError(f"{label} must be 'blake3:' + 64 lowercase hex")

        def run(txn: JournalTransaction) -> TrainingSessionRecord:
            existing = self._session_row(txn, session_id)
            if existing is not None:
                initial = txn.execute(
                    "SELECT manifest_digest FROM training_checkpoints"
                    " WHERE session_id = ? AND operation_id = ''",
                    (session_id,),
                ).fetchone()
                same = (
                    existing["scope"] == scope
                    and existing["principal"] == principal_id
                    and existing["config_digest"] == config_digest
                    and existing["extension_snapshot_digest"] == extension_snapshot_digest
                    and initial is not None
                    and initial["manifest_digest"] == initial_manifest_digest
                )
                if same:
                    return _session_from_row(existing)
                raise TrainingLineageConflict(
                    f"session {session_id!r} already exists with different identity facts"
                )
            written_at = _written_at(timestamp)
            seq = _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.SESSION_CREATED,
                timestamp=written_at,
                fence_epoch=1,
                data={
                    "scope": scope,
                    "principalId": principal_id,
                    "configDigest": config_digest,
                    "sessionExtensionSnapshotDigest": extension_snapshot_digest,
                    "initialManifestDigest": initial_manifest_digest,
                },
            )
            txn.execute(
                "INSERT INTO training_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    scope,
                    principal_id,
                    written_at,
                    1,
                    config_digest,
                    extension_snapshot_digest,
                    initial_manifest_digest,
                    0,
                    seq,
                    "active",
                ),
            )
            txn.execute(
                "INSERT INTO training_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, initial_manifest_digest, "", 0, seq, "", written_at),
            )
            row = self._require_session(txn, session_id)
            return _session_from_row(row)

        return self._journal.transact(run)

    def acquire_fence(
        self, session_id: str, *, timestamp: float | None = None
    ) -> TrainingSessionRecord:
        """Bump the fence epoch: the caller becomes the session's exclusive
        writer and every holder of an older epoch is fenced out."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")

        def run(txn: JournalTransaction) -> TrainingSessionRecord:
            row = self._require_session(txn, session_id)
            if row["state"] != "active":
                raise TrainingLineageConflict(f"session {session_id!r} is completed")
            new_epoch = row["fence_epoch"] + 1
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.PHASE_CHANGED,
                timestamp=written_at,
                fence_epoch=new_epoch,
                phase="fence-acquired",
                data={"previousFenceEpoch": row["fence_epoch"]},
            )
            txn.execute(
                "UPDATE training_sessions SET fence_epoch = ? WHERE session_id = ?",
                (new_epoch, session_id),
            )
            return _session_from_row(self._require_session(txn, session_id))

        return self._journal.transact(run)

    def claim_advance(
        self,
        session_id: str,
        operation_id: str,
        *,
        input_manifest_digest: str,
        fence_epoch: int,
        timestamp: float | None = None,
    ) -> TrainingOperationRecord:
        """Claim one advance operation for execution under the current
        fence. A committed identical claim returns its recorded outcome; a
        live claim under the same fence raises retryable AdvanceInProgress;
        a stale started claim under an older fence is taken over by this
        fence holder (resumed, not restarted)."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not is_training_operation_id(operation_id):
            raise ValueError("operation_id must be 32-64 lowercase hex characters")
        if not is_training_checkpoint_digest(input_manifest_digest):
            raise ValueError("input_manifest_digest must be 'blake3:' + 64 lowercase hex")

        def run(txn: JournalTransaction) -> TrainingOperationRecord:
            session = self._require_session(txn, session_id)
            existing = self._operation_row(txn, session_id, operation_id)
            if existing is not None:
                if existing["input_manifest_digest"] != input_manifest_digest:
                    raise TrainingLineageConflict(
                        f"operation {operation_id!r} was claimed with a different input checkpoint"
                    )
                # Recovery replay: a committed operation returns its unique
                # recorded outcome under ANY fence and session state - a
                # lost reply stays recoverable after supersession or
                # completion.
                if existing["status"] == "committed":
                    return _operation_from_row(existing)
                if existing["status"] == "aborted":
                    raise TrainingLineageConflict(
                        f"operation {operation_id!r} was aborted; claim a new operation id"
                    )
                if session["state"] != "active":
                    raise TrainingLineageConflict(f"session {session_id!r} is completed")
                self._check_fence(session, fence_epoch)
                if existing["fence_epoch"] == fence_epoch:
                    # Only a durably paused operation is safe to continue:
                    # a started-and-not-paused claim may still be running,
                    # and continuing it would double-step the trajectory.
                    if not existing["paused"]:
                        raise AdvanceInProgress(
                            f"operation {operation_id!r} is already claimed under"
                            f" fence epoch {fence_epoch}"
                        )
                    written_at = _written_at(timestamp)
                    _append_event(
                        txn,
                        session_id=session_id,
                        name=TrainingEventName.ADVANCE_RESUMED,
                        timestamp=written_at,
                        fence_epoch=fence_epoch,
                        advance_id=operation_id,
                        data={"resumedFromPause": True},
                    )
                    txn.execute(
                        "UPDATE training_operations SET paused = 0"
                        " WHERE session_id = ? AND operation_id = ?",
                        (session_id, operation_id),
                    )
                    row = self._operation_row(txn, session_id, operation_id)
                    assert row is not None
                    return _operation_from_row(row)
                # Started under an older fence: the current fence holder
                # takes the claim over and resumes it.
                written_at = _written_at(timestamp)
                _append_event(
                    txn,
                    session_id=session_id,
                    name=TrainingEventName.ADVANCE_RESUMED,
                    timestamp=written_at,
                    fence_epoch=fence_epoch,
                    advance_id=operation_id,
                    data={"previousFenceEpoch": existing["fence_epoch"]},
                )
                txn.execute(
                    "UPDATE training_operations SET fence_epoch = ?, paused = 0"
                    " WHERE session_id = ? AND operation_id = ?",
                    (fence_epoch, session_id, operation_id),
                )
                row = self._operation_row(txn, session_id, operation_id)
                assert row is not None
                return _operation_from_row(row)
            if session["state"] != "active":
                raise TrainingLineageConflict(f"session {session_id!r} is completed")
            self._check_fence(session, fence_epoch)
            blocking = txn.execute(
                "SELECT operation_id FROM training_operations"
                " WHERE session_id = ? AND status = 'started' LIMIT 1",
                (session_id,),
            ).fetchone()
            if blocking is not None:
                raise TrainingLineageConflict(
                    f"operation {blocking['operation_id']!r} is still started;"
                    " abort it under the fence to supersede it"
                )
            if input_manifest_digest != session["committed_manifest_digest"]:
                raise TrainingLineageConflict(
                    "input checkpoint is stale or unknown: the committed head is"
                    f" {session['committed_manifest_digest']!r}"
                )
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.ADVANCE_STARTED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={"inputManifestDigest": input_manifest_digest},
            )
            txn.execute(
                "INSERT INTO training_operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    operation_id,
                    input_manifest_digest,
                    "started",
                    fence_epoch,
                    "",
                    "",
                    0,
                    0,
                    written_at,
                    None,
                    0,
                ),
            )
            row = self._operation_row(txn, session_id, operation_id)
            assert row is not None
            return _operation_from_row(row)

        return self._journal.transact(run)

    def set_recovery_checkpoint(
        self,
        session_id: str,
        operation_id: str,
        *,
        fence_epoch: int,
        manifest_digest: str,
        timestamp: float | None = None,
    ) -> TrainingOperationRecord:
        """Publish the operation's latest mid-advance recovery checkpoint:
        a resume point, never a lineage commit."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not is_training_operation_id(operation_id):
            raise ValueError("operation_id must be 32-64 lowercase hex characters")
        if not is_training_checkpoint_digest(manifest_digest):
            raise ValueError("manifest_digest must be 'blake3:' + 64 lowercase hex")

        def run(txn: JournalTransaction) -> TrainingOperationRecord:
            session = self._require_session(txn, session_id)
            if session["state"] != "active":
                raise TrainingLineageConflict(f"session {session_id!r} is completed")
            self._check_fence(session, fence_epoch)
            existing = self._operation_row(txn, session_id, operation_id)
            if existing is None:
                raise TrainingLineageConflict(f"operation {operation_id!r} was never claimed")
            if existing["status"] != "started":
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} is {existing['status']}, not started"
                )
            if existing["fence_epoch"] != fence_epoch:
                raise StaleTrainingFence(
                    f"operation {operation_id!r} is held under fence epoch"
                    f" {existing['fence_epoch']}; claim it to resume"
                )
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={"manifestDigest": manifest_digest},
            )
            txn.execute(
                "UPDATE training_operations SET recovery_checkpoint_digest = ?"
                " WHERE session_id = ? AND operation_id = ?",
                (manifest_digest, session_id, operation_id),
            )
            row = self._operation_row(txn, session_id, operation_id)
            assert row is not None
            return _operation_from_row(row)

        return self._journal.transact(run)

    def pause_advance(
        self,
        session_id: str,
        operation_id: str,
        *,
        fence_epoch: int,
        reason: str = "",
        timestamp: float | None = None,
    ) -> TrainingOperationRecord:
        """Durably acknowledge that the operation reached a safe point and
        stopped (training-design.md 9.3). The operation stays ``started``
        holding its recovery pointer, and resuming is an ordinary claim
        (the same fence may continue only a paused claim; a new fence takes
        any started claim over). Requires a published recovery checkpoint:
        without a resume point a pause acknowledgement would be a lie.
        Idempotent while already paused."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not is_training_operation_id(operation_id):
            raise ValueError("operation_id must be 32-64 lowercase hex characters")

        def run(txn: JournalTransaction) -> TrainingOperationRecord:
            session = self._require_session(txn, session_id)
            if session["state"] != "active":
                raise TrainingLineageConflict(f"session {session_id!r} is completed")
            self._check_fence(session, fence_epoch)
            existing = self._operation_row(txn, session_id, operation_id)
            if existing is None:
                raise TrainingLineageConflict(f"operation {operation_id!r} was never claimed")
            if existing["status"] != "started":
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} is {existing['status']}, not started"
                )
            if existing["fence_epoch"] != fence_epoch:
                raise StaleTrainingFence(
                    f"operation {operation_id!r} is held under fence epoch"
                    f" {existing['fence_epoch']}; claim it to resume"
                )
            if existing["paused"]:
                return _operation_from_row(existing)
            if not existing["recovery_checkpoint_digest"]:
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} has no recovery checkpoint;"
                    " publish one before pausing"
                )
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.ADVANCE_PAUSED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={"reason": reason} if reason else {},
            )
            txn.execute(
                "UPDATE training_operations SET paused = 1"
                " WHERE session_id = ? AND operation_id = ?",
                (session_id, operation_id),
            )
            row = self._operation_row(txn, session_id, operation_id)
            assert row is not None
            return _operation_from_row(row)

        return self._journal.transact(run)

    def commit_advance(
        self,
        session_id: str,
        operation_id: str,
        *,
        fence_epoch: int,
        output_manifest_digest: str,
        output_step_cursor: int,
        covered_journal_seq: int,
        timestamp: float | None = None,
    ) -> TrainingOperationRecord:
        """Publish the operation's output checkpoint and advance the
        committed head. Idempotent for an identical repeat. The step cursor
        strictly increases; ``covered_journal_seq`` is the durable watermark
        the checkpoint covers and can neither regress below the session's
        committed watermark nor exceed the stream's latest seq."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not is_training_operation_id(operation_id):
            raise ValueError("operation_id must be 32-64 lowercase hex characters")
        if not is_training_checkpoint_digest(output_manifest_digest):
            raise ValueError("output_manifest_digest must be 'blake3:' + 64 lowercase hex")
        if type(output_step_cursor) is not int or output_step_cursor < 0:
            raise ValueError("output_step_cursor must be a non-negative int")
        if type(covered_journal_seq) is not int or covered_journal_seq < 0:
            raise ValueError("covered_journal_seq must be a non-negative int")

        def run(txn: JournalTransaction) -> TrainingOperationRecord:
            session = self._require_session(txn, session_id)
            existing = self._operation_row(txn, session_id, operation_id)
            if existing is None:
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} was never claimed; commit refused"
                )
            # Recovery replay: an identical repeat of a recorded commit
            # returns the outcome under ANY fence and session state - a
            # lost commit reply stays recoverable after supersession or
            # completion.
            if existing["status"] == "committed":
                if (
                    existing["output_manifest_digest"] == output_manifest_digest
                    and existing["output_step_cursor"] == output_step_cursor
                    and existing["output_journal_seq"] == covered_journal_seq
                ):
                    return _operation_from_row(existing)
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} already committed a different output"
                )
            if existing["status"] == "aborted":
                raise TrainingLineageConflict(f"operation {operation_id!r} was aborted")
            if session["state"] != "active":
                raise TrainingLineageConflict(f"session {session_id!r} is completed")
            self._check_fence(session, fence_epoch)
            if existing["fence_epoch"] != fence_epoch:
                raise StaleTrainingFence(
                    f"operation {operation_id!r} is held under fence epoch"
                    f" {existing['fence_epoch']}; claim it to resume"
                )
            if output_step_cursor <= session["committed_step_cursor"]:
                raise TrainingLineageConflict(
                    f"step cursor must increase: committed head is at"
                    f" {session['committed_step_cursor']}, commit offered {output_step_cursor}"
                )
            latest = txn.latest_seq(training_session_stream(session_id))
            if covered_journal_seq < session["committed_journal_seq"]:
                raise TrainingLineageConflict(
                    "covered journal watermark regresses below the committed"
                    f" watermark {session['committed_journal_seq']}"
                )
            if covered_journal_seq > latest:
                raise TrainingLineageConflict(
                    f"covered journal watermark {covered_journal_seq} exceeds the"
                    f" stream's latest seq {latest}"
                )
            duplicate = txn.execute(
                "SELECT 1 FROM training_checkpoints WHERE session_id = ? AND manifest_digest = ?",
                (session_id, output_manifest_digest),
            ).fetchone()
            if duplicate is not None:
                raise TrainingLineageConflict(
                    f"checkpoint {output_manifest_digest!r} is already recorded"
                    " in this session's lineage"
                )
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.CHECKPOINT_PUBLISHED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={
                    "manifestDigest": output_manifest_digest,
                    "parentDigest": existing["input_manifest_digest"],
                    "stepCursor": output_step_cursor,
                    "coveredJournalSeq": covered_journal_seq,
                },
            )
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.ADVANCE_COMMITTED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={
                    "outputManifestDigest": output_manifest_digest,
                    "stepCursor": output_step_cursor,
                },
            )
            txn.execute(
                "INSERT INTO training_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    output_manifest_digest,
                    existing["input_manifest_digest"],
                    output_step_cursor,
                    covered_journal_seq,
                    operation_id,
                    written_at,
                ),
            )
            txn.execute(
                "UPDATE training_operations SET status = 'committed', paused = 0,"
                " output_manifest_digest = ?, output_step_cursor = ?,"
                " output_journal_seq = ?, resolved_at = ?"
                " WHERE session_id = ? AND operation_id = ?",
                (
                    output_manifest_digest,
                    output_step_cursor,
                    covered_journal_seq,
                    written_at,
                    session_id,
                    operation_id,
                ),
            )
            txn.execute(
                "UPDATE training_sessions SET committed_manifest_digest = ?,"
                " committed_step_cursor = ?, committed_journal_seq = ?"
                " WHERE session_id = ?",
                (output_manifest_digest, output_step_cursor, covered_journal_seq, session_id),
            )
            row = self._operation_row(txn, session_id, operation_id)
            assert row is not None
            return _operation_from_row(row)

        return self._journal.transact(run)

    def abort_advance(
        self,
        session_id: str,
        operation_id: str,
        *,
        fence_epoch: int,
        reason: str = "",
        timestamp: float | None = None,
    ) -> TrainingOperationRecord:
        """Resolve a started operation as aborted. The CURRENT fence holder
        may abort any started claim, including one held under an older
        epoch - that is how a stale claim is superseded so a new operation
        id becomes claimable. Idempotent when already aborted."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        if not is_training_operation_id(operation_id):
            raise ValueError("operation_id must be 32-64 lowercase hex characters")

        def run(txn: JournalTransaction) -> TrainingOperationRecord:
            session = self._require_session(txn, session_id)
            existing = self._operation_row(txn, session_id, operation_id)
            if existing is None:
                raise TrainingLineageConflict(f"operation {operation_id!r} was never claimed")
            if existing["status"] == "aborted":
                return _operation_from_row(existing)
            if existing["status"] == "committed":
                raise TrainingLineageConflict(
                    f"operation {operation_id!r} already committed; commits are final"
                )
            self._check_fence(session, fence_epoch)
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.ADVANCE_ABORTED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
                advance_id=operation_id,
                data={"reason": reason} if reason else {},
            )
            txn.execute(
                "UPDATE training_operations SET status = 'aborted', paused = 0, resolved_at = ?"
                " WHERE session_id = ? AND operation_id = ?",
                (written_at, session_id, operation_id),
            )
            row = self._operation_row(txn, session_id, operation_id)
            assert row is not None
            return _operation_from_row(row)

        return self._journal.transact(run)

    def complete_session(
        self, session_id: str, *, fence_epoch: int, timestamp: float | None = None
    ) -> TrainingSessionRecord:
        """Mark the session terminal. Refused while any operation is still
        started (resolve it first); idempotent once completed."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")

        def run(txn: JournalTransaction) -> TrainingSessionRecord:
            session = self._require_session(txn, session_id)
            if session["state"] == "completed":
                return _session_from_row(session)
            self._check_fence(session, fence_epoch)
            started = txn.execute(
                "SELECT operation_id FROM training_operations"
                " WHERE session_id = ? AND status = 'started' LIMIT 1",
                (session_id,),
            ).fetchone()
            if started is not None:
                raise TrainingLineageConflict(
                    f"operation {started['operation_id']!r} is still started;"
                    " commit or abort it before completing the session"
                )
            written_at = _written_at(timestamp)
            _append_event(
                txn,
                session_id=session_id,
                name=TrainingEventName.SESSION_COMPLETED,
                timestamp=written_at,
                fence_epoch=fence_epoch,
            )
            txn.execute(
                "UPDATE training_sessions SET state = 'completed' WHERE session_id = ?",
                (session_id,),
            )
            return _session_from_row(self._require_session(txn, session_id))

        return self._journal.transact(run)

    # -- reads -------------------------------------------------------------

    def get_session(self, session_id: str) -> TrainingSessionRecord | None:
        """Lookup by id; a malformed id is a plain miss (reads fail soft)."""
        if not is_training_session_id(session_id):
            return None

        def run(txn: JournalTransaction) -> TrainingSessionRecord | None:
            row = self._session_row(txn, session_id)
            return _session_from_row(row) if row is not None else None

        return self._journal.transact(run)

    def get_operation(self, session_id: str, operation_id: str) -> TrainingOperationRecord | None:
        if not is_training_session_id(session_id) or not is_training_operation_id(operation_id):
            return None

        def run(txn: JournalTransaction) -> TrainingOperationRecord | None:
            row = self._operation_row(txn, session_id, operation_id)
            return _operation_from_row(row) if row is not None else None

        return self._journal.transact(run)

    def list_checkpoints(self, session_id: str) -> tuple[TrainingCheckpointRecord, ...]:
        """The committed lineage, oldest first (the initial checkpoint's
        step cursor is 0 and cursors strictly increase)."""
        if not is_training_session_id(session_id):
            return ()

        def run(txn: JournalTransaction) -> tuple[TrainingCheckpointRecord, ...]:
            rows = txn.execute(
                "SELECT * FROM training_checkpoints WHERE session_id = ? ORDER BY step_cursor ASC",
                (session_id,),
            ).fetchall()
            return tuple(_checkpoint_from_row(row) for row in rows)

        return self._journal.transact(run)

    def read_events(self, session_id: str, *, after: int = 0, limit: int = 1000) -> JournalPage:
        """Replay the session's journal stream (see JournalStore.read)."""
        if not is_training_session_id(session_id):
            raise ValueError("session_id must be 32-64 lowercase hex characters")
        return self._journal.read(training_session_stream(session_id), after=after, limit=limit)


# -- HTTP surface --------------------------------------------------------


TRAINING_SESSIONS_KEY: web.AppKey[TrainingSessionStore] = web.AppKey("training_sessions")


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _training_scope(request: web.Request) -> str:
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
        return resolve_scope(principal, "training:read", None)
    if not value or value != value.strip() or any(ch.isspace() for ch in value):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "scope must be a non-empty whitespace-free string"}),
            content_type="application/json",
        )
    return value


def _training_scope_allowed(request: web.Request, scope: str) -> bool:
    principal = principal_for(request)
    return principal is LOCAL_PRINCIPAL or principal.allows_in(scope, "training:read")


async def _scoped_session(request: web.Request) -> TrainingSessionRecord | None:
    """The request's session, or None when it does not exist IN THIS SCOPE.
    A wrong scope is indistinguishable from a miss."""
    store = request.app[TRAINING_SESSIONS_KEY]
    scope = _training_scope(request)
    if not _training_scope_allowed(request, scope):
        return None
    record = await asyncio.to_thread(store.get_session, request.match_info["session_id"])
    if record is None or record.scope != scope:
        return None
    return record


async def handle_training_session_get(request: web.Request) -> web.Response:
    record = await _scoped_session(request)
    if record is None:
        return _json_error(404, "no such training session")
    wire = record.to_wire()
    wire["handle"] = record.handle().to_wire()
    return web.json_response(wire)


async def handle_training_session_events(request: web.Request) -> web.Response:
    record = await _scoped_session(request)
    if record is None:
        return _json_error(404, "no such training session")
    store = request.app[TRAINING_SESSIONS_KEY]
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
    page = await asyncio.to_thread(
        lambda: store.read_events(record.session_id, after=after, limit=limit)
    )
    return web.json_response(
        {
            "records": [event.to_wire() for event in page.records],
            "latestSeq": page.latest_seq,
            "coalescedBelow": page.coalesced_below,
            "handle": record.handle().to_wire(),
        }
    )


async def handle_training_session_checkpoints(request: web.Request) -> web.Response:
    record = await _scoped_session(request)
    if record is None:
        return _json_error(404, "no such training session")
    store = request.app[TRAINING_SESSIONS_KEY]
    checkpoints = await asyncio.to_thread(store.list_checkpoints, record.session_id)
    return web.json_response({"checkpoints": [checkpoint.to_wire() for checkpoint in checkpoints]})


def add_training_routes(app: web.Application, store: TrainingSessionStore) -> None:
    """Routes only - the store's lifecycle belongs to create_app's cleanup,
    which closes it (and with it the journal file it owns) after the queue
    drains."""
    app[TRAINING_SESSIONS_KEY] = store
    app.router.add_get("/api/training/sessions/{session_id}", handle_training_session_get)
    app.router.add_get("/api/training/sessions/{session_id}/events", handle_training_session_events)
    app.router.add_get(
        "/api/training/sessions/{session_id}/checkpoints", handle_training_session_checkpoints
    )
