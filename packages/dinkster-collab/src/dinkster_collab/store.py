"""SQLite persistence for collaborative document sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

from .sessions import DocumentSession, SessionOp


class SessionStore:
    """SQLite-backed storage for session snapshots and retained operations."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    document_kind TEXT NOT NULL DEFAULT 'workflow',
                    snapshot TEXT NOT NULL,
                    snapshot_revision INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            columns = {
                cast(str, row["name"])
                for row in self._conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "document_kind" not in columns:
                self._conn.execute(
                    "ALTER TABLE sessions ADD COLUMN document_kind TEXT NOT NULL DEFAULT 'workflow'"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ops (
                    session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    op_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    base_revision INTEGER NOT NULL,
                    patch TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY(session_id, revision),
                    UNIQUE(session_id, op_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actors (
                    session_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, actor_id)
                )
                """
            )
            op_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(ops)")}
            if "principal_id" not in op_columns:
                self._conn.execute(
                    "ALTER TABLE ops ADD COLUMN principal_id TEXT NOT NULL DEFAULT 'local'"
                )
                self._conn.execute(
                    "UPDATE ops SET principal_id = COALESCE((SELECT principal_id FROM actors"
                    " WHERE actors.session_id = ops.session_id"
                    " AND actors.actor_id = ops.actor_id), 'local')"
                )
            if "actor_kind" not in op_columns:
                self._conn.execute(
                    "ALTER TABLE ops ADD COLUMN actor_kind TEXT NOT NULL DEFAULT 'human'"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_acl (
                    session_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY (session_id, principal_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_policy (
                    session_id TEXT PRIMARY KEY,
                    default_role TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put_session(self, session: DocumentSession) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions"
                " (session_id, scope, document_id, document_kind, snapshot,"
                " snapshot_revision, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    session.scope,
                    session.document_id,
                    session.document_kind,
                    json.dumps(session.snapshot),
                    session.snapshot_revision,
                    session.created_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO session_policy (session_id, default_role) VALUES (?, ?)",
                (session.session_id, session.default_role),
            )
            self._conn.executemany(
                "INSERT INTO session_acl (session_id, principal_id, role) VALUES (?, ?, ?)",
                [
                    (session.session_id, principal_id, role)
                    for principal_id, role in session.acl.items()
                ],
            )

    def replace_acl(self, session_id: str, default_role: str, entries: dict[str, str]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM session_acl WHERE session_id = ?", (session_id,))
            self._conn.executemany(
                "INSERT INTO session_acl (session_id, principal_id, role) VALUES (?, ?, ?)",
                [(session_id, principal_id, role) for principal_id, role in entries.items()],
            )
            self._conn.execute(
                "UPDATE session_policy SET default_role = ? WHERE session_id = ?",
                (default_role, session_id),
            )

    def append_op(self, session_id: str, op: SessionOp, principal_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO actors (session_id, actor_id, principal_id) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id, actor_id) DO NOTHING",
                (session_id, op.actor_id, principal_id),
            )
            self._conn.execute(
                "INSERT INTO ops"
                " (session_id, revision, op_id, actor_id, base_revision, patch, timestamp,"
                " principal_id, actor_kind)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    op.revision,
                    op.op_id,
                    op.actor_id,
                    op.base_revision,
                    json.dumps(op.patch),
                    op.timestamp,
                    principal_id,
                    op.actor_kind,
                ),
            )

    def put_actor(self, session_id: str, actor_id: str, principal_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO actors (session_id, actor_id, principal_id) VALUES (?, ?, ?)",
                (session_id, actor_id, principal_id),
            )

    def checkpoint(self, session_id: str, snapshot_json_text: str, snapshot_revision: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET snapshot = ?, snapshot_revision = ? WHERE session_id = ?",
                (snapshot_json_text, snapshot_revision, session_id),
            )
            self._conn.execute(
                "DELETE FROM ops WHERE session_id = ? AND revision <= ?",
                (session_id, snapshot_revision),
            )

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM ops WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM actors WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM session_acl WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM session_policy WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def load_all(self) -> list[DocumentSession]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                """
                SELECT
                    sessions.session_id,
                    sessions.scope,
                    sessions.document_id,
                    sessions.document_kind,
                    sessions.snapshot,
                    sessions.snapshot_revision,
                    sessions.created_at,
                    ops.revision,
                    ops.op_id,
                    ops.actor_id,
                    ops.base_revision,
                    ops.patch,
                    ops.timestamp,
                    ops.principal_id,
                    ops.actor_kind
                FROM sessions
                LEFT JOIN ops ON ops.session_id = sessions.session_id
                ORDER BY sessions.session_id, ops.revision
                """
            ).fetchall()
            actor_rows = self._conn.execute(
                "SELECT session_id, actor_id, principal_id FROM actors"
            ).fetchall()
            policy_rows = self._conn.execute(
                "SELECT session_id, default_role FROM session_policy"
            ).fetchall()
            acl_rows = self._conn.execute(
                "SELECT session_id, principal_id, role FROM session_acl"
            ).fetchall()

        sessions: dict[str, DocumentSession] = {}
        for row in rows:
            session_id = cast(str, row["session_id"])
            session = sessions.get(session_id)
            if session is None:
                session = DocumentSession(
                    session_id=session_id,
                    scope=cast(str, row["scope"]),
                    document_id=cast(str, row["document_id"]),
                    snapshot=json.loads(cast(str, row["snapshot"])),
                    document_kind=cast(str, row["document_kind"]),
                    snapshot_revision=cast(int, row["snapshot_revision"]),
                    created_at=cast(float, row["created_at"]),
                )
                sessions[session_id] = session
            if row["revision"] is None:
                continue
            patch = cast(list[dict[str, Any]], json.loads(cast(str, row["patch"])))
            op = SessionOp(
                op_id=cast(str, row["op_id"]),
                actor_id=cast(str, row["actor_id"]),
                base_revision=cast(int, row["base_revision"]),
                revision=cast(int, row["revision"]),
                patch=tuple(patch),
                timestamp=cast(float, row["timestamp"]),
                principal_id=cast(str, row["principal_id"]),
                actor_kind=cast(str, row["actor_kind"]),
            )
            session.ops.append(op)
            session.ops_by_id[op.op_id] = op
        for row in actor_rows:
            session = sessions[cast(str, row["session_id"])]
            session.actor_bindings[cast(str, row["actor_id"])] = cast(str, row["principal_id"])
        for row in policy_rows:
            session = sessions[cast(str, row["session_id"])]
            session.default_role = cast(str, row["default_role"])
        for row in acl_rows:
            session = sessions[cast(str, row["session_id"])]
            session.acl[cast(str, row["principal_id"])] = cast(str, row["role"])
        return list(sessions.values())
