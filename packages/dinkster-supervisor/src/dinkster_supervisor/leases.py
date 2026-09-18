"""Persistent, non-expiring ingress ownership assignments."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class LeaseConflictError(RuntimeError):
    """A job reference was already recorded against another owner."""


class LeaseStoreError(RuntimeError):
    """The ownership database could not complete an operation."""


class LeaseStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS key_assignments (
              scope TEXT NOT NULL DEFAULT 'default', client_id TEXT NOT NULL,
              job_id TEXT NOT NULL, owner_id TEXT NOT NULL, created REAL NOT NULL,
              PRIMARY KEY(scope, client_id, job_id));
            CREATE TABLE IF NOT EXISTS job_assignments (
              job_ref TEXT PRIMARY KEY, owner_id TEXT NOT NULL, created REAL NOT NULL);
            """
        )
        return connection

    async def _thread(self, operation: Callable[[], T]) -> T:
        try:
            return await asyncio.to_thread(operation)
        except (sqlite3.Error, OSError) as exc:
            raise LeaseStoreError(f"ownership store unavailable: {exc}") from exc

    async def initialize(self) -> None:
        await self._thread(lambda: self._connect().close())

    async def claim_key(self, scope: str, client_id: str, job_id: str, candidate_owner: str) -> str:
        def work() -> str:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT OR IGNORE INTO key_assignments VALUES (?, ?, ?, ?, ?)",
                    (scope, client_id, job_id, candidate_owner, time.time()),
                )
                row = db.execute(
                    "SELECT owner_id FROM key_assignments "
                    "WHERE scope=? AND client_id=? AND job_id=?",
                    (scope, client_id, job_id),
                ).fetchone()
                db.commit()
                assert row is not None
                return str(row[0])

        return await self._thread(work)

    async def record_job(self, job_ref: str, owner_id: str) -> None:
        def work() -> None:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT OR IGNORE INTO job_assignments VALUES (?, ?, ?)",
                    (job_ref, owner_id, time.time()),
                )
                row = db.execute(
                    "SELECT owner_id FROM job_assignments WHERE job_ref=?", (job_ref,)
                ).fetchone()
                if row is None or row[0] != owner_id:
                    db.rollback()
                    raise LeaseConflictError(
                        f"job {job_ref!r} belongs to "
                        f"{row[0] if row else 'no owner'}, not {owner_id}"
                    )
                db.commit()

        await self._thread(work)

    async def lookup_key(self, scope: str, client_id: str, job_id: str) -> str | None:
        return await self._thread(lambda: self._lookup_key(scope, client_id, job_id))

    def _lookup_key(self, scope: str, client_id: str, job_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT owner_id FROM key_assignments WHERE scope=? AND client_id=? AND job_id=?",
                (scope, client_id, job_id),
            ).fetchone()
            return None if row is None else str(row[0])

    async def lookup_job(self, job_ref: str) -> str | None:
        def work() -> str | None:
            with self._connect() as db:
                row = db.execute(
                    "SELECT owner_id FROM job_assignments WHERE job_ref=?", (job_ref,)
                ).fetchone()
                return None if row is None else str(row[0])

        return await self._thread(work)

    async def delete_key_if_owner(
        self, scope: str, client_id: str, job_id: str, owner_id: str
    ) -> bool:
        def work() -> bool:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                result = db.execute(
                    "DELETE FROM key_assignments WHERE scope=? AND client_id=? "
                    "AND job_id=? AND owner_id=?",
                    (scope, client_id, job_id, owner_id),
                )
                db.commit()
                return result.rowcount == 1

        return await self._thread(work)

    async def delete_job_if_owner(self, job_ref: str, owner_id: str) -> bool:
        def work() -> bool:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                result = db.execute(
                    "DELETE FROM job_assignments WHERE job_ref=? AND owner_id=?",
                    (job_ref, owner_id),
                )
                db.commit()
                return result.rowcount == 1

        return await self._thread(work)
