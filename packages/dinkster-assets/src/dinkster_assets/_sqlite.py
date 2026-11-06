"""Connections for the shared produced-asset journals."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Literal


def connect_journal(path: Path, *, mode: Literal["rwc", "rw", "ro"] = "rwc") -> sqlite3.Connection:
    """Keep WAL persistent while giving each connection a bounded busy wait.

    Read-only opens never initialize or repair journal mode. WAL readers need
    existing readable -wal/-shm sidecars or permission to create them in the
    containing directory; mode=ro does not make a live database immutable.
    """
    db = sqlite3.connect(path.as_uri() + f"?mode={mode}", uri=True, timeout=30)
    try:
        if mode != "ro":
            deadline = time.monotonic() + 30
            while True:
                try:
                    if db.execute("PRAGMA journal_mode").fetchone() != ("wal",):
                        db.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    # Switching to WAL can return BUSY without invoking SQLite's
                    # busy handler when concurrent creators hold shared locks.
                    remaining = deadline - time.monotonic()
                    if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY or remaining <= 0:
                        raise
                    time.sleep(min(0.01, remaining))
        return db
    except BaseException:
        db.close()
        raise
