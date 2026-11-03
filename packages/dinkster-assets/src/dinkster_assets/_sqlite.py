"""Connections for the shared produced-asset journals."""

from __future__ import annotations

import sqlite3
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
        if mode != "ro" and db.execute("PRAGMA journal_mode").fetchone() != ("wal",):
            db.execute("PRAGMA journal_mode=WAL")
        return db
    except BaseException:
        db.close()
        raise
