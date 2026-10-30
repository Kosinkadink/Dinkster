"""Stdlib-only admission check for Windows SQLite mapped-shm recovery."""

from __future__ import annotations

import mmap
import sqlite3
import sys
import tempfile
from pathlib import Path


def check() -> None:
    """Read a WAL database while its old 64 KiB shared-memory mapping survives."""
    if sys.platform != "win32":
        raise RuntimeError("The mapped-shm gate requires native Windows")
    print(f"CPython {sys.version}; SQLite {sqlite3.sqlite_version}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="dinkster-sqlite-") as temporary:
            path = Path(temporary) / "wal.sqlite"
            db = sqlite3.connect(path)
            try:
                if db.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
                    raise RuntimeError("SQLite did not enable WAL")
                db.execute("CREATE TABLE items (value TEXT)")
                db.execute("INSERT INTO items VALUES ('persisted')")
                db.commit()
                with Path(str(path) + "-shm").open("r+b") as shm:
                    shm.truncate(65536)
                    with mmap.mmap(shm.fileno(), 0) as mapping:
                        if len(mapping) != 65536:
                            raise RuntimeError("SQLite probe requires a 64 KiB shm mapping")
                        db.close()
                        for state in ("mapped", "released"):
                            if state == "released":
                                mapping.close()
                            connection = sqlite3.connect(path)
                            try:
                                rows = connection.execute("SELECT * FROM items").fetchall()
                                integrity = connection.execute("PRAGMA integrity_check").fetchall()
                                if rows != [("persisted",)] or integrity != [("ok",)]:
                                    raise RuntimeError(f"{state}: {rows}, {integrity}")
                                print(f"{state}: persisted; integrity_check=ok")
                            finally:
                                connection.close()
            finally:
                db.close()
    except Exception as error:
        raise RuntimeError(
            f"Windows SQLite mapped-shm recovery failed: {error}; "
            "https://sqlite.org/forum/forumpost/f9c68d01e2d7f638; "
            "fix https://sqlite.org/src/info/7c103a0b7d"
        ) from error


if __name__ == "__main__":
    check()
