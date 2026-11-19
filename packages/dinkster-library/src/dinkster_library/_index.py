"""Account-local file identities, serialized across helpers before reading bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

from blake3 import blake3
from filelock import FileLock

from .client import LibraryError


def fingerprint(value: os.stat_result) -> str:
    # Windows stat and fstat can expose different meanings for ctime.
    # Device, file identity, size and mtime agree for an unchanged open file.
    return json.dumps(
        [
            os.name,
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns if os.name != "nt" else None,
        ]
    )


def verification(digest: str, value: os.stat_result) -> dict[str, object] | None:
    if value.st_dev == 0 or value.st_ino == 0 or os.name not in ("posix", "nt"):
        return None
    return {
        "digest": digest,
        "scheme": "windows-v1" if os.name == "nt" else "posix-v1",
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtimeNs": value.st_mtime_ns,
        **({"ctimeNs": value.st_ctime_ns} if os.name != "nt" else {}),
    }


def index_file(
    value: object,
    *,
    seeds: tuple[dict[str, object], ...] = (),
    shipped: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise LibraryError("invalid", "file path must be absolute")
    path = Path(os.path.normcase(str(Path(value).resolve(strict=True))))
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    locks = home / "index-locks"
    locks.mkdir(parents=True, exist_ok=True)
    folder = hashlib.sha256(str(path.parent).encode("utf-8")).hexdigest()
    # A waiter must read the committed cache after taking the folder lock,
    # not retain the miss it observed before the other helper finished.
    with (
        FileLock(locks / f"{folder}.lock"),
        closing(sqlite3.connect(home / "digest-index.sqlite", timeout=30)) as db,
    ):
        with db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise LibraryError("format", f"Unsupported digest index format: {version}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS files "
                "(path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            db.execute("PRAGMA user_version=1")
        current = path.stat()
        if not stat.S_ISREG(current.st_mode):
            raise LibraryError("invalid", "hashing requires a regular file")
        expected = fingerprint(current)
        row = db.execute(
            "SELECT fingerprint,digest FROM files WHERE path=?", (str(path),)
        ).fetchone()
        if row and row[0] == expected and current.st_dev != 0 and current.st_ino != 0:
            return {
                "size": current.st_size,
                "mtimeNs": current.st_mtime_ns,
                "digest": row[1],
                "verification": verification(row[1], current),
            }, False
        for seed in seeds:
            digest = seed.get("digest")
            if (
                isinstance(digest, str)
                and re.fullmatch(r"blake3:[0-9a-f]{64}", digest)
                and seed.get("size") == current.st_size
                and seed.get("mtimeNs") == current.st_mtime_ns
                and verification(digest, current) is not None
                and seed.get("verification") == verification(digest, current)
            ):
                with db:
                    db.execute(
                        "INSERT INTO files VALUES (?,?,?) ON CONFLICT(path) "
                        "DO UPDATE SET fingerprint=excluded.fingerprint,digest=excluded.digest",
                        (str(path), expected, digest),
                    )
                return dict(seed), False
        if row is None and shipped is not None:
            digest = shipped.get("digest")
            if (
                isinstance(digest, str)
                and re.fullmatch(r"blake3:[0-9a-f]{64}", digest)
                and type(shipped.get("size")) is int
                and type(shipped.get("mtimeNs")) is int
                and shipped["size"] == current.st_size
                and shipped["mtimeNs"] == current.st_mtime_ns
                and shipped.get("missing") is not True
            ):
                # A copied list locates content, but does not prove its bytes.
                # Never turn its metadata into a verified account-cache row.
                return {
                    "size": current.st_size,
                    "mtimeNs": current.st_mtime_ns,
                    "digest": digest,
                }, False
        # No SQLite write lock is held while bytes are read. Metadata comes
        # from the same descriptor as the digest, and the path must still name it.
        with path.open("rb") as source:
            before = fingerprint(os.fstat(source.fileno()))
            hasher = blake3()
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
            after = fingerprint(os.fstat(source.fileno()))
        published = path.stat()
        if before != after or after != fingerprint(published):
            raise LibraryError("changed", "file changed while being hashed; retry the scan")
        digest = "blake3:" + hasher.hexdigest()
        with db:
            db.execute(
                "INSERT INTO files VALUES (?,?,?) ON CONFLICT(path) "
                "DO UPDATE SET fingerprint=excluded.fingerprint,digest=excluded.digest",
                (str(path), after, digest),
            )
        return {
            "size": published.st_size,
            "mtimeNs": published.st_mtime_ns,
            "digest": digest,
            "verification": verification(digest, published),
        }, True


def hash_file(value: object) -> dict[str, object]:
    row, hashed = index_file(value)
    return {"digest": row["digest"], "hashed": hashed}
