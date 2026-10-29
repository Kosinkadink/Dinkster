"""Persistent inventory of pending and verified engine-side produced assets."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .identity import AssetError, require_digest


@dataclass(frozen=True)
class ProducedAssetAdoption:
    digest: str
    size: int
    adopted_at_ns: int | None


class ProducedAssetInventory:
    """Record adoption independently of source acknowledgement and scratch lifetime.

    An entry is not a reference or a deletion grant. Pending entries retain
    recovery state; repeated adoption preserves the original adoption order.
    """

    def __init__(self, root: Path) -> None:
        self._path = root.absolute() / ".produced-assets.sqlite"
        if self._path.is_symlink():
            raise AssetError("produced asset inventory must not be a symlink")

    def pending(self, digest: str, size: int) -> None:
        digest = require_digest(digest)
        if type(size) is not int or size < 0:
            raise AssetError("produced asset size must be a nonnegative integer")
        try:
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS adoptions ("
                    "digest TEXT PRIMARY KEY, size INTEGER NOT NULL, adopted_at_ns INTEGER)"
                )
                db.execute(
                    "INSERT INTO adoptions (digest, size) VALUES (?, ?) "
                    "ON CONFLICT(digest) DO UPDATE SET size = excluded.size "
                    "WHERE adoptions.adopted_at_ns IS NULL",
                    (digest, size),
                )
                row = db.execute(
                    "SELECT size FROM adoptions WHERE digest = ?", (digest,)
                ).fetchone()
                if row is None or row[0] != size:
                    raise AssetError("produced asset adoption sizes disagree")
        except sqlite3.Error as exc:
            raise AssetError("produced asset inventory is unavailable") from exc

    def adopted(self, digest: str, size: int) -> None:
        digest = require_digest(digest)
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True)) as db, db:
                changed = db.execute(
                    "UPDATE adoptions SET adopted_at_ns = COALESCE(adopted_at_ns, ?) "
                    "WHERE digest = ? AND size = ?",
                    (time.time_ns(), digest, size),
                ).rowcount
                if changed != 1:
                    raise AssetError("produced asset adoption has no pending record")
        except sqlite3.Error as exc:
            raise AssetError("produced asset inventory is unavailable") from exc

    def confirm_existing(self, digest: str, size: int) -> None:
        """Repair interrupted adoption after the caller verifies existing vault bytes.

        Existing untracked inputs are not reclassified as produced assets.
        """
        digest = require_digest(digest)
        if not self._path.exists():
            return
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True)) as db, db:
                row = db.execute(
                    "SELECT size, adopted_at_ns FROM adoptions WHERE digest = ?", (digest,)
                ).fetchone()
                if row is None:
                    return
                if row[1] is not None and row[0] != size:
                    raise AssetError("produced asset adoption sizes disagree")
                db.execute(
                    "UPDATE adoptions SET size = ?, adopted_at_ns = COALESCE(adopted_at_ns, ?) "
                    "WHERE digest = ?",
                    (size, time.time_ns(), digest),
                )
        except sqlite3.Error as exc:
            raise AssetError("produced asset inventory is unavailable") from exc

    def entries(self) -> tuple[ProducedAssetAdoption, ...]:
        """Read existing recovery state; missing or unreadable state is not emptiness."""
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute(
                    "SELECT digest, size, adopted_at_ns FROM adoptions "
                    "ORDER BY adopted_at_ns, digest"
                ).fetchall()
        except sqlite3.Error as exc:
            raise AssetError("produced asset inventory is unavailable") from exc
        entries: list[ProducedAssetAdoption] = []
        for digest, size, adopted_at in rows:
            if not isinstance(digest, str):
                raise AssetError("produced asset inventory digest is invalid")
            require_digest(digest)
            if (
                type(size) is not int
                or size < 0
                or (adopted_at is not None and (type(adopted_at) is not int or adopted_at <= 0))
            ):
                raise AssetError("produced asset inventory is invalid")
            entries.append(ProducedAssetAdoption(digest, size, adopted_at))
        return tuple(entries)
