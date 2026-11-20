"""Finished-history reference reservations, independent of SQLite write locks."""

from __future__ import annotations

import bisect
import json
import re
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from .client import LibraryError


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", value) is None:
        raise LibraryError("references", "history asset reference is invalid")
    return value


class HistoryReferences:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = FileLock(root / "records-references.lock")
        self._lease: str | None = None
        self._snapshot: tuple[str, ...] = ()

    @contextmanager
    def writing(self) -> Generator[None]:
        # A helper's own reservation must not make its mutations reentrant.
        if self._lease is not None:
            raise LibraryError("busy", "history references are reserved for collection")
        try:
            with self._lock.acquire(timeout=30):
                yield
        except Timeout as error:
            raise LibraryError("busy", "history references are reserved for collection") from error

    def close(self) -> None:
        if self._lease is not None:
            self._lock.release()
            self._lease = None
            self._snapshot = ()

    def dispatch(
        self, db: sqlite3.Connection, operation: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        if operation == "history.references.acquire":
            if set(arguments) - {"limit"}:
                raise LibraryError("invalid", "unknown history reference field")
            if (self._root / "history.sqlite").exists():
                raise LibraryError("migration", "Finish the complete library migration first")
            if self._lease is not None:
                raise LibraryError("busy", "this helper already holds history references")
            try:
                self._lock.acquire(timeout=0)
            except Timeout as error:
                raise LibraryError("busy", "history references are already reserved") from error
            try:
                roots: set[str] = set()
                for (raw,) in db.execute(
                    "SELECT document FROM finished_history WHERE deleted=0"
                ).fetchall():
                    loaded: object = json.loads(raw)
                    if not isinstance(loaded, dict):
                        raise LibraryError("references", "history record is invalid")
                    document = cast("dict[str, object]", loaded)
                    source = document.get("sourceDocument", "")
                    if source != "":
                        roots.add(_digest(source))
                    digests = document.get("assetDigests")
                    if not isinstance(digests, list):
                        raise LibraryError("references", "history asset references are unknown")
                    roots.update(_digest(value) for value in cast("list[object]", digests))
                self._snapshot = tuple(sorted(roots))
                self._lease = uuid.uuid4().hex
                return self._page(arguments)
            except BaseException:
                self._lease = None
                self._snapshot = ()
                self._lock.release()
                raise
        if operation not in ("history.references.page", "history.references.release"):
            raise LibraryError("operation", f"unknown library operation: {operation}")
        allowed = {"lease", "cursor", "limit"} if operation.endswith(".page") else {"lease"}
        if set(arguments) - allowed:
            raise LibraryError("invalid", "unknown history reference field")
        if self._lease is None or arguments.get("lease") != self._lease:
            raise LibraryError("invalid", "history reference reservation is no longer active")
        if operation.endswith(".release"):
            self.close()
            return {}
        return self._page(arguments)

    def _page(self, arguments: dict[str, object]) -> dict[str, object]:
        limit = arguments.get("limit", 200)
        if type(limit) is not int or not 1 <= limit <= 200:
            raise LibraryError("invalid", "history reference page size must be from 1 to 200")
        cursor = arguments.get("cursor")
        start = 0
        if cursor is not None:
            cursor = _digest(cursor)
            start = bisect.bisect_right(self._snapshot, cursor)
            if start == 0 or self._snapshot[start - 1] != cursor:
                raise LibraryError("invalid", "history reference cursor is not in this reservation")
        selected = self._snapshot[start : start + limit]
        return {
            "lease": self._lease,
            "digests": list(selected),
            "cursor": selected[-1] if start + limit < len(self._snapshot) else None,
        }
