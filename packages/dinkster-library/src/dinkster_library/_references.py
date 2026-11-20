"""Library reference reservations and engine-owned reference manifests."""

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
        raise LibraryError("references", "asset reference is invalid")
    return value


def _manifest(raw: str) -> dict[str, object]:
    value: object = json.loads(raw)
    if not isinstance(value, dict):
        raise LibraryError("references", "engine reference manifest is invalid")
    document = cast("dict[str, object]", value)
    revision = document.get("revision")
    if type(revision) is not int or revision < 1:
        raise LibraryError("references", "engine reference revision is invalid")
    return document


class References:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = FileLock(root / "records-references.lock")
        self._lease: str | None = None
        self._snapshot: tuple[str, ...] = ()
        self._engine: str | None = None
        self._engine_lock: FileLock | None = None

    @contextmanager
    def writing(self) -> Generator[None]:
        # A helper's own reservation must not make its mutations reentrant.
        if self._lease is not None:
            raise LibraryError("busy", "asset references are reserved for collection")
        try:
            with self._lock.acquire(timeout=30):
                yield
        except Timeout as error:
            raise LibraryError("busy", "asset references are reserved for collection") from error

    def close(self) -> None:
        self._release()
        if self._engine_lock is not None:
            self._engine_lock.release()
            self._engine_lock = None
            self._engine = None

    def _release(self) -> None:
        if self._lease is not None:
            self._lock.release()
            self._lease = None
            self._snapshot = ()

    def engine(
        self, db: sqlite3.Connection, operation: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        if operation == "references.engine.attach":
            identifier = arguments.get("engine")
            if (
                set(arguments) != {"engine"}
                or not isinstance(identifier, str)
                or re.fullmatch(r"[0-9a-f]{32}", identifier) is None
            ):
                raise LibraryError("invalid", "engine identity must be 32 lowercase hex digits")
            if self._engine is not None:
                raise LibraryError("invalid", "this helper is already bound to an engine")
            lock = FileLock(self._root / f"engine-references-{identifier}.lock")
            try:
                lock.acquire(timeout=0)
            except Timeout as error:
                raise LibraryError(
                    "busy", "this engine already has a reference publisher"
                ) from error
            try:
                with self.writing(), db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute(
                        "SELECT document FROM engine_reference_manifests WHERE engine=?",
                        (identifier,),
                    ).fetchone()
                    document: dict[str, object] = (
                        _manifest(row[0])
                        if row
                        else {"engine": identifier, "revision": 0, "digests": []}
                    )
                    document["revision"] = cast(int, document["revision"]) + 1
                    document["complete"] = False
                    db.execute(
                        "INSERT INTO engine_reference_manifests VALUES (?,?) "
                        "ON CONFLICT(engine) DO UPDATE SET document=excluded.document",
                        (identifier, json.dumps(document, allow_nan=False)),
                    )
                    db.execute(
                        "INSERT INTO changes(kind,id,revision) VALUES ('engine-references',?,?)",
                        (identifier, document["revision"]),
                    )
                self._engine, self._engine_lock = identifier, lock
                return {"manifest": document}
            except BaseException:
                lock.release()
                raise
        if operation not in ("references.engine.get", "references.engine.patch"):
            raise LibraryError("operation", f"unknown library operation: {operation}")
        if self._engine is None:
            raise LibraryError("invalid", "attach this helper to its engine first")
        allowed: set[str] = set() if operation.endswith(".get") else {"revision", "patch"}
        if set(arguments) - allowed:
            raise LibraryError("invalid", "unknown engine reference field")
        if operation.endswith(".get"):
            row = db.execute(
                "SELECT document FROM engine_reference_manifests WHERE engine=?", (self._engine,)
            ).fetchone()
            assert row is not None
            return {"manifest": _manifest(row[0])}
        with self.writing(), db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT document FROM engine_reference_manifests WHERE engine=?", (self._engine,)
            ).fetchone()
            assert row is not None
            document = _manifest(row[0])
            revision = arguments.get("revision")
            if type(revision) is not int or revision != document["revision"]:
                raise LibraryError("stale", "engine references changed; read them again")
            patch = arguments.get("patch")
            if not isinstance(patch, dict) or set(cast("dict[str, object]", patch)) - {
                "digests",
                "complete",
            }:
                raise LibraryError("invalid", "unknown engine reference patch field")
            fields = dict(cast("dict[str, object]", patch))
            if "digests" in fields:
                digests = fields["digests"]
                if not isinstance(digests, list):
                    raise LibraryError("invalid", "engine reference digests must be a list")
                fields["digests"] = sorted(
                    {_digest(value) for value in cast("list[object]", digests)}
                )
            if "complete" in fields and type(fields["complete"]) is not bool:
                raise LibraryError("invalid", "engine reference completeness must be a boolean")
            document.update(fields)
            document["revision"] = cast(int, document["revision"]) + 1
            db.execute(
                "UPDATE engine_reference_manifests SET document=? WHERE engine=?",
                (json.dumps(document, allow_nan=False), self._engine),
            )
            db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('engine-references',?,?)",
                (self._engine, document["revision"]),
            )
            return {"manifest": document}

    def dispatch(
        self, db: sqlite3.Connection, operation: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        if operation.startswith("references.engine."):
            return self.engine(db, operation, arguments)
        if operation == "references.acquire":
            if set(arguments) - {"limit"}:
                raise LibraryError("invalid", "unknown asset reference field")
            if any((self._root / name).exists() for name in ("history.sqlite", "library.sqlite")):
                raise LibraryError("migration", "Finish the complete library migration first")
            if self._lease is not None:
                raise LibraryError("busy", "this helper already holds asset references")
            try:
                self._lock.acquire(timeout=0)
            except Timeout as error:
                raise LibraryError("busy", "asset references are already reserved") from error
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
                for (raw,) in db.execute(
                    "SELECT document FROM engine_reference_manifests"
                ).fetchall():
                    manifest = _manifest(raw)
                    if manifest.get("complete") is not True or not isinstance(
                        digests := manifest.get("digests"), list
                    ):
                        raise LibraryError("references", "engine asset references are incomplete")
                    roots.update(_digest(value) for value in cast("list[object]", digests))
                for (raw,) in db.execute("SELECT document FROM saved_records WHERE deleted=0"):
                    loaded = json.loads(raw)
                    if not isinstance(loaded, dict):
                        raise LibraryError("references", "saved record is invalid")
                    roots.add(_digest(cast("dict[str, object]", loaded).get("digest")))
                for (raw,) in db.execute("SELECT asset_digests FROM derivation_manifests"):
                    loaded = json.loads(raw) if raw is not None else None
                    if not isinstance(loaded, list):
                        raise LibraryError("references", "derivation asset references are unknown")
                    roots.update(_digest(value) for value in cast("list[object]", loaded))
                pending = list(roots)
                while pending:
                    row = db.execute(
                        "SELECT document FROM dependency_manifests WHERE digest=?",
                        (pending.pop(),),
                    ).fetchone()
                    if row is None:
                        continue
                    loaded = json.loads(row[0])
                    if not isinstance(loaded, list):
                        raise LibraryError("references", "asset dependencies are invalid")
                    for raw_dependency in cast("list[object]", loaded):
                        if not isinstance(raw_dependency, dict):
                            raise LibraryError("references", "asset dependency is invalid")
                        dependency = cast("dict[str, object]", raw_dependency)
                        child = _digest(dependency.get("digest"))
                        if child not in roots:
                            roots.add(child)
                            pending.append(child)
                self._snapshot = tuple(sorted(roots))
                self._lease = uuid.uuid4().hex
                return self._page(arguments)
            except BaseException:
                self._lease = None
                self._snapshot = ()
                self._lock.release()
                raise
        if operation not in ("references.page", "references.release"):
            raise LibraryError("operation", f"unknown library operation: {operation}")
        allowed = {"lease", "cursor", "limit"} if operation.endswith(".page") else {"lease"}
        if set(arguments) - allowed:
            raise LibraryError("invalid", "unknown asset reference field")
        if self._lease is None or arguments.get("lease") != self._lease:
            raise LibraryError("invalid", "asset reference reservation is no longer active")
        if operation.endswith(".release"):
            self._release()
            return {}
        return self._page(arguments)

    def _page(self, arguments: dict[str, object]) -> dict[str, object]:
        limit = arguments.get("limit", 200)
        if type(limit) is not int or not 1 <= limit <= 200:
            raise LibraryError("invalid", "asset reference page size must be from 1 to 200")
        cursor = arguments.get("cursor")
        start = 0
        if cursor is not None:
            cursor = _digest(cursor)
            start = bisect.bisect_right(self._snapshot, cursor)
            if start == 0 or self._snapshot[start - 1] != cursor:
                raise LibraryError("invalid", "asset reference cursor is not in this reservation")
        selected = self._snapshot[start : start + limit]
        return {
            "lease": self._lease,
            "digests": list(selected),
            "cursor": selected[-1] if start + limit < len(self._snapshot) else None,
        }
