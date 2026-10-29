"""OS-backed use leases for managed produced scratch directories."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Generator, Iterable
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from .identity import AssetError, require_digest


class ProducedScratchState:
    """Journal a scratch root's job owner and publications offered for handoff.

    Offered bytes remain recoverable until durable adoption is established.
    This journal does not prove reader liveness or authorize deletion.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.absolute()
        self._path = self._root / ".produced-scratch.sqlite"
        if self._root.is_symlink() or self._path.is_symlink():
            raise AssetError("produced scratch state must not be a symlink")

    def bind(self, job: str, attempt: int) -> None:
        if not job or type(attempt) is not int or attempt < 1:
            raise AssetError("produced scratch requires a job and positive attempt")
        try:
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                tables = {
                    row[0]
                    for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if not tables:
                    metadata = {self._path.name, self._path.name + "-journal"}
                    if any(path.name not in metadata for path in self._root.iterdir()):
                        raise AssetError("produced scratch contains untracked data")
                    db.execute(
                        "CREATE TABLE owner (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                        "job TEXT NOT NULL, attempt INTEGER NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE publications (digest TEXT PRIMARY KEY, "
                        "size INTEGER NOT NULL, "
                        "offered INTEGER NOT NULL DEFAULT 0 CHECK (offered IN (0, 1)))"
                    )
                    db.execute("INSERT INTO owner VALUES (1, ?, ?)", (job, attempt))
                elif tables != {"owner", "publications"}:
                    raise AssetError("produced scratch ownership state is incomplete")
                if db.execute("SELECT job, attempt FROM owner").fetchall() != [(job, attempt)]:
                    raise AssetError("produced scratch job ownership is unavailable or changed")
        except sqlite3.Error as exc:
            raise AssetError("produced scratch state is unavailable") from exc

    def published(self, digest: str, size: int) -> None:
        digest = require_digest(digest)
        if type(size) is not int or size < 0:
            raise AssetError("produced scratch size must be a nonnegative integer")
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True)) as db, db:
                db.execute(
                    "INSERT OR IGNORE INTO publications (digest, size) VALUES (?, ?)",
                    (digest, size),
                )
                if db.execute(
                    "SELECT size FROM publications WHERE digest = ?", (digest,)
                ).fetchone() != (size,):
                    raise AssetError("produced scratch publication sizes disagree")
        except sqlite3.Error as exc:
            raise AssetError("produced scratch state is unavailable") from exc

    def offer(self, digests: Iterable[str]) -> None:
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True)) as db, db:
                for digest in digests:
                    db.execute(
                        "UPDATE publications SET offered = 1 WHERE digest = ?",
                        (require_digest(digest),),
                    )
        except sqlite3.Error as exc:
            raise AssetError("produced scratch state is unavailable") from exc

    def publications(self) -> tuple[tuple[str, int, bool], ...]:
        """Read existing publication state without creating or repairing a journal."""
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute(
                    "SELECT digest, size, offered FROM publications ORDER BY digest"
                ).fetchall()
        except sqlite3.Error as exc:
            raise AssetError("produced scratch state is unavailable") from exc
        result: list[tuple[str, int, bool]] = []
        for digest, size, offered in rows:
            if not isinstance(digest, str):
                raise AssetError("produced scratch digest is invalid")
            require_digest(digest)
            if (
                type(size) is not int
                or size < 0
                or type(offered) is not int
                or offered not in (0, 1)
            ):
                raise AssetError("produced scratch publication state is invalid")
            result.append((digest, size, bool(offered)))
        return tuple(result)


def _lease_directory(root: Path, *, create: bool = True) -> Path:
    if root.is_symlink():
        raise AssetError("produced scratch root must not be a symlink")
    # The gate must survive removal of the claimed data directory.
    parent = root.parent / ".leases"
    folder = parent / root.name
    if parent.is_symlink() or folder.is_symlink():
        raise AssetError("produced scratch leases must not be a symlink")
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    elif not folder.is_dir() or not (folder / "gate.lock").is_file():
        raise AssetError("produced scratch ownership state is unavailable")
    return folder


def _lock(path: Path) -> FileLock:
    # Soft locks cannot prove a crashed writer is gone. Retaining lock files
    # also prevents two contenders from locking different replacement inodes.
    return FileLock(
        path,
        thread_local=False,
        fallback_to_soft=False,
        preserve_lock_file=True,
    )


class ProducedScratchLease:
    """Keep a scratch directory busy until its owner explicitly releases it."""

    def __init__(self, root: Path) -> None:
        folder = _lease_directory(root)
        self._lock = _lock(folder / f"owner-{uuid.uuid4().hex}.lock")
        try:
            # Other producers briefly hold this gate while registering too.
            # Only reconciliation attempts must be strictly nonblocking.
            with _lock(folder / "gate.lock").acquire(timeout=5):
                self._lock.acquire(timeout=0)
        except Timeout:
            raise AssetError("produced scratch is busy with reconciliation") from None

    def close(self) -> None:
        self._lock.release()


@contextmanager
def claim_inactive_produced_scratch(root: Path) -> Generator[bool]:
    """Hold off new users and yield whether every existing OS lease is free.

    A successful claim is a writer-liveness proof, not permission to discard
    publications: adoption, reference and reader protection still apply.
    """
    folder = _lease_directory(root, create=False)
    with ExitStack() as held:
        try:
            held.enter_context(_lock(folder / "gate.lock").acquire(timeout=0))
            for path in sorted(folder.iterdir()):
                if path.name == "gate.lock":
                    continue
                if path.is_symlink() or not path.is_file():
                    raise AssetError("produced scratch has unsafe lease state")
                owner = path.name.removeprefix("owner-").removesuffix(".lock")
                if (
                    path.name != f"owner-{owner}.lock"
                    or len(owner) != 32
                    or any(char not in "0123456789abcdef" for char in owner)
                ):
                    raise AssetError("produced scratch has unknown lease state")
                held.enter_context(_lock(path).acquire(timeout=0))
        except Timeout:
            yield False
        else:
            yield True
