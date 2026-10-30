"""OS-backed use leases for managed produced scratch directories."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
from pathlib import Path

from filelock import FileLock, Timeout

from .identity import AssetError, digest_bytes, is_digest, require_digest


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

    def forget_publications(self, publications: tuple[tuple[str, int, bool], ...]) -> None:
        """Retire exact handoff rows only after owned scratch bytes have been removed."""
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True)) as db, db:
                for digest, size, offered in publications:
                    changed = db.execute(
                        "DELETE FROM publications WHERE digest = ? AND size = ? AND offered = ?",
                        (digest, size, int(offered)),
                    ).rowcount
                    if changed != 1:
                        raise AssetError("produced scratch publication changed during removal")
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

    def owner(self) -> tuple[str, int]:
        """Read an existing owner without binding or repairing publication state."""
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute("SELECT singleton, job, attempt FROM owner").fetchall()
        except sqlite3.Error as exc:
            raise AssetError("produced scratch owner is unavailable") from exc
        if len(rows) != 1:
            raise AssetError("produced scratch owner is invalid")
        singleton, job, attempt = rows[0]
        if (
            singleton != 1
            or not isinstance(job, str)
            or not job
            or type(attempt) is not int
            or attempt < 1
        ):
            raise AssetError("produced scratch owner is invalid")
        return job, attempt


def produced_job_scratch_root(scratch: Path, job: str, attempt: int) -> Path:
    """Use one canonical job/attempt mapping for publication and reconciliation."""
    if not job or type(attempt) is not int or attempt < 1:
        raise AssetError("produced scratch requires a job and positive attempt")
    key = digest_bytes(json.dumps((job, attempt), ensure_ascii=True).encode("ascii")).split(":", 1)[
        1
    ]
    return scratch / "produced-assets" / "jobs" / key


def _lease_directory(root: Path, *, create: bool = True) -> Path:
    if root.is_symlink():
        raise AssetError("produced scratch root must not be a symlink")
    # The gate must survive removal of the claimed data directory.
    parent = root.parent / ".leases"
    folder = parent / root.name
    if parent.is_symlink() or folder.is_symlink() or (folder / "gate.lock").is_symlink():
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
    """Keep managed directories busy until their owner explicitly releases them."""

    def __init__(self, root: Path, *additional_roots: Path) -> None:
        self._locks: list[FileLock] = []
        try:
            for directory in (root, *additional_roots):
                folder = _lease_directory(directory)
                lock = _lock(folder / f"owner-{uuid.uuid4().hex}.lock")
                self._locks.append(lock)
                # Other producers briefly hold this gate while registering too.
                # Only reconciliation attempts must be strictly nonblocking.
                with _lock(folder / "gate.lock").acquire(timeout=5):
                    lock.acquire(timeout=0)
        except BaseException as exc:
            self.close()
            if isinstance(exc, Timeout):
                raise AssetError("produced scratch is busy with reconciliation") from None
            raise

    def close(self) -> None:
        while self._locks:
            self._locks.pop().release()


def hold_produced_vault(root: Path) -> ProducedScratchLease:
    """Retain the host's produced vault through publication and reference handoff."""
    root = root.absolute()
    if root.is_symlink():
        raise AssetError("produced vault ownership must not follow symlinks")
    return ProducedScratchLease(root / ".produced-collection")


@contextmanager
def claim_inactive_produced_vault(root: Path) -> Generator[bool]:
    """Exclude managed readers, publishers and host runs, not reference owners.

    Callers must also hold every reference owner through their final operation;
    missing ownership metadata or inventory cannot certify a quiet vault.
    """
    root = root.absolute()
    inventory = root / ".produced-assets.sqlite"
    if root.is_symlink() or inventory.is_symlink() or not inventory.is_file():
        raise AssetError("produced vault ownership state is unavailable")
    with claim_inactive_produced_scratch(root / ".produced-collection") as inactive:
        yield inactive


def hold_produced_asset_read(path: Path) -> ProducedScratchLease | None:
    """Protect managed scratch or an adopted CAS blob while its descriptor is open.

    Ordinary mounted/imported files acquire no lease and create no metadata.
    Scratch readers retain the whole job root; engine readers retain the vault
    and one digest, in that order.
    Missing publication state cannot authorize collection of either kind.
    """
    path = path.absolute()
    if not is_digest(f"blake3:{path.name}") or path.parent.name != path.name[:2]:
        return None
    root = path.parent.parent
    scratch = root / ".produced-scratch.sqlite"
    inventory = root / ".produced-assets.sqlite"
    if not any(marker.exists() or marker.is_symlink() for marker in (scratch, inventory)):
        return None
    if any(candidate.is_symlink() for candidate in (root, path.parent, path, scratch, inventory)):
        raise AssetError("produced asset reads must not follow symlinks")
    if scratch.exists():
        return ProducedScratchLease(root)
    readers = root / ".produced-readers"
    if readers.is_symlink():
        raise AssetError("produced asset reader ownership must not be a symlink")
    return ProducedScratchLease(root / ".produced-collection", readers / path.name)


_path_owner: ContextVar[Callable[[Path], None] | None] = ContextVar(
    "dinkster_produced_asset_path_owner", default=None
)


@contextmanager
def use_produced_asset_path_owner(retain: Callable[[Path], None]) -> Generator[None]:
    """Bind verified path retention to the invocation's existing lifetime owner."""
    token = _path_owner.set(retain)
    try:
        yield
    finally:
        _path_owner.reset(token)


def retain_produced_asset_path(path: Path) -> None:
    """Extend a managed path's read ownership beyond its verification handle."""
    owner = _path_owner.get()
    if owner is not None:
        owner(path)


def _job_lease_root(scratch: Path, job: str) -> Path:
    if not job:
        raise AssetError("produced scratch requires a nonempty job")
    if scratch.is_symlink() or (scratch / "produced-jobs").is_symlink():
        raise AssetError("produced job ownership root must not be a symlink")
    key = digest_bytes(job.encode("utf-8")).split(":", 1)[1]
    return scratch / "produced-jobs" / key


def hold_produced_job(scratch: Path, job: str) -> ProducedScratchLease:
    """Protect every pack's scratch between invocations until the host ends a job."""
    return ProducedScratchLease(_job_lease_root(scratch, job))


@contextmanager
def claim_inactive_produced_job(scratch: Path, job: str) -> Generator[bool]:
    """Refuse reconciliation across invocation gaps in any live host job.

    Missing host ownership metadata is not evidence of an orphaned job.
    Per-pack writer leases and publication recovery must also be checked.
    """
    with claim_inactive_produced_scratch(_job_lease_root(scratch, job)) as inactive:
        yield inactive


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
