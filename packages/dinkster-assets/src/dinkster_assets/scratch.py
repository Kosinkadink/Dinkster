"""OS-backed use leases for managed produced scratch directories."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from .identity import AssetError


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
            with _lock(folder / "gate.lock").acquire(timeout=0):
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
