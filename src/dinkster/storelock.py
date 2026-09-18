"""Advisory cross-process file locking for content stores.

One writer at a time per store: staging (``acquire``/``apply``) and
collection (``gc``) mutate the same content-addressed directories, and
once a store is shared between install roots those writers live in
different processes. The lock is a plain advisory file lock on a
``.lock`` file at the store's content root - ``flock`` on POSIX,
``msvcrt.locking`` on Windows - so it needs no daemon, dies with its
holder (no stale-lock cleanup), and costs nothing when uncontended.

Readers never take it: activation is one ``os.replace`` of a pointer,
``packs_for_serving`` reads immutable digest-addressed content, and gc
only deletes content no registered root references - so a serving
engine never races a writer on anything it can observe.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["StoreLockTimeout", "hold_lock"]

_POLL_INTERVAL = 0.1


class StoreLockTimeout(Exception):
    """The store lock stayed held past the timeout - another process is
    mid-stage or mid-gc; the caller should say so, not corrupt."""


if os.name == "nt":  # pragma: no cover - exercised on Windows machines
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def hold_lock(path: Path, timeout: float) -> Iterator[None]:
    """Hold the exclusive lock at ``path`` for the with-block.

    Polls non-blocking acquisition until ``timeout`` seconds pass, then
    raises :class:`StoreLockTimeout`. Poll-with-timeout rather than a
    blocking wait so a wedged holder produces a loud, attributable
    failure instead of a silently hung CLI.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise StoreLockTimeout(
                    f"{path}: store lock held by another process for over "
                    f"{timeout:.0f}s (a concurrent install/gc may be in "
                    "progress; retry when it finishes)"
                )
            time.sleep(_POLL_INTERVAL)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
