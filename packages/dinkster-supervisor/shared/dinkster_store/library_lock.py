"""Engine-process ownership of a library, including unchanged older engines."""

from __future__ import annotations

import errno
import json
import os
import runpy
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path

_held: set[tuple[int, Path]] = set()


class LibraryInUse(RuntimeError):
    """Another engine owns the library's lifetime lock."""


@contextmanager
def engine_lock(root: Path, name: str) -> Generator[None]:
    """Refuse competing engines; nested entry in the owning process is harmless."""
    root = root.resolve()
    owner = (os.getpid(), root)
    if owner in _held:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".engine.lock").open("a+b", buffering=0) as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
        started = time.monotonic()
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                handle.seek(1)
                holder = "another engine"
                try:
                    record = json.loads(handle.read(8192))
                    holder = str(record["name"])
                except (OSError, ValueError, KeyError, TypeError):
                    if time.monotonic() - started < 0.1:
                        time.sleep(0.01)
                        continue
                raise LibraryInUse(
                    f"Library {root} is already in use by {holder}. "
                    "Stop that engine before starting another; "
                    "all engine and project records are kept."
                ) from exc
        try:
            handle.seek(1)
            handle.truncate(1)
            handle.write(json.dumps({"name": name, "pid": os.getpid()}).encode("utf-8"))
            _held.add(owner)
            yield
        finally:
            _held.discard(owner)
            handle.truncate(1)
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    # The selected interpreter runs this stdlib-only entry before importing any
    # engine version. A modern engine reuses this exact in-process lock owner.
    sys.modules["dinkster_store.library_lock"] = sys.modules[__name__]
    library, label, *arguments = sys.argv[1:]
    sys.argv = ["dinkster-serve", *arguments]
    compatibility = runpy.run_path(str(Path(__file__).with_name("library_compatibility.py")))
    try:
        with engine_lock(Path(library), label) if library else nullcontext():
            if library:
                compatibility["check_library"](Path(library))
            runpy.run_module("dinkster.serve", run_name="__main__", alter_sys=True)
    except (LibraryInUse, compatibility["LibraryCompatibilityError"]) as exc:
        raise SystemExit(str(exc)) from exc
