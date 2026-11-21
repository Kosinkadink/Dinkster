"""Engine-process ownership of a library, including unchanged older engines."""

from __future__ import annotations

import errno
import json
import os
import re
import runpy
import sys
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

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


@contextmanager
def engine_state(root: Path, library: Path, snapshot: int) -> Generator[Path]:
    """Select working state using the tools' code before importing any engine."""
    library = library.resolve()
    if not root.is_absolute() or ".." in root.parts or type(snapshot) is not int or snapshot < 1:
        raise ValueError("engine state needs an absolute directory and a positive snapshot")
    if any(path.is_symlink() or path.is_junction() for path in (root, *root.parents)):
        raise ValueError("engine state must not contain linked directories")
    if root.is_relative_to(library) or library.is_relative_to(root):
        raise ValueError("engine state must not overlap the library")
    with engine_lock(root, "managed engine state"):
        marker = root / "state.json"
        pending = root / ".state.pending"
        if marker.is_symlink() or marker.is_junction():
            raise ValueError("engine state selection must not be linked")
        selected = "working-" + uuid4().hex
        record: dict[str, object] = {
            "format": "dinkster.engine-state-selection/1",
            "library": str(library),
        }
        if marker.exists():
            previous: object = json.loads(marker.read_text("utf-8"))
            if not isinstance(previous, dict):
                raise ValueError("invalid engine state selection")
            record = cast("dict[str, object]", previous)
            stored_library = record.get("library")
            last_snapshot = record.get("lastSnapshot")
            directory = record.get("directory")
            if (
                record.get("format") != "dinkster.engine-state-selection/1"
                or not isinstance(stored_library, str)
                or not Path(stored_library).is_absolute()
                or type(last_snapshot) is not int
                or last_snapshot < 1
                or not isinstance(directory, str)
                or re.fullmatch(r"working-[a-f0-9]{32}", directory) is None
            ):
                raise ValueError("invalid engine state selection")
            if not Path(stored_library).samefile(library):
                raise ValueError("engine state belongs to another library")
            if snapshot >= last_snapshot:
                selected = directory
        elif any(path.name not in {".engine.lock", pending.name} for path in root.iterdir()):
            raise ValueError("existing engine state has no snapshot selection")
        working = root / selected
        if working.is_symlink() or working.is_junction():
            raise ValueError("engine working state must not be linked")
        if (
            pending.is_symlink()
            or pending.is_junction()
            or (pending.exists() and (not pending.is_file() or pending.stat().st_nlink != 1))
        ):
            raise ValueError("pending engine state selection must be a regular file")
        record.update(lastSnapshot=snapshot, directory=selected)
        # Publish before stores open: a killed first use or downgrade can retry
        # its selected empty directory without touching the prior working files.
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, marker)
        if os.name != "nt":
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        working.mkdir(mode=0o700, exist_ok=True)
        yield working


if __name__ == "__main__":
    # The selected interpreter runs this stdlib-only entry before importing any
    # engine version. A modern engine reuses this exact in-process lock owner.
    sys.modules["dinkster_store.library_lock"] = sys.modules[__name__]
    library, label, *arguments = sys.argv[1:]
    compatibility = runpy.run_path(str(Path(__file__).with_name("library_compatibility.py")))
    try:
        with ExitStack() as stack:
            if library:
                stack.enter_context(engine_lock(Path(library), label))
                compatibility["check_library"](Path(library))
            if arguments[:1] == ["--engine-state"]:
                try:
                    if len(arguments) < 4 or arguments[3] != "--" or not library:
                        raise ValueError(
                            "managed engine state requires a library, directory and snapshot"
                        )
                    if any(
                        value.split("=", 1)[0] == "--engine-state-root" for value in arguments[4:]
                    ):
                        raise ValueError("managed engine state cannot be overridden")
                    working = stack.enter_context(
                        engine_state(Path(arguments[1]), Path(library), int(arguments[2]))
                    )
                    arguments = [*arguments[4:], "--engine-state-root", str(working)]
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"Cannot select engine state: {exc}") from exc
            sys.argv = ["dinkster-serve", *arguments]
            runpy.run_module("dinkster.serve", run_name="__main__", alter_sys=True)
    except (LibraryInUse, compatibility["LibraryCompatibilityError"]) as exc:
        raise SystemExit(str(exc)) from exc
