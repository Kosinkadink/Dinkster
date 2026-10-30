"""Durable launch cancellation and supervisor-owned child cleanup receipts."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .environment_state import EnvironmentError, read_document, write_document, writer_lock


def prepare_process(directory: Path, instance: str) -> None:
    directory.mkdir(parents=True, mode=0o700)
    _write(directory, instance, "pending", 0)


def process_state(directory: Path, instance: str, *, pid: int = 0) -> str:
    document = read_document(directory / "state.json")
    state, recorded_pid = document.get("state"), document.get("pid")
    if (
        document.get("instance") != instance
        or not isinstance(state, str)
        or state not in {"pending", "running", "stopped", "cancelled"}
        or type(recorded_pid) is not int
        or recorded_pid < 0
        or (state in {"running", "stopped"} and recorded_pid == 0)
        or (state in {"pending", "cancelled"} and recorded_pid != 0)
        or (state == "stopped" and pid not in {0, recorded_pid})
    ):
        raise EnvironmentError("invalid supervisor lifecycle record")
    return str(document["state"])


def _write(directory: Path, instance: str, state: str, pid: int) -> None:
    write_document(directory / "state.json", {"instance": instance, "state": state, "pid": pid})


@contextmanager
def process_lifetime(directory: Path, instance: str) -> Generator[bool]:
    """Claim a pending launch; hold its lock until child cleanup and supervisor exit."""
    with writer_lock(directory):
        state = process_state(directory, instance)
        if state == "cancelled":
            yield False
            return
        if state != "pending":
            raise EnvironmentError("supervisor launch was already claimed")
        _write(directory, instance, "running", os.getpid())
        yield True


def complete_process(directory: Path, instance: str) -> None:
    """Called only by the lock-holding supervisor after listener and child cleanup."""
    if process_state(directory, instance) != "running":
        raise EnvironmentError("supervisor lifecycle was not running")
    _write(directory, instance, "stopped", os.getpid())


def cancel_pending_process(directory: Path, instance: str) -> bool:
    """Cancel unclaimed launches; return True while a supervisor owns the guard."""
    lock = writer_lock(directory)
    try:
        lock.__enter__()
    except EnvironmentError:
        return True
    try:
        state = process_state(directory, instance)
        if state == "pending":
            _write(directory, instance, "cancelled", 0)
        return False
    finally:
        lock.__exit__(None, None, None)
