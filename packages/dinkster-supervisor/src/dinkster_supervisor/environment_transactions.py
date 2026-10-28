"""Journaled generation activation and deterministic interruption recovery."""

from __future__ import annotations

import shutil
import socket
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from .environment_runtime import (
    engine_assignments,
    preserve_stopped_engine,
    require_idle,
    start_engine,
    stop_engine,
)
from .environment_state import EnvironmentError, Generation, Journal, read_current, set_current


def begin_activation(root: Path, target: int, operation: str) -> Path:
    """Persist prior bindings before building; caller holds the project writer lock."""
    if operation not in {"update", "rollback", "repair"}:
        raise EnvironmentError("activation requires update, rollback or repair")
    previous = read_current(root)
    if previous is None or previous == target:
        raise EnvironmentError("activation requires distinct previous and target generations")
    if any(Journal.read(path).incomplete for path in (root / "journal").glob("*.json")):
        raise EnvironmentError("incomplete environment operation requires recovery first")
    target_root = root / "generations" / str(target)
    if operation == "rollback":
        if Generation.read(target_root / "generation.json").number != target:
            raise EnvironmentError("generation directory and recorded number disagree")
    elif target_root.exists():
        raise EnvironmentError("new generation target already exists")
    require_idle(root)
    bindings = engine_assignments(root, previous)
    sockets: list[socket.socket] = []
    try:
        for binding in bindings:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
            binding["targetPort"] = sock.getsockname()[1]
        path = root / "journal" / f"{time.time_ns()}.json"
        Journal(operation, previous, target, "planned", bindings=tuple(bindings)).write(path)
        Journal.read(path)
        return path
    finally:
        for sock in sockets:
            sock.close()


def activate(root: Path, path: Path, *, revalidate: Callable[[], None]) -> None:
    """Health-check every candidate before switching; preserve the old tree on failure."""
    journal = Journal.read(path)
    if not journal.incomplete or journal.operation not in {"update", "rollback", "repair"}:
        raise EnvironmentError("journal does not describe a pending generation activation")
    if read_current(root) != journal.previous:
        raise EnvironmentError("current changed after activation was planned")
    try:
        target_root = root / "generations" / str(journal.target)
        generation = Generation.read(target_root / "generation.json")
        if generation.number != journal.target or generation.drift(target_root):
            raise EnvironmentError("target generation is incomplete or has drift")
        journal = replace(journal, stage="starting")
        journal.write(path)
        for binding in journal.bindings:
            if binding["running"]:
                start_engine(
                    root,
                    journal.target,
                    cast("str", binding["name"]),
                    cast("int", binding["targetPort"]),
                    tuple(cast("list[str]", binding["gpus"])),
                )
            else:
                preserve_stopped_engine(
                    root,
                    cast("str", binding["name"]),
                    cast("int", journal.previous),
                    journal.target,
                    cast("int", binding["targetPort"]),
                )
        revalidate()
        require_idle(root)
        journal = replace(journal, stage="switching")
        journal.write(path)
        set_current(root, journal.target)
        for binding in journal.bindings:
            if binding["running"]:
                stop_engine(root, cast("str", binding["name"]), generation=journal.previous)
        replace(journal, stage="complete").write(path)
    except Exception as exc:
        replace(Journal.read(path), error=str(exc)).write(path)
        recover_activation(root, path)
        raise


def recover_activation(root: Path, path: Path) -> None:
    """Restore previous current and processes before completing another mutation."""
    journal = Journal.read(path)
    if not journal.incomplete:
        return
    if journal.operation not in {"update", "rollback", "repair"} or journal.previous is None:
        raise EnvironmentError("journal operation does not support activation recovery")
    if read_current(root) not in {journal.previous, journal.target}:
        raise EnvironmentError("current does not match interrupted activation")
    for binding in journal.bindings:
        stop_engine(root, cast("str", binding["name"]), generation=journal.target)
    set_current(root, journal.previous)
    for binding in journal.bindings:
        if binding["running"]:
            start_engine(
                root,
                journal.previous,
                cast("str", binding["name"]),
                cast("int", binding["port"]),
                tuple(cast("list[str]", binding["gpus"])),
            )
    target_root = root / "generations" / str(journal.target)
    if journal.operation != "rollback" and target_root.exists():
        shutil.rmtree(target_root)
    replace(journal, stage="rolled-back").write(path)
