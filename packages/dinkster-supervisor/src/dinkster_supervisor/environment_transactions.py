"""Journaled generation activation and deterministic interruption recovery."""

from __future__ import annotations

import shutil
import socket
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from .environment_history import prepare_entry, publish_entry, record_failed_restore
from .environment_runtime import (
    check_library_compatibility,
    engine_assignments,
    preserve_stopped_engine,
    require_idle,
    start_engine,
    stop_engine,
    validate_assignment,
)
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    atomic_write,
    durable_unlink,
    journal_paths,
    read_current,
    read_document,
    set_current,
)
from .environment_stores import restore_setup_registration


def next_generation(root: Path) -> int:
    """Never reuse the number of a successfully published tree after collection."""
    numbers = [int(path.name) for path in (root / "builds").glob("*") if path.name.isdecimal()]
    for path in journal_paths(root):
        journal = Journal.read(path)
        if journal.stage == "complete":
            numbers.append(journal.target)
    return max(numbers, default=0) + 1


def previous_activation(root: Path, current: int) -> Journal | None:
    """Find the actual successful predecessor, ignoring collection and failed attempts."""
    for path in sorted(journal_paths(root), reverse=True):
        journal = Journal.read(path)
        if (
            journal.operation in {"update", "rollback", "repair", "restore"}
            and journal.stage == "complete"
            and journal.target == current
            and journal.previous is not None
        ):
            return journal
    return None


def activation_status(root: Path) -> dict[str, object]:
    """Describe the last activation without recovering or changing software."""
    for path in sorted(journal_paths(root), reverse=True):
        journal = Journal.read(path)
        if journal.operation not in {"setup", "update", "rollback", "repair", "restore"}:
            continue
        if journal.stage == "complete":
            state, action = "applied", "Inspect the selected snapshot before starting new work."
        elif journal.stage == "rolled-back":
            state = (
                "cancelled"
                if journal.error in {"pack preparation cancelled", "setup cancelled"}
                else "failed"
            )
            action = "Current selection was restored; request and approve a new repair plan."
        else:
            state, action = "interrupted", "Run an explicit environment repair before retrying."
        return {
            "state": state,
            "operation": journal.operation,
            "previous": journal.previous,
            "target": journal.target,
            "current": read_current(root),
            "error": journal.error,
            "action": action,
        }
    return {"state": "not-attempted", "action": "Preview a repair before approving any change."}


def begin_setup(
    root: Path,
    target: int,
    port: int,
    gpus: tuple[str, ...] = (),
    *,
    name: str = "main",
    registration: dict[str, object] | None = None,
    start: bool = True,
) -> Path:
    """Record the initial binding before construction; caller holds the writer lock."""
    if read_current(root) is not None:
        raise EnvironmentError("environment already has an active snapshot")
    if any(Journal.read(path).incomplete for path in journal_paths(root)):
        raise EnvironmentError("incomplete environment operation requires recovery first")
    if (root / "builds" / str(target)).exists():
        raise EnvironmentError("new snapshot target already exists")
    validate_assignment(target, name, port, gpus)
    require_idle(root)
    binding: dict[str, object] = {
        "name": name,
        "port": port,
        "targetPort": port,
        "gpus": list(gpus),
        "running": start,
    }
    path = root / "journal" / f"{time.time_ns()}.json"
    Journal(
        "setup", None, target, "planned", bindings=(binding,), setup_registration=registration
    ).write(path)
    return path


def begin_activation(
    root: Path,
    target: int,
    operation: str,
    *,
    project_settings: dict[str, str] | None = None,
    restored_from: int | None = None,
) -> Path:
    """Persist prior bindings before building; caller holds the project writer lock."""
    if operation not in {"update", "rollback", "repair", "restore"}:
        raise EnvironmentError("activation requires update, restore previous, repair or restore")
    previous = read_current(root)
    if previous is None or previous == target:
        raise EnvironmentError("activation requires distinct previous and target snapshots")
    if any(Journal.read(path).incomplete for path in journal_paths(root)):
        raise EnvironmentError("incomplete environment operation requires recovery first")
    target_root = root / "builds" / str(target)
    if target_root.exists():
        raise EnvironmentError("new snapshot target already exists")
    require_idle(root)
    bindings = engine_assignments(root, previous)
    stopped_ports = {binding["port"] for binding in bindings if not binding["running"]}
    sockets: list[socket.socket] = []
    try:
        for binding in bindings:
            if not binding["running"]:
                binding["targetPort"] = binding["port"]
                continue
            while True:
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", 0))
                if sock.getsockname()[1] not in stopped_ports:
                    break
            binding["targetPort"] = sock.getsockname()[1]
        path = root / "journal" / f"{time.time_ns()}.json"
        Journal(
            operation,
            previous,
            target,
            "planned",
            bindings=tuple(bindings),
            project_settings=project_settings,
            restored_from=restored_from,
        ).write(path)
        Journal.read(path)
        return path
    finally:
        for sock in sockets:
            sock.close()


def activate(root: Path, path: Path, *, revalidate: Callable[[], None]) -> None:
    """Check helper-era candidates beside the old engine; recover on failure."""
    journal = Journal.read(path)
    if not journal.incomplete or journal.operation not in {
        "setup",
        "update",
        "rollback",
        "repair",
        "restore",
    }:
        raise EnvironmentError("journal does not describe a pending snapshot activation")
    if read_current(root) != journal.previous:
        raise EnvironmentError("current changed after activation was planned")
    try:
        target_root = root / "builds" / str(journal.target)
        generation = Generation.read(target_root / "build.json")
        if generation.number != journal.target or generation.drift(target_root):
            raise EnvironmentError("target snapshot is incomplete or has drift")
        from .environment_tools import library_command

        manifest = target_root / "engine-manifest.json"
        target_manifest = read_document(manifest) if manifest.is_file() else {}
        library_command(root, target_manifest)
        previous_manifest = root / "builds" / str(journal.previous) / "engine-manifest.json"
        target_capabilities = target_manifest.get("capabilities", [])
        previous_capabilities: object = (
            read_document(previous_manifest).get("capabilities", [])
            if previous_manifest.is_file()
            else []
        )
        # Earlier helper engines still take the library's exclusive lifetime lock.
        overlap = (
            isinstance(target_capabilities, list)
            and "library-helper/1" in target_capabilities
            and "library-overlap/1" in target_capabilities
            and isinstance(previous_capabilities, list)
            and "library-helper/1" in previous_capabilities
            and "library-overlap/1" in previous_capabilities
        )
        if journal.history_entry is None:
            journal = replace(
                journal,
                history_entry=prepare_entry(
                    root, target_root, operation=path.stem, reason=journal.operation
                ),
            )
        check_library_compatibility(root, journal.target)
        require_idle(root)
        journal = replace(journal, stage="starting")
        journal.write(path)
        for binding in journal.bindings:
            if binding["running"] and journal.previous is not None and not overlap:
                stop_engine(root, cast("str", binding["name"]), generation=journal.previous)
        for binding in journal.bindings:
            if binding["running"]:
                start_engine(
                    root,
                    journal.target,
                    cast("str", binding["name"]),
                    cast("int", binding["targetPort"]),
                    tuple(cast("list[str]", binding["gpus"])),
                    activation_journal=path if journal.previous is not None else None,
                    sandbox_packs=binding.get("sandboxPacks") is True,
                )
                journal = Journal.read(path)
            elif journal.previous is not None:
                preserve_stopped_engine(
                    root,
                    cast("str", binding["name"]),
                    journal.previous,
                    journal.target,
                    cast("int", binding["targetPort"]),
                )
        revalidate()
        require_idle(root)
        if overlap:
            for binding in journal.bindings:
                if binding["running"] and journal.previous is not None:
                    stop_engine(root, cast("str", binding["name"]), generation=journal.previous)
        journal = replace(journal, stage="switching")
        journal.write(path)
        if journal.project_settings is not None:
            atomic_write(root / "environment.json", journal.project_settings["target"])
        set_current(root, journal.target)
        replace(journal, stage="complete").write(path)
    except Exception as exc:
        replace(Journal.read(path), error=str(exc)).write(path)
        recover_activation(root, path)
        raise
    assert journal.history_entry is not None
    publish_entry(root, journal.history_entry)


def recover_activation(root: Path, path: Path) -> None:
    """Restore state; caller holds project and any recorded registry/store locks."""
    journal = Journal.read(path)
    if not journal.incomplete:
        if journal.stage == "complete" and journal.history_entry is not None:
            publish_entry(root, journal.history_entry)
        return
    if journal.operation not in {"setup", "update", "rollback", "repair", "restore"}:
        raise EnvironmentError("journal operation does not support activation recovery")
    if (journal.previous is None) != (journal.operation == "setup"):
        raise EnvironmentError("journal previous snapshot does not match its operation")
    if read_current(root) not in {journal.previous, journal.target}:
        raise EnvironmentError("current does not match interrupted activation")
    if journal.project_settings is not None and journal.stage == "switching":
        actual = (root / "environment.json").read_bytes().decode("utf-8")
        if actual not in journal.project_settings.values():
            raise EnvironmentError("environment settings changed during interrupted selection")
    for binding in journal.bindings:
        stop_engine(root, cast("str", binding["name"]), generation=journal.target)
    if journal.project_settings is not None and journal.stage == "switching":
        atomic_write(root / "environment.json", journal.project_settings["previous"])
    if journal.previous is None:
        durable_unlink(root / "current")
    else:
        set_current(root, journal.previous)
    for binding in journal.bindings:
        if binding["running"] and journal.previous is not None:
            start_engine(
                root,
                journal.previous,
                cast("str", binding["name"]),
                cast("int", binding["port"]),
                tuple(cast("list[str]", binding["gpus"])),
                sandbox_packs=binding.get("sandboxPacks") is True,
            )
    target_root = root / "builds" / str(journal.target)
    if target_root.exists() and (
        journal.operation != "rollback" or journal.restored_from is not None
    ):
        shutil.rmtree(target_root)
    if journal.setup_registration is not None:
        restore_setup_registration(root, journal.setup_registration)
    if journal.restored_from is not None:
        record_failed_restore(
            root,
            path.stem,
            journal.restored_from,
            journal.error or "restore interrupted before successful activation",
        )
    replace(journal, stage="rolled-back").write(path)
