"""Quarantine obsolete generations before committing their disposable-tree deletion."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

from .environment_runtime import engine_assignments
from .environment_state import EnvironmentError, Generation, Journal, durable_rename, read_current
from .environment_transactions import previous_activation


def collection_plan(root: Path) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if (root / "generations").is_symlink():
        raise EnvironmentError("collection refuses a linked generations directory")
    current = read_current(root)
    if current is None:
        raise EnvironmentError("project has no active generation")
    previous = previous_activation(root, current)
    retained = {current}
    if previous is not None and previous.previous is not None:
        retained.add(previous.previous)
    for number in retained:
        if Generation.read(root / "generations" / str(number) / "generation.json").number != number:
            raise EnvironmentError("generation directory and recorded number disagree")
    collected: list[int] = []
    for path in sorted((root / "generations").iterdir()):
        if not path.name.isdecimal() or str(int(path.name)) != path.name or int(path.name) < 1:
            raise EnvironmentError("generation directory must have a positive canonical number")
        if path.is_symlink() or not path.is_dir():
            raise EnvironmentError("collection refuses linked or non-directory generation trees")
        number = int(path.name)
        if number not in retained:
            if any(binding["running"] for binding in engine_assignments(root, number)):
                raise EnvironmentError(f"generation {number} has a live engine; stop it before gc")
            collected.append(number)
    return current, tuple(sorted(retained)), tuple(sorted(collected))


def _quarantine(root: Path, path: Path) -> Path:
    if not path.stem.isdecimal():
        raise EnvironmentError("collection journal must have a numeric timestamp name")
    directory = root / ".garbage" / path.stem
    if (root / ".garbage").is_symlink() or directory.is_symlink():
        raise EnvironmentError("collection refuses linked quarantine directories")
    return directory


def cleanup_collection(root: Path, path: Path) -> None:
    journal = Journal.read(path)
    if journal.operation != "gc" or journal.stage != "complete":
        raise EnvironmentError("only committed collection trash can be deleted")
    directory = _quarantine(root, path)
    for number in journal.collected_generations:
        tree = directory / str(number)
        if tree.is_symlink():
            raise EnvironmentError("collection refuses linked quarantine trees")
        if tree.exists():
            shutil.rmtree(tree)


def recover_collection(root: Path, path: Path) -> None:
    journal = Journal.read(path)
    if journal.operation != "gc" or not journal.incomplete:
        return
    if read_current(root) != journal.target:
        raise EnvironmentError("current changed during generation collection")
    if (root / "generations").is_symlink():
        raise EnvironmentError("collection refuses a linked generations directory")
    directory = _quarantine(root, path)
    for number in journal.collected_generations:
        original, parked = root / "generations" / str(number), directory / str(number)
        if original.is_symlink() or parked.is_symlink():
            raise EnvironmentError("collection recovery refuses linked generation trees")
        if parked.exists():
            if original.exists():
                raise EnvironmentError("collection recovery found both original and parked trees")
            durable_rename(parked, original)
        elif not original.is_dir():
            raise EnvironmentError("collection recovery cannot find an original generation")
    replace(journal, stage="rolled-back").write(path)


def collect_generations(root: Path, approved: tuple[int, tuple[int, ...], tuple[int, ...]]) -> None:
    """Caller holds the project lock and has recovered all pending operations."""
    if collection_plan(root) != approved:
        raise EnvironmentError("generation collection changed after planning; request a new plan")
    for old in sorted((root / "journal").glob("*.json")):
        record = Journal.read(old)
        if record.incomplete:
            raise EnvironmentError("incomplete environment operation requires recovery first")
        if record.operation == "gc" and record.stage == "complete":
            cleanup_collection(root, old)
    current, _, numbers = approved
    if not numbers:
        return
    path = root / "journal" / f"{time.time_ns()}.json"
    journal = Journal("gc", current, current, "planned", collected_generations=numbers)
    journal.write(path)
    directory = _quarantine(root, path)
    try:
        for parent in (directory.parent, directory):
            parent.mkdir(exist_ok=True)
            if os.name != "nt":
                descriptor = os.open(parent.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        replace(journal, stage="building").write(path)
        for number in numbers:
            durable_rename(root / "generations" / str(number), directory / str(number))
        replace(journal, stage="complete").write(path)
    except Exception as exc:
        replace(Journal.read(path), error=str(exc)).write(path)
        recover_collection(root, path)
        raise
    cleanup_collection(root, path)
