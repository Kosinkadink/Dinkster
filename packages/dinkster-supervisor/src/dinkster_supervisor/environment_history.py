"""Software-only snapshots and durable, append-only installation history."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

from .environment_state import (
    EnvironmentError,
    Generation,
    content_digest,
    generation_number,
    read_document,
)


def capture_software(generation: Path) -> dict[str, object]:
    """Read only sealed software metadata, never project libraries or mounts."""
    record = Generation.read(generation / "generation.json")
    expected = dict(record.files)
    names = {"constraints.txt"}
    names.update(
        name
        for name in expected
        if name in {"base-manifest.json", "engine-manifest.json", "dinkster.lock"}
        or (
            len(Path(name).parts) == 3
            and Path(name).parts[0] == "packs"
            and Path(name).parts[2] == "pack-lock.json"
        )
    )
    metadata: dict[str, str] = {}
    for name in sorted(names):
        path = generation / name
        if (
            any(parent.is_symlink() for parent in (generation, *path.parents[:2]))
            or path.is_symlink()
            or not path.is_file()
            or content_digest(path) != expected[name]
        ):
            raise EnvironmentError(f"snapshot requires intact software metadata: {name}")
        metadata[name] = path.read_bytes().decode("utf-8")
    return {
        "format": "dinkster.install-snapshot/1",
        "engine": {"version": record.engine_version, "release": record.engine_release},
        "base": {"id": record.base, "variant": record.variant},
        "constraints": record.constraints,
        "workerProtocol": record.worker_protocol,
        "source": record.source,
        "inProcessPacks": list(record.in_process_packs),
        "tools": {"version": version("dinkster-supervisor")},
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "metadata": metadata,
    }


def _changes(previous: object, current: object, field: str = "") -> list[dict[str, object]]:
    if isinstance(previous, dict) and isinstance(current, dict):
        before, after = cast("dict[str, object]", previous), cast("dict[str, object]", current)
        return [
            change
            for name in sorted(before.keys() | after.keys())
            for change in _changes(
                before.get(name), after.get(name), f"{field}.{name}" if field else name
            )
        ]
    return [] if previous == current else [{"field": field, "before": previous, "after": current}]


def history(root: Path) -> list[dict[str, object]]:
    """Return successful entries without consulting materialized generations."""
    directory = root / "history"
    if directory.is_symlink():
        raise EnvironmentError("history must not be a symlink")
    entries: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink():
            raise EnvironmentError("history entries must not be symlinks")
        entry = read_document(path)
        number = generation_number(entry.get("id"))
        if (
            path.name != f"{number:020d}.json"
            or number != len(entries) + 1
            or entry.get("format") != "dinkster.install-history/1"
            or not isinstance(entry.get("snapshot"), dict)
        ):
            raise EnvironmentError("invalid installation history entry")
        entries.append(entry)
    return entries


def prepare_entry(
    root: Path,
    generation: Path,
    *,
    operation: str,
    reason: str,
    restored_from: int | None = None,
    substitutions: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Prepare under the project writer lock, before attempting activation."""
    entries = history(root)
    snapshot = capture_software(generation)
    previous: dict[str, object] = (
        cast("dict[str, object]", entries[-1]["snapshot"]) if entries else {}
    )
    return {
        "format": "dinkster.install-history/1",
        "id": len(entries) + 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "reason": reason,
        "generation": int(generation.name),
        "snapshot": snapshot,
        "diff": _changes(previous, snapshot),
        "restoredFrom": restored_from,
        "exact": not substitutions,
        "substitutions": list(substitutions),
    }


def publish_entry(root: Path, entry: dict[str, object]) -> None:
    """Publish once; an activation retry cannot replace an existing entry."""
    number = generation_number(entry.get("id"))
    directory = root / "history"
    if directory.is_symlink():
        raise EnvironmentError("history must not be a symlink")
    directory.mkdir(exist_ok=True)
    destination = directory / f"{number:020d}.json"
    if destination.is_symlink():
        raise EnvironmentError("history entries must not be symlinks")
    if destination.exists():
        if read_document(destination) != entry:
            raise EnvironmentError("cannot replace an installation history entry")
        return
    entries = history(root)
    if number != len(entries) + 1:
        raise EnvironmentError("history entries must append consecutively")
    fd, temporary = tempfile.mkstemp(prefix=".entry-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        # Linking publishes fully flushed bytes without replace semantics on either OS.
        os.link(temporary, destination)
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)
