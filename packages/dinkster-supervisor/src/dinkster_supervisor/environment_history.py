"""Software-only snapshots and durable, append-only installation history."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

import packaging

from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    generation_number,
    interpreter,
    read_current,
    read_document,
    relative_file,
)


def is_software_metadata(name: str) -> bool:
    """Allow only portable sealed software inputs, never project settings or data."""
    relative_file(name)
    parts = Path(name).parts
    return name in {
        "constraints.txt",
        "base-manifest.json",
        "engine-manifest.json",
        "artifact-sources.json",
        "dinkster.lock",
    } or (
        len(parts) == 3
        and (
            (parts[0] == "packs" and parts[2] == "pack-lock.json")
            or (parts[0] == "pack-sources" and parts[2] == "dinkster-pack.toml")
        )
    )


def _sealed_software(generation: Path) -> dict[str, object]:
    """Read portable software identities without launching an interpreter."""
    record = Generation.read(generation / "generation.json")
    expected = dict(record.files)
    names = {"constraints.txt"}
    names.update(name for name in expected if is_software_metadata(name))
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
        "metadata": metadata,
    }


def capture_software(generation: Path, *, inspect_wheel_tags: bool = True) -> dict[str, object]:
    """Read sealed software and compatibility observations, never project data."""
    snapshot = _sealed_software(generation)
    metadata = cast("dict[str, str]", snapshot["metadata"])
    host: dict[str, object] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if inspect_wheel_tags and any(name.startswith("packs/") for name in metadata):
        try:
            result = subprocess.run(
                [
                    str(interpreter(generation)),
                    "-I",
                    "-B",
                    "-c",
                    "import json, sys; sys.path.insert(0, sys.argv[1]); "
                    "from packaging.tags import sys_tags; "
                    "print(json.dumps([str(tag) for tag in sys_tags()]))",
                    str(Path(packaging.__file__).parent.parent),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise EnvironmentError("snapshot wheel compatibility inspection timed out") from exc
        if result.returncode:
            raise EnvironmentError("snapshot cannot inspect the generation's wheel compatibility")
        tags: object = json.loads(result.stdout)
        if (
            not isinstance(tags, list)
            or not tags
            or any(not isinstance(tag, str) for tag in cast("list[object]", tags))
        ):
            raise EnvironmentError("snapshot requires the generation's wheel compatibility tags")
        host["wheelTags"] = tags
    return {
        **snapshot,
        "tools": {"version": version("dinkster-supervisor")},
        "host": host,
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
    """Return activations and explicit imports without consulting generation trees."""
    return _records(root, "history")


def activation_entries(root: Path) -> list[dict[str, object]]:
    """Imports preserve software, but never change the active chronology."""
    return [entry for entry in history(root) if entry.get("kind", "activation") == "activation"]


def _activation_provenance(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, Journal], dict[int, str]]:
    journals: dict[str, Journal] = {}
    directory = root / "journal"
    if any(
        path.is_symlink() or path.is_junction()
        for path in (root, *root.parents, directory, root / "current")
    ):
        raise EnvironmentError("legacy import requires unlinked activation journals")
    entries = history(root)
    for path in sorted(directory.glob("*.json")):
        if (
            path.is_symlink()
            or path.is_junction()
            or not path.stem.isascii()
            or not path.stem.isdecimal()
        ):
            raise EnvironmentError("legacy import requires unlinked numeric journals")
        journal = Journal.read(path)
        if journal.incomplete:
            raise EnvironmentError("legacy import requires completed operation recovery")
        if journal.operation in {"setup", "update", "rollback", "repair", "restore"}:
            journals[path.stem] = journal
    for journal in journals.values():
        if journal.stage == "complete" and journal.history_entry is not None:
            if journal.history_entry not in entries:
                raise EnvironmentError("legacy import requires completed history publication")
    seen: set[str] = set()
    for entry in entries:
        operation = entry.get("operation")
        if not isinstance(operation, str) or operation in seen:
            raise EnvironmentError("legacy import requires unique activation identities")
        seen.add(operation)
        generation_number(entry.get("generation"))
        if (
            cast("dict[str, object]", entry["snapshot"]).get("format")
            != "dinkster.install-snapshot/1"
        ):
            raise EnvironmentError("legacy import requires supported software snapshots")
        journal = journals.get(operation)
        if journal is None or journal.stage != "complete" or journal.error is not None:
            raise EnvironmentError("legacy import requires completed activation provenance")
        if entry.get("generation") != journal.target or (
            entry.get("kind", "activation") == "activation" and journal.history_entry != entry
        ):
            raise EnvironmentError("legacy import conflicts with recorded activation provenance")
        if entry.get("kind") == "import" and entry.get("provenance") != _provenance(
            root, operation, journal
        ):
            raise EnvironmentError("legacy import conflicts with recorded import provenance")
    ancestry: dict[int, str] = {}
    current = read_current(root)
    while current is not None:
        matches = [
            (operation, journal)
            for operation, journal in journals.items()
            if journal.stage == "complete" and journal.error is None and journal.target == current
        ]
        if current in ancestry or len(matches) != 1:
            raise EnvironmentError("legacy import requires unambiguous completed ancestry")
        operation, journal = matches[0]
        if (journal.previous is None) != (journal.operation == "setup"):
            raise EnvironmentError("legacy import requires connected setup ancestry")
        ancestry[current] = operation
        current = journal.previous
    return entries, journals, ancestry


def validate_import(root: Path, entry: dict[str, object]) -> None:
    """Recheck import provenance read-only, even after its tree was collected."""
    if entry.get("kind") != "import":
        return
    entries, _, ancestry = _activation_provenance(root)
    if entry not in entries or ancestry.get(
        generation_number(entry.get("generation"))
    ) != entry.get("operation"):
        raise EnvironmentError("legacy import generation is disconnected from current ancestry")


def import_generation(root: Path, number: int) -> dict[str, object]:
    """Capture legacy software under the caller's project writer lock or refuse.

    EnvironmentError means retain/defer. Recovery and collection belong to the
    caller; this operation only appends history after complete provenance checks.
    """
    generation_number(number)
    entries, journals, ancestry = _activation_provenance(root)
    operation = ancestry.get(number)
    if operation is None:
        raise EnvironmentError("legacy import generation is disconnected from current ancestry")
    journal = journals[operation]
    tree = root / "generations" / str(number)
    seal = tree / "generation.json"
    if any(path.is_symlink() or path.is_junction() for path in (root / "generations", tree, seal)):
        raise EnvironmentError("legacy import requires an unlinked generation seal")
    record = Generation.read(seal)
    if record.number != number:
        raise EnvironmentError("generation directory and recorded number disagree")
    python = interpreter(tree)
    if not python.is_file() or python.relative_to(tree).as_posix() not in dict(record.files):
        raise EnvironmentError("legacy import requires the sealed generation interpreter")
    for name, _ in record.files:
        path = tree / name
        if any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)):
            raise EnvironmentError("legacy import refuses linked sealed software")
        # User-data paths are never a legitimate software seal input.
        if Path(name).parts[0] in {"workflows", "models", "outputs", "mounts", "user", "libraries"}:
            raise EnvironmentError("legacy import refuses user data in a software seal")
    if record.drift(tree):
        raise EnvironmentError("legacy import requires an intact generation seal")
    snapshot = capture_software(tree, inspect_wheel_tags=False)
    matching = [entry for entry in entries if entry.get("operation") == operation]
    if matching:
        recorded = cast("dict[str, object]", matching[0]["snapshot"])
        if len(matching) != 1 or {
            key: value for key, value in recorded.items() if key not in {"tools", "host"}
        } != {key: value for key, value in snapshot.items() if key not in {"tools", "host"}}:
            raise EnvironmentError("legacy import conflicts with recorded software")
        return matching[0]
    observed = {"tools": snapshot["tools"], "host": snapshot["host"]}
    snapshot["tools"] = {"version": "unknown"}
    snapshot["host"] = {
        "system": "unknown",
        "machine": "unknown",
        "python": "unknown",
    }
    entry: dict[str, object] = {
        "format": "dinkster.install-history/1",
        "kind": "import",
        "id": len(entries) + 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "reason": "legacy-import",
        "generation": number,
        "snapshot": snapshot,
        "provenance": _provenance(root, operation, journal),
        "observedWith": observed,
        "diff": [],
        "restoredFrom": None,
        "exact": True,
        "substitutions": [],
    }
    publish_entry(root, entry)
    return entry


def _provenance(root: Path, operation: str, journal: Journal) -> dict[str, object]:
    return {
        "journal": operation,
        "digest": content_digest(root / "journal" / f"{operation}.json"),
        "operation": journal.operation,
        "previous": journal.previous,
        "target": journal.target,
        "stage": journal.stage,
    }


def attempts(root: Path) -> list[dict[str, object]]:
    """Return failed restores separately from successful installation history."""
    return _records(root, "attempts")


def generation_history_coverage(root: Path, number: int) -> int:
    """Return the covering entry ID, or refuse unproven software/provenance; never write."""
    generation_number(number)
    if (root / "history").is_junction() or not (root / "history").is_dir():
        raise EnvironmentError("history coverage requires published software history")
    entries = history(root)
    journals: dict[str, Journal] = {}
    directory = root / "journal"
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
        raise EnvironmentError("history coverage requires unlinked activation journals")
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or path.is_junction() or not path.stem.isdecimal():
            raise EnvironmentError("history coverage requires unlinked numeric journals")
        journal = Journal.read(path)
        if journal.operation in {"setup", "update", "rollback", "repair", "restore"}:
            journals[path.stem] = journal
    if any(
        journal.stage == "complete"
        and journal.history_entry is not None
        and journal.history_entry not in entries
        for journal in journals.values()
    ):
        raise EnvironmentError("history coverage requires completed history publication")
    for entry in entries:
        generation_number(entry.get("generation"))
        snapshot = cast("dict[str, object]", entry["snapshot"])
        if snapshot.get("format") != "dinkster.install-snapshot/1":
            raise EnvironmentError("history coverage requires supported software snapshots")
        if entry.get("kind") == "import":
            validate_import(root, entry)
            continue
        operation = entry.get("operation")
        journal = journals.get(operation) if isinstance(operation, str) else None
        if (
            journal is None
            or journal.stage != "complete"
            or journal.error is not None
            or journal.history_entry != entry
            or journal.target != entry.get("generation")
            or journal.operation != entry.get("reason")
        ):
            raise EnvironmentError(
                "history coverage requires matching completed activation provenance"
            )
    matching = [entry for entry in entries if entry.get("generation") == number]
    activations = [journal for journal in journals.values() if journal.target == number]
    if len(matching) != 1 or len(activations) != 1:
        raise EnvironmentError("history coverage requires one unambiguous published activation")
    tree = root / "generations" / str(number)
    seal = tree / "generation.json"
    if any(path.is_symlink() or path.is_junction() for path in (root / "generations", tree, seal)):
        raise EnvironmentError("history coverage requires an unlinked generation seal")
    record = Generation.read(seal)
    if record.number != number:
        raise EnvironmentError("generation directory and recorded number disagree")
    python = interpreter(tree)
    if not python.is_file() or python.relative_to(tree).as_posix() not in dict(record.files):
        raise EnvironmentError("history coverage requires the generation interpreter")
    for name, _ in record.files:
        path = tree / name
        if any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)):
            raise EnvironmentError("history coverage refuses linked sealed software")
    if record.drift(tree):
        raise EnvironmentError("history coverage requires an intact generation seal")
    snapshot = _sealed_software(tree)
    recorded = cast("dict[str, object]", matching[0]["snapshot"])
    # Host and tools describe the observer, not the sealed generation's software.
    if {key: value for key, value in recorded.items() if key not in {"host", "tools"}} != snapshot:
        raise EnvironmentError("history coverage software differs from the sealed generation")
    return generation_number(matching[0]["id"])


def _records(root: Path, kind: str) -> list[dict[str, object]]:
    directory = root / kind
    if directory.is_symlink() or directory.is_junction():
        raise EnvironmentError("history must not be a symlink")
    entries: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or path.is_junction():
            raise EnvironmentError("history entries must not be symlinks")
        entry = read_document(path)
        number = generation_number(entry.get("id"))
        if (
            path.name != f"{number:020d}.json"
            or number != len(entries) + 1
            or entry.get("format") != f"dinkster.install-{kind}/1"
            or (kind == "history" and not isinstance(entry.get("snapshot"), dict))
            or (
                kind == "history"
                and entry.get("kind", "activation") not in ("activation", "import")
            )
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
    activations = activation_entries(root)
    previous: dict[str, object] = (
        cast("dict[str, object]", activations[-1]["snapshot"]) if activations else {}
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
    _publish_record(root, entry, "history")


def record_failed_restore(root: Path, operation: str, entry_id: int, error: str) -> None:
    """Append one immutable failure receipt under the project writer lock."""
    entries = attempts(root)
    if any(entry.get("operation") == operation for entry in entries):
        return
    _publish_record(
        root,
        {
            "format": "dinkster.install-attempts/1",
            "id": len(entries) + 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "restoredFrom": entry_id,
            "current": read_current(root),
            "error": error,
        },
        "attempts",
    )


def _publish_record(root: Path, entry: dict[str, object], kind: str) -> None:
    number = generation_number(entry.get("id"))
    directory = root / kind
    if directory.is_symlink() or directory.is_junction():
        raise EnvironmentError("history must not be a symlink")
    directory.mkdir(exist_ok=True)
    destination = directory / f"{number:020d}.json"
    if destination.is_symlink() or destination.is_junction():
        raise EnvironmentError("history entries must not be symlinks")
    if destination.exists():
        if read_document(destination) != entry:
            raise EnvironmentError("cannot replace an installation history entry")
        return
    entries = _records(root, kind)
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
