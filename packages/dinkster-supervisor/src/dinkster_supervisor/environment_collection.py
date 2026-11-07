"""Quarantine obsolete generations before committing their disposable-tree deletion."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from .environment_constraints import requirement_hashes
from .environment_generations import pack_operation
from .environment_runtime import generation_runtime_lock, generation_runtime_reason
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    atomic_write,
    content_digest,
    digest,
    durable_rename,
    durable_unlink,
    read_current,
    read_document,
    store_cache_path,
    writer_lock,
)
from .environment_stores import store_projects
from .environment_transactions import previous_activation


def store_collection_plan(root: Path, store: Path, excluded: tuple[int, ...]) -> tuple[str, ...]:
    """Unreadable references refuse deletion; other projects keep all extant generations."""
    projects = store_projects(store)
    if store.is_symlink() or store.is_relative_to(root) or root.is_relative_to(store):
        raise EnvironmentError("collection store and project must be unlinked separate trees")
    if str(root) not in projects:
        raise EnvironmentError("collection project is not registered with its store")
    bases: set[str] = set()
    engines: set[str] = set()
    wheels: set[str] = set()
    for name in projects:
        project = Path(name)
        if project.is_symlink() or not project.is_dir():
            raise EnvironmentError(f"collection cannot read registered project: {project}")
        pointer = project / "store"
        if pointer.is_symlink() or pointer.read_text(encoding="utf-8").strip() != str(store):
            raise EnvironmentError("registered project uses another store")
        if (project / "generations").is_symlink() or not (project / "generations").is_dir():
            raise EnvironmentError("collection requires readable registered generations")
        for path in (project / "journal").glob("*.json"):
            record = Journal.read(path)
            if record.incomplete and not (project == root and record.operation == "gc"):
                raise EnvironmentError(
                    "registered project requires operation recovery before collection"
                )
        for tree in (project / "generations").iterdir():
            if project == root and tree.name in {str(number) for number in excluded}:
                continue
            if tree.is_symlink() or not tree.is_dir():
                raise EnvironmentError("collection refuses linked generation references")
            generation = Generation.read(tree / "generation.json")
            if str(generation.number) != tree.name:
                raise EnvironmentError("generation directory and recorded number disagree")
            constraints = tree / "constraints.txt"
            if constraints.is_symlink() or content_digest(constraints) != generation.constraints:
                raise EnvironmentError("collection requires intact retained constraints")
            bases.add(generation.base[7:])
            engines.add(generation.engine_release[7:])
            hashes = requirement_hashes(constraints.read_text(encoding="utf-8"))
            if not hashes:
                raise EnvironmentError("collection requires hashed retained constraints")
            wheels.update(hashes)
            packs = tree / "packs"
            if packs.is_symlink():
                raise EnvironmentError("collection refuses linked pack references")
            expected_packs = set(generation.in_process_packs) | {
                Path(name).parts[1]
                for name, _ in generation.files
                if name.startswith(("pack-sources/", "packs/"))
            }
            if any(not (packs / name / "pack-lock.json").is_file() for name in expected_packs):
                raise EnvironmentError(
                    "collection requires recorded dependency locks for admitted packs"
                )
            if (tree / "packs").exists():
                if generation.drift(tree):
                    raise EnvironmentError(
                        "collection refuses a drifted pack reference interpreter"
                    )
                if any(
                    p.is_symlink() or not p.is_dir() or (p / "pack-lock.json").is_symlink()
                    for p in (tree / "packs").iterdir()
                ):
                    raise EnvironmentError("collection refuses linked dependency lock references")
                report = pack_operation(
                    "references",
                    generation=tree,
                    store=store,
                    variant=generation.variant,
                    worker_protocol=generation.worker_protocol,
                    own_environment_size=0,
                    in_process_packs=generation.in_process_packs,
                )
                hashes = report.get("wheelHashes")
                if not isinstance(hashes, list):
                    raise EnvironmentError("pack references must contain wheelHashes")
                for sha in cast("list[object]", hashes):
                    wheels.add(digest("sha256:" + str(sha))[7:])
    collected: list[str] = []
    for category, referenced in (
        ("bases", bases),
        ("downloads", bases),
        ("engines", engines),
        ("wheels", wheels),
    ):
        directory = store / category
        if directory.is_symlink():
            raise EnvironmentError("collection refuses linked cache directories")
        if not directory.exists():
            continue
        for entry in directory.iterdir():
            relative = store_cache_path(f"{category}/{entry.name}")
            if entry.is_symlink() or not entry.is_dir():
                raise EnvironmentError("collection refuses linked or non-directory cache entries")
            if entry.name not in referenced:
                collected.append(relative)
    if (store / "uv-cache").is_symlink():
        raise EnvironmentError("collection refuses linked uv cache")
    if (store / "uv-cache").exists():
        collected.append("uv-cache")
    return tuple(sorted(collected))


def _store_trees(path: Path, journal: Journal) -> tuple[tuple[Path, Path], ...]:
    if journal.collection_store is None:
        return ()
    store = Path(journal.collection_store)
    if store.is_symlink() or (store / ".garbage").is_symlink():
        raise EnvironmentError("collection refuses linked store quarantine")
    directory = _quarantine(store, path)
    trees = tuple((store / name, directory / name) for name in journal.collected_store_paths)
    for original, parked in trees:
        if any(p.is_symlink() for p in (original.parent, original, parked.parent, parked)):
            raise EnvironmentError("collection refuses linked cache trees")
    return trees


def _release_store(path: Path, journal: Journal) -> None:
    if journal.collection_store is not None:
        marker = Path(journal.collection_store) / "collection"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == str(path):
            durable_unlink(marker)


def recover_store_collection(store: Path, locked_root: Path | None = None) -> None:
    """Caller holds the store writer lock before another cache writer can proceed."""
    marker = store / "collection"
    if not marker.exists():
        return
    if marker.is_symlink():
        raise EnvironmentError("collection refuses a linked journal pointer")
    path = Path(marker.read_text(encoding="utf-8").strip())
    if marker.is_symlink() or not path.is_absolute() or path.parent.name != "journal":
        raise EnvironmentError("invalid store collection journal pointer")
    journal = Journal.read(path)
    if journal.operation != "gc" or journal.collection_store != str(store):
        raise EnvironmentError("store collection pointer names another operation")
    owner = path.parent.parent

    def recover() -> None:
        if journal.incomplete:
            recover_collection(owner, path)
        elif journal.stage == "complete":
            cleanup_collection(owner, path)
        else:
            _release_store(path, journal)

    if owner == locked_root:
        recover()
    else:
        with writer_lock(owner):
            recover()


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
            reason = generation_runtime_reason(root, number)
            if reason is not None:
                raise EnvironmentError(f"generation {number} retained: {reason}")
            collected.append(number)
    return current, tuple(sorted(retained)), tuple(sorted(collected))


def collection_policy(root: Path) -> dict[str, object]:
    """Keep cleanup preferences independent of software activation and rollback."""
    path = root / "cleanup.json"
    if path.is_symlink() or path.is_junction():
        raise EnvironmentError("cleanup policy must not be linked")
    automatic = True
    if path.exists():
        document = read_document(path)
        if (
            document.get("format") != "dinkster.cleanup/1"
            or type(document.get("automatic")) is not bool
        ):
            raise EnvironmentError("invalid cleanup policy")
        automatic = cast("bool", document["automatic"])
    return {
        "automatic": automatic,
        "previousSuccessfulActivations": 1,
        "minimumAgeSeconds": 0,
    }


def collection_preview(root: Path) -> dict[str, object]:
    """Observe generation safety without locking, recovering or authorizing deletion."""
    policy = collection_policy(root)
    directory = root / "generations"
    if directory.is_symlink() or directory.is_junction():
        raise EnvironmentError("collection refuses a linked generations directory")
    current = read_current(root)
    if current is None:
        raise EnvironmentError("project has no active generation")
    journal_directory = root / "journal"
    if journal_directory.is_symlink() or journal_directory.is_junction():
        raise EnvironmentError("collection refuses linked journals")
    journals: list[tuple[Path, Journal]] = []
    for path in sorted(journal_directory.glob("*.json")):
        if path.is_symlink():
            raise EnvironmentError("collection refuses linked journals")
        journals.append((path, Journal.read(path)))
    previous = previous_activation(root, current)
    predecessor = previous.previous if previous is not None else None
    warnings: list[str] = []
    if any(journal.incomplete for _, journal in journals):
        warnings.append("pending environment operation requires recovery")
    if not any(
        journal.stage == "complete"
        and journal.operation in {"setup", "update", "rollback", "repair", "restore"}
        and journal.target == current
        for _, journal in journals
    ):
        warnings.append("current generation has no committed activation record")
    quarantine: list[dict[str, object]] = []
    for path, journal in journals:
        if journal.operation == "gc" and journal.stage in {"complete", "planned", "building"}:
            parked = _quarantine(root, path)
            numbers = [
                number
                for number in journal.collected_generations
                if (parked / str(number)).exists()
            ]
            if numbers:
                quarantine.append(
                    {"journal": path.name, "stage": journal.stage, "generations": numbers}
                )
                warnings.append(
                    f"collection {path.stem} has quarantined generations "
                    "awaiting recovery or cleanup"
                )
    rows: list[dict[str, object]] = []
    present: set[int] = set()
    for tree in sorted(directory.iterdir()):
        if not tree.name.isdecimal() or str(int(tree.name)) != tree.name or int(tree.name) < 1:
            raise EnvironmentError("generation directory must have a positive canonical number")
        number = int(tree.name)
        present.add(number)
        reasons: list[str] = []
        state = "candidate"
        if number == current:
            state, reasons = "retained", ["current activation"]
        elif number == predecessor:
            state, reasons = "retained", ["previous successful activation"]
        try:
            if tree.is_symlink() or tree.is_junction() or not tree.is_dir():
                raise EnvironmentError("generation tree must be an unlinked directory")
            if Generation.read(tree / "generation.json").number != number:
                raise EnvironmentError("generation directory and recorded number disagree")
            reason = generation_runtime_reason(root, number)
            if reason is not None:
                state = "retained"
                reasons.append(reason)
        except (OSError, ValueError) as exc:
            state = "deferred"
            reasons.append(str(exc))
        if warnings and state == "candidate":
            state = "deferred"
            reasons.extend(warnings)
        if state == "candidate":
            reasons.append(
                "outside current and previous; durable history coverage required "
                "before automatic deletion"
            )
        rows.append({"generation": number, "status": state, "reasons": reasons})
    if current not in present or (predecessor is not None and predecessor not in present):
        raise EnvironmentError("current or previous generation tree is missing")
    return {
        "policy": policy,
        "current": current,
        "generations": rows,
        "quarantine": quarantine,
        "warnings": warnings,
    }


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
    for _, tree in _store_trees(path, journal):
        if tree.exists():
            shutil.rmtree(tree)
    _release_store(path, journal)


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
    for original, parked in _store_trees(path, journal):
        if parked.exists():
            if original.exists():
                raise EnvironmentError("collection recovery found both original and parked cache")
            durable_rename(parked, original)
        elif not original.is_dir():
            raise EnvironmentError("collection recovery cannot find an original cache entry")
    replace(journal, stage="rolled-back").write(path)
    _release_store(path, journal)


def collect_generations(
    root: Path,
    approved: tuple[int, tuple[int, ...], tuple[int, ...]],
    *,
    store: Path | None = None,
    store_paths: tuple[str, ...] = (),
) -> None:
    """Caller holds the project lock and has recovered all pending operations."""
    with generation_runtime_lock(root):
        _collect_generations(root, approved, store=store, store_paths=store_paths)


def _collect_generations(
    root: Path,
    approved: tuple[int, tuple[int, ...], tuple[int, ...]],
    *,
    store: Path | None,
    store_paths: tuple[str, ...],
) -> None:
    if collection_plan(root) != approved:
        raise EnvironmentError("generation collection changed after planning; request a new plan")
    if store is not None and store_collection_plan(root, store, approved[2]) != store_paths:
        raise EnvironmentError("store collection changed after planning; request a new plan")
    for old in sorted((root / "journal").glob("*.json")):
        record = Journal.read(old)
        if record.incomplete:
            raise EnvironmentError("incomplete environment operation requires recovery first")
        if record.operation == "gc" and record.stage == "complete":
            cleanup_collection(root, old)
    current, _, numbers = approved
    if not numbers and not store_paths:
        return
    path = root / "journal" / f"{time.time_ns()}.json"
    journal = Journal(
        "gc",
        current,
        current,
        "planned",
        collected_generations=numbers,
        collection_store=str(store) if store is not None else None,
        collected_store_paths=store_paths,
    )
    journal.write(path)
    directory = _quarantine(root, path)
    try:
        if store is not None:
            atomic_write(store / "collection", str(path) + "\n")
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
        for original, parked in _store_trees(path, journal):
            parents: list[Path] = []
            parent = parked.parent
            while not parent.exists():
                parents.append(parent)
                parent = parent.parent
            for parent in reversed(parents):
                parent.mkdir()
                if os.name != "nt":
                    descriptor = os.open(parent.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            durable_rename(original, parked)
        replace(journal, stage="complete").write(path)
    except Exception as exc:
        replace(Journal.read(path), error=str(exc)).write(path)
        recover_collection(root, path)
        raise
    cleanup_collection(root, path)
