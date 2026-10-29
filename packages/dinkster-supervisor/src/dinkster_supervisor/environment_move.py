"""Relocate a project without abandoning its original data before destination health."""

from __future__ import annotations

import os
import shutil
import socket
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from .environment_collection import recover_store_collection
from .environment_constraints import requirement_hashes
from .environment_generations import build_generation, link_file, pack_operation, verify_base_inputs
from .environment_runtime import engine_assignments, require_idle, start_engine, stop_engine
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    atomic_write,
    content_digest,
    digest,
    durable_rename,
    read_current,
    write_document,
    writer_lock,
)
from .environment_stores import (
    register_project,
    registry_path,
    restore_setup_registration,
    select_store,
    setup_registration,
    store_projects,
    volume,
)


def copy_tree(
    source: Path,
    target: Path,
    *,
    dirs_exist_ok: bool = False,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
    copy_function: Callable[[str, str], str] = shutil.copy2,
) -> None:
    """Copy a tree without following links or losing Windows directory-link types."""
    links: list[tuple[Path, Path, bool]] = []

    def collect(directory: str, names: list[str]) -> set[str]:
        ignored = set() if ignore is None else ignore(directory, names)
        for name in set(names) - ignored:
            path = Path(directory) / name
            if path.is_symlink():
                links.append((path, target / path.relative_to(source), path.is_dir()))
                ignored.add(name)
        return ignored

    # CPython 3.12 copytree(symlinks=True) recreates directory links as file links.
    shutil.copytree(
        source, target, dirs_exist_ok=dirs_exist_ok, ignore=collect, copy_function=copy_function
    )
    for first, second, directory in links:
        os.symlink(os.readlink(first), second, target_is_directory=directory)
        shutil.copystat(first, second, follow_symlinks=False)
    for parent in {first.parent for first, _, _ in links}:
        shutil.copystat(parent, target / parent.relative_to(source))


def _files(root: Path, journal: str = "") -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if name == ".lock" or name == journal or name.startswith("runtime/"):
            continue
        if path.is_symlink():
            if Path(os.readlink(path)).is_absolute() or not path.resolve().is_relative_to(root):
                raise EnvironmentError("move requires relative links within the project")
            files[name] = "link:" + os.readlink(path)
        elif path.is_file():
            files[name] = content_digest(path)
    return files


@dataclass(frozen=True)
class MovePlan:
    source: Path
    target: Path
    source_store: Path
    target_store: Path
    current: int
    files: dict[str, str]

    @property
    def cross_volume(self) -> bool:
        return volume(self.source) != volume(self.target)


def plan_move(source: Path, target: Path, data: Path | None) -> MovePlan:
    if not target.is_absolute() or target.is_symlink():
        raise EnvironmentError("move destination must be absolute and unlinked")
    target = target.resolve()
    source = source.resolve()
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise EnvironmentError("move requires a new destination outside the project")
    if not target.parent.is_dir():
        raise EnvironmentError("move destination parent must exist")
    current = read_current(source)
    if current is None:
        raise EnvironmentError("project has no active generation")
    store = Path((source / "store").read_text(encoding="utf-8").strip())
    if not store.is_absolute() or store.is_symlink() or (source / "store").is_symlink():
        raise EnvironmentError("move requires an absolute unlinked store")
    destination_store = select_store(
        target, data or (store if volume(source) == volume(target) else None)
    )
    if str(source) not in store_projects(store):
        raise EnvironmentError("move source is not registered with its store")
    for path in (source / "journal").glob("*.json"):
        if Journal.read(path).incomplete:
            raise EnvironmentError("incomplete environment operation requires recovery first")
    for tree in (source / "generations").iterdir():
        generation = Generation.read(tree / "generation.json")
        if tree.is_symlink() or str(generation.number) != tree.name or generation.drift(tree):
            raise EnvironmentError("move requires intact retained generations")
        if generation.number != current and any(
            row["running"] for row in engine_assignments(source, generation.number)
        ):
            raise EnvironmentError("stop engines in inactive generations before move")
    require_idle(source)
    return MovePlan(source, target, store, destination_store, current, _files(source))


def moved_to(root: Path) -> Path | None:
    for path in sorted((root / "journal").glob("*.json")):
        record = Journal.read(path)
        if record.relocation is not None and record.stage == "complete":
            if record.relocation["sourceRoot"] == str(root.resolve()):
                return Path(cast("str", record.relocation["targetRoot"]))
    return None


def _resume(root: Path, journal: Journal) -> None:
    for binding in journal.bindings:
        if binding["running"]:
            start_engine(
                root,
                journal.target,
                cast("str", binding["name"]),
                cast("int", binding["port"]),
                tuple(cast("list[str]", binding["gpus"])),
            )


def _clean_source(path: Path, journal: Journal) -> None:
    relocation = cast("dict[str, object]", journal.relocation)
    source = Path(cast("str", relocation["sourceRoot"]))
    target = Path(cast("str", relocation["targetRoot"]))
    copied = cast("dict[str, str]", relocation["sourceFiles"])
    remaining = _files(source, "journal/" + path.name)
    if any(copied.get(name) != identity for name, identity in remaining.items()):
        raise EnvironmentError("source changed after move; refusing source cleanup")
    if any(row["running"] for row in engine_assignments(source, journal.target)):
        raise EnvironmentError("source engine restarted; refusing source cleanup")
    journal.write(target / "journal" / path.name)
    store = Path(cast("str", relocation["sourceStore"]))
    projects = [name for name in store_projects(store) if name != str(source)]
    write_document(store / "projects.json", {"format": "dinkster.store/1", "projects": projects})
    for entry in source.iterdir():
        if entry.name not in {"journal", ".lock"}:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def _recover(path: Path) -> None:
    journal = Journal.read(path)
    relocation = cast("dict[str, object]", journal.relocation)
    source = Path(cast("str", relocation["sourceRoot"]))
    target = Path(cast("str", relocation["targetRoot"]))
    if journal.stage == "complete":
        _clean_source(path, journal)
        return
    if not journal.incomplete:
        return
    if target.exists():
        marker = target / "journal" / path.name
        if not marker.is_file() or Journal.read(marker).relocation != journal.relocation:
            if any(target.iterdir()):
                raise EnvironmentError("move destination ownership cannot be verified")
        else:
            for binding in journal.bindings:
                stop_engine(target, cast("str", binding["name"]), generation=journal.target)
        if journal.setup_registration is not None:
            restore_setup_registration(target, journal.setup_registration)
        shutil.rmtree(target)
    _resume(source, journal)
    reservation = target.parent / (".dinkster-move-" + path.stem)
    if reservation.is_symlink():
        raise EnvironmentError("move reservation must not be linked")
    if reservation.exists():
        shutil.rmtree(reservation)
    replace(journal, stage="rolled-back").write(path)


def recover_moves(root: Path) -> None:
    for local in sorted((root / "journal").glob("*.json")):
        record = Journal.read(local)
        if record.relocation is None or record.stage == "rolled-back":
            continue
        source = Path(cast("str", record.relocation["sourceRoot"]))
        path = source / "journal" / local.name
        if root.resolve() not in {source, Path(cast("str", record.relocation["targetRoot"]))}:
            if not record.incomplete:
                continue
            raise EnvironmentError("move journal belongs to another project")
        authoritative = Journal.read(path)
        stores = {
            Path(cast("str", record.relocation[key])) for key in ("sourceStore", "targetStore")
        }
        with ExitStack() as locks:
            locks.enter_context(writer_lock(registry_path().parent))
            for store in sorted(stores):
                if store.exists():
                    locks.enter_context(writer_lock(store))
                    recover_store_collection(store)
            locks.enter_context(writer_lock(source))
            _recover(path)
        if authoritative.stage == "complete" and source == root.resolve():
            raise EnvironmentError(f"project moved to {record.relocation['targetRoot']}")


def _prepare(plan: MovePlan, path: Path, journal: Journal) -> None:
    source, target = plan.source, plan.target
    reservation = target.parent / (".dinkster-move-" + path.stem)
    reservation.mkdir()
    journal.write(reservation / "journal" / path.name)
    if target.exists():
        raise EnvironmentError("move destination appeared after planning")
    durable_rename(reservation, target)

    def ignore(directory: str, names: list[str]) -> set[str]:
        parent = Path(directory)
        if parent == source:
            return {".lock", "generations", "current"} if plan.cross_volume else {".lock"}
        return {path.name} if parent == source / "journal" else set()

    def copy_file(first: str, second: str) -> str:
        relative = Path(first).relative_to(source)
        if (
            not plan.cross_volume
            and relative.parts[0] == "generations"
            and "python" in relative.parts
        ):
            return link_file(first, second)
        return shutil.copy2(first, second)

    copy_tree(source, target, dirs_exist_ok=True, ignore=ignore, copy_function=copy_file)
    atomic_write(target / "store", str(plan.target_store) + "\n")
    if plan.cross_volume:
        required: dict[str, set[str]] = {"bases": set(), "engines": set(), "wheels": set()}
        for tree in (source / "generations").iterdir():
            record = Generation.read(tree / "generation.json")
            required["bases"].add(record.base[7:])
            required["engines"].add(record.engine_release[7:])
            required["wheels"].update(requirement_hashes((tree / "constraints.txt").read_text()))
            if (tree / "packs").exists():
                references = pack_operation(
                    "references",
                    generation=tree,
                    store=plan.source_store,
                    variant=record.variant,
                    worker_protocol=record.worker_protocol,
                    own_environment_size=0,
                    in_process_packs=record.in_process_packs,
                )
                hashes = references.get("wheelHashes")
                if not isinstance(hashes, list):
                    raise EnvironmentError("pack references require wheelHashes")
                required["wheels"].update(
                    digest("sha256:" + str(sha))[7:] for sha in cast("list[object]", hashes)
                )
        for category, identities in required.items():
            for identity in sorted(identities):
                entry = plan.source_store / category / identity
                if not entry.exists() and category in {"wheels", "engines"}:
                    continue
                destination = plan.target_store / category / entry.name
                if entry.is_symlink() or not entry.is_dir():
                    raise EnvironmentError("move requires digest-directory local caches")
                if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                    raise EnvironmentError("move requires digest-directory local caches")
                if not destination.exists():
                    staging = (
                        plan.target_store
                        / ".garbage"
                        / path.stem
                        / "transfer"
                        / category
                        / entry.name
                    )
                    if staging.exists():
                        shutil.rmtree(staging)
                    copy_tree(entry, staging)
                    if any(
                        p.is_symlink()
                        and (category != "bases" or not p.is_relative_to(staging / "python"))
                        for p in staging.rglob("*")
                    ):
                        raise EnvironmentError("move refuses linked cache inputs")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    durable_rename(staging, destination)
    for tree in sorted((source / "generations").iterdir()):
        generation = Generation.read(tree / "generation.json")
        destination = target / "generations" / tree.name
        if plan.cross_volume:
            base = plan.target_store / "bases" / generation.base[7:] / "python"
            expected = dict(generation.files)
            verify_base_inputs(base, expected, operation="move")
            release = plan.target_store / "engines" / generation.engine_release[7:]
            build_generation(
                root=target,
                number=generation.number,
                store=plan.target_store,
                base=generation.base,
                engine_version=generation.engine_version,
                engine_release=generation.engine_release,
                constraints=tree / "constraints.txt",
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                source_generation=tree if (tree / "packs").exists() else None,
                in_process_packs=generation.in_process_packs,
                own_environment_size=sum(
                    p.stat().st_size for p in (tree / "python").rglob("*") if p.is_file()
                ),
                catalog_lock=tree / "dinkster.lock" if (tree / "dinkster.lock").exists() else None,
                artifacts=release / "artifacts" if (tree / "dinkster.lock").exists() else None,
                refresh_cache=True,
            )
        else:
            for cfg in (destination / "packs").glob("*/pyvenv.cfg"):
                home = destination / "python" / ("" if os.name == "nt" else "bin")
                lines = cfg.read_text(encoding="utf-8").splitlines()
                atomic_write(
                    cfg,
                    "\n".join(
                        f"home = {home}" if line.startswith("home = ") else line for line in lines
                    )
                    + "\n",
                )
        checked = Generation.read(destination / "generation.json")
        if plan.cross_volume:
            if set(dict(checked.files)) != set(dict(generation.files)):
                raise EnvironmentError("moved file inventory differs from the retained generation")
            for name, sha in generation.files:
                if not name.startswith("python/") and dict(checked.files)[name] != sha:
                    raise EnvironmentError(f"moved retained input differs: {name}")
        if checked.drift(destination):
            raise EnvironmentError("moved generation has drift")
        if (destination / "packs").exists():
            pack_operation(
                "doctor",
                generation=destination,
                store=plan.target_store,
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                own_environment_size=0,
                in_process_packs=generation.in_process_packs,
                offline=True,
            )
    if plan.cross_volume:
        atomic_write(target / "current", f"{plan.current}\n")


def execute_move(plan: MovePlan) -> None:
    source, target = plan.source, plan.target
    registry_path().parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(writer_lock(registry_path().parent))
        for store in sorted({plan.source_store, plan.target_store}):
            store.mkdir(parents=True, exist_ok=True)
            locks.enter_context(writer_lock(store))
            recover_store_collection(store)
        locks.enter_context(writer_lock(source))
        if plan_move(source, target, plan.target_store) != plan:
            raise EnvironmentError("project changed after planning; request a new plan")
        bindings = engine_assignments(source, plan.current)
        if not any(binding["running"] for binding in bindings):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            bindings.append(
                {
                    "name": f"move-health-{time.time_ns()}",
                    "port": port,
                    "gpus": [],
                    "running": False,
                    "healthProbe": True,
                }
            )
        journal = Journal(
            "move",
            plan.current,
            plan.current,
            "planned",
            bindings=tuple(binding | {"targetPort": binding["port"]} for binding in bindings),
            setup_registration=setup_registration(target, plan.target_store),
            relocation={
                "sourceRoot": str(source),
                "targetRoot": str(target),
                "sourceStore": str(plan.source_store),
                "targetStore": str(plan.target_store),
            },
        )
        path = source / "journal" / f"{time.time_ns()}.json"
        journal.write(path)
        try:
            for binding in bindings:
                stop_engine(source, cast("str", binding["name"]), generation=plan.current)
            journal = replace(
                journal,
                stage="building",
                relocation=cast("dict[str, object]", journal.relocation)
                | {"sourceFiles": _files(source, "journal/" + path.name)},
            )
            journal.write(path)
            _prepare(plan, path, journal)
            register_project(target, plan.target_store)
            journal = replace(journal, stage="starting")
            journal.write(path)
            _resume(target, journal)
            for binding in journal.bindings:
                if binding.get("healthProbe"):
                    name = cast("str", binding["name"])
                    start_engine(target, journal.target, name, cast("int", binding["port"]))
                    stop_engine(target, name, generation=journal.target)
                    (target / "runtime" / str(journal.target) / f"{name}.json").unlink()
            if (
                _files(source, "journal/" + path.name)
                != cast("dict[str, object]", journal.relocation)["sourceFiles"]
            ):
                raise EnvironmentError("source changed while preparing move")
            journal = replace(journal, stage="complete")
            journal.write(path)
        except Exception as exc:
            replace(Journal.read(path), error=str(exc)).write(path)
            _recover(path)
            raise
        _clean_source(path, journal)
