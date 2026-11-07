"""Retained additive software plans for a host-owned project installation prompt."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from .environment_generations import build_generation, pack_operation, prepare_registry_inputs
from .environment_runtime import engine_status
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    read_current,
    read_document,
    relative_file,
    write_document,
    writer_lock,
)
from .environment_transactions import (
    activate,
    begin_activation,
    next_generation,
    recover_activation,
)


def _approval(record: dict[str, object]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )


def _project_inputs(root: Path) -> dict[str, str | None]:
    inputs: dict[str, str | None] = {}
    for name in ("current", "store", "project.json", "registries.toml"):
        path = root / name
        if path.is_symlink():
            raise EnvironmentError("pack installation refuses linked project metadata")
        inputs[name] = content_digest(path) if path.exists() else None
    return inputs


def _current(root: Path) -> tuple[Path, Generation, Path]:
    number = read_current(root)
    if number is None:
        raise EnvironmentError("pack installation requires an active generation")
    source = root / "generations" / str(number)
    generation = Generation.read(source / "generation.json")
    pointer = root / "store"
    store = Path(pointer.read_text(encoding="utf-8").strip())
    if (
        generation.number != number
        or pointer.is_symlink()
        or not store.is_absolute()
        or store.is_symlink()
        or source.is_symlink()
        or generation.drift(source)
    ):
        raise EnvironmentError("pack installation requires intact current software and store")
    return source, generation, store


def plan_pack_install(root: Path, selections: tuple[str, ...], output: Path) -> dict[str, object]:
    """Retain verified inputs in a new host-owned disposable directory, without activation."""
    root, output = root.resolve(), output.absolute()
    if not selections or len(selections) > 64:
        raise EnvironmentError("pack installation requires 1 to 64 exact selections")
    source, generation, store = _current(root)
    if output.is_relative_to(root) or output.is_relative_to(store):
        raise EnvironmentError("pack review inputs must be outside project and shared store")
    with writer_lock(store), writer_lock(root):
        source, generation, actual_store = _current(root)
        if actual_store != store:
            raise EnvironmentError("project changed after planning; request a new plan")
        inputs = _project_inputs(root)
        software = content_digest(source / "generation.json")
        size = sum(
            path.stat().st_size
            for path in (store / "bases" / generation.base[7:] / "python").rglob("*")
            if path.is_file()
        )
        with prepare_registry_inputs(
            store=store,
            base=generation.base,
            constraints=source / "constraints.txt",
            variant=generation.variant,
            engine_version=generation.engine_version,
            project=root,
            installed_lock=source / "dinkster.lock",
            additions=selections,
        ) as (catalog, artifacts, locks, _unavailable):
            preflight = pack_operation(
                "preflight",
                generation=catalog.parent / "engine",
                store=store,
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                own_environment_size=size,
                source_generation=source,
                locks=locks,
                in_process_packs=generation.in_process_packs,
            )
            if (
                _project_inputs(root) != inputs
                or content_digest(source / "generation.json") != software
                or generation.drift(source)
            ):
                raise EnvironmentError("project changed after planning; request a new plan")
            output.mkdir()
            try:
                shutil.copyfile(catalog, output / "dinkster.lock")
                shutil.copytree(artifacts, output / "artifacts")
                (output / "locks").mkdir()
                for lock in locks:
                    shutil.copyfile(lock, output / "locks" / lock.name)
                record: dict[str, object] = {
                    "format": "dinkster.project-pack-plan/1",
                    "root": str(root),
                    "store": str(store),
                    "current": generation.number,
                    "software": software,
                    "projectInputs": inputs,
                    "expires": time.time() + 900,
                    "engine": {
                        "version": generation.engine_version,
                        "release": generation.engine_release,
                    },
                    "base": generation.base,
                    "constraints": generation.constraints,
                    "inProcessPacks": list(generation.in_process_packs),
                    "sources": read_document(catalog),
                    "dependencyLocks": {
                        path.parent.name: read_document(path)
                        for path in sorted((source / "packs").glob("*/pack-lock.json"))
                    }
                    | {
                        cast("str", read_document(path)["pack"]): read_document(path)
                        for path in locks
                    },
                    "preflight": preflight,
                    "ownEnvironmentBytes": size,
                    "files": {
                        path.relative_to(output).as_posix(): content_digest(path)
                        for path in sorted(output.rglob("*"))
                        if path.is_file()
                    },
                }
                write_document(output / "plan.json", record)
            except BaseException:
                shutil.rmtree(output)
                raise
    return {
        "current": generation.number,
        "status": "planned",
        "plan": [f"add {selection}" for selection in selections]
        + ["retain engine, base, existing pack locks and placement; preserve project data"],
        "packPlan": record,
        "approval": _approval(record),
        "expiresInSeconds": 900,
    }


def apply_pack_install(root: Path, inputs: Path, approval: str) -> dict[str, object]:
    """Apply the exact host-approved plan once, refusing stale software before journaling."""
    root, inputs = root.resolve(), inputs.absolute()
    record = read_document(inputs / "plan.json")
    if record.get("format") != "dinkster.project-pack-plan/1" or _approval(record) != approval:
        raise EnvironmentError("pack installation approval does not match the reviewed plan")
    if record.get("root") != str(root):
        raise EnvironmentError("pack installation plan belongs to another project")
    source, generation, store = _current(root)

    def revalidate() -> None:
        if (
            read_current(root) != record["current"]
            or str(store) != record["store"]
            or _project_inputs(root) != record["projectInputs"]
            or content_digest(source / "generation.json") != record["software"]
            or generation.drift(source)
        ):
            raise EnvironmentError("project changed after planning; request a new plan")
        files = cast("dict[str, str]", record["files"])
        actual = {
            path.relative_to(inputs).as_posix()
            for path in inputs.rglob("*")
            if path.is_file()
            and path.relative_to(inputs).as_posix() not in {"plan.json", "consumed"}
        }
        if actual != set(files):
            raise EnvironmentError("reviewed pack inputs changed; request a new plan")
        for name, expected in files.items():
            path = inputs / relative_file(name)
            if (
                path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents)
                or content_digest(path) != expected
            ):
                raise EnvironmentError("reviewed pack inputs changed; request a new plan")

    with writer_lock(store), writer_lock(root):
        revalidate()
        if time.time() >= cast("float", record["expires"]) or (inputs / "consumed").exists():
            raise EnvironmentError("pack installation confirmation expired or was already used")
        with (inputs / "consumed").open("x", encoding="ascii") as handle:
            handle.write(approval)
        target = next_generation(root)
        journal = begin_activation(root, target, "update")
        try:
            replace(Journal.read(journal), stage="building").write(journal)
            build_generation(
                root=root,
                number=target,
                store=store,
                base=generation.base,
                engine_version=generation.engine_version,
                engine_release=generation.engine_release,
                constraints=source / "constraints.txt",
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                source_generation=source,
                pack_locks=tuple(sorted((inputs / "locks").glob("*.json"))),
                in_process_packs=generation.in_process_packs,
                own_environment_size=cast("int", record["ownEnvironmentBytes"]),
                catalog_lock=inputs / "dinkster.lock",
                artifacts=inputs / "artifacts",
                acquire_pack_wheels=True,
                source_kind=generation.source,
                artifact_sources=read_document(source / "artifact-sources.json")
                if (source / "artifact-sources.json").exists()
                else None,
            )
            activate(root, journal, revalidate=revalidate)
        except Exception as exc:
            replace(Journal.read(journal), error=str(exc)).write(journal)
            recover_activation(root, journal)
            raise
    return {
        "current": target,
        "status": "applied",
        "plan": [],
        "engines": list(engine_status(root)),
    }
