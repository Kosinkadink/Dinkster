"""Retained software plans for a host-owned project installation prompt."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_store import object_path

from .environment_constraints import requirement_hashes, selected_wheel_requirements
from .environment_generations import (
    build_generation,
    pack_operation,
    prepare_engine_interpreter,
    restore_pack_sources,
    retained_pack_archive,
)
from .environment_runtime import engine_status
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    interpreter,
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


def _current(root: Path, *, allow_lock_drift: bool = False) -> tuple[Path, Generation, Path]:
    number = read_current(root)
    if number is None:
        raise EnvironmentError("pack installation requires an active generation")
    source = root / "generations" / str(number)
    generation = Generation.read(source / "generation.json")
    pointer = root / "store"
    store = Path(pointer.read_text(encoding="utf-8").strip())
    allowed_drift: set[str] = (
        {
            f"{kind}: {name}"
            for name, _ in generation.files
            if name.startswith("packs/") and name.endswith("/pack-lock.json")
            for kind in ("missing", "modified")
        }
        if allow_lock_drift
        else set()
    )
    if (
        generation.number != number
        or pointer.is_symlink()
        or not store.is_absolute()
        or store.is_symlink()
        or source.is_symlink()
        or any(change not in allowed_drift for change in generation.drift(source))
    ):
        raise EnvironmentError("pack installation requires intact current software and store")
    return source, generation, store


def inspect_pack_install(
    root: Path,
    selections: tuple[str, ...],
    *,
    updates: tuple[str, ...] | None = None,
    registry_arguments: tuple[str, ...] = (),
) -> dict[str, object]:
    """Read the active engine's published metadata without acquiring payloads or writing state."""
    root = root.resolve()
    source, generation, store = _current(root, allow_lock_drift=True)
    drift = generation.drift(source)
    inputs = _project_inputs(root)
    registry_inputs: dict[str, str] = {}
    for flag, value in zip(registry_arguments[::2], registry_arguments[1::2], strict=True):
        if flag == "--registries":
            path = Path(value)
            if path.is_symlink():
                raise EnvironmentError("pack installation refuses linked registry configuration")
            registry_inputs[str(path)] = content_digest(path)
    software = content_digest(source / "generation.json")
    command = [
        str(interpreter(source)),
        "-I",
        "-B",
        "-m",
        "dinkster.manager",
        "--root",
        str(root),
        *registry_arguments,
        "plan-registry-sources",
        "--collect-errors",
        "--engine-manifest",
        str(source / "engine-manifest.json"),
        "--engine-release",
        generation.engine_release,
        "--installed-lock",
        str(source / "dinkster.lock"),
        "--variant",
        generation.variant,
        "--base-id",
        generation.base,
    ]
    for selection in selections:
        command.extend(["--add", selection])
    if updates is not None:
        command.extend(["--update", *updates])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    errors: list[str] = []
    if result.returncode:
        errors.append(result.stderr.strip() or result.stdout.strip())
    if errors:
        raise EnvironmentError("published pack review refused:\n" + "\n".join(errors))
    try:
        raw: object = json.loads(result.stdout)
        if not isinstance(raw, dict):
            raise ValueError("expected an object")
        metadata = cast("dict[str, object]", raw)
        if (
            metadata.get("engineRelease") != generation.engine_release
            or metadata.get("engineVersion") != generation.engine_version
            or metadata.get("variant") != generation.variant
            or metadata.get("baseId") != generation.base
            or not isinstance(metadata.get("sources"), dict)
            or not isinstance(metadata.get("dependencyLocks"), dict)
            or not isinstance(metadata.get("plan"), list)
        ):
            raise ValueError("target engine returned inconsistent identities")
    except ValueError as exc:
        raise EnvironmentError(f"published pack review returned invalid metadata: {exc}") from exc
    base = object_path(store / "bases", generation.base) / "python"
    if not base.is_dir() or base.is_symlink():
        cast("list[str]", metadata.setdefault("errors", [])).append(
            f"retained base {generation.base} is missing or linked"
        )
    source_costs: list[dict[str, object]] = []
    for entry in cast(
        "list[dict[str, object]]", cast("dict[str, object]", metadata["sources"])["packs"]
    ):
        name, identity = cast("str", entry["pack"]), cast("str", entry["artifactDigest"])
        cached = retained_pack_archive(store, identity)
        wheel = cast("dict[str, dict[str, object]]", metadata["sourceWheels"]).get(name)
        source_costs.append(
            {
                "pack": name,
                "cached": cached is not None,
                "bytes": cached.stat().st_size
                if cached is not None
                else wheel["size"]
                if wheel
                else None,
            }
        )
    metadata["sourceCosts"] = source_costs
    dependency_costs: dict[tuple[str, str], dict[str, object]] = {}
    retained_locks: dict[str, dict[str, object]] = {}
    lock_paths = {
        name
        for name, _ in generation.files
        if name.startswith("packs/") and name.endswith("/pack-lock.json")
    } | {
        path.relative_to(source).as_posix() for path in (source / "packs").glob("*/pack-lock.json")
    }
    for name in sorted(lock_paths):
        path = source / name
        try:
            if (
                not path.is_file()
                or path.is_symlink()
                or content_digest(path) != dict(generation.files).get(name)
            ):
                raise EnvironmentError("recorded dependency lock missing or modified")
            if path.parent.name in cast("dict[str, object]", metadata["dependencyLocks"]):
                continue
            retained_locks[path.parent.name] = read_document(path)
        except (OSError, ValueError) as exc:
            cast("list[str]", metadata.setdefault("errors", [])).append(
                f"pack {path.parent.name}: {exc}"
            )
    metadata["retainedDependencyLocks"] = retained_locks
    retained_locks = retained_locks | cast(
        "dict[str, dict[str, object]]", metadata["dependencyLocks"]
    )
    for lock in retained_locks.values():
        for package in cast("list[dict[str, object]]", lock["packages"]):
            for wheel in cast("list[dict[str, object]]", package["wheels"]):
                identity, filename = cast("str", wheel["sha256"]), cast("str", wheel["filename"])
                cached = any(
                    path.is_file()
                    and not path.is_symlink()
                    and path.stat().st_size == wheel["size"]
                    and content_digest(path) == "sha256:" + identity
                    for path in (
                        object_path(store / "wheels", "sha256:" + identity) / filename,
                        store / "wheels" / filename,
                    )
                )
                dependency_costs[identity, filename] = {**wheel, "cached": cached}
    metadata["dependencyCosts"] = list(dependency_costs.values())
    metadata["maximumPackPayloadBytes"] = sum(
        cast("int", row["size"]) for row in dependency_costs.values()
    )
    if (
        _project_inputs(root) != inputs
        or any(content_digest(Path(path)) != digest for path, digest in registry_inputs.items())
        or content_digest(source / "generation.json") != software
        or generation.drift(source) != drift
    ):
        raise EnvironmentError("project changed after planning; request a new plan")
    return {
        "root": str(root),
        "store": str(store),
        "current": generation.number,
        "next": next_generation(root),
        "software": software,
        "projectInputs": inputs,
        "registryInputs": registry_inputs,
        "metadata": metadata,
        "ownEnvironmentBytes": sum(
            path.stat().st_size for path in base.rglob("*") if path.is_file()
        ),
    }


def plan_pack_install(
    root: Path,
    selections: tuple[str, ...],
    output: Path,
    *,
    updates: tuple[str, ...] | None = None,
    registry_arguments: tuple[str, ...] = (),
) -> dict[str, object]:
    """Retain verified inputs in a new host-owned disposable directory, without activation."""
    root, output = root.resolve(), output.absolute()
    if updates is None and (not selections or len(selections) > 64):
        raise EnvironmentError("pack installation requires 1 to 64 exact selections")
    source, generation, store = _current(root, allow_lock_drift=True)
    if output.is_relative_to(root) or output.is_relative_to(store):
        raise EnvironmentError("pack review inputs must be outside project and shared store")
    review = inspect_pack_install(
        root, selections, updates=updates, registry_arguments=registry_arguments
    )
    metadata = cast("dict[str, object]", review["metadata"])
    with writer_lock(store), writer_lock(root):
        source, generation, actual_store = _current(root, allow_lock_drift=True)
        if actual_store != store:
            raise EnvironmentError("project changed after planning; request a new plan")
        inputs = _project_inputs(root)
        software = content_digest(source / "generation.json")
        if (
            inputs != review["projectInputs"]
            or software != review["software"]
            or generation.number != review["current"]
        ):
            raise EnvironmentError("project changed after planning; request a new plan")
        size = cast("int", review["ownEnvironmentBytes"])
        with tempfile.TemporaryDirectory(dir=store, prefix=".pack-inputs-") as directory:
            scratch = Path(directory)
            catalog, artifacts = scratch / "dinkster.lock", scratch / "artifacts"
            write_document(catalog, cast("dict[str, object]", metadata["sources"]))
            lock_root = scratch / "locks"
            lock_root.mkdir()
            for name, lock in cast("dict[str, object]", metadata["dependencyLocks"]).items():
                write_document(lock_root / f"{name}.json", cast("dict[str, object]", lock))
            locks = tuple(sorted(lock_root.glob("*.json")))
            errors = list(cast("list[str]", metadata.get("errors", [])))
            constraints = (source / "constraints.txt").read_text(encoding="utf-8")
            try:
                selected_wheel_requirements(
                    interpreter(source),
                    constraints,
                    [
                        store / "wheels",
                        *(
                            object_path(store / "wheels", "sha256:" + identity)
                            for identity in requirement_hashes(constraints)
                        ),
                    ],
                    errors=errors,
                )
            except (OSError, ValueError) as exc:
                errors.append(f"engine {generation.engine_release}: {exc}")
            try:
                restore_pack_sources(
                    source,
                    catalog_lock=catalog,
                    artifacts=artifacts,
                    project=root,
                    store=store,
                    registry_arguments=registry_arguments,
                )
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
            wheels = scratch / "store" / "wheels"
            wheels.mkdir(parents=True)
            all_locks = cast(
                "dict[str, dict[str, object]]", metadata["retainedDependencyLocks"]
            ) | cast("dict[str, dict[str, object]]", metadata["dependencyLocks"])
            for lock in all_locks.values():
                for package in cast("list[dict[str, object]]", lock["packages"]):
                    for wheel in cast("list[dict[str, object]]", package["wheels"]):
                        identity, filename = (
                            cast("str", wheel["sha256"]),
                            cast("str", wheel["filename"]),
                        )
                        cached = object_path(store / "wheels", "sha256:" + identity) / filename
                        if not cached.is_file():
                            cached = store / "wheels" / filename
                        if cached.is_file() and content_digest(cached) == "sha256:" + identity:
                            target = wheels / identity / filename
                            target.parent.mkdir(exist_ok=True)
                            shutil.copyfile(cached, target)
            try:
                pack_operation(
                    "collect",
                    generation=source,
                    store=scratch / "store",
                    variant=generation.variant,
                    worker_protocol=generation.worker_protocol,
                    own_environment_size=size,
                    source_generation=source,
                    locks=locks,
                    in_process_packs=generation.in_process_packs,
                )
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
            if errors:
                raise EnvironmentError("pack inputs unavailable:\n" + "\n".join(errors))
            engine = scratch / "engine"
            prepare_engine_interpreter(
                engine,
                store=store,
                base=generation.base,
                constraints=source / "constraints.txt",
                uv_cache=scratch / "uv-cache",
            )
            preflight = pack_operation(
                "preflight",
                generation=engine,
                store=store,
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                own_environment_size=size,
                source_generation=source,
                locks=locks,
                in_process_packs=generation.in_process_packs,
            )
            admitted = subprocess.run(
                [
                    str(interpreter(engine)),
                    "-I",
                    "-B",
                    "-c",
                    """
import sys
from pathlib import Path
from dinkster_registry import Lockfile
from dinkster_store import pack_archive_path
from dinkster.registries import registry_source_lock
lock = Lockfile.from_record_json(Path(sys.argv[1]).read_text())
entries = [registry_source_lock(entry, pack_archive_path(Path(sys.argv[2]), entry.artifact_digest),
                               engine_version=sys.argv[3])
           if entry.source == 'registry' or entry.source.startswith('registry:') else entry
           for entry in lock.packs]
print(Lockfile.of(entries).record_json())
""",
                    str(catalog),
                    str(artifacts),
                    generation.engine_version,
                ],
                capture_output=True,
                text=True,
            )
            if admitted.returncode:
                raise EnvironmentError("pack admission refused: " + admitted.stderr.strip())
            write_document(catalog, cast("dict[str, object]", json.loads(admitted.stdout)))
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
                shutil.copytree(wheels, output / "wheels")
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
                    "dependencyLocks": all_locks,
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
        "plan": [f"select {selection}" for selection in selections]
        + ["retain engine, base, unselected pack locks and placement; preserve project data"],
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
        for wheel in (inputs / "wheels").rglob("*.whl"):
            identity = content_digest(wheel)
            destination = object_path(store / "wheels", identity) / wheel.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wheel, destination)
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
                acquire_pack_wheels=False,
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
