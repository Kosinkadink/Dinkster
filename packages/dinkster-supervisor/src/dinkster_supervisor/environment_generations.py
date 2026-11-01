"""Construct complete generations exclusively from verified local store inputs."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from .environment_constraints import requirement_hashes
from .environment_state import (
    EnvironmentError,
    Generation,
    content_digest,
    digest,
    generation_number,
    interpreter,
)


def verify_base_inputs(base: Path, expected: dict[str, str], *, operation: str) -> None:
    """Check retained interpreter payloads and internal aliases before rebuilding."""
    if not base.is_dir():
        raise EnvironmentError(f"{operation} requires the retained base in the local store")
    if base.is_symlink():
        raise EnvironmentError(f"{operation} refuses linked local base inputs")
    boundary = base.resolve(strict=True)

    def verify_directory(directory: Path, ancestors: tuple[Path, ...]) -> None:
        resolved = directory.resolve(strict=True)
        if resolved in ancestors:
            raise EnvironmentError(f"{operation} refuses cyclic local base inputs")
        entries = list(directory.iterdir())
        for path in entries:
            if path.is_symlink() and (
                Path(os.readlink(path)).is_absolute()
                or not path.resolve(strict=True).is_relative_to(boundary)
            ):
                raise EnvironmentError(f"{operation} refuses linked local base inputs")
        for path in entries:
            if (
                path.name == "__pycache__"
                or "__pycache__" in path.resolve().relative_to(boundary).parts
            ):
                continue
            if path.is_dir():
                verify_directory(path, (*ancestors, resolved))
            elif path.is_file():
                name = "python/" + path.relative_to(base).as_posix()
                if content_digest(path) != expected.get(name):
                    raise EnvironmentError(f"{operation} requires intact local base input: {name}")

    try:
        verify_directory(base, ())
    except (OSError, RuntimeError) as exc:
        raise EnvironmentError(f"{operation} cannot verify local base inputs: {exc}") from exc


def link_file(source: str, destination: str) -> str:
    # Dereference interpreter-internal aliases so every generation is self-contained.
    original = Path(source).resolve(strict=True)
    if sys.platform == "linux":
        import fcntl

        try:
            with original.open("rb") as source_file, open(destination, "xb") as target_file:
                fcntl.ioctl(target_file.fileno(), 0x40049409, source_file.fileno())  # FICLONE
            shutil.copymode(original, destination)
            return destination
        except OSError as exc:
            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV}:
                raise
            Path(destination).unlink(missing_ok=True)
    os.link(original, destination)
    return destination


def pack_operation(
    operation: str,
    *,
    generation: Path,
    store: Path,
    variant: str,
    worker_protocol: int,
    own_environment_size: int,
    source_generation: Path | None = None,
    locks: tuple[Path, ...] = (),
    in_process_packs: tuple[str, ...] = (),
    offline: bool = False,
    uv: str = "uv",
) -> dict[str, object]:
    """Invoke the generation-owned provisioner without importing engine packages."""
    if operation not in {"preflight", "acquire", "rebuild", "doctor", "references"}:
        raise EnvironmentError(f"unsupported pack lifecycle operation: {operation}")
    command = [
        str(interpreter(generation)),
        "-B",
        "-m",
        "dinkster_workers.pack_environment",
        operation,
        "--generation",
        str(generation),
        "--generation-python",
        str(interpreter(generation)),
        "--constraints",
        str(generation / "constraints.txt"),
        "--variant",
        variant,
        "--worker-protocol",
        str(worker_protocol),
        "--wheel-store",
        str(store / "wheels"),
        "--uv-cache",
        str(store / "uv-cache"),
        "--own-environment-size",
        str(own_environment_size),
        "--uv",
        uv,
    ]
    if source_generation is not None:
        command.extend(["--source-generation", str(source_generation)])
    if offline:
        command.append("--offline")
    for lock in locks:
        command.extend(["--lock", str(lock)])
    for pack in in_process_packs:
        command.extend(["--in-process-pack", pack])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if operation == "acquire" and result.returncode and not result.stdout.strip():
        raise EnvironmentError(
            "selected engine cannot acquire missing pack wheels; "
            f"select a release supporting pack acquisition: {result.stderr.strip()}"
        )
    try:
        raw: object = json.loads(result.stdout)
        if not isinstance(raw, dict):
            raise ValueError("expected an object")
        document = cast("dict[str, object]", raw)
        packs, changes = document.get("packs"), document.get("changes")
        if not isinstance(packs, list) or not isinstance(changes, list):
            raise ValueError("packs and changes must be arrays")
        messages = cast("list[object]", changes)
        if any(not isinstance(message, str) for message in messages):
            raise ValueError("changes must contain strings")
        errors = list(cast("list[str]", messages)) if operation in {"doctor", "references"} else []
        names: set[str] = set()
        for raw_pack in cast("list[object]", packs):
            if not isinstance(raw_pack, dict):
                raise ValueError("pack summary must be an object")
            row = cast("dict[str, object]", raw_pack)
            name, conflicts = row.get("pack"), row.get("conflicts")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("pack summaries must have distinct nonempty ids")
            names.add(name)
            if not isinstance(conflicts, list):
                raise ValueError("pack conflicts must be an array")
            conflicts = cast("list[object]", conflicts)
            if any(not isinstance(conflict, str) for conflict in conflicts):
                raise ValueError("pack conflicts must contain strings")
            for key in ("downloadBytes", "ownEnvironmentBytes"):
                size = row.get(key)
                if type(size) is not int or size < 0:
                    raise ValueError(f"{key} must be a nonnegative integer")
            errors.extend(f"{name}: {conflict}" for conflict in conflicts)
    except ValueError as exc:
        raise EnvironmentError(f"pack {operation}: invalid provisioner result: {exc}") from exc
    if result.returncode or errors:
        details = errors or cast("list[str]", messages)
        raise EnvironmentError(
            f"pack {operation} refused: {'; '.join(details) or 'provisioner failed'}"
        )
    return document


def project_operation(
    root: Path,
    *,
    generation: Path,
    variant: str,
    catalog_lock: Path | None = None,
    artifacts: Path | None = None,
    approved: dict[str, object] | None = None,
    target_generation: Path | None = None,
    requirements_only: bool = False,
) -> dict[str, object]:
    """Plan or apply workflow requirements through the generation's core adapter."""
    command = [
        str(interpreter(generation)),
        "-B",
        "-m",
        "dinkster.workflow_requirements",
        "project-apply" if approved is not None else "project-plan",
        "--project-root",
        str(root),
        "--source-generation",
        str(generation),
        "--variant",
        variant,
        "--json",
    ]
    if catalog_lock is not None:
        command.extend(["--catalog-lock", str(catalog_lock)])
    if artifacts is not None:
        command.extend(["--artifacts", str(artifacts)])
    if target_generation is not None:
        command[0] = str(interpreter(target_generation))
        command.extend(["--generation", str(target_generation)])
    if approved is not None:
        if target_generation is None:
            raise EnvironmentError("workflow execution requires a target generation")
        command.append("--yes")
    if requirements_only:
        command = [
            str(interpreter(generation)),
            "-I",
            "-B",
            "-c",
            "import json, sys\n"
            "from pathlib import Path\n"
            "from dinkster.project_requirements import load_project_requirements\n"
            "requirements, inputs = load_project_requirements(Path(sys.argv[1]))\n"
            "print(json.dumps({'requirements': requirements.to_wire(), 'inputs': list(inputs), "
            "'plan': {'packs': [], 'assets': [], 'errors': []}}))\n",
            str(root),
        ]
    result = subprocess.run(
        command,
        input=json.dumps(approved) if approved is not None else None,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        raw: object = json.loads(result.stdout)
        if not isinstance(raw, dict):
            raise ValueError("expected an object")
        document = cast("dict[str, object]", raw)
        requirements, plan, inputs = (
            document.get("requirements"),
            document.get("plan"),
            document.get("inputs"),
        )
        if not isinstance(requirements, dict) or not isinstance(plan, dict):
            raise ValueError("requirements and plan must be objects")
        if not isinstance(inputs, list):
            raise ValueError("inputs must be an array")
        actions = cast("dict[str, object]", plan)
        for kind in ("packs", "assets", "errors"):
            if not isinstance(actions.get(kind), list):
                raise ValueError(f"plan {kind} must be an array")
        errors = cast("list[object]", actions["errors"])
        if any(not isinstance(error, str) for error in errors):
            raise ValueError("plan errors must contain strings")
    except ValueError as exc:
        detail = result.stderr.strip() or str(exc)
        raise EnvironmentError(f"workflow planning failed: {detail}") from exc
    if result.returncode and not errors:
        raise EnvironmentError(f"workflow planning failed: {result.stderr.strip()}")
    return document


def stage_pack_sources(generation: Path, *, catalog_lock: Path, artifacts: Path) -> tuple[str, ...]:
    """Run source admission in the unsealed target, never in the supervisor interpreter."""
    if not all(path.is_absolute() for path in (generation, catalog_lock, artifacts)):
        raise EnvironmentError("source staging requires absolute generation, lock and cache paths")
    result = subprocess.run(
        [
            str(interpreter(generation)),
            "-B",
            "-m",
            "dinkster.generation_packs",
            "--generation",
            str(generation),
            "--catalog-lock",
            str(catalog_lock),
            "--artifacts",
            str(artifacts),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        raw: object = json.loads(result.stdout)
        if not isinstance(raw, dict):
            raise ValueError("expected an object")
        document = cast("dict[str, object]", raw)
        packs, errors = document.get("packs"), document.get("errors")
        if not isinstance(packs, list) or not isinstance(errors, list):
            raise ValueError("packs and errors must be arrays")
        names, messages = cast("list[object]", packs), cast("list[object]", errors)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("packs must contain nonempty ids")
        if len(set(cast("list[str]", names))) != len(names):
            raise ValueError("packs must not repeat an id")
        if any(not isinstance(message, str) for message in messages):
            raise ValueError("errors must contain strings")
    except ValueError as exc:
        raise EnvironmentError(f"pack source staging: invalid admission result: {exc}") from exc
    if result.returncode or messages:
        details = "; ".join(cast("list[str]", messages)) or "admission failed"
        raise EnvironmentError(f"pack source staging refused: {details}")
    if not (generation / "dinkster.lock").is_file():
        raise EnvironmentError("pack source staging did not write the installed source lock")
    return tuple(cast("list[str]", names))


def prepare_pack_catalogs(generation: Path, in_process_packs: tuple[str, ...]) -> None:
    """Prepare admitted pack catalogs in the target before sealing its file inventory."""
    command = [
        str(interpreter(generation)),
        "-B",
        "-m",
        "dinkster.manager",
        "--root",
        "",
        "prepare-catalogs",
        "--generation",
        str(generation),
    ]
    for pack in in_process_packs:
        command.extend(["--in-process-pack", pack])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode:
        raise EnvironmentError(
            f"pack catalog preparation failed: {result.stderr.strip() or result.stdout.strip()}"
        )


def prepare_engine_interpreter(
    destination: Path,
    *,
    store: Path,
    base: str,
    constraints: Path,
    uv: str = "uv",
    refresh_cache: bool = False,
) -> Path:
    """Prepare acquired engine inputs in a new caller-owned, unsealed directory.

    The caller holds the store writer lock and owns destination cleanup on both
    success and failure. No project, journal, pack or generation record is written.
    """
    digest(base)
    source = store / "bases" / base.removeprefix("sha256:") / "python"
    if destination.exists():
        raise EnvironmentError(f"interpreter destination already exists: {destination}")
    if not source.is_dir():
        raise EnvironmentError(f"base {base} is not in the local store")
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if source.stat().st_dev != parent.stat().st_dev:
        raise EnvironmentError("project and wheel store must be on the same volume")
    if not constraints.is_file():
        raise EnvironmentError("effective constraints.txt is missing")

    def ignore_bytecode(directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name == "__pycache__"
            or "__pycache__"
            in (Path(directory) / name).resolve().relative_to(source.resolve()).parts
        }

    destination.mkdir(parents=True)
    shutil.copytree(
        source,
        destination / "python",
        copy_function=link_file,
        ignore=ignore_bytecode,
    )
    shutil.copyfile(constraints, destination / "constraints.txt")
    python = interpreter(destination)
    if not python.is_file():
        raise EnvironmentError(f"base has no generation interpreter: {python}")
    base_python = source / python.relative_to(destination / "python")
    link_mode = "hardlink" if python.samefile(base_python) else "clone"
    wheels = store / "wheels"
    hashes = requirement_hashes(constraints.read_text(encoding="utf-8"))
    links = [
        wheels,
        *sorted({path.parent for path in wheels.glob("*/*.whl") if path.parent.name in hashes}),
    ]
    command = [
        uv,
        "pip",
        "sync",
        "--python",
        str(python),
        # Standalone bases may carry a manager marker; only this owned target is writable.
        "--break-system-packages",
        "--offline",
        "--no-index",
        "--require-hashes",
        "--only-binary",
        ":all:",
        "--link-mode",
        link_mode,
    ]
    for directory in links:
        command.extend(["--find-links", str(directory)])
    command.append(str(destination / "constraints.txt"))
    environment = {
        **os.environ,
        "UV_CACHE_DIR": str(store / "uv-cache"),
        "UV_OFFLINE": "1",
        "UV_NO_INDEX": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if refresh_cache:
        # A hand-edited hardlink can also corrupt uv's extracted cache.
        cleaned = subprocess.run(
            [uv, "cache", "clean"], env=environment, capture_output=True, text=True
        )
        if cleaned.returncode:
            raise EnvironmentError(f"cannot refresh generation cache: {cleaned.stderr.strip()}")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise EnvironmentError(f"offline generation sync failed: {result.stderr.strip()}")
    return python


def build_generation(
    *,
    root: Path,
    number: int,
    store: Path,
    base: str,
    engine_version: str,
    engine_release: str,
    constraints: Path,
    variant: str,
    worker_protocol: int,
    in_process_packs: tuple[str, ...] = (),
    source_generation: Path | None = None,
    pack_locks: tuple[Path, ...] = (),
    own_environment_size: int | None = None,
    catalog_lock: Path | None = None,
    artifacts: Path | None = None,
    prepare_packs: Callable[[Path], None] | None = None,
    acquire_pack_wheels: bool = False,
    refresh_cache: bool = False,
    uv: str = "uv",
    source_kind: str = "published",
) -> Generation:
    """Build a target without activating it or touching another generation.

    The caller has already verified/extracted the base and fetched the engine
    artifacts, and holds the project/store writer locks. ``constraints`` is
    the effective hashed requirements assembled for the selected base variant.
    Pack preflight, optional acquisition, offline rebuild and doctor run before
    source admission and inventory. Repair and move use only retained local wheels.
    Additional staging may run through ``prepare_packs``; any failure removes the
    incomplete target.
    """
    generation_number(number)
    digest(base)
    digest(engine_release)
    if (catalog_lock is None) != (artifacts is None):
        raise EnvironmentError("source staging requires both catalog lock and artifact cache")
    source = store / "bases" / base.removeprefix("sha256:") / "python"
    destination = root / "generations" / str(number)
    if destination.exists():
        raise EnvironmentError(f"generation {number} already exists")
    try:
        prepare_engine_interpreter(
            destination,
            store=store,
            base=base,
            constraints=constraints,
            uv=uv,
            refresh_cache=refresh_cache,
        )
        if source_kind == "local" or (source.parent / "manifest.json").is_file():
            shutil.copyfile(source.parent / "manifest.json", destination / "base-manifest.json")
        provisioned: set[str] = set()
        if (
            source_generation is not None
            or pack_locks
            or (in_process_packs and prepare_packs is None)
        ):
            if own_environment_size is None:
                raise EnvironmentError("pack planning requires the measured own-environment size")
            operations = (
                ("preflight", "acquire", "rebuild", "doctor")
                if acquire_pack_wheels
                else ("preflight", "rebuild", "doctor")
            )
            missing_pack_wheels = False
            for operation in operations:
                if operation == "acquire" and not missing_pack_wheels:
                    continue
                provision = pack_operation(
                    operation,
                    generation=destination,
                    store=store,
                    variant=variant,
                    worker_protocol=worker_protocol,
                    own_environment_size=own_environment_size,
                    source_generation=source_generation if operation != "doctor" else None,
                    locks=pack_locks if operation != "doctor" else (),
                    in_process_packs=in_process_packs,
                    offline=operation != "acquire",
                    uv=uv,
                )
                if operation == "preflight":
                    missing_pack_wheels = any(
                        cast("int", row["downloadBytes"]) > 0
                        for row in cast("list[dict[str, object]]", provision["packs"])
                    )
                provisioned = {
                    cast("str", row["pack"])
                    for row in cast("list[dict[str, object]]", provision["packs"])
                }
        if catalog_lock is not None and artifacts is not None:
            selected = stage_pack_sources(
                destination, catalog_lock=catalog_lock, artifacts=artifacts
            )
            missing = set(selected) - provisioned
            if missing:
                raise EnvironmentError(
                    f"selected packs have no dependency lock: {', '.join(sorted(missing))}"
                )
            prepare_pack_catalogs(destination, in_process_packs)
        if prepare_packs is not None:
            prepare_packs(destination)
        files = tuple(
            (path.relative_to(destination).as_posix(), content_digest(path))
            for path in sorted(destination.rglob("*"))
            if path.is_file()
            and (
                path.relative_to(destination).parts[0] != "packs"
                or (len(path.relative_to(destination).parts) == 3 and path.name == "pack-lock.json")
            )
            and (
                "__pycache__" not in path.relative_to(destination).parts
                or "dinkster-schema-catalog" in path.relative_to(destination).parts
            )
        )
        generation = Generation(
            number,
            base,
            engine_version,
            engine_release,
            content_digest(destination / "constraints.txt"),
            variant,
            worker_protocol,
            files,
            in_process_packs,
            source_kind,
        )
        generation.write(destination / "generation.json")
        return generation
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
