"""Resume the install-root layout rename without rewriting software history."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from .environment_runtime import generation_runtime_lock, generation_runtime_reason
from .environment_state import (
    LEGACY_TOOLS_REFUSAL,
    LEGACY_TOOLS_REFUSAL_BYTES,
    EnvironmentError,
    Generation,
    Journal,
    atomic_write,
    content_digest,
    digest,
    durable_rename,
    durable_unlink,
    journal_paths,
    read_current,
    read_document,
    relative_file,
    write_document,
    writer_lock,
)

_MIGRATION = "layout-migration.json"
_FORMAT = "dinkster.environment-layout-migration/1"
_REFUSAL: dict[str, object] = {
    "format": "dinkster.environment-moved/1",
    "message": "Settings moved to environment.json; use dinkster-supervisor 0.0.13 or newer.",
}


def _unlinked(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or candidate.is_junction():
            raise EnvironmentError(f"environment migration refuses linked paths: {candidate}")


def _plan(root: Path) -> dict[str, object]:
    settings = root / "project.json"
    directory = root / "generations"
    for path in (settings, directory, root / "current", root / "journal"):
        _unlinked(path)
    if read_document(settings).get("format") != "dinkster.project/1":
        raise EnvironmentError("environment migration requires supported environment settings")
    if not directory.is_dir():
        raise EnvironmentError("environment migration requires the existing software directory")
    if (root / "environment.json").exists() or (root / "builds").exists():
        raise EnvironmentError("environment migration destination already exists")
    for path in journal_paths(root):
        _unlinked(path)
        if Journal.read(path).incomplete:
            raise EnvironmentError("recover the interrupted environment operation before migration")
    builds: dict[str, object] = {}
    configurations: dict[str, str] = {}
    restart_proofs: list[Path] = []
    for tree in sorted(directory.iterdir()):
        _unlinked(tree)
        if not tree.is_dir() or not tree.name.isascii() or not tree.name.isdecimal():
            raise EnvironmentError("environment migration requires numbered software directories")
        number = int(tree.name)
        if number < 1 or str(number) != tree.name:
            raise EnvironmentError("environment migration requires canonical snapshot numbers")
        reason = generation_runtime_reason(root, number, restart_proofs=restart_proofs)
        if reason is not None:
            raise EnvironmentError(f"stop the environment before migration: {reason}")
        seal = tree / "generation.json"
        _unlinked(seal)
        if (tree / "build.json").exists():
            raise EnvironmentError("environment migration found existing build metadata")
        record = Generation.read(seal)
        if record.number != number or record.drift(tree):
            raise EnvironmentError(f"environment migration requires intact snapshot {number}")
        manifest = tree / "engine-manifest.json"
        capabilities: object = (
            read_document(manifest).get("capabilities") if manifest.is_file() else None
        )
        builds[tree.name] = {
            "metadataDigest": content_digest(seal),
            "legacy": not isinstance(capabilities, list) or "build-metadata/1" not in capabilities,
        }
        for cfg in (
            *(tree / "packs").glob("*/pyvenv.cfg"),
            *(tree / "inference").glob("*/pyvenv.cfg"),
        ):
            _unlinked(cfg.parent)
            _unlinked(cfg)
            if cfg.relative_to(tree).as_posix() in dict(record.files):
                raise EnvironmentError("environment migration cannot relocate a sealed venv home")
            configurations[cfg.relative_to(directory).as_posix()] = cfg.read_text("utf-8")
    current = read_current(root)
    if current is not None and str(current) not in builds:
        raise EnvironmentError("environment migration requires the active snapshot")
    for lifecycle in restart_proofs:
        from .macos_process_family import RESTART_REQUIRED, supersede_stopped_receipt

        with writer_lock(lifecycle):
            if not supersede_stopped_receipt(lifecycle, publish=True):
                raise EnvironmentError(RESTART_REQUIRED)
    return {
        "format": _FORMAT,
        "settingsDigest": content_digest(settings),
        "builds": builds,
        "configurations": configurations,
    }


def _validate_plan(plan: dict[str, object]) -> None:
    if (
        set(plan) != {"format", "settingsDigest", "builds", "configurations"}
        or plan.get("format") != _FORMAT
    ):
        raise EnvironmentError("unsupported environment layout migration")
    digest(plan["settingsDigest"])
    builds, configurations = plan["builds"], plan["configurations"]
    if not isinstance(builds, dict) or not isinstance(configurations, dict):
        raise EnvironmentError("invalid environment layout migration")
    for name, build in cast("dict[str, object]", builds).items():
        if not name.isascii() or not name.isdecimal() or str(int(name)) != name or int(name) < 1:
            raise EnvironmentError("migration requires canonical snapshot numbers")
        if not isinstance(build, dict):
            raise EnvironmentError("invalid migration software record")
        record = cast("dict[str, object]", build)
        if set(record) != {"metadataDigest", "legacy"} or type(record["legacy"]) is not bool:
            raise EnvironmentError("invalid migration software record")
        digest(record["metadataDigest"])
    for name, original in cast("dict[str, object]", configurations).items():
        parts = Path(relative_file(name)).parts
        if (
            len(parts) != 4
            or parts[0] not in builds
            or parts[1] not in {"packs", "inference"}
            or parts[3] != "pyvenv.cfg"
            or not isinstance(original, str)
        ):
            raise EnvironmentError("invalid migration venv configuration")


def migrate_environment(root: Path) -> bool:
    """Rename one stopped environment once; an interrupted migration resumes on retry."""
    root = root.absolute()
    receipt = root / _MIGRATION
    settings = root / "environment.json"
    old_settings = root / "project.json"
    sentinel = root / "journal" / LEGACY_TOOLS_REFUSAL
    if not any(
        path.exists() or path.is_symlink() or path.is_junction()
        for path in (receipt, old_settings, sentinel)
    ):
        return False
    for path in (root, *root.parents):
        _unlinked(path)
    for path in (receipt, settings, old_settings, sentinel):
        _unlinked(path)
    if sentinel.exists() and sentinel.read_bytes() != LEGACY_TOOLS_REFUSAL_BYTES:
        raise EnvironmentError("environment tools refusal sentinel changed")
    if not receipt.exists() and not old_settings.exists():
        return False
    with writer_lock(root, operation="layout migration"), generation_runtime_lock(root):
        if not receipt.exists() and settings.is_file() and read_document(old_settings) == _REFUSAL:
            if not sentinel.exists():
                atomic_write(sentinel, LEGACY_TOOLS_REFUSAL_BYTES)
            return False
        if receipt.exists():
            plan = read_document(receipt)
            _validate_plan(plan)
        else:
            plan = _plan(root)
            _validate_plan(plan)
            write_document(receipt, plan)
        settings = root / "environment.json"
        old_settings = root / "project.json"
        directory = root / "builds"
        old_directory = root / "generations"
        for path in (settings, old_settings, directory, old_directory):
            _unlinked(path)
        if directory.exists() and old_directory.exists():
            raise EnvironmentError("environment migration found both software directories")
        source_settings = settings if settings.exists() else old_settings
        if content_digest(source_settings) != plan["settingsDigest"]:
            raise EnvironmentError("environment settings changed during migration")
        source_directory = directory if directory.exists() else old_directory
        builds = cast("dict[str, dict[str, object]]", plan["builds"])
        if {tree.name for tree in source_directory.iterdir()} != set(builds):
            raise EnvironmentError("software inventory changed during environment migration")
        # A receipt describes the original tree; it does not authorize changed software on retry.
        for name, build in builds.items():
            tree = source_directory / name
            _unlinked(tree)
            metadata = tree / "build.json"
            source = metadata if metadata.exists() else tree / "generation.json"
            _unlinked(source)
            if content_digest(source) != build["metadataDigest"]:
                raise EnvironmentError("software metadata changed during environment migration")
            record = Generation.read(source)
            if record.number != int(name) or record.drift(tree):
                raise EnvironmentError(f"environment migration requires intact snapshot {name}")
            reason = generation_runtime_reason(root, int(name))
            if reason is not None:
                raise EnvironmentError(f"stop the environment before migration: {reason}")
        # Released writers must refuse even if a layout change is interrupted.
        atomic_write(sentinel, LEGACY_TOOLS_REFUSAL_BYTES)
        if old_directory.exists():
            durable_rename(old_directory, directory)
        for name, build in builds.items():
            tree = directory / name
            _unlinked(tree)
            reason = generation_runtime_reason(root, int(name))
            if reason is not None:
                raise EnvironmentError(f"stop the environment before migration: {reason}")
            metadata, legacy = tree / "build.json", tree / "generation.json"
            _unlinked(metadata)
            _unlinked(legacy)
            source = metadata if metadata.exists() else legacy
            if content_digest(source) != build["metadataDigest"]:
                raise EnvironmentError("software metadata changed during environment migration")
            if not metadata.exists():
                atomic_write(metadata, source.read_bytes())
            if legacy.exists() and content_digest(legacy) != build["metadataDigest"]:
                raise EnvironmentError("legacy software metadata differs from build.json")
            if build["legacy"] is False:
                durable_unlink(legacy)
        for name, original in cast("dict[str, str]", plan["configurations"]).items():
            cfg = directory / name
            _unlinked(cfg.parent)
            _unlinked(cfg)
            home = directory / Path(name).parts[0] / "python" / ("" if os.name == "nt" else "bin")
            replacement = (
                "\n".join(
                    f"home = {home}" if line.startswith("home = ") else line
                    for line in original.splitlines()
                )
                + "\n"
            )
            if cfg.read_text("utf-8") not in (original, replacement):
                raise EnvironmentError("venv configuration changed during environment migration")
            atomic_write(cfg, replacement)
        if (
            content_digest(old_settings) != plan["settingsDigest"]
            and read_document(old_settings) != _REFUSAL
        ):
            raise EnvironmentError("old environment settings changed during migration")
        if not settings.exists():
            atomic_write(settings, old_settings.read_bytes())
        # Old tools must never observe a root with neither settings nor a refusal marker.
        write_document(old_settings, _REFUSAL)
        durable_unlink(receipt)
    return True
