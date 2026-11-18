"""Managed scratch selection without granting the selected host directory."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import cast

from .environment_state import EnvironmentError, read_document, write_document


def read_scratch_storage(value: object) -> Path | None:
    if not isinstance(value, dict):
        raise EnvironmentError("scratchStorage must contain only directory")
    policy = cast("dict[str, object]", value)
    if set(policy) - {"directory"}:
        raise EnvironmentError("scratchStorage must contain only directory")
    if not policy:
        return None
    directory = policy.get("directory")
    if (
        not isinstance(directory, str)
        or not directory
        or not Path(directory).is_absolute()
        or ".." in Path(directory).parts
    ):
        raise EnvironmentError("scratchStorage.directory must be an absolute directory")
    return Path(directory)


def project_scratch_storage(project: Path) -> Path | None:
    settings_path = project / "environment.json"
    settings = read_document(settings_path) if settings_path.is_file() else {}
    directory = read_scratch_storage(settings.get("scratchStorage", {}))
    if directory is None:
        return None
    library = Path(os.environ.get("DINKSTER_LIBRARY_ROOT", str(project / "dinkster-library")))
    return prepare_scratch_storage(directory, project, library)


def prepare_scratch_storage(directory: Path, project: Path, library: Path) -> Path:
    try:
        for path in (directory, *directory.parents):
            if path.is_symlink() or path.is_junction():
                raise EnvironmentError("scratch storage must not contain linked directories")
        if not directory.is_dir():
            raise EnvironmentError("scratch storage directory is unavailable")
        identity = hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:24]
        assigned = directory / ("dinkster-scratch-" + identity)
        for retained in (library.resolve(), project.resolve()):
            if assigned.is_relative_to(retained) or retained.is_relative_to(assigned):
                raise EnvironmentError("scratch storage overlaps retained project or library data")
        paths = (assigned, assigned / "staging", assigned / "workers", assigned / "packs")
        for path in paths:
            if path.is_symlink() or path.is_junction() or (path.exists() and not path.is_dir()):
                raise EnvironmentError("assigned scratch storage must be a real directory")
        owner = assigned / "owner.json"
        record: dict[str, object] = {
            "format": "dinkster.scratch/1",
            "project": str(project.resolve()),
        }
        if assigned.exists():
            if owner.is_symlink() or not owner.is_file() or read_document(owner) != record:
                raise EnvironmentError("assigned scratch storage is not owned by this project")
        else:
            assigned.mkdir(mode=0o700)
            write_document(owner, record)
        for path in paths:
            path.mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)
            with tempfile.TemporaryFile(dir=path) as probe:
                probe.write(b"scratch storage probe")
        return assigned
    except OSError as exc:
        raise EnvironmentError(f"scratch storage is unavailable: {exc}") from exc
