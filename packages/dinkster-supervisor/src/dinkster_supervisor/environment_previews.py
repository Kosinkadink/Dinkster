"""Managed saved-preview storage, with retained locations for existing references."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from .environment_state import EnvironmentError, read_document, write_document


def read_preview_storage(value: object) -> tuple[Path | None, tuple[Path, ...], str | None]:
    if not isinstance(value, dict):
        raise EnvironmentError("previewStorage must be an object")
    policy = cast("dict[str, object]", value)
    if set(policy) - {"directory", "previousDirectories", "identity"}:
        raise EnvironmentError("previewStorage contains unknown fields")
    identity = policy.get("identity")
    if identity is not None and (
        not isinstance(identity, str)
        or len(identity) != 32
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise EnvironmentError(
            "previewStorage.identity must be a 32-character lowercase hex string"
        )

    def directory(value: object) -> Path:
        if (
            not isinstance(value, str)
            or not value
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise EnvironmentError("preview storage locations must be absolute directories")
        return Path(value)

    selected = directory(policy["directory"]) if "directory" in policy else None
    previous = policy.get("previousDirectories", [])
    if not isinstance(previous, list):
        raise EnvironmentError("previewStorage.previousDirectories must be an array")
    retained = tuple(directory(item) for item in cast("list[object]", previous))
    if len(set(retained)) != len(retained) or selected in retained:
        raise EnvironmentError("preview storage locations must be distinct")
    if (selected is not None or retained) and identity is None:
        raise EnvironmentError("previewStorage locations require an identity")
    return selected, retained, identity


def prepare_preview_storage(
    directory: Path, project: Path, library: Path, identity: str, *, create: bool = False
) -> Path:
    """Validate only a project-owned child, never grant the operator's parent."""
    try:
        for path in (directory, *directory.parents):
            if path.is_symlink() or path.is_junction():
                raise EnvironmentError("preview storage must not contain linked directories")
        if not directory.is_dir():
            raise EnvironmentError("preview storage directory is unavailable")
        assigned = directory / ("dinkster-previews-" + identity)
        for retained in (project.resolve(), library.resolve()):
            if assigned.is_relative_to(retained) or retained.is_relative_to(assigned):
                raise EnvironmentError("preview storage overlaps retained project or library data")
        for path in (assigned, assigned / "vault"):
            if path.is_symlink() or path.is_junction() or (path.exists() and not path.is_dir()):
                raise EnvironmentError("assigned preview storage must be a real directory")
        owner = assigned / "owner.json"
        record: dict[str, object] = {
            "format": "dinkster.previews/1",
            "identity": identity,
        }
        if assigned.exists():
            if (
                owner.is_symlink()
                or owner.is_junction()
                or not owner.is_file()
                or read_document(owner) != record
            ):
                raise EnvironmentError("assigned preview storage is not owned by this project")
        elif create:
            assigned.mkdir(mode=0o700)
            write_document(owner, record)
            (assigned / "vault").mkdir(mode=0o700)
        else:
            raise EnvironmentError("assigned preview storage is unavailable")
        if not (assigned / "vault").is_dir():
            raise EnvironmentError("assigned preview vault is unavailable")
        for path in (assigned, assigned / "vault"):
            if path.is_dir():
                with tempfile.TemporaryFile(dir=path) as probe:
                    probe.write(b"preview storage probe")
        return assigned / "vault"
    except OSError as exc:
        raise EnvironmentError(f"preview storage is unavailable: {exc}") from exc


def project_preview_storage(
    project: Path, library: Path | None = None
) -> tuple[Path | None, tuple[Path, ...]]:
    settings_path = project / "environment.json"
    settings = read_document(settings_path) if settings_path.is_file() else {}
    selected, previous, identity = read_preview_storage(settings.get("previewStorage", {}))
    if identity is None:
        return None, ()
    if library is None:
        library = Path(os.environ.get("DINKSTER_LIBRARY_ROOT", str(project / "dinkster-library")))
    current = (
        prepare_preview_storage(selected, project, library, identity)
        if selected is not None
        else None
    )
    retained = tuple(prepare_preview_storage(path, project, library, identity) for path in previous)
    return current, retained
