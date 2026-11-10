"""Capture explicitly selected working folders without changing their contents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .environment_state import EnvironmentError, content_digest

DEVELOPMENT_DESCRIPTION = (
    "development: built from local folders, not reviewed by the registry; "
    "dependency ranges resolved at build time"
)
_EXCLUDED = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".amp",
        ".venv",
        ".tox",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".pyright",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
    }
)


def _ignored(name: str) -> bool:
    return name in _EXCLUDED or name.endswith((".egg-info", ".pyc", ".pyo"))


def _files(root: Path) -> dict[str, str]:
    if root.is_symlink():
        raise EnvironmentError(f"development source must not be a symlink: {root}")
    if root.is_file():
        if root.suffix != ".whl":
            raise EnvironmentError(f"development library input must be a folder or wheel: {root}")
        return {root.name: content_digest(root)}
    if not root.is_dir():
        raise EnvironmentError(f"development source folder is missing: {root}")
    files: dict[str, str] = {}
    for directory, folders, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        folders[:] = sorted(name for name in folders if not _ignored(name))
        for name in (*folders, *sorted(names)):
            if _ignored(name):
                continue
            path = parent / name
            if path.is_symlink():
                raise EnvironmentError(f"development source contains a symlink: {path}")
            if path.is_file():
                files[path.relative_to(root).as_posix()] = content_digest(path)
    return files


def _identity(document: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _git_state(root: Path) -> dict[str, object]:
    if shutil.which("git") is None:
        return {"commit": None, "dirty": None}
    directory = root if root.is_dir() else root.parent
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def read(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            env=environment,
        )

    commit = read("rev-parse", "HEAD")
    if commit.returncode:
        return {"commit": None, "dirty": None}
    status = read("status", "--porcelain", "--untracked-files=all", "--", str(root))
    if status.returncode:
        raise EnvironmentError(f"cannot inspect development git state: {root}")
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout)}


def development_selection(
    source: Path, packs: tuple[Path, ...], libraries: tuple[Path, ...]
) -> dict[str, Any]:
    """Read source identities for a plan without creating a project or cache."""
    if not packs:
        raise EnvironmentError("development requires at least one explicit --pack-source")
    rows: list[dict[str, Any]] = []
    for kind, paths in (("engine", (source,)), ("pack", packs), ("library", libraries)):
        for path in paths:
            if path.is_symlink():
                raise EnvironmentError(f"development source must not be a symlink: {path}")
            root = path.resolve()
            files = _files(root)
            required = "dinkster-pack.toml" if kind == "pack" else "pyproject.toml"
            if not root.is_file() and required not in files:
                raise EnvironmentError(f"development {kind} requires {root / required}")
            if kind != "library" and root.is_file():
                raise EnvironmentError(f"development {kind} requires a folder: {root}")
            if any(row["path"] == str(root) for row in rows):
                raise EnvironmentError(f"duplicate development source: {root}")
            rows.append(
                {
                    "kind": kind,
                    "path": str(root),
                    "packageRoots": [".", "packages/dinkster-api", "packages/dinkster-supervisor"]
                    if kind == "engine"
                    else ["."],
                    "contentDigest": _identity(files),
                    "files": files,
                    **_git_state(root),
                }
            )
    return {"format": "dinkster.development-sources/1", "sources": rows}


def verify_development_sources(selection: dict[str, Any]) -> None:
    """Refuse changed author inputs before any generation is built."""
    for row in selection["sources"]:
        if _identity(_files(Path(row["path"]))) != row["contentDigest"]:
            raise EnvironmentError(
                f"development source changed after planning: {row['path']}; request a new plan"
            )


@contextmanager
def captured_development_sources(
    selection: dict[str, Any], store: Path
) -> Generator[tuple[Path, tuple[Path, ...], tuple[Path, ...]]]:
    """Build from verified copies, including dirty and untracked selected files."""
    verify_development_sources(selection)
    with tempfile.TemporaryDirectory(dir=store, prefix=".development-") as directory:
        work = Path(directory)
        roots: dict[str, list[Path]] = {"engine": [], "pack": [], "library": []}
        for number, row in enumerate(selection["sources"]):
            source = Path(row["path"])
            target = work / str(number)
            if source.is_file():
                target.mkdir()
                target /= source.name
                shutil.copyfile(source, target)
            else:
                shutil.copytree(
                    source,
                    target,
                    symlinks=True,
                    ignore=lambda _directory, names: [name for name in names if _ignored(name)],
                )
            if _identity(_files(target)) != row["contentDigest"]:
                raise EnvironmentError(f"development source changed during capture: {source}")
            roots[row["kind"]].append(target)
        yield roots["engine"][0], tuple(roots["pack"]), tuple(roots["library"])
