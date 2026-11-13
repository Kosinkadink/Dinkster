"""Capture explicitly selected working folders without changing their contents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .environment_state import EnvironmentError, content_digest

DEVELOPMENT_DESCRIPTION = (
    "development: built from local folders, not reviewed by the registry; "
    "dependency ranges resolved at build time, not reproducible from ranges; "
    "captured runtime artifacts remain restorable"
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
    if commit.returncode or read("check-ignore", "--quiet", "--", str(root)).returncode == 0:
        return {"commit": None, "dirty": None}
    status = read("status", "--porcelain", "--untracked-files=all", "--", str(root))
    if status.returncode:
        raise EnvironmentError(f"cannot inspect development git state: {root}")
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout)}


@contextmanager
def acquired_development_sources(args: Any) -> Generator[Any]:
    """Keep fetched folders alive only for this explicit development operation."""
    import copy

    from .git_sources import fetched_git_source

    selected = copy.copy(args)
    selected.git_origins = {}
    with ExitStack() as stack:

        def acquire(value: str | Path) -> Path:
            text = str(value)
            if "://" not in text and not text.startswith("git+"):
                return Path(value)
            folder, origin = stack.enter_context(fetched_git_source(text))
            selected.git_origins[str(folder.resolve())] = origin
            return folder

        selected.source = acquire(args.source)
        selected.pack_source = [acquire(value) for value in args.pack_source]
        selected.library_source = [acquire(value) for value in args.library_source]
        yield selected


def development_selection(
    source: Path,
    packs: tuple[Path, ...],
    libraries: tuple[Path, ...],
    git_origins: dict[str, dict[str, object]] | None = None,
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
                    **(
                        {
                            "commit": git_origins[str(root)]["commit"],
                            "dirty": False,
                            "git": {**git_origins[str(root)], "contentDigest": _identity(files)},
                        }
                        if git_origins and str(root) in git_origins
                        else _git_state(root)
                    ),
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
        from .environment_local import run_local_command

        libraries: list[Path] = []
        for number, source in enumerate(roots["library"]):
            if source.is_file():
                libraries.append(source)
                continue
            output = work / "library-wheels" / str(number)
            run_local_command(
                ["uv", "build", "--wheel", "--no-sources", "--out-dir", str(output), str(source)],
                cwd=source,
            )
            wheels = list(output.glob("*.whl"))
            if len(wheels) != 1:
                raise EnvironmentError(f"expected one development library wheel: {source}")
            libraries.append(wheels[0])
        yield roots["engine"][0], tuple(roots["pack"]), tuple(libraries)


def prepare_development_packs(
    generation: Path,
    *,
    packs: tuple[Path, ...],
    libraries: tuple[Path, ...],
    store: Path,
    release: str,
    selection: dict[str, Any],
    in_process: tuple[str, ...] = (),
) -> None:
    """Use the ordinary pack provisioner with captured, generation-built inputs."""
    from dinkster_store import object_path

    from .environment_generations import pack_operation, prepare_pack_catalogs, stage_pack_sources
    from .environment_local import run_local_command
    from .environment_state import interpreter, write_document

    artifacts = object_path(store / "engines", release) / "artifacts"
    result = json.loads(
        run_local_command(
            [
                str(interpreter(generation)),
                "-I",
                "-B",
                "-m",
                "dinkster_workers.development_sources",
            ],
            cwd=generation,
            input=json.dumps(
                {
                    "packs": [str(path) for path in packs],
                    "packPaths": [
                        row["path"]
                        for row in selection["development"]["sources"]
                        if row["kind"] == "pack"
                    ],
                    "libraries": [str(path) for path in libraries],
                    "store": str(store),
                    "artifacts": str(artifacts),
                    "generation": str(generation),
                    "variant": selection["variant"],
                    "accelerator": "cpu"
                    if selection["variant"].endswith("cpu")
                    else "mps"
                    if selection["platform"]["os"] == "macos"
                    else "cuda",
                    "index": "https://pypi.org/simple",
                }
            ),
        )
    )
    locks = tuple(Path(path) for path in result["locks"])
    for operation in ("preflight", "acquire", "rebuild", "doctor"):
        pack_operation(
            operation,
            generation=generation,
            store=store,
            variant=selection["variant"],
            worker_protocol=selection["workerProtocol"],
            own_environment_size=0,
            locks=locks if operation != "doctor" else (),
            source_generation=generation,
            in_process_packs=in_process,
            offline=operation != "acquire",
        )
    stage_pack_sources(
        generation, catalog_lock=artifacts / "development-packs.lock", artifacts=artifacts
    )
    write_document(
        generation / "development-sources.json",
        {**selection["development"], "builtWheels": result["builtWheels"]},
    )
    prepare_pack_catalogs(generation, in_process)
