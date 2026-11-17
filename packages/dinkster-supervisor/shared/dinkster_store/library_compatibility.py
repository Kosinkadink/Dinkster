"""Read-only checks before an immutable engine opens an existing library."""

from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path


class LibraryCompatibilityError(RuntimeError):
    """The selected engine cannot read the library's existing formats."""


def engine_readers() -> tuple[tuple[str, ...], bool]:
    """Inspect installed source without importing engines or running migrations."""
    assets = importlib.util.find_spec("dinkster_assets")
    revisions: list[str] = []
    if assets is not None and assets.submodule_search_locations:
        for location in assets.submodule_search_locations:
            for path in (Path(location) / "migrations" / "versions").glob("*.py"):
                for node in ast.parse(path.read_text("utf-8")).body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "revision"
                        for target in node.targets
                    ):
                        value = ast.literal_eval(node.value)
                        if isinstance(value, str):
                            revisions.append(value)
    engine = importlib.util.find_spec("dinkster")
    hidden_models = False
    if engine is not None and engine.submodule_search_locations:
        for location in engine.submodule_search_locations:
            path = Path(location) / "project_folders.py"
            if path.is_file():
                hidden_models = any(
                    isinstance(node, ast.Constant) and node.value == "hiddenModels"
                    for node in ast.walk(ast.parse(path.read_text("utf-8")))
                )
    return tuple(sorted(revisions)), hidden_models


def library_requirements(root: Path) -> tuple[tuple[str, ...], bool]:
    """Read the live SQLite revision, including WAL, without applying migrations."""
    revisions: tuple[str, ...] = ()
    database = root / "library.sqlite"
    if database.exists():
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='alembic_version'"
            ).fetchone()
            if present:
                revisions = tuple(
                    row[0] for row in connection.execute("SELECT version_num FROM alembic_version")
                )
    folders = root / "project-folders.json"
    hidden_models = folders.exists() and "hiddenModels" in json.loads(folders.read_text("utf-8"))
    return revisions, hidden_models


def check_library(root: Path, *, python: Path | None = None) -> None:
    """Refuse before serve imports, and before activation stops the previous owner."""
    revisions, hidden_models = library_requirements(root)
    if not revisions and not hidden_models:
        return
    if python is None:
        supported, supports_hidden = engine_readers()
    else:
        result = subprocess.run(
            [str(python), "-I", "-B", str(Path(__file__).resolve())],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        supported, supports_hidden = json.loads(result.stdout)
    missing = sorted(set(revisions) - set(supported))
    if missing:
        required = (
            "Dinkster 0.0.4 or newer"
            if missing == ["0011_retention_clients"]
            else "a Dinkster version supporting library schema " + ", ".join(missing)
        )
        raise LibraryCompatibilityError(
            f"Library {root} requires {required}. The selected engine cannot read it; "
            "no engine was started and library data was not changed."
        )
    if hidden_models and not supports_hidden:
        raise LibraryCompatibilityError(
            f"Library {root} contains hidden-model preferences and requires "
            "Dinkster 0.0.15 or newer with hidden-model support. "
            "The selected engine cannot read it; no engine was started "
            "and all saved preferences are kept."
        )


if __name__ == "__main__":
    print(json.dumps(engine_readers()))
