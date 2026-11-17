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


def engine_readers(
    source: Path | None = None, *, python: Path | None = None
) -> tuple[tuple[str, ...], bool]:
    """Inspect source without importing engines or running migrations."""
    if python is not None:
        result = subprocess.run(
            [str(python), "-I", "-B", str(Path(__file__).resolve())],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        revisions, hidden = json.loads(result.stdout)
        return tuple(revisions), bool(hidden)
    assets = importlib.util.find_spec("dinkster_assets") if source is None else None
    asset_locations = (
        [str(source / "packages/dinkster-assets/src/dinkster_assets")]
        if source is not None
        else list(assets.submodule_search_locations or ())
        if assets
        else []
    )
    revisions: list[str] = []
    if asset_locations:
        for location in asset_locations:
            for path in (Path(location) / "migrations" / "versions").glob("*.py"):
                for node in ast.parse(path.read_text("utf-8")).body:
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                        if isinstance(node, ast.AnnAssign)
                        else []
                    )
                    if (
                        isinstance(node, (ast.Assign, ast.AnnAssign))
                        and node.value is not None
                        and any(
                            isinstance(target, ast.Name) and target.id == "revision"
                            for target in targets
                        )
                    ):
                        value = ast.literal_eval(node.value)
                        if isinstance(value, str):
                            revisions.append(value)
    engine = importlib.util.find_spec("dinkster") if source is None else None
    engine_locations = (
        [str(source / "src/dinkster")]
        if source is not None
        else list(engine.submodule_search_locations or ())
        if engine
        else []
    )
    hidden_models = False
    if engine_locations:
        for location in engine_locations:
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


def minimum_release(revisions: tuple[str, ...], hidden_models: bool) -> str | None:
    releases = {
        "0001_asset_records": 2,
        "0002_asset_observations": 2,
        "0003_asset_retentions": 3,
        "0004_derivation_references": 3,
        "0005_retention_completeness": 3,
        "0006_cleanup_policy": 3,
        "0007_result_offers": 3,
        "0008_terminal_results": 3,
        "0009_artifact_offers": 3,
        "0010_result_renditions": 3,
        "0011_retention_clients": 4,
    }
    if any(revision not in releases for revision in revisions):
        return None
    return f"0.0.{max([15 if hidden_models else 1, *(releases[r] for r in revisions)])}"


def check_library(root: Path, *, python: Path | None = None) -> None:
    """Refuse before serve imports, and before activation stops the previous owner."""
    revisions, hidden_models = library_requirements(root)
    if not revisions and not hidden_models:
        return
    supported, supports_hidden = engine_readers(python=python)
    missing = sorted(set(revisions) - set(supported))
    if missing:
        release = minimum_release(tuple(missing), False)
        required = (
            f"Dinkster {release} or newer"
            if release is not None
            else "a Dinkster version supporting library schema " + ", ".join(missing)
        )
        raise LibraryCompatibilityError(
            f"Library {root} requires {required}. The selected engine cannot read it; "
            "no engine was started and library data was not changed."
        )
    if hidden_models and not supports_hidden:
        document = json.loads((root / "project-folders.json").read_text("utf-8"))
        projects = sorted(project for project, models in document["hiddenModels"].items() if models)
        remedy = (
            "Unhide models in projects " + ", ".join(projects) + " using the current engine "
            "to let this older engine open the library again."
            if projects
            else "Open the library once with the current engine to normalize empty preferences."
        )
        raise LibraryCompatibilityError(
            f"Library {root} contains hidden-model preferences and requires "
            "Dinkster 0.0.15 or newer with hidden-model support. "
            f"{remedy} No engine was started and all saved preferences are kept."
        )


if __name__ == "__main__":
    print(json.dumps(engine_readers()))
