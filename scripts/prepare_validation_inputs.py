"""Prepare the exact evidence checkout used by local and hosted validation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / "tools/evidence-revision.txt").read_text().strip()
REPOSITORY = "https://github.com/Kosinkadink/dinkster-evidence.git"
MATERIALIZED_DIRECTORY = Path("scripts/comfyui_benchmark_nodes")
MATERIALIZED_FILE_PATTERN = "workflow_benchmark*.py"
MATERIALIZED_FILES = tuple(
    path.relative_to(ROOT) for path in sorted((ROOT / "tools").glob(MATERIALIZED_FILE_PATTERN))
)


def run(*arguments: str) -> str:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return result.stdout.rstrip()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def reset_materialized_sources(root: Path, status: str) -> None:
    directory = str(MATERIALIZED_DIRECTORY)
    changed_paths = [line[3:] for line in status.splitlines()]
    if any(
        path != directory
        and not path.startswith(f"{directory}/")
        and not Path(path).match(f"tools/{MATERIALIZED_FILE_PATTERN}")
        for path in changed_paths
    ):
        raise RuntimeError(f"refusing to replace a dirty validation input checkout: {root}")

    remove_path(root / MATERIALIZED_DIRECTORY)
    for path in (root / "tools").glob(MATERIALIZED_FILE_PATTERN):
        remove_path(path)
    run("git", "-C", str(root), "checkout", "--", str(MATERIALIZED_DIRECTORY))


def materialize_core_sources(root: Path) -> None:
    target_directory = root / MATERIALIZED_DIRECTORY
    remove_path(target_directory)
    shutil.copytree(ROOT / MATERIALIZED_DIRECTORY, target_directory)
    for relative_path in MATERIALIZED_FILES:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)


def prepare(root: Path) -> None:
    if not root.exists():
        run("git", "clone", "--filter=blob:none", REPOSITORY, str(root))
    elif run("git", "-C", str(root), "rev-parse", "--is-inside-work-tree") != "true":
        raise RuntimeError(f"validation input path is not a Git checkout: {root}")

    status = run("git", "-C", str(root), "status", "--porcelain", "--untracked-files=all")
    if status:
        reset_materialized_sources(root, status)
    if run("git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"refusing to replace a dirty validation input checkout: {root}")

    revision_exists = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{REVISION}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if revision_exists.returncode != 0:
        run("git", "-C", str(root), "fetch", "--quiet", "origin", REVISION)
    run("git", "-C", str(root), "checkout", "--quiet", "--detach", REVISION)
    actual = run("git", "-C", str(root), "rev-parse", "HEAD")
    if actual != REVISION:
        raise RuntimeError(f"expected validation input {REVISION}, found {actual}")
    materialize_core_sources(root)
    print(f"Prepared dinkster-evidence {REVISION} at {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root", type=Path, default=ROOT.parent / ".dinkster-evidence-source"
    )
    arguments = parser.parse_args()
    prepare(arguments.evidence_root.resolve())


if __name__ == "__main__":
    main()
