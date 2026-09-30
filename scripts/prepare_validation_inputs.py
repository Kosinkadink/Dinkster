"""Prepare the exact evidence checkout used by local and hosted validation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / "tools/evidence-revision.txt").read_text().strip()
REPOSITORY = "https://github.com/Kosinkadink/dinkster-evidence.git"


def run(*arguments: str) -> str:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def prepare(root: Path) -> None:
    if not root.exists():
        run("git", "clone", "--filter=blob:none", REPOSITORY, str(root))
    elif run("git", "-C", str(root), "rev-parse", "--is-inside-work-tree") != "true":
        raise RuntimeError(f"validation input path is not a Git checkout: {root}")

    if run("git", "-C", str(root), "status", "--porcelain"):
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
    print(f"Prepared dinkster-evidence {REVISION} at {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=ROOT / ".evidence-source")
    arguments = parser.parse_args()
    prepare(arguments.evidence_root.resolve())


if __name__ == "__main__":
    main()
