"""Select pull-request tests from changed paths and declared workspace dependencies."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def select(root: Path, paths: list[str]) -> dict[str, object]:
    code = [
        path for path in paths if not (path.endswith((".md", ".rst")) or path.startswith("docs/"))
    ]
    projects = {
        path.parent: tomllib.loads(path.read_text(encoding="utf-8"))
        for path in (root / "packages").glob("*/pyproject.toml")
    }
    names = {path: data["project"]["name"] for path, data in projects.items()}
    changed = {
        path
        for path in projects
        if any(name.startswith(f"{path.relative_to(root).as_posix()}/") for name in code)
    }
    library_paths = {
        path
        for path, data in projects.items()
        if data.get("tool", {}).get("dinkster", {}).get("official-library") is True
    }
    selected = changed & library_paths
    while True:
        dependencies = {names[path] for path in selected}
        dependents = {
            path
            for path, data in projects.items()
            if any(
                re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0] in dependencies
                for requirement in data["project"].get("dependencies", [])
            )
        }
        if dependents <= selected:
            break
        selected |= dependents
    selected |= changed
    pack_paths = {path for path in projects if (path / "dinkster-pack.toml").is_file()}
    tested = pack_paths | {path for path in library_paths if (path / "tests").is_dir()}
    return {
        "docs": not code,
        "engine": bool(code),
        "packs": sorted(path.relative_to(root).as_posix() for path in selected & tested),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    args = parser.parse_args()
    root = Path.cwd()
    paths = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", args.base, "HEAD"], text=True
        ).split("\0")
        if args.base
        else ["pyproject.toml"]
    )
    plan = select(root, list(filter(None, paths)))
    print(json.dumps(plan, sort_keys=True))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for key, value in plan.items():
            output.write(f"{key}={json.dumps(value)}\n")


if __name__ == "__main__":
    main()
