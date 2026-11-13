"""Select pull-request tests from changed paths and declared workspace dependencies."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def runtime_path(path: str) -> bool:
    return (
        path.startswith(
            ("packages/dinkster-supervisor/", "src/dinkster/installer", "src/dinkster/environment")
        )
        or re.match(
            r"tests/test_(environment_|supervisor|installer\.|release_install\.|setup_envs\.|run_scripts\.|launch\.)",
            path,
        )
        is not None
        or path.startswith(("setup_env", "run_dinkster"))
        or path
        in {
            "scripts/setup_envs.py",
            "scripts/base_bundles.json",
            "src/dinkster/cli.py",
            "src/dinkster/__main__.py",
            "src/dinkster/setup.py",
            "src/dinkster/manager.py",
            "src/dinkster/launch.py",
            "src/dinkster/generation_packs.py",
            "packages/dinkster-registry/src/dinkster_registry/install.py",
        }
    )


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
    owned = pack_paths | library_paths
    shared = any(
        path
        in {
            "pyproject.toml",
            "uv.lock",
            "hatch_build.py",
            "scripts/ci-fast.sh",
            "tools/ci_selection.py",
        }
        or path.startswith((".github/workflows/ci.yml", ".github/actions/"))
        for path in code
    )
    engine = (
        shared
        or bool(changed & library_paths)
        or any(
            not any(
                path.startswith(f"{package.relative_to(root).as_posix()}/") for package in owned
            )
            for path in code
        )
    )
    tested = pack_paths | {path for path in library_paths if (path / "tests").is_dir()}
    return {
        "docs": not code,
        "engine": engine,
        "runtime": shared
        or any(
            path
            in {
                "tests/test_remote.py",
                "tests/test_durable_delegations.py",
                "tests/test_produced_video_assets.py",
            }
            for path in code
        )
        or any(runtime_path(path) for path in code),
        "packs": sorted(path.relative_to(root).as_posix() for path in selected & tested),
    }


def pytest_collection_modifyitems(config, items) -> None:
    """Keep install/environment cases out of the ordinary engine pull-request lane."""
    if os.environ.get("DINKSTER_CI_RUNTIME") != "false":
        return
    selected, deselected = [], []
    for item in items:
        path = Path(item.path).relative_to(config.rootpath).as_posix()
        (deselected if runtime_path(path) else selected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


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
