"""Select pull-request tests from changed paths and declared workspace dependencies."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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
        if any(name.startswith(f"{path.relative_to(root)}/") for name in code)
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
            not any(path.startswith(f"{package.relative_to(root)}/") for package in owned)
            for path in code
        )
    )
    tested = pack_paths | {path for path in library_paths if (path / "tests").is_dir()}
    return {
        "docs": not code,
        "engine": engine,
        "runtime": shared or any(runtime_path(path) for path in code),
        "packs": sorted(str(path.relative_to(root)) for path in selected & tested),
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


def test_paths(root: Path, packages: list[str]) -> list[str]:
    # Admission and checkout testing share the existing pack-to-root-test ownership map.
    from scripts.run_selected_pack_tests import _SUITES

    tests = set()
    for package in packages:
        path = root / package
        manifest = path / "dinkster-pack.toml"
        patterns = [f"{package}/tests/test_*.py"]
        if manifest.is_file():
            identity = tomllib.loads(manifest.read_text(encoding="utf-8"))["pack"]["name"]
            patterns.extend(_SUITES.get(identity, ()))
        found = {str(test.relative_to(root)) for pattern in patterns for test in root.glob(pattern)}
        if not found:
            raise ValueError(f"selected package has no behavioral tests: {package}")
        tests.update(found)
    return sorted(tests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--packs", help="JSON array of selected workspace package directories")
    args = parser.parse_args()
    root = Path.cwd()
    if args.packs is not None:
        tests = test_paths(root, json.loads(args.packs))
        print("Selected pack tests: " + json.dumps(tests), flush=True)
        # Separate directories can reuse test module and sibling helper names.
        groups: dict[Path, list[str]] = {}
        for test in tests:
            groups.setdefault(Path(test).parent, []).append(test)
        for group in groups.values():
            subprocess.run([sys.executable, "-m", "pytest", "-q", *group], check=True)
        return
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
