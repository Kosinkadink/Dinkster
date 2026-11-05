"""Finalize the independent supervisor tools manifest after all native builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from dinkster_supervisor.environment_tools import TOOLS_ARCHIVES, validate_tools_manifest
from packaging.version import Version

from scripts.build_release import git_head, project_metadata, sha256


def inputs_changed(root: Path, before: str, after: str = "HEAD") -> bool:
    """Ignore shared lock changes outside the tools' dependency closure."""
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", before, after], cwd=root, text=True
    ).splitlines()
    direct = {
        ".github/workflows/release-tools.yml",
        "scripts/build_tools_release.py",
        "scripts/publish_tools_release.py",
        "scripts/build_native_tools.py",
        "scripts/build_environment_tools.py",
        "scripts/build_release.py",
        "scripts/publish_environment_channel.py",
    }
    if any(name in direct or name.startswith("packages/dinkster-supervisor/") for name in changed):
        return True
    if not {"pyproject.toml", "uv.lock"}.intersection(changed):
        return False

    def selected(revision: str) -> list[dict]:
        lock = tomllib.loads(
            subprocess.check_output(["git", "show", f"{revision}:uv.lock"], cwd=root, text=True)
        )
        names = set()
        pending = [("dinkster-supervisor", None)]
        visited = set()
        while pending:
            name, extra = pending.pop()
            if (name, extra) in visited:
                continue
            visited.add((name, extra))
            names.add(name)
            for package in lock["package"]:
                if package["name"] != name:
                    continue
                dependencies = list(package.get("dependencies", []))
                if extra is not None:
                    dependencies += package.get("optional-dependencies", {}).get(extra, [])
                if name == "dinkster-supervisor":
                    dependencies += package.get("dev-dependencies", {}).get("environment-tools", [])
                for dependency in dependencies:
                    pending.append((dependency["name"], None))
                    pending.extend(
                        (dependency["name"], extra) for extra in dependency.get("extra", [])
                    )
        return sorted(
            (package for package in lock["package"] if package["name"] in names),
            key=lambda package: json.dumps(package, sort_keys=True),
        )

    return selected(before) != selected(after)


def build(root: Path, output: Path) -> dict[str, object]:
    """Hash native ZIPs without deriving the tools version from an engine release."""
    actual = {path.name for path in output.iterdir()}
    if actual != TOOLS_ARCHIVES:
        raise ValueError("tools finalization requires exactly the three native ZIPs")
    _, version = project_metadata(root / "packages/dinkster-supervisor/pyproject.toml")
    built_at = datetime.now(UTC)
    manifest: dict[str, object] = {
        "format": "dinkster.tools/1",
        "repository": "Kosinkadink/Dinkster",
        "commit": git_head(root),
        "version": version,
        "builtAt": built_at.isoformat().replace("+00:00", "Z"),
        "label": f"{built_at.date().isoformat()}-{Version(version).public}",
        "artifacts": [
            {"name": name, "size": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in sorted(TOOLS_ARCHIVES)
        ],
    }
    data = (json.dumps(manifest, indent=2) + "\n").encode("ascii")
    validate_tools_manifest(data, hashlib.sha256(data).hexdigest())
    (output / "tools-manifest.json").write_bytes(data)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(output / name)}  {name}\n"
            for name in sorted([*TOOLS_ARCHIVES, "tools-manifest.json"])
        ),
        encoding="ascii",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--changed-from")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.changed_from:
        print(f"changed={str(inputs_changed(root, args.changed_from)).lower()}")
    else:
        build(root, args.output)
