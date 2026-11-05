"""Finalize the independent supervisor tools manifest after all native builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from dinkster_supervisor.environment_tools import TOOLS_ARCHIVES, validate_tools_manifest
from packaging.version import Version

from scripts.build_release import git_head, project_metadata, sha256


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output)
