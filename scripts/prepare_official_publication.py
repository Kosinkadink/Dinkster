"""Acquire a commit-bound engine and prepare own-version official pack sources."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from dinkster_registry import Lockfile, unpack_artifact
from dinkster_supervisor.environment_artifacts import validate_manifest
from dinkster_supervisor.environment_downloads import fetch_artifact
from dinkster_supervisor.environment_state import content_digest
from dinkster_workers import load_manifest
from dinkster_workers.boundary import PROTOCOL_VERSION

from scripts.acquire_engine_bases import acquire
from scripts.build_official_packs import build
from scripts.build_release import development_version, select_dependency_wheels


def prepare_inputs(
    root: Path, work: Path, commit: str, packs: list[str], *, candidate: bool = False
) -> list[dict[str, str]]:
    """Use candidate-head dependency pins for preflight, exact published identity for main."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("publication engine commit must be a full Git identity")
    selections = json.loads((root / "scripts/base_bundles.json").read_bytes())
    work.mkdir()
    bases = acquire(
        root,
        [f"base-{variant}-{identity}" for variant, identity in selections.items()],
        work / "bases",
    )
    tag = f"main-{commit}"
    engine_dir = work / "engine"
    if candidate:
        engine_dir.mkdir()
        engine = {
            "format": "dinkster.engine/1",
            "repository": "Kosinkadink/Dinkster",
            "tag": tag,
            "commit": commit,
            "version": development_version(root),
            "workerProtocol": PROTOCOL_VERSION,
            "artifacts": [],
            "dependencies": select_dependency_wheels(root, "uv", bases),
        }
        (engine_dir / "release-manifest.json").write_text(json.dumps(engine), encoding="ascii")
    else:
        engine = published_engine(work, commit)
    sources = work / "sources"
    build(root, sources, packs=packs)
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    for pack in catalog.packs:
        tree = unpack_artifact(
            sources / f"{pack.pack}-{pack.version}.zip",
            work / "admission" / pack.pack,
            pack.artifact_digest,
        )
        manifest = load_manifest(tree / "dinkster-pack.toml")
        if manifest.compatibility is None:
            raise ValueError(f"pack {pack.pack} lacks engine compatibility")
        manifest.compatibility.require_engine(pack.pack, engine["version"])
    rows = []
    for path in bases:
        base = json.loads(path.read_bytes())
        runner = {"linux": "ubuntu-latest", "windows": "windows-latest", "macos": "macos-latest"}[
            base["platform"]["os"]
        ]
        rows.append({"variant": base["variant"], "runner": runner})
    return rows


def published_engine(work: Path, commit: str) -> dict:
    """Acquire and verify only the engine release produced by this commit."""
    tag = f"main-{commit}"
    engine_dir = work / "engine"
    release = json.loads(
        subprocess.check_output(["gh", "api", f"repos/Kosinkadink/Dinkster/releases/tags/{tag}"])
    )
    if release["draft"] or not release["published_at"] or release["tag_name"] != tag:
        raise ValueError(f"official packs require the published engine release for {commit}")
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            "Kosinkadink/Dinkster",
            "--pattern",
            "release-manifest.json",
            "--dir",
            str(engine_dir),
        ],
        check=True,
    )
    metadata = engine_dir / "release-manifest.json"
    engine = validate_manifest(metadata.read_bytes(), content_digest(metadata), None)
    if (
        engine["repository"] != "Kosinkadink/Dinkster"
        or engine["tag"] != tag
        or engine["commit"] != commit
    ):
        raise ValueError("published engine identity mismatch")
    mirror = "https://github.com/Kosinkadink/Dinkster/releases/download/" + quote(tag, safe="")
    for row in engine["artifacts"]:
        fetch_artifact((mirror,), row["name"], engine_dir / row["name"], row["sha256"], row["size"])
    return engine
