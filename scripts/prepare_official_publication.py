"""Acquire a commit-bound engine and prepare own-version official pack sources."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_supervisor.environment_artifacts import validate_manifest
from dinkster_supervisor.environment_channels import engine_publication_names
from dinkster_supervisor.environment_downloads import fetch_artifact, read_metadata
from dinkster_supervisor.environment_state import digest
from dinkster_workers import load_manifest
from dinkster_workers.boundary import PROTOCOL_VERSION
from dinkster_workers.pack_wheel import pack_manifest_path

from scripts.acquire_engine_bases import acquire
from scripts.build_official_packs import build, library_metadata
from scripts.build_release import (
    development_version,
    select_dependency_wheels,
    sha256,
    versioned_sources,
)


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
    engine_dir = work / "engine"
    if candidate:
        engine_dir.mkdir()
        version = development_version(root)
        with versioned_sources(root, version) as staged:
            subprocess.run(
                [
                    "uv",
                    "build",
                    "--package",
                    "dinkster-api",
                    "--wheel",
                    "--out-dir",
                    str(engine_dir),
                ],
                cwd=staged,
                check=True,
            )
        engine = {
            "format": "dinkster.engine/1",
            "repository": "Kosinkadink/Dinkster",
            "tag": f"main-{commit}",
            "commit": commit,
            "version": version,
            "workerProtocol": PROTOCOL_VERSION,
            "artifacts": [
                {"name": wheel.name, "size": wheel.stat().st_size, "sha256": sha256(wheel)}
                for wheel in sorted(engine_dir.glob("*.whl"))
            ],
            "dependencies": select_dependency_wheels(root, "uv", bases),
        }
        (engine_dir / "release-manifest.json").write_text(json.dumps(engine), encoding="ascii")
    else:
        engine = published_engine(work, commit)
    sources = work / "sources"
    build(root, sources, packs=packs)
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    for pack in catalog.packs:
        archive = sources / (
            f"{pack.pack.replace('-', '_').replace('.', '_')}-{pack.version}-py3-none-any.whl"
        )
        if artifact_digest(archive.read_bytes()) != pack.artifact_digest:
            raise ValueError(f"publication source digest mismatch: {pack.pack}")
        with zipfile.ZipFile(archive) as wheel:
            library = not any(Path(name).name == "dinkster-pack.toml" for name in wheel.namelist())
        if library:
            name, version, _ = library_metadata(archive)
            if (name, version) != (pack.pack, pack.version) or pack.claims:
                raise ValueError(f"publication library identity mismatch: {pack.pack}")
            continue
        tree = unpack_artifact(
            archive,
            work / "admission" / pack.pack,
            pack.artifact_digest,
        )
        manifest = load_manifest(pack_manifest_path(tree))
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
    index = json.loads(read_metadata("https://releases.dinkster.org", "engine/index.json"))
    if not isinstance(index, list) or any(not isinstance(row, dict) for row in index):
        raise ValueError("engine index must contain release objects")
    matches = [row for row in index if row.get("commit") == commit]
    if len(matches) != 1:
        raise ValueError(f"engine index must identify one published release for {commit}")
    selected = matches[0]
    identity = digest(f"sha256:{selected.get('sha256')}")
    tag = f"main-{commit}"
    if selected.get("label") is not None:
        _, tag = engine_publication_names(selected, identity, "main")
    engine_dir = work / "engine"
    release = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/Kosinkadink/Dinkster/releases/tags/{quote(tag, safe='')}"]
        )
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
    engine = validate_manifest(metadata.read_bytes(), identity, None)
    name, expected_tag = engine_publication_names(engine, identity, "main")
    if (
        engine["repository"] != "Kosinkadink/Dinkster"
        or expected_tag != tag
        or engine["commit"] != commit
        or selected.get("path") != f"engine/{name}"
        or selected.get("version") != engine["version"]
    ):
        raise ValueError("published engine identity mismatch")
    mirror = "https://github.com/Kosinkadink/Dinkster/releases/download/" + quote(tag, safe="")
    for row in engine["artifacts"]:
        fetch_artifact((mirror,), row["name"], engine_dir / row["name"], row["sha256"], row["size"])
    return engine
