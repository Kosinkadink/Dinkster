"""Download and verify the pinned frontend CI bundle without a Node toolchain."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = "Kosinkadink/Dinkster-Frontend"


def github_json(endpoint: str) -> dict:
    return json.loads(subprocess.check_output(["gh", "api", endpoint]))


def download(sources: dict, output: Path) -> None:
    if sources.get("repository") != REPOSITORY:
        raise ValueError(f"frontend repository must be {REPOSITORY}")
    commit = sources.get("commit", "")
    digest = sources.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("frontend pin requires a full commit and sha256")
    name = f"dinkster-frontend-{digest}"
    page = 1
    while True:
        artifacts = github_json(
            f"repos/{REPOSITORY}/actions/artifacts?name={name}&per_page=100&page={page}"
        )["artifacts"]
        for artifact in artifacts:
            if artifact["expired"] or artifact["workflow_run"]["head_sha"] != commit:
                continue
            run = github_json(f"repos/{REPOSITORY}/actions/runs/{artifact['workflow_run']['id']}")
            if (
                run["conclusion"] != "success"
                or run["path"] != ".github/workflows/ci.yml"
                or run["event"] not in {"push", "pull_request"}
                or run["head_repository"]["full_name"] != REPOSITORY
            ):
                continue
            payload = subprocess.check_output(
                ["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"]
            )
            verify_and_extract(payload, sources, output)
            print(f"Verified {name} for {commit} from run {run['id']}")
            return
        if len(artifacts) < 100:
            break
        page += 1
    raise ValueError(f"Missing published frontend artifact {name} for commit {commit}")


def verify_and_extract(payload: bytes, sources: dict, output: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as artifact:
        if sorted(artifact.namelist()) != ["bundle.zip", "manifest.json"]:
            raise ValueError("frontend artifact must contain bundle.zip and manifest.json")
        manifest = json.loads(artifact.read("manifest.json"))
        archive = artifact.read("bundle.zip")
    if manifest.get("commit") != sources["commit"]:
        raise ValueError("frontend artifact commit does not match the pin")
    if (
        manifest.get("sha256") != sources["sha256"]
        or hashlib.sha256(archive).hexdigest() != sources["sha256"]
        or manifest.get("size") != len(archive)
    ):
        raise ValueError("frontend artifact hash or size does not match the pin")
    if not isinstance(manifest.get("version"), str):
        raise ValueError("frontend artifact must name its package version")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        if "index.html" not in names or len(names) != len(set(names)):
            raise ValueError("frontend bundle must contain a unique index.html")
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or entry.is_dir()
                or entry.external_attr >> 16 != 0o100644
            ):
                raise ValueError(f"unsafe frontend bundle entry: {entry.filename}")
        output.mkdir(parents=True, exist_ok=False)
        bundle.extractall(output)
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("scripts/release_sources.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        download(json.loads(args.sources.read_text(encoding="utf-8")), args.output)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
