"""Publish native supervisor tools without building or moving an engine release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from dinkster_supervisor.environment_artifacts import artifact_rows, parse_metadata
from dinkster_supervisor.environment_downloads import fetch_artifact, read_metadata
from dinkster_supervisor.environment_state import Channel, write_document
from dinkster_supervisor.environment_tools import tools_reference, validate_tools_manifest

from scripts.publish_environment_channel import check_collision, object_store, upload, write_index


def publish(release: Path) -> None:
    data = (release / "tools-manifest.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest = validate_tools_manifest(data, digest)
    if (
        os.environ.get("GITHUB_REPOSITORY") != "Kosinkadink/Dinkster"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_SHA") != manifest["commit"]
    ):
        raise ValueError("tools publication requires the exact repository main push")
    public = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
    label = f"{manifest['label']}-{digest[:8]}"
    prefix, tag = f"tools/{label}", f"tools-{label}"
    rows = artifact_rows(manifest["artifacts"], "name")
    expected_checksums = {f"{row['sha256']}  {row['name']}" for row in rows} | {
        f"{digest}  tools-manifest.json"
    }
    checksums = (release / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    if set(checksums) != expected_checksums or len(checksums) != len(expected_checksums):
        raise ValueError("tools checksums differ from the complete release")
    files = [release / str(row["name"]) for row in rows]
    for path, row in zip(files, rows, strict=True):
        if (
            path.stat().st_size != row["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError(f"tools payload differs from its manifest: {path.name}")
    files.extend([release / "tools-manifest.json", release / "SHA256SUMS"])
    if {path.name for path in release.iterdir()} != {path.name for path in files}:
        raise ValueError("unexpected tools publication input")
    client = object_store()
    check_collision(client, "sha256:" + digest, "tools")
    existing = subprocess.run(
        [
            "gh",
            "release",
            "view",
            tag,
            "--repo",
            "Kosinkadink/Dinkster",
            "--json",
            "tagName,targetCommitish",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0:
        identity = json.loads(existing.stdout)
        if identity["tagName"] != tag or identity["targetCommitish"] != manifest["commit"]:
            raise ValueError("existing tools release targets a different commit")
    else:
        # A failed lookup never permits overwriting a release or its assets.
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                "Kosinkadink/Dinkster",
                "--target",
                str(manifest["commit"]),
                "--prerelease",
                "--latest=false",
                "--title",
                f"Supervisor tools {label}",
                "--notes",
                f"Native supervisor tools {manifest['version']}.",
                *(str(path) for path in files),
            ],
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix="tools-publication-") as directory:
        work = Path(directory)
        github = f"https://github.com/Kosinkadink/Dinkster/releases/download/{quote(tag, safe='')}"
        for path in files:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
            fetch_artifact((github,), path.name, work / "github" / path.name, sha, size)
            upload(client, path, f"{prefix}/{path.name}")
            fetch_artifact(
                (public + "/" + prefix,), path.name, work / "mirror" / path.name, sha, size
            )
        index = work / "tools-index.json"
        write_index(client, index, "tools")
        upload(
            client,
            index,
            "tools/index.json",
            extra={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        if read_metadata(public, "tools/index.json") != index.read_bytes():
            raise ValueError("public tools index differs after publication")
        channel = parse_metadata(read_metadata(public, "channels/main.json"))
        Channel.parse(channel)
        reference = tools_reference(
            {
                "label": label,
                "sha256": digest,
                "manifest": f"{public}/{prefix}/tools-manifest.json",
                "release": f"{public}/{prefix}",
            }
        )
        if "tools" in channel:
            previous = tools_reference(channel["tools"])
            previous_data = read_metadata(previous["release"], "tools-manifest.json")
            previous_manifest = validate_tools_manifest(previous_data, previous["sha256"])
            if previous == reference:
                print(f"Verified main channel tools {label}: {digest}")
                return
            request = Request(
                "https://api.github.com/repos/Kosinkadink/Dinkster/compare/"
                f"{previous_manifest['commit']}...{manifest['commit']}",
                headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
            )
            with urlopen(request, timeout=120) as response:
                status = json.load(response)["status"]
            if status in {"behind", "diverged"}:
                print(f"Tools channel skipped: candidate {manifest['commit']} is {status}")
                return
            if status != "ahead":
                raise ValueError(f"different tools publication for the same source: {status}")
        channel["tools"] = reference
        path = work / "main.json"
        write_document(path, channel)
        upload(
            client,
            path,
            "channels/main.json",
            extra={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        if read_metadata(public, "channels/main.json") != path.read_bytes():
            raise ValueError("public tools channel differs after publication")
        print(f"Published main channel tools {label}: {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    publish(args.release)
