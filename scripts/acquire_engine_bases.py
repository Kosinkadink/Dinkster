"""Acquire explicitly selected published bases compatible with this engine lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from scripts.build_release import LINUX_PLATFORMS, read_base_manifest

REPOSITORY = "Kosinkadink/Dinkster-Environments"
TORCH_PACKAGES = {"torch", "torchvision"}


def qualify(root: Path, path: Path, tag: str) -> dict:
    manifest = read_base_manifest(path)
    variant = manifest["variant"]
    if tag != f"base-{variant}-{manifest['id'].removeprefix('sha256:')}":
        raise ValueError("base tag does not match its variant and canonical id")
    platform = manifest["platform"]
    if platform not in (
        {"arch": "x86_64", "os": "windows"},
        {"arch": "x86_64", "os": "linux"},
    ):
        raise ValueError("base platform is incompatible with engine")
    python = Version(manifest["python"])
    project = tomllib.loads((root / "pyproject.toml").read_text())
    project_python = Requirement(f"dinkster{project['project']['requires-python']}").specifier
    if not project_python.contains(python):
        raise ValueError("base Python is incompatible with the project")
    boundary = (root / "packages/dinkster-workers/src/dinkster_workers/boundary.py").read_text()
    protocol = re.search(r"(?m)^PROTOCOL_VERSION = ([0-9]+)$", boundary)
    if (
        protocol is None
        or manifest["workerProtocol"] != int(protocol[1])
    ):
        raise ValueError("base worker protocol is incompatible with engine")
    interpreter = manifest["interpreter"]
    triple = "pc-windows-msvc" if platform["os"] == "windows" else "unknown-linux-gnu"
    filename = (
        f"cpython-{manifest['python']}+{interpreter['release']}-x86_64-{triple}"
        "-install_only_stripped.tar.gz"
    )
    if (
        interpreter["filename"] != filename
        or unquote(interpreter["url"])
        != f"https://github.com/astral-sh/python-build-standalone/releases/download/{interpreter['release']}/{filename}"
        or not re.fullmatch(r"[0-9a-f]{64}", interpreter["sha256"])
        or type(interpreter["size"]) is not int
        or interpreter["size"] <= 0
        or interpreter["executable"]
        != ("python/python.exe" if platform["os"] == "windows" else "python/bin/python3")
    ):
        raise ValueError("base interpreter does not match its target or source identity")
    platforms = ["win_amd64"] if platform["os"] == "windows" else LINUX_PLATFORMS
    compatible = set(cpython_tags((python.major, python.minor), platforms=platforms)) | set(
        compatible_tags(
            (python.major, python.minor),
            interpreter=f"cp{python.major}{python.minor}",
            platforms=platforms,
        )
    )
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(canonicalize_name(package["name"]), []).append(package)
    members = {}
    for wheel in manifest["wheels"]:
        name = canonicalize_name(wheel["name"])
        wheel_name, version, _, tags = parse_wheel_filename(wheel["filename"])
        if (
            name in members
            or wheel_name != name
            or str(version) != wheel["version"]
            or not tags.intersection(compatible)
            or not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
            or type(wheel["size"]) is not int
            or wheel["size"] <= 0
        ):
            raise ValueError(f"invalid or duplicate base wheel: {name}")
        members[name] = version
        locked = packages.get(name, [])
        matching = [p for p in locked if p["version"] == wheel["lockedVersion"]]
        if name in TORCH_PACKAGES:
            index = f"https://download.pytorch.org/whl/{version.local}"
            if (
                not matching
                or version.public != wheel["lockedVersion"]
                or not version.local
                or wheel["index"] != index
                or wheel["verification"] != "index"
            ):
                raise ValueError(f"base torch version or variant index disagrees with lock: {name}")
        elif locked:
            identity = (wheel["size"], "sha256:" + wheel["sha256"])
            if (
                wheel["verification"] != "lock"
                or str(version) != wheel["lockedVersion"]
                or not any(
                    (row.get("size"), row.get("hash")) == identity
                    and unquote(urlsplit(row["url"]).path).rsplit("/", 1)[-1] == wheel["filename"]
                    for package in matching
                    for row in package.get("wheels", [])
                )
            ):
                raise ValueError(f"base wheel does not match engine lock: {name}")
        elif wheel["lockedVersion"] is not None or wheel["verification"] != "index":
            raise ValueError(f"base transitive wheel falsely claims lock verification: {name}")
    if not TORCH_PACKAGES.issubset(members):
        raise ValueError("base is missing a required torch wheel")
    for text in project["dependency-groups"]["engine-base"]:
        requirement = Requirement(text)
        version = members.get(canonicalize_name(requirement.name))
        if version is None or not requirement.specifier.contains(Version(version.public)):
            raise ValueError(f"base is missing a compatible engine-base pin: {requirement.name}")
    return manifest


def acquire(root: Path, tags: list[str], output: Path) -> list[Path]:
    if not tags or output.exists():
        raise ValueError("select at least one exact base tag and a new output directory")
    with tempfile.TemporaryDirectory(
        prefix="dinkster-base-selection-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "selected"
        staged.mkdir()
        for tag in tags:
            selection = re.fullmatch(r"base-([a-z0-9-]+)-[0-9a-f]{64}", tag)
            if selection is None:
                raise ValueError("base selection requires an exact variant/id release tag")
            endpoint = f"repos/{REPOSITORY}/releases/tags/{tag}"
            release = json.loads(subprocess.check_output(["gh", "api", endpoint]))
            if release["draft"] or not release["published_at"] or release["tag_name"] != tag:
                raise ValueError("selected base is not a published release of the requested tag")
            assets = {}
            for asset in release["assets"]:
                if asset["name"] in assets or asset["state"] != "uploaded":
                    raise ValueError("base release assets must be unique and uploaded")
                assets[asset["name"]] = asset
            if "manifest.json" not in assets or "extras.json" not in assets:
                raise ValueError("base release is missing manifest.json or extras.json")
            asset = assets["manifest.json"]
            if type(asset["id"]) is not int:
                raise ValueError("base release asset id must be numeric")
            payload = subprocess.check_output(
                [
                    "gh",
                    "api",
                    "-H",
                    "Accept: application/octet-stream",
                    f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
                ]
            )
            if len(payload) != asset["size"]:
                raise ValueError("downloaded base manifest size disagrees with release")
            path = staged / "manifest.json"
            path.write_bytes(payload)
            manifest = qualify(root, path, tag)
            asset = assets["extras.json"]
            if type(asset["id"]) is not int:
                raise ValueError("base release asset id must be numeric")
            extras_bytes = subprocess.check_output(
                [
                    "gh",
                    "api",
                    "-H",
                    "Accept: application/octet-stream",
                    f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
                ]
            )
            if len(extras_bytes) != asset["size"] or (
                asset.get("digest") is not None
                and asset["digest"] != "sha256:" + hashlib.sha256(extras_bytes).hexdigest()
            ):
                raise ValueError("downloaded extras index disagrees with release")
            extras = json.loads(extras_bytes)
            if (
                extras["format"] != "dinkster.extras/1"
                or extras["base"] != manifest["id"]
                or extras["variant"] != manifest["variant"]
                or not isinstance(extras["wheels"], list)
            ):
                raise ValueError("extras index does not match selected base")
            if not manifest["archive"]["parts"]:
                raise ValueError("planning-only base manifests cannot be selected")
            for part in manifest["archive"]["parts"]:
                asset = assets.get(part["filename"])
                if asset is None or asset["size"] != part["size"] or type(asset["id"]) is not int:
                    raise ValueError("selected base release is missing a complete archive part")
                expected = "sha256:" + part["sha256"]
                if asset.get("digest") is None:
                    command = [
                        "gh",
                        "api",
                        "-H",
                        "Accept: application/octet-stream",
                        f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
                    ]
                    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
                        assert process.stdout is not None
                        digest = hashlib.sha256()
                        size = 0
                        while block := process.stdout.read(1024 * 1024):
                            size += len(block)
                            digest.update(block)
                        if process.wait() or size != part["size"]:
                            raise ValueError("base archive part download failed or is incomplete")
                    actual = "sha256:" + digest.hexdigest()
                else:
                    actual = asset["digest"]
                if actual != expected:
                    raise ValueError("published base archive part hash disagrees with manifest")
            destination = staged / manifest["variant"]
            destination.mkdir()
            path.rename(destination / "manifest.json")
        staged.rename(output)
    return sorted(output.glob("*/manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("engine-base-manifests"))
    args = parser.parse_args()
    acquire(args.root.resolve(), args.tag, args.output.resolve())
