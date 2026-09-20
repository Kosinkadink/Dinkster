"""Build a backend installer archive from an immutable Git revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any


def git_archive(root: Path, revision: str, destination: Path) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "-C",
            str(root),
            "archive",
            "--format=zip",
            f"--output={destination}",
            revision,
        ],
        check=True,
    )


def worker_protocol(source: Path) -> int:
    boundary = source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    match = re.search(r"^PROTOCOL_VERSION = ([0-9]+)$", boundary.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError("cannot determine the backend worker protocol")
    return int(match[1])


def desktop_windows_runtime(source: Path) -> dict[str, Any]:
    profile = json.loads((source / "scripts/desktop_windows_runtime.json").read_text("utf-8"))
    if not isinstance(profile, dict) or set(profile) != {"aimdo", "cudaTorch"}:
        raise ValueError("Desktop Windows runtime must contain aimdo and cudaTorch objects")
    if not all(isinstance(value, dict) for value in profile.values()):
        raise ValueError("Desktop Windows runtime must contain aimdo and cudaTorch objects")
    lock = tomllib.loads((source / "uv.lock").read_text("utf-8"))
    aimdo = profile["aimdo"]
    locked_aimdo = next(entry for entry in lock["package"] if entry["name"] == "dinkster-aimdo")
    if aimdo.get("version") != locked_aimdo["version"]:
        raise ValueError("Desktop Windows runtime aimdo.version does not match uv.lock")
    aimdo_wheels = [
        wheel
        for wheel in locked_aimdo["wheels"]
        if wheel["url"].endswith(f"/{aimdo.get('archive')}")
    ]
    if len(aimdo_wheels) != 1:
        raise ValueError("Desktop Windows runtime aimdo.archive does not match uv.lock")
    aimdo_wheel = aimdo_wheels[0]
    if aimdo.get("sha256") != aimdo_wheel["hash"].removeprefix("sha256:"):
        raise ValueError("Desktop Windows runtime aimdo.sha256 does not match uv.lock")
    if aimdo.get("size") != aimdo_wheel["size"]:
        raise ValueError("Desktop Windows runtime aimdo.size does not match uv.lock")
    torch = profile["cudaTorch"]
    version = torch.get("version")
    locked = {entry["version"] for entry in lock["package"] if entry["name"] == "torch"}
    if not isinstance(version, str) or locked != {version.split("+", 1)[0]}:
        raise ValueError("Desktop Windows runtime cudaTorch.version does not match uv.lock")
    pack_path = "packages/dinkster-nodes-vision/dinkster_vision_birefnet_pack/dinkster-pack.toml"
    pack = tomllib.loads((source / pack_path).read_text("utf-8"))["pack"]
    torchvision_version = torch.get("torchvisionVersion")
    if (
        not isinstance(torchvision_version, str)
        or f"torchvision=={torchvision_version}" not in pack["requires"]
    ):
        raise ValueError(
            "Desktop Windows runtime cudaTorch.torchvisionVersion does not match "
            f"the exact requirement in {pack_path} [pack].requires"
        )
    return profile


def vendor_identity(source: Path, repository: Path, scratch: Path, uv: str) -> str:
    lock = tomllib.loads((source / "uv.lock").read_text())
    identity = next(
        package for package in lock["package"] if package["name"] == "dinkster-identity"
    )
    match = re.fullmatch(
        r"https://github.com/Kosinkadink/dinkster-identity.git\?rev=([0-9a-f]{40})#([0-9a-f]{40})",
        identity["source"]["git"],
    )
    if match is None or match[1] != match[2]:
        raise ValueError("identity dependency must name an immutable Kosinkadink source revision")
    commit = match[1]
    if not repository.exists():
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "fetch",
                "--quiet",
                "--depth=1",
                "https://github.com/Kosinkadink/dinkster-identity.git",
                commit,
            ],
            check=True,
        )
    identity_archive = scratch / "identity.zip"
    git_archive(repository, commit, identity_archive)
    with zipfile.ZipFile(identity_archive) as archive:
        archive.extractall(source / "vendor/dinkster-identity")
    server_project = source / "packages/dinkster-server/pyproject.toml"
    requirement = (
        f"dinkster-identity @ git+https://github.com/Kosinkadink/dinkster-identity.git@{commit}"
    )
    text = server_project.read_text()
    if text.count(requirement) != 1:
        raise ValueError("server identity requirement does not match the locked source")
    text = text.replace(requirement, f"dinkster-identity=={identity['version']}")
    text = text.replace(
        "[tool.uv.sources]\n",
        '[tool.uv.sources]\ndinkster-identity = { path = "../../vendor/dinkster-identity" }\n',
    )
    server_project.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run([uv, "lock", "--project", str(source)], check=True)
    released = tomllib.loads((source / "uv.lock").read_text())

    def versions(data):
        return sorted((package["name"], package["version"]) for package in data["package"])

    if versions(lock) != versions(released):
        raise ValueError("vendoring identity changed dependency versions; refusing release")
    if any("git" in package["source"] for package in released["package"]):
        raise ValueError("release lockfile still requires a Git dependency")
    return commit


def build(
    root: Path, output: Path, revision: str, *, uv: str = "uv", identity_source: Path | None = None
) -> Path:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"], text=True
    ).strip()
    output.mkdir(parents=True, exist_ok=True)
    archive = output.resolve() / f"dinkster-backend-{commit}.zip"
    with tempfile.TemporaryDirectory(prefix="dinkster-release-build-") as directory:
        scratch = Path(directory)
        source = scratch / "source"
        git_archive(root, commit, scratch / "source.zip")
        with zipfile.ZipFile(scratch / "source.zip") as original:
            original.extractall(source)
        if not (source / "scripts/install.py").is_file():
            raise ValueError("selected backend revision does not contain scripts/install.py")
        project = tomllib.loads((source / "pyproject.toml").read_text())
        requires_python = project["project"]["requires-python"]
        protocol = worker_protocol(source)
        windows_runtime = desktop_windows_runtime(source)
        identity_commit = vendor_identity(
            source, identity_source or scratch / "identity-repository", scratch, uv
        )
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as release:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(
                        f"dinkster-backend-{commit}/{path.relative_to(source).as_posix()}"
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    release.writestr(info, path.read_bytes())
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    bootstrap_python = "3.12"
    manifest = {
        "repository": "Kosinkadink/Dinkster",
        "commit": commit,
        "releaseTag": f"backend-{commit}",
        "workerProtocol": protocol,
        "requiresPython": requires_python,
        "desktopWindowsRuntime": windows_runtime,
        "bootstrap": {
            "tool": "uv",
            "python": bootstrap_python,
            "requiresInternet": True,
            "requiresGit": False,
        },
        "identityCommit": identity_commit,
        "archive": archive.name,
        "sha256": digest,
        "size": archive.stat().st_size,
        "install": f"uv run --no-project --python {bootstrap_python} scripts/install.py",
    }
    (output / "backend-release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--uv", default="uv")
    parser.add_argument(
        "--identity-source", type=Path, help="existing private identity Git repository"
    )
    args = parser.parse_args()
    print(
        build(
            Path(__file__).resolve().parent.parent,
            args.output,
            args.revision,
            uv=args.uv,
            identity_source=args.identity_source,
        )
    )
