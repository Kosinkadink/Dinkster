"""Acquire exact frontend CI bundles or build explicitly selected checkout assets."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .environment_downloads import fetch_artifact
from .environment_local import run_local_command
from .environment_state import EnvironmentError
from .limits import ENVIRONMENT_METADATA_LIMIT_BYTES

_REPOSITORY = "Kosinkadink/Dinkster-Frontend"


def _github_json(path: str) -> dict[str, Any]:
    request = Request(
        "https://api.github.com/" + path,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "dinkster-env"},
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read(ENVIRONMENT_METADATA_LIMIT_BYTES + 1)
    except HTTPError as exc:
        hint = (
            "; GitHub may require GITHUB_TOKEN when its anonymous rate limit is exhausted"
            if exc.code in {403, 429}
            else ""
        )
        raise EnvironmentError(
            f"frontend CI metadata request failed: {request.full_url}: HTTP {exc.code}{hint}"
        ) from exc
    if len(payload) > ENVIRONMENT_METADATA_LIMIT_BYTES:
        raise EnvironmentError("frontend CI metadata exceeds eight MiB")
    return json.loads(payload)


def frontend_ci_artifact(origin: dict[str, Any]) -> dict[str, Any] | None:
    """Select an unexpired bundle from this exact commit's successful official CI."""
    if (
        origin["url"].removesuffix(".git") != "https://github.com/" + _REPOSITORY
        or origin.get("subdirectory", ".") != "."
    ):
        return None
    page = 1
    while True:
        runs = _github_json(
            f"repos/{_REPOSITORY}/actions/runs?head_sha={origin['commit']}&per_page=100&page={page}"
        )["workflow_runs"]
        for run in runs:
            if (
                run["head_sha"] != origin["commit"]
                or run["conclusion"] != "success"
                or run["path"] != ".github/workflows/ci.yml"
                or run["event"] not in {"push", "pull_request"}
                or run["head_repository"]["full_name"] != _REPOSITORY
            ):
                continue
            artifact_page = 1
            while True:
                artifacts = _github_json(
                    f"repos/{_REPOSITORY}/actions/runs/{run['id']}/artifacts"
                    f"?per_page=100&page={artifact_page}"
                )["artifacts"]
                for artifact in artifacts:
                    name = artifact["name"]
                    if not artifact["expired"] and re.fullmatch(
                        r"dinkster-frontend-[0-9a-f]{64}", name
                    ):
                        digest = artifact.get("digest", "")
                        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                            raise EnvironmentError(
                                "frontend CI artifact has no authenticated digest"
                            )
                        return {**artifact, "run": run["id"]}
                if len(artifacts) < 100:
                    break
                artifact_page += 1
        if len(runs) < 100:
            return None
        page += 1


def _bundle_files(payload: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if "index.html" not in names or len(names) != len(set(names)):
            raise EnvironmentError("frontend bundle requires a unique index.html")
        if sum(entry.file_size for entry in entries) > 256 * 1024 * 1024:
            raise EnvironmentError("frontend bundle exceeds 256 MiB unpacked")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (
                not entry.filename
                or path.is_absolute()
                or path.as_posix() != entry.filename
                or ".." in path.parts
                or "\\" in entry.orig_filename
                or ":" in entry.filename
                or entry.is_dir()
                or entry.external_attr >> 16 != 0o100644
            ):
                raise EnvironmentError(f"unsafe frontend bundle entry: {entry.filename}")
        return {entry.filename: bundle.read(entry) for entry in entries}


def prepare_frontend_wheel(
    checkout: Path, origin: dict[str, Any], output: Path, artifact: dict[str, Any] | None
) -> tuple[Path, dict[str, Any]]:
    """Package verified assets without running producer Python or requiring system Git."""
    version = json.loads((checkout / "package.json").read_text(encoding="utf-8"))["version"]
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise EnvironmentError("frontend source version must be X.Y.Z")
    api = json.loads((checkout / "frontend-release.json").read_text(encoding="utf-8"))["engineApi"]
    output.mkdir(parents=True, exist_ok=True)
    if artifact is not None:
        archive = output / "artifact.zip"
        fetch_artifact(
            (f"https://nightly.link/{_REPOSITORY}/actions/artifacts",),
            f"{artifact['id']}.zip",
            archive,
            artifact["digest"][7:],
            artifact["size_in_bytes"],
        )
        with zipfile.ZipFile(archive) as transport:
            if sorted(transport.namelist()) != ["bundle.zip", "manifest.json"]:
                raise EnvironmentError("frontend CI artifact requires bundle.zip and manifest.json")
            if sum(entry.file_size for entry in transport.infolist()) > 256 * 1024 * 1024:
                raise EnvironmentError("frontend CI artifact exceeds 256 MiB unpacked")
            manifest = json.loads(transport.read("manifest.json"))
            payload = transport.read("bundle.zip")
        digest = hashlib.sha256(payload).hexdigest()
        if (
            manifest.get("commit") != origin["commit"]
            or manifest.get("version") != version
            or manifest.get("sha256") != digest
            or manifest.get("size") != len(payload)
            or artifact["name"] != "dinkster-frontend-" + digest
        ):
            raise EnvironmentError("frontend CI bundle identity differs from selected commit")
        assets = _bundle_files(payload)
        build = {"kind": "ci", "run": artifact["run"], "artifact": artifact["id"], "sha256": digest}
    else:
        missing = [tool for tool in ("node", "pnpm") if shutil.which(tool) is None]
        if missing:
            raise EnvironmentError(
                "no frontend CI bundle for selected commit; local build requires "
                + " and ".join(missing)
            )
        run_local_command(["pnpm", "install", "--frozen-lockfile"], cwd=checkout)
        run_local_command(["pnpm", "--filter", "@dinkster/app", "build"], cwd=checkout)
        dist = checkout / "packages/app/dist"
        if not (dist / "index.html").is_file():
            raise EnvironmentError("frontend build did not produce packages/app/dist/index.html")
        paths = sorted(dist.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise EnvironmentError("frontend build must not contain symlinks")
        assets = {
            path.relative_to(dist).as_posix(): path.read_bytes() for path in paths if path.is_file()
        }
        build = {"kind": "local"}
    metadata = {
        "format": "dinkster.frontend/1",
        "repository": _REPOSITORY,
        "version": version,
        "commit": origin["commit"],
        "engineApi": api,
    }
    files = {
        "dinkster_frontend/__init__.py": b'"""Explicit Git frontend assets."""\n',
        "dinkster_frontend/manifest.json": (json.dumps(metadata, sort_keys=True) + "\n").encode(),
        **{"dinkster_frontend/static/" + name: payload for name, payload in assets.items()},
    }
    branding = checkout / "branding"
    for path in sorted(branding.rglob("*")):
        if path.is_symlink():
            raise EnvironmentError("frontend branding must not contain symlinks")
        if path.is_file():
            files["dinkster_frontend/static/branding/" + path.relative_to(branding).as_posix()] = (
                path.read_bytes()
            )
    info = f"dinkster_frontend-{version}.dist-info"
    files[info + "/METADATA"] = (
        f"Metadata-Version: 2.1\nName: dinkster-frontend\nVersion: {version}\n"
    ).encode()
    files[info + "/WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((info + "/RECORD", "", ""))
    files[info + "/RECORD"] = record.getvalue().encode()
    wheel = output / f"dinkster_frontend-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name)
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return wheel, build
