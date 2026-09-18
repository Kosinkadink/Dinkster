#!/usr/bin/env python3
"""Install the pinned native dinkster-aimdo wheel from its private release."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.parse import urlsplit

from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version

REPOSITORY = "Kosinkadink/dinkster-aimdo"
RELEASE_TAG = "v0.5.5.post1"
RELEASE_COMMIT = "6ae5b951e333a52dba825487587d674bd0be410e"
PACKAGE_VERSION = "0.5.5.post1"
MANIFEST_FILENAME = "dinkster_aimdo-0.5.5.post1-manifest.json"
MANIFEST_SHA256 = "48eeac65ae42925e98eacb6b6368d531faab99bb40db4f8b8b5a482568396fbc"
MANIFEST_SIZE = 1598

EXPECTED_PLATFORM_GROUPS = {
    frozenset({"manylinux2014_aarch64", "manylinux_2_17_aarch64"}),
    frozenset({"manylinux2014_x86_64", "manylinux_2_17_x86_64"}),
    frozenset({"win_amd64"}),
    frozenset({"win_arm64"}),
}
MANIFEST_KEYS = {
    "package",
    "release_tag",
    "schema_version",
    "source_commit",
    "source_repository",
    "version",
    "wheels",
}
WHEEL_KEYS = {
    "abi_tag",
    "filename",
    "platform_tags",
    "python_tag",
    "sha256",
    "size",
}


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int


@dataclass(frozen=True)
class WheelSpec:
    filename: str
    sha256: str
    size: int
    tags: frozenset[Tag]


class PrivateReleaseUnavailableError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


class _PrivateReleaseRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            redirected.remove_header("Authorization")
        return redirected


def _request(url: str, token: str, *, accept: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "User-Agent": "Dinkster-aimdo-installer",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def _api_json(
    url: str,
    token: str,
    opener: urllib.request.OpenerDirector,
) -> object:
    try:
        with opener.open(
            _request(url, token, accept="application/vnd.github+json"), timeout=30
        ) as response:
            body = response.read(1024 * 1024 + 1)
    except (OSError, http.client.HTTPException) as error:
        raise PrivateReleaseUnavailableError(
            "private dinkster-aimdo release is unavailable"
        ) from error
    if len(body) > 1024 * 1024:
        raise RuntimeError("private dinkster-aimdo release metadata exceeds the size limit")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("private dinkster-aimdo release metadata is invalid") from error


def _tag_commit(token: str, opener: urllib.request.OpenerDirector) -> str:
    url = f"https://api.github.com/repos/{REPOSITORY}/git/ref/tags/{RELEASE_TAG}"
    payload = _api_json(url, token, opener)
    if not isinstance(payload, dict) or payload.get("ref") != f"refs/tags/{RELEASE_TAG}":
        raise RuntimeError("private dinkster-aimdo release tag metadata is invalid")
    git_object = payload.get("object")
    for _ in range(5):
        if not isinstance(git_object, dict):
            break
        object_type = git_object.get("type")
        sha = git_object.get("sha")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            break
        if object_type == "commit":
            return sha
        if object_type != "tag":
            break
        payload = _api_json(
            f"https://api.github.com/repos/{REPOSITORY}/git/tags/{sha}", token, opener
        )
        if not isinstance(payload, dict):
            break
        git_object = payload.get("object")
    raise RuntimeError("private dinkster-aimdo release tag target is invalid")


def _asset_url_is_valid(url: str) -> bool:
    parsed = urlsplit(url)
    prefix = f"/repos/{REPOSITORY}/releases/assets/"
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.github.com"
        and parsed.port is None
        and parsed.query == ""
        and parsed.fragment == ""
        and parsed.path.startswith(prefix)
        and parsed.path[len(prefix) :].isdigit()
    )


def _release_assets(token: str, opener: urllib.request.OpenerDirector) -> dict[str, ReleaseAsset]:
    payload = _api_json(
        f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}", token, opener
    )
    if (
        not isinstance(payload, dict)
        or payload.get("tag_name") != RELEASE_TAG
        or payload.get("draft") is not False
        or payload.get("prerelease") is not False
    ):
        raise RuntimeError("private dinkster-aimdo release metadata does not match the pin")
    if _tag_commit(token, opener) != RELEASE_COMMIT:
        raise RuntimeError("private dinkster-aimdo release tag does not target the pinned commit")
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise RuntimeError("private dinkster-aimdo release has no asset list")

    assets: dict[str, ReleaseAsset] = {}
    for raw in raw_assets:
        if not isinstance(raw, dict):
            raise RuntimeError("private dinkster-aimdo release asset metadata is invalid")
        name = raw.get("name")
        url = raw.get("url")
        size = raw.get("size")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or not isinstance(url, str)
            or not _asset_url_is_valid(url)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or raw.get("state") != "uploaded"
            or name in assets
        ):
            raise RuntimeError("private dinkster-aimdo release asset metadata is invalid")
        assets[name] = ReleaseAsset(name, url, size)
    return assets


def _download_bytes(
    asset: ReleaseAsset,
    *,
    token: str,
    opener: urllib.request.OpenerDirector,
) -> bytes:
    try:
        response = opener.open(
            _request(asset.url, token, accept="application/octet-stream"), timeout=120
        )
    except (OSError, http.client.HTTPException) as error:
        raise PrivateReleaseUnavailableError(
            "private dinkster-aimdo release asset is unavailable"
        ) from error
    with response:
        try:
            payload = response.read(asset.size + 1)
        except (OSError, http.client.HTTPException) as error:
            raise PrivateReleaseUnavailableError(
                "private dinkster-aimdo release asset is unavailable"
            ) from error
    if len(payload) != asset.size:
        raise RuntimeError(
            f"private dinkster-aimdo asset size mismatch: expected {asset.size}, got {len(payload)}"
        )
    return payload


def _parse_manifest(payload: bytes) -> tuple[WheelSpec, ...]:
    if len(payload) != MANIFEST_SIZE or sha256_bytes(payload) != MANIFEST_SHA256:
        raise RuntimeError("private dinkster-aimdo manifest does not match its hash pin")
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("private dinkster-aimdo manifest is invalid") from error
    if not isinstance(raw, dict) or set(raw) != MANIFEST_KEYS:
        raise RuntimeError("private dinkster-aimdo manifest schema is invalid")
    expected_identity = {
        "package": "dinkster-aimdo",
        "release_tag": RELEASE_TAG,
        "schema_version": 1,
        "source_commit": RELEASE_COMMIT,
        "source_repository": REPOSITORY,
        "version": PACKAGE_VERSION,
    }
    if any(raw.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("private dinkster-aimdo manifest identity does not match the pin")
    raw_wheels = raw.get("wheels")
    if not isinstance(raw_wheels, list) or len(raw_wheels) != 4:
        raise RuntimeError("private dinkster-aimdo manifest wheel matrix is invalid")

    specs: list[WheelSpec] = []
    platform_groups: set[frozenset[str]] = set()
    filenames: set[str] = set()
    for raw_wheel in raw_wheels:
        if not isinstance(raw_wheel, dict) or set(raw_wheel) != WHEEL_KEYS:
            raise RuntimeError("private dinkster-aimdo manifest wheel entry is invalid")
        filename = raw_wheel.get("filename")
        digest = raw_wheel.get("sha256")
        size = raw_wheel.get("size")
        python_tag = raw_wheel.get("python_tag")
        abi_tag = raw_wheel.get("abi_tag")
        platform_tags = raw_wheel.get("platform_tags")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or filename in filenames
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or python_tag != "cp39"
            or abi_tag != "abi3"
            or not isinstance(platform_tags, list)
            or not platform_tags
            or not all(isinstance(tag, str) and tag for tag in platform_tags)
            or len(platform_tags) != len(set(platform_tags))
        ):
            raise RuntimeError("private dinkster-aimdo manifest wheel entry is invalid")
        platform_group = frozenset(platform_tags)
        if platform_group not in EXPECTED_PLATFORM_GROUPS or platform_group in platform_groups:
            raise RuntimeError("private dinkster-aimdo manifest wheel platform is invalid")
        try:
            name, version, build, filename_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename as error:
            raise RuntimeError(
                "private dinkster-aimdo manifest wheel filename is invalid"
            ) from error
        declared_tags = frozenset(Tag(python_tag, abi_tag, tag) for tag in platform_tags)
        if (
            canonicalize_name(name) != "dinkster-aimdo"
            or version != Version(PACKAGE_VERSION)
            or build != ()
            or filename_tags != declared_tags
        ):
            raise RuntimeError(
                "private dinkster-aimdo manifest wheel filename does not match its tags"
            )
        filenames.add(filename)
        platform_groups.add(platform_group)
        specs.append(WheelSpec(filename, digest, size, filename_tags))

    if platform_groups != EXPECTED_PLATFORM_GROUPS:
        raise RuntimeError("private dinkster-aimdo manifest wheel matrix is incomplete")
    return tuple(specs)


def select_wheel(specs: Iterable[WheelSpec], supported_tags: Iterable[Tag]) -> WheelSpec:
    supported = frozenset(supported_tags)
    matches = [spec for spec in specs if spec.tags.intersection(supported)]
    if len(matches) != 1:
        raise RuntimeError(
            "private dinkster-aimdo release has "
            f"{len(matches)} compatible wheels; expected exactly one"
        )
    return matches[0]


def resolve_wheel(
    *,
    token: str,
    opener: urllib.request.OpenerDirector | None = None,
    supported_tags: Iterable[Tag] | None = None,
) -> tuple[WheelSpec, ReleaseAsset]:
    active_opener = opener or urllib.request.build_opener(_PrivateReleaseRedirectHandler())
    assets = _release_assets(token, active_opener)
    manifest_asset = assets.get(MANIFEST_FILENAME)
    if manifest_asset is None or manifest_asset.size != MANIFEST_SIZE:
        raise RuntimeError("private dinkster-aimdo release manifest asset does not match the pin")
    specs = _parse_manifest(_download_bytes(manifest_asset, token=token, opener=active_opener))
    if set(assets) != {MANIFEST_FILENAME, *(spec.filename for spec in specs)}:
        raise RuntimeError("private dinkster-aimdo release assets do not match the manifest")
    for spec in specs:
        if assets[spec.filename].size != spec.size:
            raise RuntimeError("private dinkster-aimdo wheel size does not match the manifest")
    spec = select_wheel(specs, supported_tags if supported_tags is not None else sys_tags())
    return spec, assets[spec.filename]


def download_wheel(
    spec: WheelSpec,
    asset: ReleaseAsset,
    *,
    token: str,
    cache_dir: Path,
    opener: urllib.request.OpenerDirector | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / spec.filename
    if (
        destination.is_file()
        and destination.stat().st_size == spec.size
        and sha256_file(destination) == spec.sha256
    ):
        return destination

    active_opener = opener or urllib.request.build_opener(_PrivateReleaseRedirectHandler())
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            payload = _download_bytes(asset, token=token, opener=active_opener)
            output.write(payload)
        if sha256_file(temporary) != spec.sha256:
            raise RuntimeError("private dinkster-aimdo wheel SHA-256 mismatch")
        temporary.replace(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def github_token() -> str | None:
    for name in ("DINKSTER_AIMDO_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    gh = shutil.which("gh")
    if gh is None:
        return None
    result = subprocess.run((gh, "auth", "token"), check=False, capture_output=True, text=True)
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("DINKSTER_AIMDO_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        environment.pop(name, None)
    return environment


def install_wheel(path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install private dinkster-aimdo")
    subprocess.run(
        (
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--reinstall",
            "--no-deps",
            str(path),
        ),
        check=True,
        env=_subprocess_environment(),
    )
    verification = f"""
import sys
from importlib.metadata import distribution, version
from pathlib import Path

assert version('dinkster-aimdo') == {PACKAGE_VERSION!r}
files = {{Path(str(path)).name for path in distribution('dinkster-aimdo').files or ()}}
if sys.platform == 'linux':
    expected = {{'aimdo.so', 'aimdo_rocm.so'}}
else:
    expected = {{'aimdo.dll', 'aimdo_rocm.dll'}}
assert expected <= files
"""
    subprocess.run(
        (sys.executable, "-c", verification),
        check=True,
        env=_subprocess_environment(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "dinkster" / "dinkster-aimdo" / RELEASE_TAG,
    )
    args = parser.parse_args(argv)
    try:
        token = github_token()
        if token is None:
            raise PrivateReleaseUnavailableError(
                "private dinkster-aimdo access unavailable; provide DINKSTER_AIMDO_TOKEN, "
                "GH_TOKEN, GITHUB_TOKEN, or an authenticated gh CLI"
            )
        spec, asset = resolve_wheel(token=token)
        wheel = download_wheel(
            spec,
            asset,
            token=token,
            cache_dir=args.cache_dir.expanduser(),
        )
        print(f"==> private dinkster-aimdo wheel: {wheel}")
        install_wheel(wheel)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
