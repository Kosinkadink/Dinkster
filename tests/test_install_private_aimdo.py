from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from packaging.tags import Tag

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "install_dinkster_aimdo.py"
SPEC = importlib.util.spec_from_file_location("install_dinkster_aimdo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)

COMMIT = "1" * 40
PLATFORMS = (
    "manylinux2014_aarch64.manylinux_2_17_aarch64",
    "manylinux2014_x86_64.manylinux_2_17_x86_64",
    "win_amd64",
    "win_arm64",
)


class FakeOpener:
    def __init__(self, responses: list[bytes | Exception]) -> None:
        self.responses = responses
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, *, timeout: int) -> io.BytesIO:
        assert timeout in (30, 120)
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(response)


def make_manifest() -> tuple[bytes, dict[str, bytes]]:
    wheels: dict[str, bytes] = {}
    entries = []
    for index, platform_tag in enumerate(PLATFORMS):
        filename = f"dinkster_aimdo-{installer.PACKAGE_VERSION}-cp39-abi3-{platform_tag}.whl"
        payload = f"wheel-{index}".encode()
        wheels[filename] = payload
        entries.append(
            {
                "abi_tag": "abi3",
                "filename": filename,
                "platform_tags": platform_tag.split("."),
                "python_tag": "cp39",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "package": "dinkster-aimdo",
        "release_tag": installer.RELEASE_TAG,
        "schema_version": 1,
        "source_commit": COMMIT,
        "source_repository": installer.REPOSITORY,
        "version": installer.PACKAGE_VERSION,
        "wheels": entries,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), wheels


def release_payload(manifest: bytes, wheels: dict[str, bytes], *, omit: str = "") -> bytes:
    assets = [
        {
            "name": installer.MANIFEST_FILENAME,
            "size": len(manifest),
            "state": "uploaded",
            "url": (f"https://api.github.com/repos/{installer.REPOSITORY}/releases/assets/100"),
        }
    ]
    for index, (filename, payload) in enumerate(wheels.items(), start=101):
        if filename == omit:
            continue
        assets.append(
            {
                "name": filename,
                "size": len(payload),
                "state": "uploaded",
                "url": (
                    f"https://api.github.com/repos/{installer.REPOSITORY}/releases/assets/{index}"
                ),
            }
        )
    return json.dumps(
        {
            "tag_name": installer.RELEASE_TAG,
            "draft": False,
            "prerelease": False,
            "assets": assets,
        }
    ).encode()


def tag_payload(commit: str = COMMIT) -> bytes:
    return json.dumps(
        {
            "ref": f"refs/tags/{installer.RELEASE_TAG}",
            "object": {"type": "commit", "sha": commit},
        }
    ).encode()


def apply_manifest_pin(monkeypatch: pytest.MonkeyPatch, manifest: bytes) -> None:
    monkeypatch.setattr(installer, "RELEASE_COMMIT", COMMIT)
    monkeypatch.setattr(installer, "MANIFEST_SIZE", len(manifest))
    monkeypatch.setattr(installer, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())


def test_release_manifest_selects_one_exact_compatible_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, wheels = make_manifest()
    apply_manifest_pin(monkeypatch, manifest)
    opener = FakeOpener([release_payload(manifest, wheels), tag_payload(), manifest])
    spec, asset = installer.resolve_wheel(
        token="private-token",
        opener=opener,
        supported_tags=[Tag("cp39", "abi3", "manylinux_2_17_x86_64")],
    )
    assert spec.filename.endswith("manylinux_2_17_x86_64.whl")
    assert spec.sha256 == hashlib.sha256(wheels[spec.filename]).hexdigest()
    assert asset.size == len(wheels[spec.filename])
    assert len(opener.requests) == 3
    assert all(
        request.get_header("Authorization") == "Bearer private-token" for request in opener.requests
    )


def test_release_rejects_missing_manifest_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, wheels = make_manifest()
    apply_manifest_pin(monkeypatch, manifest)
    missing = next(iter(wheels))
    opener = FakeOpener([release_payload(manifest, wheels, omit=missing), tag_payload(), manifest])
    with pytest.raises(RuntimeError, match="assets do not match"):
        installer.resolve_wheel(
            token="private-token",
            opener=opener,
            supported_tags=[Tag("cp39", "abi3", "win_amd64")],
        )


def test_release_rejects_incomplete_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, wheels = make_manifest()
    apply_manifest_pin(monkeypatch, manifest)
    release = json.loads(release_payload(manifest, wheels))
    release["assets"][0]["state"] = "new"
    opener = FakeOpener([json.dumps(release).encode(), tag_payload()])
    with pytest.raises(RuntimeError, match="asset metadata is invalid"):
        installer._release_assets("private-token", opener)


def test_manifest_rejects_filename_tag_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _ = make_manifest()
    raw = json.loads(manifest)
    raw["wheels"][0]["platform_tags"] = ["win_amd64"]
    invalid = (json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()
    apply_manifest_pin(monkeypatch, invalid)
    with pytest.raises(RuntimeError, match="filename does not match"):
        installer._parse_manifest(invalid)


def test_ambiguous_and_unsupported_platforms_fail_closed() -> None:
    specs = (
        installer.WheelSpec("linux.whl", "0" * 64, 1, frozenset({Tag("cp39", "abi3", "linux")})),
        installer.WheelSpec("win.whl", "1" * 64, 1, frozenset({Tag("cp39", "abi3", "win")})),
    )
    with pytest.raises(RuntimeError, match="2 compatible wheels"):
        installer.select_wheel(
            specs,
            [Tag("cp39", "abi3", "linux"), Tag("cp39", "abi3", "win")],
        )
    with pytest.raises(RuntimeError, match="0 compatible wheels"):
        installer.select_wheel(specs, [Tag("cp39", "abi3", "macos")])


def test_wheel_download_verifies_digest_before_caching(tmp_path: Path) -> None:
    payload = b"native-wheel"
    spec = installer.WheelSpec(
        "dinkster_aimdo.whl", hashlib.sha256(payload).hexdigest(), len(payload), frozenset()
    )
    asset = installer.ReleaseAsset(
        spec.filename,
        f"https://api.github.com/repos/{installer.REPOSITORY}/releases/assets/123",
        len(payload),
    )
    opener = FakeOpener([payload])
    wheel = installer.download_wheel(
        spec,
        asset,
        token="private-token",
        cache_dir=tmp_path,
        opener=opener,
    )
    assert wheel.read_bytes() == payload
    assert opener.requests[0].get_header("Authorization") == "Bearer private-token"

    bad = installer.WheelSpec(spec.filename, "0" * 64, len(payload), frozenset())
    wheel.unlink()
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        installer.download_wheel(
            bad,
            asset,
            token="private-token",
            cache_dir=tmp_path,
            opener=FakeOpener([payload]),
        )
    assert not wheel.exists()


def test_cross_origin_redirect_drops_private_token() -> None:
    handler = installer._PrivateReleaseRedirectHandler()
    request = installer._request(
        f"https://api.github.com/repos/{installer.REPOSITORY}/releases/assets/123",
        "private-token",
        accept="application/octet-stream",
    )
    redirected = handler.redirect_request(
        request,
        io.BytesIO(),
        302,
        "Found",
        http.client.HTTPMessage(),
        "https://objects.githubusercontent.com/private-wheel.whl",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_install_uses_current_interpreter_and_verifies_native_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "dinkster_aimdo.whl"
    wheel.write_bytes(b"wheel")
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("check") is True
        assert "private-token" not in kwargs.get("env", {}).values()
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    monkeypatch.setenv("DINKSTER_AIMDO_TOKEN", "private-token")
    installer.install_wheel(wheel)
    assert calls[0] == (
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        str(wheel),
    )
    assert calls[1][0:2] == (sys.executable, "-c")
    assert "aimdo_rocm.so" in calls[1][2]


def test_setup_uses_private_installer_in_both_linux_torch_envs() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_envs.sh").read_text()
    assert setup.count("install_dinkster_aimdo .venv-") == 2
    assert setup.count("scripts/install_dinkster_aimdo.py") == 2
    assert setup.index("aimdo_token=${DINKSTER_AIMDO_TOKEN") < setup.index("export -n aimdo_token")
    assert setup.index("export -n aimdo_token") < setup.index(
        "unset DINKSTER_AIMDO_TOKEN GH_TOKEN GITHUB_TOKEN"
    )
    assert "unset DINKSTER_AIMDO_TOKEN GH_TOKEN GITHUB_TOKEN" in setup
    assert '"comfy-aimdo==0.4.13"' not in setup
    assert "comfy-kitchen==0.2.32 comfy-aimdo==0.4.13" not in setup


@pytest.mark.skipif(sys.platform == "win32", reason="setup_envs.sh is a POSIX setup path")
@pytest.mark.parametrize("target", [".venv-torch", ".venv-gpu"])
def test_setup_root_sync_cannot_target_a_torch_environment(tmp_path: Path, target: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$UV_PROJECT_ENVIRONMENT" "$@" > "$SYNC_PROBE"\nexit 73\n'
    )
    uv.chmod(0o755)
    probe = tmp_path / "sync.txt"
    environment = os.environ | {
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "UV_PROJECT_ENVIRONMENT": str(REPO_ROOT / target),
        "UV_PROJECT": str(tmp_path),
        "SYNC_PROBE": str(probe),
    }
    result = subprocess.run(
        (bash, str(REPO_ROOT / "scripts" / "setup_envs.sh")),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 73, result.stdout + result.stderr
    assert probe.read_text().splitlines() == [
        str(REPO_ROOT / ".venv"),
        "sync",
        "--project",
        str(REPO_ROOT),
        "--all-packages",
    ]
