"""Pinned frontend artifacts fail closed before release packaging."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.build_release import build_frontend_wheel
from scripts.download_frontend import REPOSITORY, download, verify_and_extract


def artifact(*, entry: str = "assets/app.js", **overrides: object) -> tuple[bytes, dict]:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in [("index.html", b'<div id="app"></div>'), (entry, b"app();")]:
            info = zipfile.ZipInfo(name)
            # Preserve unsafe input that ZipInfo otherwise normalizes on Windows.
            info.filename = name
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data)
    data = archive.getvalue()
    pin = {"repository": REPOSITORY, "commit": "a" * 40, "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "commit": pin["commit"],
        "sha256": pin["sha256"],
        "size": len(data),
        "version": "0.0.1",
    }
    manifest.update(overrides)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as wrapper:
        wrapper.writestr("manifest.json", json.dumps(manifest))
        wrapper.writestr("bundle.zip", data)
    return payload.getvalue(), pin


def test_extract_preserves_the_published_files(tmp_path: Path) -> None:
    payload, pin = artifact()
    verify_and_extract(payload, pin, tmp_path / "bundle")
    assert (tmp_path / "bundle/index.html").read_bytes() == b'<div id="app"></div>'
    assert (tmp_path / "bundle/assets/app.js").read_bytes() == b"app();"
    assert json.loads((tmp_path / "manifest.json").read_text())["sha256"] == pin["sha256"]


@pytest.mark.parametrize("version", ["0.0.2", "0.0.2.dev123+g" + "a" * 40])
def test_wheel_preserves_the_published_bundle_and_manifest(tmp_path: Path, version: str) -> None:
    payload, pin = artifact()
    published = tmp_path / "frontend.zip"
    published.write_bytes(payload)
    uv = shutil.which("uv")
    assert uv is not None, "release packaging requires uv"
    wheel = build_frontend_wheel(published, version, pin, tmp_path, uv)
    with zipfile.ZipFile(wheel) as installed, zipfile.ZipFile(io.BytesIO(payload)) as wrapper:
        with zipfile.ZipFile(io.BytesIO(wrapper.read("bundle.zip"))) as bundle:
            assert {
                name.removeprefix("dinkster_frontend/static/"): installed.read(name)
                for name in installed.namelist()
                if name.startswith("dinkster_frontend/static/")
            } == {name: bundle.read(name) for name in bundle.namelist()}
        assert json.loads(installed.read("dinkster_frontend/manifest.json")) == json.loads(
            wrapper.read("manifest.json")
        )
    assert wheel.name == f"dinkster_frontend-{version}-py3-none-any.whl"


@pytest.mark.parametrize(
    "overrides",
    [
        {"commit": "b" * 40},
        {"sha256": "0" * 64},
        {"size": 1},
    ],
)
def test_manifest_mismatch_creates_no_bundle(tmp_path: Path, overrides: dict) -> None:
    payload, pin = artifact(**overrides)
    with pytest.raises(ValueError, match="does not match"):
        verify_and_extract(payload, pin, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_pinned_hash_is_checked_independently_of_manifest(tmp_path: Path) -> None:
    payload, pin = artifact(sha256="0" * 64)
    pin["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash or size"):
        verify_and_extract(payload, pin, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("entry", ["../escape.js", "/absolute.js", "assets\\escape.js"])
def test_archive_paths_cannot_escape_the_bundle(tmp_path: Path, entry: str) -> None:
    payload, pin = artifact(entry=entry)
    with zipfile.ZipFile(io.BytesIO(payload)) as wrapper:
        with zipfile.ZipFile(io.BytesIO(wrapper.read("bundle.zip"))) as bundle:
            assert bundle.infolist()[1].orig_filename == entry
    with pytest.raises(ValueError, match="unsafe frontend bundle entry"):
        verify_and_extract(payload, pin, tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_missing_commit_names_the_required_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pin = artifact()
    monkeypatch.setattr("scripts.download_frontend.github_json", lambda endpoint: {"artifacts": []})
    with pytest.raises(ValueError) as error:
        download(pin, tmp_path / "bundle")
    assert f"Missing published frontend artifact dinkster-frontend-{pin['sha256']}" in str(
        error.value
    )
    assert f"commit {pin['commit']}" in str(error.value)
    assert not (tmp_path / "bundle").exists()


def test_download_selects_only_the_pinned_successful_frontend_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, pin = artifact()
    candidates = [
        {
            "id": number,
            "expired": number == 1,
            "workflow_run": {"id": number, "head_sha": "b" * 40 if number == 2 else pin["commit"]},
        }
        for number in range(1, 8)
    ]

    def github(arguments: list[str]) -> bytes:
        endpoint = arguments[2]
        if "/artifacts?" in endpoint:
            return json.dumps({"artifacts": candidates}).encode()
        if "/runs/" in endpoint:
            number = int(endpoint.rsplit("/", 1)[1])
            return json.dumps(
                {
                    "id": number,
                    "conclusion": "failure" if number == 3 else "success",
                    "path": "unrelated.yml" if number == 4 else ".github/workflows/ci.yml",
                    "event": "workflow_dispatch" if number == 5 else "push",
                    "head_repository": {
                        "full_name": "other/frontend" if number == 6 else REPOSITORY
                    },
                }
            ).encode()
        if endpoint.endswith("/artifacts/7/zip"):
            return payload
        raise AssertionError(f"attempted to download an ineligible artifact: {endpoint}")

    monkeypatch.setattr("scripts.download_frontend.subprocess.check_output", github)
    output = tmp_path / "frontend.zip"
    download(pin, output)
    assert output.read_bytes() == payload
