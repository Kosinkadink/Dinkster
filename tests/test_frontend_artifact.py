"""Pinned frontend artifacts fail closed before release packaging."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_frontend import REPOSITORY, download, verify_and_extract


def artifact(*, entry: str = "assets/app.js", **overrides: object) -> tuple[bytes, dict]:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in [("index.html", b'<div id="app"></div>'), (entry, b"app();")]:
            info = zipfile.ZipInfo(name)
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
