"""Pinned frontend artifacts fail closed before release packaging."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.build_release import versioned_sources
from scripts.download_frontend import REPOSITORY, verify_and_extract


def artifact(*, entry: str = "assets/app.js", **overrides: object) -> tuple[bytes, dict]:
    manifest = {
        "format": "dinkster.frontend/1",
        "repository": REPOSITORY,
        "commit": "a" * 40,
        "version": "0.1.0",
        "engineApi": {"min": 1, "max": 1},
    }
    manifest.update(overrides)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dinkster_frontend/manifest.json", json.dumps(manifest))
        for name, data in [("index.html", b'<div id="app"></div>'), (entry, b"app();")]:
            info = zipfile.ZipInfo("dinkster_frontend/static/" + name)
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data)
    data = archive.getvalue()
    pin = {
        "repository": REPOSITORY,
        "version": "0.1.0",
        "engineApi": {"min": 1, "max": 1},
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return data, pin


@pytest.mark.parametrize("version", ["0.0.2", "0.0.2.dev123+g" + "a" * 40])
def test_wheel_preserves_the_published_bundle_and_manifest(tmp_path: Path, version: str) -> None:
    payload, pin = artifact()
    published = tmp_path / "frontend.zip"
    published.write_bytes(payload)
    uv = shutil.which("uv")
    assert uv is not None, "release packaging requires uv"
    with versioned_sources(Path(__file__).resolve().parents[1], version) as staged:
        verify_and_extract(payload, pin, staged / "src/dinkster/frontend_bundle/static")
        subprocess.run(
            [uv, "build", "--package", "dinkster", "--wheel", "--out-dir", str(tmp_path)],
            cwd=staged,
            check=True,
        )
    [wheel] = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as installed, zipfile.ZipFile(io.BytesIO(payload)) as released:
        assert {
            name.removeprefix("dinkster/frontend_bundle/static/"): installed.read(name)
            for name in installed.namelist()
            if name.startswith("dinkster/frontend_bundle/static/")
        } == {
            name.removeprefix("dinkster_frontend/static/"): released.read(name)
            for name in released.namelist()
            if name.startswith("dinkster_frontend/static/")
        }
        assert json.loads(installed.read("dinkster/frontend_bundle/manifest.json")) == json.loads(
            released.read("dinkster_frontend/manifest.json")
        )
    assert wheel.name == f"dinkster-{version}-py3-none-any.whl"
