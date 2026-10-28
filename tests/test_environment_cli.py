"""Desktop and humans share the actual subprocess result contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_environment_artifacts import _mirror
from test_environment_state import _generation


def _run(root: Path, *arguments: str) -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(root),
            *arguments,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    document = json.loads(result.stdout)
    assert document["format"] == "dinkster.environment-result/1"
    assert document["root"] == str(root)
    assert document["plan"] == []
    return result.returncode, document


def test_catalog_uses_common_envelope_and_leaves_prospective_root_absent(tmp_path: Path):
    mirror = tmp_path / "mirror"
    base, _, channel = _mirror(mirror)
    root = tmp_path / "prospective project"
    code, result = _run(root, "catalog", "--channel", "main", "--mirror", str(mirror))
    assert code == 0
    assert result["operation"] == "catalog"
    assert result["status"] == "clean"
    assert result["channel"] == "main"
    assert result["channelManifest"] == channel
    assert result["current"] is None
    assert result["variants"][0]["manifest"] == base
    assert not root.exists()


def test_status_returns_configured_channel_and_variant_without_starting_anything(tmp_path: Path):
    root = tmp_path / "project"
    code, result = _run(root, "status")
    assert code == 0
    assert result["engines"] == []
    assert result["current"] is result["channel"] is result["variant"] is None
    assert not root.exists()
    root.mkdir()
    settings = {
        "format": "dinkster.project/1",
        "channel": "main",
        "variant": "win-cpu",
        "port": 3639,
        "mirrors": [str(tmp_path / "mirror")],
    }
    (root / "project.json").write_text(json.dumps(settings))
    before = (root / "project.json").read_bytes()
    code, result = _run(root, "status")
    assert code == 0
    assert result["channel"] == "main"
    assert result["variant"] == "win-cpu"
    assert result["engines"] == []
    assert (root / "project.json").read_bytes() == before
    assert not (root / "runtime").exists()


def test_malformed_existing_project_returns_error_not_empty_status(tmp_path: Path):
    (tmp_path / "project.json").write_text('{"format":"dinkster.project/1","channel":null}')
    code, result = _run(tmp_path, "status")
    assert code == 1
    assert result["status"] == "error"
    assert "project channel" in result["error"]


def test_doctor_reports_actual_content_drift_with_nonzero_exit_and_no_repair(tmp_path: Path):
    generation = _generation(tmp_path, 1).parent
    (tmp_path / "current").write_text("1\n")
    code, result = _run(tmp_path, "doctor")
    assert code == 0
    assert result["status"] == "clean"
    module = generation / "python/module.py"
    module.write_bytes(b"ANSWER = 13\n")
    code, result = _run(tmp_path, "doctor")
    assert code == 1
    assert result["status"] == "drift"
    assert result["changes"] == ["modified: python/module.py"]
    assert module.read_bytes() == b"ANSWER = 13\n"
    assert (tmp_path / "current").read_bytes() == b"1\n"
