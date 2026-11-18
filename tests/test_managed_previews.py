"""Managed preview placement preserves originals and earlier preview locations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_environment_runtime import free_port


def test_preview_storage_changes_keep_previous_locations_and_originals(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = project / "environment.json"
    settings.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": free_port(),
            }
        )
    )
    original = project / "dinkster-library" / "outputs" / "original.txt"
    original.parent.mkdir(parents=True)
    original.write_text("retained original")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    sibling = first / "unrelated.txt"
    sibling.write_text("not a preview")
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(project),
        "previews",
    ]

    def configure(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([*command, *arguments, "--json"], capture_output=True, text=True)

    before = settings.read_bytes()
    plan = configure("--directory", str(first), "--plan")
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert settings.read_bytes() == before and list(first.iterdir()) == [sibling]
    applied = configure("--directory", str(first), "--yes")
    assert applied.returncode == 0, applied.stdout + applied.stderr
    vault = Path(json.loads(applied.stdout)["previewRoot"])
    assert vault.is_dir()
    vault.rmdir()
    missing_vault = configure("--directory", str(first), "--yes")
    assert missing_vault.returncode != 0 and "vault is unavailable" in missing_vault.stdout
    assert not vault.exists()
    vault.mkdir()
    saved = vault / "kept-preview"
    saved.write_bytes(b"earlier preview bytes")
    unavailable = configure("--directory", str(tmp_path / "missing"), "--yes")
    assert unavailable.returncode != 0 and "unavailable" in unavailable.stdout
    assert json.loads(settings.read_text())["previewStorage"] == {"directory": str(first)}
    changed = configure("--directory", str(second), "--yes")
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "directory": str(second),
        "previousDirectories": [str(first)],
    }
    cleared = configure("--yes")
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "previousDirectories": [str(first), str(second)]
    }
    restored = configure("--directory", str(first), "--yes")
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "directory": str(first),
        "previousDirectories": [str(second)],
    }
    assert saved.read_bytes() == b"earlier preview bytes"
    assert original.read_text() == "retained original"
    assert sibling.read_text() == "not a preview"
    before = settings.read_bytes()
    owner = vault.parent / "owner.json"
    owner.write_text("{}")
    refused = configure("--yes")
    assert refused.returncode != 0 and "not owned" in refused.stdout
    assert settings.read_bytes() == before
