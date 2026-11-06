"""Installation history survives materialization loss and cannot replace earlier entries."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_supervisor.environment_history import history, prepare_entry, publish_entry
from dinkster_supervisor.environment_state import EnvironmentError, Generation
from test_environment_cli import _run
from test_environment_state import _generation


def test_history_preserves_software_and_diffs_after_generation_is_pruned(tmp_path: Path):
    first_generation = _generation(tmp_path, 1).parent
    for name in ("workflows", "models", "outputs"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "private").write_bytes(b"user data must not enter the software snapshot")
    first = prepare_entry(tmp_path, first_generation, operation="install", reason="setup")
    publish_entry(tmp_path, first)
    initial_bytes = (tmp_path / "history/00000000000000000001.json").read_bytes()
    second_generation = _generation(tmp_path, 2).parent
    record = Generation.read(second_generation / "generation.json")
    replace(record, engine_version="0.4.2", engine_release="sha256:" + "d" * 64).write(
        second_generation / "generation.json"
    )
    second = prepare_entry(tmp_path, second_generation, operation="update", reason="update")
    publish_entry(tmp_path, second)
    shutil.rmtree(first_generation)
    code, result = _run(tmp_path, "history")
    assert code == 0
    assert result["history"] == [first, second]
    assert second["diff"] == [
        {"field": "engine.release", "before": "sha256:" + "b" * 64, "after": "sha256:" + "d" * 64},
        {"field": "engine.version", "before": "0.3.1", "after": "0.4.2"},
    ]
    code, result = _run(tmp_path, "history", "show", "1")
    assert code == 0 and result["entry"] == first
    snapshot = cast("dict[str, object]", first["snapshot"])
    assert snapshot["metadata"] == {"constraints.txt": "demo==1\n"}
    assert b"user data must not enter" not in initial_bytes
    for name in ("workflows", "models", "outputs"):
        assert (tmp_path / name / "private").read_bytes() == (
            b"user data must not enter the software snapshot"
        )
    publish_entry(tmp_path, first)
    assert (tmp_path / "history/00000000000000000001.json").read_bytes() == initial_bytes
    with pytest.raises(EnvironmentError, match="cannot replace"):
        publish_entry(tmp_path, {**first, "reason": "rewritten"})
    assert history(tmp_path) == [first, second]
    assert _run(tmp_path, "history", "show", "99")[0] == 1


def test_snapshot_refuses_modified_metadata_and_linked_pack_parents(tmp_path: Path):
    generation = _generation(tmp_path, 1).parent
    (generation / "constraints.txt").write_text("demo==2\n")
    with pytest.raises(EnvironmentError, match="intact software metadata: constraints.txt"):
        prepare_entry(tmp_path, generation, operation="install", reason="setup")
    assert history(tmp_path) == []
    assert not (tmp_path / "history").exists()
    (generation / "constraints.txt").write_text("demo==1\n")
    target = tmp_path / "user-files"
    target.mkdir()
    (target / "private").write_text("not software")
    (generation / "packs").symlink_to(target, target_is_directory=True)
    document = json.loads((generation / "generation.json").read_bytes())
    document["files"]["packs/private/pack-lock.json"] = "sha256:" + "e" * 64
    (generation / "generation.json").write_text(json.dumps(document))
    with pytest.raises(EnvironmentError, match="intact software metadata: packs/private"):
        prepare_entry(tmp_path, generation, operation="install", reason="setup")
    assert (target / "private").read_text() == "not software"


def test_restore_rebuilds_pruned_software_without_changing_user_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
            ],
            text=True,
            timeout=120,
        )
    )
    root = Path(fixture["root"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = history(root)[0]
        preserved = (root / "history/00000000000000000001.json").read_bytes()
        for name in ("workflows", "models", "outputs"):
            directory = root / name
            directory.mkdir(exist_ok=True)
            (directory / "sentinel").write_bytes(name.encode("ascii") + b" private data")
        shutil.copyfile(fixture["updatedChannel"], Path(fixture["mirror"]) / "channels/main.json")
        code, result = _run(root, "update", "--yes")
        assert code == 0 and result["current"] == 2, result
        second = (root / "history/00000000000000000002.json").read_bytes()
        shutil.rmtree(root / "generations/1")
        settings = (root / "project.json").read_bytes()
        code, result = _run(root, "restore", "1", "--plan")
        assert code == 0 and result["status"] == "planned", result
        assert len(history(root)) == 2 and not (root / "generations/3").exists()
        code, result = _run(root, "restore", "1", "--yes")
        assert code == 0 and result["current"] == 3 and result["exact"] is True, result
        entries = history(root)
        assert len(entries) == 3 and entries[-1]["restoredFrom"] == 1
        assert entries[-1]["snapshot"] == first["snapshot"]
        assert entries[-1]["recordedTools"] == cast("dict", first["snapshot"])["tools"]
        assert (root / "history/00000000000000000001.json").read_bytes() == preserved
        assert (root / "history/00000000000000000002.json").read_bytes() == second
        assert (root / "project.json").read_bytes() == settings
        monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", "refuse")
        code, failed = _run(root, "restore", "1", "--yes")
        assert code == 1 and "recorded by tools" in failed["error"], failed
        assert (root / "current").read_text() == "3\n"
        assert history(root) == entries and not (root / "generations/4").exists()
        code, queried = _run(root, "history")
        assert code == 0 and len(queried["attempts"]) == 1
        assert queried["attempts"][0]["restoredFrom"] == 1
        assert queried["attempts"][0]["error"] == failed["error"]
        monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
        for name in ("workflows", "models", "outputs"):
            assert (root / name / "sentinel").read_bytes() == name.encode(
                "ascii"
            ) + b" private data"
        assert _run(root, "doctor")[1]["status"] == "clean"
    finally:
        _run(root, "stop", "--yes")
