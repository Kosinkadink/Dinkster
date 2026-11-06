"""Installation history survives materialization loss and cannot replace earlier entries."""

from __future__ import annotations

import json
import shutil
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
