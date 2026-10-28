"""Project pointers and generation drift are shared with non-CLI clients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from dinkster_supervisor.environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    read_current,
    set_current,
)


def _generation(root: Path, number: int) -> Path:
    directory = root / "generations" / str(number)
    directory.mkdir(parents=True)
    (directory / "constraints.txt").write_bytes(b"demo==1\n")
    (directory / "python").mkdir()
    (directory / "python" / "module.py").write_bytes(b"ANSWER = 42\n")
    metadata = {
        "format": "dinkster.generation/1",
        "generation": number,
        "base": "sha256:" + "a" * 64,
        "engine": {"version": "0.3.1", "release": "sha256:" + "b" * 64},
        "constraints": "sha256:" + hashlib.sha256(b"demo==1\n").hexdigest(),
        "variant": "linux-cpu",
        "workerProtocol": 1,
        "files": {
            "constraints.txt": "sha256:" + hashlib.sha256(b"demo==1\n").hexdigest(),
            "python/module.py": "sha256:" + hashlib.sha256(b"ANSWER = 42\n").hexdigest(),
        },
    }
    path = directory / "generation.json"
    path.write_text(json.dumps(metadata))
    return path


def test_pointer_requires_a_complete_generation_and_preserves_previous_on_failure(tmp_path: Path):
    assert read_current(tmp_path) is None
    first = _generation(tmp_path, 1)
    set_current(tmp_path, 1)
    assert (tmp_path / "current").read_bytes() == b"1\n"
    assert read_current(tmp_path) == 1
    with pytest.raises(EnvironmentError, match="cannot read generation.json"):
        set_current(tmp_path, 2)
    assert read_current(tmp_path) == 1
    second = _generation(tmp_path, 2)
    second.write_text(first.read_text())
    with pytest.raises(EnvironmentError, match="number disagree"):
        set_current(tmp_path, 2)
    assert read_current(tmp_path) == 1


def test_doctor_detects_edits_missing_and_added_files_without_reporting_pack_files(tmp_path: Path):
    path = _generation(tmp_path, 7)
    root = path.parent
    generation = Generation.read(path)
    assert generation.drift(root) == ()
    (root / "packs" / "example").mkdir(parents=True)
    (root / "packs" / "example" / "pyvenv.cfg").write_text("home = python\n")
    (root / "python" / "__pycache__").mkdir()
    (root / "python" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    assert generation.drift(root) == ()
    (root / "python" / "module.py").write_bytes(b"ANSWER = 99\n")
    (root / "constraints.txt").unlink()
    (root / "python" / "unexpected.py").write_bytes(b"injected")
    assert generation.drift(root) == (
        "missing: constraints.txt",
        "modified: python/module.py",
        "unexpected: python/unexpected.py",
    )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "C:/escape", "a\\b"])
def test_generation_refuses_file_hashes_outside_its_tree(tmp_path: Path, name: str):
    path = _generation(tmp_path, 1)
    metadata = json.loads(path.read_text())
    metadata["files"][name] = "sha256:" + "c" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(EnvironmentError, match="relative file path"):
        Generation.read(path)


def test_journal_records_recovery_state_and_retains_failure_diagnostics(tmp_path: Path):
    path = tmp_path / "journal" / "update.json"
    Journal("update", 2, 3, "switching").write(path)
    interrupted = Journal.read(path)
    assert interrupted.incomplete
    assert (interrupted.previous, interrupted.target) == (2, 3)
    Journal("update", 2, 3, "rolled-back", "target failed health check").write(path)
    recovered = Journal.read(path)
    assert not recovered.incomplete
    assert recovered.error == "target failed health check"
    assert json.loads(path.read_text())["format"] == "dinkster.journal/1"
