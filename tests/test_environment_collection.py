"""Confirmed collection preserves journal ancestry and recovers interrupted quarantine."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_supervisor.environment_state import (
    Generation,
    Journal,
    content_digest,
    write_document,
)
from test_environment_cli import _run
from test_environment_state import _generation


@pytest.mark.parametrize(
    "interruption", [None, "quarantine", "cleanup", "missing-project", "corrupt-constraints"]
)
def test_shared_store_collection_keeps_all_registered_references_and_recovers_from_another_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str | None
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user-data"))
    store = tmp_path / "store"
    first, second = tmp_path / "first", tmp_path / "second"
    references = (
        (first, 1, "a", "b", "e"),
        (first, 2, "2", "3", "1"),
        (first, 3, "a", "b", "f"),
        (second, 1, "c", "d", "9"),
        (second, 5, "c", "d", "8"),
    )
    for project, number, base, engine, wheel in references:
        record = _generation(project, number)
        tree = record.parent
        (tree / "constraints.txt").write_text(f"demo==1 --hash=sha256:{wheel * 64}\n")
        generation = Generation.read(record)
        files = dict(generation.files)
        files["constraints.txt"] = content_digest(tree / "constraints.txt")
        replace(
            generation,
            base="sha256:" + base * 64,
            engine_release="sha256:" + engine * 64,
            constraints=files["constraints.txt"],
            files=tuple(sorted(files.items())),
        ).write(record)
        for category, identity in (
            ("bases", base),
            ("engines", engine),
            ("downloads", base),
            ("wheels", wheel),
        ):
            directory = store / category / (identity * 64)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "payload").write_bytes(identity.encode())
    for project, current in ((first, 3), (second, 1)):
        (project / "current").write_text(f"{current}\n")
        (project / "store").write_text(str(store) + "\n")
        write_document(
            project / "project.json",
            {
                "format": "dinkster.project/1",
                "channel": "main",
                "variant": "linux-cpu",
                "mirrors": [str(tmp_path / "mirror")],
                "port": 3639,
            },
        )
        (project / "library").mkdir()
        (project / "library/keep").write_bytes(b"project asset")
    Journal("update", 1, 3, "complete").write(first / "journal/100.json")
    write_document(
        store / "projects.json",
        {"format": "dinkster.store/1", "projects": [str(first), str(second)]},
    )
    for category in ("wheels", "bases", "engines", "downloads"):
        extra = store / category / ("7" * 64)
        extra.mkdir()
        (extra / "payload").write_bytes(b"unreferenced")
    (store / "uv-cache").mkdir()
    (store / "uv-cache/entry").write_bytes(b"disposable extracted cache")
    if interruption == "missing-project":
        second.rename(tmp_path / "unavailable-project")
    elif interruption == "corrupt-constraints":
        (second / "generations/5/constraints.txt").write_bytes(b"changed recorded constraints")
    before = {p: content_digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    code, planned = _run(first, "gc", "--plan")
    if interruption in {"missing-project", "corrupt-constraints"}:
        assert code == 1 and planned["status"] == "error", planned
        assert (
            "registered project"
            if interruption == "missing-project"
            else "intact retained constraints"
        ) in planned["error"]
        assert {p: content_digest(p) for p in tmp_path.rglob("*") if p.is_file()} == before
        assert not (first / ".lock").exists() and not (store / ".lock").exists()
        return
    assert code == 0 and planned["status"] == "planned", planned
    assert {p: content_digest(p) for p in tmp_path.rglob("*") if p.is_file()} == before
    if interruption is not None:
        script = """
import os, sys
from pathlib import Path
from dinkster_supervisor import environment_collection as c, environment
store = Path(sys.argv[2])
if sys.argv[3] == 'quarantine':
    original = c.durable_rename
    def stop(source, target):
        original(source, target)
        if source.is_relative_to(store): os._exit(73)
    c.durable_rename = stop
else:
    original = c.shutil.rmtree
    def stop(path, *args, **kwargs):
        if path.is_relative_to(store): os._exit(73)
        return original(path, *args, **kwargs)
    c.shutil.rmtree = stop
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'gc', '--yes', '--json']
environment.main()
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(first), str(store), interruption], timeout=30
        )
        assert child.returncode == 73
        journal = next(
            p for p in (first / "journal").glob("*.json") if Journal.read(p).operation == "gc"
        )
        assert (store / "collection").read_text().strip() == str(journal)
        code, stopped = _run(second, "stop", "--yes")
        assert code == 0, stopped
        assert not (store / "collection").exists()
        assert Journal.read(journal).stage == (
            "rolled-back" if interruption == "quarantine" else "complete"
        )
        if interruption == "quarantine":
            for category in ("wheels", "bases", "engines", "downloads"):
                assert (store / category / ("7" * 64) / "payload").read_bytes() == b"unreferenced"
            assert (first / "generations/2").is_dir()
    code, applied = _run(first, "gc", "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    assert not (first / "generations/2").exists()
    for category in ("wheels", "bases", "engines", "downloads"):
        assert not (store / category / ("7" * 64)).exists()
    for project, number, base, engine, wheel in references:
        if project == first and number == 2:
            continue
        for category, identity in (("bases", base), ("engines", engine), ("wheels", wheel)):
            assert (
                store / category / (identity * 64) / "payload"
            ).read_bytes() == identity.encode()
        assert (
            Generation.read(project / "generations" / str(number) / "generation.json").drift(
                project / "generations" / str(number)
            )
            == ()
        )
    assert (second / "generations/5").is_dir()
    assert not (store / "uv-cache").exists()
    assert all((p / "library/keep").read_bytes() == b"project asset" for p in (first, second))


@pytest.mark.parametrize("interruption", [None, "quarantine", "cleanup"])
def test_collection_keeps_actual_predecessor_and_recovers_before_other_mutations(
    tmp_path: Path, interruption: str | None
):
    for number in (1, 2, 3, 7):
        _generation(tmp_path, number)
    (tmp_path / "current").write_bytes(b"3\n")
    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "main",
                "variant": "linux-cpu",
                "mirrors": [str(tmp_path / "mirror")],
                "port": 3639,
            }
        )
    )
    Journal("update", 1, 3, "complete").write(tmp_path / "journal/100.json")
    Journal("update", 3, 7, "rolled-back").write(tmp_path / "journal/200.json")
    (tmp_path / "library").mkdir()
    (tmp_path / "library/keep.bin").write_bytes(b"never collect project assets")
    retained = {
        p.relative_to(tmp_path): content_digest(p)
        for directory in (
            tmp_path / "generations/1",
            tmp_path / "generations/3",
            tmp_path / "library",
        )
        for p in directory.rglob("*")
        if p.is_file()
    }
    before = {
        p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    for flags in ((), ("--plan",)):
        code, planned = _run(tmp_path, "gc", *flags)
        assert code == 0 and planned["status"] == "planned", planned
        assert "retain generations 1, 3" in planned["plan"]
        assert "collect obsolete generations 2, 7" in planned["plan"]
        assert {
            p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
        } == before
    if interruption is not None:
        script = """
import os, sys
from pathlib import Path
from dinkster_supervisor import environment_collection as collection
from dinkster_supervisor import environment
if sys.argv[2] == 'quarantine':
    original = collection.durable_rename
    def interrupt(source, destination):
        original(source, destination)
        os._exit(73)
    collection.durable_rename = interrupt
else:
    collection.shutil.rmtree = lambda *a, **kw: os._exit(73)
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'gc', '--yes', '--json']
environment.main()
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), interruption], timeout=30
        )
        assert child.returncode == 73
        journal_path = next(
            p for p in (tmp_path / "journal").glob("*.json") if Journal.read(p).operation == "gc"
        )
        journal = Journal.read(journal_path)
        assert journal.stage == ("building" if interruption == "quarantine" else "complete")
        assert (tmp_path / ".garbage" / journal_path.stem / "2").is_dir()
        if interruption == "quarantine":
            interactive = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdin.isatty = lambda: True; "
                    "from dinkster_supervisor.environment import main; main()",
                    "--root",
                    str(tmp_path),
                    "gc",
                ],
                input="y\n",
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert interactive.returncode != 0
            assert "changed" in interactive.stdout + interactive.stderr
            assert Journal.read(journal_path).stage == "rolled-back"
            assert (tmp_path / "generations/2").is_dir()
            assert (tmp_path / "generations/7").is_dir()
        # Confirmed stop first recovers pending operations without starting a process.
        code, stopped = _run(tmp_path, "stop", "--yes")
        assert code == 0, stopped
        if interruption == "quarantine":
            assert Journal.read(journal_path).stage == "rolled-back"
            assert (tmp_path / "generations/2").is_dir()
            assert (tmp_path / "generations/7").is_dir()
        else:
            assert not list((tmp_path / ".garbage").rglob("generation.json"))
    code, collected = _run(tmp_path, "gc", "--yes")
    assert code == 0 and collected["status"] == "applied", collected
    assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["1", "3"]
    assert (tmp_path / "current").read_bytes() == b"3\n"
    assert _run(tmp_path, "doctor")[1]["status"] == "clean"
    for name, digest in retained.items():
        assert content_digest(tmp_path / name) == digest
    code, rollback = _run(tmp_path, "rollback", "--plan")
    assert code == 0 and "reactivate retained generation 1" in rollback["plan"][0], rollback


@pytest.mark.parametrize("numbers", [[1], [2, 2], [0], [True], ["2"]])
def test_collection_journal_refuses_unsafe_generation_sets(tmp_path: Path, numbers: list):
    path = tmp_path / "100.json"
    path.write_text(
        json.dumps(
            {
                "format": "dinkster.journal/1",
                "operation": "gc",
                "previous": 1,
                "target": 1,
                "stage": "building",
                "collectedGenerations": numbers,
            }
        )
    )
    with pytest.raises(ValueError):
        Journal.read(path)
