"""Installation history survives materialization loss and cannot replace earlier entries."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_supervisor.environment_history import history, prepare_entry, publish_entry
from dinkster_supervisor.environment_runtime import engine_status
from dinkster_supervisor.environment_state import EnvironmentError, Generation
from test_environment_cli import _run
from test_environment_runtime import free_port
from test_environment_state import _generation

from tests.fixtures.environments.build import canonical_json, executable_release


def test_missing_base_offers_only_compatible_substitution_without_upgrading_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
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
    root, store, mirror = Path(fixture["root"]), Path(fixture["store"]), Path(fixture["mirror"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = history(root)[0]
        snapshot = cast("dict", first["snapshot"])
        base_record = snapshot["base"]
        original = json.loads(snapshot["metadata"]["base-manifest.json"])
        published = mirror / "base" / base_record["variant"] / base_record["id"][7:]
        index = [{"id": "sha256:" + "f" * 64, "path": f"base/{base_record['variant']}/missing"}]
        replacement_id = None
        for compatible in (False, True):
            base = json.loads(json.dumps(original))
            base["interpreter"]["release"] = "replacement" if compatible else "incompatible"
            if not compatible:
                base["python"] = "3.99.1"
            identity_fields = {
                key: value for key, value in base.items() if key not in {"id", "signature"}
            }
            identity_fields["archive"] = {"format": "tar+zstd"}
            identity = hashlib.sha256(canonical_json(identity_fields)).hexdigest()
            base["id"] = "sha256:" + identity
            destination = published.parent / identity
            shutil.copytree(published, destination)
            (destination / "manifest.json").write_bytes(canonical_json(base))
            index.append({"id": base["id"], "path": f"base/{base_record['variant']}/{identity}"})
            if compatible:
                replacement_id = base["id"]
        (published.parent / "index.json").write_text(json.dumps(index))
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        code, result = _run(root, "update", "--yes")
        assert code == 0 and result["current"] == 2, result
        preserved = {path.name: path.read_bytes() for path in (root / "history").glob("*.json")}
        shutil.rmtree(published)
        shutil.rmtree(store / "bases" / base_record["id"][7:])
        shutil.rmtree(store / "downloads" / base_record["id"][7:])
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".lock"
            and "runtime" not in path.relative_to(root).parts
        }
        code, result = _run(root, "restore", "1", "--yes")
        assert code == 0 and result["status"] == "approval-required", result
        alternatives = result["restorePlan"]["alternatives"]
        assert len(alternatives) == 1
        chosen = alternatives[0]
        assert chosen["snapshot"]["engine"] == snapshot["engine"]
        assert chosen["snapshot"]["base"]["id"] == replacement_id
        assert chosen["substitutions"] == [
            {
                "field": "base",
                "before": base_record,
                "after": {"id": replacement_id, "variant": base_record["variant"]},
            }
        ]
        assert {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".lock"
            and "runtime" not in path.relative_to(root).parts
        } == before
        code, result = _run(root, "restore", "1", "--approve", chosen["approval"])
        assert code == 0 and result["current"] == 3 and result["exact"] is False, result
        restored = history(root)[-1]
        assert cast("dict", restored["snapshot"])["engine"] == snapshot["engine"]
        assert cast("dict", restored["snapshot"])["base"]["id"] == replacement_id
        assert restored["substitutions"] == chosen["substitutions"]
        assert {
            path.name: path.read_bytes()
            for path in (root / "history").glob("*.json")
            if path.name in preserved
        } == preserved
    finally:
        _run(root, "stop", "--yes")


def test_local_restore_reuses_retained_inputs_without_checkout_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor.environment_generations import build_generation
    from dinkster_supervisor.environment_state import content_digest, write_document
    from dinkster_supervisor.environment_transactions import activate, begin_activation

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
    root, store = Path(fixture["root"]), Path(fixture["store"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = Generation.read(root / "generations/1/generation.json")
        old_base = store / "bases" / first.base[7:]
        staging = tmp_path / "local-base"
        shutil.copytree(old_base / "python", staging / "python", symlinks=False)
        base = json.loads((old_base / "manifest.json").read_bytes())
        base.pop("id")
        base.pop("archive")
        base.update(source="local")
        base["interpreter"]["files"] = {
            path.relative_to(staging).as_posix(): content_digest(path)
            for path in (staging / "python").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        base_id = "sha256:" + hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
        base["id"] = base_id
        write_document(staging / "manifest.json", base)
        shutil.move(str(staging), store / "bases" / base_id[7:])
        published = json.loads((root / "generations/1/engine-manifest.json").read_bytes())
        engine = {
            "format": "dinkster.engine/1",
            "source": "local",
            "version": first.engine_version,
            "workerProtocol": first.worker_protocol,
            "wheels": [
                {**row, "filename": row["name"], "url": "local"}
                for row in published["artifacts"]
                if row["name"].endswith(".whl")
            ],
        }
        engine_id = (
            "sha256:" + hashlib.sha256(json.dumps(engine, sort_keys=True).encode()).hexdigest()
        )
        write_document(store / "engines" / engine_id[7:] / "release-manifest.json", engine)
        journal = begin_activation(root, 2, "update")
        build_generation(
            root=root,
            number=2,
            store=store,
            base=base_id,
            engine_version=first.engine_version,
            engine_release=engine_id,
            constraints=root / "generations/1/constraints.txt",
            variant=first.variant,
            worker_protocol=first.worker_protocol,
            source_kind="local",
        )
        activate(root, journal, revalidate=lambda: None)
        before = history(root)
        assert cast("dict[str, object]", before[-1]["snapshot"])["source"] == "local"
        code, result = _run(root, "restore", "1", "--yes")
        assert code == 0 and result["current"] == 3, result
        shutil.rmtree(root / "generations/2")
        code, planned = _run(root, "restore", "2", "--plan")
        assert code == 0 and planned["restorePlan"]["exact"] is True, planned
        code, restored = _run(root, "restore", "2", "--yes")
        assert code == 0 and restored["current"] == 4 and restored["exact"] is True, restored
        assert history(root)[-1]["snapshot"] == before[-1]["snapshot"]
        assert history(root)[:2] == before
        for path, kind in (
            (
                store / "bases" / base_id[7:] / next(iter(base["interpreter"]["files"])),
                "local base",
            ),
            (
                store / "wheels" / engine["wheels"][0]["sha256"] / engine["wheels"][0]["filename"],
                "local wheel",
            ),
        ):
            payload = path.read_bytes()
            path.unlink()
            code, unavailable = _run(root, "restore", "2", "--yes")
            assert code == 0 and unavailable["status"] == "approval-required", unavailable
            assert any(item["kind"] == kind for item in unavailable["restorePlan"]["missing"])
            assert (root / "current").read_text() == "4\n"
            assert len(history(root)) == 4
            path.write_bytes(payload)
    finally:
        _run(root, "stop", "--yes")


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


@pytest.mark.parametrize("operation", ["restore", "rollback"])
def test_restore_rebuilds_pruned_software_without_changing_user_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
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
        parked_port = free_port()
        assert (
            _run(root, "start", "--engine", "parked", "--port", str(parked_port), "--yes")[0] == 0
        )
        assert _run(root, "stop", "--engine", "parked", "--yes")[0] == 0
        shutil.rmtree(root / "generations/1")
        project = json.loads((root / "project.json").read_bytes())
        project.update(channel="stable", custom={"preserve": "edited after engine update"})
        (root / "project.json").write_text(json.dumps(project))
        settings = (root / "project.json").read_bytes()
        command = ("restore", "1") if operation == "restore" else ("rollback",)
        code, result = _run(root, *command, "--plan")
        assert code == 0 and result["status"] == "planned", result
        assert len(history(root)) == 2 and not (root / "generations/3").exists()
        code, result = _run(root, *command, "--yes")
        assert code == 0 and result["current"] == 3 and result["exact"] is True, result
        entries = history(root)
        assert len(entries) == 3 and entries[-1]["restoredFrom"] == 1
        assert entries[-1]["reason"] == operation
        assert entries[-1]["snapshot"] == first["snapshot"]
        assert entries[-1]["recordedTools"] == cast("dict", first["snapshot"])["tools"]
        assert engine_status(root)[1] == {
            "name": "parked",
            "generation": 3,
            "state": "stopped",
            "url": f"http://127.0.0.1:{parked_port}",
        }
        assert (root / "history/00000000000000000001.json").read_bytes() == preserved
        assert (root / "history/00000000000000000002.json").read_bytes() == second
        assert (root / "project.json").read_bytes() == settings
        monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", "refuse")
        code, failed = _run(root, *command, "--yes")
        assert code == 1 and "recorded by tools" in failed["error"], failed
        assert (root / "current").read_text() == "3\n"
        assert history(root) == entries and not (root / "generations/4").exists()
        code, queried = _run(root, "history")
        assert code == 0 and len(queried["attempts"]) == 1
        failed_source = 1 if operation == "restore" else 2
        assert queried["attempts"][0]["restoredFrom"] == failed_source
        assert queried["attempts"][0]["error"] == failed["error"]
        monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
        interrupt_restore = """
import os
import sys
from dinkster_supervisor import environment, environment_restore
built = sys.argv[2] == 'built'
original = environment_restore.build_generation
def interrupt(**kwargs):
    if built:
        original(**kwargs)
    os._exit(73)
environment_restore.build_generation = interrupt
command = ['restore', '1'] if sys.argv[3] == 'restore' else ['rollback']
sys.argv = ['dinkster-env', '--root', sys.argv[1], *command, '--yes', '--json']
environment.main()
"""
        for count, stage in enumerate(("building", "built"), 2):
            interrupted = subprocess.run(
                [sys.executable, "-c", interrupt_restore, str(root), stage, operation], timeout=120
            )
            assert interrupted.returncode == 73
            assert (root / "current").read_text() == "3\n" and history(root) == entries
            if stage == "building":
                interrupted_recovery = """
import os
import sys
from dinkster_supervisor import environment, environment_transactions
original = environment_transactions.record_failed_restore
def interrupt(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(74)
environment_transactions.record_failed_restore = interrupt
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'stop', '--yes', '--json']
environment.main()
"""
                recovered_once = subprocess.run(
                    [sys.executable, "-c", interrupted_recovery, str(root)], timeout=120
                )
                assert recovered_once.returncode == 74
                assert len(_run(root, "history")[1]["attempts"]) == count
            assert _run(root, "stop", "--yes")[0] == 0
            assert not (root / "generations/4").exists()
            code, recovered = _run(root, "history")
            assert code == 0 and len(recovered["attempts"]) == count
            assert recovered["attempts"][-1]["restoredFrom"] == failed_source
            assert "interrupted" in recovered["attempts"][-1]["error"]
            assert history(root) == entries
            assert _run(root, "stop", "--yes")[0] == 0
            assert _run(root, "history")[1]["attempts"] == recovered["attempts"]
        old_snapshot = cast("dict[str, object]", first["snapshot"])
        old_engine = cast("dict[str, str]", old_snapshot["engine"])
        old_metadata = cast("dict[str, str]", old_snapshot["metadata"])
        old_manifest = json.loads(old_metadata["engine-manifest.json"])
        store = Path(fixture["store"])
        mirror = Path(fixture["mirror"])
        executable_release(
            mirror, json.loads(Path(fixture["baseManifest"]).read_bytes()), "3.0", 73
        )
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        shutil.rmtree(mirror / "engine" / old_engine["release"][7:])
        (mirror / "engine/index.json").write_text(
            json.dumps(
                [
                    {
                        "path": f"engine/{old_engine['release'][7:]}",
                        "sha256": old_engine["release"][7:],
                    },
                    *[
                        {"path": f"engine/{path.parent.name}", "sha256": path.parent.name}
                        for path in sorted((mirror / "engine").glob("*/release-manifest.json"))
                    ],
                ]
            )
        )
        for artifact in old_manifest["artifacts"]:
            if artifact["name"].endswith(".whl"):
                cached = store / "wheels" / artifact["sha256"]
                if cached.exists():
                    shutil.rmtree(cached)
        prior_files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.parts[-1] != ".lock"
            and "runtime" not in path.relative_to(root).parts
        }
        for flags in (("--plan",), (), ("--yes",)):
            code, proposed = _run(root, "restore", "1", *flags)
            assert code == 0 and proposed["status"] == "approval-required", proposed
            assert any(
                item["identity"] == old_engine["release"]
                for item in proposed["restorePlan"]["missing"]
            )
            assert {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
                and path.parts[-1] != ".lock"
                and "runtime" not in path.relative_to(root).parts
            } == prior_files
        alternatives = proposed["restorePlan"]["alternatives"]
        assert len(alternatives) == 2
        assert {item["snapshot"]["engine"]["version"] for item in alternatives} == {"2.0", "3.0"}
        for item in alternatives:
            assert {change["field"] for change in item["substitutions"]} == {
                "engine",
                "constraints",
            }
        chosen = next(
            item for item in alternatives if item["snapshot"]["engine"]["version"] == "2.0"
        )
        approval = chosen["approval"]
        (root / "project.json").write_bytes(settings + b"\n")
        code, stale = _run(root, "restore", "1", "--approve", approval)
        assert code == 1 and "stale" in stale["error"], stale
        assert (root / "current").read_text() == "3\n"
        (root / "project.json").write_bytes(settings)
        code, restored = _run(root, "restore", "1", "--approve", approval)
        assert code == 0 and restored["current"] == 4 and restored["exact"] is False, restored
        assert len(history(root)) == 4
        approximate = history(root)[-1]
        assert approximate["exact"] is False and approximate["restoredFrom"] == 1
        assert approximate["substitutions"] == chosen["substitutions"]
        assert cast("dict[str, object]", approximate["snapshot"])["base"] == old_snapshot["base"]
        for name in ("workflows", "models", "outputs"):
            assert (root / name / "sentinel").read_bytes() == name.encode(
                "ascii"
            ) + b" private data"
        assert _run(root, "doctor")[1]["status"] == "clean"
    finally:
        _run(root, "stop", "--yes")
