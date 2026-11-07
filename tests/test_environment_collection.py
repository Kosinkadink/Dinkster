"""Confirmed collection preserves journal ancestry and recovers interrupted quarantine."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import cast
from urllib.request import Request, urlopen

import pytest
from dinkster_store import object_path
from dinkster_supervisor.environment_state import (
    Generation,
    Journal,
    content_digest,
    write_document,
)
from test_environment_cli import _run
from test_environment_state import _generation


@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
@pytest.mark.parametrize("command", ["gc", "cleanup"])
@pytest.mark.parametrize("worker_kind", ["isolated", "grouped", "descendant"])
def test_supervisor_stop_proves_blocked_import_worker_exit_before_collection(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, worker_kind: str, command: str
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from test_environment_runtime import free_port

    tmp_path = cleanup_project
    if command == "gc":
        (tmp_path / "store").unlink()
    launch = subprocess.Popen
    script = (
        "from dinkster_supervisor import environment_collection as c, supervisor\n"
        "async def disabled(root, stopped):\n"
        "    await stopped.wait()\n"
        "c.maintain_generations=disabled\n"
        "supervisor.main()\n"
    )

    def no_periodic_maintenance(command, **kwargs):
        if "dinkster_supervisor.supervisor" not in command:
            return launch(command, **kwargs)
        return launch([command[0], "-c", script, *command[3:]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", no_periodic_maintenance)
    protected = {
        p: p.read_bytes()
        for name in (
            "history",
            "workflows",
            "models",
            "outputs",
            "mounts",
            "cache",
            "generations/1",
            "generations/3",
        )
        for p in (tmp_path / name).rglob("*")
        if p.is_file()
    }
    pack = tmp_path / "fixture-pack"
    pack.mkdir()
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname="cleanup-fixture"\n[pack.entry]\nnodes="cleanup_fixture:NODES"\n'
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(20)
        blocked_import = (
            "import socket\n"
            f"connection=socket.create_connection(('127.0.0.1',{listener.getsockname()[1]}))\n"
            "connection.sendall(b'ready')\n"
            "while True:\n"
            " data=connection.recv(1)\n"
            " if not data or data==b'x': break\n"
            " connection.sendall(b'p')\n"
            "connection.close()\nNODES=[]\n"
        )
        if worker_kind == "descendant":
            blocked_import = (
                "import signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n" + blocked_import
            )
            (pack / "cleanup_fixture.py").write_text(
                f"import subprocess,sys\nsubprocess.run([sys.executable,'-c',{blocked_import!r}])\n"
                "NODES=[]\n"
            )
        else:
            (pack / "cleanup_fixture.py").write_text(blocked_import)
        serve = tmp_path / "dinkster/serve.py"
        worker = (
            f"GroupIsolatedWorker('cleanup-group', [{str(manifest)!r}], registry, use_shm=False)"
            if worker_kind == "grouped"
            else f"IsolatedWorker({str(manifest)!r}, registry, use_shm=False)"
        )
        serve.write_text(
            serve.read_text().replace(
                "ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()",
                "import asyncio,threading\n"
                "from dinkster_workers import IsolatedWorker,GroupIsolatedWorker\n"
                "from dinkster_values import TypeRegistry,register_core_types\n"
                "async def launch_worker():\n"
                " registry=TypeRegistry();register_core_types(registry)\n"
                f" worker={worker}\n"
                " await worker.start()\n"
                " await asyncio.Event().wait()\n"
                "threading.Thread(target=lambda:asyncio.run(launch_worker()),daemon=True).start()\n"
                "ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()",
            )
        )
        connection = None
        try:
            port = free_port()
            start_engine(tmp_path, 2, "old", port)
            connection, _ = listener.accept()
            connection.settimeout(10)
            assert connection.recv(5) == b"ready"
            # Worker imports may refresh interpreter bytecode before collection begins.
            protected.update(
                {p: p.read_bytes() for p in (tmp_path / "generations/2").rglob("*") if p.is_file()}
            )
            binding = tmp_path / "runtime/2/old.json"
            binding_bytes = binding.read_bytes()
            code, refused = _run(tmp_path, command, "--yes")
            if command == "cleanup":
                assert code == 0 and refused["status"] == "complete", refused
                retained = next(
                    row for row in refused["cleanup"]["generations"] if row["generation"] == 2
                )
                assert retained["status"] == "retained", retained
                assert any("cleanup is unproven" in reason for reason in retained["reasons"])
                assert not (tmp_path / "generations/7").exists()
            else:
                assert code == 1 and "cleanup is unproven" in refused["error"], refused
            assert {p: p.read_bytes() for p in protected} == protected
            assert binding.read_bytes() == binding_bytes
            with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
                assert response.status == 200
            connection.sendall(b"p")
            assert connection.recv(1) == b"p"
            assert (tmp_path / "generations/2").is_dir()
            if sys.platform == "linux" or os.name == "nt":
                request = Request(
                    f"http://127.0.0.1:{port}/supervisor/engine/restart",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=25) as response:
                    assert response.status == 200
                try:
                    assert connection.recv(1) == b""
                except ConnectionResetError:
                    pass
                connection.close()
                connection, _ = listener.accept()
                connection.settimeout(10)
                assert connection.recv(5) == b"ready"
                binding_bytes = binding.read_bytes()
                code, refused = _run(tmp_path, command, "--yes")
                if command == "cleanup":
                    assert code == 0 and refused["status"] == "complete", refused
                    retained = next(
                        row for row in refused["cleanup"]["generations"] if row["generation"] == 2
                    )
                    assert retained["status"] == "retained", retained
                    assert any("cleanup is unproven" in reason for reason in retained["reasons"])
                else:
                    assert code == 1 and "cleanup is unproven" in refused["error"], refused
                assert {p: p.read_bytes() for p in protected} == protected
                assert binding.read_bytes() == binding_bytes
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
                    assert response.status == 200
                connection.sendall(b"p")
                assert connection.recv(1) == b"p"
            stop_engine(tmp_path, "old", generation=2)
            if sys.platform != "linux" and os.name != "nt":
                code, refused = _run(tmp_path, command, "--yes")
                if command == "gc":
                    assert code == 1 and "descendant cleanup is unproven" in refused["error"], (
                        refused
                    )
                else:
                    assert code == 0, refused
                    retained = next(
                        row for row in refused["cleanup"]["generations"] if row["generation"] == 2
                    )
                    assert "managed worker descendant cleanup is unproven" in retained["reasons"]
                assert (tmp_path / "generations/2").is_dir()
                return
            try:
                assert connection.recv(1) == b""
            except ConnectionResetError:
                pass
            code, applied = _run(tmp_path, command, "--yes")
            assert code == 0 and applied["status"] == (
                "complete" if command == "cleanup" else "applied"
            ), applied
            assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["1", "3"]
            assert {
                p: p.read_bytes()
                for p in protected
                if not p.is_relative_to(tmp_path / "generations/2")
            } == {
                p: data
                for p, data in protected.items()
                if not p.is_relative_to(tmp_path / "generations/2")
            }
        finally:
            stop_engine(tmp_path, "old", generation=2)
            if connection is not None:
                try:
                    connection.sendall(b"x")
                except OSError:
                    pass
                connection.close()


def test_collection_retains_legacy_direct_child_only_stop_receipt(tmp_path: Path):
    for number in (1, 2, 3):
        _generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 2, 3, "complete").write(tmp_path / "journal/100.json")
    instance = "legacy-supervisor"
    directory = tmp_path / "runtime/1/old.process" / hashlib.sha256(instance.encode()).hexdigest()
    directory.mkdir(parents=True)
    write_document(directory / "state.json", {"instance": instance, "state": "stopped", "pid": 1})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    code, refused = _run(tmp_path, "gc", "--yes")
    assert code == 1 and "descendant cleanup is unproven" in refused["error"], refused
    assert (tmp_path / "generations/1").is_dir()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
@pytest.mark.parametrize("command", ["gc", "cleanup"])
def test_failed_owned_family_cleanup_cannot_make_generation_collectible(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, command: str
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_state import EnvironmentError
    from test_environment_runtime import free_port

    tmp_path = cleanup_project
    if command == "gc":
        (tmp_path / "store").unlink()
    original = subprocess.Popen
    processes: list[subprocess.Popen] = []
    script = (
        "from dinkster_supervisor import environment_collection as c, supervisor\n"
        "from dinkster_supervisor.process_family import ProcessFamily\n"
        "async def disabled(root, stopped): await stopped.wait()\n"
        "c.maintain_generations=disabled\n"
        "def refuse(self): raise RuntimeError('owned cleanup fixture failure')\n"
        "ProcessFamily.stop_children=refuse\n"
        "supervisor.main()\n"
    )

    def launch(command, **kwargs):
        if "dinkster_supervisor.supervisor" not in command:
            return original(command, **kwargs)
        process = original([command[0], "-c", script, *command[3:]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    try:
        runtime.start_engine(tmp_path, 2, "old", free_port())
        with pytest.raises(EnvironmentError, match="without proving child cleanup"):
            runtime.stop_engine(tmp_path, "old", generation=2)
        assert processes[0].wait(timeout=5) != 0
    finally:
        # The injected failure occurs after direct-engine shutdown; only the fixture host remains.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    monkeypatch.undo()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for _ in range(2):
        code, result = _run(tmp_path, command, "--yes")
        if command == "gc":
            assert code == 1 and "cleanup is unproven" in result["error"], result
            assert {p: p.read_bytes() for p in before} == before
        else:
            assert code == 0 and result["status"] == "complete", result
            assert (tmp_path / "generations/2").is_dir()
            assert not (tmp_path / "generations/7").exists()
            code, status = _run(tmp_path, "status")
            assert code == 0, status
            rows = {row["generation"]: row for row in status["cleanup"]["generations"]}
            assert rows[2]["status"] == "retained", rows[2]
            assert any("cleanup is unproven" in reason for reason in rows[2]["reasons"])
            protected = {
                p: data
                for p, data in before.items()
                if not p.is_relative_to(tmp_path / "generations/7")
            }
            assert {p: p.read_bytes() for p in protected} == protected


def test_cleanup_policy_persists_without_collecting_or_changing_software(tmp_path: Path):
    from dinkster_supervisor.environment_state import writer_lock

    for number in (1, 2, 3):
        _generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 1, 3, "complete").write(tmp_path / "journal/100.json")
    for name in ("workflows", "models", "outputs", "mounts", "history", "store"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sentinel").write_text(name)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    default = {"automatic": True, "previousSuccessfulActivations": 1, "minimumAgeSeconds": 0}
    code, preview = _run(tmp_path, "cleanup", "--plan")
    assert code == 0 and preview["cleanup"]["policy"] == default, preview
    for flags in (("--plan",), ()):
        code, planned = _run(tmp_path, "cleanup", "--automatic", "off", *flags)
        assert code == 0 and planned["status"] == "planned", planned
        assert planned["requestedAutomaticCleanup"] is False
        assert planned["cleanup"]["policy"] == default
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    with writer_lock(tmp_path):
        code, refused = _run(tmp_path, "cleanup", "--automatic", "off", "--yes")
    assert code == 1 and "another environment writer holds" in refused["error"], refused
    assert not (tmp_path / "cleanup.json").exists()
    for option, enabled in (("off", False), ("on", True), ("off", False)):
        code, applied = _run(tmp_path, "cleanup", "--automatic", option, "--yes")
        assert code == 0 and applied["status"] == "applied", applied
        policy = default | {"automatic": enabled}
        assert applied["cleanup"]["policy"] == policy
        for command in ("cleanup", "status"):
            code, read = _run(tmp_path, command)
            assert code == 0 and read["cleanup"]["policy"] == policy, read
        assert all(p.read_bytes() == value for p, value in before.items())
        assert not (tmp_path / ".garbage").exists()
    assert json.loads((tmp_path / "cleanup.json").read_text()) == {
        "format": "dinkster.cleanup/1",
        "automatic": False,
    }


@pytest.mark.parametrize("condition", ["json", "format", "string", "number", "linked"])
def test_invalid_cleanup_policy_refuses_preview_and_policy_changes(tmp_path: Path, condition: str):
    _generation(tmp_path, 1)
    (tmp_path / "current").write_text("1\n")
    Journal("setup", None, 1, "complete").write(tmp_path / "journal/100.json")
    policy = tmp_path / "cleanup.json"
    target = tmp_path / "keep-policy"
    if condition == "linked":
        target.write_text('{"format":"dinkster.cleanup/1","automatic":false}')
        policy.symlink_to(target)
    elif condition == "json":
        policy.write_text("broken")
    else:
        policy.write_text(
            json.dumps(
                {
                    "format": "unknown" if condition == "format" else "dinkster.cleanup/1",
                    "automatic": "false"
                    if condition == "string"
                    else (0 if condition == "number" else False),
                }
            )
        )
    before = policy.read_bytes()
    for arguments in (("--plan",), ("--automatic", "on", "--yes")):
        code, refused = _run(tmp_path, "cleanup", *arguments)
        assert code == 1, refused
        assert "policy" in refused["error"] or "cannot read cleanup.json" in refused["error"], (
            refused
        )
        assert policy.read_bytes() == before
        assert (tmp_path / "generations/1/generation.json").is_file()
    code, status = _run(tmp_path, "status")
    assert code == 0 and status["cleanup"]["warnings"], status
    assert "policy" not in status["cleanup"]
    assert policy.read_bytes() == before
    assert policy.is_symlink() == (condition == "linked")


@pytest.mark.parametrize(
    "condition", [None, "pending", "ambiguous", "quarantine", "corrupt-journal"]
)
def test_cleanup_preview_and_status_are_read_only_and_explain_retention(
    tmp_path: Path, condition: str | None
):
    for number in (1, 2, 3, 7):
        _generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 1, 3, "complete").write(tmp_path / "journal/100.json")
    Journal("update", 3, 7, "rolled-back").write(tmp_path / "journal/200.json")
    for name in ("workflows", "models", "outputs", "mounts", "history"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sentinel").write_text(name)
    if condition == "pending":
        Journal("update", 3, 9, "planned").write(tmp_path / "journal/300.json")
    elif condition == "ambiguous":
        directory = tmp_path / "runtime/2"
        directory.mkdir(parents=True)
        (directory / "old.json").write_text("{}")
    elif condition == "quarantine":
        directory = tmp_path / ".garbage/300"
        directory.mkdir(parents=True)
        (tmp_path / "generations/7").rename(directory / "7")
        Journal("gc", 3, 3, "complete", collected_generations=(7,)).write(
            tmp_path / "journal/300.json"
        )
    elif condition == "corrupt-journal":
        (tmp_path / "journal/300.json").write_text("broken")
    before = {
        p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    code, preview = _run(tmp_path, "cleanup", "--plan")
    status_code, status = _run(tmp_path, "status")
    if condition == "corrupt-journal":
        assert code == 1 and "cannot read 300.json" in preview["error"], preview
        assert status_code == 1 and status["status"] == "error", status
        assert "cannot read 300.json" in status["error"]
    else:
        assert status_code == 0 and status["status"] == "clean", status
        assert code == 0 and preview["status"] == "planned", preview
        assert status["cleanup"] == preview["cleanup"]
        rows = {row["generation"]: row for row in preview["cleanup"]["generations"]}
        assert rows[1]["reasons"] == ["previous successful activation"]
        assert rows[3]["reasons"] == ["current activation"]
        assert rows[2]["status"] == "deferred"
        if condition == "ambiguous":
            assert "invalid" in rows[2]["reasons"][0]
            assert rows[7]["status"] == "deferred"
        elif condition == "pending":
            assert rows[7]["status"] == "deferred"
            assert "requires recovery" in rows[7]["reasons"][0]
        elif condition == "quarantine":
            assert preview["cleanup"]["quarantine"] == [
                {"journal": "300.json", "stage": "complete", "generations": [7]}
            ]
        else:
            assert "history coverage required" in rows[2]["reasons"][0]
    assert {
        p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
    } == before
    assert not (tmp_path / ".lock").exists()
    assert not (tmp_path / "runtime/.lock").exists()


@pytest.mark.parametrize(
    "condition",
    [
        "covered",
        "observer-changed",
        "packed-no-launch",
        "unsealed-pack-runtime",
        "missing-history",
        "prepared-only",
        "missing-provenance",
        "duplicate-provenance",
        "failed-provenance",
        "changed-provenance",
        "corrupt-later-entry",
        "missing-later-entry",
        "unpublished-tail",
        "missing-interpreter",
        "drifted-interpreter",
        "engine",
        "base",
        "constraints",
        "workerProtocol",
        "source",
        "inProcessPacks",
        "metadata",
    ],
)
def test_cleanup_preview_requires_published_sealed_software_and_completed_provenance(
    tmp_path: Path, condition: str
):
    from dinkster_supervisor.environment_history import prepare_entry, publish_entry
    from dinkster_supervisor.environment_state import interpreter

    entries = []
    # Activation ancestry, not numeric folder order, protects generation 1.
    for index, number in enumerate((2, 1, 3), 1):
        tree = _generation(tmp_path, number).parent
        python = interpreter(tree)
        python.parent.mkdir(exist_ok=True)
        python.write_bytes(b"sealed interpreter fixture")
        seal = tree / "generation.json"
        record = Generation.read(seal)
        replace(
            record,
            files=(*record.files, (python.relative_to(tree).as_posix(), content_digest(python))),
        ).write(seal)
        operation = "setup" if index == 1 else "update"
        entry = prepare_entry(tmp_path, tree, operation=str(index * 100), reason=operation)
        if condition == "observer-changed":
            snapshot = cast("dict[str, object]", entry["snapshot"])
            snapshot["host"] = {"system": "other", "python": "different"}
            snapshot["tools"] = {"version": "older"}
        if condition in {"packed-no-launch", "unsealed-pack-runtime"} and number == 2:
            lock = tree / "packs/fixture/pack-lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text("{}")
            record = Generation.read(seal)
            replace(
                record,
                files=(*record.files, ("packs/fixture/pack-lock.json", content_digest(lock))),
            ).write(seal)
            snapshot = cast("dict[str, object]", entry["snapshot"])
            metadata = cast("dict[str, str]", snapshot["metadata"])
            metadata["packs/fixture/pack-lock.json"] = "{}"
            # The sealed interpreter fixture cannot execute; preview must only read it.
            if condition == "unsealed-pack-runtime":
                (lock.parent / "runtime.py").write_bytes(b"modified runtime, unchanged lock")
        publish_entry(tmp_path, entry)
        Journal(
            operation,
            None if index == 1 else (2 if index == 2 else 1),
            number,
            "complete",
            history_entry=entry,
        ).write(tmp_path / f"journal/{index * 100}.json")
        entries.append(entry)
    (tmp_path / "current").write_text("3\n")
    first_history = tmp_path / "history/00000000000000000001.json"
    first_journal = tmp_path / "journal/100.json"
    tree = tmp_path / "generations/2"
    if condition == "missing-history":
        first_history.unlink()
    elif condition == "prepared-only":
        for path in (tmp_path / "history").glob("*.json"):
            path.unlink()
    elif condition == "missing-provenance":
        first_journal.unlink()
    elif condition == "duplicate-provenance":
        Journal("update", 1, 2, "complete", history_entry=entries[0]).write(
            tmp_path / "journal/400.json"
        )
    elif condition == "failed-provenance":
        replace(Journal.read(first_journal), stage="rolled-back").write(first_journal)
    elif condition == "changed-provenance":
        entry = {**entries[0], "timestamp": "changed receipt"}
        replace(Journal.read(first_journal), history_entry=entry).write(first_journal)
    elif condition == "corrupt-later-entry":
        (tmp_path / "history/00000000000000000003.json").write_text("broken")
    elif condition == "missing-later-entry":
        (tmp_path / "history/00000000000000000002.json").unlink()
    elif condition == "unpublished-tail":
        (tmp_path / "history/00000000000000000003.json").unlink()
    elif condition == "missing-interpreter":
        interpreter(tree).unlink()
    elif condition == "drifted-interpreter":
        interpreter(tree).write_bytes(b"changed interpreter")
    elif condition in {"engine", "base", "workerProtocol", "source", "inProcessPacks"}:
        document = json.loads((tree / "generation.json").read_bytes())
        document[condition] = {
            "engine": {"version": "9.0", "release": "sha256:" + "d" * 64},
            "base": "sha256:" + "e" * 64,
            "workerProtocol": 2,
            "source": "local",
            "inProcessPacks": ["other"],
        }[condition]
        write_document(tree / "generation.json", document)
    elif condition in {"constraints", "metadata"}:
        name = "constraints.txt" if condition == "constraints" else "artifact-sources.json"
        (tree / name).write_text("demo==2\n" if condition == "constraints" else "{}")
        record = Generation.read(tree / "generation.json")
        files = dict(record.files)
        files[name] = content_digest(tree / name)
        replace(record, constraints=files["constraints.txt"], files=tuple(files.items())).write(
            tree / "generation.json"
        )
    for name in ("workflows", "models", "outputs", "mounts"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sentinel").write_bytes(b"private user data")
    before = {
        p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    for command in (("cleanup", "--plan"), ("status",)):
        code, result = _run(tmp_path, *command)
        assert code == 0, result
        rows = {row["generation"]: row for row in result["cleanup"]["generations"]}
        assert rows[1]["reasons"] == ["previous successful activation"]
        assert rows[3]["reasons"] == ["current activation"]
        if condition in {"covered", "observer-changed", "packed-no-launch"}:
            assert rows[2]["status"] == "candidate", rows[2]
            assert rows[2]["reasons"] == [
                "outside current and previous; covered by history entry 1"
            ]
        else:
            assert rows[2]["status"] == "deferred", rows[2]
            assert "history coverage required" in rows[2]["reasons"][0]
        assert {
            p.relative_to(tmp_path): content_digest(p) for p in tmp_path.rglob("*") if p.is_file()
        } == before
    assert not (tmp_path / ".lock").exists()
    assert not (tmp_path / "runtime/.lock").exists()


@pytest.fixture
def cleanup_project(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    from dinkster_supervisor.environment_history import prepare_entry, publish_entry
    from dinkster_supervisor.environment_state import interpreter

    for index, number in enumerate((2, 7, 1, 3), 1):
        if getattr(request, "param", False):
            from test_environment_transactions import _target

            tree = _target(tmp_path, number)
        else:
            tree = _generation(tmp_path, number).parent
            python = interpreter(tree)
            python.parent.mkdir(exist_ok=True)
            python.write_bytes(b"sealed interpreter, not executable")
            seal = tree / "generation.json"
            generation = Generation.read(seal)
            replace(
                generation,
                files=(
                    *generation.files,
                    (python.relative_to(tree).as_posix(), content_digest(python)),
                ),
            ).write(seal)
        operation = "setup" if index == 1 else "update"
        entry = prepare_entry(tmp_path, tree, operation=str(index * 100), reason=operation)
        Journal(
            operation,
            None if index == 1 else (2, 7, 1)[index - 2],
            number,
            "complete",
            history_entry=entry,
        ).write(tmp_path / "journal" / f"{index * 100}.json")
        publish_entry(tmp_path, entry)
    (tmp_path / "current").write_text("3\n")
    for name in ("workflows", "models", "outputs", "mounts", "cache/uv-cache", "cache/wheels"):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "sentinel").write_text(name)
    (tmp_path / "store").write_text(str(tmp_path / "cache") + "\n")
    return tmp_path


@pytest.mark.parametrize("fault", [None, "provenance", "malformed", "payload"])
def test_cleanup_captures_legacy_software_without_repairing_conflicting_imports(
    cleanup_project: Path, fault: str | None
):
    from dinkster_supervisor.environment_history import activation_entries, history, validate_import
    from dinkster_supervisor.environment_state import writer_lock

    root = cleanup_project
    for path in (root / "history").glob("*.json"):
        path.unlink()
    for path in (root / "journal").glob("*.json"):
        replace(Journal.read(path), history_entry=None).write(path)
    legacy = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code, disabled = _run(root, "cleanup", "--automatic", "off", "--yes")
    assert code == 0, disabled
    code, disabled = _run(root, "cleanup", "--yes")
    assert code == 0 and disabled["status"] == "disabled", disabled
    with writer_lock(root):
        code, busy = _run(root, "cleanup", "--yes")
    assert code == 0 and busy["status"] == "deferred", busy
    assert history(root) == []
    assert {p: p.read_bytes() for p in legacy} == legacy
    code, enabled = _run(root, "cleanup", "--automatic", "on", "--yes")
    assert code == 0, enabled
    code, imported = _run(root, "import", "2", "--yes")
    assert code == 0 and imported["entry"]["kind"] == "import", imported
    receipt = root / "history/00000000000000000001.json"
    original = receipt.read_bytes()
    payload = root / "generations/2/packs/fixture/runtime.py"
    if fault == "provenance":
        entry = json.loads(original)
        entry["provenance"]["digest"] = "sha256:" + "0" * 64
        write_document(receipt, entry)
    elif fault == "malformed":
        entry = json.loads(original)
        del entry["generation"]
        write_document(receipt, entry)
    elif fault == "payload":
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"unrecorded runtime is not covered by an import")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code, preview = _run(root, "cleanup", "--plan")
    assert code == 0, preview
    rows = {row["generation"]: row for row in preview["cleanup"]["generations"]}
    assert rows[2]["status"] == ("deferred" if fault else "candidate"), rows[2]
    assert rows[7]["status"] == "deferred", rows[7]
    assert rows[1]["reasons"] == ["previous successful activation"]
    assert rows[3]["reasons"] == ["current activation"]
    if fault:
        assert "history coverage required" in rows[2]["reasons"][0], rows[2]
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    code, result = _run(root, "cleanup", "--yes")
    assert code == 0 and result["status"] == ("deferred" if fault else "complete"), result
    assert (root / "generations/2").exists() == bool(fault)
    assert (root / "generations/7").exists() == (fault in {"provenance", "malformed"})
    protected = {
        p: data
        for p, data in before.items()
        if not any(p.is_relative_to(root / "generations" / str(n)) for n in (2, 7))
        and not p.is_relative_to(root / "runtime/maintenance")
    }
    assert {p: p.read_bytes() for p in protected} == protected
    if fault:
        assert {p: p.read_bytes() for p in before if p.is_relative_to(root / "generations/2")} == {
            p: data for p, data in before.items() if p.is_relative_to(root / "generations/2")
        }
        receipt.write_bytes(original)
        if fault == "payload":
            payload.unlink()
        code, result = _run(root, "cleanup", "--yes")
        assert code == 0 and result["status"] == "complete", result
        assert not (root / "generations/2").exists()
        protected[receipt] = original
    assert {entry["generation"] for entry in history(root)} == {1, 2, 3, 7}
    assert all(entry["kind"] == "import" for entry in history(root))
    assert activation_entries(root) == []
    for entry in history(root):
        validate_import(root, entry)
    receipts = {p: p.read_bytes() for p in (root / "history").glob("*.json")}
    code, result = _run(root, "cleanup", "--yes")
    assert code == 0 and result["status"] == "complete", result
    assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "3"]
    assert {p: p.read_bytes() for p in protected} == protected
    assert {p: p.read_bytes() for p in (root / "history").glob("*.json")} == receipts


@pytest.mark.parametrize("fault", [None, "pending", "conflicting", "unreadable"])
def test_cleanup_publishes_only_completed_activation_history(
    cleanup_project: Path, fault: str | None
):
    root = cleanup_project
    receipt = root / "history/00000000000000000004.json"
    expected = receipt.read_bytes()
    receipt.unlink()
    journal = root / "journal/400.json"
    original_journal = journal.read_bytes()
    if fault == "pending":
        replace(Journal.read(journal), stage="switching").write(journal)
    elif fault == "conflicting":
        conflicting = json.loads(expected)
        conflicting["reason"] = "conflicting receipt must not be replaced"
        write_document(receipt, conflicting)
    elif fault == "unreadable":
        receipt.mkdir()
    protected = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code, result = _run(root, "cleanup", "--yes")
    assert code == 0 and result["status"] == ("deferred" if fault else "complete"), result
    if fault:
        assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "2", "3", "7"]
        assert {p: p.read_bytes() for p in protected} == protected
        assert result["cleanupAttempt"]["warnings"], result
        if fault == "pending":
            assert not receipt.exists()
            assert "requires recovery" in result["cleanupAttempt"]["warnings"][0]
        elif fault == "conflicting":
            assert "cannot replace" in result["cleanupAttempt"]["warnings"][0]
        else:
            assert receipt.is_dir()
        before_status = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        code, status = _run(root, "status")
        assert code == 0, status
        assert status["cleanup"]["lastAttempt"] == result["cleanupAttempt"]
        assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before_status
        if fault == "pending":
            journal.write_bytes(original_journal)
            protected[journal] = original_journal
        elif fault == "conflicting":
            receipt.write_bytes(expected)
            protected[receipt] = expected
        else:
            receipt.rmdir()
        code, result = _run(root, "cleanup", "--yes")
        assert code == 0 and result["status"] == "complete", result
        assert result["cleanupAttempt"]["warnings"] == []
    assert receipt.read_bytes() == expected
    assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "3"]
    protected = {
        p: data
        for p, data in protected.items()
        if not any(p.is_relative_to(root / "generations" / str(n)) for n in (2, 7))
    }
    assert {p: p.read_bytes() for p in protected} == protected


@pytest.mark.parametrize(
    "blocked",
    [None, "history", "unreadable-history", "generation", "runtime", "payload", "disabled", "busy"],
)
def test_generation_cleanup_retry_preserves_history_user_data_and_shared_caches(
    cleanup_project: Path, blocked: str | None
):
    from contextlib import nullcontext

    from dinkster_supervisor.environment_state import writer_lock

    tmp_path = cleanup_project
    damaged: Path | None = None
    original: bytes | None = None
    if blocked == "history":
        (tmp_path / "history/00000000000000000001.json").unlink()
    elif blocked == "unreadable-history":
        damaged = tmp_path / "history/00000000000000000001.json"
        original = damaged.read_bytes()
        damaged.unlink()
        damaged.mkdir()
    elif blocked == "generation":
        damaged = tmp_path / "generations/2/generation.json"
        original = damaged.read_bytes()
        damaged.write_bytes(b"{corrupt generation")
    elif blocked == "runtime":
        damaged = tmp_path / "runtime/2/old.json"
        damaged.parent.mkdir(parents=True)
        damaged.write_bytes(b"{corrupt runtime binding")
    elif blocked == "payload":
        payload = tmp_path / "generations/2/packs/fixture/runtime.py"
        payload.parent.mkdir(parents=True)
        payload.write_text("hand edited runtime")
    elif blocked == "disabled":
        write_document(
            tmp_path / "cleanup.json", {"format": "dinkster.cleanup/1", "automatic": False}
        )
    old_tree = {p: p.read_bytes() for p in (tmp_path / "generations/2").rglob("*") if p.is_file()}
    protected = {
        p: p.read_bytes()
        for parent in (
            "history",
            "workflows",
            "models",
            "outputs",
            "mounts",
            "cache",
            "runtime/2",
            "generations/1",
            "generations/3",
        )
        for p in (tmp_path / parent).rglob("*")
        if p.is_file()
    }
    protected[tmp_path / "current"] = (tmp_path / "current").read_bytes()
    with writer_lock(tmp_path) if blocked == "busy" else nullcontext():
        code, result = _run(tmp_path, "cleanup", "--yes")
    assert code == 0, result
    expected = "disabled" if blocked == "disabled" else ("deferred" if blocked else "complete")
    assert result["status"] == expected, result
    assert {p: p.read_bytes() for p in protected} == protected
    for number in (2, 7):
        retained = blocked in {"history", "unreadable-history", "disabled", "busy"} or (
            blocked in {"payload", "generation", "runtime"} and number == 2
        )
        assert (tmp_path / "generations" / str(number)).exists() == retained
    if blocked:
        assert {p: p.read_bytes() for p in old_tree} == old_tree
    before_status = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    code, status = _run(tmp_path, "status")
    assert code == 0, status
    assert status["cleanup"]["lastAttempt"] == result["cleanupAttempt"]
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before_status
    if blocked == "busy":
        assert "another environment writer holds" in result["cleanupAttempt"]["warnings"][0]
        code, retried = _run(tmp_path, "cleanup", "--yes")
        assert code == 0 and retried["status"] == "complete", retried
        assert not (tmp_path / "generations/2").exists()
    assert {p: p.read_bytes() for p in protected} == protected
    if damaged is not None:
        rows = {row["generation"]: row for row in status["cleanup"]["generations"]}
        assert rows[2]["status"] == "deferred" and rows[2]["reasons"], rows[2]
        if damaged.is_dir():
            damaged.rmdir()
        else:
            damaged.unlink()
        if original is not None:
            damaged.write_bytes(original)
        preserved = {
            p: p.read_bytes()
            for p in protected
            if p.exists() and not p.is_relative_to(tmp_path / "generations/2")
        }
        preserved.update({p: p.read_bytes() for p in (tmp_path / "history").glob("*.json")})
        code, retried = _run(tmp_path, "cleanup", "--yes")
        assert code == 0 and retried["status"] == "complete", retried
        assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["1", "3"]
        assert {p: p.read_bytes() for p in preserved} == preserved


def test_generation_cleanup_coalesces_and_does_not_stop_pending_launches(cleanup_project: Path):
    from dinkster_supervisor.environment_process import cancel_pending_process, prepare_process
    from dinkster_supervisor.environment_state import writer_lock

    root = cleanup_project
    maintenance = root / "runtime/maintenance"
    maintenance.mkdir(parents=True)
    with writer_lock(maintenance):
        code, result = _run(root, "cleanup", "--yes")
    assert code == 0 and result["status"] == "coalesced", result
    assert not (maintenance / "cleanup").exists()
    assert all((root / "generations" / str(n)).is_dir() for n in (1, 2, 3, 7))
    instance = "pending-generation-launch"
    lifecycle = root / "runtime/2/old.process" / hashlib.sha256(instance.encode()).hexdigest()
    prepare_process(lifecycle, instance)
    pending = {p: p.read_bytes() for p in lifecycle.rglob("*") if p.is_file()}
    code, result = _run(root, "cleanup", "--yes")
    assert code == 0, result
    assert (root / "generations/2").is_dir()
    assert not (root / "generations/7").exists()
    assert {p: p.read_bytes() for p in pending} == pending
    assert not cancel_pending_process(lifecycle, instance)
    code, result = _run(root, "cleanup", "--yes")
    assert code == 0 and result["status"] == "complete", result
    assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "3"]


@pytest.mark.parametrize(
    "interrupted", ["before-quarantine", "quarantine", "cleanup", "sharing", "open-file"]
)
def test_generation_cleanup_retry_recovers_without_shared_store_cleanup(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, interrupted: str
):
    from dinkster_supervisor import environment_collection as collection

    root = cleanup_project
    store = root / "cache"
    parked = store / ".garbage/500/uv-cache"
    parked.mkdir(parents=True)
    (parked / "sentinel").write_bytes(b"explicit cache GC only")
    Journal(
        "gc", 3, 3, "complete", collection_store=str(store), collected_store_paths=("uv-cache",)
    ).write(root / "journal/500.json")
    write_document(store / "collection-state.json", {"keep": True})
    protected = {
        p: p.read_bytes()
        for name in (
            "history",
            "workflows",
            "models",
            "outputs",
            "mounts",
            "cache",
            "generations/1",
            "generations/3",
        )
        for p in (root / name).rglob("*")
        if p.is_file()
    }
    store_journal = root / "journal/500.json"
    protected[store_journal] = store_journal.read_bytes()
    journal = None
    if interrupted == "sharing":
        original = collection.shutil.rmtree

        def refuse(path, *args, **kwargs):
            if path.is_relative_to(root / ".garbage"):
                raise PermissionError("fixture Windows sharing violation")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(collection.shutil, "rmtree", refuse)
        result = collection.retry_generation_collection(root)
        assert result["status"] == "deferred", result
        assert "sharing violation" in cast("list[str]", result["warnings"])[0]
        code, status = _run(root, "status")
        assert code == 0 and status["cleanup"]["lastAttempt"] == result, status
        assert status["cleanup"]["quarantine"], status
        monkeypatch.undo()
    else:
        script = """
import os, sys
from pathlib import Path
from dinkster_supervisor import environment_collection as collection, environment
root, interruption = Path(sys.argv[1]), sys.argv[2]
original = collection.durable_rename
def interrupt(source, destination):
    if interruption == 'before-quarantine': os._exit(73)
    original(source, destination)
    os._exit(73)
if interruption in {'cleanup', 'open-file'}:
    collection.shutil.rmtree = lambda *args, **kwargs: os._exit(73)
else:
    collection.durable_rename = interrupt
sys.argv = ['dinkster-env', '--root', str(root), 'cleanup', '--yes', '--json']
environment.main()
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(root), interrupted],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 73, child.stdout + child.stderr
        journal = next(
            p
            for p in (root / "journal").glob("*.json")
            if (record := Journal.read(p)).operation == "gc" and record.collection_store is None
        )
        record = Journal.read(journal)
        assert record.collected_generations == (2, 7)
        committed = interrupted in {"cleanup", "open-file"}
        assert record.stage == ("complete" if committed else "building")
        directory = root / ".garbage" / journal.stem
        for number in (2, 7):
            moved = committed or (interrupted == "quarantine" and number == 2)
            assert (directory / str(number)).is_dir() == moved
            assert (root / "generations" / str(number)).is_dir() != moved
        before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        code, status = _run(root, "status")
        assert code == 0, status
        assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
        assert {p: p.read_bytes() for p in protected} == protected
        if interrupted == "open-file":
            payload = directory / "2/constraints.txt"
            original_bytes = payload.read_bytes()
            with payload.open("rb") as held:
                code, attempt = _run(root, "cleanup", "--yes")
                assert code == 0, attempt
                assert held.read() == original_bytes
                code, status = _run(root, "status")
                assert code == 0, status
                assert status["cleanup"]["lastAttempt"] == attempt["cleanupAttempt"]
                if os.name == "nt":
                    assert attempt["status"] == "deferred", attempt
                    assert attempt["cleanupAttempt"]["warnings"], attempt
                    assert payload.is_file()
                    assert status["cleanup"]["quarantine"] == [
                        {"journal": journal.name, "stage": "complete", "generations": [2, 7]}
                    ]
                else:
                    assert attempt["status"] == "complete", attempt
                    assert not payload.exists()
                    assert status["cleanup"]["quarantine"] == []
    code, retried = _run(root, "cleanup", "--yes")
    assert code == 0 and retried["status"] == "complete", retried
    assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "3"]
    assert {p: p.read_bytes() for p in protected} == protected
    if journal is not None:
        assert Journal.read(journal).stage == (
            "complete" if interrupted in {"cleanup", "open-file"} else "rolled-back"
        )
    assert (parked / "sentinel").read_bytes() == b"explicit cache GC only"
    assert (store / "uv-cache/sentinel").read_text() == "cache/uv-cache"
    assert not list((root / ".garbage").rglob("generation.json"))


@pytest.mark.parametrize("failure", [False, True])
def test_successful_cli_stop_attempts_cleanup_without_turning_failure_into_failed_stop(
    cleanup_project: Path, failure: bool
):
    root = cleanup_project
    write_document(
        root / "project.json",
        {
            "format": "dinkster.project/1",
            "channel": "main",
            "variant": "linux-cpu",
            "mirrors": [str(root / "mirror")],
            "port": 3639,
        },
    )
    if failure:
        (root / "runtime").mkdir()
        (root / "runtime/maintenance").write_bytes(b"keep this unexpected file")
    code, planned = _run(root, "stop", "--plan")
    assert code == 0 and planned["status"] == "planned", planned
    assert all((root / "generations" / str(n)).exists() for n in (1, 2, 3, 7))
    code, stopped = _run(root, "stop", "--yes")
    assert code == 0 and stopped["status"] == "applied", stopped
    assert stopped["cleanupAttempt"]["status"] == ("deferred" if failure else "complete")
    assert (root / "current").read_text() == "3\n"
    assert (root / "generations/2").exists() == failure
    assert (root / "generations/7").exists() == failure
    if failure:
        assert (root / "runtime/maintenance").read_bytes() == b"keep this unexpected file"


@pytest.mark.parametrize(("legacy", "capture_fault"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
def test_managed_supervisor_startup_and_acknowledged_shutdown_collect_unused_generations(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, capture_fault: bool
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_history import history, validate_import
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from test_environment_runtime import free_port

    root = cleanup_project
    if legacy:
        for path in (root / "history").glob("*.json"):
            path.unlink()
        for path in (root / "journal").glob("*.json"):
            replace(Journal.read(path), history_entry=None).write(path)
    launch = subprocess.Popen
    script = (
        "from functools import partial\n"
        "from dinkster_supervisor import environment_collection as c, supervisor\n"
        "c.maintain_generations=partial(c.maintain_generations,interval=0.05)\n"
        "supervisor.main()\n"
    )

    def short_interval(command, **kwargs):
        if "--generation" not in command:
            return launch(command, **kwargs)
        selected = script
        if command[command.index("--generation") + 1] == str(root / "generations/2"):
            selected = (
                "from dinkster_supervisor import environment_collection as c, supervisor\n"
                "async def disabled(root, stopped):\n"
                "    await stopped.wait()\n"
                "c.maintain_generations=disabled\n"
                "supervisor.main()\n"
            )
        return launch([command[0], "-c", selected, *command[3:]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", short_interval)
    write_document(root / "cleanup.json", {"format": "dinkster.cleanup/1", "automatic": False})
    old_port, current_port = free_port(), free_port()

    def collection_finished(number: int) -> bool:
        # Quarantine hides the tree before its collection journal commits.
        return not (root / "generations" / str(number)).exists() and not any(
            Journal.read(path).incomplete for path in (root / "journal").glob("*.json")
        )

    try:
        start_engine(root, 2, "old", old_port)
        start_engine(root, 3, "main", current_port)
        import time

        assert (root / "generations/7").exists()
        constraints = root / "generations/7/constraints.txt"
        original_constraints = constraints.read_bytes()
        if capture_fault:
            constraints.write_bytes(b"unsealed software must survive failed capture")
        old_tree = {p: p.read_bytes() for p in (root / "generations/7").rglob("*") if p.is_file()}
        protected = {
            p: p.read_bytes()
            for name in (
                "workflows",
                "models",
                "outputs",
                "mounts",
                "cache",
                "journal",
                "generations/1",
                "generations/3",
            )
            for p in (root / name).rglob("*")
            if p.is_file()
        }
        protected.update({root / name: (root / name).read_bytes() for name in ("current", "store")})
        imported: dict[Path, bytes] = {}
        write_document(root / "cleanup.json", {"format": "dinkster.cleanup/1", "automatic": True})
        if capture_fault:
            attempt = root / "runtime/maintenance/cleanup"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if attempt.exists() and any(
                    "generation 7 history capture deferred" in warning
                    for warning in json.loads(attempt.read_bytes())["warnings"]
                ):
                    break
                time.sleep(0.05)
            code, status = _run(root, "status")
            assert code == 0, status
            assert status["cleanup"]["lastAttempt"]["status"] == "deferred", status
            assert any(
                "generation 7 history capture deferred" in warning
                for warning in status["cleanup"]["lastAttempt"]["warnings"]
            ), status
            assert {p: p.read_bytes() for p in old_tree} == old_tree
            assert {p: p.read_bytes() for p in protected} == protected
            assert (root / "generations/2").is_dir()
            assert {entry["generation"] for entry in history(root)} == {1}
            imported = {p: p.read_bytes() for p in (root / "history").glob("*.json")}
            constraints.write_bytes(original_constraints)
        deadline = time.monotonic() + 20
        while not collection_finished(7) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert collection_finished(7)
        if capture_fault:
            assert {p: p.read_bytes() for p in imported} == imported
        assert (root / "generations/2").is_dir()
        with urlopen(f"http://127.0.0.1:{old_port}/api/health", timeout=5) as response:
            assert response.status == 200
        stop_engine(root, "old", generation=2)
        if sys.platform == "linux" or os.name == "nt":
            deadline = time.monotonic() + 20
            while not collection_finished(2) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert collection_finished(2)
        else:
            assert (root / "generations/2").is_dir()
        assert (root / "generations/1").is_dir()
        assert {p: p.read_bytes() for p in protected} == protected
        if legacy:
            assert {entry["generation"] for entry in history(root)} == {1, 2, 7}
            assert all(entry["kind"] == "import" for entry in history(root))
            for entry in history(root):
                validate_import(root, entry)
        with urlopen(f"http://127.0.0.1:{current_port}/api/health", timeout=5) as response:
            assert response.status == 200
    finally:
        stop_engine(root, "old", generation=2)
        stop_engine(root, "main", generation=3)
    if legacy:
        assert {entry["generation"] for entry in history(root)} == {1, 2, 3, 7}
        for entry in history(root):
            validate_import(root, entry)


@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
def test_acknowledged_shutdown_collects_without_periodic_maintenance(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from test_environment_runtime import free_port

    root = cleanup_project
    launch = subprocess.Popen
    script = (
        "from dinkster_supervisor import environment_collection as c, supervisor\n"
        "async def disabled(root, stopped):\n"
        "    await stopped.wait()\n"
        "c.maintain_generations=disabled\n"
        "supervisor.main()\n"
    )

    def no_periodic_maintenance(command, **kwargs):
        return launch([command[0], "-c", script, *command[3:]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", no_periodic_maintenance)
    write_document(root / "cleanup.json", {"format": "dinkster.cleanup/1", "automatic": False})
    try:
        start_engine(root, 2, "old", free_port())
        assert (root / "generations/2").is_dir()
        write_document(root / "cleanup.json", {"format": "dinkster.cleanup/1", "automatic": True})
        stop_engine(root, "old", generation=2)
        if sys.platform == "linux" or os.name == "nt":
            assert not (root / "generations/2").exists()
            assert not (root / "generations/7").exists()
        else:
            assert (root / "generations/2").is_dir()
        assert (root / "generations/1").is_dir()
        assert (root / "generations/3").is_dir()
    finally:
        stop_engine(root, "old", generation=2)


def test_repeated_updates_collect_old_trees_and_restore_their_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor.environment_history import history
    from dinkster_supervisor.environment_state import interpreter
    from test_environment_runtime import free_port

    from tests.fixtures.environments.build import build, executable_release

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    second = tmp_path / "other-project"
    setup = fixture["commands"]["setup"]
    code, installed = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, installed
    try:
        second_setup = setup[setup.index("setup") : -1]
        second_setup[second_setup.index("--port") + 1] = str(free_port())
        code, other_installed = _run(second, *second_setup)
        assert code == 0 and other_installed["current"] == 1, other_installed
        assert Path((second / "store").read_text().strip()) == store
        other_engine = _run(second, "status")[1]["engines"][0]
        assert other_engine["state"] == "ready", other_engine
        other_history = history(second)
        protected = {}
        for path in (
            second / "project.json",
            second / "runtime/1/main.json",
            *(second / "history").glob("*.json"),
        ):
            protected[path] = path.read_bytes()
        for name in ("workflows", "models", "outputs", "mounts"):
            directory = root / name
            directory.mkdir(exist_ok=True)
            protected[directory / "sentinel"] = name.encode("ascii") + b" private data"
        cache = store / "uv-cache/sentinel"
        cache.parent.mkdir(exist_ok=True)
        protected[cache] = b"explicit cache GC only"
        for path, data in protected.items():
            path.write_bytes(data)
        indexes = {
            store / name / ".objects.json": json.loads(
                (store / name / ".objects.json").read_bytes()
            )["objects"]
            for name in ("wheels", "engines")
        }
        protected.update(
            {
                path: path.read_bytes()
                for name in ("wheels", "engines")
                for path in (store / name).rglob("*")
                if path.is_file() and path not in indexes
            }
        )
        base = json.loads(Path(fixture["baseManifest"]).read_bytes())
        for number, value in ((2, 59), (3, 73), (4, 91)):
            preserved = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
            if number == 2:
                (mirror / "channels/main.json").write_bytes(
                    Path(fixture["updatedChannel"]).read_bytes()
                )
            else:
                executable_release(mirror, base, f"{number}.0", value)
            code, updated = _run(root, "update", "--yes")
            assert code == 0 and updated["current"] == number, updated
            assert sorted(path.name for path in (root / "generations").iterdir()) == [
                str(number - 1),
                str(number),
            ]
            assert all(path.read_bytes() == data for path, data in preserved.items())
            assert len(history(root)) == number
            assert not [
                str(path.relative_to(tmp_path))
                for path, data in protected.items()
                if path.read_bytes() != data
            ]
            for path, objects in indexes.items():
                document = json.loads(path.read_bytes())
                assert document["format"] == "dinkster.store-paths/1"
                assert objects.items() <= document["objects"].items()
                indexes[path] = document["objects"]
            assert history(second) == other_history
            assert _run(second, "status")[1]["engines"] == [other_engine]
            with urlopen(f"{other_engine['url']}/api/health", timeout=5) as response:
                assert json.load(response) == {"ok": True}
        entries = history(root)
        preserved = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
        executable_release(mirror, base, "5.0", 113)
        channel = json.loads((mirror / "channels/main.json").read_bytes())
        release = mirror / "engine" / channel["engine"]["release"][7:]
        next(release.glob("*.whl")).write_bytes(b"corrupt next release")
        code, failed = _run(root, "update", "--yes")
        assert code == 1 and "corrupt or short data" in failed["error"], failed
        assert (root / "current").read_text() == "4\n" and history(root) == entries
        assert sorted(path.name for path in (root / "generations").iterdir()) == ["3", "4"]
        assert not (root / "generations/1").exists()
        code, restored = _run(root, "restore", "1", "--yes")
        assert code == 0 and restored["current"] == 5 and restored["exact"] is True, restored
        assert sorted(path.name for path in (root / "generations").iterdir()) == ["4", "5"]
        assert all(path.read_bytes() == data for path, data in preserved.items())
        assert not [
            str(path.relative_to(tmp_path))
            for path, data in protected.items()
            if path.read_bytes() != data
        ]
        for path, objects in indexes.items():
            document = json.loads(path.read_bytes())
            assert document["format"] == "dinkster.store-paths/1"
            assert objects.items() <= document["objects"].items()
        assert history(root)[-1]["restoredFrom"] == 1
        assert history(root)[-1]["snapshot"] == entries[0]["snapshot"]
        assert (
            subprocess.check_output(
                [
                    str(interpreter(root / "generations/5")),
                    "-B",
                    "-c",
                    "import fixture_engine; print(fixture_engine.VALUE)",
                ]
            ).strip()
            == b"41"
        )
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        assert history(second) == other_history
        assert _run(second, "status")[1]["engines"] == [other_engine]
        assert sorted(path.name for path in (second / "generations").iterdir()) == ["1"]
        with urlopen(f"{other_engine['url']}/api/health", timeout=5) as response:
            assert json.load(response) == {"ok": True}
        assert (
            subprocess.check_output(
                [
                    str(interpreter(second / "generations/1")),
                    "-B",
                    "-c",
                    "import fixture_engine; print(fixture_engine.VALUE)",
                ]
            ).strip()
            == b"41"
        )
    finally:
        assert _run(root, "stop", "--yes")[0] == 0
        if (second / "current").exists():
            assert _run(second, "stop", "--yes")[0] == 0


def test_pending_launch_without_runtime_binding_keeps_its_generation(tmp_path: Path):
    from dinkster_supervisor.environment_process import cancel_pending_process, prepare_process

    for number in (1, 2, 3):
        _generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 2, 3, "complete").write(tmp_path / "journal/100.json")
    instance = "unpublished-launch"
    lifecycle = tmp_path / "runtime/1/old.process" / hashlib.sha256(instance.encode()).hexdigest()
    prepare_process(lifecycle, instance)
    code, refused = _run(tmp_path, "gc", "--yes")
    assert code == 1 and "child cleanup is unproven" in refused["error"], refused
    assert (tmp_path / "generations/1/generation.json").is_file()
    assert not cancel_pending_process(lifecycle, instance)
    code, applied = _run(tmp_path, "gc", "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    assert not (tmp_path / "generations/1").exists()


@pytest.mark.parametrize("pending_attempt", [False, True])
def test_inherited_stopped_assignment_does_not_pin_an_unused_generation(
    tmp_path: Path, pending_attempt: bool
):
    from dinkster_supervisor.environment_process import cancel_pending_process, prepare_process
    from dinkster_supervisor.environment_runtime import engine_status, start_engine, stop_engine
    from dinkster_supervisor.environment_transactions import activate, begin_activation
    from test_environment_runtime import free_port
    from test_environment_transactions import _target

    _target(tmp_path, 1)
    port = free_port()
    start_engine(tmp_path, 1, "parked", port)
    stop_engine(tmp_path, "parked", generation=1)
    instance = "interrupted-target-launch"
    lifecycle = (
        tmp_path / "runtime/2/parked.process" / hashlib.sha256(instance.encode()).hexdigest()
    )
    if pending_attempt:
        prepare_process(lifecycle, instance)
    for number in (2, 3, 4):
        path = begin_activation(tmp_path, number, "update")
        _target(tmp_path, number)
        activate(tmp_path, path, revalidate=lambda: None)
        rows = engine_status(tmp_path)
        assert [(row["generation"], row["state"], row["url"]) for row in rows] == [
            (number, "stopped", f"http://127.0.0.1:{port}")
        ]
    if pending_attempt:
        code, refused = _run(tmp_path, "gc", "--yes")
        assert code == 1 and "child cleanup is unproven" in refused["error"], refused
        assert all((tmp_path / "generations" / str(number)).is_dir() for number in (1, 2, 3, 4))
        assert not cancel_pending_process(lifecycle, instance)
    code, applied = _run(tmp_path, "gc", "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    assert not (tmp_path / "generations/1").exists()
    assert not (tmp_path / "generations/2").exists()
    assert (tmp_path / "generations/3").is_dir()
    assert (tmp_path / "generations/4").is_dir()
    try:
        row = start_engine(tmp_path, 4, "parked", port)
        assert row == {
            "name": "parked",
            "generation": 4,
            "state": "ready",
            "url": f"http://127.0.0.1:{port}",
        }
    finally:
        stop_engine(tmp_path, "parked", generation=4)


@pytest.mark.parametrize("command", ["gc", "cleanup"])
def test_collection_defers_while_an_old_generation_is_being_acquired(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, command: str
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_state import EnvironmentError

    tmp_path = cleanup_project
    if command == "gc":
        (tmp_path / "store").unlink()
    protected = {
        p: p.read_bytes()
        for name in ("generations", "history", "workflows", "models", "outputs", "mounts", "cache")
        for p in (tmp_path / name).rglob("*")
        if p.is_file()
    }
    entered, release = Event(), Event()
    prepare = runtime.prepare_process
    launch = subprocess.Popen

    def acquiring(directory, instance):
        entered.set()
        assert release.wait(20), "test did not release the acquiring engine"
        prepare(directory, instance)

    def cancel_launch(command, **kwargs):
        if "dinkster_supervisor.supervisor" in command:
            raise OSError("fixture did not launch a process")
        return launch(command, **kwargs)

    monkeypatch.setattr(runtime, "prepare_process", acquiring)
    monkeypatch.setattr(runtime.subprocess, "Popen", cancel_launch)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(runtime.start_engine, tmp_path, 2, "old", 3639)
        try:
            assert entered.wait(20), "engine did not reach acquisition"
            code, refused = _run(tmp_path, command, "--yes")
            if command == "cleanup":
                assert code == 0 and refused["status"] == "deferred", refused
                assert (
                    "another environment writer holds" in refused["cleanupAttempt"]["warnings"][0]
                )
            else:
                assert code == 1 and "another environment writer holds" in refused["error"], refused
            assert {p: p.read_bytes() for p in protected} == protected
            assert not list((tmp_path / ".garbage").glob("*"))
        finally:
            release.set()
        with pytest.raises(OSError, match="fixture did not launch"):
            pending.result(timeout=20)
    code, applied = _run(tmp_path, command, "--yes")
    assert code == 0 and applied["status"] == ("complete" if command == "cleanup" else "applied"), (
        applied
    )
    assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["1", "3"]
    with pytest.raises(EnvironmentError, match="cannot read generation.json"):
        runtime.start_engine(tmp_path, 2, "old", 3639)


@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
@pytest.mark.parametrize("command", ["gc", "cleanup"])
@pytest.mark.parametrize("operation", ["update", "restore"])
def test_collection_defers_during_activation_and_rejects_its_old_retention_plan(
    cleanup_project: Path, command: str, operation: str
):
    from dinkster_supervisor.environment_collection import collect_generations, collection_plan
    from dinkster_supervisor.environment_state import EnvironmentError, read_current, writer_lock
    from dinkster_supervisor.environment_transactions import activate, begin_activation
    from test_environment_transactions import _target

    tmp_path = cleanup_project
    if command == "gc":
        (tmp_path / "store").unlink()
    approved = collection_plan(tmp_path)
    entered, release = Event(), Event()

    def revalidate():
        entered.set()
        assert release.wait(20), "test did not release activation"

    def activation():
        with writer_lock(tmp_path):
            path = begin_activation(
                tmp_path, 4, operation, restored_from=1 if operation == "restore" else None
            )
            _target(tmp_path, 4)
            activate(tmp_path, path, revalidate=revalidate)

    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(activation)
        try:
            assert entered.wait(20), "activation did not reach validation"
            lock = tmp_path / ".lock"
            assert lock.is_file()
            maintenance = tmp_path / "runtime/maintenance"
            before = {
                p: p.read_bytes()
                for p in tmp_path.rglob("*")
                if p.is_file() and p != lock and not p.is_relative_to(maintenance)
            }
            code, refused = _run(tmp_path, command, "--yes")
            if command == "cleanup":
                assert code == 0 and refused["status"] == "deferred", refused
                assert (
                    "another environment writer holds" in refused["cleanupAttempt"]["warnings"][0]
                )
            else:
                assert code == 1 and "another environment writer holds" in refused["error"], refused
            assert {
                p: p.read_bytes()
                for p in tmp_path.rglob("*")
                if p.is_file() and p != lock and not p.is_relative_to(maintenance)
            } == before
            assert lock.is_file()
            assert not list((tmp_path / ".garbage").glob("*"))
        finally:
            release.set()
        pending.result(timeout=20)
    assert read_current(tmp_path) == 4
    with writer_lock(tmp_path):
        with pytest.raises(EnvironmentError, match="collection changed after planning"):
            collect_generations(tmp_path, approved)
    assert all((tmp_path / "generations" / str(number)).is_dir() for number in (1, 2, 3, 4, 7))
    code, collected = _run(tmp_path, command, "--yes")
    assert code == 0 and collected["status"] == (
        "complete" if command == "cleanup" else "applied"
    ), collected
    assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["3", "4"]


@pytest.mark.parametrize("cleanup_project", [True], indirect=True)
@pytest.mark.parametrize("command", ["gc", "cleanup"])
def test_collection_defers_during_move_without_touching_source_generations(
    cleanup_project: Path, monkeypatch: pytest.MonkeyPatch, command: str
):
    from dinkster_supervisor import environment_move as move
    from dinkster_supervisor.environment_stores import register_project

    root = cleanup_project
    for name in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(root.parent / (root.name + "-user-data")))
    store, destination = (
        root.parent / (root.name + suffix) for suffix in ("-store", "-destination")
    )
    store.mkdir()
    (root / "store").write_text(str(store) + "\n")
    register_project(root, store)
    entered, release = Event(), Event()

    def interrupt(plan, path, journal):
        entered.set()
        assert release.wait(20), "test did not release move"
        raise OSError("fixture interrupted destination preparation")

    monkeypatch.setattr(move, "_prepare", interrupt)
    plan = move.plan_move(root, destination, store)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(move.execute_move, plan)
        try:
            assert entered.wait(20), "move did not reach destination preparation"
            lock = root / ".lock"
            assert lock.is_file()
            maintenance = root / "runtime/maintenance"
            before = {
                p: p.read_bytes()
                for p in root.rglob("*")
                if p.is_file() and p != lock and not p.is_relative_to(maintenance)
            }
            code, refused = _run(root, command, "--yes")
            if command == "cleanup":
                assert code == 0 and refused["status"] == "deferred", refused
                assert (
                    "another environment writer holds" in refused["cleanupAttempt"]["warnings"][0]
                )
            else:
                assert code == 1 and "another environment writer holds" in refused["error"], refused
            assert {
                p: p.read_bytes()
                for p in root.rglob("*")
                if p.is_file() and p != lock and not p.is_relative_to(maintenance)
            } == before
            assert lock.is_file()
            assert not list((root / ".garbage").glob("*"))
        finally:
            release.set()
        with pytest.raises(OSError, match="interrupted destination preparation"):
            pending.result(timeout=20)
    assert (root / "current").read_text() == "3\n"
    assert all((root / "generations" / str(number)).is_dir() for number in (1, 2, 3, 7))
    assert not destination.exists()
    assert all(not Journal.read(path).incomplete for path in (root / "journal").glob("*.json"))
    if command == "gc":
        (root / "store").unlink()
    code, collected = _run(root, command, "--yes")
    assert code == 0 and collected["status"] == (
        "complete" if command == "cleanup" else "applied"
    ), collected
    assert sorted(p.name for p in (root / "generations").iterdir()) == ["1", "3"]


@pytest.mark.parametrize(
    "interruption", [None, "quarantine", "cleanup", "missing-project", "corrupt-constraints"]
)
@pytest.mark.parametrize("readable", [False, True])
def test_shared_store_collection_keeps_all_registered_references_and_recovers_from_another_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str | None, readable: bool
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user-data"))
    store = tmp_path / "store"
    paths: dict[tuple[str, str], Path] = {}

    def cache(category: str, identity: str) -> Path:
        key = (category, identity)
        if key not in paths:
            label = f"2026-11-03-1.0-{identity}"
            if category in {"bases", "downloads"}:
                label = "linux-cpu/" + label
            paths[key] = object_path(
                store / category, "sha256:" + identity * 64, label=label if readable else None
            )
        return paths[key]

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
            directory = cache(category, identity)
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
        extra = cache(category, "7")
        extra.mkdir(parents=True)
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
                assert (cache(category, "7") / "payload").read_bytes() == b"unreferenced"
            assert (first / "generations/2").is_dir()
    code, applied = _run(first, "gc", "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    assert not (first / "generations/2").exists()
    for category in ("wheels", "bases", "engines", "downloads"):
        assert not cache(category, "7").exists()
    for project, number, base, engine, wheel in references:
        if project == first and number == 2:
            continue
        for category, identity in (("bases", base), ("engines", engine), ("wheels", wheel)):
            assert (cache(category, identity) / "payload").read_bytes() == identity.encode()
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
    assert code == 1 and "no previous successful history entry" in rollback["error"], rollback
    assert (tmp_path / "current").read_bytes() == b"3\n"


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
