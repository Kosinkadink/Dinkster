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
from urllib.request import Request, urlopen

import pytest
from dinkster_supervisor.environment_state import (
    Generation,
    Journal,
    content_digest,
    write_document,
)
from test_environment_cli import _run
from test_environment_state import _generation


@pytest.mark.parametrize("worker_kind", ["isolated", "grouped", "descendant"])
def test_supervisor_stop_proves_blocked_import_worker_exit_before_collection(
    tmp_path: Path, worker_kind: str
):
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from test_environment_runtime import free_port, make_generation

    for number in (1, 2, 3):
        make_generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 2, 3, "complete").write(tmp_path / "journal/100.json")
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
            start_engine(tmp_path, 1, "old", port)
            connection, _ = listener.accept()
            connection.settimeout(10)
            assert connection.recv(5) == b"ready"
            code, refused = _run(tmp_path, "gc", "--yes")
            assert code == 1 and "cleanup is unproven" in refused["error"], refused
            connection.sendall(b"p")
            assert connection.recv(1) == b"p"
            assert (tmp_path / "generations/1").is_dir()
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
                code, refused = _run(tmp_path, "gc", "--yes")
                assert code == 1 and "cleanup is unproven" in refused["error"], refused
                connection.sendall(b"p")
                assert connection.recv(1) == b"p"
            stop_engine(tmp_path, "old", generation=1)
            if sys.platform != "linux" and os.name != "nt":
                code, refused = _run(tmp_path, "gc", "--yes")
                assert code == 1 and "descendant cleanup is unproven" in refused["error"], refused
                assert (tmp_path / "generations/1").is_dir()
                return
            try:
                assert connection.recv(1) == b""
            except ConnectionResetError:
                pass
            code, applied = _run(tmp_path, "gc", "--yes")
            assert code == 0 and applied["status"] == "applied", applied
            assert not (tmp_path / "generations/1").exists()
            assert all((tmp_path / "generations" / str(number)).is_dir() for number in (2, 3))
        finally:
            stop_engine(tmp_path, "old", generation=1)
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


def test_failed_owned_family_cleanup_cannot_make_generation_collectible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_state import EnvironmentError
    from test_environment_runtime import free_port, make_generation

    for number in (1, 2, 3):
        make_generation(tmp_path, number)
    (tmp_path / "current").write_text("3\n")
    Journal("update", 2, 3, "complete").write(tmp_path / "journal/100.json")
    original = subprocess.Popen
    processes: list[subprocess.Popen] = []
    script = (
        "from dinkster_supervisor import supervisor\n"
        "from dinkster_supervisor.process_family import ProcessFamily\n"
        "def refuse(self): raise RuntimeError('owned cleanup fixture failure')\n"
        "ProcessFamily.stop_children=refuse\n"
        "supervisor.main()\n"
    )

    def launch(command, **kwargs):
        process = original([command[0], "-c", script, *command[3:]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    try:
        runtime.start_engine(tmp_path, 1, "old", free_port())
        with pytest.raises(EnvironmentError, match="without proving child cleanup"):
            runtime.stop_engine(tmp_path, "old", generation=1)
        assert processes[0].wait(timeout=5) != 0
    finally:
        # The injected failure occurs after direct-engine shutdown; only the fixture host remains.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    monkeypatch.undo()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    code, refused = _run(tmp_path, "gc", "--yes")
    assert code == 1 and "cleanup is unproven" in refused["error"], refused
    assert (tmp_path / "generations/1").is_dir()
    assert {p: p.read_bytes() for p in before} == before


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
        assert rows[2]["status"] == ("candidate" if condition is None else "deferred")
        if condition == "ambiguous":
            assert "invalid" in rows[2]["reasons"][0]
            assert rows[7]["status"] == "candidate"
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


def test_collection_defers_while_an_old_generation_is_being_acquired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor import environment_runtime as runtime
    from dinkster_supervisor.environment_state import EnvironmentError, interpreter

    for number in (1, 2, 3):
        _generation(tmp_path, number)
    python = interpreter(tmp_path / "generations/1")
    python.parent.mkdir(exist_ok=True)
    python.write_bytes(b"unused fixture interpreter")
    (tmp_path / "current").write_text("3\n")
    Journal("update", 2, 3, "complete").write(tmp_path / "journal/100.json")
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
        pending = threads.submit(runtime.start_engine, tmp_path, 1, "old", 3639)
        try:
            assert entered.wait(20), "engine did not reach acquisition"
            code, refused = _run(tmp_path, "gc", "--yes")
            assert code == 1 and "another environment writer holds" in refused["error"], refused
            assert (tmp_path / "generations/1/generation.json").is_file()
            assert not list((tmp_path / ".garbage").glob("*"))
        finally:
            release.set()
        with pytest.raises(OSError, match="fixture did not launch"):
            pending.result(timeout=20)
    code, applied = _run(tmp_path, "gc", "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    assert not (tmp_path / "generations/1").exists()
    assert (tmp_path / "generations/2/generation.json").is_file()
    assert (tmp_path / "generations/3/generation.json").is_file()
    with pytest.raises(EnvironmentError, match="cannot read generation.json"):
        runtime.start_engine(tmp_path, 1, "old", 3639)


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
