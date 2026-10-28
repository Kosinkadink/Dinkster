"""Activation keeps real supervisors recoverable on both sides of current replacement."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import pytest
from dinkster_supervisor.environment_runtime import engine_status, start_engine, stop_engine
from dinkster_supervisor.environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    interpreter,
    read_current,
)
from dinkster_supervisor.environment_transactions import (
    activate,
    begin_activation,
    begin_setup,
    recover_activation,
)
from test_environment_cli import _run
from test_environment_generations import _standalone_tree
from test_environment_runtime import free_port, make_generation


def _target(root: Path, number: int) -> Path:
    make_generation(root, number)
    target = root / "generations" / str(number)
    shutil.rmtree(target / "python")
    _standalone_tree(target / "python")
    generation = Generation.read(target / "generation.json")
    hashes = tuple(
        sorted(
            (path.relative_to(target).as_posix(), content_digest(path))
            for path in target.rglob("*")
            if path.is_file() and path.name != "generation.json"
        )
    )
    replace(generation, constraints=content_digest(target / "constraints.txt"), files=hashes).write(
        target / "generation.json"
    )
    return target


@pytest.mark.parametrize("interruption", [None, "before-switch", "after-switch"])
def test_first_setup_publishes_only_a_healthy_engine_and_recovers_without_previous_generation(
    tmp_path: Path, interruption: str | None
) -> None:
    port = free_port()
    path = begin_setup(tmp_path, 1, port, ("GPU-aB12",), name="compute")
    assert read_current(tmp_path) is None
    assert Journal.read(path).previous is None
    _target(tmp_path, 1)
    (tmp_path / "current").unlink()
    script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_transactions as transactions
root, journal = Path(sys.argv[1]), Path(sys.argv[2])
def interrupted():
    os._exit(71)
if sys.argv[3] == 'after-switch':
    original = transactions.set_current
    def switch_then_interrupt(root, number):
        original(root, number)
        interrupted()
    transactions.set_current = switch_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
else:
    transactions.activate(root, journal, revalidate=interrupted)
"""
    try:
        if interruption is None:
            activate(tmp_path, path, revalidate=lambda: None)
            assert read_current(tmp_path) == 1
            assert engine_status(tmp_path) == (
                {
                    "name": "compute",
                    "generation": 1,
                    "state": "ready",
                    "url": f"http://127.0.0.1:{port}",
                },
            )
            with urlopen(f"http://127.0.0.1:{port}/api/cuda") as response:  # noqa: S310
                assert json.loads(response.read()) == {"cuda": "GPU-aB12"}
            assert Journal.read(path).stage == "complete"
            with pytest.raises(EnvironmentError, match="already has an active"):
                begin_setup(tmp_path, 2, free_port())
        else:
            exited = subprocess.run(
                [sys.executable, "-c", script, str(tmp_path), str(path), interruption],
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert exited.returncode == 71, exited.stderr
            assert Journal.read(path).incomplete
            recover_activation(tmp_path, path)
            assert read_current(tmp_path) is None
            assert engine_status(tmp_path) == (
                {
                    "name": "compute",
                    "generation": 1,
                    "state": "stopped",
                    "url": f"http://127.0.0.1:{port}",
                },
            )
            assert not (tmp_path / "generations/1").exists()
            assert Journal.read(path).stage == "rolled-back"
            recover_activation(tmp_path, path)
            next_path = begin_setup(tmp_path, 1, port)
            assert Journal.read(next_path).incomplete
            recover_activation(tmp_path, next_path)
    finally:
        stop_engine(tmp_path, "compute", generation=1)


def test_update_and_rollback_reactivate_retained_tree_without_rebuild(tmp_path: Path) -> None:
    retained = _target(tmp_path, 1)
    before = (retained / "generation.json").read_bytes()
    identity = interpreter(retained).stat()
    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "main",
                "variant": "linux-cpu",
                "port": free_port(),
                "mirrors": [str(tmp_path / "mirror")],
            }
        )
    )
    start_engine(tmp_path, 1, "main", free_port(), ("GPU-aB12",))
    start_engine(tmp_path, 1, "parked", free_port())
    parked_path = tmp_path / "runtime/1/parked.json"
    stale_parked = parked_path.read_bytes()
    stop_engine(tmp_path, "parked", generation=1)
    parked_path.write_bytes(stale_parked)
    path = begin_activation(tmp_path, 2, "update")
    _target(tmp_path, 2)

    def revalidate() -> None:
        assert read_current(tmp_path) == 1
        binding = next(row for row in Journal.read(path).bindings if row["name"] == "main")
        with urlopen(f"http://127.0.0.1:{binding['targetPort']}/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "generations/2")}

    try:
        activate(tmp_path, path, revalidate=revalidate)
        assert read_current(tmp_path) == 2
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert engine_status(tmp_path)[0]["generation"] == 2
        assert Journal.read(path).stage == "complete"
        assert (tmp_path / "generations/1").is_dir()
        journals = sorted((tmp_path / "journal").glob("*.json"))
        code, planned = _run(tmp_path, "rollback", "--plan")
        assert code == 0 and planned["status"] == "planned"
        assert read_current(tmp_path) == 2
        assert sorted((tmp_path / "journal").glob("*.json")) == journals
        code, applied = _run(tmp_path, "rollback", "--yes")
        assert code == 0, applied
        assert applied["status"] == "applied"
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert engine_status(tmp_path)[1]["state"] == "stopped"
        with urlopen(str(applied["engines"][0]["url"]) + "/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "generations/1")}
        assert (retained / "generation.json").read_bytes() == before
        after = interpreter(retained).stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
            identity.st_ino,
            identity.st_size,
            identity.st_mtime_ns,
        )
        assert (tmp_path / "generations/2").is_dir()
        (tmp_path / "generations/2/constraints.txt").write_text("changed\n")
        code, refused = _run(tmp_path, "rollback", "--yes")
        assert code == 1 and "drift" in refused["error"]
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert (tmp_path / "generations/2").is_dir()
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)


@pytest.mark.parametrize(
    "interruption",
    ["before-spawn", "before-switch", "after-switch", "after-shutdown", "during-recovery"],
)
def test_interrupted_activation_restores_previous_serving_generation(
    tmp_path: Path,
    interruption: str,
) -> None:
    make_generation(tmp_path)
    prior = start_engine(tmp_path, 1, "main", free_port())
    path = begin_activation(tmp_path, 2, "update")
    _target(tmp_path, 2)

    script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_transactions as transactions
from dinkster_supervisor import environment_runtime as runtime
root, journal = Path(sys.argv[1]), Path(sys.argv[2])
def interrupted():
    os._exit(71)
if sys.argv[3] == 'after-switch':
    original = transactions.set_current
    def switch_then_interrupt(root, number):
        original(root, number)
        interrupted()
    transactions.set_current = switch_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
elif sys.argv[3] in {'before-switch', 'during-recovery'}:
    transactions.activate(root, journal, revalidate=interrupted)
elif sys.argv[3] == 'before-spawn':
    runtime.subprocess.Popen = lambda *a, **kw: interrupted()
    transactions.activate(root, journal, revalidate=lambda: None)
else:
    original = runtime._write_record
    def stopped_then_interrupt(root, name, record):
        if record['state'] == 'stopped' and record['generation'] == 1:
            interrupted()
        original(root, name, record)
    runtime._write_record = stopped_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
"""
    try:
        exited = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), str(path), interruption],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert exited.returncode == 71, exited.stderr
        assert Journal.read(path).incomplete
        if interruption == "during-recovery":
            recovery_script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_runtime as runtime
from dinkster_supervisor.environment_transactions import recover_activation
original = runtime._write_record
def stopped_then_interrupt(root, name, record):
    if record['state'] == 'stopped' and record['generation'] == 2:
        os._exit(72)
    original(root, name, record)
runtime._write_record = stopped_then_interrupt
recover_activation(Path(sys.argv[1]), Path(sys.argv[2]))
"""
            exited = subprocess.run(
                [sys.executable, "-c", recovery_script, str(tmp_path), str(path)],
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert exited.returncode == 72, exited.stderr
            assert Journal.read(path).incomplete
        recover_activation(tmp_path, path)
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path) == (prior,)
        assert not (tmp_path / "generations/2").exists()
        assert Journal.read(path).stage == "rolled-back"
        recover_activation(tmp_path, path)
        assert engine_status(tmp_path) == (prior,)
    finally:
        for number in (1, 2):
            with suppress(EnvironmentError):
                stop_engine(tmp_path, "main", generation=number)


def test_busy_refusal_and_failed_final_revalidation_preserve_current(tmp_path: Path) -> None:
    make_generation(tmp_path)
    prior = start_engine(tmp_path, 1, "main", free_port())
    try:
        (tmp_path / "queue.json").write_text('{"running":[{"jobId":"active"}],"queued":[]}')
        with pytest.raises(EnvironmentError, match="Finish or cancel"):
            begin_activation(tmp_path, 2, "update")
        assert not (tmp_path / "journal").exists()
        (tmp_path / "queue.json").write_text(json.dumps({"running": [], "queued": []}))
        path = begin_activation(tmp_path, 2, "update")
        _target(tmp_path, 2)

        def changed() -> None:
            raise EnvironmentError("project changed after confirmation")

        with pytest.raises(EnvironmentError, match="project changed"):
            activate(tmp_path, path, revalidate=changed)
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path) == (prior,)
        assert not (tmp_path / "generations/2").exists()
        assert Journal.read(path).error == "project changed after confirmation"
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)
