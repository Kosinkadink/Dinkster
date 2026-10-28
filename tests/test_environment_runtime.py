from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest
from dinkster_supervisor.environment_process import cancel_pending_process, prepare_process
from dinkster_supervisor.environment_runtime import (
    engine_assignments,
    engine_status,
    require_idle,
    start_engine,
    stop_engine,
)
from dinkster_supervisor.environment_state import EnvironmentError

SERVE = """
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--generation', required=True)
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/health':
            body = {'ok': True}
        elif self.path == '/api/queue':
            with open('queue.json') as source:
                body = json.load(source)
        elif self.path == '/api/cuda':
            body = {'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}
        elif self.path == '/api/generation':
            body = {'generation': args.generation}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_generation(root: Path, number: int = 1) -> None:
    generation = root / "generations" / str(number)
    python = generation / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
    python.parent.mkdir(parents=True)
    if os.name == "nt":
        python.write_bytes(Path(sys.executable).read_bytes())
    else:
        python.symlink_to(sys.executable)
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "generation": number,
                "base": "sha256:" + "a" * 64,
                "engine": {"version": "test", "release": "sha256:" + "b" * 64},
                "constraints": "sha256:" + "c" * 64,
                "variant": "linux-cpu",
                "workerProtocol": 1,
                "files": {"constraints.txt": "sha256:" + "c" * 64},
            }
        )
    )
    (generation / "constraints.txt").write_text("test\n")
    package = root / "dinkster"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "serve.py").write_text(SERVE)
    (root / "queue.json").write_text('{"running": [], "queued": []}')
    if not (root / "current").exists():
        (root / "current").write_text(f"{number}\n")


def test_start_status_idempotence_and_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dinkster_supervisor.environment_runtime.secrets.token_urlsafe", lambda _: "-token"
    )
    make_generation(tmp_path)
    port = free_port()
    row = start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12"))
    try:
        assert row == {
            "name": "main",
            "generation": 1,
            "state": "ready",
            "url": f"http://127.0.0.1:{port}",
        }
        with urlopen(f"http://127.0.0.1:{port}/api/cuda") as response:  # noqa: S310
            assert json.loads(response.read()) == {"cuda": "0,GPU-aB12"}
        with urlopen(f"http://127.0.0.1:{port}/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "generations/1")}
        assert engine_status(tmp_path) == (row,)
        assert start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12")) == row
        with pytest.raises(EnvironmentError, match="stop it first"):
            start_engine(tmp_path, 1, "main", free_port())
        require_idle(tmp_path)
    finally:
        stop_engine(tmp_path, "main")
    assert engine_status(tmp_path)[0]["state"] == "stopped"


def test_stop_refuses_wrong_instance(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    record_path = tmp_path / "runtime" / "1" / "main.json"
    original = json.loads(record_path.read_text())
    record = dict(original)
    record["instance"] = "not-the-supervisor"
    record_path.write_text(json.dumps(record))
    try:
        with pytest.raises(EnvironmentError, match="refusing to stop"):
            stop_engine(tmp_path, "main")
    finally:
        record_path.write_text(json.dumps(original))
        stop_engine(tmp_path, "main")


def test_status_rejects_malformed_record_without_mutating(tmp_path: Path) -> None:
    record_path = tmp_path / "runtime" / "1" / "main.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}")

    with pytest.raises(EnvironmentError, match="runtime record"):
        engine_status(tmp_path)

    assert record_path.read_text() == "{}"


@pytest.mark.parametrize("live", [False, True])
def test_unverified_supervisor_exit_does_not_prove_idle(tmp_path: Path, live: bool) -> None:
    make_generation(tmp_path)
    with subprocess.Popen([sys.executable, "-c", "pass"]) as exited:
        assert exited.wait(timeout=10) == 0
    runtime = tmp_path / "runtime" / "1"
    runtime.mkdir(parents=True)
    record = {
        "name": "main",
        "generation": 1,
        "port": free_port(),
        "gpus": [],
        "instance": "original",
        "shutdownToken": "private",
        "pid": os.getpid() if live else exited.pid,
        "state": "stopped" if live else "ready",
    }
    record["url"] = f"http://127.0.0.1:{record['port']}"
    (runtime / "main.json").write_text(json.dumps(record))
    assert engine_status(tmp_path)[0]["state"] == "failed"
    with pytest.raises(EnvironmentError, match="cannot verify"):
        require_idle(tmp_path)
    with pytest.raises(EnvironmentError, match="not safely reachable"):
        start_engine(tmp_path, 1, "main", record["port"])


def test_zero_pid_recovers_identity_for_authenticated_stop(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    record_path = tmp_path / "runtime" / "1" / "main.json"
    record = json.loads(record_path.read_text())
    record["pid"] = 0
    record_path.write_text(json.dumps(record))
    try:
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert json.loads(record_path.read_text())["pid"] == 0
    finally:
        stop_engine(tmp_path, "main")


def test_require_idle_refuses_running_or_malformed_queue(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    try:
        (tmp_path / "queue.json").write_text('{"running": [{"jobId": "one"}], "queued": []}')
        with pytest.raises(EnvironmentError, match="Finish or cancel"):
            require_idle(tmp_path)
        (tmp_path / "queue.json").write_text('{"not": "a queue"}')
        with pytest.raises(EnvironmentError, match="cannot verify"):
            require_idle(tmp_path)
    finally:
        stop_engine(tmp_path, "main")


def test_same_name_generations_coexist_and_current_selects_status(tmp_path: Path) -> None:
    make_generation(tmp_path, 1)
    make_generation(tmp_path, 2)
    first_port, second_port = free_port(), free_port()
    first = start_engine(tmp_path, 1, "main", first_port)
    try:
        second = start_engine(tmp_path, 2, "main", second_port, ("GPU-aB12",))
        try:
            assert engine_status(tmp_path) == (first,)
            assert engine_assignments(tmp_path, 2) == [
                {"name": "main", "port": second_port, "gpus": ["GPU-aB12"], "running": True}
            ]
            require_idle(tmp_path)
            (tmp_path / "current").write_text("2\n")
            assert engine_status(tmp_path) == (second,)
            stop_engine(tmp_path, "main", generation=1)
            assert engine_status(tmp_path) == (second,)
            assert engine_assignments(tmp_path, 1) == [
                {"name": "main", "port": first_port, "gpus": [], "running": False}
            ]
        finally:
            stop_engine(tmp_path, "main", generation=2)
    finally:
        stop_engine(tmp_path, "main", generation=1)


def test_cancelled_delayed_supervisor_cannot_spawn_its_engine(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle"
    prepare_process(lifecycle, "cancelled-launch")
    assert not cancel_pending_process(lifecycle, "cancelled-launch")
    sentinel = tmp_path / "engine-started"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.supervisor",
            "--instance=cancelled-launch",
            "--process-state",
            str(lifecycle),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(sentinel),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()


def test_failed_candidate_listener_proves_cleanup_and_can_restart(tmp_path: Path) -> None:
    make_generation(tmp_path, 1)
    make_generation(tmp_path, 2)
    port = free_port()
    prior = start_engine(tmp_path, 1, "main", port)
    try:
        with pytest.raises(EnvironmentError, match="before becoming ready"):
            start_engine(tmp_path, 2, "main", port)
        stop_engine(tmp_path, "main", generation=2)
        assert engine_status(tmp_path) == (prior,)
        candidate = start_engine(tmp_path, 2, "main", free_port())
        assert candidate["generation"] == 2 and candidate["state"] == "ready"
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)
