from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import venv
from pathlib import Path
from urllib.request import Request, urlopen

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
from test_environment_generations import _workspace_tree

SERVE = """
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--generation', required=True)
parser.add_argument('--library-root')
parser.add_argument('--prepare-stale-catalogs', action='store_true')
parser.add_argument('--allow-mount-changes', action='store_true')
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


def make_generation(root: Path, number: int = 1, *, fake_serve: bool = True) -> None:
    generation = root / "generations" / str(number)
    site = _workspace_tree(generation / "python")
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
    if fake_serve:
        package = root / "dinkster"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "serve.py").write_text(SERVE)
        (site / "000-project.pth").write_text(str(root.resolve()) + "\n")
    (root / "queue.json").write_text('{"running": [], "queued": []}')
    if not (root / "current").exists():
        (root / "current").write_text(f"{number}\n")


@pytest.mark.parametrize("custom_library", [False, True])
def test_managed_generation_command_uses_local_project_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_library: bool
) -> None:
    root = tmp_path / "project"
    make_generation(root)
    library = tmp_path / "chosen library" if custom_library else root / "dinkster-library"
    if custom_library:
        monkeypatch.setenv("DINKSTER_LIBRARY_ROOT", str(library))
    else:
        monkeypatch.delenv("DINKSTER_LIBRARY_ROOT", raising=False)
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        raise OSError("captured launch")

    monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    with pytest.raises(OSError, match="captured launch"):
        start_engine(root, 1, "main", free_port())
    command = commands[0]
    assert command[command.index("--") + 1 :] == [
        str(root / "generations/1/python" / ("python.exe" if os.name == "nt" else "bin/python3")),
        "-I",
        "-B",
        "-m",
        "dinkster.serve",
        "--generation",
        str(root / "generations/1"),
        "--library-root",
        str(library),
        "--prepare-stale-catalogs",
        "--allow-mount-changes",
    ]


@pytest.mark.parametrize("host_contamination", [False, True])
def test_managed_generation_persists_folder_grant_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_contamination: bool
) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog
    from test_serve import write_shutdown_pack

    from dinkster.generation_packs import stage_generation_packs

    monkeypatch.delenv("DINKSTER_LIBRARY_ROOT", raising=False)
    root = tmp_path / "project"
    generation = root / "generations/1"
    generation.mkdir(parents=True)
    manifest = write_shutdown_pack(tmp_path / "source", 57)
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    pack = "shutdown-pack-57"
    lock = Lockfile.of([LockedPack(pack, "1.2.3", digest, "dinkster", ("shutdown57",))])
    stage_generation_packs(generation, lock, {pack: archive})
    source = generation / "pack-sources" / pack
    assert prepare_catalog(
        source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
    ).ok
    make_generation(root, fake_serve=False)
    if host_contamination:
        shadow = root / "dinkster"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("raise RuntimeError('host engine shadow imported')\n")
        monkeypatch.setenv("PYTHONPATH", str(root))
    record = json.loads((generation / "generation.json").read_text())
    record["inProcessPacks"] = [pack]
    (generation / "generation.json").write_text(json.dumps(record))
    granted = tmp_path / "user files"
    granted.mkdir()
    port = free_port()
    url = f"http://127.0.0.1:{port}/api/mounts"
    try:
        start_engine(root, 1, "main", port)
        with urlopen(url, timeout=5) as response:  # noqa: S310
            initial = json.load(response)
        assert initial["mountChangesAllowed"] is True
        assert {row["id"] for row in initial["mounts"]} == {"imports", "output"}
        assert initial["outputMount"] == "output"
        request = Request(
            url,
            data=json.dumps({"id": "chosen", "path": str(granted)}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            assert response.status == 201
            assert json.load(response)["mode"] == "read"
        stop_engine(root, "main")
        start_engine(root, 1, "main", port)
        with urlopen(url, timeout=5) as response:  # noqa: S310
            listing = json.load(response)
        assert listing["mountChangesAllowed"] is True
        selected = next(row for row in listing["mounts"] if row["id"] == "chosen")
        assert selected["path"] == str(granted)
        assert selected["mode"] == "read"
        assert listing["restartRequired"] is False
    finally:
        stop_engine(root, "main")


@pytest.mark.parametrize(
    ("redirecting_launcher", "health_body_timeout"),
    [(False, False), (True, False), (False, True)],
)
def test_start_status_idempotence_and_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redirecting_launcher: bool,
    health_body_timeout: bool,
) -> None:
    monkeypatch.setattr(
        "dinkster_supervisor.environment_runtime.secrets.token_urlsafe", lambda _: "-token"
    )
    if health_body_timeout:
        original_read = http.client.HTTPResponse.read

        def read(response, *args, **kwargs):
            if response.geturl().endswith("/api/health"):
                monkeypatch.setattr(http.client.HTTPResponse, "read", original_read)
                raise TimeoutError("health response body stalled")
            return original_read(response, *args, **kwargs)

        monkeypatch.setattr(http.client.HTTPResponse, "read", read)
    if redirecting_launcher:
        launcher = tmp_path / "launcher.py"
        launcher.write_text(
            "import subprocess, sys\n"
            "with subprocess.Popen([sys.executable, *sys.argv[1:]]) as child:\n"
            "    sys.exit(child.wait())\n"
        )
        original = subprocess.Popen

        def launch(command, **kwargs):
            return original([command[0], str(launcher), *command[1:]], **kwargs)

        monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    make_generation(tmp_path)
    port = free_port()
    try:
        row = start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12"))
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
        with urlopen(f"http://127.0.0.1:{port}/supervisor/status") as response:  # noqa: S310
            supervisor_pid = json.loads(response.read())["supervisor"]["pid"]
        persisted = json.loads((tmp_path / "runtime/1/main.json").read_bytes())
        assert persisted["pid"] == supervisor_pid and supervisor_pid > 0
        assert engine_status(tmp_path) == (row,)
        assert start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12")) == row
        with pytest.raises(EnvironmentError, match="stop it first"):
            start_engine(tmp_path, 1, "main", free_port())
        require_idle(tmp_path)
    finally:
        stop_engine(tmp_path, "main")
    assert engine_status(tmp_path)[0]["state"] == "stopped"


def test_start_engine_uses_generation_built_from_a_virtual_environment(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(launcher)
    python = launcher / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    root = tmp_path / "project"
    subprocess.run(
        [
            str(python),
            "-c",
            "import sys; from pathlib import Path; "
            "from test_environment_runtime import make_generation; "
            "make_generation(Path(sys.argv[1]))",
            str(root),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(str(Path(path).resolve()) for path in sys.path if path),
        },
        check=True,
        capture_output=True,
        timeout=30,
    )
    start_engine(root, 1, "main", free_port())
    try:
        assert engine_status(root)[0]["state"] == "ready"
        with urlopen(str(engine_status(root)[0]["url"]) + "/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(root / "generations/1")}
    finally:
        stop_engine(root, "main")
    assert engine_status(root)[0]["state"] == "stopped"


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


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM cleanup is a POSIX contract")
def test_sigterm_stops_managed_engine_and_allows_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_popen = subprocess.Popen
    supervisors: list[subprocess.Popen] = []

    def launch(command, **kwargs):
        process = original_popen(command, **kwargs)
        if command[1:3] == ["-m", "dinkster_supervisor.supervisor"]:
            supervisors.append(process)
        return process

    def process_field(pid: int, field: str) -> str:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in {0, 1}
        return result.stdout.strip()

    def child_alive(pid: int) -> bool:
        state = process_field(pid, "stat")
        return bool(state) and not state.startswith("Z")

    monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    make_generation(tmp_path)
    port = free_port()
    child_pid = 0
    child_command = ""
    try:
        row = start_engine(tmp_path, 1, "main", port)
        record = json.loads((tmp_path / "runtime/1/main.json").read_text())
        with urlopen(f"http://127.0.0.1:{port}/supervisor/status") as response:  # noqa: S310
            status = json.loads(response.read())
        assert status["instance"] == record["instance"]
        assert status["supervisor"]["pid"] == record["pid"] == supervisors[0].pid
        child_pid = status["engine"]["pid"]
        child_command = process_field(child_pid, "args")
        assert str(tmp_path / "generations/1") in child_command
        lifecycle = next((tmp_path / "runtime/1/main.process").glob("*/state.json"))
        assert json.loads(lifecycle.read_text())["state"] == "running"

        os.kill(supervisors[0].pid, signal.SIGTERM)
        exit_code = supervisors[0].wait(timeout=10)
        deadline = time.monotonic() + 10
        while child_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        receipt = json.loads(lifecycle.read_text())["state"]
        assert not child_alive(child_pid), f"engine child survived SIGTERM; receipt={receipt}"
        assert exit_code == 0
        assert receipt == "stopped"
        assert engine_status(tmp_path)[0]["state"] == "stopped"
        assert start_engine(tmp_path, 1, "main", port) == row
        stop_engine(tmp_path, "main")
    finally:
        for process in supervisors:
            if process.poll() is None:
                stop_engine(tmp_path, "main")
            process.wait(timeout=10)
        if child_pid and child_alive(child_pid):
            assert process_field(child_pid, "args") == child_command
            os.kill(child_pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while child_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not child_alive(child_pid), "owned fixture child did not exit during cleanup"


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs process lifetime contract")
def test_supervisor_exit_during_status_read_refuses_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_generation(tmp_path)
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    ) as child:
        stat_path = Path("/proc") / str(child.pid) / "stat"
        runtime = tmp_path / "runtime/1"
        runtime.mkdir(parents=True)
        (runtime / "main.json").write_text(
            json.dumps(
                {
                    "name": "main",
                    "generation": 1,
                    "port": 1,
                    "gpus": [],
                    "instance": "exiting-supervisor",
                    "shutdownToken": "private",
                    "pid": child.pid,
                    "state": "ready",
                    "url": "http://127.0.0.1:1",
                }
            )
        )
        original = Path.read_text

        def read_text(path: Path, *args, **kwargs) -> str:
            if path != stat_path:
                return original(path, *args, **kwargs)
            with path.open(encoding="utf-8") as stream:
                assert child.stdin is not None
                child.stdin.close()
                assert child.wait(timeout=10) == 0
                return stream.read()

        monkeypatch.setattr(Path, "read_text", read_text)
        assert engine_status(tmp_path)[0]["state"] == "failed"
        with pytest.raises(EnvironmentError, match="cannot verify"):
            require_idle(tmp_path)


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
