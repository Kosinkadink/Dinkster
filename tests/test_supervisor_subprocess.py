"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import venv
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from supervisor_subprocess import _request_snapshot, operation_reports, run


def wait_until(ready: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 30
    while not ready():
        assert time.monotonic() < deadline, "diagnostic child did not reach the required state"
        threading.Event().wait(0.01)


def test_completed_and_refused_operations_retain_separate_timings(tmp_path: Path) -> None:
    (tmp_path / "source").write_bytes(b"payload")
    result = run(
        [
            sys.executable,
            "-c",
            "import hashlib, shutil\n"
            "from pathlib import Path\n"
            "from dinkster_supervisor.environment_downloads import fetch_artifact\n"
            "from dinkster_supervisor.environment_state import atomic_write\n"
            "from dinkster_supervisor.environment_runtime import _request\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "from threading import Thread\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.end_headers()\n"
            '        self.wfile.write(b\'{"state": "ready"}\')\n'
            "    def log_message(self, *args):\n"
            "        pass\n"
            "server = HTTPServer(('127.0.0.1', 0), Handler)\n"
            "Thread(target=server.handle_request, daemon=True).start()\n"
            "assert _request(f'http://127.0.0.1:{server.server_port}/supervisor/status') "
            "== {'state': 'ready'}\n"
            "server.server_close()\n"
            f"root = Path({str(tmp_path)!r})\n"
            "atomic_write(root / 'scratch' / 'state', b'journal bytes')\n"
            "shutil.rmtree(root / 'scratch')\n"
            "fetch_artifact((str(root),), 'source', root / 'copied', "
            "hashlib.sha256(b'payload').hexdigest(), 7)\n"
            "try:\n"
            "    fetch_artifact((str(root),), 'source', root / 'refused', '0' * 64, 7)\n"
            "except ValueError:\n"
            "    print('refused')\n",
        ]
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "refused\n", "")
    assert (tmp_path / "copied").read_bytes() == b"payload"
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "refused").exists()
    report = operation_reports[-1]
    for operation in (
        "environment_state.atomic_write",
        "shutil.rmtree",
        "environment_runtime._request",
    ):
        events = [line.split() for line in report.splitlines() if operation in line]
        assert [row[1] for row in events] == ["start", "return"]
    events = [line.split() for line in report.splitlines() if "fetch_artifact" in line]
    assert [row[1] for row in events] == ["start", "return", "start", "unwind"]
    stamps = [float(row[0].removesuffix("s")) for row in events]
    assert stamps[0] < stamps[1] <= stamps[2] < stamps[3]
    assert "supervisor child wait" in report


def test_timeout_retains_unfinished_operation(tmp_path: Path, expire_after_ready) -> None:
    (tmp_path / "source").write_bytes(b"payload")
    expire_after_ready(lambda process: wait_until((tmp_path / "ready").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run(
            [
                sys.executable,
                "-c",
                "import hashlib, threading\n"
                "from pathlib import Path\n"
                "from dinkster_supervisor import environment_downloads as downloads\n"
                f"root = Path({str(tmp_path)!r})\n"
                "def blocked(*args, **kwargs):\n"
                "    (root / 'ready').touch()\n"
                "    threading.Event().wait(90)\n"
                "downloads._download_local = blocked\n"
                "downloads.fetch_artifact((str(root),), 'source', root / 'copied', "
                "hashlib.sha256(b'payload').hexdigest(), 7)\n",
            ]
        )
    assert raised.value.timeout == 0
    notes = "\n".join(raised.value.__notes__)
    assert "start dinkster_supervisor.environment_downloads.fetch_artifact" in notes
    assert "return dinkster_supervisor.environment_downloads.fetch_artifact" not in notes
    assert "unwind dinkster_supervisor.environment_downloads.fetch_artifact" not in notes
    assert "Requested timeout snapshot" in notes


@pytest.fixture
def expire_after_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[subprocess.Popen[str]], None]], None]:
    communicate = subprocess.Popen.communicate

    def install(ready: Callable[[subprocess.Popen[str]], None]) -> None:
        def expire(process, input=None, timeout=None):
            if timeout is None:
                return communicate(process, input=input)
            assert timeout == 120
            ready(process)
            # Exercise real communicate expiration, but only once the child is ready.
            return communicate(process, input=input, timeout=0)

        monkeypatch.setattr(subprocess.Popen, "communicate", expire)

    return install


@pytest.mark.parametrize("entry", ["module", "inline"])
def test_child_output_and_timeout_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, expire_after_ready
) -> None:
    module = tmp_path / "diagnostic_child.py"
    module.write_text(
        "import sys, threading\n"
        "assert sys.argv[2] == 'space \\\"quoted\\\" value'\n"
        "ready = threading.Event()\n"
        "def changing_frames(depth):\n"
        "    values = [str(depth), {'depth': depth}]\n"
        "    return changing_frames(depth - 1) if depth else values\n"
        "def active_child():\n"
        "    ready.set()\n"
        "    while True:\n"
        "        changing_frames(24)\n"
        "def blocked_child():\n"
        "    threading.Thread(target=active_child, daemon=True).start()\n"
        "    ready.wait()\n"
        f"    __import__('pathlib').Path({str(tmp_path / 'ready')!r}).touch()\n"
        "    threading.Event().wait(90)\n"
        "if sys.argv[1] == 'blocked':\n"
        "    blocked_child()\n"
        "print('child stdout')\n"
        "print('child stderr', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    command = (
        [sys.executable, "-m", "diagnostic_child"]
        if entry == "module"
        else [sys.executable, "-c", module.read_text()]
    )
    result = run([*command, "complete", 'space "quoted" value'])
    assert (result.returncode, result.stdout, result.stderr) == (
        7,
        "child stdout\n",
        "child stderr\n",
    )
    expire_after_ready(lambda process: wait_until((tmp_path / "ready").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([*command, "blocked", 'space "quoted" value'])
    assert raised.value.timeout == 0
    notes = raised.value.__notes__[0]
    assert "returncode before cleanup=None" in notes
    assert "Child stack before cleanup:" in notes
    assert "blocked_child" in notes
    assert "active_child" in notes
    assert (str(module) if entry == "module" else "<string>") in notes
    assert "Requested timeout snapshot" in notes
    assert "Snapshot dump completed" in notes


@pytest.mark.parametrize("failure", ["dump", "acknowledgment"])
def test_child_snapshot_failure_is_reported(
    tmp_path: Path, failure: str, expire_after_ready
) -> None:
    script = (
        "import faulthandler, threading\n"
        "from pathlib import Path\n"
        "touch = Path.touch\n"
        "def failing_dump(*args, **kwargs):\n"
        "    raise ValueError('controlled dump failure')\n"
        "def failing_touch(path, *args, **kwargs):\n"
        "    if path.name == 'child.snapshot':\n"
        "        raise OSError('controlled acknowledgment failure')\n"
        "    return touch(path, *args, **kwargs)\n"
        f"if {failure!r} == 'dump':\n"
        "    faulthandler.dump_traceback = failing_dump\n"
        "else:\n"
        "    Path.touch = failing_touch\n"
        f"Path({str(tmp_path / 'ready')!r}).touch()\n"
        "threading.Event().wait(90)\n"
    )
    expire_after_ready(lambda process: wait_until((tmp_path / "ready").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([sys.executable, "-c", script])
    assert raised.value.timeout == 0
    notes = "\n".join(raised.value.__notes__)
    assert "Timeout snapshot not acknowledged" in notes
    assert "Snapshot dump call started" in notes
    assert f"controlled {failure} failure" in notes
    if failure == "dump":
        assert "Snapshot dump failed" in notes
        assert "Snapshot dump completed" not in notes
    else:
        assert "Snapshot dump completed" in notes
        assert "Snapshot acknowledgment failed" in notes


def test_late_child_snapshot_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire_after_ready
) -> None:
    script = (
        "import faulthandler, threading, time\n"
        "from pathlib import Path\n"
        "dump = faulthandler.dump_traceback\n"
        "def held_dump(*args, **kwargs):\n"
        "    deadline = time.monotonic() + 30\n"
        f"    while not Path({str(tmp_path / 'release')!r}).exists():\n"
        "        assert time.monotonic() < deadline, 'dump release absent'\n"
        "        time.sleep(0.01)\n"
        "    dump(*args, **kwargs)\n"
        f"    Path({str(tmp_path / 'dumped')!r}).touch()\n"
        "faulthandler.dump_traceback = held_dump\n"
        f"Path({str(tmp_path / 'ready')!r}).touch()\n"
        "def late_snapshot_child():\n"
        "    threading.Event().wait(90)\n"
        "late_snapshot_child()\n"
    )
    kill = subprocess.Popen.kill

    def release_before_kill(process):
        (tmp_path / "release").touch()
        wait_until((tmp_path / "dumped").exists)
        kill(process)

    monkeypatch.setattr(subprocess.Popen, "kill", release_before_kill)
    expire_after_ready(lambda process: wait_until((tmp_path / "ready").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([sys.executable, "-c", script])
    assert raised.value.timeout == 0
    notes = "\n".join(raised.value.__notes__)
    assert "Timeout snapshot not acknowledged" in notes
    assert "late_snapshot_child" not in raised.value.__notes__[0]
    assert "Child diagnostics collected after cleanup:" in raised.value.__notes__[-1]
    assert "late_snapshot_child" in raised.value.__notes__[-1]


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("kind", ["probe", "manager", "manager-help"])
def test_unwrapped_child_lifecycle_preserves_command_and_output(
    tmp_path: Path, blocked: bool, kind: str, expire_after_ready
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        arguments = (
            ["unchanged argument"]
            if kind == "probe"
            else [
                "--root",
                str(tmp_path),
                "prepare-registry-sources",
                *(["--help"] if kind == "manager-help" else []),
                "unchanged argument",
            ]
        )
        probe = (
            "import os, socket, sys\n"
            "from pathlib import Path\n"
            "assert sys.flags.isolated\n"
            "assert sys.dont_write_bytecode\n"
            f"assert sys.argv[1:] == {arguments!r}\n"
            f"root = Path({str(tmp_path)!r})\n"
            "(root / 'pid').write_text(str(os.getpid()))\n"
            f"connection = socket.create_connection({listener.getsockname()!r}, timeout=30)\n"
            "connection.settimeout(None)\n"
            "connection.sendall(b'ready')\n"
            f"if {blocked!r}:\n"
            "    (root / 'ready').touch()\n"
            "    assert connection.recv(1) == b'x'\n"
            "connection.sendall(b'completed')\n"
            "connection.close()\n"
            "print('probe stdout')\n"
            "print('probe stderr', file=sys.stderr)\n"
            "sys.exit(7)\n"
        )
        python = str(vars(sys)["_base_executable"])
        if kind == "probe":
            command = [python, "-I", "-B", "-c", probe, *arguments]
            category = "interpreter probe"
        else:
            environment = tmp_path / "interpreter"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = str(
                environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            )
            site = subprocess.check_output(
                [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
            package = Path(site) / "dinkster"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "manager.py").write_text(probe)
            command = [python, "-I", "-B", "-m", "dinkster.manager", *arguments]
            category = (
                "manager source help" if kind == "manager-help" else "manager source preparation"
            )
        parent = (
            "import subprocess, threading\n"
            "from pathlib import Path\n"
            f"child = subprocess.Popen({command!r}, stdout=subprocess.PIPE, "
            "stderr=subprocess.PIPE, text=True)\n"
            f"Path({str(tmp_path / 'launched-pid')!r}).write_text(str(child.pid))\n"
            "stdout, stderr = child.communicate(timeout=60)\n"
            "assert (child.returncode, stdout, stderr) == "
            "(7, 'probe stdout\\n', 'probe stderr\\n'), (child.returncode, stdout, stderr)\n"
            f"__import__('pathlib').Path({str(tmp_path / 'checked')!r}).touch()\n"
            "threading.Event().wait(90)\n"
        )

        def ready(process: subprocess.Popen[str]) -> None:
            wait_until((tmp_path / "launched-pid").exists)
            wait_until((tmp_path / ("ready" if blocked else "checked")).exists)

        expire_after_ready(ready)
        try:
            with pytest.raises(subprocess.TimeoutExpired) as raised:
                run([str(vars(sys)["_base_executable"]), "-c", parent])
            notes = "\n".join(raised.value.__notes__)
            assert "unchanged argument" not in notes
            # A Windows venv launcher can differ from the executing interpreter.
            assert f"Unwrapped {category} pid={(tmp_path / 'launched-pid').read_text()}," in notes
            assert int((tmp_path / "pid").read_text()) > 0
            assert (
                "state before cleanup=alive, returncode before cleanup=None"
                if blocked
                else "state before cleanup=exited, returncode before cleanup=7"
            ) in notes.split(f"Unwrapped {category}", 1)[1]
            lifecycle = json.loads(
                notes.split("Child lifecycle (seconds since launch started): ", 1)[1].splitlines()[
                    0
                ]
            )
            assert lifecycle["spawned"] >= 0
            if blocked:
                assert "observed_exit" not in lifecycle
            else:
                assert lifecycle["observed_exit"] >= lifecycle["spawned"]
                assert (tmp_path / "checked").exists()
        finally:
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as messages:
                connection.settimeout(30)
                assert messages.read(5) == b"ready"
                if blocked:
                    connection.sendall(b"x")
                assert messages.read(9) == b"completed"


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows process handle state")
def test_exited_probe_retains_windows_exit_code_259(tmp_path: Path, expire_after_ready) -> None:
    parent = (
        "import ctypes, subprocess, threading\n"
        "from ctypes import wintypes\n"
        "from pathlib import Path\n"
        f"probe = subprocess.Popen([{str(vars(sys)['_base_executable'])!r}, "
        "'-I', '-B', '-c', 'import sys; sys.exit(259)'], "
        "stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
        "kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]\n"
        "kernel32.WaitForSingleObject.restype = wintypes.DWORD\n"
        "assert kernel32.WaitForSingleObject(int(probe._handle), 30000) == 0\n"
        f"root = Path({str(tmp_path)!r})\n"
        "(root / 'pid').write_text(str(probe.pid))\n"
        "(root / 'ready').touch()\n"
        "threading.Event().wait(90)\n"
    )
    # Keep the probe's handle open without populating the Popen observer's exit record.
    expire_after_ready(lambda process: wait_until((tmp_path / "ready").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([str(vars(sys)["_base_executable"]), "-c", parent])
    notes = "\n".join(raised.value.__notes__)
    assert (
        f"Unwrapped interpreter probe pid={(tmp_path / 'pid').read_text()}, "
        "state before cleanup=exited, returncode before cleanup=259"
    ) in notes
    lifecycle = json.loads(
        notes.split("Child lifecycle (seconds since launch started): ", 1)[1].splitlines()[0]
    )
    assert "observed_exit" not in lifecycle


def test_offline_uv_lifecycle_preserves_sync_result(tmp_path: Path, expire_after_ready) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    environment = tmp_path / "empty-environment"
    subprocess.run(
        [uv, "venv", "--offline", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    command = [uv, "pip", "sync", "--offline", "--python", str(python), "-"]
    parent = (
        "import subprocess, threading\n"
        f"result = subprocess.run({command!r}, input='', capture_output=True, text=True)\n"
        "assert result.returncode == 0 and result.stdout == '', result\n"
        f"__import__('pathlib').Path({str(tmp_path / 'checked')!r}).touch()\n"
        "threading.Event().wait(90)\n"
    )
    expire_after_ready(lambda process: wait_until((tmp_path / "checked").exists))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([str(vars(sys)["_base_executable"]), "-c", parent])
    notes = "\n".join(raised.value.__notes__)
    assert "Unwrapped offline uv sync pid=" in notes
    assert "state before cleanup=exited, returncode before cleanup=0" in notes


@pytest.mark.parametrize(
    ("module_name", "blocked", "late", "entry"),
    [
        ("dinkster.generation_packs", True, False, "module"),
        ("dinkster.generation_packs", True, False, "inline"),
        ("dinkster.manager", False, False, "module"),
        ("dinkster.manager", True, True, "module"),
    ],
)
def test_nested_child_state_and_stack_at_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    blocked: bool,
    late: bool,
    entry: str,
    expire_after_ready,
) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(30)
    port = listener.getsockname()[1]
    package = tmp_path / "dinkster"
    package.mkdir()
    (package / "__init__.py").write_text("")
    nested = package / (module_name.split(".")[-1] + ".py")
    nested.write_text(
        "import faulthandler, os, socket, sys\n"
        "from pathlib import Path\n"
        "assert faulthandler.is_enabled()\n"
        "assert os.environ['PYTHONFAULTHANDLER'] == '1'\n"
        "root = Path(sys.argv[-1])\n"
        "(root / 'runtime-pid').write_text(str(os.getpid()))\n"
        f"release = socket.create_connection(('127.0.0.1', {port}), timeout=30)\n"
        "release.settimeout(None)\n"
        "release.sendall(b'ready')\n"
        "def blocked_nested_child():\n"
        "    (root / 'blocked-ready').touch()\n"
        "    assert release.recv(1) == b'x'\n"
        f"if {blocked!r}:\n"
        "    blocked_nested_child()\n"
        "(root / 'completed').touch()\n"
        "release.sendall(b'completed')\n"
        "release.close()\n"
        "print('nested stdout')\n"
        "print('nested stderr', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    arguments = ["--root", "", "prepare-catalogs"] if module_name == "dinkster.manager" else []
    (tmp_path / "diagnostic_parent.py").write_text(
        "import faulthandler, subprocess, threading\n"
        f"if {late!r}:\n"
        "    sampled = threading.Event()\n"
        "    dump = faulthandler.dump_traceback\n"
        "    def record_sample(*args, **kwargs):\n"
        "        dump(*args, **kwargs)\n"
        "        sampled.set()\n"
        "    faulthandler.dump_traceback = record_sample\n"
        f"    __import__('pathlib').Path({str(tmp_path / 'awaiting-snapshot')!r}).touch()\n"
        "    assert sampled.wait(30), 'supervisor sample absent'\n"
        f"result = subprocess.run([{sys.executable!r}, '-B', '-m', {module_name!r}, "
        f"*{arguments!r}, {str(tmp_path)!r}], capture_output=True, text=True)\n"
        "assert (result.returncode, result.stdout, result.stderr) == "
        "(7, 'nested stdout\\n', 'nested stderr\\n'), result\n"
        "threading.Event().wait(90)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # Killing a Windows venv launcher can leave its interpreter holding the output pipes.
    parent_python = str(vars(sys)["_base_executable"])

    def ready(process: subprocess.Popen[str]) -> None:
        directory = Path(cast(list[str], process.args)[4])
        if late:
            wait_until((tmp_path / "awaiting-snapshot").exists)
            assert _request_snapshot(directory) == "Timeout snapshot acknowledged"
            (directory / "snapshot.request").unlink()
        if blocked:
            wait_until((tmp_path / "blocked-ready").exists)
        else:
            wait_until(lambda: bool(list(directory.glob("nested-*.exit"))))
        if late:
            for acknowledgment in directory.glob("*.snapshot"):
                acknowledgment.unlink()

    expire_after_ready(ready)
    result = None
    try:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            result = run(
                [parent_python, "-m", "diagnostic_parent"]
                if entry == "module"
                else [
                    parent_python,
                    "-c",
                    "import runpy; runpy.run_module('diagnostic_parent', run_name='__main__')",
                ]
            )
            pytest.fail(f"Diagnostic parent exited before timeout: {result!r}")
        assert raised.value.timeout == 0
        notes = "\n".join(raised.value.__notes__)
        assert f"module={module_name}" in notes
        assert f"Nested CLI child pid={(tmp_path / 'runtime-pid').read_text()}," in notes
        assert (
            "state before cleanup=alive" in notes
            if blocked
            else ("state before cleanup=exited, returncode before cleanup=7" in notes)
        )
        if blocked:
            assert "blocked_nested_child" in notes
            assert str(nested) in notes
            assert (
                "Requested timeout snapshot"
                in notes.split("Nested child stack before cleanup:\n", 1)[1]
            )
        else:
            assert "Timeout (0:00:05)" not in notes
    finally:
        with listener:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(30)
                with connection.makefile("rb") as messages:
                    assert messages.read(5) == b"ready", result
                    if blocked:
                        connection.sendall(b"x")
                    assert messages.read(9) == b"completed", result
        assert (tmp_path / "completed").exists(), result
