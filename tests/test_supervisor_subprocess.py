"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from supervisor_subprocess import _request_snapshot, run


def wait_until(ready: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 30
    while not ready():
        assert time.monotonic() < deadline, "diagnostic child did not reach the required state"
        threading.Event().wait(0.01)


@pytest.fixture
def expire_after_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[subprocess.Popen[str]], None]], None]:
    communicate = subprocess.Popen.communicate

    def install(ready: Callable[[subprocess.Popen[str]], None]) -> None:
        def expire(process, input=None, timeout=None):
            if timeout is None:
                return communicate(process, input=input)
            assert timeout == 30
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
    parent_python = (
        str(vars(sys)["_base_executable"]) if sys.platform == "win32" else sys.executable
    )

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
