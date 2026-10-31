"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest
from supervisor_subprocess import run


@pytest.mark.parametrize("entry", ["module", "inline"])
def test_child_output_and_timeout_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
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
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([*command, "blocked", 'space "quoted" value'])
    assert raised.value.timeout == 30
    notes = "\n".join(raised.value.__notes__)
    assert "returncode before cleanup=None" in notes
    assert "Child stack before cleanup:" in notes
    assert "blocked_child" in notes
    assert "active_child" in notes
    assert (str(module) if entry == "module" else "<string>") in notes
    assert "Requested timeout snapshot" in notes


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
        assert raised.value.timeout == 30
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
