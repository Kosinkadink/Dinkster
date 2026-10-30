"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from supervisor_subprocess import run


def test_child_output_and_timeout_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "diagnostic_child.py"
    module.write_text(
        "import os, shutil, sys, threading\n"
        "from pathlib import Path\n"
        "def blocked_child():\n"
        f"    root = Path({str(tmp_path)!r})\n"
        "    (root / 'original').write_text('filesystem diagnostic')\n"
        "    os.link(root / 'original', root / 'first-link')\n"
        "    os.link(root / 'original', root / 'second-link')\n"
        "    shutil.copy2(root / 'original', root / 'copied')\n"
        "    def pending_link(event, arguments):\n"
        "        if event == 'os.link':\n"
        "            threading.Event().wait(90)\n"
        "    sys.addaudithook(pending_link)\n"
        "    os.link(root / 'original', root / 'pending-link')\n"
        "if sys.argv[1] == 'blocked':\n"
        "    blocked_child()\n"
        "print('child stdout')\n"
        "print('child stderr', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    command = [sys.executable, "-m", "diagnostic_child"]
    result = run([*command, "complete"])
    assert (result.returncode, result.stdout, result.stderr) == (
        7,
        "child stdout\n",
        "child stderr\n",
    )
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run([*command, "blocked"])
    assert raised.value.timeout == 30
    notes = "\n".join(raised.value.__notes__)
    assert "returncode before cleanup=None" in notes
    assert "Child stack before cleanup:" in notes
    assert "blocked_child" in notes
    assert str(module) in notes
    samples = [json.loads(line) for line in notes.splitlines() if line.startswith('{"elapsed":')]
    assert samples, notes
    operations = samples[-1]["operations"]
    assert operations["os.link"]["started"] == 3
    assert operations["os.link"]["source"] == str(tmp_path / "original")
    assert operations["os.link"]["destination"] == str(tmp_path / "pending-link")
    assert operations["shutil.copyfile"]["started"] == 1
    assert operations["shutil.copyfile"]["destination"] == str(tmp_path / "copied")
    assert (tmp_path / "copied").read_text() == "filesystem diagnostic"
    assert not (tmp_path / "pending-link").exists()


@pytest.mark.parametrize(
    ("module_name", "blocked"),
    [("dinkster.generation_packs", True), ("dinkster.manager", False)],
)
def test_nested_child_state_and_stack_at_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, blocked: bool
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
        "os.link(root / 'runtime-pid', root / 'linked-pid')\n"
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
        "import subprocess, threading\n"
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
            result = run([parent_python, "-m", "diagnostic_parent"])
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
            samples = [
                json.loads(line) for line in notes.splitlines() if line.startswith('{"elapsed":')
            ]
            assert any(
                sample["operations"].get("os.link", {}).get("destination")
                == str(tmp_path / "linked-pid")
                for sample in samples
            ), notes
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
