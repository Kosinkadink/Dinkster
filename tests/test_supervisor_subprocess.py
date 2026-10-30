"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from supervisor_subprocess import run


def test_child_output_and_timeout_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "diagnostic_child.py"
    module.write_text(
        "import sys, threading\n"
        "def blocked_child():\n"
        "    threading.Event().wait(90)\n"
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


@pytest.mark.parametrize(
    ("module_name", "blocked"),
    [("dinkster.generation_packs", True), ("dinkster.manager", False)],
)
def test_nested_child_state_and_stack_at_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, blocked: bool
) -> None:
    package = tmp_path / "dinkster"
    package.mkdir()
    (package / "__init__.py").write_text("")
    nested = package / (module_name.split(".")[-1] + ".py")
    nested.write_text(
        "import faulthandler, os, sys, threading, time\n"
        "from pathlib import Path\n"
        "assert faulthandler.is_enabled()\n"
        "assert os.environ['PYTHONFAULTHANDLER'] == '1'\n"
        "root = Path(sys.argv[-1])\n"
        "(root / 'runtime-pid').write_text(str(os.getpid()))\n"
        "def blocked_nested_child():\n"
        "    deadline = time.monotonic() + 90\n"
        "    while not (root / 'release').exists() and time.monotonic() < deadline:\n"
        "        threading.Event().wait(0.05)\n"
        f"if {blocked!r}:\n"
        "    blocked_nested_child()\n"
        "(root / 'completed').touch()\n"
        "print('nested stdout')\n"
        "print('nested stderr', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    arguments = ["--root", "", "prepare-catalogs"] if module_name == "dinkster.manager" else []
    (tmp_path / "diagnostic_parent.py").write_text(
        "import subprocess, sys, threading\n"
        f"result = subprocess.run([sys.executable, '-B', '-m', {module_name!r}, "
        f"*{arguments!r}, {str(tmp_path)!r}], capture_output=True, text=True)\n"
        "assert (result.returncode, result.stdout, result.stderr) == "
        "(7, 'nested stdout\\n', 'nested stderr\\n'), result\n"
        "threading.Event().wait(90)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    try:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run([sys.executable, "-m", "diagnostic_parent"])
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
        else:
            assert "Timeout (0:00:05)" not in notes
    finally:
        (tmp_path / "release").touch()
        deadline = time.monotonic() + 30
        while not (tmp_path / "completed").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "completed").exists()
