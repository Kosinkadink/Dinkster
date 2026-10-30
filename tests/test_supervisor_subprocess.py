"""Supervisor CLI diagnostics preserve output and explain a real child timeout."""

from __future__ import annotations

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
