from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("hosted", [True, False])
def test_only_audited_root_suites_skip_on_github_actions(tmp_path: Path, hosted: bool) -> None:
    source = Path(__file__).with_name("conftest.py")
    (tmp_path / "conftest.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    names = (
        "test_environment_history.py",
        "test_environment_generations.py",
        "test_environment_transactions.py",
        "test_release_install.py",
    )
    failing = "def test_failure():\n    assert False\n"
    for name in names:
        (tmp_path / name).write_text(failing, encoding="utf-8")
    (tmp_path / "test_environment_runtime.py").write_text(failing, encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / names[0]).write_text(failing, encoding="utf-8")
    (tmp_path / "test_control.py").write_text("def test_pass():\n    pass\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib"],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_ACTIONS": "true" if hosted else "false"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    expected = "2 failed, 1 passed, 4 skipped" if hosted else "6 failed, 1 passed"
    assert expected in result.stdout, result.stdout + result.stderr
