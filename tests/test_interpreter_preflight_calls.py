from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from dinkster_workers import doctor
from dinkster_workers.interpreter import InterpreterPreflightError


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _reject(_interpreter: Path | str) -> tuple[int, int]:
    raise InterpreterPreflightError("unsupported selected interpreter")


def _unexpected(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("child import, probe, or launcher was reached")


def _manifest(root: Path, name: str = "test-pack") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    package = root / name.replace("-", "_")
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("NODES = []\n")
    path = root / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "{name}"\n[pack.sandbox]\n[pack.entry]\n'
        f'nodes = "{name.replace("-", "_")}:NODES"\n'
    )
    return path


def test_doctor_reports_interpreter_timeout_before_child_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "pack")

    def timeout(command: list[str], **kwargs: object) -> None:
        assert command[:4] == ["/selected/python", "-I", "-S", "-c"]
        assert kwargs["timeout"] == 5.0
        raise subprocess.TimeoutExpired(command, 5.0)

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", timeout)
    report = doctor.diagnose(manifest, interpreter="/selected/python")
    assert not report.ok
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "doctor.interpreter"
    assert finding.severity == "error"
    assert "version check timed out after 5s" in finding.message
    assert "Python >=3.12" in finding.fix


class _UnexpectedLauncher:
    async def launch(self, _spec: object) -> None:
        _unexpected()
