"""Engine pack contracts over a test-only specimen, not an author template."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dinkster_workers.doctor import diagnose, render_text

PACK = Path(__file__).resolve().parent / "fixtures" / "pack-contract"


def test_pack_is_doctor_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def current_interpreter_version(selected: Path | str) -> tuple[int, int]:
        assert str(selected) == sys.executable
        assert sys.version_info[:2] >= (3, 12)
        return sys.version_info[:2]

    # Pack health is independent of version-probe startup latency. Interpreter
    # admission has its own real-process and timeout tests; the pack probe stays real.
    monkeypatch.setattr(
        "dinkster_workers.doctor.preflight_interpreter", current_interpreter_version
    )
    report = diagnose(PACK / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    assert not report.findings, render_text(report)
