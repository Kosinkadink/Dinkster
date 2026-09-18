from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "video_conformance.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("video_conformance", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_rss_generates_fixture_in_separate_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "--generate-source" in command:
            Path(command[-1]).write_bytes(b"fixture")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"baseline_rss": 10, "peak_rss": 20, "rss_growth": 10}), ""
        )

    monkeypatch.setattr(tool.subprocess, "run", run)
    tool.main(["--rss"])
    assert "--generate-source" in commands[0]
    assert "--source" in commands[1]
    assert json.loads(capsys.readouterr().out)["corpus"] == {
        "generator": "PyAV 16.0.1 libx264 1080p 30fps 10s RGB ramp",
        "bytes": 7,
        "sha256": "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d",
    }


def test_rss_gate_detects_real_allocation_after_fixture_generation() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--rss", "--self-test-allocation-mib", "256"],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode != 0, result.stdout
    measured = json.loads(result.stderr.rsplit("RuntimeError: ", 1)[-1].strip())
    assert measured["rss_growth"] >= 200 * 1024 * 1024
    assert measured["limit"] == 200 * 1024 * 1024
