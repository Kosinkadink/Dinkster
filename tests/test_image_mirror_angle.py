from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_image_mirror_angle_parity.py"
UNAVAILABLE_EXIT = 77


def test_declared_glsl_mirrors_match_cpu_corpora_through_angle() -> None:
    missing = [name for name in ("comfy_angle", "OpenGL") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(
            f"install the angle extra to run headless mirror parity (missing {', '.join(missing)})"
        )

    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode == UNAVAILABLE_EXIT:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ANGLE mirror parity passed: 27 cases" in result.stdout
