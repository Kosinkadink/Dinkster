from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("published", ["pass", "fail", "absent"])
def test_selected_archive_runner_preserves_execution_and_unavailable_exports(
    tmp_path: Path, published: str
) -> None:
    runner = Path(__file__).parents[1] / "scripts/_selected_pack_test_child.py"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-n 2 --max-worker-restart=0"\n', encoding="ascii"
    )
    source = tmp_path / "test_selected.py"
    code = (
        "from _selected_pack_test_child import UnavailableExport\n"
        "def test_unavailable():\n"
        "    bool(UnavailableExport('fixture:unpublished'))\n"
    )
    if published != "absent":
        code += f"def test_published():\n    assert {published == 'pass'}\n"
    if published == "fail":
        code += "def test_keeper():\n    assert 17 + 4 == 21\n"
    source.write_text(code, encoding="ascii")
    config = tmp_path / "selection.json"
    config.write_text(json.dumps({"modules": {}, "packs": [], "tests": [str(source)]}))
    result = subprocess.run(
        [sys.executable, str(runner), str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    expected = {"pass": 0, "fail": 1, "absent": 5}[published]
    assert result.returncode == expected, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    if published != "absent":
        assert ("1 passed" if published == "pass" else "1 failed") in result.stdout
