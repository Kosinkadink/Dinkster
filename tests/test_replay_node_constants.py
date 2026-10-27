from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "replay_node_constants.py"


def _write_fake_comfyui(root: Path) -> None:
    (root / "nodes.py").write_text(
        "class AddNode:\n"
        "    def execute(self, left, right):\n"
        "        return {'total': left + right}\n",
        encoding="utf-8",
    )


def _write_node_test(path: Path, expected: int) -> None:
    metadata = '{"class": "nodes.AddNode", "method": "execute", "inputs": {"left": 1, "right": 3}}'
    path.write_text(
        "def test_add_node():\n"
        "    actual = {'total': 4}\n"
        "    # Hand-derived synthetic fixture constant; this is not ComfyUI provenance.\n"
        f"    # comfy-replay: {metadata}\n"
        f"    assert actual == {{'total': {expected}}}\n",
        encoding="utf-8",
    )


def _run(comfyui: Path, test: Path) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), str(comfyui), str(test)],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )


def test_replay_reports_matching_and_mismatching_constants_without_writes(tmp_path: Path) -> None:
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    _write_fake_comfyui(comfyui)
    matching = tmp_path / "test_matching.py"
    mismatching = tmp_path / "test_mismatching.py"
    _write_node_test(matching, expected=4)
    _write_node_test(mismatching, expected=5)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    matched = _run(comfyui, matching)
    mismatched = _run(comfyui, mismatching)

    assert matched.returncode == 0
    assert "MATCH line 5: nodes.AddNode.execute -> {'total': 4}" in matched.stdout
    assert mismatched.returncode == 1
    assert "MISMATCH line 5: nodes.AddNode.execute" in mismatched.stdout
    assert "expected: {'total': 5}" in mismatched.stdout
    assert "actual:   {'total': 4}" in mismatched.stdout
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_replay_rejects_unannotated_tests(tmp_path: Path) -> None:
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    test = tmp_path / "test_unannotated.py"
    test.write_text("def test_value():\n    assert 2 + 2 == 4\n", encoding="utf-8")

    result = _run(comfyui, test)

    assert result.returncode == 2
    assert "no # comfy-replay: annotation" in result.stderr


def test_replay_rejects_nonliteral_expected_value(tmp_path: Path) -> None:
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    test = tmp_path / "test_nonliteral.py"
    metadata = '{"class": "nodes.AddNode", "method": "execute", "inputs": {}}'
    test.write_text(
        "def test_value():\n"
        "    expected = 4\n"
        f"    # comfy-replay: {metadata}\n"
        "    assert value == expected\n",
        encoding="utf-8",
    )

    result = _run(comfyui, test)

    assert result.returncode == 2
    assert "expected value must be a Python literal" in result.stderr


@pytest.mark.parametrize(
    "metadata, message",
    [
        ("{}", "class must be a dotted class path"),
        (
            '{"class": "nodes.AddNode", "method": "", "inputs": {}}',
            "method must be a non-empty string",
        ),
        (
            '{"class": "nodes.AddNode", "method": "execute", "inputs": []}',
            "inputs must be a JSON object",
        ),
    ],
)
def test_replay_rejects_invalid_metadata(tmp_path: Path, metadata: str, message: str) -> None:
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    test = tmp_path / "test_invalid_metadata.py"
    test.write_text(
        f"def test_value():\n    # comfy-replay: {metadata}\n    assert value == 4\n",
        encoding="utf-8",
    )

    result = _run(comfyui, test)

    assert result.returncode == 2
    assert message in result.stderr
