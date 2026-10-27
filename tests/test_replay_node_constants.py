from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.replay_node_constants import parse_cases

SCRIPT = Path(__file__).parents[1] / "tools" / "replay_node_constants.py"


def _write_fake_comfyui(root: Path) -> None:
    (root / "nodes.py").write_text(
        "class NodeOutput:\n"
        "    def __init__(self, *values):\n"
        "        self.result = values\n\n"
        "class AddNode:\n"
        "    def execute(self, left, right):\n"
        "        return {'total': left + right}\n\n"
        "class SeedNode:\n"
        "    def execute(self, seed):\n"
        "        return NodeOutput(seed)\n",
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


@pytest.mark.parametrize(
    ("expected", "returncode", "status"),
    [(9, 0, "MATCH"), (10, 1, "MISMATCH")],
)
def test_replay_projects_explicit_source_result_slot(
    tmp_path: Path, expected: int, returncode: int, status: str
) -> None:
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    _write_fake_comfyui(comfyui)
    test = tmp_path / "test_seed.py"
    metadata = (
        '{"class": "nodes.SeedNode", "method": "execute", "inputs": {"seed": 9}, '
        '"result": {"0": "seed"}}'
    )
    test.write_text(
        "def test_seed():\n"
        "    # Hand-derived synthetic fixture constant; this is not ComfyUI provenance.\n"
        f"    # comfy-replay: {metadata}\n"
        f"    assert result == {{'seed': {expected}}}\n",
        encoding="utf-8",
    )

    result = _run(comfyui, test)

    assert result.returncode == returncode
    assert f"{status} line 4: nodes.SeedNode.execute" in result.stdout
    if returncode == 0:
        assert "MATCH line 4: nodes.SeedNode.execute -> {'seed': 9}" in result.stdout
    else:
        assert "expected: {'seed': 10}" in result.stdout
        assert "actual:   {'seed': 9}" in result.stdout


def test_seed_annotation_declares_the_explicit_native_output_projection() -> None:
    utility_tests = Path(__file__).parent / "test_comfy_utility_nodes.py"

    cases = parse_cases(utility_tests)

    assert len(cases) == 1
    assert cases[0].class_path == "comfy_extras.nodes_seed.SeedNode"
    assert cases[0].method == "execute"
    assert cases[0].inputs == {"seed": 9223372036854775807}
    assert cases[0].result_projection == {0: "seed"}
    assert cases[0].expected == {"seed": 9223372036854775807}


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
        (
            (
                '{"class": "nodes.AddNode", "method": "execute", "inputs": {}, '
                '"result": {"zero": "total"}}'
            ),
            "result must use non-negative slots and names",
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
