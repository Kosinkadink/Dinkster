from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.pytest_file_shard import file_shard, parse_file_shard

REPO_ROOT = Path(__file__).parents[1]


def _run_pytest(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    pythonpath = os.pathsep.join(
        path for path in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if path
    )
    return subprocess.run(
        (sys.executable, "-m", "pytest", "-vv", "-p", "tools.pytest_file_shard", *args),
        cwd=directory,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _passed_node_ids(result: subprocess.CompletedProcess[str]) -> set[str]:
    return {
        line.split()[0] for line in result.stdout.splitlines() if "::" in line and " PASSED" in line
    }


def test_file_shards_are_stable_disjoint_and_complete() -> None:
    expected = {
        "tests/test_serve.py": 2,
        "tests/test_compose.py": 1,
        "packages/dinkster-nodes-remote/tests/test_remote_nodes.py": 2,
    }
    actual = {path: file_shard(path, 2) for path in expected}
    first = {path for path, shard in actual.items() if shard == 1}
    second = {path for path, shard in actual.items() if shard == 2}

    assert actual == expected
    assert first.isdisjoint(second)
    assert first | second == set(expected)


@pytest.mark.parametrize("count", (3, 4, 8))
def test_file_shards_support_extended_counts(count: int) -> None:
    expected = {
        3: {
            "tests/test_environment_archives.py": 1,
            "tests/test_environment_cli.py": 2,
            "tests/test_environment_transactions.py": 3,
        },
        4: {
            "tests/test_serve.py": 2,
            "tests/test_compose.py": 1,
            "packages/dinkster-nodes-remote/tests/test_remote_nodes.py": 4,
        },
        8: {
            "tests/test_serve.py": 2,
            "tests/test_compose.py": 5,
            "packages/dinkster-nodes-remote/tests/test_remote_nodes.py": 8,
        },
    }

    actual = {path: file_shard(path, count) for path in expected[count]}
    shards = [
        {path for path, shard in actual.items() if shard == index} for index in range(1, count + 1)
    ]

    assert actual == expected[count]
    assert all(
        left.isdisjoint(right) for index, left in enumerate(shards) for right in shards[index + 1 :]
    )
    assert set().union(*shards) == set(expected[count])


@pytest.mark.parametrize(
    "value", ("0/2", "3/2", "0/3", "4/3", "0/4", "5/4", "0/8", "9/8", "1/5", "one/two", "1")
)
def test_parse_file_shard_refuses_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="file shard"):
        parse_file_shard(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    (("1/2", 1), ("2/2", 2)),
)
def test_parse_file_shard_accepts_two_shards(value: str, expected: int) -> None:
    assert parse_file_shard(value) == (expected, 2)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("1/3", (1, 3)),
        ("3/3", (3, 3)),
        ("1/4", (1, 4)),
        ("4/4", (4, 4)),
        ("1/8", (1, 8)),
        ("8/8", (8, 8)),
    ),
)
def test_parse_file_shard_accepts_supported_counts(value: str, expected: tuple[int, int]) -> None:
    assert parse_file_shard(value) == expected


def test_pytest_plugin_selects_whole_files_and_explicit_shared_tests(tmp_path: Path) -> None:
    (tmp_path / "test_alpha.py").write_text(
        "import pytest\n"
        "def test_alpha_one(): pass\n"
        "@pytest.mark.all_file_shards\n"
        "def test_alpha_two(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_beta.py").write_text("def test_beta(): pass\n", encoding="utf-8")
    all_nodes = {
        "test_alpha.py::test_alpha_one",
        "test_alpha.py::test_alpha_two",
        "test_beta.py::test_beta",
    }

    unsharded = _run_pytest(tmp_path)
    first = _run_pytest(tmp_path, "--file-shard", "1/2")
    second = _run_pytest(tmp_path, "--file-shard", "2/2")

    assert unsharded.returncode == 0, unsharded.stdout + unsharded.stderr
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert _passed_node_ids(unsharded) == all_nodes
    assert "deselected" not in unsharded.stdout
    assert _passed_node_ids(first) == {
        "test_alpha.py::test_alpha_one",
        "test_alpha.py::test_alpha_two",
    }
    assert _passed_node_ids(second) == {
        "test_alpha.py::test_alpha_two",
        "test_beta.py::test_beta",
    }
    assert _passed_node_ids(first) & _passed_node_ids(second) == {"test_alpha.py::test_alpha_two"}
    assert _passed_node_ids(first) | _passed_node_ids(second) == all_nodes
    assert "1 deselected" in first.stdout
    assert "1 deselected" in second.stdout

    malformed = _run_pytest(tmp_path, "--file-shard", "3/2")
    assert malformed.returncode == pytest.ExitCode.USAGE_ERROR
    assert "file shard must be INDEX/COUNT, with COUNT 2, 3, 4, or 8" in malformed.stderr


def test_case_sharding_sorts_node_ids_and_keeps_other_files_whole(tmp_path: Path) -> None:
    (tmp_path / "test_cases.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('name', "
        "['zeta', 'alpha', 'kappa', 'beta', 'eta', 'delta', 'theta'])\n"
        "def test_case(name): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_whole.py").write_text(
        "def test_first(): pass\ndef test_second(): pass\n", encoding="utf-8"
    )
    expected_cases = (("alpha", "eta", "zeta"), ("beta", "kappa"), ("delta", "theta"))
    whole = {"test_whole.py::test_first", "test_whole.py::test_second"}
    shards = []
    for index, names in enumerate(expected_cases, start=1):
        result = _run_pytest(
            tmp_path, "--file-shard", f"{index}/3", "--case-shard-file", "test_cases.py"
        )
        assert result.returncode == 0, result.stdout + result.stderr
        selected = _passed_node_ids(result)
        assert selected - whole == {f"test_cases.py::test_case[{name}]" for name in names}
        assert selected & whole in (set(), whole)
        shards.append(selected)
    assert all(left.isdisjoint(right) for i, left in enumerate(shards) for right in shards[i + 1 :])
    assert set().union(*shards) == whole | {
        f"test_cases.py::test_case[{name}]" for names in expected_cases for name in names
    }
    reversed_files = _run_pytest(
        tmp_path,
        "test_whole.py",
        "test_cases.py",
        "--file-shard",
        "2/3",
        "--case-shard-file",
        "test_cases.py",
    )
    assert reversed_files.returncode == 0, reversed_files.stdout + reversed_files.stderr
    assert _passed_node_ids(reversed_files) == shards[1]
    missing_shard = _run_pytest(tmp_path, "--case-shard-file", "test_cases.py")
    assert missing_shard.returncode == pytest.ExitCode.USAGE_ERROR
    missing_file = _run_pytest(
        tmp_path, "--file-shard", "1/3", "--case-shard-file", "test_missing.py"
    )
    assert missing_file.returncode == pytest.ExitCode.USAGE_ERROR
