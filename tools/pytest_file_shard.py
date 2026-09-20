from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_SHARD_COUNT = 2
# The namespace balances whole files by the root suite's measured phase times.
_HASH_PREFIX = b"dinkster-windows-pytest-v1:22340:"
_ALL_FILE_SHARDS_MARKER = "all_file_shards"


def file_shard(path: str) -> int:
    digest = hashlib.sha256(_HASH_PREFIX + path.encode("utf-8")).digest()
    return digest[0] % _SHARD_COUNT + 1


def parse_file_shard(value: str) -> int:
    try:
        index, count = (int(part) for part in value.split("/", 1))
    except ValueError as exc:
        raise ValueError("file shard must be INDEX/2") from exc
    if count != _SHARD_COUNT or not 1 <= index <= count:
        raise ValueError("file shard must be 1/2 or 2/2")
    return index


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--file-shard", metavar="INDEX/2", help="run one whole-file test shard")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_ALL_FILE_SHARDS_MARKER}: run this test on every file shard",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    value = config.getoption("file_shard")
    if value is None:
        return

    try:
        index = parse_file_shard(value)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc

    root = Path(config.rootpath)
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        path = Path(item.path).relative_to(root).as_posix()
        runs_on_all_shards = item.get_closest_marker(_ALL_FILE_SHARDS_MARKER) is not None
        (selected if runs_on_all_shards or file_shard(path) == index else deselected).append(item)

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
