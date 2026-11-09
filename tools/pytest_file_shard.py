from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_SHARD_COUNTS = frozenset({2, 3, 4, 6, 8})
# The namespace balances whole files by the root suite's measured phase times.
_HASH_PREFIX = b"dinkster-windows-pytest-v1:22340:"
ALL_FILE_SHARDS_MARKER = "all_file_shards"


def file_shard(path: str, count: int = 2) -> int:
    if count not in _SHARD_COUNTS:
        raise ValueError("file shard count must be 2, 3, 4, 6, or 8")
    # Three-way runtime placement balances native Windows durations independently of root shards.
    prefix = b"dinkster-windows-runtime-v1:4003:" if count == 3 else _HASH_PREFIX
    digest = hashlib.sha256(prefix + path.encode("utf-8")).digest()
    return digest[0] % count + 1


def parse_file_shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/", 1))
    except ValueError as exc:
        raise ValueError("file shard must be INDEX/COUNT, with COUNT 2, 3, 4, 6, or 8") from exc
    if count not in _SHARD_COUNTS or not 1 <= index <= count:
        raise ValueError("file shard must be INDEX/COUNT, with COUNT 2, 3, 4, 6, or 8")
    return index, count


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--file-shard",
        metavar="INDEX/COUNT",
        help="run one of 2, 3, 4, 6, or 8 shards (whole-file unless opted into case splitting)",
    )
    parser.addoption(
        "--case-shard-file",
        action="append",
        default=[],
        metavar="PATH",
        help="split a root-relative test file's sorted node IDs across --file-shard shards",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{ALL_FILE_SHARDS_MARKER}: run this test on every file shard",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    value = config.getoption("file_shard")
    if value is None:
        if config.getoption("case_shard_file"):
            raise pytest.UsageError("--case-shard-file requires --file-shard")
        return

    try:
        index, count = parse_file_shard(value)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc

    root = Path(config.rootpath)
    paths = {item: Path(item.path).relative_to(root).as_posix() for item in items}
    case_files = set(config.getoption("case_shard_file"))
    missing = case_files - set(paths.values())
    if missing:
        raise pytest.UsageError(
            f"case-shard files were not collected: {', '.join(sorted(missing))}"
        )
    case_shards = {
        item.nodeid: position % count + 1
        for path in sorted(case_files)
        for position, item in enumerate(
            sorted((item for item in items if paths[item] == path), key=lambda item: item.nodeid)
        )
    }
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        path = paths[item]
        runs_on_all_shards = item.get_closest_marker(ALL_FILE_SHARDS_MARKER) is not None
        shard = case_shards[item.nodeid] if path in case_files else file_shard(path, count)
        (selected if runs_on_all_shards or shard == index else deselected).append(item)

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
