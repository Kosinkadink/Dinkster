"""Readable names never replace full content identity or legacy cache locations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dinkster_store import (
    object_entries,
    object_path,
)


def test_content_identity_survives_label_changes_collisions_and_collection(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = "sha256:" + "a" * 64
    second = "sha256:" + "b" * 64
    legacy = "sha256:" + "c" * 64
    label = "linux-cpu/2026-11-03-py3.14.10-torch2.14.1"
    allocated = object_path(store, first, label=label)
    assert allocated == store / label
    # Reservations remain distinct even before a download creates either directory.
    other = object_path(store, second, label=label)
    assert other == store / (label + "--2")
    allocated.mkdir(parents=True)
    other.mkdir(parents=True)
    old = store / ("c" * 64)
    old.mkdir()
    assert object_path(store, legacy, label=label) == old
    assert object_path(store, first, label="renamed") == allocated
    assert object_path(store, second) == other
    assert dict(object_entries(store)) == {first: allocated, second: other, legacy: old}
    allocated.rmdir()
    assert object_path(store, first, label="renamed") == allocated
    assert dict(object_entries(store)) == {second: other, legacy: old}


def test_allocation_is_serialized_across_writers(tmp_path: Path) -> None:
    def allocate(number: int) -> Path:
        return object_path(tmp_path, f"sha256:{number:064x}", label="2026-11-03-0.0.3")

    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(allocate, range(1, 13)))
    assert len(set(paths)) == 12
    assert {path.name for path in paths} == {
        "2026-11-03-0.0.3",
        *(f"2026-11-03-0.0.3--{number}" for number in range(2, 13)),
    }
    for number, path in enumerate(paths, 1):
        assert object_path(tmp_path, f"sha256:{number:064x}") == path
