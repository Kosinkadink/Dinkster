"""Readable names never replace full content identity or legacy cache locations."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dinkster_store import (
    StorePathError,
    object_entries,
    object_path,
    readable_label,
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


def test_pending_allocation_is_recovered_without_reusing_its_name(tmp_path: Path) -> None:
    first = "sha256:" + "d" * 64
    pending = tmp_path / ".objects.pending.json"
    pending.write_text(
        json.dumps({"format": "dinkster.store-paths/1", "objects": {first: "pack/2026-11-03-1.0"}})
    )
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert (
        object_path(tmp_path, first, label="renamed", reserve=False)
        == tmp_path / "pack/2026-11-03-1.0"
    )
    assert (
        object_path(tmp_path, "sha256:" + "e" * 64, label="pack/2026-11-03-1.0", reserve=False)
        == tmp_path / "pack/2026-11-03-1.0--2"
    )
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert object_path(tmp_path, first) == tmp_path / "pack/2026-11-03-1.0"
    second = object_path(tmp_path, "sha256:" + "e" * 64, label="pack/2026-11-03-1.0")
    assert second == tmp_path / "pack/2026-11-03-1.0--2"
    assert not pending.exists()
    assert object_path(tmp_path, first, label="different-label") == tmp_path / "pack/2026-11-03-1.0"


@pytest.mark.parametrize("label", ["../outside", "/absolute", "a/../b", "a\\b", "CON", "a."])
def test_unsafe_allocations_are_refused(tmp_path: Path, label: str) -> None:
    with pytest.raises(StorePathError):
        object_path(tmp_path, "blake3:" + "f" * 64, label=label)
    assert list(tmp_path.iterdir()) == []


def test_portable_labels_and_case_insensitive_collisions(tmp_path: Path) -> None:
    assert readable_label("../CON", "2.1+cpu / invalid") == "object-CON/2.1+cpu-invalid"
    (tmp_path / "Pack").mkdir()
    path = object_path(tmp_path, "sha256:" + "1" * 64, label="pack")
    assert path.name == "pack--2"
    assert object_path(tmp_path, "sha256:" + "2" * 64, label="PACK").name == "PACK--3"


@pytest.mark.parametrize("paths", [["pack", "pack/1.0"], ["pack/1.0", "PACK/1.0"], ["../outside"]])
def test_corrupt_identity_mapping_never_resolves_outside_or_overlapping_objects(
    tmp_path: Path, paths: list[str]
) -> None:
    (tmp_path / ".objects.json").write_text(
        json.dumps(
            {
                "format": "dinkster.store-paths/1",
                "objects": {f"sha256:{number:064x}": path for number, path in enumerate(paths)},
            }
        )
    )
    with pytest.raises(StorePathError):
        object_entries(tmp_path)


def test_content_objects_cannot_be_used_as_namespace_parents(tmp_path: Path) -> None:
    object_path(tmp_path, "sha256:" + "4" * 64, label="pack")
    with pytest.raises(StorePathError, match="parent is already a content object"):
        object_path(tmp_path, "sha256:" + "5" * 64, label="pack/1.0")


@pytest.mark.skipif(os.name == "nt", reason="requires unprivileged symlinks")
def test_linked_object_or_index_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    (store / "pack").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StorePathError):
        object_path(store, "sha256:" + "3" * 64, label="pack/1.0")
    (store / ".objects.json").symlink_to(outside / "index")
    with pytest.raises(StorePathError):
        object_entries(store)
