from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from dinkster_assets import (
    AssetVault,
    AssetWriter,
    LocalAssetLibrary,
    SaveTarget,
    VaultError,
    digest_bytes,
    load_write_records,
)
from dinkster_assets.library import append_write_record


class _Authority:
    def __init__(self, root: Path) -> None:
        self.root = root

    def writable_root(self, mount_id: str) -> Path:
        return self.root


def test_writer_never_publishes_empty_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "image_00001.bin"
    real_link = os.link
    observed: list[int | None] = []

    def slow_link(source: Path, destination: Path) -> None:
        observed.append(final.stat().st_size if final.exists() else None)
        time.sleep(0.02)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", slow_link)
    ref = AssetWriter(_Authority(tmp_path)).save_bytes(
        SaveTarget(mount="output", prefix="image"), b"complete", suffix=".bin"
    )

    assert observed == [None]
    assert final.read_bytes() == b"complete"
    assert ref.size == len(b"complete")


def test_append_during_prune_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_assets.library as library_module

    old_digest = digest_bytes(b"old")
    new_digest = digest_bytes(b"new")
    append_write_record(tmp_path, "old.bin", old_digest, 3, 1)
    entered = threading.Event()
    real_load = library_module.load_write_records

    def slow_load(root: Path) -> dict[str, dict[str, object]]:
        entered.set()
        time.sleep(0.03)
        return real_load(root)

    monkeypatch.setattr(library_module, "load_write_records", slow_load)
    appender = threading.Thread(
        target=lambda: (
            entered.wait(),
            append_write_record(tmp_path, "new.bin", new_digest, 3, 2),
        )
    )
    appender.start()
    LocalAssetLibrary(tmp_path)._prune_sidecar(  # pyright: ignore[reportPrivateUsage]
        {"old.bin": {"digest": old_digest, "size": 3, "mtimeNs": 1}}
    )
    appender.join()

    assert load_write_records(tmp_path)["new.bin"]["digest"] == new_digest


def test_vault_replace_failure_rejects_corrupt_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"expected"
    vault = AssetVault(tmp_path)
    digest = digest_bytes(payload)
    target = vault._path(digest)  # pyright: ignore[reportPrivateUsage]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt!")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("forced replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with vault.writer(digest) as writer:
        writer.write(payload)
        with pytest.raises(VaultError, match="could not store verified asset"):
            writer.commit()
