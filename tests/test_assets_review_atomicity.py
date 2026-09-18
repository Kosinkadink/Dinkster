from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from dinkster_assets import (
    AssetError,
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


def test_writer_refuses_record_for_rebound_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_land = AssetWriter._claim_and_land  # pyright: ignore[reportPrivateUsage]

    def rebind_after_land(folder: Path, stem: str, suffix: str, tmp: Path) -> Path:
        final = real_land(folder, stem, suffix, tmp)
        replacement = folder / "replacement.tmp"
        replacement.write_bytes(b"attacker")
        os.replace(replacement, final)
        return final

    monkeypatch.setattr(AssetWriter, "_claim_and_land", staticmethod(rebind_after_land))
    with pytest.raises(AssetError, match="rebound"):
        AssetWriter(_Authority(tmp_path)).save_bytes(
            SaveTarget(mount="output", prefix="image"), b"complete", suffix=".bin"
        )
    assert load_write_records(tmp_path) == {}


def test_writer_refuses_temp_rebound_after_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_land = AssetWriter._claim_and_land  # pyright: ignore[reportPrivateUsage]

    def rebind_before_land(folder: Path, stem: str, suffix: str, tmp: Path) -> Path:
        replacement = folder / "replacement.tmp"
        replacement.write_bytes(b"attacker")
        os.replace(replacement, tmp)
        return real_land(folder, stem, suffix, tmp)

    monkeypatch.setattr(AssetWriter, "_claim_and_land", staticmethod(rebind_before_land))
    with pytest.raises(AssetError, match="rebound"):
        AssetWriter(_Authority(tmp_path)).save_bytes(
            SaveTarget(mount="output", prefix="image"), b"complete", suffix=".bin"
        )
    assert load_write_records(tmp_path) == {}


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


def test_vault_refuses_record_for_rebound_published_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"expected"
    vault = AssetVault(tmp_path)
    digest = digest_bytes(payload)
    target = vault._path(digest)  # pyright: ignore[reportPrivateUsage]
    real_replace = os.replace

    def rebind_after_publish(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if destination == target and source.name.startswith(".ingest-"):
            replacement = tmp_path / "replacement.tmp"
            replacement.write_bytes(b"attacker")
            real_replace(replacement, destination)

    monkeypatch.setattr(os, "replace", rebind_after_publish)
    with vault.writer(digest) as writer:
        writer.write(payload)
        with pytest.raises(AssetError, match="rebound"):
            writer.commit()
    assert vault.resolve_asset(digest).verification is None  # type: ignore[union-attr]


def test_vault_missing_target_during_record_write_is_best_effort(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path)
    digest = digest_bytes(b"payload")
    with vault.writer(digest) as writer:
        writer.write(b"payload")
        writer.commit()
    resolution = vault.resolve_asset(digest)
    assert resolution is not None and resolution.verification is not None
    resolution.path.unlink()

    writer._persist_verification(  # pyright: ignore[reportPrivateUsage]
        resolution.verification
    )
