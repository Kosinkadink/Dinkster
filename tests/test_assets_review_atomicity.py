from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from dinkster_assets import (
    AssetVault,
    AssetWriter,
    SaveTarget,
    VaultError,
    digest_bytes,
)


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
