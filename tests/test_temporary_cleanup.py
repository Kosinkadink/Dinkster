"""Engine-local temporary vault collection behavior."""

from __future__ import annotations

import os
import time
from contextlib import closing
from pathlib import Path

import pytest
from dinkster_assets import AssetVault, digest_bytes
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import hold_produced_asset_read
from dinkster_server.temporary_cleanup import inspect_temporary

from tests.platform_support import symlink_or_skip


def _adopt(vault: AssetVault, data: bytes) -> str:
    digest = digest_bytes(data)
    inventory = ProducedAssetInventory(vault.root)
    inventory.pending(digest, len(data))
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    inventory.adopted(digest, len(data))
    return digest


def _vault(tmp_path: Path, name: str) -> AssetVault:
    vault = AssetVault(tmp_path / name)
    ProducedAssetInventory.initialize_vault(vault.root, existing_empty=True)
    return vault


def test_collects_oldest_entries_across_vaults_until_under_limit(tmp_path: Path) -> None:
    first, second = _vault(tmp_path, "first"), _vault(tmp_path, "second")
    old, middle, newest = (
        _adopt(first, b"a" * 3),
        _adopt(second, b"b" * 11),
        _adopt(first, b"c" * 5),
    )
    for vault, digest, modified in ((first, old, 1), (second, middle, 2), (first, newest, 3)):
        path = vault.resolve(digest)
        assert path is not None
        os.utime(path, ns=(modified, modified))
    newest_path = first.resolve(newest)
    assert newest_path is not None
    newest_bytes = sum(
        path.stat().st_size for path in newest_path.parent.glob(newest_path.name + "*")
    )
    old_bytes = (
        sum(
            path.stat().st_size
            for vault in (first, second)
            for path in vault.root.glob("*/*")
            if path.is_file()
        )
        - newest_bytes
    )
    result = inspect_temporary((first, second), collect=True, limit_bytes=newest_bytes)
    assert result["reclaimedBytes"] == old_bytes
    assert result["vaultBytes"] == newest_bytes
    assert first.resolve(old) is None
    assert second.resolve(middle) is None
    assert first.resolve(newest) is not None


def test_limit_counts_conversion_sidecars_and_reclaims_them_with_the_source(tmp_path: Path) -> None:
    vault = _vault(tmp_path, "engine")
    old = _adopt(vault, b"old")
    newest = _adopt(vault, b"newer")
    path = vault.resolve(old)
    assert path is not None
    sidecar = path.with_name(path.name + ".preview.webp")
    sidecar.write_bytes(b"x" * 11)
    newest_path = vault.resolve(newest)
    assert newest_path is not None
    old_bytes = sum(item.stat().st_size for item in path.parent.glob(path.name + "*"))
    newest_bytes = sum(
        item.stat().st_size for item in newest_path.parent.glob(newest_path.name + "*")
    )
    measured = inspect_temporary((vault,))
    assert measured["vaultBytes"] == old_bytes + newest_bytes
    result = inspect_temporary((vault,), collect=True, limit_bytes=old_bytes + newest_bytes - 1)
    assert result["reclaimedBytes"] == old_bytes
    assert result["vaultBytes"] == newest_bytes
    assert result["adoptedBytes"] == 5
    assert vault.resolve(old) is None
    assert not sidecar.exists()
    assert vault.resolve(newest) is not None


@pytest.mark.parametrize("source_present", [False, True])
def test_collects_orphaned_sidecars_without_inventory(tmp_path: Path, source_present: bool) -> None:
    vault = _vault(tmp_path, "engine")
    data = b"interrupted publication"
    digest = digest_bytes(data)
    hexpart = digest.split(":", 1)[1]
    path = vault.root / hexpart[:2] / hexpart
    path.parent.mkdir()
    if source_present:
        path.write_bytes(data)
    sidecar = path.with_name(path.name + ".preview.webp")
    sidecar.write_bytes(b"preview bytes")
    expected = len(b"preview bytes") + (len(data) if source_present else 0)
    assert inspect_temporary((vault,))["vaultBytes"] == expected
    result = inspect_temporary((vault,), collect=True)
    assert result["reclaimedBytes"] == expected
    assert result["vaultBytes"] == 0
    assert not path.exists()
    assert not sidecar.exists()


def test_busy_digest_does_not_block_an_unrelated_eligible_digest(tmp_path: Path) -> None:
    vault = _vault(tmp_path, "engine")
    busy, free = _adopt(vault, b"busy"), _adopt(vault, b"free bytes")
    path = vault.resolve(busy)
    assert path is not None
    lease = hold_produced_asset_read(path)
    assert lease is not None
    with closing(lease):
        result = inspect_temporary((vault,), collect=True)
    assert result["reclaimedFiles"] == 1
    assert result["retainedFiles"] == 1
    assert vault.resolve(busy) is not None
    assert vault.resolve(free) is None


def test_protects_explicit_and_active_and_pending_adoptions(tmp_path: Path) -> None:
    vault = _vault(tmp_path, "engine")
    explicit = _adopt(vault, b"explicit")
    active_since_ns = time.time_ns()
    active, pending = _adopt(vault, b"active"), digest_bytes(b"pending")
    inventory = ProducedAssetInventory(vault.root)
    inventory.pending(pending, len(b"pending"))
    with vault.writer(pending) as writer:
        writer.write(b"pending")
        writer.commit()
    result = inspect_temporary(
        (vault,), frozenset({explicit}), collect=True, active_since_ns=active_since_ns
    )
    assert result["reclaimedBytes"] == 0
    assert vault.resolve(explicit) is not None
    assert vault.resolve(active) is not None
    assert vault.resolve(pending) is not None
    assert {row.digest for row in inventory.entries()} == {explicit, active, pending}


def test_never_collects_unsupplied_durable_vault(tmp_path: Path) -> None:
    temporary, durable = _vault(tmp_path, "temporary"), _vault(tmp_path, "durable")
    temporary_digest, durable_digest = _adopt(temporary, b"temporary"), _adopt(durable, b"durable")
    result = inspect_temporary((temporary,), collect=True)
    assert result["reclaimedFiles"] == 1
    assert temporary.resolve(temporary_digest) is None
    assert durable.resolve(durable_digest) is not None


def test_unsafe_links_are_preserved_without_touching_safe_entries(tmp_path: Path) -> None:
    vault = _vault(tmp_path, "engine")
    safe = _adopt(vault, b"safe")
    unsafe = _adopt(vault, b"unsafe")
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    path = vault.resolve(unsafe)
    assert path is not None
    path.unlink()
    symlink_or_skip(path, target)
    result = inspect_temporary((vault,), collect=True)
    assert result["reclaimedFiles"] == 1
    assert isinstance(result["failedItems"], int) and result["failedItems"] >= 1
    assert vault.resolve(safe) is None
    assert path.is_symlink()
    assert target.read_bytes() == b"outside"
