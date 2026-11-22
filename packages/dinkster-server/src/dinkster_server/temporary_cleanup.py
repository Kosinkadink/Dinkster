"""Engine-local collection of temporary vault content."""

from __future__ import annotations

import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dinkster_assets import AssetError, AssetVault
from dinkster_assets.produced import ProducedAssetAdoption, ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchLease,
    claim_inactive_produced_scratch,
    is_link_or_junction,
)

_HEX = frozenset("0123456789abcdef")


def _blob_path(vault: AssetVault, digest: str) -> Path:
    hexpart = digest.split(":", 1)[1]
    return vault.root / hexpart[:2] / hexpart


def _normal_file(path: Path, size: int | None) -> bool:
    try:
        details = path.stat(follow_symlinks=False)
        return stat.S_ISREG(details.st_mode) and details.st_size == size
    except FileNotFoundError:
        return size is None
    except OSError:
        return False


def _vault_entries(root: Path) -> tuple[int, int, list[tuple[str, int | None, int]]]:
    """Measure temporary files and enumerate their digest, size and modification time."""
    if is_link_or_junction(root):
        return 0, 1, []
    total = failed = 0
    entries: dict[str, tuple[int | None, int]] = {}
    try:
        children = tuple(root.iterdir())
    except OSError:
        return 0, 1, []
    for shard in children:
        if shard.name.startswith(".") or shard.name == ".produced-assets.sqlite":
            continue
        if len(shard.name) != 2 or any(char not in _HEX for char in shard.name):
            continue
        try:
            shard_mode = shard.stat(follow_symlinks=False).st_mode
        except OSError:
            failed += 1
            continue
        if is_link_or_junction(shard) or not stat.S_ISDIR(shard_mode):
            failed += 1
            continue
        try:
            blobs = tuple(shard.iterdir())
        except OSError:
            failed += 1
            continue
        for blob in blobs:
            digest_name = blob.name.split(".", 1)[0]
            if len(digest_name) != 64 or any(char not in _HEX for char in digest_name):
                continue
            try:
                blob_stat = blob.stat(follow_symlinks=False)
            except OSError:
                failed += 1
                continue
            if (
                is_link_or_junction(blob)
                or not stat.S_ISREG(blob_stat.st_mode)
                or not blob.name.startswith(shard.name)
            ):
                failed += 1
                continue
            total += blob_stat.st_size
            digest = f"blake3:{digest_name}"
            if blob.name == digest_name:
                entries[digest] = (blob_stat.st_size, blob_stat.st_mtime_ns)
            else:
                size, modified = entries.get(digest, (None, blob_stat.st_mtime_ns))
                if size is None:
                    entries[digest] = (None, min(modified, blob_stat.st_mtime_ns))
    return total, failed, [(digest, size, modified) for digest, (size, modified) in entries.items()]


@contextmanager
def _claim_digest(root: Path, digest: str) -> Generator[bool]:
    """Claim exactly one digest without making a whole vault unavailable."""
    hexpart = digest.split(":", 1)[1]
    readers = root / ".produced-readers"
    gate = readers / hexpart
    try:
        if is_link_or_junction(readers) or is_link_or_junction(gate):
            yield False
            return
        if not (readers / ".leases" / hexpart / "gate.lock").is_file():
            # No reader metadata exists until an asset is first opened. Seed the
            # gate so collection can distinguish that case from missing ownership.
            lease = ProducedScratchLease(gate)
            lease.close()
    except AssetError:
        yield False
        return
    claim = claim_inactive_produced_scratch(gate)
    try:
        inactive = claim.__enter__()
    except AssetError:
        yield False
        return
    try:
        yield inactive
    finally:
        claim.__exit__(None, None, None)


def _deletion_bytes(path: Path, size: int | None) -> int | None:
    if (
        is_link_or_junction(path.parent)
        or is_link_or_junction(path)
        or not _normal_file(path, size)
    ):
        return None
    total = size or 0
    try:
        for sibling in path.parent.glob(f"{path.name}.*"):
            details = sibling.stat(follow_symlinks=False)
            if is_link_or_junction(sibling) or not stat.S_ISREG(details.st_mode):
                return None
            total += details.st_size
    except OSError:
        return None
    return total


def inspect_temporary(
    vaults: tuple[AssetVault, ...],
    protected: frozenset[str] = frozenset(),
    *,
    collect: bool = False,
    limit_bytes: int | None = None,
    active_since_ns: int | None = None,
) -> dict[str, object]:
    """Measure or collect only files in the supplied engine-local temporary vaults."""
    if type(collect) is not bool or (
        limit_bytes is not None and (type(limit_bytes) is not int or limit_bytes < 0)
    ):
        raise ValueError("collection needs a boolean and nonnegative byte limit")
    if active_since_ns is not None and (type(active_since_ns) is not int or active_since_ns < 0):
        raise ValueError("active job time needs a nonnegative integer")

    blocked: list[str] = []
    entries: list[tuple[AssetVault, str, int | None, int]] = []
    adoptions: dict[tuple[Path, str], ProducedAssetAdoption] = {}
    vault_bytes = adopted_bytes = 0
    failed_items = 0
    retained = 0
    for vault in vaults:
        if is_link_or_junction(vault.root):
            blocked.append("temporary_vault_unavailable")
            continue
        size, unsafe, files = _vault_entries(vault.root)
        vault_bytes += size
        failed_items += unsafe
        entries.extend((vault, digest, size, modified) for digest, size, modified in files)
        try:
            rows = ProducedAssetInventory(vault.root).entries()
        except AssetError:
            rows = ()
        for row in rows:
            if row.adopted_at_ns is not None:
                adopted_bytes += row.size
                adoptions[vault.root, row.digest] = row

    eligible: list[tuple[AssetVault, str, int | None, int]] = []
    for vault, digest, size, modified in entries:
        if digest in protected or (active_since_ns is not None and modified >= active_since_ns):
            retained += 1
            continue
        path = _blob_path(vault, digest)
        if _deletion_bytes(path, size) is None:
            failed_items += 1
            continue
        eligible.append((vault, digest, size, modified))
    eligible.sort(key=lambda item: (item[3], item[1], str(item[0].root)))

    remaining = vault_bytes
    reclaimable = 0
    reclaimed_bytes = reclaimed_files = reclaimed_adopted_bytes = 0
    for vault, digest, content_size, _modified in eligible:
        path = _blob_path(vault, digest)
        with _claim_digest(vault.root, digest) as inactive:
            if not inactive:
                retained += 1
                continue
            size = _deletion_bytes(path, content_size)
            if size is None:
                failed_items += 1
                continue
            if not collect or (limit_bytes is not None and remaining <= limit_bytes):
                reclaimable += size
                continue
            try:
                deleted = vault.delete(digest)
                if (not deleted and content_size is not None) or path.exists():
                    raise AssetError("temporary vault deletion did not remove the bytes")
            except (AssetError, OSError):
                failed_items += 1
                continue
            adoption = adoptions.get((vault.root, digest))
            if adoption is not None:
                try:
                    ProducedAssetInventory(vault.root).forget_adoption(adoption)
                except (AssetError, OSError):
                    failed_items += 1
                reclaimed_adopted_bytes += adoption.size
            reclaimed_bytes += size
            reclaimed_files += 1
            remaining -= size

    status: dict[str, object] = {
        "temporaryBytes": vault_bytes if not collect else remaining,
        "vaultBytes": vault_bytes if not collect else remaining,
        "adoptedBytes": adopted_bytes - reclaimed_adopted_bytes,
        "reclaimableBytes": reclaimable,
        "knownRetainedAssets": len(protected),
        "vaultOwnershipAvailable": not blocked,
        "blocked": sorted(set(blocked)),
        "collectionEnabled": not blocked,
    }
    if collect:
        status.update(
            reclaimedBytes=reclaimed_bytes,
            reclaimedFiles=reclaimed_files,
            retainedFiles=retained,
            failedItems=failed_items,
        )
    return status
