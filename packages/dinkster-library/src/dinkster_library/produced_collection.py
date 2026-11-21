"""Produced-vault accounting and collection under reference and OS reservations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from .asset_identity import AssetError, digest_file
from .client import LibraryError
from .produced_assets import ProducedAssetInventory
from .produced_scratch import claim_inactive_produced_vault, is_link_or_junction


def produced_vault_bytes(root: Path, *, progress: Callable[[], None] | None = None) -> int:
    if is_link_or_junction(root):
        raise AssetError("vault accounting must not follow symlinks")
    size = 0
    for shard in root.iterdir():
        if len(shard.name) != 2 or any(char not in "0123456789abcdef" for char in shard.name):
            continue
        if is_link_or_junction(shard) or not shard.is_dir():
            raise AssetError("vault shard accounting is unavailable")
        for path in shard.iterdir():
            if progress is not None:
                progress()
            if len(path.name) != 64 or any(char not in "0123456789abcdef" for char in path.name):
                continue
            if (
                is_link_or_junction(path)
                or not path.is_file()
                or not path.name.startswith(shard.name)
            ):
                raise AssetError("vault blob accounting is unavailable")
            size += path.stat().st_size
    return size


def _path(root: Path, digest: str) -> Path:
    hexpart = digest.split(":", 1)[1]
    return root / hexpart[:2] / hexpart


def collect_vault(
    root: Path,
    references: frozenset[str],
    limit_bytes: int | None,
    *,
    collect: bool,
    delete: Callable[[str], object] | None = None,
    progress: Callable[[], None] | None = None,
) -> dict[str, int | None]:
    """Measure or delete verified adoptions under held roots and the vault claim."""
    inventory = ProducedAssetInventory(root)
    entries = inventory.entries()
    if any(row.adopted_at_ns is None for row in entries):
        raise AssetError("produced adoption is pending")
    produced_vault_bytes(root, progress=progress)
    remaining = sum(
        path.stat().st_size for row in entries if (path := _path(root, row.digest)).is_file()
    )
    reclaimed_bytes = reclaimed_files = reclaimable_bytes = failed = 0
    for row in entries:
        if progress is not None:
            progress()
        if row.digest in references:
            continue
        path = _path(root, row.digest)
        removed: list[tuple[Path, int]] = []
        try:
            if is_link_or_junction(path.parent) or is_link_or_junction(path):
                raise AssetError("collection must not follow symlinks")
            siblings = list(path.parent.glob(f"{path.name}.*"))
            if any(is_link_or_junction(p) or not p.is_file() for p in siblings):
                raise AssetError("collection sidecars are unsafe")
            if path.exists():
                # The exclusive vault claim replaces managed reader acquisition here.
                if (
                    path.stat().st_size != row.size
                    or digest_file(path, progress=progress) != row.digest
                ):
                    raise AssetError("produced bytes disagree with adoption")
                removed.append((path, row.size))
            removed.extend((p, p.stat().st_size) for p in siblings)
            if not collect or (limit_bytes is not None and remaining <= limit_bytes):
                reclaimable_bytes += sum(size for _, size in removed)
                continue
            if delete is None:
                for candidate, _ in removed:
                    candidate.unlink()
            else:
                delete(row.digest)
            inventory.forget_adoption(row)
        except (AssetError, OSError):
            failed += 1
        finally:
            for candidate, size in removed:
                if not candidate.exists():
                    reclaimed_bytes += size
                    reclaimed_files += 1
                    if candidate == path:
                        remaining -= size
    return {
        "reclaimedBytes": reclaimed_bytes,
        "reclaimedFiles": reclaimed_files,
        "reclaimableBytes": None if failed else reclaimable_bytes,
        "retainedFiles": sum(row.digest in references for row in entries),
        "failedItems": failed,
    }


def collect_reserved(
    arguments: dict[str, object],
    references: frozenset[str],
    *,
    progress: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run inside the helper that owns the library reference reservation."""
    if set(arguments) - {"lease", "vaults", "collect", "limitBytes"}:
        raise LibraryError("invalid", "unknown collection field")
    selected = arguments.get("vaults")
    if not isinstance(selected, list) or not selected:
        raise LibraryError("invalid", "collection needs produced vault directories")
    roots: list[Path] = []
    for value in cast("list[object]", selected):
        if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise LibraryError("invalid", "produced vault directories must be absolute")
        root = Path(value)
        if any(is_link_or_junction(path) for path in (root, *root.parents)):
            raise LibraryError("invalid", "produced vault directories must not contain links")
        if any(root.samefile(previous) for previous in roots):
            raise LibraryError("invalid", "produced vault directories must be distinct")
        roots.append(root)
    collect = arguments.get("collect", False)
    limit = arguments.get("limitBytes")
    if type(collect) is not bool or (limit is not None and (type(limit) is not int or limit < 0)):
        raise LibraryError("invalid", "collection needs a boolean and nonnegative byte limit")
    measured: dict[str, object] = {
        "reclaimedBytes": 0,
        "reclaimedFiles": 0,
        "reclaimableBytes": 0,
        "retainedFiles": 0,
        "failedItems": 0,
    }
    try:
        with ExitStack() as owners:
            for root in sorted(roots):
                if not owners.enter_context(claim_inactive_produced_vault(root)):
                    raise LibraryError("busy", "produced vault is in use")
            # Validate every vault before allowing deletion in any of them.
            remaining: dict[Path, int] = {}
            for root in roots:
                produced_vault_bytes(root, progress=progress)
                entries = ProducedAssetInventory(root).entries()
                if any(row.adopted_at_ns is None for row in entries):
                    raise AssetError("produced adoption is pending")
                remaining[root] = sum(
                    row.size for row in entries if _path(root, row.digest).is_file()
                )
            for root in roots:
                allowed = (
                    max(0, limit - (sum(remaining.values()) - remaining[root]))
                    if limit is not None
                    else None
                )
                result = collect_vault(
                    root, references, allowed, collect=collect, progress=progress
                )
                for key, value in result.items():
                    current = measured[key]
                    measured[key] = (
                        None if current is None or value is None else cast(int, current) + value
                    )
                remaining[root] = sum(
                    row.size
                    for row in ProducedAssetInventory(root).entries()
                    if _path(root, row.digest).is_file()
                )
            measured["vaultBytes"] = sum(
                produced_vault_bytes(root, progress=progress) for root in roots
            )
            measured["adoptedBytes"] = sum(
                row.size for root in roots for row in ProducedAssetInventory(root).entries()
            )
    except (AssetError, OSError) as error:
        raise LibraryError("collection", str(error)) from error
    return measured
