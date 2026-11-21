"""Produced-vault accounting and collection under reference and OS reservations."""

from __future__ import annotations

import stat
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import cast

from .asset_identity import AssetError, digest_file
from .client import LibraryError
from .produced_assets import ProducedAssetInventory
from .produced_scratch import (
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    claim_inactive_produced_vault,
    is_link_or_junction,
    produced_job_scratch_root,
)


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


@contextmanager
def claim_produced_vaults(selected: object) -> Generator[list[Path]]:
    """Keep trusted host-selected vaults stable for collection or quarantine."""
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
    with ExitStack() as owners:
        for root in sorted(roots):
            if not owners.enter_context(claim_inactive_produced_vault(root)):
                raise LibraryError("busy", "produced vault is in use")
        yield roots


def retire_reserved_scratch(
    arguments: dict[str, object],
    references: frozenset[str],
    *,
    progress: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Retire one trusted host-selected job store inside its reservation owner."""
    if set(arguments) != {"lease", "scratch", "scope", "pack", "job", "attempt", "vaults"}:
        raise LibraryError("invalid", "scratch retirement needs an exact job location and vaults")
    scratch, scope, pack = arguments["scratch"], arguments["scope"], arguments["pack"]
    job, attempt = arguments["job"], arguments["attempt"]
    if (
        not isinstance(scratch, str)
        or not Path(scratch).is_absolute()
        or ".." in Path(scratch).parts
        or scope not in ("packs", "groups")
        or not isinstance(pack, str)
        or not pack
        or pack in (".", "..")
        or "/" in pack
        or "\\" in pack
        or ":" in pack
        or not isinstance(job, str)
        or not job
        or type(attempt) is not int
        or attempt < 1
    ):
        raise LibraryError("invalid", "scratch retirement needs a canonical job location")
    scratch = Path(scratch)
    root = produced_job_scratch_root(scratch / cast(str, scope) / pack, job, attempt)
    if any(is_link_or_junction(path) for path in (root, *root.parents)):
        raise LibraryError("invalid", "scratch retirement must not follow links")
    reclaimed = {"reclaimedBytes": 0, "reclaimedFiles": 0}
    try:
        with (
            claim_inactive_produced_job(scratch, job) as inactive_job,
            claim_inactive_produced_scratch(root) as inactive_store,
        ):
            if not inactive_job or not inactive_store:
                raise LibraryError("busy", "produced scratch job or store is in use")
            state = ProducedScratchState(root)
            if state.owner() != (job, attempt):
                raise AssetError("scratch job location disagrees with its owner")
            publications = state.publications()
            with claim_produced_vaults(arguments["vaults"]) as vaults:
                inventories = {
                    vault: {row.digest: row for row in ProducedAssetInventory(vault).entries()}
                    for vault in vaults
                }
                for digest, size, offered in publications:
                    if not offered:
                        continue
                    if digest not in references:
                        raise AssetError("offered scratch publication has no reserved reference")
                    path = next(
                        (_path(vault, digest) for vault in vaults if _path(vault, digest).exists()),
                        None,
                    )
                    if path is None:
                        raise AssetError("offered publication has no durable engine copy")
                    adoption = inventories[path.parent.parent].get(digest)
                    if (
                        adoption is None
                        or adoption.adopted_at_ns is None
                        or adoption.size != size
                        or is_link_or_junction(path.parent)
                        or is_link_or_junction(path)
                        or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
                        or path.stat().st_size != size
                        or digest_file(path, progress=progress) != digest
                    ):
                        raise AssetError("durable publication disagrees with scratch")
                state.retire(reclaimed, progress=progress)
    except (AssetError, OSError) as error:
        raise LibraryError("collection", str(error)) from error
    return dict(reclaimed)


def collect_reserved(
    arguments: dict[str, object],
    references: frozenset[str],
    *,
    progress: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run inside the helper that owns the library reference reservation."""
    if set(arguments) - {"lease", "vaults", "collect", "limitBytes"}:
        raise LibraryError("invalid", "unknown collection field")
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
        with claim_produced_vaults(arguments.get("vaults")) as roots:
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
