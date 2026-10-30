"""Operator accounting with reference owners held, without destructive collection."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Generator
from contextlib import AsyncExitStack, ExitStack, contextmanager
from pathlib import Path

from aiohttp import web
from dinkster_assets import AssetError, AssetVault, LibraryStore, open_verified
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    claim_inactive_produced_vault,
    produced_job_scratch_root,
)
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_protocol import CacheStore
from dinkster_values import MEBIBYTE

from .auth import principal_for
from .history import HistoryStore
from .library import run_owned_thread

RECOVERY_KEY = web.AppKey("produced_publication_recovery", dict[str, int])


async def inspect_cleanup(
    vault: AssetVault,
    store: LibraryStore,
    history: HistoryStore | None,
    cache: CacheStore,
    *,
    scratch_root: Path | None = None,
) -> dict[str, object]:
    """Hold memory, then disk, history and library in publication order.

    Blocking owners are acquired and released in one owned worker; cancellation
    must not release RAM ownership before that worker settles. This inspects
    known owners only, not global completeness or eligibility for deletion.
    """
    memories: dict[int, MemoryLRUCache] = {}
    disks: dict[Path, DiskCacheStore] = {}
    blocked = ["owner_completeness_unknown"]

    def visit(layer: CacheStore) -> None:
        if isinstance(layer, LayeredCache):
            for child in layer.layers:
                visit(child)
        elif isinstance(layer, MemoryLRUCache):
            memories[id(layer)] = layer
        elif isinstance(layer, DiskCacheStore):
            disks[layer.root.resolve()] = layer
        else:
            blocked.append("cache_reference_state_unavailable")

    visit(cache)
    async with AsyncExitStack() as owners:
        roots: set[str] = set()
        try:
            for key in sorted(memories):
                roots.update(
                    await owners.enter_async_context(memories[key].hold_asset_references())
                )
        except AssetError:
            blocked.append("cache_reference_state_unavailable")
        return await run_owned_thread(
            lambda: _inspect(vault, store, history, disks, roots, blocked, scratch_root)
        )


def _scratch_roots(scratch: Path) -> Generator[tuple[Path, Path]]:
    def present_directory(path: Path) -> bool:
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(mode):
            raise AssetError("scratch accounting requires a real directory")
        return True

    if not present_directory(scratch):
        return

    for scope in ("packs", "groups"):
        folder = scratch / scope
        if not present_directory(folder):
            continue
        for pack in folder.iterdir():
            if pack.is_symlink() or not pack.is_dir():
                raise AssetError("scratch pack accounting is unavailable")
            produced = pack / "produced-assets"
            jobs = produced / "jobs"
            if not present_directory(produced):
                continue
            if any(path.name != "jobs" for path in produced.iterdir()):
                raise AssetError("unscoped scratch accounting is unavailable")
            if not present_directory(jobs):
                continue
            for root in jobs.iterdir():
                if root.name == ".leases":
                    if root.is_symlink() or not root.is_dir():
                        raise AssetError("scratch lease accounting is unavailable")
                    continue
                if root.is_symlink() or not root.is_dir():
                    raise AssetError("scratch job accounting is unavailable")
                yield pack, root


@contextmanager
def _hold_scratch_job(scratch: Path, pack: Path, root: Path) -> Generator[ProducedScratchState]:
    state = ProducedScratchState(root)
    job, attempt = state.owner()
    if root != produced_job_scratch_root(pack, job, attempt):
        raise AssetError("scratch job location disagrees with its owner")
    with claim_inactive_produced_job(scratch, job) as inactive_job:
        if not inactive_job:
            raise AssetError("scratch job is active")
        with claim_inactive_produced_scratch(root) as inactive_store:
            if not inactive_store or state.owner() != (job, attempt):
                raise AssetError("scratch store is active or changed")
            state.publications()
            yield state


def _scratch_bytes(scratch: Path) -> int:
    """Observe only recognized, inactive job stores without repairing their state."""

    def unreadable(error: OSError) -> None:
        raise error

    size = 0
    for pack, root in _scratch_roots(scratch):
        with _hold_scratch_job(scratch, pack, root):
            for directory, folders, files in root.walk(follow_symlinks=False, on_error=unreadable):
                for name in (*folders, *files):
                    path = directory / name
                    if path.is_symlink():
                        raise AssetError("scratch accounting must not follow symlinks")
                for name in files:
                    path = directory / name
                    if not path.is_file():
                        raise AssetError("scratch accounting contains an unsafe file")
                    size += path.stat().st_size
    return size


def recover_produced_publications(
    vault: AssetVault, scratch: Path, *, job: str | None = None
) -> dict[str, int]:
    """Complete offered handoffs under inactive-job claims, preserving all scratch."""
    recovered = 0
    blocked = 0
    try:
        for pack, root in _scratch_roots(scratch):
            try:
                if job is not None and ProducedScratchState(root).owner()[0] != job:
                    continue
                with _hold_scratch_job(scratch, pack, root) as state:
                    if job is not None and state.owner()[0] != job:
                        raise AssetError("scratch job changed during terminal recovery")
                    for digest, size, offered in state.publications():
                        if offered:
                            _recover_publication(vault, root, digest, size)
                            recovered += 1
            except (AssetError, OSError):
                blocked += 1
    except (AssetError, OSError):
        blocked += 1
    return {"verifiedPublications": recovered, "blockedInspections": blocked}


def _recover_publication(vault: AssetVault, scratch: Path, digest: str, size: int) -> None:
    hexpart = digest.split(":", 1)[1]
    destination = vault.root / hexpart[:2] / hexpart
    if any(path.is_symlink() for path in (vault.root, destination.parent, destination)):
        raise AssetError("publication recovery must not follow symlinks")
    inventory = ProducedAssetInventory(vault.root)
    try:
        destination_stat = destination.stat(follow_symlinks=False)
    except FileNotFoundError:
        destination_stat = None
    if destination_stat is not None:
        if not stat.S_ISREG(destination_stat.st_mode):
            raise AssetError("publication recovery requires a regular destination file")
        with open_verified(destination, digest) as handle:
            if os.fstat(handle.fileno()).st_size != size:
                raise AssetError("recovered publication size disagrees with its journal")
            inventory.confirm_existing(digest, size)
        return
    source = scratch / hexpart[:2] / hexpart
    if source.parent.is_symlink() or source.is_symlink():
        raise AssetError("publication recovery must not follow symlinks")
    if not stat.S_ISREG(source.stat(follow_symlinks=False).st_mode):
        raise AssetError("publication recovery requires a regular source file")
    # The exclusive scratch claim already owns this descriptor's lifetime.
    # Acquiring a second managed-read lease here would conflict with that claim.
    with source.open("rb") as handle:
        if os.fstat(handle.fileno()).st_size != size:
            raise AssetError("recoverable publication size disagrees with its journal")
        inventory.pending(digest, size)
        with vault.writer(digest) as writer:
            count = 0
            while chunk := handle.read(MEBIBYTE):
                count += len(chunk)
                if count > size:
                    raise AssetError("recoverable publication grew during adoption")
                writer.write(chunk)
            if count != size:
                raise AssetError("recoverable publication is truncated")
            writer.commit()
            inventory.adopted(digest, size)


def _vault_bytes(vault: AssetVault) -> int:
    if vault.root.is_symlink():
        raise AssetError("vault accounting must not follow symlinks")
    size = 0
    for shard in vault.root.iterdir():
        if len(shard.name) != 2 or any(char not in "0123456789abcdef" for char in shard.name):
            continue
        if shard.is_symlink() or not shard.is_dir():
            raise AssetError("vault shard accounting is unavailable")
        for path in shard.iterdir():
            if len(path.name) != 64 or any(char not in "0123456789abcdef" for char in path.name):
                continue
            if path.is_symlink() or not path.is_file() or not path.name.startswith(shard.name):
                raise AssetError("vault blob accounting is unavailable")
            size += path.stat().st_size
    return size


def _inspect(
    vault: AssetVault,
    store: LibraryStore,
    history: HistoryStore | None,
    disks: dict[Path, DiskCacheStore],
    roots: set[str],
    blocked: list[str],
    scratch_root: Path | None,
) -> dict[str, object]:
    temporary_bytes: int | None = None
    vault_bytes: int | None = None
    adopted_bytes: int | None = None
    retained_assets: int | None = None
    vault_ownership_available: bool | None = None
    try:
        if scratch_root is None:
            raise AssetError("scratch root is not configured")
        temporary_bytes = _scratch_bytes(scratch_root)
    except (AssetError, OSError):
        blocked.append("temporary_accounting_unavailable")
    try:
        vault_bytes = _vault_bytes(vault)
    except (AssetError, OSError):
        blocked.append("vault_accounting_unavailable")
    try:
        if vault.root.is_symlink():
            raise AssetError("adoption accounting must not follow symlinks")
        inventory = ProducedAssetInventory(vault.root).entries()
        adopted_bytes = sum(row.size for row in inventory if row.adopted_at_ns is not None)
        if any(row.adopted_at_ns is None for row in inventory):
            blocked.append("adoption_pending")
    except (AssetError, OSError):
        blocked.append("adoption_state_unavailable")
    try:
        with ExitStack() as owners:
            for root in sorted(disks):
                roots.update(owners.enter_context(disks[root].hold_asset_references()))
            if history is None:
                blocked.append("history_reference_state_unavailable")
            else:
                roots.update(owners.enter_context(history.hold_asset_references()))
            references = owners.enter_context(store.hold_asset_references(sorted(roots)))
            retained_assets = len(references)
            try:
                vault_ownership_available = owners.enter_context(
                    claim_inactive_produced_vault(vault.root)
                )
                if not vault_ownership_available:
                    blocked.append("vault_in_use")
            except (AssetError, OSError):
                blocked.append("vault_ownership_state_unavailable")
    except (AssetError, OSError):
        blocked.append("reference_state_unavailable")
    return {
        "temporaryBytes": temporary_bytes,
        "vaultBytes": vault_bytes,
        "adoptedBytes": adopted_bytes,
        "reclaimableBytes": None,
        "knownRetainedAssets": retained_assets,
        "vaultOwnershipAvailable": vault_ownership_available,
        "blocked": sorted(set(blocked)),
        "collectionEnabled": False,
    }


async def handle_cleanup_status(request: web.Request) -> web.Response:
    from .app import STATE_KEY
    from .library import LIBRARY_KEY

    if not principal_for(request).is_operator:
        raise web.HTTPForbidden(text="asset cleanup accounting requires an operator")
    if request.query:
        raise web.HTTPBadRequest(text="cleanup accounting accepts no filesystem targets")
    library = request.app[LIBRARY_KEY]
    state = request.app[STATE_KEY]
    status = await inspect_cleanup(
        library.vault,
        library.store,
        state.history,
        state.engine.cache,
        scratch_root=library.scratch_root,
    )
    status["lastPublicationRecovery"] = request.app.get(RECOVERY_KEY)
    try:
        status["limitBytes"] = await run_owned_thread(library.store.cleanup_limit)
        status["policyAvailable"] = True
    except (AssetError, sqlite3.Error):
        status["limitBytes"] = None
        status["policyAvailable"] = False
    return web.json_response(status, headers={"Cache-Control": "no-store"})
