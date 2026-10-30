"""Operator accounting with reference owners held, without destructive collection."""

from __future__ import annotations

from contextlib import AsyncExitStack, ExitStack
from pathlib import Path

from aiohttp import web
from dinkster_assets import AssetError, AssetVault, LibraryStore
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    produced_job_scratch_root,
)
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_protocol import CacheStore

from .auth import principal_for
from .history import HistoryStore
from .library import run_owned_thread


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


def _scratch_bytes(scratch: Path) -> int:
    """Observe only recognized, inactive job stores without repairing their state."""
    if scratch.is_symlink() or (scratch.exists() and not scratch.is_dir()):
        raise AssetError("scratch accounting must not follow symlinks")

    def unreadable(error: OSError) -> None:
        raise error

    size = 0
    for scope in ("packs", "groups"):
        folder = scratch / scope
        if folder.is_symlink():
            raise AssetError("scratch accounting must not follow symlinks")
        if not folder.exists():
            continue
        for pack in folder.iterdir():
            if pack.is_symlink() or not pack.is_dir():
                raise AssetError("scratch pack accounting is unavailable")
            produced = pack / "produced-assets"
            jobs = produced / "jobs"
            if produced.is_symlink() or jobs.is_symlink():
                raise AssetError("scratch accounting must not follow symlinks")
            if not produced.exists():
                continue
            if any(path.name != "jobs" for path in produced.iterdir()):
                raise AssetError("unscoped scratch accounting is unavailable")
            if not jobs.exists():
                continue
            for root in jobs.iterdir():
                if root.name == ".leases":
                    if root.is_symlink() or not root.is_dir():
                        raise AssetError("scratch lease accounting is unavailable")
                    continue
                if root.is_symlink() or not root.is_dir():
                    raise AssetError("scratch job accounting is unavailable")
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
                        for directory, folders, files in root.walk(
                            follow_symlinks=False, on_error=unreadable
                        ):
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
    except (AssetError, OSError):
        blocked.append("reference_state_unavailable")
    return {
        "temporaryBytes": temporary_bytes,
        "vaultBytes": vault_bytes,
        "adoptedBytes": adopted_bytes,
        "reclaimableBytes": None,
        "knownRetainedAssets": retained_assets,
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
    return web.json_response(status, headers={"Cache-Control": "no-store"})
