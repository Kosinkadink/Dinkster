"""Operator accounting with reference owners held, without destructive collection."""

from __future__ import annotations

from contextlib import AsyncExitStack, ExitStack
from pathlib import Path

from aiohttp import web
from dinkster_assets import AssetError, AssetVault, LibraryStore
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_protocol import CacheStore

from .auth import principal_for
from .history import HistoryStore
from .library import run_owned_thread


async def inspect_cleanup(
    vault: AssetVault, store: LibraryStore, history: HistoryStore | None, cache: CacheStore
) -> dict[str, object]:
    """Hold memory, then disk, history and library in publication order.

    Blocking owners are acquired and released in one owned worker; cancellation
    must not release RAM ownership before that worker settles. This inspects
    known owners only, not global completeness or eligibility for deletion.
    """
    memories: dict[int, MemoryLRUCache] = {}
    disks: dict[Path, DiskCacheStore] = {}
    blocked = ["owner_completeness_unknown", "temporary_accounting_unavailable"]

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
            lambda: _inspect(vault, store, history, disks, roots, blocked)
        )


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
) -> dict[str, object]:
    vault_bytes: int | None = None
    adopted_bytes: int | None = None
    retained_assets: int | None = None
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
        "temporaryBytes": None,
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
    status = await inspect_cleanup(library.vault, library.store, state.history, state.engine.cache)
    return web.json_response(status, headers={"Cache-Control": "no-store"})
