"""Operator accounting and gated collection with reference owners held."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import threading
from collections.abc import Generator, Sequence
from contextlib import AsyncExitStack, ExitStack, closing, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, AssetVault, LibraryStore, open_verified
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    claim_inactive_produced_vault,
    hold_produced_vault,
    is_link_or_junction,
    produced_job_scratch_root,
)
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_library import LibraryClient, LibraryError
from dinkster_library.produced_collection import collect_vault, produced_vault_bytes
from dinkster_protocol import CacheStore
from dinkster_values import MEBIBYTE, TypeRegistry

from .auth import principal_for
from .history import HistoryStore
from .library import run_owned_thread

RECOVERY_KEY = web.AppKey("produced_publication_recovery", dict[str, int])
POLICY_KEY = web.AppKey("produced_policy_collection", dict[str, object])
POLICY_WAKE_KEY = web.AppKey("produced_policy_wakeup", asyncio.Event)


class CleanupPolicy:
    """A server's saved-edit view; the supplied helper remains caller-owned."""

    def __init__(self, library: LibraryClient) -> None:
        if "cleanup.policy" not in cast("list[str]", library.hello["capabilities"]):
            raise LibraryError("capability", "library helper does not support cleanup policy")
        self._library = library
        self._lock = threading.Lock()
        self._revision = 0
        self.cleanup_limit()

    def cleanup_limit(self, *, observe: bool = True) -> int | None:
        with self._lock:
            record = cast("dict[str, object]", self._library.call("cleanup.policy.get")["policy"])
            if observe:
                self._revision = cast(int, record["revision"])
            return cast("int | None", cast("dict[str, object]", record["values"])["limitBytes"])

    def set_cleanup_limit(self, limit: int | None) -> None:
        with self._lock:
            record = cast(
                "dict[str, object]",
                self._library.call(
                    "cleanup.policy.patch", revision=self._revision, patch={"limitBytes": limit}
                )["policy"],
            )
            self._revision = cast(int, record["revision"])


class EngineReferences:
    """Publish one engine's roots; the caller retains the helper and store lifetimes."""

    def __init__(
        self,
        library: LibraryClient,
        engine: str,
        cache: CacheStore,
        history: HistoryStore,
        registry: TypeRegistry,
        scratch_root: Path,
        *,
        vaults: tuple[AssetVault, ...] = (),
    ) -> None:
        self._library = library
        self._vaults = tuple(item.root.absolute() for item in vaults)
        self._history = history
        self._registry = registry
        self._scratch_root = scratch_root
        self._memories: dict[int, MemoryLRUCache] = {}
        self._disks: dict[Path, DiskCacheStore] = {}
        self._lock = threading.Lock()
        self._publication_lock = asyncio.Lock()
        self._inspecting = False
        self._complete = False
        manifest = cast(
            "dict[str, object]", library.call("references.engine.attach", engine=engine)["manifest"]
        )
        self._revision = cast(int, manifest["revision"])

        def bind(layer: CacheStore) -> None:
            if isinstance(layer, LayeredCache):
                for child in layer.layers:
                    bind(child)
            elif isinstance(layer, MemoryLRUCache):
                layer.set_asset_reference_invalidator(self.invalidate)
                self._memories[id(layer)] = layer
            elif isinstance(layer, DiskCacheStore):
                # Every alias must invalidate, though a shared root is locked only once.
                layer.set_asset_reference_invalidator(self.invalidate)
                self._disks[layer.root.resolve()] = layer
            else:
                raise AssetError("cache asset references cannot be published")

        bind(cache)
        history.set_asset_reference_invalidator(library, self.invalidate)

    def _patch(self, **patch: object) -> dict[str, object]:
        # A failed reply can follow a committed complete view. Writers must retry
        # invalidation rather than trusting an unknown outcome to be incomplete.
        self._complete = True
        manifest = cast(
            "dict[str, object]",
            self._library.call("references.engine.patch", revision=self._revision, patch=patch)[
                "manifest"
            ],
        )
        self._revision = cast(int, manifest["revision"])
        self._complete = manifest["complete"] is True
        return manifest

    def invalidate(self) -> None:
        """Refuse the writer if the helper cannot invalidate a previously complete view."""
        with self._lock:
            # A confirmed incomplete view cannot be reserved. Otherwise inspection
            # owns the pipe, so inline admission must refuse without waiting for it.
            if self._inspecting and self._complete:
                raise LibraryError("busy", "asset references are reserved for collection")
            if self._complete:
                self._patch(complete=False)

    async def publish(self) -> dict[str, object]:
        """No new admission can pass the held barriers before the helper commits its view."""
        async with self._publication_lock:
            await run_owned_thread(self.invalidate)
            async with AsyncExitStack() as owners:
                roots: set[str] = set()
                for key in sorted(self._memories):
                    roots.update(
                        await owners.enter_async_context(
                            self._memories[key].hold_asset_references(self._registry)
                        )
                    )
                return await run_owned_thread(lambda: self._publish(roots))

    async def inspect(
        self, *, collect: bool = False, limit_bytes: int | None = None
    ) -> dict[str, object]:
        """Collect only the host-bound vaults, inside the helper owning the reservation."""
        if type(collect) is not bool or (
            limit_bytes is not None and (type(limit_bytes) is not int or limit_bytes < 0)
        ):
            raise ValueError("collection needs a boolean and nonnegative byte limit")
        status: dict[str, object] = {
            "temporaryBytes": None,
            "vaultBytes": None,
            "adoptedBytes": None,
            "reclaimableBytes": None,
            "knownRetainedAssets": None,
            "vaultOwnershipAvailable": None,
            "blocked": [],
            "collectionEnabled": False,
        }
        if collect:
            status.update(reclaimedBytes=0, reclaimedFiles=0)
        try:
            if not self._vaults:
                raise AssetError("engine produced vaults are not configured")
            await self.publish()
            async with self._publication_lock:
                with self._lock:
                    self._inspecting = True
                try:
                    return await run_owned_thread(
                        lambda: self._inspect(status, collect, limit_bytes)
                    )
                finally:
                    with self._lock:
                        self._inspecting = False
        except (AssetError, OSError, LibraryError):
            status["blocked"] = ["reference_state_unavailable"]
            status["collectionEnabled"] = False
            status["reclaimableBytes"] = None
            return status

    def _inspect(
        self, status: dict[str, object], collect: bool, limit_bytes: int | None
    ) -> dict[str, object]:
        status["temporaryBytes"] = _scratch_bytes(self._scratch_root)
        vaults = [str(root) for root in self._vaults]
        self._library.call("retention.census", vaults=vaults)
        reservation = self._library.call("references.acquire")
        lease = reservation["lease"]
        try:
            retained = len(cast("list[str]", reservation["digests"]))
            while reservation["cursor"] is not None:
                reservation = self._library.call(
                    "references.page", lease=lease, cursor=reservation["cursor"]
                )
                retained += len(cast("list[str]", reservation["digests"]))
            status["knownRetainedAssets"] = retained
            if collect:
                for pack, root in _scratch_roots(self._scratch_root):
                    job, attempt = ProducedScratchState(root).owner()
                    previous = {
                        key: cast(int, status[key]) for key in ("reclaimedBytes", "reclaimedFiles")
                    }
                    status.update(reclaimedBytes=None, reclaimedFiles=None)
                    result = self._library.call(
                        "references.retire",
                        lease=lease,
                        scratch=str(self._scratch_root.absolute()),
                        scope=pack.parent.name,
                        pack=pack.name,
                        job=job,
                        attempt=attempt,
                        vaults=vaults,
                    )
                    for key in ("reclaimedBytes", "reclaimedFiles"):
                        status[key] = previous[key] + cast(int, result[key])
            previous = (
                {key: cast(int, status[key]) for key in ("reclaimedBytes", "reclaimedFiles")}
                if collect
                else {}
            )
            if collect:
                status.update(reclaimedBytes=None, reclaimedFiles=None)
            result = self._library.call(
                "references.collect",
                lease=lease,
                vaults=vaults,
                collect=collect,
                limitBytes=limit_bytes,
            )
            for key in ("vaultBytes", "adoptedBytes", "reclaimableBytes"):
                status[key] = result[key]
            if collect:
                for key in ("reclaimedBytes", "reclaimedFiles"):
                    status[key] = previous[key] + cast(int, result[key])
                status["temporaryBytes"] = _scratch_bytes(self._scratch_root)
            status["vaultOwnershipAvailable"] = True
            if result["failedItems"]:
                status["blocked"] = ["candidate_accounting_unavailable"]
            else:
                status["collectionEnabled"] = True
            return status
        finally:
            self._library.call("references.release", lease=lease)

    def _publish(self, roots: set[str]) -> dict[str, object]:
        with ExitStack() as owners:
            for root in sorted(self._disks):
                roots.update(owners.enter_context(self._disks[root].hold_asset_references()))
            accepted, idle = owners.enter_context(self._history.hold_engine_references())
            roots.update(accepted)
            held_jobs: set[str] = set()
            for pack, root in _scratch_roots(self._scratch_root):
                state = owners.enter_context(
                    _hold_scratch_job(self._scratch_root, pack, root, held_jobs=held_jobs)
                )
                roots.update(digest for digest, _, offered in state.publications() if offered)
            with self._lock:
                # Accepted jobs can create scratch directories after this scan.
                return self._patch(digests=sorted(roots), complete=idle)


async def maintain_cleanup_policy(app: web.Application) -> None:
    """Recover deferred publications and apply policy without bypassing owners."""
    from .app import STATE_KEY
    from .library import LIBRARY_KEY, recover_asset_observations, run_media_publication

    library = app[LIBRARY_KEY]
    while True:
        try:
            await asyncio.wait_for(app[POLICY_WAKE_KEY].wait(), timeout=60)
        except TimeoutError:
            pass
        app[POLICY_WAKE_KEY].clear()
        await run_media_publication(library, lambda: recover_asset_observations(library))
        result: dict[str, object] = {
            "collectionEnabled": False,
            "reclaimedBytes": 0,
            "reclaimedFiles": 0,
        }
        try:
            policy = library.cleanup_policy
            limit = await run_owned_thread(
                library.store.cleanup_limit
                if policy is None
                else partial(policy.cleanup_limit, observe=False)
            )
        except (AssetError, sqlite3.Error, LibraryError):
            result["blocked"] = ["policy_state_unavailable"]
        else:
            result["limitBytes"] = limit
            if limit is None:
                result["blocked"] = ["policy_unlimited"]
            else:
                state = app[STATE_KEY]
                result.update(
                    await inspect_cleanup(
                        library.vault,
                        library.store,
                        state.history,
                        state.engine.cache,
                        registry=state.engine.registry,
                        scratch_root=library.scratch_root,
                        preview_vaults=library.preview_vaults,
                        collect=True,
                        limit_bytes=limit,
                        collection_blocked=("helper_collection_unavailable",)
                        if library.cleanup_policy is not None
                        else (),
                    )
                )
        app[POLICY_KEY].clear()
        app[POLICY_KEY].update(result)


async def inspect_cleanup(
    vault: AssetVault,
    store: LibraryStore,
    history: HistoryStore | None,
    cache: CacheStore,
    *,
    registry: TypeRegistry | None = None,
    scratch_root: Path | None = None,
    preview_vaults: tuple[AssetVault, ...] = (),
    collect: bool = False,
    limit_bytes: int | None = None,
    collection_blocked: Sequence[str] = (),
) -> dict[str, object]:
    """Hold memory, then disk, history and library in publication order.

    Blocking owners are acquired and released in one owned worker; cancellation
    must not release RAM ownership before that worker settles. This inspects
    known owners only. Collection additionally requires an evaluated complete
    client census and every reference/publication barrier held through deletion.
    """
    if registry is None:
        registry = TypeRegistry()
    memories: dict[int, MemoryLRUCache] = {}
    disks: dict[Path, DiskCacheStore] = {}
    blocked = list(collection_blocked)

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
                    await owners.enter_async_context(memories[key].hold_asset_references(registry))
                )
        except AssetError:
            blocked.append("cache_reference_state_unavailable")
        return await run_owned_thread(
            lambda: _inspect(
                vault,
                store,
                history,
                disks,
                roots,
                blocked,
                scratch_root,
                collect,
                limit_bytes,
                preview_vaults,
            )
        )


def _scratch_roots(scratch: Path) -> Generator[tuple[Path, Path]]:
    def present_directory(path: Path) -> bool:
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return False
        if is_link_or_junction(path) or not stat.S_ISDIR(mode):
            raise AssetError("scratch accounting requires a real directory")
        return True

    if not present_directory(scratch):
        return

    for scope in ("packs", "groups"):
        folder = scratch / scope
        if not present_directory(folder):
            continue
        for pack in folder.iterdir():
            if is_link_or_junction(pack) or not pack.is_dir():
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
                    if is_link_or_junction(root) or not root.is_dir():
                        raise AssetError("scratch lease accounting is unavailable")
                    continue
                if is_link_or_junction(root) or not root.is_dir():
                    raise AssetError("scratch job accounting is unavailable")
                yield pack, root


@contextmanager
def _hold_scratch_job(
    scratch: Path, pack: Path, root: Path, *, held_jobs: set[str] | None = None
) -> Generator[ProducedScratchState]:
    state = ProducedScratchState(root)
    job, attempt = state.owner()
    if root != produced_job_scratch_root(pack, job, attempt):
        raise AssetError("scratch job location disagrees with its owner")
    # One inspection may retain several pack stores belonging to the same job.
    already_held = held_jobs is not None and job in held_jobs
    claim = nullcontext(True) if already_held else claim_inactive_produced_job(scratch, job)
    with claim as inactive_job:
        if not inactive_job:
            raise AssetError("scratch job is active")
        if held_jobs is not None:
            held_jobs.add(job)
        try:
            with claim_inactive_produced_scratch(root) as inactive_store:
                if not inactive_store or state.owner() != (job, attempt):
                    raise AssetError("scratch store is active or changed")
                state.publications()
                yield state
        finally:
            if held_jobs is not None and not already_held:
                held_jobs.remove(job)


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
                    if is_link_or_junction(path):
                        raise AssetError("scratch accounting must not follow symlinks")
                for name in files:
                    path = directory / name
                    if not path.is_file():
                        raise AssetError("scratch accounting contains an unsafe file")
                    size += path.stat().st_size
    return size


def recover_produced_publications(
    vault: AssetVault,
    scratch: Path,
    *,
    job: str | None = None,
    store: LibraryStore | None = None,
    preview_vaults: tuple[AssetVault, ...] = (),
) -> dict[str, int]:
    """Complete handoffs and reconcile inactive scratch behind the integration gate."""
    recovered = 0
    blocked = 0
    reclaimed = {"reclaimedBytes": 0, "reclaimedFiles": 0}
    eligible = False
    try:
        for pack, root in _scratch_roots(scratch):
            try:
                if job is not None and ProducedScratchState(root).owner()[0] != job:
                    continue
                with _hold_scratch_job(scratch, pack, root) as state:
                    if job is not None and state.owner()[0] != job:
                        raise AssetError("scratch job changed during terminal recovery")
                    publications = state.publications()
                    destinations = {
                        digest: next(
                            (item for item in preview_vaults if item.resolve(digest) is not None),
                            vault,
                        )
                        for digest, _, offered in publications
                        if offered
                    }
                    for digest, size, offered in publications:
                        if offered:
                            _recover_publication(destinations[digest], root, digest, size)
                            recovered += 1
                    if store is not None:
                        reasons: list[str] = []
                        with store.hold_asset_references(collection_blocked=reasons):
                            if not reasons:
                                eligible = True
                                _remove_scratch((vault, *preview_vaults), root, state, reclaimed)
            except (AssetError, OSError):
                blocked += 1
    except (AssetError, OSError):
        blocked += 1
    result = {"verifiedPublications": recovered, "blockedInspections": blocked}
    if eligible:
        result.update(reclaimed)
    return result


def _remove_scratch(
    vaults: tuple[AssetVault, ...],
    root: Path,
    state: ProducedScratchState,
    reclaimed: dict[str, int],
) -> None:
    """Remove only the claimed job's temporary files while durable handoffs stay held."""
    publications = state.publications()
    with ExitStack() as claims:
        for vault in sorted(vaults, key=lambda item: str(item.root)):
            claims.enter_context(closing(hold_produced_vault(vault.root)))
        for digest, size, offered in publications:
            if not offered:
                continue
            path = next(
                (path for item in vaults if (path := item.resolve(digest)) is not None), None
            )
            if path is None:
                raise AssetError("offered publication has no durable engine copy")
            with open_verified(path, digest) as handle:
                if os.fstat(handle.fileno()).st_size != size:
                    raise AssetError("durable publication size disagrees with scratch")
        # The vault-wide lease keeps verified copies safe without retaining N descriptors.

        state.retire(reclaimed)


def _recover_publication(vault: AssetVault, scratch: Path, digest: str, size: int) -> None:
    hexpart = digest.split(":", 1)[1]
    destination = vault.root / hexpart[:2] / hexpart
    if any(is_link_or_junction(path) for path in (vault.root, destination.parent, destination)):
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
    if is_link_or_junction(source.parent) or is_link_or_junction(source):
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


def _inspect(
    vault: AssetVault,
    store: LibraryStore,
    history: HistoryStore | None,
    disks: dict[Path, DiskCacheStore],
    roots: set[str],
    blocked: list[str],
    scratch_root: Path | None,
    collect: bool,
    limit_bytes: int | None,
    preview_vaults: tuple[AssetVault, ...],
) -> dict[str, object]:
    vaults = (vault, *preview_vaults)
    temporary_bytes: int | None = None
    vault_bytes: int | None = None
    adopted_bytes: int | None = None
    reclaimable_bytes: int | None = None
    retained_assets: int | None = None
    vault_ownership_available: bool | None = None
    collection: dict[str, int | None] = {"reclaimedBytes": 0, "reclaimedFiles": 0}
    try:
        if scratch_root is None:
            raise AssetError("scratch root is not configured")
        temporary_bytes = _scratch_bytes(scratch_root)
    except (AssetError, OSError):
        blocked.append("temporary_accounting_unavailable")
    try:
        vault_bytes = sum(produced_vault_bytes(item.root) for item in vaults)
    except (AssetError, OSError):
        blocked.append("vault_accounting_unavailable")
    try:
        if any(is_link_or_junction(item.root) for item in vaults):
            raise AssetError("adoption accounting must not follow symlinks")
        inventory = tuple(
            row for item in vaults for row in ProducedAssetInventory(item.root).entries()
        )
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
            if scratch_root is not None:
                held_jobs: set[str] = set()
                for pack, root in _scratch_roots(scratch_root):
                    state = owners.enter_context(
                        _hold_scratch_job(scratch_root, pack, root, held_jobs=held_jobs)
                    )
                    # Partial scratch removal must remain retryable until journal retirement.
                    roots.update(digest for digest, _, offered in state.publications() if offered)
            references = owners.enter_context(
                store.hold_asset_references(sorted(roots), collection_blocked=blocked)
            )
            retained_assets = len(references)
            try:
                claims = [
                    owners.enter_context(claim_inactive_produced_vault(item.root))
                    for item in sorted(vaults, key=lambda item: str(item.root))
                ]
                vault_ownership_available = all(claims)
                if not vault_ownership_available:
                    blocked.append("vault_in_use")
                elif not blocked:
                    remaining = {
                        item.root: sum(
                            row.size
                            for row in ProducedAssetInventory(item.root).entries()
                            if row.adopted_at_ns is not None
                            and item.resolve(row.digest) is not None
                        )
                        for item in vaults
                    }
                    reclaimable_bytes = 0
                    for item in vaults:
                        limit = (
                            max(0, limit_bytes - (sum(remaining.values()) - remaining[item.root]))
                            if limit_bytes is not None
                            else None
                        )
                        measured = collect_vault(
                            item.root, references, limit, collect=collect, delete=item.delete
                        )
                        candidate_bytes = measured.pop("reclaimableBytes")
                        if candidate_bytes is None:
                            blocked.append("candidate_accounting_unavailable")
                        else:
                            reclaimable_bytes += candidate_bytes
                        for key, value in measured.items():
                            assert value is not None
                            collection[key] = (collection.get(key) or 0) + value
                        remaining[item.root] = sum(
                            row.size
                            for row in ProducedAssetInventory(item.root).entries()
                            if row.adopted_at_ns is not None
                            and item.resolve(row.digest) is not None
                        )
                    # A deletion response describes the remaining store, not its old snapshot.
                    vault_bytes = sum(produced_vault_bytes(item.root) for item in vaults)
                    adopted_bytes = sum(
                        row.size
                        for item in vaults
                        for row in ProducedAssetInventory(item.root).entries()
                        if row.adopted_at_ns is not None
                    )
            except (AssetError, OSError):
                blocked.append("vault_ownership_state_unavailable")
    except (AssetError, OSError):
        blocked.append("reference_state_unavailable")
    status: dict[str, object] = {
        "temporaryBytes": temporary_bytes,
        "vaultBytes": vault_bytes,
        "adoptedBytes": adopted_bytes,
        "reclaimableBytes": None if blocked else reclaimable_bytes,
        "knownRetainedAssets": retained_assets,
        "vaultOwnershipAvailable": vault_ownership_available,
        "blocked": sorted(set(blocked)),
        "collectionEnabled": not blocked,
    }
    if collect:
        status.update(collection)
    return status


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
        registry=state.engine.registry,
        scratch_root=library.scratch_root,
        preview_vaults=library.preview_vaults,
        collection_blocked=("helper_collection_unavailable",)
        if library.cleanup_policy is not None
        else (),
    )
    status["lastPublicationRecovery"] = request.app.get(RECOVERY_KEY)
    status["lastPolicyCollection"] = request.app[POLICY_KEY] or None
    try:
        status["limitBytes"] = await run_owned_thread(library.storage_policy.cleanup_limit)
        status["policyAvailable"] = True
    except (AssetError, sqlite3.Error, LibraryError):
        status["limitBytes"] = None
        status["policyAvailable"] = False
    return web.json_response(status, headers={"Cache-Control": "no-store"})
