"""Engine reference publication and temporary storage cleanup."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import threading
from collections.abc import Generator
from contextlib import AsyncExitStack, ExitStack, closing, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, AssetVault, LibraryStore, open_verified
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchLease,
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    hold_produced_vault,
    is_link_or_junction,
    produced_job_scratch_root,
)
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_library import EngineLifetime, LibraryClient, LibraryError
from dinkster_protocol import CacheStore
from dinkster_values import MEBIBYTE, TypeRegistry

from .auth import principal_for
from .history import HistoryStore
from .library import run_owned_thread

RECOVERY_KEY = web.AppKey("produced_publication_recovery", dict[str, int])
POLICY_KEY = web.AppKey("produced_policy_collection", dict[str, object])
POLICY_WAKE_KEY = web.AppKey("produced_policy_wakeup", asyncio.Event)


def inspect_engine_temporary(
    app: web.Application,
    *,
    collect: bool = False,
    limit_bytes: int | None = None,
    clear_staging: bool = False,
) -> dict[str, object]:
    from .app import STATE_KEY
    from .library import LIBRARY_KEY
    from .preflight import graph_asset_names
    from .temporary_cleanup import inspect_temporary

    library = app[LIBRARY_KEY]
    protected: set[str] = set()
    started: list[int] = []
    state = app.get(STATE_KEY)
    for job in state.queue.jobs() if state is not None else ():
        if job.state not in ("queued", "running"):
            continue
        protected.update(graph_asset_names(job.graph))
        if job.compiled_graph is not None:
            protected.update(graph_asset_names(job.compiled_graph.graph))
        if job.state == "running":
            started.append(int((job.started_at or job.submitted_at) * 1_000_000_000))
        for outputs in job.result_offers.values():
            for offer in outputs.values():
                protected.update(cast("list[str]", offer["digests"]))
    if clear_staging and not started:
        # Startup and clean stop have no writers; an interrupted ingest has no CAS name.
        for vault in library.temporary_vaults:
            if is_link_or_junction(vault.root):
                continue
            for path in vault.root.glob(".ingest-*"):
                if not is_link_or_junction(path) and path.is_file():
                    path.unlink(missing_ok=True)
    # Selection and unlink stay on the queue's event loop: a new submission
    # cannot arrive between taking its input snapshot and deleting a file.
    return inspect_temporary(
        library.temporary_vaults,
        frozenset(protected),
        collect=collect,
        limit_bytes=limit_bytes,
        active_since_ns=min(started) if started else None,
    )


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
        lifetime: EngineLifetime | None = None,
    ) -> None:
        self._library = library
        self._lifetime = lifetime
        self._configured_vaults = tuple(item.root.absolute() for item in vaults)
        self._history = history
        self._cache = cache
        self._registry = registry
        self._scratch_root = scratch_root
        self._memories: dict[int, MemoryLRUCache] = {}
        self._disks: dict[Path, DiskCacheStore] = {}
        self._lock = threading.Lock()
        self._publication_lock = asyncio.Lock()
        self._complete = False
        attachment: dict[str, object] = {"engine": engine}
        if lifetime is not None:
            attachment["lifetime"] = lifetime.token
        manifest = cast(
            "dict[str, object]", library.call("references.engine.attach", **attachment)["manifest"]
        )
        self._revision = cast(int, manifest["revision"])
        self._patch(vaults=[str(root) for root in self._configured_vaults])

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

    def validate_binding(
        self,
        cache: CacheStore,
        history: HistoryStore | None,
        registry: TypeRegistry,
        scratch_root: Path | None,
        vaults: tuple[AssetVault, ...],
    ) -> None:
        if (
            cache is not self._cache
            or history is not self._history
            or registry is not self._registry
            or scratch_root is None
            or scratch_root.absolute() != self._scratch_root.absolute()
            or {item.root.absolute() for item in vaults} != set(self._configured_vaults)
        ):
            raise ValueError("cleanup references must bind the server's owners, vaults and scratch")

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
    """Recover deferred publications and apply the engine's temporary size limit."""
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
                result.update(inspect_engine_temporary(app, collect=True, limit_bytes=limit))
        app[POLICY_KEY].clear()
        app[POLICY_KEY].update(result)


def _scratch_roots(
    scratch: Path, *, owner: tuple[str, int] | None = None
) -> Generator[tuple[Path, Path]]:
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
            selected = produced_job_scratch_root(pack, *owner) if owner is not None else None
            if selected is not None and not selected.exists() and not is_link_or_junction(selected):
                continue
            if is_link_or_junction(pack) or not pack.is_dir():
                raise AssetError("scratch pack accounting is unavailable")
            produced = pack / "produced-assets"
            jobs = produced / "jobs"
            if not present_directory(produced):
                continue
            if selected is None and any(path.name != "jobs" for path in produced.iterdir()):
                raise AssetError("unscoped scratch accounting is unavailable")
            if not present_directory(jobs):
                continue
            if selected is not None:
                if present_directory(selected):
                    yield pack, selected
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


def grant_produced_publications(
    scratch: Path, store: LibraryStore, *, job: str, attempt: int, scope: str
) -> None:
    """Grant a job's offered writer publications without claiming its readers are idle."""
    for _pack, root in _scratch_roots(scratch, owner=(job, attempt)):
        with closing(ProducedScratchLease(root)):
            state = ProducedScratchState(root)
            if state.owner() != (job, attempt):
                raise AssetError("scratch job changed during publication")
            vault = AssetVault(root)
            for digest, size, offered in state.publications():
                if not offered or store.has_asset_grant(scope, digest):
                    continue
                path = vault.resolve(digest)
                if path is None:
                    raise AssetError("offered publication has no scratch bytes")
                with open_verified(path, digest) as handle:
                    if os.fstat(handle.fileno()).st_size != size:
                        raise AssetError("publication size disagrees with its journal")
                    store.grant_asset(scope, digest, path)


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
                            (
                                item
                                for item in (*preview_vaults, vault)
                                if item.resolve(digest) is not None
                            ),
                            preview_vaults[0] if preview_vaults else vault,
                        )
                        for digest, _, offered in publications
                        if offered
                    }
                    for digest, size, offered in publications:
                        if offered:
                            _recover_publication(destinations[digest], root, digest, size)
                            recovered += 1
                    if store is not None:
                        with store.hold_asset_references():
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


async def handle_cleanup_status(request: web.Request) -> web.Response:
    from .library import LIBRARY_KEY

    if not principal_for(request).is_operator:
        raise web.HTTPForbidden(text="asset cleanup accounting requires an operator")
    if request.query:
        raise web.HTTPBadRequest(text="cleanup accounting accepts no filesystem targets")
    library = request.app[LIBRARY_KEY]
    status = inspect_engine_temporary(request.app)
    status["lastPublicationRecovery"] = request.app.get(RECOVERY_KEY)
    status["lastPolicyCollection"] = request.app[POLICY_KEY] or None
    try:
        status["limitBytes"] = await run_owned_thread(library.storage_policy.cleanup_limit)
        status["policyAvailable"] = True
    except (AssetError, sqlite3.Error, LibraryError):
        status["limitBytes"] = None
        status["policyAvailable"] = False
    return web.json_response(status, headers={"Cache-Control": "no-store"})
