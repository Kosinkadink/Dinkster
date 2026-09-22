"""LAN discovery, grant reconciliation, and resolver integration."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from dinkster_assets import (
    P2P_FORMAT_POLICY_VERSION,
    AssetError,
    AssetVault,
    P2PDescriptorV1,
    P2PGrantReconciler,
    P2PGrantSnapshot,
    P2PLocalFileMapping,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionReceiptV1,
    PublicSwarmDeclarationV1,
    ResolverSubscriptionStore,
    SeedGrantV1,
    TransferStatus,
    TransportCandidate,
    TransportResolver,
    open_verified,
    verify_p2p_descriptor,
)
from dinkster_assets.p2p_global import (
    ProviderP2PSnapshotV1,
    provider_transport_candidates_for_snapshots,
)
from dinkster_p2p import (
    DOWNLOAD_LEASE_VERSION,
    AuthorizedGlobalLease,
    DownloadLease,
    P2PManagerError,
    P2PSidecarManager,
    SeedLease,
    authorized_global_leases_for_snapshots,
)
from dinkster_p2p import (
    current_lan_policy as plugin_current_lan_policy,
)
from dinkster_server import (
    ActiveSeedMapping,
    LanInterface,
    LanMappingService,
    LanMdnsDiscovery,
    LanNetworkPolicy,
    RunningLanMappingServer,
    discover_lan_mappings,
    start_lan_mapping_server,
)

_RESOLUTION_TIMEOUT_SECONDS = 120.0
_MONITOR_INTERVAL_SECONDS = 0.5
_LOG = logging.getLogger(__name__)


def current_lan_policy() -> LanNetworkPolicy:
    return LanNetworkPolicy(
        tuple(
            LanInterface(interface.name, interface.address, interface.network)
            for interface in plugin_current_lan_policy().interfaces
        )
    )


class LanP2PBackend:
    """Adapt sidecar leases to the progress contract used by TransportResolver."""

    def __init__(
        self,
        manager: P2PSidecarManager,
        *,
        request_global: Callable[[str, str], Awaitable[str | None]] | None = None,
        release_global: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._manager = manager
        self._failed_leases: dict[tuple[str, str], str] = {}
        self._request_global = request_global
        self._release_global = release_global

    async def transfer(
        self,
        digest: str,
        candidate: TransportCandidate,
        *,
        stall_timeout: float,
    ) -> AsyncGenerator[TransferStatus]:
        del stall_timeout
        if candidate.kind not in {"lan-p2p", "global-p2p"}:
            yield "failed"
            return
        assert candidate.descriptor is not None
        assert candidate.size_bytes is not None
        assert candidate.expires_at is not None
        global_request = candidate.kind == "global-p2p"
        if global_request:
            if self._request_global is None:
                yield "failed"
                return
            try:
                lease_id = await self._request_global(digest, candidate.source)
            except P2PManagerError:
                lease_id = None
            if lease_id is None:
                yield "failed"
                return
        else:
            lease_id = "lan-download-" + uuid.uuid4().hex
            has_peer_hint = candidate.peer_address is not None and candidate.peer_port is not None
            lease = DownloadLease(
                version=DOWNLOAD_LEASE_VERSION if has_peer_hint else 1,
                kind="download",
                lease_id=lease_id,
                digest=digest,
                size_bytes=candidate.size_bytes,
                descriptor=candidate.descriptor,
                staging_path=(f"{candidate.descriptor.info_hash}/{digest.removeprefix('blake3:')}"),
                scope="lan-only",
                expires_at=candidate.expires_at,
                peer_address=candidate.peer_address,
                peer_port=candidate.peer_port,
            )
            try:
                await self._manager.grant_download(lease)
            except P2PManagerError:
                with contextlib.suppress(P2PManagerError):
                    await self._manager.revoke(lease_id)
                yield "failed"
                return
        last_progress = 0
        terminal = "stalled"
        global_release_task: asyncio.Future[None] | None = None

        async def release_requested_global() -> None:
            nonlocal global_release_task
            assert self._release_global is not None
            if global_release_task is None:
                global_release_task = asyncio.ensure_future(
                    self._release_global(digest, candidate.source)
                )
            cancelled = False
            while not global_release_task.done():
                try:
                    await asyncio.shield(global_release_task)
                except asyncio.CancelledError:
                    cancelled = True
                except P2PManagerError:
                    break
            with contextlib.suppress(P2PManagerError):
                global_release_task.result()
            if cancelled:
                raise asyncio.CancelledError

        try:
            while True:
                try:
                    status = await self._manager.lease_status(lease_id)
                except P2PManagerError:
                    terminal = "failed"
                    self._failed_leases[(digest, candidate.source)] = lease_id
                    yield "failed"
                    return
                state = status.get("state")
                durable = status.get("durableBytes", 0)
                verified = status.get("verifiedBytes", 0)
                if state == "complete":
                    terminal = "complete"
                    if global_request and self._release_global is not None:
                        await release_requested_global()
                    yield "complete"
                    return
                if state in {"failed", "expired", "disabled", "inactive"}:
                    _LOG.warning("P2P transfer ended before publication: %s", status)
                    terminal = "failed"
                    self._failed_leases[(digest, candidate.source)] = lease_id
                    yield "failed"
                    return
                progress = max(
                    durable if isinstance(durable, int) else 0,
                    verified if isinstance(verified, int) else 0,
                )
                if progress > last_progress:
                    last_progress = progress
                    yield "progress"
                await asyncio.sleep(0.1)
        finally:
            if terminal != "failed":
                if global_request and self._release_global is not None:
                    await release_requested_global()
                else:
                    with contextlib.suppress(P2PManagerError):
                        await self._manager.revoke(lease_id)

    async def discard_partial(self, digest: str, candidate: TransportCandidate) -> None:
        lease_id = self._failed_leases.pop((digest, candidate.source), None)
        if lease_id is None:
            return
        with contextlib.suppress(P2PManagerError):
            await self._manager.remove_partial(lease_id)
        if candidate.kind == "global-p2p" and self._release_global is not None:
            with contextlib.suppress(P2PManagerError):
                await self._release_global(digest, candidate.source)


class LanP2PController:
    """Own one sidecar and every LAN socket for a server instance."""

    def __init__(
        self,
        *,
        vault: AssetVault,
        resolver_indexes: ResolverSubscriptionStore,
        receipts: PublicAcquisitionReceiptStore,
        local_path_for: Callable[[str], Path | None],
        installation_root: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._vault = vault
        self._resolver_indexes = resolver_indexes
        self._receipts = receipts
        self._local_path_for = local_path_for
        self._clock = clock
        self._manager = P2PSidecarManager(
            vault_root=vault.root,
            installation_root=installation_root,
        )
        self._settings: dict[str, object] = {}
        self._grants = P2PGrantReconciler(clock=clock)
        self._snapshot = P2PGrantSnapshot(False, (), ())
        self._provider_snapshots: tuple[ProviderP2PSnapshotV1, ...] = ()
        self._global_transport_candidates: dict[str, tuple[TransportCandidate, ...]] = {}
        self._requested_global_downloads: dict[str, str] = {}
        self._authority_signature: object = None
        self._next_authority_expiry = 0.0
        self._authorities: dict[str, tuple[tuple[int, P2PDescriptorV1, float], ...]] = {}
        self._seed_leases: dict[str, tuple[SeedLease, P2PLocalFileMapping]] = {}
        self._seed_mappings: dict[str, ActiveSeedMapping] = {}
        self._network_policy: LanNetworkPolicy | None = None
        self._network_paused = False
        self._discovery: LanMdnsDiscovery | None = None
        self._mapping_server: RunningLanMappingServer | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = asyncio.Lock()
        self._instance_id = "dinkster-" + uuid.uuid4().hex[:12]
        self._backend = LanP2PBackend(
            self._manager,
            request_global=self._request_global_download,
            release_global=self._release_global_download,
        )
        self._resolver = TransportResolver(
            vault,
            vault,
            self._known_transports,
            self._discover,
            self._backend,
            self._validate_complete,
            clock=clock,
        )

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._manager.process

    @property
    def settings(self) -> dict[str, object]:
        return dict(self._settings)

    @property
    def network_paused(self) -> bool:
        return self._network_paused

    @property
    def global_network_policy(self) -> tuple[str, bool]:
        return self._manager.global_network_policy

    @property
    def grant_snapshot(self) -> P2PGrantSnapshot:
        return self._snapshot

    def _enabled(self, capability: str) -> bool:
        return bool(self._settings.get(capability, False))

    def _network_allowed(self) -> bool:
        return not self._network_paused

    def _global_scope_enabled(self) -> bool:
        return self._settings.get("scope") == "lan-and-internet"

    def _global_downloads_allowed(self) -> bool:
        cost, paused = self._manager.global_network_policy
        return bool(
            self._global_scope_enabled()
            and self._enabled("downloadsEnabled")
            and self._network_allowed()
            and cost in {"metered", "unmetered"}
            and not paused
        )

    async def start(self, settings: object, *, network_paused: bool = False) -> None:
        self._loop = asyncio.get_running_loop()
        async with self._state_lock:
            self._settings = dict(settings) if isinstance(settings, dict) else {}
            self._network_paused = network_paused
            await self._manager.start(settings, network_paused=network_paused)
            await self._apply_network_state()
        self._monitor = asyncio.create_task(self._monitor_state())

    async def update(self, settings: object, *, network_paused: bool | None = None) -> None:
        async with self._state_lock:
            self._settings = dict(settings) if isinstance(settings, dict) else {}
            if network_paused is not None:
                self._network_paused = network_paused
            await self._manager.update(settings, network_paused=network_paused)
            await self._apply_network_state()

    async def set_network_paused(self, paused: bool) -> None:
        async with self._state_lock:
            self._network_paused = paused
            await self._manager.set_network_paused(paused)
            await self._apply_network_state()

    async def set_global_network_policy(self, cost: str, paused: bool) -> None:
        async with self._state_lock:
            await self._manager.set_global_network_policy(cost, paused)

    async def close(self) -> None:
        self._loop = None
        monitor = self._monitor
        self._monitor = None
        if monitor is not None:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
        async with self._state_lock:
            await self._stop_lan()
            await self._manager.close()

    async def pause(self) -> dict[str, object]:
        async with self._state_lock:
            await self._stop_lan()
            return await self._manager.pause()

    async def resume(self) -> dict[str, object]:
        async with self._state_lock:
            result = await self._manager.resume()
            await self._apply_network_state()
            return result

    async def pause_transfer(self, digest: str) -> dict[str, object]:
        return await self._manager.pause_transfer(digest)

    async def resume_transfer(self, digest: str) -> dict[str, object]:
        return await self._manager.resume_transfer(digest)

    async def resume_seed_transfers(self, digests: Sequence[str]) -> None:
        """Explicitly recover current seeds without clearing unrelated safety latches."""
        async with self._state_lock:
            await self._reconcile_locked()
            if self._network_paused or self._manager.global_network_policy[1]:
                raise P2PManagerError("seed recovery is blocked by network policy")
            current = {
                grant.digest
                for grant in self._snapshot.seed_grants
                if grant.expires_at > self._clock()
                and self._local_path_for(grant.digest) is not None
            }
            for digest in sorted(current.intersection(digests)):
                await self._manager.resume_transfer(digest)

    async def stop_transfer(self, digest: str) -> dict[str, object]:
        return await self._manager.stop_transfer(digest)

    async def remove_transfer_partial(self, digest: str) -> dict[str, object]:
        return await self._manager.remove_transfer_partial(digest)

    async def reset_transfer_budget(self, digest: str) -> dict[str, object]:
        return await self._manager.reset_transfer_budget(digest)

    async def make_transfer_continuous(self, digest: str) -> dict[str, object]:
        return await self._manager.make_transfer_continuous(digest)

    async def status(self) -> dict[str, object]:
        async with self._state_lock:
            status = await self._manager.status()
            status["lan"] = {
                "networkAllowed": self._network_allowed(),
                "mappingPort": self._mapping_server.port if self._mapping_server else None,
                "mappedDigests": sorted(self._seed_mappings),
            }
            return status

    async def _apply_network_state(self) -> None:
        enabled = self._enabled("downloadsEnabled") or self._enabled("seedingEnabled")
        policy = current_lan_policy()
        self._network_policy = policy
        if not enabled or not self._network_allowed():
            await self._stop_lan()
            if enabled and self._manager.process is not None:
                with contextlib.suppress(P2PManagerError):
                    await self._manager.set_network_paused(True)
            return
        if self._manager.process is not None:
            with contextlib.suppress(P2PManagerError):
                await self._manager.set_network_paused(False)
        if policy.interfaces and self._discovery is None:
            self._discovery = LanMdnsDiscovery(self._instance_id, policy)
        if self._enabled("downloadsEnabled") and self._discovery is not None:
            await self._discovery.start()
        try:
            await self._reconcile_locked()
        except P2PManagerError:
            pass

    async def _stop_lan(self) -> None:
        mapping = self._mapping_server
        self._mapping_server = None
        if mapping is not None:
            await mapping.close()
        discovery = self._discovery
        self._discovery = None
        if discovery is not None:
            await discovery.close()
        self._seed_mappings.clear()

    def _known_transports(self, digest: str) -> tuple[TransportCandidate, ...]:
        if not self._global_downloads_allowed():
            return ()
        return self._global_transport_candidates.get(digest, ())

    @staticmethod
    def _index_global_transports(
        provider_snapshots: Sequence[ProviderP2PSnapshotV1],
        grants: P2PGrantSnapshot,
        *,
        now: float,
    ) -> dict[str, tuple[TransportCandidate, ...]]:
        grant_digests = {grant.grant_id: grant.digest for grant in grants.public_grants}
        indexed: dict[str, list[TransportCandidate]] = {}
        candidates = provider_transport_candidates_for_snapshots(
            provider_snapshots,
            grants,
            trusted_provider_ids=frozenset(snapshot.provider_id for snapshot in provider_snapshots),
            now=now,
            global_downloads_allowed=True,
        )
        for candidate in candidates:
            digest = grant_digests.get(candidate.source)
            if digest is not None:
                indexed.setdefault(digest, []).append(candidate)
        return {digest: tuple(candidates) for digest, candidates in indexed.items()}

    async def _request_global_download(self, digest: str, grant_id: str) -> str | None:
        async with self._state_lock:
            if not any(
                candidate.source == grant_id for candidate in self._known_transports(digest)
            ):
                return None
            self._requested_global_downloads[digest] = grant_id
            self._authority_signature = None
            try:
                await self._reconcile_locked()
            except BaseException:
                self._requested_global_downloads.pop(digest, None)
                self._authority_signature = None
                raise
            return f"global:download:{grant_id}"

    async def _release_global_download(self, digest: str, grant_id: str) -> None:
        async with self._state_lock:
            if self._requested_global_downloads.get(digest) != grant_id:
                return
            self._requested_global_downloads.pop(digest, None)
            self._authority_signature = None
            await self._reconcile_locked()

    async def _monitor_state(self) -> None:
        last_error: str | None = None
        while True:
            try:
                async with self._state_lock:
                    await self._refresh_network_policy()
                    await self._reconcile_locked()
                    await self._refresh_seed_mappings()
            except (AssetError, OSError, P2PManagerError) as error:
                reason = str(error)
                if reason != last_error:
                    _LOG.warning("P2P monitor reconciliation failed: %s", reason)
                last_error = reason
            else:
                last_error = None
            await asyncio.sleep(_MONITOR_INTERVAL_SECONDS)

    async def _refresh_network_policy(self) -> None:
        policy = current_lan_policy()
        if policy == self._network_policy:
            return
        self._network_policy = policy
        enabled = self._enabled("downloadsEnabled") or self._enabled("seedingEnabled")
        if not enabled or not self._network_allowed():
            return
        await self._stop_lan()
        await self._manager.close()
        await self._manager.start(
            self._settings,
            network_paused=self._network_paused,
        )
        await self._apply_network_state()

    def _authority_input(
        self,
    ) -> tuple[
        tuple[PublicSwarmDeclarationV1, ...],
        tuple[PublicAcquisitionReceiptV1, ...],
        tuple[ProviderP2PSnapshotV1, ...],
        bool,
    ]:
        declarations = self._resolver_indexes.public_swarm_declarations()
        receipts = self._receipts.records()
        provider_snapshots = self._resolver_indexes.provider_p2p_snapshots()
        enabled = self._enabled("downloadsEnabled") or self._enabled("seedingEnabled")
        return declarations, receipts, provider_snapshots, enabled

    async def reconcile(self, *, local_files_changed: bool = False) -> None:
        async with self._state_lock:
            if local_files_changed:
                self._authority_signature = None
            await self._reconcile_locked()

    async def _reconcile_locked(self) -> None:
        declarations, receipts, provider_snapshots, enabled = await asyncio.to_thread(
            self._authority_input
        )
        now = self._clock()
        signature = (
            declarations,
            receipts,
            provider_snapshots,
            enabled,
            self._enabled("downloadsEnabled"),
            self._enabled("seedingEnabled"),
            self._global_scope_enabled(),
            self._network_allowed(),
            self._network_policy,
        )
        if signature == self._authority_signature and now < self._next_authority_expiry:
            return
        result = await asyncio.to_thread(
            self._grants.reconcile,
            declarations,
            receipts,
            self._local_path_for,
            enabled=enabled,
        )
        self._snapshot = result.snapshot
        self._provider_snapshots = provider_snapshots
        self._global_transport_candidates = await asyncio.to_thread(
            self._index_global_transports,
            provider_snapshots,
            self._snapshot,
            now=now,
        )
        expiries = [
            declaration.expires_at for declaration in declarations if declaration.expires_at > now
        ]
        self._next_authority_expiry = min(expiries, default=float("inf"))
        by_grant = {
            (
                declaration.digest,
                declaration.source_type,
                declaration.source_id,
                declaration.source_revision,
                declaration.descriptor,
            ): declaration
            for declaration in declarations
        }
        authorities: dict[str, list[tuple[int, P2PDescriptorV1, float]]] = {}
        for grant in self._snapshot.public_grants:
            declaration = by_grant.get(
                (
                    grant.digest,
                    grant.source_type,
                    grant.source_id,
                    grant.source_revision,
                    grant.descriptor,
                )
            )
            if declaration is not None:
                authorities.setdefault(grant.digest, []).append(
                    (declaration.size_bytes, grant.descriptor, grant.expires_at)
                )
        self._authorities = {digest: tuple(rows) for digest, rows in authorities.items()}
        global_leases = await self._desired_global_leases(provider_snapshots, now=now)
        await self._reconcile_seed_leases(declarations, global_leases=global_leases)
        self._authority_signature = signature

    async def _desired_global_leases(
        self,
        provider_snapshots: Sequence[ProviderP2PSnapshotV1],
        *,
        now: float,
    ) -> tuple[AuthorizedGlobalLease, ...]:
        return (
            await asyncio.to_thread(
                self._select_global_leases,
                provider_snapshots,
                self._snapshot,
                requested_global_downloads=dict(self._requested_global_downloads),
                downloads_enabled=self._enabled("downloadsEnabled"),
                seeding_enabled=self._enabled("seedingEnabled"),
                now=now,
                local_path_for=self._local_path_for,
            )
            if self._global_scope_enabled()
            else ()
        )

    @staticmethod
    def _select_global_leases(
        provider_snapshots: Sequence[ProviderP2PSnapshotV1],
        grants: P2PGrantSnapshot,
        *,
        requested_global_downloads: Mapping[str, str],
        downloads_enabled: bool,
        seeding_enabled: bool,
        now: float,
        local_path_for: Callable[[str], Path | None],
    ) -> tuple[AuthorizedGlobalLease, ...]:
        desired: dict[str, AuthorizedGlobalLease] = {}
        requested_digests = frozenset(requested_global_downloads)
        authorizations = authorized_global_leases_for_snapshots(
            provider_snapshots,
            grants,
            trusted_provider_ids=frozenset(snapshot.provider_id for snapshot in provider_snapshots),
            requested_download_digests=requested_digests,
            now=now,
            local_path_for=local_path_for,
        )
        for authorization in authorizations:
            lease = authorization.lease
            if isinstance(lease, DownloadLease):
                if not downloads_enabled or requested_global_downloads.get(
                    lease.digest
                ) != lease.lease_id.removeprefix("global:download:"):
                    continue
            elif not seeding_enabled:
                continue
            desired.setdefault(lease.digest, authorization)
        return tuple(desired.values())

    async def _reconcile_seed_leases(
        self,
        declarations: Sequence[PublicSwarmDeclarationV1],
        *,
        global_leases: Sequence[AuthorizedGlobalLease] = (),
    ) -> None:
        desired: dict[str, tuple[SeedLease, P2PLocalFileMapping]] = {}
        declaration_sizes = {
            (
                row.digest,
                row.source_type,
                row.source_id,
                row.source_revision,
                row.descriptor,
            ): row.size_bytes
            for row in declarations
        }
        by_digest: dict[str, list[SeedGrantV1]] = {}
        for grant in self._snapshot.seed_grants:
            by_digest.setdefault(grant.digest, []).append(grant)
        if (
            self._enabled("seedingEnabled")
            and self._network_allowed()
            and self._discovery is not None
        ):
            for digest, grants in by_digest.items():
                descriptors = {grant.descriptor for grant in grants}
                sizes = {
                    declaration_sizes.get(
                        (
                            grant.digest,
                            grant.source_type,
                            grant.source_id,
                            grant.source_revision,
                            grant.descriptor,
                        )
                    )
                    for grant in grants
                }
                if len(descriptors) != 1 or len(sizes) != 1 or None in sizes:
                    continue
                path = self._local_path_for(digest)
                if path is None:
                    continue
                descriptor = descriptors.pop()
                size = sizes.pop()
                assert isinstance(size, int)
                try:
                    mapping = await asyncio.to_thread(
                        self._vault.verify_p2p_local_file,
                        digest,
                        size,
                        path,
                        P2P_FORMAT_POLICY_VERSION,
                    )
                except (AssetError, OSError) as error:
                    _LOG.warning("P2P seed mapping rejected for %s: %s", digest, error)
                    continue
                lease = SeedLease(
                    version=1,
                    kind="seed",
                    lease_id="lan-seed-" + digest.removeprefix("blake3:"),
                    digest=digest,
                    size_bytes=size,
                    descriptor=descriptor,
                    grant_ids=tuple(sorted(grant.grant_id for grant in grants)),
                    local_path=path,
                    scope="lan-only",
                    expires_at=min(grant.expires_at for grant in grants),
                )
                desired[digest] = (lease, mapping)
        # Retire both scopes before any admission can fail at the shared capacity limit.
        for digest, (lease, _mapping) in tuple(self._seed_leases.items()):
            if digest in desired and desired[digest][0] == lease:
                continue
            with contextlib.suppress(P2PManagerError):
                await self._manager.revoke(lease.lease_id)
            self._seed_leases.pop(digest)
            self._seed_mappings.pop(digest, None)
        await self._manager.reconcile_global(global_leases, revoke_only=True)
        for digest, (lease, mapping) in desired.items():
            if digest in self._seed_leases:
                continue
            await self._manager.grant_seed(lease)
            self._seed_leases[digest] = (lease, mapping)
        await self._manager.reconcile_global(global_leases)

    async def _refresh_seed_mappings(self) -> None:
        active: dict[str, ActiveSeedMapping] = {}
        invalid: list[str] = []
        manager_status = await self._manager.status()
        sidecar = manager_status.get("sidecar")
        raw_endpoints = sidecar.get("listenEndpoints") if isinstance(sidecar, dict) else None
        peer_endpoints: list[tuple[str, int]] = []
        policy_addresses = (
            self._network_policy.addresses if self._network_policy is not None else ()
        )
        if isinstance(raw_endpoints, list):
            for raw_endpoint in raw_endpoints:
                if not isinstance(raw_endpoint, dict):
                    continue
                address = raw_endpoint.get("address")
                port = raw_endpoint.get("port")
                if (
                    isinstance(address, str)
                    and address in policy_addresses
                    and type(port) is int
                    and 1 <= port <= 65535
                ):
                    peer_endpoints.append((address, port))
        for digest, (lease, local_mapping) in self._seed_leases.items():
            if not local_mapping.is_current():
                with contextlib.suppress(P2PManagerError):
                    await self._manager.revoke(lease.lease_id)
                invalid.append(digest)
                continue
            try:
                status = await self._manager.lease_status(lease.lease_id)
            except P2PManagerError:
                invalid.append(digest)
                continue
            if status.get("state") == "inactive":
                invalid.append(digest)
                continue
            if (
                status.get("state") == "ready"
                and lease.expires_at > self._clock()
                and peer_endpoints
            ):
                active[digest] = ActiveSeedMapping(
                    digest,
                    lease.size_bytes,
                    lease.descriptor,
                    lease.expires_at,
                    tuple(peer_endpoints),
                )
        for digest in invalid:
            self._seed_leases.pop(digest, None)
        if invalid:
            self._authority_signature = None
        self._seed_mappings = active
        if active and self._mapping_server is None and self._discovery is not None:
            service = LanMappingService(
                self._mapping_for,
                self._discovery.network_policy,
                network_allowed=self._network_allowed,
                clock=self._clock,
            )
            self._mapping_server = await start_lan_mapping_server(service, self._discovery)
        elif not active and self._mapping_server is not None:
            mapping = self._mapping_server
            self._mapping_server = None
            await mapping.close()

    def _mapping_for(self, digest: str) -> ActiveSeedMapping | None:
        process = self._manager.process
        if process is None or process.returncode is not None:
            return None
        mapping = self._seed_mappings.get(digest)
        if mapping is None:
            return None
        if not any(
            size == mapping.size_bytes
            and descriptor == mapping.descriptor
            and expires_at > self._clock()
            for size, descriptor, expires_at in self._authorities.get(digest, ())
        ):
            return None
        return mapping

    async def _discover(self, digest: str) -> tuple[TransportCandidate, ...]:
        discovery = self._discovery
        if (
            discovery is None
            or not self._enabled("downloadsEnabled")
            or not self._network_allowed()
        ):
            return ()
        try:
            mappings = await discover_lan_mappings(
                discovery,
                digest,
                network_allowed=self._network_allowed,
            )
        except RuntimeError:
            return ()
        candidates: list[TransportCandidate] = []
        for mapping in mappings:
            matching_expiries = [
                expiry
                for size, descriptor, expiry in self._authorities.get(digest, ())
                if size == mapping.size_bytes and descriptor == mapping.descriptor
            ]
            if not matching_expiries:
                _LOG.debug("LAN mapping rejected for %s: no matching authority", digest)
                continue
            candidates.append(
                TransportCandidate(
                    "lan-p2p",
                    mapping.endpoint.origin,
                    size_bytes=mapping.size_bytes,
                    descriptor=mapping.descriptor,
                    expires_at=min(matching_expiries),
                    peer_address=(
                        str(mapping.endpoint.address) if mapping.peer_port is not None else None
                    ),
                    peer_port=mapping.peer_port,
                )
            )
        _LOG.debug(
            "LAN discovery for %s: %d mappings, %d candidates",
            digest,
            len(mappings),
            len(candidates),
        )
        return tuple(candidates)

    def _validate_complete(
        self,
        digest: str,
        candidate: TransportCandidate | None,
        path: Path,
    ) -> bool:
        if candidate is None:
            with open_verified(path, digest):
                return True
        if candidate.kind == "http":
            with open_verified(path, digest):
                return True
        assert candidate.descriptor is not None and candidate.size_bytes is not None
        verify_p2p_descriptor(
            path,
            candidate.descriptor,
            asset_digest=digest,
            size=candidate.size_bytes,
        )
        self._vault.verify_p2p_local_file(
            digest,
            candidate.size_bytes,
            path,
            P2P_FORMAT_POLICY_VERSION,
        )
        return True

    def resolve_sync(self, digest: str) -> Path | None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return None
        future = asyncio.run_coroutine_threadsafe(self._resolver.resolve(digest), loop)
        try:
            return future.result(timeout=_RESOLUTION_TIMEOUT_SECONDS)
        except (concurrent.futures.TimeoutError, AssetError, OSError, P2PManagerError):
            future.cancel()
            return None

    def notify_acquired(self) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        def reconcile() -> None:
            self._authority_signature = None
            asyncio.create_task(self._reconcile_after_acquisition())

        loop.call_soon_threadsafe(reconcile)

    async def _reconcile_after_acquisition(self) -> None:
        try:
            await self.reconcile()
        except (AssetError, OSError, P2PManagerError):
            pass
