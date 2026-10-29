"""Host integration for P2P lifecycle and status."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from typing import Protocol, TypeGuard, cast

from aiohttp import web
from dinkster_assets import AssetError, P2PGrantSnapshot
from dinkster_p2p import (
    P2PManagerConflict,
    P2PManagerError,
    P2PManagerNotFound,
)
from dinkster_server import (
    NetworkCost,
    P2PAction,
    P2PActivityStatus,
    P2PNetworkStatus,
    P2PProviderConflict,
    P2PProviderUnavailable,
    P2PSeedAuthorization,
    P2PTotals,
    P2PTransfer,
    P2PTransferNotFound,
    RuntimeSettings,
    detect_network_cost,
)
from dinkster_server.p2p_activity import (
    NetworkOverride,
    TransferState,
    add_p2p_action_routes,
)


class P2PService(Protocol):
    @property
    def settings(self) -> dict[str, object]: ...

    @property
    def network_paused(self) -> bool: ...

    @property
    def global_network_policy(self) -> tuple[str, bool]: ...

    async def start(self, settings: object, *, network_paused: bool = False) -> None: ...

    async def update(self, settings: object, *, network_paused: bool | None = None) -> None: ...

    async def close(self) -> None: ...

    async def status(self) -> dict[str, object]: ...

    async def set_network_paused(self, paused: bool) -> None: ...

    async def set_global_network_policy(self, cost: str, paused: bool) -> None: ...

    async def pause_transfer(self, digest: str) -> dict[str, object]: ...

    async def resume_transfer(self, digest: str) -> dict[str, object]: ...

    async def stop_transfer(self, digest: str) -> dict[str, object]: ...

    async def remove_transfer_partial(self, digest: str) -> dict[str, object]: ...

    async def reset_transfer_budget(self, digest: str) -> dict[str, object]: ...

    async def make_transfer_continuous(self, digest: str) -> dict[str, object]: ...


P2P_KEY: web.AppKey[P2PService | None] = web.AppKey("dinkster_p2p")

_TRANSFER_FIELDS = {
    "digest",
    "state",
    "sizeBytes",
    "peers",
    "downloadRateBytesPerSecond",
    "uploadRateBytesPerSecond",
    "downloadedBytes",
    "uploadedBytes",
    "partialBytes",
    "seedGrantIds",
    "authorizedSeedGrantIds",
    "remainingSeedRatio",
    "remainingSeedTimeSeconds",
}


def _object(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    body = {str(key): item for key, item in value.items()}
    if set(body) != fields:
        raise ValueError(f"{label} has an unsupported shape")
    return body


def _integer(body: Mapping[str, object], key: str) -> int:
    value = body[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_number(body: Mapping[str, object], key: str) -> float | None:
    value = body[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


def _optional_integer(body: Mapping[str, object], key: str) -> int | None:
    value = body[key]
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer or null")
    return value


def _grant_ids(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not all(
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdef" for character in item)
            for item in value
        )
        or len(set(cast("list[str]", value))) != len(value)
    ):
        raise ValueError(f"{label} must contain unique canonical grant ids")
    return tuple(cast("list[str]", value))


def _is_digest(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value.startswith("blake3:"):
        return False
    encoded = value.removeprefix("blake3:")
    return len(encoded) == 64 and all(character in "0123456789abcdef" for character in encoded)


def _lan_status(value: object, network: P2PNetworkStatus) -> dict[str, object]:
    if value is None:
        return {
            "networkAllowed": not network.paused,
            "mappingPort": None,
            "mappedDigests": [],
        }
    body = _object(value, {"networkAllowed", "mappingPort", "mappedDigests"}, "LAN status")
    mapping_port = body["mappingPort"]
    digests = body["mappedDigests"]
    if (
        not isinstance(body["networkAllowed"], bool)
        or body["networkAllowed"] != (not network.paused)
        or (
            mapping_port is not None
            and (type(mapping_port) is not int or not 1 <= mapping_port <= 65535)
        )
        or not isinstance(digests, list)
        or not all(_is_digest(digest) for digest in digests)
        or len(set(cast("list[str]", digests))) != len(digests)
    ):
        raise ValueError("LAN status is malformed")
    return dict(body)


def _transfer(value: object, snapshot: P2PGrantSnapshot) -> P2PTransfer:
    body = _object(value, _TRANSFER_FIELDS, "transfer")
    digest, state = body["digest"], body["state"]
    if not isinstance(digest, str) or not isinstance(state, str):
        raise ValueError("transfer digest and state must be strings")
    known_ids = _grant_ids(body["seedGrantIds"], "seedGrantIds")
    active_ids = frozenset(_grant_ids(body["authorizedSeedGrantIds"], "authorizedSeedGrantIds"))
    if not active_ids <= set(known_ids):
        raise ValueError("authorizedSeedGrantIds must be known to the transfer")
    current = {grant.grant_id: grant for grant in snapshot.seed_grants if grant.digest == digest}
    if len(current) != len([grant for grant in snapshot.seed_grants if grant.digest == digest]):
        raise ValueError("canonical seed grant ids must be unique")
    authorization_ids = sorted(set(known_ids) | set(current))
    return P2PTransfer(
        digest=digest,
        state=cast(TransferState, state),
        size_bytes=_integer(body, "sizeBytes"),
        peers=_integer(body, "peers"),
        download_rate_bytes_per_second=_integer(body, "downloadRateBytesPerSecond"),
        upload_rate_bytes_per_second=_integer(body, "uploadRateBytesPerSecond"),
        downloaded_bytes=_integer(body, "downloadedBytes"),
        uploaded_bytes=_integer(body, "uploadedBytes"),
        partial_bytes=_integer(body, "partialBytes"),
        seed_authorizations=tuple(
            P2PSeedAuthorization(
                grant_id=grant_id,
                state=(
                    "active"
                    if grant_id in active_ids and grant_id in current
                    else "inactive"
                    if grant_id in current
                    else "revoked"
                ),
                grant=current.get(grant_id),
            )
            for grant_id in authorization_ids
        ),
        remaining_seed_ratio=_optional_number(body, "remainingSeedRatio"),
        remaining_seed_time_seconds=_optional_integer(body, "remainingSeedTimeSeconds"),
    )


_DORMANT_GRANTS = P2PGrantSnapshot(enabled=False, public_grants=(), seed_grants=())


def _dormant_grants() -> P2PGrantSnapshot:
    return _DORMANT_GRANTS


class P2PSidecarActivityProvider:
    def __init__(
        self,
        manager: P2PService,
        network_cost: Callable[[], NetworkCost],
        grant_snapshot: Callable[[], P2PGrantSnapshot] = _dormant_grants,
    ) -> None:
        self._manager = manager
        self._network_cost = network_cost
        self._grant_snapshot = grant_snapshot
        self._policy_lock = asyncio.Lock()
        self._policy_generation = 0

    @staticmethod
    def _enabled(settings: Mapping[str, object]) -> bool:
        return bool(settings["downloadsEnabled"] or settings["seedingEnabled"])

    @staticmethod
    def _network_status(
        settings: Mapping[str, object], system: NetworkCost, *, paused: bool
    ) -> P2PNetworkStatus:
        override = settings["networkCostOverride"]
        if not isinstance(override, str):
            raise ValueError("network cost override must be a string")
        effective = system if override == "auto" else cast(NetworkCost, override)
        return P2PNetworkStatus(
            system=system,
            override=cast(NetworkOverride, override),
            effective=effective,
            paused=paused,
        )

    @classmethod
    def _global_network_policy(
        cls, settings: Mapping[str, object], network: P2PNetworkStatus
    ) -> tuple[NetworkCost, bool]:
        global_enabled = cls._enabled(settings) and settings["scope"] == "lan-and-internet"
        paused = bool(
            global_enabled
            and (
                network.effective == "unknown"
                or (settings["pauseOnMetered"] and network.effective == "metered")
            )
        )
        return network.effective, paused

    async def _apply_global_network_policy(
        self, settings: Mapping[str, object], network: P2PNetworkStatus
    ) -> None:
        policy = self._global_network_policy(settings, network)
        if self._manager.global_network_policy != policy:
            await self._manager.set_global_network_policy(*policy)

    async def network_status(
        self, settings: Mapping[str, object] | None = None
    ) -> P2PNetworkStatus:
        policy = settings or self._manager.settings
        system = await asyncio.to_thread(self._network_cost)
        return self._network_status(policy, system, paused=self._manager.network_paused)

    async def start(self, settings: Mapping[str, object]) -> None:
        async with self._policy_lock:
            self._policy_generation += 1
            if not self._enabled(settings):
                await self._manager.start(settings)
                await self._manager.set_global_network_policy("unknown", False)
                return
            network = await self.network_status(settings)
            await self._manager.set_global_network_policy(
                *self._global_network_policy(settings, network)
            )
            await self._manager.start(settings)

    async def update(self, settings: Mapping[str, object]) -> None:
        async with self._policy_lock:
            self._policy_generation += 1
            generation = self._policy_generation
            enabled = self._enabled(settings)
            if not enabled:
                await self._manager.update(settings, network_paused=False)
                await self._manager.set_global_network_policy("unknown", False)
                return
        network = await self.network_status(settings)
        async with self._policy_lock:
            if generation != self._policy_generation:
                return
            policy = self._global_network_policy(settings, network)
            if policy[1]:
                await self._apply_global_network_policy(settings, network)
            await self._manager.update(settings)
            if not policy[1]:
                await self._apply_global_network_policy(settings, network)

    async def reconcile_network_policy(self) -> None:
        async with self._policy_lock:
            generation = self._policy_generation
            settings = self._manager.settings
            if not self._enabled(settings):
                return
        network = await self.network_status(settings)
        async with self._policy_lock:
            if generation != self._policy_generation:
                return
            await self._apply_global_network_policy(settings, network)

    async def status(self) -> dict[str, object]:
        try:
            while True:
                async with self._policy_lock:
                    generation = self._policy_generation
                    settings = self._manager.settings
                network = (
                    await self.network_status(settings)
                    if self._enabled(settings)
                    else self._network_status(
                        settings,
                        "unknown",
                        paused=self._manager.network_paused,
                    )
                )
                async with self._policy_lock:
                    if generation != self._policy_generation:
                        continue
                    await self._apply_global_network_policy(settings, network)
                    status = await self._manager.status()
                    snapshot = self._grant_snapshot()
                    if not isinstance(snapshot, P2PGrantSnapshot):
                        raise TypeError("grant snapshot provider returned an invalid value")
                    break
            lan = _lan_status(status.get("lan"), network)
            sidecar = status.get("sidecar")
            if sidecar is None:
                return {**status, "network": network.to_wire(), "lan": lan}
            if not isinstance(sidecar, Mapping):
                raise ValueError("sidecar status must be an object")
            if sidecar.get("networkPaused") != network.paused:
                raise ValueError("sidecar network pause state does not match the host")
            totals = _object(sidecar.get("totals"), {"downloadedBytes", "uploadedBytes"}, "totals")
            transfers = sidecar.get("transfers")
            if not isinstance(transfers, list):
                raise ValueError("transfers must be a list")
            activity = P2PActivityStatus(
                network=network,
                totals=P2PTotals(
                    downloaded_bytes=_integer(totals, "downloadedBytes"),
                    uploaded_bytes=_integer(totals, "uploadedBytes"),
                ),
                transfers=tuple(_transfer(value, snapshot) for value in transfers),
            )
            activity_wire = activity.to_wire()
            return {
                **status,
                "network": activity_wire["network"],
                "lan": lan,
                "sidecar": {
                    **sidecar,
                    "totals": activity_wire["totals"],
                    "transfers": activity_wire["transfers"],
                },
            }
        except (AssetError, P2PManagerError, TypeError, ValueError) as error:
            raise P2PProviderUnavailable("sidecar activity contract is unavailable") from error

    async def perform_action(self, digest: str, action: P2PAction) -> None:
        operations = {
            "pause": self._manager.pause_transfer,
            "resume": self._manager.resume_transfer,
            "stop": self._manager.stop_transfer,
            "remove-partial": self._manager.remove_transfer_partial,
            "reset-budget": self._manager.reset_transfer_budget,
            "continuous-seed": self._manager.make_transfer_continuous,
        }
        try:
            await operations[action](digest)
        except P2PManagerConflict as error:
            raise P2PProviderConflict("transfer action conflicts with sidecar state") from error
        except P2PManagerNotFound as error:
            raise P2PTransferNotFound("transfer does not exist") from error
        except P2PManagerError as error:
            raise P2PProviderUnavailable("sidecar action is unavailable") from error


def add_p2p_routes(
    app: web.Application,
    manager: P2PService | None,
    settings: RuntimeSettings,
    *,
    network_cost: Callable[[], NetworkCost] = detect_network_cost,
    grant_snapshot: Callable[[], P2PGrantSnapshot] = _dormant_grants,
    network_poll_interval: float = 5.0,
) -> None:
    if network_poll_interval <= 0:
        raise ValueError("network_poll_interval must be positive")
    app[P2P_KEY] = manager
    provider = (
        None
        if manager is None
        else P2PSidecarActivityProvider(manager, network_cost, grant_snapshot)
    )
    latest_update: asyncio.Task[None] | None = None

    async def get_status(_request: web.Request) -> web.Response:
        nonlocal latest_update
        if provider is None:
            return web.json_response(
                {
                    "state": "unavailable",
                    "message": "P2P requires a persistent library vault",
                    "sidecar": None,
                }
            )
        try:
            pending_update = latest_update
            if pending_update is not None:
                # Only the newest write gates subsequent reads. A disconnected
                # reader must not cancel the host's settings update.
                try:
                    await asyncio.shield(pending_update)
                except P2PManagerError as error:
                    raise P2PProviderUnavailable("P2P settings application failed") from error
                finally:
                    if pending_update.done() and latest_update is pending_update:
                        latest_update = None
            return web.json_response(await provider.status())
        except P2PProviderUnavailable:
            return web.json_response(
                {"error": {"code": "p2p-unavailable", "message": "P2P activity is unavailable"}},
                status=503,
            )
        except Exception:
            return web.json_response(
                {"error": {"code": "p2p-internal-error", "message": "P2P activity failed"}},
                status=500,
            )

    app.router.add_get("/api/p2p/status", get_status)
    add_p2p_action_routes(app, provider, p2p_granted="p2p" in settings.granted)
    if manager is None:
        return
    assert provider is not None

    updates: set[asyncio.Task[None]] = set()
    monitor: asyncio.Task[None] | None = None

    def update(value: dict[str, object]) -> None:
        nonlocal latest_update
        task = asyncio.create_task(provider.update(value))
        latest_update = task
        updates.add(task)
        task.add_done_callback(updates.discard)

    async def start(_: web.Application) -> None:
        nonlocal monitor
        await provider.start(settings.p2p)

        async def monitor_network() -> None:
            while True:
                await asyncio.sleep(network_poll_interval)
                with contextlib.suppress(P2PManagerError):
                    await provider.reconcile_network_policy()

        monitor = asyncio.create_task(monitor_network())

    async def close(_: web.Application) -> None:
        if monitor is not None:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
        if updates:
            await asyncio.gather(*updates)
        await manager.close()

    app.on_startup.append(start)
    app.on_cleanup.append(close)
    settings.bind_p2p(update)
