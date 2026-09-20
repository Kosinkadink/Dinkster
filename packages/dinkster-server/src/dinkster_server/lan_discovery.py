"""LAN-only interface policy and mDNS peer discovery."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from ipaddress import IPv4Address, ip_address
from types import TracebackType
from typing import cast

from dinkster_p2p import LanInterface, LanNetworkPolicy, lan_interfaces
from zeroconf import IPVersion, ServiceInfo, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

__all__ = [
    "LanInterface",
    "LanMdnsDiscovery",
    "LanNetworkPolicy",
    "LanPeerEndpoint",
    "lan_interfaces",
]

LAN_P2P_SERVICE_TYPE = "_dinkster-p2p._tcp.local."
LAN_P2P_SERVICE_VERSION = b"1"
DEFAULT_LAN_DISCOVERY_TIMEOUT = 3.0
DEFAULT_MDNS_RESOLVE_TIMEOUT_MS = 1000
MAX_LAN_PEERS = 32

_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")


@dataclass(frozen=True)
class LanPeerEndpoint:
    service_name: str
    address: IPv4Address
    port: int

    @property
    def origin(self) -> str:
        return f"http://{self.address}:{self.port}"


class LanMdnsDiscovery:
    """Advertise one mapping endpoint and maintain a bounded LAN peer cache."""

    def __init__(
        self,
        instance_id: str,
        policy: LanNetworkPolicy,
        *,
        resolve_timeout_ms: int = DEFAULT_MDNS_RESOLVE_TIMEOUT_MS,
        max_peers: int = MAX_LAN_PEERS,
    ) -> None:
        if _INSTANCE_ID.fullmatch(instance_id) is None:
            raise ValueError("LAN P2P instance id must be 1-63 alphanumeric/hyphen characters")
        if not policy.interfaces:
            raise ValueError("LAN P2P requires at least one eligible interface")
        if resolve_timeout_ms <= 0:
            raise ValueError("mDNS resolve timeout must be positive")
        if max_peers <= 0:
            raise ValueError("maximum LAN peers must be positive")
        self._instance_id = instance_id
        self._service_name = f"{instance_id}.{LAN_P2P_SERVICE_TYPE}"
        self._policy = policy
        self._resolve_timeout_ms = resolve_timeout_ms
        self._max_peers = max_peers
        self._zeroconf: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._advertisement: ServiceInfo | None = None
        self._peers: dict[str, tuple[LanPeerEndpoint, ...]] = {}
        self._refresh_tasks: dict[str, asyncio.Task[None]] = {}
        self._changed = asyncio.Event()
        self._closed = False

    @property
    def network_policy(self) -> LanNetworkPolicy:
        return self._policy

    async def start(self) -> None:
        if self._zeroconf is not None:
            return
        if self._closed:
            raise RuntimeError("LAN discovery is closed")
        zeroconf = AsyncZeroconf(
            interfaces=list(self._policy.addresses),
            ip_version=IPVersion.V4Only,
        )
        self._zeroconf = zeroconf
        try:
            self._browser = AsyncServiceBrowser(
                zeroconf.zeroconf,
                LAN_P2P_SERVICE_TYPE,
                handlers=[self._service_changed],
            )
        except BaseException:
            self._zeroconf = None
            await zeroconf.async_close()
            raise

    async def advertise(self, port: int) -> None:
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("LAN mapping port must be between 1 and 65535")
        await self.start()
        await self.withdraw()
        info = ServiceInfo(
            LAN_P2P_SERVICE_TYPE,
            self._service_name,
            port=port,
            parsed_addresses=list(self._policy.addresses),
            properties={b"version": LAN_P2P_SERVICE_VERSION},
            server=f"{self._instance_id}.local.",
        )
        assert self._zeroconf is not None
        announcement = cast(
            "Awaitable[object]",
            await self._zeroconf.async_register_service(info),  # pyright: ignore[reportUnknownMemberType]
        )
        self._advertisement = info
        try:
            await announcement
        except BaseException:
            with contextlib.suppress(Exception):
                await self.withdraw()
            raise

    async def withdraw(self) -> None:
        info = self._advertisement
        if info is None:
            return
        assert self._zeroconf is not None
        zeroconf = self._zeroconf.zeroconf
        goodbye = cast(
            "Awaitable[object]",
            await self._zeroconf.async_unregister_service(info),  # pyright: ignore[reportUnknownMemberType]
        )
        await goodbye
        # Goodbye completion only sends packets; zeroconf can retain its received PTR locally.
        pointer = zeroconf.cache.current_entry_with_name_and_alias(info.type, info.name)
        if pointer is not None:
            zeroconf.cache.async_remove_records((pointer,))
        if self._advertisement is info:
            self._advertisement = None

    def _service_changed(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        del zeroconf
        if self._closed or service_type != LAN_P2P_SERVICE_TYPE or name == self._service_name:
            return
        if state_change is ServiceStateChange.Removed:
            task = self._refresh_tasks.pop(name, None)
            if task is not None:
                task.cancel()
            if self._peers.pop(name, None) is not None:
                self._changed.set()
            return
        if name in self._refresh_tasks or len(self._refresh_tasks) >= self._max_peers:
            return
        task = asyncio.create_task(self._refresh_service(name))
        self._refresh_tasks[name] = task
        task.add_done_callback(lambda finished: self._refresh_done(name, finished))

    def _refresh_done(self, name: str, task: asyncio.Task[None]) -> None:
        if self._refresh_tasks.get(name) is task:
            self._refresh_tasks.pop(name)
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.result()

    async def _refresh_service(self, name: str) -> None:
        zeroconf = self._zeroconf
        if zeroconf is None or self._closed:
            return
        info = await zeroconf.async_get_service_info(
            LAN_P2P_SERVICE_TYPE,
            name,
            timeout=self._resolve_timeout_ms,
        )
        port = None if info is None else info.port
        if info is None or port is None or not 1 <= port <= 65535:
            self._peers.pop(name, None)
            self._changed.set()
            return
        if info.properties.get(b"version") != LAN_P2P_SERVICE_VERSION:
            self._peers.pop(name, None)
            self._changed.set()
            return
        endpoints: list[LanPeerEndpoint] = []
        seen: set[IPv4Address] = set()
        for raw_address in info.parsed_addresses(IPVersion.V4Only):
            if not self._policy.allows_peer(raw_address):
                continue
            address = cast("IPv4Address", ip_address(raw_address))
            if address in seen:
                continue
            seen.add(address)
            endpoints.append(LanPeerEndpoint(name, address, port))
        if endpoints:
            self._peers[name] = tuple(endpoints)
        else:
            self._peers.pop(name, None)
        self._trim_peers()
        self._changed.set()

    def _trim_peers(self) -> None:
        endpoint_count = sum(len(endpoints) for endpoints in self._peers.values())
        if endpoint_count <= self._max_peers:
            return
        retained: dict[str, tuple[LanPeerEndpoint, ...]] = {}
        remaining = self._max_peers
        for name in sorted(self._peers):
            if remaining == 0:
                break
            endpoints = self._peers[name][:remaining]
            if endpoints:
                retained[name] = endpoints
                remaining -= len(endpoints)
        self._peers = retained

    async def peers(
        self,
        timeout: float = DEFAULT_LAN_DISCOVERY_TIMEOUT,
        *,
        exclude: frozenset[LanPeerEndpoint] = frozenset(),
    ) -> tuple[LanPeerEndpoint, ...]:
        if timeout <= 0:
            raise ValueError("LAN discovery timeout must be positive")
        await self.start()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._changed.clear()
            peers = tuple(
                endpoint
                for name in sorted(self._peers)
                for endpoint in self._peers[name]
                if endpoint not in exclude
            )[: self._max_peers]
            if peers:
                return peers
            remaining = deadline - loop.time()
            if remaining <= 0:
                return ()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except TimeoutError:
                return ()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        zeroconf = self._zeroconf
        if zeroconf is None:
            return
        with contextlib.suppress(Exception):
            await self.withdraw()
        browser = self._browser
        self._browser = None
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.async_cancel()
        tasks = tuple(self._refresh_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._refresh_tasks.clear()
        self._peers.clear()
        self._advertisement = None
        self._zeroconf = None
        await zeroconf.async_close()

    async def __aenter__(self) -> LanMdnsDiscovery:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
