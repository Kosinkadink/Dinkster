"""Digest-only LAN mapping service for active P2P seed leases."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Protocol, cast

import aiohttp
from aiohttp import web
from dinkster_assets import (
    AssetError,
    P2PDescriptorV1,
    require_digest,
    validate_p2p_descriptor,
)

from .lan_discovery import MAX_LAN_PEERS, LanMdnsDiscovery, LanNetworkPolicy, LanPeerEndpoint

LAN_MAPPING_VERSION = 2
LAN_MAPPING_PATH = "/dinkster-p2p/v2/mappings/{digest}"
LAN_MAPPING_V1_PATH = "/dinkster-p2p/v1/mappings/{digest}"
MAX_LAN_MAPPING_BYTES = 16 * 1024
DEFAULT_LAN_MAPPING_TIMEOUT = 2.0


class PeerNetworkPolicy(Protocol):
    def allows_peer(self, value: str | IPv4Address) -> bool: ...


class PeerDiscovery(Protocol):
    @property
    def network_policy(self) -> LanNetworkPolicy: ...

    async def peers(self, timeout: float) -> tuple[LanPeerEndpoint, ...]: ...


@dataclass(frozen=True)
class ActiveSeedMapping:
    digest: str
    size_bytes: int
    descriptor: P2PDescriptorV1
    expires_at: float
    peer_endpoints: tuple[tuple[str, int], ...]
    active: bool = True

    def __post_init__(self) -> None:
        require_digest(self.digest)
        if isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise AssetError("LAN seed mapping size must be a positive integer")
        if (
            isinstance(self.expires_at, bool)
            or not math.isfinite(self.expires_at)
            or self.expires_at <= 0
        ):
            raise AssetError("LAN seed mapping expiry must be positive")
        if not self.peer_endpoints:
            raise AssetError("LAN seed mapping must have a peer endpoint")
        for address, port in self.peer_endpoints:
            try:
                canonical_address = str(IPv4Address(address))
            except ValueError as error:
                raise AssetError("LAN seed mapping peer address must be IPv4") from error
            if canonical_address != address:
                raise AssetError("LAN seed mapping peer address must be canonical")
            if type(port) is not int or not 1 <= port <= 65535:
                raise AssetError("LAN seed mapping peer port must be between 1 and 65535")
        if len(dict(self.peer_endpoints)) != len(self.peer_endpoints):
            raise AssetError("LAN seed mapping peer addresses must be unique")

    def to_wire(
        self, local_address: str | None = None, *, version: int = LAN_MAPPING_VERSION
    ) -> dict[str, object]:
        if version == 1:
            return {
                "version": 1,
                "digest": self.digest,
                "sizeBytes": self.size_bytes,
                "descriptor": self.descriptor.to_wire(),
            }
        if version != LAN_MAPPING_VERSION:
            raise AssetError("unsupported LAN mapping version")
        endpoints = dict(self.peer_endpoints)
        if local_address is None:
            if len(endpoints) != 1:
                raise AssetError("LAN seed mapping requires the requested local address")
            peer_port = next(iter(endpoints.values()))
        else:
            try:
                peer_port = endpoints[str(IPv4Address(local_address))]
            except (KeyError, ValueError) as error:
                message = "LAN seed mapping has no peer on the requested interface"
                raise AssetError(message) from error
        return {
            "version": LAN_MAPPING_VERSION,
            "digest": self.digest,
            "sizeBytes": self.size_bytes,
            "descriptor": self.descriptor.to_wire(),
            "peerPort": peer_port,
        }


MappingLookup = Callable[[str], ActiveSeedMapping | None]
NetworkAllowed = Callable[[], bool]


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _strict_json_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("LAN mapping contains a duplicate or non-string field")
        result[key] = value
    return result


class LanMappingService:
    """Expose no inventory: one requested digest either maps now or is 404."""

    def __init__(
        self,
        lookup: MappingLookup,
        network_policy: LanNetworkPolicy,
        *,
        network_allowed: NetworkAllowed = lambda: True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lookup = lookup
        self._network_policy = network_policy
        self._network_allowed = network_allowed
        self._clock = clock

    @property
    def network_policy(self) -> LanNetworkPolicy:
        return self._network_policy

    def response_for(
        self,
        digest: str,
        remote: str | None,
        local_address: str | None = None,
        *,
        version: int = LAN_MAPPING_VERSION,
    ) -> dict[str, object] | None:
        if (
            remote is None
            or not self._network_allowed()
            or not self._network_policy.allows_peer(remote)
        ):
            return None
        try:
            require_digest(digest)
            mapping = self._lookup(digest)
        except (AssetError, OSError):
            return None
        if (
            mapping is None
            or mapping.digest != digest
            or not mapping.active
            or mapping.expires_at <= self._clock()
        ):
            return None
        try:
            descriptor = validate_p2p_descriptor(
                mapping.descriptor.to_wire(),
                asset_digest=mapping.digest,
                size=mapping.size_bytes,
            )
        except (AssetError, TypeError, ValueError):
            return None
        if not self._network_allowed():
            return None
        try:
            return ActiveSeedMapping(
                mapping.digest,
                mapping.size_bytes,
                descriptor,
                mapping.expires_at,
                mapping.peer_endpoints,
            ).to_wire(local_address, version=version)
        except AssetError:
            return None

    async def _handle_mapping(self, request: web.Request, *, version: int) -> web.Response:
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        local_address = (
            sockname[0]
            if isinstance(sockname, tuple) and sockname and isinstance(sockname[0], str)
            else None
        )
        wire = self.response_for(
            request.match_info["digest"], request.remote, local_address, version=version
        )
        if wire is None:
            raise web.HTTPNotFound(headers={"Cache-Control": "no-store"})
        return web.json_response(wire, headers={"Cache-Control": "no-store"})

    async def handle_mapping(self, request: web.Request) -> web.Response:
        return await self._handle_mapping(request, version=LAN_MAPPING_VERSION)

    async def handle_mapping_v1(self, request: web.Request) -> web.Response:
        return await self._handle_mapping(request, version=1)

    def application(self) -> web.Application:
        app = web.Application()
        app.router.add_get(LAN_MAPPING_V1_PATH, self.handle_mapping_v1, allow_head=False)
        app.router.add_get(LAN_MAPPING_PATH, self.handle_mapping, allow_head=False)
        return app


@dataclass(frozen=True)
class LanMapping:
    endpoint: LanPeerEndpoint
    digest: str
    size_bytes: int
    descriptor: P2PDescriptorV1
    peer_port: int | None


async def fetch_lan_mapping(
    endpoint: LanPeerEndpoint,
    digest: str,
    network_policy: PeerNetworkPolicy,
    *,
    network_allowed: NetworkAllowed = lambda: True,
    timeout: float = DEFAULT_LAN_MAPPING_TIMEOUT,
    session: aiohttp.ClientSession | None = None,
) -> LanMapping | None:
    """Fetch and strictly validate one untrusted peer mapping without redirects."""
    require_digest(digest)
    if timeout <= 0:
        raise ValueError("LAN mapping timeout must be positive")
    if not network_allowed() or not network_policy.allows_peer(endpoint.address):
        return None
    own_session = session is None
    http = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    body = b""
    response_version = LAN_MAPPING_VERSION
    try:
        for version, path in (
            (LAN_MAPPING_VERSION, LAN_MAPPING_PATH),
            (1, LAN_MAPPING_V1_PATH),
        ):
            remaining = deadline - loop.time()
            if remaining <= 0 or not network_allowed():
                return None
            url = endpoint.origin + path.format(digest=digest)
            request_timeout = aiohttp.ClientTimeout(total=remaining)
            async with http.get(url, allow_redirects=False, timeout=request_timeout) as response:
                if version == LAN_MAPPING_VERSION and response.status in {404, 405}:
                    continue
                if response.status != 200 or response.content_type != "application/json":
                    return None
                if (
                    response.content_length is not None
                    and response.content_length > MAX_LAN_MAPPING_BYTES
                ):
                    return None
                body = await response.content.read(MAX_LAN_MAPPING_BYTES + 1)
                if len(body) > MAX_LAN_MAPPING_BYTES or not network_allowed():
                    return None
                response_version = version
                break
    except (aiohttp.ClientError, TimeoutError, OSError):
        return None
    finally:
        if own_session:
            await http.close()
    try:
        parsed_object = cast(
            "object",
            json.loads(
                body,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        return None
    if not isinstance(parsed_object, dict):
        return None
    parsed = cast("dict[str, object]", parsed_object)
    expected_fields = {
        "version",
        "digest",
        "sizeBytes",
        "descriptor",
    }
    if response_version == LAN_MAPPING_VERSION:
        expected_fields.add("peerPort")
    if set(parsed) != expected_fields:
        return None
    version = parsed.get("version")
    mapped_digest = parsed.get("digest")
    size_bytes = parsed.get("sizeBytes")
    descriptor_wire = parsed.get("descriptor")
    peer_port = parsed.get("peerPort")
    if (
        isinstance(version, bool)
        or version != response_version
        or mapped_digest != digest
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not isinstance(descriptor_wire, Mapping)
        or (
            response_version == LAN_MAPPING_VERSION
            and (type(peer_port) is not int or not 1 <= peer_port <= 65535)
        )
    ):
        return None
    try:
        descriptor = validate_p2p_descriptor(
            cast("Mapping[str, object]", descriptor_wire),
            asset_digest=digest,
            size=size_bytes,
        )
    except (AssetError, TypeError, ValueError):
        return None
    if not network_allowed():
        return None
    return LanMapping(
        endpoint,
        digest,
        size_bytes,
        descriptor,
        cast("int", peer_port) if response_version == LAN_MAPPING_VERSION else None,
    )


class RunningLanMappingServer:
    def __init__(
        self,
        runner: web.AppRunner,
        discovery: LanMdnsDiscovery,
        port: int,
    ) -> None:
        self._runner = runner
        self._discovery = discovery
        self.port = port
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._discovery.withdraw()
        finally:
            await self._runner.cleanup()

    async def __aenter__(self) -> RunningLanMappingServer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


async def start_lan_mapping_server(
    service: LanMappingService,
    discovery: LanMdnsDiscovery,
    *,
    port: int = 0,
) -> RunningLanMappingServer:
    """Bind only eligible LAN addresses, then advertise the actual port."""
    if isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("LAN mapping port must be between 0 and 65535")
    network_policy = service.network_policy
    if discovery.network_policy != network_policy:
        raise ValueError("LAN mapping and discovery policies must match")
    runner = web.AppRunner(service.application(), access_log=None)
    await runner.setup()
    try:
        bound_port = port
        for position, address in enumerate(network_policy.addresses):
            site = web.TCPSite(runner, address, bound_port)
            await site.start()
            if position == 0 and bound_port == 0:
                _, separator, port_text = site.name.rpartition(":")
                if not separator or not port_text.isdecimal():
                    raise RuntimeError("LAN mapping server bound no address")
                bound_port = int(port_text)
        if bound_port == 0:
            raise RuntimeError("LAN mapping server requires an eligible interface")
        await discovery.advertise(bound_port)
    except BaseException:
        await runner.cleanup()
        raise
    return RunningLanMappingServer(runner, discovery, bound_port)


async def discover_lan_mappings(
    discovery: PeerDiscovery,
    digest: str,
    *,
    network_allowed: NetworkAllowed = lambda: True,
    discovery_timeout: float = DEFAULT_LAN_MAPPING_TIMEOUT,
) -> tuple[LanMapping, ...]:
    """Probe discovered peers concurrently inside one overall deadline."""
    if discovery_timeout <= 0:
        raise ValueError("LAN discovery timeout must be positive")
    if not network_allowed():
        return ()
    network_policy = discovery.network_policy
    loop = asyncio.get_running_loop()
    deadline = loop.time() + discovery_timeout
    seen_endpoints: set[LanPeerEndpoint] = set()
    seen_peers: set[tuple[IPv4Address, int]] = set()
    probes: set[asyncio.Task[LanMapping | None]] = set()
    mappings: list[LanMapping] = []
    pending_peers = asyncio.create_task(discovery.peers(discovery_timeout))
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=discovery_timeout)
    ) as session:
        try:
            while network_allowed():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                waiting: set[asyncio.Task[object]] = set(probes)
                if pending_peers is not None:
                    waiting.add(pending_peers)
                if not waiting:
                    break
                done, _pending = await asyncio.wait(
                    waiting, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                for probe in probes.intersection(done):
                    probes.remove(probe)
                    try:
                        mapping = probe.result()
                    except (AssetError, OSError):
                        continue
                    if mapping is not None:
                        mappings.append(mapping)
                if mappings and not probes:
                    return tuple(mappings) if network_allowed() else ()
                if pending_peers is not None and pending_peers in done:
                    try:
                        peers = pending_peers.result()
                    except (TimeoutError, OSError):
                        peers = ()
                    pending_peers = None
                    new_endpoints = set(peers[:MAX_LAN_PEERS]) - seen_endpoints
                    seen_endpoints.update(new_endpoints)
                    for peer in peers[:MAX_LAN_PEERS]:
                        key = (peer.address, peer.port)
                        if key in seen_peers or len(seen_peers) >= MAX_LAN_PEERS:
                            continue
                        seen_peers.add(key)
                        probes.add(
                            asyncio.create_task(
                                fetch_lan_mapping(
                                    peer,
                                    digest,
                                    network_policy,
                                    network_allowed=network_allowed,
                                    timeout=remaining,
                                    session=session,
                                )
                            )
                        )
                    if (
                        isinstance(discovery, LanMdnsDiscovery)
                        and new_endpoints
                        and len(seen_endpoints) < MAX_LAN_PEERS
                    ):
                        pending_peers = asyncio.create_task(
                            discovery.peers(remaining, exclude=frozenset(seen_endpoints))
                        )
        finally:
            cleanup: set[asyncio.Task[object]] = set(probes)
            if pending_peers is not None:
                cleanup.add(pending_peers)
            for task in cleanup:
                task.cancel()
            await asyncio.gather(*cleanup, return_exceptions=True)
    return tuple(mappings) if network_allowed() else ()
