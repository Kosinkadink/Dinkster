from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import aiohttp
import pytest
from aiohttp import web
from dinkster_assets import (
    P2P_PIECE_LENGTH,
    P2PDescriptorV1,
    canonical_p2p_info,
    digest_bytes,
)
from dinkster_server import (
    LAN_MAPPING_PATH,
    ActiveSeedMapping,
    LanInterface,
    LanMappingService,
    LanMdnsDiscovery,
    LanNetworkPolicy,
    LanPeerEndpoint,
    discover_lan_mappings,
    fetch_lan_mapping,
    lan_interfaces,
    start_lan_mapping_server,
)
from dinkster_server import lan_mapping as lan_mapping_module

DATA = b"LAN mapping fixture"
DIGEST = digest_bytes(DATA)
SIZE_BYTES = len(DATA)
FILE_ROOT = hashlib.sha256(DATA).hexdigest()
INFO_HASH = hashlib.sha256(
    canonical_p2p_info(asset_digest=DIGEST, size=SIZE_BYTES, file_root=FILE_ROOT)
).hexdigest()
DESCRIPTOR = P2PDescriptorV1(
    protocol="bittorrent-v2",
    info_hash=INFO_HASH,
    file_root=FILE_ROOT,
    piece_length=P2P_PIECE_LENGTH,
)
DESCRIPTOR_WIRE = DESCRIPTOR.to_wire()
LAN_MAPPING_V1_PATH = "/dinkster-p2p/v1/mappings/{digest}"


def _policy(address: str = "192.168.70.10") -> LanNetworkPolicy:
    return LanNetworkPolicy(
        (
            LanInterface(
                "ethernet",
                IPv4Address(address),
                IPv4Network("192.168.70.0/24"),
            ),
        )
    )


def _lease(
    *,
    digest: str = DIGEST,
    size_bytes: int = SIZE_BYTES,
    descriptor: P2PDescriptorV1 | None = None,
    expires_at: float = 200.0,
    peer_address: str = "192.168.70.1",
    peer_port: int = 42069,
    active: bool = True,
) -> ActiveSeedMapping:
    return ActiveSeedMapping(
        digest,
        size_bytes,
        descriptor or DESCRIPTOR,
        expires_at,
        ((peer_address, peer_port),),
        active,
    )


class AllowAllPeerPolicy:
    def allows_peer(self, value: str | IPv4Address) -> bool:
        del value
        return True


@dataclass(frozen=True)
class FakeDiscovery:
    network_policy: LanNetworkPolicy
    endpoints: tuple[LanPeerEndpoint, ...]

    async def peers(self, timeout: float) -> tuple[LanPeerEndpoint, ...]:
        assert timeout > 0
        return self.endpoints


def test_mapping_service_returns_only_exact_active_requested_digest() -> None:
    leases = {DIGEST: _lease()}
    service = LanMappingService(
        leases.get,
        _policy(),
        clock=lambda: 100.0,
    )
    assert service.response_for(DIGEST, "192.168.70.44") == {
        "version": 2,
        "digest": DIGEST,
        "sizeBytes": SIZE_BYTES,
        "descriptor": DESCRIPTOR_WIRE,
        "peerPort": 42069,
    }
    assert service.response_for("blake3:" + "0" * 64, "192.168.70.44") is None
    assert service.response_for("not-a-digest", "192.168.70.44") is None
    assert service.response_for(DIGEST, "8.8.8.8") is None
    assert service.response_for(DIGEST, "192.168.71.44") is None

    routes = {
        (route.method, route.resource.canonical)
        for route in service.application().router.routes()
        if route.resource is not None
    }
    assert routes == {
        ("GET", LAN_MAPPING_V1_PATH),
        ("GET", LAN_MAPPING_PATH),
    }


def test_mapping_service_preserves_exact_version_1_response() -> None:
    service = LanMappingService(lambda _digest: _lease(), _policy(), clock=lambda: 100.0)

    assert service.response_for(DIGEST, "192.168.70.44", version=1) == {
        "version": 1,
        "digest": DIGEST,
        "sizeBytes": SIZE_BYTES,
        "descriptor": DESCRIPTOR_WIRE,
    }


def test_mapping_service_returns_port_for_requested_local_interface() -> None:
    lease = ActiveSeedMapping(
        DIGEST,
        SIZE_BYTES,
        DESCRIPTOR,
        200.0,
        (("192.168.70.1", 42069), ("192.168.80.1", 42070)),
    )
    service = LanMappingService(lambda _digest: lease, _policy(), clock=lambda: 100.0)

    assert service.response_for(DIGEST, "192.168.70.44", "192.168.70.1") == {
        "version": 2,
        "digest": DIGEST,
        "sizeBytes": SIZE_BYTES,
        "descriptor": DESCRIPTOR_WIRE,
        "peerPort": 42069,
    }
    assert service.response_for(DIGEST, "192.168.70.44", "192.168.80.1") == {
        "version": 2,
        "digest": DIGEST,
        "sizeBytes": SIZE_BYTES,
        "descriptor": DESCRIPTOR_WIRE,
        "peerPort": 42070,
    }
    assert service.response_for(DIGEST, "192.168.70.44", "192.168.90.1") is None


@pytest.mark.parametrize(
    "lease",
    [
        _lease(active=False),
        _lease(expires_at=100.0),
        _lease(digest="blake3:" + "0" * 64),
        _lease(
            descriptor=P2PDescriptorV1(
                protocol="bittorrent-v2",
                info_hash="1" * 64,
                file_root="2" * 64,
                piece_length=P2P_PIECE_LENGTH,
            )
        ),
    ],
)
def test_mapping_service_hides_revoked_expired_mismatched_and_forged_leases(
    lease: ActiveSeedMapping,
) -> None:
    service = LanMappingService(
        lambda _digest: lease,
        _policy(),
        clock=lambda: 100.0,
    )
    assert service.response_for(DIGEST, "192.168.70.44") is None


def test_mapping_service_hides_everything_after_metered_transition() -> None:
    allowed = [True]
    service = LanMappingService(
        lambda _digest: _lease(),
        _policy(),
        network_allowed=lambda: allowed[0],
        clock=lambda: 100.0,
    )
    assert service.response_for(DIGEST, "192.168.70.44") is not None
    allowed[0] = False
    assert service.response_for(DIGEST, "192.168.70.44") is None


def test_mapping_service_rechecks_metered_state_after_lookup() -> None:
    allowed = [True]

    def lookup(_digest: str) -> ActiveSeedMapping:
        allowed[0] = False
        return _lease()

    service = LanMappingService(
        lookup,
        _policy(),
        network_allowed=lambda: allowed[0],
        clock=lambda: 100.0,
    )
    assert service.response_for(DIGEST, "192.168.70.44") is None


def test_fetch_mapping_rejects_forged_oversized_redirect_and_disappearing_peer() -> None:
    async def scenario() -> None:
        current_response: list[tuple[int, object, str]] = [
            (200, _lease().to_wire(), "application/json")
        ]

        async def mapping(_request: web.Request) -> web.Response:
            status, body, content_type = current_response[0]
            if status == 302:
                raise web.HTTPFound("https://public.example/forbidden")
            if body == "disconnect":
                raise ConnectionResetError("seeder disappeared")
            if isinstance(body, bytes):
                return web.Response(body=body, content_type=content_type, status=status)
            return web.json_response(body, status=status)

        app = web.Application()
        app.router.add_get(LAN_MAPPING_PATH, mapping, allow_head=False)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = int(runner.addresses[0][1])
        endpoint = LanPeerEndpoint("peer", IPv4Address("127.0.0.1"), port)
        policy = AllowAllPeerPolicy()
        try:
            found = await fetch_lan_mapping(endpoint, DIGEST, policy)
            assert found is not None and found.descriptor.to_wire() == DESCRIPTOR_WIRE
            assert found.peer_port == 42069

            forged = _lease().to_wire()
            forged["digest"] = "blake3:" + "0" * 64
            current_response[0] = (200, forged, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            invalid_peer = _lease().to_wire()
            invalid_peer["peerPort"] = 0
            current_response[0] = (200, invalid_peer, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            wrong = _lease().to_wire()
            cast("dict[str, object]", wrong["descriptor"])["fileRoot"] = "3" * 64
            current_response[0] = (200, wrong, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            duplicate = (
                b'{"version":1,"version":1,"digest":"'
                + DIGEST.encode()
                + f'","sizeBytes":{SIZE_BYTES},"descriptor":{{}}}}'.encode()
            )
            current_response[0] = (200, duplicate, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            nested = b"[" * 1100 + b"0" + b"]" * 1100
            current_response[0] = (200, nested, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            current_response[0] = (200, b"x" * 20_000, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            current_response[0] = (302, {}, "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None

            current_response[0] = (200, "disconnect", "application/json")
            assert await fetch_lan_mapping(endpoint, DIGEST, policy) is None
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_fetch_mapping_rechecks_metered_state_after_response() -> None:
    async def scenario() -> None:
        allowed = [True]

        async def mapping(_request: web.Request) -> web.Response:
            allowed[0] = False
            return web.json_response(_lease().to_wire())

        app = web.Application()
        app.router.add_get(LAN_MAPPING_PATH, mapping, allow_head=False)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        endpoint = LanPeerEndpoint("peer", IPv4Address("127.0.0.1"), runner.addresses[0][1])
        policy = AllowAllPeerPolicy()
        try:
            async with aiohttp.ClientSession() as session:
                assert (
                    await fetch_lan_mapping(
                        endpoint,
                        DIGEST,
                        policy,
                        network_allowed=lambda: allowed[0],
                        session=session,
                    )
                    is None
                )
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_version_2_client_falls_back_only_for_unavailable_version_2_route() -> None:
    async def scenario() -> None:
        requests: list[str] = []
        version_2_status = [404]
        version_1_delay = [0.0]
        disable_after_version_2 = [False]
        network_allowed = [True]

        async def mapping(request: web.Request) -> web.Response:
            requests.append(request.path)
            if request.path.startswith("/dinkster-p2p/v2/"):
                if version_2_status[0] == 200:
                    return web.json_response({"version": 2})
                if disable_after_version_2[0]:
                    network_allowed[0] = False
                return web.Response(status=version_2_status[0])
            await asyncio.sleep(version_1_delay[0])
            wire = _lease().to_wire()
            wire["version"] = 1
            wire.pop("peerPort")
            return web.json_response(wire)

        app = web.Application()
        app.router.add_get(LAN_MAPPING_PATH, mapping, allow_head=False)
        app.router.add_get(LAN_MAPPING_V1_PATH, mapping, allow_head=False)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        endpoint = LanPeerEndpoint("peer", IPv4Address("127.0.0.1"), runner.addresses[0][1])
        try:
            for unavailable_status in (404, 405):
                requests.clear()
                version_2_status[0] = unavailable_status
                found = await fetch_lan_mapping(endpoint, DIGEST, AllowAllPeerPolicy())
                assert found is not None and found.peer_port is None
                assert requests == [
                    f"/dinkster-p2p/v2/mappings/{DIGEST}",
                    f"/dinkster-p2p/v1/mappings/{DIGEST}",
                ]

            for status in (200, 403, 500):
                requests.clear()
                version_2_status[0] = status
                assert await fetch_lan_mapping(endpoint, DIGEST, AllowAllPeerPolicy()) is None
                assert requests == [f"/dinkster-p2p/v2/mappings/{DIGEST}"]

            requests.clear()
            version_2_status[0] = 404
            disable_after_version_2[0] = True
            assert (
                await fetch_lan_mapping(
                    endpoint,
                    DIGEST,
                    AllowAllPeerPolicy(),
                    network_allowed=lambda: network_allowed[0],
                )
                is None
            )
            assert requests == [f"/dinkster-p2p/v2/mappings/{DIGEST}"]

            requests.clear()
            disable_after_version_2[0] = False
            network_allowed[0] = True
            version_2_status[0] = 404
            version_1_delay[0] = 0.04
            assert (
                await fetch_lan_mapping(
                    endpoint,
                    DIGEST,
                    AllowAllPeerPolicy(),
                    timeout=0.03,
                )
                is None
            )
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_two_lan_instances_discover_and_fetch_digest_mapping() -> None:
    interfaces = lan_interfaces()
    if not interfaces:
        pytest.skip("host has no eligible multicast LAN interface")
    policy = LanNetworkPolicy((interfaces[0],))
    namespace = uuid4().hex
    data = f"LAN mapping fixture {namespace}".encode()
    digest = digest_bytes(data)
    file_root = hashlib.sha256(data).hexdigest()
    descriptor = P2PDescriptorV1(
        protocol="bittorrent-v2",
        info_hash=hashlib.sha256(
            canonical_p2p_info(asset_digest=digest, size=len(data), file_root=file_root)
        ).hexdigest(),
        file_root=file_root,
        piece_length=P2P_PIECE_LENGTH,
    )
    lease = _lease(
        digest=digest,
        size_bytes=len(data),
        descriptor=descriptor,
        peer_address=str(interfaces[0].address),
    )

    async def scenario() -> None:
        seeder_discovery = LanMdnsDiscovery(f"dinkster-map-seeder-{namespace}", policy)
        downloader_discovery = LanMdnsDiscovery(f"dinkster-map-downloader-{namespace}", policy)
        service = LanMappingService(
            lambda requested: lease if requested == digest else None,
            policy,
            clock=lambda: 100.0,
        )
        server = None
        try:
            await downloader_discovery.start()
            server = await start_lan_mapping_server(service, seeder_discovery)
            mappings = await discover_lan_mappings(
                downloader_discovery,
                digest,
                discovery_timeout=5.0,
            )
            assert len(mappings) == 1
            assert mappings[0].endpoint.address == interfaces[0].address
            assert mappings[0].endpoint.port == server.port
            assert mappings[0].descriptor.to_wire() == descriptor.to_wire()
        finally:
            if server is not None:
                await server.close()
            await seeder_discovery.close()
            await downloader_discovery.close()

    asyncio.run(scenario())


def test_mapping_discovery_retains_fast_peer_when_another_stalls() -> None:
    async def scenario() -> None:
        async def mapping(_request: web.Request) -> web.Response:
            return web.json_response(_lease().to_wire())

        async def stalled(_request: web.Request) -> web.Response:
            await asyncio.sleep(0.2)
            return web.json_response(_lease().to_wire())

        app = web.Application()
        app.router.add_get(LAN_MAPPING_PATH, mapping, allow_head=False)
        stalled_app = web.Application()
        stalled_app.router.add_get(LAN_MAPPING_PATH, stalled, allow_head=False)
        runner = web.AppRunner(app, access_log=None)
        stalled_runner = web.AppRunner(stalled_app, access_log=None)
        await runner.setup()
        await stalled_runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        stalled_site = web.TCPSite(stalled_runner, "127.0.0.1", 0)
        await site.start()
        await stalled_site.start()
        policy = _policy()
        discovery = FakeDiscovery(
            policy,
            (
                LanPeerEndpoint(
                    "fast",
                    IPv4Address("127.0.0.1"),
                    runner.addresses[0][1],
                ),
                LanPeerEndpoint(
                    "stalled",
                    IPv4Address("127.0.0.1"),
                    stalled_runner.addresses[0][1],
                ),
            ),
        )
        try:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(
                    LanNetworkPolicy,
                    "allows_peer",
                    lambda _self, _value: True,
                )
                mappings = await discover_lan_mappings(
                    discovery,
                    DIGEST,
                    discovery_timeout=0.1,
                )
            assert [mapping.endpoint.service_name for mapping in mappings] == ["fast"]
        finally:
            await runner.cleanup()
            await stalled_runner.cleanup()

    asyncio.run(scenario())


def test_mapping_discovery_caps_and_deduplicates_untrusted_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = _policy()
        endpoints = tuple(
            LanPeerEndpoint(
                f"peer-{position}",
                IPv4Address(f"192.168.70.{position + 20}"),
                41000 + position,
            )
            for position in range(40)
        )
        discovery = FakeDiscovery(policy, (endpoints[0], endpoints[0], *endpoints[1:]))
        called: list[LanPeerEndpoint] = []

        async def fetch(
            endpoint: LanPeerEndpoint,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            called.append(endpoint)

        monkeypatch.setattr(lan_mapping_module, "fetch_lan_mapping", fetch)
        assert (
            await discover_lan_mappings(
                discovery,
                DIGEST,
                discovery_timeout=0.1,
            )
            == ()
        )
        assert called == list(endpoints[:31])

    asyncio.run(scenario())


@pytest.mark.parametrize("stalled", [False, True])
def test_mapping_discovery_includes_target_arriving_after_unrelated_cached_peer(
    monkeypatch: pytest.MonkeyPatch, stalled: bool
) -> None:
    async def scenario() -> None:
        discovery = LanMdnsDiscovery("late-target", _policy())
        monkeypatch.setattr(discovery, "start", AsyncMock())
        monkeypatch.setattr(LanNetworkPolicy, "allows_peer", lambda *_args: True)
        unrelated_requested = asyncio.Event()
        never = asyncio.Event()

        async def unrelated(_request: web.Request) -> web.Response:
            unrelated_requested.set()
            if stalled:
                await never.wait()
            return web.Response(status=404)

        async def target(_request: web.Request) -> web.Response:
            return web.json_response(_lease().to_wire())

        app = web.Application()
        app.router.add_get(LAN_MAPPING_PATH, target)
        other = web.Application()
        other.router.add_get(LAN_MAPPING_PATH, unrelated)
        runner = web.AppRunner(app)
        other_runner = web.AppRunner(other)
        await runner.setup()
        await other_runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        await web.TCPSite(other_runner, "127.0.0.1", 0).start()
        old = LanPeerEndpoint("unrelated", IPv4Address("127.0.0.1"), other_runner.addresses[0][1])
        late = LanPeerEndpoint("target", IPv4Address("127.0.0.1"), runner.addresses[0][1])
        discovery._peers = {old.service_name: (old,)}

        async def announce_target() -> None:
            await unrelated_requested.wait()
            discovery._peers[late.service_name] = (late,)
            discovery._changed.set()

        announce = asyncio.create_task(announce_target())
        try:
            mappings = await discover_lan_mappings(discovery, DIGEST)
            assert [mapping.endpoint for mapping in mappings] == [late]
            assert announce.done()
        finally:
            announce.cancel()
            await asyncio.gather(announce, return_exceptions=True)
            never.set()
            await runner.cleanup()
            await other_runner.cleanup()

    asyncio.run(scenario())
