from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable
from ipaddress import IPv4Address, IPv4Network
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from dinkster_p2p import (
    LanInterface as PluginLanInterface,
)
from dinkster_p2p import (
    LanNetworkPolicy as PluginLanNetworkPolicy,
)
from dinkster_p2p import (
    lan_interfaces as plugin_lan_interfaces,
)
from dinkster_server import (
    LAN_P2P_SERVICE_TYPE,
    LanInterface,
    LanMdnsDiscovery,
    LanNetworkPolicy,
    lan_interfaces,
)
from zeroconf import NonUniqueNameException, ServiceInfo


def _interface(
    address: str = "192.168.50.10",
    network: str = "192.168.50.0/24",
) -> LanInterface:
    return LanInterface("ethernet", IPv4Address(address), IPv4Network(network))


def test_lan_network_policy_accepts_only_same_private_interface() -> None:
    policy = LanNetworkPolicy((_interface(),))
    assert policy.addresses == ("192.168.50.10",)
    assert policy.allows_peer("192.168.50.200")
    assert not policy.allows_peer("192.168.51.2")
    assert not policy.allows_peer("100.64.0.2")
    assert not policy.allows_peer("8.8.8.8")
    assert not policy.allows_peer("127.0.0.1")
    assert not policy.allows_peer("169.254.1.2")
    assert not policy.allows_peer("192.168.50.0")
    assert not policy.allows_peer("192.168.50.255")
    assert not policy.allows_peer("not-an-address")


def test_server_and_plugin_lan_policies_match_at_network_boundaries() -> None:
    server_policy = LanNetworkPolicy((_interface(),))
    plugin_policy = PluginLanNetworkPolicy(
        (
            PluginLanInterface(
                "ethernet",
                IPv4Address("192.168.50.10"),
                IPv4Network("192.168.50.0/24"),
            ),
        )
    )
    addresses = (
        "192.168.50.0",
        "192.168.50.1",
        "192.168.50.200",
        "192.168.50.255",
        "192.168.51.1",
        "10.0.0.1",
        "127.0.0.1",
        "not-an-address",
    )
    assert tuple(server_policy.allows_peer(value) for value in addresses) == tuple(
        plugin_policy.allows_peer(value) for value in addresses
    )


def test_lan_interfaces_excludes_public_loopback_down_and_point_to_point() -> None:
    up = SimpleNamespace(isup=True, flags="up,broadcast,multicast")
    down = SimpleNamespace(isup=False, flags="up,broadcast,multicast")
    point_to_point = SimpleNamespace(isup=True, flags="up,pointopoint")

    def address(value: str, netmask: str) -> SimpleNamespace:
        return SimpleNamespace(family=socket.AF_INET, address=value, netmask=netmask)

    with (
        patch(
            "dinkster_p2p.lan.psutil.net_if_stats",
            return_value={
                "lan": up,
                "public": up,
                "shared": up,
                "loopback": up,
                "down": down,
                "tunnel": point_to_point,
            },
        ),
        patch(
            "dinkster_p2p.lan.psutil.net_if_addrs",
            return_value={
                "lan": [address("192.168.1.10", "255.255.255.0")],
                "public": [address("8.8.8.8", "255.255.255.0")],
                "shared": [address("100.64.1.2", "255.255.255.0")],
                "loopback": [address("127.0.0.1", "255.0.0.0")],
                "down": [address("10.1.0.2", "255.255.0.0")],
                "tunnel": [address("100.64.1.2", "255.255.255.255")],
            },
        ),
        patch(
            "dinkster_server.lan_discovery.p2p_plugin",
            return_value=SimpleNamespace(lan_interfaces=plugin_lan_interfaces),
        ),
    ):
        assert lan_interfaces() == (
            LanInterface(
                "lan",
                IPv4Address("192.168.1.10"),
                IPv4Network("192.168.1.0/24"),
            ),
        )


def test_lan_mdns_requires_closed_local_configuration() -> None:
    with pytest.raises(ValueError, match="eligible interface"):
        LanMdnsDiscovery("instance", LanNetworkPolicy(()))
    with pytest.raises(ValueError, match="instance id"):
        LanMdnsDiscovery("bad.instance", LanNetworkPolicy((_interface(),)))

    async def scenario() -> None:
        discovery = LanMdnsDiscovery("instance", LanNetworkPolicy((_interface(),)))
        with pytest.raises(ValueError, match="port"):
            await discovery.advertise(0)
        with pytest.raises(ValueError, match="timeout"):
            await discovery.peers(0)
        await discovery.close()
        await discovery.close()

    asyncio.run(scenario())


def test_lan_mdns_constructor_opens_no_socket_until_started() -> None:
    with patch("dinkster_server.lan_discovery.AsyncZeroconf") as zeroconf:
        discovery = LanMdnsDiscovery("disabled-instance", LanNetworkPolicy((_interface(),)))
        assert isinstance(zeroconf, Mock)
        zeroconf.assert_not_called()
        asyncio.run(discovery.close())


def test_lan_mdns_readvertises_after_its_withdrawn_record_remains_cached() -> None:
    class StaleCache:
        def __init__(self, events: list[str]) -> None:
            self.pointer: object | None = None
            self.events = events

        def current_entry_with_name_and_alias(self, name: str, alias: str) -> object | None:
            assert name == LAN_P2P_SERVICE_TYPE
            assert alias == f"same-instance.{LAN_P2P_SERVICE_TYPE}"
            return self.pointer

        def async_remove_records(self, records: tuple[object, ...]) -> None:
            assert records == (self.pointer,)
            self.events.append("evict")
            self.pointer = None

    class StaleZeroconf:
        def __init__(self) -> None:
            self.zeroconf = self
            self.events: list[str] = []
            self.cache = StaleCache(self.events)
            self.own_pointer: object | None = None
            self.registration_attempts = 0

        async def async_register_service(self, info: ServiceInfo) -> Awaitable[None]:
            self.registration_attempts += 1
            if self.registration_attempts == 2:
                self.cache.pointer = self.own_pointer
                self.events.append("probe-cache")
            if self.cache.current_entry_with_name_and_alias(info.type, info.name) is not None:
                raise NonUniqueNameException
            self.cache.pointer = info.dns_pointer()
            self.own_pointer = self.cache.pointer
            self.events.append("register")
            return asyncio.sleep(0)

        async def async_unregister_service(self, _info: ServiceInfo) -> Awaitable[None]:
            async def goodbye() -> None:
                self.events.append("goodbye")

            return goodbye()

    async def scenario() -> None:
        discovery = LanMdnsDiscovery("same-instance", LanNetworkPolicy((_interface(),)))
        stale_zeroconf = StaleZeroconf()
        discovery._zeroconf = cast("object", stale_zeroconf)  # type: ignore[assignment]

        await discovery.advertise(41001)
        await discovery.withdraw()
        stale_zeroconf.cache.pointer = stale_zeroconf.own_pointer
        stale_zeroconf.events.append("late-cache")
        await discovery.advertise(41002)
        await discovery.withdraw()
        assert stale_zeroconf.events == [
            "register",
            "goodbye",
            "evict",
            "late-cache",
            "evict",
            "probe-cache",
            "evict",
            "register",
            "goodbye",
            "evict",
        ]

    asyncio.run(scenario())


def test_lan_mdns_readvertisement_preserves_real_name_conflicts() -> None:
    class EmptyCache:
        @staticmethod
        def current_entry_with_name_and_alias(_name: str, _alias: str) -> None:
            return None

    class ConflictingZeroconf:
        def __init__(self) -> None:
            self.zeroconf = self
            self.cache = EmptyCache()
            self.registration_attempts = 0

        async def async_register_service(self, _info: ServiceInfo) -> Awaitable[None]:
            self.registration_attempts += 1
            raise NonUniqueNameException

    async def scenario() -> None:
        discovery = LanMdnsDiscovery("conflicting-instance", LanNetworkPolicy((_interface(),)))
        conflicting_zeroconf = ConflictingZeroconf()
        discovery._zeroconf = cast("object", conflicting_zeroconf)  # type: ignore[assignment]

        with pytest.raises(NonUniqueNameException):
            await discovery.advertise(41001)
        assert conflicting_zeroconf.registration_attempts == 2

    asyncio.run(scenario())


def test_two_mdns_instances_discover_only_lan_addresses() -> None:
    interfaces = lan_interfaces()
    if not interfaces:
        pytest.skip("host has no eligible multicast LAN interface")
    policy = LanNetworkPolicy((interfaces[0],))
    first_id = f"dinkster-test-first-{uuid4().hex}"
    second_id = f"dinkster-test-second-{uuid4().hex}"

    async def scenario() -> None:
        first = LanMdnsDiscovery(first_id, policy)
        second = LanMdnsDiscovery(second_id, policy)
        try:
            await first.start()
            await second.start()
            await first.advertise(41001)
            await second.advertise(41002)
            first_peers, second_peers = await asyncio.gather(first.peers(5), second.peers(5))
            assert any(
                peer.service_name == f"{second_id}.{LAN_P2P_SERVICE_TYPE}"
                and peer.address == interfaces[0].address
                and peer.port == 41002
                for peer in first_peers
            )
            assert any(
                peer.service_name == f"{first_id}.{LAN_P2P_SERVICE_TYPE}"
                and peer.address == interfaces[0].address
                and peer.port == 41001
                for peer in second_peers
            )
            assert all(policy.allows_peer(peer.address) for peer in (*first_peers, *second_peers))
        finally:
            await first.close()
            await second.close()

    asyncio.run(scenario())
