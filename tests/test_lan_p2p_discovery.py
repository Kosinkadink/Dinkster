from __future__ import annotations

import asyncio
import socket
from ipaddress import IPv4Address, IPv4Network
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from dinkster_server import (
    LAN_P2P_SERVICE_TYPE,
    LanInterface,
    LanMdnsDiscovery,
    LanNetworkPolicy,
    lan_interfaces,
)


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
