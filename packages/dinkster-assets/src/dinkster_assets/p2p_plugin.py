"""Registration seam for the optional P2P runtime plugin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from typing import Protocol


class P2PLanInterface(Protocol):
    name: str
    address: IPv4Address
    network: IPv4Network


@dataclass(frozen=True)
class P2PPluginRegistration:
    default_settings: Callable[[], dict[str, object]]
    normalize_settings: Callable[[object], dict[str, object]]
    lan_interfaces: Callable[[], tuple[P2PLanInterface, ...]]


_registration: P2PPluginRegistration | None = None


def register_p2p_plugin(registration: P2PPluginRegistration) -> None:
    """Register the process-wide P2P implementation."""
    global _registration
    _registration = registration


def p2p_plugin() -> P2PPluginRegistration | None:
    return _registration
