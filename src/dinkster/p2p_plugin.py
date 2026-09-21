"""Activation boundary for the optional P2P runtime plugin."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from dinkster_assets import P2PPluginRegistration, register_p2p_plugin


@dataclass(frozen=True)
class P2PPlugin:
    controller: type[Any]
    add_routes: Callable[..., None]


def _enabled_defaults(plugin: ModuleType) -> dict[str, object]:
    settings = plugin.default_p2p_settings()
    settings["downloadsEnabled"] = True
    settings["seedingEnabled"] = True
    return settings


def load_p2p_plugin() -> P2PPlugin | None:
    """Register and return the installed P2P plugin, if present."""
    try:
        plugin = importlib.import_module("dinkster_p2p")
    except ModuleNotFoundError as error:
        if error.name != "dinkster_p2p":
            raise
        return None
    register_p2p_plugin(
        P2PPluginRegistration(
            default_settings=lambda: _enabled_defaults(plugin),
            normalize_settings=plugin.normalize_p2p_settings,
            lan_interfaces=plugin.lan_interfaces,
        )
    )
    lan_p2p = importlib.import_module("dinkster.lan_p2p")
    p2p_api = importlib.import_module("dinkster.p2p_api")
    return P2PPlugin(lan_p2p.LanP2PController, p2p_api.add_p2p_routes)
