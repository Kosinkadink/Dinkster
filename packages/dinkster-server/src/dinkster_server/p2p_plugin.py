"""Registration seam for the optional P2P runtime plugin."""

from __future__ import annotations

from dinkster_assets import p2p_plugin
from dinkster_assets.p2p_settings import P2PSettingsError as P2PSettingsError
from dinkster_assets.p2p_settings import default_p2p_settings as _default_p2p_settings
from dinkster_assets.p2p_settings import normalize_p2p_settings as _normalize_p2p_settings


def default_p2p_settings() -> dict[str, object]:
    registration = p2p_plugin()
    return registration.default_settings() if registration is not None else _default_p2p_settings()


def normalize_p2p_settings(value: object) -> dict[str, object]:
    registration = p2p_plugin()
    if registration is None:
        return _normalize_p2p_settings(value)
    try:
        return registration.normalize_settings(value)
    except ValueError as error:
        raise P2PSettingsError(str(error)) from error
