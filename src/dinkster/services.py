"""Engine-owned hosted account and remote-node service addresses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Services:
    name: str
    identity_base: str
    catalog_base: str
    gateway_base: str
    gateway_audience: str


_HOSTED = {
    "production": Services(
        name="production",
        identity_base="https://id.dinkster.org",
        catalog_base="https://gateway.dinkster.org",
        gateway_base="https://gateway.dinkster.org",
        gateway_audience="gateway",
    ),
}


def resolve_services(
    name: str = "production",
    *,
    catalog_base: str | None = None,
    gateway_base: str | None = None,
) -> Services:
    """Resolve one hosted selection; explicit URL overrides are custom services.

    An explicit empty catalog disables discovery. A catalog override also supplies
    the job endpoint unless a separate gateway override is given.
    """
    try:
        hosted = _HOSTED[name]
    except KeyError:
        raise ValueError(f"unknown services {name!r}; allowed: {', '.join(_HOSTED)}") from None
    if catalog_base is None and gateway_base is None:
        return hosted
    catalog = hosted.catalog_base if catalog_base is None else catalog_base
    return Services(
        name="custom",
        identity_base=hosted.identity_base,
        catalog_base=catalog,
        gateway_base=gateway_base or catalog,
        gateway_audience=hosted.gateway_audience,
    )
