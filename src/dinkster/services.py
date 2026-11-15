"""Engine-owned hosted account and remote-node service addresses."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


def _service_base(value: str) -> str:
    if not value:
        return ""
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ValueError("service addresses must be valid HTTP(S) URLs") from None
    if (
        not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.scheme not in {"https", "http"}
        or (url.scheme == "http" and url.hostname not in {"127.0.0.1", "::1", "localhost"})
    ):
        raise ValueError("service addresses require HTTPS or loopback HTTP, without credentials")
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    if port is not None and port != (443 if url.scheme == "https" else 80):
        host += f":{port}"
    return f"{url.scheme}://{host}{url.path.rstrip('/')}"


@dataclass(frozen=True, slots=True)
class Services:
    name: str
    identity_base: str
    catalog_base: str
    gateway_base: str
    gateway_audience: str

    def __post_init__(self) -> None:
        for name in ("identity_base", "catalog_base", "gateway_base"):
            object.__setattr__(self, name, _service_base(getattr(self, name)))


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
    identity_base: str | None = None,
    catalog_base: str | None = None,
    gateway_base: str | None = None,
    gateway_audience: str | None = None,
) -> Services:
    """Resolve one hosted selection; explicit URL overrides are custom services.

    An explicit empty catalog disables discovery. A catalog override also supplies
    the job endpoint unless a separate gateway override is given.
    """
    try:
        hosted = _HOSTED[name]
    except KeyError:
        raise ValueError(f"unknown services {name!r}; allowed: {', '.join(_HOSTED)}") from None
    if all(
        value is None for value in (identity_base, catalog_base, gateway_base, gateway_audience)
    ):
        return hosted
    catalog = hosted.catalog_base if catalog_base is None else catalog_base
    return Services(
        name="custom",
        identity_base=hosted.identity_base if identity_base is None else identity_base,
        catalog_base=catalog,
        gateway_base=gateway_base or catalog,
        gateway_audience=hosted.gateway_audience if gateway_audience is None else gateway_audience,
    )
