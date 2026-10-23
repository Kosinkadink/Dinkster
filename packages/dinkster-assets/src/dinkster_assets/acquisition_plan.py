"""Trusted acquisition source facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar
from urllib.parse import urlsplit

from .identity import AssetError, require_digest
from .kind import require_asset_kind
from .resolution import SourceIdentity

_REASON_LIMIT = 256
_OPAQUE_ID_LIMIT = 256
_T = TypeVar("_T")

SourceState = Literal["available", "unavailable"]


def _require_type(value: object, expected: type[_T], message: str) -> _T:
    if not isinstance(value, expected):
        raise AssetError(message)
    return value


def _digest(value: object, field_name: str = "digest") -> str:
    if not isinstance(value, str):
        raise AssetError(f"{field_name} must be a canonical asset digest string")
    return require_digest(value)


def _asset_kind(value: object) -> str:
    if not isinstance(value, str):
        raise AssetError("asset kind must be a string")
    return require_asset_kind(value)


def _opaque_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetError(f"{field_name} must be a non-empty opaque id")
    if len(value) > _OPAQUE_ID_LIMIT or any(ord(char) < 32 for char in value):
        raise AssetError(f"{field_name} must be a bounded printable opaque id")
    return value


def _reason(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AssetError(f"{field_name} must be a string")
    if len(value) > _REASON_LIMIT:
        raise AssetError(f"{field_name} exceeds {_REASON_LIMIT} characters")
    return value


def _integer(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssetError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise AssetError(f"{field_name} must be an integer >= {minimum}")
    return value


def _http_locator(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetError("authorized locator must be a non-empty http(s) URL")
    if len(value) > 2048 or any(char.isspace() or ord(char) < 32 for char in value):
        raise AssetError("authorized locator must be a bounded URL without raw whitespace")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError as exc:
        raise AssetError("authorized locator is malformed") from exc
    if parsed.scheme not in ("http", "https") or not parsed.netloc or hostname is None:
        raise AssetError("authorized locator must be an absolute http(s) URL")
    if username is not None or password is not None:
        raise AssetError("authorized locator must not contain userinfo credentials")
    if parsed.query or parsed.fragment:
        raise AssetError("authorized locator must not contain a query or fragment")
    if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
        raise AssetError("authorized locator contains an invalid port")
    return value


@dataclass(frozen=True)
class AcquisitionSource:
    """Trusted source authority, separate from advisory provider metadata.

    The locator is an absolute HTTP(S) URL with a hostname, optional valid
    port, and optional path. Raw whitespace, userinfo, query, and fragment are
    forbidden so credentials and bearer-style signed URL values cannot enter
    the durable source fact.
    """

    source: SourceIdentity
    digest: str
    asset_kind: str
    state: SourceState
    authorized_locator: str
    expected_size: int
    media_type: str
    provider_priority: int = 0
    unavailable_reason: str = ""
    credential_requirement_id: str | None = None
    license_requirement_id: str | None = None
    cost_requirement_id: str | None = None
    policy_override_requirement_id: str | None = None

    def __post_init__(self) -> None:
        _require_type(self.source, SourceIdentity, "source must be a SourceIdentity")
        object.__setattr__(self, "digest", _digest(self.digest))
        object.__setattr__(self, "asset_kind", _asset_kind(self.asset_kind))
        if self.state not in ("available", "unavailable"):
            raise AssetError(f"unknown acquisition source state: {self.state!r}")
        object.__setattr__(self, "authorized_locator", _http_locator(self.authorized_locator))
        object.__setattr__(
            self,
            "expected_size",
            _integer(self.expected_size, "expected size", minimum=0),
        )
        media_type = _require_type(self.media_type, str, "media type must be a non-empty string")
        if not media_type.strip():
            raise AssetError("media type must be a non-empty string")
        object.__setattr__(self, "media_type", media_type.strip())
        object.__setattr__(
            self,
            "provider_priority",
            _integer(self.provider_priority, "provider priority"),
        )
        reason = _reason(self.unavailable_reason, "unavailable reason")
        if self.state == "available" and reason:
            raise AssetError("available acquisition source cannot carry an unavailable reason")
        if self.state == "unavailable" and not reason:
            raise AssetError("unavailable acquisition source requires a reason")
        object.__setattr__(self, "unavailable_reason", reason)
        for field_name in (
            "credential_requirement_id",
            "license_requirement_id",
            "cost_requirement_id",
            "policy_override_requirement_id",
        ):
            value: object = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _opaque_id(value, field_name))
