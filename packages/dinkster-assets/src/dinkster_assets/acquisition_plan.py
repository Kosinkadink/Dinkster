"""Pure provider-to-managed-destination acquisition planning.

The records in this module are trusted policy facts, not provider catalog
metadata and not caller-supplied URL leads. Planning only selects authority;
it never fetches, opens, writes, publishes, or reports transfer progress.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeVar, cast
from urllib.parse import urlsplit

from .catalog import validate_virtual_path
from .identity import AssetError, require_digest
from .kind import require_asset_kind
from .resolution import ProviderMirror, SourceIdentity

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_REASON_LIMIT = 256
_OPAQUE_ID_LIMIT = 256
_T = TypeVar("_T")

SourceState = Literal["available", "unavailable"]
AcquisitionPlanStatus = Literal[
    "credential-required",
    "license-required",
    "cost-required",
    "policy-override-required",
    "unauthorized",
    "source-unavailable",
    "no-compatible-destination",
    "ambiguous",
    "ready",
]


def _require_type(value: object, expected: type[_T], message: str) -> _T:
    if not isinstance(value, expected):
        raise AssetError(message)
    return value


def _typed_rows(values: object, row_type: type[_T], field_name: str) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)):
        raise AssetError(f"{field_name} must be a collection")
    try:
        rows = tuple(cast(Iterable[object], values))
    except TypeError as exc:
        raise AssetError(f"{field_name} must be a collection") from exc
    if any(not isinstance(row, row_type) for row in rows):
        raise AssetError(f"{field_name} contain an invalid row")
    return cast("tuple[_T, ...]", rows)


def _stable_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetError(f"{field_name} must be a non-empty string")
    cleaned = value.strip().casefold()
    if _ID_RE.fullmatch(cleaned) is None:
        raise AssetError(f"{field_name} must contain only lowercase id characters")
    return cleaned


def _stable_ids(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise AssetError(f"{field_name} must be a collection of stable ids")
    return frozenset(_stable_id(item, field_name) for item in cast(Iterable[object], value))


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


def _opaque_ids(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise AssetError(f"{field_name} must be a collection of opaque ids")
    return frozenset(_opaque_id(item, field_name) for item in cast(Iterable[object], value))


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


def _availability_fact(value: object, reason: str, field_name: str) -> tuple[bool, str]:
    if not isinstance(value, bool):
        raise AssetError(f"{field_name} must be a boolean")
    bounded = _reason(reason, f"{field_name} reason")
    if value and bounded:
        raise AssetError(f"{field_name} fact cannot carry a refusal reason")
    if not value and not bounded:
        raise AssetError(f"{field_name} refusal requires a reason")
    return value, bounded


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


@dataclass(frozen=True)
class DestinationIdentity:
    mount_id: str
    virtual_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mount_id", _stable_id(self.mount_id, "mount id"))
        virtual_path = _require_type(self.virtual_path, str, "virtual path must be a string")
        object.__setattr__(self, "virtual_path", validate_virtual_path(virtual_path))


@dataclass(frozen=True)
class ManagedDestination:
    identity: DestinationIdentity
    priority: int
    asset_kind: str
    ready: bool
    readwrite: bool
    compatible: bool
    ready_reason: str = ""
    readwrite_reason: str = ""
    compatibility_reason: str = ""

    def __post_init__(self) -> None:
        _require_type(
            self.identity,
            DestinationIdentity,
            "destination identity must be a DestinationIdentity",
        )
        object.__setattr__(self, "priority", _integer(self.priority, "destination priority"))
        object.__setattr__(self, "asset_kind", _asset_kind(self.asset_kind))
        for value_name, reason_name in (
            ("ready", "ready_reason"),
            ("readwrite", "readwrite_reason"),
            ("compatible", "compatibility_reason"),
        ):
            value, reason = _availability_fact(
                getattr(self, value_name), getattr(self, reason_name), value_name
            )
            object.__setattr__(self, value_name, value)
            object.__setattr__(self, reason_name, reason)


@dataclass(frozen=True)
class AcquisitionRequest:
    digest: str
    asset_kind: str
    authorized_providers: frozenset[str]
    selected_source: SourceIdentity | None = None
    selected_destination: DestinationIdentity | None = None
    credential_grants: frozenset[str] = frozenset()
    license_grants: frozenset[str] = frozenset()
    cost_grants: frozenset[str] = frozenset()
    policy_override_grants: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _digest(self.digest))
        object.__setattr__(self, "asset_kind", _asset_kind(self.asset_kind))
        object.__setattr__(
            self,
            "authorized_providers",
            _stable_ids(self.authorized_providers, "authorized providers"),
        )
        if self.selected_source is not None:
            _require_type(
                self.selected_source,
                SourceIdentity,
                "selected source must be a SourceIdentity",
            )
        if self.selected_destination is not None:
            _require_type(
                self.selected_destination,
                DestinationIdentity,
                "selected destination must be a DestinationIdentity",
            )
        for field_name in (
            "credential_grants",
            "license_grants",
            "cost_grants",
            "policy_override_grants",
        ):
            object.__setattr__(self, field_name, _opaque_ids(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class SourceConsideration:
    candidate: AcquisitionSource
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        _require_type(self.candidate, AcquisitionSource, "invalid source consideration")
        _require_type(self.eligible, bool, "invalid source consideration")
        reason = _reason(self.reason, "source consideration reason")
        if self.eligible != (reason == "eligible"):
            raise AssetError("source consideration eligibility and reason disagree")
        if not self.eligible and not reason:
            raise AssetError("ineligible source consideration requires a reason")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class DestinationConsideration:
    candidate: ManagedDestination
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        _require_type(self.candidate, ManagedDestination, "invalid destination consideration")
        _require_type(self.eligible, bool, "invalid destination consideration")
        reason = _reason(self.reason, "destination consideration reason")
        if self.eligible != (reason == "eligible"):
            raise AssetError("destination consideration eligibility and reason disagree")
        if not self.eligible and not reason:
            raise AssetError("ineligible destination consideration requires a reason")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class AcquisitionPlan:
    status: AcquisitionPlanStatus
    reason: str
    sources: tuple[SourceConsideration, ...]
    destinations: tuple[DestinationConsideration, ...]
    selected_source: AcquisitionSource | None = None
    selected_destination: ManagedDestination | None = None
    required_grant_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (
            "credential-required",
            "license-required",
            "cost-required",
            "policy-override-required",
            "unauthorized",
            "source-unavailable",
            "no-compatible-destination",
            "ambiguous",
            "ready",
        ):
            raise AssetError(f"unknown acquisition plan status: {self.status!r}")
        reason = _reason(self.reason, "plan reason")
        if not reason or reason != reason.strip() or _ID_RE.fullmatch(reason) is None:
            raise AssetError("plan reason must be a non-empty machine reason")
        object.__setattr__(self, "reason", reason)
        source_rows = _typed_rows(self.sources, SourceConsideration, "plan sources")
        destination_rows = _typed_rows(
            self.destinations, DestinationConsideration, "plan destinations"
        )
        object.__setattr__(self, "sources", source_rows)
        object.__setattr__(self, "destinations", destination_rows)
        if self.selected_source is not None:
            _require_type(
                self.selected_source,
                AcquisitionSource,
                "selected source must be an AcquisitionSource",
            )
        if self.selected_destination is not None:
            _require_type(
                self.selected_destination,
                ManagedDestination,
                "selected destination must be a ManagedDestination",
            )
        eligible_sources = frozenset(row.candidate for row in source_rows if row.eligible)
        eligible_destinations = frozenset(row.candidate for row in destination_rows if row.eligible)
        if self.selected_source is not None and self.selected_source not in eligible_sources:
            raise AssetError("selected source must be an eligible considered source")
        if (
            self.selected_destination is not None
            and self.selected_destination not in eligible_destinations
        ):
            raise AssetError("selected destination must be an eligible considered destination")
        if self.required_grant_id is not None:
            object.__setattr__(
                self,
                "required_grant_id",
                _opaque_id(self.required_grant_id, "required grant id"),
            )
        grant_statuses = (
            "credential-required",
            "license-required",
            "cost-required",
            "policy-override-required",
        )
        if self.status == "ready":
            if self.selected_source is None or self.selected_destination is None:
                raise AssetError("ready plan requires selected source and destination")
            if self.required_grant_id is not None:
                raise AssetError("ready plan cannot require a grant")
        elif self.selected_destination is not None:
            raise AssetError("only a ready plan may select a destination")
        if self.status in grant_statuses:
            if self.selected_source is None or self.required_grant_id is None:
                raise AssetError("grant-required plan requires selected source and grant id")
        elif self.required_grant_id is not None:
            raise AssetError("non-grant plan cannot require a grant id")
        if self.status in ("unauthorized", "source-unavailable"):
            if self.selected_source is not None:
                raise AssetError(f"{self.status} plan cannot select a source")
        if self.status == "no-compatible-destination" and self.selected_source is None:
            raise AssetError("destination refusal requires a selected source")


def _source_sort_key(source: AcquisitionSource) -> tuple[object, ...]:
    return (
        source.provider_priority,
        source.source.provider_id,
        source.source.source_id,
        source.digest,
        source.asset_kind,
        source.authorized_locator,
        source.expected_size,
        source.media_type,
        source.state,
        source.unavailable_reason,
        source.credential_requirement_id or "",
        source.license_requirement_id or "",
        source.cost_requirement_id or "",
        source.policy_override_requirement_id or "",
    )


def _destination_sort_key(destination: ManagedDestination) -> tuple[object, ...]:
    return (
        destination.priority,
        destination.identity.mount_id,
        destination.identity.virtual_path,
        destination.asset_kind,
        destination.ready,
        destination.readwrite,
        destination.compatible,
        destination.ready_reason,
        destination.readwrite_reason,
        destination.compatibility_reason,
    )


def _source_reason(
    request: AcquisitionRequest,
    source: AcquisitionSource,
    mirrors: tuple[ProviderMirror, ...],
    conflicting: frozenset[SourceIdentity],
) -> str:
    if source.source.provider_id not in request.authorized_providers:
        return "unauthorized-provider"
    if source.state != "available":
        return "source-unavailable"
    if source.digest != request.digest:
        return "digest-mismatch"
    if source.asset_kind != request.asset_kind:
        return "kind-mismatch"
    matching_identity = tuple(row for row in mirrors if row.source == source.source)
    if not matching_identity:
        return "mirror-missing"
    matching_digest = tuple(row for row in matching_identity if row.digest == source.digest)
    if not matching_digest:
        return "mirror-digest-mismatch"
    if not any(row.state == "available" for row in matching_digest):
        return "mirror-unavailable"
    if request.selected_source is not None and source.source != request.selected_source:
        return "not-explicit-source"
    if source.source in conflicting:
        return "conflicting-source-facts"
    return "eligible"


def _destination_reason(
    request: AcquisitionRequest,
    destination: ManagedDestination,
    conflicting: frozenset[DestinationIdentity],
) -> str:
    if destination.asset_kind != request.asset_kind:
        return "kind-mismatch"
    if not destination.ready:
        return "not-ready"
    if not destination.readwrite:
        return "read-only"
    if not destination.compatible:
        return "incompatible"
    if (
        request.selected_destination is not None
        and destination.identity != request.selected_destination
    ):
        return "not-explicit-destination"
    if destination.identity in conflicting:
        return "conflicting-destination-facts"
    return "eligible"


def plan_acquisition(
    request: AcquisitionRequest,
    sources: Iterable[AcquisitionSource],
    mirrors: Iterable[ProviderMirror],
    destinations: Iterable[ManagedDestination],
) -> AcquisitionPlan:
    """Select trusted source and destination facts without performing I/O."""

    _require_type(request, AcquisitionRequest, "request must be an AcquisitionRequest")
    source_rows = _typed_rows(sources, AcquisitionSource, "sources")
    mirror_rows = _typed_rows(mirrors, ProviderMirror, "mirrors")
    destination_rows = _typed_rows(destinations, ManagedDestination, "destinations")
    for mirror in mirror_rows:
        _require_type(mirror.source, SourceIdentity, "mirror source must be a SourceIdentity")

    unique_sources = tuple(sorted(set(source_rows), key=_source_sort_key))
    sources_by_identity: dict[SourceIdentity, set[AcquisitionSource]] = {}
    for source in unique_sources:
        sources_by_identity.setdefault(source.source, set()).add(source)
    conflicting_sources = frozenset(
        identity for identity, rows in sources_by_identity.items() if len(rows) > 1
    )
    source_considerations = tuple(
        SourceConsideration(
            source,
            (reason := _source_reason(request, source, mirror_rows, conflicting_sources))
            == "eligible",
            reason,
        )
        for source in unique_sources
    )

    unique_destinations = tuple(sorted(set(destination_rows), key=_destination_sort_key))
    destinations_by_identity: dict[DestinationIdentity, set[ManagedDestination]] = {}
    for destination in unique_destinations:
        destinations_by_identity.setdefault(destination.identity, set()).add(destination)
    conflicting_destinations = frozenset(
        identity for identity, rows in destinations_by_identity.items() if len(rows) > 1
    )
    destination_considerations = tuple(
        DestinationConsideration(
            destination,
            (reason := _destination_reason(request, destination, conflicting_destinations))
            == "eligible",
            reason,
        )
        for destination in unique_destinations
    )

    eligible_sources = tuple(row.candidate for row in source_considerations if row.eligible)
    if not eligible_sources:
        if any(row.reason == "conflicting-source-facts" for row in source_considerations):
            status: AcquisitionPlanStatus = "ambiguous"
            reason = "conflicting-source-facts"
        elif request.selected_source is not None and (
            request.selected_source.provider_id not in request.authorized_providers
        ):
            status = "unauthorized"
            reason = "explicit-provider-unauthorized"
        elif request.selected_source is None and any(
            row.candidate.digest == request.digest
            and row.candidate.asset_kind == request.asset_kind
            and row.reason == "unauthorized-provider"
            for row in source_considerations
        ):
            status = "unauthorized"
            reason = "provider-unauthorized"
        else:
            status = "source-unavailable"
            reason = "no-eligible-source"
        return AcquisitionPlan(
            status,
            reason,
            source_considerations,
            destination_considerations,
        )

    selected_source = eligible_sources[0]
    grant_checks: tuple[tuple[AcquisitionPlanStatus, str | None, frozenset[str]], ...] = (
        (
            "credential-required",
            selected_source.credential_requirement_id,
            request.credential_grants,
        ),
        (
            "license-required",
            selected_source.license_requirement_id,
            request.license_grants,
        ),
        ("cost-required", selected_source.cost_requirement_id, request.cost_grants),
        (
            "policy-override-required",
            selected_source.policy_override_requirement_id,
            request.policy_override_grants,
        ),
    )
    for status, requirement_id, grants in grant_checks:
        if requirement_id is not None and requirement_id not in grants:
            return AcquisitionPlan(
                status,
                status,
                source_considerations,
                destination_considerations,
                selected_source=selected_source,
                required_grant_id=requirement_id,
            )

    eligible_destinations = tuple(
        row.candidate for row in destination_considerations if row.eligible
    )
    if not eligible_destinations:
        relevant_conflict = any(
            row.reason == "conflicting-destination-facts" for row in destination_considerations
        )
        return AcquisitionPlan(
            "ambiguous" if relevant_conflict else "no-compatible-destination",
            ("conflicting-destination-facts" if relevant_conflict else "no-compatible-destination"),
            source_considerations,
            destination_considerations,
            selected_source=selected_source,
        )

    return AcquisitionPlan(
        "ready",
        "ready",
        source_considerations,
        destination_considerations,
        selected_source,
        eligible_destinations[0],
    )
