"""Network-free trusted-provider facts, grant mapping, and P2P counters."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from .identity import AssetError, require_digest
from .p2p_descriptor import P2PDescriptorV1, validate_p2p_descriptor
from .p2p_grants import (
    P2P_GRANT_VERSION,
    P2P_REMOTE_GRANT_MAX_SECONDS,
    P2PGrantSnapshot,
    PublicSwarmDeclarationV1,
    PublicSwarmGrantV1,
    public_swarm_grant_id,
)
from .transport import TransportCandidate

GLOBAL_PROVIDER_SNAPSHOT_VERSION = 1
MAX_PROVIDER_ARTIFACTS = 100_000
MAX_PROVIDER_LOCATIONS = 64
MAX_PROVIDER_TRACKERS = 16

NatOutcome = Literal["reachable", "outbound-only", "unreachable"]

_INT64_MAX = 2**63 - 1


class GlobalP2PPolicyError(AssetError):
    """Trusted provider or internet P2P policy facts are invalid."""


def _object(
    value: object,
    field: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GlobalP2PPolicyError(f"{field} must be an object")
    raw = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        raise GlobalP2PPolicyError(f"{field} keys must be strings")
    result = cast("Mapping[str, object]", value)
    missing = required - set(result)
    unknown = set(result) - required - optional
    if missing:
        raise GlobalP2PPolicyError(f"{field} is missing fields: {sorted(missing)}")
    if unknown:
        raise GlobalP2PPolicyError(f"{field} has unknown fields: {sorted(unknown)}")
    return result


def _array(value: object, field: str, *, maximum: int) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise GlobalP2PPolicyError(f"{field} must be an array")
    rows = tuple(cast("Sequence[object]", value))
    if len(rows) > maximum:
        raise GlobalP2PPolicyError(f"{field} accepts at most {maximum} items")
    return rows


def _string(value: object, field: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GlobalP2PPolicyError(f"{field} must be a non-empty trimmed string")
    if len(value) > maximum or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise GlobalP2PPolicyError(f"{field} must be bounded and contain no controls")
    return value


def _license(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        raise GlobalP2PPolicyError(f"{field} must be a bounded string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise GlobalP2PPolicyError(f"{field} must not contain controls")
    return value


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= _INT64_MAX:
        qualifier = "positive" if positive else "non-negative"
        raise GlobalP2PPolicyError(f"{field} must be a {qualifier} 64-bit integer")
    return value


def _number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlobalP2PPolicyError(f"{field} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise GlobalP2PPolicyError(f"{field} must be a finite number >= {minimum}") from exc
    if not math.isfinite(result) or result < minimum:
        raise GlobalP2PPolicyError(f"{field} must be a finite number >= {minimum}")
    return result


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise GlobalP2PPolicyError(f"{field} must be a boolean")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise GlobalP2PPolicyError(f"{field} must be a canonical BLAKE3 digest")
    try:
        return require_digest(value)
    except AssetError as exc:
        raise GlobalP2PPolicyError(f"{field}: {exc}") from exc


def _descriptor(value: object, field: str, *, digest: str, size: int) -> P2PDescriptorV1:
    if isinstance(value, P2PDescriptorV1):
        parsed = value
    else:
        wire = _object(
            value,
            field,
            required=frozenset({"protocol", "infoHash", "fileRoot", "pieceLength"}),
        )
        try:
            parsed = P2PDescriptorV1.from_wire(wire)
        except AssetError as exc:
            raise GlobalP2PPolicyError(f"{field}: {exc}") from exc
    try:
        return validate_p2p_descriptor(parsed, asset_digest=digest, size=size)
    except AssetError as exc:
        raise GlobalP2PPolicyError(f"{field}: {exc}") from exc


def _public_https_url(value: object, field: str) -> str:
    url = _string(value, field, maximum=4096)
    if any(character.isspace() for character in url):
        raise GlobalP2PPolicyError(f"{field} must not contain whitespace")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise GlobalP2PPolicyError(f"{field} is malformed") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise GlobalP2PPolicyError(
            f"{field} must be an HTTPS URL without credentials, query, or fragment"
        )
    return url


def _tracker_url(value: object, field: str) -> str:
    url = _string(value, field, maximum=4096)
    if any(character.isspace() for character in url):
        raise GlobalP2PPolicyError(f"{field} must not contain whitespace")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise GlobalP2PPolicyError(f"{field} is malformed") from exc
    if (
        parsed.scheme not in {"https", "udp"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise GlobalP2PPolicyError(
            f"{field} must be an HTTPS or UDP tracker URL without credentials or query data"
        )
    return url


@dataclass(frozen=True, slots=True)
class ProviderLocationV1:
    url: str
    eligible: bool
    credential_free: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _public_https_url(self.url, "location.url"))
        _boolean(self.eligible, "location.eligible")
        _boolean(self.credential_free, "location.credentialFree")
        if self.eligible and not self.credential_free:
            raise GlobalP2PPolicyError("an eligible location must be credential-free")

    @classmethod
    def from_wire(cls, value: object, field: str = "location") -> ProviderLocationV1:
        wire = _object(
            value,
            field,
            required=frozenset({"url", "eligible", "credentialFree"}),
        )
        return cls(
            _public_https_url(wire["url"], f"{field}.url"),
            _boolean(wire["eligible"], f"{field}.eligible"),
            _boolean(wire["credentialFree"], f"{field}.credentialFree"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "url": self.url,
            "eligible": self.eligible,
            "credentialFree": self.credential_free,
        }


@dataclass(frozen=True, slots=True)
class ProviderArtifactP2PV1:
    source_id: str
    digest: str
    size_bytes: int
    descriptor: P2PDescriptorV1
    license: str
    format_safe: bool
    locations: tuple[ProviderLocationV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _string(self.source_id, "artifact.sourceId"))
        digest = _digest(self.digest, "artifact.digest")
        size = _integer(self.size_bytes, "artifact.sizeBytes", positive=True)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(
            self,
            "descriptor",
            _descriptor(self.descriptor, "artifact.p2p", digest=digest, size=size),
        )
        object.__setattr__(
            self,
            "license",
            _license(self.license, "artifact.license"),
        )
        _boolean(self.format_safe, "artifact.formatSafe")
        raw_locations = cast(object, self.locations)
        if not isinstance(raw_locations, (tuple, list)):
            raise GlobalP2PPolicyError("artifact.locations must be a collection")
        locations = tuple(cast("tuple[object, ...] | list[object]", raw_locations))
        if len(locations) > MAX_PROVIDER_LOCATIONS or any(
            not isinstance(row, ProviderLocationV1) for row in locations
        ):
            raise GlobalP2PPolicyError("artifact.locations contains invalid locations")
        typed_locations = cast("tuple[ProviderLocationV1, ...]", locations)
        if len({row.url for row in typed_locations}) != len(typed_locations):
            raise GlobalP2PPolicyError("artifact.locations must not contain duplicate URLs")
        object.__setattr__(self, "locations", typed_locations)

    @classmethod
    def from_wire(cls, value: object, field: str = "artifact") -> ProviderArtifactP2PV1:
        wire = _object(
            value,
            field,
            required=frozenset(
                {
                    "sourceId",
                    "digest",
                    "sizeBytes",
                    "p2p",
                    "license",
                    "formatSafe",
                    "locations",
                }
            ),
        )
        digest = _digest(wire["digest"], f"{field}.digest")
        size = _integer(wire["sizeBytes"], f"{field}.sizeBytes", positive=True)
        return cls(
            source_id=_string(wire["sourceId"], f"{field}.sourceId"),
            digest=digest,
            size_bytes=size,
            descriptor=_descriptor(wire["p2p"], f"{field}.p2p", digest=digest, size=size),
            license=_license(wire["license"], f"{field}.license"),
            format_safe=_boolean(wire["formatSafe"], f"{field}.formatSafe"),
            locations=tuple(
                ProviderLocationV1.from_wire(row, f"{field}.locations[{position}]")
                for position, row in enumerate(
                    _array(
                        wire["locations"],
                        f"{field}.locations",
                        maximum=MAX_PROVIDER_LOCATIONS,
                    )
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderP2PEnumerationV1:
    grant_id: str
    digest: str
    size_bytes: int
    descriptor: P2PDescriptorV1
    expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "grant_id", _string(self.grant_id, "p2pArtifact.grantId"))
        digest = _digest(self.digest, "p2pArtifact.digest")
        size = _integer(self.size_bytes, "p2pArtifact.sizeBytes", positive=True)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(
            self,
            "descriptor",
            _descriptor(self.descriptor, "p2pArtifact.p2p", digest=digest, size=size),
        )
        object.__setattr__(self, "expires_at", _number(self.expires_at, "p2pArtifact.expiresAt"))

    @classmethod
    def from_wire(cls, value: object, field: str = "p2pArtifact") -> ProviderP2PEnumerationV1:
        wire = _object(
            value,
            field,
            required=frozenset({"grantId", "digest", "sizeBytes", "p2p", "expiresAt"}),
        )
        digest = _digest(wire["digest"], f"{field}.digest")
        size = _integer(wire["sizeBytes"], f"{field}.sizeBytes", positive=True)
        return cls(
            grant_id=_string(wire["grantId"], f"{field}.grantId"),
            digest=digest,
            size_bytes=size,
            descriptor=_descriptor(wire["p2p"], f"{field}.p2p", digest=digest, size=size),
            expires_at=_number(wire["expiresAt"], f"{field}.expiresAt"),
        )


@dataclass(frozen=True, slots=True)
class ProviderP2PTombstoneV1:
    digest: str
    observed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _digest(self.digest, "tombstone.digest"))
        object.__setattr__(self, "observed_at", _number(self.observed_at, "tombstone.observedAt"))

    @classmethod
    def from_wire(cls, value: object, field: str = "tombstone") -> ProviderP2PTombstoneV1:
        wire = _object(
            value,
            field,
            required=frozenset({"digest", "observedAt"}),
        )
        return cls(
            _digest(wire["digest"], f"{field}.digest"),
            _number(wire["observedAt"], f"{field}.observedAt"),
        )


@dataclass(frozen=True, slots=True)
class ProviderP2PSnapshotV1:
    provider_id: str
    source_revision: str
    refreshed_at: float
    artifacts: tuple[ProviderArtifactP2PV1, ...]
    p2p_artifacts: tuple[ProviderP2PEnumerationV1, ...]
    p2p_trackers: tuple[str, ...] = ()
    tombstones: tuple[ProviderP2PTombstoneV1, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _string(self.provider_id, "providerId"))
        object.__setattr__(self, "source_revision", _string(self.source_revision, "sourceRevision"))
        object.__setattr__(self, "refreshed_at", _number(self.refreshed_at, "refreshedAt"))
        artifacts = tuple(cast("tuple[object, ...]", self.artifacts))
        enumeration = tuple(cast("tuple[object, ...]", self.p2p_artifacts))
        tombstones = tuple(cast("tuple[object, ...]", self.tombstones))
        if len(artifacts) > MAX_PROVIDER_ARTIFACTS or any(
            not isinstance(row, ProviderArtifactP2PV1) for row in artifacts
        ):
            raise GlobalP2PPolicyError("artifacts contains invalid rows")
        if len(enumeration) > MAX_PROVIDER_ARTIFACTS or any(
            not isinstance(row, ProviderP2PEnumerationV1) for row in enumeration
        ):
            raise GlobalP2PPolicyError("p2pArtifacts contains invalid rows")
        if len(tombstones) > MAX_PROVIDER_ARTIFACTS or any(
            not isinstance(row, ProviderP2PTombstoneV1) for row in tombstones
        ):
            raise GlobalP2PPolicyError("tombstones contains invalid rows")
        typed_artifacts = cast("tuple[ProviderArtifactP2PV1, ...]", artifacts)
        typed_enumeration = cast("tuple[ProviderP2PEnumerationV1, ...]", enumeration)
        typed_tombstones = cast("tuple[ProviderP2PTombstoneV1, ...]", tombstones)
        for field, digests in (
            ("artifacts", [row.digest for row in typed_artifacts]),
            ("p2pArtifacts", [row.digest for row in typed_enumeration]),
            ("tombstones", [row.digest for row in typed_tombstones]),
        ):
            if len(digests) != len(set(digests)):
                raise GlobalP2PPolicyError(f"{field} must not contain duplicate digests")
        trackers = tuple(
            _tracker_url(value, f"p2pTrackers[{position}]")
            for position, value in enumerate(self.p2p_trackers)
        )
        if len(trackers) > MAX_PROVIDER_TRACKERS or len(trackers) != len(set(trackers)):
            raise GlobalP2PPolicyError("p2pTrackers must be unique and within the size limit")
        object.__setattr__(self, "artifacts", typed_artifacts)
        object.__setattr__(self, "p2p_artifacts", typed_enumeration)
        object.__setattr__(self, "p2p_trackers", trackers)
        object.__setattr__(self, "tombstones", typed_tombstones)

    @classmethod
    def from_wire(cls, value: object) -> ProviderP2PSnapshotV1:
        wire = _object(
            value,
            "provider snapshot",
            required=frozenset(
                {
                    "version",
                    "providerId",
                    "sourceRevision",
                    "refreshedAt",
                    "artifacts",
                    "p2pArtifacts",
                    "p2pTrackers",
                    "tombstones",
                }
            ),
        )
        if wire["version"] != GLOBAL_PROVIDER_SNAPSHOT_VERSION or type(wire["version"]) is not int:
            raise GlobalP2PPolicyError(
                f"provider snapshot version must be {GLOBAL_PROVIDER_SNAPSHOT_VERSION}"
            )
        artifacts = _array(wire["artifacts"], "artifacts", maximum=MAX_PROVIDER_ARTIFACTS)
        enumeration = _array(wire["p2pArtifacts"], "p2pArtifacts", maximum=MAX_PROVIDER_ARTIFACTS)
        trackers = _array(wire["p2pTrackers"], "p2pTrackers", maximum=MAX_PROVIDER_TRACKERS)
        tombstones = _array(wire["tombstones"], "tombstones", maximum=MAX_PROVIDER_ARTIFACTS)
        return cls(
            provider_id=_string(wire["providerId"], "providerId"),
            source_revision=_string(wire["sourceRevision"], "sourceRevision"),
            refreshed_at=_number(wire["refreshedAt"], "refreshedAt"),
            artifacts=tuple(
                ProviderArtifactP2PV1.from_wire(row, f"artifacts[{position}]")
                for position, row in enumerate(artifacts)
            ),
            p2p_artifacts=tuple(
                ProviderP2PEnumerationV1.from_wire(row, f"p2pArtifacts[{position}]")
                for position, row in enumerate(enumeration)
            ),
            p2p_trackers=tuple(
                _tracker_url(row, f"p2pTrackers[{position}]")
                for position, row in enumerate(trackers)
            ),
            tombstones=tuple(
                ProviderP2PTombstoneV1.from_wire(row, f"tombstones[{position}]")
                for position, row in enumerate(tombstones)
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderDeclarationDecision:
    digest: str
    eligible: bool
    reason: str
    declaration: PublicSwarmDeclarationV1 | None = None


def provider_declarations(
    snapshot: ProviderP2PSnapshotV1,
    *,
    trusted_provider_ids: frozenset[str],
    now: float,
    digests: frozenset[str] | None = None,
) -> tuple[ProviderDeclarationDecision, ...]:
    """Adapt eligible provider facts into the shared acquisition grant contract."""
    current = _number(now, "now")
    trusted = frozenset(_string(item, "trusted provider id") for item in trusted_provider_ids)
    selected = (
        None
        if digests is None
        else frozenset(_digest(item, "provider declaration digest") for item in digests)
    )
    artifacts = {
        row.digest: row for row in snapshot.artifacts if selected is None or row.digest in selected
    }
    enumeration = {
        row.digest: row
        for row in snapshot.p2p_artifacts
        if selected is None or row.digest in selected
    }
    tombstones = {
        row.digest for row in snapshot.tombstones if selected is None or row.digest in selected
    }
    decisions: list[ProviderDeclarationDecision] = []

    for digest in sorted(set(artifacts) | set(enumeration)):
        artifact = artifacts.get(digest)
        entry = enumeration.get(digest)
        reason = "eligible"
        declaration: PublicSwarmDeclarationV1 | None = None
        if snapshot.provider_id not in trusted:
            reason = "untrusted-provider"
        elif entry is None:
            reason = "not-enumerated"
        elif artifact is None:
            reason = "artifact-missing"
        elif digest in tombstones:
            reason = "tombstoned"
        elif snapshot.refreshed_at > current:
            reason = "future-refresh"
        else:
            lease_expires_at = min(
                entry.expires_at,
                snapshot.refreshed_at + P2P_REMOTE_GRANT_MAX_SECONDS,
                current + P2P_REMOTE_GRANT_MAX_SECONDS,
            )
            if current >= lease_expires_at:
                reason = "expired"
            elif artifact.size_bytes != entry.size_bytes or artifact.descriptor != entry.descriptor:
                reason = "malformed-enumeration"
            elif not artifact.format_safe:
                reason = "unsafe-format"
            else:
                declaration = PublicSwarmDeclarationV1(
                    version=P2P_GRANT_VERSION,
                    digest=digest,
                    size_bytes=entry.size_bytes,
                    source_type="official-provider",
                    source_id=artifact.source_id,
                    source_revision=snapshot.source_revision,
                    license=artifact.license,
                    descriptor=entry.descriptor,
                    refreshed_at=snapshot.refreshed_at,
                    expires_at=lease_expires_at,
                    evidence_id=entry.grant_id,
                )
        decisions.append(
            ProviderDeclarationDecision(digest, declaration is not None, reason, declaration)
        )
    return tuple(decisions)


def matching_provider_grant(
    declaration: PublicSwarmDeclarationV1,
    grants: P2PGrantSnapshot,
    *,
    now: float,
) -> PublicSwarmGrantV1 | None:
    """Return the current grant that exactly preserves one provider declaration."""
    current = _number(now, "now")
    if not grants.enabled:
        return None
    return next(
        (
            grant
            for grant in grants.public_grants
            if _provider_grant_matches(declaration, grant, current=current)
        ),
        None,
    )


def _provider_grant_matches(
    declaration: PublicSwarmDeclarationV1,
    grant: PublicSwarmGrantV1,
    *,
    current: float,
) -> bool:
    return bool(
        current < grant.expires_at
        and grant.grant_id == public_swarm_grant_id(declaration)
        and grant.digest == declaration.digest
        and grant.source_type == declaration.source_type
        and grant.source_id == declaration.source_id
        and grant.source_revision == declaration.source_revision
        and grant.license == declaration.license
        and grant.descriptor == declaration.descriptor
        and grant.expires_at == declaration.expires_at
    )


def matching_provider_grants(
    snapshot: ProviderP2PSnapshotV1,
    grants: P2PGrantSnapshot,
    *,
    trusted_provider_ids: frozenset[str],
    now: float,
    digests: frozenset[str] | None = None,
) -> tuple[tuple[PublicSwarmDeclarationV1, PublicSwarmGrantV1], ...]:
    """Match one complete provider snapshot to canonical grants."""
    return tuple(
        (declaration, grant)
        for _, declaration, grant in matching_provider_grants_for_snapshots(
            (snapshot,),
            grants,
            trusted_provider_ids=trusted_provider_ids,
            now=now,
            digests=digests,
        )
    )


def matching_provider_grants_for_snapshots(
    snapshots: Sequence[ProviderP2PSnapshotV1],
    grants: P2PGrantSnapshot,
    *,
    trusted_provider_ids: frozenset[str],
    now: float,
    digests: frozenset[str] | None = None,
) -> tuple[tuple[ProviderP2PSnapshotV1, PublicSwarmDeclarationV1, PublicSwarmGrantV1], ...]:
    """Match complete provider snapshots with one shared linear grant index."""
    current = _number(now, "now")
    if not grants.enabled:
        return ()
    by_id: dict[str, PublicSwarmGrantV1] = {}
    duplicate_ids: set[str] = set()
    for grant in grants.public_grants:
        if grant.grant_id in by_id:
            duplicate_ids.add(grant.grant_id)
        else:
            by_id[grant.grant_id] = grant
    matched: list[tuple[ProviderP2PSnapshotV1, PublicSwarmDeclarationV1, PublicSwarmGrantV1]] = []
    for snapshot in snapshots:
        for decision in provider_declarations(
            snapshot,
            trusted_provider_ids=trusted_provider_ids,
            now=current,
            digests=digests,
        ):
            declaration = decision.declaration
            if declaration is None:
                continue
            grant_id = public_swarm_grant_id(declaration)
            grant = by_id.get(grant_id)
            if (
                grant_id not in duplicate_ids
                and grant is not None
                and _provider_grant_matches(declaration, grant, current=current)
            ):
                matched.append((snapshot, declaration, grant))
    return tuple(matched)


def provider_transport_candidates(
    snapshot: ProviderP2PSnapshotV1,
    grants: P2PGrantSnapshot,
    *,
    trusted_provider_ids: frozenset[str],
    now: float,
    global_downloads_allowed: bool,
) -> tuple[TransportCandidate, ...]:
    """Map one provider snapshot into global transport leads."""
    return provider_transport_candidates_for_snapshots(
        (snapshot,),
        grants,
        trusted_provider_ids=trusted_provider_ids,
        now=now,
        global_downloads_allowed=global_downloads_allowed,
    )


def provider_transport_candidates_for_snapshots(
    snapshots: Sequence[ProviderP2PSnapshotV1],
    grants: P2PGrantSnapshot,
    *,
    trusted_provider_ids: frozenset[str],
    now: float,
    global_downloads_allowed: bool,
) -> tuple[TransportCandidate, ...]:
    """Map provider snapshots with one shared grant index."""
    current = _number(now, "now")
    if not grants.enabled or not global_downloads_allowed:
        return ()
    candidates: list[TransportCandidate] = []
    for _, declaration, matching in matching_provider_grants_for_snapshots(
        snapshots,
        grants,
        trusted_provider_ids=trusted_provider_ids,
        now=current,
    ):
        candidates.append(
            TransportCandidate(
                kind="global-p2p",
                source=matching.grant_id,
                size_bytes=declaration.size_bytes,
                descriptor=matching.descriptor,
                expires_at=matching.expires_at,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class GlobalP2PCounters:
    digest: str
    downloaded_bytes: int = 0
    uploaded_bytes: int = 0
    ratio_equivalent_bytes: int = 0
    active_seed_seconds: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _digest(self.digest, "counter.digest"))
        for field in (
            "downloaded_bytes",
            "uploaded_bytes",
            "ratio_equivalent_bytes",
            "active_seed_seconds",
        ):
            _integer(getattr(self, field), f"counter.{field}")


class GlobalP2PCounterStore:
    """Durable monotonic internet transfer counters and idempotent ratio credits."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._conn:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise GlobalP2PPolicyError(f"unsupported global counter schema version {version}")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS counters (
                    digest TEXT PRIMARY KEY,
                    downloaded_bytes INTEGER NOT NULL,
                    uploaded_bytes INTEGER NOT NULL,
                    ratio_equivalent_bytes INTEGER NOT NULL,
                    active_seed_seconds INTEGER NOT NULL
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS ratio_credits (
                    evidence_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL,
                    bytes INTEGER NOT NULL
                )"""
            )
            self._conn.execute("PRAGMA user_version = 1")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> GlobalP2PCounterStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, digest: str) -> GlobalP2PCounters:
        row = self._conn.execute(
            """SELECT downloaded_bytes, uploaded_bytes, ratio_equivalent_bytes,
                      active_seed_seconds FROM counters WHERE digest = ?""",
            (digest,),
        ).fetchone()
        if row is None:
            return GlobalP2PCounters(digest)
        return GlobalP2PCounters(digest, *cast("tuple[int, int, int, int]", row))

    def get(self, digest: str) -> GlobalP2PCounters:
        canonical = _digest(digest, "counter digest")
        with self._lock:
            return self._get(canonical)

    @staticmethod
    def _add(current: int, delta: int, field: str) -> int:
        value = current + delta
        if value > _INT64_MAX:
            raise GlobalP2PPolicyError(f"{field} exceeds the durable counter range")
        return value

    def record_transfer(
        self,
        digest: str,
        *,
        downloaded_bytes: int = 0,
        uploaded_bytes: int = 0,
        active_seed_seconds: int = 0,
    ) -> GlobalP2PCounters:
        canonical = _digest(digest, "counter digest")
        downloaded = _integer(downloaded_bytes, "downloaded byte delta")
        uploaded = _integer(uploaded_bytes, "uploaded byte delta")
        active = _integer(active_seed_seconds, "active seed second delta")
        with self._lock, self._conn:
            current = self._get(canonical)
            updated = GlobalP2PCounters(
                canonical,
                self._add(current.downloaded_bytes, downloaded, "downloaded bytes"),
                self._add(current.uploaded_bytes, uploaded, "uploaded bytes"),
                current.ratio_equivalent_bytes,
                self._add(current.active_seed_seconds, active, "active seed seconds"),
            )
            self._conn.execute(
                """INSERT INTO counters VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(digest) DO UPDATE SET
                     downloaded_bytes=excluded.downloaded_bytes,
                     uploaded_bytes=excluded.uploaded_bytes,
                     ratio_equivalent_bytes=excluded.ratio_equivalent_bytes,
                     active_seed_seconds=excluded.active_seed_seconds""",
                (
                    updated.digest,
                    updated.downloaded_bytes,
                    updated.uploaded_bytes,
                    updated.ratio_equivalent_bytes,
                    updated.active_seed_seconds,
                ),
            )
            return updated

    def credit_ratio_equivalent(
        self, evidence_id: str, digest: str, byte_count: int
    ) -> GlobalP2PCounters:
        evidence = _string(evidence_id, "evidence id")
        canonical = _digest(digest, "counter digest")
        credited = _integer(byte_count, "ratio-equivalent bytes")
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT digest, bytes FROM ratio_credits WHERE evidence_id = ?", (evidence,)
            ).fetchone()
            if existing is not None:
                if existing != (canonical, credited):
                    raise GlobalP2PPolicyError(
                        "evidence id is already bound to different ratio credit"
                    )
                return self._get(canonical)
            current = self._get(canonical)
            updated = GlobalP2PCounters(
                canonical,
                current.downloaded_bytes,
                current.uploaded_bytes,
                self._add(
                    current.ratio_equivalent_bytes,
                    credited,
                    "ratio-equivalent bytes",
                ),
                current.active_seed_seconds,
            )
            self._conn.execute(
                "INSERT INTO ratio_credits VALUES (?, ?, ?)", (evidence, canonical, credited)
            )
            self._conn.execute(
                """INSERT INTO counters VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(digest) DO UPDATE SET
                     downloaded_bytes=excluded.downloaded_bytes,
                     uploaded_bytes=excluded.uploaded_bytes,
                     ratio_equivalent_bytes=excluded.ratio_equivalent_bytes,
                     active_seed_seconds=excluded.active_seed_seconds""",
                (
                    updated.digest,
                    updated.downloaded_bytes,
                    updated.uploaded_bytes,
                    updated.ratio_equivalent_bytes,
                    updated.active_seed_seconds,
                ),
            )
            return updated


@dataclass(frozen=True, slots=True)
class NatObservation:
    mapping_attempted: bool
    mapping_created: bool
    outbound_connection_observed: bool
    inbound_connection_observed: bool

    def __post_init__(self) -> None:
        for field in (
            "mapping_attempted",
            "mapping_created",
            "outbound_connection_observed",
            "inbound_connection_observed",
        ):
            _boolean(getattr(self, field), field)
        if self.mapping_created and not self.mapping_attempted:
            raise GlobalP2PPolicyError("a NAT mapping cannot be created without an attempt")


def classify_nat_outcome(observation: NatObservation) -> NatOutcome:
    """Report observed reachability without treating a mapping claim as success."""
    if observation.inbound_connection_observed:
        return "reachable"
    if observation.outbound_connection_observed:
        return "outbound-only"
    return "unreachable"
