"""Revocable public-swarm and seed authorization derived from current facts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .identity import AssetError, require_digest
from .p2p_descriptor import (
    P2PDescriptorError,
    P2PDescriptorV1,
    validate_p2p_descriptor,
    verify_p2p_descriptor,
)
from .public_acquisition import (
    PublicAcquisitionReceiptV1,
    PublicAcquisitionSourceV1,
    PublicSourceType,
)

P2P_GRANT_VERSION = 1
P2P_REMOTE_GRANT_MAX_SECONDS = 6 * 60 * 60

EvidenceType = Literal[
    "public-acquisition-receipt",
    "provider-enumeration",
    "consented-set",
    "manual-attestation",
]

_EVIDENCE_FOR_SOURCE: Mapping[PublicSourceType, EvidenceType] = {
    "official-provider": "provider-enumeration",
    "declarative-resolver": "public-acquisition-receipt",
    "code-resolver": "consented-set",
    "manual": "manual-attestation",
}


def _timestamp(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetError(f"{field} must be a finite non-negative timestamp")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise AssetError(f"{field} must be a finite non-negative timestamp") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise AssetError(f"{field} must be a finite non-negative timestamp")
    return parsed


def _opaque(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetError(f"{field} must be a non-empty trimmed string")
    if len(value) > 512 or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AssetError(f"{field} must be bounded and contain no control characters")
    return value


def _license(value: object) -> str:
    if not isinstance(value, str):
        raise AssetError("license must be a string")
    if len(value) > 4096 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise AssetError("license must be bounded and contain no control characters")
    return value


@dataclass(frozen=True)
class PublicSwarmDeclarationV1:
    """One currently trusted source fact before normalization into a grant."""

    version: int
    digest: str
    size_bytes: int
    source_type: PublicSourceType
    source_id: str
    source_revision: str
    license: str
    descriptor: P2PDescriptorV1
    refreshed_at: float
    expires_at: float
    evidence_id: str = ""
    listed_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != P2P_GRANT_VERSION:
            raise AssetError(f"unsupported public swarm declaration version {self.version}")
        object.__setattr__(self, "digest", require_digest(self.digest))
        size_bytes = cast("object", self.size_bytes)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise AssetError("public swarm declaration size must be a positive integer")
        if size_bytes <= 0:
            raise AssetError("public swarm declaration size must be a positive integer")
        object.__setattr__(self, "size_bytes", size_bytes)
        if self.source_type not in _EVIDENCE_FOR_SOURCE:
            raise AssetError(f"unknown public source type: {self.source_type!r}")
        object.__setattr__(self, "source_id", _opaque(self.source_id, "sourceId"))
        object.__setattr__(
            self,
            "source_revision",
            _opaque(self.source_revision, "sourceRevision"),
        )
        object.__setattr__(self, "license", _license(self.license))
        descriptor = cast("object", self.descriptor)
        if not isinstance(descriptor, P2PDescriptorV1):
            raise AssetError("descriptor must be a P2PDescriptorV1")
        object.__setattr__(self, "descriptor", descriptor)
        try:
            validate_p2p_descriptor(
                self.descriptor,
                asset_digest=self.digest,
                size=self.size_bytes,
            )
        except P2PDescriptorError as exc:
            raise AssetError(f"public swarm descriptor is invalid: {exc}") from exc
        refreshed_at = _timestamp(self.refreshed_at, "refreshedAt")
        expires_at = _timestamp(self.expires_at, "expiresAt")
        if expires_at <= refreshed_at:
            raise AssetError("public swarm grant expiry must follow its refresh")
        if expires_at > refreshed_at + P2P_REMOTE_GRANT_MAX_SECONDS:
            raise AssetError("public swarm grants cannot exceed six hours from refresh")
        object.__setattr__(self, "refreshed_at", refreshed_at)
        object.__setattr__(self, "expires_at", expires_at)
        evidence_id = self.evidence_id
        try:
            listed_urls = tuple(self.listed_urls)
        except TypeError as exc:
            raise AssetError("listed URLs must be a sequence") from exc
        if self.source_type == "declarative-resolver":
            if evidence_id:
                raise AssetError(
                    "declarative resolver evidence must come from an acquisition receipt"
                )
            if not listed_urls:
                raise AssetError("declarative resolver declarations require listed URLs")
        else:
            evidence_id = _opaque(evidence_id, "evidenceId")
            if listed_urls:
                raise AssetError("equivalent trusted grants do not carry receipt URLs")
        object.__setattr__(self, "evidence_id", evidence_id)
        if self.source_type == "declarative-resolver":
            source = PublicAcquisitionSourceV1(
                self.digest,
                self.size_bytes,
                self.source_type,
                self.source_id,
                self.source_revision,
                listed_urls,
            )
            object.__setattr__(self, "listed_urls", source.listed_urls)
        else:
            object.__setattr__(self, "listed_urls", ())


def public_swarm_grant_id(declaration: PublicSwarmDeclarationV1) -> str:
    identity = {
        "digest": declaration.digest,
        "sourceType": declaration.source_type,
        "sourceId": declaration.source_id,
        "sourceRevision": declaration.source_revision,
        "license": declaration.license,
        "descriptor": declaration.descriptor.to_wire(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicSwarmGrantV1:
    version: int
    grant_id: str
    digest: str
    source_type: PublicSourceType
    source_id: str
    source_revision: str
    license: str
    descriptor: P2PDescriptorV1
    expires_at: float

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != P2P_GRANT_VERSION:
            raise AssetError(f"unsupported public swarm grant version {self.version}")
        grant_id = _opaque(self.grant_id, "grantId")
        if len(grant_id) != 64 or any(
            character not in "0123456789abcdef" for character in grant_id
        ):
            raise AssetError("grantId must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "grant_id", grant_id)
        object.__setattr__(self, "digest", require_digest(self.digest))
        if self.source_type not in _EVIDENCE_FOR_SOURCE:
            raise AssetError(f"unknown public source type: {self.source_type!r}")
        object.__setattr__(self, "source_id", _opaque(self.source_id, "sourceId"))
        object.__setattr__(
            self,
            "source_revision",
            _opaque(self.source_revision, "sourceRevision"),
        )
        object.__setattr__(self, "license", _license(self.license))
        descriptor = cast("object", self.descriptor)
        if not isinstance(descriptor, P2PDescriptorV1):
            raise AssetError("descriptor must be a P2PDescriptorV1")
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expiresAt"))

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "grantId": self.grant_id,
            "digest": self.digest,
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "sourceRevision": self.source_revision,
            "license": self.license,
            "descriptor": self.descriptor.to_wire(),
            "expiresAt": self.expires_at,
        }


@dataclass(frozen=True)
class SeedGrantV1(PublicSwarmGrantV1):
    evidence_type: EvidenceType
    evidence_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = _EVIDENCE_FOR_SOURCE[self.source_type]
        if self.evidence_type != expected:
            raise AssetError(
                f"{self.source_type} requires {expected!r} evidence, got {self.evidence_type!r}"
            )
        object.__setattr__(self, "evidence_id", _opaque(self.evidence_id, "evidenceId"))

    def to_wire(self) -> dict[str, object]:
        return {
            **super().to_wire(),
            "evidenceType": self.evidence_type,
            "evidenceId": self.evidence_id,
        }


@dataclass(frozen=True)
class P2PGrantSnapshot:
    enabled: bool
    public_grants: tuple[PublicSwarmGrantV1, ...]
    seed_grants: tuple[SeedGrantV1, ...]

    def staging_download_grants(
        self,
        digest: str,
        *,
        already_local: bool,
    ) -> tuple[PublicSwarmGrantV1, ...]:
        """Authorize staging only while the complete digest is not local."""
        digest = require_digest(digest)
        if not self.enabled or already_local:
            return ()
        return tuple(grant for grant in self.public_grants if grant.digest == digest)


@dataclass(frozen=True)
class P2PGrantReconciliation:
    snapshot: P2PGrantSnapshot
    revoked_public_grant_ids: frozenset[str]
    revoked_seed_grant_ids: frozenset[str]
    revoked_download_digests: frozenset[str]
    revoked_seed_digests: frozenset[str]


def _public_grant(declaration: PublicSwarmDeclarationV1) -> PublicSwarmGrantV1:
    return PublicSwarmGrantV1(
        version=P2P_GRANT_VERSION,
        grant_id=public_swarm_grant_id(declaration),
        digest=declaration.digest,
        source_type=declaration.source_type,
        source_id=declaration.source_id,
        source_revision=declaration.source_revision,
        license=declaration.license,
        descriptor=declaration.descriptor,
        expires_at=declaration.expires_at,
    )


def _receipt_evidence(
    declaration: PublicSwarmDeclarationV1,
    receipts: Iterable[PublicAcquisitionReceiptV1],
    now: float,
) -> str:
    matching = sorted(
        (
            receipt
            for receipt in receipts
            if receipt.fetched_at <= now
            and receipt.digest == declaration.digest
            and receipt.size_bytes == declaration.size_bytes
            and receipt.source_type == declaration.source_type
            and receipt.source_id == declaration.source_id
            and receipt.source_revision == declaration.source_revision
            and receipt.listed_url in declaration.listed_urls
        ),
        key=lambda receipt: (receipt.fetched_at, receipt.receipt_id),
        reverse=True,
    )
    return matching[0].receipt_id if matching else ""


class P2PGrantReconciler:
    """Compute effective grants and report authority removed since the last snapshot."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._snapshot = P2PGrantSnapshot(False, (), ())

    @property
    def snapshot(self) -> P2PGrantSnapshot:
        return self._snapshot

    def reconcile(
        self,
        declarations: Iterable[PublicSwarmDeclarationV1],
        receipts: Iterable[PublicAcquisitionReceiptV1],
        local_path_for: Callable[[str], Path | None],
        *,
        enabled: bool = False,
    ) -> P2PGrantReconciliation:
        now = _timestamp(self._clock(), "P2P grant clock")
        enabled_value = cast("object", enabled)
        if not isinstance(enabled_value, bool):
            raise AssetError("P2P enabled state must be a boolean")
        declaration_values = tuple(cast("Iterable[object]", declarations))
        if any(not isinstance(row, PublicSwarmDeclarationV1) for row in declaration_values):
            raise AssetError("P2P declarations contain an invalid row")
        declaration_rows = cast("tuple[PublicSwarmDeclarationV1, ...]", declaration_values)
        receipt_values = tuple(cast("Iterable[object]", receipts))
        if any(not isinstance(row, PublicAcquisitionReceiptV1) for row in receipt_values):
            raise AssetError("P2P receipts contain an invalid row")
        receipt_rows = cast("tuple[PublicAcquisitionReceiptV1, ...]", receipt_values)

        current: dict[str, tuple[PublicSwarmDeclarationV1, PublicSwarmGrantV1]] = {}
        if enabled:
            for declaration in declaration_rows:
                if declaration.refreshed_at > now or declaration.expires_at <= now:
                    continue
                grant = _public_grant(declaration)
                previous = current.get(grant.grant_id)
                if previous is not None and previous != (declaration, grant):
                    raise AssetError("conflicting P2P declarations have the same grant identity")
                current[grant.grant_id] = (declaration, grant)

        by_digest: dict[str, list[tuple[PublicSwarmDeclarationV1, PublicSwarmGrantV1]]] = {}
        for row in current.values():
            by_digest.setdefault(row[1].digest, []).append(row)

        seed_grants: list[SeedGrantV1] = []
        verified: dict[tuple[str, int, P2PDescriptorV1], bool] = {}
        for digest, rows in by_digest.items():
            if len({(row.size_bytes, row.descriptor) for row, _ in rows}) != 1:
                for _, grant in rows:
                    del current[grant.grant_id]
                continue
            local_path = local_path_for(digest)
            if local_path is None:
                continue
            for declaration, public in rows:
                if declaration.source_type == "declarative-resolver":
                    evidence_id = _receipt_evidence(declaration, receipt_rows, now)
                    if not evidence_id:
                        continue
                else:
                    evidence_id = declaration.evidence_id
                verification_key = (
                    str(local_path),
                    declaration.size_bytes,
                    declaration.descriptor,
                )
                valid = verified.get(verification_key)
                if valid is None:
                    try:
                        verify_p2p_descriptor(
                            local_path,
                            declaration.descriptor,
                            asset_digest=digest,
                            size=declaration.size_bytes,
                        )
                    except (OSError, AssetError):
                        valid = False
                    else:
                        valid = True
                    verified[verification_key] = valid
                if not valid:
                    continue
                seed_grants.append(
                    SeedGrantV1(
                        public.version,
                        public.grant_id,
                        public.digest,
                        public.source_type,
                        public.source_id,
                        public.source_revision,
                        public.license,
                        public.descriptor,
                        public.expires_at,
                        evidence_type=_EVIDENCE_FOR_SOURCE[public.source_type],
                        evidence_id=evidence_id,
                    )
                )

        public_grants = tuple(sorted((row[1] for row in current.values()), key=_grant_sort_key))
        seeds = tuple(sorted(seed_grants, key=_grant_sort_key))
        next_snapshot = P2PGrantSnapshot(enabled, public_grants, seeds)
        previous = self._snapshot
        previous_public_ids = {grant.grant_id for grant in previous.public_grants}
        next_public_ids = {grant.grant_id for grant in public_grants}
        previous_seed_ids = {grant.grant_id for grant in previous.seed_grants}
        next_seed_ids = {grant.grant_id for grant in seeds}
        previous_download_digests = {grant.digest for grant in previous.public_grants}
        next_download_digests = {grant.digest for grant in public_grants}
        previous_seed_digests = {grant.digest for grant in previous.seed_grants}
        next_seed_digests = {grant.digest for grant in seeds}
        self._snapshot = next_snapshot
        return P2PGrantReconciliation(
            next_snapshot,
            frozenset(previous_public_ids - next_public_ids),
            frozenset(previous_seed_ids - next_seed_ids),
            frozenset(previous_download_digests - next_download_digests),
            frozenset(previous_seed_digests - next_seed_digests),
        )


def _grant_sort_key(
    grant: PublicSwarmGrantV1 | SeedGrantV1,
) -> tuple[str, str, str, str]:
    return (grant.digest, grant.source_type, grant.source_id, grant.grant_id)
