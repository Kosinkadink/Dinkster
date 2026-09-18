"""Internal logical-model, variant, artifact, and provider resolution facts.

This module deliberately owns no HTTP, acquisition, repair, or execution
surface. It merges already-observed facts by explicit stable identity and
derives orthogonal availability and compatibility without turning names,
paths, aliases, or metadata into content or variant authority.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from .catalog import validate_virtual_path
from .identity import AssetError, require_digest
from .json_metadata import freeze_json, thaw_json
from .kind import require_asset_kind
from .model import AssetRef

RESOLUTION_SCHEMA_VERSION = 2
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_REASON_LIMIT = 256
_SCHEMA_INIT_LOCK = threading.Lock()

MirrorState = Literal["available", "unavailable"]
Availability = Literal["local", "downloadable", "unavailable"]
Compatibility = Literal["compatible", "incompatible", "unknown"]
ResolutionStatus = Literal["resolved", "missing", "ambiguous", "incompatible"]
ResolutionTier = Literal[
    "mapping", "selected_variant", "trusted_source", "expected_digest", "compatibility"
]
AliasKind = Literal["logical", "variant"]


def _nonempty(value: object, field_name: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetError(f"{field_name} must be a non-empty string")
    cleaned = value.strip()
    if len(cleaned) > limit:
        raise AssetError(f"{field_name} exceeds {limit} characters")
    return cleaned


def _stable_id(value: object, field_name: str) -> str:
    cleaned = _nonempty(value, field_name).casefold()
    if _ID_RE.fullmatch(cleaned) is None:
        raise AssetError(f"{field_name} must contain only lowercase id characters")
    return cleaned


def _literal_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AssetError(f"{field_name} must be an integer >= {minimum}")
    return value


def _timestamp(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise AssetError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _reason(value: object, field_name: str = "reason") -> str:
    if not isinstance(value, str):
        raise AssetError(f"{field_name} must be a string")
    if len(value) > _REASON_LIMIT:
        raise AssetError(f"{field_name} exceeds {_REASON_LIMIT} characters")
    return value


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, (set, frozenset, tuple, list)):
        raise AssetError(f"{field_name} must be a collection")
    items = cast("set[object] | frozenset[object] | tuple[object, ...] | list[object]", value)
    return frozenset(_stable_id(item, field_name) for item in items)


def _metadata(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssetError("metadata must be an object")
    return cast(
        "Mapping[str, object]",
        freeze_json(cast("Mapping[object, object]", value)),
    )


def _metadata_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_object(text: object) -> Mapping[str, object]:
    if not isinstance(text, (str, bytes, bytearray)):
        raise AssetError("stored metadata is not valid JSON")
    try:
        loaded: object = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise AssetError("stored metadata is not valid JSON") from exc
    return _metadata(loaded)


def _complete_ref(ref: object) -> AssetRef:
    if not isinstance(ref, AssetRef):
        raise AssetError("local materialization ref must be an AssetRef")
    require_digest(ref.digest)
    _nonempty(ref.name, "ref name")
    _literal_int(ref.size, "ref size")
    _nonempty(ref.media_type, "ref media type")
    validate_virtual_path(ref.virtual_path)
    if ref.resolver is not None:
        raise AssetError("local materialization refs must not carry a resolver")
    return ref


@dataclass(frozen=True)
class SourceIdentity:
    provider_id: str
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _stable_id(self.provider_id, "provider id"))
        object.__setattr__(self, "source_id", _stable_id(self.source_id, "source id"))


@dataclass(frozen=True)
class LogicalModel:
    logical_id: str
    family: str
    asset_kind: str
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        object.__setattr__(self, "family", _stable_id(self.family, "family"))
        object.__setattr__(self, "asset_kind", require_asset_kind(self.asset_kind))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated at"))


@dataclass(frozen=True)
class DeclaredRequirements:
    loaders: frozenset[str] = frozenset()
    runtimes: frozenset[str] = frozenset()
    hardware: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "loaders", _string_set(self.loaders, "loader requirement"))
        object.__setattr__(self, "runtimes", _string_set(self.runtimes, "runtime requirement"))
        object.__setattr__(self, "hardware", _string_set(self.hardware, "hardware requirement"))


@dataclass(frozen=True)
class CompatibilityContext:
    loaders: frozenset[str] = frozenset()
    runtimes: frozenset[str] = frozenset()
    hardware: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "loaders", _string_set(self.loaders, "loader capability"))
        object.__setattr__(self, "runtimes", _string_set(self.runtimes, "runtime capability"))
        object.__setattr__(self, "hardware", _string_set(self.hardware, "hardware capability"))


@dataclass(frozen=True)
class ModelVariant:
    logical_id: str
    variant_id: str
    dtype: str
    quantization: str
    format: str
    role: str
    requirements: DeclaredRequirements = DeclaredRequirements()
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        object.__setattr__(self, "variant_id", _stable_id(self.variant_id, "variant id"))
        object.__setattr__(self, "dtype", _stable_id(self.dtype, "dtype"))
        object.__setattr__(self, "quantization", _stable_id(self.quantization, "quantization"))
        object.__setattr__(self, "format", _stable_id(self.format, "format"))
        object.__setattr__(self, "role", _stable_id(self.role, "role"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated at"))


@dataclass(frozen=True)
class Artifact:
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", require_digest(self.digest))


@dataclass(frozen=True)
class VariantArtifact:
    logical_id: str
    variant_id: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        object.__setattr__(self, "variant_id", _stable_id(self.variant_id, "variant id"))
        object.__setattr__(self, "digest", require_digest(self.digest))


@dataclass(frozen=True)
class AdvisoryAlias:
    kind: AliasKind
    alias: str
    logical_id: str
    variant_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("logical", "variant"):
            raise AssetError(f"unknown alias kind: {self.kind!r}")
        object.__setattr__(self, "alias", _nonempty(self.alias, "alias"))
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        if self.kind == "logical" and self.variant_id is not None:
            raise AssetError("logical alias cannot target a variant")
        if self.kind == "variant" and self.variant_id is None:
            raise AssetError("variant alias requires a variant id")
        if self.variant_id is not None:
            object.__setattr__(self, "variant_id", _stable_id(self.variant_id, "variant id"))


@dataclass(frozen=True)
class ProviderMirror:
    source: SourceIdentity
    digest: str
    state: MirrorState
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=lambda: dict[str, object]())
    observed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", require_digest(self.digest))
        if self.state not in ("available", "unavailable"):
            raise AssetError(f"unknown mirror state: {self.state!r}")
        object.__setattr__(self, "reason", _reason(self.reason))
        if self.state == "available" and self.reason:
            raise AssetError("available mirror cannot carry an unavailable reason")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed at"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated at"))


@dataclass(frozen=True)
class LocalMaterialization:
    scope: str
    ref: AssetRef
    observed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))
        object.__setattr__(self, "ref", _complete_ref(self.ref))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed at"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated at"))


@dataclass(frozen=True)
class MountMaterialization:
    scope: str
    mount_id: str
    priority: int
    asset_kind: str
    ref: AssetRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))
        object.__setattr__(self, "mount_id", _stable_id(self.mount_id, "mount id"))
        priority = cast("object", self.priority)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise AssetError("mount priority must be an integer")
        asset_kind = cast("object", self.asset_kind)
        if not isinstance(asset_kind, str):
            raise AssetError("mount asset kind must be a string")
        if asset_kind:
            object.__setattr__(self, "asset_kind", require_asset_kind(asset_kind))
        ref = _complete_ref(self.ref)
        if not ref.virtual_path.startswith(f"mounts/{self.mount_id}/"):
            raise AssetError("mount ref virtual path must match its mount id")
        object.__setattr__(self, "ref", ref)


@dataclass(frozen=True)
class ReferenceMapping:
    scope: str
    reference_key: str
    logical_id: str
    variant_id: str
    digest: str
    trusted_source: SourceIdentity | None = None
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))
        object.__setattr__(self, "reference_key", _nonempty(self.reference_key, "reference key"))
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        object.__setattr__(self, "variant_id", _stable_id(self.variant_id, "variant id"))
        object.__setattr__(self, "digest", require_digest(self.digest))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated at"))


@dataclass(frozen=True)
class ResolutionRequest:
    scope: str
    reference_key: str
    asset_kind: str
    authorized_providers: frozenset[str]
    logical_id: str | None = None
    variant_id: str | None = None
    expected_digest: str | None = None
    trusted_source: SourceIdentity | None = None
    trusted_digest: str | None = None
    compatibility: CompatibilityContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))
        object.__setattr__(self, "reference_key", _nonempty(self.reference_key, "reference key"))
        object.__setattr__(self, "asset_kind", require_asset_kind(self.asset_kind))
        object.__setattr__(
            self,
            "authorized_providers",
            frozenset(
                _stable_id(item, "authorized provider") for item in self.authorized_providers
            ),
        )
        if self.logical_id is not None:
            object.__setattr__(
                self, "logical_id", _nonempty(self.logical_id, "logical id or alias")
            )
        if self.variant_id is not None:
            object.__setattr__(
                self, "variant_id", _nonempty(self.variant_id, "variant id or alias")
            )
        if self.variant_id is not None and self.logical_id is None:
            raise AssetError("variant selection requires a logical selection")
        if self.expected_digest is not None:
            object.__setattr__(self, "expected_digest", require_digest(self.expected_digest))
        if (self.trusted_source is None) != (self.trusted_digest is None):
            raise AssetError("trusted source and trusted digest must be supplied together")
        if self.trusted_digest is not None:
            object.__setattr__(self, "trusted_digest", require_digest(self.trusted_digest))


@dataclass(frozen=True)
class ResolutionSnapshot:
    logical_models: tuple[LogicalModel, ...] = ()
    variants: tuple[ModelVariant, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    links: tuple[VariantArtifact, ...] = ()
    aliases: tuple[AdvisoryAlias, ...] = ()
    mirrors: tuple[ProviderMirror, ...] = ()
    locals: tuple[LocalMaterialization, ...] = ()
    mounts: tuple[MountMaterialization, ...] = ()

    def __post_init__(self) -> None:
        row_types: tuple[tuple[str, type[object]], ...] = (
            ("logical_models", LogicalModel),
            ("variants", ModelVariant),
            ("artifacts", Artifact),
            ("links", VariantArtifact),
            ("aliases", AdvisoryAlias),
            ("mirrors", ProviderMirror),
            ("locals", LocalMaterialization),
            ("mounts", MountMaterialization),
        )
        for field_name, row_type in row_types:
            value: object = getattr(self, field_name)
            if not isinstance(value, (tuple, list)):
                raise AssetError(f"snapshot {field_name} must be a sequence")
            rows = tuple(cast("tuple[object, ...] | list[object]", value))
            if any(not isinstance(row, row_type) for row in rows):
                raise AssetError(f"snapshot {field_name} contains an invalid row")
            object.__setattr__(self, field_name, rows)

        identities: tuple[tuple[str, list[object]], ...] = (
            ("logical model", [row.logical_id for row in self.logical_models]),
            (
                "variant",
                [(row.logical_id, row.variant_id) for row in self.variants],
            ),
            ("artifact", [row.digest for row in self.artifacts]),
            (
                "variant artifact",
                [(row.logical_id, row.variant_id, row.digest) for row in self.links],
            ),
            (
                "alias",
                [(row.kind, row.alias, row.logical_id, row.variant_id) for row in self.aliases],
            ),
            ("provider source", [row.source for row in self.mirrors]),
            (
                "local materialization",
                [(row.scope, row.ref.digest) for row in self.locals],
            ),
            (
                "mount materialization",
                [(row.scope, row.mount_id, row.ref.virtual_path) for row in self.mounts],
            ),
        )
        for identity_name, keys in identities:
            if len(keys) != len(set(keys)):
                raise AssetError(f"snapshot contains duplicate {identity_name} identity")


@dataclass(frozen=True)
class ResolutionCandidate:
    logical_id: str
    variant_id: str
    digest: str
    availability: Availability
    availability_reason: str
    compatibility: Compatibility
    compatibility_reason: str
    ref: AssetRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_id", _stable_id(self.logical_id, "logical id"))
        object.__setattr__(self, "variant_id", _stable_id(self.variant_id, "variant id"))
        object.__setattr__(self, "digest", require_digest(self.digest))
        if self.availability not in ("local", "downloadable", "unavailable"):
            raise AssetError(f"unknown availability: {self.availability!r}")
        object.__setattr__(
            self,
            "availability_reason",
            _reason(self.availability_reason, "availability reason"),
        )
        if self.compatibility not in ("compatible", "incompatible", "unknown"):
            raise AssetError(f"unknown compatibility: {self.compatibility!r}")
        object.__setattr__(
            self,
            "compatibility_reason",
            _reason(self.compatibility_reason, "compatibility reason"),
        )
        if self.compatibility == "compatible" and self.compatibility_reason:
            raise AssetError("compatible candidate cannot carry a compatibility reason")
        if self.compatibility != "compatible" and not self.compatibility_reason:
            raise AssetError("non-compatible candidate requires a compatibility reason")
        if (self.availability == "local") != (self.ref is not None):
            raise AssetError("only local candidates carry an admitted AssetRef")
        if self.ref is not None:
            object.__setattr__(self, "ref", _complete_ref(self.ref))
            if self.ref.digest != self.digest:
                raise AssetError("candidate ref digest disagrees with artifact")


@dataclass(frozen=True)
class ResolutionOutcome:
    status: ResolutionStatus
    tier: ResolutionTier | None
    selected: ResolutionCandidate | None
    candidates: tuple[ResolutionCandidate, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "missing", "ambiguous", "incompatible"):
            raise AssetError(f"unknown resolution status: {self.status!r}")
        object.__setattr__(self, "reason", _reason(self.reason))
        if self.status == "resolved" and (
            self.selected is None
            or self.selected.availability != "local"
            or self.selected.compatibility != "compatible"
            or self.tier is None
        ):
            raise AssetError(
                "resolved outcome requires a tier and compatible local selected candidate"
            )


def _compatibility(
    variant: ModelVariant, context: CompatibilityContext | None
) -> tuple[Compatibility, str]:
    requirements = variant.requirements
    if context is None:
        return "unknown", "compatibility context required"
    missing_loaders = requirements.loaders - context.loaders
    missing_runtimes = requirements.runtimes - context.runtimes
    missing_hardware = requirements.hardware - context.hardware
    parts: list[str] = []
    if missing_loaders:
        parts.append("loader:" + ",".join(sorted(missing_loaders)))
    if missing_runtimes:
        parts.append("runtime:" + ",".join(sorted(missing_runtimes)))
    if missing_hardware:
        parts.append("hardware:" + ",".join(sorted(missing_hardware)))
    detailed = "; ".join(parts)
    if not detailed:
        return "compatible", ""
    if len(detailed) <= _REASON_LIMIT:
        return "incompatible", detailed
    return "incompatible", (
        "missing compatibility requirements: "
        f"loaders={len(missing_loaders)},runtimes={len(missing_runtimes)},"
        f"hardware={len(missing_hardware)}"
    )


def _canonical_selection(
    value: str,
    canonical: set[str],
    aliases: Iterable[AdvisoryAlias],
    *,
    kind: AliasKind,
    logical_id: str | None = None,
) -> tuple[str | None, bool]:
    normalized = value.strip().casefold()
    if normalized in canonical:
        return normalized, False
    matches = {
        alias.variant_id if kind == "variant" else alias.logical_id
        for alias in aliases
        if alias.kind == kind
        and alias.alias == value.strip()
        and (logical_id is None or alias.logical_id == logical_id)
    }
    return (next(iter(matches)), False) if len(matches) == 1 else (None, len(matches) > 1)


def _candidate_facts(
    request: ResolutionRequest, snapshot: ResolutionSnapshot
) -> tuple[ResolutionCandidate, ...]:
    logical = {row.logical_id: row for row in snapshot.logical_models}
    variants = {(row.logical_id, row.variant_id): row for row in snapshot.variants}
    local_by_digest = {row.ref.digest: row for row in snapshot.locals if row.scope == request.scope}
    mount_by_kind_digest: dict[tuple[str, str], MountMaterialization] = {}
    for row in sorted(
        (item for item in snapshot.mounts if item.scope == request.scope),
        key=lambda item: (item.priority, item.mount_id, item.ref.virtual_path),
    ):
        kind = _mount_asset_kind(row)
        mount_by_kind_digest.setdefault((kind, row.ref.digest), row)
    mirrors_by_digest: dict[str, list[ProviderMirror]] = {}
    for mirror in snapshot.mirrors:
        if mirror.source.provider_id in request.authorized_providers:
            mirrors_by_digest.setdefault(mirror.digest, []).append(mirror)
    candidates: list[ResolutionCandidate] = []
    for link in snapshot.links:
        variant = variants.get((link.logical_id, link.variant_id))
        if variant is None:
            continue
        compatibility, compatibility_reason = _compatibility(variant, request.compatibility)
        local = local_by_digest.get(link.digest)
        mount = (
            mount_by_kind_digest.get((logical[link.logical_id].asset_kind, link.digest))
            if link.logical_id in logical
            else None
        )
        mirrors = mirrors_by_digest.get(link.digest, [])
        if local is not None:
            availability: Availability = "local"
            availability_reason = "local materialization"
            ref = local.ref
        elif mount is not None:
            availability = "local"
            availability_reason = "ready local mount"
            ref = mount.ref
        elif any(mirror.state == "available" for mirror in mirrors):
            availability = "downloadable"
            availability_reason = "authorized provider mirror available"
            ref = None
        else:
            availability = "unavailable"
            reasons = sorted({mirror.reason for mirror in mirrors if mirror.reason})
            availability_reason = (
                reasons[0] if reasons else "no local or authorized provider materialization"
            )
            ref = None
        candidates.append(
            ResolutionCandidate(
                link.logical_id,
                link.variant_id,
                link.digest,
                availability,
                availability_reason,
                compatibility,
                compatibility_reason,
                ref,
            )
        )
    return tuple(sorted(candidates, key=lambda row: (row.logical_id, row.variant_id, row.digest)))


def _mount_asset_kind(row: MountMaterialization) -> str:
    if row.asset_kind:
        return row.asset_kind
    major = row.ref.media_type.partition("/")[0]
    if major in {"audio", "image", "video"}:
        return f"media/{major}"
    return "asset/file"


def _finish(
    candidates: tuple[ResolutionCandidate, ...], tier: ResolutionTier, reason: str
) -> ResolutionOutcome:
    if not candidates:
        return ResolutionOutcome("missing", tier, None, (), reason)
    bindable = tuple(row for row in candidates if row.compatibility == "compatible")
    if len(bindable) > 1:
        return ResolutionOutcome("ambiguous", tier, None, candidates, reason)
    if not bindable:
        if len(candidates) > 1:
            return ResolutionOutcome("ambiguous", tier, None, candidates, reason)
        selected = next(iter(candidates))
        if selected.compatibility == "unknown":
            return ResolutionOutcome(
                "missing", tier, selected, candidates, selected.compatibility_reason
            )
        return ResolutionOutcome(
            "incompatible", tier, selected, candidates, selected.compatibility_reason
        )
    selected = next(iter(bindable))
    if selected.availability == "local":
        return ResolutionOutcome("resolved", tier, selected, candidates, reason)
    return ResolutionOutcome("missing", tier, selected, candidates, selected.availability_reason)


def _trusted_source_problem(
    source: SourceIdentity,
    digest: str,
    authorized_providers: frozenset[str],
    mirrors: tuple[ProviderMirror, ...],
) -> str:
    if source.provider_id not in authorized_providers:
        return "trusted provider unauthorized"
    observations = tuple(mirror for mirror in mirrors if mirror.source == source)
    if not observations:
        return "trusted source unavailable"
    if any(mirror.digest != digest for mirror in observations):
        return "trusted source digest mismatch"
    if not any(mirror.state == "available" for mirror in observations):
        return "trusted source unavailable"
    return ""


def resolve(
    request: ResolutionRequest,
    mapping: ReferenceMapping | None,
    snapshot: ResolutionSnapshot,
) -> ResolutionOutcome:
    """Resolve validated facts without file IO, hashing, network, or mutation."""

    logical = {row.logical_id: row for row in snapshot.logical_models}
    variants = {(row.logical_id, row.variant_id): row for row in snapshot.variants}
    facts = _candidate_facts(request, snapshot)

    selected_logical: str | None = None
    selected_variant: str | None = None
    if request.logical_id is not None:
        selected_logical, alias_ambiguous = _canonical_selection(
            request.logical_id, set(logical), snapshot.aliases, kind="logical"
        )
        if alias_ambiguous:
            return ResolutionOutcome("ambiguous", None, None, facts, "logical alias ambiguous")
        if selected_logical is None:
            return ResolutionOutcome("missing", None, None, facts, "logical model unknown")
        if logical[selected_logical].asset_kind != request.asset_kind:
            return ResolutionOutcome(
                "incompatible", None, None, facts, "logical model kind mismatch"
            )
    if request.variant_id is not None and selected_logical is not None:
        canonical_variants = {
            variant_id for logical_id, variant_id in variants if logical_id == selected_logical
        }
        selected_variant, alias_ambiguous = _canonical_selection(
            request.variant_id,
            canonical_variants,
            snapshot.aliases,
            kind="variant",
            logical_id=selected_logical,
        )
        if alias_ambiguous:
            return ResolutionOutcome("ambiguous", None, None, facts, "variant alias ambiguous")
        if selected_variant is None:
            return ResolutionOutcome("missing", None, None, facts, "variant unknown")

    applicable_mapping = (
        mapping
        if mapping is not None
        and mapping.scope == request.scope
        and mapping.reference_key == request.reference_key
        else None
    )
    if applicable_mapping is not None:
        model = logical.get(applicable_mapping.logical_id)
        if model is None or model.asset_kind != request.asset_kind:
            return ResolutionOutcome(
                "incompatible", "mapping", None, facts, "mapping kind mismatch"
            )
        if (selected_logical is not None and selected_logical != applicable_mapping.logical_id) or (
            selected_variant is not None and selected_variant != applicable_mapping.variant_id
        ):
            return ResolutionOutcome(
                "incompatible", "mapping", None, facts, "mapping selection mismatch"
            )
        if (
            request.expected_digest is not None
            and request.expected_digest != applicable_mapping.digest
        ):
            return ResolutionOutcome(
                "incompatible", "mapping", None, facts, "mapping digest mismatch"
            )
        if (
            request.trusted_digest is not None
            and request.trusted_digest != applicable_mapping.digest
        ):
            return ResolutionOutcome(
                "incompatible", "mapping", None, facts, "mapping trusted mismatch"
            )
        if (
            request.trusted_source is not None
            and applicable_mapping.trusted_source is not None
            and request.trusted_source != applicable_mapping.trusted_source
        ):
            return ResolutionOutcome(
                "incompatible", "mapping", None, facts, "mapping source mismatch"
            )
        required_sources = tuple(
            source
            for source in (applicable_mapping.trusted_source, request.trusted_source)
            if source is not None
        )
        for source in required_sources:
            problem = _trusted_source_problem(
                source,
                applicable_mapping.digest,
                request.authorized_providers,
                snapshot.mirrors,
            )
            if problem:
                return ResolutionOutcome("incompatible", "mapping", None, facts, problem)
        exact = tuple(
            row
            for row in facts
            if (row.logical_id, row.variant_id, row.digest)
            == (
                applicable_mapping.logical_id,
                applicable_mapping.variant_id,
                applicable_mapping.digest,
            )
        )
        return _finish(exact, "mapping", "exact reference mapping")

    filtered = tuple(
        row
        for row in facts
        if (selected_logical is None or row.logical_id == selected_logical)
        and (selected_variant is None or row.variant_id == selected_variant)
    )
    wrong_kind = tuple(
        row
        for row in filtered
        if logical.get(row.logical_id) is None
        or logical[row.logical_id].asset_kind != request.asset_kind
    )
    if wrong_kind and len(wrong_kind) == len(filtered):
        return ResolutionOutcome(
            "incompatible", None, None, wrong_kind, "logical model kind mismatch"
        )
    filtered = tuple(row for row in filtered if row not in wrong_kind)

    if request.trusted_source is not None and request.trusted_digest is not None:
        problem = _trusted_source_problem(
            request.trusted_source,
            request.trusted_digest,
            request.authorized_providers,
            snapshot.mirrors,
        )
        if problem:
            return ResolutionOutcome("incompatible", "trusted_source", None, filtered, problem)
        trusted = tuple(row for row in filtered if row.digest == request.trusted_digest)
        return _finish(trusted, "trusted_source", "exact trusted source and digest")
    if request.expected_digest is not None:
        expected = tuple(row for row in filtered if row.digest == request.expected_digest)
        return _finish(expected, "expected_digest", "exact expected digest")
    if selected_variant is not None:
        return _finish(filtered, "selected_variant", "explicit variant selection")
    if request.compatibility is not None:
        return _finish(filtered, "compatibility", "explicit compatibility leaves one candidate")
    return ResolutionOutcome(
        "missing" if not filtered else "ambiguous",
        None,
        None,
        filtered,
        "no sibling variant preference without explicit compatibility",
    )


_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS logical_models (
        logical_id TEXT PRIMARY KEY, family TEXT NOT NULL, asset_kind TEXT NOT NULL,
        updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS variants (
        logical_id TEXT NOT NULL, variant_id TEXT NOT NULL, dtype TEXT NOT NULL,
        quantization TEXT NOT NULL, format TEXT NOT NULL, role TEXT NOT NULL,
        requirements_json TEXT NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY (logical_id, variant_id))""",
    "CREATE TABLE IF NOT EXISTS artifacts (digest TEXT PRIMARY KEY)",
    """CREATE TABLE IF NOT EXISTS variant_artifacts (
        logical_id TEXT NOT NULL, variant_id TEXT NOT NULL, digest TEXT NOT NULL,
        PRIMARY KEY (logical_id, variant_id, digest))""",
    """CREATE TABLE IF NOT EXISTS aliases (
        kind TEXT NOT NULL, alias TEXT NOT NULL, logical_id TEXT NOT NULL,
        variant_id TEXT NOT NULL, PRIMARY KEY (kind, alias, logical_id, variant_id))""",
    """CREATE TABLE IF NOT EXISTS mirrors (
        provider_id TEXT NOT NULL, source_id TEXT NOT NULL, digest TEXT NOT NULL,
        state TEXT NOT NULL, reason TEXT NOT NULL, metadata_json TEXT NOT NULL,
        observed_at REAL NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY (provider_id, source_id))""",
    """CREATE TABLE IF NOT EXISTS local_materializations (
        scope TEXT NOT NULL, digest TEXT NOT NULL, name TEXT NOT NULL, size INTEGER NOT NULL,
        media_type TEXT NOT NULL, virtual_path TEXT NOT NULL, observed_at REAL NOT NULL,
        updated_at REAL NOT NULL, PRIMARY KEY (scope, digest))""",
    """CREATE TABLE IF NOT EXISTS mount_materializations (
        scope TEXT NOT NULL, mount_id TEXT NOT NULL, priority INTEGER NOT NULL,
        asset_kind TEXT NOT NULL, digest TEXT NOT NULL, name TEXT NOT NULL,
        size INTEGER NOT NULL, media_type TEXT NOT NULL, virtual_path TEXT NOT NULL,
        PRIMARY KEY (scope, mount_id, virtual_path))""",
    """CREATE TABLE IF NOT EXISTS mappings (
        scope TEXT NOT NULL, reference_key TEXT NOT NULL, logical_id TEXT NOT NULL,
        variant_id TEXT NOT NULL, digest TEXT NOT NULL, trusted_provider_id TEXT,
        trusted_source_id TEXT, updated_at REAL NOT NULL,
        PRIMARY KEY (scope, reference_key))""",
    "CREATE INDEX IF NOT EXISTS links_digest ON variant_artifacts(digest)",
    "CREATE INDEX IF NOT EXISTS mirrors_digest ON mirrors(digest, state)",
    "CREATE INDEX IF NOT EXISTS mirrors_provider ON mirrors(provider_id)",
)


class ResolutionStore:
    """Thread-safe dedicated SQLite store for normalized resolution facts."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        try:
            with _SCHEMA_INIT_LOCK, self._lock:
                version = self._schema_version()
                self._require_supported_version(version)
                if version != RESOLUTION_SCHEMA_VERSION:
                    self._initialize_schema()
                self._conn.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            self._conn.close()
            raise

    def _schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _require_supported_version(version: int) -> None:
        if version not in (0, 1, RESOLUTION_SCHEMA_VERSION):
            raise AssetError(
                f"resolution database has schema version {version}, "
                f"this build speaks {RESOLUTION_SCHEMA_VERSION}"
            )

    def _initialize_schema(self) -> None:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            version = self._schema_version()
            self._require_supported_version(version)
            if version in (0, 1):
                for statement in _SCHEMA_STATEMENTS:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {RESOLUTION_SCHEMA_VERSION}")
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_logical_model(self, model: LogicalModel) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT family, asset_kind FROM logical_models WHERE logical_id = ?",
                    (model.logical_id,),
                ).fetchone()
                if existing is not None and (
                    existing["family"] != model.family or existing["asset_kind"] != model.asset_kind
                ):
                    raise AssetError("logical identity family/kind cannot change")
                self._conn.execute(
                    "INSERT INTO logical_models VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(logical_id) DO UPDATE SET updated_at=excluded.updated_at",
                    (model.logical_id, model.family, model.asset_kind, model.updated_at),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def upsert_variant(self, variant: ModelVariant) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT * FROM variants WHERE logical_id = ? AND variant_id = ?",
                    (variant.logical_id, variant.variant_id),
                ).fetchone()
                requirements = variant.requirements
                if existing is not None:
                    if (
                        existing["dtype"] != variant.dtype
                        or existing["quantization"] != variant.quantization
                        or existing["format"] != variant.format
                        or existing["role"] != variant.role
                    ):
                        raise AssetError("stable variant execution identity cannot change")
                    stored = _json_object(existing["requirements_json"])
                    requirements = DeclaredRequirements(
                        loaders=(
                            variant.requirements.loaders
                            | _string_set(stored.get("loaders", ()), "loader requirement")
                        ),
                        runtimes=(
                            variant.requirements.runtimes
                            | _string_set(stored.get("runtimes", ()), "runtime requirement")
                        ),
                        hardware=(
                            variant.requirements.hardware
                            | _string_set(stored.get("hardware", ()), "hardware requirement")
                        ),
                    )
                requirements_json = json.dumps(
                    {
                        "hardware": sorted(requirements.hardware),
                        "loaders": sorted(requirements.loaders),
                        "runtimes": sorted(requirements.runtimes),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self._conn.execute(
                    """INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(logical_id, variant_id) DO UPDATE SET
                        requirements_json=excluded.requirements_json,
                        updated_at=excluded.updated_at""",
                    (
                        variant.logical_id,
                        variant.variant_id,
                        variant.dtype,
                        variant.quantization,
                        variant.format,
                        variant.role,
                        requirements_json,
                        variant.updated_at,
                    ),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def upsert_artifact(self, artifact: Artifact) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO artifacts VALUES (?)", (artifact.digest,))

    def link_variant_artifact(self, link: VariantArtifact) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO artifacts VALUES (?)", (link.digest,))
            self._conn.execute(
                "INSERT OR IGNORE INTO variant_artifacts VALUES (?, ?, ?)",
                (link.logical_id, link.variant_id, link.digest),
            )

    def add_alias(self, alias: AdvisoryAlias) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases VALUES (?, ?, ?, ?)",
                (alias.kind, alias.alias, alias.logical_id, alias.variant_id or ""),
            )

    def upsert_mirror(self, mirror: ProviderMirror) -> None:
        with self._lock, self._conn:
            self._insert_mirror(mirror, replace=True)

    def replace_provider_snapshot(
        self, provider_id: str, mirrors: Iterable[ProviderMirror]
    ) -> None:
        provider_id = _stable_id(provider_id, "provider id")
        rows = tuple(mirrors)
        if any(row.source.provider_id != provider_id for row in rows):
            raise AssetError("provider snapshot rows must match its provider")
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM mirrors WHERE provider_id = ?", (provider_id,))
            for row in rows:
                self._insert_mirror(row, replace=False)

    def _insert_mirror(self, mirror: ProviderMirror, *, replace: bool) -> None:
        self._conn.execute("INSERT OR IGNORE INTO artifacts VALUES (?)", (mirror.digest,))
        values = (
            mirror.source.provider_id,
            mirror.source.source_id,
            mirror.digest,
            mirror.state,
            mirror.reason,
            _metadata_json(mirror.metadata),
            mirror.observed_at,
            mirror.updated_at,
        )
        if replace:
            self._conn.execute(
                """INSERT INTO mirrors VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id, source_id) DO UPDATE SET digest=excluded.digest,
                    state=excluded.state, reason=excluded.reason,
                    metadata_json=excluded.metadata_json, observed_at=excluded.observed_at,
                    updated_at=excluded.updated_at""",
                values,
            )
        else:
            self._conn.execute("INSERT INTO mirrors VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)

    def upsert_local_materialization(self, materialization: LocalMaterialization) -> None:
        ref = materialization.ref
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO artifacts VALUES (?)", (ref.digest,))
            self._conn.execute(
                """INSERT INTO local_materializations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, digest) DO UPDATE SET name=excluded.name,
                    size=excluded.size, media_type=excluded.media_type,
                    virtual_path=excluded.virtual_path, observed_at=excluded.observed_at,
                    updated_at=excluded.updated_at""",
                (
                    materialization.scope,
                    ref.digest,
                    ref.name,
                    ref.size,
                    ref.media_type,
                    ref.virtual_path,
                    materialization.observed_at,
                    materialization.updated_at,
                ),
            )

    def replace_mount_snapshot(
        self,
        scopes: Iterable[str],
        materializations: Iterable[MountMaterialization],
    ) -> None:
        scope_rows = tuple(_nonempty(scope, "scope") for scope in scopes)
        if len(scope_rows) != len(set(scope_rows)):
            raise AssetError("mount snapshot contains duplicate scope")
        scope_set = frozenset(scope_rows)
        rows: tuple[object, ...] = tuple(cast("Iterable[object]", materializations))
        if any(not isinstance(row, MountMaterialization) for row in rows):
            raise AssetError("mount snapshot contains an invalid row")
        typed_rows = cast("tuple[MountMaterialization, ...]", rows)
        if any(row.scope not in scope_set for row in typed_rows):
            raise AssetError("mount snapshot row scope is not configured")
        identities = [(row.scope, row.mount_id, row.ref.virtual_path) for row in typed_rows]
        if len(identities) != len(set(identities)):
            raise AssetError("mount snapshot contains duplicate materialization identity")
        projections = {
            scope: {
                (row.mount_id, row.priority, row.asset_kind, row.ref)
                for row in typed_rows
                if row.scope == scope
            }
            for scope in scope_set
        }
        if projections and len({frozenset(rows) for rows in projections.values()}) != 1:
            raise AssetError("mount snapshot does not cover every configured scope")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for scope in scope_rows:
                    self._conn.execute(
                        "DELETE FROM mount_materializations WHERE scope = ?", (scope,)
                    )
                for row in typed_rows:
                    ref = row.ref
                    self._conn.execute(
                        "INSERT INTO mount_materializations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            row.scope,
                            row.mount_id,
                            row.priority,
                            row.asset_kind,
                            ref.digest,
                            ref.name,
                            ref.size,
                            ref.media_type,
                            ref.virtual_path,
                        ),
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def upsert_mapping(self, mapping: ReferenceMapping) -> None:
        provider = mapping.trusted_source.provider_id if mapping.trusted_source else None
        source = mapping.trusted_source.source_id if mapping.trusted_source else None
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO mappings VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, reference_key) DO UPDATE SET
                    logical_id=excluded.logical_id, variant_id=excluded.variant_id,
                    digest=excluded.digest, trusted_provider_id=excluded.trusted_provider_id,
                    trusted_source_id=excluded.trusted_source_id,
                    updated_at=excluded.updated_at""",
                (
                    mapping.scope,
                    mapping.reference_key,
                    mapping.logical_id,
                    mapping.variant_id,
                    mapping.digest,
                    provider,
                    source,
                    mapping.updated_at,
                ),
            )

    def get_mapping(self, scope: str, reference_key: str) -> ReferenceMapping | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mappings WHERE scope = ? AND reference_key = ?",
                (_nonempty(scope, "scope"), _nonempty(reference_key, "reference key")),
            ).fetchone()
        if row is None:
            return None
        provider = row["trusted_provider_id"]
        source = row["trusted_source_id"]
        if (provider is None) != (source is None):
            raise AssetError("stored mapping has incomplete trusted source")
        return ReferenceMapping(
            row["scope"],
            row["reference_key"],
            row["logical_id"],
            row["variant_id"],
            row["digest"],
            SourceIdentity(provider, source) if provider is not None else None,
            row["updated_at"],
        )

    def snapshot(self, scope: str) -> ResolutionSnapshot:
        scope = _nonempty(scope, "scope")
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                logical_rows = self._conn.execute(
                    "SELECT * FROM logical_models ORDER BY logical_id"
                ).fetchall()
                variant_rows = self._conn.execute(
                    "SELECT * FROM variants ORDER BY logical_id, variant_id"
                ).fetchall()
                artifact_rows = self._conn.execute(
                    "SELECT * FROM artifacts ORDER BY digest"
                ).fetchall()
                link_rows = self._conn.execute(
                    "SELECT * FROM variant_artifacts ORDER BY logical_id, variant_id, digest"
                ).fetchall()
                alias_rows = self._conn.execute(
                    "SELECT * FROM aliases ORDER BY kind, alias, logical_id, variant_id"
                ).fetchall()
                mirror_rows = self._conn.execute(
                    "SELECT * FROM mirrors ORDER BY provider_id, source_id"
                ).fetchall()
                local_rows = self._conn.execute(
                    "SELECT * FROM local_materializations WHERE scope = ? ORDER BY digest",
                    (scope,),
                ).fetchall()
                mount_rows = self._conn.execute(
                    """SELECT * FROM mount_materializations WHERE scope = ?
                    ORDER BY priority, mount_id, virtual_path""",
                    (scope,),
                ).fetchall()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        variants: list[ModelVariant] = []
        for row in variant_rows:
            requirements = _json_object(row["requirements_json"])
            variants.append(
                ModelVariant(
                    row["logical_id"],
                    row["variant_id"],
                    row["dtype"],
                    row["quantization"],
                    row["format"],
                    row["role"],
                    DeclaredRequirements(
                        loaders=_string_set(requirements.get("loaders", ()), "loader requirement"),
                        runtimes=_string_set(
                            requirements.get("runtimes", ()), "runtime requirement"
                        ),
                        hardware=_string_set(
                            requirements.get("hardware", ()), "hardware requirement"
                        ),
                    ),
                    row["updated_at"],
                )
            )
        return ResolutionSnapshot(
            logical_models=tuple(
                LogicalModel(row["logical_id"], row["family"], row["asset_kind"], row["updated_at"])
                for row in logical_rows
            ),
            variants=tuple(variants),
            artifacts=tuple(Artifact(row["digest"]) for row in artifact_rows),
            links=tuple(
                VariantArtifact(row["logical_id"], row["variant_id"], row["digest"])
                for row in link_rows
            ),
            aliases=tuple(
                AdvisoryAlias(
                    cast("AliasKind", row["kind"]),
                    row["alias"],
                    row["logical_id"],
                    row["variant_id"] or None,
                )
                for row in alias_rows
            ),
            mirrors=tuple(
                ProviderMirror(
                    SourceIdentity(row["provider_id"], row["source_id"]),
                    row["digest"],
                    cast("MirrorState", row["state"]),
                    row["reason"],
                    _json_object(row["metadata_json"]),
                    row["observed_at"],
                    row["updated_at"],
                )
                for row in mirror_rows
            ),
            locals=tuple(
                LocalMaterialization(
                    row["scope"],
                    AssetRef(
                        row["digest"],
                        row["name"],
                        row["size"],
                        row["media_type"],
                        row["virtual_path"],
                    ),
                    row["observed_at"],
                    row["updated_at"],
                )
                for row in local_rows
            ),
            mounts=tuple(
                MountMaterialization(
                    row["scope"],
                    row["mount_id"],
                    row["priority"],
                    row["asset_kind"],
                    AssetRef(
                        row["digest"],
                        row["name"],
                        row["size"],
                        row["media_type"],
                        row["virtual_path"],
                    ),
                )
                for row in mount_rows
            ),
        )
