"""Canonical provenance for one resolved node-pack composition."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from dinkster_schema import canonical_name, validate_name

from .model import canonical_json, validate_artifact_digest, validate_version

COMPOSITION_FORMAT = "dinkster.composition/1"
CompositionMode = Literal["production", "development"]
RequirementKind = Literal["host", "api", "inference", "pack", "registry", "capability"]


class CompositionRecordError(ValueError):
    """A composition record is incomplete or non-canonical."""


@dataclass(frozen=True)
class ComposedPack:
    """One manifest pack and the release bytes used for composition."""

    pack: str
    version: str = ""
    artifact_digest: str = ""


@dataclass(frozen=True)
class ResolvedRequirement:
    """One pack requirement bound to its exact provider."""

    pack: str
    kind: RequirementKind
    requirement: str
    provider: str


@dataclass(frozen=True)
class CompositionGeneration:
    """Immutable, content-addressed provenance for one composed pack set."""

    mode: CompositionMode
    packs: tuple[ComposedPack, ...]
    resolutions: tuple[ResolvedRequirement, ...]

    @classmethod
    def of(
        cls,
        mode: CompositionMode,
        packs: tuple[ComposedPack, ...] | list[ComposedPack],
        resolutions: tuple[ResolvedRequirement, ...] | list[ResolvedRequirement] = (),
    ) -> CompositionGeneration:
        if mode not in ("production", "development"):
            raise CompositionRecordError(f"invalid composition mode {mode!r}")
        ordered_packs = tuple(sorted(packs, key=lambda item: item.pack))
        seen: set[str] = set()
        for entry in ordered_packs:
            problem = validate_name(entry.pack)
            if problem is not None:
                raise CompositionRecordError(f"pack id {entry.pack!r} {problem}")
            canonical = canonical_name(entry.pack)
            if entry.pack != canonical:
                raise CompositionRecordError(
                    f"pack id {entry.pack!r} is not canonical (expected {canonical!r})"
                )
            if canonical in seen:
                raise CompositionRecordError(f"composition repeats pack {entry.pack!r}")
            seen.add(canonical)
            if entry.version and validate_version(entry.version) is not None:
                raise CompositionRecordError(
                    f"pack {entry.pack!r} has invalid version {entry.version!r}"
                )
            if (
                entry.artifact_digest
                and validate_artifact_digest(entry.artifact_digest) is not None
            ):
                raise CompositionRecordError(
                    f"pack {entry.pack!r} has invalid artifact digest {entry.artifact_digest!r}"
                )
            if mode == "production" and not entry.artifact_digest:
                raise CompositionRecordError(
                    f"production composition pack {entry.pack!r} has no artifact digest"
                )

        valid_kinds = {"host", "api", "inference", "pack", "registry", "capability"}
        ordered_resolutions = tuple(
            sorted(
                resolutions,
                key=lambda item: (item.pack, item.kind, item.requirement, item.provider),
            )
        )
        resolution_keys: set[tuple[str, str, str]] = set()
        for item in ordered_resolutions:
            if item.pack not in seen:
                raise CompositionRecordError(
                    f"requirement resolution names unknown pack {item.pack!r}"
                )
            if item.kind not in valid_kinds:
                raise CompositionRecordError(f"invalid requirement kind {item.kind!r}")
            if not item.requirement or not item.provider:
                raise CompositionRecordError("requirement and provider must be non-empty")
            key = (item.pack, item.kind, item.requirement)
            if key in resolution_keys:
                raise CompositionRecordError(
                    f"composition resolves {item.pack!r} {item.kind} requirement "
                    f"{item.requirement!r} more than once"
                )
            resolution_keys.add(key)
        return cls(mode, ordered_packs, ordered_resolutions)

    def record_json(self) -> str:
        return canonical_json(
            {
                "format": COMPOSITION_FORMAT,
                "mode": self.mode,
                "packs": [
                    {
                        "pack": entry.pack,
                        "version": entry.version,
                        "artifactDigest": entry.artifact_digest,
                    }
                    for entry in self.packs
                ],
                "resolutions": [
                    {
                        "pack": item.pack,
                        "kind": item.kind,
                        "requirement": item.requirement,
                        "provider": item.provider,
                    }
                    for item in self.resolutions
                ],
            }
        )

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.record_json().encode()).hexdigest()


__all__ = [
    "COMPOSITION_FORMAT",
    "ComposedPack",
    "CompositionGeneration",
    "CompositionMode",
    "CompositionRecordError",
    "RequirementKind",
    "ResolvedRequirement",
]
