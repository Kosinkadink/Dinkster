"""Asset needs: what a template or workflow requires, before bytes exist
locally (DESIGN 3.12 + roadmap "templates/asset distribution").

A need is the declaration side of distribution: a human-facing display
name ALWAYS (users think in names), an optional content digest (the
authority whenever present), an optional kind, expected size/media type,
a metadata snapshot for substitution UX, and SOURCE HINTS - places the
bytes might be obtainable. Two source shapes exist:

- ``PackagedSource``: a file shipped inside an installed pack's artifact
  (small starter assets ride the pack; the 10 GB-pip-package mistake is
  exactly what the remote shape exists to avoid).
- ``RemoteSource``: an http(s) URL. A lead, never an authority - fetched
  bytes verify against the digest or land nowhere.

A need WITHOUT a digest is deliberately representable: imported ComfyUI
workflows know names before anyone hashed the file. Such a need can never
be verified, so acquisition refuses it (status "unverifiable") - the
reserved slot for a future name-guess compatibility tier with its own
explicit acceptance step, never a silent fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from .catalog import validate_virtual_path
from .component_manifest import AssetComponentManifest, asset_kind_matches
from .identity import AssetError, require_digest
from .kind import require_asset_kind


@dataclass(frozen=True)
class PackagedSource:
    """Bytes shipped inside an installed pack: ``pack`` is the pack id,
    ``path`` a '/'-separated path relative to the pack's artifact root.
    The path grammar is the catalog's virtual-path grammar (no '..', no
    empty segments), so a manifest can never point outside its pack."""

    pack: str
    path: str

    def __post_init__(self) -> None:
        if not self.pack:
            raise AssetError("packaged source requires a pack id")
        validate_virtual_path(self.path)

    def to_wire(self) -> dict[str, object]:
        return {"type": "packaged", "pack": self.pack, "path": self.path}


@dataclass(frozen=True)
class RemoteSource:
    """Bytes obtainable from an http(s) URL. Verification is the digest's
    job; the URL is only a lead."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise AssetError(f"remote asset sources must be http(s) URLs, got: {self.url!r}")

    def to_wire(self) -> dict[str, object]:
        return {"type": "remote", "url": self.url}


AssetSource = PackagedSource | RemoteSource


def source_from_wire(wire: Mapping[str, object]) -> AssetSource:
    source_type = wire.get("type")
    if source_type == "packaged":
        return PackagedSource(pack=str(wire.get("pack", "")), path=str(wire.get("path", "")))
    if source_type == "remote":
        return RemoteSource(url=str(wire.get("url", "")))
    raise AssetError(f"unknown asset source type: {source_type!r}")


@dataclass(frozen=True)
class AssetNeed:
    """One required asset, as a template/workflow declares it.

    ``name`` is mandatory and human-facing. ``digest`` is optional but
    authoritative when present; everything else is advisory metadata for
    display, filtering, and substitution suggestions."""

    name: str
    digest: str = ""
    kind: str = ""
    size: int = -1
    """Expected byte size, -1 when unknown. Display/plan data only - the
    digest is the verification, never the size."""
    media_type: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    sources: tuple[AssetSource, ...] = ()
    component_manifest: AssetComponentManifest | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise AssetError("asset needs require a non-empty display name")
        if self.digest:
            require_digest(self.digest)
        if self.kind:
            require_asset_kind(self.kind)
        if self.size < -1:
            raise AssetError(f"asset need size must be >= -1, got {self.size}")
        component_manifest = cast("object", self.component_manifest)
        if component_manifest is not None and not isinstance(
            component_manifest, AssetComponentManifest
        ):
            raise AssetError("asset need component manifest must be an AssetComponentManifest")

    def matches_kind(self, accepted_kind: str) -> bool:
        if not self.kind:
            return False
        return asset_kind_matches(self.kind, accepted_kind, self.component_manifest)

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"name": self.name}
        if self.digest:
            wire["digest"] = self.digest
        if self.kind:
            wire["kind"] = self.kind
        if self.size >= 0:
            wire["size"] = self.size
        if self.media_type:
            wire["mediaType"] = self.media_type
        if self.metadata:
            wire["metadata"] = dict(self.metadata)
        if self.component_manifest is not None:
            wire["components"] = self.component_manifest.to_wire()
        if self.sources:
            wire["sources"] = [source.to_wire() for source in self.sources]
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> AssetNeed:
        """Strict decode: a malformed need raises AssetError. Callers with
        warn-and-drop surfaces (manifest loaders) catch per entry."""
        sources_raw = wire.get("sources", [])
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, str):
            raise AssetError("asset need 'sources' must be a list")
        sources: list[AssetSource] = []
        for entry in cast("Sequence[object]", sources_raw):
            if not isinstance(entry, Mapping):
                raise AssetError("asset need sources must be objects")
            sources.append(source_from_wire(cast("Mapping[str, object]", entry)))
        size = wire.get("size", -1)
        if not isinstance(size, int) or isinstance(size, bool):
            raise AssetError("asset need 'size' must be an integer")
        metadata_raw = wire.get("metadata", {})
        if not isinstance(metadata_raw, Mapping):
            raise AssetError("asset need 'metadata' must be an object")
        components_raw = wire.get("components")
        return cls(
            name=str(wire.get("name", "")),
            digest=str(wire.get("digest", "")),
            kind=str(wire.get("kind", "")),
            size=size,
            media_type=str(wire.get("mediaType", "")),
            metadata={str(k): v for k, v in cast("Mapping[object, object]", metadata_raw).items()},
            component_manifest=AssetComponentManifest.from_wire(components_raw)
            if components_raw is not None
            else None,
            sources=tuple(sources),
        )
