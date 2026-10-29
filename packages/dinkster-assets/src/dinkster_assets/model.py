"""AssetRef: the node-facing asset object (DESIGN 3.12).

Nodes never receive filesystem paths from the graph; they receive AssetRefs.
A ref carries identity (content digest) and cheap metadata everywhere; the
bytes are materialized only where and when a node actually reads them,
through whatever resolver the hosting worker was configured with. Crossing
a boundary drops the resolver and rebinds the receiving side's - the ref's
identity, equality, and fingerprint never change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from .identity import AssetError, require_digest
from .integrity import AssetVerificationRecord, open_verified, verified_local_path


class AssetResolver(Protocol):
    """Where digests become local bytes. Implementations: local directory
    libraries, disk CAS, HTTP stores, remote peers (planned). Returns None when
    the digest is not locally materializable."""

    def resolve(self, digest: str) -> Path | None: ...


@dataclass(frozen=True)
class AssetResolution:
    path: Path
    verification: AssetVerificationRecord | None = None


@runtime_checkable
class RecordedAssetResolver(Protocol):
    def resolve_asset(self, digest: str) -> AssetResolution | None: ...


@dataclass(frozen=True)
class AssetRef:
    """Identity plus metadata; never a path. Node authors receive these for
    schema inputs of type ``dinkster.asset`` and call ``local_path()`` /
    ``open()`` when they need bytes (hazard H9: writing a node stays boring).

    Equality and hashing are identity-and-metadata only: the resolver is a
    location-specific binding, not part of what the value *is*."""

    digest: str
    name: str
    size: int
    media_type: str = "application/octet-stream"
    virtual_path: str = ""
    resolver: AssetResolver | None = field(default=None, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        require_digest(self.digest)
        if self.size < 0:
            raise AssetError(f"asset size must be >= 0, got {self.size}")

    def _resolution(self) -> AssetResolution:
        """Locate a candidate path via the bound resolver - a location hint,
        not yet a content guarantee (resolvers use metadata heuristics)."""
        if self.resolver is None:
            raise AssetError(
                f"asset '{self.name}' ({self.digest}) has no resolver bound in "
                "this process; the hosting worker was not configured with an "
                "asset store"
            )
        if isinstance(self.resolver, RecordedAssetResolver):
            resolution = self.resolver.resolve_asset(self.digest)
        else:
            path = self.resolver.resolve(self.digest)
            resolution = None if path is None else AssetResolution(path)
        if resolution is None:
            raise AssetError(
                f"asset '{self.name}' ({self.digest}) is not materializable "
                "here: no configured store holds its content"
            )
        return resolution

    def _resolved_path(self) -> Path:
        return self._resolution().path

    def local_path(self) -> Path:
        """Materialize the asset and return a real local path whose content
        was verified at ingest and whose descriptor still matches that record.

        The path is a read-only view of content: treat it as scoped to this
        invocation, never store it in outputs (identity is the digest).
        Legacy and direct resolvers without an ingest record hash from the open
        descriptor. A path is rebindable by nature - after return the only
        remaining window is a concurrent writer, which is outside the threat
        model. Prefer :meth:`open` where a handle works."""
        resolution = self._resolution()
        return verified_local_path(resolution.path, self.digest, resolution.verification)

    def open(self) -> BinaryIO:
        """Open the asset for reading. The returned handle's bytes are
        bound to the ingest record through descriptor metadata. Resolvers
        without a record hash the same descriptor. Raises AssetIntegrityError
        on mismatch, never silently falls back from a stale record."""
        resolution = self._resolution()
        return open_verified(resolution.path, self.digest, resolution.verification)

    def read_bytes(self) -> bytes:
        """Read the asset's full content, verified like :meth:`open`."""
        with self.open() as handle:
            return handle.read()

    def to_wire(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "name": self.name,
            "size": self.size,
            "mediaType": self.media_type,
            "virtualPath": self.virtual_path,
        }

    @classmethod
    def from_wire(
        cls, wire: Mapping[str, object], resolver: AssetResolver | None = None
    ) -> AssetRef:
        digest = wire.get("digest")
        if not isinstance(digest, str):
            raise AssetError(f"asset wire form requires a 'digest' string, got: {wire!r}")
        size = wire.get("size", 0)
        if not isinstance(size, int):
            raise AssetError(f"asset 'size' must be an integer, got {size!r}")
        return cls(
            digest=digest,
            name=str(wire.get("name", "")),
            size=size,
            media_type=str(wire.get("mediaType", "application/octet-stream")),
            virtual_path=str(wire.get("virtualPath", "")),
            resolver=resolver,
        )
