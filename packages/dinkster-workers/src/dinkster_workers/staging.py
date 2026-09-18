"""Asset staging for remote worker daemons (DESIGN 3.12 across machines).

A daemon executes nodes whose pack declares ``[[pack.assets]]``, but the
declaring manifest lives daemon-side and the bytes may not: the engine
host is where consented acquisition landed them (or where provenance says
they live). Staging closes that gap BEFORE dispatch, over the existing
authenticated worker socket: the engine asks which digests the daemon
holds (``assetQuery``), then names candidate HTTP(S) sources for the
missing ones (``stageAssets``); the daemon pulls them through
``fetch_asset`` into its own vault - streamed, digest-verified, rolled
back to nothing on any failure. Asset bytes NEVER ride the worker socket
itself: sources are HTTP pull only, so the socket stays a control plane
and verification stays in the vault's writer where it always is.

Execution-time reads stay downloads-free: ``declared_asset`` resolves
what staging (or an operator's pre-seeded store) already landed, exactly
like local execution. Staging is the remote replacement for the job
preflight that materializes assets host-side."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetVault,
    DeclaredAsset,
    fetch_asset,
    require_digest,
)
from dinkster_assets.fetch import DEFAULT_FETCH_TIMEOUT
from dinkster_assets.model import AssetResolver

__all__ = [
    "AssetStagingService",
    "StageAsset",
    "StagingSource",
    "declared_assets_from_wire",
]


@dataclass(frozen=True)
class StagingSource:
    """One candidate HTTP(S) URL for staged bytes, with request headers.

    Headers carry credentials (the engine's scoped bearer for its own
    /assets endpoint) - never place them in the URL, where they would
    leak into logs and error messages."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise AssetError(f"staging sources must be http(s) URLs, got: {self.url!r}")

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"url": self.url}
        if self.headers:
            wire["headers"] = dict(self.headers)
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> StagingSource:
        url = wire.get("url")
        if not isinstance(url, str) or not url:
            raise AssetError("staging source requires a non-empty 'url' string")
        headers_raw = wire.get("headers", {})
        if not isinstance(headers_raw, Mapping):
            raise AssetError("staging source 'headers' must be an object")
        headers = {
            str(name): str(value)
            for name, value in cast("Mapping[object, object]", headers_raw).items()
        }
        return cls(url=url, headers=headers)


@dataclass(frozen=True)
class StageAsset:
    """One asset a stageAssets frame asks the daemon to materialize."""

    digest: str
    name: str = ""
    size: int = -1
    """Expected byte size, -1 when unknown; progress display only - the
    digest is the verification, never the size."""
    sources: tuple[StagingSource, ...] = ()

    def __post_init__(self) -> None:
        require_digest(self.digest)
        if self.size < -1:
            raise AssetError(f"stage asset size must be >= -1, got {self.size}")

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"digest": self.digest}
        if self.name:
            wire["name"] = self.name
        if self.size >= 0:
            wire["size"] = self.size
        if self.sources:
            wire["sources"] = [source.to_wire() for source in self.sources]
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> StageAsset:
        """Strict decode: a malformed entry raises AssetError."""
        digest = wire.get("digest")
        if not isinstance(digest, str) or not digest:
            raise AssetError("stage asset requires a 'digest' string")
        size = wire.get("size", -1)
        if not isinstance(size, int) or isinstance(size, bool):
            raise AssetError("stage asset 'size' must be an integer")
        sources_raw = wire.get("sources", [])
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, str):
            raise AssetError("stage asset 'sources' must be a list")
        sources: list[StagingSource] = []
        for entry in cast("Sequence[object]", sources_raw):
            if not isinstance(entry, Mapping):
                raise AssetError("stage asset sources must be objects")
            sources.append(StagingSource.from_wire(cast("Mapping[str, object]", entry)))
        return cls(
            digest=digest,
            name=str(wire.get("name", "")),
            size=size,
            sources=tuple(sources),
        )


class AssetStagingService:
    """Daemon-side answerer for assetQuery/stageAssets frames.

    ``resolver`` is the same environment-assembled chain the hosted pack
    reads through (``resolver_from_env``), so "held" here means exactly
    "``declared_asset`` will resolve it". ``vault`` is where fetched
    bytes land; without one the daemon still answers queries honestly
    but refuses to stage."""

    def __init__(
        self,
        *,
        vault: AssetVault | None,
        resolver: AssetResolver | None,
        timeout: float = DEFAULT_FETCH_TIMEOUT,
    ) -> None:
        self._vault = vault
        self._resolver = resolver
        self._timeout = timeout

    def held(self, digests: Sequence[str]) -> tuple[list[str], list[str]]:
        """Partition digests into (held, missing) against the read chain.
        Synchronous (disk stats); callers on an event loop use a thread."""
        held: list[str] = []
        missing: list[str] = []
        for digest in digests:
            require_digest(digest)
            path: Path | None = None
            if self._resolver is not None:
                with contextlib.suppress(AssetError):
                    path = self._resolver.resolve(digest)
            (held if path is not None else missing).append(digest)
        return held, missing

    def stage(
        self,
        asset: StageAsset,
        *,
        should_abort: Callable[[], bool] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        """Materialize one asset into the vault, or raise AssetError with
        the reason (FetchAborted escapes on caller-requested abort).
        Synchronous and blocking for the download's duration - run in a
        thread from the conversation loop."""
        if self._vault is None:
            raise AssetError(
                "this service has no asset vault to stage into (start it with --asset-vault)"
            )
        if not asset.sources:
            raise AssetError(f"no sources offered for {asset.digest}")
        headers_by_url = {source.url: source.headers for source in asset.sources}

        def headers_for(url: str) -> Mapping[str, str]:
            return headers_by_url.get(url, {})

        path = fetch_asset(
            asset.digest,
            [source.url for source in asset.sources],
            self._vault,
            timeout=self._timeout,
            headers_for=headers_for,
            should_abort=should_abort,
            on_progress=on_progress,
        )
        if path is None:
            raise AssetError(
                f"no candidate source yielded bytes verifying as "
                f"{asset.digest} ({len(asset.sources)} tried)"
            )
        return path


def declared_assets_from_wire(raw: object) -> tuple[DeclaredAsset, ...]:
    """Strict decode of a hello's ``declaredAssets`` descriptors.

    The engine never sees the daemon's manifest, so the pack's
    ``[[pack.assets]]`` declarations ride the handshake as the same wire
    descriptors the packs table serves (:meth:`DeclaredAsset.descriptor`).
    A malformed entry raises AssetError - a peer that garbles its own
    declarations must not silently lose staging."""
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise AssetError("declaredAssets must be a list")
    assets: list[DeclaredAsset] = []
    for entry in cast("Sequence[object]", raw):
        if not isinstance(entry, Mapping):
            raise AssetError("declaredAssets entries must be objects")
        wire = cast("Mapping[str, object]", entry)
        asset_id = wire.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise AssetError("declared asset requires an 'id' string")
        nodes_raw = wire.get("nodes", [])
        if not isinstance(nodes_raw, Sequence) or isinstance(nodes_raw, str):
            raise AssetError("declared asset 'nodes' must be a list")
        nodes = tuple(str(node) for node in cast("Sequence[object]", nodes_raw))
        assets.append(DeclaredAsset(id=asset_id, need=AssetNeed.from_wire(wire), nodes=nodes))
    return tuple(assets)
