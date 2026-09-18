"""Pack-declared assets: distribution metadata a pack ships (roadmap
"templates/asset distribution"; the auxiliary-model story).

A pack that needs model files - a controlnet preprocessor's dozen small
lineart/depth networks, a starter LoRA - declares them as DATA in its
manifest instead of shipping downloader code. Each declaration is an
:class:`AssetNeed` (digest as the sole authority, packaged and/or remote
sources as leads) plus a pack-local id and, optionally, the node types
whose INSTANTIATION requires the asset. Declarations feed exactly the
machinery that already exists: job preflight surfaces them in the
acquisition plan, consent stays digest-exact, verified acquisition lands
the bytes or nothing. Nothing here downloads at install, at composition,
or silently during execution.

Identity boundary: declarations are distribution records, never execution
identity. What a node actually loads is decided by the pack's CODE, and
the pack's version/digest already pins that code (environment stamping) -
the manifest entry only tells the host where those bytes can come from
and when to preflight them. A declaration that disagrees with the code is
a doctor finding, not a semantics change. Node code reads the landed
bytes through the worker-side hook (declared.py: ``declared_asset``),
which resolves the pack's own declarations by pack-local id over the
same vault/resolver chain - acquired-or-loud-error, never a download.

:class:`PackAssetCatalog` is the live registry the composed surface
maintains: pack artifact roots (for packaged acquisition) plus the
declared needs, indexed by digest, requiring node type, and selected
vision provider. The composer owns writes (packs compose, reload, and
remove); job preflight reads from worker threads - hence the lock and
snapshot semantics.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .identity import AssetError
from .need import AssetNeed, PackagedSource, RemoteSource

__all__ = ["DeclaredAsset", "PackAssetCatalog"]


@dataclass(frozen=True)
class DeclaredAsset:
    """One ``[[pack.assets]]`` declaration: a need with a pack-local handle.

    ``id`` is the stable pack-local name authors use to talk about the
    entry (docs, doctor findings); the digest inside ``need`` remains the
    only identity that matters for bytes. ``nodes`` lists node types whose
    instantiation requires the asset - the fixed-internal-model hook: a
    submitted graph containing one of those types preflights this asset
    exactly like a graph-literal reference (visible plan, digest-exact
    consent, never a silent download during execution). The association is
    author-declared data; the backend never derives it from node code."""

    id: str
    need: AssetNeed
    nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise AssetError("declared assets require a non-empty id")
        if not self.need.digest:
            raise AssetError(
                "declared assets require a digest - a pack distributing "
                "bytes must pin them (digestless needs belong to workflow "
                "import, not distribution)"
            )

    def descriptor(self) -> dict[str, object]:
        """The wire form for the packs table: the need's wire shape plus
        the pack-local id and the requiring node types."""
        wire: dict[str, object] = {"id": self.id}
        wire.update(self.need.to_wire())
        if self.nodes:
            wire["nodes"] = list(self.nodes)
        return wire


def _merge_needs(needs: Sequence[AssetNeed]) -> AssetNeed:
    """One digest declared by several packs is one need with merged source
    leads (mirrors). Descriptive fields come from the first declaration
    that filled them - they are display data, never authority."""
    base = needs[0]
    if len(needs) == 1:
        return base
    sources: list[PackagedSource | RemoteSource] = []
    for need in needs:
        for source in need.sources:
            if source not in sources:
                sources.append(source)
    kind = next((n.kind for n in needs if n.kind), "")
    size = next((n.size for n in needs if n.size >= 0), -1)
    media_type = next((n.media_type for n in needs if n.media_type), "")
    metadata = next((n.metadata for n in needs if n.metadata), base.metadata)
    component_manifest = next(
        (n.component_manifest for n in needs if n.component_manifest is not None),
        base.component_manifest,
    )
    return AssetNeed(
        name=base.name,
        digest=base.digest,
        kind=kind,
        size=size,
        media_type=media_type,
        metadata=metadata,
        component_manifest=component_manifest,
        sources=tuple(sources),
    )


class PackAssetCatalog:
    """Composed-surface registry of pack-declared assets and pack roots.

    Writes come from the composer under its own mutation lock (add,
    reload, remove all end in :meth:`replace_all`); reads come from job
    preflight in worker threads. Every read returns a snapshot computed
    under the lock, so a reload mid-preflight yields either the old
    surface or the new one, never a torn mix."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roots: dict[str, Path] = {}
        self._assets: dict[str, tuple[DeclaredAsset, ...]] = {}
        self._provider_assets: dict[tuple[str, str], tuple[DeclaredAsset, ...]] = {}

    def replace_all(
        self,
        entries: Iterable[tuple[str, Path | None, Sequence[DeclaredAsset]]],
        *,
        provider_entries: Iterable[tuple[str, str, Sequence[DeclaredAsset]]] = (),
    ) -> None:
        """Swap the whole catalog: ``(pack_id, artifact_root, assets)``
        per pack. Derive-don't-unpick, mirroring the composer's registry
        rebuild: records are the source of truth, so removal and reload
        both just rebuild. A ``None`` root registers the needs without
        packaged resolution (remote leads still work).

        ``provider_entries`` associates assets with a selected
        ``(node_type, provider_id)``. Unlike ``DeclaredAsset.nodes``, these
        are conditional: job preflight requests only the selected provider's
        artifacts when the selector is a literal."""
        roots: dict[str, Path] = {}
        assets: dict[str, tuple[DeclaredAsset, ...]] = {}
        for pack_id, root, declared in entries:
            if root is not None:
                roots[pack_id] = root
            if declared:
                assets[pack_id] = tuple(declared)
        providers = {
            (node_type, provider_id): tuple(declared)
            for node_type, provider_id, declared in provider_entries
        }
        with self._lock:
            self._roots = roots
            self._assets = assets
            self._provider_assets = providers

    def pack_roots(self) -> dict[str, Path]:
        """Snapshot of pack artifact roots for packaged acquisition."""
        with self._lock:
            return dict(self._roots)

    def all_declared(self) -> dict[str, tuple[DeclaredAsset, ...]]:
        """Snapshot of every declaration, by pack id. The name-guess
        surface reads display names and packaged paths from here;
        preflight keeps using the digest and node-type indexes."""
        with self._lock:
            return dict(self._assets)

    def need_for(self, digest: str) -> AssetNeed | None:
        """The merged declared need for a digest, or None: every pack that
        declared the digest contributes its sources as mirrors."""
        with self._lock:
            matches = [
                asset.need
                for declared in self._assets.values()
                for asset in declared
                if asset.need.digest == digest
            ]
        return _merge_needs(matches) if matches else None

    def needs_for_nodes(
        self,
        node_types: Iterable[str],
        *,
        provider_selections: Iterable[tuple[str, str]] = (),
        linked_provider_nodes: Iterable[str] = (),
    ) -> dict[str, AssetNeed]:
        """Return one atomic snapshot of pack-declared graph requirements.

        Ordinary ``nodes`` associations always contribute. A literal vision
        provider contributes only its selected artifact set; a linked
        provider selector contributes every installed provider for that node
        because its value is unknown until execution.
        """
        wanted = set(node_types)
        selected = set(provider_selections)
        linked = set(linked_provider_nodes)
        with self._lock:
            matches: dict[str, list[AssetNeed]] = {}
            for declared in self._assets.values():
                for asset in declared:
                    if asset.nodes and not wanted.isdisjoint(asset.nodes):
                        matches.setdefault(asset.need.digest, []).append(asset.need)
            for key in selected:
                for asset in self._provider_assets.get(key, ()):
                    matches.setdefault(asset.need.digest, []).append(asset.need)
            for (node_type, _provider), assets in self._provider_assets.items():
                if node_type not in linked:
                    continue
                for asset in assets:
                    matches.setdefault(asset.need.digest, []).append(asset.need)
        return {digest: _merge_needs(needs) for digest, needs in matches.items()}
