"""Job asset preflight: no run starts with unresolvable assets, and no
byte downloads without digest-exact consent (roadmap "templates/asset
distribution"; the reproduce plan/apply principle applied to submission).

A submitted graph names assets by identity (``dinkster.asset`` literals carry
digests). Before a job is queued, every referenced identity must be
locally materializable; anything else fails the submit with a
machine-readable acquisition plan (409 ``assets-missing``) listing each
missing identity, its display name, and the source leads this server
knows. The caller - UI or remote API client alike - decides, then
resubmits with ``acquireAssets: [digest, ...]``: consent per identity,
never a blanket "download whatever you need".

Consent is TOCTOU-proof by construction: a digest names exact bytes, so
consenting to it cannot authorize different content no matter what any
source serves - acquisition verifies through the vault writer or lands
nothing. ``assetSources`` (digest -> [urls]) lets the submission carry
its own leads (a template's pinned URLs); they merge with the provenance
store's and are recorded there only after bytes actually verified.

A need with no digest cannot occur here (asset literals require one);
the "unverifiable" tier is reserved for the future name-guess import
path, which will carry its own explicit acceptance step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetSource,
    PackagedSource,
    RemoteSource,
    acquire_need,
    is_digest,
)
from dinkster_graph import Graph, Link, RegionNode, TypedLiteral

from .library import ServerLibrary

__all__ = [
    "asset_preflight",
    "graph_asset_names",
    "graph_node_types",
    "graph_provider_selections",
    "parse_asset_consent",
]


def graph_asset_names(graph: Graph) -> dict[str, str]:
    """Every asset identity a graph references, with a display name:
    digest -> name. Scans literal inputs at every nesting level (region
    inputs and bodies included); an asset literal is a mapping carrying a
    canonical digest, matching the ``dinkster.asset`` wire shape."""
    found: dict[str, str] = {}
    _scan_graph(graph, found)
    return found


def _scan_graph(graph: Graph, found: dict[str, str]) -> None:
    for node in graph.nodes.values():
        _scan_inputs(node.inputs, found)
        if isinstance(node, RegionNode):
            _scan_graph(node.body, found)


def graph_node_types(graph: Graph) -> set[str]:
    """Every node type a graph instantiates, at every nesting level.

    Feeds the node-declared asset hook: a ``[[pack.assets]]`` entry
    naming node types makes those assets preflight requirements for any
    graph that instantiates them - the fixed-internal-model story, where
    exposing an asset input would be noise but a silent mid-execution
    download would be worse."""
    types: set[str] = set()
    _scan_types(graph, types)
    return types


def _scan_types(graph: Graph, types: set[str]) -> None:
    for node in graph.nodes.values():
        if isinstance(node, RegionNode):
            _scan_types(node.body, types)
        else:
            types.add(node.node_type)


def graph_provider_selections(graph: Graph) -> tuple[set[tuple[str, str]], set[str]]:
    """Return literal provider selections and nodes with linked selectors.

    A literal selects one provider's artifact set. A link is resolved only
    during execution, so preflight must conservatively cover every installed
    provider for that node type. Invalid or missing literals remain graph
    validation's responsibility and do not invent an asset requirement.
    """
    selected: set[tuple[str, str]] = set()
    linked: set[str] = set()

    def scan(current: Graph) -> None:
        for node in current.nodes.values():
            if isinstance(node, RegionNode):
                scan(node.body)
                continue
            value = node.inputs.get("provider")
            if isinstance(value, TypedLiteral):
                value = value.value
            if isinstance(value, str):
                selected.add((node.node_type, value))
            elif isinstance(value, Link):
                linked.add(node.node_type)
            elif isinstance(node.inputs.get("model"), Link):
                linked.add(node.node_type)

    scan(graph)
    return selected, linked


def _scan_inputs(inputs: Mapping[str, Link | object], found: dict[str, str]) -> None:
    for value in inputs.values():
        if isinstance(value, Link):
            continue
        # A typed literal is a stamped literal: the asset descriptor (if
        # any) lives in its value, so preflight sees through the stamp.
        if isinstance(value, TypedLiteral):
            value = value.value
        _scan_literal(value, found)


def _scan_literal(value: object, found: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[str, object]", value)
        digest = mapping.get("digest")
        if isinstance(digest, str) and is_digest(digest):
            name = mapping.get("name")
            found[digest] = name if isinstance(name, str) and name else digest[:15] + "..."
        else:
            for child in mapping.values():
                _scan_literal(child, found)
        return
    if isinstance(value, list):
        for element in cast("list[object]", value):
            _scan_literal(element, found)


def parse_asset_consent(
    body: Mapping[str, Any],
) -> tuple[set[str], dict[str, list[str]]] | str:
    """Decode the submit body's consent fields, or return an error string.

    ``acquireAssets``: digests the caller explicitly authorizes acquiring.
    ``assetSources``: digest -> candidate http(s) URLs, submission-supplied
    leads on top of the provenance store's."""
    consent_raw = body.get("acquireAssets", [])
    if not isinstance(consent_raw, list) or not all(
        isinstance(entry, str) and is_digest(entry) for entry in cast("list[object]", consent_raw)
    ):
        return "'acquireAssets' must be a list of canonical asset digests"
    sources_raw = body.get("assetSources", {})
    if not isinstance(sources_raw, dict):
        return "'assetSources' must be an object mapping digests to URL lists"
    sources: dict[str, list[str]] = {}
    for key, value in cast("dict[object, object]", sources_raw).items():
        if not isinstance(key, str) or not is_digest(key):
            return "'assetSources' keys must be canonical asset digests"
        if not isinstance(value, list) or not all(
            isinstance(url, str) and url.startswith(("http://", "https://"))
            for url in cast("list[object]", value)
        ):
            return "'assetSources' values must be lists of http(s) URLs"
        sources[key] = cast("list[str]", value)
    return set(cast("list[str]", consent_raw)), sources


def asset_preflight(
    library: ServerLibrary,
    referenced: Mapping[str, str],
    consented: set[str],
    hinted_sources: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    """Resolve every referenced identity or say exactly why not.

    Returns the acquisition plan for identities that are still missing
    after consented acquisitions - empty means the job may run. Runs in a
    worker thread: consented acquisitions may download for a while.

    Pack declarations ([[pack.assets]]) join here: when the composed
    surface's catalog knows a referenced digest, its declared sources -
    packaged files inside installed packs, remote mirrors - merge with
    the submission's own leads, and packaged acquisition resolves against
    the catalog's pack roots. Every path stays verified and consented."""
    required = dict(referenced)
    pending = list(required)
    while pending:
        parent = pending.pop()
        dependencies = library.store.get_dependencies(parent)
        if dependencies is None:
            continue
        for dependency in dependencies:
            digest = cast(str, dependency["digest"])
            if digest in required:
                continue
            required[digest] = cast(str, dependency["resourceId"])
            pending.append(digest)
    catalog = library.pack_assets
    pack_roots = catalog.pack_roots() if catalog is not None else {}
    plan: list[dict[str, object]] = []
    for digest in sorted(required):
        if library.locate(digest) is not None:
            continue
        name = required[digest]
        declared = catalog.need_for(digest) if catalog is not None else None
        sources: list[AssetSource] = list(declared.sources) if declared else []
        for url in hinted_sources.get(digest, ()):
            if not any(
                isinstance(source, RemoteSource) and source.url == url for source in sources
            ):
                sources.append(RemoteSource(url))
        need = AssetNeed(
            name=name,
            digest=digest,
            kind=declared.kind if declared is not None else "",
            size=declared.size if declared is not None else -1,
            media_type=declared.media_type if declared is not None else "",
            metadata=declared.metadata if declared is not None else {},
            component_manifest=declared.component_manifest if declared is not None else None,
            sources=tuple(sources),
        )
        if digest in consented:
            try:
                result = acquire_need(
                    need,
                    library.vault,
                    resolver=library.resolver,
                    pack_roots=pack_roots,
                    provenance=library.provenance,
                    public_sources=(
                        library.public_sources_for(digest)
                        if library.public_sources_for is not None
                        else ()
                    ),
                    receipts=library.receipts,
                    lan_resolve=library.lan_resolve,
                    materialized=library.p2p_acquired,
                )
            except AssetError as exc:  # defensive: acquire reports, not raises
                result = None
                detail = str(exc)
            else:
                detail = result.detail
            if result is not None and result.ok:
                continue
            plan.append(_plan_entry(library, digest, name, need, "failed", detail))
            continue
        plan.append(_plan_entry(library, digest, name, need, "missing", ""))
    return plan


def _plan_entry(
    library: ServerLibrary,
    digest: str,
    name: str,
    need: AssetNeed,
    status: str,
    detail: str,
) -> dict[str, object]:
    urls = [source.url for source in need.sources if isinstance(source, RemoteSource)]
    if library.provenance is not None:
        urls.extend(url for url in library.provenance.sources(digest) if url not in urls)
    packaged = sorted(
        {source.pack for source in need.sources if isinstance(source, PackagedSource)}
    )
    entry: dict[str, object] = {
        "digest": digest,
        "name": name,
        "status": status,
        "sources": urls,
        "fetchable": bool(urls) or bool(packaged) or library.lan_resolve is not None,
    }
    if need.kind:
        entry["kind"] = need.kind
    if need.size >= 0:
        entry["size"] = need.size
    if need.metadata:
        entry["metadata"] = dict(need.metadata)
    if need.component_manifest is not None:
        entry["components"] = need.component_manifest.to_wire()
    if packaged:
        # Consent UX: "ships with pack X" - acquisition copies out of the
        # installed artifact, no network involved.
        entry["packagedFrom"] = packaged
    if detail:
        entry["detail"] = detail
    return entry
