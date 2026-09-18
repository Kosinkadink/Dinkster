"""Name-guess API: best-guess candidates for legacy asset names, with an
optional exact-digest lookup for names whose identity is already known
(ROADMAP "name-based best-guess asset resolver").

Umbrella-owned wiring like mounts_api: the corpus is composed here -
mount catalogs (files this host has actually hashed) plus pack-declared
assets ([[pack.assets]] names, whose digests exist before bytes do) plus
subscribed resolver indexes - and dinkster-server stays guess-agnostic.

- POST /api/assets/guess   {"names": ["SD15\\model.safetensors", ...],
                            "digestHints": {"SD15\\model.safetensors":
                            "blake3:<64 lowercase hex>"}}
                           -> {"matches": [{"query": ..., "candidates":
                           [...]}]} in request order

Unhinted names preserve the existing confidence tiers (see
dinkster_assets.guess: path > name > name-insensitive > stem) across local
catalogs and pack declarations. A hinted name instead returns only ready,
authorized local mount entries with that exact digest, confidence ``digest``,
and no name fallback. Each cataloged candidate carries ``virtualPath`` /
``mountId`` and ``held`` true; unhinted declared candidates carry
``declaredBy`` and whether their bytes are held locally.

This endpoint is a LOOKUP, deliberately read-only: the digest tier
stays the only authority, acquisition keeps refusing digestless needs,
and accepting a guess is an explicit client gesture that rewrites the
imported document with the chosen digest. Registered unconditionally -
an empty composition answers with empty candidate lists, never a 404
the client must special-case.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

from aiohttp import web
from dinkster_assets import (
    MOUNT_NAMESPACE,
    AssetEntry,
    DeclaredAsset,
    GuessCandidate,
    MountsError,
    PackagedSource,
    ResolverSuggestion,
    guess_matches,
    is_digest,
    match_confidence,
    normalize_guess_query,
)
from dinkster_server import LIBRARY_KEY

from .mounts_api import MOUNTS_KEY
from .resolver_api import RESOLVER_INDEXES_KEY

__all__ = ["add_guess_routes"]

_NAMES_MAX = 64
"""One imported workflow's worth of references, not a bulk-scan API."""

_NAME_LENGTH_MAX = 1024

_CANDIDATES_MAX = 10
"""Per name. An acceptance UX showing more than this is a search box,
and the mounts entries endpoint already is one."""

_TIER_RANK = {"path": 0, "name": 1, "name-insensitive": 2, "stem": 3}


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _mount_id(virtual_path: str) -> str:
    """The mount a cataloged path belongs to: ``mounts/<id>/...``."""
    segments = virtual_path.split("/")
    if len(segments) >= 2 and segments[0] == MOUNT_NAMESPACE:
        return segments[1]
    return ""


def _cataloged_wire(candidate: GuessCandidate) -> dict[str, object]:
    wire: dict[str, object] = {
        "digest": candidate.digest,
        "name": candidate.name,
        "confidence": candidate.confidence,
        "virtualPath": candidate.virtual_path,
        "held": True,
    }
    mount_id = _mount_id(candidate.virtual_path)
    if mount_id:
        wire["mountId"] = mount_id
    if candidate.size >= 0:
        wire["size"] = candidate.size
    if candidate.media_type:
        wire["mediaType"] = candidate.media_type
    return wire


def _declared_confidence(query: str, declared: DeclaredAsset) -> str | None:
    """Best tier across everything a declaration names: the display name
    and each packaged file path (the latter is a real filename, often the
    exact string a legacy workflow used)."""
    best: str | None = None
    candidates = [normalize_guess_query(declared.need.name)]
    candidates.extend(
        source.path for source in declared.need.sources if isinstance(source, PackagedSource)
    )
    for candidate in candidates:
        tier = match_confidence(query, candidate)
        if tier is not None and (best is None or _TIER_RANK[tier] < _TIER_RANK[best]):
            best = tier
    return best


def _declared_wire(
    declared: DeclaredAsset, confidence: str, packs: list[str], held: bool
) -> dict[str, object]:
    wire: dict[str, object] = {
        "digest": declared.need.digest,
        "name": declared.need.name,
        "confidence": confidence,
        "declaredBy": sorted(packs),
        "held": held,
    }
    if declared.need.kind:
        wire["kind"] = declared.need.kind
    if declared.need.size >= 0:
        wire["size"] = declared.need.size
    if declared.need.media_type:
        wire["mediaType"] = declared.need.media_type
    return wire


def _resolver_wire(suggestion: ResolverSuggestion, held: bool) -> dict[str, object]:
    wire: dict[str, object] = {
        "digest": suggestion.digest,
        "name": suggestion.name,
        "confidence": "name",
        "resolverIndex": suggestion.source_id,
        "resolverSource": suggestion.source,
        "held": held,
    }
    if suggestion.index_name:
        wire["resolverName"] = suggestion.index_name
    if suggestion.kind:
        wire["kind"] = suggestion.kind
    if suggestion.size >= 0:
        wire["size"] = suggestion.size
    if suggestion.component_manifest is not None:
        wire["components"] = suggestion.component_manifest.to_wire()
    return wire


def _guess_one(request: web.Request, raw_query: str) -> list[dict[str, object]]:
    """Ranked candidate wires for one legacy name. Blocking (mount table
    locks, vault stats) - the handler runs it on a thread."""
    query = normalize_guess_query(raw_query)
    if not query:
        return []
    ranked: list[tuple[int, int, str, dict[str, object]]] = []
    taken: set[str] = set()
    service = request.app.get(MOUNTS_KEY)
    if service is not None:
        entries: list[AssetEntry] = []
        for row in service.table.descriptors():
            try:
                entries.extend(service.table.entries(str(row["id"])))
            except MountsError:
                continue  # revoked between the listing and this read
        for candidate in guess_matches(raw_query, entries):
            taken.add(candidate.digest)
            ranked.append(
                (
                    _TIER_RANK[candidate.confidence],
                    0,  # held bytes outrank a declaration on ties
                    candidate.virtual_path,
                    _cataloged_wire(candidate),
                )
            )
    library = request.app.get(LIBRARY_KEY)
    catalog = library.pack_assets if library is not None else None
    if catalog is not None:
        matched: dict[str, tuple[DeclaredAsset, str, list[str]]] = {}
        for pack_id, assets in catalog.all_declared().items():
            for declared in assets:
                if declared.need.digest in taken:
                    continue
                tier = _declared_confidence(query, declared)
                if tier is None:
                    continue
                digest = declared.need.digest
                previous = matched.get(digest)
                if previous is None:
                    matched[digest] = (declared, tier, [pack_id])
                else:
                    best = min((previous[1], tier), key=lambda t: _TIER_RANK[t])
                    matched[digest] = (previous[0], best, [*previous[2], pack_id])
        for digest, (declared, tier, packs) in matched.items():
            held = library is not None and library.locate(digest) is not None
            taken.add(digest)
            ranked.append(
                (
                    _TIER_RANK[tier],
                    1,
                    declared.need.name,
                    _declared_wire(declared, tier, packs, held),
                )
            )
    resolver_indexes = request.app.get(RESOLVER_INDEXES_KEY)
    if resolver_indexes is not None:
        for suggestion in resolver_indexes.suggest(query):
            if suggestion.digest in taken:
                continue
            taken.add(suggestion.digest)
            held = library is not None and library.locate(suggestion.digest) is not None
            ranked.append(
                (
                    _TIER_RANK["name"],
                    2,
                    suggestion.name,
                    _resolver_wire(suggestion, held),
                )
            )
    ranked.sort(key=lambda row: row[:3])
    return [wire for _, _, _, wire in ranked[:_CANDIDATES_MAX]]


def _digest_candidates(request: web.Request, digest: str) -> list[dict[str, object]]:
    """Exact-digest references from ready authorized local mounts only."""
    service = request.app.get(MOUNTS_KEY)
    if service is None:
        return []
    ranked: list[tuple[int, str, str, AssetEntry]] = []
    for descriptor in service.table.descriptors():
        if descriptor["state"] != "ready":
            continue
        mount_id = str(descriptor["id"])
        try:
            entries = service.table.entries(mount_id)
        except MountsError:
            continue  # revoked between the descriptor and catalog reads
        for entry in entries:
            if entry.digest == digest:
                ranked.append(
                    (
                        cast("int", descriptor["priority"]),
                        mount_id,
                        entry.virtual_path,
                        entry,
                    )
                )
    ranked.sort(key=lambda row: row[:3])
    return [
        _cataloged_wire(
            GuessCandidate(
                confidence="digest",
                digest=entry.digest,
                name=entry.name,
                virtual_path=entry.virtual_path,
                size=entry.size,
                media_type=entry.media_type,
            )
        )
        for _, _, _, entry in ranked[:_CANDIDATES_MAX]
    ]


async def handle_assets_guess(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
    except json.JSONDecodeError as exc:
        return _json_error(400, f"invalid JSON: {exc}")
    if not isinstance(raw, dict):
        return _json_error(400, "request body must be an object")
    unknown = set(raw) - {"names", "digestHints"}
    if unknown:
        return _json_error(400, f"unknown request fields: {sorted(unknown)}")
    names_raw = raw.get("names")
    if (
        not isinstance(names_raw, list)
        or not names_raw
        or not all(
            isinstance(name, str) and name.strip() for name in cast("list[object]", names_raw)
        )
    ):
        return _json_error(400, "'names' must be a non-empty list of strings")
    names = cast("list[str]", names_raw)
    if len(names) > _NAMES_MAX:
        return _json_error(400, f"'names' accepts at most {_NAMES_MAX} entries")
    if any(len(name) > _NAME_LENGTH_MAX for name in names):
        return _json_error(400, f"names must be at most {_NAME_LENGTH_MAX} characters")
    hints_raw = raw.get("digestHints", {})
    if not isinstance(hints_raw, dict):
        return _json_error(400, "'digestHints' must be an object")
    hints = cast("dict[object, object]", hints_raw)
    if len(hints) > _NAMES_MAX:
        return _json_error(400, f"'digestHints' accepts at most {_NAMES_MAX} entries")
    if any(not isinstance(name, str) or name not in names for name in hints):
        return _json_error(400, "'digestHints' keys must exactly match a requested name")
    if any(not isinstance(digest, str) or not is_digest(digest) for digest in hints.values()):
        return _json_error(400, "'digestHints' values must be canonical lowercase BLAKE3 digests")
    digest_hints = cast("dict[str, str]", hints)

    def resolve_all() -> list[dict[str, object]]:
        return [
            {
                "query": name,
                "candidates": (
                    _digest_candidates(request, digest_hints[name])
                    if name in digest_hints
                    else _guess_one(request, name)
                ),
            }
            for name in names
        ]

    matches = await asyncio.to_thread(resolve_all)
    return web.json_response({"matches": matches})


def add_guess_routes(app: web.Application) -> None:
    app.router.add_post("/api/assets/guess", handle_assets_guess)
