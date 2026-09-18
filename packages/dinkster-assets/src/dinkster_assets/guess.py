"""Name-based best-guess asset matching (ROADMAP "name-based best-guess
asset resolver"; the reserved tier from need.py and acquire.py).

Imported ComfyUI workflows name assets the only way ComfyUI can: by
filename, sometimes under a subfolder ("SD15\\model.safetensors"). Nobody
hashed those files, so such references can never be VERIFIED - digest
identity stays the only authority, and acquisition keeps refusing
digestless needs. This module is the lookup half of the compatibility
tier: given a legacy name and the entries a host can see, produce ranked
candidates that carry real digests. Acceptance stays with the caller (an
explicit user gesture rewrites the imported document with the chosen
digest); nothing here mutates state, downloads bytes, or silently
substitutes same-named content.

Confidence is a TIER, not a score - four falsifiable statements about
how the name relates to a candidate, ordered strongest first:

- ``path``: the query has a folder part and it matches the tail of the
  candidate's path, case-insensitively ("SD15/model.safetensors" against
  "mounts/checkpoints/SD15/model.safetensors").
- ``name``: the file names are identical.
- ``name-insensitive``: the file names differ only by case (the query
  usually came from a Windows install).
- ``stem``: the names differ only by extension ("model.ckpt" against
  "model.safetensors" - the file was re-encoded, a suggestion at best).

A numeric score would imply precision this tier cannot have; a tier
names exactly what the user is being asked to believe.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .catalog import AssetEntry

__all__ = [
    "GUESS_CONFIDENCE",
    "GuessCandidate",
    "guess_matches",
    "match_confidence",
    "normalize_guess_query",
]

GUESS_CONFIDENCE: tuple[str, ...] = ("path", "name", "name-insensitive", "stem")
"""Every confidence tier, strongest first. Wire values."""

_TIER_RANK = {tier: rank for rank, tier in enumerate(GUESS_CONFIDENCE)}


def normalize_guess_query(raw: str) -> str:
    """A legacy name as a comparable '/'-path: backslashes become '/',
    empty and '.' segments drop. Returns "" for a query with no usable
    segments (matches nothing)."""
    segments = [
        segment for segment in raw.replace("\\", "/").split("/") if segment not in ("", ".")
    ]
    if any(segment == ".." for segment in segments):
        return ""  # never resolvable: legacy names are plain subpaths
    return "/".join(segments)


def _stem(name: str) -> str:
    stem, _, _ = name.rpartition(".")
    return stem or name


def match_confidence(query: str, candidate: str) -> str | None:
    """The confidence tier relating a normalized query to a candidate
    '/'-path (or bare name), or None when they are unrelated. ``query``
    must already be normalized (see :func:`normalize_guess_query`)."""
    if not query or not candidate:
        return None
    query_segments = query.split("/")
    candidate_segments = candidate.split("/")
    query_name = query_segments[-1]
    candidate_name = candidate_segments[-1]
    if len(query_segments) > 1 and len(candidate_segments) >= len(query_segments):
        tail = candidate_segments[-len(query_segments) :]
        if [s.lower() for s in tail] == [s.lower() for s in query_segments]:
            return "path"
    if candidate_name == query_name:
        return "name"
    if candidate_name.lower() == query_name.lower():
        return "name-insensitive"
    if _stem(candidate_name).lower() == _stem(query_name).lower():
        return "stem"
    return None


@dataclass(frozen=True)
class GuessCandidate:
    """One ranked match: a real identity plus where the name led. The
    digest is what an accepting client writes into the document; the rest
    is display data for the acceptance UX."""

    confidence: str
    digest: str
    name: str
    virtual_path: str = ""
    size: int = -1
    media_type: str = ""


def guess_matches(raw_query: str, entries: Iterable[AssetEntry]) -> tuple[GuessCandidate, ...]:
    """Rank catalog entries against one legacy name. Deterministic:
    confidence tier first, virtual path as the tiebreak. One candidate
    per digest - aliases of the same bytes are the same answer, and the
    best-ranked path represents it."""
    query = normalize_guess_query(raw_query)
    if not query:
        return ()
    scored: list[tuple[int, str, AssetEntry]] = []
    for entry in entries:
        confidence = match_confidence(query, entry.virtual_path)
        if confidence is not None:
            scored.append((_TIER_RANK[confidence], entry.virtual_path, entry))
    scored.sort(key=lambda row: (row[0], row[1]))
    seen: set[str] = set()
    out: list[GuessCandidate] = []
    for rank, _, entry in scored:
        if entry.digest in seen:
            continue
        seen.add(entry.digest)
        out.append(
            GuessCandidate(
                confidence=GUESS_CONFIDENCE[rank],
                digest=entry.digest,
                name=entry.name,
                virtual_path=entry.virtual_path,
                size=entry.size,
                media_type=entry.media_type,
            )
        )
    return tuple(out)
