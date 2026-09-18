"""Registry model: versions, artifacts, releases, review lifecycle.

The pure half of the registry - data shapes and invariants with no server,
storage, or network. Everything here is computable on both sides of the
trust boundary: a publisher can run the same checks locally that the
registry runs at admission, so "publishable" and "accepted" can never
drift apart (the doctor's one-predicate-two-enforcement-points property,
scaled up to distribution).

Identity decisions, each answering a recorded ComfyUI failure:

- Pack names, publisher ids, and namespace claims share the one closed
  grammar (``dinkster_schema.names``); uniqueness is ``canonical_name``.
- Release versions use a closed numeric ``major.minor.patch`` grammar (no
  leading zeros). Prerelease tags and channels are a deliberate later
  addition, never a reinterpretation.
- Release artifacts are identified by ``sha256:<64 hex>`` - computable
  with the stdlib everywhere, and deliberately a *different* digest space
  from assets (``blake3:``, DESIGN 3.12) so an artifact digest can never
  be mistaken for an asset digest or vice versa.
- A (pack, version) release is immutable: once published, its digest can
  never change. Re-publishing identical bytes is idempotent; different
  bytes under an existing version are refused, never silently versioned
  over. The admission verdict binds to this digest.
- Review is a state machine whose every transition carries an actor and,
  wherever judgment was involved, a reason - "under review" with no cause
  and no recourse is unrepresentable. Deadlines and escalation are server
  policy layered on these states, not modeled here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal


class RegistryError(Exception):
    """A registry-model invariant was violated."""


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------


def canonical_json(payload: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII escapes.

    The registry digests metadata records, so two encodings of one record
    must be one byte sequence - key order and formatting can never leak
    into identity.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Release versions (closed grammar, grows additively)
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def validate_version(text: str) -> str | None:
    """The problem with ``text`` as a release version, or None."""
    if not _VERSION_RE.fullmatch(text):
        return (
            "must be 'major.minor.patch' with decimal numbers and no "
            "leading zeros (e.g. '1.0.0', '0.4.12')"
        )
    return None


@dataclass(frozen=True, order=True)
class Version:
    """A parsed release version; ordering is (major, minor, patch)."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> Version:
        problem = validate_version(text)
        if problem is not None:
            raise RegistryError(f"invalid version {text!r}: {problem}")
        major, minor, patch = (int(part) for part in text.split("."))
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


# ---------------------------------------------------------------------------
# Artifact digests
# ---------------------------------------------------------------------------

ARTIFACT_DIGEST_PREFIX = "sha256:"
_ARTIFACT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_artifact_digest(text: str) -> str | None:
    """The problem with ``text`` as a release artifact digest, or None."""
    if not _ARTIFACT_DIGEST_RE.fullmatch(text):
        return "must be 'sha256:<64 lowercase hex>' computed over the artifact bytes"
    return None


def artifact_digest(data: bytes) -> str:
    """The canonical digest of release artifact bytes."""
    return ARTIFACT_DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Releases (immutable once published)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseTemplate:
    """One template descriptor recorded at publish: browse metadata the
    registry's OWN probe read from the artifact bytes.

    The registry never re-opens artifacts to answer browse queries -
    descriptors ride the immutable release record, exactly like node
    types. ``digest`` (``sha256:<hex>`` over the document bytes) is the
    same immutability contract the local server's /api/templates speaks;
    ``path`` is the POSIX member path inside the release artifact, an
    internal serving locator that never rides any wire descriptor.
    ``assets`` are pack-local ``[[pack.assets]]`` ids, passed through as
    ids exactly as the composed server passes them."""

    id: str
    name: str
    digest: str
    path: str
    description: str = ""
    tags: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        """The canonical record payload - one serializer, shared by the
        release record and the store's submission persistence, so the
        two can never drift."""
        return {
            "id": self.id,
            "name": self.name,
            "digest": self.digest,
            "path": self.path,
            "description": self.description,
            "tags": list(self.tags),
            "assets": list(self.assets),
        }


@dataclass(frozen=True)
class Release:
    """One published (pack, version): what the registry's own probe saw.

    ``claims`` and ``node_types`` are recorded from admission - the
    registry indexes what its sandboxed doctor probe actually loaded from
    the artifact bytes, never what a manifest asserted
    (manifests claim, the registry grants).
    """

    pack: str
    """Canonical pack name (``canonical_name`` form)."""
    version: str
    """Canonical version string (``str(Version.parse(...))``)."""
    artifact_digest: str
    publisher: str
    """Canonical publisher id."""
    claims: tuple[str, ...]
    """Canonical namespace claims in force for this release."""
    node_types: tuple[str, ...]
    """Node types the admission probe loaded from the artifact."""
    templates: tuple[ReleaseTemplate, ...] = ()
    """Template descriptors the probe read from the artifact's manifest -
    the registry's browse surface for packs nobody has installed. Older
    records simply lack the field and rehydrate empty."""

    def record_json(self) -> str:
        return canonical_json(
            {
                "pack": self.pack,
                "version": self.version,
                "artifactDigest": self.artifact_digest,
                "publisher": self.publisher,
                "claims": list(self.claims),
                "nodeTypes": list(self.node_types),
                "templates": [template.record() for template in self.templates],
            }
        )

    def record_digest(self) -> str:
        """Digest of the release *metadata* record (not the artifact) -
        the stable key for signed metadata and mirror verification."""
        return ARTIFACT_DIGEST_PREFIX + hashlib.sha256(self.record_json().encode()).hexdigest()


class ReleaseIndex:
    """All published releases, enforcing (pack, version) immutability."""

    def __init__(self, releases: tuple[Release, ...] = ()) -> None:
        self._by_key: dict[tuple[str, str], Release] = {}
        for release in releases:
            self.add(release)

    def get(self, pack: str, version: str) -> Release | None:
        return self._by_key.get((pack, version))

    def add(self, release: Release) -> Release:
        """Record a release. Identical re-publication is idempotent (same
        bytes -> same verdict); a different digest under an existing
        (pack, version) is refused - published versions never mutate."""
        key = (release.pack, release.version)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing.artifact_digest == release.artifact_digest:
                return existing
            raise RegistryError(
                f"release {release.pack} {release.version} is already published "
                f"with digest {existing.artifact_digest}; publishing different "
                f"bytes requires a new version"
            )
        self._by_key[key] = release
        return release

    def releases(self) -> tuple[Release, ...]:
        return tuple(self._by_key.values())


# ---------------------------------------------------------------------------
# Repository bindings (provenance metadata, never identity)
# ---------------------------------------------------------------------------


class RepoBindings:
    """Verified repository -> pack bindings, keyed by immutable repo id.

    The key is the hosting platform's immutable repository id (e.g. a
    GitHub numeric id), never the URL string - renames and mirrors cannot
    fork identity, and nobody can point their pack at someone else's repo
    for borrowed credibility. Verification itself (OAuth/App
    or challenge file) is server machinery; this model only refuses the
    representable corruption: one repo claiming to be two packs. URLs are
    display metadata and key nothing.
    """

    def __init__(self) -> None:
        self._by_repo: dict[str, str] = {}

    def bind(self, repo_id: str, pack: str) -> None:
        if not repo_id:
            raise RegistryError("repo binding requires an immutable repository id")
        existing = self._by_repo.get(repo_id)
        if existing is not None and existing != pack:
            raise RegistryError(
                f"repository {repo_id!r} is already bound to pack {existing!r}; "
                f"a repository is one pack's provenance, never two"
            )
        self._by_repo[repo_id] = pack

    def pack_for(self, repo_id: str) -> str | None:
        return self._by_repo.get(repo_id)


# ---------------------------------------------------------------------------
# Review lifecycle (a state machine, never an opaque status)
# ---------------------------------------------------------------------------

VersionState = Literal["submitted", "accepted", "needs_review", "rejected", "yanked"]

_TRANSITIONS: dict[VersionState, frozenset[VersionState]] = {
    "submitted": frozenset({"accepted", "needs_review", "rejected"}),
    "needs_review": frozenset({"accepted", "rejected"}),
    "rejected": frozenset({"needs_review"}),  # appeal: back into visible review
    "accepted": frozenset({"yanked"}),  # post-acceptance revocation
    "yanked": frozenset(),
}

_REASON_REQUIRED: frozenset[VersionState] = frozenset({"needs_review", "rejected", "yanked"})
"""States a human judgment produces must carry the judgment's reason;
deterministic acceptance is its own evidence (the doctor report)."""


@dataclass(frozen=True)
class ReviewTransition:
    """One lifecycle step: where to, who did it, why, and when."""

    to_state: VersionState
    actor: str
    reason: str
    at: str
    """Caller-supplied timestamp (ISO 8601) - the model owns no clock."""


@dataclass(frozen=True)
class ReviewLog:
    """Append-only lifecycle history for one submitted version.

    The log IS the audit trail: every state a version was ever in, with
    actor attribution and reasons, immutable by construction. "Pending
    with no visible cause" cannot be represented - entering needs_review
    without a reason is refused.
    """

    transitions: tuple[ReviewTransition, ...] = field(default_factory=tuple)

    @classmethod
    def start(cls, actor: str, at: str) -> ReviewLog:
        if not actor:
            raise RegistryError("review transitions require an actor")
        return cls((ReviewTransition("submitted", actor, "", at),))

    @property
    def state(self) -> VersionState:
        if not self.transitions:
            raise RegistryError("review log has no transitions; use ReviewLog.start()")
        return self.transitions[-1].to_state

    def advance(self, to_state: VersionState, actor: str, at: str, reason: str = "") -> ReviewLog:
        if not actor:
            raise RegistryError("review transitions require an actor")
        allowed = _TRANSITIONS[self.state]
        if to_state not in allowed:
            raise RegistryError(
                f"cannot move a version from {self.state!r} to {to_state!r}; "
                f"allowed: {sorted(allowed) or 'none (terminal state)'}"
            )
        if to_state in _REASON_REQUIRED and not reason:
            raise RegistryError(
                f"entering {to_state!r} requires an explicit reason - opaque "
                f"'under review' states are exactly what this model exists to prevent"
            )
        return ReviewLog((*self.transitions, ReviewTransition(to_state, actor, reason, at)))


__all__ = [
    "ARTIFACT_DIGEST_PREFIX",
    "RegistryError",
    "Release",
    "ReleaseIndex",
    "ReleaseTemplate",
    "RepoBindings",
    "ReviewLog",
    "ReviewTransition",
    "Version",
    "VersionState",
    "artifact_digest",
    "canonical_json",
    "validate_artifact_digest",
    "validate_version",
]
