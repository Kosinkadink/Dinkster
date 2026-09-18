"""Torch-free canonical distributed-execution manifest and consensus.

One distributed invocation has exactly one canonical manifest.
Subsystems contribute compiled-plan slots; no subsystem mints a
parallel identity scheme. Ordering is enforced by construction:

1. a ``CompiledPlanSlot`` can only be built from an already-compiled
   plan's canonical facts, so plan compilation precedes the freeze;
2. ``CanonicalManifest`` is immutable - construction is the freeze;
3. ``prove_manifest_consensus`` is the only mint of
   ``ManifestConsensusToken``, and collective issue points must
   demand that token, so consensus precedes every collective.

Every rank derives the manifest independently and the group proves
sha256 digest equality before any collective. A rank that observes a
differing digest refuses; there is no majority override.
"""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from enum import StrEnum
from typing import Protocol

__all__ = [
    "CanonicalManifest",
    "CompiledPlanSlot",
    "ConsensusTransport",
    "ManifestConsensusToken",
    "ManifestRefusal",
    "ManifestRefusalCode",
    "build_canonical_manifest",
    "prove_manifest_consensus",
]

_SLOT_DOMAIN = "dinkster.manifest.slot.v1"
_MANIFEST_DOMAIN = "dinkster.manifest.v1"


class ManifestRefusalCode(StrEnum):
    INVALID_SLOT = "invalid-slot"
    DUPLICATE_SLOT = "duplicate-slot"
    INVALID_MANIFEST = "invalid-manifest"
    CONSENSUS_GROUP_INVALID = "consensus-group-invalid"
    CONSENSUS_DIGEST_MISMATCH = "consensus-digest-mismatch"
    FORGED_CONSENSUS_TOKEN = "forged-consensus-token"


class ManifestRefusal(ValueError):
    def __init__(self, code: ManifestRefusalCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _sha256_lines(lines: tuple[str, ...]) -> str:
    """sha256 of the exact preimage: every line followed by one LF."""
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _require_fact(code: ManifestRefusalCode, name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise ManifestRefusal(code, f"{name} must be a non-empty string")
    if "\n" in value:
        raise ManifestRefusal(code, f"{name} must not contain newlines")


def _require_sha256(code: ManifestRefusalCode, name: str, digest: str) -> None:
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ManifestRefusal(code, f"{name} must be a lowercase sha256 hex digest")


def _require_runtime_identity(value: str) -> None:
    _require_fact(ManifestRefusalCode.INVALID_MANIFEST, "runtime identity", value)
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != "native" or not parts[1]:
        raise ManifestRefusal(
            ManifestRefusalCode.INVALID_MANIFEST,
            "runtime identity must follow the native:<family>:<sha256> convention",
        )
    _require_sha256(ManifestRefusalCode.INVALID_MANIFEST, "runtime identity digest", parts[2])


@dataclass(frozen=True, slots=True)
class CompiledPlanSlot:
    """One subsystem's compiled plan, frozen as canonical facts.

    A slot exists only downstream of plan compilation: its facts are
    the compiled plan's canonical serialization, so every fail-closed
    compile check has already run by the time a slot value exists.
    """

    slot: str
    facts: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_fact(ManifestRefusalCode.INVALID_SLOT, "slot name", self.slot)
        if type(self.facts) is not tuple or not self.facts:
            raise ManifestRefusal(
                ManifestRefusalCode.INVALID_SLOT,
                f"slot {self.slot!r} must carry at least one canonical fact",
            )
        for fact in self.facts:
            _require_fact(ManifestRefusalCode.INVALID_SLOT, f"slot {self.slot!r} fact", fact)

    @property
    def digest(self) -> str:
        return _sha256_lines((_SLOT_DOMAIN, f"slot={self.slot}", *self.facts))


def _require_compiled_slots(slots: tuple[CompiledPlanSlot, ...]) -> None:
    if type(slots) is not tuple:
        raise ManifestRefusal(ManifestRefusalCode.INVALID_MANIFEST, "slots must be a tuple")
    for slot in slots:
        if type(slot) is not CompiledPlanSlot:
            raise ManifestRefusal(
                ManifestRefusalCode.INVALID_MANIFEST,
                "slots must contain only CompiledPlanSlot values",
            )


@dataclass(frozen=True, slots=True)
class CanonicalManifest:
    """The one canonical manifest of a distributed invocation.

    Immutable by construction; building it is the manifest freeze.
    ``slots`` is canonically ordered by ascending slot name, never by
    contribution arrival order. ``rank_plan_digests`` carries every
    rank's expected rank-local compiled-plan digests, indexed by
    rank, so consensus proves each rank compiled the same plans, not
    merely that each rank has some plan. The group size is
    ``len(rank_plan_digests)``.
    """

    runtime_identity: str
    invocation_facts: tuple[str, ...]
    slots: tuple[CompiledPlanSlot, ...]
    rank_plan_digests: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require_runtime_identity(self.runtime_identity)
        if type(self.invocation_facts) is not tuple or not self.invocation_facts:
            raise ManifestRefusal(
                ManifestRefusalCode.INVALID_MANIFEST,
                "a manifest must carry at least one invocation fact",
            )
        for fact in self.invocation_facts:
            _require_fact(ManifestRefusalCode.INVALID_MANIFEST, "invocation fact", fact)
        _require_compiled_slots(self.slots)
        names = [slot.slot for slot in self.slots]
        if names != sorted(names):
            raise ManifestRefusal(
                ManifestRefusalCode.INVALID_MANIFEST,
                "slots must be in canonical ascending slot-name order",
            )
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ManifestRefusal(
                ManifestRefusalCode.DUPLICATE_SLOT,
                f"duplicate slot contribution: {', '.join(duplicates)}",
            )
        if type(self.rank_plan_digests) is not tuple or not self.rank_plan_digests:
            raise ManifestRefusal(
                ManifestRefusalCode.INVALID_MANIFEST,
                "a manifest must carry expected plan digests for at least one rank",
            )
        for rank, digests in enumerate(self.rank_plan_digests):
            if type(digests) is not tuple or not digests:
                raise ManifestRefusal(
                    ManifestRefusalCode.INVALID_MANIFEST,
                    f"rank {rank} must carry at least one expected plan digest",
                )
            for digest in digests:
                _require_sha256(
                    ManifestRefusalCode.INVALID_MANIFEST,
                    f"rank {rank} expected plan digest",
                    digest,
                )

    @property
    def group_size(self) -> int:
        return len(self.rank_plan_digests)

    def manifest_lines(self) -> tuple[str, ...]:
        """Exact digest preimage, line by line.

        ``digest`` is the sha256 of exactly these lines, each followed
        by one LF; the versioned domain tag is the first line.
        """
        lines = [_MANIFEST_DOMAIN, f"runtime_identity={self.runtime_identity}"]
        lines.extend(f"fact={fact}" for fact in self.invocation_facts)
        lines.extend(f"slot={slot.slot}:{slot.digest}" for slot in self.slots)
        for rank, digests in enumerate(self.rank_plan_digests):
            lines.extend(f"rank[{rank}].plan={digest}" for digest in digests)
        return tuple(lines)

    @property
    def digest(self) -> str:
        return _sha256_lines(self.manifest_lines())


def build_canonical_manifest(
    *,
    runtime_identity: str,
    invocation_facts: tuple[str, ...],
    slots: tuple[CompiledPlanSlot, ...],
    rank_plan_digests: tuple[tuple[str, ...], ...],
) -> CanonicalManifest:
    """Freeze the canonical manifest from compiled contributions.

    Slot contributions may arrive in any order; the manifest always
    stores them in canonical ascending slot-name order, so every rank
    derives byte-identical manifest lines from equal facts.
    """
    _require_compiled_slots(slots)
    ordered = tuple(sorted(slots, key=lambda slot: slot.slot))
    return CanonicalManifest(
        runtime_identity=runtime_identity,
        invocation_facts=invocation_facts,
        slots=ordered,
        rank_plan_digests=rank_plan_digests,
    )


class ConsensusTransport(Protocol):
    """Transport seam for the pre-collective digest all-gather.

    Implementations must return every rank's submitted digest indexed
    by rank, blocking until the whole group has submitted. The seam is
    deliberately digest-only: no plan content crosses the transport,
    so any byte-faithful channel (CPU/Gloo store, TCP, in-memory test
    group) can carry it without a torch dependency here.
    """

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]: ...


_MINT_AUTHORITY: object = object()


@dataclass(frozen=True, slots=True)
class ManifestConsensusToken:
    """Proof that the group agreed on one manifest digest.

    Only ``prove_manifest_consensus`` mints this token: construction
    demands the module-private mint authority and refuses anything
    else, so a collective issue point that requires the token makes
    "consensus precedes collectives" a data dependency rather than a
    calling convention.
    """

    manifest_digest: str
    group_size: int
    rank: int
    mint_authority: InitVar[object]

    def __post_init__(self, mint_authority: object) -> None:
        if mint_authority is not _MINT_AUTHORITY:
            raise ManifestRefusal(
                ManifestRefusalCode.FORGED_CONSENSUS_TOKEN,
                "consensus tokens are minted only by prove_manifest_consensus",
            )


def prove_manifest_consensus(
    manifest: CanonicalManifest,
    *,
    rank: int,
    transport: ConsensusTransport,
) -> ManifestConsensusToken:
    """Prove group-wide manifest digest equality; refuse on any mismatch.

    Every rank calls this with its independently derived manifest.
    The group's digests are gathered over the transport and must all
    equal the local digest; a rank observing any difference refuses.
    There is no majority override: one divergent rank fails the whole
    group before any collective is issued.
    """
    if type(manifest) is not CanonicalManifest:
        raise ManifestRefusal(
            ManifestRefusalCode.INVALID_MANIFEST,
            "consensus is proven only over an exact CanonicalManifest value",
        )
    group_size = manifest.group_size
    if type(rank) is not int or not 0 <= rank < group_size:
        raise ManifestRefusal(
            ManifestRefusalCode.CONSENSUS_GROUP_INVALID,
            f"rank {rank!r} is not a member of a group of {group_size}",
        )
    local_digest = manifest.digest
    gathered = tuple(transport.exchange_digest(rank, local_digest))
    if len(gathered) != group_size:
        raise ManifestRefusal(
            ManifestRefusalCode.CONSENSUS_GROUP_INVALID,
            f"transport gathered {len(gathered)} digests for a group of {group_size}",
        )
    for peer, digest in enumerate(gathered):
        _require_sha256(
            ManifestRefusalCode.CONSENSUS_GROUP_INVALID, f"rank {peer} gathered digest", digest
        )
    if gathered[rank] != local_digest:
        raise ManifestRefusal(
            ManifestRefusalCode.CONSENSUS_GROUP_INVALID,
            f"transport returned a foreign digest at local rank {rank}",
        )
    divergent = tuple(peer for peer, digest in enumerate(gathered) if digest != local_digest)
    if divergent:
        raise ManifestRefusal(
            ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH,
            "manifest digest disagreement at rank"
            f" {rank}: divergent ranks {', '.join(str(peer) for peer in divergent)}",
        )
    return ManifestConsensusToken(
        manifest_digest=local_digest,
        group_size=group_size,
        rank=rank,
        mint_authority=_MINT_AUTHORITY,
    )
