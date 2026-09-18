from __future__ import annotations

import hashlib

from dinkster_inference import (
    ManifestConsensusToken,
    build_canonical_manifest,
    prove_manifest_consensus,
)


class _AgreeingTransport:
    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        del rank
        return (digest,) * self.group_size


def minted_consensus_token(group_size: int, rank: int) -> ManifestConsensusToken:
    rank_plan_digests = tuple(
        (hashlib.sha256(f"test-rank-plan:{peer}".encode()).hexdigest(),)
        for peer in range(group_size)
    )
    manifest = build_canonical_manifest(
        runtime_identity=f"native:test:{'0' * 64}",
        invocation_facts=("test=sequence-exchange",),
        slots=(),
        rank_plan_digests=rank_plan_digests,
    )
    return prove_manifest_consensus(
        manifest,
        rank=rank,
        transport=_AgreeingTransport(group_size),
    )
