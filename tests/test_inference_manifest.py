from __future__ import annotations

import hashlib

import pytest
from dinkster_inference import (
    CanonicalManifest,
    CompiledPlanSlot,
    ManifestConsensusToken,
    ManifestRefusal,
    ManifestRefusalCode,
    build_canonical_manifest,
    prove_manifest_consensus,
)

RUNTIME_IDENTITY = f"native:dinkster.test:{hashlib.sha256(b'runtime').hexdigest()}"
PLAN_DIGEST = hashlib.sha256(b"rank plan").hexdigest()


class InMemoryConsensusGroup:
    """Digest-only all-gather over shared memory, one group per test."""

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.submitted: dict[int, str] = {}
        self.exchanges = 0

    def submit(self, rank: int, digest: str) -> None:
        self.submitted[rank] = digest

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        self.submitted[rank] = digest
        assert len(self.submitted) == self.group_size, "gather ran before the group submitted"
        self.exchanges += 1
        return tuple(self.submitted[peer] for peer in range(self.group_size))


def example_usp_slot(*, chunk: int = 128) -> CompiledPlanSlot:
    return CompiledPlanSlot(
        slot="usp",
        facts=(
            "process_mesh_axes=cfg1xtp1xu2xr2",
            f"partition_chunk={chunk}",
        ),
    )


def example_window_slot() -> CompiledPlanSlot:
    return CompiledPlanSlot(
        slot="windowed-evaluation",
        facts=(f"composite_plan={hashlib.sha256(b'composite').hexdigest()}",),
    )


def build_manifest(
    *,
    slots: tuple[CompiledPlanSlot, ...] | None = None,
    invocation_facts: tuple[str, ...] = (
        "sigma_table=1.0,0.5,0.0",
        "seed_stream=dinkster.rng.v1:7",
    ),
    rank_plan_digests: tuple[tuple[str, ...], ...] = ((PLAN_DIGEST,), (PLAN_DIGEST,)),
) -> CanonicalManifest:
    return build_canonical_manifest(
        runtime_identity=RUNTIME_IDENTITY,
        invocation_facts=invocation_facts,
        slots=(example_window_slot(), example_usp_slot()) if slots is None else slots,
        rank_plan_digests=rank_plan_digests,
    )


def test_slot_digest_is_deterministic_and_fact_sensitive() -> None:
    assert example_usp_slot().digest == example_usp_slot().digest
    assert example_usp_slot().digest != example_usp_slot(chunk=64).digest
    assert example_usp_slot().digest != example_window_slot().digest


def test_slot_refuses_invalid_names_and_facts() -> None:
    for slot, facts in (
        ("", ("fact=1",)),
        ("usp\nextra", ("fact=1",)),
        ("usp", ()),
        ("usp", ("",)),
        ("usp", ("fact=1\nfact=2",)),
    ):
        with pytest.raises(ManifestRefusal) as error:
            CompiledPlanSlot(slot=slot, facts=facts)
        assert error.value.code is ManifestRefusalCode.INVALID_SLOT


def test_builder_orders_slots_canonically_regardless_of_arrival() -> None:
    forward = build_manifest(slots=(example_usp_slot(), example_window_slot()))
    reversed_arrival = build_manifest(slots=(example_window_slot(), example_usp_slot()))

    assert [slot.slot for slot in forward.slots] == ["usp", "windowed-evaluation"]
    assert forward.manifest_lines() == reversed_arrival.manifest_lines()
    assert forward.digest == reversed_arrival.digest


def test_duplicate_slot_contribution_refuses() -> None:
    with pytest.raises(ManifestRefusal) as error:
        build_manifest(slots=(example_usp_slot(), example_usp_slot(chunk=64)))
    assert error.value.code is ManifestRefusalCode.DUPLICATE_SLOT


def test_manifest_requires_canonical_construction_facts() -> None:
    cases: list[dict[str, object]] = [
        {"invocation_facts": ()},
        {"invocation_facts": ("",)},
        {"invocation_facts": ("fact\nfact",)},
        {"rank_plan_digests": ()},
        {"rank_plan_digests": ((),)},
        {"rank_plan_digests": (("not-a-digest",),)},
    ]
    for overrides in cases:
        with pytest.raises(ManifestRefusal) as error:
            build_manifest(**overrides)  # type: ignore[arg-type]
        assert error.value.code is ManifestRefusalCode.INVALID_MANIFEST


def test_manifest_refuses_malformed_runtime_identity() -> None:
    digest = hashlib.sha256(b"identity").hexdigest()
    for identity in (
        "",
        "native:dinkster.test",
        f"other:x:{digest}",
        "native::abc",
        f"native:x:extra:{digest}",
        f"native:x\ny:{digest}",
        f"native:x:{digest}\n",
    ):
        with pytest.raises(ManifestRefusal) as error:
            build_canonical_manifest(
                runtime_identity=identity,
                invocation_facts=("fact=1",),
                slots=(example_usp_slot(),),
                rank_plan_digests=((PLAN_DIGEST,),),
            )
        assert error.value.code is ManifestRefusalCode.INVALID_MANIFEST


def test_manifest_refuses_non_canonical_slot_order() -> None:
    with pytest.raises(ManifestRefusal) as error:
        CanonicalManifest(
            runtime_identity=RUNTIME_IDENTITY,
            invocation_facts=("fact=1",),
            slots=(example_window_slot(), example_usp_slot()),
            rank_plan_digests=((PLAN_DIGEST,),),
        )
    assert error.value.code is ManifestRefusalCode.INVALID_MANIFEST


def test_manifest_refuses_non_slot_values() -> None:
    class DuckSlot:
        def __init__(self) -> None:
            self.slot = "usp"
            self.facts = ("fact=1",)
            self.digest = hashlib.sha256(b"duck").hexdigest()

    for build in (
        lambda: build_manifest(slots=(DuckSlot(),)),  # type: ignore[arg-type]
        lambda: CanonicalManifest(
            runtime_identity=RUNTIME_IDENTITY,
            invocation_facts=("fact=1",),
            slots=(DuckSlot(),),  # type: ignore[arg-type]
            rank_plan_digests=((PLAN_DIGEST,),),
        ),
    ):
        with pytest.raises(ManifestRefusal) as error:
            build()
        assert error.value.code is ManifestRefusalCode.INVALID_MANIFEST


def test_manifest_lines_are_the_exact_digest_preimage() -> None:
    manifest = build_manifest()
    lines = manifest.manifest_lines()
    assert lines[0] == "dinkster.manifest.v1"
    preimage = b"".join(line.encode() + b"\n" for line in lines)
    assert hashlib.sha256(preimage).hexdigest() == manifest.digest


def test_manifest_digest_binds_every_contribution() -> None:
    base = build_manifest()
    variants = (
        build_manifest(slots=(example_usp_slot(chunk=64), example_window_slot())),
        build_manifest(slots=(example_usp_slot(),)),
        build_manifest(
            invocation_facts=("sigma_table=1.0,0.5,0.0", "seed_stream=dinkster.rng.v1:8")
        ),
        build_manifest(
            rank_plan_digests=((PLAN_DIGEST,), (hashlib.sha256(b"other plan").hexdigest(),))
        ),
    )
    digests = {base.digest, *(variant.digest for variant in variants)}
    assert len(digests) == len(variants) + 1


def test_consensus_agreement_mints_token_for_every_rank() -> None:
    group = InMemoryConsensusGroup(2)
    manifests = [build_manifest() for _ in range(2)]
    group.submit(0, manifests[0].digest)
    group.submit(1, manifests[1].digest)

    tokens = [
        prove_manifest_consensus(manifest, rank=rank, transport=group)
        for rank, manifest in enumerate(manifests)
    ]

    for rank, token in enumerate(tokens):
        assert token.manifest_digest == manifests[0].digest
        assert token.group_size == 2
        assert token.rank == rank


def test_consensus_refuses_non_manifest_values() -> None:
    class EchoTransport:
        def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
            return (digest,)

    class FakeManifest:
        group_size = 1
        digest = "0" * 64

    class SubclassManifest(CanonicalManifest):
        pass

    subclassed = SubclassManifest(
        runtime_identity=RUNTIME_IDENTITY,
        invocation_facts=("fact=1",),
        slots=(example_usp_slot(),),
        rank_plan_digests=((PLAN_DIGEST,),),
    )

    for manifest in (FakeManifest(), subclassed):
        with pytest.raises(ManifestRefusal) as error:
            prove_manifest_consensus(
                manifest,  # type: ignore[arg-type]
                rank=0,
                transport=EchoTransport(),
            )
        assert error.value.code is ManifestRefusalCode.INVALID_MANIFEST


def test_consensus_token_cannot_be_forged() -> None:
    with pytest.raises(TypeError):
        ManifestConsensusToken("0" * 64, 999, -123)  # type: ignore[call-arg]

    with pytest.raises(ManifestRefusal) as error:
        ManifestConsensusToken(
            manifest_digest="0" * 64,
            group_size=999,
            rank=-123,
            mint_authority=object(),
        )
    assert error.value.code is ManifestRefusalCode.FORGED_CONSENSUS_TOKEN


def test_rank_with_differing_fact_refuses_before_any_collective() -> None:
    agreeing = build_manifest()
    divergent = build_manifest(
        invocation_facts=("sigma_table=1.0,0.5,0.0", "seed_stream=dinkster.rng.v1:8")
    )
    assert agreeing.digest != divergent.digest

    group = InMemoryConsensusGroup(2)
    group.submit(0, agreeing.digest)
    group.submit(1, divergent.digest)
    collectives_issued: list[ManifestConsensusToken] = []

    for rank, manifest in ((0, agreeing), (1, divergent)):
        with pytest.raises(ManifestRefusal) as error:
            token = prove_manifest_consensus(manifest, rank=rank, transport=group)
            collectives_issued.append(token)
        assert error.value.code is ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH

    assert collectives_issued == []


def test_no_majority_override() -> None:
    agreeing = build_manifest(rank_plan_digests=((PLAN_DIGEST,),) * 3)
    divergent = build_manifest(
        invocation_facts=("sigma_table=1.0,0.5,0.0", "seed_stream=dinkster.rng.v1:8"),
        rank_plan_digests=((PLAN_DIGEST,),) * 3,
    )

    group = InMemoryConsensusGroup(3)
    group.submit(0, agreeing.digest)
    group.submit(1, agreeing.digest)
    group.submit(2, divergent.digest)

    for rank, manifest in ((0, agreeing), (1, agreeing), (2, divergent)):
        with pytest.raises(ManifestRefusal) as error:
            prove_manifest_consensus(manifest, rank=rank, transport=group)
        assert error.value.code is ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH


def test_consensus_refuses_invalid_groups_and_transports() -> None:
    manifest = build_manifest()

    class ShortTransport:
        def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
            return (digest,)

    class ForeignTransport:
        def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
            other = hashlib.sha256(b"foreign").hexdigest()
            return (other, other)

    class MalformedTransport:
        def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
            return (digest, "not-a-digest")

    for rank in (-1, 2, "0"):
        with pytest.raises(ManifestRefusal) as error:
            prove_manifest_consensus(
                manifest,
                rank=rank,  # type: ignore[arg-type]
                transport=InMemoryConsensusGroup(2),
            )
        assert error.value.code is ManifestRefusalCode.CONSENSUS_GROUP_INVALID

    for transport in (ShortTransport(), ForeignTransport(), MalformedTransport()):
        with pytest.raises(ManifestRefusal) as error:
            prove_manifest_consensus(manifest, rank=0, transport=transport)
        assert error.value.code is ManifestRefusalCode.CONSENSUS_GROUP_INVALID
