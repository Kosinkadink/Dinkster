from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

import pytest
from dinkster_assets import P2PGrantReconciler, TransportCandidate, rank_transport_candidates
from dinkster_assets.p2p_descriptor import (
    P2PDescriptorV1,
    canonical_p2p_info,
    derive_p2p_descriptor,
)
from dinkster_assets.p2p_global import (
    GlobalP2PCounterStore,
    GlobalP2PPolicyError,
    NatObservation,
    ProviderArtifactP2PV1,
    ProviderLocationV1,
    ProviderP2PEnumerationV1,
    ProviderP2PSnapshotV1,
    ProviderP2PTombstoneV1,
    classify_nat_outcome,
    provider_declarations,
    provider_transport_candidates,
    provider_transport_candidates_for_snapshots,
)
from dinkster_assets.p2p_grants import (
    P2P_REMOTE_GRANT_MAX_SECONDS,
    P2PGrantSnapshot,
    PublicSwarmGrantV1,
)

NOW = 1_800_000_000.0
TRUSTED = frozenset({"official.fixture"})
DescriptorFixture: TypeAlias = tuple[Path, str, int, P2PDescriptorV1]


class _CountingGrants(tuple[PublicSwarmGrantV1, ...]):
    iterations = 0

    def __iter__(self) -> Iterator[PublicSwarmGrantV1]:
        self.iterations += 1
        return super().__iter__()


@pytest.fixture
def descriptor(tmp_path: Path) -> DescriptorFixture:
    path = tmp_path / "fixture.safetensors"
    header = json.dumps(
        {"weight": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    result = derive_p2p_descriptor(path)
    return path, result.asset_digest, result.size, result.descriptor


def location(*, eligible: bool = True, credential_free: bool = True) -> ProviderLocationV1:
    return ProviderLocationV1(
        "https://models.example/fixture.safetensors",
        eligible,
        credential_free,
    )


def snapshot(
    descriptor: DescriptorFixture,
    *,
    provider_id: str = "official.fixture",
    license: str = "Apache-2.0",
    format_safe: bool = True,
    locations: tuple[ProviderLocationV1, ...] | None = None,
    expires_at: float = NOW + 24 * 60 * 60,
    refreshed_at: float = NOW,
    trackers: tuple[str, ...] = (),
    tombstoned: bool = False,
    enumerated: bool = True,
    enumeration_descriptor: P2PDescriptorV1 | None = None,
) -> ProviderP2PSnapshotV1:
    _, digest, size, p2p = descriptor
    return ProviderP2PSnapshotV1(
        provider_id=provider_id,
        source_revision="fixture-r1",
        refreshed_at=refreshed_at,
        artifacts=(
            ProviderArtifactP2PV1(
                source_id="artifact-1",
                digest=digest,
                size_bytes=size,
                descriptor=p2p,
                license=license,
                format_safe=format_safe,
                locations=(location(),) if locations is None else locations,
            ),
        ),
        p2p_artifacts=(
            ProviderP2PEnumerationV1(
                "fixture-enumeration",
                digest,
                size,
                enumeration_descriptor or p2p,
                expires_at,
            ),
        )
        if enumerated
        else (),
        p2p_trackers=trackers,
        tombstones=(ProviderP2PTombstoneV1(digest, NOW + 1),) if tombstoned else (),
    )


def grant_snapshot(
    provider: ProviderP2PSnapshotV1, descriptor: DescriptorFixture
) -> P2PGrantSnapshot:
    path, digest, _, _ = descriptor
    declarations = tuple(
        decision.declaration
        for decision in provider_declarations(provider, trusted_provider_ids=TRUSTED, now=NOW)
        if decision.declaration is not None
    )
    return (
        P2PGrantReconciler(clock=lambda: NOW)
        .reconcile(
            declarations,
            (),
            lambda requested: path if requested == digest else None,
            enabled=True,
        )
        .snapshot
    )


def eligible_grants(descriptor: DescriptorFixture) -> P2PGrantSnapshot:
    grants = grant_snapshot(snapshot(descriptor), descriptor)
    assert len(grants.public_grants) == len(grants.seed_grants) == 1
    return grants


def test_provider_facts_adapt_to_shared_six_hour_grants(descriptor: DescriptorFixture) -> None:
    provider = snapshot(descriptor)
    decision = provider_declarations(provider, trusted_provider_ids=TRUSTED, now=NOW)[0]
    assert decision.eligible
    assert decision.declaration is not None
    assert decision.declaration.source_type == "official-provider"
    assert decision.declaration.evidence_id == "fixture-enumeration"
    assert decision.declaration.expires_at == NOW + P2P_REMOTE_GRANT_MAX_SECONDS

    grants = grant_snapshot(provider, descriptor)
    assert grants.public_grants[0].expires_at == NOW + P2P_REMOTE_GRANT_MAX_SECONDS
    assert grants.seed_grants[0].expires_at == NOW + P2P_REMOTE_GRANT_MAX_SECONDS

    future = provider_declarations(
        snapshot(descriptor, refreshed_at=NOW + 1),
        trusted_provider_ids=TRUSTED,
        now=NOW,
    )[0]
    assert not future.eligible
    assert future.reason == "future-refresh"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"format_safe": False}, "unsafe-format"),
        ({"expires_at": NOW}, "expired"),
        ({"enumerated": False}, "not-enumerated"),
        ({"provider_id": "untrusted.fixture"}, "untrusted-provider"),
    ],
)
def test_ineligible_provider_artifacts_never_produce_a_grant(
    descriptor: DescriptorFixture,
    changes: dict[str, object],
    reason: str,
) -> None:
    provider = snapshot(descriptor, **changes)  # type: ignore[arg-type]
    decision = provider_declarations(provider, trusted_provider_ids=TRUSTED, now=NOW)[0]
    assert not decision.eligible
    assert decision.declaration is None
    assert decision.reason == reason
    assert grant_snapshot(provider, descriptor).public_grants == ()


@pytest.mark.parametrize(
    "license_name", ["", "unknown", "All rights reserved", "Apache-2.0 AND MIT"]
)
@pytest.mark.parametrize(
    "locations",
    [(), (location(eligible=False, credential_free=False),), (location(eligible=False),)],
)
def test_provider_metadata_does_not_authorize_or_prevent_verified_seeding(
    descriptor: DescriptorFixture,
    license_name: str,
    locations: tuple[ProviderLocationV1, ...],
) -> None:
    provider = snapshot(descriptor, license=license_name, locations=locations)
    grants = grant_snapshot(provider, descriptor)
    assert len(grants.public_grants) == len(grants.seed_grants) == 1
    assert grants.public_grants[0].license == license_name
    assert grants.seed_grants[0].grant_id == grants.public_grants[0].grant_id
    candidates = provider_transport_candidates(
        provider, grants, trusted_provider_ids=TRUSTED, now=NOW, global_downloads_allowed=True
    )
    assert len(candidates) == 1
    assert candidates[0].source == grants.public_grants[0].grant_id

    path, _, _, _ = descriptor
    original = path.read_bytes()
    path.write_bytes(bytes(len(original)))
    mutated = grant_snapshot(provider, descriptor)
    assert mutated.public_grants == grants.public_grants
    assert mutated.seed_grants == ()
    path.unlink()
    missing = grant_snapshot(provider, descriptor)
    assert missing.public_grants == grants.public_grants
    assert missing.seed_grants == ()


def test_mismatched_enumeration_and_malformed_wire_fail_closed(
    descriptor: DescriptorFixture,
) -> None:
    _, digest, size, _ = descriptor
    alternate_root = "c" * 64
    alternate = P2PDescriptorV1(
        "bittorrent-v2",
        hashlib.sha256(
            canonical_p2p_info(asset_digest=digest, size=size, file_root=alternate_root)
        ).hexdigest(),
        alternate_root,
        8 * 1024 * 1024,
    )
    decision = provider_declarations(
        snapshot(descriptor, enumeration_descriptor=alternate),
        trusted_provider_ids=TRUSTED,
        now=NOW,
    )[0]
    assert not decision.eligible
    assert decision.reason == "malformed-enumeration"

    valid = snapshot(descriptor)
    malformed = {
        "version": 1,
        "providerId": valid.provider_id,
        "sourceRevision": valid.source_revision,
        "refreshedAt": valid.refreshed_at,
        "artifacts": [
            {
                "sourceId": valid.artifacts[0].source_id,
                "digest": digest,
                "sizeBytes": size,
                "p2p": {**valid.artifacts[0].descriptor.to_wire(), "unexpected": True},
                "license": "Apache-2.0",
                "formatSafe": True,
                "locations": [location().to_wire()],
            }
        ],
        "p2pArtifacts": [],
        "p2pTrackers": [],
        "tombstones": [],
    }
    with pytest.raises(GlobalP2PPolicyError, match="unknown fields"):
        ProviderP2PSnapshotV1.from_wire(malformed)
    with pytest.raises(GlobalP2PPolicyError, match="whitespace"):
        ProviderLocationV1.from_wire(
            {
                "url": "https://models.example/bad path",
                "eligible": True,
                "credentialFree": True,
            }
        )


def test_observed_tombstone_immediately_revokes_provider_declaration(
    descriptor: DescriptorFixture,
) -> None:
    path, digest, _, _ = descriptor
    reconciler = P2PGrantReconciler(clock=lambda: NOW)
    initial_declaration = provider_declarations(
        snapshot(descriptor), trusted_provider_ids=TRUSTED, now=NOW
    )[0].declaration
    assert initial_declaration is not None
    initial = reconciler.reconcile(
        (initial_declaration,), (), lambda _: path, enabled=True
    ).snapshot
    assert initial.seed_grants

    tombstoned = provider_declarations(
        snapshot(descriptor, tombstoned=True),
        trusted_provider_ids=TRUSTED,
        now=NOW,
    )[0]
    assert tombstoned.reason == "tombstoned"
    revoked = reconciler.reconcile((), (), lambda _: path, enabled=True)
    assert revoked.snapshot.public_grants == revoked.snapshot.seed_grants == ()
    assert revoked.revoked_download_digests == frozenset({digest})
    assert revoked.revoked_seed_digests == frozenset({digest})


def test_provider_grants_map_to_global_transport_only_while_eligible(
    descriptor: DescriptorFixture,
) -> None:
    provider = snapshot(descriptor)
    grants = grant_snapshot(provider, descriptor)
    candidates = provider_transport_candidates(
        provider,
        grants,
        trusted_provider_ids=TRUSTED,
        now=NOW,
        global_downloads_allowed=True,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    grant = grants.public_grants[0]
    assert candidate == TransportCandidate(
        kind="global-p2p",
        source=grant.grant_id,
        size_bytes=descriptor[2],
        descriptor=descriptor[3],
        expires_at=grant.expires_at,
    )
    http = TransportCandidate("http", "https://models.example/fixture.safetensors")
    assert rank_transport_candidates((http, candidate)) == (candidate, http)

    unavailable_cases = (
        (snapshot(descriptor, tombstoned=True), True),
        (snapshot(descriptor, license="unknown"), True),
        (provider, False),
    )
    for unavailable, allowed in unavailable_cases:
        assert (
            provider_transport_candidates(
                unavailable,
                grants,
                trusted_provider_ids=TRUSTED,
                now=NOW,
                global_downloads_allowed=allowed,
            )
            == ()
        )


def test_provider_transport_rejects_noncanonical_grant_id(
    descriptor: DescriptorFixture,
) -> None:
    provider = snapshot(descriptor)
    grants = grant_snapshot(provider, descriptor)
    grant = grants.public_grants[0]
    noncanonical_id = "f" * 64
    assert grant.grant_id != noncanonical_id
    forged = replace(grants, public_grants=(replace(grant, grant_id=noncanonical_id),))

    assert (
        provider_transport_candidates(
            provider,
            forged,
            trusted_provider_ids=TRUSTED,
            now=NOW,
            global_downloads_allowed=True,
        )
        == ()
    )
    duplicated = replace(grants, public_grants=(grant, grant))
    assert (
        provider_transport_candidates(
            provider,
            duplicated,
            trusted_provider_ids=TRUSTED,
            now=NOW,
            global_downloads_allowed=True,
        )
        == ()
    )


def test_provider_transport_indexes_multiple_catalogs_grants_once() -> None:
    count = 64
    artifacts: list[ProviderArtifactP2PV1] = []
    enumeration: list[ProviderP2PEnumerationV1] = []
    for position in range(count):
        digest = "blake3:" + hashlib.sha256(f"digest-{position}".encode()).hexdigest()
        file_root = hashlib.sha256(f"root-{position}".encode()).hexdigest()
        size = position + 1
        p2p = P2PDescriptorV1(
            "bittorrent-v2",
            hashlib.sha256(
                canonical_p2p_info(asset_digest=digest, size=size, file_root=file_root)
            ).hexdigest(),
            file_root,
            8 * 1024 * 1024,
        )
        artifacts.append(
            ProviderArtifactP2PV1(
                source_id=f"artifact-{position}",
                digest=digest,
                size_bytes=size,
                descriptor=p2p,
                license="Apache-2.0",
                format_safe=True,
                locations=(location(),),
            )
        )
        enumeration.append(
            ProviderP2PEnumerationV1(
                hashlib.sha256(f"grant-{position}".encode()).hexdigest(),
                digest,
                size,
                p2p,
                NOW + P2P_REMOTE_GRANT_MAX_SECONDS,
            )
        )
    provider = ProviderP2PSnapshotV1(
        provider_id="official.fixture",
        source_revision="large-r1",
        refreshed_at=NOW,
        artifacts=tuple(artifacts),
        p2p_artifacts=tuple(enumeration),
    )
    second = replace(
        provider,
        provider_id="official.second",
        source_revision="large-r2",
        artifacts=tuple(
            replace(artifact, source_id=f"second-{position}")
            for position, artifact in enumerate(provider.artifacts)
        ),
    )
    providers = (provider, second)
    trusted = TRUSTED | {second.provider_id}
    declarations = tuple(
        decision.declaration
        for snapshot_row in providers
        for decision in provider_declarations(
            snapshot_row,
            trusted_provider_ids=trusted,
            now=NOW,
        )
        if decision.declaration is not None
    )
    grants = (
        P2PGrantReconciler(clock=lambda: NOW)
        .reconcile(
            declarations,
            (),
            lambda _digest: None,
            enabled=True,
        )
        .snapshot
    )

    counted = _CountingGrants(grants.public_grants)
    candidates = provider_transport_candidates_for_snapshots(
        providers,
        replace(grants, public_grants=counted),
        trusted_provider_ids=trusted,
        now=NOW,
        global_downloads_allowed=True,
    )

    assert len(candidates) == count * len(providers)
    assert counted.iterations == 1


def test_durable_counters_survive_restart(
    tmp_path: Path,
    descriptor: DescriptorFixture,
) -> None:
    grant = eligible_grants(descriptor).seed_grants[0]
    store_path = tmp_path / "p2p-counters.sqlite"
    with GlobalP2PCounterStore(store_path) as store:
        credited = store.credit_ratio_equivalent("receipt-fixture", grant.digest, 100)
        assert store.credit_ratio_equivalent("receipt-fixture", grant.digest, 100) == credited
        with pytest.raises(GlobalP2PPolicyError, match="different ratio credit"):
            store.credit_ratio_equivalent("receipt-fixture", grant.digest, 101)
        store.record_transfer(
            grant.digest,
            downloaded_bytes=40,
            uploaded_bytes=100,
            active_seed_seconds=10,
        )

    with GlobalP2PCounterStore(store_path) as reopened:
        persisted = reopened.get(grant.digest)
        assert persisted.downloaded_bytes == 40
        assert persisted.uploaded_bytes == 100
        assert persisted.ratio_equivalent_bytes == 100
        assert persisted.active_seed_seconds == 10


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (NatObservation(True, True, True, True), "reachable"),
        (NatObservation(True, True, True, False), "outbound-only"),
        (NatObservation(True, True, False, False), "unreachable"),
        (NatObservation(False, False, True, False), "outbound-only"),
        (NatObservation(False, False, False, False), "unreachable"),
    ],
)
def test_nat_outcome_matrix_reports_only_observed_reachability(
    observation: NatObservation,
    expected: str,
) -> None:
    assert classify_nat_outcome(observation) == expected
