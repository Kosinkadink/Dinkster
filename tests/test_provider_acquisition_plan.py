from __future__ import annotations

import builtins
import itertools
import socket
from dataclasses import FrozenInstanceError

import pytest
from dinkster_assets.acquisition_plan import (
    AcquisitionPlan,
    AcquisitionRequest,
    AcquisitionSource,
    DestinationConsideration,
    DestinationIdentity,
    ManagedDestination,
    SourceConsideration,
    plan_acquisition,
)
from dinkster_assets.identity import AssetError
from dinkster_assets.resolution import ProviderMirror, SourceIdentity

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
KIND = "model/diffusion"


def source(
    provider: str = "provider-a",
    source_id: str = "source-a",
    *,
    digest: str = DIGEST_A,
    kind: str = KIND,
    priority: int = 0,
    state: str = "available",
    reason: str = "",
    credential: str | None = None,
    license: str | None = None,
    cost: str | None = None,
    policy: str | None = None,
) -> AcquisitionSource:
    return AcquisitionSource(
        SourceIdentity(provider, source_id),
        digest,
        kind,
        state,  # type: ignore[arg-type]
        f"https://{provider}.example/{source_id}",
        100,
        "application/octet-stream",
        priority,
        reason,
        credential,
        license,
        cost,
        policy,
    )


def mirror(
    provider: str = "provider-a",
    source_id: str = "source-a",
    *,
    digest: str = DIGEST_A,
    state: str = "available",
    metadata: dict[str, object] | None = None,
) -> ProviderMirror:
    return ProviderMirror(
        SourceIdentity(provider, source_id),
        digest,
        state,  # type: ignore[arg-type]
        "offline" if state == "unavailable" else "",
        metadata or {},
        1,
        2,
    )


def destination(
    mount: str = "models-a",
    path: str = "models/model.safetensors",
    *,
    priority: int = 0,
    kind: str = KIND,
    ready: bool = True,
    readwrite: bool = True,
    compatible: bool = True,
) -> ManagedDestination:
    return ManagedDestination(
        DestinationIdentity(mount, path),
        priority,
        kind,
        ready,
        readwrite,
        compatible,
        "" if ready else "mount-not-ready",
        "" if readwrite else "mount-read-only",
        "" if compatible else "storage-policy-incompatible",
    )


def request(**changes: object) -> AcquisitionRequest:
    values: dict[str, object] = {
        "digest": DIGEST_A,
        "asset_kind": KIND,
        "authorized_providers": frozenset(("provider-a", "provider-b")),
    }
    values.update(changes)
    return AcquisitionRequest(**values)  # type: ignore[arg-type]


def test_same_digest_mirrors_deduplicate_and_order_by_trusted_priority_identity() -> None:
    sources = (
        source("provider-b", "z", priority=0),
        source("provider-a", "z", priority=1),
        source("provider-a", "a", priority=1),
        source("provider-a", "a", priority=1),
    )
    mirrors = (
        mirror("provider-a", "a"),
        mirror("provider-a", "a"),
        mirror("provider-a", "z"),
        mirror("provider-b", "z"),
    )
    plan = plan_acquisition(request(), sources, mirrors, (destination(),))
    assert plan.status == "ready"
    assert plan.selected_source == sources[0]
    assert [row.candidate.source for row in plan.sources] == [
        SourceIdentity("provider-b", "z"),
        SourceIdentity("provider-a", "a"),
        SourceIdentity("provider-a", "z"),
    ]


def test_same_source_different_digest_is_ambiguous_and_cannot_bypass() -> None:
    facts = (source(), source(digest=DIGEST_B))
    plan = plan_acquisition(request(), facts, (mirror(), mirror(digest=DIGEST_B)), (destination(),))
    assert plan.status == "ambiguous"
    assert plan.reason == "conflicting-source-facts"
    assert {row.reason for row in plan.sources} == {
        "conflicting-source-facts",
        "digest-mismatch",
    }


def test_irrelevant_source_conflicts_do_not_override_closed_refusals() -> None:
    unauthorized = (
        source("provider-c", "same"),
        source("provider-c", "same", priority=1),
    )
    unauthorized_plan = plan_acquisition(
        request(
            authorized_providers=frozenset(("provider-a",)),
            selected_source=SourceIdentity("provider-c", "same"),
        ),
        unauthorized,
        (mirror("provider-c", "same"),),
        (destination(),),
    )
    assert unauthorized_plan.status == "unauthorized"

    unavailable = (
        source("provider-a", "same", state="unavailable", reason="offline"),
        source(
            "provider-a",
            "same",
            priority=1,
            state="unavailable",
            reason="maintenance",
        ),
    )
    unavailable_plan = plan_acquisition(
        request(),
        unavailable,
        (mirror("provider-a", "same", state="unavailable"),),
        (destination(),),
    )
    assert unavailable_plan.status == "source-unavailable"
    assert {row.reason for row in unavailable_plan.sources} == {"source-unavailable"}


def test_mirror_is_only_identity_digest_availability_corroboration() -> None:
    trusted = source("provider-b", "source-b", priority=10)
    advisory = mirror(
        "provider-a",
        "source-a",
        metadata={
            "priority": -1000,
            "authorizedLocator": "https://attacker.invalid/winner",
            "credentialGrant": "forged",
            "displayName": "preferred sibling",
        },
    )
    plan = plan_acquisition(
        request(),
        (source(), trusted),
        (advisory, mirror("provider-b", "source-b", metadata={"priority": -9999})),
        (destination(),),
    )
    assert plan.status == "ready"
    assert plan.selected_source == source()
    assert plan.selected_source is not None
    assert plan.selected_source.authorized_locator == "https://provider-a.example/source-a"

    wrong_digest = plan_acquisition(
        request(), (source(),), (mirror(digest=DIGEST_B),), (destination(),)
    )
    assert wrong_digest.status == "source-unavailable"
    assert wrong_digest.sources[0].reason == "mirror-digest-mismatch"
    unavailable = plan_acquisition(
        request(), (source(),), (mirror(state="unavailable"),), (destination(),)
    )
    assert unavailable.sources[0].reason == "mirror-unavailable"


def test_authority_digest_kind_and_explicit_source_fail_closed() -> None:
    rows = (
        source("provider-a", "good"),
        source("provider-c", "unauthorized"),
        source("provider-a", "wrong-digest", digest=DIGEST_B),
        source("provider-a", "wrong-kind", kind="model/lora"),
        source("provider-a", "offline", state="unavailable", reason="maintenance"),
    )
    mirrors = tuple(
        mirror(row.source.provider_id, row.source.source_id, digest=row.digest) for row in rows
    )
    explicit = plan_acquisition(
        request(selected_source=SourceIdentity("provider-a", "wrong-digest")),
        rows,
        mirrors,
        (destination(),),
    )
    assert explicit.status == "source-unavailable"
    assert explicit.selected_source is None
    assert {row.candidate.source.source_id: row.reason for row in explicit.sources} == {
        "good": "not-explicit-source",
        "offline": "source-unavailable",
        "wrong-digest": "digest-mismatch",
        "wrong-kind": "kind-mismatch",
        "unauthorized": "unauthorized-provider",
    }
    unauthorized = plan_acquisition(
        request(
            selected_source=SourceIdentity("provider-c", "unauthorized"),
            authorized_providers=frozenset(("provider-a",)),
        ),
        rows,
        mirrors,
        (destination(),),
    )
    assert unauthorized.status == "unauthorized"
    assert unauthorized.reason == "explicit-provider-unauthorized"


@pytest.mark.parametrize(
    ("grants", "expected_status", "expected_requirement"),
    [
        ({}, "credential-required", "credential-X"),
        ({"credential_grants": frozenset(("credential-X",))}, "license-required", "license-X"),
        (
            {
                "credential_grants": frozenset(("credential-X",)),
                "license_grants": frozenset(("license-X",)),
            },
            "cost-required",
            "cost-X",
        ),
        (
            {
                "credential_grants": frozenset(("credential-X",)),
                "license_grants": frozenset(("license-X",)),
                "cost_grants": frozenset(("cost-X",)),
            },
            "policy-override-required",
            "policy-X",
        ),
    ],
)
def test_exceptional_grants_are_exact_and_have_stable_precedence(
    grants: dict[str, object], expected_status: str, expected_requirement: str
) -> None:
    gated = source(credential="credential-X", license="license-X", cost="cost-X", policy="policy-X")
    plan = plan_acquisition(request(**grants), (gated,), (mirror(),), (destination(),))
    assert plan.status == expected_status
    assert plan.required_grant_id == expected_requirement
    assert plan.selected_source == gated

    ready = plan_acquisition(
        request(
            credential_grants=frozenset(("credential-X",)),
            license_grants=frozenset(("license-X",)),
            cost_grants=frozenset(("cost-X",)),
            policy_override_grants=frozenset(("policy-X",)),
        ),
        (gated,),
        (mirror(),),
        (destination(),),
    )
    assert ready.status == "ready"


def test_destinations_filter_order_and_explicit_selection_fail_closed() -> None:
    rows = (
        destination("models-z", priority=1),
        destination("models-b", priority=0),
        destination("models-a", priority=0),
        destination("not-ready", ready=False, priority=-10),
        destination("readonly", readwrite=False, priority=-10),
        destination("incompatible", compatible=False, priority=-10),
        destination("wrong-kind", kind="model/lora", priority=-10),
    )
    plan = plan_acquisition(request(), (source(),), (mirror(),), rows)
    assert plan.status == "ready"
    assert plan.selected_destination == destination("models-a", priority=0)
    assert {row.candidate.identity.mount_id: row.reason for row in plan.destinations} == {
        "not-ready": "not-ready",
        "readonly": "read-only",
        "incompatible": "incompatible",
        "wrong-kind": "kind-mismatch",
        "models-a": "eligible",
        "models-b": "eligible",
        "models-z": "eligible",
    }

    explicit = plan_acquisition(
        request(selected_destination=DestinationIdentity("readonly", "models/model.safetensors")),
        (source(),),
        (mirror(),),
        rows,
    )
    assert explicit.status == "no-compatible-destination"
    assert explicit.selected_destination is None
    assert any(row.reason == "not-explicit-destination" for row in explicit.destinations)


def test_irrelevant_destination_conflicts_do_not_override_closed_refusal() -> None:
    selected = destination("selected", ready=False)
    unrelated_conflict = (
        destination("other"),
        destination("other", priority=1),
    )
    plan = plan_acquisition(
        request(selected_destination=selected.identity),
        (source(),),
        (mirror(),),
        (selected, *unrelated_conflict),
    )
    assert plan.status == "no-compatible-destination"
    assert {row.candidate.identity.mount_id: row.reason for row in plan.destinations} == {
        "selected": "not-ready",
        "other": "not-explicit-destination",
    }

    all_not_ready = (
        destination("same", ready=False),
        destination("same", priority=1, ready=False),
    )
    unavailable = plan_acquisition(request(), (source(),), (mirror(),), all_not_ready)
    assert unavailable.status == "no-compatible-destination"
    assert {row.reason for row in unavailable.destinations} == {"not-ready"}


def test_permutations_foreign_hints_and_sibling_names_cannot_change_plan() -> None:
    sources = (source("provider-a", "fp8"), source("provider-b", "nvfp4"))
    mirrors = (
        mirror("provider-a", "fp8", metadata={"variant": "nvfp4", "filename": "z"}),
        mirror(
            "provider-b",
            "nvfp4",
            metadata={"variant": "fp8", "filename": "aaa", "templateAlias": "winner"},
        ),
    )
    destinations = (
        destination("models-b", "variants/nvfp4/model.safetensors"),
        destination("models-a", "variants/fp8/model.safetensors"),
    )
    signatures = set()
    for source_order, mirror_order, destination_order in itertools.product(
        itertools.permutations(sources),
        itertools.permutations(mirrors),
        itertools.permutations(destinations),
    ):
        plan = plan_acquisition(request(), source_order, mirror_order, destination_order)
        signatures.add(
            (
                plan.status,
                plan.selected_source,
                plan.selected_destination,
                plan.sources,
                plan.destinations,
            )
        )
    assert len(signatures) == 1


def test_records_validate_freeze_copy_inputs_and_store_no_secret_fields() -> None:
    providers = {"provider-a"}
    req = request(authorized_providers=providers)
    providers.clear()
    assert req.authorized_providers == frozenset(("provider-a",))
    with pytest.raises(FrozenInstanceError):
        req.digest = DIGEST_B  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        source().expected_size = 9  # type: ignore[misc]
    with pytest.raises(AssetError, match="userinfo credentials"):
        AcquisitionSource(
            SourceIdentity("provider-a", "source-a"),
            DIGEST_A,
            KIND,
            "available",
            "https://bearer:secret@example.invalid/model",
            1,
            "application/octet-stream",
        )
    with pytest.raises(AssetError, match="expected size"):
        source().__class__(
            SourceIdentity("provider-a", "source-a"),
            DIGEST_A,
            KIND,
            "available",
            "https://example.invalid/model",
            -1,
            "application/octet-stream",
        )
    with pytest.raises(AssetError, match="media type"):
        source().__class__(
            SourceIdentity("provider-a", "source-a"),
            DIGEST_A,
            KIND,
            "available",
            "https://example.invalid/model",
            1,
            42,  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="source must be"):
        source().__class__(
            "provider-a/source-a",  # type: ignore[arg-type]
            DIGEST_A,
            KIND,
            "available",
            "https://example.invalid/model",
            1,
            "application/octet-stream",
        )
    assert not ({"secret", "bearer", "price", "accepted", "permission"} & set(source().__dict__))
    plan = plan_acquisition(req, (source(),), (mirror(),), (destination(),))
    assert hash(req)
    assert hash(source())
    assert hash(destination())
    assert hash(plan.sources[0])
    assert hash(plan.destinations[0])
    assert hash(plan)


@pytest.mark.parametrize(
    "locator",
    (
        "https://user@example.invalid/model",
        "https://user:secret@example.invalid/model",
        "https://example.invalid/model?token=secret",
        "https://example.invalid/model#bearer",
        "https://example.invalid/model\nnext",
        "https://[::1",
        "https://example.invalid:/model",
    ),
)
def test_authorized_locator_excludes_secret_bearing_url_shapes(locator: str) -> None:
    with pytest.raises(AssetError, match="authorized locator"):
        source().__class__(
            SourceIdentity("provider-a", "source-a"),
            DIGEST_A,
            KIND,
            "available",
            locator,
            1,
            "application/octet-stream",
        )
    assert source().authorized_locator == "https://provider-a.example/source-a"


@pytest.mark.parametrize(
    "locator",
    (
        "http://provider.example:8080/models/model.safetensors",
        "https://provider.example",
    ),
)
def test_authorized_locator_accepts_only_documented_credential_free_shape(
    locator: str,
) -> None:
    accepted = source().__class__(
        SourceIdentity("provider-a", "source-a"),
        DIGEST_A,
        KIND,
        "available",
        locator,
        1,
        "application/octet-stream",
    )
    assert accepted.authorized_locator == locator


def test_direct_record_construction_enforces_closed_outcome_invariants() -> None:
    selected_source = source()
    selected_destination = destination()
    eligible_source = SourceConsideration(selected_source, True, "eligible")
    eligible_destination = DestinationConsideration(selected_destination, True, "eligible")

    with pytest.raises(AssetError, match="eligibility and reason disagree"):
        SourceConsideration(selected_source, False, "eligible")
    with pytest.raises(AssetError, match="eligibility and reason disagree"):
        DestinationConsideration(selected_destination, True, "read-only")
    with pytest.raises(AssetError, match="ineligible source.*requires a reason"):
        SourceConsideration(selected_source, False, "")
    with pytest.raises(AssetError, match="ineligible destination.*requires a reason"):
        DestinationConsideration(selected_destination, False, "")
    with pytest.raises(AssetError, match="invalid source consideration"):
        SourceConsideration("bad", False, "bad-candidate")  # type: ignore[arg-type]
    with pytest.raises(AssetError, match="invalid destination consideration"):
        DestinationConsideration("bad", False, "bad-candidate")  # type: ignore[arg-type]
    with pytest.raises(AssetError, match="destination identity must be"):
        ManagedDestination(
            "models-a/path",  # type: ignore[arg-type]
            0,
            KIND,
            True,
            True,
            True,
        )
    with pytest.raises(AssetError, match="virtual path must be a string"):
        DestinationIdentity("models-a", 42)  # type: ignore[arg-type]
    with pytest.raises(AssetError, match="digest must be"):
        source().__class__(
            SourceIdentity("provider-a", "source-a"),
            42,  # type: ignore[arg-type]
            KIND,
            "available",
            "https://example.invalid/model",
            1,
            "application/octet-stream",
        )
    with pytest.raises(AssetError, match="digest must be"):
        AcquisitionRequest(
            42,  # type: ignore[arg-type]
            KIND,
            frozenset(("provider-a",)),
        )
    with pytest.raises(AssetError, match="authorized providers must be a collection"):
        AcquisitionRequest(
            DIGEST_A,
            KIND,
            None,  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="credential_grants must be a collection"):
        AcquisitionRequest(
            DIGEST_A,
            KIND,
            frozenset(("provider-a",)),
            credential_grants=None,  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="selected source must be"):
        AcquisitionRequest(
            DIGEST_A,
            KIND,
            frozenset(("provider-a",)),
            selected_source="provider-a/source-a",  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="selected destination must be"):
        AcquisitionRequest(
            DIGEST_A,
            KIND,
            frozenset(("provider-a",)),
            selected_destination="models-a/path",  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="unknown acquisition plan status"):
        AcquisitionPlan(
            "invented",  # type: ignore[arg-type]
            "invented",
            (eligible_source,),
            (eligible_destination,),
        )
    for invalid_reason in ("", "human reason", "x" * 257):
        with pytest.raises(AssetError, match="plan reason"):
            AcquisitionPlan(
                "source-unavailable",
                invalid_reason,
                (eligible_source,),
                (eligible_destination,),
            )
    with pytest.raises(AssetError, match="plan sources must be a collection"):
        AcquisitionPlan(
            "source-unavailable",
            "source-unavailable",
            None,  # type: ignore[arg-type]
            (eligible_destination,),
        )
    with pytest.raises(AssetError, match="plan destinations must be a collection"):
        AcquisitionPlan(
            "source-unavailable",
            "source-unavailable",
            (eligible_source,),
            "not-rows",  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="plan sources must be a collection"):
        AcquisitionPlan(
            "source-unavailable",
            "source-unavailable",
            b"not-rows",  # type: ignore[arg-type]
            (eligible_destination,),
        )
    with pytest.raises(AssetError, match="ready plan requires"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (eligible_destination,),
            selected_destination=selected_destination,
        )
    with pytest.raises(AssetError, match="ready plan requires"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (eligible_destination,),
            selected_source=selected_source,
        )
    with pytest.raises(AssetError, match="ready plan cannot require a grant"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (eligible_destination,),
            selected_source,
            selected_destination,
            "credential-X",
        )
    with pytest.raises(AssetError, match="grant-required plan requires"):
        AcquisitionPlan(
            "credential-required",
            "credential-required",
            (eligible_source,),
            (eligible_destination,),
            selected_source,
        )
    for status in (
        "credential-required",
        "license-required",
        "cost-required",
        "policy-override-required",
    ):
        with pytest.raises(AssetError, match="grant-required plan requires"):
            AcquisitionPlan(
                status,  # type: ignore[arg-type]
                status,
                (eligible_source,),
                (eligible_destination,),
                selected_source,
            )
        with pytest.raises(AssetError, match="grant-required plan requires"):
            AcquisitionPlan(
                status,  # type: ignore[arg-type]
                status,
                (eligible_source,),
                (eligible_destination,),
                required_grant_id="grant-X",
            )
    with pytest.raises(AssetError, match="only a ready plan"):
        AcquisitionPlan(
            "source-unavailable",
            "source-unavailable",
            (eligible_source,),
            (eligible_destination,),
            selected_destination=selected_destination,
        )
    with pytest.raises(AssetError, match="selected source must be"):
        AcquisitionPlan(
            "source-unavailable",
            "source-unavailable",
            (eligible_source,),
            (eligible_destination,),
            selected_source="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="selected destination must be"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (eligible_destination,),
            selected_source=selected_source,
            selected_destination="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(AssetError, match="non-grant plan cannot require"):
        AcquisitionPlan(
            "unauthorized",
            "unauthorized",
            (eligible_source,),
            (eligible_destination,),
            required_grant_id="credential-X",
        )
    with pytest.raises(AssetError, match="destination refusal requires"):
        AcquisitionPlan(
            "no-compatible-destination",
            "no-compatible-destination",
            (eligible_source,),
            (eligible_destination,),
        )
    with pytest.raises(AssetError, match="selected source must be an eligible considered"):
        AcquisitionPlan(
            "no-compatible-destination",
            "no-compatible-destination",
            (SourceConsideration(selected_source, False, "unavailable"),),
            (eligible_destination,),
            selected_source=selected_source,
        )
    with pytest.raises(AssetError, match="selected source must be an eligible considered"):
        AcquisitionPlan(
            "no-compatible-destination",
            "no-compatible-destination",
            (SourceConsideration(source("provider-b"), True, "eligible"),),
            (eligible_destination,),
            selected_source=selected_source,
        )
    with pytest.raises(AssetError, match="selected destination must be an eligible considered"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (DestinationConsideration(selected_destination, False, "read-only"),),
            selected_source=selected_source,
            selected_destination=selected_destination,
        )
    with pytest.raises(AssetError, match="selected destination must be an eligible considered"):
        AcquisitionPlan(
            "ready",
            "ready",
            (eligible_source,),
            (DestinationConsideration(destination("models-b"), True, "eligible"),),
            selected_source=selected_source,
            selected_destination=selected_destination,
        )
    for status in ("unauthorized", "source-unavailable"):
        with pytest.raises(AssetError, match="cannot select a source"):
            AcquisitionPlan(
                status,  # type: ignore[arg-type]
                status,
                (eligible_source,),
                (eligible_destination,),
                selected_source=selected_source,
            )
    destination_ambiguous = AcquisitionPlan(
        "ambiguous",
        "conflicting-destination-facts",
        (eligible_source,),
        (eligible_destination,),
        selected_source=selected_source,
    )
    assert destination_ambiguous.selected_source == selected_source
    for status in ("source-unavailable", "unauthorized", "ambiguous"):
        outcome = AcquisitionPlan(
            status,  # type: ignore[arg-type]
            status,
            (eligible_source,),
            (eligible_destination,),
        )
        assert outcome.selected_source is None


def test_source_cannot_self_corroborate_as_provider_mirror() -> None:
    with pytest.raises(AssetError, match="mirrors contain an invalid row"):
        plan_acquisition(
            request(),
            (source(),),
            (source(),),  # type: ignore[arg-type]
            (destination(),),
        )


def test_planning_performs_zero_file_network_or_service_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("planner attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    plan = plan_acquisition(request(), (source(),), (mirror(),), (destination(),))
    assert plan.status == "ready"
