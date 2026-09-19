"""Authenticated attention route-token substrate proofs."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import cast

import pytest
from dinkster_inference import (
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    build_runtime_identity_from_facts,
)
from dinkster_protocol import (
    ATTENTION_ROLES,
    AttentionCapabilityEvidence,
    AttentionPolicy,
    AttentionPolicyConfig,
    AttentionRoute,
    AttentionRouteToken,
    attention_capability_evidence_from_wire,
    attention_capability_evidence_to_wire,
    attention_policy_config_from_wire,
    attention_policy_config_to_wire,
    attention_route_token_from_wire,
    attention_route_token_to_wire,
    automatic_attention_route,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
    resolve_attention_runtime_status,
    resolve_role_policy,
)
from dinkster_workers.host import AttentionRouteDiscoveryError, discover_attention_route_token


def route_token(*, policy: str = "auto", torch_version: str = "2.13.0") -> AttentionRouteToken:
    return AttentionRouteToken(
        version=1,
        routes=tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
        provider_versions=(("torch", torch_version),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime=torch_version,
        requested_policy=policy,  # type: ignore[arg-type]
    )


def route_token_v2(
    *, overrides: tuple[tuple[str, str], ...] = (("flux", "dinkster_kitchen_int8"),)
) -> AttentionRouteToken:
    overridden = dict(overrides)
    return AttentionRouteToken(
        version=2,
        routes=tuple(
            AttentionRoute(role, "dinkster_kitchen_int8", "sdpa")
            if overridden.get(role) == "dinkster_kitchen_int8"
            else AttentionRoute(role, "sdpa")
            for role in ATTENTION_ROLES
        ),
        provider_versions=(("dinkster-kitchen", "0.2.31"), ("torch", "2.13.0")),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        requested_policy="auto",
        requested_role_policies=overrides,  # type: ignore[arg-type]
    )


def capability_evidence(
    *,
    kitchen: bool = False,
    sage: bool = False,
    sol: bool = False,
    device_kind: str = "cpu",
) -> AttentionCapabilityEvidence:
    policies: tuple[AttentionPolicy, ...] = ("sdpa",)
    if sage:
        policies = (*policies, "sage")
    if sol:
        policies = (*policies, "sol")
    if kitchen:
        policies = (*policies, "dinkster_kitchen_int8")
    providers = (("torch", "2.13.0"),)
    if device_kind == "rocm":
        providers = (("hip", "7.0"), *providers)
    if sage:
        providers = (("sageattention", "2.2.0"), *providers)
    if kitchen or sol:
        providers = (("dinkster-kitchen", "0.2.31"), *providers)
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=policies,
        provider_versions=providers,
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind=device_kind,
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
    )


def identity(token: AttentionRouteToken | None = None) -> str:
    return build_runtime_identity_from_facts(
        "dinkster.sd15",
        ("family=dinkster.sd15", "component=diffusion"),
        diffusion_dtype="float16",
        text_dtype="float32",
        vae_dtype="float32",
        fp8_matmul=False,
        attention_policy="auto" if token is None else token.requested_policy,
        attention_route_token=token,
    )


def test_route_token_is_frozen_canonical_and_strictly_round_trips() -> None:
    token = route_token()
    assert attention_route_token_from_wire(attention_route_token_to_wire(token)) == token
    assert tuple(route.role for route in token.routes) == ATTENTION_ROLES
    with pytest.raises(FrozenInstanceError):
        token.device_kind = "cuda"  # type: ignore[misc]
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(token, provider_versions=(("torch", "2"), ("torch", "1")))
    with pytest.raises(ValueError, match="canonical role ordering"):
        replace(token, routes=tuple(reversed(token.routes)))


def test_capability_evidence_is_frozen_canonical_and_strictly_round_trips() -> None:
    evidence = capability_evidence(kitchen=True)
    wire = attention_capability_evidence_to_wire(evidence)
    assert attention_capability_evidence_from_wire(wire) == evidence
    assert json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=True) == (
        '{"adapterContractRevision":"dinkster.attention-kernel.v1",'
        '"availablePolicies":["sdpa","dinkster_kitchen_int8"],"deviceKind":"cpu",'
        '"deviceSm":null,"providerVersions":[["dinkster-kitchen","0.2.31"],'
        '["torch","2.13.0"]],"sdpaTorchRuntime":"2.13.0","version":1}'
    )
    with pytest.raises(FrozenInstanceError):
        evidence.device_kind = "cuda"  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonical policy ordering"):
        replace(evidence, available_policies=tuple(reversed(evidence.available_policies)))
    with pytest.raises(ValueError, match="must be concrete"):
        replace(evidence, available_policies=("auto", "sdpa"))
    with pytest.raises(ValueError, match="must include sdpa"):
        replace(
            evidence,
            available_policies=("dinkster_kitchen_int8",),
        )
    with pytest.raises(ValueError, match="must match"):
        replace(evidence, available_policies=("sdpa",))
    with pytest.raises(ValueError, match="names must be unique"):
        replace(
            evidence,
            provider_versions=(("torch", "2.13.0"), ("torch", "2.14.0")),
            available_policies=("sdpa",),
        )
    with pytest.raises(ValueError, match="requires a torch provider"):
        replace(evidence, provider_versions=(), available_policies=("sdpa",))
    with pytest.raises(ValueError, match="must match sdpa_torch_runtime"):
        replace(
            evidence,
            provider_versions=(("torch", "2.14.0"),),
            available_policies=("sdpa",),
        )


def test_attention_policy_config_wire_is_strict_and_canonical() -> None:
    raw = {
        "requestedPolicy": "auto",
        "requestedRolePolicies": [
            ["qwen", "flash"],
            ["flux", "dinkster_kitchen_int8"],
        ],
    }
    config = attention_policy_config_from_wire(raw)
    assert config.requested_role_policies == (
        ("flux", "dinkster_kitchen_int8"),
        ("qwen", "flash"),
    )
    assert attention_policy_config_to_wire(config) == {
        "requestedPolicy": "auto",
        "requestedRolePolicies": [
            ["flux", "dinkster_kitchen_int8"],
            ["qwen", "flash"],
        ],
    }


@pytest.mark.parametrize(
    ("role_policies", "message"),
    (
        (["flux"], "must be a pair"),
        ([["missing", "flash"]], "unknown attention role"),
        ([["flux", "flash"], ["flux", "sage"]], "at most once"),
        ([["flux", "missing"]], "unsupported attention policy"),
        ([["flux", "auto"]], "must name a concrete policy"),
        ([["flux", "sdpa"]], "is a no-op"),
    ),
)
def test_attention_policy_config_wire_rejects_malformed_role_overrides(
    role_policies: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        attention_policy_config_from_wire(
            {
                "requestedPolicy": "auto",
                "requestedRolePolicies": role_policies,
            }
        )


@pytest.mark.parametrize(
    "raw",
    (
        None,
        {},
        {"requestedPolicy": "auto", "requestedRolePolicies": [], "extra": True},
        {"requestedPolicy": "auto", "requestedRolePolicies": {}},
    ),
)
def test_attention_policy_config_wire_rejects_invalid_envelopes(raw: object) -> None:
    with pytest.raises(ValueError):
        attention_policy_config_from_wire(raw)


def test_policy_config_is_frozen_sparse_canonical_and_normalizes_noops() -> None:
    config = AttentionPolicyConfig(requested_role_policies=(("flux", "dinkster_kitchen_int8"),))
    with pytest.raises(FrozenInstanceError):
        config.requested_policy = "sdpa"  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable pairs"):
        AttentionPolicyConfig(requested_role_policies=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical role ordering"):
        AttentionPolicyConfig(
            requested_role_policies=(
                ("vae", "dinkster_kitchen_int8"),
                ("flux", "dinkster_kitchen_int8"),
            )
        )
    assert (
        AttentionPolicyConfig(requested_role_policies=(("flux", "sdpa"),)).requested_role_policies
        == ()
    )
    assert AttentionPolicyConfig(
        requested_policy="dinkster_kitchen_int8",
        requested_role_policies=(("flux", "sdpa"),),
    ).requested_role_policies == (("flux", "sdpa"),)
    with pytest.raises(ValueError, match="leave at least one role"):
        AttentionPolicyConfig(
            requested_role_policies=tuple(
                (role, "dinkster_kitchen_int8") for role in ATTENTION_ROLES
            )  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad-version", "bad-policies"])
def test_capability_evidence_wire_refuses_malformed_fields(mutation: str) -> None:
    wire = attention_capability_evidence_to_wire(capability_evidence())
    if mutation == "missing":
        del wire["deviceKind"]
    elif mutation == "extra":
        wire["forged"] = True
    elif mutation == "bad-version":
        wire["version"] = True
    else:
        wire["availablePolicies"] = "sdpa"
    with pytest.raises((TypeError, ValueError)):
        attention_capability_evidence_from_wire(wire)


def test_derivation_preserves_explicit_v1_bytes_and_mints_sparse_v4() -> None:
    evidence = capability_evidence(kitchen=True)
    derived_v1 = derive_attention_route_token(
        evidence, AttentionPolicyConfig(requested_policy="sdpa")
    )
    normalized_v1 = derive_attention_route_token(
        capability_evidence(),
        AttentionPolicyConfig(requested_policy="sdpa", requested_role_policies=(("flux", "sdpa"),)),
    )
    expected_v1 = route_token(policy="sdpa")
    assert canonical_attention_route_token_bytes(
        derived_v1
    ) == canonical_attention_route_token_bytes(expected_v1)
    assert canonical_attention_route_token_bytes(
        normalized_v1
    ) == canonical_attention_route_token_bytes(expected_v1)
    assert derived_v1.provider_versions == (("torch", "2.13.0"),)

    config = AttentionPolicyConfig(
        requested_policy="auto",
        requested_role_policies=(("flux", "dinkster_kitchen_int8"),),
    )
    derived_v4 = derive_attention_route_token(evidence, config)
    routes = {route.role: route for route in derived_v4.routes}
    assert derived_v4.version == 4
    assert routes["flux"] == AttentionRoute("flux", "dinkster_kitchen_int8", "sdpa")
    assert routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")


@pytest.mark.parametrize("device_kind", ("cpu", "cuda", "xpu", "mps"))
def test_derivation_routes_auto_from_capabilities_and_falls_back_unavailable_policies(
    device_kind: str,
) -> None:
    evidence = capability_evidence(device_kind=device_kind)
    auto = derive_attention_route_token(evidence, AttentionPolicyConfig())
    auto_routes = {route.role: route for route in auto.routes}
    assert auto.version == 4
    assert auto.requested_policy == "auto"
    assert (auto_routes["vae"].primary, auto_routes["vae"].fallback) == (
        "sdpa",
        "bounded",
    )
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in auto_routes.items()
        if role != "vae"
    )
    assert (
        derive_attention_route_token(
            evidence, AttentionPolicyConfig(requested_policy="sdpa")
        ).requested_policy
        == "sdpa"
    )
    flash = derive_attention_route_token(
        evidence,
        AttentionPolicyConfig(requested_policy="flash"),
    )
    assert flash.version == 3
    assert flash.requested_policy == "flash"
    assert flash.requested_role_policies == ()
    assert all((route.primary, route.fallback) == ("sdpa", None) for route in flash.routes)
    assert flash.provider_versions == (("torch", "2.13.0"),)

    kitchen_override = derive_attention_route_token(
        evidence,
        AttentionPolicyConfig(requested_role_policies=(("flux", "dinkster_kitchen_int8"),)),
    )
    assert kitchen_override.version == 4
    assert kitchen_override.requested_role_policies == (("flux", "dinkster_kitchen_int8"),)
    kitchen_routes = {route.role: route for route in kitchen_override.routes}
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in kitchen_routes.items()
        if role != "vae"
    )
    assert kitchen_routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")
    assert kitchen_override.provider_versions == (("torch", "2.13.0"),)


def test_capability_evidence_does_not_split_selected_runtime_identity() -> None:
    plain = capability_evidence()
    kitchen_capable = capability_evidence(kitchen=True)
    config = AttentionPolicyConfig()
    plain_token = derive_attention_route_token(plain, config)
    kitchen_capable_token = derive_attention_route_token(kitchen_capable, config)
    assert plain_token == kitchen_capable_token
    assert identity(plain_token) == identity(kitchen_capable_token)
    encoded_evidence = json.dumps(
        attention_capability_evidence_to_wire(kitchen_capable),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert encoded_evidence not in identity(kitchen_capable_token)


def test_sage_evidence_routes_and_provider_filtering() -> None:
    evidence = capability_evidence(sage=True)
    with pytest.raises(ValueError, match="sageattention provider evidence must match"):
        replace(evidence, provider_versions=(("torch", "2.13.0"),))
    plain = capability_evidence()
    with pytest.raises(ValueError, match="sageattention provider evidence must match"):
        replace(plain, provider_versions=(("sageattention", "2.2.0"), ("torch", "2.13.0")))
    token = derive_attention_route_token(evidence, AttentionPolicyConfig(requested_policy="sage"))
    assert all((route.primary, route.fallback) == ("sage", "sdpa") for route in token.routes)
    assert dict(token.provider_versions) == {"sageattention": "2.2.0", "torch": "2.13.0"}
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(token, routes=tuple(AttentionRoute(route.role, "sage") for route in token.routes))
    auto = derive_attention_route_token(evidence, AttentionPolicyConfig())
    assert all((route.primary, route.fallback) == ("sage", "sdpa") for route in auto.routes)
    assert dict(auto.provider_versions) == {"sageattention": "2.2.0", "torch": "2.13.0"}
    # Sage evidence appearing on a worker must not change the token bytes of
    # any non-sage request, kitchen included.
    assert derive_attention_route_token(
        capability_evidence(kitchen=True, sage=True),
        AttentionPolicyConfig(requested_policy="dinkster_kitchen_int8"),
    ) == derive_attention_route_token(
        capability_evidence(kitchen=True),
        AttentionPolicyConfig(requested_policy="dinkster_kitchen_int8"),
    )
    missing_sage = derive_attention_route_token(
        plain, AttentionPolicyConfig(requested_policy="sage")
    )
    assert missing_sage.version == 3
    assert all((route.primary, route.fallback) == ("sdpa", None) for route in missing_sage.routes)
    assert missing_sage.provider_versions == (("torch", "2.13.0"),)
    overridden = derive_attention_route_token(
        evidence,
        AttentionPolicyConfig(requested_role_policies=(("flux", "sage"),)),
    )
    assert overridden.version == 4
    routes = {route.role: route for route in overridden.routes}
    assert (routes["flux"].primary, routes["flux"].fallback) == ("sage", "sdpa")
    assert all(
        (route.primary, route.fallback) == ("sage", "sdpa")
        for role, route in routes.items()
        if role != "flux"
    )
    assert dict(overridden.provider_versions) == {"sageattention": "2.2.0", "torch": "2.13.0"}


def test_sol_evidence_routes_and_shares_kitchen_provider_identity() -> None:
    evidence = capability_evidence(sol=True)
    with pytest.raises(ValueError, match="kitchen attention availability"):
        replace(evidence, provider_versions=(("torch", "2.13.0"),))
    token = derive_attention_route_token(evidence, AttentionPolicyConfig(requested_policy="sol"))
    assert all((route.primary, route.fallback) == ("sol", "sdpa") for route in token.routes)
    assert dict(token.provider_versions) == {"dinkster-kitchen": "0.2.31", "torch": "2.13.0"}
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(token, routes=tuple(AttentionRoute(route.role, "sol") for route in token.routes))
    assert derive_attention_route_token(
        evidence, AttentionPolicyConfig()
    ) == derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())

    kitchen = capability_evidence(kitchen=True, sol=True)
    overridden = derive_attention_route_token(
        kitchen,
        AttentionPolicyConfig(
            requested_policy="dinkster_kitchen_int8",
            requested_role_policies=(("flux", "sol"),),
        ),
    )
    routes = {route.role: route for route in overridden.routes}
    assert (routes["flux"].primary, routes["flux"].fallback) == ("sol", "sdpa")
    assert all(
        (route.primary, route.fallback) == ("dinkster_kitchen_int8", "sdpa")
        for role, route in routes.items()
        if role != "flux"
    )
    assert dict(overridden.provider_versions) == {
        "dinkster-kitchen": "0.2.31",
        "torch": "2.13.0",
    }


def test_auto_route_uses_bounded_vae_on_rocm_and_ignores_explicit_only_providers() -> None:
    rocm = capability_evidence(device_kind="rocm")
    routes = {
        role: automatic_attention_route(role, rocm.available_policies, rocm.device_kind)
        for role in ATTENTION_ROLES
    }
    assert routes["vae"] == AttentionRoute("vae", "bounded")
    assert all(route.primary == "sdpa" for role, route in routes.items() if role != "vae")

    portable = derive_attention_route_token(
        capability_evidence(kitchen=True, sol=True), AttentionPolicyConfig()
    )
    assert portable == derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())
    assert identity(portable) == identity(
        derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())
    )


def test_v4_auto_route_round_trip_and_forged_route_refusal() -> None:
    plain = derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())
    token = derive_attention_route_token(capability_evidence(sage=True), AttentionPolicyConfig())
    rocm = derive_attention_route_token(
        capability_evidence(device_kind="rocm"), AttentionPolicyConfig()
    )
    assert len({identity(plain), identity(token), identity(rocm)}) == 3
    wire = attention_route_token_to_wire(token)
    assert wire["version"] == 4
    assert attention_route_token_from_wire(wire) == token
    forged = dict(wire)
    forged["routes"] = [
        dict(route, primary="sdpa", fallback=None) if route["role"] == "flux" else route
        for route in cast("list[dict[str, object]]", wire["routes"])
    ]
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        attention_route_token_from_wire(forged)


def test_v3_derivation_preserves_supported_roles_and_only_their_providers() -> None:
    token = derive_attention_route_token(
        capability_evidence(kitchen=True, sage=True),
        AttentionPolicyConfig(
            requested_policy="flash",
            requested_role_policies=(("flux", "dinkster_kitchen_int8"),),
        ),
    )
    routes = {route.role: route for route in token.routes}
    assert token.version == 3
    assert (routes["flux"].primary, routes["flux"].fallback) == (
        "dinkster_kitchen_int8",
        "sdpa",
    )
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in routes.items()
        if role != "flux"
    )
    assert token.provider_versions == (("dinkster-kitchen", "0.2.31"), ("torch", "2.13.0"))


@pytest.mark.parametrize("mutation", ["missing", "extra", "bad-version", "bad-route"])
def test_route_token_wire_refuses_missing_extra_and_malformed_fields(mutation: str) -> None:
    wire = attention_route_token_to_wire(route_token())
    if mutation == "missing":
        del wire["deviceKind"]
    elif mutation == "extra":
        wire["forged"] = True
    elif mutation == "bad-version":
        wire["version"] = True
    else:
        wire["routes"] = [{"role": "unet", "primary": "sdpa", "fallback": None}]
    with pytest.raises((TypeError, ValueError)):
        attention_route_token_from_wire(wire)


def test_v1_wire_bytes_are_pinned() -> None:
    encoded = canonical_attention_route_token_bytes(route_token())
    assert encoded == (
        b'{"adapterContractRevision":"dinkster.attention-kernel.v1","deviceKind":"cpu",'
        b'"deviceSm":null,"providerVersions":[["torch","2.13.0"]],"requestedPolicy":"auto",'
        b'"routes":[{"fallback":null,"primary":"sdpa","role":"unet"},'
        b'{"fallback":null,"primary":"sdpa","role":"flux"},'
        b'{"fallback":null,"primary":"sdpa","role":"vae"},'
        b'{"fallback":null,"primary":"sdpa","role":"clip"},'
        b'{"fallback":null,"primary":"sdpa","role":"t5"},'
        b'{"fallback":null,"primary":"sdpa","role":"qwen"}],'
        b'"sdpaTorchRuntime":"2.13.0","version":1}'
    )


def test_v2_wire_bytes_remain_pinned() -> None:
    assert canonical_attention_route_token_bytes(route_token_v2()) == (
        b'{"adapterContractRevision":"dinkster.attention-kernel.v1","deviceKind":"cpu",'
        b'"deviceSm":null,"providerVersions":[["dinkster-kitchen","0.2.31"],'
        b'["torch","2.13.0"]],"requestedPolicy":"auto",'
        b'"requestedRolePolicies":[["flux","dinkster_kitchen_int8"]],'
        b'"routes":[{"fallback":null,"primary":"sdpa","role":"unet"},'
        b'{"fallback":"sdpa","primary":"dinkster_kitchen_int8","role":"flux"},'
        b'{"fallback":null,"primary":"sdpa","role":"vae"},'
        b'{"fallback":null,"primary":"sdpa","role":"clip"},'
        b'{"fallback":null,"primary":"sdpa","role":"t5"},'
        b'{"fallback":null,"primary":"sdpa","role":"qwen"}],'
        b'"sdpaTorchRuntime":"2.13.0","version":2}'
    )


def test_v3_wire_round_trip_requires_role_policies_even_when_empty() -> None:
    token = derive_attention_route_token(
        capability_evidence(), AttentionPolicyConfig(requested_policy="flash")
    )
    wire = attention_route_token_to_wire(token)
    assert wire["version"] == 3
    assert wire["requestedRolePolicies"] == []
    assert attention_route_token_from_wire(wire) == token

    missing = dict(wire)
    del missing["requestedRolePolicies"]
    with pytest.raises(ValueError, match="missing or extra fields"):
        attention_route_token_from_wire(missing)

    malformed = dict(wire)
    malformed["routes"] = [
        dict(route, primary="sage") if route["role"] == "unet" else route
        for route in cast("list[dict[str, object]]", wire["routes"])
    ]
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        attention_route_token_from_wire(malformed)

    for legacy_version in (1, 2):
        legacy = dict(wire)
        legacy["version"] = legacy_version
        if legacy_version == 1:
            del legacy["requestedRolePolicies"]
        with pytest.raises(ValueError):
            attention_route_token_from_wire(legacy)


def test_v3_validator_requires_real_fallback_and_requested_routes() -> None:
    token = derive_attention_route_token(
        capability_evidence(), AttentionPolicyConfig(requested_policy="flash")
    )
    with pytest.raises(ValueError, match="requires a portable fallback"):
        replace(
            token,
            routes=tuple(AttentionRoute(route.role, "flash") for route in token.routes),
        )
    with pytest.raises(ValueError, match="requires a portable fallback"):
        replace(
            token,
            requested_policy="sdpa",
        )
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(
            token,
            routes=(AttentionRoute("unet", "sage"), *token.routes[1:]),
        )


def test_role_policy_overrides_are_version_keyed_on_the_wire() -> None:
    v1_wire = attention_route_token_to_wire(route_token())
    assert "requestedRolePolicies" not in v1_wire
    with pytest.raises(ValueError, match="missing or extra fields"):
        attention_route_token_from_wire(dict(v1_wire) | {"requestedRolePolicies": []})
    token = route_token_v2()
    v2_wire = attention_route_token_to_wire(token)
    assert v2_wire["requestedRolePolicies"] == [["flux", "dinkster_kitchen_int8"]]
    assert attention_route_token_from_wire(v2_wire) == token
    missing = dict(v2_wire)
    del missing["requestedRolePolicies"]
    with pytest.raises(ValueError, match="missing or extra fields"):
        attention_route_token_from_wire(missing)


def test_role_policy_overrides_refuse_non_canonical_and_no_op_forms() -> None:
    token = route_token_v2()
    with pytest.raises(ValueError, match="cannot carry role policy overrides"):
        replace(token, version=1)
    with pytest.raises(ValueError, match="requires role policy overrides"):
        route_token_v2(overrides=())
    with pytest.raises(ValueError, match="unknown attention role"):
        route_token_v2(overrides=(("forged", "dinkster_kitchen_int8"),))
    with pytest.raises(ValueError, match="at most once"):
        route_token_v2(overrides=(("flux", "dinkster_kitchen_int8"), ("flux", "sdpa")))
    with pytest.raises(ValueError, match="leave at least one role"):
        route_token_v2(overrides=tuple((role, "dinkster_kitchen_int8") for role in ATTENTION_ROLES))
    with pytest.raises(ValueError, match="canonical role ordering"):
        route_token_v2(
            overrides=(("vae", "dinkster_kitchen_int8"), ("flux", "dinkster_kitchen_int8"))
        )
    with pytest.raises(ValueError, match="not 'auto'"):
        replace(route_token(policy="sdpa"), version=2, requested_role_policies=(("flux", "auto"),))
    with pytest.raises(ValueError, match="is a no-op"):
        replace(route_token(policy="sdpa"), version=2, requested_role_policies=(("flux", "sdpa"),))
    with pytest.raises(ValueError, match="is a no-op"):
        route_token_v2(overrides=(("flux", "sdpa"),))


def test_routes_must_match_effective_policy_at_construction() -> None:
    token = derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(
            token,
            routes=tuple(
                replace(route, fallback="dinkster_kitchen_int8") if route.role == "flux" else route
                for route in token.routes
            ),
        )
    version_two = route_token_v2()
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(
            version_two,
            routes=tuple(
                replace(route, primary="sdpa") if route.role == "flux" else route
                for route in version_two.routes
            ),
        )
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(
            version_two,
            routes=tuple(
                replace(route, fallback=None) if route.role == "flux" else route
                for route in version_two.routes
            ),
        )


def test_resolve_role_policy_is_the_single_merge_point() -> None:
    overrides: tuple[tuple[str, AttentionPolicy], ...] = (("flux", "dinkster_kitchen_int8"),)
    assert resolve_role_policy("auto", overrides, "flux") == "dinkster_kitchen_int8"
    assert resolve_role_policy("auto", overrides, "vae") == "auto"
    assert resolve_role_policy("sdpa", (), "unet") == "sdpa"
    with pytest.raises(ValueError, match="unknown attention role"):
        resolve_role_policy("auto", overrides, "forged")
    status = resolve_attention_runtime_status("auto", route_token_v2(overrides=overrides))
    assert status.requested_role_policies == overrides


def test_role_policy_overrides_rotate_identity() -> None:
    plain = route_token()
    overridden = route_token_v2()
    assert identity(overridden) != identity(plain)
    assert identity(overridden) != identity()


def test_fallback_request_identity_differs_from_explicit_sdpa() -> None:
    evidence = capability_evidence()
    fallback = derive_attention_route_token(
        evidence, AttentionPolicyConfig(requested_policy="flash")
    )
    explicit_sdpa = derive_attention_route_token(
        evidence, AttentionPolicyConfig(requested_policy="sdpa")
    )
    assert fallback.routes == explicit_sdpa.routes
    assert identity(fallback) != identity(explicit_sdpa)


def test_missing_token_is_bounded_auto_only_and_preserves_legacy_identity() -> None:
    status = resolve_attention_runtime_status("auto", None)
    assert status.authenticated is False
    assert {route.primary for route in status.routes} == {"sdpa"}
    assert status.provider_versions == ()
    with pytest.raises(ValueError, match="requires an authenticated"):
        resolve_attention_runtime_status("sdpa", None)
    assert identity() == identity(None)


def test_authenticated_token_rotates_identity_with_runtime_facts() -> None:
    first = route_token(torch_version="2.13.0")
    second = route_token(torch_version="2.14.0")
    assert identity(first) != identity()
    assert identity(first) != identity(second)


def test_authenticated_identity_has_unambiguous_provider_encoding() -> None:
    token = route_token()
    provider_colon_left = identity(replace(token, provider_versions=(("a:b", "c"),)))
    provider_colon_right = identity(replace(token, provider_versions=(("a", "b:c"),)))

    assert provider_colon_left != provider_colon_right


def test_reconstruction_recipe_preserves_and_authenticates_route_facts() -> None:
    token = route_token()
    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef(
                    digest="blake3:" + "0" * 64,
                    name="weights.safetensors",
                    size=1,
                ),
            ),
        ),
        family_id="dinkster.sd15",
        component_identity=("family=dinkster.sd15", "component=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float16",
            text_dtype="float32",
            vae_dtype="float32",
            fp8_matmul=False,
            attention_route_token=token,
        ),
    )
    assert recipe.knobs.attention_route_token == token
    assert recipe.runtime_identity == identity(token)
    with pytest.raises(ValueError, match="does not match"):
        replace(recipe.knobs, attention_policy="sdpa")


def test_worker_discovery_accepts_absent_top_level_only_and_rejects_nested_failure() -> None:
    def missing_top_level(_name: str) -> object:
        raise ModuleNotFoundError(name="dinkster_inference_torch")

    def missing_nested(_name: str) -> object:
        raise ModuleNotFoundError(name="torch")

    assert discover_attention_route_token(missing_top_level) is None
    with pytest.raises(AttentionRouteDiscoveryError, match="nested import"):
        discover_attention_route_token(missing_nested)


def test_worker_discovery_rejects_missing_probe_and_malformed_evidence() -> None:
    with pytest.raises(AttentionRouteDiscoveryError, match="no capability discovery export"):
        discover_attention_route_token(lambda _name: SimpleNamespace())
    with pytest.raises(AttentionRouteDiscoveryError, match="malformed evidence"):
        discover_attention_route_token(
            lambda _name: SimpleNamespace(
                discover_attention_capabilities=lambda: {"forged": True},
                discover_attention_route_token=lambda _policy: route_token(),
            )
        )
    with pytest.raises(AttentionRouteDiscoveryError, match="malformed token"):
        discover_attention_route_token(
            lambda _name: SimpleNamespace(
                discover_attention_capabilities=lambda: capability_evidence(),
                discover_attention_route_token=lambda _policy: {"forged": True},
            )
        )
    token = derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())
    assert (
        discover_attention_route_token(
            lambda _name: SimpleNamespace(
                discover_attention_capabilities=lambda: capability_evidence(),
                discover_attention_route_token=lambda _policy: token,
            )
        )
        == token
    )
    with pytest.raises(AttentionRouteDiscoveryError, match="does not match"):
        discover_attention_route_token(
            lambda _name: SimpleNamespace(
                discover_attention_capabilities=lambda: capability_evidence(),
                discover_attention_route_token=lambda _policy: route_token(torch_version="forged"),
            )
        )


def test_worker_discovery_ignores_environment_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def probe(policy: object) -> AttentionRouteToken:
        seen.append(policy)
        assert policy == "auto"
        return derive_attention_route_token(capability_evidence(), AttentionPolicyConfig())

    module = SimpleNamespace(
        configure_amd_miopen=lambda: seen.append("configure"),
        discover_attention_capabilities=lambda: capability_evidence(),
        discover_attention_route_token=probe,
    )
    monkeypatch.setenv("DINKSTER_ATTENTION_POLICY", "fast")
    assert discover_attention_route_token(lambda _name: module) == derive_attention_route_token(
        capability_evidence(), AttentionPolicyConfig()
    )
    assert seen == ["configure", "auto"]
