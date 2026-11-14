"""Authenticated attention route-token substrate proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest
from dinkster_protocol import (
    ATTENTION_ROLES,
    AttentionCapabilityEvidence,
    AttentionPolicy,
    AttentionPolicyConfig,
    AttentionRoute,
    attention_policy_config_from_wire,
    attention_route_token_from_wire,
    attention_route_token_to_wire,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
)
from dinkster_workers import ExecutionContext
from dinkster_workers.execution import use_execution_context

from tests.async_scenarios import run_scenario


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
        policies = (*policies, "comfy_kitchen_int8")
    providers = (("torch", "2.13.0"),)
    if device_kind == "rocm":
        providers = (("hip", "7.0"), *providers)
    if sage:
        providers = (("sageattention", "2.2.0"), *providers)
    if kitchen or sol:
        providers = (("comfy-kitchen", "0.2.31"), *providers)
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=policies,
        provider_versions=providers,
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind=device_kind,
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
    )


def test_execution_context_rejects_cross_worker_route_token_replay() -> None:
    sm80 = replace(capability_evidence(device_kind="cuda"), device_sm=80)
    sm90 = replace(sm80, device_sm=90)
    sm80_token = derive_attention_route_token(sm80, AttentionPolicyConfig())
    sm80_token = replace(sm80_token, routes=tuple(reversed(sm80_token.routes)))
    sm90_token = derive_attention_route_token(sm90, AttentionPolicyConfig())

    ExecutionContext(
        "native",
        None,
        attention_route_token=sm80_token,
        attention_capabilities=sm80,
    )
    assert canonical_attention_route_token_bytes(
        sm80_token
    ) == canonical_attention_route_token_bytes(
        derive_attention_route_token(sm80, AttentionPolicyConfig())
    )
    with pytest.raises(ValueError, match="does not match worker capabilities"):
        ExecutionContext(
            "native",
            None,
            attention_route_token=sm80_token,
            attention_capabilities=sm90,
        )
    with pytest.raises(ValueError, match="does not match worker capabilities"):
        ExecutionContext(
            "native",
            None,
            attention_route_token=sm90_token,
            attention_capabilities=sm80,
        )


@pytest.mark.parametrize(
    ("requested", "overrides", "expected"),
    (
        ("sdpa", [["flux", "sdpa"]], ("sdpa", None)),
        ("sdpa", [[role, "sage"] for role in ATTENTION_ROLES], ("sage", "sdpa")),
        ("auto", [["vae", "sdpa"]], ("sdpa", None)),
    ),
)
def test_policy_wire_admits_redundant_and_complete_role_overrides(
    requested: str, overrides: list[list[str]], expected: tuple[str, str | None]
) -> None:
    evidence = capability_evidence(sage=True)
    config = attention_policy_config_from_wire(
        {"requestedPolicy": requested, "requestedRolePolicies": overrides}
    )
    token = derive_attention_route_token(evidence, config)
    if token.version == 1:
        wire = attention_route_token_to_wire(token)
        wire.update(version=2, requestedRolePolicies=overrides)
        redundant_token = attention_route_token_from_wire(wire)
        assert canonical_attention_route_token_bytes(
            redundant_token
        ) == canonical_attention_route_token_bytes(token)
        token = redundant_token
    ExecutionContext(
        "native",
        None,
        attention_policy=config.requested_policy,
        attention_route_token=token,
        attention_capabilities=evidence,
    )
    routes = {route.role: (route.primary, route.fallback) for route in token.routes}
    assert routes["vae"] == expected
    if requested == "sdpa":
        assert all(route == expected for route in routes.values())
    else:
        assert config.requested_role_policies == (("vae", "sdpa"),)


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
        AttentionPolicyConfig(requested_role_policies=(("flux", "comfy_kitchen_int8"),)),
    )
    assert kitchen_override.version == 4
    assert kitchen_override.requested_role_policies == (("flux", "comfy_kitchen_int8"),)
    kitchen_routes = {route.role: route for route in kitchen_override.routes}
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in kitchen_routes.items()
        if role != "vae"
    )
    assert kitchen_routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")
    assert kitchen_override.provider_versions == (("torch", "2.13.0"),)


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
    auto_routes = {route.role: route for route in auto.routes}
    assert auto_routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in auto_routes.items()
        if role != "vae"
    )
    assert auto.provider_versions == (("torch", "2.13.0"),)
    # Sage evidence appearing on a worker must not change the token bytes of
    # any non-sage request, kitchen included.
    assert derive_attention_route_token(
        capability_evidence(kitchen=True, sage=True),
        AttentionPolicyConfig(requested_policy="comfy_kitchen_int8"),
    ) == derive_attention_route_token(
        capability_evidence(kitchen=True),
        AttentionPolicyConfig(requested_policy="comfy_kitchen_int8"),
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
    assert routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in routes.items()
        if role not in ("flux", "vae")
    )
    assert dict(overridden.provider_versions) == {"sageattention": "2.2.0", "torch": "2.13.0"}


@pytest.mark.parametrize("nested", (False, True))
def test_in_process_sampling_preserves_invocation_distributed_configuration(nested: bool) -> None:
    from collections.abc import Mapping
    from contextlib import nullcontext

    from dinkster_protocol import Invocation, SingleJobMultiGpuExecution
    from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr
    from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker, current_execution_context

    class Probe(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.distributed-context",
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            context = current_execution_context()
            assert context is not None
            return cls.outputs(value=str(context.single_job_multi_gpu_execution))

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = Probe.schema()
        worker = InProcessWorker({schema.node_type: Probe}, registry)
        config = SingleJobMultiGpuExecution(1, 2, "sequence")
        outer = use_execution_context(ExecutionContext(None, None)) if nested else nullcontext()
        with outer:
            result = await worker.invoke(
                Invocation(
                    invocation_id="sampling",
                    node_id="probe",
                    node_type=schema.node_type,
                    inputs={},
                    effective_schema=schema,
                    single_job_multi_gpu_execution=config,
                )
            )
        assert result.error is None
        assert result.outputs is not None
        assert result.outputs["value"].resolve() == str(config)

    run_scenario(scenario())
