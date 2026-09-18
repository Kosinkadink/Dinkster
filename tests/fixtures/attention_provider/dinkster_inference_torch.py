"""Torch-free authenticated attention evidence for isolated worker tests."""

from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    derive_attention_route_token,
)


def discover_attention_capabilities() -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )


def discover_attention_route_token(policy: str = "auto") -> AttentionRouteToken:
    if policy not in ("auto", "sdpa"):
        raise RuntimeError(f"attention policy {policy!r} is unavailable")
    return derive_attention_route_token(
        discover_attention_capabilities(), AttentionPolicyConfig(policy)
    )
