"""Authenticated, RPC-clean attention routing evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

AttentionPolicy = Literal[
    "auto", "sdpa", "flash", "xformers", "sage", "sage3", "sol", "comfy_kitchen_int8"
]
ATTENTION_ROLES = ("unet", "flux", "vae", "clip", "t5", "qwen")
ATTENTION_POLICIES = (
    "auto",
    "sdpa",
    "flash",
    "xformers",
    "sage",
    "sage3",
    "sol",
    "comfy_kitchen_int8",
)
ATTENTION_CAPABILITY_EVIDENCE_VERSION = 1
ATTENTION_ROUTE_TOKEN_VERSION = 1
ATTENTION_ROUTE_TOKEN_VERSIONS = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class AttentionRoute:
    role: str
    primary: str
    fallback: str | None = None

    def __post_init__(self) -> None:
        role = cast("object", self.role)
        primary = cast("object", self.primary)
        fallback = cast("object", self.fallback)
        if role not in ATTENTION_ROLES:
            raise ValueError("unknown attention role")
        if not isinstance(primary, str) or not primary:
            raise ValueError("attention primary route must be non-empty")
        if fallback is not None and (not isinstance(fallback, str) or not fallback):
            raise ValueError("attention fallback route must be non-empty or None")


def automatic_attention_route(
    role: str,
    available_policies: Sequence[AttentionPolicy],
    device_kind: str,
) -> AttentionRoute:
    """Choose the deterministic automatic route from authenticated capabilities."""
    if role not in ATTENTION_ROLES:
        raise ValueError("unknown attention role")
    available = {validate_attention_policy(policy) for policy in available_policies}
    if "sdpa" not in available:
        raise ValueError("automatic attention requires SDPA capability")
    if "sage" in available:
        return AttentionRoute(role, "sage", "sdpa")
    if role == "vae" and device_kind == "rocm":
        return AttentionRoute(role, "bounded")
    if role == "vae":
        return AttentionRoute(role, "sdpa", "bounded")
    return AttentionRoute(role, "sdpa")


def _validate_requested_role_policies(
    requested_policy: AttentionPolicy,
    requested_role_policies: object,
    *,
    normalize_noops: bool = False,
) -> tuple[tuple[str, AttentionPolicy], ...]:
    validate_attention_policy(requested_policy)
    if not isinstance(requested_role_policies, tuple) or not all(
        isinstance(pair, tuple) and len(cast("tuple[object, ...]", pair)) == 2
        for pair in cast("tuple[object, ...]", requested_role_policies)
    ):
        raise TypeError("requested role policies must be immutable pairs")
    override_pairs = cast("tuple[tuple[object, object], ...]", requested_role_policies)
    override_roles = tuple(role for role, _ in override_pairs)
    if any(role not in ATTENTION_ROLES for role in override_roles):
        raise ValueError("unknown attention role in requested role policies")
    if len(set(override_roles)) != len(override_roles):
        raise ValueError("requested role policies must name each role at most once")
    canonical = tuple(role for role in ATTENTION_ROLES if role in set(override_roles))
    if override_roles != canonical:
        raise ValueError("requested role policies must use canonical role ordering")
    default_policy = "sdpa" if requested_policy == "auto" else requested_policy
    normalized: list[tuple[str, AttentionPolicy]] = []
    for role, policy_raw in override_pairs:
        policy = validate_attention_policy(policy_raw)
        if policy == "auto":
            raise ValueError("a role policy override must name a concrete policy, not 'auto'")
        if policy == default_policy:
            if normalize_noops:
                continue
            raise ValueError(
                "a role policy override equivalent to the requested policy is a no-op; "
                "omit the role instead"
            )
        normalized.append((cast("str", role), policy))
    if len(normalized) == len(ATTENTION_ROLES):
        raise ValueError(
            "requested role policies must leave at least one role to the default policy"
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class AttentionPolicyConfig:
    requested_policy: AttentionPolicy = "auto"
    requested_role_policies: tuple[tuple[str, AttentionPolicy], ...] = ()

    def __post_init__(self) -> None:
        normalized = _validate_requested_role_policies(
            self.requested_policy,
            cast("object", self.requested_role_policies),
            normalize_noops=True,
        )
        object.__setattr__(self, "requested_role_policies", normalized)


@dataclass(frozen=True, slots=True)
class AttentionCapabilityEvidence:
    version: int
    device_kind: str
    device_sm: int | None
    sdpa_torch_runtime: str
    adapter_contract_revision: str
    available_policies: tuple[AttentionPolicy, ...]
    provider_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != ATTENTION_CAPABILITY_EVIDENCE_VERSION:
            raise ValueError("unsupported attention capability evidence version")
        policies = cast("object", self.available_policies)
        if not isinstance(policies, tuple):
            raise TypeError("available attention policies must be an immutable tuple")
        validated = tuple(
            validate_attention_policy(policy) for policy in cast("tuple[object, ...]", policies)
        )
        if "auto" in validated:
            raise ValueError("available attention policies must be concrete")
        if "sdpa" not in validated:
            raise ValueError("available attention policies must include sdpa")
        canonical = tuple(policy for policy in ATTENTION_POLICIES if policy in set(validated))
        if validated != canonical:
            raise ValueError("available attention policies must use canonical policy ordering")
        providers = cast("object", self.provider_versions)
        if not isinstance(providers, tuple) or not all(
            isinstance(pair, tuple) and len(cast("tuple[object, ...]", pair)) == 2
            for pair in cast("tuple[object, ...]", providers)
        ):
            raise TypeError("provider versions must be immutable pairs")
        if self.provider_versions != tuple(sorted(set(self.provider_versions))):
            raise ValueError("provider versions must be sorted and unique")
        provider_pairs = cast("tuple[tuple[object, object], ...]", providers)
        if not all(
            isinstance(name, str) and name and isinstance(version, str) and version
            for name, version in provider_pairs
        ):
            raise ValueError("provider version names and versions must be non-empty")
        provider_names = tuple(name for name, _ in provider_pairs)
        if len(set(provider_names)) != len(provider_names):
            raise ValueError("provider version names must be unique")
        kitchen_available = bool({"sol", "comfy_kitchen_int8"}.intersection(validated))
        kitchen_versioned = any(name == "comfy-kitchen" for name, _ in provider_pairs)
        if kitchen_available != kitchen_versioned:
            raise ValueError(
                "comfy-kitchen provider evidence must match kitchen attention availability"
            )
        sage_available = "sage" in validated
        sage_versioned = any(name == "sageattention" for name, _ in provider_pairs)
        if sage_available != sage_versioned:
            raise ValueError("sageattention provider evidence must match sage availability")
        for name in ("adapter_contract_revision", "device_kind", "sdpa_torch_runtime"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        torch_version = dict(cast("tuple[tuple[str, str], ...]", provider_pairs)).get("torch")
        if torch_version is None:
            raise ValueError("attention capability evidence requires a torch provider version")
        if torch_version.split("+")[0] != self.sdpa_torch_runtime:
            raise ValueError("torch provider version must match sdpa_torch_runtime")
        if self.device_sm is not None and (type(self.device_sm) is not int or self.device_sm < 0):
            raise ValueError("device_sm must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class AttentionRouteToken:
    version: int
    routes: tuple[AttentionRoute, ...]
    provider_versions: tuple[tuple[str, str], ...]
    adapter_contract_revision: str
    device_kind: str
    device_sm: int | None
    sdpa_torch_runtime: str
    requested_policy: AttentionPolicy
    requested_role_policies: tuple[tuple[str, AttentionPolicy], ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in ATTENTION_ROUTE_TOKEN_VERSIONS:
            raise ValueError("unsupported attention route token version")
        routes = cast("object", self.routes)
        providers = cast("object", self.provider_versions)
        if not isinstance(routes, tuple) or not all(
            isinstance(route, AttentionRoute) for route in cast("tuple[object, ...]", routes)
        ):
            raise TypeError("attention routes must be an immutable tuple")
        if tuple(route.role for route in self.routes) != ATTENTION_ROLES:
            raise ValueError("attention routes must use canonical role ordering")
        if not isinstance(providers, tuple) or not all(
            isinstance(pair, tuple) and len(cast("tuple[object, ...]", pair)) == 2
            for pair in cast("tuple[object, ...]", providers)
        ):
            raise TypeError("provider versions must be immutable pairs")
        if self.provider_versions != tuple(sorted(set(self.provider_versions))):
            raise ValueError("provider versions must be sorted and unique")
        provider_pairs = cast("tuple[tuple[object, object], ...]", providers)
        if not all(
            isinstance(name, str) and name and isinstance(version, str) and version
            for name, version in provider_pairs
        ):
            raise ValueError("provider version names and versions must be non-empty")
        for name in ("adapter_contract_revision", "device_kind", "sdpa_torch_runtime"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.device_sm is not None and (type(self.device_sm) is not int or self.device_sm < 0):
            raise ValueError("device_sm must be a non-negative integer or None")
        _validate_requested_role_policies(
            self.requested_policy,
            cast("object", self.requested_role_policies),
        )
        if self.version == 1 and self.requested_role_policies:
            raise ValueError("a version 1 attention route token cannot carry role policy overrides")
        if self.version == 2 and not self.requested_role_policies:
            raise ValueError("a version 2 attention route token requires role policy overrides")
        if self.version == 4 and self.requested_policy != "auto":
            raise ValueError("a version 4 attention route token requires automatic policy")
        portable_fallback = False
        available: list[AttentionPolicy] = ["sdpa"]
        if any(name == "sageattention" for name, _ in self.provider_versions):
            available.append("sage")
        for route in self.routes:
            effective = resolve_role_policy(
                self.requested_policy, self.requested_role_policies, route.role
            )
            if effective == "auto" and self.version == 4:
                selected = automatic_attention_route(route.role, available, self.device_kind)
                expected_route = (selected.primary, selected.fallback)
            elif effective in ("auto", "sdpa"):
                expected_route = ("sdpa", None)
            elif effective in ("comfy_kitchen_int8", "sage", "sol"):
                expected_route = (effective, "sdpa")
            else:
                expected_route = (effective, None)
            if (
                self.version in (3, 4)
                and effective not in ("auto", "sdpa")
                and (route.primary, route.fallback) == ("sdpa", None)
                and (
                    (effective == "sage" and "sageattention" not in dict(self.provider_versions))
                    or (
                        effective in ("sol", "comfy_kitchen_int8")
                        and "comfy-kitchen" not in dict(self.provider_versions)
                    )
                    or effective in ("flash", "xformers", "sage3")
                )
            ):
                portable_fallback = True
                continue
            if (route.primary, route.fallback) != expected_route:
                raise ValueError(
                    f"attention route for role {route.role!r} is inconsistent with "
                    f"its effective policy {effective!r}"
                )
        if self.version == 3 and not portable_fallback:
            raise ValueError("a version 3 attention route token requires a portable fallback")


@dataclass(frozen=True, slots=True)
class AttentionRuntimeStatus:
    authenticated: bool
    requested_policy: AttentionPolicy
    routes: tuple[AttentionRoute, ...]
    provider_versions: tuple[tuple[str, str], ...]
    adapter_contract_revision: str
    device_kind: str
    device_sm: int | None
    sdpa_torch_runtime: str
    requested_role_policies: tuple[tuple[str, AttentionPolicy], ...] = ()


def validate_attention_policy(policy: object) -> AttentionPolicy:
    if not isinstance(policy, str) or policy not in ATTENTION_POLICIES:
        raise ValueError("unsupported attention policy")
    return policy


def attention_policy_config_to_wire(config: AttentionPolicyConfig) -> dict[str, object]:
    if not isinstance(cast("object", config), AttentionPolicyConfig):
        raise TypeError("config must be AttentionPolicyConfig")
    return {
        "requestedPolicy": config.requested_policy,
        "requestedRolePolicies": [
            [role, policy] for role, policy in config.requested_role_policies
        ],
    }


def attention_policy_config_from_wire(raw: object) -> AttentionPolicyConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("attention policy config must be an object")
    wire = cast("Mapping[object, object]", raw)
    if set(wire) != {"requestedPolicy", "requestedRolePolicies"}:
        raise ValueError("attention policy config has missing or extra fields")
    requested_policy = validate_attention_policy(wire["requestedPolicy"])
    role_policies_raw = wire["requestedRolePolicies"]
    if not isinstance(role_policies_raw, Sequence) or isinstance(role_policies_raw, (str, bytes)):
        raise ValueError("requested role policies must be an array")
    parsed: list[tuple[str, AttentionPolicy]] = []
    seen: set[str] = set()
    for pair in cast("Sequence[object]", role_policies_raw):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(cast("Sequence[object]", pair)) != 2
        ):
            raise ValueError("requested role policy must be a pair")
        pair_values = cast("Sequence[object]", pair)
        role = pair_values[0]
        if not isinstance(role, str) or role not in ATTENTION_ROLES:
            raise ValueError("unknown attention role in requested role policies")
        if role in seen:
            raise ValueError("requested role policies must name each role at most once")
        seen.add(role)
        parsed.append((role, validate_attention_policy(pair_values[1])))
    by_role = dict(parsed)
    canonical: tuple[tuple[str, AttentionPolicy], ...] = tuple(
        (role, by_role[role]) for role in ATTENTION_ROLES if role in by_role
    )
    _validate_requested_role_policies(requested_policy, canonical)
    return AttentionPolicyConfig(requested_policy, canonical)


def resolve_role_policy(
    requested_policy: AttentionPolicy,
    requested_role_policies: Sequence[tuple[str, AttentionPolicy]],
    role: str,
) -> AttentionPolicy:
    """Return a role override when present, otherwise the token-wide policy."""
    if role not in ATTENTION_ROLES:
        raise ValueError("unknown attention role")
    for override_role, policy in requested_role_policies:
        if override_role == role:
            return validate_attention_policy(policy)
    return validate_attention_policy(requested_policy)


def derive_attention_route_token(
    evidence: AttentionCapabilityEvidence,
    config: AttentionPolicyConfig,
) -> AttentionRouteToken:
    """Derive the unique route token for capability evidence and job policy.

    The observable derivation is part of the route-token version contract.
    Changing it for the same inputs requires a new route-token version.
    """
    if not isinstance(cast("object", evidence), AttentionCapabilityEvidence):
        raise TypeError("evidence must be AttentionCapabilityEvidence")
    if not isinstance(cast("object", config), AttentionPolicyConfig):
        raise TypeError("config must be AttentionPolicyConfig")

    routes: list[AttentionRoute] = []
    portable_fallback = False
    for role in ATTENTION_ROLES:
        requested = resolve_role_policy(
            config.requested_policy,
            config.requested_role_policies,
            role,
        )
        if requested == "auto":
            route = automatic_attention_route(
                role, evidence.available_policies, evidence.device_kind
            )
        else:
            effective = requested
            route = AttentionRoute(
                role,
                effective,
                "sdpa" if effective in ("comfy_kitchen_int8", "sage", "sol") else None,
            )
            if effective not in evidence.available_policies:
                route = AttentionRoute(role, "sdpa")
                portable_fallback = True
        routes.append(route)

    kitchen_requested = any(route.primary in ("sol", "comfy_kitchen_int8") for route in routes)
    sage_requested = any(route.primary == "sage" for route in routes)
    providers = tuple(
        pair
        for pair in evidence.provider_versions
        if (pair[0] != "comfy-kitchen" or kitchen_requested)
        and (pair[0] != "sageattention" or sage_requested)
    )
    return AttentionRouteToken(
        version=(
            4
            if config.requested_policy == "auto"
            else 3
            if portable_fallback
            else 2
            if config.requested_role_policies
            else ATTENTION_ROUTE_TOKEN_VERSION
        ),
        routes=tuple(routes),
        provider_versions=providers,
        adapter_contract_revision=evidence.adapter_contract_revision,
        device_kind=evidence.device_kind,
        device_sm=evidence.device_sm,
        sdpa_torch_runtime=evidence.sdpa_torch_runtime,
        requested_policy=config.requested_policy,
        requested_role_policies=config.requested_role_policies,
    )


def resolve_attention_runtime_status(
    policy: AttentionPolicy, token: AttentionRouteToken | None
) -> AttentionRuntimeStatus:
    policy = validate_attention_policy(policy)
    if token is None:
        if policy != "auto":
            raise ValueError("a named attention policy requires an authenticated route token")
        routes = tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES)
        return AttentionRuntimeStatus(
            False, policy, routes, (), "unavailable", "unknown", None, "unknown"
        )
    if not isinstance(cast("object", token), AttentionRouteToken):
        raise TypeError("attention_route_token must be AttentionRouteToken or None")
    if token.requested_policy != policy:
        raise ValueError("attention policy does not match route token")
    return AttentionRuntimeStatus(
        True,
        policy,
        token.routes,
        token.provider_versions,
        token.adapter_contract_revision,
        token.device_kind,
        token.device_sm,
        token.sdpa_torch_runtime,
        requested_role_policies=token.requested_role_policies,
    )


def attention_capability_evidence_to_wire(
    evidence: AttentionCapabilityEvidence,
) -> dict[str, object]:
    if not isinstance(cast("object", evidence), AttentionCapabilityEvidence):
        raise TypeError("evidence must be AttentionCapabilityEvidence")
    return {
        "version": evidence.version,
        "availablePolicies": list(evidence.available_policies),
        "providerVersions": [[name, version] for name, version in evidence.provider_versions],
        "adapterContractRevision": evidence.adapter_contract_revision,
        "deviceKind": evidence.device_kind,
        "deviceSm": evidence.device_sm,
        "sdpaTorchRuntime": evidence.sdpa_torch_runtime,
    }


def attention_capability_evidence_from_wire(raw: object) -> AttentionCapabilityEvidence:
    if not isinstance(raw, Mapping):
        raise ValueError("attention capability evidence must be an object")
    wire = cast("Mapping[object, object]", raw)
    expected = {
        "version",
        "availablePolicies",
        "providerVersions",
        "adapterContractRevision",
        "deviceKind",
        "deviceSm",
        "sdpaTorchRuntime",
    }
    if set(wire) != expected:
        raise ValueError("attention capability evidence has missing or extra fields")
    version = wire["version"]
    if type(version) is not int or version != ATTENTION_CAPABILITY_EVIDENCE_VERSION:
        raise ValueError("unsupported attention capability evidence version")
    policies_raw = wire["availablePolicies"]
    providers_raw = wire["providerVersions"]
    if not isinstance(policies_raw, Sequence) or isinstance(policies_raw, (str, bytes)):
        raise ValueError("available attention policies must be an array")
    if not isinstance(providers_raw, Sequence) or isinstance(providers_raw, (str, bytes)):
        raise ValueError("provider versions must be an array")
    providers: list[tuple[str, str]] = []
    for pair in cast("Sequence[object]", providers_raw):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(cast("Sequence[object]", pair)) != 2
        ):
            raise ValueError("provider version must be a pair")
        pair_values = cast("Sequence[object]", pair)
        providers.append((cast("str", pair_values[0]), cast("str", pair_values[1])))
    return AttentionCapabilityEvidence(
        version=version,
        available_policies=tuple(
            validate_attention_policy(policy) for policy in cast("Sequence[object]", policies_raw)
        ),
        provider_versions=tuple(providers),
        adapter_contract_revision=cast("str", wire["adapterContractRevision"]),
        device_kind=cast("str", wire["deviceKind"]),
        device_sm=cast("int | None", wire["deviceSm"]),
        sdpa_torch_runtime=cast("str", wire["sdpaTorchRuntime"]),
    )


def attention_route_token_to_wire(token: AttentionRouteToken) -> dict[str, object]:
    if not isinstance(cast("object", token), AttentionRouteToken):
        raise TypeError("token must be AttentionRouteToken")
    wire: dict[str, object] = {
        "version": token.version,
        "routes": [
            {"role": route.role, "primary": route.primary, "fallback": route.fallback}
            for route in token.routes
        ],
        "providerVersions": [[name, version] for name, version in token.provider_versions],
        "adapterContractRevision": token.adapter_contract_revision,
        "deviceKind": token.device_kind,
        "deviceSm": token.device_sm,
        "sdpaTorchRuntime": token.sdpa_torch_runtime,
        "requestedPolicy": token.requested_policy,
    }
    if token.version >= 2:
        wire["requestedRolePolicies"] = [
            [role, policy] for role, policy in token.requested_role_policies
        ]
    return wire


def canonical_attention_route_token_bytes(token: AttentionRouteToken) -> bytes:
    """Encode a route token for exact process-boundary comparison."""
    return json.dumps(
        attention_route_token_to_wire(token),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def attention_route_token_from_wire(raw: object) -> AttentionRouteToken:
    if not isinstance(raw, Mapping):
        raise ValueError("attention route token must be an object")
    wire = cast("Mapping[object, object]", raw)
    version_raw = wire.get("version")
    if type(version_raw) is not int or version_raw not in ATTENTION_ROUTE_TOKEN_VERSIONS:
        raise ValueError("unsupported attention route token version")
    expected = {
        "version",
        "routes",
        "providerVersions",
        "adapterContractRevision",
        "deviceKind",
        "deviceSm",
        "sdpaTorchRuntime",
        "requestedPolicy",
    }
    if version_raw >= 2:
        expected.add("requestedRolePolicies")
    if set(wire) != expected:
        raise ValueError("attention route token has missing or extra fields")
    routes_raw = wire["routes"]
    providers_raw = wire["providerVersions"]
    if not isinstance(routes_raw, Sequence) or isinstance(routes_raw, (str, bytes)):
        raise ValueError("attention routes must be an array")
    routes: list[AttentionRoute] = []
    for item in cast("Sequence[object]", routes_raw):
        if not isinstance(item, Mapping) or set(cast("Mapping[object, object]", item)) != {
            "role",
            "primary",
            "fallback",
        }:
            raise ValueError("attention route has malformed fields")
        routes.append(
            AttentionRoute(
                cast("str", item["role"]),
                cast("str", item["primary"]),
                cast("str | None", item["fallback"]),
            )
        )
    if not isinstance(providers_raw, Sequence) or isinstance(providers_raw, (str, bytes)):
        raise ValueError("provider versions must be an array")
    providers: list[tuple[str, str]] = []
    for pair in cast("Sequence[object]", providers_raw):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(cast("Sequence[object]", pair)) != 2
        ):
            raise ValueError("provider version must be a pair")
        pair_values = cast("Sequence[object]", pair)
        providers.append((cast("str", pair_values[0]), cast("str", pair_values[1])))
    role_policies: list[tuple[str, AttentionPolicy]] = []
    if version_raw >= 2:
        role_policies_raw = wire["requestedRolePolicies"]
        if not isinstance(role_policies_raw, Sequence) or isinstance(
            role_policies_raw, (str, bytes)
        ):
            raise ValueError("requested role policies must be an array")
        for pair in cast("Sequence[object]", role_policies_raw):
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(cast("Sequence[object]", pair)) != 2
            ):
                raise ValueError("requested role policy must be a pair")
            pair_values = cast("Sequence[object]", pair)
            role_policies.append(
                (cast("str", pair_values[0]), validate_attention_policy(pair_values[1]))
            )
    return AttentionRouteToken(
        version_raw,
        tuple(routes),
        tuple(providers),
        cast("str", wire["adapterContractRevision"]),
        cast("str", wire["deviceKind"]),
        cast("int | None", wire["deviceSm"]),
        cast("str", wire["sdpaTorchRuntime"]),
        validate_attention_policy(wire["requestedPolicy"]),
        tuple(role_policies),
    )
