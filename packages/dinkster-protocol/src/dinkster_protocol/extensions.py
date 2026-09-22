"""RPC-clean declarative contracts for Dinkster extensions.

This module belongs in ``dinkster-protocol`` because activation will cross the
same worker boundary as invocations: workers, engines, and future remote hosts
must share these frozen data shapes without importing one another. Pack
authors receive the public subset through ``dinkster_api.v1``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, TypeGuard, cast

from .frontend_modules import FrontendModule, validate_frontend_modules
from .pack_surfaces import (
    PACK_EVENTS_SURFACE,
    PACK_ROUTES_SURFACE,
    PackEvent,
    PackRoute,
    pack_surfaces_to_wire,
    validate_pack_surfaces,
)


class ExtensionScope(StrEnum):
    """An independently authorized extension process/surface scope."""

    SCHEMA = "schema"
    SERVER = "server"
    INFERENCE = "inference"
    FRONTEND = "frontend"
    TRAINING = "training"


EXTENSION_SCOPES = tuple(scope.value for scope in ExtensionScope)
"""Closed manifest vocabulary for extension privilege declarations."""


EXTENSION_CAPABILITIES = (
    "accelerator",
    "artifacts",
    "background-jobs",
    "downloads",
    "filesystem",
    "model-family-registration",
    "routes",
)
"""Closed capability vocabulary from extension-design.md section 5.

``accelerator`` (GPU/compute device use for long-running work) and
``artifacts`` (checkpoint/adapter artifact read/write) are the training
authorities from training-design.md section 7.

These names are authorization and audit metadata only. S0-A validates and
carries declarations; enforcement begins with the first server-scope slice.
"""


def _runtime_value(value: object) -> object:
    """Erase a static annotation so frozen dataclass inputs can be validated."""
    return value


@dataclass(frozen=True)
class ExtensionEntryPoints:
    """Scope-separated ``module:attr`` references, never resolved at parse time."""

    schema: str | None = None
    server: str | None = None
    inference: str | None = None
    frontend: str | None = None
    training: str | None = None

    def __post_init__(self) -> None:
        for scope in ExtensionScope:
            entry = _runtime_value(self.for_scope(scope))
            if entry is not None and (
                not isinstance(entry, str)
                or entry.count(":") != 1
                or not all(entry.split(":"))
                or any(char.isspace() for char in entry)
            ):
                raise ValueError(f"{scope.value} extension entry must be a 'module:attr' string")

    def for_scope(self, scope: ExtensionScope) -> str | None:
        return cast("str | None", getattr(self, scope.value))


@dataclass(frozen=True)
class ExtensionDeclaration:
    """One pack's declarative extension entries and requested authority."""

    entries: ExtensionEntryPoints = ExtensionEntryPoints()
    privileges: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    routes: tuple[PackRoute, ...] = ()
    events: tuple[PackEvent, ...] = ()
    frontend_modules: tuple[FrontendModule, ...] = ()

    def __post_init__(self) -> None:
        entries = _runtime_value(self.entries)
        if not isinstance(entries, ExtensionEntryPoints):
            raise TypeError("entries must be ExtensionEntryPoints")
        _require_declared_vocabulary("privileges", self.privileges, EXTENSION_SCOPES)
        _require_declared_vocabulary("capabilities", self.capabilities, EXTENSION_CAPABILITIES)
        validate_pack_surfaces(self.routes, self.events)
        if self.routes and ("server" not in self.privileges or "routes" not in self.capabilities):
            raise ValueError("pack routes require server privilege and routes capability")
        if self.events and "schema" not in self.privileges:
            raise ValueError("pack events require schema privilege")
        validate_frontend_modules(self.frontend_modules)
        if self.frontend_modules and "frontend" not in self.privileges:
            raise ValueError("frontend modules require frontend privilege")
        missing_privileges = tuple(
            scope
            for scope in EXTENSION_SCOPES
            if getattr(entries, scope) is not None and scope not in self.privileges
        )
        if missing_privileges:
            raise ValueError(
                f"extension entries require matching privileges: {', '.join(missing_privileges)}"
            )


class CompositionMode(StrEnum):
    """The complete composition vocabulary required by design principle P1."""

    ORDERED_LIST = "ordered_list"
    WRAPPER_CHAIN = "wrapper_chain"
    EXCLUSIVE = "exclusive"
    OBSERVERS = "observers"
    KEYED_REGISTRY = "keyed_registry"


class GuidancePhase(StrEnum):
    CONDITION_EVALUATION = "condition-evaluation"
    PRE_CFG = "pre-cfg"
    REDUCE = "reduce"
    POST_CFG = "post-cfg"


class GuidancePhaseParticipation(StrEnum):
    COMPOSE = "compose"
    BYPASS_TRANSFORMS = "bypass-transforms"


GUIDANCE_SURFACES = (
    "inference.guidance.condition-evaluation",
    "inference.guidance.pre-cfg",
    "inference.guidance.strategy",
    "inference.guidance.post-cfg",
    "inference.guidance.plan-augmentation",
    "inference.guidance.attention",
)

ATTENTION_QKV_SURFACE = "inference.attention.qkv"
ATTENTION_WRAPPER_SURFACE = "inference.attention.wrapper"
ATTENTION_OUTPUT_SURFACE = "inference.attention.output"
ATTENTION_BACKEND_SURFACE = "inference.attention.backend"
BLOCK_INJECTION_SURFACE = "inference.block.injection"

ATTENTION_SURFACES = (
    ATTENTION_QKV_SURFACE,
    ATTENTION_WRAPPER_SURFACE,
    ATTENTION_OUTPUT_SURFACE,
    ATTENTION_BACKEND_SURFACE,
    BLOCK_INJECTION_SURFACE,
)

GRAPH_COMPILERS_SURFACE = "inference.graph-compilers"
GRAPH_COMPILE_REQUEST_TYPE = "compileGraph"
GRAPH_COMPILE_RESULT_TYPE = "compileGraphResult"
GRAPH_COMPILE_CANCEL_TYPE = "cancelCompile"

# These adjudicated limits are durable protocol facts. Changing any limit
# requires durable adjudication before implementation.
GRAPH_COMPILE_MAX_NODES = 4096
GRAPH_COMPILE_MAX_LINKS = 16384
GRAPH_COMPILE_MAX_PASSES = 32
GRAPH_COMPILE_MAX_DEPTH = 16
GRAPH_COMPILE_MAX_GENERATED_PER_PASS = 1024
GRAPH_COMPILE_MAX_REPLY_BYTES = 8 * 1024 * 1024
GRAPH_COMPILE_TIMEOUT_SECONDS = 30.0
GENERATED_NODE_ID_PREFIX = "$gen-"

GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION = "compile-unknown-generation"
GRAPH_COMPILE_ERROR_TIMEOUT = "compile-timeout"
GRAPH_COMPILE_ERROR_REPLY_OVERSIZE = "compile-reply-oversize"
GRAPH_COMPILE_ERROR_MALFORMED_REPLY = "compile-malformed-reply"
GRAPH_COMPILE_ERROR_NODE_LIMIT = "compile-node-limit"
GRAPH_COMPILE_ERROR_LINK_LIMIT = "compile-link-limit"
GRAPH_COMPILE_ERROR_PASS_LIMIT = "compile-pass-limit"
GRAPH_COMPILE_ERROR_DEPTH_LIMIT = "compile-depth-limit"
GRAPH_COMPILE_ERROR_GENERATED_LIMIT = "compile-generated-limit"
GRAPH_COMPILE_ERROR_TARGET_MISMATCH = "compile-target-mismatch"
GRAPH_COMPILE_ERROR_SELECTOR_EMITTED = "compile-selector-emitted"
GRAPH_COMPILE_ERROR_SELECTOR_INPUT = "compile-selector-input"
GRAPH_COMPILE_ERROR_ID_COLLISION = "compile-id-collision"
GRAPH_COMPILE_ERROR_ID_FORMAT = "compile-id-format"
GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE = "compile-origin-coverage"
GRAPH_COMPILE_ERROR_GENERATION_MISMATCH = "compile-generation-mismatch"
GRAPH_COMPILE_ERROR_COMPILER_FAILURE = "compile-compiler-failure"


@dataclass(frozen=True)
class ContributionSurfaceDescriptor:
    """Declares how one named contribution surface composes.

    ``ordered_list`` preserves host-resolved order; ``wrapper_chain`` delegates
    explicitly to the next wrapper; ``exclusive`` has one owner-selected
    implementation; ``observers`` fans out; and ``keyed_registry`` composes by
    unique key. S0-A publishes this descriptor without migrating any existing
    seam onto it.
    """

    surface_id: str
    mode: CompositionMode
    routes: tuple[PackRoute, ...] = ()
    events: tuple[PackEvent, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("surface_id", self.surface_id)
        mode = _runtime_value(self.mode)
        if not isinstance(mode, CompositionMode):
            raise TypeError("mode must be a CompositionMode")
        validate_pack_surfaces(self.routes, self.events)
        if self.routes and (
            self.surface_id != PACK_ROUTES_SURFACE or self.mode != CompositionMode.KEYED_REGISTRY
        ):
            raise ValueError("routes require the keyed server.pack-routes surface")
        if self.events and (
            self.surface_id != PACK_EVENTS_SURFACE or self.mode != CompositionMode.OBSERVERS
        ):
            raise ValueError("events require the observers schema.pack-events surface")


BehaviorValue: TypeAlias = str | int | bool | None


@dataclass(frozen=True)
class KeyedContribution:
    """One RPC-clean declaration on a keyed extension surface.

    The callable implementation remains in its execution process. Package
    identity on :class:`ActiveExtension` binds its code bytes; this value
    carries only the registered id, aliases, and behavior-affecting metadata.
    """

    surface_id: str
    id: str
    aliases: tuple[str, ...] = ()
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("surface_id", self.surface_id)
        _require_nonempty("id", self.id)
        _require_string_tuple("aliases", self.aliases)
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("aliases must be unique")
        _require_behavior_configuration("behavior_metadata", self.behavior_metadata)


@dataclass(frozen=True)
class SamplerRegistrySnapshot:
    """RPC-clean effective sampler registry pinned by an execution runtime."""

    samplers: tuple[KeyedContribution, ...] = ()

    def __post_init__(self) -> None:
        samplers = _runtime_value(self.samplers)
        if not isinstance(samplers, tuple) or not all(
            isinstance(sampler, KeyedContribution)
            for sampler in cast("tuple[object, ...]", samplers)
        ):
            raise TypeError("samplers must be a tuple of KeyedContribution values")
        ids = [sampler.id for sampler in self.samplers]
        if len(set(ids)) != len(ids):
            raise ValueError("sampler ids must be unique")


@dataclass(frozen=True)
class GuidanceRegistrySnapshot:
    """Canonical RPC-clean declarations for all guidance phases."""

    contributions: tuple[KeyedContribution, ...] = ()

    def __post_init__(self) -> None:
        values = _runtime_value(self.contributions)
        if not isinstance(values, tuple) or not all(
            isinstance(value, KeyedContribution) for value in cast("tuple[object, ...]", values)
        ):
            raise TypeError("contributions must be a tuple of KeyedContribution values")
        seen: set[str] = set()
        strategy_count = 0
        previous: tuple[int, int, str] | None = None
        for contribution in self.contributions:
            if contribution.surface_id not in GUIDANCE_SURFACES:
                raise ValueError(f"unknown guidance surface {contribution.surface_id!r}")
            if (
                "." not in contribution.id
                or contribution.id.startswith(".")
                or contribution.id.endswith(".")
            ):
                raise ValueError("guidance contribution ids must be namespace-qualified")
            if contribution.id in seen:
                raise ValueError("guidance contribution ids must be globally unique")
            seen.add(contribution.id)
            if contribution.aliases:
                raise ValueError("guidance contribution aliases must be empty")
            metadata = dict(contribution.behavior_metadata)
            if type(metadata.get("contractVersion")) is not int or metadata["contractVersion"] != 1:
                raise ValueError("guidance contractVersion must be 1")
            config = tuple(key for key in metadata if key.startswith("config."))
            reserved = set(metadata) - set(config)
            surface_index = GUIDANCE_SURFACES.index(contribution.surface_id)
            if surface_index == 2:
                strategy_count += 1
                if reserved != {"contractVersion", "participation", "requiresUncond"}:
                    raise ValueError("guidance strategy metadata is malformed")
                if metadata["participation"] not in tuple(
                    item.value for item in GuidancePhaseParticipation
                ):
                    raise ValueError("guidance strategy participation is invalid")
                order = 0
            else:
                if reserved != {"contractVersion", "order", "requiresUncond"}:
                    raise ValueError("ordered guidance metadata is malformed")
                order_value = metadata["order"]
                if type(order_value) is not int or not -(2**31) <= order_value < 2**31:
                    raise ValueError("guidance order must be a signed 32-bit integer")
                order = order_value
            if type(metadata["requiresUncond"]) is not bool:
                raise TypeError("guidance requiresUncond must be bool")
            key = (surface_index, order, contribution.id)
            if previous is not None and key < previous:
                raise ValueError("guidance contributions are not in canonical order")
            previous = key
        if strategy_count > 1:
            raise ValueError("guidance strategy is exclusive")


@dataclass(frozen=True)
class GraphCompilerRegistrySnapshot:
    """Canonical RPC-clean declarations for graph compiler contributions."""

    contributions: tuple[KeyedContribution, ...] = ()

    def __post_init__(self) -> None:
        values = _runtime_value(self.contributions)
        if not isinstance(values, tuple) or not all(
            isinstance(value, KeyedContribution) for value in cast("tuple[object, ...]", values)
        ):
            raise TypeError("contributions must be a tuple of KeyedContribution values")
        seen: set[str] = set()
        previous: tuple[int, str] | None = None
        for contribution in self.contributions:
            if contribution.surface_id != GRAPH_COMPILERS_SURFACE:
                raise ValueError(f"unknown graph compiler surface {contribution.surface_id!r}")
            if (
                "." not in contribution.id
                or contribution.id.startswith(".")
                or contribution.id.endswith(".")
            ):
                raise ValueError("graph compiler contribution ids must be namespace-qualified")
            if contribution.id in seen:
                raise ValueError("graph compiler contribution ids must be globally unique")
            seen.add(contribution.id)
            if contribution.aliases:
                raise ValueError("graph compiler contribution aliases must be empty")
            metadata = dict(contribution.behavior_metadata)
            config = tuple(key for key in metadata if key.startswith("config."))
            reserved = set(metadata) - set(config)
            if reserved != {"contractVersion", "order"}:
                raise ValueError("graph compiler metadata is malformed")
            if type(metadata["contractVersion"]) is not int or metadata["contractVersion"] != 1:
                raise ValueError("graph compiler contractVersion must be 1")
            order = metadata["order"]
            if type(order) is not int or not -(2**31) <= order < 2**31:
                raise ValueError("graph compiler order must be a signed 32-bit integer")
            key = (order, contribution.id)
            if previous is not None and key < previous:
                raise ValueError("graph compiler contributions are not in canonical order")
            previous = key


def generated_node_id(compiler_id: str, sources: Sequence[str], local_key: str) -> str:
    """Derive a grammar-safe id from canonical compiler facts.

    The hash input is UTF-8 JSON with sorted object keys, compact separators,
    ASCII escaping, and ``sources`` preserved in caller-supplied order. Its
    object keys are exactly ``compiler``, ``key``, and ``sources``. At least
    one source is required so every generated node remains attributable.
    """
    runtime_compiler_id = _runtime_value(compiler_id)
    if (
        not isinstance(runtime_compiler_id, str)
        or "." not in runtime_compiler_id
        or runtime_compiler_id.startswith(".")
        or runtime_compiler_id.endswith(".")
    ):
        raise ValueError("compiler_id must be namespace-qualified")
    _require_nonempty("local_key", local_key)
    runtime_sources = _runtime_value(sources)
    if isinstance(runtime_sources, str) or not isinstance(runtime_sources, Sequence):
        raise TypeError("sources must be a sequence of non-empty strings")
    normalized_sources = cast("Sequence[object]", runtime_sources)
    if not normalized_sources or not all(
        isinstance(source, str) and source for source in normalized_sources
    ):
        raise ValueError("sources must be a sequence of non-empty strings")
    document: dict[str, object] = {
        "compiler": runtime_compiler_id,
        "key": local_key,
        "sources": list(normalized_sources),
    }
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return GENERATED_NODE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:16]


def canonical_compile_reply_bytes(reply: Mapping[str, object]) -> bytes:
    """Canonicalize a compile reply without validating its engine-owned shape."""
    return json.dumps(
        reply,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ActiveExtension:
    """Behavior-affecting facts for one active extension.

    The tuple fields are deliberately RPC-clean: only strings, integers,
    booleans, ``None``, and tuples may appear. ``contribution_ids`` preserves
    resolved composition order because changing wrapper/transform order changes
    behavior. Selector resolutions, service-provider choices, capabilities,
    and configuration keys are canonicalized as sorted, unique-key tuples.

    Presentation metadata (display names, descriptions, icons, search terms)
    has no field here and is therefore explicitly excluded from behavior
    identity. Included fields are exactly the extension/package identity,
    resolved contribution identity and order, selector resolutions, service
    providers, declared capabilities, and behavior-affecting configuration.
    """

    id: str
    version: str
    package_digest: str
    contribution_ids: tuple[str, ...] = ()
    keyed_contributions: tuple[KeyedContribution, ...] = ()
    selector_resolutions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    service_providers: tuple[tuple[str, str], ...] = ()
    capabilities: tuple[str, ...] = ()
    behavior_configuration: tuple[tuple[str, BehaviorValue], ...] = ()
    routes: tuple[PackRoute, ...] = ()
    events: tuple[PackEvent, ...] = ()
    frontend_modules: tuple[FrontendModule, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("id", self.id)
        _require_nonempty("version", self.version)
        _require_nonempty("package_digest", self.package_digest)
        _require_string_tuple("contribution_ids", self.contribution_ids)
        if len(set(self.contribution_ids)) != len(self.contribution_ids):
            raise ValueError("contribution_ids must be unique")
        keyed = _runtime_value(self.keyed_contributions)
        if not isinstance(keyed, tuple) or not all(
            isinstance(contribution, KeyedContribution)
            for contribution in cast("tuple[object, ...]", keyed)
        ):
            raise TypeError("keyed_contributions must be a tuple of KeyedContribution values")
        keyed_ids = [
            (contribution.surface_id, contribution.id) for contribution in self.keyed_contributions
        ]
        if len(set(keyed_ids)) != len(keyed_ids):
            raise ValueError("keyed_contributions must have unique surface/id pairs")
        _require_selector_resolutions(self.selector_resolutions)
        _require_string_pairs("service_providers", self.service_providers)
        _require_canonical_string_tuple("capabilities", self.capabilities)
        unknown_capabilities = tuple(
            capability
            for capability in self.capabilities
            if capability not in EXTENSION_CAPABILITIES
        )
        if unknown_capabilities:
            raise ValueError(f"unknown capabilities: {', '.join(unknown_capabilities)}")
        _require_behavior_configuration("behavior_configuration", self.behavior_configuration)
        validate_pack_surfaces(self.routes, self.events)
        validate_frontend_modules(self.frontend_modules, pack=self.id)
        if any(not module.module_digest for module in self.frontend_modules):
            raise ValueError("active frontend modules require content digests")


@dataclass(frozen=True)
class ExtensionSnapshot:
    """An immutable, canonically ordered active-extension generation.

    Extensions are sorted by id and ids are unique. Runtime snapshot
    production and per-execution pinning intentionally begin in S0-B; this
    type and its canonical behavior identity are the S0-A contract.
    """

    extensions: tuple[ActiveExtension, ...] = ()
    frontend_api: str = "1.0.0"
    format_version: int = 1

    def __post_init__(self) -> None:
        extensions = _runtime_value(self.extensions)
        if not isinstance(extensions, tuple):
            raise TypeError("extensions must be a tuple")
        if not all(
            isinstance(extension, ActiveExtension)
            for extension in cast("tuple[object, ...]", extensions)
        ):
            raise TypeError("extensions must contain ActiveExtension values")
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("format_version must be 1")
        _require_nonempty("frontend_api", self.frontend_api)
        ids = [extension.id for extension in self.extensions]
        if ids != sorted(set(ids)):
            raise ValueError("extensions must be sorted by id with unique ids")


def canonical_extension_snapshot(snapshot: ExtensionSnapshot) -> str:
    """Serialize behavior facts with fixed fields and collection ordering.

    JSON object field order below is part of the v1 canonical format. Active
    extensions, capabilities, and configuration are constructor-validated as
    sorted and unique. Contribution order is retained rather than sorted
    because it is behavior-affecting composition order.
    """
    document = {
        "format": "dinkster.extension-snapshot",
        "version": snapshot.format_version,
        "frontendApi": snapshot.frontend_api,
        "extensions": [
            {
                "id": extension.id,
                "version": extension.version,
                "packageDigest": extension.package_digest,
                "contributionIds": list(extension.contribution_ids),
                **(
                    {
                        "keyedContributions": [
                            {
                                "surfaceId": contribution.surface_id,
                                "id": contribution.id,
                                "aliases": list(contribution.aliases),
                                "behaviorMetadata": [
                                    {"key": key, "value": value}
                                    for key, value in contribution.behavior_metadata
                                ],
                            }
                            for contribution in extension.keyed_contributions
                        ]
                    }
                    if extension.keyed_contributions
                    else {}
                ),
                "selectorResolutions": [
                    {"selector": selector, "points": list(points)}
                    for selector, points in extension.selector_resolutions
                ],
                "serviceProviders": [
                    {"service": service, "provider": provider}
                    for service, provider in extension.service_providers
                ],
                "capabilities": list(extension.capabilities),
                "behaviorConfiguration": [
                    {"key": key, "value": value} for key, value in extension.behavior_configuration
                ],
                **pack_surfaces_to_wire(extension.routes, extension.events),
                **(
                    {
                        "frontend": [
                            module.to_wire(extension.id) for module in extension.frontend_modules
                        ]
                    }
                    if extension.frontend_modules
                    else {}
                ),
            }
            for extension in snapshot.extensions
        ],
    }
    return json.dumps(document, ensure_ascii=True, separators=(",", ":"))


def extension_behavior_hash(snapshot: ExtensionSnapshot) -> str:
    """Return lowercase sha256 over the UTF-8 canonical serialization."""
    canonical = canonical_extension_snapshot(snapshot).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_extension_snapshot_digest(value: object) -> TypeGuard[str]:
    """Whether value is the opaque ``sha256:<lowercase hex>`` snapshot handle."""
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_string_tuple(name: str, values: object) -> None:
    runtime_values = _runtime_value(values)
    if not isinstance(runtime_values, tuple):
        raise TypeError(f"{name} must be a tuple of non-empty strings")
    if not all(
        isinstance(value, str) and value for value in cast("tuple[object, ...]", runtime_values)
    ):
        raise TypeError(f"{name} must be a tuple of non-empty strings")


def _require_canonical_string_tuple(name: str, values: object) -> None:
    _require_string_tuple(name, values)
    normalized = cast("tuple[str, ...]", values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_declared_vocabulary(name: str, values: object, known: tuple[str, ...]) -> None:
    _require_string_tuple(name, values)
    normalized = cast("tuple[str, ...]", values)
    unknown = tuple(value for value in normalized if value not in known)
    if unknown:
        raise ValueError(f"unknown {name}: {', '.join(unknown)}")
    canonical = tuple(value for value in known if value in normalized)
    if normalized != canonical:
        raise ValueError(f"{name} must use vocabulary order without duplicates")


def _require_behavior_configuration(name: str, configuration: object) -> None:
    runtime_configuration = _runtime_value(configuration)
    if not isinstance(runtime_configuration, tuple):
        raise TypeError(f"{name} must be a tuple")
    keys: list[str] = []
    for raw_setting in cast("tuple[object, ...]", runtime_configuration):
        if not isinstance(raw_setting, tuple):
            raise TypeError(f"{name} entries must be (key, value) tuples")
        setting_values = cast("tuple[object, ...]", raw_setting)
        if len(setting_values) != 2:
            raise TypeError(f"{name} entries must be (key, value) tuples")
        key, value = setting_values
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} key must be a non-empty string")
        if value is not None and type(value) not in (str, int, bool):
            raise TypeError(f"{name} values must be str, int, bool, or None")
        keys.append(key)
    if keys != sorted(set(keys)):
        raise ValueError(f"{name} keys must be sorted and unique")


def _require_string_pairs(name: str, values: object) -> None:
    runtime_values = _runtime_value(values)
    if not isinstance(runtime_values, tuple):
        raise TypeError(f"{name} must be a tuple")
    pairs: list[tuple[str, str]] = []
    for raw_pair in cast("tuple[object, ...]", runtime_values):
        if not isinstance(raw_pair, tuple):
            raise TypeError(f"{name} entries must be non-empty string pairs")
        pair_values = cast("tuple[object, ...]", raw_pair)
        if len(pair_values) != 2 or not all(
            isinstance(value, str) and value for value in pair_values
        ):
            raise TypeError(f"{name} entries must be non-empty string pairs")
        pairs.append(cast("tuple[str, str]", pair_values))
    if pairs != sorted(set(pairs)) or len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} must be sorted by unique key")


def _require_selector_resolutions(values: object) -> None:
    runtime_values = _runtime_value(values)
    if not isinstance(runtime_values, tuple):
        raise TypeError("selector_resolutions must be a tuple")
    selectors: list[str] = []
    for raw_resolution in cast("tuple[object, ...]", runtime_values):
        if not isinstance(raw_resolution, tuple):
            raise TypeError("selector_resolutions entries must be (selector, points) tuples")
        resolution_values = cast("tuple[object, ...]", raw_resolution)
        if len(resolution_values) != 2:
            raise TypeError("selector_resolutions entries must be (selector, points) tuples")
        selector, points = resolution_values
        if not isinstance(selector, str) or not selector:
            raise ValueError("selector resolution names must be non-empty strings")
        _require_canonical_string_tuple("selector resolution points", points)
        selectors.append(selector)
    if selectors != sorted(set(selectors)):
        raise ValueError("selector_resolutions must be sorted by unique selector")


__all__ = [
    "EXTENSION_CAPABILITIES",
    "EXTENSION_SCOPES",
    "GENERATED_NODE_ID_PREFIX",
    "GRAPH_COMPILERS_SURFACE",
    "GRAPH_COMPILE_CANCEL_TYPE",
    "GRAPH_COMPILE_ERROR_COMPILER_FAILURE",
    "GRAPH_COMPILE_ERROR_DEPTH_LIMIT",
    "GRAPH_COMPILE_ERROR_GENERATED_LIMIT",
    "GRAPH_COMPILE_ERROR_GENERATION_MISMATCH",
    "GRAPH_COMPILE_ERROR_ID_COLLISION",
    "GRAPH_COMPILE_ERROR_ID_FORMAT",
    "GRAPH_COMPILE_ERROR_LINK_LIMIT",
    "GRAPH_COMPILE_ERROR_MALFORMED_REPLY",
    "GRAPH_COMPILE_ERROR_NODE_LIMIT",
    "GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE",
    "GRAPH_COMPILE_ERROR_PASS_LIMIT",
    "GRAPH_COMPILE_ERROR_REPLY_OVERSIZE",
    "GRAPH_COMPILE_ERROR_SELECTOR_EMITTED",
    "GRAPH_COMPILE_ERROR_SELECTOR_INPUT",
    "GRAPH_COMPILE_ERROR_TARGET_MISMATCH",
    "GRAPH_COMPILE_ERROR_TIMEOUT",
    "GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION",
    "GRAPH_COMPILE_MAX_DEPTH",
    "GRAPH_COMPILE_MAX_GENERATED_PER_PASS",
    "GRAPH_COMPILE_MAX_LINKS",
    "GRAPH_COMPILE_MAX_NODES",
    "GRAPH_COMPILE_MAX_PASSES",
    "GRAPH_COMPILE_MAX_REPLY_BYTES",
    "GRAPH_COMPILE_REQUEST_TYPE",
    "GRAPH_COMPILE_RESULT_TYPE",
    "GRAPH_COMPILE_TIMEOUT_SECONDS",
    "ActiveExtension",
    "BehaviorValue",
    "CompositionMode",
    "ContributionSurfaceDescriptor",
    "ExtensionDeclaration",
    "ExtensionEntryPoints",
    "ExtensionScope",
    "ExtensionSnapshot",
    "GraphCompilerRegistrySnapshot",
    "KeyedContribution",
    "SamplerRegistrySnapshot",
    "canonical_compile_reply_bytes",
    "canonical_extension_snapshot",
    "extension_behavior_hash",
    "generated_node_id",
    "is_extension_snapshot_digest",
]
