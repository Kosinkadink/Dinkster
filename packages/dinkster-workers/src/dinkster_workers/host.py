"""Isolated worker host: the child-process side of IsolatedWorker.

    python -m dinkster_workers.host --endpoint ENDPOINT --manifest PATH \
        [--shm-threshold N] [--no-shm] [--aimdo-init] \
        [--aimdo-arm {auto,on,off}] [--reserve-vram BYTES] \
        [--vram-budget INDEX=BYTES ...]

Loads the pack named by the manifest inside this process, connects back to
the engine over the boundary endpoint (``unix:<path>`` or authenticated
``tcp:<host>:<port>`` - see transport.py), announces the pack's schemas in
the schema wire format (hazard H1: the one interface description, same
format the server API ships), then serves invocations until shutdown.

The node-facing shim is a plain InProcessWorker: envelope unwrapping,
family grouping, and output validation behave identically in and out of
process because they are literally the same code (hazards H9/H10).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias, cast

from dinkster_assets import (
    AssetError,
    DeclaredAsset,
    FetchAborted,
    install_declared_assets,
    resolver_from_env,
    use_declared_asset_pack,
)
from dinkster_memory import (
    DEFAULT_ACCELERATOR_HEADROOM_BYTES,
    AcceleratorMemoryPolicy,
    DetailedConsumer,
    FullReleasableConsumer,
    FullReleaseConsumer,
    MeasuredMemory,
    PressureSignal,
    ReleasableConsumer,
    ReleaseCandidate,
    Shedder,
)
from dinkster_protocol import (
    GRAPH_COMPILE_CANCEL_TYPE,
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_REQUEST_TYPE,
    GRAPH_COMPILE_RESULT_TYPE,
    WORKGROUP_CAPABILITY,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    BeginWorkGroup,
    CommitWorkGroup,
    CompatGateDiagnostic,
    CompositionMode,
    ContributionSurfaceDescriptor,
    ExtensionScope,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusWorker,
    NodeError,
    PrepareReplica,
    ReplicaRefused,
    RunWorkUnit,
    WorkGroupMessage,
    WorkGroupRefused,
    WorkUnitFailed,
    attention_capability_evidence_to_wire,
    attention_route_token_to_wire,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
)
from dinkster_protocol.pack_surfaces import (
    PACK_EVENTS_SURFACE,
    PACK_ROUTE_TIMEOUT,
    PACK_ROUTES_SURFACE,
    PackRoute,
    pack_surfaces_to_wire,
)
from dinkster_schema import (
    SCHEMA_WIRE_SERVE_VERSIONS,
    ComfyAliasRegistry,
    ComfyGroupRegistry,
    Node,
    NodeSchema,
    build_node_types,
    build_schemas,
    claim_covers,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    comfy_group_registry_problems,
    comfy_group_registry_to_wire,
    configure_logging_from_env,
    install_stream_capture,
    schema_signature,
    schema_to_wire,
    validate_name,
)
from dinkster_values import (
    InvalidRenditionRequest,
    RenditionUnavailable,
    TypeRegistry,
    default_decode,
    process_instance_token,
    register_core_types,
    stamp_resource_producer_arm,
    value_resource_ids,
)

from .aimdo_bootstrap import bootstrap_aimdo
from .blobs import BlobTransfer, attribute_moved_wire
from .boundary import (
    DEFAULT_SHM_THRESHOLD,
    BoundaryError,
    ValueCodec,
    ValueStore,
    decode_invocation,
    encode_result,
    read_frame,
    release_segment,
    write_frame,
)
from .execution import (
    ExecutionContext,
    SourceStagingProvider,
    SourceStagingSession,
    use_execution_context,
)
from .governed import ReservationPlanner
from .in_process import (
    InProcessWorker,
    attention_route_token_matches_capabilities,
    validated_choice_values,
)
from .manifest import (
    GenerationProvider,
    ManifestError,
    PackManifest,
    VisionProvider,
    add_pack_root_to_import_path,
    generation_provider_to_wire,
    load_manifest,
    resolve_entry,
    vision_provider_to_wire,
)
from .produced_assets import source_asset_files
from .relay import consumer_item_to_wire
from .resume import InvocationKey, ResumableConversation
from .staging import AssetStagingService, StageAsset
from .transport import connect_endpoint
from .workgroup_session import (
    WORKGROUP_FRAME_TYPE,
    WORKGROUP_HELLO_FIELD,
    WorkGroupCommandHandler,
    handle_workgroup_command,
    workgroup_command_from_frame,
    workgroup_frame,
)

log = logging.getLogger("dinkster.workers.host")
_ACCELERATOR_BUDGETS_ENV = "DINKSTER_ACCELERATOR_BUDGETS"
_ACCELERATOR_HEADROOM_ENV = "DINKSTER_ACCELERATOR_HEADROOM_BYTES"
_AIMDO_HEADROOM_TARGET_ENV = "DINKSTER_AIMDO_HEADROOM_TARGET"


def _legacy_checkpoint_conversion_capable() -> bool:
    """Advertise conversion only from a worker with the complete runtime."""
    try:
        if not all(
            importlib.util.find_spec(module) is not None
            for module in ("dinkster_compat_comfy", "torch", "safetensors.torch")
        ):
            return False
        legacy_sources = importlib.import_module("dinkster_compat_comfy.legacy_sources")
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")
        return (
            callable(getattr(legacy_sources, "resolve_weight_source", None))
            and callable(getattr(legacy_sources, "classify_conversion_error", None))
            and isinstance(getattr(legacy_sources, "LegacyCheckpointError", None), type)
            and callable(getattr(torch, "load", None))
            and isinstance(getattr(torch, "Tensor", None), type)
            and callable(getattr(safetensors_torch, "save_file", None))
        )
    except Exception:  # noqa: BLE001 - broken optional runtime omits capability
        return False


def _convert_legacy_checkpoint(path: Path, logical_name: str) -> tuple[str, str | None]:
    """Run the canonical compat converter and classify its stable result."""
    try:
        legacy_sources = importlib.import_module("dinkster_compat_comfy.legacy_sources")
    except Exception as exc:  # noqa: BLE001 - unavailable worker runtime is retryable
        return "error", f"{type(exc).__name__}: {exc}"
    try:
        legacy_sources.resolve_weight_source(path, logical_name)
    except Exception as exc:  # noqa: BLE001 - converter owns its refusal type
        status = (
            "refused" if legacy_sources.classify_conversion_error(exc) == "refused" else "error"
        )
        return status, f"{type(exc).__name__}: {exc}"
    return "success", None


_accelerator_headroom_base: int | None = None
_aimdo_bootstrap_headroom_base: int | None = None
_aimdo_headroom_extra_bytes = 0
_MISSING = object()

PackLoad: TypeAlias = tuple[
    InProcessWorker, TypeRegistry, list[type[Node]], dict[str, InProcessWorker]
]


class AttentionRouteDiscoveryError(RuntimeError):
    """The installed inference runtime could not produce valid route evidence."""


def _discover_attention_routing(
    import_module: Callable[[str], object] = importlib.import_module,
) -> tuple[AttentionCapabilityEvidence | None, AttentionRouteToken | None]:
    """Probe capability evidence and its automatic compatibility route together."""
    try:
        module = import_module("dinkster_inference_torch")
    except ModuleNotFoundError as exc:
        if exc.name == "dinkster_inference_torch":
            return None, None
        raise AttentionRouteDiscoveryError("attention runtime nested import failed") from exc
    except Exception as exc:
        raise AttentionRouteDiscoveryError("attention runtime import failed") from exc
    capability_probe = getattr(module, "discover_attention_capabilities", None)
    route_probe = getattr(module, "discover_attention_route_token", None)
    configure_amd = getattr(module, "configure_amd_miopen", None)
    if not callable(capability_probe):
        raise AttentionRouteDiscoveryError("attention runtime has no capability discovery export")
    if not callable(route_probe):
        raise AttentionRouteDiscoveryError("attention runtime has no route discovery export")
    try:
        if callable(configure_amd):
            configure_amd()
        capabilities = capability_probe()
        token = route_probe("auto")
    except Exception as exc:
        raise AttentionRouteDiscoveryError("attention routing discovery failed") from exc
    if not isinstance(capabilities, AttentionCapabilityEvidence):
        raise AttentionRouteDiscoveryError(
            "attention capability discovery returned malformed evidence"
        )
    if not isinstance(token, AttentionRouteToken):
        raise AttentionRouteDiscoveryError("attention route discovery returned malformed token")
    try:
        derived = derive_attention_route_token(capabilities, AttentionPolicyConfig())
    except (TypeError, ValueError) as exc:
        raise AttentionRouteDiscoveryError(
            "attention route token cannot be derived from discovered capabilities"
        ) from exc

    if canonical_attention_route_token_bytes(token) != canonical_attention_route_token_bytes(
        derived
    ):
        raise AttentionRouteDiscoveryError(
            "attention route discovery does not match discovered capabilities"
        )
    return capabilities, token


def discover_attention_route_token(
    import_module: Callable[[str], object] = importlib.import_module,
) -> AttentionRouteToken | None:
    """Return the automatic compatibility route after capability cross-checking."""
    return _discover_attention_routing(import_module)[1]


def _bootstrap_aimdo(enabled: bool, *, simple_vram_headroom: int | None = None) -> bool:
    """Bootstrap Aimdo and retain the headroom state used by worker controls."""
    global _aimdo_bootstrap_headroom_base, _aimdo_headroom_extra_bytes
    _aimdo_headroom_extra_bytes = 0
    succeeded, _aimdo_bootstrap_headroom_base = bootstrap_aimdo(
        enabled,
        simple_vram_headroom=simple_vram_headroom,
    )
    return succeeded


def _prepare_accelerator_runtime(
    enabled: bool,
    import_module: Callable[[str], object] = importlib.import_module,
) -> bool:
    """Resolve accelerator kernels at worker startup instead of first execution."""
    if not enabled:
        return False
    try:
        runtime = import_module("dinkster_inference_torch")
        prepare = cast(Any, runtime).prepare_fp8_matmul_runtime
        if not callable(prepare):
            raise TypeError("prepare_fp8_matmul_runtime is not callable")
        prepare()
    except Exception:  # noqa: BLE001 - optional accelerator runtime remains best-effort
        log.warning("accelerator runtime preparation failed; worker continues", exc_info=True)
        return False
    return True


def _handle_aimdo_headroom(extra_bytes: object, base_bytes: object = _MISSING) -> None:
    """Apply one physical-headroom update and AIMDO reservation projection.

    ``extra_bytes`` deliberately includes every granted reservation on the
    mapped device, including this worker's own grant. That safe overestimate
    protects the granted-but-not-yet-allocated window. ``base_bytes`` replaces
    the process-global bootstrap base when present; omitting it retains the
    current base for compatibility with older parent frames.
    """
    global _accelerator_headroom_base, _aimdo_bootstrap_headroom_base
    global _aimdo_headroom_extra_bytes
    base = _accelerator_headroom_base
    if base is None:
        return
    if isinstance(extra_bytes, bool) or not isinstance(extra_bytes, int):
        log.warning("ignored malformed aimdoHeadroom extraBytes %r", extra_bytes)
        return
    if extra_bytes < 0:
        log.warning("ignored negative aimdoHeadroom extraBytes %d", extra_bytes)
        return
    if base_bytes is not _MISSING:
        if isinstance(base_bytes, bool) or not isinstance(base_bytes, int):
            log.warning("ignored malformed aimdoHeadroom baseBytes %r", base_bytes)
            return
        if base_bytes < 0:
            log.warning("ignored negative aimdoHeadroom baseBytes %d", base_bytes)
            return
        base = base_bytes
    _accelerator_headroom_base = base
    _aimdo_headroom_extra_bytes = extra_bytes
    os.environ[_ACCELERATOR_HEADROOM_ENV] = str(base)
    if _aimdo_bootstrap_headroom_base is None:
        return
    _aimdo_bootstrap_headroom_base = AcceleratorMemoryPolicy(
        physical_headroom_bytes=base
    ).minimum_free_bytes
    target = _aimdo_bootstrap_headroom_base + _aimdo_headroom_extra_bytes
    os.environ[_AIMDO_HEADROOM_TARGET_ENV] = str(target)
    try:
        inference_torch = importlib.import_module("dinkster_inference_torch")
        applied = inference_torch.set_simple_vram_headroom(target)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - runtime headroom is best-effort control
        log.warning("aimdo runtime headroom setter raised", exc_info=True)
        return
    if applied is True:
        os.environ.pop(_AIMDO_HEADROOM_TARGET_ENV, None)


def _parse_vram_budget(entry: str) -> tuple[int, int]:
    index_text, sep, bytes_text = entry.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected INDEX=BYTES")
    try:
        index = int(index_text)
        nbytes = int(bytes_text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected integer INDEX=BYTES") from None
    if index < 0 or nbytes < 0:
        raise argparse.ArgumentTypeError("INDEX and BYTES must be non-negative")
    return index, nbytes


def parse_comfy_args(entry: str) -> tuple[str, ...]:
    try:
        value = json.loads(entry)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON array of strings: {exc}") from None
    if not isinstance(value, list):
        raise argparse.ArgumentTypeError("expected a JSON array of strings")
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        raise argparse.ArgumentTypeError("expected a JSON array of strings")
    return tuple(cast("list[str]", items))


def _node_classes(entry: str, value: object) -> list[type[Node]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"entry '{entry}' must be a sequence of Node classes")
    classes: list[type[Node]] = []
    for item in cast("Sequence[object]", value):
        if not (isinstance(item, type) and issubclass(item, Node)):
            raise ManifestError(f"entry '{entry}' contains a non-Node item: {item!r}")
        classes.append(item)
    return classes


def load_pack(
    manifest: PackManifest,
    *,
    pack_context: Callable[[], contextlib.AbstractContextManager[None]] | None = None,
    import_from_pack_root: bool = False,
    registry: TypeRegistry | None = None,
) -> tuple[
    InProcessWorker,
    TypeRegistry,
    list[type[Node]],
    dict[str, InProcessWorker],
]:
    has_native_arm = any(arm == "native" for arm, _node_types in manifest.arms)
    attention_capabilities, attention_route_token = (
        _discover_attention_routing() if has_native_arm else (None, None)
    )
    if has_native_arm and (attention_capabilities is None or attention_route_token is None):
        raise AttentionRouteDiscoveryError(
            "native arm requires authenticated attention route evidence"
        )
    # Declared assets install BEFORE any pack entry runs, so schema-time
    # and execute-time code both reach the pack's own [[pack.assets]]
    # through dinkster_api.v1.declared_asset. The active context keeps tables
    # isolated when several manifests share this process.
    if import_from_pack_root:
        root = str(manifest.root.resolve())
        entries = tuple(
            entry
            for entry in (
                manifest.nodes_entry,
                manifest.types_entry,
                manifest.arm_nodes_entry,
                manifest.choices_entry,
                manifest.consumers_entry,
            )
            if entry is not None
        )
        for entry in entries:
            module_name = entry.partition(":")[0]
            loaded = sys.modules.get(module_name)
            module_file = getattr(loaded, "__file__", None)
            if module_file is not None and not Path(module_file).resolve().is_relative_to(
                manifest.root.resolve()
            ):
                raise ManifestError(
                    f"in-process entry module {module_name!r} for pack {manifest.name!r} "
                    f"is already loaded from outside the pack root: {module_file}"
                )
        if root not in sys.path:
            # In-process packs have no editable venv installation. Retain the
            # root for lazy imports after composition; removal intentionally
            # leaves imported modules resident for the serving lifetime.
            sys.path.insert(0, root)

    install_declared_assets(manifest.name, manifest.assets, resolver_from_env())

    def resolve_pack_entry(entry: str) -> object:
        value = resolve_entry(entry)
        if not import_from_pack_root:
            return value
        module_name = entry.partition(":")[0]
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            manifest.root.resolve()
        ):
            raise ManifestError(
                f"in-process entry module {module_name!r} for pack "
                f"{manifest.name!r} did not resolve inside {manifest.root}"
            )
        return value

    context = pack_context or contextlib.nullcontext

    @contextlib.contextmanager
    def worker_context() -> Generator[None]:
        with use_declared_asset_pack(manifest.name), context():
            yield

    with use_declared_asset_pack(manifest.name), context():
        if registry is None:
            registry = TypeRegistry()
            register_core_types(registry)
        if manifest.types_entry is not None:
            register = resolve_pack_entry(manifest.types_entry)
            if not callable(register):
                raise ManifestError(f"entry '{manifest.types_entry}' is not callable")
            register(registry)
        node_classes = _node_classes(manifest.nodes_entry, resolve_pack_entry(manifest.nodes_entry))
        # The same startup enumeration as the isolated worker path: static
        # lists are validated now, lazy providers are held uninvoked and
        # served per fetch by the in-process worker itself.
        choices = load_choices(manifest, resolve=resolve_pack_entry)
        consumers = load_consumers(manifest, resolve=resolve_pack_entry)
        worker = InProcessWorker(
            build_node_types(node_classes),
            registry,
            pack_context=worker_context,
            attention_capabilities=attention_capabilities,
            attention_route_token=attention_route_token,
            combo_choices=choices.static,
            lazy_choices=choices.lazy,
            memory_consumers=consumers,
        )
        default_schemas = build_schemas(node_classes)
        if manifest.comfy_aliases is not None:
            for record in manifest.comfy_aliases.records:
                if not any(claim_covers(claim, record.carrier) for claim in manifest.namespaces):
                    raise ManifestError(
                        f"{manifest.path}: comfy alias record {record.id!r} carrier "
                        f"{record.carrier!r} is not owned by this pack"
                    )
            problems = comfy_alias_registry_problems(manifest.comfy_aliases, default_schemas)
            if problems:
                raise ManifestError(f"{manifest.path}: invalid comfy alias registry: {problems[0]}")
        if manifest.comfy_groups is not None:
            for record in manifest.comfy_groups.records:
                if not any(claim_covers(claim, record.carrier) for claim in manifest.namespaces):
                    raise ManifestError(
                        f"{manifest.path}: comfy group record {record.id!r} carrier "
                        f"{record.carrier!r} is not owned by this pack"
                    )
            problems = comfy_group_registry_problems(manifest.comfy_groups, default_schemas)
            if problems:
                raise ManifestError(f"{manifest.path}: invalid comfy group registry: {problems[0]}")
        arm_workers: dict[str, InProcessWorker] = {}
        declared_arms = dict(manifest.arms)
    if manifest.arm_nodes_entry is not None:
        with use_declared_asset_pack(manifest.name), context():
            arm_nodes_obj = resolve_pack_entry(manifest.arm_nodes_entry)
            if not isinstance(arm_nodes_obj, Mapping):
                raise ManifestError(
                    f"entry '{manifest.arm_nodes_entry}' must be a mapping of "
                    "arm name -> sequence of Node classes"
                )
            registered = cast("Mapping[object, object]", arm_nodes_obj)
            if set(registered) != set(declared_arms):
                raise ManifestError(
                    f"entry '{manifest.arm_nodes_entry}' arm keys must exactly match [pack.arms]"
                )
            for arm, declared_types in declared_arms.items():
                classes = _node_classes(manifest.arm_nodes_entry, registered[arm])
                node_types = build_node_types(classes)
                if set(node_types) != set(declared_types):
                    raise ManifestError(
                        f"entry '{manifest.arm_nodes_entry}' arm {arm!r} node types "
                        "must exactly match [pack.arms]"
                    )
                for node_type, cls in node_types.items():
                    default_schema = default_schemas.get(node_type)
                    if default_schema is None:
                        raise ManifestError(
                            f"arm {arm!r} node type {node_type!r} is absent from "
                            f"the default nodes entry '{manifest.nodes_entry}'"
                        )
                    arm_signature = schema_signature(cls.schema())
                    if arm_signature != schema_signature(default_schema):
                        raise ManifestError(
                            f"arm {arm!r} node type {node_type!r} does not match "
                            "the default body's schema signature"
                        )
                arm_workers[arm] = InProcessWorker(
                    node_types,
                    registry,
                    pack_context=worker_context,
                    attention_capabilities=attention_capabilities,
                    attention_route_token=attention_route_token,
                )
    return worker, registry, node_classes, arm_workers


def load_planner(manifest: PackManifest) -> ReservationPlanner | None:
    if manifest.reservations_entry is None:
        return None
    planner = resolve_entry(manifest.reservations_entry)
    if not callable(planner):
        raise ManifestError(f"entry '{manifest.reservations_entry}' is not callable")
    return cast("ReservationPlanner", planner)


def load_workgroup_handler(manifest: PackManifest) -> WorkGroupCommandHandler | None:
    if manifest.workgroup_handler_entry is None:
        return None
    factory = resolve_entry(manifest.workgroup_handler_entry)
    if not callable(factory):
        raise ManifestError(f"entry '{manifest.workgroup_handler_entry}' is not callable")
    handler = factory()
    if not callable(handler):
        raise ManifestError(
            f"entry '{manifest.workgroup_handler_entry}' must return a callable handler"
        )
    return cast("WorkGroupCommandHandler", handler)


def load_consumers(
    manifest: PackManifest,
    *,
    resolve: Callable[[str], object] = resolve_entry,
) -> dict[str, Shedder]:
    """The pack's governed memory consumers, keyed by name. Names travel in
    the hello handshake and become the governor-facing identities of the
    parent's relay proxies."""
    if manifest.consumers_entry is None:
        return {}
    factory = resolve(manifest.consumers_entry)
    if not callable(factory):
        raise ManifestError(f"entry '{manifest.consumers_entry}' is not callable")
    consumers_obj = factory()
    if not isinstance(consumers_obj, Mapping):
        raise ManifestError(
            f"entry '{manifest.consumers_entry}' must return a mapping of name -> Shedder"
        )
    consumers: dict[str, Shedder] = {}
    for name, consumer in cast("Mapping[object, object]", consumers_obj).items():
        if not isinstance(name, str) or not name:
            raise ManifestError(
                f"entry '{manifest.consumers_entry}' has a non-string consumer name"
            )
        if not isinstance(consumer, Shedder):
            raise ManifestError(
                f"entry '{manifest.consumers_entry}' consumer {name!r} is not a "
                "Shedder (needs footprint() and shed())"
            )
        consumers[name] = consumer
    return consumers


def load_telemetry(
    manifest: PackManifest,
) -> Callable[[], Mapping[str, MeasuredMemory]] | None:
    """The pack's measured-memory probe: a zero-arg callable returning
    ``Mapping[str, MeasuredMemory]`` in this process's device namespace.
    Resolution failures are loud (a miswired manifest), but the callable
    itself is invoked defensively at measure time - a probe that starts
    failing later must never take the worker down."""
    if manifest.telemetry_entry is None:
        return None
    probe = resolve_entry(manifest.telemetry_entry)
    if not callable(probe):
        raise ManifestError(f"entry '{manifest.telemetry_entry}' is not callable")
    return cast("Callable[[], Mapping[str, MeasuredMemory]]", probe)


def load_schema_reload(manifest: PackManifest) -> Callable[[], Awaitable[None]] | None:
    if manifest.schema_reload_entry is None:
        return None
    watcher = resolve_entry(manifest.schema_reload_entry)
    if not callable(watcher):
        raise ManifestError(f"entry '{manifest.schema_reload_entry}' is not callable")
    return cast("Callable[[], Awaitable[None]]", watcher)


def load_source_staging(manifest: PackManifest) -> SourceStagingProvider | None:
    if manifest.source_staging_entry is None:
        return None
    factory = resolve_entry(manifest.source_staging_entry)
    if not callable(factory):
        raise ManifestError(f"entry '{manifest.source_staging_entry}' is not callable")
    provider = factory()
    if not all(callable(getattr(provider, name, None)) for name in ("sweep", "open")):
        raise ManifestError(
            f"entry '{manifest.source_staging_entry}' must return a source staging provider"
        )
    return cast("SourceStagingProvider", provider)


class LoadedChoices(NamedTuple):
    """Combo choice lists split by evaluation time: static lists are
    enumerated at startup and ride the hello; lazy providers are announced
    by id only and invoked once per parent fetch, never at startup."""

    static: dict[str, tuple[str, ...]]
    lazy: dict[str, Callable[[], Sequence[str]]]


def load_choices(
    manifest: PackManifest, *, resolve: Callable[[str], object] | None = None
) -> LoadedChoices:
    """The pack's combo choice lists, keyed by namespaced choice-list id.

    Enumerated once, at worker startup, AFTER the pack's entries have
    imported - so import-time registrations (a legacy pack appending to
    ComfyUI's sampler list) are visible. A mapping value is either a
    sequence of strings (a static list, validated here and published in
    the hello) or a zero-arg callable (a lazy provider, announced by id
    only and invoked once per parent fetch - never here, so startup and
    discovery cannot touch devices or other live resources). Ids are
    grammar-validated here; whether they fall under the pack's namespace
    claims is composition's check, same as node types."""
    if manifest.choices_entry is None:
        return LoadedChoices({}, {})
    factory = (resolve or resolve_entry)(manifest.choices_entry)
    if not callable(factory):
        raise ManifestError(f"entry '{manifest.choices_entry}' is not callable")
    choices_obj = factory()
    if not isinstance(choices_obj, Mapping):
        raise ManifestError(
            f"entry '{manifest.choices_entry}' must return a mapping of "
            "choice-list id -> sequence of value strings or zero-arg provider"
        )
    static: dict[str, tuple[str, ...]] = {}
    lazy: dict[str, Callable[[], Sequence[str]]] = {}
    for choice_id, values in cast("Mapping[object, object]", choices_obj).items():
        if not isinstance(choice_id, str) or not choice_id:
            raise ManifestError(f"entry '{manifest.choices_entry}' has a non-string choice id")
        id_problem = validate_name(choice_id)
        if id_problem is not None:
            raise ManifestError(
                f"entry '{manifest.choices_entry}' choice id {choice_id!r} {id_problem}"
            )
        if callable(values):
            lazy[choice_id] = cast("Callable[[], Sequence[str]]", values)
            continue
        try:
            static[choice_id] = validated_choice_values(
                values,
                subject=f"entry '{manifest.choices_entry}' choice {choice_id!r}",
            )
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
    return LoadedChoices(static, lazy)


def load_skips(manifest: PackManifest) -> dict[str, CompatGateDiagnostic]:
    """The pack's classified compat skips, keyed by source node name.

    Enumerated once beside combo choices at worker startup. The entry is
    foreign pack code, so validate its claimed typed mapping shape
    before it crosses the boundary; empty mappings are legal."""
    if manifest.skips_entry is None:
        return {}
    factory = resolve_entry(manifest.skips_entry)
    if not callable(factory):
        raise ManifestError(f"entry '{manifest.skips_entry}' is not callable")
    skips_obj = factory()
    if not isinstance(skips_obj, Mapping):
        raise ManifestError(
            f"entry '{manifest.skips_entry}' must return a mapping of "
            "source node name -> CompatGateDiagnostic"
        )
    skips: dict[str, CompatGateDiagnostic] = {}
    for node_name, diagnostic in cast("Mapping[object, object]", skips_obj).items():
        if not isinstance(node_name, str) or not node_name:
            raise ManifestError(
                f"entry '{manifest.skips_entry}' has a non-string or empty node name"
            )
        if not isinstance(diagnostic, CompatGateDiagnostic):
            raise ManifestError(
                f"entry '{manifest.skips_entry}' skip {node_name!r} has a "
                "non-CompatGateDiagnostic value"
            )
        if diagnostic.source_node != node_name:
            raise ManifestError(
                f"entry '{manifest.skips_entry}' skip {node_name!r} source_node differs"
            )
        skips[node_name] = diagnostic
    return skips


def load_extension_contributions(
    manifest: PackManifest,
) -> tuple[tuple[ExtensionScope, ContributionSurfaceDescriptor], ...]:
    """Resolve declarative extension registration inside the pack worker.

    Each resolvable scope entry names a zero-argument registration function
    returning a sequence of ContributionSurfaceDescriptor values. Inference
    entries are deliberately skipped here: the sampling worker resolves them
    through its local catalog so sampler callables never enter the pack worker
    or parent. Training entries are likewise skipped: the dedicated training
    worker process resolves them so trainer code never loads into the pack
    worker. Only RPC-clean data crosses hello.
    """
    if not manifest.extension_declared:
        return ()
    contributions: list[tuple[ExtensionScope, ContributionSurfaceDescriptor]] = []
    declaration = manifest.extension
    if declaration.routes:
        for route in declaration.routes:
            if not callable(resolve_entry(route.handler)):
                raise ManifestError(f"route handler {route.handler!r} is not callable")
        contributions.append(
            (
                ExtensionScope.SERVER,
                ContributionSurfaceDescriptor(
                    PACK_ROUTES_SURFACE,
                    CompositionMode.KEYED_REGISTRY,
                    routes=declaration.routes,
                ),
            )
        )
    if declaration.events:
        contributions.append(
            (
                ExtensionScope.SCHEMA,
                ContributionSurfaceDescriptor(
                    PACK_EVENTS_SURFACE,
                    CompositionMode.OBSERVERS,
                    events=declaration.events,
                ),
            )
        )
    for scope in ExtensionScope:
        if scope is ExtensionScope.INFERENCE or scope is ExtensionScope.TRAINING:
            continue
        entry = manifest.extension.entries.for_scope(scope)
        if entry is None:
            continue
        register = resolve_entry(entry)
        if not callable(register):
            raise ManifestError(f"extension entry '{entry}' is not callable")
        declared = register()
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
            raise ManifestError(
                f"extension entry '{entry}' must return a sequence of "
                "ContributionSurfaceDescriptor values"
            )
        seen: set[str] = set()
        for item in cast("Sequence[object]", declared):
            if not isinstance(item, ContributionSurfaceDescriptor):
                raise ManifestError(
                    f"extension entry '{entry}' contains a non-descriptor item: {item!r}"
                )
            if item.surface_id in seen:
                raise ManifestError(
                    f"extension entry '{entry}' declares surface {item.surface_id!r} more than once"
                )
            seen.add(item.surface_id)
            contributions.append((scope, item))
    return tuple(contributions)


async def serve(endpoint: str, manifest_path: str, *, shm_threshold: int, use_shm: bool) -> None:
    manifest = load_manifest(Path(manifest_path))
    add_pack_root_to_import_path(manifest, sys.path)
    worker, registry, node_classes, arm_workers = load_pack(manifest)
    with use_declared_asset_pack(manifest.name):
        planner = load_planner(manifest)
        consumers = worker.memory_consumers
        telemetry = load_telemetry(manifest)
        schema_reload = load_schema_reload(manifest)
        source_staging = load_source_staging(manifest)
        choices = load_choices(manifest)
        skips = load_skips(manifest)
        extension_contributions = load_extension_contributions(manifest)
        workgroup_handler = load_workgroup_handler(manifest)
        schemas = build_schemas(node_classes)
    codec = ValueCodec(registry, shm_threshold=shm_threshold, use_shm=use_shm)
    reader, writer = await connect_endpoint(endpoint)
    with use_declared_asset_pack(manifest.name):
        await serve_connection(
            reader,
            writer,
            pack_name=manifest.name,
            worker=worker,
            schemas=schemas,
            comfy_aliases=manifest.comfy_aliases,
            comfy_groups=manifest.comfy_groups,
            planner=planner,
            consumers=consumers,
            telemetry=telemetry,
            schema_reload=schema_reload,
            source_staging=source_staging,
            choices=choices.static,
            lazy_choices=choices.lazy,
            skips=skips,
            extension_contributions=extension_contributions,
            arm_workers=arm_workers,
            body_arms=dict(manifest.arms),
            vision_providers=manifest.vision_providers,
            generation_providers=manifest.generation_providers,
            codec=codec,
            workgroup_handler=workgroup_handler,
        )


async def serve_many(
    endpoints: Sequence[str],
    manifest_paths: Sequence[str],
    *,
    shm_threshold: int,
    use_shm: bool,
) -> None:
    """Load every pack before opening any boundary, then serve all peers."""
    process_resource_tasks: set[asyncio.Task[None]] = set()
    process_maintenance_operations: set[tuple[object, str]] = set()
    loaded: list[tuple[Any, ...]] = []
    for path in manifest_paths:
        manifest = load_manifest(Path(path))
        add_pack_root_to_import_path(manifest, sys.path)
        pack_load = load_pack(manifest)
        with use_declared_asset_pack(manifest.name):
            loaded.append(
                (
                    manifest,
                    pack_load,
                    load_planner(manifest),
                    pack_load[0].memory_consumers,
                    load_telemetry(manifest),
                    load_schema_reload(manifest),
                    load_source_staging(manifest),
                    load_choices(manifest),
                    load_skips(manifest),
                    load_extension_contributions(manifest),
                    load_workgroup_handler(manifest),
                    build_schemas(pack_load[2]),
                )
            )

    async def one(endpoint: str, item: tuple[Any, ...]) -> None:
        (
            manifest,
            (worker, registry, _node_classes, arm_workers),
            planner,
            consumers,
            telemetry,
            schema_reload,
            source_staging,
            choices,
            skips,
            extensions,
            workgroup_handler,
            schemas,
        ) = item
        reader, writer = await connect_endpoint(endpoint)
        with use_declared_asset_pack(manifest.name):
            await serve_connection(
                reader,
                writer,
                pack_name=manifest.name,
                worker=worker,
                schemas=schemas,
                comfy_aliases=manifest.comfy_aliases,
                comfy_groups=manifest.comfy_groups,
                planner=planner,
                consumers=consumers,
                telemetry=telemetry,
                schema_reload=schema_reload,
                source_staging=source_staging,
                choices=choices.static,
                lazy_choices=choices.lazy,
                skips=skips,
                extension_contributions=extensions,
                arm_workers=arm_workers,
                body_arms=dict(manifest.arms),
                vision_providers=manifest.vision_providers,
                generation_providers=manifest.generation_providers,
                codec=ValueCodec(registry, shm_threshold=shm_threshold, use_shm=use_shm),
                workgroup_handler=workgroup_handler,
                process_resource_tasks=process_resource_tasks,
                process_maintenance_operations=process_maintenance_operations,
            )

    await asyncio.gather(
        *(one(endpoint, item) for endpoint, item in zip(endpoints, loaded, strict=True))
    )


def _retrieve_task_exception(task: asyncio.Task[object]) -> None:
    """Mark an abandoned task's exception as observed."""
    if not task.cancelled():
        task.exception()


async def serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    pack_name: str,
    worker: InProcessWorker,
    schemas: Mapping[str, NodeSchema],
    comfy_aliases: ComfyAliasRegistry | None = None,
    comfy_groups: ComfyGroupRegistry | None = None,
    planner: ReservationPlanner | None,
    consumers: Mapping[str, Shedder],
    codec: ValueCodec,
    telemetry: Callable[[], Mapping[str, MeasuredMemory]] | None = None,
    schema_reload: Callable[[], Awaitable[None]] | None = None,
    source_staging: SourceStagingProvider | None = None,
    choices: Mapping[str, Sequence[str]] | None = None,
    lazy_choices: Mapping[str, Callable[[], Sequence[str]]] | None = None,
    skips: Mapping[str, CompatGateDiagnostic] | None = None,
    extension_contributions: Sequence[tuple[ExtensionScope, ContributionSurfaceDescriptor]] = (),
    arm_workers: Mapping[str, InProcessWorker] | None = None,
    body_arms: Mapping[str, Sequence[str]] | None = None,
    vision_providers: Sequence[VisionProvider] | None = None,
    generation_providers: Sequence[GenerationProvider] | None = None,
    hello_extra: Mapping[str, object] | None = None,
    workgroup_handler: WorkGroupCommandHandler | None = None,
    asset_staging: AssetStagingService | None = None,
    declared_assets: Sequence[DeclaredAsset] = (),
    value_store: ValueStore | None = None,
    resume: ResumableConversation | None = None,
    process_resource_tasks: set[asyncio.Task[None]] | None = None,
    process_maintenance_operations: set[tuple[object, str]] | None = None,
) -> None:
    """Serve one boundary conversation on an already-established stream.

    This is the whole worker-side protocol: hello, invocations, leases, the
    memory relay, and the ram-release gate. ``serve`` (the launched child)
    and the remote service (service.py) both run it - the conversation is
    identical whether the peer is a parent process on this machine or an
    engine across the network. Pack state (``worker``, ``consumers``) may
    outlive one connection; everything conversation-scoped lives in the
    locals below and dies with the stream.
    """
    # Residency classes the parent accounts (sent as memoryDevices, already
    # in this process's namespace): footprints are evaluated on these plus
    # whatever detail items reveal, so a plain Shedder is still reported
    # honestly on every device the governor cares about.
    if source_staging is not None:
        source_staging.sweep()
    declared_devices: set[str] = set()
    send_lock = asyncio.Lock()
    sent_segments: dict[str, SharedMemory] = {}
    tasks: dict[str, asyncio.Task[None]] = {}
    cancellation_events: dict[str, threading.Event] = {}
    compile_tasks: dict[str, tuple[asyncio.Task[None], threading.Event]] = {}
    # In-flight stageAssets requests: requestId -> (task, abort event). The
    # event reaches into the fetch thread (checked between chunks), so a
    # cancelStage stops the download at the next chunk boundary and the
    # vault writer rolls the partial file back to nothing.
    stage_tasks: dict[str, tuple[asyncio.Task[None], threading.Event]] = {}
    # Memory leases in flight: requestId -> future resolving to None (granted)
    # or a denial message. requestId is the invocationId, so the parent can
    # correlate leases with cancellations without extra bookkeeping.
    pending_grants: dict[str, asyncio.Future[str | None]] = {}
    # Resource references named by results still in flight to the parent:
    # invocationId -> resource ids. Registered before the result frame is
    # sent, then cleared by the parent's validated resultAck or cancellation.
    # A resident named here is unreleasable - the parent-side gate cannot see
    # a stub it has not decoded yet, so this side must hold the resident until
    # the parent vouches for it (DESIGN 3.10: the in-flight half of the gate).
    inflight_result_refs: dict[str, frozenset[str]] = {}
    full_release_operations: set[str] = set()
    full_release_commits: set[str] = set()
    connection_token = object()
    resource_tasks: set[asyncio.Task[None]] = (
        process_resource_tasks if process_resource_tasks is not None else set()
    )
    maintenance_operations: set[tuple[object, str]] = (
        process_maintenance_operations if process_maintenance_operations is not None else set()
    )

    def track_resource_task(task: asyncio.Task[None]) -> asyncio.Task[None]:
        resource_tasks.add(task)
        task.add_done_callback(resource_tasks.discard)
        return task

    async def await_task_settlement(task: asyncio.Future[Any]) -> Any:
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                pass
        if cancelled:
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    async def run_sync_resource(
        call: Callable[..., object], /, *args: object, **kwargs: object
    ) -> Any:
        task = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
        return await await_task_settlement(task)

    def held_resource_ids() -> set[str]:
        held: set[str] = set()
        for refs in inflight_result_refs.values():
            held.update(refs)
        return held

    def finish_resumable_cancellation(invocation_id: str) -> None:
        inflight_result_refs.pop(invocation_id, None)
        assert resume is not None
        resume.finish_cancelled(invocation_id)

    async def send(
        header: dict[str, object],
        blobs: Sequence[bytes] = (),
        segments: Sequence[SharedMemory] = (),
    ) -> None:
        async with send_lock:
            for segment in segments:
                # Handle stays open until the parent acks: Windows frees a
                # segment when its last handle closes (see release_segment).
                sent_segments[segment.name] = segment
            if resume is not None:
                if segments:
                    raise BoundaryError("remote resumable transport cannot carry shared memory")
                await resume.send(header, blobs)
            else:
                await write_frame(writer, header, blobs)

    blob_transfer = (
        BlobTransfer(
            value_store,
            lambda header, blobs: send(header, blobs),
            closed_exc=lambda: ConnectionError("conversation ended mid blob transfer"),
        )
        if value_store is not None
        else None
    )
    result_transfer_lock = asyncio.Lock()

    async def ensure_peer_holds(
        pending: Sequence[tuple[str, bytes | Path]],
    ) -> dict[str, tuple[int, float]]:
        assert blob_transfer is not None
        if resume is None:
            return await blob_transfer.ensure_peer_holds(pending)
        while True:
            loss_event = resume.transport_loss_event()
            transfer = asyncio.create_task(blob_transfer.ensure_peer_holds(pending))
            lost = asyncio.create_task(loss_event.wait())
            try:
                done, _ = await asyncio.wait((transfer, lost), return_when=asyncio.FIRST_COMPLETED)
                if transfer in done:
                    try:
                        return transfer.result()
                    except ConnectionError:
                        if not loss_event.is_set():
                            raise
                transfer.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                    await transfer
                await resume.wait_ready()
            finally:
                if not transfer.done():
                    transfer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await transfer
                lost.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await lost

    def build_report() -> dict[str, object]:
        """Footprints (and items, for detail-contract consumers) for every
        consumer - the snapshot the parent's relay proxies serve. Keys stay
        in this process's namespace; the parent translates at receipt."""
        report: dict[str, object] = {}
        for name, consumer in consumers.items():
            devices = set(declared_devices)
            body: dict[str, object] = {}
            if isinstance(consumer, DetailedConsumer):
                items = tuple(consumer.details())
                for item in items:
                    devices.update(item.bytes_by_residency)
                body["items"] = [consumer_item_to_wire(item) for item in items]
            footprints: dict[str, int] = {}
            for device in devices:
                nbytes = consumer.footprint(device)
                if nbytes:
                    footprints[device] = nbytes
            body["footprints"] = footprints
            report[name] = body
        return report

    def measure() -> dict[str, dict[str, int]]:
        """The pack probe's snapshot in wire shape ({device: {freeBytes,
        totalBytes}}), keys still in this process's namespace. Defensive
        end to end: a raising probe measures nothing, and malformed
        entries are dropped item-by-item - bad telemetry is absent
        telemetry, never a dead worker."""
        if telemetry is None:
            return {}
        try:
            # The probe is foreign pack code; its annotation is a claim,
            # not a guarantee - treat the result as untyped data.
            snapshot = cast("object", telemetry())
        except Exception:  # noqa: BLE001 - a failing probe is an absent probe
            return {}
        if not isinstance(snapshot, Mapping):
            return {}
        measured: dict[str, dict[str, int]] = {}
        for device, memory in cast("Mapping[object, object]", snapshot).items():
            if not (isinstance(device, str) and device and isinstance(memory, MeasuredMemory)):
                continue
            # The dataclass came from foreign pack code, so validate its
            # runtime fields before carrying them across the boundary.
            wire = memory.to_wire()
            if wire is not None:
                measured[device] = wire
        return measured

    async def send_report() -> None:
        if not consumers and telemetry is None:
            return
        header: dict[str, object] = {
            "type": "memoryReport",
            "consumers": build_report(),
        }
        if telemetry is not None:
            header["measured"] = measure()
        with contextlib.suppress(Exception):
            await send(header)

    hello: dict[str, object] = {
        "type": "hello",
        "pack": pack_name,
        "schemas": {
            t: schema_to_wire(s, wire_version=max(SCHEMA_WIRE_SERVE_VERSIONS))
            for t, s in schemas.items()
        },
        "lazyStatus": True,
        # This process lifetime's identity: the same token resident codecs
        # stamp as RESOURCE_OWNER_META_KEY on the envelopes they produce, so
        # the parent can map an owner token back to a live session (and know
        # a token from an earlier lifetime is dead without a round trip).
        "workerInstance": process_instance_token(),
        "bodyArms": {arm: sorted(node_types) for arm, node_types in (body_arms or {}).items()},
        "renditions": [
            {
                "typeId": spec.type_id,
                "kind": spec.kind,
                "mime": spec.mime if isinstance(spec.mime, str) else "application/octet-stream",
                "default": spec.default,
                **({"version": spec.version} if spec.version is not None else {}),
                **({"parameters": list(spec.parameters)} if spec.parameters else {}),
                **({"defaults": dict(spec.defaults)} if spec.defaults is not None else {}),
                **({"limits": dict(spec.limits)} if spec.limits is not None else {}),
            }
            for spec in worker.registry.registered_renditions()
        ],
    }
    if comfy_aliases is not None:
        hello["comfyAliases"] = comfy_alias_registry_to_wire(comfy_aliases)
    if comfy_groups is not None:
        hello["comfyGroups"] = comfy_group_registry_to_wire(comfy_groups)
    if worker.attention_route_token is not None:
        hello["attentionRouteToken"] = attention_route_token_to_wire(worker.attention_route_token)
    if worker.attention_capabilities is not None:
        hello["attentionCapabilities"] = attention_capability_evidence_to_wire(
            worker.attention_capabilities
        )
    if _legacy_checkpoint_conversion_capable():
        hello["convertLegacyCheckpoint"] = True
    if choices:
        # Combo choice lists ride the handshake like schemas do: announced
        # once, validated by the parent, published behind server-owned
        # /api/choices/{id} routes. UI vocabulary, never identity.
        hello["comboChoices"] = {choice_id: list(values) for choice_id, values in choices.items()}
    if lazy_choices:
        # Lazy choice lists are announced by id only: the provider runs in
        # this process, once per parent fetch, never at startup - the hello
        # carries no values, just ownership of the ids.
        hello["lazyChoiceIds"] = sorted(lazy_choices)
    if skips:
        # Classified compat translation refusals ride the same startup
        # snapshot as schemas and choices. They are diagnostics, not
        # identity, and change only when this worker is restarted/reloaded.
        hello["compatSkips"] = {
            node_name: diagnostic.to_wire() for node_name, diagnostic in skips.items()
        }
    if extension_contributions:
        hello["extensionContributions"] = [
            {
                "scope": scope.value,
                "surfaceId": descriptor.surface_id,
                "mode": descriptor.mode.value,
                **pack_surfaces_to_wire(descriptor.routes, descriptor.events),
            }
            for scope, descriptor in extension_contributions
        ]
    if asset_staging is not None:
        # Announced whenever this build answers the staging frames. Whether
        # a fetch can actually land depends on the service's vault; a
        # vault-less daemon still answers assetQuery honestly and refuses
        # stageAssets with a per-digest reason.
        hello["assetStaging"] = True
    if declared_assets:
        # The pack's [[pack.assets]] declarations ride the handshake so the
        # engine - which never sees the daemon's manifest - can compute
        # which digests a dispatched node type requires and stage them
        # before the invoke crosses.
        hello["declaredAssets"] = [asset.descriptor() for asset in declared_assets]
    if vision_providers is not None:
        hello["visionProviders"] = [vision_provider_to_wire(item) for item in vision_providers]
    if generation_providers is not None:
        hello["generationProviders"] = [
            generation_provider_to_wire(item) for item in generation_providers
        ]
    if hello_extra and WORKGROUP_HELLO_FIELD in hello_extra:
        raise ValueError(f"{WORKGROUP_HELLO_FIELD} is owned by the workgroup handler")
    if hello_extra:
        hello.update(hello_extra)
    if workgroup_handler is not None:
        advertised_obj: object = getattr(
            workgroup_handler,
            "workgroup_capabilities",
            (WORKGROUP_CAPABILITY,),
        )
        if type(advertised_obj) is not tuple:
            raise ValueError("workgroup handler capabilities must be unique strings")
        advertised = cast("tuple[object, ...]", advertised_obj)
        if (
            WORKGROUP_CAPABILITY not in advertised
            or any(type(capability) is not str for capability in advertised)
            or len(advertised) != len(set(advertised))
        ):
            raise ValueError("workgroup handler capabilities must be unique strings")
        hello[WORKGROUP_HELLO_FIELD] = list(cast("tuple[str, ...]", advertised))
    hello["memoryConsumers"] = {
        name: {
            "details": isinstance(consumer, DetailedConsumer),
            "fullRelease": isinstance(consumer, (FullReleaseConsumer, FullReleasableConsumer)),
        }
        for name, consumer in consumers.items()
    }
    if telemetry is not None:
        # Measured device memory rides the handshake so the parent sees
        # ground truth before the first invocation; refreshed by every
        # memoryReport/memoryShedResult. Present (possibly empty) exactly
        # when the pack declares a telemetry entry.
        hello["measured"] = measure()
    if schema_reload is not None:
        hello["schemaReload"] = True
    await send(hello)
    await send_report()

    async def notify_schema_reload() -> None:
        assert schema_reload is not None
        while True:
            await schema_reload()
            await send({"type": "schemaReloadRequest"})

    schema_reload_task = (
        asyncio.create_task(notify_schema_reload()) if schema_reload is not None else None
    )

    def schema_reload_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error("pack %s schema reload watcher failed: %s", pack_name, error)

    if schema_reload_task is not None:
        schema_reload_task.add_done_callback(schema_reload_done)

    async def send_error(invocation_id: str, header: dict[str, Any], message: str) -> None:
        await send(
            {
                "type": "result",
                "invocationId": invocation_id,
                "executeMs": 0.0,
                "error": {
                    "nodeId": str(header.get("nodeId", "")),
                    "nodeType": str(header.get("nodeType", "")),
                    "message": message,
                    "traceback": "",
                },
            }
        )

    async def run_invocation(
        header: dict[str, Any], blobs: list[bytes], cancellation: threading.Event
    ) -> None:
        invocation_id = str(header["invocationId"])
        reserve_sent = False
        staging_session: SourceStagingSession | None = None
        staging_lock = threading.RLock()
        staging_closed = threading.Event()

        def materialize_source(asset: Any, kind: str, category: str) -> str:
            nonlocal staging_session
            if staging_closed.is_set() or cancellation.is_set():
                raise RuntimeError("source staging session is closing or closed")
            with staging_lock:
                if staging_closed.is_set() or cancellation.is_set():
                    raise RuntimeError("source staging session is closing or closed")
                if source_staging is None:
                    raise RuntimeError("this pack has no source staging provider")
                if staging_session is None:
                    staging_session = source_staging.open(invocation_id, invocation.media_sources)
                return staging_session.materialize(asset, kind, category)

        def close_staging() -> None:
            nonlocal staging_session
            staging_closed.set()
            with staging_lock:
                session, staging_session = staging_session, None
            if session is not None:
                session.close()

        # Node reports stream back as invocationEvent frames while execute()
        # runs. Each emit becomes its own send task: created in emit order and
        # serialized FIFO by send_lock, so per-invocation order holds; the
        # result send below first awaits them all, so events always reach the
        # parent before the result does (the ordering the reporting contract
        # promises). A failed event send is dropped - chatter must not fail
        # the node. flush_events() closes the window: late emits from a
        # leaked thread after the result is sent go nowhere.
        event_tasks: list[asyncio.Task[None]] = []
        event_window_open = True
        event_loop = asyncio.get_running_loop()

        async def send_event(event: InvocationEvent) -> None:
            frame: dict[str, object] = {
                "type": "invocationEvent",
                "invocationId": invocation_id,
                "name": event.name,
                "data": dict(event.data),
            }
            event_blobs = [event.blob] if event.blob is not None else []
            with contextlib.suppress(Exception):
                await send(frame, event_blobs)

        def on_event(event: InvocationEvent) -> None:
            if event_window_open:
                event_tasks.append(event_loop.create_task(send_event(event)))

        async def flush_events() -> None:
            nonlocal event_window_open
            event_window_open = False
            if event_tasks:
                await asyncio.gather(*event_tasks, return_exceptions=True)

        result_send_locked = False
        try:
            consumed: list[str] = []
            invocation = decode_invocation(codec, header, blobs, consumed)
            if consumed:
                await send({"type": "shmAck", "segments": consumed})
            # Memory lease (DESIGN 3.10): the pack's planner names what this
            # invocation is about to materialize; the parent owns the governor
            # and grants, denies, or delays. This side allocates only after
            # the grant, and releases in the finally below.
            selected_worker: InProcessWorker | None = worker
            if invocation.arm is not None:
                selected_worker = (arm_workers or {}).get(invocation.arm)
                if selected_worker is None or invocation.node_type not in set(
                    (body_arms or {}).get(invocation.arm, ())
                ):
                    await send_error(
                        invocation_id,
                        header,
                        f"unknown body arm {invocation.arm!r} for node type "
                        f"{invocation.node_type!r}",
                    )
                    return
            assert selected_worker is not None
            if not attention_route_token_matches_capabilities(
                selected_worker.attention_capabilities,
                invocation.attention_route_token,
            ):
                await send_error(
                    invocation_id,
                    header,
                    "attention route token does not match worker startup evidence",
                )
                return
            parent_manages_reservations = getattr(
                workgroup_handler,
                "parent_manages_invocation_reservations",
                lambda: False,
            )
            requests = (
                ()
                if parent_manages_reservations()
                else tuple(planner(invocation))
                if planner is not None
                else ()
            )
            if requests:
                grant: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
                pending_grants[invocation_id] = grant
                reserve_sent = True
                await send(
                    {
                        "type": "memoryReserve",
                        "requestId": invocation_id,
                        "requests": [
                            {"residency": r.residency, "nbytes": r.nbytes} for r in requests
                        ],
                    }
                )
                try:
                    denial = await grant
                finally:
                    pending_grants.pop(invocation_id, None)
                if denial is not None:
                    await send_error(invocation_id, header, f"memory admission failed: {denial}")
                    return
            started = time.perf_counter()
            workgroup_cancelled = getattr(workgroup_handler, "invocation_cancelled", None)
            inference_registries = None
            if invocation.extension_snapshot_digest is not None:
                inference = importlib.import_module("dinkster_inference")
                inference_registries = inference.materialize_inference_generation(
                    invocation.extension_snapshot_digest
                ).registries
            with use_execution_context(
                ExecutionContext(
                    arm=invocation.arm,
                    expected_execution_identity=invocation.expected_execution_identity,
                    extension_snapshot_digest=invocation.extension_snapshot_digest,
                    inference_registries=inference_registries,
                    fp8_matmul=invocation.fp8_matmul,
                    diffusion_dtype=invocation.diffusion_dtype,
                    text_dtype=invocation.text_dtype,
                    vae_dtype=invocation.vae_dtype,
                    attention_policy=invocation.attention_policy,
                    attention_route_token=invocation.attention_route_token,
                    preview_mode=invocation.preview_mode,
                    preview_animation=invocation.preview_animation,
                    cancelled=lambda: (
                        cancellation.is_set()
                        or (workgroup_cancelled is not None and workgroup_cancelled(invocation_id))
                    ),
                    node_id=invocation.node_id,
                    export_snapshot=invocation.export_snapshot,
                    materialize_source=(materialize_source if source_staging is not None else None),
                )
            ):
                before_invocation = getattr(workgroup_handler, "before_invocation", None)
                after_invocation = getattr(workgroup_handler, "after_invocation", None)
                if (before_invocation is None) != (after_invocation is None):
                    raise RuntimeError("workgroup invocation hooks must be paired")
                if before_invocation is not None:
                    await before_invocation(invocation_id)
                try:
                    result = await selected_worker.invoke(invocation, on_event=on_event)
                except BaseException as error:
                    if after_invocation is not None:
                        await after_invocation(
                            invocation_id,
                            f"{type(error).__name__}: {error}",
                        )
                    raise
                if after_invocation is not None:
                    await after_invocation(
                        invocation_id,
                        None if result.error is None else result.error.message,
                    )
            execute_ms = (time.perf_counter() - started) * 1000.0
            await flush_events()
            if result.outputs is not None:
                producer_arm = (
                    pack_name if invocation.arm is None else f"{pack_name}@{invocation.arm}"
                )
                result = InvocationResult(
                    outputs={
                        name: stamp_resource_producer_arm(
                            value, process_instance_token(), producer_arm
                        )
                        for name, value in result.outputs.items()
                    },
                    artifact_candidates=result.artifact_candidates,
                )
            if blob_transfer is not None:
                await result_transfer_lock.acquire()
                result_send_locked = True
            source_files = (
                source_asset_files(result.outputs, resolver_from_env())
                if blob_transfer is not None and result.outputs
                else []
            )
            conversation_checkpoint = codec.conversation_checkpoint()
            out_header, out_blobs, out_segments = encode_result(
                codec, result, invocation_id, execute_ms
            )
            pending_blobs: list[tuple[str, bytes | Path]] = list(codec.take_pending_store_blobs())
            pending_blobs.extend(source_files)
            if consumers and result.outputs:
                refs: set[str] = set()
                for value in result.outputs.values():
                    refs.update(value_resource_ids(value))
                if refs:
                    inflight_result_refs[invocation_id] = frozenset(refs)
            moved: dict[str, tuple[int, float]] = {}
            if blob_transfer is not None and pending_blobs:
                try:
                    # Staged input sources can be returned unchanged; transfer
                    # their bytes before closing the invocation's staging lease.
                    moved = await ensure_peer_holds(pending_blobs)
                except BaseException:
                    inflight_result_refs.pop(invocation_id, None)
                    for segment in out_segments:
                        release_segment(segment)
                    raise
            try:
                await run_sync_resource(close_staging)
            except Exception as exc:  # noqa: BLE001 - cleanup is execution correctness
                for segment in out_segments:
                    release_segment(segment)
                codec.restore_conversation(conversation_checkpoint)
                if result.error is None:
                    result = InvocationResult(
                        error=NodeError(
                            invocation.node_id,
                            invocation.node_type,
                            f"source staging cleanup failed: {exc}",
                            traceback=traceback.format_exc(),
                        )
                    )
                else:
                    result = InvocationResult(
                        error=NodeError(
                            result.error.node_id,
                            result.error.node_type,
                            result.error.message,
                            traceback=(
                                result.error.traceback
                                + "\nSource staging cleanup also failed:\n"
                                + traceback.format_exc()
                            ),
                            hints=result.error.hints,
                        )
                    )
                out_header, out_blobs, out_segments = encode_result(
                    codec, result, invocation_id, execute_ms
                )
                codec.take_pending_store_blobs()
                inflight_result_refs.pop(invocation_id, None)
            attribute_moved_wire(out_header, moved, out_blobs)
            await send(out_header, out_blobs, out_segments)
        except asyncio.CancelledError:
            raise  # engine gave up; nothing to report
        except Exception as exc:  # noqa: BLE001 - host must survive any invocation
            await flush_events()
            primary_traceback = traceback.format_exc()
            cleanup_traceback = ""
            try:
                await run_sync_resource(close_staging)
            except Exception:  # noqa: BLE001 - retain the primary host error
                cleanup_traceback = (
                    "\nSource staging cleanup also failed:\n" + traceback.format_exc()
                )
            error = NodeError(
                node_id=str(header.get("nodeId", "")),
                node_type=str(header.get("nodeType", "")),
                message=f"worker host error: {exc}",
                traceback=primary_traceback + cleanup_traceback,
            )
            with contextlib.suppress(Exception):
                await send(
                    {
                        "type": "result",
                        "invocationId": invocation_id,
                        "executeMs": 0.0,
                        "error": {
                            "nodeId": error.node_id,
                            "nodeType": error.node_type,
                            "message": error.message,
                            "traceback": error.traceback,
                        },
                    }
                )
        finally:
            if result_send_locked:
                result_transfer_lock.release()
            # Refuse late calls synchronously before cleanup can suspend. A
            # disconnect may cancel this task again while to_thread is still
            # queued, but copied execution contexts must never reopen staging.
            staging_closed.set()
            cleanup_task = asyncio.create_task(asyncio.to_thread(close_staging))
            try:
                await await_task_settlement(cleanup_task)
            except asyncio.CancelledError:
                # The invocation remains resource-active until off-loop
                # cleanup actually settles, even when its caller is gone.
                if not cleanup_task.cancelled() and isinstance(
                    (error := cleanup_task.exception()), Exception
                ):
                    log.warning(
                        "source staging cleanup failed after cancellation",
                        exc_info=error,
                    )
            except Exception:
                # Normal and host-error paths already applied cleanup failure
                # precedence above; the idempotent final attempt is best effort.
                pass
            # Close the event window on every exit so no orphaned send task
            # outlives its invocation. On the normal path flush_events()
            # already drained these; on cancellation, cancel rather than
            # await - a frame is buffered whole before write_frame's only
            # suspension (drain), so cancelling a send cannot tear a frame.
            event_window_open = False
            pending_events = [t for t in event_tasks if not t.done()]
            for task in pending_events:
                task.cancel()
            if pending_events:
                await asyncio.gather(*pending_events, return_exceptions=True)
            if reserve_sent:
                # Idempotent on the parent: releases a held lease, or tells a
                # still-acquiring one to let go the moment it is granted. Sent
                # on every exit path, including cancellation mid-grant.
                with contextlib.suppress(Exception):
                    await send({"type": "memoryRelease", "requestId": invocation_id})
            # Invocations are what change consumer state (a loader admitted a
            # resident): refresh the parent's snapshot at every one.
            await send_report()
            tasks.pop(invocation_id, None)
            cancellation_events.pop(invocation_id, None)
            if resume is not None and cancellation.is_set():
                finish_resumable_cancellation(invocation_id)

    async def run_lazy_status(header: dict[str, Any], blobs: list[bytes]) -> None:
        request_id = str(header.get("requestId", ""))
        reply: dict[str, object] = {
            "type": "lazyStatusResult",
            "requestId": request_id,
        }
        cancelled = False
        try:
            if request_id != str(header.get("invocationId", "")):
                raise ValueError("lazy-protocol-skew: request id mismatch")
            consumed: list[str] = []
            ordinary = decode_invocation(codec, header, blobs, consumed)
            if consumed:
                await send({"type": "shmAck", "segments": consumed})
            raw_hidden = cast("object", header.get("connectedUndemandedInputs"))
            if (
                not isinstance(raw_hidden, list)
                or not all(isinstance(item, str) for item in cast("list[object]", raw_hidden))
                or len(set(cast("list[object]", raw_hidden)))
                != len(cast("list[object]", raw_hidden))
            ):
                raise ValueError("lazy-protocol-malformed: invalid connected inputs")
            invocation = LazyStatusInvocation(
                request_id=request_id,
                node_id=ordinary.node_id,
                node_type=ordinary.node_type,
                available_inputs=ordinary.inputs,
                connected_undemanded_inputs=tuple(cast("list[str]", raw_hidden)),
                effective_schema=ordinary.effective_schema,
                executor=ordinary.executor,
                arm=ordinary.arm,
                expected_execution_identity=ordinary.expected_execution_identity,
                fp8_matmul=ordinary.fp8_matmul,
                diffusion_dtype=ordinary.diffusion_dtype,
                text_dtype=ordinary.text_dtype,
                vae_dtype=ordinary.vae_dtype,
                attention_policy=ordinary.attention_policy,
                attention_route_token=ordinary.attention_route_token,
                extension_snapshot_digest=ordinary.extension_snapshot_digest,
            )
            if invocation.arm is not None:
                raise ValueError("lazy-dispatch-unsupported: alternate body arm")
            result = await cast("LazyStatusWorker", worker).check_lazy_status(invocation)
            if result.error is not None:
                reply["error"] = {
                    "nodeId": result.error.node_id,
                    "nodeType": result.error.node_type,
                    "message": result.error.message,
                    "traceback": result.error.traceback,
                }
            else:
                requested_inputs: list[str] = []
                for index, item in enumerate(result.requested_inputs or ()):
                    if not isinstance(item, str):
                        reply["error"] = {
                            "nodeId": ordinary.node_id,
                            "nodeType": ordinary.node_type,
                            "message": (
                                "lazy-request-invalid: requested input at index "
                                f"{index} must be a string"
                            ),
                            "traceback": "",
                        }
                        break
                    requested_inputs.append(item)
                else:
                    reply["requestedInputs"] = requested_inputs
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:  # noqa: BLE001 - classify malformed peer requests
            reply["error"] = {
                "nodeId": str(header.get("nodeId", "")),
                "nodeType": str(header.get("nodeType", "")),
                "message": f"lazy-protocol-malformed: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
            if not cancelled:
                try:
                    await send(reply)
                except Exception:  # noqa: BLE001 - the parent must observe terminal failure
                    # This is the sole terminal frame for the request. If it
                    # cannot be serialized or sent, end the conversation so
                    # the parent's read loop fails and drains its pending
                    # lazy-status future instead of awaiting it forever.
                    log.exception("failed to send terminal lazy-status reply")
                    writer.close()
                    with contextlib.suppress(Exception):
                        await writer.wait_closed()
            tasks.pop(request_id, None)

    async def run_sampler_materialization(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        key = str(header.get("catalogKey", ""))
        reply: dict[str, object] = {
            "type": "samplerRegistryResult",
            "requestId": request_id,
        }
        try:
            inference = importlib.import_module("dinkster_inference")
            materialized = await run_sync_resource(inference.materialize_inference_generation, key)
            reply["extensions"] = [
                {
                    "id": extension_id,
                    "contributions": [
                        {
                            "surfaceId": sampler.surface_id,
                            "id": sampler.id,
                            "aliases": list(sampler.aliases),
                            "behaviorMetadata": [list(item) for item in sampler.behavior_metadata],
                        }
                        for sampler in samplers
                    ],
                }
                for extension_id, samplers in materialized.extensions
            ]
        except Exception as exc:  # noqa: BLE001 - staging needs the worker's refusal
            reply["error"] = f"{type(exc).__name__}: {exc}"
        with contextlib.suppress(Exception):
            await send(reply)

    async def run_inference_release(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        key = str(header.get("catalogKey", ""))
        reply: dict[str, object] = {
            "type": "inferenceReleaseResult",
            "requestId": request_id,
        }
        try:
            inference = importlib.import_module("dinkster_inference")
            await run_sync_resource(inference.release_inference_generation, key)
        except Exception as exc:  # noqa: BLE001 - release refusal must reach the parent
            reply["error"] = f"{type(exc).__name__}: {exc}"
        with contextlib.suppress(Exception):
            await send(reply)

    async def run_graph_compile(header: dict[str, Any], cancel_event: threading.Event) -> None:
        request_id = str(header.get("requestId", ""))
        reply: dict[str, object]
        try:
            try:
                inference = importlib.import_module("dinkster_inference")
                result = await run_sync_resource(
                    inference.compile_inference_graph,
                    header.get("generationKey"),
                    header.get("graph"),
                    header.get("targets"),
                    cancelled=cancel_event.is_set,
                )
                if not isinstance(result, Mapping):
                    raise TypeError("graph compiler returned a non-Mapping result")
                reply = dict(cast("Mapping[str, object]", result))
                reply["type"] = GRAPH_COMPILE_RESULT_TYPE
                reply["requestId"] = request_id
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - compiler failures cross as data
                error_name = getattr(exc, "error_name", None)
                reply = {
                    "type": GRAPH_COMPILE_RESULT_TYPE,
                    "requestId": request_id,
                    "errorName": (
                        error_name
                        if isinstance(error_name, str) and error_name
                        else GRAPH_COMPILE_ERROR_COMPILER_FAILURE
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            await send(reply)
        finally:
            current = asyncio.current_task()
            entry = compile_tasks.get(request_id)
            if entry is not None and entry[0] is current:
                compile_tasks.pop(request_id, None)

    async def run_asset_query(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        assert asset_staging is not None
        reply: dict[str, object] = {"type": "assetQueryResult", "requestId": request_id}
        try:
            digests_raw = header.get("digests")
            if not isinstance(digests_raw, list):
                raise AssetError("assetQuery requires a 'digests' list")
            digests = [str(digest) for digest in cast("list[object]", digests_raw)]
            held, missing = await asyncio.to_thread(asset_staging.held, digests)
            reply["held"] = held
            reply["missing"] = missing
        except AssetError as exc:
            reply["error"] = str(exc)
        await send(reply)

    async def run_fetch_choices(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        reply: dict[str, object] = {"type": "choicesResult", "requestId": request_id}
        choice_id = str(header.get("choiceId", ""))
        provider = (lazy_choices or {}).get(choice_id)
        if provider is None:
            reply["error"] = f"unknown lazy choice list {choice_id!r}"
        else:
            try:
                # Foreign pack code, invoked synchronously off the event
                # loop; exactly one invocation per parent fetch. Results are
                # held to the same grammar and JSON budget as static lists.
                values = await run_sync_resource(provider)
                reply["values"] = list(
                    validated_choice_values(values, subject=f"lazy choice {choice_id!r}")
                )
            except Exception as exc:
                reply["error"] = f"{type(exc).__name__}: {exc}"
        await send(reply)

    async def run_stage_assets(header: dict[str, Any], abort: threading.Event) -> None:
        request_id = str(header.get("requestId", ""))
        assert asset_staging is not None
        reply: dict[str, object] = {"type": "stageAssetsResult", "requestId": request_id}
        try:
            entries: list[StageAsset] = []
            try:
                assets_raw = header.get("assets")
                if not isinstance(assets_raw, list):
                    raise AssetError("stageAssets requires an 'assets' list")
                for entry in cast("list[object]", assets_raw):
                    if not isinstance(entry, Mapping):
                        raise AssetError("stageAssets entries must be objects")
                    entries.append(StageAsset.from_wire(cast("Mapping[str, object]", entry)))
            except AssetError as exc:
                reply["error"] = str(exc)
                await send(reply)
                return
            staged: list[str] = []
            failed: dict[str, str] = {}
            for asset in entries:
                totals = {"totalBytes": asset.size} if asset.size >= 0 else {}
                received = 0

                def on_progress(count: int) -> None:
                    nonlocal received
                    received = count  # written from the fetch thread, read here

                await send(
                    {
                        "type": "stageEvent",
                        "requestId": request_id,
                        "digest": asset.digest,
                        "event": "fetching",
                        **totals,
                    }
                )
                fetch = asyncio.create_task(
                    asyncio.to_thread(
                        asset_staging.stage,
                        asset,
                        should_abort=abort.is_set,
                        on_progress=on_progress,
                    )
                )
                # A cancelStage (or connection teardown) cancels THIS task
                # while the fetch thread drains to its next abort check; the
                # callback retrieves the eventual FetchAborted so the
                # orphaned task never logs an unretrieved exception.
                fetch.add_done_callback(_retrieve_task_exception)
                try:
                    reported = -1
                    while True:
                        done, _ = await asyncio.wait({fetch}, timeout=0.2)
                        if done:
                            break
                        if received != reported:
                            reported = received
                            await send(
                                {
                                    "type": "stageEvent",
                                    "requestId": request_id,
                                    "digest": asset.digest,
                                    "event": "progress",
                                    "bytesReceived": reported,
                                    **totals,
                                }
                            )
                    fetch.result()
                except asyncio.CancelledError:
                    abort.set()
                    try:
                        await await_task_settlement(fetch)
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise
                except FetchAborted:
                    # The engine asked to stop; it is not waiting for a
                    # result frame, and the writer already rolled back.
                    return
                except AssetError as exc:
                    # Fail fast: the whole staging request fails with the
                    # first digest that cannot land, and later entries are
                    # reported unattempted (absent from both lists).
                    failed[asset.digest] = str(exc)
                    await send(
                        {
                            "type": "stageEvent",
                            "requestId": request_id,
                            "digest": asset.digest,
                            "event": "failed",
                            "message": str(exc),
                        }
                    )
                    break
                staged.append(asset.digest)
                await send(
                    {
                        "type": "stageEvent",
                        "requestId": request_id,
                        "digest": asset.digest,
                        "event": "staged",
                        **totals,
                    }
                )
            reply["staged"] = staged
            if failed:
                reply["failed"] = failed
            await send(reply)
        finally:
            current = asyncio.current_task()
            stage_entry = stage_tasks.get(request_id)
            if stage_entry is not None and stage_entry[0] is current:
                stage_tasks.pop(request_id, None)

    async def run_legacy_checkpoint_conversion(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        source_path = str(header.get("sourcePath", ""))
        logical_name = str(header.get("logicalName", ""))
        reply: dict[str, object] = {
            "type": "legacyCheckpointConversionResult",
            "requestId": request_id,
        }
        status, error = await run_sync_resource(
            _convert_legacy_checkpoint, Path(source_path), logical_name
        )
        reply["status"] = status
        if error is not None:
            reply["error"] = error
        with contextlib.suppress(Exception):
            await send(reply)

    async def run_shed(header: dict[str, Any]) -> None:
        """Service one memoryShed frame. Its own task, like invocations: a
        slow unload must not stall the read loop that carries results and
        lease grants. The reply carries a fresh report so the parent's
        admission rescore sees the freed bytes the moment shed() returns."""
        request_id = str(header.get("requestId"))
        name = str(header.get("consumer", ""))
        device = str(header.get("device", ""))
        items_raw = header.get("items")
        items = (
            tuple(str(item) for item in cast("list[object]", items_raw))
            if isinstance(items_raw, list)
            else None
        )
        freed = 0
        consumer = consumers.get(name)
        # Item IDs are one consumer's namespace: pressure a consumer cannot
        # resolve must free nothing (mirrors the governor's own routing).
        if consumer is not None and (items is None or isinstance(consumer, DetailedConsumer)):
            try:
                freed = await consumer.shed(
                    PressureSignal(
                        device=device,
                        bytes_needed=int(header.get("bytesNeeded", 0)),
                        items=items,
                    )
                )
            except Exception:  # noqa: BLE001 - a raising shedder freed nothing provable
                freed = 0
        reply: dict[str, object] = {
            "type": "memoryShedResult",
            "requestId": request_id,
            "freedBytes": freed,
            "consumers": build_report(),
        }
        if telemetry is not None:
            # A shed just changed free memory: refresh the parent's view
            # now rather than waiting for the next report.
            reply["measured"] = measure()
        with contextlib.suppress(Exception):
            await send(reply)

    async def run_release_query(header: dict[str, Any]) -> None:
        """Propose ram-lane release candidates for the parent's gate.

        Residents named by results still in flight to the parent are
        excluded here AND at commit: the parent cannot pin a stub it has
        not decoded, so this side must not offer (or drop) what it has
        not yet been vouched for."""
        request_id = str(header.get("requestId"))
        name = str(header.get("consumer", ""))
        device = str(header.get("device", ""))
        items_raw = header.get("items")
        items = (
            tuple(str(item) for item in cast("list[object]", items_raw))
            if isinstance(items_raw, list)
            else None
        )
        candidates: Sequence[ReleaseCandidate] = ()
        consumer = consumers.get(name)
        if isinstance(consumer, ReleasableConsumer) and (
            items is None or isinstance(consumer, DetailedConsumer)
        ):
            try:
                candidates = tuple(
                    consumer.propose_release(
                        PressureSignal(
                            device=device,
                            bytes_needed=int(header.get("bytesNeeded", 0)),
                            items=items,
                        )
                    )
                )
            except Exception:  # noqa: BLE001 - a raising consumer proposes nothing
                candidates = ()
            held = held_resource_ids()
            candidates = tuple(c for c in candidates if c.resource_id not in held)
        with contextlib.suppress(Exception):
            await send(
                {
                    "type": "memoryReleaseCandidates",
                    "requestId": request_id,
                    "candidates": [
                        {
                            "itemId": c.item_id,
                            "resourceId": c.resource_id,
                            "nbytes": c.nbytes,
                            "token": c.token,
                        }
                        for c in candidates
                    ],
                }
            )

    def snapshot_commit_candidates(
        header: dict[str, Any],
    ) -> list[ReleaseCandidate]:
        """The commit's in-flight filter, run *synchronously in the read
        loop* the moment the commit frame is read. Ordering is the point:
        the parent sends its resultAck only after the commit (its send
        lock serializes them), so a hold registered before the result
        frame left is still registered when the commit frame is read -
        but only if this check runs before the read loop can process the
        following resultAck. Deferring it into the commit task would let
        the ack overtake it and release a resident whose stub the parent
        just decoded (DESIGN 3.10, the in-flight half of the gate)."""
        held = held_resource_ids()
        wanted: list[ReleaseCandidate] = []
        for wire_raw in cast("list[object]", header.get("candidates", [])):
            if not isinstance(wire_raw, Mapping):
                continue
            wire = cast("Mapping[str, Any]", wire_raw)
            try:
                candidate = ReleaseCandidate(
                    item_id=str(wire["itemId"]),
                    resource_id=str(wire["resourceId"]),
                    nbytes=int(wire.get("nbytes", 0)),
                    token=str(wire["token"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if candidate.resource_id in held:
                continue  # a result naming this resident is in flight
            wanted.append(candidate)
        return wanted

    async def run_release_commit(
        header: dict[str, Any], wanted: Sequence[ReleaseCandidate]
    ) -> None:
        """Release the candidates the parent's gate approved. The pool's
        token check refuses anything used since proposal; the in-flight
        filter already ran synchronously at frame read (see
        snapshot_commit_candidates). The reply names what was actually
        released so the parent can roll back the rest."""
        request_id = str(header.get("requestId"))
        name = str(header.get("consumer", ""))
        device = str(header.get("device", ""))
        consumer = consumers.get(name)
        freed = 0
        released: list[str] = []
        if isinstance(consumer, ReleasableConsumer):
            resource_by_item = {c.item_id: c.resource_id for c in wanted}
            outcome: Mapping[str, int]
            try:
                outcome = await consumer.release(device, list(wanted))
            except Exception:  # noqa: BLE001 - a raising consumer released nothing provable
                outcome = {}
            freed = sum(
                nbytes for item_id, nbytes in outcome.items() if item_id in resource_by_item
            )
            # Filter, don't index: a consumer reporting an item it was never
            # asked about must not crash this task - the parent is awaiting
            # exactly one reply per request, forever.
            released = [
                resource_by_item[item_id] for item_id in outcome if item_id in resource_by_item
            ]
        with contextlib.suppress(Exception):
            await send(
                {
                    "type": "memoryReleaseResult",
                    "requestId": request_id,
                    "freedBytes": freed,
                    "released": released,
                    "consumers": build_report(),
                }
            )

    async def run_full_release_query(header: dict[str, Any], admitted: bool) -> None:
        request_id = str(header.get("requestId", ""))
        operation_id = str(header.get("operationRequestId", ""))
        worker_instance = str(header.get("workerInstance", ""))
        reply: dict[str, object] = {
            "type": "memoryFreeCandidates",
            "requestId": request_id,
            "operationRequestId": operation_id,
            "workerInstance": process_instance_token(),
            "status": "complete",
            "consumers": [],
        }
        if (
            not admitted
            or not request_id
            or not operation_id
            or worker_instance != process_instance_token()
        ):
            reply["status"] = "error"
            reply["error"] = (
                "worker full release is already active"
                if not admitted
                else "full-release request identity is invalid"
            )
            if admitted:
                full_release_operations.discard(operation_id)
                maintenance_operations.discard((connection_token, operation_id))
            await send(reply)
            return
        active_work = bool(
            cancellation_events
            or workgroup_tasks
            or inflight_result_refs
            or any(not task.done() for task in resource_tasks)
        )
        if active_work:
            reply["status"] = "busy"
        results: list[dict[str, object]] = []
        held = held_resource_ids()
        for name, consumer in consumers.items():
            if active_work:
                results.append({"consumer": name, "status": "busy", "candidates": []})
            elif isinstance(consumer, FullReleasableConsumer):
                try:
                    proposed = tuple(consumer.propose_full_release())
                    available = tuple(
                        candidate for candidate in proposed if candidate.resource_id not in held
                    )
                    results.append(
                        {
                            "consumer": name,
                            "status": "busy" if len(available) != len(proposed) else "ready",
                            "candidates": [
                                {
                                    "itemId": candidate.item_id,
                                    "resourceId": candidate.resource_id,
                                    "nbytes": candidate.nbytes,
                                    "token": candidate.token,
                                }
                                for candidate in available
                            ],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - explicit incomplete result
                    results.append({"consumer": name, "status": "error", "error": str(exc)})
            elif isinstance(consumer, FullReleaseConsumer):
                results.append({"consumer": name, "status": "ready", "candidates": []})
            else:
                results.append({"consumer": name, "status": "unsupported"})
        reply["consumers"] = results
        await send(reply)

    def snapshot_full_release_commit(
        header: dict[str, Any],
    ) -> tuple[str, dict[str, tuple[str, tuple[ReleaseCandidate, ...]]], str | None]:
        selected: dict[str, tuple[str, tuple[ReleaseCandidate, ...]]] = {}
        held = held_resource_ids()
        active_work = bool(
            cancellation_events
            or workgroup_tasks
            or inflight_result_refs
            or any(not task.done() for task in resource_tasks)
        )
        query_worker_status = header.get("workerStatus")
        raw_consumers = header.get("consumers", ())
        operation_id = header.get("operationRequestId")
        if (
            not isinstance(operation_id, str)
            or operation_id not in full_release_operations
            or query_worker_status not in {"complete", "busy"}
            or not isinstance(raw_consumers, list)
        ):
            return "error", selected, "full-release commit is invalid"
        worker_status = "busy" if active_work or query_worker_status == "busy" else "complete"
        for raw in cast("list[object]", raw_consumers):
            if not isinstance(raw, Mapping):
                return "error", {}, "full-release consumer selection is invalid"
            wire = cast("Mapping[str, Any]", raw)
            name = wire.get("consumer")
            query_status = wire.get("queryStatus")
            candidates_raw = wire.get("candidates")
            if (
                not isinstance(name, str)
                or not name
                or name in selected
                or name not in consumers
                or query_status not in {"ready", "busy", "unsupported", "error"}
                or not isinstance(candidates_raw, list)
                or (query_status != "ready" and candidates_raw)
            ):
                return "error", {}, "full-release consumer selection is invalid"
            candidates: list[ReleaseCandidate] = []
            item_ids: set[str] = set()
            for candidate_raw in cast("list[object]", candidates_raw):
                if not isinstance(candidate_raw, Mapping):
                    return "error", {}, "full-release candidate is invalid"
                candidate_wire = cast("Mapping[str, Any]", candidate_raw)
                item_id = candidate_wire.get("itemId")
                resource_id = candidate_wire.get("resourceId")
                nbytes = candidate_wire.get("nbytes", 0)
                token = candidate_wire.get("token")
                if (
                    not isinstance(item_id, str)
                    or not item_id
                    or item_id in item_ids
                    or not isinstance(resource_id, str)
                    or not resource_id
                    or type(nbytes) is not int
                    or nbytes < 0
                    or not isinstance(token, str)
                    or not token
                ):
                    return "error", {}, "full-release candidate is invalid"
                candidate = ReleaseCandidate(
                    item_id=item_id,
                    resource_id=resource_id,
                    nbytes=nbytes,
                    token=token,
                )
                item_ids.add(item_id)
                if candidate.resource_id not in held:
                    candidates.append(candidate)
            selected[name] = (
                "busy" if worker_status != "complete" else query_status,
                () if worker_status != "complete" else tuple(candidates),
            )
        if set(selected) != set(consumers):
            return "error", {}, "full-release consumer declarations changed"
        return worker_status, selected, None

    async def run_full_release_commit(
        header: dict[str, Any],
        worker_status: str,
        selected: dict[str, tuple[str, tuple[ReleaseCandidate, ...]]],
        validation_error: str | None,
    ) -> None:
        request_id = str(header.get("requestId", ""))
        operation_id = str(header.get("operationRequestId", ""))
        worker_instance = str(header.get("workerInstance", ""))
        reply: dict[str, object] = {
            "type": "memoryFreeResult",
            "requestId": request_id,
            "operationRequestId": operation_id,
            "workerInstance": process_instance_token(),
            "status": worker_status,
            "consumers": [],
        }
        results: list[dict[str, object]] = []
        try:
            if not request_id or not operation_id or worker_instance != process_instance_token():
                reply["status"] = "error"
                reply["error"] = "full-release commit identity is invalid"
                await send(reply)
                return
            if validation_error is not None:
                reply["error"] = validation_error
            for name, consumer in consumers.items():
                query_status, candidates = selected.get(name, ("error", ()))
                released: list[str] = []
                if query_status in {"busy", "unsupported", "error"}:
                    result: dict[str, object] = {"consumer": name, "status": query_status}
                    if query_status == "error":
                        result["error"] = "consumer selection was not valid"
                    results.append(result)
                    continue
                try:
                    if isinstance(consumer, FullReleasableConsumer):
                        resource_by_item = {
                            candidate.item_id: candidate.resource_id for candidate in candidates
                        }
                        outcome = await consumer.release_full(candidates)
                        released = [
                            resource_by_item[item_id]
                            for item_id in outcome.released
                            if item_id in resource_by_item
                        ]
                        result = {
                            "consumer": name,
                            "status": outcome.result.status,
                            "released": released,
                        }
                        if outcome.result.error is not None:
                            result["error"] = outcome.result.error
                        results.append(result)
                    elif isinstance(consumer, FullReleaseConsumer):
                        outcome = await consumer.full_release()
                        status = getattr(outcome, "status", None)
                        error = getattr(outcome, "error", None)
                        if (
                            status not in {"complete", "busy", "unsupported", "error"}
                            or (status == "error" and (not isinstance(error, str) or not error))
                            or (status != "error" and error is not None)
                        ):
                            raise ValueError("consumer returned an invalid full release result")
                        result: dict[str, object] = {
                            "consumer": name,
                            "status": status,
                            "released": [],
                        }
                        if status == "error":
                            result["error"] = error
                        results.append(result)
                    else:
                        results.append({"consumer": name, "status": "unsupported", "released": []})
                except Exception as exc:  # noqa: BLE001 - explicit incomplete result
                    results.append(
                        {
                            "consumer": name,
                            "status": "error",
                            "error": str(exc),
                            "released": released,
                        }
                    )
            reply["consumers"] = results
            await send(reply)
        finally:
            full_release_commits.discard(operation_id)
            full_release_operations.discard(operation_id)
            maintenance_operations.discard((connection_token, operation_id))

    async def run_full_release_abort(header: dict[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        operation_id = str(header.get("operationRequestId", ""))
        worker_instance = header.get("workerInstance")
        valid = (
            bool(request_id)
            and bool(operation_id)
            and worker_instance == process_instance_token()
            and operation_id in full_release_operations
            and operation_id not in full_release_commits
        )
        if valid:
            full_release_operations.discard(operation_id)
            maintenance_operations.discard((connection_token, operation_id))
        reply: dict[str, object] = {
            "type": "memoryFreeAborted",
            "requestId": request_id,
            "operationRequestId": operation_id,
            "workerInstance": process_instance_token(),
            "status": "complete" if valid else "error",
        }
        if not valid:
            reply["error"] = "full-release abort identity is invalid or commit is active"
        await send(reply)

    shed_tasks: set[asyncio.Task[None]] = set()
    full_release_tasks: set[asyncio.Task[None]] = set()
    workgroup_tasks: set[asyncio.Task[None]] = set()
    route_tasks: dict[str, asyncio.Task[None]] = {}
    rendition_tasks: dict[str, asyncio.Task[None]] = {}
    pack_routes = {route.id: route for _, item in extension_contributions for route in item.routes}

    async def run_pack_route(header: Mapping[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        reply: dict[str, object] = {"type": "packRouteResult", "requestId": request_id}
        try:
            requested = PackRoute.from_wire(header.get("route"))
            route = pack_routes.get(requested.id)
            if route != requested:
                raise ValueError("route does not match this worker's declaration")
            assert route is not None
            data = route.request.validate(header.get("data"))
            handler = cast(Callable[[Mapping[str, object]], object], resolve_entry(route.handler))
            async with asyncio.timeout(PACK_ROUTE_TIMEOUT):
                result = await run_sync_resource(handler, data)
                if inspect.isawaitable(result):
                    result = await cast(Awaitable[object], result)
            reply["data"] = route.response.validate(result)
        except Exception:
            reply["error"] = "pack-route-failed"
        finally:
            route_tasks.pop(request_id, None)
        await send(reply)

    def requested_rendition(header: Mapping[str, Any]):
        type_id = header.get("typeId")
        kind = header.get("kind")
        if type(type_id) is not str or not type_id or type(kind) is not str or not kind:
            raise InvalidRenditionRequest("rendition type and kind must be non-empty strings")
        spec = next(
            (item for item in worker.registry.renditions_of(type_id) if item.kind == kind), None
        )
        if spec is None:
            raise InvalidRenditionRequest(f"{type_id}: no rendition {kind!r}")
        return spec

    async def run_rendition(header: Mapping[str, Any], blobs: Sequence[bytes]) -> None:
        request_id = str(header.get("requestId", ""))
        reply: dict[str, object] = {"type": "renditionResult", "requestId": request_id}
        result_blobs: list[bytes] = []
        try:
            spec = requested_rendition(header)
            parameters_raw = header.get("parameters", {})
            if not isinstance(parameters_raw, Mapping) or any(
                type(name) is not str or type(value) is not str
                for name, value in cast("Mapping[object, object]", parameters_raw).items()
            ):
                raise InvalidRenditionRequest("rendition parameters must be strings")
            parameters = cast("Mapping[str, str]", parameters_raw)
            if header.get("type") == "resolveRendition":
                meta_blob = header.get("metaBlob")
                if type(meta_blob) is not int or not 0 <= meta_blob < len(blobs):
                    raise InvalidRenditionRequest("rendition metadata is missing")
                metadata = default_decode(blobs[meta_blob])
                if not isinstance(metadata, Mapping):
                    raise InvalidRenditionRequest("rendition metadata must be a mapping")
                meta = cast("Mapping[str, object]", metadata)
                reply["mime"] = spec.mime_for(meta)
                if header.get("normalize") is True:
                    reply["parameters"] = dict(spec.normalize_parameters(parameters, meta))
            else:
                value_wire = header.get("value")
                if not isinstance(value_wire, Mapping):
                    raise InvalidRenditionRequest("rendition value is missing")
                consumed: list[str] = []
                value, _ = codec.decode(cast("Mapping[str, Any]", value_wire), blobs, consumed)
                if consumed:
                    await send({"type": "shmAck", "segments": consumed})
                rendition = await run_sync_resource(
                    worker.registry.render, value, spec.kind, parameters or None
                )
                reply["mime"] = rendition.mime
                reply["dataBlob"] = 0
                result_blobs.append(rendition.data)
        except InvalidRenditionRequest as exc:
            reply.update(error="invalid-request", message=str(exc))
        except RenditionUnavailable as exc:
            reply.update(error="unavailable", message=str(exc))
        except Exception as exc:
            reply.update(error="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            rendition_tasks.pop(request_id, None)
        await send(reply, result_blobs)

    async def run_workgroup(command: WorkGroupMessage) -> None:
        assert workgroup_handler is not None
        for reply in await handle_workgroup_command(workgroup_handler, command):
            await send(reply)

    def workgroup_done(task: asyncio.Task[None]) -> None:
        workgroup_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                log.error("workgroup command handler failed", exc_info=error)
                writer.close()

    try:
        while True:
            frame = await (resume.read() if resume is not None else read_frame(reader))
            if frame is None:
                break
            header, blobs = frame
            kind = header.get("type")
            maintenance_active = bool(maintenance_operations)
            if maintenance_active and kind == WORKGROUP_FRAME_TYPE:
                command = workgroup_command_from_frame(header, blobs)
                refusal: WorkGroupMessage | None = None
                if type(command) is PrepareReplica:
                    refusal = ReplicaRefused(
                        worker=command.worker,
                        replica=command.replica,
                        group=command.group,
                        attempt=command.attempt,
                        device=command.device,
                        reason="worker memory maintenance is active",
                    )
                elif type(command) in {BeginWorkGroup, CommitWorkGroup}:
                    refusal = WorkGroupRefused(
                        worker=command.worker,
                        replica=command.replica,
                        group=command.group,
                        attempt=command.attempt,
                        device=command.device,
                        reason="worker memory maintenance is active",
                    )
                elif type(command) is RunWorkUnit:
                    refusal = WorkUnitFailed(
                        worker=command.worker,
                        replica=command.replica,
                        group=command.group,
                        attempt=command.attempt,
                        device=command.device,
                        unit=command.unit,
                        slot=command.slot,
                        reason="worker memory maintenance is active",
                    )
                if refusal is not None:
                    await send(workgroup_frame(refusal))
                    continue
            if maintenance_active and kind == "invoke":
                await send_error(
                    str(header.get("invocationId", "")),
                    header,
                    "worker memory maintenance is active",
                )
                continue
            if maintenance_active and kind == "fetchChoices":
                await send(
                    {
                        "type": "choicesResult",
                        "requestId": str(header.get("requestId", "")),
                        "error": "worker memory maintenance is active",
                    }
                )
                continue
            if maintenance_active and kind == "checkLazyStatus":
                await send(
                    {
                        "type": "lazyStatusResult",
                        "requestId": str(header.get("requestId", "")),
                        "error": {
                            "nodeId": str(header.get("nodeId", "")),
                            "nodeType": str(header.get("nodeType", "")),
                            "message": "worker memory maintenance is active",
                            "traceback": "",
                        },
                    }
                )
                continue
            if maintenance_active and kind == GRAPH_COMPILE_REQUEST_TYPE:
                await send(
                    {
                        "type": GRAPH_COMPILE_RESULT_TYPE,
                        "requestId": str(header.get("requestId", "")),
                        "errorName": GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
                        "error": "worker memory maintenance is active",
                    }
                )
                continue
            if maintenance_active and kind == "materializeSamplerRegistry":
                await send(
                    {
                        "type": "samplerRegistryResult",
                        "requestId": str(header.get("requestId", "")),
                        "error": "worker memory maintenance is active",
                    }
                )
                continue
            if maintenance_active and kind == "convertLegacyCheckpoint":
                await send(
                    {
                        "type": "legacyCheckpointConversionResult",
                        "requestId": str(header.get("requestId", "")),
                        "status": "error",
                        "error": "worker memory maintenance is active",
                    }
                )
                continue
            if maintenance_active and kind == "packRoute":
                await send(
                    {
                        "type": "packRouteResult",
                        "requestId": str(header.get("requestId", "")),
                        "error": "pack-route-failed",
                    }
                )
                continue
            if kind == "packRoute":
                request_id = str(header.get("requestId", ""))
                if request_id in route_tasks:
                    raise BoundaryError("duplicate pack route request id")
                route_tasks[request_id] = track_resource_task(
                    asyncio.create_task(run_pack_route(header))
                )
            elif kind == "cancelPackRoute":
                task = route_tasks.get(str(header.get("requestId", "")))
                if task is not None:
                    task.cancel()
            elif kind in ("resolveRendition", "renderRendition"):
                request_id = str(header.get("requestId", ""))
                if not request_id or request_id in rendition_tasks:
                    raise BoundaryError("duplicate rendition request id")
                rendition_tasks[request_id] = track_resource_task(
                    asyncio.create_task(run_rendition(header, blobs))
                )
            elif kind == "invoke":
                invocation_id = str(header["invocationId"])
                if maintenance_operations:
                    await send_error(
                        invocation_id,
                        header,
                        "worker memory maintenance is active",
                    )
                    continue
                if resume is not None:
                    key = resume.key_for_invocation(invocation_id)
                    if key is None:
                        raise BoundaryError("invoke was not admitted for resume")
                    await send({"type": "invokeAccepted", **key.to_wire()})
                cancellation = threading.Event()
                cancellation_events[invocation_id] = cancellation
                tasks[invocation_id] = track_resource_task(
                    asyncio.create_task(run_invocation(header, blobs, cancellation))
                )
            elif kind == "rebind":
                if resume is None:
                    raise BoundaryError("rebind received on a non-resumable conversation")
                response, requested, cancellations = resume.plan_rebind(header)
                last_sequences = {
                    InvocationKey.from_header(cast("Mapping[str, object]", entry)): cast(
                        "int", cast("Mapping[str, object]", entry)["lastEventSeq"]
                    )
                    for entry in cast("list[object]", header["invocations"])
                }
                cancelled_tasks: list[asyncio.Task[None]] = []
                for key in cancellations:
                    cancellation = cancellation_events.get(key.invocation_id)
                    if cancellation is not None:
                        cancellation.set()
                    task = tasks.get(key.invocation_id)
                    if task is not None:
                        task.cancel()
                        cancelled_tasks.append(task)
                if cancelled_tasks:
                    await asyncio.gather(*cancelled_tasks, return_exceptions=True)
                for key in cancellations:
                    finish_resumable_cancellation(key.invocation_id)
                await resume.complete_rebind(response, requested, last_sequences)
            elif kind == WORKGROUP_FRAME_TYPE:
                if workgroup_handler is None:
                    raise BoundaryError("workgroup frame received without a negotiated handler")
                command = workgroup_command_from_frame(header, blobs)
                workgroup_task = asyncio.create_task(run_workgroup(command))
                track_resource_task(workgroup_task)
                workgroup_tasks.add(workgroup_task)
                workgroup_task.add_done_callback(workgroup_done)
            elif kind == "checkLazyStatus":
                request_id = str(header.get("requestId", ""))
                if not request_id or request_id in tasks:
                    await send(
                        {
                            "type": "lazyStatusResult",
                            "requestId": request_id,
                            "error": {
                                "nodeId": str(header.get("nodeId", "")),
                                "nodeType": str(header.get("nodeType", "")),
                                "message": "lazy-protocol-skew: duplicate request id",
                                "traceback": "",
                            },
                        }
                    )
                else:
                    tasks[request_id] = track_resource_task(
                        asyncio.create_task(run_lazy_status(header, blobs))
                    )
            elif kind == "cancelLazyStatus":
                request_id = str(header.get("requestId", ""))
                task = tasks.get(request_id)
                if task is not None:
                    task.cancel()
            elif kind == "materializeSamplerRegistry":
                sampler_task = asyncio.create_task(run_sampler_materialization(header))
                track_resource_task(sampler_task)
                shed_tasks.add(sampler_task)
                sampler_task.add_done_callback(shed_tasks.discard)
            elif kind == "releaseInferenceGeneration":
                release_task = asyncio.create_task(run_inference_release(header))
                shed_tasks.add(release_task)
                release_task.add_done_callback(shed_tasks.discard)
            elif kind == "convertLegacyCheckpoint":
                conversion_task = asyncio.create_task(run_legacy_checkpoint_conversion(header))
                track_resource_task(conversion_task)
                shed_tasks.add(conversion_task)
                conversion_task.add_done_callback(shed_tasks.discard)
            elif kind == GRAPH_COMPILE_REQUEST_TYPE:
                request_id = str(header.get("requestId", ""))
                cancel_event = threading.Event()
                compile_task = asyncio.create_task(run_graph_compile(header, cancel_event))
                track_resource_task(compile_task)
                compile_tasks[request_id] = (compile_task, cancel_event)
            elif kind == GRAPH_COMPILE_CANCEL_TYPE:
                request_id = str(header.get("requestId", ""))
                compile_entry = compile_tasks.get(request_id)
                if compile_entry is not None:
                    compile_entry[1].set()
                    compile_tasks.pop(request_id, None)
                    compile_entry[0].cancel()
                    await asyncio.gather(compile_entry[0], return_exceptions=True)
            elif kind == "assetQuery":
                if asset_staging is None:
                    raise BoundaryError("assetQuery received without negotiated asset staging")
                query_task = asyncio.create_task(run_asset_query(header))
                shed_tasks.add(query_task)
                query_task.add_done_callback(shed_tasks.discard)
            elif kind == "fetchChoices":
                if not lazy_choices:
                    raise BoundaryError("fetchChoices received without announced lazy choices")
                fetch_task = asyncio.create_task(run_fetch_choices(header))
                track_resource_task(fetch_task)
                shed_tasks.add(fetch_task)
                fetch_task.add_done_callback(shed_tasks.discard)
            elif kind == "stageAssets":
                if asset_staging is None:
                    raise BoundaryError("stageAssets received without negotiated asset staging")
                request_id = str(header.get("requestId", ""))
                stage_abort = threading.Event()
                stage_task = asyncio.create_task(run_stage_assets(header, stage_abort))
                track_resource_task(stage_task)
                stage_tasks[request_id] = (stage_task, stage_abort)
            elif kind == "cancelStage":
                request_id = str(header.get("requestId", ""))
                stage_entry = stage_tasks.get(request_id)
                if stage_entry is not None:
                    stage_tasks.pop(request_id, None)
                    stage_entry[1].set()
                    stage_entry[0].cancel()
                    await asyncio.gather(stage_entry[0], return_exceptions=True)
            elif kind == "cancel":
                invocation_id = str(header.get("invocationId"))
                cancel_key: InvocationKey | None = None
                if resume is not None:
                    cancel_key = InvocationKey.from_header(header)
                    if resume.key_for_invocation(invocation_id) != cancel_key:
                        raise BoundaryError("cancel names no resumable invocation")
                cancellation = cancellation_events.get(invocation_id)
                if cancellation is not None:
                    cancellation.set()
                task = tasks.get(invocation_id)
                if task is not None:
                    task.cancel()
                if resume is not None and cancel_key is not None:

                    async def complete_cancel(
                        invocation_task: asyncio.Task[None] | None,
                        key: InvocationKey,
                    ) -> None:
                        if invocation_task is not None:
                            await asyncio.gather(invocation_task, return_exceptions=True)
                        finish_resumable_cancellation(key.invocation_id)
                        await send({"type": "cancelAck", **key.to_wire()})

                    cancel_task = asyncio.create_task(complete_cancel(task, cancel_key))
                    shed_tasks.add(cancel_task)
                    cancel_task.add_done_callback(shed_tasks.discard)
            elif kind == "memoryGrant":
                if resume is not None:
                    resume.grant_received(header.get("requestId"))
                grant_future = pending_grants.get(str(header.get("requestId")))
                if grant_future is not None and not grant_future.done():
                    grant_future.set_result(None)
            elif kind == "aimdoHeadroom":
                if "baseBytes" in header:
                    _handle_aimdo_headroom(header.get("extraBytes"), header.get("baseBytes"))
                else:
                    _handle_aimdo_headroom(header.get("extraBytes"))
            elif kind == "memoryDeny":
                if resume is not None:
                    resume.grant_received(header.get("requestId"))
                grant_future = pending_grants.get(str(header.get("requestId")))
                if grant_future is not None and not grant_future.done():
                    grant_future.set_result(str(header.get("message", "denied")))
            elif kind == "memoryDevices":
                devices_raw = header.get("devices")
                declared_devices.clear()
                if isinstance(devices_raw, list):
                    declared_devices.update(
                        str(device) for device in cast("list[object]", devices_raw)
                    )
                await send_report()
            elif kind == "memoryShed":
                shed_task = asyncio.create_task(run_shed(header))
                shed_tasks.add(shed_task)
                shed_task.add_done_callback(shed_tasks.discard)
            elif kind == "memoryReleaseQuery":
                release_task = asyncio.create_task(run_release_query(header))
                shed_tasks.add(release_task)
                release_task.add_done_callback(shed_tasks.discard)
            elif kind == "memoryReleaseCommit":
                # The held filter MUST run before the next frame is read: a
                # resultAck right behind this commit would lift the hold the
                # filter exists to observe (see snapshot_commit_candidates).
                wanted = snapshot_commit_candidates(header)
                release_task = asyncio.create_task(run_release_commit(header, wanted))
                shed_tasks.add(release_task)
                release_task.add_done_callback(shed_tasks.discard)
            elif kind == "memoryFreeQuery":
                operation_id = str(header.get("operationRequestId", ""))
                admitted = (
                    bool(operation_id)
                    and not full_release_operations
                    and not maintenance_operations
                )
                if admitted:
                    full_release_operations.add(operation_id)
                    maintenance_operations.add((connection_token, operation_id))
                release_task = asyncio.create_task(run_full_release_query(header, admitted))
                full_release_tasks.add(release_task)
                release_task.add_done_callback(full_release_tasks.discard)
            elif kind == "memoryFreeCommit":
                operation_id = str(header.get("operationRequestId", ""))
                if operation_id in full_release_commits:
                    await send(
                        {
                            "type": "memoryFreeResult",
                            "requestId": str(header.get("requestId", "")),
                            "operationRequestId": operation_id,
                            "workerInstance": process_instance_token(),
                            "status": "error",
                            "error": "full-release commit is already active",
                            "consumers": [],
                        }
                    )
                    continue
                worker_status, selected, validation_error = snapshot_full_release_commit(header)
                if operation_id in full_release_operations:
                    full_release_commits.add(operation_id)
                release_task = asyncio.create_task(
                    run_full_release_commit(header, worker_status, selected, validation_error)
                )
                full_release_tasks.add(release_task)
                release_task.add_done_callback(full_release_tasks.discard)
            elif kind == "memoryFreeAbort":
                release_task = asyncio.create_task(run_full_release_abort(header))
                full_release_tasks.add(release_task)
                release_task.add_done_callback(full_release_tasks.discard)
            elif kind == "blobQuery":
                if blob_transfer is not None:
                    await blob_transfer.answer_query(header)
            elif kind == "blobQueryResult":
                if blob_transfer is not None:
                    blob_transfer.resolve_query(header)
            elif kind == "blobData":
                if blob_transfer is not None:
                    await blob_transfer.accept_chunk(header, blobs)
            elif kind == "heartbeat":
                # Lease renewal (service.py counts every received frame;
                # this reply is the peer's own liveness evidence).
                await send({"type": "heartbeatAck"})
            elif kind == "resultAck":
                # The parent decoded and pinned the result's references;
                # its gate can see them now, so this side's hold ends.
                invocation_id = str(header.get("invocationId"))
                if resume is not None:
                    key = InvocationKey.from_header(header)
                    resume.acknowledge_result(header)
                    inflight_result_refs.pop(invocation_id, None)
                    await send({"type": "resultAcked", **key.to_wire()})
                else:
                    inflight_result_refs.pop(invocation_id, None)
            elif kind == "shmAck":
                for name in header.get("segments", ()):
                    segment = sent_segments.pop(str(name), None)
                    if segment is not None:
                        release_segment(segment)
            elif kind == "shutdown":
                break
    finally:
        pending = (
            list(tasks.values())
            + list(shed_tasks)
            + list(workgroup_tasks)
            + list(route_tasks.values())
            + list(rendition_tasks.values())
            + [task for task, _ in compile_tasks.values()]
            + [task for task, _ in stage_tasks.values()]
            + ([schema_reload_task] if schema_reload_task is not None else [])
        )
        for cancellation in cancellation_events.values():
            cancellation.set()
        for _, cancellation in compile_tasks.values():
            cancellation.set()
        for _, stage_abort in stage_tasks.values():
            stage_abort.set()
        for task in pending:
            task.cancel()
        pending_settlement = asyncio.gather(*pending, return_exceptions=True)
        try:
            await await_task_settlement(pending_settlement)
        except asyncio.CancelledError:
            pass
        release_settlement = asyncio.gather(*full_release_tasks, return_exceptions=True)
        try:
            await await_task_settlement(release_settlement)
        except asyncio.CancelledError:
            pass
        for operation_id in tuple(full_release_operations - full_release_commits):
            full_release_operations.discard(operation_id)
            maintenance_operations.discard((connection_token, operation_id))
        compile_tasks.clear()
        stage_tasks.clear()
        if blob_transfer is not None:
            blob_transfer.close()
        # Unacknowledged segments die with the conversation (hazard H14).
        for segment in sent_segments.values():
            release_segment(segment)
        sent_segments.clear()
        if resume is None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dinkster isolated worker host")
    parser.add_argument(
        "--endpoint", required=True, action="append", help="boundary endpoint to connect back to"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        action="append",
        help="path to the pack's dinkster-pack.toml",
    )
    parser.add_argument("--shm-threshold", type=int, default=DEFAULT_SHM_THRESHOLD)
    parser.add_argument("--no-shm", action="store_true")
    parser.add_argument("--aimdo-init", action="store_true")
    parser.add_argument("--aimdo-arm", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--reserve-vram", type=int, metavar="BYTES")
    parser.add_argument(
        "--comfy-args-json",
        type=parse_comfy_args,
        default=(),
        metavar="JSON",
        help="validated ComfyUI argv as one JSON array",
    )
    parser.add_argument(
        "--vram-budget",
        action="append",
        default=[],
        type=_parse_vram_budget,
        metavar="INDEX=BYTES",
    )
    args = parser.parse_args()
    if len(args.endpoint) != len(args.manifest):
        parser.error("--endpoint and --manifest must be supplied in equal counts")
    if args.reserve_vram is not None and args.reserve_vram < 0:
        parser.error("--reserve-vram must be non-negative")
    vram_budgets: dict[int, int] = {}
    for index, nbytes in args.vram_budget:
        if index in vram_budgets:
            parser.error(f"duplicate --vram-budget index {index}")
        vram_budgets[index] = nbytes
    # Launched children inherit the host environment and stderr, so the
    # host's DINKSTER_LOG_LEVEL/DINKSTER_LOG settings apply here without any
    # forwarding machinery (origin names survive the process boundary).
    configure_logging_from_env(os.environ)
    # Pack stdout/stderr still reaches the shared terminal, and while a node
    # executes it is also forwarded as attributed execution log events.
    install_stream_capture()
    global _accelerator_headroom_base  # noqa: PLW0603 - process policy source
    _accelerator_headroom_base = args.reserve_vram
    aimdo_armed = args.aimdo_arm in ("auto", "on")
    if _accelerator_headroom_base is None and aimdo_armed:
        _accelerator_headroom_base = DEFAULT_ACCELERATOR_HEADROOM_BYTES
    os.environ.pop(_ACCELERATOR_HEADROOM_ENV, None)
    if _accelerator_headroom_base is not None:
        os.environ[_ACCELERATOR_HEADROOM_ENV] = str(_accelerator_headroom_base)
    os.environ.pop(_AIMDO_HEADROOM_TARGET_ENV, None)
    simple_vram_headroom = None
    if aimdo_armed:
        assert _accelerator_headroom_base is not None
        simple_vram_headroom = AcceleratorMemoryPolicy(
            physical_headroom_bytes=_accelerator_headroom_base
        ).minimum_free_bytes
    _bootstrap_aimdo(
        args.aimdo_init or aimdo_armed,
        simple_vram_headroom=simple_vram_headroom,
    )
    _prepare_accelerator_runtime(args.aimdo_init or aimdo_armed)
    os.environ.pop(_ACCELERATOR_BUDGETS_ENV, None)
    if vram_budgets:
        os.environ[_ACCELERATOR_BUDGETS_ENV] = ",".join(
            f"{index}={nbytes}" for index, nbytes in sorted(vram_budgets.items())
        )
    # Only validated argv, including its default, selects the worker policy.
    os.environ["DINKSTER_AIMDO_ARM"] = args.aimdo_arm
    # Pin the pack bootstrap's argv to the explicit parent-supplied ComfyUI
    # arguments. This happens after this host parsed its own CLI and applies
    # even to the empty tuple, so no ambient worker argument can leak into
    # ComfyUI's import-time parser.
    sys.argv[:] = [sys.argv[0], *args.comfy_args_json]
    if len(args.endpoint) == 1:
        asyncio.run(
            serve(
                args.endpoint[0],
                args.manifest[0],
                shm_threshold=args.shm_threshold,
                use_shm=not args.no_shm,
            )
        )
    else:
        asyncio.run(
            serve_many(
                args.endpoint,
                args.manifest,
                shm_threshold=args.shm_threshold,
                use_shm=not args.no_shm,
            )
        )


if __name__ == "__main__":
    main()
