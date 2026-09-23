"""The engine loop: plan, resolve, cache-check, invoke, store.

Location-agnostic by construction: the only way node code runs is
Worker.invoke (hazard H3), and cache keys derive from schema signature, input
fingerprints, and any structural region occurrence scope (hazard H4), so
entries are shareable across processes and machines.

Parallel by construction (hazard H12): the plan is a DAG and the scheduler
is ready-set - every node whose dependencies are satisfied dispatches
concurrently, bounded only by max_concurrency (a resource declaration, not
node code). run() is reentrant, so whole workflows execute in parallel as
plain concurrent run() calls; an engine-wide single-flight table coalesces
identical computations across nodes and runs. Any interleaving produces
identical results - concurrency changes wall-clock time and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Container, Iterable, Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal, cast

from dinkster_graph import (
    PORTS_NODE_ID,
    REGION_INDEX_PORT_ID,
    REGION_INDEX_PORT_TYPE,
    Diagnostic,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    TypedLiteral,
    dependency_map,
    elaborate_graph,
    has_errors,
    migrate_pure_node_type_replacements,
    region_interface,
    snapshot_graph,
    validate,
)
from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    AttentionPolicy,
    AttentionPolicyConfig,
    AttentionRouteToken,
    CacheStore,
    ExportSnapshot,
    ExtensionSnapshot,
    GraphCompilerRegistrySnapshot,
    Invocation,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    LazyStatusWorker,
    MediaSourceAuthority,
    NodeError,
    PreviewPolicy,
    SamplerRegistrySnapshot,
    SavedArtifact,
    Worker,
    canonical_attention_route_token_bytes,
    extension_behavior_hash,
    resolve_attention_runtime_status,
)
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    TypeExpr,
    combo_type_mismatch_is_error,
    plan_asset_coercion,
    plan_type_equivalence,
    schema_signature,
)
from dinkster_schema.media import media_diagnostics
from dinkster_values import (
    ABSENT_ORIGIN_META_KEY,
    ABSENT_REASON_META_KEY,
    ASSET_BASE_TYPE,
    LENGTH_META_KEY,
    RESOURCES_META_KEY,
    ResourcePins,
    TypeRegistry,
    Value,
    is_absent,
    iter_value_tree,
    list_children,
    make_absent_value,
    make_list_value,
    parse_list_type_id,
    stable_hash,
    value_resource_ids,
    value_resource_refs,
)

from .compile import (
    CompiledGraph,
    GraphCompileError,
    GraphCompileTransport,
    compile_graph,
)
from .events import EngineEvent, EventListener

logger = logging.getLogger(__name__)

ExecutionArm = Literal["native", "comfyui"]


class GraphValidationError(Exception):
    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        lines = [f"{d.severity}[{d.code}] {d.node_id or '-'}: {d.message}" for d in diagnostics]
        super().__init__("graph validation failed:\n" + "\n".join(lines))


class ProviderResolutionError(ValueError):
    """A graph node has no compatible implementation at admission."""

    def __init__(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str,
        capability: str,
        remedy: str,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.title = title
        self.capability = capability
        self.remedy = remedy
        super().__init__(
            f"{title} cannot run because {capability} is unavailable on this server. {remedy}"
        )


class ExecutionError(Exception):
    def __init__(self, error: NodeError) -> None:
        self.error = error
        super().__init__(f"node {error.node_id} ({error.node_type}) failed: {error.message}")


class ActiveRunIdError(Exception):
    """Raised when a caller reuses an id belonging to an active run."""


@dataclass(frozen=True)
class ExecutionSelection:
    """One node attempt's execution decision (stage 6 dispatch).

    ``target`` is the arm name the dispatching worker facade must invoke;
    ``cache_tag`` is the arm's execution identity - a stable string naming
    the executing implementation's identity (e.g. the native runtime
    identity including its runtime facts). These fields and the selected
    attention route are cache-key components (the binding is structural, not
    a naming convention), so different implementations or float-visible
    attention routes never share an entry, and rotating either identity rotates
    its keys.
    Ephemeral facts (process tokens, PIDs) must NOT ride either field - a
    worker restart is not a different computation. ``attention_diagnostic``
    is receipt evidence, not execution identity, and does not affect caching."""

    target: str
    cache_tag: str
    worker: str | None = None
    provider: str | None = None
    pack: str | None = None
    execution_arm: ExecutionArm = "native"
    fp8_matmul: bool = False
    diffusion_dtype: str | None = None
    text_dtype: str | None = None
    vae_dtype: str | None = None
    attention_policy: AttentionPolicy = "auto"
    attention_route_token: AttentionRouteToken | None = None
    attention_diagnostic: str | None = None

    def __post_init__(self) -> None:
        # Planning runs BEFORE cache lookup: a blank selection would key a
        # cache entry and only fail later (or never) at the dispatcher.
        # Malformed selections are wiring errors, refused at the source.
        if not self.target:
            raise ValueError("ExecutionSelection.target must be non-empty")
        if not self.cache_tag:
            raise ValueError("ExecutionSelection.cache_tag must be non-empty")
        if self.worker == "":
            raise ValueError("ExecutionSelection.worker must be non-empty when present")
        if self.provider == "":
            raise ValueError("ExecutionSelection.provider must be non-empty when present")
        if self.pack == "":
            raise ValueError("ExecutionSelection.pack must be non-empty when present")
        if self.execution_arm not in ("native", "comfyui"):
            raise ValueError("ExecutionSelection.execution_arm must be 'native' or 'comfyui'")
        if self.attention_diagnostic == "":
            raise ValueError("ExecutionSelection.attention_diagnostic must be a non-empty string")
        resolve_attention_runtime_status(self.attention_policy, self.attention_route_token)


PlanExecution = Callable[
    [str, str, NodeSchema, Mapping[str, Value], str, AttentionPolicyConfig | None],
    "Awaitable[ExecutionSelection | None]",
]
"""Engine hook: decide where one node attempt executes, BEFORE cache
lookup. Called with (node_id, node_type, effective schema, resolved inputs,
run_id, attention policy config) after input resolution and absent policies;
returns None for node types not enrolled in dispatch (the invocation proceeds
exactly as without the hook). Raises ExecutionError to fail the node loudly
(conflicting or dead resident owners). Called exactly once per attempt; the
returned selection feeds both the cache key and the invocation, so they cannot
diverge."""

ResolveProviders = Callable[[Graph], Graph]
"""Admission hook that binds implementation providers on a transient graph."""

PreflightProviderAssets = Callable[
    [Graph, frozenset[str], Mapping[str, Sequence[str]]],
    "Awaitable[tuple[frozenset[str], list[dict[str, object]]]]",
]
"""Admission hook returning selected remote digests and their unresolved plan."""


@dataclass(frozen=True)
class ExecutionRuntime:
    """Immutable worker topology and extension generation for one run.

    Hosts may atomically replace the current runtime while admitted runs keep
    this value. Only the digest crosses the worker boundary; worker/planner
    objects remain process-local implementation details.
    """

    worker: Worker
    plan_execution: PlanExecution | None = None
    resolve_providers: ResolveProviders | None = None
    preflight_provider_assets: PreflightProviderAssets | None = None
    run_finished: Callable[[str], None] | None = None
    owner_alive: Callable[[str], bool] | None = None
    extension_snapshot: ExtensionSnapshot = ExtensionSnapshot()
    sampler_registry_snapshot: SamplerRegistrySnapshot = SamplerRegistrySnapshot()
    graph_compiler_registry: GraphCompilerRegistrySnapshot = GraphCompilerRegistrySnapshot()
    graph_compile_transport: GraphCompileTransport | None = None
    schemas: Mapping[str, NodeSchema] | None = None
    prepare_host_types: Callable[[set[str]], Awaitable[None]] | None = None
    known_types: Container[str] | None = None
    placement_worker: Callable[[str], str | None] | None = None
    """Run-local worker attribution for placed nodes without a dispatch selection."""

    def __post_init__(self) -> None:
        if bool(self.graph_compiler_registry.contributions) != (
            self.graph_compile_transport is not None
        ):
            raise ValueError("graph compiler declarations and transport must be present together")

    @cached_property
    def extension_behavior_hash(self) -> str | None:
        """None is the compatibility-preserving no-extension sentinel."""
        if not self.extension_snapshot.extensions:
            return None
        return extension_behavior_hash(self.extension_snapshot)

    @cached_property
    def extension_snapshot_digest(self) -> str:
        behavior_hash = self.extension_behavior_hash
        if behavior_hash is None:
            behavior_hash = extension_behavior_hash(self.extension_snapshot)
        return "sha256:" + behavior_hash


PinExecution = Callable[[], ExecutionRuntime]
_execution_runtime: ContextVar[ExecutionRuntime | None] = ContextVar(
    "dinkster_engine_execution_runtime", default=None
)


@dataclass(frozen=True)
class RunResult:
    """What a run produced.

    The determinism invariant (hazard H12) is over `outputs`: same document,
    same values, under any schedule. `executed` and `cached` are completion-
    ordered observability data and legitimately vary with interleaving - e.g.
    which of two concurrent runs owns a coalesced computation (executed) and
    which consumes it (cached) is a race by design.
    """

    run_id: str
    outputs: Mapping[str, Mapping[str, Value]]  # target node_id -> output_id -> Value
    executed: tuple[str, ...]
    cached: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]
    skipped: tuple[str, ...] = ()
    """Nodes that did not execute because a skip-policy input arrived absent
    (DESIGN 3.15). An engine decision recorded here and in node_skipped
    events - never a value smuggled through node code. Completion-ordered
    like executed/cached."""
    artifacts: tuple[SavedArtifact, ...] = ()
    """Fresh output-node files imported during this run. Cache hits never
    replay artifacts because they do not repeat the filesystem side effect."""


def output_summary(
    outputs: Mapping[str, Value], registry: TypeRegistry | None = None
) -> dict[str, dict[str, object]]:
    """Per-output ``{typeId, length?, value?}`` carried on node_finished/
    node_cached event details, so frontends can render runtime type/element-
    count badges on intermediate edges mid-run instead of waiting for job
    completion.

    Cheap by construction: reads only the envelope's type id and its length
    meta (present exactly when the value is a list) - plus, given a
    registry, the inline scalar for types that declared one (an int/short
    string decode, never a rich payload; see TypeRegistry.inline_of).
    ``value`` omitted means "not inline", never absence - absence rides
    node_skipped and typed core.absent envelopes."""
    summary: dict[str, dict[str, object]] = {}
    for output_id, value in outputs.items():
        entry: dict[str, object] = {"typeId": value.type_id}
        media_meta = {
            key: value.meta.get(key)
            for key in ("shape", "dtype", "channels", "color", "polarity", "semantic")
            if value.meta.get(key) is not None
        }
        if media_meta:
            entry["meta"] = media_meta
        length = value.meta.get(LENGTH_META_KEY)
        if isinstance(length, int):
            entry["length"] = length
        if registry is not None:
            scalar = registry.inline_of(value)
            if scalar is not None:
                entry["value"] = scalar
        summary[output_id] = entry
    return summary


# Cache-miss explanation memory: one record per runtime node id (iteration
# ids included). Bounds the dev-mode bookkeeping, not any cache.
_KEY_MEMORY_CAP = 4096


class _ProducedReferences:
    """Keep outputs only while a possible consumer or DAG export needs them.

    Lazy edges count until their consumer finishes. Removing an unused branch
    also retires its ancestors, without executing them or dropping a shared
    ancestor still reachable by another consumer. Resource pins and cache
    residency have independent owners; this only releases Python references.
    """

    def __init__(
        self,
        deps: Mapping[str, set[str]],
        targets: Sequence[str],
        produced: dict[str, Mapping[str, Value]],
    ) -> None:
        self._deps = dict(deps)
        self._targets = set(targets)
        self._produced = produced
        self._uses = dict.fromkeys(deps, 0)
        for dependencies in deps.values():
            for dependency in dependencies:
                self._uses[dependency] += 1

    def finished(self, node_id: str) -> None:
        retired = [node_id]
        while retired:
            current = retired.pop()
            if self._uses[current] == 0 and current not in self._targets:
                self._produced.pop(current, None)
            for dependency in self._deps.pop(current, ()):
                self._uses[dependency] -= 1
                if self._uses[dependency] == 0 and dependency not in self._targets:
                    retired.append(dependency)


class Engine:
    def __init__(
        self,
        *,
        schemas: Mapping[str, NodeSchema],
        registry: TypeRegistry,
        worker: Worker,
        cache: CacheStore,
        on_event: EventListener | None = None,
        max_concurrency: int | None = None,
        resource_capacities: Mapping[str, int] | None = None,
        pins: ResourcePins | None = None,
        explain_misses: bool = False,
        plan_execution: PlanExecution | None = None,
        run_finished: Callable[[str], None] | None = None,
        owner_alive: Callable[[str], bool] | None = None,
        extension_snapshot: ExtensionSnapshot | None = None,
        pin_execution: PinExecution | None = None,
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        for kind, capacity in (resource_capacities or {}).items():
            if not kind:
                raise ValueError("resource kinds must be non-empty strings")
            if capacity < 1:
                raise ValueError(f"resource capacity for {kind!r} must be >= 1")
        self._schemas = dict(schemas)
        self._registry = registry
        self._worker = worker
        self._cache = cache
        self._on_event = on_event
        # Bounds concurrent worker invocations engine-wide (across runs).
        # One global knob for now; per-worker/per-device slots arrive with placement.
        self._invoke_slots = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        # Resource admission (hazard H12): schemas declare what a node
        # occupies (e.g. "gpu"); the engine owns how many may occupy each
        # lane at once. Undeclared lanes default to capacity 1 - the sane
        # default for a GPU (one occupying invocation at a time) and for the
        # implicit "compute" lane plain local nodes share (parallel node
        # execution limited by default). Configure {"gpu": 2} or
        # {"compute": 4} to widen deliberately. Engine-wide, so admission
        # holds across concurrent runs. Distinct from memory reservation: a
        # slot bounds simultaneous execution, VRAM budgets bound whether
        # allocations fit (DESIGN.md 3.10). With placement, abstract kinds
        # resolve to concrete devices ("cuda:0") and admission moves to the
        # machine that owns the hardware.
        self._resource_capacities = dict(resource_capacities or {})
        self._resource_slots: dict[str, asyncio.Semaphore] = {}
        self._lane_in_use: dict[str, int] = {}  # observability for resource_status()
        # Single-flight: cache key -> future of that computation's outputs.
        # Shared across runs, so two workflows needing the same computation
        # execute it once. Non-idempotent nodes have unique keys, so they are
        # never coalesced - by construction, not special-casing.
        self._inflight: dict[str, asyncio.Future[Mapping[str, Value]]] = {}
        # Resource pins (DESIGN 3.10): every resource-referencing envelope a run
        # routes is pinned for the run's lifetime, so a ram-lane release in
        # another process can see references that live only in this run's
        # variables. Share the same registry with the worker's ReleaseGuard;
        # None (the default) is correct when no release path exists. Node
        # authors never see any of this (hazard H2).
        self._pins = pins
        # Dev-mode cache-miss explanations (DESIGN 3.9): remember the key
        # COMPONENTS of each node's last computed key - schema signature +
        # per-input fingerprints - so a miss can name what changed ("missed
        # because input 'strength' fingerprint changed") instead of showing
        # two opaque hashes. Advisory observability, bounded LRU, never
        # part of cache identity (hazard H4).
        self._explain_misses = explain_misses
        self._key_memory: OrderedDict[
            str,
            tuple[
                str,
                ExecutionSelection | None,
                dict[str, str],
                tuple[str, ...] | None,
            ],
        ] = OrderedDict()
        # Execution dispatch (stage 6): plan_execution decides where an
        # enrolled node type executes BEFORE cache lookup, so the selection
        # (arm + cache tag) is part of the key and the invocation carries
        # the target.
        # owner_alive answers whether a resident envelope's owner token
        # (RESOURCE_OWNER_META_KEY) names a live worker lifetime - output
        # admission rejects cached entries whose owner died (the stub could
        # never resolve) and fails fresh results loudly (contract violation).
        # Both default to None: without them the engine behaves exactly as
        # before.
        self._plan_execution = plan_execution
        self._owner_alive = owner_alive
        self._base_runtime = ExecutionRuntime(
            worker=worker,
            plan_execution=plan_execution,
            run_finished=run_finished,
            owner_alive=owner_alive,
            extension_snapshot=extension_snapshot or ExtensionSnapshot(),
        )
        self._pin_execution = pin_execution
        # Run-scoped schema captures (hazard H10): each run pins the schema
        # mapping it entered with, so replace_schemas (hot reload) can swap
        # self._schemas without a running job ever observing the change -
        # "jobs execute compiled snapshots" stays true by construction, not
        # by hoping no lookup happens mid-swap.
        self._run_schemas: dict[str, Mapping[str, NodeSchema]] = {}
        self._run_media_sources: dict[str, tuple[MediaSourceAuthority, ...]] = {}
        self._run_attempts: dict[str, int] = {}
        self._run_preview_policy: dict[str, PreviewPolicy] = {}
        self._run_attention_config: dict[str, AttentionPolicyConfig] = {}
        self._run_artifacts: dict[str, list[SavedArtifact]] = {}
        self._run_prepared_types: dict[str, set[str]] = {}
        self._run_prepare_locks: dict[str, asyncio.Lock] = {}
        self._compiling_run_ids: set[str] = set()

    def announce_schemas(self, schemas: Mapping[str, NodeSchema]) -> None:
        """Additively grow the node surface (progressive announcement;
        hot-reload re-announces through the same seam).

        Strictly additive: redefining an existing node type is refused -
        composition already guarantees type uniqueness, so a collision here
        is host miswiring, never something to resolve silently. The merged
        mapping replaces ``self._schemas`` in one reference swap, so runs
        in flight (which only ever look up types that existed when they
        were submitted - announcement never removes) see either the old or
        the new mapping, both consistent, never a half-mutated dict.
        """
        for node_type in schemas:
            if node_type in self._schemas:
                raise ValueError(
                    f"announce_schemas would redefine node type {node_type!r}; "
                    "the surface grows additively, never in place"
                )
        self._schemas = {**self._schemas, **schemas}

    def replace_schemas(self, remove: Sequence[str], add: Mapping[str, NodeSchema]) -> None:
        """Atomically swap one pack's slice of the node surface (hot
        reload, DESIGN 3.9). The removed types and the re-announced types
        land in ONE reference swap; runs in flight are untouched because
        every run pinned the mapping it entered with (see _run_schemas) -
        a reload can never make a running job's lookups drift mid-run.

        Validation mirrors announce_schemas: every removed type must
        exist, and an added type may redefine only a type being removed in
        the same swap (the reloaded pack's own). Removal without re-add is
        legal - a reloaded pack may drop node types; new runs naming a
        dropped type fail graph validation, exactly like any unknown type.
        """
        removed = set(remove)
        for node_type in removed:
            if node_type not in self._schemas:
                raise ValueError(f"replace_schemas would remove unknown node type {node_type!r}")
        for node_type in add:
            if node_type in self._schemas and node_type not in removed:
                raise ValueError(
                    f"replace_schemas would redefine node type {node_type!r}, which "
                    "is not being removed in this swap"
                )
        merged = {t: s for t, s in self._schemas.items() if t not in removed}
        merged.update(add)
        self._schemas = merged

    def _schemas_for(self, run_id: str) -> Mapping[str, NodeSchema]:
        """The schema mapping a run pinned at entry (hazard H10); the live
        mapping only for lookups outside any run."""
        return self._run_schemas.get(run_id, self._schemas)

    def pin_execution(self) -> ExecutionRuntime:
        """Capture the current immutable execution generation at admission."""
        runtime = self._pin_execution() if self._pin_execution is not None else self._base_runtime
        return (
            runtime
            if runtime.schemas is not None
            else replace(runtime, schemas=dict(self._schemas))
        )

    def _normalize_execution(self, runtime: ExecutionRuntime) -> ExecutionRuntime:
        return runtime if runtime.schemas else replace(runtime, schemas=dict(self._schemas))

    def _runtime(self) -> ExecutionRuntime:
        return _execution_runtime.get() or self.pin_execution()

    async def _prepare_node_types(self, run_id: str, node_types: Sequence[str]) -> None:
        async with self._run_prepare_locks[run_id]:
            prepared = self._run_prepared_types[run_id]
            missing = sorted(set(node_types) - prepared)
            if missing:
                await self._runtime().worker.prepare(missing)
                prepared.update(missing)

    def _emit(self, event: EngineEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.exception("engine event listener failed for %s", event.kind)

    def _emit_media_diagnostics(
        self,
        run_id: str,
        node_id: str,
        schema: NodeSchema,
        inputs: Mapping[str, Value],
        outputs: Mapping[str, Value],
    ) -> None:
        recorded = next(
            (
                value.meta.get("valueDiagnostics")
                for value in outputs.values()
                if value.meta.get("valueDiagnostics") is not None
            ),
            None,
        )
        diagnostics = (
            [
                deepcopy(dict(cast("Mapping[str, object]", item)))
                for item in cast("list[object]", recorded)
                if isinstance(item, Mapping)
            ]
            if isinstance(recorded, list)
            else media_diagnostics(schema, inputs, outputs)
        )
        if diagnostics:
            self._emit(
                EngineEvent(
                    "value_diagnostics",
                    run_id,
                    node_id,
                    {
                        "diagnostics": [
                            {**deepcopy(item), "nodeId": node_id} for item in diagnostics
                        ]
                    },
                )
            )

    def _emit_node_event(
        self,
        run_id: str,
        node_id: str,
        event: InvocationEvent,
        selection: ExecutionSelection | None = None,
    ) -> None:
        """A running node reported (progress, preview, pack event): surface
        it as a node_event with run/node provenance the worker cannot know.
        The blob rides in detail as bytes - listeners that serialize to JSON
        must lift it out (the server's binary frame path does)."""
        detail: dict[str, object] = {"name": event.name, "data": dict(event.data)}
        if event.blob is not None:
            detail["blob"] = event.blob
        detail = self._execution_detail(node_id, selection, **detail)
        if "worker" not in detail:
            del detail["executionArm"]
        runtime = self._runtime()
        declared = next(
            (
                (extension, item)
                for extension in runtime.extension_snapshot.extensions
                for item in extension.events
                if item.name == event.name
            ),
            None,
        )
        if declared is not None:
            extension, contract = declared
            if detail.get("pack") != extension.id or event.blob is not None:
                return
            try:
                contract.payload.validate(event.data)
            except ValueError:
                return
            detail["schemaVersion"] = 1
            detail["extensionSnapshotDigest"] = runtime.extension_snapshot_digest
        self._emit(
            EngineEvent(
                "node_event",
                run_id,
                node_id,
                detail,
            )
        )

    def _execution_detail(
        self,
        node_id: str,
        selection: ExecutionSelection | None,
        **detail: object,
    ) -> dict[str, object]:
        detail["executionArm"] = selection.execution_arm if selection is not None else "native"
        worker = selection.worker if selection is not None else None
        placement_worker = self._runtime().placement_worker
        if worker is None and placement_worker is not None:
            worker = placement_worker(node_id)
        if worker is not None:
            detail["worker"] = worker
        if selection is not None and selection.provider is not None:
            detail["provider"] = selection.provider
        if selection is not None and selection.pack is not None:
            detail["pack"] = selection.pack
        if selection is not None and selection.attention_diagnostic is not None:
            detail["attentionDiagnostic"] = selection.attention_diagnostic
        return detail

    def _decrement_lane(self, lane: str) -> None:
        self._lane_in_use[lane] -= 1

    def _resource_slot(self, lane: str) -> asyncio.Semaphore:
        slot = self._resource_slots.get(lane)
        if slot is None:
            capacity = self._resource_capacities.get(lane)
            if capacity is None:
                # A bound lane ("gpu:cuda:1") inherits its kind's capacity.
                capacity = self._resource_capacities.get(lane.split(":", 1)[0], 1)
            slot = asyncio.Semaphore(capacity)
            self._resource_slots[lane] = slot
        return slot

    @staticmethod
    def _occupancy(
        schema: NodeSchema,
        inputs: Mapping[str, Value],
        selection: ExecutionSelection | None,
    ) -> tuple[str, ...]:
        """The admission lanes this invocation occupies.

        Three default admission classes, all declaration-driven (hazard H12):

        - **Hardware-occupying** (occupies declared): which concrete instance
          a node occupies is a fact of its *inputs*, never of the node type -
          a loaded model knows which device(s) it lives on and declares them
          in its envelope's resources meta ({"gpu": "cuda:1"}, or a list for
          a model spanning devices). Each occupied kind binds to every
          instance declared by any input - a spanning model or two models on
          different devices occupy every lane involved - falling back to the
          abstract kind when no input declares one.
        - **io_bound**: waits, doesn't compute (partner/API nodes); occupies
          nothing and overlaps freely under max_concurrency.
        - **Plain local nodes**: share the implicit "compute" lane when local.
          Dispatched nodes bind it to the selected execution arm so distinct
          workers overlap while each worker retains its own capacity.
        """
        if schema.io_bound:
            return ()
        if not schema.occupies:
            return (f"compute:{selection.target}",) if selection is not None else ("compute",)
        lanes: set[str] = set()
        for kind in schema.occupies:
            bound = False
            # Tree traversal: a resource-owning value inside a list binds
            # lanes exactly like a top-level one (DESIGN 3.13).
            for top in inputs.values():
                for value in iter_value_tree(top):
                    resources = value.meta.get(RESOURCES_META_KEY)
                    if not isinstance(resources, Mapping):
                        continue
                    # Residency meta is runtime data from type registrations:
                    # a device string or a sequence of them (multigpu).
                    instance = cast("Mapping[str, object]", resources).get(kind)
                    if instance is None:
                        continue
                    instances = (
                        [instance]
                        if isinstance(instance, str)
                        else list(cast("Iterable[str]", instance))
                    )
                    for one in instances:
                        lanes.add(f"{kind}:{one}")
                        bound = True
            if not bound:
                lanes.add(kind)
        return tuple(sorted(lanes))

    @contextlib.asynccontextmanager
    async def _admission(
        self,
        schema: NodeSchema,
        inputs: Mapping[str, Value],
        selection: ExecutionSelection | None,
    ):
        """Hold one permit per occupied admission lane plus the global
        invocation slot for the full invocation. Everyone acquires in the
        same total order - lanes sorted, global slot last - so multi-lane
        nodes can never deadlock, and a node blocked on a scarce permit is
        not sitting on a global slot that a runnable node needs. The exit
        stack releases in reverse and unwinds partial acquisition on
        cancellation. Only reached by the single-flight owner on a cache
        miss - waiters and cache hits never consume permits."""
        async with contextlib.AsyncExitStack() as stack:
            for lane in self._occupancy(schema, inputs, selection):
                await stack.enter_async_context(self._resource_slot(lane))
                self._lane_in_use[lane] = self._lane_in_use.get(lane, 0) + 1
                stack.callback(self._decrement_lane, lane)
            if self._invoke_slots is not None:
                await stack.enter_async_context(self._invoke_slots)
            yield

    def _remember_key_components(
        self,
        label: str,
        schema: NodeSchema,
        signature: str,
        inputs: Mapping[str, Value],
        selection: ExecutionSelection | None,
        connected_undemanded_inputs: tuple[str, ...] | None,
    ) -> (
        tuple[
            str,
            ExecutionSelection | None,
            dict[str, str],
            tuple[str, ...] | None,
        ]
        | None
    ):
        """Dev-mode bookkeeping for cache-miss explanations: return this
        label's PREVIOUS key components and record the current ones (the
        next run compares against the latest computation, hit or miss).
        Non-idempotent schemas are not recorded - their keys are unique by
        construction, so components can never explain anything."""
        previous = self._key_memory.get(label)
        if previous is not None:
            self._key_memory.move_to_end(label)
        if schema.idempotent:
            fingerprints = {i: value.fingerprint for i, value in inputs.items()}
            self._key_memory[label] = (
                signature,
                selection,
                fingerprints,
                connected_undemanded_inputs,
            )
            while len(self._key_memory) > _KEY_MEMORY_CAP:
                self._key_memory.popitem(last=False)
        return previous

    @staticmethod
    def _miss_detail(
        key: str,
        schema: NodeSchema,
        signature: str,
        inputs: Mapping[str, Value],
        selection: ExecutionSelection | None,
        connected_undemanded_inputs: tuple[str, ...] | None,
        previous: tuple[
            str,
            ExecutionSelection | None,
            dict[str, str],
            tuple[str, ...] | None,
        ]
        | None,
    ) -> dict[str, object]:
        """Why did the store have nothing for this key? Named from the key's
        own composite parts (DESIGN 3.9): 'missed because input X
        fingerprint changed', not two opaque hashes."""
        detail: dict[str, object] = {"cache_key": key}
        if connected_undemanded_inputs is not None:
            detail["lazy_state"] = {
                "connected_undemanded_inputs": list(connected_undemanded_inputs)
            }
        if not schema.idempotent:
            detail["reason"] = "never-cacheable"
            return detail
        if previous is None:
            detail["reason"] = "first-seen"
            return detail
        prev_signature, prev_selection, prev_fingerprints, prev_lazy_state = previous
        changed = sorted(
            input_id
            for input_id, value in inputs.items()
            if input_id in prev_fingerprints and prev_fingerprints[input_id] != value.fingerprint
        )
        added = sorted(set(inputs) - set(prev_fingerprints))
        removed = sorted(set(prev_fingerprints) - set(inputs))
        if changed:
            detail["changed_inputs"] = changed
        if added:
            detail["added_inputs"] = added
        if removed:
            detail["removed_inputs"] = removed
        if prev_signature != signature:
            detail["reason"] = "schema-changed"
        elif prev_selection != selection:
            # A deliberate execution-identity rotation (dispatch selected a
            # different arm, or an implementation's identity changed) - name
            # it instead of misreporting "evicted".
            detail["reason"] = "executor-changed"
        elif prev_lazy_state != connected_undemanded_inputs:
            detail["reason"] = "lazy-state-changed"
            detail["previous_lazy_state"] = (
                None
                if prev_lazy_state is None
                else {"connected_undemanded_inputs": list(prev_lazy_state)}
            )
        elif changed or added or removed:
            detail["reason"] = "inputs-changed"
        else:
            # Identical components produce an identical key, and that key
            # was computed (and its result stored) before: the entry is
            # gone from the store, not the computation changed.
            detail["reason"] = "evicted"
        return detail

    def _cache_key(
        self,
        schema: NodeSchema,
        signature: str,
        inputs: Mapping[str, Value],
        selection: ExecutionSelection | None = None,
        connected_undemanded_inputs: tuple[str, ...] | None = None,
        cache_scope: str | None = None,
    ) -> str:
        parts: list[bytes] = [signature.encode("utf-8")]
        if cache_scope is not None:
            parts.extend((b"region-occurrence", cache_scope.encode("utf-8")))
        behavior_hash = self._runtime().extension_behavior_hash
        if behavior_hash is not None:
            parts.append(b"extensions")
            parts.append(behavior_hash.encode("ascii"))
        if selection is not None:
            # The selected implementation is part of the computation's
            # identity (stage 6 dispatch): the same inputs through compat
            # and native are different computations, and an implementation
            # rotating its execution identity rotates its keys. BOTH the arm
            # name and its cache tag are separately framed components - two
            # arms sharing a tag must still never share an entry, and the
            # binding is structural, not a docstring convention. Framed with
            # a marker so an unenrolled node's key never collides with an
            # enrolled one's.
            parts.append(b"executor")
            parts.append(selection.target.encode("utf-8"))
            parts.append(selection.cache_tag.encode("utf-8"))
            if selection.attention_route_token is not None:
                parts.append(b"attention-route-token")
                parts.append(canonical_attention_route_token_bytes(selection.attention_route_token))
        if connected_undemanded_inputs is not None:
            # A lazy consumer's identity is finalized only after its bounded
            # hook reaches a fixed demand set. Values that are visible then
            # are framed below like ordinary inputs. Connected lazy sockets
            # that remain undemanded have no Value, so frame their ordered
            # schema ids explicitly: this distinguishes their worker-visible
            # None from omission/default/unconnected shapes without hashing a
            # producer value the hook deliberately never demanded.
            parts.append(b"connected-undemanded-lazy-inputs")
            parts.append(len(connected_undemanded_inputs).to_bytes(8, "big"))
            parts.extend(input_id.encode("utf-8") for input_id in connected_undemanded_inputs)
        for input_id in sorted(inputs):
            value = inputs[input_id]
            parts.append(input_id.encode("utf-8"))
            # The stamp is part of the input's identity, not just the bytes:
            # the same AssetRef under asset<A> vs asset<B> shares a content
            # fingerprint but solves type variables (and therefore output
            # stamps) differently, so it must never share a cache entry.
            parts.append(value.type_id.encode("utf-8"))
            parts.append(value.fingerprint.encode("utf-8"))
        if not schema.idempotent:
            # Never cacheable: a unique component makes every key distinct.
            parts.append(uuid.uuid4().bytes)
        return stable_hash(parts)

    def _admit_input_value(
        self, node_id: str, node_type: str, spec: InputSpec, value: Value
    ) -> Value:
        """Apply an input's registered type bridge or asset coercion identity.

        Equivalent atoms are restamped before both cache identity and worker
        transport, so the destination receives only the spelling declared by
        its interface. Asset coercions restamp the fingerprint with their
        provider identity (typed assets, joint contract 2026-07-26: coerced
        cache identity = source digest(s) + decoder identity + merger
        identity; decoded tensors are never fingerprinted).

        Asset payloads stay encoded: the WORKER's shim applies the same plan
        with its own providers at kwargs time. A plan whose providers this
        registry lacks fails before invocation, mirroring the worker's own
        refusal."""
        if is_absent(value) or spec.type.accepts_concrete(value.type_id):
            return value
        equivalent = plan_type_equivalence(value.type_id, spec.type, self._registry)
        if equivalent is not None:
            return self._registry.bridge_equivalent(value, equivalent)
        plan = plan_asset_coercion(value.type_id, spec.type)
        if plan is None:
            if combo_type_mismatch_is_error(value.type_id, spec.type):
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"input '{spec.id}' rejects runtime type "
                        f"{value.type_id}: core.combo mismatches require an "
                        "explicit converter",
                    )
                )
            return value
        decoder = self._registry.asset_decoder_for(plan.target_type_id)
        merger = (
            self._registry.batch_merge_for(plan.merge_type_id)
            if plan.merge_type_id is not None
            else None
        )
        if decoder is None or (plan.merge_type_id is not None and merger is None):
            raise ExecutionError(
                NodeError(
                    node_id,
                    node_type,
                    f"input '{spec.id}' carries {value.type_id} and needs "
                    + " and ".join(plan.missing_providers(self._registry))
                    + ", which is not registered",
                )
            )
        parts = [
            b"asset-coerce",
            plan.kind.encode("utf-8"),
            decoder.provider_id.encode("utf-8"),
        ]
        if merger is not None:
            parts.append(merger.provider_id.encode("utf-8"))
        parts.append(value.fingerprint.encode("utf-8"))
        return replace(value, fingerprint=stable_hash(parts))

    def _resolve_inputs(
        self,
        graph: Graph,
        node_id: str,
        schema: NodeSchema,
        produced: Mapping[str, Mapping[str, Value]],
    ) -> dict[str, Value]:
        node = graph.nodes[node_id]
        assert isinstance(node, GraphNode), f"region {node_id!r} in _resolve_inputs"
        resolved: dict[str, Value] = {}
        for spec in schema.inputs:
            if spec.id in node.inputs:
                raw = node.inputs[spec.id]
            elif spec.default is not None or not spec.required:
                if spec.default is None:
                    continue
                raw = spec.default
            else:  # pragma: no cover - validation rejects this earlier
                raise KeyError(f"{node_id}: missing input {spec.id}")
            if isinstance(raw, Link):
                resolved[spec.id] = self._admit_input_value(
                    node_id,
                    node.node_type,
                    spec,
                    produced[raw.node_id][raw.output_id],
                )
            elif isinstance(raw, TypedLiteral):
                # A typed literal carries its own runtime type id (validated:
                # syntax, registration, shape), so it wraps with the stamp
                # instead of the destination type - the wildcard/union input
                # case where the destination has no runtime type to offer.
                resolved[spec.id] = self._admit_input_value(
                    node_id,
                    node.node_type,
                    spec,
                    self._registry.wrap(raw.type_id, raw.value),
                )
            else:
                # Literals are wrapped by the engine using the declared
                # runtime-resolvable input type - concrete or list-of-concrete
                # (validated); node authors and frontends send plain values.
                runtime = spec.type.runtime_type_id()
                if runtime is None:  # pragma: no cover - validation rejects this
                    raise ValueError(
                        f"{node_id}: literal on non-runtime-resolvable input {spec.id}"
                    )
                resolved[spec.id] = self._registry.wrap(runtime, raw)
        return resolved

    def _apply_absent_policies(
        self,
        node_id: str,
        node_type: str,
        schema: NodeSchema,
        inputs: dict[str, Value],
    ) -> tuple[str, str, str] | None:
        """Apply each input's absent policy (DESIGN 3.15) before invocation.

        Mutates ``inputs``: omit-policy absents become exactly the unconnected
        shape - the wrapped schema default when one exists, otherwise no
        kwarg - so the cache key matches a document where the input was never
        linked. Accept-policy absents stay (the worker resolves them to plain
        None). Returns (input_id, origin, reason) when a skip-policy absent
        means the node must not execute. Raises ExecutionError for fail-policy
        absents - fail always beats skip, loud beats silent. Node code never
        sees an absent envelope it did not opt into."""
        skip: tuple[str, str, str] | None = None
        for spec in schema.inputs:
            value = inputs.get(spec.id)
            if value is None or not is_absent(value):
                continue
            origin = str(value.meta.get(ABSENT_ORIGIN_META_KEY, ""))
            reason = str(value.meta.get(ABSENT_REASON_META_KEY, ""))
            policy = spec.absent_policy()
            if policy == "fail":
                detail = f" ({reason})" if reason else ""
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"input '{spec.id}' is absent - no value was produced "
                        f"by {origin or 'its producer'}{detail} - and this "
                        "input declares on_absent='fail'",
                    )
                )
            if policy == "omit":
                if spec.default is not None:
                    runtime = spec.type.runtime_type_id()
                    if runtime is None:  # pragma: no cover - validation rejects
                        raise ValueError(
                            f"{node_id}: default on non-runtime-resolvable input {spec.id}"
                        )
                    inputs[spec.id] = self._registry.wrap(runtime, spec.default)
                else:
                    del inputs[spec.id]
            elif policy == "skip" and skip is None:
                skip = (spec.id, origin, reason)
        return skip

    async def _prepare_host_types(
        self, expressions: Iterable[TypeExpr], values: Iterable[object]
    ) -> None:
        prepare = self._runtime().prepare_host_types
        if prepare is None:
            return
        atoms: set[str] = set()
        pending = list(expressions)
        while pending:
            expression = pending.pop()
            atoms.update(expression.types)
            if expression.kind == "asset":
                atoms.add(ASSET_BASE_TYPE)
            if expression.element is not None:
                pending.append(expression.element)
        for value in values:
            if isinstance(value, (TypedLiteral, Value)):
                if (atom := TypeExpr.runtime_type_atom(value.type_id)) is not None:
                    atoms.add(atom)
                if "asset<" in value.type_id:
                    atoms.add(ASSET_BASE_TYPE)
        await prepare(atoms)

    async def _resolve_inputs_lazy(
        self,
        graph: Graph,
        node_id: str,
        schema: NodeSchema,
        produced: Mapping[str, Mapping[str, Value]],
        demanded: set[str],
    ) -> dict[str, Value]:
        node = cast("GraphNode", graph.nodes[node_id])
        hidden = {
            spec.id
            for spec in schema.inputs
            if spec.lazy and isinstance(node.inputs.get(spec.id), Link) and spec.id not in demanded
        }
        visible = replace(
            schema,
            inputs=tuple(spec for spec in schema.inputs if spec.id not in hidden),
            selector=None,
        )
        values = (node.inputs.get(spec.id, spec.default) for spec in visible.inputs)
        await self._prepare_host_types(
            (spec.type for spec in (*visible.inputs, *visible.outputs)),
            (
                produced[value.node_id][value.output_id] if isinstance(value, Link) else value
                for value in values
            ),
        )
        return self._resolve_inputs(graph, node_id, visible, produced)

    @staticmethod
    def _validate_lazy_requests(
        label: str, node: GraphNode, schema: NodeSchema, raw: Sequence[object]
    ) -> list[str]:
        by_id = {spec.id: spec for spec in schema.inputs}
        for item in raw:
            if not isinstance(item, str):
                code, cause = "lazy-request-invalid", f"non-string value {item!r}"
            elif item not in by_id:
                code, cause = "lazy-request-unknown-input", f"unknown input {item!r}"
            elif not by_id[item].lazy:
                code, cause = "lazy-request-non-lazy", f"non-lazy input {item!r}"
            elif item not in node.inputs:
                code, cause = "lazy-request-unconnected", f"unconnected input {item!r}"
            elif not isinstance(node.inputs[item], Link):
                code, cause = "lazy-request-literal", f"literal input {item!r}"
            else:
                continue
            raise ExecutionError(
                NodeError(
                    label,
                    node.node_type,
                    f"{code}: lazy input hook requested {cause}",
                )
            )
        names = set(cast("Sequence[str]", raw))
        return [spec.id for spec in schema.inputs if spec.id in names]

    @staticmethod
    def _cache_hit_valid(schema: NodeSchema, hit: Mapping[str, Value]) -> bool:
        """Cache hits honor the same output contract as worker results: exact
        elaborated output ids, and concrete declared types match the stored
        Value's type_id. An invalid entry (corrupt/foreign cache) is treated
        as a miss and overwritten, never handed downstream."""
        if set(hit) != {out.id for out in schema.outputs}:
            return False
        for out in schema.outputs:
            expected = out.type.runtime_type_id()
            if expected is not None and hit[out.id].type_id != expected:
                # A deliberate absence is a valid cached value for an output
                # declared optional (DESIGN 3.15) - it replays exactly like
                # the value it stands in for.
                if out.optional and is_absent(hit[out.id]):
                    continue
                return False
        return True

    def _pin_outputs(self, outputs: Mapping[str, Value], pinned: list[str]) -> bool:
        """Pin every resource reference in one node's outputs for the run's
        lifetime (recorded in ``pinned``; run() unpins at the end). False
        means a referenced resource is condemned - released, or mid-release
        - so the outputs must not be used: the caller treats them as a miss
        and recomputes through the ordinary path. Pins taken before the
        refusal was noticed stay in ``pinned`` and unwind at run end like
        any other."""
        if self._pins is None:
            return True
        live = True
        for value in outputs.values():
            # Tree traversal: a resource stub inside a list output is pinned
            # exactly like a top-level one (DESIGN 3.13).
            for rid in value_resource_ids(value):
                if self._pins.pin(rid):
                    pinned.append(rid)
                else:
                    live = False
        return live

    async def _plan(
        self,
        run_id: str,
        label: str,
        node_type: str,
        schema: NodeSchema,
        inputs: Mapping[str, Value],
    ) -> ExecutionSelection | None:
        """Run the plan_execution hook (stage 6 dispatch) for one attempt.
        None (no hook, or unenrolled type) leaves the attempt exactly as
        without dispatch. Hook failures fail the node loudly - a node whose
        execution cannot be decided must not fall through to a guess."""
        planner = self._runtime().plan_execution
        if planner is None:
            return None
        try:
            return await planner(
                label,
                node_type,
                schema,
                inputs,
                run_id,
                self._run_attention_config.get(run_id),
            )
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                NodeError(label, node_type, f"execution planning failed: {exc}")
            ) from exc

    def _dead_owner(self, outputs: Mapping[str, Value]) -> tuple[str, str] | None:
        """The first (resource id, owner token) in ``outputs`` whose owner
        lifetime ended, or None when every stamped owner is live (or no
        owner_alive predicate is configured). Unstamped references are not
        judged here - their liveness is the pin/release layers' concern.
        Companion to _pin_outputs at the same admission seam: a reference
        to a dead worker lifetime can never resolve, so cached entries
        carrying one are misses and fresh results carrying one are
        contract violations."""
        owner_alive = self._runtime().owner_alive
        if owner_alive is None:
            return None
        for value in outputs.values():
            for rid, owner in value_resource_refs(value):
                if owner is not None and not owner_alive(owner):
                    return (rid, owner)
        return None

    async def _run_node(
        self,
        run_id: str,
        graph: Graph,
        node_id: str,
        effective: Mapping[str, NodeSchema],
        signatures: Mapping[str, str],
        produced: dict[str, Mapping[str, Value]],
        executed: list[str],
        cached: list[str],
        skipped: list[str],
        pinned: list[str],
        export_snapshot: ExportSnapshot | None,
        prefix: str = "",
        ensure_node: Callable[[str], Awaitable[None]] | None = None,
        cache_enabled: bool = True,
        cache_scope: str | None = None,
    ) -> None:
        """Produce one node's outputs: coalesce, hit cache, or invoke.

        Single-flight discipline: whoever registers the inflight future for a
        key is the owner and computes; everyone else awaits that future. The
        future resolves with the outputs, fails with the owner's error (so
        coalesced runs fail identically), or is cancelled when the owner's
        run is torn down - in which case a waiter takes over ownership and
        recomputes. Ownership is registered before the first await, so two
        tasks can never both own a key.

        ``prefix`` namespaces the node id in events, bookkeeping lists, and
        errors when this node runs inside a region iteration
        (``region[3]/node``). Nodes in a nested region also carry their stable
        parent occurrence as cache scope, matching explicit nested-loop
        ancestry while still coalescing identical work within that occurrence
        and across runs.
        """
        node = graph.nodes[node_id]
        assert isinstance(node, GraphNode), f"region {node_id!r} dispatched to _run_node"
        label = prefix + node_id
        schema = effective[node_id]
        lazy_inputs = tuple(spec for spec in schema.inputs if spec.lazy)
        lazy_links = {
            spec.id: cast("Link", node.inputs[spec.id])
            for spec in lazy_inputs
            if isinstance(node.inputs.get(spec.id), Link)
        }

        def refuse_lazy(message: str) -> None:
            error = NodeError(label, node.node_type, message)
            self._emit(EngineEvent("node_failed", run_id, label, {"message": error.message}))
            raise ExecutionError(error)

        demanded: set[str] = set()
        inputs: dict[str, Value]
        skip: tuple[str, str, str] | None
        if lazy_inputs:
            base_schema = self._schemas_for(run_id)[node.node_type]
            static_inputs = {spec.id for spec in base_schema.inputs}
            for spec in lazy_inputs:
                if spec.id in static_inputs:
                    continue
                family = base_schema.family_of_input(spec.id)
                if (
                    family is None
                    or len(family.template) != 1
                    or not isinstance(family.template[0], InputSpec)
                ):
                    refuse_lazy("lazy-dynamic-unsupported")
            round_number = 0
            while True:
                round_number += 1
                inputs = await self._resolve_inputs_lazy(graph, node_id, schema, produced, demanded)
                skip = self._apply_absent_policies(label, node.node_type, schema, inputs)
                if skip is not None:
                    break
                if schema.selector is not None:
                    selector = schema.selector
                    selected = inputs[selector.input].resolve()
                    if type(selected) is not bool:
                        result = LazyStatusResult(
                            error=NodeError(
                                label,
                                node.node_type,
                                "selector input did not resolve to a boolean",
                            )
                        )
                    else:
                        branch = selector.branches["true" if selected else "false"]
                        result = LazyStatusResult(
                            requested_inputs=() if branch in inputs else (branch,)
                        )
                else:
                    runtime_worker = self._runtime().worker
                    lazy_status = getattr(runtime_worker, "check_lazy_status", None)
                    if not callable(lazy_status):
                        result = LazyStatusResult(
                            error=NodeError(
                                label,
                                node.node_type,
                                "lazy-protocol-skew: worker does not support lazy status",
                            )
                        )
                    else:
                        result = await cast("LazyStatusWorker", runtime_worker).check_lazy_status(
                            LazyStatusInvocation(
                                request_id=uuid.uuid4().hex,
                                node_id=label,
                                node_type=node.node_type,
                                available_inputs=inputs,
                                connected_undemanded_inputs=tuple(
                                    spec.id
                                    for spec in schema.inputs
                                    if spec.id in lazy_links and spec.id not in demanded
                                ),
                                effective_schema=schema,
                                extension_snapshot_digest=(
                                    self._runtime().extension_snapshot_digest
                                    if self._runtime().extension_behavior_hash is not None
                                    else None
                                ),
                            )
                        )
                if result.error is not None:
                    error = NodeError(
                        label,
                        node.node_type,
                        result.error.message,
                        traceback=result.error.traceback,
                        hints=result.error.hints,
                    )
                    self._emit(
                        EngineEvent(
                            "node_failed",
                            run_id,
                            label,
                            {"message": error.message},
                        )
                    )
                    raise ExecutionError(error)
                assert result.requested_inputs is not None
                try:
                    requested = self._validate_lazy_requests(
                        label, node, schema, result.requested_inputs
                    )
                except ExecutionError as exc:
                    self._emit(
                        EngineEvent(
                            "node_failed",
                            run_id,
                            label,
                            {"message": exc.error.message},
                        )
                    )
                    raise
                new = [name for name in requested if name not in demanded]
                demanded.update(new)
                producers = sorted({lazy_links[name].node_id for name in new})
                self._emit_node_event(
                    run_id,
                    label,
                    InvocationEvent(
                        "lazy_demand",
                        {
                            "round": round_number,
                            "status": "waiting" if new else "ready",
                            "requestedInputs": requested,
                            "newInputs": new,
                            "demandedInputs": [
                                spec.id for spec in schema.inputs if spec.id in demanded
                            ],
                            "producerNodes": producers,
                        },
                    ),
                )
                if not new:
                    break
                assert ensure_node is not None
                await asyncio.gather(
                    *(ensure_node(producer) for producer in producers if producer not in produced)
                )
        else:
            inputs = await self._resolve_inputs_lazy(graph, node_id, schema, produced, demanded)
            skip = self._apply_absent_policies(label, node.node_type, schema, inputs)
        if skip is not None:
            # Skips are pure graph logic: no invocation, no cache entry (they
            # cost nothing to rederive), no pins (absents carry no resources).
            # Outputs become absent with the ROOT origin propagated unchanged,
            # so diagnostics always name the producer that decided no value
            # exists - never this node, which is a bystander (DESIGN 3.15).
            input_id, origin, reason = skip
            produced[node_id] = {
                out.id: make_absent_value(
                    origin=origin,
                    reason=reason,
                    stands_for=out.type.runtime_type_id(),
                )
                for out in schema.outputs
            }
            skipped.append(label)
            self._emit(
                EngineEvent(
                    "node_skipped",
                    run_id,
                    label,
                    {"input": input_id, "origin": origin, "reason": reason},
                )
            )
            return
        # Selectors are engine-owned graph control and never enter a worker
        # residency domain. Other nodes select an implementation before the
        # cache lookup so that identity participates in the key.
        selection = (
            None
            if schema.selector is not None
            else await self._plan(run_id, label, node.node_type, schema, inputs)
        )
        lazy_consumer = bool(lazy_inputs)
        connected_undemanded_inputs = (
            tuple(
                spec.id
                for spec in schema.inputs
                if spec.id in lazy_links and spec.id not in demanded
            )
            if lazy_consumer
            else None
        )
        key = self._cache_key(
            schema,
            signatures[node_id],
            inputs,
            selection,
            connected_undemanded_inputs,
            cache_scope,
        )
        previous_components = (
            self._remember_key_components(
                label,
                schema,
                signatures[node_id],
                inputs,
                selection,
                connected_undemanded_inputs,
            )
            if self._explain_misses
            else None
        )
        use_cache = cache_enabled and (not lazy_consumer or schema.idempotent)

        while use_cache and (inflight := self._inflight.get(key)) is not None:
            try:
                # shield: cancelling THIS run must not cancel the owner's
                # computation, which other runs may be waiting on.
                outputs = await asyncio.shield(inflight)
            except asyncio.CancelledError:
                # Both "owner died" and "this run is being cancelled" surface
                # here as CancelledError, and both can be true at once. Our
                # own cancellation wins: taking ownership while being torn
                # down would swallow the cancellation and could leave the
                # TaskGroup waiting on a replacement computation.
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise  # this run was cancelled
                if inflight.cancelled():
                    continue  # owner's run died; try to take ownership
                raise
            # Dead-owner check FIRST: a known-unresolvable result must not
            # take pins it would only hold until run cleanup.
            if self._dead_owner(outputs) is not None or not self._pin_outputs(outputs, pinned):
                # The owning worker lifetime ended, or a referenced resource
                # was released between the owner's run ending and this waiter
                # resuming (the orphaned-future window): the stub would
                # dangle, so recompute instead.
                await asyncio.sleep(0)
                continue
            produced[node_id] = outputs
            cached.append(label)
            self._emit_media_diagnostics(run_id, label, schema, inputs, outputs)
            self._emit(
                EngineEvent(
                    "node_cached",
                    run_id,
                    label,
                    self._execution_detail(
                        label,
                        selection,
                        cache_key=key,
                        coalesced=True,
                        outputs=output_summary(outputs, self._registry),
                    ),
                )
            )
            return

        future: asyncio.Future[Mapping[str, Value]] = asyncio.get_running_loop().create_future()
        if use_cache:
            self._inflight[key] = future
        try:
            hit = await self._cache.get(key) if use_cache else None
            cache_layer: str | None = None
            if hit is not None:
                take_hit_layer = getattr(self._cache, "take_hit_layer", None)
                if callable(take_hit_layer):
                    reported_layer = take_hit_layer()
                    if isinstance(reported_layer, str):
                        cache_layer = reported_layer
            entry_problem: str | None = None
            if hit is not None and not self._cache_hit_valid(schema, hit):
                entry_problem = "invalid-entry"
                hit = None
            # A hit referencing a dead worker lifetime can never resolve
            # (the resident died with its process): miss, recompute, and
            # the fresh result overwrites the entry under the same key.
            # Checked BEFORE pinning so a known-dead entry never takes
            # pins it would only hold until run cleanup.
            elif hit is not None and self._dead_owner(hit) is not None:
                entry_problem = "dead-owner"
                hit = None
            # A hit referencing a condemned resource is a stale read
            # (e.g. from a cache whose get() truly suspends across the
            # release's invalidation): treat as a miss and recompute.
            elif hit is not None and not self._pin_outputs(hit, pinned):
                entry_problem = "stale-resource"
                hit = None
            if hit is not None:
                produced[node_id] = hit
                cached.append(label)
                future.set_result(hit)
                self._emit_media_diagnostics(run_id, label, schema, inputs, hit)
                cache_detail: dict[str, object] = {}
                if cache_layer is not None:
                    cache_detail["cacheLayer"] = cache_layer
                self._emit(
                    EngineEvent(
                        "node_cached",
                        run_id,
                        label,
                        self._execution_detail(
                            label,
                            selection,
                            **cache_detail,
                            cache_key=key,
                            outputs=output_summary(hit, self._registry),
                        ),
                    )
                )
                return

            if self._explain_misses:
                detail = self._miss_detail(
                    key,
                    schema,
                    signatures[node_id],
                    inputs,
                    selection,
                    connected_undemanded_inputs,
                    previous_components,
                )
                if entry_problem is not None:
                    detail["reason"] = entry_problem
                self._emit(EngineEvent("cache_miss", run_id, label, detail))
            self._emit(
                EngineEvent(
                    "node_started",
                    run_id,
                    label,
                    self._execution_detail(label, selection),
                )
            )
            started = time.perf_counter()
            preview_policy = self._run_preview_policy.get(run_id)
            invocation = Invocation(
                invocation_id=uuid.uuid4().hex,
                node_id=label,
                node_type=node.node_type,
                inputs=inputs,
                effective_schema=schema,
                job_ref=run_id,
                attempt_id=self._run_attempts[run_id],
                connected_undemanded_inputs=connected_undemanded_inputs or (),
                output_members=tuple(
                    (fam.id, tuple(node.output_members.get(fam.id, ())))
                    for fam in self._schemas_for(run_id)[node.node_type].output_families
                ),
                executor=selection.target if selection is not None else None,
                expected_execution_identity=(
                    selection.cache_tag if selection is not None else None
                ),
                extension_snapshot_digest=(
                    self._runtime().extension_snapshot_digest
                    if self._runtime().extension_behavior_hash is not None
                    else None
                ),
                export_snapshot=(deepcopy(export_snapshot) if schema.output_node else None),
                fp8_matmul=(selection.fp8_matmul if selection is not None else False),
                diffusion_dtype=(selection.diffusion_dtype if selection is not None else None),
                text_dtype=(selection.text_dtype if selection is not None else None),
                vae_dtype=(selection.vae_dtype if selection is not None else None),
                attention_policy=(selection.attention_policy if selection is not None else "auto"),
                attention_route_token=(
                    selection.attention_route_token if selection is not None else None
                ),
                media_sources=self._run_media_sources.get(run_id, ()),
                preview_mode=(
                    # Schema capability gates the policy: unflagged node types
                    # never pay for preview emitters, provider matching, or
                    # decodes, even under an explicit per-node override.
                    preview_policy.resolve(label)
                    if preview_policy is not None and schema.emits_previews
                    else "off"
                ),
                preview_animation=(
                    preview_policy.animation if preview_policy is not None else "ring"
                ),
            )
            if schema.selector is not None:
                selector = schema.selector
                selected = inputs[selector.input].resolve()
                assert type(selected) is bool
                branch = selector.branches["true" if selected else "false"]
                result = InvocationResult(outputs={schema.outputs[0].id: inputs[branch]})
            else:
                async with self._admission(schema, inputs, selection):
                    result = await self._runtime().worker.invoke(
                        invocation,
                        on_event=lambda ev: self._emit_node_event(run_id, label, ev, selection),
                    )
            duration_ms = (time.perf_counter() - started) * 1000.0

            if result.error is not None:
                self._emit(
                    EngineEvent(
                        "node_failed",
                        run_id,
                        label,
                        self._execution_detail(
                            label,
                            selection,
                            message=result.error.message,
                        ),
                    )
                )
                raise ExecutionError(result.error)
            if result.artifact_candidates:
                message = (
                    "untrusted saved artifact candidates reached the engine without host validation"
                )
                self._emit(
                    EngineEvent(
                        "node_failed",
                        run_id,
                        label,
                        self._execution_detail(label, selection, message=message),
                    )
                )
                raise ExecutionError(NodeError(label, node.node_type, message))
            assert result.outputs is not None
            # A fresh result must never reference a resident whose owning
            # worker lifetime already ended - the producing side stamps its
            # own live token, so a dead one here means some layer
            # misattributed provenance. Checked BEFORE pinning so the
            # doomed result never takes run-lifetime pins.
            if (dead := self._dead_owner(result.outputs)) is not None:
                rid, owner = dead
                message = (
                    f"fresh result of {label!r} references resource {rid!r} "
                    f"owned by dead worker lifetime {owner!r} (ownership "
                    "contract violation)"
                )
                self._emit(
                    EngineEvent(
                        "node_failed",
                        run_id,
                        label,
                        self._execution_detail(label, selection, message=message),
                    )
                )
                raise ExecutionError(NodeError(label, node.node_type, message))
            # Fresh worker results should never reference a condemned
            # resource: the child holds a result's residents until this side
            # pins and acks them, and a consumer's use token advances when a
            # result names an item (the in-flight half of the gate). If a
            # pin is refused anyway, some layer broke that contract - fail
            # the node loudly rather than put an unpinned stub in `produced`
            # where a later release would leave it dangling.
            if not self._pin_outputs(result.outputs, pinned):
                message = (
                    f"resource referenced by fresh result of {label!r} was "
                    "released during handoff (release-gate contract violation)"
                )
                self._emit(
                    EngineEvent(
                        "node_failed",
                        run_id,
                        label,
                        self._execution_detail(label, selection, message=message),
                    )
                )
                raise ExecutionError(NodeError(label, node.node_type, message))
            self._run_artifacts[run_id].extend(result.artifacts)
            produced[node_id] = result.outputs
            executed.append(label)
            if use_cache:
                await self._cache.put(key, result.outputs)
            future.set_result(result.outputs)
            self._emit_media_diagnostics(run_id, label, schema, inputs, result.outputs)
            self._emit(
                EngineEvent(
                    "node_finished",
                    run_id,
                    label,
                    self._execution_detail(
                        label,
                        selection,
                        duration_ms=round(duration_ms, 3),
                        cache_key=key,
                        outputs=output_summary(result.outputs, self._registry),
                    ),
                )
            )
        except BaseException as exc:
            # Retained error tracebacks must not own the invocation's inputs.
            inputs.clear()
            if not future.done():
                if isinstance(exc, asyncio.CancelledError):
                    future.cancel()  # waiters retry and take ownership
                else:
                    future.set_exception(exc)  # computation failures propagate to waiters
                    future.exception()  # mark retrieved; our own raise suffices
            raise
        finally:
            if use_cache and self._inflight.get(key) is future:
                del self._inflight[key]

    async def _execute_dag(
        self,
        run_id: str,
        graph: Graph,
        effective: Mapping[str, NodeSchema],
        signatures: Mapping[str, str],
        order: Sequence[str],
        deps: Mapping[str, set[str]],
        produced: dict[str, Mapping[str, Value]],
        executed: list[str],
        cached: list[str],
        skipped: list[str],
        pinned: list[str],
        export_snapshot: ExportSnapshot | None,
        prefix: str = "",
        targets: Sequence[str] | None = None,
        cache_enabled: bool = True,
        cache_scope: str | None = None,
    ) -> None:
        """Ready-set scheduler over one DAG level (hazard H12): dispatch
        every node whose dependencies are satisfied; completions release
        dependents. Used for the top-level document and, recursively, for
        each region iteration - the SAME scheduler, cache, single-flight
        table, admission lanes, and pin list govern both."""
        initial_targets = (
            list(targets)
            if targets is not None
            else [node_id for node_id in order if not any(node_id in ds for ds in deps.values())]
        )
        planned = set(order)
        planned_types = sorted(
            {
                node.node_type
                for node_id in order
                if isinstance((node := graph.nodes[node_id]), GraphNode)
            }
        )
        if planned_types:
            await self._prepare_node_types(run_id, planned_types)
        if not prefix:
            self._emit(EngineEvent("run_started", run_id, detail={"planned": list(order)}))
        has_connected_lazy = any(
            isinstance((node := graph.nodes[node_id]), GraphNode)
            and any(
                spec.lazy and isinstance(node.inputs.get(spec.id), Link)
                for spec in effective[node_id].inputs
            )
            for node_id in order
        )
        if not has_connected_lazy:
            try:
                await self._execute_static_dag(
                    run_id,
                    graph,
                    effective,
                    signatures,
                    order,
                    deps,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix,
                    _ProducedReferences(deps, initial_targets, produced),
                    cache_enabled,
                    cache_scope,
                )
            except BaseException:
                produced.clear()
                raise
            return

        tasks: dict[str, asyncio.Task[None]] = {}
        references = _ProducedReferences(
            dependency_map(graph, initial_targets), initial_targets, produced
        )

        def ordinary_dependencies(node_id: str) -> set[str]:
            node = graph.nodes[node_id]
            if not isinstance(node, GraphNode):
                return {
                    value.node_id
                    for value in node.inputs.values()
                    if isinstance(value, Link) and value.node_id in graph.nodes
                }
            lazy = {spec.id for spec in effective[node_id].inputs if spec.lazy}
            return {
                value.node_id
                for name, value in node.inputs.items()
                if (isinstance(value, Link) and name not in lazy and value.node_id in graph.nodes)
            }

        async def run_one(node_id: str) -> None:
            node = graph.nodes[node_id]
            if isinstance(node, GraphNode) and node_id not in planned:
                await self._prepare_node_types(run_id, (node.node_type,))
            await asyncio.gather(
                *(ensure_node(dep) for dep in sorted(ordinary_dependencies(node_id)))
            )
            if isinstance(node, RegionNode):
                await self._run_region(
                    run_id,
                    node_id,
                    node,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix,
                )
            else:
                await self._run_node(
                    run_id,
                    graph,
                    node_id,
                    effective,
                    signatures,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix,
                    ensure_node,
                    cache_enabled,
                    cache_scope,
                )
            references.finished(node_id)

        async def ensure_node(node_id: str) -> None:
            task = tasks.get(node_id)
            if task is None:
                task = asyncio.create_task(run_one(node_id))
                tasks[node_id] = task
            await asyncio.shield(task)

        try:
            await asyncio.gather(*(ensure_node(node_id) for node_id in initial_targets))
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            produced.clear()
            raise

    async def _execute_static_dag(
        self,
        run_id: str,
        graph: Graph,
        effective: Mapping[str, NodeSchema],
        signatures: Mapping[str, str],
        order: Sequence[str],
        deps: Mapping[str, set[str]],
        produced: dict[str, Mapping[str, Value]],
        executed: list[str],
        cached: list[str],
        skipped: list[str],
        pinned: list[str],
        export_snapshot: ExportSnapshot | None,
        prefix: str,
        references: _ProducedReferences,
        cache_enabled: bool,
        cache_scope: str | None,
    ) -> None:
        """Original ready-set scheduler for graphs without deferred edges."""
        dependents: dict[str, list[str]] = {node_id: [] for node_id in deps}
        for node_id, node_deps in deps.items():
            for dependency in node_deps:
                dependents[dependency].append(node_id)
        remaining = {node_id: len(node_deps) for node_id, node_deps in deps.items()}

        async def step(node_id: str, task_group: asyncio.TaskGroup) -> None:
            node = graph.nodes[node_id]
            if isinstance(node, RegionNode):
                await self._run_region(
                    run_id,
                    node_id,
                    node,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix,
                )
            else:
                await self._run_node(
                    run_id,
                    graph,
                    node_id,
                    effective,
                    signatures,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix,
                    cache_enabled=cache_enabled,
                    cache_scope=cache_scope,
                )
            references.finished(node_id)
            for dependent in dependents[node_id]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    task_group.create_task(step(dependent, task_group))

        try:
            async with asyncio.TaskGroup() as task_group:
                for node_id in order:
                    if remaining[node_id] == 0:
                        task_group.create_task(step(node_id, task_group))
        except* ExecutionError as group:
            raise group.exceptions[0] from None

    async def _run_region(
        self,
        run_id: str,
        node_id: str,
        region: RegionNode,
        produced: dict[str, Mapping[str, Value]],
        executed: list[str],
        cached: list[str],
        skipped: list[str],
        pinned: list[str],
        export_snapshot: ExportSnapshot | None,
        prefix: str = "",
    ) -> None:
        """Expand one region (DESIGN 3.13): the single repetition primitive
        under the map/fold/while profiles.

        Every iteration's body nodes run through _run_node on the shared
        engine state, so caching, single-flight, admission, pins, and
        diagnostics behave exactly as at top level; iteration node ids are
        namespaced ``region[3]/node``. A nested region scopes its body cache to
        the parent occurrence; bindings still flow through ordinary input
        fingerprints, so changing one list element re-executes only that
        occurrence's dependents.
        """
        label = prefix + node_id
        region_type = f"region:{region.kind}"

        def region_error(message: str) -> ExecutionError:
            self._emit(EngineEvent("node_failed", run_id, label, {"message": message}))
            return ExecutionError(NodeError(label, region_type, message))

        # Outer inputs resolve with node-input semantics: links read sibling
        # outputs, literals wrap with the declared port type (an element port
        # declared T takes a list<T> literal).
        await self._prepare_host_types(region.ports.values(), region.inputs.values())
        inputs: dict[str, Value] = {}
        for input_id, raw in region.inputs.items():
            if isinstance(raw, Link):
                inputs[input_id] = produced[raw.node_id][raw.output_id]
            elif isinstance(raw, TypedLiteral):
                # Same stamp-wins rule as node inputs: the literal names its
                # own runtime type, covering ports whose declared type does
                # not resolve to one.
                inputs[input_id] = self._registry.wrap(raw.type_id, raw.value)
            else:
                port_type = region.ports[input_id]
                expected = (
                    TypeExpr.list_of(port_type) if input_id in region.element_ports else port_type
                )
                runtime = expected.runtime_type_id()
                if runtime is None:  # pragma: no cover - validation rejects this
                    raise ValueError(f"{label}: literal on non-runtime-resolvable port {input_id}")
                inputs[input_id] = self._registry.wrap(runtime, raw)

        interface = region_interface(region, self._schemas_for(run_id))

        # Whole-region absence (DESIGN 3.15): any absent input skips the
        # region - zero iterations, no body work - and every output becomes
        # absent with the ROOT origin propagated unchanged.
        absent_input = next((iid for iid in sorted(inputs) if is_absent(inputs[iid])), None)
        if absent_input is not None:
            value = inputs[absent_input]
            origin = str(value.meta.get(ABSENT_ORIGIN_META_KEY, ""))
            reason = str(value.meta.get(ABSENT_REASON_META_KEY, ""))
            produced[node_id] = {
                out_id: make_absent_value(
                    origin=origin,
                    reason=reason,
                    stands_for=(None if (t := interface[out_id]) is None else t.runtime_type_id()),
                )
                for out_id in region.outputs
            }
            skipped.append(label)
            self._emit(
                EngineEvent(
                    "node_skipped",
                    run_id,
                    label,
                    {"input": absent_input, "origin": origin, "reason": reason},
                )
            )
            return

        # Body preparation happens once per region, not per iteration: every
        # iteration sees the SAME elaborated interfaces (hazard H10).
        body_effective, body_diags = elaborate_graph(region.body, self._schemas_for(run_id))
        if has_errors(body_diags):  # pragma: no cover - validated before run
            raise region_error("region body failed elaboration")
        body_signatures = {nid: schema_signature(s) for nid, s in body_effective.items()}
        sources = [out.source for out in region.outputs.values()]
        if region.continue_source is not None:
            sources.append(region.continue_source)
        body_targets = sorted({src.node_id for src in sources if src.node_id != PORTS_NODE_ID})
        body_order, body_deps = self._strong_plan(region.body, body_targets, body_effective)

        broadcast = {
            input_id: value
            for input_id, value in inputs.items()
            if input_id not in region.element_ports and input_id not in region.state_ports
        }
        state: dict[str, Value] = {p: inputs[p] for p in region.state_ports}

        # Binding sets. zip: one iteration per index, all element lists must
        # have equal length (loud error otherwise). broadcast: iterate to the
        # longest list, repeating each shorter list's final element. cross:
        # the cartesian product, deterministic in declared element-port order
        # with the last port varying fastest.
        bindings: list[dict[str, Value]] = []
        if region.element_ports:
            elements: dict[str, tuple[Value, ...]] = {}
            for port in region.element_ports:
                children = list_children(inputs[port])
                if children is None:
                    raise region_error(
                        f"element port '{port}' received a non-list value ({inputs[port].type_id})"
                    )
                elements[port] = children
            if region.binding == "zip":
                lengths = {port: len(ch) for port, ch in elements.items()}
                if len(set(lengths.values())) > 1:
                    raise region_error(
                        "zip binding requires equal-length element lists, got "
                        + ", ".join(f"{p}={n}" for p, n in lengths.items())
                    )
                count = next(iter(lengths.values()))
                bindings = [
                    {port: elements[port][i] for port in region.element_ports} for i in range(count)
                ]
            elif region.binding == "cross":
                bindings = [
                    dict(zip(region.element_ports, combo, strict=True))
                    for combo in itertools.product(*(elements[p] for p in region.element_ports))
                ]
            elif region.binding == "broadcast":
                lengths = {port: len(ch) for port, ch in elements.items()}
                count = max(lengths.values())
                empty_ports = [port for port, length in lengths.items() if length == 0]
                if empty_ports and count:
                    raise region_error(
                        "broadcast binding cannot repeat empty element ports: "
                        + ", ".join(empty_ports)
                    )
                bindings = [
                    {
                        port: elements[port][min(i, lengths[port] - 1)]
                        for port in region.element_ports
                    }
                    for i in range(count)
                ]
            else:  # pragma: no cover - validation rejects this
                raise region_error(f"unknown binding mode {region.binding!r}")
            if region.max_iterations is not None and len(bindings) > region.max_iterations:
                raise region_error(
                    f"{len(bindings)} bindings exceed max_iterations={region.max_iterations}"
                )

        self._emit(
            EngineEvent(
                "region_expanded",
                run_id,
                label,
                {
                    "kind": region.kind,
                    "binding": region.binding,
                    "iterations": None if region.kind == "while" else len(bindings),
                },
            )
        )

        gathered: dict[str, list[Value]] = {
            out_id: []
            for out_id, out in region.outputs.items()
            if out.mode in ("gather", "compact", "flatten")
        }
        last_values: dict[str, Value] = {}

        async def run_iteration(
            index: int,
            binding: Mapping[str, Value],
            iteration_state: Mapping[str, Value],
        ) -> dict[str, Mapping[str, Value]]:
            self._emit(EngineEvent("region_iteration_started", run_id, label, {"iteration": index}))
            body_ports = {**broadcast, **binding, **iteration_state}
            if REGION_INDEX_PORT_ID not in region.ports:
                index_type_id = REGION_INDEX_PORT_TYPE.runtime_type_id()
                assert index_type_id is not None
                body_ports[REGION_INDEX_PORT_ID] = self._registry.wrap(index_type_id, index)
            body_produced: dict[str, Mapping[str, Value]] = {PORTS_NODE_ID: body_ports}
            try:
                await self._execute_dag(
                    run_id,
                    region.body,
                    body_effective,
                    body_signatures,
                    body_order,
                    body_deps,
                    body_produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    prefix=f"{label}[{index}]/",
                    targets=body_targets,
                    cache_enabled=region.cache_policy == "reuse",
                    cache_scope=label if prefix else None,
                )
            except BaseException:
                body_produced.clear()
                body_ports.clear()
                raise
            self._emit(
                EngineEvent("region_iteration_finished", run_id, label, {"iteration": index})
            )
            return body_produced

        def source_value(body_produced: Mapping[str, Mapping[str, Value]], source: Link) -> Value:
            return body_produced[source.node_id][source.output_id]

        def collect(body_produced: Mapping[str, Mapping[str, Value]]) -> None:
            for out_id, out in region.outputs.items():
                if out.mode in ("gather", "compact", "flatten"):
                    gathered[out_id].append(source_value(body_produced, out.source))
                elif out.mode == "last":
                    last_values[out_id] = source_value(body_produced, out.source)

        def advance_state(body_produced: Mapping[str, Mapping[str, Value]]) -> None:
            for out_id, out in region.outputs.items():
                if out.mode == "state":
                    state[out_id] = source_value(body_produced, out.source)

        iterations_done = 0
        if region.kind == "map":
            # Independent iterations dispatch concurrently; admission lanes
            # still bound real resource use, and identical bindings coalesce
            # through single-flight. Gather order is index order regardless
            # of completion order.
            results: list[dict[str, Mapping[str, Value]] | None] = [None] * len(bindings)

            async def map_step(index: int, binding: Mapping[str, Value]) -> None:
                results[index] = await run_iteration(index, binding, {})

            try:
                async with asyncio.TaskGroup() as tg:
                    for index, binding in enumerate(bindings):
                        tg.create_task(map_step(index, binding))
            except* ExecutionError as group:
                raise group.exceptions[0] from None
            for body_produced in results:
                assert body_produced is not None
                collect(body_produced)
                body_produced.clear()
            iterations_done = len(bindings)
        elif region.kind == "fold":
            for index, binding in enumerate(bindings):
                body_produced = await run_iteration(index, binding, state)
                collect(body_produced)
                advance_state(body_produced)
                body_produced.clear()
            iterations_done = len(bindings)
        elif region.kind == "while":
            assert region.continue_source is not None
            assert region.max_iterations is not None
            while True:
                body_produced = {}
                cont_value = None
                cont = None
                try:
                    body_produced = await run_iteration(iterations_done, {}, state)
                    cont_value = source_value(body_produced, region.continue_source)
                    if is_absent(cont_value):
                        raise region_error("continue source produced no value (absent)")
                    cont = cont_value.resolve()
                    if not isinstance(cont, bool):
                        raise region_error(
                            f"continue source produced {type(cont).__name__}, expected a boolean"
                        )
                    collect(body_produced)
                    advance_state(body_produced)
                    iterations_done += 1
                    if not cont:
                        break
                    if iterations_done >= region.max_iterations:
                        raise region_error(
                            f"reached max_iterations={region.max_iterations} with continue "
                            "still true (infinite-loop guard)"
                        )
                except BaseException:
                    gathered.clear()
                    state.clear()
                    raise
                finally:
                    body_produced.clear()
                    cont_value = None
                    cont = None
        else:  # pragma: no cover - validation rejects this
            raise region_error(f"unknown region kind {region.kind!r}")

        # Assemble region outputs. A gather or flatten with an absent child
        # makes the WHOLE collection absent with the child's root provenance.
        # Compact is the explicit opt-in that omits typed-absent children.
        outputs: dict[str, Value] = {}
        for out_id, out in region.outputs.items():
            if out.mode == "state":
                outputs[out_id] = state[out_id]
                continue
            if out.mode == "last":
                if out_id in last_values:
                    outputs[out_id] = last_values[out_id]
                else:
                    output_type = interface[out_id]
                    outputs[out_id] = make_absent_value(
                        origin=f"{label}/{out_id}",
                        reason="region completed zero iterations",
                        stands_for=(None if output_type is None else output_type.runtime_type_id()),
                    )
                continue
            if out.mode not in ("gather", "compact", "flatten"):  # pragma: no cover
                raise region_error(f"output '{out_id}' has unknown mode {out.mode!r}")
            list_type = interface[out_id]
            list_runtime = None if list_type is None else list_type.runtime_type_id()
            if list_runtime is None:  # pragma: no cover - validation rejects this
                raise region_error(f"region output '{out_id}' has no runtime-resolvable type")
            element_type = parse_list_type_id(list_runtime)
            assert element_type is not None
            iteration_values = gathered[out_id]
            absent_child = next((c for c in iteration_values if is_absent(c)), None)
            if absent_child is not None and out.mode != "compact":
                outputs[out_id] = make_absent_value(
                    origin=str(absent_child.meta.get(ABSENT_ORIGIN_META_KEY, "")),
                    reason=str(absent_child.meta.get(ABSENT_REASON_META_KEY, "")),
                    stands_for=list_runtime,
                )
            else:
                collected_children: list[Value] = [
                    child for child in iteration_values if not is_absent(child)
                ]
                if out.mode == "flatten":
                    collected_children = []
                    for iteration_value in iteration_values:
                        iteration_children = list_children(iteration_value)
                        if iteration_children is None:  # pragma: no cover
                            raise region_error(
                                f"flatten output '{out_id}' received a non-list body value"
                            )
                        collected_children.extend(iteration_children)
                outputs[out_id] = make_list_value(element_type, collected_children)
        produced[node_id] = outputs
        self._emit(EngineEvent("region_finished", run_id, label, {"iterations": iterations_done}))

    @staticmethod
    def _strong_plan(
        graph: Graph, targets: Sequence[str], effective: Mapping[str, NodeSchema]
    ) -> tuple[list[str], dict[str, set[str]]]:
        """Initial closure and topo order with lazy links deferred."""

        def dependencies(node_id: str) -> set[str]:
            node = graph.nodes[node_id]
            if not isinstance(node, GraphNode):
                return {
                    value.node_id
                    for value in node.inputs.values()
                    if isinstance(value, Link) and value.node_id in graph.nodes
                }
            lazy = {spec.id for spec in effective[node_id].inputs if spec.lazy}
            return {
                value.node_id
                for name, value in node.inputs.items()
                if isinstance(value, Link) and name not in lazy and value.node_id in graph.nodes
            }

        needed: set[str] = set()
        stack = list(targets)
        while stack:
            node_id = stack.pop()
            if node_id in needed:
                continue
            needed.add(node_id)
            stack.extend(dependencies(node_id))
        deps = {node_id: dependencies(node_id) & needed for node_id in needed}
        remaining = {node_id: set(items) for node_id, items in deps.items()}
        order: list[str] = []
        while ready := sorted(
            node_id for node_id, items in remaining.items() if not items and node_id not in order
        ):
            node_id = ready[0]
            order.append(node_id)
            for items in remaining.values():
                items.discard(node_id)
        return order, deps

    @property
    def registry(self) -> TypeRegistry:
        """The engine's type registry - read-side consumers (the server's
        descriptor/rendition endpoints) share the same registrations that
        wrapped the values."""
        return self._registry

    @property
    def cache(self) -> CacheStore:
        """The engine's cache store, read-only by convention: the server's
        frozen-view value retrieval (DESIGN 3.5) looks up retained
        intermediate outputs by the cache keys engine events reported.
        Writing stays the engine's job."""
        return self._cache

    def resource_status(self) -> dict[str, dict[str, int]]:
        """Admission lanes and their occupancy - the engine's contribution to
        GET /memory/status (DESIGN 3.10). Lanes appear once first used."""
        status: dict[str, dict[str, int]] = {}
        for lane in self._resource_slots:
            capacity = self._resource_capacities.get(lane)
            if capacity is None:
                capacity = self._resource_capacities.get(lane.split(":", 1)[0], 1)
            status[lane] = {
                "executionCapacity": capacity,
                "executionInUse": self._lane_in_use.get(lane, 0),
            }
        return status

    async def compile_for_execution(
        self,
        graph: Graph,
        targets: Sequence[str],
        *,
        execution: ExecutionRuntime | None = None,
    ) -> CompiledGraph:
        """Compile and parent-validate a graph against one pinned runtime."""
        runtime = self._normalize_execution(
            self.pin_execution() if execution is None else execution
        )
        assert runtime.schemas is not None
        return await compile_graph(
            graph,
            targets,
            generation_key=runtime.extension_snapshot_digest,
            registry=runtime.graph_compiler_registry,
            transport=runtime.graph_compile_transport,
            schemas=runtime.schemas,
        )

    async def run(
        self,
        graph: Graph,
        targets: Sequence[str],
        *,
        run_id: str | None = None,
        attempt_id: int = 1,
        execution: ExecutionRuntime | None = None,
        export_snapshot: ExportSnapshot | None = None,
        media_sources: Sequence[MediaSourceAuthority] = (),
        preview_policy: PreviewPolicy | None = None,
        attention_config: AttentionPolicyConfig | None = None,
    ) -> RunResult:
        """Execute a raw graph, compiling once when this runtime declares compilers."""
        # Preserve legacy raw-run admission order: run-id errors precede a
        # host pin callback, even when the runtime ultimately has compilers.
        if run_id is not None and not run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(attempt_id) is not int or attempt_id < 1:
            raise ValueError("attempt_id must be a positive integer")
        run_id = run_id if run_id is not None else uuid.uuid4().hex[:12]
        if run_id in self._run_schemas or run_id in self._compiling_run_ids:
            raise ActiveRunIdError(f"run_id {run_id!r} is already active")
        runtime = (
            self.pin_execution() if execution is None else self._normalize_execution(execution)
        )
        graph = migrate_pure_node_type_replacements(graph, runtime.schemas or self._schemas)
        graph = snapshot_graph(graph)
        if runtime.resolve_providers is not None:
            graph = runtime.resolve_providers(graph)
        if not runtime.graph_compiler_registry.contributions:
            return await self._run_graph(
                graph,
                targets,
                run_id=run_id,
                attempt_id=attempt_id,
                execution=runtime,
                export_snapshot=export_snapshot,
                media_sources=media_sources,
                preview_policy=preview_policy,
                attention_config=attention_config,
            )
        self._compiling_run_ids.add(run_id)
        try:
            compiled = await self.compile_for_execution(graph, targets, execution=runtime)
        finally:
            self._compiling_run_ids.remove(run_id)
        return await self.run_compiled(
            compiled,
            run_id=run_id,
            attempt_id=attempt_id,
            execution=runtime,
            export_snapshot=export_snapshot,
            media_sources=media_sources,
            preview_policy=preview_policy,
            attention_config=attention_config,
        )

    async def run_compiled(
        self,
        compiled: CompiledGraph,
        *,
        run_id: str | None = None,
        attempt_id: int = 1,
        execution: ExecutionRuntime | None = None,
        export_snapshot: ExportSnapshot | None = None,
        media_sources: Sequence[MediaSourceAuthority] = (),
        preview_policy: PreviewPolicy | None = None,
        attention_config: AttentionPolicyConfig | None = None,
    ) -> RunResult:
        """Execute a validated artifact without invoking graph compilation."""
        if type(attempt_id) is not int or attempt_id < 1:
            raise ValueError("attempt_id must be a positive integer")
        runtime = (
            self.pin_execution() if execution is None else self._normalize_execution(execution)
        )
        if compiled.extension_snapshot_digest != runtime.extension_snapshot_digest:
            raise GraphCompileError(
                GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
                "compiled graph generation does not match the execution runtime",
            )
        return await self._run_graph(
            compiled.graph,
            compiled.targets,
            run_id=run_id,
            attempt_id=attempt_id,
            execution=runtime,
            export_snapshot=export_snapshot,
            media_sources=media_sources,
            preview_policy=preview_policy,
            attention_config=attention_config,
        )

    async def _run_graph(
        self,
        graph: Graph,
        targets: Sequence[str],
        *,
        run_id: str | None = None,
        attempt_id: int = 1,
        execution: ExecutionRuntime | None = None,
        export_snapshot: ExportSnapshot | None = None,
        media_sources: Sequence[MediaSourceAuthority] = (),
        preview_policy: PreviewPolicy | None = None,
        attention_config: AttentionPolicyConfig | None = None,
    ) -> RunResult:
        """Execute targets. run_id defaults to a fresh unique id; a caller
        that owns job identity (the server's queue) supplies its own so
        engine events correlate with the job without a side channel."""
        if run_id is not None and not run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(attempt_id) is not int or attempt_id < 1:
            raise ValueError("attempt_id must be a positive integer")
        run_id = run_id if run_id is not None else uuid.uuid4().hex[:12]
        if run_id in self._run_schemas or run_id in self._compiling_run_ids:
            raise ActiveRunIdError(f"run_id {run_id!r} is already active")
        runtime = execution or self.pin_execution()
        # Every admission hook reads an immutable graph captured at run entry.
        graph = snapshot_graph(graph)
        if runtime.resolve_providers is not None:
            graph = runtime.resolve_providers(graph)
        # Run-scoped snapshot (hazard H10): a deep immutable copy of the
        # admission result. The caller may keep mutating its graph (dicts of
        # inputs/output members) while this coroutine is suspended at any
        # await; every consumer below - the elaboration pass, validation,
        # planning, input resolution, and invocations - reads ONLY this
        # snapshot, so a run's topology is fixed at entry and can never drift
        # mid-run.
        graph = snapshot_graph(graph)
        targets = tuple(targets)
        # Pin the schema mapping alongside the graph snapshot: every mid-run
        # lookup goes through _schemas_for(run_id), so a hot reload swapping
        # self._schemas mid-run changes nothing this run can see.
        schemas = runtime.schemas or self._schemas
        sources = tuple(media_sources)
        if any(type(source) is not MediaSourceAuthority for source in sources):
            raise ValueError("media_sources must contain MediaSourceAuthority values")
        if len({source.digest for source in sources}) != len(sources):
            raise ValueError("media_sources must have unique digests")
        if preview_policy is not None and type(preview_policy) is not PreviewPolicy:
            raise ValueError("preview_policy must be a PreviewPolicy")
        if attention_config is not None and type(attention_config) is not AttentionPolicyConfig:
            raise ValueError("attention_config must be an AttentionPolicyConfig")
        runtime_token = _execution_runtime.set(runtime)
        self._run_schemas[run_id] = schemas
        self._run_media_sources[run_id] = sources
        self._run_attempts[run_id] = attempt_id
        self._run_prepared_types[run_id] = set()
        self._run_prepare_locks[run_id] = asyncio.Lock()
        if preview_policy is not None:
            self._run_preview_policy[run_id] = preview_policy
        if attention_config is not None:
            self._run_attention_config[run_id] = attention_config
        self._run_artifacts[run_id] = []
        try:
            # One elaboration pass per run (hazard H10): the SAME effective map
            # feeds validation, cache keys, and invocations, so what validation
            # approved is exactly what executes - no re-elaboration anywhere.
            effective, diagnostics = elaborate_graph(graph, schemas)
            # The registry supplies type names, equivalences, and coercion providers.
            diagnostics = diagnostics + validate(
                graph,
                schemas,
                targets,
                effective,
                known_types=runtime.known_types
                if runtime.known_types is not None
                else self._registry,
            )
            if has_errors(diagnostics):
                raise GraphValidationError(diagnostics)
            signatures = {node_id: schema_signature(s) for node_id, s in effective.items()}

            # Validation above used every edge. Scheduling deliberately starts
            # with only the ordinary closure; demanded producers are added by
            # the worker-bound hook fixpoint.
            order, deps = self._strong_plan(graph, targets, effective)

            produced: dict[str, Mapping[str, Value]] = {}
            executed: list[str] = []  # completion order, not semantic
            cached: list[str] = []
            skipped: list[str] = []
            # Resource references this run holds (DESIGN 3.10): pinned as envelopes
            # enter `produced`, unpinned when the run ends. Between invocations a
            # reference lives only in these Python structures - the pin is what
            # makes it visible to a ram-lane release. RunResult outputs returned
            # to the caller are envelopes for display/serialization; a caller
            # that resolves a resource reference *after* the run must tolerate
            # recompute semantics (the resident may have been released).

            pinned: list[str] = []

            # The ready-set scheduler lives in _execute_dag (hazard H12): the
            # topological order is only the deterministic linearization for
            # events; execution follows the DAG, and regions recurse into the
            # same scheduler per iteration.
            try:
                await self._execute_dag(
                    run_id,
                    graph,
                    effective,
                    signatures,
                    order,
                    deps,
                    produced,
                    executed,
                    cached,
                    skipped,
                    pinned,
                    export_snapshot,
                    targets=targets,
                )
                outputs = {t: produced[t] for t in targets}
            finally:
                # Recursive scheduler closures can survive until cyclic GC.
                # Only RunResult, not a completed scheduler, owns the exports.
                produced.clear()
                if self._pins is not None:
                    for resource_id in pinned:
                        self._pins.unpin(resource_id)

            self._emit(
                EngineEvent(
                    "run_finished",
                    run_id,
                    detail={
                        "executed": len(executed),
                        "cached": len(cached),
                        "skipped": len(skipped),
                    },
                )
            )
            return RunResult(
                run_id=run_id,
                outputs=outputs,
                executed=tuple(executed),
                cached=tuple(cached),
                diagnostics=tuple(diagnostics),
                skipped=tuple(skipped),
                artifacts=tuple(self._run_artifacts[run_id]),
            )
        finally:
            self._run_schemas.pop(run_id, None)
            self._run_media_sources.pop(run_id, None)
            self._run_attempts.pop(run_id, None)
            self._run_preview_policy.pop(run_id, None)
            self._run_attention_config.pop(run_id, None)
            self._run_artifacts.pop(run_id, None)
            self._run_prepared_types.pop(run_id, None)
            self._run_prepare_locks.pop(run_id, None)
            _execution_runtime.reset(runtime_token)
            if runtime.run_finished is not None:
                runtime.run_finished(run_id)

    @property
    def extension_snapshot(self) -> ExtensionSnapshot:
        return self.pin_execution().extension_snapshot

    @property
    def extension_snapshot_digest(self) -> str:
        return self.pin_execution().extension_snapshot_digest
