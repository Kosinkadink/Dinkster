"""Multi-pack composition: pack manifests -> one serving engine.

Umbrella-owned wiring, like serve.py itself: dinkster-server knows nothing
about workers, dinkster-workers knows nothing about the HTTP app, and this
module is where a host turns "these packs are installed" into the three
things create_app needs - merged schemas, a packs table, and per-node
attribution - plus an engine factory over one composed Worker.

The host kernel has no nodes. Packs run as an IsolatedWorker by default
(schemas arrive over the hello handshake); trusted first-party packs may run
in-process through the same manifest path. RoutingWorker composes both paths
by node type behind the one Worker protocol, so the engine cannot tell the
difference. Provenance comes from the host-side loading record (which
manifest a schema arrived through), never from a schema's own claims.

Composition is configuration, so it fails loudly: a pack named "core", two
packs with the same name (canonically - separators are one identity), two
packs whose namespace claims overlap, a pack announcing a node type outside
its declared claims, or two packs announcing the same node type are
CompositionError at startup, never a silent precedence pick. Reserved
namespace roots (std, comfy, core, dinkster) compose only under a spec's
explicit trust_reserved - the local analog of the registry's grant table
(manifests claim, authority grants).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import tomllib
import uuid
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

from dinkster_assets import DeclaredAsset, PackAssetCatalog
from dinkster_caches import (
    DEFAULT_DISK_CACHE_BYTES,
    DiskCacheStore,
    LayeredCache,
    MemoryLRUCache,
)
from dinkster_engine import (
    Engine,
    EventListener,
    ExecutionArm,
    ExecutionError,
    ExecutionRuntime,
    ExecutionSelection,
    GraphCompileTransport,
    ProviderResolutionError,
    Worker,
)
from dinkster_graph import Graph, GraphNode, Link, RegionNode, TypedLiteral, top_level_node_id
from dinkster_inference import (
    INFERENCE_SAMPLERS_SURFACE,
    INFERENCE_SCHEDULERS_SURFACE,
    SAMPLER_CATALOG_ENV,
    Registry,
    RegistryError,
    SamplerExtensionEntry,
    builtin_registries,
    builtin_sampler_snapshot,
    register_inference_types,
    registry_choice_values,
    remove_sampler_catalog_record,
    sampler_choice_values,
    scheduler_declaration,
    write_sampler_catalog,
)
from dinkster_memory import (
    FullReleaseResult,
    MemoryGovernor,
    ModelTenantHandle,
    ModelTenantRegistry,
    ReportedTelemetry,
    ReservationRequest,
    ReservationService,
    TenantRegistration,
    use_model_tenant_registry,
)
from dinkster_native.memory import plan_reservations
from dinkster_native.native_residency import NativeComponentPublisher
from dinkster_native.pool import default_pool
from dinkster_protocol import (
    GRAPH_COMPILERS_SURFACE,
    GUIDANCE_SURFACES,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    ActiveExtension,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    BehaviorValue,
    CompatGateDiagnostic,
    CompositionMode,
    ContributionSurfaceDescriptor,
    DeviceResourceId,
    ExtensionDeclaration,
    ExtensionScope,
    ExtensionSnapshot,
    GraphCompilerRegistrySnapshot,
    GuidanceRegistrySnapshot,
    Invocation,
    InvocationResult,
    KeyedContribution,
    LazyStatusInvocation,
    LazyStatusResult,
    NodeError,
    OnInvocationEvent,
    ReplicaId,
    ReplicaRecipeId,
    SamplerRegistrySnapshot,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupId,
    WorkGroupLifecycle,
    WorkGroupState,
    WorkUnitId,
    attention_policy_config_to_wire,
    derive_attention_route_token,
    extension_behavior_hash,
)
from dinkster_protocol.frontend_modules import FrontendModule
from dinkster_protocol.pack_surfaces import PACK_EVENTS_SURFACE, PACK_ROUTES_SURFACE, PackRoute
from dinkster_registry import (
    ArtifactError,
    ComposedPack,
    CompositionGeneration,
    CompositionRecordError,
    InstallError,
    LockedPack,
    Lockfile,
    RequirementKind,
    ResolvedRequirement,
    build_artifact,
)
from dinkster_registry import (
    CompositionMode as PackCompositionMode,
)
from dinkster_schema import (
    ComboOption,
    ComboWidget,
    ComfyAliasRegistry,
    ComfyGroupRegistry,
    Node,
    NodeSchema,
    build_node_types,
    build_schemas,
    canonical_name,
    claim_covers,
    claims_conflict,
    combo_choices_json_bytes,
    comfy_alias_collision_problems,
    comfy_alias_registry_problems,
    comfy_group_collision_problems,
    comfy_group_registry_problems,
    core_logger,
    reserved_root,
    schema_signature,
    validate_remote_choice_authority,
)
from dinkster_server import (
    CORE_PACK_ID,
    ChoiceOwnerGone,
    LazyChoiceFetcher,
    PackInferenceUnavailable,
    PackInfo,
    UnavailableInferenceProvider,
    WorkerInfo,
    validate_comfy_args,
)
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCES_META_KEY,
    ListPayload,
    Rendition,
    ResourcePins,
    TypeRegistry,
    Value,
    ValueMeta,
    list_children,
    process_instance_token,
    register_core_types,
)
from dinkster_workers import (
    PACK_AUTHOR_API_CONTRACT,
    PACK_HOST_CONTRACT,
    PACK_INFERENCE_CONTRACT,
    ArmWorker,
    BoundaryDiagnostic,
    BubblewrapCapability,
    BubblewrapLauncher,
    DeviceMap,
    DiagnosticListener,
    DispatchWorker,
    EdgeCost,
    GenerationProvider,
    GroupIsolatedWorker,
    GroupMemberWorker,
    HeadroomMirror,
    InProcessWorker,
    IsolatedWorker,
    PackManifest,
    ReleaseGuard,
    RemoteWorker,
    RoutingWorker,
    SandboxPolicy,
    ValueStore,
    VisionProvider,
    WorkGroupCoordinator,
    WorkGroupWorkerLane,
    compose_workgroup_configuration,
    detect_bubblewrap,
    egress_origin_from_url,
    load_manifest,
    normalize_egress_origin,
    resident_devices,
    unmatched_registry_providers,
)
from dinkster_workers.catalog import (
    PackCatalog,
    read_catalog,
    source_digest,
    worker_declarations_match_catalog,
)
from dinkster_workers.host import load_pack as load_host_pack
from dinkster_workers.session import WorkerDied

from .extension_assets import read_module, resolve_frontend_modules
from .lazy_worker import CatalogTypeRegistry, LazyWorker
from .native_policy import NativeDispatchPolicy, select_resident_producer
from .packs import pack_info_from_manifest
from .remotes import RemoteSpec

T = TypeVar("T")

__all__ = [
    "Composition",
    "CompositionError",
    "PackDelta",
    "PackSpec",
    "ReloadResult",
    "RemoteReattachResult",
    "ServingComposer",
    "UnknownPackError",
    "cuda_vram_budgets",
    "compose_serving",
    "default_pack_ids",
    "default_pack_spec",
    "default_pack_specs",
    "model_pack_specs",
    "resolve_manifest_path",
    "training_pack_specs",
]

FRONTEND_API_VERSION = "1.0.0"

MODEL_FAMILY_REGISTRY = "dinkster.model-families"
SAMPLER_REGISTRY = "dinkster.samplers"
SCHEDULER_REGISTRY = "dinkster.schedulers"

ExecutionCacheMode = Literal["memory", "layered"]

_SANDBOX_RO_PATH_ENV = (
    "DINKSTER_ASSET_ROOT",
    "DINKSTER_ASSET_VAULT",
    "DINKSTER_COMFYUI_ROOT",
    "DINKSTER_MOUNTS_SNAPSHOT",
    "DINKSTER_REMOTE_AUTH_TOKEN_FILE",
)
_SANDBOX_RO_PATH_LIST_ENV = ("DINKSTER_LEGACY_PACKS",)
_SANDBOX_RW_FILE_ENV = ("DINKSTER_TRAINING_JOURNAL",)
_SANDBOX_RW_FILE_URI_PARENT_ENV = ("DINKSTER_SINGLE_JOB_RENDEZVOUS",)
_PACK_SCRATCH_ENV = "DINKSTER_PACK_SCRATCH"
_ATTENTION_POLICY_ENV = "DINKSTER_ATTENTION_POLICY"


_FIRST_PARTY_PACK_MODULES = MappingProxyType(
    {
        "dinkster-nodes-foundation": "dinkster_nodes_foundation",
        "dinkster-nodes-media-io": "dinkster_nodes_media_io",
        "dinkster-nodes-image": "dinkster_nodes_image",
        "dinkster-nodes-remote": "dinkster_nodes_remote",
        "dinkster-nodes-generation": "dinkster_nodes_generation",
        "dinkster-model-qwen-image": "dinkster_model_qwen_image",
        "dinkster-model-triposplat": "dinkster_model_triposplat",
        "dinkster-model-wan": "dinkster_model_wan",
        "dinkster-vision-birefnet": "dinkster_nodes_vision.birefnet",
        "dinkster-vision-depth-anything-v2": "dinkster_nodes_vision.depth_anything_v2",
        "dinkster-vision-depth-anything-v3": "dinkster_nodes_vision.depth_anything_v3",
        "dinkster-vision-detr": "dinkster_nodes_vision.detr",
        "dinkster-vision-efficient-sam": "dinkster_nodes_vision.efficient_sam",
        "dinkster-vision-hed": "dinkster_nodes_vision.hed",
        "dinkster-vision-rtdetr": "dinkster_nodes_vision.rtdetr",
        "dinkster-vision-sam31": "dinkster_nodes_vision.sam31",
        "dinkster-vision-upscale": "dinkster_nodes_vision.upscale",
    }
)
_FIRST_PARTY_PACK_DISTRIBUTIONS = MappingProxyType(
    {
        pack_id: "dinkster-nodes-vision"
        for pack_id in _FIRST_PARTY_PACK_MODULES
        if pack_id.startswith("dinkster-vision-")
    }
)
_ISOLATED_FIRST_PARTY_PACKS = frozenset(
    pack_id for pack_id in _FIRST_PARTY_PACK_MODULES if pack_id.startswith("dinkster-vision-")
)
_PACK_ARTIFACT_SIDECARS = ("comfy-aliases.json", "comfy-groups.json")
_DEFAULT_SUITE_DISTRIBUTION = "dinkster-nodes-std"
_DEFAULT_SUITE_LOCK = "dinkster_nodes_std_suite/dinkster.lock"
_MODEL_PACK_IDS = (
    "dinkster-model-qwen-image",
    "dinkster-model-triposplat",
    "dinkster-model-wan",
)
SAMPLING_WORKER_NAME = "dinkster.ksampler"
"""Native worker whose arm materializes every inference extension surface."""
INFERENCE_UNAVAILABLE_REASON = (
    "no live native dinkster.ksampler worker was composed (the native inference "
    "pack did not load), so this pack's samplers, schedulers, graph compilers and "
    "guidance strategies cannot run"
)


def _trusted_reserved_claims_can_overlap(first: str, second: str) -> bool:
    root = reserved_root(first)
    return root is not None and reserved_root(second) == root


class CompositionError(Exception):
    """Pack composition is misconfigured: reserved/duplicate pack names or
    node-type collisions. Raised at startup, never mid-workflow."""


class UnknownPackError(CompositionError):
    """A reload named a pack this composer never composed."""


@dataclass(frozen=True)
class _ReplicaLane:
    cuda_index: int
    worker: IsolatedWorker | LazyWorker


class _ReplicaWorkerPool:
    """One homogeneous pack worker process per parent-visible CUDA device."""

    def __init__(self, lanes: Sequence[_ReplicaLane]) -> None:
        self.lanes = tuple(lanes)
        if len(self.lanes) < 2:
            raise ValueError("a replica worker pool requires at least two lanes")

    @property
    def _first(self) -> IsolatedWorker | LazyWorker:
        return self.lanes[0].worker

    @property
    def catalog(self) -> PackCatalog | None:
        return self._first.catalog if isinstance(self._first, LazyWorker) else None

    @property
    def cold(self) -> bool:
        return all(isinstance(lane.worker, LazyWorker) and lane.worker.cold for lane in self.lanes)

    def validate_schema(self, node_type: str) -> None:
        for lane in self.lanes:
            if isinstance(lane.worker, LazyWorker):
                lane.worker.validate_schema(node_type)

    async def ensure_started(self) -> None:
        await asyncio.gather(
            *(
                lane.worker.ensure_started()
                for lane in self.lanes
                if isinstance(lane.worker, LazyWorker)
            )
        )

    @property
    def alive(self) -> bool:
        return all(lane.worker.alive for lane in self.lanes)

    @property
    def instance_token(self) -> str | None:
        return self._first.instance_token

    @property
    def attention_route_token(self) -> AttentionRouteToken | None:
        return cast("AttentionRouteToken | None", self._first.attention_route_token)

    @property
    def attention_capabilities(self) -> AttentionCapabilityEvidence | None:
        return cast("AttentionCapabilityEvidence | None", self._first.attention_capabilities)

    @property
    def schemas(self) -> Mapping[str, NodeSchema]:
        return self._first.schemas

    @property
    def combo_choices(self) -> Mapping[str, tuple[str, ...]]:
        return self._first.combo_choices

    @property
    def lazy_choice_ids(self) -> tuple[str, ...]:
        return self._first.lazy_choice_ids

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        return await self._first.fetch_choices(choice_id)

    async def call_pack_route(
        self, route: PackRoute, data: Mapping[str, object]
    ) -> dict[str, object]:
        return await self._first.call_pack_route(route, data)

    @property
    def compat_skips(self) -> Mapping[str, CompatGateDiagnostic]:
        return self._first.compat_skips

    @property
    def body_arms(self) -> Mapping[str, tuple[str, ...]] | None:
        return self._first.body_arms

    @property
    def extension_contributions(self) -> object:
        return self._first.extension_contributions

    @property
    def renditions(self) -> object:
        return getattr(self._first, "renditions", ())

    async def resolve_rendition(
        self,
        type_id: str,
        kind: str,
        metadata: Mapping[str, object],
        parameters: Mapping[str, str],
    ) -> tuple[str, Mapping[str, str]]:
        return await self._first.resolve_rendition(type_id, kind, metadata, parameters)

    async def resolve_rendition_mime(
        self, type_id: str, kind: str, metadata: Mapping[str, object]
    ) -> str:
        return await self._first.resolve_rendition_mime(type_id, kind, metadata)

    async def render_rendition(
        self, value: Value, kind: str, parameters: Mapping[str, str]
    ) -> Rendition:
        return await self._first.render_rendition(value, kind, parameters)

    @property
    def can_convert_legacy_checkpoint(self) -> bool:
        return self._first.can_convert_legacy_checkpoint

    async def start(self) -> None:
        results = await asyncio.gather(
            *(lane.worker.start() for lane in self.lanes), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            await self._close_lanes()
            raise failures[0]
        first = self._first
        for lane in self.lanes[1:]:
            worker = lane.worker
            if (
                dict(worker.schemas) != dict(first.schemas)
                or dict(worker.combo_choices) != dict(first.combo_choices)
                or worker.lazy_choice_ids != first.lazy_choice_ids
                or dict(worker.compat_skips) != dict(first.compat_skips)
                or worker.body_arms != first.body_arms
                or worker.extension_contributions != first.extension_contributions
                or getattr(worker, "renditions", ()) != getattr(first, "renditions", ())
                or worker.attention_capabilities != first.attention_capabilities
                or worker.attention_route_token != first.attention_route_token
            ):
                await self._close_lanes()
                raise RuntimeError("replica workers announced different pack surfaces")

    async def _close_lanes(self) -> None:
        await asyncio.gather(*(lane.worker.close() for lane in self.lanes), return_exceptions=True)

    async def close(self) -> None:
        await self._close_lanes()

    async def materialize_inference_generation(self, key: str) -> object:
        results = await asyncio.gather(
            *(lane.worker.materialize_inference_generation(key) for lane in self.lanes)
        )
        if any(result != results[0] for result in results[1:]):
            raise RuntimeError("replica workers materialized different inference generations")
        return results[0]

    async def release_inference_generation(self, key: str) -> None:
        results = await asyncio.gather(
            *(lane.worker.release_inference_generation(key) for lane in self.lanes),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]

    async def compile_graph(
        self, generation_key: str, graph: Mapping[str, object], targets: Sequence[str]
    ) -> Mapping[str, object]:
        return await self._first.compile_graph(generation_key, graph, targets)

    async def convert_legacy_checkpoint(
        self, path: Path, logical_name: str
    ) -> tuple[str, str | None] | None:
        return await self._first.convert_legacy_checkpoint(path, logical_name)


class _SingleJobWorkerPool(_ReplicaWorkerPool):
    """Fan one native invocation across fixed, process-isolated CUDA ranks."""

    def __init__(
        self,
        lanes: Sequence[_ReplicaLane],
        rendezvous_path: Path,
        mode: str,
        reservations: ReservationService | None = None,
        lane_factory: Callable[[], Sequence[_ReplicaLane]] | None = None,
        rendezvous_directory: Path | None = None,
    ) -> None:
        super().__init__(lanes)
        self._rank_resources: dict[str, tuple[Value, ...]] = {}
        self._rendezvous_path = rendezvous_path
        self._rendezvous_directory = rendezvous_directory
        self._reservations = reservations
        self._lane_factory = lane_factory
        self._failed_lanes = False
        self._permanently_closed = False
        self._recovery_token = f"single-job-recovery-{uuid.uuid4().hex}"
        self._invoke_lock = asyncio.Lock()
        self.mode = mode

    @property
    def alive(self) -> bool:
        return not self._permanently_closed and (
            super().alive or (self._failed_lanes and self._lane_factory is not None)
        )

    @property
    def instance_token(self) -> str | None:
        if super().alive:
            return super().instance_token
        if self.alive:
            return self._recovery_token
        return None

    async def close(self) -> None:
        self._permanently_closed = True
        await super().close()
        self._rendezvous_path.unlink(missing_ok=True)
        if self._rendezvous_directory is not None:
            shutil.rmtree(self._rendezvous_directory, ignore_errors=True)

    async def _close_failed_lanes(self) -> None:
        await self._close_lanes()
        self._rendezvous_path.unlink(missing_ok=True)
        self._failed_lanes = True

    async def _replace_failed_lanes(self) -> None:
        if self._lane_factory is None:
            return
        self.lanes = tuple(self._lane_factory())
        self._rank_resources.clear()
        await self.start()
        self._failed_lanes = False

    def _rank_value(self, value: Value, rank: int) -> Value:
        children = list_children(value)
        if children is not None:
            replaced = tuple(self._rank_value(child, rank) for child in children)
            if replaced != children:
                payload = value.payload
                assert isinstance(payload, ListPayload)
                value = replace(value, payload=replace(payload, children=replaced))
        resource_id = value.meta.get(RESOURCE_ID_META_KEY)
        if isinstance(resource_id, str):
            values = self._rank_resources.get(resource_id)
            if values is not None:
                return values[rank]
        return value

    async def prepare(self, node_types: Sequence[str]) -> None:
        await asyncio.gather(*(lane.worker.prepare(node_types) for lane in self.lanes))

    async def _invoke_ranks(
        self,
        invocations: tuple[Invocation, ...],
        on_event: OnInvocationEvent | None,
        primary_failure: asyncio.Future[str] | None = None,
    ) -> tuple[InvocationResult, ...]:
        tasks = {
            asyncio.create_task(
                lane.worker.invoke(invocation, on_event=on_event if rank == 0 else None)
            ): rank
            for rank, (lane, invocation) in enumerate(zip(self.lanes, invocations, strict=True))
        }
        results: list[InvocationResult | None] = [None] * len(tasks)
        try:
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    rank = tasks.pop(task)
                    result = task.result()
                    results[rank] = result
                    if result.error is not None:
                        if primary_failure is not None and not primary_failure.done():
                            primary_failure.set_result("rank")
                        pending_tasks = tuple(tasks)
                        for pending in pending_tasks:
                            pending.cancel()
                        await self._close_failed_lanes()
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        return tuple(result if item is None else item for item in results)
        except BaseException as exc:
            if (
                not isinstance(exc, asyncio.CancelledError)
                and primary_failure is not None
                and not primary_failure.done()
            ):
                primary_failure.set_result("rank")
            pending_tasks = tuple(tasks)
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            await self._close_failed_lanes()
            raise
        return cast("tuple[InvocationResult, ...]", tuple(results))

    @staticmethod
    def _workgroup_conclusion_error(
        lifecycle: WorkGroupLifecycle,
        unexplained: str,
    ) -> RuntimeError:
        if lifecycle.failures:
            reasons = "; ".join(f"{unit.value}: {reason}" for unit, reason in lifecycle.failures)
            return RuntimeError(f"workgroup concluded with rank failures: {reasons}")
        if lifecycle.refusals:
            reasons = "; ".join(
                f"{replica.value}: {reason}" for replica, reason in lifecycle.refusals
            )
            return RuntimeError(f"workgroup refused: {reasons}")
        return RuntimeError(f"{unexplained} (state {lifecycle.state.value})")

    async def _invoke_workgroup(
        self,
        invocations: tuple[Invocation, ...],
        on_event: OnInvocationEvent | None,
    ) -> tuple[InvocationResult, ...]:
        if self._reservations is None:
            raise RuntimeError("single-job multi-GPU execution requires reservation service")
        if any(
            WORKGROUP_DATA_PLANE_CAPABILITY not in lane.worker.workgroup_capabilities
            for lane in self.lanes
        ):
            raise RuntimeError("single-job rank did not negotiate the workgroup v2 capability")
        group = WorkGroupId(f"single-{uuid.uuid4().hex}")
        attempt = WorkGroupAttempt(1)
        recipe = ReplicaRecipeId(
            "sha256:"
            + hashlib.sha256(f"single-job:{self.mode}:{len(self.lanes)}".encode()).hexdigest()
        )
        workgroup_lanes = tuple(
            WorkGroupWorkerLane(
                ReplicaId(f"rank-{rank}"),
                WorkerInstanceId(cast("str", lane.worker.instance_token)),
                DeviceResourceId(f"cuda-{lane.cuda_index}"),
                recipe,
                WorkUnitId(f"sample-{rank}"),
                SemanticSlot.SINGLE,
                lane.worker,
            )
            for rank, lane in enumerate(self.lanes)
        )
        definition, endpoints = compose_workgroup_configuration(
            group,
            attempt,
            workgroup_lanes,
        )
        reservation_requests = self._reservation_requests(invocations)
        started = asyncio.Event()
        primary_failure: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def coordinator_failed() -> None:
            if not primary_failure.done():
                primary_failure.set_result("coordinator")

        coordinator = asyncio.create_task(
            WorkGroupCoordinator(self._reservations).execute(
                definition,
                endpoints,
                reservation_requests,
                started=started,
                on_failure=coordinator_failed,
            )
        )
        workgroup_start = asyncio.create_task(started.wait())
        rank_invocations: asyncio.Task[tuple[InvocationResult, ...]] | None = None
        try:
            done, _ = await asyncio.wait(
                (coordinator, workgroup_start),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if coordinator in done:
                lifecycle = await coordinator
                raise self._workgroup_conclusion_error(
                    lifecycle, "workgroup ended before rank dispatch"
                )
            await workgroup_start
            rank_invocations = asyncio.create_task(
                self._invoke_ranks(
                    invocations,
                    on_event,
                    primary_failure,
                )
            )
            done, _ = await asyncio.wait(
                (coordinator, rank_invocations, primary_failure),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if primary_failure in done and primary_failure.result() == "rank":
                results = await rank_invocations
                coordinator.cancel()
                await asyncio.gather(coordinator, return_exceptions=True)
                return results
            if primary_failure in done:
                try:
                    lifecycle = await coordinator
                finally:
                    rank_invocations.cancel()
                    await self._close_failed_lanes()
                    await asyncio.gather(rank_invocations, return_exceptions=True)
                raise self._workgroup_conclusion_error(
                    lifecycle, "workgroup failed without coordinator error"
                )
            if rank_invocations not in done:
                lifecycle = await coordinator
                if lifecycle.state is not WorkGroupState.SUCCEEDED:
                    raise self._workgroup_conclusion_error(
                        lifecycle, "workgroup ended before rank invocations"
                    )
            results = await rank_invocations
            if any(result.error is not None for result in results):
                coordinator.cancel()
                await asyncio.gather(coordinator, return_exceptions=True)
                return results
            await coordinator
            return results
        except BaseException:
            coordinator.cancel()
            if rank_invocations is not None:
                rank_invocations.cancel()
            await asyncio.gather(
                coordinator,
                *(task for task in (rank_invocations,) if task is not None),
                return_exceptions=True,
            )
            if started.is_set():
                await self._close_failed_lanes()
            raise
        finally:
            primary_failure.cancel()
            workgroup_start.cancel()
            await asyncio.gather(workgroup_start, return_exceptions=True)
            for lane, member in zip(self.lanes, definition.members, strict=True):
                lane.worker.unbind_workgroup_endpoint(definition, member.replica)

    @staticmethod
    def _reservation_requests(
        invocations: tuple[Invocation, ...],
    ) -> tuple[ReservationRequest, ...]:
        return tuple(
            request for invocation in invocations for request in plan_reservations(invocation)
        )

    def _record_rank_values(self, values: tuple[Value, ...]) -> None:
        children = tuple(list_children(value) for value in values)
        if any(items is None for items in children) != all(items is None for items in children):
            raise RuntimeError("single-job ranks returned different list value types")
        if all(items is not None for items in children):
            lengths = {len(cast("tuple[Value, ...]", items)) for items in children}
            if len(lengths) != 1:
                raise RuntimeError("single-job ranks returned different list value shapes")
            for rank_children in zip(
                *(cast("tuple[Value, ...]", items) for items in children), strict=True
            ):
                self._record_rank_values(rank_children)
        resource_id = values[0].meta.get(RESOURCE_ID_META_KEY)
        if isinstance(resource_id, str):
            if any(
                not isinstance(value.meta.get(RESOURCE_ID_META_KEY), str) for value in values[1:]
            ):
                raise RuntimeError("single-job ranks returned different resource value types")
            self._rank_resources[resource_id] = values

    @classmethod
    def _parent_value(cls, values: tuple[Value, ...]) -> Value:
        leader = values[0]
        children = tuple(list_children(value) for value in values)
        if all(items is not None for items in children):
            leader_children = cast("tuple[Value, ...]", children[0])
            merged_children = tuple(
                cls._parent_value(
                    tuple(cast("tuple[Value, ...]", items)[index] for items in children)
                )
                for index in range(len(leader_children))
            )
            if merged_children != leader_children:
                payload = leader.payload
                assert isinstance(payload, ListPayload)
                leader = replace(leader, payload=replace(payload, children=merged_children))

        resources: dict[str, list[str]] = {}
        for value in values:
            declared = value.meta.get(RESOURCES_META_KEY)
            if not isinstance(declared, Mapping):
                continue
            for kind, instances in cast("Mapping[str, object]", declared).items():
                if not isinstance(kind, str):
                    continue
                candidates = (
                    (instances,)
                    if isinstance(instances, str)
                    else instances
                    if isinstance(instances, (tuple, list))
                    else ()
                )
                selected = resources.setdefault(kind, [])
                for instance in candidates:
                    if isinstance(instance, str) and instance not in selected:
                        selected.append(instance)
        resources = {kind: instances for kind, instances in resources.items() if instances}
        if not resources:
            return leader
        entries = dict(leader.meta.entries)
        entries[RESOURCES_META_KEY] = {
            kind: instances[0] if len(instances) == 1 else tuple(instances)
            for kind, instances in resources.items()
        }
        return replace(leader, meta=ValueMeta(entries))

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        async with self._invoke_lock:
            if self._failed_lanes:
                await self._replace_failed_lanes()
            invocations = tuple(
                replace(
                    invocation,
                    inputs={
                        name: self._rank_value(value, rank)
                        for name, value in invocation.inputs.items()
                    },
                )
                for rank in range(len(self.lanes))
            )
            try:
                results = await self._invoke_workgroup(invocations, on_event)
            except BaseException:
                if self._failed_lanes:
                    await self._replace_failed_lanes()
                raise
            for result in results:
                if result.error is not None:
                    await self._replace_failed_lanes()
                    return result
            outputs = results[0].outputs
            if outputs is None:
                return results[0]
            if any(
                result.outputs is None or result.outputs.keys() != outputs.keys()
                for result in results
            ):
                raise RuntimeError("single-job ranks returned different output surfaces")
            parent_outputs: dict[str, Value] = {}
            for name in outputs:
                values = tuple(
                    cast("Mapping[str, Value]", result.outputs)[name] for result in results
                )
                self._record_rank_values(values)
                parent_outputs[name] = self._parent_value(values)
            return replace(results[0], outputs=parent_outputs)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        return await self._first.check_lazy_status(invocation, on_event=on_event)


def _format_edge_costs(label: str, costs: Sequence[EdgeCost]) -> str:
    parts: list[str] = []
    for edge in costs:
        note = ""
        if edge.reused:
            note = " reused"
        elif not edge.declared_codec:
            note = " FALLBACK-CODEC"
        if edge.transport == "persistentCas":
            note += f" moved={edge.network_bytes}B" if edge.network_bytes else " store-hit"
        parts.append(
            f"{edge.edge_id}={edge.type_id} {edge.size_bytes}B "
            f"{edge.transport} {edge.codec_ms:.1f}ms{note}"
        )
    return f" {label} " + ", ".join(parts) if parts else ""


def log_boundary_diagnostic(diagnostic: BoundaryDiagnostic) -> None:
    """Dev-mode consumer for the boundary's per-invocation cost breakdown
    (DESIGN 3.9): one structured log line per crossing - execute vs
    boundary time, per-edge transport/size/codec cost, and a loud marker
    on fallback-codec crossings (the 'declare a codec to fix' signal)."""
    core_logger("dev.boundary").info(
        "%s (%s, pack %s): execute %.1fms, boundary %.1fms;%s%s",
        diagnostic.node_id,
        diagnostic.node_type,
        diagnostic.pack,
        diagnostic.execute_ms,
        diagnostic.boundary_ms,
        _format_edge_costs("in", diagnostic.inputs),
        _format_edge_costs("out", diagnostic.outputs),
    )


def resolve_manifest_path(path: Path | str) -> Path:
    """Accept a ``dinkster-pack.toml`` or the directory containing one."""
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "dinkster-pack.toml"
    if not resolved.is_file():
        raise CompositionError(f"pack manifest not found: {resolved}")
    return resolved


_BUILTIN_GENERATION_PROVIDER = "builtin"


def _provider_arms(
    arms: Sequence[ArmRecord],
    preferred_worker: str | None,
    remote_names: Collection[str],
) -> tuple[ArmRecord, ...]:
    if preferred_worker is None:
        return tuple(arm for arm in arms if arm.remote is None or arm.name == arm.remote)
    if preferred_worker in remote_names:
        return tuple(arm for arm in arms if arm.remote == preferred_worker)
    return tuple(arm for arm in arms if arm.remote is None)


def _select_vision_provider(
    node_type: str,
    model: str,
    arms: Sequence[ArmRecord],
    routes: Mapping[tuple[str, str], str],
    models: Mapping[tuple[str, str], str | None],
    requested_provider: str | None = None,
) -> tuple[str, str]:
    candidates: list[tuple[bool, bool, str, str]] = []
    annotated_targets: set[str] = set()
    for arm in arms:
        if arm.vision_provider is None:
            continue
        annotated_targets.add(arm.name)
        if requested_provider is not None and arm.vision_provider != requested_provider:
            continue
        if model != "auto" and arm.vision_model is not None and arm.vision_model != model:
            continue
        if not arm.available:
            continue
        candidates.append(
            (
                not arm.provider_declared,
                arm.remote is not None,
                arm.vision_provider,
                arm.name,
            )
        )
    for (managed_type, provider_id), target in routes.items():
        if managed_type != node_type:
            continue
        if requested_provider is not None and provider_id != requested_provider:
            continue
        if target in annotated_targets:
            continue
        if model != "auto" and models.get((node_type, provider_id)) != model:
            continue
        arm = next((candidate for candidate in arms if candidate.name == target), None)
        if arm is None:
            continue
        if not arm.available:
            continue
        candidates.append((False, arm.remote is not None, provider_id, target))
    if candidates:
        _inferred, _remote, selected_provider, target = min(candidates)
        return selected_provider, target
    requested = "an automatic model" if model == "auto" else f"model {model!r}"
    raise ValueError(
        f"{node_type} has no compatible live vision implementation for {requested}; "
        "install or connect a compatible vision pack"
    )


def cuda_vram_budgets(budgets: Mapping[str, int]) -> dict[str, int]:
    """Project global governor budgets into the compat worker namespace."""
    prefix = "vram:cuda:"
    return {
        residency: nbytes
        for residency, nbytes in budgets.items()
        if residency.startswith(prefix) and residency.removeprefix(prefix).isdigit()
    }


@dataclass(frozen=True)
class PackSpec:
    """How one pack worker launches and attributes, beyond its manifest.

    The plain-``Path`` case (host interpreter, manifest-name attribution)
    needs none of this; a spec exists for packs like the ComfyUI compat
    workers, which run on a foreign interpreter with an environment
    contract and attribute one worker's nodes to several provenance
    entries (``comfy.<pack>`` per legacy pack).

    ``packs`` are the provenance-table entries this worker contributes and
    ``attribute`` maps each announced node type onto one of them; the
    defaults are the manifest's own name/presentation and constant
    attribution, i.e. exactly the plain-path behavior. Attributing to an
    id outside ``packs`` is a CompositionError - provenance is host
    configuration and fails loudly, never a silent fallback.
    """

    manifest: Path | str
    python: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    aimdo: str = "off"
    vram_budgets: Mapping[str, int] = field(default_factory=dict)
    reserve_vram: int | None = None
    comfy_args: tuple[str, ...] = ()
    runtime_settings: bool = False
    """Refresh worker policy and ComfyUI argv from effective settings on reload."""
    require_catalog: bool = False
    """Refuse missing or stale installed declarations instead of importing at boot."""
    start_timeout: float = 60.0
    packs: Mapping[str, PackInfo] | None = None
    attribute: Callable[[str], str] | None = None
    trust_reserved: bool = False
    """Allow this pack's manifest to claim reserved namespace roots
    (``std``, ``comfy``, ``core``, ``dinkster``). The local analog of the
    registry's grant table: manifests only claim, and a reserved claim
    composes only because the HOST vouched for the pack here - first-party
    compat/std wiring sets it, ordinary ``--pack`` entries never do."""
    asset_roots: Mapping[str, Path] = field(default_factory=dict)
    """Artifact roots for packaged asset acquisition, per pack-table id.
    Plain packs default to the manifest directory for the manifest's own
    id; specs that carry their own ``packs`` table (the compat workers'
    legacy entries) supply roots here for entries whose declarations name
    packaged files. An entry without a root still composes - its packaged
    sources just report "not present" at acquisition while remote leads
    keep working."""
    host_types: Callable[[TypeRegistry], None] | None = None
    """Host-side type registrations that accompany this worker's surface.

    A foreign-interpreter worker's value types normally register only
    inside the worker; the host still carries, caches and relays what it
    cannot load (DESIGN 3.2). When the HOST must load or render such
    values - the compat workers' comfy.IMAGE previews - the spec declares
    those registrations here and add_pack applies them to the composed
    registry at its commit point. Must be idempotent (guard with
    ``type_id in registry``): two specs may share one hook, reload
    re-applies specs, and the registry is additive for the process
    lifetime - remove_pack never unregisters a type."""
    extension_config: Mapping[str, object] = field(default_factory=dict)
    """Host-resolved behavior configuration for this pack's extension.

    Float values are accepted at this producer boundary but canonicalized to
    strings before entering BehaviorValue, whose vocabulary deliberately has
    no float. Other non-RPC-clean values are refused before publication.
    """
    execution_config: Mapping[str, str] = field(default_factory=dict)
    """Non-secret settings that rotate this pack's execution cache identity."""
    worker_group: str | None = None
    """Host-only H2 process group name, absent for a solo worker."""
    group_manifests: tuple[Path, ...] = ()
    """All manifests in worker_group, supplied by generation topology."""
    in_process: bool = False
    """Host-only H3 placement; pack code loads in the serving process."""
    optional_execution: tuple[str, ...] = ()
    """Schema-only types retained as metadata when no provider is configured.

    These remain absent from the executable catalog until a provider composes.
    Other schema-only types still require a provider for a complete generation.
    """
    runtime_pins: Mapping[str, str] = field(default_factory=dict)
    """Exact torch/dinkster-aimdo baseline recorded by the install generation."""
    asset_vault_write: bool = False
    """Allow this trusted isolated pack to ingest verified assets into the host vault."""
    replica_cuda_indices: tuple[int, ...] = ()
    """Parent-visible CUDA indices for homogeneous process replicas."""
    single_job_cuda_indices: tuple[int, ...] = ()
    """Parent-visible CUDA indices for one process-isolated sampling workgroup."""
    single_job_mode: str = "auto"

    def __post_init__(self) -> None:
        if self.aimdo not in ("off", "auto", "on"):
            raise ValueError(f"PackSpec aimdo must be 'off', 'auto', or 'on', got {self.aimdo!r}")
        if self.replica_cuda_indices and (
            len(self.replica_cuda_indices) < 2
            or any(type(index) is not int or index < 0 for index in self.replica_cuda_indices)
            or len(set(self.replica_cuda_indices)) != len(self.replica_cuda_indices)
        ):
            raise ValueError(
                "PackSpec replica_cuda_indices must contain at least two unique non-negative ints"
            )
        if self.single_job_cuda_indices and (
            len(self.single_job_cuda_indices) < 2
            or any(type(index) is not int or index < 0 for index in self.single_job_cuda_indices)
            or len(set(self.single_job_cuda_indices)) != len(self.single_job_cuda_indices)
        ):
            raise ValueError(
                "PackSpec single_job_cuda_indices must contain at least two unique "
                "non-negative ints"
            )
        if self.replica_cuda_indices and self.single_job_cuda_indices:
            raise ValueError("replica and single-job CUDA indices are mutually exclusive")
        if self.in_process and self.asset_vault_write:
            raise ValueError("an in-process PackSpec cannot request asset vault write access")
        if self.single_job_mode not in ("auto", "guidance", "sequence", "window"):
            raise ValueError("PackSpec single_job_mode is invalid")
        budgets: dict[str, int] = {}
        for residency, nbytes in self.vram_budgets.items():
            prefix = "vram:cuda:"
            index = residency.removeprefix(prefix)
            if not residency.startswith(prefix) or not index.isdigit():
                raise ValueError(
                    f"PackSpec vram_budgets keys must be vram:cuda:N, got {residency!r}"
                )
            value = int(nbytes)
            if value < 0:
                raise ValueError("PackSpec vram_budgets values must be non-negative")
            budgets[residency] = value
        object.__setattr__(self, "vram_budgets", MappingProxyType(budgets))
        if self.reserve_vram is not None and self.reserve_vram < 0:
            raise ValueError("PackSpec reserve_vram must be non-negative")
        object.__setattr__(
            self,
            "comfy_args",
            validate_comfy_args(self.comfy_args, require_tuple=True),
        )
        object.__setattr__(self, "runtime_pins", MappingProxyType(dict(self.runtime_pins)))
        execution_config = dict(self.execution_config)
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in execution_config.items()
        ):
            raise ValueError("PackSpec execution_config must map non-empty strings to strings")
        object.__setattr__(self, "execution_config", MappingProxyType(execution_config))


@lru_cache(maxsize=1)
def _default_pack_lock() -> Lockfile:
    """Load the independently released standard-suite selection."""
    try:
        distribution = importlib.metadata.distribution(_DEFAULT_SUITE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompositionError(
            f"installed default suite {_DEFAULT_SUITE_DISTRIBUTION!r} is unavailable"
        ) from exc
    path = Path(str(distribution.locate_file(_DEFAULT_SUITE_LOCK)))
    if not path.is_file():
        raise CompositionError(
            f"installed default suite {_DEFAULT_SUITE_DISTRIBUTION!r} has no {_DEFAULT_SUITE_LOCK}"
        )
    try:
        lock = Lockfile.from_record_json(path.read_text(encoding="utf-8"))
    except (InstallError, OSError) as exc:
        raise CompositionError(f"installed default suite lock is invalid: {exc}") from exc
    if not lock.packs:
        raise CompositionError("installed default suite lock selects no packs")
    unsupported = tuple(
        entry.pack for entry in lock.packs if entry.pack not in _FIRST_PARTY_PACK_MODULES
    )
    if unsupported:
        raise CompositionError(
            f"installed default suite lock selects unsupported packs {unsupported!r}"
        )
    unexpected_publishers = tuple(
        entry.pack for entry in lock.packs if entry.publisher != "dinkster"
    )
    if unexpected_publishers:
        raise CompositionError(
            "installed default suite lock contains non-Dinkster publishers for "
            f"{unexpected_publishers!r}"
        )
    return lock


def default_pack_ids() -> tuple[str, ...]:
    """Return the locked first-party pack ids in composition order."""
    return tuple(entry.pack for entry in _default_pack_lock().packs)


def _installed_pack_digest(manifest: Path, module_root: Path | None) -> str:
    """Digest source and wheel installations as the same pack artifact."""
    with tempfile.TemporaryDirectory(prefix="dinkster-default-pack-") as directory:
        temporary = Path(directory)
        root = manifest.parent
        if module_root is not None:
            root = temporary / "artifact"
            root.mkdir()
            shutil.copy2(manifest, root / "dinkster-pack.toml")
            if module_root.parent.name == "dinkster_nodes_vision":
                namespace = root / module_root.parent.name
                namespace.mkdir()
                bundled_namespace = manifest.parent / module_root.parent.name
                shutil.copy2(bundled_namespace / "__init__.py", namespace / "__init__.py")
                shutil.copytree(module_root, namespace / module_root.name)
                for sidecar in manifest.parent.iterdir():
                    if sidecar.is_file() and sidecar != manifest:
                        target = root / sidecar.name
                        shutil.copy2(sidecar, target)
                        if sidecar.name.endswith("_LICENSE"):
                            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
            else:
                shutil.copytree(module_root, root / module_root.name)
            for filename in _PACK_ARTIFACT_SIDECARS:
                sidecar = manifest.parent / filename
                if sidecar.is_file():
                    shutil.copy2(sidecar, root / filename)
            pack = tomllib.loads(manifest.read_text(encoding="utf-8")).get("pack", {})
            docs = pack.get("docs", {}) if isinstance(pack, dict) else {}
            docs_dir = docs.get("dir") if isinstance(docs, dict) else None
            if isinstance(docs_dir, str):
                relative_docs = Path(docs_dir)
                source_docs = manifest.parent / relative_docs
                if (
                    not relative_docs.is_absolute()
                    and ".." not in relative_docs.parts
                    and source_docs.is_symlink()
                ):
                    raise ArtifactError(f"artifact contains symlink: {relative_docs}")
                if (
                    not relative_docs.is_absolute()
                    and ".." not in relative_docs.parts
                    and source_docs.is_dir()
                    and source_docs.resolve().is_relative_to(manifest.parent.resolve())
                ):
                    target_docs = root / relative_docs
                    target_docs.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_docs, target_docs, symlinks=True)
            source_locales = manifest.parent / "locales"
            if source_locales.is_symlink():
                raise ArtifactError("artifact contains symlink: locales")
            if source_locales.is_dir():
                shutil.copytree(source_locales, root / "locales", symlinks=True)
        return build_artifact(root, temporary / "pack.zip")


def _installed_pack_spec(
    pack_id: str,
    module_name: str,
    *,
    in_process: bool,
    env: Mapping[str, str] | None = None,
    locked: LockedPack | None = None,
    asset_vault_write: bool = False,
) -> PackSpec:
    """Resolve one installed first-party pack and its exact provenance."""
    distribution_id = _FIRST_PARTY_PACK_DISTRIBUTIONS.get(pack_id, pack_id)
    try:
        distribution = importlib.metadata.distribution(distribution_id)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompositionError(
            f"installed pack distribution {distribution_id!r} is unavailable"
        ) from exc
    sidecar = pack_id.replace("-", "_")
    bundled = Path(str(distribution.locate_file(f"{sidecar}_pack/dinkster-pack.toml")))
    module = importlib.util.find_spec(module_name)
    if module is None or module.origin is None:
        raise CompositionError(f"installed {pack_id} package is not discoverable")
    source_manifest = next(
        (
            parent / "dinkster-pack.toml"
            for parent in Path(module.origin).resolve().parents
            if (parent / "dinkster-pack.toml").is_file()
        ),
        None,
    )
    if source_manifest is None:
        source_manifest = next(
            (
                parent / f"{sidecar}_pack" / "dinkster-pack.toml"
                for parent in Path(module.origin).resolve().parents
                if (parent / "pyproject.toml").is_file()
                and (parent / f"{sidecar}_pack" / "dinkster-pack.toml").is_file()
            ),
            None,
        )
    if source_manifest is not None:
        manifest = source_manifest
        source = f"local:{manifest.parent.resolve()}"
        source_install = True
    elif bundled.is_file():
        manifest = bundled
        source = f"python:{distribution_id}=={distribution.version}"
        source_install = False
    else:
        raise CompositionError(f"installed {pack_id} has no bundled artifact or source manifest")
    module_root = (
        Path(module.origin).resolve().parent if source_install and locked is not None else None
    )
    digest = _installed_pack_digest(manifest, module_root)
    manifest_record = load_manifest(manifest)
    if canonical_name(manifest_record.name) != canonical_name(pack_id):
        raise CompositionError(f"installed {pack_id} manifest names pack {manifest_record.name!r}")
    if locked is not None:
        if distribution.version != locked.version:
            raise CompositionError(
                f"default suite locks {pack_id} version {locked.version}, "
                f"but installed version is {distribution.version}"
            )
        if digest != locked.artifact_digest:
            raise CompositionError(
                f"default suite locks {pack_id} artifact {locked.artifact_digest}, "
                f"but installed artifact is {digest}"
            )
        if tuple(canonical_name(claim) for claim in manifest_record.namespaces) != locked.claims:
            raise CompositionError(
                f"default suite locks {pack_id} claims {locked.claims!r}, "
                f"but its manifest claims {manifest_record.namespaces!r}"
            )
        if locked.publisher != "dinkster":
            raise CompositionError(
                f"default suite pack {pack_id} has unexpected publisher {locked.publisher!r}"
            )
        source = locked.source
    manifest_info = pack_info_from_manifest(manifest_record)
    info = PackInfo(
        display_name=pack_id,
        version=distribution.version,
        artifact_digest=digest,
        source=source,
        publisher="dinkster",
        assets=manifest_record.assets,
        docs=manifest_info.docs,
        locale_catalogs=manifest_info.locale_catalogs,
        templates=manifest_info.templates,
        comfy_aliases=manifest_record.comfy_aliases,
        comfy_groups=manifest_record.comfy_groups,
    )
    return PackSpec(
        manifest=manifest,
        packs=MappingProxyType({pack_id: info}),
        trust_reserved=True,
        in_process=in_process,
        env={} if env is None else env,
        asset_vault_write=asset_vault_write,
    )


def default_pack_spec(pack_id: str) -> PackSpec:
    """Resolve one installed first-party pack and its exact provenance."""
    module_name = _FIRST_PARTY_PACK_MODULES.get(pack_id)
    if module_name is None:
        raise CompositionError(f"unknown first-party pack {pack_id!r}")
    return _installed_pack_spec(
        pack_id,
        module_name,
        in_process=(
            pack_id not in _ISOLATED_FIRST_PARTY_PACKS and pack_id != "dinkster-nodes-remote"
        ),
        env=(
            {"DINKSTER_REMOTE_CATALOG_BASE": "", "DINKSTER_REMOTE_GATEWAY_BASE": ""}
            if pack_id == "dinkster-nodes-remote"
            else None
        ),
        locked=_default_pack_lock().get(pack_id),
        asset_vault_write=pack_id == "dinkster-nodes-remote",
    )


@lru_cache(maxsize=1)
def default_pack_specs() -> tuple[PackSpec, ...]:
    """Resolve the complete installed first-party pack set in dependency order."""
    specs = tuple(default_pack_spec(pack_id) for pack_id in default_pack_ids())
    entries: dict[str, PackContractInput] = {}
    by_name: dict[str, PackSpec] = {}
    for spec in specs:
        manifest = load_manifest(resolve_manifest_path(spec.manifest))
        name = canonical_name(manifest.name)
        entries[name] = (manifest, spec)
        by_name[name] = spec
    order, _receipts = _resolve_pack_contracts(entries, _builtin_registry_providers())
    return tuple(by_name[name] for name in order)


def model_pack_specs() -> tuple[PackSpec, ...]:
    """Resolve the installed first-party model packs in stable order."""
    return tuple(default_pack_spec(pack_id) for pack_id in _MODEL_PACK_IDS)


def training_pack_specs(journal_path: Path | str) -> tuple[PackSpec, PackSpec]:
    """Resolve the in-process schemas and isolated training executor."""
    journal = str(Path(journal_path).resolve())
    return (
        _installed_pack_spec(
            "dinkster-nodes-training",
            "dinkster_nodes_training",
            in_process=True,
        ),
        _installed_pack_spec(
            "dinkster-training-worker",
            "dinkster_training_worker",
            in_process=False,
            env={
                "DINKSTER_TRAINING_BACKEND": "fake",
                "DINKSTER_TRAINING_JOURNAL": journal,
            },
        ),
    )


def _worker_group_contract(spec: PackSpec, worker_env: Mapping[str, str]) -> tuple[object, ...]:
    environment = {**worker_env, **spec.env}
    environment.pop(_ATTENTION_POLICY_ENV, None)
    return (
        spec.python,
        tuple(sorted(environment.items())),
        spec.aimdo,
        tuple(sorted(spec.vram_budgets.items())),
        spec.reserve_vram,
        spec.comfy_args,
        spec.start_timeout,
        spec.asset_vault_write,
        tuple(Path(path) for path in spec.group_manifests),
    )


def _validate_worker_group_specs(
    entries: Sequence[PackSpec | Path | str], worker_env: Mapping[str, str]
) -> None:
    contracts: dict[str, tuple[object, ...]] = {}
    for entry in entries:
        if (
            isinstance(entry, PackSpec)
            and entry.in_process
            and (
                entry.worker_group is not None or entry.group_manifests or entry.python is not None
            )
        ):
            raise CompositionError(
                "an in-process PackSpec cannot set worker_group, group_manifests, or python"
            )
        if not isinstance(entry, PackSpec) or entry.worker_group is None:
            continue
        contract = _worker_group_contract(entry, worker_env)
        previous = contracts.setdefault(entry.worker_group, contract)
        if previous != contract:
            raise CompositionError(
                f"worker group {entry.worker_group!r} members have incompatible launch contracts"
            )


@dataclass
class Composition:
    """Everything a server host needs, built by :func:`compose_serving`.

    ``schemas``/``packs``/``node_packs`` and the host-attributed authority
    maps feed create_app; ``make_engine`` is its engine factory. Call
    :meth:`close` on shutdown to reap the pack processes (an aiohttp
    ``on_cleanup`` hook is the natural place).
    """

    schemas: dict[str, NodeSchema]
    packs: dict[str, PackInfo]
    node_packs: dict[str, str]
    schema_owners: dict[str, str]
    choice_owners: dict[str, str]
    execution_arms: dict[str, tuple[ExecutionArm, ...]]
    _registry: TypeRegistry
    _worker: Worker
    _plan_execution: Callable[
        [str, NodeSchema, Mapping[str, Value], str],
        Awaitable[ExecutionSelection | None],
    ]
    _owner_alive: Callable[[str], bool]
    _pin_execution: Callable[[], ExecutionRuntime]
    cache_mode: ExecutionCacheMode = "memory"
    cache_memory_entries: int = 1024
    cache_dir: Path | None = None
    cache_disk_budget: int = DEFAULT_DISK_CACHE_BYTES
    generation: CompositionGeneration = field(
        default_factory=lambda: CompositionGeneration.of("development", [])
    )
    """Canonical pack-set provenance. It never participates in native numerical identity."""
    _isolated: list[Any] = field(default_factory=list)
    _tenant_registries: list[Any] = field(default_factory=list)
    _component_publishers: list[NativeComponentPublisher] = field(default_factory=list)
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Combo choice lists across the composed surface (choice-list id ->
    values), for create_app's /api/choices routes."""
    lazy_choices: dict[str, LazyChoiceFetcher] = field(default_factory=dict)
    """Lazy combo choice fetchers across the composed surface (choice-list
    id -> per-request fetcher bound to the owning worker), for create_app's
    /api/choices routes. Invoked only when the route is fetched, never at
    compose time."""
    compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = field(default_factory=dict)
    """Classified compat translation skips grouped by attributed pack id,
    for create_app's /api/diagnostics response."""
    dev: bool = False
    explain_misses: bool = False
    asset_catalog: PackAssetCatalog = field(default_factory=PackAssetCatalog)
    """Pack-declared assets on the composed surface ([[pack.assets]]):
    pack artifact roots plus declared needs, maintained across add,
    reload, and remove. Hosts hand it to ServerLibrary so job preflight
    can resolve declared digests through packaged and remote sources -
    always verified, always behind digest-exact consent."""
    _cleanup_paths: list[Path] = field(default_factory=list)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _engine_caches: weakref.WeakSet[MemoryLRUCache | LayeredCache] = field(
        default_factory=weakref.WeakSet, init=False, repr=False
    )
    _resource_pins: ResourcePins = field(default_factory=ResourcePins, init=False, repr=False)

    def make_engine(self, on_event: EventListener) -> Engine:
        runtime = self._pin_execution()
        memory = MemoryLRUCache(self.cache_memory_entries)
        cache: MemoryLRUCache | LayeredCache = memory
        if self.cache_mode == "layered":
            assert self.cache_dir is not None
            cache = LayeredCache(
                memory,
                DiskCacheStore(
                    self.cache_dir,
                    self._registry,
                    max_bytes=self.cache_disk_budget,
                ),
            )
        pins = self._resource_pins
        pool = default_pool()
        pool.register_invalidator(cache.drop_referencing)
        pool.register_pins(pins)
        self._engine_caches.add(cache)
        return Engine(
            schemas=self.schemas,
            registry=self._registry,
            worker=runtime.worker,
            cache=cache,
            pins=pins,
            on_event=on_event,
            explain_misses=self.dev or self.explain_misses,
            plan_execution=runtime.plan_execution,
            run_finished=runtime.run_finished,
            owner_alive=runtime.owner_alive,
            extension_snapshot=runtime.extension_snapshot,
            pin_execution=self._pin_execution,
        )

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError as cancelled:
            while not self._close_task.done():
                try:
                    await asyncio.shield(self._close_task)
                except asyncio.CancelledError:
                    continue
            await self._close_task
            raise cancelled

    async def _close(self) -> None:
        registries = tuple(self._tenant_registries)
        publishers = tuple(self._component_publishers)
        workers = tuple(self._isolated)
        self._tenant_registries.clear()
        self._component_publishers.clear()
        self._isolated.clear()
        lazy_results = await asyncio.gather(
            *(worker.close() for worker in workers if isinstance(worker, LazyWorker)),
            return_exceptions=True,
        )
        registry_results = await asyncio.gather(
            *(registry.close() for registry in registries), return_exceptions=True
        )
        publisher_results: list[BaseException | None] = []
        for publisher in publishers:
            try:
                publisher.close()
            except BaseException as exc:
                publisher_results.append(exc)
            else:
                publisher_results.append(None)
        worker_results = await asyncio.gather(
            *(worker.close() for worker in workers if not isinstance(worker, LazyWorker)),
            return_exceptions=True,
        )
        for path in self._cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
        self._cleanup_paths.clear()
        errors = [
            result
            for result in (*lazy_results, *publisher_results, *registry_results, *worker_results)
            if isinstance(result, BaseException)
        ]
        if errors:
            raise errors[0]


def _merge_pack_entry(
    packs: dict[str, PackInfo], pack_id: str, info: PackInfo, source: str
) -> None:
    """Table entries may repeat across specs only when identical (two compat
    workers both contributing the shared "comfy" entry); a conflicting
    redefinition is misconfiguration."""
    if pack_id == CORE_PACK_ID:
        raise CompositionError(f"{source}: pack id {CORE_PACK_ID!r} is reserved")
    existing = packs.get(pack_id)
    if existing is not None and existing != info:
        raise CompositionError(
            f"{source}: pack id {pack_id!r} already declared with different info"
        )
    candidate = {**packs, pack_id: info}
    registries = {
        existing_id: existing_info.comfy_aliases
        for existing_id, existing_info in candidate.items()
        if existing_info.comfy_aliases is not None
    }
    collisions = comfy_alias_collision_problems(registries)
    if collisions:
        raise CompositionError(f"{source}: {collisions[0]}")
    group_registries = {
        existing_id: existing_info.comfy_groups
        for existing_id, existing_info in candidate.items()
        if existing_info.comfy_groups is not None
    }
    group_collisions = comfy_group_collision_problems(group_registries)
    if group_collisions:
        raise CompositionError(f"{source}: {group_collisions[0]}")
    packs[pack_id] = info


def _validate_registry_carriers(
    owner: str,
    spec_packs: Mapping[str, PackInfo],
    node_packs: Mapping[str, str],
) -> None:
    """Every comfy alias/group record carrier must be a node attributed to
    the pack entry announcing the registry.

    This is the compose-time net for provider-gated carriers: the served
    surface withholds schema-only types until an execution provider
    composes, so the server can only enforce carrier ownership once the
    carrier publishes. Here ``node_packs`` carries the pack's complete
    declared attribution (no publication gating), so a registry naming a
    node the pack never declares is refused before any worker serves it.
    """
    for pack_id, info in spec_packs.items():
        for label, registry in (("alias", info.comfy_aliases), ("group", info.comfy_groups)):
            if registry is None:
                continue
            for record in registry.records:
                if node_packs.get(record.carrier) != pack_id:
                    raise CompositionError(
                        f"{owner}: comfy {label} record {record.id!r} carrier "
                        f"{record.carrier!r} is not declared by pack {pack_id!r}"
                    )


def _validate_remote_authority(
    owner: str,
    schemas: Mapping[str, NodeSchema],
    choices: Mapping[str, Sequence[str]],
    lazy_ids: Collection[str] = (),
) -> None:
    try:
        for choice_id, values in choices.items():
            combo_choices_json_bytes(
                values,
                subject=f"{owner} choice {choice_id!r}",
            )
        validate_remote_choice_authority(
            schemas, {**choices, **dict.fromkeys(lazy_ids, ())}, owner=owner
        )
    except ValueError as exc:
        raise CompositionError(str(exc)) from exc


def _lazy_choice_fetcher(
    worker: Any, choice_id: str, composer: ServingComposer
) -> LazyChoiceFetcher:
    """Bind one announced lazy choice id to its owning worker session.

    A dead owner surfaces as :class:`ChoiceOwnerGone` so the server can
    answer 503 without knowing about worker session types."""

    async def fetch() -> Sequence[str]:
        try:
            async with composer._mutate:
                return await worker.fetch_choices(choice_id)
        except WorkerDied as exc:
            raise ChoiceOwnerGone(f"choice list {choice_id!r} owner is not connected") from exc

    return fetch


def _validated_remote_body_arms(
    name: str,
    worker: Any,
    announced: Mapping[str, NodeSchema],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Check the hello's bodyArms are internally coherent and return them in
    deterministic order. The daemon's manifest is its own authority for which
    arms exist (it validated them against [pack.arms] before announcing);
    this side only requires that every arm node type is announced and that a
    native arm carries attention route evidence, mirroring the local
    _validate_body_arms evidence rule."""
    body_arms = worker.body_arms or {}
    if "native" in body_arms and worker.attention_route_token is None:
        raise CompositionError(
            f"remote worker {name!r}: native bodyArms hello omitted attention route evidence"
        )
    for arm_name, node_types in body_arms.items():
        unknown = sorted(set(node_types) - set(announced))
        if unknown:
            raise CompositionError(
                f"remote worker {name!r}: bodyArms arm {arm_name!r} names node types "
                f"not announced in schemas: {', '.join(unknown)}"
            )
    return tuple(sorted((arm, tuple(sorted(types))) for arm, types in body_arms.items()))


@dataclass(frozen=True)
class PackDelta:
    """What one pack's announcement adds to the serving surface - exactly
    the arguments ServerState.announce takes, so a progressive host can
    forward each delta to a live server verbatim."""

    pack: str
    schemas: dict[str, NodeSchema]
    packs: dict[str, PackInfo]
    node_packs: dict[str, str]
    schema_owners: dict[str, str] = field(default_factory=dict)
    """Host-attributed schema authority, including retained schema-only owners."""
    choice_owners: dict[str, str] = field(default_factory=dict)
    """Host-attributed authority for authored, lazy, and derived choice updates."""
    execution_arms: dict[str, tuple[ExecutionArm, ...]] = field(default_factory=dict)
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Combo choice lists the pack announced (choice-list id -> values),
    served behind /api/choices/{id}. Exclusively owned per pack, like
    node types; UI vocabulary, never identity."""
    lazy_choices: dict[str, LazyChoiceFetcher] = field(default_factory=dict)
    """Lazy combo choice fetchers the pack announced (choice-list id ->
    per-request fetcher bound to the owning worker). Same exclusive
    ownership as ``choices``; the two tables never share an id."""
    derived_choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Choice replacements derived from the composed surface.

    These ride live publication separately from pack-authored ``choices``
    so they retain the authority of the original static registration rather
    than the pack whose addition happened to change their values."""
    compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = field(default_factory=dict)
    """Classified compat skips grouped by attributed pack id."""


@dataclass(frozen=True)
class ReloadResult:
    """What one pack's reload swaps on the serving surface - exactly the
    arguments ServerState.replace takes. Uniform remove-then-add:
    ``removed_types`` are all previously routable affected node types and
    ``removed_packs`` ALL of its exclusive pack-table ids (shared entries
    like the compat workers' "comfy" stay); ``delta`` re-announces the
    routable survivors. Schema-only definitions stay internal until a body
    composes, so they never create an unroutable public node. Types dropped
    from the serving surface are ``removed_types`` minus ``delta.schemas``."""

    pack: str
    removed_types: tuple[str, ...]
    removed_packs: tuple[str, ...]
    delta: PackDelta
    removed_choices: tuple[str, ...] = ()
    """ALL of the old version's choice-list ids; ``delta.choices``
    re-announces the survivors (the same uniform remove-then-add as
    node types)."""
    removed_compat_skips: tuple[tuple[str, str], ...] = ()
    """Every (pack id, source node name) skip owned by the old worker."""
    reloaded_packs: tuple[str, ...] = ()
    """Every pack replaced by this transaction; empty means only ``pack``."""


@dataclass(frozen=True)
class RemoteReattachResult:
    """What one remote's session replacement swaps on the serving surface,
    plus whether the daemon process survived. A changed instance token
    means the daemon restarted: engine result caches fingerprint schema
    signature + inputs, never implementation (hazard H4), so the caller
    should clear them - the reload precedent. An unchanged instance is a
    transport-only reconnect and cached results stay valid."""

    result: ReloadResult
    same_instance: bool


@dataclass(frozen=True)
class RemoveResult:
    """What one pack's removal retracts from the serving surface - the
    removal-only case of :class:`ReloadResult`: its routable node types,
    any schema-only types that lose their last provider, and its exclusive
    pack-table ids (shared entries like the compat workers' "comfy" stay
    while another worker still declares them), with no incoming delta."""

    pack: str
    removed_types: tuple[str, ...]
    removed_packs: tuple[str, ...]
    execution_arms: dict[str, tuple[ExecutionArm, ...]] = field(default_factory=dict)
    removed_choices: tuple[str, ...] = ()
    """All of the pack's choice-list ids (exclusively owned, so every
    one retracts)."""
    derived_choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Choice replacements derived after this removal."""
    choice_owners: dict[str, str] = field(default_factory=dict)
    """Retained authority for every derived choice replacement."""
    removed_compat_skips: tuple[tuple[str, str], ...] = ()
    """Every (pack id, source node name) skip owned by the removed worker."""
    packs: dict[str, PackInfo] = field(default_factory=dict)
    """Pack-table rows this removal changed on other packs. Removing the native
    inference pack degrades every remaining pack that declares an inference
    entry, so those rows must ride the removal announcement."""


@dataclass(frozen=True, eq=False)
class _ResidencyDomain:
    """Opaque identity and liveness seat for one worker session."""

    worker: Any = field(compare=False)
    label: str

    @property
    def default_arm(self) -> str:
        """Compatibility name for diagnostics; dispatch defaults live on arms."""
        return self.label


@dataclass(frozen=True)
class ArmRecord:
    """One dispatch arm in an immutable composer topology snapshot."""

    name: str
    worker: Any
    domain: _ResidencyDomain
    instance_token: Callable[[], str | None]
    default_cache_tag: str
    default_arm: str
    owner_worker: Any
    execution_arm: ExecutionArm = "native"
    attention_route_token: AttentionRouteToken | None = None
    attention_capabilities: AttentionCapabilityEvidence | None = None
    remote: str | None = None
    """Owning remote worker name; None for a local arm."""
    implementation_pack: str | None = None
    vision_provider: str | None = None
    vision_provider_choice: str | None = None
    vision_model: str | None = None
    generation_provider: str | None = None
    generation_provider_choice: str | None = None
    generation_provider_label: str | None = None
    provider_declared: bool = True
    """False only for an older remote whose compatible pack identity is inferred."""

    @property
    def execution_worker(self) -> Any:
        return self.worker.worker if isinstance(self.worker, ArmWorker) else self.owner_worker

    @property
    def available(self) -> bool:
        return getattr(self.execution_worker, "cold", False) or (
            self.execution_worker.alive and self.instance_token() is not None
        )


Topology = Mapping[str, tuple[ArmRecord, ...]]


@dataclass(frozen=True)
class _PackRecord:
    """Composer-side bookkeeping for one composed pack: everything a
    reload needs to start a fresh worker from the same spec and retract
    exactly what the old worker contributed."""

    spec: PackSpec
    worker: Any
    delta: PackDelta
    claims: tuple[str, ...]
    canonical: str
    manifest: PackManifest
    executes: tuple[str, ...] = ()
    schema_only: tuple[str, ...] = ()
    default_cache_tag: str = "unversioned"
    body_arms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    domain: _ResidencyDomain | None = None
    asset_roots: Mapping[str, Path] = field(default_factory=dict)
    """Artifact root per pack-table id this record contributes, for
    packaged asset acquisition (spec-declared, or the manifest directory
    for the manifest's own id)."""
    extension: ExtensionDeclaration | None = None
    extension_contributions: tuple[tuple[ExtensionScope, ContributionSurfaceDescriptor], ...] = ()
    tenant_registrations: _PackTenantRegistry | None = None
    owns_tenant_registrations: bool = True
    component_publisher: NativeComponentPublisher | None = None
    owns_component_publisher: bool = True


PackContractInput = tuple[PackManifest, PackSpec]


def _pack_release(manifest: PackManifest, spec: PackSpec) -> tuple[str, str]:
    """Return the manifest pack's release version and artifact digest when pinned."""
    if spec.packs is None:
        return "", ""
    info = spec.packs.get(manifest.name) or spec.packs.get(canonical_name(manifest.name))
    if info is None:
        return "", ""
    return info.version, info.artifact_digest


def _pack_provider_identity(manifest: PackManifest, spec: PackSpec) -> str:
    version, digest = _pack_release(manifest, spec)
    identity = canonical_name(manifest.name)
    if version:
        identity += f"@{version}"
    if digest:
        identity += f"#{digest}"
    return identity


def _builtin_registry_providers() -> dict[str, dict[str, str]]:
    provider = PACK_INFERENCE_CONTRACT
    registries = builtin_registries()
    return {
        canonical_name(MODEL_FAMILY_REGISTRY): {
            canonical_name(id_): provider for id_ in registries.families.ids()
        },
        canonical_name(SAMPLER_REGISTRY): {
            canonical_name(item.id): provider for item in builtin_sampler_snapshot().samplers
        },
        canonical_name(SCHEDULER_REGISTRY): {
            canonical_name(item.id): provider for item in registries.schedulers
        },
    }


def _resolve_pack_contracts(
    entries: Mapping[str, PackContractInput],
    registry_providers: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[str, ...], dict[str, tuple[ResolvedRequirement, ...]]]:
    """Validate one complete pack set and return dependency order plus exact receipts."""
    capabilities: dict[str, tuple[str, str]] = {}
    dependencies: dict[str, set[str]] = {}
    receipts: dict[str, list[ResolvedRequirement]] = {name: [] for name in entries}
    providers = {registry: dict(items) for registry, items in registry_providers.items()}
    provider_packs: dict[tuple[str, str], str] = {}

    for name in sorted(entries):
        manifest, spec = entries[name]
        contracts = manifest.contracts
        if contracts is not None:
            expected: dict[RequirementKind, str] = {
                "host": PACK_HOST_CONTRACT,
                "api": PACK_AUTHOR_API_CONTRACT,
                "inference": PACK_INFERENCE_CONTRACT,
            }
            declared: dict[RequirementKind, str | None] = {
                "host": contracts.host,
                "api": contracts.api,
                "inference": contracts.inference,
            }
            for kind, value in declared.items():
                if value is None:
                    continue
                if value != expected[kind]:
                    raise CompositionError(
                        f"pack {manifest.name!r} requires {kind} contract {value!r}, "
                        f"but this host provides {expected[kind]!r}"
                    )
                receipts[name].append(ResolvedRequirement(name, kind, value, expected[kind]))
        dependencies[name] = {dependency.pack for dependency in manifest.dependencies}
        for provider in manifest.provides.registry:
            registry_id = canonical_name(provider.registry)
            descriptor_id = canonical_name(provider.id)
            registry = providers.setdefault(registry_id, {})
            previous = registry.get(descriptor_id)
            if previous is not None:
                raise CompositionError(
                    f"registry descriptor {provider.registry}:{provider.id} is provided by both "
                    f"{previous!r} and {manifest.name!r}"
                )
            registry[descriptor_id] = _pack_provider_identity(manifest, spec)
            provider_packs[(registry_id, descriptor_id)] = name
        for capability in manifest.capabilities:
            key = canonical_name(capability.id)
            previous = capabilities.get(key)
            if previous is not None:
                previous_pack, _ = previous
                raise CompositionError(
                    f"capability {capability.id!r} is provided by both "
                    f"{previous_pack!r} and {manifest.name!r}"
                )
            capabilities[key] = (name, capability.version)

    for name in sorted(entries):
        manifest, _spec = entries[name]
        for dependency in manifest.dependencies:
            provider = entries.get(dependency.pack)
            if provider is None:
                raise CompositionError(
                    f"pack {manifest.name!r} requires pack {dependency.pack!r} "
                    f"{dependency.version}, but it is not composed"
                )
            provider_manifest, provider_spec = provider
            version, _digest = _pack_release(provider_manifest, provider_spec)
            if not version:
                raise CompositionError(
                    f"pack {manifest.name!r} requires versioned pack {dependency.pack!r}, "
                    "but the provider is an unversioned development pack"
                )
            if not dependency.accepts(version):
                raise CompositionError(
                    f"pack {manifest.name!r} requires {dependency.pack!r} "
                    f"{dependency.version}, but the composed version is {version}"
                )
            receipts[name].append(
                ResolvedRequirement(
                    name,
                    "pack",
                    f"{dependency.pack}{dependency.version}",
                    _pack_provider_identity(provider_manifest, provider_spec),
                )
            )

        for requirement in manifest.requirements.registry:
            registry_id = canonical_name(requirement.registry)
            descriptor_id = canonical_name(requirement.id)
            registry = providers.get(registry_id)
            provider = registry.get(descriptor_id) if registry is not None else None
            if provider is None:
                raise CompositionError(
                    f"pack {manifest.name!r} requires registry descriptor "
                    f"{requirement.registry}:{requirement.id}, but it is unavailable"
                )
            receipts[name].append(
                ResolvedRequirement(
                    name,
                    "registry",
                    f"{requirement.registry}:{requirement.id}",
                    provider,
                )
            )
            provider_pack = provider_packs.get((registry_id, descriptor_id))
            if provider_pack is not None and provider_pack != name:
                dependencies[name].add(provider_pack)

        for requirement in manifest.requirements.capabilities:
            provider = capabilities.get(canonical_name(requirement.id))
            if provider is None:
                raise CompositionError(
                    f"pack {manifest.name!r} requires capability {requirement.id!r} "
                    f"{requirement.version}, but no pack provides it"
                )
            provider_name, version = provider
            if not requirement.accepts(version):
                raise CompositionError(
                    f"pack {manifest.name!r} requires capability {requirement.id!r} "
                    f"{requirement.version}, but {provider_name!r} provides {version}"
                )
            if provider_name != name:
                dependencies[name].add(provider_name)
            provider_manifest, provider_spec = entries[provider_name]
            receipts[name].append(
                ResolvedRequirement(
                    name,
                    "capability",
                    f"{requirement.id}{requirement.version}",
                    f"{_pack_provider_identity(provider_manifest, provider_spec)}:{version}",
                )
            )

    remaining = {name: set(required) for name, required in dependencies.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, required in remaining.items() if not required)
        if not ready:
            cycle = ", ".join(
                f"{name} -> {', '.join(sorted(required))}"
                for name, required in sorted(remaining.items())
            )
            raise CompositionError(f"pack dependency cycle: {cycle}")
        for name in ready:
            order.append(name)
            del remaining[name]
        for required in remaining.values():
            required.difference_update(ready)
    return tuple(order), {name: tuple(receipts[name]) for name in sorted(receipts)}


def _composition_generation(
    entries: Mapping[str, PackContractInput],
    receipts: Mapping[str, tuple[ResolvedRequirement, ...]],
    requested_mode: PackCompositionMode,
) -> CompositionGeneration:
    packs = []
    for name, (manifest, spec) in entries.items():
        version, digest = _pack_release(manifest, spec)
        packs.append(ComposedPack(name, version, digest))
    mode = requested_mode
    if mode == "production" and any(not pack.artifact_digest for pack in packs):
        mode = "development"
    try:
        return CompositionGeneration.of(
            mode,
            packs,
            [item for pack_receipts in receipts.values() for item in pack_receipts],
        )
    except CompositionRecordError as exc:
        raise CompositionError(str(exc)) from exc


@dataclass(frozen=True)
class _RemoteSurface:
    """One dialed daemon's hello, validated against the composed surface
    and reduced to exactly what a :class:`_RemoteRecord` needs."""

    composed: dict[str, NodeSchema]
    body_arms: tuple[tuple[str, tuple[str, ...]], ...]
    schemas: dict[str, NodeSchema]
    node_packs: dict[str, str]
    comfy_aliases: ComfyAliasRegistry | None
    comfy_groups: ComfyGroupRegistry | None
    choices: dict[str, tuple[str, ...]]
    lazy_choice_ids: tuple[str, ...]
    compat_skips: dict[str, dict[str, CompatGateDiagnostic]]
    vision_providers: tuple[VisionProvider, ...] | None
    generation_providers: tuple[GenerationProvider, ...] | None


@dataclass(frozen=True)
class _RemoteRecord:
    """Composer-side bookkeeping for one composed remote worker.

    Deliberately parallel to (not inside) ``_records``: every _records
    consumer - pack_specs, watch_targets, reload_pack, the asset catalog,
    extension snapshots - is manifest-shaped, and a remote has no manifest
    and no hot reload."""

    spec: RemoteSpec
    worker: RemoteWorker
    delta: PackDelta
    domain: _ResidencyDomain
    node_types: tuple[str, ...]
    pack_name: str = ""
    body_arms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    vision_providers: tuple[VisionProvider, ...] | None = None
    generation_providers: tuple[GenerationProvider, ...] | None = None
    instance_token: str | None = None
    """The daemon's process-lifetime token at compose time. The session
    clears its own copy on death, so reattach compares against this record
    to tell a daemon restart from a transport-only reconnect."""


class _PackTenantRegistry(ModelTenantRegistry):
    """Pack-attributing proxy that owns cleanup of successful registrations."""

    def __init__(self, pack: str, registry: ModelTenantRegistry) -> None:
        self.pack = pack
        self._registry = registry
        self._registrations: list[TenantRegistration] = []
        self._lock = asyncio.Lock()
        self._closed = False
        self._operations = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def _begin(self) -> None:
        async with self._lock:
            if self._closed:
                raise CompositionError(
                    f"model tenant registry for removed pack {self.pack!r} is closed"
                )
            self._operations += 1
            self._idle.clear()

    async def _end(self) -> None:
        async with self._lock:
            self._operations -= 1
            if self._operations == 0:
                self._idle.set()

    async def register(self, handle: ModelTenantHandle) -> TenantRegistration:
        if handle.pack != self.pack:
            raise CompositionError(
                f"pack {self.pack!r} attempted tenant registration attributed to {handle.pack!r}"
            )
        await self._begin()
        registration: TenantRegistration | None = None
        try:
            registration = await self._registry.register(handle)
        finally:
            await self._end()
        async with self._lock:
            closed = self._closed
            if not closed:
                self._registrations.append(registration)
        if closed:
            await self._registry.unregister(registration)
            raise CompositionError(
                f"model tenant registry for removed pack {self.pack!r} is closed"
            )
        return registration

    async def notify_resident(self, registration: TenantRegistration) -> None:
        """Attribute and forward engine-side residence admission."""
        await self._begin()
        try:
            async with self._lock:
                if registration not in self._registrations:
                    raise CompositionError(
                        f"pack {self.pack!r} attempted residence notification "
                        "for an unowned tenant registration"
                    )
            await self._registry.notify_resident(registration)
        finally:
            await self._end()

    async def unregister(self, registration: TenantRegistration) -> None:
        await self._begin()
        try:
            async with self._lock:
                if registration not in self._registrations:
                    raise CompositionError(
                        f"pack {self.pack!r} attempted to unregister an unowned tenant registration"
                    )
                self._registrations.remove(registration)
            await self._registry.unregister(registration)
        except BaseException:
            async with self._lock:
                if registration not in self._registrations:
                    self._registrations.append(registration)
            raise
        finally:
            await self._end()

    async def terminal_release(self, registration: TenantRegistration) -> None:
        await self._begin()
        try:
            async with self._lock:
                if registration not in self._registrations:
                    raise CompositionError(
                        f"pack {self.pack!r} attempted to release an unowned tenant registration"
                    )
            await self._registry.terminal_release(registration)
            async with self._lock:
                self._registrations.remove(registration)
        finally:
            await self._end()

    async def full_release(self) -> FullReleaseResult:
        """Terminally release every tenant registration owned by this pack."""
        async with self._lock:
            registrations = tuple(reversed(self._registrations))
        failures: list[str] = []
        for registration in registrations:
            try:
                await self.terminal_release(registration)
            except Exception as exc:  # noqa: BLE001 - continue every owned tenant
                failures.append(str(exc) or "tenant terminal release failed")
        if failures:
            return FullReleaseResult("error", "; ".join(failures))
        async with self._lock:
            return FullReleaseResult("busy" if self._registrations else "complete")

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self._idle.wait()
        async with self._lock:
            registrations = tuple(reversed(self._registrations))
            self._registrations.clear()
        for registration in registrations:
            await self._registry.unregister(registration)


def _probe_torch_capability() -> bool:
    """Import torch in the serving process; discoverability alone is not capability."""
    try:
        importlib.import_module("torch")
    except Exception:  # noqa: BLE001 - a broken torch install is not capable
        return False
    return True


def _pack_asset_roots(
    spec: PackSpec, manifest: PackManifest, spec_packs: Mapping[str, PackInfo]
) -> dict[str, Path]:
    """Resolve the artifact root for each pack-table id a spec
    contributes: the spec's explicit declaration first, the manifest
    directory for the manifest's own id, otherwise no root (packaged
    sources for that id report "not present"; remote leads still work)."""
    roots: dict[str, Path] = {}
    for pack_id in spec_packs:
        root = spec.asset_roots.get(pack_id)
        if root is None and pack_id == manifest.name:
            root = manifest.root
        if root is not None:
            roots[pack_id] = Path(root)
    return roots


class _RuntimeSeat:
    """One-reference publication seat shared by a composer and its engines."""

    def __init__(self, runtime: ExecutionRuntime) -> None:
        self._runtime = runtime

    def pin(self) -> ExecutionRuntime:
        return self._runtime

    def publish(self, runtime: ExecutionRuntime) -> None:
        self._runtime = runtime


class ServingComposer:
    """Incremental composition over a healthy zero-node host kernel.

    Two hosts drive it. :func:`compose_serving` runs every pack to
    completion before anything serves (the fixed-surface path tests and
    embedders use). dinkster-serve drives it PROGRESSIVELY: bind the port on
    the empty diagnostic surface first, then :meth:`add_pack` each spec and
    announce the returned delta to the live server. The standard node pack
    follows this same path; it is not boot-critical host code.
    Both paths apply identical validation because they are the same code.

    Failure semantics are unchanged from the all-at-once path: any
    validation or startup failure raises (CompositionError - loudly, never
    a silent precedence pick); workers already composed keep running and
    the caller decides whether that is fatal. add_pack is ATOMIC: a
    failing pack merges nothing - schemas, pack table, name registry,
    claim registry all stay as they were - so a host may record the
    failure and keep composing the rest (dinkster-serve's default), or abort
    (compose_serving, --strict-packs). :meth:`close` reaps every started
    worker; an add_pack candidate that fails after startup is closed before
    the failure escapes.
    """

    def __init__(
        self,
        *,
        worker_env: Mapping[str, str] | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        sandbox_gpu_grants: Iterable[str] = (),
        sandbox_network_grants: Mapping[str, Iterable[str]] | None = None,
        sandbox_writable_mounts: Sequence[str] = (),
        pack_scratch_root: Path | str | None = None,
        dev: bool = False,
        on_diagnostic: DiagnosticListener | None = None,
        explain_misses: bool = False,
        governor: MemoryGovernor | None = None,
        reservations: ReservationService | None = None,
        telemetry: ReportedTelemetry | None = None,
        native_policy: NativeDispatchPolicy | None = None,
        runtime_worker_settings: Callable[[], tuple[str, int, Mapping[str, int], tuple[str, ...]]]
        | None = None,
        headroom_base: int = 256 * 1024**2,
        registry: TypeRegistry | None = None,
        tenant_registry: ModelTenantRegistry | None = None,
        torch_capable: Callable[[], bool] | None = None,
        runtime_versions: Callable[[], Mapping[str, str]] | None = None,
        remote_asset_endpoint: str | None = None,
        remote_asset_token: str | None = None,
        remote_value_store: ValueStore | None = None,
        on_schema_reload: Callable[[str], None] | None = None,
        cache_mode: ExecutionCacheMode = "memory",
        cache_memory_entries: int = 1024,
        cache_dir: Path | str | None = None,
        cache_disk_budget: int = DEFAULT_DISK_CACHE_BYTES,
        composition_mode: PackCompositionMode = "development",
        _private_staging: bool = False,
        _reusable_in_process: Mapping[str, _PackRecord] | None = None,
    ) -> None:
        # Memory governance (DESIGN 3.10): when the host supplies a
        # governor, every isolated worker composes governed - its manifest
        # consumers become relayed shedders, its reservation planner holds
        # governor-backed leases, and the ram-release gate runs parent-side.
        # Declared accounting only: devices without an explicit budget
        # admit every reservation, so a budgetless setup never blocks on
        # a budget nobody declared (status/shedding still work). A
        # telemetry store makes every worker's measured reports land
        # parent-side - informational only, never admission.
        self._governor = governor
        self._dev = dev
        self._explain_misses = explain_misses
        self._headroom_base = headroom_base
        self._headroom_mirror = (
            HeadroomMirror(governor, base_bytes=headroom_base) if governor is not None else None
        )
        self._reservations = reservations
        self._telemetry = telemetry
        self._engine_instance_id = uuid.uuid4().hex
        self._native_policy = native_policy
        self._runtime_worker_settings = runtime_worker_settings
        self._tenant_registry = tenant_registry
        self._torch_capable = torch_capable or _probe_torch_capability
        self._runtime_versions = runtime_versions or (
            lambda: {name: importlib.metadata.version(name) for name in ("torch", "dinkster-aimdo")}
        )
        self._sandbox_policy = sandbox_policy
        self._sandbox_gpu_grants = frozenset(canonical_name(name) for name in sandbox_gpu_grants)
        network_grants: dict[str, list[str]] = {}
        for name, origins in (sandbox_network_grants or {}).items():
            network_grants.setdefault(canonical_name(name), []).extend(
                normalize_egress_origin(origin) for origin in origins
            )
        self._sandbox_network_grants = MappingProxyType(
            {name: tuple(dict.fromkeys(origins)) for name, origins in network_grants.items()}
        )
        self._sandbox_writable_mounts = tuple(sandbox_writable_mounts)
        self._pack_scratch_root = (
            Path(os.path.abspath(str(pack_scratch_root))) if pack_scratch_root is not None else None
        )
        self._sandbox_capability: BubblewrapCapability | None = None
        self._on_diagnostic = on_diagnostic
        diagnostic_sinks: list[DiagnosticListener] = []
        if dev:
            diagnostic_sinks.append(log_boundary_diagnostic)
        if on_diagnostic is not None:
            diagnostic_sinks.append(on_diagnostic)
        self._diagnostic_listener: DiagnosticListener | None = None
        if len(diagnostic_sinks) == 1:
            self._diagnostic_listener = diagnostic_sinks[0]
        elif diagnostic_sinks:

            def fan_out(diagnostic: BoundaryDiagnostic) -> None:
                for sink in diagnostic_sinks:
                    sink(diagnostic)

            self._diagnostic_listener = fan_out
        self._worker_env = dict(worker_env or {})
        # The asset base URL composed remote workers offer their daemons
        # as a staging source (pull-over-HTTP; peer_asset_sources URL
        # shape), plus the bearer credential for it when this engine
        # serves with auth. Engine-global: one advertised surface, every
        # remote hears the same address.
        self._remote_asset_endpoint = remote_asset_endpoint
        self._remote_asset_token = remote_asset_token
        # Engine-side persistent value store for remote workers: bulk
        # boundary values land here once, so re-runs and reconnects send
        # digest references instead of bytes. One store, every remote.
        self._remote_value_store = remote_value_store
        self._on_schema_reload = on_schema_reload
        if cache_mode not in ("memory", "layered"):
            raise ValueError(f"invalid execution cache mode {cache_mode!r}")
        if cache_memory_entries < 1:
            raise ValueError("cache_memory_entries must be >= 1")
        if cache_disk_budget < 1:
            raise ValueError("cache_disk_budget must be >= 1")
        if cache_mode == "layered" and cache_dir is None:
            raise ValueError("layered execution cache requires cache_dir")
        self._cache_mode: ExecutionCacheMode = cache_mode
        self._cache_memory_entries = cache_memory_entries
        self._cache_dir = Path(os.path.abspath(str(cache_dir))) if cache_dir is not None else None
        self._cache_disk_budget = cache_disk_budget
        self._private_staging = _private_staging
        self._reusable_in_process = dict(_reusable_in_process or {})
        if composition_mode not in ("production", "development"):
            raise ValueError(f"invalid composition mode {composition_mode!r}")
        self._composition_mode: PackCompositionMode = composition_mode
        self._registry_providers = _builtin_registry_providers()
        catalog_path = self._worker_env.get(SAMPLER_CATALOG_ENV)
        self._sampler_catalog_root: Path | None = None
        if catalog_path is None:
            self._sampler_catalog_root = Path(tempfile.mkdtemp(prefix="dinkster-sampler-catalog-"))
            catalog_path = str(self._sampler_catalog_root / "catalog.json")
            self._worker_env[SAMPLER_CATALOG_ENV] = catalog_path
        self._sampler_catalog_path = Path(catalog_path)
        self._published_generation_digest: str | None = None
        # Published generations remain loaded for this worker lifetime so
        # runtimes pinned by already-admitted jobs can still resolve them.
        # The set is one entry per digest; close() provides the full cleanup.
        self._retired_generation_digests: set[str] = set()
        registry_was_supplied = registry is not None
        registry = registry or TypeRegistry()
        if not registry_was_supplied:
            register_core_types(registry)
        register_inference_types(registry)
        # The host kernel starts with no nodes; nodes arrive through manifests.
        core_nodes: list[type[Node]] = []
        # The native inference registries are core vocabulary on every
        # surface: /api/choices/dinkster.samplers and .schedulers serve the
        # ported catalog ids (dinkster.euler, dinkster.karras, ...) so remote
        # COMBO widgets and future registerable-sampler packs share one
        # source of truth (stage 3b; DESIGN first-class registries).
        builtin_sampler_view = builtin_sampler_snapshot()
        self._core_choices: dict[str, tuple[str, ...]] = {
            "dinkster.samplers": tuple(d.id for d in builtin_sampler_view.samplers),
            "dinkster.schedulers": tuple(d.id for d in builtin_registries().schedulers),
        }
        self._core_packs: dict[str, PackInfo] = {}
        self._base_registry = registry.copy()
        schemas: dict[str, NodeSchema] = dict(build_schemas(core_nodes))
        _validate_remote_authority(CORE_PACK_ID, schemas, self._core_choices)
        self._core_schemas = dict(schemas)
        self._owners: dict[str, str] = dict.fromkeys(schemas, CORE_PACK_ID)
        self._seen_names: dict[str, str] = {}
        self._claim_owners: dict[str, str] = {}
        # Choice-list ownership mirrors node-type ownership: exclusive per
        # pack, collisions refused at composition. Core choices are owned
        # by the core surface itself.
        self._choice_owners: dict[str, str] = dict.fromkeys(self._core_choices, CORE_PACK_ID)
        self._compat_skip_owners: dict[tuple[str, str], str] = {}
        # Per-pack bookkeeping for reload; keyed by manifest name. Guarded
        # by _mutate: startup's drive task adds packs sequentially, but a
        # reload request can arrive while composition is still in flight.
        self._records: dict[str, _PackRecord] = {}
        # Packs whose [pack.extension] inference surface has no native sampling
        # worker to materialize it. Rebuilt at every commit point from the
        # snapshot builder, so reload, remote attach and removal all recompute it
        # against the topology that actually exists at that moment.
        self._inference_unavailable: dict[str, PackInferenceUnavailable] = {}
        # Composed remote workers, keyed by their configured name. Parallel
        # to _records on purpose: remotes have no manifest and no reload.
        self._remotes: dict[str, _RemoteRecord] = {}
        self._group_owners: dict[str, GroupIsolatedWorker] = {}
        self._group_activations: dict[GroupIsolatedWorker, asyncio.Task[None]] = {}
        self._group_contracts: dict[str, tuple[object, ...]] = {}
        self._in_process_domain: _ResidencyDomain | None = None
        self._topology: Topology = MappingProxyType({})
        self._mutate = asyncio.Lock()
        self._publication = asyncio.Lock()
        # Always a RoutingWorker (even before any pack routes exist) so the
        # surface can grow behind the one Worker the engine already holds.
        self._core_worker = InProcessWorker(build_node_types(core_nodes), registry)
        self._core_domain = _ResidencyDomain(self._core_worker, "local")
        self._routing = RoutingWorker({}, default=self._core_worker)

        async def plan_core(
            _node_id: str,
            node_type: str,
            schema: NodeSchema,
            inputs: Mapping[str, Value],
            run_id: str,
            attention_config: AttentionPolicyConfig | None,
        ) -> ExecutionSelection | None:
            return await self._plan_execution(
                node_type,
                schema,
                inputs,
                run_id,
                attention_config,
            )

        empty_runtime = ExecutionRuntime(
            worker=self._routing,
            plan_execution=plan_core,
            owner_alive=self._owner_alive,
            extension_snapshot=ExtensionSnapshot(frontend_api=FRONTEND_API_VERSION),
            sampler_registry_snapshot=builtin_sampler_view,
            schemas=MappingProxyType(dict(schemas)),
        )
        self._runtime_seat = _RuntimeSeat(empty_runtime)
        self.composition = Composition(
            schemas=schemas,
            packs=dict(self._core_packs),
            choices=dict(self._core_choices),
            compat_skips={},
            node_packs={},
            schema_owners=self._owners,
            choice_owners=self._choice_owners,
            execution_arms={node_type: ("native",) for node_type in schemas},
            _registry=registry,
            _worker=self._routing,
            _plan_execution=self._plan_execution,
            _owner_alive=self._owner_alive,
            _pin_execution=self._runtime_seat.pin,
            cache_mode=self._cache_mode,
            cache_memory_entries=self._cache_memory_entries,
            cache_dir=self._cache_dir,
            cache_disk_budget=self._cache_disk_budget,
            generation=CompositionGeneration.of(composition_mode, []),
            _isolated=[],
            dev=dev,
            explain_misses=explain_misses,
        )

    def spawn_empty(
        self, *, composition_mode: PackCompositionMode | None = None
    ) -> ServingComposer:
        """Build an isolated staging composer with this host configuration."""
        worker_env = dict(self._worker_env)
        worker_env.pop(SAMPLER_CATALOG_ENV, None)
        selected_mode: PackCompositionMode = (
            self._composition_mode if composition_mode is None else composition_mode
        )
        return ServingComposer(
            worker_env=worker_env,
            sandbox_policy=self._sandbox_policy,
            sandbox_gpu_grants=self._sandbox_gpu_grants,
            sandbox_network_grants=self._sandbox_network_grants,
            sandbox_writable_mounts=self._sandbox_writable_mounts,
            pack_scratch_root=self._pack_scratch_root,
            dev=self._dev,
            on_diagnostic=self._on_diagnostic,
            explain_misses=self._explain_misses,
            governor=self._governor,
            reservations=self._reservations,
            telemetry=self._telemetry,
            native_policy=self._native_policy,
            runtime_worker_settings=self._runtime_worker_settings,
            headroom_base=self._headroom_base,
            registry=self._base_registry.copy(),
            tenant_registry=self._tenant_registry,
            torch_capable=self._torch_capable,
            runtime_versions=self._runtime_versions,
            on_schema_reload=self._on_schema_reload,
            cache_mode=self._cache_mode,
            cache_memory_entries=self._cache_memory_entries,
            cache_dir=self._cache_dir,
            cache_disk_budget=self._cache_disk_budget,
            composition_mode=selected_mode,
            _private_staging=True,
            _reusable_in_process={
                canonical_name(name): record
                for name, record in self._records.items()
                if record.spec.in_process
            },
        )

    def _scratch_directory(
        self,
        spec: PackSpec,
        manifests: Sequence[PackManifest],
    ) -> Path | None:
        root = self._pack_scratch_root
        if root is None:
            return None
        if spec.worker_group is None:
            if len(manifests) != 1:
                raise CompositionError("a solo worker scratch requires exactly one manifest")
            scope = "packs"
            name = canonical_name(manifests[0].name)
        else:
            scope = "groups"
            name = spec.worker_group
            if (
                not name
                or name[0] in ".-"
                or any(
                    char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                    for char in name
                )
            ):
                raise CompositionError(f"worker group {name!r} is not path-safe")
        scratch = root / scope / name
        for directory in (root, root / scope, scratch):
            try:
                directory.mkdir(mode=0o700, parents=directory == root)
            except FileExistsError:
                pass
            except OSError as exc:
                raise CompositionError(
                    f"cannot create pack scratch directory {directory}: {exc}"
                ) from exc
            if directory.is_symlink() or not directory.is_dir():
                raise CompositionError(
                    f"pack scratch directory {directory} is not a real directory"
                )
            try:
                directory.chmod(0o700)
            except OSError as exc:
                raise CompositionError(
                    f"cannot protect pack scratch directory {directory}: {exc}"
                ) from exc
        return scratch

    def _worker_environment(
        self,
        spec: PackSpec,
        manifests: Sequence[PackManifest],
    ) -> dict[str, str]:
        environment = {**self._worker_env, **spec.env}
        environment.pop(_ATTENTION_POLICY_ENV, None)
        if scratch := self._scratch_directory(spec, manifests):
            environment[_PACK_SCRATCH_ENV] = str(scratch)
        else:
            environment.pop(_PACK_SCRATCH_ENV, None)
        return environment

    def _sandbox_launcher(
        self,
        spec: PackSpec,
        manifests: Sequence[PackManifest],
        env: Mapping[str, str],
    ) -> BubblewrapLauncher | None:
        base = self._sandbox_policy
        if base is None:
            return None

        effective_env = dict(env)
        passthrough = list(base.env_passthrough)
        for name in (*_SANDBOX_RO_PATH_ENV, *_SANDBOX_RO_PATH_LIST_ENV, "PYTHONPATH"):
            if name not in effective_env and (value := os.environ.get(name)):
                effective_env[name] = value
                passthrough.append(name)

        ro_binds = list(base.ro_binds)
        rw_binds = list(base.rw_binds)
        protected_ro_exceptions = list(base.protected_ro_exceptions)
        protected_rw_exceptions = list(base.protected_rw_exceptions)
        if scratch := self._scratch_directory(spec, manifests):
            rw_binds.append(str(scratch))
            protected_rw_exceptions.append(str(scratch))
        ro_binds.extend(str(manifest.root) for manifest in manifests)
        protected_ro_exceptions.extend(str(manifest.root) for manifest in manifests)
        python = Path(spec.python or sys.executable).absolute()
        protected_ro_exceptions.append(str(python.parent.parent))
        for name in _SANDBOX_RO_PATH_ENV:
            if value := effective_env.get(name):
                ro_binds.append(value)
                if name == "DINKSTER_REMOTE_AUTH_TOKEN_FILE":
                    protected_ro_exceptions.append(value)
        if spec.asset_vault_write and (vault_path := effective_env.get("DINKSTER_ASSET_VAULT")):
            ro_binds = [path for path in ro_binds if path != vault_path]
            rw_binds.append(vault_path)
            protected_rw_exceptions.append(vault_path)
        for name in _SANDBOX_RO_PATH_LIST_ENV:
            ro_binds.extend(path for path in effective_env.get(name, "").split(os.pathsep) if path)
        if catalog := effective_env.get(SAMPLER_CATALOG_ENV):
            ro_binds.append(str(Path(catalog).parent))
        for name in _SANDBOX_RW_FILE_ENV:
            if value := effective_env.get(name):
                files = (value, value + "-wal", value + "-shm")
                rw_binds.extend(files)
                protected_rw_exceptions.extend(files)
        for name in _SANDBOX_RW_FILE_URI_PARENT_ENV:
            value = effective_env.get(name, "")
            if value.startswith("file://"):
                rw_binds.append(str(Path(value.removeprefix("file://")).parent))

        gpu_requesters = frozenset(
            canonical_name(manifest.name) for manifest in manifests if manifest.sandbox.gpu
        )
        network_requesters = frozenset(
            canonical_name(manifest.name) for manifest in manifests if manifest.sandbox.network
        )
        if any(manifest.sandbox.writable_mounts for manifest in manifests):
            writable_mounts = {
                os.path.realpath(os.path.abspath(path)) for path in self._sandbox_writable_mounts
            }
            ro_binds = [
                path
                for path in ro_binds
                if os.path.realpath(os.path.abspath(path)) not in writable_mounts
            ]
            rw_binds.extend(self._sandbox_writable_mounts)

        visible_devices = effective_env.get(
            "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
        )
        gpu_host_grant = base.gpu or bool(
            spec.replica_cuda_indices
            or spec.single_job_cuda_indices
            or spec.vram_budgets
            or spec.aimdo != "off"
            or visible_devices not in ("", "-1")
        )
        if not gpu_host_grant and not gpu_requesters.issubset(self._sandbox_gpu_grants):
            denied = ", ".join(sorted(gpu_requesters - self._sandbox_gpu_grants))
            raise CompositionError(f"sandbox GPU access requested but not granted for: {denied}")
        gpu = bool(gpu_requesters) and (
            gpu_host_grant or gpu_requesters.issubset(self._sandbox_gpu_grants)
        )
        global_egress = tuple(
            dict.fromkeys(normalize_egress_origin(origin) for origin in base.egress_allowlist)
        )
        implicit_egress_list: list[str] = []
        if network_requesters:
            for name in (
                "DINKSTER_OPENAI_BASE_URL",
                "DINKSTER_REMOTE_CATALOG_BASE",
                "DINKSTER_REMOTE_GATEWAY_BASE",
            ):
                if api_base := effective_env.get(name):
                    try:
                        implicit_egress_list.append(egress_origin_from_url(api_base))
                    except ValueError as exc:
                        raise CompositionError(f"invalid {name} egress origin: {exc}") from exc
        implicit_egress = tuple(dict.fromkeys(implicit_egress_list))
        missing_network_grants = {
            requester
            for requester in network_requesters
            if not global_egress
            and not implicit_egress
            and not self._sandbox_network_grants.get(requester)
        }
        if missing_network_grants:
            denied = ", ".join(sorted(missing_network_grants))
            raise CompositionError(
                f"sandbox network access requested but not granted for: {denied}"
            )
        egress_allowlist = tuple(
            dict.fromkeys(
                (
                    *global_egress,
                    *implicit_egress,
                    *(
                        origin
                        for requester in sorted(network_requesters)
                        for origin in self._sandbox_network_grants.get(requester, ())
                    ),
                )
            )
        )

        if self._sandbox_capability is None:
            self._sandbox_capability = detect_bubblewrap()
        policy = replace(
            base,
            egress_allowlist=egress_allowlist if network_requesters else (),
            gpu=gpu,
            ro_binds=tuple(dict.fromkeys(ro_binds)),
            rw_binds=tuple(dict.fromkeys(rw_binds)),
            env_passthrough=tuple(dict.fromkeys(passthrough)),
            protected_ro_exceptions=tuple(dict.fromkeys(protected_ro_exceptions)),
            protected_rw_exceptions=tuple(dict.fromkeys(protected_rw_exceptions)),
        )
        return BubblewrapLauncher(policy, capability=self._sandbox_capability)

    def _catalog_group_worker(
        self,
        group_name: str,
        owner: GroupIsolatedWorker,
        manifest: PackManifest,
        catalog: PackCatalog,
    ) -> LazyWorker:
        worker = owner.members[manifest.name]

        async def activate_group() -> None:
            members = [
                record
                for record in self._records.values()
                if record.spec.worker_group == group_name and isinstance(record.worker, LazyWorker)
            ]
            for member in members:
                member.worker.validate_source()
            try:
                await owner.start()
                for member in members:
                    live = owner.members[member.manifest.name]
                    self._validate_body_arms(member.manifest, live)
                    member.worker.adopt(live)
            except BaseException:
                await owner.close()
                raise

        async def start_execution() -> Any:
            if not worker.alive:
                if owner._close_task is not None:
                    raise CompositionError(f"worker group {group_name!r} is closed")
                task = self._group_activations.get(owner)
                if task is None:
                    task = asyncio.create_task(activate_group())
                    self._group_activations[owner] = task
                await asyncio.shield(task)
            self._validate_body_arms(manifest, worker)
            return worker

        async def close_execution() -> None:
            pass  # The composition owns the shared process, not an individual member.

        return LazyWorker(manifest, catalog, start_execution, close_execution)

    def _catalog_worker(
        self, worker: Any, manifest: PackManifest, catalog: PackCatalog | None
    ) -> Any:
        if catalog is None:
            return worker

        async def activate() -> Any:
            await worker.start()
            self._validate_body_arms(manifest, worker)
            return worker

        def declarations_changed() -> None:
            assert self._on_schema_reload is not None
            self._on_schema_reload(manifest.name)

        return LazyWorker(
            manifest,
            catalog,
            activate,
            worker.close,
            declarations_changed
            if manifest.schema_reload_entry is not None and self._on_schema_reload is not None
            else None,
        )

    def _isolated_worker(
        self,
        spec: PackSpec,
        manifest: PackManifest,
        registry: TypeRegistry,
    ) -> Any:
        catalog = read_catalog(manifest) if spec.require_catalog else None
        if spec.require_catalog and catalog is None:
            raise CompositionError(
                f"{manifest.name}: schema catalog is missing or stale; rerun doctor"
            )
        worker_kwargs = {
            "python": spec.python,
            "aimdo_arm": spec.aimdo,
            "vram_budgets": spec.vram_budgets,
            "reserve_vram": spec.reserve_vram,
            "comfy_args": spec.comfy_args,
            "start_timeout": spec.start_timeout,
            "on_diagnostic": self._diagnostic_listener,
            "governor": self._governor,
            "reservations": self._reservations,
            "telemetry": self._telemetry,
            "headroom_mirror": self._headroom_mirror,
            "on_schema_reload": self._schema_reload_requested,
        }
        selected_cuda_indices = spec.replica_cuda_indices or spec.single_job_cuda_indices
        if not selected_cuda_indices:
            worker_env = self._worker_environment(spec, (manifest,))
            worker = IsolatedWorker(
                manifest.path,
                registry,
                extra_env=worker_env,
                launcher=self._sandbox_launcher(spec, (manifest,), worker_env),
                **worker_kwargs,
            )
            return self._catalog_worker(worker, manifest, catalog)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_devices = None if visible is None else tuple(visible.split(","))
        rendezvous_directory: Path | None = None
        rendezvous_path: Path | None = None
        rank_environment: dict[str, str] = {}
        if spec.single_job_cuda_indices:
            rendezvous_directory = Path(tempfile.mkdtemp(prefix="dinkster-single-job-"))
            rendezvous_path = rendezvous_directory / "rendezvous"
            rank_environment = {
                "DINKSTER_SINGLE_JOB_MULTI_GPU_MODE": spec.single_job_mode,
                "DINKSTER_SINGLE_JOB_WORLD_SIZE": str(len(selected_cuda_indices)),
                "DINKSTER_SINGLE_JOB_RENDEZVOUS": f"file://{rendezvous_path}",
                "DINKSTER_SINGLE_JOB_TOKEN": uuid.uuid4().hex,
            }
            if spec.single_job_mode == "sequence":
                rank_environment.update(
                    {
                        "DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES": str(len(selected_cuda_indices)),
                        "DINKSTER_SINGLE_JOB_SEQUENCE_RING": "1",
                        "DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE": "1",
                    }
                )

        worker_env = self._worker_environment(spec, (manifest,))

        def make_lanes() -> tuple[_ReplicaLane, ...]:
            lanes: list[_ReplicaLane] = []
            for rank, cuda_index in enumerate(selected_cuda_indices):
                if visible_devices is not None and cuda_index >= len(visible_devices):
                    raise CompositionError(
                        f"replica CUDA index {cuda_index} is outside CUDA_VISIBLE_DEVICES"
                    )
                child_device = (
                    str(cuda_index) if visible_devices is None else visible_devices[cuda_index]
                )
                lane_env = {
                    **worker_env,
                    "CUDA_VISIBLE_DEVICES": child_device,
                    **rank_environment,
                }
                if spec.single_job_cuda_indices:
                    lane_env["DINKSTER_SINGLE_JOB_RANK"] = str(rank)
                lane = IsolatedWorker(
                    manifest.path,
                    registry,
                    extra_env=lane_env,
                    device_map=DeviceMap({"cuda:0": f"cuda:{cuda_index}"}),
                    launcher=self._sandbox_launcher(spec, (manifest,), lane_env),
                    **worker_kwargs,
                )
                lanes.append(
                    _ReplicaLane(cuda_index, self._catalog_worker(lane, manifest, catalog))
                )
            return tuple(lanes)

        try:
            lanes = make_lanes()
        except BaseException:
            if rendezvous_directory is not None:
                shutil.rmtree(rendezvous_directory, ignore_errors=True)
            raise
        if not spec.single_job_cuda_indices:
            return _ReplicaWorkerPool(lanes)
        return _SingleJobWorkerPool(
            lanes,
            cast("Path", rendezvous_path),
            spec.single_job_mode,
            self._reservations,
            make_lanes,
            rendezvous_directory,
        )

    def _schema_reload_requested(self, pack: str, instance_token: str) -> None:
        record = self._records.get(pack)
        if record is None or self._on_schema_reload is None:
            return
        worker = record.worker
        if isinstance(worker, _ReplicaWorkerPool):
            active = any(lane.worker.instance_token == instance_token for lane in worker.lanes)
        else:
            active = worker.instance_token == instance_token
        if active:
            self._on_schema_reload(pack)

    def validate_specs(self, entries: Sequence[PackSpec | Path | str]) -> None:
        """Fail closed on contradictory group launch contracts before startup."""
        _validate_worker_group_specs(entries, self._worker_env)

    def validate_catalogs(self, entries: Sequence[PackSpec | Path | str]) -> None:
        """Require installed declarations before a serving host binds."""
        for entry in entries:
            if not isinstance(entry, PackSpec) or not entry.require_catalog:
                continue
            paths = (resolve_manifest_path(entry.manifest), *entry.group_manifests)
            for path in paths:
                manifest = load_manifest(path)
                if read_catalog(manifest) is None:
                    raise CompositionError(
                        f"{manifest.name}: schema catalog is missing or stale; "
                        "run dinkster-pack prepare-catalogs "
                        "(or dinkster-pack prepare-catalogs --defaults) before serving"
                    )

    def _contract_inputs(
        self,
        *,
        replacing: tuple[PackManifest, PackSpec] | None = None,
        removing: str | None = None,
    ) -> dict[str, PackContractInput]:
        entries = {
            canonical_name(record.manifest.name): (record.manifest, record.spec)
            for record in self._records.values()
            if record.manifest.name != removing
        }
        if replacing is not None:
            manifest, spec = replacing
            entries[canonical_name(manifest.name)] = (manifest, spec)
        return entries

    def _resolve_contract_inputs(
        self, entries: Mapping[str, PackContractInput]
    ) -> tuple[tuple[str, ...], CompositionGeneration]:
        _composition_generation(entries, {}, "development")
        order, receipts = _resolve_pack_contracts(entries, self._registry_providers)
        return order, _composition_generation(entries, receipts, self._composition_mode)

    def order_pack_entries(self, entries: Sequence[PackSpec | Path | str]) -> tuple[PackSpec, ...]:
        """Resolve a complete candidate set and order providers before consumers."""
        specs = tuple(
            entry if isinstance(entry, PackSpec) else PackSpec(manifest=entry) for entry in entries
        )
        self.validate_specs(specs)
        candidates = self._contract_inputs()
        new: dict[str, PackSpec] = {}
        for spec in specs:
            manifest = load_manifest(resolve_manifest_path(spec.manifest))
            name = canonical_name(manifest.name)
            if name in candidates or name in new:
                raise CompositionError(
                    f"duplicate pack name {manifest.name!r} in candidate set; "
                    "separator-equivalent spellings are one identity"
                )
            candidates[name] = (manifest, spec)
            new[name] = spec
        order, _generation = self._resolve_contract_inputs(candidates)
        return tuple(new[name] for name in order if name in new)

    def adopt(self, staged: ServingComposer) -> Composition:
        """Adopt a fully validated generation without awaiting or rebuilding.

        Returns the previous composition so its workers can be retired after
        the server publishes the new generation.
        """
        old = self.composition
        seat = self._runtime_seat
        mutate = self._mutate
        publication = self._publication
        runtime = staged._runtime_seat.pin()
        old_catalog = old.asset_catalog
        old_catalog_root = self._sampler_catalog_root
        for cache in staged.composition._engine_caches:
            old._engine_caches.add(cache)
        staged.composition._engine_caches = old._engine_caches
        staged.composition._resource_pins = old._resource_pins
        transferred_tenants = {
            staged_record.tenant_registrations
            for name, staged_record in staged._records.items()
            if staged_record.tenant_registrations is not None
            and (current := self._records.get(name)) is not None
            and current.tenant_registrations is staged_record.tenant_registrations
        }
        transferred_publishers = {
            staged_record.component_publisher
            for name, staged_record in staged._records.items()
            if staged_record.component_publisher is not None
            and (current := self._records.get(name)) is not None
            and current.component_publisher is staged_record.component_publisher
        }
        staged._runtime_seat = seat
        staged._mutate = mutate
        staged._publication = publication
        staged.composition._pin_execution = seat.pin
        staged._private_staging = False
        self.__dict__.update(staged.__dict__)
        for tenant in transferred_tenants:
            if tenant in old._tenant_registries:
                old._tenant_registries.remove(tenant)
            if tenant not in self.composition._tenant_registries:
                self.composition._tenant_registries.append(tenant)
        for publisher in transferred_publishers:
            if publisher in old._component_publishers:
                old._component_publishers.remove(publisher)
            if publisher not in self.composition._component_publishers:
                self.composition._component_publishers.append(publisher)
        for name, record in tuple(self._records.items()):
            if record.tenant_registrations in transferred_tenants:
                record = replace(record, owns_tenant_registrations=True)
            if record.component_publisher in transferred_publishers:
                record = replace(record, owns_component_publisher=True)
            self._records[name] = record
        if old_catalog_root is not None:
            old._cleanup_paths.append(old_catalog_root)
        seat.publish(runtime)
        self.composition.asset_catalog = old_catalog
        self._rebuild_asset_catalog()
        return old

    @asynccontextmanager
    async def generation_transaction(self) -> AsyncIterator[None]:
        """Exclude incremental pack mutations while a generation stages."""
        async with self._publication:
            async with self._mutate:
                yield

    @asynccontextmanager
    async def publication_transaction(self) -> AsyncIterator[None]:
        """Serialize composer mutation through matching server publication."""
        async with self._publication:
            yield

    async def finish_publication(self, publication: Awaitable[T]) -> T:
        """Finish state publication after composer commit before honoring cancellation."""
        task = asyncio.ensure_future(publication)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise cancelled

    def set_memory_headroom(self, base_bytes: int) -> None:
        """Push a new process-global base to every registered policy worker."""
        self._headroom_base = base_bytes
        if self._headroom_mirror is not None:
            self._headroom_mirror.set_base(base_bytes)

    def residency_memory_budgets(self) -> dict[str, dict[str, int]]:
        """Restart-bound accelerator budgets by active policy worker."""
        return {
            name: dict(record.spec.vram_budgets)
            for name, record in self._records.items()
            if record.spec.runtime_settings
        }

    @staticmethod
    def _vision_provider_models(
        records: Mapping[str, _PackRecord],
    ) -> dict[tuple[str, str], str | None]:
        return {
            (provider.node, record.manifest.name): provider.model
            for record in records.values()
            for provider in record.manifest.vision_providers
        }

    @staticmethod
    def _vision_provider_nodes(
        schemas: Mapping[str, NodeSchema], generation_nodes: Collection[str]
    ) -> frozenset[str]:
        return frozenset(
            node_type
            for node_type, schema in schemas.items()
            if node_type not in generation_nodes
            and (provider_input := schema.input("provider")) is not None
            and provider_input.hidden
        )

    @staticmethod
    def _generation_provider_nodes(
        records: Mapping[str, _PackRecord], schemas: Mapping[str, NodeSchema]
    ) -> frozenset[str]:
        candidates = {
            provider.node
            for record in records.values()
            for provider in record.manifest.generation_providers
        }
        candidates.update(
            node_type
            for record in records.values()
            if any(
                capability.id == "dinkster.generation.schemas"
                for capability in record.manifest.capabilities
            )
            for node_type in record.schema_only
        )
        return frozenset(
            node_type
            for node_type in candidates
            if (schema := schemas.get(node_type)) is not None
            and (provider_input := schema.input("provider")) is not None
            and provider_input.hidden
            and not provider_input.advanced
        )

    def _resolve_providers(
        self,
        graph: Graph,
        *,
        topology: Topology,
        routes: Mapping[tuple[str, str], str],
        models: Mapping[tuple[str, str], str | None],
        provider_nodes: Collection[str],
        generation_nodes: Collection[str],
        placement: Mapping[str, str],
        remote_names: Collection[str],
        schemas: Mapping[str, NodeSchema],
    ) -> Graph:
        def unavailable(node_id: str, node_type: str) -> ProviderResolutionError:
            schema = schemas[node_type]
            return ProviderResolutionError(
                node_id=node_id,
                node_type=node_type,
                title=schema.display_name or node_type,
                capability="a compatible vision-processing implementation",
                remedy="Install or reconnect the standard vision components, then retry.",
            )

        def resolve(
            current: Graph,
            inherited_worker: str | None = None,
            prefix: str = "",
        ) -> Graph:
            nodes: dict[str, GraphNode | RegionNode] = {}
            for node_id, node in current.nodes.items():
                stable_node_id = f"{prefix}/{node_id}" if prefix else node_id
                preferred_worker = inherited_worker or placement.get(top_level_node_id(node_id))
                if isinstance(node, RegionNode):
                    nodes[node_id] = replace(
                        node,
                        body=resolve(node.body, preferred_worker, stable_node_id),
                    )
                    continue
                if node.node_type in generation_nodes:
                    generation_provider = node.inputs.get("provider")
                    if isinstance(generation_provider, TypedLiteral):
                        generation_provider = generation_provider.value
                    if generation_provider == _BUILTIN_GENERATION_PROVIDER:
                        node = replace(
                            node,
                            inputs={
                                input_id: value
                                for input_id, value in node.inputs.items()
                                if input_id != "provider"
                            },
                        )
                if node.node_type not in provider_nodes:
                    nodes[node_id] = node
                    continue
                arms = _provider_arms(
                    topology.get(node.node_type, ()),
                    preferred_worker,
                    remote_names,
                )
                provider_value = node.inputs.get("provider")
                if provider_value is not None:
                    if isinstance(provider_value, Link):
                        nodes[node_id] = node
                        continue
                    if isinstance(provider_value, TypedLiteral):
                        provider_value = provider_value.value
                    if not isinstance(provider_value, str):
                        raise ValueError(f"{node.node_type} vision provider must be a string id")
                    try:
                        _select_vision_provider(
                            node.node_type,
                            "auto",
                            arms,
                            routes,
                            models,
                            provider_value,
                        )
                    except ValueError as exc:
                        raise unavailable(stable_node_id, node.node_type) from exc
                    nodes[node_id] = node
                    continue
                model_value = node.inputs.get("model", "auto")
                if isinstance(model_value, TypedLiteral):
                    model_value = model_value.value
                if isinstance(model_value, Link):
                    nodes[node_id] = node
                    continue
                if not isinstance(model_value, str):
                    raise ValueError(f"{node.node_type} model choice must be a string")
                try:
                    provider, _target = _select_vision_provider(
                        node.node_type,
                        model_value,
                        arms,
                        routes,
                        models,
                    )
                except ValueError as exc:
                    raise unavailable(stable_node_id, node.node_type) from exc
                nodes[node_id] = replace(node, inputs={**node.inputs, "provider": provider})
            return Graph(nodes=nodes)

        return resolve(graph)

    async def _preflight_provider_assets(
        self,
        graph: Graph,
        consented: frozenset[str],
        hinted_sources: Mapping[str, Sequence[str]],
        *,
        topology: Topology,
        routes: Mapping[tuple[str, str], str],
        models: Mapping[tuple[str, str], str | None],
        provider_nodes: Collection[str],
        placement: Mapping[str, str],
        remote_names: Collection[str],
    ) -> tuple[frozenset[str], list[dict[str, object]]]:
        requirements: dict[str, tuple[RemoteWorker, set[str]]] = {}

        def add_arm(node_type: str, arm: ArmRecord) -> None:
            if arm.remote is None or not arm.provider_declared:
                return
            worker = arm.owner_worker
            if not isinstance(worker, RemoteWorker):
                return
            declaration = next(
                (
                    provider
                    for provider in worker.vision_providers or ()
                    if provider.node == node_type and arm.vision_provider == worker.pack
                ),
                None,
            )
            if declaration is None or not declaration.artifacts:
                return
            entry = requirements.setdefault(arm.remote, (worker, set()))
            entry[1].update(declaration.artifacts)

        def scan(current: Graph, inherited_worker: str | None = None) -> None:
            for node_id, node in current.nodes.items():
                preferred_worker = inherited_worker or placement.get(top_level_node_id(node_id))
                if isinstance(node, RegionNode):
                    scan(node.body, preferred_worker)
                    continue
                if node.node_type not in provider_nodes:
                    continue
                arms = _provider_arms(
                    topology.get(node.node_type, ()),
                    preferred_worker,
                    remote_names,
                )
                provider_value = node.inputs.get("provider")
                if isinstance(provider_value, TypedLiteral):
                    provider_value = provider_value.value
                if isinstance(provider_value, Link):
                    for arm in arms:
                        if arm.vision_provider is not None:
                            add_arm(node.node_type, arm)
                    continue
                if isinstance(provider_value, str):
                    requested = provider_value
                    model = "auto"
                else:
                    requested = None
                    model_value = node.inputs.get("model", "auto")
                    if isinstance(model_value, TypedLiteral):
                        model_value = model_value.value
                    if isinstance(model_value, Link):
                        for arm in arms:
                            if arm.vision_provider is not None:
                                add_arm(node.node_type, arm)
                        continue
                    model = model_value if isinstance(model_value, str) else "auto"
                _provider, target = _select_vision_provider(
                    node.node_type,
                    model,
                    arms,
                    routes,
                    models,
                    requested,
                )
                selected = next((arm for arm in arms if arm.name == target), None)
                if selected is not None:
                    add_arm(node.node_type, selected)

        scan(graph)
        plan: list[dict[str, object]] = []
        selected_digests: set[str] = set()
        for remote_name in sorted(requirements):
            worker, asset_ids = requirements[remote_name]
            selected_digests.update(
                asset.need.digest for asset in worker.declared_assets if asset.id in asset_ids
            )
            plan.extend(
                await worker.preflight_provider_assets(
                    sorted(asset_ids),
                    consented,
                    hinted_sources,
                )
            )
        return frozenset(selected_digests), plan

    async def _plan_execution(
        self,
        node_type: str,
        schema: NodeSchema,
        inputs: Mapping[str, Value],
        run_id: str | None = None,
        attention_config: AttentionPolicyConfig | None = None,
        *,
        topology: Topology | None = None,
        extension_hash: str | None = None,
        preferred_worker: str | None = None,
        remote_names: frozenset[str] | None = None,
        vision_provider_routes: Mapping[tuple[str, str], str] | None = None,
        vision_provider_models: Mapping[tuple[str, str], str | None] | None = None,
        vision_provider_nodes: Collection[str] | None = None,
        generation_provider_routes: Mapping[tuple[str, str], str] | None = None,
        generation_provider_nodes: Collection[str] | None = None,
    ) -> ExecutionSelection | None:
        all_arms = (self._topology if topology is None else topology).get(node_type)
        if all_arms is None:
            return None
        if not all_arms:
            raise RuntimeError(f"node type {node_type!r} has no execution provider")
        self._require_available_inference(inputs)
        known_remotes = frozenset(self._remotes) if remote_names is None else remote_names
        remote_preference = preferred_worker is not None and preferred_worker in known_remotes
        arms = _provider_arms(all_arms, preferred_worker, known_remotes)
        if not arms:
            raise RuntimeError(
                f"worker {preferred_worker!r} has no dispatch arm for node type {node_type!r}"
            )
        default_attention_config = AttentionPolicyConfig()
        effective_attention_config = attention_config or default_attention_config
        planned_arms: list[ArmRecord] = []
        attention_routes: dict[str, AttentionRouteToken | None] = {}
        attention_diagnostics: dict[str, str] = {}
        for arm in arms:
            fallback_reason: str | None = None
            execution_worker = arm.execution_worker
            if isinstance(execution_worker, (LazyWorker, _SingleJobWorkerPool)):
                arm = replace(
                    arm,
                    attention_capabilities=execution_worker.attention_capabilities,
                    attention_route_token=execution_worker.attention_route_token,
                )
            if getattr(execution_worker, "cold", False):
                token = None
            elif arm.attention_capabilities is not None:
                try:
                    token = derive_attention_route_token(
                        arm.attention_capabilities, effective_attention_config
                    )
                except ValueError as exc:
                    token = arm.attention_route_token
                    if token is None:
                        token = derive_attention_route_token(
                            arm.attention_capabilities, default_attention_config
                        )
                    fallback_reason = f"requested route unavailable ({exc})"
            else:
                token = arm.attention_route_token
                if effective_attention_config != default_attention_config:
                    fallback_reason = "missing capability evidence"
            if fallback_reason is not None:
                requested = attention_policy_config_to_wire(effective_attention_config)
                actual = (
                    "default auto route without an attention token"
                    if token is None
                    else ", ".join(
                        f"{route.role}={route.primary}"
                        + (f" (fallback={route.fallback})" if route.fallback is not None else "")
                        for route in token.routes
                    )
                )
                attention_diagnostics[arm.name] = (
                    f"Attention request {requested!r} for arm {arm.name!r}: "
                    f"{fallback_reason}; using {actual}"
                )
            planned_arms.append(arm)
            attention_routes[arm.name] = token
        arms = tuple(planned_arms)
        first_arm = arms[0]
        provider_routes = {} if vision_provider_routes is None else vision_provider_routes
        provider_models = {} if vision_provider_models is None else vision_provider_models
        provider_nodes = (
            {managed_type for managed_type, _provider in provider_routes}
            if vision_provider_nodes is None
            else set(vision_provider_nodes)
        )
        provider_target: str | None = None
        selected_provider: str | None = None
        if node_type in provider_nodes:
            provider_value = inputs.get("provider")
            if provider_value is None:
                model_value = inputs.get("model")
                model = "auto" if model_value is None else model_value.resolve()
                if not isinstance(model, str):
                    raise RuntimeError(f"{node_type} model choice must be a string")
                selected_provider, provider_target = _select_vision_provider(
                    node_type,
                    model,
                    arms,
                    provider_routes,
                    provider_models,
                )
            else:
                provider = provider_value.resolve()
                if not isinstance(provider, str):
                    raise RuntimeError(f"{node_type} vision provider must be a string id")
                try:
                    selected_provider, provider_target = _select_vision_provider(
                        node_type,
                        "auto",
                        arms,
                        provider_routes,
                        provider_models,
                        provider,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"{node_type} names unavailable vision provider {provider!r}"
                    ) from exc
        generation_routes = {} if generation_provider_routes is None else generation_provider_routes
        generation_nodes = (
            {managed_type for managed_type, _provider in generation_routes}
            if generation_provider_nodes is None
            else set(generation_provider_nodes)
        )
        generation_input = schema.input("provider")
        if (
            generation_input is not None
            and not generation_input.required
            and not generation_input.hidden
            and isinstance(generation_input.widget, ComboWidget)
            and generation_input.widget.remote_route is not None
        ):
            generation_nodes.add(node_type)
        if node_type in generation_nodes and (provider_value := inputs.get("provider")) is not None:
            provider = provider_value.resolve()
            if not isinstance(provider, str):
                raise RuntimeError(f"{node_type} generation provider must be a string id")
            if provider == _BUILTIN_GENERATION_PROVIDER:
                raise RuntimeError(
                    f"{node_type} built-in generation service must be selected directly, "
                    "not supplied by a link"
                )
            else:
                candidates = tuple(
                    arm for arm in arms if arm.generation_provider == provider and arm.available
                )
                if not candidates:
                    target = generation_routes.get((node_type, provider))
                    candidates = tuple(arm for arm in arms if arm.name == target)
                if not candidates:
                    raise RuntimeError(
                        f"{node_type} names unavailable generation provider {provider!r}"
                    )
                chosen = min(candidates, key=lambda arm: (arm.remote is not None, arm.name))
                provider_target = chosen.name
                selected_provider = provider
        if node_type in generation_nodes and provider_target is None:
            builtin_arms = tuple(
                arm for arm in arms if arm.generation_provider == _BUILTIN_GENERATION_PROVIDER
            )
            if builtin_arms:
                arms = builtin_arms
                selected_provider = _BUILTIN_GENERATION_PROVIDER
            else:
                uncertain_arms = tuple(
                    arm
                    for arm in arms
                    if arm.generation_provider is not None and not arm.provider_declared
                )
                if uncertain_arms:
                    arms = uncertain_arms
                    selected_provider = min(
                        cast("tuple[str, ...]", tuple(arm.generation_provider for arm in arms))
                    )
                    arms = tuple(
                        arm for arm in arms if arm.generation_provider == selected_provider
                    )
            if not arms or selected_provider is None:
                raise RuntimeError(
                    f"{node_type} has no default generation execution provider; "
                    "select an installed provider"
                )
            first_arm = arms[0]
        selected: ExecutionSelection | None = None
        if provider_target is None and self._native_policy is not None:
            available_arms = tuple(
                arm
                for index, arm in enumerate(arms)
                if index == 0 or isinstance(arm.domain.worker, GroupMemberWorker) or arm.available
            )
            candidates = (
                first_arm.name,
                MappingProxyType({arm.name: arm.default_cache_tag for arm in available_arms}),
            )
            available_attention_routes = MappingProxyType(
                {arm.name: attention_routes[arm.name] for arm in available_arms}
            )
            if extension_hash is None:
                selected = await self._native_policy.select(
                    node_type,
                    inputs,
                    candidates,
                    run_id=run_id,
                    attention_routes=available_attention_routes,
                )
            else:
                selected = await self._native_policy.select(
                    node_type,
                    inputs,
                    candidates,
                    run_id=run_id,
                    extension_behavior_hash=extension_hash,
                    attention_routes=available_attention_routes,
                )
        if provider_target is None and selected is None:
            selected = select_resident_producer(
                inputs,
                {arm.name: arm.default_cache_tag for arm in arms},
                attention_routes,
            )
        target = selected.target if selected is not None else first_arm.name
        if provider_target is not None:
            target = provider_target
        arm = next((candidate for candidate in arms if candidate.name == target), None)
        if arm is None and remote_preference and selected is not None:
            # A worker hint pins dispatch to this remote's arms. The policy
            # still runs first - managed types derive their execution
            # identity from its selection - but a selection naming an arm
            # this remote does not offer cannot be honored, so the hint wins
            # with the arm's default identity.
            selected = None
            arm = first_arm
        if arm is None:
            raise RuntimeError(
                f"execution policy selected unknown arm {target!r} for {node_type!r}"
            )
        if getattr(arm.execution_worker, "cold", False):
            await arm.execution_worker.ensure_started()
            # Re-select with live capability evidence so policy cache identities
            # and fallback decisions never use a catalog as runtime authority.
            return await self._plan_execution(
                node_type,
                schema,
                inputs,
                run_id,
                attention_config,
                topology=topology,
                extension_hash=extension_hash,
                preferred_worker=preferred_worker,
                remote_names=remote_names,
                vision_provider_routes=vision_provider_routes,
                vision_provider_models=vision_provider_models,
                vision_provider_nodes=vision_provider_nodes,
                generation_provider_routes=generation_provider_routes,
                generation_provider_nodes=generation_provider_nodes,
            )
        if isinstance(arm.execution_worker, (LazyWorker, _ReplicaWorkerPool)):
            arm.execution_worker.validate_schema(node_type)
        if not isinstance(arm.domain.worker, GroupMemberWorker) and (
            not arm.domain.worker.alive or arm.instance_token() is None
        ):
            raise RuntimeError(f"dispatch arm {arm.name!r} for {node_type!r} is not live")
        if selected is not None:
            return replace(
                selected,
                worker=(arm.remote or "local"),
                provider=selected_provider,
                pack=arm.implementation_pack or arm.default_arm,
                execution_arm=arm.execution_arm,
                attention_diagnostic=attention_diagnostics.get(arm.name),
            )
        token = attention_routes[arm.name]
        return ExecutionSelection(
            target=arm.name,
            cache_tag=arm.default_cache_tag,
            worker=(arm.remote or "local"),
            provider=selected_provider,
            pack=arm.implementation_pack or arm.default_arm,
            execution_arm=arm.execution_arm,
            attention_policy=("auto" if token is None else token.requested_policy),
            attention_route_token=token,
            attention_diagnostic=attention_diagnostics.get(arm.name),
        )

    def _publish_runtime(
        self,
        topology: Topology,
        snapshot: ExtensionSnapshot,
        sampler_registry: SamplerRegistrySnapshot,
        graph_compiler_registry: GraphCompilerRegistrySnapshot,
        graph_compile_transport: GraphCompileTransport | None,
    ) -> None:
        routes = self._dispatch_routes(topology, tuple(topology))
        worker = RoutingWorker(routes, default=self._core_worker)
        behavior_hash = None if not snapshot.extensions else extension_behavior_hash(snapshot)
        _choices, vision_provider_routes = self._validated_vision_providers(self._records, topology)
        vision_provider_models = self._vision_provider_models(self._records)
        _choices, generation_provider_routes = self._validated_generation_providers(
            self._records, topology
        )
        generation_provider_nodes = self._generation_provider_nodes(
            self._records, self.composition.schemas
        ) | {node_type for node_type, _provider in generation_provider_routes}
        vision_provider_nodes = self._vision_provider_nodes(
            self.composition.schemas, generation_provider_nodes
        )
        remote_names = frozenset(self._remotes)
        runtime_schemas = MappingProxyType(dict(self.composition.schemas))
        host_type_workers = tuple(
            record.worker
            for record in self._records.values()
            if record.spec.in_process
            and record.manifest.types_entry is not None
            and isinstance(record.worker, LazyWorker)
        )

        async def prepare_host_types(atoms: set[str]) -> None:
            for host_worker in host_type_workers:
                if atoms.intersection(host_worker.catalog.types.atoms):
                    await host_worker.ensure_started()

        def resolve_providers(graph: Graph) -> Graph:
            return self._resolve_providers(
                graph,
                topology=topology,
                routes=vision_provider_routes,
                models=vision_provider_models,
                provider_nodes=vision_provider_nodes,
                generation_nodes=generation_provider_nodes,
                placement={},
                remote_names=remote_names,
                schemas=runtime_schemas,
            )

        async def preflight_provider_assets(
            graph: Graph,
            consented: frozenset[str],
            hinted_sources: Mapping[str, Sequence[str]],
        ) -> tuple[frozenset[str], list[dict[str, object]]]:
            return await self._preflight_provider_assets(
                graph,
                consented,
                hinted_sources,
                topology=topology,
                routes=vision_provider_routes,
                models=vision_provider_models,
                provider_nodes=vision_provider_nodes,
                placement={},
                remote_names=remote_names,
            )

        async def plan_execution(
            _node_id: str,
            node_type: str,
            schema: NodeSchema,
            inputs: Mapping[str, Value],
            run_id: str,
            attention_config: AttentionPolicyConfig | None,
        ) -> ExecutionSelection | None:
            return await self._plan_execution(
                node_type,
                schema,
                inputs,
                run_id,
                attention_config,
                topology=topology,
                extension_hash=behavior_hash,
                vision_provider_routes=vision_provider_routes,
                vision_provider_models=vision_provider_models,
                vision_provider_nodes=vision_provider_nodes,
                generation_provider_routes=generation_provider_routes,
                generation_provider_nodes=generation_provider_nodes,
            )

        self._runtime_seat.publish(
            ExecutionRuntime(
                worker=worker,
                plan_execution=plan_execution,
                resolve_providers=resolve_providers,
                preflight_provider_assets=preflight_provider_assets,
                run_finished=getattr(self._native_policy, "release_run", None),
                owner_alive=lambda token: token in self._live_token_owners(topology),
                extension_snapshot=snapshot,
                sampler_registry_snapshot=sampler_registry,
                graph_compiler_registry=graph_compiler_registry,
                graph_compile_transport=graph_compile_transport,
                schemas=runtime_schemas,
                prepare_host_types=prepare_host_types if host_type_workers else None,
                known_types=CatalogTypeRegistry(
                    self.composition._registry,
                    tuple(worker.catalog.types for worker in host_type_workers),
                ),
            )
        )

    @staticmethod
    def _hinted_worker(hints: Mapping[str, str], node_id: str) -> str | None:
        return hints.get(top_level_node_id(node_id))

    def place_execution(
        self,
        runtime: ExecutionRuntime,
        hints: Mapping[str, str],
    ) -> ExecutionRuntime:
        """Bind immutable submission hints to one already-pinned runtime."""
        copied = MappingProxyType(dict(hints))
        topology = self._topology
        remote_names = frozenset(self._remotes)
        base_planner = runtime.plan_execution
        _choices, vision_provider_routes = self._validated_vision_providers(self._records, topology)
        vision_provider_models = self._vision_provider_models(self._records)
        _choices, generation_provider_routes = self._validated_generation_providers(
            self._records, topology
        )
        generation_provider_nodes = self._generation_provider_nodes(
            self._records, self.composition.schemas
        ) | {node_type for node_type, _provider in generation_provider_routes}
        vision_provider_nodes = self._vision_provider_nodes(
            self.composition.schemas, generation_provider_nodes
        )

        def resolve_providers(graph: Graph) -> Graph:
            return self._resolve_providers(
                graph,
                topology=topology,
                routes=vision_provider_routes,
                models=vision_provider_models,
                provider_nodes=vision_provider_nodes,
                generation_nodes=generation_provider_nodes,
                placement=copied,
                remote_names=remote_names,
                schemas=cast("Mapping[str, NodeSchema]", runtime.schemas),
            )

        async def preflight_provider_assets(
            graph: Graph,
            consented: frozenset[str],
            hinted_sources: Mapping[str, Sequence[str]],
        ) -> tuple[frozenset[str], list[dict[str, object]]]:
            return await self._preflight_provider_assets(
                graph,
                consented,
                hinted_sources,
                topology=topology,
                routes=vision_provider_routes,
                models=vision_provider_models,
                provider_nodes=vision_provider_nodes,
                placement=copied,
                remote_names=remote_names,
            )

        async def plan_execution(
            node_id: str,
            node_type: str,
            schema: NodeSchema,
            inputs: Mapping[str, Value],
            run_id: str,
            attention_config: AttentionPolicyConfig | None,
        ) -> ExecutionSelection | None:
            worker_name = self._hinted_worker(copied, node_id)
            if worker_name is None:
                return (
                    None
                    if base_planner is None
                    else await base_planner(
                        node_id, node_type, schema, inputs, run_id, attention_config
                    )
                )
            hint_id = top_level_node_id(node_id)
            devices = resident_devices(inputs)
            unowned = sorted(
                device
                for device in devices
                if "@" in device and device.rpartition("@")[2] not in remote_names
            )
            if unowned:
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"placement hint {hint_id!r} cannot use inputs resident on device(s) "
                        f"{', '.join(unowned)} that no configured worker owns",
                    )
                )
            resident_workers = {
                qualifier if separator else "local"
                for device in devices
                for _, separator, qualifier in (device.rpartition("@"),)
            }
            if len(resident_workers) > 1:
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"placement hint {hint_id!r} has inputs resident on different workers "
                        f"({', '.join(sorted(resident_workers))}); cross-worker transfer of "
                        "resident state does not exist",
                    )
                )
            if resident_workers and worker_name not in resident_workers:
                resident_worker = next(iter(resident_workers))
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"placement hint {hint_id!r} names worker {worker_name!r}, but its "
                        f"inputs are resident on worker {resident_worker!r}; resident state "
                        "cannot cross workers",
                    )
                )
            try:
                selected = await self._plan_execution(
                    node_type,
                    schema,
                    inputs,
                    run_id,
                    attention_config,
                    topology=topology,
                    extension_hash=runtime.extension_behavior_hash,
                    preferred_worker=worker_name,
                    remote_names=remote_names,
                    vision_provider_routes=vision_provider_routes,
                    vision_provider_models=vision_provider_models,
                    vision_provider_nodes=vision_provider_nodes,
                    generation_provider_routes=generation_provider_routes,
                    generation_provider_nodes=generation_provider_nodes,
                )
                if selected is None:
                    raise RuntimeError(f"node type {node_type!r} has no execution selection")
                return selected
            except Exception as exc:
                raise ExecutionError(
                    NodeError(
                        node_id,
                        node_type,
                        f"placement hint {hint_id!r} naming worker {worker_name!r} "
                        f"cannot be satisfied: {exc}",
                    )
                ) from exc

        return replace(
            runtime,
            plan_execution=plan_execution,
            resolve_providers=resolve_providers,
            preflight_provider_assets=preflight_provider_assets,
            placement_worker=lambda node_id: self._hinted_worker(copied, node_id),
        )

    def workers(self, configured: Sequence[RemoteSpec]) -> tuple[WorkerInfo, ...]:
        """Return configured worker state without initiating network work."""
        local_types = set(self._core_schemas)
        for record in self._records.values():
            local_types.update(record.delta.schemas)
            local_types.update(record.executes)
        result = [
            WorkerInfo(
                name="local",
                status="connected",
                node_types=tuple(sorted(local_types)),
            )
        ]
        for spec in configured:
            record = self._remotes.get(spec.name)
            if record is None:
                status = "configured"
            elif record.worker.alive:
                status = "connected"
            else:
                status = "disconnected"
            result.append(
                WorkerInfo(
                    name=spec.name,
                    status=status,
                    node_types=(
                        tuple(sorted(record.node_types))
                        if record is not None
                        else tuple(sorted(spec.nodes or ()))
                    ),
                    device_qualifiers=(f"@{spec.name}",),
                )
            )
        return tuple(result)

    async def full_free(self, request_id: str) -> tuple[dict[str, object], ...]:
        """Release volatile caches and every currently live worker consumer."""
        async with self._mutate:
            operation = asyncio.create_task(self._full_free_locked(request_id))
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError as cancelled:
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                raise cancelled

    async def _full_free_locked(self, request_id: str) -> tuple[dict[str, object], ...]:
        local_consumers: list[dict[str, object]] = []
        cache_errors: list[str] = []
        for cache in self.composition._engine_caches:
            try:
                if isinstance(cache, MemoryLRUCache):
                    cache.clear()
                else:
                    for layer in cache.layers:
                        if isinstance(layer, MemoryLRUCache):
                            layer.clear()
            except Exception as exc:  # noqa: BLE001 - continue every cache
                cache_errors.append(str(exc))
        cache_error = "; ".join(cache_errors) if cache_errors else None
        local_consumers.append(
            {
                "consumer": "execution-cache",
                "status": "complete" if cache_error is None else "error",
                **({"error": cache_error} if cache_error is not None else {}),
            }
        )
        pool_result = await default_pool().full_release()
        local_consumers.append(
            {
                "consumer": "comfy-models",
                "status": pool_result.status,
                **({"error": pool_result.error} if pool_result.error is not None else {}),
            }
        )
        results: list[dict[str, object]] = [
            {
                "worker": "local",
                "workerInstance": process_instance_token(),
                "deviceMap": {"mapping": {}, "qualifier": None},
                "status": "complete",
                "consumers": local_consumers,
            }
        ]
        release_guard = ReleaseGuard(
            pins=self.composition._resource_pins,
            invalidators=tuple(cache.drop_referencing for cache in self.composition._engine_caches),
        )
        targets: list[tuple[str, Any, str]] = []

        def record_missing_worker(name: str, worker: Any) -> None:
            results.append(
                {
                    "worker": name,
                    "workerInstance": worker.instance_token,
                    "deviceMap": worker.device_map_wire,
                    "status": "error",
                    "error": "worker is unavailable at the maintenance snapshot",
                    "consumers": [],
                }
            )

        for name, record in self._records.items():
            worker = record.worker
            if isinstance(worker, LazyWorker) and worker.cold:
                continue
            if record.spec.in_process:
                consumers: list[dict[str, object]] = []
                result: dict[str, object] = {
                    "worker": name,
                    "workerInstance": process_instance_token(),
                    "deviceMap": {"mapping": {}, "qualifier": None},
                    "status": "error",
                    "consumers": consumers,
                }
                try:
                    consumers.extend(await worker.full_release())
                    if record.tenant_registrations is not None:
                        tenant_result = await record.tenant_registrations.full_release()
                        tenant: dict[str, object] = {
                            "consumer": "model-tenants",
                            "status": tenant_result.status,
                        }
                        if tenant_result.error is not None:
                            tenant["error"] = tenant_result.error
                        consumers.append(tenant)
                except Exception as exc:  # noqa: BLE001 - one worker cannot hide others
                    result["error"] = str(exc) or "in-process worker full release failed"
                else:
                    result["status"] = "complete"
                results.append(result)
                continue
            if isinstance(worker, _ReplicaWorkerPool):
                for lane in worker.lanes:
                    lane_worker = lane.worker
                    if isinstance(lane_worker, LazyWorker) and lane_worker.cold:
                        continue
                    lane_name = f"{name}@cuda:{lane.cuda_index}"
                    token = lane_worker.instance_token
                    if lane_worker.alive and token is not None:
                        targets.append((lane_name, lane_worker, token))
                    else:
                        record_missing_worker(lane_name, lane_worker)
                continue
            token = worker.instance_token
            if worker.alive and token is not None:
                targets.append((name, worker, token))
            else:
                record_missing_worker(name, worker)
        for name, record in self._remotes.items():
            worker = record.worker
            token = worker.instance_token
            if worker.alive and token is not None:
                targets.append((name, worker, token))
            else:
                record_missing_worker(name, worker)

        process_locks: dict[str, asyncio.Lock] = {}

        async def release_one(name: str, worker: Any, token: str) -> dict[str, object]:
            result: dict[str, object] = {
                "worker": name,
                "workerInstance": token,
                "deviceMap": worker.device_map_wire,
                "status": "error",
                "consumers": [],
            }
            async with process_locks.setdefault(token, asyncio.Lock()):
                try:
                    outcome = await worker.full_release(
                        request_id, token, release_guard=release_guard
                    )
                except Exception as exc:  # noqa: BLE001 - one worker cannot hide others
                    result["error"] = str(exc) or "worker full release failed"
                    return result
            result["consumers"] = list(outcome.consumers)
            if not worker.alive or worker.instance_token != token:
                result["error"] = "worker instance changed during full release"
                return result
            result["status"] = outcome.status
            if outcome.error is not None:
                result["error"] = outcome.error
            return result

        if targets:
            results.extend(await asyncio.gather(*(release_one(*target) for target in targets)))
        return tuple(results)

    async def _commit_published_generation(
        self, snapshot: ExtensionSnapshot, retired_worker: IsolatedWorker | None
    ) -> None:
        """Retain generations while their worker lives; forget them on rotation."""
        new_digest = (
            None if not snapshot.extensions else "sha256:" + extension_behavior_hash(snapshot)
        )
        old_digest = self._published_generation_digest
        self._published_generation_digest = new_digest
        if self._private_staging and old_digest is not None and old_digest != new_digest:
            worker = self._sampling_worker(self._topology)
            if worker is not None:
                await worker.release_inference_generation(old_digest)
            remove_sampler_catalog_record(self._sampler_catalog_path, old_digest)
            return
        if retired_worker is not None:
            retired = set(self._retired_generation_digests)
            if old_digest is not None:
                retired.add(old_digest)
            for digest in retired - {new_digest}:
                remove_sampler_catalog_record(self._sampler_catalog_path, digest)
            self._retired_generation_digests.clear()
            return
        if old_digest is None or old_digest == new_digest:
            return
        self._retired_generation_digests.add(old_digest)

    async def _rollback_unpublished_generation(
        self, snapshot: ExtensionSnapshot, topology: Topology
    ) -> None:
        """Remove a materialized generation that never reached publication."""
        if not snapshot.extensions:
            return
        digest = "sha256:" + extension_behavior_hash(snapshot)
        if (
            digest == self._published_generation_digest
            or digest in self._retired_generation_digests
        ):
            return
        worker = self._sampling_worker(topology)
        if worker is not None:
            with contextlib.suppress(Exception):
                await worker.release_inference_generation(digest)
        remove_sampler_catalog_record(self._sampler_catalog_path, digest)

    def _live_token_owners(self, topology: Topology) -> dict[str, _ResidencyDomain]:
        owners: dict[str, _ResidencyDomain] = {}
        seen_domains: set[int] = set()
        for arms in topology.values():
            for arm in arms:
                domain_id = id(arm.domain)
                if domain_id in seen_domains:
                    continue
                if not arm.domain.worker.alive:
                    continue
                token = arm.instance_token()
                if token is None:
                    continue
                seen_domains.add(domain_id)
                existing = owners.get(token)
                if existing is not None and existing is not arm.domain:
                    raise RuntimeError(
                        f"distinct live worker sessions for {existing.label!r} "
                        f"and {arm.domain.label!r} "
                        f"share instance token {token!r}"
                    )
                owners[token] = arm.domain
        return owners

    def _owner_alive(self, token: str) -> bool:
        return token in self._live_token_owners(self._topology)

    def _resolve_owner(self, topology: Topology, token: str) -> _ResidencyDomain | None:
        return self._live_token_owners(topology).get(token)

    def _execution_identity(
        self,
        manifest: PackManifest,
        packs: Mapping[str, PackInfo],
        spec: PackSpec,
    ) -> str:
        info = packs.get(manifest.name)
        artifact = (
            info.artifact_digest if info is not None and info.artifact_digest else "unversioned"
        )
        if not spec.execution_config:
            return artifact
        hasher = hashlib.sha256()
        for value in (
            artifact,
            *(item for pair in sorted(spec.execution_config.items()) for item in pair),
        ):
            encoded = value.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
        return "sha256:" + hasher.hexdigest()

    @staticmethod
    def _development_package_digest(manifest: PackManifest) -> str:
        """Bind an unpublished extension to its deterministic source tree."""
        hasher = hashlib.sha256()
        for path in sorted(manifest.root.rglob("*")):
            relative = path.relative_to(manifest.root)
            if not path.is_file() or any(
                part == "__pycache__" or part.startswith(".") for part in relative.parts
            ):
                continue
            encoded = relative.as_posix().encode("utf-8")
            content = path.read_bytes()
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
            hasher.update(len(content).to_bytes(8, "big"))
            hasher.update(content)
        return "sha256:" + hasher.hexdigest()

    @staticmethod
    def _behavior_configuration(spec: PackSpec) -> tuple[tuple[str, BehaviorValue], ...]:
        """Canonicalize host config into BehaviorValue's float-free vocabulary."""
        settings: list[tuple[str, BehaviorValue]] = []
        for key, raw in sorted(spec.extension_config.items()):
            if not isinstance(key, str) or not key:
                raise CompositionError(
                    "extension behavior configuration keys must be non-empty strings"
                )
            value: BehaviorValue
            if isinstance(raw, float):
                if not math.isfinite(raw):
                    raise CompositionError(
                        f"extension behavior configuration {key!r} has a non-finite float"
                    )
                # BehaviorValue deliberately excludes float. The producer
                # owns this stable round-trip spelling before the value can
                # cross an RPC or enter canonical snapshot JSON.
                value = format(raw, ".17g")
            elif raw is None or type(raw) in (str, int, bool):
                value = cast("BehaviorValue", raw)
            else:
                raise CompositionError(
                    f"extension behavior configuration {key!r} must be "
                    "str, int, bool, float, or None"
                )
            settings.append((key, value))
        return tuple(settings)

    @staticmethod
    def _sampling_worker(topology: Topology) -> Any | None:
        for arm in topology.get(SAMPLING_WORKER_NAME, ()):
            if "@native" in arm.name:
                return arm.owner_worker
        return None

    @staticmethod
    def _inference_unavailable_detail(record: _PackRecord) -> PackInferenceUnavailable:
        """Describe a pack whose inference entry has no worker to materialize it.

        The declared provider ids come from the manifest because nothing was
        materialized, so they are reported with ``available`` false to explain a
        name a saved workflow already carries rather than to offer it."""
        entry = record.extension
        assert entry is not None and entry.entries.inference is not None
        return PackInferenceUnavailable(
            reason=INFERENCE_UNAVAILABLE_REASON,
            entry=entry.entries.inference,
            worker=SAMPLING_WORKER_NAME,
            providers=tuple(
                UnavailableInferenceProvider(registry=item.registry, id=item.id)
                for item in unmatched_registry_providers(record.manifest.provides, ())
            ),
        )

    async def _materialize_inference_contributions(
        self,
        records: Mapping[str, _PackRecord],
        topology: Topology,
    ) -> tuple[
        tuple[SamplerExtensionEntry, ...],
        dict[str, tuple[KeyedContribution, ...]],
        dict[str, PackInferenceUnavailable],
    ]:
        entries = tuple(
            SamplerExtensionEntry(name, inference_entry)
            for name, record in sorted(records.items())
            if record.extension is not None
            and (inference_entry := record.extension.entries.inference) is not None
        )
        if not entries:
            return (), {}, {}
        worker = self._sampling_worker(topology)
        if worker is None:
            # The native inference pack did not load, so nothing can execute these
            # declarations. Composing them anyway would register samplers whose
            # every run fails inside the worker, and refusing the whole pack would
            # take its nodes, routes and events with it. Degrade the one surface.
            return (
                (),
                {},
                {
                    entry.extension_id: self._inference_unavailable_detail(
                        records[entry.extension_id]
                    )
                    for entry in entries
                },
            )
        if all(
            getattr(records[entry.extension_id].worker, "catalog", None) is not None
            for entry in entries
        ):
            return entries, {
                entry.extension_id: tuple(
                    KeyedContribution(
                        surface_id=item["surface_id"],
                        id=item["id"],
                        aliases=tuple(item["aliases"]),
                        behavior_metadata=tuple(tuple(pair) for pair in item["behavior_metadata"]),
                    )
                    for item in records[entry.extension_id].worker.catalog.declarations[
                        "inferenceContributions"
                    ]
                )
                for entry in entries
            }, {}
        candidate_key = f"candidate:{uuid.uuid4().hex}"
        write_sampler_catalog(self._sampler_catalog_path, candidate_key, entries)
        try:
            try:
                returned = await worker.materialize_inference_generation(candidate_key)
            except Exception as exc:
                if "multiple guidance strategies were materialized" in str(exc):
                    owners = ", ".join(entry.extension_id for entry in entries)
                    raise CompositionError(
                        "exclusive extension surface inference:inference.guidance.strategy "
                        f"has multiple contributors: {owners}"
                    ) from exc
                raise CompositionError(f"inference extension activation failed: {exc}") from exc
        finally:
            try:
                await worker.release_inference_generation(candidate_key)
            except Exception:
                # Preserve the activation error, if any; closing the worker is
                # the rollback backstop when explicit candidate release fails.
                pass
            remove_sampler_catalog_record(self._sampler_catalog_path, candidate_key)
        by_extension: dict[str, tuple[KeyedContribution, ...]] = {}
        for extension_id, contributions in returned:
            if extension_id in by_extension:
                raise CompositionError(
                    f"inference worker returned extension {extension_id!r} more than once"
                )
            by_extension[extension_id] = contributions
        expected_ids = tuple(entry.extension_id for entry in entries)
        if tuple(by_extension) != expected_ids:
            raise CompositionError(
                "inference worker extension set does not match the staged generation: "
                f"expected {expected_ids}, got {tuple(by_extension)}"
            )
        return entries, by_extension

    async def call_pack_route(
        self, pack: str, route: PackRoute, data: Mapping[str, object], snapshot_digest: str
    ) -> dict[str, object]:
        async with self._mutate:
            if self._runtime_seat.pin().extension_snapshot_digest != snapshot_digest:
                raise KeyError("pack route snapshot is no longer active")
            record = self._records.get(pack)
            if record is None or record.extension is None or route not in record.extension.routes:
                raise KeyError("pack route is no longer active")
            return await record.worker.call_pack_route(route, data)

    async def read_frontend_module(self, pack: str, module: FrontendModule) -> bytes:
        async with self._mutate:
            record = self._records.get(pack)
            if (
                record is None
                or record.extension is None
                or not any(
                    item.id == module.id and item.module == module.module
                    for item in record.extension.frontend_modules
                )
            ):
                raise KeyError("frontend module is no longer active")
            return await asyncio.to_thread(read_module, record.manifest, module)

    async def _build_extension_snapshot(
        self, records: Mapping[str, _PackRecord], topology: Topology
    ) -> tuple[
        ExtensionSnapshot,
        SamplerRegistrySnapshot,
        tuple[KeyedContribution, ...],
        GraphCompilerRegistrySnapshot,
        GraphCompileTransport | None,
        dict[str, PackInferenceUnavailable],
    ]:
        """Derive and worker-validate one RPC-clean behavior generation.

        The last element names the packs whose inference surface could not
        materialize because no native sampling worker is live. Their other
        surfaces compose normally, so the generation is published with those
        packs present and their inference contributions simply absent."""
        (
            inference_entries,
            inference_contributions,
            inference_unavailable,
        ) = await self._materialize_inference_contributions(records, topology)
        sampler_registry: Registry[KeyedContribution] = Registry()
        for declaration in builtin_sampler_snapshot().samplers:
            sampler_registry.register(declaration)
        scheduler_registry: Registry[KeyedContribution] = Registry()
        for descriptor in builtin_registries().schedulers:
            scheduler_registry.register(scheduler_declaration(descriptor))
        surfaces: dict[
            tuple[ExtensionScope, str],
            tuple[CompositionMode, list[str]],
        ] = {}
        active: list[ActiveExtension] = []
        for name, record in sorted(records.items()):
            keyed_contributions = inference_contributions.get(name, ())
            degraded = name in inference_unavailable
            unmatched_providers = unmatched_registry_providers(
                record.manifest.provides,
                ((item.surface_id, item.id) for item in keyed_contributions),
            )
            if degraded:
                # Every declared provider is unregistered here, which is exactly
                # what the degraded record reports; the mismatch is not a
                # misconfiguration until a worker exists to answer the declaration.
                unmatched_providers = ()
            if unmatched_providers:
                provider = unmatched_providers[0]
                raise CompositionError(
                    f"pack {record.manifest.name!r} declares registry provider "
                    f"{provider.registry}:{provider.id}, but its inference contribution "
                    "does not register it"
                )
            declaration = record.extension
            if declaration is None:
                if record.extension_contributions:
                    raise CompositionError(
                        f"pack {name!r} announced extension contributions without a "
                        "[pack.extension] declaration in the host manifest"
                    )
                continue
            contribution_ids: list[str] = []
            seen_ids: set[str] = set()
            announced_routes = tuple(
                route for _, item in record.extension_contributions for route in item.routes
            )
            announced_events = tuple(
                event for _, item in record.extension_contributions for event in item.events
            )
            if announced_routes != declaration.routes or announced_events != declaration.events:
                raise CompositionError(
                    f"extension {name!r} route/event declarations differ from its manifest"
                )
            for event in declaration.events:
                if not any(claim_covers(claim, event.name) for claim in record.claims):
                    raise CompositionError(
                        f"extension {name!r} event {event.name!r} is outside its namespaces"
                    )
            for scope, descriptor in record.extension_contributions:
                manifest_surface = (
                    scope == ExtensionScope.SERVER
                    and descriptor.surface_id == PACK_ROUTES_SURFACE
                    and bool(descriptor.routes)
                ) or (
                    scope == ExtensionScope.SCHEMA
                    and descriptor.surface_id == PACK_EVENTS_SURFACE
                    and bool(descriptor.events)
                )
                if declaration.entries.for_scope(scope) is None and not manifest_surface:
                    raise CompositionError(
                        f"extension {name!r} announced undeclared {scope.value!r} contributions"
                    )
                contribution_id = f"{scope.value}:{descriptor.surface_id}"
                if contribution_id in seen_ids:
                    raise CompositionError(
                        f"extension {name!r} contributes {contribution_id!r} more than once"
                    )
                seen_ids.add(contribution_id)
                contribution_ids.append(contribution_id)
                surface_key = (scope, descriptor.surface_id)
                existing = surfaces.get(surface_key)
                if existing is None:
                    surfaces[surface_key] = (descriptor.mode, [name])
                else:
                    mode, owners = existing
                    if mode != descriptor.mode:
                        raise CompositionError(
                            f"extension surface {scope.value}:{descriptor.surface_id} "
                            f"has incompatible composition modes {mode.value!r} and "
                            f"{descriptor.mode.value!r}"
                        )
                    owners.append(name)
            inference_entry = declaration.entries.inference
            # A degraded pack registers no inference:* contribution id; its
            # declaration is reported through the pack's inferenceUnavailable
            # record instead, which is also why the empty-contribution check
            # below must not fire for it.
            if inference_entry is not None and not degraded:
                if not keyed_contributions:
                    raise CompositionError(
                        f"extension {name!r} declared an inference entry but produced nothing"
                    )
                for surface_id in dict.fromkeys(item.surface_id for item in keyed_contributions):
                    contribution_id = f"{ExtensionScope.INFERENCE.value}:{surface_id}"
                    contribution_ids.append(contribution_id)
                    mode = (
                        CompositionMode.EXCLUSIVE
                        if surface_id == GUIDANCE_SURFACES[2]
                        else CompositionMode.KEYED_REGISTRY
                        if surface_id in (INFERENCE_SAMPLERS_SURFACE, INFERENCE_SCHEDULERS_SURFACE)
                        else CompositionMode.ORDERED_LIST
                        if surface_id == GRAPH_COMPILERS_SURFACE
                        else CompositionMode.WRAPPER_CHAIN
                        if surface_id == GUIDANCE_SURFACES[0]
                        else CompositionMode.ORDERED_LIST
                    )
                    surface_key = (ExtensionScope.INFERENCE, surface_id)
                    previous = surfaces.get(surface_key)
                    surfaces[surface_key] = (
                        mode,
                        [*(previous[1] if previous is not None else []), name],
                    )
                    if mode is CompositionMode.EXCLUSIVE and len(surfaces[surface_key][1]) > 1:
                        owners = surfaces[surface_key][1]
                        raise CompositionError(
                            f"exclusive extension surface inference:{surface_id} has multiple "
                            f"contributors: {', '.join(sorted(owners))}"
                        )
            elif keyed_contributions:
                raise CompositionError(
                    f"extension {name!r} produced contributions without an inference declaration"
                )
            for contribution in keyed_contributions:
                if contribution.surface_id not in (
                    INFERENCE_SAMPLERS_SURFACE,
                    INFERENCE_SCHEDULERS_SURFACE,
                    GRAPH_COMPILERS_SURFACE,
                    *GUIDANCE_SURFACES,
                ):
                    raise CompositionError(
                        f"extension {name!r} produced contribution {contribution.id!r} on "
                        f"unknown surface {contribution.surface_id!r}"
                    )
                if not any(claim_covers(claim, contribution.id) for claim in record.claims):
                    raise CompositionError(
                        f"extension {name!r} contribution {contribution.id!r} is outside "
                        f"the pack's declared namespaces ({', '.join(record.claims)})"
                    )
                if contribution.surface_id == INFERENCE_SAMPLERS_SURFACE:
                    try:
                        sampler_registry.register(contribution)
                    except RegistryError as exc:
                        raise CompositionError(
                            f"extension {name!r} sampler registry collision: {exc}"
                        ) from exc
                if contribution.surface_id == INFERENCE_SCHEDULERS_SURFACE:
                    try:
                        scheduler_registry.register(contribution)
                    except RegistryError as exc:
                        raise CompositionError(
                            f"extension {name!r} scheduler registry collision: {exc}"
                        ) from exc
            info = record.delta.packs.get(name)
            if info is None:
                raise CompositionError(f"extension {name!r} has no matching pack identity")
            package_digest = info.artifact_digest
            if not package_digest:
                # Unpublished development packs have no registry artifact
                # pin. Bind the source tree without generated/hidden files so
                # implementation edits rotate behavior identity on reload.
                package_digest = self._development_package_digest(record.manifest)
            try:
                GraphCompilerRegistrySnapshot(
                    tuple(
                        sorted(
                            (
                                contribution
                                for contribution in keyed_contributions
                                if contribution.surface_id == GRAPH_COMPILERS_SURFACE
                            ),
                            key=lambda item: (
                                dict(item.behavior_metadata).get("order", 0),
                                item.id,
                            ),
                        )
                    )
                )
            except (TypeError, ValueError) as exc:
                raise CompositionError(f"invalid graph compiler composition: {exc}") from exc
            active.append(
                ActiveExtension(
                    id=name,
                    version=info.version or "0.0.0+unversioned",
                    package_digest=package_digest,
                    contribution_ids=tuple(contribution_ids),
                    keyed_contributions=keyed_contributions,
                    capabilities=tuple(sorted(declaration.capabilities)),
                    behavior_configuration=self._behavior_configuration(record.spec),
                    routes=declaration.routes,
                    events=declaration.events,
                    frontend_modules=resolve_frontend_modules(record.manifest),
                )
            )
        for (scope, surface_id), (mode, owners) in sorted(
            surfaces.items(), key=lambda item: (item[0][0].value, item[0][1])
        ):
            if mode is CompositionMode.EXCLUSIVE and len(owners) > 1:
                raise CompositionError(
                    f"exclusive extension surface {scope.value}:{surface_id} has "
                    f"multiple contributors: {', '.join(sorted(owners))}"
                )
        snapshot = ExtensionSnapshot(extensions=tuple(active), frontend_api=FRONTEND_API_VERSION)
        sampler_snapshot = SamplerRegistrySnapshot(tuple(sampler_registry))
        scheduler_snapshot = tuple(scheduler_registry)
        try:
            GuidanceRegistrySnapshot(
                tuple(
                    sorted(
                        (
                            contribution
                            for extension in active
                            for contribution in extension.keyed_contributions
                            if contribution.surface_id in GUIDANCE_SURFACES
                        ),
                        key=lambda item: (
                            GUIDANCE_SURFACES.index(item.surface_id),
                            dict(item.behavior_metadata).get("order", 0),
                            item.id,
                        ),
                    )
                )
            )
        except (TypeError, ValueError) as exc:
            raise CompositionError(f"invalid guidance composition: {exc}") from exc
        try:
            graph_compiler_registry = GraphCompilerRegistrySnapshot(
                tuple(
                    sorted(
                        (
                            contribution
                            for extension in active
                            for contribution in extension.keyed_contributions
                            if contribution.surface_id == GRAPH_COMPILERS_SURFACE
                        ),
                        key=lambda item: (
                            dict(item.behavior_metadata).get("order", 0),
                            item.id,
                        ),
                    )
                )
            )
        except (TypeError, ValueError) as exc:
            raise CompositionError(f"invalid graph compiler composition: {exc}") from exc
        graph_compile_transport: GraphCompileTransport | None = None
        if snapshot.extensions:
            digest = "sha256:" + extension_behavior_hash(snapshot)
            if digest in self._retired_generation_digests:
                raise CompositionError(
                    f"inference generation {digest} is retired and cannot be materialized "
                    "or activated for new work"
                )
            write_sampler_catalog(
                self._sampler_catalog_path,
                digest,
                inference_entries,
                sampler_snapshot,
                expected_extensions=tuple(inference_contributions.items()),
            )
            if inference_entries:
                worker = self._sampling_worker(topology)
                assert worker is not None
                try:
                    final = (
                        tuple(inference_contributions.items())
                        if getattr(worker, "cold", False)
                        else await worker.materialize_inference_generation(digest)
                    )
                except BaseException as exc:
                    release = asyncio.create_task(worker.release_inference_generation(digest))
                    with contextlib.suppress(BaseException):
                        await asyncio.shield(release)
                    remove_sampler_catalog_record(self._sampler_catalog_path, digest)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise CompositionError(
                        f"inference extension declaration validation failed: {exc}"
                    ) from exc
                if final != tuple(inference_contributions.items()):
                    try:
                        await worker.release_inference_generation(digest)
                    finally:
                        remove_sampler_catalog_record(self._sampler_catalog_path, digest)
                    raise CompositionError(
                        "inference worker changed contribution declarations during activation"
                    )
                if graph_compiler_registry.contributions:

                    async def compile_graph(
                        generation_key: str,
                        graph: Mapping[str, object],
                        targets: Sequence[str],
                        *,
                        owning_digest: str = digest,
                        owning_worker: Any = worker,
                    ) -> Mapping[str, object]:
                        if generation_key != owning_digest:
                            raise AssertionError(
                                "graph compile transport generation key does not match "
                                f"its owning digest: {generation_key!r} != {owning_digest!r}"
                            )
                        async with self._mutate:
                            return await owning_worker.compile_graph(owning_digest, graph, targets)

                    graph_compile_transport = compile_graph
        return (
            snapshot,
            sampler_snapshot,
            scheduler_snapshot,
            graph_compiler_registry,
            graph_compile_transport,
            inference_unavailable,
        )

    def _apply_inference_unavailable(
        self,
        unavailable: Mapping[str, PackInferenceUnavailable],
        announced: Mapping[str, PackInfo],
    ) -> None:
        """Publish the inference degradation state and mirror it into the delta.

        The state lives on ``PackInfo``, so it reaches the client only through a
        packs-table entry. ``announced`` is that entry map for the delta being
        committed, which normally carries only the pack being added or reloaded:
        a pack degraded by this commit, or healed by it, is added so the client
        sees the row change. The same object is written back to the owning
        record's delta so a later full resync announces the identical row.
        """
        self._inference_unavailable = dict(unavailable)
        for pack_id, info in list(self.composition.packs.items()):
            detail = unavailable.get(pack_id)
            if info.inference_unavailable == detail:
                continue
            updated = replace(info, inference_unavailable=detail)
            self.composition.packs[pack_id] = updated
            announced[pack_id] = updated
            record = self._records.get(pack_id)
            if record is not None and pack_id in record.delta.packs:
                record.delta.packs[pack_id] = updated
            if detail is not None:
                core_logger("compose").warning(
                    "pack %s composed without a native sampling worker: %s",
                    pack_id,
                    detail.reason,
                )

    def _require_available_inference(self, inputs: Mapping[str, Value]) -> None:
        """Fail a plan that selects a provider a degraded pack no longer offers.

        Keyed on the registry vocabulary rather than any node id: a saved
        workflow can carry the sampler name in whatever input a sampler node or a
        decomposed sampler seam uses, and the reason the user is told is the one
        recorded when the surface degraded."""
        if not self._inference_unavailable:
            return
        for pack_id, detail in sorted(self._inference_unavailable.items()):
            declared = {provider.id for provider in detail.providers}
            for input_name, value in inputs.items():
                selected = value.resolve() if isinstance(value, Value) else value
                if isinstance(selected, str) and selected in declared:
                    raise RuntimeError(
                        f"input {input_name!r} selects {selected!r} from pack "
                        f"{pack_id!r}, whose inference surface is unavailable: {detail.reason}"
                    )

    def _build_topology(
        self,
        records: Mapping[str, _PackRecord],
        remotes: Mapping[str, _RemoteRecord] | None = None,
    ) -> Topology:
        arms_by_type: dict[str, list[ArmRecord]] = {}
        records_by_owned_type: dict[str, _PackRecord] = {}
        body_records_by_pack: dict[str, list[tuple[tuple[str, ...], ArmRecord]]] = {}
        replica_domains: dict[int, _ResidencyDomain] = {}

        def worker_domain(worker: Any, label: str) -> _ResidencyDomain:
            if not isinstance(worker, (IsolatedWorker, LazyWorker)):
                return _ResidencyDomain(worker, label)
            return replica_domains.setdefault(id(worker), _ResidencyDomain(worker, label))

        schemas = dict(self._core_schemas)
        for record in records.values():
            schemas.update(record.delta.schemas)
        generation_nodes = self._generation_provider_nodes(records, schemas)
        vision_nodes = self._vision_provider_nodes(schemas, generation_nodes)

        def annotate_local(arm: ArmRecord, record: _PackRecord, node_type: str) -> ArmRecord:
            arm = replace(arm, implementation_pack=record.manifest.name)
            vision = next(
                (
                    provider
                    for provider in record.manifest.vision_providers
                    if provider.node == node_type
                ),
                None,
            )
            if vision is not None:
                return replace(
                    arm,
                    vision_provider=record.manifest.name,
                    vision_provider_choice=vision.choice,
                    vision_model=vision.model,
                )
            generation = next(
                (
                    provider
                    for provider in record.manifest.generation_providers
                    if provider.node == node_type
                ),
                None,
            )
            if generation is not None:
                return replace(
                    arm,
                    generation_provider=record.manifest.name,
                    generation_provider_choice=generation.choice,
                    generation_provider_label=generation.label,
                )
            if node_type in generation_nodes and node_type not in record.schema_only:
                return replace(arm, generation_provider=_BUILTIN_GENERATION_PROVIDER)
            return arm

        def primary_worker(record: _PackRecord) -> tuple[Any, _ResidencyDomain]:
            assert record.domain is not None
            owner_worker = (
                record.worker
                if isinstance(record.worker, _SingleJobWorkerPool)
                else record.worker.lanes[0].worker
                if isinstance(record.worker, _ReplicaWorkerPool)
                else record.worker
            )
            # A single-job pool is one owner session spanning all of its
            # ranks: wrapping it in a fresh cuda-labeled domain here would
            # put its pack-level and body arms in two live domains that
            # share the pool's instance token.
            owner_domain = (
                worker_domain(
                    owner_worker,
                    f"{record.delta.pack}:cuda:{record.worker.lanes[0].cuda_index}",
                )
                if isinstance(record.worker, _ReplicaWorkerPool)
                and not isinstance(record.worker, _SingleJobWorkerPool)
                else record.domain
            )
            return owner_worker, owner_domain

        for record in records.values():
            owner_worker, owner_domain = primary_worker(record)
            arm = ArmRecord(
                name=record.delta.pack,
                worker=ArmWorker(owner_worker, None),
                domain=owner_domain,
                instance_token=lambda worker=owner_worker: worker.instance_token,
                default_cache_tag=record.default_cache_tag,
                default_arm=record.manifest.name,
                owner_worker=owner_worker,
                execution_arm=(
                    "comfyui" if record.manifest.name.startswith("dinkster-compat-") else "native"
                ),
                attention_route_token=cast(
                    "AttentionRouteToken | None", owner_worker.attention_route_token
                ),
                attention_capabilities=cast(
                    "AttentionCapabilityEvidence | None", owner_worker.attention_capabilities
                ),
            )
            for node_type in record.delta.schemas:
                arms_by_type[node_type] = (
                    []
                    if node_type in record.schema_only
                    else [annotate_local(arm, record, node_type)]
                )
                records_by_owned_type[node_type] = record
            body_records: list[tuple[tuple[str, ...], ArmRecord]] = []
            for local_arm, node_types in record.body_arms:
                if isinstance(record.worker, _SingleJobWorkerPool):
                    lanes = (_ReplicaLane(-1, cast("Any", record.worker)),)
                elif isinstance(record.worker, _ReplicaWorkerPool):
                    lanes = record.worker.lanes
                else:
                    lanes = (_ReplicaLane(-1, record.worker),)
                for lane in lanes:
                    lane_worker = lane.worker
                    name = f"{record.delta.pack}@{local_arm}"
                    if lane.cuda_index >= 0:
                        name += f":cuda:{lane.cuda_index}"
                    elif isinstance(record.worker, _SingleJobWorkerPool):
                        name += f":single-job:{len(record.worker.lanes)}:{record.worker.mode}"
                    selected = ArmRecord(
                        name=name,
                        worker=ArmWorker(lane_worker, local_arm, name),
                        domain=(
                            worker_domain(lane_worker, name)
                            if lane.cuda_index >= 0
                            else owner_domain
                        ),
                        instance_token=lambda worker=lane_worker: worker.instance_token,
                        default_cache_tag=record.default_cache_tag,
                        default_arm=(name if lane.cuda_index >= 0 else record.manifest.name),
                        owner_worker=record.worker,
                        execution_arm="native" if local_arm == "native" else "comfyui",
                        attention_route_token=cast(
                            "AttentionRouteToken | None", lane_worker.attention_route_token
                        ),
                        attention_capabilities=cast(
                            "AttentionCapabilityEvidence | None", lane_worker.attention_capabilities
                        ),
                    )
                    body_records.append((node_types, selected))
                    for node_type in node_types:
                        if node_type in record.delta.schemas:
                            arms_by_type[node_type].append(
                                annotate_local(selected, record, node_type)
                            )
                        elif node_type not in record.executes:
                            raise RuntimeError(
                                f"same-session arm {local_arm!r} lost node type {node_type!r}"
                            )
            body_records_by_pack[record.manifest.name] = body_records
        for record in records.values():
            assert record.domain is not None
            owner_worker, owner_domain = primary_worker(record)
            arm = ArmRecord(
                name=record.delta.pack,
                worker=ArmWorker(owner_worker, None),
                domain=owner_domain,
                instance_token=lambda worker=owner_worker: worker.instance_token,
                default_cache_tag=record.default_cache_tag,
                default_arm=record.manifest.name,
                owner_worker=owner_worker,
                execution_arm=(
                    "comfyui" if record.manifest.name.startswith("dinkster-compat-") else "native"
                ),
                attention_route_token=owner_worker.attention_route_token,
                attention_capabilities=owner_worker.attention_capabilities,
            )
            for node_type in record.executes:
                if node_type not in records_by_owned_type:
                    raise RuntimeError(f"dispatch topology lost owner for {node_type!r}")
                selected = annotate_local(arm, record, node_type)
                arms_by_type[node_type].append(
                    replace(selected, execution_arm="native")
                    if node_type.startswith("dinkster.")
                    else selected
                )
            for node_types, selected in body_records_by_pack[record.manifest.name]:
                for node_type in node_types:
                    if node_type in record.executes:
                        arms_by_type[node_type].append(annotate_local(selected, record, node_type))

        def annotate_remote(
            arm: ArmRecord,
            node_type: str,
            remote: _RemoteRecord,
            vision_providers: tuple[VisionProvider, ...] | None,
            generation_providers: tuple[GenerationProvider, ...] | None,
            provider_declared: bool,
            generation_declared: bool,
        ) -> ArmRecord:
            vision = next(
                (provider for provider in vision_providers or () if provider.node == node_type),
                None,
            )
            if vision is not None:
                return replace(
                    arm,
                    vision_provider=remote.pack_name,
                    vision_provider_choice=vision.choice,
                    vision_model=vision.model,
                    provider_declared=provider_declared,
                )
            if (
                remote.vision_providers is None
                and node_type in vision_nodes
                and node_type not in remote.delta.schemas
            ):
                return replace(
                    arm,
                    vision_provider=remote.pack_name,
                    provider_declared=False,
                )
            generation = next(
                (provider for provider in generation_providers or () if provider.node == node_type),
                None,
            )
            if generation is not None:
                return replace(
                    arm,
                    generation_provider=remote.pack_name,
                    generation_provider_choice=generation.choice,
                    generation_provider_label=generation.label,
                    provider_declared=generation_declared,
                )
            if node_type in generation_nodes:
                if remote.generation_providers is not None or node_type in remote.delta.schemas:
                    return replace(
                        arm,
                        generation_provider=_BUILTIN_GENERATION_PROVIDER,
                    )
                return replace(
                    arm,
                    generation_provider=remote.pack_name,
                    provider_declared=False,
                )
            return arm

        for remote in (self._remotes if remotes is None else remotes).values():
            remote_worker = remote.worker
            remote_token = cast("AttentionRouteToken | None", remote_worker.attention_route_token)
            compat_pack = remote.pack_name.startswith("dinkster-compat-")
            matching_local = next(
                (record for record in records.values() if record.manifest.name == remote.pack_name),
                None,
            )
            vision_providers = remote.vision_providers
            generation_providers = remote.generation_providers
            provider_declared = vision_providers is not None
            generation_declared = generation_providers is not None
            if vision_providers is None and matching_local is not None:
                vision_providers = matching_local.manifest.vision_providers
            if generation_providers is None and matching_local is not None:
                generation_providers = matching_local.manifest.generation_providers

            remote_arm = ArmRecord(
                name=remote.spec.name,
                # The daemon stamps resident provenance with ITS local pack
                # name, which is meaningless in this composition's arm
                # namespace - requalify this session's outputs to the remote
                # arm so producer-arm affinity checks compare engine names.
                worker=ArmWorker(remote_worker, None, remote.spec.name),
                domain=remote.domain,
                instance_token=lambda worker=remote_worker: worker.instance_token,
                default_cache_tag="unversioned",
                default_arm=remote.spec.name,
                owner_worker=remote_worker,
                execution_arm="comfyui" if compat_pack else "native",
                attention_route_token=remote_token,
                attention_capabilities=cast(
                    "AttentionCapabilityEvidence | None", remote_worker.attention_capabilities
                ),
                remote=remote.spec.name,
                implementation_pack=remote.pack_name,
            )
            for node_type in remote.node_types:
                appended = (
                    replace(remote_arm, execution_arm="native")
                    if compat_pack and node_type.startswith("dinkster.")
                    else remote_arm
                )
                appended = annotate_remote(
                    appended,
                    node_type,
                    remote,
                    vision_providers,
                    generation_providers,
                    provider_declared,
                    generation_declared,
                )
                arms = arms_by_type.get(node_type)
                if arms is None and node_type in self._core_schemas:
                    arms = [
                        ArmRecord(
                            name="local",
                            worker=ArmWorker(self._core_worker, None),
                            domain=self._core_domain,
                            instance_token=lambda: self._core_worker.instance_token,
                            default_cache_tag="unversioned",
                            default_arm="local",
                            owner_worker=self._core_worker,
                            implementation_pack="local",
                            attention_capabilities=cast(
                                "AttentionCapabilityEvidence | None",
                                self._core_worker.attention_capabilities,
                            ),
                        )
                    ]
                    arms_by_type[node_type] = arms
                if arms is None:
                    arms_by_type[node_type] = [appended]
                else:
                    arms.append(appended)
            for local_arm, node_types in remote.body_arms:
                arm_name = f"{remote.spec.name}@{local_arm}"
                selected = ArmRecord(
                    name=arm_name,
                    worker=ArmWorker(remote_worker, local_arm, arm_name),
                    domain=remote.domain,
                    instance_token=lambda worker=remote_worker: worker.instance_token,
                    default_cache_tag="unversioned",
                    default_arm=remote.spec.name,
                    owner_worker=remote_worker,
                    execution_arm="native" if local_arm == "native" else "comfyui",
                    attention_route_token=remote_token,
                    attention_capabilities=cast(
                        "AttentionCapabilityEvidence | None", remote_worker.attention_capabilities
                    ),
                    remote=remote.spec.name,
                    implementation_pack=remote.pack_name,
                )
                for node_type in node_types:
                    arms_by_type[node_type].append(
                        annotate_remote(
                            selected,
                            node_type,
                            remote,
                            vision_providers,
                            generation_providers,
                            provider_declared,
                            generation_declared,
                        )
                    )
        for node_type in generation_nodes:
            arms = arms_by_type.get(node_type)
            if arms is None:
                continue
            arms.sort(
                key=lambda arm: arm.generation_provider not in (None, _BUILTIN_GENERATION_PROVIDER)
            )
        return MappingProxyType(
            {node_type: tuple(arms) for node_type, arms in arms_by_type.items()}
        )

    def _dispatch_routes(self, topology: Topology, node_types: Sequence[str]) -> dict[str, Worker]:
        routes: dict[str, Worker] = {}
        for node_type in node_types:
            arms = topology[node_type]
            if not arms:
                continue
            # prepare() reaches every arm. Arm preparation must remain
            # side-effect-free and must never allocate runtime state such
            # as models or VRAM; a future native arm depends on this.
            routes[node_type] = DispatchWorker(
                {arm.name: arm.worker for arm in arms},
                resolve_owner=lambda token, snapshot=topology: self._resolve_owner(snapshot, token),
                arm_domains={arm.name: arm.domain for arm in arms},
                default_arms={arm.name: arm.default_arm for arm in arms},
            )
        return routes

    def _validate_body_arms(self, manifest: PackManifest, worker: Any) -> None:
        expected = {arm: tuple(sorted(node_types)) for arm, node_types in manifest.arms}
        if manifest.arms and worker.body_arms is None:
            raise CompositionError(
                f"{manifest.name}: worker hello omitted required bodyArms capability"
            )
        actual = worker.body_arms or {}
        if actual != expected:
            raise CompositionError(
                f"{manifest.name}: worker hello bodyArms does not exactly match [pack.arms]"
            )
        if (
            "native" in expected
            and not getattr(worker, "cold", False)
            and worker.attention_route_token is None
        ):
            raise CompositionError(
                f"{manifest.name}: native worker hello omitted attention route evidence"
            )

    def _validate_schema_only(self, manifest: PackManifest, worker: Any) -> None:
        missing = tuple(
            node_type for node_type in manifest.schema_only if node_type not in worker.schemas
        )
        if missing:
            raise CompositionError(
                f"{manifest.name}: schema-only node types are missing from the worker schemas: "
                f"{', '.join(missing)}"
            )

    def _validate_executes(
        self,
        manifest: PackManifest,
        worker: Any,
        *,
        replacing: str | None = None,
    ) -> None:
        for node_type in manifest.executes:
            arm_schema = worker.schemas.get(node_type)
            if arm_schema is None:
                raise CompositionError(
                    f"{manifest.name}: executes claim {node_type!r} is not "
                    "present in the worker hello schemas"
                )
            owner = self._owners.get(node_type)
            if (
                owner is None
                or owner == CORE_PACK_ID
                or owner == replacing
                or owner in self._remotes
            ):
                raise CompositionError(
                    f"{manifest.name}: executes claim {node_type!r} has no "
                    "owning pack already serving it"
                )
            owner_schema = self.composition.schemas[node_type]
            if schema_signature(arm_schema) != schema_signature(owner_schema):
                raise CompositionError(
                    f"{manifest.name}: executes claim {node_type!r} does not "
                    f"match owning pack {owner!r}'s schema signature"
                )

    def _owner_blockers(
        self, owner: str, schemas: Mapping[str, NodeSchema] | None
    ) -> dict[str, list[str]]:
        blockers: dict[str, list[str]] = {}
        record = self._records[owner]
        for node_type in record.delta.schemas:
            for other_name, other in self._records.items():
                if other_name == owner or node_type not in other.executes:
                    continue
                new_schema = schemas.get(node_type) if schemas is not None else None
                if new_schema is None or schema_signature(new_schema) != schema_signature(
                    other.worker.schemas[node_type]
                ):
                    blockers.setdefault(other_name, []).append(node_type)
            for remote_name, remote in self._remotes.items():
                if node_type not in remote.node_types:
                    continue
                new_schema = schemas.get(node_type) if schemas is not None else None
                if new_schema is None or schema_signature(new_schema) != schema_signature(
                    remote.worker.schemas[node_type]
                ):
                    blockers.setdefault(remote_name, []).append(node_type)
        return blockers

    def _raise_owner_blockers(
        self, owner: str, blockers: Mapping[str, Sequence[str]], action: str
    ) -> None:
        if not blockers:
            return
        detail = ", ".join(
            f"{name} ({', '.join(types)})" for name, types in sorted(blockers.items())
        )
        raise CompositionError(
            f"cannot {action} owning pack {owner!r}; execution providers still "
            f"depend on its schemas: {detail}; remove or reload them first"
        )

    async def add_pack(self, entry: PackSpec | Path | str) -> PackDelta:
        """Merge one pack's catalog or live announcement into the surface.

        Validates against everything composed SO FAR - the same rules
        whether packs arrive all at once or one at a time - and returns
        the delta for a live server to announce. The delta's packs map may
        repeat an existing entry only when identical (two compat workers
        both contributing the shared "comfy" entry)."""
        async with self._mutate:
            return await self._add_pack_locked(entry)

    async def _add_pack_locked(self, entry: PackSpec | Path | str) -> PackDelta:
        composition = self.composition
        spec = entry if isinstance(entry, PackSpec) else PackSpec(manifest=entry)
        self.validate_specs((spec,))
        manifest = load_manifest(resolve_manifest_path(spec.manifest))
        catalog = read_catalog(manifest) if spec.require_catalog else None
        if (
            catalog is not None
            and spec.worker_group is not None
            and any(read_catalog(load_manifest(path)) is None for path in spec.group_manifests)
        ):
            catalog = None
        if spec.require_catalog and catalog is None:
            raise CompositionError(
                f"{manifest.name}: schema catalog is missing or stale; "
                "run dinkster-pack prepare-catalogs "
                "(or dinkster-pack prepare-catalogs --defaults) before serving"
            )
        canonical = canonical_name(manifest.name)
        reused_in_process: _PackRecord | None = None
        _order, staged_generation = self._resolve_contract_inputs(
            self._contract_inputs(replacing=(manifest, spec))
        )
        tenant_proxy: _PackTenantRegistry | None = None
        component_publisher: NativeComponentPublisher | None = None
        new_component_publisher: NativeComponentPublisher | None = None
        if spec.in_process:
            if self._private_staging:
                reused_in_process = self._reusable_in_process.get(canonical)
                if reused_in_process is None or reused_in_process.spec != spec:
                    raise CompositionError(
                        f"cannot stage new or changed in-process pack {manifest.name!r} "
                        "for live activation; restart dinkster-serve"
                    )
            if manifest.extension_declared:
                raise CompositionError(
                    f"in-process pack {manifest.name!r} may contribute nodes only; "
                    "sampler, extension, and model-registry contributions are refused"
                )
            needs_residency = (
                manifest.reservations_entry is not None or manifest.consumers_entry is not None
            )
            if needs_residency and self._tenant_registry is None:
                raise CompositionError(
                    f"in-process pack {manifest.name!r} declares residency needs, but "
                    "the engine provides no model tenant registry"
                )
            if needs_residency and not spec.runtime_pins:
                raise CompositionError(
                    f"in-process pack {manifest.name!r} has no exact torch/dinkster-aimdo "
                    "generation baseline"
                )
            if spec.runtime_pins:
                if not self._torch_capable():
                    raise CompositionError(
                        f"in-process pack {manifest.name!r} requires a torch-capable "
                        "serve environment"
                    )
                if set(spec.runtime_pins) != {"torch", "dinkster-aimdo"}:
                    raise CompositionError(
                        f"in-process pack {manifest.name!r} has no exact torch/dinkster-aimdo "
                        "generation baseline"
                    )
                try:
                    actual_versions = dict(self._runtime_versions())
                except Exception as exc:
                    raise CompositionError(
                        f"cannot verify in-process runtime pins for {manifest.name!r}: {exc}"
                    ) from exc
                if actual_versions != dict(spec.runtime_pins):
                    raise CompositionError(
                        f"in-process runtime drift for {manifest.name!r}: expected "
                        f"{dict(spec.runtime_pins)!r}, installed {actual_versions!r}"
                    )
                component_publisher = (
                    reused_in_process.component_publisher
                    if reused_in_process is not None
                    else NativeComponentPublisher()
                )
                if reused_in_process is None:
                    new_component_publisher = component_publisher
            if self._tenant_registry is not None and reused_in_process is None:
                tenant_proxy = _PackTenantRegistry(manifest.name, self._tenant_registry)
        group_owner: GroupIsolatedWorker | None = None
        group_new = False
        contract: tuple[object, ...] = ()
        if spec.worker_group is not None:
            if not spec.group_manifests:
                raise CompositionError(f"worker group {spec.worker_group!r} has no group manifests")
            group_names = tuple(load_manifest(path).name for path in spec.group_manifests)
            if manifest.name not in group_names:
                raise CompositionError(
                    f"pack {manifest.name!r} is not a member of worker group {spec.worker_group!r}"
                )
            contract = _worker_group_contract(spec, self._worker_env)
            previous = self._group_contracts.get(spec.worker_group)
            if previous is not None and previous != contract:
                raise CompositionError(
                    f"worker group {spec.worker_group!r} members have incompatible launch contracts"
                )
            group_owner = self._group_owners.get(spec.worker_group)
        if manifest.name == CORE_PACK_ID:
            raise CompositionError(f"{manifest.path}: pack name {CORE_PACK_ID!r} is reserved")
        other_spelling = self._seen_names.get(canonical)
        if other_spelling is not None:
            same = (
                "duplicate pack name"
                if other_spelling == manifest.name
                else f"pack name collides with {other_spelling!r} - "
                f"separators '-', '_' and '.' are one identity"
            )
            raise CompositionError(f"{manifest.path}: {same} {manifest.name!r}")
        for claim in manifest.namespaces:
            root = reserved_root(claim)
            if root is not None and not spec.trust_reserved:
                raise CompositionError(
                    f"{manifest.path}: namespace claim {claim!r} falls "
                    f"under the reserved root {root!r}; composing it "
                    f"requires explicit host trust "
                    f"(PackSpec(trust_reserved=True))"
                )
            for held, holder in self._claim_owners.items():
                if not claims_conflict(held, claim):
                    continue
                # Reserved roots are host-multiplexed across trusted packs;
                # concrete node type collisions still fail below.
                if _trusted_reserved_claims_can_overlap(held, claim):
                    continue
                raise CompositionError(
                    f"namespace {claim!r} claimed by {manifest.name!r} "
                    f"overlaps {held!r} claimed by {holder!r}"
                )
        # The plain-path default is a raw development pack: expose where
        # the bytes genuinely live (source) but no version/digest pin -
        # nothing pinned them, and omission MEANS unpinned on the wire.
        # Specs that provide their own packs table (installer-managed,
        # compat workers) carry their own provenance untouched.
        spec_packs = (
            dict(spec.packs)
            if spec.packs is not None
            else {
                manifest.name: replace(
                    pack_info_from_manifest(manifest),
                    source=f"local:{manifest.root.resolve()}",
                )
            }
        )
        # Validate against a STAGED copy; nothing below this line mutates
        # composed state until the commit point at the end. add_pack is
        # atomic: a pack that fails to start or validate leaves the
        # surface, the pack table, the name registry, and the claim
        # registry exactly as they were, so a continue-on-failure host
        # serves the survivors from an unpolluted composition (only the
        # dead worker process remains, reaped by close()).
        staged_packs = dict(composition.packs)
        for pack_id, info in spec_packs.items():
            _merge_pack_entry(staged_packs, pack_id, info, str(manifest.path))
        staged_registry = composition._registry.copy()
        worker: Any = None
        if spec.in_process:
            pack_tenant_registry = (
                reused_in_process.tenant_registrations
                if reused_in_process is not None
                else tenant_proxy
            )

            @contextlib.contextmanager
            def pack_context() -> Generator[None, None, None]:
                with contextlib.ExitStack() as contexts:
                    contexts.enter_context(use_model_tenant_registry(pack_tenant_registry))
                    if component_publisher is not None:
                        inference_torch = importlib.import_module("dinkster_inference_torch")
                        contexts.enter_context(
                            inference_torch.use_component_publisher(component_publisher)
                        )
                    yield

            loaded: list[Any] = []

            async def start_in_process() -> Any:
                current_registry = composition._registry.copy()
                loaded_worker, _, _, _ = load_host_pack(
                    manifest,
                    pack_context=pack_context,
                    import_from_pack_root=True,
                    registry=current_registry,
                )
                loaded.append(loaded_worker)
                self._validate_body_arms(manifest, loaded_worker)
                assert catalog is not None
                if not worker_declarations_match_catalog(loaded_worker, catalog):
                    raise CompositionError(f"{manifest.name}: declarations changed; rerun doctor")
                composition._registry.replace_from(current_registry)
                loaded_worker.bind_registry(composition._registry)
                return loaded_worker

            async def close_in_process() -> None:
                for loaded_worker in loaded:
                    await loaded_worker.close()

            try:
                if catalog is not None:
                    worker = LazyWorker(manifest, catalog, start_in_process, close_in_process)
                else:
                    worker, _, _, _ = load_host_pack(
                        manifest,
                        pack_context=pack_context,
                        import_from_pack_root=True,
                        registry=staged_registry,
                    )
            except BaseException:
                if tenant_proxy is not None:
                    await tenant_proxy.close()
                if new_component_publisher is not None:
                    new_component_publisher.close()
                raise
        elif spec.worker_group is None:
            worker = self._isolated_worker(spec, manifest, composition._registry)
        if group_owner is not None:
            worker = group_owner.members[manifest.name]
        elif spec.worker_group is not None:
            group_manifests = tuple(load_manifest(path) for path in spec.group_manifests)
            group_env = self._worker_environment(spec, group_manifests)
            group_owner = GroupIsolatedWorker(
                spec.worker_group,
                spec.group_manifests,
                composition._registry,
                python=spec.python,
                extra_env=group_env,
                launcher=self._sandbox_launcher(spec, group_manifests, group_env),
                aimdo_arm=spec.aimdo,
                vram_budgets=spec.vram_budgets,
                reserve_vram=spec.reserve_vram,
                comfy_args=spec.comfy_args,
                start_timeout=spec.start_timeout,
                on_diagnostic=self._diagnostic_listener,
                governor=self._governor,
                reservations=self._reservations,
                telemetry=self._telemetry,
                headroom_mirror=self._headroom_mirror,
                on_schema_reload=self._schema_reload_requested,
            )
            worker = group_owner.members[manifest.name]
            group_new = True
            composition._isolated.append(group_owner)  # type: ignore[arg-type]
        elif not spec.in_process:
            composition._isolated.append(worker)
        assert worker is not None
        if catalog is not None and spec.worker_group is not None:
            assert group_owner is not None
            worker = self._catalog_group_worker(spec.worker_group, group_owner, manifest, catalog)
        try:
            if group_new and catalog is None:
                assert group_owner is not None
                await group_owner.start()
            elif spec.worker_group is None and not spec.in_process:
                await cast("Any", worker).start()
            self._validate_body_arms(manifest, worker)
            self._validate_schema_only(manifest, worker)
            self._validate_executes(manifest, worker)
            delta_schemas: dict[str, NodeSchema] = {}
            delta_node_packs: dict[str, str] = {}
            for node_type, schema in worker.schemas.items():
                if node_type in manifest.executes:
                    continue
                if not any(claim_covers(claim, node_type) for claim in manifest.namespaces):
                    raise CompositionError(
                        f"{manifest.name}: node type {node_type!r} is outside "
                        f"the pack's declared namespaces "
                        f"({', '.join(manifest.namespaces)})"
                    )
                owner = self._owners.get(node_type)
                if owner is not None:
                    raise CompositionError(
                        f"node type {node_type!r} is declared by both "
                        f"{owner!r} and {manifest.name!r}"
                    )
                pack_id = spec.attribute(node_type) if spec.attribute else manifest.name
                if pack_id not in spec_packs:
                    raise CompositionError(
                        f"{manifest.name}: node type {node_type!r} attributes to "
                        f"{pack_id!r}, which is not among this spec's pack entries"
                    )
                delta_schemas[node_type] = schema
                delta_node_packs[node_type] = pack_id
            _validate_registry_carriers(
                manifest.name,
                spec_packs,
                {**composition.node_packs, **delta_node_packs},
            )
            delta_choices: dict[str, tuple[str, ...]] = {}
            for choice_id, values in worker.combo_choices.items():
                if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                    raise CompositionError(
                        f"{manifest.name}: choice list {choice_id!r} is outside "
                        f"the pack's declared namespaces "
                        f"({', '.join(manifest.namespaces)})"
                    )
                choice_owner = self._choice_owners.get(choice_id)
                if choice_owner is not None:
                    raise CompositionError(
                        f"choice list {choice_id!r} is declared by both "
                        f"{choice_owner!r} and {manifest.name!r}"
                    )
                delta_choices[choice_id] = values
            delta_lazy: dict[str, LazyChoiceFetcher] = {}
            for choice_id in worker.lazy_choice_ids:
                if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                    raise CompositionError(
                        f"{manifest.name}: choice list {choice_id!r} is outside "
                        f"the pack's declared namespaces "
                        f"({', '.join(manifest.namespaces)})"
                    )
                choice_owner = self._choice_owners.get(choice_id)
                if choice_owner is not None:
                    raise CompositionError(
                        f"choice list {choice_id!r} is declared by both "
                        f"{choice_owner!r} and {manifest.name!r}"
                    )
                delta_lazy[choice_id] = _lazy_choice_fetcher(worker, choice_id, self)
            _validate_remote_authority(manifest.name, delta_schemas, delta_choices, delta_lazy)
            delta_compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = {}
            for source_name, diagnostic in worker.compat_skips.items():
                pack_id = spec.attribute(source_name) if spec.attribute else manifest.name
                if pack_id not in spec_packs:
                    raise CompositionError(
                        f"{manifest.name}: compat skip {source_name!r} attributes "
                        f"to {pack_id!r}, which is not among this spec's pack entries"
                    )
                node_id = source_name.removeprefix(pack_id + ".")
                owner = self._compat_skip_owners.get((pack_id, node_id))
                if owner is not None:
                    raise CompositionError(
                        f"compat skip {pack_id}.{node_id!r} is declared by both "
                        f"{owner!r} and {manifest.name!r}"
                    )
                delta_compat_skips.setdefault(pack_id, {})[node_id] = diagnostic
        except BaseException:
            if group_new:
                assert group_owner is not None
                composition._isolated.remove(group_owner)  # type: ignore[arg-type]
                await group_owner.close()
            elif spec.worker_group is None and not spec.in_process:
                composition._isolated.remove(worker)
                await cast("Any", worker).close()
            if tenant_proxy is not None:
                await tenant_proxy.close()
            if new_component_publisher is not None:
                new_component_publisher.close()
            raise
        delta = PackDelta(
            pack=manifest.name,
            schemas=delta_schemas,
            packs=spec_packs,
            node_packs=delta_node_packs,
            schema_owners=dict.fromkeys(delta_schemas, manifest.name),
            choice_owners=dict.fromkeys((*delta_choices, *delta_lazy), manifest.name),
            choices=delta_choices,
            lazy_choices=delta_lazy,
            compat_skips=delta_compat_skips,
        )
        if spec.in_process:
            domain = self._in_process_domain or self._core_domain
        else:
            domain = next(
                (
                    current.domain
                    for current in self._records.values()
                    if spec.worker_group is not None
                    and current.spec.worker_group == spec.worker_group
                ),
                None,
            ) or _ResidencyDomain(worker, spec.worker_group or manifest.name)
        record = _PackRecord(
            spec=spec,
            worker=worker,
            delta=delta,
            claims=tuple(manifest.namespaces),
            canonical=canonical,
            manifest=manifest,
            executes=manifest.executes,
            schema_only=manifest.schema_only,
            default_cache_tag=self._execution_identity(manifest, spec_packs, spec),
            body_arms=manifest.arms,
            domain=domain,
            asset_roots=_pack_asset_roots(spec, manifest, spec_packs),
            extension=manifest.extension if manifest.extension_declared else None,
            extension_contributions=cast(
                "tuple[tuple[ExtensionScope, ContributionSurfaceDescriptor], ...]",
                worker.extension_contributions,
            ),
            tenant_registrations=(
                reused_in_process.tenant_registrations
                if reused_in_process is not None
                else tenant_proxy
            ),
            owns_tenant_registrations=reused_in_process is None,
            component_publisher=component_publisher,
            owns_component_publisher=reused_in_process is None,
        )
        staged_records = {**self._records, manifest.name: record}
        snapshot: ExtensionSnapshot | None = None
        topology: Topology | None = None
        try:
            topology = self._build_topology(staged_records)
            (
                snapshot,
                sampler_registry,
                scheduler_registry,
                graph_compiler_registry,
                graph_compile_transport,
                inference_unavailable,
            ) = await self._build_extension_snapshot(staged_records, topology)
            derived_choices = self._validated_derived_choices(
                sampler_registry, scheduler_registry, staged_records, topology
            )
            staged_registry.remove_relayed_renditions(manifest.name)
            if not spec.in_process:
                self._register_worker_renditions(staged_registry, manifest.name, worker)
            if spec.host_types is not None:
                spec.host_types(staged_registry)
        except BaseException:
            if snapshot is not None and topology is not None:
                await self._rollback_unpublished_generation(snapshot, topology)
            if group_new:
                assert group_owner is not None
                composition._isolated.remove(group_owner)  # type: ignore[arg-type]
                await group_owner.close()
            elif spec.worker_group is None and not spec.in_process:
                composition._isolated.remove(worker)
                await cast("Any", worker).close()
            if tenant_proxy is not None:
                await tenant_proxy.close()
            if new_component_publisher is not None:
                new_component_publisher.close()
            raise
        old_derived_choices = self._current_derived_choices()
        # Publish validated declarations and routes atomically.
        if isinstance(worker, LazyWorker) and (spec.in_process or spec.worker_group is not None):
            composition._isolated.append(worker)
        composition._registry.replace_from(staged_registry)
        if spec.in_process and not isinstance(worker, LazyWorker):
            cast("InProcessWorker", worker).bind_registry(composition._registry)
        self._seen_names[canonical] = manifest.name
        for claim in manifest.namespaces:
            self._claim_owners.setdefault(claim, manifest.name)
        composition.packs.update(staged_packs)
        for node_type, schema in delta_schemas.items():
            composition.schemas[node_type] = schema
            self._owners[node_type] = manifest.name
            composition.node_packs[node_type] = delta_node_packs[node_type]
        composition.choices.update(delta_choices)
        for choice_id in delta_choices:
            self._choice_owners[choice_id] = manifest.name
        composition.lazy_choices.update(delta_lazy)
        for choice_id in delta_lazy:
            self._choice_owners[choice_id] = manifest.name
        for pack_id, skips in delta_compat_skips.items():
            composition.compat_skips.setdefault(pack_id, {}).update(skips)
            for node_id in skips:
                self._compat_skip_owners[(pack_id, node_id)] = manifest.name
        changed_types = (*manifest.executes, *delta.schemas)
        published_types = tuple(
            node_type
            for node_type in dict.fromkeys(changed_types)
            if topology.get(node_type)
            and (node_type in delta.schemas or not self._topology.get(node_type))
        )
        routed_executes = tuple(
            node_type for node_type in manifest.executes if self._routing.has_route(node_type)
        )
        self._routing.swap_routes(
            routed_executes,
            self._dispatch_routes(topology, changed_types),
        )
        old_inference_worker = self._sampling_worker(self._topology)
        new_inference_worker = self._sampling_worker(topology)
        retired_inference_worker = (
            old_inference_worker
            if old_inference_worker is not None and old_inference_worker is not new_inference_worker
            else None
        )
        self._records[manifest.name] = record
        composition.generation = staged_generation
        if spec.in_process and self._in_process_domain is None:
            self._in_process_domain = domain
        if tenant_proxy is not None:
            composition._tenant_registries.append(tenant_proxy)
        if new_component_publisher is not None:
            composition._component_publishers.append(new_component_publisher)
        if group_new:
            assert group_owner is not None and spec.worker_group is not None
            self._group_owners[spec.worker_group] = group_owner
            self._group_contracts[spec.worker_group] = contract
        self._topology = topology
        self._sync_execution_arms()
        self._apply_derived_choices(derived_choices)
        changed_derived_choices = self._changed_derived_choices(old_derived_choices)
        delta = replace(
            delta,
            schemas={node_type: composition.schemas[node_type] for node_type in published_types},
            node_packs={
                node_type: composition.node_packs[node_type] for node_type in published_types
            },
            schema_owners={node_type: self._owners[node_type] for node_type in published_types},
            choice_owners=self._published_choice_owners(
                (*delta.choices, *delta.lazy_choices, *changed_derived_choices)
            ),
            execution_arms={
                node_type: composition.execution_arms[node_type]
                for node_type in dict.fromkeys(changed_types)
                if topology.get(node_type)
            },
            derived_choices=changed_derived_choices,
        )
        self._apply_inference_unavailable(inference_unavailable, delta.packs)
        self._publish_runtime(
            topology,
            snapshot,
            sampler_registry,
            graph_compiler_registry,
            graph_compile_transport,
        )
        await self._commit_published_generation(snapshot, retired_inference_worker)
        self._rebuild_asset_catalog()
        return delta

    async def add_remote(self, spec: RemoteSpec) -> PackDelta:
        """Connect one configured remote worker daemon and merge its
        announced surface into the composition.

        The same atomicity contract as :meth:`add_pack`: any startup or
        validation failure closes the connection (never the daemon) and
        merges nothing, so a continue-on-failure host serves the survivors
        from an unpolluted composition. The remote's facts come from the
        daemon's hello - there is no manifest on this side."""
        async with self._mutate:
            return await self._add_remote_locked(spec)

    async def _add_remote_locked(self, spec: RemoteSpec) -> PackDelta:
        composition = self.composition
        source = f"remote:{spec.host}:{spec.port}"
        if spec.name == CORE_PACK_ID or canonical_name(spec.name) == "local":
            raise CompositionError(f"{source}: remote worker name {spec.name!r} is reserved")
        canonical = canonical_name(spec.name)
        other_spelling = self._seen_names.get(canonical)
        if other_spelling is not None:
            same = (
                "duplicate pack name"
                if other_spelling == spec.name
                else f"remote worker name collides with {other_spelling!r} - "
                f"separators '-', '_' and '.' are one identity"
            )
            raise CompositionError(f"{source}: {same} {spec.name!r}")
        try:
            token = spec.token_file.read_text("utf-8").strip()
        except OSError as exc:
            raise CompositionError(
                f"{source}: cannot read token file {spec.token_file}: {exc}"
            ) from exc
        if spec.tls_ca_file is not None:
            # Checked before dialing so a misconfigured path fails with the
            # file named, not a handshake error.
            try:
                spec.tls_ca_file.read_bytes()
            except OSError as exc:
                raise CompositionError(
                    f"{source}: cannot read TLS CA file {spec.tls_ca_file}: {exc}"
                ) from exc
        # One pack-table entry per remote. No version/digest - omission
        # means unpinned, which is honest for a remote surface this side
        # never installed; the source field carries where it runs.
        worker = RemoteWorker(
            spec.host,
            spec.port,
            token,
            composition._registry,
            name=spec.name,
            on_diagnostic=self._diagnostic_listener,
            governor=self._governor,
            reservations=self._reservations,
            telemetry=self._telemetry,
            asset_endpoint=self._remote_asset_endpoint,
            asset_endpoint_token=self._remote_asset_token,
            value_store=self._remote_value_store,
            tls_ca_file=spec.tls_ca_file,
            engine_instance_id=self._engine_instance_id,
        )
        composition._isolated.append(worker)
        snapshot: ExtensionSnapshot | None = None
        topology: Topology | None = None
        try:
            await worker.start()
            surface = self._validated_remote_surface(spec, worker)
            info = PackInfo(
                display_name=spec.name,
                source=source,
                comfy_aliases=surface.comfy_aliases,
                comfy_groups=surface.comfy_groups,
            )
            staged_packs = dict(composition.packs)
            _merge_pack_entry(staged_packs, spec.name, info, source)
            delta = PackDelta(
                pack=spec.name,
                schemas=surface.schemas,
                packs={spec.name: info},
                node_packs=surface.node_packs,
                schema_owners=dict.fromkeys(surface.schemas, spec.name),
                choice_owners=dict.fromkeys(
                    (*surface.choices, *surface.lazy_choice_ids), spec.name
                ),
                choices=surface.choices,
                lazy_choices={
                    choice_id: _lazy_choice_fetcher(worker, choice_id, self)
                    for choice_id in surface.lazy_choice_ids
                },
                compat_skips=surface.compat_skips,
            )
            record = _RemoteRecord(
                spec=spec,
                worker=worker,
                delta=delta,
                domain=_ResidencyDomain(worker, spec.name),
                node_types=tuple(surface.composed),
                pack_name=worker.pack,
                body_arms=surface.body_arms,
                vision_providers=surface.vision_providers,
                generation_providers=surface.generation_providers,
                instance_token=worker.instance_token,
            )
            staged_remotes = {**self._remotes, spec.name: record}
            topology = self._build_topology(self._records, staged_remotes)
            # Remotes contribute nothing to the extension snapshot (their
            # contributions were refused above), so deriving it over the
            # unchanged records reproduces the published generation.
            (
                snapshot,
                sampler_registry,
                scheduler_registry,
                graph_compiler_registry,
                graph_compile_transport,
                inference_unavailable,
            ) = await self._build_extension_snapshot(self._records, topology)
            derived_choices = self._validated_derived_choices(
                sampler_registry, scheduler_registry, self._records, topology
            )
            staged_registry = composition._registry.copy()
            staged_registry.remove_relayed_renditions(spec.name)
            self._register_worker_renditions(staged_registry, spec.name, worker)
        except BaseException:
            if snapshot is not None and topology is not None:
                await self._rollback_unpublished_generation(snapshot, topology)
            composition._isolated.remove(worker)
            await worker.close()
            raise
        old_derived_choices = self._current_derived_choices()
        # Commit point: every validation passed, the connection is up -
        # now merge everything at once (mirrors add_pack).
        self._seen_names[canonical] = spec.name
        composition._registry.replace_from(staged_registry)
        composition.packs.update(staged_packs)
        for node_type, schema in surface.schemas.items():
            composition.schemas[node_type] = schema
            self._owners[node_type] = spec.name
            composition.node_packs[node_type] = surface.node_packs[node_type]
        composition.choices.update(surface.choices)
        for choice_id in surface.choices:
            self._choice_owners[choice_id] = spec.name
        composition.lazy_choices.update(delta.lazy_choices)
        for choice_id in delta.lazy_choices:
            self._choice_owners[choice_id] = spec.name
        for pack_id, skips in surface.compat_skips.items():
            composition.compat_skips.setdefault(pack_id, {}).update(skips)
            for node_id in skips:
                self._compat_skip_owners[(pack_id, node_id)] = spec.name
        changed_types = record.node_types
        published_types = tuple(
            node_type
            for node_type in changed_types
            if topology.get(node_type)
            and node_type not in self._core_schemas
            and (node_type in delta.schemas or not self._topology.get(node_type))
        )
        replaced_types = tuple(
            node_type for node_type in changed_types if self._routing.has_route(node_type)
        )
        self._routing.swap_routes(
            replaced_types,
            self._dispatch_routes(topology, changed_types),
        )
        self._remotes[spec.name] = record
        self._topology = topology
        self._sync_execution_arms()
        self._apply_derived_choices(derived_choices)
        changed_derived_choices = self._changed_derived_choices(old_derived_choices)
        delta = replace(
            delta,
            schemas={node_type: composition.schemas[node_type] for node_type in published_types},
            node_packs={
                node_type: composition.node_packs[node_type] for node_type in published_types
            },
            schema_owners={node_type: self._owners[node_type] for node_type in published_types},
            choice_owners=self._published_choice_owners(
                (*delta.choices, *delta.lazy_choices, *changed_derived_choices)
            ),
            execution_arms={
                node_type: composition.execution_arms[node_type] for node_type in changed_types
            },
            derived_choices=changed_derived_choices,
        )
        self._apply_inference_unavailable(inference_unavailable, delta.packs)
        self._publish_runtime(
            topology,
            snapshot,
            sampler_registry,
            graph_compiler_registry,
            graph_compile_transport,
        )
        # The records did not change, so the sampling worker cannot have
        # rotated: no generation retires with this announcement.
        await self._commit_published_generation(snapshot, None)
        return delta

    def _validated_remote_surface(
        self, spec: RemoteSpec, worker: RemoteWorker, *, replacing: str | None = None
    ) -> _RemoteSurface:
        """Validate one dialed daemon's hello against the composed surface.

        ``replacing`` names a composed remote whose own prior claims do not
        count as collisions - the reattach path, where the new announcement
        replaces that record's whole surface. With ``replacing=None`` these
        are exactly add_remote's rules."""
        composition = self.composition
        if worker.extension_contributions:
            raise CompositionError(
                f"remote worker {spec.name!r}: remote extension contributions are not composed"
            )
        announced = dict(worker.schemas)
        body_arms = _validated_remote_body_arms(spec.name, worker, announced)
        if spec.nodes is not None:
            missing = sorted(set(spec.nodes) - set(announced))
            if missing:
                raise CompositionError(
                    f"remote worker {spec.name!r}: configured nodes "
                    f"{', '.join(missing)} are not served by "
                    f"{spec.host}:{spec.port}"
                )
        composed = (
            announced
            if spec.nodes is None
            else {node_type: announced[node_type] for node_type in spec.nodes}
        )
        vision_providers = worker.vision_providers
        generation_providers = worker.generation_providers
        declared_asset_ids = {asset.id for asset in worker.declared_assets}
        if vision_providers is not None and generation_providers is not None:
            overlap = sorted(
                {provider.node for provider in vision_providers}
                & {provider.node for provider in generation_providers}
            )
            if overlap:
                raise CompositionError(
                    f"remote worker {spec.name!r}: nodes cannot use both vision and generation "
                    f"provider metadata: {', '.join(overlap)}"
                )

        def validate_provider(provider: VisionProvider | GenerationProvider, kind: str) -> None:
            if provider.node not in announced:
                raise CompositionError(
                    f"remote worker {spec.name!r}: {kind} provider names unannounced node "
                    f"{provider.node!r}"
                )
            if provider.node not in composed:
                return
            schema = self._core_schemas.get(provider.node)
            if schema is None:
                schema = next(
                    (
                        record.delta.schemas[provider.node]
                        for record in self._records.values()
                        if provider.node in record.delta.schemas
                    ),
                    announced[provider.node],
                )
            provider_input = schema.input("provider")
            expected_route = f"/api/choices/{provider.choice}"
            if (
                provider_input is None
                or provider_input.required
                or not isinstance(provider_input.widget, ComboWidget)
                or bool(provider_input.widget.options)
                or provider_input.widget.remote_route != expected_route
                or not provider_input.hidden
                or provider_input.advanced
            ):
                raise CompositionError(
                    f"remote worker {spec.name!r}: {kind} provider declaration does not match "
                    f"{provider.node!r} provider input"
                )
            choices = self._core_choices.get(provider.choice)
            if choices is None:
                choices = next(
                    (
                        record.delta.choices[provider.choice]
                        for record in self._records.values()
                        if provider.choice in record.delta.choices
                    ),
                    None,
                )
            if choices is None:
                choices = worker.combo_choices.get(provider.choice)
            if choices is None or choices:
                raise CompositionError(
                    f"remote worker {spec.name!r}: {kind} provider choice "
                    f"{provider.choice!r} must have an empty owner declaration"
                )
            if kind != "vision":
                return
            vision = cast("VisionProvider", provider)
            missing_assets = sorted(set(vision.artifacts) - declared_asset_ids)
            if missing_assets:
                raise CompositionError(
                    f"remote worker {spec.name!r}: vision provider for {provider.node!r} "
                    f"references undeclared assets: {', '.join(missing_assets)}"
                )
            model_input = schema.input("model")
            if model_input is not None:
                model_options = (
                    {
                        option.value if isinstance(option, ComboOption) else option
                        for option in model_input.widget.options
                    }
                    if isinstance(model_input.widget, ComboWidget)
                    else set()
                )
                if vision.model is None or vision.model not in model_options - {"auto"}:
                    raise CompositionError(
                        f"remote worker {spec.name!r}: vision provider model does not match "
                        f"{provider.node!r} model choices"
                    )
            elif vision.model is not None:
                raise CompositionError(
                    f"remote worker {spec.name!r}: vision provider declares a model for "
                    f"{provider.node!r}, which has no model choice"
                )

        for provider in vision_providers or ():
            validate_provider(provider, "vision")
        if generation_providers and worker.pack == _BUILTIN_GENERATION_PROVIDER:
            raise CompositionError(
                f"remote worker {spec.name!r}: generation provider pack name "
                f"{_BUILTIN_GENERATION_PROVIDER!r} is reserved"
            )
        for provider in generation_providers or ():
            validate_provider(provider, "generation")
        vision_providers = (
            None
            if vision_providers is None
            else tuple(provider for provider in vision_providers if provider.node in composed)
        )
        generation_providers = (
            None
            if generation_providers is None
            else tuple(provider for provider in generation_providers if provider.node in composed)
        )
        # No namespace-claim check for remotes: the daemon side already
        # enforced its own manifest claims, and engine-side claims do
        # not cross the wire. Reserved roots still gate on host trust.
        delta_schemas: dict[str, NodeSchema] = {}
        delta_node_packs: dict[str, str] = {}
        for node_type, schema in composed.items():
            root = reserved_root(node_type)
            if root is not None and not spec.trust_reserved:
                raise CompositionError(
                    f"remote worker {spec.name!r}: node type {node_type!r} "
                    f"falls under the reserved root {root!r}; composing it "
                    "requires explicit host trust (trust_reserved)"
                )
            owner = self._owners.get(node_type)
            if owner is not None and owner != replacing:
                # Signature, not full equality: hello schemas for
                # executes-claimed types carry only the computational
                # interface, and presentation prose never changes what a
                # node computes (same basis as _validate_executes).
                if schema_signature(composition.schemas[node_type]) != schema_signature(schema):
                    raise CompositionError(
                        f"remote worker {spec.name!r}: node type {node_type!r} "
                        f"does not match the schema signature served by {owner!r}"
                    )
                continue
            delta_schemas[node_type] = schema
            delta_node_packs[node_type] = spec.name
        delta_choices: dict[str, tuple[str, ...]] = {}
        for choice_id, values in worker.combo_choices.items():
            choice_owner = self._choice_owners.get(choice_id)
            if choice_owner is not None and choice_owner != replacing:
                if composition.choices.get(choice_id) != values:
                    raise CompositionError(
                        f"remote worker {spec.name!r}: choice list {choice_id!r} "
                        f"does not match the values served by {choice_owner!r}"
                    )
                continue
            delta_choices[choice_id] = values
        delta_lazy_ids: list[str] = []
        for choice_id in worker.lazy_choice_ids:
            choice_owner = self._choice_owners.get(choice_id)
            if choice_owner is not None and choice_owner != replacing:
                raise CompositionError(
                    f"choice list {choice_id!r} is declared by both "
                    f"{choice_owner!r} and {spec.name!r}"
                )
            delta_lazy_ids.append(choice_id)
        _validate_remote_authority(spec.name, delta_schemas, delta_choices, delta_lazy_ids)
        aliases = worker.comfy_aliases
        if aliases is not None:
            records = tuple(record for record in aliases.records if record.carrier in delta_schemas)
            source_types = {record.source.node_type for record in records}
            aliases = (
                replace(
                    aliases,
                    records=records,
                    source_schemas=tuple(
                        snapshot
                        for snapshot in aliases.source_schemas
                        if snapshot.schema.node_type in source_types
                    ),
                )
                if records
                else None
            )
        if aliases is not None:
            alias_problems = comfy_alias_registry_problems(aliases, delta_schemas)
            if alias_problems:
                raise CompositionError(
                    f"remote worker {spec.name!r}: invalid comfy alias registry: "
                    f"{alias_problems[0]}"
                )
        groups = worker.comfy_groups
        if groups is not None:
            records = tuple(record for record in groups.records if record.carrier in delta_schemas)
            source_types = {
                node.source.node_type for record in records for _, node in record.pattern.nodes
            }
            group_types = {record.pattern.group_type for record in records}
            groups = (
                replace(
                    groups,
                    records=records,
                    source_schemas=tuple(
                        snapshot
                        for snapshot in groups.source_schemas
                        if snapshot.schema.node_type in source_types
                    ),
                    group_schemas=tuple(
                        snapshot
                        for snapshot in groups.group_schemas
                        if snapshot.schema.node_type in group_types
                    ),
                )
                if records
                else None
            )
        if groups is not None:
            group_problems = comfy_group_registry_problems(groups, delta_schemas)
            if group_problems:
                raise CompositionError(
                    f"remote worker {spec.name!r}: invalid comfy group registry: "
                    f"{group_problems[0]}"
                )
        delta_compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = {}
        for source_name, diagnostic in worker.compat_skips.items():
            node_id = source_name.removeprefix(spec.name + ".")
            skip_owner = self._compat_skip_owners.get((spec.name, node_id))
            if skip_owner is not None and skip_owner != replacing:
                raise CompositionError(
                    f"compat skip {spec.name}.{node_id!r} is declared by "
                    f"both {skip_owner!r} and {spec.name!r}"
                )
            delta_compat_skips.setdefault(spec.name, {})[node_id] = diagnostic
        return _RemoteSurface(
            composed=composed,
            body_arms=tuple(
                (arm_name, kept)
                for arm_name, node_types in body_arms
                if (kept := tuple(t for t in node_types if t in composed))
            ),
            schemas=delta_schemas,
            node_packs=delta_node_packs,
            comfy_aliases=aliases,
            comfy_groups=groups,
            choices=delta_choices,
            lazy_choice_ids=tuple(delta_lazy_ids),
            compat_skips=delta_compat_skips,
            vision_providers=vision_providers,
            generation_providers=generation_providers,
        )

    def _remote_owner_blockers(
        self, name: str, schemas: Mapping[str, NodeSchema]
    ) -> dict[str, list[str]]:
        """Dependents of the schemas remote ``name`` owns - local packs
        that execute one of its types and other remotes announcing it. A
        replacement announcement must keep every depended-on schema
        signature-identical, the same rule as :meth:`_owner_blockers`."""
        blockers: dict[str, list[str]] = {}
        record = self._remotes[name]
        for node_type in record.delta.schemas:
            new_schema = schemas.get(node_type)
            for other_name, other in self._records.items():
                if node_type not in other.executes:
                    continue
                if new_schema is None or schema_signature(new_schema) != schema_signature(
                    other.worker.schemas[node_type]
                ):
                    blockers.setdefault(other_name, []).append(node_type)
            for remote_name, remote in self._remotes.items():
                if remote_name == name or node_type not in remote.node_types:
                    continue
                if new_schema is None or schema_signature(new_schema) != schema_signature(
                    remote.worker.schemas[node_type]
                ):
                    blockers.setdefault(remote_name, []).append(node_type)
        return blockers

    def remote_connected(self, name: str) -> bool | None:
        """Whether remote ``name`` has a live session; None when it is
        configured but never composed. The reconnect supervisor's probe."""
        record = self._remotes.get(name)
        return None if record is None else record.worker.alive

    async def reattach_remote(self, spec: RemoteSpec) -> RemoteReattachResult:
        """Replace one composed remote's dead session with a freshly dialed
        one - the reconnect half of :meth:`add_remote`.

        Dial-new-first: the new hello is revalidated with add_remote's
        rules minus this remote's own prior claims (a redeployed daemon may
        announce a different surface), dependents of its owned schemas
        block a signature change exactly like reload, and the whole swap
        lands at one commit point. Any failure leaves the composed record
        and the served surface exactly as they were, so a supervisor can
        retry forever against a down daemon. The old session is reaped
        just before the swap; a close never stops the daemon."""
        async with self._mutate:
            return await self._reattach_remote_locked(spec)

    async def _reattach_remote_locked(self, spec: RemoteSpec) -> RemoteReattachResult:
        composition = self.composition
        source = f"remote:{spec.host}:{spec.port}"
        record = self._remotes.get(spec.name)
        if record is None:
            raise UnknownPackError(f"no composed remote named {spec.name!r} to reattach")
        try:
            token = spec.token_file.read_text("utf-8").strip()
        except OSError as exc:
            raise CompositionError(
                f"{source}: cannot read token file {spec.token_file}: {exc}"
            ) from exc
        if spec.tls_ca_file is not None:
            try:
                spec.tls_ca_file.read_bytes()
            except OSError as exc:
                raise CompositionError(
                    f"{source}: cannot read TLS CA file {spec.tls_ca_file}: {exc}"
                ) from exc
        worker = RemoteWorker(
            spec.host,
            spec.port,
            token,
            composition._registry,
            name=spec.name,
            on_diagnostic=self._diagnostic_listener,
            governor=self._governor,
            reservations=self._reservations,
            telemetry=self._telemetry,
            asset_endpoint=self._remote_asset_endpoint,
            asset_endpoint_token=self._remote_asset_token,
            value_store=self._remote_value_store,
            tls_ca_file=spec.tls_ca_file,
            engine_instance_id=self._engine_instance_id,
            resume_from=record.worker,
        )
        snapshot: ExtensionSnapshot | None = None
        topology: Topology | None = None
        try:
            await worker.start()
            surface = self._validated_remote_surface(spec, worker, replacing=spec.name)
            info = PackInfo(
                display_name=spec.name,
                source=source,
                comfy_aliases=surface.comfy_aliases,
                comfy_groups=surface.comfy_groups,
            )
            staged_packs = {
                pack_id: pack_info
                for pack_id, pack_info in composition.packs.items()
                if pack_id != spec.name
            }
            _merge_pack_entry(staged_packs, spec.name, info, source)
            self._raise_owner_blockers(
                spec.name,
                self._remote_owner_blockers(spec.name, surface.schemas),
                "reattach",
            )
            delta = PackDelta(
                pack=spec.name,
                schemas=surface.schemas,
                packs={spec.name: info},
                node_packs=surface.node_packs,
                schema_owners=dict.fromkeys(surface.schemas, spec.name),
                choice_owners=dict.fromkeys(
                    (*surface.choices, *surface.lazy_choice_ids), spec.name
                ),
                choices=surface.choices,
                lazy_choices={
                    choice_id: _lazy_choice_fetcher(worker, choice_id, self)
                    for choice_id in surface.lazy_choice_ids
                },
                compat_skips=surface.compat_skips,
            )
            new_record = _RemoteRecord(
                spec=spec,
                worker=worker,
                delta=delta,
                domain=_ResidencyDomain(worker, spec.name),
                node_types=tuple(surface.composed),
                pack_name=worker.pack,
                body_arms=surface.body_arms,
                vision_providers=surface.vision_providers,
                generation_providers=surface.generation_providers,
                instance_token=worker.instance_token,
            )
            staged_remotes = {**self._remotes, spec.name: new_record}
            topology = self._build_topology(self._records, staged_remotes)
            (
                snapshot,
                sampler_registry,
                scheduler_registry,
                graph_compiler_registry,
                graph_compile_transport,
                inference_unavailable,
            ) = await self._build_extension_snapshot(self._records, topology)
            derived_choices = self._validated_derived_choices(
                sampler_registry, scheduler_registry, self._records, topology
            )
            staged_registry = composition._registry.copy()
            staged_registry.remove_relayed_renditions(spec.name)
            self._register_worker_renditions(staged_registry, spec.name, worker)
            # Reap the old session before the first published mutation:
            # everything from the route swap to the return is then free
            # of suspension points, so the caller publishes the surface
            # and clears the cache in the same task step as the swap - no
            # job can be admitted against the new runtime while the old
            # cache or ServerState surface is still live, and a
            # cancellation cannot strand a half-committed composer.
            old_worker = record.worker
            if old_worker.session_identity is not worker.session_identity:
                await old_worker.close()
        except BaseException:
            # Failed reattach: the OLD record keeps its place (routes and
            # surface untouched); the new connection is reaped NOW.
            if snapshot is not None and topology is not None:
                await self._rollback_unpublished_generation(snapshot, topology)
            await worker.rollback_start()
            raise
        old_derived_choices = self._current_derived_choices()
        old_choice_owners = dict(self._choice_owners)
        # Commit point: swap routes in one reference swap (concurrent
        # invocations see old or new, never half), then rewrite the
        # registries from the updated records (mirrors reload_pack's
        # commit). No await may appear between here and the return.
        candidates = tuple(dict.fromkeys((*record.node_types, *new_record.node_types)))
        removed_routes = tuple(
            node_type for node_type in candidates if self._routing.has_route(node_type)
        )
        added_routes = tuple(node_type for node_type in candidates if topology.get(node_type))
        self._routing.swap_routes(removed_routes, self._dispatch_routes(topology, added_routes))
        self._remotes[spec.name] = new_record
        composition._registry.replace_from(staged_registry)
        self._topology = topology
        composition._isolated.append(worker)
        if old_worker in composition._isolated:
            composition._isolated.remove(old_worker)
        self._rebuild_registries()
        self._apply_derived_choices(derived_choices)
        changed_derived_choices = self._changed_derived_choices(old_derived_choices)
        # Core-scaffolding types the remote merely arms stay off the
        # schema swap (they are served by core and carry no pack
        # attribution, same rule as add_remote); their arm updates ride
        # execution_arms over every re-routed type.
        removed_surface_types = tuple(
            node_type for node_type in removed_routes if node_type not in self._core_schemas
        )
        published_types = tuple(
            node_type for node_type in added_routes if node_type not in self._core_schemas
        )
        delta = replace(
            delta,
            schemas={node_type: composition.schemas[node_type] for node_type in published_types},
            node_packs={
                node_type: composition.node_packs[node_type] for node_type in published_types
            },
            schema_owners={node_type: self._owners[node_type] for node_type in published_types},
            choice_owners=self._published_choice_owners(
                (*delta.choices, *delta.lazy_choices, *changed_derived_choices),
                previous=old_choice_owners,
            ),
            derived_choices=changed_derived_choices,
            execution_arms={
                node_type: composition.execution_arms[node_type] for node_type in added_routes
            },
        )
        self._apply_inference_unavailable(inference_unavailable, delta.packs)
        self._publish_runtime(
            topology,
            snapshot,
            sampler_registry,
            graph_compiler_registry,
            graph_compile_transport,
        )
        # The pack records are unchanged, so no generation retires with
        # this swap and this call returns without suspending (the
        # commit-to-return section stays await-free).
        await self._commit_published_generation(snapshot, None)
        same_instance = (
            record.instance_token is not None and record.instance_token == new_record.instance_token
        )
        return RemoteReattachResult(
            result=ReloadResult(
                pack=spec.name,
                removed_types=removed_surface_types,
                removed_packs=(spec.name,),
                delta=delta,
                removed_choices=tuple({**record.delta.choices, **record.delta.lazy_choices}),
                removed_compat_skips=tuple(
                    (pack_id, node_id)
                    for pack_id, skips in record.delta.compat_skips.items()
                    for node_id in skips
                ),
            ),
            same_instance=same_instance,
        )

    async def reload_pack(self, name: str, spec: PackSpec | None = None) -> ReloadResult:
        """Replace one pack's validated declarations and execution routes.

        Cold managed packs remain cold. A dynamic worker with a newer live
        announcement can publish it without restarting the first invocation.

        ``spec`` replaces the recorded spec for this and future reloads -
        the live-activation path, where a new install generation moves
        the pack to a different content-addressed store directory (and
        possibly interpreter). Omitted means reload from the recorded
        spec: the dev edit-in-place flow.

        Other live replacements start-new-first, swap, then stop old: the new worker starts and
        validates while the OLD one keeps serving, and any failure -
        manifest error, worker death, schema refusal, namespace violation
        - closes the new worker and leaves the running surface exactly as
        it was. Only after every validation passes do routes swap (one
        reference swap), the registries rewrite, and the old worker close.

        Validation is add_pack's, minus this pack's own prior claims and
        node types (a pack cannot collide with itself). Renaming the pack
        in its manifest is refused - reload swaps a known identity; a
        rename is remove-and-add, which reload cannot express.

        In-flight jobs pinned their schema mapping at run entry, so they
        finish against the OLD definitions - but mid-run invocations of
        this pack's types route to the NEW worker (or fail loudly for
        dropped types). Dev-mode semantics: reload while jobs run on the
        reloaded pack may fail those jobs, never corrupt them.
        """
        async with self._mutate:
            record = self._records.get(name)
            if record is None:
                raise UnknownPackError(f"no composed pack named {name!r} to reload")
            if record.spec.in_process:
                raise CompositionError(
                    f"cannot reload in-process pack {name!r}; replace the generation "
                    "and restart dinkster-serve"
                )
            composition = self.composition
            if spec is None:
                spec = record.spec
            if spec.runtime_settings and self._runtime_worker_settings is not None:
                aimdo, reserve_vram, memory_budgets, comfy_args = self._runtime_worker_settings()
                spec = replace(
                    spec,
                    aimdo=aimdo,
                    reserve_vram=reserve_vram,
                    vram_budgets=cuda_vram_budgets(memory_budgets),
                    comfy_args=comfy_args,
                )
            if record.spec.worker_group is not None:
                if spec.worker_group != record.spec.worker_group:
                    raise CompositionError(
                        f"pack {name!r} cannot change worker-group placement during "
                        "reload; replace the hosting generation"
                    )
                return await self._reload_group_locked(name, record, spec)
            manifest = load_manifest(resolve_manifest_path(spec.manifest))
            if canonical_name(manifest.name) != record.canonical:
                raise CompositionError(
                    f"{manifest.path}: reload would rename pack {name!r} "
                    f"to {manifest.name!r}; a rename is remove-and-add, "
                    "not a reload"
                )
            _order, staged_generation = self._resolve_contract_inputs(
                self._contract_inputs(replacing=(manifest, spec), removing=name)
            )
            for claim in manifest.namespaces:
                root = reserved_root(claim)
                if root is not None and not spec.trust_reserved:
                    raise CompositionError(
                        f"{manifest.path}: namespace claim {claim!r} falls "
                        f"under the reserved root {root!r}; composing it "
                        f"requires explicit host trust "
                        f"(PackSpec(trust_reserved=True))"
                    )
                for held, holder in self._claim_owners.items():
                    if holder == name:
                        continue  # a pack cannot collide with itself
                    if not claims_conflict(held, claim):
                        continue
                    if _trusted_reserved_claims_can_overlap(held, claim):
                        continue
                    raise CompositionError(
                        f"namespace {claim!r} claimed by {manifest.name!r} "
                        f"overlaps {held!r} claimed by {holder!r}"
                    )
            spec_packs = (
                dict(spec.packs)
                if spec.packs is not None
                else {
                    manifest.name: replace(
                        pack_info_from_manifest(manifest),
                        source=f"local:{manifest.root.resolve()}",
                    )
                }
            )
            # Stage the pack table as it would look after the swap: current
            # entries minus this pack's exclusive ids, then the new
            # declarations merged under the identical-or-new rule (a shared
            # entry like "comfy" must be re-declared identically).
            shared_ids = {
                pack_id
                for other, other_record in self._records.items()
                if other != name
                for pack_id in other_record.delta.packs
            }
            staged_packs = {
                pack_id: info
                for pack_id, info in composition.packs.items()
                if pack_id in shared_ids or pack_id not in record.delta.packs
            }
            for pack_id, info in spec_packs.items():
                _merge_pack_entry(staged_packs, pack_id, info, str(manifest.path))
            promoted = (
                isinstance(record.worker, LazyWorker)
                and record.worker.declarations_changed
                and spec == record.spec
                and source_digest(manifest) == record.worker.catalog.source
            )
            if promoted:
                worker = await record.worker.ensure_started()
                if not worker.alive:
                    raise CompositionError(f"pack {name!r} died before its schemas were refreshed")
            else:
                launch_spec = (
                    spec
                    if getattr(record.worker, "cold", False)
                    else replace(spec, require_catalog=False)
                )
                worker = self._isolated_worker(launch_spec, manifest, composition._registry)
            try:
                if not promoted:
                    await worker.start()
                self._validate_body_arms(manifest, worker)
                self._validate_schema_only(manifest, worker)
                self._validate_executes(manifest, worker, replacing=name)
                delta_schemas: dict[str, NodeSchema] = {}
                delta_node_packs: dict[str, str] = {}
                for node_type, schema in worker.schemas.items():
                    if node_type in manifest.executes:
                        continue
                    if not any(claim_covers(claim, node_type) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{manifest.name}: node type {node_type!r} is "
                            f"outside the pack's declared namespaces "
                            f"({', '.join(manifest.namespaces)})"
                        )
                    owner = self._owners.get(node_type)
                    if owner is not None and owner != name:
                        raise CompositionError(
                            f"node type {node_type!r} is declared by both "
                            f"{owner!r} and {manifest.name!r}"
                        )
                    pack_id = spec.attribute(node_type) if spec.attribute else manifest.name
                    if pack_id not in spec_packs:
                        raise CompositionError(
                            f"{manifest.name}: node type {node_type!r} "
                            f"attributes to {pack_id!r}, which is not among "
                            f"this spec's pack entries"
                        )
                    delta_schemas[node_type] = schema
                    delta_node_packs[node_type] = pack_id
                _validate_registry_carriers(
                    manifest.name,
                    spec_packs,
                    {
                        node_type: pack_id
                        for node_type, pack_id in composition.node_packs.items()
                        if node_type not in record.delta.node_packs
                    }
                    | delta_node_packs,
                )
                delta_choices: dict[str, tuple[str, ...]] = {}
                for choice_id, values in worker.combo_choices.items():
                    if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{manifest.name}: choice list {choice_id!r} is "
                            f"outside the pack's declared namespaces "
                            f"({', '.join(manifest.namespaces)})"
                        )
                    choice_owner = self._choice_owners.get(choice_id)
                    if choice_owner is not None and choice_owner != name:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both "
                            f"{choice_owner!r} and {manifest.name!r}"
                        )
                    delta_choices[choice_id] = values
                delta_lazy: dict[str, LazyChoiceFetcher] = {}
                for choice_id in worker.lazy_choice_ids:
                    if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{manifest.name}: choice list {choice_id!r} is "
                            f"outside the pack's declared namespaces "
                            f"({', '.join(manifest.namespaces)})"
                        )
                    choice_owner = self._choice_owners.get(choice_id)
                    if choice_owner is not None and choice_owner != name:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both "
                            f"{choice_owner!r} and {manifest.name!r}"
                        )
                    delta_lazy[choice_id] = _lazy_choice_fetcher(worker, choice_id, self)
                _validate_remote_authority(manifest.name, delta_schemas, delta_choices, delta_lazy)
                delta_compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = {}
                for source_name, diagnostic in worker.compat_skips.items():
                    pack_id = spec.attribute(source_name) if spec.attribute else manifest.name
                    if pack_id not in spec_packs:
                        raise CompositionError(
                            f"{manifest.name}: compat skip {source_name!r} "
                            f"attributes to {pack_id!r}, which is not among "
                            "this spec's pack entries"
                        )
                    node_id = source_name.removeprefix(pack_id + ".")
                    owner = self._compat_skip_owners.get((pack_id, node_id))
                    if owner is not None and owner != name:
                        raise CompositionError(
                            f"compat skip {pack_id}.{node_id!r} is declared by "
                            f"both {owner!r} and {manifest.name!r}"
                        )
                    delta_compat_skips.setdefault(pack_id, {})[node_id] = diagnostic
                self._raise_owner_blockers(
                    name,
                    self._owner_blockers(name, delta_schemas),
                    "reload",
                )
            except BaseException:
                # Failed reload: the OLD worker keeps serving; the new one
                # is reaped NOW (a dev session may retry reload many times -
                # leaking one live process per attempt is not acceptable).
                if not promoted:
                    await worker.close()
                raise
            # Commit point: swap routes in one reference swap (concurrent
            # invocations see old or new, never half), rewrite the
            # registries from the updated records, then stop the old worker.
            old_types = tuple(record.delta.schemas)
            delta = PackDelta(
                pack=manifest.name,
                schemas=delta_schemas,
                packs=spec_packs,
                node_packs=delta_node_packs,
                schema_owners=dict.fromkeys(delta_schemas, manifest.name),
                choice_owners=dict.fromkeys((*delta_choices, *delta_lazy), manifest.name),
                choices=delta_choices,
                lazy_choices=delta_lazy,
                compat_skips=delta_compat_skips,
            )
            new_record = _PackRecord(
                spec=spec,
                worker=worker,
                delta=delta,
                claims=tuple(manifest.namespaces),
                canonical=record.canonical,
                manifest=manifest,
                executes=manifest.executes,
                schema_only=manifest.schema_only,
                default_cache_tag=self._execution_identity(manifest, spec_packs, spec),
                body_arms=manifest.arms,
                domain=record.domain if promoted else _ResidencyDomain(worker, manifest.name),
                asset_roots=_pack_asset_roots(spec, manifest, spec_packs),
                extension=manifest.extension if manifest.extension_declared else None,
                extension_contributions=worker.extension_contributions,
            )
            staged_records = dict(self._records)
            del staged_records[name]
            staged_records[manifest.name] = new_record
            snapshot: ExtensionSnapshot | None = None
            topology: Topology | None = None
            try:
                topology = self._build_topology(staged_records)
                (
                    snapshot,
                    sampler_registry,
                    scheduler_registry,
                    graph_compiler_registry,
                    graph_compile_transport,
                    inference_unavailable,
                ) = await self._build_extension_snapshot(staged_records, topology)
                derived_choices = self._validated_derived_choices(
                    sampler_registry, scheduler_registry, staged_records, topology
                )
            except BaseException:
                if snapshot is not None and topology is not None:
                    await self._rollback_unpublished_generation(snapshot, topology)
                if not promoted:
                    await worker.close()
                raise
            old_derived_choices = self._current_derived_choices()
            old_choice_owners = dict(self._choice_owners)
            staged_registry = composition._registry.copy()
            staged_registry.remove_relayed_renditions(name)
            if not spec.in_process:
                self._register_worker_renditions(staged_registry, manifest.name, worker)
            removed_routes = tuple(
                node_type
                for node_type in dict.fromkeys((*old_types, *record.executes, *manifest.executes))
                if self._routing.has_route(node_type)
            )
            added_routes = tuple(
                node_type
                for node_type in dict.fromkeys(
                    (*old_types, *record.executes, *manifest.executes, *delta.schemas)
                )
                if topology.get(node_type)
            )
            self._routing.swap_routes(removed_routes, self._dispatch_routes(topology, added_routes))
            old_inference_worker = self._sampling_worker(self._topology)
            new_inference_worker = self._sampling_worker(topology)
            retired_inference_worker = (
                old_inference_worker
                if old_inference_worker is not None
                and old_inference_worker is not new_inference_worker
                else None
            )
            del self._records[name]
            self._records[manifest.name] = new_record
            composition._registry.replace_from(staged_registry)
            composition.generation = staged_generation
            self._topology = topology
            composition._isolated.append(worker)
            self._rebuild_registries()
            self._apply_derived_choices(derived_choices)
            changed_derived_choices = self._changed_derived_choices(old_derived_choices)
            delta = replace(
                delta,
                schemas={node_type: composition.schemas[node_type] for node_type in added_routes},
                node_packs={
                    node_type: composition.node_packs[node_type] for node_type in added_routes
                },
                schema_owners={node_type: self._owners[node_type] for node_type in added_routes},
                choice_owners=self._published_choice_owners(
                    (*delta.choices, *delta.lazy_choices, *changed_derived_choices),
                    previous=old_choice_owners,
                ),
                derived_choices=changed_derived_choices,
                execution_arms={
                    node_type: composition.execution_arms[node_type] for node_type in added_routes
                },
            )
            self._apply_inference_unavailable(inference_unavailable, delta.packs)
            self._publish_runtime(
                topology,
                snapshot,
                sampler_registry,
                graph_compiler_registry,
                graph_compile_transport,
            )
            await self._commit_published_generation(snapshot, retired_inference_worker)
            old_worker = record.worker
            if old_worker in composition._isolated:
                composition._isolated.remove(old_worker)
            if promoted:
                old_worker.detach()
            else:
                await old_worker.close()
            return ReloadResult(
                pack=manifest.name,
                removed_types=removed_routes,
                removed_packs=tuple(
                    pack_id for pack_id in record.delta.packs if pack_id not in shared_ids
                ),
                delta=delta,
                removed_choices=tuple({**record.delta.choices, **record.delta.lazy_choices}),
                removed_compat_skips=tuple(
                    (pack_id, node_id)
                    for pack_id, skips in record.delta.compat_skips.items()
                    for node_id in skips
                ),
            )

    async def _reload_group_locked(
        self, name: str, record: _PackRecord, replacement: PackSpec
    ) -> ReloadResult:
        """Replace every session in one process group at one commit point."""
        group_name = replacement.worker_group
        assert group_name is not None
        old_owner = self._group_owners.get(group_name)
        if old_owner is None:
            raise CompositionError(f"worker group {group_name!r} is not running")
        group_records = {
            pack: current
            for pack, current in self._records.items()
            if current.spec.worker_group == group_name
        }
        specs = {
            pack: (replacement if pack == name else current.spec)
            for pack, current in group_records.items()
        }
        contract_values = {
            _worker_group_contract(item, self._worker_env) for item in specs.values()
        }
        if len(contract_values) != 1:
            raise CompositionError(
                f"worker group {group_name!r} members have incompatible launch contracts"
            )
        manifest_paths = replacement.group_manifests
        manifests = tuple(load_manifest(path) for path in manifest_paths)
        manifest_by_name = {manifest.name: manifest for manifest in manifests}
        if set(manifest_by_name) != set(group_records):
            raise CompositionError(
                f"worker group {group_name!r} manifest members changed during reload"
            )
        contract_inputs = {
            canonical_name(current.manifest.name): (current.manifest, current.spec)
            for pack, current in self._records.items()
            if pack not in group_records
        }
        for pack, manifest in manifest_by_name.items():
            contract_inputs[canonical_name(manifest.name)] = (manifest, specs[pack])
        _order, staged_generation = self._resolve_contract_inputs(contract_inputs)
        catalogs: dict[str, PackCatalog] = {}
        if all(
            isinstance(current.worker, LazyWorker)
            and current.worker.cold
            and specs[pack].require_catalog
            for pack, current in group_records.items()
        ):
            for manifest in manifests:
                catalog = read_catalog(manifest)
                if catalog is None:
                    raise CompositionError(
                        f"{manifest.name}: schema catalog is missing or stale; rerun doctor"
                    )
                catalogs[manifest.name] = catalog
        group_env = self._worker_environment(replacement, manifests)
        owner = GroupIsolatedWorker(
            group_name,
            manifest_paths,
            self.composition._registry,
            python=replacement.python,
            extra_env=group_env,
            launcher=self._sandbox_launcher(replacement, manifests, group_env),
            aimdo_arm=replacement.aimdo,
            vram_budgets=replacement.vram_budgets,
            reserve_vram=replacement.reserve_vram,
            comfy_args=replacement.comfy_args,
            start_timeout=replacement.start_timeout,
            on_diagnostic=self._diagnostic_listener,
            governor=self._governor,
            reservations=self._reservations,
            telemetry=self._telemetry,
            headroom_mirror=self._headroom_mirror,
            on_schema_reload=self._schema_reload_requested,
        )
        snapshot: ExtensionSnapshot | None = None
        topology: Topology | None = None
        try:
            members: dict[str, Any] = dict(owner.members)
            if catalogs:
                members = {
                    manifest.name: self._catalog_group_worker(
                        group_name, owner, manifest, catalogs[manifest.name]
                    )
                    for manifest in manifests
                }
            else:
                await owner.start()
            staged_records = dict(self._records)
            group_domain = _ResidencyDomain(next(iter(owner.members.values())), group_name)
            group_names = set(group_records)
            seen_nodes: dict[str, str] = {}
            seen_choices: dict[str, str] = {}
            seen_compat_skips: dict[tuple[str, str], str] = {}
            seen_claims: list[tuple[str, str]] = []
            staged_pack_table = {
                pack_id: info
                for pack_id, info in self.composition.packs.items()
                if any(
                    pack_id in current.delta.packs
                    for other, current in self._records.items()
                    if other not in group_names
                )
                or not any(pack_id in current.delta.packs for current in group_records.values())
            }
            for pack, old_record in group_records.items():
                member = members[pack]
                manifest = manifest_by_name[pack]
                for claim in manifest.namespaces:
                    root = reserved_root(claim)
                    member_spec = specs[pack]
                    if root is not None and not member_spec.trust_reserved:
                        raise CompositionError(
                            f"{manifest.path}: namespace claim {claim!r} requires "
                            "explicit host trust"
                        )
                    for held, holder in self._claim_owners.items():
                        if holder in group_names or not claims_conflict(held, claim):
                            continue
                        if _trusted_reserved_claims_can_overlap(held, claim):
                            continue
                        raise CompositionError(
                            f"namespace {claim!r} claimed by {pack!r} overlaps "
                            f"{held!r} claimed by {holder!r}"
                        )
                    for held, holder in seen_claims:
                        if claims_conflict(held, claim):
                            if _trusted_reserved_claims_can_overlap(held, claim):
                                continue
                            raise CompositionError(
                                f"namespace {claim!r} claimed by {pack!r} overlaps "
                                f"{held!r} claimed by {holder!r}"
                            )
                    seen_claims.append((claim, pack))
                self._validate_body_arms(manifest, member)
                self._validate_schema_only(manifest, member)
                announced = {
                    node_type: schema
                    for node_type, schema in member.schemas.items()
                    if node_type not in manifest.executes
                }
                for node_type in announced:
                    if not any(claim_covers(claim, node_type) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{pack}: node type {node_type!r} is outside the pack's "
                            "declared namespaces"
                        )
                    external_owner = self._owners.get(node_type)
                    if external_owner is not None and external_owner not in group_names:
                        raise CompositionError(
                            f"node type {node_type!r} is declared by both "
                            f"{external_owner!r} and {pack!r}"
                        )
                    prior = seen_nodes.setdefault(node_type, pack)
                    if prior != pack:
                        raise CompositionError(
                            f"node type {node_type!r} is declared by both {prior!r} and {pack!r}"
                        )
                member_spec = specs[pack]
                spec_packs = (
                    dict(member_spec.packs)
                    if member_spec.packs is not None
                    else {
                        pack: replace(
                            pack_info_from_manifest(manifest),
                            source=f"local:{manifest.root.resolve()}",
                        )
                    }
                )
                for pack_id, info in spec_packs.items():
                    _merge_pack_entry(staged_pack_table, pack_id, info, str(manifest.path))
                node_packs = {
                    node_type: (
                        member_spec.attribute(node_type)
                        if member_spec.attribute is not None
                        else pack
                    )
                    for node_type in announced
                }
                if any(pack_id not in spec_packs for pack_id in node_packs.values()):
                    raise CompositionError(
                        f"worker group {group_name!r} reload attributed outside its pack table"
                    )
                choices = dict(member.combo_choices)
                for choice_id in choices:
                    if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{pack}: choice list {choice_id!r} is outside the pack's "
                            "declared namespaces"
                        )
                    external_owner = self._choice_owners.get(choice_id)
                    if external_owner is not None and external_owner not in group_names:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both "
                            f"{external_owner!r} and {pack!r}"
                        )
                    prior = seen_choices.setdefault(choice_id, pack)
                    if prior != pack:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both {prior!r} and {pack!r}"
                        )
                member_lazy: dict[str, LazyChoiceFetcher] = {}
                for choice_id in member.lazy_choice_ids:
                    if not any(claim_covers(claim, choice_id) for claim in manifest.namespaces):
                        raise CompositionError(
                            f"{pack}: choice list {choice_id!r} is outside the pack's "
                            "declared namespaces"
                        )
                    external_owner = self._choice_owners.get(choice_id)
                    if external_owner is not None and external_owner not in group_names:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both "
                            f"{external_owner!r} and {pack!r}"
                        )
                    prior = seen_choices.setdefault(choice_id, pack)
                    if prior != pack:
                        raise CompositionError(
                            f"choice list {choice_id!r} is declared by both {prior!r} and {pack!r}"
                        )
                    member_lazy[choice_id] = _lazy_choice_fetcher(member, choice_id, self)
                _validate_remote_authority(pack, announced, choices, member_lazy)
                compat_skips: dict[str, dict[str, CompatGateDiagnostic]] = {}
                for source_name, diagnostic in member.compat_skips.items():
                    pack_id = (
                        member_spec.attribute(source_name)
                        if member_spec.attribute is not None
                        else pack
                    )
                    if pack_id not in spec_packs:
                        raise CompositionError(
                            f"{pack}: compat skip {source_name!r} attributes outside "
                            "this spec's pack entries"
                        )
                    node_id = source_name.removeprefix(pack_id + ".")
                    external_owner = self._compat_skip_owners.get((pack_id, node_id))
                    if external_owner is not None and external_owner not in group_names:
                        raise CompositionError(
                            f"compat skip {pack_id}.{node_id!r} is declared by both "
                            f"{external_owner!r} and {pack!r}"
                        )
                    prior = seen_compat_skips.setdefault((pack_id, node_id), pack)
                    if prior != pack:
                        raise CompositionError(
                            f"compat skip {pack_id}.{node_id!r} is declared by both "
                            f"{prior!r} and {pack!r}"
                        )
                    compat_skips.setdefault(pack_id, {})[node_id] = diagnostic
                delta = replace(
                    old_record.delta,
                    schemas=announced,
                    packs=spec_packs,
                    node_packs=node_packs,
                    schema_owners=dict.fromkeys(announced, pack),
                    choice_owners=dict.fromkeys((*choices, *member_lazy), pack),
                    choices=choices,
                    lazy_choices=member_lazy,
                    compat_skips=compat_skips,
                )
                staged_records[pack] = replace(
                    old_record,
                    spec=member_spec,
                    worker=member,
                    delta=delta,
                    claims=tuple(manifest.namespaces),
                    manifest=manifest,
                    executes=manifest.executes,
                    schema_only=manifest.schema_only,
                    default_cache_tag=self._execution_identity(manifest, spec_packs, member_spec),
                    body_arms=manifest.arms,
                    domain=group_domain,
                    asset_roots=_pack_asset_roots(member_spec, manifest, spec_packs),
                    extension=manifest.extension if manifest.extension_declared else None,
                    extension_contributions=member.extension_contributions,
                )
            replaced_types = {
                node_type
                for current in group_records.values()
                for node_type in current.delta.node_packs
            }
            _validate_registry_carriers(
                group_name,
                staged_pack_table,
                {
                    node_type: pack_id
                    for node_type, pack_id in self.composition.node_packs.items()
                    if node_type not in replaced_types
                }
                | {
                    node_type: pack_id
                    for pack in group_records
                    for node_type, pack_id in staged_records[pack].delta.node_packs.items()
                },
            )
            staged_owners = {
                node_type: pack
                for pack, current in staged_records.items()
                for node_type in current.delta.schemas
            }
            for pack, current in staged_records.items():
                for node_type in current.executes:
                    arm_schema = current.worker.schemas.get(node_type)
                    owner_name = staged_owners.get(node_type)
                    if arm_schema is None:
                        raise CompositionError(
                            f"{pack}: executes claim {node_type!r} is not present "
                            "in the worker hello schemas"
                        )
                    if owner_name is None or owner_name == pack:
                        raise CompositionError(
                            f"{pack}: executes claim {node_type!r} has no owning "
                            "isolated pack in the staged group topology"
                        )
                    owner_schema = staged_records[owner_name].delta.schemas[node_type]
                    if schema_signature(arm_schema) != schema_signature(owner_schema):
                        raise CompositionError(
                            f"{pack}: executes claim {node_type!r} does not match "
                            f"owning pack {owner_name!r}'s staged schema signature"
                        )
            topology = self._build_topology(staged_records)
            (
                snapshot,
                sampler_registry,
                scheduler_registry,
                graph_compiler_registry,
                graph_compile_transport,
                inference_unavailable,
            ) = await self._build_extension_snapshot(staged_records, topology)
            derived_choices = self._validated_derived_choices(
                sampler_registry, scheduler_registry, staged_records, topology
            )
            route_candidates = tuple(
                dict.fromkeys(
                    tuple(
                        node_type
                        for current in group_records.values()
                        for node_type in (*current.delta.schemas, *current.executes)
                    )
                    + tuple(
                        node_type
                        for current in group_records
                        for node_type in (
                            *staged_records[current].delta.schemas,
                            *staged_records[current].executes,
                        )
                    )
                )
            )
            prepared_routes = self._dispatch_routes(
                topology,
                tuple(node_type for node_type in route_candidates if topology.get(node_type)),
            )
            staged_registry = self.composition._registry.copy()
            for pack, member_spec in specs.items():
                staged_registry.remove_relayed_renditions(pack)
                self._register_worker_renditions(staged_registry, pack, members[pack])
                if member_spec.host_types is not None:
                    member_spec.host_types(staged_registry)
        except BaseException:
            if snapshot is not None and topology is not None:
                await self._rollback_unpublished_generation(snapshot, topology)
            await owner.close()
            raise

        old_derived_choices = self._current_derived_choices()
        old_choice_owners = dict(self._choice_owners)
        removed_routes = tuple(
            node_type
            for node_type in dict.fromkeys(
                node_type
                for current in group_records.values()
                for node_type in (*current.delta.schemas, *current.executes)
            )
            if self._routing.has_route(node_type)
        )
        self._routing.swap_routes(removed_routes, prepared_routes)
        old_inference_worker = self._sampling_worker(self._topology)
        new_inference_worker = self._sampling_worker(topology)
        self._records = staged_records
        self.composition._registry.replace_from(staged_registry)
        self.composition.generation = staged_generation
        self._topology = topology
        self._group_owners[group_name] = owner
        self._group_contracts[group_name] = next(iter(contract_values))
        self.composition._isolated.append(owner)
        self.composition._isolated.extend(
            member for member in members.values() if isinstance(member, LazyWorker)
        )
        if old_owner in self.composition._isolated:
            self.composition._isolated.remove(old_owner)
        self._rebuild_registries()
        self._apply_derived_choices(derived_choices)
        changed_derived_choices = self._changed_derived_choices(old_derived_choices)
        aggregate_packs: dict[str, PackInfo] = {}
        aggregate_choices: dict[str, tuple[str, ...]] = {}
        aggregate_lazy: dict[str, LazyChoiceFetcher] = {}
        aggregate_skips: dict[str, dict[str, CompatGateDiagnostic]] = {}
        for pack in group_records:
            delta = staged_records[pack].delta
            aggregate_packs.update(delta.packs)
            aggregate_choices.update(delta.choices)
            aggregate_lazy.update(delta.lazy_choices)
            for pack_id, skips in delta.compat_skips.items():
                aggregate_skips.setdefault(pack_id, {}).update(skips)
        target_delta = PackDelta(
            pack=name,
            schemas={
                node_type: self.composition.schemas[node_type] for node_type in prepared_routes
            },
            packs=aggregate_packs,
            node_packs={
                node_type: self.composition.node_packs[node_type] for node_type in prepared_routes
            },
            schema_owners={node_type: self._owners[node_type] for node_type in prepared_routes},
            choice_owners=self._published_choice_owners(
                (*aggregate_choices, *aggregate_lazy, *changed_derived_choices),
                previous=old_choice_owners,
            ),
            execution_arms={
                node_type: self.composition.execution_arms[node_type]
                for node_type in prepared_routes
            },
            choices=aggregate_choices,
            lazy_choices=aggregate_lazy,
            derived_choices=changed_derived_choices,
            compat_skips=aggregate_skips,
        )
        retired_inference_worker = (
            old_inference_worker
            if old_inference_worker is not None and old_inference_worker is not new_inference_worker
            else None
        )
        self._apply_inference_unavailable(inference_unavailable, delta.packs)
        self._publish_runtime(
            topology,
            snapshot,
            sampler_registry,
            graph_compiler_registry,
            graph_compile_transport,
        )
        await self._commit_published_generation(snapshot, retired_inference_worker)
        for current in group_records.values():
            if isinstance(current.worker, LazyWorker):
                if current.worker in self.composition._isolated:
                    self.composition._isolated.remove(current.worker)
                await current.worker.close()
        await old_owner.close()
        self._group_activations.pop(old_owner, None)
        return ReloadResult(
            pack=name,
            removed_types=removed_routes,
            removed_packs=tuple(
                dict.fromkeys(
                    pack_id for current in group_records.values() for pack_id in current.delta.packs
                )
            ),
            delta=target_delta,
            removed_choices=tuple(
                dict.fromkeys(
                    choice_id
                    for current in group_records.values()
                    for choice_id in (*current.delta.choices, *current.delta.lazy_choices)
                )
            ),
            removed_compat_skips=tuple(
                (pack_id, node_id)
                for current in group_records.values()
                for pack_id, skips in current.delta.compat_skips.items()
                for node_id in skips
            ),
            reloaded_packs=tuple(group_records),
        )

    async def remove_pack(self, name: str) -> RemoveResult:
        """Retract one composed pack from the surface and stop its worker
        (the removal half of hot reload, DESIGN 3.9).

        The reload sequence minus the new worker: routes drop in one
        reference swap (invocations of a removed type fail loudly), the
        record goes, the registries rebuild from the survivors, and the
        old worker closes. Node types are exclusively owned so every one
        of this pack's types retracts; pack-table ids retract only when
        no other worker still declares them (the compat workers' shared
        "comfy" entry stays). Deliberately manifest-free: removal must
        work when the sources are already gone from disk.
        """
        async with self._mutate:
            record = self._records.get(name)
            if record is None:
                raise UnknownPackError(f"no composed pack named {name!r} to remove")
            if record.spec.worker_group is not None and any(
                other != name and current.spec.worker_group == record.spec.worker_group
                for other, current in self._records.items()
            ):
                raise CompositionError(
                    f"cannot remove pack {name!r} alone from worker group "
                    f"{record.spec.worker_group!r}; replace the hosting generation"
                )
            composition = self.composition
            self._raise_owner_blockers(name, self._owner_blockers(name, None), "remove")
            _order, staged_generation = self._resolve_contract_inputs(
                self._contract_inputs(removing=name)
            )
            shared_ids = {
                pack_id
                for other, other_record in self._records.items()
                if other != name
                for pack_id in other_record.delta.packs
            }
            old_types = tuple(record.delta.schemas)
            staged_records = dict(self._records)
            del staged_records[name]
            topology = self._build_topology(staged_records)
            snapshot: ExtensionSnapshot | None = None
            try:
                (
                    snapshot,
                    sampler_registry,
                    scheduler_registry,
                    graph_compiler_registry,
                    graph_compile_transport,
                    inference_unavailable,
                ) = await self._build_extension_snapshot(staged_records, topology)
                derived_choices = self._validated_derived_choices(
                    sampler_registry, scheduler_registry, staged_records, topology
                )
            except BaseException:
                if snapshot is not None:
                    await self._rollback_unpublished_generation(snapshot, topology)
                raise
            assert snapshot is not None
            old_derived_choices = self._current_derived_choices()
            old_choice_owners = dict(self._choice_owners)
            staged_registry = composition._registry.copy()
            staged_registry.remove_relayed_renditions(name)
            route_candidates = tuple(dict.fromkeys((*old_types, *record.executes)))
            removed_routes = tuple(
                node_type for node_type in route_candidates if self._routing.has_route(node_type)
            )
            added_routes = tuple(
                node_type for node_type in route_candidates if topology.get(node_type)
            )
            removed_surface_types = tuple(
                node_type
                for node_type in route_candidates
                if self._topology.get(node_type)
                and (node_type in old_types or not topology.get(node_type))
            )
            self._routing.swap_routes(removed_routes, self._dispatch_routes(topology, added_routes))
            old_inference_worker = self._sampling_worker(self._topology)
            new_inference_worker = self._sampling_worker(topology)
            retired_inference_worker = (
                old_inference_worker
                if old_inference_worker is not None
                and old_inference_worker is not new_inference_worker
                else None
            )
            del self._records[name]
            composition._registry.replace_from(staged_registry)
            composition.generation = staged_generation
            self._topology = topology
            self._rebuild_registries()
            self._apply_derived_choices(derived_choices)
            changed_derived_choices = self._changed_derived_choices(old_derived_choices)
            changed_packs: dict[str, PackInfo] = {}
            self._apply_inference_unavailable(inference_unavailable, changed_packs)
            self._publish_runtime(
                topology,
                snapshot,
                sampler_registry,
                graph_compiler_registry,
                graph_compile_transport,
            )
            await self._commit_published_generation(snapshot, retired_inference_worker)
            old_worker = record.worker
            group_name = record.spec.worker_group
            if record.tenant_registrations is not None and record.owns_tenant_registrations:
                await record.tenant_registrations.close()
                if record.tenant_registrations in composition._tenant_registries:
                    composition._tenant_registries.remove(record.tenant_registrations)
            if record.component_publisher is not None and record.owns_component_publisher:
                record.component_publisher.close()
                if record.component_publisher in composition._component_publishers:
                    composition._component_publishers.remove(record.component_publisher)
            if record.spec.in_process:
                pass
            elif group_name is None:
                if old_worker in composition._isolated:
                    composition._isolated.remove(old_worker)
                await cast("Any", old_worker).close()
            elif not any(
                current.spec.worker_group == group_name for current in self._records.values()
            ):
                owner = self._group_owners.pop(group_name)
                self._group_contracts.pop(group_name, None)
                if owner in composition._isolated:
                    composition._isolated.remove(owner)
                await owner.close()
                self._group_activations.pop(owner, None)
            return RemoveResult(
                pack=name,
                removed_types=removed_surface_types,
                removed_packs=tuple(
                    pack_id for pack_id in record.delta.packs if pack_id not in shared_ids
                ),
                execution_arms={
                    node_type: composition.execution_arms[node_type] for node_type in added_routes
                },
                removed_choices=tuple({**record.delta.choices, **record.delta.lazy_choices}),
                derived_choices=changed_derived_choices,
                choice_owners=self._published_choice_owners(
                    changed_derived_choices,
                    previous=old_choice_owners,
                ),
                removed_compat_skips=tuple(
                    (pack_id, node_id)
                    for pack_id, skips in record.delta.compat_skips.items()
                    for node_id in skips
                ),
                packs=changed_packs,
            )

    def _current_sampler_choices(self) -> dict[str, tuple[str, ...]]:
        return {
            choice_id: self.composition.choices[choice_id]
            for choice_id in (
                "dinkster.samplers",
                "comfy.samplers",
                "dinkster.schedulers",
                "comfy.schedulers",
            )
            if choice_id in self.composition.choices
        }

    def _current_derived_choices(self) -> dict[str, tuple[str, ...]]:
        out = self._current_sampler_choices()
        for provider_choices in (
            self._validated_vision_providers(self._records, self._topology)[0],
            self._validated_generation_providers(self._records, self._topology)[0],
        ):
            for choice_id in provider_choices:
                choices = self.composition.choices.get(choice_id)
                if choices is not None:
                    out[choice_id] = choices
        return out

    def _published_choice_owners(
        self,
        choice_ids: Collection[str],
        *,
        previous: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Resolve authority for every choice value sent to the live server."""
        owners: dict[str, str] = {}
        for choice_id in choice_ids:
            owner = self._choice_owners.get(choice_id)
            if owner is None and previous is not None:
                owner = previous.get(choice_id)
            if owner is None:
                raise RuntimeError(f"published choice {choice_id!r} has no authority owner")
            owners[choice_id] = owner
        return owners

    def _changed_derived_choices(
        self, before: Mapping[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        after = self._current_derived_choices()
        return {
            choice_id: after.get(choice_id, ())
            for choice_id in before.keys() | after.keys()
            if before.get(choice_id) != after.get(choice_id)
        }

    def _validated_sampler_choices(
        self, sampler_registry: SamplerRegistrySnapshot
    ) -> dict[str, tuple[str, ...]]:
        """Build and bound derived choices before a composition commit."""
        canonical = tuple(sampler.id for sampler in sampler_registry.samplers)
        compat = sampler_choice_values(sampler_registry)
        try:
            combo_choices_json_bytes(canonical, subject="derived choice 'dinkster.samplers'")
            combo_choices_json_bytes(compat, subject="derived choice 'comfy.samplers'")
        except ValueError as exc:
            raise CompositionError(str(exc)) from exc
        return {"dinkster.samplers": canonical, "comfy.samplers": compat}

    def _validated_providers(
        self,
        records: Mapping[str, _PackRecord],
        topology: Topology,
        *,
        provider_kind: str,
        required_input: bool,
    ) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], str]]:
        vision_nodes = {
            provider.node
            for record in records.values()
            for provider in record.manifest.vision_providers
        }
        generation_nodes = {
            provider.node
            for record in records.values()
            for provider in record.manifest.generation_providers
        }
        if overlap := sorted(vision_nodes & generation_nodes):
            raise CompositionError(
                "nodes cannot use both vision and generation provider metadata: "
                + ", ".join(overlap)
            )
        schemas = dict(self._core_schemas)
        schema_owners = dict.fromkeys(self._core_schemas, CORE_PACK_ID)
        choice_owners = dict.fromkeys(self._core_choices, CORE_PACK_ID)
        static_choices = dict(self._core_choices)
        for pack_id, record in records.items():
            schemas.update(record.delta.schemas)
            schema_owners.update(dict.fromkeys(record.delta.schemas, pack_id))
            choice_owners.update(dict.fromkeys(record.delta.choices, pack_id))
            static_choices.update(record.delta.choices)

        provider_choices: dict[str, set[str]] = {}
        routes: dict[tuple[str, str], str] = {}
        vision_providers_by_choice: set[tuple[str, str]] = set()
        for record in records.values():
            provider_id = record.manifest.name
            providers = (
                record.manifest.vision_providers
                if provider_kind == "vision"
                else record.manifest.generation_providers
            )
            if (
                provider_kind == "generation"
                and providers
                and provider_id == _BUILTIN_GENERATION_PROVIDER
            ):
                raise CompositionError(
                    f"generation provider pack name {_BUILTIN_GENERATION_PROVIDER!r} is reserved"
                )
            for provider in providers:
                schema = schemas.get(provider.node)
                if schema is None:
                    raise CompositionError(
                        f"{provider_kind} provider {provider_id!r} names unknown node "
                        f"{provider.node!r}"
                    )
                provider_input = schema.input("provider")
                expected_route = f"/api/choices/{provider.choice}"
                if (
                    provider_input is None
                    or provider_input.required is not required_input
                    or not isinstance(provider_input.widget, ComboWidget)
                    or bool(provider_input.widget.options)
                    or provider_input.widget.remote_route != expected_route
                    or not provider_input.hidden
                    or provider_input.advanced
                ):
                    raise CompositionError(
                        f"{provider_kind} provider {provider_id!r} choice {provider.choice!r} "
                        f"does not match {provider.node!r} provider input"
                    )
                if provider_kind == "vision":
                    vision_provider = cast("VisionProvider", provider)
                    model_input = schema.input("model")
                    if model_input is not None:
                        model_options = (
                            {
                                option.value if isinstance(option, ComboOption) else option
                                for option in model_input.widget.options
                            }
                            if isinstance(model_input.widget, ComboWidget)
                            else set()
                        )
                        if (
                            vision_provider.model is None
                            or vision_provider.model not in model_options - {"auto"}
                        ):
                            raise CompositionError(
                                f"vision provider {provider_id!r} model does not match "
                                f"{provider.node!r} model choices"
                            )
                    elif vision_provider.model is not None:
                        raise CompositionError(
                            f"vision provider {provider_id!r} declares a model for "
                            f"{provider.node!r}, which has no model choice"
                        )
                choice_owner = choice_owners.get(provider.choice)
                if choice_owner is None:
                    raise CompositionError(
                        f"{provider_kind} provider node {provider.node!r} names choice id "
                        f"{provider.choice!r} without a static choice owner"
                    )
                if static_choices[provider.choice]:
                    raise CompositionError(
                        f"{provider_kind} provider choice id {provider.choice!r} "
                        "must be declared empty"
                    )
                if choice_owner != schema_owners[provider.node]:
                    raise CompositionError(
                        f"{provider_kind} provider node {provider.node!r} and choice id "
                        f"{provider.choice!r} must have the same owning pack"
                    )
                provider_key = (provider.choice, provider_id)
                if provider_kind == "vision":
                    if provider_key in vision_providers_by_choice:
                        raise CompositionError(
                            f"duplicate vision provider {provider_id!r} for choice "
                            f"{provider.choice!r}"
                        )
                    vision_providers_by_choice.add(provider_key)
                route_key = (provider.node, provider_id)
                if route_key in routes:
                    raise CompositionError(
                        f"duplicate {provider_kind} provider {provider_id!r} for node "
                        f"{provider.node!r}"
                    )
                arms = topology.get(provider.node, ())
                if not any(arm.name == provider_id for arm in arms):
                    raise CompositionError(
                        f"{provider_kind} provider {provider_id!r} has no dispatch arm for "
                        f"{provider.node!r}"
                    )
                provider_choices.setdefault(provider.choice, set())
                provider_choices[provider.choice].add(provider_id)
                routes[route_key] = provider_id

        for node_type, arms in topology.items():
            for arm in arms:
                if provider_kind == "vision":
                    provider_id = arm.vision_provider
                    choice_id = arm.vision_provider_choice
                else:
                    provider_id = arm.generation_provider
                    choice_id = arm.generation_provider_choice
                if provider_id is None or choice_id is None:
                    continue
                provider_choices.setdefault(choice_id, set()).add(provider_id)
                routes.setdefault((node_type, provider_id), arm.name)

        choices: dict[str, tuple[str, ...]] = {}
        for choice_id, provider_ids in provider_choices.items():
            values = tuple(sorted(provider_ids))
            try:
                combo_choices_json_bytes(values, subject=f"derived choice {choice_id!r}")
            except ValueError as exc:
                raise CompositionError(str(exc)) from exc
            choices[choice_id] = values
        return choices, routes

    def _validated_vision_providers(
        self,
        records: Mapping[str, _PackRecord],
        topology: Topology,
    ) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], str]]:
        return self._validated_providers(
            records,
            topology,
            provider_kind="vision",
            required_input=False,
        )

    def _validated_generation_providers(
        self,
        records: Mapping[str, _PackRecord],
        topology: Topology,
    ) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], str]]:
        return self._validated_providers(
            records,
            topology,
            provider_kind="generation",
            required_input=False,
        )

    def _validated_derived_choices(
        self,
        sampler_registry: SamplerRegistrySnapshot,
        scheduler_registry: tuple[KeyedContribution, ...],
        records: Mapping[str, _PackRecord],
        topology: Topology,
    ) -> dict[str, tuple[str, ...]]:
        choices = self._validated_sampler_choices(sampler_registry)
        canonical_schedulers = tuple(item.id for item in scheduler_registry)
        compat_schedulers = registry_choice_values(scheduler_registry)
        try:
            combo_choices_json_bytes(
                canonical_schedulers, subject="derived choice 'dinkster.schedulers'"
            )
            combo_choices_json_bytes(compat_schedulers, subject="derived choice 'comfy.schedulers'")
        except ValueError as exc:
            raise CompositionError(str(exc)) from exc
        choices.update(
            {
                "dinkster.schedulers": canonical_schedulers,
                "comfy.schedulers": compat_schedulers,
            }
        )
        for provider_kind, provider_choices in (
            ("vision", self._validated_vision_providers(records, topology)[0]),
            ("generation", self._validated_generation_providers(records, topology)[0]),
        ):
            for choice_id, values in provider_choices.items():
                if choice_id in choices:
                    raise CompositionError(
                        f"choice id {choice_id!r} has multiple derived owners including "
                        f"{provider_kind} providers"
                    )
                choices[choice_id] = values
        return choices

    def _apply_sampler_choices(self, choices: Mapping[str, tuple[str, ...]]) -> None:
        """Publish native ids and the aliases consumed by native KSampler arms.

        Once a pack declares ``comfy.samplers``, that route is the legacy name
        surface for the effective native registry, not a literal ComfyUI list.
        """
        canonical = choices["dinkster.samplers"]
        self._core_choices["dinkster.samplers"] = canonical
        self.composition.choices["dinkster.samplers"] = canonical
        if "comfy.samplers" in self.composition.choices:
            self.composition.choices["comfy.samplers"] = choices["comfy.samplers"]
        scheduler_ids = choices["dinkster.schedulers"]
        self._core_choices["dinkster.schedulers"] = scheduler_ids
        self.composition.choices["dinkster.schedulers"] = scheduler_ids
        if "comfy.schedulers" in self.composition.choices:
            self.composition.choices["comfy.schedulers"] = choices["comfy.schedulers"]

    def _apply_derived_choices(self, choices: Mapping[str, tuple[str, ...]]) -> None:
        self._apply_sampler_choices(choices)
        for provider_choices in (
            self._validated_vision_providers(self._records, self._topology)[0],
            self._validated_generation_providers(self._records, self._topology)[0],
        ):
            for choice_id in provider_choices:
                self.composition.choices[choice_id] = choices.get(choice_id, ())
        labels: dict[str, dict[str, str]] = {
            node_type: {}
            for node_type in self._generation_provider_nodes(
                self._records, self.composition.schemas
            )
        }
        declared_schemas = dict(self._core_schemas)
        for record in self._records.values():
            declared_schemas.update(record.delta.schemas)
            for provider in record.manifest.generation_providers:
                labels.setdefault(provider.node, {})[record.manifest.name] = (
                    provider.label or record.manifest.name
                )
        for node_type, arms in self._topology.items():
            for arm in arms:
                if arm.generation_provider_choice is not None and arm.generation_provider not in (
                    None,
                    _BUILTIN_GENERATION_PROVIDER,
                ):
                    labels.setdefault(node_type, {})[arm.generation_provider] = (
                        arm.generation_provider_label or arm.generation_provider
                    )
        for node_type, provider_labels in labels.items():
            schema = self.composition.schemas.get(node_type)
            declared_schema = declared_schemas.get(node_type)
            if schema is None or declared_schema is None:
                continue
            provider_input = schema.input("provider")
            declared_input = declared_schema.input("provider")
            if (
                provider_input is None
                or declared_input is None
                or not isinstance(provider_input.widget, ComboWidget)
                or not isinstance(declared_input.widget, ComboWidget)
            ):
                continue
            displayed_input = replace(
                provider_input,
                widget=replace(
                    provider_input.widget,
                    options=(
                        ComboOption(_BUILTIN_GENERATION_PROVIDER, "Built-in"),
                        *(
                            ComboOption(provider_id, label)
                            for provider_id, label in sorted(provider_labels.items())
                        ),
                    ),
                    remote_route="",
                ),
            )
            self.composition.schemas[node_type] = replace(
                schema,
                inputs=tuple(
                    displayed_input if input_spec.id == "provider" else input_spec
                    for input_spec in schema.inputs
                ),
            )

    def _rebuild_registries(self) -> None:
        """Recompute the composed surface and the ownership registries from
        the host kernel, every live pack
        record, and every composed remote. Reload uses this instead
        of incremental removal because shared state (the compat workers'
        one "comfy" table entry and claim) has no per-pack decrement - the
        records are the source of truth, so derive, don't unpick."""
        composition = self.composition
        composition.schemas.clear()
        composition.schemas.update(self._core_schemas)
        composition.packs.clear()
        composition.packs.update(self._core_packs)
        composition.node_packs.clear()
        composition.execution_arms.clear()
        composition.choices.clear()
        composition.choices.update(self._core_choices)
        composition.lazy_choices.clear()
        composition.compat_skips.clear()
        self._owners.clear()
        self._owners.update(dict.fromkeys(self._core_schemas, CORE_PACK_ID))
        self._seen_names = {}
        self._claim_owners = {}
        self._choice_owners.clear()
        self._choice_owners.update(dict.fromkeys(self._core_choices, CORE_PACK_ID))
        self._compat_skip_owners = {}
        for name, record in self._records.items():
            self._merge_delta_registrations(name, record.canonical, record.delta)
            for claim in record.claims:
                self._claim_owners.setdefault(claim, name)
        for name, remote in self._remotes.items():
            self._merge_delta_registrations(name, canonical_name(name), remote.delta)
        self._sync_execution_arms()
        self._rebuild_asset_catalog()

    def _merge_delta_registrations(self, name: str, canonical: str, delta: PackDelta) -> None:
        """Fold one live delta's surface and ownership rows into freshly
        cleared registries (packs and remotes share this shape)."""
        composition = self.composition
        composition.schemas.update(delta.schemas)
        composition.packs.update(delta.packs)
        composition.node_packs.update(delta.node_packs)
        composition.choices.update(delta.choices)
        composition.lazy_choices.update(delta.lazy_choices)
        for pack_id, skips in delta.compat_skips.items():
            composition.compat_skips.setdefault(pack_id, {}).update(skips)
            for node_id in skips:
                self._compat_skip_owners[(pack_id, node_id)] = name
        for node_type in delta.schemas:
            self._owners[node_type] = name
        for choice_id in delta.choices:
            self._choice_owners[choice_id] = name
        for choice_id in delta.lazy_choices:
            self._choice_owners[choice_id] = name
        self._seen_names[canonical] = name

    def _sync_execution_arms(self) -> None:
        """Project internal topology into stable user-facing arm names."""
        composition = self.composition
        composition.execution_arms.clear()
        for node_type in composition.schemas:
            arms = self._topology.get(node_type)
            composition.execution_arms[node_type] = (
                tuple(dict.fromkeys(arm.execution_arm for arm in arms))
                if arms is not None
                else ("native",)
            )

    @staticmethod
    def _register_worker_renditions(registry: TypeRegistry, owner: str, worker: Any) -> None:
        for declaration in worker.renditions:
            existing = next(
                (
                    item
                    for item in registry.renditions_of(declaration.type_id)
                    if item.kind == declaration.kind
                ),
                None,
            )
            if existing is not None:
                if existing.relay_owner is None:
                    continue
                if (
                    existing.mime != declaration.mime
                    or existing.default != declaration.default
                    or existing.version != declaration.version
                    or existing.parameters != declaration.parameters
                    or dict(existing.defaults or {}) != dict(declaration.defaults or {})
                    or dict(existing.limits or {}) != dict(declaration.limits or {})
                ):
                    raise ValueError(
                        f"{declaration.type_id}: conflicting rendition declaration "
                        f"for {declaration.kind!r}"
                    )
                continue

            async def resolve(
                metadata: Mapping[str, object],
                parameters: Mapping[str, str],
                *,
                type_id: str = declaration.type_id,
                kind: str = declaration.kind,
            ) -> tuple[str, Mapping[str, str]]:
                return await worker.resolve_rendition(type_id, kind, metadata, parameters)

            async def resolve_mime(
                metadata: Mapping[str, object],
                *,
                type_id: str = declaration.type_id,
                kind: str = declaration.kind,
            ) -> str:
                return await worker.resolve_rendition_mime(type_id, kind, metadata)

            async def render(
                value: Value,
                parameters: Mapping[str, str],
                *,
                kind: str = declaration.kind,
            ) -> Rendition:
                return await worker.render_rendition(value, kind, parameters)

            registry.register_relayed_rendition(
                declaration.type_id,
                declaration.kind,
                mime=declaration.mime,
                default=declaration.default,
                version=declaration.version,
                parameters=declaration.parameters,
                defaults=declaration.defaults,
                limits=declaration.limits,
                owner=owner,
                resolve_mime=resolve_mime,
                resolve=resolve,
                render=render,
            )

    def _rebuild_asset_catalog(self) -> None:
        """Recompute the pack-asset catalog from the live records - the
        same derive-don't-unpick move as the registries: add, reload, and
        remove all end in one wholesale swap, so preflight sees the old
        surface or the new one, never a torn mix."""
        entries: list[tuple[str, Path | None, tuple[DeclaredAsset, ...]]] = []
        provider_entries: list[tuple[str, str, tuple[DeclaredAsset, ...]]] = []
        for record in self._records.values():
            for pack_id, info in record.delta.packs.items():
                if info.assets or pack_id in record.asset_roots:
                    entries.append((pack_id, record.asset_roots.get(pack_id), info.assets))
            assets_by_id = {asset.id: asset for asset in record.manifest.assets}
            for provider in record.manifest.vision_providers:
                provider_entries.append(
                    (
                        provider.node,
                        record.manifest.name,
                        tuple(assets_by_id[asset_id] for asset_id in provider.artifacts),
                    )
                )
        self.composition.asset_catalog.replace_all(
            entries,
            provider_entries=provider_entries,
        )

    def pack_specs(self) -> dict[str, PackSpec]:
        """The spec every composed pack currently runs from, keyed by
        pack name - what live activation diffs against a new install
        generation's specs to decide add/remove/reload/unchanged."""
        return {name: record.spec for name, record in self._records.items()}

    def validate_complete_generation(self) -> None:
        """Require providers except for explicitly optional schema-only types."""
        missing = tuple(
            node_type
            for record in self._records.values()
            for node_type in record.schema_only
            if node_type not in record.spec.optional_execution and not self._topology.get(node_type)
        )
        if missing:
            raise CompositionError(
                "schema-only node types have no execution provider: " + ", ".join(sorted(missing))
            )

    def incomplete_generation_removals(self) -> dict[str, tuple[str, ...]]:
        """Return dependent-first pack removals for an incomplete generation."""
        records = {canonical_name(name): (name, record) for name, record in self._records.items()}
        missing = {
            canonical: tuple(
                node_type
                for node_type in record.schema_only
                if node_type not in record.spec.optional_execution
                and not self._topology.get(node_type)
            )
            for canonical, (_name, record) in records.items()
        }
        missing = {name: nodes for name, nodes in missing.items() if nodes}
        if not missing:
            return {}

        capability_owners = {
            canonical_name(capability.id): canonical
            for canonical, (_name, record) in records.items()
            for capability in record.manifest.capabilities
        }
        removals = set(missing)
        while True:
            previous = len(removals)
            removed_schemas = {
                node_type
                for canonical in removals
                for node_type in records[canonical][1].delta.schemas
            }
            for canonical, (_name, record) in records.items():
                if canonical in removals:
                    continue
                pack_dependencies = {
                    canonical_name(dependency.pack) for dependency in record.manifest.dependencies
                }
                capability_dependencies = {
                    owner
                    for requirement in record.manifest.requirements.capabilities
                    if (owner := capability_owners.get(canonical_name(requirement.id))) is not None
                }
                if removals.intersection(pack_dependencies | capability_dependencies) or set(
                    record.executes
                ).intersection(removed_schemas):
                    removals.add(canonical)
            if len(removals) == previous:
                break

        dependencies: dict[str, set[str]] = {canonical: set() for canonical in removals}
        for canonical in removals:
            record = records[canonical][1]
            dependencies[canonical].update(
                dependency
                for declared in record.manifest.dependencies
                if (dependency := canonical_name(declared.pack)) in removals
            )
            dependencies[canonical].update(
                provider
                for requirement in record.manifest.requirements.capabilities
                if (provider := capability_owners.get(canonical_name(requirement.id))) in removals
            )
            dependencies[canonical].update(
                owner
                for owner in removals
                if set(record.executes).intersection(records[owner][1].delta.schemas)
            )
        remaining = {canonical: set(required) for canonical, required in dependencies.items()}
        composition_order: list[str] = []
        while remaining:
            ready = sorted(canonical for canonical, required in remaining.items() if not required)
            if not ready:
                raise CompositionError(
                    "incomplete generation packs cannot be retracted independently: "
                    + ", ".join(sorted(remaining))
                )
            composition_order.extend(ready)
            for canonical in ready:
                del remaining[canonical]
            for required in remaining.values():
                required.difference_update(ready)
        return {
            records[canonical][0]: missing.get(canonical, ())
            for canonical in reversed(composition_order)
        }

    def watch_targets(self) -> dict[str, Path]:
        """Source root of every composed pack, for the dev file watcher:
        pack name -> the directory containing its manifest. Reads the live
        records, so packs composed after startup (or reloaded) appear on
        the next call. A record whose manifest no longer resolves (deleted
        mid-session) is skipped - the watcher cannot watch what is gone,
        and an explicit reload of it would fail loudly anyway."""
        targets: dict[str, Path] = {}
        for name, record in self._records.items():
            try:
                manifest_path = resolve_manifest_path(record.spec.manifest)
            except CompositionError:
                continue
            targets[name] = manifest_path.parent
        return targets

    async def close(self) -> None:
        try:
            for record in self._records.values():
                if record.tenant_registrations is not None and record.owns_tenant_registrations:
                    await record.tenant_registrations.close()
            await self.composition.close()
        finally:
            self._group_activations.clear()
            if self._sampler_catalog_root is not None:
                shutil.rmtree(self._sampler_catalog_root, ignore_errors=True)
                self._sampler_catalog_root = None


async def compose_serving(
    pack_manifests: Sequence[PackSpec | Path | str] = (),
    *,
    include_default_packs: bool = True,
    worker_env: Mapping[str, str] | None = None,
    sandbox_policy: SandboxPolicy | None = None,
    pack_scratch_root: Path | str | None = None,
    dev: bool = False,
    on_diagnostic: DiagnosticListener | None = None,
    explain_misses: bool = False,
    governor: MemoryGovernor | None = None,
    reservations: ReservationService | None = None,
    telemetry: ReportedTelemetry | None = None,
    cache_mode: ExecutionCacheMode = "memory",
    cache_memory_entries: int = 1024,
    cache_dir: Path | str | None = None,
    cache_disk_budget: int = DEFAULT_DISK_CACHE_BYTES,
    composition_mode: PackCompositionMode = "development",
) -> Composition:
    """Compose the installed default pack set and every requested pack.

    Entries are manifest paths (host interpreter, manifest-name
    attribution) or :class:`PackSpec` for anything richer. ``worker_env``
    is merged into every child's environment, under any spec's own ``env``
    (the child inherits the host's environment either way - this is for
    extras like a test PYTHONPATH). ``pack_scratch_root`` assigns stable
    per-pack or per-group directories through ``DINKSTER_PACK_SCRATCH``.
    Per-pack venvs (``ensure_pack_venv``) are a host policy this first cut
    does not wire. ``dev`` turns on the dev-mode diagnostics (DESIGN 3.9):
    engines explain cache misses as ``cache_miss`` events and every isolated
    worker logs its per-invocation boundary costs through
    :func:`log_boundary_diagnostic`.

    ``on_diagnostic`` is a host-supplied BoundaryDiagnostic consumer wired
    to every isolated worker alongside (not instead of) the dev logger -
    benchmark assembly (DESIGN 3.9) attaches here. ``explain_misses``
    turns on cache-miss explanation events without the rest of dev mode,
    so benchmark records can attribute misses on a production posture.

    On any failure, workers already started are closed before the error
    propagates - no orphaned pack processes.

    This convenience constructor is atomic: failure to resolve or compose any
    default pack fails the call. The production server uses ``ServingComposer``
    progressively so it can report one failed pack while serving the others.
    """
    entries = (
        (*default_pack_specs(), *pack_manifests) if include_default_packs else tuple(pack_manifests)
    )
    _validate_worker_group_specs(entries, worker_env or {})
    composer = ServingComposer(
        worker_env=worker_env,
        sandbox_policy=sandbox_policy,
        pack_scratch_root=pack_scratch_root,
        dev=dev,
        on_diagnostic=on_diagnostic,
        explain_misses=explain_misses,
        governor=governor,
        reservations=reservations,
        telemetry=telemetry,
        cache_mode=cache_mode,
        cache_memory_entries=cache_memory_entries,
        cache_dir=cache_dir,
        cache_disk_budget=cache_disk_budget,
        composition_mode=composition_mode,
    )
    try:
        for entry in composer.order_pack_entries(entries):
            await composer.add_pack(entry)
        composer.validate_complete_generation()
    except BaseException:
        await composer.close()
        raise
    return composer.composition
