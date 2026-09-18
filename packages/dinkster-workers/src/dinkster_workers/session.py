"""BoundarySession: the engine side of one boundary conversation.

IsolatedWorker (a child process this engine launched) and RemoteWorker (a
service on another machine) speak the *same* protocol: hello, invoke/result,
cancel, shm/result acks, memory leases, the consumer relay, and the
two-phase ram-release gate. This class is that protocol, once - the workers
own only what genuinely differs: how the connection comes to exist and what
dying means (a subprocess to reap vs a socket to close).

The session begins on an already-authenticated (reader, writer) pair, reads
the peer's hello, and serves invocations until the stream ends. Everything
conservative about failure lives here: a resumable remote retains admitted
invocations only for its negotiated grace, every other dead peer fails pending
invocations with a NodeError, relay footprints collapse to zero, and a pending
release never claims bytes were freed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, TypeVar, cast

from dinkster_assets import AssetError, AssetResolver, AssetVault, DeclaredAsset, resolver_from_env
from dinkster_memory import (
    BudgetExceeded,
    MeasuredMemory,
    MemoryGovernor,
    ReportedTelemetry,
    ReservationRequest,
    ReservationService,
    ReservationTimeout,
    Shedder,
)
from dinkster_protocol import (
    GRAPH_COMPILE_CANCEL_TYPE,
    GRAPH_COMPILE_REQUEST_TYPE,
    GRAPH_COMPILE_RESULT_TYPE,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    CompatGateDiagnostic,
    CompositionMode,
    ContributionSurfaceDescriptor,
    ExtensionScope,
    Invocation,
    InvocationEvent,
    InvocationResult,
    KeyedContribution,
    LazyStatusInvocation,
    LazyStatusResult,
    NodeError,
    OnInvocationEvent,
    ReplicaId,
    SavedArtifact,
    SavedArtifactCandidate,
    WorkGroupDefinition,
    attention_capability_evidence_from_wire,
    attention_route_token_from_wire,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
)
from dinkster_protocol.pack_surfaces import PackRoute, pack_surfaces_from_wire
from dinkster_schema import (
    ComfyAliasRegistry,
    ComfyGroupRegistry,
    NodeSchema,
    combo_choices_json_bytes,
    comfy_alias_registry_from_wire,
    comfy_group_registry_from_wire,
    schema_from_wire,
    validate_name,
)
from dinkster_values import TypeRegistry, value_resource_ids

from .blobs import BlobTransfer, attribute_moved, await_file_operation
from .boundary import (
    BoundaryError,
    TransferStat,
    ValueCodec,
    ValueStore,
    decode_result_artifact_candidates,
    decode_result_outputs,
    encode_invocation,
    error_from_wire,
    read_frame,
    release_segment,
    write_frame,
)
from .devices import DeviceMap
from .diagnostics import BoundaryDiagnostic, DiagnosticListener, EdgeCost
from .manifest import (
    GenerationProvider,
    VisionProvider,
    generation_providers_from_wire,
    vision_providers_from_wire,
)
from .produced_assets import ProducedAssetAuthority, result_asset_digests
from .relay import MemoryRelay, ReleaseGuard, WorkerFullReleaseResult
from .resume import InvocationKey
from .staging import StageAsset, declared_assets_from_wire
from .workgroup import ReplicaEndpoint
from .workgroup_session import WORKGROUP_FRAME_TYPE, WorkGroupSession

ArtifactAuthority = Callable[
    [Sequence[SavedArtifactCandidate], str, int], tuple[SavedArtifact, ...]
]
SchemaReloadListener = Callable[[str, str], None]
_T = TypeVar("_T")


def _retrieve_future_exception(task: asyncio.Future[_T]) -> None:
    if not task.cancelled():
        task.exception()


def _combo_choices_from_hello(
    header: Mapping[str, Any], *, role: str, pack: str
) -> dict[str, tuple[str, ...]]:
    """Parse the hello's optional ``comboChoices`` field strictly: the
    producing side is our own worker host (which already validated), so a
    malformed table means a miswired or incompatible peer - fail the
    handshake loudly, never serve a half-parsed choice list."""
    raw = header.get("comboChoices")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{role} '{pack}' hello carried a malformed comboChoices table")
    choices: dict[str, tuple[str, ...]] = {}
    for choice_id, values in cast("Mapping[Any, Any]", raw).items():
        if (
            not isinstance(choice_id, str)
            or not choice_id
            or validate_name(choice_id) is not None
            or not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not all(isinstance(v, str) and v for v in cast("Sequence[Any]", values))
        ):
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed comboChoices entry for {choice_id!r}"
            )
        parsed = tuple(cast("Sequence[str]", values))
        try:
            combo_choices_json_bytes(
                parsed,
                subject=f"{role} '{pack}' hello choice {choice_id!r}",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed comboChoices "
                f"entry for {choice_id!r}: {exc}"
            ) from exc
        choices[choice_id] = parsed
    return choices


def _lazy_choice_ids_from_hello(
    header: Mapping[str, Any], *, role: str, pack: str, static_ids: Collection[str]
) -> tuple[str, ...]:
    """Parse the hello's optional ``lazyChoiceIds`` field strictly, same
    posture as ``_combo_choices_from_hello``: ids only (values are fetched
    per request), grammar-valid, no duplicates, and disjoint from the same
    hello's static comboChoices - one id has exactly one evaluation mode."""
    raw = header.get("lazyChoiceIds")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError(f"{role} '{pack}' hello carried a malformed lazyChoiceIds list")
    ids: list[str] = []
    seen: set[str] = set()
    for choice_id in cast("Sequence[Any]", raw):
        if not isinstance(choice_id, str) or not choice_id or validate_name(choice_id) is not None:
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed lazyChoiceIds entry {choice_id!r}"
            )
        if choice_id in seen:
            raise RuntimeError(
                f"{role} '{pack}' hello carried duplicate lazyChoiceIds entry {choice_id!r}"
            )
        if choice_id in static_ids:
            raise RuntimeError(
                f"{role} '{pack}' hello announced {choice_id!r} as both a "
                "static comboChoices list and a lazy choice id"
            )
        seen.add(choice_id)
        ids.append(choice_id)
    return tuple(ids)


def _compat_skips_from_hello(
    header: Mapping[str, Any], *, role: str, pack: str
) -> dict[str, CompatGateDiagnostic]:
    """Parse the hello's optional classified compat skip mapping."""
    raw = header.get("compatSkips")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{role} '{pack}' hello carried a malformed compatSkips table")
    skips: dict[str, CompatGateDiagnostic] = {}
    for node_name, diagnostic_wire in cast("Mapping[Any, Any]", raw).items():
        if not isinstance(node_name, str) or not node_name:
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed compatSkips entry for {node_name!r}"
            )
        try:
            diagnostic = CompatGateDiagnostic.from_wire(diagnostic_wire)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed compatSkips "
                f"entry for {node_name!r}: {exc}"
            ) from exc
        if diagnostic.source_node != node_name:
            raise RuntimeError(
                f"{role} '{pack}' hello carried a malformed compatSkips "
                f"entry for {node_name!r}: sourceNode differs"
            )
        skips[node_name] = diagnostic
    return skips


def _body_arms_from_hello(
    header: Mapping[str, Any], *, role: str, pack: str
) -> dict[str, tuple[str, ...]] | None:
    raw = header.get("bodyArms")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{role} '{pack}' hello carried malformed bodyArms")
    arms: dict[str, tuple[str, ...]] = {}
    for arm, node_types in cast("Mapping[Any, Any]", raw).items():
        if (
            not isinstance(arm, str)
            or not isinstance(node_types, Sequence)
            or isinstance(node_types, (str, bytes))
            or not all(
                isinstance(node_type, str) and node_type
                for node_type in cast("Sequence[Any]", node_types)
            )
        ):
            raise RuntimeError(f"{role} '{pack}' hello carried malformed bodyArms entry {arm!r}")
        arms[arm] = tuple(cast("Sequence[str]", node_types))
    return arms


def _extension_contributions_from_hello(
    header: Mapping[str, Any], *, role: str, pack: str
) -> tuple[tuple[ExtensionScope, ContributionSurfaceDescriptor], ...]:
    raw = header.get("extensionContributions")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError(f"{role} '{pack}' hello carried malformed extensionContributions")
    contributions: list[tuple[ExtensionScope, ContributionSurfaceDescriptor]] = []
    for item in cast("Sequence[object]", raw):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"{role} '{pack}' hello carried malformed extension contribution")
        fields = cast("Mapping[object, object]", item)
        if not {"scope", "surfaceId", "mode"} <= set(fields) or set(fields) - {
            "scope",
            "surfaceId",
            "mode",
            "routes",
            "events",
        }:
            raise RuntimeError(
                f"{role} '{pack}' hello carried malformed extension contribution fields"
            )
        try:
            scope = ExtensionScope(fields["scope"])
            routes, events = pack_surfaces_from_wire(cast(Mapping[str, object], fields))
            descriptor = ContributionSurfaceDescriptor(
                surface_id=cast("str", fields["surfaceId"]),
                mode=CompositionMode(fields["mode"]),
                routes=routes,
                events=events,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{role} '{pack}' hello carried malformed extension contribution: {exc}"
            ) from exc
        contributions.append((scope, descriptor))
    return tuple(contributions)


class WorkerDied(Exception):
    """The peer is gone; the caller decides what that means."""


class PackDeclarations:
    """Decode the portable subset shared by catalogs and worker announcements."""

    def __init__(
        self, declarations: Mapping[str, Any], *, role: str = "catalog", pack: str = "pack"
    ) -> None:
        self.declarations = dict(declarations)
        schemas_wire = declarations.get("schemas")
        if not isinstance(schemas_wire, dict):
            raise RuntimeError(f"{role} '{pack}' hello carried no schemas")
        self.schemas = {
            str(name): schema_from_wire(dict(cast("Mapping[str, Any]", schema)))
            for name, schema in cast("dict[str, Any]", schemas_wire).items()
        }
        if any(name != schema.node_type for name, schema in self.schemas.items()):
            raise ValueError("schema key does not match node type")
        self.combo_choices = _combo_choices_from_hello(declarations, role=role, pack=pack)
        self.lazy_choice_ids = _lazy_choice_ids_from_hello(
            declarations, static_ids=self.combo_choices, role=role, pack=pack
        )
        self.compat_skips = _compat_skips_from_hello(declarations, role=role, pack=pack)
        self.body_arms = _body_arms_from_hello(declarations, role=role, pack=pack)
        self.extension_contributions = _extension_contributions_from_hello(
            declarations, role=role, pack=pack
        )


class BoundarySession:
    """One protocol conversation with a worker peer, engine side."""

    def __init__(
        self,
        registry: TypeRegistry,
        *,
        role: str,
        pack: str,
        codec: ValueCodec,
        on_diagnostic: DiagnosticListener | None = None,
        reservations: ReservationService | None = None,
        device_map: DeviceMap | None = None,
        governor: MemoryGovernor | None = None,
        consumer_priority: int = 10,
        release_guard: ReleaseGuard | None = None,
        telemetry: ReportedTelemetry | None = None,
        death_detail: Callable[[], str] | None = None,
        artifact_authority: ArtifactAuthority | None = None,
        produced_asset_source: AssetResolver | None = None,
        on_schema_reload: SchemaReloadListener | None = None,
        resumable: bool = False,
    ) -> None:
        self._registry = registry
        self._role = role
        self._pack = pack
        self._codec = codec
        self._on_diagnostic = on_diagnostic
        self._death_detail = death_detail
        self._artifact_authority = artifact_authority
        vault_root = os.environ.get("DINKSTER_ASSET_VAULT")
        self._produced_assets = ProducedAssetAuthority(
            AssetVault(vault_root) if vault_root else None,
            produced_asset_source,
            resolver_from_env(),
        )
        self._received_asset_pins: dict[str, object] = {}
        self._asset_adoptions: set[asyncio.Task[None]] = set()
        self._on_schema_reload = on_schema_reload
        self._resumable = resumable
        self._closing = False
        self._owner_epoch = 0
        self._resume_grace = 0.0
        self._resume_timeout_task: asyncio.Task[None] | None = None
        self._rebind_future: asyncio.Future[dict[str, Any]] | None = None
        self._rebinding = False
        self._invocation_keys: dict[str, InvocationKey] = {}
        self._last_event_sequences: dict[str, int] = {}
        self._cancelled_invocations: set[str] = set()
        self._accepted_invocations: dict[str, asyncio.Event] = {}
        self._received_results: set[str] = set()
        self._result_ack_ready: set[str] = set()
        self._result_acknowledgements: dict[str, asyncio.Future[None]] = {}
        self._result_ack_tasks: dict[str, asyncio.Task[None]] = {}
        self._resume_outbox: list[tuple[dict[str, object], list[bytes]]] = []
        self._lease_decisions: dict[str, dict[str, object]] = {}
        self._pending: dict[str, asyncio.Future[tuple[dict[str, Any], list[bytes]]]] = {}
        self._lazy_pending: dict[str, asyncio.Future[tuple[dict[str, Any], list[bytes]]]] = {}
        self._sampler_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._sampler_request_index = 0
        self._conversion_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._conversion_request_index = 0
        self._graph_compile_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._graph_compile_request_index = 0
        self._route_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._route_request_index = 0
        # Asset staging (declared assets on a remote daemon): request
        # futures for assetQuery/stageAssets replies, plus per-request
        # sinks for the daemon's stageEvent progress frames.
        self._staging_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._staging_request_index = 0
        # Lazy choice fetches: request futures for choicesResult replies.
        # A future is popped by its awaiter's finally, so a reply landing
        # after a timeout finds nothing and is discarded.
        self._choices_pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._choices_request_index = 0
        self._stage_event_sinks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._asset_staging = False
        self._declared_assets: tuple[DeclaredAsset, ...] = ()
        # Per-invocation event sinks: invocationEvent frames from the peer
        # are routed here by invocationId while the invocation is pending.
        # Events for an id with no sink (caller passed none, or the result
        # already landed) are dropped - they are chatter by contract.
        self._event_sinks: dict[str, OnInvocationEvent] = {}
        # Memory leases (DESIGN 3.10): the peer names what it is about to
        # materialize; this side holds the governor-backed reservation until
        # the peer releases it. None means grant-without-accounting - the
        # unbudgeted posture, never a block.
        self._reservations = reservations
        self._leases: dict[str, tuple[asyncio.Task[None], asyncio.Event]] = {}
        # Device namespacing (see devices.py): the peer's "cuda:0" is
        # translated into the parent's namespace as facts cross the boundary
        # - output meta (lanes, budgets) and lease residency classes.
        self._device_map = device_map
        # Memory relay (relay.py): consumers the peer announces in hello
        # become governed proxies here, registered/unregistered with the
        # governor across this session's lifetime - the composing host cannot
        # forget to wire a pool it never sees. Worker consumers default to a
        # later shed priority than caches (register_shedder's convention:
        # caches shed before resource pools).
        self._governor = governor
        self._consumer_priority = consumer_priority
        # The cross-process ram-release gate (relay.py): without it, ram
        # pressure on this worker's consumers is refused at the proxy. The
        # guard's pins must be the same registry the Engine pins run-held
        # envelopes into; invoke() also takes short handoff pins in it so a
        # result being decoded is never invisible to a concurrent release.
        self._release_guard = release_guard
        # Worker-reported measurements (reported.py): "measured" mappings
        # from hello/memoryReport/memoryShedResult frames land here,
        # DeviceMap-translated, keyed by this session. Informational only
        # - the governor reads them for status, never admission - and the
        # snapshot dies with the session (close AND read-loop death).
        self._telemetry = telemetry
        self._relay: MemoryRelay | None = None
        self._closed_relays: list[MemoryRelay] = []
        self._relay_proxies: list[Shedder] = []
        self._memory_consumers: dict[str, bool] | None = None
        self._sent_segments: dict[str, SharedMemory] = {}
        self._send_lock = asyncio.Lock()
        self._blob_store: ValueStore | None = None
        self._blob_transfer: BlobTransfer | None = None
        self._schemas: dict[str, NodeSchema] | None = None
        self._comfy_aliases: ComfyAliasRegistry | None = None
        self._comfy_groups: ComfyGroupRegistry | None = None
        self._combo_choices: dict[str, tuple[str, ...]] = {}
        self._lazy_choice_ids: tuple[str, ...] = ()
        self._compat_skips: dict[str, CompatGateDiagnostic] = {}
        self._body_arms: dict[str, tuple[str, ...]] | None = None
        self._vision_providers: tuple[VisionProvider, ...] | None = None
        self._generation_providers: tuple[GenerationProvider, ...] | None = None
        self._lazy_status = False
        self._convert_legacy_checkpoint = False
        self._schema_reload = False
        self._extension_contributions: tuple[
            tuple[ExtensionScope, ContributionSurfaceDescriptor], ...
        ] = ()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._last_received = 0.0
        self._alive = False
        self._instance_token: str | None = None
        self._attention_capabilities: AttentionCapabilityEvidence | None = None
        self._attention_route_token: AttentionRouteToken | None = None
        self._workgroup = WorkGroupSession(self.send)

    @property
    def pack(self) -> str:
        return self._pack

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def instance_token(self) -> str | None:
        """The peer's process-lifetime token from the hello handshake
        (``workerInstance``); None when the peer announced none. Matches
        the RESOURCE_OWNER_META_KEY the peer's resident codecs stamp, so
        owner tokens resolve to sessions. A token never survives the
        session: a restarted peer minted a new one."""
        return self._instance_token

    @property
    def device_map_wire(self) -> dict[str, object]:
        mapping = {} if self._device_map is None else dict(self._device_map.mapping)
        qualifier = None if self._device_map is None else self._device_map.qualifier
        return {"mapping": mapping, "qualifier": qualifier}

    async def full_release(
        self,
        request_id: str,
        worker_instance: str,
        *,
        release_guard: ReleaseGuard | None = None,
    ) -> WorkerFullReleaseResult:
        """Release every consumer declared by this exact worker lifetime."""
        if not self._alive or self._instance_token != worker_instance:
            raise WorkerDied()
        if self._memory_consumers is None or self._relay is None:
            raise RuntimeError("worker memory consumer declarations are unavailable")
        result = await self._relay.full_release(
            request_id, worker_instance, release_guard=release_guard
        )
        if result is None:
            raise WorkerDied()
        names = [str(item.get("consumer", "")) for item in result.consumers]
        if len(names) != len(set(names)) or set(names) != set(self._memory_consumers):
            raise WorkerDied()
        return result

    @property
    def attention_route_token(self) -> AttentionRouteToken | None:
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._attention_route_token

    @property
    def attention_capabilities(self) -> AttentionCapabilityEvidence | None:
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._attention_capabilities

    @property
    def schemas(self) -> Mapping[str, NodeSchema]:
        """The pack's node schemas, as announced by the hello handshake.
        Feed these to the Engine - this side never imports the pack."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._schemas

    @property
    def comfy_aliases(self) -> ComfyAliasRegistry | None:
        """Maintained ComfyUI import translations announced by the pack."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._comfy_aliases

    @property
    def comfy_groups(self) -> ComfyGroupRegistry | None:
        """ComfyUI group translations announced by the pack."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._comfy_groups

    @property
    def combo_choices(self) -> Mapping[str, tuple[str, ...]]:
        """The pack's combo choice lists (choice-list id -> values), as
        announced by the hello handshake; empty when the pack declares
        none. UI vocabulary for remote ComboWidget routes, never
        identity."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._combo_choices

    @property
    def lazy_choice_ids(self) -> tuple[str, ...]:
        """Choice-list ids whose values the pack computes per fetch (see
        ``fetch_choices``), announced by the hello handshake; disjoint from
        ``combo_choices`` by construction."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._lazy_choice_ids

    @property
    def compat_skips(self) -> Mapping[str, CompatGateDiagnostic]:
        """Classified compat translation skips announced at startup."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._compat_skips

    @property
    def body_arms(self) -> Mapping[str, tuple[str, ...]] | None:
        """Same-session body capabilities, or None for an older peer that
        omitted the capability entirely."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._body_arms

    @property
    def vision_providers(self) -> tuple[VisionProvider, ...] | None:
        """Provider declarations, or None when an older peer omitted them."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._vision_providers

    @property
    def generation_providers(self) -> tuple[GenerationProvider, ...] | None:
        """Provider declarations, or None when an older peer omitted them."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._generation_providers

    @property
    def asset_staging(self) -> bool:
        """Whether the peer's hello negotiated the asset staging frames
        (assetQuery/stageAssets/cancelStage). A peer that predates staging
        announces nothing and gets no staging frames."""
        return self._asset_staging

    @property
    def declared_assets(self) -> tuple[DeclaredAsset, ...]:
        """The peer pack's ``[[pack.assets]]`` declarations from the hello:
        the engine-side source of which digests a dispatched node type
        requires (each declaration's ``nodes``), since the declaring
        manifest lives on the peer."""
        return self._declared_assets

    @property
    def can_convert_legacy_checkpoint(self) -> bool:
        """Whether this peer advertised the optional conversion RPC."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._convert_legacy_checkpoint

    @property
    def extension_contributions(
        self,
    ) -> tuple[tuple[ExtensionScope, ContributionSurfaceDescriptor], ...]:
        """RPC-clean extension descriptors resolved inside the peer."""
        if self._schemas is None:
            raise RuntimeError("BoundarySession.begin() has not completed")
        return self._extension_contributions

    @property
    def workgroup_capabilities(self) -> frozenset[str]:
        """Workgroup capabilities negotiated by the authenticated hello."""
        return self._workgroup.capabilities

    def bind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> ReplicaEndpoint:
        """Bind one parent endpoint to this authenticated worker lifetime."""
        if not self._alive:
            raise RuntimeError(f"{self._role} '{self._pack}' is not running")
        return self._workgroup.bind(
            definition,
            replica,
            worker_instance=self._instance_token,
        )

    def enable_blob_transfer(self, store: ValueStore) -> None:
        """Attach the persistentCas blob transport - called only after both
        hellos advertised it, alongside the codec's enable_persistent_cas."""
        if self._blob_store is store and self._blob_transfer is not None:
            return
        if self._blob_transfer is not None:
            self._blob_transfer.close()
        self._blob_store = store
        self._reset_blob_transfer()

    def _reset_blob_transfer(self) -> None:
        if self._blob_store is None:
            self._blob_transfer = None
            return
        self._blob_transfer = BlobTransfer(
            self._blob_store,
            lambda header, blobs: self.send(header, blobs),
            closed_exc=WorkerDied,
            pin_owner=self,
        )

    def unbind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> None:
        """Remove only a pristine registration, including after transport loss."""
        self._workgroup.unbind(definition, replica)

    async def begin(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        timeout: float,
        resuming: bool = False,
        owner_epoch: int = 0,
        resume_grace: float = 0.0,
    ) -> dict[str, Any]:
        """Adopt an authenticated stream pair: read the peer's hello, wire
        the relay, start the read loop. Returns the raw hello header so a
        caller can validate negotiated fields (protocol, transports); on
        any failure the caller still owns close()."""
        previous_instance = self._instance_token
        if self._resume_timeout_task is not None:
            timeout_task = self._resume_timeout_task
            self._resume_timeout_task = None
            timeout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timeout_task
        if resuming and (not self._resumable or self._alive or not self._invocation_keys):
            raise RuntimeError(f"{self._role} '{self._pack}' has no invocation state to resume")
        self._owner_epoch = owner_epoch
        self._resume_grace = resume_grace
        self._closing = False
        self._rebinding = resuming
        self._reader = reader
        self._writer = writer
        frame = await asyncio.wait_for(read_frame(reader), timeout)
        if frame is not None and frame[0].get("type") == "error":
            # The peer refused us and said why (busy, protocol mismatch):
            # surface its words, not a generic handshake failure.
            raise RuntimeError(
                f"{self._role} '{self._pack}' refused: {frame[0].get('message', 'no reason given')}"
            )
        if frame is None:
            # A hangup before hello, not a malformed one: connection-level,
            # so the caller can say what a silent close means on its
            # transport (the remote service hangs up on a bad token).
            raise ConnectionError(
                f"{self._role} '{self._pack}' closed the stream before the hello handshake"
            )
        if frame[0].get("type") != "hello":
            raise RuntimeError(f"{self._role} '{self._pack}' failed to start: no hello handshake")
        header, _ = frame
        self._pack = str(header.get("pack", self._pack))
        declarations = PackDeclarations(header, role=self._role, pack=self._pack)
        self._schemas = declarations.schemas
        aliases_wire = header.get("comfyAliases")
        if aliases_wire is not None:
            try:
                self._comfy_aliases = comfy_alias_registry_from_wire(aliases_wire)
            except (KeyError, OverflowError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed comfyAliases: {exc}"
                ) from exc
        groups_wire = header.get("comfyGroups")
        if groups_wire is not None:
            try:
                self._comfy_groups = comfy_group_registry_from_wire(groups_wire)
            except (KeyError, OverflowError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed comfyGroups: {exc}"
                ) from exc
        self._combo_choices = declarations.combo_choices
        self._lazy_choice_ids = declarations.lazy_choice_ids
        self._compat_skips = declarations.compat_skips
        self._body_arms = declarations.body_arms
        self._vision_providers = None
        if "visionProviders" in header:
            try:
                self._vision_providers = vision_providers_from_wire(header["visionProviders"])
            except ValueError as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed visionProviders: {exc}"
                ) from exc
        self._generation_providers = None
        if "generationProviders" in header:
            try:
                self._generation_providers = generation_providers_from_wire(
                    header["generationProviders"]
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed generationProviders: "
                    f"{exc}"
                ) from exc
        self._lazy_status = header.get("lazyStatus") is True
        self._convert_legacy_checkpoint = header.get("convertLegacyCheckpoint") is True
        self._schema_reload = header.get("schemaReload") is True
        self._asset_staging = header.get("assetStaging") is True
        declared_raw = header.get("declaredAssets")
        if declared_raw is not None:
            try:
                self._declared_assets = declared_assets_from_wire(declared_raw)
            except AssetError as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed declaredAssets: {exc}"
                ) from exc
        self._extension_contributions = declarations.extension_contributions
        if "attentionRouteToken" in header:
            try:
                self._attention_route_token = attention_route_token_from_wire(
                    header["attentionRouteToken"]
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed "
                    f"attentionRouteToken: {exc}"
                ) from exc
        if "attentionCapabilities" in header:
            try:
                self._attention_capabilities = attention_capability_evidence_from_wire(
                    header["attentionCapabilities"]
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried malformed "
                    f"attentionCapabilities: {exc}"
                ) from exc
            if self._attention_route_token is None:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried attentionCapabilities "
                    "without attentionRouteToken"
                )
            try:
                expected_token = derive_attention_route_token(
                    self._attention_capabilities,
                    AttentionPolicyConfig(
                        self._attention_route_token.requested_policy,
                        self._attention_route_token.requested_role_policies,
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried inconsistent "
                    f"attentionCapabilities: {exc}"
                ) from exc
            if canonical_attention_route_token_bytes(
                expected_token
            ) != canonical_attention_route_token_bytes(self._attention_route_token):
                raise RuntimeError(
                    f"{self._role} '{self._pack}' hello carried attentionCapabilities "
                    "inconsistent with attentionRouteToken"
                )
        # The peer's process-lifetime token (optional: a remote peer may
        # predate it). Malformed is treated as absent, never coerced -
        # dispatch enrollment requires a real token and refuses without one.
        instance = header.get("workerInstance")
        self._instance_token = instance if isinstance(instance, str) and instance else None
        if resuming and self._instance_token != previous_instance:
            raise RuntimeError(f"{self._role} '{self._pack}' daemon identity changed during resume")
        if resuming:
            self._workgroup = WorkGroupSession(self.send)
        self._workgroup.negotiate(header)
        self._wire_relay(header)
        self._apply_measured(header)
        self._alive = True
        self._reader_task = asyncio.create_task(self._read_loop())
        if self._relay is not None and self._governor is not None:
            # Tell the peer which residency classes the governor accounts
            # (translated into the peer's namespace; classes the peer
            # cannot see are dropped, not guessed), so even a consumer
            # without the detail contract reports honest footprints.
            devices = [
                child_device
                for device in sorted(self._governor.status())
                if (child_device := self._relay.to_child_residency(device)) is not None
            ]
            with contextlib.suppress(WorkerDied):
                await self.send({"type": "memoryDevices", "devices": devices}, [])
        return header

    async def rebind(self) -> None:
        """Rebind admitted invocations to this session's new physical stream."""
        if not self._resumable or not self._alive:
            raise RuntimeError(f"{self._role} '{self._pack}' cannot rebind")
        rebind_future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._rebind_future = rebind_future
        entries = [
            {
                **key.to_wire(),
                "lastEventSeq": self._last_event_sequences.get(invocation_id, 0),
                "cancelled": invocation_id in self._cancelled_invocations,
            }
            for invocation_id, key in self._invocation_keys.items()
        ]
        try:
            await self.send(
                {
                    "type": "rebind",
                    "ownerEpoch": self._owner_epoch,
                    "invocations": entries,
                },
                [],
            )
            response = await rebind_future
        finally:
            self._rebind_future = None
        response_epoch = response.get("ownerEpoch")
        if type(response_epoch) is not int or response_epoch != self._owner_epoch:
            raise BoundaryError("rebind result owner epoch is stale")
        raw_statuses = response.get("invocations")
        if not isinstance(raw_statuses, list):
            raise BoundaryError("rebind result invocations must be a list")
        expected_keys = set(self._invocation_keys.values())
        seen: set[InvocationKey] = set()
        for raw in cast("list[object]", raw_statuses):
            if not isinstance(raw, Mapping):
                raise BoundaryError("rebind result invocation must be an object")
            status_entry = cast("Mapping[str, object]", raw)
            if set(status_entry) != {"jobRef", "attemptId", "invocationId", "status"}:
                raise BoundaryError("rebind result invocation has invalid fields")
            key = InvocationKey.from_header(status_entry)
            status = status_entry.get("status")
            if status not in ("running", "completed", "missing", "cancelled") or key in seen:
                raise BoundaryError("rebind result invocation state is malformed")
            seen.add(key)
            expected = self._invocation_keys.get(key.invocation_id)
            if expected != key:
                raise BoundaryError("rebind result names an unknown invocation")
            if status in ("missing", "cancelled"):
                pending_future = self._pending.pop(key.invocation_id, None)
                acknowledgement = self._result_acknowledgements.get(key.invocation_id)
                if status == "missing" and key.invocation_id in self._received_results:
                    if acknowledgement is not None and not acknowledgement.done():
                        acknowledgement.set_result(None)
                elif pending_future is not None and not pending_future.done():
                    if status == "cancelled":
                        pending_future.cancel()
                    else:
                        pending_future.set_exception(WorkerDied())
                lease = self._leases.get(key.invocation_id)
                if lease is not None:
                    lease[1].set()
                self._lease_decisions.pop(key.invocation_id, None)
                self._forget_invocation(key.invocation_id)
        if seen != expected_keys:
            raise BoundaryError("rebind result omitted invocation state")
        outbox, self._resume_outbox = self._resume_outbox, []
        self._rebinding = False
        for header, blobs in outbox:
            await self.send(header, blobs)

    def _forget_invocation(self, invocation_id: str) -> None:
        self._release_asset_pin(invocation_id)
        self._invocation_keys.pop(invocation_id, None)
        self._last_event_sequences.pop(invocation_id, None)
        self._cancelled_invocations.discard(invocation_id)
        self._accepted_invocations.pop(invocation_id, None)
        self._received_results.discard(invocation_id)
        self._result_ack_ready.discard(invocation_id)
        self._result_acknowledgements.pop(invocation_id, None)
        task = self._result_ack_tasks.pop(invocation_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._event_sinks.pop(invocation_id, None)

    def _release_asset_pin(self, invocation_id: str) -> None:
        owner = self._received_asset_pins.pop(invocation_id, None)
        if owner is not None and self._blob_store is not None:
            self._blob_store.unpin(owner)

    async def _confirm_result_acknowledgement(self, key: InvocationKey) -> None:
        acknowledgement = self._result_acknowledgements.get(key.invocation_id)
        if acknowledgement is None:
            raise BoundaryError("result acknowledgement has no invocation state")
        self._result_ack_ready.add(key.invocation_id)
        with contextlib.suppress(WorkerDied):
            await self.send({"type": "resultAck", **key.to_wire()}, [])
        await asyncio.shield(acknowledgement)

    async def _cancel_invocation(self, key: InvocationKey) -> None:
        self._release_asset_pin(key.invocation_id)
        if self._resumable:
            self._cancelled_invocations.add(key.invocation_id)
        else:
            self._pending.pop(key.invocation_id, None)
        with contextlib.suppress(Exception):
            await self.send({"type": "cancel", **key.to_wire()}, [])

    def _resend_result_acknowledgement(self, key: InvocationKey) -> None:
        current = self._result_ack_tasks.get(key.invocation_id)
        if current is not None and not current.done():
            return

        async def resend() -> None:
            try:
                await self.send({"type": "resultAck", **key.to_wire()}, [])
            except WorkerDied:
                pass

        task = asyncio.create_task(resend())
        self._result_ack_tasks[key.invocation_id] = task
        task.add_done_callback(_retrieve_future_exception)

    def _apply_measured(self, header: Mapping[str, Any]) -> None:
        """Land a frame's optional ``measured`` mapping in the telemetry
        store, translated into the parent's device namespace. Tolerant by
        contract: a remote peer's malformed entry is dropped, never a dead
        session - measurements are observability, not protocol."""
        if self._telemetry is None:
            return
        raw = header.get("measured")
        if not isinstance(raw, Mapping):
            return
        parsed: dict[str, MeasuredMemory] = {}
        for device, body in cast("Mapping[Any, Any]", raw).items():
            if not isinstance(device, str) or not device:
                continue
            measured = MeasuredMemory.from_wire(body)
            if measured is None:
                continue
            key = self._device_map.residency(device) if self._device_map is not None else device
            parsed[key] = measured
        self._telemetry.update(self, parsed)

    def _wire_relay(self, hello: Mapping[str, Any]) -> None:
        consumers_wire = hello.get("memoryConsumers")
        if not isinstance(consumers_wire, Mapping):
            return
        declared: dict[str, bool] = {}
        for name, info in cast("Mapping[object, object]", consumers_wire).items():
            if not isinstance(name, str) or not name or not isinstance(info, Mapping):
                return
            declared[name] = cast("Mapping[str, Any]", info).get("fullRelease") is True
        self._memory_consumers = declared
        self._relay = MemoryRelay(
            lambda header: self.send(header, []),
            self._device_map,
            release_guard=self._release_guard,
        )
        for name, info in cast("Mapping[str, Any]", consumers_wire).items():
            detailed = isinstance(info, Mapping) and bool(
                cast("Mapping[str, Any]", info).get("details")
            )
            proxy = self._relay.proxy(str(name), detailed=detailed)
            self._relay_proxies.append(proxy)
            if self._governor is not None:
                self._governor.register_shedder(
                    proxy,
                    priority=self._consumer_priority,
                    name=f"{self._pack}:{name}",
                )

    async def prepare(self, node_types: Sequence[str]) -> None:
        schemas = self.schemas
        missing = [t for t in node_types if t not in schemas]
        if missing:
            raise KeyError(f"worker has no implementation for: {', '.join(missing)}")

    async def materialize_sampler_registry(
        self, key: str
    ) -> tuple[tuple[str, tuple[KeyedContribution, ...]], ...]:
        """Compatibility alias for materialize_inference_generation."""
        return await self.materialize_inference_generation(key)

    async def convert_legacy_checkpoint(
        self, path: Path, logical_name: str
    ) -> tuple[str, str | None] | None:
        """Ask a capable peer to publish the deterministic converted sidecar."""
        if not self._convert_legacy_checkpoint:
            return None
        if not self._alive:
            raise WorkerDied()
        request_id = f"legacy-{self._conversion_request_index}"
        self._conversion_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._conversion_pending[request_id] = future
        try:
            await self.send(
                {
                    "type": "convertLegacyCheckpoint",
                    "requestId": request_id,
                    "sourcePath": str(path),
                    "logicalName": logical_name,
                },
                [],
            )
            reply = await future
        finally:
            self._conversion_pending.pop(request_id, None)
        status = reply.get("status")
        error = reply.get("error")
        if status == "success":
            return "success", None
        if status == "refused" and isinstance(error, str) and error:
            return "refused", error
        if status == "error" and isinstance(error, str) and error:
            raise RuntimeError(error)
        raise RuntimeError("legacy checkpoint conversion result is malformed")

    async def materialize_inference_generation(
        self, key: str
    ) -> tuple[tuple[str, tuple[KeyedContribution, ...]], ...]:
        """Ask this worker to import and validate every inference surface."""
        if not self._alive:
            raise WorkerDied()
        request_id = f"sampler-{self._sampler_request_index}"
        self._sampler_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._sampler_pending[request_id] = future
        try:
            await self.send(
                {
                    "type": "materializeSamplerRegistry",
                    "requestId": request_id,
                    "catalogKey": key,
                },
                [],
            )
            reply = await future
        finally:
            self._sampler_pending.pop(request_id, None)
        error = reply.get("error")
        if isinstance(error, str) and error:
            raise RuntimeError(error)
        raw_extensions = reply.get("extensions")
        if not isinstance(raw_extensions, list):
            raise RuntimeError("sampler registry result has malformed extensions")
        extensions: list[tuple[str, tuple[KeyedContribution, ...]]] = []
        for raw_extension in cast("list[object]", raw_extensions):
            if not isinstance(raw_extension, Mapping):
                raise RuntimeError("sampler registry result has malformed extension")
            extension = cast("Mapping[str, object]", raw_extension)
            extension_id = extension.get("id")
            raw_samplers = extension.get("contributions", extension.get("samplers"))
            if not isinstance(extension_id, str) or not isinstance(raw_samplers, list):
                raise RuntimeError("sampler registry result has malformed extension fields")
            samplers: list[KeyedContribution] = []
            for raw_sampler in cast("list[object]", raw_samplers):
                if not isinstance(raw_sampler, Mapping):
                    raise RuntimeError("sampler registry result has malformed sampler")
                sampler = cast("Mapping[str, object]", raw_sampler)
                aliases = sampler.get("aliases")
                metadata = sampler.get("behaviorMetadata")
                if not isinstance(aliases, list) or not isinstance(metadata, list):
                    raise RuntimeError("sampler registry result has malformed declaration")
                metadata_pairs: list[tuple[str, str | int | bool | None]] = []
                for raw_pair in cast("list[object]", metadata):
                    if not isinstance(raw_pair, list):
                        raise RuntimeError("sampler registry result has malformed metadata")
                    pair = cast("list[object]", raw_pair)
                    if len(pair) != 2 or not isinstance(pair[0], str):
                        raise RuntimeError("sampler registry result has malformed metadata")
                    value = pair[1]
                    if value is not None and type(value) not in (str, int, bool):
                        raise RuntimeError("sampler registry result metadata is not RPC-clean")
                    metadata_pairs.append((pair[0], cast("str | int | bool | None", value)))
                samplers.append(
                    KeyedContribution(
                        surface_id=str(sampler.get("surfaceId", "")),
                        id=str(sampler.get("id", "")),
                        aliases=tuple(str(alias) for alias in cast("list[object]", aliases)),
                        behavior_metadata=tuple(metadata_pairs),
                    )
                )
            extensions.append((extension_id, tuple(samplers)))
        return tuple(extensions)

    async def release_inference_generation(self, key: str) -> None:
        """Explicitly release an unpinned worker-local generation."""
        request_id = f"sampler-{self._sampler_request_index}"
        self._sampler_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._sampler_pending[request_id] = future
        try:
            await self.send(
                {
                    "type": "releaseInferenceGeneration",
                    "requestId": request_id,
                    "catalogKey": key,
                },
                [],
            )
            reply = await future
        finally:
            self._sampler_pending.pop(request_id, None)
        error = reply.get("error")
        if isinstance(error, str) and error:
            raise RuntimeError(error)

    async def call_pack_route(
        self, route: PackRoute, data: Mapping[str, object]
    ) -> dict[str, object]:
        if not self._alive:
            raise WorkerDied()
        request_id = f"route-{self._route_request_index}"
        self._route_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._route_pending[request_id] = future
        try:
            await self.send(
                {
                    "type": "packRoute",
                    "requestId": request_id,
                    "route": route.to_wire(),
                    "data": route.request.validate(data),
                },
                [],
            )
            reply = await future
            if reply.get("error"):
                raise RuntimeError("pack-route-failed")
            return route.response.validate(reply.get("data"))
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.send({"type": "cancelPackRoute", "requestId": request_id}, [])
            raise
        finally:
            self._route_pending.pop(request_id, None)

    async def compile_graph(
        self,
        generation_key: str,
        graph: Mapping[str, Any],
        targets: Sequence[str],
    ) -> dict[str, Any]:
        """Send one generation-keyed graph compile request to the peer."""
        if not self._alive:
            raise WorkerDied()
        request_id = f"compile-{self._graph_compile_request_index}"
        self._graph_compile_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._graph_compile_pending[request_id] = future
        try:
            await self.send(
                {
                    "type": GRAPH_COMPILE_REQUEST_TYPE,
                    "requestId": request_id,
                    "generationKey": generation_key,
                    "graph": graph,
                    "targets": list(targets),
                },
                [],
            )
            return await future
        except asyncio.CancelledError:
            if request_id in self._graph_compile_pending:
                with contextlib.suppress(Exception):
                    await self.send(
                        {"type": GRAPH_COMPILE_CANCEL_TYPE, "requestId": request_id}, []
                    )
            raise
        finally:
            self._graph_compile_pending.pop(request_id, None)

    async def query_assets(self, digests: Sequence[str]) -> dict[str, Any]:
        """Ask the peer which digests its asset store already resolves.
        Returns the raw assetQueryResult header (held/missing/error)."""
        if not self._alive:
            raise WorkerDied()
        request_id = f"assets-{self._staging_request_index}"
        self._staging_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._staging_pending[request_id] = future
        try:
            await self.send(
                {"type": "assetQuery", "requestId": request_id, "digests": list(digests)},
                [],
            )
            return await future
        finally:
            self._staging_pending.pop(request_id, None)

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        """Ask the peer to run the lazy choice provider for ``choice_id``
        and return its values. Exactly one provider invocation per call -
        no caching on either side. Raises WorkerDied when the peer is gone,
        ValueError when the provider failed or returned an invalid result,
        and RuntimeError on a malformed reply frame. Cancellation (e.g. the
        server's fetch timeout) leaves the peer's invocation running; its
        late reply finds no pending future and is discarded."""
        if choice_id not in self._lazy_choice_ids:
            raise ValueError(f"unknown lazy choice list {choice_id!r}")
        if not self._alive:
            raise WorkerDied()
        request_id = f"choices-{self._choices_request_index}"
        self._choices_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._choices_pending[request_id] = future
        try:
            await self.send(
                {"type": "fetchChoices", "requestId": request_id, "choiceId": choice_id},
                [],
            )
            header = await future
        finally:
            self._choices_pending.pop(request_id, None)
        error = header.get("error")
        if error is not None:
            raise ValueError(
                f"{self._role} '{self._pack}' lazy choice {choice_id!r} failed: {error}"
            )
        values = header.get("values")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not all(isinstance(v, str) and v for v in cast("Sequence[Any]", values))
        ):
            raise RuntimeError(
                f"{self._role} '{self._pack}' sent a malformed choicesResult for {choice_id!r}"
            )
        return tuple(cast("Sequence[str]", values))

    async def stage_assets(
        self,
        assets: Sequence[StageAsset],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Ask the peer to fetch the named assets into its vault from the
        offered HTTP(S) sources. Returns the raw stageAssetsResult header
        (staged/failed/error); ``on_event`` observes the peer's per-digest
        stageEvent progress frames while the request is pending.
        Cancellation sends cancelStage so the peer aborts its download and
        rolls the partial file back to nothing."""
        if not self._alive:
            raise WorkerDied()
        request_id = f"stage-{self._staging_request_index}"
        self._staging_request_index += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._staging_pending[request_id] = future
        if on_event is not None:
            self._stage_event_sinks[request_id] = on_event
        try:
            await self.send(
                {
                    "type": "stageAssets",
                    "requestId": request_id,
                    "assets": [asset.to_wire() for asset in assets],
                },
                [],
            )
            return await future
        except asyncio.CancelledError:
            if request_id in self._staging_pending:
                with contextlib.suppress(Exception):
                    await self.send({"type": "cancelStage", "requestId": request_id}, [])
            raise
        finally:
            self._staging_pending.pop(request_id, None)
            self._stage_event_sinks.pop(request_id, None)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        if not self._alive:
            return self._dead_result(invocation)
        key = InvocationKey(
            invocation.job_ref or invocation.invocation_id,
            invocation.attempt_id,
            invocation.invocation_id,
        )
        if self._resumable:
            if invocation.invocation_id in self._invocation_keys:
                return InvocationResult(
                    error=NodeError(
                        invocation.node_id,
                        invocation.node_type,
                        "duplicate resumable invocation identity",
                    )
                )
            self._invocation_keys[invocation.invocation_id] = key
            self._last_event_sequences[invocation.invocation_id] = 0
            self._accepted_invocations[invocation.invocation_id] = asyncio.Event()
            acknowledgement: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            acknowledgement.add_done_callback(_retrieve_future_exception)
            self._result_acknowledgements[invocation.invocation_id] = acknowledgement
        future: asyncio.Future[tuple[dict[str, Any], list[bytes]]] = (
            asyncio.get_running_loop().create_future()
        )
        if self._resumable:
            future.add_done_callback(_retrieve_future_exception)
        self._pending[invocation.invocation_id] = future
        if on_event is not None:
            self._event_sinks[invocation.invocation_id] = on_event
        started_at_ns = time.time_ns()
        started = time.perf_counter()
        try:
            try:
                input_stats = await self._send_invocation(invocation)
            except WorkerDied:
                if not self._resumable:
                    raise
                input_stats = {}
            reply_header, reply_blobs = await asyncio.shield(future)
            lease = self._leases.get(invocation.invocation_id)
            if lease is not None:
                # The result frame precedes memoryRelease on the wire. Invocation
                # completion includes unwinding its parent-held reservation.
                lease_task = lease[0]
                try:
                    await asyncio.shield(lease_task)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if (
                        current is None
                        or current.cancelling()
                        or self._alive
                        or not lease_task.cancelled()
                    ):
                        raise
        except BoundaryError as exc:
            self._pending.pop(invocation.invocation_id, None)
            self._forget_invocation(invocation.invocation_id)
            return InvocationResult(
                error=NodeError(invocation.node_id, invocation.node_type, str(exc))
            )
        except WorkerDied:
            self._pending.pop(invocation.invocation_id, None)
            self._forget_invocation(invocation.invocation_id)
            return self._dead_result(invocation)
        except asyncio.CancelledError:
            await self._cancel_invocation(key)
            raise
        finally:
            self._event_sinks.pop(invocation.invocation_id, None)
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        execute_ms = float(reply_header.get("executeMs", 0.0))
        error_wire = reply_header.get("error")
        if error_wire is not None:
            try:
                if self._resumable:
                    with contextlib.suppress(BoundaryError, WorkerDied):
                        await self._confirm_result_acknowledgement(key)
                else:
                    with contextlib.suppress(WorkerDied):
                        await self.send({"type": "resultAck", **key.to_wire()}, [])
            except asyncio.CancelledError:
                await self._cancel_invocation(key)
                raise
            self._forget_invocation(invocation.invocation_id)
            self._emit(invocation, input_stats, {}, execute_ms, round_trip_ms)
            return InvocationResult(error=error_from_wire(error_wire))

        consumed: list[str] = []
        try:
            outputs, output_stats = decode_result_outputs(
                self._codec, reply_header, reply_blobs, consumed
            )
            if self._device_map is not None:
                outputs = {name: self._device_map.value(value) for name, value in outputs.items()}
            # Handoff pins (DESIGN 3.10): from the moment this side can see the
            # result's resource references until the engine pins them at
            # produced-assignment, a concurrent ram release must not treat
            # them as unreferenced. The awaits below are the only true
            # suspension points on that path (the return itself resumes the
            # caller synchronously), so pin around them and ack the peer -
            # the resultAck is what lets it lift its own in-flight hold.
            refs: list[str] = []
            for value in outputs.values():
                # Tree traversal: resource stubs inside list outputs get the
                # same handoff pin as top-level ones (DESIGN 3.13).
                refs.extend(value_resource_ids(value))
            handoff: list[str] = []
            if self._release_guard is not None:
                handoff = [rid for rid in refs if self._release_guard.pins.pin(rid)]
            try:
                adoption = asyncio.create_task(
                    asyncio.to_thread(
                        self._produced_assets.capture,
                        outputs,
                        self._blob_store,
                        invocation.inputs,
                    )
                )
                self._asset_adoptions.add(adoption)
                adoption.add_done_callback(self._asset_adoptions.discard)
                await await_file_operation(adoption)
                if consumed:
                    with contextlib.suppress(WorkerDied):
                        await self.send({"type": "shmAck", "segments": consumed}, [])
                if self._resumable:
                    await self._confirm_result_acknowledgement(key)
                else:
                    # Peer cleanup cannot invalidate an already decoded and adopted result.
                    with contextlib.suppress(WorkerDied):
                        await self.send({"type": "resultAck", **key.to_wire()}, [])
            finally:
                if self._release_guard is not None:
                    for rid in handoff:
                        self._release_guard.pins.unpin(rid)
        except asyncio.CancelledError:
            if consumed:
                with contextlib.suppress(WorkerDied):
                    await self.send({"type": "shmAck", "segments": consumed}, [])
            if not self._resumable:
                with contextlib.suppress(WorkerDied):
                    await self.send({"type": "resultAck", **key.to_wire()}, [])
            await self._cancel_invocation(key)
            raise
        except (AssetError, BoundaryError, OSError, WorkerDied) as exc:
            with contextlib.suppress(BoundaryError, WorkerDied):
                if consumed:
                    await self.send({"type": "shmAck", "segments": consumed}, [])
                if self._resumable:
                    await self._confirm_result_acknowledgement(key)
                else:
                    await self.send({"type": "resultAck", **key.to_wire()}, [])
            self._forget_invocation(invocation.invocation_id)
            return InvocationResult(
                error=NodeError(invocation.node_id, invocation.node_type, str(exc))
            )
        finally:
            self._release_asset_pin(invocation.invocation_id)
        self._forget_invocation(invocation.invocation_id)
        self._emit(invocation, input_stats, output_stats, execute_ms, round_trip_ms)
        try:
            candidates = decode_result_artifact_candidates(reply_header)
            if candidates and self._artifact_authority is None:
                raise BoundaryError("saved artifact candidates have no local host authority")
            artifacts = (
                await asyncio.to_thread(
                    self._artifact_authority,
                    candidates,
                    invocation.node_id,
                    started_at_ns,
                )
                if candidates and self._artifact_authority is not None
                else ()
            )
        except (AssetError, BoundaryError, OSError) as exc:
            return InvocationResult(
                error=NodeError(invocation.node_id, invocation.node_type, str(exc))
            )
        return InvocationResult(outputs=outputs, artifacts=artifacts)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        del on_event
        if not self._lazy_status:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: worker does not advertise lazy status",
                )
            )
        if not self._alive:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    f"{self._role} '{self._pack}' is not running",
                )
            )
        ordinary = Invocation(
            invocation_id=invocation.request_id,
            node_id=invocation.node_id,
            node_type=invocation.node_type,
            inputs=invocation.available_inputs,
            effective_schema=invocation.effective_schema,
            executor=invocation.executor,
            arm=invocation.arm,
            expected_execution_identity=invocation.expected_execution_identity,
            fp8_matmul=invocation.fp8_matmul,
            diffusion_dtype=invocation.diffusion_dtype,
            text_dtype=invocation.text_dtype,
            vae_dtype=invocation.vae_dtype,
            attention_policy=invocation.attention_policy,
            attention_route_token=invocation.attention_route_token,
            extension_snapshot_digest=invocation.extension_snapshot_digest,
        )
        future: asyncio.Future[tuple[dict[str, Any], list[bytes]]] = (
            asyncio.get_running_loop().create_future()
        )
        if invocation.request_id in self._lazy_pending:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: duplicate request id",
                )
            )
        self._lazy_pending[invocation.request_id] = future
        try:
            await self._send_invocation(
                ordinary,
                {
                    "type": "checkLazyStatus",
                    "requestId": invocation.request_id,
                    "connectedUndemandedInputs": list(invocation.connected_undemanded_inputs),
                },
            )
            reply, reply_blobs = await future
            if reply_blobs:
                raise BoundaryError("lazy status result must not contain blobs")
            error_wire = reply.get("error")
            if error_wire is not None:
                if not isinstance(error_wire, Mapping):
                    raise BoundaryError("lazy status error must be an object")
                return LazyStatusResult(
                    error=error_from_wire(cast("Mapping[str, Any]", error_wire))
                )
            raw = reply.get("requestedInputs")
            if not isinstance(raw, list):
                raise BoundaryError("lazy status result requestedInputs must be a list")
            return LazyStatusResult(requested_inputs=tuple(cast("list[object]", raw)))
        except WorkerDied:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    f"{self._role} '{self._pack}' is not running",
                )
            )
        except BoundaryError as exc:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    f"lazy-protocol-malformed: {exc}",
                )
            )
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.send(
                    {"type": "cancelLazyStatus", "requestId": invocation.request_id}, []
                )
            raise
        finally:
            self._lazy_pending.pop(invocation.request_id, None)

    async def close(self) -> None:
        """End the conversation and unwind everything session-owned: the
        read loop, relay proxies, leases (their reservations unwind through
        their context managers), pending invocations, unacknowledged shm
        segments, and the stream itself."""
        self._closing = True
        self._alive = False
        self._workgroup.fail()
        # The lifetime token dies with the session (its contract): a
        # resolver that forgot to pair it with `alive` must still never
        # match a dead lifetime.
        self._instance_token = None
        if self._telemetry is not None:
            # The peer's measurements die with it: a closed worker must
            # never leave a stale "free" number in the parent's status.
            self._telemetry.clear(self)
        if (
            self._relay is not None
            and self._reader_task is not None
            and self._reader_task is not asyncio.current_task()
        ):
            await self._relay.drain_releases()
        if self._relay is not None:
            self._relay.close()
            self._closed_relays.append(self._relay)
        for proxy in self._relay_proxies:
            if self._governor is not None:
                self._governor.unregister_shedder(proxy)
        self._relay_proxies.clear()
        self._relay = None
        self._memory_consumers = None
        if self._resume_timeout_task is not None:
            self._resume_timeout_task.cancel()
            if self._resume_timeout_task is not asyncio.current_task():
                with contextlib.suppress(asyncio.CancelledError):
                    await self._resume_timeout_task
            self._resume_timeout_task = None
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
        if self._reader_task is not None:
            reader_task, self._reader_task = self._reader_task, None
            if reader_task is not asyncio.current_task():
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader_task
        await self._unwind_leases()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._pending.clear()
        if self._asset_adoptions:
            await await_file_operation(
                asyncio.gather(*self._asset_adoptions, return_exceptions=True)
            )
        # Received source pins belong to invoke(), even before adoption starts after EOF.
        for future in self._lazy_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._lazy_pending.clear()
        for future in self._sampler_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._sampler_pending.clear()
        for future in self._conversion_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._conversion_pending.clear()
        for future in self._graph_compile_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._graph_compile_pending.clear()
        for future in self._route_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._route_pending.clear()
        for future in self._staging_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._staging_pending.clear()
        for future in self._choices_pending.values():
            if not future.done():
                future.set_exception(WorkerDied())
        self._choices_pending.clear()
        self._stage_event_sinks.clear()
        self._event_sinks.clear()
        self._invocation_keys.clear()
        self._last_event_sequences.clear()
        self._cancelled_invocations.clear()
        self._accepted_invocations.clear()
        self._received_results.clear()
        self._result_ack_ready.clear()
        for acknowledgement in self._result_acknowledgements.values():
            if not acknowledgement.done():
                acknowledgement.set_exception(WorkerDied())
        self._result_acknowledgements.clear()
        result_ack_tasks = list(self._result_ack_tasks.values())
        for task in result_ack_tasks:
            task.cancel()
        self._result_ack_tasks.clear()
        if result_ack_tasks:
            await asyncio.gather(*result_ack_tasks, return_exceptions=True)
        self._resume_outbox.clear()
        self._lease_decisions.clear()
        if self._blob_transfer is not None:
            self._blob_transfer.close()
            self._blob_transfer = None
        # Segments the peer never acknowledged die with us (hazard H14).
        for segment in self._sent_segments.values():
            release_segment(segment)
        self._sent_segments.clear()
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
        self._reader = None

    def process_died(self) -> None:
        if self._relay is not None:
            self._relay.process_died()
        for relay in self._closed_relays:
            relay.process_died()
        self._closed_relays.clear()

    async def detach_transport(self) -> None:
        """Drop the physical stream while preserving resumable invocation state."""
        if not self._resumable or not self._invocation_keys:
            await self.close()
            return
        reader_task = self._reader_task
        if self._writer is not None:
            self._writer.transport.abort()
        if reader_task is not None and reader_task is not asyncio.current_task():
            await reader_task

    def _dead_result(self, invocation: Invocation) -> InvocationResult:
        detail = self._death_detail() if self._death_detail is not None else ""
        return InvocationResult(
            error=NodeError(
                node_id=invocation.node_id,
                node_type=invocation.node_type,
                message=f"{self._role} '{self._pack}' is not running{detail}",
            )
        )

    async def send(
        self,
        header: Mapping[str, object],
        blobs: Sequence[bytes],
        segments: Sequence[SharedMemory] = (),
    ) -> None:
        body = dict(header)
        kind = body.get("type")
        resumable_control = kind in ("memoryGrant", "memoryDeny")
        if resumable_control:
            key = self._invocation_keys.get(str(body.get("requestId")))
            if key is not None:
                body.update(key.to_wire())
        if resumable_control and (self._writer is None or self._rebinding):
            if self._resumable and not segments:
                self._resume_outbox.append((body, list(blobs)))
                return
            for segment in segments:
                release_segment(segment)
            raise WorkerDied()
        async with self._send_lock:
            if self._writer is None or (segments and not self._alive):
                for segment in segments:
                    release_segment(segment)
                raise WorkerDied()
            try:
                await self._send_unlocked(body, blobs, segments)
            except WorkerDied:
                if self._resumable and resumable_control and not segments:
                    self._resume_outbox.append((body, list(blobs)))
                    return
                raise
            except BoundaryError:
                for segment in segments:
                    retained = self._sent_segments.pop(segment.name, None)
                    if retained is not None:
                        release_segment(retained)
                raise

    async def _send_unlocked(
        self,
        header: Mapping[str, object],
        blobs: Sequence[bytes],
        segments: Sequence[SharedMemory],
    ) -> None:
        if self._writer is None:
            raise WorkerDied()
        try:
            for segment in segments:
                # Hold the handle open until the receiver acknowledges: on
                # Windows a segment is freed when its last handle closes, so
                # closing at send time would destroy it before the receiver
                # attaches (POSIX only needs the name for unlink, but the
                # lifetime rule must be the portable one).
                self._sent_segments[segment.name] = segment
            await write_frame(self._writer, header, blobs)
        except ConnectionError as exc:
            raise WorkerDied() from exc

    async def _send_invocation(
        self,
        invocation: Invocation,
        header_updates: Mapping[str, object] | None = None,
    ) -> dict[str, TransferStat]:
        """Serialize CAS mutation, invocation encoding, and the first frame write."""
        pending: list[tuple[str, bytes]] = []
        async with self._send_lock:
            if not self._alive or self._writer is None:
                raise WorkerDied()
            checkpoint = self._codec.conversation_checkpoint()
            segments: list[SharedMemory] = []
            try:
                header, blobs, segments, stats = encode_invocation(self._codec, invocation)
                pending = self._codec.take_pending_store_blobs()
                if header_updates is not None:
                    header.update(header_updates)
                if not pending:
                    await self._send_unlocked(header, blobs, segments)
            except BoundaryError:
                # Boundary refusal occurs before bytes or an await make CAS state peer-visible.
                self._codec.restore_conversation(checkpoint)
                for segment in segments:
                    held = self._sent_segments.pop(segment.name, segment)
                    release_segment(held)
                raise
        if not pending:
            return stats
        # persistentCas references must resolve before the frame that names
        # them: land the missing blobs first, then send the invoke. The send
        # lock is NOT held across the transfer, so result frames and the
        # peer's own blob queries keep flowing. Persistent encoding never
        # touches the conversation cas index, so releasing the lock between
        # encode and send leaks no transactional state.
        assert self._blob_transfer is not None  # pending implies negotiation
        try:
            moved = await self._blob_transfer.ensure_peer_holds(pending)
        except BaseException:
            # The invoke frame was never sent, so its segments were never
            # registered for shmAck release: free them here.
            for segment in segments:
                release_segment(segment)
            raise
        await self.send(header, blobs, segments)
        return attribute_moved(stats, cast("Mapping[str, Any]", header["inputs"]), moved)

    def start_keepalive(self, interval: float, ttl: float) -> None:
        """Send a ``heartbeat`` frame every ``interval`` seconds and treat
        the peer as dead when nothing at all is received for ``ttl``
        seconds: the transport is aborted, which unwinds the read loop
        through its normal conservative teardown (pending invocations fail
        as WorkerDied, footprints collapse, ``alive`` goes False). Renewal
        counts every received frame, not just acks, so a peer busy
        streaming results never needs its acks to arrive on time."""
        if self._keepalive_task is not None or not self._alive:
            return
        self._last_received = time.monotonic()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(interval, ttl))

    async def _keepalive_loop(self, interval: float, ttl: float) -> None:
        while self._alive:
            await asyncio.sleep(interval)
            if not self._alive:
                return
            if time.monotonic() - self._last_received > ttl:
                # A silent peer past the lease: abort rather than close, so
                # a half-open connection unblocks the read loop NOW instead
                # of when the OS notices.
                if self._writer is not None:
                    self._writer.transport.abort()
                return
            with contextlib.suppress(Exception):
                # A failed send means the stream is dying; the read loop
                # notices and unwinds - nothing for this task to add.
                await self.send({"type": "heartbeat"}, [])

    async def _read_loop(self) -> None:
        assert self._reader is not None
        transport_loss = True
        try:
            while True:
                frame = await read_frame(self._reader)
                if frame is None:
                    break
                self._last_received = time.monotonic()
                header, blobs = frame
                kind = header.get("type")
                if kind == "result":
                    if "resultAlgebra" in header:
                        raise BoundaryError("unexpected result algebra frame")
                    invocation_id = str(header.get("invocationId"))
                    key: InvocationKey | None = None
                    if self._resumable:
                        key = InvocationKey.from_header(header)
                        if self._invocation_keys.get(invocation_id) != key:
                            raise BoundaryError("result names no resumable invocation")
                    future = self._pending.get(invocation_id)
                    if future is not None and not future.done():
                        if self._blob_store is not None:
                            try:
                                digests = result_asset_digests(header, blobs)
                            except AssetError as exc:
                                raise BoundaryError(str(exc)) from exc
                            if digests:
                                owner = object()
                                self._received_asset_pins[invocation_id] = owner
                                self._blob_store.pin(owner, digests)
                        if self._resumable:
                            self._received_results.add(invocation_id)
                        self._pending.pop(invocation_id, None)
                        future.set_result((header, blobs))
                    elif self._resumable:
                        assert key is not None
                        if invocation_id not in self._received_results:
                            raise BoundaryError("replayed result was not previously received")
                        # Receipt alone cannot release the producer's hold while adoption runs.
                        if invocation_id in self._result_ack_ready:
                            self._resend_result_acknowledgement(key)
                elif kind == "lazyStatusResult":
                    future = self._lazy_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result((header, blobs))
                elif kind == "invocationEvent":
                    invocation_id = str(header.get("invocationId"))
                    if self._resumable:
                        key = InvocationKey.from_header(header)
                        sequence = header.get("eventSeq")
                        if (
                            self._invocation_keys.get(invocation_id) != key
                            or type(sequence) is not int
                            or sequence < 1
                        ):
                            raise BoundaryError("invocation event identity is malformed")
                        if sequence <= self._last_event_sequences.get(invocation_id, 0):
                            continue
                        self._last_event_sequences[invocation_id] = sequence
                    sink = self._event_sinks.get(invocation_id)
                    if sink is not None:
                        data = header.get("data")
                        event = InvocationEvent(
                            name=str(header.get("name", "")),
                            data=(
                                dict(cast("Mapping[str, object]", data))
                                if isinstance(data, Mapping)
                                else {}
                            ),
                            blob=blobs[0] if blobs else None,
                        )
                        # A misbehaving listener must not kill the read loop
                        # (which would take the whole session down with it).
                        with contextlib.suppress(Exception):
                            sink(event)
                elif kind == "schemaReloadRequest":
                    if not self._schema_reload or blobs:
                        raise BoundaryError("unnegotiated schema reload request")
                    if self._on_schema_reload is not None and self._instance_token is not None:
                        with contextlib.suppress(Exception):
                            self._on_schema_reload(self._pack, self._instance_token)
                elif kind in ("samplerRegistryResult", "inferenceReleaseResult"):
                    future = self._sampler_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind == "legacyCheckpointConversionResult":
                    future = self._conversion_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind == "packRouteResult":
                    future = self._route_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind == GRAPH_COMPILE_RESULT_TYPE:
                    future = self._graph_compile_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind in ("assetQueryResult", "stageAssetsResult"):
                    future = self._staging_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind == "choicesResult":
                    future = self._choices_pending.pop(str(header.get("requestId")), None)
                    if future is not None and not future.done():
                        future.set_result(header)
                elif kind == "stageEvent":
                    stage_sink = self._stage_event_sinks.get(str(header.get("requestId")))
                    if stage_sink is not None:
                        # A misbehaving listener must not kill the read loop.
                        with contextlib.suppress(Exception):
                            stage_sink(header)
                elif kind == "memoryReserve":
                    if self._resumable:
                        key = InvocationKey.from_header(header)
                        if (
                            str(header.get("requestId")) != key.invocation_id
                            or self._invocation_keys.get(key.invocation_id) != key
                        ):
                            raise BoundaryError("memory reservation names no invocation")
                    self._start_lease(header)
                elif kind == "memoryRelease":
                    if self._resumable:
                        key = InvocationKey.from_header(header)
                        if (
                            str(header.get("requestId")) != key.invocation_id
                            or self._invocation_keys.get(key.invocation_id) != key
                        ):
                            raise BoundaryError("memory release names no invocation")
                    lease = self._leases.get(str(header.get("requestId")))
                    if lease is not None:
                        lease[1].set()
                    self._lease_decisions.pop(str(header.get("requestId")), None)
                elif kind == "memoryReport":
                    consumers = header.get("consumers")
                    if self._relay is not None and isinstance(consumers, dict):
                        self._relay.apply_report(cast("dict[str, Any]", consumers))
                    self._apply_measured(header)
                elif kind == "memoryShedResult":
                    if self._relay is not None:
                        self._relay.on_shed_result(header)
                    self._apply_measured(header)
                elif kind in (
                    "memoryReleaseCandidates",
                    "memoryReleaseResult",
                    "memoryFreeCandidates",
                    "memoryFreeResult",
                    "memoryFreeAborted",
                ):
                    if self._relay is not None:
                        self._relay.on_release_reply(header)
                elif kind == "blobQuery":
                    if self._blob_transfer is not None:
                        await self._blob_transfer.answer_query(header)
                elif kind == "blobQueryResult":
                    if self._blob_transfer is not None:
                        self._blob_transfer.resolve_query(header)
                elif kind == "blobData":
                    if self._blob_transfer is not None:
                        await self._blob_transfer.accept_chunk(header, blobs)
                elif kind == "shmAck":
                    for name in header.get("segments", ()):
                        segment = self._sent_segments.pop(str(name), None)
                        if segment is not None:
                            release_segment(segment)
                elif kind == WORKGROUP_FRAME_TYPE:
                    self._workgroup.accept(header, blobs)
                elif kind == "heartbeatAck":
                    pass  # receipt itself already renewed the keepalive clock
                elif kind == "rebindResult":
                    future = self._rebind_future
                    if future is None or future.done():
                        raise BoundaryError("unexpected rebind result")
                    future.set_result(header)
                elif kind == "cancelAck":
                    key = InvocationKey.from_header(header)
                    if self._invocation_keys.get(key.invocation_id) != key:
                        raise BoundaryError("cancel acknowledgement names no invocation")
                    pending = self._pending.pop(key.invocation_id, None)
                    if pending is not None and not pending.done():
                        pending.cancel()
                    lease = self._leases.get(key.invocation_id)
                    if lease is not None:
                        lease[1].set()
                    self._lease_decisions.pop(key.invocation_id, None)
                    self._forget_invocation(key.invocation_id)
                elif kind == "resultAcked":
                    if not self._resumable:
                        continue
                    key = InvocationKey.from_header(header)
                    acknowledgement = self._result_acknowledgements.get(key.invocation_id)
                    if (
                        self._invocation_keys.get(key.invocation_id) != key
                        or key.invocation_id not in self._result_ack_ready
                        or acknowledgement is None
                        or acknowledgement.done()
                    ):
                        raise BoundaryError("result acknowledgement names no received result")
                    acknowledgement.set_result(None)
                    self._forget_invocation(key.invocation_id)
                elif kind == "invokeAccepted":
                    if not self._resumable:
                        continue
                    key = InvocationKey.from_header(header)
                    accepted = self._accepted_invocations.get(key.invocation_id)
                    if self._invocation_keys.get(key.invocation_id) != key or accepted is None:
                        raise BoundaryError("invocation acceptance names no invocation")
                    accepted.set()
        except WorkerDied:
            pass
        except BoundaryError:
            transport_loss = False
        finally:
            self._alive = False
            if self._resumable and transport_loss and not self._closing and self._invocation_keys:
                await self._detach_for_resume()
            else:
                await self.close()

    async def _detach_for_resume(self) -> None:
        self._workgroup.fail()
        if self._rebind_future is not None and not self._rebind_future.done():
            self._rebind_future.set_exception(WorkerDied())
        if self._telemetry is not None:
            self._telemetry.clear(self)
        if self._relay is not None:
            self._relay.close()
            self._closed_relays.append(self._relay)
        for proxy in self._relay_proxies:
            if self._governor is not None:
                self._governor.unregister_shedder(proxy)
        self._relay_proxies.clear()
        self._relay = None
        self._memory_consumers = None
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        for pending in (
            self._lazy_pending,
            self._sampler_pending,
            self._conversion_pending,
            self._graph_compile_pending,
            self._route_pending,
            self._staging_pending,
            self._choices_pending,
        ):
            for future in pending.values():
                if not future.done():
                    future.set_exception(WorkerDied())
            pending.clear()
        self._stage_event_sinks.clear()
        if self._blob_transfer is not None:
            self._blob_transfer.close(preserve_pins=True)
            self._reset_blob_transfer()
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._reader = None
        if self._resume_timeout_task is None:
            self._resume_timeout_task = asyncio.create_task(self._expire_resume())

    async def _expire_resume(self) -> None:
        try:
            await asyncio.sleep(self._resume_grace)
            await self.close()
        except asyncio.CancelledError:
            pass

    async def _unwind_leases(self) -> None:
        """Release every reservation owned by this session."""
        lease_tasks = [task for task, _ in self._leases.values()]
        for task in lease_tasks:
            task.cancel()
        if lease_tasks:
            await asyncio.gather(*lease_tasks, return_exceptions=True)
        self._leases.clear()

    def _start_lease(self, header: dict[str, Any]) -> None:
        """Service one memoryReserve frame. The lease runs as its own task so
        the read loop keeps processing results and releases while a grant
        waits on the governor - a blocked reserve must never stall the frames
        that would unblock it."""
        request_id = str(header.get("requestId"))
        decision = self._lease_decisions.get(request_id)
        if decision is not None:
            task = asyncio.create_task(self.send(decision, []))
            task.add_done_callback(_retrieve_future_exception)
            return
        if request_id in self._leases:
            return
        try:
            requests = tuple(
                ReservationRequest(
                    # The peer asks in its own device namespace; the governor
                    # accounts in the parent's.
                    residency=(
                        self._device_map.residency(str(wire["residency"]))
                        if self._device_map is not None
                        else str(wire["residency"])
                    ),
                    nbytes=int(wire["nbytes"]),
                )
                for wire in cast("list[dict[str, Any]]", header.get("requests", []))
            )
        except (KeyError, TypeError, ValueError) as exc:
            malformed_decision: dict[str, object] = {
                "type": "memoryDeny",
                "requestId": request_id,
                "message": f"malformed reservation request: {exc}",
            }
            self._lease_decisions[request_id] = malformed_decision
            task = asyncio.create_task(self.send(malformed_decision, []))
            self._leases[request_id] = (task, asyncio.Event())
            task.add_done_callback(lambda _: self._leases.pop(request_id, None))
            return
        released = asyncio.Event()
        task = asyncio.create_task(self._hold_lease(request_id, requests, released))
        self._leases[request_id] = (task, released)

    async def _hold_lease(
        self,
        request_id: str,
        requests: tuple[ReservationRequest, ...],
        released: asyncio.Event,
    ) -> None:
        try:
            if released.is_set():
                # The peer already gave up (cancellation raced ahead of this
                # task): acquiring would grab budget only to drop it.
                return
            if self._reservations is None:
                decision: dict[str, object] = {
                    "type": "memoryGrant",
                    "requestId": request_id,
                }
                self._lease_decisions[request_id] = decision
                await self.send(decision, [])
                await released.wait()
                return
            try:
                async with self._reservations.reserve(requests):
                    decision = {"type": "memoryGrant", "requestId": request_id}
                    self._lease_decisions[request_id] = decision
                    await self.send(decision, [])
                    await released.wait()
            except (BudgetExceeded, ReservationTimeout) as exc:
                decision = {
                    "type": "memoryDeny",
                    "requestId": request_id,
                    "message": str(exc),
                }
                self._lease_decisions[request_id] = decision
                await self.send(decision, [])
        except WorkerDied:
            pass  # reservation already unwound; nobody left to notify
        finally:
            self._leases.pop(request_id, None)

    def _emit(
        self,
        invocation: Invocation,
        input_stats: Mapping[str, TransferStat],
        output_stats: Mapping[str, TransferStat],
        execute_ms: float,
        round_trip_ms: float,
    ) -> None:
        if self._on_diagnostic is None:
            return

        def costs(
            stats: Mapping[str, TransferStat], types: Mapping[str, str]
        ) -> tuple[EdgeCost, ...]:
            return tuple(
                EdgeCost(
                    edge_id=edge_id,
                    type_id=types.get(edge_id, ""),
                    transport=stat.transport,
                    size_bytes=stat.size_bytes,
                    codec_ms=stat.codec_ms,
                    declared_codec=stat.declared_codec,
                    reused=stat.reused,
                    network_bytes=stat.network_bytes,
                    transfer_ms=stat.transfer_ms,
                )
                for edge_id, stat in stats.items()
            )

        input_types = {input_id: value.type_id for input_id, value in invocation.inputs.items()}
        output_types = {
            out.id: (out.type.runtime_type_id() or "")
            for out in invocation.effective_schema.outputs
        }
        self._on_diagnostic(
            BoundaryDiagnostic(
                invocation_id=invocation.invocation_id,
                node_id=invocation.node_id,
                node_type=invocation.node_type,
                pack=self._pack,
                inputs=costs(input_stats, input_types),
                outputs=costs(output_stats, output_types),
                execute_ms=execute_ms,
                round_trip_ms=round_trip_ms,
            )
        )
