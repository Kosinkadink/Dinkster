"""The native Dinkster HTTP/WS protocol (DESIGN 3.5).

This is NOT ComfyUI's v1 surface - no /prompt, no /object_info. A v1 facade,
if ever built, is a quarantined adapter package, never this one.

Surface:

- GET    /api/nodes                      schemaVersion + dinkster environment
                                         header (server version + schema wire
                                         version). The schema wire version is
                                         fixed for a release. Plus packs table
                                         (pack id ->
                                         displayName/abbr/mark/color badge
                                         presentation + omitted-when-unknown
                                         provenance: version/artifactDigest/
                                         source/publisher) + native schema
                                         wire; every node entry carries
                                         "pack" (host-attached attribution,
                                         "core" included - never
                                         self-claimed) and an opaque
                                         "signature" (interface identity for
                                         drift diagnostics; equality-compare
                                         only, never parse)
- GET    /api/diagnostics                instance-level schema diagnostics
                                         (cross-pack replacement-rule
                                         problems + classified compat
                                         translation skips); advisory,
                                         never fatal
- GET    /api/packs/{packId}/icon        raster badge bytes for packs whose
                                         table entry declares an icon; digest
                                         ETag + If-None-Match 304, immutable
                                         cache headers (bytes for a digest
                                         never change); 404 for unknown packs
                                         and packs without icons
- GET    /packs/{packId}/static/{path}   exact bytes from a pack-declared
                                         frontend asset tree; no directory
                                         listing or traversal
- GET/PUT /api/packs/{packId}/settings   declared schema plus effective values;
                                         complete PUT objects are validated and
                                         atomically persisted
- GET    /api/packs/{packId}/blueprints/{id}
                                         blueprint document bytes (JSON) for
                                         packs whose table entry advertises
                                         the descriptor ({id, name,
                                         description?, tags?, digest} inline
                                         in "blueprints"); same immutable
                                         digest/ETag contract as icons; 404
                                         for unknown packs/blueprints
- GET    /api/templates?q=&tag=&pack=&limit=&cursor=
                                         query-first paged index of pack-
                                         shipped starter workflow templates:
                                         descriptors only ({pack, id, name,
                                         description?, tags?, assets?,
                                         digest}; "assets" are pack-local
                                         [[pack.assets]] ids to join against
                                         the packs table), cursor bound to
                                         the query (mismatch -> 400)
- GET    /api/packs/{packId}/templates/{id}
                                         template document bytes (JSON);
                                         same immutable digest/ETag contract
                                         as blueprints; 404 for unknown
                                         packs/templates
- GET    /api/docs?q=&kind=&pack=&id=&limit=&cursor=
                                         paged localized node and guide
                                         descriptors; Markdown bodies and
                                         referenced media are fetched by
                                         immutable digest from pack-scoped
                                         /docs/pages/{digest} and
                                         /docs/assets/{digest} routes
- POST   /api/jobs                       submit {clientId, jobId, graph,
                                         targets, priority?, attention?,
                                         sourceDocument?, scope?, placement?}
                                         (placement maps top-level node ids
                                         to worker names; scope is
                                         selected from jobs:submit grants;
                                         sourceDocument:
                                         canonical asset digest of the
                                         workflow document the job came
                                         from - execution-opaque provenance,
                                         echoed on the job wire). The 202
                                         response carries jobRef: the
                                         server-assigned globally unique
                                         job reference (platform plan;
                                         runId is the same value under its
                                         legacy name), plus submittedBy
                                         {principalId, kind}. An active duplicate
                                         with identical content returns the
                                         same job with duplicate=true
- GET    /api/jobs/{clientId}/{jobId}    job state, node state map, result descriptors
- DELETE /api/jobs/{clientId}/{jobId}    cancel (queued or running)
- GET    /api/jobs/by-ref/{jobRef}       the same status without client affinity
- DELETE /api/jobs/by-ref/{jobRef}       the same cancel without client affinity
- GET    /api/jobs/by-ref/{jobRef}/events?after=N
                                         buffered job events after sequence N
- GET    /api/workers                    configured execution locations,
                                         routed node types, and device
                                         qualifiers
- GET    /api/values?clientId=..&jobId=..&nodeId=..&outputId=..
                                         completed-run value retrieval for
                                         frozen-view peeking: descriptor +
                                         available renditions as JSON, or
                                         browser-renderable bytes with
                                         &rendition=<advertised-cacheKey>;
                                         parameterized renditions advertise
                                         accepted selectors/defaults/limits;
                                         AUDIO supports &waveform=WxH or
                                         &window=start,duration plus &batch=N;
                                         kind/default remain revalidating
                                         compatibility aliases; &element=
                                         i[,j...] descends lists. Targets
                                         resolve from the retained job
                                         result, intermediates through the
                                         run's own cache keys; evicted ->
                                         410, never a newer value. ETag is
                                         fingerprint/cacheKey. Only a URL
                                         carrying the advertised cacheKey is
                                         immutable.
- GET    /api/events?clientId=...        additive WebSocket event stream;
                                         job events carry per-job seq
- POST   /api/auth/ws-ticket             mint a 30-second single-use browser
                                         WebSocket upgrade credential

Cross-instance memory coordination (DESIGN 3.10) - present only when the
instance runs governed (a MemoryGovernor was supplied); ungoverned
instances answer 409 so a peer's probe gets a clean "no", never a fake:

- GET    /memory/status                  admission occupancy, queue, governor
                                         budgets/footprints/consumers, measured
                                         free/total, live leases; ?details=1
                                         adds per-consumer item lists (stable
                                         IDs, display names, residency bytes,
                                         page maps with geometry)
- POST   /memory/shed                    {device, bytes} -> {freedBytes}
- POST   /memory/free                    paused-queue full release across live workers
- POST   /memory/reserve                 {device, bytes, ttlSeconds?, timeoutSeconds?}
                                         -> TTL lease (201); 507 when it can never
                                         fit, 503 when it did not fit in time
- POST   /memory/reserve/{id}/renew      {ttlSeconds?} -> extended lease; the
                                         heartbeat that keeps a lease alive
- DELETE /memory/reserve/{id}            release (204); 404 when expired/unknown
- POST   /cache/trim                     {device, bytes?, consumers?, items?} ->
                                         shed named consumers only (default:
                                         caches and everything else registered);
                                         items (stable IDs, one consumer) is how
                                         "unload this model" is expressed

Cache sharing (DESIGN 3.4) - present only when a ``cache_export`` was
supplied (a DiskCacheStore, typically). Read-only pull: a peer fetches this
instance's computed entries; nothing writes into another instance's cache.
Same trust domain as workers - manifests carry default-codec meta bytes and
blobs carry payload codec bytes, so expose these only to peers you would
hand a worker token to:

- GET    /cache/entry/{key}              entry manifest (wire.py format);
                                         404 on miss
- GET    /cache/cas/{digest}             payload bytes for a blake3 digest;
                                         400 on a malformed digest, 404 on miss

Workflow persistence - present only when a ``library`` was supplied; see
library.py for the full contract:

- POST   /api/assets                     verified client upload -> {"digest"}
- GET    /api/assets/{digest}            bytes back, immutable-cache headers
- POST   /api/library                    create scoped record
- GET    /api/library                    query-first cursor-paged browse
- GET/PATCH/DELETE /api/library/{id}     one record (scoped, optimistic
                                         concurrency on PATCH)

Persistent execution history - present only when a ``history`` store was
supplied; see history.py for the full contract:

- GET    /api/history                    terminal runs, query-first and
                                         cursor-paged (scope required)
- DELETE /api/history                    bulk clear (same filters + before=)
- GET    /api/history/{runId}            one durable run record
- DELETE /api/history/{runId}            remove one record, never bytes

Training sessions - present only when a ``training_sessions`` store was
supplied; read-only (mutation crosses the worker boundary, never HTTP);
see training_sessions.py for the full contract:

- GET /api/training/sessions/{sessionId}             record + current handle
- GET /api/training/sessions/{sessionId}/events      journal replay page
- GET /api/training/sessions/{sessionId}/checkpoints committed lineage

Asset sharing (DESIGN 3.12) - present only when an ``asset_export`` was
supplied (an AssetVault or LocalAssetLibrary, typically). Assets move by
identity: a peer that lacks a model's bytes fetches them by digest and
verifies against the digest it already knew, so these endpoints can waste a
peer's bandwidth at worst, never plant wrong bytes. The digest list is the
announcement placement can query to prefer machines where a model already
lives:

- GET    /assets                         {"digests": [...]} held here
- GET    /assets/{digest}                asset bytes, streamed from disk;
                                         400 on a malformed digest, 404 on miss

Coordination is cooperative and advisory: leases expire on their TTLs, so
a crashed peer cannot pin memory, and measured free memory remains ground
truth on the devices themselves.

Job results carry value *descriptors* (typeId, fingerprint, meta) - payload
bytes move through a dedicated transfer endpoint later, not through job
status JSON. That keeps the envelope-first value model intact on the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import ipaddress
import json
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from aiohttp import WSMsgType, web
from dinkster_assets import (
    AssetError,
    AssetIntegrityError,
    DeclaredAsset,
    is_digest,
)
from dinkster_assets.acquisition_plan import AcquisitionSource
from dinkster_assets.resolution import ResolutionStore
from dinkster_engine import (
    CompiledGraph,
    Engine,
    EngineEvent,
    EventListener,
    ExecutionArm,
    ExecutionRuntime,
    GraphCompileError,
    ProviderResolutionError,
)
from dinkster_graph import (
    Graph,
    GraphNode,
    GraphWireError,
    RegionNode,
    graph_from_wire,
    lower_selectors,
    migrate_pure_node_type_replacements,
    top_level_node_id,
)
from dinkster_memory import (
    AcceleratorMemoryPolicy,
    BudgetExceeded,
    ConsumerItem,
    Lease,
    LeaseBroker,
    MemoryGovernor,
    ReservationTimeout,
)
from dinkster_protocol import (
    AttentionPolicy,
    AttentionPolicyConfig,
    CompatGateDiagnostic,
    PackSettingsSchema,
    PreviewPolicy,
    attention_policy_config_from_wire,
    attention_policy_config_to_wire,
    canonical_extension_snapshot,
    validate_attention_policy,
    validate_preview_animation,
    validate_preview_mode,
)
from dinkster_schema import (
    PROGRESS_EVENT,
    SCHEMA_WIRE_VERSION,
    ComfyAliasRegistry,
    ComfyGroupRegistry,
    NodeSchema,
    ReplacementProblem,
    combo_choices_json_bytes,
    comfy_alias_collision_problems,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    comfy_group_collision_problems,
    comfy_group_registry_problems,
    comfy_group_registry_to_wire,
    core_logger,
    problem_to_wire,
    schema_signature,
    schema_to_wire,
    validate_name,
    validate_remote_choice_authority,
    validate_replacement_references,
)
from dinkster_values import (
    InvalidRenditionRequest,
    RenditionUnavailable,
    TypeRegistry,
    UnresolvablePayload,
    Value,
    list_children,
)

from ._federated_asset_catalog import add_federated_asset_routes
from .asset_stream import open_verified_sized, stream_verified
from .auth import (
    Authenticator,
    Principal,
    PrincipalPermissionStore,
    add_principal_routes,
    handle_ws_ticket,
    install_auth,
    principal_for,
    resolve_scope,
)
from .events import (
    BINARY_BLOB_KEY,
    DROPPABLE_KINDS,
    EventHub,
    ProgressThrottle,
    encode_binary_event,
    engine_event_to_wire,
)
from .execution_journal import ExecutionJournal, add_execution_journal_routes
from .history import HistoryStore, add_history_routes
from .library import LIBRARY_KEY, ServerLibrary, add_library_routes
from .p2p_plugin import default_p2p_settings
from .pack_settings import PackSettingsStore
from .pack_surfaces import FrontendModuleRead, PackRouteDispatch, install_pack_surfaces
from .paging import decode_cursor, encode_cursor
from .preflight import (
    asset_preflight,
    graph_asset_names,
    graph_node_types,
    graph_provider_selections,
    parse_asset_consent,
)
from .queue import TERMINAL_STATES, Job, JobGraphAdmissionError, JobQueue
from .redaction import PathRedactor
from .settings import SETTINGS_CATEGORIES, RuntimeSettings, add_settings_routes
from .training_sessions import TrainingSessionStore, add_training_routes

_log = core_logger("server")

SCHEMA_VERSION = 1
JOB_EVENT_BUFFER = 512

PlaceExecution = Callable[
    [ExecutionRuntime, Mapping[str, str]],
    ExecutionRuntime,
]
FullFree = Callable[[str], Awaitable[Sequence[dict[str, object]]]]

STATE_KEY: web.AppKey[ServerState] = web.AppKey("state")
PACK_SETTINGS_KEY: web.AppKey[PackSettingsStore] = web.AppKey("pack_settings")

# One lazy choice list: awaited per /api/choices fetch, returning the values
# to serve. The composition binds these to worker sessions; the server owns
# the fetch timeout so a hung provider yields a deterministic error before
# the frontend's own request deadline races it.
LazyChoiceFetcher = Callable[[], Awaitable[Sequence[str]]]

LAZY_CHOICE_TIMEOUT_SECONDS = 4.0


class ChoiceOwnerGone(Exception):
    """The worker owning a lazy choice list is dead or disconnected."""


def _validated_lazy_choices(
    lazy_choices: Mapping[str, LazyChoiceFetcher] | None,
    *,
    subject: str,
) -> dict[str, LazyChoiceFetcher]:
    validated: dict[str, LazyChoiceFetcher] = {}
    for choice_id, fetcher in cast("Mapping[object, object]", lazy_choices or {}).items():
        if not isinstance(choice_id, str):
            raise ValueError(f"{subject} lazy choice id must be a string")
        problem = validate_name(choice_id)
        if problem is not None:
            raise ValueError(f"{subject} lazy choice id {choice_id!r} {problem}")
        if not callable(fetcher):
            raise ValueError(f"{subject} lazy choice {choice_id!r} must be a callable fetcher")
        validated[choice_id] = cast("LazyChoiceFetcher", fetcher)
    return validated


def _merged_choice_authority(
    choices: Mapping[str, Sequence[str]],
    lazy_choices: Mapping[str, LazyChoiceFetcher],
) -> dict[str, Sequence[str]]:
    """The choice ids one owner registered, static and lazy together, in
    the shape validate_remote_choice_authority checks (it looks at keys
    only; lazy values are unknown until fetched)."""
    return {**choices, **dict.fromkeys(lazy_choices, ())}


def _validated_authority_owners(
    identifiers: Sequence[str],
    owners: Mapping[str, object] | None,
    *,
    subject: str,
    fallback: object,
    retained: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = set(identifiers)
    if owners is None:
        return {
            identifier: (retained or {}).get(identifier, fallback) for identifier in identifiers
        }
    supplied = dict(owners)
    if missing := sorted(expected - supplied.keys()):
        raise ValueError(f"{subject} omits authority owners for {missing!r}")
    if extra := sorted(supplied.keys() - expected):
        raise ValueError(f"{subject} names authority owners for unknown ids {extra!r}")
    for identifier, owner in supplied.items():
        if not isinstance(owner, str) or not owner:
            raise ValueError(
                f"{subject} authority owner for {identifier!r} must be a non-empty string"
            )
    return dict(supplied)


def _validate_attributed_remote_choice_authority(
    schemas: Mapping[str, NodeSchema],
    choices: Mapping[str, Sequence[str]],
    schema_owners: Mapping[str, object],
    choice_owners: Mapping[str, object],
    *,
    subject: str,
) -> None:
    """Validate remote routes against choices from the same host-attributed owner."""
    for authority in dict.fromkeys(schema_owners.values()):
        validate_remote_choice_authority(
            {
                node_type: schema
                for node_type, schema in schemas.items()
                if schema_owners[node_type] == authority
            },
            {
                choice_id: values
                for choice_id, values in choices.items()
                if choice_owners[choice_id] == authority
            },
            owner=subject,
        )


def _validated_choices(
    choices: Mapping[str, Sequence[str]] | None, *, subject: str
) -> dict[str, tuple[str, ...]]:
    validated: dict[str, tuple[str, ...]] = {}
    for choice_id, raw_values in cast("Mapping[object, object]", choices or {}).items():
        if not isinstance(choice_id, str):
            raise ValueError(f"{subject} choice id must be a string")
        problem = validate_name(choice_id)
        if problem is not None:
            raise ValueError(f"{subject} choice id {choice_id!r} {problem}")
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            raise ValueError(f"{subject} choice {choice_id!r} must be a sequence of strings")
        values = tuple(cast("Sequence[object]", raw_values))
        combo_choices_json_bytes(values, subject=f"{subject} choice {choice_id!r}")
        validated[choice_id] = tuple(cast("Sequence[str]", values))
    return validated


def _dinkster_version() -> str:
    """The running Dinkster release, from installed distribution metadata.

    Authoritative fact for the /api/nodes environment header: clients stamp
    it into saved workflows as an advisory record (never identity), so it
    must be what is genuinely installed - when no distribution metadata
    exists (bare source tree), report "unknown" rather than fabricate a
    version."""
    try:
        return importlib.metadata.version("dinkster")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class CacheExport(Protocol):
    """What the cache-sharing endpoints need from a persistent cache store
    (DiskCacheStore satisfies it structurally)."""

    async def entry_wire(self, key: str) -> Mapping[str, Any] | None: ...

    async def blob(self, digest: str) -> bytes | None: ...


CACHE_EXPORT_KEY: web.AppKey[CacheExport] = web.AppKey("cache_export")


class AssetExport(Protocol):
    """What the asset-sharing endpoints need from a local asset holding
    (AssetVault satisfies it structurally; LocalAssetLibrary too). Keep
    exports *local* - a fetching resolver here would turn an incoming
    request into an outgoing download on the event loop."""

    def digests(self) -> list[str]: ...

    def resolve(self, digest: str) -> Path | None: ...


ASSET_EXPORT_KEY: web.AppKey[AssetExport] = web.AppKey("asset_export")

# Node execution states, updated from engine events. A map, not a cursor:
# parallel execution means many nodes can be running at once.
_NODE_STATE_FOR_KIND = {
    "node_started": "running",
    "node_cached": "cached",
    "node_finished": "completed",
    "node_failed": "failed",
    "node_skipped": "skipped",
}


DESCRIPTOR_ELEMENT_CAP = 64
"""Per-level bound on child descriptors in job-result value descriptors.
Nested structure (per-element types/lengths, arbitrary depth) stays remotely
interrogable without letting a 10k-element list balloon a status response;
"length" is always the full count, and truncation is explicit."""


def value_descriptor(value: Value, registry: TypeRegistry | None = None) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "typeId": value.type_id,
        "fingerprint": value.fingerprint,
        "meta": dict(value.meta.entries),
    }
    if registry is not None:
        # Inline small scalars (DESIGN 3.5): declared-safe serialization
        # only (TypeRegistry.inline_of enforces the scalar/size policy), so
        # peeking an int or short string never needs a payload round trip.
        # Omitted means "not inline" - absence has its own channel.
        scalar = registry.inline_of(value)
        if scalar is not None:
            descriptor["value"] = scalar
    children = list_children(value)
    if children is not None:
        # Runtime length is data, never schema (DESIGN 3.13): the count rides
        # execution-scoped result payloads so frontends can badge list edges.
        descriptor["length"] = len(children)
        # Child envelopes are full values, so descriptors recurse: nested
        # lists expose their shape (ragged lengths and all) level by level.
        descriptor["elements"] = [
            value_descriptor(child, registry) for child in children[:DESCRIPTOR_ELEMENT_CAP]
        ]
        if len(children) > DESCRIPTOR_ELEMENT_CAP:
            descriptor["elementsTruncated"] = True
    return descriptor


def job_to_wire(
    job: Job,
    node_states: Mapping[str, str] | None,
    registry: TypeRegistry | None = None,
) -> dict[str, object]:
    wire: dict[str, object] = {
        "clientId": job.key.client_id,
        "jobId": job.key.job_id,
        "scope": job.scope,
        "submittedBy": {
            "principalId": job.principal_id,
            "kind": job.principal_kind,
        },
        "state": job.state,
        "priority": job.priority,
        "attemptId": job.attempt,
        # jobRef is the server-assigned global job reference (platform plan);
        # runId is the same value under its historical name. One identity,
        # two wire names until clients migrate off runId.
        "jobRef": job.run_id,
        "runId": job.run_id,
        "submittedAt": job.submitted_at,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "nodeStates": dict(node_states or {}),
        "latestSeq": job.latest_seq,
    }
    if job.execution is not None:
        wire["extensionSnapshotDigest"] = job.execution.extension_snapshot_digest
    if job.source_document:
        wire["sourceDocument"] = job.source_document
    if job.error is not None:
        wire["error"] = job.error
    if job.result is not None:
        wire["outputs"] = {
            node_id: {out: value_descriptor(v, registry) for out, v in outs.items()}
            for node_id, outs in job.result.outputs.items()
        }
        wire["executed"] = list(job.result.executed)
        wire["cached"] = list(job.result.cached)
        wire["skipped"] = list(job.result.skipped)
        wire["artifacts"] = [
            {
                "nodeId": artifact.node_id,
                "digest": artifact.digest,
                "name": artifact.name,
                "size": artifact.size,
                "mediaType": artifact.media_type,
                "virtualPath": artifact.virtual_path,
            }
            for artifact in job.result.artifacts
        ]
    return wire


CORE_PACK_ID = "core"
"""Reserved pack id for host-kernel surfaces such as dev scaffolding."""


def _validated_execution_arms(
    schemas: Mapping[str, NodeSchema],
    execution_arms: Mapping[str, Sequence[ExecutionArm]] | None,
    *,
    subject: str,
    include_defaults: bool = True,
) -> dict[str, tuple[ExecutionArm, ...]]:
    result: dict[str, tuple[ExecutionArm, ...]] = (
        {node_type: ("native",) for node_type in schemas} if include_defaults else {}
    )
    for node_type, arms in (execution_arms or {}).items():
        if node_type not in schemas:
            raise ValueError(f"{subject} names unknown node type {node_type!r}")
        normalized = tuple(dict.fromkeys(arms))
        if not normalized or any(arm not in ("native", "comfyui") for arm in normalized):
            raise ValueError(f"{subject} has invalid execution arms for {node_type!r}")
        result[node_type] = normalized
    return result


@dataclass(frozen=True)
class PackIconAsset:
    """A pack's raster badge as served by /api/packs/{packId}/icon.

    ``digest`` (``sha256:<hex>``) is the immutability contract: the bytes
    served for a digest never change - a changed icon is a new digest in
    the packs table - which is what licenses clients to cache decoded
    bitmaps forever. ``data`` is held in memory (validation caps it at
    64 KiB) so the served bytes can never drift from the advertised
    digest, no matter what happens to the file on disk."""

    digest: str
    media_type: str
    data: bytes = field(repr=False, default=b"")


@dataclass(frozen=True)
class PackBlueprintAsset:
    """One pack blueprint as served by /api/packs/{packId}/blueprints/{id}.

    A blueprint is a starter workflow document the pack ships as plain
    data; the server never interprets it (document semantics are the
    frontend's). The packs table advertises the full descriptor inline
    ({id, name, description?, tags?, boundaryInputs?, boundaryOutputs?,
    digest}) so search palettes list
    every blueprint with zero extra fetches; only the graph body rides
    the endpoint. ``digest`` (``sha256:<hex>``) is the immutability
    contract shared with icons: bytes served for a digest never change -
    a changed blueprint is a new digest in the packs table - which
    licenses clients to cache decoded documents forever. ``data`` is held
    in memory (validation caps it at 1 MiB per blueprint, 16 MiB per
    pack) so the served bytes can never drift from the advertised
    digest."""

    id: str
    name: str
    digest: str
    description: str = ""
    tags: tuple[str, ...] = ()
    boundary_inputs: tuple[str, ...] = ()
    """Author-declared boundary input type ids, passed through verbatim:
    search-affordance hints so frontends can port-filter blueprints without
    fetching bodies. The server never derives them from the document or
    checks them against it - declaration-vs-document semantics are the
    frontend's."""
    boundary_outputs: tuple[str, ...] = ()
    """Author-declared boundary output type ids; same verbatim contract."""
    data: bytes = field(repr=False, default=b"")

    def descriptor(self) -> dict[str, object]:
        wire: dict[str, object] = {"id": self.id, "name": self.name}
        if self.description:
            wire["description"] = self.description
        if self.tags:
            wire["tags"] = list(self.tags)
        if self.boundary_inputs:
            wire["boundaryInputs"] = list(self.boundary_inputs)
        if self.boundary_outputs:
            wire["boundaryOutputs"] = list(self.boundary_outputs)
        wire["digest"] = self.digest
        return wire


@dataclass(frozen=True)
class PackTemplateAsset:
    """One pack template as served by /api/templates (descriptor) and
    /api/packs/{packId}/templates/{id} (body bytes).

    A template is a complete starter workflow document the pack ships as
    plain data; the server never interprets it (document semantics are
    the frontend's). Unlike blueprints, template descriptors do NOT ride
    the /api/nodes packs table inline: templates are a browse surface, so
    they get the query-first paged index instead - dumping an unbounded
    collection into the schema table would bloat every schema refresh.
    ``assets`` lists the pack-local [[pack.assets]] ids the template
    requires; clients join them against the pack's asset descriptors
    (which DO ride the packs table) for the needs/consent UI. ``digest``
    (``sha256:<hex>``) is the same immutability contract as icons and
    blueprints: bytes served for a digest never change, licensing clients
    to cache decoded documents forever. ``data`` is held in memory
    (validation caps it at 1 MiB per template, 16 MiB per pack) so the
    served bytes can never drift from the advertised digest."""

    id: str
    name: str
    digest: str
    description: str = ""
    tags: tuple[str, ...] = ()
    family: str = ""
    models: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    """Pack-local [[pack.assets]] ids, passed through verbatim: the
    manifest already validated they exist among the pack's surviving
    declarations."""
    thumbnail: PackIconAsset | None = None
    data: bytes = field(repr=False, default=b"")

    def descriptor(self, pack_id: str) -> dict[str, object]:
        wire: dict[str, object] = {
            "pack": pack_id,
            "id": self.id,
            "name": self.name,
        }
        if self.description:
            wire["description"] = self.description
        if self.tags:
            wire["tags"] = list(self.tags)
        if self.family:
            wire["family"] = self.family
        if self.models:
            wire["models"] = list(self.models)
        if self.assets:
            wire["assets"] = list(self.assets)
        if self.thumbnail is not None:
            wire["thumbnail"] = {
                "digest": self.thumbnail.digest,
                "mediaType": self.thumbnail.media_type,
            }
        wire["digest"] = self.digest
        return wire


@dataclass(frozen=True)
class PackDocAsset:
    """One validated documentation asset addressed by its digest."""

    source: str
    digest: str
    media_type: str
    data: bytes = field(repr=False, default=b"")

    def descriptor(self) -> dict[str, str]:
        return {"digest": self.digest, "mediaType": self.media_type}


@dataclass(frozen=True)
class PackDocPageAsset:
    """One validated localized Markdown page addressed by its digest."""

    kind: str
    id: str
    locale: str
    title: str
    summary: str
    digest: str
    assets: tuple[PackDocAsset, ...] = ()
    schema_version: int | None = None
    order: int | None = None
    tags: tuple[str, ...] = ()
    guide_kind: str | None = None
    data: bytes = field(repr=False, default=b"")


@dataclass(frozen=True)
class PackDocsAsset:
    """One pack's validated docs collection."""

    default_locale: str
    pages: tuple[PackDocPageAsset, ...] = ()


@dataclass(frozen=True)
class PackLocaleCatalogAsset:
    """One pack translation catalog served by its exact digest."""

    locale: str
    digest: str
    node_references: tuple[str, ...] = ()
    data: bytes = field(repr=False, default=b"")


@dataclass(frozen=True)
class PackFrontendAsset:
    """One validated static file served from a pack's frontend namespace."""

    path: str
    media_type: str
    data: bytes = field(repr=False, default=b"")


@dataclass(frozen=True)
class PackInfo:
    """One entry of the /api/nodes packs table: authoritative pack identity
    plus author-declared presentation (a compact badge for node headers).

    Attribution is attached by the host at schema collection - a node
    schema never claims its own pack. Presentation is pixels only: abbr and
    mark may collide across packs, packId is the sole lookup key, and none
    of this joins schema signatures or cache identity.

    Provenance (version/artifact_digest/source/publisher) is the host's
    record of what is actually installed, for workflow environment stamping
    and drift diagnostics. Every field is omitted from the wire when
    unknown, never null - omission MEANS unpinned (a dev --pack has a
    source but no release version; fabricating one would turn an honest
    "unpinned" badge into a lie). Like presentation, provenance never joins
    schema signatures or execution identity."""

    display_name: str
    abbr: str = ""
    mark: str = ""
    color: str = ""
    icon: PackIconAsset | None = None
    version: str = ""
    """Registry release version (strict semver). Empty for unpublished
    packs - local/git installs have no release identity, only a digest."""
    artifact_digest: str = ""
    """``sha256:<hex>`` of the installed artifact bytes - the real pin."""
    source: str = ""
    """Install provenance ('registry', 'git:<url>@<commit>',
    'local:<path>') - where the bytes came from, never identity."""
    publisher: str = ""
    """Canonical publisher id ('local' for unpublished installs)."""
    blueprints: tuple[PackBlueprintAsset, ...] = ()
    """Starter workflow documents the pack ships; descriptors ride the
    packs table inline, bodies ride the lazy per-pack endpoint. Never
    identity - excluded from schema signatures and execution identity."""
    assets: tuple[DeclaredAsset, ...] = ()
    """Digest-pinned assets the pack distributes ([[pack.assets]]):
    declarations only, exposed as descriptors so clients can narrate
    "this pack ships/needs these models" with zero extra fetches. Bytes
    never ride the table; acquisition stays behind job preflight's
    digest-exact consent. Never identity."""
    templates: tuple[PackTemplateAsset, ...] = ()
    """Complete starter workflow documents the pack ships. Deliberately
    ABSENT from to_wire(): templates are a browse surface, so descriptors
    ride the query-first paged /api/templates index and bodies ride the
    lazy per-pack endpoint - never the schema table, which every client
    refetches on every schema change. Never identity."""
    docs: PackDocsAsset | None = None
    """Pack docs bytes and descriptors. Deliberately absent from
    ``to_wire``; descriptors ride the paged ``/api/docs`` index."""
    locale_catalogs: tuple[PackLocaleCatalogAsset, ...] = ()
    """Pack translations listed as locale-to-digest descriptors;
    exact bytes ride the immutable per-pack endpoint."""
    comfy_aliases: ComfyAliasRegistry | None = None
    """Maintained ComfyUI import translations. Dedicated pack metadata:
    rules are never copied into native node schemas or execution surfaces."""
    comfy_groups: ComfyGroupRegistry | None = None
    """Maintained ComfyUI group patterns, separate from executable schemas."""
    frontend_assets: tuple[PackFrontendAsset, ...] = ()
    """Validated static frontend files, served lazily outside the packs table."""
    settings_schema: PackSettingsSchema | None = None
    """Declared user settings schema; values live in the host library."""

    def to_wire(
        self,
        *,
        schemas: Mapping[str, NodeSchema] | None = None,
        published_types: frozenset[str] | None = None,
    ) -> dict[str, object]:
        wire: dict[str, object] = {"displayName": self.display_name}
        if self.abbr:
            wire["abbr"] = self.abbr
        if self.mark:
            wire["mark"] = self.mark
        if self.color:
            wire["color"] = self.color
        if self.version:
            wire["version"] = self.version
        if self.artifact_digest:
            wire["artifactDigest"] = self.artifact_digest
        if self.source:
            wire["source"] = self.source
        if self.publisher:
            wire["publisher"] = self.publisher
        if self.icon is not None:
            # Descriptor only - bytes ride /api/packs/{packId}/icon, never
            # inline (the table would re-download on every schema refresh).
            wire["icon"] = {
                "digest": self.icon.digest,
                "mediaType": self.icon.media_type,
            }
        if self.blueprints:
            # Descriptors only - bodies ride the lazy per-pack endpoint
            # /api/packs/{packId}/blueprints/{id}, never the table (the
            # table would re-download every body on each schema refresh).
            wire["blueprints"] = [bp.descriptor() for bp in self.blueprints]
        if self.assets:
            # Descriptors only, same reasoning: identity + leads, no bytes.
            wire["assets"] = [asset.descriptor() for asset in self.assets]
        if self.locale_catalogs:
            catalogs = self.locale_catalogs
            if published_types is not None:
                catalogs = tuple(
                    catalog
                    for catalog in catalogs
                    if all(node_type in published_types for node_type in catalog.node_references)
                )
            if catalogs:
                wire["locales"] = {
                    catalog.locale: catalog.digest
                    for catalog in sorted(catalogs, key=lambda item: item.locale)
                }
        if self.comfy_aliases is not None:
            aliases = self.comfy_aliases
            if published_types is not None:
                alias_records = tuple(r for r in aliases.records if r.carrier in published_types)
                source_types = {r.source.node_type for r in alias_records}
                aliases = replace(
                    aliases,
                    records=alias_records,
                    source_schemas=tuple(
                        s for s in aliases.source_schemas if s.schema.node_type in source_types
                    ),
                )
            wire["comfyAliases"] = comfy_alias_registry_to_wire(
                aliases,
                schemas=schemas,
            )
        if self.comfy_groups is not None:
            groups = self.comfy_groups
            if published_types is not None:
                group_records = tuple(r for r in groups.records if r.carrier in published_types)
                source_types = {
                    node.source.node_type for r in group_records for _, node in r.pattern.nodes
                }
                group_types = {r.pattern.group_type for r in group_records}
                groups = replace(
                    groups,
                    records=group_records,
                    source_schemas=tuple(
                        s for s in groups.source_schemas if s.schema.node_type in source_types
                    ),
                    group_schemas=tuple(
                        s for s in groups.group_schemas if s.schema.node_type in group_types
                    ),
                )
            wire["comfyGroups"] = comfy_group_registry_to_wire(
                groups,
                schemas=schemas,
            )
        if self.settings_schema is not None:
            wire["settings"] = True
        return wire


def _registry_validation_schemas(
    schemas: Mapping[str, NodeSchema],
    node_packs: Mapping[str, str],
    import_types: set[str],
) -> dict[str, NodeSchema]:
    """Exclude live ComfyUI source schemas from import-only collision checks."""
    return {
        node_type: schema
        for node_type, schema in schemas.items()
        if not (
            node_type in import_types
            and (
                (owner := node_packs.get(node_type, CORE_PACK_ID)) == "comfy"
                or owner.startswith("comfy.")
            )
        )
    }


def _validate_comfy_registry_packs(
    packs: Mapping[str, PackInfo],
    schemas: Mapping[str, NodeSchema],
    node_packs: Mapping[str, str],
    *,
    subject: str,
    defer_unpublished_carriers: bool = False,
) -> None:
    """Check every pack's comfy alias/group registry against the served surface.

    ``defer_unpublished_carriers`` is the incremental-growth relaxation: a
    pack's registry arrives with its pack entry, but provider-gated
    schema-only carriers publish only once an execution provider composes
    (the generation schema owner precedes the compat worker that executes
    its nodes). Records naming a carrier that is not in ``schemas`` yet are
    skipped; the replace that publishes the carrier re-runs this validation
    over the complete merged state, so a carrier owned by the wrong pack is
    still refused - just at publication time. Complete-surface callers keep
    the strict check: there, an unknown carrier is a miswired registry.
    """
    registries = {
        pack_id: info.comfy_aliases
        for pack_id, info in packs.items()
        if info.comfy_aliases is not None
    }
    collisions = comfy_alias_collision_problems(registries)
    if collisions:
        raise ValueError(f"{subject}: {collisions[0]}")
    for pack_id, registry in registries.items():
        for record in registry.records:
            if record.carrier not in schemas:
                if defer_unpublished_carriers:
                    continue
                raise ValueError(
                    f"{subject}: comfy alias record {record.id!r} carrier "
                    f"{record.carrier!r} is not owned by pack {pack_id!r}"
                )
            if node_packs.get(record.carrier, CORE_PACK_ID) != pack_id:
                raise ValueError(
                    f"{subject}: comfy alias record {record.id!r} carrier "
                    f"{record.carrier!r} is not owned by pack {pack_id!r}"
                )
        source_types = {snapshot.schema.node_type for snapshot in registry.source_schemas}
        problems = comfy_alias_registry_problems(
            registry,
            _registry_validation_schemas(schemas, node_packs, source_types),
            ignore_unknown_carriers=defer_unpublished_carriers,
        )
        if problems:
            raise ValueError(
                f"{subject}: invalid comfy alias registry for {pack_id!r}: {problems[0]}"
            )
    group_registries = {
        pack_id: info.comfy_groups
        for pack_id, info in packs.items()
        if info.comfy_groups is not None
    }
    group_collisions = comfy_group_collision_problems(group_registries)
    if group_collisions:
        raise ValueError(f"{subject}: {group_collisions[0]}")
    for pack_id, registry in group_registries.items():
        for record in registry.records:
            if record.carrier not in schemas:
                if defer_unpublished_carriers:
                    continue
                raise ValueError(
                    f"{subject}: comfy group record {record.id!r} carrier "
                    f"{record.carrier!r} is not owned by pack {pack_id!r}"
                )
            if node_packs.get(record.carrier, CORE_PACK_ID) != pack_id:
                raise ValueError(
                    f"{subject}: comfy group record {record.id!r} carrier "
                    f"{record.carrier!r} is not owned by pack {pack_id!r}"
                )
        import_types = {
            snapshot.schema.node_type
            for snapshot in (*registry.source_schemas, *registry.group_schemas)
        }
        problems = comfy_group_registry_problems(
            registry,
            _registry_validation_schemas(schemas, node_packs, import_types),
            ignore_unknown_carriers=defer_unpublished_carriers,
        )
        if problems:
            raise ValueError(
                f"{subject}: invalid comfy group registry for {pack_id!r}: {problems[0]}"
            )


@dataclass(frozen=True)
class WorkerInfo:
    """One configured execution location exposed by ``GET /api/workers``."""

    name: str
    status: str
    node_types: tuple[str, ...]
    device_qualifiers: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "routedNodeTypes": list(self.node_types),
            "deviceQualifiers": list(self.device_qualifiers),
        }


class _JobReplay:
    """One jobRef's soft-bounded, gap-detecting event replay buffer."""

    def __init__(self, limit: int = JOB_EVENT_BUFFER) -> None:
        self._limit = limit
        self._events: deque[tuple[dict[str, object], bool]] = deque()
        self.floor = 0

    def append(self, event: Mapping[str, object], *, droppable: bool) -> None:
        wire = dict(event)
        seq = wire["seq"]
        assert isinstance(seq, int)
        if BINARY_BLOB_KEY in wire:
            del wire[BINARY_BLOB_KEY]
            wire["blobOmitted"] = True
        if len(self._events) >= self._limit:
            # Evict the oldest queued droppable, not the incoming event: a
            # replaying client needs the newest state (the final progress
            # update has no later superseder), and evicting the oldest keeps
            # the floor as low as possible.
            for index, (queued, queued_droppable) in enumerate(self._events):
                if queued_droppable:
                    queued_seq = queued["seq"]
                    assert isinstance(queued_seq, int)
                    self.floor = max(self.floor, queued_seq)
                    del self._events[index]
                    break
            else:
                if droppable:
                    self.floor = max(self.floor, seq)
                    return
                # With only job_state entries, exceed the soft bound rather
                # than lose an authoritative transition.
        self._events.append((wire, droppable))

    def after(self, seq: int) -> list[dict[str, object]] | None:
        if seq < self.floor:
            return None
        result: list[dict[str, object]] = []
        for event, _ in self._events:
            event_seq = event["seq"]
            assert isinstance(event_seq, int)
            if event_seq > seq:
                result.append(dict(event))
        return result


def _placement_worker_for_node(hints: Mapping[str, str], node_id: str) -> str | None:
    return hints.get(top_level_node_id(node_id))


def _region_node_types(region: RegionNode) -> set[str]:
    result: set[str] = set()
    for node in region.body.nodes.values():
        if isinstance(node, GraphNode):
            result.add(node.node_type)
        else:
            result.update(_region_node_types(node))
    return result


@dataclass(frozen=True)
class _SurfaceValidation:
    source_epoch: int
    schemas: Mapping[str, NodeSchema]
    packs: Mapping[str, PackInfo]
    node_packs: Mapping[str, str]
    replacement_problems: tuple[ReplacementProblem, ...]


class ServerState:
    """Wires engine events -> job correlation -> hub, and tracks per-job
    node state maps. Owned by the app; handlers reach it via app[STATE_KEY]."""

    def __init__(
        self,
        make_engine: Callable[[EventListener], Engine],
        schemas: Mapping[str, NodeSchema],
        *,
        max_running_jobs: int = 1,
        governor: MemoryGovernor | None = None,
        history_limit: int = 256,
        packs: Mapping[str, PackInfo] | None = None,
        node_packs: Mapping[str, str] | None = None,
        execution_arms: Mapping[str, Sequence[ExecutionArm]] | None = None,
        history: HistoryStore | None = None,
        choices: Mapping[str, Sequence[str]] | None = None,
        lazy_choices: Mapping[str, LazyChoiceFetcher] | None = None,
        schema_owners: Mapping[str, str] | None = None,
        choice_owners: Mapping[str, str] | None = None,
        compat_skips: Mapping[str, Mapping[str, CompatGateDiagnostic]] | None = None,
        settings: RuntimeSettings | None = None,
        memory_headroom_changed: Callable[[int], None] | None = None,
        residency_memory_budgets: Callable[[], Mapping[str, Mapping[str, int]]] | None = None,
        workers: Callable[[], Sequence[WorkerInfo]] | None = None,
        place_execution: PlaceExecution | None = None,
        debug_errors: bool = False,
        preview_default: str = "cheap",
        preview_animation: str = "ring",
        attention_policy: AttentionPolicy = "auto",
        redactor: PathRedactor | None = None,
        execution_journal: ExecutionJournal | None = None,
        full_free: FullFree | None = None,
    ) -> None:
        # Outbound text (errors, pack failures, node-reported data) is
        # path-redacted through this instance; hosts that know their
        # filesystem roots pass a configured redactor for stable tokens.
        self.redactor = redactor if redactor is not None else PathRedactor()
        # Durable replay copy of published job events (None: live-only).
        self.execution_journal = execution_journal
        self.full_free = full_free
        self.schemas = dict(schemas)
        self.preview_default = validate_preview_mode(preview_default)
        self.preview_animation = validate_preview_animation(preview_animation)
        self.attention_default = AttentionPolicyConfig(validate_attention_policy(attention_policy))
        self._workers = workers
        self._place_execution = place_execution
        self.residency_memory_budgets = residency_memory_budgets
        # Combo choice lists (choice-list id -> values) behind
        # /api/choices/{id}: UI vocabulary for remote ComboWidget routes,
        # never identity - a value outside the served list is the
        # frontend's raw-value fallback, not a server error.
        initial_choices = _validated_choices(choices, subject="server construction")
        initial_lazy = _validated_lazy_choices(lazy_choices, subject="server construction")
        for choice_id in initial_lazy:
            if choice_id in initial_choices:
                raise ValueError(
                    f"server construction registers choice list {choice_id!r} as "
                    "both static and lazy"
                )
        initial_authority = object()
        initial_schema_owners = _validated_authority_owners(
            tuple(self.schemas),
            schema_owners,
            subject="server construction schemas",
            fallback=initial_authority,
        )
        initial_choice_authority = _merged_choice_authority(initial_choices, initial_lazy)
        initial_choice_owners = _validated_authority_owners(
            tuple(initial_choice_authority),
            choice_owners,
            subject="server construction choices",
            fallback=initial_authority,
        )
        _validate_attributed_remote_choice_authority(
            self.schemas,
            initial_choice_authority,
            initial_schema_owners,
            initial_choice_owners,
            subject="server construction",
        )
        self.choices = initial_choices
        # Lazy choice lists behind the same /api/choices/{id} namespace:
        # the fetcher runs the owning worker's provider once per request.
        self.lazy_choices = initial_lazy
        self._schema_owners = initial_schema_owners
        self._choice_owners = initial_choice_owners
        self.compat_skips: dict[tuple[str, str], CompatGateDiagnostic] = {
            (pack_id, node_id): diagnostic
            for pack_id, skips in (compat_skips or {}).items()
            for node_id, diagnostic in skips.items()
        }
        # Node-surface generation, monotonic per process. Fixed at 1 while
        # composition happens before create_app; progressive announcement
        # and hot-reload bump it (and emit schema_changed) when they land.
        self.schema_epoch = 1
        # Pack provenance: the packs table always carries the reserved
        # "core" entry, and every published node attributes to some table
        # entry (unattributed node types fall back to "core"). Misconfigured
        # wiring fails loudly here - provenance is host configuration, not
        # untrusted pack input.
        self.packs = {CORE_PACK_ID: PackInfo(display_name="Dinkster Core")}
        self.packs.update(packs or {})
        self.node_packs = dict(node_packs or {})
        self.execution_arms = _validated_execution_arms(
            self.schemas,
            execution_arms,
            subject="server construction",
        )
        for node_type, pack_id in self.node_packs.items():
            if node_type not in self.schemas:
                raise ValueError(f"node_packs names unknown node type {node_type!r}")
            if pack_id not in self.packs:
                raise ValueError(f"node_packs attributes {node_type!r} to unknown pack {pack_id!r}")
        _validate_comfy_registry_packs(
            self.packs,
            self.schemas,
            self.node_packs,
            subject="server construction",
        )
        # Cross-pack replacement-rule check over the COMPLETE mapping this
        # instance actually serves: doctor can only see one pack at a time,
        # so references between installed packs are verifiable exactly here.
        # Diagnostics, never fatal - a stale migration rule must not take the
        # instance down (the frontend treats these as review-forcing, not
        # unloadable), so they are logged and served at /api/diagnostics.
        self.replacement_problems = validate_replacement_references(self.schemas)
        for problem in self.replacement_problems:
            _log.warning("replacement rule problem: %s", problem.message)
        self.hub = EventHub()
        self.governor = governor
        # Leases exist exactly when a governor does: the broker adds TTL
        # mortality to reservations for holders on the other end of a wire.
        self.leases = LeaseBroker(governor) if governor is not None else None
        self.engine = make_engine(self._on_engine_event)
        # Persistent execution history (optional): the queue records every
        # accepted job in the store before acknowledging it and retires the
        # record at terminal, so a crash leaves durable leftovers. Sweep
        # them into "interrupted" history BEFORE the queue exists - nothing
        # is ever re-executed automatically (user ruling on issue #556).
        self.history = history
        if history is not None:
            interrupted = history.recover_interrupted()
            if interrupted:
                _log.warning(
                    "previous server process left %d unfinished job(s);"
                    " recorded as interrupted history, none re-run",
                    len(interrupted),
                )
        self.queue = JobQueue(
            self.engine,
            max_running_jobs=max_running_jobs,
            on_job_event=self._on_job_event,
            history_limit=history_limit,
            debug_errors=debug_errors,
            redactor=self.redactor,
            store=history,
        )
        default_budgets = {
            device: budget
            for device, report in (governor.status() if governor is not None else {}).items()
            if isinstance((budget := report["budgetBytes"]), int)
        }
        self.settings = settings or RuntimeSettings(
            {
                "memory-budgets": default_budgets,
                "memory-headroom": 256 * 1024**2,
                "aimdo-policy": "auto",
                "dtype-policy": {
                    "diffusion": "auto",
                    "textEncoder": "auto",
                    "vae": "auto",
                },
                "fp8-matmul": False,
                "worker-comfy-args": (),
                "jobs": {"maxRunningJobs": max_running_jobs},
                "logging": {"level": "info", "overrides": {}},
                "p2p": default_p2p_settings(),
            },
            {category: "default" for category in SETTINGS_CATEGORIES},
        )
        self.settings.bind(
            self.governor,
            self.queue,
            memory_headroom_changed=memory_headroom_changed,
        )
        self._node_states: dict[str, dict[str, str]] = {}  # run_id -> node -> state
        self._job_events: dict[str, _JobReplay] = {}
        self.progress_throttle = ProgressThrottle()
        # Composition-progress narration, single source of truth for three
        # read paths: /api/health's "composition" object (supervisor engine
        # protocol), /api/nodes' "composing" flag, and the composition_*
        # events. Hosts drive it through narrate_composition /
        # complete_composition; None means the surface is fully composed.
        # Advisory - never gates any route.
        self.composition_progress: dict[str, object] | None = None
        # Per-pack composition report, alive for the process lifetime: the
        # startup-error cache /api/composition serves. Keyed by pack label,
        # entries are {"state": "pending"} -> {"state": "announced",
        # "epoch": N} | {"state": "failed", "error": str}. A failed pack
        # stays recorded so clients can correlate missing node types with
        # "failed to load: <why>" instead of "unknown" (via the workflow
        # environment stamp's node->pack mapping). Advisory, never
        # identity, never gates a route.
        self.composition_packs: dict[str, dict[str, object]] = {}
        self.composition_started = False
        # run_id -> runtime node id -> engine cache key. Recorded from
        # node_finished/node_cached events; this is what makes intermediate
        # outputs (including region iterations like "r[3]/node") addressable
        # by execution-scoped identity after the run (DESIGN 3.5). A cache
        # key is derived from the invocation's content fingerprints, so a
        # lookup through it can only ever see the exact value this run saw -
        # or nothing (evicted) - never a newer one.
        self._run_keys: dict[str, dict[str, str]] = {}
        self._job_admission_locks: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}

    def workers(self) -> tuple[WorkerInfo, ...]:
        if self._workers is None:
            return (
                WorkerInfo(
                    name="local",
                    status="connected",
                    node_types=tuple(sorted(self.schemas)),
                ),
            )
        workers = tuple(self._workers())
        names = [worker.name for worker in workers]
        if len(names) != len(set(names)):
            raise RuntimeError("worker provider returned duplicate names")
        return workers

    def validate_placement(
        self,
        graph: Graph,
        hints: Mapping[str, str],
        *,
        submitted_graph: Graph | None = None,
    ) -> None:
        workers = {worker.name: worker for worker in self.workers()}
        for node_id, worker_name in hints.items():
            if "/" in node_id:
                raise ValueError(
                    f"placement hint {node_id!r} is invalid: keys must be top-level "
                    "node ids without '/'"
                )
            node = graph.nodes.get(node_id)
            if node is None:
                submitted_node = (
                    None if submitted_graph is None else submitted_graph.nodes.get(node_id)
                )
                if isinstance(submitted_node, GraphNode):
                    schema = self.schemas.get(submitted_node.node_type)
                    if schema is not None and schema.selector is not None:
                        raise ValueError(
                            f"placement hint {node_id!r} names a selector node removed by "
                            "selector lowering"
                        )
                if submitted_node is not None:
                    raise ValueError(
                        f"placement hint {node_id!r} names a node that was pruned by "
                        "selector lowering"
                    )
                raise ValueError(f"placement hint {node_id!r} names unknown top-level node id")
            worker = workers.get(worker_name)
            if worker is None:
                raise ValueError(f"placement hint {node_id!r} names unknown worker {worker_name!r}")
            if worker.status != "connected":
                raise ValueError(
                    f"placement hint {node_id!r} names worker {worker_name!r}, "
                    "which is not connected"
                )
            node_types = (
                {node.node_type} if isinstance(node, GraphNode) else _region_node_types(node)
            )
            unsupported = sorted(node_types - set(worker.node_types))
            if unsupported:
                subject = "region body node type" if isinstance(node, RegionNode) else "node type"
                raise ValueError(
                    f"placement hint {node_id!r} names worker {worker_name!r}, "
                    f"which cannot execute {subject} {unsupported[0]!r}"
                )
            lazy_types = sorted(
                node_type
                for node_type in node_types
                if (schema := self.schemas.get(node_type)) is not None
                and any(input_spec.lazy for input_spec in schema.inputs)
            )
            if worker_name != "local" and lazy_types:
                raise ValueError(
                    f"placement hint {node_id!r} names worker {worker_name!r}, but remote "
                    f"placement does not support lazy node type {lazy_types[0]!r}"
                )

    def place_execution(
        self,
        execution: ExecutionRuntime,
        hints: Mapping[str, str],
    ) -> ExecutionRuntime:
        if not hints:
            return execution
        copied = dict(hints)
        if self._place_execution is not None:
            return self._place_execution(execution, copied)

        def placement_worker(node_id: str) -> str | None:
            return _placement_worker_for_node(copied, node_id)

        return replace(
            execution,
            placement_worker=placement_worker,
        )

    @contextlib.asynccontextmanager
    async def job_admission(self, client_id: str, job_id: str) -> AsyncGenerator[None]:
        """Serialize admission for one idempotency key without blocking others."""
        key = (client_id, job_id)
        lock, users = self._job_admission_locks.get(key, (asyncio.Lock(), 0))
        self._job_admission_locks[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            current_lock, current_users = self._job_admission_locks[key]
            assert current_lock is lock and current_users > 0
            if current_users == 1:
                del self._job_admission_locks[key]
            else:
                self._job_admission_locks[key] = (lock, current_users - 1)

    def announce(
        self,
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
        *,
        execution_arms: Mapping[str, Sequence[ExecutionArm]] | None = None,
        choices: Mapping[str, Sequence[str]] | None = None,
        lazy_choices: Mapping[str, LazyChoiceFetcher] | None = None,
        schema_owners: Mapping[str, str] | None = None,
        choice_owners: Mapping[str, str] | None = None,
        compat_skips: Mapping[str, Mapping[str, CompatGateDiagnostic]] | None = None,
    ) -> int:
        """Additively grow the served node surface (progressive announcement;
        pack hot-reload and live install activation ride the same seam).

        Validates the delta against the same rules construction applies -
        node-type collisions refused, pack-table entries may repeat only
        when identical (two compat workers both contributing "comfy"),
        every attribution names a known type and a known table entry -
        then merges state, re-derives the cross-pack replacement
        diagnostics over the COMPLETE mapping, bumps the surface epoch,
        and broadcasts {"type": "schema_changed", "epoch": N}.

        Ordering is the settled frontend contract: state mutates BEFORE the
        event publishes, and both happen synchronously on the event loop,
        so any /api/nodes fetch a client issues after seeing epoch N
        observes at least that surface. Returns the new epoch.
        """
        for node_type in schemas:
            if node_type in self.schemas:
                raise ValueError(f"announce would redefine node type {node_type!r}")
        merged_packs = dict(self.packs)
        for pack_id, info in packs.items():
            existing = merged_packs.get(pack_id)
            if existing is not None and existing != info:
                raise ValueError(f"announce redeclares pack {pack_id!r} with different info")
            merged_packs[pack_id] = info
        for node_type, pack_id in node_packs.items():
            if node_type not in schemas:
                raise ValueError(
                    f"announce attributes {node_type!r}, which is not in this "
                    "announcement's schemas"
                )
            if pack_id not in merged_packs:
                raise ValueError(f"announce attributes {node_type!r} to unknown pack {pack_id!r}")
        incoming_choices = _validated_choices(choices, subject="server announcement")
        incoming_lazy = _validated_lazy_choices(lazy_choices, subject="server announcement")
        incoming_authority = object()
        incoming_schema_owners = _validated_authority_owners(
            tuple(schemas),
            schema_owners,
            subject="server announcement schemas",
            fallback=incoming_authority,
        )
        incoming_choice_authority = _merged_choice_authority(incoming_choices, incoming_lazy)
        incoming_choice_owners = _validated_authority_owners(
            tuple(incoming_choice_authority),
            choice_owners,
            subject="server announcement choices",
            fallback=incoming_authority,
        )
        incoming_execution_arms: dict[str, tuple[ExecutionArm, ...]] = {
            node_type: ("native",) for node_type in schemas
        }
        incoming_execution_arms.update(
            _validated_execution_arms(
                {**self.schemas, **schemas},
                execution_arms,
                subject="server announcement",
                include_defaults=False,
            )
        )
        for choice_id in incoming_choices:
            if choice_id in self.choices or choice_id in self.lazy_choices:
                raise ValueError(f"announce would redefine choice list {choice_id!r}")
        for choice_id in incoming_lazy:
            if (
                choice_id in self.choices
                or choice_id in self.lazy_choices
                or choice_id in incoming_choices
            ):
                raise ValueError(f"announce would redefine choice list {choice_id!r}")
        _validate_attributed_remote_choice_authority(
            {**self.schemas, **schemas},
            _merged_choice_authority(
                {**self.choices, **incoming_choices},
                {**self.lazy_choices, **incoming_lazy},
            ),
            {**self._schema_owners, **incoming_schema_owners},
            {**self._choice_owners, **incoming_choice_owners},
            subject="server announcement result",
        )
        incoming_skips = {
            (pack_id, node_id): diagnostic
            for pack_id, skips in (compat_skips or {}).items()
            for node_id, diagnostic in skips.items()
        }
        for key in incoming_skips:
            if key in self.compat_skips:
                raise ValueError(f"announce would redefine compat skip {key!r}")
        _validate_comfy_registry_packs(
            merged_packs,
            {**self.schemas, **schemas},
            {**self.node_packs, **node_packs},
            subject="server announcement",
            defer_unpublished_carriers=True,
        )
        # The engine holds its own copy of the surface; grow it first so a
        # failure there (host miswiring) leaves the served state untouched.
        self.engine.announce_schemas(schemas)
        self.schemas = {**self.schemas, **schemas}
        self.packs = merged_packs
        self.node_packs = {**self.node_packs, **node_packs}
        self.execution_arms = {**self.execution_arms, **incoming_execution_arms}
        self.choices = {**self.choices, **incoming_choices}
        self.lazy_choices = {**self.lazy_choices, **incoming_lazy}
        self._schema_owners = {**self._schema_owners, **incoming_schema_owners}
        self._choice_owners = {**self._choice_owners, **incoming_choice_owners}
        self.compat_skips = {**self.compat_skips, **incoming_skips}
        self.replacement_problems = validate_replacement_references(self.schemas)
        for problem in self.replacement_problems:
            _log.warning("replacement rule problem: %s", problem.message)
        self.schema_epoch += 1
        self.hub.publish(
            {"type": "schema_changed", "epoch": self.schema_epoch},
            client_id=None,
            droppable=False,
        )
        return self.schema_epoch

    def publish_generation(
        self,
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
        *,
        execution_arms: Mapping[str, Sequence[ExecutionArm]] | None = None,
        choices: Mapping[str, Sequence[str]] | None = None,
        lazy_choices: Mapping[str, LazyChoiceFetcher] | None = None,
        schema_owners: Mapping[str, str] | None = None,
        choice_owners: Mapping[str, str] | None = None,
        compat_skips: Mapping[str, Mapping[str, CompatGateDiagnostic]] | None = None,
        _validation: _SurfaceValidation | None = None,
    ) -> int:
        """Atomically replace the complete published composition generation."""
        next_schemas = dict(schemas)
        next_packs = {CORE_PACK_ID: PackInfo(display_name="Dinkster Core")}
        next_packs.update(packs)
        next_node_packs = dict(node_packs)
        next_execution_arms = _validated_execution_arms(
            next_schemas,
            execution_arms,
            subject="server generation",
        )
        for node_type, pack_id in next_node_packs.items():
            if node_type not in next_schemas:
                raise ValueError(f"generation attributes unknown node type {node_type!r}")
            if pack_id not in next_packs:
                raise ValueError(f"generation attributes {node_type!r} to unknown pack {pack_id!r}")
        next_choices = _validated_choices(choices, subject="server generation")
        next_lazy = _validated_lazy_choices(lazy_choices, subject="server generation")
        for choice_id in next_lazy:
            if choice_id in next_choices:
                raise ValueError(
                    f"generation registers choice list {choice_id!r} as both static and lazy"
                )
        next_authority = object()
        next_schema_owners = _validated_authority_owners(
            tuple(next_schemas),
            schema_owners,
            subject="server generation schemas",
            fallback=next_authority,
        )
        next_choice_authority = _merged_choice_authority(next_choices, next_lazy)
        next_choice_owners = _validated_authority_owners(
            tuple(next_choice_authority),
            choice_owners,
            subject="server generation choices",
            fallback=next_authority,
        )
        _validate_attributed_remote_choice_authority(
            next_schemas,
            next_choice_authority,
            next_schema_owners,
            next_choice_owners,
            subject="server generation",
        )
        next_skips = {
            (pack_id, node_id): diagnostic
            for pack_id, skips in (compat_skips or {}).items()
            for node_id, diagnostic in skips.items()
        }
        if _validation is None:
            _validate_comfy_registry_packs(
                next_packs,
                next_schemas,
                next_node_packs,
                subject="server generation",
            )
            replacement_problems = tuple(validate_replacement_references(next_schemas))
        else:
            if (
                _validation.source_epoch != self.schema_epoch
                or _validation.schemas != next_schemas
                or _validation.packs != next_packs
                or _validation.node_packs != next_node_packs
            ):
                raise RuntimeError("prepared generation no longer matches the served surface")
            replacement_problems = _validation.replacement_problems
        # Engine schemas replace in one reference swap. The remaining state
        # assignments and event publication are synchronous, so no request can
        # observe a torn generation on the event loop.
        self.engine.replace_schemas(tuple(self.schemas), next_schemas)
        self.schemas = next_schemas
        self.packs = next_packs
        self.node_packs = next_node_packs
        self.execution_arms = next_execution_arms
        self.choices = next_choices
        self.lazy_choices = next_lazy
        self._schema_owners = next_schema_owners
        self._choice_owners = next_choice_owners
        self.compat_skips = next_skips
        self.replacement_problems = replacement_problems
        for problem in self.replacement_problems:
            _log.warning("replacement rule problem: %s", problem.message)
        self.schema_epoch += 1
        self.hub.publish(
            {"type": "schema_changed", "epoch": self.schema_epoch},
            client_id=None,
            droppable=False,
        )
        return self.schema_epoch

    async def _prepare_surface_validation(
        self,
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
        *,
        subject: str,
        defer_unpublished_carriers: bool = False,
    ) -> _SurfaceValidation:
        source_epoch = self.schema_epoch
        frozen_schemas = MappingProxyType(dict(schemas))
        frozen_packs = MappingProxyType(dict(packs))
        frozen_node_packs = MappingProxyType(dict(node_packs))

        def validate() -> tuple[ReplacementProblem, ...]:
            _validate_comfy_registry_packs(
                frozen_packs,
                frozen_schemas,
                frozen_node_packs,
                subject=subject,
                defer_unpublished_carriers=defer_unpublished_carriers,
            )
            return tuple(validate_replacement_references(frozen_schemas))

        replacement_problems = await asyncio.to_thread(validate)
        return _SurfaceValidation(
            source_epoch,
            frozen_schemas,
            frozen_packs,
            frozen_node_packs,
            replacement_problems,
        )

    async def prepare_generation(
        self,
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
    ) -> _SurfaceValidation:
        """Validate a complete publication generation off the event loop."""
        next_packs = {CORE_PACK_ID: PackInfo(display_name="Dinkster Core")}
        next_packs.update(packs)
        return await self._prepare_surface_validation(
            schemas,
            next_packs,
            node_packs,
            subject="server generation",
        )

    async def prepare_replace(
        self,
        remove_types: Sequence[str],
        remove_packs: Sequence[str],
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
    ) -> _SurfaceValidation:
        """Validate the CPU-bound complete replacement surface off the event loop."""
        removed_types = set(remove_types)
        removed_packs = set(remove_packs)
        merged_schemas = {
            node_type: schema
            for node_type, schema in self.schemas.items()
            if node_type not in removed_types
        }
        merged_schemas.update(schemas)
        merged_packs = {
            pack_id: info for pack_id, info in self.packs.items() if pack_id not in removed_packs
        }
        merged_packs.update(packs)
        merged_node_packs = {
            node_type: pack_id
            for node_type, pack_id in self.node_packs.items()
            if node_type not in removed_types
        }
        merged_node_packs.update(node_packs)

        return await self._prepare_surface_validation(
            merged_schemas,
            merged_packs,
            merged_node_packs,
            subject="server replacement",
            defer_unpublished_carriers=True,
        )

    def replace(
        self,
        remove_types: Sequence[str],
        remove_packs: Sequence[str],
        schemas: Mapping[str, NodeSchema],
        packs: Mapping[str, PackInfo],
        node_packs: Mapping[str, str],
        *,
        execution_arms: Mapping[str, Sequence[ExecutionArm]] | None = None,
        remove_choices: Sequence[str] = (),
        choices: Mapping[str, Sequence[str]] | None = None,
        lazy_choices: Mapping[str, LazyChoiceFetcher] | None = None,
        schema_owners: Mapping[str, str] | None = None,
        choice_owners: Mapping[str, str] | None = None,
        remove_compat_skips: Sequence[tuple[str, str]] = (),
        compat_skips: Mapping[str, Mapping[str, CompatGateDiagnostic]] | None = None,
        _validation: _SurfaceValidation | None = None,
    ) -> int:
        """Swap one pack's slice of the served surface (hot reload): the
        removal half announce never needed. Removed types vanish from
        /api/nodes, removed pack ids leave the table, and the incoming
        delta lands - all in one epoch bump and ONE schema_changed event,
        so clients see a single refetch, not a flicker of vanished types.

        Validation mirrors announce, relaxed exactly where an atomic swap
        needs it: an incoming schema or choice list may redefine an id being
        removed in the same swap, and an incoming choice may introduce an id
        that is derived from another incoming list. The one new invariant:
        after the swap, every surviving attribution must still name a table
        entry - a removal can never orphan another pack's nodes. The reserved
        core entry is irremovable.

        Ordering is announce's settled contract: the engine's surface
        swaps first (failure there leaves served state untouched), state
        mutates, THEN the event publishes - a fetch after seeing epoch N
        observes at least that surface. Returns the new epoch.
        """
        removed_types = set(remove_types)
        for node_type in removed_types:
            if node_type not in self.schemas:
                raise ValueError(f"replace would remove unknown node type {node_type!r}")
        removed_packs = set(remove_packs)
        for pack_id in removed_packs:
            if pack_id == CORE_PACK_ID:
                raise ValueError(f"replace cannot remove the {CORE_PACK_ID!r} entry")
            if pack_id not in self.packs:
                raise ValueError(f"replace would remove unknown pack {pack_id!r}")
        for node_type in schemas:
            if node_type in self.schemas and node_type not in removed_types:
                raise ValueError(
                    f"replace would redefine node type {node_type!r}, which is not "
                    "being removed in this swap"
                )
        merged_packs = {
            pack_id: info for pack_id, info in self.packs.items() if pack_id not in removed_packs
        }
        for pack_id, info in packs.items():
            existing = merged_packs.get(pack_id)
            if existing is not None and existing != info:
                raise ValueError(f"replace redeclares pack {pack_id!r} with different info")
            merged_packs[pack_id] = info
        for node_type in node_packs:
            if node_type not in schemas:
                raise ValueError(
                    f"replace attributes {node_type!r}, which is not in this announcement's schemas"
                )
        merged_node_packs = {
            node_type: pack_id
            for node_type, pack_id in self.node_packs.items()
            if node_type not in removed_types
        }
        merged_node_packs.update(node_packs)
        for node_type, pack_id in merged_node_packs.items():
            if pack_id not in merged_packs:
                raise ValueError(
                    f"replace would orphan {node_type!r}: its pack {pack_id!r} would "
                    "no longer be in the table"
                )
        incoming_choices = _validated_choices(choices, subject="server replacement")
        incoming_lazy = _validated_lazy_choices(lazy_choices, subject="server replacement")
        incoming_authority = object()
        incoming_schema_owners = _validated_authority_owners(
            tuple(schemas),
            schema_owners,
            subject="server replacement schemas",
            fallback=incoming_authority,
            retained=self._schema_owners,
        )
        incoming_choice_authority = _merged_choice_authority(incoming_choices, incoming_lazy)
        incoming_choice_owners = _validated_authority_owners(
            tuple(incoming_choice_authority),
            choice_owners,
            subject="server replacement choices",
            fallback=incoming_authority,
            retained=self._choice_owners,
        )
        for choice_id in incoming_lazy:
            if choice_id in incoming_choices:
                raise ValueError(
                    f"replace registers choice list {choice_id!r} as both static and lazy"
                )
        removed_choices = set(remove_choices)
        for choice_id in removed_choices:
            if (
                choice_id not in self.choices
                and choice_id not in self.lazy_choices
                and choice_id not in incoming_choices
                and choice_id not in incoming_lazy
            ):
                raise ValueError(f"replace would remove unknown choice list {choice_id!r}")
        for choice_id in (*incoming_choices, *incoming_lazy):
            if (
                choice_id in self.choices or choice_id in self.lazy_choices
            ) and choice_id not in removed_choices:
                raise ValueError(
                    f"replace would redefine choice list {choice_id!r}, which is not "
                    "being removed in this swap"
                )
        removed_skips = set(remove_compat_skips)
        for key in removed_skips:
            if key not in self.compat_skips:
                raise ValueError(f"replace would remove unknown compat skip {key!r}")
        incoming_skips = {
            (pack_id, node_id): diagnostic
            for pack_id, skips in (compat_skips or {}).items()
            for node_id, diagnostic in skips.items()
        }
        for key in incoming_skips:
            if key in self.compat_skips and key not in removed_skips:
                raise ValueError(
                    f"replace would redefine compat skip {key!r}, which is not being "
                    "removed in this swap"
                )
        merged_schemas = {
            node_type: schema
            for node_type, schema in self.schemas.items()
            if node_type not in removed_types
        }
        merged_schemas.update(schemas)
        merged_schema_owners = {
            node_type: owner
            for node_type, owner in self._schema_owners.items()
            if node_type not in removed_types
        }
        merged_schema_owners.update(incoming_schema_owners)
        incoming_execution_arms: dict[str, tuple[ExecutionArm, ...]] = {
            node_type: ("native",) for node_type in schemas
        }
        incoming_execution_arms.update(
            _validated_execution_arms(
                merged_schemas,
                execution_arms,
                subject="server replacement",
                include_defaults=False,
            )
        )
        merged_execution_arms = {
            node_type: arms
            for node_type, arms in self.execution_arms.items()
            if node_type not in removed_types
        }
        merged_execution_arms.update(incoming_execution_arms)
        merged_choices = {
            choice_id: values
            for choice_id, values in self.choices.items()
            if choice_id not in removed_choices
        }
        merged_choices.update(incoming_choices)
        merged_lazy = {
            choice_id: fetcher
            for choice_id, fetcher in self.lazy_choices.items()
            if choice_id not in removed_choices
        }
        merged_lazy.update(incoming_lazy)
        merged_choice_owners = {
            choice_id: owner
            for choice_id, owner in self._choice_owners.items()
            if choice_id not in removed_choices
        }
        merged_choice_owners.update(incoming_choice_owners)
        _validate_attributed_remote_choice_authority(
            merged_schemas,
            _merged_choice_authority(merged_choices, merged_lazy),
            merged_schema_owners,
            merged_choice_owners,
            subject="server replacement result",
        )
        if _validation is None:
            _validate_comfy_registry_packs(
                merged_packs,
                merged_schemas,
                merged_node_packs,
                subject="server replacement",
                defer_unpublished_carriers=True,
            )
            replacement_problems = tuple(validate_replacement_references(merged_schemas))
        else:
            if (
                _validation.source_epoch != self.schema_epoch
                or _validation.schemas != merged_schemas
                or _validation.packs != merged_packs
                or _validation.node_packs != merged_node_packs
            ):
                raise RuntimeError("prepared replacement no longer matches the served surface")
            replacement_problems = _validation.replacement_problems
        # The engine holds its own copy of the surface; swap it first so a
        # failure there (host miswiring) leaves the served state untouched.
        self.engine.replace_schemas(tuple(removed_types), schemas)
        self.schemas = merged_schemas
        self.packs = merged_packs
        self.node_packs = merged_node_packs
        self.execution_arms = merged_execution_arms
        self.choices = merged_choices
        self.lazy_choices = merged_lazy
        self._schema_owners = merged_schema_owners
        self._choice_owners = merged_choice_owners
        merged_skips = {
            key: diagnostic
            for key, diagnostic in self.compat_skips.items()
            if key not in removed_skips
        }
        merged_skips.update(incoming_skips)
        self.compat_skips = merged_skips
        self.replacement_problems = replacement_problems
        for problem in self.replacement_problems:
            _log.warning("replacement rule problem: %s", problem.message)
        self.schema_epoch += 1
        self.hub.publish(
            {"type": "schema_changed", "epoch": self.schema_epoch},
            client_id=None,
            droppable=False,
        )
        return self.schema_epoch

    def seed_composition(self, labels: Sequence[str]) -> list[str]:
        """Record every pack the host intends to compose as "pending" and
        return the report keys, one per label in order. Duplicate labels
        (two specs whose manifests resolve to the same name - the second
        will be refused as a duplicate) get "#n" suffixed keys so the
        failure record cannot clobber the survivor's entry."""
        self.composition_started = True
        keys: list[str] = []
        counts: dict[str, int] = {}
        for label in labels:
            n = counts.get(label, 0)
            counts[label] = n + 1
            key = label if n == 0 else f"{label}#{n + 1}"
            keys.append(key)
            self.composition_packs[key] = {"state": "pending"}
        return keys

    def mark_pack_announced(self, pack: str, epoch: int) -> None:
        """Record one pack's successful announcement in the report."""
        self.composition_packs[pack] = {"state": "announced", "epoch": epoch}

    def mark_pack_removed(self, pack: str, epoch: int) -> None:
        """Record one pack's deliberate removal in the report. The row
        stays (state "removed" plus the retracting epoch) rather than
        vanishing: /api/composition tells the process-lifetime story, and
        "was live, then removed" is a different fact from "never
        composed" - a client correlating missing node types needs the
        distinction."""
        self.composition_packs[pack] = {"state": "removed", "epoch": epoch}

    def mark_pack_failed(self, pack: str, error: str) -> None:
        """Record one pack's composition failure - the cached startup error
        /api/composition serves for the process lifetime - and broadcast
        non-droppable {"type": "pack_failed", "pack", "error"} so live
        clients learn WHY a surface stopped growing, not just that it did.
        The error text is raiser-written (import failures name .py files);
        redact it once here so the cached row and the event agree."""
        error = self.redactor.redact_text(error)
        self.composition_packs[pack] = {"state": "failed", "error": error}
        self.hub.publish(
            {"type": "pack_failed", "pack": pack, "error": error},
            client_id=None,
            droppable=False,
        )

    def narrate_composition(self, done: int, total: int, phase: str = "packs") -> None:
        """Record in-flight composition and broadcast it: /api/health grows
        the "composition" object, /api/nodes the "composing" flag, and
        subscribers get {"type": "composition_progress", "done", "total",
        "phase"} - droppable chatter, each update supersedes the last."""
        self.composition_progress = {"done": done, "total": total, "phase": phase}
        self.hub.publish(
            {
                "type": "composition_progress",
                "done": done,
                "total": total,
                "phase": phase,
            },
            client_id=None,
            droppable=True,
        )

    def complete_composition(self) -> None:
        """Mark the surface fully composed. If composition was being
        narrated, broadcasts {"type": "composition_complete", "epoch": N}
        (non-droppable) - the positive "loading actually finished" moment,
        carrying the FINAL epoch so a client holding epoch >= N knows its
        table is the complete surface. Emitted only after the last
        announcement's schema_changed, so fetch-on-complete always
        observes everything. Complete does NOT mean everything loaded:
        packs recorded as failed are named in an additive "failed" list
        (omitted when empty) and stay queryable on /api/composition. No-op
        when nothing was composing (a zero-pack host has no loading to
        finish)."""
        if self.composition_progress is None:
            return
        self.composition_progress = None
        event: dict[str, object] = {
            "type": "composition_complete",
            "epoch": self.schema_epoch,
        }
        failed = sorted(
            pack for pack, entry in self.composition_packs.items() if entry.get("state") == "failed"
        )
        if failed:
            event["failed"] = failed
        self.hub.publish(event, client_id=None, droppable=False)

    def node_states(self, run_id: str) -> dict[str, str]:
        return self._node_states.get(run_id, {})

    def latest_seq(self, run_id: str) -> int:
        job = self.queue.job_for_run(run_id)
        return job.latest_seq if job is not None else 0

    def replay_events(self, run_id: str, after: int) -> list[dict[str, object]] | None:
        replay = self._job_events.get(run_id)
        return [] if replay is None else replay.after(after)

    def run_cache_key(self, run_id: str, node_id: str) -> str | None:
        return self._run_keys.get(run_id, {}).get(node_id)

    def _on_engine_event(self, event: EngineEvent) -> None:  # listener is sync
        if event.kind == "node_event" and event.detail.get("name") == PROGRESS_EVENT:
            data = event.detail.get("data")
            if isinstance(data, Mapping) and not self.progress_throttle.admit(
                event.run_id, event.node_id or "", cast("Mapping[str, object]", data)
            ):
                return
        state = _NODE_STATE_FOR_KIND.get(event.kind)
        if state is not None and event.node_id is not None:
            self._node_states.setdefault(event.run_id, {})[event.node_id] = state
        if event.kind in ("node_finished", "node_cached") and event.node_id is not None:
            key = event.detail.get("cache_key")
            if isinstance(key, str):
                self._run_keys.setdefault(event.run_id, {})[event.node_id] = key
        job = self.queue.job_for_run(event.run_id)
        if job is not None and event.node_id is not None:
            disposition = {
                "node_started": "running",
                "node_cached": "cached",
                "node_finished": "executed",
                "node_failed": "failed",
                "node_skipped": "skipped",
            }.get(event.kind)
            if disposition is not None:
                receipt = {"nodeId": event.node_id, "disposition": disposition}
                execution_arm = event.detail.get("executionArm")
                if execution_arm in ("native", "comfyui"):
                    receipt["executionArm"] = execution_arm
                worker = event.detail.get("worker")
                if isinstance(worker, str):
                    receipt["worker"] = worker
                provider = event.detail.get("provider")
                if isinstance(provider, str):
                    receipt["provider"] = provider
                cache_layer = event.detail.get("cacheLayer")
                if isinstance(cache_layer, str):
                    receipt["cacheLayer"] = cache_layer
                pack = event.detail.get("pack")
                if isinstance(pack, str):
                    receipt["pack"] = pack
                attention_diagnostic = event.detail.get("attentionDiagnostic")
                if isinstance(attention_diagnostic, str):
                    receipt["attentionDiagnostic"] = self.redactor.redact_text(attention_diagnostic)
                job.node_receipts[event.node_id] = receipt
        client_id = job.key.client_id if job is not None else None
        job_id = job.key.job_id if job is not None else None
        wire = engine_event_to_wire(
            event, client_id=client_id, job_id=job_id, redactor=self.redactor
        )
        if job is None:
            self.hub.publish(
                wire,
                client_id=None,
                droppable=event.kind in DROPPABLE_KINDS,
            )
        else:
            wire["jobRef"] = job.run_id
            self._publish_job_event(job, wire, droppable=event.kind in DROPPABLE_KINDS)

    def _on_job_event(self, job: Job) -> None:
        wire: dict[str, object] = {
            "type": "job_state",
            "clientId": job.key.client_id,
            "jobId": job.key.job_id,
            "state": job.state,
            "jobRef": job.run_id,  # canonical name; runId is the legacy alias
            "runId": job.run_id,
            "attemptId": job.attempt,
        }
        if job.error is not None:
            wire["error"] = job.error
        # Job transitions are the contract the frontend keys off; never shed.
        # (Terminal history rows are written by the queue itself, inline
        # through its QueuePersistence store, before this listener runs.)
        self._publish_job_event(job, wire, droppable=False)
        # Run-scoped maps live exactly as long as their job is queryable.
        # Check on every transition because terminal-key resubmission retires
        # its predecessor while emitting the new job's queued transition.
        known = {j.run_id for j in self.queue.jobs()}
        for run_id in [r for r in self._node_states if r not in known]:
            del self._node_states[run_id]
        for run_id in [r for r in self._run_keys if r not in known]:
            del self._run_keys[run_id]
        for run_id in [r for r in self._job_events if r not in known]:
            del self._job_events[run_id]
        self.progress_throttle.retain(known)

    def _publish_job_event(self, job: Job, event: Mapping[str, object], *, droppable: bool) -> None:
        """Assign one per-job sequence before the shared replay/hub fanout."""
        job.latest_seq += 1
        wire = dict(event)
        wire["seq"] = job.latest_seq
        self._job_events.setdefault(job.run_id, _JobReplay()).append(wire, droppable=droppable)
        self.hub.publish(wire, client_id=job.key.client_id, droppable=droppable)
        if self.execution_journal is not None:
            # The journal keeps the published dict itself (redacted, seq'd):
            # replayed events are the live contract, not a translation.
            self.execution_journal.observe(wire, scope=job.scope)


def _bad_request(message: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(text=json.dumps({"error": message}), content_type="application/json")


def _provider_resolution_response(error: ProviderResolutionError) -> web.Response:
    return web.json_response(
        {
            "error": "capability-unavailable",
            "diagnostics": [
                {
                    "severity": "error",
                    "code": "capability-unavailable",
                    "message": str(error),
                    "nodeId": error.node_id,
                    "nodeType": error.node_type,
                    "title": error.title,
                    "capability": error.capability,
                    "remedy": error.remedy,
                }
            ],
        },
        status=400,
    )


def cors_origin(request: web.Request, allowed: frozenset[str]) -> str | None:
    """The Access-Control-Allow-Origin value for this request, or None for
    no CORS response headers (no Origin, or an origin not allowed)."""
    origin = request.headers.get("Origin")
    if origin is None or not allowed:
        return None
    if "*" in allowed:
        return "*"
    return origin if origin in allowed else None


def _authority_hostname(authority: str) -> str | None:
    if not authority or authority != authority.strip():
        return None
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            return None
        hostname = authority[1:closing]
        suffix = authority[closing + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return None
    else:
        if authority.count(":") > 1:
            return None
        hostname, separator, port = authority.partition(":")
        if separator and not port.isdigit():
            return None
    if not hostname or any(character.isspace() or character in "/\\@?#" for character in hostname):
        return None
    hostname = hostname.rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        return hostname


def _configured_hostname(host: str) -> str | None:
    hostname = _authority_hostname(host)
    if hostname is not None:
        return hostname
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def install_browser_request_security(
    app: web.Application,
    *,
    allow_hosts: Sequence[str],
    allow_origins: Sequence[str],
) -> None:
    allowed_hosts: set[str] = set()
    for host in allow_hosts:
        hostname = _configured_hostname(host)
        if hostname is None:
            raise ValueError("allowed hosts must be hostnames or IP address literals")
        allowed_hosts.add(hostname)
    allowed_origins = frozenset(allow_origins)

    @web.middleware
    async def validate_browser_boundary(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        host_headers = request.headers.getall("Host", [])
        hostname = _authority_hostname(host_headers[0]) if len(host_headers) == 1 else None
        try:
            loopback = hostname is not None and ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
        if hostname is None or (not loopback and hostname not in allowed_hosts):
            return web.json_response(
                {
                    "error": "host-not-allowed",
                    "message": "the request Host is not allowed by this server",
                },
                status=421,
            )

        origin = request.headers.get("Origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin is not None:
            same_origin_proxy = request.headers.get("Sec-Fetch-Site", "").lower() == "same-origin"
            if (
                "*" not in allowed_origins
                and origin not in allowed_origins
                and not same_origin_proxy
            ):
                return web.json_response(
                    {
                        "error": "origin-not-allowed",
                        "message": "the request Origin is not allowed for this operation",
                    },
                    status=403,
                )
        return await handler(request)

    app.middlewares.append(validate_browser_boundary)


def install_cors(app: web.Application, allow_origins: Sequence[str]) -> None:
    """Opt-in CORS (DESIGN: owned by whoever owns the public port). Off by
    default - a local instance must not be reachable from any website the
    user happens to visit; explicit origins (or '*') are a deliberate
    deployment decision. Preflights are answered before routing (so they
    work on every path, including 404s), and actual responses are stamped
    in on_response_prepare so streamed responses are covered too."""
    allowed = frozenset(allow_origins)
    if not allowed:
        return

    @web.middleware
    async def preflight(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if request.method == "OPTIONS" and "Access-Control-Request-Method" in request.headers:
            origin = cors_origin(request, allowed)
            if origin is not None:
                headers = {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": request.headers[
                        "Access-Control-Request-Method"
                    ],
                    "Access-Control-Max-Age": "600",
                }
                requested = request.headers.get("Access-Control-Request-Headers")
                if requested:
                    headers["Access-Control-Allow-Headers"] = requested
                if origin != "*":
                    headers["Vary"] = "Origin"
                return web.Response(status=204, headers=headers)
        return await handler(request)

    async def stamp(request: web.Request, response: web.StreamResponse) -> None:
        origin = cors_origin(request, allowed)
        if origin is None:
            return
        response.headers["Access-Control-Allow-Origin"] = origin
        # ETag drives the immutable icon/blueprint/value caching contract;
        # cross-origin scripts must be able to read it.
        response.headers["Access-Control-Expose-Headers"] = "ETag"
        if origin != "*" and not any(
            "origin" in value.lower() for value in response.headers.getall("Vary", [])
        ):
            response.headers.add("Vary", "Origin")

    app.middlewares.append(preflight)
    app.on_response_prepare.append(stamp)


async def handle_health(request: web.Request) -> web.Response:
    """Liveness/readiness plus composition state and progress.

    200 means the app is serving - with progressive announcement that is
    the diagnostic host surface, bound before pack workers start, so the port
    opens in milliseconds instead of after every pack import. ``ok`` is false
    when no node types composed or any pack failed. ``compositionState`` is
    stable after startup; the optional ``composition`` object retains the
    existing in-flight progress contract for supervisors."""
    state = request.app[STATE_KEY]
    if not state.composition_started:
        body: dict[str, object] = {"ok": True}
        if state.composition_progress is not None:
            body["composition"] = dict(state.composition_progress)
        return web.json_response(body)
    failed = sum(entry.get("state") == "failed" for entry in state.composition_packs.values())
    composed = len(state.schemas)
    body = {
        "ok": composed > 0 and failed == 0,
        "compositionState": {
            "composed": composed,
            "failed": failed,
            "epoch": state.schema_epoch,
        },
    }
    if state.composition_progress is not None:
        body["composition"] = dict(state.composition_progress)
    return web.json_response(body)


async def handle_composition(request: web.Request) -> web.Response:
    """Pollable composition state: the whole startup story in one cheap
    fetch, alive for the process lifetime. A client that missed (or never
    subscribed to) the composition_* events reads the same truth here:
    current epoch, whether more packs are expected ("composing" +
    "progress", both omitted once done), and the per-pack report -
    "pending" | {"announced", "epoch"} | {"failed", "error"}. Failed
    entries are the cached startup errors: joined with a workflow
    environment stamp's node->pack mapping, a missing node type resolves
    to "pack X failed to load: <error>" instead of "unknown node"."""
    state = request.app[STATE_KEY]
    wire: dict[str, object] = {"epoch": state.schema_epoch}
    if state.composition_progress is not None:
        wire["composing"] = True
        wire["progress"] = dict(state.composition_progress)
    wire["packs"] = {pack: dict(entry) for pack, entry in state.composition_packs.items()}
    return web.json_response(wire)


async def handle_workers(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    return web.json_response({"workers": [worker.to_wire() for worker in state.workers()]})


async def handle_nodes(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    nodes: dict[str, dict[str, object]] = {}
    documented_nodes: set[tuple[str, str]] = set()
    documented_nodes = {
        (pack_id, page.id)
        for pack_id in state.packs
        for page in _valid_doc_pages(state, pack_id)
        if page.kind == "node"
    }
    for type_id, schema in state.schemas.items():
        entry = schema_to_wire(schema, replacement_schemas=state.schemas)
        # Attribution rides the publication, never the schema: the host
        # attaches pack ids at collection, so a node cannot claim a pack.
        # The key is always present ("core" included) by frontend contract.
        pack_id = state.node_packs.get(type_id, CORE_PACK_ID)
        entry["pack"] = pack_id
        if (pack_id, type_id) in documented_nodes:
            entry["hasDocs"] = True
        entry["executionArms"] = list(state.execution_arms[type_id])
        # Interface identity for environment stamping and drift diagnostics:
        # ALWAYS present (frontend contract - stamping never has holes) and
        # opaque (clients compare for equality, never parse). Transport
        # metadata computed FROM the schema wire, never part of it - it
        # rides the entry alongside "pack", outside schema_to_wire, so the
        # signature can never feed back into its own input.
        entry["signature"] = schema_signature(schema)
        nodes[type_id] = entry
    wire: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        # Surface epoch (settled contract with the frontend, distinct
        # from schemaVersion = API surface version above; the schema
        # WIRE version rides dinkster.schemaWire and each node entry, never
        # this top-level field): a monotonic
        # per-engine-process generation counter for the composed node
        # surface. Starts at 1 (the surface create_app was built with);
        # every ServerState.announce - progressive startup today,
        # hot-reload/live activation later - bumps it and emits
        # {"type": "schema_changed", "epoch": N} on /api/events, always
        # AFTER /api/nodes serves the new surface, so a client holding
        # epoch >= N may skip the refetch.
        "epoch": state.schema_epoch,
        "extensionSnapshotDigest": state.engine.extension_snapshot_digest,
        # Environment header: authoritative runtime facts clients stamp
        # into saved workflows (advisory record, never load/execution
        # identity). schemaWire is the schema encoding epoch every node
        # entry below also carries per-entry. graphFeatures advertises
        # additive graph/job DOCUMENT wire capabilities (decoupled from the
        # schema wire, joint contract 2026-07-25): clients gate optional
        # emission forms on membership, never on version comparison.
        # mergeableTypes (joint contract 2026-07-26, additive - no wire
        # bump): the sorted atom type ids with a registered batch-merge
        # provider, derived from the composed registry at envelope build
        # time so compat packs contribute automatically. The frontend
        # gates the scalar-T multi-select merge arm of asset widgets on
        # membership (list<asset<T>> -> T needs a merge provider; the
        # coercion planner enforces server-side either way).
        "dinkster": {
            "version": _dinkster_version(),
            "schemaWire": SCHEMA_WIRE_VERSION,
            "graphFeatures": ["typedLiteral", "decimalInt", "regions", "placement"],
            "mergeableTypes": sorted(
                type_id
                for type_id in state.engine.registry.type_ids()
                if state.engine.registry.batch_merge_for(type_id) is not None
            ),
        },
        "packs": {
            pack_id: info.to_wire(
                schemas=state.schemas,
                # Keep the stored registries complete for provider reloads, but
                # publish only carriers this response actually attributes here.
                published_types=frozenset(
                    type_id for type_id, entry in nodes.items() if entry["pack"] == pack_id
                ),
            )
            for pack_id, info in state.packs.items()
        },
        "nodes": nodes,
    }
    # Present exactly while the host is still announcing packs: this table
    # is real but not final - expect more epochs, then a
    # composition_complete event naming the last one. Omitted (never
    # false) once the surface is fully composed, so table-in-hand clients
    # can tell "loading finished" positively at fetch time.
    if state.composition_progress is not None:
        wire["composing"] = True
    return web.json_response(wire)


async def handle_extension_snapshot(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    return web.Response(
        text=canonical_extension_snapshot(state.engine.extension_snapshot),
        content_type="application/json",
    )


async def handle_diagnostics(request: web.Request) -> web.Response:
    """Instance-level schema diagnostics: problems only visible with the
    complete schema mapping plus classified compat translation skips.
    Advisory, mirroring the frontend's stance: diagnostics render and force
    review, they never make anything unloadable."""
    state = request.app[STATE_KEY]
    return web.json_response(
        {
            "replacementProblems": [
                problem_to_wire(problem) for problem in state.replacement_problems
            ],
            "compatSkips": [
                {
                    "packId": pack_id,
                    "nodeId": node_id,
                    **diagnostic.to_wire(),
                    "schemaEpoch": state.schema_epoch,
                    "extensionSnapshotDigest": state.engine.extension_snapshot_digest,
                }
                for (pack_id, node_id), diagnostic in sorted(state.compat_skips.items())
            ],
        }
    )


async def handle_choices(request: web.Request) -> web.Response:
    """One combo choice list as a JSON array of strings - the remote half
    of the wire-v9 ComboWidget contract: a successful fetch REPLACES the
    schema's baked static options, an unknown id is a plain 404. Choice
    ids are single namespaced tokens (``comfy.samplers``), so an ordinary
    path variable carries them. Static lists are enumerated at worker
    startup and change only on pack add/reload/remove; lazy lists run the
    owning worker's provider on every fetch. The response is deliberately
    uncached: refresh cadence is frontend policy."""
    state = request.app[STATE_KEY]
    choice_id = request.match_info["choice_id"]
    values: Sequence[str] | None = state.choices.get(choice_id)
    if values is None:
        fetcher = state.lazy_choices.get(choice_id)
        if fetcher is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "no such choice list"}),
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        # Lazy list: run the owning worker's provider, exactly once per
        # fetch, under the server-owned timeout. Provider failures are the
        # owner's fault (502), a hung provider is a timeout (504), and a
        # dead owner is unavailability (503) - all uncached JSON, so a
        # refresh retries against live state.
        try:
            values = tuple(await asyncio.wait_for(fetcher(), LAZY_CHOICE_TIMEOUT_SECONDS))
        except TimeoutError:
            raise web.HTTPGatewayTimeout(
                text=json.dumps({"error": f"choice list {choice_id!r} timed out"}),
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            ) from None
        except ChoiceOwnerGone:
            raise web.HTTPServiceUnavailable(
                text=json.dumps({"error": f"choice list {choice_id!r} owner is not connected"}),
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            ) from None
        except Exception as exc:
            raise web.HTTPBadGateway(
                text=json.dumps({"error": f"choice list {choice_id!r} provider failed: {exc}"}),
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            ) from None
    try:
        body = combo_choices_json_bytes(
            values,
            subject=f"HTTP choice {choice_id!r}",
        )
    except ValueError as exc:
        raise web.HTTPBadGateway(
            text=json.dumps({"error": f"choice list {choice_id!r} is invalid: {exc}"}),
            content_type="application/json",
            headers={"Cache-Control": "no-store"},
        ) from None
    return web.Response(
        body=body,
        content_type="application/json",
        charset="utf-8",
        headers={"Cache-Control": "no-store"},
    )


async def handle_pack_icon(request: web.Request) -> web.Response:
    """Raster badge bytes for one packs-table entry. Exists exactly for
    packs whose /api/nodes entry declares an icon - the descriptor's
    presence is the client's only signal, so unknown packs and packs
    without icons are both plain 404s (no probing contract). Bytes for a
    digest are immutable, hence the rendition-style caching: quoted digest
    ETag, If-None-Match -> 304, and a forever cache lifetime."""
    state = request.app[STATE_KEY]
    info = state.packs.get(request.match_info["pack_id"])
    if info is None or info.icon is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack icon"}),
            content_type="application/json",
        )
    icon = info.icon
    etag = f'"{icon.digest}"'
    if any(tag.value in (icon.digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.Response(
        body=icon.data,
        content_type=icon.media_type,
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


async def handle_pack_static(request: web.Request) -> web.Response:
    """Serve one exact immutable file from a pack's validated asset tree."""
    info = request.app[STATE_KEY].packs.get(request.match_info["pack_id"])
    relative = request.match_info["path"]
    parts = relative.replace("\\", "/").split("/")
    asset = None
    if info is not None and relative and all(part not in ("", ".", "..") for part in parts):
        normalized = "/".join(parts)
        asset = next((item for item in info.frontend_assets if item.path == normalized), None)
    if asset is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack static asset"}),
            content_type="application/json",
        )
    return web.Response(
        body=asset.data,
        content_type=asset.media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


def _pack_settings_info(request: web.Request) -> tuple[str, PackInfo, PackSettingsSchema]:
    pack_id = request.match_info["pack_id"]
    info = request.app[STATE_KEY].packs.get(pack_id)
    if info is None or info.settings_schema is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack settings"}),
            content_type="application/json",
        )
    return pack_id, info, info.settings_schema


def _pack_settings_wire(
    pack_id: str, info: PackInfo, schema: PackSettingsSchema, values: Mapping[str, object]
) -> dict[str, object]:
    return {
        "packId": pack_id,
        "displayName": info.display_name,
        "schema": schema.to_wire(),
        "values": dict(values),
    }


async def handle_pack_settings_get(request: web.Request) -> web.Response:
    pack_id, info, schema = _pack_settings_info(request)
    try:
        values = await asyncio.to_thread(request.app[PACK_SETTINGS_KEY].read, pack_id, schema)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(_pack_settings_wire(pack_id, info, schema, values))


async def handle_pack_settings_put(request: web.Request) -> web.Response:
    pack_id, info, schema = _pack_settings_info(request)
    try:
        body = await request.json()
        if not isinstance(body, Mapping):
            raise ValueError("pack settings must be a JSON object")
        values = await asyncio.to_thread(
            request.app[PACK_SETTINGS_KEY].write,
            pack_id,
            schema,
            cast(Mapping[str, object], body),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError as exc:
        return web.json_response(
            {"error": f"pack settings could not be persisted: {exc}"}, status=500
        )
    return web.json_response(_pack_settings_wire(pack_id, info, schema, values))


async def handle_pack_blueprint(request: web.Request) -> web.Response:
    """Blueprint document bytes for one packs-table descriptor. Copies the
    icon endpoint's contract verbatim: the descriptor's presence in
    /api/nodes is the client's only signal, so unknown packs and unknown
    blueprint ids are both plain 404s (no probing contract). Bytes for a
    digest are immutable - a changed blueprint is a new digest in the
    table - hence the rendition-style caching: quoted digest ETag,
    If-None-Match -> 304, and a forever cache lifetime."""
    state = request.app[STATE_KEY]
    info = state.packs.get(request.match_info["pack_id"])
    blueprint = None
    if info is not None:
        blueprint_id = request.match_info["blueprint_id"]
        blueprint = next((bp for bp in info.blueprints if bp.id == blueprint_id), None)
    if blueprint is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack blueprint"}),
            content_type="application/json",
        )
    etag = f'"{blueprint.digest}"'
    if any(tag.value in (blueprint.digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.Response(
        body=blueprint.data,
        content_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


async def handle_pack_template(request: web.Request) -> web.Response:
    """Template document bytes for one /api/templates descriptor. Copies
    the blueprint endpoint's contract verbatim: unknown packs and unknown
    template ids are both plain 404s (no probing contract), and bytes for
    a digest are immutable - a changed template is a new digest in the
    index - hence the rendition-style caching: quoted digest ETag,
    If-None-Match -> 304, and a forever cache lifetime."""
    state = request.app[STATE_KEY]
    info = state.packs.get(request.match_info["pack_id"])
    template = None
    if info is not None:
        template_id = request.match_info["template_id"]
        template = next((tp for tp in info.templates if tp.id == template_id), None)
    if template is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack template"}),
            content_type="application/json",
        )
    etag = f'"{template.digest}"'
    if any(tag.value in (template.digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.Response(
        body=template.data,
        content_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


async def handle_pack_template_thumbnail(request: web.Request) -> web.Response:
    """Immutable thumbnail bytes for one template descriptor."""
    state = request.app[STATE_KEY]
    info = state.packs.get(request.match_info["pack_id"])
    template = None
    if info is not None:
        template_id = request.match_info["template_id"]
        template = next((tp for tp in info.templates if tp.id == template_id), None)
    if template is None or template.thumbnail is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such template thumbnail"}),
            content_type="application/json",
        )
    return _immutable_pack_response(
        request,
        template.thumbnail.digest,
        template.thumbnail.data,
        template.thumbnail.media_type,
    )


def _valid_doc_pages(state: ServerState, pack_id: str) -> tuple[PackDocPageAsset, ...]:
    info = state.packs.get(pack_id)
    if info is None or info.docs is None:
        return ()
    return tuple(
        page
        for page in info.docs.pages
        if page.kind != "node"
        or (page.id in state.schemas and state.node_packs.get(page.id, CORE_PACK_ID) == pack_id)
    )


def _doc_descriptor(
    pack_id: str, default_locale: str, pages: Sequence[PackDocPageAsset]
) -> dict[str, object]:
    first = pages[0]
    descriptor: dict[str, object] = {
        "pack": pack_id,
        "kind": first.kind,
        "id": first.id,
        "defaultLocale": default_locale,
    }
    if first.kind == "guide":
        if first.order is not None:
            descriptor["order"] = first.order
        if first.tags:
            descriptor["tags"] = list(first.tags)
        if first.guide_kind is not None:
            descriptor["guideKind"] = first.guide_kind
    locales: dict[str, object] = {}
    for page in sorted(pages, key=lambda item: item.locale):
        locale: dict[str, object] = {
            "title": page.title,
            "summary": page.summary,
            "digest": page.digest,
            "assets": {asset.source: asset.descriptor() for asset in page.assets},
        }
        if page.kind == "node" and page.schema_version is not None:
            locale["schemaVersion"] = page.schema_version
        locales[page.locale] = locale
    descriptor["locales"] = locales
    return descriptor


def _doc_matches(
    descriptor: Mapping[str, object], text: str, kind: str, pack: str, content_id: str
) -> bool:
    if kind and descriptor["kind"] != kind:
        return False
    if pack and descriptor["pack"] != pack:
        return False
    if content_id and descriptor["id"] != content_id:
        return False
    if not text:
        return True
    needle = text.lower()
    values = [str(descriptor["id"])]
    tags = descriptor.get("tags")
    if isinstance(tags, list):
        values.extend(str(tag) for tag in cast("list[object]", tags))
    for locale in cast("Mapping[str, Mapping[str, object]]", descriptor["locales"]).values():
        values.extend((str(locale["title"]), str(locale["summary"])))
    return any(needle in value.lower() for value in values)


async def handle_docs_list(request: web.Request) -> web.Response:
    """Paged descriptors for pack-shipped node pages and guides."""
    state = request.app[STATE_KEY]
    text = request.query.get("q", "")
    kind = request.query.get("kind", "")
    pack_filter = request.query.get("pack", "")
    content_id = request.query.get("id", "")
    if kind not in ("", "node", "guide"):
        raise _bad_request("'kind' must be 'node' or 'guide'")
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        raise _bad_request("'limit' must be an integer") from None
    bound = {"q": text, "k": kind, "p": pack_filter, "i": content_id}
    after_id = ""
    cursor = request.query.get("cursor")
    if cursor:
        after_id = decode_cursor(cursor, bound)[1]

    rows: list[tuple[str, dict[str, object]]] = []
    for pack_id, info in state.packs.items():
        if info.docs is None:
            continue
        grouped: dict[tuple[str, str], list[PackDocPageAsset]] = {}
        for page in _valid_doc_pages(state, pack_id):
            grouped.setdefault((page.kind, page.id), []).append(page)
        for (page_kind, page_id), pages in grouped.items():
            qualified = f"{pack_id}/{page_kind}/{page_id}"
            if after_id and qualified <= after_id:
                continue
            descriptor = _doc_descriptor(pack_id, info.docs.default_locale, pages)
            if _doc_matches(descriptor, text, kind, pack_filter, content_id):
                rows.append((qualified, descriptor))
    rows.sort(key=lambda row: row[0])
    wire: dict[str, object] = {"docs": [descriptor for _, descriptor in rows[:limit]]}
    if len(rows) > limit:
        wire["cursor"] = encode_cursor(bound, (0.0, rows[limit - 1][0]))
    return web.json_response(wire)


def _doc_by_digest(
    state: ServerState, pack_id: str, digest: str, *, asset: bool
) -> PackDocAsset | PackDocPageAsset | None:
    pages = _valid_doc_pages(state, pack_id)
    if asset:
        return next(
            (item for page in pages for item in page.assets if item.digest == digest),
            None,
        )
    return next((page for page in pages if page.digest == digest), None)


def _immutable_pack_response(
    request: web.Request, digest: str, data: bytes, media_type: str
) -> web.Response:
    etag = f'"{digest}"'
    if any(tag.value in (digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.Response(
        body=data,
        content_type=media_type,
        headers={"ETag": etag, "Cache-Control": "private, max-age=31536000, immutable"},
    )


async def handle_pack_doc_page(request: web.Request) -> web.Response:
    page = _doc_by_digest(
        request.app[STATE_KEY],
        request.match_info["pack_id"],
        request.match_info["digest"],
        asset=False,
    )
    if not isinstance(page, PackDocPageAsset):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack doc page"}),
            content_type="application/json",
        )
    return _immutable_pack_response(request, page.digest, page.data, "text/markdown")


async def handle_pack_doc_asset(request: web.Request) -> web.Response:
    asset = _doc_by_digest(
        request.app[STATE_KEY],
        request.match_info["pack_id"],
        request.match_info["digest"],
        asset=True,
    )
    if not isinstance(asset, PackDocAsset):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack doc asset"}),
            content_type="application/json",
        )
    return _immutable_pack_response(request, asset.digest, asset.data, asset.media_type)


def _valid_locale_catalogs(state: ServerState, pack_id: str) -> tuple[PackLocaleCatalogAsset, ...]:
    info = state.packs.get(pack_id)
    if info is None:
        return ()
    return tuple(
        catalog
        for catalog in info.locale_catalogs
        if all(
            node_type in state.schemas and state.node_packs.get(node_type, CORE_PACK_ID) == pack_id
            for node_type in catalog.node_references
        )
    )


async def handle_pack_locale_catalog(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    catalog = next(
        (
            item
            for item in _valid_locale_catalogs(
                request.app[STATE_KEY], request.match_info["pack_id"]
            )
            if item.digest == digest
        ),
        None,
    )
    if catalog is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such pack locale catalog"}),
            content_type="application/json",
        )
    return _immutable_pack_response(request, catalog.digest, catalog.data, "application/json")


def _template_matches(template: PackTemplateAsset, text: str, tag_filter: str) -> bool:
    if tag_filter and tag_filter not in template.tags:
        return False
    if text:
        needle = text.lower()
        hay = (
            template.id,
            template.name,
            template.description,
            *template.tags,
        )
        if not any(needle in value.lower() for value in hay):
            return False
    return True


async def handle_templates_list(request: web.Request) -> web.Response:
    """Query-first paged template index over every pack's shipped
    templates. The frontend collection contract: listings own their
    query (q= substring over id/name/description/tags, tag= exact tag,
    pack= exact pack id), the cursor BINDS the query that minted it
    (mismatch is a loud 400 via decode_cursor), and descriptors stay
    kilobyte-scale - bodies ride the lazy per-pack endpoint. Order is
    deterministic (pack id, then template id); the keyset cursor carries
    the last row's qualified id, so pages stay stable as long as the
    composition does. Asset requirements ride as pack-local ids that
    clients join against the packs table's asset descriptors."""
    state = request.app[STATE_KEY]
    text = request.query.get("q", "")
    tag_filter = request.query.get("tag", "")
    pack_filter = request.query.get("pack", "")
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "limit must be an integer"}),
            content_type="application/json",
        ) from None
    bound = {"q": text, "t": tag_filter, "p": pack_filter}
    after_id = ""
    cursor = request.query.get("cursor")
    if cursor:
        after_id = decode_cursor(cursor, bound)[1]

    # Collect-then-sort over the qualified "pack/template" id: the SAME
    # key the cursor carries, so keyset resumption can never skip or
    # repeat a row (iterating packs in dict-sorted order would disagree
    # with qualified-string order for pack ids containing '-' or '.').
    # The collection is composition-bounded and descriptors are tiny, so
    # materializing the filtered set is cheap.
    rows: list[tuple[str, str, PackTemplateAsset]] = []
    for pack_id, info in state.packs.items():
        if pack_filter and pack_id != pack_filter:
            continue
        for template in info.templates:
            if not _template_matches(template, text, tag_filter):
                continue
            qualified = f"{pack_id}/{template.id}"
            if after_id and qualified <= after_id:
                continue
            rows.append((qualified, pack_id, template))
    rows.sort(key=lambda row: row[0])

    wire: dict[str, object] = {
        "templates": [template.descriptor(pack_id) for _, pack_id, template in rows[:limit]]
    }
    if len(rows) > limit:
        wire["cursor"] = encode_cursor(bound, (0.0, rows[limit - 1][0]))
    return web.json_response(wire)


def _parse_previews(
    raw: object, *, default_mode: str, default_animation: str = "ring"
) -> PreviewPolicy | None | str:
    """Parse the submit body's optional 'previews' field into a policy.

    An absent field falls back to the server's default preview mode
    ("off" parses to no policy at all, so zero plumbing runs). A present
    field must be an object: {"mode": <mode>, "nodes": {nodeId: <mode>},
    "animation": <transport>} with modes drawn from off/cheap/quality/auto
    and the optional animation transport from ring/encoded (absent falls
    back to the server default). Returns an error string for a malformed
    field; node ids are syntax-checked only, exactly like placement - an
    id absent from the graph simply never matches."""
    if raw is None:
        if default_mode == "off":
            return None
        return PreviewPolicy(
            mode=validate_preview_mode(default_mode),
            animation=validate_preview_animation(default_animation),
        )
    if not isinstance(raw, Mapping):
        return (
            "'previews' must be an object with 'mode', optional 'nodes', and optional 'animation'"
        )
    body = cast("Mapping[object, object]", raw)
    if not set(body) <= {"mode", "nodes", "animation"}:
        return "'previews' accepts only 'mode', 'nodes', and 'animation'"
    try:
        mode = validate_preview_mode(body.get("mode", "off"))
        animation = validate_preview_animation(body.get("animation", default_animation))
    except ValueError as exc:
        return f"'previews' is invalid: {exc}"
    raw_nodes = body.get("nodes", {})
    if not isinstance(raw_nodes, Mapping):
        return "'previews' 'nodes' must be an object mapping node ids to modes"
    node_modes: dict[str, str] = {}
    for node_id, node_mode in cast("Mapping[object, object]", raw_nodes).items():
        if not isinstance(node_id, str) or not node_id:
            return "'previews' 'nodes' keys must be non-empty node id strings"
        try:
            node_modes[node_id] = validate_preview_mode(node_mode)
        except ValueError as exc:
            return f"'previews' node {node_id!r} is invalid: {exc}"
    try:
        return PreviewPolicy(
            mode=mode, node_modes=cast("Mapping[str, Any]", node_modes), animation=animation
        )
    except ValueError as exc:
        return f"'previews' is invalid: {exc}"


async def handle_submit(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    body = await _json_body(request)
    principal = principal_for(request)
    scope = resolve_scope(
        principal,
        "jobs:submit",
        None if principal.local else body.get("scope"),
    )
    client_id = body.get("clientId")
    job_id = body.get("jobId")
    if not isinstance(client_id, str) or not client_id:
        raise _bad_request("'clientId' must be a non-empty string")
    if not isinstance(job_id, str) or not job_id:
        raise _bad_request("'jobId' must be a non-empty string")
    raw_targets = body.get("targets")
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or not all(isinstance(t, str) for t in cast(list[Any], raw_targets))
    ):
        raise _bad_request("'targets' must be a non-empty list of node ids")
    targets = cast(list[str], raw_targets)
    priority = body.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise _bad_request("'priority' must be an integer")
    try:
        submitted_attention = (
            attention_policy_config_from_wire(body["attention"]) if "attention" in body else None
        )
    except (TypeError, ValueError) as exc:
        raise _bad_request(f"'attention' is invalid: {exc}") from exc
    attention_config = submitted_attention or state.attention_default
    # Execution-opaque provenance (frontend contract 2026-07): the canonical
    # asset digest of the workflow document this job was compiled from.
    # Syntax-checked only - it never joins execution or cache identity, so
    # whether the asset exists is not this endpoint's business.
    source_document = body.get("sourceDocument", "")
    if not isinstance(source_document, str) or (source_document and not is_digest(source_document)):
        raise _bad_request("'sourceDocument' must be a canonical asset digest ('blake3:<64 hex>')")
    try:
        graph = graph_from_wire(body.get("graph"))
    except GraphWireError as exc:
        raise _bad_request(str(exc)) from exc
    raw_placement = body.get("placement", {})
    if not isinstance(raw_placement, Mapping):
        raise _bad_request("'placement' must be an object mapping node ids to worker names")
    if any(
        not isinstance(node_id, str) or not node_id or not isinstance(worker, str) or not worker
        for node_id, worker in cast("Mapping[object, object]", raw_placement).items()
    ):
        raise _bad_request("'placement' keys and values must be non-empty strings")
    placement = dict(cast("Mapping[str, str]", raw_placement))
    raw_previews = body.get("previews")
    if "previews" in body and raw_previews is None:
        raise _bad_request(
            "'previews' must be an object with 'mode', optional 'nodes', and optional 'animation'"
        )
    previews = _parse_previews(
        raw_previews,
        default_mode=state.preview_default,
        default_animation=state.preview_animation,
    )
    if isinstance(previews, str):
        raise _bad_request(previews)
    library = request.app.get(LIBRARY_KEY)
    parsed = parse_asset_consent(body)
    if isinstance(parsed, str):
        raise _bad_request(parsed)
    consented, hinted_sources = parsed
    fingerprint_payload: dict[str, object] = {
        "graph": body.get("graph"),
        "targets": targets,
        "priority": priority,
        "scope": scope,
        "sourceDocument": source_document,
    }
    if placement:
        # Placement is opaque to graph/document and node cache identity, but
        # it is content of the active job idempotency request. Reusing one
        # client/job key with different requested workers must conflict
        # instead of returning the first placement as a duplicate.
        fingerprint_payload["placement"] = placement
    if raw_previews is not None and previews is not None:
        # An explicitly requested preview policy is likewise idempotency
        # content, never cache identity: resubmitting one job key with a
        # different policy must conflict instead of silently reusing the
        # first. The server default applied to an absent field stays out
        # of the fingerprint, exactly like other absent optional fields.
        fingerprint_payload["previews"] = {
            "mode": previews.mode,
            "nodes": dict(previews.node_modes),
            "animation": previews.animation,
        }
    if submitted_attention is not None and submitted_attention != state.attention_default:
        fingerprint_payload["attention"] = attention_policy_config_to_wire(submitted_attention)
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    async with state.job_admission(client_id, job_id):
        existing = state.queue.get(client_id, job_id)
        if existing is not None and existing.state not in TERMINAL_STATES:
            if existing.fingerprint != fingerprint:
                return web.json_response(
                    {
                        "error": "job-key-in-use",
                        "message": (
                            f"job key {existing.key} is in use by {existing.state} job "
                            "with different content"
                        ),
                        "clientId": client_id,
                        "jobId": job_id,
                    },
                    status=409,
                )
            wire = job_to_wire(
                existing,
                None,
                state.engine.registry,
            )
            wire["duplicate"] = True
            return web.json_response(wire, status=202)
        execution = state.engine.pin_execution()
        assert execution.schemas is not None
        graph = migrate_pure_node_type_replacements(graph, execution.schemas)
        lowered = lower_selectors(graph, targets, execution.schemas)
        if lowered.problems:
            return web.json_response(
                {
                    "error": "selector-lowering",
                    "problems": [
                        {
                            "code": p.code,
                            "message": p.message,
                            "nodeId": p.node_id,
                            **({"inputId": p.input_id} if p.input_id else {}),
                        }
                        for p in lowered.problems
                    ],
                },
                status=400,
            )
        try:
            state.validate_placement(
                lowered.graph,
                placement,
                submitted_graph=graph,
            )
        except ValueError as exc:
            raise _bad_request(str(exc)) from exc
        graph = lowered.graph
        targets = list(lowered.targets)
        execution = state.place_execution(execution, placement)
        execution_graph = graph
        if execution.resolve_providers is not None:
            try:
                execution_graph = execution.resolve_providers(execution_graph)
            except ProviderResolutionError as exc:
                return _provider_resolution_response(exc)
            except ValueError as exc:
                return web.json_response(
                    {"error": "vision-provider-unavailable", "message": str(exc)},
                    status=400,
                )
        compiled_graph = None
        if execution.graph_compiler_registry.contributions:
            try:
                compiled_graph = await state.engine.compile_for_execution(
                    execution_graph,
                    targets,
                    execution=execution,
                )
            except GraphCompileError as exc:
                return web.json_response(
                    {"error": exc.code, "message": str(exc)},
                    status=400,
                )
            preflight_graph = compiled_graph.graph
            if execution.resolve_providers is not None:
                try:
                    preflight_graph = execution.resolve_providers(preflight_graph)
                except ProviderResolutionError as exc:
                    return _provider_resolution_response(exc)
                except ValueError as exc:
                    return web.json_response(
                        {"error": "vision-provider-unavailable", "message": str(exc)},
                        status=400,
                    )
                compiled_graph = CompiledGraph(
                    preflight_graph,
                    compiled_graph.targets,
                    compiled_graph.extension_snapshot_digest,
                    compiled_graph.origins,
                )
            targets = list(compiled_graph.targets)
        else:
            preflight_graph = execution_graph
        remote_provider_digests = frozenset[str]()
        remote_provider_plan: list[dict[str, object]] = []
        if execution.preflight_provider_assets is not None:
            (
                remote_provider_digests,
                remote_provider_plan,
            ) = await execution.preflight_provider_assets(
                preflight_graph, frozenset(consented), hinted_sources
            )
        remote_missing = {
            digest
            for entry in remote_provider_plan
            if isinstance((digest := entry.get("digest")), str)
        }
        local_plan: list[dict[str, object]] = []
        # Asset preflight (roadmap "templates/asset distribution"): with a
        # library configured, every asset identity the graph references must
        # be locally materializable before the job queues. Missing identities
        # answer 409 with a machine-readable acquisition plan; the caller
        # consents per digest via 'acquireAssets' (never a blanket boolean),
        # optionally supplying its own leads via 'assetSources'. Digest-exact
        # consent is TOCTOU-proof: verified ingest means a consented identity
        # can only ever land the exact bytes it names.
        if library is not None:
            referenced = graph_asset_names(preflight_graph)
            # Node-declared requirements ([[pack.assets]] nodes = [...]): a
            # fixed internal model a node type needs joins the preflight set
            # the moment the graph instantiates that type - no widget, no
            # dropdown, and no silent download during execution. Graph
            # literals win the display name when both reference a digest.
            if library.pack_assets is not None:
                catalog = library.pack_assets
                provider_selections, linked_provider_nodes = graph_provider_selections(
                    preflight_graph
                )
                declared = catalog.needs_for_nodes(
                    graph_node_types(preflight_graph),
                    provider_selections=provider_selections,
                    linked_provider_nodes=linked_provider_nodes,
                )
                for digest, need in declared.items():
                    if digest in remote_provider_digests and digest not in remote_missing:
                        continue
                    referenced.setdefault(digest, need.name)
            if referenced:
                local_plan = await asyncio.to_thread(
                    asset_preflight,
                    library,
                    referenced,
                    consented,
                    hinted_sources,
                )
        if (
            execution.preflight_provider_assets is not None
            and remote_provider_plan
            and any(digest in consented for digest in remote_missing)
        ):
            _digests, remote_provider_plan = await execution.preflight_provider_assets(
                preflight_graph, frozenset(consented), hinted_sources
            )
        missing_assets: list[dict[str, object]] = []
        seen_missing: set[str] = set()
        for entry in [*local_plan, *remote_provider_plan]:
            digest = entry.get("digest")
            if isinstance(digest, str) and digest in seen_missing:
                continue
            if isinstance(digest, str):
                seen_missing.add(digest)
            missing_assets.append(entry)
        if missing_assets:
            return web.json_response(
                {"error": "assets-missing", "assets": missing_assets}, status=409
            )
        try:
            job = state.queue.submit(
                client_id,
                job_id,
                graph,
                targets,
                priority=priority,
                scope=scope,
                principal_id=principal.principal_id,
                principal_kind=principal.kind,
                source_document=source_document,
                fingerprint=fingerprint,
                execution=execution,
                compiled_graph=compiled_graph,
                previews=previews,
                attention_config=attention_config,
            )
        except JobGraphAdmissionError as exc:
            return web.json_response(
                {"error": "selector-admission", "message": str(exc)},
                status=400,
            )
        return web.json_response(
            job_to_wire(job, None, state.engine.registry),
            status=202,
        )


async def handle_job_status(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    job = state.queue.get(request.match_info["client_id"], request.match_info["job_id"])
    if job is None or not principal_for(request).allows_in(job.scope, "jobs:read"):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such job"}), content_type="application/json"
        )
    return web.json_response(
        job_to_wire(
            job,
            state.node_states(job.run_id),
            state.engine.registry,
        )
    )


def _job_by_ref(
    state: ServerState,
    job_ref: str,
    *,
    principal: Principal | None = None,
    preserve_keyed_route: bool = False,
) -> Job:
    job = state.queue.job_for_run(job_ref)
    if job is None and preserve_keyed_route:
        # `/api/jobs/by-ref/{jobRef}` overlaps the pre-existing keyed route
        # for clientId="by-ref". Preserve those legacy keys when the segment
        # does not resolve as a global reference.
        job = state.queue.get("by-ref", job_ref)
    if job is None or (principal is not None and not principal.allows_in(job.scope, "jobs:read")):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such job"}), content_type="application/json"
        )
    return job


async def handle_job_status_by_ref(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    job = _job_by_ref(
        state,
        request.match_info["job_ref"],
        principal=principal_for(request),
        preserve_keyed_route=True,
    )
    return web.json_response(
        job_to_wire(
            job,
            state.node_states(job.run_id),
            state.engine.registry,
        )
    )


async def handle_job_events(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    job = _job_by_ref(state, request.match_info["job_ref"], principal=principal_for(request))
    raw_after = request.query.get("after", "0")
    if not raw_after.isdigit():
        return web.json_response({"error": "after must be a non-negative integer"}, status=400)
    after = int(raw_after)
    events = state.replay_events(job.run_id, after)
    if events is None:
        return web.json_response({"error": "resync-required"}, status=410)
    return web.json_response({"events": events, "latestSeq": state.latest_seq(job.run_id)})


def _value_unavailable(status: int, reason: str, message: str) -> web.Response:
    """The peek contract's safety half (DESIGN 3.5): a value that cannot be
    served resolves to a structured refusal - never silently to a
    different/newer value."""
    return web.json_response(
        {"available": False, "reason": reason, "error": message}, status=status
    )


def _rendition_problem(status: int, reason: str, message: str) -> web.Response:
    return web.json_response(
        {"available": False, "reason": reason, "error": message},
        status=status,
        content_type="application/problem+json",
    )


async def handle_value(request: web.Request) -> web.Response:
    """GET /api/values - completed-run output value retrieval (DESIGN 3.5).

    Query: clientId, jobId, nodeId (runtime id, region iteration paths
    included), outputId; optional element (comma-separated list indices),
    rendition (registered kind, or "default"). Parameterized renditions
    advertise their accepted selectors, defaults, and limits in discovery.

    Lookup authority is execution-scoped identity (clientId/jobId/nodeId/
    outputId); fingerprints are cache-validation tags (the ETag), not the
    lookup key. Target outputs come from the retained job result;
    intermediates resolve through the cache key the run's own events
    reported - which is content-derived, so it can only yield the exact
    value this run saw, or nothing."""
    state = request.app[STATE_KEY]
    query = request.rel_url.query
    missing = [p for p in ("clientId", "jobId", "nodeId", "outputId") if p not in query]
    if missing:
        raise _bad_request(f"missing query parameter(s): {', '.join(missing)}")
    node_id = query["nodeId"]
    output_id = query["outputId"]

    job = state.queue.get(query["clientId"], query["jobId"])
    if job is None or not principal_for(request).allows_in(job.scope, "jobs:read"):
        return _value_unavailable(404, "unknown-job", "no such job")
    if job.state not in TERMINAL_STATES:
        # Frozen-view contract: peek serves finished runs. Live values ride
        # the event stream; serving a mid-run snapshot here would blur the
        # "exact value this run saw" guarantee while states still mutate.
        return _value_unavailable(
            409, "not-complete", f"job is {job.state}; values are served once terminal"
        )

    outputs: Mapping[str, Value] | None = None
    if job.result is not None and node_id in job.result.outputs:
        outputs = job.result.outputs[node_id]
    else:
        cache_key = state.run_cache_key(job.run_id, node_id)
        if cache_key is None:
            return _value_unavailable(
                404,
                "not-retained",
                f"run has no retained outputs for node {node_id!r}",
            )
        outputs = await state.engine.cache.get(cache_key)
        if outputs is None:
            return _value_unavailable(410, "evicted", f"outputs of node {node_id!r} were evicted")
    if output_id not in outputs:
        return _value_unavailable(
            404, "unknown-output", f"node {node_id!r} has no output {output_id!r}"
        )
    value = outputs[output_id]

    for index_text in [e for e in query.get("element", "").split(",") if e]:
        try:
            index = int(index_text)
        except ValueError:
            # Malformed query, not a data condition: 400 regardless of what
            # the value turned out to be.
            raise _bad_request(f"element indices must be integers: {index_text!r}") from None
        children = list_children(value)
        if children is None:
            return _value_unavailable(404, "bad-element", f"{value.type_id} is not a list")
        if not 0 <= index < len(children):
            return _value_unavailable(
                404,
                "bad-element",
                f"element {index} out of range (length {len(children)})",
            )
        value = children[index]

    registry = state.engine.registry
    kind = query.get("rendition")
    renditions: list[tuple[Any, str]] = []
    for rendition_spec in registry.renditions_of(value.type_id):
        try:
            mime = await registry.rendition_mime(rendition_spec, value.meta.entries)
        except ValueError:
            continue
        renditions.append((rendition_spec, mime))
    if kind is None:
        return web.json_response(
            {
                "available": True,
                "descriptor": value_descriptor(value, registry),
                "renditions": [
                    {
                        "kind": s.kind,
                        "mime": mime,
                        "default": s.default,
                        "cacheKey": s.selector,
                        **(
                            {
                                "version": s.version,
                                "parameters": list(s.parameters),
                                "defaults": dict(s.defaults or {}),
                                "limits": dict(s.limits or {}),
                            }
                            if s.parameters
                            else ({"limits": dict(s.limits)} if s.limits is not None else {})
                        ),
                    }
                    for s, mime in renditions
                ],
            }
        )

    spec = next(
        (
            s
            for s, _mime in renditions
            if (s.default if kind == "default" else kind in (s.kind, s.selector))
        ),
        None,
    )
    if spec is None:
        return web.json_response(
            {
                "available": False,
                "reason": "no-rendition",
                "error": f"{value.type_id} has no rendition {kind!r}",
                "renditions": [s.kind for s, _mime in renditions],
            },
            status=406,
        )
    request_parameters = set(query) - {
        "clientId",
        "jobId",
        "nodeId",
        "outputId",
        "element",
        "rendition",
    }
    unsupported = request_parameters - set(spec.parameters)
    if unsupported:
        return _rendition_problem(
            400,
            "invalid_rendition_request",
            f"{spec.kind} rendition does not accept parameter(s): {', '.join(sorted(unsupported))}",
        )
    raw_parameters: dict[str, str] = {}
    for name in spec.parameters:
        values = query.getall(name, [])
        if len(values) > 1:
            return _rendition_problem(
                400, "invalid_rendition_request", f"rendition parameter {name!r} is repeated"
            )
        if values:
            raw_parameters[name] = values[0]
    try:
        _mime, normalized_parameters = await registry.resolve_rendition(
            spec,
            value.meta.entries,
            {**dict(spec.defaults or {}), **raw_parameters},
        )
    except InvalidRenditionRequest as error:
        return _rendition_problem(400, "invalid_rendition_request", str(error))
    except RenditionUnavailable as error:
        return _rendition_problem(404, "rendition_unavailable", str(error))
    # Immutable responses require the version in the URL; ETag alone cannot
    # rotate a fresh immutable cache entry. Kind/default aliases revalidate.
    etag_value = spec.cache_key(value.fingerprint, normalized_parameters)
    cache_control = (
        "private, max-age=31536000, immutable"
        if kind == spec.selector and kind != "default"
        else "private, no-cache"
    )
    if any(tag.value in (etag_value, "*") for tag in (request.if_none_match or ())):
        return web.Response(
            status=304,
            headers={"ETag": f'"{etag_value}"', "Cache-Control": cache_control},
        )
    try:
        rendition = await registry.render_async(value, spec.kind, normalized_parameters or None)
    except InvalidRenditionRequest as error:
        return _rendition_problem(400, "invalid_rendition_request", str(error))
    except RenditionUnavailable as error:
        return _rendition_problem(404, "rendition_unavailable", str(error))
    except AssetError:
        return _rendition_problem(404, "rendition_unavailable", "rendition source is unavailable")
    except UnresolvablePayload:
        return web.json_response(
            {
                "available": False,
                "reason": "type-not-loadable",
                "error": f"{value.type_id} is not loadable in this process "
                "(pack type registered only in an isolated worker)",
            },
            status=406,
        )
    return web.Response(
        body=rendition.data,
        content_type=rendition.mime,
        headers={
            "ETag": f'"{etag_value}"',
            "Cache-Control": cache_control,
            "X-Dinkster-Type-Id": value.type_id,
            "X-Dinkster-Fingerprint": value.fingerprint,
            "X-Dinkster-Rendition": rendition.kind,
        },
    )


async def handle_cancel(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    principal = principal_for(request)
    job = state.queue.get(request.match_info["client_id"], request.match_info["job_id"])
    if job is None or not (
        principal.allows_in(job.scope, "jobs:read")
        and principal.allows_in(job.scope, "jobs:cancel")
    ):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such job"}), content_type="application/json"
        )
    job = state.queue.cancel(job.key.client_id, job.key.job_id)
    assert job is not None
    return web.json_response(
        job_to_wire(
            job,
            state.node_states(job.run_id),
            state.engine.registry,
        )
    )


async def handle_cancel_by_ref(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    job = _job_by_ref(
        state,
        request.match_info["job_ref"],
        principal=principal_for(request),
        preserve_keyed_route=True,
    )
    if not principal_for(request).allows_in(job.scope, "jobs:cancel"):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such job"}), content_type="application/json"
        )
    cancelled = state.queue.cancel(job.key.client_id, job.key.job_id)
    assert cancelled is job
    return web.json_response(
        job_to_wire(
            job,
            state.node_states(job.run_id),
            state.engine.registry,
        )
    )


async def handle_jobs_list(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    client_id = request.query.get("clientId")  # absent -> every client's jobs
    jobs = state.queue.jobs(client_id)
    principal = principal_for(request)
    return web.json_response(
        {
            "jobs": [
                job_to_wire(
                    job,
                    None,
                    state.engine.registry,
                )
                for job in jobs
                if principal.allows_in(job.scope, "jobs:read")
            ]
        }
    )


def _queue_wire(state: ServerState) -> dict[str, object]:
    status = state.queue.status()
    status["queued"] = [
        job_to_wire(
            job,
            None,
            state.engine.registry,
        )
        for job in state.queue.pending()
    ]
    status["running"] = [
        job_to_wire(
            job,
            None,
            state.engine.registry,
        )
        for job in state.queue.running()
    ]
    return status


def _publish_queue_state(state: ServerState) -> None:
    # Control-state changes are rare and load-bearing (a paused queue that
    # looks live is a support ticket): broadcast, never shed.
    wire: dict[str, object] = {"type": "queue_state"}
    wire.update(state.queue.status())
    state.hub.publish(wire, client_id=None, droppable=False)


async def handle_queue_status(request: web.Request) -> web.Response:
    return web.json_response(_queue_wire(request.app[STATE_KEY]))


async def handle_queue_pause(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    state.queue.pause()
    _publish_queue_state(state)
    return web.json_response(state.queue.status())


async def handle_queue_resume(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    try:
        state.queue.resume()
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    _publish_queue_state(state)
    return web.json_response(state.queue.status())


async def handle_queue_clear(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    body = await _json_body(request) if request.can_read_body else {}
    client_id = body.get("clientId")
    if client_id is not None and (not isinstance(client_id, str) or not client_id):
        raise _bad_request("'clientId' must be a non-empty string when present")
    cleared = state.queue.clear(client_id)
    # Each cleared job already emitted its cancelled job_state transition.
    return web.json_response(
        {"cleared": [{"clientId": j.key.client_id, "jobId": j.key.job_id} for j in cleared]}
    )


async def handle_events(request: web.Request) -> web.WebSocketResponse:
    state = request.app[STATE_KEY]
    client_id = request.query.get("clientId")  # absent -> observe all clients
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    def visible(event: Mapping[str, object]) -> bool:
        job = None
        job_ref = event.get("jobRef")
        if isinstance(job_ref, str):
            job = state.queue.job_for_run(job_ref)
        elif isinstance(event.get("clientId"), str) and isinstance(event.get("jobId"), str):
            job = state.queue.get(cast("str", event["clientId"]), cast("str", event["jobId"]))
        if job is not None:
            return principal_for(request).allows_in(job.scope, "jobs:read")
        return "jobRef" not in event and not ("clientId" in event and "jobId" in event)

    sub = state.hub.subscribe(client_id, event_filter=visible)

    async def pump() -> None:
        while True:
            if not principal_for(request).allows("jobs:read"):
                await ws.close(code=1008, message=b"authorization-expired")
                return
            try:
                event = await asyncio.wait_for(sub.get(), timeout=1)
            except TimeoutError:
                continue
            if event is None:
                break
            if not principal_for(request).allows("jobs:read"):
                await ws.close(code=1008, message=b"authorization-expired")
                return
            if not visible(event):
                continue
            if BINARY_BLOB_KEY in event:
                # Preview payloads ship as one self-describing binary frame
                # (length-prefixed JSON header + raw bytes) - never base64
                # through the JSON path.
                await ws.send_bytes(encode_binary_event(event))
            else:
                await ws.send_json(event)

    async def receive() -> None:
        async for msg in ws:  # inbound messages are ignored; drain until close
            if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break

    pump_task = asyncio.create_task(pump())
    receive_task = asyncio.create_task(receive())
    try:
        done, _ = await asyncio.wait((pump_task, receive_task), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        state.hub.unsubscribe(sub)
        for task in (pump_task, receive_task):
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await ws.close()
    return ws


def _lease_wire(lease: Lease) -> dict[str, object]:
    return {
        "reservationId": lease.lease_id,
        "device": lease.device,
        "bytes": lease.nbytes,
        "expiresInSeconds": round(lease.ttl_remaining(), 3),
    }


def _governed(state: ServerState) -> tuple[MemoryGovernor, LeaseBroker]:
    """The coordination endpoints exist only on governed instances; a peer
    probing an ungoverned one gets a clean 409, never a pretend-yes."""
    if state.governor is None or state.leases is None:
        raise web.HTTPConflict(
            text=json.dumps({"error": "this instance runs ungoverned"}),
            content_type="application/json",
        )
    return state.governor, state.leases


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        raw = cast(object, await request.json())
    except json.JSONDecodeError as exc:
        raise _bad_request(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise _bad_request("request body must be an object")
    return cast(dict[str, Any], raw)


def _require_device(body: Mapping[str, Any]) -> str:
    device = body.get("device")
    if not isinstance(device, str) or not device:
        raise _bad_request("'device' must be a non-empty string")
    return device


def _require_bytes(body: Mapping[str, Any], key: str = "bytes") -> int:
    nbytes = body.get(key)
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise _bad_request(f"'{key}' must be a non-negative integer")
    return nbytes


def _optional_seconds(body: Mapping[str, Any], key: str) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise _bad_request(f"'{key}' must be a positive number of seconds")
    return float(value)


def _item_wire(item: ConsumerItem) -> dict[str, object]:
    wire: dict[str, object] = {
        "itemId": item.item_id,
        "displayName": item.display_name,
        "bytesByResidency": dict(item.bytes_by_residency),
    }
    if item.pages is not None:
        # Geometry travels with the data: no client hardcodes a page size.
        wire["pages"] = {
            "pageBytes": item.pages.page_bytes,
            "pageCount": item.pages.page_count,
            "flags": list(item.pages.flags),
        }
    return wire


def _memory_status_payload(state: ServerState) -> dict[str, object]:
    """The instance's memory picture: one shape for the HTTP endpoint and
    the pushed memory_status event, so a memory panel (live allocation
    visualization, DESIGN 3.10 telemetry) renders either without caring
    whether it polled or subscribed."""
    governor_status = state.governor.status() if state.governor is not None else None
    return {
        "devices": state.engine.resource_status(),
        "queue": state.queue.status(),
        # The instance's governor (DESIGN 3.10): budgets, reservations,
        # consumer footprints per residency class. Null when the instance
        # runs ungoverned - honest, not a placeholder.
        "memoryGovernor": governor_status,
        "acceleratorPolicy": _accelerator_policy_payload(state, governor_status),
        # Live cross-instance leases, TTLs included: a peer reading
        # status sees exactly what is pinned and for how long.
        "leases": (
            [_lease_wire(lease) for lease in state.leases.leases()]
            if state.leases is not None
            else None
        ),
    }


def _resolved_accelerator_policy(
    policy: AcceleratorMemoryPolicy,
    total_bytes: int,
    budget_bytes: int | None,
) -> dict[str, object]:
    resolved = policy.resolve(total_bytes, budget_bytes)
    return {
        "budgetBytes": resolved.hard_budget_bytes,
        "effectiveBudgetBytes": resolved.effective_budget_bytes,
        "budgetHeadroomBytes": resolved.budget_headroom_bytes,
        "classicWeightCapacityBytes": resolved.residency_capacity_bytes,
        "insufficientTotal": resolved.insufficient_total,
        "budgetExceedsTotal": resolved.budget_exceeds_total,
    }


def _accelerator_policy_payload(
    state: ServerState,
    governor_status: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    policy = AcceleratorMemoryPolicy(
        physical_headroom_bytes=state.settings.memory_headroom,
    )
    live_budgets = state.settings.memory_budgets
    applied_by_worker = (
        {}
        if state.residency_memory_budgets is None
        else {worker: dict(budgets) for worker, budgets in state.residency_memory_budgets().items()}
    )
    devices = {
        device
        for device in (
            *(governor_status or {}),
            *live_budgets,
            *(device for budgets in applied_by_worker.values() for device in budgets),
        )
        if device.startswith("vram:cuda:") and device.removeprefix("vram:cuda:").isdigit()
    }
    device_status: dict[str, dict[str, object]] = {}
    for device in sorted(devices):
        measured = (governor_status or {}).get(device, {}).get("measured")
        total: object = (
            cast("Mapping[object, object]", measured).get("totalBytes")
            if isinstance(measured, Mapping)
            else None
        )
        total_bytes = total if type(total) is int and total > 0 else None
        live_budget = live_budgets.get(device)
        worker_budgets = {
            worker: budgets.get(device) for worker, budgets in sorted(applied_by_worker.items())
        }
        distinct_applied = set(worker_budgets.values())
        applied_consistent = len(distinct_applied) <= 1
        applied_budget = next(iter(distinct_applied)) if len(distinct_applied) == 1 else None
        admission: dict[str, object] = {"budgetBytes": live_budget}
        residency: dict[str, object] = {
            "budgetBytes": applied_budget,
            "budgetsByWorker": worker_budgets,
            "consistent": applied_consistent,
        }
        if total_bytes is not None:
            admission = _resolved_accelerator_policy(policy, total_bytes, live_budget)
            if worker_budgets and applied_consistent:
                residency.update(_resolved_accelerator_policy(policy, total_bytes, applied_budget))
        device_status[device] = {
            "totalBytes": total_bytes,
            "governorAdmission": admission,
            "residencyApplied": residency,
            "residencyBudgetStale": any(
                budget != live_budget for budget in worker_budgets.values()
            ),
        }
    return {
        "physicalHeadroomBytes": policy.physical_headroom_bytes,
        "inferenceReserveBytes": policy.inference_reserve_bytes,
        "minimumFreeBytes": policy.minimum_free_bytes,
        "aimdoSimpleHeadroomBaseBytes": policy.physical_headroom_bytes,
        "devices": device_status,
    }


async def handle_memory_status(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    payload = _memory_status_payload(state)
    # Item-level contents are opt-in (?details=1): peers polling for
    # admission stay cheap; a memory panel asks for the full picture.
    if request.query.get("details") in ("1", "true") and state.governor is not None:
        payload["consumerDetails"] = {
            name: [_item_wire(item) for item in items]
            for name, items in state.governor.details().items()
        }
    return web.json_response(payload)


async def handle_memory_shed(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    governor, _ = _governed(state)
    body = await _json_body(request)
    device = _require_device(body)
    nbytes = _require_bytes(body)
    freed = await governor.shed(device, nbytes)
    # Freed is what consumers actually gave back - possibly short. The
    # caller decides what short means; this endpoint never inflates.
    return web.json_response({"device": device, "requestedBytes": nbytes, "freedBytes": freed})


def _full_free_worker_complete(worker: Mapping[str, object]) -> bool:
    consumers = worker.get("consumers")
    if worker.get("status") != "complete" or not isinstance(consumers, list):
        return False
    for consumer in cast("list[object]", consumers):
        if not isinstance(consumer, Mapping):
            return False
        if cast("Mapping[str, object]", consumer).get("status") != "complete":
            return False
    return True


async def handle_memory_free(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    body = await _json_body(request)
    request_id = body.get("requestId")
    if not isinstance(request_id, str) or not request_id:
        raise _bad_request("'requestId' must be a non-empty string")
    refusal = state.queue.begin_maintenance()
    if refusal is not None:
        return web.json_response(
            {
                "requestId": request_id,
                "completed": False,
                "queuePaused": state.queue.paused,
                "workers": [],
            },
            status=409,
            reason=refusal,
        )
    try:
        if state.full_free is None:
            workers: list[dict[str, object]] = []
        else:
            workers = list(await state.full_free(request_id))
        completed = bool(workers) and all(_full_free_worker_complete(worker) for worker in workers)
        return web.json_response(
            {
                "requestId": request_id,
                "completed": completed,
                "queuePaused": state.queue.paused,
                "workers": workers,
            }
        )
    finally:
        state.queue.end_maintenance()


DEFAULT_LEASE_TTL = 60.0


async def handle_memory_reserve(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    _, broker = _governed(state)
    body = await _json_body(request)
    device = _require_device(body)
    nbytes = _require_bytes(body)
    ttl = _optional_seconds(body, "ttlSeconds") or DEFAULT_LEASE_TTL
    timeout = _optional_seconds(body, "timeoutSeconds")
    try:
        lease = await broker.acquire(device, nbytes, ttl=ttl, timeout=timeout)
    except BudgetExceeded as exc:
        # It can never fit: no amount of waiting or shedding helps.
        return web.json_response({"error": str(exc)}, status=507)
    except ReservationTimeout as exc:
        # It did not fit in time: try later, or shed and retry.
        return web.json_response({"error": str(exc)}, status=503)
    return web.json_response(_lease_wire(lease), status=201)


async def handle_memory_reserve_renew(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    _, broker = _governed(state)
    body = await _json_body(request)
    ttl = _optional_seconds(body, "ttlSeconds") or DEFAULT_LEASE_TTL
    lease = broker.renew(request.match_info["lease_id"], ttl=ttl)
    if lease is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such reservation (expired or released)"}),
            content_type="application/json",
        )
    return web.json_response(_lease_wire(lease))


async def handle_memory_release(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    _, broker = _governed(state)
    if not await broker.release(request.match_info["lease_id"]):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such reservation (expired or released)"}),
            content_type="application/json",
        )
    return web.Response(status=204)


async def handle_cache_trim(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    governor, _ = _governed(state)
    body = await _json_body(request)
    device = _require_device(body)
    # Bytes omitted means full maintenance: everything sheddable goes.
    nbytes = _require_bytes(body) if "bytes" in body else governor.footprint(device)
    consumers = body.get("consumers")
    if consumers is not None and (
        not isinstance(consumers, list)
        or not all(isinstance(c, str) for c in cast(list[Any], consumers))
    ):
        raise _bad_request("'consumers' must be a list of shedder names")
    # Item IDs come from the detail contract (?details=1 on /memory/status);
    # "unload this model" is a trim naming its consumer and its item.
    items = body.get("items")
    if items is not None:
        if not isinstance(items, list) or not all(
            isinstance(i, str) for i in cast(list[Any], items)
        ):
            raise _bad_request("'items' must be a list of item IDs")
        if not isinstance(consumers, list) or len(cast(list[Any], consumers)) != 1:
            raise _bad_request(
                "'items' requires exactly one consumer: item IDs are meaningful only "
                "within one consumer's namespace"
            )
    freed = await governor.shed(
        device,
        nbytes,
        consumers=cast("list[str] | None", consumers),
        items=cast("list[str] | None", items),
    )
    return web.json_response({"device": device, "requestedBytes": nbytes, "freedBytes": freed})


async def handle_cache_entry(request: web.Request) -> web.Response:
    export = request.app[CACHE_EXPORT_KEY]
    manifest = await export.entry_wire(request.match_info["key"])
    if manifest is None:
        return web.json_response({"error": "cache miss"}, status=404)
    return web.json_response(dict(manifest))


async def handle_cache_blob(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return web.json_response(
            {"error": "malformed digest (expected 'blake3:<64 hex>')"}, status=400
        )
    export = request.app[CACHE_EXPORT_KEY]
    data = await export.blob(digest)
    if data is None:
        return web.json_response({"error": "unknown blob"}, status=404)
    return web.Response(body=data, content_type="application/octet-stream")


async def handle_assets_list(request: web.Request) -> web.Response:
    export = request.app[ASSET_EXPORT_KEY]
    digests = await asyncio.to_thread(export.digests)
    return web.json_response({"digests": sorted(digests)})


async def handle_asset_bytes(request: web.Request) -> web.StreamResponse:
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return web.json_response(
            {"error": "malformed digest (expected 'blake3:<64 hex>')"}, status=400
        )
    export = request.app[ASSET_EXPORT_KEY]
    path = await asyncio.to_thread(export.resolve, digest)
    if path is None:
        return web.json_response({"error": "asset not held here"}, status=404)

    # The response streams from the SAME descriptor the digest was verified
    # on - see asset_stream.py for the invariant and cancellation story.
    try:
        handle, size = await open_verified_sized(path, digest)
    except AssetIntegrityError:
        # Deliberately no filesystem path in the public body.
        return web.json_response(
            {"error": f"asset content on disk no longer matches {digest}"},
            status=409,
            headers={"Cache-Control": "no-store"},
        )
    except AssetError as error:
        return web.json_response({"error": str(error)}, status=500)
    except OSError:
        return web.json_response({"error": "asset not held here"}, status=404)
    return await stream_verified(request, handle, size)


def create_app(
    make_engine: Callable[[EventListener], Engine],
    schemas: Mapping[str, NodeSchema],
    *,
    pack_route_dispatch: PackRouteDispatch | None = None,
    frontend_module_read: FrontendModuleRead | None = None,
    max_running_jobs: int = 1,
    governor: MemoryGovernor | None = None,
    cache_export: CacheExport | None = None,
    asset_export: AssetExport | None = None,
    memory_status_interval: float | None = 1.0,
    history_limit: int = 256,
    packs: Mapping[str, PackInfo] | None = None,
    node_packs: Mapping[str, str] | None = None,
    execution_arms: Mapping[str, Sequence[ExecutionArm]] | None = None,
    allow_hosts: Sequence[str] = (),
    allow_origins: Sequence[str] = (),
    authenticator: Authenticator | None = None,
    principal_permissions: PrincipalPermissionStore | None = None,
    library: ServerLibrary | None = None,
    history: HistoryStore | None = None,
    training_sessions: TrainingSessionStore | None = None,
    choices: Mapping[str, Sequence[str]] | None = None,
    lazy_choices: Mapping[str, LazyChoiceFetcher] | None = None,
    schema_owners: Mapping[str, str] | None = None,
    choice_owners: Mapping[str, str] | None = None,
    compat_skips: Mapping[str, Mapping[str, CompatGateDiagnostic]] | None = None,
    settings: RuntimeSettings | None = None,
    pack_settings_root: Path | None = None,
    memory_headroom_changed: Callable[[int], None] | None = None,
    residency_memory_budgets: Callable[[], Mapping[str, Mapping[str, int]]] | None = None,
    workers: Callable[[], Sequence[WorkerInfo]] | None = None,
    place_execution: PlaceExecution | None = None,
    debug_errors: bool = False,
    preview_default: str = "cheap",
    preview_animation: str = "ring",
    attention_policy: AttentionPolicy = "auto",
    redactor: PathRedactor | None = None,
    execution_journal: ExecutionJournal | None = None,
    full_free: FullFree | None = None,
    federated_asset_paths: Mapping[str, str] | None = None,
    federated_asset_store: ResolutionStore | None = None,
    federated_asset_sources: Sequence[AcquisitionSource] = (),
    federated_asset_provider_policy: Mapping[str, frozenset[str]] | None = None,
    _federated_asset_cursor_key: bytes | None = None,
    _federated_asset_clock: Callable[[], float] | None = None,
) -> web.Application:
    """Build the Dinkster server app.

    The server owns event wiring, so it constructs the engine: pass a factory
    that accepts the server's event listener (mirrors how tests build engines).

    ``packs``/``node_packs`` carry pack provenance for /api/nodes: the packs
    table (pack id -> PackInfo) and per-node attribution (node type -> pack
    id). Both optional - unattributed nodes publish as "core".
    ``memory_headroom_changed`` is an optional composition-owned live-apply
    hook; embedders that do not host armed workers leave it unset.
    """
    state = ServerState(
        make_engine,
        schemas,
        max_running_jobs=max_running_jobs,
        governor=governor,
        history_limit=history_limit,
        packs=packs,
        node_packs=node_packs,
        execution_arms=execution_arms,
        history=history,
        choices=choices,
        lazy_choices=lazy_choices,
        schema_owners=schema_owners,
        choice_owners=choice_owners,
        compat_skips=compat_skips,
        settings=settings,
        memory_headroom_changed=memory_headroom_changed,
        residency_memory_budgets=residency_memory_budgets,
        workers=workers,
        place_execution=place_execution,
        debug_errors=debug_errors,
        preview_default=preview_default,
        preview_animation=preview_animation,
        attention_policy=attention_policy,
        redactor=redactor,
        execution_journal=execution_journal,
        full_free=full_free,
    )
    app = web.Application()
    app[STATE_KEY] = state
    app[PACK_SETTINGS_KEY] = PackSettingsStore(pack_settings_root)
    permission_store = principal_permissions or PrincipalPermissionStore()
    install_browser_request_security(
        app,
        allow_hosts=allow_hosts,
        allow_origins=allow_origins,
    )
    install_cors(app, allow_origins)
    route_capabilities = install_pack_surfaces(
        app, lambda: state.engine.extension_snapshot, pack_route_dispatch, frontend_module_read
    )
    federated_paths = frozenset[str]()
    if federated_asset_paths is not None:
        if federated_asset_store is None or federated_asset_provider_policy is None:
            raise ValueError("federated asset routes require a store and provider policy")
        route_capabilities.update(
            add_federated_asset_routes(
                app,
                paths=federated_asset_paths,
                store=federated_asset_store,
                sources=federated_asset_sources,
                provider_policy=federated_asset_provider_policy,
                cursor_key=_federated_asset_cursor_key,
                clock=_federated_asset_clock,
            )
        )
        federated_paths = frozenset(federated_asset_paths.values())
    install_auth(
        app,
        authenticator,
        route_capabilities=route_capabilities,
        federated_asset_paths=federated_paths,
        permission_store=permission_store,
    )
    app.router.add_get("/api/health", handle_health)
    app.router.add_post("/api/auth/ws-ticket", handle_ws_ticket)
    add_principal_routes(app, authenticator, permission_store)
    app.router.add_get("/api/nodes", handle_nodes)
    app.router.add_get("/api/workers", handle_workers)
    app.router.add_get("/api/extensions/snapshot", handle_extension_snapshot)
    app.router.add_get("/api/composition", handle_composition)
    app.router.add_get("/api/diagnostics", handle_diagnostics)
    app.router.add_get("/api/choices/{choice_id}", handle_choices)
    app.router.add_get("/api/packs/{pack_id}/icon", handle_pack_icon)
    app.router.add_get("/packs/{pack_id}/static/{path:.*}", handle_pack_static)
    app.router.add_get("/api/packs/{pack_id}/settings", handle_pack_settings_get)
    app.router.add_put("/api/packs/{pack_id}/settings", handle_pack_settings_put)
    app.router.add_get("/api/packs/{pack_id}/blueprints/{blueprint_id}", handle_pack_blueprint)
    app.router.add_get("/api/templates", handle_templates_list)
    app.router.add_get("/api/packs/{pack_id}/templates/{template_id}", handle_pack_template)
    app.router.add_get(
        "/api/packs/{pack_id}/templates/{template_id}/thumbnail",
        handle_pack_template_thumbnail,
    )
    app.router.add_get("/api/docs", handle_docs_list)
    app.router.add_get("/api/packs/{pack_id}/docs/pages/{digest}", handle_pack_doc_page)
    app.router.add_get("/api/packs/{pack_id}/docs/assets/{digest}", handle_pack_doc_asset)
    app.router.add_get("/api/packs/{pack_id}/locales/{digest}", handle_pack_locale_catalog)
    app.router.add_post("/api/jobs", handle_submit)
    app.router.add_get("/api/jobs", handle_jobs_list)
    app.router.add_get("/api/jobs/by-ref/{job_ref}/events", handle_job_events)
    app.router.add_get("/api/jobs/by-ref/{job_ref}", handle_job_status_by_ref)
    app.router.add_delete("/api/jobs/by-ref/{job_ref}", handle_cancel_by_ref)
    app.router.add_get("/api/jobs/{client_id}/{job_id}", handle_job_status)
    app.router.add_delete("/api/jobs/{client_id}/{job_id}", handle_cancel)
    app.router.add_get("/api/values", handle_value)
    app.router.add_get("/api/queue", handle_queue_status)
    app.router.add_post("/api/queue/pause", handle_queue_pause)
    app.router.add_post("/api/queue/resume", handle_queue_resume)
    app.router.add_post("/api/queue/clear", handle_queue_clear)
    app.router.add_get("/api/events", handle_events)
    add_settings_routes(app, state.settings)
    app.router.add_get("/memory/status", handle_memory_status)
    app.router.add_post("/memory/shed", handle_memory_shed)
    app.router.add_post("/memory/free", handle_memory_free)
    app.router.add_post("/memory/reserve", handle_memory_reserve)
    app.router.add_post("/memory/reserve/{lease_id}/renew", handle_memory_reserve_renew)
    app.router.add_delete("/memory/reserve/{lease_id}", handle_memory_release)
    app.router.add_post("/cache/trim", handle_cache_trim)
    if cache_export is not None:
        app[CACHE_EXPORT_KEY] = cache_export
        app.router.add_get("/cache/entry/{key}", handle_cache_entry)
        app.router.add_get("/cache/cas/{digest}", handle_cache_blob)
    if asset_export is not None:
        app[ASSET_EXPORT_KEY] = asset_export
        app.router.add_get("/assets", handle_assets_list)
        app.router.add_get("/assets/{digest}", handle_asset_bytes)
    if library is not None:
        add_library_routes(app, library)
    if history is not None:
        add_history_routes(app, history)
    if training_sessions is not None:
        add_training_routes(app, training_sessions)
    if execution_journal is not None:
        add_execution_journal_routes(app, execution_journal)

    # Native memory telemetry (DESIGN 3.10): a governed instance broadcasts
    # its memory picture over the event stream at a low fixed rate, so a
    # live memory panel needs no polling loop and no custom pack shipping
    # events through server internals. Droppable chatter: a slow subscriber
    # sheds stale samples; the next one supersedes them. None disables.
    sampler_task: asyncio.Task[None] | None = None

    async def sample_memory(interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            wire: dict[str, object] = {"type": "memory_status"}
            wire.update(_memory_status_payload(state))
            state.hub.publish(wire, client_id=None, droppable=True)

    async def on_startup(app: web.Application) -> None:
        state.queue.start()
        if execution_journal is not None:
            execution_journal.start()
        nonlocal sampler_task
        if governor is not None and memory_status_interval is not None:
            sampler_task = asyncio.get_running_loop().create_task(
                sample_memory(memory_status_interval)
            )

    async def on_cleanup(app: web.Application) -> None:
        if sampler_task is not None:
            sampler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler_task
        await state.queue.close()
        if history is not None:
            # Queue close just wrote its terminal transitions inline
            # through the store; nothing writes after this point.
            await asyncio.to_thread(history.close)
        if training_sessions is not None:
            # The store owns its journal file (TrainingSessionStore.close
            # closes it); nothing here writes after queue close.
            await asyncio.to_thread(training_sessions.close)
        if execution_journal is not None:
            # Queue close just emitted terminal transitions; the journal's
            # close lands them (and their finalize pruning) before the
            # store goes away.
            await execution_journal.close()
        await asyncio.to_thread(permission_store.close)
        if state.leases is not None:
            await state.leases.close()
        state.hub.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
