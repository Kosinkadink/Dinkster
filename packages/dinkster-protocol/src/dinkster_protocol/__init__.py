"""The two protocols everything plugs into: Worker and CacheStore.

The engine never calls node code (hazard H3); it emits Invocations to a
Worker. In-process, another venv, and another machine are three
implementations of the same protocol.

This package is the execution boundary itself - the leaf both sides of
that boundary depend on. The engine (scheduler) consumes these protocols;
dinkster_workers and dinkster_caches implement them. Neither side needs the
other's package to name the contract, so a worker interpreter never
transitively installs the scheduler. Only dinkster_schema and dinkster_values
belong below this layer; anything more is a layering regression.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from dinkster_schema import NodeSchema
from dinkster_values import Value

from .attention import (
    ATTENTION_CAPABILITY_EVIDENCE_VERSION,
    ATTENTION_POLICIES,
    ATTENTION_ROLES,
    AttentionCapabilityEvidence,
    AttentionPolicy,
    AttentionPolicyConfig,
    AttentionRoute,
    AttentionRouteToken,
    AttentionRuntimeStatus,
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
    validate_attention_policy,
)
from .extensions import (
    EXTENSION_CAPABILITIES,
    EXTENSION_SCOPES,
    GENERATED_NODE_ID_PREFIX,
    GRAPH_COMPILE_CANCEL_TYPE,
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_DEPTH_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    GRAPH_COMPILE_ERROR_ID_COLLISION,
    GRAPH_COMPILE_ERROR_ID_FORMAT,
    GRAPH_COMPILE_ERROR_LINK_LIMIT,
    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
    GRAPH_COMPILE_ERROR_NODE_LIMIT,
    GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
    GRAPH_COMPILE_ERROR_PASS_LIMIT,
    GRAPH_COMPILE_ERROR_REPLY_OVERSIZE,
    GRAPH_COMPILE_ERROR_SELECTOR_EMITTED,
    GRAPH_COMPILE_ERROR_SELECTOR_INPUT,
    GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
    GRAPH_COMPILE_ERROR_TIMEOUT,
    GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
    GRAPH_COMPILE_MAX_DEPTH,
    GRAPH_COMPILE_MAX_GENERATED_PER_PASS,
    GRAPH_COMPILE_MAX_LINKS,
    GRAPH_COMPILE_MAX_NODES,
    GRAPH_COMPILE_MAX_PASSES,
    GRAPH_COMPILE_MAX_REPLY_BYTES,
    GRAPH_COMPILE_REQUEST_TYPE,
    GRAPH_COMPILE_RESULT_TYPE,
    GRAPH_COMPILE_TIMEOUT_SECONDS,
    GRAPH_COMPILERS_SURFACE,
    ATTENTION_QKV_SURFACE,
    ATTENTION_WRAPPER_SURFACE,
    ATTENTION_OUTPUT_SURFACE,
    ATTENTION_BACKEND_SURFACE,
    BLOCK_INJECTION_SURFACE,
    ATTENTION_SURFACES,
    GUIDANCE_SURFACES,
    ActiveExtension,
    BehaviorValue,
    CompositionMode,
    ContributionSurfaceDescriptor,
    ExtensionDeclaration,
    ExtensionEntryPoints,
    ExtensionScope,
    ExtensionSnapshot,
    GraphCompilerRegistrySnapshot,
    GuidancePhase,
    GuidancePhaseParticipation,
    GuidanceRegistrySnapshot,
    KeyedContribution,
    SamplerRegistrySnapshot,
    canonical_compile_reply_bytes,
    canonical_extension_snapshot,
    extension_behavior_hash,
    generated_node_id,
    is_extension_snapshot_digest,
)
from .frontend_modules import FrontendContribution, FrontendModule
from .pack_surfaces import (
    JsonField,
    JsonObjectSchema,
    PackEvent,
    PackRoute,
    PackSettingField,
    PackSettingsSchema,
    report_pack_event,
)
from .preview import (
    PREVIEW_ANIMATIONS,
    PREVIEW_MODES,
    PreviewAnimation,
    PreviewMode,
    PreviewPolicy,
    validate_preview_animation,
    validate_preview_mode,
)
from .training import (
    COALESCIBLE_TRAINING_EVENTS,
    DURABLE_TRAINING_EVENTS,
    MAX_TRAINING_EVENT_DATA_BYTES,
    TRAINING_SESSION_HANDLE_SCHEMA_VERSION,
    TrainingEventName,
    TrainingJournalEvent,
    TrainingSessionHandle,
    is_durable_training_event,
    is_training_checkpoint_digest,
    is_training_operation_id,
    is_training_session_id,
    training_session_stream,
)
from .workgroup import (
    MAX_REASON_BYTES,
    MAX_WORKGROUP_REPLICAS,
    MAX_WORKGROUP_UNITS,
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    WORKGROUP_VERSION,
    AbortWorkGroup,
    BeginAdmission,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    DispatchWork,
    GatherFailed,
    GatherSucceeded,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaBinding,
    ReplicaId,
    ReplicaReady,
    ReplicaRecipeId,
    ReplicaRefused,
    RequestCancellation,
    RunWorkUnit,
    SemanticSlot,
    SettleFailure,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupEvent,
    WorkGroupId,
    WorkGroupLifecycle,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupProtocolError,
    WorkGroupRefused,
    WorkGroupReleased,
    WorkGroupState,
    WorkUnitDefinition,
    WorkUnitFailed,
    WorkUnitId,
    WorkUnitProgress,
    WorkUnitResult,
    negotiate_workgroup_capabilities,
    reduce_workgroup,
    workgroup_message_from_wire,
    workgroup_message_to_wire,
)

CacheKey = str

_MEDIA_SOURCE_ROWS = frozenset(
    {
        ("media/image", "image/png", "png"),
        ("media/image", "image/jpeg", "jpg"),
        ("media/image", "image/webp", "webp"),
        ("media/audio", "audio/wav", "wav"),
        ("media/audio", "audio/flac", "flac"),
        ("media/audio", "audio/mpeg", "mp3"),
        ("media/audio", "audio/ogg", "ogg"),
        ("media/audio", "audio/webm", "webm"),
        ("media/audio", "audio/mp4", "m4a"),
        ("media/video", "video/mp4", "mp4"),
        ("media/video", "video/webm", "webm"),
    }
)


@dataclass(frozen=True)
class MediaSourceAuthority:
    """Exact byte-derived media authority for one host-authored invocation."""

    digest: str
    kind: str
    media_type: str
    extension: str
    byte_size: int

    def __post_init__(self) -> None:
        if (
            type(self.digest) is not str
            or not self.digest.startswith("blake3:")
            or len(self.digest) != 71
            or any(c not in "0123456789abcdef" for c in self.digest[7:])
        ):
            raise ValueError("MediaSourceAuthority.digest must be a lowercase blake3 digest")
        if (self.kind, self.media_type, self.extension) not in _MEDIA_SOURCE_ROWS:
            raise ValueError("MediaSourceAuthority media facts must be canonical")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("MediaSourceAuthority.byte_size must be a non-negative integer")


@dataclass(frozen=True)
class CompatGateDiagnostic:
    """One source-attributed, fail-closed compat translation refusal."""

    code: str
    source_node: str
    reason: str
    source_generation: Literal["v1", "v3"]
    path_kind: Literal["declared", "dynamic-family"]
    input_id: str | None = None
    input_path: tuple[str, ...] = ()
    lazy: bool | None = None
    input_is_list: bool | None = None
    output_is_list: bool | None = None
    raw_link: bool | None = None
    accept_all: bool | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code:
            raise ValueError("CompatGateDiagnostic.code must be a non-empty string")
        if type(self.source_node) is not str or not self.source_node:
            raise ValueError("CompatGateDiagnostic.source_node must be a non-empty string")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("CompatGateDiagnostic.reason must be a non-empty string")
        if self.source_generation not in ("v1", "v3"):
            raise ValueError("CompatGateDiagnostic.source_generation must be 'v1' or 'v3'")
        if self.path_kind not in ("declared", "dynamic-family"):
            raise ValueError("CompatGateDiagnostic.path_kind is invalid")
        if self.input_id is not None and (type(self.input_id) is not str or not self.input_id):
            raise ValueError("CompatGateDiagnostic.input_id must be non-empty or None")
        if type(self.input_path) is not tuple or any(
            type(part) is not str or not part for part in self.input_path
        ):
            raise ValueError("CompatGateDiagnostic.input_path must contain non-empty strings")
        expected_input_id = self.input_path[-1] if self.input_path else None
        if self.input_id != expected_input_id:
            raise ValueError("CompatGateDiagnostic.input_id must match the end of input_path")
        for name in ("lazy", "input_is_list", "output_is_list", "raw_link", "accept_all"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"CompatGateDiagnostic.{name} must be bool or None")

    def to_wire(self) -> dict[str, object]:
        return {
            "code": self.code,
            "sourceNode": self.source_node,
            "reason": self.reason,
            "sourceGeneration": self.source_generation,
            "pathKind": self.path_kind,
            "inputId": self.input_id,
            "inputPath": list(self.input_path),
            "lazy": self.lazy,
            "inputIsList": self.input_is_list,
            "outputIsList": self.output_is_list,
            "rawLink": self.raw_link,
            "acceptAll": self.accept_all,
        }

    @classmethod
    def from_wire(cls, value: object) -> CompatGateDiagnostic:
        keys = {
            "code",
            "sourceNode",
            "reason",
            "sourceGeneration",
            "pathKind",
            "inputId",
            "inputPath",
            "lazy",
            "inputIsList",
            "outputIsList",
            "rawLink",
            "acceptAll",
        }
        if not isinstance(value, Mapping) or set(cast("Mapping[object, object]", value)) != keys:
            raise ValueError("compat gate diagnostic must contain the exact field set")
        raw = cast("Mapping[str, object]", value)
        path = raw["inputPath"]
        if type(path) is not list or any(
            type(part) is not str for part in cast("list[object]", path)
        ):
            raise ValueError("compat gate diagnostic inputPath must be a string list")
        return cls(
            code=cast("str", raw["code"]),
            source_node=cast("str", raw["sourceNode"]),
            reason=cast("str", raw["reason"]),
            source_generation=cast("Literal['v1', 'v3']", raw["sourceGeneration"]),
            path_kind=cast("Literal['declared', 'dynamic-family']", raw["pathKind"]),
            input_id=cast("str | None", raw["inputId"]),
            input_path=tuple(cast("list[str]", path)),
            lazy=cast("bool | None", raw["lazy"]),
            input_is_list=cast("bool | None", raw["inputIsList"]),
            output_is_list=cast("bool | None", raw["outputIsList"]),
            raw_link=cast("bool | None", raw["rawLink"]),
            accept_all=cast("bool | None", raw["acceptAll"]),
        )


@dataclass(frozen=True)
class ExportSnapshot:
    """Opaque compat export payload for save-node synthesis.

    The engine and workers carry this payload without interpreting it. Only
    compat save-node wrappers interpret it when synthesizing hidden inputs.
    Validation is deliberately shallow because submitted JSON remains opaque.
    """

    prompt: Mapping[str, object]
    extra_pnginfo: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        prompt = cast("object", self.prompt)
        if not isinstance(prompt, Mapping):
            raise ValueError("ExportSnapshot.prompt must be a JSON-object mapping")
        prompt_mapping = cast("Mapping[object, object]", prompt)
        if not all(isinstance(key, str) for key in prompt_mapping):
            raise ValueError("ExportSnapshot.prompt must be a JSON-object mapping")
        extra_pnginfo = cast("object", self.extra_pnginfo)
        if extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, Mapping):
                raise ValueError("ExportSnapshot.extra_pnginfo must be a mapping or None")


@dataclass(frozen=True)
class Invocation:
    """One node execution request.

    effective_schema is the node's *elaborated* concrete interface (hazard
    H10): the worker validates and wraps execute() results against it, never
    against the base schema, so dynamic members are handled exactly like
    static ones. It is derivable from the same pure elaboration over the same
    stored ids, so carrying it is a convenience, not a second source of truth.

    output_members is the canonical family provenance for dynamic outputs:
    ordered (family_id, (suffix, ...)) pairs taken from the document. The
    worker builds the node-facing OutputInterface from it and cross-checks it
    against effective_schema - it never re-derives membership by parsing
    output ids, which would make a version-skewed worker silently trusted.
    """

    invocation_id: str
    node_id: str
    node_type: str
    inputs: Mapping[str, Value]
    effective_schema: NodeSchema
    output_members: tuple[tuple[str, tuple[str, ...]], ...] = ()
    connected_undemanded_inputs: tuple[str, ...] = ()
    executor: str | None = None
    """Host-side execution selection (stage 6 dispatch): the arm name the
    engine's plan_execution hook chose, or None when the node type is not
    enrolled in dispatch. Consumed by the dispatching worker facade to pick
    an implementation; NEVER serialized across the boundary. A same-session
    ArmWorker translates it into the separate wire ``arm`` selector. Immutable
    with the invocation, so the cache key computed from the same selection and
    the body actually invoked cannot diverge."""
    arm: str | None = None
    """Worker-side body selector. None selects the pack's default body;
    a non-empty name selects a same-session body registered by that pack.
    Unlike ``executor``, this field crosses the worker boundary."""
    expected_execution_identity: str | None = None
    """Host-authoritative assertion of the cache identity chosen for the
    executing body. It crosses the worker boundary unchanged and is exposed
    to node code through the worker execution context."""
    fp8_matmul: bool = False
    """Host-requested native fp8 matrix-multiplication policy. It crosses
    with ``expected_execution_identity`` so the worker loads exactly the
    numerics named by that identity, or refuses the request loudly."""
    diffusion_dtype: str | None = None
    text_dtype: str | None = None
    vae_dtype: str | None = None
    attention_policy: AttentionPolicy = "auto"
    attention_route_token: AttentionRouteToken | None = None
    """Worker-authenticated attention route evidence selected by the host."""
    extension_snapshot_digest: str | None = None
    """The extension generation pinned when this execution was admitted.
    It crosses the worker boundary as an opaque ``sha256:<hex>`` handle; the
    immutable snapshot itself remains on the catalog transport."""
    export_snapshot: ExportSnapshot | None = None
    """Opaque compat export payload, present only for output-node invocations."""
    media_sources: tuple[MediaSourceAuthority, ...] = ()
    """Exact current-run media authorities. They are execution data only and
    deliberately do not participate in graph, schema, or cache identity."""
    preview_mode: PreviewMode = "off"
    """This invocation's resolved sampling-preview spend. Execution data
    only: previews are droppable side effects, so the mode never joins
    graph, schema, or cache identity."""
    preview_animation: PreviewAnimation = "ring"
    """The run's animated-preview transport. Execution data only, exactly
    like ``preview_mode``; meaningless when previews are off."""
    job_ref: str | None = None
    attempt_id: int = 1
    """Stable server-job identity used to fence resumable boundary work."""

    def __post_init__(self) -> None:
        if self.job_ref is not None and (type(self.job_ref) is not str or not self.job_ref):
            raise ValueError("Invocation.job_ref must be a non-empty string when present")
        if type(self.attempt_id) is not int or self.attempt_id < 1:
            raise ValueError("Invocation.attempt_id must be a positive integer")
        validate_preview_mode(self.preview_mode)
        validate_preview_animation(self.preview_animation)
        if len(set(self.connected_undemanded_inputs)) != len(self.connected_undemanded_inputs):
            raise ValueError("Invocation.connected_undemanded_inputs must not contain duplicates")
        if type(self.media_sources) is not tuple:
            raise ValueError("Invocation.media_sources must be a tuple")
        if any(type(authority) is not MediaSourceAuthority for authority in self.media_sources):
            raise ValueError("Invocation.media_sources must contain media authorities")
        digests = tuple(authority.digest for authority in self.media_sources)
        if len(set(digests)) != len(digests):
            raise ValueError("Invocation.media_sources must have unique digests")
        if self.extension_snapshot_digest is not None and not is_extension_snapshot_digest(
            self.extension_snapshot_digest
        ):
            raise ValueError("Invocation.extension_snapshot_digest must be a sha256 digest")
        if self.fp8_matmul and self.expected_execution_identity is None:
            raise ValueError("Invocation.fp8_matmul requires expected_execution_identity")
        dtypes = (self.diffusion_dtype, self.text_dtype, self.vae_dtype)
        if any(dtype is not None for dtype in dtypes) and (
            self.expected_execution_identity is None
            or not all(isinstance(dtype, str) and dtype for dtype in dtypes)
        ):
            raise ValueError(
                "Invocation component dtypes must be complete and require "
                "expected_execution_identity"
            )
        resolve_attention_runtime_status(self.attention_policy, self.attention_route_token)


@dataclass(frozen=True)
class InvocationEvent:
    """One typed report from a running node: progress, a preview frame, or
    a pack-defined event (DESIGN 3.5). Chatter, not results - consumers may
    drop or coalesce, so nothing correctness-critical rides here. ``blob``
    carries binary payloads (encoded preview images) out-of-band so they
    never transit as JSON."""

    name: str
    data: Mapping[str, object] = field(default_factory=dict[str, object])
    blob: bytes | None = None


OnInvocationEvent = Callable[[InvocationEvent], None]
"""Per-invocation event sink. Called on the engine's event loop, possibly
while the invocation is still executing; must not block and must not raise."""


@dataclass(frozen=True)
class ErrorHint:
    code: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class NodeError:
    node_id: str
    node_type: str
    message: str
    traceback: str = ""
    hints: tuple[ErrorHint, ...] = ()


@dataclass(frozen=True)
class SavedArtifact:
    """A guarded output-node file exposed as an execution asset."""

    node_id: str
    digest: str
    name: str
    size: int
    media_type: str
    virtual_path: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.node_id, self.name, self.media_type, self.virtual_path)
        ):
            raise ValueError("SavedArtifact text fields must be non-empty")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("SavedArtifact.size must be nonnegative")
        if (
            type(self.digest) is not str
            or not self.digest.startswith("blake3:")
            or len(self.digest) != 71
            or any(char not in "0123456789abcdef" for char in self.digest[7:])
        ):
            raise ValueError("SavedArtifact.digest must be a canonical blake3 digest")


@dataclass(frozen=True)
class SavedArtifactCandidate:
    """An untrusted worker report naming one ComfyUI saved output."""

    node_id: str
    filename: str
    subfolder: str
    folder_type: str


@dataclass(frozen=True)
class InvocationResult:
    outputs: Mapping[str, Value] | None = None
    error: NodeError | None = None
    artifacts: tuple[SavedArtifact, ...] = ()
    artifact_candidates: tuple[SavedArtifactCandidate, ...] = ()


@dataclass(frozen=True)
class LazyStatusInvocation:
    """A generation-pinned request to evaluate one node's synchronous hook."""

    request_id: str
    node_id: str
    node_type: str
    available_inputs: Mapping[str, Value]
    connected_undemanded_inputs: tuple[str, ...]
    effective_schema: NodeSchema
    executor: str | None = None
    arm: str | None = None
    expected_execution_identity: str | None = None
    fp8_matmul: bool = False
    diffusion_dtype: str | None = None
    text_dtype: str | None = None
    vae_dtype: str | None = None
    attention_policy: AttentionPolicy = "auto"
    attention_route_token: AttentionRouteToken | None = None
    extension_snapshot_digest: str | None = None

    def __post_init__(self) -> None:
        if self.extension_snapshot_digest is not None and not is_extension_snapshot_digest(
            self.extension_snapshot_digest
        ):
            raise ValueError(
                "LazyStatusInvocation.extension_snapshot_digest must be a sha256 digest"
            )
        if len(set(self.connected_undemanded_inputs)) != len(self.connected_undemanded_inputs):
            raise ValueError("connected_undemanded_inputs must not contain duplicates")
        if self.fp8_matmul and self.expected_execution_identity is None:
            raise ValueError("LazyStatusInvocation.fp8_matmul requires expected execution identity")
        dtypes = (self.diffusion_dtype, self.text_dtype, self.vae_dtype)
        if any(dtype is not None for dtype in dtypes) and (
            self.expected_execution_identity is None
            or not all(isinstance(dtype, str) and dtype for dtype in dtypes)
        ):
            raise ValueError(
                "LazyStatusInvocation component dtypes must be complete and require "
                "expected execution identity"
            )
        resolve_attention_runtime_status(self.attention_policy, self.attention_route_token)


@dataclass(frozen=True)
class LazyStatusResult:
    requested_inputs: tuple[object, ...] | None = None
    error: NodeError | None = None

    def __post_init__(self) -> None:
        if (self.requested_inputs is None) == (self.error is None):
            raise ValueError(
                "LazyStatusResult must contain exactly one of requested_inputs or error"
            )


class Worker(Protocol):
    async def prepare(self, node_types: Sequence[str]) -> None: ...

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        """Execute one invocation. ``on_event`` receives the node's typed
        reports (progress/previews) as they happen; None means the caller
        does not observe them. Events for an invocation are delivered
        before its result returns."""
        ...


class LazyStatusWorker(Protocol):
    """Optional worker capability used only for nodes with deferred inputs."""

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult: ...


class CacheStore(Protocol):
    async def get(self, key: CacheKey) -> Mapping[str, Value] | None: ...

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None: ...


from .result_algebra import (  # noqa: E402 - NodeError is the algebra's leaf error
    RESULT_ALGEBRA_CAPABILITY,
    RESULT_ALGEBRA_MAX_COUNT,
    RESULT_ALGEBRA_MAX_DOCUMENT_BYTES,
    RESULT_ALGEBRA_MAX_LITERAL_DEPTH,
    RESULT_ALGEBRA_MAX_LITERAL_ITEMS,
    RESULT_ALGEBRA_MAX_NESTING,
    RESULT_ALGEBRA_VERSION,
    BlockedOutput,
    CurrentNodeRef,
    CurrentOutputRef,
    DirectReturn,
    ExpandedReturn,
    InvocationFailure,
    InvocationOutcome,
    InvocationReturn,
    JsonObject,
    LiteralInput,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    PresentOutput,
    ReturnBatch,
    negotiate_result_capabilities,
)

__all__ = [
    "FrontendContribution",
    "FrontendModule",
    "JsonField",
    "JsonObjectSchema",
    "PackEvent",
    "PackRoute",
    "PackSettingField",
    "PackSettingsSchema",
    "report_pack_event",
    "EXTENSION_CAPABILITIES",
    "EXTENSION_SCOPES",
    "GENERATED_NODE_ID_PREFIX",
    "GRAPH_COMPILERS_SURFACE",
    "ATTENTION_QKV_SURFACE",
    "ATTENTION_WRAPPER_SURFACE",
    "ATTENTION_OUTPUT_SURFACE",
    "ATTENTION_BACKEND_SURFACE",
    "BLOCK_INJECTION_SURFACE",
    "ATTENTION_SURFACES",
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
    "COALESCIBLE_TRAINING_EVENTS",
    "DURABLE_TRAINING_EVENTS",
    "MAX_TRAINING_EVENT_DATA_BYTES",
    "TRAINING_SESSION_HANDLE_SCHEMA_VERSION",
    "ActiveExtension",
    "ATTENTION_CAPABILITY_EVIDENCE_VERSION",
    "ATTENTION_POLICIES",
    "ATTENTION_ROLES",
    "AttentionCapabilityEvidence",
    "AttentionPolicy",
    "AttentionPolicyConfig",
    "AttentionRoute",
    "AttentionRouteToken",
    "AttentionRuntimeStatus",
    "BehaviorValue",
    "CacheKey",
    "CacheStore",
    "CompatGateDiagnostic",
    "CompositionMode",
    "ContributionSurfaceDescriptor",
    "ErrorHint",
    "ExtensionDeclaration",
    "ExtensionEntryPoints",
    "ExtensionScope",
    "ExtensionSnapshot",
    "GUIDANCE_SURFACES",
    "GraphCompilerRegistrySnapshot",
    "GuidancePhase",
    "GuidancePhaseParticipation",
    "GuidanceRegistrySnapshot",
    "Invocation",
    "InvocationEvent",
    "InvocationResult",
    "LazyStatusInvocation",
    "LazyStatusResult",
    "LazyStatusWorker",
    "MediaSourceAuthority",
    "KeyedContribution",
    "NodeError",
    "OnInvocationEvent",
    "PREVIEW_ANIMATIONS",
    "PREVIEW_MODES",
    "PreviewAnimation",
    "PreviewMode",
    "PreviewPolicy",
    "validate_preview_animation",
    "validate_preview_mode",
    "SamplerRegistrySnapshot",
    "Worker",
    "RESULT_ALGEBRA_CAPABILITY",
    "RESULT_ALGEBRA_MAX_COUNT",
    "RESULT_ALGEBRA_MAX_DOCUMENT_BYTES",
    "RESULT_ALGEBRA_MAX_LITERAL_DEPTH",
    "RESULT_ALGEBRA_MAX_LITERAL_ITEMS",
    "RESULT_ALGEBRA_MAX_NESTING",
    "RESULT_ALGEBRA_VERSION",
    "BlockedOutput",
    "CurrentOutputRef",
    "CurrentNodeRef",
    "DirectReturn",
    "ExpandedReturn",
    "InvocationFailure",
    "InvocationOutcome",
    "InvocationReturn",
    "JsonObject",
    "LiteralInput",
    "LocalExpansion",
    "LocalNode",
    "LocalNodeRef",
    "LocalOutputRef",
    "PresentOutput",
    "ReturnBatch",
    "negotiate_result_capabilities",
    "canonical_compile_reply_bytes",
    "canonical_extension_snapshot",
    "extension_behavior_hash",
    "generated_node_id",
    "is_extension_snapshot_digest",
    "attention_capability_evidence_from_wire",
    "attention_capability_evidence_to_wire",
    "attention_policy_config_from_wire",
    "attention_policy_config_to_wire",
    "attention_route_token_from_wire",
    "attention_route_token_to_wire",
    "automatic_attention_route",
    "canonical_attention_route_token_bytes",
    "derive_attention_route_token",
    "resolve_attention_runtime_status",
    "resolve_role_policy",
    "validate_attention_policy",
    "TrainingEventName",
    "TrainingJournalEvent",
    "TrainingSessionHandle",
    "is_durable_training_event",
    "is_training_checkpoint_digest",
    "is_training_operation_id",
    "is_training_session_id",
    "training_session_stream",
    "MAX_REASON_BYTES",
    "MAX_WORKGROUP_REPLICAS",
    "MAX_WORKGROUP_UNITS",
    "WORKGROUP_CAPABILITY",
    "WORKGROUP_VERSION",
    "AbortWorkGroup",
    "BeginAdmission",
    "BeginWorkGroup",
    "CancelWorkGroup",
    "CommitWorkGroup",
    "DeviceResourceId",
    "DispatchWork",
    "GatherFailed",
    "GatherSucceeded",
    "PrepareReplica",
    "ReleaseWorkGroup",
    "ReplicaBinding",
    "ReplicaId",
    "ReplicaReady",
    "ReplicaRecipeId",
    "ReplicaRefused",
    "RequestCancellation",
    "RunWorkUnit",
    "SemanticSlot",
    "SettleFailure",
    "WorkerInstanceId",
    "WorkGroupAttempt",
    "WorkGroupCancelled",
    "WorkGroupDefinition",
    "WorkGroupEvent",
    "WorkGroupId",
    "WorkGroupLifecycle",
    "WorkGroupMessage",
    "WorkGroupPrepared",
    "WorkGroupProtocolError",
    "WorkGroupRefused",
    "WorkGroupReleased",
    "WorkGroupState",
    "WorkUnitDefinition",
    "WorkUnitFailed",
    "WorkUnitId",
    "WorkUnitProgress",
    "WorkUnitResult",
    "negotiate_workgroup_capabilities",
    "reduce_workgroup",
    "workgroup_message_from_wire",
    "workgroup_message_to_wire",
    "WORKGROUP_DATA_PLANE_CAPABILITY",
]
