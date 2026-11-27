"""InProcessWorker: the trivial Worker - the boundary contract without the
boundary cost.

This shim is where envelopes are unwrapped and wrapped (hazard H9): node
execute() receives plain Python values and returns plain values keyed by
output id; everything envelope-shaped happens here, driven by the schema.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from dinkster_assets import AssetError, AssetVault, resolver_from_env
from dinkster_assets.declared import VAULT_ENV, use_produced_output_job
from dinkster_assets.privacy import check_secret_data
from dinkster_memory import FullReleaseConsumer, Shedder
from dinkster_memory.resident import ResidentPool
from dinkster_protocol import (
    AdmissionInvocation,
    AdmissionResult,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    BlockedOutput,
    CompatGateDiagnostic,
    CurrentNodeRef,
    CurrentOutputRef,
    ExpandedReturn,
    ExternalNodeRef,
    ExternalOutputRef,
    Invocation,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    LiteralInput,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    NodeError,
    OnInvocationEvent,
    PresentOutput,
    SavedArtifactCandidate,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
)
from dinkster_protocol.result_algebra import (
    RESULT_ALGEBRA_MAX_LITERAL_DEPTH,
    RESULT_ALGEBRA_MAX_LITERAL_ITEMS,
    InputBinding,
    JsonLiteral,
    JsonObject,
    OutputBinding,
)
from dinkster_schema import (
    AbsentOutput,
    Expansion,
    ExpansionOutput,
    Node,
    NodeSchema,
    Reporter,
    TypeSolveError,
    bind_type_variables,
    combo_choices_json_bytes,
    combo_type_mismatch_is_error,
    elaborate,
    install_stream_capture,
    plan_asset_coercion,
    resolved_type_expr,
    resolved_type_id,
    schema_signature,
    use_capture_budget,
    use_reporter,
)
from dinkster_schema.body import all_rows
from dinkster_schema.flatten import described_ids, flatten_body
from dinkster_schema.media import media_diagnostics
from dinkster_schema.model import Output
from dinkster_schema.nest import OutputShapeError, flatten_outputs, nest_inputs
from dinkster_schema.reporting import (
    capture_image_documents,
    capture_preview_artifacts,
    capture_saved_artifacts,
    capture_value_diagnostics,
)
from dinkster_values import (
    TypeRegistry,
    UnresolvablePayload,
    Value,
    ValueMeta,
    is_absent,
    list_children,
    make_absent_value,
    process_instance_token,
)

from .error_hints import hints_for
from .execution import ExecutionContext, current_execution_context, use_execution_context
from .media import apply_alpha_policy, coercion_drops_alpha, media_input_value, prepare_media_output
from .preview_artifacts import capture_saved_file_previews
from .produced_assets import ProducedAssetAuthority, image_document_values
from .run_control import PreparedRunContext
from .saved_artifacts import SavedArtifactAuthority


async def _run_sync(call: Callable[..., object], /, *args: object, **kwargs: object) -> object:
    task = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
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


class InputCoercionError(Exception):
    """An input failed worker-side admission or asset coercion.

    This includes the hard COMBO boundary and asset inputs whose
    provider is missing or whose decode/merge fails. It is a fact about THIS
    node's inputs, exactly like UnresolvablePayload - reported as a NodeError,
    never a worker crash.
    """


def validated_choice_values(values: object, *, subject: str) -> tuple[str, ...]:
    """Combo choice value validation shared by startup lists and lazy
    provider results: a sequence of non-empty strings, deduplicated
    preserving declaration order, bounded by the combo JSON budget. An
    empty list is legal (a registered but currently empty source)."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{subject} must be a sequence of strings")
    deduped: list[str] = []
    seen: set[str] = set()
    for value in cast("Sequence[object]", values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{subject} has a non-string or empty value")
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    combo_choices_json_bytes(deduped, subject=subject)
    return tuple(deduped)


def attention_route_token_matches_capabilities(
    capabilities: AttentionCapabilityEvidence | None,
    token: AttentionRouteToken | None,
) -> bool:
    """Check a job route against immutable worker capability evidence."""
    if capabilities is None or token is None:
        return capabilities is None and token is None
    try:
        expected = derive_attention_route_token(
            capabilities,
            AttentionPolicyConfig(
                token.requested_policy,
                token.requested_role_policies,
            ),
        )
    except ValueError:
        return False
    return canonical_attention_route_token_bytes(expected) == canonical_attention_route_token_bytes(
        token
    )


class InProcessWorker:
    def __init__(
        self,
        node_types: Mapping[str, type[Node]],
        registry: TypeRegistry,
        *,
        pack_context: Callable[[], contextlib.AbstractContextManager[None]] | None = None,
        attention_capabilities: AttentionCapabilityEvidence | None = None,
        attention_route_token: AttentionRouteToken | None = None,
        attention_runtime: object | None = None,
        combo_choices: Mapping[str, tuple[str, ...]] | None = None,
        lazy_choices: Mapping[str, Callable[[], Sequence[str]]] | None = None,
        memory_consumers: Mapping[str, Shedder] | None = None,
        registered_types: tuple[str, ...] = (),
        saved_preview_vault: AssetVault | None = None,
    ) -> None:
        self._node_types = dict(node_types)
        self._saved_preview_vault = saved_preview_vault
        self._registry = registry
        self.registered_types = registered_types
        self.resident_types = tuple(
            tid for tid in registry.type_ids() if registry.spec(tid).may_hold_resident
        )
        self._pack_context = pack_context or contextlib.nullcontext
        if attention_route_token is not None and not isinstance(
            cast("object", attention_route_token), AttentionRouteToken
        ):
            raise TypeError("attention_route_token must be AttentionRouteToken or None")
        if attention_capabilities is not None and not isinstance(
            cast("object", attention_capabilities), AttentionCapabilityEvidence
        ):
            raise TypeError("attention_capabilities must be AttentionCapabilityEvidence or None")
        if (attention_capabilities is None) != (attention_route_token is None):
            raise ValueError("attention capabilities and route token must be provided together")
        if attention_route_token is not None and not attention_route_token_matches_capabilities(
            attention_capabilities, attention_route_token
        ):
            raise ValueError("attention capabilities do not match route token")
        self._attention_capabilities = attention_capabilities
        self._attention_route_token = attention_route_token
        self._attention_runtime = attention_runtime
        self._combo_choices = dict(combo_choices or {})
        self._lazy_choices = dict(lazy_choices or {})
        self._memory_consumers = dict(memory_consumers or {})
        for index, table in enumerate(registry.resident_tables):
            if not isinstance(table, ResidentPool) or any(
                consumer is table for consumer in self._memory_consumers.values()
            ):
                continue
            name = f"resident-values-{index}"
            if name in self._memory_consumers:
                raise ValueError(f"reserved resident consumer name: {name}")
            self._memory_consumers[name] = table
        with self._pack_context():
            self._schemas: dict[str, NodeSchema] = {
                node_type: cls.schema() for node_type, cls in self._node_types.items()
            }

    @property
    def schemas(self) -> Mapping[str, NodeSchema]:
        return self._schemas

    @property
    def combo_choices(self) -> Mapping[str, tuple[str, ...]]:
        return self._combo_choices

    @property
    def lazy_choice_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._lazy_choices))

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        """Run the pack's lazy choice provider for ``choice_id`` and return
        its values; one invocation per call, no caching. The provider is
        foreign pack code, so it runs off the event loop, under the pack
        context, held to the same grammar and budget as static lists."""
        provider = self._lazy_choices.get(choice_id)
        if provider is None:
            raise ValueError(f"unknown lazy choice list {choice_id!r}")

        def invoke() -> tuple[str, ...]:
            with self._pack_context():
                values = provider()
            return validated_choice_values(values, subject=f"lazy choice {choice_id!r}")

        return cast("tuple[str, ...]", await _run_sync(invoke))

    @property
    def compat_skips(self) -> Mapping[str, CompatGateDiagnostic]:
        return {}

    @property
    def extension_contributions(self) -> tuple[tuple[object, object], ...]:
        return ()

    @property
    def body_arms(self) -> Mapping[str, tuple[str, ...]]:
        return {}

    @property
    def alive(self) -> bool:
        return True

    @property
    def memory_consumers(self) -> Mapping[str, Shedder]:
        return self._memory_consumers

    @property
    def instance_token(self) -> str:
        return process_instance_token()

    @property
    def attention_route_token(self) -> AttentionRouteToken | None:
        return self._attention_route_token

    @property
    def attention_capabilities(self) -> AttentionCapabilityEvidence | None:
        return self._attention_capabilities

    @property
    def attention_runtime(self) -> object | None:
        return self._attention_runtime

    def bind_registry(self, registry: TypeRegistry) -> None:
        """Bind a staged worker to the host registry before publication."""
        self._registry = registry

    @property
    def registry(self) -> TypeRegistry:
        return self._registry

    @property
    def renditions(self) -> tuple[object, ...]:
        return self._registry.registered_renditions()

    async def start(self) -> None:
        # Node stdout/stderr becomes attributed execution log events while
        # still reaching the terminal (idempotent, process-global).
        install_stream_capture()
        return None

    async def close(self) -> None:
        return None

    async def full_release(self) -> tuple[dict[str, object], ...]:
        """Run every declared in-process consumer's explicit release operation."""
        results: list[dict[str, object]] = []
        for name, consumer in self._memory_consumers.items():
            if not isinstance(consumer, FullReleaseConsumer):
                results.append({"consumer": name, "status": "unsupported"})
                continue
            try:
                outcome = await consumer.full_release()
            except Exception as exc:  # noqa: BLE001 - one consumer cannot hide another
                results.append(
                    {
                        "consumer": name,
                        "status": "error",
                        "error": str(exc) or "consumer full release failed",
                    }
                )
                continue
            try:
                status = getattr(outcome, "status", None)
                error = getattr(outcome, "error", None)
                valid = not (
                    status not in {"complete", "busy", "unsupported", "error"}
                    or (status == "error" and (not isinstance(error, str) or not error))
                    or (status != "error" and error is not None)
                )
            except Exception:  # noqa: BLE001 - malformed result is isolated to its consumer
                valid = False
                status = None
                error = None
            if not valid:
                results.append(
                    {
                        "consumer": name,
                        "status": "error",
                        "error": "consumer returned an invalid full release result",
                    }
                )
                continue
            result: dict[str, object] = {"consumer": name, "status": status}
            if status == "error":
                result["error"] = error
            results.append(result)
        return tuple(results)

    async def prepare(self, node_types: Sequence[str]) -> None:
        missing = [t for t in node_types if t not in self._node_types]
        if missing:
            raise KeyError(f"worker has no implementation for: {', '.join(missing)}")

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        """Invoke with the pack context active across all worker-side work."""
        if any(spec.secret for spec in invocation.effective_schema.inputs):
            # Refuse before pack setup can produce side effects or errors.
            # Credential names are not a trusted resolution channel.
            return InvocationResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "Secret inputs require a trusted credential resolver",
                )
            )
        started_at_ns = time.time_ns()
        with (
            self._pack_context(),
            use_produced_output_job(
                invocation.job_ref or invocation.invocation_id, invocation.attempt_id
            ) as owns_adoption,
        ):
            control: PreparedRunContext | None = None
            try:
                authority: ProducedAssetAuthority | None = None
                if owns_adoption and os.environ.get("DINKSTER_PACK_SCRATCH"):
                    root = os.environ.get(VAULT_ENV)
                    vault = self._saved_preview_vault or (AssetVault(root) if root else None)
                    existing_env = dict(os.environ)
                    existing_env.pop("DINKSTER_PACK_SCRATCH", None)
                    authority = ProducedAssetAuthority(
                        vault,
                        resolver_from_env(),
                        resolver_from_env(existing_env),
                        preview_vault=self._saved_preview_vault,
                    )

                    async def prepare_values(values: Mapping[str, Value]) -> Mapping[str, Value]:
                        assert authority is not None
                        values = dict(values)
                        await _run_sync(authority.capture, values, None, invocation.inputs)
                        return values

                    if invocation.run_control is not None:
                        control = PreparedRunContext(invocation.run_control, prepare_values)
                        invocation = replace(invocation, run_control=control)
                result = await self._invoke_in_context(invocation, on_event)
                if result.outputs is not None and result.saved_artifact_claims:
                    outputs = await _run_sync(
                        SavedArtifactAuthority(
                            os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
                        ).document_outputs,
                        self.registry,
                        result.outputs,
                        result.saved_artifact_claims,
                        started_at_ns,
                    )
                    result = replace(result, outputs=cast("dict[str, Value]", outputs))
                if authority is not None and result.outputs is not None:
                    await _run_sync(authority.capture, result.outputs, None, invocation.inputs)
                    if result.image_documents:
                        documents = await _run_sync(image_document_values, result.image_documents)
                        await _run_sync(authority.capture, documents, None, invocation.inputs)
                return result
            except (AssetError, OSError) as error:
                return InvocationResult(
                    error=NodeError(invocation.node_id, invocation.node_type, str(error))
                )
            finally:
                if control is not None:
                    control.close()

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        del on_event
        if any(spec.secret for spec in invocation.effective_schema.inputs):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "Secret inputs require a trusted credential resolver",
                )
            )
        if invocation.executor is not None or invocation.arm is not None:
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-dispatch-unsupported: lazy hooks require the default owner body",
                )
            )

        cls = self._node_types.get(invocation.node_type)
        if cls is None:
            return LazyStatusResult(
                error=NodeError(invocation.node_id, invocation.node_type, "lazy-protocol-skew")
            )
        hook = getattr(cls, "check_lazy_status", None)
        if not callable(hook):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-hook-invalid: check_lazy_status must be callable",
                )
            )
        ordinary = Invocation(
            invocation_id=invocation.request_id,
            node_id=invocation.node_id,
            node_type=invocation.node_type,
            inputs=invocation.available_inputs,
            effective_schema=invocation.effective_schema,
            connected_undemanded_inputs=invocation.connected_undemanded_inputs,
            resolve_secret=invocation.resolve_secret,
            expected_execution_identity=invocation.expected_execution_identity,
            fp8_matmul=invocation.fp8_matmul,
            diffusion_dtype=invocation.diffusion_dtype,
            text_dtype=invocation.text_dtype,
            vae_dtype=invocation.vae_dtype,
            attention_policy=invocation.attention_policy,
            attention_route_token=invocation.attention_route_token,
            extension_snapshot_digest=invocation.extension_snapshot_digest,
            asset_folders=invocation.asset_folders,
        )
        outer_context = current_execution_context()
        execution = ExecutionContext(
            arm=invocation.arm,
            expected_execution_identity=invocation.expected_execution_identity,
            fp8_matmul=invocation.fp8_matmul,
            diffusion_dtype=invocation.diffusion_dtype,
            text_dtype=invocation.text_dtype,
            vae_dtype=invocation.vae_dtype,
            attention_policy=invocation.attention_policy,
            attention_route_token=invocation.attention_route_token,
            attention_capabilities=(
                self._attention_capabilities
                if invocation.attention_route_token is not None
                else None
            ),
            attention_runtime=(
                self._attention_runtime if invocation.attention_route_token is not None else None
            ),
            extension_snapshot_digest=invocation.extension_snapshot_digest,
            node_id=invocation.node_id,
            asset_folders=invocation.asset_folders,
            cancelled=(outer_context.cancelled if outer_context is not None else lambda: False),
        )
        try:
            with self._pack_context(), use_execution_context(execution):
                kwargs = self._build_kwargs(self._schemas[invocation.node_type], ordinary)
                raw = hook(**kwargs)
                if inspect.isawaitable(raw):
                    raw = await raw
            if raw is None:
                raw = ()
            if not isinstance(raw, (list, tuple)):
                raise TypeError("check_lazy_status must return a list or tuple")
            return LazyStatusResult(
                requested_inputs=tuple(cast("list[object] | tuple[object, ...]", raw))
            )
        except Exception as exc:  # noqa: BLE001 - hook failures are node failures
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    f"lazy-hook-failed: {exc}",
                    traceback=traceback.format_exc(),
                )
            )

    async def validate_admission(
        self,
        invocation: AdmissionInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> AdmissionResult:
        del on_event
        cls = self._node_types.get(invocation.node_type)
        if cls is None:
            return AdmissionResult("admission-protocol-skew: unknown node type")
        if not invocation.effective_schema.admission_validation:
            return AdmissionResult(
                "admission-protocol-skew: node did not opt into admission validation"
            )
        try:
            explanation = cls.validate_admission(
                literal_inputs=dict(invocation.literal_inputs),
                linked_input_types=dict(invocation.linked_input_types),
            )
        except Exception as exc:  # noqa: BLE001 - hook failures refuse admission
            return AdmissionResult(f"admission-hook-failed: {exc}")
        return AdmissionResult(explanation)

    async def _invoke_in_context(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None,
    ) -> InvocationResult:
        if not attention_route_token_matches_capabilities(
            self._attention_capabilities, invocation.attention_route_token
        ):
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message="attention route token does not match worker startup evidence",
                )
            )
        cls = self._node_types[invocation.node_type]
        base_schema = self._schemas[invocation.node_type]
        # Output validation/wrapping uses the *elaborated* interface carried
        # by the invocation, never the base schema (hazard H10).
        schema = invocation.effective_schema

        skew = self._check_body_provenance(base_schema, invocation)
        if skew is not None:
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=skew,
                )
            )

        coercion_diagnostics: list[dict[str, object]] = []
        media_inputs = dict(invocation.inputs)
        try:
            decoded_inputs: frozenset[str] | None = None
            if base_schema.input_values:
                parameters = inspect.signature(cls.execute).parameters
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                    decoded_inputs = frozenset(parameters)
            kwargs = self._build_kwargs(
                base_schema, invocation, coercion_diagnostics, media_inputs, decoded_inputs
            )
        except (UnresolvablePayload, InputCoercionError) as exc:
            # A linked value produced elsewhere whose type has no codec in
            # this process (a wildcard input fed a resident/tensor type,
            # say), or an asset-typed input whose coercion misses or fails
            # here. Either way a fact about THIS node's inputs, not a worker
            # crash: fail the node with the cause named, keep the run's
            # error shape ordinary.
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=str(exc),
                )
            )

        close_reporter: Callable[[], Awaitable[None]] | None = None
        if on_event is None:
            reporter = contextlib.nullcontext()
            capture = contextlib.nullcontext()
        else:
            report, close_reporter = self._make_reporter(on_event)
            reporter = use_reporter(report)
            # Exits before the reporter does, so captured output flushed at
            # invocation end still delivers.
            capture = use_capture_budget()
        artifact_candidates: list[SavedArtifactCandidate] = []

        def capture_artifact(node_id: str, filename: str, subfolder: str, folder_type: str) -> None:
            artifact_candidates.append(
                SavedArtifactCandidate(node_id, filename, subfolder, folder_type)
            )

        try:
            outer_context = current_execution_context()
            started_at_ns = (
                outer_context.started_at_ns
                if outer_context is not None and outer_context.started_at_ns
                else time.time_ns()
            )
            if outer_context is None:
                execution = ExecutionContext(
                    arm=invocation.arm,
                    expected_execution_identity=invocation.expected_execution_identity,
                    extension_snapshot_digest=invocation.extension_snapshot_digest,
                    fp8_matmul=invocation.fp8_matmul,
                    diffusion_dtype=invocation.diffusion_dtype,
                    text_dtype=invocation.text_dtype,
                    vae_dtype=invocation.vae_dtype,
                    attention_policy=invocation.attention_policy,
                    attention_route_token=invocation.attention_route_token,
                    attention_capabilities=(
                        self._attention_capabilities
                        if invocation.attention_route_token is not None
                        else None
                    ),
                    attention_runtime=(
                        self._attention_runtime
                        if invocation.attention_route_token is not None
                        else None
                    ),
                    single_job_multi_gpu_execution=invocation.single_job_multi_gpu_execution,
                    preview_mode=invocation.preview_mode,
                    preview_animation=invocation.preview_animation,
                    node_id=invocation.node_id,
                    job_ref=invocation.job_ref,
                    export_snapshot=invocation.export_snapshot,
                    asset_folders=invocation.asset_folders,
                    artifact_sink=capture_artifact,
                    started_at_ns=started_at_ns,
                )
            else:
                execution = ExecutionContext(
                    arm=invocation.arm,
                    expected_execution_identity=invocation.expected_execution_identity,
                    extension_snapshot_digest=invocation.extension_snapshot_digest,
                    fp8_matmul=invocation.fp8_matmul,
                    diffusion_dtype=invocation.diffusion_dtype,
                    text_dtype=invocation.text_dtype,
                    vae_dtype=invocation.vae_dtype,
                    attention_policy=invocation.attention_policy,
                    attention_route_token=invocation.attention_route_token,
                    attention_capabilities=(
                        self._attention_capabilities
                        if invocation.attention_route_token is not None
                        else None
                    ),
                    attention_runtime=(
                        self._attention_runtime
                        if invocation.attention_route_token is not None
                        else None
                    ),
                    single_job_multi_gpu_execution=invocation.single_job_multi_gpu_execution,
                    preview_mode=invocation.preview_mode,
                    preview_animation=invocation.preview_animation,
                    node_id=invocation.node_id,
                    job_ref=invocation.job_ref,
                    export_snapshot=invocation.export_snapshot,
                    asset_folders=invocation.asset_folders,
                    cancelled=outer_context.cancelled,
                    materialize_source=outer_context.materialize_source,
                    artifact_sink=capture_artifact,
                    started_at_ns=started_at_ns,
                )
            with (
                use_execution_context(execution),
                reporter,
                capture,
                capture_value_diagnostics() as pack_diagnostics,
                capture_image_documents() as image_documents,
                capture_preview_artifacts() as preview_artifacts,
                capture_saved_artifacts() as saved_artifact_claims,
                capture_saved_file_previews(
                    enabled=invocation.saved_file_previews_enabled,
                    vault=self._saved_preview_vault,
                ) as saved_file_previews,
            ):
                if inspect.iscoroutinefunction(cls.execute):
                    raw = await cls.execute(**kwargs)
                else:
                    raw = await _run_sync(cls.execute, **kwargs)
                    if inspect.isawaitable(raw):
                        raw = await raw
        except Exception as exc:  # noqa: BLE001 - node failures become NodeError
            result = InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    hints=hints_for(exc),
                )
            )
        else:
            result = self._wrap_outputs(invocation, schema, raw, media_inputs)
            if result.outputs is not None:
                diagnostics = media_diagnostics(schema, media_inputs, result.outputs)
                diagnostics.extend(
                    {**item, "outputId": output_id}
                    for output_id in result.outputs
                    for item in coercion_diagnostics
                )
                diagnostics.extend(pack_diagnostics)
                if diagnostics:
                    result = replace(
                        result,
                        outputs={
                            key: replace(
                                value,
                                meta=ValueMeta(
                                    {
                                        **value.meta.entries,
                                        "valueDiagnostics": diagnostics,
                                    }
                                ),
                            )
                            for key, value in result.outputs.items()
                        },
                    )
                result = replace(
                    result,
                    artifact_candidates=tuple(artifact_candidates),
                    image_documents=tuple(image_documents),
                    preview_artifacts=tuple(preview_artifacts),
                    saved_artifact_claims=tuple(saved_artifact_claims),
                    saved_file_previews=tuple(saved_file_previews),
                )
        finally:
            if close_reporter is not None:
                await close_reporter()
        return result

    @staticmethod
    def _make_reporter(
        on_event: OnInvocationEvent,
    ) -> tuple[Reporter, Callable[[], Awaitable[None]]]:
        """Bridge the node's ambient report_* calls to the caller's sink.

        The reporter contract says emission never blocks or raises into node
        code and is callable from any thread; the sink contract says it runs
        on the engine's loop. So: same-thread calls go straight through, and
        calls from a worker thread (asyncio.to_thread carries the context
        there) are tracked and marshaled with call_soon_threadsafe. Closing
        rejects later chatter and drains accepted calls before invoke returns.
        Sink exceptions are swallowed - a broken observer must not fail the
        node."""
        loop = asyncio.get_running_loop()
        lock = threading.Lock()
        open_ = True
        pending = 0
        waiter: asyncio.Future[None] | None = None

        def deliver(event: InvocationEvent) -> None:
            with contextlib.suppress(Exception):
                on_event(event)

        def deliver_cross_thread(event: InvocationEvent) -> None:
            nonlocal pending, waiter
            try:
                deliver(event)
            finally:
                ready: asyncio.Future[None] | None = None
                with lock:
                    pending -= 1
                    if pending == 0:
                        ready = waiter
                        waiter = None
                if ready is not None and not ready.done():
                    ready.set_result(None)

        def reporter(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
            nonlocal pending
            event = InvocationEvent(name=name, data=dict(data), blob=blob)
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                with lock:
                    accepted = open_
                if not accepted:
                    return
                deliver(event)
            else:
                with lock:
                    if not open_:
                        return
                    pending += 1
                    try:
                        loop.call_soon_threadsafe(deliver_cross_thread, event)
                    except RuntimeError:
                        pending -= 1

        async def close() -> None:
            nonlocal open_, waiter
            with lock:
                open_ = False
                if pending == 0:
                    return
                waiter = loop.create_future()
                pending_waiter = waiter
            await pending_waiter

        return reporter, close

    def _check_body_provenance(self, base_schema: NodeSchema, invocation: Invocation) -> str | None:
        """Flatten the body again from the run's recorded members and choices.

        The worker knows neither link state nor the values of unrequested
        inputs, so count and leader formulas are not worked out here; only the
        descriptions of described rows are read, because they decide rows."""
        supplied = dict.fromkeys((*invocation.inputs, *invocation.connected_undemanded_inputs))
        # Lazy demand changes the order inputs arrive in, not the run's member order.
        stored: dict[str, object] = {
            spec.id: None for spec in invocation.effective_schema.inputs if spec.id in supplied
        }
        stored.update((name, None) for name in supplied if name not in stored)
        # A tuple selector's stored index decides its output's type.
        selectors: set[str] = set()
        types = [row.type for row in all_rows(base_schema.body) if isinstance(row, Output)]
        while types:
            expr = types.pop()
            if expr.kind == "tuple_element" and isinstance(expr.index, str):
                selectors.add(expr.index)
            if expr.element is not None:
                types.append(expr.element)
            types.extend(expr.elements)
        try:
            for input_id in described_ids(base_schema.body, invocation.inputs) | (
                selectors & set(invocation.inputs)
            ):
                stored[input_id] = invocation.inputs[input_id].resolve()
            expected = elaborate(
                base_schema,
                stored,
                dict(invocation.output_members),
                dict(invocation.effective_schema.slot_choices),
                formulas=False,
            )
        except (KeyError, ValueError, UnresolvablePayload) as error:
            return f"schema skew: invalid authored inputs: {error}"
        actual = invocation.effective_schema
        # Host-derived menus and display metadata are not input authority.
        if schema_signature(expected) != schema_signature(replace(expected, inputs=actual.inputs)):
            return "schema skew: input interface differs from worker projection"
        if expected.outputs != actual.outputs:
            return "schema skew: output interface differs from worker projection"
        admitted = {spec.id for spec in expected.inputs}
        if any(name not in admitted for name in supplied):
            return "schema skew: supplied input is absent from the effective interface"
        return None

    def _build_body_kwargs(
        self,
        base_schema: NodeSchema,
        invocation: Invocation,
        unwrap: Callable[[str, Value], object],
        wants: Callable[[str], bool],
    ) -> dict[str, object]:
        """The body's tree of values: groups as members in the run's order,
        a choice as its key and the chosen option's rows, a described row as
        a mapping from entry id to value."""
        effective = invocation.effective_schema
        values: dict[str, object] = {}
        for input_id, value in invocation.inputs.items():
            if wants(input_id.split(".", 1)[0]):
                values[input_id] = unwrap(input_id, value)
        for input_id in invocation.connected_undemanded_inputs:
            spec = effective.input(input_id)
            values[input_id] = (None,) if spec is not None and spec.type.kind == "list" else None
        stored: dict[str, object] = {spec.id: None for spec in effective.inputs}
        for input_id in described_ids(base_schema.body, invocation.inputs):
            stored[input_id] = invocation.inputs[input_id].resolve()
        flat = flatten_body(
            base_schema.body,
            stored,
            dict(invocation.output_members),
            dict(effective.slot_choices),
            formulas=False,
        )
        kwargs = nest_inputs(
            base_schema.body,
            values,
            dict(flat.members),
            dict(effective.slot_choices),
            {output.id for output in flat.outputs},
        )
        if base_schema.additional_inputs is not None or not base_schema.body:
            # Inputs the body never declared arrive by their stored ids.
            declared = {spec.id for spec in flat.inputs}
            kwargs.update((name, value) for name, value in values.items() if name not in declared)
        kwargs = {name: value for name, value in kwargs.items() if wants(name)}
        if base_schema.execution_context:
            if invocation.run_control is None:
                raise InputCoercionError("node requires a live run execution context")
            kwargs["execution_context"] = invocation.run_control
        if base_schema.input_values:
            kwargs["input_values"] = MappingProxyType(dict(invocation.inputs))
        return kwargs

    def _unwrap(
        self,
        invocation: Invocation,
        input_id: str,
        value: Value,
        diagnostics: list[dict[str, object]] | None = None,
        media_inputs: dict[str, Value] | None = None,
    ) -> object:
        try:
            obj = self._unwrap_value(invocation, input_id, value)
        except UnresolvablePayload as exc:
            raise InputCoercionError(f"input '{input_id}': {exc}") from exc
        spec = invocation.effective_schema.input(input_id)
        if spec is None or is_absent(value):
            return obj
        if (
            diagnostics is not None
            and spec.alpha_policy != "drop"
            and plan_asset_coercion(value.type_id, spec.type) is not None
            and coercion_drops_alpha(value, obj)
        ):
            diagnostics.append({"code": "alpha_dropped", "inputId": input_id})
        type_id = spec.type.runtime_type_id() or value.type_id
        try:
            result = apply_alpha_policy(obj, type_id, spec.alpha_policy, value)
            if media_inputs is not None and (type_id != value.type_id or result is not obj):
                media_inputs[input_id] = media_input_value(result, type_id, value)
            return result
        except ValueError as exc:
            raise InputCoercionError(f"input '{input_id}': {exc}") from exc

    def _unwrap_value(self, invocation: Invocation, input_id: str, value: Value) -> object:
        """The plain object node code receives for one input: the resolved
        payload - or, when the value is asset-typed and the declared input
        expects its decode target, the coerced result. Decode runs HERE,
        worker-side, with THIS registry's providers: the engine plans and
        folds provider identity into the cache key, it never decodes. The
        plan is re-derived from
        the same pure planner over the same effective schema, so engine and
        worker can never disagree about which coercion applies."""
        spec = invocation.effective_schema.input(input_id)
        if spec is None or is_absent(value):
            return value.resolve()

        def input_object(type_id: str, obj: object) -> object:
            if spec.accepts_storage:
                return obj.resolve() if isinstance(obj, Value) else obj
            return self._registry.input_object(type_id, obj)

        if spec.type.accepts_concrete(value.type_id):
            return input_object(value.type_id, value)
        plan = plan_asset_coercion(value.type_id, spec.type)
        if plan is None:
            if combo_type_mismatch_is_error(value.type_id, spec.type):
                raise InputCoercionError(
                    f"input '{input_id}' rejects runtime type {value.type_id}: "
                    "COMBO mismatches require an explicit converter"
                )
            return input_object(value.type_id, value)
        decoder = self._registry.asset_decoder_for(plan.target_type_id)
        merger = (
            self._registry.batch_merge_for(plan.merge_type_id)
            if plan.merge_type_id is not None
            else None
        )
        if decoder is None or (plan.merge_type_id is not None and merger is None):
            raise InputCoercionError(
                f"input '{input_id}' carries {value.type_id} and needs "
                + " and ".join(plan.missing_providers(self._registry))
                + ", which this worker does not register"
            )
        try:
            if plan.kind == "decode":
                return input_object(plan.target_type_id, decoder.decode(value.resolve()))
            children = list_children(value)
            if children is None:
                raise InputCoercionError(
                    f"input '{input_id}': {value.type_id} arrived without "
                    "list structure; its elements cannot be decoded"
                )
            decoded = [
                input_object(plan.target_type_id, decoder.decode(child.resolve()))
                for child in children
            ]
            if plan.kind == "lift":
                return decoded
            assert merger is not None  # merge plans always name a merger
            return merger.merge(decoded)
        except (UnresolvablePayload, InputCoercionError):
            raise
        except Exception as exc:
            raise InputCoercionError(
                f"input '{input_id}': coercing {value.type_id} to "
                f"{plan.result_type_id} failed: {exc}"
            ) from exc

    def _build_kwargs(
        self,
        base_schema: NodeSchema,
        invocation: Invocation,
        diagnostics: list[dict[str, object]] | None = None,
        media_inputs: dict[str, Value] | None = None,
        decoded_inputs: frozenset[str] | None = None,
    ) -> dict[str, object]:
        """Plain values for execute(), boundary-obliviously (hazard H9): the
        body's tree of values, as ``_build_body_kwargs`` builds it."""

        def unwrap(input_id: str, value: Value) -> object:
            return self._unwrap(invocation, input_id, value, diagnostics, media_inputs)

        def wants(name: str) -> bool:
            return decoded_inputs is None or name in decoded_inputs

        return self._build_body_kwargs(base_schema, invocation, unwrap, wants)

    def _wrap_outputs(
        self,
        invocation: Invocation,
        schema: NodeSchema,
        raw: object,
        media_inputs: Mapping[str, Value],
    ) -> InvocationResult:
        # Check before wrapping can fingerprint or encode a credential-bearing return.
        check_secret_data(raw)

        def contract_error(message: str) -> InvocationResult:
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=message,
                )
            )

        if isinstance(raw, Expansion):
            if not schema.may_expand_graph:
                return contract_error("execute() expanded without declaring may_expand_graph")
            try:
                raw = self._lower_expansion(invocation, schema, raw, media_inputs)
            except (KeyError, TypeError, ValueError) as exc:
                return contract_error(f"invalid expansion: {exc}")
            if isinstance(raw, InvocationResult):
                return raw
            return InvocationResult(expansion=raw)

        if isinstance(raw, ExpandedReturn):
            if not schema.may_expand_graph:
                return contract_error("execute() expanded without declaring may_expand_graph")
            if {name for name, _ in raw.bindings} != {out.id for out in schema.outputs}:
                return contract_error("expansion bindings must match declared outputs")
            declared_outputs = {out.id: out for out in schema.outputs}
            for name, binding in raw.bindings:
                if isinstance(binding, PresentOutput):
                    output = declared_outputs[name]
                    if is_absent(binding.value) and output.optional:
                        continue
                    if not output.type.accepts_concrete(binding.value.type_id):
                        return contract_error(f"output '{name}' rejects {binding.value.type_id}")
            outputs = {
                name: replace(
                    binding.value,
                    meta=ValueMeta({**binding.value.meta.entries, "absent.forwarded": True}),
                )
                if is_absent(binding.value)
                else binding.value
                for name, binding in raw.bindings
                if isinstance(binding, PresentOutput)
            }
            return InvocationResult(outputs=outputs, expansion=raw)

        if not isinstance(raw, Mapping):
            return contract_error(
                f"execute() must return a mapping keyed by output id, got {type(raw).__name__}"
            )

        # Flatten the returned tree into output ids, then validate exactly
        # against the elaborated schema - a wrong member set fails here, not
        # downstream. Node returns are runtime data, checked at this boundary.
        base_schema = self._schemas[invocation.node_type]
        if base_schema.body:
            try:
                raw = flatten_outputs(base_schema.body, cast("Mapping[str, object]", raw))
            except OutputShapeError as error:
                return contract_error(str(error))
        raw = {str(key): value for key, value in cast("Mapping[object, object]", raw).items()}

        declared = {out.id for out in schema.outputs}
        returned = set(raw.keys())
        if declared != returned:
            missing = declared - returned
            extra = returned - declared
            problems: list[str] = []
            if missing:
                problems.append("missing outputs: " + ", ".join(sorted(missing)))
            if extra:
                problems.append("undeclared outputs: " + ", ".join(sorted(str(k) for k in extra)))
            return contract_error("; ".join(problems))

        outputs: dict[str, Value] = {}
        bindings: Mapping[str, str] | None = None
        for out in schema.outputs:
            expected = out.type.runtime_type_id()
            if expected is None:
                # Generic interface (DESIGN 3.13): solve template variables
                # once per invocation from the authoritative runtime type ids
                # on the input envelopes (absent inputs bind nothing), then
                # resolve the output type under those bindings. Deterministic:
                # both the elaborated interface and the input envelopes are
                # fixed before execute() ran.
                if bindings is None:
                    try:
                        bindings = bind_type_variables(
                            schema.inputs,
                            {
                                input_id: value.type_id
                                for input_id, value in invocation.inputs.items()
                                if not is_absent(value)
                            },
                        )
                    except TypeSolveError as exc:
                        return contract_error(str(exc))
                expected = resolved_type_id(out.type, bindings)
            returned_value = raw[out.id]
            if isinstance(returned_value, AbsentOutput):
                # Blocking applies to required outputs too; ordinary absence
                # still requires an optional output declaration.
                if not out.optional and not returned_value.blocked:
                    return contract_error(
                        f"output '{out.id}' returned ABSENT but is not declared optional"
                    )
                outputs[out.id] = make_absent_value(
                    origin=f"{invocation.node_id}/{out.id}",
                    reason=returned_value.reason,
                    stands_for=expected,
                )
                continue
            if isinstance(returned_value, Value):
                if out.optional and is_absent(returned_value):
                    outputs[out.id] = replace(
                        returned_value,
                        meta=ValueMeta({**returned_value.meta.entries, "absent.forwarded": True}),
                    )
                    continue
                resolved = resolved_type_expr(out.type, bindings or {}) or out.type
                if not resolved.accepts_concrete(returned_value.type_id):
                    return contract_error(f"output '{out.id}' rejects {returned_value.type_id}")
                outputs[out.id] = returned_value
                continue
            if expected is None:
                # Plain results need a resolved type to wrap. A forwarded
                # envelope already carries its type and ownership unchanged.
                return contract_error(
                    f"output '{out.id}' type could not be resolved from this invocation's inputs"
                )
            try:
                output = prepare_media_output(raw[out.id], expected, out, schema, media_inputs)
                outputs[out.id] = self._registry.wrap(expected, output)
            except (TypeError, ValueError) as exc:
                # Shape violations (a list<T> output returning a non-sequence)
                # are contract errors at this boundary, not engine crashes.
                return contract_error(f"output '{out.id}': {exc}")
        return InvocationResult(outputs=outputs)

    def _lower_expansion(
        self,
        invocation: Invocation,
        schema: NodeSchema,
        declaration: Expansion,
        media_inputs: Mapping[str, Value],
    ) -> ExpandedReturn | InvocationResult:
        if set(declaration.outputs) != {output.id for output in schema.outputs}:
            raise ValueError("expansion bindings must match declared outputs")

        def reference(
            value: ExpansionOutput,
        ) -> CurrentOutputRef | LocalOutputRef | ExternalOutputRef:
            if value.external:
                if value.node_id is None:
                    raise ValueError("external reference requires a node id")
                return ExternalOutputRef(value.node_id, value.output_id)
            if value.node_id is None:
                if not isinstance(value.output_id, str):
                    raise ValueError("current output reference requires an output id")
                return CurrentOutputRef(value.output_id)
            return LocalOutputRef(value.node_id, value.output_id)

        literal_items = 0

        def literal(value: object, depth: int = 0) -> JsonLiteral:
            nonlocal literal_items
            literal_items += 1
            if (
                depth > RESULT_ALGEBRA_MAX_LITERAL_DEPTH
                or literal_items > RESULT_ALGEBRA_MAX_LITERAL_ITEMS
            ):
                raise ValueError("expansion literal exceeds item or depth limit")
            if value is None or type(value) in (bool, int, float, str):
                return cast("JsonLiteral", value)
            if isinstance(value, (list, tuple)):
                return tuple(literal(item, depth + 1) for item in cast("Sequence[object]", value))
            if isinstance(value, Mapping):
                items = cast("Mapping[object, object]", value)
                if not all(type(key) is str for key in items):
                    raise ValueError("expansion literal object keys must be strings")
                return JsonObject(
                    tuple(sorted((cast("str", k), literal(v, depth + 1)) for k, v in items.items()))
                )
            raise ValueError("expansion inputs must be JSON literals or output references")

        nodes: list[LocalNode] = []
        for node in declaration.nodes:
            display = node.display
            bindings: list[tuple[str, InputBinding]] = []
            for name, value in node.inputs:
                binding = (
                    reference(value)
                    if isinstance(value, ExpansionOutput)
                    else LiteralInput(literal(value))
                )
                bindings.append((name, binding))
            nodes.append(
                LocalNode(
                    node.local_id,
                    node.node_type,
                    tuple(bindings),
                    display=(
                        ExternalNodeRef(display[0])
                        if display[1]
                        else CurrentNodeRef()
                        if display[0] == invocation.node_id
                        else LocalNodeRef(display[0])
                    )
                    if display is not None
                    else None,
                    output_members=node.output_members,
                    slot_variants=node.slot_variants,
                )
            )
        present = {
            name: value
            for name, value in declaration.outputs.items()
            if not isinstance(value, ExpansionOutput)
            and not (isinstance(value, AbsentOutput) and value.blocked)
        }
        result = self._wrap_outputs(
            invocation,
            replace(schema, outputs=tuple(out for out in schema.outputs if out.id in present)),
            present,
            media_inputs,
        )
        if result.error is not None:
            return result
        assert result.outputs is not None
        outputs: list[tuple[str, OutputBinding]] = []
        for name, value in sorted(declaration.outputs.items()):
            returned: OutputBinding
            if isinstance(value, ExpansionOutput):
                returned = reference(value)
            elif isinstance(value, AbsentOutput) and value.blocked:
                returned = BlockedOutput(value.reason)
            else:
                returned = PresentOutput(result.outputs[name])
            outputs.append((name, returned))
        return ExpandedReturn(
            LocalExpansion(tuple(nodes), declaration.cache_policy), tuple(outputs)
        )
