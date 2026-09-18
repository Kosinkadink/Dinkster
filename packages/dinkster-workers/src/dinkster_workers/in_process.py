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
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import cast

from dinkster_memory import FullReleaseConsumer, Shedder
from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    CompatGateDiagnostic,
    Invocation,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    NodeError,
    OnInvocationEvent,
    SavedArtifactCandidate,
    canonical_attention_route_token_bytes,
    derive_attention_route_token,
)
from dinkster_schema import (
    AbsentOutput,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    Node,
    NodeSchema,
    OutputInterface,
    Reporter,
    SlotValue,
    TypeSolveError,
    bind_type_variables,
    combo_choices_json_bytes,
    combo_type_mismatch_is_error,
    elaborate,
    install_stream_capture,
    plan_asset_coercion,
    resolved_type_id,
    use_capture_budget,
    use_reporter,
)
from dinkster_schema.media import media_diagnostics
from dinkster_schema.reporting import capture_value_diagnostics
from dinkster_values import (
    TypeRegistry,
    UnresolvablePayload,
    Value,
    ValueMeta,
    is_absent,
    list_children,
    make_absent_value,
)

from .error_hints import hints_for
from .execution import ExecutionContext, current_execution_context, use_execution_context
from .media import apply_alpha_policy, coercion_drops_alpha, media_input_value, prepare_media_output


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


def _is_combo_choice(entries: Sequence[DynamicEntry], parts: Sequence[str]) -> bool:
    """Whether a materialized choice path names a declared DynamicCombo.

    Family member suffixes occupy one path segment but are document state,
    not schema entries, so recurse past that wildcard segment. All option and
    slot branches are searched because this is classification only; the
    effective schema already records which branch is active.
    """
    if not parts:
        return False
    for entry in entries:
        if entry.id != parts[0]:
            continue
        rest = parts[1:]
        if isinstance(entry, DynamicComboSpec):
            if not rest:
                return True
            return any(_is_combo_choice(option.inputs, rest) for option in entry.options)
        if isinstance(entry, InputFamilySpec):
            return len(rest) >= 2 and _is_combo_choice(entry.template, rest[1:])
        if isinstance(entry, DynamicSlotSpec) and rest:
            if _is_combo_choice(entry.inputs, rest):
                return True
            return any(_is_combo_choice(variant.inputs, rest) for variant in entry.variants or ())
        return False
    return False


class InputCoercionError(Exception):
    """An input failed worker-side admission or asset coercion.

    This includes the hard core.combo boundary and asset inputs whose
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
        combo_choices: Mapping[str, tuple[str, ...]] | None = None,
        lazy_choices: Mapping[str, Callable[[], Sequence[str]]] | None = None,
        memory_consumers: Mapping[str, Shedder] | None = None,
    ) -> None:
        self._node_types = dict(node_types)
        self._registry = registry
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
        self._combo_choices = dict(combo_choices or {})
        self._lazy_choices = dict(lazy_choices or {})
        self._memory_consumers = dict(memory_consumers or {})
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
        return "in-process"

    @property
    def attention_route_token(self) -> AttentionRouteToken | None:
        return self._attention_route_token

    @property
    def attention_capabilities(self) -> AttentionCapabilityEvidence | None:
        return self._attention_capabilities

    def bind_registry(self, registry: TypeRegistry) -> None:
        """Bind a staged worker to the host registry before publication."""
        self._registry = registry

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
        with self._pack_context():
            return await self._invoke_in_context(invocation, on_event)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        del on_event
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
            expected_execution_identity=invocation.expected_execution_identity,
            fp8_matmul=invocation.fp8_matmul,
            diffusion_dtype=invocation.diffusion_dtype,
            text_dtype=invocation.text_dtype,
            vae_dtype=invocation.vae_dtype,
            attention_policy=invocation.attention_policy,
            attention_route_token=invocation.attention_route_token,
            extension_snapshot_digest=invocation.extension_snapshot_digest,
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
            extension_snapshot_digest=invocation.extension_snapshot_digest,
            node_id=invocation.node_id,
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

        skew = self._check_output_provenance(base_schema, invocation)
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
            kwargs = self._build_kwargs(base_schema, invocation, coercion_diagnostics, media_inputs)
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
                    preview_mode=invocation.preview_mode,
                    preview_animation=invocation.preview_animation,
                    node_id=invocation.node_id,
                    export_snapshot=invocation.export_snapshot,
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
                    preview_mode=invocation.preview_mode,
                    preview_animation=invocation.preview_animation,
                    node_id=invocation.node_id,
                    export_snapshot=invocation.export_snapshot,
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
                result = replace(result, artifact_candidates=tuple(artifact_candidates))
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

    def _check_output_provenance(
        self, base_schema: NodeSchema, invocation: Invocation
    ) -> str | None:
        """Guard against engine/worker schema skew: the invocation's family
        provenance, flattened, must produce exactly the dynamic outputs of the
        effective schema (effective outputs minus this worker's static base
        outputs). Membership is never re-derived by parsing output ids."""
        if base_schema.output_descriptors is not None:
            source_ids = {base_schema.output_descriptors.input}
            source_ids.update(
                family.count.input
                for family in base_schema.output_families
                if family.count is not None
            )
            stored: dict[str, object] = {
                input_id: None
                for input_id in (*invocation.inputs, *invocation.connected_undemanded_inputs)
            }
            try:
                for input_id in source_ids:
                    stored[input_id] = invocation.inputs[input_id].resolve()
                expected = elaborate(
                    base_schema,
                    stored,
                    dict(invocation.output_members),
                    dict(invocation.effective_schema.slot_choices),
                )
            except (KeyError, ValueError, UnresolvablePayload) as error:
                return f"schema skew: invalid stored output descriptors: {error}"
            if expected.outputs != invocation.effective_schema.outputs:
                return "schema skew: output descriptor interface differs from worker projection"
            return None
        provenance_families = {fid for fid, _ in invocation.output_members}
        declared_families = {fam.id for fam in base_schema.output_families}
        if provenance_families != declared_families:
            return (
                "schema skew: invocation output families "
                f"{sorted(provenance_families)} != worker's declared "
                f"{sorted(declared_families)}"
            )
        flattened: set[str] = set()
        for fid, suffixes in invocation.output_members:
            fam = base_schema.output_family(fid)
            assert fam is not None  # checked above
            flattened.update(fam.member_id(s) for s in suffixes)
        static_ids = {out.id for out in base_schema.outputs}
        effective_dynamic = {
            out.id for out in invocation.effective_schema.outputs if out.id not in static_ids
        }
        if flattened != effective_dynamic:
            return (
                "schema skew: flattened output members "
                f"{sorted(flattened)} != effective dynamic outputs "
                f"{sorted(effective_dynamic)}"
            )
        return None

    def _unwrap(
        self,
        invocation: Invocation,
        input_id: str,
        value: Value,
        diagnostics: list[dict[str, object]] | None = None,
        media_inputs: dict[str, Value] | None = None,
    ) -> object:
        obj = self._unwrap_value(invocation, input_id, value)
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
            result = apply_alpha_policy(obj, type_id, spec.alpha_policy)
            if media_inputs is not None and (type_id != value.type_id or result is not obj):
                media_inputs[input_id] = media_input_value(result, type_id, value)
            return result
        except ValueError as exc:
            raise InputCoercionError(f"input '{input_id}': {exc}") from exc

    def _unwrap_value(self, invocation: Invocation, input_id: str, value: Value) -> object:
        """The plain object node code receives for one input: the resolved
        payload - or, when the value is asset-typed and the declared input
        expects its decode target, the coerced result (typed assets, joint
        contract 2026-07-26). Decode runs HERE, worker-side, with THIS
        registry's providers: the engine plans and folds provider identity
        into the cache key, it never decodes. The plan is re-derived from
        the same pure planner over the same effective schema, so engine and
        worker can never disagree about which coercion applies."""
        spec = invocation.effective_schema.input(input_id)
        if spec is None or is_absent(value) or spec.type.accepts_concrete(value.type_id):
            return value.resolve()
        plan = plan_asset_coercion(value.type_id, spec.type)
        if plan is None:
            if combo_type_mismatch_is_error(value.type_id, spec.type):
                raise InputCoercionError(
                    f"input '{input_id}' rejects runtime type {value.type_id}: "
                    "core.combo mismatches require an explicit converter"
                )
            return value.resolve()
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
                return decoder.decode(value.resolve())
            children = list_children(value)
            if children is None:
                raise InputCoercionError(
                    f"input '{input_id}': {value.type_id} arrived without "
                    "list structure; its elements cannot be decoded"
                )
            decoded = [decoder.decode(child.resolve()) for child in children]
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
    ) -> dict[str, object]:
        """Plain values for execute(), boundary-obliviously (hazard H9).

        Static inputs arrive as ordinary kwargs. Input-family members are
        grouped into one mapping per family, keyed by member suffix in
        effective-interface (document) order - a node with family "operands"
        receives ``operands={"x": ..., "y": ...}``. A dynamic slot arrives as
        one SlotValue kwarg under the slot id: the connected value, the
        active variant key (explicit dispatch), and the variant's dependents
        under their LOCAL names; an optional slot with no stored choice is
        simply not passed (declare a default). Nodes with output families
        additionally receive the reserved ``output_spec`` parameter: an
        OutputInterface built from the invocation's canonical provenance.
        """

        def unwrap(input_id: str, value: Value) -> object:
            return self._unwrap(invocation, input_id, value, diagnostics, media_inputs)

        kwargs: dict[str, object] = {fam.id: {} for fam in base_schema.input_families}
        slot_choice = dict(invocation.effective_schema.slot_choices)
        for choice_id, choice in slot_choice.items():
            if _is_combo_choice(
                (*base_schema.input_families, *base_schema.combos, *base_schema.slots),
                choice_id.split("."),
            ):
                # DynamicCombo choices are document state, not ordinary
                # inputs. Compat wrappers need the active key under its
                # materialized construct path to rebuild ComfyUI's nested
                # submission dictionary. Top-level variant slots keep the
                # established SlotValue calling convention below.
                kwargs[choice_id] = choice
        slot_socket: dict[str, object] = {}
        slot_options: dict[str, dict[str, object]] = {
            slot.id: {} for slot in base_schema.slots if slot.id in slot_choice
        }
        for input_id, value in invocation.inputs.items():
            owner = base_schema.slot_of_input(input_id)
            if owner is not None:
                slot, local = owner
                if slot.variants is None:
                    kwargs[input_id] = unwrap(input_id, value)
                    continue
                assert slot.id in slot_options, (
                    f"slot input {input_id!r} without a slot choice on the "
                    "effective schema (engine/worker skew)"
                )
                if local is None:
                    slot_socket[slot.id] = unwrap(input_id, value)
                else:
                    slot_options[slot.id][local] = unwrap(input_id, value)
                continue
            fam = base_schema.family_of_input(input_id)
            if fam is None:
                kwargs[input_id] = unwrap(input_id, value)
            else:
                suffix = fam.member_suffix(input_id)
                assert suffix is not None
                members = kwargs[fam.id]
                assert isinstance(members, dict)
                members[suffix] = unwrap(input_id, value)
        for slot_id, options in slot_options.items():
            kwargs[slot_id] = SlotValue(
                variant=slot_choice[slot_id],
                value=slot_socket.get(slot_id),
                options=options,
            )
        for input_id in invocation.connected_undemanded_inputs:
            spec = invocation.effective_schema.input(input_id)
            marker = (None,) if spec is not None and spec.type.kind == "list" else None
            family = base_schema.family_of_input(input_id)
            if family is None:
                kwargs[input_id] = marker
            else:
                suffix = family.member_suffix(input_id)
                assert suffix is not None
                members = kwargs[family.id]
                assert isinstance(members, dict)
                members[suffix] = marker
        for family in base_schema.input_families:
            members = kwargs[family.id]
            assert isinstance(members, dict)
            ordered_members: dict[str, object] = {}
            for spec in invocation.effective_schema.inputs:
                suffix = family.member_suffix(spec.id)
                if suffix is not None and suffix in members:
                    ordered_members[suffix] = members[suffix]
            kwargs[family.id] = ordered_members
        if base_schema.output_families or base_schema.output_descriptors is not None:
            kwargs["output_spec"] = OutputInterface(
                members=invocation.output_members,
                outputs=invocation.effective_schema.outputs,
            )
        return kwargs

    def _wrap_outputs(
        self,
        invocation: Invocation,
        schema: NodeSchema,
        raw: object,
        media_inputs: Mapping[str, Value],
    ) -> InvocationResult:
        def contract_error(message: str) -> InvocationResult:
            return InvocationResult(
                error=NodeError(
                    node_id=invocation.node_id,
                    node_type=invocation.node_type,
                    message=message,
                )
            )

        if not isinstance(raw, Mapping):
            return contract_error(
                f"execute() must return a mapping keyed by output id, got {type(raw).__name__}"
            )

        # Flatten family-grouped returns ({family_id: {suffix: value}}) into
        # member output ids, then validate exactly against the elaborated
        # schema - a wrong member set fails here, not downstream.
        base_schema = self._schemas[invocation.node_type]
        flat: dict[str, object] = {}
        # Node returns are runtime data: keys/values are whatever the node
        # author produced, validated here at the contract boundary.
        for key, value in cast("Mapping[object, object]", raw).items():
            fam = base_schema.output_family(str(key))
            if fam is not None:
                if not isinstance(value, Mapping):
                    return contract_error(
                        f"output family '{key}' must be a mapping of member "
                        f"suffix -> value, got {type(value).__name__}"
                    )
                for suffix, member_value in cast("Mapping[object, object]", value).items():
                    flat[fam.member_id(str(suffix))] = member_value
            else:
                flat[str(key)] = value
        raw = flat

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
                # Deliberate absence (DESIGN 3.15): only outputs declared
                # optional may carry it, and this node is the origin - the
                # provenance downstream diagnostics will name.
                if not out.optional:
                    return contract_error(
                        f"output '{out.id}' returned ABSENT but is not declared optional"
                    )
                outputs[out.id] = make_absent_value(
                    origin=f"{invocation.node_id}/{out.id}",
                    reason=returned_value.reason,
                    stands_for=expected,
                )
                continue
            if expected is None:
                # Workers wrap outputs by declared type: after variable
                # solving, anything still unresolved (an unmentioned variable,
                # a wildcard/union output) is a schema bug in the node.
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
