"""Materialized torch guidance registry and deterministic phase executor."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import torch
from dinkster_inference import (
    AttentionGuidanceDescriptor,
    CancellationToken,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    DualSamplingGuidance,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidanceEvaluationRequest,
    GuidanceExtensionError,
    GuidancePlanContext,
    GuidancePostCFGContext,
    GuidancePreCFGContext,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceReduceContext,
    GuidanceResult,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    ModelTokenLayout,
    Parameterization,
    SamplingExecutionContext,
    TokenGridTransform,
    TokenLayoutError,
    cfg_needs_uncond,
    map_transforms,
)
from dinkster_inference.guidance import GuidancePhaseParticipation

from .cfg import cfg_combine
from .memory import get_free_memory
from .parameterizations import calculate_denoised
from .sampling_cache import active_guidance_evaluation_cache

Evaluator = Callable[[GuidanceEvaluationRequest[torch.Tensor]], GuidancePredictions[torch.Tensor]]
ReplicaEvaluator = Callable[
    [torch.Tensor, float, GuidanceEvaluationRequest[torch.Tensor]],
    GuidancePredictions[torch.Tensor],
]
ReplicaEvaluatorFactory = Callable[[ReplicaEvaluator], ReplicaEvaluator]
PreparedCondition = TypeVar("PreparedCondition")

_CONDITIONING_MEMORY_SAFETY_FACTOR = 1.5
_DEFAULT_CONDITIONING_BATCHING = ConditioningBatching()


def estimate_standard_activation_memory(
    input_shape: Sequence[int],
    conditions: tuple[object, ...],
    *,
    memory_usage_factor: float,
) -> int:
    """Estimate denoiser activation bytes without performing memory management.

    This is ComfyUI's conservative non-flash estimate: latent area times 0.15
    MiB and a family-specific factor. Families may declare a more specific pure
    estimator through :class:`ConditioningEvaluation`.
    """

    if not conditions:
        raise ValueError("activation memory estimation requires at least one condition")
    if not math.isfinite(memory_usage_factor) or memory_usage_factor <= 0.0:
        raise ValueError("activation memory usage factor must be finite and positive")
    shape = tuple(int(value) for value in input_shape)
    if len(shape) < 2 or any(value < 1 for value in shape):
        raise ValueError("activation memory estimation requires a nonempty latent shape")
    area = len(conditions) * shape[0] * math.prod(shape[2:])
    return math.ceil(area * 0.15 * memory_usage_factor * 1024 * 1024)


def _default_free_memory(device: torch.device) -> int:
    return get_free_memory(device).free_total


def _conditioning_shape(value: object) -> object | None:
    if type(value) is torch.Tensor:
        return ("tensor", tuple(value.shape))
    if type(value) is tuple:
        shapes = tuple(_conditioning_shape(item) for item in cast("tuple[object, ...]", value))
        if any(shape is None for shape in shapes):
            return None
        return (
            "tuple",
            shapes,
        )
    if value is None:
        return ("none",)
    return None


def _equal_conditioning_shapes(conditions: tuple[object, ...]) -> bool:
    if not conditions:
        return False
    first = _conditioning_shape(conditions[0])
    return first is not None and all(
        _conditioning_shape(condition) == first for condition in conditions[1:]
    )


class ConditioningValidationPath(StrEnum):
    LAYOUT_BACKED = "layout-backed"
    LEGACY_SHAPE_PREDICATE = "legacy-shape-predicate"
    LAYOUT_ABSENT = "layout-absent"
    SYNTHETIC_ZERO = "synthetic-zero"


@dataclass(frozen=True, slots=True)
class CompiledConditioningLane:
    lane_id: str
    role: GuidanceRole
    layout_digest: str | None
    token_transforms: tuple[tuple[str, str], ...]
    validation: ConditioningValidationPath
    call_index: int | None
    physical_index: int | None
    inner_calls: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()

    @property
    def fact_line(self) -> str:
        facts: dict[str, object] = {
            "call": self.call_index,
            "lane": self.lane_id,
            "layout": self.layout_digest,
            "physical": self.physical_index,
            "role": self.role.value,
            "transforms": [
                {"digest": digest, "identity": identity}
                for identity, digest in self.token_transforms
            ],
            "validation": self.validation.value,
        }
        if self.inner_calls:
            facts["inner_calls"] = [
                {
                    "index": index,
                    "layout": layout_digest,
                    "transforms": [
                        {"digest": digest, "identity": identity} for identity, digest in transforms
                    ],
                }
                for index, (layout_digest, transforms) in enumerate(self.inner_calls)
            ]
        return json.dumps(facts, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CompiledConditioningCall:
    lane_ids: tuple[str, ...]
    evaluator_identity: str
    layout_digest: str | None
    validation: ConditioningValidationPath
    inner_layout_digests: tuple[str, ...] = ()

    @property
    def fact_line(self) -> str:
        facts: dict[str, object] = {
            "evaluator": self.evaluator_identity,
            "lanes": self.lane_ids,
            "layout": self.layout_digest,
            "validation": self.validation.value,
        }
        if self.inner_layout_digests:
            facts["inner_calls"] = [
                {"index": index, "layout": digest}
                for index, digest in enumerate(self.inner_layout_digests)
            ]
        return json.dumps(facts, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CompiledConditioningPlan:
    """One immutable lane assignment and physical model-call plan."""

    lanes: tuple[CompiledConditioningLane, ...]
    calls: tuple[CompiledConditioningCall, ...]

    @property
    def fact_lines(self) -> tuple[str, ...]:
        return (
            "conditioning-lane-plan.v1",
            *(lane.fact_line for lane in self.lanes),
            *(call.fact_line for call in self.calls),
        )

    @property
    def plan_digest(self) -> str:
        facts = "\n".join((*self.fact_lines, ""))
        return hashlib.sha256(facts.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _PreparedLane(Generic[PreparedCondition]):
    index: int
    source: object
    role: GuidanceRole
    condition: PreparedCondition
    evaluator_identity: str
    layout: ModelTokenLayout | None
    token_transforms: tuple[TokenGridTransform, ...]
    validation: ConditioningValidationPath
    inner_calls: tuple[tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]], ...]


@dataclass(frozen=True)
class _PreparedEvaluationPlan(Generic[PreparedCondition]):
    sources: tuple[tuple[str, GuidanceRole, object | None], ...]
    groups: tuple[tuple[_PreparedLane[PreparedCondition], ...], ...]
    compiled: CompiledConditioningPlan

    def matches(self, plan: GuidanceEvaluationPlan[torch.Tensor]) -> bool:
        return len(self.sources) == len(plan.lanes) and all(
            lane_id == lane.id and role is lane.role and source is lane.conditioning
            for (lane_id, role, source), lane in zip(self.sources, plan.lanes, strict=True)
        )


@dataclass(frozen=True)
class ConditioningBatch(Generic[PreparedCondition]):
    latent: torch.Tensor
    model_input: torch.Tensor
    timestep: torch.Tensor
    sigma: float
    conditions: tuple[PreparedCondition, ...]
    context: object | None = None

    @property
    def batch_size(self) -> int:
        return self.latent.shape[0]

    @property
    def lane_count(self) -> int:
        return len(self.conditions)


@runtime_checkable
class ConditioningBatchAdapter(Protocol[PreparedCondition]):
    """Required hooks for conditioning evaluation.

    Optional stage hooks are ``_conditioning_model_input``,
    ``_stack_conditioning_model_input``, ``_conditioning_timestep_tensor``,
    ``_conditioning_timestep``, ``_conditioning_model_output``, and
    ``_conditioning_denoised``. Without an input hook, an adapter declares
    ``compute_dtype`` or ``_compute_dtype``.
    """

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[PreparedCondition, ...],
    ) -> None: ...

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[PreparedCondition],
    ) -> torch.Tensor: ...


def evaluate_conditioning_batch(
    evaluator: object,
    x: torch.Tensor,
    sigma: float,
    conditions: tuple[PreparedCondition, ...],
    context: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Evaluate one compatible conditioning batch through a family adapter."""

    # Specialized adapters are composition wrappers only: they must delegate inward to
    # this evaluator for casting, stacking, timesteps, and denoising. Context is adapter
    # data for the standard hook path and cannot cross this composition boundary.
    specialized = getattr(evaluator, "_evaluate_conditioning_batch", None)
    if specialized is not None:
        if context is not None:
            raise GuidanceContractError(
                "specialized conditioning batch adapter does not accept evaluation context"
            )
        outputs = specialized(x, sigma, conditions)
    else:
        if not isinstance(evaluator, ConditioningBatchAdapter):
            raise GuidanceContractError(
                "conditioning batch adapter must validate conditions and evaluate the model"
            )
        evaluator._validate_conditioning_batch(  # pyright: ignore[reportPrivateUsage]
            x, conditions
        )
        input_adapter = getattr(evaluator, "_conditioning_model_input", None)
        if input_adapter is None:
            compute_dtype = getattr(evaluator, "compute_dtype", None)
            if compute_dtype is None:
                compute_dtype = getattr(evaluator, "_compute_dtype", None)
            if compute_dtype is None:
                raise GuidanceContractError(
                    "conditioning batch adapter must declare its compute dtype"
                )
            model_input = x.to(dtype=compute_dtype)
        else:
            model_input = input_adapter(x, sigma)
        if model_input.shape[0] != x.shape[0]:
            raise GuidanceContractError(
                "conditioning batch model input changed the latent batch size"
            )
        stack_adapter = getattr(evaluator, "_stack_conditioning_model_input", None)
        if stack_adapter is not None:
            model_input = stack_adapter(model_input, conditions)
        elif len(conditions) > 1:
            model_input = torch.cat((model_input,) * len(conditions), dim=0)
        timestep_tensor_adapter = getattr(evaluator, "_conditioning_timestep_tensor", None)
        timestep_adapter = getattr(evaluator, "_conditioning_timestep", None)
        timestep_value = (
            timestep_tensor_adapter(sigma, x.device)
            if timestep_tensor_adapter is not None
            else sigma
            if timestep_adapter is None
            else timestep_adapter(sigma)
        )
        timestep = (
            timestep_value.to(device=x.device, dtype=torch.float32).expand(model_input.shape[0])
            if isinstance(timestep_value, torch.Tensor)
            else torch.full(
                (model_input.shape[0],),
                timestep_value,
                dtype=torch.float32,
                device=x.device,
            )
        )
        batch = ConditioningBatch(x, model_input, timestep, sigma, conditions, context)
        output = evaluator._evaluate_conditioning_model(  # pyright: ignore[reportPrivateUsage]
            batch
        )
        output_adapter = getattr(evaluator, "_conditioning_model_output", None)
        if output_adapter is not None:
            output = output_adapter(batch, output)
        flow_input = x if len(conditions) == 1 else torch.cat((x,) * len(conditions), dim=0)
        denoised_adapter = getattr(evaluator, "_conditioning_denoised", None)
        denoised = (
            calculate_denoised(Parameterization.FLOW, sigma, output, flow_input)
            if denoised_adapter is None
            else denoised_adapter(batch, output, flow_input)
        )
        outputs = tuple(denoised.chunk(len(conditions)))
    if type(outputs) is not tuple or len(outputs) != len(conditions):
        raise GuidanceContractError(
            "conditioning batch evaluation must return one tensor per condition"
        )
    expected_shape = tuple(x.shape)
    if any(
        type(output) is not torch.Tensor or tuple(output.shape) != expected_shape
        for output in outputs
    ):
        raise GuidanceContractError("conditioning batch evaluation returned an incompatible tensor")
    return outputs


@dataclass(frozen=True)
class ConditioningEvaluation(Generic[PreparedCondition]):
    """Family-declared conditioning evaluation used by shared guidance.

    A family that supplies ``evaluate_batch`` may omit ``batchable`` when its
    prepared values are tensors or tuples of tensors; equal tensor shapes are
    then the compatibility rule. Families bind their catalog
    ``memory_factor`` through ``standard_activation_memory_factor`` or supply
    a custom ``estimate_activation_memory`` function. Both contracts are pure:
    they report bytes for a candidate lane tuple and never read free memory or
    move model state.
    """

    prepare: Callable[[object, GuidanceRole], PreparedCondition]
    evaluate: Callable[[torch.Tensor, float, PreparedCondition], torch.Tensor]
    batchable: Callable[[tuple[PreparedCondition, ...]], bool] | None = None
    evaluate_batch: (
        Callable[[torch.Tensor, float, tuple[PreparedCondition, ...]], tuple[torch.Tensor, ...]]
        | None
    ) = None
    evaluate_batch_attention: (
        Callable[
            [
                torch.Tensor,
                float,
                tuple[PreparedCondition, ...],
                tuple[GuidanceRole, ...],
                tuple[AttentionGuidanceDescriptor[torch.Tensor], ...],
            ],
            tuple[torch.Tensor, ...],
        ]
        | None
    ) = None
    evaluator_identity: Callable[[GuidanceRole], str] | None = None
    layout: Callable[[object], ModelTokenLayout | None] | None = None
    fused_layout: Callable[[tuple[ModelTokenLayout, ...]], ModelTokenLayout] | None = None
    token_transforms: Callable[[object], tuple[TokenGridTransform, ...]] | None = None
    validate_layout: Callable[[PreparedCondition, ModelTokenLayout], None] | None = None
    inner_calls: (
        Callable[
            [PreparedCondition],
            tuple[tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]], ...],
        ]
        | None
    ) = None
    estimate_activation_memory: (
        Callable[[Sequence[int], tuple[PreparedCondition, ...]], int] | None
    ) = None
    standard_activation_memory_factor: float | None = None

    def __post_init__(self) -> None:
        if self.batchable is not None and self.evaluate_batch is None:
            raise TypeError("batchable requires evaluate_batch")
        if self.evaluate_batch_attention is not None and self.evaluate_batch is None:
            raise TypeError("evaluate_batch_attention requires evaluate_batch")
        factor = self.standard_activation_memory_factor
        if factor is not None and (not math.isfinite(factor) or factor <= 0.0):
            raise ValueError("standard activation memory factor must be finite and positive")
        if factor is not None and self.estimate_activation_memory is not None:
            raise TypeError("a custom activation memory estimator cannot use the standard factor")

    def activation_memory_required(
        self,
        input_shape: Sequence[int],
        conditions: tuple[PreparedCondition, ...],
    ) -> int:
        factor = self.standard_activation_memory_factor
        if factor is not None:
            return estimate_standard_activation_memory(
                input_shape,
                cast("tuple[object, ...]", conditions),
                memory_usage_factor=factor,
            )
        if self.estimate_activation_memory is not None:
            return self.estimate_activation_memory(input_shape, conditions)
        raise GuidanceContractError(
            "AUTO conditioning batching requires a family activation-memory estimate"
        )

    def evaluate_request(
        self,
        x: torch.Tensor,
        sigma: float,
        request: GuidanceEvaluationRequest[torch.Tensor],
        on_plan: Callable[[CompiledConditioningPlan], None] | None = None,
        attention: tuple[AttentionGuidanceDescriptor[torch.Tensor], ...] = (),
    ) -> GuidancePredictions[torch.Tensor]:
        return ConditioningPlanCompiler(
            self,
            request.plan.lanes,
            request.input,
            ConditioningBatching(),
        ).evaluate_request(
            x,
            sigma,
            request,
            on_plan,
            attention,
        )


@dataclass(frozen=True, kw_only=True)
class RoutedConditioning(Conditioning[torch.Tensor]):
    """Conditioning evaluated by a different model on its guidance lane."""

    evaluation: ConditioningEvaluation[Any]
    source: Conditioning[torch.Tensor] | None = None


@dataclass(frozen=True)
class _RoutedPreparedConditioning:
    evaluation: ConditioningEvaluation[Any]
    value: object


def routed_conditioning_evaluation(
    primary: ConditioningEvaluation[Any],
) -> ConditioningEvaluation[_RoutedPreparedConditioning]:
    def prepare(source: object, role: GuidanceRole) -> _RoutedPreparedConditioning:
        evaluation = source.evaluation if isinstance(source, RoutedConditioning) else primary
        value = (
            source.source
            if isinstance(source, RoutedConditioning) and source.source is not None
            else source
        )
        return _RoutedPreparedConditioning(evaluation, evaluation.prepare(value, role))

    def evaluate(
        x: torch.Tensor,
        sigma: float,
        prepared: _RoutedPreparedConditioning,
    ) -> torch.Tensor:
        return prepared.evaluation.evaluate(x, sigma, prepared.value)

    return ConditioningEvaluation(
        prepare,
        evaluate,
        evaluator_identity=lambda _role: "dinkster.routed-conditioning.v1",
    )


class ConditioningPlanCompiler(Generic[PreparedCondition]):
    def __init__(
        self,
        evaluation: ConditioningEvaluation[PreparedCondition],
        lanes: tuple[GuidanceCondition[torch.Tensor], ...],
        input: torch.Tensor,
        batching: ConditioningBatching,
        free_memory_reader: Callable[[torch.device], int] = _default_free_memory,
    ) -> None:
        self._evaluation = evaluation
        self._input_shape = tuple(input.shape)
        self._input_device = input.device
        self._batching = batching
        self._free_memory_reader = free_memory_reader
        self._prepared: list[_PreparedLane[PreparedCondition]] = []
        self._plans: list[_PreparedEvaluationPlan[PreparedCondition]] = []
        for index, lane in enumerate(lanes):
            if lane.conditioning is not None:
                self._prepare(index, lane)

    def _prepare(
        self,
        index: int,
        lane: GuidanceCondition[torch.Tensor],
    ) -> _PreparedLane[PreparedCondition]:
        source = lane.conditioning
        if source is None:
            raise GuidanceContractError("synthetic conditioning lanes cannot be prepared")
        for entry in self._prepared:
            if entry.source is source and entry.role is lane.role:
                return replace(entry, index=index)

        identity = (
            "legacy.single-conditioning.v1"
            if self._evaluation.evaluator_identity is None
            else self._evaluation.evaluator_identity(lane.role)
        )
        if (
            type(identity) is not str
            or not identity
            or identity != identity.strip()
            or any(character in identity for character in "\r\n")
        ):
            raise GuidanceContractError(
                "conditioning evaluator identity must be a non-empty single-line string"
            )
        layout = None if self._evaluation.layout is None else self._evaluation.layout(source)
        if layout is not None and type(layout) is not ModelTokenLayout:
            raise GuidanceContractError(
                "conditioning layout must be an exact ModelTokenLayout or None"
            )
        condition = self._evaluation.prepare(source, lane.role)
        transforms = (
            ()
            if self._evaluation.token_transforms is None
            else self._evaluation.token_transforms(source)
        )
        if type(transforms) is not tuple or any(
            type(transform) is not TokenGridTransform for transform in transforms
        ):
            raise GuidanceContractError(
                "conditioning token transforms must be exact TokenGridTransform values"
            )
        if layout is None:
            if transforms:
                raise GuidanceContractError(
                    "conditioning token transforms require a declared layout"
                )
            validation = (
                ConditioningValidationPath.LEGACY_SHAPE_PREDICATE
                if self._evaluation.batchable is not None
                else ConditioningValidationPath.LAYOUT_ABSENT
            )
        else:
            if self._evaluation.validate_layout is None:
                raise GuidanceContractError(
                    "layout-backed conditioning requires a tensor consistency validator"
                )
            try:
                transforms = tuple(map_transforms(layout, transforms).values())
                self._evaluation.validate_layout(condition, layout)
            except TokenLayoutError as error:
                raise GuidanceContractError(
                    f"conditioning layout validation failed: {error}"
                ) from None
            validation = ConditioningValidationPath.LAYOUT_BACKED
        inner_calls: tuple[tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]], ...] = ()
        if self._evaluation.inner_calls is not None:
            try:
                declared_inner_calls = self._evaluation.inner_calls(condition)
                if type(declared_inner_calls) is not tuple or any(
                    type(call) is not tuple
                    or len(call) != 2
                    or type(call[0]) is not ModelTokenLayout
                    or type(call[1]) is not tuple
                    or any(type(transform) is not TokenGridTransform for transform in call[1])
                    for call in declared_inner_calls
                ):
                    raise TokenLayoutError(
                        "inner calls must contain exact layouts and transform tuples"
                    )
                inner_calls = tuple(
                    (inner_layout, tuple(map_transforms(inner_layout, inner_transforms).values()))
                    for inner_layout, inner_transforms in declared_inner_calls
                )
            except TokenLayoutError as error:
                raise GuidanceContractError(
                    f"conditioning inner-call declaration failed: {error}"
                ) from None
            if not inner_calls:
                raise GuidanceContractError(
                    "conditioning inner-call declaration must contain at least one call"
                )
        prepared = _PreparedLane(
            -1,
            source,
            lane.role,
            condition,
            identity,
            layout,
            transforms,
            validation,
            inner_calls,
        )
        self._prepared.append(prepared)
        return replace(prepared, index=index)

    def prepare_plan(
        self,
        plan: GuidanceEvaluationPlan[torch.Tensor],
    ) -> CompiledConditioningPlan:
        for prepared_plan in self._plans:
            if prepared_plan.matches(plan):
                return prepared_plan.compiled
        prepared_plan = self._compile(plan)
        self._plans.append(prepared_plan)
        return prepared_plan.compiled

    def _compile(
        self,
        plan: GuidanceEvaluationPlan[torch.Tensor],
    ) -> _PreparedEvaluationPlan[PreparedCondition]:
        prepared = [
            self._prepare(index, lane)
            for index, lane in enumerate(plan.lanes)
            if lane.conditioning is not None
        ]
        has_declared_layout = any(entry.layout is not None for entry in prepared)
        if has_declared_layout and any(entry.layout is None for entry in prepared):
            raise GuidanceContractError("conditioning lanes cannot mix declared and absent layouts")
        has_inner_calls = any(entry.inner_calls for entry in prepared)
        if has_inner_calls and any(not entry.inner_calls for entry in prepared):
            raise GuidanceContractError(
                "conditioning lanes cannot mix direct and inner-call evaluation"
            )
        compatible_groups: list[list[_PreparedLane[PreparedCondition]]] = []
        if self._evaluation.evaluate_batch is None:
            compatible_groups.extend([entry] for entry in prepared)
        else:
            for entry in prepared:
                matched = False
                for group in compatible_groups:
                    candidate = (*group, entry)
                    conditions = tuple(item.condition for item in candidate)
                    compatible = (
                        _equal_conditioning_shapes(cast("tuple[object, ...]", conditions))
                        if self._evaluation.batchable is None
                        else self._evaluation.batchable(conditions)
                    )
                    if not compatible:
                        continue
                    if has_declared_layout:
                        layouts = tuple(item.layout for item in candidate)
                        assert all(layout is not None for layout in layouts)
                        typed_layouts = cast("tuple[ModelTokenLayout, ...]", layouts)
                        if self._evaluation.fused_layout is None:
                            if any(layout != typed_layouts[0] for layout in typed_layouts[1:]):
                                raise GuidanceContractError(
                                    "conditioning layout mismatch prevents fused evaluation"
                                )
                        else:
                            try:
                                fused_layout = self._evaluation.fused_layout(typed_layouts)
                            except TokenLayoutError as error:
                                raise GuidanceContractError(
                                    f"conditioning fused layout failed: {error}"
                                ) from None
                            if type(fused_layout) is not ModelTokenLayout:
                                raise GuidanceContractError(
                                    "conditioning fused layout must be an exact ModelTokenLayout"
                                )
                    if has_inner_calls and any(
                        tuple(layout for layout, _transforms in item.inner_calls)
                        != tuple(layout for layout, _transforms in entry.inner_calls)
                        for item in group
                    ):
                        raise GuidanceContractError(
                            "conditioning inner-call layout mismatch prevents fused evaluation"
                        )
                    if any(item.evaluator_identity != entry.evaluator_identity for item in group):
                        raise GuidanceContractError(
                            "conditioning evaluator identity mismatch prevents fused evaluation"
                        )
                    group.append(entry)
                    matched = True
                    break
                if not matched:
                    compatible_groups.append([entry])

        groups = [
            batch
            for compatible_group in compatible_groups
            for batch in self._size_compatible_group(compatible_group)
        ]

        ordered_groups = tuple(
            tuple(reversed(group)) if len(group) > 1 else tuple(group) for group in groups
        )
        call_assignments = {
            entry.index: (call_index, physical_index)
            for call_index, group in enumerate(ordered_groups)
            for physical_index, entry in enumerate(group)
        }
        by_index = {entry.index: entry for entry in prepared}
        compiled_lanes = []
        for index, lane in enumerate(plan.lanes):
            entry = by_index.get(index)
            if entry is None:
                compiled_lanes.append(
                    CompiledConditioningLane(
                        lane.id,
                        lane.role,
                        None,
                        (),
                        ConditioningValidationPath.SYNTHETIC_ZERO,
                        None,
                        None,
                    )
                )
                continue
            call_index, physical_index = call_assignments[index]
            compiled_lanes.append(
                CompiledConditioningLane(
                    lane.id,
                    lane.role,
                    None if entry.layout is None else entry.layout.digest,
                    tuple(
                        (transform.transform, transform.digest)
                        for transform in entry.token_transforms
                    ),
                    entry.validation,
                    call_index,
                    physical_index,
                    tuple(
                        (
                            inner_layout.digest,
                            tuple(
                                (transform.transform, transform.digest)
                                for transform in inner_transforms
                            ),
                        )
                        for inner_layout, inner_transforms in entry.inner_calls
                    ),
                )
            )
        compiled = CompiledConditioningPlan(
            tuple(compiled_lanes),
            tuple(
                CompiledConditioningCall(
                    tuple(plan.lanes[entry.index].id for entry in group),
                    group[0].evaluator_identity,
                    (
                        None
                        if group[0].layout is None
                        else (
                            group[0].layout.digest
                            if self._evaluation.fused_layout is None
                            else self._evaluation.fused_layout(
                                cast(
                                    "tuple[ModelTokenLayout, ...]",
                                    tuple(entry.layout for entry in group),
                                )
                            ).digest
                        )
                    ),
                    group[0].validation,
                    tuple(layout.digest for layout, _transforms in group[0].inner_calls),
                )
                for group in ordered_groups
            ),
        )
        return _PreparedEvaluationPlan(
            tuple((lane.id, lane.role, lane.conditioning) for lane in plan.lanes),
            ordered_groups,
            compiled,
        )

    def _size_compatible_group(
        self,
        group: list[_PreparedLane[PreparedCondition]],
    ) -> tuple[tuple[_PreparedLane[PreparedCondition], ...], ...]:
        if len(group) < 2:
            return (tuple(group),)
        mode = self._batching.mode
        if mode is ConditioningBatchingMode.FORCE_SEPARATE:
            limit = 1
        elif mode is ConditioningBatchingMode.MAX_FUSED_LANES:
            assert self._batching.max_fused_lanes is not None
            limit = self._batching.max_fused_lanes
        else:
            limit = len(group)
        remaining = list(group)
        batches: list[tuple[_PreparedLane[PreparedCondition], ...]] = []
        while remaining:
            largest = min(limit, len(remaining))
            if mode is ConditioningBatchingMode.AUTO and largest > 1:
                free_memory = self._free_memory_reader(self._input_device)
                if type(free_memory) is not int or free_memory < 0:
                    raise GuidanceContractError(
                        "conditioning free-memory reader must return a non-negative integer"
                    )
                while largest > 1:
                    conditions = tuple(item.condition for item in remaining[:largest])
                    required = self._evaluation.activation_memory_required(
                        self._input_shape,
                        conditions,
                    )
                    if type(required) is not int or required < 0:
                        raise GuidanceContractError(
                            "conditioning activation-memory estimator must return a"
                            " non-negative integer"
                        )
                    # ComfyUI's calc_cond_batch rule admits a fused call only when
                    # memory_required * 1.5 is strictly below current free memory.
                    if required * _CONDITIONING_MEMORY_SAFETY_FACTOR < free_memory:
                        break
                    largest -= 1
            batches.append(tuple(remaining[:largest]))
            del remaining[:largest]
        return tuple(batches)

    def _attention_group(
        self,
        plan: GuidanceEvaluationPlan[torch.Tensor],
        prepared_plan: _PreparedEvaluationPlan[PreparedCondition],
        attention: tuple[AttentionGuidanceDescriptor[torch.Tensor], ...],
    ) -> tuple[_PreparedLane[PreparedCondition], ...]:
        ids = ", ".join(item.id for item in attention)
        if self._evaluation.evaluate_batch_attention is None:
            raise GuidanceContractError(
                f"attention-kind guidance contributions ({ids}) require a family"
                " evaluator that consumes them; this evaluator declares none"
            )
        conditional = [
            index for index, lane in enumerate(plan.lanes) if lane.role is GuidanceRole.CONDITIONAL
        ]
        if len(conditional) != 1:
            raise GuidanceContractError(
                f"attention-kind guidance contributions ({ids}) require exactly one"
                f" conditional lane; the plan has {len(conditional)}"
            )
        unconditional = [
            index
            for index, lane in enumerate(plan.lanes)
            if lane.role is GuidanceRole.UNCONDITIONAL
        ]
        if len(unconditional) != 1 or plan.lanes[unconditional[0]].conditioning is None:
            raise GuidanceContractError(
                f"attention-kind guidance contributions ({ids}) require an"
                " unconditional lane carrying real conditioning; a synthetic"
                " unconditional prediction has no attention output to rewrite"
            )
        for group in prepared_plan.groups:
            indices = {entry.index for entry in group}
            if conditional[0] in indices and unconditional[0] in indices:
                return group
        # The reference (calc_cond_batch at pin b78cec87) splits the
        # conditional and unconditional lanes into separate forwards on
        # shape mismatch or free-memory pressure, and its attn1 output
        # patch then silently no-ops, so upstream application varies
        # with VRAM. Dinkster deliberately refuses instead of silently
        # skipping the rewrite.
        raise GuidanceContractError(
            f"attention-kind guidance contributions ({ids}) require the"
            " conditional and unconditional lanes to evaluate in one fused"
            " forward, and this plan evaluates them separately"
        )

    def evaluate_request(
        self,
        x: torch.Tensor,
        sigma: float,
        request: GuidanceEvaluationRequest[torch.Tensor],
        on_plan: Callable[[CompiledConditioningPlan], None] | None = None,
        attention: tuple[AttentionGuidanceDescriptor[torch.Tensor], ...] = (),
    ) -> GuidancePredictions[torch.Tensor]:
        prepared_plan = next(
            (item for item in self._plans if item.matches(request.plan)),
            None,
        )
        if prepared_plan is None:
            prepared_plan = self._compile(request.plan)
            self._plans.append(prepared_plan)
        if on_plan is not None:
            on_plan(prepared_plan.compiled)

        attention_group = (
            self._attention_group(request.plan, prepared_plan, attention) if attention else None
        )
        predictions = {
            index: GuidancePrediction(
                lane.id,
                torch.zeros_like(x),
                GuidancePredictionSource.SYNTHETIC_ZERO,
            )
            for index, lane in enumerate(request.plan.lanes)
            if lane.conditioning is None
        }
        for group in prepared_plan.groups:
            if attention_group is not None and group is attention_group:
                evaluate_batch_attention = self._evaluation.evaluate_batch_attention
                if evaluate_batch_attention is None:
                    raise GuidanceContractError("conditioning attention evaluator is missing")
                values = evaluate_batch_attention(
                    x,
                    sigma,
                    tuple(item.condition for item in group),
                    tuple(request.plan.lanes[item.index].role for item in group),
                    attention,
                )
                if (
                    not isinstance(values, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
                    or len(values) != len(group)
                ):
                    raise GuidanceContractError(
                        "conditioning attention evaluator returned the wrong number of predictions"
                    )
            elif len(group) == 1:
                entry = group[0]
                values = (self._evaluation.evaluate(x, sigma, entry.condition),)
            else:
                evaluate_batch = self._evaluation.evaluate_batch
                if evaluate_batch is None:
                    raise GuidanceContractError("conditioning batch evaluator is missing")
                values = evaluate_batch(
                    x,
                    sigma,
                    tuple(item.condition for item in group),
                )
                if (
                    not isinstance(values, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
                    or len(values) != len(group)
                ):
                    raise GuidanceContractError(
                        "conditioning batch evaluator returned the wrong number of predictions"
                    )
            for entry, value in zip(group, values, strict=True):
                predictions[entry.index] = GuidancePrediction(
                    request.plan.lanes[entry.index].id,
                    value,
                    GuidancePredictionSource.MODEL,
                )
        return GuidancePredictions(
            tuple(predictions[index] for index in range(len(request.plan.lanes)))
        )


@dataclass(frozen=True)
class _Owned:
    extension: str
    value: Any


class GuidanceRegistry:
    """Process-local callbacks retained with their owning extension identity."""

    def __init__(
        self, contributions: tuple[tuple[str, GuidanceContribution[torch.Tensor]], ...] = ()
    ):
        self.contributions = tuple(contributions)
        wrappers: list[_Owned] = []
        scale: list[_Owned] = []
        pre: list[_Owned] = []
        post: list[_Owned] = []
        attention: list[_Owned] = []
        strategy: _Owned | None = None
        plan_augmentations: list[_Owned] = []
        for extension, contribution in contributions:
            plan_augmentations.extend(
                _Owned(extension, item) for item in contribution.plan_augmentations
            )
            wrappers.extend(_Owned(extension, item) for item in contribution.evaluation_wrappers)
            scale.extend(_Owned(extension, item) for item in contribution.scale)
            pre.extend(_Owned(extension, item) for item in contribution.pre_cfg)
            post.extend(_Owned(extension, item) for item in contribution.post_cfg)
            if contribution.attention is not None:
                attention.append(_Owned(extension, contribution.attention))
            if contribution.strategy is not None:
                if strategy is not None:
                    raise GuidanceContractError(
                        "multiple guidance strategies were materialized: "
                        f"extension={strategy.extension} contribution={strategy.value.id}; "
                        f"extension={extension} contribution={contribution.strategy.id}"
                    )
                strategy = _Owned(extension, contribution.strategy)
        self.plan_augmentations = tuple(
            sorted(plan_augmentations, key=lambda x: (x.value.order, x.value.id))
        )
        if self.plan_augmentations and (wrappers or pre):
            paired = wrappers[0] if wrappers else pre[0]
            phase = "evaluation-wrapper" if wrappers else "pre-CFG"
            raise GuidanceContractError(
                "plan-augmentation guidance contribution "
                f"extension={self.plan_augmentations[0].extension}"
                f" contribution={self.plan_augmentations[0].value.id}"
                f" cannot compose with {phase} guidance contribution "
                f"extension={paired.extension} contribution={paired.value.id}"
            )
        if (
            strategy is not None
            and self.plan_augmentations
            and strategy.value.participation is GuidancePhaseParticipation.BYPASS_TRANSFORMS
        ):
            raise GuidanceContractError(
                "guidance strategy "
                f"extension={strategy.extension} contribution={strategy.value.id}"
                " bypasses transforms and cannot compose with plan-augmentation contribution "
                f"extension={self.plan_augmentations[0].extension}"
                f" contribution={self.plan_augmentations[0].value.id}"
            )
        if strategy is not None and attention:
            # No executed reference coverage pins how a plan-owning
            # strategy composes with attention-level rewrites, so the
            # pairing refuses rather than shipping untested semantics.
            raise GuidanceContractError(
                "guidance strategy "
                f"extension={strategy.extension} contribution={strategy.value.id}"
                " cannot compose with attention-kind contribution "
                f"extension={attention[0].extension} contribution={attention[0].value.id}"
            )
        if strategy is not None and strategy.value.compatible_post_cfg_kinds:
            compatible_kinds = set(strategy.value.compatible_post_cfg_kinds)
            incompatible = [*wrappers, *pre, *plan_augmentations] + [
                item for item in post if item.value.composition_kind not in compatible_kinds
            ]
            if incompatible:
                item = incompatible[0]
                raise GuidanceContractError(
                    "guidance strategy cannot compose with guidance contribution "
                    f"strategy={strategy.value.id} "
                    f"extension={item.extension} contribution={item.value.id}"
                )
            if len(post) > 1:
                raise GuidanceContractError(
                    "guidance strategy accepts at most one compatible post-CFG contribution: "
                    f"strategy={strategy.value.id}"
                )
        self.wrappers = tuple(sorted(wrappers, key=lambda x: (x.value.order, x.value.id)))  # type: ignore[attr-defined]
        self.scale = tuple(sorted(scale, key=lambda x: (x.value.order, x.value.id)))  # type: ignore[attr-defined]
        self.pre = tuple(sorted(pre, key=lambda x: (x.value.order, x.value.id)))  # type: ignore[attr-defined]
        self.post = tuple(sorted(post, key=lambda x: (x.value.order, x.value.id)))  # type: ignore[attr-defined]
        # Attention rewrites keep contribution order: the reference
        # chains attn1 output patches in attachment order.
        self.attention = tuple(attention)
        self.strategy = strategy

    @property
    def active(self) -> bool:
        return bool(
            self.plan_augmentations
            or self.wrappers
            or self.scale
            or self.pre
            or self.post
            or self.strategy
            or self.attention
        )

    @property
    def requires_uncond(self) -> bool:
        values = (
            *self.plan_augmentations,
            *self.wrappers,
            *self.scale,
            *self.pre,
            *self.post,
            *self.attention,
        )
        return any(item.value.requires_uncond for item in values) or bool(
            self.strategy and self.strategy.value.requires_uncond
        )


def merged_guidance_executor(
    executor: GuidanceExecutor | None,
    transforms: tuple[tuple[str, GuidanceContribution[torch.Tensor]], ...],
) -> GuidanceExecutor | None:
    """Merge per-run guidance contributions into one executor.

    Without ``transforms`` the load-time ``executor`` is returned
    unchanged (identity preserved, including None). With transforms,
    the merged registry holds the load-time contributions followed by
    the per-run ones in their given order. Owner ids are opaque
    strings (extension ids, node-derived identities); an owner id
    holding contributions on both sides is refused because the merge
    would silently double its callbacks."""
    if not transforms:
        return executor
    base = executor.registry.contributions if executor is not None else ()
    registered = {owner for owner, _ in base}
    for owner, _ in transforms:
        if owner in registered:
            raise GuidanceContractError(
                f"per-run guidance transform owner {owner!r} is already registered"
            )
        registered.add(owner)
    return GuidanceExecutor(GuidanceRegistry(base + transforms))


def _standard_plan_values(
    conditions: tuple[GuidanceCondition[torch.Tensor], ...],
    cfg_scale: float,
    force_uncond: bool,
) -> GuidanceEvaluationPlan[torch.Tensor]:
    lanes = tuple(lane for lane in conditions if lane.role is GuidanceRole.CONDITIONAL)
    uncond = next((lane for lane in conditions if lane.role is GuidanceRole.UNCONDITIONAL), None)
    # Reference sampling_function semantics (CfgDenoiser): with no
    # unconditional CONDITIONING the primary prediction is returned
    # untouched - no synthetic-zero CFG. Only force_uncond (CFG++
    # samplers, requires_uncond contributions) admits a synthetic
    # lane, whose prediction is calc_cond_batch's untouched zeros.
    need_uncond = force_uncond or (
        uncond is not None and uncond.conditioning is not None and cfg_needs_uncond(cfg_scale)
    )
    if uncond is not None and need_uncond:
        lanes += (uncond,)
    unconditional_id = uncond.id if uncond is not None and need_uncond else None
    return GuidanceEvaluationPlan(lanes, lanes[0].id, unconditional_id)


def _standard_plan(
    context: GuidancePlanContext[torch.Tensor],
) -> GuidanceEvaluationPlan[torch.Tensor]:
    return _standard_plan_values(
        context.conditions,
        context.cfg_scale,
        context.force_uncond,
    )


def _validate_predictions(
    request: GuidanceEvaluationRequest[torch.Tensor], predictions: object, owner: str
) -> None:
    if not isinstance(predictions, GuidancePredictions):
        raise GuidanceContractError(f"{owner} returned invalid predictions type")
    expected = tuple(lane.id for lane in request.plan.lanes)
    actual = tuple(item.lane_id for item in predictions.items)
    if actual != expected:
        raise GuidanceContractError(
            f"{owner} prediction lane order/count mismatch: expected {expected}, got {actual}"
        )
    for lane, item in zip(request.plan.lanes, predictions.items, strict=True):
        value = item.value
        source_value: object = item.source
        if not isinstance(source_value, GuidancePredictionSource):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise GuidanceContractError(f"{owner} prediction {item.lane_id!r} has invalid source")
        if item.source is GuidancePredictionSource.SYNTHETIC_ZERO and lane.conditioning is not None:
            raise GuidanceContractError(
                f"{owner} prediction {item.lane_id!r} marked synthetic-zero for a model lane"
            )
        if item.source is GuidancePredictionSource.MODEL and lane.conditioning is None:
            raise GuidanceContractError(
                f"{owner} prediction {item.lane_id!r} marked model for a synthetic lane"
            )
        if not isinstance(value, torch.Tensor):
            raise GuidanceContractError(f"{owner} prediction {item.lane_id!r} is not a tensor")
        if value.shape != request.input.shape:
            raise GuidanceContractError(f"{owner} prediction {item.lane_id!r} shape mismatch")
        if value.dtype != request.input.dtype:
            raise GuidanceContractError(f"{owner} prediction {item.lane_id!r} dtype mismatch")
        if value.device != request.input.device:
            raise GuidanceContractError(f"{owner} prediction {item.lane_id!r} device mismatch")
        if item.source is GuidancePredictionSource.SYNTHETIC_ZERO and torch.count_nonzero(value):
            raise GuidanceContractError(
                f"{owner} prediction {item.lane_id!r} synthetic-zero value is not zero"
            )


def _validate_plan(
    context: GuidancePlanContext[torch.Tensor],
    plan: object,
    owner: str,
    *,
    allows_unconditional_primary: bool = False,
) -> GuidanceEvaluationPlan[torch.Tensor]:
    if not isinstance(plan, GuidanceEvaluationPlan):
        raise GuidanceContractError(f"{owner} plan returned invalid output type")
    available = {lane.id: lane for lane in context.conditions}
    if not plan.lanes:
        raise GuidanceContractError(f"{owner} plan has no lanes")
    for lane in plan.lanes:
        source = available.get(lane.id)
        if source is None:
            raise GuidanceContractError(
                f"{owner} plan lane {lane.id!r} is not an available condition"
            )
        if lane.role is not source.role:
            raise GuidanceContractError(f"{owner} plan lane {lane.id!r} changed its role")
        if lane.scale_vector is not None:
            raise GuidanceContractError(
                "condition scale vectors are not supported by the torch runtime"
            )
    primary = next((lane for lane in plan.lanes if lane.id == plan.primary_id), None)
    if primary is None:
        raise GuidanceContractError(f"{owner} plan primary lane is not present")
    if primary.role is not GuidanceRole.CONDITIONAL and not allows_unconditional_primary:
        raise GuidanceContractError(f"{owner} plan primary lane must be conditional")
    if plan.unconditional_id is not None:
        uncond = next((lane for lane in plan.lanes if lane.id == plan.unconditional_id), None)
        if uncond is None:
            raise GuidanceContractError(f"{owner} plan unconditional lane is not present")
        if uncond.role is not GuidanceRole.UNCONDITIONAL:
            raise GuidanceContractError(f"{owner} plan unconditional lane has the wrong role")
    if context.force_uncond and plan.unconditional_id is None:
        raise GuidanceContractError(f"{owner} plan must provide an unconditional lane")
    return plan


def _validate_plan_augmentation(
    original: GuidanceEvaluationPlan[torch.Tensor],
    plan: object,
    owner: str,
) -> GuidanceEvaluationPlan[torch.Tensor]:
    if not isinstance(plan, GuidanceEvaluationPlan):
        raise GuidanceContractError(f"{owner} returned invalid output type")
    if (
        plan.primary_id != original.primary_id
        or plan.unconditional_id != original.unconditional_id
        or len(plan.lanes) < len(original.lanes)
        or any(
            lane is not original_lane
            for lane, original_lane in zip(plan.lanes, original.lanes, strict=False)
        )
    ):
        raise GuidanceContractError(
            f"{owner} must preserve the existing plan and append only auxiliary lanes"
        )
    existing_ids = {lane.id for lane in original.lanes}
    for lane in plan.lanes[len(original.lanes) :]:
        if lane.id in existing_ids:
            raise GuidanceContractError(f"{owner} appended duplicate lane id {lane.id!r}")
        existing_ids.add(lane.id)
        if lane.role is not GuidanceRole.AUXILIARY or lane.conditioning is None:
            raise GuidanceContractError(
                f"{owner} appended lane {lane.id!r} must be an auxiliary model evaluation"
            )
        if lane.scale_vector is not None:
            raise GuidanceContractError(
                "condition scale vectors are not supported by the torch runtime"
            )
    return plan


def _validate_wrapper_request(
    original: GuidanceEvaluationRequest[torch.Tensor], request: object, owner: str
) -> GuidanceEvaluationRequest[torch.Tensor]:
    if not isinstance(request, GuidanceEvaluationRequest):
        raise GuidanceContractError(f"{owner} wrapper next received invalid request type")
    if (
        request.input is not original.input
        or request.sigma is not original.sigma
        or request.execution is not original.execution
    ):
        raise GuidanceContractError(
            f"{owner} wrapper next changed request execution tensors or context"
        )
    expected = tuple((lane.id, lane.role) for lane in original.plan.lanes)
    actual = tuple((lane.id, lane.role) for lane in request.plan.lanes)
    if (
        actual != expected
        or request.plan.primary_id != original.plan.primary_id
        or request.plan.unconditional_id != original.plan.unconditional_id
    ):
        raise GuidanceContractError(f"{owner} wrapper next changed plan lane set/order or roles")
    for lane in request.plan.lanes:
        if lane.scale_vector is not None:
            raise GuidanceContractError(
                "condition scale vectors are not supported by the torch runtime"
            )
    return request


class GuidanceExecutor:
    def __init__(self, registry: GuidanceRegistry):
        self.registry = registry

    def execute(
        self,
        context: GuidancePlanContext[torch.Tensor],
        evaluate: Evaluator,
        *,
        attention_consumed: bool = False,
    ) -> GuidanceResult[torch.Tensor]:
        cancel = context.execution.cancellation.check
        for owned in self.registry.scale:
            value = self._call(
                cancel,
                owned.value.id,
                owned,
                "scale",
                owned.value.transform,
                context,
            )
            if type(value) is not float:
                raise GuidanceContractError(
                    "guidance scale transform "
                    f"extension={owned.extension} contribution={owned.value.id}"
                    " returned a non-float scale"
                )
            context = replace(context, cfg_scale=value)
        # Attention-kind contributions are not an execution phase: they
        # rewrite attention outputs inside model evaluation, upstream of
        # every CFG phase, so a caller that does not thread them to a
        # consuming evaluator must refuse rather than silently sample
        # without them.
        if self.registry.attention and not attention_consumed:
            ids = ", ".join(item.value.id for item in self.registry.attention)
            raise GuidanceContractError(
                f"attention-kind guidance contributions ({ids}) must be consumed"
                " by the family's model evaluation; this execution path does not"
                " consume them"
            )
        strategy = self.registry.strategy
        # _standard_plan follows the reference: an unconditional lane
        # carrying no CONDITIONING is never planned, so the conditional
        # prediction is returned untouched and CFG-scale combination
        # never happens. Contributions registered around that
        # combination would silently run on the untouched prediction,
        # and planning a synthetic-zero lane instead would move wired
        # values, so the ambiguous
        # pairing refuses. A wholly absent unconditional lane keeps
        # its cond-only plan (that surface never offered CFG), and a
        # contribution that needs an unconditional lane declares
        # requires_uncond, which plans the synthetic lane explicitly
        # through force_uncond.
        if (
            strategy is None
            and self.registry.active
            and not context.force_uncond
            and cfg_needs_uncond(context.cfg_scale)
        ):
            uncond = next(
                (lane for lane in context.conditions if lane.role is GuidanceRole.UNCONDITIONAL),
                None,
            )
            if uncond is not None and uncond.conditioning is None:
                raise GuidanceContractError(
                    "guidance contributions with a cfg scale other than one"
                    " require unconditional conditioning"
                )
        plan_fn = _standard_plan if strategy is None else strategy.value.plan
        plan_owner = (
            "builtin strategy"
            if strategy is None
            else f"extension={strategy.extension} contribution={strategy.value.id} phase=plan"
        )
        plan = _validate_plan(
            context,
            self._call(cancel, "strategy", strategy, "plan", plan_fn, context),
            plan_owner,
            allows_unconditional_primary=bool(
                strategy and strategy.value.allows_unconditional_primary
            ),
        )
        for owned in self.registry.plan_augmentations:
            owner = (
                f"extension={owned.extension} contribution={owned.value.id} phase=plan-augmentation"
            )
            plan = _validate_plan_augmentation(
                plan,
                self._call(
                    cancel,
                    owned.value.id,
                    owned,
                    "plan-augmentation",
                    owned.value.augment,
                    context,
                    plan,
                ),
                owner,
            )
        request = GuidanceEvaluationRequest(context.input, context.sigma, plan, context.execution)
        invoke_stack: list[
            Callable[
                [int, GuidanceEvaluationRequest[torch.Tensor]],
                GuidancePredictions[torch.Tensor],
            ]
        ] = []

        def invoke(
            index: int, req: GuidanceEvaluationRequest[torch.Tensor]
        ) -> GuidancePredictions[torch.Tensor]:
            if index == len(self.registry.wrappers):
                cancel()
                result = evaluate(req)
                cancel()
                _validate_predictions(req, result, "core evaluation")
                return result
            owned = self.registry.wrappers[index]
            owner = f"extension={owned.extension} contribution={owned.value.id} phase=wrapper"
            calls = [0]
            delegated = [req]
            delegated_error: list[Exception | None] = [None]

            def next_(
                replacement: GuidanceEvaluationRequest[torch.Tensor],
            ) -> GuidancePredictions[torch.Tensor]:
                calls[0] += 1
                if calls[0] != 1:
                    raise GuidanceContractError(f"{owner} called next multiple times")
                delegated[0] = _validate_wrapper_request(req, replacement, owner)
                try:
                    return invoke_stack[0](index + 1, delegated[0])
                except Exception as error:
                    # The outer wrapper did not originate this failure. Keep
                    # its exact object and attribution when it escapes.
                    delegated_error[0] = error
                    raise

            try:
                result = self._call(
                    cancel, owned.value.id, owned, "wrapper", owned.value.wrapper, req, next_
                )
            except GuidanceExtensionError as error:
                if delegated_error[0] is not None and error.__cause__ is delegated_error[0]:
                    raise delegated_error[0] from None
                raise
            if calls[0] != 1:
                raise GuidanceContractError(f"{owner} did not call next")
            _validate_predictions(delegated[0], result, owner)
            return cast("GuidancePredictions[torch.Tensor]", result)

        invoke_stack.append(invoke)
        try:
            predictions = invoke(0, request)
        finally:
            invoke_stack.clear()
        compose = (
            strategy is None or strategy.value.participation is GuidancePhaseParticipation.COMPOSE
        )
        if compose:
            for owned in self.registry.pre:
                predictions = cast(
                    "GuidancePredictions[torch.Tensor]",
                    self._call(
                        cancel,
                        owned.value.id,
                        owned,
                        "pre",
                        owned.value.transform,
                        GuidancePreCFGContext(request, predictions, context.cfg_scale),
                    ),
                )
                _validate_predictions(
                    request,
                    predictions,
                    f"extension={owned.extension} contribution={owned.value.id} phase=pre",
                )
        if strategy is None:
            by_id = {item.lane_id: item.value for item in predictions.items}
            reduced = by_id[plan.primary_id]
            if plan.unconditional_id is not None and plan.unconditional_id in by_id:
                reduced = cfg_combine(reduced, by_id[plan.unconditional_id], context.cfg_scale)
        else:
            reduced = cast(
                "torch.Tensor",
                self._call(
                    cancel,
                    strategy.value.id,
                    strategy,
                    "reduce",
                    strategy.value.reduce,
                    GuidanceReduceContext(request, predictions, context.cfg_scale),
                ),
            )
        reducer_owner = (
            "builtin reducer"
            if strategy is None
            else f"extension={strategy.extension} contribution={strategy.value.id} phase=reduce"
        )
        self._validate_tensor(request.input, reduced, reducer_owner)
        if compose:
            for owned in self.registry.post:
                reduced = cast(
                    "torch.Tensor",
                    self._call(
                        cancel,
                        owned.value.id,
                        owned,
                        "post",
                        owned.value.transform,
                        GuidancePostCFGContext(request, predictions, reduced, context.cfg_scale),
                    ),
                )
                self._validate_tensor(
                    request.input,
                    reduced,
                    f"extension={owned.extension} contribution={owned.value.id} phase=post",
                )
        uncond = next(
            (item.value for item in predictions.items if item.lane_id == plan.unconditional_id),
            None,
        )
        return GuidanceResult(reduced, uncond, predictions)

    @staticmethod
    def _validate_tensor(reference: torch.Tensor, value: object, phase: str) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise GuidanceContractError(f"{phase} returned incompatible tensor")

    @staticmethod
    def _call(
        cancel: Callable[[], None],
        contribution: str,
        owned: _Owned | None,
        phase: str,
        fn: Callable[..., object],
        *args: object,
    ):
        cancel()
        try:
            result = fn(*args)
        except Exception as error:
            from dinkster_inference import SamplingCancelled

            if isinstance(
                error, (SamplingCancelled, GuidanceContractError, GuidanceExtensionError)
            ):
                raise
            extension = "builtin" if owned is None else owned.extension
            raise GuidanceExtensionError(
                f"extension={extension} contribution={contribution} phase={phase}: {error}"
            ) from error
        cancel()
        return result


def dual_cfg_executor(guidance: DualSamplingGuidance[object]) -> GuidanceExecutor:
    """Build ComfyUI-compatible three-lane regular or nested CFG execution."""

    def plan(context: GuidancePlanContext[torch.Tensor]) -> GuidanceEvaluationPlan[torch.Tensor]:
        conditions = {lane.id: lane for lane in context.conditions}
        positive = conditions["positive"]
        middle = conditions["middle"]
        negative = conditions["negative"]
        needs_middle = cfg_needs_uncond(guidance.scale)
        needs_negative = cfg_needs_uncond(guidance.middle_scale)
        includes_negative = guidance.nested or context.force_uncond or needs_negative
        if guidance.nested or context.force_uncond:
            lanes = (negative, middle, positive)
        else:
            lanes = (
                *((negative,) if needs_negative else ()),
                *((middle,) if needs_negative or needs_middle else ()),
                positive,
            )
        return GuidanceEvaluationPlan(
            lanes,
            "positive",
            "middle" if guidance.nested else ("negative" if includes_negative else None),
        )

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        predictions = {item.lane_id: item.value for item in context.predictions.items}
        positive = predictions["positive"]
        middle = predictions.get("middle")
        if middle is None:
            middle = torch.zeros_like(positive)
        negative = predictions.get("negative")
        if negative is None:
            negative = torch.zeros_like(positive)
        if guidance.nested:
            text = cfg_combine(positive, middle, guidance.scale)
            return negative + guidance.middle_scale * (text - negative)
        return (
            cfg_combine(middle, negative, guidance.middle_scale)
            + (positive - middle) * guidance.scale
        )

    strategy = GuidanceStrategyDescriptor(
        "dinkster.dual-cfg",
        plan,
        reduce,
        participation=GuidancePhaseParticipation.COMPOSE,
        behavior_metadata=(("config.style", "nested" if guidance.nested else "regular"),),
    )
    return GuidanceExecutor(
        GuidanceRegistry((("dinkster.dual-cfg", GuidanceContribution(strategy=strategy)),))
    )


class GuidedDenoiser:
    """Shared Flux/SD adapter that executes one materialized guidance graph."""

    def __init__(
        self,
        conditioning: ConditioningEvaluation[Any],
        executor: GuidanceExecutor,
        conditions: tuple[GuidanceCondition[torch.Tensor], ...],
        *,
        cfg_scale: float,
        force_uncond: bool,
        input: torch.Tensor,
        execution: SamplingExecutionContext,
        batching: ConditioningBatching = _DEFAULT_CONDITIONING_BATCHING,
        replica_evaluator_factory: ReplicaEvaluatorFactory | None = None,
        free_memory_reader: Callable[[torch.device], int] = _default_free_memory,
    ) -> None:
        self._executor = executor
        self._conditions = conditions
        self._cfg_scale = cfg_scale
        self._force_uncond = force_uncond
        self._execution = execution
        self._evaluation = 0
        self._attention = tuple(item.value for item in executor.registry.attention)
        if self._attention and replica_evaluator_factory is not None:
            ids = ", ".join(item.id for item in self._attention)
            raise GuidanceContractError(
                f"attention-kind guidance contributions ({ids}) are not supported"
                " on replica-evaluated (distributed) sampling"
            )
        admission_plan = (
            None
            if executor.registry.strategy is not None or executor.registry.plan_augmentations
            else _standard_plan_values(conditions, cfg_scale, force_uncond)
        )
        admitted_lanes = conditions if admission_plan is None else admission_plan.lanes
        self._conditioning = ConditioningPlanCompiler(
            conditioning,
            admitted_lanes,
            input,
            batching,
            free_memory_reader,
        )
        self._conditioning_plan = (
            None if admission_plan is None else self._conditioning.prepare_plan(admission_plan)
        )
        self._replica_evaluator = (
            None
            if replica_evaluator_factory is None
            else replica_evaluator_factory(self._conditioning.evaluate_request)
        )

    @property
    def conditioning_plan(self) -> CompiledConditioningPlan | None:
        return self._conditioning_plan

    def _execute(self, x: torch.Tensor, sigma: float) -> GuidanceResult[torch.Tensor]:
        execution = replace(
            self._execution,
            model_evaluation=self._evaluation,
            current_sigma=float(sigma),
        )
        from .distributed import (
            distributed_sampling_config,
            rank_zero_sampling_active,
            synchronized_sampling_call,
        )

        if distributed_sampling_config() is not None and torch.distributed.is_initialized():
            local_cancellation = execution.cancellation

            def cancelled() -> bool:
                synchronized_sampling_call(local_cancellation.check, x.device, "cancellation")
                return False

            cancellation = CancellationToken(cancelled)
            execution = replace(
                execution,
                cancellation=cancellation,
                progress=replace(execution.progress, cancellation=cancellation),
            )
        self._evaluation += 1
        context = GuidancePlanContext(
            x,
            torch.full((), sigma, device=x.device, dtype=x.dtype),
            self._cfg_scale,
            self._conditions,
            self._force_uncond,
            execution,
        )
        evaluator = None if rank_zero_sampling_active() else self._replica_evaluator

        def evaluate(
            request: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            if evaluator is not None:
                return evaluator(x, sigma, request)
            plan = self._conditioning.prepare_plan(request.plan)
            self._conditioning_plan = plan
            cache = active_guidance_evaluation_cache()

            def evaluate_conditioning(
                value: GuidanceEvaluationRequest[torch.Tensor],
            ) -> GuidancePredictions[torch.Tensor]:
                return self._conditioning.evaluate_request(
                    x,
                    sigma,
                    value,
                    attention=self._attention,
                )

            if cache is not None:
                return cache.evaluate(request, plan, evaluate_conditioning)
            return evaluate_conditioning(request)

        return self._executor.execute(
            context,
            evaluate,
            attention_consumed=True,
        )

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return self._execute(x, sigma).denoised

    def call_with_uncond(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        result = self._execute(x, sigma)
        if result.unconditional is None:
            raise GuidanceContractError("guidance execution did not produce unconditional output")
        return result.denoised, result.unconditional


__all__ = [
    "CompiledConditioningCall",
    "CompiledConditioningLane",
    "CompiledConditioningPlan",
    "ConditioningBatch",
    "ConditioningEvaluation",
    "ConditioningValidationPath",
    "dual_cfg_executor",
    "evaluate_conditioning_batch",
    "GuidanceExecutor",
    "GuidanceRegistry",
    "GuidedDenoiser",
]
