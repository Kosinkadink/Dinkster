"""Torch-free authoring and runtime contracts for typed guidance phases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, cast

from dinkster_protocol import (
    BehaviorValue,
)
from dinkster_protocol import (
    GuidancePhaseParticipation as GuidancePhaseParticipation,
)

from .patches import SizedTensor
from .registry import validate_registry_id
from .sampling import SamplingExecutionContext
from .text_encoders import Conditioning

T = TypeVar("T", bound=SizedTensor)


class GuidanceRole(StrEnum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"
    AUXILIARY = "auxiliary"


class GuidancePredictionSource(StrEnum):
    MODEL = "model"
    SYNTHETIC_ZERO = "synthetic-zero"


@dataclass(frozen=True)
class ConditionScaleVector(Generic[T]):
    values: T


@dataclass(frozen=True)
class GuidanceCondition(Generic[T]):
    id: str
    role: GuidanceRole
    conditioning: Conditioning[T] | None
    scale_vector: ConditionScaleVector[T] | None = None


@dataclass(frozen=True)
class GuidancePlanContext(Generic[T]):
    input: T
    sigma: T
    cfg_scale: float
    conditions: tuple[GuidanceCondition[T], ...]
    force_uncond: bool
    execution: SamplingExecutionContext


@dataclass(frozen=True)
class GuidanceEvaluationPlan(Generic[T]):
    lanes: tuple[GuidanceCondition[T], ...]
    primary_id: str
    unconditional_id: str | None

    def __post_init__(self) -> None:
        ids = tuple(lane.id for lane in self.lanes)
        if len(ids) != len(set(ids)):
            raise ValueError("guidance lane ids must be unique")
        if self.primary_id not in ids:
            raise ValueError("primary_id must reference a lane")
        if self.unconditional_id is not None and self.unconditional_id not in ids:
            raise ValueError("unconditional_id must reference a lane")


@dataclass(frozen=True)
class GuidanceEvaluationRequest(Generic[T]):
    input: T
    sigma: T
    plan: GuidanceEvaluationPlan[T]
    execution: SamplingExecutionContext


@dataclass(frozen=True)
class GuidancePrediction(Generic[T]):
    lane_id: str
    value: T
    source: GuidancePredictionSource


@dataclass(frozen=True)
class GuidancePredictions(Generic[T]):
    items: tuple[GuidancePrediction[T], ...]

    def __post_init__(self) -> None:
        ids = tuple(item.lane_id for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("prediction lane ids must be unique")


@dataclass(frozen=True)
class GuidancePreCFGContext(Generic[T]):
    request: GuidanceEvaluationRequest[T]
    predictions: GuidancePredictions[T]
    cfg_scale: float


@dataclass(frozen=True)
class GuidanceReduceContext(Generic[T]):
    request: GuidanceEvaluationRequest[T]
    predictions: GuidancePredictions[T]
    cfg_scale: float


@dataclass(frozen=True)
class GuidancePostCFGContext(Generic[T]):
    request: GuidanceEvaluationRequest[T]
    predictions: GuidancePredictions[T]
    reduced: T
    cfg_scale: float
    evaluate_conditions: Callable[[GuidanceEvaluationRequest[T]], GuidancePredictions[T]] | None = (
        None
    )
    """Bounded non-reentrant raw model-evaluation service for auxiliary
    predictions (PAG-style), provided by the torch guidance executor; it
    bypasses guidance wrappers and the pre/reduce/post phases. The caller
    may pass a replaced request/plan but must preserve the original
    request's input, sigma, and execution context. ``None`` keeps the
    previous shape for existing constructors."""


@dataclass(frozen=True)
class GuidanceResult(Generic[T]):
    denoised: T
    unconditional: T | None
    predictions: GuidancePredictions[T]


class GuidancePlanFn(Protocol[T]):
    def __call__(self, context: GuidancePlanContext[T]) -> GuidanceEvaluationPlan[T]: ...


class GuidancePlanAugmentationFn(Protocol[T]):
    def __call__(
        self,
        context: GuidancePlanContext[T],
        plan: GuidanceEvaluationPlan[T],
    ) -> GuidanceEvaluationPlan[T]: ...


class GuidanceEvaluateNext(Protocol[T]):
    def __call__(self, request: GuidanceEvaluationRequest[T]) -> GuidancePredictions[T]: ...


class GuidanceEvaluationWrapper(Protocol[T]):
    def __call__(
        self, request: GuidanceEvaluationRequest[T], next: GuidanceEvaluateNext[T]
    ) -> GuidancePredictions[T]: ...


class GuidanceScaleTransform(Protocol[T]):
    def __call__(self, context: GuidancePlanContext[T]) -> float: ...


class GuidancePreCFGTransform(Protocol[T]):
    def __call__(self, context: GuidancePreCFGContext[T]) -> GuidancePredictions[T]: ...


class GuidanceReduceFn(Protocol[T]):
    def __call__(self, context: GuidanceReduceContext[T]) -> T: ...


class GuidancePostCFGTransform(Protocol[T]):
    def __call__(self, context: GuidancePostCFGContext[T]) -> T: ...


class GuidanceAttentionTransform(Protocol[T]):
    """Rewrite one self-attention output inside model evaluation.

    ``positive`` and ``negative`` are the self-attention output rows of
    the conditional and unconditional streams of one fused model call,
    at one attention site. The returned value replaces the conditional
    rows; the unconditional rows are left untouched. The transform must
    not mutate its arguments."""

    def __call__(self, positive: T, negative: T) -> T: ...


def _validate_descriptor(
    id: str,
    callback: object,
    metadata: tuple[tuple[str, BehaviorValue], ...],
    label: str = "guidance",
) -> None:
    validate_registry_id(id)
    if "." not in id:
        raise ValueError(f"{label} descriptor id must be namespace-qualified")
    if not callable(callback):
        raise TypeError(f"{label} callback must be callable")
    raw_metadata = cast("object", metadata)
    valid_metadata = isinstance(raw_metadata, tuple)
    if valid_metadata:
        for raw_item in cast("tuple[object, ...]", raw_metadata):
            if not isinstance(raw_item, tuple):
                valid_metadata = False
                break
            item = cast("tuple[object, ...]", raw_item)
            if (
                len(item) != 2
                or not isinstance(item[0], str)
                or not (item[1] is None or type(item[1]) in (str, int, bool))
            ):
                valid_metadata = False
                break
    if not valid_metadata:
        raise TypeError("behavior_metadata must contain (str, BehaviorValue) tuples")
    keys = tuple(key for key, _ in metadata)
    if keys != tuple(sorted(set(keys))) or any(not key.startswith("config.") for key in keys):
        raise ValueError("behavior_metadata keys must be sorted unique config.* names")


@dataclass(frozen=True)
class GuidanceEvaluationWrapperDescriptor(Generic[T]):
    id: str
    wrapper: GuidanceEvaluationWrapper[T]
    order: int = 0
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.wrapper, self.behavior_metadata)
        _validate_order(self.order)
        _validate_requires_uncond(self.requires_uncond)


@dataclass(frozen=True)
class GuidancePlanAugmentationDescriptor(Generic[T]):
    id: str
    augment: GuidancePlanAugmentationFn[T]
    order: int = 0
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.augment, self.behavior_metadata)
        _validate_order(self.order)
        _validate_requires_uncond(self.requires_uncond)


@dataclass(frozen=True)
class GuidanceScaleDescriptor(Generic[T]):
    id: str
    transform: GuidanceScaleTransform[T]
    order: int = 0
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.transform, self.behavior_metadata)
        _validate_order(self.order)
        _validate_requires_uncond(self.requires_uncond)


@dataclass(frozen=True)
class GuidancePreCFGDescriptor(Generic[T]):
    id: str
    transform: GuidancePreCFGTransform[T]
    order: int = 0
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.transform, self.behavior_metadata)
        _validate_order(self.order)
        _validate_requires_uncond(self.requires_uncond)


@dataclass(frozen=True)
class GuidanceStrategyDescriptor(Generic[T]):
    id: str
    plan: GuidancePlanFn[T]
    reduce: GuidanceReduceFn[T]
    participation: GuidancePhaseParticipation = GuidancePhaseParticipation.BYPASS_TRANSFORMS
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()
    allows_unconditional_primary: bool = False
    compatible_post_cfg_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.plan, self.behavior_metadata)
        if not callable(self.reduce):
            raise TypeError("guidance reduce must be callable")
        if not isinstance(self.participation, GuidancePhaseParticipation):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("participation must be GuidancePhaseParticipation")
        _validate_requires_uncond(self.requires_uncond)
        if type(self.allows_unconditional_primary) is not bool:
            raise TypeError("allows_unconditional_primary must be bool")
        raw_kinds = cast("object", self.compatible_post_cfg_kinds)
        if not isinstance(raw_kinds, tuple) or not all(
            isinstance(value, str) and value for value in cast("tuple[object, ...]", raw_kinds)
        ):
            raise TypeError("compatible_post_cfg_kinds must contain nonempty strings")
        if self.compatible_post_cfg_kinds != tuple(sorted(set(self.compatible_post_cfg_kinds))):
            raise ValueError("compatible_post_cfg_kinds must be sorted and unique")


@dataclass(frozen=True)
class GuidancePostCFGDescriptor(Generic[T]):
    id: str
    transform: GuidancePostCFGTransform[T]
    order: int = 0
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()
    composition_kind: str | None = None

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.transform, self.behavior_metadata)
        _validate_order(self.order)
        _validate_requires_uncond(self.requires_uncond)
        raw_kind = cast("object", self.composition_kind)
        if raw_kind is not None and (not isinstance(raw_kind, str) or not raw_kind):
            raise TypeError("composition_kind must be a nonempty string or None")


@dataclass(frozen=True)
class AttentionGuidanceDescriptor(Generic[T]):
    """Declare one attention-level guidance rewrite.

    Unlike the phase descriptors, this declaration is never executed by
    the guidance executor - it is not a phase. It is consumed inside
    model evaluation by a family that positively opts in; a compiled
    plan carrying attention-kind contributions that the active
    evaluator does not consume refuses before sampling. Descriptors
    apply in contribution order, each observing the previous rewrite,
    matching the reference's chained attn1 output patches."""

    id: str
    transform: GuidanceAttentionTransform[T]
    requires_uncond: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_descriptor(self.id, self.transform, self.behavior_metadata)
        _validate_requires_uncond(self.requires_uncond)


def _validate_order(order: int) -> None:
    if type(order) is not int or not -(2**31) <= order < 2**31:
        raise ValueError("order must be a signed 32-bit integer")


def _validate_requires_uncond(value: bool) -> None:
    if type(value) is not bool:
        raise TypeError("requires_uncond must be bool")


@dataclass(frozen=True)
class GuidanceContribution(Generic[T]):
    evaluation_wrappers: tuple[GuidanceEvaluationWrapperDescriptor[T], ...] = ()
    pre_cfg: tuple[GuidancePreCFGDescriptor[T], ...] = ()
    strategy: GuidanceStrategyDescriptor[T] | None = None
    post_cfg: tuple[GuidancePostCFGDescriptor[T], ...] = ()
    attention: AttentionGuidanceDescriptor[T] | None = None
    plan_augmentations: tuple[GuidancePlanAugmentationDescriptor[T], ...] = ()
    scale: tuple[GuidanceScaleDescriptor[T], ...] = ()

    def __post_init__(self) -> None:
        collections = (
            (
                "plan_augmentations",
                self.plan_augmentations,
                GuidancePlanAugmentationDescriptor,
            ),
            ("evaluation_wrappers", self.evaluation_wrappers, GuidanceEvaluationWrapperDescriptor),
            ("scale", self.scale, GuidanceScaleDescriptor),
            ("pre_cfg", self.pre_cfg, GuidancePreCFGDescriptor),
            ("post_cfg", self.post_cfg, GuidancePostCFGDescriptor),
        )
        for name, values, descriptor_type in collections:
            raw_values = cast("object", values)
            if not isinstance(raw_values, tuple) or not all(
                isinstance(value, descriptor_type)
                for value in cast("tuple[object, ...]", raw_values)
            ):
                raise TypeError(f"{name} must be a tuple of {descriptor_type.__name__} values")
        raw_strategy = cast("object", self.strategy)
        if raw_strategy is not None and not isinstance(raw_strategy, GuidanceStrategyDescriptor):
            raise TypeError("strategy must be a GuidanceStrategyDescriptor or None")
        raw_attention = cast("object", self.attention)
        if raw_attention is not None and not isinstance(raw_attention, AttentionGuidanceDescriptor):
            raise TypeError("attention must be an AttentionGuidanceDescriptor or None")
        if not (
            self.plan_augmentations
            or self.evaluation_wrappers
            or self.scale
            or self.pre_cfg
            or self.strategy
            or self.post_cfg
            or self.attention
        ):
            raise ValueError("guidance contribution must be nonempty")
        ids = (
            tuple(
                descriptor.id
                for descriptors in (
                    self.plan_augmentations,
                    self.evaluation_wrappers,
                    self.scale,
                    self.pre_cfg,
                    self.post_cfg,
                )
                for descriptor in descriptors
            )
            + (() if self.strategy is None else (self.strategy.id,))
            + (() if self.attention is None else (self.attention.id,))
        )
        if len(ids) != len(set(ids)):
            raise ValueError("guidance descriptor ids must be globally unique")


class GuidanceContractError(RuntimeError):
    pass


class GuidanceExtensionError(RuntimeError):
    pass
