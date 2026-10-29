"""Backend-agnostic classifier-free guidance declarations and policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from .guidance import GuidanceContribution

CondT = TypeVar("CondT")


class ConditioningBatchingMode(Enum):
    """User-selected policy for compatible conditioning lanes."""

    AUTO = "auto"
    FORCE_SEPARATE = "force-separate"
    MAX_FUSED_LANES = "max-fused-lanes"


@dataclass(frozen=True, slots=True)
class ConditioningBatching:
    """Typed override for the shared guidance batch planner.

    ``AUTO`` sizes batches from the family's activation-memory estimate and
    current free memory. ``FORCE_SEPARATE`` evaluates one lane per model call.
    ``MAX_FUSED_LANES`` caps each compatible batch at ``max_fused_lanes`` and
    bypasses automatic memory sizing.
    """

    mode: ConditioningBatchingMode = ConditioningBatchingMode.AUTO
    max_fused_lanes: int | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not ConditioningBatchingMode:
            raise TypeError("conditioning batching mode must be a ConditioningBatchingMode")
        if self.mode is ConditioningBatchingMode.MAX_FUSED_LANES:
            if type(self.max_fused_lanes) is not int or self.max_fused_lanes < 1:
                raise ValueError("max-fused-lanes batching requires a positive lane limit")
        elif self.max_fused_lanes is not None:
            raise ValueError("only max-fused-lanes batching accepts a lane limit")


def cfg_needs_uncond(cond_scale: float, *, disable_cfg1_optimization: bool = False) -> bool:
    """Whether the unconditional prediction must be computed at all
    (comfy/samplers.py sampling_function @ b78cec87: at scale 1 the
    uncond term cancels, so the model call is skipped)."""
    if disable_cfg1_optimization:
        return True
    return not math.isclose(cond_scale, 1.0)


def _validate_transforms(value: object) -> None:
    """Validate per-run guidance transforms as (owner_id, contribution) pairs."""
    if not isinstance(value, tuple):
        raise TypeError("transforms must be a tuple of (owner_id, contribution) pairs")
    for entry in cast("tuple[object, ...]", value):
        if not isinstance(entry, tuple):
            raise TypeError("transforms must be a tuple of (owner_id, contribution) pairs")
        items = cast("tuple[object, ...]", entry)
        if len(items) != 2:
            raise TypeError("transforms must be a tuple of (owner_id, contribution) pairs")
        owner, contribution = items
        if not isinstance(owner, str) or not owner:
            raise TypeError("transform owner id must be a non-empty string")
        if not isinstance(contribution, GuidanceContribution):
            raise TypeError("transform contribution must be a GuidanceContribution")


@dataclass(frozen=True, slots=True)
class SamplingGuidance(Generic[CondT]):
    """Classifier-free guidance inputs for one sampling run.

    Callers bundle the negative conditioning payload and the CFG
    scale here so a family sampling surface declares one guidance
    input instead of re-declaring uncond/cfg_scale. A family
    validates and transforms the ``uncond`` payload (device
    movement, packing) in its own conditioning type and may consult
    :attr:`needs_unconditional_lane` for those decisions; the cfg==1
    optimization, lane fan-out, and the combination math belong to
    the shared guidance executor. The default instance means plain
    conditional sampling.

    ``transforms`` carries per-run guidance contributions as
    ``(owner_id, contribution)`` pairs. The owner id is an opaque
    non-empty string naming whoever attached the contribution - an
    extension id, or an identity derived from a node for
    node-attached transforms. They apply to this sampling run only:
    the executor merges them with any load-time registry, refusing
    an owner id present in both.
    """

    uncond: CondT | None = None
    scale: float = 1.0
    transforms: tuple[tuple[str, GuidanceContribution[Any]], ...] = ()
    batching: ConditioningBatching = ConditioningBatching()
    disable_cfg1_optimization: bool = False

    def __post_init__(self) -> None:
        _validate_transforms(cast("object", self.transforms))
        if type(self.batching) is not ConditioningBatching:
            raise TypeError("batching must be a ConditioningBatching")
        if type(self.disable_cfg1_optimization) is not bool:
            raise TypeError("disable_cfg1_optimization must be a bool")

    @property
    def needs_unconditional_lane(self) -> bool:
        """Whether a real unconditional model evaluation is required:
        a payload is present and the scale does not cancel it
        (:func:`cfg_needs_uncond`). CFG++ samplers and guidance
        extensions can still force the lane; that decision belongs to
        the executor, not to this predicate."""
        return self.uncond is not None and cfg_needs_uncond(
            self.scale,
            disable_cfg1_optimization=self.disable_cfg1_optimization,
        )


@dataclass(frozen=True, slots=True)
class DualSamplingGuidance(Generic[CondT]):
    """Three-lane guidance inputs for positive, middle, and negative predictions."""

    middle: CondT
    uncond: CondT
    scale: float
    middle_scale: float
    nested: bool = False
    batching: ConditioningBatching = ConditioningBatching()
    disable_cfg1_optimization: bool = False
    transforms: tuple[tuple[str, GuidanceContribution[Any]], ...] = ()

    def __post_init__(self) -> None:
        if type(self.batching) is not ConditioningBatching:
            raise TypeError("batching must be a ConditioningBatching")
        _validate_transforms(cast("object", self.transforms))
        if type(self.disable_cfg1_optimization) is not bool:
            raise TypeError("disable_cfg1_optimization must be a bool")


@dataclass(frozen=True, slots=True)
class PerpNegSamplingGuidance(Generic[CondT]):
    """Perpendicular-negative guidance inputs
    (comfy_extras/nodes_perpneg.py Guider_PerpNeg @ b78cec87).

    Three lanes: the conditional prediction, the negative prediction
    at ``uncond``, and the empty-prompt prediction at ``empty``. The
    reduction removes from the negative delta its projection onto
    the positive delta (both relative to the empty prediction),
    scales the residual by ``neg_scale``, and applies classifier-free
    guidance at ``scale`` from the empty prediction.

    ``transforms`` carries per-run guidance contributions exactly as
    on :class:`SamplingGuidance`; the reference guider applies the
    sampler pre/post CFG hook chains around its combination, so
    per-run transforms compose with this guidance rather than
    replacing it.
    """

    uncond: CondT
    empty: CondT
    scale: float
    neg_scale: float
    transforms: tuple[tuple[str, GuidanceContribution[Any]], ...] = ()
    batching: ConditioningBatching = ConditioningBatching()
    disable_cfg1_optimization: bool = False

    def __post_init__(self) -> None:
        _validate_transforms(cast("object", self.transforms))
        if type(self.batching) is not ConditioningBatching:
            raise TypeError("batching must be a ConditioningBatching")
        if type(self.disable_cfg1_optimization) is not bool:
            raise TypeError("disable_cfg1_optimization must be a bool")


__all__ = [
    "ConditioningBatching",
    "ConditioningBatchingMode",
    "SamplingGuidance",
    "cfg_needs_uncond",
]
