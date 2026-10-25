"""Native conditioning and CLIP hook nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    HOOKS_TYPE,
    MASK_TYPE,
    TIMESTEPS_RANGE_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_CATEGORY = "advanced/hooks"
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_HOOKS = TypeExpr.concrete(HOOKS_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_TIMESTEPS_RANGE = TypeExpr.concrete(TIMESTEPS_RANGE_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)


def _hooks() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.hooks")


def _properties_inputs(*inputs: InputSpec) -> tuple[InputSpec, ...]:
    return (
        *inputs,
        InputSpec(
            "strength",
            _FLOAT,
            required=False,
            default=1.0,
            widget=NumberWidget(min=0.0, max=10.0, step=0.01),
        ),
        InputSpec(
            "set_cond_area",
            _COMBO,
            required=False,
            default="default",
            widget=ComboWidget(("default", "mask bounds")),
        ),
        InputSpec("mask", _MASK, required=False),
        InputSpec("hooks", _HOOKS, required=False),
        InputSpec("timesteps", _TIMESTEPS_RANGE, required=False),
    )


def _properties(
    conds: Sequence[Sequence[Any]],
    strength: float,
    set_cond_area: str,
    mask: object | None,
    hooks: object | None,
    timesteps: tuple[float, float] | None,
) -> list[list[Any]]:
    return _hooks().set_conds_props(
        list(conds),
        strength,
        set_cond_area,
        mask=None if mask is None else materialize_image(mask),
        hooks=hooks,
        timesteps_range=timesteps,
    )


class ConditioningSetProperties(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConditioningSetProperties",
            display_name="Cond Set Props",
            category=f"{_CATEGORY}/cond single",
            inputs=_properties_inputs(InputSpec("cond_NEW", _CONDITIONING)),
            outputs=(OutputSpec("conditioning", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        cond_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        return cls.outputs(
            conditioning=_properties([cond_NEW], strength, set_cond_area, mask, hooks, timesteps)[0]
        )


class ConditioningSetPropertiesAndCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConditioningSetPropertiesAndCombine",
            display_name="Cond Set Props Combine",
            category=f"{_CATEGORY}/cond single",
            inputs=_properties_inputs(
                InputSpec("cond", _CONDITIONING), InputSpec("cond_NEW", _CONDITIONING)
            ),
            outputs=(OutputSpec("conditioning", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        cond: Sequence[Sequence[Any]],
        cond_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        return cls.outputs(
            conditioning=_hooks().set_conds_props_and_combine(
                [cond],
                [cond_NEW],
                strength,
                set_cond_area,
                mask=None if mask is None else materialize_image(mask),
                hooks=hooks,
                timesteps_range=timesteps,
            )[0]
        )


class PairConditioningSetProperties(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PairConditioningSetProperties",
            display_name="Cond Pair Set Props",
            category=f"{_CATEGORY}/cond pair",
            inputs=_properties_inputs(
                InputSpec("positive_NEW", _CONDITIONING), InputSpec("negative_NEW", _CONDITIONING)
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("negative", _CONDITIONING)),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive_NEW: Sequence[Sequence[Any]],
        negative_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        positive, negative = _properties(
            [positive_NEW, negative_NEW], strength, set_cond_area, mask, hooks, timesteps
        )
        return cls.outputs(positive=positive, negative=negative)


class PairConditioningSetPropertiesAndCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PairConditioningSetPropertiesAndCombine",
            display_name="Cond Pair Set Props Combine",
            category=f"{_CATEGORY}/cond pair",
            inputs=_properties_inputs(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("positive_NEW", _CONDITIONING),
                InputSpec("negative_NEW", _CONDITIONING),
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("negative", _CONDITIONING)),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        positive_NEW: Sequence[Sequence[Any]],
        negative_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        positive, negative = _hooks().set_conds_props_and_combine(
            [positive, negative],
            [positive_NEW, negative_NEW],
            strength,
            set_cond_area,
            mask=None if mask is None else materialize_image(mask),
            hooks=hooks,
            timesteps_range=timesteps,
        )
        return cls.outputs(positive=positive, negative=negative)


class ConditioningSetDefaultCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConditioningSetDefaultCombine",
            display_name="Cond Set Default Combine",
            category=f"{_CATEGORY}/cond single",
            inputs=(
                InputSpec("cond", _CONDITIONING),
                InputSpec("cond_DEFAULT", _CONDITIONING),
                InputSpec("hooks", _HOOKS, required=False),
            ),
            outputs=(OutputSpec("conditioning", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        cond: Sequence[Sequence[Any]],
        cond_DEFAULT: Sequence[Sequence[Any]],
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        return cls.outputs(
            conditioning=_hooks().set_default_conds_and_combine(
                [cond], [cond_DEFAULT], hooks=hooks
            )[0]
        )


class PairConditioningSetDefaultCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PairConditioningSetDefaultCombine",
            display_name="Cond Pair Set Default Combine",
            category=f"{_CATEGORY}/cond pair",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("positive_DEFAULT", _CONDITIONING),
                InputSpec("negative_DEFAULT", _CONDITIONING),
                InputSpec("hooks", _HOOKS, required=False),
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("negative", _CONDITIONING)),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        positive_DEFAULT: Sequence[Sequence[Any]],
        negative_DEFAULT: Sequence[Sequence[Any]],
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        positive, negative = _hooks().set_default_conds_and_combine(
            [positive, negative], [positive_DEFAULT, negative_DEFAULT], hooks=hooks
        )
        return cls.outputs(positive=positive, negative=negative)


class PairConditioningCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PairConditioningCombine",
            display_name="Cond Pair Combine",
            category=f"{_CATEGORY}/cond pair",
            inputs=(
                InputSpec("positive_A", _CONDITIONING),
                InputSpec("negative_A", _CONDITIONING),
                InputSpec("positive_B", _CONDITIONING),
                InputSpec("negative_B", _CONDITIONING),
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("negative", _CONDITIONING)),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive_A: Sequence[Sequence[Any]],
        negative_A: Sequence[Sequence[Any]],
        positive_B: Sequence[Sequence[Any]],
        negative_B: Sequence[Sequence[Any]],
    ) -> Mapping[str, object]:
        positive, negative = _hooks().set_conds_props_and_combine(
            [positive_A, negative_A], [positive_B, negative_B]
        )
        return cls.outputs(positive=positive, negative=negative)


class SetClipHooks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SetClipHooks",
            display_name="Set CLIP Hooks",
            category=f"{_CATEGORY}/clip",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec("apply_to_conds", _BOOLEAN, required=False, default=True, advanced=True),
                InputSpec("schedule_clip", _BOOLEAN, required=False, default=False, advanced=True),
                InputSpec("hooks", _HOOKS, required=False),
            ),
            outputs=(OutputSpec("clip", _CLIP),),
            occupies=("gpu",),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        apply_to_conds: bool = True,
        schedule_clip: bool = False,
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        if hooks is None:
            return cls.outputs(clip=clip)
        inference_hooks = _hooks()
        hook_group = cast(Any, hooks)
        result = clip.clone(disable_dynamic=True)
        if apply_to_conds:
            result.apply_hooks_to_conds = hook_group
        result.patcher.forced_hooks = hook_group.clone()
        result.use_clip_schedule = schedule_clip
        if not result.use_clip_schedule:
            result.patcher.forced_hooks.set_keyframes_on_hooks(None)
        result.patcher.register_all_hook_patches(
            hook_group, inference_hooks.create_target_dict(inference_hooks.EnumWeightTarget.Clip)
        )
        return cls.outputs(clip=result)


class ConditioningTimestepsRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConditioningTimestepsRange",
            display_name="Timesteps Range",
            category=_CATEGORY,
            inputs=(
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(
                OutputSpec("TIMESTEPS_RANGE", _TIMESTEPS_RANGE),
                OutputSpec("BEFORE_RANGE", _TIMESTEPS_RANGE),
                OutputSpec("AFTER_RANGE", _TIMESTEPS_RANGE),
            ),
        )

    @classmethod
    def execute(
        cls, *, start_percent: float = 0.0, end_percent: float = 1.0
    ) -> Mapping[str, object]:
        return cls.outputs(
            TIMESTEPS_RANGE=(start_percent, end_percent),
            BEFORE_RANGE=(0.0, start_percent),
            AFTER_RANGE=(end_percent, 1.0),
        )


HOOK_CONDITIONING_NODES = (
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    ConditioningSetDefaultCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningCombine,
    SetClipHooks,
    ConditioningTimestepsRange,
)
