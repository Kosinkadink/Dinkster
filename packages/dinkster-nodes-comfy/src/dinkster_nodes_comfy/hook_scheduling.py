"""Native scheduling and composition for inference hook groups."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_FLOATS = TypeExpr.concrete(FLOATS_TYPE)
_HOOK_KEYFRAMES = TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
_HOOKS = TypeExpr.concrete(HOOKS_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_CATEGORY = "advanced/hooks/scheduling"


def _hooks() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.hooks")


def retain_hook_weights(result: Any, *sources: Any) -> Any:
    if result is None or any(result is source for source in sources):
        return result
    retained: list[object] = []
    for source in (result, *sources):
        if source is None or not any(
            hook == cloned
            and all(
                getattr(hook, field, None) is getattr(cloned, field, None)
                for field in ("weights", "weights_clip")
            )
            for hook in source.hooks
            for cloned in result.hooks
        ):
            continue
        derived = getattr(source, "_dinkster_hook_keyframe_refs", ())
        for owner in getattr(source, "_dinkster_resident_refs", ()):
            # Only explicit owners survive cloning; normalization derives the new schedule.
            if not any(owner is old for old in (*derived, *retained)):
                retained.append(owner)
    result._dinkster_resident_refs = tuple(retained)
    return result


def _keyframes(previous: Any | None) -> Any:
    hooks = _hooks()
    return hooks.HookKeyframeGroup() if previous is None else previous.clone()


def _append_keyframes(
    previous: Any | None,
    strengths: Any,
    start_percent: float,
    end_percent: float,
    *,
    print_keyframes: bool,
) -> Any:
    hooks = _hooks()
    result = _keyframes(previous)
    if type(strengths) in (int, float):
        values: Any = (float(strengths),)
    else:
        values = materialize_latent(strengths)
    try:
        count = len(values)
    except TypeError as exc:
        raise TypeError("floats_strength must be a scalar or sized sequence") from exc
    percents = hooks.InterpolationMethod.get_weights(
        num_from=start_percent,
        num_to=end_percent,
        length=count,
        method=hooks.InterpolationMethod.LINEAR,
    )
    for index, (percent, strength) in enumerate(zip(percents, values, strict=True)):
        result.add(
            hooks.HookKeyframe(
                strength=strength,
                start_percent=percent,
                guarantee_steps=1 if index == 0 else 0,
            )
        )
        if print_keyframes:
            logging.info("Hook Keyframe - start_percent:%s = %s", percent, strength)
    return result


class SetHookKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SetHookKeyframes",
            display_name="Set Hook Keyframes",
            category=_CATEGORY,
            inputs=(
                InputSpec("hooks", _HOOKS),
                InputSpec("hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            outputs=(OutputSpec("HOOKS", _HOOKS),),
        )

    @classmethod
    def execute(cls, *, hooks: Any, hook_kf: Any | None = None) -> Mapping[str, object]:
        if hook_kf is None:
            return cls.outputs(HOOKS=hooks)
        result = hooks.clone()
        result.set_keyframes_on_hooks(hook_kf)
        return cls.outputs(HOOKS=retain_hook_weights(result, hooks))


class CreateHookKeyframe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CreateHookKeyframe",
            display_name="Create Hook Keyframe",
            category=_CATEGORY,
            search_terms=("hook scheduling", "strength animation", "timed hook"),
            inputs=(
                InputSpec(
                    "strength_mult",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-20.0, max=20.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("prev_hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            outputs=(OutputSpec("HOOK_KF", _HOOK_KEYFRAMES),),
        )

    @classmethod
    def execute(
        cls, *, strength_mult: float, start_percent: float, prev_hook_kf: Any | None = None
    ) -> Mapping[str, object]:
        result = _keyframes(prev_hook_kf)
        result.add(_hooks().HookKeyframe(strength=strength_mult, start_percent=start_percent))
        return cls.outputs(HOOK_KF=result)


class CreateHookKeyframesInterpolated(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        interpolation = ("linear", "ease_in", "ease_out", "ease_in_out")
        return NodeSchema(
            node_type="CreateHookKeyframesInterpolated",
            display_name="Create Hook Keyframes Interp.",
            category=_CATEGORY,
            search_terms=("ease hook strength", "smooth hook transition", "interpolate keyframes"),
            inputs=(
                InputSpec(
                    "strength_start",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.001),
                ),
                InputSpec(
                    "strength_end",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.001),
                ),
                InputSpec(
                    "interpolation",
                    _COMBO,
                    widget=ComboWidget(tuple(ComboOption(value) for value in interpolation)),
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("keyframes_count", _INT, default=5, widget=NumberWidget(min=2, max=100)),
                InputSpec("print_keyframes", _BOOLEAN, default=False, advanced=True),
                InputSpec("prev_hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            outputs=(OutputSpec("HOOK_KF", _HOOK_KEYFRAMES),),
        )

    @classmethod
    def execute(
        cls,
        *,
        strength_start: float,
        strength_end: float,
        interpolation: str,
        start_percent: float,
        end_percent: float,
        keyframes_count: int,
        print_keyframes: bool = False,
        prev_hook_kf: Any | None = None,
    ) -> Mapping[str, object]:
        hooks = _hooks()
        result = _keyframes(prev_hook_kf)
        percents = hooks.InterpolationMethod.get_weights(
            start_percent, end_percent, keyframes_count, hooks.InterpolationMethod.LINEAR
        )
        strengths = hooks.InterpolationMethod.get_weights(
            strength_start, strength_end, keyframes_count, interpolation
        )
        for index, (percent, strength) in enumerate(zip(percents, strengths, strict=True)):
            result.add(
                hooks.HookKeyframe(
                    strength=strength,
                    start_percent=percent,
                    guarantee_steps=1 if index == 0 else 0,
                )
            )
            if print_keyframes:
                logging.info("Hook Keyframe - start_percent:%s = %s", percent, strength)
        return cls.outputs(HOOK_KF=result)


class CreateHookKeyframesFromFloats(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CreateHookKeyframesFromFloats",
            display_name="Create Hook Keyframes From Floats",
            category=_CATEGORY,
            search_terms=("batch keyframes", "strength list to keyframes"),
            inputs=(
                InputSpec("floats_strength", _FLOATS, default=-1, force_input=True),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("print_keyframes", _BOOLEAN, default=False, advanced=True),
                InputSpec("prev_hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            outputs=(OutputSpec("HOOK_KF", _HOOK_KEYFRAMES),),
        )

    @classmethod
    def execute(
        cls,
        *,
        floats_strength: object,
        start_percent: float,
        end_percent: float,
        print_keyframes: bool = False,
        prev_hook_kf: Any | None = None,
    ) -> Mapping[str, object]:
        result = _append_keyframes(
            prev_hook_kf,
            floats_strength,
            start_percent,
            end_percent,
            print_keyframes=print_keyframes,
        )
        return cls.outputs(HOOK_KF=result)


def _combine_schema(node_type: str, count: int) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=f"Combine Hooks [{count}]",
        category="advanced/hooks/combine",
        search_terms=("merge hooks",) if count == 2 else (),
        inputs=tuple(
            InputSpec(f"hooks_{letter}", _HOOKS, required=False) for letter in "ABCDEFGH"[:count]
        ),
        outputs=(OutputSpec("HOOKS", _HOOKS),),
    )


class CombineHooks2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _combine_schema("CombineHooks2", 2)

    @classmethod
    def execute(
        cls, *, hooks_A: Any | None = None, hooks_B: Any | None = None
    ) -> Mapping[str, object]:
        candidates = [hooks_A, hooks_B]
        result = _hooks().HookGroup.combine_all_hooks(candidates)
        return cls.outputs(HOOKS=retain_hook_weights(result, *candidates))


class CombineHooks4(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _combine_schema("CombineHooks4", 4)

    @classmethod
    def execute(cls, **inputs: Any) -> Mapping[str, object]:
        candidates = [inputs.get(f"hooks_{letter}") for letter in "ABCD"]
        result = _hooks().HookGroup.combine_all_hooks(candidates)
        return cls.outputs(HOOKS=retain_hook_weights(result, *candidates))


class CombineHooks8(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _combine_schema("CombineHooks8", 8)

    @classmethod
    def execute(cls, **inputs: Any) -> Mapping[str, object]:
        candidates = [inputs.get(f"hooks_{letter}") for letter in "ABCDEFGH"]
        result = _hooks().HookGroup.combine_all_hooks(candidates)
        return cls.outputs(HOOKS=retain_hook_weights(result, *candidates))


HOOK_SCHEDULING_NODES = (
    SetHookKeyframes,
    CreateHookKeyframe,
    CreateHookKeyframesInterpolated,
    CreateHookKeyframesFromFloats,
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
)
