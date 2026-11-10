"""CPU behavior for native hook scheduling and combination nodes."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    FLOATS_TYPE,
    HOOKS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_models.hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    SetHookKeyframes,
)
from dinkster_values import EncodedLatentTensor


def _hooks() -> Any:
    from dinkster_inference import hooks

    return hooks


def _group(strength: float) -> Any:
    hooks = _hooks()
    group = hooks.HookGroup()
    group.add(hooks.WeightHook(strength_model=strength))
    return group


def test_create_keyframes_preserves_source_and_interpolation_endpoints() -> None:
    initial = CreateHookKeyframe.execute(strength_mult=-1.25, start_percent=0.75)["HOOK_KF"]
    result = CreateHookKeyframesInterpolated.execute(
        strength_start=0.25,
        strength_end=3.25,
        interpolation="ease_in",
        start_percent=0.1,
        end_percent=0.7,
        keyframes_count=3,
        prev_hook_kf=initial,
    )["HOOK_KF"]

    assert [(keyframe.strength, keyframe.start_percent) for keyframe in initial.keyframes] == [
        (-1.25, 0.75)
    ]
    assert result is not initial
    assert [float(keyframe.strength) for keyframe in result.keyframes] == [0.25, 1.0, 3.25, -1.25]
    torch.testing.assert_close(
        torch.tensor([keyframe.start_percent for keyframe in result.keyframes]),
        torch.tensor([0.1, 0.4, 0.7, 0.75]),
    )
    assert [keyframe.guarantee_steps for keyframe in result.keyframes] == [1, 0, 0, 1]


def test_float_strengths_materialize_encoded_tensor_without_losing_dtype_or_order() -> None:
    strengths = torch.tensor([0.5, -3.25, 1.75], dtype=torch.bfloat16)
    encoded = EncodedLatentTensor(
        "bfloat16",
        tuple(strengths.shape),
        strengths.contiguous().view(torch.uint8).numpy().tobytes(),
    )
    result = CreateHookKeyframesFromFloats.execute(
        floats_strength=encoded,
        start_percent=0.2,
        end_percent=0.8,
    )["HOOK_KF"]

    values = [keyframe.strength for keyframe in result.keyframes]
    assert [value.dtype for value in values] == [torch.bfloat16] * 3
    assert [float(value) for value in values] == [0.5, -3.25, 1.75]
    assert [keyframe.start_percent for keyframe in result.keyframes] == [
        0.20000000298023224,
        0.5,
        0.800000011920929,
    ]
    assert [keyframe.guarantee_steps for keyframe in result.keyframes] == [1, 0, 0]


def test_set_keyframes_clones_hooks_and_tracks_retained_keyframe_identity() -> None:
    source = _group(0.75)
    keyframes = CreateHookKeyframe.execute(strength_mult=0.625, start_percent=0.25)["HOOK_KF"]
    result = SetHookKeyframes.execute(hooks=source, hook_kf=keyframes)["HOOKS"]

    assert result is not source
    assert source.hooks[0].hook_keyframe is not keyframes
    retained = result.hooks[0].hook_keyframe
    assert retained is not keyframes
    assert [(keyframe.strength, keyframe.start_percent) for keyframe in retained.keyframes] == [
        (0.625, 0.25)
    ]


@pytest.mark.parametrize(
    ("node", "count"),
    ((CombineHooks2, 2), (CombineHooks4, 4), (CombineHooks8, 8)),
)
def test_combine_hooks_preserves_singleton_and_clones_multi_group_inputs(
    node: Any, count: int
) -> None:
    groups = [_group(float(index + 1)) for index in range(count)]
    inputs = {
        f"hooks_{letter}": group for letter, group in zip("ABCDEFGH"[:count], groups, strict=True)
    }
    combined = node.execute(**inputs)["HOOKS"]

    assert len(combined.hooks) == count
    assert [hook.strength_model for hook in combined.hooks] == [
        float(index + 1) for index in range(count)
    ]
    assert all(
        result is not source.hooks[0] for result, source in zip(combined.hooks, groups, strict=True)
    )
    assert CombineHooks2.execute(hooks_A=groups[0], hooks_B=None)["HOOKS"] is groups[0]
    assert CombineHooks2.execute(hooks_A=None, hooks_B=None)["HOOKS"] is None


class HookSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-scheduling-fixture.source",
            outputs=(
                OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                OutputSpec("strengths", TypeExpr.concrete(FLOATS_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(hooks=_group(0.75), strengths=torch.tensor([0.5, -1.25]))


class HookInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-scheduling-fixture.inspect",
            inputs=(InputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, hooks: Any) -> dict[str, object]:
        assert len(hooks.hooks) == 1
        keyframes = hooks.hooks[0].hook_keyframe.keyframes
        assert [(float(keyframe.strength), keyframe.guarantee_steps) for keyframe in keyframes] == [
            (0.25, 1),
            (0.25, 1),
            (1.25, 0),
            (0.5, 1),
            (-1.25, 0),
        ]
        return cls.outputs(strength=hooks.hooks[0].strength_model)
