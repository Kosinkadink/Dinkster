"""Public schemas for native hook scheduling nodes."""

from dinkster_api.v1 import FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, NumberWidget, TypeExpr
from dinkster_models import NODES
from dinkster_models.hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    SetHookKeyframes,
)


def test_hook_scheduling_nodes_publish_the_fixed_core_ids_and_sockets() -> None:
    expected = {
        "dinkster.models.set_hook_keyframes": ("hooks", "hook_kf"),
        "dinkster.models.create_hook_keyframe": ("strength_mult", "start_percent", "prev_hook_kf"),
        "dinkster.models.create_hook_keyframes_interpolated": (
            "strength_start",
            "strength_end",
            "interpolation",
            "start_percent",
            "end_percent",
            "keyframes_count",
            "print_keyframes",
            "prev_hook_kf",
        ),
        "dinkster.models.create_hook_keyframes_from_floats": (
            "floats_strength",
            "start_percent",
            "end_percent",
            "print_keyframes",
            "prev_hook_kf",
        ),
        "dinkster.models.combine_hooks2": ("hooks_A", "hooks_B"),
        "dinkster.models.combine_hooks4": ("hooks_A", "hooks_B", "hooks_C", "hooks_D"),
        "dinkster.models.combine_hooks8": (
            "hooks_A",
            "hooks_B",
            "hooks_C",
            "hooks_D",
            "hooks_E",
            "hooks_F",
            "hooks_G",
            "hooks_H",
        ),
    }
    nodes = (
        SetHookKeyframes,
        CreateHookKeyframe,
        CreateHookKeyframesInterpolated,
        CreateHookKeyframesFromFloats,
        CombineHooks2,
        CombineHooks4,
        CombineHooks8,
    )
    assert tuple(node.schema().node_type for node in nodes) == tuple(expected)
    for node in nodes:
        schema = node.schema()
        assert node in NODES
        assert tuple(input.id for input in schema.inputs) == expected[schema.node_type]
        assert all(
            input.type == TypeExpr.concrete(HOOKS_TYPE)
            for input in schema.inputs
            if input.id.startswith("hooks_")
        )

    set_keyframes, create_one, interpolated, from_floats, *_ = (node.schema() for node in nodes)
    assert set_keyframes.inputs[0].type == TypeExpr.concrete(HOOKS_TYPE)
    assert set_keyframes.inputs[1].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert create_one.outputs[0].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert interpolated.outputs[0].id == from_floats.outputs[0].id == "HOOK_KF"
    assert from_floats.inputs[0].type == TypeExpr.concrete(FLOATS_TYPE)
    assert from_floats.inputs[0].force_input
    assert from_floats.inputs[0].default == -1
    assert from_floats.inputs[0].widget == NumberWidget(min=-1, step=0.001)
    assert interpolated.inputs[5].widget == NumberWidget(min=2, max=100, step=1)
