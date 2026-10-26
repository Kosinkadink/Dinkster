"""Public schemas for native hook scheduling nodes."""

from dinkster_api.v1 import FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hook_scheduling import HOOK_SCHEDULING_NODES
from dinkster_workers import load_manifest


def test_hook_scheduling_nodes_publish_the_fixed_core_ids_and_sockets() -> None:
    expected = {
        "SetHookKeyframes": ("hooks", "hook_kf"),
        "CreateHookKeyframe": ("strength_mult", "start_percent", "prev_hook_kf"),
        "CreateHookKeyframesInterpolated": (
            "strength_start",
            "strength_end",
            "interpolation",
            "start_percent",
            "end_percent",
            "keyframes_count",
            "print_keyframes",
            "prev_hook_kf",
        ),
        "CreateHookKeyframesFromFloats": (
            "floats_strength",
            "start_percent",
            "end_percent",
            "print_keyframes",
            "prev_hook_kf",
        ),
        "CombineHooks2": ("hooks_A", "hooks_B"),
        "CombineHooks4": ("hooks_A", "hooks_B", "hooks_C", "hooks_D"),
        "CombineHooks8": (
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
    for node in HOOK_SCHEDULING_NODES:
        schema = node.schema()
        assert node in COMFY_NODES
        assert tuple(input.id for input in schema.inputs) == expected[schema.node_type]
        assert all(
            input.type == TypeExpr.concrete(HOOKS_TYPE)
            for input in schema.inputs
            if input.id.startswith("hooks_")
        )

    set_keyframes, create_one, interpolated, from_floats, *_ = (
        node.schema() for node in HOOK_SCHEDULING_NODES
    )
    assert set_keyframes.inputs[0].type == TypeExpr.concrete(HOOKS_TYPE)
    assert set_keyframes.inputs[1].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert create_one.outputs[0].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert interpolated.outputs[0].id == from_floats.outputs[0].id == "HOOK_KF"
    assert from_floats.inputs[0].type == TypeExpr.concrete(FLOATS_TYPE)
    assert from_floats.inputs[0].force_input
    assert from_floats.inputs[0].default == -1
    assert from_floats.inputs[0].widget == NumberWidget(min=-1, step=0.001)
    assert interpolated.inputs[5].widget == NumberWidget(min=2, max=100, step=1)


def test_complete_manifest_declares_hook_scheduling_namespaces() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert {node.schema().node_type for node in HOOK_SCHEDULING_NODES}.issubset(manifest.namespaces)
