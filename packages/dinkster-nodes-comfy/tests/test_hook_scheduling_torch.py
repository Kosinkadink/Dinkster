"""CPU behavior for native hook scheduling and combination nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
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
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    SetHookKeyframes,
)
from dinkster_values import EncodedLatentTensor

from dinkster.compose import PackSpec, ServingComposer


@pytest.fixture(autouse=True)
def _cpu_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    initialize_inference_process()


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


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifests_execute_hook_scheduling_chain_twice(
    tmp_path: Any, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).parents[1]
        composer = ServingComposer(
            worker_env={
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(root / "src"))
                ),
            }
        )
        try:
            module = f"hook_scheduling_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from dinkster_api.v1 import register_inference_types\n"
                "from test_hook_scheduling_torch import HookInspect, HookSource\n"
                "NODES = (HookSource, HookInspect)\n",
                encoding="ascii",
            )
            manifest = root / "dinkster-pack.toml"
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                "[pack]\n"
                'name = "hook-scheduling-fixture"\n'
                'publisher = "dinkster"\n'
                'namespaces = ["hook-scheduling-fixture"]\n'
                "\n[pack.entry]\n"
                f'nodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            for pack_manifest in (manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=pack_manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "hook-scheduling-fixture",
                        group_manifests=() if in_process else (manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-scheduling-fixture.source"),
                    "single": GraphNode(
                        "CreateHookKeyframe", {"strength_mult": 0.25, "start_percent": 0.1}
                    ),
                    "interpolated": GraphNode(
                        "CreateHookKeyframesInterpolated",
                        {
                            "strength_start": 0.25,
                            "strength_end": 1.25,
                            "interpolation": "linear",
                            "start_percent": 0.1,
                            "end_percent": 0.5,
                            "keyframes_count": 2,
                            "prev_hook_kf": Link("single", "HOOK_KF"),
                        },
                    ),
                    "from-floats": GraphNode(
                        "CreateHookKeyframesFromFloats",
                        {
                            "floats_strength": Link("source", "strengths"),
                            "start_percent": 0.6,
                            "end_percent": 0.8,
                            "prev_hook_kf": Link("interpolated", "HOOK_KF"),
                        },
                    ),
                    "set": GraphNode(
                        "SetHookKeyframes",
                        {
                            "hooks": Link("source", "hooks"),
                            "hook_kf": Link("from-floats", "HOOK_KF"),
                        },
                    ),
                    "combined": GraphNode(
                        "CombineHooks2",
                        {"hooks_A": Link("set", "HOOKS"), "hooks_B": Link("source", "hooks")},
                    ),
                    "combined-four": GraphNode(
                        "CombineHooks4", {"hooks_D": Link("combined", "HOOKS")}
                    ),
                    "combined-eight": GraphNode(
                        "CombineHooks8", {"hooks_H": Link("combined-four", "HOOKS")}
                    ),
                    "inspect": GraphNode(
                        "hook-scheduling-fixture.inspect",
                        {"hooks": Link("combined-eight", "HOOKS")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for iteration in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 0.1875
                assert set(result.executed if iteration == 0 else result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
