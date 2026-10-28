"""Actual hook-bearing conditioning through native nodes and ordinary hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    HOOKS_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    TIMESTEPS_RANGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hook_conditioning import (
    ConditioningSetDefaultCombine,
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    ConditioningTimestepsRange,
    PairConditioningCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    SetClipHooks,
)

from dinkster.compose import PackSpec, ServingComposer


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    clip = CLIP(no_init=True)
    clip.cond_stage_model = torch.nn.Linear(2, 3)
    clip.patcher = ModelPatcher(
        clip.cond_stage_model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    clip.patcher.is_clip = True
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return clip


class HookConditioningSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("original", CONDITIONING_TYPE),
                    ("hooked", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        group = hooks.HookGroup()
        hook = hooks.WeightHook(strength_model=0.37, strength_clip=-0.8)
        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=1.7, start_percent=0.3))
        hook.hook_keyframe = keyframes
        group.add(hook)
        original = [[torch.tensor([[2.0, -7.0]], dtype=torch.float64), {"strength": 0.6}]]
        hooked = hooks.set_hooks_for_conditioning(original, group.clone())
        return cls.outputs(
            original=original,
            hooked=hooked,
            latent={"samples": torch.tensor([[[[3.0, -11.0]]]], dtype=torch.float64)},
        )


class HookConditioningInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("original", "hooked", "referenced")
            ),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Any, hooked: Any, referenced: Any) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        assert original[0][1] == {"strength": 0.6}
        assert hooked[0][1]["strength"] == referenced[0][1]["strength"] == 0.6
        assert hooked[0][1]["hooks"] is referenced[0][1]["hooks"]
        group = referenced[0][1]["hooks"]
        assert isinstance(group, hooks.HookGroup)
        assert len(group.hooks) == 1
        hook = group.hooks[0]
        assert isinstance(hook, hooks.WeightHook)
        assert hook.strength_model == 0.629
        assert hook.strength_clip == -1.36
        assert len(hook.hook_keyframe.keyframes) == 1
        assert hook.hook_keyframe.keyframes[-1].strength == 1.7
        assert hook.hook_keyframe.keyframes[-1].start_percent == 0.3
        assert "reference_latents" not in hooked[0][1]
        assert torch.equal(referenced[0][0], torch.tensor([[2.0, -7.0]], dtype=torch.float64))
        assert torch.equal(
            referenced[0][1]["reference_latents"][0],
            torch.tensor([[[[3.0, -11.0]]]], dtype=torch.float64),
        )
        return cls.outputs(strength=referenced[0][1]["strength"])


class HookConditioningPropertiesSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-properties.source",
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("existing", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                OutputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        group = hooks.HookGroup()
        hook = hooks.WeightHook(strength_model=0.25, strength_clip=-0.5)
        hook.need_weight_init = False
        hook.weights_clip = {}
        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=0.75, start_percent=0.4))
        hook.hook_keyframe = keyframes
        group.add(hook)
        return cls.outputs(
            positive=[[torch.tensor([[1.0, 2.0]], dtype=torch.float64), {"side": "positive"}]],
            negative=[[torch.tensor([[-3.0, 4.0]], dtype=torch.float64), {"side": "negative"}]],
            existing=[[torch.tensor([[5.0, -6.0]], dtype=torch.float64), {"source": "existing"}]],
            hooks=group,
            mask=torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float64),
            clip=_clip(),
        )


class HookConditioningPropertiesInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-properties.inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("single", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("combined", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("defaults", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("pair_combined", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("pair_properties_combined", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("pair_defaults", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("before", TypeExpr.concrete(TIMESTEPS_RANGE_TYPE)),
                InputSpec("after", TypeExpr.concrete(TIMESTEPS_RANGE_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("no_hooks", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("scheduled", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("unscheduled", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("source_hooks", TypeExpr.concrete(HOOKS_TYPE)),
            ),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        negative: Any,
        single: Any,
        combined: Any,
        defaults: Any,
        pair_combined: Any,
        pair_properties_combined: Any,
        pair_defaults: Any,
        before: tuple[float, float],
        after: tuple[float, float],
        clip: Any,
        no_hooks: Any,
        scheduled: Any,
        unscheduled: Any,
        source_hooks: Any,
    ) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        for result, side in ((positive, "positive"), (negative, "negative")):
            tensor, metadata = result[0]
            assert metadata["side"] == side
            assert metadata["start_percent"] == 0.2
            assert metadata["end_percent"] == 0.7
            assert metadata["set_area_to_bounds"] is True
            assert metadata["mask_strength"] == 1.25
            assert torch.equal(
                tensor,
                torch.tensor(
                    [[1.0, 2.0]] if side == "positive" else [[-3.0, 4.0]], dtype=torch.float64
                ),
            )
            assert torch.allclose(
                metadata["mask"],
                torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=metadata["mask"].dtype).unsqueeze(0),
            )
            assert isinstance(metadata["hooks"], hooks.HookGroup)
            assert metadata["hooks"].hooks[0].strength_model == 0.1875
            assert metadata["hooks"].hooks[0].hook_keyframe.keyframes[0].start_percent == 0.4
        assert [entry[1].get("source") for entry in combined] == ["existing", None]
        assert combined[1][1]["start_percent"] == 0.2
        assert combined[1][1]["end_percent"] == 0.7
        assert defaults[0][1] == {"source": "existing"}
        assert defaults[1][1]["default"] is True
        assert defaults[1][1]["hooks"].hooks[0].strength_clip == -0.375
        assert torch.equal(single[0][0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))
        assert "strength" not in single[0][1]
        assert single[0][1]["start_percent"] == 0.0
        assert single[0][1]["end_percent"] == 0.2
        assert [entry[1]["side"] for entry in pair_combined] == ["positive", "negative"]
        assert [entry[1]["side"] for entry in pair_properties_combined] == [
            "positive",
            "negative",
        ]
        assert [entry[1].get("default", False) for entry in pair_defaults] == [False, True]
        assert before == (0.0, 0.2)
        assert after == (0.7, 1.0)
        assert no_hooks is clip
        assert scheduled is not clip and unscheduled is not clip
        assert scheduled.cond_stage_model is clip.cond_stage_model
        assert scheduled.apply_hooks_to_conds is None
        assert scheduled.use_clip_schedule is True
        assert len(scheduled.patcher.forced_hooks.hooks[0].hook_keyframe.keyframes) == 1
        assert scheduled.patcher.forced_hooks.hooks[0].hook_keyframe.keyframes[0].strength == 0.75
        assert unscheduled.apply_hooks_to_conds is source_hooks
        assert unscheduled.use_clip_schedule is False
        assert not unscheduled.patcher.forced_hooks.hooks[0].hook_keyframe.keyframes
        assert len(scheduled.patcher.hook_patches) == len(unscheduled.patcher.hook_patches) == 1
        assert clip.patcher.forced_hooks is None
        assert clip.patcher.hook_patches == {}
        assert clip.apply_hooks_to_conds is None
        assert len(source_hooks.hooks[0].hook_keyframe.keyframes) == 1
        return cls.outputs(strength=positive[0][1]["mask_strength"])


def test_hook_conditioning_nodes_preserve_asymmetric_values() -> None:
    initialize_inference_process()
    from dinkster_inference import hooks

    group = hooks.HookGroup()
    group.add(hooks.WeightHook(strength_model=0.25, strength_clip=-0.5))
    positive = [[torch.tensor([[1.0]], dtype=torch.float64), {"source": "positive"}]]
    negative = [[torch.tensor([[-2.0]], dtype=torch.float64), {"source": "negative"}]]
    mask = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    ranges = ConditioningTimestepsRange.execute(start_percent=0.2, end_percent=0.7)
    properties = PairConditioningSetProperties.execute(
        positive_NEW=positive,
        negative_NEW=negative,
        strength=1.25,
        set_cond_area="mask bounds",
        mask=mask,
        hooks=group,
        timesteps=ranges["TIMESTEPS_RANGE"],
    )
    assert torch.equal(properties["positive"][0][1]["mask"], mask.unsqueeze(0))
    assert properties["positive"][0][1]["hooks"].hooks[0].strength_model == 0.25
    assert properties["negative"][0][1]["start_percent"] == 0.2
    assert torch.equal(positive[0][0], torch.tensor([[1.0]], dtype=torch.float64))
    assert positive[0][1] == {"source": "positive"}
    assert torch.equal(negative[0][0], torch.tensor([[-2.0]], dtype=torch.float64))
    assert negative[0][1] == {"source": "negative"}

    single = ConditioningSetProperties.execute(
        cond_NEW=positive,
        strength=0.9,
        set_cond_area="default",
        timesteps=ranges["BEFORE_RANGE"],
    )["conditioning"]
    assert torch.equal(single[0][0], positive[0][0])
    assert single[0][1] == {
        "source": "positive",
        "start_percent": 0.0,
        "end_percent": 0.2,
    }

    combined = ConditioningSetPropertiesAndCombine.execute(
        cond=positive,
        cond_NEW=negative,
        strength=0.4,
        set_cond_area="default",
        timesteps=ranges["AFTER_RANGE"],
    )["conditioning"]
    assert [entry[1]["source"] for entry in combined] == ["positive", "negative"]
    assert combined[1][1]["start_percent"] == 0.7
    defaults = ConditioningSetDefaultCombine.execute(
        cond=positive, cond_DEFAULT=negative, hooks=group
    )["conditioning"]
    assert defaults[1][1]["default"] is True
    assert (
        PairConditioningCombine.execute(
            positive_A=positive, negative_A=negative, positive_B=negative, negative_B=positive
        )["positive"][1][1]["source"]
        == "negative"
    )
    assert (
        PairConditioningSetPropertiesAndCombine.execute(
            positive=positive,
            negative=negative,
            positive_NEW=negative,
            negative_NEW=positive,
        )["negative"][1][1]["source"]
        == "positive"
    )
    assert (
        PairConditioningSetDefaultCombine.execute(
            positive=positive,
            negative=negative,
            positive_DEFAULT=negative,
            negative_DEFAULT=positive,
        )["positive"][1][1]["default"]
        is True
    )

    group.hooks[0].need_weight_init = False
    group.hooks[0].weights_clip = {}
    keyframes = hooks.HookKeyframeGroup()
    keyframes.add(hooks.HookKeyframe(strength=0.75, start_percent=0.4))
    group.hooks[0].hook_keyframe = keyframes
    clip = _clip()
    assert SetClipHooks.execute(clip=clip)["clip"] is clip
    applied = SetClipHooks.execute(
        clip=clip, apply_to_conds=True, schedule_clip=False, hooks=group
    )["clip"]
    assert applied is not clip
    assert applied.apply_hooks_to_conds is group
    assert len(applied.patcher.hook_patches) == 1
    assert not applied.patcher.forced_hooks.hooks[0].hook_keyframe.keyframes
    scheduled = SetClipHooks.execute(
        clip=clip, apply_to_conds=False, schedule_clip=True, hooks=group
    )["clip"]
    assert scheduled.apply_hooks_to_conds is None
    assert scheduled.use_clip_schedule is True
    assert scheduled.patcher.forced_hooks.hooks[0].hook_keyframe is keyframes
    assert len(keyframes.keyframes) == 1
    assert clip.patcher.forced_hooks is None
    assert clip.patcher.hook_patches == {}


@pytest.mark.parametrize("in_process", (True, False))
def test_hook_conditioning_properties_cross_ordinary_hosts_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src")))}
        )
        try:
            (tmp_path / "hook_properties_fixture.py").write_text(
                "from test_hook_conditioning_torch import (\n"
                "    HookConditioningPropertiesInspect,\n"
                "    HookConditioningPropertiesSource,\n"
                ")\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (HookConditioningPropertiesSource, HookConditioningPropertiesInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "hook-properties-dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "hook-properties-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["hook-properties"]\n'
                '[pack.entry]\nnodes = "hook_properties_fixture:NODES"\n'
                'types = "hook_properties_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = root / "dinkster-pack.toml"
            manifests = (native_manifest, fixture_manifest)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="hook-properties",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-properties.source"),
                    "range": GraphNode(
                        "ConditioningTimestepsRange", {"start_percent": 0.2, "end_percent": 0.7}
                    ),
                    "no_hooks": GraphNode("SetClipHooks", {"clip": Link("source", "clip")}),
                    "scheduled": GraphNode(
                        "SetClipHooks",
                        {
                            "clip": Link("source", "clip"),
                            "hooks": Link("source", "hooks"),
                            "apply_to_conds": False,
                            "schedule_clip": True,
                        },
                    ),
                    "unscheduled": GraphNode(
                        "SetClipHooks",
                        {
                            "clip": Link("source", "clip"),
                            "hooks": Link("source", "hooks"),
                            "apply_to_conds": True,
                            "schedule_clip": False,
                        },
                    ),
                    "properties": GraphNode(
                        "PairConditioningSetProperties",
                        {
                            "positive_NEW": Link("source", "positive"),
                            "negative_NEW": Link("source", "negative"),
                            "strength": 1.25,
                            "set_cond_area": "mask bounds",
                            "mask": Link("source", "mask"),
                            "hooks": Link("source", "hooks"),
                            "timesteps": Link("range", "TIMESTEPS_RANGE"),
                        },
                    ),
                    "single": GraphNode(
                        "ConditioningSetProperties",
                        {
                            "cond_NEW": Link("source", "positive"),
                            "strength": 0.9,
                            "set_cond_area": "default",
                            "timesteps": Link("range", "BEFORE_RANGE"),
                        },
                    ),
                    "combined": GraphNode(
                        "ConditioningSetPropertiesAndCombine",
                        {
                            "cond": Link("source", "existing"),
                            "cond_NEW": Link("source", "positive"),
                            "strength": 1.0,
                            "set_cond_area": "default",
                            "timesteps": Link("range", "TIMESTEPS_RANGE"),
                        },
                    ),
                    "defaults": GraphNode(
                        "ConditioningSetDefaultCombine",
                        {
                            "cond": Link("source", "existing"),
                            "cond_DEFAULT": Link("source", "negative"),
                            "hooks": Link("source", "hooks"),
                        },
                    ),
                    "pair_combined": GraphNode(
                        "PairConditioningCombine",
                        {
                            "positive_A": Link("source", "positive"),
                            "negative_A": Link("source", "negative"),
                            "positive_B": Link("source", "negative"),
                            "negative_B": Link("source", "positive"),
                        },
                    ),
                    "pair_properties_combined": GraphNode(
                        "PairConditioningSetPropertiesAndCombine",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "positive_NEW": Link("source", "negative"),
                            "negative_NEW": Link("source", "positive"),
                            "strength": 0.5,
                            "set_cond_area": "default",
                        },
                    ),
                    "pair_defaults": GraphNode(
                        "PairConditioningSetDefaultCombine",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "positive_DEFAULT": Link("source", "negative"),
                            "negative_DEFAULT": Link("source", "positive"),
                            "hooks": Link("source", "hooks"),
                        },
                    ),
                    "inspect": GraphNode(
                        "hook-properties.inspect",
                        {
                            "positive": Link("properties", "positive"),
                            "negative": Link("properties", "negative"),
                            "single": Link("single", "conditioning"),
                            "combined": Link("combined", "conditioning"),
                            "defaults": Link("defaults", "conditioning"),
                            "pair_combined": Link("pair_combined", "positive"),
                            "pair_properties_combined": Link(
                                "pair_properties_combined", "positive"
                            ),
                            "pair_defaults": Link("pair_defaults", "positive"),
                            "before": Link("range", "BEFORE_RANGE"),
                            "after": Link("range", "AFTER_RANGE"),
                            "clip": Link("source", "clip"),
                            "no_hooks": Link("no_hooks", "clip"),
                            "scheduled": Link("scheduled", "clip"),
                            "unscheduled": Link("unscheduled", "clip"),
                            "source_hooks": Link("source", "hooks"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 1.25
                if attempt:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", (True,))
def test_hook_conditioning_reaches_native_reference_consumer(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src")))}
        )
        try:
            (tmp_path / "hook_fixture.py").write_text(
                "from test_hook_conditioning_torch import HookConditioningSource\n"
                "from test_hook_conditioning_torch import HookConditioningInspect\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (HookConditioningSource, HookConditioningInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "hook-fixture-dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "hook-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["hook-fixture"]\n'
                '[pack.entry]\nnodes = "hook_fixture:NODES"\n'
                'types = "hook_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            manifests = (root / "dinkster-pack.toml", fixture_manifest)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="hook-reference",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-fixture.source"),
                    "reference": GraphNode(
                        "ReferenceLatent",
                        {
                            "conditioning": Link("source", "hooked"),
                            "latent": Link("source", "latent"),
                        },
                    ),
                    "inspect": GraphNode(
                        "hook-fixture.inspect",
                        {
                            "original": Link("source", "original"),
                            "hooked": Link("source", "hooked"),
                            "referenced": Link("reference", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 0.6
                if attempt:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
