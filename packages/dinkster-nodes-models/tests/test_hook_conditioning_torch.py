"""Actual hook-bearing conditioning through native nodes and ordinary hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_models.hook_conditioning import (
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
                        "dinkster.conditioning.reference_latent",
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
