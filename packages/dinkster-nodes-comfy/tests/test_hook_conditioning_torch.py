"""Actual hook-bearing conditioning through native nodes and ordinary hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    HOOKS_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
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
    ConditioningSetPropertiesAndCombine,
    ConditioningTimestepsRange,
    PairConditioningCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    SetClipHooks,
)

from dinkster.compose import PackSpec, ServingComposer


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
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        group = hooks.HookGroup()
        hook = hooks.WeightHook(strength_model=0.25, strength_clip=-0.5)
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
        )


class HookConditioningPropertiesInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-properties.inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("combined", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("defaults", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls, *, positive: Any, negative: Any, combined: Any, defaults: Any
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

    class Patcher:
        def __init__(self) -> None:
            self.forced_hooks: Any = None
            self.calls: list[tuple[object, object]] = []

        def register_all_hook_patches(self, value: object, target: object) -> None:
            self.calls.append((value, target))

    class Clip:
        def __init__(self) -> None:
            self.patcher = Patcher()

        def clone(self, *, disable_dynamic: bool) -> Clip:
            assert disable_dynamic is True
            return Clip()

    clip = Clip()
    assert SetClipHooks.execute(clip=clip)["clip"] is clip
    applied = SetClipHooks.execute(
        clip=clip, apply_to_conds=True, schedule_clip=False, hooks=group
    )["clip"]
    assert applied is not clip
    assert applied.apply_hooks_to_conds is group
    assert len(applied.patcher.calls) == 1


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
                "    HookConditioningPropertiesInspect, HookConditioningPropertiesSource,\n"
                ")\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, HookConditioningPropertiesSource,\n"
                "         HookConditioningPropertiesInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hook-properties",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hook_properties_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hook_properties_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-properties.source"),
                    "range": GraphNode(
                        "ConditioningTimestepsRange", {"start_percent": 0.2, "end_percent": 0.7}
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
                    "inspect": GraphNode(
                        "hook-properties.inspect",
                        {
                            "positive": Link("properties", "positive"),
                            "negative": Link("properties", "negative"),
                            "combined": Link("combined", "conditioning"),
                            "defaults": Link("defaults", "conditioning"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 1.25
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", (True, False))
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
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, HookConditioningSource, HookConditioningInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hook-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hook_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hook_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
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
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 0.6
        finally:
            await composer.close()

    asyncio.run(scenario())
