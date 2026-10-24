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
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link

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
