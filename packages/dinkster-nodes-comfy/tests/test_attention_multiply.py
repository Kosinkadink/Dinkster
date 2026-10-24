"""CPU and hosted execution behavior for attention multiplier nodes."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from dinkster_api.v1 import CORE_FLOAT, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.attention_multiply import (
    CLIPAttentionMultiply,
    UNetCrossAttentionMultiply,
    UNetSelfAttentionMultiply,
    UNetTemporalAttentionMultiply,
)

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = dict(weights)

    def clone(self) -> Patcher:
        return Patcher(self.weights)

    def model_state_dict(self) -> dict[str, float]:
        return self.weights.copy()

    def add_patches(
        self, patches: Mapping[str, tuple[None]], strength_patch: float, strength_model: float
    ) -> None:
        assert strength_patch == 0.0
        for key in patches:
            self.weights[key] *= strength_model


class Clip:
    def __init__(self, patcher: Patcher) -> None:
        self.patcher = patcher

    def clone(self) -> Clip:
        return Clip(self.patcher.clone())

    def add_patches(
        self, patches: Mapping[str, tuple[None]], strength_patch: float, strength_model: float
    ) -> None:
        self.patcher.add_patches(patches, strength_patch, strength_model)


WEIGHTS = {
    "unet.attn1.to_q.weight": 1.0,
    "unet.attn1.to_k.bias": 2.0,
    "unet.attn1.to_v.weight": 3.0,
    "unet.attn1.to_out.0.bias": 4.0,
    "unet.attn2.to_q.weight": 5.0,
    "unet.attn2.to_k.bias": 6.0,
    "unet.attn2.to_v.weight": 7.0,
    "unet.attn2.to_out.0.bias": 8.0,
    "unet.attn10.to_q.weight": 9.0,
    "text.self_attn.q_proj.weight": 10.0,
    "text.self_attn.k_proj.bias": 11.0,
    "text.self_attn.v_proj.weight": 12.0,
    "text.self_attn.out_proj.bias": 13.0,
    "text.self_attn.q_proj.weight.distractor": 14.0,
    "unet.attn1.to_out.0.weight": 15.0,
    "unet.time_stack.attn1.to_out.0.weight": 16.0,
    "unet.attn2.to_out.0.weight": 17.0,
    "unet.time_stack.attn2.to_out.0.weight": 18.0,
    "unet.time_stack.attn1.to_q.weight": 19.0,
}


def test_attention_multipliers_assign_independent_strengths() -> None:
    source = Patcher(WEIGHTS)
    self_attention = cast(
        Patcher,
        UNetSelfAttentionMultiply.execute(model=source, q=1.1, k=1.2, v=1.3, out=1.4)["MODEL"],
    )
    cross_attention = cast(
        Patcher,
        UNetCrossAttentionMultiply.execute(model=source, q=2.1, k=2.2, v=2.3, out=2.4)["MODEL"],
    )
    clip_attention = cast(
        Clip,
        CLIPAttentionMultiply.execute(clip=Clip(source), q=3.1, k=3.2, v=3.3, out=3.4)["CLIP"],
    )

    assert self_attention.weights["unet.attn1.to_q.weight"] == pytest.approx(1.1)
    assert self_attention.weights["unet.attn1.to_k.bias"] == pytest.approx(2.4)
    assert self_attention.weights["unet.attn1.to_v.weight"] == pytest.approx(3.9)
    assert self_attention.weights["unet.attn1.to_out.0.bias"] == pytest.approx(5.6)
    assert self_attention.weights["unet.attn2.to_q.weight"] == 5.0
    assert self_attention.weights["unet.attn10.to_q.weight"] == 9.0
    assert cross_attention.weights["unet.attn2.to_q.weight"] == pytest.approx(10.5)
    assert cross_attention.weights["unet.attn2.to_k.bias"] == pytest.approx(13.2)
    assert cross_attention.weights["unet.attn2.to_v.weight"] == pytest.approx(16.1)
    assert cross_attention.weights["unet.attn2.to_out.0.bias"] == pytest.approx(19.2)
    assert clip_attention.patcher.weights["text.self_attn.q_proj.weight"] == pytest.approx(31.0)
    assert clip_attention.patcher.weights["text.self_attn.k_proj.bias"] == pytest.approx(35.2)
    assert clip_attention.patcher.weights["text.self_attn.v_proj.weight"] == pytest.approx(39.6)
    assert clip_attention.patcher.weights["text.self_attn.out_proj.bias"] == pytest.approx(44.2)
    assert clip_attention.patcher.weights["text.self_attn.q_proj.weight.distractor"] == 14.0
    assert source.weights == WEIGHTS


def test_temporal_attention_multiplies_only_output_projections_by_stack() -> None:
    temporal = cast(
        Patcher,
        UNetTemporalAttentionMultiply.execute(
            model=Patcher(WEIGHTS),
            self_structural=1.1,
            self_temporal=1.2,
            cross_structural=1.3,
            cross_temporal=1.4,
        )["MODEL"],
    )
    assert temporal.weights["unet.attn1.to_out.0.weight"] == pytest.approx(16.5)
    assert temporal.weights["unet.time_stack.attn1.to_out.0.weight"] == pytest.approx(19.2)
    assert temporal.weights["unet.attn2.to_out.0.weight"] == pytest.approx(22.1)
    assert temporal.weights["unet.time_stack.attn2.to_out.0.weight"] == pytest.approx(25.2)
    assert temporal.weights["unet.time_stack.attn1.to_q.weight"] == 19.0


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete("MODEL")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher({"unet.attn1.to_q.weight": 2.0}))


class WeightReader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.read",
            inputs=(InputSpec("model", TypeExpr.concrete("MODEL")),),
            outputs=(OutputSpec("weight", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        return cls.outputs(weight=model.weights["unet.attn1.to_q.weight"])


FIXTURE_NODES = (ModelSource, WeightReader)


@pytest.mark.parametrize("in_process", (True, False))
def test_complete_manifest_executes_attention_multiplier_in_each_worker_mode(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root)))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "attention_fixture_sources.py").write_text(
                "from test_attention_multiply import FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["attention-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "attention_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "attention_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("attention-fixture.source"),
                        "multiply": GraphNode(
                            "UNetSelfAttentionMultiply",
                            {
                                "model": Link("source", "MODEL"),
                                "q": 1.5,
                                "k": 2.5,
                                "v": 3.5,
                                "out": 4.5,
                            },
                        ),
                        "read": GraphNode(
                            "attention-fixture.read", {"model": Link("multiply", "MODEL")}
                        ),
                    }
                ),
                ["read"],
            )
            assert result.outputs["read"]["weight"].resolve() == 3.0
        finally:
            await composer.close()

    asyncio.run(scenario())
