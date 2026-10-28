"""CPU execution and hosted transport coverage for attention multiplier nodes."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_FLOAT,
    CORE_STRING,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.attention_multiply import (
    CLIPAttentionMultiply,
    UNetCrossAttentionMultiply,
    UNetSelfAttentionMultiply,
    UNetTemporalAttentionMultiply,
)

from dinkster.compose import PackSpec, ServingComposer

ATTENTION_STRENGTHS = {"q": 1.11, "k": 1.23, "v": 1.37, "out": 1.41}
TEMPORAL_STRENGTHS = {
    "self_structural": 1.13,
    "self_temporal": 1.29,
    "cross_structural": 1.47,
    "cross_temporal": 1.61,
}


def _tensor(value: float) -> torch.Tensor:
    return torch.tensor([value, value + 0.25], dtype=torch.float32)


def _weights() -> dict[str, torch.Tensor]:
    names = (
        "unet.attn1.to_q.weight",
        "unet.attn1.to_q.bias",
        "unet.attn1.to_k.weight",
        "unet.attn1.to_k.bias",
        "unet.attn1.to_v.weight",
        "unet.attn1.to_v.bias",
        "unet.attn1.to_out.0.weight",
        "unet.attn1.to_out.0.bias",
        "unet.attn2.to_q.weight",
        "unet.attn2.to_q.bias",
        "unet.attn2.to_k.weight",
        "unet.attn2.to_k.bias",
        "unet.attn2.to_v.weight",
        "unet.attn2.to_v.bias",
        "unet.attn2.to_out.0.weight",
        "unet.attn2.to_out.0.bias",
        "unet.time_stack.attn1.to_out.0.weight",
        "unet.time_stack.attn1.to_out.0.bias",
        "unet.time_stack.attn2.to_out.0.weight",
        "unet.time_stack.attn2.to_out.0.bias",
        "unet.time_stack.attn1.to_q.weight",
        "unet.attn10.to_q.weight",
        "unet.attn1.to_q.weight.distractor",
        "text.self_attn.q_proj.weight",
        "text.self_attn.q_proj.bias",
        "text.self_attn.k_proj.weight",
        "text.self_attn.k_proj.bias",
        "text.self_attn.v_proj.weight",
        "text.self_attn.v_proj.bias",
        "text.self_attn.out_proj.weight",
        "text.self_attn.out_proj.bias",
        "text.self_attn.q_proj.weight.distractor",
    )
    return {name: _tensor(index + 1.0) for index, name in enumerate(names)}


class Patcher:
    def __init__(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.weights = dict(weights)

    def clone(self) -> Patcher:
        return Patcher(self.weights)

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return self.weights.copy()

    def add_patches(
        self,
        patches: Mapping[str, tuple[None]],
        strength_patch: float,
        strength_model: float,
    ) -> None:
        assert strength_patch == 0.0
        for key in patches:
            self.weights[key] = self.weights[key] * strength_model


class Clip:
    def __init__(self, patcher: Patcher) -> None:
        self.patcher = patcher

    def clone(self) -> Clip:
        return Clip(self.patcher.clone())

    def add_patches(
        self,
        patches: Mapping[str, tuple[None]],
        strength_patch: float,
        strength_model: float,
    ) -> None:
        self.patcher.add_patches(patches, strength_patch, strength_model)


def _assert_preserves_source(
    source: Patcher, original: Mapping[str, torch.Tensor], transformed: Patcher
) -> None:
    assert transformed is not source
    for key, value in original.items():
        assert torch.equal(source.weights[key], value), key


def _assert_scaled(
    transformed: Patcher,
    original: Mapping[str, torch.Tensor],
    strengths: Mapping[str, float],
) -> None:
    for key, value in original.items():
        strength = next(
            (factor for suffix, factor in strengths.items() if key.endswith(suffix)), None
        )
        expected = value if strength is None else value * strength
        assert torch.equal(transformed.weights[key], expected), key


def _unet_strengths(attention: str) -> dict[str, float]:
    return {
        f"{attention}.to_q.weight": ATTENTION_STRENGTHS["q"],
        f"{attention}.to_q.bias": ATTENTION_STRENGTHS["q"],
        f"{attention}.to_k.weight": ATTENTION_STRENGTHS["k"],
        f"{attention}.to_k.bias": ATTENTION_STRENGTHS["k"],
        f"{attention}.to_v.weight": ATTENTION_STRENGTHS["v"],
        f"{attention}.to_v.bias": ATTENTION_STRENGTHS["v"],
        f"{attention}.to_out.0.weight": ATTENTION_STRENGTHS["out"],
        f"{attention}.to_out.0.bias": ATTENTION_STRENGTHS["out"],
    }


def _clip_strengths() -> dict[str, float]:
    return {
        "self_attn.q_proj.weight": ATTENTION_STRENGTHS["q"],
        "self_attn.q_proj.bias": ATTENTION_STRENGTHS["q"],
        "self_attn.k_proj.weight": ATTENTION_STRENGTHS["k"],
        "self_attn.k_proj.bias": ATTENTION_STRENGTHS["k"],
        "self_attn.v_proj.weight": ATTENTION_STRENGTHS["v"],
        "self_attn.v_proj.bias": ATTENTION_STRENGTHS["v"],
        "self_attn.out_proj.weight": ATTENTION_STRENGTHS["out"],
        "self_attn.out_proj.bias": ATTENTION_STRENGTHS["out"],
    }


def _temporal_strengths(original: Mapping[str, torch.Tensor]) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for key in original:
        if key.endswith(("attn1.to_out.0.weight", "attn1.to_out.0.bias")):
            strengths[key] = (
                TEMPORAL_STRENGTHS["self_temporal"]
                if ".time_stack." in key
                else TEMPORAL_STRENGTHS["self_structural"]
            )
        elif key.endswith(("attn2.to_out.0.weight", "attn2.to_out.0.bias")):
            strengths[key] = (
                TEMPORAL_STRENGTHS["cross_temporal"]
                if ".time_stack." in key
                else TEMPORAL_STRENGTHS["cross_structural"]
            )
    return strengths


def _verify_model(
    kind: str,
    original: Patcher,
    transformed: Patcher,
    expected_weights: Mapping[str, torch.Tensor],
) -> None:
    _assert_preserves_source(original, expected_weights, transformed)
    if kind == "self":
        _assert_scaled(transformed, expected_weights, _unet_strengths("attn1"))
    elif kind == "cross":
        _assert_scaled(transformed, expected_weights, _unet_strengths("attn2"))
    elif kind == "temporal":
        _assert_scaled(transformed, expected_weights, _temporal_strengths(expected_weights))
    else:
        raise AssertionError(f"unknown attention fixture kind: {kind}")


def _verify_clip(
    original: Clip, transformed: Clip, expected_weights: Mapping[str, torch.Tensor]
) -> None:
    _assert_preserves_source(original.patcher, expected_weights, transformed.patcher)
    _assert_scaled(transformed.patcher, expected_weights, _clip_strengths())


@pytest.mark.parametrize(
    ("node", "kind"),
    (
        (UNetSelfAttentionMultiply, "self"),
        (UNetCrossAttentionMultiply, "cross"),
        (UNetTemporalAttentionMultiply, "temporal"),
    ),
)
def test_unet_attention_multipliers_scale_exact_tensor_suffixes(
    node: type[Node], kind: str
) -> None:
    source = Patcher(_weights())
    expected_weights = {key: value.clone() for key, value in source.weights.items()}
    inputs: dict[str, Any] = {"model": source}
    inputs.update(TEMPORAL_STRENGTHS if kind == "temporal" else ATTENTION_STRENGTHS)
    transformed = cast(Patcher, node.execute(**inputs)["MODEL"])
    _verify_model(kind, source, transformed, expected_weights)


def test_clip_attention_multiplier_scales_exact_tensor_suffixes() -> None:
    source = Clip(Patcher(_weights()))
    expected_weights = {key: value.clone() for key, value in source.patcher.weights.items()}
    transformed = cast(
        Clip, CLIPAttentionMultiply.execute(clip=source, **ATTENTION_STRENGTHS)["CLIP"]
    )
    _verify_clip(source, transformed, expected_weights)


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.model-source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher(_weights()))


class ClipSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.clip-source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(CLIP=Clip(Patcher(_weights())))


class ModelReader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.model-reader",
            inputs=(
                InputSpec("original", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("transformed", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("kind", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("verified", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Patcher, transformed: Patcher, kind: str) -> Mapping[str, object]:
        _verify_model(kind, original, transformed, _weights())
        return cls.outputs(verified=1.0)


class ClipReader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-fixture.clip-reader",
            inputs=(
                InputSpec("original", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("transformed", TypeExpr.concrete(CLIP_TYPE)),
            ),
            outputs=(OutputSpec("verified", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Clip, transformed: Clip) -> Mapping[str, object]:
        _verify_clip(original, transformed, _weights())
        return cls.outputs(verified=1.0)


FIXTURE_NODES = (ModelSource, ClipSource, ModelReader, ClipReader)


def _model_graph(node_type: str, kind: str, inputs: Mapping[str, float]) -> Graph:
    return Graph(
        nodes={
            "source": GraphNode("attention-fixture.model-source"),
            "multiply": GraphNode(node_type, {"model": Link("source", "MODEL"), **inputs}),
            "reader": GraphNode(
                "attention-fixture.model-reader",
                {
                    "original": Link("source", "MODEL"),
                    "transformed": Link("multiply", "MODEL"),
                    "kind": kind,
                },
            ),
        }
    )


def _clip_graph() -> Graph:
    return Graph(
        nodes={
            "source": GraphNode("attention-fixture.clip-source"),
            "multiply": GraphNode(
                "CLIPAttentionMultiply", {"clip": Link("source", "CLIP"), **ATTENTION_STRENGTHS}
            ),
            "reader": GraphNode(
                "attention-fixture.clip-reader",
                {
                    "original": Link("source", "CLIP"),
                    "transformed": Link("multiply", "CLIP"),
                },
            ),
        }
    )


GRAPH_CASES = (
    ("UNetSelfAttentionMultiply", "self", ATTENTION_STRENGTHS),
    ("UNetCrossAttentionMultiply", "cross", ATTENTION_STRENGTHS),
    ("UNetTemporalAttentionMultiply", "temporal", TEMPORAL_STRENGTHS),
)


@pytest.mark.parametrize("in_process", (True,))
def test_complete_manifest_executes_all_attention_multipliers_in_each_worker_mode(
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
            engine = composer.composition.make_engine(lambda _event: None)
            graphs = [
                _model_graph(node_type, kind, inputs) for node_type, kind, inputs in GRAPH_CASES
            ] + [_clip_graph()]
            for _ in range(2):
                for graph in graphs:
                    result = await engine.run(graph, ["reader"])
                    assert result.outputs["reader"]["verified"].resolve() == 1.0
        finally:
            await composer.close()

    asyncio.run(scenario())
