"""Native merging preserves coefficients, source values and excluded CLIP keys."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_merging import (
    CLIPMergeAdd,
    CLIPMergeSimple,
    CLIPMergeSubtract,
    ModelMergeAdd,
    ModelMergeBlocks,
    ModelMergeSimple,
    ModelMergeSubtract,
)

from dinkster.compose import PackSpec, ServingComposer


class ScalarPatcher:
    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = dict(weights)

    def clone(self) -> ScalarPatcher:
        return ScalarPatcher(self.weights)

    def get_key_patches(self, prefix: str = "") -> dict[str, float]:
        return {key: value for key, value in self.weights.items() if key.startswith(prefix)}

    def add_patches(
        self, patches: Mapping[str, float], strength_patch: float, strength_model: float
    ) -> None:
        for key, value in patches.items():
            if key in self.weights:
                self.weights[key] = self.weights[key] * strength_model + value * strength_patch


@pytest.mark.parametrize(
    ("coefficient", "simple", "subtract"),
    [(0.0, -4.0, 0.0), (0.25, -0.5, 3.5), (1.0, 10.0, 14.0), (-2.0, -32.0, -28.0)],
)
def test_model_merges_do_not_reverse_coefficients_or_mutate_sources(
    coefficient: float, simple: float, subtract: float
) -> None:
    first = ScalarPatcher({"diffusion_model.weight": 10.0, "other.weight": 31.0})
    second = ScalarPatcher({"diffusion_model.weight": -4.0, "other.weight": 57.0})
    merged = cast(
        ScalarPatcher,
        ModelMergeSimple.execute(model1=first, model2=second, ratio=coefficient)["MODEL"],
    )
    difference = cast(
        ScalarPatcher,
        ModelMergeSubtract.execute(model1=first, model2=second, multiplier=coefficient)["MODEL"],
    )
    added = cast(ScalarPatcher, ModelMergeAdd.execute(model1=first, model2=second)["MODEL"])
    assert merged.weights == {"diffusion_model.weight": simple, "other.weight": 31.0}
    assert difference.weights == {"diffusion_model.weight": subtract, "other.weight": 31.0}
    assert added.weights == {"diffusion_model.weight": 6.0, "other.weight": 31.0}
    assert first.weights == {"diffusion_model.weight": 10.0, "other.weight": 31.0}
    assert second.weights == {"diffusion_model.weight": -4.0, "other.weight": 57.0}
    assert merged is not first and difference is not first and added is not first


def test_clip_merges_skip_only_the_two_exact_suffixes() -> None:
    first = ScalarPatcher(
        {"text.weight": 10.0, "x.position_ids": 3.0, "x.logit_scale": 7.0, "logit_scale": 11.0}
    )
    second = ScalarPatcher(
        {"text.weight": -4.0, "x.position_ids": 99.0, "x.logit_scale": 88.0, "logit_scale": -5.0}
    )
    merged = cast(
        ScalarPatcher, CLIPMergeSimple.execute(clip1=first, clip2=second, ratio=0.25)["CLIP"]
    )
    difference = cast(
        ScalarPatcher, CLIPMergeSubtract.execute(clip1=first, clip2=second, multiplier=-2.0)["CLIP"]
    )
    added = cast(ScalarPatcher, CLIPMergeAdd.execute(clip1=first, clip2=second)["CLIP"])
    assert merged.weights == {
        "text.weight": -0.5,
        "x.position_ids": 3.0,
        "x.logit_scale": 7.0,
        "logit_scale": -1.0,
    }
    assert difference.weights == {
        "text.weight": -28.0,
        "x.position_ids": 3.0,
        "x.logit_scale": 7.0,
        "logit_scale": -32.0,
    }
    assert added.weights == {
        "text.weight": 6.0,
        "x.position_ids": 3.0,
        "x.logit_scale": 7.0,
        "logit_scale": 6.0,
    }
    assert first.weights["text.weight"] == 10.0
    assert second.weights["x.position_ids"] == 99.0


def test_block_merge_uses_longest_prefix_and_first_coefficient_fallback() -> None:
    weights = {
        "diffusion_model.input.weight": 10.0,
        "diffusion_model.input.deep.weight": 10.0,
        "diffusion_model.middle.weight": 10.0,
        "diffusion_model.unmatched": 10.0,
        "outside": 23.0,
    }
    first = ScalarPatcher(weights)
    second = ScalarPatcher({key: -6.0 for key in weights})
    merged = cast(
        ScalarPatcher,
        ModelMergeBlocks.execute(
            model1=first,
            model2=second,
            **{"input": 0.25, "input.deep.": 0.75, "middle": 0.5, "out": 1.0},
        )["MODEL"],
    )
    assert merged.weights == {
        "diffusion_model.input.weight": -2.0,
        "diffusion_model.input.deep.weight": 6.0,
        "diffusion_model.middle.weight": 2.0,
        "diffusion_model.unmatched": -2.0,
        "outside": 23.0,
    }
    assert first.weights == weights


class ScalarModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="merge-fixture.model",
            inputs=(InputSpec("weights", TypeExpr.concrete("DICT")),),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete("MODEL")),),
        )

    @classmethod
    def execute(cls, *, weights: Mapping[str, float]) -> Mapping[str, object]:
        return cls.outputs(MODEL=ScalarPatcher(weights))


class ScalarModelWeights(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="merge-fixture.weights",
            inputs=(InputSpec("model", TypeExpr.concrete("MODEL")),),
            outputs=(OutputSpec("weights", TypeExpr.concrete("DICT")),),
        )

    @classmethod
    def execute(cls, *, model: ScalarPatcher) -> Mapping[str, object]:
        return cls.outputs(weights=model.weights)


PATCHER_NODES = (ScalarModelSource, ScalarModelWeights)


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_registers_model_types_for_graph_execution(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "merging_fixture_sources.py").write_text(
                "from test_comfy_model_merging_nodes import PATCHER_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, *PATCHER_NODES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["merge-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "merging_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "merging_fixture_sources:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "merging_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "first": GraphNode(
                            "merge-fixture.model", {"weights": {"diffusion_model.weight": 10.0}}
                        ),
                        "second": GraphNode(
                            "merge-fixture.model", {"weights": {"diffusion_model.weight": -4.0}}
                        ),
                        "merge": GraphNode(
                            "ModelMergeSimple",
                            {
                                "model1": Link("first", "MODEL"),
                                "model2": Link("second", "MODEL"),
                                "ratio": 0.25,
                            },
                        ),
                        "merged-weights": GraphNode(
                            "merge-fixture.weights", {"model": Link("merge", "MODEL")}
                        ),
                        "source-weights": GraphNode(
                            "merge-fixture.weights", {"model": Link("first", "MODEL")}
                        ),
                    }
                ),
                ["merged-weights", "source-weights"],
            )
            assert result.outputs["merged-weights"]["weights"].resolve() == {
                "diffusion_model.weight": -0.5
            }
            assert result.outputs["source-weights"]["weights"].resolve() == {
                "diffusion_model.weight": 10.0
            }
        finally:
            await composer.close()

    asyncio.run(scenario())
