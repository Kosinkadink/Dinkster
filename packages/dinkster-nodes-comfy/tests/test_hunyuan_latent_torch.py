"""CPU Hunyuan image latent behavior and ordinary manifest hosting."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hunyuan_latent import EmptyHunyuanImageLatent, HunyuanRefinerLatent

from dinkster.compose import PackSpec, ServingComposer


def test_hunyuan_image_latents_preserve_source_values_and_allocate_on_cpu() -> None:
    empty = EmptyHunyuanImageLatent.execute(width=195, height=99, batch_size=2)["LATENT"]
    empty_samples = empty["samples"]
    assert tuple(empty_samples.shape) == (2, 64, 3, 6)
    assert empty_samples.dtype == torch.float32
    assert empty_samples.device.type == "cpu"
    assert not empty_samples.count_nonzero()

    samples = torch.arange(2 * 16 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 16, 3, 5, 7)
    source = {"samples": samples, "batch_index": [3, 1], "unrelated": "retained"}
    positive = [[torch.ones((1, 2, 4)), {"positive": True}]]
    negative = [[torch.full((1, 2, 4), 2.0), {"negative": True}]]
    result = HunyuanRefinerLatent.execute(
        positive=positive, negative=negative, latent=source, noise_augmentation=0.35
    )
    refined = result["latent"]
    refined_samples = refined["samples"]
    assert tuple(refined_samples.shape) == (2, 32, 3, 5, 7)
    assert refined_samples.dtype == torch.float32
    assert refined_samples.device.type == "cpu"
    assert not refined_samples.count_nonzero()
    assert refined["batch_index"] == [3, 1]
    assert refined["unrelated"] == "retained"
    assert source["samples"] is samples
    assert source["batch_index"] == [3, 1]
    assert positive[0][1] == {"positive": True}
    assert negative[0][1] == {"negative": True}
    for conditioning, metadata_key in (
        (result["positive"], "positive"),
        (result["negative"], "negative"),
    ):
        assert conditioning[0][0] is (positive if metadata_key == "positive" else negative)[0][0]
        assert conditioning[0][1][metadata_key] is True
        assert conditioning[0][1]["concat_latent_image"] is samples
        assert conditioning[0][1]["noise_augmentation"] == 0.35


class HunyuanSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-fixture.source",
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        samples = torch.arange(2 * 16 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 16, 3, 5, 7)
        return cls.outputs(
            positive=[[torch.ones((1, 2, 4)), {"positive": True}]],
            negative=[[torch.full((1, 2, 4), 2.0), {"negative": True}]],
            latent={"samples": samples, "batch_index": [3, 1], "unrelated": "retained"},
        )


class HunyuanInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-fixture.inspect",
            inputs=(
                InputSpec("empty", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, empty: object, positive: Any, negative: Any, latent: object
    ) -> dict[str, object]:
        empty_samples = materialize_latent(empty)["samples"]
        refined = materialize_latent(latent)
        samples = refined["samples"]
        assert tuple(empty_samples.shape) == (2, 64, 3, 6)
        assert tuple(samples.shape) == (2, 32, 3, 5, 7)
        assert all(
            value.dtype == torch.float32 and value.device.type == "cpu"
            for value in (empty_samples, samples)
        )
        assert not empty_samples.count_nonzero() and not samples.count_nonzero()
        assert refined["batch_index"] == [3, 1]
        assert refined["unrelated"] == "retained"
        for conditioning, expected in ((positive, "positive"), (negative, "negative")):
            metadata = conditioning[0][1]
            assert metadata[expected] is True
            assert tuple(metadata["concat_latent_image"].shape) == (2, 16, 3, 5, 7)
            assert metadata["noise_augmentation"] == 0.35
        return cls.outputs(summary="ok")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_hunyuan_latents_twice(in_process: bool, tmp_path: Path) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "hunyuan_fixture.py").write_text(
                "from test_hunyuan_latent_torch import HunyuanInspect, HunyuanSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, HunyuanSource, HunyuanInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hunyuan-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hunyuan_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hunyuan_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hunyuan-fixture.source"),
                    "empty": GraphNode(
                        "EmptyHunyuanImageLatent", {"width": 195, "height": 99, "batch_size": 2}
                    ),
                    "refiner": GraphNode(
                        "HunyuanRefinerLatent",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "latent": Link("source", "latent"),
                            "noise_augmentation": 0.35,
                        },
                    ),
                    "inspect": GraphNode(
                        "hunyuan-fixture.inspect",
                        {
                            "empty": Link("empty", "LATENT"),
                            "positive": Link("refiner", "positive"),
                            "negative": Link("refiner", "negative"),
                            "latent": Link("refiner", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "ok"
        finally:
            await composer.close()

    asyncio.run(scenario())
