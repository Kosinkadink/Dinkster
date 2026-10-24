"""CPU video latent creation and ordinary pack execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
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
from dinkster_nodes_comfy.video_latent import VIDEO_LATENT_NODES

from dinkster.compose import PackSpec, ServingComposer


class VideoLatentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-latent-fixture.inspect",
            inputs=tuple(
                InputSpec(node.schema().node_type, TypeExpr.concrete(LATENT_TYPE))
                for node in VIDEO_LATENT_NODES
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **latents: object) -> dict[str, object]:
        expected = (
            ((2, 16, 2, 11, 18), {"downscale_ratio_spacial": 8}),
            ((2, 32, 3, 6, 9), {"downscale_ratio_spacial": 16}),
            ((2, 128, 2, 3, 2), {"downscale_ratio_spacial": 32}),
            ((2, 16, 2, 8, 9), {}),
            ((2, 16, 2, 8, 9), {}),
        )
        shapes = []
        for node, (shape, metadata) in zip(VIDEO_LATENT_NODES, expected, strict=True):
            latent = materialize_latent(latents[node.schema().node_type])
            samples = latent["samples"]
            assert tuple(samples.shape) == shape
            assert samples.device.type == "cpu"
            assert samples.dtype == torch.float32
            assert not samples.count_nonzero()
            assert {key: value for key, value in latent.items() if key != "samples"} == metadata
            shapes.append(str(shape))
        return cls.outputs(summary=";".join(shapes))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_video_latents_twice(in_process: bool, tmp_path: Path) -> None:
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
            (tmp_path / "video_latent_fixture.py").write_text(
                "from test_video_latent_torch import VideoLatentInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, VideoLatentInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["video-latent-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "video_latent_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "video_latent_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            inputs = (
                {"width": 151, "height": 95, "length": 8, "batch_size": 2},
                {"width": 159, "height": 111, "length": 9, "batch_size": 2},
                {"width": 95, "height": 127, "length": 16, "batch_size": 2},
                {"width": 79, "height": 71, "length": 16, "batch_size": 2},
                {"width": 79, "height": 71, "length": 8, "batch_size": 2},
            )
            graph = Graph(
                nodes={
                    **{
                        node.schema().node_type: GraphNode(node.schema().node_type, inputs[index])
                        for index, node in enumerate(VIDEO_LATENT_NODES)
                    },
                    "inspect": GraphNode(
                        "video-latent-fixture.inspect",
                        {
                            node.schema().node_type: Link(node.schema().node_type, "LATENT")
                            for node in VIDEO_LATENT_NODES
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve()
        finally:
            await composer.close()

    asyncio.run(scenario())
