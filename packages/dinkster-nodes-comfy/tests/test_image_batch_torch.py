"""Public CPU image dimension and rebatch behavior."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.rebatch import RebatchImages, RebatchLatents

from dinkster.compose import PackSpec, ServingComposer


def test_rebatch_preserves_frame_order_and_latent_indices() -> None:
    first = torch.arange(2 * 3 * 5 * 3, dtype=torch.float32).reshape(2, 3, 5, 3)
    second = torch.full((1, 3, 5, 3), 123.0)
    images = RebatchImages.execute(images=[first, second], batch_size=[2])["IMAGE"]
    assert len(images) == 2
    assert torch.equal(images[0], first)
    assert torch.equal(images[1], second)

    samples = torch.arange(3 * 4 * 2 * 5, dtype=torch.float32).reshape(3, 4, 2, 5)
    latents = RebatchLatents.execute(
        latents=[{"samples": samples, "batch_index": [8, 3, 11]}], batch_size=[2]
    )["LATENT"]
    assert [item["batch_index"] for item in latents] == [[8, 3], [11]]
    assert torch.equal(latents[0]["samples"], samples[:2])
    assert torch.equal(latents[1]["samples"], samples[2:])
    assert all("noise_mask" not in item for item in latents)


class DimensionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-dimension-fixture.source",
            inputs=(InputSpec("video", TypeExpr.concrete(CORE_BOOLEAN)),),
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, video: bool) -> Any:
        return cls.outputs(
            image=torch.zeros((2, 3, 7, 4)),
            latent={"samples": torch.zeros((3, 4, 5, 6, 9) if video else (3, 4, 6, 9))},
        )


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("video", (False, True))
def test_image_dimensions_ordinary_manifest_twice(
    tmp_path: Path, in_process: bool, video: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = f"dimension_fixture_{in_process}_{video}"
            (tmp_path / f"{module}.py").write_text(
                "from test_image_batch_torch import DimensionSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, DimensionSource)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["image-dimension-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("image-dimension-fixture.source", {"video": video}),
                    "image": GraphNode("GetImageSize", {"image": Link("source", "image")}),
                    "latent": GraphNode("GetLatentSize", {"latent": Link("source", "latent")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["image", "latent"])
                assert {
                    name: {key: value.resolve() for key, value in outputs.items()}
                    for name, outputs in result.outputs.items()
                } == {
                    "image": {"width": 7, "height": 3, "batch_size": 2},
                    "latent": {
                        "width": 9,
                        "height": 6,
                        "frames": 5 if video else 1,
                        "batch_size": 3,
                    },
                }
        finally:
            await composer.close()

    asyncio.run(scenario())
