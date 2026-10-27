"""CPU tensor and ordinary-pack execution behavior for SD upscaler conditioning."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sd_upscale_conditioning import SD_4XUpscale_Conditioning

from dinkster.compose import PackSpec, ServingComposer

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


def test_sd_upscale_rounds_asymmetric_dimensions_and_preserves_conditioning_sources() -> None:
    image = torch.arange(2 * 7 * 11 * 3, dtype=torch.float32).reshape(2, 7, 11, 3) / 511
    image_before = image.clone()
    positive = [[torch.tensor([[[1.0]]]), {"source": "positive"}]]
    negative = [[torch.tensor([[[2.0]]]), {"source": "negative"}]]
    result = SD_4XUpscale_Conditioning.execute(
        images=image.numpy(),
        positive=positive,
        negative=negative,
        scale_ratio=1.5,
        noise_augmentation=0.125,
    )
    assert torch.equal(image, image_before)
    assert positive[0][1] == {"source": "positive"}
    assert negative[0][1] == {"source": "negative"}
    assert result["positive"][0][0] is positive[0][0]
    assert result["negative"][0][0] is negative[0][0]
    assert result["positive"][0][1]["concat_image"].shape == (2, 3, 2, 4)
    expected_pixels = torch.tensor(
        [
            [
                [
                    [
                        (batch * 231 + row * 33 + column * 3 + channel) / 511 * 2 - 1
                        for column in (0.875, 3.625, 6.375, 9.125)
                    ]
                    for row in (1.75, 4.25)
                ]
                for channel in range(3)
            ]
            for batch in range(2)
        ]
    )
    assert torch.allclose(result["positive"][0][1]["concat_image"], expected_pixels)
    assert result["positive"][0][1]["noise_augmentation"] == 0.125
    assert result["negative"][0][1]["concat_image"] is result["positive"][0][1]["concat_image"]
    latent: Any = result["latent"]
    assert latent["samples"].shape == (2, 4, 2, 4)
    assert latent["samples"].dtype == torch.float32
    assert torch.count_nonzero(latent["samples"]) == 0


class UpscaleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sd-upscale-fixture.source",
            outputs=(
                OutputSpec("images", _IMAGE),
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            images=np.arange(2 * 7 * 11 * 3, dtype=np.float32).reshape(2, 7, 11, 3) / 511,
            positive=[[torch.ones((1, 1, 1)), {"source": "positive"}]],
            negative=[[torch.zeros((1, 1, 1)), {"source": "negative"}]],
        )


class UpscaleInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sd-upscale-fixture.inspect",
            inputs=(
                InputSpec("image", _IMAGE),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(
                OutputSpec("pixel", _FLOAT),
                OutputSpec("height", _FLOAT),
                OutputSpec("width", _FLOAT),
                OutputSpec("latent_batch", _FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, positive: Any, negative: Any, latent: Any) -> dict[str, object]:
        image = materialize_image(image)
        latent = materialize_latent(latent)
        assert positive[0][1]["noise_augmentation"] == 0.25
        assert negative[0][1]["concat_image"].shape == (2, 3, 2, 4)
        return cls.outputs(
            pixel=float(image[0, 0, 0, 0]),
            height=float(positive[0][1]["concat_image"].shape[-2]),
            width=float(positive[0][1]["concat_image"].shape[-1]),
            latent_batch=float(latent["samples"].shape[0]),
        )


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_executes_sd_upscale_conditioning_with_decoded_values(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(source_root), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "sd_upscale_fixture.py").write_text(
                "from test_sd_upscale_conditioning_torch import UpscaleInspect, UpscaleSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, UpscaleSource, UpscaleInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["sd-upscale-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "sd_upscale_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "sd_upscale_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("sd-upscale-fixture.source"),
                    "upscale": GraphNode(
                        "SD_4XUpscale_Conditioning",
                        {
                            "images": Link("source", "images"),
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "scale_ratio": 1.5,
                            "noise_augmentation": 0.25,
                        },
                    ),
                    "inspect": GraphNode(
                        "sd-upscale-fixture.inspect",
                        {
                            "image": Link("source", "images"),
                            "positive": Link("upscale", "positive"),
                            "negative": Link("upscale", "negative"),
                            "latent": Link("upscale", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                output = result.outputs["inspect"]
                assert output["pixel"].resolve() == 0.0
                assert output["height"].resolve() == 2.0
                assert output["width"].resolve() == 4.0
                assert output["latent_batch"].resolve() == 2.0
        finally:
            await composer.close()

    asyncio.run(scenario())
