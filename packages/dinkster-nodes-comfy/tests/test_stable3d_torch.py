"""CPU execution and ordinary manifest hosting for Stable Zero123 and SV3D."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.stable3d import (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)

from dinkster.compose import PackSpec, ServingComposer


class FakeClipVision:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode_image(self, image: torch.Tensor) -> SimpleNamespace:
        self.images.append(image.clone())
        return SimpleNamespace(image_embeds=image.mean(dim=(1, 2)))


class FakeVAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image.mean(-1, keepdim=True).movedim(-1, 1)


def _image() -> np.ndarray:
    image = np.zeros((1, 35, 51, 4), dtype=np.float32)
    image[..., 0] = 0.25
    image[..., 1] = 0.5
    image[..., 2] = 0.75
    image[..., 3] = 0.9
    return image


def test_zero123_conditioning_preserves_source_and_negative_shape() -> None:
    image = _image()
    before = image.copy()
    clip_vision = FakeClipVision()
    vae = FakeVAE()
    result = StableZero123_Conditioning.execute(
        clip_vision=clip_vision,
        init_image=image,
        vae=vae,
        width=50,
        height=34,
        batch_size=3,
        elevation=12.5,
        azimuth=-30.0,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert positive[0].shape == (1, 1, 8)
    assert negative[0].shape == (1, 1, 4)
    assert result["latent"]["samples"].shape == (3, 4, 4, 6)
    assert clip_vision.images[0].shape[-1] == 4
    assert vae.images[0].shape == (1, 34, 50, 3)
    assert not negative[0].count_nonzero()
    assert not negative[1]["concat_latent_image"].count_nonzero()
    np.testing.assert_array_equal(image, before)


def test_batched_zero123_increments_camera_and_sets_batch_index() -> None:
    result = StableZero123_Conditioning_Batched.execute(
        clip_vision=FakeClipVision(),
        init_image=_image(),
        vae=FakeVAE(),
        width=50,
        height=34,
        batch_size=3,
        elevation=10.0,
        azimuth=20.0,
        elevation_batch_increment=15.0,
        azimuth_batch_increment=30.0,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert positive[0].shape == (3, 1, 8)
    assert negative[0].shape == (1, 1, 4)
    assert result["latent"]["batch_index"] == [0, 0, 0]
    expected = torch.deg2rad(torch.tensor([-10.0, -25.0, -40.0]))
    torch.testing.assert_close(positive[0][:, 0, 4], expected)
    torch.testing.assert_close(
        positive[0][:, 0, 5], torch.sin(torch.deg2rad(torch.tensor([20.0, 50.0, 80.0])))
    )


def test_sv3d_conditioning_records_azimuth_and_elevation() -> None:
    result = SV3D_Conditioning.execute(
        clip_vision=FakeClipVision(),
        init_image=_image(),
        vae=FakeVAE(),
        width=50,
        height=34,
        video_frames=3,
        elevation=17.5,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert result["latent"]["samples"].shape == (3, 4, 4, 6)
    assert positive[1]["elevation"] == [17.5, 17.5, 17.5]
    assert positive[1]["azimuth"] == [0.0, 180.0, 360.0]
    assert negative[1]["elevation"] == positive[1]["elevation"]
    assert negative[1]["azimuth"] == positive[1]["azimuth"]
    assert positive[0].shape == negative[0].shape == (1, 1, 4)


_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class Stable3DSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="stable3d-fixture.source",
            outputs=(
                OutputSpec("clip_vision", _CLIP_VISION),
                OutputSpec("image", _IMAGE),
                OutputSpec("vae", _VAE),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip_vision=FakeClipVision(), image=_image(), vae=FakeVAE())


class Stable3DInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="stable3d-fixture.inspect",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, latent: Any) -> dict[str, object]:
        latent = materialize_image(materialize_latent(latent)["samples"])
        positive_tensor, positive_metadata = positive[0]
        negative_tensor, negative_metadata = negative[0]
        assert positive_tensor.shape == (2, 1, 8)
        assert negative_tensor.shape == (1, 1, 4)
        assert positive_metadata["concat_latent_image"].shape == (1, 1, 34, 50)
        assert negative_metadata["concat_latent_image"].shape == (1, 1, 34, 50)
        assert latent.shape == (2, 4, 4, 6)
        return cls.outputs(summary="stable3d")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_stable3d_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "stable3d_fixture.py").write_text(
                "from test_stable3d_torch import Stable3DInspect, Stable3DSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, Stable3DSource, Stable3DInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["stable3d-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "stable3d_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "stable3d_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("stable3d-fixture.source"),
                    "conditioning": GraphNode(
                        "StableZero123_Conditioning_Batched",
                        {
                            "clip_vision": Link("source", "clip_vision"),
                            "init_image": Link("source", "image"),
                            "vae": Link("source", "vae"),
                            "width": 50,
                            "height": 34,
                            "batch_size": 2,
                            "elevation": 10.0,
                            "azimuth": 20.0,
                            "elevation_batch_increment": 5.0,
                            "azimuth_batch_increment": 10.0,
                        },
                    ),
                    "inspect": GraphNode(
                        "stable3d-fixture.inspect",
                        {
                            "positive": Link("conditioning", "positive"),
                            "negative": Link("conditioning", "negative"),
                            "latent": Link("conditioning", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "stable3d"
        finally:
            await composer.close()

    asyncio.run(scenario())
