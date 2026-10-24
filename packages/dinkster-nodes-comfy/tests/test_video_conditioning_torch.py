"""CPU behavior and manifest hosting for Stable Video image conditioning."""

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
from dinkster_nodes_comfy.video_conditioning import SVD_img2vid_Conditioning

from dinkster.compose import PackSpec, ServingComposer


class ClipVision:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode_image(self, image: torch.Tensor) -> SimpleNamespace:
        self.images.append(image.clone())
        return SimpleNamespace(image_embeds=image.mean((1, 2)))


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image.movedim(-1, 1)


def _image() -> np.ndarray:
    return np.arange(1 * 7 * 13 * 4, dtype=np.float32).reshape(1, 7, 13, 4) / 127.0


def _execute(augmentation_level: float) -> tuple[dict[str, object], ClipVision, VAE]:
    clip_vision = ClipVision()
    vae = VAE()
    result = SVD_img2vid_Conditioning.execute(
        clip_vision=clip_vision,
        init_image=_image(),
        vae=vae,
        width=10,
        height=6,
        video_frames=3,
        motion_bucket_id=127,
        fps=6,
        augmentation_level=augmentation_level,
    )
    return dict(result), clip_vision, vae


def test_stable_video_conditioning_preserves_source_operations() -> None:
    result, clip_vision, vae = _execute(0.0)
    positive = result["positive"][0]
    negative = result["negative"][0]
    assert clip_vision.images[0].shape == (1, 7, 13, 4)
    assert vae.images[0].shape == (1, 6, 10, 3)
    assert result["latent"]["samples"].shape == (3, 4, 0, 1)
    assert result["latent"]["samples"].dtype == torch.get_default_dtype()
    assert result["latent"]["samples"].device.type == "cpu"
    assert positive[1] == {
        "motion_bucket_id": 127,
        "fps": 6,
        "augmentation_level": 0.0,
        "concat_latent_image": positive[1]["concat_latent_image"],
    }
    assert negative[1]["motion_bucket_id"] == positive[1]["motion_bucket_id"]
    assert negative[1]["fps"] == positive[1]["fps"]
    assert negative[1]["augmentation_level"] == positive[1]["augmentation_level"]
    assert not negative[0].count_nonzero()
    assert not negative[1]["concat_latent_image"].count_nonzero()


def test_stable_video_conditioning_uses_source_augmentation_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _image()[..., :3]
    noise = torch.arange(1 * 6 * 10 * 3, dtype=torch.float32).reshape(1, 6, 10, 3) / 19.0
    monkeypatch.setattr(torch, "randn_like", lambda pixels: noise)
    vae = VAE()
    result = SVD_img2vid_Conditioning.execute(
        clip_vision=ClipVision(),
        init_image=image,
        vae=vae,
        width=10,
        height=6,
        video_frames=3,
        motion_bucket_id=127,
        fps=6,
        augmentation_level=0.25,
    )
    from dinkster_inference.utils import common_upscale

    pixels = common_upscale(
        torch.as_tensor(image).movedim(-1, 1), 10, 6, "bilinear", "center"
    ).movedim(1, -1)
    expected = pixels[:, :, :, :3] + noise * 0.25
    torch.testing.assert_close(vae.images[0], expected)
    torch.testing.assert_close(
        result["positive"][0][1]["concat_latent_image"], expected.movedim(-1, 1)
    )


_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-conditioning-fixture.source",
            outputs=(
                OutputSpec("clip_vision", _CLIP_VISION),
                OutputSpec("image", _IMAGE),
                OutputSpec("vae", _VAE),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip_vision=ClipVision(), image=_image(), vae=VAE())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-conditioning-fixture.inspect",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, latent: Any) -> dict[str, object]:
        assert positive[0][1]["motion_bucket_id"] == negative[0][1]["motion_bucket_id"] == 11
        assert materialize_image(materialize_latent(latent)["samples"]).shape == (2, 4, 1, 1)
        return cls.outputs(summary="stable-video")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_stable_video_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src"))
                ),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "video_conditioning_fixture.py").write_text(
                "from test_video_conditioning_torch import Inspect, Source\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, Source, Inspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["video-conditioning-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "video_conditioning_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "video_conditioning_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("video-conditioning-fixture.source"),
                    "conditioning": GraphNode(
                        "SVD_img2vid_Conditioning",
                        {
                            "clip_vision": Link("source", "clip_vision"),
                            "init_image": Link("source", "image"),
                            "vae": Link("source", "vae"),
                            "width": 8,
                            "height": 8,
                            "video_frames": 2,
                            "motion_bucket_id": 11,
                            "fps": 7,
                            "augmentation_level": 0.0,
                        },
                    ),
                    "inspect": GraphNode(
                        "video-conditioning-fixture.inspect",
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
                assert (await engine.run(graph, ["inspect"])).outputs["inspect"][
                    "summary"
                ].resolve() == "stable-video"
        finally:
            await composer.close()

    asyncio.run(scenario())
