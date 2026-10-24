"""CPU Hunyuan image-to-video conditioning and manifest execution."""

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
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hunyuan_video_image import HunyuanImageToVideo, HunyuanVideo15ImageToVideo

from dinkster.compose import PackSpec, ServingComposer


class VAE:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        spatial = 8 if self.channels == 16 else 16
        base = image[::4, ::spatial, ::spatial, 0].unsqueeze(0).unsqueeze(0)
        return base.repeat(1, self.channels, 1, 1, 1)


def _image(frames: int = 6) -> np.ndarray:
    values = np.arange(frames * 4 * 6 * 4, dtype=np.float64).reshape(frames, 4, 6, 4)
    return values / 37.0


def _conditioning(value: float) -> list[list[Any]]:
    return [[torch.tensor([[value]], dtype=torch.float64), {"source": value}]]


def test_hunyuan_image_to_video_guidance_modes_preserve_asymmetric_values() -> None:
    image = _image()
    source = _conditioning(3.0)
    for guidance, key in (
        ("v1 (concat)", "concat_latent_image"),
        ("v2 (replace)", "guiding_frame_index"),
        ("custom", "ref_latent"),
    ):
        vae = VAE(16)
        result = HunyuanImageToVideo.execute(
            positive=source,
            vae=vae,
            width=16,
            height=16,
            length=9,
            batch_size=2,
            guidance_type=guidance,
            start_image=image,
        )
        samples = result["latent"]["samples"]
        metadata = result["positive"][0][1]
        assert samples.shape == (2, 16, 3, 2, 2)
        assert samples.dtype == torch.float32 and samples.device.type == "cpu"
        assert metadata["source"] == 3.0 and key in metadata
        assert source[0][1] == {"source": 3.0}
        assert vae.images[0].shape == (6, 16, 16, 3)
        assert vae.images[0].dtype is torch.float64
        if guidance == "v1 (concat)":
            assert metadata["concat_mask"].dtype is torch.float64
            assert metadata["concat_mask"].device.type == "cpu"
            assert not metadata["concat_mask"][:, :, :2].count_nonzero()
            assert metadata["concat_mask"][:, :, 2:].count_nonzero()
        elif guidance == "v2 (replace)":
            assert metadata == {"source": 3.0, "guiding_frame_index": 0}
            assert result["latent"]["noise_mask"].dtype is torch.float64
            assert samples[:, :, :2].count_nonzero()
        else:
            assert "noise_mask" not in result["latent"]
    no_image = HunyuanImageToVideo.execute(
        positive=source,
        vae=VAE(16),
        width=16,
        height=8,
        length=5,
        batch_size=1,
        guidance_type="custom",
    )
    assert no_image["positive"] is source and no_image["latent"] == {
        "samples": no_image["latent"]["samples"]
    }


def test_hunyuan_video15_materializes_vision_and_preserves_optional_identity() -> None:
    positive, negative = _conditioning(1.0), _conditioning(2.0)
    no_optional = HunyuanVideo15ImageToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(32),
        width=16,
        height=16,
        length=5,
        batch_size=2,
    )
    assert no_optional["positive"] is positive and no_optional["negative"] is negative
    assert no_optional["latent"] == {"samples": no_optional["latent"]["samples"]}

    vae = VAE(32)
    projected = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    result = HunyuanVideo15ImageToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=16,
        length=5,
        batch_size=2,
        start_image=_image(),
        clip_vision_output=SimpleNamespace(mm_projected=projected),
    )
    samples = result["latent"]["samples"]
    assert samples.shape == (2, 32, 2, 1, 1)
    assert vae.images[0].shape == (5, 16, 16, 3)
    for conditioning, value in ((result["positive"], 1.0), (result["negative"], 2.0)):
        metadata = conditioning[0][1]
        assert metadata["source"] == value
        assert metadata["concat_latent_image"].shape == (2, 32, 2, 1, 1)
        assert metadata["concat_mask"].dtype is torch.float64
        assert metadata["clip_vision_output"].mm_projected is projected
    assert positive[0][1] == {"source": 1.0} and negative[0][1] == {"source": 2.0}


_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_VISION = TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-video-image-fixture.source",
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("image", _IMAGE),
                OutputSpec("vae16", _VAE),
                OutputSpec("vae32", _VAE),
                OutputSpec("vision", _VISION),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=_conditioning(1.0),
            negative=_conditioning(2.0),
            image=_image(),
            vae16=VAE(16),
            vae32=VAE(32),
            vision=SimpleNamespace(
                mm_projected=torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
            ),
        )


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-video-image-fixture.inspect",
            inputs=(
                InputSpec("old", _CONDITIONING),
                InputSpec("old_latent", TypeExpr.concrete("LATENT")),
                InputSpec("new_positive", _CONDITIONING),
                InputSpec("new_negative", _CONDITIONING),
                InputSpec("new_latent", TypeExpr.concrete("LATENT")),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        old: Any,
        old_latent: object,
        new_positive: Any,
        new_negative: Any,
        new_latent: object,
    ) -> dict[str, object]:
        old_samples = materialize_latent(old_latent)["samples"]
        new_samples = materialize_latent(new_latent)["samples"]
        assert old_samples.shape == (1, 16, 2, 2, 2)
        assert new_samples.shape == (1, 32, 2, 1, 1)
        assert old[0][1]["ref_latent"].shape == (1, 16, 2, 2, 2)
        assert new_positive[0][1]["concat_latent_image"].shape == (1, 32, 2, 1, 1)
        assert new_negative[0][1]["clip_vision_output"].mm_projected.dtype is torch.float64
        return cls.outputs(summary="hunyuan video image")


@pytest.mark.parametrize("in_process", (True, False))
def test_production_manifest_registers_both_hunyuan_video_image_nodes_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(package / "src"))),
            }
        )
        try:
            (tmp_path / "hunyuan_video_image_fixture.py").write_text(
                "from test_hunyuan_video_image_torch import Inspect, Source\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, Source, Inspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (package / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hunyuan-video-image-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hunyuan_video_image_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hunyuan_video_image_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hunyuan-video-image-fixture.source"),
                    "old": GraphNode(
                        "HunyuanImageToVideo",
                        {
                            "positive": Link("source", "positive"),
                            "vae": Link("source", "vae16"),
                            "width": 16,
                            "height": 16,
                            "length": 5,
                            "batch_size": 1,
                            "guidance_type": "custom",
                            "start_image": Link("source", "image"),
                        },
                    ),
                    "new": GraphNode(
                        "HunyuanVideo15ImageToVideo",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae32"),
                            "width": 16,
                            "height": 16,
                            "length": 5,
                            "batch_size": 1,
                            "start_image": Link("source", "image"),
                            "clip_vision_output": Link("source", "vision"),
                        },
                    ),
                    "inspect": GraphNode(
                        "hunyuan-video-image-fixture.inspect",
                        {
                            "old": Link("old", "positive"),
                            "old_latent": Link("old", "latent"),
                            "new_positive": Link("new", "positive"),
                            "new_negative": Link("new", "negative"),
                            "new_latent": Link("new", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "hunyuan video image"
        finally:
            await composer.close()

    asyncio.run(scenario())
