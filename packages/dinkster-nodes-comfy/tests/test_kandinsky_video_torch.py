"""Kandinsky start conditioning through public CPU and pack interfaces."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
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
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.kandinsky_video import Kandinsky5ImageToVideo

from dinkster.compose import PackSpec, ServingComposer


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[-1] == 3
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def image() -> torch.Tensor:
    return torch.arange(6 * 6 * 12 * 4, dtype=torch.float64).reshape(6, 6, 12, 4) / 37


def conditioning(value: float) -> list[list[Any]]:
    return [[torch.tensor([[value]]), {"source": value}]]


def check(
    positive: Any,
    negative: Any,
    latent: object,
    clean: object,
    length: int,
    start: bool,
    image_dtype: torch.dtype = torch.float64,
) -> None:
    samples = materialize_latent(latent)["samples"]
    assert samples.shape == (2, 16, (length - 1) // 4 + 1, 2, 4)
    assert samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()
    clean = materialize_latent(clean)
    if not start:
        assert clean == {}
        assert positive[0][1] == {"source": 3.0}
        assert negative[0][1] == {"source": -7.0}
        return
    pixels = torch.nn.functional.interpolate(
        image().to(image_dtype)[:length].movedim(-1, 1),
        size=(16, 32),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    encoded = pixels[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)
    torch.testing.assert_close(clean["samples"], encoded, rtol=0, atol=0)
    mask = torch.ones((1, 1, samples.shape[2], 2, 4), dtype=image_dtype)
    mask[:, :, : (min(length, 6) - 1) // 4 + 1] = 0
    for cond, source in ((positive, 3.0), (negative, -7.0)):
        assert cond[0][1]["source"] == source
        torch.testing.assert_close(cond[0][1]["time_dim_replace"], encoded, rtol=0, atol=0)
        torch.testing.assert_close(cond[0][1]["concat_mask"], mask, rtol=0, atol=0)


@pytest.mark.parametrize("length", (5, 9))
def test_public_start_and_empty_conditioning_preserve_sources(length: int) -> None:
    positive, negative, pixels = conditioning(3.0), conditioning(-7.0), image()
    before = pixels.clone()
    for start in (False, True):
        result = Kandinsky5ImageToVideo.execute(
            positive=positive,
            negative=negative,
            vae=VAE(),
            width=32,
            height=16,
            length=length,
            batch_size=2,
            start_image=pixels if start else None,
        )
        check(
            result["positive"],
            result["negative"],
            result["latent"],
            result["cond_latent"],
            length,
            start,
        )
    assert positive[0][1] == {"source": 3.0} and negative[0][1] == {"source": -7.0}
    assert torch.equal(pixels, before)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="kandinsky-video-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("image", IMAGE_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=conditioning(3.0), negative=conditioning(-7.0), image=image(), vae=VAE()
        )


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="kandinsky-video-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                    ("clean", LATENT_TYPE),
                )
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, positive: Any, negative: Any, latent: object, clean: object
    ) -> dict[str, object]:
        check(positive, negative, latent, clean, 9, True, torch.float32)
        return cls.outputs(value="verified")


@pytest.mark.parametrize("in_process", (True,))
def test_unchanged_manifest_start_conditioning_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "kandinsky_fixture.py").write_text(
            "from test_kandinsky_video_torch import Source, Inspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (Source, Inspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname="kandinsky-video-fixture"\npublisher="dinkster"\n'
            'namespaces=["kandinsky-video-fixture"]\n[pack.entry]\n'
            'nodes="kandinsky_fixture:NODES"\ntypes="kandinsky_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), *sys.path)),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            native_manifest = package / "dinkster-pack.toml"
            if in_process:
                await composer.add_pack(PackSpec(manifest=manifest, in_process=True))
                await composer.add_pack(PackSpec(manifest=native_manifest, in_process=True))
            else:
                group = (manifest, native_manifest)
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        worker_group="kandinsky-video-fixture",
                        group_manifests=group,
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="kandinsky-video-fixture",
                        group_manifests=group,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("kandinsky-video-fixture.source"),
                    "video": GraphNode(
                        "Kandinsky5ImageToVideo",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "start_image": Link("source", "image"),
                            "width": 32,
                            "height": 16,
                            "length": 9,
                            "batch_size": 2,
                        },
                    ),
                    "consumer": GraphNode(
                        "kandinsky-video-fixture.inspect",
                        {
                            "positive": Link("video", "positive"),
                            "negative": Link("video", "negative"),
                            "latent": Link("video", "latent"),
                            "clean": Link("video", "cond_latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["value"].resolve() == "verified"
        finally:
            await composer.close()

    asyncio.run(scenario())
