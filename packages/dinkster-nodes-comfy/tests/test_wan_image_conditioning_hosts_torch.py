"""Native Wan image conditioning through both ordinary pack hosts."""

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
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class VideoEncoder:
    latent_channels = 16

    def spacial_compression_encode(self) -> int:
        return 8

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image[::4, ::8, ::8, :1].movedim(-1, 0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-image-fixture.source",
            outputs=tuple(
                OutputSpec(n, TypeExpr.concrete(t))
                for n, t in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("start", IMAGE_TYPE),
                    ("end", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[[torch.ones(1, 2, 3), {"label": "positive"}]],
            negative=[[torch.ones(1, 2, 3), {"label": "negative"}]],
            vae=VideoEncoder(),
            start=torch.full((1, 16, 32, 3), 0.25),
            end=torch.full((1, 16, 32, 3), 0.75),
        )


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-image-fixture.consumer",
            inputs=tuple(
                InputSpec(n, TypeExpr.concrete(t))
                for n, t in (
                    ("image", CONDITIONING_TYPE),
                    ("first_last", CONDITIONING_TYPE),
                    ("inpaint", CONDITIONING_TYPE),
                    ("control", CONDITIONING_TYPE),
                    ("control22", CONDITIONING_TYPE),
                    ("vace", CONDITIONING_TYPE),
                    ("phantom", CONDITIONING_TYPE),
                    ("phantom_negative", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: Any,
        first_last: Any,
        inpaint: Any,
        control: Any,
        control22: Any,
        vace: Any,
        phantom: Any,
        phantom_negative: Any,
        latent: Any,
    ) -> dict[str, object]:
        samples = materialize_latent(latent)["samples"]
        assert samples.shape == (2, 16, 3, 2, 4) and not samples.count_nonzero()
        for value in (image, first_last, inpaint):
            assert value[0][1]["label"] == "positive"
            encoded = value[0][1]["concat_latent_image"]
            assert encoded.shape == (1, 16, 3, 2, 4)
            torch.testing.assert_close(
                encoded[:, :, 0], torch.full((1, 16, 2, 4), 0.25), rtol=0, atol=0
            )
        torch.testing.assert_close(
            image[0][1]["concat_latent_image"][:, :, -1],
            torch.full((1, 16, 2, 4), 0.5),
            rtol=0,
            atol=0,
        )
        for value in (first_last, inpaint):
            torch.testing.assert_close(
                value[0][1]["concat_latent_image"][:, :, -1],
                torch.full((1, 16, 2, 4), 0.75),
                rtol=0,
                atol=0,
            )
            mask = value[0][1]["concat_mask"]
            assert mask.shape == (1, 4, 3, 2, 4)
            assert not mask[:, :, 0].count_nonzero()
            assert mask[0, 2, 2, 0, 0] == 1 and mask[0, 3, 2, 0, 0] == 0
        for value in (control, control22):
            encoded = value[0][1]["concat_latent_image"]
            assert encoded.shape == (2, 32, 3, 2, 4)
            torch.testing.assert_close(
                encoded[:, :16, :1], torch.full((2, 16, 1, 2, 4), 0.25), rtol=0, atol=0
            )
        torch.testing.assert_close(
            control[0][1]["concat_latent_image"][:, 16:, :1],
            torch.full((2, 16, 1, 2, 4), 0.75),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            control22[0][1]["reference_latents"][0],
            torch.full((1, 16, 1, 2, 4), 0.75),
            rtol=0,
            atol=0,
        )
        vace_metadata = vace[0][1]
        assert vace_metadata["vace_frames"][0].shape == (1, 32, 4, 2, 4)
        assert vace_metadata["vace_mask"][0].shape == (1, 64, 4, 2, 4)
        assert not vace_metadata["vace_mask"][0][:, :, :1].count_nonzero()
        assert vace_metadata["vace_strength"] == [0.75]
        torch.testing.assert_close(
            phantom[0][1]["time_dim_concat"],
            torch.full((1, 16, 1, 2, 4), 0.25),
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            phantom[0][1]["time_dim_concat"], phantom_negative[0][1]["time_dim_concat"]
        )
        return cls.outputs(summary="wan-image-first-last-inpaint")


HOST_NODES = (Source, Consumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_wan_image_conditioning_runs_twice_in_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "wan_image_host_entry.py").write_text(
            "from test_wan_image_conditioning_hosts_torch import (\n"
            "    HOST_NODES, register_inference_types,\n)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wan-image-fixture"\npublisher="dinkster"\n'
            'namespaces=["wan-image-fixture"]\n[pack.entry]\n'
            'nodes="wan_image_host_entry:HOST_NODES"\n'
            'types="wan_image_host_entry:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "wan-image-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            inputs = {
                "positive": Link("source", "positive"),
                "negative": Link("source", "negative"),
                "vae": Link("source", "vae"),
                "width": 32,
                "height": 16,
                "length": 9,
                "batch_size": 2,
                "start_image": Link("source", "start"),
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("wan-image-fixture.source"),
                    "image": GraphNode("WanImageToVideo", inputs),
                    "first_last": GraphNode(
                        "WanFirstLastFrameToVideo", {**inputs, "end_image": Link("source", "end")}
                    ),
                    "inpaint": GraphNode(
                        "WanFunInpaintToVideo", {**inputs, "end_image": Link("source", "end")}
                    ),
                    "control": GraphNode(
                        "WanFunControlToVideo",
                        {
                            **inputs,
                            "start_image": Link("source", "end"),
                            "control_video": Link("source", "start"),
                        },
                    ),
                    "control22": GraphNode(
                        "Wan22FunControlToVideo",
                        {
                            k: v
                            for k, v in {
                                **inputs,
                                "ref_image": Link("source", "end"),
                                "control_video": Link("source", "start"),
                            }.items()
                            if k != "start_image"
                        },
                    ),
                    "vace": GraphNode(
                        "WanVaceToVideo",
                        {
                            k: v
                            for k, v in {
                                **inputs,
                                "strength": 0.75,
                                "reference_image": Link("source", "end"),
                                "control_video": Link("source", "start"),
                            }.items()
                            if k != "start_image"
                        },
                    ),
                    "phantom": GraphNode(
                        "WanPhantomSubjectToVideo",
                        {
                            k: v
                            for k, v in {**inputs, "images": Link("source", "start")}.items()
                            if k != "start_image"
                        },
                    ),
                    "consumer": GraphNode(
                        "wan-image-fixture.consumer",
                        {
                            "image": Link("image", "positive"),
                            "first_last": Link("first_last", "positive"),
                            "inpaint": Link("inpaint", "positive"),
                            "control": Link("control", "positive"),
                            "control22": Link("control22", "positive"),
                            "vace": Link("vace", "positive"),
                            "phantom": Link("phantom", "positive"),
                            "phantom_negative": Link("phantom", "negative_img_text"),
                            "latent": Link("first_last", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve()
                    == "wan-image-first-last-inpaint"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
