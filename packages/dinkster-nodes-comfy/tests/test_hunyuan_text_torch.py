"""Native Hunyuan scheduled text conditioning through both manifest hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hunyuan_text import (
    CLIPTextEncodeHunyuanDiT,
    TextEncodeHunyuanVideo_ImageToVideo,
)

from dinkster.compose import PackSpec, ServingComposer


class HunyuanClip:
    def __init__(self) -> None:
        self.hidden = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3) / 7
        self.metadata = {"scheduled": (0.25, 0.75)}

    def tokenize(self, text: str, **options: Any) -> dict[str, Any]:
        return {"bert": text + ":bert", "mt5xl": text + ":mt5xl", **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        return [[self.hidden, {**self.metadata, **tokens}]]


def test_hunyuan_image_uses_independent_prompt_streams() -> None:
    clip = HunyuanClip()
    result = CLIPTextEncodeHunyuanDiT.execute(clip=clip, bert="red sail", mt5xl="blue cloud")[
        "CONDITIONING"
    ]
    hidden, meta = result[0]
    assert hidden is clip.hidden and hidden.dtype is torch.float64
    assert meta == {"scheduled": (0.25, 0.75), "bert": "red sail:bert", "mt5xl": "blue cloud:mt5xl"}
    assert clip.metadata == {"scheduled": (0.25, 0.75)}


@pytest.mark.parametrize("interleave", (1, 7, 512))
def test_hunyuan_video_passes_projected_vision_and_template(interleave: int) -> None:
    clip = HunyuanClip()
    projected = torch.arange(10, dtype=torch.float64).reshape(1, 2, 5) - 3
    before = projected.clone()
    result = TextEncodeHunyuanVideo_ImageToVideo.execute(
        clip=clip,
        clip_vision_output=SimpleNamespace(mm_projected=projected),
        prompt="turn left",
        image_interleave=interleave,
    )["CONDITIONING"]
    hidden, meta = result[0]
    assert hidden is clip.hidden and hidden.device.type == "cpu"
    assert meta["bert"] == "turn left:bert" and meta["mt5xl"] == "turn left:mt5xl"
    assert meta["image_interleave"] == interleave
    torch.testing.assert_close(meta["image_embeds"], before, rtol=0, atol=0)
    assert meta["image_embeds"].dtype is torch.float64
    assert (
        meta["llama_template"]
        .format("turn left")
        .endswith(
            "<|start_header_id|>user<|end_header_id|>\n\nturn left<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    )
    assert "<image>\nDescribe the video" in meta["llama_template"]
    assert torch.equal(projected, before) and clip.metadata == {"scheduled": (0.25, 0.75)}


class HunyuanTextSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-text-fixture.source",
            outputs=(
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("vision", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            clip=HunyuanClip(),
            vision=SimpleNamespace(
                mm_projected=torch.arange(10, dtype=torch.float64).reshape(1, 2, 5) - 3
            ),
        )


class HunyuanTextInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-text-fixture.inspect",
            inputs=(
                InputSpec("image", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("video", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any, video: Any) -> dict[str, object]:
        for conditioning in (image, video):
            hidden, meta = materialize_latent(conditioning)[0]
            torch.testing.assert_close(
                hidden, torch.arange(12, dtype=torch.float64).reshape(1, 4, 3) / 7, rtol=0, atol=0
            )
            assert hidden.dtype is torch.float64 and hidden.device.type == "cpu"
            assert tuple(meta["scheduled"]) == (0.25, 0.75)
        _, image_meta = materialize_latent(image)[0]
        assert (image_meta["bert"], image_meta["mt5xl"]) == ("red sail:bert", "blue cloud:mt5xl")
        _, video_meta = materialize_latent(video)[0]
        assert video_meta["bert"] == "turn left:bert" and video_meta["image_interleave"] == 7
        torch.testing.assert_close(
            video_meta["image_embeds"],
            torch.arange(10, dtype=torch.float64).reshape(1, 2, 5) - 3,
            rtol=0,
            atol=0,
        )
        assert video_meta["image_embeds"].dtype is torch.float64
        assert "<image>\nDescribe the video" in video_meta["llama_template"]
        return cls.outputs(summary="hunyuan text exact")


@pytest.mark.parametrize("in_process", (True, False))
def test_hunyuan_text_both_production_hosts_twice(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "hunyuan_text_fixture.py").write_text(
                "from test_hunyuan_text_torch import HunyuanTextSource, HunyuanTextInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, HunyuanTextSource, HunyuanTextInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (package / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hunyuan-text-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hunyuan_text_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hunyuan_text_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hunyuan-text-fixture.source"),
                    "image": GraphNode(
                        "CLIPTextEncodeHunyuanDiT",
                        {"clip": Link("source", "clip"), "bert": "red sail", "mt5xl": "blue cloud"},
                    ),
                    "video": GraphNode(
                        "TextEncodeHunyuanVideo_ImageToVideo",
                        {
                            "clip": Link("source", "clip"),
                            "clip_vision_output": Link("source", "vision"),
                            "prompt": "turn left",
                            "image_interleave": 7,
                        },
                    ),
                    "inspect": GraphNode(
                        "hunyuan-text-fixture.inspect",
                        {
                            "image": Link("image", "CONDITIONING"),
                            "video": Link("video", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "hunyuan text exact"
        finally:
            await composer.close()

    asyncio.run(scenario())
