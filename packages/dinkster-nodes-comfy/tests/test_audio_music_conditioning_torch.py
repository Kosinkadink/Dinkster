"""CPU conditioning values and ordinary production-host transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
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
from dinkster_nodes_comfy.audio_music_conditioning import (
    MiniMaxMusic3TextEncode,
    ReferenceTimbreAudio,
)

from dinkster.compose import PackSpec, ServingComposer


class MusicEncoder:
    def tokenize(self, caption: str, **options: object) -> dict[str, object]:
        return {"caption": caption, **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        return [
            [torch.arange(42, dtype=torch.float64).reshape(2, 7, 3) / 8, dict(tokens)],
            [
                torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) - 3,
                {**tokens, "end_percent": 0.75},
            ],
        ]


@pytest.mark.parametrize(("duration", "frames"), ((0.0, 1), (0.1, 2), (0.14, 4), (361.0, 9000)))
def test_music_frame_rounding_options_scales_and_actual_duration(
    duration: float, frames: int
) -> None:
    result = MiniMaxMusic3TextEncode.execute(
        clip=MusicEncoder(),
        caption="quiet brass",
        lyrics="one\ntwo",
        seed=37,
        max_duration=duration,
        cfg_scale=2.75,
        top_k=17,
    )
    assert result["seconds"] == 0.28
    for index, (hidden, metadata) in enumerate(result["CONDITIONING"]):
        assert {
            k: metadata[k]
            for k in ("caption", "lyrics", "seed", "max_audio_frames", "cfg_scale", "top_k")
        } == {
            "caption": "quiet brass",
            "lyrics": "one\ntwo",
            "seed": 37,
            "max_audio_frames": frames,
            "cfg_scale": 2.75,
            "top_k": 17,
        }
        expected = (
            torch.arange(42, dtype=torch.float64).reshape(2, 7, 3) / 8
            if index == 0
            else torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) - 3
        )
        assert torch.equal(hidden, expected)
        scale = metadata["conditioning_scale"]
        assert scale.shape == (hidden.shape[0], 1, 1)
        assert scale.dtype == hidden.dtype and scale.device == hidden.device
        assert torch.equal(scale, torch.ones_like(scale))


def test_reference_absence_identity_append_and_metadata_copy() -> None:
    opaque = object()
    assert ReferenceTimbreAudio.execute(conditioning=opaque)["CONDITIONING"] is opaque  # type: ignore[arg-type]
    hidden = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    old = torch.tensor([-7.0])
    source = [
        [hidden, {"reference_audio_timbre_latents": [old], "tag": "kept"}],
        [hidden + 1, {"reference_audio_timbre_latents": None}],
    ]
    latent = torch.arange(10, dtype=torch.float64).reshape(1, 2, 5)
    original = latent.clone()
    result = ReferenceTimbreAudio.execute(conditioning=source, latent={"samples": latent})[
        "CONDITIONING"
    ]
    assert result is not source and result[0][1] is not source[0][1]
    assert result[0][0] is hidden and result[0][1]["tag"] == "kept"
    assert len(result[0][1]["reference_audio_timbre_latents"]) == 2
    assert result[0][1]["reference_audio_timbre_latents"][0] is old
    for entry in result:
        assert entry[1]["reference_audio_timbre_latents"][-1] is latent
    assert len(source[0][1]["reference_audio_timbre_latents"]) == 1
    assert source[1][1]["reference_audio_timbre_latents"] is None
    assert torch.equal(latent, original) and latent.dtype is torch.float64


class MusicSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="music-fixture.source",
            outputs=(
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            clip=MusicEncoder(),
            latent={"samples": torch.arange(10, dtype=torch.float64).reshape(1, 2, 5)},
        )


class MusicInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="music-fixture.inspect",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("seconds", TypeExpr.concrete(CORE_FLOAT)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any, seconds: float) -> dict[str, object]:
        assert seconds == 0.28
        for index, (hidden, meta) in enumerate(materialize_latent(conditioning)):
            expected = (
                torch.arange(42, dtype=torch.float64).reshape(2, 7, 3) / 8
                if index == 0
                else torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) - 3
            )
            assert torch.equal(hidden, expected) and hidden.dtype == expected.dtype
            assert hidden.device.type == "cpu"
            assert meta["caption"] == "quiet brass" and meta["lyrics"] == "one\ntwo"
            assert (meta["seed"], meta["max_audio_frames"], meta["cfg_scale"], meta["top_k"]) == (
                37,
                4,
                2.75,
                17,
            )
            scale = meta["conditioning_scale"]
            assert scale.shape == (hidden.shape[0], 1, 1) and scale.dtype == hidden.dtype
            assert scale.device == hidden.device and torch.equal(scale, torch.ones_like(scale))
            references = meta["reference_audio_timbre_latents"]
            assert len(references) == 2
            for tensor in references:
                assert tensor.dtype is torch.float64 and tensor.device.type == "cpu"
                assert torch.equal(tensor, torch.arange(10, dtype=torch.float64).reshape(1, 2, 5))
        return cls.outputs(result="music conditioning exact")


@pytest.mark.parametrize("in_process", (True,))
def test_music_conditioning_chain_twice_in_both_hosts(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "music_fixture.py").write_text(
                "from test_audio_music_conditioning_torch import MusicSource, MusicInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, MusicSource, MusicInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (package / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["music-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "music_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "music_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("music-fixture.source"),
                    "encode": GraphNode(
                        "MiniMaxMusic3TextEncode",
                        {
                            "clip": Link("source", "clip"),
                            "caption": "quiet brass",
                            "lyrics": "one\ntwo",
                            "seed": 37,
                            "max_duration": 0.14,
                            "cfg_scale": 2.75,
                            "top_k": 17,
                        },
                    ),
                    "ref": GraphNode(
                        "ReferenceTimbreAudio",
                        {
                            "conditioning": Link("encode", "CONDITIONING"),
                            "latent": Link("source", "latent"),
                        },
                    ),
                    "ref2": GraphNode(
                        "ReferenceTimbreAudio",
                        {
                            "conditioning": Link("ref", "CONDITIONING"),
                            "latent": Link("source", "latent"),
                        },
                    ),
                    "absent": GraphNode(
                        "ReferenceTimbreAudio", {"conditioning": Link("ref2", "CONDITIONING")}
                    ),
                    "inspect": GraphNode(
                        "music-fixture.inspect",
                        {
                            "conditioning": Link("absent", "CONDITIONING"),
                            "seconds": Link("encode", "seconds"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "music conditioning exact"
        finally:
            await composer.close()

    asyncio.run(scenario())
