"""CPU duration boundaries and ordinary-host audio generator execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_audio,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_generators import EmptyAudio, EmptyMiniMaxMusic3LatentAudio

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("duration", "rate", "channels", "length"),
    ((0.0, 11025, 1, 0), (0.3125, 8, 2, 2), (0.4375, 8, 1, 4), (0.01, 11025, 2, 110)),
)
def test_empty_audio_sample_rounding(
    duration: float, rate: int, channels: int, length: int
) -> None:
    audio = EmptyAudio.execute(duration=duration, sample_rate=rate, channels=channels)["AUDIO"]
    assert set(audio) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == rate
    waveform = audio["waveform"]
    assert waveform.shape == (1, channels, length)
    assert waveform.dtype is torch.float32
    assert waveform.device.type == "cpu"
    assert not waveform.count_nonzero()


@pytest.mark.parametrize(
    ("seconds", "batch", "length"),
    ((0.0, 3, 3), (0.1, 2, 6), (0.14, 2, 13), (360.04, 1, 31007)),
)
def test_minimax_frames_round_clamp_then_convert(seconds: float, batch: int, length: int) -> None:
    latent = EmptyMiniMaxMusic3LatentAudio.execute(seconds=seconds, batch_size=batch)["LATENT"]
    assert set(latent) == {"samples", "type", "downscale_ratio_temporal"}
    assert latent["type"] == "audio"
    assert latent["downscale_ratio_temporal"] == 512
    samples = latent["samples"]
    assert samples.shape == (batch, 128, length)
    assert samples.dtype is torch.float32
    assert samples.device.type == "cpu"
    assert not samples.count_nonzero()


class AudioGeneratorInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-generator-fixture.inspect",
            inputs=(
                InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, audio: Any, latent: Any) -> dict[str, object]:
        audio, latent = materialize_audio(audio), materialize_latent(latent)
        assert set(audio) == {"waveform", "sample_rate"}
        assert audio["sample_rate"] == 8
        assert audio["waveform"].shape == (1, 2, 2)
        assert set(latent) == {"samples", "type", "downscale_ratio_temporal"}
        assert latent["type"] == "audio" and latent["downscale_ratio_temporal"] == 512
        assert latent["samples"].shape == (2, 128, 13)
        for tensor in (audio["waveform"], latent["samples"]):
            assert tensor.dtype is torch.float32 and tensor.device.type == "cpu"
            assert not tensor.count_nonzero()
        return cls.outputs(result="audio generators exact")


@pytest.mark.parametrize("in_process", (True,))
def test_audio_generators_feed_consumer_twice_in_both_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        manifest = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(package / "src"))),
            }
        )
        try:
            (tmp_path / "audio_generator_fixture.py").write_text(
                "from test_audio_generators_torch import AudioGeneratorInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_media_io import register_media_types\n"
                "NODES = (*COMFY_NODES, AudioGeneratorInspect)\n"
                "def register_types(registry):\n"
                "    register_inference_types(registry)\n"
                "    register_media_types(registry)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-generator-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_generator_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "audio_generator_fixture:register_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(
                PackSpec(
                    manifest=package.parent / "dinkster-nodes-media-io/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "audio": GraphNode(
                        "EmptyAudio", {"duration": 0.3125, "sample_rate": 8, "channels": 2}
                    ),
                    "latent": GraphNode(
                        "EmptyMiniMaxMusic3LatentAudio", {"seconds": 0.14, "batch_size": 2}
                    ),
                    "inspect": GraphNode(
                        "audio-generator-fixture.inspect",
                        {"audio": Link("audio", "AUDIO"), "latent": Link("latent", "LATENT")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "audio generators exact"
        finally:
            await composer.close()

    asyncio.run(scenario())
