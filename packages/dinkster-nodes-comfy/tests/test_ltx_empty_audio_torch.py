"""CPU tensor contracts for the native empty LTXV audio latent node."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_nodes_comfy.ltx_audio import LTXVEmptyLatentAudio

from dinkster.compose import PackSpec, ServingComposer

_LATENT = TypeExpr.concrete(LATENT_TYPE)


class FirstStage:
    latent_frequency_bins = 5

    def __init__(self) -> None:
        self.calls: list[tuple[int, float | int]] = []

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: float | int) -> int:
        self.calls.append((frames_number, frame_rate))
        return frames_number + int(frame_rate)


class LTXAudioVAE:
    latent_channels = 3

    def __init__(self) -> None:
        self.first_stage_model = FirstStage()


@pytest.mark.parametrize("frame_rate", (25, 2.5))
def test_empty_audio_preserves_frame_rate_and_uses_default_dtype(frame_rate: float | int) -> None:
    vae = LTXAudioVAE()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        output = LTXVEmptyLatentAudio.execute(
            frames_number=7, frame_rate=frame_rate, batch_size=2, audio_vae=vae
        )["LATENT"]
    finally:
        torch.set_default_dtype(previous)
    samples = output["samples"]
    assert vae.first_stage_model.calls == [(7, frame_rate)]
    assert tuple(samples.shape) == (2, 3, 7 + int(frame_rate), 5)
    assert samples.dtype is torch.float64
    assert samples.device.type == "cpu"
    assert torch.count_nonzero(samples) == 0
    assert output["type"] == "audio"


def test_empty_audio_requires_a_vae() -> None:
    with pytest.raises(AssertionError, match="Audio VAE model is required"):
        LTXVEmptyLatentAudio.execute(frames_number=1, frame_rate=1.0, batch_size=1, audio_vae=None)


class LTXVAESource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-empty-audio-fixture.source",
            outputs=(OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(vae=LTXAudioVAE())


class LTXLatentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-empty-audio-fixture.inspect",
            inputs=(InputSpec("latent", _LATENT),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> dict[str, object]:
        value = materialize_latent(latent)
        samples = value["samples"]
        assert tuple(samples.shape) == (2, 3, 9, 5)
        assert samples.dtype is torch.float32
        assert samples.device.type == "cpu"
        assert torch.count_nonzero(samples) == 0
        assert value["type"] == "audio"
        return cls.outputs(summary="ltx-empty-audio")


@pytest.mark.parametrize("in_process", (True,))
def test_empty_audio_executes_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "ltx_empty_audio_fixture.py").write_text(
                "from test_ltx_empty_audio_torch import LTXLatentInspect, LTXVAESource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, LTXVAESource, LTXLatentInspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["ltx-empty-audio-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "ltx_empty_audio_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "ltx_empty_audio_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-empty-audio-fixture.source"),
                    "empty": GraphNode(
                        "LTXVEmptyLatentAudio",
                        {
                            "frames_number": 7,
                            "frame_rate": TypedLiteral("FLOAT", 2.5),
                            "batch_size": 2,
                            "audio_vae": Link("source", "vae"),
                        },
                    ),
                    "inspect": GraphNode(
                        "ltx-empty-audio-fixture.inspect", {"latent": Link("empty", "LATENT")}
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "ltx-empty-audio"
        finally:
            await composer.close()

    asyncio.run(scenario())
