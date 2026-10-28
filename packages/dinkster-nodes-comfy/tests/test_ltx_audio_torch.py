"""CPU tensor contracts for native LTXV audio VAE nodes."""

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
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_audio,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_audio import LTXVAudioVAEDecode, LTXVAudioVAEEncode

from dinkster.compose import PackSpec, ServingComposer

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class FirstStage:
    output_sample_rate = 24_000


class LTXAudioVAE:
    audio_sample_rate = 16_000

    def __init__(self) -> None:
        self.first_stage_model = FirstStage()
        self.encoded: Any = None
        self.decoded: Any = None

    def encode(self, waveform: Any) -> Any:
        self.encoded = waveform
        return waveform * 2.0 + 1.0

    def decode(self, latent: Any) -> Any:
        self.decoded = latent
        return latent.movedim(1, -1) * 3.0


@pytest.mark.parametrize(("source_rate", "vae_rate"), ((8_000, 16_000), (16_000, 8_000)))
def test_encode_delegates_to_audio_vae_with_pinned_resampling(
    source_rate: int, vae_rate: int
) -> None:
    import dinkster_inference.audio

    class RateVAE(LTXAudioVAE):
        audio_sample_rate = vae_rate

    waveform = torch.tensor([[[0.0, 1.0, -0.5]], [[0.5, -1.0, 0.25]]], dtype=torch.float64)
    before = waveform.clone()
    vae = RateVAE()
    output = LTXVAudioVAEEncode.execute(
        audio={"waveform": waveform, "sample_rate": source_rate}, audio_vae=vae
    )["LATENT"]
    expected = dinkster_inference.audio.resample(waveform, source_rate, vae_rate).movedim(1, -1)
    torch.testing.assert_close(vae.encoded, expected)
    torch.testing.assert_close(output["samples"], expected * 2.0 + 1.0)
    assert output == {"samples": output["samples"]}
    assert output["samples"].dtype is torch.float64
    assert output["samples"].device == waveform.device
    assert torch.equal(waveform, before)


def test_decode_uses_last_nested_latent_without_normalization() -> None:
    vae = LTXAudioVAE()
    first = torch.full((1, 2, 3, 4), 9.0, dtype=torch.float64)
    last = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5) / 10
    nested = torch.nested.nested_tensor([first, last])
    source = {"samples": nested, "extra": "source"}
    output = LTXVAudioVAEDecode.execute(audio_vae=vae, samples=source)["AUDIO"]
    expected = last * 3.0
    torch.testing.assert_close(vae.decoded, last)
    torch.testing.assert_close(output["waveform"], expected, rtol=0, atol=0)
    assert output["sample_rate"] == 24_000
    assert type(output["sample_rate"]) is int
    assert output["waveform"].dtype is torch.float64
    assert output["waveform"].device == last.device
    assert source["extra"] == "source"
    assert torch.equal(nested.unbind()[0], first)


class LTXAudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-audio-fixture.source",
            outputs=(
                OutputSpec("audio", _AUDIO),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            audio={
                "waveform": torch.tensor(
                    [[[0.0, 2.0, -1.0], [1.0, -2.0, 0.5]]], dtype=torch.float64
                ),
                "sample_rate": 16_000,
            },
            vae=LTXAudioVAE(),
        )


class LTXAudioInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-audio-fixture.inspect",
            inputs=(InputSpec("audio", _AUDIO),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, audio: Any) -> dict[str, object]:
        decoded = materialize_audio(audio)
        expected = torch.tensor([[[3.0, 9.0], [15.0, -9.0], [-3.0, 6.0]]], dtype=torch.float32)
        torch.testing.assert_close(decoded["waveform"], expected, rtol=0, atol=0)
        assert decoded["sample_rate"] == 24_000
        assert decoded["waveform"].dtype is torch.float32
        return cls.outputs(summary="ltx-audio")


@pytest.mark.parametrize("in_process", (True,))
def test_ltx_audio_nodes_execute_twice_through_both_production_hosts(
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
            (tmp_path / "ltx_audio_fixture.py").write_text(
                "from test_ltx_audio_torch import LTXAudioInspect, LTXAudioSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_media_io import register_media_types\n"
                "NODES = (*COMFY_NODES, LTXAudioSource, LTXAudioInspect)\n"
                "\ndef register_types(registry):\n"
                "    register_inference_types(registry)\n"
                "    register_media_types(registry)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["ltx-audio-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "ltx_audio_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "ltx_audio_fixture:register_types",
                ),
                encoding="ascii",
            )
            media_manifest = manifest.parent.parent / "dinkster-nodes-media-io/dinkster-pack.toml"
            await composer.add_pack(
                PackSpec(manifest=media_manifest, in_process=in_process, trust_reserved=True)
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-audio-fixture.source"),
                    "encode": GraphNode(
                        "LTXVAudioVAEEncode",
                        {"audio": Link("source", "audio"), "audio_vae": Link("source", "vae")},
                    ),
                    "decode": GraphNode(
                        "LTXVAudioVAEDecode",
                        {"samples": Link("encode", "LATENT"), "audio_vae": Link("source", "vae")},
                    ),
                    "inspect": GraphNode(
                        "ltx-audio-fixture.inspect", {"audio": Link("decode", "AUDIO")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "ltx-audio"
        finally:
            await composer.close()

    asyncio.run(scenario())
