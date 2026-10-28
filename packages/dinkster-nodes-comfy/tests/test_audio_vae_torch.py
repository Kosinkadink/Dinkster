"""CPU behavior for native ComfyUI audio VAE nodes."""

from __future__ import annotations

import asyncio
import math
import os
import statistics
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
from dinkster_nodes_comfy.audio_vae import VAEDecodeAudio, VAEDecodeAudioTiled, VAEEncodeAudio

from dinkster.compose import PackSpec, ServingComposer

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class AudioVAE:
    audio_sample_rate = 24000
    audio_sample_rate_output = 32000

    def __init__(self) -> None:
        self.encoded: Any = None
        self.decoded: Any = None
        self.tiled: tuple[Any, int, int, int | None] | None = None

    def encode(self, waveform: Any) -> Any:
        self.encoded = waveform
        return waveform + 2

    def decode(self, latent: Any) -> Any:
        self.decoded = latent
        return latent.movedim(1, -1)

    def decode_tiled(self, latent: Any, *, tile_x: int, tile_y: int, overlap: int | None) -> Any:
        self.tiled = (latent, tile_x, tile_y, overlap)
        return latent.movedim(1, -1)


class GraphAudioVAE(AudioVAE):
    def encode(self, waveform: Any) -> Any:
        return waveform.movedim(-1, 1)


def test_encode_preserves_layout_and_rejects_no_audio() -> None:
    vae = AudioVAE()
    waveform = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    source = {"waveform": waveform, "sample_rate": 24000, "extra": "source"}
    encoded = VAEEncodeAudio.execute(audio=source, vae=vae)["LATENT"]
    assert torch.equal(vae.encoded, waveform.movedim(1, -1))
    assert torch.equal(encoded["samples"], waveform.movedim(1, -1) + 2)
    assert encoded == {"samples": encoded["samples"]}
    assert source["waveform"] is waveform and source["extra"] == "source"
    with pytest.raises(ValueError, match="VAEEncodeAudio: input audio is None"):
        VAEEncodeAudio.execute(audio=None, vae=vae)


@pytest.mark.parametrize(("source_rate", "vae_rate"), ((8_000, 16_000), (16_000, 8_000)))
def test_encode_uses_pinned_resample_for_unequal_rates(source_rate: int, vae_rate: int) -> None:
    import dinkster_inference.audio

    class RateVAE(AudioVAE):
        audio_sample_rate = vae_rate

    waveform = torch.tensor([[[0.0, 1.0, -0.5]], [[0.5, -1.0, 0.25]]], dtype=torch.float64)
    before = waveform.clone()
    vae = RateVAE()
    result = VAEEncodeAudio.execute(
        audio={"waveform": waveform, "sample_rate": source_rate}, vae=vae
    )["LATENT"]
    expected = dinkster_inference.audio.resample(waveform, source_rate, vae_rate).movedim(1, -1)
    torch.testing.assert_close(result["samples"], expected + 2)
    assert result["samples"].dtype is torch.float64
    assert result["samples"].device == waveform.device
    assert torch.equal(waveform, before)


def test_decode_uses_last_nested_stream_normalizes_each_batch_and_selects_rate() -> None:
    vae = AudioVAE()
    first = torch.full((1, 2, 3), 9.0, dtype=torch.float64)
    last = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]],
        dtype=torch.float64,
    )
    expected = last.clone()
    expected[0] /= 5 * math.sqrt(3.5)
    nested = torch.nested.nested_tensor([first, last])
    samples = {"samples": nested, "sample_rate": 48000, "extra": "source"}
    output = VAEDecodeAudio.execute(samples=samples, vae=vae)["AUDIO"]
    torch.testing.assert_close(vae.decoded, expected)
    torch.testing.assert_close(output["waveform"], expected)
    torch.testing.assert_close(nested.unbind()[-1], expected)
    assert torch.equal(nested.unbind()[0], first)
    assert output["sample_rate"] == 48000
    assert output["waveform"].dtype == last.dtype and output["waveform"].device == last.device
    assert samples["extra"] == "source"


def test_decode_rate_falls_back_to_vae_output_rate() -> None:
    samples = {"samples": torch.ones(1, 2, 3)}
    output = VAEDecodeAudio.execute(samples=samples, vae=AudioVAE())["AUDIO"]
    assert output["sample_rate"] == 32000

    class InputRateVAE:
        audio_sample_rate = 22050

        def decode(self, latent: Any) -> Any:
            return latent.movedim(1, -1)

    output = VAEDecodeAudio.execute(samples=samples, vae=InputRateVAE())["AUDIO"]
    assert output["sample_rate"] == 22050

    class DefaultRateVAE:
        def decode(self, latent: Any) -> Any:
            return latent.movedim(1, -1)

    output = VAEDecodeAudio.execute(samples=samples, vae=DefaultRateVAE())["AUDIO"]
    assert output["sample_rate"] == 44100


def test_tiled_decode_delegates_exact_arguments() -> None:
    vae = AudioVAE()
    samples = {"samples": torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)}
    output = VAEDecodeAudioTiled.execute(samples=samples, vae=vae, tile_size=520, overlap=72)[
        "AUDIO"
    ]
    assert vae.tiled is not None
    latent, tile_x, tile_y, overlap = vae.tiled
    assert torch.equal(latent, samples["samples"])
    assert (tile_x, tile_y, overlap) == (520, 520, 72)
    assert output["sample_rate"] == 32000


class AudioVAESource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-vae-fixture.source",
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
                    [
                        [[0.0, 8.0, -4.0], [4.0, -8.0, 2.0]],
                        [[0.0, 0.01, -0.005], [0.005, -0.01, 0.0025]],
                    ],
                    dtype=torch.float64,
                ),
                "sample_rate": 24000,
            },
            vae=GraphAudioVAE(),
        )


class AudioVAEInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-vae-fixture.inspect",
            inputs=(InputSpec("ordinary", _AUDIO), InputSpec("tiled", _AUDIO)),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, ordinary: Any, tiled: Any) -> dict[str, object]:
        expected = torch.tensor(
            [[[0.0, 8.0, -4.0], [4.0, -8.0, 2.0]], [[0.0, 0.01, -0.005], [0.005, -0.01, 0.0025]]],
            dtype=torch.float32,
        )
        expected[0] /= 5 * statistics.stdev([0.0, 8.0, -4.0, 4.0, -8.0, 2.0])
        for output in (ordinary, tiled):
            audio = materialize_audio(output)
            assert set(audio) == {"waveform", "sample_rate"}
            assert audio["sample_rate"] == 32000
            assert audio["waveform"].dtype is torch.float32
            assert audio["waveform"].device.type == "cpu"
            assert tuple(audio["waveform"].shape) == (2, 2, 3)
            torch.testing.assert_close(audio["waveform"], expected)
        return cls.outputs(summary="audio-vae")


@pytest.mark.parametrize("in_process", (True,))
def test_audio_vae_nodes_execute_twice_through_both_production_hosts(
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
            (tmp_path / "audio_vae_fixture.py").write_text(
                "from test_audio_vae_torch import AudioVAEInspect, AudioVAESource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_media_io import register_media_types\n"
                "NODES = (*COMFY_NODES, AudioVAESource, AudioVAEInspect)\n"
                "\ndef register_types(registry):\n"
                "    register_inference_types(registry)\n"
                "    register_media_types(registry)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-vae-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_vae_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "audio_vae_fixture:register_types",
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
                    "source": GraphNode("audio-vae-fixture.source"),
                    "encode": GraphNode(
                        "VAEEncodeAudio",
                        {"audio": Link("source", "audio"), "vae": Link("source", "vae")},
                    ),
                    "decode": GraphNode(
                        "VAEDecodeAudio",
                        {"samples": Link("encode", "LATENT"), "vae": Link("source", "vae")},
                    ),
                    "tiled": GraphNode(
                        "VAEDecodeAudioTiled",
                        {
                            "samples": Link("encode", "LATENT"),
                            "vae": Link("source", "vae"),
                            "tile_size": 512,
                            "overlap": 64,
                        },
                    ),
                    "inspect": GraphNode(
                        "audio-vae-fixture.inspect",
                        {"ordinary": Link("decode", "AUDIO"), "tiled": Link("tiled", "AUDIO")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "audio-vae"
        finally:
            await composer.close()

    asyncio.run(scenario())
