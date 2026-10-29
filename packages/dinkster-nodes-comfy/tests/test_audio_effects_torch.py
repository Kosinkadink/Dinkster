"""CPU behavior for native ComfyUI audio effect nodes."""

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
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_audio,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_effects import AudioEqualizer3Band, TrimAudioDuration

from dinkster.compose import PackSpec, ServingComposer

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


def _audio(values: list[list[list[float]]], rate: int = 8) -> dict[str, object]:
    return {
        "waveform": torch.tensor(values, dtype=torch.float64),
        "sample_rate": rate,
        "extra": object(),
    }


def _equalize(audio: object, **gains: object) -> object:
    return AudioEqualizer3Band.execute(
        audio=audio,
        low_gain_dB=gains.get("low_gain_dB", 0.0),
        low_freq=100,
        mid_gain_dB=gains.get("mid_gain_dB", 0.0),
        mid_freq=2,
        mid_q=0.9,
        high_gain_dB=gains.get("high_gain_dB", 0.0),
        high_freq=3,
    )["AUDIO"]


def test_trim_handles_null_rounding_negative_starts_clamps_empty_and_invalid_ranges() -> None:
    assert TrimAudioDuration.execute(audio=None, start_index=0.0, duration=1.0) == {"AUDIO": None}
    audio = _audio([[[0, 1, 2, 3, 4, 5]]])
    original = audio["waveform"].clone()
    positive = TrimAudioDuration.execute(audio=audio, start_index=0.1875, duration=0.3125)["AUDIO"]
    assert torch.equal(positive["waveform"], torch.tensor([[[2.0, 3.0]]], dtype=torch.float64))
    negative = TrimAudioDuration.execute(audio=audio, start_index=-0.375, duration=0.25)["AUDIO"]
    assert torch.equal(negative["waveform"], torch.tensor([[[3.0, 4.0]]], dtype=torch.float64))
    clamped = TrimAudioDuration.execute(audio=audio, start_index=-2.0, duration=2.0)["AUDIO"]
    assert torch.equal(clamped["waveform"], original)
    empty = _audio([[[]]])
    assert TrimAudioDuration.execute(audio=empty, start_index=0.0, duration=1.0)["AUDIO"] is empty
    with pytest.raises(ValueError, match="TrimAudioDuration: Start time must be less"):
        TrimAudioDuration.execute(audio=audio, start_index=1.0, duration=0.0)
    assert torch.equal(audio["waveform"], original)


def test_equalizer_uses_pinned_dsp_in_source_order_and_preserves_source() -> None:
    audio = _audio(
        [
            [[0.1, -0.2, 0.3, -0.4], [0.4, -0.3, 0.2, -0.1]],
            [[-0.2, 0.1, -0.4, 0.3], [0.3, -0.4, 0.1, -0.2]],
        ]
    )
    original = audio["waveform"].clone()
    assert AudioEqualizer3Band.execute(
        audio=None,
        low_gain_dB=1.0,
        low_freq=100,
        mid_gain_dB=0.0,
        mid_freq=2,
        mid_q=0.9,
        high_gain_dB=0.0,
        high_freq=3,
    ) == {"AUDIO": None}
    zero = _equalize(audio)
    assert zero["waveform"] is not audio["waveform"]
    assert torch.equal(zero["waveform"], original)
    import dinkster_inference.audio as filters

    expected = filters.bass_biquad(original.clone(), 8, gain=3.0, central_freq=100.0, Q=0.707)
    expected = filters.equalizer_biquad(expected, 8, center_freq=2.0, gain=-2.0, Q=0.9)
    expected = filters.treble_biquad(expected, 8, gain=1.5, central_freq=3.0, Q=0.707)
    actual = _equalize(audio, low_gain_dB=3.0, mid_gain_dB=-2.0, high_gain_dB=1.5)
    assert set(actual) == {"waveform", "sample_rate"}
    assert actual["sample_rate"] == 8
    assert actual["waveform"].dtype is torch.float64
    assert actual["waveform"].device.type == "cpu"
    assert torch.equal(actual["waveform"], expected)
    assert torch.equal(audio["waveform"], original)
    empty = _audio([[[]]])
    assert _equalize(empty) is empty


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-effects-fixture.source", outputs=(OutputSpec("audio", _AUDIO),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            audio={"waveform": torch.tensor([[[0.1, -0.2, 0.3, -0.4]]]), "sample_rate": 8}
        )


class AudioInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-effects-fixture.inspect",
            inputs=(InputSpec("audio", _AUDIO),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, audio: Any) -> dict[str, object]:
        audio = materialize_audio(audio)
        expected = torch.tensor([[[0.1, -0.2]]])
        expected = __import__("dinkster_inference.audio", fromlist=["bass_biquad"]).bass_biquad(
            expected, 8, gain=3.0, central_freq=100.0, Q=0.707
        )
        assert set(audio) == {"waveform", "sample_rate"}
        assert audio["sample_rate"] == 8
        assert audio["waveform"].dtype is torch.float32
        assert audio["waveform"].device.type == "cpu"
        assert torch.equal(audio["waveform"], expected)
        return cls.outputs(summary="audio-effects")


@pytest.mark.parametrize("in_process", (True,))
def test_audio_effect_nodes_execute_twice_in_both_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        media_source_root = source_root.parent.parent / "dinkster-nodes-media-io" / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(source_root), str(media_source_root))
                ),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "audio_effects_fixture.py").write_text(
                "from test_audio_effects_torch import AudioInspect, AudioSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "from dinkster_nodes_media_io import register_media_types\n"
                "NODES = (*COMFY_NODES, AudioSource, AudioInspect)\n"
                "\ndef register_types(registry):\n"
                "    register_inference_types(registry)\n"
                "    register_media_types(registry)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-effects-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_effects_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "audio_effects_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "audio_effects_fixture:register_types",
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
                    "source": GraphNode("audio-effects-fixture.source"),
                    "trim": GraphNode(
                        "TrimAudioDuration",
                        {"audio": Link("source", "audio"), "start_index": 0.0, "duration": 0.25},
                    ),
                    "equalizer": GraphNode(
                        "AudioEqualizer3Band",
                        {
                            "audio": Link("trim", "AUDIO"),
                            "low_gain_dB": 3.0,
                            "low_freq": 100,
                            "mid_gain_dB": 0.0,
                            "mid_freq": 1000,
                            "mid_q": 0.707,
                            "high_gain_dB": 0.0,
                            "high_freq": 5000,
                        },
                    ),
                    "inspect": GraphNode(
                        "audio-effects-fixture.inspect", {"audio": Link("equalizer", "AUDIO")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                assert (await engine.run(graph, ["inspect"])).outputs["inspect"][
                    "summary"
                ].resolve() == "audio-effects"
        finally:
            await composer.close()

    asyncio.run(scenario())
