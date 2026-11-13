"""CPU duration boundaries and ordinary-host audio generator execution."""

from __future__ import annotations

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
from dinkster_audio.audio_generators import EmptyAudio
from dinkster_minimax.audio_generators import EmptyMiniMaxMusic3LatentAudio


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
