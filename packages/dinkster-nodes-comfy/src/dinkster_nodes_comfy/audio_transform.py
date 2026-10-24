"""Native ComfyUI audio channel and volume nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_INT,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_audio,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_INT = TypeExpr.concrete(CORE_INT)


class SplitAudioChannels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplitAudioChannels",
            display_name="Split Audio Channels",
            category="audio",
            inputs=(InputSpec("audio", _AUDIO),),
            outputs=(OutputSpec("left", _AUDIO), OutputSpec("right", _AUDIO)),
        )

    @classmethod
    def execute(cls, *, audio: Any) -> Mapping[str, object]:
        if audio is None:
            return cls.outputs(left=None, right=None)
        audio = materialize_audio(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.shape[1] != 2:
            raise ValueError(
                "AudioSplit: Input audio must be stereo (2 channels), "
                f"got {waveform.shape[1]} channel(s)."
            )
        return cls.outputs(
            left={"waveform": waveform[..., 0:1, :], "sample_rate": sample_rate},
            right={"waveform": waveform[..., 1:2, :], "sample_rate": sample_rate},
        )


class AudioAdjustVolume(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioAdjustVolume",
            display_name="Adjust Audio Volume",
            category="audio",
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "volume",
                    _INT,
                    default=1,
                    widget=NumberWidget(min=-100, max=100),
                    doc=(
                        "Volume adjustment in decibels (dB). 0 = no change, "
                        "+6 = double, -6 = half, etc"
                    ),
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(cls, *, audio: Any, volume: int) -> Mapping[str, object]:
        if audio is None:
            return cls.outputs(AUDIO=None)
        if volume == 0:
            return cls.outputs(AUDIO=audio)
        audio = materialize_audio(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        gain = 10 ** (volume / 20)
        return cls.outputs(AUDIO={"waveform": waveform * gain, "sample_rate": sample_rate})


AUDIO_TRANSFORM_NODES = (SplitAudioChannels, AudioAdjustVolume)
