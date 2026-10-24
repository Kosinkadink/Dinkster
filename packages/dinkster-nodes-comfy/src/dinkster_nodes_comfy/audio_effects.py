"""Native ComfyUI audio duration trimming and equalization nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)


class TrimAudioDuration(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TrimAudioDuration",
            display_name="Trim Audio Duration",
            description="Trim audio tensor into chosen time range.",
            category="audio",
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "start_index",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(step=0.01),
                    doc=(
                        "Start time in seconds, can be negative to count from the end "
                        "(supports sub-seconds)."
                    ),
                ),
                InputSpec(
                    "duration",
                    _FLOAT,
                    default=60.0,
                    widget=NumberWidget(min=0.0, step=0.01),
                    doc="Duration in seconds",
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(cls, *, audio: Any, start_index: float, duration: float) -> Mapping[str, object]:
        if audio is None:
            return cls.outputs(AUDIO=None)
        initialize_inference_process()
        audio = materialize_audio(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        audio_length = waveform.shape[-1]
        if audio_length == 0:
            return cls.outputs(AUDIO=audio)
        if start_index < 0:
            start_frame = audio_length + int(round(start_index * sample_rate))
        else:
            start_frame = int(round(start_index * sample_rate))
        start_frame = max(0, min(start_frame, audio_length))
        end_frame = start_frame + int(round(duration * sample_rate))
        end_frame = max(0, min(end_frame, audio_length))
        if start_frame >= end_frame:
            raise ValueError(
                "TrimAudioDuration: Start time must be less than end time and be within "
                "the audio length."
            )
        return cls.outputs(
            AUDIO={"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}
        )


class AudioEqualizer3Band(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioEqualizer3Band",
            display_name="Audio Equalizer (3-Band)",
            category="audio",
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "low_gain_dB",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                    doc="Gain for Low frequencies (Bass)",
                ),
                InputSpec(
                    "low_freq",
                    _INT,
                    default=100,
                    widget=NumberWidget(min=20, max=500),
                    doc="Cutoff frequency for Low shelf",
                ),
                InputSpec(
                    "mid_gain_dB",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                    doc="Gain for Mid frequencies",
                ),
                InputSpec(
                    "mid_freq",
                    _INT,
                    default=1000,
                    widget=NumberWidget(min=200, max=4000),
                    doc="Center frequency for Mids",
                ),
                InputSpec(
                    "mid_q",
                    _FLOAT,
                    default=0.707,
                    widget=NumberWidget(min=0.1, max=10.0, step=0.1),
                    doc="Q factor (bandwidth) for Mids",
                ),
                InputSpec(
                    "high_gain_dB",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                    doc="Gain for High frequencies (Treble)",
                ),
                InputSpec(
                    "high_freq",
                    _INT,
                    default=5000,
                    widget=NumberWidget(min=1000, max=15000),
                    doc="Cutoff frequency for High shelf",
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: Any,
        low_gain_dB: float,
        low_freq: int,
        mid_gain_dB: float,
        mid_freq: int,
        mid_q: float,
        high_gain_dB: float,
        high_freq: int,
    ) -> Mapping[str, object]:
        if audio is None:
            return cls.outputs(AUDIO=None)
        initialize_inference_process()
        audio = materialize_audio(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.shape[-1] == 0:
            return cls.outputs(AUDIO=audio)
        equalized = waveform.clone()
        filters = importlib.import_module("dinkster_inference.audio")
        if low_gain_dB != 0:
            equalized = filters.bass_biquad(
                equalized, sample_rate, gain=low_gain_dB, central_freq=float(low_freq), Q=0.707
            )
        if mid_gain_dB != 0:
            equalized = filters.equalizer_biquad(
                equalized, sample_rate, center_freq=float(mid_freq), gain=mid_gain_dB, Q=mid_q
            )
        if high_gain_dB != 0:
            equalized = filters.treble_biquad(
                equalized, sample_rate, gain=high_gain_dB, central_freq=float(high_freq), Q=0.707
            )
        return cls.outputs(AUDIO={"waveform": equalized, "sample_rate": sample_rate})


AUDIO_EFFECT_NODES = (TrimAudioDuration, AudioEqualizer3Band)
