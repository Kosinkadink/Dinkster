"""Native ComfyUI audio channel joining, concatenation, and merging."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_COMBO,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)


def _match_audio_sample_rates(
    waveform_1: Any, sample_rate_1: int, waveform_2: Any, sample_rate_2: int
) -> tuple[Any, Any, int]:
    if sample_rate_1 > sample_rate_2:
        waveform_2 = importlib.import_module("dinkster_inference.audio").resample(
            waveform_2, sample_rate_2, sample_rate_1
        )
        return waveform_1, waveform_2, sample_rate_1
    if sample_rate_2 > sample_rate_1:
        waveform_1 = importlib.import_module("dinkster_inference.audio").resample(
            waveform_1, sample_rate_1, sample_rate_2
        )
        return waveform_1, waveform_2, sample_rate_2
    return waveform_1, waveform_2, sample_rate_1


def _audio_pair(audio1: object, audio2: object) -> tuple[Any, int, Any, int]:
    initialize_inference_process()
    first = materialize_audio(audio1)
    second = materialize_audio(audio2)
    return (
        first["waveform"],
        cast("int", first["sample_rate"]),
        second["waveform"],
        cast("int", second["sample_rate"]),
    )


class JoinAudioChannels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="JoinAudioChannels",
            display_name="Join Audio Channels",
            description="Joins left and right mono audio channels into a stereo audio.",
            category="audio",
            inputs=(InputSpec("audio_left", _AUDIO), InputSpec("audio_right", _AUDIO)),
            outputs=(OutputSpec("audio", _AUDIO),),
        )

    @classmethod
    def execute(
        cls, *, audio_left: object | None, audio_right: object | None
    ) -> Mapping[str, object]:
        if audio_left is None and audio_right is None:
            return cls.outputs(audio=None)
        if audio_left is None:
            return cls.outputs(audio=audio_right)
        if audio_right is None:
            return cls.outputs(audio=audio_left)
        waveform_left, sample_rate_left, waveform_right, sample_rate_right = _audio_pair(
            audio_left, audio_right
        )
        if waveform_left.shape[1] != 1 or waveform_right.shape[1] != 1:
            raise ValueError("AudioJoin: Both input audios must be mono.")
        waveform_left, waveform_right, sample_rate = _match_audio_sample_rates(
            waveform_left, sample_rate_left, waveform_right, sample_rate_right
        )
        length = min(waveform_left.shape[-1], waveform_right.shape[-1])
        torch = importlib.import_module("torch")
        return cls.outputs(
            audio={
                "waveform": torch.cat(
                    (waveform_left[..., 0:1, :length], waveform_right[..., 0:1, :length]), dim=1
                ),
                "sample_rate": sample_rate,
            }
        )


class AudioConcat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioConcat",
            display_name="Concatenate Audio",
            description="Concatenates the audio1 to audio2 in the specified direction.",
            category="audio",
            inputs=(
                InputSpec("audio1", _AUDIO),
                InputSpec("audio2", _AUDIO),
                InputSpec(
                    "direction",
                    _COMBO,
                    default="after",
                    widget=ComboWidget((ComboOption("after"), ComboOption("before"))),
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(
        cls, *, audio1: object | None, audio2: object | None, direction: str
    ) -> Mapping[str, object]:
        if audio1 is None and audio2 is None:
            return cls.outputs(AUDIO=None)
        if audio1 is None:
            return cls.outputs(AUDIO=audio2)
        if audio2 is None:
            return cls.outputs(AUDIO=audio1)
        waveform_1, sample_rate_1, waveform_2, sample_rate_2 = _audio_pair(audio1, audio2)
        if waveform_1.shape[1] == 1:
            waveform_1 = waveform_1.repeat(1, 2, 1)
        if waveform_2.shape[1] == 1:
            waveform_2 = waveform_2.repeat(1, 2, 1)
        waveform_1, waveform_2, sample_rate = _match_audio_sample_rates(
            waveform_1, sample_rate_1, waveform_2, sample_rate_2
        )
        torch = importlib.import_module("torch")
        waveforms = (waveform_1, waveform_2) if direction == "after" else (waveform_2, waveform_1)
        return cls.outputs(
            AUDIO={"waveform": torch.cat(waveforms, dim=2), "sample_rate": sample_rate}
        )


class AudioMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioMerge",
            display_name="Merge Audio",
            description="Combine two audio tracks by overlaying their waveforms.",
            category="audio",
            inputs=(
                InputSpec("audio1", _AUDIO),
                InputSpec("audio2", _AUDIO),
                InputSpec(
                    "merge_method",
                    _COMBO,
                    widget=ComboWidget(
                        tuple(
                            ComboOption(method)
                            for method in ("add", "mean", "subtract", "multiply")
                        )
                    ),
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(
        cls, *, audio1: object | None, audio2: object | None, merge_method: str
    ) -> Mapping[str, object]:
        if audio1 is None and audio2 is None:
            return cls.outputs(AUDIO=None)
        if audio1 is None:
            return cls.outputs(AUDIO=audio2)
        if audio2 is None:
            return cls.outputs(AUDIO=audio1)
        waveform_1, sample_rate_1, waveform_2, sample_rate_2 = _audio_pair(audio1, audio2)
        waveform_1, waveform_2, sample_rate = _match_audio_sample_rates(
            waveform_1, sample_rate_1, waveform_2, sample_rate_2
        )
        length_1 = waveform_1.shape[-1]
        if length_1 == 0 or waveform_2.shape[-1] == 0:
            return cls.outputs(AUDIO={"waveform": waveform_1, "sample_rate": sample_rate})
        if waveform_2.shape[-1] > length_1:
            waveform_2 = waveform_2[..., :length_1]
        elif waveform_2.shape[-1] < length_1:
            torch = importlib.import_module("torch")
            waveform_2 = torch.cat(
                (
                    waveform_2,
                    torch.zeros(
                        (*waveform_2.shape[:-1], length_1 - waveform_2.shape[-1]),
                        dtype=waveform_2.dtype,
                        device=waveform_2.device,
                    ),
                ),
                dim=-1,
            )
        if merge_method == "add":
            waveform = waveform_1 + waveform_2
        elif merge_method == "subtract":
            waveform = waveform_1 - waveform_2
        elif merge_method == "multiply":
            waveform = waveform_1 * waveform_2
        else:
            waveform = (waveform_1 + waveform_2) / 2
        maximum = waveform.abs().max()
        if maximum > 1.0:
            waveform = waveform / maximum
        return cls.outputs(AUDIO={"waveform": waveform, "sample_rate": sample_rate})


AUDIO_COMBINE_NODES = (JoinAudioChannels, AudioConcat, AudioMerge)
