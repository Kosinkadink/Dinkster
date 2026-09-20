"""Value-level audio editing ops, a 3-band biquad equalizer, and envelope extraction.

The equalizer is a float64 port of the pinned torchaudio biquad chain
(RBJ cookbook shelf/peaking coefficients, per-stage [-1, 1] clamp); it is
value-close to the float32 reference, replayed by the pinned goldens in
tests/test_audio_ops_goldens.py. Mixed-sample-rate inputs are matched
through the pack's libswresample seam, which differs numerically from
torchaudio's sinc resampler; that gap is declared on the alias records.

ExtractAudioEnvelope is the native replacement for the audio-reactive
amplitude idiom (per-frame band-limited FFT magnitude aggregated to one
value per video frame). It has no core ComfyUI equivalent and therefore
no alias record. Where the third-party idiom quantizes to int16 mono via
channel squeeze and divides by a possibly-zero peak, this node downmixes
channels by mean, keeps float precision throughout, and normalizes
silence to zeros.
"""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    AudioWindowReader,
    ComboWidget,
    Curve,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    append_audio_edit,
    audio_window,
    effective_audio_facts,
)
from scipy.signal import lfilter

from .audio import (
    AUDIO,
    COMBO,
    FLOAT,
    INT,
    MAX_EMPTY_AUDIO_SECONDS,
    resample_audio,
)
from .video import MAX_DECODED_AUDIO_BYTES

_CONCAT_DIRECTIONS = ("after", "before")
_MERGE_METHODS = ("add", "mean", "subtract", "multiply")
_FADE_CURVES = ("linear", "cosine")
_ENVELOPE_OPERATIONS = ("avg", "max", "sum")
_MAX_ENVELOPE_FPS = 240.0
_MAX_ENVELOPE_HZ = 100_000.0

BOOLEAN = TypeExpr.concrete("core.boolean")
FLOAT_LIST = TypeExpr.list_of(FLOAT)


def _audio_value(audio: object, name: str) -> tuple[np.ndarray, int]:
    """Validate an audio mapping for editing ops; empty waveforms pass."""
    if isinstance(audio, Mapping) and "source" in audio:
        facts = effective_audio_facts(audio)
        if facts["frames"] is None:
            raise ValueError("this audio operation requires a trim duration")
        window = audio_window(audio, 0, facts["frames"])
        return window["waveform"], window["sample_rate"]
    if not isinstance(audio, Mapping):
        raise ValueError(f"{name} must contain waveform and sample_rate")
    value = cast("Mapping[str, object]", audio)
    source = value.get("waveform")
    if not isinstance(source, np.ndarray):
        raise ValueError(f"{name} waveform must be a numpy array")
    waveform = source
    sample_rate = value.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{name} sample_rate must be a positive integer")
    if waveform.ndim != 3 or waveform.shape[0] < 1:
        raise ValueError(f"{name} waveform must have nonempty [B,C,T] layout")
    if waveform.shape[1] < 1:
        raise ValueError(f"{name} must have channels")
    if waveform.nbytes > MAX_DECODED_AUDIO_BYTES:
        raise ValueError(f"{name} waveform exceeds the 256 MiB input limit")
    if not bool(np.isfinite(waveform).all()):
        raise ValueError(f"{name} contains non-finite samples")
    return np.asarray(waveform, dtype=np.float32), sample_rate


def _bounded_float(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and in {minimum}..{maximum}, got {value!r}")
    return number


def _bounded_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}, got {value!r}")
    return value


def _choice(value: object, name: str, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}")
    return value


def _check_output_budget(waveform: np.ndarray) -> np.ndarray:
    if waveform.nbytes > MAX_DECODED_AUDIO_BYTES:
        raise ValueError("audio result exceeds the 256 MiB limit")
    return waveform


def _resample_batch(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if waveform.shape[-1] == 0:
        return waveform
    return np.stack([resample_audio(item, source_rate, target_rate) for item in waveform])


def _match_sample_rates(
    waveform_1: np.ndarray, rate_1: int, waveform_2: np.ndarray, rate_2: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resample the lower-rate waveform to the higher rate, as pinned."""
    if rate_1 == rate_2:
        return waveform_1, waveform_2, rate_1
    if rate_1 > rate_2:
        return waveform_1, _resample_batch(waveform_2, rate_2, rate_1), rate_1
    return _resample_batch(waveform_1, rate_1, rate_2), waveform_2, rate_2


class TrimAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.trim",
            display_name="Trim Audio",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "start",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(
                        min=-MAX_EMPTY_AUDIO_SECONDS, max=MAX_EMPTY_AUDIO_SECONDS, step=0.01
                    ),
                ),
                InputSpec(
                    "duration",
                    FLOAT,
                    required=False,
                    default=60.0,
                    widget=NumberWidget(min=0.0, max=MAX_EMPTY_AUDIO_SECONDS, step=0.01),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("cut audio", "audio clip", "shorten audio"),
        )

    @classmethod
    def execute(
        cls, *, audio: object, start: object = 0.0, duration: object = 60.0
    ) -> Mapping[str, object]:
        if not isinstance(audio, Mapping) or "source" not in audio:
            _audio_value(audio, "audio")
        facts = effective_audio_facts(audio)
        sample_rate = facts["sample_rate"]
        start_seconds = _bounded_float(
            start, "start", minimum=-MAX_EMPTY_AUDIO_SECONDS, maximum=MAX_EMPTY_AUDIO_SECONDS
        )
        duration_seconds = _bounded_float(
            duration, "duration", minimum=0.0, maximum=MAX_EMPTY_AUDIO_SECONDS
        )
        length = facts["frames"]
        if length == 0:
            return cls.outputs(audio=audio)
        start_frame = int(round(start_seconds * sample_rate))
        if start_seconds < 0:
            if length is None:
                raise ValueError("negative trim start requires a known duration")
            start_frame += length
        start_frame = max(0, min(start_frame, length) if length is not None else start_frame)
        end_frame = start_frame + int(round(duration_seconds * sample_rate))
        end_frame = max(0, min(end_frame, length) if length is not None else end_frame)
        if start_frame >= end_frame:
            raise ValueError("start must land before the trimmed range's end within the audio")
        return cls.outputs(
            audio=append_audio_edit(
                audio,
                {"trim": {"start_sample": start_frame, "sample_count": end_frame - start_frame}},
            )
        )


class SplitAudioChannels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.split_channels",
            display_name="Split Audio Channels",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "left_index", INT, required=False, default=0, widget=NumberWidget(min=0, step=1)
                ),
                InputSpec(
                    "right_index",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=0, step=1),
                ),
            ),
            outputs=(
                OutputSpec("left", AUDIO, preview=True),
                OutputSpec("right", AUDIO),
            ),
            search_terms=("stereo to mono", "separate channels"),
        )

    @classmethod
    def execute(
        cls, *, audio: object, left_index: int = 0, right_index: int = 1
    ) -> Mapping[str, object]:
        channels = effective_audio_facts(audio)["channels"]
        left = _bounded_int(left_index, "left_index", minimum=0, maximum=channels - 1)
        right = _bounded_int(right_index, "right_index", minimum=0, maximum=channels - 1)
        return cls.outputs(
            **{
                name: append_audio_edit(
                    audio,
                    {
                        "channel_map": {
                            "matrix": [[float(channel == index) for channel in range(channels)]],
                            "layout": "mono",
                        }
                    },
                )
                for name, index in (("left", left), ("right", right))
            }
        )


class JoinAudioChannels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.join_channels",
            display_name="Join Audio Channels",
            category="audio",
            inputs=(
                InputSpec("audio_left", AUDIO),
                InputSpec("audio_right", AUDIO),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("mono to stereo", "combine channels"),
        )

    @classmethod
    def execute(cls, *, audio_left: object, audio_right: object) -> Mapping[str, object]:
        left, rate_left = _audio_value(audio_left, "audio_left")
        right, rate_right = _audio_value(audio_right, "audio_right")
        if left.shape[0] != right.shape[0]:
            raise ValueError("both inputs must have the same batch size")
        left, right, sample_rate = _match_sample_rates(left, rate_left, right, rate_right)
        min_length = min(left.shape[-1], right.shape[-1])
        stereo = np.concatenate((left[..., :min_length], right[..., :min_length]), axis=1)
        joined = _check_output_budget(np.ascontiguousarray(stereo, dtype=np.float32))
        return cls.outputs(audio={"waveform": joined, "sample_rate": sample_rate})


class ConcatAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.concat",
            display_name="Concatenate Audio",
            category="audio",
            inputs=(
                InputSpec("audio1", AUDIO),
                InputSpec("audio2", AUDIO),
                InputSpec(
                    "direction",
                    COMBO,
                    required=False,
                    default="after",
                    widget=ComboWidget(options=_CONCAT_DIRECTIONS),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("join audio", "combine audio", "append audio"),
        )

    @classmethod
    def execute(
        cls, *, audio1: object, audio2: object, direction: object = "after"
    ) -> Mapping[str, object]:
        waveform_1, rate_1 = _audio_value(audio1, "audio1")
        waveform_2, rate_2 = _audio_value(audio2, "audio2")
        chosen = _choice(direction, "direction", _CONCAT_DIRECTIONS)
        if waveform_1.shape[0] != waveform_2.shape[0]:
            raise ValueError("both inputs must have the same batch size")
        # Mono is duplicated to the other input's channels, with stereo for mono+mono.
        channels = max(2, waveform_1.shape[1], waveform_2.shape[1])
        if waveform_1.shape[1] == 1:
            waveform_1 = np.repeat(waveform_1, channels, axis=1)
        if waveform_2.shape[1] == 1:
            waveform_2 = np.repeat(waveform_2, channels, axis=1)
        if waveform_1.shape[1] != waveform_2.shape[1]:
            raise ValueError(
                "concat requires matching channels; use an explicit channel map or downmix"
            )
        waveform_1, waveform_2, sample_rate = _match_sample_rates(
            waveform_1, rate_1, waveform_2, rate_2
        )
        ordered = (waveform_1, waveform_2) if chosen == "after" else (waveform_2, waveform_1)
        concatenated = _check_output_budget(
            np.ascontiguousarray(np.concatenate(ordered, axis=2), dtype=np.float32)
        )
        return cls.outputs(audio={"waveform": concatenated, "sample_rate": sample_rate})


class MergeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.merge",
            display_name="Merge Audio",
            category="audio",
            inputs=(
                InputSpec("audio1", AUDIO),
                InputSpec("audio2", AUDIO),
                InputSpec(
                    "merge_method",
                    COMBO,
                    required=False,
                    default="add",
                    widget=ComboWidget(options=_MERGE_METHODS),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("mix audio", "overlay audio", "layer audio"),
        )

    @classmethod
    def execute(
        cls, *, audio1: object, audio2: object, merge_method: object = "add"
    ) -> Mapping[str, object]:
        waveform_1, rate_1 = _audio_value(audio1, "audio1")
        waveform_2, rate_2 = _audio_value(audio2, "audio2")
        method = _choice(merge_method, "merge_method", _MERGE_METHODS)
        if waveform_1.shape[0] != waveform_2.shape[0]:
            raise ValueError("both inputs must have the same batch size")
        waveform_1, waveform_2, sample_rate = _match_sample_rates(
            waveform_1, rate_1, waveform_2, rate_2
        )
        length_1 = waveform_1.shape[-1]
        length_2 = waveform_2.shape[-1]
        if length_1 == 0 or length_2 == 0:
            return cls.outputs(audio={"waveform": waveform_1, "sample_rate": sample_rate})
        if length_2 > length_1:
            waveform_2 = waveform_2[..., :length_1]
        elif length_2 < length_1:
            pad_shape = list(waveform_2.shape)
            pad_shape[-1] = length_1 - length_2
            waveform_2 = np.concatenate(
                (waveform_2, np.zeros(pad_shape, dtype=waveform_2.dtype)), axis=-1
            )
        if method == "add":
            merged = waveform_1 + waveform_2
        elif method == "subtract":
            merged = waveform_1 - waveform_2
        elif method == "multiply":
            merged = waveform_1 * waveform_2
        else:
            merged = (waveform_1 + waveform_2) / np.float32(2.0)
        peak = float(np.abs(merged).max())
        if peak > 1.0:
            merged = merged / np.float32(peak)
        result = _check_output_budget(np.ascontiguousarray(merged, dtype=np.float32))
        return cls.outputs(audio={"waveform": result, "sample_rate": sample_rate})


class AdjustAudioVolume(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.volume",
            display_name="Adjust Audio Volume",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "gain_db",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("audio gain", "loudness", "audio level"),
        )

    @classmethod
    def execute(cls, *, audio: object, gain_db: object = 0.0) -> Mapping[str, object]:
        if not isinstance(audio, Mapping) or "source" not in audio:
            _audio_value(audio, "audio")
        gain = _bounded_float(gain_db, "gain_db", minimum=-100.0, maximum=100.0)
        return cls.outputs(
            audio=append_audio_edit(audio, {"gain": float(np.float32(10.0 ** (gain / 20.0)))})
        )


def _fade_ramp(samples: int, curve: str) -> np.ndarray:
    """Rising gain envelope ending exactly at 1.0; reversed for fade-out."""
    ramp = (np.arange(samples, dtype=np.float64) + 1.0) / samples
    if curve == "cosine":
        ramp = np.sin(ramp * (math.pi / 2.0))
    return ramp


class FadeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.fade",
            display_name="Fade Audio",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "fade_in_seconds",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=MAX_EMPTY_AUDIO_SECONDS, step=0.01),
                ),
                InputSpec(
                    "fade_out_seconds",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=MAX_EMPTY_AUDIO_SECONDS, step=0.01),
                ),
                InputSpec(
                    "curve",
                    COMBO,
                    required=False,
                    default="linear",
                    widget=ComboWidget(options=_FADE_CURVES),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("fade in", "fade out", "audio envelope"),
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        fade_in_seconds: object = 0.0,
        fade_out_seconds: object = 0.0,
        curve: object = "linear",
    ) -> Mapping[str, object]:
        waveform, sample_rate = _audio_value(audio, "audio")
        fade_in = _bounded_float(
            fade_in_seconds, "fade_in_seconds", minimum=0.0, maximum=MAX_EMPTY_AUDIO_SECONDS
        )
        fade_out = _bounded_float(
            fade_out_seconds, "fade_out_seconds", minimum=0.0, maximum=MAX_EMPTY_AUDIO_SECONDS
        )
        shape = _choice(curve, "curve", _FADE_CURVES)
        length = waveform.shape[-1]
        samples_in = min(int(round(fade_in * sample_rate)), length)
        samples_out = min(int(round(fade_out * sample_rate)), length)
        envelope = np.ones(length, dtype=np.float64)
        if samples_in > 0:
            envelope[:samples_in] *= _fade_ramp(samples_in, shape)
        if samples_out > 0:
            envelope[length - samples_out :] *= _fade_ramp(samples_out, shape)[::-1]
        faded = np.ascontiguousarray(waveform.astype(np.float64) * envelope, dtype=np.float32)
        return cls.outputs(audio={"waveform": faded, "sample_rate": sample_rate})


def _shelf_coefficients(
    *, kind: str, gain_db: float, frequency: float, q: float, sample_rate: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """RBJ cookbook low/high shelf coefficients, as in torchaudio."""
    w0 = 2.0 * math.pi * frequency / sample_rate
    alpha = math.sin(w0) / 2.0 / q
    big_a = math.exp(gain_db / 40.0 * math.log(10.0))
    temp1 = 2.0 * math.sqrt(big_a) * alpha
    temp2 = (big_a - 1.0) * math.cos(w0)
    temp3 = (big_a + 1.0) * math.cos(w0)
    if kind == "bass":
        numerator = (
            big_a * ((big_a + 1.0) - temp2 + temp1),
            2.0 * big_a * ((big_a - 1.0) - temp3),
            big_a * ((big_a + 1.0) - temp2 - temp1),
        )
        denominator = (
            (big_a + 1.0) + temp2 + temp1,
            -2.0 * ((big_a - 1.0) + temp3),
            (big_a + 1.0) + temp2 - temp1,
        )
    else:
        numerator = (
            big_a * ((big_a + 1.0) + temp2 + temp1),
            -2.0 * big_a * ((big_a - 1.0) + temp3),
            big_a * ((big_a + 1.0) + temp2 - temp1),
        )
        denominator = (
            (big_a + 1.0) - temp2 + temp1,
            2.0 * ((big_a - 1.0) - temp3),
            (big_a + 1.0) - temp2 - temp1,
        )
    return numerator, denominator


def _peaking_coefficients(
    *, gain_db: float, frequency: float, q: float, sample_rate: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """RBJ cookbook peaking EQ coefficients, as in torchaudio."""
    w0 = 2.0 * math.pi * frequency / sample_rate
    big_a = math.exp(gain_db / 40.0 * math.log(10.0))
    alpha = math.sin(w0) / 2.0 / q
    numerator = (1.0 + alpha * big_a, -2.0 * math.cos(w0), 1.0 - alpha * big_a)
    denominator = (1.0 + alpha / big_a, -2.0 * math.cos(w0), 1.0 - alpha / big_a)
    return numerator, denominator


def _biquad_stage(
    waveform: np.ndarray,
    coefficients: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> np.ndarray:
    """One normalized biquad pass with the reference's [-1, 1] output clamp."""
    numerator, denominator = coefficients
    scale = denominator[0]
    b = np.asarray(numerator, dtype=np.float64) / scale
    a = np.asarray(denominator, dtype=np.float64) / scale
    filtered = np.asarray(lfilter(b, a, waveform, axis=-1), dtype=np.float64)
    return np.clip(filtered, -1.0, 1.0)


class EqualizeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.equalizer",
            display_name="Audio Equalizer (3-Band)",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "low_gain_db",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                ),
                InputSpec(
                    "low_freq",
                    INT,
                    required=False,
                    default=100,
                    widget=NumberWidget(min=20, max=500, step=1),
                ),
                InputSpec(
                    "mid_gain_db",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                ),
                InputSpec(
                    "mid_freq",
                    INT,
                    required=False,
                    default=1000,
                    widget=NumberWidget(min=200, max=4000, step=1),
                ),
                InputSpec(
                    "mid_q",
                    FLOAT,
                    required=False,
                    default=0.707,
                    widget=NumberWidget(min=0.1, max=10.0, step=0.1),
                ),
                InputSpec(
                    "high_gain_db",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-24.0, max=24.0, step=0.1),
                ),
                InputSpec(
                    "high_freq",
                    INT,
                    required=False,
                    default=5000,
                    widget=NumberWidget(min=1000, max=15000, step=1),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("eq", "bass boost", "treble boost", "equalizer"),
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        low_gain_db: object = 0.0,
        low_freq: object = 100,
        mid_gain_db: object = 0.0,
        mid_freq: object = 1000,
        mid_q: object = 0.707,
        high_gain_db: object = 0.0,
        high_freq: object = 5000,
    ) -> Mapping[str, object]:
        waveform, sample_rate = _audio_value(audio, "audio")
        low_gain = _bounded_float(low_gain_db, "low_gain_db", minimum=-24.0, maximum=24.0)
        low_hz = _bounded_int(low_freq, "low_freq", minimum=20, maximum=500)
        mid_gain = _bounded_float(mid_gain_db, "mid_gain_db", minimum=-24.0, maximum=24.0)
        mid_hz = _bounded_int(mid_freq, "mid_freq", minimum=200, maximum=4000)
        mid_quality = _bounded_float(mid_q, "mid_q", minimum=0.1, maximum=10.0)
        high_gain = _bounded_float(high_gain_db, "high_gain_db", minimum=-24.0, maximum=24.0)
        high_hz = _bounded_int(high_freq, "high_freq", minimum=1000, maximum=15000)
        if waveform.shape[-1] == 0:
            return cls.outputs(audio={"waveform": waveform, "sample_rate": sample_rate})
        equalized = waveform.astype(np.float64)
        if low_gain != 0.0:
            equalized = _biquad_stage(
                equalized,
                _shelf_coefficients(
                    kind="bass",
                    gain_db=low_gain,
                    frequency=float(low_hz),
                    q=0.707,
                    sample_rate=sample_rate,
                ),
            )
        if mid_gain != 0.0:
            equalized = _biquad_stage(
                equalized,
                _peaking_coefficients(
                    gain_db=mid_gain,
                    frequency=float(mid_hz),
                    q=mid_quality,
                    sample_rate=sample_rate,
                ),
            )
        if high_gain != 0.0:
            equalized = _biquad_stage(
                equalized,
                _shelf_coefficients(
                    kind="treble",
                    gain_db=high_gain,
                    frequency=float(high_hz),
                    q=0.707,
                    sample_rate=sample_rate,
                ),
            )
        result = np.ascontiguousarray(equalized, dtype=np.float32)
        return cls.outputs(audio={"waveform": result, "sample_rate": sample_rate})


def _bool_option(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _frame_bounds(total_samples: int, samples_per_frame: float) -> tuple[int, ...]:
    """Rectangular per-frame slice boundaries, rounded as in the reference idiom."""
    total_frames = int(math.ceil(total_samples / samples_per_frame))
    edges = [round(index * samples_per_frame) for index in range(total_frames)]
    edges.append(total_samples)
    return tuple(min(edge, total_samples) for edge in edges)


def _envelope_curve(values: list[float], edges: tuple[int, ...], sample_rate: int) -> Curve:
    if not values:
        return Curve(((0.0, 0.0),))
    group_size = max(1, math.ceil(len(values) / 4096))
    points: list[tuple[float, float]] = []
    for start in range(0, len(values), group_size):
        stop = min(start + group_size, len(values))
        peak = max(range(start, stop), key=values.__getitem__)
        points.append((edges[peak] / sample_rate, values[peak]))
    return Curve(tuple(points))


class ExtractAudioEnvelope(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.envelope",
            editor_role="audio-envelope",
            display_name="Extract Audio Envelope",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "frames_per_second",
                    FLOAT,
                    required=False,
                    default=12.0,
                    widget=NumberWidget(min=0.1, max=_MAX_ENVELOPE_FPS, step=0.1),
                ),
                InputSpec(
                    "band_low_hz",
                    FLOAT,
                    required=False,
                    default=500.0,
                    widget=NumberWidget(min=0.0, max=_MAX_ENVELOPE_HZ, step=1.0),
                ),
                InputSpec(
                    "band_high_hz",
                    FLOAT,
                    required=False,
                    default=4000.0,
                    widget=NumberWidget(min=0.0, max=_MAX_ENVELOPE_HZ, step=1.0),
                ),
                InputSpec(
                    "operation",
                    COMBO,
                    required=False,
                    default="max",
                    widget=ComboWidget(options=_ENVELOPE_OPERATIONS),
                ),
                InputSpec("normalize", BOOLEAN, required=False, default=True),
                InputSpec("invert", BOOLEAN, required=False, default=False),
            ),
            outputs=(
                OutputSpec("envelope", FLOAT_LIST),
                OutputSpec("frames", INT),
                OutputSpec("curve", TypeExpr.concrete("dinkster.curve")),
            ),
            search_terms=("amplitude", "audio reactive", "envelope", "beat"),
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        frames_per_second: object = 12.0,
        band_low_hz: object = 500.0,
        band_high_hz: object = 4000.0,
        operation: object = "max",
        normalize: object = True,
        invert: object = False,
    ) -> Mapping[str, object]:
        waveform, sample_rate = _audio_value(audio, "audio")
        fps = _bounded_float(
            frames_per_second, "frames_per_second", minimum=0.1, maximum=_MAX_ENVELOPE_FPS
        )
        band_low = _bounded_float(band_low_hz, "band_low_hz", minimum=0.0, maximum=_MAX_ENVELOPE_HZ)
        band_high = _bounded_float(
            band_high_hz, "band_high_hz", minimum=0.0, maximum=_MAX_ENVELOPE_HZ
        )
        if band_high <= band_low:
            raise ValueError("band_high_hz must be greater than band_low_hz")
        method = _choice(operation, "operation", _ENVELOPE_OPERATIONS)
        normalized = _bool_option(normalize, "normalize")
        inverted = _bool_option(invert, "invert")
        if inverted and not normalized:
            raise ValueError("invert requires normalize")
        if waveform.shape[0] != 1:
            raise ValueError("audio must contain a single batch item")
        total_samples = waveform.shape[-1]
        if total_samples == 0:
            return cls.outputs(envelope=[], frames=0, curve=_envelope_curve([], (0,), sample_rate))
        mono = waveform[0].astype(np.float64).mean(axis=0)
        edges = _frame_bounds(total_samples, sample_rate / fps)
        values = np.zeros(len(edges) - 1, dtype=np.float64)
        for index in range(len(edges) - 1):
            frame = mono[edges[index] : edges[index + 1]]
            if frame.size == 0:
                continue
            spectrum = (2.0 / frame.size) * np.abs(np.fft.rfft(frame))
            frequencies = np.fft.rfftfreq(frame.size, 1.0 / sample_rate)
            band = spectrum[(frequencies >= band_low) & (frequencies < band_high)]
            if band.size == 0:
                continue
            if method == "avg":
                values[index] = float(np.mean(band))
            elif method == "max":
                values[index] = float(np.max(band))
            else:
                values[index] = float(np.sum(band))
        if normalized:
            peak = float(np.max(values))
            if peak > 0.0:
                values = values / peak
            if inverted:
                values = 1.0 - values
        envelope = [float(value) for value in values]
        return cls.outputs(
            envelope=envelope,
            frames=int(values.size),
            curve=_envelope_curve(envelope, edges, sample_rate),
        )


class ResampleAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.resample",
            display_name="Resample Audio",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "sample_rate",
                    INT,
                    required=False,
                    default=48_000,
                    widget=NumberWidget(min=1, step=1),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
        )

    @classmethod
    def execute(cls, *, audio: object, sample_rate: int = 48_000) -> Mapping[str, object]:
        return cls.outputs(audio=append_audio_edit(audio, {"resample": sample_rate}))


class DownmixAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.downmix",
            display_name="Downmix Audio",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "layout",
                    COMBO,
                    required=False,
                    default="stereo",
                    widget=ComboWidget(options=("mono", "stereo")),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
        )

    @classmethod
    def execute(cls, *, audio: object, layout: str = "stereo") -> Mapping[str, object]:
        import av

        chosen = _choice(layout, "layout", ("mono", "stereo"))
        facts = effective_audio_facts(audio)
        channels = facts["channels"]
        if chosen == "mono":
            matrix = [[1.0 / channels] * channels]
        elif channels == 1:
            matrix = [[1.0], [1.0]]
        else:
            try:
                labels = [channel.name for channel in av.AudioLayout(facts["layout"]).channels]
            except ValueError:
                labels = []
            if len(labels) != channels:
                warnings.warn(
                    "unknown audio layout; distributing channels equally to stereo", stacklevel=2
                )
                labels = [""] * channels
            matrix = [[0.0] * channels for _ in range(2)]
            for index, label in enumerate(labels):
                if label in ("FL", "BL", "SL"):
                    matrix[0][index] = 1.0 if label == "FL" else math.sqrt(0.5)
                elif label in ("FR", "BR", "SR"):
                    matrix[1][index] = 1.0 if label == "FR" else math.sqrt(0.5)
                elif label != "LFE":
                    matrix[0][index] = matrix[1][index] = math.sqrt(0.5)
            scale = max(1.0, *(sum(row) for row in matrix))
            matrix = [[weight / scale for weight in row] for row in matrix]
        return cls.outputs(
            audio=append_audio_edit(audio, {"channel_map": {"matrix": matrix, "layout": chosen}})
        )


class AudioOnsets(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.audio.onsets",
            display_name="Audio Onsets",
            category="audio",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "frames_per_second",
                    FLOAT,
                    required=False,
                    default=50.0,
                    widget=NumberWidget(min=1.0, max=240.0, step=1.0),
                ),
            ),
            outputs=(OutputSpec("curve", TypeExpr.concrete("dinkster.curve")),),
            search_terms=("beat", "audio reactive", "spectral flux"),
        )

    @classmethod
    def execute(cls, *, audio: object, frames_per_second: float = 50.0) -> Mapping[str, object]:
        fps = _bounded_float(frames_per_second, "frames_per_second", minimum=1.0, maximum=240.0)
        facts = effective_audio_facts(audio)
        if facts["frames"] is None:
            raise ValueError("onset detection requires a trim duration")
        hop = max(1, round(facts["sample_rate"] / fps))
        # CURVE carries at most 4096 control points; retain the strongest onset in each group.
        group_size = max(1, math.ceil(math.ceil(facts["frames"] / hop) / 4096))
        if group_size > 1:
            warnings.warn("onsets are peak-aggregated to the 4096-point CURVE budget", stacklevel=2)
        previous: np.ndarray | None = None
        values: list[float] = []
        times: list[float] = []
        with AudioWindowReader(audio) as reader:
            for start in range(0, facts["frames"], hop):
                chunk = reader.read(start, min(hop, facts["frames"] - start), batch_index=0)[
                    "waveform"
                ][0]
                mono = chunk.mean(axis=0)
                spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=hop))
                flux = (
                    float(np.maximum(spectrum - previous, 0).sum())
                    if previous is not None
                    else float(spectrum.sum())
                )
                if (start // hop) % group_size == 0:
                    values.append(flux)
                    times.append(start / facts["sample_rate"])
                elif flux > values[-1]:
                    values[-1] = flux
                    times[-1] = start / facts["sample_rate"]
                previous = spectrum
        peak = max(values, default=0.0)
        if peak:
            values = [value / peak for value in values]
        points = tuple(zip(times, values, strict=True)) or ((0.0, 0.0),)
        return cls.outputs(curve=Curve(points))


AUDIO_OPS_NODES: tuple[type[Node], ...] = (
    TrimAudio,
    SplitAudioChannels,
    JoinAudioChannels,
    ConcatAudio,
    MergeAudio,
    AdjustAudioVolume,
    FadeAudio,
    EqualizeAudio,
    ExtractAudioEnvelope,
    ResampleAudio,
    DownmixAudio,
    AudioOnsets,
)

__all__ = [
    "AUDIO_OPS_NODES",
    "AudioOnsets",
    "DownmixAudio",
    "ResampleAudio",
    "AdjustAudioVolume",
    "ConcatAudio",
    "EqualizeAudio",
    "ExtractAudioEnvelope",
    "FadeAudio",
    "JoinAudioChannels",
    "MergeAudio",
    "SplitAudioChannels",
    "TrimAudio",
]
