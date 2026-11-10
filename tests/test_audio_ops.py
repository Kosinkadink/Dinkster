"""First-party audio editing contracts: trim, split, join, concat, merge, volume and EQ."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pytest
from dinkster_audio.audio_combine import AudioConcat, AudioMerge, JoinAudioChannels
from dinkster_audio.audio_effects import AudioEqualizer3Band, TrimAudioDuration
from dinkster_audio.audio_transform import AudioAdjustVolume, SplitAudioChannels


def _audio(
    samples: int = 4410,
    *,
    channels: int = 1,
    sample_rate: int = 44_100,
    amplitude: float = 0.5,
    batch: int = 1,
) -> dict[str, object]:
    timeline = np.arange(samples, dtype=np.float64) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
    waveform = np.broadcast_to(tone, (batch, channels, samples)).copy()
    return {"waveform": waveform, "sample_rate": sample_rate}


def _waveform(result: object) -> np.ndarray:
    return np.asarray(cast("dict[str, object]", result)["waveform"])


def _rate(result: object) -> int:
    return cast("int", cast("dict[str, object]", result)["sample_rate"])


@pytest.mark.parametrize(
    ("start", "duration", "expected_range"),
    [
        (0.0, 0.05, (0, 2205)),
        (0.05, 0.02, (2205, 3087)),
        (-0.05, 0.02, (2205, 3087)),
        (0.05, 60.0, (2205, 4410)),
        (-0.0123, 0.006, (3868, 4133)),
    ],
)
def test_trim_selects_the_pinned_rounded_sample_range(
    start: float, duration: float, expected_range: tuple[int, int]
) -> None:
    source = _audio()
    result = TrimAudioDuration.execute(audio=source, start_index=start, duration=duration)
    first, last = expected_range
    expected = _waveform(source)[..., first:last]
    assert np.array_equal(_waveform(result["AUDIO"]), expected)
    assert _rate(result["AUDIO"]) == 44_100


def test_trim_passes_empty_audio_through() -> None:
    result = TrimAudioDuration.execute(audio=_audio(0), start_index=1.0, duration=1.0)
    assert _waveform(result["AUDIO"]).shape == (1, 1, 0)


@pytest.mark.parametrize(("start", "duration"), [(0.2, 0.05), (0.05, 0.0), (0.0, -1.0)])
def test_trim_rejects_a_range_outside_the_audio(start: float, duration: float) -> None:
    with pytest.raises(ValueError, match="Start time must be less than end time"):
        TrimAudioDuration.execute(audio=_audio(), start_index=start, duration=duration)


def test_trim_rejects_non_finite_start() -> None:
    with pytest.raises(ValueError, match="NaN"):
        TrimAudioDuration.execute(audio=_audio(), start_index=float("nan"), duration=1.0)


def test_split_returns_left_and_right_mono_channels() -> None:
    source = _audio(channels=2)
    waveform = _waveform(source)
    waveform[0, 1, :] *= 0.5
    result = SplitAudioChannels.execute(audio=source)
    assert np.array_equal(_waveform(result["left"]), waveform[..., 0:1, :])
    assert np.array_equal(_waveform(result["right"]), waveform[..., 1:2, :])


def test_split_requires_stereo() -> None:
    with pytest.raises(ValueError, match="must be stereo"):
        SplitAudioChannels.execute(audio=_audio())


def test_join_trims_to_the_shorter_input_and_keeps_channel_order() -> None:
    left = _audio(4410)
    right = _audio(3000, amplitude=0.25)
    result = JoinAudioChannels.execute(audio_left=left, audio_right=right)
    joined = _waveform(result["audio"])
    assert joined.shape == (1, 2, 3000)
    assert np.array_equal(joined[:, 0:1, :], _waveform(left)[..., :3000])
    assert np.array_equal(joined[:, 1:2, :], _waveform(right))


def test_join_resamples_the_lower_rate_input_to_the_higher_rate() -> None:
    result = JoinAudioChannels.execute(
        audio_left=_audio(4410, sample_rate=44_100),
        audio_right=_audio(2205, sample_rate=22_050),
    )
    assert _rate(result["audio"]) == 44_100
    assert _waveform(result["audio"]).shape == (1, 2, 4410)


def test_join_rejects_stereo_inputs_and_batch_mismatches() -> None:
    with pytest.raises(ValueError, match="must be mono"):
        JoinAudioChannels.execute(audio_left=_audio(channels=2), audio_right=_audio())
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        JoinAudioChannels.execute(audio_left=_audio(batch=2), audio_right=_audio())


def test_concat_orders_by_direction() -> None:
    first = _audio(channels=2)
    second = _audio(3000, channels=2, amplitude=0.25)
    after = _waveform(AudioConcat.execute(audio1=first, audio2=second, direction="after")["AUDIO"])
    before = _waveform(
        AudioConcat.execute(audio1=first, audio2=second, direction="before")["AUDIO"]
    )
    assert after.shape == (1, 2, 7410)
    assert np.array_equal(after[..., :4410], _waveform(first))
    assert np.array_equal(before[..., :3000], _waveform(second))


def test_concat_always_duplicates_mono_to_stereo_as_pinned() -> None:
    result = AudioConcat.execute(audio1=_audio(), audio2=_audio(3000), direction="after")
    joined = _waveform(result["AUDIO"])
    assert joined.shape == (1, 2, 7410)
    assert np.array_equal(joined[:, 0], joined[:, 1])


def test_concat_resamples_mixed_rates_to_the_higher_rate() -> None:
    result = AudioConcat.execute(
        audio1=_audio(2205, sample_rate=22_050),
        audio2=_audio(4410, sample_rate=44_100),
        direction="after",
    )
    assert _rate(result["AUDIO"]) == 44_100
    assert _waveform(result["AUDIO"]).shape == (1, 2, 8820)


def test_concat_rejects_batch_mismatches() -> None:
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        AudioConcat.execute(audio1=_audio(batch=2), audio2=_audio(), direction="after")


def test_merge_pads_audio2_with_silence_to_audio1_length() -> None:
    first = _audio(4410)
    second = _audio(3000, amplitude=0.25)
    merged = _waveform(
        AudioMerge.execute(audio1=first, audio2=second, merge_method="subtract")["AUDIO"]
    )
    assert merged.shape == (1, 1, 4410)
    assert np.array_equal(merged[..., 3000:], _waveform(first)[..., 3000:])


def test_merge_trims_audio2_to_audio1_length() -> None:
    first, second = _audio(3000), _audio(4410)
    merged = _waveform(
        AudioMerge.execute(audio1=first, audio2=second, merge_method="multiply")["AUDIO"]
    )
    assert merged.shape == (1, 1, 3000)
    assert np.array_equal(merged, _waveform(first) * _waveform(second)[..., :3000])


@pytest.mark.parametrize(
    ("method", "combine"),
    [
        ("add", lambda a, b: a + b),
        ("subtract", lambda a, b: a - b),
        ("multiply", lambda a, b: a * b),
        ("mean", lambda a, b: (a + b) / np.float32(2.0)),
    ],
)
def test_merge_methods_match_their_definitions_below_the_normalize_peak(
    method: Literal["add", "subtract", "multiply", "mean"],
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    first = _audio(amplitude=0.4)
    second = _audio(amplitude=0.3)
    merged = _waveform(
        AudioMerge.execute(audio1=first, audio2=second, merge_method=method)["AUDIO"]
    )
    assert np.array_equal(merged, combine(_waveform(first), _waveform(second)))


def test_merge_peak_normalizes_only_above_unity() -> None:
    hot = _audio(amplitude=0.9)
    merged = _waveform(AudioMerge.execute(audio1=hot, audio2=hot, merge_method="add")["AUDIO"])
    assert float(np.abs(merged).max()) == pytest.approx(1.0)
    quiet = _audio(amplitude=0.2)
    unscaled = _waveform(
        AudioMerge.execute(audio1=quiet, audio2=quiet, merge_method="add")["AUDIO"]
    )
    assert float(np.abs(unscaled).max()) < 1.0


def test_merge_returns_audio1_when_either_side_is_empty() -> None:
    first = _audio()
    result = AudioMerge.execute(audio1=first, audio2=_audio(0), merge_method="add")
    assert np.array_equal(_waveform(result["AUDIO"]), _waveform(first))
    empty_first = AudioMerge.execute(audio1=_audio(0), audio2=first, merge_method="add")
    assert _waveform(empty_first["AUDIO"]).shape == (1, 1, 0)


@pytest.mark.parametrize("volume", [0, -6])
def test_volume_applies_decibel_gain(volume: int) -> None:
    source = _audio(amplitude=0.5)
    result = AudioAdjustVolume.execute(audio=source, volume=volume)
    expected = _waveform(source) * np.float32(10.0 ** (volume / 20.0))
    assert np.array_equal(_waveform(result["AUDIO"]), expected)


@pytest.mark.parametrize(
    ("samples", "amplitude", "gain"), [(4410, 0.5, 0.0), (0, 0.5, 6.0), (4410, 0.999, 24.0)]
)
def test_equalizer_preserves_empty_and_zero_gain_and_clamps_high_gain(
    samples: int, amplitude: float, gain: float
) -> None:
    source = _audio(samples, channels=2, amplitude=amplitude)
    result = AudioEqualizer3Band.execute(
        audio=source,
        low_gain_dB=gain,
        low_freq=100,
        mid_gain_dB=gain,
        mid_freq=1000,
        mid_q=0.707,
        high_gain_dB=gain,
        high_freq=5000,
    )
    waveform = _waveform(result["AUDIO"])
    assert waveform.shape == (1, 2, samples)
    assert waveform.dtype == np.float32
    if gain == 0 or samples == 0:
        assert np.array_equal(waveform, _waveform(source))
    else:
        assert float(np.abs(waveform).max()) <= 1.0


def test_equalizer_boost_raises_band_energy() -> None:
    source = _audio(amplitude=0.1)
    result = AudioEqualizer3Band.execute(
        audio=source,
        low_gain_dB=0.0,
        low_freq=100,
        mid_gain_dB=12.0,
        mid_freq=440,
        mid_q=0.707,
        high_gain_dB=0.0,
        high_freq=5000,
    )
    assert (
        float(np.abs(_waveform(result["AUDIO"])).max())
        > float(np.abs(_waveform(source)).max()) * 2.0
    )
