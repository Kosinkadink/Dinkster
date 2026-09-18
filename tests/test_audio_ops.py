"""Audio editing-op contracts: trim, split, join, concat, merge, volume, fade, EQ, envelope."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_foundation import CURVE_TYPE, Curve, CurveEditor
from dinkster_nodes_media_io import (
    AdjustAudioVolume,
    ConcatAudio,
    EqualizeAudio,
    ExtractAudioEnvelope,
    FadeAudio,
    JoinAudioChannels,
    MergeAudio,
    SplitAudioChannels,
    TrimAudio,
)


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
    return cast("np.ndarray", cast("dict[str, object]", result)["waveform"])


def _rate(result: object) -> int:
    return cast("int", cast("dict[str, object]", result)["sample_rate"])


def _empty(channels: int = 1) -> dict[str, object]:
    return {"waveform": np.zeros((1, channels, 0), dtype=np.float32), "sample_rate": 44_100}


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
    result = TrimAudio.execute(audio=source, start=start, duration=duration)
    first, last = expected_range
    expected = cast("np.ndarray", source["waveform"])[..., first:last]
    assert np.array_equal(_waveform(result["audio"]), expected)
    assert _rate(result["audio"]) == 44_100


def test_trim_passes_empty_audio_through() -> None:
    result = TrimAudio.execute(audio=_empty(), start=1.0, duration=1.0)
    assert _waveform(result["audio"]).shape == (1, 1, 0)


def test_trim_rejects_a_range_outside_the_audio() -> None:
    with pytest.raises(ValueError, match="start must land before"):
        TrimAudio.execute(audio=_audio(), start=0.2, duration=0.05)
    with pytest.raises(ValueError, match="start must land before"):
        TrimAudio.execute(audio=_audio(), start=0.05, duration=0.0)


def test_trim_rejects_non_finite_and_out_of_range_arguments() -> None:
    with pytest.raises(ValueError, match="start"):
        TrimAudio.execute(audio=_audio(), start=float("nan"), duration=1.0)
    with pytest.raises(ValueError, match="duration"):
        TrimAudio.execute(audio=_audio(), start=0.0, duration=-1.0)


def test_split_returns_left_and_right_mono_channels() -> None:
    source = _audio(channels=2)
    waveform = cast("np.ndarray", source["waveform"])
    waveform[0, 1, :] *= 0.5
    result = SplitAudioChannels.execute(audio=source)
    assert np.array_equal(_waveform(result["left"]), waveform[..., 0:1, :])
    assert np.array_equal(_waveform(result["right"]), waveform[..., 1:2, :])


def test_split_rejects_out_of_range_channel_index() -> None:
    with pytest.raises(ValueError, match="right_index"):
        SplitAudioChannels.execute(audio=_audio())


def test_join_trims_to_the_shorter_input_and_keeps_channel_order() -> None:
    left = _audio(4410)
    right = _audio(3000, amplitude=0.25)
    result = JoinAudioChannels.execute(audio_left=left, audio_right=right)
    joined = _waveform(result["audio"])
    assert joined.shape == (1, 2, 3000)
    assert np.array_equal(joined[:, 0:1, :], cast("np.ndarray", left["waveform"])[..., :3000])
    assert np.array_equal(joined[:, 1:2, :], cast("np.ndarray", right["waveform"]))


def test_join_resamples_the_lower_rate_input_to_the_higher_rate() -> None:
    result = JoinAudioChannels.execute(
        audio_left=_audio(4410, sample_rate=44_100),
        audio_right=_audio(2205, sample_rate=22_050),
    )
    assert _rate(result["audio"]) == 44_100
    assert _waveform(result["audio"]).shape[1] == 2


def test_join_accepts_stereo_inputs_and_rejects_batch_mismatches() -> None:
    result = JoinAudioChannels.execute(audio_left=_audio(channels=2), audio_right=_audio())
    assert _waveform(result["audio"]).shape[1] == 3
    with pytest.raises(ValueError, match="batch size"):
        JoinAudioChannels.execute(audio_left=_audio(batch=2), audio_right=_audio())


def test_concat_orders_by_direction() -> None:
    first = _audio(channels=2)
    second = _audio(3000, channels=2, amplitude=0.25)
    after = _waveform(ConcatAudio.execute(audio1=first, audio2=second)["audio"])
    before = _waveform(
        ConcatAudio.execute(audio1=first, audio2=second, direction="before")["audio"]
    )
    assert after.shape == (1, 2, 7410)
    assert np.array_equal(after[..., :4410], cast("np.ndarray", first["waveform"]))
    assert np.array_equal(before[..., :3000], cast("np.ndarray", second["waveform"]))


def test_concat_always_duplicates_mono_to_stereo_as_pinned() -> None:
    result = ConcatAudio.execute(audio1=_audio(), audio2=_audio(3000))
    joined = _waveform(result["audio"])
    assert joined.shape == (1, 2, 7410)
    assert np.array_equal(joined[:, 0], joined[:, 1])


def test_concat_resamples_mixed_rates_to_the_higher_rate() -> None:
    result = ConcatAudio.execute(
        audio1=_audio(2205, sample_rate=22_050), audio2=_audio(4410, sample_rate=44_100)
    )
    assert _rate(result["audio"]) == 44_100


def test_concat_rejects_bad_directions_and_batch_mismatches() -> None:
    with pytest.raises(ValueError, match="direction"):
        ConcatAudio.execute(audio1=_audio(), audio2=_audio(), direction="sideways")
    with pytest.raises(ValueError, match="batch size"):
        ConcatAudio.execute(audio1=_audio(batch=2), audio2=_audio())


def test_merge_pads_audio2_with_silence_to_audio1_length() -> None:
    first = _audio(4410)
    second = _audio(3000, amplitude=0.25)
    merged = _waveform(
        MergeAudio.execute(audio1=first, audio2=second, merge_method="subtract")["audio"]
    )
    assert merged.shape == (1, 1, 4410)
    expected_tail = cast("np.ndarray", first["waveform"])[..., 3000:]
    assert np.array_equal(merged[..., 3000:], expected_tail)


def test_merge_trims_audio2_to_audio1_length() -> None:
    merged = _waveform(
        MergeAudio.execute(audio1=_audio(3000), audio2=_audio(4410), merge_method="multiply")[
            "audio"
        ]
    )
    assert merged.shape == (1, 1, 3000)


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
    method: str, combine: object
) -> None:
    first = _audio(amplitude=0.4)
    second = _audio(amplitude=0.3)
    merged = _waveform(
        MergeAudio.execute(audio1=first, audio2=second, merge_method=method)["audio"]
    )
    a = cast("np.ndarray", first["waveform"])
    b = cast("np.ndarray", second["waveform"])
    assert np.array_equal(merged, cast("np.ndarray", combine(a, b)))  # type: ignore[operator]


def test_merge_peak_normalizes_only_above_unity() -> None:
    hot = _audio(amplitude=0.9)
    merged = _waveform(MergeAudio.execute(audio1=hot, audio2=hot, merge_method="add")["audio"])
    assert float(np.abs(merged).max()) == pytest.approx(1.0)
    quiet = _audio(amplitude=0.2)
    unscaled = _waveform(
        MergeAudio.execute(audio1=quiet, audio2=quiet, merge_method="add")["audio"]
    )
    assert float(np.abs(unscaled).max()) < 1.0


def test_merge_returns_audio1_when_either_side_is_empty() -> None:
    first = _audio()
    result = MergeAudio.execute(audio1=first, audio2=_empty(), merge_method="add")
    assert np.array_equal(_waveform(result["audio"]), cast("np.ndarray", first["waveform"]))
    empty_first = MergeAudio.execute(audio1=_empty(), audio2=first, merge_method="add")
    assert _waveform(empty_first["audio"]).shape == (1, 1, 0)


def test_volume_zero_gain_is_a_bit_exact_passthrough() -> None:
    source = _audio()
    result = AdjustAudioVolume.execute(audio=source, gain_db=0.0)
    assert np.array_equal(_waveform(result["audio"]), cast("np.ndarray", source["waveform"]))


def test_volume_applies_decibel_gain() -> None:
    source = _audio(amplitude=0.5)
    result = AdjustAudioVolume.execute(audio=source, gain_db=-6.0)
    expected = cast("np.ndarray", source["waveform"]) * np.float32(10.0 ** (-6.0 / 20.0))
    assert np.array_equal(_waveform(result["audio"]), expected)


def test_volume_rejects_out_of_range_gain() -> None:
    with pytest.raises(ValueError, match="gain_db"):
        AdjustAudioVolume.execute(audio=_audio(), gain_db=101.0)


def test_fade_zero_durations_keep_the_signal() -> None:
    source = _audio()
    result = FadeAudio.execute(audio=source)
    assert np.array_equal(_waveform(result["audio"]), cast("np.ndarray", source["waveform"]))


@pytest.mark.parametrize("curve", ["linear", "cosine"])
def test_fade_ramps_rise_to_unity_and_mirror_on_the_way_out(curve: str) -> None:
    source = {"waveform": np.ones((1, 1, 1000), dtype=np.float32), "sample_rate": 1000}
    result = FadeAudio.execute(audio=source, fade_in_seconds=0.1, fade_out_seconds=0.2, curve=curve)
    faded = _waveform(result["audio"])[0, 0]
    assert faded[0] < 0.05
    assert faded[99] == pytest.approx(1.0)
    assert np.all(faded[100:800] == 1.0)
    assert faded[800] == pytest.approx(1.0)
    assert faded[-1] < 0.05
    assert np.all(np.diff(faded[:100]) > 0)
    assert np.all(np.diff(faded[800:]) < 0)


def test_fade_overlapping_windows_multiply() -> None:
    source = {"waveform": np.ones((1, 1, 100), dtype=np.float32), "sample_rate": 100}
    result = FadeAudio.execute(audio=source, fade_in_seconds=1.0, fade_out_seconds=1.0)
    faded = _waveform(result["audio"])[0, 0]
    expected = ((np.arange(100) + 1.0) / 100.0) * ((np.arange(100) + 1.0) / 100.0)[::-1]
    assert np.allclose(faded, expected.astype(np.float32), atol=0.0)


def test_fade_rejects_unknown_curves() -> None:
    with pytest.raises(ValueError, match="curve"):
        FadeAudio.execute(audio=_audio(), curve="exponential")


def test_equalizer_zero_gains_pass_the_signal_through_bit_exactly() -> None:
    source = _audio(channels=2)
    result = EqualizeAudio.execute(audio=source)
    assert np.array_equal(_waveform(result["audio"]), cast("np.ndarray", source["waveform"]))


def test_equalizer_passes_empty_audio_through() -> None:
    result = EqualizeAudio.execute(audio=_empty(2), low_gain_db=6.0)
    assert _waveform(result["audio"]).shape == (1, 2, 0)


def test_equalizer_output_stays_clamped_at_high_gain() -> None:
    source = _audio(channels=2, amplitude=0.999)
    result = EqualizeAudio.execute(
        audio=source, low_gain_db=24.0, mid_gain_db=24.0, high_gain_db=24.0
    )
    waveform = _waveform(result["audio"])
    assert float(np.abs(waveform).max()) <= 1.0
    assert waveform.dtype == np.float32


def test_equalizer_boost_raises_band_energy() -> None:
    source = _audio(amplitude=0.1)
    boosted = _waveform(
        EqualizeAudio.execute(audio=source, mid_gain_db=12.0, mid_freq=440)["audio"]
    )
    original = cast("np.ndarray", source["waveform"])
    assert float(np.abs(boosted).max()) > float(np.abs(original).max()) * 2.0


def test_equalizer_rejects_out_of_range_parameters() -> None:
    with pytest.raises(ValueError, match="low_freq"):
        EqualizeAudio.execute(audio=_audio(), low_freq=10)
    with pytest.raises(ValueError, match="mid_q"):
        EqualizeAudio.execute(audio=_audio(), mid_q=0.0)
    with pytest.raises(ValueError, match="high_gain_db"):
        EqualizeAudio.execute(audio=_audio(), high_gain_db=25.0)
    with pytest.raises(ValueError, match="mid_freq"):
        EqualizeAudio.execute(audio=_audio(), mid_freq=8000)


@pytest.mark.parametrize(
    "bad_audio",
    [
        "not audio",
        {"waveform": [1, 2, 3], "sample_rate": 44_100},
        {"waveform": np.zeros((4410,), dtype=np.float32), "sample_rate": 44_100},
        {"waveform": np.zeros((1, 0, 100), dtype=np.float32), "sample_rate": 44_100},
        {"waveform": np.zeros((1, 1, 100), dtype=np.float32), "sample_rate": 0},
        {"waveform": np.zeros((1, 1, 100), dtype=np.float32), "sample_rate": True},
        {"waveform": np.full((1, 1, 100), np.nan, dtype=np.float32), "sample_rate": 44_100},
    ],
)
def test_ops_reject_malformed_audio_values(bad_audio: object) -> None:
    with pytest.raises(ValueError):
        TrimAudio.execute(audio=bad_audio, start=0.0, duration=1.0)


def test_concat_rejects_results_beyond_the_256_mib_budget() -> None:
    samples = 33_000_000
    big = {"waveform": np.zeros((1, 2, samples), dtype=np.float32), "sample_rate": 44_100}
    with pytest.raises(ValueError, match="256 MiB"):
        ConcatAudio.execute(audio1=big, audio2=big)


def _envelope(result: object) -> list[float]:
    return cast("list[float]", cast("dict[str, object]", result)["envelope"])


def _envelope_curve(result: object) -> Curve:
    return cast("Curve", cast("dict[str, object]", result)["curve"])


def test_envelope_recovers_a_bin_aligned_tone_amplitude() -> None:
    # 44_100 samples at fps 12 gives 3675-sample frames with 12 Hz bins, so a
    # 996 Hz half-amplitude tone lands exactly on a bin: max magnitude is the
    # tone amplitude in every frame, with no rectangular-window leakage.
    source = _audio(samples=44_100, amplitude=0.5)
    tone = cast("np.ndarray", source["waveform"])
    timeline = np.arange(44_100, dtype=np.float64) / 44_100
    tone[:] = (0.5 * np.sin(2 * np.pi * 996 * timeline)).astype(np.float32)
    result = ExtractAudioEnvelope.execute(audio=source, normalize=False)
    envelope = np.asarray(_envelope(result))
    assert cast("dict[str, object]", result)["frames"] == 12
    np.testing.assert_allclose(envelope, np.full(12, 0.5), rtol=0.0, atol=1e-4)
    curve = _envelope_curve(result)
    assert curve.points == tuple((index / 12, value) for index, value in enumerate(envelope))
    assert CurveEditor.execute(curve=curve)["curve"] == curve


def test_envelope_ignores_out_of_band_energy() -> None:
    source = _audio(samples=44_100, amplitude=0.5)  # 440 Hz tone
    result = ExtractAudioEnvelope.execute(
        audio=source, band_low_hz=5000.0, band_high_hz=9000.0, normalize=False
    )
    assert max(_envelope(result)) < 1e-2


def test_envelope_normalizes_peak_to_one_and_inverts() -> None:
    source = _audio(samples=44_100, amplitude=0.5, channels=2)
    tone = cast("np.ndarray", source["waveform"])
    tone[:, :, 22_050:] = 0.0
    normalized = _envelope(ExtractAudioEnvelope.execute(audio=source))
    assert max(normalized) == 1.0
    assert min(normalized) >= 0.0
    inverted = _envelope(ExtractAudioEnvelope.execute(audio=source, invert=True))
    np.testing.assert_allclose(
        np.asarray(inverted), 1.0 - np.asarray(normalized), rtol=0.0, atol=0.0
    )


def test_envelope_normalizes_silence_to_zeros() -> None:
    silent = {"waveform": np.zeros((1, 1, 44_100), dtype=np.float32), "sample_rate": 44_100}
    envelope = _envelope(ExtractAudioEnvelope.execute(audio=silent))
    assert set(envelope) == {0.0}
    assert all(math.isfinite(value) for value in envelope)


def test_envelope_frame_count_covers_a_partial_tail_frame() -> None:
    source = _audio(samples=40_000)
    result = ExtractAudioEnvelope.execute(audio=source, frames_per_second=12.0)
    # ceil(40_000 / 3675) frames, the last one shorter than the rest
    assert cast("dict[str, object]", result)["frames"] == 11
    assert len(_envelope(result)) == 11


def test_envelope_passes_empty_audio_through() -> None:
    result = ExtractAudioEnvelope.execute(audio=_empty())
    assert _envelope(result) == []
    assert cast("dict[str, object]", result)["frames"] == 0
    assert _envelope_curve(result) == Curve(((0.0, 0.0),))


def test_envelope_curve_reduces_to_bounded_time_aligned_peaks() -> None:
    sample_rate = 48_000
    fps = 240
    frame_samples = sample_rate // fps
    frame_count = 4_100
    phase = np.arange(frame_samples, dtype=np.float64) / sample_rate
    tone = np.sin(2 * np.pi * 960 * phase)
    amplitudes = np.where(np.arange(frame_count) % 2, 0.75, 0.25)
    amplitudes[0] = amplitudes[1]
    waveform = (amplitudes[:, None] * tone).reshape(1, 1, -1).astype(np.float32)
    result = ExtractAudioEnvelope.execute(
        audio={"waveform": waveform, "sample_rate": sample_rate},
        frames_per_second=fps,
        band_low_hz=900,
        band_high_hz=1100,
        normalize=False,
    )
    envelope = _envelope(result)
    curve = _envelope_curve(result)
    expected_indexes = (0, *range(3, frame_count, 2))
    assert len(curve.points) == 2_050
    assert curve.points == tuple(
        (index * frame_samples / sample_rate, envelope[index]) for index in expected_indexes
    )


def test_envelope_curve_schema_and_graph_link_round_trip() -> None:
    schema = ExtractAudioEnvelope.define_schema()
    assert [output.id for output in schema.outputs] == ["envelope", "frames", "curve"]
    assert schema.outputs[2].type.types == (CURVE_TYPE,)
    graph = Graph(
        nodes={
            "envelope": GraphNode("dinkster.audio.envelope", {}),
            "editor": GraphNode("dinkster.curve.editor", {"curve": Link("envelope", "curve")}),
        }
    )
    restored = graph_from_wire(graph_to_wire(graph))
    assert restored == graph
    assert restored.nodes["editor"].inputs["curve"] == Link("envelope", "curve")


def test_envelope_rejects_invalid_options() -> None:
    source = _audio()
    with pytest.raises(ValueError, match="band_high_hz"):
        ExtractAudioEnvelope.execute(audio=source, band_low_hz=500.0, band_high_hz=500.0)
    with pytest.raises(ValueError, match="invert requires normalize"):
        ExtractAudioEnvelope.execute(audio=source, normalize=False, invert=True)
    with pytest.raises(ValueError, match="single batch"):
        ExtractAudioEnvelope.execute(audio=_audio(batch=2))
    with pytest.raises(ValueError, match="operation"):
        ExtractAudioEnvelope.execute(audio=source, operation="median")
    with pytest.raises(ValueError, match="normalize must be a boolean"):
        ExtractAudioEnvelope.execute(audio=source, normalize=1)
    with pytest.raises(ValueError, match="frames_per_second"):
        ExtractAudioEnvelope.execute(audio=source, frames_per_second=0.0)
