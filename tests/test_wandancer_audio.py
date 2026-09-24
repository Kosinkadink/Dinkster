import base64
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from dinkster_model_wan.wandancer_audio import (
    WanDancerAudioFeatures,
    encode_wandancer_audio_features,
    plan_wandancer_keyframe_list,
    plan_wandancer_keyframes,
    quick_tempo_estimate,
)
from threadpoolctl import threadpool_limits

_DIRECT_OPERATION_GOLDEN = json.loads(
    (Path(__file__).parent / "goldens" / "comfy_direct_operations_b78cec87.json").read_text()
)

_AUDIO_FEATURE_DIGEST_PATH = Path(__file__).parent / "goldens" / "wandancer_audio_digest.json"
_AUDIO_FEATURE_MAX_PROVIDER_DRIFT = 9.1553e-5
_AUDIO_FEATURE_DRIFT_MARGIN = 8.447e-6
_AUDIO_FEATURE_THREAD_COUNT = 4
# The 1e-4 quantum is the measured drift plus 9.2% headroom.
_AUDIO_FEATURE_COMPARISON_QUANTUM = np.float32(
    _AUDIO_FEATURE_MAX_PROVIDER_DRIFT + _AUDIO_FEATURE_DRIFT_MARGIN
)


def _canonical_audio_feature_digest(feature: np.ndarray) -> str:
    onset_mfcc = np.ascontiguousarray(
        np.rint(feature[..., :21] / _AUDIO_FEATURE_COMPARISON_QUANTUM), dtype="<i4"
    )
    chroma_events = np.ascontiguousarray(
        np.rint(feature[..., 21:] / _AUDIO_FEATURE_COMPARISON_QUANTUM), dtype="<i4"
    )
    return hashlib.sha256(onset_mfcc.tobytes() + chroma_events.tobytes()).hexdigest()


def _expected_audio_feature() -> np.ndarray:
    document = json.loads(_AUDIO_FEATURE_DIGEST_PATH.read_text(encoding="utf-8"))
    assert document["format"] == "dinkster-wandancer-audio-features/2"
    return np.frombuffer(base64.b64decode(document["float32Base64"]), dtype="<f4").reshape(
        document["shape"]
    )


def _wandancer_audio_feature() -> np.ndarray:
    samples = 15_360
    time = np.arange(samples, dtype=np.float32) / 15_360
    waveform = (
        0.3 * np.sin(2 * np.pi * 220 * time)
        + 0.2 * np.sin(2 * np.pi * 330 * time)
        + (np.arange(samples) % 2048 < 24) * 0.5
    ).astype(np.float32)
    thread_limit = (
        threadpool_limits(limits=_AUDIO_FEATURE_THREAD_COUNT)
        if sys.platform.startswith("linux")
        else nullcontext()
    )
    with thread_limit:
        return encode_wandancer_audio_features(waveform, 15_360, waveform, 31).audio_feature


def test_audio_features_match_pinned_comfyui_reference() -> None:
    feature = _wandancer_audio_feature()

    assert feature.shape == (1, 31, 35)
    assert feature.dtype == np.float32
    # Hosted providers differed by at most 9.1553e-05; Windows also produced
    # digest e16d9f95b357eb7ff23174170222764c8a05d900bdb513faa8fdbe99670e6162.
    # The 2e-04 limit is more than twice that spread, with a 1e-05 relative floor.
    np.testing.assert_allclose(feature, _expected_audio_feature(), rtol=1e-5, atol=2e-4)
    assert np.flatnonzero(feature[0, :, 33]).tolist() == [5, 9, 13, 17, 21, 25, 29]
    assert np.flatnonzero(feature[0, :, 34]).tolist() == [5, 17, 29]


def test_audio_feature_canonical_digest_preserves_its_numerical_boundaries() -> None:
    feature = _wandancer_audio_feature()
    baseline = _canonical_audio_feature_digest(feature)

    provider_low = feature.copy()
    provider_high = feature.copy()
    for index in (5, 21):
        provider_center = (
            np.rint(feature[0, 0, index] / _AUDIO_FEATURE_COMPARISON_QUANTUM)
            * _AUDIO_FEATURE_COMPARISON_QUANTUM
        )
        provider_low[0, 0, index] = provider_center - _AUDIO_FEATURE_MAX_PROVIDER_DRIFT / 2
        provider_high[0, 0, index] = provider_center + _AUDIO_FEATURE_MAX_PROVIDER_DRIFT / 2
        assert (
            provider_high[0, 0, index] - provider_low[0, 0, index]
            <= _AUDIO_FEATURE_COMPARISON_QUANTUM
        )
    assert _canonical_audio_feature_digest(provider_low) == baseline
    assert _canonical_audio_feature_digest(provider_high) == baseline

    for index in (0, 21):
        material_change = feature.copy()
        material_change[..., index] += _AUDIO_FEATURE_COMPARISON_QUANTUM * 2
        assert _canonical_audio_feature_digest(material_change) != baseline


def test_audio_features_shape_dtype_mono_mix_and_fps() -> None:
    samples = 15_360
    time = np.arange(samples, dtype=np.float32) / 15_360
    mono = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    stereo = np.stack((mono * 0.5, mono * 1.5))[None]
    mixed = encode_wandancer_audio_features(stereo, 15_360, stereo, 31, 2.5)
    direct = encode_wandancer_audio_features(mono, 15_360, mono, 31, 2.5)

    assert isinstance(mixed, WanDancerAudioFeatures)
    assert mixed.audio_feature.shape == (1, 31, 35)
    assert mixed.audio_feature.dtype == np.float32
    np.testing.assert_array_equal(mixed.audio_feature, direct.audio_feature)
    assert mixed.fps == 30.0
    assert mixed.audio_inject_scale == 2.5


def test_audio_features_are_deterministic() -> None:
    rng = np.random.default_rng(123)
    waveform = rng.normal(0, 0.01, 7_680).astype(np.float32)
    first = encode_wandancer_audio_features(waveform, 15_360, waveform, 16)
    second = encode_wandancer_audio_features(waveform, 15_360, waveform, 16)

    np.testing.assert_array_equal(first.audio_feature, second.audio_feature)


def test_short_audio_tempo_and_invalid_video_boundary() -> None:
    assert quick_tempo_estimate(np.zeros(100, np.float32), 44_100) == 120.0
    with pytest.raises(ValueError, match="too short"):
        encode_wandancer_audio_features(
            np.ones(100, np.float32), 15_360, np.ones(100, np.float32), 149
        )


def test_keyframes_normal_final_empty_and_audio_slices() -> None:
    images = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    waveform = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
    first = plan_wandancer_keyframes(images, 4, 0, waveform, 30)
    final = plan_wandancer_keyframes(images, 4, 1, waveform, 30)

    assert np.flatnonzero(first.mask[:, 0, 0]).tolist() == [0, 3]
    assert np.flatnonzero(final.mask[:, 0, 0]).tolist() == [0, 3]
    np.testing.assert_array_equal(first.audio_waveform, waveform[:, :, :4])
    np.testing.assert_array_equal(final.audio_waveform, waveform[:, :, 4:8])

    empty = plan_wandancer_keyframes(images[:0], 4, 0, waveform[:, :, :5], 30)
    assert not empty.mask.any()
    assert not empty.keyframes.any()


def test_keyframe_list_count_and_boundaries() -> None:
    images = np.ones((1, 2, 2, 3), np.float32)
    waveform = np.ones((1, 1, 10), np.float32)
    segments = plan_wandancer_keyframe_list(images, 4, 3, waveform, 30)
    assert len(segments) == 3
    assert segments[-1].audio_waveform.shape[-1] == 2

    with pytest.raises(ValueError):
        plan_wandancer_keyframes(images, 0, 0, waveform, 30)
    with pytest.raises(ValueError):
        plan_wandancer_keyframes(images, 4, -1, waveform, 30)
    with pytest.raises(ValueError):
        plan_wandancer_keyframe_list(images, 4, 0, waveform, 30)


def test_keyframe_list_matches_pinned_comfyui_operation_goldens() -> None:
    assert _DIRECT_OPERATION_GOLDEN["format"] == "dinkster-comfy-direct-operation-golden/1"
    assert _DIRECT_OPERATION_GOLDEN["referenceCommit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    operation = next(
        item
        for item in _DIRECT_OPERATION_GOLDEN["operations"]
        if item["sourceNode"] == "WanDancerPadKeyframesList"
    )
    assert operation["scope"] == "operation-only-not-family-parity"
    for case in operation["cases"]:
        inputs = case["inputs"]
        image_record = inputs["images"]
        waveform_record = inputs["waveform"]
        images = np.asarray(image_record["data"], dtype=image_record["dtype"]).reshape(
            image_record["shape"]
        )
        waveform = np.asarray(waveform_record["data"], dtype=waveform_record["dtype"]).reshape(
            waveform_record["shape"]
        )
        segments = plan_wandancer_keyframe_list(
            images,
            inputs["segment_length"],
            inputs["num_segments"],
            waveform,
            inputs["sample_rate"],
        )

        actual = []
        for segment in segments:
            values = {}
            for name, array in (
                ("keyframes", segment.keyframes),
                ("mask", segment.mask),
                ("waveform", segment.audio_waveform),
            ):
                contiguous = np.ascontiguousarray(array)
                values[name] = {
                    "shape": list(contiguous.shape),
                    "dtype": contiguous.dtype.name,
                    "nonzero": int(np.count_nonzero(contiguous)),
                    "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                }
            actual.append({**values, "sampleRate": segment.sample_rate})
        assert actual == case["outputs"]
