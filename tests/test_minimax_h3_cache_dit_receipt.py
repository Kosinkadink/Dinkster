from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tools import minimax_h3_cache_dit_receipt as receipt


def test_receipt_matrix_uses_eight_asymmetric_prompt_seed_cases() -> None:
    assert receipt.SHAPES == ((672, 384, 56), (1344, 768, 124))
    assert receipt.STEPS == 20
    assert len(receipt.CASES) == 8
    assert len({case["seed"] for case in receipt.CASES}) == 8
    assert len({case["prompt"] for case in receipt.CASES}) == 8
    categories = {category for case in receipt.CASES for category in case["category"].split(",")}
    assert categories == {"motion", "text", "faces", "audio-transients"}


def test_tensor_metrics_report_cosine_and_relative_rmse_independently() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([3.0, 4.0])
    candidate = torch.tensor([0.0, 4.0])

    metrics = receipt._tensor_metrics(reference, candidate)  # pyright: ignore[reportPrivateUsage]

    assert metrics["shape_exact"] is True
    assert metrics["finite"] is True
    assert metrics["cosine_similarity"] == pytest.approx(0.8)
    assert metrics["rmse"] == pytest.approx(3.0 / 2.0**0.5)
    assert metrics["relative_rmse"] == pytest.approx(0.6)


def test_quality_floors_accept_boundaries_and_reject_each_miss() -> None:
    video = {
        "shape_exact": True,
        "finite": True,
        "cosine_similarity": 0.95,
        "relative_rmse": 0.30,
    }
    audio = {
        "shape_exact": True,
        "finite": True,
        "cosine_similarity": 0.98,
        "relative_rmse": 0.20,
    }

    assert receipt._metric_floor_failures(video, audio, 0.90) == []  # pyright: ignore[reportPrivateUsage]

    for metric, value, message in (
        ("cosine_similarity", 0.949, "video latent cosine below 0.95"),
        ("relative_rmse", 0.301, "video latent relative RMSE above 0.30"),
    ):
        changed = {**video, metric: value}
        assert message in receipt._metric_floor_failures(changed, audio, 0.90)  # pyright: ignore[reportPrivateUsage]
    for metric, value, message in (
        ("cosine_similarity", 0.979, "audio latent cosine below 0.98"),
        ("relative_rmse", 0.201, "audio latent relative RMSE above 0.20"),
    ):
        changed = {**audio, metric: value}
        assert message in receipt._metric_floor_failures(video, changed, 0.90)  # pyright: ignore[reportPrivateUsage]
    assert "decoded video SSIM below 0.90" in receipt._metric_floor_failures(  # pyright: ignore[reportPrivateUsage]
        video, audio, 0.899
    )


def test_ssim_and_duplicate_checks_distinguish_degraded_frames() -> None:
    reference = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[1, :, :, 0] = 255

    assert receipt._ssim_windows(reference, reference) == pytest.approx(1.0)  # pyright: ignore[reportPrivateUsage]
    assert receipt._ssim_windows(reference, candidate) < 0.90  # pyright: ignore[reportPrivateUsage]
    assert receipt._adjacent_duplicate_frames(reference) == [1]  # pyright: ignore[reportPrivateUsage]
    assert receipt._adjacent_duplicate_frames(candidate) == []  # pyright: ignore[reportPrivateUsage]


def test_cache_summary_counts_computed_and_skipped_blocks_by_step() -> None:
    summary = cast(
        "dict[str, Any]",
        receipt._cache_summary(  # pyright: ignore[reportPrivateUsage]
            [
                {
                    "hits": 1,
                    "misses": 1,
                    "invalidations": 0,
                    "events": [
                        {
                            "step": 0,
                            "computed_blocks": [0, 1, 2],
                            "skipped_blocks": [],
                        },
                        {"step": 1, "computed_blocks": [0], "skipped_blocks": [1, 2]},
                    ],
                }
            ],
        ),
    )

    assert summary["hits"] == 1
    assert summary["misses"] == 1
    assert summary["computed_by_block"] == {0: 2, 1: 1, 2: 1}
    assert summary["skipped_by_block"] == {1: 1, 2: 1}
    assert summary["steps"][1]["step"] == 1


def test_quality_summary_preserves_failed_profile_with_unavailable_metric() -> None:
    case = {
        "video_latent": {"cosine_similarity": None, "relative_rmse": None},
        "audio_latent": {"cosine_similarity": 0.99, "relative_rmse": 0.1},
        "decoded_video": {"ssim_8x8_data_range_1": 0.95},
    }

    summary = receipt._quality_summary([case])  # pyright: ignore[reportPrivateUsage]

    assert summary["video_latent_cosine"] == {"median": None, "worst": None}
    assert summary["video_latent_relative_rmse"] == {"median": None, "worst": None}
    assert summary["audio_latent_cosine"] == {"median": 0.99, "worst": 0.99}


def test_audio_and_spectrogram_artifacts_are_reviewable(tmp_path: Path) -> None:
    samples = np.stack(
        (
            np.linspace(-0.5, 0.5, 2048, dtype=np.float32),
            np.linspace(0.5, -0.5, 2048, dtype=np.float32),
        )
    )
    wav_path = tmp_path / "audio.wav"
    png_path = tmp_path / "spectrogram.png"

    wav = receipt._write_wav(wav_path, samples, 32_000)  # pyright: ignore[reportPrivateUsage]
    spectrogram = receipt._write_spectrogram(  # pyright: ignore[reportPrivateUsage]
        png_path, samples, samples * 0.5
    )

    with wave.open(str(wav_path), "rb") as opened:
        assert opened.getnchannels() == 2
        assert opened.getframerate() == 32_000
        assert opened.getnframes() == 2048
    assert wav["sha256"] == receipt._sha256(wav_path)  # pyright: ignore[reportPrivateUsage]
    assert spectrogram["sha256"] == receipt._sha256(png_path)  # pyright: ignore[reportPrivateUsage]
    assert png_path.is_file()


def test_side_by_side_video_round_trips_all_frames(tmp_path: Path) -> None:
    reference = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    candidate = np.zeros_like(reference)
    for index in range(3):
        reference[index, :, :, 0] = index * 60
        candidate[index, :, :, 1] = index * 70

    artifact = receipt._write_video(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "comparison.mp4", reference, candidate
    )

    assert artifact["frames"] == 3
    assert artifact["dropped_frames"] == 0
    assert artifact["duplicated_frames"] == []


def test_deliberate_invalidation_changes_conditioning_key() -> None:
    pytest.importorskip("torch")
    result = receipt._deliberate_invalidation()  # pyright: ignore[reportPrivateUsage]

    assert result["status"] == "PASS"
    assert result["change"] == "conditioning bytes"
    assert result["invalidations"] == 1
