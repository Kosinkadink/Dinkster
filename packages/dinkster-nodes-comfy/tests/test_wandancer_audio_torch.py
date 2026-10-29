"""CPU contracts for WanDancer audio feature encoding."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    schema_to_wire,
)
from dinkster_nodes_comfy.wandancer_audio import WanDancerEncodeAudio


def _waveform(sample_rate: int, seconds: int, channels: int = 1) -> torch.Tensor:
    samples = sample_rate * seconds
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    left = 0.3 * torch.sin(2 * torch.pi * 220 * time) + 0.1 * torch.sin(2 * torch.pi * 330 * time)
    right = 0.2 * torch.sin(2 * torch.pi * 440 * time) + (torch.arange(samples) % 997 < 19) * 0.4
    return torch.stack((left, right)[:channels]).unsqueeze(0)


def _encode(
    waveform: torch.Tensor, sample_rate: int, video_frames: int, scale: float
) -> dict[str, Any]:
    return dict(
        WanDancerEncodeAudio.execute(
            audio={"waveform": waveform, "sample_rate": sample_rate},
            video_frames=video_frames,
            audio_inject_scale=scale,
        )
    )


def test_schema_matches_fixed_comfyui_node() -> None:
    expected = NodeSchema(
        node_type="WanDancerEncodeAudio",
        category="model/conditioning/wan/dancer",
        inputs=(
            InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
            InputSpec(
                "video_frames",
                TypeExpr.concrete(CORE_INT),
                default=149,
                widget=NumberWidget(min=1, max=16384, step=4),
            ),
            InputSpec(
                "audio_inject_scale",
                TypeExpr.concrete(CORE_FLOAT),
                default=1.0,
                widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                doc="The scale for the audio features when injected into the video model.",
            ),
        ),
        outputs=(
            OutputSpec(
                "audio_encoder_output",
                TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE),
                display_name="audio_encoder_output",
            ),
            OutputSpec(
                "fps_string",
                TypeExpr.concrete(CORE_STRING),
                display_name="fps_string",
                doc=(
                    "The calculated fps based on the audio length and "
                    "the number of video frames. "
                    "Used in the prompt."
                ),
            ),
        ),
    )
    assert schema_to_wire(WanDancerEncodeAudio.define_schema()) == schema_to_wire(expected)


def test_stereo_first_batch_feature_order_preservation_and_non_thirty_fps() -> None:
    # ComfyUI WanDancerEncodeAudio at bc5f880dbdd43e426077f42ccec53bb577ffee9f,
    # comfy_extras/nodes_wandancer.py:711-782, exact input: 4 s asymmetric stereo,
    # 22050 Hz, 59 video frames, scale 0.375.
    stereo = _waveform(22050, 4, channels=2)
    source = torch.cat((stereo, torch.full_like(stereo, -9.0)))
    before = source.clone()
    mixed = _encode(source, 22050, 59, 0.375)
    mono = _encode(stereo[:, 0].add(stereo[:, 1]).div(2).unsqueeze(1), 22050, 59, 0.375)

    feature = mixed["audio_encoder_output"]["audio_feature"]
    assert feature.shape == (1, 121, 35)
    assert feature.dtype == torch.float32 and feature.device.type == "cpu"
    torch.testing.assert_close(
        feature, mono["audio_encoder_output"]["audio_feature"], rtol=0, atol=0
    )
    assert feature[:, :, :21].abs().sum() > 0
    assert feature[:, :, 21:33].abs().sum() > 0
    assert set(feature[:, :, 33].unique().tolist()) <= {0.0, 1.0}
    assert set(feature[:, :, 34].unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(source, before)
    assert mixed["audio_encoder_output"]["audio_inject_scale"] == 0.375
    assert mixed["audio_encoder_output"]["fps"] == 15.0
    assert mixed["fps_string"] == " \u5e27\u7387\u662f15.0000"


def test_mono_thirty_fps_prompt_and_injection_scale() -> None:
    # ComfyUI WanDancerEncodeAudio at bc5f880dbdd43e426077f42ccec53bb577ffee9f,
    # comfy_extras/nodes_wandancer.py:711-782, exact input: 5 s mono, 24000 Hz,
    # 149 video frames, scale 1.25.
    result = _encode(_waveform(24000, 5), 24000, 149, 1.25)
    output = result["audio_encoder_output"]
    feature = output["audio_feature"]
    assert feature.shape == (1, 151, 35)
    assert output["fps"] == 30.0
    assert output["audio_inject_scale"] == 1.25
    assert result["fps_string"] == ", \u5e27\u7387\u662f30fps\u3002"


@pytest.mark.parametrize(
    ("feature_frames", "expected_fps", "expected_prompt"),
    (
        (8, 30.0, ", \u5e27\u7387\u662f30fps\u3002"),
        (9, 15.0, " \u5e27\u7387\u662f15.0000"),
        (10, 15.0, " \u5e27\u7387\u662f15.0000"),
        (15, 10.0, " \u5e27\u7387\u662f10.0000"),
    ),
)
def test_feature_column_order_and_half_up_fps(
    monkeypatch: pytest.MonkeyPatch,
    feature_frames: int,
    expected_fps: float,
    expected_prompt: str,
) -> None:
    # Hand-derived controlled features exercise the public node's assembly, not DSP parity.
    # With six video frames, 9/6 and 15/6 distinguish half-up from truncation and ties-to-even.
    from dinkster_nodes_comfy import wandancer_audio_features as features

    monkeypatch.setattr(features, "quick_tempo_estimate", lambda *a, **k: 120.0)
    monkeypatch.setattr(
        features, "_compute_mel_spectrogram", lambda *a, **k: np.zeros((128, feature_frames))
    )
    monkeypatch.setattr(features, "power_to_db", lambda mel, **k: mel)
    monkeypatch.setattr(
        features,
        "compute_onset_envelope",
        lambda *a, **k: np.arange(feature_frames, dtype=np.float32) + 1,
    )
    monkeypatch.setattr(
        features,
        "compute_mfcc",
        lambda *a, **k: np.arange(feature_frames * 20, dtype=np.float32).reshape(-1, 20) + 100,
    )
    monkeypatch.setattr(
        features,
        "compute_chroma_cens",
        lambda *a, **k: np.arange(feature_frames * 12, dtype=np.float32).reshape(12, -1) + 500,
    )
    monkeypatch.setattr(
        features, "detect_onset_peaks", lambda *a, **k: np.array([1, feature_frames - 2])
    )
    monkeypatch.setattr(features, "estimate_tempo_from_onset", lambda *a, **k: 120.0)
    monkeypatch.setattr(features, "track_beats", lambda *a, **k: np.array([0, feature_frames - 1]))

    result = _encode(torch.zeros((1, 1, 512)), 15360, 6, 0.625)
    expected = torch.tensor(
        [
            [
                row + 1,
                *(100 + row * 20 + channel for channel in range(20)),
                *(500 + channel * feature_frames + row for channel in range(12)),
                int(row in (1, feature_frames - 2)),
                int(row in (0, feature_frames - 1)),
            ]
            for row in range(feature_frames)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    output = result["audio_encoder_output"]
    torch.testing.assert_close(output["audio_feature"], expected, rtol=0, atol=0)
    assert output["fps"] == expected_fps
    assert output["audio_inject_scale"] == 0.625
    assert result["fps_string"] == expected_prompt


def test_node_discovery_does_not_load_torch_or_scipy() -> None:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_nodes_comfy.wandancer_audio; "
            "assert 'torch' not in sys.modules and 'scipy' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
