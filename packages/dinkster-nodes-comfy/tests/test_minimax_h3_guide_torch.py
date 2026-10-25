"""MiniMax H3 guide conditioning uses the native AV streams."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import LatentStream, MultiStreamLatent
from dinkster_inference.audio import resample
from dinkster_nodes_comfy.minimax_h3_latent import MiniMaxH3AddGuide


def _latent(video_tokens: int = 7, audio_frames: int = 37) -> dict[str, object]:
    return {
        "samples": MultiStreamLatent(
            (
                LatentStream("video", torch.zeros((1, 24, video_tokens, 2, 3))),
                LatentStream("audio", torch.zeros((1, 32, 2, audio_frames))),
            )
        )
    }


class VideoVae:
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.movedim(-1, 1).unsqueeze(2)


class AudioVae:
    audio_sample_rate = 32000

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.repeat(1, 4, 1).movedim(-1, 1).unsqueeze(2)


class DefaultRateAudioVae:
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.repeat(1, 4, 1).movedim(-1, 1).unsqueeze(2)


def _positive() -> list[list[object]]:
    return [[torch.tensor([1.0]), {"keep": "metadata", "minimax_keyframes": [{"source": 1}]}]]


def test_image_audio_guide_truncates_clip_resamples_and_preserves_source() -> None:
    image = torch.arange(23, dtype=torch.float32).view(23, 1, 1, 1).expand(23, 16, 48, 4) / 255
    waveform = torch.linspace(-1, 1, 17, dtype=torch.float32).reshape(1, 1, 17)
    image_before, waveform_before = image.clone(), waveform.clone()
    positive = _positive()
    vae, audio_vae = VideoVae(), AudioVae()

    result = MiniMaxH3AddGuide.execute(
        positive=positive,
        latent=_latent(video_tokens=12),
        frame_idx=1,
        vae=vae,
        audio_vae=audio_vae,
        image=image,
        audio={"waveform": waveform, "sample_rate": 16000},
    )["positive"]

    keyframes = result[0][1]["minimax_keyframes"]
    assert keyframes[0] == {"source": 1}
    keyframe = keyframes[1]
    assert keyframe["resolved_frame_index"] == 1
    expected_image = torch.arange(22, dtype=torch.float32).view(22, 1, 1, 1, 1)
    expected_image = expected_image.expand(22, 3, 1, 32, 48) / 255
    torch.testing.assert_close(keyframe["latent"], expected_image, rtol=0, atol=0)
    expected_audio = resample(waveform, 16000, 32000).repeat(1, 1, 4).unsqueeze(2)[..., :35]
    torch.testing.assert_close(keyframe["audio_latent"], expected_audio, rtol=0, atol=0)
    assert result[0][1]["keep"] == "metadata"
    assert positive[0][1] == {"keep": "metadata", "minimax_keyframes": [{"source": 1}]}
    assert torch.equal(image, image_before) and torch.equal(waveform, waveform_before)


@pytest.mark.parametrize("frames,expected", ((1, 1), (4, 1), (5, 5), (21, 5), (22, 22)))
def test_image_only_guide_uses_valid_clip_lengths(frames: int, expected: int) -> None:
    image = torch.ones((frames, 16, 48, 3))
    vae = VideoVae()
    result = MiniMaxH3AddGuide.execute(
        positive=[[torch.tensor([1.0]), {}]], latent=_latent(), frame_idx=0, vae=vae, image=image
    )["positive"]
    assert result[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    encoded = result[0][1]["minimax_keyframes"][0]["latent"]
    assert encoded.shape == (expected, 3, 1, 32, 48)
    assert torch.equal(encoded, torch.ones_like(encoded))


def test_audio_only_uses_first_batch_and_default_sample_rate() -> None:
    waveform = torch.arange(2 * 2 * 7, dtype=torch.float32).reshape(2, 2, 7)
    audio_vae = DefaultRateAudioVae()
    result = MiniMaxH3AddGuide.execute(
        positive=[[torch.tensor([1.0]), {}]],
        latent=_latent(),
        frame_idx=-1,
        audio_vae=audio_vae,
        audio={"waveform": waveform, "sample_rate": 32000},
    )["positive"]
    keyframe = result[0][1]["minimax_keyframes"][0]
    assert keyframe["resolved_frame_index"] == 21
    assert torch.equal(keyframe["audio_latent"], waveform[:1, :, :2].unsqueeze(2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({}, "needs an image or an audio"),
        ({"image": torch.ones((1, 16, 48, 3))}, "needs the vae"),
        (
            {"audio": {"waveform": torch.ones((1, 1, 5)), "sample_rate": 32000}},
            "needs the audio_vae",
        ),
        ({"image": torch.ones((5, 16, 48, 3)), "vae": VideoVae(), "frame_idx": 21}, "does not fit"),
        ({"image": torch.ones((1, 16, 48, 3)), "vae": VideoVae(), "frame_idx": -26}, "outside"),
    ),
)
def test_guide_rejects_invalid_inputs(kwargs: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {
        "positive": [[torch.tensor([1.0]), {}]],
        "latent": _latent(),
        "frame_idx": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MiniMaxH3AddGuide.execute(**values)


def test_guide_rejects_non_minimax_latent_and_audio_past_track_end() -> None:
    with pytest.raises(ValueError, match="expects a MiniMax H3 AV latent"):
        MiniMaxH3AddGuide.execute(
            positive=[[torch.tensor([1.0]), {}]],
            latent={"samples": torch.zeros((1, 4))},
            frame_idx=0,
            vae=VideoVae(),
            image=torch.ones((1, 16, 48, 3)),
        )
    with pytest.raises(ValueError, match="expects a MiniMax H3 AV latent"):
        MiniMaxH3AddGuide.execute(
            positive=[[torch.tensor([1.0]), {}]],
            latent={
                "samples": MultiStreamLatent(
                    (
                        LatentStream("video", torch.zeros((1, 24, 7, 2, 3))),
                        LatentStream("audio", torch.zeros((1, 32, 2, 37))),
                        LatentStream("extra", torch.zeros((1, 1))),
                    )
                )
            },
            frame_idx=0,
            vae=VideoVae(),
            image=torch.ones((1, 16, 48, 3)),
        )
    with pytest.raises(ValueError, match="past the end"):
        MiniMaxH3AddGuide.execute(
            positive=[[torch.tensor([1.0]), {}]],
            latent=_latent(audio_frames=1),
            frame_idx=1,
            audio_vae=AudioVae(),
            audio={"waveform": torch.ones((1, 1, 5)), "sample_rate": 32000},
        )
