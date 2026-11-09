"""MiniMax reference presentation and encoded payload contracts."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import MultiStreamLatent
from dinkster_inference.audio import resample
from dinkster_nodes_comfy.minimax_h3_reference import MiniMaxH3ReferenceToVideo


class Clip:
    def tokenize(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        return [[torch.tensor([7.0]), {"tokens": tokens, "keep": "metadata"}]]


class VideoVae:
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.permute(3, 0, 1, 2).unsqueeze(0)


class AudioVae:
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.movedim(-1, 1).unsqueeze(2)


def test_mixed_references_pair_soundtracks_by_suffix_and_preserve_sources() -> None:
    image = torch.arange(2.0).view(2, 1, 1, 1).expand(2, 32, 64, 4) / 8
    video = torch.arange(41.0).view(41, 1, 1, 1).expand(41, 32, 64, 4) / 64
    waveform = torch.arange(2 * 2 * 11.0).reshape(2, 2, 11) / 64
    before = [value.clone() for value in (image, video, waveform)]
    audio = {"waveform": waveform, "sample_rate": 16000}
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=Clip(),
        prompt="ordered references",
        width=64,
        height=32,
        length=23,
        vae=VideoVae(),
        audio_vae=AudioVae(),
        ref_images={"ref_image_8": image, "ref_image_1": None},
        ref_videos={"ref_video_2": video, "ref_video_1": video[:5] / 2},
        ref_video_audios={"ref_video_audio_1": None, "ref_video_audio_2": audio},
        ref_audios={"ref_audio_3": audio},
    )
    metadata = result["positive"][0][1]
    items = metadata["tokens"]["minimax_ref_items"]
    assert [item["type"] for item in items] == ["image", "audio", "video", "video", "audio"]
    assert metadata["tokens"]["prompt"] == "ordered references"
    assert metadata["keep"] == "metadata"
    assert torch.equal(items[0]["data"], image[:1, ..., :3])
    quantized = (video[..., :3] * 255).to(torch.uint8).float() / 255
    assert torch.equal(items[2]["data"], quantized[[0, 12, 24, 36]])
    assert items[2]["timestamps"] == [0.0, 0.5, 1.0, 1.5]
    assert torch.equal(items[3]["data"], video[:1, ..., :3] / 2)
    blocks = metadata["minimax_refs"]
    assert [block["kind"] for block in blocks] == ["image", "video_audio", "video", "audio"]
    assert (blocks[1]["latent_t"], blocks[1]["latent_h"], blocks[1]["latent_w"]) == (39, 2, 4)
    assert torch.equal(blocks[1]["latent"], quantized[:39].permute(3, 0, 1, 2).unsqueeze(0))
    assert blocks[2]["ref_audio_t"] == 0 and blocks[2]["audio_latent"] is None
    expected_audio = resample(waveform, 16000, 32000)[:1].unsqueeze(2)
    assert torch.equal(blocks[1]["audio_latent"], expected_audio)
    assert torch.equal(blocks[3]["audio_latent"], expected_audio)
    assert blocks[1]["ref_audio_t"] == blocks[3]["ref_audio_t"] == expected_audio.shape[-1]
    latent = result["LATENT"]["samples"]
    assert isinstance(latent, MultiStreamLatent)
    assert latent.by_role("video").shape == (1, 24, 12, 2, 4)
    assert latent.by_role("audio").shape == (1, 32, 2, 65)
    assert all(
        torch.equal(value, original)
        for value, original in zip((image, video, waveform), before, strict=False)
    )


@pytest.mark.parametrize("video_vae,audio_vae", ((False, False), (True, False), (False, True)))
def test_optional_vaes_keep_reference_labels_without_fabricating_payloads(
    video_vae: bool,
    audio_vae: bool,
) -> None:
    frames = torch.ones((22, 32, 64, 3))
    audio = {"waveform": torch.ones((1, 1, 7)), "sample_rate": 32000}
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=Clip(),
        prompt="refs",
        width=64,
        height=32,
        length=5,
        vae=VideoVae() if video_vae else None,
        audio_vae=AudioVae() if audio_vae else None,
        ref_images={"ref_image_1": frames[:1]},
        ref_videos={"ref_video_3": frames},
        ref_video_audios={"ref_video_audio_3": audio},
        ref_audios={"ref_audio_2": audio},
    )["positive"]
    metadata = result[0][1]
    items = metadata["tokens"]["minimax_ref_items"]
    assert [item["type"] for item in items] == ["image", "audio", "video", "audio"]
    assert items[2]["data"].shape == (1, 32, 64, 3)
    assert items[2]["timestamps"] == [0.0]
    if not video_vae and not audio_vae:
        assert "minimax_refs" not in metadata
    else:
        assert [block["kind"] for block in metadata["minimax_refs"]] == (
            ["image", "video"] if video_vae else ["audio"]
        )


@pytest.mark.parametrize("frames,expected", ((5, 5), (21, 5), (22, 22), (38, 22), (39, 39)))
def test_reference_video_rounds_down_to_valid_clip_lengths(frames: int, expected: int) -> None:
    video = torch.arange(float(frames)).view(frames, 1, 1, 1).expand(frames, 32, 64, 3) / 64
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=Clip(),
        prompt="clip",
        width=64,
        height=32,
        length=124,
        vae=VideoVae(),
        ref_videos={"ref_video_1": video},
    )["positive"]
    block = result[0][1]["minimax_refs"][0]
    assert block["latent_t"] == expected
    quantized = (video[:expected] * 255).to(torch.uint8).float() / 255
    assert torch.equal(block["latent"], quantized.permute(3, 0, 1, 2).unsqueeze(0))


@pytest.mark.parametrize("mode,shape", (("match", (32, 64)), ("max", (64, 128))))
def test_image_size_modes_preserve_aspect_and_first_image(
    mode: str, shape: tuple[int, int]
) -> None:
    image = torch.full((2, 64, 128, 4), 0.25)
    image[1] = 0.75
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=Clip(),
        prompt="image",
        width=64,
        height=32,
        length=5,
        ref_image_size=mode,
        ref_images={"ref_image_1": image},
    )["positive"]
    pixels = result[0][1]["tokens"]["minimax_ref_items"][0]["data"]
    assert pixels.shape == (1, *shape, 3)
    assert torch.equal(pixels, torch.full_like(pixels, 63 / 255))


def test_reference_video_rejects_short_clip_without_mutating_input() -> None:
    frames = torch.ones((4, 32, 64, 3))
    with pytest.raises(ValueError, match="at least 5 frames"):
        MiniMaxH3ReferenceToVideo.execute(
            clip=Clip(),
            prompt="short",
            width=64,
            height=32,
            length=5,
            ref_videos={"ref_video_1": frames},
        )
    assert torch.equal(frames, torch.ones_like(frames))
