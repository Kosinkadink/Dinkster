"""LTXV AV latent role, mask, and audio-fitting behavior."""

from __future__ import annotations

import torch
from dinkster_api.v1 import MultiStreamLatent
from dinkster_nodes_comfy.ltx_av_latent import LTXVConcatAVLatent, LTXVSeparateAVLatent


def _latent(samples: object, **metadata: object) -> dict[str, object]:
    return {"samples": samples, **metadata}


def test_concat_and_separate_preserve_explicit_roles_metadata_and_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "dinkster_nodes_comfy.ltx_av_latent.initialize_inference_process", lambda: None
    )
    video = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3)
    audio = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    video_mask = torch.zeros_like(video)
    output = LTXVConcatAVLatent.execute(
        video_latent=_latent(video, noise_mask=video_mask, owner="video", shared="video"),
        audio_latent=_latent(audio, owner="audio", shared="audio"),
    )["LATENT"]
    streams = output["samples"]
    assert type(streams) is MultiStreamLatent
    assert streams.roles == ("video", "audio")
    assert streams.by_role("video") is video and streams.by_role("audio") is audio
    assert output["owner"] == "audio" and output["shared"] == "audio"
    masks = output["noise_mask"]
    assert masks.roles == ("video", "audio")
    assert torch.equal(masks.by_role("video"), video_mask)
    assert torch.equal(masks.by_role("audio"), torch.ones_like(audio))

    split = LTXVSeparateAVLatent.execute(av_latent=output)
    assert split["VIDEO_LATENT"]["samples"] is video
    assert split["AUDIO_LATENT"]["samples"] is audio
    assert split["VIDEO_LATENT"]["noise_mask"] is masks.by_role("video")
    assert split["AUDIO_LATENT"]["noise_mask"] is masks.by_role("audio")


def test_concat_replaces_audio_by_role_and_pads_mask_tail(monkeypatch) -> None:
    monkeypatch.setattr(
        "dinkster_nodes_comfy.ltx_av_latent.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr("dinkster_inference.utils.reshape_mask", lambda mask, _shape: mask)
    video = torch.full((1, 2, 2, 3), 9.0, dtype=torch.float64)
    previous_audio = torch.full((1, 2, 6), 4.0, dtype=torch.float32)
    new_audio = torch.tensor(
        [[[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]]], dtype=torch.float32
    )
    audio_mask = torch.zeros_like(new_audio)
    av = _latent(MultiStreamLatent.from_pairs((("audio", previous_audio), ("video", video))))
    output = LTXVConcatAVLatent.execute(
        video_latent=av, audio_latent=_latent(new_audio, noise_mask=audio_mask)
    )["LATENT"]
    streams = output["samples"]
    assert streams.roles == ("video", "audio")
    assert streams.by_role("video") is video
    assert torch.equal(
        streams.by_role("audio"),
        torch.tensor([[[7.0, 8.0, 9.0, 10.0, 0.0, 0.0], [11.0, 12.0, 13.0, 14.0, 0.0, 0.0]]]),
    )
    assert torch.equal(
        output["noise_mask"].by_role("audio"),
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]]),
    )
