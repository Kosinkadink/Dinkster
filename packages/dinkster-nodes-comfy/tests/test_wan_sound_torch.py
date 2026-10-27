"""CPU output contracts for Wan sound and HuMo conditioning."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_api.v1 import AUDIO_ENCODER_OUTPUT_TYPE, TypeRegistry, register_inference_types
from dinkster_nodes_comfy.wan_sound import (
    WanHuMoImageToVideo,
    WanSoundImageToVideo,
    WanSoundImageToVideoExtend,
)


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning() -> list[list[Any]]:
    return [[torch.tensor([[2.0, 7.0]]), {"reference_latents": [torch.tensor([11.0])]}]]


def test_sound_public_audio_layers_reference_append_and_source_preservation() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    audio = {
        "encoded_audio": torch.ones(1, 50, 3),
        "encoded_audio_all_layers": (torch.full((1, 50, 3), 2.0), torch.full((1, 50, 3), 7.0)),
        "audio_samples": 16000,
    }
    spec = registry.spec(AUDIO_ENCODER_OUTPUT_TYPE)
    encoded = spec.decode(spec.encode(audio))
    image = torch.full((2, 16, 24, 4), 0.75)
    positive, negative = _conditioning(), _conditioning()
    result = WanSoundImageToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=24,
        height=16,
        length=5,
        batch_size=2,
        audio_encoder_output=encoded,
        ref_image=image,
    )
    metadata = result["positive"][0][1]
    assert metadata["audio_embed"].shape == (1, 2, 3, 8)
    assert torch.equal(metadata["audio_embed"][0, :, :, 0], torch.tensor([[2.0] * 3, [7.0] * 3]))
    assert not result["negative"][0][1]["audio_embed"].count_nonzero()
    assert len(metadata["reference_latents"]) == 2
    assert metadata["reference_latents"][0].item() == 11
    assert torch.all(metadata["reference_latents"][1] == 0.75)
    assert len(positive[0][1]["reference_latents"]) == 1
    assert "audio_embed" not in negative[0][1]
    assert result["latent"]["samples"].shape == (2, 16, 2, 2, 3)
    assert torch.all(image == 0.75)
    assert torch.all(audio["encoded_audio_all_layers"][1] == 7)


def test_sound_motion_padding_and_control_temporal_placement() -> None:
    vae = VAE()
    motion = torch.full((5, 16, 24, 3), 0.25)
    control = torch.stack([torch.full((16, 24, 4), value) for value in (0.1, 0.2, 0.3, 0.4, 0.9)])
    result = WanSoundImageToVideo.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=vae,
        width=24,
        height=16,
        length=9,
        batch_size=1,
        ref_motion=motion,
        control_video=control,
    )
    assert vae.images[0].shape == (73, 16, 24, 3)
    assert torch.all(vae.images[0][:68] == 0.5)
    assert torch.all(vae.images[0][68:] == 0.25)
    metadata = result["positive"][0][1]
    assert metadata["reference_motion"].shape[2] == 19
    assert torch.all(metadata["reference_motion"][:, :, :17] == 0.5)
    assert torch.all(metadata["reference_motion"][:, :, 17:] == 0.25)
    assert torch.all(metadata["control_video"][:, :, 0] == 0.1)
    assert torch.all(metadata["control_video"][:, :, 1] == 0.9)
    assert torch.all(motion == 0.25)


def test_sound_extend_uses_latent_dimensions_tail_and_audio_offset() -> None:
    samples = torch.arange(2 * 16 * 21 * 2 * 3, dtype=torch.float32).reshape(2, 16, 21, 2, 3)
    before = samples.clone()
    ramp = torch.arange(300, dtype=torch.float32).reshape(1, 300, 1).expand(1, 300, 2)
    audio = {
        "encoded_audio_all_layers": [ramp, ramp + 1000],
    }
    result = WanSoundImageToVideoExtend.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=VAE(),
        length=5,
        video_latent={"samples": samples},
        audio_encoder_output=audio,
    )
    metadata = result["positive"][0][1]
    assert result["latent"]["samples"].shape == (2, 16, 2, 2, 3)
    assert torch.equal(metadata["reference_motion"], samples[:, :, -19:])
    assert metadata["audio_embed"].shape == (1, 2, 2, 8)
    # Offset 84 maps to source bucket index round(84 * 30 / 16) = 158.
    torch.testing.assert_close(
        metadata["audio_embed"][0, 0, :, 0], torch.full((2,), 158 * 299 / 179)
    )
    torch.testing.assert_close(
        metadata["audio_embed"][0, 1, :, 0], torch.full((2,), 1000 + 158 * 299 / 179)
    )
    torch.testing.assert_close(
        metadata["audio_embed"][0, 0, :, -1], torch.full((2,), 171 * 299 / 179)
    )
    assert torch.equal(samples, before)


def test_humo_absent_inputs_keep_reference_list_and_causal_grid() -> None:
    result = WanHuMoImageToVideo.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=VAE(),
        width=24,
        height=16,
        length=6,
        batch_size=2,
    )
    assert result["latent"]["samples"].shape == (2, 16, 2, 2, 3)
    metadata = result["positive"][0][1]
    assert metadata["audio_embed"].shape == (2, 3, 8, 5, 1280)
    assert not metadata["audio_embed"].count_nonzero()
    assert metadata["reference_latents"][0].item() == 11
    assert metadata["reference_latents"][1].shape == (2, 16, 1, 2, 3)


def test_humo_layer_groups_audio_windows_and_negative_reference() -> None:
    audio = {
        "encoded_audio_all_layers": [torch.full((1, 12, 3), float(i)) for i in range(33)],
        "audio_samples": 640 * 6,
    }
    image = torch.full((1, 16, 24, 3), 0.625)
    result = WanHuMoImageToVideo.execute(
        positive=_conditioning(),
        negative=_conditioning(),
        vae=VAE(),
        width=24,
        height=16,
        length=5,
        batch_size=1,
        audio_encoder_output=audio,
        ref_image=image,
    )
    positive, negative = result["positive"][0][1], result["negative"][0][1]
    embedding = positive["audio_embed"]
    assert embedding.shape == (1, 2, 8, 5, 3)
    assert not embedding[0, 0, :5].count_nonzero()
    assert torch.equal(embedding[0, 0, 5, :, 0], torch.tensor([3.5, 11.5, 19.5, 27.5, 32.0]))
    assert not embedding[0, 1, 0].count_nonzero()
    assert not embedding[0, 1, -1].count_nonzero()
    assert torch.equal(embedding[0, 1, 1, :, 2], torch.tensor([3.5, 11.5, 19.5, 27.5, 32.0]))
    assert torch.all(positive["reference_latents"][-1] == 0.625)
    assert not negative["reference_latents"][-1].count_nonzero()
    assert not negative["audio_embed"].count_nonzero()
    assert torch.all(image == 0.625)
    assert torch.all(audio["encoded_audio_all_layers"][32] == 32)
