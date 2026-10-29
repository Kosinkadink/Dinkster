"""CPU conditioning/noise contracts with controlled VAE encoders.

Hand-derived from VOIDQuadmaskPreprocess/VOIDInpaintConditioning/
VOIDWarpedNoiseSource in ComfyUI comfy_extras/nodes_void.py and
WanDancerVideo in comfy_extras/nodes_wandancer.py at immutable revision
bc5f880dbdd43e426077f42ccec53bb577ffee9f. Exact inputs are below.
Quadmask thresholds 31/95/191 select 0/63/127/255, then invert /255;
dilation of the primary pixel uses a radius-one square. VOID lengths
1/9 become 5 (two latent frames), 13 remains 13 (four latent frames).
The controlled encoder selects RGB channel zero every four frames/eight
pixels; temporal matching repeats its last frame or truncates its end.
WanDancer length9 has three latent frames; the optional mask packs
[m0,m0,m0,m0,m1,m2,m3,m4,m5,m6,m7,m8] into four channels.
All expectations are independently derived, not captured F outputs.
No real-model or GPU claim.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.void_conditioning import (
    VOIDInpaintConditioning,
    VOIDQuadmaskPreprocess,
    VOIDWarpedNoiseSource,
)
from dinkster_nodes_comfy.wandancer_video import WanDancerVideo


class _VAE:
    def __init__(self, temporal: int | None = None) -> None:
        self.temporal = temporal
        self.images: list[torch.Tensor] = []

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.images.append(images.clone())
        values = images[::4, ::8, ::8, 0]
        if self.temporal is not None:
            values = values[:1].repeat(self.temporal, 1, 1)
            values = values + torch.arange(self.temporal).reshape(-1, 1, 1)
        return values.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning() -> list[list[Any]]:
    return [[torch.tensor([[3.0]]), {"label": "retained"}]]


def test_quadmask_threshold_edges_and_normalized_input_preserve_source() -> None:
    source = torch.tensor([0, 31, 32, 95, 96, 191, 192, 255], dtype=torch.float32)
    expected = (255 - torch.tensor([0, 0, 63, 63, 127, 127, 255, 255])) / 255
    original = source.clone()
    for mask in (source.reshape(1, 2, 4), (source / 255).reshape(1, 2, 4)):
        result = VOIDQuadmaskPreprocess.execute(mask=mask)["quadmask"]
        torch.testing.assert_close(result.flatten(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_quadmask_dilation_is_spatial_and_keeps_other_frames() -> None:
    mask = torch.full((2, 4, 5), 255.0)
    mask[0, 1, 2] = 63
    expected = torch.zeros_like(mask)
    expected[0, :3, 1:4] = 1
    torch.testing.assert_close(
        VOIDQuadmaskPreprocess.execute(mask=mask, dilate_width=1)["quadmask"],
        expected,
        rtol=0,
        atol=0,
    )
    assert mask[0, 1, 2] == 63 and torch.count_nonzero(mask == 63) == 1


@pytest.mark.parametrize(("length", "frames", "latent_t"), ((1, 5, 2), (9, 5, 2), (13, 13, 4)))
@pytest.mark.parametrize("encoded_t", (1, 6))
def test_void_conditioning_rounds_length_and_matches_encoder_time(
    length: int, frames: int, latent_t: int, encoded_t: int
) -> None:
    video = torch.arange(13 * 16 * 24 * 4, dtype=torch.float32).reshape(13, 16, 24, 4) / 20000
    mask = torch.full((13, 16, 24), 0.25)
    original = video.clone()
    positive, negative = _conditioning(), _conditioning()
    vae = _VAE(encoded_t)
    result = VOIDInpaintConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        video=video,
        quadmask=mask,
        width=24,
        height=16,
        length=length,
        batch_size=2,
    )
    torch.testing.assert_close(vae.images[0], torch.full((frames, 16, 24, 3), 0.75))
    torch.testing.assert_close(vae.images[1], video[:frames, ..., :3] * 0.75)
    steps = torch.arange(latent_t).clamp(max=encoded_t - 1).reshape(1, 1, -1, 1, 1)
    expected_mask = (0.75 + steps).expand(1, 16, latent_t, 2, 3)
    expected_video = video[0, ::8, ::8, 0].reshape(1, 1, 1, 2, 3) * 0.75 + steps
    expected = torch.cat((expected_mask, expected_video.expand(1, 16, latent_t, 2, 3)), dim=1)
    for label in ("positive", "negative"):
        assert result[label][0][1]["label"] == "retained"
        torch.testing.assert_close(result[label][0][1]["concat_latent_image"], expected)
    assert result["latent"]["samples"].shape == (2, 16, latent_t, 2, 3)
    assert torch.count_nonzero(result["latent"]["samples"]) == 0
    assert positive[0][1] == negative[0][1] == {"label": "retained"}
    torch.testing.assert_close(video, original, rtol=0, atol=0)
    assert torch.all(mask == 0.25)


def test_warped_noise_source_returns_independent_cpu_noise() -> None:
    source = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4)
    noise = VOIDWarpedNoiseSource.execute(warped_noise={"samples": source})["NOISE"]
    first = noise.generate_noise({"samples": torch.ones_like(source)})
    assert noise.seed == 0 and first.device.type == "cpu"
    torch.testing.assert_close(first, source, rtol=0, atol=0)
    first.zero_()
    torch.testing.assert_close(noise.generate_noise(None), source, rtol=0, atol=0)
    assert source.flatten()[-1] == 47


@pytest.mark.parametrize("with_mask", (False, True))
def test_wandancer_packs_mask_and_preserves_audio_vision_and_source(with_mask: bool) -> None:
    positive, negative = _conditioning(), _conditioning()
    images = torch.full((2, 16, 24, 4), 0.375)
    mask = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1).expand(9, 16, 24) / 8
    vision = {"penultimate_hidden_states": torch.tensor([[[4.0]]])}
    reference = {"penultimate_hidden_states": torch.tensor([[[7.0]]])}
    features = torch.tensor([[[2.0, -1.0]]])
    vae = _VAE()
    result = WanDancerVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=24,
        height=16,
        length=9,
        start_image=images,
        mask=mask if with_mask else None,
        clip_vision_output=vision,
        clip_vision_output_ref=reference,
        audio_encoder_output={"audio_feature": features, "fps": 15.0, "audio_inject_scale": 0.375},
    )
    expected_image = torch.zeros(9, 16, 24, 3)
    expected_image[:2] = 0.375
    torch.testing.assert_close(vae.images[0], expected_image, rtol=0, atol=0)
    expected_mask = (
        (
            1 - torch.tensor([[0, 1, 5], [0, 2, 6], [0, 3, 7], [0, 4, 8]]) / 8
            if with_mask
            else torch.tensor([[0.0, 1.0, 1.0]])
        )
        .reshape(1, 4 if with_mask else 1, 3, 1, 1)
        .expand(-1, -1, -1, 2, 3)
    )
    for label in ("positive", "negative"):
        meta = result[label][0][1]
        torch.testing.assert_close(meta["concat_mask"], expected_mask, rtol=0, atol=0)
        assert meta["fps"] == 15 and meta["audio_inject_scale"] == 0.375
        torch.testing.assert_close(meta["audio_embed"], features, rtol=0, atol=0)
        torch.testing.assert_close(
            meta["clip_vision_output"].penultimate_hidden_states,
            vision["penultimate_hidden_states"],
        )
        torch.testing.assert_close(
            meta["clip_vision_output_ref"].penultimate_hidden_states,
            reference["penultimate_hidden_states"],
        )
    assert result["latent"]["samples"].shape == (1, 16, 3, 2, 3)
    assert positive[0][1] == negative[0][1] == {"label": "retained"}
    assert torch.all(images == 0.375)
