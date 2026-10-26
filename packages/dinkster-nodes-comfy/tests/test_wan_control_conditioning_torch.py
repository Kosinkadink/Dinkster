"""CPU contracts for native Wan control conditioning."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_nodes_comfy.wan_control_conditioning import (
    Wan22FunControlToVideo,
    WanFunControlToVideo,
    WanPhantomSubjectToVideo,
    WanVaceToVideo,
)


class VAE:
    def __init__(self, channels: int = 16, scale: int = 8) -> None:
        self.latent_channels = channels
        self.scale = scale
        self.images: list[torch.Tensor] = []

    def spacial_compression_encode(self) -> int:
        return self.scale

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return (
            image[::4, :: self.scale, :: self.scale, :1]
            .movedim(-1, 0)
            .unsqueeze(0)
            .repeat(1, self.latent_channels, 1, 1, 1)
        )


def _conditioning(label: str, **metadata: object) -> list[list[Any]]:
    return [[torch.tensor([[1.0]]), {"label": label, **metadata}]]


def _image(values: list[float], channels: int = 4) -> torch.Tensor:
    return torch.stack([torch.full((8, 16, channels), value) for value in values])


def test_fun_control_uses_asymmetric_start_and_control_halves() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    vae = VAE()
    output = WanFunControlToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=8,
        length=9,
        batch_size=2,
        start_image=_image([2.0]),
        control_video=_image([7.0, 8.0]),
        clip_vision_output={"image_embeds": torch.tensor([[3.0]])},
    )
    concat = output["positive"][0][1]["concat_latent_image"]
    assert concat.shape == (2, 32, 3, 1, 2)
    torch.testing.assert_close(concat[:, :16, :1], torch.full((2, 16, 1, 1, 2), 7.0))
    torch.testing.assert_close(concat[:, 16:, :1], torch.full((2, 16, 1, 1, 2), 2.0))
    torch.testing.assert_close(
        output["positive"][0][1]["clip_vision_output"].image_embeds, torch.tensor([[3.0]])
    )
    assert positive == _conditioning("positive") and negative == _conditioning("negative")
    assert output["latent"]["samples"].shape == (2, 16, 3, 1, 2)


def test_wan22_uses_vae_layout_and_appends_references_without_mutating_sources() -> None:
    positive = _conditioning("positive", reference_latents=["before"])
    negative = _conditioning("negative", reference_latents=["before"])
    ref, start, control = _image([3.0]), _image([5.0]), _image([9.0, 10.0])
    ref_before, start_before, control_before = ref.clone(), start.clone(), control.clone()
    vae = VAE(channels=48, scale=16)
    output = Wan22FunControlToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=32,
        height=16,
        length=9,
        batch_size=1,
        ref_image=ref,
        start_image=start,
        control_video=control,
    )
    metadata = output["positive"][0][1]
    concat = metadata["concat_latent_image"]
    assert output["latent"]["samples"].shape == (1, 48, 3, 1, 2)
    assert concat.shape == (1, 96, 3, 1, 2)
    torch.testing.assert_close(concat[:, :48, :1], torch.full((1, 48, 1, 1, 2), 9.0))
    torch.testing.assert_close(concat[:, 48:, :1], torch.full((1, 48, 1, 1, 2), 5.0))
    assert metadata["concat_mask_index"] == 48
    assert metadata["reference_latents"][0] == "before"
    torch.testing.assert_close(metadata["reference_latents"][1], torch.full((1, 48, 1, 1, 2), 3.0))
    assert metadata["concat_mask"].shape == (1, 4, 3, 1, 2)
    assert not metadata["concat_mask"][:, :, 0].count_nonzero()
    assert positive[0][1]["reference_latents"] == ["before"]
    assert negative[0][1]["reference_latents"] == ["before"]
    assert torch.equal(ref, ref_before)
    assert torch.equal(start, start_before)
    assert torch.equal(control, control_before)


def test_wan22_normalizes_non_48_channel_zero_control() -> None:
    output = Wan22FunControlToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=VAE(channels=16, scale=8),
        width=16,
        height=8,
        length=1,
        batch_size=1,
    )
    concat = output["positive"][0][1]["concat_latent_image"]
    assert concat.shape == (1, 32, 1, 1, 2)
    assert not torch.equal(concat, torch.zeros_like(concat))


def test_vace_pads_short_video_and_mask_and_preserves_existing_metadata() -> None:
    positive = _conditioning(
        "positive", vace_frames=["old"], vace_mask=["old"], vace_strength=[0.5]
    )
    negative = _conditioning(
        "negative", vace_frames=["old"], vace_mask=["old"], vace_strength=[0.5]
    )
    video, reference = _image([0.0, 1.0]), _image([4.0])
    mask = torch.zeros((1, 1, 8, 16))
    mask[:, :, :, 8:] = 1.0
    vae = VAE()
    output = WanVaceToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=8,
        length=5,
        batch_size=2,
        strength=2.0,
        control_video=video,
        control_masks=mask,
        reference_image=reference,
    )
    metadata = output["positive"][0][1]
    assert output["trim_latent"] == 1
    assert output["latent"]["samples"].shape == (2, 16, 3, 1, 2)
    assert metadata["vace_frames"][0] == "old"
    assert metadata["vace_strength"] == [0.5, 2.0]
    frames, latent_mask = metadata["vace_frames"][1], metadata["vace_mask"][1]
    assert frames.shape == (1, 32, 3, 1, 2)
    assert latent_mask.shape == (1, 64, 3, 1, 2)
    assert not latent_mask[:, :, 0].count_nonzero()
    assert latent_mask[:, :, 1:].count_nonzero()
    assert positive[0][1]["vace_frames"] == ["old"]
    assert negative[0][1]["vace_strength"] == [0.5]


def test_vace_preserves_nchw_mask_axis_when_resizing() -> None:
    mask = torch.zeros((2, 1, 8, 16))
    mask[0, :, :, 8:] = 1.0
    output = WanVaceToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=VAE(),
        width=32,
        height=8,
        length=5,
        batch_size=1,
        strength=1.0,
        control_video=_image([0.0, 1.0]),
        control_masks=mask,
    )
    frames = output["positive"][0][1]["vace_frames"][0]
    torch.testing.assert_close(frames[:, :16, 0, :, 0], torch.zeros((1, 16, 1)))
    torch.testing.assert_close(frames[:, :16, 0, :, -1], torch.full((1, 16, 1), 0.5))
    torch.testing.assert_close(frames[:, 16:, 0, :, 0], torch.full((1, 16, 1), 0.5))
    torch.testing.assert_close(frames[:, 16:, 0, :, -1], torch.zeros((1, 16, 1)))


def test_vace_defaults_and_phantom_image_paths() -> None:
    vace = WanVaceToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=VAE(),
        width=16,
        height=8,
        length=1,
        batch_size=1,
        strength=1.0,
    )
    assert vace["trim_latent"] == 0
    assert vace["positive"][0][1]["vace_mask"][0].shape == (1, 64, 1, 1, 2)
    assert vace["positive"][0][1]["vace_mask"][0].count_nonzero()

    positive, negative = _conditioning("positive"), _conditioning("negative")
    images = _image([3.0, 6.0])
    vae = VAE()
    phantom = WanPhantomSubjectToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=8,
        length=9,
        batch_size=1,
        images=images,
    )
    positive_concat = phantom["positive"][0][1]["time_dim_concat"]
    negative_text = phantom["negative_text"][0][1]["time_dim_concat"]
    negative_image = phantom["negative_img_text"][0][1]["time_dim_concat"]
    torch.testing.assert_close(positive_concat, negative_text)
    assert not torch.equal(positive_concat, negative_image)
    torch.testing.assert_close(positive_concat[:, :, 0], torch.full((1, 16, 1, 2), 3.0))
    torch.testing.assert_close(positive_concat[:, :, 1], torch.full((1, 16, 1, 2), 6.0))
    no_images = WanPhantomSubjectToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=16,
        height=8,
        length=1,
        batch_size=1,
    )
    assert no_images["positive"] is positive
    assert no_images["negative_text"] is negative
    assert no_images["negative_img_text"] is negative
