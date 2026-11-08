from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from dinkster_sd.stable3d import (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)


class FakeClipVision:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode_image(self, image: torch.Tensor) -> SimpleNamespace:
        self.images.append(image.clone())
        return SimpleNamespace(image_embeds=image.mean(dim=(1, 2)))


class FakeVAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image.mean(-1, keepdim=True).movedim(-1, 1)


def _image() -> np.ndarray:
    image = np.zeros((1, 35, 51, 4), dtype=np.float32)
    image[..., 0] = 0.25
    image[..., 1] = 0.5
    image[..., 2] = 0.75
    image[..., 3] = 0.9
    return image


def test_zero123_conditioning_preserves_source_and_negative_shape() -> None:
    image = _image()
    before = image.copy()
    clip_vision = FakeClipVision()
    vae = FakeVAE()
    result = StableZero123_Conditioning.execute(
        clip_vision=clip_vision,
        init_image=image,
        vae=vae,
        width=50,
        height=34,
        batch_size=3,
        elevation=12.5,
        azimuth=-30.0,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert positive[0].shape == (1, 1, 8)
    assert negative[0].shape == (1, 1, 4)
    assert result["latent"]["samples"].shape == (3, 4, 4, 6)
    assert clip_vision.images[0].shape[-1] == 4
    assert vae.images[0].shape == (1, 34, 50, 3)
    assert not negative[0].count_nonzero()
    assert not negative[1]["concat_latent_image"].count_nonzero()
    np.testing.assert_array_equal(image, before)


def test_batched_zero123_increments_camera_and_sets_batch_index() -> None:
    result = StableZero123_Conditioning_Batched.execute(
        clip_vision=FakeClipVision(),
        init_image=_image(),
        vae=FakeVAE(),
        width=50,
        height=34,
        batch_size=3,
        elevation=10.0,
        azimuth=20.0,
        elevation_batch_increment=15.0,
        azimuth_batch_increment=30.0,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert positive[0].shape == (3, 1, 8)
    assert negative[0].shape == (1, 1, 4)
    assert result["latent"]["batch_index"] == [0, 0, 0]
    expected = torch.deg2rad(torch.tensor([-10.0, -25.0, -40.0]))
    torch.testing.assert_close(positive[0][:, 0, 4], expected)
    torch.testing.assert_close(
        positive[0][:, 0, 5], torch.sin(torch.deg2rad(torch.tensor([20.0, 50.0, 80.0])))
    )


def test_sv3d_conditioning_records_azimuth_and_elevation() -> None:
    result = SV3D_Conditioning.execute(
        clip_vision=FakeClipVision(),
        init_image=_image(),
        vae=FakeVAE(),
        width=50,
        height=34,
        video_frames=3,
        elevation=17.5,
    )
    positive, negative = result["positive"][0], result["negative"][0]
    assert result["latent"]["samples"].shape == (3, 4, 4, 6)
    assert positive[1]["elevation"] == [17.5, 17.5, 17.5]
    assert positive[1]["azimuth"] == [0.0, 180.0, 360.0]
    assert negative[1]["elevation"] == positive[1]["elevation"]
    assert negative[1]["azimuth"] == positive[1]["azimuth"]
    assert positive[0].shape == negative[0].shape == (1, 1, 4)
