from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dinkster_sd.video_conditioning import SVD_img2vid_Conditioning


class ClipVision:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode_image(self, image: torch.Tensor) -> SimpleNamespace:
        self.images.append(image.clone())
        return SimpleNamespace(image_embeds=image.mean((1, 2)))


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image.movedim(-1, 1)


def _image() -> np.ndarray:
    rows, columns, channels = np.indices((16, 30, 4), dtype=np.float32)
    return (rows * 100 + columns * 10 + channels)[None] / 1000.0


def _execute(augmentation_level: float) -> tuple[dict[str, object], ClipVision, VAE]:
    clip_vision = ClipVision()
    vae = VAE()
    result = SVD_img2vid_Conditioning.execute(
        clip_vision=clip_vision,
        init_image=_image(),
        vae=vae,
        width=24,
        height=16,
        video_frames=3,
        motion_bucket_id=127,
        fps=6,
        augmentation_level=augmentation_level,
    )
    return dict(result), clip_vision, vae


def test_stable_video_conditioning_preserves_source_operations() -> None:
    result, clip_vision, vae = _execute(0.0)
    positive = result["positive"][0]
    negative = result["negative"][0]
    expected_source = torch.as_tensor(_image())
    expected_pixels = expected_source[:, :, 3:27, :]
    expected_pooled = expected_source.mean((1, 2)).unsqueeze(0)
    expected_encoded = expected_pixels[:, :, :, :3].movedim(-1, 1)
    assert clip_vision.images[0].shape == (1, 16, 30, 4)
    torch.testing.assert_close(positive[0], expected_pooled)
    torch.testing.assert_close(vae.images[0], expected_pixels[:, :, :, :3])
    torch.testing.assert_close(positive[1]["concat_latent_image"], expected_encoded)
    assert result["latent"]["samples"].shape == (3, 4, 2, 3)
    assert result["latent"]["samples"].dtype == torch.get_default_dtype()
    assert result["latent"]["samples"].device.type == "cpu"
    assert set(positive[1]) == {
        "motion_bucket_id",
        "fps",
        "augmentation_level",
        "concat_latent_image",
    }
    assert set(negative[1]) == set(positive[1])
    assert positive[1]["motion_bucket_id"] == negative[1]["motion_bucket_id"] == 127
    assert positive[1]["fps"] == negative[1]["fps"] == 6
    assert positive[1]["augmentation_level"] == negative[1]["augmentation_level"] == 0.0
    assert not negative[0].count_nonzero()
    assert not negative[1]["concat_latent_image"].count_nonzero()


def test_stable_video_conditioning_uses_source_augmentation_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _image()[..., :3]
    noise = torch.arange(1 * 16 * 24 * 3, dtype=torch.float32).reshape(1, 16, 24, 3) / 19.0
    monkeypatch.setattr(torch, "randn_like", lambda pixels: noise)
    vae = VAE()
    result = SVD_img2vid_Conditioning.execute(
        clip_vision=ClipVision(),
        init_image=image,
        vae=vae,
        width=24,
        height=16,
        video_frames=3,
        motion_bucket_id=127,
        fps=6,
        augmentation_level=0.25,
    )
    from dinkster_inference.utils import common_upscale

    pixels = common_upscale(
        torch.as_tensor(image).movedim(-1, 1), 24, 16, "bilinear", "center"
    ).movedim(1, -1)
    expected = pixels[:, :, :, :3] + noise * 0.25
    torch.testing.assert_close(vae.images[0], expected)
    torch.testing.assert_close(
        result["positive"][0][1]["concat_latent_image"], expected.movedim(-1, 1)
    )
