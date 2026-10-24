"""CPU contracts for native LTXV video conditioning nodes."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.ltx_video_conditioning import (
    LTXVConditioning,
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
)


class VAE:
    downscale_index_formula = (8, 32, 32)

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(pixels.clone())
        return pixels[:, 0, 0, 0].reshape(1, 1, -1, 1, 1).repeat(1, 128, 1, 1, 1)


def _image(frames: int = 3, height: int = 32, width: int = 32) -> torch.Tensor:
    return torch.stack(
        [
            torch.full((height, width, 4), float(frame), dtype=torch.float64)
            for frame in range(frames)
        ]
    )


def test_img_to_video_encodes_all_frames_and_preserves_conditioning() -> None:
    vae = VAE()
    positive, negative = [["positive", {"keep": 1}]], [["negative", {"keep": 2}]]
    output = LTXVImgToVideo.execute(
        positive=positive,
        negative=negative,
        image=_image(),
        vae=vae,
        width=32,
        height=32,
        length=17,
        batch_size=2,
        strength=0.25,
    )
    latent = output["latent"]
    assert output["positive"] is positive
    assert output["negative"] is negative
    assert vae.calls[0].shape == (3, 32, 32, 3)
    assert torch.equal(vae.calls[0][:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert latent["samples"].dtype is torch.get_default_dtype()
    assert latent["samples"].shape == (2, 128, 3, 1, 1)
    assert torch.equal(latent["samples"][0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert latent["noise_mask"].dtype is torch.float32
    assert torch.equal(latent["noise_mask"][:, :, :, 0, 0], torch.full((2, 1, 3), 0.75))


@torch.no_grad()
def test_img_to_video_strength_extremes() -> None:
    for strength, expected in ((0.0, 1.0), (1.0, 0.0)):
        latent = LTXVImgToVideo.execute(
            positive=[],
            negative=[],
            image=_image(1),
            vae=VAE(),
            width=32,
            height=32,
            length=17,
            batch_size=1,
            strength=strength,
        )["latent"]
        assert torch.equal(latent["noise_mask"][:, :, :1], torch.full((1, 1, 1, 1, 1), expected))
        assert torch.equal(latent["noise_mask"][:, :, 1:], torch.ones((1, 1, 2, 1, 1)))


def test_inplace_bypass_returns_identity_without_touching_vae() -> None:
    vae = VAE()
    latent = {"samples": torch.ones((1, 128, 3, 1, 1)), "metadata": {"ignored": True}}
    output = LTXVImgToVideoInplace.execute(
        vae=vae, image=_image(), latent=latent, strength=0.5, bypass=True
    )
    assert output["latent"] is latent
    assert vae.calls == []


def test_inplace_clones_noise_mask_and_discards_unrelated_metadata() -> None:
    vae = VAE()
    original_mask = torch.full((1, 1, 3, 1, 1), 0.4, dtype=torch.float64)
    source = {
        "samples": torch.full((1, 128, 3, 1, 1), 9.0),
        "noise_mask": original_mask,
        "extra": 4,
    }
    output = LTXVImgToVideoInplace.execute(vae=vae, image=_image(2), latent=source, strength=0.5)[
        "latent"
    ]
    assert set(output) == {"samples", "noise_mask"}
    assert output["samples"] is not source["samples"]
    assert output["noise_mask"] is not original_mask
    assert torch.equal(original_mask, torch.full_like(original_mask, 0.4))
    assert torch.equal(output["samples"][0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 9.0]))
    assert torch.equal(
        output["noise_mask"][0, 0, :, 0, 0], torch.tensor([0.5, 0.5, 0.4], dtype=torch.float64)
    )


def test_inplace_creates_float32_noise_mask_and_conditioning_preserves_inputs() -> None:
    source = {"samples": torch.zeros((1, 128, 3, 1, 1))}
    latent = LTXVImgToVideoInplace.execute(
        vae=VAE(), image=_image(1), latent=source, strength=0.25
    )["latent"]
    assert latent["noise_mask"].dtype is torch.float32
    assert torch.equal(latent["noise_mask"][0, 0, :, 0, 0], torch.tensor([0.75, 1.0, 1.0]))
    positive, negative = [["a", {"old": 1}]], [["b", {"old": 2}]]
    output = LTXVConditioning.execute(positive=positive, negative=negative, frame_rate=23.976)
    assert positive == [["a", {"old": 1}]]
    assert negative == [["b", {"old": 2}]]
    assert output["positive"] == [["a", {"old": 1, "frame_rate": 23.976}]]
    assert output["negative"] == [["b", {"old": 2, "frame_rate": 23.976}]]
