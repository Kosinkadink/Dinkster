"""CPU contracts for Wan 2.2 image-to-video latents."""

from __future__ import annotations

import torch
from dinkster_inference.latent_formats import Wan22
from dinkster_nodes_comfy.wan22_latent import Wan22ImageToVideoLatent


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        tokens = ((image.shape[0] - 1) // 4) + 1
        values = image[::4].sum(dim=(1, 2, 3))
        return values[:tokens].reshape(1, 1, tokens, 1, 1).repeat(1, 48, 1, 2, 3)


def _image(frames: int, height: int = 32, width: int = 48) -> torch.Tensor:
    values = torch.arange(frames * height * width * 4, dtype=torch.float64)
    return values.reshape(frames, height, width, 4) / 37.0


def _encoded(image: torch.Tensor) -> torch.Tensor:
    tokens = ((image.shape[0] - 1) // 4) + 1
    values = image[::4].sum(dim=(1, 2, 3))
    return values[:tokens].reshape(1, 1, tokens, 1, 1).repeat(1, 48, 1, 2, 3).float()


def _normalized_zeros(shape: tuple[int, ...]) -> torch.Tensor:
    format = Wan22()
    zeros = torch.zeros(shape)
    return zeros * format.latents_std + format.latents_mean


def _resize_center(image: torch.Tensor) -> torch.Tensor:
    pixels = image.movedim(-1, 1)
    pixels = pixels.narrow(-2, 8, 16)
    return torch.nn.functional.interpolate(pixels, size=(32, 48), mode="bilinear").movedim(1, -1)


def test_wan22_schema_matches_fixed_reference() -> None:
    schema = Wan22ImageToVideoLatent.define_schema()
    assert schema.node_type == "Wan22ImageToVideoLatent"
    assert schema.category == "model/conditioning/wan"
    assert [input.id for input in schema.inputs] == [
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
    ]
    assert [input.default for input in schema.inputs[1:5]] == [1280, 704, 49, 1]
    assert [schema.inputs[index].widget.step for index in (1, 2, 3)] == [32, 32, 4]
    assert not schema.inputs[-1].required
    assert [output.id for output in schema.outputs] == ["LATENT"]


def test_wan22_without_image_keeps_one_unmasked_source_batch() -> None:
    latent = Wan22ImageToVideoLatent.execute(
        vae=VAE(), width=48, height=32, length=13, batch_size=3
    )["LATENT"]
    assert set(latent) == {"samples"}
    samples = latent["samples"]
    assert samples.shape == (1, 48, 4, 2, 3)
    assert samples.dtype is torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()


def test_wan22_partial_start_image_preserves_all_channels_and_normalizes_only_unfilled_tokens() -> (
    None
):
    image = _image(5, width=24)
    before = image.clone()
    latent = Wan22ImageToVideoLatent.execute(
        vae=VAE(), width=48, height=32, length=13, batch_size=3, start_image=image
    )["LATENT"]
    expected_encoded = _encoded(_resize_center(image))
    expected = _normalized_zeros((1, 48, 4, 2, 3))
    expected[:, :, :2] = expected_encoded
    expected_mask = torch.ones((1, 1, 4, 2, 3))
    expected_mask[:, :, :2] = 0.0
    assert set(latent) == {"samples", "noise_mask"}
    torch.testing.assert_close(latent["samples"], expected.repeat(3, 1, 1, 1, 1), rtol=0, atol=0)
    torch.testing.assert_close(
        latent["noise_mask"], expected_mask.repeat(3, 1, 1, 1, 1), rtol=0, atol=0
    )
    assert torch.equal(latent["samples"][0], latent["samples"][2])
    torch.testing.assert_close(image, before, rtol=0, atol=0)


def test_wan22_full_start_image_fills_temporal_boundary_without_normalizing_encoded_values() -> (
    None
):
    image = _image(20)
    latent = Wan22ImageToVideoLatent.execute(
        vae=VAE(), width=48, height=32, length=9, batch_size=2, start_image=image
    )["LATENT"]
    expected = _encoded(image[:9])
    assert expected.shape == (1, 48, 3, 2, 3)
    torch.testing.assert_close(latent["samples"], expected.repeat(2, 1, 1, 1, 1), rtol=0, atol=0)
    assert not latent["noise_mask"].count_nonzero()
    assert torch.equal(latent["samples"][0], latent["samples"][1])
