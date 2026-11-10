from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    materialize_latent,
)
from dinkster_misc_models.kandinsky_video import Kandinsky5ImageToVideo


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[-1] == 3
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def image() -> torch.Tensor:
    return torch.arange(6 * 6 * 12 * 4, dtype=torch.float64).reshape(6, 6, 12, 4) / 37


def conditioning(value: float) -> list[list[Any]]:
    return [[torch.tensor([[value]]), {"source": value}]]


def check(
    positive: Any,
    negative: Any,
    latent: object,
    clean: object,
    length: int,
    start: bool,
    image_dtype: torch.dtype = torch.float64,
) -> None:
    samples = materialize_latent(latent)["samples"]
    assert samples.shape == (2, 16, (length - 1) // 4 + 1, 2, 4)
    assert samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()
    clean = materialize_latent(clean)
    if not start:
        assert clean == {}
        assert positive[0][1] == {"source": 3.0}
        assert negative[0][1] == {"source": -7.0}
        return
    pixels = torch.nn.functional.interpolate(
        image().to(image_dtype)[:length].movedim(-1, 1),
        size=(16, 32),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    encoded = pixels[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)
    torch.testing.assert_close(clean["samples"], encoded, rtol=0, atol=0)
    mask = torch.ones((1, 1, samples.shape[2], 2, 4), dtype=image_dtype)
    mask[:, :, : (min(length, 6) - 1) // 4 + 1] = 0
    for cond, source in ((positive, 3.0), (negative, -7.0)):
        assert cond[0][1]["source"] == source
        torch.testing.assert_close(cond[0][1]["time_dim_replace"], encoded, rtol=0, atol=0)
        torch.testing.assert_close(cond[0][1]["concat_mask"], mask, rtol=0, atol=0)


@pytest.mark.parametrize("length", (5, 9))
def test_public_start_and_empty_conditioning_preserve_sources(length: int) -> None:
    positive, negative, pixels = conditioning(3.0), conditioning(-7.0), image()
    before = pixels.clone()
    for start in (False, True):
        result = Kandinsky5ImageToVideo.execute(
            positive=positive,
            negative=negative,
            vae=VAE(),
            width=32,
            height=16,
            length=length,
            batch_size=2,
            start_image=pixels if start else None,
        )
        check(
            result["positive"],
            result["negative"],
            result["latent"],
            result["cond_latent"],
            length,
            start,
        )
    assert positive[0][1] == {"source": 3.0} and negative[0][1] == {"source": -7.0}
    assert torch.equal(pixels, before)
