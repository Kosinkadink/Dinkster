from __future__ import annotations

from typing import Any

import torch
from dinkster_misc_models.reference_conditioning import InstructPixToPixConditioning


class ImageVAE:
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.movedim(-1, 1) * 2.0 - 1.0


def test_image_edit_crops_without_mutating_inputs() -> None:
    pixels = torch.arange(11 * 21 * 4, dtype=torch.float64).reshape(1, 11, 21, 4) / 1024
    before = pixels.clone()
    positive = [[torch.tensor([3.0]), {"strength": 0.7}]]
    negative = [[torch.tensor([-2.0]), {"concat_latent_image": "replace", "strength": 0.2}]]
    result: Any = InstructPixToPixConditioning.execute(
        vae=ImageVAE(), pixels=pixels.numpy(), positive=positive, negative=negative
    )
    expected = pixels[:, 1:9, 2:18, :].movedim(-1, 1) * 2 - 1
    assert torch.equal(result["positive"][0][1]["concat_latent_image"], expected)
    assert torch.equal(result["negative"][0][1]["concat_latent_image"], expected)
    assert result["positive"][0][1]["strength"] == 0.7
    assert result["negative"][0][1]["strength"] == 0.2
    assert result["latent"]["samples"].dtype == torch.float64
    assert torch.equal(result["latent"]["samples"], torch.zeros(1, 4, 8, 16, dtype=torch.float64))
    assert torch.equal(pixels, before)
    assert positive[0][1] == {"strength": 0.7}
    assert negative[0][1]["concat_latent_image"] == "replace"
