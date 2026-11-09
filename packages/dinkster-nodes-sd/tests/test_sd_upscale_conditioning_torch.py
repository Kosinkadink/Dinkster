from __future__ import annotations

from typing import Any

import torch
from dinkster_sd.sd_upscale_conditioning import SD_4XUpscale_Conditioning


def test_sd_upscale_rounds_asymmetric_dimensions_and_preserves_conditioning_sources() -> None:
    image = torch.arange(2 * 7 * 11 * 3, dtype=torch.float32).reshape(2, 7, 11, 3) / 511
    image_before = image.clone()
    positive = [[torch.tensor([[[1.0]]]), {"source": "positive"}]]
    negative = [[torch.tensor([[[2.0]]]), {"source": "negative"}]]
    result = SD_4XUpscale_Conditioning.execute(
        images=image.numpy(),
        positive=positive,
        negative=negative,
        scale_ratio=1.5,
        noise_augmentation=0.125,
    )
    assert torch.equal(image, image_before)
    assert positive[0][1] == {"source": "positive"}
    assert negative[0][1] == {"source": "negative"}
    assert result["positive"][0][0] is positive[0][0]
    assert result["negative"][0][0] is negative[0][0]
    assert result["positive"][0][1]["concat_image"].shape == (2, 3, 2, 4)
    expected_pixels = torch.tensor(
        [
            [
                [
                    [
                        (batch * 231 + row * 33 + column * 3 + channel) / 511 * 2 - 1
                        for column in (0.875, 3.625, 6.375, 9.125)
                    ]
                    for row in (1.75, 4.25)
                ]
                for channel in range(3)
            ]
            for batch in range(2)
        ]
    )
    assert torch.allclose(result["positive"][0][1]["concat_image"], expected_pixels)
    assert result["positive"][0][1]["noise_augmentation"] == 0.125
    assert result["negative"][0][1]["concat_image"] is result["positive"][0][1]["concat_image"]
    latent: Any = result["latent"]
    assert latent["samples"].shape == (2, 4, 2, 4)
    assert latent["samples"].dtype == torch.float32
    assert torch.count_nonzero(latent["samples"]) == 0
