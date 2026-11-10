from __future__ import annotations

import numpy as np
import torch
from dinkster_sd.stable_cascade import (
    StableCascade_EmptyLatentImage,
    StableCascade_StageB_Conditioning,
    StableCascade_StageC_VAEEncode,
    StableCascade_SuperResolutionControlnet,
)


def test_latent_shapes_and_compression_rounding() -> None:
    stages = StableCascade_EmptyLatentImage.execute(
        width=264, height=280, compression=42, batch_size=2
    )
    assert stages["stage_c"]["samples"].shape == (2, 16, 6, 6)
    assert stages["stage_b"]["samples"].shape == (2, 4, 70, 66)
    assert not stages["stage_c"]["samples"].count_nonzero()
    assert not stages["stage_b"]["samples"].count_nonzero()
    assert stages["stage_c"]["samples"].dtype == torch.float32


class CascadeVAE:
    downscale_ratio = 4

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.mean(-1, keepdim=True).movedim(-1, 1)[:, :, ::4, ::4].repeat(1, 16, 1, 1)


def test_vae_encoding_preserves_crop_rgb_and_stage_dimensions() -> None:
    # Deliberately non-divisible dimensions distinguish stage B's two rounding rules.
    image = np.zeros((2, 35, 51, 4), dtype=np.float32)
    image[..., :3] = 0.25
    image[..., 3] = 0.9
    before = image.copy()
    encoded = StableCascade_StageC_VAEEncode.execute(image=image, vae=CascadeVAE(), compression=8)
    assert encoded["stage_c"]["samples"].shape == (2, 16, 4, 6)
    torch.testing.assert_close(encoded["stage_c"]["samples"], torch.full((2, 16, 4, 6), 0.25))
    assert encoded["stage_b"]["samples"].shape == (2, 4, 8, 12)
    control = StableCascade_SuperResolutionControlnet.execute(image=image, vae=CascadeVAE())
    assert control["controlnet_input"].shape == (2, 9, 13, 16)
    torch.testing.assert_close(control["controlnet_input"], torch.full((2, 9, 13, 16), 0.25))
    assert control["stage_c"]["samples"].shape == (2, 16, 2, 3)
    assert control["stage_b"]["samples"].shape == (2, 4, 17, 25)
    assert not control["stage_c"]["samples"].count_nonzero()
    assert not control["stage_b"]["samples"].count_nonzero()
    np.testing.assert_array_equal(image, before)


def test_prior_conditioning_materializes_without_source_mutation() -> None:
    samples = np.arange(96, dtype=np.float32).reshape(1, 16, 2, 3)
    text = torch.ones(1, 2, 4)
    conditioning = [[text, {"strength": 0.7}], [text + 1, {"other": True}]]
    result = StableCascade_StageB_Conditioning.execute(
        conditioning=conditioning, stage_c={"samples": samples}
    )["CONDITIONING"]
    assert result[0][0] is text
    assert result[0][1]["strength"] == 0.7
    assert result[1][1]["other"] is True
    torch.testing.assert_close(result[1][1]["stable_cascade_prior"], torch.from_numpy(samples))
    assert conditioning[0][1] == {"strength": 0.7}
    assert conditioning[1][1] == {"other": True}
