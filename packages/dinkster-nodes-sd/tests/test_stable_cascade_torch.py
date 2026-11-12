from __future__ import annotations

import torch
from dinkster_sd.stable_cascade import (
    StableCascade_EmptyLatentImage,
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
