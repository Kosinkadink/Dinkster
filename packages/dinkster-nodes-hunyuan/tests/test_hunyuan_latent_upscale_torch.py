"""Hunyuan latent upscale dimension ordering and metadata."""

import torch
from dinkster_hunyuan.latent_upscale import HunyuanVideo15LatentUpscaleWithModel


def test_hunyuan_zero_dimensions_returns_the_original_public_latent() -> None:
    source = {"samples": torch.ones((1, 2, 3, 5)), "metadata": "kept"}
    result = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=object(),
        samples=source,
        upscale_method="bilinear",
        width=0,
        height=0,
        crop="disabled",
    )["LATENT"]
    assert result is source
