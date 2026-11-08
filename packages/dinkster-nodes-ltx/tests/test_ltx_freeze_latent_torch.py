"""LTX frozen latent masks preserve samples and source metadata."""

import math

import pytest
import torch
from dinkster_ltx.video_latent_ops import LTXVFreezeLatent


@pytest.mark.parametrize(
    ("shape", "mask_shape"),
    (
        ((2, 3, 5, 7, 11), (2, 1, 5, 1, 1)),
        ((2, 3, 5, 11), (2, 1, 5, 1)),
    ),
)
def test_freeze_latent_replaces_the_mask_without_mutating_the_source(
    shape: tuple[int, ...], mask_shape: tuple[int, ...]
) -> None:
    samples = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    original_mask = torch.full(mask_shape, 0.75, dtype=torch.float64)
    latent = {"samples": samples, "noise_mask": original_mask, "metadata": {"keep": True}}
    samples_before, mask_before = samples.clone(), original_mask.clone()

    result = LTXVFreezeLatent.execute(latent=latent)["latent"]

    assert result is not latent
    assert result["samples"] is samples
    assert result["metadata"] is latent["metadata"]
    assert result["noise_mask"].shape == mask_shape
    assert result["noise_mask"].dtype == torch.float32
    assert result["noise_mask"].device == samples.device
    assert not torch.count_nonzero(result["noise_mask"])
    assert torch.equal(samples, samples_before)
    assert torch.equal(original_mask, mask_before)
