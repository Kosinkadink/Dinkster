"""Empty Trellis latents preserve their layout and independent zero storage."""

from __future__ import annotations

import torch
from dinkster_3d_models.latent_model_specific import EmptyTrellis2LatentStructure


def test_empty_trellis_latent_layout() -> None:
    result = EmptyTrellis2LatentStructure.execute(batch_size=2)["LATENT"]
    samples = result["samples"]
    assert tuple(samples.shape) == (2, 32, 16, 16, 16)
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert torch.count_nonzero(samples).item() == 0
    assert {key: value for key, value in result.items() if key != "samples"} == {"type": "trellis2"}
    with torch.inference_mode():
        samples.fill_(3)
    fresh = EmptyTrellis2LatentStructure.execute(batch_size=2)["LATENT"]
    assert torch.count_nonzero(fresh["samples"]).item() == 0
