from __future__ import annotations

import torch
from dinkster_misc_models.mochi import EmptyMochiLatentVideo


def test_mochi_temporal_boundary_and_asymmetric_dimensions() -> None:
    for length, frames in ((12, 2), (13, 3)):
        result = EmptyMochiLatentVideo.execute(width=40, height=24, length=length, batch_size=2)
        samples = result["LATENT"]["samples"]
        assert samples.shape == (2, 12, frames, 3, 5)
        assert samples.device.type == "cpu"
        assert samples.dtype == torch.float32
        assert not samples.count_nonzero()
