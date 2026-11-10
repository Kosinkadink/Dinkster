"""CPU LTX video latent layout and metadata."""

import torch
from dinkster_ltx.video_latent import EmptyLTXVLatentVideo


def test_empty_ltx_video_latent_preserves_layout_and_metadata() -> None:
    latent = EmptyLTXVLatentVideo.execute(width=95, height=127, length=16, batch_size=2)["LATENT"]
    samples = latent["samples"]
    assert tuple(samples.shape) == (2, 128, 2, 3, 2)
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert not samples.count_nonzero()
    assert {key: value for key, value in latent.items() if key != "samples"} == {
        "downscale_ratio_spacial": 32
    }
