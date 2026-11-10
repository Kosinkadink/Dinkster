"""CPU Cosmos video latent layout and metadata."""

import torch
from dinkster_cosmos.video_latent import EmptyCosmosLatentVideo


def test_empty_cosmos_video_latent_preserves_layout_and_metadata() -> None:
    latent = EmptyCosmosLatentVideo.execute(width=79, height=71, length=8, batch_size=2)["LATENT"]
    samples = latent["samples"]
    assert tuple(samples.shape) == (2, 16, 1, 8, 9)
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert not samples.count_nonzero()
    assert {key: value for key, value in latent.items() if key != "samples"} == {}
