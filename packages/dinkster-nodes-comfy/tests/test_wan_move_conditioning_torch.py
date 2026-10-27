"""CPU Wan Move conditioning behavior."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.wan_move_conditioning import WanMoveTrackToVideo


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return (
            image[..., :1]
            .mean(dim=-1)
            .reshape(1, 1, image.shape[0], image.shape[1], image.shape[2])
        )


def test_wan_move_conditioning_tracks_vision_and_sources() -> None:
    images = torch.tensor([[[[0.2, 0.3, 0.4]]], [[[0.8, 0.1, 0.2]]]])
    tracks = {
        "track_path": torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]]),
        "track_visibility": torch.tensor([[True], [True]]),
    }
    before = images.clone()
    torch.manual_seed(3)
    result = WanMoveTrackToVideo.execute(
        positive=[[torch.tensor([1.0]), {"side": "positive"}]],
        negative=[[torch.tensor([2.0]), {"side": "negative"}]],
        vae=VAE(),
        width=1,
        height=1,
        length=2,
        batch_size=1,
        strength=1.0,
        tracks=tracks,
        start_image=images,
        clip_vision_output={"vision": 4},
    )
    assert result["latent"]["samples"].shape == (1, 16, 1, 0, 0)
    assert result["positive"][0][1]["side"] == "positive"
    assert result["negative"][0][1]["side"] == "negative"
    assert result["positive"][0][1]["clip_vision_output"] is not None
    assert result["positive"][0][1]["concat_mask"].shape == (1, 1, 1, 1, 1)
    assert torch.equal(images, before)


def test_wan_move_conditioning_no_optional_values_preserves_conditioning() -> None:
    positive, negative = [[torch.tensor([1.0]), {"a": 1}]], [[torch.tensor([2.0]), {"b": 2}]]
    result = WanMoveTrackToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=8,
        height=8,
        length=1,
        batch_size=2,
        strength=0.0,
    )
    assert result["positive"] is positive and result["negative"] is negative
    assert result["latent"]["samples"].shape == (2, 16, 1, 1, 1)
