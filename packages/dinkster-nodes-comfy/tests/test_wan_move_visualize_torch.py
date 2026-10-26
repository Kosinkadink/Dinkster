"""CPU Wan Move visualization behavior."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.wan_move_visualize import WanMoveVisualizeTracks


def test_visualization_draws_tracks_without_mutating_sources() -> None:
    images = torch.zeros((2, 8, 8, 3))
    tracks = {
        "track_path": torch.tensor([[[2.0, 2.0]], [[5.0, 2.0]]]),
        "track_visibility": torch.tensor([[True], [True]]),
    }
    before = images.clone()
    result = WanMoveVisualizeTracks.execute(
        images=images, tracks=tracks, line_resolution=24, circle_size=1, opacity=1.0, line_width=2
    )["IMAGE"]
    assert result.shape == images.shape and result.dtype is torch.float32
    assert result[0].sum() > 0 and result[1].sum() > 0
    assert torch.equal(images, before)


def test_visualization_absent_tracks_is_identity() -> None:
    images = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 24
    assert (
        WanMoveVisualizeTracks.execute(
            images=images, tracks=None, line_resolution=1, circle_size=1, opacity=0.0, line_width=1
        )["IMAGE"]
        is images
    )
