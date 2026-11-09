"""CPU Hunyuan video latent layouts and metadata."""

from typing import Any

import pytest
import torch
from dinkster_hunyuan.video_latent import EmptyHunyuanLatentVideo, EmptyHunyuanVideo15Latent


@pytest.mark.parametrize(
    ("node", "inputs", "shape", "ratio"),
    (
        (
            EmptyHunyuanLatentVideo,
            {"width": 151, "height": 95, "length": 8, "batch_size": 2},
            (2, 16, 2, 11, 18),
            8,
        ),
        (
            EmptyHunyuanVideo15Latent,
            {"width": 159, "height": 111, "length": 9, "batch_size": 2},
            (2, 32, 3, 6, 9),
            16,
        ),
    ),
)
def test_empty_hunyuan_video_latent_preserves_layout_and_metadata(
    node: type[Any], inputs: dict[str, int], shape: tuple[int, ...], ratio: int
) -> None:
    latent = node.execute(**inputs)["LATENT"]
    samples = latent["samples"]
    assert tuple(samples.shape) == shape
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert not samples.count_nonzero()
    assert {key: value for key, value in latent.items() if key != "samples"} == {
        "downscale_ratio_spacial": ratio
    }
