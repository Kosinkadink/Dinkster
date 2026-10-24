"""CPU tensor contracts for core VAE nodes."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from dinkster_nodes_comfy.vae import VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled

torch: Any = importlib.import_module("torch")


class ImageVAE:
    def encode(self, pixels: Any) -> Any:
        return pixels.movedim(-1, 1) * 3.0 - 2.0

    def decode(self, samples: Any) -> Any:
        return ((samples + 2.0) / 3.0).movedim(1, -1)

    def spacial_compression_decode(self) -> int:
        return 8

    def temporal_compression_decode(self) -> int | None:
        return None

    def encode_tiled(
        self,
        pixels: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
        tile_t: int,
        overlap_t: int,
    ) -> Any:
        return self.encode(pixels) + tile_x + 2 * tile_y + 3 * overlap + 5 * tile_t + 7 * overlap_t

    def decode_tiled(
        self,
        samples: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
        tile_t: int | None,
        overlap_t: int | None,
    ) -> Any:
        return (
            self.decode(samples)
            + tile_x
            + 2 * tile_y
            + 3 * overlap
            + 5 * (tile_t or 0)
            + 7 * (overlap_t or 0)
        )


class VideoVAE(ImageVAE):
    def temporal_compression_decode(self) -> int:
        return 4

    def decode(self, samples: Any) -> Any:
        return samples.movedim(1, -1)


def test_encode_decode_preserve_numpy_transport_and_source() -> None:
    pixels = torch.arange(2 * 7 * 11 * 3, dtype=torch.float32).reshape(2, 7, 11, 3) / 512
    original = pixels.clone()
    encoded: Any = VAEEncode.execute(vae=ImageVAE(), pixels=pixels.numpy())["LATENT"]
    assert torch.equal(encoded["samples"], pixels.movedim(-1, 1) * 3 - 2)
    decoded: Any = VAEDecode.execute(
        vae=ImageVAE(), samples={"samples": encoded["samples"].numpy()}
    )["IMAGE"]
    torch.testing.assert_close(decoded, pixels, rtol=0, atol=3e-8)
    assert torch.equal(pixels, original)


@pytest.mark.parametrize("video", (False, True))
def test_tiled_spatial_and_temporal_units(video: bool) -> None:
    pixels = torch.full((1, 7, 11, 3), 0.25)
    encoded: Any = VAEEncodeTiled.execute(
        vae=ImageVAE(),
        pixels=pixels.numpy(),
        tile_size=192,
        overlap=32,
        temporal_size=12,
        temporal_overlap=4,
    )["LATENT"]
    assert torch.equal(encoded["samples"], torch.full((1, 3, 7, 11), 758.75))
    samples = torch.full((2, 3, 5, 7, 11) if video else (2, 3, 7, 11), 0.25)
    result: Any = VAEDecodeTiled.execute(
        vae=VideoVAE() if video else ImageVAE(),
        samples={"samples": samples},
        tile_size=64,
        overlap=48,
        temporal_size=8,
        temporal_overlap=12,
    )["IMAGE"]
    # Spatial overlap becomes 16 pixels, then 2 latent cells. Temporal overlap
    # becomes 6 frames, then 1 cell after video compression.
    expected = 47.25 if video else 30.75
    assert torch.equal(result, torch.full((10, 7, 11, 3) if video else (2, 7, 11, 3), expected))


def test_decode_nested_latent_uses_first_stream_and_flattens_video() -> None:
    first = torch.arange(2 * 3 * 5 * 7 * 11, dtype=torch.float32).reshape(2, 3, 5, 7, 11)
    second = torch.zeros(1, 4, 3, 2, 2)
    nested = torch.nested.nested_tensor([first, second])
    result: Any = VAEDecode.execute(vae=VideoVAE(), samples={"samples": nested})["IMAGE"]
    np.testing.assert_array_equal(
        result.numpy(), first.movedim(1, -1).reshape(10, 7, 11, 3).numpy()
    )
