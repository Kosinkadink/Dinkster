from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from dinkster_image.tiles import ImageTileMerge, ImageTileSplit


@pytest.mark.parametrize(("overlap", "extra"), [(0.0, 0), (0.25, 0), (0.0, 1)])
def test_image_tile_split_merge_round_trip(overlap: float, extra: int) -> None:
    image = np.random.default_rng(7).random((2, 8, 12, 3), dtype=np.float32)
    split = ImageTileSplit.execute(
        image=image,
        rows=2,
        columns=3,
        overlap=overlap,
        overlap_x=extra,
        overlap_y=extra,
    )
    tiles = np.asarray(split["tiles"])
    assert tiles.shape[0] == 12
    merged = ImageTileMerge.execute(
        tiles=tiles,
        rows=2,
        columns=3,
        overlap_x=cast("int", split["overlap_x"]),
        overlap_y=cast("int", split["overlap_y"]),
    )["image"]
    np.testing.assert_allclose(np.asarray(merged), image, rtol=0, atol=2e-7)


def test_tile_split_crops_remainder_like_essentials() -> None:
    image = np.arange(7 * 10, dtype=np.float32).reshape((1, 7, 10, 1))
    split = ImageTileSplit.execute(image=image, rows=2, columns=3)
    merged = ImageTileMerge.execute(
        tiles=split["tiles"], rows=2, columns=3, overlap_x=0, overlap_y=0
    )["image"]
    np.testing.assert_array_equal(np.asarray(merged), image[:, :6, :9])
