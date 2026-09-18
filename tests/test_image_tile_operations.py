from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from dinkster_nodes_image.tiles import ImageTileMerge, ImageTileSplit, _tile_bounds, _tile_geometry


def test_tile_geometry_matches_essentials_overlap_limits() -> None:
    assert _tile_geometry(8, 12, 2, 3, 0.25, 1, 1) == (4, 4, 2, 2)
    assert _tile_geometry(8, 12, 1, 1, 0.5, 100, 100) == (8, 12, 0, 0)
    with pytest.raises(ValueError, match="between 0 and 0.5"):
        _tile_geometry(8, 12, 2, 3, 0.75, 0, 0)


def test_tile_bounds_match_essentials_far_edge_adjustment() -> None:
    bounds = [
        _tile_bounds(
            row,
            column,
            rows=2,
            columns=3,
            tile_height=4,
            tile_width=4,
            overlap_y=1,
            overlap_x=1,
        )
        for row in range(2)
        for column in range(3)
    ]
    assert bounds == [
        (0, 0, 5, 5),
        (3, 0, 8, 5),
        (7, 0, 12, 5),
        (0, 3, 5, 8),
        (3, 3, 8, 8),
        (7, 3, 12, 8),
    ]


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


def test_tile_merge_validates_count_and_overlap() -> None:
    tiles = np.zeros((3, 4, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of 4"):
        ImageTileMerge.execute(tiles=tiles, rows=2, columns=2)
    with pytest.raises(ValueError, match="smaller than the tile"):
        ImageTileMerge.execute(
            tiles=np.zeros((4, 4, 4, 1), dtype=np.float32),
            rows=2,
            columns=2,
            overlap_x=4,
        )
