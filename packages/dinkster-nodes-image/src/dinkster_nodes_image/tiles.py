"""Deterministic overlapping image tiles."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec

from .geometry import FLOAT, IMAGE, INT
from .support import check_output_size, image_array, number_input


def _tile_geometry(
    height: int,
    width: int,
    rows: int,
    columns: int,
    overlap: float,
    overlap_x: int,
    overlap_y: int,
) -> tuple[int, int, int, int]:
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if not math.isfinite(overlap) or not 0.0 <= overlap <= 0.5:
        raise ValueError("overlap must be between 0 and 0.5")
    if overlap_x < 0 or overlap_y < 0:
        raise ValueError("overlap_x and overlap_y must be non-negative")
    tile_height, tile_width = height // rows, width // columns
    if tile_height < 1 or tile_width < 1:
        raise ValueError("rows and columns exceed the image dimensions")
    real_overlap_y = min(tile_height // 2, int(tile_height * overlap) + overlap_y)
    real_overlap_x = min(tile_width // 2, int(tile_width * overlap) + overlap_x)
    if rows == 1:
        real_overlap_y = 0
    if columns == 1:
        real_overlap_x = 0
    return tile_height, tile_width, real_overlap_y, real_overlap_x


def _tile_bounds(
    row: int,
    column: int,
    *,
    rows: int,
    columns: int,
    tile_height: int,
    tile_width: int,
    overlap_y: int,
    overlap_x: int,
) -> tuple[int, int, int, int]:
    output_height, output_width = rows * tile_height, columns * tile_width
    top, left = row * tile_height, column * tile_width
    if row > 0:
        top -= overlap_y
    if column > 0:
        left -= overlap_x
    bottom, right = top + tile_height + overlap_y, left + tile_width + overlap_x
    if bottom > output_height:
        bottom = output_height
        top = bottom - tile_height - overlap_y
    if right > output_width:
        right = output_width
        left = right - tile_width - overlap_x
    return left, top, right, bottom


class ImageTileSplit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.tiles.split",
            display_name="Split Image into Tiles",
            category="image/tiles",
            inputs=(
                InputSpec("image", IMAGE),
                number_input("rows", INT, 2, minimum=1, maximum=256, step=1),
                number_input("columns", INT, 2, minimum=1, maximum=256, step=1),
                number_input("overlap", FLOAT, 0.0, minimum=0.0, maximum=0.5, step=0.01),
                number_input("overlap_x", INT, 0, minimum=0, maximum=8192, step=1),
                number_input("overlap_y", INT, 0, minimum=0, maximum=8192, step=1),
            ),
            outputs=(
                OutputSpec("tiles", IMAGE, preview=True),
                OutputSpec("tile_width", INT),
                OutputSpec("tile_height", INT),
                OutputSpec("overlap_x", INT),
                OutputSpec("overlap_y", INT),
            ),
            search_terms=("ImageTile", "split image", "tile image"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        rows: int = 2,
        columns: int = 2,
        overlap: float = 0.0,
        overlap_x: int = 0,
        overlap_y: int = 0,
    ) -> Mapping[str, object]:
        array = image_array(image)
        tile_height, tile_width, real_overlap_y, real_overlap_x = _tile_geometry(
            int(array.shape[1]),
            int(array.shape[2]),
            rows,
            columns,
            overlap,
            overlap_x,
            overlap_y,
        )
        count = len(array) * rows * columns
        check_output_size(
            (
                count,
                tile_height + real_overlap_y,
                tile_width + real_overlap_x,
                int(array.shape[3]),
            )
        )
        tiles: list[np.ndarray] = []
        for row in range(rows):
            for column in range(columns):
                left, top, right, bottom = _tile_bounds(
                    row,
                    column,
                    rows=rows,
                    columns=columns,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    overlap_y=real_overlap_y,
                    overlap_x=real_overlap_x,
                )
                tiles.append(array[:, top:bottom, left:right, :])
        return cls.outputs(
            tiles=np.ascontiguousarray(np.concatenate(tiles, axis=0)),
            tile_width=tile_width + real_overlap_x,
            tile_height=tile_height + real_overlap_y,
            overlap_x=real_overlap_x,
            overlap_y=real_overlap_y,
        )


class ImageTileMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.tiles.merge",
            display_name="Merge Image Tiles",
            category="image/tiles",
            inputs=(
                InputSpec("tiles", IMAGE),
                number_input("overlap_x", INT, 0, minimum=0, maximum=8192, step=1),
                number_input("overlap_y", INT, 0, minimum=0, maximum=8192, step=1),
                number_input("rows", INT, 2, minimum=1, maximum=256, step=1),
                number_input("columns", INT, 2, minimum=1, maximum=256, step=1),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("ImageUntile", "merge image tiles", "stitch tiles"),
        )

    @classmethod
    def execute(
        cls,
        *,
        tiles: object,
        overlap_x: int = 0,
        overlap_y: int = 0,
        rows: int = 2,
        columns: int = 2,
    ) -> Mapping[str, object]:
        array = image_array(tiles, subject="tiles")
        if rows < 1 or columns < 1:
            raise ValueError("rows and columns must be positive")
        if overlap_x < 0 or overlap_y < 0:
            raise ValueError("overlap_x and overlap_y must be non-negative")
        tile_count = rows * columns
        if len(array) % tile_count:
            raise ValueError(
                f"tile batch must contain a multiple of {tile_count} frames, got {len(array)}"
            )
        batch = len(array) // tile_count
        tile_height = int(array.shape[1]) - overlap_y
        tile_width = int(array.shape[2]) - overlap_x
        if tile_height < 1 or tile_width < 1:
            raise ValueError("tile overlap must be smaller than the tile dimensions")
        output_height, output_width = rows * tile_height, columns * tile_width
        check_output_size((batch, output_height, output_width, int(array.shape[3])))
        grouped = array.reshape(tile_count, batch, *array.shape[1:])
        output = np.zeros(
            (batch, output_height, output_width, int(array.shape[3])), dtype=np.float32
        )
        for row in range(rows):
            for column in range(columns):
                tile = grouped[row * columns + column]
                left, top, right, bottom = _tile_bounds(
                    row,
                    column,
                    rows=rows,
                    columns=columns,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    overlap_y=overlap_y,
                    overlap_x=overlap_x,
                )
                mask = np.ones((1, tile.shape[1], tile.shape[2], 1), dtype=np.float32)
                if row > 0 and overlap_y:
                    mask[:, :overlap_y, :, :] *= np.linspace(0.0, 1.0, overlap_y, dtype=np.float32)[
                        None, :, None, None
                    ]
                if column > 0 and overlap_x:
                    mask[:, :, :overlap_x, :] *= np.linspace(0.0, 1.0, overlap_x, dtype=np.float32)[
                        None, None, :, None
                    ]
                region = output[:, top:bottom, left:right, :]
                output[:, top:bottom, left:right, :] = region * (1.0 - mask) + tile * mask
        return cls.outputs(image=np.ascontiguousarray(output))


TILE_NODES: tuple[type[Node], ...] = (ImageTileSplit, ImageTileMerge)


__all__ = ["TILE_NODES", "ImageTileMerge", "ImageTileSplit"]
