"""Native ComfyUI image tile nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_IMAGE_LIST = TypeExpr.list_of(_IMAGE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 16_384


def _integer(name: str, default: int, minimum: int, maximum: int) -> InputSpec:
    return InputSpec(name, _INT, default=default, widget=NumberWidget(min=minimum, max=maximum))


def _grid_coords(
    width: int, height: int, tile_width: int, tile_height: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    coords: list[tuple[int, int, int, int]] = []
    stride_x = round(max(tile_width * 0.25, tile_width - overlap))
    stride_y = round(max(tile_height * 0.25, tile_height - overlap))
    y = 0
    while y < height:
        x = 0
        y_end = min(y + tile_height, height)
        y_start = max(0, y_end - tile_height)
        while x < width:
            x_end = min(x + tile_width, width)
            x_start = max(0, x_end - tile_width)
            coords.append((x_start, y_start, x_end, y_end))
            if x_end >= width:
                break
            x += stride_x
        if y_end >= height:
            break
        y += stride_y
    return coords


class SplitImageToTileList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplitImageToTileList",
            display_name="Split Image into List of Tiles",
            description="Splits an image into a batched list of tiles with a specified overlap.",
            category="image/batch",
            search_terms=("split image", "tile image", "slice image"),
            inputs=(
                InputSpec("image", _IMAGE),
                _integer("tile_width", 1024, 64, _MAX_RESOLUTION),
                _integer("tile_height", 1024, 64, _MAX_RESOLUTION),
                _integer("overlap", 128, 0, 4096),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE_LIST),),
        )

    @staticmethod
    def get_grid_coords(
        width: int, height: int, tile_width: int, tile_height: int, overlap: int
    ) -> list[tuple[int, int, int, int]]:
        return _grid_coords(width, height, tile_width, tile_height, overlap)

    @classmethod
    def execute(
        cls, *, image: Any, tile_width: int, tile_height: int, overlap: int
    ) -> Mapping[str, object]:
        image = materialize_image(image)
        _, height, width, _ = image.shape
        tiles = [
            image[:, y_start:y_end, x_start:x_end, :]
            for x_start, y_start, x_end, y_end in _grid_coords(
                width, height, tile_width, tile_height, overlap
            )
        ]
        return cls.outputs(IMAGE=tiles)


class ImageMergeTileList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageMergeTileList",
            display_name="Merge List of Tiles to Image",
            category="image/batch",
            search_terms=("split image", "tile image", "slice image"),
            inputs=(
                InputSpec("image_list", _IMAGE_LIST),
                _integer("final_width", 1024, 64, 32_768),
                _integer("final_height", 1024, 64, 32_768),
                _integer("overlap", 128, 0, 4096),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(
        cls, *, image_list: list[Any], final_width: int, final_height: int, overlap: int
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        tiles = [materialize_image(tile) for tile in image_list]
        first_tile = tiles[0]
        batch, tile_height, tile_width, channels = first_tile.shape
        coords = _grid_coords(final_width, final_height, tile_width, tile_height, overlap)
        canvas = torch.zeros(
            (batch, final_height, final_width, channels),
            device=first_tile.device,
            dtype=first_tile.dtype,
        )
        weights = torch.zeros(
            (batch, final_height, final_width, 1), device=first_tile.device, dtype=first_tile.dtype
        )
        if overlap > 0:
            vertical = torch.sin(
                math.pi
                * torch.linspace(
                    0, 1, tile_height, device=first_tile.device, dtype=first_tile.dtype
                )
            ).clamp(min=1e-5)
            horizontal = torch.sin(
                math.pi
                * torch.linspace(0, 1, tile_width, device=first_tile.device, dtype=first_tile.dtype)
            ).clamp(min=1e-5)
            weight_mask = (vertical[:, None] * horizontal[None, :]).unsqueeze(0).unsqueeze(-1)
        else:
            weight_mask = torch.ones(
                (1, tile_height, tile_width, 1), device=first_tile.device, dtype=first_tile.dtype
            )
        for index, (x_start, y_start, x_end, y_end) in enumerate(coords):
            if index >= len(tiles):
                break
            tile = tiles[index]
            real_height = min(y_end - y_start, tile.shape[1])
            real_width = min(x_end - x_start, tile.shape[2])
            canvas[:, y_start : y_start + real_height, x_start : x_start + real_width] += (
                tile[:, :real_height, :real_width] * weight_mask[:, :real_height, :real_width]
            )
            weights[:, y_start : y_start + real_height, x_start : x_start + real_width] += (
                weight_mask[:, :real_height, :real_width]
            )
        weights[weights == 0] = 1.0
        return cls.outputs(IMAGE=canvas / weights)
