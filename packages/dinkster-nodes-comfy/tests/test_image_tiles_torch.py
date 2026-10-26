"""CPU behavior for native image tiles and mask preview."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.image_tiles import ImageMergeTileList, SplitImageToTileList
from dinkster_nodes_comfy.mask_preview import MaskPreview
from dinkster_schema import use_reporter


def test_split_and_merge_asymmetric_tiles_preserve_dtype_device_and_pixels() -> None:
    image = torch.arange(5 * 7, dtype=torch.float64).reshape(1, 5, 7, 1)
    tiles = SplitImageToTileList.execute(image=image, tile_width=4, tile_height=3, overlap=1)[
        "IMAGE"
    ]
    assert [tile.shape for tile in tiles] == [(1, 3, 4, 1)] * 4
    assert all(tile.dtype == image.dtype and tile.device == image.device for tile in tiles)
    merged = ImageMergeTileList.execute(image_list=tiles, final_width=7, final_height=5, overlap=1)[
        "IMAGE"
    ]
    assert torch.allclose(merged, image)


def test_merge_stops_at_missing_tiles_and_crops_partial_tiles() -> None:
    tile = torch.ones((1, 2, 2, 1), dtype=torch.float32)
    merged = ImageMergeTileList.execute(
        image_list=[tile[:, :1]], final_width=4, final_height=2, overlap=0
    )["IMAGE"]
    assert torch.equal(merged[0, 0, :, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(merged[0, 1, :, 0], torch.zeros(4))


def test_mask_preview_passes_through_mask_and_reports_png() -> None:
    mask = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
    reports: list[tuple[str, dict[str, object], bytes | None]] = []
    with use_reporter(lambda name, data, blob: reports.append((name, dict(data), blob))):
        result = MaskPreview.execute(mask=mask)["mask"]
    assert result is mask
    assert reports[0][:2] == ("preview", {"mime": "image/png", "width": 2, "height": 2})
    assert reports[0][2] is not None and reports[0][2].startswith(b"\x89PNG\r\n\x1a\n")
