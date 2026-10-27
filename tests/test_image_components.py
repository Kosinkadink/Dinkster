from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import ImageComponents, decode_image_components
from dinkster_assets import AssetRef, digest_file
from PIL import Image


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == digest_file(self.path) else None


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        resolver=_Resolver(path),
    )


def _first_frame(path: Path) -> Any:
    with av.open(path) as container:
        return next(container.decode(video=0))


def test_decode_image_components_preserves_rgb_rgba_and_odd_width(tmp_path: Path) -> None:
    rgb = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]],
        dtype=np.uint8,
    )
    rgba = np.concatenate(
        (rgb, np.array([[[0], [128], [255]], [[17], [34], [51]]], dtype=np.uint8)), axis=-1
    )
    for name, pixels, expected_alpha in (
        ("rgb.png", rgb, None),
        ("rgba.png", rgba, rgba[..., 3:] / 255),
    ):
        path = tmp_path / name
        Image.fromarray(pixels).save(path)

        decoded = decode_image_components(_asset(path))

        assert isinstance(decoded, ImageComponents)
        assert decoded.pixels.dtype == np.float32
        np.testing.assert_array_equal(decoded.pixels, pixels[..., :3].astype(np.float32) / 255)
        if expected_alpha is None:
            assert decoded.alpha is None
        else:
            assert decoded.alpha is not None
            np.testing.assert_array_equal(decoded.alpha, expected_alpha.astype(np.float32))


def test_decode_image_components_preserves_sixteen_bit_grayscale(tmp_path: Path) -> None:
    grayscale = np.linspace(0, 65535, 64, dtype=np.uint16)[None]
    path = tmp_path / "sixteen-bit.png"
    Image.fromarray(grayscale).save(path)

    decoded = decode_image_components(_asset(path))

    assert decoded.pixels.dtype == np.float32
    np.testing.assert_allclose(
        decoded.pixels,
        np.repeat(grayscale[..., None], 3, axis=-1).astype(np.float32) / 65535,
    )
    assert decoded.alpha is None


@pytest.mark.parametrize(
    ("orientation", "quadrant"),
    ((1, 0), (2, 2), (3, 2), (4, 0), (5, 3), (6, 3), (7, 1), (8, 1)),
)
def test_decode_image_components_matches_reference_exif_rotation(
    tmp_path: Path, orientation: int, quadrant: int
) -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
            [[0, 255, 255], [255, 0, 255]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = orientation
    path = tmp_path / f"orientation-{orientation}.jpg"
    image.save(path, quality=100, subsampling=0, exif=exif)

    frame = _first_frame(path)
    expected = np.rot90(frame.to_ndarray(format="rgb24").astype(np.float32) / 255, k=quadrant)
    decoded = decode_image_components(_asset(path))

    assert int(round(frame.rotation // 90)) % 4 == quadrant
    np.testing.assert_array_equal(decoded.pixels, expected)


def test_decode_image_components_opens_source_once(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(path)

    class Source:
        opens = 0

        def open(self):
            self.opens += 1
            return path.open("rb")

    source = Source()
    decode_image_components(source)

    assert source.opens == 1
