from __future__ import annotations

from pathlib import Path

import numpy as np
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
        ("rgba.png", rgba, rgba[..., 3] / 255),
    ):
        path = tmp_path / name
        Image.fromarray(pixels).save(path)

        decoded = decode_image_components(_asset(path))

        assert isinstance(decoded, ImageComponents)
        np.testing.assert_array_equal(decoded.pixels, pixels[..., :3])
        if expected_alpha is None:
            assert decoded.alpha is None
        else:
            assert decoded.alpha is not None
            np.testing.assert_allclose(decoded.alpha, expected_alpha, atol=1 / 65535)


def test_decode_image_components_preserves_sixteen_bit_grayscale(tmp_path: Path) -> None:
    grayscale = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "sixteen-bit.png"
    Image.fromarray(grayscale).save(path)

    decoded = decode_image_components(_asset(path))

    assert decoded.pixels.dtype == np.uint16
    np.testing.assert_array_equal(decoded.pixels, np.repeat(grayscale[..., None], 3, axis=-1))
    assert decoded.alpha is None


def test_decode_image_components_applies_orientation(tmp_path: Path) -> None:
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]], [[0, 255, 255], [255, 0, 255]]],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.jpg"
    image.save(path, quality=100, subsampling=0, exif=exif)

    decoded = decode_image_components(_asset(path))

    with Image.open(path) as source:
        expected = np.asarray(source.transpose(Image.Transpose.ROTATE_270))
    np.testing.assert_array_equal(decoded.pixels, expected)
