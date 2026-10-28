from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from av.error import ArgumentError
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


def _reference_nonbyte_pixels(path: Path) -> np.ndarray:
    with av.open(path) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        if frame.width % 32 == 0:
            return frame.to_ndarray(format="gbrpf32le")
        width = ((frame.width + 31) // 32) * 32
        height = ((frame.height + 31) // 32) * 32
        graph = cast(Any, av).filter.Graph()
        source = graph.add_buffer(
            width=frame.width,
            height=frame.height,
            format=frame.format.name,
            time_base=stream.time_base,
        )
        pad = graph.add("pad", f"{width}:{height}:0:0")
        fill = graph.add(
            "fillborders",
            f"left=0:right={width - frame.width}:top=0:bottom={height - frame.height}:mode=smear",
        )
        sink = graph.add("buffersink")
        source.link_to(pad)
        pad.link_to(fill)
        fill.link_to(sink)
        graph.configure()
        source.push(frame)
        return sink.pull().to_ndarray(format="gbrpf32le")[: frame.height, : frame.width]


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


def test_decode_image_components_matches_reference_nonbyte_odd_width(tmp_path: Path) -> None:
    grayscale = np.linspace(0, 65535, 17 * 17, dtype=np.uint16).reshape(17, 17)
    path = tmp_path / "sixteen-bit-17x17.png"
    Image.fromarray(grayscale).save(path)

    expected = _reference_nonbyte_pixels(path)
    decoded = decode_image_components(_asset(path))

    np.testing.assert_array_equal(decoded.pixels, expected[..., :3])
    assert decoded.alpha is None


@pytest.mark.parametrize(("height", "width"), ((3, 17), (9, 33), (3, 65)))
def test_decode_image_components_matches_reference_small_nonbyte_failure(
    tmp_path: Path, height: int, width: int
) -> None:
    path = tmp_path / f"sixteen-bit-{width}x{height}.png"
    Image.fromarray(np.zeros((height, width), dtype=np.uint16)).save(path)

    with pytest.raises(ArgumentError, match="Invalid argument returned 22"):
        _reference_nonbyte_pixels(path)
    with pytest.raises(ArgumentError, match="Invalid argument returned 22"):
        decode_image_components(_asset(path))


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
