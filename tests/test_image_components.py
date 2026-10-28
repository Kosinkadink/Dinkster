from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from av.error import ArgumentError
from dinkster_api.v1 import ImageComponents, decode_image_components, open_image_components
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


def test_decode_image_components_admission_boundaries(tmp_path: Path) -> None:
    pixels = np.array(
        [[[3, 17, 251], [44, 92, 10], [125, 231, 6]], [[29, 7, 33], [240, 80, 1], [9, 180, 63]]],
        dtype=np.uint8,
    )
    path = tmp_path / "bounded.png"
    Image.fromarray(pixels).save(path)
    asset = _asset(path)
    # Six pixels at 96 bytes each plus the fixed missing-alpha mask allowance.
    result = decode_image_components(
        asset, max_dimension=3, max_source_bytes=asset.size, max_working_set_bytes=33_344
    )
    np.testing.assert_array_equal(result.pixels, pixels.astype(np.float32) / 255)
    assert result.alpha is None
    with pytest.raises(ValueError, match="input byte limit"):
        decode_image_components(asset, max_source_bytes=asset.size - 1)
    with pytest.raises(ValueError, match="dimensions"):
        decode_image_components(asset, max_dimension=2)
    with pytest.raises(ValueError, match="working-set limit"):
        decode_image_components(asset, max_working_set_bytes=33_343)


def test_decode_image_components_source_admission_precedes_parser(tmp_path: Path) -> None:
    path = tmp_path / "not-an-image"
    path.write_bytes(b"invalid encoded image")

    class Source:
        handle: Any = None

        def open(self):
            self.handle = path.open("rb")
            return self.handle

    source = Source()
    with pytest.raises(ValueError, match="input byte limit"):
        decode_image_components(source, max_source_bytes=1)
    assert source.handle.closed


def test_decode_image_components_default_dimension_admission(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    Image.new("RGB", (16_385, 1)).save(path)
    with pytest.raises(ValueError, match="dimensions must be between 1 and 16384"):
        decode_image_components(_asset(path))


def test_decode_image_components_admits_alignment_working_set(tmp_path: Path) -> None:
    path = tmp_path / "aligned.png"
    Image.fromarray(np.arange(66, dtype=np.uint16).reshape(2, 33)).save(path)
    # Unaligned pixels fit; the 64x32 conversion scratch does not.
    with pytest.raises(ValueError, match="working-set limit"):
        decode_image_components(_asset(path), max_working_set_bytes=200_000)


@pytest.mark.parametrize("suffix", ("gif", "png"))
def test_open_image_components_preserves_animation_order_and_alpha(
    tmp_path: Path, suffix: str
) -> None:
    colors = ((19, 73, 211), (241, 5, 91), (37, 183, 13))
    arrays = [
        np.full(
            (2, 32 if suffix == "gif" else 3, 4),
            (*color, 255 if suffix == "gif" else 71 + i * 61),
            np.uint8,
        )
        for i, color in enumerate(colors)
    ]
    path = tmp_path / f"animation.{suffix}"
    images = [Image.fromarray(array) for array in arrays]
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0)
    expected_frames = [array.astype(np.float32) / 255 for array in arrays]
    if suffix == "gif":
        # GIF decodes as BGRA, whose float conversion is not byte normalization.
        with av.open(path) as container:
            expected_frames = [
                frame.to_ndarray(format="gbrapf32le") for frame in container.decode(video=0)
            ]

    with open_image_components(_asset(path)) as frames:
        decoded = list(frames)

    assert len(decoded) == 3
    for result, expected in zip(decoded, expected_frames, strict=True):
        assert result.pixels.dtype == np.float32
        np.testing.assert_array_equal(result.pixels, expected[..., :3])
        assert result.alpha is not None
        np.testing.assert_array_equal(result.alpha, expected[..., 3:])
    first = decode_image_components(_asset(path))
    np.testing.assert_array_equal(first.pixels, decoded[0].pixels)
    np.testing.assert_array_equal(first.alpha, decoded[0].alpha)


@pytest.mark.parametrize("finish", ("exhaust", "stop", "error"))
def test_open_image_components_closes_source_and_iterator_on_exit(
    tmp_path: Path, finish: str
) -> None:
    path = tmp_path / "animation.png"
    Image.new("RGB", (3, 2), (17, 92, 211)).save(
        path, save_all=True, append_images=[Image.new("RGB", (3, 2), (201, 3, 47))], duration=80
    )

    class Source:
        handle: Any = None

        def open(self):
            assert self.handle is None
            self.handle = path.open("rb")
            return self.handle

    source = Source()
    frames = None
    first = None
    try:
        with open_image_components(source) as frames:
            first = next(frames)
            assert not source.handle.closed
            if finish == "exhaust":
                assert len(list(frames)) == 1
            elif finish == "error":
                raise RuntimeError("consumer failed")
    except RuntimeError as error:
        assert finish == "error" and str(error) == "consumer failed"
    assert source.handle.closed
    assert frames is not None and first is not None
    with pytest.raises(StopIteration):
        next(frames)
    np.testing.assert_array_equal(first.pixels[0, 0], np.array([17, 92, 211], np.float32) / 255)


def test_open_image_components_checks_each_frame_before_conversion(tmp_path: Path) -> None:
    import io

    path = tmp_path / "varying.mjpeg"
    with path.open("wb") as handle:
        for size in ((3, 2), (5, 2)):
            encoded = io.BytesIO()
            Image.new("RGB", size, (61, 122, 183)).save(encoded, format="JPEG")
            handle.write(encoded.getvalue())
    with open_image_components(_asset(path), max_dimension=3) as frames:
        assert next(frames).pixels.shape == (2, 3, 3)
        with pytest.raises(ValueError, match="dimensions"):
            next(frames)
