"""Decode one bound image source into its RGB and straight-alpha components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, cast

from .limits import IMAGE_WORKING_SET_LIMIT_BYTES

if TYPE_CHECKING:
    import numpy as np


class ReadableImageSource(Protocol):
    def open(self) -> BinaryIO: ...


@dataclass(frozen=True)
class ImageComponents:
    pixels: np.ndarray
    alpha: np.ndarray | None


def _rotation_quadrant(frame: Any) -> int:
    return int(round(frame.rotation // 90)) % 4 if frame.rotation else 0


def _float_pixels(frame: Any) -> np.ndarray:
    import numpy as np

    byte_order = ">" if frame.format.is_big_endian else "<"
    dtype = np.dtype(f"{byte_order}f{frame.format.components[0].bits // 8}")
    planes: list[Any] = []
    for component in frame.format.components:
        plane = frame.planes[component.plane]
        planes.append(
            np.ndarray(
                (frame.height, frame.width),
                dtype=dtype,
                buffer=plane,
                strides=(plane.line_size, dtype.itemsize),
            )
        )
    return np.stack(planes * 3 if len(planes) == 1 else planes, axis=-1, dtype=np.float32)


def _convert(frame: Any, image_format: str, time_base: object) -> np.ndarray:
    import av
    import numpy as np

    if frame.format.name in {
        "gbrpf16le",
        "gbrpf16be",
        "gbrapf16le",
        "gbrapf16be",
        "grayf16le",
        "grayf16be",
        "gbrpf32le",
        "gbrpf32be",
        "gbrapf32le",
        "gbrapf32be",
        "grayf32le",
        "grayf32be",
    }:
        return _float_pixels(frame)
    if image_format not in {"gbrpf32le", "gbrapf32le"} or frame.width % 32 == 0:
        return cast(np.ndarray, frame.to_ndarray(format=image_format))
    width = ((frame.width + 31) // 32) * 32
    height = ((frame.height + 31) // 32) * 32
    graph = cast(Any, av).filter.Graph()
    source = graph.add_buffer(
        width=frame.width,
        height=frame.height,
        format=frame.format.name,
        time_base=time_base,
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
    return np.ascontiguousarray(
        sink.pull().to_ndarray(format=image_format)[: frame.height, : frame.width]
    )


def _check_dimensions(
    width: int, height: int, max_dimension: int, max_working_set_bytes: int
) -> None:
    if min(width, height) < 1 or max(width, height) > max_dimension:
        raise ValueError(f"image dimensions must be between 1 and {max_dimension}")
    # Decoded planes, conversion/rotation scratch, RGB/mask lists and stacked
    # outputs may coexist. The fixed allowance covers a missing-alpha 64x64 mask.
    if width * height * 96 + 32_768 > max_working_set_bytes:
        raise ValueError("decoded image exceeds the image working-set limit")


def decode_image_components(
    source: ReadableImageSource,
    *,
    max_dimension: int = 16_384,
    max_working_set_bytes: int = IMAGE_WORKING_SET_LIMIT_BYTES,
    max_source_bytes: int = IMAGE_WORKING_SET_LIMIT_BYTES,
) -> ImageComponents:
    """Decode bounded first-frame RGB/alpha components from a seekable asset.

    Source length is checked before FFmpeg opens the asset. Stream dimensions
    are checked before explicit decoding; frame dimensions and aligned scratch
    are checked before conversion. FFmpeg probing is not an allocation sandbox.
    """
    for name, limit in (
        ("max_dimension", max_dimension),
        ("max_working_set_bytes", max_working_set_bytes),
        ("max_source_bytes", max_source_bytes),
    ):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    opener = getattr(source, "open", None)
    if not callable(opener):
        raise TypeError(
            f"decode_image_components expects an asset with open(), got {type(source).__name__}"
        )
    with cast(Any, opener()) as handle:
        position = handle.tell()
        source_size = handle.seek(0, 2)
        handle.seek(position)
        if source_size > max_source_bytes:
            raise ValueError("image asset exceeds the input byte limit")
        return _decode_components(handle, max_dimension, max_working_set_bytes)


def _decode_components(
    handle: BinaryIO, max_dimension: int, max_working_set_bytes: int
) -> ImageComponents:
    import av
    import numpy as np

    with av.open(handle, mode="r") as opened:
        container = cast(Any, opened)
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            raise ValueError("image source has no decodable video stream")
        _check_dimensions(
            stream.codec_context.width,
            stream.codec_context.height,
            max_dimension,
            max_working_set_bytes,
        )
        frame = cast(Any, next(container.decode(stream), None))
        if frame is None:
            raise ValueError("image source contains no decodable frame")
        _check_dimensions(frame.width, frame.height, max_dimension, max_working_set_bytes)
        alpha = (
            any(component.is_alpha for component in frame.format.components)
            or frame.format.name == "pal8"
        )
        byte_format = frame.format.name in {
            "yuvj420p",
            "yuvj422p",
            "yuvj444p",
            "rgb24",
            "rgba",
            "pal8",
        }
        image_format = (
            "rgba"
            if alpha and byte_format
            else "rgb24"
            if byte_format
            else "gbrapf32le"
            if alpha
            else "gbrpf32le"
        )
        if image_format in {"gbrpf32le", "gbrapf32le"} and frame.width % 32:
            aligned_width = ((frame.width + 31) // 32) * 32
            aligned_height = ((frame.height + 31) // 32) * 32
            _check_dimensions(
                aligned_width, aligned_height, max_dimension + 31, max_working_set_bytes
            )
        image = _convert(frame, image_format, stream.time_base)
        if byte_format:
            image = image.astype(np.float32) / 255.0
        quadrant = _rotation_quadrant(frame)
        if quadrant:
            image = np.rot90(image, k=quadrant, axes=(0, 1)).copy()
    return ImageComponents(image[..., :3], image[..., -1:] if alpha else None)
