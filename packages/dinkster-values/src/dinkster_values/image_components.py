"""Decode one bound image source into its RGB and straight-alpha components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, cast

if TYPE_CHECKING:
    import numpy as np


class ReadableImageSource(Protocol):
    def open(self) -> BinaryIO: ...


@dataclass(frozen=True)
class ImageComponents:
    pixels: np.ndarray
    alpha: np.ndarray | None


def _alpha_pixels(frame: Any) -> np.ndarray:
    import numpy as np

    component = next(component for component in frame.format.components if component.is_alpha)
    if frame.format.is_planar:
        plane = frame.planes[component.plane]
        dtype = (
            "<f4"
            if "f32" in frame.format.name
            else "<f2"
            if "f16" in frame.format.name
            else "u1"
            if component.bits <= 8
            else ">u2"
            if frame.format.name.endswith("be")
            else "<u2"
        )
        array = np.frombuffer(plane, dtype=dtype).reshape(plane.height, -1)[:, : frame.width]
        return array.astype(np.float32) / (1 if "f" in dtype else 2**component.bits - 1)
    array = cast(np.ndarray, frame.to_ndarray())
    return array[..., -1].astype(np.float32) / (2**component.bits - 1)


def _rotation(source: ReadableImageSource) -> int:
    try:
        from PIL import Image

        with source.open() as handle, Image.open(handle) as image:
            orientation = image.getexif().get(274, 1)
    except ImportError:
        return 1
    return int(orientation)


def _orient(array: np.ndarray, orientation: int) -> np.ndarray:
    import numpy as np

    if orientation == 2:
        return array[:, ::-1]
    if orientation == 3:
        return array[::-1, ::-1]
    if orientation == 4:
        return array[::-1]
    if orientation == 5:
        return array.transpose(1, 0, *range(2, array.ndim))
    if orientation == 6:
        return np.rot90(array, -1)
    if orientation == 7:
        return np.rot90(array.transpose(1, 0, *range(2, array.ndim)), 2)
    if orientation == 8:
        return np.rot90(array)
    return array


def decode_image_components(source: ReadableImageSource) -> ImageComponents:
    """Decode an openable asset without admitting it as a VIDEO value."""
    import av
    import numpy as np

    opener = getattr(source, "open", None)
    if not callable(opener):
        raise TypeError(
            f"decode_image_components expects an asset with open(), got {type(source).__name__}"
        )
    with cast(Any, opener()) as handle, av.open(handle, mode="r") as opened:
        container = cast(Any, opened)
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            raise ValueError("image source has no decodable video stream")
        frame = cast(Any, next(container.decode(stream), None))
        if frame is None:
            raise ValueError("image source contains no decodable frame")
    depth = max(component.bits for component in frame.format.components if not component.is_alpha)
    alpha = any(component.is_alpha for component in frame.format.components)
    byte_output = depth <= 8 and (frame.format.is_rgb or frame.color_range == 2)
    format = "rgb24" if byte_output else "gbrpf32le"
    converted = (
        frame
        if frame.format.name == format
        else frame.reformat(
            format=format,
            src_colorspace=frame.colorspace if frame.colorspace not in (0, 2) else 5,
            dst_colorspace=0,
            src_color_range=2 if frame.format.is_rgb else frame.color_range or 1,
            dst_color_range=2,
        )
    )
    pixels = cast(np.ndarray, converted.to_ndarray())
    if not byte_output:
        maximum = 255 if depth <= 8 else 65535
        pixels = np.rint(np.clip(pixels, 0, 1) * maximum).astype(
            np.uint8 if depth <= 8 else np.uint16
        )
    components_alpha = _alpha_pixels(frame) if alpha else None
    orientation = _rotation(source)
    return ImageComponents(
        _orient(pixels, orientation),
        _orient(components_alpha, orientation) if components_alpha is not None else None,
    )
