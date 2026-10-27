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


def decode_image_components(source: ReadableImageSource) -> ImageComponents:
    """Decode an openable asset with ComfyUI's first-frame component policy."""
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
        image = _convert(frame, image_format, stream.time_base)
        if byte_format:
            image = image.astype(np.float32) / 255.0
        quadrant = _rotation_quadrant(frame)
        if quadrant:
            image = np.rot90(image, k=quadrant, axes=(0, 1)).copy()
    return ImageComponents(image[..., :3], image[..., -1:] if alpha else None)
