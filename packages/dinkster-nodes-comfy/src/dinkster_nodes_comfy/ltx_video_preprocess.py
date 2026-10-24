"""Native LTXV image preprocessing."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from io import BytesIO
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
    initialize_inference_process,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)


def _encode_single_frame(output_file: BytesIO, image_array: Any, crf: int) -> None:
    av = importlib.import_module("av")
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    finally:
        container.close()


def _decode_single_frame(video_file: BytesIO) -> Any:
    av = importlib.import_module("av")
    container = av.open(video_file)
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def _preprocess(frame: Any, compression: int, torch: Any) -> Any:
    if compression == 0:
        return frame
    image_array = (
        (frame[: (frame.shape[0] // 2) * 2, : (frame.shape[1] // 2) * 2] * 255.0)
        .byte()
        .cpu()
        .numpy()
    )
    with BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, compression)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        decoded = _decode_single_frame(video_file)
    return torch.tensor(decoded, dtype=frame.dtype, device=frame.device) / 255.0


class LTXVPreprocess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVPreprocess",
            display_name="LTXV Preprocess",
            category="video/preprocessors",
            inputs=(
                InputSpec("image", _IMAGE),
                InputSpec(
                    "img_compression",
                    _INT,
                    default=35,
                    widget=NumberWidget(min=0, max=100),
                    doc="Amount of compression to apply on image.",
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE, display_name="output_image"),),
        )

    @classmethod
    def execute(cls, *, image: object, img_compression: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        frames = materialize_image(image)
        output = [_preprocess(frame, img_compression, torch) for frame in frames]
        return cls.outputs(IMAGE=torch.stack(output))


LTX_VIDEO_PREPROCESS_NODES = (LTXVPreprocess,)
