"""Native WEBM image-sequence saving through mounted assets."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, BinaryIO, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
    IMAGE_TYPE,
    AssetError,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    current_export_snapshot,
    image_input,
)


class _EncodedVideo(tempfile.SpooledTemporaryFile[bytes]):
    def __init__(self) -> None:
        super().__init__(max_size=512 * 1024, mode="w+b")

    def write(self, s: Any) -> int:
        position = self.tell()
        self.seek(0, os.SEEK_END)
        size = self.tell()
        self.seek(position)
        if max(position + len(s), size) > ENCODED_MEDIA_LIMIT_BYTES:
            raise ValueError("encoded video exceeds the output byte limit")
        return super().write(s)


class SaveWEBM(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveWEBM",
            display_name="Save WEBM",
            category="video",
            search_terms=("export webm",),
            inputs=(
                InputSpec(
                    "images",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc=(
                        "RGBA images are saved with their alpha channel as transparency "
                        "(vp9 codec only)."
                    ),
                ),
                InputSpec(
                    "filename_prefix",
                    TypeExpr.concrete(CORE_STRING),
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "codec",
                    TypeExpr.concrete(CORE_COMBO),
                    default="vp9",
                    widget=ComboWidget(options=("vp9", "av1")),
                ),
                InputSpec(
                    "fps",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=24.0,
                    widget=NumberWidget(min=0.01, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "crf",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=32.0,
                    widget=NumberWidget(min=0.0, max=63.0, step=1.0),
                    doc=(
                        "Higher crf means lower quality with a smaller file size, "
                        "lower crf means higher quality higher filesize."
                    ),
                ),
                InputSpec(
                    "mount",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            outputs=(OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE), preview=True),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        filename_prefix: str,
        codec: str,
        fps: float,
        crf: float,
        mount: str | None = None,
    ) -> Mapping[str, object]:
        if codec not in ("vp9", "av1"):
            raise ValueError(f"Unsupported WEBM codec: {codec!r}")
        av: Any = importlib.import_module("av")
        np: Any = importlib.import_module("numpy")
        pixels = np.asarray(image_input(images))
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        target = writer.output_target(
            filename_prefix, mount, image_width=pixels.shape[2], image_height=pixels.shape[1]
        )
        save_alpha = pixels.shape[-1] == 4 and codec == "vp9"
        with _EncodedVideo() as encoded:
            with av.open(encoded, "w", format="webm") as container:
                submitted = current_export_snapshot()
                if submitted is not None:
                    container.metadata["prompt"] = json.dumps(submitted.prompt)
                    for name, value in (submitted.extra_pnginfo or {}).items():
                        container.metadata[name] = json.dumps(value)
                stream = container.add_stream(
                    {"vp9": "libvpx-vp9", "av1": "libsvtav1"}[codec],
                    rate=Fraction(round(fps * 1000), 1000),
                )
                stream.width, stream.height = pixels.shape[2], pixels.shape[1]
                stream.pix_fmt = (
                    "yuva420p" if save_alpha else "yuv420p10le" if codec == "av1" else "yuv420p"
                )
                stream.bit_rate = 0
                stream.thread_count = 1
                stream.options = {"crf": str(crf)}
                if codec == "av1":
                    stream.options.update({"preset": "6", "svtav1-params": "lp=1"})
                for frame in pixels:
                    packed = np.clip(frame[..., : 4 if save_alpha else 3] * 255, 0, 255).astype(
                        np.uint8
                    )
                    item = av.VideoFrame.from_ndarray(
                        packed, format="rgba" if save_alpha else "rgb24"
                    )
                    for packet in stream.encode(item):
                        container.mux(packet)
                container.mux(stream.encode())
            encoded.seek(0)
            writer.save_stream(
                target,
                cast(BinaryIO, encoded),
                suffix=".webm",
                media_type="video/webm",
                limit=ENCODED_MEDIA_LIMIT_BYTES,
            )
        return cls.outputs(images=images)
