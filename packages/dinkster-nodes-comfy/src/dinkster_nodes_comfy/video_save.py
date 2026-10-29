"""Native video saving through the public mounted-asset boundary."""

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
    VIDEO_TYPE,
    AssetError,
    AssetWriter,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    coerce_video,
    current_export_snapshot,
    effective_video_facts,
    image_input,
    report_value_diagnostic,
    save_video_stream,
)

_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_CODEC_DOC = (
    "The output video codec. Auto preserves a compatible source stream. "
    "H.264 and AV1 re-encoding support SDR, HDR (HLG), and HDR PQ."
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


def _codec_input(*, webm: bool = False, optional: bool = False) -> DynamicComboSpec:
    return DynamicComboSpec(
        "codec",
        (
            DynamicComboOption("auto"),
            *(
                DynamicComboOption(
                    codec,
                    (
                        DynamicComboSpec(
                            "encoding",
                            (
                                DynamicComboOption("auto"),
                                DynamicComboOption(
                                    "re-encode",
                                    (
                                        InputSpec(
                                            "crf",
                                            _FLOAT,
                                            default=24.0 if codec == "av1" else 18.0,
                                            widget=NumberWidget(
                                                min=0.0,
                                                max=63.0 if codec == "av1" else 51.0,
                                                step=1.0,
                                            ),
                                            doc=(
                                                "Lower values produce higher quality "
                                                "and larger files."
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            required=False,
                            display_name="encoding mode",
                            doc=(
                                "Automatic preserves compatible "
                                f"{'AV1' if codec == 'av1' else 'H.264'} "
                                "streams. Re-encode applies custom encoding options."
                            ),
                        ),
                    ),
                )
                for codec in (("av1",) if webm else ("h264", "av1"))
            ),
        ),
        required=not optional,
        doc=_CODEC_DOC,
    )


class SaveVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveVideo",
            display_name="Save Video",
            category="video",
            description="Saves the input videos to your ComfyUI output directory.",
            search_terms=("export video",),
            inputs=(
                InputSpec("video", _VIDEO, doc="The video to save."),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="video/ComfyUI",
                    widget=StringWidget(multiline=False),
                    doc=(
                        "The prefix for the file to save. This may include formatting information "
                        "such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values "
                        "from nodes."
                    ),
                ),
                InputSpec(
                    "mount",
                    _STRING,
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "format",
                    tuple(
                        DynamicComboOption(name, (_codec_input(webm=name == "webm"),))
                        for name in ("auto", "mp4", "mkv", "webm")
                    ),
                    doc=(
                        "The output container. Auto uses MP4 for Auto/H.264 and WebM for AV1. "
                        "MP4, MKV, and WebM select a specific container."
                    ),
                ),
                _codec_input(optional=True),
            ),
            input_order=("video", "filename_prefix", "format", "codec", "mount"),
            outputs=(OutputSpec("video", _VIDEO, preview=True, doc="The input video, unchanged."),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        filename_prefix: str,
        format: str | Mapping[str, Any],
        codec: str | Mapping[str, Any] | None = None,
        mount: str | None = None,
        **values: Any,
    ) -> Mapping[str, object]:
        if isinstance(format, Mapping):
            format_name = format["format"]
            codec = format.get("codec") or codec
        else:
            format_name = format
            codec = values.get("format.codec") or codec
        if isinstance(codec, Mapping):
            codec_name = codec["codec"]
            encoding: Mapping[str, Any] = codec.get("encoding") or {}
            crf = encoding.get("crf")
        else:
            codec_name = codec or "auto"
            prefix = "format.codec." if "format.codec" in values else "codec."
            crf = values.get(prefix + "encoding.crf")
        if format_name == "auto":
            format_name = "webm" if codec_name == "av1" else "mp4"
        if format_name not in ("mp4", "mkv", "webm"):
            raise ValueError(f"Unsupported video container: {format_name!r}")
        if codec_name not in ("auto", "h264", "av1"):
            raise ValueError(f"Unsupported video codec: {codec_name!r}")
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        facts = effective_video_facts(coerce_video(video))
        target = writer.output_target(
            filename_prefix,
            mount,
            image_width=cast(int, facts["width"]),
            image_height=cast(int, facts["height"]),
        )
        submitted = current_export_snapshot()
        metadata: dict[str, object] = {}
        if submitted is not None:
            metadata.update(submitted.extra_pnginfo or {})
            metadata["prompt"] = submitted.prompt
        with _EncodedVideo() as encoded:
            suffix, media_type = save_video_stream(
                video,
                cast(BinaryIO, encoded),
                container=format_name,
                codec=codec_name,
                crf=crf,
                metadata=metadata or None,
                on_diagnostic=lambda record: report_value_diagnostic(
                    str(record["code"]),
                    {"outputId": "video", **{k: v for k, v in record.items() if k != "code"}},
                ),
            )
            encoded.seek(0)
            writer.save_stream(
                target,
                cast(BinaryIO, encoded),
                suffix=suffix,
                media_type=media_type,
                limit=ENCODED_MEDIA_LIMIT_BYTES,
            )
        return cls.outputs(video=video)


class SaveWEBM(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveWEBM",
            display_name="Save WEBM",
            category="video",
            search_terms=("export webm",),
            inputs=(
                InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "filename_prefix",
                    _STRING,
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
                    _FLOAT,
                    default=24.0,
                    widget=NumberWidget(min=0.01, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "crf",
                    _FLOAT,
                    default=32.0,
                    widget=NumberWidget(min=0.0, max=63.0, step=1.0),
                ),
                InputSpec(
                    "mount",
                    _STRING,
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
                stream.options = {"crf": str(crf)}
                if codec == "av1":
                    stream.options["preset"] = "6"
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


__all__ = ["SaveVideo", "SaveWEBM"]
