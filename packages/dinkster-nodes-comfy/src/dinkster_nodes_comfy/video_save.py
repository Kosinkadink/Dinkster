"""Native video saving through the public mounted-asset boundary."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any, BinaryIO, cast

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
    VIDEO_TYPE,
    AssetError,
    AssetWriter,
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
    effective_video_facts,
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
                    widget=StringWidget(),
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
        # The current public encoder accepts integral CRF only; never silently round.
        if crf is not None:
            if isinstance(crf, bool) or not isinstance(crf, (float, int)) or int(crf) != crf:
                raise ValueError("fractional CRF is not supported by the public video encoder")
            crf = int(crf)
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
        with _EncodedVideo() as encoded:
            suffix, media_type = save_video_stream(
                video,
                cast(BinaryIO, encoded),
                container=format_name,
                codec=codec_name,
                crf=crf,
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


__all__ = ["SaveVideo"]
