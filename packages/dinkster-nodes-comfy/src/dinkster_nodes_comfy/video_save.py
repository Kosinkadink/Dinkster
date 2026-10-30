"""Save canonical VIDEO values through mounted output targets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import BinaryIO, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
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
    attach_saved_artifact,
    bind_video_value,
    current_export_snapshot,
    effective_video_facts,
    resolver_from_env,
    save_video_stream,
)

from .webm_save import EncodedVideoSpool

_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


def _codec_input(codecs: tuple[str, ...], *, required: bool = True) -> DynamicComboSpec:
    options: list[DynamicComboOption] = []
    for codec in codecs:
        if codec == "auto":
            options.append(DynamicComboOption(codec))
            continue
        options.append(
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
                                        TypeExpr.concrete(CORE_FLOAT),
                                        default=18.0 if codec == "h264" else 24.0,
                                        widget=NumberWidget(
                                            min=0.0, max=51.0 if codec == "h264" else 63.0, step=1.0
                                        ),
                                        doc="Lower values produce higher quality and larger files.",
                                    ),
                                ),
                            ),
                        ),
                        required=False,
                        display_name="encoding mode",
                        doc=(
                            f"Automatic preserves compatible {codec.upper()} streams. "
                            "Re-encode applies custom encoding options."
                        ),
                    ),
                ),
            )
        )
    return DynamicComboSpec(
        "codec",
        tuple(options),
        required=required,
        doc=(
            "The output video codec. Auto preserves a compatible source stream. "
            "H.264 and AV1 re-encoding support SDR, HDR (HLG), and HDR PQ."
        ),
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
                    "codec",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=("auto", "h264", "av1")),
                    hidden=True,
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
                        DynamicComboOption(
                            kind,
                            (
                                _codec_input(
                                    ("auto", "av1") if kind == "webm" else ("auto", "h264", "av1")
                                ),
                            ),
                        )
                        for kind in ("auto", "mp4", "mkv", "webm")
                    ),
                    doc=(
                        "The output container. Auto uses MP4 for Auto/H.264 and WebM for AV1. "
                        "MP4, MKV, and WebM select a specific container."
                    ),
                ),
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
        filename_prefix: str = "video/ComfyUI",
        format: str | Mapping[str, object] = "auto",
        codec: str | Mapping[str, object] | None = None,
        mount: str | None = None,
        **values: object,
    ) -> Mapping[str, object]:
        if isinstance(format, Mapping):
            kind = format["format"]
            selected = format.get("codec") or codec
            encoding_prefix = "codec.encoding."
        else:
            kind = format
            nested = values.get("format.codec")
            selected = nested or codec
            encoding_prefix = "format.codec.encoding." if nested else "codec.encoding."
        if isinstance(selected, Mapping):
            fields = cast(Mapping[str, object], selected)
            name = fields["codec"]
            encoding = cast(Mapping[str, object], fields.get("encoding") or {})
            crf = encoding.get("crf")
        else:
            name = selected or "auto"
            crf = values.get(encoding_prefix + "crf")
        if kind not in ("auto", "mp4", "mkv", "webm"):
            raise ValueError(f"Unsupported video container: {kind!r}")
        if name not in (("auto", "av1") if kind == "webm" else ("auto", "h264", "av1")):
            raise ValueError(f"Unsupported video codec: {name!r}")
        if kind == "auto":
            kind = "webm" if name == "av1" else "mp4"
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        value = bind_video_value(video, resolver_from_env())
        facts = effective_video_facts(value)
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        target = writer.output_target(
            filename_prefix,
            mount,
            image_width=cast(int, facts["width"]),
            image_height=cast(int, facts["height"]),
        )
        submitted = current_export_snapshot()
        metadata = None
        if submitted is not None:
            metadata = {**(submitted.extra_pnginfo or {}), "prompt": submitted.prompt}
        with EncodedVideoSpool() as encoded:
            suffix, mime = save_video_stream(
                value,
                cast(BinaryIO, encoded),
                container=cast(str, kind),
                codec=cast(str, name),
                crf=cast(float | None, crf),
                metadata=metadata,
                source_components="components" in value and not value["edits"],
            )
            encoded.seek(0)
            ref = writer.save_stream(
                target,
                cast(BinaryIO, encoded),
                suffix=suffix,
                media_type=mime,
                limit=ENCODED_MEDIA_LIMIT_BYTES,
            )
            attach_saved_artifact(ref.to_wire())
        return cls.outputs(video=video)
