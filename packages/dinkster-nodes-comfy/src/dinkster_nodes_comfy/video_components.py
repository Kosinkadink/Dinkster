"""Native video assembly and explicit component extraction."""

from __future__ import annotations

import importlib
import io
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    IMAGE_TYPE,
    VIDEO_TYPE,
    ComboOption,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    assemble_video,
    bind_video_value,
    concatenate_video,
    copy_media_semantics,
    disassemble_video,
    resolver_from_env,
    save_video_stream,
    video_from_source,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_COMBO = TypeExpr.concrete(CORE_COMBO)


class CreateVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CreateVideo",
            display_name="Create Video",
            category="video",
            description="Create a video from images.",
            search_terms=("images to video",),
            inputs=(
                InputSpec("images", _IMAGE, doc="The images to create a video from."),
                InputSpec(
                    "fps",
                    _FLOAT,
                    default=30.0,
                    widget=NumberWidget(min=1.0, max=120.0, step=1.0),
                ),
                InputSpec("audio", _AUDIO, required=False, doc="The audio to add to the video."),
                InputSpec(
                    "bit_depth",
                    _COMBO,
                    default="auto",
                    required=False,
                    widget=ComboWidget((ComboOption("auto"), ComboOption(8), ComboOption(10))),
                    doc=(
                        "Auto uses 8-bit for sRGB and 10-bit for HDR. Explicit 8-bit and "
                        "10-bit choices are independent of colorspace."
                    ),
                ),
                InputSpec(
                    "color_space",
                    _COMBO,
                    default="sRGB",
                    required=False,
                    widget=ComboWidget(options=("sRGB", "HDR", "HDR PQ")),
                    doc=(
                        "Colorspace of the input images. HDR selects BT.2020/HLG and "
                        "HDR PQ selects BT.2020/PQ."
                    ),
                ),
                InputSpec(
                    "codec",
                    _COMBO,
                    default="none",
                    required=False,
                    advanced=True,
                    widget=ComboWidget(options=("none", "auto", "h264", "av1")),
                    doc=(
                        "Optionally encode the video immediately. None keeps the images "
                        "in tensor form; Auto uses H.264."
                    ),
                ),
            ),
            outputs=(OutputSpec("video", _VIDEO),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        fps: float,
        audio: object | None = None,
        bit_depth: int | str = "auto",
        color_space: str = "sRGB",
        codec: str = "none",
    ) -> Mapping[str, object]:
        np = importlib.import_module("numpy")
        pixels = copy_media_semantics(
            images,
            np.asarray(images.detach().cpu().numpy() if hasattr(images, "detach") else images),
        )
        if bit_depth == "auto":
            bit_depth = 10 if color_space in ("HDR", "HDR PQ") else 8
        video = assemble_video(
            pixels,
            fps=Fraction(fps),
            audio=audio,
            bit_depth=str(bit_depth),
            color_space=color_space,
        )
        if codec != "none":
            if codec not in ("auto", "h264", "av1"):
                raise ValueError(f"Unsupported video codec: {codec}")
            with io.BytesIO() as encoded:
                save_video_stream(
                    video, encoded, container="mkv", codec=codec, source_components=True
                )
                video = video_from_source(encoded.getvalue())
        return cls.outputs(video=video)


class GetVideoComponents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetVideoComponents",
            display_name="Get Video Components",
            category="video",
            description="Extracts video frames, audio, frame rate, bit depth, and color space.",
            search_terms=("extract frames", "split video", "video to images", "demux"),
            inputs=(InputSpec("video", _VIDEO, doc="The video to extract components from."),),
            outputs=(
                OutputSpec("images", _IMAGE),
                OutputSpec("audio", _AUDIO),
                OutputSpec("fps", _FLOAT),
                OutputSpec("bit_depth", _COMBO),
                OutputSpec("color_space", _COMBO),
            ),
        )

    @classmethod
    def execute(cls, *, video: object) -> Mapping[str, object]:
        value = bind_video_value(video, resolver_from_env(), for_audio_extraction=True)
        components = cast(
            "Mapping[str, Any]",
            value["components"]
            if "components" in value and not value["edits"]
            else disassemble_video(value),
        )
        return cls.outputs(
            images=components["images"],
            audio=components["audio"],
            fps=float(components["fps"]),
            bit_depth=components["bit_depth"],
            color_space=components["color_space"],
        )


class ConcatenateVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConcatenateVideo",
            display_name="Concatenate Video",
            category="video",
            description="Concatenates videos in order without decoding compatible encoded inputs.",
            input_families=(
                InputFamilySpec(
                    "videos",
                    (
                        InputSpec(
                            "video",
                            TypeExpr.list_of(_VIDEO),
                            doc="VIDEO lists to concatenate in input order.",
                        ),
                    ),
                    min_members=1,
                    max_members=100,
                    member_prefix="video",
                ),
            ),
            inputs=(
                InputSpec(
                    "codec",
                    _COMBO,
                    default="auto",
                    advanced=True,
                    widget=ComboWidget(options=("auto", "h264", "av1")),
                ),
                InputSpec("complete_audio", _AUDIO, required=False, advanced=True),
            ),
            outputs=(OutputSpec("video", _VIDEO),),
            input_order=("videos", "codec", "complete_audio"),
        )

    @classmethod
    def execute(
        cls,
        *,
        videos: Mapping[str, list[object]],
        codec: str = "auto",
        complete_audio: object | None = None,
    ) -> Mapping[str, object]:
        resolver = resolver_from_env()
        bound = [bind_video_value(video, resolver) for group in videos.values() for video in group]
        return cls.outputs(
            video=concatenate_video(bound, codec=codec, complete_audio=complete_audio)
        )
