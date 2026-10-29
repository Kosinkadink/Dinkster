"""ComfyUI video component nodes backed by portable VIDEO values."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from io import BytesIO
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    AUDIO_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    IMAGE_TYPE,
    VIDEO_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    assemble_video,
    coerce_audio,
    coerce_video,
    disassemble_video,
    edit_video,
    effective_video_facts,
    image_input,
    save_video_stream,
    video_from_source,
)

IMAGE = TypeExpr.concrete(IMAGE_TYPE)
AUDIO = TypeExpr.concrete(AUDIO_TYPE)
VIDEO = TypeExpr.concrete(VIDEO_TYPE)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
COMBO = TypeExpr.concrete(CORE_COMBO)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


class CreateVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CreateVideo",
            display_name="Create Video",
            category="video",
            description="Create a video from images.",
            inputs=(
                InputSpec("images", IMAGE, doc="The images to create a video from."),
                InputSpec(
                    "fps",
                    FLOAT,
                    default=30.0,
                    widget=NumberWidget(min=1.0, max=120.0, step=1.0),
                ),
                InputSpec("audio", AUDIO, required=False, doc="The audio to add to the video."),
                InputSpec(
                    "bit_depth",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=("auto", 8, 10)),
                    doc=(
                        "Auto uses 8-bit for sRGB and 10-bit for HDR. Explicit 8-bit and "
                        "10-bit choices are independent of colorspace."
                    ),
                ),
                InputSpec(
                    "color_space",
                    COMBO,
                    required=False,
                    default="sRGB",
                    widget=ComboWidget(options=("sRGB", "HDR", "HDR PQ")),
                    doc=(
                        "Colorspace of the input images. HDR selects BT.2020/HLG and HDR PQ "
                        "selects BT.2020/PQ."
                    ),
                ),
                InputSpec(
                    "codec",
                    COMBO,
                    required=False,
                    default="none",
                    advanced=True,
                    widget=ComboWidget(options=("none", "auto", "h264", "av1")),
                    doc=(
                        "Optionally encode the video immediately. None keeps the images in "
                        "tensor form; Auto uses H.264."
                    ),
                ),
            ),
            outputs=(OutputSpec("VIDEO", VIDEO, preview=True),),
            search_terms=("images to video",),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        fps: float,
        audio: object | None = None,
        bit_depth: str | int = "auto",
        color_space: str = "sRGB",
        codec: str = "none",
    ) -> Mapping[str, object]:
        depth = 10 if bit_depth == "auto" and color_space in ("HDR", "HDR PQ") else 8
        if bit_depth != "auto":
            depth = int(bit_depth)
        video = assemble_video(
            np.asarray(image_input(images)),
            fps=fps,
            audio=coerce_audio(audio) if audio is not None else None,
            bit_depth=str(depth),
            color_space=color_space,
        )
        if codec == "none":
            return cls.outputs(VIDEO=video)
        encoded = BytesIO()
        save_video_stream(video, encoded, container="mkv", codec=codec)
        return cls.outputs(VIDEO=video_from_source(encoded.getvalue()))


class GetVideoComponents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetVideoComponents",
            display_name="Get Video Components",
            category="video",
            description="Extracts video frames, audio, frame rate, bit depth, and color space.",
            inputs=(InputSpec("video", VIDEO, doc="The video to extract components from."),),
            outputs=(
                OutputSpec("images", IMAGE, preview=True),
                OutputSpec("audio", AUDIO, optional=True, preview=True),
                OutputSpec("fps", FLOAT),
                OutputSpec("bit_depth", COMBO),
                OutputSpec("color_space", COMBO),
            ),
            search_terms=("extract frames", "split video", "video to images", "demux"),
        )

    @classmethod
    def execute(cls, *, video: object) -> Mapping[str, object]:
        value = coerce_video(video)
        if "components" in value and not value["edits"]:
            components = cast(Mapping[str, object], value["components"])
            audio = components["audio"]
            return cls.outputs(
                images=np.asarray(image_input(components["images"])),
                audio=coerce_audio(audio) if audio is not None else ABSENT,
                fps=float(cast(Fraction, components["fps"])),
                bit_depth=components["bit_depth"],
                color_space=components["color_space"],
            )

        components = disassemble_video(value)
        audio = components["audio"]
        return cls.outputs(
            images=np.asarray(image_input(components["images"])),
            audio=coerce_audio(audio) if audio is not None else ABSENT,
            fps=components["fps"],
            bit_depth=components["bit_depth"],
            color_space=components["color_space"],
        )


class VideoSlice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Video Slice",
            display_name="Trim Video",
            category="video",
            inputs=(
                InputSpec("video", VIDEO),
                InputSpec(
                    "start_time",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-100_000.0, max=100_000.0, step=0.001),
                    doc="Start time in seconds",
                ),
                InputSpec(
                    "duration",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.001),
                    doc="Duration in seconds, or 0 for unlimited duration",
                ),
                InputSpec(
                    "strict_duration",
                    BOOLEAN,
                    default=False,
                    doc=(
                        "If True, when the specified duration is not possible, an error will "
                        "be raised."
                    ),
                ),
            ),
            outputs=(OutputSpec("VIDEO", VIDEO, preview=True),),
            search_terms=(
                "trim video duration",
                "skip first frames",
                "frame load cap",
                "start time",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        start_time: float,
        duration: float,
        strict_duration: bool,
    ) -> Mapping[str, object]:
        trimmed = edit_video(
            video,
            {
                "trim": {"start_time": start_time, "duration": duration},
                "strict_duration": strict_duration,
            },
        )
        effective_video_facts(trimmed)
        return cls.outputs(VIDEO=trimmed)


VIDEO_COMPONENT_NODES = (CreateVideo, GetVideoComponents, VideoSlice)
