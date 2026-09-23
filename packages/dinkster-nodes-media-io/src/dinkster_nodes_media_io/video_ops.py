"""Value-level video operations over frame batches and comfy.VIDEO values."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from fractions import Fraction

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    ComboWidget,
    CustomWidgetDescriptor,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    assemble_video,
    bind_video_value,
    coerce_video,
    disassemble_video,
    edit_video,
    resolver_from_env,
    video_meta,
)

from .video import (
    AUDIO,
    COMBO,
    FLOAT,
    IMAGE,
    INT,
    VIDEO_TYPE,
    positive_finite,
)

VIDEO = TypeExpr.concrete(VIDEO_TYPE)


def _frame_batch(images: object) -> np.ndarray:
    """Validate a frame batch for pure selection ops.

    Unlike the encode-side validator this accepts odd dimensions and does
    not scan pixels: selection never re-encodes or interprets values.
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("images must be a numpy frame batch")
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("images must be a nonempty [B,H,W,3|4] frame batch")
    if images.shape[1] < 1 or images.shape[2] < 1:
        raise ValueError("video frames must have nonzero dimensions")
    return images


def _video_value(video: object) -> dict[str, object]:
    try:
        return coerce_video(video)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"video is not a valid comfy.VIDEO value: {exc}") from exc


class VideoFrameWindow(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.window",
            display_name="Video Frame Window",
            category="video",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "skip_first_frames",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, step=1),
                ),
                InputSpec(
                    "select_every_nth",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, step=1),
                ),
                InputSpec(
                    "frame_load_cap",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, step=1),
                ),
            ),
            outputs=(
                OutputSpec("images", IMAGE, preview=True),
                OutputSpec("frame_count", INT),
            ),
            search_terms=("skip frames", "every nth", "frame cap", "trim frames", "VHS"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        skip_first_frames: object = 0,
        select_every_nth: object = 1,
        frame_load_cap: object = 0,
    ) -> Mapping[str, object]:
        batch = _frame_batch(images)
        if (
            isinstance(skip_first_frames, bool)
            or not isinstance(skip_first_frames, int)
            or skip_first_frames < 0
        ):
            raise ValueError("skip_first_frames must be a nonnegative integer")
        if (
            isinstance(select_every_nth, bool)
            or not isinstance(select_every_nth, int)
            or select_every_nth < 1
        ):
            raise ValueError("select_every_nth must be an integer of at least one")
        if (
            isinstance(frame_load_cap, bool)
            or not isinstance(frame_load_cap, int)
            or frame_load_cap < 0
        ):
            raise ValueError("frame_load_cap must be a nonnegative integer")
        selected = batch[skip_first_frames::select_every_nth]
        if frame_load_cap:
            selected = selected[:frame_load_cap]
        if selected.shape[0] < 1:
            raise ValueError("frame selection contains no frames")
        result = np.ascontiguousarray(selected)
        return cls.outputs(images=result, frame_count=int(result.shape[0]))


class VideoFrameRate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.rate",
            display_name="Video Frame Rate",
            category="video",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "fps",
                    FLOAT,
                    required=False,
                    default=24.0,
                    widget=NumberWidget(min=0.01, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "target_fps",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("images", IMAGE, preview=True),
                OutputSpec("frame_count", INT),
                OutputSpec("fps", FLOAT),
                OutputSpec("duration", FLOAT),
            ),
            search_terms=("force rate", "fps", "resample frames", "retime"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        fps: float = 24.0,
        target_fps: float = 0.0,
    ) -> Mapping[str, object]:
        batch = _frame_batch(images)
        source_rate = positive_finite(fps, "fps", minimum=0.01, maximum=1000.0)
        requested = float(target_fps)
        if requested != 0:
            positive_finite(requested, "target_fps", minimum=0.01, maximum=1000.0)
        frame_count = int(batch.shape[0])
        if requested == 0 or requested == source_rate:
            effective = source_rate
            result = np.ascontiguousarray(batch)
        else:
            # Nearest-neighbor time resample matching dinkster.load_video's
            # force_rate exactly: source frame i sits at i/fps, output ticks
            # at j/target_fps for every tick strictly before the source end
            # frame_count/fps, and ties round to the earlier frame.
            source = Fraction(source_rate).limit_denominator(1_000_000)
            target = Fraction(requested).limit_denominator(1_000_000)
            limit = Fraction(frame_count) * target / source
            total = int(limit) if limit.denominator == 1 else math.floor(limit) + 1
            indices = [
                min(math.ceil(Fraction(j) * source / target - Fraction(1, 2)), frame_count - 1)
                for j in range(total)
            ]
            effective = float(target)
            result = np.ascontiguousarray(batch[indices])
        count = int(result.shape[0])
        return cls.outputs(
            images=result,
            frame_count=count,
            fps=effective,
            duration=count / effective,
        )


class AssembleVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.assemble",
            display_name="Assemble Video",
            category="video",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "fps",
                    FLOAT,
                    required=False,
                    default=24.0,
                    widget=NumberWidget(min=0.01, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "bit_depth",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=("auto", "8", "10")),
                ),
                InputSpec(
                    "color_space",
                    COMBO,
                    required=False,
                    widget=ComboWidget(options=("sRGB", "HDR", "HDR PQ")),
                ),
                InputSpec(
                    "codec",
                    COMBO,
                    required=False,
                    default="none",
                    advanced=True,
                    widget=ComboWidget(options=("none", "auto", "h264", "av1")),
                ),
                InputSpec("audio", AUDIO, required=False),
            ),
            outputs=(OutputSpec("video", VIDEO, preview=True),),
            search_terms=("create video", "images to video", "encode", "mp4", "webm"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        fps: float = 24.0,
        bit_depth: str = "auto",
        color_space: str | None = None,
        codec: str = "none",
        audio: object = None,
    ) -> Mapping[str, object]:
        rate = positive_finite(fps, "fps", minimum=0.01, maximum=1000.0)
        video = assemble_video(
            images, fps=rate, bit_depth=bit_depth, color_space=color_space, audio=audio
        )
        if codec != "none":
            video = _video_value({**video, "preferred_codec": "h264" if codec == "auto" else codec})
        return cls.outputs(video=video)


class DisassembleVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.disassemble",
            display_name="Disassemble Video",
            category="video",
            inputs=(InputSpec("video", VIDEO),),
            outputs=(
                OutputSpec("images", IMAGE, preview=True),
                OutputSpec("frame_count", INT),
                OutputSpec("audio", AUDIO, optional=True, preview=True),
                OutputSpec("fps", FLOAT),
                OutputSpec("duration", FLOAT),
                OutputSpec("bit_depth", INT),
                OutputSpec("color_space", TypeExpr.concrete("core.string")),
            ),
            search_terms=("video to images", "extract frames", "components", "demux"),
        )

    @classmethod
    def execute(cls, *, video: object) -> Mapping[str, object]:
        value = bind_video_value(
            _video_value(video), resolver_from_env(), for_audio_extraction=True
        )
        result = disassemble_video(value)
        if result["audio"] is None:
            result["audio"] = ABSENT
        return cls.outputs(**result)


class TrimVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.trim",
            editor_role="video-trim",
            display_name="Trim Video",
            category="video",
            inputs=(
                InputSpec("video", VIDEO),
                InputSpec(
                    "video_edit",
                    TypeExpr.concrete("comfy.VIDEO_EDIT"),
                    required=False,
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["trim"]}),
                ),
                InputSpec(
                    "start_time",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-100_000.0, max=100_000.0, step=0.001),
                ),
                InputSpec(
                    "duration",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.001),
                ),
                InputSpec(
                    "strict_duration",
                    TypeExpr.concrete("core.boolean"),
                    required=False,
                    default=False,
                ),
            ),
            outputs=(OutputSpec("video", VIDEO, preview=True),),
            search_terms=("video slice", "trim video duration", "start time"),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        start_time: float = 0.0,
        duration: float = 0.0,
        strict_duration: object = False,
        video_edit: object = None,
    ) -> Mapping[str, object]:
        value = _video_value(video)
        if video_edit is not None:
            if not isinstance(video_edit, Mapping):
                raise ValueError("VIDEO_EDIT must be an object")
            trim = video_edit.get("trim")
            if trim is None:
                return cls.outputs(video=value)
            if not isinstance(trim, Mapping):
                raise ValueError("VIDEO_EDIT trim must be an object")
            start_time, duration = trim.get("start_time", 0), trim.get("duration", 0)
        return cls.outputs(
            video=edit_video(
                value,
                {
                    "trim": {"start_time": start_time, "duration": duration},
                    "strict_duration": strict_duration,
                },
            )
        )


class CropVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.crop",
            editor_role="video-crop",
            display_name="Crop Video",
            category="video",
            inputs=(
                InputSpec("video", VIDEO),
                InputSpec(
                    "video_edit",
                    TypeExpr.concrete("comfy.VIDEO_EDIT"),
                    required=False,
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["crop"]}),
                ),
                *(
                    InputSpec(key, INT, required=False, default=0, widget=NumberWidget(step=1))
                    for key in ("x", "y", "width", "height")
                ),
            ),
            outputs=(OutputSpec("video", VIDEO, preview=True),),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
        video_edit: object = None,
    ) -> Mapping[str, object]:
        value = _video_value(video)
        if video_edit is not None:
            if not isinstance(video_edit, Mapping):
                raise ValueError("VIDEO_EDIT must be an object")
            crop = video_edit.get("crop")
            if crop is None:
                return cls.outputs(video=value)
            if not isinstance(crop, Mapping):
                raise ValueError("VIDEO_EDIT crop must be an object")
            x, y = crop.get("x", 0), crop.get("y", 0)
            width, height = crop.get("width", 0), crop.get("height", 0)
        return cls.outputs(
            video=edit_video(value, {"crop": {"x": x, "y": y, "width": width, "height": height}})
        )


class ConcatenateVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.concatenate",
            display_name="Concatenate Video",
            category="video",
            input_families=(
                InputFamilySpec(
                    "videos", VIDEO, min_members=1, max_members=100, member_prefix="video"
                ),
            ),
            inputs=(
                InputSpec(
                    "codec",
                    COMBO,
                    required=False,
                    default="auto",
                    advanced=True,
                    widget=ComboWidget(options=("auto", "h264", "av1")),
                ),
                InputSpec("complete_audio", AUDIO, required=False, advanced=True),
            ),
            outputs=(OutputSpec("video", VIDEO, preview=True),),
            search_terms=("append video", "join video", "combine video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        videos: Mapping[str, object],
        codec: object = "auto",
        complete_audio: object = None,
    ) -> Mapping[str, object]:
        values = [
            _video_value(video)
            for group in videos.values()
            for video in (group if isinstance(group, list) else [group])
        ]
        if not values:
            raise ValueError("videos must not be empty")
        if isinstance(codec, list):
            codec = codec[0] if codec else "auto"
        if not isinstance(codec, str):
            raise ValueError("codec must be auto, h264, or av1")
        if isinstance(complete_audio, list):
            complete_audio = complete_audio[0] if complete_audio else None
        video = values[0] if len(values) == 1 else edit_video(values[0], {"concat": values[1:]})
        if complete_audio is not None:
            video = _video_value({**video, "complete_audio": complete_audio})
        if codec != "auto":
            video = _video_value({**video, "preferred_codec": codec})
        return cls.outputs(video=video)


class VideoInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.video.info",
            display_name="Video Info",
            category="video",
            inputs=(InputSpec("video", VIDEO),),
            outputs=(
                OutputSpec("info", TypeExpr.concrete("core.string")),
                OutputSpec("width", INT),
                OutputSpec("height", INT),
                OutputSpec("fps", FLOAT, optional=True),
                OutputSpec("frame_count", INT, optional=True),
                OutputSpec("duration", FLOAT, optional=True),
            ),
        )

    @classmethod
    def execute(cls, *, video: object) -> Mapping[str, object]:
        metadata = video_meta(_video_value(video))
        facts = metadata["effective"]
        assert isinstance(facts, Mapping)

        def rational(key: str) -> object:
            value = facts[key]
            return value[0] / value[1] if value is not None else ABSENT

        return cls.outputs(
            info=json.dumps(dict(metadata), sort_keys=True),
            width=facts["width"],
            height=facts["height"],
            fps=rational("fps"),
            duration=rational("duration"),
            frame_count=facts["frame_count"] if facts["frame_count"] is not None else ABSENT,
        )


VIDEO_OPS_NODES = (
    VideoFrameWindow,
    VideoFrameRate,
    AssembleVideo,
    DisassembleVideo,
    TrimVideo,
    CropVideo,
    ConcatenateVideo,
    VideoInfo,
)

__all__ = [
    "VIDEO_OPS_NODES",
    "AssembleVideo",
    "ConcatenateVideo",
    "DisassembleVideo",
    "TrimVideo",
    "CropVideo",
    "VideoInfo",
    "VideoFrameRate",
    "VideoFrameWindow",
]
