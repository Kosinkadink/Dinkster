"""Native temporal and spatial editing of component and encoded VIDEO values."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    VIDEO_EDIT_TYPE,
    VIDEO_TYPE,
    CustomWidgetDescriptor,
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
    effective_video_facts,
    resolver_from_env,
    save_video_stream,
    video_from_source,
)

_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


def _trim_video(
    video: object, start_time: float, duration: float, strict_duration: bool, action: str
) -> dict[str, object]:
    value = bind_video_value(video, resolver_from_env())
    source_duration = effective_video_facts(value)["duration"]
    if "components" in value and not value["edits"]:
        # Source component trims reject overruns even when strict_duration is false.
        if float(cast(float, source_duration)) < start_time + duration:
            raise ValueError(
                f"Failed to {action} video:\nSource duration: {float(cast(float, source_duration))}"
                f"\nStart time: {start_time}\nTarget duration: {duration}"
            )
        with io.BytesIO() as encoded:
            save_video_stream(value, encoded, container="mp4", source_components=True)
            value = video_from_source(encoded.getvalue())
    else:
        # File trims replace the previous duration and add raw start offsets.
        edits = cast(list[Mapping[str, object]], value["edits"])
        for edit in edits:
            if "trim" in edit:
                start_time += float(cast(Mapping[str, float], edit["trim"])["start_time"])
        value["edits"] = [edit for edit in edits if "trim" not in edit]
    return edit_video(
        value,
        {
            "trim": {"start_time": start_time, "duration": duration},
            "strict_duration": strict_duration,
        },
    )


class VideoSlice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Video Slice",
            display_name="Trim Video",
            category="video",
            search_terms=(
                "trim video duration",
                "skip first frames",
                "frame load cap",
                "start time",
            ),
            inputs=(
                InputSpec("video", _VIDEO),
                InputSpec(
                    "start_time",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-1e5, max=1e5, step=0.001),
                    doc="Start time in seconds",
                ),
                InputSpec(
                    "duration",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.001),
                    doc="Duration in seconds, or 0 for unlimited duration",
                ),
                InputSpec("strict_duration", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("video", _VIDEO),),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        start_time: float = 0.0,
        duration: float = 0.0,
        strict_duration: bool = False,
    ) -> Mapping[str, object]:
        return cls.outputs(video=_trim_video(video, start_time, duration, strict_duration, "slice"))


class VideoTrim(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoTrim",
            display_name="Trim Video (Advanced)",
            category="video",
            editor_role="video-trim",
            output_node=True,
            search_terms=("trim video duration", "skip first frames", "cut video", "start time"),
            inputs=(
                InputSpec("video", _VIDEO),
                InputSpec(
                    "trim",
                    TypeExpr.concrete(VIDEO_EDIT_TYPE),
                    default={"trim": {"start_time": 0.0, "duration": 0.0}},
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["trim"]}),
                    doc="Trim window using start/end frames.",
                ),
                InputSpec("strict_duration", _BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("video", _VIDEO, preview=True),),
        )

    @classmethod
    def execute(
        cls, *, video: object, trim: Mapping[str, object] | None, strict_duration: bool = False
    ) -> Mapping[str, object]:
        window = cast(Mapping[str, float], (trim or {}).get("trim") or {})
        start_time, duration = float(window.get("start_time", 0)), float(window.get("duration", 0))
        if duration < 0:
            raise ValueError(f"Trim duration must be >= 0, got {duration}")
        if start_time == 0 and duration == 0:
            return cls.outputs(video=bind_video_value(video, resolver_from_env()))
        return cls.outputs(video=_trim_video(video, start_time, duration, strict_duration, "trim"))


class VideoCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoCrop",
            display_name="Crop Video",
            category="video",
            editor_role="video-crop",
            output_node=True,
            search_terms=("crop video", "cut region", "spatial crop"),
            inputs=(
                InputSpec("video", _VIDEO),
                InputSpec(
                    "crop",
                    TypeExpr.concrete(VIDEO_EDIT_TYPE),
                    default={"crop": {"x": 0, "y": 0, "width": 0, "height": 0}},
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["crop"]}),
                    doc="Crop region in pixels. Zero width/height keeps the full frame.",
                ),
            ),
            outputs=(OutputSpec("video", _VIDEO, preview=True),),
        )

    @classmethod
    def execute(cls, *, video: object, crop: Mapping[str, object] | None) -> Mapping[str, object]:
        value = bind_video_value(video, resolver_from_env())
        rectangle = cast(Mapping[str, object], (crop or {}).get("crop") or {})
        edited = edit_video(
            value,
            {
                "crop": {
                    key: int(cast(int, rectangle.get(key, 0)))
                    for key in ("x", "y", "width", "height")
                }
            },
        )
        before, after = effective_video_facts(value), effective_video_facts(edited)
        if (before["width"], before["height"]) == (after["width"], after["height"]):
            return cls.outputs(video=value)
        if "components" in value and not value["edits"]:
            # Component cropping clones pixels but retains the complete original audio.
            components = dict(cast(Mapping[str, object], value["components"]))
            pixels = coerce_video(
                {"components": {**components, "audio": None}, "edits": edited["edits"]}
            )
            # Source component reconstruction uses its default encoding color space.
            edited = assemble_video(
                disassemble_video(pixels)["images"],
                audio=components["audio"],
                fps=components["fps"],
                bit_depth=str(components["bit_depth"]),
                color_space="sRGB",
            )
        return cls.outputs(video=edited)
