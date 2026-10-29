"""Native trim and crop controls over the public VIDEO edit plan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    VIDEO_EDIT_TYPE,
    VIDEO_TYPE,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    edit_video,
)

_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_EDIT = TypeExpr.concrete(VIDEO_EDIT_TYPE)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


class VideoTrim(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoTrim",
            display_name="Trim Video (Advanced)",
            category="video",
            editor_role="video-trim",
            search_terms=("trim video duration", "skip first frames", "cut video", "start time"),
            inputs=(
                InputSpec("video", _VIDEO),
                InputSpec(
                    "trim",
                    _EDIT,
                    default={"trim": {"start_time": 0.0, "duration": 0.0}},
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["trim"]}),
                    doc="Trim window using start/end frames.",
                ),
                InputSpec(
                    "strict_duration",
                    _BOOLEAN,
                    default=False,
                    advanced=True,
                    doc=(
                        "If True, when the specified duration is not possible, "
                        "an error will be raised."
                    ),
                ),
            ),
            outputs=(OutputSpec("VIDEO", _VIDEO, preview=True),),
            output_node=True,
        )

    @classmethod
    def execute(
        cls, *, video: object, trim: Mapping[str, Any] | None, strict_duration: bool
    ) -> Mapping[str, object]:
        parameters: Mapping[str, Any] = (trim or {}).get("trim") or {}
        start = float(parameters.get("start_time", 0.0))
        duration = float(parameters.get("duration", 0.0))
        if duration < 0:
            raise ValueError(f"Trim duration must be >= 0, got {duration}")
        if start == 0 and duration == 0:
            return cls.outputs(VIDEO=video)
        return cls.outputs(
            VIDEO=edit_video(
                video,
                {
                    "trim": {"start_time": start, "duration": duration},
                    "strict_duration": strict_duration,
                },
            )
        )


class VideoCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VideoCrop",
            display_name="Crop Video",
            category="video",
            editor_role="video-crop",
            search_terms=("crop video", "cut region", "spatial crop"),
            inputs=(
                InputSpec("video", _VIDEO),
                InputSpec(
                    "crop",
                    _EDIT,
                    default={"crop": {"x": 0, "y": 0, "width": 0, "height": 0}},
                    widget=CustomWidgetDescriptor("VIDEO_EDIT", {"features": ["crop"]}),
                    doc="Crop region in pixels. Zero width/height keeps the full frame.",
                ),
            ),
            outputs=(OutputSpec("VIDEO", _VIDEO, preview=True),),
            output_node=True,
        )

    @classmethod
    def execute(cls, *, video: object, crop: Mapping[str, Any] | None) -> Mapping[str, object]:
        parameters: Mapping[str, Any] = (crop or {}).get("crop") or {}
        return cls.outputs(
            VIDEO=edit_video(
                video,
                {
                    "crop": {
                        key: int(parameters.get(key, 0)) for key in ("x", "y", "width", "height")
                    }
                },
            )
        )


VIDEO_EDIT_NODES = (VideoTrim, VideoCrop)
