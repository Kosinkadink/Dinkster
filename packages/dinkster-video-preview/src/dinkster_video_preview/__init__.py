"""Initialize a bounded frontend preview from portable VIDEO metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    InputSpec,
    JsonField,
    JsonObjectSchema,
    Node,
    NodeSchema,
    OutputSpec,
    PackEvent,
    TypeExpr,
    TypeRegistry,
    effective_video_facts,
    register_video_value_type,
    report_pack_event,
    resolver_from_env,
)

INITIALIZED = PackEvent(
    "video-preview.initialized",
    JsonObjectSchema(
        (
            JsonField("fps", "number"),
            JsonField("frameCount", "integer"),
            JsonField("height", "integer"),
            JsonField("width", "integer"),
        )
    ),
)


def preview_policy(_request: Mapping[str, object]) -> dict[str, object]:
    """The same bounds used by preview initialization, available before execution."""
    return {"defaultFps": 24.0, "maxFrames": 120, "maxWidth": 512}


class InitializePreview(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        video = TypeExpr.concrete("comfy.VIDEO")
        return NodeSchema(
            node_type="video-preview.initialize",
            display_name="Initialize Video Preview",
            inputs=(InputSpec("video", video),),
            outputs=(OutputSpec("video", video),),
        )

    @classmethod
    async def execute(cls, *, video: Mapping[str, object]) -> Mapping[str, object]:
        facts = effective_video_facts(video)
        policy = preview_policy({})
        fps = float(cast(float, facts["fps"] or policy["defaultFps"]))
        width, height = cast(int, facts["width"]), cast(int, facts["height"])
        ratio = min(1.0, cast(int, policy["maxWidth"]) / width)
        frames = facts["frame_count"]
        if frames is None and facts["duration"] is not None:
            frames = math.ceil(cast(float, facts["duration"]) * fps)
        report_pack_event(
            INITIALIZED,
            {
                "fps": fps,
                "frameCount": min(
                    cast(int, policy["maxFrames"]),
                    cast(int, frames) if frames is not None else cast(int, policy["maxFrames"]),
                ),
                "height": max(1, round(height * ratio)),
                "width": max(1, round(width * ratio)),
            },
        )
        return cls.outputs(video=video)


def register_types(registry: TypeRegistry) -> None:
    register_video_value_type(registry, "comfy.VIDEO", resolver_from_env())


NODES = [InitializePreview]
