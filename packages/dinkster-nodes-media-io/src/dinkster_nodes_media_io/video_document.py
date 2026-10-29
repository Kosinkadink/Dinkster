"""Graph equivalents of timeline document authoring and interchange commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from dinkster_api.v1 import (
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    resolver_from_env,
    timeline_document,
    timeline_render,
    timeline_runtime,
    timeline_video,
    video_document,
)

DOCUMENT_TYPE = video_document.DOCUMENT_TYPE
document = video_document.document
parse_json = video_document.parse_json
video_reference = video_document.video_reference
COMMAND_NODES = timeline_document.COMMAND_NODES
make = timeline_document.make
mutate = timeline_document.mutate
import_otio = timeline_document.import_otio
export_otio = timeline_document.export_otio
compile_timeline = timeline_render.compile_timeline
single_clip_video = timeline_render.single_clip_video
SourceMedia = timeline_runtime.SourceMedia
TimelineVideo = timeline_video.TimelineVideo

DOCUMENT = TypeExpr.concrete(DOCUMENT_TYPE)
STRING = TypeExpr.concrete("core.string")


class MakeVideoDocument(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES["make"],
            display_name="Make Video Document",
            category="video/timeline",
            inputs=(
                InputSpec(
                    "params",
                    STRING,
                    required=False,
                    default="{}",
                    widget=StringWidget(multiline=True),
                ),
            ),
            outputs=(OutputSpec("document", DOCUMENT),),
        )

    @classmethod
    def execute(cls, *, params: str = "{}") -> Mapping[str, object]:
        return cls.outputs(document=make(**parse_json(params)))


class _Mutation(Node):
    command: ClassVar[str]

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES[cls.command],
            display_name="Timeline " + cls.command.replace("_", " ").title(),
            category="video/timeline",
            inputs=(
                InputSpec("document", DOCUMENT),
                InputSpec("params", STRING, widget=StringWidget(multiline=True)),
            ),
            outputs=(OutputSpec("document", DOCUMENT),),
        )

    @classmethod
    def execute(cls, *, document: object, params: str) -> Mapping[str, object]:
        return cls.outputs(document=mutate(document, cls.command, parse_json(params)))


class AddTimelineClip(_Mutation):
    command = "add_clip"


class AddTimelineTrack(_Mutation):
    command = "add_track"


class SetTimelineEffect(_Mutation):
    command = "set_effect"


class TimelineTransition(_Mutation):
    command = "transition"


class RetimeTimelineClip(_Mutation):
    command = "retime"


class MixTimelineAudio(_Mutation):
    command = "mix_audio"


class SplitTimelineClip(_Mutation):
    command = "split"


class MoveTimelineClip(_Mutation):
    command = "move"


class TrimTimelineClip(_Mutation):
    command = "trim"


class RippleTimelineClip(_Mutation):
    command = "ripple"


class RollTimelineClip(_Mutation):
    command = "roll"


class BindTimelineSource(_Mutation):
    command = "bind_source"

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES[cls.command],
            display_name="Bind Timeline Source",
            category="video/timeline",
            inputs=(
                InputSpec("document", DOCUMENT),
                InputSpec("params", STRING, widget=StringWidget(multiline=True)),
                InputSpec("video", TypeExpr.concrete("comfy.VIDEO"), required=False),
            ),
            outputs=(OutputSpec("document", DOCUMENT),),
        )

    @classmethod
    def execute(
        cls, *, document: object, params: str, video: object = None
    ) -> Mapping[str, object]:
        fields = parse_json(params)
        if video is not None:
            if "reference" in fields:
                raise ValueError("bind_source accepts either a VIDEO port or a reference, not both")
            fields["reference"] = video_reference(video)
        return cls.outputs(document=mutate(document, cls.command, fields))


class RenderVideoDocument(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES["render"],
            display_name="Render Video Document",
            category="video/timeline",
            inputs=(InputSpec("document", DOCUMENT),),
            outputs=(OutputSpec("video", TypeExpr.concrete("comfy.VIDEO")),),
        )

    @classmethod
    def execute(cls, *, document: object) -> Mapping[str, object]:
        return cls.outputs(video=render_document(document))


def render_document(value: object) -> dict[str, object]:
    obj = document(value)
    resolver = resolver_from_env()

    def factory(wire: Mapping[str, object]) -> Any:
        return AssetRef.from_wire(wire, resolver)

    media = SourceMedia(factory)
    single = single_clip_video(obj, media)
    if single is not None:
        return single
    compile_timeline(obj)
    return TimelineVideo(obj, factory)


class ImportOTIO(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES["import_otio"],
            display_name="Import OTIO",
            category="video/timeline",
            inputs=(InputSpec("otio", STRING, widget=StringWidget(multiline=True)),),
            outputs=(OutputSpec("document", DOCUMENT),),
        )

    @classmethod
    def execute(cls, *, otio: str) -> Mapping[str, object]:
        return cls.outputs(document=import_otio(otio))


class ExportOTIO(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=COMMAND_NODES["export_otio"],
            display_name="Export OTIO",
            category="video/timeline",
            inputs=(InputSpec("document", DOCUMENT),),
            outputs=(OutputSpec("otio", STRING),),
        )

    @classmethod
    def execute(cls, *, document: object) -> Mapping[str, object]:
        return cls.outputs(otio=export_otio(document))


VIDEO_DOCUMENT_NODES = [
    MakeVideoDocument,
    AddTimelineClip,
    AddTimelineTrack,
    SetTimelineEffect,
    TimelineTransition,
    RetimeTimelineClip,
    MixTimelineAudio,
    SplitTimelineClip,
    MoveTimelineClip,
    TrimTimelineClip,
    RippleTimelineClip,
    RollTimelineClip,
    BindTimelineSource,
    RenderVideoDocument,
    ImportOTIO,
    ExportOTIO,
]
