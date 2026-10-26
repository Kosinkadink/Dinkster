"""Public fixture nodes for portable native VIDEO transport."""

from __future__ import annotations

import json
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_STRING,
    VIDEO_EDIT_TYPE,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    admit_video_source,
    edit_video,
    effective_video_facts,
    register_video_edit_type,
    register_video_value_type,
    resolver_from_env,
    video_fingerprint,
)
from test_video_runtime import _source

VIDEO = TypeExpr.concrete(VIDEO_TYPE)
VIDEO_EDIT = TypeExpr.concrete(VIDEO_EDIT_TYPE)

EDIT = {"trim": {"duration": 1.0, "start_time": 0.5}, "unused": {"preserved": True}}


def register_public_types(registry: TypeRegistry) -> None:
    if VIDEO_TYPE not in registry:
        register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_video_edit_type(registry)


class VideoSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-native-value-fixture.source",
            outputs=(OutputSpec("video", VIDEO), OutputSpec("video_edit", VIDEO_EDIT)),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(video=admit_video_source(_source(depth=10)), video_edit=EDIT)


class VideoConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-native-value-fixture.consumer",
            inputs=(InputSpec("video", VIDEO), InputSpec("video_edit", VIDEO_EDIT)),
            outputs=(
                OutputSpec("source", VIDEO),
                OutputSpec("edited", VIDEO),
                OutputSpec("video_edit", VIDEO_EDIT),
                OutputSpec("facts", TypeExpr.concrete(CORE_STRING)),
                OutputSpec("fingerprint", TypeExpr.concrete(CORE_STRING)),
            ),
        )

    @classmethod
    def execute(cls, *, video: object, video_edit: object) -> Mapping[str, object]:
        assert isinstance(video_edit, Mapping)
        edited = edit_video(video, {"trim": video_edit["trim"]})
        return cls.outputs(
            source=video,
            edited=edited,
            video_edit=video_edit,
            facts=json.dumps(effective_video_facts(edited), default=str, sort_keys=True),
            fingerprint=video_fingerprint(VIDEO_TYPE)(edited),
        )


NODES = (VideoSource, VideoConsumer)
