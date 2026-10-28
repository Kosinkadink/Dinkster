"""Native video input through indexed, invocation-granted assets."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    VIDEO_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    SourceFilenameSpec,
    TypeExpr,
    admit_video_source,
)


class LoadVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadVideo",
            display_name="Load Video",
            category="video",
            search_terms=("import video", "open video", "video file"),
            inputs=(
                InputSpec(
                    "file",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(kind="media/video", accept=("video/*",), allow_upload=True),
                    source_filename=SourceFilenameSpec("media/video", "input"),
                ),
            ),
            outputs=(OutputSpec("VIDEO", TypeExpr.concrete(VIDEO_TYPE), preview=True),),
        )

    @classmethod
    def execute(cls, *, file: object) -> Mapping[str, object]:
        if not isinstance(file, (AssetRef, LocalAssetRef)):
            raise TypeError("file must be an AssetRef or LocalAssetRef")
        return cls.outputs(VIDEO=admit_video_source(file))
