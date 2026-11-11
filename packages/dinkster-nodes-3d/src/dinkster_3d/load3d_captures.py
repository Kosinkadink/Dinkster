"""Load original 3D files and authored viewport captures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_INT,
    CORE_STRING,
    FILE_3D_TYPE,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    LOAD_3D_TYPE,
    MASK_TYPE,
    VIDEO_TYPE,
    AbsentOutput,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SourceFilenameSpec,
    TypeExpr,
    admit_video_source,
    load_file3d,
    with_identity_replacement,
)
from dinkster_image.ported.image_file_tools import LoadImage


class Load3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.3d.load3_d",
            display_name="Load 3D & Animation",
            category="3d",
            inputs=(
                InputSpec(
                    "model_file",
                    TypeExpr.concrete(ASSET_TYPE),
                    default="none",
                    widget=AssetWidget(kind="media/model3d", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/model3d", "input"),
                ),
                InputSpec("image", TypeExpr.concrete(LOAD_3D_TYPE)),
                *(
                    InputSpec(
                        axis,
                        TypeExpr.concrete(CORE_INT),
                        default=1024,
                        widget=NumberWidget(min=1, max=4096, step=1),
                    )
                    for axis in ("width", "height")
                ),
            ),
            outputs=tuple(
                OutputSpec(
                    name,
                    TypeExpr.concrete(type_id),
                    preview=name == "model_3d",
                    optional=name == "recording_video",
                )
                for name, type_id in (
                    ("image", IMAGE_TYPE),
                    ("mask", MASK_TYPE),
                    ("mesh_path", CORE_STRING),
                    ("normal", IMAGE_TYPE),
                    ("camera_info", LOAD3D_CAMERA_TYPE),
                    ("recording_video", VIDEO_TYPE),
                    ("model_3d", FILE_3D_TYPE),
                    ("model_3d_info", LOAD3D_MODEL_INFO_TYPE),
                )
            ),
        )
        return with_identity_replacement(schema, "Load3D")

    @classmethod
    def execute(
        cls,
        *,
        model_file: AssetRef | LocalAssetRef | str | None,
        image: Mapping[str, object],
        width: int = 1024,
        height: int = 1024,
    ) -> Mapping[str, object]:
        output_image = LoadImage.execute(image=cast("AssetRef | LocalAssetRef", image["image"]))[
            "IMAGE"
        ]
        output_mask = LoadImage.execute(image=cast("AssetRef | LocalAssetRef", image["mask"]))[
            "MASK"
        ]
        normal = LoadImage.execute(image=cast("AssetRef | LocalAssetRef", image["normal"]))["IMAGE"]
        recording = image["recording"]
        video = (
            admit_video_source(cast("AssetRef | LocalAssetRef", recording))
            if recording != ""
            else AbsentOutput("no recording")
        )
        packet = load_file3d(model_file)
        source = cast("AssetRef | LocalAssetRef", packet["source"]) if packet is not None else None
        return cls.outputs(
            image=output_image,
            mask=output_mask,
            mesh_path=source.name if source is not None else "",
            normal=normal,
            camera_info=image["camera_info"],
            recording_video=video,
            model_3d=packet,
            model_3d_info=image.get("model_3d_info", []),
        )
