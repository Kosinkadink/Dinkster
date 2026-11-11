"""Load original 3D files through granted asset inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_INT,
    FILE_3D_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    LOAD_3D_TYPE,
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
    load_file3d,
    with_identity_replacement,
)


class Load3DAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.3d.load3_d_advanced",
            display_name="Load 3D (Advanced)",
            category="3d",
            search_terms=("load mesh", "load gltf", "load glb", "load obj", "load fbx", "load stl"),
            inputs=(
                InputSpec(
                    "model_file",
                    TypeExpr.concrete(ASSET_TYPE),
                    default="none",
                    widget=AssetWidget(kind="media/model3d", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/model3d", "input"),
                ),
                InputSpec("viewport_state", TypeExpr.concrete(LOAD_3D_TYPE)),
                *(
                    InputSpec(
                        axis,
                        TypeExpr.concrete(CORE_INT),
                        default=1024,
                        widget=NumberWidget(min=1, max=4096, step=1),
                        doc=f"Render {axis} of the viewport in pixels.",
                    )
                    for axis in ("width", "height")
                ),
            ),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id), preview=name == "model_3d")
                for name, type_id in (
                    ("model_3d", FILE_3D_TYPE),
                    ("model_3d_info", LOAD3D_MODEL_INFO_TYPE),
                    ("camera_info", LOAD3D_CAMERA_TYPE),
                    ("width", CORE_INT),
                    ("height", CORE_INT),
                )
            ),
        )
        return with_identity_replacement(schema, "Load3DAdvanced")

    @classmethod
    def execute(
        cls,
        *,
        model_file: AssetRef | LocalAssetRef | str | None,
        viewport_state: object,
        width: int = 1024,
        height: int = 1024,
    ) -> Mapping[str, object]:
        state = (
            cast("dict[str, object]", viewport_state) if isinstance(viewport_state, dict) else {}
        )
        return cls.outputs(
            model_3d=load_file3d(model_file),
            model_3d_info=state.get("model_3d_info", []),
            camera_info=state.get("camera_info"),
            width=width,
            height=height,
        )
