"""Original-file 3D previews with viewport camera and placement outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

from dinkster_api.v1 import (
    CORE_INT,
    FILE_3D_FBX_TYPE,
    FILE_3D_GLB_TYPE,
    FILE_3D_GLTF_TYPE,
    FILE_3D_KSPLAT_TYPE,
    FILE_3D_OBJ_TYPE,
    FILE_3D_PLY_TYPE,
    FILE_3D_POINT_CLOUD_ANY_TYPE,
    FILE_3D_SPLAT_ANY_TYPE,
    FILE_3D_SPLAT_TYPE,
    FILE_3D_SPZ_TYPE,
    FILE_3D_STL_TYPE,
    FILE_3D_TYPE,
    FILE_3D_USDZ_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    LOAD_3D_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)


class _Preview3D(Node):
    formats: ClassVar[tuple[str, ...]]
    output_type: ClassVar[str]
    label: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]]
    description: ClassVar[str]

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=cls.__name__,
            display_name=cls.label,
            search_terms=cls.aliases,
            category="3d",
            description=cls.description,
            output_node=True,
            inputs=(
                InputSpec("model_3d", TypeExpr.union(*cls.formats)),
                InputSpec(
                    "model_3d_info",
                    TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE),
                    required=False,
                    advanced=True,
                ),
                InputSpec("viewport_state", TypeExpr.concrete(LOAD_3D_TYPE)),
                InputSpec(
                    "camera_info",
                    TypeExpr.concrete(LOAD3D_CAMERA_TYPE),
                    required=False,
                    advanced=True,
                ),
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
                OutputSpec(name, TypeExpr.concrete(type_id), preview=name == "model_3d")
                for name, type_id in (
                    ("model_3d", cls.output_type),
                    ("model_3d_info", LOAD3D_MODEL_INFO_TYPE),
                    ("camera_info", LOAD3D_CAMERA_TYPE),
                    ("width", CORE_INT),
                    ("height", CORE_INT),
                )
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model_3d: object,
        viewport_state: object,
        width: int,
        height: int,
        model_3d_info: object = None,
        camera_info: object = None,
    ) -> Mapping[str, object]:
        state = (
            cast("dict[str, object]", viewport_state) if isinstance(viewport_state, dict) else {}
        )
        return cls.outputs(
            model_3d=model_3d,
            model_3d_info=model_3d_info
            if model_3d_info is not None
            else state.get("model_3d_info", []),
            camera_info=camera_info if camera_info is not None else state.get("camera_info"),
            width=width,
            height=height,
        )


class Preview3DAdvanced(_Preview3D):
    formats = (
        FILE_3D_GLB_TYPE,
        FILE_3D_GLTF_TYPE,
        FILE_3D_FBX_TYPE,
        FILE_3D_OBJ_TYPE,
        FILE_3D_STL_TYPE,
        FILE_3D_USDZ_TYPE,
        FILE_3D_TYPE,
    )
    output_type = FILE_3D_TYPE
    label = "Preview 3D (Advanced)"
    aliases = ("preview", "preview 3d", "3d viewer", "view mesh", "frame 3d", "3d camera output")
    description = "Preview a 3D model file without saving it to the ComfyUI output directory."


class PreviewGaussianSplat(_Preview3D):
    formats = (
        FILE_3D_SPLAT_ANY_TYPE,
        FILE_3D_PLY_TYPE,
        FILE_3D_SPLAT_TYPE,
        FILE_3D_SPZ_TYPE,
        FILE_3D_KSPLAT_TYPE,
    )
    output_type = FILE_3D_SPLAT_ANY_TYPE
    label = "Preview Splat"
    aliases = (
        "preview",
        "preview splat",
        "preview gaussian",
        "preview gaussian splat",
        "preview 3dgs",
        "preview ply",
        "preview spz",
        "preview ksplat",
        "view splat",
        "view gaussian",
        "view gaussian splat",
        "view 3dgs",
    )
    description = (
        "Preview a gaussian splat 3D file without saving it to the ComfyUI output directory."
    )


class PreviewPointCloud(_Preview3D):
    formats = (FILE_3D_POINT_CLOUD_ANY_TYPE, FILE_3D_PLY_TYPE)
    output_type = FILE_3D_POINT_CLOUD_ANY_TYPE
    label = "Preview Point Cloud"
    aliases = (
        "preview",
        "preview point cloud",
        "preview pointcloud",
        "preview ply",
        "view point cloud",
        "view pointcloud",
    )
    description = "Preview a point cloud 3D file without saving it to the ComfyUI output directory."
