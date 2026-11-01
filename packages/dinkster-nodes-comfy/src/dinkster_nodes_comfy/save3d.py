"""Save original 3D files with authored viewport metadata."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import ClassVar, cast

from dinkster_api.v1 import (
    CORE_INT,
    CORE_STRING,
    ENCODED_MEDIA_LIMIT_BYTES,
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
    AssetError,
    AssetRef,
    AssetWriter,
    InputSpec,
    LocalAssetRef,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    attach_saved_artifact,
)


class _Save3D(Node):
    formats: ClassVar[tuple[str, ...]]
    output_type: ClassVar[str]
    label: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]]
    model_description: ClassVar[str]

    @classmethod
    def define_schema(cls) -> NodeSchema:
        detailed = cls.__name__ == "Save3DAdvanced"
        return NodeSchema(
            node_type=cls.__name__,
            display_name=cls.label,
            search_terms=cls.aliases,
            category="3d",
            output_node=True,
            idempotent=False,
            inputs=(
                InputSpec("model_3d", TypeExpr.union(*cls.formats), doc=cls.model_description),
                InputSpec(
                    "filename_prefix",
                    TypeExpr.concrete(CORE_STRING),
                    default="3d/ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec("viewport_state", TypeExpr.concrete(LOAD_3D_TYPE)),
                InputSpec(
                    "model_3d_info",
                    TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE),
                    required=False,
                    advanced=True,
                    doc=(
                        "Placement of each model in the scene: position, rotation, and scale "
                        "(Y-up world space)."
                        if detailed
                        else ""
                    ),
                ),
                InputSpec(
                    "camera_info",
                    TypeExpr.concrete(LOAD3D_CAMERA_TYPE),
                    required=False,
                    advanced=True,
                    doc=(
                        "Viewport camera information: position, look-at target, zoom, and type."
                        if detailed
                        else ""
                    ),
                ),
                *(
                    InputSpec(
                        axis,
                        TypeExpr.concrete(CORE_INT),
                        default=1024,
                        widget=NumberWidget(min=1, max=4096, step=1),
                        doc=f"Render {axis} of the viewport in pixels." if detailed else "",
                    )
                    for axis in ("width", "height")
                ),
                InputSpec(
                    "mount",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
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
        model_3d: Mapping[str, object],
        viewport_state: object,
        width: int = 1024,
        height: int = 1024,
        filename_prefix: str = "3d/ComfyUI",
        model_3d_info: object = None,
        camera_info: object = None,
        mount: str | None = None,
    ) -> Mapping[str, object]:
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        source = model_3d["source"]
        if not isinstance(source, (AssetRef, LocalAssetRef)):
            raise TypeError("FILE_3D source must be a public asset identity")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        target = writer.output_target(filename_prefix, mount=mount)
        with source.open() as stream:
            refs = writer.save_stream_batch(
                target,
                (stream,),
                suffix=f".{model_3d['format'] or 'glb'}",
                media_type=source.media_type or "application/octet-stream",
                limit=ENCODED_MEDIA_LIMIT_BYTES,
            )
        state = (
            cast("dict[str, object]", viewport_state) if isinstance(viewport_state, dict) else {}
        )
        camera = camera_info if camera_info is not None else state.get("camera_info")
        placement = model_3d_info if model_3d_info is not None else state.get("model_3d_info", [])
        for ref in refs:
            attach_saved_artifact(ref.to_wire())
        return cls.outputs(
            model_3d=model_3d,
            model_3d_info=placement,
            camera_info=camera,
            width=width,
            height=height,
        )


class Save3DAdvanced(_Save3D):
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
    label = "Save 3D (Advanced)"
    aliases = ("save 3d", "export 3d model", "save mesh advanced")
    model_description = "3D model file from an upstream 3D node."


class SaveGaussianSplat(_Save3D):
    formats = (
        FILE_3D_SPLAT_ANY_TYPE,
        FILE_3D_PLY_TYPE,
        FILE_3D_SPLAT_TYPE,
        FILE_3D_SPZ_TYPE,
        FILE_3D_KSPLAT_TYPE,
    )
    output_type = FILE_3D_SPLAT_ANY_TYPE
    label = "Save Splat"
    aliases = ("save splat", "save gaussian splat", "export gaussian", "export splat")
    model_description = "A gaussian splat 3D file."


class SavePointCloud(_Save3D):
    formats = (FILE_3D_POINT_CLOUD_ANY_TYPE, FILE_3D_PLY_TYPE)
    output_type = FILE_3D_POINT_CLOUD_ANY_TYPE
    label = "Save Point Cloud"
    aliases = ("save point cloud", "save pointcloud", "export point cloud")
    model_description = "Point cloud file (.ply)"
