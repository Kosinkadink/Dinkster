"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    KIMODO_POSE_DATA_TYPE,
    LOAD3D_CAMERA_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3_TRACK_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_box_types,
    register_file3d_types,
    register_load3d_camera_type,
    register_media_types,
    register_resident_type,
    register_vision_packet_types,
    resolver_from_env,
)
from dinkster_inference_wire.sam3_wire import register_sam3_track_type

from .sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth
from .sam3d_export import BuildPoseFile
from .sam3d_expression import SAM3DBody_FaceExpression
from .sam3d_prediction import SAM3DBody_Predict
from .sam3d_render import SAM3DBody_Render

NODES = (
    BuildPoseFile,
    SAM3DBody_FaceExpression,
    SAM3DBody_Loader,
    SAM3DBody_Predict,
    SAM3DBody_Render,
    SAM3DBody_Smooth,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_box_types(registry)
    register_file3d_types(registry, resolver_from_env())
    if LOAD3D_CAMERA_TYPE not in registry:
        register_load3d_camera_type(registry)
    if SAM3_TRACK_DATA_TYPE not in registry:
        with registry.registering_pack(
            "dinkster-vision-sam3", (SAM3_TRACK_DATA_TYPE,), allow_bare=True
        ):
            register_sam3_track_type(registry)
    if SAM3D_BODY_MODEL_TYPE not in registry:
        register_resident_type(
            registry, SAM3D_BODY_MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta
        )
    register_vision_packet_types(registry, type_ids=(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE))


inference_nodes(NODES)
