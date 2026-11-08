"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    MODEL_TYPE,
    POSE_KEYPOINT_TYPE,
    VAE_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_box_types,
    register_media_types,
    register_resident_type,
)
from dinkster_inference_wire.pose_keypoint import register_pose_keypoint_type

from .sdpose import SDPoseDrawKeypoints, SDPoseFaceBBoxes
from .sdpose_extract import SDPoseKeypointExtractor

NODES = (
    SDPoseDrawKeypoints,
    SDPoseFaceBBoxes,
    SDPoseKeypointExtractor,
)


def register_types(registry: TypeRegistry) -> None:
    register_media_types(registry)
    register_box_types(registry)
    for type_id in (MODEL_TYPE, VAE_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )
    if POSE_KEYPOINT_TYPE not in registry:
        register_pose_keypoint_type(registry)


inference_nodes(NODES)
