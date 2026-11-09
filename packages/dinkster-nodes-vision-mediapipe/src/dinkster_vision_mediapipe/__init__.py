"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    COLOR_TYPE,
    FACE_DETECTION_MODEL_TYPE,
    FACE_LANDMARKS_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_box_types,
    register_media_types,
    register_resident_type,
    register_vision_packet_types,
    resolver_from_env,
)

from .mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize
from .mediapipe_detection import LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker

NODES = (
    LoadMediaPipeFaceLandmarker,
    MediaPipeFaceLandmarker,
    MediaPipeFaceMask,
    MediaPipeFaceMeshVisualize,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_box_types(registry)
    if COLOR_TYPE not in registry:
        registry.register(COLOR_TYPE)
    if FACE_DETECTION_MODEL_TYPE not in registry:
        register_resident_type(
            registry, FACE_DETECTION_MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta
        )
    register_vision_packet_types(registry, type_ids=(FACE_LANDMARKS_TYPE,))


inference_nodes(NODES)
