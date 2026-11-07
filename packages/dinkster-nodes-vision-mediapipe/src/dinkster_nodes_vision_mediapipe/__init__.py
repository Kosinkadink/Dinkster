"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    TypeRegistry,
    register_asset_type,
    register_inference_types,
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
    register_inference_types(registry)
