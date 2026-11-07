"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize
from .mediapipe_detection import LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker

NODES = (
    LoadMediaPipeFaceLandmarker,
    MediaPipeFaceLandmarker,
    MediaPipeFaceMask,
    MediaPipeFaceMeshVisualize,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
