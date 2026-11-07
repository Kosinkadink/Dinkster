"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .sdpose import SDPoseDrawKeypoints, SDPoseFaceBBoxes
from .sdpose_extract import SDPoseKeypointExtractor

NODES = (
    SDPoseDrawKeypoints,
    SDPoseFaceBBoxes,
    SDPoseKeypointExtractor,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
