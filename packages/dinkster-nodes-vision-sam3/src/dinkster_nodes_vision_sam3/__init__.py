"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .sam3 import SAM3_TrackToMask, SAM3_VideoTrack
from .sam3_detection import SAM3_Detect
from .sam3_preview import SAM3_TrackPreview

NODES = (
    SAM3_Detect,
    SAM3_TrackPreview,
    SAM3_TrackToMask,
    SAM3_VideoTrack,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
