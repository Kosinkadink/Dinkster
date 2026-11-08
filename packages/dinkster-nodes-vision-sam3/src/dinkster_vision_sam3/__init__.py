"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    MODEL_TYPE,
    SAM3_TRACK_DATA_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_box_types,
    register_media_types,
    register_resident_type,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.sam3_wire import register_sam3_track_type

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
    register_media_types(registry)
    register_box_types(registry)
    if CONDITIONING_TYPE not in registry:
        register_conditioning_type(registry, resident_table=default_pool())
    if MODEL_TYPE not in registry:
        register_resident_type(registry, MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta)
    if SAM3_TRACK_DATA_TYPE not in registry:
        register_sam3_track_type(registry)


inference_nodes(NODES)
