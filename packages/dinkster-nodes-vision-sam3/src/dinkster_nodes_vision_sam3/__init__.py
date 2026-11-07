"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    TypeRegistry,
    register_asset_type,
    register_inference_types,
    resolver_from_env,
)

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
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_inference_types(registry)
