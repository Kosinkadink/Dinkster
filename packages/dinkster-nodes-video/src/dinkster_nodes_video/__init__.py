"""Video input, assembly, editing, interpolation and mounted output."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_TYPE,
    INTERP_MODEL_TYPE,
    OPTICAL_FLOW_TYPE,
    VIDEO_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_audio_value_type,
    register_media_types,
    register_resident_type,
    register_video_edit_type,
    register_video_value_type,
    resolver_from_env,
)

from .interpolation import FrameInterpolate
from .interpolation_loader import FrameInterpolationModelLoader
from .optical_flow import OpticalFlowLoader
from .video_components import ConcatenateVideo, CreateVideo, GetVideoComponents
from .video_edits import VideoCrop, VideoSlice, VideoTrim
from .video_load import LoadVideo
from .video_save import SaveVideo
from .webm_save import SaveWEBM

NODES = (
    ConcatenateVideo,
    CreateVideo,
    FrameInterpolate,
    FrameInterpolationModelLoader,
    GetVideoComponents,
    LoadVideo,
    OpticalFlowLoader,
    SaveVideo,
    SaveWEBM,
    VideoCrop,
    VideoSlice,
    VideoTrim,
)


def register_types(registry: TypeRegistry) -> None:
    """Keep media authority and model objects in their existing codec owners."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    if VIDEO_TYPE not in registry:
        register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_media_types(registry)
    register_video_edit_type(registry)
    for type_id in (INTERP_MODEL_TYPE, OPTICAL_FLOW_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
