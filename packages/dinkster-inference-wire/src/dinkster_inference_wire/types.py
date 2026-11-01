"""Value type registration retained outside the fork-backed runtime."""

from __future__ import annotations

from functools import partial

from dinkster_memory.resident import default_pool
from dinkster_values import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_ENCODER_TYPE,
    BACKGROUND_REMOVAL_TYPE,
    BOUNDING_BOXES_TYPE,
    CAMERA_INFO_STATE_TYPE,
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CLIP_VISION_TYPE,
    COLOR_TYPE,
    COLORS_TYPE,
    COMPOSITOR_TYPE,
    CONTROL_NET_TYPE,
    DA3_MODEL_TYPE,
    FACE_DETECTION_MODEL_TYPE,
    FLOATS_TYPE,
    GLIGEN_TYPE,
    GUIDER_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    IC_LORA_PARAMETERS_TYPE,
    IMAGECOMPARE_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    INTERP_MODEL_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    LAYERS_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    MOGE_GEOMETRY_TYPE,
    MOGE_MODEL_TYPE,
    NOISE_TYPE,
    OPTICAL_FLOW_TYPE,
    PHOTOMAKER_TYPE,
    POSE_KEYPOINT_TYPE,
    RESOLUTION_PREVIEW_TYPE,
    SAM3_TRACK_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    STYLE_MODEL_TYPE,
    TIMESTEPS_RANGE_TYPE,
    TRACKS_TYPE,
    UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    VIDEO_EDIT_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    latent_fingerprint,
    register_box_types,
    register_resident_type,
    register_splat_type,
    register_video_edit_type,
    validate_latent_encoded,
)

from .camera_wire import register_load3d_camera_type, register_load3d_model_info_type
from .conditioning_wire import CONDITIONING_TYPE_ID, register_conditioning_type
from .geometry_wire import materialize_mesh, register_mesh_type
from .hook_values import (
    normalize_clip_hook_owners,
    normalize_hook_owners,
    register_hook_structural_type,
)
from .ic_lora_parameters import register_ic_lora_parameters_type
from .media_wire import (
    materialize_audio,
    materialize_image,
    materialize_latent,
    register_media_types,
)
from .pose_keypoint import register_pose_keypoint_type
from .sam3_wire import register_sam3_track_type
from .sampling_wire import register_sampling_type
from .splat_wire import materialize_splat
from .vision_wire import (
    clip_vision_output_fingerprint,
    decode_clip_vision_output,
    encode_clip_vision_output,
    materialize_clip_vision_output,
    validate_clip_vision_output,
)
from .voxel_wire import materialize_voxel, register_voxel_type
from .wan_wire import register_wan_type

MODEL_TYPE_ID = MODEL_TYPE
CLIP_TYPE_ID = CLIP_TYPE
CLIP_VISION_TYPE_ID = CLIP_VISION_TYPE
VAE_TYPE_ID = VAE_TYPE
SAMPLER_TYPE_ID = SAMPLER_TYPE
SIGMAS_TYPE_ID = SIGMAS_TYPE
GUIDER_TYPE_ID = GUIDER_TYPE
NOISE_TYPE_ID = NOISE_TYPE

_RESIDENT_TYPE_IDS = (
    AUDIO_ENCODER_TYPE,
    BACKGROUND_REMOVAL_TYPE,
    MODEL_TYPE_ID,
    MOGE_MODEL_TYPE,
    DA3_MODEL_TYPE,
    FACE_DETECTION_MODEL_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    MODEL_PATCH_TYPE,
    OPTICAL_FLOW_TYPE,
    INTERP_MODEL_TYPE,
    CLIP_TYPE_ID,
    CLIP_VISION_TYPE_ID,
    VAE_TYPE_ID,
    GUIDER_TYPE_ID,
    CONTROL_NET_TYPE,
    STYLE_MODEL_TYPE,
    GLIGEN_TYPE,
    PHOTOMAKER_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    UPSCALE_MODEL_TYPE,
    HOOKS_TYPE,
    HOOK_KEYFRAMES_TYPE,
)
_SAMPLING_TYPE_IDS = (SAMPLER_TYPE_ID, SIGMAS_TYPE_ID, NOISE_TYPE_ID)
_STRUCTURAL_TYPE_IDS = (TIMESTEPS_RANGE_TYPE, FLOATS_TYPE)


def register_inference_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Idempotently register shared wire and resident value types."""
    table = default_pool()
    registered = [
        registry.spec(CONDITIONING_TYPE_ID)
        if CONDITIONING_TYPE_ID in registry
        else register_conditioning_type(registry, resident_table=table)
    ]
    for type_id in _RESIDENT_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_resident_type(
                registry,
                type_id,
                table=table,
                coerce=(
                    normalize_hook_owners
                    if type_id == HOOKS_TYPE
                    else normalize_clip_hook_owners
                    if type_id == CLIP_TYPE
                    else None
                ),
            )
        )
    for type_id in _SAMPLING_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_sampling_type(registry, type_id)
        )
    registered.extend(register_box_types(registry))
    for type_id in (
        BOUNDING_BOXES_TYPE,
        CAMERA_INFO_STATE_TYPE,
        COLOR_TYPE,
        COLORS_TYPE,
        COMPOSITOR_TYPE,
        IMAGECOMPARE_TYPE,
        RESOLUTION_PREVIEW_TYPE,
    ):
        registered.append(
            registry.spec(type_id) if type_id in registry else registry.register(type_id)
        )
    for type_id in _STRUCTURAL_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_hook_structural_type(registry, type_id)
        )
    registered.append(
        registry.spec(IC_LORA_PARAMETERS_TYPE)
        if IC_LORA_PARAMETERS_TYPE in registry
        else register_ic_lora_parameters_type(registry)
    )
    if CLIP_VISION_OUTPUT_TYPE not in registry:
        registry.register(
            CLIP_VISION_OUTPUT_TYPE,
            encode=encode_clip_vision_output,
            decode=decode_clip_vision_output,
            fingerprint=clip_vision_output_fingerprint,
            validate_encoded=validate_clip_vision_output,
        )
    registered.append(registry.spec(CLIP_VISION_OUTPUT_TYPE))
    registered.append(
        registry.spec(SAM3_TRACK_DATA_TYPE)
        if SAM3_TRACK_DATA_TYPE in registry
        else register_sam3_track_type(registry)
    )
    registered.append(
        registry.spec(LOAD3D_CAMERA_TYPE)
        if LOAD3D_CAMERA_TYPE in registry
        else register_load3d_camera_type(registry)
    )
    registered.append(
        registry.spec(LOAD3D_MODEL_INFO_TYPE)
        if LOAD3D_MODEL_INFO_TYPE in registry
        else register_load3d_model_info_type(registry)
    )
    registered.append(
        registry.spec(POSE_KEYPOINT_TYPE)
        if POSE_KEYPOINT_TYPE in registry
        else register_pose_keypoint_type(registry)
    )
    if MOGE_GEOMETRY_TYPE not in registry:
        registry.register(
            MOGE_GEOMETRY_TYPE,
            encode=encode_latent,
            decode=partial(decode_latent, require_samples=False),
            fingerprint=latent_fingerprint(MOGE_GEOMETRY_TYPE),
            validate_encoded=partial(validate_latent_encoded, require_samples=False),
        )
    registered.append(registry.spec(MOGE_GEOMETRY_TYPE))
    if LAYERS_TYPE not in registry:
        registry.register(
            LAYERS_TYPE,
            encode=encode_latent,
            decode=partial(decode_latent, require_samples=False),
            fingerprint=latent_fingerprint(LAYERS_TYPE),
            validate_encoded=partial(validate_latent_encoded, require_samples=False),
        )
    registered.append(registry.spec(LAYERS_TYPE))
    for type_id in (WAN_CAMERA_EMBEDDING_TYPE, TRACKS_TYPE, AUDIO_ENCODER_OUTPUT_TYPE):
        registered.append(
            registry.spec(type_id) if type_id in registry else register_wan_type(registry, type_id)
        )
    registered.extend(register_media_types(registry))
    registered.append(register_mesh_type(registry))
    registered.append(register_splat_type(registry))
    registered.append(register_voxel_type(registry))
    register_video_edit_type(registry)
    registered.append(registry.spec(VIDEO_EDIT_TYPE))
    return tuple(registered)


__all__ = [
    "CLIP_TYPE_ID",
    "CLIP_VISION_TYPE_ID",
    "GUIDER_TYPE_ID",
    "MODEL_TYPE_ID",
    "NOISE_TYPE_ID",
    "SAMPLER_TYPE_ID",
    "SIGMAS_TYPE_ID",
    "VAE_TYPE_ID",
    "materialize_audio",
    "materialize_clip_vision_output",
    "materialize_image",
    "materialize_latent",
    "materialize_mesh",
    "materialize_splat",
    "materialize_voxel",
    "register_inference_types",
]
