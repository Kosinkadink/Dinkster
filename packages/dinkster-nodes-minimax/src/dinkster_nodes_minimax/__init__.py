"""First-party minimax model nodes."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_audio_value_type,
    register_media_types,
    register_resident_type,
    resolver_from_env,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.hook_values import normalize_clip_hook_owners

from .audio_generators import EmptyMiniMaxMusic3LatentAudio
from .audio_music_conditioning import MiniMaxMusic3TextEncode, ReferenceTimbreAudio
from .minimax_h3_control import MiniMaxH3FunControlNetApply
from .minimax_h3_latent import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide, MiniMaxH3ImageToVideo
from .minimax_h3_reference import MiniMaxH3ReferenceToVideo
from .minimax_h3_sampling import MiniMaxH3SigmaShift

NODES = (
    EmptyMiniMaxH3LatentAV,
    EmptyMiniMaxMusic3LatentAudio,
    MiniMaxH3AddGuide,
    MiniMaxH3FunControlNetApply,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3SigmaShift,
    MiniMaxMusic3TextEncode,
    ReferenceTimbreAudio,
)


def register_types(registry: TypeRegistry) -> None:
    """Install consumed canonical codecs in this pack's isolated worker."""
    register_media_types(registry)
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    if CONDITIONING_TYPE not in registry:
        register_conditioning_type(registry, resident_table=default_pool())
    for type_id in (CLIP_TYPE, MODEL_TYPE, MODEL_PATCH_TYPE, VAE_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry,
                type_id,
                table=default_pool(),
                meta=comfy_resident_meta,
                coerce=normalize_clip_hook_owners if type_id == CLIP_TYPE else None,
            )


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
