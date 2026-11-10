"""Audio input, editing, generation, preview and mounted output."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_RECORD_TYPE,
    AUDIO_TYPE,
    TypeRegistry,
    inference_nodes,
    register_asset_type,
    register_audio_value_type,
    resolver_from_env,
)

from .audio_combine import AUDIO_COMBINE_NODES
from .audio_effects import AUDIO_EFFECT_NODES
from .audio_generators import AUDIO_GENERATOR_NODES
from .audio_io import LoadAudio, PreviewAudio, RecordAudio
from .audio_save import SaveAudio, SaveAudioAdvanced, SaveAudioMP3, SaveAudioOpus
from .audio_transform import AUDIO_TRANSFORM_NODES

NODES = (
    *AUDIO_COMBINE_NODES,
    *AUDIO_EFFECT_NODES,
    *AUDIO_GENERATOR_NODES,
    *AUDIO_TRANSFORM_NODES,
    LoadAudio,
    PreviewAudio,
    RecordAudio,
    SaveAudio,
    SaveAudioAdvanced,
    SaveAudioMP3,
    SaveAudioOpus,
)


def register_types(registry: TypeRegistry) -> None:
    """Preserve the host's first media registration and granted recording codec."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if AUDIO_RECORD_TYPE not in registry:
        register_asset_type(registry, resolver_from_env(), type_id=AUDIO_RECORD_TYPE)
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
