from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .ace_text_conditioning import TextEncodeAceStepAudio, TextEncodeAceStepAudio15
from .audio_encoder import SheetSage2AudioToABC
from .audio_latent import EmptyAceStep15LatentAudio, EmptyAceStepLatentAudio, EmptyYuE2LatentAudio
from .conditioning import ConditioningStableAudio
from .yue2_conditioning import YuE2GenerateABC, YuE2GenerateMusic

NODES = (
    TextEncodeAceStepAudio,
    TextEncodeAceStepAudio15,
    SheetSage2AudioToABC,
    EmptyAceStepLatentAudio,
    EmptyAceStep15LatentAudio,
    EmptyYuE2LatentAudio,
    ConditioningStableAudio,
    YuE2GenerateABC,
    YuE2GenerateMusic,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
