"""First-party minimax model nodes."""

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
