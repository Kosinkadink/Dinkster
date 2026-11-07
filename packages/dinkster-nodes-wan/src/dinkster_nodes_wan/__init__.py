"""First-party wan model nodes."""

from dinkster_api.v1 import TypeRegistry, TypeSpec
from dinkster_inference_wire.wan_wire import register_wan_type

from .compile import wanBlockSwap
from .context_windows import WanContextWindowsManual
from .scail_masks import SCAIL2ColoredMask
from .wan22_latent import Wan22ImageToVideoLatent
from .wan_animate import WanAnimate2Cache, WanAnimate2ToVideo, WanAnimateToVideo
from .wan_camera import WanCameraEmbedding
from .wan_camera_conditioning import WanCameraImageToVideo
from .wan_control_conditioning import (
    Wan22FunControlToVideo,
    WanFunControlToVideo,
    WanPhantomSubjectToVideo,
    WanVaceToVideo,
)
from .wan_image_conditioning import WanFirstLastFrameToVideo, WanFunInpaintToVideo, WanImageToVideo
from .wan_infinite_talk import WanInfiniteTalkToVideo
from .wan_move_conditioning import WanMoveTrackToVideo
from .wan_move_tracks import GenerateTracks, WanMoveConcatTrack, WanMoveTracksFromCoords
from .wan_move_visualize import WanMoveVisualizeTracks
from .wan_scail_conditioning import WanSCAILToVideo
from .wan_sound import WanHuMoImageToVideo, WanSoundImageToVideo, WanSoundImageToVideoExtend
from .wan_track_conditioning import WanTrackToVideo
from .wan_uni3c import WanUni3CControlnetApply
from .wandancer_audio import WanDancerEncodeAudio
from .wandancer_keyframes import WanDancerPadKeyframes, WanDancerPadKeyframesList
from .wandancer_video import WanDancerVideo


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    return tuple(
        registry.spec(type_id) if type_id in registry else register_wan_type(registry, type_id)
        for type_id in ("WAN_CAMERA_EMBEDDING", "TRACKS")
    )


NODES = (
    GenerateTracks,
    SCAIL2ColoredMask,
    Wan22FunControlToVideo,
    Wan22ImageToVideoLatent,
    WanAnimate2Cache,
    WanAnimate2ToVideo,
    WanAnimateToVideo,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanContextWindowsManual,
    WanDancerEncodeAudio,
    WanDancerPadKeyframes,
    WanDancerPadKeyframesList,
    WanDancerVideo,
    WanFirstLastFrameToVideo,
    WanFunControlToVideo,
    WanFunInpaintToVideo,
    WanHuMoImageToVideo,
    WanImageToVideo,
    WanInfiniteTalkToVideo,
    WanMoveConcatTrack,
    WanMoveTrackToVideo,
    WanMoveTracksFromCoords,
    WanMoveVisualizeTracks,
    WanPhantomSubjectToVideo,
    WanSCAILToVideo,
    WanSoundImageToVideo,
    WanSoundImageToVideoExtend,
    WanTrackToVideo,
    WanUni3CControlnetApply,
    WanVaceToVideo,
    wanBlockSwap,
)
