"""First-party ltx model nodes."""

from dinkster_api.v1 import TypeRegistry, TypeSpec, inference_nodes, register_inference_types
from dinkster_inference_wire.ic_lora_parameters import register_ic_lora_parameters_type

from .context_windows import LTXVContextWindows
from .latent_upscale import LTXVLatentUpsampler
from .ltx_audio import LTXVAudioVAEDecode, LTXVAudioVAEEncode, LTXVEmptyLatentAudio
from .ltx_av_latent import LTXVConcatAVLatent, LTXVSeparateAVLatent
from .ltx_dual_cfg import LTXVDualCFGGuider
from .ltx_duration import LTXVDurationPredictor
from .ltx_guidance import LTXVModalityGuidance, LTXVReferenceAudio, LTXVSpatioTemporalGuidance
from .ltx_guides import GetICLoRAParameters, LTXVAddGuide, LTXVAddLatentGuide
from .ltx_keyframe_guides import (
    LTXVAddGeneratedKeyframes,
    LTXVGeneratedKeyframesToGuides,
    LTXVSeparateGeneratedKeyframes,
)
from .ltx_sampling import LTXVScheduler, ModelSamplingLTXV
from .ltx_video_conditioning import (
    LTXVConditioning,
    LTXVCropGuides,
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
)
from .ltx_video_preprocess import LTXVPreprocess
from .model_file_loaders import LTXAVTextEncoderLoader, LTXVAudioVAELoader
from .text_generation import TextGenerateLTX2Prompt
from .video_latent import EmptyLTXVLatentVideo
from .video_latent_ops import LTXVFreezeLatent


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    register_inference_types(registry)
    return tuple(
        registry.spec(type_id)
        if type_id in registry
        else register_ic_lora_parameters_type(registry)
        for type_id in ("IC_LORA_PARAMETERS",)
    )


NODES = (
    EmptyLTXVLatentVideo,
    GetICLoRAParameters,
    LTXAVTextEncoderLoader,
    LTXVAddGeneratedKeyframes,
    LTXVAddGuide,
    LTXVAddLatentGuide,
    LTXVAudioVAEDecode,
    LTXVAudioVAEEncode,
    LTXVAudioVAELoader,
    LTXVConcatAVLatent,
    LTXVConditioning,
    LTXVContextWindows,
    LTXVCropGuides,
    LTXVDualCFGGuider,
    LTXVDurationPredictor,
    LTXVEmptyLatentAudio,
    LTXVFreezeLatent,
    LTXVGeneratedKeyframesToGuides,
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
    LTXVLatentUpsampler,
    LTXVModalityGuidance,
    LTXVPreprocess,
    LTXVReferenceAudio,
    LTXVScheduler,
    LTXVSeparateAVLatent,
    LTXVSeparateGeneratedKeyframes,
    LTXVSpatioTemporalGuidance,
    ModelSamplingLTXV,
    TextGenerateLTX2Prompt,
)

inference_nodes(NODES)
