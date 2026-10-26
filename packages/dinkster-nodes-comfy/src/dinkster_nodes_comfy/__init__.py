"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_TYPE,
    TypeRegistry,
    register_asset_type,
    register_audio_value_type,
    resolver_from_env,
)
from dinkster_api.v1 import register_inference_types as _register_inference_types

from .ace_text_conditioning import ACE_TEXT_CONDITIONING_NODES
from .advanced_samplers import ADVANCED_SAMPLER_NODES
from .ar_video import ARVideoI2V, SamplerARVideo
from .attention_guidance import ATTENTION_GUIDANCE_NODES
from .attention_multiply import ATTENTION_MULTIPLY_NODES
from .audio_combine import AUDIO_COMBINE_NODES
from .audio_effects import AUDIO_EFFECT_NODES
from .audio_generators import AUDIO_GENERATOR_NODES
from .audio_latent import AUDIO_LATENT_NODES, GetAudioLatentSize
from .audio_music_conditioning import AUDIO_MUSIC_CONDITIONING_NODES
from .audio_transform import AUDIO_TRANSFORM_NODES
from .audio_vae import AUDIO_VAE_NODES
from .cache import CACHE_NODES
from .cfg_guidance import CFG_GUIDANCE_NODES
from .cfg_guiders import CFG_GUIDER_NODES
from .chroma_options import ChromaRadianceOptions
from .clip import CLIP_NODES
from .clip_model_specific import MODEL_TEXT_ENCODER_NODES
from .clip_sd3 import CLIPTextEncodeSD3
from .clip_sdxl import CLIP_SDXL_NODES
from .color_resolution import COLOR_RESOLUTION_NODES
from .compile import COMPILE_NODES
from .compositing import COMPOSITING_NODES
from .conditioning import CONDITIONING_NODES
from .context_windows import CONTEXT_WINDOW_NODES
from .controlnet import CONTROLNET_NODES
from .cosmos_conditioning import COSMOS_CONDITIONING_NODES
from .dataset_collections import DATASET_COLLECTION_NODES
from .dataset_groups import DATASET_GROUP_NODES
from .dataset_images import DATASET_IMAGE_NODES
from .dataset_text import DATASET_TEXT_NODES
from .device_routing import DEVICE_ROUTING_NODES
from .flux_conditioning import FLUX_CONDITIONING_NODES
from .flux_image_layout import FLUX_IMAGE_LAYOUT_NODES
from .freeu import FREEU_NODES
from .gligen_conditioning import GLIGENTextBoxApply
from .hook_checkpoint import HOOK_CHECKPOINT_NODES
from .hook_conditioning import HOOK_CONDITIONING_NODES
from .hook_lora import HOOK_LORA_NODES
from .hook_scheduling import HOOK_SCHEDULING_NODES
from .hunyuan3d import HUNYUAN3D_CONDITIONING_NODES
from .hunyuan_latent import HUNYUAN_LATENT_NODES
from .hunyuan_text import HUNYUAN_TEXT_NODES
from .hunyuan_video_image import HUNYUAN_VIDEO_IMAGE_NODES
from .hypernetwork import HypernetworkLoader
from .image_color_conversion import IMAGE_COLOR_CONVERSION_NODES
from .image_core import IMAGE_CORE_NODES
from .image_dimensions import IMAGE_DIMENSION_NODES
from .image_file_tools import (
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    PreviewImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from .image_filters import IMAGE_FILTER_NODES
from .image_transform import IMAGE_TRANSFORM_NODES
from .inpaint import INPAINT_NODES
from .kandinsky_video import Kandinsky5ImageToVideo
from .latent import LATENT_NODES
from .latent_advanced import LATENT_ADVANCED_NODES
from .latent_file_tools import LATENT_FILE_NODES
from .latent_model_specific import MODEL_LATENT_NODES
from .latent_operations import LATENT_OPERATION_NODES
from .latent_upscale import LATENT_UPSCALE_NODES
from .latent_upscale_loader import LATENT_UPSCALE_LOADER_NODES
from .logic import LOGIC_NODES
from .loops import LOOP_NODES
from .lora_extract_save import LoraSave
from .ltx_audio import LTX_AUDIO_NODES
from .ltx_av_latent import LTX_AV_LATENT_NODES
from .ltx_duration import LTXVDurationPredictor
from .ltx_sampling import LTX_SAMPLING_NODES
from .ltx_video_conditioning import LTX_VIDEO_CONDITIONING_NODES
from .ltx_video_preprocess import LTX_VIDEO_PREPROCESS_NODES
from .marigold import MarigoldV2PostProcess
from .mask import MASK_NODES
from .math import ComfyMathExpression
from .minimax_h3_latent import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide, MiniMaxH3ImageToVideo
from .minimax_h3_sampling import MiniMaxH3SigmaShift
from .mochi import EmptyMochiLatentVideo
from .model_advanced import MODEL_ADVANCED_NODES
from .model_file_savers import MODEL_FILE_SAVE_NODES
from .model_merging import MODEL_MERGE_NODES
from .model_merging_specific import MODEL_SPECIFIC_MERGE_NODES
from .model_patch_loader import MODEL_PATCH_LOADER_NODES
from .model_specific_schedulers import MODEL_SPECIFIC_SCHEDULER_NODES
from .morphology import MORPHOLOGY_NODES
from .noise_guidance import NOISE_GUIDANCE_NODES
from .perpneg import PERPNEG_NODES
from .pid_conditioning import PiDConditioning
from .post_processing import POST_PROCESSING_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .rebatch import REBATCH_NODES
from .reference_conditioning import REFERENCE_CONDITIONING_NODES
from .rope_and_mask import ROPE_AND_MASK_NODES
from .sampler_factories import SAMPLER_FACTORY_NODES
from .save_text import SaveText
from .sd_upscale_conditioning import SD_UPSCALE_CONDITIONING_NODES
from .seedvr_conditioning import SeedVR2Conditioning
from .seedvr_postprocess import SeedVR2PostProcessing
from .seedvr_preprocess import SeedVR2Preprocess
from .seedvr_temporal import SEEDVR_TEMPORAL_NODES
from .sensenova_options import SenseNovaSamplingOptions
from .sigma_operations import SIGMA_OPERATION_NODES
from .sigma_schedulers import SIGMA_SCHEDULER_NODES
from .sparse_attention import SPARSE_ATTENTION_NODES
from .spatial_model_patches import SPATIAL_MODEL_PATCH_NODES
from .stable3d import STABLE3D_NODES
from .stable_cascade import STABLE_CASCADE_NODES
from .style_model import STYLE_MODEL_NODES
from .switch import ComfySwitchNode
from .text import TEXT_NODES
from .utilities import ComfyNumberConvert, SeedNode
from .vae import VAE_NODES
from .video_conditioning import VIDEO_CONDITIONING_NODES
from .video_guidance import VIDEO_GUIDANCE_NODES
from .video_latent import VIDEO_LATENT_NODES
from .video_latent_ops import VIDEO_LATENT_OPERATION_NODES
from .wandancer_keyframes import WANDANCER_KEYFRAME_NODES

COMFY_NODES = (
    *TEXT_NODES,
    *DATASET_TEXT_NODES,
    SaveText,
    *DATASET_COLLECTION_NODES,
    PrimitiveString,
    PrimitiveStringMultiline,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
    *LOGIC_NODES,
    ComfySwitchNode,
    *LOOP_NODES,
    ComfyNumberConvert,
    SeedNode,
    ComfyMathExpression,
    *ATTENTION_MULTIPLY_NODES,
    *ATTENTION_GUIDANCE_NODES,
    *CACHE_NODES,
    *MODEL_MERGE_NODES,
    *MODEL_SPECIFIC_MERGE_NODES,
    *MODEL_ADVANCED_NODES,
    ChromaRadianceOptions,
    SenseNovaSamplingOptions,
    HypernetworkLoader,
    *MODEL_PATCH_LOADER_NODES,
    *DEVICE_ROUTING_NODES,
    *COMPILE_NODES,
    *MODEL_SPECIFIC_SCHEDULER_NODES,
    *CFG_GUIDER_NODES,
    *CFG_GUIDANCE_NODES,
    *PERPNEG_NODES,
    *VIDEO_GUIDANCE_NODES,
    *FREEU_NODES,
    *SPATIAL_MODEL_PATCH_NODES,
    *ROPE_AND_MASK_NODES,
    *NOISE_GUIDANCE_NODES,
    *SPARSE_ATTENTION_NODES,
    *HOOK_SCHEDULING_NODES,
    *SIGMA_OPERATION_NODES,
    *LATENT_NODES,
    *LATENT_ADVANCED_NODES,
    *LATENT_OPERATION_NODES,
    *LATENT_UPSCALE_NODES,
    *LATENT_UPSCALE_LOADER_NODES,
    *MODEL_LATENT_NODES,
    *CLIP_NODES,
    *STYLE_MODEL_NODES,
    GLIGENTextBoxApply,
    *CLIP_SDXL_NODES,
    CLIPTextEncodeSD3,
    *MODEL_TEXT_ENCODER_NODES,
    *FLUX_CONDITIONING_NODES,
    *FLUX_IMAGE_LAYOUT_NODES,
    *VAE_NODES,
    *CONDITIONING_NODES,
    PiDConditioning,
    *HOOK_CHECKPOINT_NODES,
    *HOOK_LORA_NODES,
    *CONTEXT_WINDOW_NODES,
    *HOOK_CONDITIONING_NODES,
    *REFERENCE_CONDITIONING_NODES,
    *SD_UPSCALE_CONDITIONING_NODES,
    *HUNYUAN3D_CONDITIONING_NODES,
    *INPAINT_NODES,
    *CONTROLNET_NODES,
    *MASK_NODES,
    *COMPOSITING_NODES,
    *IMAGE_CORE_NODES,
    *IMAGE_DIMENSION_NODES,
    *IMAGE_COLOR_CONVERSION_NODES,
    MarigoldV2PostProcess,
    *IMAGE_FILTER_NODES,
    *IMAGE_TRANSFORM_NODES,
    *MORPHOLOGY_NODES,
    *STABLE_CASCADE_NODES,
    *STABLE3D_NODES,
    *HUNYUAN_LATENT_NODES,
    *HUNYUAN_TEXT_NODES,
    *HUNYUAN_VIDEO_IMAGE_NODES,
    Kandinsky5ImageToVideo,
    *LTX_VIDEO_CONDITIONING_NODES,
    SeedVR2Preprocess,
    SeedVR2Conditioning,
    *SEEDVR_TEMPORAL_NODES,
    MiniMaxH3SigmaShift,
    EmptyMiniMaxH3LatentAV,
    MiniMaxH3ImageToVideo,
    *VIDEO_CONDITIONING_NODES,
    *COSMOS_CONDITIONING_NODES,
    *VIDEO_LATENT_NODES,
    *VIDEO_LATENT_OPERATION_NODES,
    *POST_PROCESSING_NODES,
    *REBATCH_NODES,
    EmptyMochiLatentVideo,
    *AUDIO_COMBINE_NODES,
    *AUDIO_EFFECT_NODES,
    *AUDIO_GENERATOR_NODES,
    *AUDIO_LATENT_NODES,
    *DATASET_GROUP_NODES,
    *ACE_TEXT_CONDITIONING_NODES,
    *AUDIO_MUSIC_CONDITIONING_NODES,
    *AUDIO_TRANSFORM_NODES,
    *AUDIO_VAE_NODES,
    GetAudioLatentSize,
    *LTX_AUDIO_NODES,
    *LTX_AV_LATENT_NODES,
    *LTX_VIDEO_PREPROCESS_NODES,
    *LTX_SAMPLING_NODES,
    *DATASET_IMAGE_NODES,
    *SIGMA_SCHEDULER_NODES,
    *SAMPLER_FACTORY_NODES,
    *ADVANCED_SAMPLER_NODES,
    SamplerARVideo,
    ARVideoI2V,
    SeedVR2PostProcessing,
    *WANDANCER_KEYFRAME_NODES,
    LTXVDurationPredictor,
    MiniMaxH3AddGuide,
    *COLOR_RESOLUTION_NODES,
    SaveImage,
    PreviewImage,
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    *LATENT_FILE_NODES,
    *MODEL_FILE_SAVE_NODES,
    SaveAnimatedWEBP,
    SaveAnimatedPNG,
    LoraSave,
)


def register_inference_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    _register_inference_types(registry)


__all__ = ["COMFY_NODES", "register_inference_types"]
