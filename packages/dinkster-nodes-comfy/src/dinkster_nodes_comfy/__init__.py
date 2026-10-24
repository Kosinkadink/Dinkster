"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import register_inference_types

from .ace_text_conditioning import ACE_TEXT_CONDITIONING_NODES
from .advanced_samplers import ADVANCED_SAMPLER_NODES
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
from .clip import CLIP_NODES
from .clip_model_specific import MODEL_TEXT_ENCODER_NODES
from .clip_sdxl import CLIP_SDXL_NODES
from .compile import COMPILE_NODES
from .compositing import COMPOSITING_NODES
from .conditioning import CONDITIONING_NODES
from .controlnet import CONTROLNET_NODES
from .cosmos_conditioning import COSMOS_CONDITIONING_NODES
from .dataset_collections import DATASET_COLLECTION_NODES
from .dataset_images import DATASET_IMAGE_NODES
from .dataset_text import DATASET_TEXT_NODES
from .device_routing import DEVICE_ROUTING_NODES
from .freeu import FREEU_NODES
from .hook_conditioning import HOOK_CONDITIONING_NODES
from .hunyuan_latent import HUNYUAN_LATENT_NODES
from .hunyuan_text import HUNYUAN_TEXT_NODES
from .hunyuan_video_image import HUNYUAN_VIDEO_IMAGE_NODES
from .image_core import IMAGE_CORE_NODES
from .image_filters import IMAGE_FILTER_NODES
from .image_transform import IMAGE_TRANSFORM_NODES
from .inpaint import INPAINT_NODES
from .latent import LATENT_NODES
from .latent_advanced import LATENT_ADVANCED_NODES
from .latent_model_specific import MODEL_LATENT_NODES
from .latent_operations import LATENT_OPERATION_NODES
from .logic import LOGIC_NODES
from .ltx_audio import LTX_AUDIO_NODES
from .ltx_video_conditioning import LTX_VIDEO_CONDITIONING_NODES
from .mask import MASK_NODES
from .math import ComfyMathExpression
from .mochi import EmptyMochiLatentVideo
from .model_advanced import MODEL_ADVANCED_NODES
from .model_merging import MODEL_MERGE_NODES
from .model_merging_specific import MODEL_SPECIFIC_MERGE_NODES
from .model_specific_schedulers import MODEL_SPECIFIC_SCHEDULER_NODES
from .morphology import MORPHOLOGY_NODES
from .noise_guidance import NOISE_GUIDANCE_NODES
from .perpneg import PERPNEG_NODES
from .post_processing import POST_PROCESSING_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .reference_conditioning import REFERENCE_CONDITIONING_NODES
from .rope_and_mask import ROPE_AND_MASK_NODES
from .sampler_factories import SAMPLER_FACTORY_NODES
from .save_text import SaveText
from .sigma_operations import SIGMA_OPERATION_NODES
from .sigma_schedulers import SIGMA_SCHEDULER_NODES
from .sparse_attention import SPARSE_ATTENTION_NODES
from .spatial_model_patches import SPATIAL_MODEL_PATCH_NODES
from .stable3d import STABLE3D_NODES
from .stable_cascade import STABLE_CASCADE_NODES
from .style_model import STYLE_MODEL_NODES
from .text import TEXT_NODES
from .utilities import ComfyNumberConvert, SeedNode
from .vae import VAE_NODES
from .video_conditioning import VIDEO_CONDITIONING_NODES
from .video_guidance import VIDEO_GUIDANCE_NODES
from .video_latent import VIDEO_LATENT_NODES
from .video_latent_ops import VIDEO_LATENT_OPERATION_NODES

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
    ComfyNumberConvert,
    SeedNode,
    ComfyMathExpression,
    *ATTENTION_MULTIPLY_NODES,
    *ATTENTION_GUIDANCE_NODES,
    *CACHE_NODES,
    *MODEL_MERGE_NODES,
    *MODEL_SPECIFIC_MERGE_NODES,
    *MODEL_ADVANCED_NODES,
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
    *SIGMA_OPERATION_NODES,
    *LATENT_NODES,
    *LATENT_ADVANCED_NODES,
    *LATENT_OPERATION_NODES,
    *MODEL_LATENT_NODES,
    *CLIP_NODES,
    *STYLE_MODEL_NODES,
    *CLIP_SDXL_NODES,
    *MODEL_TEXT_ENCODER_NODES,
    *VAE_NODES,
    *CONDITIONING_NODES,
    *HOOK_CONDITIONING_NODES,
    *REFERENCE_CONDITIONING_NODES,
    *INPAINT_NODES,
    *CONTROLNET_NODES,
    *MASK_NODES,
    *COMPOSITING_NODES,
    *IMAGE_CORE_NODES,
    *IMAGE_FILTER_NODES,
    *IMAGE_TRANSFORM_NODES,
    *MORPHOLOGY_NODES,
    *STABLE_CASCADE_NODES,
    *STABLE3D_NODES,
    *HUNYUAN_LATENT_NODES,
    *HUNYUAN_TEXT_NODES,
    *HUNYUAN_VIDEO_IMAGE_NODES,
    *LTX_VIDEO_CONDITIONING_NODES,
    *VIDEO_CONDITIONING_NODES,
    *COSMOS_CONDITIONING_NODES,
    *VIDEO_LATENT_NODES,
    *VIDEO_LATENT_OPERATION_NODES,
    *POST_PROCESSING_NODES,
    EmptyMochiLatentVideo,
    *AUDIO_COMBINE_NODES,
    *AUDIO_EFFECT_NODES,
    *AUDIO_GENERATOR_NODES,
    *AUDIO_LATENT_NODES,
    *ACE_TEXT_CONDITIONING_NODES,
    *AUDIO_MUSIC_CONDITIONING_NODES,
    *AUDIO_TRANSFORM_NODES,
    *AUDIO_VAE_NODES,
    GetAudioLatentSize,
    *LTX_AUDIO_NODES,
    *DATASET_IMAGE_NODES,
    *SIGMA_SCHEDULER_NODES,
    *SAMPLER_FACTORY_NODES,
    *ADVANCED_SAMPLER_NODES,
)

__all__ = ["COMFY_NODES", "register_inference_types"]
