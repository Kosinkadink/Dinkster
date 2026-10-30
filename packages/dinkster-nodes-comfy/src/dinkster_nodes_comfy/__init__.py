"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_RECORD_TYPE,
    AUDIO_TYPE,
    VIDEO_TYPE,
    Node,
    TypeRegistry,
    register_asset_type,
    register_audio_value_type,
    register_curve_type,
    register_file3d_types,
    register_load3d_type,
    register_video_value_type,
    resolver_from_env,
)
from dinkster_api.v1 import register_inference_types as _register_inference_types

from .ace_text_conditioning import ACE_TEXT_CONDITIONING_NODES
from .advanced_samplers import ADVANCED_SAMPLER_NODES
from .anima_lllite import AnimaLLLiteApply
from .ar_video import ARVideoI2V, SamplerARVideo
from .attention_guidance import ATTENTION_GUIDANCE_NODES
from .attention_multiply import ATTENTION_MULTIPLY_NODES
from .audio_combine import AUDIO_COMBINE_NODES
from .audio_effects import AUDIO_EFFECT_NODES
from .audio_encoder import AUDIO_ENCODER_NODES
from .audio_generators import AUDIO_GENERATOR_NODES
from .audio_io import LoadAudio, PreviewAudio, RecordAudio
from .audio_latent import AUDIO_LATENT_NODES, GetAudioLatentSize
from .audio_music_conditioning import AUDIO_MUSIC_CONDITIONING_NODES
from .audio_save import SaveAudio, SaveAudioAdvanced, SaveAudioMP3, SaveAudioOpus
from .audio_transform import AUDIO_TRANSFORM_NODES
from .audio_vae import AUDIO_VAE_NODES
from .background_removal import BACKGROUND_REMOVAL_NODES
from .bbox_crop import CropByBBoxes
from .bernini_conditioning import BERNINI_CONDITIONING_NODES
from .boogu_encoders import TextEncodeBooguEdit
from .cache import CACHE_NODES
from .camera_angle import CameraAngle
from .camera_info import CreateCameraInfo
from .cfg_guidance import CFG_GUIDANCE_NODES
from .cfg_guiders import CFG_GUIDER_NODES
from .checkpoint_save import ImageOnlyCheckpointSave
from .chroma_options import ChromaRadianceOptions
from .clip import CLIP_NODES
from .clip_loaders import QuadrupleCLIPLoader, TripleCLIPLoader
from .clip_model_specific import MODEL_TEXT_ENCODER_NODES
from .clip_sd3 import CLIPTextEncodeSD3
from .clip_sdxl import CLIP_SDXL_NODES
from .color_resolution import COLOR_RESOLUTION_NODES
from .compile import COMPILE_NODES
from .compositing import COMPOSITING_NODES
from .conditioning import CONDITIONING_NODES
from .conditioning_files import ConditioningLoader, SaveConditioning
from .context_windows import CONTEXT_WINDOW_NODES
from .controlnet import CONTROLNET_NODES
from .cosmos_conditioning import COSMOS_CONDITIONING_NODES
from .curve_editor import CurveEditor
from .custom_sampling import CUSTOM_SAMPLING_NODES
from .device_routing import DEVICE_ROUTING_NODES
from .fibo2_encoders import TextEncodeFibo2
from .flux_conditioning import FLUX_CONDITIONING_NODES
from .flux_image_layout import FLUX_IMAGE_LAYOUT_NODES
from .freeu import FREEU_NODES
from .gaussian_splat import GetSplatCount, MergeSplat, TransformSplat
from .gligen_conditioning import GLIGENTextBoxApply
from .hidream_patches import HiDreamO1PatchSeamSmoothing, HiDreamO1ReferenceImages
from .hook_checkpoint import HOOK_CHECKPOINT_NODES
from .hook_conditioning import HOOK_CONDITIONING_NODES
from .hook_lora import HOOK_LORA_NODES
from .hook_scheduling import HOOK_SCHEDULING_NODES
from .hunyuan3d import HUNYUAN3D_CONDITIONING_NODES, VAEDecodeHunyuan3D
from .hunyuan_latent import HUNYUAN_LATENT_NODES
from .hunyuan_text import HUNYUAN_TEXT_NODES
from .hunyuan_video_image import HUNYUAN_VIDEO_IMAGE_NODES
from .hypernetwork import HypernetworkLoader
from .image_color_conversion import IMAGE_COLOR_CONVERSION_NODES
from .image_control_patches import IMAGE_CONTROL_PATCH_NODES
from .image_core import IMAGE_CORE_NODES
from .image_crop import ImageCropToMask, ImageCropV2, PrimitiveBoundingBox
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
from .image_layers import AddLayer
from .image_tiles import ImageMergeTileList, SplitImageToTileList
from .image_transform import IMAGE_TRANSFORM_NODES
from .inpaint import INPAINT_NODES
from .interpolation import FrameInterpolate
from .interpolation_loader import FrameInterpolationModelLoader
from .joyimage_encoders import TextEncodeJoyImageEdit
from .json_prompt import BuildJsonPromptIdeogram
from .kandinsky_video import Kandinsky5ImageToVideo
from .latent import LATENT_NODES
from .latent_advanced import LATENT_ADVANCED_NODES
from .latent_file_tools import LATENT_FILE_NODES
from .latent_model_specific import MODEL_LATENT_NODES
from .latent_operations import LATENT_OPERATION_NODES
from .latent_upscale import LATENT_UPSCALE_NODES
from .latent_upscale_loader import LATENT_UPSCALE_LOADER_NODES
from .list_utilities import LIST_UTILITY_NODES
from .logic import LOGIC_NODES
from .loops import LOOP_NODES
from .lora_extract_save import LoraSave
from .lora_file_loaders import LORA_FILE_LOADER_NODES
from .lotus import LotusConditioning
from .ltx_audio import LTX_AUDIO_NODES
from .ltx_av_latent import LTX_AV_LATENT_NODES
from .ltx_dual_cfg import LTXVDualCFGGuider
from .ltx_duration import LTXVDurationPredictor
from .ltx_guidance import LTXVModalityGuidance, LTXVReferenceAudio, LTXVSpatioTemporalGuidance
from .ltx_guides import GetICLoRAParameters, LTXVAddGuide, LTXVAddLatentGuide
from .ltx_keyframe_guides import (
    LTXVAddGeneratedKeyframes,
    LTXVGeneratedKeyframesToGuides,
    LTXVSeparateGeneratedKeyframes,
)
from .ltx_sampling import LTX_SAMPLING_NODES
from .ltx_video_conditioning import LTX_VIDEO_CONDITIONING_NODES
from .ltx_video_preprocess import LTX_VIDEO_PREPROCESS_NODES
from .mage_image_encoders import TextEncodeMageFlowEdit
from .marigold import MarigoldV2PostProcess
from .mask import MASK_NODES
from .mask_preview import MaskPreview
from .math import ComfyMathExpression
from .mesh import MESH_NODES
from .ming_image_encoders import TextEncodeMingImageEdit
from .minimax_h3_control import MiniMaxH3FunControlNetApply
from .minimax_h3_latent import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide, MiniMaxH3ImageToVideo
from .minimax_h3_reference import MiniMaxH3ReferenceToVideo
from .minimax_h3_sampling import MiniMaxH3SigmaShift
from .mochi import EmptyMochiLatentVideo
from .model_advanced import MODEL_ADVANCED_NODES
from .model_file_loaders import MODEL_FILE_LOADER_NODES
from .model_file_savers import MODEL_FILE_SAVE_NODES, CheckpointSave, ModelSave
from .model_merging import MODEL_MERGE_NODES
from .model_merging_specific import MODEL_SPECIFIC_MERGE_NODES
from .model_patch_loader import MODEL_PATCH_LOADER_NODES
from .model_selections import MODEL_SELECTION_NODES
from .model_specific_schedulers import MODEL_SPECIFIC_SCHEDULER_NODES
from .moge import MoGeGeometryToFOV
from .moge_inference import MoGeInference
from .moge_mesh import MoGePointMapToMesh
from .moge_panorama import MoGePanoramaInference
from .morphology import MORPHOLOGY_NODES
from .moss_image import MOSSImageControl, MOSSImageLayoutPrompt, TextEncodeMOSSImageEdit
from .noise import NOISE_NODES
from .noise_guidance import NOISE_GUIDANCE_NODES
from .optical_flow import OpticalFlowLoader
from .perpneg import PERPNEG_NODES
from .photomaker import PhotoMakerEncode, PhotoMakerLoader
from .pid_conditioning import PiDConditioning
from .post_processing import POST_PROCESSING_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .qwen_image_encoders import (
    QwenImage21Cache,
    TextEncodeQwenImage21,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
)
from .rebatch import REBATCH_NODES
from .reference_conditioning import REFERENCE_CONDITIONING_NODES
from .rope_and_mask import ROPE_AND_MASK_NODES
from .rtdetr import RTDETR_NODES
from .sam3 import SAM3_TRACKING_NODES
from .sam3_detection import SAM3_Detect
from .sam3_preview import SAM3_TrackPreview
from .sampler_factories import SAMPLER_FACTORY_NODES
from .save_image_advanced import SaveImageAdvanced
from .save_text import SaveText
from .scail_masks import SCAIL2ColoredMask
from .sd_upscale_conditioning import SD_UPSCALE_CONDITIONING_NODES
from .sdpose import SDPoseDrawKeypoints, SDPoseFaceBBoxes
from .sdpose_extract import SDPoseKeypointExtractor
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
from .standard_sampling import STANDARD_SAMPLING_NODES
from .style_model import STYLE_MODEL_NODES
from .supir import SUPIRApply
from .svg_save import SaveSVGNode
from .switch import ComfySwitchNode
from .text import TEXT_NODES
from .text_generation import TextGenerate, TextGenerateLTX2Prompt
from .trellis_conditioning import Trellis2Conditioning
from .trellis_shape import Trellis2ShapeStage
from .trellis_upsample import Trellis2UpsampleStage
from .triposplat import (
    TripoSplatConditioning,
    TripoSplatPreprocessImage,
    TripoSplatSamplingPreview,
    VAEDecodeTripoSplat,
)
from .upscale_model_loader import UpscaleModelLoader
from .utilities import ComfyNumberConvert, PreviewAny, SeedNode
from .vae import VAE_NODES
from .video_components import ConcatenateVideo, CreateVideo, GetVideoComponents
from .video_conditioning import VIDEO_CONDITIONING_NODES
from .video_edits import VideoCrop, VideoSlice, VideoTrim
from .video_guidance import VIDEO_GUIDANCE_NODES
from .video_latent import VIDEO_LATENT_NODES
from .video_latent_ops import VIDEO_LATENT_OPERATION_NODES
from .video_load import LoadVideo
from .video_save import SaveVideo
from .void_conditioning import (
    VOIDInpaintConditioning,
    VOIDQuadmaskPreprocess,
    VOIDWarpedNoiseSource,
)
from .void_sampler import VOIDSampler
from .void_warped_noise import VOIDWarpedNoise
from .voxel_mesh import VOXEL_MESH_NODES
from .wan22_latent import Wan22ImageToVideoLatent
from .wan_animate import WAN_ANIMATE_NODES
from .wan_camera import WanCameraEmbedding
from .wan_camera_conditioning import WanCameraImageToVideo
from .wan_control_conditioning import WAN_CONTROL_CONDITIONING_NODES
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
from .wandancer_keyframes import WANDANCER_KEYFRAME_NODES
from .wandancer_video import WanDancerVideo
from .webm_save import SaveWEBM
from .yue2_conditioning import YUE2_CONDITIONING_NODES
from .zimage_encoders import TextEncodeZImageOmni

COMFY_NODES: tuple[type[Node], ...] = (
    *AUDIO_ENCODER_NODES,
    *TEXT_NODES,
    SaveText,
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
    HiDreamO1PatchSeamSmoothing,
    HiDreamO1ReferenceImages,
    TripleCLIPLoader,
    QuadrupleCLIPLoader,
    LotusConditioning,
    *MODEL_MERGE_NODES,
    *MODEL_SPECIFIC_MERGE_NODES,
    *MODEL_ADVANCED_NODES,
    ChromaRadianceOptions,
    SenseNovaSamplingOptions,
    HypernetworkLoader,
    *MODEL_PATCH_LOADER_NODES,
    AnimaLLLiteApply,
    SUPIRApply,
    *STANDARD_SAMPLING_NODES,
    *CUSTOM_SAMPLING_NODES,
    WanUni3CControlnetApply,
    *IMAGE_CONTROL_PATCH_NODES,
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
    *NOISE_NODES,
    *SPARSE_ATTENTION_NODES,
    *HOOK_SCHEDULING_NODES,
    *SIGMA_OPERATION_NODES,
    *LATENT_NODES,
    *LATENT_ADVANCED_NODES,
    *LATENT_OPERATION_NODES,
    *LATENT_UPSCALE_NODES,
    *LATENT_UPSCALE_LOADER_NODES,
    OpticalFlowLoader,
    FrameInterpolationModelLoader,
    FrameInterpolate,
    *MODEL_LATENT_NODES,
    *CLIP_NODES,
    *STYLE_MODEL_NODES,
    GLIGENTextBoxApply,
    *CLIP_SDXL_NODES,
    CLIPTextEncodeSD3,
    *MODEL_TEXT_ENCODER_NODES,
    *FLUX_CONDITIONING_NODES,
    *FLUX_IMAGE_LAYOUT_NODES,
    *BERNINI_CONDITIONING_NODES,
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
    *ACE_TEXT_CONDITIONING_NODES,
    *AUDIO_MUSIC_CONDITIONING_NODES,
    *AUDIO_TRANSFORM_NODES,
    *AUDIO_VAE_NODES,
    GetAudioLatentSize,
    *LTX_AUDIO_NODES,
    *LTX_AV_LATENT_NODES,
    *LTX_VIDEO_PREPROCESS_NODES,
    *LTX_SAMPLING_NODES,
    *SIGMA_SCHEDULER_NODES,
    *SAMPLER_FACTORY_NODES,
    *ADVANCED_SAMPLER_NODES,
    SamplerARVideo,
    ARVideoI2V,
    SeedVR2PostProcessing,
    *WANDANCER_KEYFRAME_NODES,
    LTXVDurationPredictor,
    GetICLoRAParameters,
    LTXVAddGuide,
    LTXVAddLatentGuide,
    LTXVAddGeneratedKeyframes,
    LTXVSeparateGeneratedKeyframes,
    LTXVGeneratedKeyframesToGuides,
    LTXVReferenceAudio,
    LTXVSpatioTemporalGuidance,
    LTXVModalityGuidance,
    LTXVDualCFGGuider,
    MiniMaxH3AddGuide,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3FunControlNetApply,
    *LIST_UTILITY_NODES,
    SplitImageToTileList,
    ImageMergeTileList,
    MaskPreview,
    PreviewAny,
    WanImageToVideo,
    WanFirstLastFrameToVideo,
    WanFunInpaintToVideo,
    *WAN_CONTROL_CONDITIONING_NODES,
    Wan22ImageToVideoLatent,
    WanSCAILToVideo,
    WanTrackToVideo,
    *WAN_ANIMATE_NODES,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanMoveTracksFromCoords,
    GenerateTracks,
    WanMoveConcatTrack,
    WanMoveTrackToVideo,
    WanMoveVisualizeTracks,
    WanSoundImageToVideo,
    WanSoundImageToVideoExtend,
    WanHuMoImageToVideo,
    WanInfiniteTalkToVideo,
    SCAIL2ColoredMask,
    *COLOR_RESOLUTION_NODES,
    BuildJsonPromptIdeogram,
    MOSSImageLayoutPrompt,
    TextEncodeMOSSImageEdit,
    TextGenerate,
    MOSSImageControl,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
    TextEncodeQwenImage21,
    QwenImage21Cache,
    TextEncodeMingImageEdit,
    TextEncodeMageFlowEdit,
    TextEncodeJoyImageEdit,
    TextEncodeBooguEdit,
    TextEncodeFibo2,
    TextGenerateLTX2Prompt,
    TextEncodeZImageOmni,
    PhotoMakerLoader,
    PhotoMakerEncode,
    ImageCropToMask,
    ImageCropV2,
    PrimitiveBoundingBox,
    *RTDETR_NODES,
    *MODEL_SELECTION_NODES,
    *SAM3_TRACKING_NODES,
    SAM3_Detect,
    SAM3_TrackPreview,
    VideoSlice,
    VideoTrim,
    VideoCrop,
    SaveVideo,
    *MODEL_FILE_LOADER_NODES,
    *LORA_FILE_LOADER_NODES,
    *BACKGROUND_REMOVAL_NODES,
    CropByBBoxes,
    CameraAngle,
    CreateCameraInfo,
    GetSplatCount,
    TransformSplat,
    MergeSplat,
    Trellis2UpsampleStage,
    MoGePanoramaInference,
    MoGeInference,
    MoGePointMapToMesh,
    TripoSplatPreprocessImage,
    TripoSplatSamplingPreview,
    TripoSplatConditioning,
    VAEDecodeTripoSplat,
    LoadAudio,
    RecordAudio,
    PreviewAudio,
    SaveAudio,
    SaveAudioMP3,
    SaveAudioOpus,
    SaveAudioAdvanced,
    SaveWEBM,
    WanDancerEncodeAudio,
    *YUE2_CONDITIONING_NODES,
    MoGeGeometryToFOV,
    UpscaleModelLoader,
    ImageOnlyCheckpointSave,
    VOIDSampler,
    VOIDQuadmaskPreprocess,
    VOIDInpaintConditioning,
    VOIDWarpedNoiseSource,
    CurveEditor,
    CreateVideo,
    ConcatenateVideo,
    GetVideoComponents,
    ConditioningLoader,
    SaveConditioning,
    SDPoseFaceBBoxes,
    SDPoseDrawKeypoints,
    SDPoseKeypointExtractor,
    AddLayer,
    WanDancerVideo,
    VOIDWarpedNoise,
    LoadVideo,
    SaveSVGNode,
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
    CheckpointSave,
    ModelSave,
    SaveImageAdvanced,
    Trellis2Conditioning,
    *MESH_NODES,
    *VOXEL_MESH_NODES,
    VAEDecodeHunyuan3D,
    Trellis2ShapeStage,
)


def register_inference_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_load3d_type(registry)
    if AUDIO_RECORD_TYPE not in registry:
        register_asset_type(registry, resolver_from_env(), type_id=AUDIO_RECORD_TYPE)
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    if VIDEO_TYPE not in registry:
        register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_curve_type(registry)
    register_file3d_types(registry, resolver_from_env())
    _register_inference_types(registry)


__all__ = ["COMFY_NODES", "register_inference_types"]
