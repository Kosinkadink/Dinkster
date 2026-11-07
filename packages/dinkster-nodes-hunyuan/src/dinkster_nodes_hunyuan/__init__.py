"""First-party hunyuan model nodes."""

from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .hunyuan3d import Hunyuan3Dv2Conditioning, Hunyuan3Dv2ConditioningMultiView, VAEDecodeHunyuan3D
from .hunyuan_latent import EmptyHunyuanImageLatent, HunyuanRefinerLatent
from .hunyuan_text import CLIPTextEncodeHunyuanDiT, TextEncodeHunyuanVideo_ImageToVideo
from .hunyuan_video_image import (
    HunyuanImageToVideo,
    HunyuanVideo15ImageToVideo,
    HunyuanVideo15SuperResolution,
)
from .latent_model_specific import EmptyLatentHunyuan3Dv2
from .latent_upscale import HunyuanVideo15LatentUpscaleWithModel
from .video_latent import EmptyHunyuanLatentVideo, EmptyHunyuanVideo15Latent

NODES = (
    CLIPTextEncodeHunyuanDiT,
    EmptyHunyuanImageLatent,
    EmptyHunyuanLatentVideo,
    EmptyHunyuanVideo15Latent,
    EmptyLatentHunyuan3Dv2,
    Hunyuan3Dv2Conditioning,
    Hunyuan3Dv2ConditioningMultiView,
    HunyuanImageToVideo,
    HunyuanRefinerLatent,
    HunyuanVideo15ImageToVideo,
    HunyuanVideo15LatentUpscaleWithModel,
    HunyuanVideo15SuperResolution,
    TextEncodeHunyuanVideo_ImageToVideo,
    VAEDecodeHunyuan3D,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
