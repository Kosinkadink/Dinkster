from dinkster_api.v1 import (
    ASSET_TYPE,
    PHOTOMAKER_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_inference_types,
    register_resident_type,
    resolver_from_env,
)

from .attention_guidance import SkipLayerGuidanceSD3
from .checkpoint_save import ImageOnlyCheckpointSave
from .clip import unCLIPConditioning
from .clip_sd3 import CLIPTextEncodeSD3
from .clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
from .controlnet import ControlNetApplySD3
from .gligen_conditioning import GLIGENTextBoxApply
from .latent_model_specific import EmptySD3LatentImage
from .model_file_loaders import GLIGENLoader, ImageOnlyCheckpointLoader, unCLIPCheckpointLoader
from .photomaker import PhotoMakerEncode, PhotoMakerLoader
from .sd_upscale_conditioning import SD_4XUpscale_Conditioning
from .stable3d import (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)
from .stable_cascade import (
    StableCascade_EmptyLatentImage,
    StableCascade_StageB_Conditioning,
    StableCascade_StageC_VAEEncode,
    StableCascade_SuperResolutionControlnet,
)
from .video_conditioning import SVD_img2vid_Conditioning

NODES = (
    SkipLayerGuidanceSD3,
    ImageOnlyCheckpointSave,
    unCLIPConditioning,
    CLIPTextEncodeSD3,
    CLIPTextEncodeSDXLRefiner,
    CLIPTextEncodeSDXL,
    ControlNetApplySD3,
    GLIGENTextBoxApply,
    EmptySD3LatentImage,
    unCLIPCheckpointLoader,
    GLIGENLoader,
    ImageOnlyCheckpointLoader,
    PhotoMakerLoader,
    PhotoMakerEncode,
    SD_4XUpscale_Conditioning,
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
    StableCascade_EmptyLatentImage,
    StableCascade_StageC_VAEEncode,
    StableCascade_StageB_Conditioning,
    StableCascade_SuperResolutionControlnet,
    SVD_img2vid_Conditioning,
)

__all__ = ["NODES"]


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_inference_types(registry)
    return (
        registry.spec(PHOTOMAKER_TYPE)
        if PHOTOMAKER_TYPE in registry
        else register_resident_type(
            registry, PHOTOMAKER_TYPE, table=default_pool(), meta=comfy_resident_meta
        ),
    )


__all__.append("register_types")

inference_nodes(NODES)
