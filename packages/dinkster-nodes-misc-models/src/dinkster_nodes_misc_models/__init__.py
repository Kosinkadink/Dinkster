from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .anima_lllite import AnimaLLLiteApply
from .ar_video import ARVideoI2V, SamplerARVideo
from .bernini_conditioning import BerniniConditioning
from .boogu_encoders import TextEncodeBooguEdit
from .chroma_options import ChromaRadianceOptions
from .clip_model_specific import (
    CLIPTextEncodeKandinsky5,
    CLIPTextEncodeLumina2,
    CLIPTextEncodePixArtAlpha,
)
from .fibo2_encoders import TextEncodeFibo2
from .image_control_patches import ZImageFunControlnet
from .joyimage_encoders import TextEncodeJoyImageEdit
from .json_prompt import BuildJsonPromptIdeogram
from .kandinsky_video import Kandinsky5ImageToVideo
from .latent_model_specific import EmptyChromaRadianceLatentImage
from .lotus import LotusConditioning
from .mage_image_encoders import TextEncodeMageFlowEdit
from .ming_image_encoders import TextEncodeMingImageEdit
from .mochi import EmptyMochiLatentVideo
from .model_specific_schedulers import Ideogram4Scheduler
from .moss_image import MOSSImageControl, MOSSImageLayoutPrompt, TextEncodeMOSSImageEdit
from .reference_conditioning import InstructPixToPixConditioning
from .sensenova_options import SenseNovaSamplingOptions
from .supir import SUPIRApply
from .video_latent import EmptyARVideoLatent
from .zimage_encoders import TextEncodeZImageOmni

NODES = (
    AnimaLLLiteApply,
    SamplerARVideo,
    ARVideoI2V,
    BerniniConditioning,
    TextEncodeBooguEdit,
    ChromaRadianceOptions,
    CLIPTextEncodePixArtAlpha,
    CLIPTextEncodeLumina2,
    CLIPTextEncodeKandinsky5,
    TextEncodeFibo2,
    ZImageFunControlnet,
    TextEncodeJoyImageEdit,
    BuildJsonPromptIdeogram,
    Kandinsky5ImageToVideo,
    EmptyChromaRadianceLatentImage,
    LotusConditioning,
    TextEncodeMageFlowEdit,
    TextEncodeMingImageEdit,
    EmptyMochiLatentVideo,
    Ideogram4Scheduler,
    MOSSImageLayoutPrompt,
    MOSSImageControl,
    TextEncodeMOSSImageEdit,
    InstructPixToPixConditioning,
    SenseNovaSamplingOptions,
    SUPIRApply,
    EmptyARVideoLatent,
    TextEncodeZImageOmni,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
