"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import register_inference_types

from .clip import CLIP_NODES
from .clip_model_specific import MODEL_TEXT_ENCODER_NODES
from .clip_sdxl import CLIP_SDXL_NODES
from .compositing import COMPOSITING_NODES
from .conditioning import CONDITIONING_NODES
from .dataset_text import DATASET_TEXT_NODES
from .freeu import FREEU_NODES
from .image_core import IMAGE_CORE_NODES
from .image_color_conversion import IMAGE_COLOR_CONVERSION_NODES
from .image_filters import IMAGE_FILTER_NODES
from .image_transform import IMAGE_TRANSFORM_NODES
from .inpaint import INPAINT_NODES
from .latent import LATENT_NODES
from .latent_advanced import LATENT_ADVANCED_NODES
from .latent_operations import LATENT_OPERATION_NODES
from .logic import LOGIC_NODES
from .mask import MASK_NODES
from .math import ComfyMathExpression
from .model_merging import MODEL_MERGE_NODES
from .model_merging_specific import MODEL_SPECIFIC_MERGE_NODES
from .morphology import MORPHOLOGY_NODES
from .post_processing import POST_PROCESSING_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .reference_conditioning import REFERENCE_CONDITIONING_NODES
from .stable_cascade import STABLE_CASCADE_NODES
from .text import TEXT_NODES
from .utilities import ComfyNumberConvert, SeedNode
from .vae import VAE_NODES

COMFY_NODES = (
    *TEXT_NODES,
    *DATASET_TEXT_NODES,
    PrimitiveString,
    PrimitiveStringMultiline,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
    *LOGIC_NODES,
    ComfyNumberConvert,
    SeedNode,
    ComfyMathExpression,
    *MODEL_MERGE_NODES,
    *MODEL_SPECIFIC_MERGE_NODES,
    *FREEU_NODES,
    *LATENT_NODES,
    *LATENT_ADVANCED_NODES,
    *LATENT_OPERATION_NODES,
    *CLIP_NODES,
    *CLIP_SDXL_NODES,
    *MODEL_TEXT_ENCODER_NODES,
    *VAE_NODES,
    *CONDITIONING_NODES,
    *REFERENCE_CONDITIONING_NODES,
    *INPAINT_NODES,
    *MASK_NODES,
    *COMPOSITING_NODES,
    *IMAGE_CORE_NODES,
    *IMAGE_COLOR_CONVERSION_NODES,
    *IMAGE_FILTER_NODES,
    *IMAGE_TRANSFORM_NODES,
    *MORPHOLOGY_NODES,
    *STABLE_CASCADE_NODES,
    *POST_PROCESSING_NODES,
)

__all__ = ["COMFY_NODES", "register_inference_types"]
