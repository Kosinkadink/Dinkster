"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import register_inference_types

from .conditioning import CONDITIONING_NODES
from .image_core import IMAGE_CORE_NODES
from .latent import LATENT_NODES
from .logic import LOGIC_NODES
from .mask import MASK_NODES
from .math import ComfyMathExpression
from .model_merging import MODEL_MERGE_NODES
from .model_merging_specific import MODEL_SPECIFIC_MERGE_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .text import TEXT_NODES
from .utilities import ComfyNumberConvert, SeedNode

COMFY_NODES = (
    *TEXT_NODES,
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
    *LATENT_NODES,
    *CONDITIONING_NODES,
    *MASK_NODES,
    *IMAGE_CORE_NODES,
)

__all__ = ["COMFY_NODES", "register_inference_types"]
