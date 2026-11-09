from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .image_control_patches import QwenImageDiffsynthControlnet
from .latent_model_specific import EmptyQwenImageLayeredLatentImage
from .qwen_image_encoders import (
    QwenImage21Cache,
    TextEncodeQwenImage21,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
)

NODES = (
    QwenImageDiffsynthControlnet,
    EmptyQwenImageLayeredLatentImage,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
    TextEncodeQwenImage21,
    QwenImage21Cache,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
