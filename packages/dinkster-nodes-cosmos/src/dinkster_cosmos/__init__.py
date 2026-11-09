"""First-party cosmos model nodes."""

from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .cosmos_conditioning import CosmosImageToVideoLatent, CosmosPredict2ImageToVideoLatent
from .video_latent import EmptyCosmosLatentVideo

NODES = (
    CosmosImageToVideoLatent,
    CosmosPredict2ImageToVideoLatent,
    EmptyCosmosLatentVideo,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
