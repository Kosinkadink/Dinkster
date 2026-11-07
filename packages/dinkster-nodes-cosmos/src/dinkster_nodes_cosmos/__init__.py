"""First-party cosmos model nodes."""

from .cosmos_conditioning import CosmosImageToVideoLatent, CosmosPredict2ImageToVideoLatent
from .video_latent import EmptyCosmosLatentVideo

NODES = (
    CosmosImageToVideoLatent,
    CosmosPredict2ImageToVideoLatent,
    EmptyCosmosLatentVideo,
)
