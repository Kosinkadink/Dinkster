from .clip_model_specific import CLIPTextEncodeHiDream
from .hidream_patches import HiDreamO1PatchSeamSmoothing, HiDreamO1ReferenceImages
from .latent_model_specific import EmptyHiDreamO1LatentImage

NODES = (
    CLIPTextEncodeHiDream,
    HiDreamO1ReferenceImages,
    HiDreamO1PatchSeamSmoothing,
    EmptyHiDreamO1LatentImage,
)

__all__ = ["NODES"]
