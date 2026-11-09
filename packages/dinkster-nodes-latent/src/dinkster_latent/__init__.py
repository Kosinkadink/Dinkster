"""First-party latent nodes."""

from dinkster_api.v1 import inference_nodes

from .audio_latent import (
    EmptyLatentAudio,
    GetAudioLatentSize,
)
from .audio_vae import (
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    VAEEncodeAudio,
)
from .image_dimensions import (
    GetLatentSize,
)
from .inpaint import (
    VAEEncodeForInpaint,
)
from .latent import (
    EmptyLatentImage,
    LatentBlend,
    LatentComposite,
    LatentCrop,
    LatentFlip,
    LatentFromBatch,
    LatentRotate,
    LatentUpscale,
    LatentUpscaleBy,
    RepeatLatentBatch,
    SetLatentNoiseMask,
)
from .latent_advanced import (
    LatentAdd,
    LatentBatch,
    LatentBatchSeedBehavior,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentInterpolate,
    LatentMultiply,
    LatentSubtract,
)
from .latent_file_tools import (
    LoadLatent,
    SaveLatent,
)
from .latent_operations import (
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationBlend,
    LatentOperationSharpen,
    LatentOperationTonemapReinhard,
    ReplaceVideoLatentFrames,
)
from .mask import (
    LatentCompositeMasked,
)
from .post_processing import (
    BatchLatentsNode,
)
from .rebatch import (
    RebatchImages,
    RebatchLatents,
)
from .vae import (
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeTiled,
)
from .video_latent_ops import (
    NormalizeVideoLatentStart,
    TrimVideoLatent,
)

NODES = (
    EmptyLatentImage,
    LatentFromBatch,
    RepeatLatentBatch,
    LatentUpscale,
    LatentUpscaleBy,
    LatentRotate,
    LatentFlip,
    SetLatentNoiseMask,
    LatentComposite,
    LatentBlend,
    LatentCrop,
    LatentAdd,
    LatentSubtract,
    LatentMultiply,
    LatentInterpolate,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentBatch,
    LatentBatchSeedBehavior,
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationBlend,
    LatentOperationTonemapReinhard,
    LatentOperationSharpen,
    ReplaceVideoLatentFrames,
    VAEDecode,
    VAEEncode,
    VAEDecodeTiled,
    VAEEncodeTiled,
    VAEEncodeForInpaint,
    LatentCompositeMasked,
    GetLatentSize,
    TrimVideoLatent,
    NormalizeVideoLatentStart,
    BatchLatentsNode,
    RebatchImages,
    RebatchLatents,
    EmptyLatentAudio,
    VAEEncodeAudio,
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    GetAudioLatentSize,
    LoadLatent,
    SaveLatent,
)

inference_nodes(NODES)
