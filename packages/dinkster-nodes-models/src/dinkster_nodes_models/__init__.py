"""First-party models nodes."""

from dinkster_api.v1 import inference_nodes

from .attention_multiply import (
    CLIPAttentionMultiply,
    UNetCrossAttentionMultiply,
    UNetSelfAttentionMultiply,
    UNetTemporalAttentionMultiply,
)
from .cache import (
    EasyCache,
    LazyCache,
)
from .clip_loaders import (
    QuadrupleCLIPLoader,
    TripleCLIPLoader,
)
from .compile import (
    TorchCompileModel,
)
from .context_windows import (
    ContextWindowsManual,
)
from .device_routing import (
    MultiGPUWorkUnits,
    SelectCLIPDevice,
    SelectModelDevice,
    SelectVAEDevice,
)
from .freeu import (
    FreeU,
    FreeU_V2,
)
from .hook_checkpoint import (
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
)
from .hook_conditioning import (
    ConditioningSetDefaultCombine,
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    ConditioningTimestepsRange,
    PairConditioningCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    SetClipHooks,
)
from .hook_lora import (
    CreateHookLora,
    CreateHookLoraModelOnly,
)
from .hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    SetHookKeyframes,
)
from .hypernetwork import (
    HypernetworkLoader,
)
from .latent_upscale_loader import (
    LatentUpscaleModelLoader,
)
from .lora_extract_save import (
    LoraSave,
)
from .lora_file_loaders import (
    LoadLoraModel,
    LoadLoraTextEncoder,
    LoraLoader,
    LoraLoaderBypass,
    LoraLoaderBypassModelOnly,
    LoraLoaderModelOnly,
)
from .model_advanced import (
    ModelAttentionBackend,
    ModelComputeDtype,
    ModelNoiseScale,
    ModelSamplingAuraFlow,
    ModelSamplingContinuousEDM,
    ModelSamplingContinuousV,
    ModelSamplingDiscrete,
    ModelSamplingFlux,
    ModelSamplingSD3,
    ModelSamplingStableCascade,
    RescaleCFG,
)
from .model_file_loaders import (
    CheckpointLoader,
    CheckpointLoaderSimple,
    CLIPLoader,
    CLIPVisionLoader,
    ControlNetLoader,
    DiffControlNetLoader,
    DualCLIPLoader,
    StyleModelLoader,
    UNETLoader,
)
from .model_file_savers import (
    CheckpointSave,
    CLIPSave,
    ModelSave,
    VAESave,
)
from .model_merging import (
    CLIPMergeAdd,
    CLIPMergeSimple,
    CLIPMergeSubtract,
    ModelMergeAdd,
    ModelMergeBlocks,
    ModelMergeSimple,
    ModelMergeSubtract,
)
from .model_merging_specific import (
    ModelMergeAuraflow,
    ModelMergeCosmos7B,
    ModelMergeCosmos14B,
    ModelMergeCosmosPredict2_2B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeFlux1,
    ModelMergeKrea2,
    ModelMergeLTXV,
    ModelMergeMochiPreview,
    ModelMergeQwenImage,
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSD3_2B,
    ModelMergeSD35_Large,
    ModelMergeSDXL,
    ModelMergeWAN2_1,
)
from .model_patch_loader import (
    ModelPatchLoader,
)
from .model_selections import (
    DiffusersLoader,
    VAELoader,
)
from .rope_and_mask import (
    DifferentialDiffusion,
    ScaleROPE,
)
from .sparse_attention import (
    BlockSparseAttention,
)
from .spatial_model_patches import (
    HyperTile,
    PatchModelAddDownscale,
    TomePatchModel,
)

NODES = (
    UNetSelfAttentionMultiply,
    UNetCrossAttentionMultiply,
    CLIPAttentionMultiply,
    UNetTemporalAttentionMultiply,
    EasyCache,
    LazyCache,
    TripleCLIPLoader,
    QuadrupleCLIPLoader,
    ModelMergeSimple,
    ModelMergeSubtract,
    ModelMergeAdd,
    CLIPMergeSimple,
    CLIPMergeSubtract,
    CLIPMergeAdd,
    ModelMergeBlocks,
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSDXL,
    ModelMergeSD3_2B,
    ModelMergeAuraflow,
    ModelMergeFlux1,
    ModelMergeSD35_Large,
    ModelMergeMochiPreview,
    ModelMergeLTXV,
    ModelMergeCosmos7B,
    ModelMergeCosmos14B,
    ModelMergeWAN2_1,
    ModelMergeCosmosPredict2_2B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeQwenImage,
    ModelMergeKrea2,
    ModelSamplingDiscrete,
    ModelSamplingContinuousEDM,
    ModelSamplingContinuousV,
    ModelSamplingStableCascade,
    ModelSamplingSD3,
    ModelSamplingAuraFlow,
    ModelSamplingFlux,
    ModelNoiseScale,
    RescaleCFG,
    ModelComputeDtype,
    ModelAttentionBackend,
    HypernetworkLoader,
    ModelPatchLoader,
    MultiGPUWorkUnits,
    SelectModelDevice,
    SelectCLIPDevice,
    SelectVAEDevice,
    TorchCompileModel,
    FreeU,
    FreeU_V2,
    TomePatchModel,
    HyperTile,
    PatchModelAddDownscale,
    ScaleROPE,
    DifferentialDiffusion,
    BlockSparseAttention,
    SetHookKeyframes,
    CreateHookKeyframe,
    CreateHookKeyframesInterpolated,
    CreateHookKeyframesFromFloats,
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    LatentUpscaleModelLoader,
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
    CreateHookLora,
    CreateHookLoraModelOnly,
    ContextWindowsManual,
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    ConditioningSetDefaultCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningCombine,
    SetClipHooks,
    ConditioningTimestepsRange,
    VAELoader,
    DiffusersLoader,
    CheckpointLoaderSimple,
    CheckpointLoader,
    ControlNetLoader,
    DiffControlNetLoader,
    UNETLoader,
    CLIPLoader,
    DualCLIPLoader,
    CLIPVisionLoader,
    StyleModelLoader,
    LoraLoader,
    LoraLoaderModelOnly,
    LoadLoraModel,
    LoadLoraTextEncoder,
    LoraLoaderBypass,
    LoraLoaderBypassModelOnly,
    CLIPSave,
    VAESave,
    LoraSave,
    CheckpointSave,
    ModelSave,
)

inference_nodes(NODES)
