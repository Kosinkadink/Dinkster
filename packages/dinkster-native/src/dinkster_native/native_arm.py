"""Native execution node registration assembled from cohesive node modules."""

# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import (
    native_arm_conditioning,
    native_arm_core,
    native_arm_latent_utils,
    native_arm_runtime,
    native_arm_scheduling,
    nodes_conditioning,
    nodes_custom_sampling,
    nodes_generation_loaders,
    nodes_guidance,
    nodes_latent,
    nodes_loaders,
    nodes_model3d,
    nodes_provider,
    nodes_samplers,
    nodes_sampling_runtime,
)
from .families import ltx as nodes_ltx
from .families import minimax_h3 as nodes_minimax
from .families import seedvr2 as nodes_vae_seedvr2
from .families import wan21 as nodes_wan
from .native_arm_registry import (
    GENERATION_PROVIDER_NODES as GENERATION_PROVIDER_NODES,
)
from .native_arm_registry import (
    NATIVE_ARM_NODES as NATIVE_ARM_NODES,
)
from .native_arm_registry import (
    NATIVE_ARM_TYPE_IDS as NATIVE_ARM_TYPE_IDS,
)
from .native_arm_registry import (
    NATIVE_SCHEDULING_NODES as NATIVE_SCHEDULING_NODES,
)
from .native_arm_registry import (
    GenerationAdaptiveProjectedGuidance as GenerationAdaptiveProjectedGuidance,
)
from .native_arm_registry import (
    GenerationAddNoise as GenerationAddNoise,
)
from .native_arm_registry import (
    GenerationAlignYourStepsScheduler as GenerationAlignYourStepsScheduler,
)
from .native_arm_registry import (
    GenerationApplyControlNet as GenerationApplyControlNet,
)
from .native_arm_registry import (
    GenerationApplyControlNetAdvanced as GenerationApplyControlNetAdvanced,
)
from .native_arm_registry import (
    GenerationApplyLoraStack as GenerationApplyLoraStack,
)
from .native_arm_registry import (
    GenerationApplyLoraStackModelOnly as GenerationApplyLoraStackModelOnly,
)
from .native_arm_registry import (
    GenerationApplyTextureToMesh as GenerationApplyTextureToMesh,
)
from .native_arm_registry import (
    GenerationAttentionSchedule as GenerationAttentionSchedule,
)
from .native_arm_registry import (
    GenerationBakeAmbientOcclusion as GenerationBakeAmbientOcclusion,
)
from .native_arm_registry import (
    GenerationBakeNormalMapFromMesh as GenerationBakeNormalMapFromMesh,
)
from .native_arm_registry import (
    GenerationBakeTextureFromVoxel as GenerationBakeTextureFromVoxel,
)
from .native_arm_registry import (
    GenerationBasicGuider as GenerationBasicGuider,
)
from .native_arm_registry import (
    GenerationBasicScheduler as GenerationBasicScheduler,
)
from .native_arm_registry import (
    GenerationBetaSamplingScheduler as GenerationBetaSamplingScheduler,
)
from .native_arm_registry import (
    GenerationCFGGuider as GenerationCFGGuider,
)
from .native_arm_registry import (
    GenerationCfgNorm as GenerationCfgNorm,
)
from .native_arm_registry import (
    GenerationCFGOverride as GenerationCFGOverride,
)
from .native_arm_registry import (
    GenerationCfgZeroStar as GenerationCfgZeroStar,
)
from .native_arm_registry import (
    GenerationChromaModelSampling as GenerationChromaModelSampling,
)
from .native_arm_registry import (
    GenerationChromaRadianceOptions as GenerationChromaRadianceOptions,
)
from .native_arm_registry import (
    GenerationClipSetLastLayer as GenerationClipSetLastLayer,
)
from .native_arm_registry import (
    GenerationClipTextEncode as GenerationClipTextEncode,
)
from .native_arm_registry import (
    GenerationClipTextEncodeControlnet as GenerationClipTextEncodeControlnet,
)
from .native_arm_registry import (
    GenerationClipTextEncodeLumina2 as GenerationClipTextEncodeLumina2,
)
from .native_arm_registry import (
    GenerationConditioningMerge as GenerationConditioningMerge,
)
from .native_arm_registry import (
    GenerationConditioningScale as GenerationConditioningScale,
)
from .native_arm_registry import (
    GenerationConditioningSetArea as GenerationConditioningSetArea,
)
from .native_arm_registry import (
    GenerationConditioningSetMask as GenerationConditioningSetMask,
)
from .native_arm_registry import (
    GenerationConditioningSetTimestepRange as GenerationConditioningSetTimestepRange,
)
from .native_arm_registry import (
    GenerationConditioningZeroOut as GenerationConditioningZeroOut,
)
from .native_arm_registry import (
    GenerationContextWindowsManual as GenerationContextWindowsManual,
)
from .native_arm_registry import (
    GenerationDecimateMesh as GenerationDecimateMesh,
)
from .native_arm_registry import (
    GenerationDisableCFG1Optimization as GenerationDisableCFG1Optimization,
)
from .native_arm_registry import (
    GenerationDisableNoise as GenerationDisableNoise,
)
from .native_arm_registry import (
    GenerationDualCFGGuider as GenerationDualCFGGuider,
)
from .native_arm_registry import (
    GenerationDualModelGuider as GenerationDualModelGuider,
)
from .native_arm_registry import (
    GenerationEasyCache as GenerationEasyCache,
)
from .native_arm_registry import (
    GenerationEmptyChromaRadianceLatentImage as GenerationEmptyChromaRadianceLatentImage,
)
from .native_arm_registry import (
    GenerationEmptyFlux2LatentImage as GenerationEmptyFlux2LatentImage,
)
from .native_arm_registry import (
    GenerationEmptyLatentImage as GenerationEmptyLatentImage,
)
from .native_arm_registry import (
    GenerationEmptyLTXAVLatent as GenerationEmptyLTXAVLatent,
)
from .native_arm_registry import (
    GenerationEmptyLTXVLatent as GenerationEmptyLTXVLatent,
)
from .native_arm_registry import (
    GenerationEmptySD3LatentImage as GenerationEmptySD3LatentImage,
)
from .native_arm_registry import (
    GenerationEmptyTrellis2LatentStructure as GenerationEmptyTrellis2LatentStructure,
)
from .native_arm_registry import (
    GenerationEpsilonScaling as GenerationEpsilonScaling,
)
from .native_arm_registry import (
    GenerationEstimateGeometry as GenerationEstimateGeometry,
)
from .native_arm_registry import (
    GenerationExponentialScheduler as GenerationExponentialScheduler,
)
from .native_arm_registry import (
    GenerationExtendIntermediateSigmas as GenerationExtendIntermediateSigmas,
)
from .native_arm_registry import (
    GenerationFlipSigmas as GenerationFlipSigmas,
)
from .native_arm_registry import (
    GenerationFlux2Scheduler as GenerationFlux2Scheduler,
)
from .native_arm_registry import (
    GenerationFluxDisableGuidance as GenerationFluxDisableGuidance,
)
from .native_arm_registry import (
    GenerationFluxGuidance as GenerationFluxGuidance,
)
from .native_arm_registry import (
    GenerationFreSca as GenerationFreSca,
)
from .native_arm_registry import (
    GenerationGeometryToFOV as GenerationGeometryToFOV,
)
from .native_arm_registry import (
    GenerationGetMeshInfo as GenerationGetMeshInfo,
)
from .native_arm_registry import (
    GenerationGITSScheduler as GenerationGITSScheduler,
)
from .native_arm_registry import (
    GenerationIdeogram4Scheduler as GenerationIdeogram4Scheduler,
)
from .native_arm_registry import (
    GenerationImageCropToMask as GenerationImageCropToMask,
)
from .native_arm_registry import (
    GenerationImpactRegionalSampler as GenerationImpactRegionalSampler,
)
from .native_arm_registry import (
    GenerationKarrasScheduler as GenerationKarrasScheduler,
)
from .native_arm_registry import (
    GenerationKSampler as GenerationKSampler,
)
from .native_arm_registry import (
    GenerationKSamplerAdvanced as GenerationKSamplerAdvanced,
)
from .native_arm_registry import (
    GenerationKSamplerSelect as GenerationKSamplerSelect,
)
from .native_arm_registry import (
    GenerationLaplaceScheduler as GenerationLaplaceScheduler,
)
from .native_arm_registry import (
    GenerationLatentApplyOperation as GenerationLatentApplyOperation,
)
from .native_arm_registry import (
    GenerationLatentApplyOperationCFG as GenerationLatentApplyOperationCFG,
)
from .native_arm_registry import (
    GenerationLatentBatch as GenerationLatentBatch,
)
from .native_arm_registry import (
    GenerationLatentCombine as GenerationLatentCombine,
)
from .native_arm_registry import (
    GenerationLatentComposite as GenerationLatentComposite,
)
from .native_arm_registry import (
    GenerationLatentCompositeMasked as GenerationLatentCompositeMasked,
)
from .native_arm_registry import (
    GenerationLatentConcat as GenerationLatentConcat,
)
from .native_arm_registry import (
    GenerationLatentCrop as GenerationLatentCrop,
)
from .native_arm_registry import (
    GenerationLatentCut as GenerationLatentCut,
)
from .native_arm_registry import (
    GenerationLatentCutToBatch as GenerationLatentCutToBatch,
)
from .native_arm_registry import (
    GenerationLatentFlip as GenerationLatentFlip,
)
from .native_arm_registry import (
    GenerationLatentFromBatch as GenerationLatentFromBatch,
)
from .native_arm_registry import (
    GenerationLatentGenerateNoise as GenerationLatentGenerateNoise,
)
from .native_arm_registry import (
    GenerationLatentInjectNoise as GenerationLatentInjectNoise,
)
from .native_arm_registry import (
    GenerationLatentMix as GenerationLatentMix,
)
from .native_arm_registry import (
    GenerationLatentMultiply as GenerationLatentMultiply,
)
from .native_arm_registry import (
    GenerationLatentOperationSharpen as GenerationLatentOperationSharpen,
)
from .native_arm_registry import (
    GenerationLatentOperationTonemapReinhard as GenerationLatentOperationTonemapReinhard,
)
from .native_arm_registry import (
    GenerationLatentRebatch as GenerationLatentRebatch,
)
from .native_arm_registry import (
    GenerationLatentRepeat as GenerationLatentRepeat,
)
from .native_arm_registry import (
    GenerationLatentReplaceFrames as GenerationLatentReplaceFrames,
)
from .native_arm_registry import (
    GenerationLatentResize as GenerationLatentResize,
)
from .native_arm_registry import (
    GenerationLatentResizeBy as GenerationLatentResizeBy,
)
from .native_arm_registry import (
    GenerationLatentRotate as GenerationLatentRotate,
)
from .native_arm_registry import (
    GenerationLatentSeedBehavior as GenerationLatentSeedBehavior,
)
from .native_arm_registry import (
    GenerationLatentSetNoiseMask as GenerationLatentSetNoiseMask,
)
from .native_arm_registry import (
    GenerationLazyCache as GenerationLazyCache,
)
from .native_arm_registry import (
    GenerationLoadBackgroundRemoval as GenerationLoadBackgroundRemoval,
)
from .native_arm_registry import (
    GenerationLoadCheckpoint as GenerationLoadCheckpoint,
)
from .native_arm_registry import (
    GenerationLoadCheckpointStack as GenerationLoadCheckpointStack,
)
from .native_arm_registry import (
    GenerationLoadControlNet as GenerationLoadControlNet,
)
from .native_arm_registry import (
    GenerationLoadDiffusionComponents as GenerationLoadDiffusionComponents,
)
from .native_arm_registry import (
    GenerationLoadDiffusionModel as GenerationLoadDiffusionModel,
)
from .native_arm_registry import (
    GenerationLoadGeometryModel as GenerationLoadGeometryModel,
)
from .native_arm_registry import (
    GenerationLoadLatentUpscaleModel as GenerationLoadLatentUpscaleModel,
)
from .native_arm_registry import (
    GenerationLoadLora as GenerationLoadLora,
)
from .native_arm_registry import (
    GenerationLoadLoraModelOnly as GenerationLoadLoraModelOnly,
)
from .native_arm_registry import (
    GenerationLoadLTXAVAudioVAE as GenerationLoadLTXAVAudioVAE,
)
from .native_arm_registry import (
    GenerationLoadLTXAVTextEncoder as GenerationLoadLTXAVTextEncoder,
)
from .native_arm_registry import (
    GenerationLTXAVAudioVAEDecode as GenerationLTXAVAudioVAEDecode,
)
from .native_arm_registry import (
    GenerationLTXAVConditioning as GenerationLTXAVConditioning,
)
from .native_arm_registry import (
    GenerationLTXAVIDLoRAReferenceAudio as GenerationLTXAVIDLoRAReferenceAudio,
)
from .native_arm_registry import (
    GenerationLTXAVReferenceAudio as GenerationLTXAVReferenceAudio,
)
from .native_arm_registry import (
    GenerationLTXVAddGuide as GenerationLTXVAddGuide,
)
from .native_arm_registry import (
    GenerationLTXVConditioning as GenerationLTXVConditioning,
)
from .native_arm_registry import (
    GenerationLTXVContextWindows as GenerationLTXVContextWindows,
)
from .native_arm_registry import (
    GenerationLTXVCropGuides as GenerationLTXVCropGuides,
)
from .native_arm_registry import (
    GenerationLTXVDualCFGGuider as GenerationLTXVDualCFGGuider,
)
from .native_arm_registry import (
    GenerationLTXVDurationPredictor as GenerationLTXVDurationPredictor,
)
from .native_arm_registry import (
    GenerationLTXVImageToVideo as GenerationLTXVImageToVideo,
)
from .native_arm_registry import (
    GenerationLTXVImageToVideoInplace as GenerationLTXVImageToVideoInplace,
)
from .native_arm_registry import (
    GenerationLTXVLatentUpsampler as GenerationLTXVLatentUpsampler,
)
from .native_arm_registry import (
    GenerationLTXVModalityGuidance as GenerationLTXVModalityGuidance,
)
from .native_arm_registry import (
    GenerationLTXVSpatioTemporalGuidance as GenerationLTXVSpatioTemporalGuidance,
)
from .native_arm_registry import (
    GenerationMahiroGuidance as GenerationMahiroGuidance,
)
from .native_arm_registry import (
    GenerationManualSigmas as GenerationManualSigmas,
)
from .native_arm_registry import (
    GenerationMeshToModel3D as GenerationMeshToModel3D,
)
from .native_arm_registry import (
    GenerationModelSamplingAuraFlow as GenerationModelSamplingAuraFlow,
)
from .native_arm_registry import (
    GenerationModelSamplingFlux as GenerationModelSamplingFlux,
)
from .native_arm_registry import (
    GenerationModelSamplingLTXV as GenerationModelSamplingLTXV,
)
from .native_arm_registry import (
    GenerationModelSamplingSD3 as GenerationModelSamplingSD3,
)
from .native_arm_registry import (
    GenerationNormalizedAttentionGuidance as GenerationNormalizedAttentionGuidance,
)
from .native_arm_registry import (
    GenerationOptimalStepsScheduler as GenerationOptimalStepsScheduler,
)
from .native_arm_registry import (
    GenerationPaintMesh as GenerationPaintMesh,
)
from .native_arm_registry import (
    GenerationPerpNegGuider as GenerationPerpNegGuider,
)
from .native_arm_registry import (
    GenerationPixal3DConditioning as GenerationPixal3DConditioning,
)
from .native_arm_registry import (
    GenerationPolyexponentialScheduler as GenerationPolyexponentialScheduler,
)
from .native_arm_registry import (
    GenerationPreviewMask as GenerationPreviewMask,
)
from .native_arm_registry import (
    GenerationPromptEnhance as GenerationPromptEnhance,
)
from .native_arm_registry import (
    GenerationRandomNoise as GenerationRandomNoise,
)
from .native_arm_registry import (
    GenerationReferenceLatent as GenerationReferenceLatent,
)
from .native_arm_registry import (
    GenerationRemeshMesh as GenerationRemeshMesh,
)
from .native_arm_registry import (
    GenerationRemoveBackground as GenerationRemoveBackground,
)
from .native_arm_registry import (
    GenerationRenderUVAtlas as GenerationRenderUVAtlas,
)
from .native_arm_registry import (
    GenerationRenormCfg as GenerationRenormCfg,
)
from .native_arm_registry import (
    GenerationRescaleCfg as GenerationRescaleCfg,
)
from .native_arm_registry import (
    GenerationSamplerCustom as GenerationSamplerCustom,
)
from .native_arm_registry import (
    GenerationSamplerCustomAdvanced as GenerationSamplerCustomAdvanced,
)
from .native_arm_registry import (
    GenerationSamplerDPMAdaptative as GenerationSamplerDPMAdaptative,
)
from .native_arm_registry import (
    GenerationSamplerDPMPP2MSDE as GenerationSamplerDPMPP2MSDE,
)
from .native_arm_registry import (
    GenerationSamplerDPMPP2SAncestral as GenerationSamplerDPMPP2SAncestral,
)
from .native_arm_registry import (
    GenerationSamplerDPMPP3MSDE as GenerationSamplerDPMPP3MSDE,
)
from .native_arm_registry import (
    GenerationSamplerDPMPPSDE as GenerationSamplerDPMPPSDE,
)
from .native_arm_registry import (
    GenerationSamplerERSDE as GenerationSamplerERSDE,
)
from .native_arm_registry import (
    GenerationSamplerEulerAncestral as GenerationSamplerEulerAncestral,
)
from .native_arm_registry import (
    GenerationSamplerEulerAncestralCFGPP as GenerationSamplerEulerAncestralCFGPP,
)
from .native_arm_registry import (
    GenerationSamplerLMS as GenerationSamplerLMS,
)
from .native_arm_registry import (
    GenerationSamplerSASolver as GenerationSamplerSASolver,
)
from .native_arm_registry import (
    GenerationSamplerSEEDS2 as GenerationSamplerSEEDS2,
)
from .native_arm_registry import (
    GenerationSamplingPercentToSigma as GenerationSamplingPercentToSigma,
)
from .native_arm_registry import (
    GenerationScheduledCFGGuider as GenerationScheduledCFGGuider,
)
from .native_arm_registry import (
    GenerationSDTurboScheduler as GenerationSDTurboScheduler,
)
from .native_arm_registry import (
    GenerationSeedVR2Conditioning as GenerationSeedVR2Conditioning,
)
from .native_arm_registry import (
    GenerationSeedVR2PostProcessing as GenerationSeedVR2PostProcessing,
)
from .native_arm_registry import (
    GenerationSeedVR2Preprocess as GenerationSeedVR2Preprocess,
)
from .native_arm_registry import (
    GenerationSeedVR2TemporalChunk as GenerationSeedVR2TemporalChunk,
)
from .native_arm_registry import (
    GenerationSeedVR2TemporalMerge as GenerationSeedVR2TemporalMerge,
)
from .native_arm_registry import (
    GenerationSetControlNetUnionType as GenerationSetControlNetUnionType,
)
from .native_arm_registry import (
    GenerationSetFirstSigma as GenerationSetFirstSigma,
)
from .native_arm_registry import (
    GenerationSmoothMeshNormals as GenerationSmoothMeshNormals,
)
from .native_arm_registry import (
    GenerationSplitSigmas as GenerationSplitSigmas,
)
from .native_arm_registry import (
    GenerationSplitSigmasDenoise as GenerationSplitSigmasDenoise,
)
from .native_arm_registry import (
    GenerationT5TokenizerOptions as GenerationT5TokenizerOptions,
)
from .native_arm_registry import (
    GenerationTCFG as GenerationTCFG,
)
from .native_arm_registry import (
    GenerationTemporalScoreRescaling as GenerationTemporalScoreRescaling,
)
from .native_arm_registry import (
    GenerationTextGenerate as GenerationTextGenerate,
)
from .native_arm_registry import (
    GenerationTrellis2Conditioning as GenerationTrellis2Conditioning,
)
from .native_arm_registry import (
    GenerationTrellis2ShapeStage as GenerationTrellis2ShapeStage,
)
from .native_arm_registry import (
    GenerationTrellis2TextureStage as GenerationTrellis2TextureStage,
)
from .native_arm_registry import (
    GenerationTrellis2UpsampleStage as GenerationTrellis2UpsampleStage,
)
from .native_arm_registry import (
    GenerationUnwrapMesh as GenerationUnwrapMesh,
)
from .native_arm_registry import (
    GenerationVAEDecode as GenerationVAEDecode,
)
from .native_arm_registry import (
    GenerationVaeDecodeShapeTrellis as GenerationVaeDecodeShapeTrellis,
)
from .native_arm_registry import (
    GenerationVaeDecodeStructureTrellis2 as GenerationVaeDecodeStructureTrellis2,
)
from .native_arm_registry import (
    GenerationVaeDecodeTextureTrellis as GenerationVaeDecodeTextureTrellis,
)
from .native_arm_registry import (
    GenerationVAEDecodeTiled as GenerationVAEDecodeTiled,
)
from .native_arm_registry import (
    GenerationVAEEncode as GenerationVAEEncode,
)
from .native_arm_registry import (
    GenerationVAEEncodeTiled as GenerationVAEEncodeTiled,
)
from .native_arm_registry import (
    GenerationVoxelToMesh as GenerationVoxelToMesh,
)
from .native_arm_registry import (
    GenerationVPScheduler as GenerationVPScheduler,
)
from .native_arm_registry import (
    GenerationWanContextWindowsManual as GenerationWanContextWindowsManual,
)
from .native_arm_registry import (
    NativeApplyZImageControlPatch as NativeApplyZImageControlPatch,
)
from .native_arm_registry import (
    NativeBerniniConditioning as NativeBerniniConditioning,
)
from .native_arm_registry import (
    NativeClipTextEncode as NativeClipTextEncode,
)
from .native_arm_registry import (
    NativeConcatAVLatent as NativeConcatAVLatent,
)
from .native_arm_registry import (
    NativeConditioningSetPropertiesAndCombine as NativeConditioningSetPropertiesAndCombine,
)
from .native_arm_registry import (
    NativeConditioningTimestepsRange as NativeConditioningTimestepsRange,
)
from .native_arm_registry import (
    NativeControlNetApply as NativeControlNetApply,
)
from .native_arm_registry import (
    NativeControlNetApplyAdvanced as NativeControlNetApplyAdvanced,
)
from .native_arm_registry import (
    NativeControlNetLoader as NativeControlNetLoader,
)
from .native_arm_registry import (
    NativeCreateHookKeyframe as NativeCreateHookKeyframe,
)
from .native_arm_registry import (
    NativeCreateHookLora as NativeCreateHookLora,
)
from .native_arm_registry import (
    NativeEmptyLTXAVLatent as NativeEmptyLTXAVLatent,
)
from .native_arm_registry import (
    NativeEmptyLTXVLatent as NativeEmptyLTXVLatent,
)
from .native_arm_registry import (
    NativeEmptyMiniMaxH3AV as NativeEmptyMiniMaxH3AV,
)
from .native_arm_registry import (
    NativeEmptyMiniMaxMusic3LatentAudio as NativeEmptyMiniMaxMusic3LatentAudio,
)
from .native_arm_registry import (
    NativeInspectLatentMask as NativeInspectLatentMask,
)
from .native_arm_registry import (
    NativeKSampler as NativeKSampler,
)
from .native_arm_registry import (
    NativeKSamplerAdvanced as NativeKSamplerAdvanced,
)
from .native_arm_registry import (
    NativeLoadCheckpoint as NativeLoadCheckpoint,
)
from .native_arm_registry import (
    NativeLoadClip as NativeLoadClip,
)
from .native_arm_registry import (
    NativeLoadDiffusionModel as NativeLoadDiffusionModel,
)
from .native_arm_registry import (
    NativeLoadDualClip as NativeLoadDualClip,
)
from .native_arm_registry import (
    NativeLoadLora as NativeLoadLora,
)
from .native_arm_registry import (
    NativeLoadLoraModelOnly as NativeLoadLoraModelOnly,
)
from .native_arm_registry import (
    NativeLoadModelProfile as NativeLoadModelProfile,
)
from .native_arm_registry import (
    NativeLoadVae as NativeLoadVae,
)
from .native_arm_registry import (
    NativeLoadVision as NativeLoadVision,
)
from .native_arm_registry import (
    NativeLoadZImageControlPatch as NativeLoadZImageControlPatch,
)
from .native_arm_registry import (
    NativeMiniMaxH3AddGuide as NativeMiniMaxH3AddGuide,
)
from .native_arm_registry import (
    NativeMiniMaxH3AVDecode as NativeMiniMaxH3AVDecode,
)
from .native_arm_registry import (
    NativeMiniMaxH3AVEncode as NativeMiniMaxH3AVEncode,
)
from .native_arm_registry import (
    NativeMiniMaxH3FL2VAConditioning as NativeMiniMaxH3FL2VAConditioning,
)
from .native_arm_registry import (
    NativeMiniMaxH3ImageToVideo as NativeMiniMaxH3ImageToVideo,
)
from .native_arm_registry import (
    NativeMiniMaxH3MotionContext as NativeMiniMaxH3MotionContext,
)
from .native_arm_registry import (
    NativeMiniMaxH3REF2VAConditioning as NativeMiniMaxH3REF2VAConditioning,
)
from .native_arm_registry import (
    NativeMiniMaxH3ReferenceToVideo as NativeMiniMaxH3ReferenceToVideo,
)
from .native_arm_registry import (
    NativeMiniMaxH3T2VAConditioning as NativeMiniMaxH3T2VAConditioning,
)
from .native_arm_registry import (
    NativeMiniMaxMusic3TextEncode as NativeMiniMaxMusic3TextEncode,
)
from .native_arm_registry import (
    NativePairConditioningSetProperties as NativePairConditioningSetProperties,
)
from .native_arm_registry import (
    NativePreviewLatentAudio as NativePreviewLatentAudio,
)
from .native_arm_registry import (
    NativePreviewLatentVisual as NativePreviewLatentVisual,
)
from .native_arm_registry import (
    NativeRuntimeHandle as NativeRuntimeHandle,
)
from .native_arm_registry import (
    NativeSeparateAVLatent as NativeSeparateAVLatent,
)
from .native_arm_registry import (
    NativeSetHookKeyframes as NativeSetHookKeyframes,
)
from .native_arm_registry import (
    NativeSetLatentMaskFromFrames as NativeSetLatentMaskFromFrames,
)
from .native_arm_registry import (
    NativeSetLatentMaskFromTimeRanges as NativeSetLatentMaskFromTimeRanges,
)
from .native_arm_registry import (
    NativeVAEDecode as NativeVAEDecode,
)
from .native_arm_registry import (
    NativeVAEDecodeAudio as NativeVAEDecodeAudio,
)
from .native_arm_registry import (
    NativeVAEDecodeAudioTiled as NativeVAEDecodeAudioTiled,
)
from .native_arm_registry import (
    NativeVAEEncode as NativeVAEEncode,
)
from .native_arm_registry import (
    NativeWan21ClipVisionEncode as NativeWan21ClipVisionEncode,
)
from .native_arm_registry import (
    NativeWan21ImageToVideo as NativeWan21ImageToVideo,
)
from .native_arm_registry import (
    NativeWan22FunControlToVideo as NativeWan22FunControlToVideo,
)
from .native_arm_registry import (
    NativeWan22ImageToVideoLatent as NativeWan22ImageToVideoLatent,
)
from .native_arm_registry import (
    NativeWanCameraEmbedding as NativeWanCameraEmbedding,
)
from .native_arm_registry import (
    NativeWanCameraImageToVideo as NativeWanCameraImageToVideo,
)
from .native_arm_registry import (
    NativeWanFirstLastFrameToVideo as NativeWanFirstLastFrameToVideo,
)
from .native_arm_registry import (
    NativeWanFunControlToVideo as NativeWanFunControlToVideo,
)
from .native_arm_registry import (
    NativeWanFunInpaintToVideo as NativeWanFunInpaintToVideo,
)
from .native_arm_registry import (
    NativeWanMoveConcatTrack as NativeWanMoveConcatTrack,
)
from .native_arm_registry import (
    NativeWanMoveGenerateTracks as NativeWanMoveGenerateTracks,
)
from .native_arm_registry import (
    NativeWanMoveTracksFromCoords as NativeWanMoveTracksFromCoords,
)
from .native_arm_registry import (
    NativeWanMoveTrackToVideo as NativeWanMoveTrackToVideo,
)
from .native_arm_registry import (
    NativeWanMoveVisualizeTracks as NativeWanMoveVisualizeTracks,
)
from .native_arm_registry import (
    NativeWanPhantomSubjectToVideo as NativeWanPhantomSubjectToVideo,
)
from .native_arm_registry import (
    NativeWanTrackToVideo as NativeWanTrackToVideo,
)
from .native_arm_registry import (
    NativeWanVaceToVideo as NativeWanVaceToVideo,
)
from .native_arm_registry import __all__ as __all__
from .native_arm_registry import (
    load_native_runtime_handle as load_native_runtime_handle,
)
from .native_arm_runtime import _NativeModelOverlay as _NativeModelOverlay

_CustomSigmasValue = native_arm_core._CustomSigmasValue

_IMPLEMENTATION_MODULES = (
    native_arm_conditioning,
    native_arm_core,
    native_arm_latent_utils,
    native_arm_runtime,
    native_arm_scheduling,
    nodes_conditioning,
    nodes_custom_sampling,
    nodes_generation_loaders,
    nodes_guidance,
    nodes_latent,
    nodes_loaders,
    nodes_ltx,
    nodes_minimax,
    nodes_model3d,
    nodes_provider,
    nodes_samplers,
    nodes_sampling_runtime,
    nodes_vae_seedvr2,
    nodes_wan,
)


def __getattr__(name: str) -> Any:
    for module in _IMPLEMENTATION_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


class _NativeArmModule(ModuleType):
    """Keep existing native_arm monkeypatch and import behavior across split modules."""

    def __setattr__(self, name: str, value: object) -> None:
        for module in _IMPLEMENTATION_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _NativeArmModule
