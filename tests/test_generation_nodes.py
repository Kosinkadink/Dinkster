from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from dinkster_api.v1 import prepare_image_array_encoding
from dinkster_assets import AssetRef
from dinkster_compat_comfy import translate_prompt
from dinkster_compat_comfy.native import GENERATION_CLAIMED_V1_NAMES, NATIVE_NODES
from dinkster_compat_comfy.native_arm import GENERATION_PROVIDER_NODES
from dinkster_compat_comfy.usdu import USDU_CARRIER_NODES
from dinkster_graph import GraphNode, Link
from dinkster_nodes_generation import (
    GENERATION_COMPAT_CARRIER_NODE_IDS,
    GENERATION_COMPAT_CARRIER_NODES,
    GENERATION_NODE_IDS,
    GENERATION_NODES,
    GENERATION_SCHEMA_NODE_IDS,
)
from dinkster_nodes_generation_openai import OPENAI_GENERATION_NODES
from dinkster_nodes_image import UpscaleWithModel
from dinkster_nodes_media_io import (
    IMAGE_TYPE,
    MASK_TYPE,
    LoadVideo,
    SaveVideo,
    register_media_types,
)
from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    derive_attention_route_token,
)
from dinkster_schema import (
    AssetWidget,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    StringWidget,
    build_node_types,
    build_schemas,
    elaborate,
    schema_signature,
    schema_to_wire,
)
from dinkster_server import STATE_KEY, create_app
from dinkster_values import TypeRegistry
from dinkster_workers import InProcessWorker, PackManifest, load_manifest

from dinkster.comfy_compose import comfy_compat_specs
from dinkster.compose import (
    CompositionError,
    PackDelta,
    PackSpec,
    ServingComposer,
    default_pack_spec,
)
from dinkster.reload_api import apply_reload

MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-nodes-generation" / "dinkster-pack.toml"
)
COMPAT_MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"
)
UPSCALE_MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-vision-upscale" / "dinkster-pack.toml"
)
OPENAI_MANIFEST = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-nodes-generation-openai"
    / "dinkster-pack.toml"
)


def _attention_token() -> AttentionRouteToken:
    return derive_attention_route_token(_attention_capabilities(), AttentionPolicyConfig())


def _attention_capabilities() -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
    )


class _ComfyArmSchema(Node):
    node_type: ClassVar[str]

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type=cls.node_type)

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("schema fixture")


class _CreateHookLora(_ComfyArmSchema):
    node_type = "dinkster.create_hook_lora"


class _CreateHookKeyframe(_ComfyArmSchema):
    node_type = "dinkster.create_hook_keyframe"


class _SetHookKeyframes(_ComfyArmSchema):
    node_type = "dinkster.set_hook_keyframes"


class _ConditioningTimestepsRange(_ComfyArmSchema):
    node_type = "dinkster.conditioning_timesteps_range"


class _ConditioningSetPropertiesAndCombine(_ComfyArmSchema):
    node_type = "dinkster.conditioning_set_properties_and_combine"


class _PairConditioningSetProperties(_ComfyArmSchema):
    node_type = "dinkster.pair_conditioning_set_properties"


_COMFY_ARM_SCHEMAS = (
    _CreateHookLora,
    _CreateHookKeyframe,
    _SetHookKeyframes,
    _ConditioningTimestepsRange,
    _ConditioningSetPropertiesAndCombine,
    _PairConditioningSetProperties,
)


class _CompatSchemaWorker(InProcessWorker):
    def __init__(self, registry: TypeRegistry) -> None:
        super().__init__(
            build_node_types((*GENERATION_PROVIDER_NODES, *NATIVE_NODES, *_COMFY_ARM_SCHEMAS)),
            registry,
            attention_capabilities=_attention_capabilities(),
            attention_route_token=_attention_token(),
        )

    @property
    def body_arms(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(sorted(types)) for name, types in load_manifest(COMPAT_MANIFEST).arms}


def test_generation_manifest_owns_native_schema_contracts() -> None:
    from dinkster_inference import builtin_samplers, builtin_schedulers

    manifest = load_manifest(MANIFEST)
    compat_manifest = load_manifest(COMPAT_MANIFEST)
    sampler_ids = tuple(item.id for item in builtin_samplers())
    scheduler_ids = tuple(item.id for item in builtin_schedulers())

    assert manifest.name == "dinkster-nodes-generation"
    assert manifest.schema_only == GENERATION_SCHEMA_NODE_IDS
    assert [(item.id, item.version) for item in manifest.capabilities] == [
        ("dinkster.generation.schemas", "1.0.0")
    ]
    for pack_manifest in (manifest, compat_manifest):
        assert {item.registry for item in pack_manifest.requirements.registry} == {
            "dinkster.samplers",
            "dinkster.schedulers",
        }
        actual_sampler_ids = tuple(
            item.id
            for item in pack_manifest.requirements.registry
            if item.registry == "dinkster.samplers"
        )
        actual_scheduler_ids = tuple(
            item.id
            for item in pack_manifest.requirements.registry
            if item.registry == "dinkster.schedulers"
        )
        assert len(actual_sampler_ids) == len(sampler_ids)
        assert set(actual_sampler_ids) == set(sampler_ids)
        assert len(actual_scheduler_ids) == len(scheduler_ids)
        assert set(actual_scheduler_ids) == set(scheduler_ids)


def test_generation_schemas_use_native_boundary_types_only() -> None:
    from dinkster_inference import builtin_samplers, builtin_schedulers

    schemas = build_schemas(GENERATION_NODES)

    assert (
        tuple(schemas)
        == GENERATION_NODE_IDS
        == (
            "dinkster.empty_trellis2_latent_structure",
            "dinkster.trellis2_conditioning",
            "dinkster.pixal3d_conditioning",
            "dinkster.vae_decode_structure_trellis2",
            "dinkster.trellis2_shape_stage",
            "dinkster.trellis2_upsample_stage",
            "dinkster.vae_decode_shape_trellis",
            "dinkster.trellis2_texture_stage",
            "dinkster.vae_decode_texture_trellis",
            "dinkster.load_geometry_model",
            "dinkster.estimate_geometry",
            "dinkster.geometry_to_fov",
            "dinkster.load_background_removal",
            "dinkster.remove_background",
            "dinkster.image_crop_to_mask",
            "dinkster.preview_mask",
            "dinkster.voxel_to_mesh",
            "dinkster.get_mesh_info",
            "dinkster.remesh_mesh",
            "dinkster.decimate_mesh",
            "dinkster.smooth_mesh_normals",
            "dinkster.unwrap_mesh",
            "dinkster.paint_mesh",
            "dinkster.bake_texture_from_voxel",
            "dinkster.bake_normal_map_from_mesh",
            "dinkster.bake_ambient_occlusion",
            "dinkster.render_uv_atlas",
            "dinkster.apply_texture_to_mesh",
            "dinkster.mesh_to_model3d",
            "dinkster.load_model_profile",
            "dinkster.load_checkpoint",
            "dinkster.load_controlnet",
            "dinkster.apply_controlnet",
            "dinkster.apply_controlnet_advanced",
            "dinkster.set_controlnet_union_type",
            "dinkster.load_checkpoint_stack",
            "dinkster.load_diffusion_model",
            "dinkster.load_diffusion_components",
            "dinkster.load_ltxav_text_encoder",
            "dinkster.load_ltxav_audio_vae",
            "dinkster.load_latent_upscale_model",
            "dinkster.ltxav_audio_vae_decode",
            "dinkster.load_lora",
            "dinkster.load_lora_model_only",
            "dinkster.apply_lora_stack",
            "dinkster.apply_lora_stack_model_only",
            "dinkster.clip_text_encode",
            "dinkster.clip_text_encode_lumina2",
            "dinkster.model_sampling_aura_flow",
            "dinkster.text_generate",
            "dinkster.prompt_enhance",
            "dinkster.clip_set_last_layer",
            "dinkster.t5_tokenizer_options",
            "dinkster.clip_text_encode_controlnet",
            "dinkster.flux_guidance",
            "dinkster.flux_disable_guidance",
            "dinkster.reference_latent",
            "dinkster.cfg_zero_star",
            "dinkster.cfg_norm",
            "dinkster.tcfg",
            "dinkster.fresca",
            "dinkster.lazy_cache",
            "dinkster.easy_cache",
            "dinkster.attention_schedule",
            "dinkster.context_windows_manual",
            "dinkster.wan_context_windows_manual",
            "dinkster.ltxv_context_windows",
            "dinkster.adaptive_projected_guidance",
            "dinkster.mahiro_guidance",
            "dinkster.epsilon_scaling",
            "dinkster.cfg_override",
            "dinkster.rescale_cfg",
            "dinkster.renorm_cfg",
            "dinkster.temporal_score_rescaling",
            "dinkster.nag",
            "dinkster.ltxav_conditioning",
            "dinkster.ltxav_reference_audio",
            "dinkster.ltxav_id_lora_reference_audio",
            "dinkster.ltxv_spatiotemporal_guidance",
            "dinkster.ltxv_modality_guidance",
            "dinkster.ltxv_duration_predictor",
            "dinkster.ltxv_dual_cfg_guider",
            "dinkster.ltxv_conditioning",
            "dinkster.ltxv_image_to_video",
            "dinkster.ltxv_image_to_video_inplace",
            "dinkster.ltxv_add_guide",
            "dinkster.ltxv_crop_guides",
            "dinkster.ltxv_latent_upsampler",
            "dinkster.conditioning_merge",
            "dinkster.conditioning_scale",
            "dinkster.conditioning_set_area",
            "dinkster.conditioning_set_mask",
            "dinkster.conditioning_set_timestep_range",
            "dinkster.conditioning_zero_out",
            "dinkster.chroma_radiance_options",
            "dinkster.chroma_model_sampling",
            "dinkster.model_sampling_sd3",
            "dinkster.model_sampling_ltxv",
            "dinkster.model_sampling_flux",
            "dinkster.empty_latent_image",
            "dinkster.empty_sd3_latent_image",
            "dinkster.empty_chroma_radiance_latent_image",
            "dinkster.empty_flux2_latent_image",
            "dinkster.empty_ltxav_latent",
            "dinkster.empty_ltxv_latent",
            "dinkster.ksampler",
            "dinkster.ksampler_advanced",
            "dinkster.ksampler_select",
            "dinkster.sampler_dpmpp_3m_sde",
            "dinkster.sampler_dpmpp_2m_sde",
            "dinkster.sampler_dpmpp_sde",
            "dinkster.sampler_dpmpp_2s_ancestral",
            "dinkster.sampler_euler_ancestral",
            "dinkster.sampler_euler_ancestral_cfg_pp",
            "dinkster.sampler_lms",
            "dinkster.sampler_dpm_adaptative",
            "dinkster.sampler_er_sde",
            "dinkster.sampler_seeds_2",
            "dinkster.sampler_sa_solver",
            "dinkster.basic_scheduler",
            "dinkster.beta_sampling_scheduler",
            "dinkster.sd_turbo_scheduler",
            "dinkster.karras_scheduler",
            "dinkster.exponential_scheduler",
            "dinkster.polyexponential_scheduler",
            "dinkster.laplace_scheduler",
            "dinkster.vp_scheduler",
            "dinkster.align_your_steps_scheduler",
            "dinkster.gits_scheduler",
            "dinkster.optimal_steps_scheduler",
            "dinkster.flux2_scheduler",
            "dinkster.ideogram4_scheduler",
            "dinkster.manual_sigmas",
            "dinkster.split_sigmas",
            "dinkster.split_sigmas_denoise",
            "dinkster.flip_sigmas",
            "dinkster.set_first_sigma",
            "dinkster.extend_intermediate_sigmas",
            "dinkster.sampling_percent_to_sigma",
            "dinkster.basic_guider",
            "dinkster.cfg_guider",
            "dinkster.dual_cfg_guider",
            "dinkster.dual_model_guider",
            "dinkster.scheduled_cfg_guider",
            "dinkster.perp_neg_guider",
            "dinkster.disable_cfg1_optimization",
            "dinkster.disable_noise",
            "dinkster.random_noise",
            "dinkster.add_noise",
            "dinkster.sampler_custom",
            "dinkster.sampler_custom_advanced",
            "dinkster.compat.impact_regional_sampler",
            "dinkster.vae_decode",
            "dinkster.vae_decode_tiled",
            "dinkster.vae_encode",
            "dinkster.vae_encode_tiled",
            "dinkster.seedvr2_preprocess",
            "dinkster.seedvr2_postprocess",
            "dinkster.seedvr2_conditioning",
            "dinkster.seedvr2_temporal_chunk",
            "dinkster.seedvr2_temporal_merge",
            "dinkster.latent.combine",
            "dinkster.latent.mix",
            "dinkster.latent.multiply",
            "dinkster.latent.rotate",
            "dinkster.latent.flip",
            "dinkster.latent.crop",
            "dinkster.latent.resize",
            "dinkster.latent.resize_by",
            "dinkster.latent.composite",
            "dinkster.latent.composite_masked",
            "dinkster.latent.concat",
            "dinkster.latent.cut",
            "dinkster.latent.cut_to_batch",
            "dinkster.latent.from_batch",
            "dinkster.latent.repeat",
            "dinkster.latent.seed_behavior",
            "dinkster.latent.batch",
            "dinkster.latent.rebatch",
            "dinkster.latent.set_noise_mask",
            "dinkster.latent.replace_frames",
            "dinkster.latent.apply_operation",
            "dinkster.latent.operation_tonemap_reinhard",
            "dinkster.latent.operation_sharpen",
            "dinkster.latent.apply_operation_cfg",
            "dinkster.latent.generate_noise",
            "dinkster.latent.inject_noise",
        )
    )
    assert {node_type: schema.display_name for node_type, schema in schemas.items()} == {
        "dinkster.empty_trellis2_latent_structure": "Empty TRELLIS.2 Latent Structure",
        "dinkster.trellis2_conditioning": "TRELLIS.2 Conditioning",
        "dinkster.pixal3d_conditioning": "Pixal3D Conditioning",
        "dinkster.vae_decode_structure_trellis2": "Decode TRELLIS.2 Structure",
        "dinkster.trellis2_shape_stage": "TRELLIS.2 Shape Stage",
        "dinkster.trellis2_upsample_stage": "TRELLIS.2 Upsample Stage",
        "dinkster.vae_decode_shape_trellis": "Decode TRELLIS.2 Shape",
        "dinkster.trellis2_texture_stage": "TRELLIS.2 Texture Stage",
        "dinkster.vae_decode_texture_trellis": "Decode TRELLIS.2 Texture",
        "dinkster.load_geometry_model": "Load Geometry Model",
        "dinkster.estimate_geometry": "Estimate Geometry",
        "dinkster.geometry_to_fov": "Geometry to Field of View",
        "dinkster.load_background_removal": "Load Background Removal Model",
        "dinkster.remove_background": "Remove Background",
        "dinkster.image_crop_to_mask": "Crop Image to Mask",
        "dinkster.preview_mask": "Preview Mask",
        "dinkster.voxel_to_mesh": "Voxel to Mesh",
        "dinkster.get_mesh_info": "Get Mesh Info",
        "dinkster.remesh_mesh": "Remesh Mesh",
        "dinkster.decimate_mesh": "Decimate Mesh",
        "dinkster.smooth_mesh_normals": "Smooth Mesh Normals",
        "dinkster.unwrap_mesh": "Unwrap Mesh UVs",
        "dinkster.paint_mesh": "Paint Mesh",
        "dinkster.bake_texture_from_voxel": "Bake Texture from Voxel",
        "dinkster.bake_normal_map_from_mesh": "Bake Normal Map from Mesh",
        "dinkster.bake_ambient_occlusion": "Bake Ambient Occlusion",
        "dinkster.render_uv_atlas": "Render UV Atlas",
        "dinkster.apply_texture_to_mesh": "Apply Texture to Mesh",
        "dinkster.mesh_to_model3d": "Mesh to 3D Model",
        "dinkster.load_model_profile": "Load Model",
        "dinkster.load_checkpoint": "Load Checkpoint",
        "dinkster.load_controlnet": "Load ControlNet Model",
        "dinkster.apply_controlnet": "Apply ControlNet (DEPRECATED)",
        "dinkster.apply_controlnet_advanced": "Apply ControlNet",
        "dinkster.set_controlnet_union_type": "Set Union ControlNet Type",
        "dinkster.load_checkpoint_stack": "Load Checkpoint Stack",
        "dinkster.load_diffusion_model": "Load Diffusion Model",
        "dinkster.load_diffusion_components": "Load Diffusion Components",
        "dinkster.load_ltxav_text_encoder": "Load LTX-2 Text Encoder",
        "dinkster.load_ltxav_audio_vae": "Load LTX-2 Audio VAE",
        "dinkster.load_latent_upscale_model": "Load Latent Upscale Model",
        "dinkster.ltxav_audio_vae_decode": "LTX-2 Audio VAE Decode",
        "dinkster.load_lora": "Load LoRA (Model and CLIP)",
        "dinkster.load_lora_model_only": "Load LoRA",
        "dinkster.apply_lora_stack": "Apply LoRA Stack (Model and CLIP)",
        "dinkster.apply_lora_stack_model_only": "Apply LoRA Stack",
        "dinkster.clip_text_encode": "CLIP Text Encode",
        "dinkster.clip_text_encode_lumina2": "CLIP Text Encode (Lumina 2)",
        "dinkster.model_sampling_aura_flow": "ModelSamplingAuraFlow",
        "dinkster.text_generate": "Generate Text",
        "dinkster.prompt_enhance": "Enhance Prompt",
        "dinkster.clip_set_last_layer": "CLIP Set Last Layer",
        "dinkster.t5_tokenizer_options": "T5 Tokenizer Options",
        "dinkster.clip_text_encode_controlnet": "CLIP Text Encode (Controlnet)",
        "dinkster.flux_guidance": "FluxGuidance",
        "dinkster.flux_disable_guidance": "Flux Disable Guidance",
        "dinkster.reference_latent": "Set Reference Latent",
        "dinkster.cfg_zero_star": "CFGZeroStar",
        "dinkster.cfg_norm": "CFGNorm",
        "dinkster.tcfg": "Tangential Damping CFG",
        "dinkster.fresca": "FreSca",
        "dinkster.lazy_cache": "LazyCache",
        "dinkster.easy_cache": "EasyCache",
        "dinkster.attention_schedule": "Attention Schedule",
        "dinkster.context_windows_manual": "Context Windows (Manual)",
        "dinkster.wan_context_windows_manual": "WAN Context Windows (Manual)",
        "dinkster.ltxv_context_windows": "LTXV Context Windows",
        "dinkster.adaptive_projected_guidance": "Adaptive Projected Guidance",
        "dinkster.mahiro_guidance": "Positive-Biased Guidance",
        "dinkster.epsilon_scaling": "Epsilon Scaling",
        "dinkster.cfg_override": "CFG Override",
        "dinkster.rescale_cfg": "RescaleCFG",
        "dinkster.renorm_cfg": "RenormCFG",
        "dinkster.temporal_score_rescaling": "TSR - Temporal Score Rescaling",
        "dinkster.nag": "Normalized Attention Guidance",
        "dinkster.ltxav_conditioning": "LTX-2 AV Conditioning",
        "dinkster.ltxav_reference_audio": "LTX-2 Reference Audio",
        "dinkster.ltxav_id_lora_reference_audio": "LTXV Reference Audio (ID-LoRA)",
        "dinkster.ltxv_spatiotemporal_guidance": "LTXV Spatio-Temporal Guidance (STG)",
        "dinkster.ltxv_modality_guidance": "LTXV Modality Guidance (A/V Coupling)",
        "dinkster.ltxv_duration_predictor": "LTXV Duration Predictor",
        "dinkster.ltxv_dual_cfg_guider": "LTXV Dual CFG Guider",
        "dinkster.ltxv_conditioning": "LTX-Video Conditioning",
        "dinkster.ltxv_image_to_video": "LTX-Video Image to Video",
        "dinkster.ltxv_image_to_video_inplace": "LTX-Video Image to Video (In-place)",
        "dinkster.ltxv_add_guide": "LTX-Video Add Guide",
        "dinkster.ltxv_crop_guides": "LTX-Video Crop Guides",
        "dinkster.ltxv_latent_upsampler": "LTXV Latent Upsampler",
        "dinkster.conditioning_merge": "Conditioning Merge",
        "dinkster.conditioning_scale": "Conditioning Scale",
        "dinkster.conditioning_set_area": "Conditioning Set Area",
        "dinkster.conditioning_set_mask": "Conditioning Set Mask",
        "dinkster.conditioning_set_timestep_range": "Conditioning Set Timestep Range",
        "dinkster.conditioning_zero_out": "Conditioning Zero Out",
        "dinkster.chroma_radiance_options": "Chroma Radiance Options",
        "dinkster.chroma_model_sampling": "Chroma Model Sampling",
        "dinkster.model_sampling_sd3": "Model Sampling SD3",
        "dinkster.model_sampling_ltxv": "ModelSamplingLTXV",
        "dinkster.model_sampling_flux": "ModelSamplingFlux",
        "dinkster.empty_latent_image": "Empty Latent Image",
        "dinkster.empty_sd3_latent_image": "Empty SD3 Latent Image",
        "dinkster.empty_chroma_radiance_latent_image": "Empty Chroma Radiance Latent Image",
        "dinkster.empty_flux2_latent_image": "Empty Flux 2 Latent",
        "dinkster.empty_ltxav_latent": "Empty LTX-2 AV Latent",
        "dinkster.empty_ltxv_latent": "Empty LTX-Video Latent",
        "dinkster.ksampler": "KSampler",
        "dinkster.ksampler_advanced": "KSampler (Advanced)",
        "dinkster.ksampler_select": "KSamplerSelect",
        "dinkster.sampler_dpmpp_3m_sde": "SamplerDPMPP_3M_SDE",
        "dinkster.sampler_dpmpp_2m_sde": "SamplerDPMPP_2M_SDE",
        "dinkster.sampler_dpmpp_sde": "SamplerDPMPP_SDE",
        "dinkster.sampler_dpmpp_2s_ancestral": "SamplerDPMPP_2S_Ancestral",
        "dinkster.sampler_euler_ancestral": "SamplerEulerAncestral",
        "dinkster.sampler_euler_ancestral_cfg_pp": "SamplerEulerAncestralCFG++",
        "dinkster.sampler_lms": "SamplerLMS",
        "dinkster.sampler_dpm_adaptative": "SamplerDPMAdaptative",
        "dinkster.sampler_er_sde": "SamplerER_SDE",
        "dinkster.sampler_seeds_2": "SamplerSEEDS2",
        "dinkster.sampler_sa_solver": "SamplerSASolver",
        "dinkster.basic_scheduler": "BasicScheduler",
        "dinkster.beta_sampling_scheduler": "BetaSamplingScheduler",
        "dinkster.sd_turbo_scheduler": "SDTurboScheduler",
        "dinkster.karras_scheduler": "KarrasScheduler",
        "dinkster.exponential_scheduler": "ExponentialScheduler",
        "dinkster.polyexponential_scheduler": "PolyexponentialScheduler",
        "dinkster.laplace_scheduler": "LaplaceScheduler",
        "dinkster.vp_scheduler": "VPScheduler",
        "dinkster.align_your_steps_scheduler": "AlignYourStepsScheduler",
        "dinkster.gits_scheduler": "GITSScheduler",
        "dinkster.optimal_steps_scheduler": "OptimalStepsScheduler",
        "dinkster.flux2_scheduler": "Flux2Scheduler",
        "dinkster.ideogram4_scheduler": "Ideogram 4 Scheduler",
        "dinkster.manual_sigmas": "ManualSigmas",
        "dinkster.split_sigmas": "SplitSigmas",
        "dinkster.split_sigmas_denoise": "SplitSigmasDenoise",
        "dinkster.flip_sigmas": "FlipSigmas",
        "dinkster.set_first_sigma": "SetFirstSigma",
        "dinkster.extend_intermediate_sigmas": "ExtendIntermediateSigmas",
        "dinkster.sampling_percent_to_sigma": "SamplingPercentToSigma",
        "dinkster.basic_guider": "Basic Guider",
        "dinkster.cfg_guider": "CFG Guider",
        "dinkster.dual_cfg_guider": "Dual CFG Guider",
        "dinkster.dual_model_guider": "Dual Model CFG Guider",
        "dinkster.scheduled_cfg_guider": "Scheduled CFG Guider",
        "dinkster.perp_neg_guider": "Perp-Neg Guider",
        "dinkster.disable_cfg1_optimization": "Disable CFG 1 Optimization",
        "dinkster.disable_noise": "DisableNoise",
        "dinkster.random_noise": "RandomNoise",
        "dinkster.add_noise": "AddNoise",
        "dinkster.sampler_custom": "SamplerCustom",
        "dinkster.sampler_custom_advanced": "SamplerCustomAdvanced",
        "dinkster.compat.impact_regional_sampler": (
            "Impact Regional Sampler Compatibility Carrier"
        ),
        "dinkster.vae_decode": "VAE Decode",
        "dinkster.vae_decode_tiled": "VAE Decode (Tiled)",
        "dinkster.vae_encode": "VAE Encode",
        "dinkster.vae_encode_tiled": "VAE Encode (Tiled)",
        "dinkster.seedvr2_preprocess": "Pre-Process SeedVR2 Input",
        "dinkster.seedvr2_postprocess": "Post-Process SeedVR2 Output",
        "dinkster.seedvr2_conditioning": "Apply SeedVR2 Conditioning",
        "dinkster.seedvr2_temporal_chunk": "Split SeedVR2 Latent",
        "dinkster.seedvr2_temporal_merge": "Merge SeedVR2 Latents",
        "dinkster.latent.combine": "Combine Latents",
        "dinkster.latent.mix": "Mix Latents",
        "dinkster.latent.multiply": "Multiply Latent",
        "dinkster.latent.rotate": "Rotate Latent",
        "dinkster.latent.flip": "Flip Latent",
        "dinkster.latent.crop": "Crop Latent",
        "dinkster.latent.resize": "Resize Latent",
        "dinkster.latent.resize_by": "Resize Latent By",
        "dinkster.latent.composite": "Composite Latents",
        "dinkster.latent.composite_masked": "Composite Latents (Masked)",
        "dinkster.latent.concat": "Concat Latents",
        "dinkster.latent.cut": "Cut Latent",
        "dinkster.latent.cut_to_batch": "Cut Latent to Batch",
        "dinkster.latent.from_batch": "Latent From Batch",
        "dinkster.latent.repeat": "Repeat Latent Batch",
        "dinkster.latent.seed_behavior": "Latent Batch Seed Behavior",
        "dinkster.latent.batch": "Batch Latents",
        "dinkster.latent.rebatch": "Rebatch Latents",
        "dinkster.latent.set_noise_mask": "Set Latent Noise Mask",
        "dinkster.latent.replace_frames": "Replace Video Latent Frames",
        "dinkster.latent.apply_operation": "Latent Apply Operation",
        "dinkster.latent.operation_tonemap_reinhard": "Latent Operation Tonemap Reinhard",
        "dinkster.latent.operation_sharpen": "Latent Operation Sharpen",
        "dinkster.latent.apply_operation_cfg": "Latent Apply Operation CFG",
        "dinkster.latent.generate_noise": "Generate Noise",
        "dinkster.latent.inject_noise": "Inject Noise To Latent",
    }
    for node_type, comfy_node_id in {
        "dinkster.load_latent_upscale_model": "LatentUpscaleModelLoader",
        "dinkster.ltxv_spatiotemporal_guidance": "LTXVSpatioTemporalGuidance",
        "dinkster.ltxv_modality_guidance": "LTXVModalityGuidance",
        "dinkster.ltxv_duration_predictor": "LTXVDurationPredictor",
        "dinkster.ltxv_dual_cfg_guider": "LTXVDualCFGGuider",
        "dinkster.dual_cfg_guider": "DualCFGGuider",
        "dinkster.add_noise": "AddNoise",
        "dinkster.empty_trellis2_latent_structure": "EmptyTrellis2LatentStructure",
        "dinkster.trellis2_conditioning": "Trellis2Conditioning",
        "dinkster.pixal3d_conditioning": "Pixal3DConditioning",
        "dinkster.vae_decode_structure_trellis2": "VaeDecodeStructureTrellis2",
        "dinkster.trellis2_shape_stage": "Trellis2ShapeStage",
        "dinkster.trellis2_upsample_stage": "Trellis2UpsampleStage",
        "dinkster.vae_decode_shape_trellis": "VaeDecodeShapeTrellis",
        "dinkster.trellis2_texture_stage": "Trellis2TextureStage",
        "dinkster.vae_decode_texture_trellis": "VaeDecodeTextureTrellis",
        "dinkster.cfg_override": "CFGOverride",
        "dinkster.model_sampling_sd3": "ModelSamplingSD3",
        "dinkster.model_sampling_ltxv": "ModelSamplingLTXV",
        "dinkster.model_sampling_flux": "ModelSamplingFlux",
        "dinkster.empty_latent_image": "EmptyLatentImage",
        "dinkster.ltxv_conditioning": "LTXVConditioning",
        "dinkster.ltxv_image_to_video": "LTXVImgToVideo",
        "dinkster.ltxv_image_to_video_inplace": "LTXVImgToVideoInplace",
        "dinkster.ltxv_add_guide": "LTXVAddGuide",
        "dinkster.ltxv_crop_guides": "LTXVCropGuides",
        "dinkster.ltxv_latent_upsampler": "LTXVLatentUpsampler",
        "dinkster.ksampler_select": "KSamplerSelect",
        "dinkster.sampler_dpmpp_3m_sde": "SamplerDPMPP_3M_SDE",
        "dinkster.sampler_dpmpp_2m_sde": "SamplerDPMPP_2M_SDE",
        "dinkster.sampler_dpmpp_sde": "SamplerDPMPP_SDE",
        "dinkster.sampler_dpmpp_2s_ancestral": "SamplerDPMPP_2S_Ancestral",
        "dinkster.sampler_euler_ancestral": "SamplerEulerAncestral",
        "dinkster.sampler_euler_ancestral_cfg_pp": "SamplerEulerAncestralCFGPP",
        "dinkster.sampler_lms": "SamplerLMS",
        "dinkster.sampler_dpm_adaptative": "SamplerDPMAdaptative",
        "dinkster.sampler_er_sde": "SamplerER_SDE",
        "dinkster.sampler_seeds_2": "SamplerSEEDS2",
        "dinkster.basic_scheduler": "BasicScheduler",
        "dinkster.beta_sampling_scheduler": "BetaSamplingScheduler",
        "dinkster.sd_turbo_scheduler": "SDTurboScheduler",
        "dinkster.karras_scheduler": "KarrasScheduler",
        "dinkster.exponential_scheduler": "ExponentialScheduler",
        "dinkster.polyexponential_scheduler": "PolyexponentialScheduler",
        "dinkster.laplace_scheduler": "LaplaceScheduler",
        "dinkster.vp_scheduler": "VPScheduler",
        "dinkster.align_your_steps_scheduler": "AlignYourStepsScheduler",
        "dinkster.gits_scheduler": "GITSScheduler",
        "dinkster.optimal_steps_scheduler": "OptimalStepsScheduler",
        "dinkster.manual_sigmas": "ManualSigmas",
        "dinkster.split_sigmas": "SplitSigmas",
        "dinkster.split_sigmas_denoise": "SplitSigmasDenoise",
        "dinkster.flip_sigmas": "FlipSigmas",
        "dinkster.set_first_sigma": "SetFirstSigma",
        "dinkster.extend_intermediate_sigmas": "ExtendIntermediateSigmas",
        "dinkster.sampling_percent_to_sigma": "SamplingPercentToSigma",
        "dinkster.basic_guider": "BasicGuider",
        "dinkster.cfg_guider": "CFGGuider",
        "dinkster.perp_neg_guider": "PerpNegGuider",
        "dinkster.disable_cfg1_optimization": "DisableModelCfg1Optimization",
        "dinkster.disable_noise": "DisableNoise",
        "dinkster.random_noise": "RandomNoise",
        "dinkster.sampler_custom": "SamplerCustom",
        "dinkster.sampler_custom_advanced": "SamplerCustomAdvanced",
        "dinkster.vae_decode_tiled": "VAEDecodeTiled",
        "dinkster.latent.generate_noise": "GenerateNoise",
        "dinkster.latent.inject_noise": "InjectNoiseToLatent",
    }.items():
        assert schemas[node_type].aliases == (comfy_node_id,)
    assert {
        node_type: (
            {item.id: item.type.types for item in schema.inputs},
            {item.id: item.type.types for item in schema.outputs},
        )
        for node_type, schema in schemas.items()
    } == {
        "dinkster.empty_trellis2_latent_structure": (
            {"batch_size": ("core.int",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.trellis2_conditioning": (
            {
                "clip_vision_model": ("dinkster.clip-vision",),
                "image": ("dinkster.image",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.pixal3d_conditioning": (
            {
                "clip_vision_model": ("dinkster.clip-vision",),
                "image": ("dinkster.image",),
                "camera_angle_x": ("core.float",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.vae_decode_structure_trellis2": (
            {
                "samples": ("dinkster.latent",),
                "vae": ("dinkster.vae",),
                "resolution": ("core.combo",),
            },
            {"voxel": ("comfy.VOXEL",)},
        ),
        "dinkster.trellis2_shape_stage": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "voxel": ("comfy.VOXEL",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.trellis2_upsample_stage": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "shape_latent": ("dinkster.latent",),
                "vae": ("dinkster.vae",),
                "target_resolution": ("core.int",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.vae_decode_shape_trellis": (
            {"samples": ("dinkster.latent",), "vae": ("dinkster.vae",)},
            {
                "mesh": ("comfy.MESH",),
                "shape_subdivides": ("comfy.SHAPE_SUBDIVIDES",),
            },
        ),
        "dinkster.trellis2_texture_stage": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "shape_latent": ("dinkster.latent",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.vae_decode_texture_trellis": (
            {
                "samples": ("dinkster.latent",),
                "vae": ("dinkster.vae",),
                "shape_subdivides": ("comfy.SHAPE_SUBDIVIDES",),
            },
            {"voxel_colors": ("comfy.VOXEL",)},
        ),
        "dinkster.load_geometry_model": (
            {"model": ("dinkster.asset",)},
            {"model": ("comfy.MOGE_MODEL",)},
        ),
        "dinkster.estimate_geometry": (
            {
                "model": ("comfy.MOGE_MODEL",),
                "image": ("dinkster.image",),
                "resolution_level": ("core.int",),
                "fov_x_degrees": ("core.float",),
                "batch_size": ("core.int",),
                "force_projection": ("core.boolean",),
                "apply_mask": ("core.boolean",),
            },
            {"geometry": ("comfy.MOGE_GEOMETRY",)},
        ),
        "dinkster.geometry_to_fov": (
            {
                "geometry": ("comfy.MOGE_GEOMETRY",),
                "axis": ("core.combo",),
                "unit": ("core.combo",),
            },
            {"fov": ("core.float",), "focal_pixels": ("core.float",)},
        ),
        "dinkster.load_background_removal": (
            {"model": ("dinkster.asset",)},
            {"model": ("comfy.BACKGROUND_REMOVAL",)},
        ),
        "dinkster.remove_background": (
            {
                "model": ("comfy.BACKGROUND_REMOVAL",),
                "image": ("dinkster.image",),
            },
            {"mask": ("dinkster.mask",)},
        ),
        "dinkster.image_crop_to_mask": (
            {
                "images": ("dinkster.image",),
                "masks": ("dinkster.mask",),
                "width": ("core.int",),
                "height": ("core.int",),
                "pad_factor": ("core.float",),
                "grow_mask": ("core.int",),
                "background": ("core.string",),
            },
            {"images": ("dinkster.image",)},
        ),
        "dinkster.preview_mask": (
            {"mask": ("dinkster.mask",)},
            {"mask": ("dinkster.mask",)},
        ),
        "dinkster.voxel_to_mesh": (
            {
                "voxel": ("comfy.VOXEL",),
                "algorithm": ("core.combo",),
                "threshold": ("core.float",),
            },
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.get_mesh_info": (
            {"mesh": ("comfy.MESH",)},
            {"mesh": ("comfy.MESH",), "info": ("core.string",)},
        ),
        "dinkster.remesh_mesh": (
            {
                "mesh": ("comfy.MESH",),
                "resolution": ("core.int",),
                "sign_mode": ("core.combo",),
                "qef": ("core.boolean",),
                "drop_inverted_components": ("core.boolean",),
                "drop_enclosed_components": ("core.boolean",),
                "manifold": ("core.boolean",),
                "band": ("core.float",),
                "project_back": ("core.float",),
                "fix_poles": ("core.boolean",),
                "smooth_iters": ("core.int",),
                "drop_small_components": ("core.float",),
                "precluster_max_verts": ("core.int",),
            },
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.decimate_mesh": (
            {
                "mesh": ("comfy.MESH",),
                "target_face_count": ("core.int",),
                "placement_mode": ("core.combo",),
            },
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.smooth_mesh_normals": (
            {"mesh": ("comfy.MESH",), "crease_angle": ("core.float",)},
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.unwrap_mesh": (
            {
                "mesh": ("comfy.MESH",),
                "segmenter": ("core.combo",),
                "resolution": ("core.int",),
                "padding": ("core.int",),
                "weld_distance": ("core.float",),
            },
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.paint_mesh": (
            {"mesh": ("comfy.MESH",), "voxel_colors": ("comfy.VOXEL",)},
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.bake_texture_from_voxel": (
            {
                "mesh": ("comfy.MESH",),
                "voxel_colors": ("comfy.VOXEL",),
                "texture_size": ("core.int",),
                "reference_mesh": ("comfy.MESH",),
            },
            {
                "base_color": ("dinkster.image",),
                "metallic": ("dinkster.image",),
                "roughness": ("dinkster.image",),
            },
        ),
        "dinkster.bake_normal_map_from_mesh": (
            {
                "low_poly": ("comfy.MESH",),
                "high_poly": ("comfy.MESH",),
                "resolution": ("core.int",),
                "cage_distance": ("core.float",),
                "ignore_backfaces": ("core.boolean",),
            },
            {"normal_map": ("dinkster.image",)},
        ),
        "dinkster.bake_ambient_occlusion": (
            {
                "low_poly": ("comfy.MESH",),
                "high_poly": ("comfy.MESH",),
                "resolution": ("core.int",),
                "samples": ("core.int",),
                "max_distance": ("core.float",),
                "strength": ("core.float",),
                "bias": ("core.float",),
            },
            {"occlusion": ("dinkster.image",)},
        ),
        "dinkster.render_uv_atlas": (
            {"mesh": ("comfy.MESH",), "resolution": ("core.int",)},
            {"image": ("dinkster.image",)},
        ),
        "dinkster.apply_texture_to_mesh": (
            {
                "mesh": ("comfy.MESH",),
                "base_color": ("dinkster.image",),
                "metallic": ("dinkster.image",),
                "roughness": ("dinkster.image",),
                "occlusion": ("dinkster.image",),
                "normal_map": ("dinkster.image",),
            },
            {"mesh": ("comfy.MESH",)},
        ),
        "dinkster.mesh_to_model3d": (
            {"mesh": ("comfy.MESH",)},
            {"model": ("dinkster.model3d",)},
        ),
        "dinkster.load_model_profile": (
            {"checkpoint": ("dinkster.asset",), "entries": ("core.string",)},
            {},
        ),
        "dinkster.load_checkpoint": (
            {"checkpoint": ("dinkster.asset",)},
            {
                "model": ("dinkster.model",),
                "clip": ("dinkster.clip",),
                "vae": ("dinkster.vae",),
            },
        ),
        "dinkster.load_controlnet": (
            {"control_net_name": ("dinkster.asset",)},
            {"control_net": ("comfy.CONTROL_NET",)},
        ),
        "dinkster.apply_controlnet": (
            {
                "conditioning": ("dinkster.conditioning",),
                "control_net": ("comfy.CONTROL_NET",),
                "image": ("dinkster.image",),
                "strength": ("core.float",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.apply_controlnet_advanced": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "control_net": ("comfy.CONTROL_NET",),
                "image": ("dinkster.image",),
                "strength": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
                "vae": ("dinkster.vae",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.set_controlnet_union_type": (
            {
                "control_net": ("comfy.CONTROL_NET",),
                "type": ("core.combo",),
            },
            {"control_net": ("comfy.CONTROL_NET",)},
        ),
        "dinkster.load_checkpoint_stack": (
            {
                "checkpoint": ("dinkster.asset",),
                "stop_at_clip_layer": ("core.int",),
                "execution_mode": ("core.combo",),
            },
            {
                "model": ("dinkster.model",),
                "clip": ("dinkster.clip",),
                "vae": ("dinkster.vae",),
            },
        ),
        "dinkster.load_diffusion_model": (
            {
                "diffusion_model": ("dinkster.asset",),
                "weight_dtype": ("core.combo",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.load_diffusion_components": (
            {"weight_dtype": ("core.combo",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.load_ltxav_text_encoder": (
            {
                "text_encoder": ("dinkster.asset",),
                "ckpt_name": ("dinkster.asset",),
                "device": ("core.combo",),
            },
            {"clip": ("dinkster.clip",)},
        ),
        "dinkster.load_ltxav_audio_vae": (
            {"ckpt_name": ("dinkster.asset",)},
            {"audio_vae": ("dinkster.vae",)},
        ),
        "dinkster.load_latent_upscale_model": (
            {"model_name": ("dinkster.asset",)},
            {"upscale_model": ("comfy.LATENT_UPSCALE_MODEL",)},
        ),
        "dinkster.ltxav_audio_vae_decode": (
            {
                "samples": ("dinkster.latent",),
                "audio_vae": ("dinkster.vae",),
            },
            {"audio": ("comfy.AUDIO",)},
        ),
        "dinkster.load_lora": (
            {
                "model": ("dinkster.model",),
                "clip": ("dinkster.clip",),
                "lora": ("dinkster.asset",),
                "strength_model": ("core.float",),
                "strength_clip": ("core.float",),
                "execution_mode": ("core.combo",),
            },
            {"model": ("dinkster.model",), "clip": ("dinkster.clip",)},
        ),
        "dinkster.load_lora_model_only": (
            {
                "model": ("dinkster.model",),
                "lora": ("dinkster.asset",),
                "strength_model": ("core.float",),
                "execution_mode": ("core.combo",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.apply_lora_stack": (
            {
                "model": ("dinkster.model",),
                "clip": ("dinkster.clip",),
                "execution_mode": ("core.combo",),
            },
            {"model": ("dinkster.model",), "clip": ("dinkster.clip",)},
        ),
        "dinkster.apply_lora_stack_model_only": (
            {
                "model": ("dinkster.model",),
                "execution_mode": ("core.combo",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.clip_text_encode": (
            {"text": ("core.string",), "clip": ("dinkster.clip",)},
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.clip_text_encode_lumina2": (
            {
                "system_prompt": ("core.combo",),
                "user_prompt": ("core.string",),
                "clip": ("dinkster.clip",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.model_sampling_aura_flow": (
            {"model": ("dinkster.model",), "shift": ("core.float",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.text_generate": (
            {
                "clip": ("dinkster.clip",),
                "provider": ("core.combo",),
                "prompt": ("core.string",),
                "image": ("dinkster.image",),
                "video": ("dinkster.image",),
                "audio": ("comfy.AUDIO",),
                "max_length": ("core.int",),
                "thinking": ("core.boolean",),
                "use_default_template": ("core.boolean",),
            },
            {"generated_text": ("core.string",)},
        ),
        "dinkster.prompt_enhance": (
            {
                "clip": ("dinkster.clip",),
                "provider": ("core.combo",),
                "prompt": ("core.string",),
                "image": ("dinkster.image",),
                "video": ("dinkster.image",),
                "audio": ("comfy.AUDIO",),
                "max_length": ("core.int",),
                "thinking": ("core.boolean",),
                "use_default_template": ("core.boolean",),
            },
            {"generated_text": ("core.string",)},
        ),
        "dinkster.clip_set_last_layer": (
            {"clip": ("dinkster.clip",), "stop_at_clip_layer": ("core.int",)},
            {"clip": ("dinkster.clip",)},
        ),
        "dinkster.t5_tokenizer_options": (
            {
                "clip": ("dinkster.clip",),
                "min_padding": ("core.int",),
                "min_length": ("core.int",),
            },
            {"clip": ("dinkster.clip",)},
        ),
        "dinkster.clip_text_encode_controlnet": (
            {
                "clip": ("dinkster.clip",),
                "conditioning": ("dinkster.conditioning",),
                "text": ("core.string",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.flux_guidance": (
            {"conditioning": ("dinkster.conditioning",), "guidance": ("core.float",)},
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.flux_disable_guidance": (
            {"conditioning": ("dinkster.conditioning",)},
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.reference_latent": (
            {
                "conditioning": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.cfg_zero_star": (
            {"model": ("dinkster.model",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.cfg_norm": (
            {
                "model": ("dinkster.model",),
                "strength": ("core.float",),
                "pre_cfg": ("core.boolean",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.tcfg": (
            {"model": ("dinkster.model",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.fresca": (
            {
                "model": ("dinkster.model",),
                "scale_low": ("core.float",),
                "scale_high": ("core.float",),
                "freq_cutoff": ("core.int",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.lazy_cache": (
            {
                "model": ("dinkster.model",),
                "reuse_threshold": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.easy_cache": (
            {
                "model": ("dinkster.model",),
                "reuse_threshold": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.attention_schedule": (
            {
                "model": ("dinkster.model",),
                "approximate_provider": ("core.combo",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
                "conditioning_sink": ("core.combo",),
                "sol_tau": ("dinkster.curve",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.context_windows_manual": (
            {
                "model": ("dinkster.model",),
                "context_length": ("core.int",),
                "context_overlap": ("core.int",),
                "context_schedule": ("core.combo",),
                "context_stride": ("core.int",),
                "closed_loop": ("core.boolean",),
                "fuse_method": ("core.combo",),
                "dim": ("core.int",),
                "freenoise": ("core.boolean",),
                "causal_window_fix": ("core.boolean",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.wan_context_windows_manual": (
            {
                "model": ("dinkster.model",),
                "context_length": ("core.int",),
                "context_overlap": ("core.int",),
                "context_schedule": ("core.combo",),
                "context_stride": ("core.int",),
                "closed_loop": ("core.boolean",),
                "fuse_method": ("core.combo",),
                "freenoise": ("core.boolean",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.ltxv_context_windows": (
            {
                "model": ("dinkster.model",),
                "context_length": ("core.int",),
                "context_overlap": ("core.int",),
                "context_schedule": ("core.combo",),
                "context_stride": ("core.int",),
                "closed_loop": ("core.boolean",),
                "fuse_method": ("core.combo",),
                "freenoise": ("core.boolean",),
                "retain_first_frame": ("core.boolean",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.adaptive_projected_guidance": (
            {
                "model": ("dinkster.model",),
                "eta": ("core.float",),
                "norm_threshold": ("core.float",),
                "momentum": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.mahiro_guidance": (
            {"model": ("dinkster.model",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.epsilon_scaling": (
            {"model": ("dinkster.model",), "scaling_factor": ("core.float",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.cfg_override": (
            {
                "model": ("dinkster.model",),
                "cfg": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.rescale_cfg": (
            {"model": ("dinkster.model",), "multiplier": ("core.float",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.renorm_cfg": (
            {
                "model": ("dinkster.model",),
                "cfg_trunc": ("core.float",),
                "renorm_cfg": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.temporal_score_rescaling": (
            {
                "model": ("dinkster.model",),
                "tsr_k": ("core.float",),
                "tsr_sigma": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.nag": (
            {
                "model": ("dinkster.model",),
                "nag_scale": ("core.float",),
                "nag_alpha": ("core.float",),
                "nag_tau": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.ltxav_conditioning": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "frame_rate": ("core.float",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.ltxav_reference_audio": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "reference_audio": ("comfy.AUDIO",),
                "audio_vae": ("dinkster.vae",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.ltxav_id_lora_reference_audio": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "reference_audio": ("comfy.AUDIO",),
                "audio_vae": ("dinkster.vae",),
                "identity_guidance_scale": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.ltxv_spatiotemporal_guidance": (
            {
                "model": ("dinkster.model",),
                "scale": ("core.float",),
                "blocks": ("core.string",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.ltxv_modality_guidance": (
            {
                "model": ("dinkster.model",),
                "modality_scale": ("core.float",),
                "start_percent": ("core.float",),
                "end_percent": ("core.float",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.ltxv_duration_predictor": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "duration_head": ("comfy.MODEL_PATCH",),
                "frame_rate": ("core.float",),
                "min_seconds": ("core.float",),
                "max_seconds": ("core.float",),
            },
            {"num_frames": ("core.int",), "seconds": ("core.float",)},
        ),
        "dinkster.ltxv_dual_cfg_guider": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "video_cfg": ("core.float",),
                "audio_cfg": ("core.float",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.ltxv_conditioning": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "frame_rate": ("core.float",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.ltxv_image_to_video": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "vae": ("dinkster.vae",),
                "image": ("dinkster.image",),
                "width": ("core.int",),
                "height": ("core.int",),
                "length": ("core.int",),
                "batch_size": ("core.int",),
                "strength": ("core.float",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.ltxv_image_to_video_inplace": (
            {
                "vae": ("dinkster.vae",),
                "image": ("dinkster.image",),
                "latent": ("dinkster.latent",),
                "strength": ("core.float",),
                "bypass": ("core.boolean",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.ltxv_add_guide": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "vae": ("dinkster.vae",),
                "latent": ("dinkster.latent",),
                "image": ("dinkster.image",),
                "frame_idx": ("core.int",),
                "strength": ("core.float",),
                "attention_mask": ("dinkster.mask",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.ltxv_crop_guides": (
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent": ("dinkster.latent",),
            },
        ),
        "dinkster.ltxv_latent_upsampler": (
            {
                "samples": ("dinkster.latent",),
                "upscale_model": ("comfy.LATENT_UPSCALE_MODEL",),
                "vae": ("dinkster.vae",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.conditioning_merge": (
            {},
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.conditioning_scale": (
            {
                "conditioning": ("dinkster.conditioning",),
                "multiplier": ("core.float",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.conditioning_set_area": (
            {
                "conditioning": ("dinkster.conditioning",),
                "strength": ("core.float",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.conditioning_set_mask": (
            {
                "conditioning": ("dinkster.conditioning",),
                "mask": ("dinkster.mask",),
                "strength": ("core.float",),
                "set_cond_area": ("core.combo",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.conditioning_set_timestep_range": (
            {
                "conditioning": ("dinkster.conditioning",),
                "start": ("core.float",),
                "end": ("core.float",),
            },
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.conditioning_zero_out": (
            {"conditioning": ("dinkster.conditioning",)},
            {"conditioning": ("dinkster.conditioning",)},
        ),
        "dinkster.chroma_radiance_options": (
            {
                "model": ("dinkster.model",),
                "preserve_wrapper": ("core.boolean",),
                "start_sigma": ("core.float",),
                "end_sigma": ("core.float",),
                "nerf_tile_size": ("core.int",),
                "force_sequential_txt_ids": ("core.boolean",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.chroma_model_sampling": (
            {"model": ("dinkster.model",), "shift": ("core.float",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.model_sampling_sd3": (
            {"model": ("dinkster.model",), "shift": ("core.float",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.model_sampling_ltxv": (
            {
                "model": ("dinkster.model",),
                "max_shift": ("core.float",),
                "base_shift": ("core.float",),
                "latent": ("dinkster.latent",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.model_sampling_flux": (
            {
                "model": ("dinkster.model",),
                "max_shift": ("core.float",),
                "base_shift": ("core.float",),
                "width": ("core.int",),
                "height": ("core.int",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.empty_latent_image": (
            {
                "width": ("core.int",),
                "height": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.empty_sd3_latent_image": (
            {
                "width": ("core.int",),
                "height": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.empty_chroma_radiance_latent_image": (
            {
                "width": ("core.int",),
                "height": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.empty_flux2_latent_image": (
            {
                "width": ("core.int",),
                "height": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.empty_ltxav_latent": (
            {
                "model": ("dinkster.model",),
                "width": ("core.int",),
                "height": ("core.int",),
                "length": ("core.int",),
                "frame_rate": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.empty_ltxv_latent": (
            {
                "model": ("dinkster.model",),
                "width": ("core.int",),
                "height": ("core.int",),
                "length": ("core.int",),
                "batch_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.ksampler": (
            {
                "model": ("dinkster.model",),
                "seed": ("core.int",),
                "steps": ("core.int",),
                "cfg": ("core.float",),
                "sampler_name": ("core.combo",),
                "scheduler": ("core.combo",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent_image": ("dinkster.latent",),
                "denoise": ("core.float",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.ksampler_advanced": (
            {
                "model": ("dinkster.model",),
                "add_noise": ("core.combo",),
                "noise_seed": ("core.int",),
                "steps": ("core.int",),
                "cfg": ("core.float",),
                "sampler_name": ("core.combo",),
                "scheduler": ("core.combo",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "latent_image": ("dinkster.latent",),
                "start_at_step": ("core.int",),
                "end_at_step": ("core.int",),
                "return_with_leftover_noise": ("core.combo",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.ksampler_select": (
            {"sampler_name": ("core.combo",)},
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_dpmpp_3m_sde": (
            {
                "eta": ("core.float",),
                "s_noise": ("core.float",),
                "noise_device": ("core.combo",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_dpmpp_2m_sde": (
            {
                "solver_type": ("core.combo",),
                "eta": ("core.float",),
                "s_noise": ("core.float",),
                "noise_device": ("core.combo",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_dpmpp_sde": (
            {
                "eta": ("core.float",),
                "s_noise": ("core.float",),
                "r": ("core.float",),
                "noise_device": ("core.combo",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_dpmpp_2s_ancestral": (
            {"eta": ("core.float",), "s_noise": ("core.float",)},
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_euler_ancestral": (
            {"eta": ("core.float",), "s_noise": ("core.float",)},
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_euler_ancestral_cfg_pp": (
            {"eta": ("core.float",), "s_noise": ("core.float",)},
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_lms": (
            {"order": ("core.int",)},
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_dpm_adaptative": (
            {
                "order": ("core.int",),
                "rtol": ("core.float",),
                "atol": ("core.float",),
                "h_init": ("core.float",),
                "pcoeff": ("core.float",),
                "icoeff": ("core.float",),
                "dcoeff": ("core.float",),
                "accept_safety": ("core.float",),
                "eta": ("core.float",),
                "s_noise": ("core.float",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_er_sde": (
            {
                "solver_type": ("core.combo",),
                "max_stage": ("core.int",),
                "eta": ("core.float",),
                "s_noise": ("core.float",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_seeds_2": (
            {
                "solver_type": ("core.combo",),
                "eta": ("core.float",),
                "s_noise": ("core.float",),
                "r": ("core.float",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.sampler_sa_solver": (
            {
                "model": ("dinkster.model",),
                "eta": ("core.float",),
                "sde_start_percent": ("core.float",),
                "sde_end_percent": ("core.float",),
                "s_noise": ("core.float",),
                "predictor_order": ("core.int",),
                "corrector_order": ("core.int",),
                "use_pece": ("core.boolean",),
                "simple_order_2": ("core.boolean",),
            },
            {"sampler": ("dinkster.sampler",)},
        ),
        "dinkster.basic_scheduler": (
            {
                "model": ("dinkster.model",),
                "scheduler": ("core.combo",),
                "steps": ("core.int",),
                "denoise": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.beta_sampling_scheduler": (
            {
                "model": ("dinkster.model",),
                "steps": ("core.int",),
                "alpha": ("core.float",),
                "beta": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.sd_turbo_scheduler": (
            {
                "model": ("dinkster.model",),
                "steps": ("core.int",),
                "denoise": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.karras_scheduler": (
            {
                "steps": ("core.int",),
                "sigma_max": ("core.float",),
                "sigma_min": ("core.float",),
                "rho": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.exponential_scheduler": (
            {
                "steps": ("core.int",),
                "sigma_max": ("core.float",),
                "sigma_min": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.polyexponential_scheduler": (
            {
                "steps": ("core.int",),
                "sigma_max": ("core.float",),
                "sigma_min": ("core.float",),
                "rho": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.laplace_scheduler": (
            {
                "steps": ("core.int",),
                "sigma_max": ("core.float",),
                "sigma_min": ("core.float",),
                "mu": ("core.float",),
                "beta": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.vp_scheduler": (
            {
                "steps": ("core.int",),
                "beta_d": ("core.float",),
                "beta_min": ("core.float",),
                "eps_s": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.align_your_steps_scheduler": (
            {
                "model_type": ("core.combo",),
                "steps": ("core.int",),
                "denoise": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.gits_scheduler": (
            {
                "coeff": ("core.float",),
                "steps": ("core.int",),
                "denoise": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.optimal_steps_scheduler": (
            {
                "model_type": ("core.combo",),
                "steps": ("core.int",),
                "denoise": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.flux2_scheduler": (
            {
                "steps": ("core.int",),
                "width": ("core.int",),
                "height": ("core.int",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.ideogram4_scheduler": (
            {
                "steps": ("core.int",),
                "width": ("core.int",),
                "height": ("core.int",),
                "mu": ("core.float",),
                "std": ("core.float",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.manual_sigmas": (
            {"sigmas": ("core.string",)},
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.split_sigmas": (
            {"sigmas": ("dinkster.sigmas",), "step": ("core.int",)},
            {
                "high_sigmas": ("dinkster.sigmas",),
                "low_sigmas": ("dinkster.sigmas",),
            },
        ),
        "dinkster.split_sigmas_denoise": (
            {"sigmas": ("dinkster.sigmas",), "denoise": ("core.float",)},
            {
                "high_sigmas": ("dinkster.sigmas",),
                "low_sigmas": ("dinkster.sigmas",),
            },
        ),
        "dinkster.flip_sigmas": (
            {"sigmas": ("dinkster.sigmas",)},
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.set_first_sigma": (
            {"sigmas": ("dinkster.sigmas",), "sigma": ("core.float",)},
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.extend_intermediate_sigmas": (
            {
                "sigmas": ("dinkster.sigmas",),
                "steps": ("core.int",),
                "start_at_sigma": ("core.float",),
                "end_at_sigma": ("core.float",),
                "spacing": ("core.combo",),
            },
            {"sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.sampling_percent_to_sigma": (
            {
                "model": ("dinkster.model",),
                "sampling_percent": ("core.float",),
                "return_actual_sigma": ("core.boolean",),
            },
            {"sigma_value": ("core.float",)},
        ),
        "dinkster.basic_guider": (
            {
                "model": ("dinkster.model",),
                "conditioning": ("dinkster.conditioning",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.cfg_guider": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "cfg": ("core.float",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.dual_cfg_guider": (
            {
                "model": ("dinkster.model",),
                "cond1": ("dinkster.conditioning",),
                "cond2": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "cfg_conds": ("core.float",),
                "cfg_cond2_negative": ("core.float",),
                "style": ("core.combo",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.dual_model_guider": (
            {
                "model": ("dinkster.model",),
                "model_negative": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "cfg": ("core.float",),
                "negative": ("dinkster.conditioning",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.scheduled_cfg_guider": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "sigmas": ("dinkster.sigmas",),
                "from_cfg": ("core.float",),
                "to_cfg": ("core.float",),
                "schedule": ("core.combo",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",), "sigmas": ("dinkster.sigmas",)},
        ),
        "dinkster.perp_neg_guider": (
            {
                "model": ("dinkster.model",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "empty_conditioning": ("dinkster.conditioning",),
                "cfg": ("core.float",),
                "neg_scale": ("core.float",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {"guider": ("dinkster.guider",)},
        ),
        "dinkster.disable_cfg1_optimization": (
            {"model": ("dinkster.model",)},
            {"model": ("dinkster.model",)},
        ),
        "dinkster.disable_noise": ({}, {"noise": ("dinkster.noise",)}),
        "dinkster.random_noise": (
            {"noise_seed": ("core.int",)},
            {"noise": ("dinkster.noise",)},
        ),
        "dinkster.add_noise": (
            {
                "model": ("dinkster.model",),
                "noise": ("dinkster.noise",),
                "sigmas": ("dinkster.sigmas",),
                "latent_image": ("dinkster.latent",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.sampler_custom": (
            {
                "model": ("dinkster.model",),
                "add_noise": ("core.boolean",),
                "noise_seed": ("core.int",),
                "cfg": ("core.float",),
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
                "sampler": ("dinkster.sampler",),
                "sigmas": ("dinkster.sigmas",),
                "latent_image": ("dinkster.latent",),
                "conditioning_batching": ("core.combo",),
                "max_fused_lanes": ("core.int",),
            },
            {
                "output": ("dinkster.latent",),
                "denoised_output": ("dinkster.latent",),
            },
        ),
        "dinkster.sampler_custom_advanced": (
            {
                "noise": ("dinkster.noise",),
                "guider": ("dinkster.guider",),
                "sampler": ("dinkster.sampler",),
                "sigmas": ("dinkster.sigmas",),
                "latent_image": ("dinkster.latent",),
            },
            {
                "output": ("dinkster.latent",),
                "denoised_output": ("dinkster.latent",),
            },
        ),
        "dinkster.compat.impact_regional_sampler": (
            {
                "base_basic_pipe": ("comfy.BASIC_PIPE",),
                "region_basic_pipe": ("comfy.BASIC_PIPE",),
                "mask": ("dinkster.mask",),
                "samples": ("dinkster.latent",),
                "seed": ("core.int",),
                "steps": ("core.int",),
                "base_only_steps": ("core.int",),
                "denoise": ("core.float",),
                "overlap_factor": ("core.int",),
                "restore_latent": ("core.boolean",),
                "base_cfg": ("core.float",),
                "base_sampler_name": ("core.combo",),
                "base_scheduler": ("core.combo",),
                "region_cfg": ("core.float",),
                "region_sampler_name": ("core.combo",),
                "region_scheduler": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.vae_decode": (
            {"samples": ("dinkster.latent",), "vae": ("dinkster.vae",)},
            {"image": ("dinkster.image",)},
        ),
        "dinkster.vae_decode_tiled": (
            {
                "samples": ("dinkster.latent",),
                "vae": ("dinkster.vae",),
                "tile_size": ("core.int",),
                "overlap": ("core.int",),
                "temporal_size": ("core.int",),
                "temporal_overlap": ("core.int",),
            },
            {"image": ("dinkster.image",)},
        ),
        "dinkster.vae_encode": (
            {"pixels": ("dinkster.image",), "vae": ("dinkster.vae",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.vae_encode_tiled": (
            {
                "pixels": ("dinkster.image",),
                "vae": ("dinkster.vae",),
                "tile_size": ("core.int",),
                "overlap": ("core.int",),
                "temporal_size": ("core.int",),
                "temporal_overlap": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.seedvr2_preprocess": (
            {"resized_images": ("dinkster.image",)},
            {"images": ("dinkster.image",)},
        ),
        "dinkster.seedvr2_postprocess": (
            {
                "images": ("dinkster.image",),
                "original_resized_images": ("dinkster.image",),
                "color_correction_method": ("core.combo",),
            },
            {"images": ("dinkster.image",)},
        ),
        "dinkster.seedvr2_conditioning": (
            {
                "model": ("dinkster.model",),
                "vae_conditioning": ("dinkster.latent",),
            },
            {
                "positive": ("dinkster.conditioning",),
                "negative": ("dinkster.conditioning",),
            },
        ),
        "dinkster.seedvr2_temporal_chunk": (
            {
                "latent": ("dinkster.latent",),
                "temporal_overlap": ("core.int",),
            },
            {"latents": (), "temporal_overlap": ("core.int",)},
        ),
        "dinkster.seedvr2_temporal_merge": (
            {"latents": (), "temporal_overlap": ("core.int",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.combine": (
            {
                "samples1": ("dinkster.latent",),
                "samples2": ("dinkster.latent",),
                "operation": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.mix": (
            {
                "samples1": ("dinkster.latent",),
                "samples2": ("dinkster.latent",),
                "operation": ("core.combo",),
                "factor": ("core.float",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.multiply": (
            {"samples": ("dinkster.latent",), "multiplier": ("core.float",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.rotate": (
            {"samples": ("dinkster.latent",), "angle": ("core.combo",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.flip": (
            {"samples": ("dinkster.latent",), "axis": ("core.combo",)},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.crop": (
            {
                "samples": ("dinkster.latent",),
                "width": ("core.int",),
                "height": ("core.int",),
                "x": ("core.int",),
                "y": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.resize": (
            {
                "samples": ("dinkster.latent",),
                "method": ("core.combo",),
                "width": ("core.int",),
                "height": ("core.int",),
                "crop": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.resize_by": (
            {
                "samples": ("dinkster.latent",),
                "method": ("core.combo",),
                "scale_by": ("core.float",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.composite": (
            {
                "destination": ("dinkster.latent",),
                "source": ("dinkster.latent",),
                "x": ("core.int",),
                "y": ("core.int",),
                "feather": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.composite_masked": (
            {
                "destination": ("dinkster.latent",),
                "source": ("dinkster.latent",),
                "x": ("core.int",),
                "y": ("core.int",),
                "resize_source": ("core.boolean",),
                "mask": ("dinkster.mask",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.concat": (
            {
                "samples1": ("dinkster.latent",),
                "samples2": ("dinkster.latent",),
                "dim": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.cut": (
            {
                "samples": ("dinkster.latent",),
                "dim": ("core.combo",),
                "index": ("core.int",),
                "amount": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.cut_to_batch": (
            {
                "samples": ("dinkster.latent",),
                "dim": ("core.combo",),
                "slice_size": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.from_batch": (
            {
                "samples": ("dinkster.latent",),
                "batch_index": ("core.int",),
                "length": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.repeat": (
            {
                "samples": ("dinkster.latent",),
                "amount": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.seed_behavior": (
            {
                "samples": ("dinkster.latent",),
                "behavior": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.batch": (
            {},
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.rebatch": (
            {
                "latents": (),
                "batch_size": ("core.int",),
            },
            {"latents": ()},
        ),
        "dinkster.latent.set_noise_mask": (
            {
                "samples": ("dinkster.latent",),
                "mask": ("dinkster.mask",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.replace_frames": (
            {
                "destination": ("dinkster.latent",),
                "source": ("dinkster.latent",),
                "index": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.apply_operation": (
            {
                "samples": ("dinkster.latent",),
                "operation": ("dinkster.latent-operation",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.operation_tonemap_reinhard": (
            {"multiplier": ("core.float",)},
            {"operation": ("dinkster.latent-operation",)},
        ),
        "dinkster.latent.operation_sharpen": (
            {
                "sharpen_radius": ("core.int",),
                "sigma": ("core.float",),
                "alpha": ("core.float",),
            },
            {"operation": ("dinkster.latent-operation",)},
        ),
        "dinkster.latent.apply_operation_cfg": (
            {
                "model": ("dinkster.model",),
                "operation": ("dinkster.latent-operation",),
            },
            {"model": ("dinkster.model",)},
        ),
        "dinkster.latent.generate_noise": (
            {
                "width": ("core.int",),
                "height": ("core.int",),
                "batch_size": ("core.int",),
                "seed": ("core.int",),
                "multiplier": ("core.float",),
                "constant_batch_noise": ("core.boolean",),
                "normalize": ("core.boolean",),
                "model": ("dinkster.model",),
                "sigmas": ("dinkster.sigmas",),
                "latent_channels": ("core.combo",),
                "shape": ("core.combo",),
            },
            {"latent": ("dinkster.latent",)},
        ),
        "dinkster.latent.inject_noise": (
            {
                "latents": ("dinkster.latent",),
                "strength": ("core.float",),
                "noise": ("dinkster.latent",),
                "normalize": ("core.boolean",),
                "average": ("core.boolean",),
                "mask": ("dinkster.mask",),
                "mix_randn_amount": ("core.float",),
                "seed": ("core.int",),
            },
            {"latent": ("dinkster.latent",)},
        ),
    }
    assert {
        type_id
        for schema in schemas.values()
        for socket in (*schema.inputs, *schema.outputs)
        for type_id in socket.type.types
        if type_id.startswith("dinkster.")
    } >= {
        "dinkster.model",
        "dinkster.clip",
        "dinkster.clip-vision",
        "dinkster.vae",
        "dinkster.conditioning",
        "dinkster.latent",
        "dinkster.image",
        "dinkster.sampler",
        "dinkster.sigmas",
        "dinkster.guider",
        "dinkster.noise",
        "dinkster.latent-operation",
        "dinkster.model3d",
    }
    assert {
        type_id
        for schema in schemas.values()
        for socket in (*schema.inputs, *schema.outputs)
        for type_id in socket.type.types
        if type_id.startswith("comfy.")
    } == {
        "comfy.AUDIO",
        "comfy.BACKGROUND_REMOVAL",
        "comfy.BASIC_PIPE",
        "comfy.CONTROL_NET",
        "comfy.LATENT_UPSCALE_MODEL",
        "comfy.MESH",
        "comfy.MODEL_PATCH",
        "comfy.MOGE_GEOMETRY",
        "comfy.MOGE_MODEL",
        "comfy.SHAPE_SUBDIVIDES",
        "comfy.VOXEL",
    }

    for node_type in ("dinkster.load_lora", "dinkster.load_lora_model_only"):
        inputs = {item.id: item for item in schemas[node_type].inputs}
        assert inputs["execution_mode"].widget == ComboWidget(options=("auto", "precalculate"))

    for node_type in ("dinkster.ksampler", "dinkster.ksampler_advanced"):
        inputs = {item.id: item for item in schemas[node_type].inputs}
        assert all(item.required for item in inputs.values())
        sampler = inputs["sampler_name"].widget
        scheduler = inputs["scheduler"].widget
        sampler_descriptors = builtin_samplers()
        scheduler_descriptors = builtin_schedulers()
        # The schema labels combo options with the id's short name; the
        # first alias must match it (later aliases carry retired legacy
        # vocabulary and never label anything).
        assert all(item.aliases[0] == item.id.split(".", 1)[1] for item in sampler_descriptors)
        assert all(item.aliases[0] == item.id.split(".", 1)[1] for item in scheduler_descriptors)
        assert sampler == ComboWidget(
            options=tuple(
                ComboOption(value=item.id, label=item.aliases[0])
                for item in sampler_descriptors
                if item.id != "dinkster.configured_sa_solver"
            )
        )
        assert scheduler == ComboWidget(
            options=tuple(
                ComboOption(value=item.id, label=item.aliases[0]) for item in scheduler_descriptors
            )
        )
        assert inputs["scheduler"].default == "dinkster.simple"

    basic_scheduler = {item.id: item for item in schemas["dinkster.basic_scheduler"].inputs}
    assert basic_scheduler["scheduler"].default == "dinkster.simple"
    sampler_option_widgets = {
        "dinkster.sampler_dpmpp_3m_sde": (
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("noise_device", "gpu", ComboWidget(options=("gpu", "cpu")), False),
        ),
        "dinkster.sampler_dpmpp_2m_sde": (
            ("solver_type", "midpoint", ComboWidget(options=("midpoint", "heun")), False),
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("noise_device", "gpu", ComboWidget(options=("gpu", "cpu")), False),
        ),
        "dinkster.sampler_dpmpp_sde": (
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("r", 0.5, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("noise_device", "gpu", ComboWidget(options=("gpu", "cpu")), False),
        ),
        "dinkster.sampler_dpmpp_2s_ancestral": (
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.sampler_euler_ancestral": (
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.sampler_euler_ancestral_cfg_pp": (
            ("eta", 1.0, NumberWidget(min=0.0, max=1.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=10.0, step=0.01), False),
        ),
        "dinkster.sampler_lms": (("order", 4, NumberWidget(min=1, max=100), False),),
        "dinkster.sampler_dpm_adaptative": (
            ("order", 3, NumberWidget(min=2, max=3), False),
            ("rtol", 0.05, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("atol", 0.0078, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("h_init", 0.05, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("pcoeff", 0.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("icoeff", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("dcoeff", 0.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("accept_safety", 0.81, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("eta", 0.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.sampler_er_sde": (
            (
                "solver_type",
                "ER-SDE",
                ComboWidget(options=("ER-SDE", "Reverse-time SDE", "ODE")),
                False,
            ),
            ("max_stage", 3, NumberWidget(min=1, max=3), False),
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.sampler_seeds_2": (
            ("solver_type", "phi_1", ComboWidget(options=("phi_1", "phi_2")), False),
            ("eta", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("s_noise", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
            ("r", 0.5, NumberWidget(min=0.01, max=1.0, step=0.01), False),
        ),
    }
    assert {
        node_type: tuple(
            (item.id, item.default, item.widget, item.advanced) for item in schema.inputs
        )
        for node_type, schema in schemas.items()
        if node_type in sampler_option_widgets
    } == sampler_option_widgets
    scheduler_widgets = {
        "dinkster.beta_sampling_scheduler": (
            ("model", None, None, False),
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("alpha", 0.6, NumberWidget(min=0.0, max=50.0, step=0.01), False),
            ("beta", 0.6, NumberWidget(min=0.0, max=50.0, step=0.01), False),
        ),
        "dinkster.sd_turbo_scheduler": (
            ("model", None, None, False),
            ("steps", 1, NumberWidget(min=1, max=10), False),
            ("denoise", 1.0, NumberWidget(min=0.0, max=1.0, step=0.01), False),
        ),
        "dinkster.karras_scheduler": (
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("sigma_max", 14.614642, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("sigma_min", 0.0291675, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("rho", 7.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.exponential_scheduler": (
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("sigma_max", 14.614642, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("sigma_min", 0.0291675, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
        ),
        "dinkster.polyexponential_scheduler": (
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("sigma_max", 14.614642, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("sigma_min", 0.0291675, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("rho", 1.0, NumberWidget(min=0.0, max=100.0, step=0.01), False),
        ),
        "dinkster.laplace_scheduler": (
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("sigma_max", 14.614642, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("sigma_min", 0.0291675, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("mu", 0.0, NumberWidget(min=-10.0, max=10.0, step=0.1), False),
            ("beta", 0.5, NumberWidget(min=0.0, max=10.0, step=0.1), False),
        ),
        "dinkster.vp_scheduler": (
            ("steps", 20, NumberWidget(min=1, max=10_000), False),
            ("beta_d", 19.9, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("beta_min", 0.1, NumberWidget(min=0.0, max=5000.0, step=0.01), False),
            ("eps_s", 0.001, NumberWidget(min=0.0, max=1.0, step=0.0001), False),
        ),
        "dinkster.align_your_steps_scheduler": (
            ("model_type", "SD1", ComboWidget(options=("SD1", "SDXL", "SVD")), False),
            ("steps", 10, NumberWidget(min=1, max=10_000), False),
            ("denoise", 1.0, NumberWidget(min=0.0, max=1.0, step=0.01), False),
        ),
        "dinkster.gits_scheduler": (
            ("coeff", 1.20, NumberWidget(min=0.80, max=1.50, step=0.05), False),
            ("steps", 10, NumberWidget(min=2, max=1000), False),
            ("denoise", 1.0, NumberWidget(min=0.0, max=1.0, step=0.01), False),
        ),
        "dinkster.optimal_steps_scheduler": (
            ("model_type", "FLUX", ComboWidget(options=("FLUX", "Wan", "Chroma")), False),
            ("steps", 20, NumberWidget(min=3, max=1000), False),
            ("denoise", 1.0, NumberWidget(min=0.0, max=1.0, step=0.01), False),
        ),
    }
    assert {
        node_type: tuple(
            (item.id, item.default, item.widget, item.advanced) for item in schema.inputs
        )
        for node_type, schema in schemas.items()
        if node_type in scheduler_widgets
    } == scheduler_widgets
    assert schemas["dinkster.manual_sigmas"].inputs[0].widget == StringWidget(multiline=False)
    assert schemas["dinkster.split_sigmas"].inputs[1].widget == NumberWidget(min=0, max=10_000)
    assert schemas["dinkster.split_sigmas_denoise"].inputs[1].widget == NumberWidget(
        min=0.0, max=1.0, step=0.01
    )
    assert schemas["dinkster.set_first_sigma"].inputs[1].widget == NumberWidget(
        min=0.0, max=20_000.0, step=0.001
    )
    extend_inputs = {
        item.id: item for item in schemas["dinkster.extend_intermediate_sigmas"].inputs
    }
    assert extend_inputs["steps"].widget == NumberWidget(min=1, max=100)
    assert extend_inputs["start_at_sigma"].widget == NumberWidget(min=-1.0, max=20_000.0, step=0.01)
    assert extend_inputs["end_at_sigma"].widget == NumberWidget(min=0.0, max=20_000.0, step=0.01)
    assert extend_inputs["spacing"].widget == ComboWidget(options=("linear", "cosine", "sine"))
    percent_inputs = {
        item.id: item for item in schemas["dinkster.sampling_percent_to_sigma"].inputs
    }
    assert percent_inputs["sampling_percent"].widget == NumberWidget(min=0.0, max=1.0, step=0.0001)
    assert percent_inputs["return_actual_sigma"].default is False

    comfy_sampling_shapes = {
        "dinkster.ksampler": (
            (
                "model",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "positive",
                "negative",
                "latent_image",
                "denoise",
                "conditioning_batching",
                "max_fused_lanes",
            ),
            ("latent",),
        ),
        "dinkster.ksampler_advanced": (
            (
                "model",
                "add_noise",
                "noise_seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "positive",
                "negative",
                "latent_image",
                "start_at_step",
                "end_at_step",
                "return_with_leftover_noise",
                "conditioning_batching",
                "max_fused_lanes",
            ),
            ("latent",),
        ),
        "dinkster.sampler_dpmpp_3m_sde": (
            ("eta", "s_noise", "noise_device"),
            ("sampler",),
        ),
        "dinkster.sampler_dpmpp_2m_sde": (
            ("solver_type", "eta", "s_noise", "noise_device"),
            ("sampler",),
        ),
        "dinkster.sampler_dpmpp_sde": (
            ("eta", "s_noise", "r", "noise_device"),
            ("sampler",),
        ),
        "dinkster.sampler_dpmpp_2s_ancestral": (
            ("eta", "s_noise"),
            ("sampler",),
        ),
        "dinkster.sampler_euler_ancestral": (
            ("eta", "s_noise"),
            ("sampler",),
        ),
        "dinkster.sampler_euler_ancestral_cfg_pp": (
            ("eta", "s_noise"),
            ("sampler",),
        ),
        "dinkster.sampler_lms": (("order",), ("sampler",)),
        "dinkster.sampler_dpm_adaptative": (
            (
                "order",
                "rtol",
                "atol",
                "h_init",
                "pcoeff",
                "icoeff",
                "dcoeff",
                "accept_safety",
                "eta",
                "s_noise",
            ),
            ("sampler",),
        ),
        "dinkster.sampler_er_sde": (
            ("solver_type", "max_stage", "eta", "s_noise"),
            ("sampler",),
        ),
        "dinkster.sampler_seeds_2": (
            ("solver_type", "eta", "s_noise", "r"),
            ("sampler",),
        ),
        "dinkster.beta_sampling_scheduler": (
            ("model", "steps", "alpha", "beta"),
            ("sigmas",),
        ),
        "dinkster.sd_turbo_scheduler": (
            ("model", "steps", "denoise"),
            ("sigmas",),
        ),
        "dinkster.karras_scheduler": (
            ("steps", "sigma_max", "sigma_min", "rho"),
            ("sigmas",),
        ),
        "dinkster.exponential_scheduler": (
            ("steps", "sigma_max", "sigma_min"),
            ("sigmas",),
        ),
        "dinkster.polyexponential_scheduler": (
            ("steps", "sigma_max", "sigma_min", "rho"),
            ("sigmas",),
        ),
        "dinkster.laplace_scheduler": (
            ("steps", "sigma_max", "sigma_min", "mu", "beta"),
            ("sigmas",),
        ),
        "dinkster.vp_scheduler": (
            ("steps", "beta_d", "beta_min", "eps_s"),
            ("sigmas",),
        ),
        "dinkster.align_your_steps_scheduler": (
            ("model_type", "steps", "denoise"),
            ("sigmas",),
        ),
        "dinkster.gits_scheduler": (
            ("coeff", "steps", "denoise"),
            ("sigmas",),
        ),
        "dinkster.optimal_steps_scheduler": (
            ("model_type", "steps", "denoise"),
            ("sigmas",),
        ),
        "dinkster.manual_sigmas": (("sigmas",), ("sigmas",)),
        "dinkster.split_sigmas": (
            ("sigmas", "step"),
            ("high_sigmas", "low_sigmas"),
        ),
        "dinkster.split_sigmas_denoise": (
            ("sigmas", "denoise"),
            ("high_sigmas", "low_sigmas"),
        ),
        "dinkster.flip_sigmas": (("sigmas",), ("sigmas",)),
        "dinkster.set_first_sigma": (("sigmas", "sigma"), ("sigmas",)),
        "dinkster.extend_intermediate_sigmas": (
            ("sigmas", "steps", "start_at_sigma", "end_at_sigma", "spacing"),
            ("sigmas",),
        ),
        "dinkster.sampling_percent_to_sigma": (
            ("model", "sampling_percent", "return_actual_sigma"),
            ("sigma_value",),
        ),
        "dinkster.sampler_custom": (
            (
                "model",
                "add_noise",
                "noise_seed",
                "cfg",
                "positive",
                "negative",
                "sampler",
                "sigmas",
                "latent_image",
                "conditioning_batching",
                "max_fused_lanes",
            ),
            ("output", "denoised_output"),
        ),
        "dinkster.sampler_custom_advanced": (
            ("noise", "guider", "sampler", "sigmas", "latent_image"),
            ("output", "denoised_output"),
        ),
        "dinkster.compat.impact_regional_sampler": (
            (
                "base_basic_pipe",
                "region_basic_pipe",
                "mask",
                "samples",
                "seed",
                "steps",
                "base_only_steps",
                "denoise",
                "overlap_factor",
                "restore_latent",
                "base_cfg",
                "base_sampler_name",
                "base_scheduler",
                "region_cfg",
                "region_sampler_name",
                "region_scheduler",
            ),
            ("latent",),
        ),
    }
    assert {
        node_type: (
            tuple(item.id for item in schemas[node_type].inputs),
            tuple(item.id for item in schemas[node_type].outputs),
        )
        for node_type in comfy_sampling_shapes
    } == comfy_sampling_shapes


def test_text_generation_schemas_match_comfy_workflow_shape() -> None:
    schemas = build_schemas(GENERATION_NODES)

    for node_type, alias in (
        ("dinkster.text_generate", "TextGenerate"),
        ("dinkster.prompt_enhance", "TextGenerateLTX2Prompt"),
    ):
        schema = schemas[node_type]
        assert schema.aliases == (alias,)
        assert tuple(item.id for item in schema.inputs) == (
            "clip",
            "provider",
            "prompt",
            "image",
            "video",
            "audio",
            "max_length",
            "thinking",
            "use_default_template",
        )
        clip = schema.input("clip")
        assert clip is not None
        assert clip == InputSpec("clip", clip.type, required=False, lazy=True)
        provider = schema.input("provider")
        assert provider is not None
        assert not provider.required
        assert provider.hidden
        assert not provider.advanced
        assert provider.widget == ComboWidget(
            remote_route="/api/choices/dinkster.generation.providers"
        )
        assert schema.outputs[0].id == "generated_text"
        sampling = schema.combos[0]
        assert sampling.id == "sampling_mode"
        assert sampling.default == "on"
        assert tuple(option.key for option in sampling.options) == ("on", "off")
        assert tuple(item.id for item in sampling.options[0].inputs) == (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "seed",
            "presence_penalty",
        )


def test_ltxv_context_windows_schema_matches_reference_surface() -> None:
    schema = build_schemas(GENERATION_NODES)["dinkster.ltxv_context_windows"]

    assert schema.category == "model/patch"
    assert [(item.id, item.default, item.widget, item.advanced) for item in schema.inputs] == [
        ("model", None, None, False),
        ("context_length", 145, NumberWidget(min=1, max=16384, step=8), False),
        ("context_overlap", 40, NumberWidget(min=0, max=16384, step=8), False),
        (
            "context_schedule",
            "standard_uniform",
            ComboWidget(
                options=("standard_static", "standard_uniform", "looped_uniform", "batched")
            ),
            False,
        ),
        ("context_stride", 1, NumberWidget(min=1, max=32, step=1), True),
        ("closed_loop", False, None, True),
        (
            "fuse_method",
            "pyramid",
            ComboWidget(options=("flat", "pyramid", "overlap-linear")),
            False,
        ),
        ("freenoise", True, None, True),
        ("retain_first_frame", False, None, False),
    ]


def test_lora_stack_families_are_ordered_structured_members() -> None:
    schemas = build_schemas(GENERATION_NODES)
    for node_type, minimum, fields in (
        (
            "dinkster.load_checkpoint_stack",
            0,
            ("lora", "strength_model", "strength_clip"),
        ),
        (
            "dinkster.apply_lora_stack",
            1,
            ("lora", "strength_model", "strength_clip"),
        ),
        (
            "dinkster.apply_lora_stack_model_only",
            1,
            ("lora", "strength_model"),
        ),
    ):
        schema = schemas[node_type]
        family = schema.input_families[0]
        assert (family.id, family.min_members, family.max_members, family.member_prefix) == (
            "loras",
            minimum,
            50,
            "lora_",
        )
        assert tuple(item.id for item in family.template) == fields
        for item in family.template:
            if isinstance(item, InputSpec) and item.id.startswith("strength_"):
                assert item.default == 1.0
                assert item.widget == NumberWidget(min=-100.0, max=100.0, step=0.01)
        wire_interface = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])
        wire_family = next(item for item in wire_interface if item["id"] == "loras")
        wire_template = cast("list[dict[str, Any]]", wire_family["template"])
        for item in wire_template:
            if cast("str", item["id"]).startswith("strength_"):
                assert item["default"] == 1.0
                assert item["widget"] == {
                    "type": "NUMBER",
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }

    effective = elaborate(
        schemas["dinkster.apply_lora_stack"],
        [
            "model",
            "clip",
            "loras.lora_2.lora",
            "loras.lora_2.strength_model",
            "loras.lora_2.strength_clip",
            "loras.lora_1.lora",
            "loras.lora_1.strength_model",
            "loras.lora_1.strength_clip",
        ],
    )
    assert [item.id for item in effective.inputs][-6:] == [
        "loras.lora_2.lora",
        "loras.lora_2.strength_model",
        "loras.lora_2.strength_clip",
        "loras.lora_1.lora",
        "loras.lora_1.strength_model",
        "loras.lora_1.strength_clip",
    ]


def test_diffusion_components_are_generic_ordered_asset_role_members() -> None:
    schema = build_schemas(GENERATION_NODES)["dinkster.load_diffusion_components"]
    family = schema.input_families[0]

    assert (family.id, family.min_members, family.max_members, family.member_prefix) == (
        "components",
        1,
        64,
        "component_",
    )
    assert tuple(item.id for item in family.template) == ("component", "role")
    component, role = cast("tuple[InputSpec, InputSpec]", family.template)
    assert component.type.types == ("dinkster.asset",)
    assert component.widget == AssetWidget(
        accept=("application/octet-stream",),
        kind="model/diffusion",
    )
    assert role.type.types == ("core.string",)
    assert role.widget == StringWidget()

    effective = elaborate(
        schema,
        [
            "weight_dtype",
            "components.component_1.component",
            "components.component_1.role",
            "components.component_0.component",
            "components.component_0.role",
        ],
    )
    assert [item.id for item in effective.inputs] == [
        "weight_dtype",
        "components.component_1.component",
        "components.component_1.role",
        "components.component_0.component",
        "components.component_0.role",
    ]


def test_comfy_numeric_widget_bounds_reach_generation_wire() -> None:
    schemas = build_schemas(GENERATION_NODES)
    strength_widget = NumberWidget(min=-100.0, max=100.0, step=0.01)
    for node_type, strength_ids in (
        ("dinkster.load_lora", ("strength_model", "strength_clip")),
        ("dinkster.load_lora_model_only", ("strength_model",)),
    ):
        schema = schemas[node_type]
        inputs = {item.id: item for item in schema.inputs}
        wire_interface = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])
        wire_inputs = {item["id"]: item for item in wire_interface}
        for input_id in strength_ids:
            assert inputs[input_id].default == 1.0
            assert inputs[input_id].widget == strength_widget
            assert wire_inputs[input_id]["default"] == 1.0
            assert wire_inputs[input_id]["widget"] == {
                "type": "NUMBER",
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
            }

    seed_widget = NumberWidget(
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        step=1,
        control_after_generate="randomize",
    )
    for node_type, input_id in (
        ("dinkster.ksampler", "seed"),
        ("dinkster.ksampler_advanced", "noise_seed"),
        ("dinkster.random_noise", "noise_seed"),
        ("dinkster.sampler_custom", "noise_seed"),
    ):
        schema = schemas[node_type]
        input_spec = next(item for item in schema.inputs if item.id == input_id)
        wire_interface = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])
        wire_input = next(item for item in wire_interface if item["id"] == input_id)
        assert input_spec.default == 0
        assert input_spec.widget == seed_widget
        assert wire_input["default"] == 0
        assert wire_input["widget"] == {
            "type": "NUMBER",
            "min": 0,
            "max": str(0xFFFFFFFFFFFFFFFF),
            "step": 1,
            "controlAfterGenerate": "randomize",
        }


def test_compat_provider_executes_every_generation_schema_exactly() -> None:
    manifest = load_manifest(COMPAT_MANIFEST)
    owner_schemas = build_schemas(GENERATION_NODES)
    provider_schemas = build_schemas(GENERATION_PROVIDER_NODES)
    carrier_owner_schemas = build_schemas(GENERATION_COMPAT_CARRIER_NODES)
    carrier_provider_schemas = build_schemas(USDU_CARRIER_NODES)

    assert manifest.executes == (*GENERATION_COMPAT_CARRIER_NODE_IDS, *GENERATION_NODE_IDS)
    assert tuple(provider_schemas) == GENERATION_NODE_IDS
    assert {
        node_type: schema_signature(schema) for node_type, schema in provider_schemas.items()
    } == {node_type: schema_signature(schema) for node_type, schema in owner_schemas.items()}
    assert {
        node_type: schema_signature(schema)
        for node_type, schema in carrier_provider_schemas.items()
    } == {
        node_type: schema_signature(schema) for node_type, schema in carrier_owner_schemas.items()
    }
    # Native-arm consumers must co-locate with the resident model handles they use.
    assert set(GENERATION_NODE_IDS).intersection(dict(manifest.arms)["native"]) == {
        "dinkster.apply_texture_to_mesh",
        "dinkster.bake_ambient_occlusion",
        "dinkster.bake_normal_map_from_mesh",
        "dinkster.bake_texture_from_voxel",
        "dinkster.basic_guider",
        "dinkster.basic_scheduler",
        "dinkster.beta_sampling_scheduler",
        "dinkster.cfg_guider",
        "dinkster.dual_cfg_guider",
        "dinkster.dual_model_guider",
        "dinkster.scheduled_cfg_guider",
        "dinkster.cfg_override",
        "dinkster.clip_set_last_layer",
        "dinkster.clip_text_encode",
        "dinkster.clip_text_encode_lumina2",
        "dinkster.model_sampling_aura_flow",
        "dinkster.clip_text_encode_controlnet",
        "dinkster.decimate_mesh",
        "dinkster.disable_cfg1_optimization",
        "dinkster.add_noise",
        "dinkster.empty_trellis2_latent_structure",
        "dinkster.estimate_geometry",
        "dinkster.geometry_to_fov",
        "dinkster.get_mesh_info",
        "dinkster.image_crop_to_mask",
        "dinkster.ksampler",
        "dinkster.ksampler_advanced",
        "dinkster.sampler_sa_solver",
        "dinkster.lazy_cache",
        "dinkster.easy_cache",
        "dinkster.attention_schedule",
        "dinkster.load_background_removal",
        "dinkster.load_diffusion_components",
        "dinkster.load_diffusion_model",
        "dinkster.load_geometry_model",
        "dinkster.load_model_profile",
        "dinkster.ltxav_audio_vae_decode",
        "dinkster.ltxav_id_lora_reference_audio",
        "dinkster.ltxav_reference_audio",
        "dinkster.ltxv_spatiotemporal_guidance",
        "dinkster.ltxv_modality_guidance",
        "dinkster.ltxv_duration_predictor",
        "dinkster.ltxv_dual_cfg_guider",
        "dinkster.ltxv_image_to_video",
        "dinkster.ltxv_image_to_video_inplace",
        "dinkster.ltxv_add_guide",
        "dinkster.ltxv_crop_guides",
        "dinkster.ltxv_latent_upsampler",
        "dinkster.mesh_to_model3d",
        "dinkster.model_sampling_sd3",
        "dinkster.model_sampling_ltxv",
        "dinkster.model_sampling_flux",
        "dinkster.paint_mesh",
        "dinkster.perp_neg_guider",
        "dinkster.pixal3d_conditioning",
        "dinkster.preview_mask",
        "dinkster.prompt_enhance",
        "dinkster.remesh_mesh",
        "dinkster.remove_background",
        "dinkster.render_uv_atlas",
        "dinkster.rescale_cfg",
        "dinkster.sampler_custom",
        "dinkster.sampler_custom_advanced",
        "dinkster.compat.impact_regional_sampler",
        "dinkster.sampling_percent_to_sigma",
        "dinkster.sd_turbo_scheduler",
        "dinkster.seedvr2_conditioning",
        "dinkster.smooth_mesh_normals",
        "dinkster.t5_tokenizer_options",
        "dinkster.text_generate",
        "dinkster.trellis2_conditioning",
        "dinkster.trellis2_shape_stage",
        "dinkster.trellis2_texture_stage",
        "dinkster.trellis2_upsample_stage",
        "dinkster.unwrap_mesh",
        "dinkster.vae_decode",
        "dinkster.vae_decode_tiled",
        "dinkster.vae_decode_shape_trellis",
        "dinkster.vae_decode_structure_trellis2",
        "dinkster.vae_decode_texture_trellis",
        "dinkster.vae_encode",
        "dinkster.vae_encode_tiled",
        "dinkster.voxel_to_mesh",
    }
    assert GENERATION_CLAIMED_V1_NAMES == (
        "EmptyTrellis2LatentStructure",
        "Trellis2Conditioning",
        "Pixal3DConditioning",
        "VaeDecodeStructureTrellis2",
        "Trellis2ShapeStage",
        "Trellis2UpsampleStage",
        "VaeDecodeShapeTrellis",
        "Trellis2TextureStage",
        "VaeDecodeTextureTrellis",
        "LoadMoGeModel",
        "MoGeInference",
        "MoGeGeometryToFOV",
        "LoadBackgroundRemovalModel",
        "RemoveBackground",
        "ImageCropToMask",
        "MaskPreview",
        "VoxelToMesh",
        "GetMeshInfo",
        "RemeshMesh",
        "DecimateMesh",
        "MeshSmoothNormals",
        "UnwrapMesh",
        "PaintMesh",
        "BakeTextureFromVoxel",
        "BakeNormalMapFromMesh",
        "BakeAmbientOcclusion",
        "RenderUVAtlas",
        "ApplyTextureToMesh",
        "MeshToFile3D",
        "CheckpointLoaderSimple",
        "ControlNetLoader",
        "ControlNetApply",
        "ControlNetApplyAdvanced",
        "SetUnionControlNetType",
        "UNETLoader",
        "LTXAVTextEncoderLoader",
        "LTXVAudioVAELoader",
        "LatentUpscaleModelLoader",
        "LoraLoader",
        "LoraLoaderModelOnly",
        "CLIPTextEncode",
        "CLIPTextEncodeLumina2",
        "ModelSamplingAuraFlow",
        "TextGenerate",
        "TextGenerateLTX2Prompt",
        "CLIPSetLastLayer",
        "T5TokenizerOptions",
        "CLIPTextEncodeControlnet",
        "FluxGuidance",
        "FluxDisableGuidance",
        "ReferenceLatent",
        "CFGZeroStar",
        "CFGNorm",
        "TCFG",
        "FreSca",
        "ContextWindowsManual",
        "WanContextWindowsManual",
        "LTXVContextWindows",
        "APG",
        "Mahiro",
        "Epsilon Scaling",
        "CFGOverride",
        "RescaleCFG",
        "RenormCFG",
        "TemporalScoreRescaling",
        "NAGuidance",
        "LTXVConditioning",
        "LTXVReferenceAudio",
        "LTXVSpatioTemporalGuidance",
        "LTXVModalityGuidance",
        "LTXVDurationPredictor",
        "LTXVDualCFGGuider",
        "LTXVImgToVideo",
        "LTXVImgToVideoInplace",
        "LTXVAddGuide",
        "LTXVCropGuides",
        "LTXVLatentUpsampler",
        "ConditioningCombine",
        "ConditioningAverage",
        "ConditioningConcat",
        "ConditioningMultiply",
        "ConditioningSetArea",
        "ConditioningSetAreaPercentage",
        "ConditioningSetAreaPercentageVideo",
        "ConditioningSetMask",
        "ConditioningSetTimestepRange",
        "ConditioningZeroOut",
        "ChromaRadianceOptions",
        "ModelSamplingSD3",
        "ModelSamplingLTXV",
        "ModelSamplingFlux",
        "EmptyLatentImage",
        "EmptySD3LatentImage",
        "EmptyChromaRadianceLatentImage",
        "EmptyFlux2LatentImage",
        "KSampler",
        "KSamplerAdvanced",
        "KSamplerSelect",
        "SamplerDPMPP_3M_SDE",
        "SamplerDPMPP_2M_SDE",
        "SamplerDPMPP_SDE",
        "SamplerDPMPP_2S_Ancestral",
        "SamplerEulerAncestral",
        "SamplerEulerAncestralCFGPP",
        "SamplerLMS",
        "SamplerDPMAdaptative",
        "SamplerER_SDE",
        "SamplerSEEDS2",
        "SamplerSASolver",
        "BasicScheduler",
        "BetaSamplingScheduler",
        "SDTurboScheduler",
        "KarrasScheduler",
        "ExponentialScheduler",
        "PolyexponentialScheduler",
        "LaplaceScheduler",
        "VPScheduler",
        "AlignYourStepsScheduler",
        "GITSScheduler",
        "OptimalStepsScheduler",
        "Flux2Scheduler",
        "ManualSigmas",
        "SplitSigmas",
        "SplitSigmasDenoise",
        "FlipSigmas",
        "SetFirstSigma",
        "ExtendIntermediateSigmas",
        "SamplingPercentToSigma",
        "BasicGuider",
        "CFGGuider",
        "DualCFGGuider",
        "PerpNegGuider",
        "DisableModelCfg1Optimization",
        "DisableNoise",
        "RandomNoise",
        "AddNoise",
        "SamplerCustom",
        "SamplerCustomAdvanced",
        "LTXVAudioVAEDecode",
        "VAEDecode",
        "VAEDecodeTiled",
        "VAEEncode",
        "VAEEncodeTiled",
        "SeedVR2Preprocess",
        "SeedVR2PostProcessing",
        "SeedVR2Conditioning",
        "SeedVR2TemporalChunk",
        "SeedVR2TemporalMerge",
    )


def test_seedvr2_comfy_aliases_translate_outputs_and_manual_chunking() -> None:
    schemas = build_schemas((*GENERATION_NODES, SaveVideo))
    translation = translate_prompt(
        {
            "preprocess": {
                "class_type": "SeedVR2Preprocess",
                "inputs": {"resized_images": []},
            },
            "encode": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["preprocess", 0],
                    "vae": "vae",
                    "tile_size": 512,
                    "overlap": 128,
                    "temporal_size": 4096,
                    "temporal_overlap": 8,
                },
            },
            "conditioning": {
                "class_type": "SeedVR2Conditioning",
                "inputs": {"model": "model", "vae_conditioning": ["encode", 0]},
            },
            "chunk": {
                "class_type": "SeedVR2TemporalChunk",
                "inputs": {
                    "latent": ["encode", 0],
                    "temporal_overlap": 4,
                    "chunking_mode": "manual",
                    "chunking_mode.frames_per_chunk": 21,
                },
            },
            "merge": {
                "class_type": "SeedVR2TemporalMerge",
                "inputs": {
                    "latents": ["chunk", 0],
                    "temporal_overlap": ["chunk", 1],
                },
            },
            "sample": {
                "class_type": "KSampler",
                "inputs": {
                    "model": "model",
                    "seed": 0,
                    "steps": 1,
                    "cfg": 1.0,
                    "sampler_name": "dinkster.euler",
                    "scheduler": "dinkster.simple",
                    "positive": ["conditioning", 0],
                    "negative": ["conditioning", 1],
                    "latent_image": ["merge", 0],
                    "denoise": 1.0,
                },
            },
            "decode": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["sample", 0],
                    "vae": "vae",
                    "tile_size": 512,
                    "overlap": 128,
                    "temporal_size": 4096,
                    "temporal_overlap": 8,
                },
            },
            "postprocess": {
                "class_type": "SeedVR2PostProcessing",
                "inputs": {
                    "images": ["decode", 0],
                    "original_resized_images": ["preprocess", 0],
                    "color_correction_method": "lab",
                },
            },
            "output": {
                "class_type": "dinkster.save_video",
                "inputs": {"images": ["postprocess", 0]},
            },
        },
        schemas,
    )

    assert translation.targets == ("output",)
    nodes = translation.graph.nodes
    seedvr_nodes = [
        cast("GraphNode", nodes[node_id])
        for node_id in ("preprocess", "postprocess", "conditioning", "chunk", "merge")
    ]
    assert [node.node_type for node in seedvr_nodes] == [
        "dinkster.seedvr2_preprocess",
        "dinkster.seedvr2_postprocess",
        "dinkster.seedvr2_conditioning",
        "dinkster.seedvr2_temporal_chunk",
        "dinkster.seedvr2_temporal_merge",
    ]
    chunk = cast("GraphNode", nodes["chunk"])
    merge = cast("GraphNode", nodes["merge"])
    sample = cast("GraphNode", nodes["sample"])
    assert cast("GraphNode", nodes["encode"]).node_type == "dinkster.vae_encode_tiled"
    assert cast("GraphNode", nodes["decode"]).node_type == "dinkster.vae_decode_tiled"
    assert chunk.slot_variants == {"chunking_mode": "manual"}
    assert chunk.inputs["chunking_mode.frames_per_chunk"] == 21
    assert merge.inputs == {
        "latents": Link("chunk", "latents"),
        "temporal_overlap": Link("chunk", "temporal_overlap"),
    }
    assert sample.inputs["positive"] == Link("conditioning", "positive")
    assert sample.inputs["negative"] == Link("conditioning", "negative")


def test_production_compat_specs_compose_with_carrier_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def isolated_worker(
        _composer: ServingComposer,
        _spec: PackSpec,
        manifest: PackManifest,
        registry: TypeRegistry,
    ) -> InProcessWorker:
        if manifest.name == "dinkster-vision-upscale":
            return InProcessWorker(build_node_types((UpscaleWithModel,)), registry)
        if manifest.name == "dinkster-compat-comfy":
            return _CompatSchemaWorker(registry)
        raise AssertionError(f"unexpected isolated pack {manifest.name}")

    monkeypatch.setattr(ServingComposer, "_isolated_worker", isolated_worker)
    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    generation, compat = comfy_compat_specs(comfy_root, python=sys.executable)
    checkpoint = AssetRef(
        digest=f"blake3:{'a' * 64}",
        name="checkpoint.safetensors",
        size=1,
        media_type="application/octet-stream",
        virtual_path="models/checkpoints/checkpoint.safetensors",
    )
    upscaler = AssetRef(
        digest=f"blake3:{'b' * 64}",
        name="upscaler.pth",
        size=1,
        media_type="application/octet-stream",
        virtual_path="models/upscale_models/upscaler.pth",
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            for pack_id in (
                "dinkster-nodes-media-io",
                "dinkster-nodes-foundation",
                "dinkster-nodes-image",
            ):
                await composer.add_pack(default_pack_spec(pack_id))
            await composer.add_pack(PackSpec(UPSCALE_MANIFEST, python=sys.executable))
            await composer.add_pack(generation)
            await composer.add_pack(compat)

            schemas = composer.composition.schemas
            assert set(GENERATION_COMPAT_CARRIER_NODE_IDS) <= set(schemas)
            assert schemas["dinkster.compat.ultimate_sd_upscale"].aliases == ("UltimateSDUpscale",)
            assert schemas["dinkster.compat.ultimate_sd_upscale_no_upscale"].aliases == (
                "UltimateSDUpscaleNoUpscale",
            )
            assert schemas["dinkster.compat.upscale_model_loader"].aliases == (
                "UpscaleModelLoader",
            )
            translation = translate_prompt(
                {
                    "source": {
                        "class_type": "dinkster.image.generate",
                        "inputs": {"width": 64, "height": 32, "batch_size": 1},
                    },
                    "checkpoint": {
                        "class_type": "dinkster.load_checkpoint",
                        "inputs": {"checkpoint": checkpoint.to_wire()},
                    },
                    "positive": {
                        "class_type": "dinkster.clip_text_encode",
                        "inputs": {"clip": ["checkpoint", 1], "text": "test"},
                    },
                    "negative": {
                        "class_type": "dinkster.clip_text_encode",
                        "inputs": {"clip": ["checkpoint", 1], "text": ""},
                    },
                    "loader": {
                        "class_type": "UpscaleModelLoader",
                        "inputs": {"model_name": upscaler.to_wire()},
                    },
                    "upscale": {
                        "class_type": "UltimateSDUpscale",
                        "inputs": {
                            "image": ["source", 0],
                            "model": ["checkpoint", 0],
                            "positive": ["positive", 0],
                            "negative": ["negative", 0],
                            "vae": ["checkpoint", 2],
                            "upscale_model": ["loader", 0],
                            "mode_type": "None",
                            "seam_fix_mode": "None",
                        },
                    },
                    "output": {
                        "class_type": "dinkster.preview_image",
                        "inputs": {"images": ["upscale", 0]},
                    },
                },
                schemas,
            )
            assert translation.targets == ("output",)
            assert "loader" not in translation.graph.nodes
            assert all(
                not isinstance(node, GraphNode)
                or node.node_type not in GENERATION_COMPAT_CARRIER_NODE_IDS
                for node in translation.graph.nodes.values()
            )
            assert any(
                isinstance(node, GraphNode) and node.node_type == "dinkster.image.upscale_model"
                for node in translation.graph.nodes.values()
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_production_generation_providers_publish_progressively_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def isolated_worker(
        _composer: ServingComposer,
        _spec: PackSpec,
        manifest: PackManifest,
        registry: TypeRegistry,
    ) -> InProcessWorker:
        if manifest.name == "dinkster-compat-comfy":
            return _CompatSchemaWorker(registry)
        if manifest.name == "dinkster-nodes-generation-openai":
            return InProcessWorker(build_node_types(OPENAI_GENERATION_NODES), registry)
        raise AssertionError(f"unexpected isolated pack {manifest.name}")

    monkeypatch.setattr(ServingComposer, "_isolated_worker", isolated_worker)
    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    generation, compat = comfy_compat_specs(comfy_root, python=sys.executable)
    openai = PackSpec(OPENAI_MANIFEST, python=sys.executable)

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                choices=composition.choices,
                schema_owners=composition.schema_owners,
                choice_owners=composition.choice_owners,
            )
            state = app[STATE_KEY]

            def publish(delta: PackDelta) -> None:
                state.replace(
                    (),
                    (),
                    delta.schemas,
                    delta.packs,
                    delta.node_packs,
                    execution_arms=delta.execution_arms,
                    remove_choices=tuple(delta.derived_choices),
                    choices={**delta.choices, **delta.derived_choices},
                    lazy_choices=delta.lazy_choices,
                    schema_owners=delta.schema_owners,
                    choice_owners=delta.choice_owners,
                    compat_skips=delta.compat_skips,
                )

            for pack_id in (
                "dinkster-nodes-foundation",
                "dinkster-nodes-media-io",
                "dinkster-nodes-image",
            ):
                publish(await composer.add_pack(default_pack_spec(pack_id)))
            generation_delta = await composer.add_pack(generation)
            assert generation_delta.choice_owners["dinkster.generation.providers"] == (
                "dinkster-nodes-generation"
            )
            publish(generation_delta)
            assert "dinkster.text_generate" not in state.schemas
            assert state.choices["dinkster.generation.providers"] == ()

            compat_delta = await composer.add_pack(compat)
            assert compat_delta.schema_owners["dinkster.text_generate"] == (
                "dinkster-nodes-generation"
            )
            publish(compat_delta)
            assert "dinkster.text_generate" in state.schemas
            assert state.node_packs["dinkster.text_generate"] == "dinkster-nodes-generation"

            openai_delta = await composer.add_pack(openai)
            assert openai_delta.choice_owners["dinkster.generation.providers"] == (
                "dinkster-nodes-generation"
            )
            publish(openai_delta)
            assert state.choices["dinkster.generation.providers"] == (
                "dinkster-nodes-generation-openai",
            )

            result = await apply_reload(state, composer, "dinkster-nodes-generation-openai")
            assert result["epoch"] == state.schema_epoch
            assert state.choices["dinkster.generation.providers"] == (
                "dinkster-nodes-generation-openai",
            )
            assert "dinkster.text_generate" in state.schemas
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_media_pack_owns_native_image_mask_and_asset_contracts() -> None:
    registry = TypeRegistry()
    register_media_types(registry)

    assert "dinkster.asset" in registry
    assert IMAGE_TYPE == "dinkster.image"
    assert IMAGE_TYPE in registry
    assert registry.spec(IMAGE_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    assert MASK_TYPE == "dinkster.mask"
    assert MASK_TYPE in registry
    assert registry.spec(MASK_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    assert [(item.kind, item.mime) for item in registry.renditions_of(MASK_TYPE)] == [
        ("png", "image/png")
    ]

    load = LoadVideo.schema()
    save = SaveVideo.schema()
    assert load.outputs[0].type.types == (IMAGE_TYPE,)
    assert save.inputs[0].type.types == ("comfy.VIDEO",)
    assert save.outputs[0].type.types == ("comfy.VIDEO",)
    assert save.outputs[1].type.element is not None
    assert save.outputs[1].type.element.types == ("comfy.VIDEO",)


def test_generation_owner_stays_unpublished_without_a_provider() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(default_pack_spec("dinkster-nodes-foundation"))
            await composer.add_pack(default_pack_spec("dinkster-nodes-media-io"))
            delta = await composer.add_pack(
                PackSpec(MANIFEST, trust_reserved=True, in_process=True)
            )
            assert delta.schemas == {}
            assert set(GENERATION_SCHEMA_NODE_IDS) <= set(composer.composition.schemas)
            assert composer.incomplete_generation_removals() == {
                "dinkster-nodes-generation": GENERATION_SCHEMA_NODE_IDS
            }
            with pytest.raises(CompositionError, match="have no execution provider"):
                composer.validate_complete_generation()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_load_latent_has_one_native_schema_and_preserves_source_metadata() -> None:
    from dinkster_compat_comfy.native import LoadLatent, merge_native_nodes
    from dinkster_compat_comfy.schema_snapshot import core_schema_snapshot
    from dinkster_schema import SourceFilenameSpec, TypeExpr, schema_from_wire

    source = schema_from_wire(core_schema_snapshot()["schemas"]["comfy.LoadLatent"])

    class TranslatedLoadLatent(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return source

    schemas = build_schemas(merge_native_nodes((TranslatedLoadLatent,)))
    assert "comfy.LoadLatent" not in schemas
    schema = schemas["dinkster.load_latent"]
    assert schema == LoadLatent.schema()
    assert schema.aliases == ("LoadLatent",)
    assert [output.id for output in schema.outputs] == ["samples", "vae_hint"]
    latent = TypeExpr.concrete("dinkster.latent")
    assert schema.outputs[0].type == latent
    assert schema.inputs[0].type == TypeExpr.asset_of(latent)
    assert schema.inputs[0].source_filename == SourceFilenameSpec("data/latent", "input")
    assert source.node_type == "comfy.LoadLatent"
    assert len(source.outputs) == 1
    assert source.inputs[0].id == "latent"
    assert "LoadLatent" in source.aliases


def test_controlnet_legacy_names_have_one_schema_owner() -> None:
    owners = {}
    for node in (*NATIVE_NODES, *GENERATION_NODES):
        schema = node.schema()
        for alias in schema.aliases:
            if alias in ("ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced"):
                assert alias not in owners
                owners[alias] = schema.node_type
    assert owners == {
        "ControlNetLoader": "dinkster.load_controlnet",
        "ControlNetApply": "dinkster.apply_controlnet",
        "ControlNetApplyAdvanced": "dinkster.apply_controlnet_advanced",
    }
