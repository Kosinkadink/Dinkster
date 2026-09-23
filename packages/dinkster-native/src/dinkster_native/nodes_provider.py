"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .families.minimax_h3 import (
    _component_runtime_with_options,
    _minimax_h3_schedule_runtime,
)
from .native_arm_core import (
    _ASSET,
    _BOOLEAN,
    _COMBO,
    _COMFY_AUDIO,
    _COMFY_LATENT_UPSCALE_MODEL,
    _COMFY_MESH,
    _COMFY_MODEL_PATCH,
    _COMFY_SHAPE_SUBDIVIDES,
    _COMFY_VOXEL,
    _DINKSTER_CLIP,
    _DINKSTER_CLIP_VISION,
    _DINKSTER_CONDITIONING,
    _DINKSTER_CONTEXT_WINDOWS,
    _DINKSTER_GUIDER,
    _DINKSTER_IMAGE,
    _DINKSTER_INPAINT_CONDITIONING,
    _DINKSTER_LATENT,
    _DINKSTER_LATENT_LIST,
    _DINKSTER_LATENT_OPERATION,
    _DINKSTER_MASK,
    _DINKSTER_MODEL,
    _DINKSTER_NOISE,
    _DINKSTER_SAMPLER,
    _DINKSTER_SIGMAS,
    _DINKSTER_VAE,
    _FLOAT,
    _INT,
    _INT_LIST,
    _NATIVE_PREPARED_CONDITIONING_KEY,
    _STRING,
    GENERATION_NODES,
    Any,
    AssetWidget,
    Callable,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    NativeRuntimeHandle,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    _active_inference_registries,
    _inference_registries,
    _split_ltx_frame_rate,
    cast,
    dataclass,
    importlib,
    partial,
)
from .native_arm_runtime import (
    _application_chain_model,
    _native_handle,
    _native_model,
    _native_model_sampling_space,
    _NativeModelOverlay,
    _sampling_space_runtime,
)


def _provider_lora_stack_family(*, model_only: bool, min_members: int) -> InputFamilySpec:
    inputs = [
        InputSpec("lora", _ASSET),
        InputSpec("strength_model", _FLOAT, required=False, default=1.0),
    ]
    if not model_only:
        inputs.append(InputSpec("strength_clip", _FLOAT, required=False, default=1.0))
    return InputFamilySpec(
        "loras",
        tuple(inputs),
        min_members=min_members,
        max_members=50,
        member_prefix="lora_",
    )


def _provider_diffusion_components_family() -> InputFamilySpec:
    return InputFamilySpec(
        "components",
        (
            InputSpec(
                "component",
                _ASSET,
                widget=AssetWidget(
                    accept=("application/octet-stream",),
                    kind="model/diffusion",
                ),
            ),
            InputSpec("role", _STRING, widget=StringWidget()),
        ),
        min_members=1,
        max_members=64,
        member_prefix="component_",
    )


def _generation_provider_text_schema(node_type: str, display_name: str) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="text",
        inputs=(
            InputSpec("clip", _DINKSTER_CLIP, required=False, lazy=True),
            InputSpec(
                "provider",
                _COMBO,
                required=False,
                widget=ComboWidget(remote_route="/api/choices/dinkster.generation.providers"),
                advanced=True,
            ),
            InputSpec(
                "prompt",
                _STRING,
                default="",
                widget=StringWidget(multiline=True, dynamic_prompts=True),
            ),
            InputSpec("image", _DINKSTER_IMAGE, required=False),
            InputSpec("video", _DINKSTER_IMAGE, required=False),
            InputSpec("audio", _COMFY_AUDIO, required=False),
            InputSpec("max_length", _INT, default=512),
            InputSpec("thinking", _BOOLEAN, required=False, default=False),
            InputSpec("use_default_template", _BOOLEAN, required=False, default=True),
            InputSpec("system_prompt", _STRING, required=False, force_input=True),
        ),
        combos=(
            DynamicComboSpec(
                "sampling_mode",
                options=(
                    DynamicComboOption(
                        "on",
                        inputs=(
                            InputSpec("temperature", _FLOAT, default=0.7),
                            InputSpec("top_k", _INT, default=64),
                            InputSpec("top_p", _FLOAT, default=0.95),
                            InputSpec("min_p", _FLOAT, default=0.05),
                            InputSpec("repetition_penalty", _FLOAT, default=1.05),
                            InputSpec("seed", _INT, default=0),
                            InputSpec(
                                "presence_penalty",
                                _FLOAT,
                                required=False,
                                default=0.0,
                            ),
                        ),
                    ),
                    DynamicComboOption("off"),
                ),
                default="on",
            ),
        ),
        outputs=(OutputSpec("generated_text", _STRING), OutputSpec("thinking", _STRING)),
    )


_GENERATION_SCHEMAS = {
    schema.node_type: schema for node in GENERATION_NODES if (schema := node.schema())
}


def _generation_provider_schema(node_type: str) -> NodeSchema:
    """Provider-side copies of the universal generation signatures."""
    owner = _GENERATION_SCHEMAS.get(node_type)
    if owner is not None:
        return owner
    if node_type == "dinkster.empty_trellis2_latent_structure":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty TRELLIS.2 Latent Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec(
                    "batch_size",
                    _INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type in ("dinkster.trellis2_conditioning", "dinkster.pixal3d_conditioning"):
        pixal3d = node_type == "dinkster.pixal3d_conditioning"
        inputs = [
            InputSpec("clip_vision_model", _DINKSTER_CLIP_VISION),
            InputSpec("image", _DINKSTER_IMAGE),
        ]
        if pixal3d:
            inputs.append(
                InputSpec(
                    "camera_angle_x",
                    _FLOAT,
                    default=49.13,
                    widget=NumberWidget(min=1.0, max=170.0, step=0.01),
                )
            )
        return NodeSchema(
            node_type=node_type,
            display_name="Pixal3D Conditioning" if pixal3d else "TRELLIS.2 Conditioning",
            category="model/conditioning/trellis2",
            inputs=tuple(inputs),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.vae_decode_structure_trellis2":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec(
                    "resolution",
                    _COMBO,
                    default="32",
                    widget=ComboWidget(options=("32", "64")),
                ),
            ),
            outputs=(OutputSpec("voxel", _COMFY_VOXEL),),
        )
    if node_type == "dinkster.trellis2_shape_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Shape Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("voxel", _COMFY_VOXEL),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.trellis2_upsample_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Upsample Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("shape_latent", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec(
                    "target_resolution",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=1024, max=2048, step=128),
                ),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.vae_decode_shape_trellis":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Shape",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(
                OutputSpec("mesh", _COMFY_MESH),
                OutputSpec("shape_subdivides", _COMFY_SHAPE_SUBDIVIDES),
            ),
        )
    if node_type == "dinkster.trellis2_texture_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Texture Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("shape_latent", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.vae_decode_texture_trellis":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Texture",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("shape_subdivides", _COMFY_SHAPE_SUBDIVIDES),
            ),
            outputs=(OutputSpec("voxel_colors", _COMFY_VOXEL),),
        )
    if node_type == "dinkster.load_checkpoint":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Checkpoint",
            category="model/loaders",
            inputs=(InputSpec("checkpoint", _ASSET),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
                OutputSpec("vae", _DINKSTER_VAE),
            ),
        )
    if node_type == "dinkster.load_checkpoint_stack":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Checkpoint Stack",
            category="model/loaders",
            inputs=(
                InputSpec("checkpoint", _ASSET),
                InputSpec("stop_at_clip_layer", _INT, default=-1),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=False, min_members=0),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
                OutputSpec("vae", _DINKSTER_VAE),
            ),
        )
    if node_type == "dinkster.load_diffusion_model":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Diffusion Model",
            category="model/loaders",
            inputs=(
                InputSpec("diffusion_model", _ASSET),
                InputSpec("weight_dtype", _COMBO, required=False, default="default"),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.load_diffusion_components":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Diffusion Components",
            category="model/loaders",
            description="Loads named diffusion components as one admitted model.",
            inputs=(InputSpec("weight_dtype", _COMBO, required=False, default="default"),),
            input_families=(_provider_diffusion_components_family(),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.load_ltxav_text_encoder":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LTX-2 Text Encoder",
            category="model/loaders",
            inputs=(
                InputSpec("text_encoder", _ASSET),
                InputSpec("ckpt_name", _ASSET),
                InputSpec("device", _COMBO, required=False, default="default"),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.load_ltxav_audio_vae":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LTX-2 Audio VAE",
            category="model/loaders",
            inputs=(InputSpec("ckpt_name", _ASSET),),
            outputs=(OutputSpec("audio_vae", _DINKSTER_VAE),),
        )
    if node_type == "dinkster.load_latent_upscale_model":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=(InputSpec("model_name", _ASSET),),
            outputs=(OutputSpec("upscale_model", _COMFY_LATENT_UPSCALE_MODEL),),
        )
    if node_type == "dinkster.ltxav_audio_vae_decode":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 Audio VAE Decode",
            category="model/latent/ltxv",
            inputs=(InputSpec("samples", _DINKSTER_LATENT), InputSpec("audio_vae", _DINKSTER_VAE)),
            outputs=(OutputSpec("audio", _COMFY_AUDIO),),
        )
    if node_type == "dinkster.load_lora":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LoRA (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("lora", _ASSET),
                InputSpec("strength_model", _FLOAT, required=False, default=1.0),
                InputSpec("strength_clip", _FLOAT, required=False, default=1.0),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
            ),
        )
    if node_type == "dinkster.load_lora_model_only":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LoRA",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("lora", _ASSET),
                InputSpec("strength_model", _FLOAT, required=False, default=1.0),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.apply_lora_stack":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply LoRA Stack (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=False, min_members=1),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
            ),
        )
    if node_type == "dinkster.apply_lora_stack_model_only":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply LoRA Stack",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=True, min_members=1),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.clip_text_encode":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode",
            category="model/conditioning",
            inputs=(
                InputSpec(
                    "text",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", _DINKSTER_CLIP),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.clip_text_encode_lumina2":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode (Lumina 2)",
            category="model/conditioning/lumina",
            inputs=(
                InputSpec(
                    "system_prompt",
                    _COMBO,
                    required=False,
                    default="superior",
                    widget=ComboWidget(options=("superior", "alignment")),
                ),
                InputSpec(
                    "user_prompt",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", _DINKSTER_CLIP),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.model_sampling_aura_flow":
        return NodeSchema(
            node_type=node_type,
            display_name="ModelSamplingAuraFlow",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec(
                    "shift",
                    _FLOAT,
                    required=False,
                    default=1.73,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.text_generate":
        return _generation_provider_text_schema(node_type, "Generate Text")
    if node_type == "dinkster.prompt_enhance":
        return _generation_provider_text_schema(node_type, "Enhance Prompt")
    if node_type == "dinkster.clip_set_last_layer":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Set Last Layer",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec(
                    "stop_at_clip_layer",
                    _INT,
                    default=-1,
                    widget=NumberWidget(min=-24, max=-1, step=1),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.t5_tokenizer_options":
        return NodeSchema(
            node_type=node_type,
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec(
                    "min_padding",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
                InputSpec(
                    "min_length",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.clip_text_encode_controlnet":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec(
                    "text",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.flux_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="FluxGuidance",
            category="model/conditioning",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("guidance", _FLOAT, default=3.5),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.flux_disable_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Flux Disable Guidance",
            category="model/conditioning/flux",
            inputs=(InputSpec("conditioning", _DINKSTER_CONDITIONING),),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.reference_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Set Reference Latent",
            category="model/conditioning",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("latent", _DINKSTER_LATENT, required=False),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.cfg_zero_star":
        return NodeSchema(
            node_type=node_type,
            display_name="CFGZeroStar",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.cfg_norm":
        return NodeSchema(
            node_type=node_type,
            display_name="CFGNorm",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("pre_cfg", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.tcfg":
        return NodeSchema(
            node_type=node_type,
            display_name="Tangential Damping CFG",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.fresca":
        return NodeSchema(
            node_type=node_type,
            display_name="FreSca",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scale_low", _FLOAT, default=1.0),
                InputSpec("scale_high", _FLOAT, default=1.25),
                InputSpec("freq_cutoff", _INT, default=20),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.context_windows_manual":
        return NodeSchema(
            node_type=node_type,
            display_name="Context Windows (Manual)",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=16),
                InputSpec("context_overlap", _INT, default=4),
                InputSpec("context_schedule", _COMBO, default="standard_static"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("dim", _INT, default=0),
                InputSpec("freenoise", _BOOLEAN, default=False),
                InputSpec("causal_window_fix", _BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.wan_context_windows_manual":
        return NodeSchema(
            node_type=node_type,
            display_name="WAN Context Windows (Manual)",
            category="model/patch/wan",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=81),
                InputSpec("context_overlap", _INT, default=30),
                InputSpec("context_schedule", _COMBO, default="standard_uniform"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("freenoise", _BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_context_windows":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Context Windows",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=145),
                InputSpec("context_overlap", _INT, default=40),
                InputSpec("context_schedule", _COMBO, default="standard_uniform"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("freenoise", _BOOLEAN, default=True),
                InputSpec("retain_first_frame", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.adaptive_projected_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Adaptive Projected Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("norm_threshold", _FLOAT, default=5.0),
                InputSpec("momentum", _FLOAT, default=0.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.mahiro_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Positive-Biased Guidance",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.epsilon_scaling":
        return NodeSchema(
            node_type=node_type,
            display_name="Epsilon Scaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scaling_factor", _FLOAT, default=1.005),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.cfg_override":
        return NodeSchema(
            node_type=node_type,
            display_name="CFG Override",
            category="model/sampling/guiders",
            description="Overrides CFG over a sampling-percent range.",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cfg", _FLOAT, default=1.0),
                InputSpec("start_percent", _FLOAT, default=0.0),
                InputSpec("end_percent", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.rescale_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="RescaleCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("multiplier", _FLOAT, default=0.7),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.renorm_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="RenormCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cfg_trunc", _FLOAT, default=100.0),
                InputSpec("renorm_cfg", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.temporal_score_rescaling":
        return NodeSchema(
            node_type=node_type,
            display_name="TSR - Temporal Score Rescaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("tsr_k", _FLOAT, default=0.95),
                InputSpec("tsr_sigma", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.nag":
        return NodeSchema(
            node_type=node_type,
            display_name="Normalized Attention Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("nag_scale", _FLOAT, default=5.0),
                InputSpec("nag_alpha", _FLOAT, default=0.5),
                InputSpec("nag_tau", _FLOAT, default=1.5),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxav_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 AV Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("frame_rate", _FLOAT, default=25.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxav_reference_audio":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 Reference Audio",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("reference_audio", _COMFY_AUDIO),
                InputSpec("audio_vae", _DINKSTER_VAE),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxav_id_lora_reference_audio":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Reference Audio (ID-LoRA)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("reference_audio", _COMFY_AUDIO),
                InputSpec("audio_vae", _DINKSTER_VAE, display_name="Audio VAE"),
                InputSpec("identity_guidance_scale", _FLOAT, default=3.0),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxv_spatiotemporal_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Spatio-Temporal Guidance (STG)",
            category="model/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scale", _FLOAT, default=1.0),
                InputSpec("blocks", _STRING, default="29"),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_modality_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Modality Guidance (A/V Coupling)",
            category="model/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("modality_scale", _FLOAT, default=3.0),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_duration_predictor":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Duration Predictor",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("duration_head", _COMFY_MODEL_PATCH),
                InputSpec("frame_rate", _FLOAT, default=24.0),
                InputSpec("min_seconds", _FLOAT, default=1.0),
                InputSpec("max_seconds", _FLOAT, default=20.0),
            ),
            outputs=(OutputSpec("num_frames", _INT), OutputSpec("seconds", _FLOAT)),
        )
    if node_type == "dinkster.ltxv_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("frame_rate", _FLOAT, default=25.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxv_image_to_video":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Image to Video",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("batch_size", _INT, default=1),
                InputSpec("strength", _FLOAT, default=1.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_image_to_video_inplace":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Image to Video (In-place)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("bypass", _BOOLEAN, required=False, default=False),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ltxv_add_guide":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Add Guide",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("frame_idx", _INT, default=0),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("attention_mask", _DINKSTER_MASK, required=False),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_add_latent_guide":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Add Latent Guide",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("guiding_latent", _DINKSTER_LATENT),
                InputSpec("latent_idx", _INT, default=0),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("attention_mask", _DINKSTER_MASK, required=False),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_freeze_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Freeze Latent",
            category="model/latent/ltxv",
            inputs=(InputSpec("latent", _DINKSTER_LATENT),),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ltxv_add_generated_keyframes":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Add Generated Keyframes",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("interval_frames", _INT, required=False, default=24),
                InputSpec("keyframes", _DINKSTER_LATENT, required=False),
                InputSpec("frame_indices", _STRING, required=False, default=""),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_separate_generated_keyframes":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Separate Generated Keyframes",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("keyframes_to_batch", _BOOLEAN, default=False),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
                OutputSpec("keyframes", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_generated_keyframes_to_guides":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Generated Keyframes to Guides",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("keyframes", _DINKSTER_LATENT),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("override_frame_indices", _STRING, required=False, default=""),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_crop_guides":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Crop Guides",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_latent_upsampler":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Latent Upsampler",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("upscale_model", _COMFY_LATENT_UPSCALE_MODEL),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.conditioning_merge":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Merge",
            category="model/conditioning/transform",
            combos=(
                DynamicComboSpec(
                    "mode",
                    options=(
                        DynamicComboOption(
                            "combine",
                            inputs=(
                                InputFamilySpec(
                                    "inputs",
                                    _DINKSTER_CONDITIONING,
                                    min_members=2,
                                    member_names=tuple(
                                        f"conditioning_{index}" for index in range(1, 9)
                                    ),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "average",
                            inputs=(
                                InputSpec("conditioning_to", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_from", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_to_strength", _FLOAT, default=1.0),
                            ),
                        ),
                        DynamicComboOption(
                            "concat",
                            inputs=(
                                InputSpec("conditioning_to", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_from", _DINKSTER_CONDITIONING),
                            ),
                        ),
                    ),
                    default="combine",
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_scale":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Scale",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("multiplier", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_area":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Area",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("strength", _FLOAT, default=1.0),
            ),
            combos=(
                DynamicComboSpec(
                    "units",
                    options=(
                        DynamicComboOption(
                            "pixels",
                            inputs=(
                                InputSpec("width", _INT, default=64),
                                InputSpec("height", _INT, default=64),
                                InputSpec("x", _INT, default=0),
                                InputSpec("y", _INT, default=0),
                            ),
                        ),
                        DynamicComboOption(
                            "percent",
                            inputs=(
                                InputSpec("width", _FLOAT, default=1.0),
                                InputSpec("height", _FLOAT, default=1.0),
                                InputSpec("x", _FLOAT, default=0.0),
                                InputSpec("y", _FLOAT, default=0.0),
                            ),
                        ),
                        DynamicComboOption(
                            "percent-video",
                            inputs=(
                                InputSpec("width", _FLOAT, default=1.0),
                                InputSpec("height", _FLOAT, default=1.0),
                                InputSpec("temporal", _FLOAT, default=1.0),
                                InputSpec("x", _FLOAT, default=0.0),
                                InputSpec("y", _FLOAT, default=0.0),
                                InputSpec("z", _FLOAT, default=0.0),
                            ),
                        ),
                    ),
                    default="pixels",
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_mask":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Mask",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("mask", _DINKSTER_MASK),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("set_cond_area", _COMBO, default="default"),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_timestep_range":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Timestep Range",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("start", _FLOAT, default=0.0),
                InputSpec("end", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_zero_out":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Zero Out",
            category="model/conditioning/transform",
            inputs=(InputSpec("conditioning", _DINKSTER_CONDITIONING),),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.chroma_radiance_options":
        return NodeSchema(
            node_type=node_type,
            display_name="Chroma Radiance Options",
            category="model/patch/chroma radiance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("preserve_wrapper", _BOOLEAN, required=False, default=True),
                InputSpec("start_sigma", _FLOAT, required=False, default=1.0, advanced=True),
                InputSpec("end_sigma", _FLOAT, required=False, default=0.0, advanced=True),
                InputSpec("nerf_tile_size", _INT, required=False, default=-1, advanced=True),
                InputSpec(
                    "force_sequential_txt_ids",
                    _BOOLEAN,
                    required=False,
                    default=False,
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.chroma_model_sampling":
        return NodeSchema(
            node_type=node_type,
            display_name="Chroma Model Sampling",
            category="model/patch/chroma",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("shift", _FLOAT, required=False, default=1.73),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.model_sampling_sd3":
        return NodeSchema(
            node_type=node_type,
            display_name="Model Sampling SD3",
            category="model/patch/stable diffusion",
            description="Applies discrete-flow sampling with a configurable shift.",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("shift", _FLOAT, default=3.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.model_sampling_flux":
        return NodeSchema(
            node_type=node_type,
            display_name="ModelSamplingFlux",
            category="model/patch/flux",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("max_shift", _FLOAT, default=1.15, advanced=True),
                InputSpec("base_shift", _FLOAT, default=0.5, advanced=True),
                InputSpec("width", _INT, default=1024),
                InputSpec("height", _INT, default=1024),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.empty_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Latent Image",
            category="model/latent",
            inputs=(
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_sd3_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty SD3 Latent Image",
            category="model/latent/stable diffusion",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_chroma_radiance_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Chroma Radiance Latent Image",
            category="model/latent/chroma radiance",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_flux2_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Flux 2 Latent",
            category="model/latent",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_ltxav_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty LTX-2 AV Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("frame_rate", _INT, default=25),
                InputSpec("batch_size", _INT, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_ltxv_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty LTX-Video Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("batch_size", _INT, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler":
        return NodeSchema(
            node_type=node_type,
            display_name="KSampler",
            category="model/sampling",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("seed", _INT, default=0),
                InputSpec("steps", _INT, default=20),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("sampler_name", _COMBO, default="dinkster.euler"),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler_advanced":
        return NodeSchema(
            node_type=node_type,
            display_name="KSampler (Advanced)",
            category="model/sampling",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("add_noise", _COMBO, default="enable"),
                InputSpec("noise_seed", _INT, default=0),
                InputSpec("steps", _INT, default=20),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("sampler_name", _COMBO, default="dinkster.euler"),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("start_at_step", _INT, default=0),
                InputSpec("end_at_step", _INT, default=10_000),
                InputSpec("return_with_leftover_noise", _COMBO, default="disable"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler_select":
        return NodeSchema(
            node_type=node_type,
            display_name="KSamplerSelect",
            category="model/sampling/samplers",
            inputs=(InputSpec("sampler_name", _COMBO, default="dinkster.euler"),),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_3m_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_3M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_2m_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_2M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="midpoint"),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("r", _FLOAT, default=0.5, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_2s_ancestral":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_2S_Ancestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("s_noise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_euler_ancestral":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerEulerAncestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_euler_ancestral_cfg_pp":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerEulerAncestralCFG++",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("s_noise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_lms":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerLMS",
            category="model/sampling/samplers",
            inputs=(InputSpec("order", _INT, default=4, advanced=True),),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpm_adaptative":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMAdaptative",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("order", _INT, default=3, advanced=True),
                InputSpec("rtol", _FLOAT, default=0.05, advanced=True),
                InputSpec("atol", _FLOAT, default=0.0078, advanced=True),
                InputSpec("h_init", _FLOAT, default=0.05, advanced=True),
                InputSpec("pcoeff", _FLOAT, default=0.0, advanced=True),
                InputSpec("icoeff", _FLOAT, default=1.0, advanced=True),
                InputSpec("dcoeff", _FLOAT, default=0.0, advanced=True),
                InputSpec("accept_safety", _FLOAT, default=0.81, advanced=True),
                InputSpec("eta", _FLOAT, default=0.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_er_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerER_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="ER-SDE"),
                InputSpec("max_stage", _INT, default=3, advanced=True),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_seeds_2":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerSEEDS2",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="phi_1"),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("r", _FLOAT, default=0.5, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_sa_solver":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerSASolver",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("sde_start_percent", _FLOAT, default=0.2, advanced=True),
                InputSpec("sde_end_percent", _FLOAT, default=0.8, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("predictor_order", _INT, default=3, advanced=True),
                InputSpec("corrector_order", _INT, default=4, advanced=True),
                InputSpec("use_pece", _BOOLEAN, default=False, advanced=True),
                InputSpec("simple_order_2", _BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.basic_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="BasicScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("steps", _INT, default=20),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.beta_sampling_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="BetaSamplingScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("steps", _INT, default=20),
                InputSpec("alpha", _FLOAT, default=0.6, advanced=True),
                InputSpec("beta", _FLOAT, default=0.6, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.sd_turbo_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="SDTurboScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("steps", _INT, default=1),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.karras_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="KarrasScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("rho", _FLOAT, default=7.0, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.exponential_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="ExponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.polyexponential_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="PolyexponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("rho", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.laplace_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="LaplaceScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("mu", _FLOAT, default=0.0, advanced=True),
                InputSpec("beta", _FLOAT, default=0.5, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.vp_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="VPScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("beta_d", _FLOAT, default=19.9, advanced=True),
                InputSpec("beta_min", _FLOAT, default=0.1, advanced=True),
                InputSpec("eps_s", _FLOAT, default=0.001, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.align_your_steps_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="AlignYourStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model_type", _COMBO, default="SD1"),
                InputSpec("steps", _INT, default=10),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.gits_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="GITSScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("coeff", _FLOAT, default=1.20, advanced=True),
                InputSpec("steps", _INT, default=10),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.optimal_steps_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="OptimalStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model_type", _COMBO, default="FLUX"),
                InputSpec("steps", _INT, default=20),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.flux2_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="Flux2Scheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("width", _INT, default=1024),
                InputSpec("height", _INT, default=1024),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.manual_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="ManualSigmas",
            category="model/sampling/sigmas",
            inputs=(InputSpec("sigmas", _STRING, default="1, 0.5"),),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.split_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="SplitSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("step", _INT, default=0),
            ),
            outputs=(
                OutputSpec("high_sigmas", _DINKSTER_SIGMAS),
                OutputSpec("low_sigmas", _DINKSTER_SIGMAS),
            ),
        )
    if node_type == "dinkster.split_sigmas_denoise":
        return NodeSchema(
            node_type=node_type,
            display_name="SplitSigmasDenoise",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(
                OutputSpec("high_sigmas", _DINKSTER_SIGMAS),
                OutputSpec("low_sigmas", _DINKSTER_SIGMAS),
            ),
        )
    if node_type == "dinkster.flip_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="FlipSigmas",
            category="model/sampling/sigmas",
            inputs=(InputSpec("sigmas", _DINKSTER_SIGMAS),),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.set_first_sigma":
        return NodeSchema(
            node_type=node_type,
            display_name="SetFirstSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("sigma", _FLOAT, default=136.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.extend_intermediate_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="ExtendIntermediateSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("steps", _INT, default=2),
                InputSpec("start_at_sigma", _FLOAT, default=-1.0),
                InputSpec("end_at_sigma", _FLOAT, default=12.0),
                InputSpec("spacing", _COMBO, default="linear"),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.sampling_percent_to_sigma":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplingPercentToSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("sampling_percent", _FLOAT, default=0.0),
                InputSpec("return_actual_sigma", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("sigma_value", _FLOAT),),
        )
    if node_type == "dinkster.basic_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Basic Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("cfg", _FLOAT, default=8.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.dual_cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cond1", _DINKSTER_CONDITIONING),
                InputSpec("cond2", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("cfg_conds", _FLOAT, default=8.0),
                InputSpec("cfg_cond2_negative", _FLOAT, default=8.0),
                InputSpec("style", _COMBO, default="regular"),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.ltxv_dual_cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("video_cfg", _FLOAT, default=3.0),
                InputSpec("audio_cfg", _FLOAT, default=7.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.perp_neg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Perp-Neg Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("empty_conditioning", _DINKSTER_CONDITIONING),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("neg_scale", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.disable_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="DisableNoise",
            category="model/sampling/noise",
            outputs=(OutputSpec("noise", _DINKSTER_NOISE),),
        )
    if node_type == "dinkster.random_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="RandomNoise",
            category="model/sampling/noise",
            inputs=(InputSpec("noise_seed", _INT, default=0),),
            outputs=(OutputSpec("noise", _DINKSTER_NOISE),),
        )
    if node_type == "dinkster.add_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="AddNoise",
            category="model/sampling/noise",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("noise", _DINKSTER_NOISE),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.sampler_custom":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerCustom",
            category="model/sampling/custom",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("add_noise", _BOOLEAN, default=True),
                InputSpec("noise_seed", _INT, default=0),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("sampler", _DINKSTER_SAMPLER),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("denoise_mask", _DINKSTER_MASK, required=False),
                InputSpec("inpaint", _DINKSTER_INPAINT_CONDITIONING, required=False),
                InputSpec("negative_inpaint", _DINKSTER_INPAINT_CONDITIONING, required=False),
                InputSpec("noise_inds", _INT_LIST, required=False),
                InputSpec("context_windows", _DINKSTER_CONTEXT_WINDOWS, required=False),
            ),
            outputs=(
                OutputSpec("output", _DINKSTER_LATENT),
                OutputSpec("denoised_output", _DINKSTER_LATENT),
            ),
            emits_previews=True,
        )
    if node_type == "dinkster.sampler_custom_advanced":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerCustomAdvanced",
            category="model/sampling/custom",
            inputs=(
                InputSpec("noise", _DINKSTER_NOISE),
                InputSpec("guider", _DINKSTER_GUIDER),
                InputSpec("sampler", _DINKSTER_SAMPLER),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("denoise_mask", _DINKSTER_MASK, required=False),
                InputSpec("inpaint", _DINKSTER_INPAINT_CONDITIONING, required=False),
                InputSpec("negative_inpaint", _DINKSTER_INPAINT_CONDITIONING, required=False),
                InputSpec("noise_inds", _INT_LIST, required=False),
                InputSpec("context_windows", _DINKSTER_CONTEXT_WINDOWS, required=False),
            ),
            outputs=(
                OutputSpec("output", _DINKSTER_LATENT),
                OutputSpec("denoised_output", _DINKSTER_LATENT),
            ),
            emits_previews=True,
        )
    if node_type == "dinkster.vae_decode":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Decode",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("image", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.vae_decode_tiled":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Decode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("tile_size", _INT, default=512, advanced=True),
                InputSpec("overlap", _INT, default=64, advanced=True),
                InputSpec("temporal_size", _INT, default=64, advanced=True),
                InputSpec("temporal_overlap", _INT, default=8, advanced=True),
            ),
            outputs=(OutputSpec("image", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.vae_encode":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Encode",
            category="model/latent",
            inputs=(
                InputSpec("pixels", _DINKSTER_IMAGE),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.vae_encode_tiled":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Encode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("pixels", _DINKSTER_IMAGE),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("tile_size", _INT, default=512, advanced=True),
                InputSpec("overlap", _INT, default=64, advanced=True),
                InputSpec("temporal_size", _INT, default=64, advanced=True),
                InputSpec("temporal_overlap", _INT, default=8, advanced=True),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.seedvr2_preprocess":
        return NodeSchema(
            node_type=node_type,
            display_name="Pre-Process SeedVR2 Input",
            category="image/pre-processors",
            inputs=(InputSpec("resized_images", _DINKSTER_IMAGE),),
            outputs=(OutputSpec("images", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.seedvr2_postprocess":
        return NodeSchema(
            node_type=node_type,
            display_name="Post-Process SeedVR2 Output",
            category="image/post-processors",
            inputs=(
                InputSpec("images", _DINKSTER_IMAGE),
                InputSpec("original_resized_images", _DINKSTER_IMAGE),
                InputSpec("color_correction_method", _COMBO, default="lab"),
            ),
            outputs=(OutputSpec("images", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.seedvr2_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply SeedVR2 Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("vae_conditioning", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.seedvr2_temporal_chunk":
        return NodeSchema(
            node_type=node_type,
            display_name="Split SeedVR2 Latent",
            category="model/latent/batch",
            inputs=(
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("temporal_overlap", _INT, default=0),
            ),
            combos=(
                DynamicComboSpec(
                    "chunking_mode",
                    options=(
                        DynamicComboOption("auto"),
                        DynamicComboOption(
                            "manual",
                            (InputSpec("frames_per_chunk", _INT, default=21),),
                        ),
                    ),
                    default="auto",
                ),
            ),
            outputs=(
                OutputSpec("latents", _DINKSTER_LATENT_LIST),
                OutputSpec("temporal_overlap", _INT),
            ),
        )
    if node_type == "dinkster.seedvr2_temporal_merge":
        return NodeSchema(
            node_type=node_type,
            display_name="Merge SeedVR2 Latents",
            category="model/latent/batch",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT_LIST),
                InputSpec("temporal_overlap", _INT, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.combine":
        return NodeSchema(
            node_type=node_type,
            display_name="Combine Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("operation", _COMBO, required=False, default="add"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.mix":
        return NodeSchema(
            node_type=node_type,
            display_name="Mix Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("operation", _COMBO, required=False, default="interpolate"),
                InputSpec("factor", _FLOAT, required=False, default=0.5),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.multiply":
        return NodeSchema(
            node_type=node_type,
            display_name="Multiply Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("multiplier", _FLOAT, required=False, default=1.0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.rotate":
        return NodeSchema(
            node_type=node_type,
            display_name="Rotate Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("angle", _COMBO, required=False, default="none"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.flip":
        return NodeSchema(
            node_type=node_type,
            display_name="Flip Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("axis", _COMBO, required=False, default="vertical"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.crop":
        return NodeSchema(
            node_type=node_type,
            display_name="Crop Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.resize":
        return NodeSchema(
            node_type=node_type,
            display_name="Resize Latent",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("method", _COMBO, required=False, default="nearest-exact"),
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("crop", _COMBO, required=False, default="disabled"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.resize_by":
        return NodeSchema(
            node_type=node_type,
            display_name="Resize Latent By",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("method", _COMBO, required=False, default="nearest-exact"),
                InputSpec("scale_by", _FLOAT, required=False, default=1.5),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.composite":
        return NodeSchema(
            node_type=node_type,
            display_name="Composite Latents",
            category="model/latent",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
                InputSpec("feather", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.composite_masked":
        return NodeSchema(
            node_type=node_type,
            display_name="Composite Latents (Masked)",
            category="model/latent",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
                InputSpec("resize_source", _BOOLEAN, required=False, default=False),
                InputSpec("mask", _DINKSTER_MASK, required=False),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.concat":
        return NodeSchema(
            node_type=node_type,
            display_name="Concat Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="x"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.cut":
        return NodeSchema(
            node_type=node_type,
            display_name="Cut Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="x"),
                InputSpec("index", _INT, required=False, default=0),
                InputSpec("amount", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.cut_to_batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Cut Latent to Batch",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="t"),
                InputSpec("slice_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.from_batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent From Batch",
            category="model/latent/batch",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("batch_index", _INT, required=False, default=0),
                InputSpec("length", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.repeat":
        return NodeSchema(
            node_type=node_type,
            display_name="Repeat Latent Batch",
            category="model/latent/batch",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("amount", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.seed_behavior":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Batch Seed Behavior",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("behavior", _COMBO, required=False, default="fixed"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Batch Latents",
            category="model/latent/batch",
            input_families=(
                InputFamilySpec(
                    "latents",
                    _DINKSTER_LATENT,
                    min_members=1,
                    max_members=50,
                    member_prefix="latent_",
                ),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.rebatch":
        return NodeSchema(
            node_type=node_type,
            display_name="Rebatch Latents",
            category="model/latent/batch",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT_LIST),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latents", _DINKSTER_LATENT_LIST),),
        )
    if node_type == "dinkster.latent.set_noise_mask":
        return NodeSchema(
            node_type=node_type,
            display_name="Set Latent Noise Mask",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("mask", _DINKSTER_MASK),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.replace_frames":
        return NodeSchema(
            node_type=node_type,
            display_name="Replace Video Latent Frames",
            category="model/latent/batch",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT, required=False),
                InputSpec("index", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.apply_operation":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Apply Operation",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("operation", _DINKSTER_LATENT_OPERATION),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.operation_tonemap_reinhard":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Operation Tonemap Reinhard",
            category="model/latent/advanced/operations",
            inputs=(InputSpec("multiplier", _FLOAT, required=False, default=1.0),),
            outputs=(OutputSpec("operation", _DINKSTER_LATENT_OPERATION),),
        )
    if node_type == "dinkster.latent.operation_sharpen":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Operation Sharpen",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("sharpen_radius", _INT, required=False, default=9),
                InputSpec("sigma", _FLOAT, required=False, default=1.0),
                InputSpec("alpha", _FLOAT, required=False, default=0.1),
            ),
            outputs=(OutputSpec("operation", _DINKSTER_LATENT_OPERATION),),
        )
    if node_type == "dinkster.latent.apply_operation_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Apply Operation CFG",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("operation", _DINKSTER_LATENT_OPERATION),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.latent.generate_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="Generate Noise",
            category="model/latent/advanced",
            inputs=(
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("batch_size", _INT, required=False, default=1),
                InputSpec("seed", _INT, required=False, default=123),
                InputSpec("multiplier", _FLOAT, required=False, default=1.0),
                InputSpec("constant_batch_noise", _BOOLEAN, required=False, default=False),
                InputSpec("normalize", _BOOLEAN, required=False, default=False),
                InputSpec("model", _DINKSTER_MODEL, required=False),
                InputSpec("sigmas", _DINKSTER_SIGMAS, required=False),
                InputSpec("latent_channels", _COMBO, required=False, default="4"),
                InputSpec("shape", _COMBO, required=False, default="BCHW"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.inject_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="Inject Noise To Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT),
                InputSpec("strength", _FLOAT, required=False, default=0.1),
                InputSpec("noise", _DINKSTER_LATENT),
                InputSpec("normalize", _BOOLEAN, required=False, default=False),
                InputSpec("average", _BOOLEAN, required=False, default=False),
                InputSpec("mask", _DINKSTER_MASK, required=False),
                InputSpec("mix_randn_amount", _FLOAT, required=False, default=0.0),
                InputSpec("seed", _INT, required=False, default=123),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    raise ValueError(f"unknown generation provider node type {node_type!r}")


def _require_provider_runtime(value: object, input_id: str) -> NativeRuntimeHandle:
    inference = importlib.import_module("dinkster_inference")
    inference.require_inference_runtime_handle(value, input_id)
    return _native_handle(value, input_id)


def _runtime_sampling_shift(
    runtime: object,
    sampling_shift: float | None,
) -> float | None:
    if sampling_shift is None:
        return None
    if not getattr(runtime, "supports_sampling_shift", False):
        raise TypeError("runtime does not support a sampling shift")
    return sampling_shift


def _bind_sampling_shift(
    method: Callable[..., Any], sampling_shift: float | None
) -> Callable[..., Any]:
    if sampling_shift is None:
        return method
    return partial(method, sampling_shift=sampling_shift)


@dataclass(frozen=True)
class _ShiftedCustomSamplingRuntime:
    shift: float
    runtime: object

    @property
    def _space(self) -> Any:
        inference = importlib.import_module("dinkster_inference")
        return inference.FlowSigmas(shift=self.shift)

    def custom_sampling_sigmas(
        self,
        scheduler_id: str,
        steps: int,
        denoise: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        inference = importlib.import_module("dinkster_inference")
        if device is None:
            scheduler = _inference_registries(inference).schedulers.get(scheduler_id)
            if scheduler is None:
                raise ValueError(f"unknown scheduler {scheduler_id!r}")
            return inference.sampling_sigmas(scheduler, self._space, steps, denoise=denoise)
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        scheduler = schedules.torch_scheduler_registry().get(scheduler_id)
        if scheduler is None:
            raise ValueError(f"unknown scheduler {scheduler_id!r}")
        return inference.sampling_sigmas(
            schedules.scheduler_on_device(scheduler, device),
            self._space,
            steps,
            denoise=denoise,
        )

    def custom_sampling_beta_sigmas(
        self,
        steps: int,
        alpha: float,
        beta: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.custom_beta_sigmas(self._space, steps, alpha, beta, device=device)

    def custom_sampling_sd_turbo_sigmas(
        self, steps: int, denoise: float, *, device: object | None = None
    ) -> tuple[float, ...]:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.sd_turbo_sigmas(self._space, steps, denoise, device=device)

    def custom_sampling_percent_to_sigma(
        self,
        percent: float,
        *,
        return_actual_sigma: bool,
    ) -> float:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.custom_percent_to_sigma(
            self._space,
            self._space.percent_to_sigma,
            percent,
            return_actual_sigma=return_actual_sigma,
        )

    def custom_sampling_add_noise(self, latent: object, noise: object, sigma: float) -> object:
        add_noise = getattr(self.runtime, "custom_sampling_add_noise", None)
        if not callable(add_noise):
            raise TypeError("model runtime does not support AddNoise")
        return add_noise(latent, noise, sigma)


def _require_custom_sampling_runtime(
    value: object, node_name: str
) -> tuple[Any, float | None, object]:
    model, applications = _application_chain_model(value, "model")
    if applications:
        raise ValueError(f"{node_name} does not accept a model application chain")
    sampling_space = _native_model_sampling_space(model)
    if sampling_space is not None:
        handle = _native_handle(model, "model")
        return _sampling_space_runtime(handle.runtime, sampling_space), None, handle.load_device
    sampling_shift = None
    if isinstance(model, _NativeModelOverlay):
        (
            handle,
            overlays,
            resolvers,
            control,
            sampling_shift,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        if handle.recipe.family_id in (inference.CHROMA.id, inference.CHROMA_RADIANCE.id):
            if (
                overlays
                or resolvers
                or control is not None
                or transforms
                or context_windows is not None
                or (sampling_shift is None and not radiance_options)
            ):
                raise ValueError(f"{node_name} does not accept a model overlay")
            descriptor = _active_inference_registries().components.get(handle.recipe.family_id)
            if descriptor is None or descriptor.execution_options is None:
                raise TypeError("model must be a native Chroma diffusion component")
            sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
            return (
                _component_runtime_with_options(
                    runtime,
                    descriptor,
                    handle.recipe.family_id,
                    handle.recipe.runtime_identity,
                    sampling_runtime.assembled.compute_dtype("diffusion"),
                    sampling_shift,
                    radiance_options,
                ),
                None,
                handle.load_device,
            )
        if (
            control is not None
            and handle.recipe.family_id == inference.Z_IMAGE_CONFIG.family_id
            and sampling_shift is None
            and not overlays
            and not resolvers
            and not transforms
            and context_windows is None
            and not radiance_options
        ):
            if handle.recipe.runtime_identity != runtime.runtime_identity or not isinstance(
                runtime, inference.CustomSamplingRuntime
            ):
                raise TypeError("model must be a native custom sampling runtime")
            return runtime, None, handle.load_device
        if (
            sampling_shift is not None
            and getattr(runtime, "supports_sampling_shift", False)
            and not overlays
            and not resolvers
            and control is None
            and not transforms
            and context_windows is None
            and not radiance_options
        ):
            if handle.recipe.runtime_identity != runtime.runtime_identity or not isinstance(
                runtime, inference.CustomSamplingRuntime
            ):
                raise TypeError("model must be a native custom sampling runtime")
            return (
                runtime,
                _runtime_sampling_shift(runtime, sampling_shift),
                handle.load_device,
            )
        if (
            overlays
            or resolvers
            or control is not None
            or transforms
            or context_windows is not None
            or radiance_options
        ):
            raise ValueError(f"{node_name} does not accept a model overlay")
        runtime = handle.runtime
        if sampling_shift is None:
            raise ValueError(f"{node_name} does not accept a model overlay")
        if handle.recipe.family_id == inference.LUMINA2_CONFIG.family_id:
            model = handle
        elif not isinstance(
            runtime, inference.CustomSamplingRuntime
        ) or not inference.is_flow_parameterization(runtime.family.sampling.parameterization):
            raise TypeError("ModelSamplingSD3 requires a flow custom-sampling runtime")
        else:
            return _ShiftedCustomSamplingRuntime(sampling_shift, runtime), None, handle.load_device
    handle = _require_provider_runtime(model, "model")
    inference = importlib.import_module("dinkster_inference")
    runtime = _minimax_h3_schedule_runtime(handle, inference)
    if runtime is None:
        runtime = handle.runtime
    if not isinstance(runtime, inference.CustomSamplingRuntime):
        raise TypeError(f"model family {runtime.family.id!r} does not support custom sampling")
    return runtime, sampling_shift, handle.load_device


def _require_base_custom_sampling_runtime(
    value: object, node_name: str
) -> tuple[Any, float | None, object]:
    model, _applications = _application_chain_model(value, "model")
    return _require_custom_sampling_runtime(model, node_name)


def _materialize_provider_conditioning(
    value: object, input_id: str, handle: NativeRuntimeHandle
) -> list[list[object]]:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    conditioning = inference_torch.materialize_basic_conditioning(
        value,
        device=handle.load_device,
    )
    metadata: dict[str, object] = {}
    if conditioning.pooled is not None:
        metadata["pooled_output"] = conditioning.pooled
    return [[conditioning.embeddings, metadata]]


def _prepare_provider_multistream_conditioning(
    value: object, input_id: str, runtime: Any, inference: Any
) -> list[list[object]]:
    prepare = getattr(runtime, "prepare_conditioning", None)
    if not callable(prepare):
        raise TypeError(f"{input_id} runtime prepare_conditioning must be callable")
    carrier, frame_rate = _split_ltx_frame_rate(value, inference)
    payload = prepare(carrier) if frame_rate is None else prepare(carrier, frame_rate=frame_rate)
    return [
        [
            inference.PreparedMultiStreamConditioning(runtime.conditioning_identity, payload),
            {},
        ]
    ]


def _prepare_provider_conditioning(
    value: object, input_id: str, runtime: Any, inference: Any
) -> list[list[object]]:
    prepare = getattr(runtime, "prepare_single_stream_conditioning", None)
    if not callable(prepare):
        raise TypeError(f"{input_id} runtime prepare_single_stream_conditioning must be callable")
    conditioning = prepare(value)
    if not isinstance(conditioning, inference.Conditioning):
        raise TypeError(f"{input_id} runtime must prepare a Conditioning value")
    typed = cast("Any", conditioning)
    return [[typed.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]


def _generation_lora_mode(execution_mode: str) -> str:
    if execution_mode == "attach":
        raise ValueError(
            "generation LoRA providers cannot attach patches to canonical conditioning; "
            "use 'auto' or 'precalculate'"
        )
    return "precalculate" if execution_mode == "auto" else execution_mode
