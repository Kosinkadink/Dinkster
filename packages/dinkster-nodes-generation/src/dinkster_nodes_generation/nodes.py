"""Universal generation node schemas, independent of execution providers."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    AssetWidget,
    BooleanWidget,
    ComboOption,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputDescriptorsSpec,
    OutputProbeSpec,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .model3d import MODEL3D_GENERATION_NODES
from .trellis2 import TRELLIS2_NODES

MODEL_TYPE = "dinkster.model"
CLIP_TYPE = "dinkster.clip"
VAE_TYPE = "dinkster.vae"
CONDITIONING_TYPE = "dinkster.conditioning"
LATENT_TYPE = "dinkster.latent"
IMAGE_TYPE = "dinkster.image"
MASK_TYPE = "dinkster.mask"
SAMPLER_TYPE = "dinkster.sampler"
SIGMAS_TYPE = "dinkster.sigmas"
GUIDER_TYPE = "dinkster.guider"
NOISE_TYPE = "dinkster.noise"
LATENT_OPERATION_TYPE = "dinkster.latent-operation"
CONTROL_NET_TYPE = "comfy.CONTROL_NET"
AUDIO_TYPE = "comfy.AUDIO"
IMPACT_BASIC_PIPE_TYPE = "comfy.BASIC_PIPE"
MODEL_PATCH_TYPE = "comfy.MODEL_PATCH"
LATENT_UPSCALE_MODEL_TYPE = "comfy.LATENT_UPSCALE_MODEL"

ASSET = TypeExpr.concrete(ASSET_TYPE)
MODEL = TypeExpr.concrete(MODEL_TYPE)
CLIP = TypeExpr.concrete(CLIP_TYPE)
VAE = TypeExpr.concrete(VAE_TYPE)
CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
LATENT = TypeExpr.concrete(LATENT_TYPE)
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
MASK = TypeExpr.concrete(MASK_TYPE)
SAMPLER = TypeExpr.concrete(SAMPLER_TYPE)
SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
GUIDER = TypeExpr.concrete(GUIDER_TYPE)
NOISE = TypeExpr.concrete(NOISE_TYPE)
LATENT_OPERATION = TypeExpr.concrete(LATENT_OPERATION_TYPE)
CONTROL_NET = TypeExpr.concrete(CONTROL_NET_TYPE)
CURVE = TypeExpr.concrete("dinkster.curve")
AUDIO = TypeExpr.concrete(AUDIO_TYPE)
IMPACT_BASIC_PIPE = TypeExpr.concrete(IMPACT_BASIC_PIPE_TYPE)
MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)
LATENT_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
LATENT_LIST = TypeExpr.list_of(LATENT)

SAMPLER_IDS = (
    "dinkster.ar_video",
    "dinkster.euler",
    "dinkster.euler_cfg_pp",
    "dinkster.euler_ancestral",
    "dinkster.euler_ancestral_cfg_pp",
    "dinkster.heun",
    "dinkster.heunpp2",
    "dinkster.exp_heun_2_x0",
    "dinkster.exp_heun_2_x0_sde",
    "dinkster.dpm_2",
    "dinkster.dpm_2_ancestral",
    "dinkster.lms",
    "dinkster.dpm_fast",
    "dinkster.dpm_adaptive",
    "dinkster.dpmpp_2s_ancestral",
    "dinkster.dpmpp_2s_ancestral_cfg_pp",
    "dinkster.dpmpp_sde",
    "dinkster.dpmpp_sde_gpu",
    "dinkster.dpmpp_2m",
    "dinkster.dpmpp_2m_cfg_pp",
    "dinkster.dpmpp_2m_sde",
    "dinkster.dpmpp_2m_sde_gpu",
    "dinkster.dpmpp_2m_sde_heun",
    "dinkster.dpmpp_2m_sde_heun_gpu",
    "dinkster.dpmpp_3m_sde",
    "dinkster.dpmpp_3m_sde_gpu",
    "dinkster.ddpm",
    "dinkster.lcm",
    "dinkster.ipndm",
    "dinkster.ipndm_v",
    "dinkster.deis",
    "dinkster.res_multistep",
    "dinkster.res_multistep_cfg_pp",
    "dinkster.res_multistep_ancestral",
    "dinkster.res_multistep_ancestral_cfg_pp",
    "dinkster.gradient_estimation",
    "dinkster.gradient_estimation_cfg_pp",
    "dinkster.er_sde",
    "dinkster.seeds_2",
    "dinkster.seeds_3",
    "dinkster.sa_solver",
    "dinkster.sa_solver_pece",
    "dinkster.ddim",
    "dinkster.uni_pc",
    "dinkster.uni_pc_bh2",
    "res4lyf.res_2m",
    "res4lyf.res_3m",
    "res4lyf.res_2s",
    "res4lyf.res_3s",
    "res4lyf.res_5s",
    "res4lyf.res_6s",
    "res4lyf.res_2m_ode",
    "res4lyf.res_3m_ode",
    "res4lyf.res_2s_ode",
    "res4lyf.res_3s_ode",
    "res4lyf.res_5s_ode",
    "res4lyf.res_6s_ode",
    "res4lyf.deis_2m",
    "res4lyf.deis_3m",
    "res4lyf.deis_2m_ode",
    "res4lyf.deis_3m_ode",
    "res4lyf.rk_beta",
)
SCHEDULER_IDS = (
    "dinkster.simple",
    "dinkster.sgm_uniform",
    "dinkster.karras",
    "dinkster.exponential",
    "dinkster.ddim_uniform",
    "dinkster.beta",
    "dinkster.normal",
    "dinkster.linear_quadratic",
    "dinkster.kl_optimal",
    "res4lyf.bong_tangent",
    "res4lyf.beta57",
)
SAMPLER_CHOICES = tuple(
    ComboOption(value=option_id, label=option_id.split(".", 1)[1]) for option_id in SAMPLER_IDS
)
SCHEDULER_CHOICES = tuple(
    ComboOption(value=option_id, label=option_id.split(".", 1)[1]) for option_id in SCHEDULER_IDS
)


def _conditioning_batching_inputs() -> tuple[InputSpec, InputSpec]:
    return (
        InputSpec(
            "conditioning_batching",
            COMBO,
            default="auto",
            widget=ComboWidget(options=("auto", "force-separate", "max-fused-lanes")),
            doc="Choose automatic memory sizing, separate evaluation, or a fixed lane cap.",
        ),
        InputSpec(
            "max_fused_lanes",
            INT,
            default=2,
            widget=NumberWidget(min=1, max=4096, step=1),
            doc="Maximum lanes fused when conditioning_batching is max_fused_lanes.",
        ),
    )


LORA_EXECUTION_MODES = ("auto", "precalculate")
LATENT_RESIZE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
CONTEXT_SCHEDULE_CHOICES = ("standard_static", "standard_uniform", "looped_uniform", "batched")
CONTEXT_FUSE_CHOICES = ("flat", "pyramid", "overlap-linear")
GENERATION_PROVIDER_CHOICE_ID = "dinkster.generation.providers"
CONTROL_NET_UNION_TYPES = (
    "auto",
    "openpose",
    "depth",
    "hed/pidi/scribble/ted",
    "canny/lineart/anime_lineart/mlsd",
    "normal",
    "segment",
    "tile",
    "repaint",
)


class _SchemaOnlyNode(Node):
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError(f"{cls.schema().node_type} requires an execution provider")


_USDU_REDRAW_MODES = ("Linear", "Chess", "None")
_USDU_SEAM_MODES = ("None", "Band Pass", "Half Tile", "Half Tile + Intersections")


def _usdu_combo(input_id: str, options: tuple[str, ...], default: str) -> InputSpec:
    return InputSpec(
        input_id,
        COMBO,
        default=default,
        widget=ComboWidget(options=tuple(ComboOption(value=value) for value in options)),
    )


def _usdu_number(
    input_id: str,
    input_type: TypeExpr,
    default: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> InputSpec:
    return InputSpec(
        input_id,
        input_type,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _usdu_inputs(*, include_upscale: bool) -> tuple[InputSpec, ...]:
    image_id = "image" if include_upscale else "upscaled_image"
    inputs: list[InputSpec] = [
        InputSpec(image_id, IMAGE),
        InputSpec("model", MODEL),
        InputSpec("positive", CONDITIONING),
        InputSpec("negative", CONDITIONING),
        InputSpec("vae", VAE),
    ]
    if include_upscale:
        inputs.append(_usdu_number("upscale_by", FLOAT, 2.0, minimum=0.05, maximum=4.0, step=0.05))
    inputs.extend(
        (
            _usdu_number("seed", INT, 0, minimum=0, maximum=2**53 - 1, step=1),
            _usdu_number("steps", INT, 20, minimum=1, maximum=10_000, step=1),
            _usdu_number("cfg", FLOAT, 8.0, minimum=0.0, maximum=100.0, step=0.1),
            _usdu_combo("sampler_name", ("euler",), "euler"),
            _usdu_combo("scheduler", ("simple",), "simple"),
            _usdu_number("denoise", FLOAT, 0.2, minimum=0.0, maximum=1.0, step=0.01),
        )
    )
    if include_upscale:
        inputs.append(InputSpec("upscale_model", ASSET))
    inputs.extend(
        (
            _usdu_combo("mode_type", _USDU_REDRAW_MODES, "Linear"),
            _usdu_number("tile_width", INT, 512, minimum=64, maximum=8192, step=8),
            _usdu_number("tile_height", INT, 512, minimum=64, maximum=8192, step=8),
            _usdu_number("mask_blur", INT, 8, minimum=0, maximum=64, step=1),
            _usdu_number("tile_padding", INT, 32, minimum=0, maximum=8192, step=8),
            _usdu_combo("seam_fix_mode", _USDU_SEAM_MODES, "None"),
            _usdu_number("seam_fix_denoise", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
            _usdu_number("seam_fix_width", INT, 64, minimum=0, maximum=8192, step=8),
            _usdu_number("seam_fix_mask_blur", INT, 8, minimum=0, maximum=64, step=1),
            _usdu_number("seam_fix_padding", INT, 16, minimum=0, maximum=8192, step=8),
            InputSpec("force_uniform_tiles", BOOLEAN, default=True),
            InputSpec("tiled_decode", BOOLEAN, default=False),
            _usdu_number("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
        )
    )
    return tuple(inputs)


class UltimateSDUpscaleOwner(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.compat.ultimate_sd_upscale",
            display_name="Ultimate SD Upscale Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=_usdu_inputs(include_upscale=True),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("UltimateSDUpscale",),
            search_visibility="hidden",
        )


class UltimateSDUpscaleNoUpscaleOwner(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.compat.ultimate_sd_upscale_no_upscale",
            display_name="Ultimate SD Upscale No-Upscale Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=_usdu_inputs(include_upscale=False),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("UltimateSDUpscaleNoUpscale",),
            search_visibility="hidden",
        )


class UpscaleModelLoaderOwner(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.compat.upscale_model_loader",
            display_name="Upscale Model Loader Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=(
                InputSpec(
                    "model_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/upscaler",
                    ),
                ),
            ),
            outputs=(OutputSpec("upscale_model", ASSET),),
            aliases=("UpscaleModelLoader",),
            search_visibility="hidden",
        )


GENERATION_COMPAT_CARRIER_NODES: tuple[type[Node], ...] = (
    UltimateSDUpscaleOwner,
    UltimateSDUpscaleNoUpscaleOwner,
    UpscaleModelLoaderOwner,
)


class LoadCheckpoint(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_checkpoint",
            display_name="Load Checkpoint",
            category="model/loaders",
            description=("Loads a model, text encoder, and codec from one checkpoint asset."),
            inputs=(
                InputSpec(
                    "checkpoint",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/checkpoint",
                    ),
                ),
            ),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("clip", CLIP),
                OutputSpec("vae", VAE),
            ),
            aliases=("CheckpointLoaderSimple",),
            search_terms=("checkpoint", "model loader"),
        )


class LoadControlNet(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_controlnet",
            display_name="Load ControlNet Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "control_net_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/controlnet",
                    ),
                ),
            ),
            outputs=(OutputSpec("control_net", CONTROL_NET),),
            aliases=("ControlNetLoader",),
            search_terms=("controlnet", "control net", "model loader"),
        )


class ApplyControlNet(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_controlnet",
            display_name="Apply ControlNet (DEPRECATED)",
            category="model/conditioning/controlnet",
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec("control_net", CONTROL_NET),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ControlNetApply",),
            search_terms=("controlnet", "control net", "conditioning"),
        )


class ApplyControlNetAdvanced(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_controlnet_advanced",
            display_name="Apply ControlNet",
            category="model/conditioning/controlnet",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("control_net", CONTROL_NET),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("vae", VAE, required=False),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("ControlNetApplyAdvanced",),
            search_terms=("controlnet", "control net", "conditioning"),
        )


class SetControlNetUnionType(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.set_controlnet_union_type",
            display_name="Set Union ControlNet Type",
            category="model/conditioning/controlnet",
            description=(
                "Selects the SD ControlNet union mode. The grouped labels map to the shared "
                "mode indices documented by SD_CONTROL_MODE_INDEX."
            ),
            inputs=(
                InputSpec("control_net", CONTROL_NET),
                InputSpec(
                    "type",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=CONTROL_NET_UNION_TYPES),
                ),
            ),
            outputs=(OutputSpec("control_net", CONTROL_NET),),
            aliases=("SetUnionControlNetType",),
            search_terms=("controlnet", "control net", "union type"),
        )


class LoadModelProfile(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_model_profile",
            display_name="Load Model",
            category="model/loaders",
            description="Loads the components detected in a model asset.",
            inputs=(
                InputSpec(
                    "checkpoint",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/checkpoint",
                    ),
                ),
                InputSpec(
                    "entries",
                    STRING,
                    default=(
                        '{"entries":[{"id":"model","name":"MODEL","type":"model"}],'
                        '"assetDigest":"","detectorRevision":"1","shapeDigest":"",'
                        '"components":{},"diagnostics":[]}'
                    ),
                    widget=StringWidget(multiline=True),
                    display_name="Outputs",
                ),
            ),
            output_descriptors=OutputDescriptorsSpec(
                input="entries",
                choices=(
                    OutputSpec("model", MODEL),
                    OutputSpec("clip", CLIP),
                    OutputSpec("vae", VAE),
                ),
                max_entries=3,
                min_entries=1,
                fixed_ids=True,
                probe=OutputProbeSpec(input="checkpoint", kind="model", revision="1"),
            ),
            search_terms=("checkpoint", "model loader", "automatic loader"),
        )


def _lora_stack_family(*, model_only: bool, min_members: int) -> InputFamilySpec:
    inputs = [
        InputSpec(
            "lora",
            ASSET,
            widget=AssetWidget(
                accept=("application/octet-stream",),
                kind="model/lora",
            ),
        ),
        InputSpec(
            "strength_model",
            FLOAT,
            required=False,
            default=1.0,
            widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
        ),
    ]
    if not model_only:
        inputs.append(
            InputSpec(
                "strength_clip",
                FLOAT,
                required=False,
                default=1.0,
                widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
            )
        )
    return InputFamilySpec(
        "loras",
        tuple(inputs),
        min_members=min_members,
        max_members=50,
        member_prefix="lora_",
    )


class LoadCheckpointStack(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_checkpoint_stack",
            display_name="Load Checkpoint Stack",
            category="model/loaders",
            description=(
                "Loads one checkpoint and applies an ordered LoRA stack before "
                "selecting the CLIP layer."
            ),
            inputs=(
                InputSpec(
                    "checkpoint",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/checkpoint",
                    ),
                ),
                InputSpec(
                    "stop_at_clip_layer",
                    INT,
                    default=-1,
                    widget=NumberWidget(min=-24, max=-1, step=1),
                    advanced=True,
                ),
                InputSpec(
                    "execution_mode",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                    hidden=True,
                ),
            ),
            input_families=(_lora_stack_family(model_only=False, min_members=0),),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("clip", CLIP),
                OutputSpec("vae", VAE),
            ),
            search_terms=("checkpoint", "lora stack", "combined loader"),
        )


class LoadDiffusionModel(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_diffusion_model",
            display_name="Load Diffusion Model",
            category="model/loaders",
            description="Loads an admitted diffusion model from a content-addressed asset.",
            inputs=(
                InputSpec(
                    "diffusion_model",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/diffusion",
                    ),
                ),
                InputSpec(
                    "weight_dtype",
                    COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(
                        options=("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")
                    ),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("UNETLoader",),
            search_terms=("unet", "dit", "diffusion model"),
        )


class LoadDiffusionComponents(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_diffusion_components",
            display_name="Load Diffusion Components",
            category="model/loaders",
            description="Loads named diffusion components as one admitted model.",
            inputs=(
                InputSpec(
                    "weight_dtype",
                    COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(
                        options=("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")
                    ),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "components",
                    (
                        InputSpec(
                            "component",
                            ASSET,
                            widget=AssetWidget(
                                accept=("application/octet-stream",),
                                kind="model/diffusion",
                            ),
                        ),
                        InputSpec("role", STRING, widget=StringWidget()),
                    ),
                    min_members=1,
                    max_members=64,
                    member_prefix="component_",
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("unet", "dit", "diffusion model", "split checkpoint"),
        )


class LoadLTXAVTextEncoder(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_ltxav_text_encoder",
            display_name="Load LTX-2 Text Encoder",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "text_encoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/text-encoder",
                    ),
                ),
                InputSpec(
                    "ckpt_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/checkpoint",
                    ),
                ),
                InputSpec(
                    "device",
                    COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(options=("default", "cpu")),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("clip", CLIP),),
            aliases=("LTXAVTextEncoderLoader",),
            search_terms=("ltx", "audio video", "gemma", "text encoder"),
        )


class LoadLTXAVAudioVAE(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_ltxav_audio_vae",
            display_name="Load LTX-2 Audio VAE",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "ckpt_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/checkpoint",
                    ),
                ),
            ),
            outputs=(OutputSpec("audio_vae", VAE),),
            aliases=("LTXVAudioVAELoader",),
            search_terms=("ltx", "audio", "vae", "vocoder"),
        )


class LoadLatentUpscaleModel(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_latent_upscale_model",
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/latent-upscaler",
                    ),
                ),
            ),
            outputs=(OutputSpec("upscale_model", LATENT_UPSCALE_MODEL),),
            aliases=("LatentUpscaleModelLoader",),
            search_terms=("ltx", "latent", "upscale", "loader"),
        )


class LTXAVAudioVAEDecode(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxav_audio_vae_decode",
            display_name="LTX-2 Audio VAE Decode",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("audio_vae", VAE, display_name="Audio VAE"),
            ),
            outputs=(OutputSpec("audio", AUDIO),),
            aliases=("LTXVAudioVAEDecode",),
            search_terms=("ltx", "audio", "decode", "vae", "vocoder"),
        )


class LoadLora(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_lora",
            display_name="Load LoRA (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("clip", CLIP),
                InputSpec(
                    "lora",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/lora",
                    ),
                ),
                InputSpec(
                    "strength_model",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "strength_clip",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "execution_mode",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                    hidden=True,
                ),
            ),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("clip", CLIP),
            ),
            aliases=("LoraLoader",),
            search_terms=("lora", "adapter"),
        )


class LoadLoraModelOnly(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_lora_model_only",
            display_name="Load LoRA",
            category="model/loaders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "lora",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/lora",
                    ),
                ),
                InputSpec(
                    "strength_model",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "execution_mode",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("LoraLoaderModelOnly",),
            search_terms=("lora", "adapter"),
        )


class ApplyLoraStack(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_lora_stack",
            display_name="Apply LoRA Stack (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("clip", CLIP),
                InputSpec(
                    "execution_mode",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                    hidden=True,
                ),
            ),
            input_families=(_lora_stack_family(model_only=False, min_members=1),),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("clip", CLIP),
            ),
            search_terms=("lora", "adapter", "stack"),
        )


class ApplyLoraStackModelOnly(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_lora_stack_model_only",
            display_name="Apply LoRA Stack",
            category="model/loaders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "execution_mode",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                    hidden=True,
                ),
            ),
            input_families=(_lora_stack_family(model_only=True, min_members=1),),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("lora", "adapter", "stack"),
        )


class CLIPTextEncode(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.clip_text_encode",
            display_name="CLIP Text Encode",
            category="model/conditioning",
            inputs=(
                InputSpec(
                    "text",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", CLIP),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("CLIPTextEncode",),
            search_terms=("text", "prompt", "conditioning"),
        )


class CLIPTextEncodeLumina2(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.clip_text_encode_lumina2",
            display_name="CLIP Text Encode (Lumina 2)",
            category="model/conditioning/lumina",
            inputs=(
                InputSpec(
                    "system_prompt",
                    COMBO,
                    required=False,
                    default="superior",
                    widget=ComboWidget(options=("superior", "alignment")),
                ),
                InputSpec(
                    "user_prompt",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", CLIP),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("CLIPTextEncodeLumina2",),
            search_terms=("lumina", "text", "prompt", "conditioning"),
        )


class ModelSamplingAuraFlow(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.model_sampling_aura_flow",
            display_name="ModelSamplingAuraFlow",
            category="model/patch",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "shift",
                    FLOAT,
                    required=False,
                    default=1.73,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ModelSamplingAuraFlow",),
            search_terms=("aura flow", "sampling", "shift"),
        )


def _text_generation_schema(
    node_type: str,
    display_name: str,
    alias: str,
    search_terms: tuple[str, ...],
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="text",
        description=(
            "Generates text with the native Anima Qwen3-0.6B component or an "
            "explicitly selected external provider. Unsupported provider options "
            "are refused rather than approximated."
        ),
        inputs=(
            InputSpec("clip", CLIP, required=False, lazy=True),
            InputSpec(
                "provider",
                COMBO,
                required=False,
                widget=ComboWidget(remote_route=f"/api/choices/{GENERATION_PROVIDER_CHOICE_ID}"),
                display_name="Service",
                hidden=True,
            ),
            InputSpec(
                "prompt",
                STRING,
                default="",
                widget=StringWidget(multiline=True, dynamic_prompts=True),
            ),
            InputSpec("image", IMAGE, required=False),
            InputSpec("video", IMAGE, required=False),
            InputSpec("audio", AUDIO, required=False),
            InputSpec(
                "max_length",
                INT,
                default=512,
                widget=NumberWidget(min=1, max=32_768, step=1),
            ),
            InputSpec("thinking", BOOLEAN, required=False, default=False),
            InputSpec(
                "use_default_template",
                BOOLEAN,
                required=False,
                default=False,
                advanced=True,
            ),
        ),
        combos=(
            DynamicComboSpec(
                "sampling_mode",
                options=(
                    DynamicComboOption(
                        "on",
                        inputs=(
                            InputSpec(
                                "temperature",
                                FLOAT,
                                default=0.7,
                                widget=NumberWidget(min=0.01, max=2.0, step=0.000001),
                            ),
                            InputSpec(
                                "top_k",
                                INT,
                                default=64,
                                widget=NumberWidget(min=0, max=1_000, step=1),
                            ),
                            InputSpec(
                                "top_p",
                                FLOAT,
                                default=0.95,
                                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                            ),
                            InputSpec(
                                "min_p",
                                FLOAT,
                                default=0.05,
                                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                            ),
                            InputSpec(
                                "repetition_penalty",
                                FLOAT,
                                default=1.05,
                                widget=NumberWidget(min=0.01, max=5.0, step=0.01),
                            ),
                            InputSpec(
                                "seed",
                                INT,
                                default=0,
                                widget=NumberWidget(
                                    min=0,
                                    max=2**53 - 1,
                                    step=1,
                                    control_after_generate="randomize",
                                ),
                            ),
                            InputSpec(
                                "presence_penalty",
                                FLOAT,
                                required=False,
                                default=0.0,
                                widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                            ),
                        ),
                    ),
                    DynamicComboOption("off"),
                ),
                default="on",
                display_name="Sampling Mode",
            ),
        ),
        outputs=(OutputSpec("generated_text", STRING),),
        aliases=(alias,),
        search_terms=search_terms,
    )


class TextGenerate(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _text_generation_schema(
            "dinkster.text_generate",
            "Generate Text",
            "TextGenerate",
            ("text", "generate", "llm", "qwen"),
        )


class PromptEnhance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _text_generation_schema(
            "dinkster.prompt_enhance",
            "Enhance Prompt",
            "TextGenerateLTX2Prompt",
            ("prompt", "enhance", "llm", "ltx", "qwen"),
        )


def generation_choices() -> dict[str, tuple[str, ...]]:
    return {GENERATION_PROVIDER_CHOICE_ID: ()}


class CLIPSetLastLayer(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.clip_set_last_layer",
            display_name="CLIP Set Last Layer",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec(
                    "stop_at_clip_layer",
                    INT,
                    default=-1,
                    widget=NumberWidget(min=-24, max=-1, step=1),
                ),
            ),
            outputs=(OutputSpec("clip", CLIP),),
            aliases=("CLIPSetLastLayer",),
            search_terms=("clip", "layer", "hidden state"),
        )


class T5TokenizerOptions(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.t5_tokenizer_options",
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec(
                    "min_padding",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
                InputSpec(
                    "min_length",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
            ),
            outputs=(OutputSpec("clip", CLIP),),
            aliases=("T5TokenizerOptions",),
            search_terms=("t5", "tokenizer", "padding", "length"),
        )


class CLIPTextEncodeControlnet(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.clip_text_encode_controlnet",
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec("conditioning", CONDITIONING),
                InputSpec(
                    "text",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("CLIPTextEncodeControlnet",),
            search_terms=("controlnet", "text", "prompt", "conditioning"),
        )


class FluxGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.flux_guidance",
            display_name="FluxGuidance",
            category="model/conditioning",
            description="Sets the distilled guidance strength on Flux-family conditioning.",
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec(
                    "guidance",
                    FLOAT,
                    default=3.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("FluxGuidance",),
            search_terms=("flux", "guidance", "conditioning"),
        )


class FluxDisableGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.flux_disable_guidance",
            display_name="Flux Disable Guidance",
            category="model/conditioning/flux",
            description="Disables the distilled guidance embed on Flux-family conditioning.",
            inputs=(InputSpec("conditioning", CONDITIONING),),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("FluxDisableGuidance",),
            search_terms=("flux", "disable", "guidance", "conditioning"),
        )


class ReferenceLatent(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.reference_latent",
            display_name="Set Reference Latent",
            category="model/conditioning",
            description=(
                "Adds an optional reference latent for image editing; chain for multiple images."
            ),
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec("latent", LATENT, required=False),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ReferenceLatent",),
            search_terms=("reference", "latent", "image edit", "conditioning"),
        )


class CFGZeroStar(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.cfg_zero_star",
            display_name="CFGZeroStar",
            category="advanced/guidance",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("CFGZeroStar",),
            search_terms=("cfg", "guidance", "zero star"),
        )


class CFGNorm(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.cfg_norm",
            display_name="CFGNorm",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec("pre_cfg", BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("CFGNorm",),
            search_terms=("cfg", "guidance", "normalize"),
        )


class TCFG(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.tcfg",
            display_name="Tangential Damping CFG",
            category="advanced/guidance",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("TCFG",),
            search_terms=("cfg", "guidance", "tangential", "damping"),
        )


class FreSca(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.fresca",
            display_name="FreSca",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "scale_low",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "scale_high",
                    FLOAT,
                    default=1.25,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "freq_cutoff", INT, default=20, widget=NumberWidget(min=1, max=10_000, step=1)
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("FreSca",),
            search_terms=("cfg", "guidance", "frequency", "scaling"),
        )


class LazyCache(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.lazy_cache",
            display_name="LazyCache",
            category="model/patch",
            description=(
                "Approximate sampling acceleration that reuses complete guided denoiser"
                " results when the estimated output change stays below the threshold."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "reuse_threshold",
                    FLOAT,
                    default=0.2,
                    widget=NumberWidget(min=0.0, max=3.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.15,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=0.95,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("LazyCache",),
            search_terms=("cache", "sampling", "acceleration", "approximate"),
        )


class EasyCache(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.easy_cache",
            display_name="EasyCache",
            category="model/patch",
            description=(
                "Approximate sampling acceleration that reuses each conditioning lane's"
                " model residual when the estimated output change stays below the threshold."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "reuse_threshold",
                    FLOAT,
                    default=0.2,
                    widget=NumberWidget(min=0.0, max=3.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.15,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=0.95,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("EasyCache",),
            search_terms=("cache", "sampling", "acceleration", "approximate"),
        )


class AttentionSchedule(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.attention_schedule",
            display_name="Attention Schedule",
            category="model/patch",
            description=(
                "Uses one selected approximate attention provider inside a sampling window and"
                " SDPA outside it. Sol tau may be driven by a Curve."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "approximate_provider",
                    COMBO,
                    default="sol",
                    widget=ComboWidget(
                        options=tuple(
                            ComboOption(value=value)
                            for value in ("sol", "sage", "comfy_kitchen_int8")
                        )
                    ),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "conditioning_sink",
                    COMBO,
                    default="off",
                    widget=ComboWidget(
                        options=tuple(ComboOption(value=value) for value in ("off", "exact_kv"))
                    ),
                ),
                InputSpec("sol_tau", CURVE, required=False),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=(
                "attention",
                "schedule",
                "curve",
                "sol",
                "sage",
                "sdpa",
                "conditioning sink",
            ),
        )


class ContextWindowsManual(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.context_windows_manual",
            display_name="Context Windows (Manual)",
            category="model/patch",
            description=(
                "Samples the latent in overlapping windows along one axis, fusing the"
                " per-window model outputs. Lengths are in latent-axis units."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "context_length",
                    INT,
                    default=16,
                    widget=NumberWidget(min=1, max=16384, step=1),
                ),
                InputSpec(
                    "context_overlap",
                    INT,
                    default=4,
                    widget=NumberWidget(min=0, max=16384, step=1),
                ),
                InputSpec(
                    "context_schedule",
                    COMBO,
                    default="standard_static",
                    widget=ComboWidget(options=CONTEXT_SCHEDULE_CHOICES),
                ),
                InputSpec(
                    "context_stride",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=32, step=1),
                ),
                InputSpec("closed_loop", BOOLEAN, default=False),
                InputSpec(
                    "fuse_method",
                    COMBO,
                    default="pyramid",
                    widget=ComboWidget(options=CONTEXT_FUSE_CHOICES),
                ),
                InputSpec("dim", INT, default=0, widget=NumberWidget(min=0, max=5, step=1)),
                InputSpec("freenoise", BOOLEAN, default=False),
                InputSpec("causal_window_fix", BOOLEAN, default=True, hidden=True),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ContextWindowsManual",),
            search_terms=("context", "windows", "sliding", "animatediff"),
        )


class WanContextWindowsManual(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_context_windows_manual",
            display_name="WAN Context Windows (Manual)",
            category="model/patch/wan",
            description=(
                "Context windows over the temporal axis of WAN video latents."
                " Lengths are in real frames and convert to latent frames."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "context_length",
                    INT,
                    default=81,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "context_overlap",
                    INT,
                    default=30,
                    widget=NumberWidget(min=0, max=16384, step=1),
                ),
                InputSpec(
                    "context_schedule",
                    COMBO,
                    default="standard_uniform",
                    widget=ComboWidget(options=CONTEXT_SCHEDULE_CHOICES),
                ),
                InputSpec(
                    "context_stride",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=32, step=1),
                    advanced=True,
                ),
                InputSpec("closed_loop", BOOLEAN, default=False, advanced=True),
                InputSpec(
                    "fuse_method",
                    COMBO,
                    default="pyramid",
                    widget=ComboWidget(options=CONTEXT_FUSE_CHOICES),
                ),
                InputSpec("freenoise", BOOLEAN, default=True, advanced=True),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("WanContextWindowsManual",),
            search_terms=("context", "windows", "wan", "video", "sliding"),
        )


class LTXVContextWindows(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_context_windows",
            display_name="LTXV Context Windows",
            category="model/patch",
            description=(
                "Context windows over the temporal axis of LTX-Video latents."
                " Lengths are in real frames and convert to latent frames."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "context_length",
                    INT,
                    default=145,
                    widget=NumberWidget(min=1, max=16384, step=8),
                ),
                InputSpec(
                    "context_overlap",
                    INT,
                    default=40,
                    widget=NumberWidget(min=0, max=16384, step=8),
                ),
                InputSpec(
                    "context_schedule",
                    COMBO,
                    default="standard_uniform",
                    widget=ComboWidget(options=CONTEXT_SCHEDULE_CHOICES),
                ),
                InputSpec(
                    "context_stride",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=32, step=1),
                    advanced=True,
                ),
                InputSpec("closed_loop", BOOLEAN, default=False, advanced=True),
                InputSpec(
                    "fuse_method",
                    COMBO,
                    default="pyramid",
                    widget=ComboWidget(options=CONTEXT_FUSE_CHOICES),
                ),
                InputSpec("freenoise", BOOLEAN, default=True, advanced=True),
                InputSpec("retain_first_frame", BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("LTXVContextWindows",),
            search_terms=("context", "windows", "ltxv", "video", "sliding"),
        )


class AdaptiveProjectedGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.adaptive_projected_guidance",
            display_name="Adaptive Projected Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "eta", FLOAT, default=1.0, widget=NumberWidget(min=-10.0, max=10.0, step=0.01)
                ),
                InputSpec(
                    "norm_threshold",
                    FLOAT,
                    default=5.0,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.1),
                ),
                InputSpec(
                    "momentum",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-5.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("APG",),
            search_terms=("apg", "cfg", "guidance", "projected"),
        )


class MahiroGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mahiro_guidance",
            display_name="Positive-Biased Guidance",
            category="advanced/guidance",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("Mahiro",),
            search_terms=("mahiro", "cfg", "guidance", "positive biased"),
        )


class EpsilonScaling(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.epsilon_scaling",
            display_name="Epsilon Scaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "scaling_factor",
                    FLOAT,
                    default=1.005,
                    widget=NumberWidget(min=0.5, max=1.5, step=0.001),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("Epsilon Scaling",),
            search_terms=("epsilon", "cfg", "guidance", "scaling"),
        )


class CFGOverride(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.cfg_override",
            display_name="CFG Override",
            category="model/sampling/guiders",
            description="Overrides CFG over a sampling-percent range.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("CFGOverride",),
            search_terms=("cfg", "guidance", "interval", "override"),
        )


class RescaleCFG(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.rescale_cfg",
            display_name="RescaleCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "multiplier",
                    FLOAT,
                    default=0.7,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("RescaleCFG",),
            search_terms=("cfg", "guidance", "rescale"),
        )


class RenormCFG(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.renorm_cfg",
            display_name="RenormCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "cfg_trunc",
                    FLOAT,
                    default=100.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "renorm_cfg",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("RenormCFG",),
            search_terms=("cfg", "guidance", "renorm", "lumina"),
        )


class TemporalScoreRescaling(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.temporal_score_rescaling",
            display_name="TSR - Temporal Score Rescaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "tsr_k",
                    FLOAT,
                    default=0.95,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.001),
                ),
                InputSpec(
                    "tsr_sigma",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("TemporalScoreRescaling",),
            search_terms=("tsr", "cfg", "guidance", "temporal", "rescaling"),
        )


class NormalizedAttentionGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.nag",
            display_name="Normalized Attention Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "nag_scale",
                    FLOAT,
                    default=5.0,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.1),
                ),
                InputSpec(
                    "nag_alpha",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "nag_tau",
                    FLOAT,
                    default=1.5,
                    widget=NumberWidget(min=1.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("NAGuidance",),
            search_terms=("nag", "attention", "guidance", "negative", "normalized"),
        )


class LTXAVConditioning(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxav_conditioning",
            display_name="LTX-2 AV Conditioning",
            category="model/conditioning",
            description=(
                "Sets the frame rate on LTX-2 audio-video text conditioning "
                "so sampling matches the intended playback speed."
            ),
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec(
                    "frame_rate",
                    FLOAT,
                    default=25.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            search_terms=("ltx", "frame rate", "conditioning"),
        )


class LTXAVReferenceAudio(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxav_reference_audio",
            display_name="LTX-2 Reference Audio",
            category="model/conditioning/ltxv",
            description="Encodes a reference audio clip into LTX-2 audio-video conditioning.",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("reference_audio", AUDIO),
                InputSpec("audio_vae", VAE),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            search_terms=("ltx", "audio", "reference", "speaker"),
        )


class LTXAVIDLoRAReferenceAudio(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxav_id_lora_reference_audio",
            display_name="LTXV Reference Audio (ID-LoRA)",
            category="model/conditioning/ltxv",
            description=(
                "Encodes speaker reference audio and applies no-reference identity guidance "
                "for LTX-2 ID-LoRA models."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("reference_audio", AUDIO),
                InputSpec("audio_vae", VAE, display_name="Audio VAE"),
                InputSpec(
                    "identity_guidance_scale",
                    FLOAT,
                    default=3.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("LTXVReferenceAudio",),
            search_terms=("ltx", "audio", "reference", "speaker", "identity", "id-lora"),
        )


class LTXVSpatioTemporalGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_spatiotemporal_guidance",
            display_name="LTXV Spatio-Temporal Guidance (STG)",
            category="model/guidance",
            description="Guides away from value-passthrough self-attention in selected blocks.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "scale",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec("blocks", STRING, default="29", widget=StringWidget(multiline=False)),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("LTXVSpatioTemporalGuidance",),
            search_terms=("ltx", "stg", "spatio temporal", "guidance"),
        )


class LTXVModalityGuidance(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_modality_guidance",
            display_name="LTXV Modality Guidance (A/V Coupling)",
            category="model/guidance",
            description="Strengthens LTX-2 audio-video coupling with an additional decoupled pass.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "modality_scale",
                    FLOAT,
                    default=3.0,
                    widget=NumberWidget(min=1.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("LTXVModalityGuidance",),
            search_terms=("ltx", "audio video", "coupling", "guidance"),
        )


class LTXVDurationPredictor(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_duration_predictor",
            display_name="LTXV Duration Predictor",
            category="model/conditioning/ltxv",
            description="Predicts prompt duration and snaps it to the LTX causal frame grid.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("duration_head", MODEL_PATCH),
                InputSpec(
                    "frame_rate",
                    FLOAT,
                    default=24.0,
                    widget=NumberWidget(min=1.0, max=120.0, step=0.01),
                ),
                InputSpec(
                    "min_seconds",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.5, max=120.0, step=0.1),
                ),
                InputSpec(
                    "max_seconds",
                    FLOAT,
                    default=20.0,
                    widget=NumberWidget(min=0.5, max=120.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("num_frames", INT), OutputSpec("seconds", FLOAT)),
            aliases=("LTXVDurationPredictor",),
            search_terms=("ltx", "duration", "frames", "auto duration"),
        )


class LTXVConditioning(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_conditioning",
            display_name="LTX-Video Conditioning",
            category="model/conditioning",
            description=(
                "Sets the frame rate on LTX-Video text conditioning so "
                "sampling matches the intended playback speed."
            ),
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec(
                    "frame_rate",
                    FLOAT,
                    default=25.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("LTXVConditioning",),
            search_terms=("ltx", "frame rate", "conditioning"),
        )


class LTXVImageToVideo(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_image_to_video",
            display_name="LTX-Video Image to Video",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec("image", IMAGE),
                InputSpec(
                    "width",
                    INT,
                    default=768,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    INT,
                    default=97,
                    widget=NumberWidget(min=9, max=16384, step=8),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("LTXVImgToVideo",),
            search_terms=("ltx", "image to video", "initial frame"),
        )


class LTXVImageToVideoInplace(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_image_to_video_inplace",
            display_name="LTX-Video Image to Video (In-place)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("vae", VAE),
                InputSpec("image", IMAGE),
                InputSpec("latent", LATENT),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec("bypass", BOOLEAN, required=False, default=False),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("LTXVImgToVideoInplace",),
            search_terms=("ltx", "image to video", "initial frame"),
        )


class LTXVAddGuide(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_add_guide",
            display_name="LTX-Video Add Guide",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec("latent", LATENT),
                InputSpec("image", IMAGE),
                InputSpec(
                    "frame_idx",
                    INT,
                    default=0,
                    widget=NumberWidget(min=-9999, max=9999, step=1),
                ),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec("attention_mask", MASK, required=False),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("LTXVAddGuide",),
            search_terms=("ltx", "guide", "first frame", "last frame"),
        )


class LTXVCropGuides(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_crop_guides",
            display_name="LTX-Video Crop Guides",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("latent", LATENT),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("LTXVCropGuides",),
            search_terms=("ltx", "guide", "crop"),
        )


class LTXVLatentUpsampler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_latent_upsampler",
            display_name="LTXV Latent Upsampler",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("upscale_model", LATENT_UPSCALE_MODEL),
                InputSpec("vae", VAE),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("LTXVLatentUpsampler",),
            search_terms=("ltx", "latent", "upscale", "spatial"),
        )


class ConditioningMerge(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_merge",
            display_name="Conditioning Merge",
            category="model/conditioning/transform",
            description=(
                "Merges conditioning: combine keeps every record side by "
                "side, average blends token payloads by strength, and "
                "concat joins them along the token axis."
            ),
            combos=(
                DynamicComboSpec(
                    "mode",
                    options=(
                        DynamicComboOption(
                            "combine",
                            inputs=(
                                InputFamilySpec(
                                    "inputs",
                                    CONDITIONING,
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
                                InputSpec("conditioning_to", CONDITIONING),
                                InputSpec("conditioning_from", CONDITIONING),
                                InputSpec(
                                    "conditioning_to_strength",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "concat",
                            inputs=(
                                InputSpec("conditioning_to", CONDITIONING),
                                InputSpec("conditioning_from", CONDITIONING),
                            ),
                        ),
                    ),
                    default="combine",
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=(
                "ConditioningCombine",
                "ConditioningAverage",
                "ConditioningConcat",
            ),
            search_terms=("combine", "average", "concat", "merge", "blend prompts"),
        )


class ConditioningScale(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_scale",
            display_name="Conditioning Scale",
            category="model/conditioning/transform",
            description="Multiplies conditioning payloads by a scalar.",
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec(
                    "multiplier",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ConditioningMultiply",),
            search_terms=("multiply", "scale conditioning", "scale prompt"),
        )


class ConditioningSetArea(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_set_area",
            display_name="Conditioning Set Area",
            category="model/conditioning/transform",
            description=(
                "Restricts conditioning to a rectangle, in pixels (snapped "
                "to the latent grid), as fractions of the image size, or to "
                "a fractional spatiotemporal video region."
            ),
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "units",
                    options=(
                        DynamicComboOption(
                            "pixels",
                            inputs=(
                                InputSpec(
                                    "width",
                                    INT,
                                    default=64,
                                    widget=NumberWidget(min=64, max=16384, step=8),
                                ),
                                InputSpec(
                                    "height",
                                    INT,
                                    default=64,
                                    widget=NumberWidget(min=64, max=16384, step=8),
                                ),
                                InputSpec(
                                    "x",
                                    INT,
                                    default=0,
                                    widget=NumberWidget(min=0, max=16384, step=8),
                                ),
                                InputSpec(
                                    "y",
                                    INT,
                                    default=0,
                                    widget=NumberWidget(min=0, max=16384, step=8),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "percent",
                            inputs=(
                                InputSpec(
                                    "width",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "height",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "x",
                                    FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "y",
                                    FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "percent-video",
                            inputs=(
                                InputSpec(
                                    "width",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "height",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "temporal",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "x",
                                    FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "y",
                                    FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "z",
                                    FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                            ),
                        ),
                    ),
                    default="pixels",
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=(
                "ConditioningSetArea",
                "ConditioningSetAreaPercentage",
                "ConditioningSetAreaPercentageVideo",
            ),
            search_terms=("regional prompt", "area prompt", "spatial conditioning"),
        )


class ConditioningSetMask(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_set_mask",
            display_name="Conditioning Set Mask",
            category="model/conditioning/transform",
            description="Restricts conditioning to a mask.",
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec("mask", MASK),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "set_cond_area",
                    COMBO,
                    default="default",
                    widget=ComboWidget(options=("default", "mask bounds")),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ConditioningSetMask",),
            search_terms=("masked prompt", "mask conditioning"),
        )


class ConditioningSetTimestepRange(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_set_timestep_range",
            display_name="Conditioning Set Timestep Range",
            category="model/conditioning/transform",
            description=(
                "Limits conditioning to a normalized sampling window; both endpoints are included."
            ),
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec(
                    "start",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ConditioningSetTimestepRange",),
            search_terms=("timestep", "schedule conditioning", "range"),
        )


class ConditioningZeroOut(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.conditioning_zero_out",
            display_name="Conditioning Zero Out",
            category="model/conditioning/transform",
            description="Zeroes every conditioning payload in place of the prompt.",
            inputs=(InputSpec("conditioning", CONDITIONING),),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("ConditioningZeroOut",),
            search_terms=("null conditioning", "clear conditioning", "negative"),
        )


class ChromaRadianceOptions(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.chroma_radiance_options",
            display_name="Chroma Radiance Options",
            category="model/patch/chroma radiance",
            description="Sets advanced Chroma Radiance execution options over a sigma range.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("preserve_wrapper", BOOLEAN, required=False, default=True),
                InputSpec(
                    "start_sigma",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_sigma",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "nerf_tile_size",
                    INT,
                    required=False,
                    default=-1,
                    widget=NumberWidget(min=-1),
                    advanced=True,
                ),
                InputSpec(
                    "force_sequential_txt_ids",
                    BOOLEAN,
                    required=False,
                    default=False,
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ChromaRadianceOptions",),
            search_terms=("chroma radiance", "nerf tile", "sequential text ids"),
        )


class ChromaModelSampling(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.chroma_model_sampling",
            display_name="Chroma Model Sampling",
            category="model/patch/chroma",
            description="Applies AuraFlow sampling with a configurable flow shift to Chroma.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "shift",
                    FLOAT,
                    required=False,
                    default=1.73,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("chroma", "aura flow", "sampling shift"),
        )


class ModelSamplingSD3(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.model_sampling_sd3",
            display_name="Model Sampling SD3",
            category="model/patch/stable diffusion",
            description="Applies discrete-flow sampling with a configurable shift.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "shift",
                    FLOAT,
                    default=3.0,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ModelSamplingSD3",),
            search_terms=("sd3", "flow", "sampling shift"),
        )


class ModelSamplingLTXV(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.model_sampling_ltxv",
            display_name="ModelSamplingLTXV",
            category="model/patch/ltxv",
            description="Applies a latent-token-dependent flow shift to LTX sampling.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "max_shift",
                    FLOAT,
                    default=2.05,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "base_shift",
                    FLOAT,
                    default=0.95,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec("latent", LATENT, required=False),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ModelSamplingLTXV",),
            search_terms=("ltx", "video", "sampling shift"),
        )


class ModelSamplingFlux(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.model_sampling_flux",
            display_name="ModelSamplingFlux",
            category="model/patch/flux",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "max_shift",
                    FLOAT,
                    default=1.15,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "base_shift",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "width",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ModelSamplingFlux",),
        )


class EmptyLatentImage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_latent_image",
            display_name="Empty Latent Image",
            category="model/latent",
            description="Creates a zeroed image latent for generation.",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=16, max=16_384, step=8),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=16, max=16_384, step=8),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyLatentImage",),
            search_terms=("empty", "latent", "noise"),
        )


class EmptySD3LatentImage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_sd3_latent_image",
            display_name="Empty SD3 Latent Image",
            category="model/latent/stable diffusion",
            description="Creates a zeroed 16-channel image latent at one-eighth resolution.",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptySD3LatentImage",),
            search_terms=("empty", "latent", "sd3", "chroma"),
        )


class EmptyChromaRadianceLatentImage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_chroma_radiance_latent_image",
            display_name="Empty Chroma Radiance Latent Image",
            category="model/latent/chroma radiance",
            description="Creates a zeroed RGB pixel latent for Chroma Radiance generation.",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyChromaRadianceLatentImage",),
            search_terms=("empty", "latent", "chroma radiance", "pixel space"),
        )


class EmptyFlux2LatentImage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_flux2_latent_image",
            display_name="Empty Flux 2 Latent",
            category="model/latent",
            description="Creates a zeroed packed latent for Flux2 generation.",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384, step=16),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyFlux2LatentImage",),
            search_terms=("empty", "latent", "flux2"),
        )


class EmptyLTXAVLatent(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_ltxav_latent",
            display_name="Empty LTX-2 AV Latent",
            category="latent/multi-stream",
            description=(
                "Creates blank video and audio latent streams sized for one "
                "LTX-2 clip. The latent geometry is model-derived, so the "
                "node takes the loaded model rather than hardcoding one "
                "checkpoint's constants."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "width",
                    INT,
                    default=768,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    INT,
                    default=97,
                    widget=NumberWidget(min=1, max=9999, step=8),
                ),
                InputSpec(
                    "frame_rate",
                    INT,
                    default=25,
                    widget=NumberWidget(min=1, max=240, step=1),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )


class EmptyLTXVLatent(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_ltxv_latent",
            display_name="Empty LTX-Video Latent",
            category="latent/multi-stream",
            description=(
                "Creates a blank video latent stream sized for one LTX-Video "
                "clip. The latent geometry is model-derived, so the node "
                "takes the loaded model rather than hardcoding one "
                "checkpoint's constants."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "width",
                    INT,
                    default=768,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    INT,
                    default=97,
                    widget=NumberWidget(min=1, max=9999, step=8),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )


class KSampler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler",
            display_name="KSampler",
            category="model/sampling",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec(
                    "steps",
                    INT,
                    default=20,
                    widget=NumberWidget(min=1, max=10_000),
                ),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "sampler_name",
                    COMBO,
                    default="dinkster.euler",
                    widget=ComboWidget(options=SAMPLER_CHOICES),
                ),
                InputSpec(
                    "scheduler",
                    COMBO,
                    default="dinkster.simple",
                    widget=ComboWidget(options=SCHEDULER_CHOICES),
                ),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("latent_image", LATENT),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("KSampler",),
            search_terms=("sample", "denoise", "generate"),
            emits_previews=True,
        )


class KSamplerAdvanced(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler_advanced",
            display_name="KSampler (Advanced)",
            category="model/sampling",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "add_noise",
                    COMBO,
                    default="enable",
                    widget=ComboWidget(options=("enable", "disable")),
                ),
                InputSpec(
                    "noise_seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "sampler_name",
                    COMBO,
                    default="dinkster.euler",
                    widget=ComboWidget(options=SAMPLER_CHOICES),
                ),
                InputSpec(
                    "scheduler",
                    COMBO,
                    default="dinkster.simple",
                    widget=ComboWidget(options=SCHEDULER_CHOICES),
                ),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("latent_image", LATENT),
                InputSpec(
                    "start_at_step",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000),
                ),
                InputSpec(
                    "end_at_step",
                    INT,
                    default=10_000,
                    widget=NumberWidget(min=0, max=10_000),
                ),
                InputSpec(
                    "return_with_leftover_noise",
                    COMBO,
                    default="disable",
                    widget=ComboWidget(options=("disable", "enable")),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("KSamplerAdvanced",),
            search_terms=("sample", "denoise", "step range"),
            emits_previews=True,
        )


class KSamplerSelect(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler_select",
            display_name="KSamplerSelect",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "sampler_name",
                    COMBO,
                    default="dinkster.euler",
                    widget=ComboWidget(options=SAMPLER_CHOICES),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("KSamplerSelect",),
        )


class SamplerDPMPP3MSDE(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_dpmpp_3m_sde",
            display_name="SamplerDPMPP_3M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "noise_device",
                    COMBO,
                    default="gpu",
                    widget=ComboWidget(options=("gpu", "cpu")),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerDPMPP_3M_SDE",),
        )


class SamplerDPMPP2MSDE(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_dpmpp_2m_sde",
            display_name="SamplerDPMPP_2M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "solver_type",
                    COMBO,
                    default="midpoint",
                    widget=ComboWidget(options=("midpoint", "heun")),
                ),
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "noise_device",
                    COMBO,
                    default="gpu",
                    widget=ComboWidget(options=("gpu", "cpu")),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerDPMPP_2M_SDE",),
        )


class SamplerDPMPPSDE(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_dpmpp_sde",
            display_name="SamplerDPMPP_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "r",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "noise_device",
                    COMBO,
                    default="gpu",
                    widget=ComboWidget(options=("gpu", "cpu")),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerDPMPP_SDE",),
        )


class SamplerDPMPP2SAncestral(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_dpmpp_2s_ancestral",
            display_name="SamplerDPMPP_2S_Ancestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerDPMPP_2S_Ancestral",),
        )


class SamplerEulerAncestral(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_euler_ancestral",
            display_name="SamplerEulerAncestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerEulerAncestral",),
        )


class SamplerEulerAncestralCFGPP(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_euler_ancestral_cfg_pp",
            display_name="SamplerEulerAncestralCFG++",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerEulerAncestralCFGPP",),
        )


class SamplerLMS(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_lms",
            display_name="SamplerLMS",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "order",
                    INT,
                    default=4,
                    widget=NumberWidget(min=1, max=100),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerLMS",),
        )


class SamplerDPMAdaptative(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_dpm_adaptative",
            display_name="SamplerDPMAdaptative",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("order", INT, default=3, widget=NumberWidget(min=2, max=3)),
                InputSpec(
                    "rtol",
                    FLOAT,
                    default=0.05,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "atol",
                    FLOAT,
                    default=0.0078,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "h_init",
                    FLOAT,
                    default=0.05,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "pcoeff",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "icoeff",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "dcoeff",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "accept_safety",
                    FLOAT,
                    default=0.81,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "eta",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerDPMAdaptative",),
        )


class SamplerERSDE(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_er_sde",
            display_name="SamplerER_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "solver_type",
                    COMBO,
                    default="ER-SDE",
                    widget=ComboWidget(options=("ER-SDE", "Reverse-time SDE", "ODE")),
                ),
                InputSpec("max_stage", INT, default=3, widget=NumberWidget(min=1, max=3)),
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerER_SDE",),
            search_terms=("sde",),
        )


class SamplerSEEDS2(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_seeds_2",
            display_name="SamplerSEEDS2",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "solver_type",
                    COMBO,
                    default="phi_1",
                    widget=ComboWidget(options=("phi_1", "phi_2")),
                ),
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "r",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.01, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerSEEDS2",),
            search_terms=("sde", "exp heun"),
        )


class SamplerSASolver(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_sa_solver",
            display_name="SamplerSASolver",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "eta",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "sde_start_percent",
                    FLOAT,
                    default=0.2,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    advanced=True,
                ),
                InputSpec(
                    "sde_end_percent",
                    FLOAT,
                    default=0.8,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    advanced=True,
                ),
                InputSpec(
                    "s_noise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "predictor_order",
                    INT,
                    default=3,
                    widget=NumberWidget(min=1, max=6),
                    advanced=True,
                ),
                InputSpec(
                    "corrector_order",
                    INT,
                    default=4,
                    widget=NumberWidget(min=0, max=6),
                    advanced=True,
                ),
                InputSpec("use_pece", BOOLEAN, default=False, advanced=True),
                InputSpec("simple_order_2", BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
            aliases=("SamplerSASolver",),
            search_terms=("sde",),
        )


class BasicScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.basic_scheduler",
            display_name="BasicScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "scheduler",
                    COMBO,
                    default="dinkster.simple",
                    widget=ComboWidget(options=SCHEDULER_CHOICES),
                ),
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("BasicScheduler",),
        )


class BetaSamplingScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.beta_sampling_scheduler",
            display_name="BetaSamplingScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "alpha",
                    FLOAT,
                    default=0.6,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.01),
                ),
                InputSpec(
                    "beta",
                    FLOAT,
                    default=0.6,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("BetaSamplingScheduler",),
        )


class SDTurboScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sd_turbo_scheduler",
            display_name="SDTurboScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("steps", INT, default=1, widget=NumberWidget(min=1, max=10)),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("SDTurboScheduler",),
        )


class KarrasScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.karras_scheduler",
            display_name="KarrasScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "sigma_max",
                    FLOAT,
                    default=14.614642,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "sigma_min",
                    FLOAT,
                    default=0.0291675,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "rho",
                    FLOAT,
                    default=7.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("KarrasScheduler",),
        )


class ExponentialScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.exponential_scheduler",
            display_name="ExponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "sigma_max",
                    FLOAT,
                    default=14.614642,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "sigma_min",
                    FLOAT,
                    default=0.0291675,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("ExponentialScheduler",),
        )


class PolyexponentialScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.polyexponential_scheduler",
            display_name="PolyexponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "sigma_max",
                    FLOAT,
                    default=14.614642,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "sigma_min",
                    FLOAT,
                    default=0.0291675,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "rho",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("PolyexponentialScheduler",),
        )


class LaplaceScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.laplace_scheduler",
            display_name="LaplaceScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "sigma_max",
                    FLOAT,
                    default=14.614642,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "sigma_min",
                    FLOAT,
                    default=0.0291675,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "mu",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.1),
                ),
                InputSpec(
                    "beta",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("LaplaceScheduler",),
        )


class VPScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vp_scheduler",
            display_name="VPScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "beta_d",
                    FLOAT,
                    default=19.9,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "beta_min",
                    FLOAT,
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=5000.0, step=0.01),
                ),
                InputSpec(
                    "eps_s",
                    FLOAT,
                    default=0.001,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.0001),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("VPScheduler",),
        )


class AlignYourStepsScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.align_your_steps_scheduler",
            display_name="AlignYourStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec(
                    "model_type",
                    COMBO,
                    default="SD1",
                    widget=ComboWidget(options=("SD1", "SDXL", "SVD")),
                ),
                InputSpec("steps", INT, default=10, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("AlignYourStepsScheduler",),
            search_terms=("AYS scheduler",),
        )


class GITSScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.gits_scheduler",
            display_name="GITSScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec(
                    "coeff",
                    FLOAT,
                    default=1.20,
                    widget=NumberWidget(min=0.80, max=1.50, step=0.05),
                ),
                InputSpec("steps", INT, default=10, widget=NumberWidget(min=2, max=1000)),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("GITSScheduler",),
        )


class OptimalStepsScheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.optimal_steps_scheduler",
            display_name="OptimalStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec(
                    "model_type",
                    COMBO,
                    default="FLUX",
                    widget=ComboWidget(options=("FLUX", "Wan", "Chroma")),
                ),
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=3, max=1000)),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("OptimalStepsScheduler",),
        )


class Flux2Scheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.flux2_scheduler",
            display_name="Flux2Scheduler",
            category="model/sampling/schedulers",
            description="Empirical-mu flow schedule sized to the Flux2 image resolution.",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=4096)),
                InputSpec(
                    "width",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=16_384),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("Flux2Scheduler",),
        )


class Ideogram4Scheduler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ideogram4_scheduler",
            display_name="Ideogram 4 Scheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=200)),
                InputSpec(
                    "width",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=256, max=8192, step=16),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=256, max=8192, step=16),
                ),
                InputSpec(
                    "mu",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.05),
                ),
                InputSpec(
                    "std",
                    FLOAT,
                    default=1.75,
                    widget=NumberWidget(min=0.1, max=5.0, step=0.05),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("Ideogram4Scheduler",),
        )


class ManualSigmas(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.manual_sigmas",
            display_name="ManualSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec(
                    "sigmas",
                    STRING,
                    default="1, 0.5",
                    widget=StringWidget(multiline=False),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("ManualSigmas",),
            search_terms=("custom noise schedule", "define sigmas"),
        )


class SplitSigmas(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.split_sigmas",
            display_name="SplitSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", SIGMAS),
                InputSpec("step", INT, default=0, widget=NumberWidget(min=0, max=10_000)),
            ),
            outputs=(
                OutputSpec("high_sigmas", SIGMAS),
                OutputSpec("low_sigmas", SIGMAS),
            ),
            aliases=("SplitSigmas",),
        )


class SplitSigmasDenoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.split_sigmas_denoise",
            display_name="SplitSigmasDenoise",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", SIGMAS),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("high_sigmas", SIGMAS),
                OutputSpec("low_sigmas", SIGMAS),
            ),
            aliases=("SplitSigmasDenoise",),
        )


class FlipSigmas(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.flip_sigmas",
            display_name="FlipSigmas",
            category="model/sampling/sigmas",
            inputs=(InputSpec("sigmas", SIGMAS),),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("FlipSigmas",),
        )


class SetFirstSigma(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.set_first_sigma",
            display_name="SetFirstSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", SIGMAS),
                InputSpec(
                    "sigma",
                    FLOAT,
                    default=136.0,
                    widget=NumberWidget(min=0.0, max=20_000.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("SetFirstSigma",),
        )


class ExtendIntermediateSigmas(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.extend_intermediate_sigmas",
            display_name="ExtendIntermediateSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", SIGMAS),
                InputSpec("steps", INT, default=2, widget=NumberWidget(min=1, max=100)),
                InputSpec(
                    "start_at_sigma",
                    FLOAT,
                    default=-1.0,
                    widget=NumberWidget(min=-1.0, max=20_000.0, step=0.01),
                ),
                InputSpec(
                    "end_at_sigma",
                    FLOAT,
                    default=12.0,
                    widget=NumberWidget(min=0.0, max=20_000.0, step=0.01),
                ),
                InputSpec(
                    "spacing",
                    COMBO,
                    default="linear",
                    widget=ComboWidget(options=("linear", "cosine", "sine")),
                ),
            ),
            outputs=(OutputSpec("sigmas", SIGMAS),),
            aliases=("ExtendIntermediateSigmas",),
            search_terms=("interpolate sigmas",),
        )


class SamplingPercentToSigma(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampling_percent_to_sigma",
            display_name="SamplingPercentToSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "sampling_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.0001),
                ),
                InputSpec(
                    "return_actual_sigma",
                    BOOLEAN,
                    default=False,
                    doc=(
                        "Return the actual sigma value instead of the value used for interval "
                        "checks. This only affects results at 0.0 and 1.0."
                    ),
                ),
            ),
            outputs=(OutputSpec("sigma_value", FLOAT),),
            aliases=("SamplingPercentToSigma",),
        )


class BasicGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.basic_guider",
            display_name="Basic Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("conditioning", CONDITIONING),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("BasicGuider",),
        )


class CFGGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.cfg_guider",
            display_name="CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("CFGGuider",),
        )


class DualCFGGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.dual_cfg_guider",
            display_name="Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("cond1", CONDITIONING),
                InputSpec("cond2", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec(
                    "cfg_conds",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "cfg_cond2_negative",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "style",
                    COMBO,
                    default="regular",
                    widget=ComboWidget(options=("regular", "nested")),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("DualCFGGuider",),
            search_terms=("dual prompt guidance",),
        )


class DualModelGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.dual_model_guider",
            display_name="Dual Model CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("model_negative", MODEL, required=False),
                InputSpec("positive", CONDITIONING),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=4.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec("negative", CONDITIONING, required=False),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("DualModelGuider",),
        )


class ScheduledCFGGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.scheduled_cfg_guider",
            display_name="Scheduled CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("sigmas", SIGMAS),
                InputSpec(
                    "from_cfg",
                    FLOAT,
                    default=6.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "to_cfg",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "schedule",
                    COMBO,
                    default="log",
                    widget=ComboWidget(options=("linear", "log", "exp", "cos")),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER), OutputSpec("sigmas", SIGMAS)),
        )


class LTXVDualCFGGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ltxv_dual_cfg_guider",
            display_name="LTXV Dual CFG Guider",
            category="model/sampling/guiders",
            description="Applies separate CFG scales to LTX-2 video and audio streams.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec(
                    "video_cfg",
                    FLOAT,
                    default=3.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "audio_cfg",
                    FLOAT,
                    default=7.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("LTXVDualCFGGuider",),
            search_terms=("ltx", "audio video", "cfg", "guidance"),
        )


class PerpNegGuider(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.perp_neg_guider",
            display_name="Perp-Neg Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("empty_conditioning", CONDITIONING),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "neg_scale",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                *_conditioning_batching_inputs(),
            ),
            outputs=(OutputSpec("guider", GUIDER),),
            aliases=("PerpNegGuider",),
        )


class DisableCFG1Optimization(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.disable_cfg1_optimization",
            display_name="Disable CFG 1 Optimization",
            category="model/sampling/guiders",
            description="Evaluates negative conditioning when CFG is 1.0.",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("DisableModelCfg1Optimization",),
            search_terms=("force negative conditioning", "unconditional"),
        )


class DisableNoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.disable_noise",
            display_name="DisableNoise",
            category="model/sampling/noise",
            outputs=(OutputSpec("noise", NOISE),),
            aliases=("DisableNoise",),
            search_terms=("zero noise",),
        )


class RandomNoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.random_noise",
            display_name="RandomNoise",
            category="model/sampling/noise",
            inputs=(
                InputSpec(
                    "noise_seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
            ),
            outputs=(OutputSpec("noise", NOISE),),
            aliases=("RandomNoise",),
        )


class AddNoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.add_noise",
            display_name="AddNoise",
            category="model/sampling/noise",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("noise", NOISE),
                InputSpec("sigmas", SIGMAS),
                InputSpec("latent_image", LATENT),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("AddNoise",),
        )


class SamplerCustom(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_custom",
            display_name="SamplerCustom",
            category="model/sampling/custom",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("add_noise", BOOLEAN, default=True),
                InputSpec(
                    "noise_seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("sampler", SAMPLER),
                InputSpec("sigmas", SIGMAS),
                InputSpec("latent_image", LATENT),
                *_conditioning_batching_inputs(),
            ),
            outputs=(
                OutputSpec("output", LATENT),
                OutputSpec("denoised_output", LATENT),
            ),
            aliases=("SamplerCustom",),
            emits_previews=True,
        )


class SamplerCustomAdvanced(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_custom_advanced",
            display_name="SamplerCustomAdvanced",
            category="model/sampling/custom",
            inputs=(
                InputSpec("noise", NOISE),
                InputSpec("guider", GUIDER),
                InputSpec("sampler", SAMPLER),
                InputSpec("sigmas", SIGMAS),
                InputSpec("latent_image", LATENT),
            ),
            outputs=(
                OutputSpec("output", LATENT),
                OutputSpec("denoised_output", LATENT),
            ),
            aliases=("SamplerCustomAdvanced",),
            emits_previews=True,
        )


class ImpactRegionalSampler(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.compat.impact_regional_sampler",
            display_name="Impact Regional Sampler Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=(
                InputSpec("base_basic_pipe", IMPACT_BASIC_PIPE),
                InputSpec("region_basic_pipe", IMPACT_BASIC_PIPE),
                InputSpec("mask", MASK),
                InputSpec("samples", LATENT),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec(
                    "steps",
                    INT,
                    default=20,
                    widget=NumberWidget(min=1, max=10_000),
                ),
                InputSpec(
                    "base_only_steps",
                    INT,
                    default=2,
                    widget=NumberWidget(min=0, max=10_000),
                ),
                InputSpec(
                    "denoise",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "overlap_factor",
                    INT,
                    default=10,
                    widget=NumberWidget(min=0, max=10_000),
                ),
                InputSpec(
                    "restore_latent",
                    BOOLEAN,
                    default=True,
                    widget=BooleanWidget(label_on="enabled", label_off="disabled"),
                ),
                InputSpec(
                    "base_cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "base_sampler_name",
                    COMBO,
                    default="dinkster.euler",
                    widget=ComboWidget(options=SAMPLER_CHOICES),
                ),
                InputSpec(
                    "base_scheduler",
                    COMBO,
                    default="dinkster.simple",
                    widget=ComboWidget(options=SCHEDULER_CHOICES),
                ),
                InputSpec(
                    "region_cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "region_sampler_name",
                    COMBO,
                    default="dinkster.euler",
                    widget=ComboWidget(options=SAMPLER_CHOICES),
                ),
                InputSpec(
                    "region_scheduler",
                    COMBO,
                    default="dinkster.simple",
                    widget=ComboWidget(options=SCHEDULER_CHOICES),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_visibility="hidden",
            emits_previews=True,
        )


class VAEDecode(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode",
            display_name="VAE Decode",
            category="model/latent",
            inputs=(InputSpec("samples", LATENT), InputSpec("vae", VAE)),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("VAEDecode",),
            search_terms=("decode", "latent to image", "vae"),
        )


class VAEDecodeTiled(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_tiled",
            display_name="VAE Decode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("vae", VAE),
                InputSpec(
                    "tile_size",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=32),
                ),
                InputSpec(
                    "overlap",
                    INT,
                    default=64,
                    widget=NumberWidget(min=0, max=4096, step=32),
                ),
                InputSpec(
                    "temporal_size",
                    INT,
                    default=64,
                    advanced=True,
                    widget=NumberWidget(min=8, max=4096, step=4),
                ),
                InputSpec(
                    "temporal_overlap",
                    INT,
                    default=8,
                    advanced=True,
                    widget=NumberWidget(min=4, max=4096, step=4),
                ),
            ),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("VAEDecodeTiled",),
            search_terms=("decode", "latent to image", "vae", "tiled"),
        )


class VAEEncode(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_encode",
            display_name="VAE Encode",
            category="model/latent",
            inputs=(InputSpec("pixels", IMAGE), InputSpec("vae", VAE)),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("VAEEncode",),
            search_terms=("encode", "image to latent", "vae"),
        )


class VAEEncodeTiled(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_encode_tiled",
            display_name="VAE Encode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("pixels", IMAGE),
                InputSpec("vae", VAE),
                InputSpec(
                    "tile_size",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=64),
                ),
                InputSpec(
                    "overlap",
                    INT,
                    default=64,
                    widget=NumberWidget(min=0, max=4096, step=32),
                ),
                InputSpec(
                    "temporal_size",
                    INT,
                    default=64,
                    advanced=True,
                    widget=NumberWidget(min=8, max=4096, step=4),
                ),
                InputSpec(
                    "temporal_overlap",
                    INT,
                    default=8,
                    advanced=True,
                    widget=NumberWidget(min=4, max=4096, step=4),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("VAEEncodeTiled",),
            search_terms=("encode", "image to latent", "vae", "tiled"),
        )


class SeedVR2Preprocess(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.seedvr2_preprocess",
            display_name="Pre-Process SeedVR2 Input",
            category="image/pre-processors",
            description="Pads resized images for SeedVR2 VAE encoding.",
            inputs=(InputSpec("resized_images", IMAGE),),
            outputs=(OutputSpec("images", IMAGE),),
            aliases=("SeedVR2Preprocess",),
            search_terms=("seedvr2", "upscale", "video upscale", "pad", "preprocess"),
        )


class SeedVR2PostProcessing(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.seedvr2_postprocess",
            display_name="Post-Process SeedVR2 Output",
            category="image/post-processors",
            description="Aligns SeedVR2 output with its resized source and transfers color.",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec("original_resized_images", IMAGE),
                InputSpec(
                    "color_correction_method",
                    COMBO,
                    default="lab",
                    widget=ComboWidget(options=("lab", "wavelet", "adain", "none")),
                ),
            ),
            outputs=(OutputSpec("images", IMAGE),),
            aliases=("SeedVR2PostProcessing",),
            search_terms=(
                "seedvr2",
                "upscale",
                "color correction",
                "color match",
                "postprocess",
            ),
        )


class SeedVR2Conditioning(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.seedvr2_conditioning",
            display_name="Apply SeedVR2 Conditioning",
            category="model/conditioning",
            description="Builds SeedVR2 positive and negative conditioning from a VAE latent.",
            inputs=(InputSpec("model", MODEL), InputSpec("vae_conditioning", LATENT)),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("SeedVR2Conditioning",),
            search_terms=("seedvr2", "upscale", "conditioning"),
        )


class SeedVR2TemporalChunk(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.seedvr2_temporal_chunk",
            display_name="Split SeedVR2 Latent",
            category="model/latent/batch",
            description="Splits a SeedVR2 video latent into overlapping temporal chunks.",
            inputs=(
                InputSpec("latent", LATENT),
                InputSpec(
                    "temporal_overlap",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "chunking_mode",
                    options=(
                        DynamicComboOption("auto"),
                        DynamicComboOption(
                            "manual",
                            (
                                InputSpec(
                                    "frames_per_chunk",
                                    INT,
                                    default=21,
                                    widget=NumberWidget(min=1, max=16_384, step=4),
                                ),
                            ),
                        ),
                    ),
                    default="auto",
                ),
            ),
            outputs=(
                OutputSpec("latents", LATENT_LIST),
                OutputSpec("temporal_overlap", INT),
            ),
            aliases=("SeedVR2TemporalChunk",),
            search_terms=(
                "seedvr2",
                "split",
                "chunk",
                "temporal",
                "video upscale",
                "rebatch",
            ),
        )


class SeedVR2TemporalMerge(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.seedvr2_temporal_merge",
            display_name="Merge SeedVR2 Latents",
            category="model/latent/batch",
            description="Merges SeedVR2 temporal chunks with a Hann crossfade.",
            inputs=(
                InputSpec("latents", LATENT_LIST),
                InputSpec(
                    "temporal_overlap",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("SeedVR2TemporalMerge",),
            search_terms=("seedvr2", "merge", "temporal", "hann", "crossfade"),
        )


class LatentCombine(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.combine",
            display_name="Combine Latents",
            category="model/latent/advanced",
            description="Adds or subtracts one latent from another elementwise.",
            inputs=(
                InputSpec("samples1", LATENT),
                InputSpec("samples2", LATENT),
                InputSpec(
                    "operation",
                    COMBO,
                    required=False,
                    default="add",
                    widget=ComboWidget(options=("add", "subtract")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("add", "subtract", "sum", "difference", "latent"),
        )


class LatentMix(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.mix",
            display_name="Mix Latents",
            category="model/latent/advanced",
            description=(
                "Mixes two latents. interpolate blends direction and magnitude "
                "separately (spherical-style); blend is a plain weighted sum. "
                "factor is the weight of samples1."
            ),
            inputs=(
                InputSpec("samples1", LATENT),
                InputSpec("samples2", LATENT),
                InputSpec(
                    "operation",
                    COMBO,
                    required=False,
                    default="interpolate",
                    widget=ComboWidget(options=("interpolate", "blend")),
                ),
                InputSpec(
                    "factor",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("interpolate", "blend", "mix", "lerp", "latent"),
        )


class LatentMultiply(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.multiply",
            display_name="Multiply Latent",
            category="model/latent/advanced",
            description="Scales a latent by a constant factor.",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "multiplier",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("multiply", "scale", "amplify", "gain", "latent"),
        )


class LatentRotate(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.rotate",
            display_name="Rotate Latent",
            category="model/latent/transform",
            description="Rotates a latent clockwise by a right-angle amount.",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "angle",
                    COMBO,
                    required=False,
                    default="none",
                    widget=ComboWidget(options=("none", "90", "180", "270")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("rotate", "turn", "latent"),
        )


class LatentFlip(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.flip",
            display_name="Flip Latent",
            category="model/latent/transform",
            description=(
                "Mirrors a latent. vertical flips top-to-bottom (across the "
                "x axis); horizontal flips left-to-right (across the y axis)."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "axis",
                    COMBO,
                    required=False,
                    default="vertical",
                    widget=ComboWidget(options=("vertical", "horizontal")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("flip", "mirror", "latent"),
        )


class LatentCrop(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.crop",
            display_name="Crop Latent",
            category="model/latent/transform",
            description=(
                "Crops a pixel-space rectangle out of a latent. Coordinates "
                "and sizes are in pixels and snap to the 8-pixel latent grid."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=64, max=16_384, step=8),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=64, max=16_384, step=8),
                ),
                InputSpec(
                    "x",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "y",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("crop", "trim", "cut", "latent"),
        )


class LatentResize(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.resize",
            display_name="Resize Latent",
            category="model/latent",
            description=(
                "Resamples a latent to a pixel-space size. A zero width or "
                "height is derived from the other side's aspect ratio; both "
                "zero returns the latent unchanged."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "method",
                    COMBO,
                    required=False,
                    default="nearest-exact",
                    widget=ComboWidget(options=LATENT_RESIZE_METHODS),
                ),
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "crop",
                    COMBO,
                    required=False,
                    default="disabled",
                    widget=ComboWidget(options=("disabled", "center")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("resize", "upscale", "enlarge", "latent"),
        )


class LatentResizeBy(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.resize_by",
            display_name="Resize Latent By",
            category="model/latent",
            description="Resamples a latent by a scale factor.",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "method",
                    COMBO,
                    required=False,
                    default="nearest-exact",
                    widget=ComboWidget(options=LATENT_RESIZE_METHODS),
                ),
                InputSpec(
                    "scale_by",
                    FLOAT,
                    required=False,
                    default=1.5,
                    widget=NumberWidget(min=0.01, max=8.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("resize", "upscale", "scale", "latent"),
        )


class LatentComposite(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.composite",
            display_name="Composite Latents",
            category="model/latent",
            description=(
                "Pastes a source latent onto a destination latent at a "
                "pixel-space offset, optionally feathering the pasted edges."
            ),
            inputs=(
                InputSpec("destination", LATENT),
                InputSpec("source", LATENT),
                InputSpec(
                    "x",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "y",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "feather",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("composite", "overlay", "paste", "layer", "latent"),
        )


class LatentCompositeMasked(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.composite_masked",
            display_name="Composite Latents (Masked)",
            category="model/latent",
            description=(
                "Pastes a source latent onto a destination latent through an "
                "optional mask. The source can be resized to cover the "
                "destination."
            ),
            inputs=(
                InputSpec("destination", LATENT),
                InputSpec("source", LATENT),
                InputSpec(
                    "x",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec(
                    "y",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384, step=8),
                ),
                InputSpec("resize_source", BOOLEAN, required=False, default=False),
                InputSpec("mask", MASK, required=False),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("composite", "overlay", "paste", "inpaint", "mask", "latent"),
        )


class LatentConcat(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.concat",
            display_name="Concat Latents",
            category="model/latent/advanced",
            description=(
                "Concatenates two latents along a spatial or temporal axis. "
                "A leading '-' places samples2 before samples1."
            ),
            inputs=(
                InputSpec("samples1", LATENT),
                InputSpec("samples2", LATENT),
                InputSpec(
                    "dim",
                    COMBO,
                    required=False,
                    default="x",
                    widget=ComboWidget(options=("x", "-x", "y", "-y", "t", "-t")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("concat", "join", "stitch", "latent"),
        )


class LatentCut(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.cut",
            display_name="Cut Latent",
            category="model/latent/advanced",
            description=(
                "Extracts a slice of latent cells along one axis. A negative "
                "index counts from the end."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "dim",
                    COMBO,
                    required=False,
                    default="x",
                    widget=ComboWidget(options=("x", "y", "t")),
                ),
                InputSpec(
                    "index",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-16_384, max=16_384),
                ),
                InputSpec(
                    "amount",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=16_384),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("cut", "slice", "extract", "latent"),
        )


class LatentCutToBatch(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.cut_to_batch",
            display_name="Cut Latent to Batch",
            category="model/latent/advanced",
            description=(
                "Splits one axis of a latent into equal slices stacked on "
                "the batch axis, truncating any remainder."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "dim",
                    COMBO,
                    required=False,
                    default="t",
                    widget=ComboWidget(options=("t", "x", "y")),
                ),
                InputSpec(
                    "slice_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=16_384),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("split", "batch", "tile", "latent"),
        )


class LatentFromBatch(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.from_batch",
            display_name="Latent From Batch",
            category="model/latent/batch",
            description=(
                "Extracts a run of entries from a latent batch, slicing the "
                "noise mask and batch index bookkeeping to match. A negative "
                "batch index counts from the end."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "batch_index",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-16_384, max=16_384),
                ),
                InputSpec(
                    "length",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=64),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("select", "from batch", "subset", "latent"),
        )


class LatentRepeat(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.repeat",
            display_name="Repeat Latent Batch",
            category="model/latent/batch",
            description=(
                "Repeats a latent batch a number of times, extending the "
                "noise mask and batch index bookkeeping to match."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "amount",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=64),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("repeat", "duplicate", "batch", "latent"),
        )


class LatentSeedBehavior(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.seed_behavior",
            display_name="Latent Batch Seed Behavior",
            category="model/latent/advanced",
            description=(
                "Controls how noise seeds vary across a latent batch: random "
                "gives every entry its own seed offset, fixed gives all "
                "entries the seed of the first."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "behavior",
                    COMBO,
                    required=False,
                    default="fixed",
                    widget=ComboWidget(options=("random", "fixed")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("seed", "batch", "noise", "latent"),
        )


class LatentBatch(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.batch",
            display_name="Batch Latents",
            category="model/latent/batch",
            description=(
                "Concatenates latents into one batch, resizing later inputs "
                "to the spatial size of the first and concatenating batch "
                "index bookkeeping."
            ),
            input_families=(
                InputFamilySpec(
                    "latents", LATENT, min_members=1, max_members=50, member_prefix="latent_"
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("batch", "merge", "combine", "latent"),
        )


class LatentRebatch(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.rebatch",
            display_name="Rebatch Latents",
            category="model/latent/batch",
            description=(
                "Regroups a list of latents into batches of a chosen size, "
                "merging compatible entries and splitting oversized ones."
            ),
            inputs=(
                InputSpec("latents", LATENT_LIST),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4_096),
                ),
            ),
            outputs=(OutputSpec("latents", LATENT_LIST),),
            search_terms=("rebatch", "batch size", "regroup", "latent"),
        )


class LatentSetNoiseMask(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.set_noise_mask",
            display_name="Set Latent Noise Mask",
            category="model/latent",
            description=(
                "Attaches a noise mask to a latent so sampling only regenerates the masked region."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("mask", MASK),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("noise mask", "inpaint", "mask", "latent"),
        )


class LatentReplaceFrames(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.replace_frames",
            display_name="Replace Video Latent Frames",
            category="model/latent/batch",
            description=(
                "Overwrites a run of temporal frames in a video latent with "
                "frames from a source latent starting at an index. A "
                "negative index counts from the end; out-of-bounds requests "
                "return the destination unchanged."
            ),
            inputs=(
                InputSpec("destination", LATENT),
                InputSpec("source", LATENT, required=False),
                InputSpec(
                    "index",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-16_384, max=16_384),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("replace", "frames", "video", "latent"),
        )


class LatentApplyOperation(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.apply_operation",
            display_name="Latent Apply Operation",
            category="model/latent/advanced/operations",
            description=(
                "Applies a latent operation to the samples of a latent, "
                "leaving the noise mask and batch index bookkeeping "
                "untouched."
            ),
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("operation", LATENT_OPERATION),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            search_terms=("apply", "operation", "transform", "latent"),
        )


class LatentOperationTonemapReinhard(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.operation_tonemap_reinhard",
            display_name="Latent Operation Tonemap Reinhard",
            category="model/latent/advanced/operations",
            description=(
                "Builds a latent operation that compresses per-position "
                "latent vector magnitudes with a Reinhard curve scaled by "
                "the batch magnitude distribution."
            ),
            inputs=(
                InputSpec(
                    "multiplier",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("operation", LATENT_OPERATION),),
            search_terms=("tonemap", "reinhard", "hdr", "operation", "latent"),
        )


class LatentOperationSharpen(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.operation_sharpen",
            display_name="Latent Operation Sharpen",
            category="model/latent/advanced/operations",
            description=(
                "Builds a latent operation that sharpens magnitude-"
                "normalized latents with an inverted gaussian kernel of the "
                "given radius, sigma, and strength."
            ),
            inputs=(
                InputSpec(
                    "sharpen_radius",
                    INT,
                    required=False,
                    default=9,
                    widget=NumberWidget(min=1, max=31, step=1),
                ),
                InputSpec(
                    "sigma",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.1, max=10.0, step=0.1),
                ),
                InputSpec(
                    "alpha",
                    FLOAT,
                    required=False,
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("operation", LATENT_OPERATION),),
            search_terms=("sharpen", "operation", "kernel", "latent"),
        )


class LatentApplyOperationCFG(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.apply_operation_cfg",
            display_name="Latent Apply Operation CFG",
            category="model/latent/advanced/operations",
            description=(
                "Patches a model so the latent operation runs on each "
                "sampling step's predictions before CFG combination: on the "
                "guidance difference when a negative prediction exists, and "
                "on the conditional prediction alone otherwise."
            ),
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("operation", LATENT_OPERATION),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("apply", "operation", "cfg", "guidance", "latent"),
        )


class LatentGenerateNoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.generate_noise",
            display_name="Generate Noise",
            category="model/latent/advanced",
            description=(
                "Generates a seeded gaussian-noise latent for injection or "
                "for samplers with add_noise disabled. Optional model and "
                "sigmas inputs scale the noise by the schedule's total sigma "
                "span in the model's latent space."
            ),
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=16, max=4_096),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=16, max=4_096),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4_096),
                ),
                InputSpec(
                    "seed",
                    INT,
                    required=False,
                    default=123,
                    widget=NumberWidget(min=0, control_after_generate="randomize"),
                ),
                InputSpec(
                    "multiplier",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=4_096.0, step=0.01),
                ),
                InputSpec("constant_batch_noise", BOOLEAN, required=False, default=False),
                InputSpec("normalize", BOOLEAN, required=False, default=False),
                InputSpec("model", MODEL, required=False),
                InputSpec("sigmas", SIGMAS, required=False),
                InputSpec(
                    "latent_channels",
                    COMBO,
                    required=False,
                    default="4",
                    widget=ComboWidget(options=("4", "16")),
                ),
                InputSpec(
                    "shape",
                    COMBO,
                    required=False,
                    default="BCHW",
                    widget=ComboWidget(options=("BCHW", "BCTHW", "BTCHW")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("GenerateNoise",),
            search_terms=("noise", "generate", "seed", "latent"),
        )


class LatentInjectNoise(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.latent.inject_noise",
            display_name="Inject Noise To Latent",
            category="model/latent/advanced",
            description=(
                "Adds a noise latent to a latent: scaled by strength, or "
                "averaged. Optionally normalizes the result, blends it "
                "through a mask, and mixes in a seeded random draw."
            ),
            inputs=(
                InputSpec("latents", LATENT),
                InputSpec(
                    "strength",
                    FLOAT,
                    required=False,
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=200.0, step=0.0001),
                ),
                InputSpec("noise", LATENT),
                InputSpec("normalize", BOOLEAN, required=False, default=False),
                InputSpec("average", BOOLEAN, required=False, default=False),
                InputSpec("mask", MASK, required=False),
                InputSpec(
                    "mix_randn_amount",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1_000.0, step=0.001),
                ),
                InputSpec(
                    "seed",
                    INT,
                    required=False,
                    default=123,
                    widget=NumberWidget(min=0, control_after_generate="randomize"),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("InjectNoiseToLatent",),
            search_terms=("noise", "inject", "mix", "latent"),
        )


GENERATION_NODES: tuple[type[Node], ...] = (
    *TRELLIS2_NODES,
    *MODEL3D_GENERATION_NODES,
    LoadModelProfile,
    LoadCheckpoint,
    LoadControlNet,
    ApplyControlNet,
    ApplyControlNetAdvanced,
    SetControlNetUnionType,
    LoadCheckpointStack,
    LoadDiffusionModel,
    LoadDiffusionComponents,
    LoadLTXAVTextEncoder,
    LoadLTXAVAudioVAE,
    LoadLatentUpscaleModel,
    LTXAVAudioVAEDecode,
    LoadLora,
    LoadLoraModelOnly,
    ApplyLoraStack,
    ApplyLoraStackModelOnly,
    CLIPTextEncode,
    CLIPTextEncodeLumina2,
    ModelSamplingAuraFlow,
    TextGenerate,
    PromptEnhance,
    CLIPSetLastLayer,
    T5TokenizerOptions,
    CLIPTextEncodeControlnet,
    FluxGuidance,
    FluxDisableGuidance,
    ReferenceLatent,
    CFGZeroStar,
    CFGNorm,
    TCFG,
    FreSca,
    LazyCache,
    EasyCache,
    AttentionSchedule,
    ContextWindowsManual,
    WanContextWindowsManual,
    LTXVContextWindows,
    AdaptiveProjectedGuidance,
    MahiroGuidance,
    EpsilonScaling,
    CFGOverride,
    RescaleCFG,
    RenormCFG,
    TemporalScoreRescaling,
    NormalizedAttentionGuidance,
    LTXAVConditioning,
    LTXAVReferenceAudio,
    LTXAVIDLoRAReferenceAudio,
    LTXVSpatioTemporalGuidance,
    LTXVModalityGuidance,
    LTXVDurationPredictor,
    LTXVDualCFGGuider,
    LTXVConditioning,
    LTXVImageToVideo,
    LTXVImageToVideoInplace,
    LTXVAddGuide,
    LTXVCropGuides,
    LTXVLatentUpsampler,
    ConditioningMerge,
    ConditioningScale,
    ConditioningSetArea,
    ConditioningSetMask,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
    ChromaRadianceOptions,
    ChromaModelSampling,
    ModelSamplingSD3,
    ModelSamplingLTXV,
    ModelSamplingFlux,
    EmptyLatentImage,
    EmptySD3LatentImage,
    EmptyChromaRadianceLatentImage,
    EmptyFlux2LatentImage,
    EmptyLTXAVLatent,
    EmptyLTXVLatent,
    KSampler,
    KSamplerAdvanced,
    KSamplerSelect,
    SamplerDPMPP3MSDE,
    SamplerDPMPP2MSDE,
    SamplerDPMPPSDE,
    SamplerDPMPP2SAncestral,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerDPMAdaptative,
    SamplerERSDE,
    SamplerSEEDS2,
    SamplerSASolver,
    BasicScheduler,
    BetaSamplingScheduler,
    SDTurboScheduler,
    KarrasScheduler,
    ExponentialScheduler,
    PolyexponentialScheduler,
    LaplaceScheduler,
    VPScheduler,
    AlignYourStepsScheduler,
    GITSScheduler,
    OptimalStepsScheduler,
    Flux2Scheduler,
    Ideogram4Scheduler,
    ManualSigmas,
    SplitSigmas,
    SplitSigmasDenoise,
    FlipSigmas,
    SetFirstSigma,
    ExtendIntermediateSigmas,
    SamplingPercentToSigma,
    BasicGuider,
    CFGGuider,
    DualCFGGuider,
    DualModelGuider,
    ScheduledCFGGuider,
    PerpNegGuider,
    DisableCFG1Optimization,
    DisableNoise,
    RandomNoise,
    AddNoise,
    SamplerCustom,
    SamplerCustomAdvanced,
    ImpactRegionalSampler,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeTiled,
    SeedVR2Preprocess,
    SeedVR2PostProcessing,
    SeedVR2Conditioning,
    SeedVR2TemporalChunk,
    SeedVR2TemporalMerge,
    LatentCombine,
    LatentMix,
    LatentMultiply,
    LatentRotate,
    LatentFlip,
    LatentCrop,
    LatentResize,
    LatentResizeBy,
    LatentComposite,
    LatentCompositeMasked,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentFromBatch,
    LatentRepeat,
    LatentSeedBehavior,
    LatentBatch,
    LatentRebatch,
    LatentSetNoiseMask,
    LatentReplaceFrames,
    LatentApplyOperation,
    LatentOperationTonemapReinhard,
    LatentOperationSharpen,
    LatentApplyOperationCFG,
    LatentGenerateNoise,
    LatentInjectNoise,
)

GENERATION_NODE_IDS = tuple(node.schema().node_type for node in GENERATION_NODES)
GENERATION_COMPAT_CARRIER_NODE_IDS = tuple(
    node.schema().node_type for node in GENERATION_COMPAT_CARRIER_NODES
)
GENERATION_SCHEMA_NODES = (*GENERATION_NODES, *GENERATION_COMPAT_CARRIER_NODES)
GENERATION_SCHEMA_NODE_IDS = (*GENERATION_NODE_IDS, *GENERATION_COMPAT_CARRIER_NODE_IDS)

__all__ = [
    "GENERATION_COMPAT_CARRIER_NODE_IDS",
    "GENERATION_COMPAT_CARRIER_NODES",
    "GENERATION_NODE_IDS",
    "GENERATION_NODES",
    "GENERATION_SCHEMA_NODE_IDS",
    "GENERATION_SCHEMA_NODES",
]
