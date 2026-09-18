"""Checkpoint assembly planning - torch-free.

ComfyUI assembles a runnable model by loading the WHOLE state dict and
guessing as it goes (comfy/sd.py load_checkpoint_guess_config /
load_state_dict_guess_config @ b78cec87). Dinkster plans first: this
module turns weight-source headers (plus explicitly modeled scalar
configuration tensors) into a typed assembly plan that records, per component,
exactly which source keys
feed which model keys, at what storage dtype, with what per-layer
quantization, and - where a rename is not enough - which
:class:`~.weights.TensorTransform` derives the model tensor. The
torch executor (dinkster_inference_torch.assemble) then reads only the
planned slices.

Scope: classic Flux dev / schnell (:func:`plan_flux_assembly` - Flux
DiT + CLIP-L + classic T5-XXL + KL VAE) and the classic SD era
(:func:`plan_sd_assembly` - SD 1.5 / SDXL base / SDXL refiner UNets +
their CLIP text encoders + KL VAE, including the OpenCLIP-format
CLIP-G layout raw SDXL checkpoints ship). Combined checkpoints, split
per-component files, and any mix; a split source overrides the
combined checkpoint for its component. Every other family or layout
refuses loudly through the underlying detectors (their messages name
the ROADMAP entry).

Mixed per-parameter storage dtypes are NORMAL, not an error: the plan
carries a per-model-key dtype inventory and quantization is per-layer
(quantization.split_quantization), never a model-wide switch.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, cast

from .anima import ANIMA_CONFIG, anima_layout
from .autoencoder_kl import (
    KL_PREFIX_RENAMES,
    KLConfig,
    detect_kl_config,
    diffusers_kl_key,
    is_diffusers_kl,
    normalize_kl_keys,
)
from .catalog import (
    FLUX2_DEV,
    FLUX2_KLEIN_4B,
    FLUX2_KLEIN_9B,
    FLUX_DEV,
    FLUX_SCHNELL,
    LTXAV,
    LUMINA2,
    QWEN_IMAGE,
    SD15,
    SDXL,
    SDXL_REFINER,
    SDXL_VARIANT_MARKERS,
    WAN21,
    WAN22,
    Z_IMAGE,
    Z_IMAGE_PIXEL_SPACE,
)
from .clip_text import (
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    CLIP_TEXT_OPTIONAL_KEYS,
    ClipTextConfig,
    detect_clip_text_config,
)
from .clip_vision import ClipVisionConfig, clip_vision_layout, detect_wan21_clip_vision
from .controlnet import (
    ControlNetDetectError,
    ControlNetLayout,
    ControlNetSourceLayout,
    SD15ControlNetConfig,
    SDXLControlLoRAConfig,
    SDXLControlNetConfig,
    SDXLControlNetUnionConfig,
    detect_sdxl_control_lora,
    normalize_sd15_controlnet,
    normalize_sdxl_controlnet,
    normalize_sdxl_controlnet_union,
    sd15_controlnet_layout,
    sdxl_control_lora_layout,
    sdxl_controlnet_layout,
    sdxl_controlnet_union_layout,
)
from .devices import BFLOAT16, FLOAT32, INT8, UINT8, DType
from .dinov3 import (
    DINOv3ViTConfig,
    NAFConfig,
    detect_dinov3_vith,
    detect_dinov3_vitl,
    detect_naf,
)
from .families import ModelFamily
from .flux import FluxConfig, detect_flux_config
from .flux2 import (
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B_CONFIG,
    FLUX2_LATENT_CHANNELS,
    Flux2DetectError,
    detect_flux2_config,
)
from .gemma_text import (
    GEMMA2_LUMINA_2B_CONFIG,
    GEMMA3_LTX_12B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    GemmaTextConfig,
    GemmaTextDetectError,
    LtxConnectorConfig,
    LtxTextProjectionKind,
    detect_gemma_text_config,
    detect_ltx_text_connectors,
    detect_ltx_text_projection,
    ltx_connector_layout,
)
from .ideogram4 import Ideogram4DetectError, detect_ideogram4_config
from .ideogram4_text import (
    IDEOGRAM4_LANGUAGE_SUBTREE,
    IDEOGRAM4_VISION_SUBTREE,
    Ideogram4TextDetectError,
    detect_ideogram4_text_config,
    ideogram4_text_layout,
)
from .ipadapter import (
    SD15IPAdapterConfig,
    detect_sd15_ipadapter,
    detect_sd15_ipadapter_clip_vision,
    sd15_ipadapter_layout,
)
from .krea2 import Krea2DetectError, detect_krea2_config
from .krea2_text import (
    KREA2_LANGUAGE_SUBTREE,
    KREA2_VISION_SUBTREE,
    Krea2TextDetectError,
    detect_krea2_text_config,
    krea2_text_layout,
)
from .ltx import (
    LTX_LATENT_UPSAMPLER_CONFIG,
    LTXAV_19B_CONFIG,
    LTXAV_19B_VAE_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V23_VAE_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_22B_V25_VAE_CONFIG,
    LTXAV_AUDIO_MAX_POS,
    LTXAV_CONNECTOR_PREFIXES,
    LTXAV_DURATION_HEAD_CONFIG,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    LTXV_2B_V095_CONFIG,
    LTXV_2B_V095_VAE_CONFIG,
    LTXV_MAX_POS,
    LTXV_THETA,
    LTXV_TIMESTEP_MULTIPLIER,
    LTXAVConfig,
    LTXDiffusionVideoVAEConfig,
    LTXDurationHeadConfig,
    LTXLatentUpsamplerConfig,
    LTXVConfig,
    LTXVideoVAEConfig,
    detect_ltxav,
    detect_ltxv,
    ltx_diffusion_video_vae_layout,
    ltx_latent_upsampler_layout,
    ltxav_duration_head_layout,
    ltxav_layout,
    ltxv_layout,
    ltxv_vae_layout,
)
from .ltx_audio import (
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_VOCODER_CONFIG,
    LTXAV_BWE_VOCODER_CONFIG,
    LTXAudioVAEConfig,
    LTXVocoderBWEConfig,
    LTXVocoderConfig,
    ltx_audio_vae_layout,
    ltx_vocoder_bwe_layout,
    ltx_vocoder_layout,
)
from .lumina2 import LUMINA2_CONFIG, lumina2_layout
from .minimax_music3 import (
    MINIMAX_MUSIC3_CONFIG,
    MINIMAX_MUSIC3_DAV_CONFIG,
    MiniMaxMusic3Config,
    MiniMaxMusic3DavConfig,
    MiniMaxMusic3TextConfig,
    detect_minimax_music3_text_config,
    minimax_music3_dav_layout,
    minimax_music3_diffusion_layout,
)
from .openclip_text import convert_openclip_text, is_openclip_text
from .quantization import (
    SUPPORTED_QUANT_FORMATS,
    LayerQuant,
    QuantizationError,
    split_quantization,
)
from .qwen_image import (
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    QwenImageConfig,
)
from .qwen_image_control import (
    QWEN_IMAGE_DIFFSYNTH,
    QWEN_IMAGE_DIFFSYNTH_INPAINT,
    QWEN_IMAGE_FUN_CONTROL,
    QWEN_IMAGE_INSTANTX_CONTROL,
    QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
    QwenImageControlConfig,
    QwenImageDiffSynthConfig,
    qwen_image_control_layout,
    qwen_image_diffsynth_layout,
    require_qwen_image_control_layout,
    require_qwen_image_diffsynth_layout,
)
from .qwen_image_text import (
    QWEN_IMAGE_TEXT_CONFIG,
    QwenImageTextConfig,
    detect_qwen_image_text_config,
    qwen_image_text_layout,
)
from .qwen_text import KLEIN_QWEN3_4B_CONFIG, QwenTextConfig, detect_qwen_text_config
from .sampling import Parameterization, SamplingDescriptor, SamplingSpace
from .seedvr2 import SeedVR2DetectError, detect_seedvr2_config
from .seedvr2_vae import SeedVR2VAEHeaderError, detect_seedvr2_vae_config
from .spaces import linear_beta_sigmas
from .t2i_adapter import (
    SD15T2IAdapterConfig,
    T2IAdapterDetectError,
    normalize_sd15_t2i_adapter,
    sd15_t2i_adapter_layout,
)
from .t5_text import (
    T5_TEXT_OPTIONAL_KEYS,
    T5_XXL_CONFIG,
    UMT5_XXL_CONFIG,
    T5Config,
    detect_t5_config,
)
from .taehv import TAEHVConfig, TAEHVDetectError, detect_taehv_decoder_config
from .taesd import TAESDConfig, TAESDDetectError, TAESDFamily, detect_taesd_config
from .trellis2 import (
    Trellis2DecoderConfig,
    Trellis2FlowConfig,
    detect_trellis2_decoder,
    detect_trellis2_flow,
)
from .triposplat import detect_triposplat_config, detect_triposplat_gaussian_decoder
from .unet import (
    SD15_INPAINT_UNET_CONFIG,
    SD15_UNET_CONFIG,
    SDXL_INPAINT_UNET_CONFIG,
    SDXL_REFINER_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    UNetConfig,
    detect_unet_config,
)
from .wan21 import (
    WAN21_ANIMATE2_14B,
    WAN21_CAMERA_1_3B,
    WAN21_CAMERA_14B,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_FLF_I2V_14B,
    WAN21_FLOW_RVS_1_3B,
    WAN21_FUN_CONTROL_1_3B,
    WAN21_FUN_INPAINT_1_3B,
    WAN21_HUMO_17B,
    WAN21_I2V_14B,
    WAN21_SCAIL2_14B,
    WAN21_SCAIL_14B,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_VACE_1_3B,
    WAN21_VACE_14B,
    WAN22_ANIMATE_14B,
    WAN22_BERNINI_14B,
    WAN22_CAMERA_14B,
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_CONTROL_14B,
    WAN22_FUN_INPAINT_5B,
    WAN22_I2V_14B,
    WAN22_S2V_14B,
    WAN22_TI2V_5B,
    WAN22_WANDANCER_14B,
    Wan21Config,
    detect_wan21,
    detect_wan22,
    wan21_layout,
)
from .wan21_multitalk import (
    WAN21_MULTITALK,
    Wan21MultiTalkConfig,
    require_wan21_multitalk_layout,
    wan21_multitalk_model_layout,
)
from .wan21_uni3c import (
    WAN21_UNI3C,
    Wan21Uni3CConfig,
    normalize_wan21_uni3c_key,
    require_wan21_uni3c_layout,
    wan21_uni3c_model_layout,
)
from .wan21_vae import (
    WAN21_FLOW_RVS_VAE_CONFIG,
    WAN21_VAE_CONFIG,
    Wan21VAEConfig,
    wan21_vae_layout,
)
from .wan22_vae import (
    WAN22_VAE_CONFIG,
    Wan22VAEConfig,
    Wan22VAEHeaderError,
    validate_wan22_vae_header,
)
from .weights import (
    AssetIdentifiedSource,
    ConfigurationPayloadSource,
    ConfigurationScalarSource,
    LinearToConv2D,
    RowChunk,
    TensorGeometry,
    TensorTransform,
    WeightSource,
)
from .z_image import (
    Z_IMAGE_CONFIG,
    Z_IMAGE_CONTROL_CONFIG,
    Z_IMAGE_PIXEL_CONFIG,
    ZImageConfig,
    ZImageControlConfig,
    ZImagePixelConfig,
    detect_z_image_control,
    z_image_control_layout,
    z_image_layout,
    z_image_pixel_layout,
)

_ASSET_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")

C = TypeVar("C")

#: Combined-checkpoint component prefixes (comfy/sd.py checkpoint
#: layout + supported_models.py Flux text-encoder naming @ b78cec87).
#: The ``transformer.`` segment is the reference's clip/t5 wrapper
#: module attribute; Dinkster's native modules sit directly at the
#: stripped keys.
FLUX_DIFFUSION_PREFIX = "model.diffusion_model."
FLUX_VAE_PREFIX = "vae."
FLUX_CLIP_L_PREFIX = "text_encoders.clip_l.transformer."
FLUX_T5XXL_PREFIX = "text_encoders.t5xxl.transformer."
FLUX_QWEN_PREFIX = "text_encoders.qwen3_2b.transformer.model."
WAN21_DIFFUSION_PREFIX = "model.diffusion_model."
WAN21_UMT5_PREFIX = "text_encoders.umt5xxl.transformer."
WAN21_UMT5_ROOT = "text_encoders.umt5xxl."
WAN21_VAE_PREFIX = "vae."
Z_IMAGE_QWEN_PREFIX = "text_encoders.qwen3_4b.transformer.model."
QWEN_IMAGE_TEXT_PREFIX = "text_encoders.qwen2_5_vl_7b."
KREA2_TEXT_PREFIX = "text_encoders.qwen3vl_4b.transformer."
IDEOGRAM4_TEXT_PREFIX = "text_encoders.qwen3vl_8b.transformer."
ANIMA_QWEN_PREFIX = "text_encoders.qwen3_06b.transformer.model."
LUMINA2_GEMMA_PREFIX = "text_encoders.gemma2_2b.transformer.model."
#: Flux2 text-encoder slots (comfy/text_encoders/flux.py Mistral3 and
#: klein_te naming @ b78cec87). The Klein 4B slot name matches
#: Z-Image's - both park a Qwen3-4B tower under ``qwen3_4b`` - so the
#: DiT geometry, not the text slot, disambiguates the family.
FLUX2_MISTRAL_PREFIX = "text_encoders.mistral3_24b.transformer.model."
FLUX2_QWEN3_4B_PREFIX = "text_encoders.qwen3_4b.transformer.model."
FLUX2_QWEN3_8B_PREFIX = "text_encoders.qwen3_8b.transformer.model."
#: LTX-2 combined-checkpoint component prefixes beside the shared
#: ``model.diffusion_model.`` / ``vae.`` roots (Lightricks/LTX-2
#: ltx-2-19b-dev.safetensors; comfy/sd.py audio VAE and vocoder
#: extraction @ b78cec87). The split Gemma 3 file parks the tower
#: under a bare ``model.`` prefix with ``spiece_model`` beside it,
#: the same convention as Wan's split UMT5 file.
LTXAV_GEMMA_PREFIX = "model."
LTXAV_PROJECTION_PREFIX = "text_embedding_projection."
#: The official Gemma 3 12B split artifact bundles the multimodal vision
#: tower beside the text stack; the reference loads it into a vision
#: tower that text-only encoding never consults, and Dinkster has no such
#: tower, so these keys are dropped rather than refused.
LTXAV_GEMMA_UNUSED_PREFIXES = (
    "vision_model.",
    "multi_modal_projector.",
    "audio_projector.",
)
LTXAV_GEMMA_UNUSED_ASSETS = frozenset(
    {
        "hf_asset__chat_template.jinja",
        "hf_asset__generation_config.json",
        "hf_asset__processor_config.json",
        "hf_asset__tokenizer_config.json",
    }
)
LTXAV_AUDIO_VAE_PREFIX = "audio_vae."
LTXAV_VOCODER_PREFIX = "vocoder."

#: Non-transformer keys the reference checkpoint format parks beside
#: each text encoder (CLIP's contrastive-training temperature; never
#: consumed at inference). Recorded as ignored, not errors.
_TEXT_ENCODER_ROOTS = ("text_encoders.clip_l.", "text_encoders.t5xxl.")

#: Role-separated non-text roots beside ``model.*`` in the official
#: Ovis multimodal artifact. Unknown siblings refuse.
_OVIS_IGNORED_ROOTS = frozenset({"logit_scale", "vision_model", "visual_tokenizer", "vte"})

#: Classic SD-era combined-checkpoint component prefixes
#: (comfy/supported_models.py SD15/SDXL/SDXLRefiner
#: process_clip_state_dict + comfy/sd.py VAE extraction @ b78cec87).
#: SD 1.5 parks its transformers-format CLIP-L under
#: ``cond_stage_model.transformer.``; SDXL base carries CLIP-L
#: (transformers format) in conditioner slot 0 and CLIP-G (OpenCLIP
#: format) in slot 1; the refiner carries only CLIP-G (OpenCLIP
#: format) in slot 0.
SD_DIFFUSION_PREFIX = "model.diffusion_model."
SD_VAE_PREFIX = "first_stage_model."
TRIPOSPLAT_DIFFUSION_PREFIX = "model.diffusion_model."
SD15_CLIP_L_PREFIX = "cond_stage_model.transformer."
SDXL_CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
SDXL_CLIP_G_PREFIX = "conditioner.embedders.1.model."
SDXL_REFINER_CLIP_G_PREFIX = "conditioner.embedders.0.model."

#: Roots widening each text encoder's ignored-key scan (the reference
#: parks ``logit_scale`` beside the wrapped module).
_SD15_CLIP_ROOT = "cond_stage_model."
_SDXL_EMBEDDER_ROOTS = ("conditioner.embedders.0.", "conditioner.embedders.1.")

#: Inert HF buffers checkpoints park inside the transformers-format
#: CLIP slice; the reference loads strict=False and drops them
#: (comfy/sd1_clip.py SDClipModel.load_sd @ b78cec87; the float32
#: rounding of position_ids in SD15.process_clip_state_dict is
#: vestigial - comfy/clip_model.py registers no such buffer). Both
#: spellings: modern (text_model.-prefixed) and legacy.
_CLIP_TEXT_INERT_KEYS = frozenset({"text_model.embeddings.position_ids", "embeddings.position_ids"})

# Standalone SD VAEs can retain EMA bookkeeping that ComfyUI
# accepts as leftover keys when loading the codec.
_SD_VAE_INERT_KEYS = frozenset({"model_ema.decay", "model_ema.num_updates"})


class AssemblyError(ValueError):
    """A checkpoint set this planner cannot assemble: a missing
    component, an unrecognized architecture, or a quantization layout
    with no port. The message names the component and the reason."""


@dataclass(frozen=True)
class NativePlanningContext:
    """Worker-environment facts a planner may bind into runtime identity.

    Planning stays a pure function of headers plus this explicit
    context: no planner reads package metadata or torch state itself.
    The loading side constructs the context from the live worker
    environment; planners that bind no environment facts ignore it.
    """

    torch_version: str
    comfy_kitchen_version: str

    def __post_init__(self) -> None:
        for name in ("torch_version", "comfy_kitchen_version"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class ComponentPlan(Generic[C]):
    """One component's loading contract.

    ``keys`` maps MODEL state-dict keys to full SOURCE keys (prefix
    and spelling normalization already applied), so the executor reads
    and renames without re-deriving either. ``quant`` is keyed by
    MODEL layer name; each :class:`LayerQuant` names its artifact
    tensors in full SOURCE keys. ``ignored`` lists source keys
    deliberately left unread (duplicate embedding aliases,
    ``logit_scale``); ``absent`` lists optional model keys the source
    legitimately lacks (the executor default-fills them).
    ``transforms`` names the model keys whose tensor is DERIVED from
    the source tensor rather than read through (OpenCLIP's fused
    in_proj split, the text_projection transpose); several model keys
    may then share one source key. Transforms never change dtype, so
    ``dtypes`` stays the source-tensor storage dtype.
    """

    component: str
    path: Path
    config: C
    keys: Mapping[str, str]
    dtypes: Mapping[str, DType]
    quant: Mapping[str, LayerQuant]
    ignored: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    transforms: Mapping[str, TensorTransform] = field(default_factory=dict[str, TensorTransform])
    identity_facts: tuple[str, ...] = field(default=(), repr=False)
    runtime_facts: tuple[str, ...] = field(default=(), repr=False)
    source_format: str = "safetensors"
    payload_source: WeightSource | None = field(default=None, repr=False, compare=False)
    payload_consumed: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
        object.__setattr__(self, "dtypes", MappingProxyType(dict(self.dtypes)))
        object.__setattr__(self, "quant", MappingProxyType(dict(self.quant)))
        object.__setattr__(self, "transforms", MappingProxyType(dict(self.transforms)))
        object.__setattr__(self, "identity_facts", tuple(self.identity_facts))
        object.__setattr__(self, "runtime_facts", tuple(self.runtime_facts))
        if self.source_format not in {"safetensors", "gguf"}:
            raise ValueError(f"unsupported component source format {self.source_format!r}")
        if self.source_format == "gguf":
            if not self.runtime_facts:
                raise ValueError("GGUF component plan requires bound runtime facts")
            if self.payload_consumed == (self.payload_source is not None):
                raise ValueError(
                    "GGUF component plan must bind its payload until loading consumes it"
                )
        elif self.payload_consumed:
            raise ValueError("only GGUF component plans consume a bound payload")
        if set(self.keys) != set(self.dtypes):
            raise ValueError(
                f"component {self.component!r}: keys and dtypes must cover the same model keys"
            )
        for key in self.transforms:
            if key not in self.keys:
                raise ValueError(
                    f"component {self.component!r}: transform for {key!r} has no mapped source"
                )
        for layer in self.quant:
            if f"{layer}.weight" not in self.keys:
                raise ValueError(
                    f"component {self.component!r}: quantized layer {layer!r} has no mapped weight"
                )
            if f"{layer}.weight" in self.transforms:
                raise ValueError(
                    f"component {self.component!r}: quantized layer {layer!r}"
                    " cannot also be transform-derived"
                )

    def without_payload_source(self) -> ComponentPlan[C]:
        """Return the realized plan metadata without retaining checkpoint bytes."""

        if self.source_format != "gguf":
            return self
        return replace(self, payload_source=None, payload_consumed=True)


@dataclass(frozen=True)
class ControlNetAssemblyPlan:
    """Complete header-only loading plan for one classic SD1.5 ControlNet.

    The family and source role are closed facts. ``claims`` contains every
    source key exactly once; there is no ignored or unclaimed surface.
    """

    controlnet: ComponentPlan[SD15ControlNetConfig]
    layout: ControlNetLayout
    source_layout: ControlNetSourceLayout
    asset_digest: str
    claims: tuple[str, ...]
    family_id: str = field(default="dinkster.sd15_controlnet", init=False)
    source_role: str = field(default="controlnet", init=False)

    def __post_init__(self) -> None:
        if self.controlnet.component != self.source_role:
            raise ValueError("ControlNet component must use the controlnet source role")
        if self.controlnet.config != self.layout.config:
            raise ValueError("ControlNet plan config and layout disagree")
        if set(self.controlnet.keys) != set(self.layout.keys):
            raise ValueError("ControlNet plan must map every canonical layout key")
        if self.source_layout not in ("canonical", "diffusers"):
            raise ValueError("ControlNet source_layout must be canonical or diffusers")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("ControlNet asset digest must be a canonical blake3 asset digest")
        if not isinstance(cast("object", self.claims), tuple):
            raise TypeError("ControlNet claims must be a tuple")
        if self.claims != tuple(sorted(self.claims)) or len(self.claims) != len(set(self.claims)):
            raise ValueError("ControlNet claims must be sorted and unique")
        mapped_sources = tuple(self.controlnet.keys.values())
        if len(mapped_sources) != len(set(mapped_sources)):
            raise ValueError("ControlNet source mapping must be one-to-one")
        if self.claims != tuple(sorted(mapped_sources)):
            raise ValueError("ControlNet claims must exactly match mapped source keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[SD15ControlNetConfig], ...]:
        """Components in canonical child-recipe identity order."""
        return (self.controlnet,)


@dataclass(frozen=True)
class T2IAdapterAssemblyPlan:
    """Complete header-only loading plan for the SD1.5 full adapter."""

    adapter: ComponentPlan[SD15T2IAdapterConfig]
    asset_digest: str
    claims: tuple[str, ...]
    family_id: str = field(default="dinkster.sd15_t2i_adapter", init=False)
    source_role: str = field(default="t2i_adapter", init=False)

    def __post_init__(self) -> None:
        if self.adapter.component != self.source_role:
            raise ValueError("T2I Adapter component must use the t2i_adapter source role")
        if set(self.adapter.keys) != set(sd15_t2i_adapter_layout(self.adapter.config)):
            raise ValueError("T2I Adapter plan must map every canonical key")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("T2I Adapter asset digest must be a canonical blake3 asset digest")
        if not isinstance(cast("object", self.claims), tuple):
            raise TypeError("T2I Adapter claims must be a tuple")
        if self.claims != tuple(sorted(self.claims)) or len(self.claims) != len(set(self.claims)):
            raise ValueError("T2I Adapter claims must be sorted and unique")
        mapped_sources = tuple(self.adapter.keys.values())
        if len(mapped_sources) != len(set(mapped_sources)):
            raise ValueError("T2I Adapter source mapping must be one-to-one")
        if self.claims != tuple(sorted(self.adapter.keys.values())):
            raise ValueError("T2I Adapter claims must exactly match mapped source keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[SD15T2IAdapterConfig], ...]:
        return (self.adapter,)


@dataclass(frozen=True)
class SD15IPAdapterAssemblyPlan:
    """Exact standard SD1.5 adapter and CLIP vision loading plan."""

    adapter: ComponentPlan[SD15IPAdapterConfig]
    clip_vision: ComponentPlan[ClipVisionConfig]
    adapter_asset_digest: str
    clip_vision_asset_digest: str
    family_id: str = field(default="dinkster.sd15", init=False)

    def __post_init__(self) -> None:
        if self.adapter.component != "ipadapter":
            raise ValueError("IP-Adapter plan must use the ipadapter component role")
        if self.clip_vision.component != "ipadapter_clip_vision":
            raise ValueError("IP-Adapter vision plan must use its canonical component role")
        if set(self.adapter.keys) != set(sd15_ipadapter_layout(self.adapter.config)):
            raise ValueError("IP-Adapter plan must map every canonical adapter key")
        expected_vision = set(clip_vision_layout(self.clip_vision.config))
        if set(self.clip_vision.keys) | set(self.clip_vision.absent) != expected_vision:
            raise ValueError("IP-Adapter vision plan must map the exact CLIP ViT-H layout")
        for digest in (self.adapter_asset_digest, self.clip_vision_asset_digest):
            if _ASSET_DIGEST_RE.fullmatch(digest) is None:
                raise ValueError("IP-Adapter asset digests must be canonical blake3 digests")

    @property
    def identity_components(
        self,
    ) -> tuple[ComponentPlan[SD15IPAdapterConfig], ComponentPlan[ClipVisionConfig]]:
        return self.adapter, self.clip_vision


@dataclass(frozen=True)
class SDXLControlLoRAAssemblyPlan:
    """Complete loading contract for an official SDXL Control-LoRA."""

    control_lora: ComponentPlan[SDXLControlLoRAConfig]
    asset_digest: str
    base_asset_digest: str
    claims: tuple[str, ...]
    family_id: str = field(default="dinkster.sdxl_control_lora", init=False)
    source_role: str = field(default="control_lora", init=False)

    def __post_init__(self) -> None:
        if self.control_lora.component != self.source_role:
            raise ValueError("Control-LoRA component must use the control_lora source role")
        if set(self.control_lora.keys) != set(sdxl_control_lora_layout(self.control_lora.config)):
            raise ValueError("Control-LoRA plan must map every canonical key")
        for name, digest in (
            ("Control-LoRA", self.asset_digest),
            ("Control-LoRA base model", self.base_asset_digest),
        ):
            if _ASSET_DIGEST_RE.fullmatch(digest) is None:
                raise ValueError(f"{name} asset digest must be a canonical blake3 asset digest")
        if self.claims != tuple(sorted(self.claims)) or len(self.claims) != len(set(self.claims)):
            raise ValueError("Control-LoRA claims must be sorted and unique")
        if self.claims != tuple(sorted(self.control_lora.keys.values())):
            raise ValueError("Control-LoRA claims must exactly match mapped source keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[SDXLControlLoRAConfig], ...]:
        return (self.control_lora,)


@dataclass(frozen=True)
class SDXLControlNetAssemblyPlan:
    """Complete header-only loading contract for classic SDXL ControlNet."""

    controlnet: ComponentPlan[SDXLControlNetConfig]
    asset_digest: str
    claims: tuple[str, ...]
    family_id: str = field(default="dinkster.sdxl_controlnet", init=False)
    source_role: str = field(default="controlnet", init=False)

    def __post_init__(self) -> None:
        if self.controlnet.component != self.source_role:
            raise ValueError("SDXL ControlNet component must use the controlnet source role")
        if set(self.controlnet.keys) != set(sdxl_controlnet_layout(self.controlnet.config)):
            raise ValueError("SDXL ControlNet plan must map every canonical key")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("SDXL ControlNet asset digest must be canonical blake3")
        if self.claims != tuple(sorted(self.claims)) or len(self.claims) != len(set(self.claims)):
            raise ValueError("SDXL ControlNet claims must be sorted and unique")
        if self.claims != tuple(sorted(self.controlnet.keys.values())):
            raise ValueError("SDXL ControlNet claims must exactly match mapped source keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[SDXLControlNetConfig], ...]:
        return (self.controlnet,)


@dataclass(frozen=True)
class SDXLControlNetUnionAssemblyPlan:
    """Complete header-only loading contract for xinsir SDXL ControlNet Union."""

    controlnet_union: ComponentPlan[SDXLControlNetUnionConfig]
    asset_digest: str
    claims: tuple[str, ...]
    family_id: str = field(default="dinkster.sdxl_controlnet_union", init=False)
    source_role: str = field(default="controlnet_union", init=False)

    def __post_init__(self) -> None:
        if self.controlnet_union.component != self.source_role:
            raise ValueError("Union component must use the controlnet_union source role")
        if set(self.controlnet_union.keys) != set(
            sdxl_controlnet_union_layout(self.controlnet_union.config)
        ):
            raise ValueError("Union plan must map every canonical key")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Union asset digest must be a canonical blake3 asset digest")
        if self.claims != tuple(sorted(self.claims)) or len(self.claims) != len(set(self.claims)):
            raise ValueError("Union claims must be sorted and unique")
        if self.claims != tuple(sorted(self.controlnet_union.keys.values())):
            raise ValueError("Union claims must exactly match mapped source keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[SDXLControlNetUnionConfig], ...]:
        return (self.controlnet_union,)


@dataclass(frozen=True)
class FluxAssemblyPlan:
    """The full Flux loading contract: which family the DiT is (dev
    or schnell - sampling and latent descriptors ride the family),
    plus either classic CLIP-L/T5 or one Qwen text component.
    ``unclaimed`` lists combined-checkpoint keys no component consumed
    (diagnostic only; the reference silently drops such keys)."""

    family: ModelFamily
    diffusion: ComponentPlan[FluxConfig]
    clip_l: ComponentPlan[ClipTextConfig] | None
    t5xxl: ComponentPlan[T5Config] | None
    vae: ComponentPlan[KLConfig]
    unclaimed: tuple[str, ...] = ()
    qwen3_2b: ComponentPlan[QwenTextConfig] | None = None

    def __post_init__(self) -> None:
        if self.diffusion.config.vec_in_dim is None and self.qwen3_2b is None:
            raise ValueError("a vector-free Ovis Flux assembly requires qwen3_2b text")
        classic_complete = self.clip_l is not None and self.t5xxl is not None
        if self.qwen3_2b is None and not classic_complete:
            raise ValueError("a classic Flux assembly requires clip_l and t5xxl")
        if self.qwen3_2b is not None and not (self.clip_l is None and self.t5xxl is None):
            raise ValueError("a Qwen Flux assembly cannot also carry classic text components")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any] | None, ...]:
        """Components in the canonical native-identity order."""
        if self.qwen3_2b is not None:
            return (self.diffusion, self.qwen3_2b, self.vae)
        return (self.diffusion, self.clip_l, self.t5xxl, self.vae)


@dataclass(frozen=True)
class Wan21AssemblyPlan:
    """Complete native loading contract for an official Wan core model."""

    family: ModelFamily
    diffusion: ComponentPlan[Wan21Config]
    umt5xxl: ComponentPlan[T5Config]
    vae: ComponentPlan[Wan21VAEConfig | Wan22VAEConfig]
    tokenizer_source_key: str
    clip_vision: ComponentPlan[ClipVisionConfig] | None = field(default=None, kw_only=True)
    tokenizer_vendored: bool = field(default=False, kw_only=True)
    unclaimed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family is WAN21:
            valid_diffusion = self.diffusion.config in (
                WAN21_T2V_1_3B,
                WAN21_CAUSAL_AR_1_3B,
                WAN21_FLOW_RVS_1_3B,
                WAN21_T2V_14B,
                WAN21_HUMO_17B,
                WAN21_FLF_I2V_14B,
                WAN21_FUN_CONTROL_1_3B,
                WAN21_FUN_INPAINT_1_3B,
                WAN21_CAMERA_1_3B,
                WAN21_CAMERA_14B,
                WAN21_ANIMATE2_14B,
                WAN21_I2V_14B,
                WAN21_SCAIL_14B,
                WAN21_SCAIL2_14B,
                WAN22_ANIMATE_14B,
                WAN22_BERNINI_14B,
                WAN22_S2V_14B,
                WAN22_WANDANCER_14B,
                WAN22_I2V_14B,
                WAN22_CAMERA_14B,
                WAN22_FUN_CONTROL_14B,
                WAN21_VACE_1_3B,
                WAN21_VACE_14B,
            )
            expected_vae = (
                WAN21_FLOW_RVS_VAE_CONFIG
                if self.diffusion.config == WAN21_FLOW_RVS_1_3B
                else WAN21_VAE_CONFIG
            )
            valid_vae = self.vae.config == expected_vae
        elif self.family is WAN22:
            valid_diffusion = self.diffusion.config in (
                WAN22_TI2V_5B,
                WAN22_FUN_CONTROL_5B,
                WAN22_FUN_INPAINT_5B,
            )
            valid_vae = self.vae.config == WAN22_VAE_CONFIG
        else:
            valid_diffusion = valid_vae = False
        if not valid_diffusion:
            raise ValueError("Wan native assembly requires an official model profile")
        if self.umt5xxl.config != UMT5_XXL_CONFIG:
            raise ValueError("Wan native assembly requires UMT5-XXL")
        if not valid_vae:
            raise ValueError("Wan native assembly requires the matching causal VAE")
        if bool(self.tokenizer_source_key) == self.tokenizer_vendored:
            raise ValueError(
                "Wan assembly requires exactly one tokenizer origin:"
                " a checkpoint source key or the vendored model"
            )
        if self.tokenizer_vendored and self.umt5xxl.source_format != "gguf":
            raise ValueError(
                "the vendored tokenizer is only for GGUF text sources;"
                " safetensors sources carry spiece_model inline"
            )
        image_to_video = self.family is WAN21 and self.diffusion.config.model_type == "i2v"
        if image_to_video != (self.clip_vision is not None):
            raise ValueError(
                "Wan 2.1 I2V requires CLIP vision and other profiles must not carry it"
            )

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return tuple(
            component
            for component in (self.diffusion, self.umt5xxl, self.clip_vision, self.vae)
            if component is not None
        )


Wan21StandaloneComponentRole = Literal["diffusion", "umt5xxl", "vae"]


@dataclass(frozen=True)
class Wan21StandaloneComponentPlan:
    """One independently loaded Wan 2.1 T2V component."""

    role: Wan21StandaloneComponentRole
    component: ComponentPlan[Wan21Config | T5Config | Wan21VAEConfig]
    tokenizer_source_key: str = ""

    def __post_init__(self) -> None:
        valid = (
            (
                self.role == "diffusion"
                and self.component.component == "diffusion"
                and self.component.config
                in (
                    WAN21_T2V_14B,
                    WAN21_CAUSAL_AR_1_3B,
                    WAN21_HUMO_17B,
                    WAN22_S2V_14B,
                    WAN22_WANDANCER_14B,
                )
            )
            or (
                self.role == "umt5xxl"
                and self.component.component == "umt5xxl"
                and self.component.config == UMT5_XXL_CONFIG
                and bool(self.tokenizer_source_key)
            )
            or (
                self.role == "vae"
                and self.component.component == "vae"
                and self.component.config == WAN21_VAE_CONFIG
            )
        )
        if not valid:
            raise ValueError("standalone Wan 2.1 components require exact supported roles")
        if self.role != "umt5xxl" and self.tokenizer_source_key:
            raise ValueError("only the standalone UMT5 component carries a tokenizer source")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.component,)


LTXVStandaloneComponentRole = Literal["diffusion", "t5xxl", "vae"]


@dataclass(frozen=True)
class LTXVStandaloneComponentPlan:
    """One independently loaded classic LTX-Video component."""

    role: LTXVStandaloneComponentRole
    component: ComponentPlan[LTXVConfig | T5Config | LTXVideoVAEConfig]

    def __post_init__(self) -> None:
        valid = (
            (
                self.role == "diffusion"
                and self.component.component == "diffusion"
                and self.component.config in (LTXV_2B_V09_CONFIG, LTXV_2B_V095_CONFIG)
            )
            or (
                self.role == "t5xxl"
                and self.component.component == "t5xxl"
                and self.component.config == T5_XXL_CONFIG
            )
            or (
                self.role == "vae"
                and self.component.component == "vae"
                and self.component.config in (LTXV_2B_V09_VAE_CONFIG, LTXV_2B_V095_VAE_CONFIG)
            )
        )
        if not valid:
            raise ValueError("standalone LTX-Video components require exact supported roles")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.component,)


@dataclass(frozen=True)
class LTXAVAudioCodecPlan:
    """One independently loaded LTX-2 audio VAE and vocoder bundle."""

    family: ModelFamily
    audio_vae: ComponentPlan[LTXAudioVAEConfig]
    vocoder: ComponentPlan[LTXVocoderConfig] | ComponentPlan[LTXVocoderBWEConfig]
    unclaimed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family is not LTXAV:
            raise ValueError("LTX-2 audio codec requires the audio-video family")
        if self.audio_vae.config != LTXAV_19B_AUDIO_VAE_CONFIG:
            raise ValueError("LTX-2 audio codec requires the official causal audio VAE")
        if self.vocoder.config not in (LTXAV_19B_VOCODER_CONFIG, LTXAV_BWE_VOCODER_CONFIG):
            raise ValueError("LTX-2 audio codec requires the official vocoder profile")
        if self.audio_vae.path != self.vocoder.path:
            raise ValueError("LTX-2 audio codec components must share one source")
        if self.unclaimed:
            raise ValueError("LTX-2 audio codec source must be fully claimed")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.audio_vae, self.vocoder)


LTXAVStandaloneComponentRole = Literal[
    "diffusion",
    "duration_head",
    "gemma3_12b",
    "gemma4_12b",
    "text_projection",
    "connectors",
    "latent_upscaler",
    "vae",
]


@dataclass(frozen=True)
class LTXAVStandaloneComponentPlan:
    """One independently loaded LTX-2 component."""

    role: LTXAVStandaloneComponentRole
    component: ComponentPlan[
        LTXAVConfig
        | LTXDurationHeadConfig
        | LTXLatentUpsamplerConfig
        | GemmaTextConfig
        | LtxTextProjectionKind
        | LtxConnectorConfig
        | LTXDiffusionVideoVAEConfig
        | LTXVideoVAEConfig
    ]
    tokenizer_source_key: str = ""

    def __post_init__(self) -> None:
        diffusion_config = self.component.config
        diffusion_profile = (
            isinstance(diffusion_config, LTXAVConfig)
            and diffusion_config
            in (
                replace(
                    LTXAV_19B_CONFIG,
                    av_ca_timestep_scale_multiplier=(
                        diffusion_config.av_ca_timestep_scale_multiplier
                    ),
                    use_keyframes_abs_pos_embedding=(
                        diffusion_config.use_keyframes_abs_pos_embedding
                    ),
                ),
                replace(
                    LTXAV_22B_V23_CONFIG,
                    av_ca_timestep_scale_multiplier=(
                        diffusion_config.av_ca_timestep_scale_multiplier
                    ),
                    use_keyframes_abs_pos_embedding=(
                        diffusion_config.use_keyframes_abs_pos_embedding
                    ),
                ),
                replace(
                    LTXAV_22B_V25_CONFIG,
                    av_ca_timestep_scale_multiplier=(
                        diffusion_config.av_ca_timestep_scale_multiplier
                    ),
                    use_keyframes_abs_pos_embedding=(
                        diffusion_config.use_keyframes_abs_pos_embedding
                    ),
                ),
            )
            and math.isfinite(diffusion_config.av_ca_timestep_scale_multiplier)
            and diffusion_config.av_ca_timestep_scale_multiplier > 0.0
        )
        valid = (
            (
                self.role == "diffusion"
                and self.component.component == "diffusion"
                and diffusion_profile
            )
            or (
                self.role == "duration_head"
                and self.component.component == "duration_head"
                and self.component.config == LTXAV_DURATION_HEAD_CONFIG
            )
            or (
                self.role in ("gemma3_12b", "gemma4_12b")
                and self.component.component == self.role
                and self.component.config
                == (GEMMA3_LTX_12B_CONFIG if self.role == "gemma3_12b" else GEMMA4_LTX_12B_CONFIG)
                and bool(self.tokenizer_source_key)
            )
            or (
                self.role == "text_projection"
                and self.component.component == "text_projection"
                and self.component.config in ("single_linear", "dual_linear", "dual_linear_gemma4")
            )
            or (
                self.role == "connectors"
                and self.component.component == "connectors"
                and self.component.config == LTX_TEXT_CONNECTOR_CONFIG
            )
            or (
                self.role == "latent_upscaler"
                and self.component.component == "latent_upscaler"
                and self.component.config == LTX_LATENT_UPSAMPLER_CONFIG
            )
            or (
                self.role == "vae"
                and self.component.component == "vae"
                and self.component.config
                in (
                    LTXAV_19B_VAE_CONFIG,
                    LTXAV_22B_V23_VAE_CONFIG,
                    LTXAV_22B_V25_VAE_CONFIG,
                )
            )
        )
        if not valid:
            raise ValueError("standalone LTX-2 components require exact supported roles")
        if self.role not in ("gemma3_12b", "gemma4_12b") and self.tokenizer_source_key:
            raise ValueError("only the standalone Gemma component carries a tokenizer source")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.component,)


@dataclass(frozen=True)
class ZImageAssemblyPlan:
    """Complete latent or pixel-space Z-Image loading contract."""

    family: ModelFamily
    diffusion: ComponentPlan[ZImageConfig | ZImagePixelConfig]
    qwen3_4b: ComponentPlan[QwenTextConfig]
    vae: ComponentPlan[KLConfig] | None
    unclaimed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = {
            Z_IMAGE_CONFIG.family_id: (Z_IMAGE, Z_IMAGE_CONFIG),
            Z_IMAGE_PIXEL_CONFIG.family_id: (Z_IMAGE_PIXEL_SPACE, Z_IMAGE_PIXEL_CONFIG),
        }
        family, config = expected[self.diffusion.config.family_id]
        if self.family is not family or self.diffusion.config is not config:
            raise ValueError(
                "Z-Image assembly requires the exact native family and diffusion profile"
            )
        if self.qwen3_4b.config.architecture != "z_image_qwen3_4b":
            raise ValueError("Z-Image assembly requires the Qwen3-4B text profile")
        if config is Z_IMAGE_CONFIG:
            if self.vae is None or self.vae.config.embed_dim != Z_IMAGE_CONFIG.latent_channels:
                raise ValueError("latent Z-Image requires a matching VAE")
        elif self.vae is not None:
            raise ValueError("pixel-space Z-Image must not carry a VAE component")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return tuple(
            component for component in (self.diffusion, self.qwen3_4b, self.vae) if component
        )


@dataclass(frozen=True)
class QwenImageAssemblyPlan:
    """Complete loading contract for one supported Qwen Image variant."""

    family: ModelFamily
    diffusion: ComponentPlan[QwenImageConfig]
    qwen2_5_vl_7b: ComponentPlan[QwenImageTextConfig]
    vae: ComponentPlan[Wan21VAEConfig]
    unclaimed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family is not QWEN_IMAGE or self.diffusion.config not in (
            QWEN_IMAGE_CONFIG,
            QWEN_IMAGE_EDIT_2511_CONFIG,
            QWEN_IMAGE_LAYERED_CONFIG,
        ):
            raise ValueError("Qwen Image assembly requires an exact diffusion profile")
        if self.qwen2_5_vl_7b.config != QWEN_IMAGE_TEXT_CONFIG:
            raise ValueError("Qwen Image assembly requires the Qwen2.5-VL-7B text profile")
        if self.vae.config != WAN21_VAE_CONFIG:
            raise ValueError("Qwen Image assembly requires the Wan 2.1 VAE profile")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.diffusion, self.qwen2_5_vl_7b, self.vae)


#: Flux2 composition wiring: family, exact DiT config, reference
#: text-encoder slot name, combined-checkpoint text prefix, accepted
#: text architectures (comfy/sd.py flux2/klein clip targets
#: @ b78cec87). Dev accepts the full Mistral3-Small tower and the
#: layer-pruned tower the official flux2-dev release ships.
_FLUX2_VARIANTS: tuple[tuple[ModelFamily, FluxConfig, str, str, tuple[str, ...]], ...] = (
    (
        FLUX2_DEV,
        FLUX2_DEV_CONFIG,
        "mistral3_24b",
        FLUX2_MISTRAL_PREFIX,
        ("mistral3_24b", "mistral3_24b_pruned"),
    ),
    (FLUX2_KLEIN_9B, FLUX2_KLEIN_9B_CONFIG, "qwen3_8b", FLUX2_QWEN3_8B_PREFIX, ("klein_qwen3_8b",)),
    (FLUX2_KLEIN_4B, FLUX2_KLEIN_4B_CONFIG, "qwen3_4b", FLUX2_QWEN3_4B_PREFIX, ("klein_qwen3_4b",)),
)


@dataclass(frozen=True)
class Flux2AssemblyPlan:
    """Complete loading contract for one published Flux2 release: the
    exact DiT geometry rides the family, the text slot carries the
    family's reference encoder (Mistral3-Small for dev, Qwen3 for
    Klein), and the VAE is the packed batch-norm KL variant."""

    family: ModelFamily
    diffusion: ComponentPlan[FluxConfig]
    text_encoder: ComponentPlan[QwenTextConfig]
    vae: ComponentPlan[KLConfig]
    unclaimed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        variant = next(
            (entry for entry in _FLUX2_VARIANTS if entry[0] is self.family),
            None,
        )
        if variant is None or self.diffusion.config is not variant[1]:
            raise ValueError("Flux2 assembly requires a published family and diffusion profile")
        if self.text_encoder.config.architecture not in variant[4]:
            raise ValueError(
                f"Flux2 assembly for {self.family.id} requires one of the"
                f" {', '.join(variant[4])} text profiles"
            )
        if (
            not self.vae.config.batch_norm_latent
            or self.vae.config.latent_channels != FLUX2_LATENT_CHANNELS
        ):
            raise ValueError("Flux2 assembly requires the 128-channel packed batch-norm KL VAE")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.diffusion, self.text_encoder, self.vae)


@dataclass(frozen=True)
class QwenImageControlPlan:
    """Exact loading contract for a maintained Qwen Image ControlNet."""

    control: ComponentPlan[QwenImageControlConfig]
    asset_digest: str
    source_role: str = field(default="qwen_image_control", init=False)

    def __post_init__(self) -> None:
        if self.control.component != self.source_role:
            raise ValueError("Qwen Image control component must use its dedicated source role")
        if all(
            self.control.config is not profile
            for profile in (
                QWEN_IMAGE_INSTANTX_CONTROL,
                QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
                QWEN_IMAGE_FUN_CONTROL,
            )
        ):
            raise ValueError("Qwen Image control plan requires an exact maintained profile")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Qwen Image control asset digest must be canonical blake3 identity")
        if set(self.control.keys) != set(qwen_image_control_layout(self.control.config)):
            raise ValueError("Qwen Image control plan must map the complete checkpoint layout")

    @property
    def identity_components(self) -> tuple[ComponentPlan[QwenImageControlConfig], ...]:
        return (self.control,)


@dataclass(frozen=True)
class QwenImageDiffSynthPlan:
    """Exact loading contract for a Qwen Image DiffSynth block patch."""

    patch: ComponentPlan[QwenImageDiffSynthConfig]
    asset_digest: str
    source_role: str = field(default="qwen_image_diffsynth", init=False)

    def __post_init__(self) -> None:
        if self.patch.component != self.source_role:
            raise ValueError("Qwen Image DiffSynth component must use its dedicated source role")
        if (
            self.patch.config is not QWEN_IMAGE_DIFFSYNTH
            and self.patch.config is not QWEN_IMAGE_DIFFSYNTH_INPAINT
        ):
            raise ValueError("Qwen Image DiffSynth plan requires an exact maintained profile")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Qwen Image DiffSynth asset digest must be canonical blake3 identity")
        if set(self.patch.keys) != set(qwen_image_diffsynth_layout(self.patch.config)):
            raise ValueError("Qwen Image DiffSynth plan must map the complete checkpoint layout")

    @property
    def identity_components(self) -> tuple[ComponentPlan[QwenImageDiffSynthConfig], ...]:
        return (self.patch,)


@dataclass(frozen=True)
class Wan21Uni3CPlan:
    """Exact loading contract for the maintained Wan 2.1 Uni3C patch."""

    patch: ComponentPlan[Wan21Uni3CConfig]
    asset_digest: str
    source_role: str = field(default="wan21_uni3c", init=False)

    def __post_init__(self) -> None:
        if self.patch.component != self.source_role:
            raise ValueError("Wan 2.1 Uni3C component must use its dedicated source role")
        if self.patch.config is not WAN21_UNI3C:
            raise ValueError("Wan 2.1 Uni3C plan requires the exact maintained profile")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Wan 2.1 Uni3C asset digest must be canonical blake3 identity")
        if set(self.patch.keys) != set(wan21_uni3c_model_layout()):
            raise ValueError("Wan 2.1 Uni3C plan must map the complete checkpoint layout")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Wan21Uni3CConfig], ...]:
        return (self.patch,)


@dataclass(frozen=True)
class Wan21MultiTalkPlan:
    """Exact loading contract for the maintained Wan 2.1 MultiTalk patch."""

    patch: ComponentPlan[Wan21MultiTalkConfig]
    asset_digest: str
    source_role: str = field(default="wan21_multitalk", init=False)

    def __post_init__(self) -> None:
        if self.patch.component != self.source_role:
            raise ValueError("Wan 2.1 MultiTalk component must use its dedicated source role")
        if self.patch.config is not WAN21_MULTITALK:
            raise ValueError("Wan 2.1 MultiTalk plan requires the exact maintained profile")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Wan 2.1 MultiTalk asset digest must be canonical blake3 identity")
        if set(self.patch.keys) != set(wan21_multitalk_model_layout()):
            raise ValueError("Wan 2.1 MultiTalk plan must map the complete checkpoint layout")

    @property
    def identity_components(self) -> tuple[ComponentPlan[Wan21MultiTalkConfig], ...]:
        return (self.patch,)


@dataclass(frozen=True)
class ZImageControlPlan:
    """Exact loading contract for a separately loaded Z-Image control patch."""

    control: ComponentPlan[ZImageControlConfig]
    asset_digest: str
    source_role: str = field(default="z_image_control", init=False)

    def __post_init__(self) -> None:
        if self.control.component != self.source_role:
            raise ValueError("Z-Image control component must use its dedicated source role")
        if self.control.config is not Z_IMAGE_CONTROL_CONFIG:
            raise ValueError("Z-Image control plan requires the exact Union profile")
        if _ASSET_DIGEST_RE.fullmatch(self.asset_digest) is None:
            raise ValueError("Z-Image control asset digest must be canonical blake3 identity")
        if set(self.control.keys) != set(z_image_control_layout()):
            raise ValueError("Z-Image control plan must map all 136 checkpoint keys")

    @property
    def identity_components(self) -> tuple[ComponentPlan[ZImageControlConfig], ...]:
        return (self.control,)


@dataclass(frozen=True)
class _Extracted:
    """A component's slice of one source: stripped-key geometries,
    stripped-key -> full-source-key mapping, quant layers re-keyed to
    stripped names, and sibling keys under the component root that
    the transformer prefix does not cover."""

    path: Path
    geometries: dict[str, TensorGeometry]
    source_keys: dict[str, str]
    quant: dict[str, LayerQuant]
    ignored: tuple[str, ...]
    source_format: str
    runtime_facts: tuple[str, ...]
    payload_source: WeightSource | None


def _extract(
    source: WeightSource,
    path: Path,
    component: str,
    prefix: str,
    *,
    root: str | None = None,
) -> _Extracted:
    """Slice ``source`` down to one component: quantization is
    classified AT COMPONENT SCOPE (split_quantization's ``prefix`` -
    combined checkpoints park the legacy marker and metadata layer
    names under each component's prefix), so detectors see pure
    stripped architecture. Artifact keys come back stripped and are
    re-prefixed here into full source keys for the executor. ``root``
    widens the ignored-key scan to a parent namespace (text encoders
    park ``logit_scale`` beside the transformer); ``root=""`` scans
    the whole source (split files, where out-of-prefix keys are
    deliberately unread)."""
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    try:
        split = split_quantization(
            geometries,
            source.metadata(),
            prefix=prefix,
            payload_reader=(
                source.read_uint8_configuration
                if isinstance(source, ConfigurationPayloadSource)
                else None
            ),
        )
    except QuantizationError as error:
        raise AssemblyError(f"{component}: {error}") from error
    source_keys = {stripped: prefix + stripped for stripped in split.architecture}
    quant = {
        entry.layer: replace(
            entry,
            weight=prefix + entry.weight,
            weight_scale=("" if not entry.weight_scale else prefix + entry.weight_scale),
            input_scale=None if entry.input_scale is None else prefix + entry.input_scale,
            weight_scale_2=(
                None if entry.weight_scale_2 is None else prefix + entry.weight_scale_2
            ),
            pre_quant_scale=(
                None if entry.pre_quant_scale is None else prefix + entry.pre_quant_scale
            ),
            config=None if entry.config is None else prefix + entry.config,
            payloads={name: prefix + key for name, key in entry.payloads.items()},
        )
        for entry in split.layers.values()
    }
    scan = prefix if root is None else root
    ignored = tuple(
        key for key in geometries if key.startswith(scan) and not key.startswith(prefix)
    )
    source_format = getattr(source, "source_format", "safetensors")
    runtime_facts = tuple(getattr(source, "runtime_facts", ()))
    payload_source = source if source_format == "gguf" else None
    return _Extracted(
        path,
        dict(split.architecture),
        source_keys,
        quant,
        ignored,
        source_format,
        runtime_facts,
        payload_source,
    )


def _checkpoint_component_root(key: str) -> str | None:
    """Identify serialization namespaces, keeping aliases of one component together."""
    for container in ("text_encoders.", "conditioner.embedders."):
        if key.startswith(container):
            return container + key[len(container) :].split(".", 1)[0] + "."
    for aliases in (
        (FLUX_DIFFUSION_PREFIX,),
        (FLUX_VAE_PREFIX, SD_VAE_PREFIX),
        (_SD15_CLIP_ROOT,),
        (LTXAV_AUDIO_VAE_PREFIX,),
        (LTXAV_VOCODER_PREFIX,),
        (LTXAV_PROJECTION_PREFIX,),
    ):
        if key.startswith(aliases):
            return aliases[0]
    return None


def _component_source(
    component: str,
    split: WeightSource | None,
    checkpoint: WeightSource | None,
    prefix: str,
    *,
    root: str | None = None,
    split_prefixes: tuple[str, ...] = ("",),
) -> _Extracted:
    """Pick the component's source (split wins over combined) and
    extract its slice, or refuse naming what was searched."""
    if split is not None:
        # Longest prefix with any keys wins; "" is the bare fallback
        # (unet_prefix_from_state_dict, comfy/model_detection.py
        # @ b78cec87 - split DiT files ship bare or re-prefixed).
        best = ""
        candidates = tuple(dict.fromkeys((*split_prefixes, prefix)))
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate and any(key.startswith(candidate) for key in split.keys()):
                best = candidate
                break
        extracted = _extract(split, _source_path(split, component), component, best, root="")
        namespace = _checkpoint_component_root(best)
        if best and namespace is not None:
            extracted = replace(
                extracted,
                ignored=tuple(
                    key
                    for key in extracted.ignored
                    if _checkpoint_component_root(key) in (None, namespace)
                    or (root is not None and key.startswith(root))
                ),
            )
        if not extracted.geometries:
            raise AssemblyError(
                f"{component}: the split source has no keys under any of {candidates!r}"
            )
        return extracted
    if checkpoint is not None:
        extracted = _extract(
            checkpoint, _source_path(checkpoint, component), component, prefix, root=root
        )
        if extracted.geometries:
            return extracted
        raise AssemblyError(
            f"{component}: the combined checkpoint has no keys under"
            f" {prefix!r} and no split source was given"
        )
    raise AssemblyError(f"{component}: no source - pass a combined checkpoint or a split file")


def _source_path(source: WeightSource, component: str) -> Path:
    path = getattr(source, "path", None)
    if not isinstance(path, Path):
        raise AssemblyError(
            f"{component}: weight source carries no file path; the executor cannot read it"
        )
    return path


def _plan(
    component: str,
    extracted: _Extracted,
    config: C,
    *,
    renames: Mapping[str, str] | None = None,
    optional: frozenset[str] = frozenset(),
    drop: frozenset[str] = frozenset(),
    transforms: Mapping[str, TensorTransform] | None = None,
) -> ComponentPlan[C]:
    """Assemble the ComponentPlan bookkeeping: apply stripped-key ->
    model-key ``renames``, drop known aliases, record optional keys
    the source lacks."""
    unsupported = sorted(
        {
            entry.format
            for entry in extracted.quant.values()
            if entry.format is not None and entry.format not in SUPPORTED_QUANT_FORMATS
        }
    )
    if unsupported:
        raise AssemblyError(
            f"{component}: typed quantization format(s)"
            f" {', '.join(unsupported)} have no runtime implementation"
            " (ROADMAP: Native inference)"
        )
    keys: dict[str, str] = {}
    dtypes: dict[str, DType] = {}
    ignored = list(extracted.ignored)
    for stripped, source_key in extracted.source_keys.items():
        if stripped in drop:
            ignored.append(source_key)
            continue
        model_key = (renames or {}).get(stripped, stripped)
        keys[model_key] = source_key
        dtypes[model_key] = extracted.geometries[stripped].dtype
    absent = tuple(sorted(key for key in optional if key not in keys))
    # A quantized layer's MODEL name follows its weight's rename
    # (rename maps are keyed by full parameter keys, not layer names).
    quant: dict[str, LayerQuant] = {}
    for layer, entry in extracted.quant.items():
        weight_key = f"{layer}.weight"
        model_weight = (renames or {}).get(weight_key, weight_key)
        quant[model_weight[: -len(".weight")]] = entry
    return ComponentPlan(
        component=component,
        path=extracted.path,
        config=config,
        keys=keys,
        dtypes=dtypes,
        quant=quant,
        source_format=extracted.source_format,
        ignored=tuple(ignored),
        absent=absent,
        transforms=transforms or {},
        runtime_facts=extracted.runtime_facts,
        payload_source=extracted.payload_source,
    )


def _norm_renames(geometries: Mapping[str, TensorGeometry]) -> dict[str, str]:
    """stripped-key -> model-key renames produced by
    :func:`normalize_flux_keys` (RMSNorm ``.scale`` spelling)."""
    renames: dict[str, str] = {}
    for key in geometries:
        if key.endswith("_norm.scale"):
            renames[key] = key[: -len(".scale")] + ".weight"
    return renames


def _kl_conversion(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[dict[str, str], dict[str, TensorTransform]]:
    """Plan canonical model keys and payload-only Diffusers reshapes."""
    renames: dict[str, str] = {}
    transforms: dict[str, TensorTransform] = {}
    # Detection calls this first, but keep the conversion seam safe on
    # its own: validation must happen before _plan can collapse aliases.
    normalize_kl_keys(geometries)
    diffusers = is_diffusers_kl(geometries)
    for key in geometries:
        model_key = diffusers_kl_key(key) if diffusers else key
        for old, new in KL_PREFIX_RENAMES.items():
            if model_key.startswith(old):
                model_key = new + model_key[len(old) :]
                break
        if model_key != key:
            renames[key] = model_key
        if (
            diffusers
            and model_key.endswith((".q.weight", ".k.weight", ".v.weight", ".proj_out.weight"))
            and len(geometries[key].shape) == 2
        ):
            transforms[model_key] = LinearToConv2D()
    return renames, transforms


def _detect_flux_family_normalized(
    *, checkpoint: WeightSource | None = None, diffusion: WeightSource | None = None
) -> tuple[_Extracted, FluxConfig, ModelFamily]:
    """Authoritative Flux diffusion probe after quant payload normalization."""
    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    try:
        config = detect_flux_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"diffusion: {error}") from error
    return extracted, config, FLUX_DEV if config.guidance_embed else FLUX_SCHNELL


def plan_flux_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    t5xxl: WeightSource | None = None,
    qwen3_2b: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> FluxAssemblyPlan:
    """Plan a classic or Ovis-text Flux assembly from headers alone.

    ``checkpoint`` is a combined file; the split component arguments
    are split files, each overriding the combined checkpoint for its
    component when both are present. ``qwen3_2b`` replaces both classic
    text slots. Raises :class:`AssemblyError`
    (chaining the underlying detector error) when any component is
    missing, unrecognized, or carries an unported quantization.
    """
    if all(source is None for source in (checkpoint, diffusion, clip_l, t5xxl, qwen3_2b, vae)):
        raise AssemblyError("no sources given")

    # --- diffusion -------------------------------------------------
    extracted, flux_config, family = _detect_flux_family_normalized(
        checkpoint=checkpoint, diffusion=diffusion
    )
    diffusion_plan = _plan(
        "diffusion",
        extracted,
        flux_config,
        renames=_norm_renames(extracted.geometries),
    )
    clip_plan: ComponentPlan[ClipTextConfig] | None = None
    t5_plan: ComponentPlan[T5Config] | None = None
    qwen_plan: ComponentPlan[QwenTextConfig] | None = None
    combined_qwen = checkpoint is not None and any(
        key.startswith(FLUX_QWEN_PREFIX) for key in checkpoint.keys()
    )
    use_qwen = qwen3_2b is not None or combined_qwen
    if flux_config.vec_in_dim is None and not use_qwen:
        raise AssemblyError("text: vector-free Ovis diffusion requires the qwen3_2b component")
    if use_qwen:
        combined_classic = checkpoint is not None and any(
            key.startswith(_TEXT_ENCODER_ROOTS) and not key.startswith("text_encoders.qwen3_2b.")
            for key in checkpoint.keys()
        )
        if clip_l is not None or t5xxl is not None or combined_classic:
            raise AssemblyError(
                "text: Qwen is an alternative component slot; do not also pass clip_l/t5xxl"
            )
        extracted = _component_source(
            "qwen3_2b",
            qwen3_2b,
            checkpoint,
            FLUX_QWEN_PREFIX,
            root="text_encoders.qwen3_2b.",
            split_prefixes=("model.", FLUX_QWEN_PREFIX),
        )
        ignored_roots = {
            key.removeprefix("text_encoders.qwen3_2b.").split(".", 1)[0]
            for key in extracted.ignored
        }
        unknown_roots = ignored_roots - _OVIS_IGNORED_ROOTS
        if unknown_roots:
            raise AssemblyError(
                "qwen3_2b: unexpected non-text artifact roots: " + ", ".join(sorted(unknown_roots))
            )
        try:
            qwen_config = detect_qwen_text_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"qwen3_2b: {error}") from error
        if qwen_config.hidden_size != flux_config.context_in_dim:
            raise AssemblyError(
                f"qwen3_2b: {qwen_config.hidden_size}-wide text output does not match"
                f" diffusion context width {flux_config.context_in_dim}"
            )
        qwen_plan = _plan("qwen3_2b", extracted, qwen_config)
    else:
        # --- clip_l ------------------------------------------------
        extracted = _component_source(
            "clip_l",
            clip_l,
            checkpoint,
            FLUX_CLIP_L_PREFIX,
            root=_TEXT_ENCODER_ROOTS[0],
        )
        try:
            clip_config = detect_clip_text_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"clip_l: {error}") from error
        clip_plan = _plan(
            "clip_l",
            extracted,
            clip_config,
            optional=CLIP_TEXT_OPTIONAL_KEYS,
        )

        # --- t5xxl -------------------------------------------------
        extracted = _component_source(
            "t5xxl",
            t5xxl,
            checkpoint,
            FLUX_T5XXL_PREFIX,
            root=_TEXT_ENCODER_ROOTS[1],
        )
        try:
            t5_config = detect_t5_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"t5xxl: {error}") from error
        if t5_config.model_type != "t5":
            raise AssemblyError("t5xxl: classic Flux requires the classic T5-XXL layout")
        # The checkpoints' encoder.embed_tokens.weight duplicates
        # shared.weight; the reference loads strict=False and drops it.
        t5_plan = _plan(
            "t5xxl",
            extracted,
            t5_config,
            drop=T5_TEXT_OPTIONAL_KEYS,
        )

    # --- vae -------------------------------------------------------
    extracted = _component_source("vae", vae, checkpoint, FLUX_VAE_PREFIX)
    try:
        kl_config = detect_kl_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"vae: {error}") from error
    vae_renames, vae_transforms = _kl_conversion(extracted.geometries)
    vae_plan = _plan(
        "vae",
        extracted,
        kl_config,
        renames=vae_renames,
        transforms=vae_transforms,
    )

    # --- unclaimed combined keys ------------------------------------
    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        claimed_prefixes = tuple(
            prefix
            for source, prefix in (
                (diffusion, FLUX_DIFFUSION_PREFIX),
                (clip_l, _TEXT_ENCODER_ROOTS[0]),
                (t5xxl, _TEXT_ENCODER_ROOTS[1]),
                (qwen3_2b, "text_encoders.qwen3_2b."),
                (vae, FLUX_VAE_PREFIX),
            )
            if source is None
        )
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))

    return FluxAssemblyPlan(
        family=family,
        diffusion=diffusion_plan,
        clip_l=clip_plan,
        t5xxl=t5_plan,
        vae=vae_plan,
        unclaimed=unclaimed,
        qwen3_2b=qwen_plan,
    )


def plan_wan21_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    umt5xxl: WeightSource | None = None,
    clip_vision: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> Wan21AssemblyPlan:
    """Plan an official Wan 2.1 T2V or I2V split or combined layout."""

    return _plan_wan_assembly(
        family=WAN21,
        checkpoint=checkpoint,
        diffusion=diffusion,
        umt5xxl=umt5xxl,
        clip_vision=clip_vision,
        vae=vae,
    )


def plan_wan22_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    umt5xxl: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> Wan21AssemblyPlan:
    """Plan an official Wan 2.2 TI2V 5B split or combined layout."""

    return _plan_wan_assembly(
        family=WAN22,
        checkpoint=checkpoint,
        diffusion=diffusion,
        umt5xxl=umt5xxl,
        clip_vision=None,
        vae=vae,
    )


def _plan_wan_diffusion(
    extracted: _Extracted,
    config: Wan21Config,
) -> ComponentPlan[Wan21Config]:
    plan = _plan("diffusion", extracted, config)
    if config.model_variant != "wandancer":
        return plan
    keys = dict(plan.keys)
    dtypes = dict(plan.dtypes)
    transforms: dict[str, TensorTransform] = {}
    for index in range(2):
        root = f"music_encoder.{index}.self_attn"
        for suffix in ("weight", "bias"):
            source_key = f"{root}.in_proj_{suffix}"
            full_source_key = keys.pop(source_key)
            dtype = dtypes.pop(source_key)
            for part, projection in enumerate(("q_proj", "k_proj", "v_proj")):
                model_key = f"{root}.{projection}.{suffix}"
                keys[model_key] = full_source_key
                dtypes[model_key] = dtype
                transforms[model_key] = RowChunk(part=part, parts=3)
    return replace(
        plan,
        keys=keys,
        dtypes=dtypes,
        transforms=transforms,
    )


def plan_wan21_standalone_component(
    source: WeightSource,
    role: Wan21StandaloneComponentRole,
) -> Wan21StandaloneComponentPlan:
    """Plan one split component used by the official Wan alpha workflow."""

    if role == "diffusion":
        evidence = detect_wan21(source)
        profile = None if evidence is None else evidence.fields.get("profile")
        configs = {
            "t2v-14b": WAN21_T2V_14B,
            "causal-ar-1.3b": WAN21_CAUSAL_AR_1_3B,
            "humo-17b": WAN21_HUMO_17B,
            "s2v-14b-2.2": WAN22_S2V_14B,
            "wandancer-14b-2.2": WAN22_WANDANCER_14B,
        }
        config = configs.get(cast("str", profile))
        if config is None:
            raise AssemblyError(
                "diffusion: standalone Wan requires T2V 14B, CausalAR 1.3B, HuMo 17B,"
                " S2V 14B, or WanDancer 14B"
            )
        extracted = _component_source(
            "diffusion",
            source,
            None,
            WAN21_DIFFUSION_PREFIX,
            split_prefixes=("", WAN21_DIFFUSION_PREFIX),
        )
        if extracted.ignored:
            raise AssemblyError(
                "diffusion: source contains unsupported or duplicate tensors: "
                + ", ".join(extracted.ignored)
            )
        layout = wan21_layout(config)
        keys = set(extracted.geometries)
        if keys != set(layout):
            missing = sorted(set(layout) - keys)
            unexpected = sorted(keys - set(layout))
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise AssemblyError(
                f"diffusion: source does not match the Wan 2.1 {profile} layout: "
                + "; ".join(details)
            )
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        for key, shape in layout.items():
            geometry = extracted.geometries[key]
            if geometry.shape != shape:
                raise AssemblyError(
                    f"diffusion: {key} expected shape {shape}, found {geometry.shape}"
                )
            if geometry.dtype.kind != "float" and key not in quantized_weights:
                raise AssemblyError(
                    f"diffusion: {key} requires floating-point storage, found {geometry.dtype.name}"
                )
        return Wan21StandaloneComponentPlan(
            role,
            cast(
                "ComponentPlan[Wan21Config | T5Config | Wan21VAEConfig]",
                _plan_wan_diffusion(extracted, config),
            ),
        )

    if role == "umt5xxl":
        tokenizer_candidates = tuple(
            key
            for key in ("spiece_model", WAN21_UMT5_ROOT + "spiece_model")
            if key in source.keys()
        )
        if len(tokenizer_candidates) != 1:
            raise AssemblyError(
                "umt5xxl: source must contain exactly one uint8 spiece_model tensor"
            )
        tokenizer_source_key = tokenizer_candidates[0]
        tokenizer_geometry = source.entry(tokenizer_source_key).geometry
        if (
            tokenizer_geometry.dtype != UINT8
            or len(tokenizer_geometry.shape) != 1
            or not 1 <= tokenizer_geometry.shape[0] <= 8 * 1024 * 1024
        ):
            raise AssemblyError("umt5xxl: spiece_model must be a nonempty rank-1 uint8 tensor")
        extracted = _component_source(
            "umt5xxl",
            source,
            None,
            WAN21_UMT5_PREFIX,
            root=WAN21_UMT5_ROOT,
            split_prefixes=("", WAN21_UMT5_PREFIX),
        )
        tokenizer_stripped = tokenizer_source_key.removeprefix(WAN21_UMT5_PREFIX)
        text = _Extracted(
            extracted.path,
            {
                key: value
                for key, value in extracted.geometries.items()
                if key != tokenizer_stripped
            },
            {
                key: value
                for key, value in extracted.source_keys.items()
                if key != tokenizer_stripped
            },
            extracted.quant,
            tuple(key for key in extracted.ignored if key != tokenizer_source_key),
            extracted.source_format,
            extracted.runtime_facts,
            extracted.payload_source,
        )
        if text.ignored:
            raise AssemblyError(
                "umt5xxl: source contains unsupported or duplicate tensors: "
                + ", ".join(text.ignored)
            )
        try:
            config = detect_t5_config(text.geometries)
        except ValueError as error:
            raise AssemblyError(f"umt5xxl: {error}") from error
        if config != UMT5_XXL_CONFIG:
            raise AssemblyError("umt5xxl: standalone Wan requires the UMT5-XXL layout")
        return Wan21StandaloneComponentPlan(
            role,
            cast(
                "ComponentPlan[Wan21Config | T5Config | Wan21VAEConfig]",
                _plan("umt5xxl", text, config, drop=T5_TEXT_OPTIONAL_KEYS),
            ),
            tokenizer_source_key,
        )

    if role == "vae":
        extracted = _component_source(
            "vae",
            source,
            None,
            WAN21_VAE_PREFIX,
            split_prefixes=("", WAN21_VAE_PREFIX, "first_stage_model."),
        )
        if extracted.ignored:
            raise AssemblyError(
                "vae: source contains unsupported or duplicate tensors: "
                + ", ".join(extracted.ignored)
            )
        layout = wan21_vae_layout(WAN21_VAE_CONFIG)
        keys = set(extracted.geometries)
        if keys != set(layout):
            missing = sorted(set(layout) - keys)
            unexpected = sorted(keys - set(layout))
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise AssemblyError(
                "vae: source does not match the Wan 2.1 causal VAE: " + "; ".join(details)
            )
        for key, shape in layout.items():
            geometry = extracted.geometries[key]
            if geometry.shape != shape:
                raise AssemblyError(f"vae: {key} expected shape {shape}, found {geometry.shape}")
            if geometry.dtype.kind != "float":
                raise AssemblyError(
                    f"vae: {key} requires floating-point storage, found {geometry.dtype.name}"
                )
        return Wan21StandaloneComponentPlan(
            role,
            cast(
                "ComponentPlan[Wan21Config | T5Config | Wan21VAEConfig]",
                _plan("vae", extracted, WAN21_VAE_CONFIG),
            ),
        )

    raise ValueError(f"unsupported standalone Wan component role {role!r}")


def plan_wan_text_component(
    *,
    checkpoint: WeightSource | None = None,
    umt5xxl: WeightSource | None = None,
) -> tuple[ComponentPlan[T5Config], str, bool]:
    """Plan UMT5 weights with the exact embedded or vendored tokenizer origin."""
    text_source = umt5xxl if umt5xxl is not None else checkpoint
    if text_source is None:
        raise AssemblyError("umt5xxl: no source")
    tokenizer_vendored = getattr(text_source, "source_format", "safetensors") == "gguf"
    tokenizer_source_key = ""
    if not tokenizer_vendored:
        tokenizer_candidates = tuple(
            key
            for key in ("spiece_model", WAN21_UMT5_ROOT + "spiece_model")
            if key in text_source.keys()
        )
        if len(tokenizer_candidates) != 1:
            raise AssemblyError(
                "umt5xxl: source must contain exactly one uint8 spiece_model tensor"
            )
        tokenizer_source_key = tokenizer_candidates[0]
        tokenizer_geometry = text_source.entry(tokenizer_source_key).geometry
        if (
            tokenizer_geometry.dtype != UINT8
            or len(tokenizer_geometry.shape) != 1
            or not 1 <= tokenizer_geometry.shape[0] <= 8 * 1024 * 1024
        ):
            raise AssemblyError("umt5xxl: spiece_model must be a nonempty rank-1 uint8 tensor")
    extracted = _component_source(
        "umt5xxl",
        umt5xxl,
        checkpoint,
        WAN21_UMT5_PREFIX,
        root=WAN21_UMT5_ROOT,
        split_prefixes=("", WAN21_UMT5_PREFIX),
    )
    if tokenizer_vendored:
        text_extracted = extracted
    else:
        tokenizer_stripped = tokenizer_source_key.removeprefix(WAN21_UMT5_PREFIX)
        text_extracted = _Extracted(
            extracted.path,
            {
                key: value
                for key, value in extracted.geometries.items()
                if key != tokenizer_stripped
            },
            {
                key: value
                for key, value in extracted.source_keys.items()
                if key != tokenizer_stripped
            },
            extracted.quant,
            tuple(key for key in extracted.ignored if key != tokenizer_source_key),
            extracted.source_format,
            extracted.runtime_facts,
            extracted.payload_source,
        )
    if text_extracted.ignored:
        raise AssemblyError(
            "umt5xxl: source contains unsupported or duplicate tensors: "
            + ", ".join(text_extracted.ignored)
        )
    try:
        text_config = detect_t5_config(text_extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"umt5xxl: {error}") from error
    if text_config != UMT5_XXL_CONFIG:
        raise AssemblyError("umt5xxl: Wan requires the UMT5-XXL layout")
    text_plan = _plan("umt5xxl", text_extracted, text_config, drop=T5_TEXT_OPTIONAL_KEYS)
    return text_plan, tokenizer_source_key, tokenizer_vendored


def plan_wan_diffusion_component(
    *,
    family: ModelFamily,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
) -> ComponentPlan[Wan21Config]:
    """Resolve and validate a Wan diffusion profile independently of its companions."""
    family_name = "Wan 2.1" if family is WAN21 else "Wan 2.2"
    detect_source = diffusion if diffusion is not None else checkpoint
    if detect_source is None:
        raise AssemblyError("diffusion: no source")
    evidence = detect_wan21(detect_source) if family is WAN21 else detect_wan22(detect_source)
    if evidence is None:
        raise AssemblyError(f"diffusion: source is not an official {family_name} profile")
    profile = evidence.fields.get("profile")
    configs: dict[str, Wan21Config]
    if family is WAN21:
        configs = {
            "t2v-1.3b": WAN21_T2V_1_3B,
            "causal-ar-1.3b": WAN21_CAUSAL_AR_1_3B,
            "flow-rvs-1.3b": WAN21_FLOW_RVS_1_3B,
            "t2v-14b": WAN21_T2V_14B,
            "humo-17b": WAN21_HUMO_17B,
            "i2v-14b": WAN21_I2V_14B,
            "scail-14b": WAN21_SCAIL_14B,
            "scail2-14b": WAN21_SCAIL2_14B,
            "animate2-14b-2.1": WAN21_ANIMATE2_14B,
            "animate-14b-2.2": WAN22_ANIMATE_14B,
            "bernini-14b-2.2": WAN22_BERNINI_14B,
            "s2v-14b-2.2": WAN22_S2V_14B,
            "wandancer-14b-2.2": WAN22_WANDANCER_14B,
            "flf-i2v-14b": WAN21_FLF_I2V_14B,
            "fun-control-1.3b": WAN21_FUN_CONTROL_1_3B,
            "fun-inpaint-1.3b": WAN21_FUN_INPAINT_1_3B,
            "camera-1.3b": WAN21_CAMERA_1_3B,
            "camera-14b": WAN21_CAMERA_14B,
            "i2v-14b-2.2": WAN22_I2V_14B,
            "camera-14b-2.2": WAN22_CAMERA_14B,
            "fun-control-14b-2.2": WAN22_FUN_CONTROL_14B,
            "vace-1.3b": WAN21_VACE_1_3B,
            "vace-14b": WAN21_VACE_14B,
        }
    else:
        configs = {
            "ti2v-5b": WAN22_TI2V_5B,
            "fun-control-5b": WAN22_FUN_CONTROL_5B,
            "fun-inpaint-5b": WAN22_FUN_INPAINT_5B,
        }
    try:
        diffusion_config = configs[cast("str", profile)]
    except (KeyError, TypeError):
        raise AssemblyError(
            f"diffusion: detector returned an unsupported {family_name} profile"
        ) from None
    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        WAN21_DIFFUSION_PREFIX,
        split_prefixes=("", WAN21_DIFFUSION_PREFIX),
    )
    if extracted.ignored:
        raise AssemblyError(
            "diffusion: source contains unsupported or duplicate tensors: "
            + ", ".join(extracted.ignored)
        )
    diffusion_layout = wan21_layout(diffusion_config)
    diffusion_keys = set(extracted.geometries)
    if diffusion_keys != set(diffusion_layout):
        missing = sorted(set(diffusion_layout) - diffusion_keys)
        unexpected = sorted(diffusion_keys - set(diffusion_layout))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AssemblyError(
            f"diffusion: source does not match the {family_name} {profile} layout: "
            + "; ".join(details)
        )
    quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
    for key, shape in diffusion_layout.items():
        geometry = extracted.geometries.get(key)
        if geometry is None or geometry.shape != shape:
            found = None if geometry is None else geometry.shape
            raise AssemblyError(f"diffusion: {key} expected shape {shape}, found {found}")
        if geometry.dtype.kind != "float" and key not in quantized_weights:
            raise AssemblyError(
                f"diffusion: {key} requires floating-point storage, found {geometry.dtype.name}"
            )
    return _plan_wan_diffusion(extracted, diffusion_config)


def plan_wan_vae_component(
    *,
    config: Wan21VAEConfig | Wan22VAEConfig,
    checkpoint: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> ComponentPlan[Wan21VAEConfig | Wan22VAEConfig]:
    """Validate a causal codec's complete geometry before constructing its key map."""
    extracted = _component_source(
        "vae",
        vae,
        checkpoint,
        WAN21_VAE_PREFIX,
        split_prefixes=("", WAN21_VAE_PREFIX, "first_stage_model."),
    )
    if extracted.ignored:
        raise AssemblyError(
            "vae: source contains unsupported or duplicate tensors: " + ", ".join(extracted.ignored)
        )
    if isinstance(config, Wan22VAEConfig):
        try:
            config = validate_wan22_vae_header(extracted.geometries)
        except Wan22VAEHeaderError as error:
            raise AssemblyError(f"vae: {error}") from error
    else:
        layout = wan21_vae_layout(config)
        keys = set(extracted.geometries)
        if keys != set(layout):
            missing = sorted(set(layout) - keys)
            unexpected = sorted(keys - set(layout))
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise AssemblyError(
                "vae: source does not match the required causal VAE: " + "; ".join(details)
            )
        for key, shape in layout.items():
            geometry = extracted.geometries.get(key)
            if geometry is None or geometry.shape != shape:
                found = None if geometry is None else geometry.shape
                raise AssemblyError(f"vae: {key} expected shape {shape}, found {found}")
            if geometry.dtype.kind != "float":
                raise AssemblyError(
                    f"vae: {key} requires floating-point storage, found {geometry.dtype.name}"
                )
    return _plan("vae", extracted, config)


def plan_wan_vision_component(source: WeightSource) -> ComponentPlan[ClipVisionConfig]:
    """Plan the image-conditioning tower independently of its diffusion companion."""
    extracted = _component_source("clip_vision", source, None, "")
    if extracted.ignored:
        raise AssemblyError(
            "clip_vision: source contains unsupported tensors: " + ", ".join(extracted.ignored)
        )
    try:
        config = detect_wan21_clip_vision(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"clip_vision: {error}") from error
    return _plan("clip_vision", extracted, config)


def _plan_wan_assembly(
    *,
    family: ModelFamily,
    checkpoint: WeightSource | None,
    diffusion: WeightSource | None,
    umt5xxl: WeightSource | None,
    clip_vision: WeightSource | None,
    vae: WeightSource | None,
) -> Wan21AssemblyPlan:
    family_name = "Wan 2.1" if family is WAN21 else "Wan 2.2"
    if all(source is None for source in (checkpoint, diffusion, umt5xxl, clip_vision, vae)):
        raise AssemblyError("no sources given")
    split_paths = tuple(
        _source_path(source, component)
        for component, source in (
            ("diffusion", diffusion),
            ("umt5xxl", umt5xxl),
            ("clip_vision", clip_vision),
            ("vae", vae),
        )
        if source is not None
    )
    if len(set(split_paths)) != len(split_paths):
        raise AssemblyError(f"{family_name} split component sources must be distinct")
    diffusion_plan = plan_wan_diffusion_component(
        family=family, checkpoint=checkpoint, diffusion=diffusion
    )
    diffusion_config = diffusion_plan.config

    vision_plan: ComponentPlan[ClipVisionConfig] | None = None
    if clip_vision is not None:
        if family is not WAN21 or diffusion_config.model_type != "i2v":
            raise AssemblyError("clip_vision: Wan 2.1 T2V does not consume CLIP vision")
        vision_plan = plan_wan_vision_component(clip_vision)
    elif family is WAN21 and diffusion_config.model_type == "i2v":
        raise AssemblyError("clip_vision: Wan 2.1 I2V requires the official CLIP ViT-H checkpoint")

    text_plan, tokenizer_source_key, tokenizer_vendored = plan_wan_text_component(
        checkpoint=checkpoint, umt5xxl=umt5xxl
    )

    vae_config = (
        WAN22_VAE_CONFIG
        if family is WAN22
        else WAN21_FLOW_RVS_VAE_CONFIG
        if diffusion_config is WAN21_FLOW_RVS_1_3B
        else WAN21_VAE_CONFIG
    )
    vae_plan = plan_wan_vae_component(config=vae_config, checkpoint=checkpoint, vae=vae)

    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        claimed_prefixes = tuple(
            prefix
            for split, prefix in (
                (diffusion, WAN21_DIFFUSION_PREFIX),
                (umt5xxl, WAN21_UMT5_ROOT),
                (vae, WAN21_VAE_PREFIX),
            )
            if split is None
        )
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))
    if unclaimed:
        raise AssemblyError(
            f"checkpoint: unsupported or duplicate {family_name} tensors: " + ", ".join(unclaimed)
        )
    return Wan21AssemblyPlan(
        family=family,
        diffusion=diffusion_plan,
        umt5xxl=text_plan,
        vae=vae_plan,
        tokenizer_source_key=tokenizer_source_key,
        clip_vision=vision_plan,
        tokenizer_vendored=tokenizer_vendored,
        unclaimed=unclaimed,
    )


#: Per-channel latent aggregates the LTX checkpoints carry beyond the two
#: the runtime consumes; the reference drops them via strict=False loading.
LTXV_VAE_UNUSED_STATISTICS = frozenset(
    {
        "per_channel_statistics.channel",
        "per_channel_statistics.mean-of-stds",
        "per_channel_statistics.mean-of-stds_over_std-of-means",
    }
)

# The official diffusion VAE carries this unused decoder tensor; the reference
# decoder has no matching parameter and ignores it during strict=False loading.
LTXAV_V25_VAE_UNUSED_TENSORS = frozenset({"decoder.type_emb"})


def _ltxv_component_layout(
    component: str,
    extracted: _Extracted,
    layout: dict[str, tuple[int, ...]],
    *,
    profile: str,
    allowed_extra: frozenset[str] = frozenset(),
    family_name: str = "LTX-Video",
) -> None:
    """Refuse anything but an exact layout match (allowing named extras),
    exact shapes, and floating-point storage for unquantized tensors."""
    keys = set(extracted.geometries)
    if keys - set(layout) - allowed_extra or set(layout) - keys:
        missing = sorted(set(layout) - keys)
        unexpected = sorted(keys - set(layout) - allowed_extra)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AssemblyError(
            f"{component}: source does not match the {family_name} {profile} layout: "
            + "; ".join(details)
        )
    quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
    for key, shape in layout.items():
        geometry = extracted.geometries[key]
        if geometry.shape != shape:
            raise AssemblyError(
                f"{component}: {key} expected shape {shape}, found {geometry.shape}"
            )
        if geometry.dtype.kind != "float" and key not in quantized_weights:
            raise AssemblyError(
                f"{component}: {key} requires floating-point storage, found {geometry.dtype.name}"
            )


def _plan_ltxv_diffusion_component(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[LTXVConfig]:
    detect_source = source if source is not None else checkpoint
    if detect_source is None:
        raise AssemblyError("diffusion: no source")
    evidence = detect_ltxv(detect_source)
    if evidence is None:
        raise AssemblyError("diffusion: source is not an official LTX-Video 2B profile")
    profile = evidence.fields.get("profile")
    configs = {"2b-v0.9": LTXV_2B_V09_CONFIG, "2b-v0.9.5": LTXV_2B_V095_CONFIG}
    try:
        config = configs[cast("str", profile)]
    except (KeyError, TypeError):
        raise AssemblyError(
            "diffusion: detector returned an unsupported LTX-Video profile"
        ) from None
    extracted = _component_source(
        "diffusion",
        source,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    if extracted.ignored:
        raise AssemblyError(
            "diffusion: source contains unsupported or duplicate tensors: "
            + ", ".join(extracted.ignored)
        )
    _ltxv_component_layout(
        "diffusion",
        extracted,
        ltxv_layout(config),
        profile=cast("str", profile),
    )
    return _plan("diffusion", extracted, config)


def _plan_ltxv_text_component(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[T5Config]:
    extracted = _component_source(
        "t5xxl",
        source,
        checkpoint,
        FLUX_T5XXL_PREFIX,
        root=_TEXT_ENCODER_ROOTS[1],
    )
    if extracted.ignored:
        raise AssemblyError(
            "t5xxl: source contains unsupported or duplicate tensors: "
            + ", ".join(extracted.ignored)
        )
    try:
        config = detect_t5_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"t5xxl: {error}") from error
    if config != T5_XXL_CONFIG:
        raise AssemblyError("t5xxl: LTX-Video requires the classic T5-XXL layout")
    quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
    for key in sorted(extracted.geometries):
        geometry = extracted.geometries[key]
        if geometry.dtype.kind != "float" and key not in quantized_weights:
            raise AssemblyError(
                f"t5xxl: {key} requires floating-point storage, found {geometry.dtype.name}"
            )
    return _plan("t5xxl", extracted, config, drop=T5_TEXT_OPTIONAL_KEYS)


def _plan_ltxv_vae_component(
    source: WeightSource | None,
    checkpoint: WeightSource | None,
) -> ComponentPlan[LTXVideoVAEConfig]:
    extracted = _component_source(
        "vae",
        source,
        checkpoint,
        FLUX_VAE_PREFIX,
        split_prefixes=("", FLUX_VAE_PREFIX),
    )
    if extracted.ignored:
        raise AssemblyError(
            "vae: source contains unsupported or duplicate tensors: " + ", ".join(extracted.ignored)
        )
    vae_keys = set(extracted.geometries) - LTXV_VAE_UNUSED_STATISTICS
    config: LTXVideoVAEConfig | None = None
    for candidate in (LTXV_2B_V09_VAE_CONFIG, LTXV_2B_V095_VAE_CONFIG):
        if vae_keys == set(ltxv_vae_layout(candidate)):
            config = candidate
            break
    if config is None:
        raise AssemblyError("vae: source does not match either LTX-Video 2B causal VAE layout")
    _ltxv_component_layout(
        "vae",
        extracted,
        ltxv_vae_layout(config),
        profile="causal VAE",
        allowed_extra=LTXV_VAE_UNUSED_STATISTICS,
    )
    for key in sorted(LTXV_VAE_UNUSED_STATISTICS & set(extracted.geometries)):
        geometry = extracted.geometries[key]
        expected = (config.latent_channels,)
        if geometry.shape != expected or geometry.dtype.kind != "float":
            raise AssemblyError(
                f"vae: {key} expected floating-point shape {expected},"
                f" found {geometry.dtype.name} {geometry.shape}"
            )
    return _plan("vae", extracted, config, drop=LTXV_VAE_UNUSED_STATISTICS)


def plan_ltxv_standalone_component(
    source: WeightSource,
    role: LTXVStandaloneComponentRole,
) -> LTXVStandaloneComponentPlan:
    """Plan one strict classic LTX-Video split component."""
    planners = {
        "diffusion": _plan_ltxv_diffusion_component,
        "t5xxl": _plan_ltxv_text_component,
        "vae": _plan_ltxv_vae_component,
    }
    try:
        planner = planners[role]
    except KeyError:
        raise ValueError(f"unsupported standalone LTX-Video component role {role!r}") from None
    component = cast(
        "ComponentPlan[LTXVConfig | T5Config | LTXVideoVAEConfig]",
        planner(source, None),
    )
    return LTXVStandaloneComponentPlan(role, component)


def _ltxav_number(value: object) -> float | None:
    """A JSON scalar as a float, refusing bools (JSON true/false are not
    numbers even though Python bool subclasses int)."""
    if type(value) is int or type(value) is float:
        return float(value)
    return None


def _ltxav_numeric_equal(value: object, expected: object) -> bool:
    """Numeric equality between a JSON value and a pinned constant,
    elementwise through nested sequences (the header stores integers
    where the port pins floats and vice versa)."""
    if isinstance(expected, tuple):
        pinned_items = cast("tuple[object, ...]", expected)
        if type(value) is not list:
            return False
        items = cast("list[object]", value)
        if len(items) != len(pinned_items):
            return False
        return all(
            _ltxav_numeric_equal(item, pinned)
            for item, pinned in zip(items, pinned_items, strict=True)
        )
    number = _ltxav_number(value)
    return number is not None and number == float(cast("int | float", expected))


def _ltxav_metadata_config(source: WeightSource, component: str) -> dict[str, object]:
    """The checkpoint's embedded config JSON; the loader refuses to fall
    back to constructor defaults, so missing or malformed metadata refuses."""
    encoded = source.metadata().get("config")
    if encoded is None:
        raise AssemblyError(
            f"{component}: the LTX-2 checkpoint metadata carries no config JSON;"
            " the loader refuses to assume the pinned constants"
        )
    try:
        decoded = cast("object", json.loads(encoded))
    except json.JSONDecodeError as error:
        raise AssemblyError(f"{component}: malformed LTX-2 config metadata: {error}") from error
    if type(decoded) is not dict:
        raise AssemblyError(f"{component}: LTX-2 config metadata must be a JSON object")
    return cast("dict[str, object]", decoded)


def _ltxav_metadata_section(
    config: dict[str, object], component: str, *path: str
) -> dict[str, object]:
    node = config
    for name in path:
        if name not in node:
            raise AssemblyError(
                f"{component}: LTX-2 config metadata lacks the {'.'.join(path)} section"
            )
        child = node[name]
        if type(child) is not dict:
            raise AssemblyError(
                f"{component}: LTX-2 config metadata {'.'.join(path)} must be a JSON object"
            )
        node = cast("dict[str, object]", child)
    return node


def _ltxav_transformer_config(source: WeightSource, profile: object) -> LTXAVConfig:
    """Select one exact transformer profile and gate its metadata constants."""
    transformer = _ltxav_metadata_section(
        _ltxav_metadata_config(source, "diffusion"), "diffusion", "transformer"
    )
    if profile == "19b":
        pinned_config = LTXAV_19B_CONFIG
    elif profile == "22b":
        pinned_config = LTXAV_22B_V23_CONFIG
    elif profile == "22b-v2.5":
        pinned_config = LTXAV_22B_V25_CONFIG
    else:
        raise AssemblyError("diffusion: detector returned an unsupported LTX-2 audio-video profile")
    for name, expected_text in (("rope_type", "split"), ("frequencies_precision", "float64")):
        value = transformer.get(name)
        if value != expected_text:
            raise AssemblyError(
                f"diffusion: LTX-2 metadata transformer {name} must be"
                f" {expected_text!r} (the Dinkster port pins it), found {value!r}"
            )
    numeric_pins: tuple[tuple[str, float | tuple[float, ...] | tuple[int, ...]], ...] = (
        ("positional_embedding_theta", LTXV_THETA),
        ("timestep_scale_multiplier", LTXV_TIMESTEP_MULTIPLIER),
        ("positional_embedding_max_pos", LTXV_MAX_POS),
        ("audio_positional_embedding_max_pos", LTXAV_AUDIO_MAX_POS),
        # The head split leaves every weight shape unchanged (q/k norms are
        # over inner_dim, before the split), so the layout gate cannot catch
        # a checkpoint that varies it - only this metadata pin can.
        ("attention_head_dim", pinned_config.attention_head_dim),
        ("num_attention_heads", pinned_config.num_attention_heads),
        ("audio_attention_head_dim", pinned_config.audio_attention_head_dim),
        ("audio_num_attention_heads", pinned_config.audio_num_attention_heads),
    )
    for name, pinned in numeric_pins:
        if not _ltxav_numeric_equal(transformer.get(name), pinned):
            raise AssemblyError(
                f"diffusion: LTX-2 metadata transformer {name} must equal"
                f" {pinned!r} (the Dinkster port pins it), found {transformer.get(name)!r}"
            )
    for name in ("causal_temporal_positioning", "use_middle_indices_grid"):
        if transformer.get(name) is not True:
            raise AssemblyError(
                f"diffusion: LTX-2 metadata transformer {name} must be true"
                f" (the Dinkster port pins it), found {transformer.get(name)!r}"
            )
    if profile in ("22b", "22b-v2.5"):
        for name in (
            "av_cross_ada_norm",
            "use_embeddings_connector",
            "connector_norm_output",
            "apply_gated_attention",
            "connector_apply_gated_attention",
            "caption_proj_before_connector",
            "cross_attention_adaln",
        ):
            if transformer.get(name) is not True:
                raise AssemblyError(
                    f"diffusion: LTX-2 22B metadata transformer {name} must be true"
                    f" (the Dinkster port pins it), found {transformer.get(name)!r}"
                )
        for name in (
            "caption_projection_first_linear",
            "caption_projection_second_linear",
            "caption_proj_input_norm",
        ):
            if transformer.get(name) is not False:
                raise AssemblyError(
                    f"diffusion: LTX-2 22B metadata transformer {name} must be false"
                    f" (the Dinkster port pins it), found {transformer.get(name)!r}"
                )
        norm_type = transformer.get("text_encoder_norm_type")
        expected_norm_types = (
            ("PER_TOKEN_RMS", "None") if profile == "22b-v2.5" else ("per_token_rms",)
        )
        if norm_type not in expected_norm_types:
            raise AssemblyError(
                "diffusion: LTX-2 22B metadata transformer text_encoder_norm_type"
                f" must be one of {expected_norm_types!r}, found {norm_type!r}"
            )
        video_connector = pinned_config.video_connector
        audio_connector = pinned_config.audio_connector
        assert video_connector is not None and audio_connector is not None
        connector_pins = (
            ("connector_attention_head_dim", video_connector.attention_head_dim),
            ("connector_num_attention_heads", video_connector.num_attention_heads),
            ("connector_num_layers", video_connector.num_layers),
            ("connector_num_learnable_registers", video_connector.num_learnable_registers),
            (
                "connector_positional_embedding_max_pos",
                (video_connector.positional_embedding_max_pos,),
            ),
            ("audio_connector_attention_head_dim", audio_connector.attention_head_dim),
            ("audio_connector_num_attention_heads", audio_connector.num_attention_heads),
        )
        for name, pinned in connector_pins:
            if not _ltxav_numeric_equal(transformer.get(name), pinned):
                raise AssemblyError(
                    f"diffusion: LTX-2 22B metadata transformer {name} must equal"
                    f" {pinned!r}, found {transformer.get(name)!r}"
                )
    multiplier = _ltxav_number(transformer.get("av_ca_timestep_scale_multiplier"))
    if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0:
        raise AssemblyError(
            "diffusion: LTX-2 metadata transformer av_ca_timestep_scale_multiplier"
            " must be a positive real number; the loader never assumes a default,"
            f" found {transformer.get('av_ca_timestep_scale_multiplier')!r}"
        )
    keyframe_embedding = any(key.endswith("keyframes_abs_pos_embedding") for key in source.keys())
    return replace(
        pinned_config,
        av_ca_timestep_scale_multiplier=multiplier,
        use_keyframes_abs_pos_embedding=keyframe_embedding,
    )


def _ltxav_vae_metadata_gate(
    checkpoint: WeightSource,
    config: LTXVideoVAEConfig | LTXDiffusionVideoVAEConfig = LTXAV_19B_VAE_CONFIG,
) -> None:
    """Refuse video-VAE metadata that contradicts the pinned config: the
    LTX-2 profile loads with timestep conditioning off (the reference builds
    the VAE from this metadata key), so a checkpoint declaring it on would
    otherwise be silently mis-modeled."""
    section = _ltxav_metadata_section(_ltxav_metadata_config(checkpoint, "vae"), "vae", "vae")
    if config is LTXAV_22B_V25_VAE_CONFIG:
        diffusion_config = cast("LTXDiffusionVideoVAEConfig", config)
        if section.get("_class_name") != "CausalDiffusionVAE":
            raise AssemblyError("vae: LTX-2.5 metadata must name CausalDiffusionVAE")
        if (
            section.get("model_output_type") != "x0"
            or section.get("spatial_padding_mode") != "zeros"
        ):
            raise AssemblyError("vae: LTX-2.5 metadata must use x0 output and zero spatial padding")
        encoder = _ltxav_metadata_section(
            _ltxav_metadata_config(checkpoint, "vae"), "vae", "vae", "encoder"
        )
        decoder = _ltxav_metadata_section(
            _ltxav_metadata_config(checkpoint, "vae"), "vae", "vae", "decoder"
        )
        expected_encoder: tuple[tuple[str, object], ...] = (
            ("_class_name", "Encoder"),
            ("dims", 3),
            ("in_channels", 3),
            ("out_channels", diffusion_config.latent_channels),
            ("patch_size", diffusion_config.patch_size),
            ("latent_log_var", "constant"),
            ("norm_layer", "pixel_norm"),
            ("base_channels", diffusion_config.base_channels),
            ("spatial_padding_mode", "zeros"),
        )
        expected_decoder: tuple[tuple[str, object], ...] = (
            ("_class_name", "NADiffusionDecoder"),
            ("in_channels", diffusion_config.latent_channels),
            ("out_channels", diffusion_config.output_channels),
            ("patch_size", diffusion_config.patch_size),
            ("head_dim", diffusion_config.head_dim),
            ("stage_channels", list(diffusion_config.stage_channels)),
            ("stage_depths", list(diffusion_config.stage_depths)),
            ("stage_kernels", [list(value) for value in diffusion_config.stage_kernels]),
            (
                "upsamples",
                [[list(stride), reduction] for stride, reduction in diffusion_config.upsamples],
            ),
            ("timestep_scale_multiplier", 1000.0),
            ("default_num_inference_steps", 1),
        )
        for name, expected in expected_encoder:
            if encoder.get(name) != expected:
                raise AssemblyError(
                    f"vae: LTX-2.5 metadata encoder {name} must equal {expected!r},"
                    f" found {encoder.get(name)!r}"
                )
        for name, expected in expected_decoder:
            if decoder.get(name) != expected:
                raise AssemblyError(
                    f"vae: LTX-2.5 metadata decoder {name} must equal {expected!r},"
                    f" found {decoder.get(name)!r}"
                )
        return

    value = section.get("timestep_conditioning")
    if value is not False:
        raise AssemblyError(
            "vae: LTX-2 metadata vae timestep_conditioning must be false"
            f" (the Dinkster port pins it), found {value!r}"
        )
    if config is LTXAV_22B_V23_VAE_CONFIG and section.get("spatial_padding_mode") != "zeros":
        raise AssemblyError(
            "vae: LTX-2.3 metadata vae spatial_padding_mode must be 'zeros'"
            f" (the Dinkster port pins it), found {section.get('spatial_padding_mode')!r}"
        )


def _ltxav_audio_vae_metadata_gate(checkpoint: WeightSource) -> None:
    """Refuse audio-VAE metadata that contradicts the port's pinned
    semantics: time-causal padding on the height axis, pixel
    normalization, no attention blocks, doubled latent channels, and the
    mel patchifier constants the runtime derives durations from."""
    config = _ltxav_metadata_config(checkpoint, "audio_vae")
    ddconfig = _ltxav_metadata_section(
        config, "audio_vae", "audio_vae", "model", "params", "ddconfig"
    )
    for name, expected_text in (("causality_axis", "height"), ("norm_type", "pixel")):
        value = ddconfig.get(name)
        if value != expected_text:
            raise AssemblyError(
                f"audio_vae: LTX-2 metadata ddconfig {name} must be"
                f" {expected_text!r} (the Dinkster port pins it), found {value!r}"
            )
    attention = ddconfig.get("attn_resolutions")
    if type(attention) is not list or attention:
        raise AssemblyError(
            "audio_vae: LTX-2 metadata ddconfig attn_resolutions must be empty"
            f" (the Dinkster port has no attention blocks), found {attention!r}"
        )
    if ddconfig.get("double_z") is not True:
        raise AssemblyError(
            "audio_vae: LTX-2 metadata ddconfig double_z must be true"
            f" (the Dinkster port halves the encoder output), found {ddconfig.get('double_z')!r}"
        )
    pinned = LTXAV_19B_AUDIO_VAE_CONFIG
    preprocessing_pins = (
        ("audio", "sampling_rate", pinned.sampling_rate),
        ("stft", "hop_length", pinned.mel_hop_length),
        ("stft", "filter_length", pinned.n_fft),
        ("mel", "n_mel_channels", pinned.mel_bins),
    )
    for section_name, name, expected in preprocessing_pins:
        section = _ltxav_metadata_section(
            config, "audio_vae", "audio_vae", "preprocessing", section_name
        )
        if not _ltxav_numeric_equal(section.get(name), float(expected)):
            raise AssemblyError(
                f"audio_vae: LTX-2 metadata preprocessing {section_name}.{name} must"
                f" equal {expected!r} (the Dinkster port pins it), found {section.get(name)!r}"
            )


def _ltxav_vocoder_metadata_gate(
    checkpoint: WeightSource,
) -> LTXVocoderConfig | LTXVocoderBWEConfig:
    """Select one exact supported LTX-2 vocoder metadata profile."""
    section = _ltxav_metadata_section(
        _ltxav_metadata_config(checkpoint, "vocoder"), "vocoder", "vocoder"
    )
    nested = section.get("vocoder")
    if isinstance(nested, Mapping):
        vocoder = _ltxav_metadata_section(section, "vocoder", "vocoder")
        bwe = _ltxav_metadata_section(section, "vocoder", "bwe")
        pinned_bwe = LTXAV_BWE_VOCODER_CONFIG

        def check_values(
            values: Mapping[str, object],
            expected: tuple[tuple[str, object], ...],
            profile: str,
            *,
            optional: bool = False,
        ) -> None:
            for name, target in expected:
                if optional and name not in values:
                    continue
                value = values.get(name)
                if type(target) is bool:
                    matches = value is target
                elif isinstance(target, (int, float, tuple)):
                    matches = _ltxav_numeric_equal(value, cast("object", target))
                else:
                    matches = value == target
                if not matches:
                    raise AssemblyError(
                        f"vocoder: LTX-2 metadata {profile}.{name} must equal {target!r}, "
                        f"found {value!r}"
                    )

        base = pinned_bwe.vocoder
        check_values(
            vocoder,
            (
                ("resblock", base.resblock),
                ("stereo", base.stereo),
                ("activation", base.activation),
                ("upsample_initial_channel", base.upsample_initial_channel),
                ("upsample_rates", base.upsample_rates),
                ("upsample_kernel_sizes", base.upsample_kernel_sizes),
                ("resblock_kernel_sizes", base.resblock_kernel_sizes),
                ("resblock_dilation_sizes", base.resblock_dilation_sizes),
                ("use_bias_at_final", base.use_bias_at_final),
                ("use_tanh_at_final", base.use_tanh_at_final),
            ),
            "vocoder",
        )
        check_values(
            vocoder,
            (
                ("apply_final_activation", base.apply_final_activation),
                ("output_sample_rate", base.output_sample_rate),
            ),
            "vocoder",
            optional=True,
        )
        generator = pinned_bwe.bwe_generator
        check_values(
            bwe,
            (
                ("resblock", generator.resblock),
                ("stereo", generator.stereo),
                ("activation", generator.activation),
                ("upsample_initial_channel", generator.upsample_initial_channel),
                ("upsample_rates", generator.upsample_rates),
                ("upsample_kernel_sizes", generator.upsample_kernel_sizes),
                ("resblock_kernel_sizes", generator.resblock_kernel_sizes),
                ("resblock_dilation_sizes", generator.resblock_dilation_sizes),
                ("use_bias_at_final", generator.use_bias_at_final),
                ("use_tanh_at_final", generator.use_tanh_at_final),
                ("apply_final_activation", generator.apply_final_activation),
                ("input_sampling_rate", pinned_bwe.input_sampling_rate),
                ("output_sampling_rate", pinned_bwe.output_sampling_rate),
                ("hop_length", pinned_bwe.hop_length),
                ("n_fft", pinned_bwe.n_fft),
                ("win_size", pinned_bwe.n_fft),
                ("num_mels", pinned_bwe.num_mels),
            ),
            "bwe",
        )
        check_values(
            bwe,
            (("output_sample_rate", generator.output_sample_rate),),
            "bwe",
            optional=True,
        )
        return pinned_bwe

    vocoder = section
    pinned = LTXAV_19B_VOCODER_CONFIG
    if vocoder.get("resblock") != pinned.resblock:
        raise AssemblyError(
            f"vocoder: LTX-2 metadata vocoder resblock must be {pinned.resblock!r}"
            f" (the Dinkster port pins it), found {vocoder.get('resblock')!r}"
        )
    if vocoder.get("stereo") is not pinned.stereo:
        raise AssemblyError(
            f"vocoder: LTX-2 metadata vocoder stereo must be {pinned.stereo!r}"
            f" (the Dinkster port pins it), found {vocoder.get('stereo')!r}"
        )
    numeric_pins: tuple[tuple[str, float | tuple[Any, ...]], ...] = (
        ("upsample_initial_channel", float(pinned.upsample_initial_channel)),
        ("upsample_rates", pinned.upsample_rates),
        ("upsample_kernel_sizes", pinned.upsample_kernel_sizes),
        ("resblock_kernel_sizes", pinned.resblock_kernel_sizes),
        ("resblock_dilation_sizes", pinned.resblock_dilation_sizes),
    )
    for name, expected in numeric_pins:
        if not _ltxav_numeric_equal(vocoder.get(name), expected):
            raise AssemblyError(
                f"vocoder: LTX-2 metadata vocoder {name} must equal {expected!r}"
                f" (the Dinkster port pins it), found {vocoder.get(name)!r}"
            )
    optional_pins: tuple[tuple[str, object], ...] = (
        ("activation", pinned.activation),
        ("use_bias_at_final", pinned.use_bias_at_final),
        ("use_tanh_at_final", pinned.use_tanh_at_final),
        ("apply_final_activation", pinned.apply_final_activation),
        ("output_sample_rate", pinned.output_sample_rate),
    )
    for name, expected in optional_pins:
        if name not in vocoder:
            continue
        value = vocoder[name]
        matches = value is expected if type(expected) is bool else value == expected
        if not matches:
            raise AssemblyError(
                f"vocoder: LTX-2 metadata vocoder {name} must equal {expected!r}"
                f" when present, found {value!r}"
            )
    return pinned


def _plan_ltxav_audio_codec_components(
    source: WeightSource,
) -> tuple[
    ComponentPlan[LTXAudioVAEConfig],
    ComponentPlan[LTXVocoderConfig] | ComponentPlan[LTXVocoderBWEConfig],
]:
    _ltxav_audio_vae_metadata_gate(source)
    audio_extracted = _component_source("audio_vae", None, source, LTXAV_AUDIO_VAE_PREFIX)
    if audio_extracted.ignored:
        raise AssemblyError(
            "audio_vae: source contains unsupported or duplicate tensors: "
            + ", ".join(audio_extracted.ignored)
        )
    _ltxv_component_layout(
        "audio_vae",
        audio_extracted,
        ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG),
        profile="causal audio VAE",
        family_name="LTX-2",
    )
    audio_vae_plan = _plan("audio_vae", audio_extracted, LTXAV_19B_AUDIO_VAE_CONFIG)

    vocoder_config = _ltxav_vocoder_metadata_gate(source)
    vocoder_extracted = _component_source("vocoder", None, source, LTXAV_VOCODER_PREFIX)
    if vocoder_extracted.ignored:
        raise AssemblyError(
            "vocoder: source contains unsupported or duplicate tensors: "
            + ", ".join(vocoder_extracted.ignored)
        )
    if isinstance(vocoder_config, LTXVocoderBWEConfig):
        vocoder_layout = ltx_vocoder_bwe_layout(vocoder_config)
    else:
        vocoder_layout = ltx_vocoder_layout(vocoder_config)
    _ltxv_component_layout(
        "vocoder",
        vocoder_extracted,
        vocoder_layout,
        profile="vocoder",
        family_name="LTX-2",
    )
    vocoder_plan = cast(
        "ComponentPlan[LTXVocoderConfig] | ComponentPlan[LTXVocoderBWEConfig]",
        _plan("vocoder", vocoder_extracted, vocoder_config),
    )
    return audio_vae_plan, vocoder_plan


def plan_ltxav_audio_codec(source: WeightSource) -> LTXAVAudioCodecPlan:
    """Plan the audio codec from either its own asset or an LTX-2 checkpoint."""
    audio_vae, vocoder = _plan_ltxav_audio_codec_components(source)
    known_prefixes = (
        LTXAV_AUDIO_VAE_PREFIX,
        LTXAV_VOCODER_PREFIX,
        FLUX_DIFFUSION_PREFIX,
        FLUX_VAE_PREFIX,
        LTXAV_PROJECTION_PREFIX,
    )
    unclaimed = tuple(key for key in source.keys() if not key.startswith(known_prefixes))
    if unclaimed:
        raise AssemblyError("audio codec: unsupported LTX-2 tensors: " + ", ".join(unclaimed))
    return LTXAVAudioCodecPlan(LTXAV, audio_vae, vocoder, unclaimed)


def plan_ltxav_standalone_component(
    source: WeightSource,
    role: LTXAVStandaloneComponentRole,
) -> LTXAVStandaloneComponentPlan:
    """Plan one strict LTX-2 component from an independently selected asset."""
    if role == "latent_upscaler":
        config = _ltxav_metadata_config(source, role)
        expected = {
            "_class_name": "LatentUpsampler",
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "spatial_scale": 2.0,
            "rational_resampler": False,
        }
        if config != expected:
            raise AssemblyError(
                "latent_upscaler: LTX-2 config metadata does not match the official "
                "x2 spatial profile"
            )
        extracted = _component_source(
            role,
            source,
            None,
            "",
            split_prefixes=("",),
        )
        if extracted.ignored:
            raise AssemblyError(
                "latent_upscaler: source contains unsupported or duplicate tensors: "
                + ", ".join(extracted.ignored)
            )
        _ltxv_component_layout(
            role,
            extracted,
            ltx_latent_upsampler_layout(),
            profile="x2 spatial latent upscaler",
            family_name="LTX-2",
        )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(role, extracted, LTX_LATENT_UPSAMPLER_CONFIG),
        )

    if role == "diffusion":
        evidence = detect_ltxav(source)
        if evidence is None:
            raise AssemblyError("diffusion: source is not an official LTX-2 audio-video profile")
        profile = evidence.fields.get("profile")
        diffusion_config = _ltxav_transformer_config(source, profile)
        extracted = _component_source(
            "diffusion",
            source,
            None,
            FLUX_DIFFUSION_PREFIX,
            split_prefixes=("", FLUX_DIFFUSION_PREFIX),
        )
        duration_keys = {key for key in extracted.geometries if key.startswith("duration_head.")}
        if duration_keys:
            duration_layout = {
                "duration_head." + key: shape
                for key, shape in ltxav_duration_head_layout(LTXAV_DURATION_HEAD_CONFIG).items()
            }
            duration_extracted = replace(
                extracted,
                geometries={
                    key: value
                    for key, value in extracted.geometries.items()
                    if key in duration_keys
                },
                source_keys={
                    key: value
                    for key, value in extracted.source_keys.items()
                    if key in duration_keys
                },
                quant={
                    key: value
                    for key, value in extracted.quant.items()
                    if key.startswith("duration_head.")
                },
            )
            _ltxv_component_layout(
                "duration_head",
                duration_extracted,
                duration_layout,
                profile="prompt duration head",
                family_name="LTX-2",
            )
            extracted = replace(
                extracted,
                geometries={
                    key: value
                    for key, value in extracted.geometries.items()
                    if key not in duration_keys
                },
                source_keys={
                    key: value
                    for key, value in extracted.source_keys.items()
                    if key not in duration_keys
                },
                quant={
                    key: value
                    for key, value in extracted.quant.items()
                    if not key.startswith("duration_head.")
                },
            )
        if profile == "19b":
            connector_keys = {
                key for key in extracted.geometries if key.startswith(LTXAV_CONNECTOR_PREFIXES)
            }
            extracted = _Extracted(
                extracted.path,
                {
                    key: value
                    for key, value in extracted.geometries.items()
                    if key not in connector_keys
                },
                {
                    key: value
                    for key, value in extracted.source_keys.items()
                    if key not in connector_keys
                },
                {
                    layer: entry
                    for layer, entry in extracted.quant.items()
                    if not layer.startswith(LTXAV_CONNECTOR_PREFIXES)
                },
                (),
                extracted.source_format,
                extracted.runtime_facts,
                extracted.payload_source,
            )
        _ltxv_component_layout(
            "diffusion",
            extracted,
            ltxav_layout(diffusion_config),
            profile=cast("str", profile),
            family_name="LTX-2 audio-video",
        )
        component: ComponentPlan[Any] = _plan("diffusion", extracted, diffusion_config)
        return LTXAVStandaloneComponentPlan(role, component)

    if role == "duration_head":
        extracted = _component_source(
            role,
            source,
            None,
            FLUX_DIFFUSION_PREFIX,
            split_prefixes=("", FLUX_DIFFUSION_PREFIX),
        )
        prefix = "duration_head."
        keys = {key for key in extracted.geometries if key.startswith(prefix)}
        duration_extracted = (
            replace(
                extracted,
                geometries={
                    key[len(prefix) :]: value
                    for key, value in extracted.geometries.items()
                    if key in keys
                },
                source_keys={
                    key[len(prefix) :]: value
                    for key, value in extracted.source_keys.items()
                    if key in keys
                },
                quant={
                    key[len(prefix) :]: value
                    for key, value in extracted.quant.items()
                    if key.startswith(prefix)
                },
                ignored=(),
            )
            if keys
            else extracted
        )
        _ltxv_component_layout(
            role,
            duration_extracted,
            ltxav_duration_head_layout(LTXAV_DURATION_HEAD_CONFIG),
            profile="prompt duration head",
            family_name="LTX-2",
        )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(role, duration_extracted, LTXAV_DURATION_HEAD_CONFIG),
        )

    if role in ("gemma3_12b", "gemma4_12b"):
        tokenizer_key = "spiece_model" if role == "gemma3_12b" else "tokenizer_json"
        tokenizer_limit = 8 * 1024 * 1024 if role == "gemma3_12b" else 64 * 1024 * 1024
        if tokenizer_key not in source.keys():
            raise AssemblyError(
                f"{role}: source must contain exactly one uint8 {tokenizer_key} tensor"
            )
        tokenizer_geometry = source.entry(tokenizer_key).geometry
        if (
            tokenizer_geometry.dtype != UINT8
            or len(tokenizer_geometry.shape) != 1
            or not 1 <= tokenizer_geometry.shape[0] <= tokenizer_limit
        ):
            raise AssemblyError(f"{role}: {tokenizer_key} must be a nonempty rank-1 uint8 tensor")
        extracted = _component_source(
            role,
            source,
            None,
            LTXAV_GEMMA_PREFIX,
            split_prefixes=(LTXAV_GEMMA_PREFIX,),
        )
        extracted = replace(
            extracted,
            ignored=tuple(
                key
                for key in extracted.ignored
                if key != tokenizer_key
                and not key.startswith(LTXAV_PROJECTION_PREFIX)
                and not key.startswith(LTXAV_GEMMA_UNUSED_PREFIXES)
                and key not in LTXAV_GEMMA_UNUSED_ASSETS
            ),
        )
        if extracted.ignored:
            raise AssemblyError(
                f"{role}: source contains unsupported or duplicate tensors: "
                + ", ".join(extracted.ignored)
            )
        try:
            text_config = detect_gemma_text_config(extracted.geometries)
        except GemmaTextDetectError as error:
            raise AssemblyError(f"{role}: {error}") from error
        expected_config = GEMMA3_LTX_12B_CONFIG if role == "gemma3_12b" else GEMMA4_LTX_12B_CONFIG
        if text_config != expected_config:
            name = "Gemma 3" if role == "gemma3_12b" else "Gemma 4"
            raise AssemblyError(f"{role}: LTX-2 requires the {name} 12B text layout")
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        for key in sorted(extracted.geometries):
            geometry = extracted.geometries[key]
            if geometry.dtype.kind != "float" and key not in quantized_weights:
                raise AssemblyError(
                    f"{role}: {key} requires floating-point storage, found {geometry.dtype.name}"
                )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(role, extracted, text_config),
            tokenizer_source_key=tokenizer_key,
        )

    if role == "text_projection":
        extracted = _extract(
            source,
            _source_path(source, role),
            role,
            LTXAV_PROJECTION_PREFIX,
        )
        try:
            kind = detect_ltx_text_projection(extracted.geometries)
        except GemmaTextDetectError as error:
            raise AssemblyError(f"text_projection: {error}") from error
        gemma_source = source.metadata().get("gemma_source_checkpoint")
        if gemma_source is not None:
            try:
                decoded_gemma_source = json.loads(gemma_source)
            except (TypeError, ValueError) as error:
                raise AssemblyError(
                    "text_projection: gemma_source_checkpoint metadata is malformed"
                ) from error
            if decoded_gemma_source != {
                "ltx_version": "2.4.0",
                "gemma_version": "gemma4-12b-ltx-v1",
            }:
                raise AssemblyError("text_projection: unsupported gemma_source_checkpoint metadata")
            if kind != "dual_linear" or source.metadata().get("model_version") != "2.4.0":
                raise AssemblyError(
                    "text_projection: Gemma 4 requires the LTX-2.4 dual projection profile"
                )
            kind = "dual_linear_gemma4"
        elif kind == "dual_linear" and identify_ltxav_text_source(source) == "gemma4_12b":
            kind = "dual_linear_gemma4"
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        for key in sorted(extracted.geometries):
            geometry = extracted.geometries[key]
            if geometry.dtype.kind != "float" and key not in quantized_weights:
                raise AssemblyError(
                    f"text_projection: {key} requires floating-point storage, "
                    f"found {geometry.dtype.name}"
                )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(
                role,
                extracted,
                kind,
                renames={"aggregate_embed.weight": "weight"} if kind == "single_linear" else None,
            ),
        )

    if role == "connectors":
        extracted = _component_source(
            role,
            source,
            None,
            FLUX_DIFFUSION_PREFIX,
            split_prefixes=("", FLUX_DIFFUSION_PREFIX),
        )
        connector_keys = {
            key for key in extracted.geometries if key.startswith(LTXAV_CONNECTOR_PREFIXES)
        }
        connector_extracted = _Extracted(
            extracted.path,
            {key: value for key, value in extracted.geometries.items() if key in connector_keys},
            {key: value for key, value in extracted.source_keys.items() if key in connector_keys},
            {
                layer: entry
                for layer, entry in extracted.quant.items()
                if layer.startswith(LTXAV_CONNECTOR_PREFIXES)
            },
            (),
            extracted.source_format,
            extracted.runtime_facts,
            extracted.payload_source,
        )
        try:
            connector_config = detect_ltx_text_connectors(
                {
                    FLUX_DIFFUSION_PREFIX + key: value
                    for key, value in connector_extracted.geometries.items()
                }
            )
        except GemmaTextDetectError as error:
            raise AssemblyError(f"connectors: {error}") from error
        if connector_config is None:
            raise AssemblyError(
                "connectors: source must carry both LTX-2 text-embedding connector towers"
            )
        layout = {
            prefix + key: shape
            for prefix in LTXAV_CONNECTOR_PREFIXES
            for key, shape in ltx_connector_layout(connector_config).items()
        }
        _ltxv_component_layout(
            "connectors",
            connector_extracted,
            layout,
            profile="text-embedding connector",
            family_name="LTX-2",
        )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(role, connector_extracted, connector_config),
        )

    if role == "vae":
        extracted = _component_source(
            role,
            source,
            None,
            FLUX_VAE_PREFIX,
            split_prefixes=("", FLUX_VAE_PREFIX),
        )
        keys = set(extracted.geometries) - LTXV_VAE_UNUSED_STATISTICS
        v25_keys = keys - LTXAV_V25_VAE_UNUSED_TENSORS
        causal_config = next(
            (
                candidate
                for candidate in (LTXAV_19B_VAE_CONFIG, LTXAV_22B_V23_VAE_CONFIG)
                if keys == set(ltxv_vae_layout(candidate))
            ),
            None,
        )
        vae_config: LTXVideoVAEConfig | LTXDiffusionVideoVAEConfig | None = causal_config
        layout = ltxv_vae_layout(causal_config) if causal_config is not None else None
        if v25_keys == set(ltx_diffusion_video_vae_layout(LTXAV_22B_V25_VAE_CONFIG)):
            vae_config = LTXAV_22B_V25_VAE_CONFIG
            layout = ltx_diffusion_video_vae_layout(LTXAV_22B_V25_VAE_CONFIG)
        if vae_config is None:
            raise AssemblyError("vae: source does not match an official LTX-2 video VAE")
        assert layout is not None
        allowed_extra = LTXV_VAE_UNUSED_STATISTICS
        if vae_config is LTXAV_22B_V25_VAE_CONFIG:
            allowed_extra |= LTXAV_V25_VAE_UNUSED_TENSORS
            geometry = extracted.geometries.get("decoder.type_emb")
            if geometry is not None:
                if geometry.shape != (128,) or geometry.dtype != BFLOAT16:
                    raise AssemblyError(
                        "vae: decoder.type_emb expected bfloat16 shape (128,),"
                        f" found {geometry.dtype.name} {geometry.shape}"
                    )
        _ltxav_vae_metadata_gate(source, vae_config)
        _ltxv_component_layout(
            role,
            extracted,
            layout,
            profile=(
                "neighborhood-attention diffusion video VAE"
                if vae_config is LTXAV_22B_V25_VAE_CONFIG
                else "causal video VAE"
            ),
            allowed_extra=allowed_extra,
            family_name="LTX-2",
        )
        return LTXAVStandaloneComponentPlan(
            role,
            _plan(role, extracted, vae_config, drop=allowed_extra),
        )

    raise ValueError(f"unsupported standalone LTX-2 component role {role!r}")


def plan_z_image_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    qwen3_4b: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> ZImageAssemblyPlan:
    """Plan exact latent or pixel-space Z-Image compositions."""
    if all(source is None for source in (checkpoint, diffusion, qwen3_4b, vae)):
        raise AssemblyError("no sources given")

    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    layouts = (
        (Z_IMAGE_PIXEL_SPACE, Z_IMAGE_PIXEL_CONFIG, z_image_pixel_layout()),
        (Z_IMAGE, Z_IMAGE_CONFIG, z_image_layout()),
    )
    selected = next(
        (
            (family, config, layout)
            for family, config, layout in layouts
            if set(extracted.geometries) == set(layout)
            and all(extracted.geometries[key].shape == shape for key, shape in layout.items())
        ),
        None,
    )
    if selected is None:
        raise AssemblyError("diffusion: source is not an exact supported Z-Image layout")
    family, config, _ = selected
    diffusion_plan = _plan(
        "diffusion",
        extracted,
        config,
        drop=(
            frozenset({"__sequential__", "__x0__"})
            if config is Z_IMAGE_PIXEL_CONFIG
            else frozenset()
        ),
    )

    extracted = _component_source(
        "qwen3_4b",
        qwen3_4b,
        checkpoint,
        Z_IMAGE_QWEN_PREFIX,
        root="text_encoders.qwen3_4b.",
        split_prefixes=("model.", Z_IMAGE_QWEN_PREFIX),
    )
    if extracted.ignored:
        raise AssemblyError("qwen3_4b: unexpected non-text keys in the text encoder source")
    try:
        qwen_config = detect_qwen_text_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"qwen3_4b: {error}") from error
    if qwen_config.architecture != "z_image_qwen3_4b":
        raise AssemblyError("qwen3_4b: Z-Image requires the Qwen3-4B text profile")
    qwen_plan = _plan("qwen3_4b", extracted, qwen_config)

    vae_plan: ComponentPlan[KLConfig] | None = None
    if config is Z_IMAGE_CONFIG:
        extracted = _component_source("vae", vae, checkpoint, FLUX_VAE_PREFIX)
        try:
            kl_config = detect_kl_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"vae: {error}") from error
        vae_renames, vae_transforms = _kl_conversion(extracted.geometries)
        vae_plan = _plan(
            "vae",
            extracted,
            kl_config,
            renames=vae_renames,
            transforms=vae_transforms,
        )
    elif vae is not None:
        raise AssemblyError("vae: pixel-space Z-Image uses identity RGB and accepts no VAE source")

    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        component_prefixes = [
            (diffusion, FLUX_DIFFUSION_PREFIX),
            (qwen3_4b, "text_encoders.qwen3_4b."),
        ]
        if config is Z_IMAGE_CONFIG:
            component_prefixes.append((vae, FLUX_VAE_PREFIX))
        claimed_prefixes = tuple(prefix for source, prefix in component_prefixes if source is None)
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))
    return ZImageAssemblyPlan(family, diffusion_plan, qwen_plan, vae_plan, unclaimed)


QwenImageComponentRole = Literal["diffusion", "qwen2_5_vl_7b", "vae"]


def plan_qwen_image_component(
    source: WeightSource,
    role: QwenImageComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied Qwen Image component."""
    prefix = (
        FLUX_DIFFUSION_PREFIX
        if role == "diffusion"
        else QWEN_IMAGE_TEXT_PREFIX
        if role == "qwen2_5_vl_7b"
        else FLUX_VAE_PREFIX
    )
    extracted = _component_source(
        role,
        source,
        None,
        prefix,
        root="text_encoders.qwen2_5_vl_7b." if role == "qwen2_5_vl_7b" else None,
    )
    return _plan_qwen_image_extracted(extracted, role)


def _plan_qwen_image_extracted(
    extracted: _Extracted,
    role: QwenImageComponentRole,
) -> ComponentPlan[object]:
    from .qwen_image_layout import qwen_image_dit_layout

    if role == "diffusion":
        layouts = tuple(
            (config, qwen_image_dit_layout(config).keys)
            for config in (
                QWEN_IMAGE_LAYERED_CONFIG,
                QWEN_IMAGE_EDIT_2511_CONFIG,
                QWEN_IMAGE_CONFIG,
            )
        )
        config = next(
            (
                config
                for config, layout in layouts
                if set(extracted.geometries) == set(layout)
                and all(extracted.geometries[key].shape == shape for key, shape in layout.items())
            ),
            None,
        )
        if config is None:
            raise AssemblyError("diffusion: source is not an exact supported Qwen Image layout")
        if any(geometry.dtype.kind != "float" for geometry in extracted.geometries.values()):
            raise AssemblyError("diffusion: Qwen Image weights require floating-point storage")
        return cast("ComponentPlan[object]", _plan(role, extracted, config))
    if role == "qwen2_5_vl_7b":
        text_layout = qwen_image_text_layout()
        text_geometries = {
            key: geometry for key, geometry in extracted.geometries.items() if key in text_layout
        }
        try:
            config = detect_qwen_image_text_config(text_geometries)
        except ValueError as error:
            raise AssemblyError(f"qwen2_5_vl_7b: {error}") from error
        extras = set(extracted.geometries) - set(text_layout)
        if extras - {"lm_head.weight"}:
            raise AssemblyError(
                "qwen2_5_vl_7b: unexpected text keys: "
                + ", ".join(sorted(extras - {"lm_head.weight"})[:3])
            )
        return cast(
            "ComponentPlan[object]",
            _plan(role, extracted, config, drop=frozenset({"lm_head.weight"})),
        )
    return cast("ComponentPlan[object]", _plan_wan21_vae(extracted))


def _plan_wan21_vae(extracted: _Extracted) -> ComponentPlan[Wan21VAEConfig]:
    """Plan the exact Wan 2.1 VAE component (Qwen Image and Anima)."""
    vae_layout = wan21_vae_layout()
    if set(extracted.geometries) != set(vae_layout) or any(
        extracted.geometries[key].shape != shape for key, shape in vae_layout.items()
    ):
        raise AssemblyError("vae: source is not the exact Wan 2.1 VAE layout")
    if any(geometry.dtype.kind != "float" for geometry in extracted.geometries.values()):
        raise AssemblyError("vae: Wan 2.1 VAE weights require floating-point storage")
    return _plan("vae", extracted, WAN21_VAE_CONFIG)


def plan_qwen_image_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    qwen2_5_vl_7b: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> QwenImageAssemblyPlan:
    """Plan an exact Qwen Image diffusion, text encoder, and VAE composition."""
    if all(source is None for source in (checkpoint, diffusion, qwen2_5_vl_7b, vae)):
        raise AssemblyError("no sources given")

    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    diffusion_plan = _plan_qwen_image_extracted(extracted, "diffusion")

    extracted = _component_source(
        "qwen2_5_vl_7b",
        qwen2_5_vl_7b,
        checkpoint,
        QWEN_IMAGE_TEXT_PREFIX,
        root="text_encoders.qwen2_5_vl_7b.",
        split_prefixes=("", QWEN_IMAGE_TEXT_PREFIX),
    )
    text_plan = _plan_qwen_image_extracted(extracted, "qwen2_5_vl_7b")

    extracted = _component_source("vae", vae, checkpoint, FLUX_VAE_PREFIX)
    vae_plan = _plan_qwen_image_extracted(extracted, "vae")

    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        claimed_prefixes = tuple(
            prefix
            for source, prefix in (
                (diffusion, FLUX_DIFFUSION_PREFIX),
                (qwen2_5_vl_7b, QWEN_IMAGE_TEXT_PREFIX),
                (vae, FLUX_VAE_PREFIX),
            )
            if source is None
        )
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))
    return QwenImageAssemblyPlan(
        QWEN_IMAGE,
        cast("ComponentPlan[QwenImageConfig]", diffusion_plan),
        cast("ComponentPlan[QwenImageTextConfig]", text_plan),
        cast("ComponentPlan[Wan21VAEConfig]", vae_plan),
        unclaimed,
    )


SeedVR2ComponentRole = Literal["diffusion", "vae"]


def plan_seedvr2_component(
    source: WeightSource,
    role: SeedVR2ComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied SeedVR2 component."""
    prefix = FLUX_DIFFUSION_PREFIX if role == "diffusion" else FLUX_VAE_PREFIX
    extracted = _component_source(role, source, None, "", split_prefixes=("", prefix))
    if role == "diffusion":
        try:
            config = detect_seedvr2_config(extracted.geometries)
        except SeedVR2DetectError as error:
            raise AssemblyError(f"diffusion: {error}") from error
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        if any(
            geometry.dtype.kind != "float"
            for key, geometry in extracted.geometries.items()
            if key not in quantized_weights
        ):
            raise AssemblyError("diffusion: SeedVR2 weights require floating-point storage")
        return cast("ComponentPlan[object]", _plan(role, extracted, config))
    try:
        config = detect_seedvr2_vae_config(extracted.geometries)
    except SeedVR2VAEHeaderError as error:
        raise AssemblyError(f"vae: {error}") from error
    if any(geometry.dtype.kind != "float" for geometry in extracted.geometries.values()):
        raise AssemblyError("vae: SeedVR2 weights require floating-point storage")
    return cast("ComponentPlan[object]", _plan(role, extracted, config))


Krea2ComponentRole = Literal["diffusion", "qwen3vl_4b"]


def plan_krea2_component(
    source: WeightSource,
    role: Krea2ComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied Krea 2 component."""
    split_prefixes = ("", FLUX_DIFFUSION_PREFIX) if role == "diffusion" else ("", KREA2_TEXT_PREFIX)
    return _plan_krea2_extracted(
        _component_source(role, source, None, "", split_prefixes=split_prefixes), role
    )


def _plan_krea2_extracted(
    extracted: _Extracted,
    role: Krea2ComponentRole,
) -> ComponentPlan[object]:
    if role == "diffusion":
        try:
            config = detect_krea2_config(extracted.geometries)
        except Krea2DetectError as error:
            raise AssemblyError(f"diffusion: {error}") from error
        if any(geometry.dtype.kind != "float" for geometry in extracted.geometries.values()):
            raise AssemblyError("diffusion: Krea 2 weights require floating-point storage")
        return cast("ComponentPlan[object]", _plan(role, extracted, config))
    text_layout = krea2_text_layout()
    text_geometries = {
        key: geometry for key, geometry in extracted.geometries.items() if key in text_layout
    }
    try:
        config = detect_krea2_text_config(text_geometries)
    except Krea2TextDetectError as error:
        raise AssemblyError(f"qwen3vl_4b: {error}") from error
    extras = set(extracted.geometries) - set(text_layout)
    if extras - {"lm_head.weight"}:
        raise AssemblyError(
            "qwen3vl_4b: unexpected text keys: "
            + ", ".join(sorted(extras - {"lm_head.weight"})[:3])
        )
    # The loaded module is the bare language tower: strip the checkpoint's
    # language subtree prefix and ignore the vision tower, whose weights
    # anchor detection but are never executed for this family.
    renames = {
        key: key.removeprefix(KREA2_LANGUAGE_SUBTREE)
        for key in text_layout
        if key.startswith(KREA2_LANGUAGE_SUBTREE)
    }
    vision_keys = frozenset(key for key in text_layout if key.startswith(KREA2_VISION_SUBTREE))
    return cast(
        "ComponentPlan[object]",
        _plan(
            role,
            extracted,
            config,
            renames=renames,
            drop=vision_keys | {"lm_head.weight"},
        ),
    )


Ideogram4ComponentRole = Literal["diffusion", "qwen3vl_8b"]


def plan_ideogram4_component(
    source: WeightSource,
    role: Ideogram4ComponentRole,
) -> ComponentPlan[object]:
    """Plan one official independently supplied Ideogram 4 component."""

    split_prefixes = (
        ("", FLUX_DIFFUSION_PREFIX) if role == "diffusion" else ("", IDEOGRAM4_TEXT_PREFIX)
    )
    extracted = _component_source(role, source, None, "", split_prefixes=split_prefixes)
    if role == "diffusion":
        try:
            config = detect_ideogram4_config(extracted.geometries)
        except Ideogram4DetectError as error:
            raise AssemblyError(f"diffusion: {error}") from error
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        if any(
            geometry.dtype.kind != "float"
            for key, geometry in extracted.geometries.items()
            if key not in quantized_weights
        ):
            raise AssemblyError("diffusion: Ideogram 4 weights require floating-point storage")
        return cast("ComponentPlan[object]", _plan(role, extracted, config))

    layout = ideogram4_text_layout()
    text_geometries = {key: value for key, value in extracted.geometries.items() if key in layout}
    try:
        config = detect_ideogram4_text_config(text_geometries)
    except Ideogram4TextDetectError as error:
        raise AssemblyError(f"qwen3vl_8b: {error}") from error
    extras = set(extracted.geometries) - set(layout)
    if extras:
        raise AssemblyError("qwen3vl_8b: unexpected text keys: " + ", ".join(sorted(extras)[:3]))
    renames = {
        key: key.removeprefix(IDEOGRAM4_LANGUAGE_SUBTREE)
        for key in layout
        if key.startswith(IDEOGRAM4_LANGUAGE_SUBTREE)
        and not key.startswith(IDEOGRAM4_VISION_SUBTREE)
    }
    drop = frozenset(
        key for key in layout if key.startswith(IDEOGRAM4_VISION_SUBTREE) or key == "lm_head.weight"
    )
    return cast(
        "ComponentPlan[object]",
        _plan(role, extracted, config, renames=renames, drop=drop),
    )


AnimaComponentRole = Literal["diffusion", "qwen3_06b"]


def plan_anima_component(
    source: WeightSource,
    role: AnimaComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied Anima component."""
    split_prefixes = (
        ("net.", "", FLUX_DIFFUSION_PREFIX)
        if role == "diffusion"
        else ("model.", ANIMA_QWEN_PREFIX, "")
    )
    extracted = _component_source(
        role,
        source,
        None,
        "",
        split_prefixes=split_prefixes,
    )
    return _plan_anima_extracted(extracted, role)


def _plan_anima_extracted(
    extracted: _Extracted,
    role: AnimaComponentRole,
) -> ComponentPlan[object]:
    if role == "diffusion":
        if extracted.ignored:
            raise AssemblyError(
                "diffusion: source contains unsupported or duplicate tensors: "
                + ", ".join(extracted.ignored)
            )
        layout = anima_layout()
        if set(extracted.geometries) != set(layout) or any(
            extracted.geometries[key].shape != shape for key, shape in layout.items()
        ):
            raise AssemblyError("diffusion: source is not the exact Anima 2B layout")
        if any(geometry.dtype.kind != "float" for geometry in extracted.geometries.values()):
            raise AssemblyError("diffusion: Anima weights require floating-point storage")
        return cast("ComponentPlan[object]", _plan("diffusion", extracted, ANIMA_CONFIG))

    # Qwen3-0.6B releases may carry the tied lm_head duplicate beside
    # model.*, and the reference wrapper parks logit_scale beside the
    # transformer (comfy/sd1_clip.py SDClipModel @ 82f839f5).
    if any(
        not key.endswith("lm_head.weight") and key.split(".")[-1] != "logit_scale"
        for key in extracted.ignored
    ):
        raise AssemblyError("qwen3_06b: unexpected non-text keys in the text encoder source")
    try:
        qwen_config = detect_qwen_text_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"qwen3_06b: {error}") from error
    if qwen_config.architecture != "anima_qwen3_06b":
        raise AssemblyError("qwen3_06b: Anima requires the Qwen3-0.6B text profile")
    return cast("ComponentPlan[object]", _plan("qwen3_06b", extracted, qwen_config))


MiniMaxMusic3ComponentRole = Literal["diffusion", "text", "vae"]


def plan_minimax_music3_component(
    source: WeightSource,
    role: MiniMaxMusic3ComponentRole,
) -> ComponentPlan[MiniMaxMusic3Config | MiniMaxMusic3TextConfig | MiniMaxMusic3DavConfig]:
    """Plan one exact independently supplied MiniMax Music 3 component."""

    extracted = _component_source(
        role,
        source,
        None,
        "",
        split_prefixes=("model.diffusion_model.", "") if role == "diffusion" else ("",),
    )
    if extracted.ignored:
        raise AssemblyError(
            f"{role}: source contains unsupported tensors: " + ", ".join(extracted.ignored[:3])
        )
    if role == "diffusion":
        layout = minimax_music3_diffusion_layout()
        if set(extracted.geometries) != set(layout) or any(
            extracted.geometries[key].shape != shape for key, shape in layout.items()
        ):
            raise AssemblyError("diffusion: source is not the exact MiniMax Music 3 DiT layout")
        quantized = {
            key.removesuffix(".weight")
            for key, geometry in extracted.geometries.items()
            if geometry.dtype is INT8
        }
        if set(extracted.quant) != quantized:
            raise AssemblyError("diffusion: MiniMax Music 3 INT8 weights require ConvRot metadata")
        quantized_weights = {f"{layer}.weight" for layer in extracted.quant}
        if any(
            geometry.dtype.kind != "float"
            for key, geometry in extracted.geometries.items()
            if key not in quantized_weights
        ):
            raise AssemblyError("diffusion: MiniMax Music 3 weights require floating-point storage")
        return _plan(role, extracted, MINIMAX_MUSIC3_CONFIG)
    if role == "text":
        try:
            config = detect_minimax_music3_text_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"text: {error}") from error
        quantized = {
            key.removesuffix(".weight")
            for key, geometry in extracted.geometries.items()
            if geometry.dtype is INT8
        }
        if set(extracted.quant) != quantized:
            raise AssemblyError("text: MiniMax Music 3 INT8 weights require ConvRot metadata")
        return _plan(role, extracted, config, drop=frozenset({"tokenizer_json"}))
    layout = minimax_music3_dav_layout()
    if set(extracted.geometries) != set(layout) or any(
        extracted.geometries[key].shape != shape or extracted.geometries[key].dtype is not FLOAT32
        for key, shape in layout.items()
    ):
        raise AssemblyError("vae: source is not the exact MiniMax Music 3 DAV layout")
    return _plan(role, extracted, MINIMAX_MUSIC3_DAV_CONFIG)


Lumina2ComponentRole = Literal["diffusion", "gemma2_2b", "vae"]


def lumina2_tokenizer_source_key(source: WeightSource) -> str:
    """Return the unique validated tokenizer tensor's serialized key."""
    candidates = tuple(
        key for key in ("spiece_model", "text_encoders.spiece_model") if key in source.keys()
    )
    if len(candidates) != 1:
        raise AssemblyError("gemma2_2b: source must contain exactly one uint8 spiece_model tensor")
    geometry = source.entry(candidates[0]).geometry
    if (
        geometry.dtype != UINT8
        or len(geometry.shape) != 1
        or not 1 <= geometry.shape[0] <= 8 * 1024 * 1024
    ):
        raise AssemblyError("gemma2_2b: spiece_model must be a nonempty rank-1 uint8 tensor")
    return candidates[0]


def _plan_lumina2_extracted(
    extracted: _Extracted,
    role: Lumina2ComponentRole,
) -> ComponentPlan[object]:
    if role == "diffusion":
        layout = lumina2_layout()
        if (
            extracted.ignored
            or set(extracted.geometries) != set(layout)
            or any(
                extracted.geometries[key].shape != shape
                or extracted.geometries[key].dtype.kind != "float"
                for key, shape in layout.items()
            )
        ):
            raise AssemblyError("diffusion: source is not the exact Lumina Image 2.0 layout")
        return cast(
            "ComponentPlan[object]",
            _plan(
                role,
                extracted,
                LUMINA2_CONFIG,
                drop=frozenset({"norm_final.weight"}),
            ),
        )
    if role == "gemma2_2b":
        if any(
            key not in ("logit_scale", "spiece_model")
            and not key.endswith((".logit_scale", ".spiece_model"))
            for key in extracted.ignored
        ):
            raise AssemblyError("gemma2_2b: unexpected non-text keys in the text encoder source")
        try:
            gemma_config = detect_gemma_text_config(extracted.geometries)
        except (GemmaTextDetectError, ValueError) as error:
            raise AssemblyError(f"gemma2_2b: {error}") from error
        if gemma_config is not GEMMA2_LUMINA_2B_CONFIG:
            raise AssemblyError("gemma2_2b: Lumina2 requires the Gemma 2 2B text profile")
        return cast("ComponentPlan[object]", _plan(role, extracted, gemma_config))
    if role == "vae":
        try:
            kl_config = detect_kl_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"vae: {error}") from error
        if (
            kl_config.latent_channels != LUMINA2_CONFIG.latent_channels
            or kl_config.spatial_downscale != 8
            or kl_config.batch_norm_latent
        ):
            raise AssemblyError("vae: Lumina2 requires the 16-channel unpacked Flux KL VAE")
        vae_renames, vae_transforms = _kl_conversion(extracted.geometries)
        return cast(
            "ComponentPlan[object]",
            _plan(
                role,
                extracted,
                kl_config,
                renames=vae_renames,
                transforms=vae_transforms,
            ),
        )
    raise AssemblyError(f"unknown Lumina2 component role {role!r}")


def plan_lumina2_component(
    source: WeightSource,
    role: Lumina2ComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied Lumina Image 2.0 component."""
    if role == "diffusion":
        extracted = _component_source(
            role,
            source,
            None,
            FLUX_DIFFUSION_PREFIX,
            split_prefixes=("", FLUX_DIFFUSION_PREFIX),
        )
    elif role == "gemma2_2b":
        lumina2_tokenizer_source_key(source)
        extracted = _component_source(
            role,
            source,
            None,
            LUMINA2_GEMMA_PREFIX,
            root="text_encoders.gemma2_2b.",
            split_prefixes=("model.", LUMINA2_GEMMA_PREFIX),
        )
    elif role == "vae":
        extracted = _component_source(
            role,
            source,
            None,
            FLUX_VAE_PREFIX,
            split_prefixes=("", FLUX_VAE_PREFIX),
        )
    else:
        raise AssemblyError(f"unknown Lumina2 component role {role!r}")
    return _plan_lumina2_extracted(extracted, role)


@dataclass(frozen=True)
class Lumina2AssemblyPlan:
    """Complete Lumina2 diffusion, Gemma 2 2B, and Flux VAE loading contract."""

    diffusion: ComponentPlan[object]
    gemma2_2b: ComponentPlan[object]
    vae: ComponentPlan[object]
    family: ModelFamily = field(default=LUMINA2, init=False)
    tokenizer_source_key: str = field(default="text_encoders.spiece_model", kw_only=True)

    def __post_init__(self) -> None:
        if self.diffusion.config is not LUMINA2_CONFIG:
            raise ValueError("Lumina2 assembly requires the exact diffusion profile")
        if self.gemma2_2b.config is not GEMMA2_LUMINA_2B_CONFIG:
            raise ValueError("Lumina2 assembly requires the Gemma 2 2B profile")
        if (
            not isinstance(self.vae.config, KLConfig)
            or self.vae.config.latent_channels != LUMINA2_CONFIG.latent_channels
            or self.vae.config.spatial_downscale != 8
            or self.vae.config.batch_norm_latent
        ):
            raise ValueError("Lumina2 assembly requires the unpacked Flux KL VAE")

    @property
    def identity_components(self) -> tuple[ComponentPlan[object], ...]:
        return self.diffusion, self.gemma2_2b, self.vae


def plan_lumina2_assembly(*, checkpoint: WeightSource | None) -> Lumina2AssemblyPlan:
    """Plan the official all-in-one checkpoint without loading model tensors."""
    if checkpoint is None:
        raise AssemblyError("Lumina2 family runtime requires an all-in-one checkpoint")
    return Lumina2AssemblyPlan(*plan_lumina2_checkpoint(checkpoint))


def plan_lumina2_checkpoint(
    source: WeightSource,
) -> tuple[ComponentPlan[object], ComponentPlan[object], ComponentPlan[object]]:
    """Decompose one exact all-in-one Lumina2 checkpoint into independent plans."""
    lumina2_tokenizer_source_key(source)
    diffusion = _plan_lumina2_extracted(
        _component_source("diffusion", None, source, FLUX_DIFFUSION_PREFIX),
        "diffusion",
    )
    text = _plan_lumina2_extracted(
        _component_source(
            "gemma2_2b",
            None,
            source,
            LUMINA2_GEMMA_PREFIX,
            root="text_encoders.gemma2_2b.",
        ),
        "gemma2_2b",
    )
    vae = _plan_lumina2_extracted(
        _component_source("vae", None, source, FLUX_VAE_PREFIX),
        "vae",
    )
    claimed = {
        source_key for component in (diffusion, text, vae) for source_key in component.keys.values()
    }
    extras = set(source.keys()) - claimed
    allowed_extras = {
        "model.diffusion_model.norm_final.weight",
        "text_encoders.gemma2_2b.logit_scale",
        "text_encoders.spiece_model",
    }
    if extras != allowed_extras:
        missing = allowed_extras - extras
        unexpected = extras - allowed_extras
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise AssemblyError("Lumina2 checkpoint extras are not exact: " + "; ".join(details))
    return diffusion, text, vae


_Flux2Variant = tuple[ModelFamily, FluxConfig, str, str, tuple[str, ...]]

#: Neutral identity scope for the packed batch-norm KL VAE all three
#: published Flux2 releases share. Not a catalog family: the VAE never
#: enters a composed execution, so its component identity must not tie
#: it to any single release.
FLUX2_SHARED_COMPONENT_FAMILY_ID = "dinkster.flux2"

Flux2ComponentRole = Literal["diffusion", "mistral3_24b", "qwen3_8b", "qwen3_4b", "vae"]

#: Reference text-encoder component role for each Flux2 diffusion
#: family (comfy/sd.py flux2/klein clip targets @ b78cec87).
FLUX2_TEXT_ROLE_BY_FAMILY: Mapping[str, Flux2ComponentRole] = MappingProxyType(
    {variant[0].id: cast("Flux2ComponentRole", variant[2]) for variant in _FLUX2_VARIANTS}
)

_FLUX2_TEXT_ROLE_BY_ARCHITECTURE: Mapping[str, Flux2ComponentRole] = MappingProxyType(
    {
        "mistral3_24b": "mistral3_24b",
        "mistral3_24b_pruned": "mistral3_24b",
        "klein_qwen3_8b": "qwen3_8b",
        "klein_qwen3_4b": "qwen3_4b",
        "z_image_qwen3_4b": "qwen3_4b",
    }
)


def _plan_flux2_diffusion(
    extracted: _Extracted,
) -> tuple[_Flux2Variant, ComponentPlan[FluxConfig]]:
    try:
        flux2_config = detect_flux2_config(extracted.geometries)
    except Flux2DetectError as error:
        raise AssemblyError(f"diffusion: {error}") from error
    variant = next(entry for entry in _FLUX2_VARIANTS if entry[1] is flux2_config)
    plan = _plan(
        "diffusion",
        extracted,
        flux2_config,
        renames=_norm_renames(extracted.geometries),
    )
    return variant, plan


def _plan_flux2_text(
    extracted: _Extracted,
    variant: _Flux2Variant,
    *,
    text_root: str,
) -> ComponentPlan[QwenTextConfig]:
    family, _, slot, _, accepted = variant

    # The official dev text file parks the unused multimodal towers and
    # the tekken tokenizer blob beside model.*; Qwen releases may carry
    # the tied lm_head duplicate, and the reference wrapper parks
    # logit_scale beside the transformer (comfy/sd1_clip.py SDClipModel
    # @ b78cec87). All land in ignored; anything else refuses.
    def tolerated_text_sibling(key: str) -> bool:
        if key.startswith(text_root):
            relative = key.removeprefix(text_root)
            return relative in ("logit_scale", "transformer.lm_head.weight")
        if key in ("lm_head.weight", "logit_scale"):
            return True
        if family is not FLUX2_DEV:
            return False
        return key == "tekken_model" or key.startswith(("vision_tower.", "multi_modal_projector."))

    unexpected = tuple(key for key in extracted.ignored if not tolerated_text_sibling(key))
    if unexpected:
        raise AssemblyError(
            f"{slot}: unexpected non-text keys in the text encoder source: "
            + ", ".join(unexpected[:3])
        )
    try:
        text_config = detect_qwen_text_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"{slot}: {error}") from error
    # The Klein 4B tower is byte-identical to Z-Image's Qwen3-4B, so
    # detection returns that profile; Flux2 swaps in the Klein prompt
    # policy over the same weights.
    if text_config.architecture == "z_image_qwen3_4b" and "klein_qwen3_4b" in accepted:
        text_config = KLEIN_QWEN3_4B_CONFIG
    if text_config.architecture not in accepted:
        raise AssemblyError(
            f"{slot}: Flux2 {family.id} requires one of the {', '.join(accepted)}"
            f" text profiles, found {text_config.architecture}"
        )
    return _plan(slot, extracted, text_config)


def _plan_flux2_vae(extracted: _Extracted) -> ComponentPlan[KLConfig]:
    try:
        kl_config = detect_kl_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"vae: {error}") from error
    if not kl_config.batch_norm_latent or kl_config.latent_channels != FLUX2_LATENT_CHANNELS:
        raise AssemblyError("vae: Flux2 requires the 128-channel packed batch-norm KL VAE")
    vae_renames, vae_transforms = _kl_conversion(extracted.geometries)
    return _plan(
        "vae",
        extracted,
        kl_config,
        renames=vae_renames,
        transforms=vae_transforms,
    )


def identify_flux2_text_source(source: WeightSource) -> Flux2ComponentRole | None:
    """Return the Flux2 text-encoder role a split source carries, or None.

    Geometry-only: the Klein 4B tower detects as Z-Image's Qwen3-4B
    profile (byte-identical weights) and maps to the qwen3_4b role;
    only the caller's stated Flux2 intent decides which prompt policy
    rides those weights.
    """
    try:
        extracted = _component_source("text_encoder", source, None, "", split_prefixes=("model.",))
        config = detect_qwen_text_config(extracted.geometries)
    except (AssemblyError, ValueError):
        return None
    return _FLUX2_TEXT_ROLE_BY_ARCHITECTURE.get(config.architecture)


def identify_ltxav_text_source(
    source: WeightSource,
) -> Literal["gemma3_12b", "gemma4_12b"] | None:
    """Return the exact LTX-2 Gemma role carried by a split source, or None.

    Geometry-only: the verdict keys off the Gemma token-embedding layout
    under the split prefix, so the classic LTXV T5 encoder (which also
    ships a spiece_model tensor) never matches.
    """
    try:
        extracted = _component_source(
            "gemma3_12b",
            source,
            None,
            LTXAV_GEMMA_PREFIX,
            split_prefixes=(LTXAV_GEMMA_PREFIX,),
        )
        config = detect_gemma_text_config(extracted.geometries)
    except (AssemblyError, GemmaTextDetectError, ValueError):
        return None
    return "gemma4_12b" if config == GEMMA4_LTX_12B_CONFIG else "gemma3_12b"


def plan_flux2_component(
    source: WeightSource,
    role: Flux2ComponentRole,
) -> tuple[str, ComponentPlan[object]]:
    """Plan one exact independently supplied Flux2 component.

    Returns the identity family alongside the plan: the detected DiT
    geometry's family for the diffusion role, the slot's family for a
    text role, and the shared Flux2 scope for the VAE.
    """
    if role == "diffusion":
        extracted = _component_source(
            "diffusion",
            source,
            None,
            FLUX_DIFFUSION_PREFIX,
            split_prefixes=("", FLUX_DIFFUSION_PREFIX),
        )
        variant, plan = _plan_flux2_diffusion(extracted)
        return variant[0].id, cast("ComponentPlan[object]", plan)
    if role == "vae":
        extracted = _component_source("vae", source, None, FLUX_VAE_PREFIX)
        return (
            FLUX2_SHARED_COMPONENT_FAMILY_ID,
            cast("ComponentPlan[object]", _plan_flux2_vae(extracted)),
        )
    variant = next((entry for entry in _FLUX2_VARIANTS if entry[2] == role), None)
    if variant is None:
        raise AssemblyError(f"unknown Flux2 component role {role!r}")
    text_root = f"text_encoders.{role}."
    extracted = _component_source(
        role,
        source,
        None,
        variant[3],
        root=text_root,
        split_prefixes=("model.", variant[3]),
    )
    return (
        variant[0].id,
        cast("ComponentPlan[object]", _plan_flux2_text(extracted, variant, text_root=text_root)),
    )


def plan_flux2_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    text_encoder: WeightSource | None = None,
    text_slot: str | None = None,
    vae: WeightSource | None = None,
) -> Flux2AssemblyPlan:
    """Plan one published Flux2 release (dev, Klein 9B, Klein 4B).

    The detected DiT geometry picks the family, and the family picks
    which reference encoder the ``text_encoder`` source (or the
    combined checkpoint's text slot) must carry: Mistral3-Small 24B
    for dev (full or layer-pruned), Qwen3-8B for Klein 9B, Qwen3-4B
    for Klein 4B. ``text_slot`` is the name of the split slot the
    caller received ``text_encoder`` through; when given, it must be
    the family's own slot. Raises :class:`AssemblyError` (chaining
    the underlying detector error) when any component is missing,
    unrecognized, or mismatched with the family.
    """
    if all(source is None for source in (checkpoint, diffusion, text_encoder, vae)):
        raise AssemblyError("no sources given")

    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        FLUX_DIFFUSION_PREFIX,
        split_prefixes=("", FLUX_DIFFUSION_PREFIX),
    )
    variant, diffusion_plan = _plan_flux2_diffusion(extracted)
    family, _, slot, text_prefix, _ = variant
    if text_slot is not None and text_slot != slot:
        raise AssemblyError(
            f"{family.id} wires only the {slot} text encoder slot;"
            f" the text source arrived as {text_slot}"
        )

    text_root = f"text_encoders.{slot}."
    extracted = _component_source(
        slot,
        text_encoder,
        checkpoint,
        text_prefix,
        root=text_root,
        split_prefixes=("model.", text_prefix),
    )
    text_plan = _plan_flux2_text(extracted, variant, text_root=text_root)

    extracted = _component_source("vae", vae, checkpoint, FLUX_VAE_PREFIX)
    vae_plan = _plan_flux2_vae(extracted)

    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        claimed_prefixes = tuple(
            prefix
            for source, prefix in (
                (diffusion, FLUX_DIFFUSION_PREFIX),
                (text_encoder, text_root),
                (vae, FLUX_VAE_PREFIX),
            )
            if source is None
        )
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))
    return Flux2AssemblyPlan(family, diffusion_plan, text_plan, vae_plan, unclaimed)


TripoSplatComponentRole = Literal["dit", "dinov3-vision-conditioner", "gaussian-decoder"]


def plan_triposplat_component(
    source: WeightSource,
    role: TripoSplatComponentRole,
) -> ComponentPlan[object]:
    """Plan one exact independently supplied TripoSplat component.

    The DiT accepts the reference checkpoint prefix or a bare split
    file; the DINOv3 vision conditioner and the octree gaussian
    decoder ship bare-keyed.
    """
    if role == "dit":
        extracted = _component_source(
            "dit",
            source,
            None,
            TRIPOSPLAT_DIFFUSION_PREFIX,
            split_prefixes=("", TRIPOSPLAT_DIFFUSION_PREFIX),
        )
        return cast(
            "ComponentPlan[object]",
            _plan("dit", extracted, detect_triposplat_config(extracted.geometries)),
        )
    if role == "dinov3-vision-conditioner":
        extracted = _component_source(role, source, None, "")
        return cast(
            "ComponentPlan[object]",
            _plan(role, extracted, detect_dinov3_vith(extracted.geometries)),
        )
    if role == "gaussian-decoder":
        extracted = _component_source(role, source, None, "")
        return cast(
            "ComponentPlan[object]",
            _plan(role, extracted, detect_triposplat_gaussian_decoder(extracted.geometries)),
        )
    raise AssemblyError(f"unknown TripoSplat component role {role!r}")


TRELLIS2_STRUCTURE_PREFIX = "model.structure_model."
TRELLIS2_SHAPE_PREFIX = "model.img2shape."
TRELLIS2_SHAPE_512_PREFIX = "model.img2shape_512."
TRELLIS2_TEXTURE_PREFIX = "model.shape2txt."

Trellis2FlowRole = Literal[
    "structure",
    "shape",
    "shape-512",
    "texture",
    "texture-512",
]
Trellis2ComponentRole = Literal[
    "structure",
    "shape",
    "shape-512",
    "texture",
    "texture-512",
    "structure-decoder",
    "shape-decoder",
    "texture-decoder",
]


@dataclass(frozen=True)
class Trellis2ModelPlan:
    """The four execution-distinct flows in one official fused model."""

    structure: ComponentPlan[Trellis2FlowConfig]
    shape: ComponentPlan[Trellis2FlowConfig]
    shape_512: ComponentPlan[Trellis2FlowConfig]
    texture: ComponentPlan[Trellis2FlowConfig]

    @property
    def identity_components(self) -> tuple[ComponentPlan[Trellis2FlowConfig], ...]:
        return (self.structure, self.shape, self.shape_512, self.texture)


@dataclass(frozen=True)
class Trellis2SplitModelPlan:
    """The five execution-distinct flows in Microsoft's split release."""

    structure: ComponentPlan[Trellis2FlowConfig]
    shape: ComponentPlan[Trellis2FlowConfig]
    shape_512: ComponentPlan[Trellis2FlowConfig]
    texture: ComponentPlan[Trellis2FlowConfig]
    texture_512: ComponentPlan[Trellis2FlowConfig]

    @property
    def identity_components(self) -> tuple[ComponentPlan[Trellis2FlowConfig], ...]:
        return (
            self.structure,
            self.shape,
            self.shape_512,
            self.texture,
            self.texture_512,
        )


@dataclass(frozen=True)
class Trellis2VisionPlan:
    """DINOv3-L plus the optional Pixal3D NAF weights from one artifact."""

    dino: ComponentPlan[DINOv3ViTConfig]
    naf: ComponentPlan[NAFConfig] | None

    @property
    def identity_components(
        self,
    ) -> tuple[ComponentPlan[DINOv3ViTConfig] | ComponentPlan[NAFConfig], ...]:
        return (self.dino,) if self.naf is None else (self.dino, self.naf)


def plan_trellis2_flow_component(
    source: WeightSource,
    role: Trellis2FlowRole,
) -> ComponentPlan[Trellis2FlowConfig]:
    """Plan one flow from a bare split file or official fused model."""
    prefixes = {
        "structure": TRELLIS2_STRUCTURE_PREFIX,
        "shape": TRELLIS2_SHAPE_PREFIX,
        "shape-512": TRELLIS2_SHAPE_512_PREFIX,
        "texture": TRELLIS2_TEXTURE_PREFIX,
        "texture-512": TRELLIS2_TEXTURE_PREFIX,
    }
    prefix = prefixes[role]
    extracted = _component_source(
        role,
        source,
        None,
        prefix,
        split_prefixes=("", prefix),
    )
    try:
        config = detect_trellis2_flow(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"{role}: {error}") from error
    expected_stage = {
        "shape-512": "shape",
        "texture-512": "texture",
    }.get(role, role)
    if config.stage != expected_stage:
        raise AssemblyError(
            f"{role}: expected TRELLIS.2 {expected_stage} flow, found {config.stage}"
        )
    return _plan(role, extracted, config)


def plan_trellis2_model(source: WeightSource) -> Trellis2ModelPlan:
    """Plan all four flows from an official fused TRELLIS.2/Pixal3D artifact."""
    return Trellis2ModelPlan(
        structure=plan_trellis2_flow_component(source, "structure"),
        shape=plan_trellis2_flow_component(source, "shape"),
        shape_512=plan_trellis2_flow_component(source, "shape-512"),
        texture=plan_trellis2_flow_component(source, "texture"),
    )


def plan_trellis2_vision(source: WeightSource) -> Trellis2VisionPlan:
    """Plan exact TRELLIS.2 DINOv3-L weights and Pixal3D's optional NAF subtree."""
    path = _source_path(source, "trellis2-vision")
    whole = _extract(source, path, "trellis2-vision", "", root="")
    dino_keys = {key for key in whole.geometries if not key.startswith("naf.")}
    dino = _Extracted(
        path,
        {key: whole.geometries[key] for key in dino_keys},
        {key: whole.source_keys[key] for key in dino_keys},
        {key: value for key, value in whole.quant.items() if not key.startswith("naf.")},
        tuple(key for key in source.keys() if key.startswith("naf.")),
        whole.source_format,
        whole.runtime_facts,
        whole.payload_source,
    )
    try:
        dino_config = detect_dinov3_vitl(dino.geometries)
    except ValueError:
        dino_config = detect_dinov3_vith(dino.geometries)
    dino_plan = _plan("dino", dino, dino_config)
    naf_plan: ComponentPlan[NAFConfig] | None = None
    if any(key.startswith("naf.") for key in source.keys()):
        naf_extracted = _extract(source, path, "naf", "naf.", root="")
        try:
            naf_config = detect_naf(naf_extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"naf: {error}") from error
        naf_plan = _plan("naf", naf_extracted, naf_config)
    return Trellis2VisionPlan(dino_plan, naf_plan)


def plan_trellis2_decoder_component(
    source: WeightSource,
    role: Literal["structure-decoder", "shape-decoder", "texture-decoder"],
) -> ComponentPlan[Trellis2DecoderConfig]:
    """Plan one official fused VAE or one Microsoft split decoder."""
    extracted = _component_source(role, source, None, "")
    try:
        config = detect_trellis2_decoder(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"{role}: {error}") from error
    expected = role.removesuffix("-decoder")
    if config.kind != expected:
        raise AssemblyError(f"{role}: expected {expected} decoder, found {config.kind}")
    return _plan(role, extracted, config)


def plan_z_image_control(source: WeightSource, *, asset_digest: str) -> ZImageControlPlan:
    """Plan the exact standalone Alibaba PAI Z-Image Union control checkpoint."""
    role = "z_image_control"
    extracted = _component_source(role, source, None, "")
    if not detect_z_image_control(source):
        raise AssemblyError(f"{role}: source is not the exact 136-key BF16 Union checkpoint")
    return ZImageControlPlan(_plan(role, extracted, Z_IMAGE_CONTROL_CONFIG), asset_digest)


def plan_qwen_image_control(source: WeightSource, *, asset_digest: str) -> QwenImageControlPlan:
    """Plan an exact separately loaded maintained Qwen Image ControlNet."""
    role = "qwen_image_control"
    config, _ = require_qwen_image_control_layout(source)
    extracted = _component_source(role, source, None, "")
    return QwenImageControlPlan(_plan(role, extracted, config), asset_digest)


def plan_qwen_image_diffsynth(source: WeightSource, *, asset_digest: str) -> QwenImageDiffSynthPlan:
    """Plan an exact separately loaded Qwen Image DiffSynth block patch."""
    role = "qwen_image_diffsynth"
    config, _ = require_qwen_image_diffsynth_layout(source)
    extracted = _component_source(role, source, None, "")
    return QwenImageDiffSynthPlan(_plan(role, extracted, config), asset_digest)


def plan_wan21_uni3c(source: WeightSource, *, asset_digest: str) -> Wan21Uni3CPlan:
    """Plan the exact separately loaded Wan 2.1 Uni3C patch."""
    role = "wan21_uni3c"
    config, layout = require_wan21_uni3c_layout(source)
    extracted = _component_source(role, source, None, "")
    renames = {key: normalize_wan21_uni3c_key(key) for key in layout}
    return Wan21Uni3CPlan(_plan(role, extracted, config, renames=renames), asset_digest)


def plan_wan21_multitalk(source: WeightSource, *, asset_digest: str) -> Wan21MultiTalkPlan:
    """Plan the exact separately loaded Wan 2.1 MultiTalk patch."""
    role = "wan21_multitalk"
    config, _ = require_wan21_multitalk_layout(source)
    extracted = _component_source(role, source, None, "")
    return Wan21MultiTalkPlan(_plan(role, extracted, config), asset_digest)


@dataclass(frozen=True)
class TAESDCodecPlan:
    """Both independently loaded halves of one TAESD/TAESDXL codec."""

    config: TAESDConfig
    encoder: ComponentPlan[TAESDConfig]
    decoder: ComponentPlan[TAESDConfig]
    scale_keys: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.encoder.config.role != "encoder" or self.decoder.config.role != "decoder":
            raise ValueError("TAESD codec plan requires encoder and decoder halves")
        if (
            self.encoder.config.family != self.config.family
            or self.decoder.config.family != self.config.family
        ):
            raise ValueError("TAESD codec plan halves must have the selected family")


SDCodecPlan = ComponentPlan[KLConfig] | TAESDCodecPlan


@dataclass(frozen=True)
class SDAssemblyPlan:
    """The classic SD-era loading contract: which family the UNet is
    (SD 1.5 / SDXL base / SDXL refiner - conditioning and latent
    descriptors ride the family), plus one :class:`ComponentPlan` per
    component the family actually wires (SD 1.5 has no CLIP-G, the
    refiner no CLIP-L). ``unclaimed`` lists combined-checkpoint keys
    no component consumed (the LDM wrapper's schedule buffers,
    ``model_ema.*``; diagnostic only - the reference silently drops
    them)."""

    family: ModelFamily
    diffusion: ComponentPlan[UNetConfig]
    clip_l: ComponentPlan[ClipTextConfig] | None
    clip_g: ComponentPlan[ClipTextConfig] | None
    vae: SDCodecPlan
    unclaimed: tuple[str, ...] = ()
    sampling: SamplingDescriptor | None = field(default=None, repr=False)
    diffusion_asset_digest: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.clip_l is None and self.clip_g is None:
            raise ValueError("an SD plan wires at least one text encoder")
        effective_sampling = self.sampling or self.family.sampling
        expected_facts = _sampling_identity_facts(
            effective_sampling, canonical=self.family.sampling
        )
        if self.diffusion.identity_facts != expected_facts:
            raise ValueError("SD sampling descriptor and diffusion identity facts disagree")
        if isinstance(self.vae, TAESDCodecPlan):
            expected = "sd15" if self.family is SD15 else "sdxl"
            if self.vae.config.family != expected:
                raise ValueError(
                    f"{self.family.id} requires {expected} TAESD, got {self.vae.config.family}"
                )

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any] | None, ...]:
        """Components in the canonical native-identity order."""
        if isinstance(self.vae, TAESDCodecPlan):
            return (
                self.diffusion,
                self.clip_l,
                self.clip_g,
                self.vae.encoder,
                self.vae.decoder,
            )
        return (self.diffusion, self.clip_l, self.clip_g, self.vae)


def _sampling_identity_facts(
    sampling: SamplingDescriptor, *, canonical: SamplingDescriptor
) -> tuple[str, ...]:
    if sampling == canonical:
        return ()
    facts = [f"parameterization={sampling.parameterization.value}"]
    if sampling.space is SamplingSpace.CONTINUOUS_EDM:
        facts.extend(
            (
                f"sampling_space={sampling.space.value}",
                f"sigma_min={sampling.sigma_min!r}",
                f"sigma_max={sampling.sigma_max!r}",
            )
        )
    else:
        facts.append(f"zsnr={sampling.zsnr}")
    return tuple(facts)


def plan_sd15_controlnet(controlnet: WeightSource, *, asset_digest: str) -> ControlNetAssemblyPlan:
    """Plan one exact canonical or Diffusers classic SD1.5 ControlNet.

    ``asset_digest`` is the content identity established by the asset system;
    planning and assembly do not hash checkpoint payload bytes.
    This function reads only header entries and metadata-free source facts.
    Unsupported quantization markers, wrappers, aliases, and leftovers are
    rejected by the finite layout detector rather than read or ignored.
    """
    source_keys = tuple(controlnet.keys())
    if len(source_keys) != len(set(source_keys)):
        raise ControlNetDetectError("duplicate ControlNet source keys")
    geometries = {key: controlnet.entry(key).geometry for key in source_keys}
    config, source_layout, source_to_model = normalize_sd15_controlnet(geometries)
    model_to_source = {model_key: source_key for source_key, model_key in source_to_model.items()}
    component = ComponentPlan(
        component="controlnet",
        path=_source_path(controlnet, "controlnet"),
        config=config,
        keys=model_to_source,
        dtypes={
            model_key: geometries[source_key].dtype
            for model_key, source_key in model_to_source.items()
        },
        quant={},
    )
    return ControlNetAssemblyPlan(
        controlnet=component,
        layout=sd15_controlnet_layout(config),
        source_layout=source_layout,
        asset_digest=asset_digest,
        claims=tuple(sorted(source_to_model)),
    )


def plan_sd15_t2i_adapter(adapter: WeightSource, *, asset_digest: str) -> T2IAdapterAssemblyPlan:
    """Plan the exact 38-key TencentARC SD1.5 full adapter without payload reads."""
    source_keys = tuple(adapter.keys())
    if len(source_keys) != len(set(source_keys)):
        raise T2IAdapterDetectError("duplicate T2I Adapter source keys")
    geometries = {key: adapter.entry(key).geometry for key in source_keys}
    config, source_to_model = normalize_sd15_t2i_adapter(geometries)
    model_to_source = {model: source for source, model in source_to_model.items()}
    component = ComponentPlan(
        component="t2i_adapter",
        path=_source_path(adapter, "t2i_adapter"),
        config=config,
        keys=model_to_source,
        dtypes={model: geometries[source].dtype for model, source in model_to_source.items()},
        quant={},
    )
    return T2IAdapterAssemblyPlan(component, asset_digest, tuple(sorted(source_to_model)))


def plan_sd15_ipadapter(
    adapter: WeightSource,
    clip_vision: WeightSource,
    *,
    adapter_asset_digest: str,
    clip_vision_asset_digest: str,
) -> SD15IPAdapterAssemblyPlan:
    """Plan the official standard SD1.5 IP-Adapter and image encoder."""
    adapter_extracted = _component_source("ipadapter", adapter, None, "")
    if adapter_extracted.ignored:
        raise AssemblyError(
            "ipadapter: source contains unsupported tensors: "
            + ", ".join(adapter_extracted.ignored)
        )
    try:
        adapter_config = detect_sd15_ipadapter(adapter_extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"ipadapter: {error}") from error

    vision_extracted = _component_source("ipadapter_clip_vision", clip_vision, None, "")
    if vision_extracted.ignored:
        raise AssemblyError(
            "ipadapter_clip_vision: source contains unsupported tensors: "
            + ", ".join(vision_extracted.ignored)
        )
    try:
        vision_config = detect_sd15_ipadapter_clip_vision(vision_extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"ipadapter_clip_vision: {error}") from error
    return SD15IPAdapterAssemblyPlan(
        _plan("ipadapter", adapter_extracted, adapter_config),
        _plan("ipadapter_clip_vision", vision_extracted, vision_config),
        adapter_asset_digest,
        clip_vision_asset_digest,
    )


def plan_taesd_decoder(decoder: WeightSource, *, family: TAESDFamily) -> ComponentPlan[TAESDConfig]:
    """Plan one standalone TAESD decoder for sampling previews.

    Accepts the bare official madebyollin half (unprefixed keys) or the
    decoder half of a combined artifact (``taesd_decoder.``-prefixed keys);
    a combined artifact's encoder half is left unread. The caller supplies
    ``family`` - the two families are byte-indistinguishable.
    This function reads only header entries, never payload bytes.
    """
    source_keys = tuple(decoder.keys())
    if len(source_keys) != len(set(source_keys)):
        raise TAESDDetectError("duplicate TAESD source keys")
    geometries = {key: decoder.entry(key).geometry for key in source_keys}
    config = detect_taesd_config(geometries, family=family, role="decoder")
    prefix = "taesd_decoder."
    model_to_source: dict[str, str] = {}
    for key in source_keys:
        if key.startswith("taesd_encoder."):
            continue
        model_key = key[len(prefix) :] if key.startswith(prefix) else key
        if model_key in model_to_source:
            raise TAESDDetectError(f"TAESD decoder key {model_key!r} appears twice in the source")
        model_to_source[model_key] = key
    return ComponentPlan(
        component="taesd_decoder",
        path=_source_path(decoder, "taesd_decoder"),
        config=config,
        keys=model_to_source,
        dtypes={
            model_key: geometries[source_key].dtype
            for model_key, source_key in model_to_source.items()
        },
        quant={},
    )


def plan_taehv_decoder(decoder: WeightSource) -> ComponentPlan[TAEHVConfig]:
    """Plan one TAEHV video-TAE decoder for sampling previews.

    Accepts the ``decoder.``-prefixed half of a combined artifact (the
    published lighttaew2 files; their encoder half is left unread) or a
    bare decoder-only artifact. The Wan variants are geometry-detected.
    This function reads only header entries, never payload bytes.
    """
    source_keys = tuple(decoder.keys())
    if len(source_keys) != len(set(source_keys)):
        raise TAEHVDetectError("duplicate TAEHV source keys")
    geometries = {key: decoder.entry(key).geometry for key in source_keys}
    config = detect_taehv_decoder_config(geometries)
    prefix = "decoder."
    model_to_source: dict[str, str] = {}
    for key in source_keys:
        if key.startswith("encoder."):
            continue
        model_key = key[len(prefix) :] if key.startswith(prefix) else key
        if model_key in model_to_source:
            raise TAEHVDetectError(f"TAEHV decoder key {model_key!r} appears twice in the source")
        model_to_source[model_key] = key
    return ComponentPlan(
        component="taehv_decoder",
        path=_source_path(decoder, "taehv_decoder"),
        config=config,
        keys=model_to_source,
        dtypes={
            model_key: geometries[source_key].dtype
            for model_key, source_key in model_to_source.items()
        },
        quant={},
    )


def plan_sdxl_control_lora(
    control_lora: WeightSource,
    *,
    asset_digest: str,
    base_asset_digest: str,
) -> SDXLControlLoRAAssemblyPlan:
    """Plan the exact official rank-128 SDXL Control-LoRA without payload reads."""
    source_keys = tuple(control_lora.keys())
    if len(source_keys) != len(set(source_keys)):
        raise ControlNetDetectError("duplicate Control-LoRA source keys")
    geometries = {key: control_lora.entry(key).geometry for key in source_keys}
    config = detect_sdxl_control_lora(geometries)
    component = ComponentPlan(
        component="control_lora",
        path=_source_path(control_lora, "control_lora"),
        config=config,
        keys={key: key for key in source_keys},
        dtypes={key: geometries[key].dtype for key in source_keys},
        quant={},
    )
    return SDXLControlLoRAAssemblyPlan(
        component,
        asset_digest,
        base_asset_digest,
        tuple(sorted(source_keys)),
    )


def plan_sdxl_controlnet(
    controlnet: WeightSource,
    *,
    asset_digest: str,
) -> SDXLControlNetAssemblyPlan:
    """Plan one classic SDXL ControlNet without payload reads."""
    source_keys = tuple(controlnet.keys())
    if len(source_keys) != len(set(source_keys)):
        raise ControlNetDetectError("duplicate SDXL ControlNet source keys")
    geometries = {key: controlnet.entry(key).geometry for key in source_keys}
    config, source_to_model = normalize_sdxl_controlnet(geometries)
    component = ComponentPlan(
        component="controlnet",
        path=_source_path(controlnet, "controlnet"),
        config=config,
        keys={model: source for source, model in source_to_model.items()},
        dtypes={model: geometries[source].dtype for source, model in source_to_model.items()},
        quant={},
    )
    return SDXLControlNetAssemblyPlan(
        component,
        asset_digest,
        tuple(sorted(source_keys)),
    )


def plan_sdxl_controlnet_union(
    controlnet_union: WeightSource,
    *,
    asset_digest: str,
) -> SDXLControlNetUnionAssemblyPlan:
    """Plan one exact xinsir SDXL ControlNet Union without payload reads."""
    source_keys = tuple(controlnet_union.keys())
    if len(source_keys) != len(set(source_keys)):
        raise ControlNetDetectError("duplicate SDXL ControlNet Union source keys")
    geometries = {key: controlnet_union.entry(key).geometry for key in source_keys}
    config, source_to_model = normalize_sdxl_controlnet_union(geometries)
    component = ComponentPlan(
        component="controlnet_union",
        path=_source_path(controlnet_union, "controlnet_union"),
        config=config,
        keys={model: source for source, model in source_to_model.items()},
        dtypes={model: geometries[source].dtype for source, model in source_to_model.items()},
        quant={},
    )
    return SDXLControlNetUnionAssemblyPlan(
        component,
        asset_digest,
        tuple(sorted(source_keys)),
    )


def _checkpoint_float_scalar(source: WeightSource, key: str) -> float:
    if not isinstance(source, ConfigurationScalarSource):
        raise AssemblyError(
            f"diffusion: {key} numeric payload is unavailable from this weight source"
        )
    try:
        value = source.read_float_scalar(key)
    except (OSError, ValueError) as error:
        raise AssemblyError(f"diffusion: invalid {key}: {error}") from error
    if not isinstance(value, float):
        raise AssemblyError(f"diffusion: {key} did not produce a float")
    return value


#: Detected UNet profile -> registered family. Exact-config equality
#: is the loud-refusal gate: SD1.5 and SDXL accept their 9-channel
#: inpaint siblings; instruct-pix2pix, SD2.x, and the pruned
#: distillates all fail it (the
#: same variants the catalog detectors exclude).
_SD_FAMILY_BY_UNET = (
    (SD15_UNET_CONFIG, SD15),
    (SD15_INPAINT_UNET_CONFIG, SD15),
    (SDXL_UNET_CONFIG, SDXL),
    (SDXL_INPAINT_UNET_CONFIG, SDXL),
    (SDXL_REFINER_UNET_CONFIG, SDXL_REFINER),
)


def _clip_text_plan(
    component: str,
    extracted: _Extracted,
    *,
    expected: ClipTextConfig | None = None,
) -> ComponentPlan[ClipTextConfig]:
    """Plan one CLIP text-encoder slice in any of the three layouts
    the reference meets: transformers format, the legacy pre-4.31
    spelling without the ``text_model.`` infix (the reference inserts
    it, SD15.process_clip_state_dict @ b78cec87), and the OpenCLIP
    format (converted per openclip_text). ``expected`` pins which
    CLIP the component slot carries - a CLIP-L in a CLIP-G slot is a
    refusal, not a plan."""
    if is_openclip_text(extracted.geometries):
        if extracted.quant:
            error = QuantizationError(
                f"{component}: quantized OpenCLIP-format text encoders"
                " are not supported (the fused in_proj split would need"
                " per-chunk scale surgery)"
            )
            raise AssemblyError(str(error)) from error
        try:
            conversion = convert_openclip_text(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"{component}: {error}") from error
        config = detect_clip_text_config(
            {key: entry.geometry for key, entry in conversion.keys.items()}
        )
        if expected is not None and config != expected:
            raise AssemblyError(
                f"{component}: detected a {config.hidden_size}-wide CLIP"
                f" where a {expected.hidden_size}-wide one was expected"
            )
        keys = {
            model_key: extracted.source_keys[entry.source]
            for model_key, entry in conversion.keys.items()
        }
        return ComponentPlan(
            component=component,
            path=extracted.path,
            config=config,
            keys=keys,
            dtypes={
                model_key: entry.geometry.dtype for model_key, entry in conversion.keys.items()
            },
            quant={},
            ignored=tuple(extracted.ignored)
            + tuple(extracted.source_keys[key] for key in conversion.ignored),
            transforms={
                model_key: entry.transform
                for model_key, entry in conversion.keys.items()
                if entry.transform is not None
            },
        )

    # Transformers format; very old checkpoints lack the text_model.
    # infix and get it inserted, exactly like the reference.
    legacy = not any(key.startswith("text_model.") for key in extracted.geometries)
    renames = {key: f"text_model.{key}" for key in extracted.geometries} if legacy else {}
    inert = frozenset(key for key in extracted.geometries if key in _CLIP_TEXT_INERT_KEYS)
    view = {
        renames.get(key, key): geometry
        for key, geometry in extracted.geometries.items()
        if key not in inert
    }
    try:
        config = detect_clip_text_config(view)
    except ValueError as error:
        raise AssemblyError(f"{component}: {error}") from error
    if expected is not None and config != expected:
        raise AssemblyError(
            f"{component}: detected a {config.hidden_size}-wide CLIP"
            f" where a {expected.hidden_size}-wide one was expected"
        )
    return _plan(
        component,
        extracted,
        config,
        renames=renames,
        optional=CLIP_TEXT_OPTIONAL_KEYS,
        drop=inert,
    )


def plan_sd_assembly(
    *,
    checkpoint: WeightSource | None = None,
    diffusion: WeightSource | None = None,
    clip_l: WeightSource | None = None,
    clip_g: WeightSource | None = None,
    vae: WeightSource | None = None,
) -> SDAssemblyPlan:
    """Plan a classic SD 1.5 / SDXL base / SDXL refiner assembly from
    headers plus explicitly modeled scalar configuration tensors.

    Family detection remains header-only; planning never materializes
    model weights.

    ``checkpoint`` is a combined file; the component arguments are
    split files, each overriding the combined checkpoint for its
    component when both are present. A split text-encoder source for
    a slot the detected family does not wire is a caller bug, refused
    loudly. Raises :class:`AssemblyError` (chaining the underlying
    detector error) when any component is missing, unrecognized,
    outside the three supported UNet profiles, an unsupported SDXL
    sampling variant (Playground markers), or
    carries an unported quantization.
    """
    if all(source is None for source in (checkpoint, diffusion, clip_l, clip_g, vae)):
        raise AssemblyError("no sources given")

    # --- diffusion -------------------------------------------------
    extracted = _component_source(
        "diffusion",
        diffusion,
        checkpoint,
        SD_DIFFUSION_PREFIX,
        split_prefixes=("", SD_DIFFUSION_PREFIX),
    )
    marker_source = diffusion if diffusion is not None else checkpoint
    assert marker_source is not None  # diffusion extraction succeeded
    marker_keys = set(marker_source.keys())
    top_level_sampling_keys = set(SDXL_VARIANT_MARKERS) | {"ztsnr"}
    misplaced_markers = tuple(
        sorted(
            source_key
            for key, source_key in extracted.source_keys.items()
            if key in top_level_sampling_keys and source_key != key
        )
    )
    if misplaced_markers:
        raise AssemblyError(
            "diffusion: SDXL sampling markers must be top-level, not under the"
            f" UNet prefix ({', '.join(misplaced_markers)})"
        )
    extracted_markers = {
        key
        for key, source_key in extracted.source_keys.items()
        if key in top_level_sampling_keys and source_key == key
    }
    if extracted_markers:
        extracted = replace(
            extracted,
            geometries={
                key: geometry
                for key, geometry in extracted.geometries.items()
                if key not in extracted_markers
            },
            source_keys={
                key: source_key
                for key, source_key in extracted.source_keys.items()
                if key not in extracted_markers
            },
        )
    try:
        unet_config = detect_unet_config(extracted.geometries)
    except ValueError as error:
        raise AssemblyError(f"diffusion: {error}") from error
    family = next((fam for cfg, fam in _SD_FAMILY_BY_UNET if cfg == unet_config), None)
    if family is None:
        raise AssemblyError(
            "diffusion: a valid SD-era UNet outside the supported"
            f" profiles (in_channels={unet_config.in_channels},"
            f" context_dim={unet_config.context_dim},"
            f" adm_in_channels={unet_config.adm_in_channels},"
            f" model_channels={unet_config.model_channels});"
            " instruct-pix2pix, unsupported widened-input, and distillate"
            " variants are not"
            " ported (ROADMAP: Native inference)"
        )
    sampling = family.sampling
    identity_facts: tuple[str, ...] = ()
    if family is not SDXL and marker_keys & top_level_sampling_keys:
        raise AssemblyError(
            f"diffusion: top-level SDXL sampling markers are invalid for {family.id}"
        )
    if family is SDXL:
        # Sampling-variant markers sit at the top level of whichever
        # source carries the UNet (comfy/supported_models.py
        # SDXL.model_type @ b78cec87).
        present = tuple(marker for marker in ("edm_mean", "edm_std") if marker in marker_keys)
        if present:
            raise AssemblyError(
                "diffusion: SDXL sampling-variant markers present"
                f" ({', '.join(present)}); Playground is not ported"
                " (ROADMAP: Native inference)"
            )
        edm_max_key = "edm_vpred.sigma_max"
        edm_min_key = "edm_vpred.sigma_min"
        if edm_min_key in marker_keys and edm_max_key not in marker_keys:
            raise AssemblyError(f"diffusion: {edm_min_key} requires {edm_max_key}")
        if edm_max_key in marker_keys:
            sigma_max = _checkpoint_float_scalar(marker_source, edm_max_key)
            sigma_min = (
                _checkpoint_float_scalar(marker_source, edm_min_key)
                if edm_min_key in marker_keys
                else 0.002
            )
            try:
                sampling = SamplingDescriptor(
                    parameterization=Parameterization.V_PREDICTION,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    space=SamplingSpace.CONTINUOUS_EDM,
                )
            except ValueError as error:
                raise AssemblyError(
                    "diffusion: invalid EDM v-pred sigma bounds:"
                    f" sigma_min={sigma_min!r}, sigma_max={sigma_max!r} ({error})"
                ) from error
            identity_facts = _sampling_identity_facts(sampling, canonical=family.sampling)
        elif "v_pred" in marker_keys:
            zsnr = "ztsnr" in marker_keys
            entries = linear_beta_sigmas(zsnr=zsnr)
            sampling = SamplingDescriptor(
                parameterization=Parameterization.V_PREDICTION,
                sigma_min=entries[0],
                sigma_max=entries[-1],
                zsnr=zsnr,
            )
            identity_facts = _sampling_identity_facts(sampling, canonical=family.sampling)
    diffusion_plan = replace(
        _plan("diffusion", extracted, unet_config),
        identity_facts=identity_facts,
    )

    # --- text encoders ---------------------------------------------
    clip_l_plan: ComponentPlan[ClipTextConfig] | None = None
    clip_g_plan: ComponentPlan[ClipTextConfig] | None = None
    if family is SD15:
        if clip_g is not None:
            raise AssemblyError("clip_g: Stable Diffusion 1.5 wires no CLIP-G text encoder")
        clip_l_plan = _clip_text_plan(
            "clip_l",
            _component_source(
                "clip_l",
                clip_l,
                checkpoint,
                SD15_CLIP_L_PREFIX,
                root=_SD15_CLIP_ROOT,
            ),
            expected=CLIP_L_TEXT_CONFIG,
        )
    elif family is SDXL:
        clip_l_plan = _clip_text_plan(
            "clip_l",
            _component_source(
                "clip_l",
                clip_l,
                checkpoint,
                SDXL_CLIP_L_PREFIX,
                root=_SDXL_EMBEDDER_ROOTS[0],
            ),
            expected=CLIP_L_TEXT_CONFIG,
        )
        clip_g_plan = _clip_text_plan(
            "clip_g",
            _component_source(
                "clip_g",
                clip_g,
                checkpoint,
                SDXL_CLIP_G_PREFIX,
                root=_SDXL_EMBEDDER_ROOTS[1],
            ),
            expected=CLIP_G_TEXT_CONFIG,
        )
    else:
        if clip_l is not None:
            raise AssemblyError("clip_l: the SDXL refiner wires no CLIP-L text encoder")
        clip_g_plan = _clip_text_plan(
            "clip_g",
            _component_source(
                "clip_g",
                clip_g,
                checkpoint,
                SDXL_REFINER_CLIP_G_PREFIX,
                root=_SDXL_EMBEDDER_ROOTS[0],
            ),
            expected=CLIP_G_TEXT_CONFIG,
        )

    # --- vae -------------------------------------------------------
    extracted = _component_source("vae", vae, checkpoint, SD_VAE_PREFIX)
    taesd_family = "sd15" if family is SD15 else "sdxl"
    if any(key.startswith("taesd_") for key in extracted.geometries):
        scalar_keys = ("vae_scale", "vae_shift")
        unexpected = tuple(
            sorted(
                key
                for key in extracted.geometries
                if not key.startswith(("taesd_encoder.", "taesd_decoder."))
                and key not in scalar_keys
            )
        )
        if unexpected:
            raise AssemblyError(
                "vae: mixed TAESD and non-TAESD layout; unexpected keys: "
                + ", ".join(unexpected[:3])
            )
        present_scalars = tuple(key for key in scalar_keys if key in extracted.geometries)
        if present_scalars and present_scalars != scalar_keys:
            raise AssemblyError("vae: TAESD vae_scale and vae_shift must appear together")
        for key in present_scalars:
            geometry = extracted.geometries[key]
            if geometry.shape != () or geometry.dtype != FLOAT32:
                raise AssemblyError(f"vae: TAESD {key} must be a float32 scalar")
        halves: dict[str, ComponentPlan[TAESDConfig]] = {}
        for role in ("encoder", "decoder"):
            prefix = f"taesd_{role}."
            geometries = {
                key[len(prefix) :]: geometry
                for key, geometry in extracted.geometries.items()
                if key.startswith(prefix)
            }
            try:
                config = detect_taesd_config(geometries, family=taesd_family, role=role)
            except TAESDDetectError as error:
                raise AssemblyError(
                    f"vae: missing or invalid TAESD {role} half: {error}"
                ) from error
            half = _Extracted(
                extracted.path,
                geometries,
                {
                    key[len(prefix) :]: extracted.source_keys[key]
                    for key in extracted.geometries
                    if key.startswith(prefix)
                },
                {},
                (),
                extracted.source_format,
                extracted.runtime_facts,
                extracted.payload_source,
            )
            halves[role] = _plan(f"taesd_{role}", half, config)
        codec_plan: SDCodecPlan = TAESDCodecPlan(
            config=TAESDConfig(taesd_family, "encoder"),
            encoder=halves["encoder"],
            decoder=halves["decoder"],
            scale_keys=(
                (
                    extracted.source_keys[scalar_keys[0]],
                    extracted.source_keys[scalar_keys[1]],
                )
                if present_scalars
                else None
            ),
        )
    else:
        try:
            kl_config = detect_kl_config(extracted.geometries)
        except ValueError as error:
            raise AssemblyError(f"vae: {error}") from error
        vae_renames, vae_transforms = _kl_conversion(extracted.geometries)
        codec_plan = _plan(
            "vae",
            extracted,
            kl_config,
            renames=vae_renames,
            drop=_SD_VAE_INERT_KEYS,
            transforms=vae_transforms,
        )

    # --- unclaimed combined keys ------------------------------------
    unclaimed: tuple[str, ...] = ()
    if checkpoint is not None:
        text_roots: tuple[str, ...]
        if family is SD15:
            text_roots = (_SD15_CLIP_ROOT,) if clip_l is None else ()
        elif family is SDXL:
            text_roots = tuple(
                root
                for source, root in zip((clip_l, clip_g), _SDXL_EMBEDDER_ROOTS, strict=True)
                if source is None
            )
        else:
            text_roots = (_SDXL_EMBEDDER_ROOTS[0],) if clip_g is None else ()
        claimed_prefixes = text_roots + tuple(
            prefix
            for source, prefix in (
                (diffusion, SD_DIFFUSION_PREFIX),
                (vae, SD_VAE_PREFIX),
            )
            if source is None
        )
        unclaimed = tuple(key for key in checkpoint.keys() if not key.startswith(claimed_prefixes))

    return SDAssemblyPlan(
        family=family,
        sampling=sampling,
        diffusion=diffusion_plan,
        clip_l=clip_l_plan,
        clip_g=clip_g_plan,
        vae=codec_plan,
        unclaimed=unclaimed,
        diffusion_asset_digest=(
            marker_source.asset_digest if isinstance(marker_source, AssetIdentifiedSource) else None
        ),
    )


__all__ = [
    "FLUX_CLIP_L_PREFIX",
    "FLUX_DIFFUSION_PREFIX",
    "FLUX_QWEN_PREFIX",
    "FLUX_T5XXL_PREFIX",
    "FLUX_VAE_PREFIX",
    "WAN21_DIFFUSION_PREFIX",
    "WAN21_UMT5_PREFIX",
    "WAN21_VAE_PREFIX",
    "SD15_CLIP_L_PREFIX",
    "SD_DIFFUSION_PREFIX",
    "SD_VAE_PREFIX",
    "SDXL_CLIP_G_PREFIX",
    "SDXL_CLIP_L_PREFIX",
    "SDXL_REFINER_CLIP_G_PREFIX",
    "TRELLIS2_SHAPE_512_PREFIX",
    "TRELLIS2_SHAPE_PREFIX",
    "TRELLIS2_STRUCTURE_PREFIX",
    "TRELLIS2_TEXTURE_PREFIX",
    "AssemblyError",
    "AnimaComponentRole",
    "Lumina2AssemblyPlan",
    "Lumina2ComponentRole",
    "ComponentPlan",
    "NativePlanningContext",
    "ControlNetAssemblyPlan",
    "SDXLControlLoRAAssemblyPlan",
    "SDXLControlNetAssemblyPlan",
    "SDXLControlNetUnionAssemblyPlan",
    "SD15IPAdapterAssemblyPlan",
    "T2IAdapterAssemblyPlan",
    "FluxAssemblyPlan",
    "LTXAVAudioCodecPlan",
    "LTXAVStandaloneComponentPlan",
    "LTXAVStandaloneComponentRole",
    "LTXVStandaloneComponentPlan",
    "LTXVStandaloneComponentRole",
    "SDAssemblyPlan",
    "SDCodecPlan",
    "SeedVR2ComponentRole",
    "TAESDCodecPlan",
    "Trellis2ComponentRole",
    "Trellis2FlowRole",
    "Trellis2ModelPlan",
    "Trellis2SplitModelPlan",
    "Trellis2VisionPlan",
    "Wan21AssemblyPlan",
    "Wan21StandaloneComponentPlan",
    "Wan21StandaloneComponentRole",
    "QwenImageControlPlan",
    "QwenImageDiffSynthPlan",
    "ZImageControlPlan",
    "plan_flux_assembly",
    "plan_anima_component",
    "plan_lumina2_assembly",
    "plan_lumina2_checkpoint",
    "plan_lumina2_component",
    "plan_ltxav_audio_codec",
    "plan_ltxav_standalone_component",
    "plan_ltxv_standalone_component",
    "plan_sd15_controlnet",
    "plan_sd15_ipadapter",
    "plan_sd15_t2i_adapter",
    "plan_sdxl_control_lora",
    "plan_sdxl_controlnet",
    "plan_sdxl_controlnet_union",
    "plan_sd_assembly",
    "plan_seedvr2_component",
    "plan_taehv_decoder",
    "plan_taesd_decoder",
    "plan_trellis2_decoder_component",
    "plan_trellis2_flow_component",
    "plan_trellis2_model",
    "plan_trellis2_vision",
    "plan_wan21_assembly",
    "plan_wan21_standalone_component",
    "plan_wan22_assembly",
    "plan_qwen_image_control",
    "plan_qwen_image_diffsynth",
    "plan_z_image_control",
]
