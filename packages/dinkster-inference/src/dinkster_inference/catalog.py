"""Grounded model family catalog.

The model family catalog includes classic UNet image families, Flux,
Chroma, Flux2, core Wan 2.1/2.2 profiles, and MiniMax H3. Every
constant is ported from the ComfyUI reference commit cited by its
family implementation; nothing here is guessed:

- detection keys/shapes: comfy/model_detection.py
  (calculate_transformer_depth, the flux branch, detect_unet_config's
  classic-UNet tail)
- match values and per-family config: comfy/supported_models.py
  (SD15, SDXLRefiner, SDXL, Flux, Chroma, ChromaRadiance, Flux2)
- latent constants: comfy/latent_formats.py
  (SD15, SDXL, Flux, ChromaRadiance, Flux2)
- sigma endpoints and parameterization: comfy/model_sampling.py
  (ModelSamplingDiscrete defaults, ModelSamplingFlux/DiscreteFlow),
  comfy/model_base.py ModelType mapping

Deliberately deferred (ROADMAP "Native inference" ledger), and kept
UNRECOGNIZED rather than misdetected by explicit exclusions below:
instruct-pix2pix variants (first-conv axis 1 pinned away from 8 for
SD/SDXL), ``img_in`` axis 1 pinned to 64 for Flux, SSD1B/Segmind
Vega/KOALA (SDXL transformer-depth profile pinned via block keys),
SDXL Playground variants (top-level ``edm_mean``/``edm_std`` markers
excluded prefix-independently), the wider family zoo (SD2/SD3 and
other video/audio families), and quantized checkpoints whose 4-bit
packing halves linear axis 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anima import ANIMA_CONFIG, detect_anima
from .chroma import (
    CHROMA_FAMILY_ID,
    CHROMA_INFERENCE_DTYPES,
    CHROMA_RADIANCE_FAMILY_ID,
    detect_chroma,
)
from .devices import BFLOAT16, FLOAT16, FLOAT32
from .families import (
    ComponentWiring,
    DetectionEvidence,
    EngineProperties,
    FamilyFeatureHook,
    FamilyRegistry,
    ModelFamily,
    PreviewDecoderProperties,
)
from .flux import FluxConfig, detect_flux_config
from .flux2 import (
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B_CONFIG,
    detect_flux2,
)
from .ideogram4 import IDEOGRAM4_CONFIG, IDEOGRAM4_SIGMAS, detect_ideogram4
from .krea2 import KREA2_CONFIG, detect_krea2
from .latents import LatentDescriptor, MultiStreamLatentDescriptor
from .ltx import LTX_SAMPLING, LTXAV_LATENT, LTXV_LATENT, LTXAVDetector, LTXVDetector
from .lumina2 import LUMINA2_CONFIG, detect_lumina2
from .minimax_h3 import (
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_FAMILY,
    MINIMAX_H3_SIGMAS,
    detect_minimax_h3,
)
from .minimax_music3 import MINIMAX_MUSIC3_CONFIG, MINIMAX_MUSIC3_LATENT, detect_minimax_music3
from .qwen_image import QWEN_IMAGE_CONFIG, detect_qwen_image
from .sampling import Parameterization, SamplingDescriptor
from .seedvr2 import SEEDVR2_3B, SEEDVR2_SIGMAS, detect_seedvr2
from .signatures import DimField, KeySignature, RankIs, ShapeIn, ShapeIs
from .spaces import FlowSigmas
from .trellis2 import TRELLIS2_FAMILY_ID, detect_trellis2_flow
from .triposplat import (
    TRIPOSPLAT_CONFIG,
    TRIPOSPLAT_FAMILY,
    TRIPOSPLAT_SIGMAS,
    detect_triposplat,
)
from .wan21 import (
    WAN21_LATENT,
    WAN21_SAMPLING,
    WAN22_LATENT,
    WAN22_SAMPLING,
    Wan21Detector,
    Wan22Detector,
)
from .weights import WeightSource
from .z_image import Z_IMAGE_CONFIG, Z_IMAGE_PIXEL_CONFIG, detect_z_image

_COMMON_DTYPES = frozenset({FLOAT16, BFLOAT16, FLOAT32})


@dataclass(frozen=True)
class _MiniMaxH3Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_minimax_h3(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=MINIMAX_H3_FAMILY.id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


@dataclass(frozen=True)
class _MiniMaxMusic3Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_minimax_music3(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=evidence.config.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


# Combined-checkpoint prefix first, bare diffusion file second.
_UNET_PREFIXES = ("model.diffusion_model.", "")

# Endpoints of the default discrete sigma table: linear beta schedule,
# linear_start=0.00085, linear_end=0.012, 1000 steps
# (comfy/model_sampling.py ModelSamplingDiscrete @ b78cec87).
# spaces.linear_beta_sigmas computes the full table; the descriptor
# pins the span.
_DISCRETE_EPS = SamplingDescriptor(
    parameterization=Parameterization.EPS,
    sigma_min=0.029167158151720367,
    sigma_max=14.614641229333646,
)

# Flux dev: ModelSamplingFlux samples 10000 timesteps of
# flux_time_shift(shift=1.15, 1.0, t) = exp(1.15)/(exp(1.15) + 1/t - 1);
# sigma_min is the t=1/10000 endpoint (comfy/model_sampling.py
# ModelSamplingFlux @ b78cec87; float64 evaluation of the formula).
_FLUX_DEV_SIGMA_MIN = 0.0003157511457805717
# Flux schnell: ModelType.FLOW -> ModelSamplingDiscreteFlow with
# shift=1.0, multiplier=1.0 over 1000 steps; sigma_min is the first
# step, t=1/1000 (comfy/model_sampling.py ModelSamplingDiscreteFlow,
# comfy/supported_models.py FluxSchnell @ b78cec87).
_SCHNELL_SIGMA_MIN = 0.001

# Shape facts (comfy/model_detection.py @ b78cec87): first conv
# carries model_channels/in_channels; the first transformer block's
# attn2.to_k carries context_dim on axis 1; proj_in rank 4 = conv
# (SD1.x), rank 2 = linear (SGM/SDXL); label_emb axis 1 carries
# adm_in_channels. SD1.x's first transformer sits in input_blocks.1;
# SDXL's (transformer_depth [0,0,2,2,10,10]) in input_blocks.4.
_FIRST_CONV = "input_blocks.0.0.weight"
_SD1_TO_K = "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
_SD1_PROJ_IN = "input_blocks.1.1.proj_in.weight"
_SDXL_TO_K = "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
_SDXL_PROJ_IN = "input_blocks.4.1.proj_in.weight"
_LABEL_EMB = "label_emb.0.0.weight"


def _block(input_block: int, transformer_block: int) -> str:
    """Key marking that input block ``input_block`` has at least
    ``transformer_block + 1`` transformer blocks - the declarative pin
    for upstream's transformer_depth profiles (comfy/supported_models.py
    @ b78cec87). Present = depth reaches it, absent = depth stops
    short; a required/absent pair brackets the exact depth."""
    return f"input_blocks.{input_block}.1.transformer_blocks.{transformer_block}.attn2.to_k.weight"


# Base-model first convs take 4 latent channels; inpaint (9) and
# instruct-pix2pix (8) variants widen axis 1 (comfy/supported_models.py
# SD15_instructpix2pix/SDXL_instructpix2pix, BASE.inpaint_model
# @ b78cec87). SD1.5 accepts only base and native inpaint; 8 remains a
# loud refusal. SDXL has the same accepted widths under its existing family.
_LATENT_IN = 4
_SD15_INPUT_WIDTHS = (_LATENT_IN, 9)
_SDXL_INPUT_WIDTHS = (_LATENT_IN, 9)

# Top-level sampling markers that make an SDXL checkpoint a v-pred,
# EDM v-pred, or Playground EDM variant rather than plain EPS SDXL
# (comfy/supported_models.py SDXL.model_type @ b78cec87). They sit
# beside the prefixed UNet, so detection inspects them independently
# of the selected UNet prefix.
SDXL_VARIANT_MARKERS = (
    "v_pred",
    "edm_mean",
    "edm_std",
    "edm_vpred.sigma_max",
    "edm_vpred.sigma_min",
)
_SDXL_UNSUPPORTED_VARIANT_MARKERS = (
    "edm_mean",
    "edm_std",
)


@dataclass(frozen=True)
class _SDXLDetector:
    architecture: KeySignature

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = self.architecture.detect(source)
        if evidence is None:
            return None
        keys = set(source.keys())
        if any(marker in keys for marker in _SDXL_UNSUPPORTED_VARIANT_MARKERS):
            return None
        edm_max = "edm_vpred.sigma_max" in keys
        edm_min = "edm_vpred.sigma_min" in keys
        if edm_min and not edm_max:
            return None
        if edm_max:
            markers = ("edm_vpred.sigma_max",) + (("edm_vpred.sigma_min",) if edm_min else ())
            if any(
                source.entry(marker).geometry.numel != 1
                or not source.entry(marker).geometry.dtype.name.startswith("float")
                and source.entry(marker).geometry.dtype.name != "bfloat16"
                for marker in markers
            ):
                return None
            fields = dict(evidence.fields)
            fields["parameterization"] = Parameterization.V_PREDICTION.value
            fields["sampling_space"] = "continuous_edm"
            return DetectionEvidence(evidence.family_id, evidence.matched_keys + markers, fields)
        if "v_pred" not in keys:
            return evidence
        fields = dict(evidence.fields)
        fields["parameterization"] = Parameterization.V_PREDICTION.value
        fields["zsnr"] = "ztsnr" in keys
        markers = ("v_pred",) + (("ztsnr",) if "ztsnr" in keys else ())
        return DetectionEvidence(evidence.family_id, evidence.matched_keys + markers, fields)


SD15 = ModelFamily(
    id="dinkster.sd15",
    display_name="Stable Diffusion 1.5",
    detector=KeySignature(
        family_id="dinkster.sd15",
        prefixes=_UNET_PREFIXES,
        required=(_FIRST_CONV, _SD1_PROJ_IN, _SD1_TO_K),
        absent=(_LABEL_EMB,),
        constraints=(
            ShapeIs(_FIRST_CONV, 0, 320),  # model_channels
            ShapeIn(_FIRST_CONV, 1, _SD15_INPUT_WIDTHS),  # excludes 8-channel IP2P
            ShapeIs(_SD1_TO_K, 1, 768),  # context_dim (excludes SD2's 1024)
            RankIs(_SD1_PROJ_IN, 4),  # conv proj = not linear_in_transformer
        ),
        fields=(
            DimField("model_channels", _FIRST_CONV, 0),
            DimField("in_channels", _FIRST_CONV, 1),
            DimField("context_dim", _SD1_TO_K, 1),
        ),
    ),
    specificity=100,
    latent=LatentDescriptor(
        channels=4,
        scale_factor=0.18215,
        rgb_factors=(
            (0.3512, 0.2297, 0.3227),
            (0.3250, 0.4974, 0.2350),
            (-0.2829, 0.1762, 0.2721),
            (-0.2120, -0.2616, -0.7177),
        ),
        taesd_decoder="taesd_decoder",
    ),
    sampling=_DISCRETE_EPS,
    wiring=ComponentWiring(text_encoders=("dinkster.clip_l",)),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=1.0,
    engine=EngineProperties(
        diffusion_dtype=FLOAT16,
        text_dtype=FLOAT32,
        regional_memory_factor=1.0,
        clip_text_profile="sd1",
        ipadapter_profile="sd15",
        controlnet_profile="sd15",
        preview_decoder=PreviewDecoderProperties("taesd", "sd15"),
        compatibility_latent_formats=("SD15",),
        gguf_architecture="sd1",
    ),
)

_SDXL_LATENT = LatentDescriptor(
    channels=4,
    scale_factor=0.13025,
    rgb_factors=(
        (0.3651, 0.4232, 0.4341),
        (-0.2533, -0.0042, 0.1068),
        (0.1076, 0.1111, -0.0362),
        (-0.3165, -0.2492, -0.2188),
    ),
    rgb_bias=(0.1084, -0.0175, -0.0011),
    taesd_decoder="taesdxl_decoder",
)

SDXL = ModelFamily(
    id="dinkster.sdxl",
    display_name="Stable Diffusion XL",
    detector=_SDXLDetector(
        KeySignature(
            family_id="dinkster.sdxl",
            prefixes=_UNET_PREFIXES,
            # transformer_depth [0, 0, 2, 2, 10, 10]: exactly 2 blocks at
            # input block 4, exactly 10 at input block 7. SSD1B ([.., 4, 4])
            # and Segmind Vega ([.., 1, 1, 2, 2]) share every checked
            # dimension, so the depth profile is what tells them apart.
            required=(
                _FIRST_CONV,
                _LABEL_EMB,
                _SDXL_PROJ_IN,
                _SDXL_TO_K,
                _block(4, 1),
                _block(7, 9),
            ),
            absent=(_block(4, 2), _block(7, 10)),
            absent_toplevel=_SDXL_UNSUPPORTED_VARIANT_MARKERS,
            constraints=(
                ShapeIs(_FIRST_CONV, 0, 320),
                ShapeIn(_FIRST_CONV, 1, _SDXL_INPUT_WIDTHS),  # excludes 8-channel IP2P
                ShapeIs(_SDXL_TO_K, 1, 2048),
                ShapeIs(_LABEL_EMB, 1, 2816),  # adm_in_channels
                RankIs(_SDXL_PROJ_IN, 2),  # linear_in_transformer
            ),
            fields=(
                DimField("model_channels", _FIRST_CONV, 0),
                DimField("in_channels", _FIRST_CONV, 1),
                DimField("context_dim", _SDXL_TO_K, 1),
                DimField("adm_in_channels", _LABEL_EMB, 1),
            ),
        ),
    ),
    specificity=100,
    latent=_SDXL_LATENT,
    sampling=_DISCRETE_EPS,
    wiring=ComponentWiring(text_encoders=("dinkster.clip_l", "dinkster.clip_g")),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=0.8,
    engine=EngineProperties(
        diffusion_dtype=FLOAT16,
        text_dtype=FLOAT32,
        regional_memory_factor=0.8,
        clip_text_profile="sdxl",
        adm_profile="sdxl",
        controlnet_profile="sdxl",
        preview_decoder=PreviewDecoderProperties("taesd", "sdxl"),
        compatibility_latent_formats=("SDXL",),
        gguf_architecture="sdxl",
    ),
)

SDXL_REFINER = ModelFamily(
    id="dinkster.sdxl_refiner",
    display_name="Stable Diffusion XL Refiner",
    detector=KeySignature(
        family_id="dinkster.sdxl_refiner",
        prefixes=_UNET_PREFIXES,
        # transformer_depth [0, 0, 4, 4, 4, 4, 0, 0]: exactly 4 blocks
        # at input block 4.
        required=(_FIRST_CONV, _LABEL_EMB, _SDXL_PROJ_IN, _SDXL_TO_K, _block(4, 3)),
        absent=(_block(4, 4),),
        constraints=(
            ShapeIs(_FIRST_CONV, 0, 384),
            ShapeIs(_FIRST_CONV, 1, _LATENT_IN),
            ShapeIs(_SDXL_TO_K, 1, 1280),
            ShapeIs(_LABEL_EMB, 1, 2560),
            RankIs(_SDXL_PROJ_IN, 2),
        ),
        fields=(
            DimField("model_channels", _FIRST_CONV, 0),
            DimField("in_channels", _FIRST_CONV, 1),
            DimField("context_dim", _SDXL_TO_K, 1),
            DimField("adm_in_channels", _LABEL_EMB, 1),
        ),
    ),
    specificity=100,
    latent=_SDXL_LATENT,
    sampling=_DISCRETE_EPS,
    wiring=ComponentWiring(text_encoders=("dinkster.clip_g",)),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=1.0,
    engine=EngineProperties(
        diffusion_dtype=FLOAT16,
        text_dtype=FLOAT32,
        regional_memory_factor=1.0,
        clip_text_profile="sdxl",
        adm_profile="sdxl_refiner",
        preview_decoder=PreviewDecoderProperties("taesd", "sdxl"),
        gguf_architecture="sdxl",
    ),
)

# Flux checkpoints ship bare (BFL layout) or combined; RMSNorm params
# appear as .scale or .weight depending on the exporter (upstream's
# any_suffix_in, comfy/model_detection.py flux branch @ b78cec87).
_FLUX_PREFIXES = ("", "model.diffusion_model.")
_FLUX_REQUIRED = ("img_in.weight", "txt_in.weight", "vector_in.in_layer.weight")
_FLUX_KEY_NORM = (
    (
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "double_blocks.0.img_attn.norm.key_norm.weight",
    ),
)
_FLUX_FIELDS = (
    DimField("hidden_size", "img_in.weight", 0),
    DimField("context_in_dim", "txt_in.weight", 1),
    DimField("vec_in_dim", "vector_in.in_layer.weight", 1),
)
# img_in takes 16 latent channels x 2x2 patch = 64; FluxInpaint widens
# it to 384 (in_channels 96, comfy/supported_models.py FluxInpaint
# @ b78cec87), so pinning it keeps inpaint unrecognized.
_FLUX_IMG_IN = (ShapeIs("img_in.weight", 1, 64),)
# Chroma keeps the double_blocks layout but replaces vector_in and
# guidance_in with distilled_guidance_layer; keep it from matching the
# guidance-free schnell signature.
_FLUX_GUIDANCE = "guidance_in.in_layer.weight"
_CHROMA_MARKERS = (
    "distilled_guidance_layer.norms.0.scale",
    "distilled_guidance_layer.norms.0.weight",
)


@dataclass(frozen=True)
class _FluxSchnellDetector:
    """Keep classic Schnell's sparse signature while admitting only the
    exact vector-free Ovis layout through the existing family matcher."""

    classic: KeySignature
    vector_free_ovis: KeySignature

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = self.classic.detect(source)
        if evidence is not None:
            return evidence
        evidence = self.vector_free_ovis.detect(source)
        if evidence is None:
            return None
        prefix = str(evidence.fields["key_prefix"])
        geometries = {
            key.removeprefix(prefix): source.entry(key).geometry
            for key in source.keys()
            if key.startswith(prefix)
        }
        try:
            config = detect_flux_config(geometries)
        except ValueError:
            return None
        if config.vec_in_dim is not None or config.guidance_embed:
            return None
        return evidence


_FLUX_LATENT = LatentDescriptor(
    channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
    rgb_factors=(
        (-0.0346, 0.0244, 0.0681),
        (0.0034, 0.0210, 0.0687),
        (0.0275, -0.0668, -0.0433),
        (-0.0174, 0.0160, 0.0617),
        (0.0859, 0.0721, 0.0329),
        (0.0004, 0.0383, 0.0115),
        (0.0405, 0.0861, 0.0915),
        (-0.0236, -0.0185, -0.0259),
        (-0.0245, 0.0250, 0.1180),
        (0.1008, 0.0755, -0.0421),
        (-0.0515, 0.0201, 0.0011),
        (0.0428, -0.0012, -0.0036),
        (0.0817, 0.0765, 0.0749),
        (-0.1264, -0.0522, -0.1103),
        (-0.0280, -0.0881, -0.0499),
        (-0.1262, -0.0982, -0.0778),
    ),
    rgb_bias=(-0.0329, -0.0718, -0.0851),
    taesd_decoder="taef1_decoder",
)

_Z_IMAGE_PIXEL_LATENT = LatentDescriptor(
    channels=3,
    spatial_downscale=1,
)

_FLUX_WIRING = ComponentWiring(
    vae_prefix="vae.",
    text_encoder_prefix="text_encoders.",
    text_encoders=("dinkster.clip_l", "dinkster.t5xxl"),
)

_CHROMA_WIRING = ComponentWiring(
    vae_prefix="vae.",
    text_encoder_prefix="text_encoders.",
    text_encoders=("dinkster.t5xxl",),
)


@dataclass(frozen=True)
class _ChromaDetector:
    family_id: str

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_chroma(source)
        if evidence is None or evidence.family_id != self.family_id:
            return None
        return evidence


_CHROMA_ENGINE = EngineProperties(
    quantized_component_load_device=True,
    attention_backends=(("diffusion", "flux"), ("t5xxl", "t5"), ("vae", "vae")),
    attention_requires_route=True,
)


CHROMA = ModelFamily(
    id=CHROMA_FAMILY_ID,
    display_name="Chroma",
    detector=_ChromaDetector(CHROMA_FAMILY_ID),
    specificity=110,
    latent=_FLUX_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_SCHNELL_SIGMA_MIN,
        sigma_max=1.0,
        shift=1.0,
    ),
    wiring=_CHROMA_WIRING,
    supported_dtypes=CHROMA_INFERENCE_DTYPES,
    memory_factor=3.2,
    engine=_CHROMA_ENGINE,
)

CHROMA_RADIANCE = ModelFamily(
    id=CHROMA_RADIANCE_FAMILY_ID,
    display_name="Chroma Radiance",
    detector=_ChromaDetector(CHROMA_RADIANCE_FAMILY_ID),
    specificity=120,
    latent=_Z_IMAGE_PIXEL_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_SCHNELL_SIGMA_MIN,
        sigma_max=1.0,
        shift=1.0,
    ),
    wiring=_CHROMA_WIRING,
    supported_dtypes=CHROMA_INFERENCE_DTYPES,
    memory_factor=0.044,
    engine=_CHROMA_ENGINE,
)

FLUX_DEV = ModelFamily(
    id="dinkster.flux_dev",
    display_name="Flux (guidance-distilled)",
    detector=KeySignature(
        family_id="dinkster.flux_dev",
        prefixes=_FLUX_PREFIXES,
        required=(*_FLUX_REQUIRED, _FLUX_GUIDANCE),
        required_any=_FLUX_KEY_NORM,
        constraints=_FLUX_IMG_IN,
        fields=_FLUX_FIELDS,
    ),
    specificity=100,
    latent=_FLUX_LATENT,
    # ModelType.FLUX -> ModelSamplingFlux, default shift 1.15
    # (comfy/model_base.py, comfy/model_sampling.py @ b78cec87).
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_FLUX_DEV_SIGMA_MIN,
        sigma_max=1.0,
        shift=1.15,
    ),
    wiring=_FLUX_WIRING,
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=3.1,
    engine=EngineProperties(
        sigma_space="flux", regional_memory_factor=3.1, gguf_architecture="flux"
    ),
)

FLUX_SCHNELL = ModelFamily(
    id="dinkster.flux_schnell",
    display_name="Flux Schnell",
    detector=_FluxSchnellDetector(
        classic=KeySignature(
            family_id="dinkster.flux_schnell",
            prefixes=_FLUX_PREFIXES,
            required=_FLUX_REQUIRED,
            required_any=_FLUX_KEY_NORM,
            absent=(_FLUX_GUIDANCE, *_CHROMA_MARKERS),
            constraints=_FLUX_IMG_IN,
            fields=_FLUX_FIELDS,
        ),
        vector_free_ovis=KeySignature(
            family_id="dinkster.flux_schnell",
            prefixes=_FLUX_PREFIXES,
            required=(
                "img_in.weight",
                "txt_in.weight",
                "double_blocks.0.img_mlp.gate_proj.weight",
            ),
            required_any=(
                *_FLUX_KEY_NORM,
                ("txt_norm.scale", "txt_norm.weight"),
            ),
            absent=(
                "vector_in.in_layer.weight",
                _FLUX_GUIDANCE,
                *_CHROMA_MARKERS,
            ),
            constraints=(
                *_FLUX_IMG_IN,
                ShapeIs("img_in.weight", 0, 3072),
                ShapeIs("txt_in.weight", 0, 3072),
                ShapeIs("txt_in.weight", 1, 2048),
            ),
            fields=(
                DimField("hidden_size", "img_in.weight", 0),
                DimField("context_in_dim", "txt_in.weight", 1),
            ),
        ),
    ),
    specificity=100,
    latent=_FLUX_LATENT,
    # ModelType.FLOW with shift=1.0, multiplier=1.0
    # (comfy/supported_models.py FluxSchnell @ b78cec87).
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_SCHNELL_SIGMA_MIN,
        sigma_max=1.0,
        shift=1.0,
    ),
    wiring=_FLUX_WIRING,
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=3.1,
    engine=EngineProperties(regional_memory_factor=3.1, gguf_architecture="flux"),
)


@dataclass(frozen=True)
class _Flux2Detector:
    family_id: str
    config: FluxConfig

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_flux2(source)
        if evidence is None or evidence.config != self.config:
            return None
        return DetectionEvidence(
            family_id=self.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


# Flux2: ModelSamplingFlux with shift 2.02 over 10000 timesteps
# (comfy/supported_models.py Flux2 sampling_settings,
# comfy/model_sampling.py ModelSamplingFlux @ b78cec87); sigma_min is
# the float64 t=1/10000 endpoint of flux_time_shift(2.02, 1.0, t),
# the same evaluation convention as _FLUX_DEV_SIGMA_MIN.
_FLUX2_SAMPLING = SamplingDescriptor(
    parameterization=Parameterization.FLOW,
    sigma_min=0.0007533399352379832,
    sigma_max=1.0,
    shift=2.02,
)

# The packed Flux2 latent: 128 channels are a 2x2 pixel shuffle of the
# VAE's 32-channel space, total spatial downscale 16, identity
# process_in/out (comfy/latent_formats.py Flux2 @ b78cec87). The
# reference's preview factors are 32 rows applied after unpacking;
# LatentDescriptor has no unpack seam, so cheap previews stay
# undeclared rather than misdeclared against the packed channels.
_FLUX2_LATENT = LatentDescriptor(
    channels=128,
    spatial_downscale=16,
)


# comfy/supported_models.py Flux2.__init__ @ b78cec87: the inherited
# Flux factor 3.1 times (2.0 * 2.0) times hidden_size / 2604.
def _flux2_memory_factor(hidden_size: int) -> float:
    return 3.1 * (2.0 * 2.0) * (hidden_size / 2604)


FLUX2_DEV = ModelFamily(
    id="dinkster.flux2_dev",
    display_name="Flux2 Dev",
    detector=_Flux2Detector("dinkster.flux2_dev", FLUX2_DEV_CONFIG),
    specificity=100,
    latent=_FLUX2_LATENT,
    sampling=_FLUX2_SAMPLING,
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.mistral3_24b",),
    ),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=_flux2_memory_factor(FLUX2_DEV_CONFIG.hidden_size),
)

FLUX2_KLEIN_9B = ModelFamily(
    id="dinkster.flux2_klein_9b",
    display_name="Flux2 Klein 9B",
    detector=_Flux2Detector("dinkster.flux2_klein_9b", FLUX2_KLEIN_9B_CONFIG),
    specificity=100,
    latent=_FLUX2_LATENT,
    sampling=_FLUX2_SAMPLING,
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3_8b",),
    ),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=_flux2_memory_factor(FLUX2_KLEIN_9B_CONFIG.hidden_size),
)

FLUX2_KLEIN_4B = ModelFamily(
    id="dinkster.flux2_klein_4b",
    display_name="Flux2 Klein 4B",
    detector=_Flux2Detector("dinkster.flux2_klein_4b", FLUX2_KLEIN_4B_CONFIG),
    specificity=100,
    latent=_FLUX2_LATENT,
    sampling=_FLUX2_SAMPLING,
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3_4b",),
    ),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=_flux2_memory_factor(FLUX2_KLEIN_4B_CONFIG.hidden_size),
)

WAN21 = ModelFamily(
    id="dinkster.wan21",
    display_name="Wan 2.1",
    detector=Wan21Detector(),
    specificity=110,
    latent=WAN21_LATENT,
    sampling=WAN21_SAMPLING,
    wiring=ComponentWiring(text_encoders=("dinkster.umt5xxl",)),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=1536 / 2222,
    engine=EngineProperties(
        preview_decoder=PreviewDecoderProperties("taehv"),
        compatibility_latent_formats=("Wan21",),
        supports_context_windows=True,
    ),
)

WAN22 = ModelFamily(
    id="dinkster.wan22",
    display_name="Wan 2.2 TI2V 5B",
    detector=Wan22Detector(),
    specificity=111,
    latent=WAN22_LATENT,
    sampling=WAN22_SAMPLING,
    wiring=ComponentWiring(text_encoders=("dinkster.umt5xxl",)),
    supported_dtypes=_COMMON_DTYPES,
    memory_factor=3072 / 2222,
    engine=EngineProperties(
        text_dtype=FLOAT32,
        vae_dtypes=(BFLOAT16, FLOAT16, FLOAT32),
        preview_decoder=PreviewDecoderProperties("taehv"),
        compatibility_latent_formats=("Wan22",),
        supports_context_windows=True,
    ),
)

LTXV = ModelFamily(
    id="dinkster.ltxv",
    display_name="LTX-Video",
    detector=LTXVDetector(),
    specificity=110,
    latent=LTXV_LATENT,
    sampling=LTX_SAMPLING,
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.t5xxl",),
    ),
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    memory_factor=5.5,
    engine=EngineProperties(supports_context_windows=True),
)

LTXAV = ModelFamily(
    id="dinkster.ltxav",
    display_name="LTX Audio-Video",
    detector=LTXAVDetector(),
    specificity=110,
    latent=LTXAV_LATENT,
    sampling=LTX_SAMPLING,
    # AV checkpoints ship encoders as split artifacts or bundled components.
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=(),
    ),
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    memory_factor=0.077,
    engine=EngineProperties(
        attention_backends=(
            ("diffusion", "flux"),
            ("gemma3_12b", "qwen"),
            ("gemma4_12b", "qwen"),
            ("connectors", "flux"),
        )
    ),
)


@dataclass(frozen=True)
class _ZImageDetector:
    family_id: str

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_z_image(source)
        if evidence is None or evidence.config.family_id != self.family_id:
            return None
        return DetectionEvidence(
            family_id=self.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


Z_IMAGE = ModelFamily(
    id=Z_IMAGE_CONFIG.family_id,
    display_name="Z-Image",
    detector=_ZImageDetector(Z_IMAGE_CONFIG.family_id),
    specificity=110,
    latent=_FLUX_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=0.0029940119760479044,
        sigma_max=1.0,
        shift=Z_IMAGE_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3_4b",),
    ),
    supported_dtypes=frozenset(Z_IMAGE_CONFIG.inference_dtypes),
    memory_factor=Z_IMAGE_CONFIG.memory_factor,
    engine=EngineProperties(text_dtype=FLOAT32),
)


@dataclass(frozen=True)
class _Lumina2Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_lumina2(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=LUMINA2_CONFIG.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


LUMINA2 = ModelFamily(
    id=LUMINA2_CONFIG.family_id,
    display_name="Lumina Image 2.0",
    detector=_Lumina2Detector(),
    specificity=120,
    latent=_FLUX_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=0.005970149253731343,
        sigma_max=1.0,
        shift=LUMINA2_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.gemma2_2b",),
    ),
    supported_dtypes=frozenset(LUMINA2_CONFIG.inference_dtypes),
    memory_factor=LUMINA2_CONFIG.memory_factor,
)

Z_IMAGE_PIXEL_SPACE = ModelFamily(
    id=Z_IMAGE_PIXEL_CONFIG.family_id,
    display_name="Z-Image Pixel Space",
    detector=_ZImageDetector(Z_IMAGE_PIXEL_CONFIG.family_id),
    specificity=120,
    latent=_Z_IMAGE_PIXEL_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=0.0029940119760479044,
        sigma_max=1.0,
        shift=Z_IMAGE_PIXEL_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3_4b",),
    ),
    supported_dtypes=frozenset(Z_IMAGE_PIXEL_CONFIG.inference_dtypes),
    memory_factor=Z_IMAGE_PIXEL_CONFIG.memory_factor,
    engine=EngineProperties(text_dtype=FLOAT32),
)


MINIMAX_H3 = ModelFamily(
    id=MINIMAX_H3_FAMILY.id,
    display_name=MINIMAX_H3_FAMILY.display_name,
    detector=_MiniMaxH3Detector(),
    specificity=100,
    latent=MultiStreamLatentDescriptor(
        (
            (
                "video",
                LatentDescriptor(
                    channels=MINIMAX_H3_CONFIG.video_latent_channels,
                    dimensions=3,
                    spatial_downscale=MINIMAX_H3_CONFIG.video_spatial_downscale,
                ),
            ),
            (
                "audio",
                LatentDescriptor(
                    channels=MINIMAX_H3_CONFIG.audio_latent_channels,
                    dimensions=1,
                ),
            ),
        )
    ),
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=MINIMAX_H3_SIGMAS.sigma_min,
        sigma_max=MINIMAX_H3_SIGMAS.sigma_max,
    ),
    # H3 ships as split artifacts rather than a combined checkpoint.
    wiring=ComponentWiring(text_encoders=()),
    supported_dtypes=frozenset({BFLOAT16}),
    aliases=MINIMAX_H3_FAMILY.aliases,
    engine=EngineProperties(
        attention_backends=(("diffusion", "flux"),),
        feature_hooks=(
            FamilyFeatureHook(
                "lora-key-map",
                "dinkster_inference:minimax_h3_lora_key_map",
                ("diffusion",),
            ),
        ),
    ),
)


MINIMAX_MUSIC3 = ModelFamily(
    id=MINIMAX_MUSIC3_CONFIG.family_id,
    display_name="MiniMax Music 3",
    detector=_MiniMaxMusic3Detector(),
    specificity=120,
    latent=MINIMAX_MUSIC3_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=0.001,
        sigma_max=1.0,
    ),
    wiring=ComponentWiring(text_encoders=("dinkster.minimax_music3_text",)),
    supported_dtypes=frozenset(MINIMAX_MUSIC3_CONFIG.inference_dtypes),
    memory_factor=MINIMAX_MUSIC3_CONFIG.memory_factor,
    aliases=("minimax_music3", "minimax-music-3"),
    engine=EngineProperties(
        attention_backends=(("diffusion", "flux"), ("text", "qwen")),
        attention_requires_route=True,
    ),
)


@dataclass(frozen=True)
class _QwenImageDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_qwen_image(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=evidence.config.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


QWEN_IMAGE = ModelFamily(
    id=QWEN_IMAGE_CONFIG.family_id,
    display_name="Qwen Image",
    detector=_QwenImageDetector(),
    specificity=115,
    latent=WAN21_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_FLUX_DEV_SIGMA_MIN,
        sigma_max=1.0,
        shift=QWEN_IMAGE_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen2_5_vl_7b",),
    ),
    supported_dtypes=frozenset(QWEN_IMAGE_CONFIG.inference_dtypes),
    memory_factor=QWEN_IMAGE_CONFIG.memory_factor,
)


@dataclass(frozen=True)
class _Krea2Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_krea2(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=evidence.config.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


KREA2 = ModelFamily(
    id=KREA2_CONFIG.family_id,
    display_name="Krea 2",
    detector=_Krea2Detector(),
    specificity=100,
    latent=WAN21_LATENT,
    # ModelType.FLUX: ModelSamplingFlux over 10000 timesteps with
    # sampling_settings shift 1.15 (comfy/model_base.py Krea2,
    # comfy/supported_models.py Krea2 @ b78cec87); the same evaluation
    # convention as _FLUX_DEV_SIGMA_MIN.
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=_FLUX_DEV_SIGMA_MIN,
        sigma_max=1.0,
        shift=KREA2_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3vl_4b",),
    ),
    supported_dtypes=frozenset(KREA2_CONFIG.inference_dtypes),
    memory_factor=KREA2_CONFIG.memory_factor,
    engine=EngineProperties(preview_decoder=PreviewDecoderProperties("taehv")),
)


@dataclass(frozen=True)
class _Ideogram4Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_ideogram4(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=evidence.config.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


IDEOGRAM4 = ModelFamily(
    id=IDEOGRAM4_CONFIG.family_id,
    display_name="Ideogram 4",
    detector=_Ideogram4Detector(),
    specificity=120,
    latent=_FLUX2_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=IDEOGRAM4_SIGMAS.sigma_min,
        sigma_max=IDEOGRAM4_SIGMAS.sigma_max,
        shift=IDEOGRAM4_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3vl_8b",),
    ),
    supported_dtypes=frozenset(IDEOGRAM4_CONFIG.inference_dtypes),
    memory_factor=IDEOGRAM4_CONFIG.memory_factor,
)


@dataclass(frozen=True)
class _SeedVR2Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_seedvr2(source)
        if evidence is None:
            return None
        return DetectionEvidence(evidence.config.family_id, evidence.matched_keys, evidence.fields)


SEEDVR2 = ModelFamily(
    id=SEEDVR2_3B.family_id,
    display_name="SeedVR2",
    detector=_SeedVR2Detector(),
    specificity=120,
    latent=LatentDescriptor(
        channels=16,
        dimensions=3,
        spatial_downscale=8,
        temporal_downscale=4,
        temporal_causal=True,
    ),
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=SEEDVR2_SIGMAS.sigma_min,
        sigma_max=SEEDVR2_SIGMAS.sigma_max,
        shift=SEEDVR2_SIGMAS.shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=(),
    ),
    supported_dtypes=frozenset(SEEDVR2_3B.inference_dtypes),
    memory_factor=SEEDVR2_3B.memory_factor,
)


@dataclass(frozen=True)
class _AnimaDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_anima(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=evidence.config.family_id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


ANIMA = ModelFamily(
    id=ANIMA_CONFIG.family_id,
    display_name="Anima",
    detector=_AnimaDetector(),
    specificity=100,
    latent=WAN21_LATENT,
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        # shift * t / (1 + (shift - 1) * t) at t = 1/1000 and t = 1
        # with shift 3.0 (ComfyUI supported_models Anima
        # sampling_settings @ abe2ec26).
        sigma_min=0.0029940119760479044,
        sigma_max=1.0,
        shift=ANIMA_CONFIG.sampling_shift,
    ),
    wiring=ComponentWiring(
        vae_prefix="vae.",
        text_encoder_prefix="text_encoders.",
        text_encoders=("dinkster.qwen3_06b",),
    ),
    supported_dtypes=frozenset(ANIMA_CONFIG.inference_dtypes),
    memory_factor=ANIMA_CONFIG.memory_factor,
    engine=EngineProperties(preview_decoder=PreviewDecoderProperties("taehv")),
)


@dataclass(frozen=True)
class _TripoSplatDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        evidence = detect_triposplat(source)
        if evidence is None:
            return None
        return DetectionEvidence(
            family_id=TRIPOSPLAT_FAMILY.id,
            matched_keys=evidence.matched_keys,
            fields=evidence.fields,
        )


TRIPOSPLAT = ModelFamily(
    id=TRIPOSPLAT_FAMILY.id,
    display_name=TRIPOSPLAT_FAMILY.display_name,
    detector=_TripoSplatDetector(),
    specificity=100,
    # The DiT jointly denoises a fixed 8192-token shape-code sequence
    # and a single 5-channel camera token (comfy/ldm/triposplat/model.py
    # @ 36408117); both are 1-dimensional token streams, not spatial
    # grids.
    latent=MultiStreamLatentDescriptor(
        (
            (
                "latent",
                LatentDescriptor(
                    channels=TRIPOSPLAT_CONFIG.latent_channels,
                    dimensions=1,
                ),
            ),
            (
                "camera",
                LatentDescriptor(
                    channels=TRIPOSPLAT_CONFIG.cam_channels,
                    dimensions=1,
                ),
            ),
        )
    ),
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=TRIPOSPLAT_SIGMAS.sigma_min,
        sigma_max=TRIPOSPLAT_SIGMAS.sigma_max,
        shift=TRIPOSPLAT_CONFIG.sampling_shift,
    ),
    # TripoSplat ships as split artifacts rather than a combined
    # checkpoint; it conditions on vision tokens, not text.
    wiring=ComponentWiring(text_encoders=()),
    supported_dtypes=frozenset(TRIPOSPLAT_CONFIG.inference_dtypes),
    memory_factor=TRIPOSPLAT_CONFIG.memory_factor,
    aliases=TRIPOSPLAT_FAMILY.aliases,
    engine=EngineProperties(
        preview_decoder=PreviewDecoderProperties("asset", "triposplat_vae_decoder")
    ),
)


@dataclass(frozen=True)
class _Trellis2Detector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        prefixes = {
            "structure": "model.structure_model.",
            "shape": "model.img2shape.",
            "shape-512": "model.img2shape_512.",
            "texture": "model.shape2txt.",
        }
        try:
            configs = {
                role: detect_trellis2_flow(
                    {
                        key.removeprefix(prefix): source.entry(key).geometry
                        for key in source.keys()
                        if key.startswith(prefix)
                    }
                )
                for role, prefix in prefixes.items()
            }
        except ValueError:
            return None
        if configs["structure"].stage != "structure":
            return None
        if any(configs[role].stage != "shape" for role in ("shape", "shape-512")):
            return None
        if configs["texture"].stage != "texture":
            return None
        attentions = {config.image_attention for config in configs.values()}
        return DetectionEvidence(
            family_id=TRELLIS2_FAMILY_ID,
            matched_keys=tuple(source.keys()),
            fields={"image_attention": ",".join(sorted(attentions))},
        )


TRELLIS2_SIGMAS = FlowSigmas(shift=3.0)
TRELLIS2 = ModelFamily(
    id=TRELLIS2_FAMILY_ID,
    display_name="TRELLIS.2 / Pixal3D",
    detector=_Trellis2Detector(),
    specificity=100,
    latent=LatentDescriptor(channels=32, dimensions=3, spatial_downscale=16),
    sampling=SamplingDescriptor(
        parameterization=Parameterization.FLOW,
        sigma_min=TRELLIS2_SIGMAS.sigma_min,
        sigma_max=TRELLIS2_SIGMAS.sigma_max,
        shift=3.0,
    ),
    wiring=ComponentWiring(text_encoders=()),
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    memory_factor=6.0,
    aliases=("trellis2", "trellis.2", "pixal3d"),
)


def builtin_families() -> tuple[ModelFamily, ...]:
    """The grounded catalog, in registration order."""
    return (
        SD15,
        SDXL,
        SDXL_REFINER,
        CHROMA,
        CHROMA_RADIANCE,
        FLUX_DEV,
        FLUX_SCHNELL,
        FLUX2_DEV,
        FLUX2_KLEIN_9B,
        FLUX2_KLEIN_4B,
        WAN21,
        WAN22,
        LTXV,
        LTXAV,
        QWEN_IMAGE,
        Z_IMAGE,
        Z_IMAGE_PIXEL_SPACE,
        MINIMAX_H3,
        MINIMAX_MUSIC3,
        KREA2,
        IDEOGRAM4,
        SEEDVR2,
        ANIMA,
        LUMINA2,
        TRIPOSPLAT,
        TRELLIS2,
    )


def builtin_family_registry() -> FamilyRegistry:
    """A fresh registry preloaded with the grounded catalog."""
    registry = FamilyRegistry()
    for family in builtin_families():
        registry.register(family)
    return registry


__all__ = [
    "ANIMA",
    "CHROMA",
    "CHROMA_RADIANCE",
    "FLUX_DEV",
    "FLUX_SCHNELL",
    "FLUX2_DEV",
    "FLUX2_KLEIN_4B",
    "FLUX2_KLEIN_9B",
    "IDEOGRAM4",
    "KREA2",
    "MINIMAX_H3",
    "MINIMAX_MUSIC3",
    "LTXAV",
    "LTXV",
    "LUMINA2",
    "QWEN_IMAGE",
    "SD15",
    "SDXL",
    "SDXL_REFINER",
    "SDXL_VARIANT_MARKERS",
    "SEEDVR2",
    "TRIPOSPLAT",
    "TRELLIS2",
    "TRELLIS2_SIGMAS",
    "WAN21",
    "WAN22",
    "Z_IMAGE",
    "Z_IMAGE_PIXEL_SPACE",
    "builtin_families",
    "builtin_family_registry",
]
