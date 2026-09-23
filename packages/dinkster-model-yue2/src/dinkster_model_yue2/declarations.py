"""Pack-owned YuE2 detection and inference declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dinkster_api.v1 import (
    FLOAT32,
    AssemblyRegistration,
    ComponentDescriptor,
    ComponentWiring,
    DetectionEvidence,
    EngineProperties,
    InferenceContribution,
    LatentDescriptor,
    ModelFamily,
    Parameterization,
    SamplingDescriptor,
)
from dinkster_inference import BFLOAT16

FAMILY_ID = "dinkster.yue2"
FRAMES_PER_SECOND = 25
LATENT_CHANNELS = 64
LATENT_DOWNSCALE = 1920
SAMPLE_RATE = 48_000

_SIGNATURES = (
    "model.diffusion_model.latent_pos_embed.pe",
    "model.diffusion_model.vae2llm.weight",
    "model.diffusion_model.llm2vae.weight",
    "text_encoders.model.embed_tokens.weight",
    "text_encoders.model.lm_head.weight",
    "text_encoders.yue2_tokenizer_json",
    "vae.decoder.layers.6.layers.1.weight_v",
)


@dataclass(frozen=True)
class YuE2Detector:
    def detect(self, source: Any) -> DetectionEvidence | None:
        keys = source.keys()
        if not all(key in keys for key in _SIGNATURES):
            return None
        position = source.entry(_SIGNATURES[0]).geometry
        projection = source.entry(_SIGNATURES[1]).geometry
        if position.shape != (24_576, 2_048) or projection.shape[1] != LATENT_CHANNELS:
            return None
        return DetectionEvidence(
            FAMILY_ID,
            _SIGNATURES,
            {
                "context": position.shape[0],
                "hidden": position.shape[1],
                "latent_channels": projection.shape[1],
                "sample_rate": SAMPLE_RATE,
            },
        )


YUE2_FAMILY = ModelFamily(
    id=FAMILY_ID,
    display_name="YuE2",
    detector=YuE2Detector(),
    specificity=130,
    latent=LatentDescriptor(channels=LATENT_CHANNELS, scale_factor=1.0, shift_factor=0.0),
    sampling=SamplingDescriptor(Parameterization.FLOW, sigma_min=0.001, sigma_max=1.0),
    wiring=ComponentWiring(text_encoders=("text",)),
    supported_dtypes=frozenset({BFLOAT16, FLOAT32}),
    aliases=("yue2",),
    engine=EngineProperties(
        attention_backends=(("diffusion", "qwen"), ("text", "qwen")),
        attention_requires_route=True,
    ),
    denoiser="dinkster_model_yue2.runtime:yue2_denoiser",
    text_encoder="dinkster_model_yue2.runtime:checkpoint_text_runtime",
    latent_codec="dinkster_model_yue2.runtime:checkpoint_codec",
    loader="dinkster_model_yue2.runtime:realize_yue2_component",
)


def _detect_components(
    source: Any, path: object, **options: object
) -> tuple[tuple[str, object], ...]:
    from .planning import detect_components

    return detect_components(source, path, **options)


def _plan_assembly(**sources: object) -> object:
    from .planning import plan_assembly

    return plan_assembly(**sources)


YUE2_COMPONENT = ComponentDescriptor(
    family=YUE2_FAMILY,
    detector=_detect_components,
    roles=("diffusion", "text", "vae"),
    text_encoder_roles=("text",),
    codec_roles=("vae",),
    loader="dinkster_model_yue2.runtime:load_yue2_component",
    runtime_class="dinkster_model_yue2.runtime:YuE2DiffusionRuntime",
    vae_dtypes=(FLOAT32,),
    tokenizer_attribute="_dinkster_yue2_tokenizer",
    prepare_conditioning="materialize_yue2_conditioning",
    checkpoint_loader=("dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint"),
    component_realizer="dinkster_model_yue2.runtime:realize_yue2_component",
    checkpoint_text_factory="dinkster_model_yue2.runtime:checkpoint_text_runtime",
    checkpoint_codec_factory="dinkster_model_yue2.runtime:checkpoint_codec",
)

YUE2_ASSEMBLY = AssemblyRegistration(
    id=FAMILY_ID,
    plan=_plan_assembly,
    load="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
)


def register_inference() -> InferenceContribution:
    return InferenceContribution(
        families=(YUE2_FAMILY,),
        components=(YUE2_COMPONENT,),
        assemblies=(YUE2_ASSEMBLY,),
    )
