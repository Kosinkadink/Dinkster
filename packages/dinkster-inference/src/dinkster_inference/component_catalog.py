"""Native component registrations and adapters for architecture planners."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

from . import catalog
from .assembly import AssemblyError
from .component_registry import ComponentDescriptor, ComponentRegistry
from .devices import BFLOAT16, FLOAT16, FLOAT32
from .families import ModelFamily
from .quantization import QuantizationError, quantization_error_cause
from .weights import WeightSource


def _split_detector(
    planner_name: str,
    roles: tuple[str, ...],
    family: ModelFamily,
    *,
    require_family: bool = False,
) -> Callable[..., tuple[tuple[str, Any], ...]]:
    def detect(
        source: WeightSource, path: Path, *, bind_asset_identity: bool = True
    ) -> tuple[tuple[str, Any], ...]:
        planner = getattr(importlib.import_module("dinkster_inference"), planner_name)
        components: list[tuple[str, Any]] = []
        diagnostics: list[str] = []
        quantization_error: QuantizationError | None = None
        for role in roles:
            try:
                planned = (
                    planner(source, role=role, path=path)
                    if bind_asset_identity
                    else planner(source, role=role, path=path, bind_asset_identity=False)
                )
            except ValueError as error:
                diagnostics.append(str(error))
                quantization_error = quantization_error or quantization_error_cause(error)
                continue
            if (
                require_family
                and (bind_asset_identity or role == "diffusion")
                and getattr(planned, "family_id", family.id) != family.id
            ):
                continue
            components.append((role, planned))
        if not components and diagnostics and family.detector.detect(source) is not None:
            logging.getLogger(__name__).warning(
                "Detected %s; component planning diagnostics: %s",
                family.display_name,
                "; ".join(diagnostics),
            )
        if not components and quantization_error is not None:
            raise quantization_error
        if not components and diagnostics and not bind_asset_identity:
            raise AssemblyError("; ".join(diagnostics))
        return tuple(components)

    return detect


def _lumina2_detector(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, Any], ...]:
    from .assembly import lumina2_tokenizer_source_key
    from .lumina2_component import (
        Lumina2CheckpointText,
        Lumina2ComponentAssemblyError,
        plan_lumina2_artifact_components,
    )

    try:
        components = plan_lumina2_artifact_components(
            source, path=path, bind_asset_identity=bind_asset_identity
        )
    except Lumina2ComponentAssemblyError as error:
        if not bind_asset_identity and catalog.LUMINA2.detector.detect(source) is not None:
            raise AssemblyError(str(error.__cause__ or error)) from error
        return ()
    return tuple(
        (
            role,
            Lumina2CheckpointText(planned, lumina2_tokenizer_source_key(source))
            if role == "gemma2_2b" and not bind_asset_identity
            else planned,
        )
        for role, planned in components
    )


def _chroma_plan_family(planned: Any) -> ModelFamily:
    if planned.component == "diffusion" and planned.config.family_id == catalog.CHROMA_RADIANCE.id:
        return catalog.CHROMA_RADIANCE
    return catalog.CHROMA


def _chroma_execution_options(
    runtime: Any, sampling_shift: float | None, option_windows: tuple[Any, ...]
) -> dict[str, Any]:
    return {
        "sampling_shift": 1.0 if sampling_shift is None else sampling_shift,
        "option_windows": option_windows,
        "attention_status": runtime.attention_status,
    }


def _descriptor(
    family: ModelFamily,
    planner: str,
    loader: str,
    runtime: str,
    text: tuple[str, ...] = (),
    codecs: tuple[str, ...] = (),
    *,
    additional_roles: tuple[str, ...] = (),
    **behavior: Any,
) -> ComponentDescriptor:
    roles = (behavior.get("model_role", "diffusion"), *text, *codecs, *additional_roles)
    behavior.setdefault("native_load", _NATIVE_COMPONENT_LOAD)
    if codecs:
        behavior.setdefault("native_decode", _NATIVE_COMPONENT_DECODE)
        behavior.setdefault("native_encode", _NATIVE_COMPONENT_ENCODE)
    return ComponentDescriptor(
        family=family,
        detector=_split_detector(planner, roles, family),
        roles=roles,
        text_encoder_roles=text,
        codec_roles=codecs,
        loader=loader,
        runtime_class=runtime,
        **behavior,
    )


_NATIVE_COMPONENT_LOAD = "dinkster_native.family_registry:load_component"
_NATIVE_COMPONENT_DECODE = "dinkster_native.family_registry:decode_component"
_NATIVE_COMPONENT_ENCODE = "dinkster_native.family_registry:encode_component"


@cache
def default_component_registry() -> ComponentRegistry:
    """The worker and host use the same architecture registrations."""
    from .ace15_text import ace15_component_descriptor
    from .control_components import control_component_descriptors
    from .hunyuan_image_text import plan_hunyuan_image_qwen
    from .ideogram4_component import ideogram4_component_uses_fp8_matmul
    from .llama3_text import detect_llama3_component
    from .ltxav_component import ltxav_component_uses_fp8_matmul
    from .lumina2_component import lumina2_checkpoint_assembly
    from .minimax_h3_component_descriptor import H3ComponentDescriptor, detect_h3_components
    from .newbie_text import plan_newbie_component
    from .qwen_image_assembly import qwen_image_checkpoint_assembly
    from .text_components import plan_text_components
    from .wan21_component import wan_checkpoint_assembly

    registry = ComponentRegistry()
    wan_roles = ("diffusion", "umt5xxl", "clip_vision", "vae")
    for descriptor in (
        ComponentDescriptor(
            family=replace(
                catalog.FLUX_DEV,
                id="dinkster.classic_text",
                display_name="CLIP and T5 text encoders",
                specificity=0,
                aliases=(),
            ),
            detector=plan_text_components,
            roles=("clip_l", "clip_g", "t5xxl", "umt5xxl", "byt5_small"),
            text_encoder_roles=("clip_l", "clip_g", "t5xxl", "umt5xxl", "byt5_small"),
            codec_roles=(),
            loader="dinkster_inference_torch.text_recipes:assemble_text_recipe",
            runtime_class="dinkster_inference_torch.text_recipes:TextRecipeRuntime",
            requires_text_recipe=True,
        ),
        ComponentDescriptor(
            family=replace(
                catalog.FLUX_DEV,
                id="dinkster.hunyuan_video",
                display_name="Hunyuan Video Llama3 text encoder",
                specificity=0,
                aliases=(),
            ),
            detector=detect_llama3_component,
            roles=("llama",),
            text_encoder_roles=("llama",),
            codec_roles=(),
            loader="dinkster_inference_torch.hunyuan_video_text:assemble_hunyuan_video_text",
            runtime_class="dinkster_inference_torch.hunyuan_video_text:HunyuanVideoTextRuntime",
            requires_text_recipe=True,
        ),
        ComponentDescriptor(
            family=replace(
                catalog.FLUX_DEV,
                id="dinkster.hunyuan_image",
                display_name="Hunyuan Image Qwen text encoder",
                specificity=0,
                aliases=(),
            ),
            detector=plan_hunyuan_image_qwen,
            roles=("qwen25_vl",),
            text_encoder_roles=("qwen25_vl",),
            codec_roles=(),
            loader="dinkster_inference_torch.hunyuan_image_text:assemble_hunyuan_image_text",
            runtime_class="dinkster_inference_torch.hunyuan_image_text:HunyuanImageTextRuntime",
            requires_text_recipe=True,
        ),
        ComponentDescriptor(
            family=replace(
                catalog.FLUX_DEV,
                id="dinkster.newbie",
                display_name="NewBie text encoders",
                specificity=0,
                aliases=(),
            ),
            detector=plan_newbie_component,
            roles=("gemma", "jina"),
            text_encoder_roles=("gemma", "jina"),
            codec_roles=(),
            loader="dinkster_inference_torch.newbie_text:assemble_newbie_text",
            runtime_class="dinkster_inference_torch.newbie_text:NewBieTextRuntime",
            requires_text_recipe=True,
        ),
        ace15_component_descriptor(
            replace(
                catalog.FLUX_DEV,
                id="dinkster.ace_step_1_5",
                display_name="ACE-Step 1.5 text encoders",
                specificity=0,
                aliases=(),
            )
        ),
        *control_component_descriptors(),
        H3ComponentDescriptor(
            catalog.MINIMAX_H3,
            detect_h3_components,
            ("diffusion", "qwen3vl-32b-conditioner", "video-vae", "audio-vae"),
            ("qwen3vl-32b-conditioner",),
            ("video-vae", "audio-vae"),
            "dinkster_inference_torch.component_runtime:load_h3_component",
            "MiniMaxH3Model",
            vae_dtypes=(FLOAT16, FLOAT32),
            role_default_dtypes=(("audio-vae", FLOAT32),),
            text_loader_hints=("minimax",),
            requires_runtime_versions=True,
            pool_model=True,
            aimdo_roles=("qwen3vl-32b-conditioner", "video-vae", "audio-vae"),
            runtime_factory="dinkster_inference_torch.component_runtime:h3_runtime",
            conditioning_format="carrier",
            execution_resolver="dinkster_native.native_arm:resolve_minimax_h3_component_execution",
            native_load="dinkster_native.families.minimax_h3:load_component",
            codec_adapter="dinkster_native.families.minimax_h3:CodecAdapter",
            native_decode=_NATIVE_COMPONENT_DECODE,
            native_encode=_NATIVE_COMPONENT_ENCODE,
            default_text_dtype=FLOAT16,
        ),
        _descriptor(
            catalog.ANIMA,
            "plan_anima_split_component",
            "load_anima_component",
            "AnimaDiffusionRuntime",
            ("qwen3_06b",),
            default_text_dtype=FLOAT32,
            vae_dtypes=(BFLOAT16, FLOAT16, FLOAT32),
            prepare_conditioning="materialize_anima_conditioning",
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer="dinkster_inference_torch.anima_component:realize_anima_component",
            checkpoint_text_factory="dinkster_inference_torch.anima_runtime:checkpoint_text_runtime",
            native_encode_text="dinkster_native.families.anima:encode_text",
            native_load="dinkster_native.families.anima:load_component",
        ),
        ComponentDescriptor(
            catalog.LUMINA2,
            _lumina2_detector,
            ("diffusion", "gemma2_2b", "vae"),
            ("gemma2_2b",),
            ("vae",),
            "load_lumina2_component",
            "Lumina2DiffusionRuntime",
            default_text_dtype=FLOAT32,
            checkpoint_loader="dinkster_inference_torch.wiring:_load_lumina2",
            checkpoint_validator=lumina2_checkpoint_assembly,
            codec_adapter="dinkster_native.families.lumina2:CodecAdapter",
            native_encode_text="dinkster_native.families.lumina2:encode_text",
            native_decode=_NATIVE_COMPONENT_DECODE,
            native_encode=_NATIVE_COMPONENT_ENCODE,
            native_load=_NATIVE_COMPONENT_LOAD,
        ),
        _descriptor(
            catalog.KREA2,
            "plan_krea2_split_component",
            "load_krea2_component",
            "Krea2DiffusionRuntime",
            ("qwen3vl_4b",),
            default_text_dtype=FLOAT32,
            vae_dtypes=(BFLOAT16, FLOAT16, FLOAT32),
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer="dinkster_inference_torch.krea2_component:realize_krea2_component",
            checkpoint_text_factory="dinkster_inference_torch.krea2_runtime:checkpoint_text_runtime",
            native_encode_text="dinkster_native.families.krea2:encode_text",
        ),
        _descriptor(
            catalog.IDEOGRAM4,
            "plan_ideogram4_split_component",
            "load_ideogram4_component",
            "Ideogram4DiffusionRuntime",
            ("qwen3vl_8b",),
            default_text_dtype=FLOAT32,
            fp8_matmul=ideogram4_component_uses_fp8_matmul,
            execution_resolver="dinkster_native.native_arm:resolve_ideogram4_component_execution",
            native_encode_text="dinkster_native.families.ideogram4:encode_text",
        ),
        _descriptor(
            catalog.SEEDVR2,
            "plan_seedvr2_split_component",
            "load_seedvr2_component",
            "SeedVR2DiffusionRuntime",
            codecs=("vae",),
            vae_dtypes=(FLOAT16, BFLOAT16, FLOAT32),
            execution_resolver="dinkster_native.native_arm:resolve_seedvr2_component_execution",
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer="dinkster_inference_torch.seedvr2_component:realize_seedvr2_component",
            checkpoint_codec_factory="dinkster_inference_torch.seedvr2_runtime:checkpoint_codec",
            codec_adapter="dinkster_native.families.seedvr2:CodecAdapter",
            native_load="dinkster_native.families.seedvr2:load_component",
        ),
        _descriptor(
            catalog.CHROMA,
            "plan_chroma_split_component",
            "load_chroma_component",
            "ChromaDiffusionRuntime",
            ("t5xxl",),
            ("vae",),
            default_text_dtype=FLOAT32,
            aliases=("dinkster.chroma_radiance",),
            aimdo_roles=("diffusion", "t5xxl", "vae"),
            fixed_promotion_roles=("diffusion",),
            plan_family=_chroma_plan_family,
            runtime_factory="dinkster_inference_torch.component_runtime:chroma_runtime",
            execution_options=_chroma_execution_options,
            shared_conditioning_families=(catalog.CHROMA.id,),
            prepare_conditioning="prepare_single_stream_conditioning",
            release_conditioning=True,
            codec_adapter="dinkster_native.families.chroma:CodecAdapter",
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer="dinkster_inference_torch.chroma_component:realize_chroma_component",
            checkpoint_text_factory="dinkster_inference_torch.chroma_component:checkpoint_text_runtime",
            checkpoint_codec_factory="dinkster_inference_torch.chroma_component:checkpoint_codec",
            native_encode_text="dinkster_native.families.chroma:encode_text",
        ),
        _descriptor(
            catalog.MINIMAX_MUSIC3,
            "plan_minimax_music3_split_component",
            "load_minimax_music3_component",
            "MiniMaxMusic3DiffusionRuntime",
            ("text",),
            ("vae",),
            vae_dtypes=(FLOAT32,),
            text_loader_hints=("minimax",),
            tokenizer_attribute="_dinkster_minimax_music3_tokenizer",
            prepare_conditioning="materialize_minimax_music3_conditioning",
            codec_adapter="dinkster_native.families.minimax_music3:CodecAdapter",
            native_load="dinkster_native.families.minimax_music3:load_component",
        ),
        _descriptor(
            catalog.QWEN_IMAGE,
            "plan_qwen_image_official_component",
            "load_qwen_image_component",
            "QwenImageDiffusionRuntime",
            ("qwen2_5_vl_7b",),
            ("vae",),
            vae_dtypes=(BFLOAT16, FLOAT16, FLOAT32),
            runtime_with_family=True,
            prepare_conditioning="prepare_single_stream_conditioning",
            checkpoint_loader="dinkster_inference_torch.wiring:_load_qwen_image",
            checkpoint_validator=qwen_image_checkpoint_assembly,
            codec_adapter="dinkster_native.families.qwen_image:CodecAdapter",
            native_encode_text="dinkster_native.families.qwen_image:encode_text",
        ),
        ComponentDescriptor(
            catalog.WAN21,
            _split_detector("plan_wan21_split_component", wan_roles, catalog.WAN21),
            wan_roles,
            ("umt5xxl",),
            ("vae",),
            "load_wan21_component",
            "Wan21DiffusionRuntime",
            default_text_dtype=FLOAT32,
            text_loader_hints=("wan",),
            vae_dtypes=(BFLOAT16, FLOAT16, FLOAT32),
            runtime_with_family=True,
            runtime_factory="dinkster_inference_torch.component_runtime:wan_runtime",
            runtime_variants=("Wan21CausalDiffusionRuntime",),
            prepare_conditioning="prepare_conditioning",
            conditioning_format="multistream",
            checkpoint_loader="dinkster_inference_torch.wiring:_load_wan",
            checkpoint_validator=wan_checkpoint_assembly,
            checkpoint_source_aliases=(("t5xxl", "umt5xxl"),),
            codec_adapter="dinkster_native.families.wan21:CodecAdapter",
            native_encode_text="dinkster_native.families.wan21:encode_text",
            native_decode=_NATIVE_COMPONENT_DECODE,
            native_encode=_NATIVE_COMPONENT_ENCODE,
            native_load=_NATIVE_COMPONENT_LOAD,
        ),
        _descriptor(
            catalog.LTXV,
            "plan_ltxv_split_component",
            "load_ltxv_component",
            "LTXVDiffusionRuntime",
            ("t5xxl",),
            ("vae",),
            prepare_conditioning="prepare_conditioning",
            conditioning_format="multistream",
            frame_rate_conditioning=True,
            allow_unbound_conditioning=True,
            codec_adapter="dinkster_native.families.ltx:CodecAdapter",
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer="dinkster_inference_torch.ltx_component:realize_ltxv_component",
            checkpoint_text_factory="dinkster_inference_torch.ltx_component:checkpoint_text_runtime",
            checkpoint_codec_factory="dinkster_inference_torch.ltxv_runtime:checkpoint_codec",
            native_encode_text="dinkster_native.families.ltxv:encode_text",
        ),
        _descriptor(
            catalog.LTXAV,
            "plan_ltxav_split_component",
            "load_ltxav_component",
            "LTXAVDiffusionRuntime",
            ("gemma3_12b", "gemma4_12b", "text_projection", "connectors"),
            ("vae", "latent_upscaler"),
            additional_roles=("duration_head",),
            fp8_matmul=ltxav_component_uses_fp8_matmul,
            conditioning_roles=("text",),
            prepare_conditioning="prepare_conditioning",
            conditioning_format="multistream",
            frame_rate_conditioning=True,
            allow_unbound_conditioning=True,
            codec_adapter="dinkster_native.families.ltx:CodecAdapter",
        ),
        _descriptor(
            catalog.TRIPOSPLAT,
            "plan_triposplat_split_component",
            "load_triposplat_component",
            "TripoSplatDiffusionRuntime",
            ("dinov3-vision-conditioner",),
            ("gaussian-decoder",),
            vae_dtypes=(FLOAT16, BFLOAT16, FLOAT32),
            model_role="dit",
            prepare_conditioning="prepare_conditioning",
            conditioning_format="raw",
        ),
        _descriptor(
            catalog.TRELLIS2,
            "plan_trellis2_artifact",
            "load_trellis2_artifact",
            "Trellis2DiffusionRuntime",
            ("vision",),
            ("structure-decoder", "shape-decoder", "texture-decoder"),
            canonicalize_runtime_facts=True,
            runtime_factory="dinkster_inference_torch.component_runtime:trellis_runtime",
            execution_resolver="dinkster_native.native_arm:resolve_trellis2_component_execution",
        ),
    ):
        registry.register(descriptor)
    from .assembly import FLUX2_SHARED_COMPONENT_FAMILY_ID
    from .flux2_assembly import FLUX2_TEXT_COMPONENT_ROLES, flux2_checkpoint_assembly

    for family in (
        catalog.FLUX2_DEV,
        catalog.FLUX2_KLEIN_4B,
        catalog.FLUX2_KLEIN_9B,
        replace(
            catalog.FLUX2_DEV,
            id=FLUX2_SHARED_COMPONENT_FAMILY_ID,
            wiring=replace(catalog.FLUX2_DEV.wiring, text_encoders=()),
        ),
    ):
        text_roles = tuple(role.removeprefix("dinkster.") for role in family.wiring.text_encoders)
        roles = ("diffusion", *text_roles, "vae")
        registry.register(
            ComponentDescriptor(
                family,
                _split_detector("plan_flux2_split_component", roles, family, require_family=True),
                roles,
                text_roles,
                ("vae",),
                "load_flux2_component",
                "Flux2DiffusionRuntime",
                text_loader_hints=("flux2",),
                runtime_with_family=True,
                aimdo_roles=FLUX2_TEXT_COMPONENT_ROLES,
                prepare_conditioning="prepare_flux2_conditioning",
                checkpoint_loader="dinkster_inference_torch.wiring:_load_flux2"
                if text_roles
                else None,
                checkpoint_validator=flux2_checkpoint_assembly,
                codec_adapter="dinkster_native.families.flux2:CodecAdapter",
                native_encode_text=(
                    "dinkster_native.families.flux2:encode_text" if text_roles else None
                ),
                native_decode=_NATIVE_COMPONENT_DECODE,
                native_encode=_NATIVE_COMPONENT_ENCODE,
                native_load="dinkster_native.families.flux2:load_component",
            )
        )
    return registry
