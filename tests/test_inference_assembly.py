"""Proving tests for torch-free Flux, SD, and Wan assembly planning.

Synthetic sources built from the exact detector layouts cover the
combined/split/mixed flows, per-component quantization scoping, key
normalization, and every documented refusal. Header-only tests against
the installed checkpoints prove the plans on real files. The synthetic
SDXL layouts (transformers-format CLIP-L in conditioner slot 0,
OpenCLIP-format CLIP-G in slot 1) were verified byte-exact against the
real sd_xl_base_1.0.safetensors header.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict, cast

import pytest
from dinkster_inference import (
    ANIMA,
    ANIMA_CONFIG,
    ANIMA_QWEN3_06B_CONFIG,
    ANIMA_QWEN_PREFIX,
    BFLOAT16,
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    FLOAT8_E8M0,
    FLOAT16,
    FLOAT32,
    FLUX2_DEV,
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B,
    FLUX2_KLEIN_9B_CONFIG,
    FLUX2_MISTRAL_PREFIX,
    FLUX_CLIP_L_PREFIX,
    FLUX_DEV,
    FLUX_DEV_CONFIG,
    FLUX_DIFFUSION_PREFIX,
    FLUX_QWEN_PREFIX,
    FLUX_SCHNELL,
    FLUX_SCHNELL_CONFIG,
    FLUX_T5XXL_PREFIX,
    FLUX_VAE_PREFIX,
    GEMMA3_LTX_12B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    INT8,
    INT32,
    INT64,
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    KREA2,
    KREA2_CONFIG,
    KREA2_TEXT_CONFIG,
    KREA2_TEXT_PREFIX,
    LTX_LATENT_UPSAMPLER_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    LTX_TEXT_STACK_FEATURES,
    LTXAV,
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_CONFIG,
    LTXAV_19B_VAE_CONFIG,
    LTXAV_19B_VOCODER_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V23_VAE_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_22B_V25_VAE_CONFIG,
    LTXAV_BWE_VOCODER_CONFIG,
    LTXAV_CONNECTOR_PREFIXES,
    LTXAV_DURATION_HEAD_CONFIG,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    LTXV_2B_V095_CONFIG,
    LTXV_2B_V095_VAE_CONFIG,
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    OVIS_QWEN3_2B_CONFIG,
    QWEN_IMAGE,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    QWEN_IMAGE_TEXT_CONFIG,
    SD15,
    SD15_CLIP_L_PREFIX,
    SD15_UNET_CONFIG,
    SD_DIFFUSION_PREFIX,
    SD_VAE_PREFIX,
    SDXL,
    SDXL_CLIP_G_PREFIX,
    SDXL_CLIP_L_PREFIX,
    SDXL_INPAINT_UNET_CONFIG,
    SDXL_REFINER,
    SDXL_REFINER_CLIP_G_PREFIX,
    SDXL_REFINER_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    T5_XXL_CONFIG,
    UINT8,
    UMT5_XXL_CONFIG,
    WAN21,
    WAN21_ANIMATE2_14B,
    WAN21_CAMERA_1_3B,
    WAN21_CAMERA_14B,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_DIFFUSION_PREFIX,
    WAN21_FLF_I2V_14B,
    WAN21_FLOW_RVS_1_3B,
    WAN21_FLOW_RVS_VAE_CONFIG,
    WAN21_FUN_CONTROL_1_3B,
    WAN21_FUN_INPAINT_1_3B,
    WAN21_HUMO_17B,
    WAN21_I2V_14B,
    WAN21_SCAIL2_14B,
    WAN21_SCAIL_14B,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_UMT5_PREFIX,
    WAN21_VACE_1_3B,
    WAN21_VACE_14B,
    WAN21_VAE_CONFIG,
    WAN21_VAE_PREFIX,
    WAN22,
    WAN22_ANIMATE_14B,
    WAN22_BERNINI_14B,
    WAN22_CAMERA_14B,
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_CONTROL_14B,
    WAN22_FUN_INPAINT_5B,
    WAN22_I2V_14B,
    WAN22_TI2V_5B,
    WAN22_VAE_CONFIG,
    WAN22_WANDANCER_14B,
    Z_IMAGE_QWEN3_4B_CONFIG,
    AssemblyError,
    ComponentPlan,
    DType,
    FamilyRegistry,
    Flux2AssemblyPlan,
    FluxAssemblyPlan,
    LinearToConv2D,
    LTXAVAudioCodecPlan,
    LTXAVConfig,
    LTXVConfig,
    LTXVideoVAEConfig,
    NativeRefusalCategory,
    NativeRefusalError,
    Parameterization,
    QwenImageAssemblyPlan,
    QwenImageConfig,
    RowChunk,
    SafetensorsSource,
    SamplingDescriptor,
    SamplingSpace,
    SDAssemblyPlan,
    T5Config,
    TAESDCodecPlan,
    TAESDDetectError,
    TensorGeometry,
    Transpose2D,
    UNetConfig,
    Wan21StandaloneComponentPlan,
    Wan21VAEConfig,
    WeightEntry,
    WeightSource,
    anima_layout,
    build_runtime_identity,
    clip_text_layout,
    clip_vision_layout,
    flux2_layout,
    flux_layout,
    gemma_text_layout,
    identify_ltxav_text_source,
    krea2_language_layout,
    krea2_layout,
    krea2_text_layout,
    load_safetensors_header,
    ltx_audio_vae_layout,
    ltx_connector_layout,
    ltx_diffusion_video_vae_layout,
    ltx_latent_upsampler_layout,
    ltx_vocoder_bwe_layout,
    ltx_vocoder_layout,
    ltxav_duration_head_layout,
    ltxav_layout,
    ltxv_layout,
    ltxv_vae_layout,
    openclip_text_layout,
    plan_anima_component,
    plan_flux2_assembly,
    plan_flux2_component,
    plan_flux_assembly,
    plan_krea2_component,
    plan_ltxav_audio_codec,
    plan_ltxav_standalone_component,
    plan_ltxv_standalone_component,
    plan_native,
    plan_qwen_image_assembly,
    plan_sd_assembly,
    plan_taesd_decoder,
    plan_wan21_assembly,
    plan_wan21_standalone_component,
    plan_wan22_assembly,
    probe_native,
    qwen_image_dit_layout,
    qwen_image_text_layout,
    qwen_text_layout,
    runtime_component_identity,
    t5_layout,
    taesd_layout,
    unet_layout,
    wan21_vae_layout,
    wan22_vae_layout,
)

from tests.test_inference_kl import diffusers_geometries
from tests.test_inference_kl import kl_geometries as synthetic_kl_geometries
from tests.test_inference_wan21 import wan21_shapes
from tests.test_inference_wan22 import wan22_shapes

MODELS = Path("/home/kosin/ComfyUI/models")
REAL_COMBINED_FP8 = MODELS / "diffusion_models/flux1-dev-fp8.safetensors"
REAL_SPLIT_DIT_FP8 = MODELS / "diffusion_models/flux1-dev-fp8-new.safetensors"
REAL_FLUX2 = MODELS / "diffusion_models/flux2_dev_fp8mixed.safetensors"
REAL_CLIP_L = MODELS / "text_encoders/clip_l.safetensors"
REAL_T5_FP16 = MODELS / "text_encoders/t5xxl_fp16.safetensors"
REAL_UMT5 = MODELS / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
REAL_AE = MODELS / "vae/ae.safetensors"
REAL_WAN22_BERNINI = Path(
    os.environ.get(
        "DINKSTER_WAN22_BERNINI",
        "/home/kosin/ComfyUI/models/diffusion_models/"
        "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors",
    )
)
WAN22_BERNINI_REVISION = "fc371005c90d24177f3658cfacd78b44a41bbd8e"
WAN22_BERNINI_ARTIFACT = (
    15_574_833_216,
    "9ff3d7369da98f8eaf71045f7d0e99d5e344eea5e7dc934a930608786fe73f52",
    "https://huggingface.co/Comfy-Org/Bernini-R/resolve/"
    f"{WAN22_BERNINI_REVISION}/diffusion_models/"
    "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors",
)

OVIS_ARTIFACT_ROOT = Path("/home/kosin/model-artifacts/dinkster-w0-flux-vector-free")
REAL_OVIS_DIFFUSION = OVIS_ARTIFACT_ROOT / "diffusion_models/ovis_image_bf16.safetensors"
REAL_OVIS_TEXT = OVIS_ARTIFACT_ROOT / "text_encoders/ovis_2.5.safetensors"
REAL_OVIS_AE = OVIS_ARTIFACT_ROOT / "vae/ae.safetensors"

REAL_SPLIT_SET = (REAL_SPLIT_DIT_FP8, REAL_CLIP_L, REAL_T5_FP16, REAL_AE)
REAL_OVIS_SET = (REAL_OVIS_DIFFUSION, REAL_OVIS_TEXT, REAL_OVIS_AE)

KL_GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "kl_goldens.json"
)


@dataclass
class FakeSource:
    """In-memory WeightSource with the path the planner requires."""

    path: Path
    geometries: dict[str, TensorGeometry]
    extra: dict[str, str] = field(default_factory=dict)
    scalar_values: dict[str, float] = field(default_factory=dict)
    payload_values: dict[str, bytes] = field(default_factory=dict)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return self.extra

    def read_float_scalar(self, key: str) -> float:
        return self.scalar_values[key]

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.payload_values[key]


@dataclass
class CountingSource(FakeSource):
    scalar_reads: int = 0

    def read_float_scalar(self, key: str) -> float:
        self.scalar_reads += 1
        return super().read_float_scalar(key)


@dataclass
class HeaderOnlySource:
    path: Path
    geometries: dict[str, TensorGeometry]

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}


@dataclass
class PathlessSource(FakeSource):
    path: None = None  # type: ignore[assignment]
    geometries: dict[str, TensorGeometry] = field(default_factory=dict)


def g(shape: tuple[int, ...], dtype: DType = BFLOAT16) -> TensorGeometry:
    return TensorGeometry(shape, dtype)


def geometrize(
    layout: Mapping[str, tuple[int, ...]], dtype: DType = BFLOAT16
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}


def prefixed(sd: dict[str, TensorGeometry], prefix: str) -> dict[str, TensorGeometry]:
    return {prefix + key: value for key, value in sd.items()}


def flux_geometries(config=FLUX_DEV_CONFIG, dtype: DType = BFLOAT16) -> dict[str, TensorGeometry]:
    return geometrize(flux_layout(config), dtype)


def clip_geometries(*, projection: bool = True) -> dict[str, TensorGeometry]:
    sd = geometrize(clip_text_layout(CLIP_L_TEXT_CONFIG), FLOAT16)
    if not projection:
        del sd["text_projection.weight"]
    return sd


def t5_geometries(*, alias: bool = False) -> dict[str, TensorGeometry]:
    sd = geometrize(t5_layout(T5_XXL_CONFIG), FLOAT16)
    if alias:
        sd["encoder.embed_tokens.weight"] = sd["shared.weight"]
    return sd


def kl_geometries() -> dict[str, TensorGeometry]:
    payload = json.loads(KL_GOLDENS.read_text())
    return {
        key: TensorGeometry(tuple(shape), FLOAT32)
        for key, shape in payload["cases"]["standard"]["state_dict"]
    }


def source(
    sd: dict[str, TensorGeometry],
    name: str = "fake.safetensors",
    *,
    scalar_values: dict[str, float] | None = None,
    payload_values: dict[str, bytes] | None = None,
) -> FakeSource:
    return FakeSource(
        Path(f"/fake/{name}"),
        sd,
        scalar_values=scalar_values or {},
        payload_values=payload_values or {},
    )


def combined_geometries(dit: dict[str, TensorGeometry] | None = None) -> dict[str, TensorGeometry]:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(dit if dit is not None else flux_geometries(), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(clip_geometries(), FLUX_CLIP_L_PREFIX))
    sd["text_encoders.clip_l.logit_scale"] = g((), FLOAT32)
    sd.update(prefixed(t5_geometries(), FLUX_T5XXL_PREFIX))
    sd["text_encoders.t5xxl.logit_scale"] = g((), FLOAT32)
    sd.update(prefixed(kl_geometries(), FLUX_VAE_PREFIX))
    return sd


def split_sources() -> dict[str, WeightSource]:
    return {
        "diffusion": source(flux_geometries(), "dit.safetensors"),
        "clip_l": source(clip_geometries(projection=False), "clip_l.safetensors"),
        "t5xxl": source(t5_geometries(alias=True), "t5.safetensors"),
        "vae": source(kl_geometries(), "ae.safetensors"),
    }


def wan21_vae_geometries(
    config: Wan21VAEConfig = WAN21_VAE_CONFIG,
) -> dict[str, TensorGeometry]:
    return geometrize(wan21_vae_layout(config), FLOAT32)


@pytest.mark.parametrize(
    "config",
    (QWEN_IMAGE_CONFIG, QWEN_IMAGE_EDIT_2511_CONFIG, QWEN_IMAGE_LAYERED_CONFIG),
)
def test_qwen_image_split_plan_admits_each_semantic_variant(config: QwenImageConfig) -> None:
    sources: dict[str, WeightSource] = {
        "diffusion": source(
            geometrize(qwen_image_dit_layout(config).keys, BFLOAT16),
            "qwen_image.safetensors",
        ),
        "qwen2_5_vl_7b": source(
            geometrize(qwen_image_text_layout(), BFLOAT16),
            "qwen_2.5_vl_7b.safetensors",
        ),
        "vae": source(wan21_vae_geometries(), "qwen_image_vae.safetensors"),
    }
    plan = plan_qwen_image_assembly(**sources)
    assert isinstance(plan, QwenImageAssemblyPlan)
    assert plan.family is QWEN_IMAGE
    assert plan.diffusion.config is config
    assert plan.qwen2_5_vl_7b.config is QWEN_IMAGE_TEXT_CONFIG
    assert plan.vae.config is WAN21_VAE_CONFIG
    assert plan.identity_components == (
        plan.diffusion,
        plan.qwen2_5_vl_7b,
        plan.vae,
    )


def test_qwen_image_split_plan_accepts_float16_diffusion_storage() -> None:
    plan = plan_qwen_image_assembly(
        diffusion=source(geometrize(qwen_image_dit_layout().keys, FLOAT16)),
        qwen2_5_vl_7b=source(geometrize(qwen_image_text_layout(), BFLOAT16)),
        vae=source(wan21_vae_geometries()),
    )
    assert set(plan.diffusion.dtypes.values()) == {FLOAT16}


def test_qwen_image_plan_refuses_wrong_layout_and_nonfloating_storage() -> None:
    diffusion = geometrize(qwen_image_dit_layout().keys, BFLOAT16)
    diffusion["img_in.weight"] = g((3071, 64), BFLOAT16)
    with pytest.raises(AssemblyError, match="diffusion.*exact supported"):
        plan_qwen_image_assembly(
            diffusion=source(diffusion),
            qwen2_5_vl_7b=source(geometrize(qwen_image_text_layout(), BFLOAT16)),
            vae=source(wan21_vae_geometries()),
        )

    diffusion = geometrize(qwen_image_dit_layout().keys, UINT8)
    with pytest.raises(AssemblyError, match="require floating-point storage"):
        plan_qwen_image_assembly(
            diffusion=source(diffusion),
            qwen2_5_vl_7b=source(geometrize(qwen_image_text_layout(), BFLOAT16)),
            vae=source(wan21_vae_geometries()),
        )


@pytest.mark.parametrize("storage_dtype", (FLOAT8_E4M3, FLOAT8_E5M2))
def test_qwen_image_plan_admits_plain_fp8_diffusion_storage(storage_dtype: DType) -> None:
    diffusion = geometrize(qwen_image_dit_layout().keys, storage_dtype)
    plan = plan_qwen_image_assembly(
        diffusion=source(diffusion),
        qwen2_5_vl_7b=source(geometrize(qwen_image_text_layout(), BFLOAT16)),
        vae=source(wan21_vae_geometries()),
    )
    assert set(plan.diffusion.dtypes.values()) == {storage_dtype}
    assert plan.diffusion.quant == {}


def test_qwen_image_plan_admits_scaled_fp8_diffusion_layer() -> None:
    diffusion = geometrize(qwen_image_dit_layout().keys, BFLOAT16)
    quantize_legacy(diffusion, "img_in")
    plan = plan_qwen_image_assembly(
        diffusion=source(diffusion),
        qwen2_5_vl_7b=source(geometrize(qwen_image_text_layout(), BFLOAT16)),
        vae=source(wan21_vae_geometries()),
    )
    assert plan.diffusion.dtypes["img_in.weight"] == FLOAT8_E4M3
    assert plan.diffusion.quant["img_in"].format == "float8_e4m3fn"


def krea2_split_sources() -> dict[str, WeightSource]:
    return {
        "diffusion": source(geometrize(krea2_layout(), BFLOAT16), "krea2_raw.safetensors"),
        "qwen3vl_4b": source(
            geometrize(krea2_text_layout(), BFLOAT16),
            "qwen3vl_4b_bf16.safetensors",
        ),
        "vae": source(wan21_vae_geometries(), "qwen_image_vae.safetensors"),
    }


def test_krea2_component_planner_maps_exact_components() -> None:
    sources = krea2_split_sources()

    diffusion = plan_krea2_component(sources["diffusion"], "diffusion")
    text = plan_krea2_component(sources["qwen3vl_4b"], "qwen3vl_4b")

    assert diffusion.config is KREA2_CONFIG
    assert len(diffusion.keys) == 430
    assert diffusion.keys["first.weight"] == "first.weight"
    assert text.config is KREA2_TEXT_CONFIG
    # Model keys are the bare language tower; the vision tower is ignored.
    assert set(text.keys) == set(krea2_language_layout())
    assert text.keys["embed_tokens.weight"] == "model.language_model.embed_tokens.weight"
    assert "model.visual.patch_embed.proj.weight" in text.ignored


def test_krea2_component_planner_accepts_checkpoint_prefixed_sources() -> None:
    diffusion = plan_krea2_component(
        source(prefixed(geometrize(krea2_layout(), BFLOAT16), FLUX_DIFFUSION_PREFIX)),
        "diffusion",
    )
    text = plan_krea2_component(
        source(prefixed(geometrize(krea2_text_layout(), BFLOAT16), KREA2_TEXT_PREFIX)),
        "qwen3vl_4b",
    )
    assert diffusion.keys["first.weight"] == FLUX_DIFFUSION_PREFIX + "first.weight"
    assert text.keys["embed_tokens.weight"] == (
        KREA2_TEXT_PREFIX + "model.language_model.embed_tokens.weight"
    )

    float32 = plan_krea2_component(source(geometrize(krea2_layout(), FLOAT32)), "diffusion")
    assert set(float32.dtypes.values()) == {FLOAT32}


def test_krea2_component_text_plan_drops_the_tied_lm_head() -> None:
    text = geometrize(krea2_text_layout(), BFLOAT16)
    text["lm_head.weight"] = g((151936, 2560))
    plan = plan_krea2_component(source(text), "qwen3vl_4b")
    assert plan.config is KREA2_TEXT_CONFIG
    assert "lm_head.weight" in plan.ignored
    assert "lm_head.weight" not in plan.keys


def test_krea2_component_planner_refuses_wrong_layout_and_foreign_text_tower() -> None:
    incomplete = geometrize(krea2_layout(), BFLOAT16)
    del incomplete["blocks.27.mlp.down.weight"]
    with pytest.raises(AssemblyError, match="diffusion: .*missing blocks.27.mlp.down.weight"):
        plan_krea2_component(source(incomplete), "diffusion")

    with pytest.raises(AssemblyError, match="Krea 2 weights require floating-point storage"):
        plan_krea2_component(source(geometrize(krea2_layout(), UINT8)), "diffusion")

    foreign = geometrize(qwen_image_text_layout(), BFLOAT16)
    with pytest.raises(AssemblyError, match="qwen3vl_4b: not a Krea 2 Qwen3-VL-4B text role"):
        plan_krea2_component(source(foreign), "qwen3vl_4b")

    extra = geometrize(krea2_text_layout(), BFLOAT16)
    extra["model.language_model.unexpected.weight"] = g((4, 4))
    with pytest.raises(AssemblyError, match="qwen3vl_4b: unexpected text keys"):
        plan_krea2_component(source(extra), "qwen3vl_4b")


def test_krea2_combined_assembly_preserves_component_keys_and_unbound_vae() -> None:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(krea2_layout(), BFLOAT16), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(geometrize(krea2_text_layout(), BFLOAT16), KREA2_TEXT_PREFIX))
    sd.update(prefixed(wan21_vae_geometries(), FLUX_VAE_PREFIX))
    checkpoint = source(sd, "krea2_checkpoint.safetensors")
    capability = probe_native(checkpoint)
    assert capability.family_id == KREA2.id
    assert capability.native
    plan = plan_native(checkpoint)
    assert isinstance(plan, ComponentCheckpointPlan)
    assert tuple(plan.components) == ("diffusion", "qwen3vl_4b")
    assert set(plan.components["diffusion"].keys.values()) == {
        FLUX_DIFFUSION_PREFIX + key for key in krea2_layout()
    }
    assert set(plan.unclaimed) == {
        f"{checkpoint.path}:{FLUX_VAE_PREFIX}{key}" for key in wan21_vae_geometries()
    }

    sources = krea2_split_sources()
    split = plan_native(diffusion=sources["diffusion"], qwen3vl_4b=sources["qwen3vl_4b"])
    assert isinstance(split, ComponentCheckpointPlan)
    for expected in (
        plan_krea2_component(sources["diffusion"], "diffusion"),
        plan_krea2_component(sources["qwen3vl_4b"], "qwen3vl_4b"),
    ):
        part = split.components[expected.component]
        assert part.keys == expected.keys
        assert part.dtypes == expected.dtypes
        assert part.quant == expected.quant
    capability = probe_native(
        diffusion=sources["diffusion"],
        qwen3vl_4b=sources["qwen3vl_4b"],
        vae=sources["vae"],
    )
    assert capability.family_id == KREA2.id
    assert not capability.native
    assert any("source slot 'vae' has no matching components" in r for r in capability.reasons)


@pytest.mark.parametrize("storage_dtype", (FLOAT8_E4M3, FLOAT8_E5M2))
def test_krea2_component_planner_admits_plain_fp8_diffusion_storage(storage_dtype: DType) -> None:
    plan = plan_krea2_component(source(geometrize(krea2_layout(), storage_dtype)), "diffusion")
    assert set(plan.dtypes.values()) == {storage_dtype}
    assert plan.quant == {}


def anima_split_sources() -> dict[str, WeightSource]:
    qwen = prefixed(geometrize(qwen_text_layout(ANIMA_QWEN3_06B_CONFIG)), "model.")
    qwen["lm_head.weight"] = g((151936, 1024))
    return {
        "diffusion": source(prefixed(geometrize(anima_layout()), "net."), "anima.safetensors"),
        "qwen3_06b": source(qwen, "qwen3_06b.safetensors"),
        "vae": source(wan21_vae_geometries(), "qwen_image_vae.safetensors"),
    }


def test_anima_component_planner_maps_exact_components() -> None:
    sources = anima_split_sources()

    diffusion = plan_anima_component(sources["diffusion"], "diffusion")
    text = plan_anima_component(sources["qwen3_06b"], "qwen3_06b")

    assert diffusion.config is ANIMA_CONFIG
    assert len(diffusion.keys) == 685
    assert diffusion.keys["x_embedder.proj.1.weight"] == "net.x_embedder.proj.1.weight"
    assert text.config is ANIMA_QWEN3_06B_CONFIG
    assert len(text.keys) == 310
    assert text.keys["embed_tokens.weight"] == "model.embed_tokens.weight"
    assert text.keys["layers.27.mlp.down_proj.weight"] == ("model.layers.27.mlp.down_proj.weight")
    assert text.ignored == ("lm_head.weight",)


def test_anima_component_planner_accepts_floating_storage_and_refuses_wrong_layout_or_tower() -> (
    None
):
    incomplete = geometrize(anima_layout())
    del incomplete["llm_adapter.norm.weight"]
    with pytest.raises(AssemblyError, match="diffusion: source is not the exact Anima 2B layout"):
        plan_anima_component(source(incomplete), "diffusion")

    fp8 = plan_anima_component(source(geometrize(anima_layout(), FLOAT8_E4M3)), "diffusion")
    assert set(fp8.dtypes.values()) == {FLOAT8_E4M3}
    float32 = plan_anima_component(source(geometrize(anima_layout(), FLOAT32)), "diffusion")
    assert set(float32.dtypes.values()) == {FLOAT32}

    stowaway = prefixed(geometrize(anima_layout()), FLUX_DIFFUSION_PREFIX)
    stowaway["stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="diffusion: source contains unsupported"):
        plan_anima_component(source(stowaway), "diffusion")

    z_tower = prefixed(geometrize(qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG)), "model.")
    with pytest.raises(AssemblyError, match="qwen3_06b: Anima requires the Qwen3-0.6B"):
        plan_anima_component(source(z_tower), "qwen3_06b")

    stray = prefixed(geometrize(qwen_text_layout(ANIMA_QWEN3_06B_CONFIG)), "model.")
    stray["visual.blocks.0.attn.qkv.weight"] = g((3072, 1024))
    with pytest.raises(AssemblyError, match="qwen3_06b: unexpected non-text keys"):
        plan_anima_component(source(stray), "qwen3_06b")


def test_anima_combined_assembly_preserves_component_keys_and_unbound_vae() -> None:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(anima_layout()), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(geometrize(qwen_text_layout(ANIMA_QWEN3_06B_CONFIG)), ANIMA_QWEN_PREFIX))
    sd.update(prefixed(wan21_vae_geometries(), FLUX_VAE_PREFIX))
    checkpoint = source(sd, "anima_checkpoint.safetensors")
    capability = probe_native(checkpoint)
    assert capability.family_id == ANIMA.id
    assert capability.native
    plan = plan_native(checkpoint)
    assert isinstance(plan, ComponentCheckpointPlan)
    assert tuple(plan.components) == ("diffusion", "qwen3_06b")
    assert set(plan.components["diffusion"].keys.values()) == {
        FLUX_DIFFUSION_PREFIX + key for key in anima_layout()
    }
    assert set(plan.unclaimed) == {
        f"{checkpoint.path}:{FLUX_VAE_PREFIX}{key}" for key in wan21_vae_geometries()
    }

    sources = anima_split_sources()
    split = plan_native(diffusion=sources["diffusion"], qwen3_06b=sources["qwen3_06b"])
    assert isinstance(split, ComponentCheckpointPlan)
    for expected in (
        plan_anima_component(sources["diffusion"], "diffusion"),
        plan_anima_component(sources["qwen3_06b"], "qwen3_06b"),
    ):
        part = split.components[expected.component]
        assert part.keys == expected.keys
        assert part.dtypes == expected.dtypes
        assert part.quant == expected.quant
    capability = probe_native(
        diffusion=sources["diffusion"],
        qwen3_06b=sources["qwen3_06b"],
        vae=sources["vae"],
    )
    assert capability.family_id == ANIMA.id
    assert not capability.native
    assert any("source slot 'vae' has no matching components" in r for r in capability.reasons)


def flux2_vae_geometries() -> dict[str, TensorGeometry]:
    return synthetic_kl_geometries(z_channels=32, embed_dim=32, batch_norm_latent=True)


def mistral_split_geometries(config=MISTRAL3_24B_PRUNED_CONFIG) -> dict[str, TensorGeometry]:
    sd = prefixed(geometrize(qwen_text_layout(config)), "model.")
    sd["vision_tower.transformer.layers.0.attention.q_proj.weight"] = g((1024, 1024))
    sd["multi_modal_projector.linear_1.weight"] = g((5120, 4096))
    sd["multi_modal_projector.norm.weight"] = g((1024,))
    sd["tekken_model"] = g((14801754,), UINT8)
    return sd


class Flux2SplitSources(TypedDict):
    diffusion: WeightSource
    text_encoder: WeightSource
    vae: WeightSource


def flux2_split_sources(
    dit_config=FLUX2_DEV_CONFIG,
    text: dict[str, TensorGeometry] | None = None,
) -> Flux2SplitSources:
    return {
        "diffusion": source(geometrize(flux2_layout(dit_config)), "flux2_dit.safetensors"),
        "text_encoder": source(
            text if text is not None else mistral_split_geometries(), "flux2_text.safetensors"
        ),
        "vae": source(flux2_vae_geometries(), "flux2_vae.safetensors"),
    }


@pytest.mark.parametrize("text_config", (MISTRAL3_24B_PRUNED_CONFIG, MISTRAL3_24B_CONFIG))
def test_flux2_dev_split_plan_maps_exact_components(text_config) -> None:
    plan = plan_flux2_assembly(**flux2_split_sources(text=mistral_split_geometries(text_config)))
    assert isinstance(plan, Flux2AssemblyPlan)
    assert plan.family is FLUX2_DEV
    assert plan.diffusion.config is FLUX2_DEV_CONFIG
    assert plan.text_encoder.component == "mistral3_24b"
    assert plan.text_encoder.config is text_config
    assert plan.vae.config.batch_norm_latent
    assert plan.vae.config.latent_channels == 128
    assert plan.diffusion.keys["img_in.weight"] == "img_in.weight"
    assert plan.text_encoder.keys["embed_tokens.weight"] == "model.embed_tokens.weight"
    assert sorted(plan.text_encoder.ignored) == [
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.norm.weight",
        "tekken_model",
        "vision_tower.transformer.layers.0.attention.q_proj.weight",
    ]
    assert plan.identity_components == (plan.diffusion, plan.text_encoder, plan.vae)
    assert plan.unclaimed == ()


@pytest.mark.parametrize(
    ("dit_config", "family", "slot", "text_config", "layout_config"),
    (
        (FLUX2_KLEIN_9B_CONFIG, FLUX2_KLEIN_9B, "qwen3_8b", KLEIN_QWEN3_8B_CONFIG, None),
        (
            FLUX2_KLEIN_4B_CONFIG,
            FLUX2_KLEIN_4B,
            "qwen3_4b",
            KLEIN_QWEN3_4B_CONFIG,
            Z_IMAGE_QWEN3_4B_CONFIG,
        ),
    ),
)
def test_flux2_klein_split_plans_pin_the_klein_text_profiles(
    dit_config, family, slot, text_config, layout_config
) -> None:
    tower = prefixed(
        geometrize(qwen_text_layout(layout_config or text_config)),
        "model.",
    )
    tower["lm_head.weight"] = g(tower["model.embed_tokens.weight"].shape)
    plan = plan_flux2_assembly(**flux2_split_sources(dit_config, text=tower))
    assert plan.family is family
    assert plan.diffusion.config is dit_config
    assert plan.text_encoder.component == slot
    assert plan.text_encoder.config is text_config
    assert plan.text_encoder.ignored == ("lm_head.weight",)
    assert plan.unclaimed == ()


@pytest.mark.parametrize(
    "role,text_config,diffusion_config",
    (
        ("mistral3_24b", MISTRAL3_24B_PRUNED_CONFIG, FLUX2_DEV_CONFIG),
        ("qwen3_4b", KLEIN_QWEN3_4B_CONFIG, FLUX2_KLEIN_4B_CONFIG),
    ),
)
def test_flux2_text_components_ignore_distinct_checkpoint_namespaces(
    role, text_config, diffusion_config
) -> None:
    prefix = f"text_encoders.{role}.transformer.model."
    text = prefixed(geometrize(qwen_text_layout(text_config)), prefix)
    combined = {
        **text,
        **prefixed(geometrize(flux2_layout(diffusion_config)), FLUX_DIFFUSION_PREFIX),
        **prefixed(flux2_vae_geometries(), FLUX_VAE_PREFIX),
    }
    expected = plan_flux2_component(source(text), role)
    assert plan_flux2_component(source(combined), role) == expected
    for key in ("model.stowaway.weight", f"text_encoders.{role}.stowaway.weight"):
        with pytest.raises(AssemblyError, match="unexpected|unsupported"):
            plan_flux2_component(source({**combined, key: g((1,))}), role)


def test_flux2_combined_checkpoint_plan_claims_all_roles() -> None:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(flux2_layout(FLUX2_DEV_CONFIG)), FLUX_DIFFUSION_PREFIX))
    sd.update(
        prefixed(geometrize(qwen_text_layout(MISTRAL3_24B_PRUNED_CONFIG)), FLUX2_MISTRAL_PREFIX)
    )
    sd["text_encoders.mistral3_24b.logit_scale"] = g((), FLOAT32)
    sd["text_encoders.mistral3_24b.transformer.lm_head.weight"] = g((131072, 5120))
    sd.update(prefixed(flux2_vae_geometries(), FLUX_VAE_PREFIX))
    sd["extra.unrelated"] = g((1,))
    plan = plan_flux2_assembly(checkpoint=source(sd, "flux2_checkpoint.safetensors"))
    assert plan.family is FLUX2_DEV
    assert plan.diffusion.keys["img_in.weight"] == FLUX_DIFFUSION_PREFIX + "img_in.weight"
    assert plan.text_encoder.keys["embed_tokens.weight"] == (
        FLUX2_MISTRAL_PREFIX + "embed_tokens.weight"
    )
    assert sorted(plan.text_encoder.ignored) == [
        "text_encoders.mistral3_24b.logit_scale",
        "text_encoders.mistral3_24b.transformer.lm_head.weight",
    ]
    assert plan.vae.config.latent_channels == 128
    assert plan.unclaimed == ("extra.unrelated",)


def test_flux2_probe_native_and_plan_native_route_split_sources() -> None:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    sources = flux2_split_sources()
    capability = probe_native(
        diffusion=sources["diffusion"],
        mistral3_24b=sources["text_encoder"],
        vae=sources["vae"],
    )
    assert capability.native and capability.family_id == FLUX2_DEV.id
    plan = plan_native(
        diffusion=sources["diffusion"],
        mistral3_24b=sources["text_encoder"],
        vae=sources["vae"],
    )
    assert isinstance(plan, ComponentCheckpointPlan)
    assert plan.family is FLUX2_DEV
    assert plan.identity_components == plan_flux2_assembly(**sources).identity_components


def test_flux2_probe_refuses_foreign_text_slots() -> None:
    sources = flux2_split_sources()
    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["text_encoder"],
        vae=sources["vae"],
    )
    assert not capability.native
    assert any("wires only" in reason for reason in capability.reasons)


def test_flux2_probe_refuses_two_split_text_sources() -> None:
    sources = flux2_split_sources()
    capability = probe_native(
        diffusion=sources["diffusion"],
        mistral3_24b=sources["text_encoder"],
        qwen3_8b=sources["text_encoder"],
        vae=sources["vae"],
    )
    assert not capability.native
    assert any(
        "source slot 'qwen3_8b' requires role 'qwen3_8b'; detected ('mistral3_24b',)" in reason
        for reason in capability.reasons
    )


def test_flux2_refuses_family_text_weights_through_a_sibling_slot() -> None:
    sources = flux2_split_sources()
    capability = probe_native(
        diffusion=sources["diffusion"],
        qwen3_8b=sources["text_encoder"],
        vae=sources["vae"],
    )
    assert not capability.native
    assert any(
        "source slot 'qwen3_8b' requires role 'qwen3_8b'; detected ('mistral3_24b',)" in reason
        for reason in capability.reasons
    )
    with pytest.raises(AssemblyError, match="wires only the mistral3_24b text encoder slot"):
        plan_flux2_assembly(
            diffusion=sources["diffusion"],
            text_encoder=sources["text_encoder"],
            text_slot="qwen3_8b",
            vae=sources["vae"],
        )


def _flux2_fp8_layer(key: str) -> str | None:
    """The official fp8mixed dev export quantizes exactly the double-block
    MLP matmuls and the single-block fused linears."""
    if not key.endswith(".weight"):
        return None
    parts = key[: -len(".weight")].split(".")
    if len(parts) == 4 and parts[0] == "double_blocks" and parts[2] in ("img_mlp", "txt_mlp"):
        if parts[3] in ("0", "2"):
            return ".".join(parts)
    if len(parts) == 3 and parts[0] == "single_blocks" and parts[2] in ("linear1", "linear2"):
        return ".".join(parts)
    return None


def test_flux2_dev_fp8mixed_metadata_shape_plans_quant_and_norm_spelling() -> None:
    sd: dict[str, TensorGeometry] = {}
    layers: list[str] = []
    for key, shape in flux2_layout(FLUX2_DEV_CONFIG).items():
        if key.endswith("_norm.weight"):
            sd[key[: -len(".weight")] + ".scale"] = g(shape)
            continue
        layer = _flux2_fp8_layer(key)
        if layer is None:
            sd[key] = g(shape)
            continue
        layers.append(layer)
        sd[key] = g(shape, FLOAT8_E4M3)
        sd[f"{layer}.weight_scale"] = g((), FLOAT32)
        sd[f"{layer}.input_scale"] = g((), FLOAT32)
    dit = source(sd, "flux2_dev_fp8mixed.safetensors")
    dit.extra["_quantization_metadata"] = json.dumps(
        {
            "format_version": "1.0",
            "layers": {layer: {"format": "float8_e4m3fn"} for layer in layers},
        }
    )
    plan = plan_flux2_assembly(**{**flux2_split_sources(), "diffusion": dit})
    assert plan.family is FLUX2_DEV
    assert len(layers) == 8 * 4 + 48 * 2
    assert sorted(plan.diffusion.quant) == sorted(layers)
    quant = plan.diffusion.quant["double_blocks.0.img_mlp.0"]
    assert quant.format == "float8_e4m3fn"
    assert quant.weight_scale == "double_blocks.0.img_mlp.0.weight_scale"
    assert quant.input_scale == "double_blocks.0.img_mlp.0.input_scale"
    assert plan.diffusion.dtypes["single_blocks.47.linear2.weight"] == FLOAT8_E4M3
    assert plan.diffusion.keys["double_blocks.0.img_attn.norm.query_norm.weight"] == (
        "double_blocks.0.img_attn.norm.query_norm.scale"
    )


def test_flux2_plan_refuses_foreign_components() -> None:
    sources = flux2_split_sources()

    with pytest.raises(AssemblyError, match="diffusion"):
        plan_flux2_assembly(**{**sources, "diffusion": source(flux_geometries())})

    classic = split_sources()
    classic["diffusion"] = source(geometrize(flux2_layout(FLUX2_DEV_CONFIG)))
    with pytest.raises(AssemblyError, match="diffusion"):
        plan_flux_assembly(**classic)

    z_tower = prefixed(geometrize(qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG)), "model.")
    with pytest.raises(
        AssemblyError,
        match="mistral3_24b: Flux2 dinkster.flux2_dev requires one of the"
        " mistral3_24b, mistral3_24b_pruned text profiles, found z_image_qwen3_4b",
    ):
        plan_flux2_assembly(**{**sources, "text_encoder": source(z_tower)})

    klein9 = flux2_split_sources(
        FLUX2_KLEIN_9B_CONFIG,
        text=prefixed(geometrize(qwen_text_layout(KLEIN_QWEN3_8B_CONFIG)), "model."),
    )
    bare_mistral = prefixed(geometrize(qwen_text_layout(MISTRAL3_24B_PRUNED_CONFIG)), "model.")
    with pytest.raises(
        AssemblyError,
        match="qwen3_8b: Flux2 dinkster.flux2_klein_9b requires one of the"
        " klein_qwen3_8b text profiles, found mistral3_24b_pruned",
    ):
        plan_flux2_assembly(**{**klein9, "text_encoder": source(bare_mistral)})

    klein_with_towers = prefixed(geometrize(qwen_text_layout(KLEIN_QWEN3_8B_CONFIG)), "model.")
    klein_with_towers["vision_tower.patch_embed.weight"] = g((1024, 3))
    with pytest.raises(AssemblyError, match="qwen3_8b: unexpected non-text keys"):
        plan_flux2_assembly(**{**klein9, "text_encoder": source(klein_with_towers)})

    stray = mistral_split_geometries()
    stray["audio_tower.encoder.weight"] = g((1024, 1024))
    with pytest.raises(AssemblyError, match="mistral3_24b: unexpected non-text keys"):
        plan_flux2_assembly(**{**sources, "text_encoder": source(stray)})

    for deceptive_key, geometry in (
        ("audio_tower.lm_head.weight", g((1024, 1024))),
        ("audio_tower.logit_scale", g((), FLOAT32)),
        ("tekken_model.extra", g((8,), UINT8)),
        ("transformer.lm_head.weight", g((1024, 1024))),
        ("transformer.tekken_model", g((8,), UINT8)),
        ("text_encoders.mistral3_24b.lm_head.weight", g((1024, 1024))),
        ("text_encoders.mistral3_24b.transformer.logit_scale", g((), FLOAT32)),
    ):
        deceptive = mistral_split_geometries()
        deceptive[deceptive_key] = geometry
        with pytest.raises(AssemblyError, match="mistral3_24b: unexpected non-text keys"):
            plan_flux2_assembly(**{**sources, "text_encoder": source(deceptive)})

    with pytest.raises(AssemblyError, match="vae: Flux2 requires the 128-channel"):
        plan_flux2_assembly(**{**sources, "vae": source(kl_geometries())})

    small_batch_norm = synthetic_kl_geometries(batch_norm_latent=True)
    with pytest.raises(AssemblyError, match="vae: Flux2 requires the 128-channel"):
        plan_flux2_assembly(**{**sources, "vae": source(small_batch_norm)})

    with pytest.raises(AssemblyError, match="no sources given"):
        plan_flux2_assembly()


def test_flux2_plan_validator_pins_family_text_and_vae() -> None:
    plan = plan_flux2_assembly(**flux2_split_sources())
    with pytest.raises(ValueError, match="published family and diffusion profile"):
        replace(plan, family=FLUX_DEV)

    klein = plan_flux2_assembly(
        **flux2_split_sources(
            FLUX2_KLEIN_9B_CONFIG,
            text=prefixed(geometrize(qwen_text_layout(KLEIN_QWEN3_8B_CONFIG)), "model."),
        )
    )
    with pytest.raises(ValueError, match="text profiles"):
        replace(plan, text_encoder=klein.text_encoder)

    classic = plan_flux_assembly(**split_sources())
    with pytest.raises(ValueError, match="batch-norm KL VAE"):
        replace(plan, vae=classic.vae)


def test_flux2_families_have_distinct_pinned_identities() -> None:
    dev = plan_flux2_assembly(**flux2_split_sources())
    klein9 = plan_flux2_assembly(
        **flux2_split_sources(
            FLUX2_KLEIN_9B_CONFIG,
            text=prefixed(geometrize(qwen_text_layout(KLEIN_QWEN3_8B_CONFIG)), "model."),
        )
    )
    klein4 = plan_flux2_assembly(
        **flux2_split_sources(
            FLUX2_KLEIN_4B_CONFIG,
            text=prefixed(geometrize(qwen_text_layout(KLEIN_QWEN3_4B_CONFIG)), "model."),
        )
    )
    classic = plan_flux_assembly(**split_sources())
    identities = {
        runtime_component_identity(plan.family.id, plan.identity_components)
        for plan in (dev, klein9, klein4)
    }
    assert len(identities) == 3
    assert runtime_component_identity(classic.family.id, classic.identity_components) not in (
        identities
    )
    assert build_runtime_identity(
        dev.family.id,
        dev.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    ) == (
        "native:dinkster.flux2_dev:49ec420567ca6c7878848bfba6dafe7a66e35f26cd18ecb2e80b3d61e0c37d30"
    )


def wan21_split_sources(profile: str = "t2v-1.3b") -> dict[str, WeightSource]:
    text = geometrize(t5_layout(UMT5_XXL_CONFIG), BFLOAT16)
    text["spiece_model"] = g((128,), UINT8)
    flow_rvs = profile == "flow-rvs-1.3b"
    sources: dict[str, WeightSource] = {
        "diffusion": source(geometrize(wan21_shapes(profile)), "wan.safetensors"),
        "umt5xxl": source(text, "umt5.safetensors"),
        "vae": source(
            wan21_vae_geometries(WAN21_FLOW_RVS_VAE_CONFIG if flow_rvs else WAN21_VAE_CONFIG),
            "wan_vae.safetensors",
        ),
    }
    if flow_rvs:
        diffusion = cast(FakeSource, sources["diffusion"])
        diffusion.extra["config"] = '{"transformer":{"model_type":"flow_rvs"}}'
    if profile == "causal-ar-1.3b":
        diffusion = cast(FakeSource, sources["diffusion"])
        diffusion.extra["config"] = '{"transformer":{"causal_ar":true}}'
    if profile == "animate2-14b-2.1":
        diffusion = cast(FakeSource, sources["diffusion"])
        diffusion.extra["config"] = '{"transformer":{"model_type":"animate2"}}'
    if profile == "bernini-14b-2.2":
        diffusion = cast(FakeSource, sources["diffusion"])
        diffusion.extra["model_type"] = "bernini_high"
    if profile in (
        "i2v-14b",
        "scail-14b",
        "scail2-14b",
        "animate2-14b-2.1",
        "animate-14b-2.2",
        "wandancer-14b-2.2",
        "flf-i2v-14b",
        "fun-control-1.3b",
        "fun-inpaint-1.3b",
        "camera-1.3b",
        "camera-14b",
    ):
        clip_vision = geometrize(clip_vision_layout(), FLOAT16)
        clip_vision["vision_model.embeddings.position_ids"] = g((1, 257), INT64)
        sources["clip_vision"] = source(clip_vision, "clip_vision_h.safetensors")
    return sources


def wan21_combined_geometries() -> dict[str, TensorGeometry]:
    values = prefixed(geometrize(wan21_shapes("t2v-1.3b")), WAN21_DIFFUSION_PREFIX)
    values.update(prefixed(geometrize(t5_layout(UMT5_XXL_CONFIG)), WAN21_UMT5_PREFIX))
    values["text_encoders.umt5xxl.spiece_model"] = g((128,), UINT8)
    values.update(prefixed(wan21_vae_geometries(), WAN21_VAE_PREFIX))
    return values


def wan22_split_sources(profile: str = "ti2v-5b") -> dict[str, WeightSource]:
    text = geometrize(t5_layout(UMT5_XXL_CONFIG), BFLOAT16)
    text["spiece_model"] = g((128,), UINT8)
    return {
        "diffusion": source(geometrize(wan22_shapes(profile=profile)), "wan22.safetensors"),
        "umt5xxl": source(text, "umt5.safetensors"),
        "vae": source(geometrize(wan22_vae_layout()), "wan22_vae.safetensors"),
    }


# --------------------------------------------------------- happy paths


def test_wan21_split_sources_plan_exact_1_3b_components() -> None:
    plan = plan_wan21_assembly(**wan21_split_sources())

    assert plan.family is WAN21
    assert plan.diffusion.config == WAN21_T2V_1_3B
    assert plan.umt5xxl.config == UMT5_XXL_CONFIG
    assert plan.vae.config == WAN21_VAE_CONFIG
    assert plan.tokenizer_source_key == "spiece_model"
    assert plan.umt5xxl.path == Path("/fake/umt5.safetensors")
    assert "spiece_model" not in plan.umt5xxl.keys
    assert set(plan.diffusion.keys) == set(wan21_shapes("t2v-1.3b"))
    assert set(plan.vae.keys) == set(wan21_vae_geometries())
    assert plan.unclaimed == ()


def test_wan21_standalone_components_plan_the_official_alpha_workflow_roles() -> None:
    sources = wan21_split_sources("t2v-14b")

    diffusion = plan_wan21_standalone_component(sources["diffusion"], "diffusion")
    text = plan_wan21_standalone_component(sources["umt5xxl"], "umt5xxl")
    vae = plan_wan21_standalone_component(sources["vae"], "vae")

    assert isinstance(diffusion, Wan21StandaloneComponentPlan)
    assert diffusion.component.config is WAN21_T2V_14B
    assert text.component.config == UMT5_XXL_CONFIG
    assert text.tokenizer_source_key == "spiece_model"
    assert "spiece_model" not in text.component.keys
    assert vae.component.config == WAN21_VAE_CONFIG
    assert tuple(plan.role for plan in (diffusion, text, vae)) == (
        "diffusion",
        "umt5xxl",
        "vae",
    )


def test_wan21_causal_ar_plans_full_and_standalone_components() -> None:
    sources = wan21_split_sources("causal-ar-1.3b")

    full = plan_wan21_assembly(**sources)
    standalone = plan_wan21_standalone_component(sources["diffusion"], "diffusion")

    assert full.diffusion.config is WAN21_CAUSAL_AR_1_3B
    assert full.vae.config == WAN21_VAE_CONFIG
    assert full.clip_vision is None
    assert standalone.component.config is WAN21_CAUSAL_AR_1_3B
    assert set(standalone.component.keys) == set(wan21_shapes("causal-ar-1.3b"))


def test_wan21_humo_plans_full_and_standalone_components() -> None:
    sources = wan21_split_sources("humo-17b")

    full = plan_wan21_assembly(**sources)
    standalone = plan_wan21_standalone_component(sources["diffusion"], "diffusion")

    assert full.diffusion.config is WAN21_HUMO_17B
    assert full.clip_vision is None
    assert full.vae.config == WAN21_VAE_CONFIG
    assert standalone.component.config is WAN21_HUMO_17B
    assert set(standalone.component.keys) == set(wan21_shapes("humo-17b"))


def test_wandancer_plans_exact_components_and_fused_music_qkv() -> None:
    sources = wan21_split_sources("wandancer-14b-2.2")

    full = plan_wan21_assembly(**sources)
    standalone = plan_wan21_standalone_component(sources["diffusion"], "diffusion")

    assert full.diffusion.config is WAN22_WANDANCER_14B
    assert full.clip_vision is not None
    assert full.vae.config == WAN21_VAE_CONFIG
    assert standalone.component.config is WAN22_WANDANCER_14B
    expected = set(wan21_shapes("wandancer-14b-2.2"))
    expected -= {
        f"music_encoder.{layer}.self_attn.in_proj_{suffix}"
        for layer in range(2)
        for suffix in ("weight", "bias")
    }
    expected |= {
        f"music_encoder.{layer}.self_attn.{projection}.{suffix}"
        for layer in range(2)
        for projection in ("q_proj", "k_proj", "v_proj")
        for suffix in ("weight", "bias")
    }
    assert set(full.diffusion.keys) == expected
    assert set(standalone.component.keys) == expected
    assert len(standalone.component.transforms) == 12
    for layer in range(2):
        for suffix in ("weight", "bias"):
            source_key = f"music_encoder.{layer}.self_attn.in_proj_{suffix}"
            for part, projection in enumerate(("q_proj", "k_proj", "v_proj")):
                target = f"music_encoder.{layer}.self_attn.{projection}.{suffix}"
                assert standalone.component.keys[target] == source_key
                assert standalone.component.transforms[target] == RowChunk(part=part, parts=3)


def test_wandancer_refuses_nearby_base_i2v_or_incomplete_variant_state() -> None:
    base_sources = wan21_split_sources("i2v-14b")
    with pytest.raises(AssemblyError, match="standalone Wan requires"):
        plan_wan21_standalone_component(base_sources["diffusion"], "diffusion")

    dancer_sources = wan21_split_sources("wandancer-14b-2.2")
    diffusion = cast(FakeSource, dancer_sources["diffusion"])
    del diffusion.geometries["music_projection.weight"]
    with pytest.raises(AssemblyError, match="standalone Wan requires"):
        plan_wan21_standalone_component(diffusion, "diffusion")


def test_wan21_standalone_diffusion_refuses_non_alpha_base_geometry() -> None:
    with pytest.raises(
        AssemblyError,
        match="standalone Wan requires T2V 14B, CausalAR 1.3B, HuMo 17B, S2V 14B, or WanDancer 14B",
    ):
        plan_wan21_standalone_component(wan21_split_sources()["diffusion"], "diffusion")


def test_wan21_flow_rvs_pairs_exact_metadata_diffusion_and_mask_vae() -> None:
    sources = wan21_split_sources("flow-rvs-1.3b")
    plan = plan_wan21_assembly(**sources)

    assert plan.family is WAN21
    assert plan.diffusion.config == WAN21_FLOW_RVS_1_3B
    assert plan.vae.config == WAN21_FLOW_RVS_VAE_CONFIG
    assert plan.clip_vision is None
    assert set(plan.diffusion.keys) == set(wan21_shapes("flow-rvs-1.3b"))
    assert set(plan.vae.keys) == set(wan21_vae_geometries(WAN21_FLOW_RVS_VAE_CONFIG))

    structurally_equal_diffusion = replace(
        plan.diffusion,
        config=replace(WAN21_FLOW_RVS_1_3B),
    )
    assert structurally_equal_diffusion.config is not WAN21_FLOW_RVS_1_3B
    replace(plan, diffusion=structurally_equal_diffusion)


def test_wan21_flow_rvs_refuses_the_base_vae() -> None:
    sources = wan21_split_sources("flow-rvs-1.3b")
    sources["vae"] = source(wan21_vae_geometries(), "base_wan_vae.safetensors")

    expected = r"decoder\.head\.2\.weight expected shape \(1, 96, 3, 3, 3\)"
    with pytest.raises(AssemblyError, match=expected):
        plan_wan21_assembly(**sources)


def test_wan22_split_sources_plan_and_probe_exact_ti2v_components() -> None:
    sources = wan22_split_sources()
    plan = plan_wan22_assembly(**sources)

    assert plan.family is WAN22
    assert plan.diffusion.config == WAN22_TI2V_5B
    assert plan.umt5xxl.config == UMT5_XXL_CONFIG
    assert plan.vae.config == WAN22_VAE_CONFIG
    assert plan.clip_vision is None
    assert set(plan.diffusion.keys) == set(wan22_shapes())
    assert set(plan.vae.keys) == set(wan22_vae_layout())
    assert plan.unclaimed == ()

    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        vae=sources["vae"],
    )
    assert capability.family_id == "dinkster.wan22"
    assert capability.native is True
    assert capability.reasons == ()


def test_wan22_planner_and_native_probe_refuse_non_floating_vae_storage() -> None:
    sources = wan22_split_sources()
    geometries = dict(cast("FakeSource", sources["vae"]).geometries)
    key = "conv2.weight"
    geometries[key] = g(geometries[key].shape, INT64)
    sources["vae"] = source(geometries, "integer-wan22-vae.safetensors")

    with pytest.raises(AssemblyError, match=f"storage dtype mismatch for {key}"):
        plan_wan22_assembly(**sources)

    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        vae=sources["vae"],
    )
    assert capability.family_id == "dinkster.wan22"
    assert capability.native is False
    assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
    assert any(
        f"vae: storage dtype mismatch for {key}: got int64, expected floating" in reason
        for reason in capability.reasons
    )


def test_wan21_combined_source_claims_only_exact_component_roots() -> None:
    checkpoint = source(wan21_combined_geometries(), "wan-combined.safetensors")
    plan = plan_wan21_assembly(checkpoint=checkpoint)

    assert all(component.path == checkpoint.path for component in plan.identity_components)
    assert plan.tokenizer_source_key == "text_encoders.umt5xxl.spiece_model"
    assert plan.diffusion.keys["patch_embedding.weight"] == (
        "model.diffusion_model.patch_embedding.weight"
    )
    assert plan.umt5xxl.keys["shared.weight"] == ("text_encoders.umt5xxl.transformer.shared.weight")
    assert plan.vae.keys["encoder.conv1.weight"] == "vae.encoder.conv1.weight"
    assert plan.unclaimed == ()


@dataclass
class GgufFakeSource(FakeSource):
    """FakeSource presenting the GGUF-source planning surface."""

    source_format: str = "gguf"
    runtime_facts: tuple[str, ...] = ("gguf.artifact.file_sha256=" + "a" * 64,)


def gguf_text_source(config: T5Config, name: str = "text.gguf") -> GgufFakeSource:
    return GgufFakeSource(Path(f"/fake/{name}"), geometrize(t5_layout(config), FLOAT32))


def test_flux_plan_carries_gguf_text_source_facts_and_payload() -> None:
    sources = split_sources()
    gguf = gguf_text_source(T5_XXL_CONFIG)
    sources["t5xxl"] = gguf
    plan = plan_flux_assembly(**sources)

    assert plan.t5xxl is not None
    assert plan.t5xxl.source_format == "gguf"
    assert plan.t5xxl.runtime_facts == gguf.runtime_facts
    assert plan.t5xxl.payload_source is gguf
    assert plan.t5xxl.config == T5_XXL_CONFIG
    assert plan.t5xxl.ignored == ()
    assert plan.diffusion.source_format == "safetensors"


def test_ltxv_plan_carries_gguf_text_source_facts_and_payload() -> None:
    gguf = gguf_text_source(T5_XXL_CONFIG)
    plan = plan_ltxv_standalone_component(gguf, "t5xxl").component

    assert plan.source_format == "gguf"
    assert plan.runtime_facts == gguf.runtime_facts
    assert plan.payload_source is gguf
    assert plan.config == T5_XXL_CONFIG
    assert plan.ignored == ()


def test_wan21_plan_vendors_the_tokenizer_for_gguf_text_sources() -> None:
    sources = wan21_split_sources()
    sources["umt5xxl"] = gguf_text_source(UMT5_XXL_CONFIG, "umt5.gguf")
    plan = plan_wan21_assembly(**sources)

    assert plan.tokenizer_vendored is True
    assert plan.tokenizer_source_key == ""
    assert plan.umt5xxl.source_format == "gguf"
    assert plan.umt5xxl.config == UMT5_XXL_CONFIG

    safetensors_plan = plan_wan21_assembly(**wan21_split_sources())
    assert safetensors_plan.tokenizer_vendored is False
    assert safetensors_plan.tokenizer_source_key == "spiece_model"


def test_wan21_plan_requires_exactly_one_tokenizer_origin() -> None:
    sources = wan21_split_sources()
    sources["umt5xxl"] = gguf_text_source(UMT5_XXL_CONFIG, "umt5.gguf")
    gguf_plan = plan_wan21_assembly(**sources)
    safetensors_plan = plan_wan21_assembly(**wan21_split_sources())

    origin = "Wan assembly requires exactly one tokenizer origin"
    with pytest.raises(ValueError, match=origin):
        replace(gguf_plan, tokenizer_source_key="spiece_model")
    with pytest.raises(ValueError, match=origin):
        replace(safetensors_plan, tokenizer_source_key="")
    with pytest.raises(ValueError, match="vendored tokenizer is only for GGUF text sources"):
        replace(safetensors_plan, tokenizer_source_key="", tokenizer_vendored=True)


@pytest.mark.parametrize(
    ("profile", "config"),
    (
        ("t2v-14b", WAN21_T2V_14B),
        ("humo-17b", WAN21_HUMO_17B),
        ("i2v-14b", WAN21_I2V_14B),
        ("scail-14b", WAN21_SCAIL_14B),
        ("scail2-14b", WAN21_SCAIL2_14B),
        ("animate2-14b-2.1", WAN21_ANIMATE2_14B),
        ("animate-14b-2.2", WAN22_ANIMATE_14B),
        ("bernini-14b-2.2", WAN22_BERNINI_14B),
        ("wandancer-14b-2.2", WAN22_WANDANCER_14B),
        ("flf-i2v-14b", WAN21_FLF_I2V_14B),
        ("fun-control-1.3b", WAN21_FUN_CONTROL_1_3B),
        ("fun-inpaint-1.3b", WAN21_FUN_INPAINT_1_3B),
        ("i2v-14b-2.2", WAN22_I2V_14B),
        ("fun-control-14b-2.2", WAN22_FUN_CONTROL_14B),
        ("vace-1.3b", WAN21_VACE_1_3B),
        ("vace-14b", WAN21_VACE_14B),
        ("camera-1.3b", WAN21_CAMERA_1_3B),
        ("camera-14b", WAN21_CAMERA_14B),
        ("camera-14b-2.2", WAN22_CAMERA_14B),
    ),
)
def test_wan21_profiles_plan_exact_components(profile: str, config: object) -> None:
    plan = plan_wan21_assembly(**wan21_split_sources(profile))

    assert plan.diffusion.config == config
    if profile != "wandancer-14b-2.2":
        assert set(plan.diffusion.keys) == set(wan21_shapes(profile))
    assert (plan.clip_vision is not None) == (
        profile
        in (
            "i2v-14b",
            "scail-14b",
            "scail2-14b",
            "animate2-14b-2.1",
            "animate-14b-2.2",
            "wandancer-14b-2.2",
            "flf-i2v-14b",
            "fun-control-1.3b",
            "fun-inpaint-1.3b",
            "camera-1.3b",
            "camera-14b",
        )
    )


@pytest.mark.skipif(
    not REAL_WAN22_BERNINI.exists(),
    reason="pinned Bernini artifact absent (set DINKSTER_WAN22_BERNINI)",
)
def test_real_bernini_header_plans_the_exact_native_profile() -> None:
    expected_size, expected_sha256, source_url = WAN22_BERNINI_ARTIFACT
    assert source_url == (
        "https://huggingface.co/Comfy-Org/Bernini-R/resolve/"
        "fc371005c90d24177f3658cfacd78b44a41bbd8e/diffusion_models/"
        "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"
    )
    assert REAL_WAN22_BERNINI.stat().st_size == expected_size
    assert len(expected_sha256) == 64
    diffusion = load_safetensors_header(REAL_WAN22_BERNINI)
    assert diffusion.metadata() == {"format": "pt", "model_type": "bernini_high"}

    sources = wan21_split_sources("bernini-14b-2.2")
    sources["diffusion"] = diffusion
    plan = plan_wan21_assembly(**sources)

    assert plan.family is WAN21
    assert plan.diffusion.config is WAN22_BERNINI_14B
    assert len(plan.diffusion.keys) == 1095
    assert len(plan.diffusion.quant) == 360
    assert plan.diffusion.ignored == ()


def test_wan21_non_quantized_admission_does_not_read_quant_payloads() -> None:
    class PayloadTrapSource(FakeSource):
        def read_uint8_configuration(self, key: str) -> bytes:
            raise AssertionError(f"plain checkpoint read quantization payload {key}")

    baseline_sources = wan21_split_sources("t2v-1.3b")
    baseline = plan_wan21_assembly(**baseline_sources)
    plain = cast(FakeSource, baseline_sources["diffusion"])
    baseline_sources["diffusion"] = PayloadTrapSource(
        plain.path,
        dict(plain.geometries),
        dict(plain.extra),
    )

    trapped = plan_wan21_assembly(**baseline_sources)

    assert trapped == baseline
    assert build_runtime_identity(
        trapped.family.id,
        trapped.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    ) == build_runtime_identity(
        baseline.family.id,
        baseline.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )


def test_wan21_animate2_metadata_spellings_have_one_plan_and_identity() -> None:
    official_sources = wan21_split_sources("animate2-14b-2.1")
    redundant_sources = wan21_split_sources("animate2-14b-2.1")
    redundant = cast(FakeSource, redundant_sources["diffusion"])
    redundant.extra["config"] = '{"transformer":{"image_model":"wan2.1","model_type":"animate2"}}'

    official = plan_wan21_assembly(**official_sources)
    redundant_plan = plan_wan21_assembly(**redundant_sources)

    assert redundant_plan == official
    assert build_runtime_identity(
        redundant_plan.family.id,
        redundant_plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    ) == build_runtime_identity(
        official.family.id,
        official.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )


def test_wan21_animate2_int8_convrot_payload_detects_and_plans() -> None:
    sources = wan21_split_sources("animate2-14b-2.1")
    diffusion = cast(FakeSource, sources["diffusion"])
    layer = "blocks.0.cross_attn.k"
    weight = f"{layer}.weight"
    diffusion.geometries[weight] = g(diffusion.geometries[weight].shape, INT8)
    diffusion.geometries[f"{layer}.weight_scale"] = g((5120, 1), FLOAT32)
    config = f"{layer}.comfy_quant"
    payload = json.dumps(
        {
            "convrot": True,
            "convrot_groupsize": 256,
            "format": "int8_tensorwise",
        },
        separators=(",", ":"),
    ).encode()
    diffusion.geometries[config] = g((len(payload),), UINT8)
    diffusion.payload_values[config] = payload

    plan = plan_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        clip_vision=sources["clip_vision"],
        vae=sources["vae"],
    )
    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        clip_vision=sources["clip_vision"],
        vae=sources["vae"],
    )

    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    assert isinstance(plan, ComponentCheckpointPlan)
    assert plan.identity_components == plan_wan21_assembly(**sources).identity_components
    assert plan.components["diffusion"].config == WAN21_ANIMATE2_14B
    assert plan.components["diffusion"].dtypes[weight] == INT8
    quant = plan.components["diffusion"].quant[layer]
    assert quant.format == "int8_tensorwise"
    assert quant.parameters == {"convrot": True, "convrot_groupsize": 256}
    assert capability.native is True
    assert capability.family_id == "dinkster.wan21"


def test_wan21_animate2_refuses_malformed_quantization_payload() -> None:
    sources = wan21_split_sources("animate2-14b-2.1")
    diffusion = cast(FakeSource, sources["diffusion"])
    layer = "blocks.0.cross_attn.k"
    weight = f"{layer}.weight"
    diffusion.geometries[weight] = g(diffusion.geometries[weight].shape, INT8)
    diffusion.geometries[f"{layer}.weight_scale"] = g((5120, 1), FLOAT32)
    config = f"{layer}.comfy_quant"
    diffusion.geometries[config] = g((8,), UINT8)
    diffusion.payload_values[config] = b"not-json"

    with pytest.raises(AssemblyError) as caught:
        plan_wan21_assembly(**sources)
    assert str(caught.value) == "diffusion: source is not an official Wan 2.1 profile"

    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        clip_vision=sources["clip_vision"],
        vae=sources["vae"],
    )
    assert capability.family_id is None
    assert capability.native is False
    assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
    assert capability.reasons == (
        "diffusion: blocks.0.cross_attn.k.comfy_quant: malformed configuration: "
        "Expecting value: line 1 column 1 (char 0)",
    )


def test_wan21_animate2_undeclared_integer_weight_refuses_exactly() -> None:
    sources = wan21_split_sources("animate2-14b-2.1")
    diffusion = cast(FakeSource, sources["diffusion"])
    weight = "blocks.0.cross_attn.k.weight"
    diffusion.geometries[weight] = g(diffusion.geometries[weight].shape, INT8)

    with pytest.raises(AssemblyError) as caught:
        plan_wan21_assembly(**sources)
    assert str(caught.value) == "diffusion: source is not an official Wan 2.1 profile"

    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        clip_vision=sources["clip_vision"],
        vae=sources["vae"],
    )
    assert capability.family_id is None
    assert capability.native is False
    assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
    assert any(
        "source is not an official Wan 2.1 profile" in reason for reason in capability.reasons
    )


@pytest.mark.parametrize(
    ("profile", "config"),
    (
        ("fun-control-5b", WAN22_FUN_CONTROL_5B),
        ("fun-inpaint-5b", WAN22_FUN_INPAINT_5B),
    ),
)
def test_wan22_fun_profiles_plan_exact_components(profile: str, config: object) -> None:
    plan = plan_wan22_assembly(**wan22_split_sources(profile))

    assert plan.family is WAN22
    assert plan.diffusion.config == config
    assert set(plan.diffusion.keys) == set(wan22_shapes(profile=profile))
    assert plan.clip_vision is None


@pytest.mark.parametrize(
    "profile",
    (
        "t2v-1.3b",
        "t2v-14b",
        "i2v-14b",
        "flf-i2v-14b",
        "fun-control-1.3b",
        "fun-inpaint-1.3b",
        "i2v-14b-2.2",
        "fun-control-14b-2.2",
        "vace-1.3b",
        "vace-14b",
        "camera-1.3b",
        "camera-14b",
        "camera-14b-2.2",
    ),
)
def test_wan21_planner_and_native_probe_refuse_non_floating_diffusion_storage(
    profile: str,
) -> None:
    sources = wan21_split_sources(profile)
    geometries = dict(cast("FakeSource", sources["diffusion"]).geometries)
    key = "blocks.0.ffn.0.bias"
    geometries[key] = g(geometries[key].shape, INT64)
    sources["diffusion"] = source(geometries, "integer-wan.safetensors")

    with pytest.raises(AssemblyError, match=f"{key} requires floating-point storage, found int64"):
        plan_wan21_assembly(**sources)

    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        clip_vision=sources.get("clip_vision"),
        vae=sources["vae"],
    )
    assert capability.family_id == "dinkster.wan21"
    assert capability.native is False
    assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
    assert any(
        f"diffusion: {key} requires floating-point storage, found int64" in reason
        for reason in capability.reasons
    )


def test_wan21_missing_duplicate_and_malformed_components_fail_closed() -> None:
    sources = wan21_split_sources()
    with pytest.raises(AssemblyError, match="umt5xxl: no source"):
        plan_wan21_assembly(diffusion=sources["diffusion"], vae=sources["vae"])

    combined = source(wan21_combined_geometries(), "duplicated-wan.safetensors")
    with pytest.raises(AssemblyError, match="split component sources must be distinct"):
        plan_wan21_assembly(diffusion=combined, umt5xxl=combined, vae=combined)

    text = source(
        {
            **cast("FakeSource", sources["umt5xxl"]).geometries,
            "text_encoders.umt5xxl.spiece_model": g((128,), UINT8),
        },
        "duplicate-tokenizer.safetensors",
    )
    with pytest.raises(AssemblyError, match="exactly one"):
        plan_wan21_assembly(diffusion=sources["diffusion"], umt5xxl=text, vae=sources["vae"])

    malformed_vae = dict(cast("FakeSource", sources["vae"]).geometries)
    malformed_vae["encoder.conv1.weight"] = g((96, 4, 3, 3, 3), FLOAT32)
    with pytest.raises(AssemblyError, match="encoder.conv1.weight expected shape"):
        plan_wan21_assembly(
            diffusion=sources["diffusion"],
            umt5xxl=sources["umt5xxl"],
            vae=source(malformed_vae),
        )

    incomplete_diffusion = dict(cast("FakeSource", sources["diffusion"]).geometries)
    del incomplete_diffusion["blocks.17.ffn.0.bias"]
    with pytest.raises(AssemblyError, match="missing blocks.17.ffn.0.bias"):
        plan_wan21_assembly(
            diffusion=source(incomplete_diffusion),
            umt5xxl=sources["umt5xxl"],
            vae=sources["vae"],
        )

    malformed_diffusion = dict(cast("FakeSource", sources["diffusion"]).geometries)
    malformed_diffusion["blocks.17.ffn.0.bias"] = g((8959,))
    with pytest.raises(AssemblyError, match="blocks.17.ffn.0.bias expected shape"):
        plan_wan21_assembly(
            diffusion=source(malformed_diffusion),
            umt5xxl=sources["umt5xxl"],
            vae=sources["vae"],
        )


def test_wan21_split_sources_reject_foreign_and_duplicate_layouts() -> None:
    sources = wan21_split_sources()
    text = cast("FakeSource", sources["umt5xxl"])
    duplicated_text = dict(text.geometries)
    duplicated_text.update(prefixed(text.geometries, WAN21_UMT5_PREFIX))
    with pytest.raises(AssemblyError, match="umt5xxl: source contains unsupported or duplicate"):
        plan_wan21_assembly(
            diffusion=sources["diffusion"],
            umt5xxl=source(duplicated_text),
            vae=sources["vae"],
        )

    vae = cast("FakeSource", sources["vae"])
    duplicated_vae = dict(vae.geometries)
    duplicated_vae.update(prefixed(vae.geometries, WAN21_VAE_PREFIX))
    with pytest.raises(AssemblyError, match="vae: source contains unsupported or duplicate"):
        plan_wan21_assembly(
            diffusion=sources["diffusion"],
            umt5xxl=sources["umt5xxl"],
            vae=source(duplicated_vae),
        )


# The per-channel aggregates the LTX checkpoints carry beyond the two
# the runtime consumes; the planner drops them instead of loading them.
LTXV_UNUSED_STATISTICS = (
    "per_channel_statistics.channel",
    "per_channel_statistics.mean-of-stds",
    "per_channel_statistics.mean-of-stds_over_std-of-means",
)


def ltxv_vae_geometries(
    config: LTXVideoVAEConfig | None = None, *, statistics: bool = True
) -> dict[str, TensorGeometry]:
    vae_config = config if config is not None else LTXV_2B_V09_VAE_CONFIG
    sd = geometrize(ltxv_vae_layout(vae_config), FLOAT32)
    if statistics:
        for key in LTXV_UNUSED_STATISTICS:
            sd[key] = g((128,), FLOAT32)
    return sd


def ltxv_split_sources(profile: str = "2b-v0.9") -> dict[str, WeightSource]:
    configs = {
        "2b-v0.9": (LTXV_2B_V09_CONFIG, LTXV_2B_V09_VAE_CONFIG),
        "2b-v0.9.5": (LTXV_2B_V095_CONFIG, LTXV_2B_V095_VAE_CONFIG),
    }
    diffusion_config, vae_config = configs[profile]
    diffusion = source(geometrize(ltxv_layout(diffusion_config)), "ltxv.safetensors")
    if profile == "2b-v0.9.5":
        diffusion.extra["config"] = json.dumps(
            {"transformer": {"causal_temporal_positioning": True}}
        )
    return {
        "diffusion": diffusion,
        "t5xxl": source(t5_geometries(alias=True), "t5.safetensors"),
        "vae": source(ltxv_vae_geometries(vae_config), "ltxv_vae.safetensors"),
    }


@pytest.mark.parametrize(
    ("profile", "diffusion_config", "vae_config"),
    (
        ("2b-v0.9", LTXV_2B_V09_CONFIG, LTXV_2B_V09_VAE_CONFIG),
        ("2b-v0.9.5", LTXV_2B_V095_CONFIG, LTXV_2B_V095_VAE_CONFIG),
    ),
)
def test_ltxv_split_sources_plan_exact_components(
    profile: str, diffusion_config: object, vae_config: object
) -> None:
    sources = ltxv_split_sources(profile)
    plans = {
        role: plan_ltxv_standalone_component(sources[role], cast("Any", role))
        for role in ("diffusion", "t5xxl", "vae")
    }

    assert plans["diffusion"].component.config is diffusion_config
    assert plans["t5xxl"].component.config == T5_XXL_CONFIG
    assert plans["vae"].component.config is vae_config
    assert set(plans["diffusion"].component.keys) == set(
        ltxv_layout(cast("LTXVConfig", diffusion_config))
    )
    assert plans["t5xxl"].component.ignored == ("encoder.embed_tokens.weight",)
    assert set(plans["vae"].component.keys) == set(
        ltxv_vae_layout(cast("LTXVideoVAEConfig", vae_config))
    )
    assert sorted(plans["vae"].component.ignored) == sorted(LTXV_UNUSED_STATISTICS)
    for role, plan in plans.items():
        assert plan.role == role
        assert plan.identity_components == (plan.component,)


@pytest.mark.parametrize("profile", ("2b-v0.9", "2b-v0.9.5"))
def test_ltxv_components_split_combined_namespaces_without_changing_plans(profile: str) -> None:
    from dinkster_inference.component_catalog import default_component_registry

    sources = cast("dict[str, FakeSource]", ltxv_split_sources(profile))
    prefixes = {
        "diffusion": FLUX_DIFFUSION_PREFIX,
        "t5xxl": FLUX_T5XXL_PREFIX,
        "vae": FLUX_VAE_PREFIX,
    }
    combined = {
        key: geometry
        for role, part in sources.items()
        for key, geometry in prefixed(part.geometries, prefixes[role]).items()
    }
    checkpoint = source(combined, "combined.safetensors")
    checkpoint.extra.update(sources["diffusion"].extra)
    for role, prefix in prefixes.items():
        expected = plan_ltxv_standalone_component(sources[role], cast("Any", role)).component
        actual = plan_ltxv_standalone_component(checkpoint, cast("Any", role)).component
        assert actual.config == expected.config
        assert actual.dtypes == expected.dtypes
        assert actual.quant == expected.quant
        assert actual.transforms == expected.transforms
        assert actual.keys == {key: prefix + value for key, value in expected.keys.items()}
        assert actual.ignored == tuple(prefix + key for key in expected.ignored)

    cast("Any", checkpoint).asset_digest = "blake3:" + "1" * 64
    cast("Any", checkpoint).asset_size = 1000
    descriptor = default_component_registry().get("dinkster.ltxv")
    assert descriptor is not None
    assert tuple(role for role, _ in descriptor.detector(checkpoint, checkpoint.path)) == tuple(
        prefixes
    )


@pytest.mark.parametrize("extra", ("stowaway.weight", "vae.stowaway.weight", "first_stage_model.x"))
def test_combined_component_selection_keeps_unknown_keys_and_same_role_aliases(extra: str) -> None:
    combined = prefixed(ltxv_vae_geometries(), FLUX_VAE_PREFIX)
    combined.update(prefixed(geometrize(ltxv_layout(LTXV_2B_V09_CONFIG)), FLUX_DIFFUSION_PREFIX))
    combined[extra] = g((1,))
    with pytest.raises(AssemblyError, match="unsupported or duplicate|layout"):
        plan_ltxv_standalone_component(source(combined), "vae")


def test_combined_text_selection_keeps_siblings_inside_its_encoder_namespace() -> None:
    combined = prefixed(t5_geometries(alias=True), FLUX_T5XXL_PREFIX)
    combined.update(prefixed(ltxv_vae_geometries(), FLUX_VAE_PREFIX))
    combined["text_encoders.t5xxl.stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unsupported or duplicate"):
        plan_ltxv_standalone_component(source(combined), "t5xxl")


def test_ltxv_vae_profile_matches_independently_of_the_diffusion_profile() -> None:
    v095 = plan_ltxv_standalone_component(
        source(ltxv_vae_geometries(LTXV_2B_V095_VAE_CONFIG), "v095_vae.safetensors"),
        "vae",
    )
    v09 = plan_ltxv_standalone_component(
        source(ltxv_vae_geometries(LTXV_2B_V09_VAE_CONFIG), "v09_vae.safetensors"),
        "vae",
    )

    assert v095.component.config is LTXV_2B_V095_VAE_CONFIG
    assert v09.component.config is LTXV_2B_V09_VAE_CONFIG


def test_ltxv_vae_statistics_are_optional_and_never_load() -> None:
    plan = plan_ltxv_standalone_component(
        source(ltxv_vae_geometries(statistics=False), "bare_vae.safetensors"), "vae"
    ).component
    assert plan.ignored == ()
    assert set(plan.keys) == set(ltxv_vae_layout(LTXV_2B_V09_VAE_CONFIG))
    for key in LTXV_UNUSED_STATISTICS:
        assert key not in plan.keys


def test_ltxv_missing_malformed_and_foreign_components_fail_closed() -> None:
    with pytest.raises(AssemblyError, match="not an official LTX-Video 2B profile"):
        plan_ltxv_standalone_component(source(geometrize(wan21_shapes("t2v-1.3b"))), "diffusion")

    incomplete = geometrize(ltxv_layout(LTXV_2B_V09_CONFIG))
    del incomplete["transformer_blocks.0.attn1.to_q.weight"]
    with pytest.raises(AssemblyError, match="missing transformer_blocks.0.attn1.to_q.weight"):
        plan_ltxv_standalone_component(source(incomplete), "diffusion")

    stowaway = geometrize(ltxv_layout(LTXV_2B_V09_CONFIG))
    stowaway["stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unexpected stowaway.weight"):
        plan_ltxv_standalone_component(source(stowaway), "diffusion")

    malformed = geometrize(ltxv_layout(LTXV_2B_V09_CONFIG))
    malformed["transformer_blocks.0.attn1.to_q.weight"] = g((2047, 2048))
    with pytest.raises(
        AssemblyError, match="transformer_blocks.0.attn1.to_q.weight expected shape"
    ):
        plan_ltxv_standalone_component(source(malformed), "diffusion")

    integer = geometrize(ltxv_layout(LTXV_2B_V09_CONFIG))
    integer["proj_out.bias"] = g((128,), INT64)
    with pytest.raises(AssemblyError, match="requires floating-point storage, found int64"):
        plan_ltxv_standalone_component(source(integer), "diffusion")

    umt5 = geometrize(t5_layout(UMT5_XXL_CONFIG), FLOAT16)
    with pytest.raises(AssemblyError, match="t5xxl: LTX-Video requires the classic T5-XXL"):
        plan_ltxv_standalone_component(source(umt5), "t5xxl")

    with pytest.raises(AssemblyError, match="does not match either LTX-Video 2B causal VAE"):
        plan_ltxv_standalone_component(source(kl_geometries()), "vae")


def test_ltxv_t5_and_vae_statistics_fail_closed() -> None:
    integer_t5 = t5_geometries()
    integer_t5["shared.weight"] = g((32128, 4096), INT64)
    with pytest.raises(
        AssemblyError, match="t5xxl: shared.weight requires floating-point storage, found int64"
    ):
        plan_ltxv_standalone_component(source(integer_t5), "t5xxl")

    misshaped = ltxv_vae_geometries()
    misshaped["per_channel_statistics.channel"] = g((64,), FLOAT32)
    with pytest.raises(
        AssemblyError, match="per_channel_statistics.channel expected floating-point shape"
    ):
        plan_ltxv_standalone_component(source(misshaped), "vae")

    integer_stats = ltxv_vae_geometries()
    integer_stats["per_channel_statistics.mean-of-stds"] = g((128,), INT64)
    with pytest.raises(
        AssemblyError, match="per_channel_statistics.mean-of-stds expected floating-point shape"
    ):
        plan_ltxv_standalone_component(source(integer_stats), "vae")


@pytest.mark.parametrize(
    "roles",
    [("diffusion",), ("diffusion", "t5xxl"), ("diffusion", "vae"), ("diffusion", "t5xxl", "vae")],
)
def test_ltxv_general_runtime_planner_preserves_detected_component_keys(
    roles: tuple[str, ...],
) -> None:
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    sources = ltxv_split_sources()
    actual = plan_native(**{role: sources[role] for role in roles})
    assert isinstance(actual, ComponentCheckpointPlan)
    assert tuple(actual.components) == roles
    for role in roles:
        expected = plan_ltxv_standalone_component(sources[role], cast("Any", role))
        assert actual.components[role].config == expected.component.config
        assert actual.components[role].keys == expected.component.keys


def test_ltxv_standalone_plan_validator_pins_role_and_profile() -> None:
    plan = plan_ltxv_standalone_component(ltxv_split_sources()["diffusion"], "diffusion")
    with pytest.raises(ValueError, match="exact supported roles"):
        replace(plan, role=cast("Any", "vae"))


def ltxav_metadata() -> dict[str, Any]:
    """The ltx-2-19b-dev checkpoint's config metadata sections the
    planner gates on, with the real header's values."""
    return {
        "transformer": {
            "rope_type": "split",
            "frequencies_precision": "float64",
            "positional_embedding_theta": 10000.0,
            "timestep_scale_multiplier": 1000,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "audio_positional_embedding_max_pos": [20],
            "causal_temporal_positioning": True,
            "use_middle_indices_grid": True,
            "av_ca_timestep_scale_multiplier": 1000.0,
            "attention_head_dim": 128,
            "num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_num_attention_heads": 32,
        },
        "vae": {"timestep_conditioning": False},
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "double_z": True,
                        "mel_bins": 64,
                        "z_channels": 8,
                        "attn_resolutions": [],
                        "norm_type": "pixel",
                        "causality_axis": "height",
                    }
                }
            },
            "preprocessing": {
                "audio": {"sampling_rate": 16000},
                "stft": {"filter_length": 1024, "hop_length": 160},
                "mel": {"n_mel_channels": 64},
            },
        },
        "vocoder": {
            "resblock": "1",
            "stereo": True,
            "upsample_initial_channel": 1024,
            "upsample_rates": [6, 5, 2, 2, 2],
            "upsample_kernel_sizes": [16, 15, 8, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        },
    }


def ltxav_22b_metadata(*, text_encoder_norm_type: str = "per_token_rms") -> dict[str, Any]:
    metadata = ltxav_metadata()
    metadata["transformer"].update(
        {
            "av_cross_ada_norm": True,
            "use_embeddings_connector": True,
            "connector_norm_output": True,
            "apply_gated_attention": True,
            "connector_apply_gated_attention": True,
            "caption_proj_before_connector": True,
            "cross_attention_adaln": True,
            "caption_projection_first_linear": False,
            "caption_projection_second_linear": False,
            "caption_proj_input_norm": False,
            "text_encoder_norm_type": text_encoder_norm_type,
            "connector_attention_head_dim": 128,
            "connector_num_attention_heads": 32,
            "connector_num_layers": 8,
            "connector_num_learnable_registers": 128,
            "connector_positional_embedding_max_pos": [4096],
            "audio_connector_attention_head_dim": 64,
            "audio_connector_num_attention_heads": 32,
        }
    )
    metadata["vae"]["spatial_padding_mode"] = "zeros"
    vocoder = LTXAV_BWE_VOCODER_CONFIG.vocoder
    bwe = LTXAV_BWE_VOCODER_CONFIG.bwe_generator
    metadata["vocoder"] = {
        "vocoder": {
            "resblock": vocoder.resblock,
            "stereo": vocoder.stereo,
            "activation": vocoder.activation,
            "upsample_initial_channel": vocoder.upsample_initial_channel,
            "upsample_rates": list(vocoder.upsample_rates),
            "upsample_kernel_sizes": list(vocoder.upsample_kernel_sizes),
            "resblock_kernel_sizes": list(vocoder.resblock_kernel_sizes),
            "resblock_dilation_sizes": [list(values) for values in vocoder.resblock_dilation_sizes],
            "use_bias_at_final": vocoder.use_bias_at_final,
            "use_tanh_at_final": vocoder.use_tanh_at_final,
        },
        "bwe": {
            "resblock": bwe.resblock,
            "stereo": bwe.stereo,
            "activation": bwe.activation,
            "upsample_initial_channel": bwe.upsample_initial_channel,
            "upsample_rates": list(bwe.upsample_rates),
            "upsample_kernel_sizes": list(bwe.upsample_kernel_sizes),
            "resblock_kernel_sizes": list(bwe.resblock_kernel_sizes),
            "resblock_dilation_sizes": [list(values) for values in bwe.resblock_dilation_sizes],
            "use_bias_at_final": bwe.use_bias_at_final,
            "use_tanh_at_final": bwe.use_tanh_at_final,
            "apply_final_activation": bwe.apply_final_activation,
            "input_sampling_rate": LTXAV_BWE_VOCODER_CONFIG.input_sampling_rate,
            "output_sampling_rate": LTXAV_BWE_VOCODER_CONFIG.output_sampling_rate,
            "hop_length": LTXAV_BWE_VOCODER_CONFIG.hop_length,
            "n_fft": LTXAV_BWE_VOCODER_CONFIG.n_fft,
            "win_size": LTXAV_BWE_VOCODER_CONFIG.n_fft,
            "num_mels": LTXAV_BWE_VOCODER_CONFIG.num_mels,
        },
    }
    return metadata


def ltxav_connector_geometries() -> dict[str, TensorGeometry]:
    tower = geometrize(ltx_connector_layout(LTX_TEXT_CONNECTOR_CONFIG))
    sd: dict[str, TensorGeometry] = {}
    for connector_prefix in LTXAV_CONNECTOR_PREFIXES:
        sd.update(prefixed(tower, connector_prefix))
    return sd


def ltxav_checkpoint_geometries() -> dict[str, TensorGeometry]:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(ltxav_layout(LTXAV_19B_CONFIG)), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(ltxav_connector_geometries(), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(ltxv_vae_geometries(LTXAV_19B_VAE_CONFIG), FLUX_VAE_PREFIX))
    sd.update(
        prefixed(
            geometrize(ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG), FLOAT32), "audio_vae."
        )
    )
    sd.update(
        prefixed(geometrize(ltx_vocoder_layout(LTXAV_19B_VOCODER_CONFIG), FLOAT32), "vocoder.")
    )
    sd["text_embedding_projection.aggregate_embed.weight"] = g((3840, LTX_TEXT_STACK_FEATURES))
    return sd


def ltxav_checkpoint(
    metadata: dict[str, Any] | None = None,
    geometries: dict[str, TensorGeometry] | None = None,
) -> FakeSource:
    checkpoint = source(
        geometries if geometries is not None else ltxav_checkpoint_geometries(),
        "ltx-2-19b-dev.safetensors",
    )
    checkpoint.extra["config"] = json.dumps(metadata if metadata is not None else ltxav_metadata())
    return checkpoint


def ltxav_22b_checkpoint() -> FakeSource:
    geometries: dict[str, TensorGeometry] = {}
    geometries.update(
        prefixed(geometrize(ltxav_layout(LTXAV_22B_V23_CONFIG)), FLUX_DIFFUSION_PREFIX)
    )
    geometries.update(
        prefixed(
            ltxv_vae_geometries(LTXAV_22B_V23_VAE_CONFIG, statistics=False),
            FLUX_VAE_PREFIX,
        )
    )
    geometries.update(
        prefixed(
            geometrize(ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG), FLOAT32),
            "audio_vae.",
        )
    )
    geometries.update(
        prefixed(
            geometrize(ltx_vocoder_bwe_layout(LTXAV_BWE_VOCODER_CONFIG), FLOAT32),
            "vocoder.",
        )
    )
    geometries.update(
        {
            "text_embedding_projection.audio_aggregate_embed.bias": g((2048,)),
            "text_embedding_projection.audio_aggregate_embed.weight": g(
                (2048, LTX_TEXT_STACK_FEATURES)
            ),
            "text_embedding_projection.video_aggregate_embed.bias": g((4096,)),
            "text_embedding_projection.video_aggregate_embed.weight": g(
                (4096, LTX_TEXT_STACK_FEATURES)
            ),
        }
    )
    return ltxav_checkpoint(ltxav_22b_metadata(), geometries)


def ltxav_22b_v25_diffusion() -> FakeSource:
    geometries = geometrize(ltxav_layout(LTXAV_22B_V25_CONFIG))
    diffusion = source(geometries, "ltx-2.4-22b-sft.safetensors")
    diffusion.extra["config"] = json.dumps(
        ltxav_22b_metadata(text_encoder_norm_type="PER_TOKEN_RMS")
    )
    return diffusion


def ltxav_audio_codec_source() -> FakeSource:
    geometries = prefixed(
        geometrize(ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG), FLOAT32),
        "audio_vae.",
    )
    geometries.update(
        prefixed(
            geometrize(ltx_vocoder_bwe_layout(LTXAV_BWE_VOCODER_CONFIG), FLOAT32),
            "vocoder.",
        )
    )
    codec = source(geometries, "ltx-2-audio-codec.safetensors")
    metadata = ltxav_22b_metadata()
    codec.extra["config"] = json.dumps(metadata)
    return codec


def ltxav_gemma_geometries(*, projection: bool = False) -> dict[str, TensorGeometry]:
    sd = prefixed(geometrize(gemma_text_layout(GEMMA3_LTX_12B_CONFIG)), "model.")
    sd["spiece_model"] = g((4096,), UINT8)
    if projection:
        sd["text_embedding_projection.aggregate_embed.weight"] = g((3840, LTX_TEXT_STACK_FEATURES))
    return sd


def ltxav_gemma4_geometries(*, projection: bool = False) -> dict[str, TensorGeometry]:
    sd = prefixed(geometrize(gemma_text_layout(GEMMA4_LTX_12B_CONFIG)), "model.")
    sd["tokenizer_json"] = g((32768,), UINT8)
    if projection:
        sd.update(
            prefixed(
                geometrize(
                    {
                        "audio_aggregate_embed.bias": (2048,),
                        "audio_aggregate_embed.weight": (2048, LTX_TEXT_STACK_FEATURES),
                        "video_aggregate_embed.bias": (4096,),
                        "video_aggregate_embed.weight": (4096, LTX_TEXT_STACK_FEATURES),
                    }
                ),
                "text_embedding_projection.",
            )
        )
    return sd


def test_ltxav_gemma4_accepts_exact_hugging_face_auxiliary_assets() -> None:
    geometries = ltxav_gemma4_geometries(projection=True)
    for key in (
        "hf_asset__chat_template.jinja",
        "hf_asset__generation_config.json",
        "hf_asset__processor_config.json",
        "hf_asset__tokenizer_config.json",
    ):
        geometries[key] = g((128,), UINT8)

    plan = plan_ltxav_standalone_component(
        source(geometries, "gemma4-ltx-2.5.safetensors"),
        "gemma4_12b",
    )

    assert plan.component.config is GEMMA4_LTX_12B_CONFIG
    assert plan.tokenizer_source_key == "tokenizer_json"


def test_ltxav_gemma4_rejects_unknown_hugging_face_auxiliary_asset() -> None:
    geometries = ltxav_gemma4_geometries()
    geometries["hf_asset__merges.txt"] = g((128,), UINT8)

    with pytest.raises(AssemblyError, match="unsupported .*hf_asset__merges.txt"):
        plan_ltxav_standalone_component(
            source(geometries, "gemma4-ltx-2.5.safetensors"),
            "gemma4_12b",
        )


def test_ltxav_19b_checkpoint_components_plan_independently() -> None:
    checkpoint = ltxav_checkpoint()
    gemma_source = source(ltxav_gemma_geometries(), "gemma3-12b.safetensors")
    diffusion = plan_ltxav_standalone_component(checkpoint, "diffusion")
    gemma = plan_ltxav_standalone_component(gemma_source, "gemma3_12b")
    projection = plan_ltxav_standalone_component(checkpoint, "text_projection")
    connectors = plan_ltxav_standalone_component(checkpoint, "connectors")
    vae = plan_ltxav_standalone_component(checkpoint, "vae")
    audio = plan_ltxav_audio_codec(checkpoint)

    diffusion_config = cast("LTXAVConfig", diffusion.component.config)
    assert diffusion_config == LTXAV_19B_CONFIG
    assert diffusion_config.av_ca_timestep_scale_multiplier == 1000.0
    assert set(diffusion.component.keys) == set(ltxav_layout(LTXAV_19B_CONFIG))
    assert not any(key.startswith(LTXAV_CONNECTOR_PREFIXES) for key in diffusion.component.keys)
    assert gemma.component.config is GEMMA3_LTX_12B_CONFIG
    assert gemma.tokenizer_source_key == "spiece_model"
    assert projection.component.config == "single_linear"
    assert dict(projection.component.keys) == {
        "weight": "text_embedding_projection.aggregate_embed.weight"
    }
    assert connectors.component.config == LTX_TEXT_CONNECTOR_CONFIG
    assert set(connectors.component.keys) == set(ltxav_connector_geometries())
    assert vae.component.config is LTXAV_19B_VAE_CONFIG
    assert set(vae.component.keys) == set(ltxv_vae_layout(LTXAV_19B_VAE_CONFIG))
    for key in LTXV_UNUSED_STATISTICS:
        assert FLUX_VAE_PREFIX + key in vae.component.ignored
    assert audio.audio_vae.config is LTXAV_19B_AUDIO_VAE_CONFIG
    assert audio.vocoder.config is LTXAV_19B_VOCODER_CONFIG
    for planned in (diffusion, gemma, projection, connectors, vae):
        assert planned.identity_components == (planned.component,)


def test_ltxav_22b_checkpoint_components_plan_independently() -> None:
    checkpoint = ltxav_22b_checkpoint()
    diffusion = plan_ltxav_standalone_component(checkpoint, "diffusion")
    projection = plan_ltxav_standalone_component(checkpoint, "text_projection")
    vae = plan_ltxav_standalone_component(checkpoint, "vae")
    audio = plan_ltxav_audio_codec(checkpoint)

    assert diffusion.component.config == LTXAV_22B_V23_CONFIG
    assert set(diffusion.component.keys) == set(ltxav_layout(LTXAV_22B_V23_CONFIG))
    assert any("video_embeddings_connector" in key for key in diffusion.component.keys)
    assert any("audio_embeddings_connector" in key for key in diffusion.component.keys)
    assert projection.component.config == "dual_linear"
    assert set(projection.component.keys) == {
        "audio_aggregate_embed.bias",
        "audio_aggregate_embed.weight",
        "video_aggregate_embed.bias",
        "video_aggregate_embed.weight",
    }
    assert vae.component.config is LTXAV_22B_V23_VAE_CONFIG
    assert audio.vocoder.config is LTXAV_BWE_VOCODER_CONFIG
    with pytest.raises(AssemblyError, match="unsupported .*_embeddings_connector variant"):
        plan_ltxav_standalone_component(checkpoint, "connectors")


@pytest.mark.parametrize("norm_type", ("PER_TOKEN_RMS", "None"))
def test_ltxav_22b_v25_diffusion_plans_without_video_feed_forward_biases(
    norm_type: str,
) -> None:
    source_value = ltxav_22b_v25_diffusion()
    source_value.extra["config"] = json.dumps(ltxav_22b_metadata(text_encoder_norm_type=norm_type))

    plan = plan_ltxav_standalone_component(source_value, "diffusion")

    assert plan.component.config == LTXAV_22B_V25_CONFIG
    assert set(plan.component.keys) == set(ltxav_layout(LTXAV_22B_V25_CONFIG))
    assert not any(key.startswith("duration_head.") for key in plan.component.keys)


@pytest.mark.parametrize(
    "prefix",
    ("", "duration_head.", FLUX_DIFFUSION_PREFIX + "duration_head."),
)
def test_ltxav_duration_head_accepts_reference_key_layouts(prefix: str) -> None:
    layout = ltxav_duration_head_layout(LTXAV_DURATION_HEAD_CONFIG)
    geometries = prefixed(geometrize(layout), prefix)
    if prefix.startswith(FLUX_DIFFUSION_PREFIX):
        geometries[FLUX_DIFFUSION_PREFIX + "diffusion_sibling.weight"] = g((1,))
    duration = source(geometries, "ltx-2-duration-head.safetensors")

    plan = plan_ltxav_standalone_component(duration, "duration_head")

    assert set(plan.component.keys) == set(layout)
    assert dict(plan.component.keys) == {key: prefix + key for key in layout}


def test_ltxav_duration_head_rejects_bare_stowaway() -> None:
    geometries = geometrize(ltxav_duration_head_layout(LTXAV_DURATION_HEAD_CONFIG))
    geometries["stowaway"] = g((1,))

    with pytest.raises(AssemblyError, match="unexpected stowaway"):
        plan_ltxav_standalone_component(
            source(geometries, "ltx-2-duration-head.safetensors"),
            "duration_head",
        )


def _ltx_latent_upscaler_source() -> FakeSource:
    upscaler = source(
        geometrize(ltx_latent_upsampler_layout(), BFLOAT16),
        "ltx-2-spatial-upscaler.safetensors",
    )
    upscaler.extra["config"] = json.dumps(
        {
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
    )
    return upscaler


def test_ltx_latent_upscaler_requires_exact_metadata_and_layout() -> None:
    upscaler = _ltx_latent_upscaler_source()

    plan = plan_ltxav_standalone_component(upscaler, "latent_upscaler")

    assert plan.component.config == LTX_LATENT_UPSAMPLER_CONFIG
    assert set(plan.component.keys) == set(ltx_latent_upsampler_layout())

    wrong_metadata = _ltx_latent_upscaler_source()
    config = json.loads(wrong_metadata.extra["config"])
    config["spatial_scale"] = 1.5
    wrong_metadata.extra["config"] = json.dumps(config)
    with pytest.raises(AssemblyError, match="config metadata does not match"):
        plan_ltxav_standalone_component(wrong_metadata, "latent_upscaler")

    incomplete = _ltx_latent_upscaler_source()
    del incomplete.geometries["initial_conv.weight"]
    with pytest.raises(AssemblyError, match="missing initial_conv.weight"):
        plan_ltxav_standalone_component(incomplete, "latent_upscaler")

    stowaway = _ltx_latent_upscaler_source()
    stowaway.geometries["stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unexpected stowaway.weight"):
        plan_ltxav_standalone_component(stowaway, "latent_upscaler")


def test_ltxav_diffusion_plans_optional_keyframe_embedding() -> None:
    source_value = ltxav_22b_v25_diffusion()
    source_value.geometries["keyframes_abs_pos_embedding"] = g(
        (1, LTXAV_22B_V25_CONFIG.hidden_size)
    )

    plan = plan_ltxav_standalone_component(source_value, "diffusion")

    config = cast("LTXAVConfig", plan.component.config)
    assert config.use_keyframes_abs_pos_embedding is True
    assert set(plan.component.keys) == set(ltxav_layout(config))


def _ltxav_v25_vae_metadata() -> dict[str, Any]:
    config = LTXAV_22B_V25_VAE_CONFIG
    return {
        "vae": {
            "_class_name": "CausalDiffusionVAE",
            "model_output_type": "x0",
            "spatial_padding_mode": "zeros",
            "encoder": {
                "_class_name": "Encoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": config.latent_channels,
                "patch_size": config.patch_size,
                "latent_log_var": "constant",
                "norm_layer": "pixel_norm",
                "base_channels": config.base_channels,
                "spatial_padding_mode": "zeros",
            },
            "decoder": {
                "_class_name": "NADiffusionDecoder",
                "in_channels": config.latent_channels,
                "out_channels": config.output_channels,
                "patch_size": config.patch_size,
                "head_dim": config.head_dim,
                "stage_channels": list(config.stage_channels),
                "stage_depths": list(config.stage_depths),
                "stage_kernels": [list(value) for value in config.stage_kernels],
                "upsamples": [[list(stride), reduction] for stride, reduction in config.upsamples],
                "timestep_scale_multiplier": 1000.0,
                "default_num_inference_steps": 1,
            },
        }
    }


def test_ltxav_22b_v25_diffusion_video_vae_plans_strictly() -> None:
    geometries = geometrize(ltx_diffusion_video_vae_layout(LTXAV_22B_V25_VAE_CONFIG))
    geometries["decoder.type_emb"] = g((128,))
    vae = source(geometries, "ltx-2.4-video-vae.safetensors")
    vae.extra["config"] = json.dumps(_ltxav_v25_vae_metadata())

    plan = plan_ltxav_standalone_component(vae, "vae")

    assert plan.component.config is LTXAV_22B_V25_VAE_CONFIG
    assert set(plan.component.keys) == set(ltx_diffusion_video_vae_layout(LTXAV_22B_V25_VAE_CONFIG))
    assert plan.component.ignored == ("decoder.type_emb",)

    metadata = _ltxav_v25_vae_metadata()
    metadata["vae"]["decoder"]["default_num_inference_steps"] = 2
    vae.extra["config"] = json.dumps(metadata)
    with pytest.raises(AssemblyError, match="default_num_inference_steps must equal 1"):
        plan_ltxav_standalone_component(vae, "vae")


@pytest.mark.parametrize(
    ("key", "geometry", "match"),
    (
        ("decoder.type_emb", g((127,)), "decoder.type_emb expected bfloat16 shape"),
        ("decoder.type_emb", g((128,), FLOAT16), "decoder.type_emb expected bfloat16 shape"),
        ("stowaway.weight", g((1,)), "source does not match an official LTX-2 video VAE"),
    ),
)
def test_ltxav_22b_v25_diffusion_video_vae_refuses_foreign_extras(
    key: str,
    geometry: TensorGeometry,
    match: str,
) -> None:
    geometries = geometrize(ltx_diffusion_video_vae_layout(LTXAV_22B_V25_VAE_CONFIG))
    geometries[key] = geometry
    vae = source(geometries, "ltx-2.4-video-vae.safetensors")
    vae.extra["config"] = json.dumps(_ltxav_v25_vae_metadata())

    with pytest.raises(AssemblyError, match=match):
        plan_ltxav_standalone_component(vae, "vae")


def test_ltxav_22b_split_vae_and_diffusion_plan_independently() -> None:
    checkpoint = ltxav_22b_checkpoint()
    checkpoint_geometries = dict(checkpoint.geometries)
    diffusion_geometries = {
        key.removeprefix(FLUX_DIFFUSION_PREFIX): checkpoint_geometries.pop(key)
        for key in tuple(checkpoint_geometries)
        if key.startswith(FLUX_DIFFUSION_PREFIX)
    }
    vae_geometries = {
        key.removeprefix(FLUX_VAE_PREFIX): checkpoint_geometries.pop(key)
        for key in tuple(checkpoint_geometries)
        if key.startswith(FLUX_VAE_PREFIX)
    }
    checkpoint.geometries = checkpoint_geometries
    diffusion = source(diffusion_geometries, "ltx-2.3-22b-dit.safetensors")
    diffusion.extra["config"] = json.dumps(ltxav_22b_metadata())
    vae = source(vae_geometries, "ltx-2.3-vae.safetensors")
    vae.extra["config"] = json.dumps(ltxav_22b_metadata())

    diffusion_plan = plan_ltxav_standalone_component(diffusion, "diffusion")
    vae_plan = plan_ltxav_standalone_component(vae, "vae")

    assert diffusion_plan.component.path == diffusion.path
    assert vae_plan.component.path == vae.path


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("apply_gated_attention", False),
        ("connector_apply_gated_attention", False),
        ("caption_proj_before_connector", False),
        ("cross_attention_adaln", False),
        ("connector_num_layers", 2),
    ),
)
def test_ltxav_22b_transformer_metadata_fails_closed(name: str, value: object) -> None:
    checkpoint = ltxav_22b_checkpoint()
    metadata = json.loads(checkpoint.extra["config"])
    metadata["transformer"][name] = value
    checkpoint.extra["config"] = json.dumps(metadata)

    with pytest.raises(AssemblyError, match=name):
        plan_ltxav_standalone_component(checkpoint, "diffusion")


def test_ltxav_22b_vae_padding_metadata_fails_closed() -> None:
    checkpoint = ltxav_22b_checkpoint()
    metadata = json.loads(checkpoint.extra["config"])
    metadata["vae"]["spatial_padding_mode"] = "reflect"
    checkpoint.extra["config"] = json.dumps(metadata)

    with pytest.raises(AssemblyError, match="spatial_padding_mode must be 'zeros'"):
        plan_ltxav_standalone_component(checkpoint, "vae")


def test_ltxav_standalone_audio_codec_claims_both_components_exactly() -> None:
    source = ltxav_audio_codec_source()

    plan = plan_ltxav_audio_codec(source)

    assert isinstance(plan, LTXAVAudioCodecPlan)
    assert plan.family is LTXAV
    assert plan.audio_vae.path == source.path
    assert plan.vocoder.path == source.path
    assert set(plan.audio_vae.keys) == set(ltx_audio_vae_layout(LTXAV_19B_AUDIO_VAE_CONFIG))
    assert plan.vocoder.config is LTXAV_BWE_VOCODER_CONFIG
    assert set(plan.vocoder.keys) == set(ltx_vocoder_bwe_layout(LTXAV_BWE_VOCODER_CONFIG))
    assert plan.identity_components == (plan.audio_vae, plan.vocoder)
    assert plan.unclaimed == ()


def test_ltxav_standalone_audio_codec_refuses_incomplete_foreign_and_stowaway_sources() -> None:
    codec = ltxav_audio_codec_source()
    incomplete = dict(codec.geometries)
    del incomplete["audio_vae.encoder.conv_in.conv.weight"]
    missing = ltxav_audio_codec_source()
    missing.geometries = incomplete
    with pytest.raises(AssemblyError, match="missing encoder.conv_in.conv.weight"):
        plan_ltxav_audio_codec(missing)

    foreign = ltxav_audio_codec_source()
    foreign.geometries["audio_vae.encoder.conv_in.conv.weight"] = g((1,), INT64)
    with pytest.raises(AssemblyError, match="expected shape"):
        plan_ltxav_audio_codec(foreign)

    stowaway = ltxav_audio_codec_source()
    stowaway.geometries["unrelated.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unsupported LTX-2 tensors: unrelated.weight"):
        plan_ltxav_audio_codec(stowaway)

    malformed_metadata = ltxav_audio_codec_source()
    metadata = json.loads(malformed_metadata.extra["config"])
    metadata["vocoder"]["bwe"]["output_sampling_rate"] = 44100
    malformed_metadata.extra["config"] = json.dumps(metadata)
    with pytest.raises(AssemblyError, match="bwe.output_sampling_rate must equal 48000"):
        plan_ltxav_audio_codec(malformed_metadata)

    mismatched_window = ltxav_audio_codec_source()
    metadata = json.loads(mismatched_window.extra["config"])
    metadata["vocoder"]["bwe"]["win_size"] = 1024
    mismatched_window.extra["config"] = json.dumps(metadata)
    with pytest.raises(AssemblyError, match="bwe.win_size must equal 512"):
        plan_ltxav_audio_codec(mismatched_window)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    (
        ("rope_type", "interleaved", "transformer rope_type must be 'split'"),
        ("frequencies_precision", "float32", "frequencies_precision must be 'float64'"),
        ("positional_embedding_theta", 20000.0, "positional_embedding_theta must equal"),
        ("timestep_scale_multiplier", 100, "transformer timestep_scale_multiplier must equal"),
        (
            "positional_embedding_max_pos",
            [20, 2048, 1024],
            "positional_embedding_max_pos must equal",
        ),
        (
            "audio_positional_embedding_max_pos",
            [40],
            "audio_positional_embedding_max_pos must equal",
        ),
        ("causal_temporal_positioning", False, "causal_temporal_positioning must be true"),
        ("use_middle_indices_grid", None, "use_middle_indices_grid must be true"),
        ("timestep_scale_multiplier", True, "transformer timestep_scale_multiplier must equal"),
        # The head split leaves every weight shape unchanged, so only the
        # metadata pin can catch a checkpoint that varies it.
        ("attention_head_dim", 64, "transformer attention_head_dim must equal"),
        ("num_attention_heads", 64, "transformer num_attention_heads must equal"),
        ("audio_attention_head_dim", 128, "transformer audio_attention_head_dim must equal"),
        ("audio_num_attention_heads", 16, "transformer audio_num_attention_heads must equal"),
    ),
)
def test_ltxav_transformer_metadata_pins_fail_closed(name: str, value: Any, match: str) -> None:
    metadata = ltxav_metadata()
    if value is None:
        del metadata["transformer"][name]
    else:
        metadata["transformer"][name] = value
    with pytest.raises(AssemblyError, match=match):
        plan_ltxav_standalone_component(ltxav_checkpoint(metadata), "diffusion")


@pytest.mark.parametrize("value", (None, True, 0.0, -5.0, "1000", float("nan")))
def test_ltxav_av_ca_multiplier_never_falls_back_to_a_default(value: Any) -> None:
    metadata = ltxav_metadata()
    if value is None:
        del metadata["transformer"]["av_ca_timestep_scale_multiplier"]
    else:
        metadata["transformer"]["av_ca_timestep_scale_multiplier"] = value
    with pytest.raises(
        AssemblyError, match="av_ca_timestep_scale_multiplier must be a positive real number"
    ):
        plan_ltxav_standalone_component(ltxav_checkpoint(metadata), "diffusion")


@pytest.mark.parametrize(
    ("value", "match"),
    (
        (True, "vae timestep_conditioning must be false"),
        (None, "vae timestep_conditioning must be false"),
        ("section", "lacks the vae section"),
    ),
)
def test_ltxav_vae_metadata_gate_fails_closed(value: Any, match: str) -> None:
    metadata = ltxav_metadata()
    if value == "section":
        del metadata["vae"]
    elif value is None:
        del metadata["vae"]["timestep_conditioning"]
    else:
        metadata["vae"]["timestep_conditioning"] = value
    with pytest.raises(AssemblyError, match=match):
        plan_ltxav_standalone_component(ltxav_checkpoint(metadata), "vae")


def test_ltxav_av_ca_multiplier_is_sourced_from_metadata() -> None:
    metadata = ltxav_metadata()
    metadata["transformer"]["av_ca_timestep_scale_multiplier"] = 250.0
    plan = plan_ltxav_standalone_component(ltxav_checkpoint(metadata), "diffusion")
    config = cast("LTXAVConfig", plan.component.config)
    assert config.av_ca_timestep_scale_multiplier == 250.0
    assert config == replace(LTXAV_19B_CONFIG, av_ca_timestep_scale_multiplier=250.0)


def test_ltxav_config_metadata_is_required() -> None:
    checkpoint = ltxav_checkpoint()

    del checkpoint.extra["config"]
    with pytest.raises(AssemblyError, match="carries no config JSON"):
        plan_ltxav_standalone_component(checkpoint, "diffusion")

    # Malformed config metadata already refuses at detection, before the
    # planner's own metadata gates run.
    checkpoint.extra["config"] = "{not json"
    with pytest.raises(AssemblyError, match="not an official LTX-2 audio-video profile"):
        plan_ltxav_standalone_component(checkpoint, "diffusion")

    checkpoint.extra["config"] = json.dumps([1])
    with pytest.raises(AssemblyError, match="not an official LTX-2 audio-video profile"):
        plan_ltxav_standalone_component(checkpoint, "diffusion")

    checkpoint.extra["config"] = json.dumps({})
    with pytest.raises(AssemblyError, match="lacks the transformer section"):
        plan_ltxav_standalone_component(checkpoint, "diffusion")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (
            ("model", "params", "ddconfig", "causality_axis"),
            "width",
            "ddconfig causality_axis must be 'height'",
        ),
        (("model", "params", "ddconfig", "norm_type"), "group", "norm_type must be 'pixel'"),
        (
            ("model", "params", "ddconfig", "attn_resolutions"),
            [8],
            "attn_resolutions must be empty",
        ),
        (
            ("model", "params", "ddconfig", "attn_resolutions"),
            None,
            "attn_resolutions must be empty",
        ),
        (("model", "params", "ddconfig", "double_z"), False, "double_z must be true"),
        (
            ("preprocessing", "audio", "sampling_rate"),
            22050,
            "preprocessing audio.sampling_rate must equal",
        ),
        (("preprocessing", "stft", "hop_length"), 256, "preprocessing stft.hop_length must equal"),
        (
            ("preprocessing", "stft", "filter_length"),
            2048,
            "preprocessing stft.filter_length must equal",
        ),
        (
            ("preprocessing", "mel", "n_mel_channels"),
            80,
            "preprocessing mel.n_mel_channels must equal",
        ),
    ),
)
def test_ltxav_audio_vae_metadata_gates_fail_closed(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    metadata = ltxav_metadata()
    node = metadata["audio_vae"]
    for name in path[:-1]:
        node = node[name]
    if value is None:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    with pytest.raises(AssemblyError, match=match):
        plan_ltxav_audio_codec(ltxav_checkpoint(metadata))

    missing = ltxav_metadata()
    del missing["audio_vae"]
    with pytest.raises(AssemblyError, match="lacks the audio_vae.model.params.ddconfig section"):
        plan_ltxav_audio_codec(ltxav_checkpoint(missing))


@pytest.mark.parametrize(
    ("name", "value", "match"),
    (
        ("resblock", "2", "vocoder resblock must be '1'"),
        ("stereo", False, "vocoder stereo must be True"),
        ("upsample_initial_channel", 512, "upsample_initial_channel must equal"),
        ("upsample_rates", [8, 5, 2, 2, 2], "upsample_rates must equal"),
        ("upsample_kernel_sizes", [16, 15, 8, 4], "upsample_kernel_sizes must equal"),
        ("resblock_kernel_sizes", [3, 7], "resblock_kernel_sizes must equal"),
        (
            "resblock_dilation_sizes",
            [[1, 3, 5], [1, 3, 5]],
            "resblock_dilation_sizes must equal",
        ),
        ("activation", "snakebeta", "activation must equal 'snake'"),
        ("use_bias_at_final", False, "use_bias_at_final must equal True"),
        ("use_tanh_at_final", False, "use_tanh_at_final must equal True"),
        ("apply_final_activation", False, "apply_final_activation must equal True"),
        ("output_sample_rate", 48000, "output_sample_rate must equal None"),
    ),
)
def test_ltxav_vocoder_metadata_gate_fails_closed(name: str, value: Any, match: str) -> None:
    metadata = ltxav_metadata()
    metadata["vocoder"][name] = value
    with pytest.raises(AssemblyError, match=match):
        plan_ltxav_audio_codec(ltxav_checkpoint(metadata))

    missing = ltxav_metadata()
    del missing["vocoder"]
    with pytest.raises(AssemblyError, match="lacks the vocoder section"):
        plan_ltxav_audio_codec(ltxav_checkpoint(missing))


def test_ltxav_connectors_are_required_and_route_text_side() -> None:
    absent = ltxav_checkpoint_geometries()
    for key in ltxav_connector_geometries():
        del absent[FLUX_DIFFUSION_PREFIX + key]
    with pytest.raises(
        AssemblyError, match="must carry both LTX-2 text-embedding connector towers"
    ):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=absent), "connectors")

    variant = ltxav_checkpoint_geometries()
    variant[
        FLUX_DIFFUSION_PREFIX + "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias"
    ] = g((1920,))
    with pytest.raises(AssemblyError, match="connectors: unsupported .*_embeddings_connector"):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=variant), "connectors")

    incomplete = ltxav_checkpoint_geometries()
    del incomplete[FLUX_DIFFUSION_PREFIX + "video_embeddings_connector.learnable_registers"]
    with pytest.raises(
        AssemblyError,
        match="connectors: .*missing model.diffusion_model.video_embeddings_connector",
    ):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=incomplete), "connectors")


def test_identify_ltxav_text_source_matches_only_the_gemma_split() -> None:
    gemma = source(ltxav_gemma_geometries(), "gemma3-12b.safetensors")
    assert identify_ltxav_text_source(gemma) == "gemma3_12b"

    bundled = ltxav_gemma_geometries()
    bundled["vision_model.embeddings.patch_embedding.weight"] = g((16, 3, 14, 14))
    assert identify_ltxav_text_source(source(bundled, "gemma3-12b.safetensors")) == "gemma3_12b"

    gemma4 = source(ltxav_gemma4_geometries(), "gemma4-12b.safetensors")
    assert identify_ltxav_text_source(gemma4) == "gemma4_12b"

    assert identify_ltxav_text_source(ltxav_checkpoint()) is None
    assert identify_ltxav_text_source(source(t5_geometries(), "t5xxl.safetensors")) is None


def test_ltxav_diffusion_strict_loading_fails_closed() -> None:
    incomplete = ltxav_checkpoint_geometries()
    del incomplete[FLUX_DIFFUSION_PREFIX + "transformer_blocks.0.attn1.to_q.weight"]
    with pytest.raises(AssemblyError, match="missing transformer_blocks.0.attn1.to_q.weight"):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=incomplete), "diffusion")

    stowaway = ltxav_checkpoint_geometries()
    stowaway[FLUX_DIFFUSION_PREFIX + "stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unexpected stowaway.weight"):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=stowaway), "diffusion")

    misshaped = ltxav_checkpoint_geometries()
    misshaped[FLUX_DIFFUSION_PREFIX + "transformer_blocks.0.attn1.to_q.weight"] = g((4095, 4096))
    with pytest.raises(
        AssemblyError, match="transformer_blocks.0.attn1.to_q.weight expected shape"
    ):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=misshaped), "diffusion")

    integer = ltxav_checkpoint_geometries()
    integer[FLUX_DIFFUSION_PREFIX + "patchify_proj.weight"] = g((4096, 128), INT64)
    with pytest.raises(AssemblyError, match="requires floating-point storage, found int64"):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=integer), "diffusion")


def test_ltxav_gemma_and_tokenizer_fail_closed() -> None:
    no_spiece = ltxav_gemma_geometries()
    del no_spiece["spiece_model"]
    with pytest.raises(AssemblyError, match="exactly one uint8 spiece_model tensor"):
        plan_ltxav_standalone_component(source(no_spiece, "g.st"), "gemma3_12b")

    for bad_tokenizer in (g((4096,), FLOAT32), g((4096, 1), UINT8), g((0,), UINT8)):
        malformed = ltxav_gemma_geometries()
        malformed["spiece_model"] = bad_tokenizer
        with pytest.raises(AssemblyError, match="nonempty rank-1 uint8 tensor"):
            plan_ltxav_standalone_component(source(malformed, "g.st"), "gemma3_12b")

    stowaway = ltxav_gemma_geometries()
    stowaway["stowaway.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="gemma3_12b: source contains unsupported or duplicate"):
        plan_ltxav_standalone_component(source(stowaway, "g.st"), "gemma3_12b")

    incomplete = ltxav_gemma_geometries()
    del incomplete["model.norm.weight"]
    with pytest.raises(AssemblyError, match="gemma3_12b: .*missing norm.weight"):
        plan_ltxav_standalone_component(source(incomplete, "g.st"), "gemma3_12b")

    integer = ltxav_gemma_geometries()
    integer["model.norm.weight"] = g((3840,), INT64)
    with pytest.raises(
        AssemblyError, match="gemma3_12b: norm.weight requires floating-point storage"
    ):
        plan_ltxav_standalone_component(source(integer, "g.st"), "gemma3_12b")


def test_ltxav_gemma_source_ignores_the_bundled_vision_tower() -> None:
    bundled = ltxav_gemma_geometries()
    bundled["vision_model.embeddings.patch_embedding.weight"] = g((16, 3, 14, 14))
    bundled["vision_model.encoder.layers.0.self_attn.q_proj.weight"] = g((16, 16))
    bundled["multi_modal_projector.mm_input_projection_weight"] = g((16, 16))

    plan = plan_ltxav_standalone_component(source(bundled, "gemma3-12b.safetensors"), "gemma3_12b")
    assert set(plan.component.keys) == set(gemma_text_layout(GEMMA3_LTX_12B_CONFIG))
    assert not any(
        key.startswith(("vision_model.", "multi_modal_projector."))
        for component in plan.identity_components
        for key in component.keys.values()
    )


def test_ltxav_gemma4_source_owns_only_text_and_tokenizer() -> None:
    bundled = ltxav_gemma4_geometries(projection=True)
    bundled["vision_model.embeddings.patch_embedding.weight"] = g((16, 3, 14, 14))
    bundled["multi_modal_projector.mm_input_projection_weight"] = g((16, 16))
    bundled["audio_projector.input_proj.weight"] = g((16, 16))

    source_value = source(bundled, "gemma4-12b.safetensors")
    plan = plan_ltxav_standalone_component(source_value, "gemma4_12b")
    projection = plan_ltxav_standalone_component(source_value, "text_projection")

    assert plan.component.config is GEMMA4_LTX_12B_CONFIG
    assert plan.tokenizer_source_key == "tokenizer_json"
    assert set(plan.component.keys) == set(gemma_text_layout(GEMMA4_LTX_12B_CONFIG))
    assert projection.component.config == "dual_linear_gemma4"
    assert not any(
        key.startswith(("vision_model.", "multi_modal_projector.", "audio_projector."))
        for component in plan.identity_components
        for key in component.keys.values()
    )


def test_ltxav_text_projection_rules() -> None:
    single = plan_ltxav_standalone_component(
        source(ltxav_gemma_geometries(projection=True), "single.safetensors"),
        "text_projection",
    )
    dual = plan_ltxav_standalone_component(ltxav_22b_checkpoint(), "text_projection")
    assert single.component.config == "single_linear"
    assert single.component.path == Path("/fake/single.safetensors")
    assert dual.component.config == "dual_linear"

    metadata = ltxav_22b_checkpoint()
    metadata.extra["model_version"] = "2.4.0"
    metadata.extra["gemma_source_checkpoint"] = json.dumps(
        {"ltx_version": "2.4.0", "gemma_version": "gemma4-12b-ltx-v1"}
    )
    assert (
        plan_ltxav_standalone_component(metadata, "text_projection").component.config
        == "dual_linear_gemma4"
    )

    malformed = ltxav_22b_checkpoint()
    malformed.extra["model_version"] = "2.4.0"
    malformed.extra["gemma_source_checkpoint"] = "{}"
    with pytest.raises(AssemblyError, match="unsupported gemma_source_checkpoint"):
        plan_ltxav_standalone_component(malformed, "text_projection")

    misshaped = ltxav_checkpoint_geometries()
    misshaped["text_embedding_projection.aggregate_embed.weight"] = g((3840, 1000))
    with pytest.raises(
        AssemblyError, match="text_projection: .*single_linear text-projection layout"
    ):
        plan_ltxav_standalone_component(ltxav_checkpoint(geometries=misshaped), "text_projection")


def test_ltxav_profile_detection_fails_closed() -> None:
    with pytest.raises(AssemblyError, match="not an official LTX-2 audio-video profile"):
        plan_ltxav_standalone_component(
            source(geometrize(wan21_shapes("t2v-1.3b")), "wan.safetensors"),
            "diffusion",
        )


def test_ltxav_split_vae_and_diffusion_sources_plan_by_role() -> None:
    sd = ltxav_checkpoint_geometries()
    vae_sd = {
        key.removeprefix(FLUX_VAE_PREFIX): sd.pop(key)
        for key in tuple(sd)
        if key.startswith(FLUX_VAE_PREFIX)
    }
    vae = source(vae_sd, "ltx-2-vae.safetensors")
    vae.extra["config"] = json.dumps(ltxav_metadata())
    vae_plan = plan_ltxav_standalone_component(vae, "vae")
    diffusion_plan = plan_ltxav_standalone_component(ltxav_checkpoint(geometries=sd), "diffusion")
    assert vae_plan.component.path == Path("/fake/ltx-2-vae.safetensors")
    assert diffusion_plan.component.path == Path("/fake/ltx-2-19b-dev.safetensors")

    with pytest.raises(AssemblyError, match="does not match an official LTX-2 video VAE"):
        plan_ltxav_standalone_component(source(kl_geometries(), "kl.safetensors"), "vae")

    sd = ltxav_checkpoint_geometries()
    dit_sd = {
        key.removeprefix(FLUX_DIFFUSION_PREFIX): sd.pop(key)
        for key in tuple(sd)
        if key.startswith(FLUX_DIFFUSION_PREFIX)
    }
    dit = source(dit_sd, "ltx-2-dit.safetensors")
    dit.extra["config"] = json.dumps(ltxav_metadata())
    plan = plan_ltxav_standalone_component(dit, "diffusion")
    connectors = plan_ltxav_standalone_component(dit, "connectors")
    assert plan.component.path == Path("/fake/ltx-2-dit.safetensors")
    assert connectors.component.path == Path("/fake/ltx-2-dit.safetensors")
    assert plan.component.keys["patchify_proj.weight"] == "patchify_proj.weight"


def test_general_runtime_planner_refuses_ltx_component_loading() -> None:
    gemma = source(ltxav_gemma_geometries(), "gemma3-12b.safetensors")
    with pytest.raises(NativeRefusalError, match="do not match an executable architecture"):
        plan_native(checkpoint=ltxav_checkpoint(), gemma3_12b=gemma)

    with pytest.raises(NativeRefusalError, match="wires no Gemma 3 slot"):
        plan_native(**split_sources(), gemma3_12b=gemma)


def test_ltxav_standalone_plan_validator_pins_role() -> None:
    plan = plan_ltxav_standalone_component(ltxav_checkpoint(), "diffusion")
    with pytest.raises(ValueError, match="exact supported roles"):
        replace(plan, role=cast("Any", "vae"))


def test_combined_checkpoint_plans_all_components() -> None:
    checkpoint = source(combined_geometries(), "combined.safetensors")
    plan = plan_flux_assembly(checkpoint=checkpoint)
    assert plan.clip_l is not None and plan.t5xxl is not None
    assert plan.family is FLUX_DEV
    assert plan.diffusion.config == FLUX_DEV_CONFIG
    assert plan.clip_l.config == CLIP_L_TEXT_CONFIG
    assert plan.t5xxl.config == T5_XXL_CONFIG
    for component in (plan.diffusion, plan.clip_l, plan.t5xxl, plan.vae):
        assert component.path == checkpoint.path
        assert not component.quant
        assert set(component.keys) == set(component.dtypes)
    assert plan.diffusion.keys["img_in.weight"] == "model.diffusion_model.img_in.weight"
    assert plan.diffusion.dtypes["img_in.weight"] == BFLOAT16
    assert plan.clip_l.ignored == ("text_encoders.clip_l.logit_scale",)
    assert plan.t5xxl.ignored == ("text_encoders.t5xxl.logit_scale",)
    assert plan.clip_l.absent == ()
    assert plan.unclaimed == ()


def test_split_sources_plan() -> None:
    plan = plan_flux_assembly(**split_sources())
    assert plan.clip_l is not None and plan.t5xxl is not None
    assert plan.family is FLUX_DEV
    assert plan.diffusion.keys["img_in.weight"] == "img_in.weight"
    assert plan.clip_l.absent == ("text_projection.weight",)
    assert plan.t5xxl.ignored == ("encoder.embed_tokens.weight",)
    assert plan.unclaimed == ()


def test_flux_assembly_refuses_umt5_after_exact_text_detection() -> None:
    sources = split_sources()
    sources["t5xxl"] = source(
        geometrize(t5_layout(UMT5_XXL_CONFIG), FLOAT8_E4M3), "umt5.safetensors"
    )
    with pytest.raises(AssemblyError, match="classic T5-XXL"):
        plan_flux_assembly(**sources)


def test_schnell_family_detected() -> None:
    sources = split_sources()
    sources["diffusion"] = source(flux_geometries(FLUX_SCHNELL_CONFIG), "dit.safetensors")
    plan = plan_flux_assembly(**sources)
    assert plan.family is FLUX_SCHNELL
    assert plan.diffusion.config == FLUX_SCHNELL_CONFIG


@pytest.mark.parametrize("spelling", ["weight", "scale"])
def test_txt_norm_variant_uses_existing_flux_family_and_distinct_identity(
    spelling: str,
) -> None:
    classic = plan_flux_assembly(**split_sources())
    sources = split_sources()
    config = replace(FLUX_DEV_CONFIG, txt_norm=True)
    geometries = flux_geometries(config)
    if spelling == "scale":
        geometries["txt_norm.scale"] = geometries.pop("txt_norm.weight")
    sources["diffusion"] = source(geometries, "dit.safetensors")
    normalized = plan_flux_assembly(**sources)
    assert normalized.family is FLUX_DEV
    assert normalized.diffusion.config == config
    assert set(normalized.diffusion.keys) - set(classic.diffusion.keys) == {"txt_norm.weight"}
    assert normalized.diffusion.keys["txt_norm.weight"] == f"txt_norm.{spelling}"
    assert runtime_component_identity(
        FLUX_DEV.id, normalized.identity_components
    ) != runtime_component_identity(FLUX_DEV.id, classic.identity_components)


def test_gated_ovis_variant_uses_existing_flux_family_and_distinct_identity() -> None:
    classic = plan_flux_assembly(**split_sources())
    sources = split_sources()
    config = replace(FLUX_DEV_CONFIG, yak_mlp=True)
    sources["diffusion"] = source(flux_geometries(config), "ovis.safetensors")
    ovis = plan_flux_assembly(**sources)
    assert ovis.family is FLUX_DEV
    assert ovis.diffusion.config == config
    assert "double_blocks.0.img_mlp.gate_proj.weight" in ovis.diffusion.keys
    assert "double_blocks.0.img_mlp.0.weight" not in ovis.diffusion.keys
    assert runtime_component_identity(
        FLUX_DEV.id, ovis.identity_components
    ) != runtime_component_identity(FLUX_DEV.id, classic.identity_components)


def ovis_text_sources() -> dict[str, WeightSource]:
    sources = split_sources()
    del sources["clip_l"]
    del sources["t5xxl"]
    sources["diffusion"] = source(
        flux_geometries(replace(FLUX_DEV_CONFIG, context_in_dim=2048)),
        "context-2048.safetensors",
    )
    text = prefixed(geometrize(qwen_text_layout(OVIS_QWEN3_2B_CONFIG)), "model.")
    text.update(
        {
            "vision_model.encoder.layers.0.weight": g((1,)),
            "visual_tokenizer.head.weight": g((1,)),
            "vte.weight": g((65536, 2048)),
        }
    )
    sources["qwen3_2b"] = source(text, "ovis_2.5.safetensors")
    return sources


OVIS_VECTOR_FREE_CONFIG = replace(
    FLUX_SCHNELL_CONFIG,
    vec_in_dim=None,
    context_in_dim=2048,
    depth=6,
    depth_single_blocks=27,
    txt_norm=True,
    yak_mlp=True,
    txt_ids_dims=(1, 2),
)


def vector_free_ovis_sources() -> dict[str, WeightSource]:
    sources = ovis_text_sources()
    sources["diffusion"] = source(
        flux_geometries(OVIS_VECTOR_FREE_CONFIG),
        "ovis_image_bf16.safetensors",
    )
    return sources


def test_vector_free_ovis_plans_exact_components_and_distinct_identity() -> None:
    sources = vector_free_ovis_sources()
    capability = probe_native(
        diffusion=sources["diffusion"],
        qwen3_2b=sources["qwen3_2b"],
        vae=sources["vae"],
    )
    assert capability.native
    assert capability.family_id == FLUX_SCHNELL.id
    plan = plan_flux_assembly(**sources)
    classic = plan_flux_assembly(**split_sources())
    assert plan.family is FLUX_SCHNELL
    assert plan.diffusion.config == OVIS_VECTOR_FREE_CONFIG
    assert len(plan.diffusion.keys) == 397
    assert plan.qwen3_2b is not None
    assert plan.clip_l is None and plan.t5xxl is None
    assert not any(key.startswith("vector_in.") for key in plan.diffusion.keys)
    assert not any(key.startswith("guidance_in.") for key in plan.diffusion.keys)
    assert runtime_component_identity(
        plan.family.id, plan.identity_components
    ) != runtime_component_identity(classic.family.id, classic.identity_components)
    assert build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    ) == (
        "native:dinkster.flux_schnell:"
        "b85f7d5686c0b91760e112973d9682ed39b90dbf3aa00a3dac018bf1eb3541af"
    )


def test_vector_free_ovis_requires_qwen_text_component() -> None:
    sources = split_sources()
    sources["diffusion"] = source(
        flux_geometries(OVIS_VECTOR_FREE_CONFIG),
        "ovis_image_bf16.safetensors",
    )
    with pytest.raises(AssemblyError, match="vector-free Ovis.*qwen3_2b"):
        plan_flux_assembly(**sources)
    plan = plan_flux_assembly(**vector_free_ovis_sources())
    with pytest.raises(ValueError, match="vector-free Ovis.*qwen3_2b"):
        replace(plan, qwen3_2b=None)


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_OVIS_SET),
    reason="digest-verified Ovis artifact set absent",
)
def test_real_digest_pinned_vector_free_ovis_header_probes_native() -> None:
    manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    workload = next(
        item for item in manifest["workloads"] if item["id"] == "W0-FLUX-GATED-OVIS-TXT2IMG"
    )
    artifacts = {item["role"]: item for item in workload["artifacts"]}
    expected = {
        "ovis-vector-free-gated-txt-norm-flux-diffusion": (
            REAL_OVIS_DIFFUSION,
            14740943536,
            "sha256:eb3d9e1b201412b3b527472cf82a2c5add6b5a40a37e94d99f11c381f18a9e2b",
        ),
        "ovis-qwen3-2b-text-encoder": (
            REAL_OVIS_TEXT,
            5140950080,
            "sha256:f453ee5e7a25cb23cf2adf7aae3e5b405f22097cb67f2cfcca029688cb3f740d",
        ),
        "flux-16-channel-kl-codec": (
            REAL_OVIS_AE,
            335304388,
            "sha256:afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        ),
    }
    assert set(artifacts) == set(expected)
    for role, (path, size, digest) in expected.items():
        assert artifacts[role]["path"] == str(path)
        assert artifacts[role]["bytes"] == size
        assert artifacts[role]["digest"] == digest
        assert path.stat().st_size == size

    capability = probe_native(
        diffusion=load_safetensors_header(REAL_OVIS_DIFFUSION),
        qwen3_2b=load_safetensors_header(REAL_OVIS_TEXT),
        vae=load_safetensors_header(REAL_OVIS_AE),
    )
    assert capability.native
    assert capability.family_id == FLUX_SCHNELL.id
    assert capability.reasons == ()


def test_split_ovis_text_artifact_extracts_exact_model_role() -> None:
    plan = plan_flux_assembly(**ovis_text_sources())
    assert plan.qwen3_2b is not None
    assert plan.clip_l is None and plan.t5xxl is None
    assert plan.qwen3_2b.config is OVIS_QWEN3_2B_CONFIG
    assert len(plan.qwen3_2b.keys) == 310
    assert plan.qwen3_2b.keys["embed_tokens.weight"] == "model.embed_tokens.weight"
    assert set(plan.qwen3_2b.ignored) == {
        "vision_model.encoder.layers.0.weight",
        "visual_tokenizer.head.weight",
        "vte.weight",
    }
    assert plan.unclaimed == ()


def test_combined_ovis_text_slot_is_detected_without_split_hint() -> None:
    split = ovis_text_sources()
    diffusion = split["diffusion"]
    qwen3_2b = split["qwen3_2b"]
    vae = split["vae"]
    combined: dict[str, TensorGeometry] = {}
    combined.update(
        {FLUX_DIFFUSION_PREFIX + key: diffusion.entry(key).geometry for key in diffusion.keys()}
    )
    for key in qwen3_2b.keys():
        prefix = (
            "text_encoders.qwen3_2b.transformer."
            if key.startswith("model.")
            else "text_encoders.qwen3_2b."
        )
        combined[prefix + key] = qwen3_2b.entry(key).geometry
    combined["text_encoders.qwen3_2b.logit_scale"] = g(())
    combined.update({FLUX_VAE_PREFIX + key: vae.entry(key).geometry for key in vae.keys()})
    plan = plan_flux_assembly(checkpoint=source(combined, "combined-ovis.safetensors"))
    assert plan.qwen3_2b is not None
    assert plan.clip_l is None and plan.t5xxl is None
    assert plan.qwen3_2b.keys["embed_tokens.weight"] == (FLUX_QWEN_PREFIX + "embed_tokens.weight")
    assert set(plan.qwen3_2b.ignored) == {
        "text_encoders.qwen3_2b.logit_scale",
        "text_encoders.qwen3_2b.vision_model.encoder.layers.0.weight",
        "text_encoders.qwen3_2b.visual_tokenizer.head.weight",
        "text_encoders.qwen3_2b.vte.weight",
    }
    assert plan.unclaimed == ()


def test_ovis_ignored_role_roots_do_not_change_structural_identity() -> None:
    mixed = plan_flux_assembly(**ovis_text_sources())
    sources = ovis_text_sources()
    qwen3_2b = sources["qwen3_2b"]
    text_only = {
        key: qwen3_2b.entry(key).geometry for key in qwen3_2b.keys() if key.startswith("model.")
    }
    sources["qwen3_2b"] = source(text_only, "ovis-text-only.safetensors")
    isolated = plan_flux_assembly(**sources)
    assert runtime_component_identity(
        mixed.family.id, mixed.identity_components
    ) == runtime_component_identity(isolated.family.id, isolated.identity_components)


def test_ovis_text_has_distinct_pinned_structural_identity() -> None:
    ovis = plan_flux_assembly(**ovis_text_sources())
    classic = plan_flux_assembly(**split_sources())
    assert runtime_component_identity(
        ovis.family.id, ovis.identity_components
    ) != runtime_component_identity(classic.family.id, classic.identity_components)
    assert build_runtime_identity(
        ovis.family.id,
        ovis.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    ) == (
        "native:dinkster.flux_dev:a471f4aa3e5cf6098fe36a4f620cc140b82231d734c3ebe0296a5ccf69781a6c"
    )


def test_ovis_text_refuses_classic_slot_mixing() -> None:
    sources = ovis_text_sources()
    sources["clip_l"] = source(clip_geometries(), "clip_l.safetensors")
    with pytest.raises(AssemblyError, match="alternative component slot"):
        plan_flux_assembly(**sources)


def test_ovis_text_refuses_unknown_artifact_role() -> None:
    sources = ovis_text_sources()
    qwen3_2b = sources["qwen3_2b"]
    assert isinstance(qwen3_2b, FakeSource)
    qwen3_2b.geometries["audio_model.weight"] = g((1,))
    with pytest.raises(AssemblyError, match="unexpected non-text artifact roots.*audio"):
        plan_flux_assembly(**sources)


def test_ovis_text_does_not_admit_non_approved_vector_free_geometry() -> None:
    sources = ovis_text_sources()
    diffusion = sources["diffusion"]
    assert isinstance(diffusion, FakeSource)
    for key in tuple(diffusion.geometries):
        if key.startswith(("vector_in.", "guidance_in.")):
            del diffusion.geometries[key]
    with pytest.raises(AssemblyError, match="only the approved Ovis"):
        plan_flux_assembly(**sources)


def test_split_overrides_combined() -> None:
    checkpoint = source(combined_geometries(), "combined.safetensors")
    dit = source(flux_geometries(), "dit.safetensors")
    plan = plan_flux_assembly(checkpoint=checkpoint, diffusion=dit)
    assert plan.diffusion.path == dit.path
    assert plan.vae.path == checkpoint.path
    assert set(plan.unclaimed) == {
        key for key in checkpoint.geometries if key.startswith(FLUX_DIFFUSION_PREFIX)
    }


def test_prefixed_split_diffusion_file() -> None:
    sources = split_sources()
    sources["diffusion"] = source(
        prefixed(flux_geometries(), FLUX_DIFFUSION_PREFIX), "dit.safetensors"
    )
    plan = plan_flux_assembly(**sources)
    assert plan.diffusion.keys["img_in.weight"] == "model.diffusion_model.img_in.weight"


def test_stray_checkpoint_key_lands_in_unclaimed() -> None:
    sd = combined_geometries()
    sd["first_stage_model.leftover"] = g((4,))
    plan = plan_flux_assembly(checkpoint=source(sd))
    assert plan.unclaimed == ("first_stage_model.leftover",)


def test_bare_scale_spelling_maps_to_model_weight() -> None:
    sd = {
        (key[: -len(".weight")] + ".scale" if key.endswith("_norm.weight") else key): value
        for key, value in flux_geometries().items()
    }
    sources = split_sources()
    sources["diffusion"] = source(sd, "dit.safetensors")
    plan = plan_flux_assembly(**sources)
    key = "double_blocks.0.img_attn.norm.query_norm.weight"
    assert plan.diffusion.keys[key] == "double_blocks.0.img_attn.norm.query_norm.scale"
    assert key in plan.diffusion.dtypes


# --------------------------------------------------- quantized layouts


def quantize_legacy(
    sd: dict[str, TensorGeometry], layer: str, prefix: str = ""
) -> dict[str, TensorGeometry]:
    """Mark ``sd`` (already prefixed with ``prefix``) as legacy
    scaled-fp8 with a single quantized ``layer``."""
    weight = f"{prefix}{layer}.weight"
    sd[weight] = TensorGeometry(sd[weight].shape, FLOAT8_E4M3)
    sd[f"{prefix}{layer}.scale_weight"] = g((), FLOAT32)
    sd[f"{prefix}{layer}.scale_input"] = g((), FLOAT32)
    sd[f"{prefix}scaled_fp8"] = g((0,), FLOAT32)
    return sd


def test_combined_legacy_scaled_fp8_diffusion() -> None:
    sd = combined_geometries()
    quantize_legacy(sd, "img_in", FLUX_DIFFUSION_PREFIX)
    plan = plan_flux_assembly(checkpoint=source(sd))
    assert plan.clip_l is not None and plan.t5xxl is not None
    assert set(plan.diffusion.quant) == {"img_in"}
    layer = plan.diffusion.quant["img_in"]
    assert layer.format == "float8_e4m3fn"
    assert layer.weight == "model.diffusion_model.img_in.weight"
    assert layer.weight_scale == "model.diffusion_model.img_in.scale_weight"
    assert layer.input_scale == "model.diffusion_model.img_in.scale_input"
    assert plan.diffusion.dtypes["img_in.weight"] == FLOAT8_E4M3
    assert not plan.clip_l.quant and not plan.t5xxl.quant and not plan.vae.quant
    assert plan.unclaimed == ()


def test_combined_text_encoder_legacy_marker_is_scoped() -> None:
    sd = combined_geometries()
    quantize_legacy(sd, "encoder.block.0.layer.0.SelfAttention.q", FLUX_T5XXL_PREFIX)
    plan = plan_flux_assembly(checkpoint=source(sd))
    assert plan.t5xxl is not None
    assert set(plan.t5xxl.quant) == {"encoder.block.0.layer.0.SelfAttention.q"}
    layer = plan.t5xxl.quant["encoder.block.0.layer.0.SelfAttention.q"]
    assert layer.weight_scale == (
        FLUX_T5XXL_PREFIX + "encoder.block.0.layer.0.SelfAttention.q.scale_weight"
    )
    assert not plan.diffusion.quant


def test_combined_metadata_quantization_names_prefixed_layers() -> None:
    sd = combined_geometries()
    weight = FLUX_DIFFUSION_PREFIX + "img_in.weight"
    sd[weight] = TensorGeometry(sd[weight].shape, FLOAT8_E4M3)
    sd[FLUX_DIFFUSION_PREFIX + "img_in.weight_scale"] = g((), FLOAT32)
    sd[FLUX_DIFFUSION_PREFIX + "img_in.input_scale"] = g((), FLOAT32)
    metadata = {
        "_quantization_metadata": json.dumps(
            {"layers": {FLUX_DIFFUSION_PREFIX + "img_in": {"format": "float8_e4m3fn"}}}
        )
    }
    checkpoint = FakeSource(Path("/fake/combined.safetensors"), sd, metadata)
    plan = plan_flux_assembly(checkpoint=checkpoint)
    assert plan.clip_l is not None and plan.t5xxl is not None
    layer = plan.diffusion.quant["img_in"]
    assert layer.format == "float8_e4m3fn"
    assert layer.weight_scale == FLUX_DIFFUSION_PREFIX + "img_in.weight_scale"
    assert layer.input_scale == FLUX_DIFFUSION_PREFIX + "img_in.input_scale"
    assert not plan.clip_l.quant and not plan.t5xxl.quant and not plan.vae.quant


# -------------------------------------------------------------- refusals


def test_no_sources_refuses() -> None:
    with pytest.raises(AssemblyError, match="no sources"):
        plan_flux_assembly()


def test_missing_component_refuses_by_name() -> None:
    sd = {
        key: value
        for key, value in combined_geometries().items()
        if not key.startswith(FLUX_VAE_PREFIX)
    }
    with pytest.raises(AssemblyError, match="vae"):
        plan_flux_assembly(checkpoint=source(sd))


def test_empty_split_source_refuses() -> None:
    sources = split_sources()
    sources["diffusion"] = source({}, "empty.safetensors")
    with pytest.raises(AssemblyError, match="diffusion"):
        plan_flux_assembly(**sources)


def test_unrecognized_diffusion_refuses_with_component_context() -> None:
    sources = split_sources()
    sources["diffusion"] = source({"blocks.0.weight": g((8, 8))}, "other.safetensors")
    with pytest.raises(AssemblyError, match="diffusion"):
        plan_flux_assembly(**sources)


def test_nvfp4_metadata_plans_logical_geometry_and_mixed_fp8() -> None:
    sd = flux_geometries()
    sd["img_in.weight"] = g((3072, 32), UINT8)
    sd["img_in.weight_scale"] = g((3072, 4), FLOAT8_E4M3)
    sd["img_in.weight_scale_2"] = g((), FLOAT32)
    sd["img_in.input_scale"] = g((), FLOAT32)
    sd["img_in.pre_quant_scale"] = g((64,), FLOAT16)
    sd["time_in.in_layer.weight"] = g((3072, 256), FLOAT8_E4M3)
    sd["time_in.in_layer.weight_scale"] = g((), FLOAT32)
    sd["time_in.in_layer.input_scale"] = g((), FLOAT32)
    sources = split_sources()
    diffusion = source(sd, "nvfp4.safetensors")
    diffusion.extra["_quantization_metadata"] = json.dumps(
        {
            "layers": {
                "img_in": {"format": "nvfp4"},
                "time_in.in_layer": {"format": "float8_e4m3fn"},
            }
        }
    )
    sources["diffusion"] = diffusion
    plan = plan_flux_assembly(**sources)
    assert plan.family is FLUX_DEV
    assert plan.diffusion.dtypes["img_in.weight"] == UINT8
    assert plan.diffusion.keys["img_in.weight"] == "img_in.weight"
    nvfp4 = plan.diffusion.quant["img_in"]
    assert nvfp4.weight_scale_2 == "img_in.weight_scale_2"
    assert nvfp4.pre_quant_scale == "img_in.pre_quant_scale"
    assert plan.diffusion.quant["time_in.in_layer"].format == "float8_e4m3fn"


def _payload_nvfp4_source(*, input_scale: bool = True) -> FakeSource:
    sd = flux_geometries()
    payloads: dict[str, bytes] = {}
    for layer, logical in (("img_in", (3072, 64)), ("txt_in", (3072, 4096))):
        rows, columns = logical
        sd[f"{layer}.weight"] = g((rows, columns // 2), UINT8)
        sd[f"{layer}.weight_scale"] = g((3072, ((columns // 16 + 3) // 4) * 4), FLOAT8_E4M3)
        sd[f"{layer}.weight_scale_2"] = g((), FLOAT32)
        if input_scale:
            sd[f"{layer}.input_scale"] = g((), FLOAT32)
        config = f"{layer}.comfy_quant"
        raw = b'{"format":"nvfp4","full_precision_matrix_mult":false}'
        sd[config] = g((len(raw),), UINT8)
        payloads[config] = raw
    return source(sd, "payload-nvfp4.safetensors", payload_values=payloads)


def _real_payload_nvfp4_source(
    tmp_path: Path,
    *,
    input_scale: bool = True,
    payload: bytes | None = None,
    metadata: dict[str, str] | None = None,
) -> SafetensorsSource:
    """Write a sparse but structurally complete safetensors fixture."""
    fake = _payload_nvfp4_source(input_scale=input_scale)
    if payload is not None:
        fake.payload_values["img_in.comfy_quant"] = payload
        fake.geometries["img_in.comfy_quant"] = g((len(payload),), UINT8)
    return _write_real_source(tmp_path, fake, metadata=metadata)


def _write_real_source(
    tmp_path: Path,
    fake: FakeSource,
    *,
    metadata: dict[str, str] | None = None,
    filename: str = "payload-nvfp4.safetensors",
) -> SafetensorsSource:
    if os.name == "nt":
        pytest.skip("large sparse safetensors fixtures require POSIX sparse-file semantics")
    dtype_names = {
        "bfloat16": "BF16",
        "float16": "F16",
        "float32": "F32",
        "float8_e4m3fn": "F8_E4M3",
        "uint8": "U8",
    }
    header: dict[str, object] = {"__metadata__": metadata or {}}
    offset = 0
    payload_offsets: dict[str, tuple[int, bytes]] = {}
    for key, geometry in fake.geometries.items():
        end = offset + geometry.nbytes
        header[key] = {
            "dtype": dtype_names[geometry.dtype.name],
            "shape": list(geometry.shape),
            "data_offsets": [offset, end],
        }
        if key in fake.payload_values:
            payload_offsets[key] = (offset, fake.payload_values[key])
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / filename
    data_start = 8 + len(encoded)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.truncate(data_start + offset)
        for payload_offset, raw in payload_offsets.values():
            handle.seek(data_start + payload_offset)
            handle.write(raw)
    return load_safetensors_header(path)


def test_plan_native_normalizes_real_payload_source_before_family_detection(
    tmp_path: Path,
) -> None:
    sources = split_sources()
    diffusion = _real_payload_nvfp4_source(tmp_path, input_scale=False)
    sources["diffusion"] = diffusion
    plan = plan_native(**sources)  # pyright: ignore[reportArgumentType]
    assert isinstance(plan, FluxAssemblyPlan)
    direct = plan_flux_assembly(**sources)
    assert plan == direct
    assert plan.family is FLUX_DEV
    assert diffusion.entry("img_in.weight").geometry.shape == (3072, 32)
    assert plan.diffusion.dtypes["img_in.weight"] == UINT8
    assert plan.diffusion.quant["img_in"].input_scale is None
    # The packed physical width normalizes to logical width 64; Flux's
    # patch_size=2 then derives 64 / 4 = 16 latent input channels.
    assert plan.diffusion.config.in_channels == 16


def test_plan_native_real_payload_requires_one_registered_normalized_family(
    tmp_path: Path,
) -> None:
    sources = split_sources()
    sources["diffusion"] = _real_payload_nvfp4_source(tmp_path)
    registry = FamilyRegistry()
    registry.register(FLUX_DEV)
    plan = plan_native(
        **sources,  # pyright: ignore[reportArgumentType]
        registry=registry,
    )
    assert plan.family is FLUX_DEV


def test_plan_native_real_non_quantized_source_is_unchanged(tmp_path: Path) -> None:
    sources = split_sources()
    plain = source(flux_geometries(), "plain-flux.safetensors")
    sources["diffusion"] = _write_real_source(tmp_path, plain, filename="plain-flux.safetensors")
    assert plan_native(
        **sources  # pyright: ignore[reportArgumentType]
    ) == plan_flux_assembly(**sources)


def test_plan_native_zero_label_match_keeps_valid_non_quantized_geometry(
    tmp_path: Path,
) -> None:
    class NoMatchDetector:
        def detect(self, source: WeightSource) -> None:
            del source
            return None

    sources = split_sources()
    plain = source(flux_geometries(), "plain-flux-zero-match.safetensors")
    sources["diffusion"] = _write_real_source(
        tmp_path, plain, filename="plain-flux-zero-match.safetensors"
    )
    registry = FamilyRegistry()
    registry.register(replace(FLUX_DEV, detector=NoMatchDetector()))
    assert plan_native(
        **sources,  # pyright: ignore[reportArgumentType]
        registry=registry,
    ) == plan_flux_assembly(**sources)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b"not-json", "malformed"),
        (b'{"format":"nvfp4","format":"nvfp4"}', "duplicate"),
        (b'{"format":"nvfp4","surprise":true}', "unexpected"),
        (b'{"format":"float8_e4m3fn"}', "does not match"),
    ),
)
def test_plan_native_real_payload_refusals(tmp_path: Path, raw: bytes, message: str) -> None:
    sources = split_sources()
    sources["diffusion"] = _real_payload_nvfp4_source(tmp_path, payload=raw)
    with pytest.raises(NativeRefusalError, match=message) as caught:
        plan_native(**sources)  # pyright: ignore[reportArgumentType]
    assert caught.value.category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY


def test_plan_native_explicit_metadata_precedes_disagreeing_real_payload(
    tmp_path: Path,
) -> None:
    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "layers": {
                    "img_in": {"format": "nvfp4"},
                    "txt_in": {"format": "nvfp4"},
                }
            }
        )
    }
    sources = split_sources()
    sources["diffusion"] = _real_payload_nvfp4_source(
        tmp_path, payload=b'{"format":"float8_e4m3fn"}', metadata=metadata
    )
    plan = plan_native(**sources)  # pyright: ignore[reportArgumentType]
    assert plan.family is FLUX_DEV
    assert plan.diffusion.quant["img_in"].format == "nvfp4"
    assert plan.diffusion.quant["img_in"].config is None


def test_payload_only_nvfp4_plans_two_layers_and_optional_input_scale() -> None:
    sources = split_sources()
    sources["diffusion"] = _payload_nvfp4_source(input_scale=False)
    plan = plan_flux_assembly(**sources)
    assert plan.family is FLUX_DEV
    assert set(plan.diffusion.quant) >= {"img_in", "txt_in"}
    assert plan.diffusion.quant["img_in"].format == "nvfp4"
    assert plan.diffusion.quant["txt_in"].config == "txt_in.comfy_quant"
    assert plan.diffusion.quant["img_in"].input_scale is None


def test_nvfp4_metadata_precedes_payload_configuration() -> None:
    diffusion = _payload_nvfp4_source()
    diffusion.extra["_quantization_metadata"] = json.dumps(
        {
            "layers": {
                layer: {"format": "nvfp4", "full_precision_matrix_mult": True}
                for layer in ("img_in", "txt_in")
            }
        }
    )
    diffusion.payload_values = {key: b"not json" for key in diffusion.payload_values}
    sources = split_sources()
    sources["diffusion"] = diffusion
    plan = plan_flux_assembly(**sources)
    assert all(plan.diffusion.quant[layer].full_precision_matmul for layer in ("img_in", "txt_in"))


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"format":"nvfp4","format":"nvfp4"}', "duplicate"),
        (b"not-json", "malformed"),
        (b'{"format":"nvfp4","full_precision_matrix_mult":1}', "must be a bool"),
        (b'{"format":"nvfp4","surprise":true}', "unexpected"),
    ),
)
def test_payload_nvfp4_refuses_duplicate_malformed_nonbool_and_unexpected_config(
    raw: bytes, message: str
) -> None:
    diffusion = _payload_nvfp4_source()
    diffusion.payload_values["img_in.comfy_quant"] = raw
    sources = split_sources()
    sources["diffusion"] = diffusion
    with pytest.raises(AssemblyError, match=message):
        plan_flux_assembly(**sources)


def test_typed_unsupported_quant_refuses_before_runtime_or_provider_selection() -> None:
    diffusion = source(flux_geometries(), "typed-mxfp8.safetensors")
    layers = {"img_in": (3072, 64), "txt_in": (3072, 4096)}
    for layer, (rows, columns) in layers.items():
        diffusion.geometries[f"{layer}.weight"] = g((rows, columns), FLOAT8_E4M3)
        diffusion.geometries[f"{layer}.weight_scale"] = g(
            (((rows + 127) // 128) * 128, ((columns // 32 + 3) // 4) * 4),
            FLOAT8_E8M0,
        )
    diffusion.extra["_quantization_metadata"] = json.dumps(
        {"layers": {layer: {"format": "mxfp8"} for layer in layers}}
    )
    sources = split_sources()
    sources["diffusion"] = diffusion
    with pytest.raises(
        AssemblyError,
        match="typed quantization format.*mxfp8.*no runtime implementation",
    ):
        plan_flux_assembly(**sources)


def test_source_without_path_refuses() -> None:
    sources = split_sources()
    sources["diffusion"] = PathlessSource(None, flux_geometries())
    with pytest.raises(AssemblyError, match="no file path"):
        plan_flux_assembly(**sources)


# ------------------------------------------------------ plan invariants


def test_component_plan_mappings_are_immutable() -> None:
    plan = plan_flux_assembly(**split_sources())
    for mapping in (
        plan.diffusion.keys,
        plan.diffusion.dtypes,
        plan.diffusion.quant,
        plan.diffusion.transforms,
    ):
        assert isinstance(mapping, MappingProxyType)


def test_component_plan_rejects_keys_dtypes_mismatch() -> None:
    with pytest.raises(ValueError, match="same model keys"):
        ComponentPlan(
            component="diffusion",
            path=Path("/fake"),
            config=None,
            keys={"a.weight": "a.weight"},
            dtypes={},
            quant={},
        )


def test_component_plan_rejects_quant_layer_without_weight() -> None:
    from dinkster_inference import LayerQuant

    with pytest.raises(ValueError, match="no mapped weight"):
        ComponentPlan(
            component="diffusion",
            path=Path("/fake"),
            config=None,
            keys={"a.weight": "a.weight"},
            dtypes={"a.weight": BFLOAT16},
            quant={
                "b": LayerQuant(
                    layer="b",
                    format="float8_e4m3fn",
                    weight="b.weight",
                    weight_scale="b.weight_scale",
                )
            },
        )


# ---------------------------------------------------------- real headers


@pytest.mark.skipif(not REAL_COMBINED_FP8.exists(), reason="combined flux1-dev-fp8 absent")
def test_real_combined_flux_dev_fp8() -> None:
    plan = plan_flux_assembly(checkpoint=load_safetensors_header(REAL_COMBINED_FP8))
    assert plan.clip_l is not None and plan.t5xxl is not None
    assert plan.family is FLUX_DEV
    assert len(plan.diffusion.keys) == 780
    assert all(dtype == FLOAT8_E4M3 for dtype in plan.diffusion.dtypes.values())
    assert len(plan.clip_l.keys) == 197
    assert all(dtype == FLOAT16 for dtype in plan.clip_l.dtypes.values())
    assert len(plan.t5xxl.keys) == 219
    assert all(dtype == FLOAT8_E4M3 for dtype in plan.t5xxl.dtypes.values())
    assert len(plan.vae.keys) == 244
    for component in (plan.diffusion, plan.clip_l, plan.t5xxl, plan.vae):
        assert not component.quant
    assert plan.clip_l.ignored == ("text_encoders.clip_l.logit_scale",)
    assert plan.t5xxl.ignored == ("text_encoders.t5xxl.logit_scale",)
    assert plan.unclaimed == ()


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_SPLIT_SET), reason="split checkpoint set absent"
)
def test_real_split_assembly() -> None:
    plan = plan_flux_assembly(
        diffusion=load_safetensors_header(REAL_SPLIT_DIT_FP8),
        clip_l=load_safetensors_header(REAL_CLIP_L),
        t5xxl=load_safetensors_header(REAL_T5_FP16),
        vae=load_safetensors_header(REAL_AE),
    )
    assert plan.clip_l is not None and plan.t5xxl is not None
    assert plan.family is FLUX_DEV
    assert len(plan.diffusion.quant) == 266
    assert all(layer.format == "float8_e4m3fn" for layer in plan.diffusion.quant.values())
    assert all(layer.input_scale is not None for layer in plan.diffusion.quant.values())
    assert plan.clip_l.absent == ("text_projection.weight",)
    assert plan.t5xxl.ignored == ("encoder.embed_tokens.weight",)
    assert all(dtype == FLOAT16 for dtype in plan.t5xxl.dtypes.values())
    assert plan.unclaimed == ()


@pytest.mark.skipif(not REAL_FLUX2.exists(), reason="flux2-dev checkpoint absent")
def test_real_flux2_refuses() -> None:
    sources = split_sources()
    sources["diffusion"] = load_safetensors_header(REAL_FLUX2)
    with pytest.raises(AssemblyError, match="diffusion"):
        plan_flux_assembly(**sources)


@pytest.mark.skipif(not REAL_UMT5.exists(), reason="umt5 checkpoint absent")
def test_real_umt5_refuses_as_t5xxl() -> None:
    sources = split_sources()
    sources["t5xxl"] = load_safetensors_header(REAL_UMT5)
    with pytest.raises(AssemblyError, match="t5xxl"):
        plan_flux_assembly(**sources)


# ============================================================ SD era


REAL_SD15 = MODELS / "checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
REAL_DREAMSHAPER = MODELS / "checkpoints/DreamShaper_8_pruned.safetensors"
REAL_NETAYUME = MODELS / "checkpoints/NetaYumev35_pretrained_all_in_one.safetensors"


def unet_geometries(config=SD15_UNET_CONFIG) -> dict[str, TensorGeometry]:
    return geometrize(unet_layout(config), FLOAT16)


def clip_l_sd_geometries() -> dict[str, TensorGeometry]:
    """SD-checkpoint CLIP-L: transformers format, no text projection,
    plus the inert position_ids buffer real checkpoints carry."""
    sd = clip_geometries(projection=False)
    sd["text_model.embeddings.position_ids"] = g((1, 77), INT64)
    return sd


def clip_g_openclip_geometries() -> dict[str, TensorGeometry]:
    sd = geometrize(openclip_text_layout(CLIP_G_TEXT_CONFIG), FLOAT16)
    sd["logit_scale"] = g((), FLOAT16)
    return sd


def sd15_combined_geometries() -> dict[str, TensorGeometry]:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(unet_geometries(), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(clip_l_sd_geometries(), SD15_CLIP_L_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    sd["alphas_cumprod"] = g((1000,), FLOAT32)
    sd["model_ema.decay"] = g((), FLOAT32)
    return sd


def sdxl_combined_geometries(
    config: UNetConfig = SDXL_UNET_CONFIG,
) -> dict[str, TensorGeometry]:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(unet_geometries(config), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(clip_l_sd_geometries(), SDXL_CLIP_L_PREFIX))
    sd.update(prefixed(clip_g_openclip_geometries(), SDXL_CLIP_G_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    return sd


def refiner_combined_geometries() -> dict[str, TensorGeometry]:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(unet_geometries(SDXL_REFINER_UNET_CONFIG), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(clip_g_openclip_geometries(), SDXL_REFINER_CLIP_G_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    return sd


# ------------------------------------------------------ SD happy paths


def test_sd15_combined_plans() -> None:
    plan = plan_sd_assembly(checkpoint=source(sd15_combined_geometries(), "sd15.safetensors"))
    assert plan.family is SD15
    assert plan.clip_g is None
    assert plan.diffusion.config == SD15_UNET_CONFIG
    assert len(plan.diffusion.keys) == len(unet_layout(SD15_UNET_CONFIG))
    assert plan.clip_l is not None
    assert plan.clip_l.keys["text_model.embeddings.token_embedding.weight"] == (
        SD15_CLIP_L_PREFIX + "text_model.embeddings.token_embedding.weight"
    )
    assert plan.clip_l.ignored == (SD15_CLIP_L_PREFIX + "text_model.embeddings.position_ids",)
    assert plan.clip_l.absent == ("text_projection.weight",)
    assert not plan.clip_l.transforms
    assert set(plan.unclaimed) == {"alphas_cumprod", "model_ema.decay"}


def test_sd15_legacy_clip_spelling_gets_infix() -> None:
    sd = sd15_combined_geometries()
    legacy = {
        SD15_CLIP_L_PREFIX + key[len("text_model.") :]: value
        for key, value in clip_l_sd_geometries().items()
    }
    sd = {key: value for key, value in sd.items() if not key.startswith(SD15_CLIP_L_PREFIX)}
    sd.update(legacy)
    plan = plan_sd_assembly(checkpoint=source(sd, "sd15-legacy.safetensors"))
    assert plan.clip_l is not None
    assert plan.clip_l.keys["text_model.embeddings.token_embedding.weight"] == (
        SD15_CLIP_L_PREFIX + "embeddings.token_embedding.weight"
    )
    assert plan.clip_l.ignored == (SD15_CLIP_L_PREFIX + "embeddings.position_ids",)


def test_sdxl_combined_plans_both_encoders() -> None:
    plan = plan_sd_assembly(checkpoint=source(sdxl_combined_geometries(), "sdxl.safetensors"))
    assert plan.family is SDXL
    assert plan.diffusion.config == SDXL_UNET_CONFIG
    assert plan.clip_l is not None and plan.clip_g is not None
    assert not plan.clip_l.transforms
    assert plan.clip_g.config == CLIP_G_TEXT_CONFIG
    assert plan.clip_g.ignored == (SDXL_CLIP_G_PREFIX + "logit_scale",)
    assert plan.unclaimed == ()
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    layers = CLIP_G_TEXT_CONFIG.num_hidden_layers
    # q/k/v weight+bias per layer, plus the projection transpose.
    assert len(plan.clip_g.transforms) == 6 * layers + 1
    for part, proj in enumerate(("q_proj", "k_proj", "v_proj")):
        key = f"text_model.encoder.layers.0.self_attn.{proj}.weight"
        assert plan.clip_g.keys[key] == (
            SDXL_CLIP_G_PREFIX + "transformer.resblocks.0.attn.in_proj_weight"
        )
        assert plan.clip_g.transforms[key] == RowChunk(part=part, parts=3)
        assert plan.clip_g.dtypes[key] == FLOAT16
    assert plan.clip_g.keys["text_projection.weight"] == (SDXL_CLIP_G_PREFIX + "text_projection")
    assert plan.clip_g.transforms["text_projection.weight"] == Transpose2D()
    assert plan.clip_g.dtypes["text_projection.weight"] == FLOAT16
    assert hidden == 1280


@pytest.mark.parametrize("split", (False, True))
def test_sdxl_plan_preserves_diffusion_source_asset_identity(split: bool) -> None:
    class IdentifiedSource(FakeSource):
        asset_digest = "blake3:" + "a" * 64
        asset_size = 123

    checkpoint = IdentifiedSource(Path("sdxl.safetensors"), sdxl_combined_geometries())
    diffusion = None
    if split:
        diffusion = IdentifiedSource(Path("unet.safetensors"), unet_geometries(SDXL_UNET_CONFIG))
        diffusion.asset_digest = "blake3:" + "b" * 64
    plan = plan_sd_assembly(checkpoint=checkpoint, diffusion=diffusion)
    assert plan.diffusion_asset_digest == (diffusion or checkpoint).asset_digest


def test_sdxl_refiner_combined_plans() -> None:
    plan = plan_sd_assembly(checkpoint=source(refiner_combined_geometries(), "refiner.safetensors"))
    assert plan.family is SDXL_REFINER
    assert plan.diffusion.config == SDXL_REFINER_UNET_CONFIG
    assert plan.clip_l is None
    assert plan.clip_g is not None
    assert plan.clip_g.ignored == (SDXL_REFINER_CLIP_G_PREFIX + "logit_scale",)
    assert plan.unclaimed == ()


def test_sd_split_sources_plan() -> None:
    plan = plan_sd_assembly(
        diffusion=source(unet_geometries(), "unet.safetensors"),
        clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.family is SD15
    assert plan.diffusion.path == Path("/fake/unet.safetensors")
    assert plan.clip_l is not None
    assert plan.clip_l.keys["text_model.embeddings.token_embedding.weight"] == (
        "text_model.embeddings.token_embedding.weight"
    )


@pytest.mark.parametrize(
    ("checkpoint_geometries", "family_id"),
    [
        (sd15_combined_geometries, SD15.id),
        (sdxl_combined_geometries, SDXL.id),
    ],
)
def test_sd_diffusers_vae_plans_canonical_keys_shapes_and_identity(
    checkpoint_geometries, family_id: str
) -> None:
    """Header-only planning covers both SD family wiring paths; source
    spelling and linear attention storage do not alter KL structure."""
    canonical_vae = kl_geometries()
    canonical_vae["model_ema.decay"] = TensorGeometry((), FLOAT32)
    canonical_vae["model_ema.num_updates"] = TensorGeometry((), INT32)
    canonical = plan_sd_assembly(
        checkpoint=source(checkpoint_geometries(), "checkpoint.safetensors"),
        vae=source(canonical_vae, "canonical-vae.safetensors"),
    )
    diffusers_vae = diffusers_geometries(canonical_vae)
    diffusers = plan_sd_assembly(
        checkpoint=source(checkpoint_geometries(), "checkpoint.safetensors"),
        vae=source(diffusers_vae, "diffusers-vae.safetensors"),
    )
    assert diffusers.family.id == family_id
    assert isinstance(canonical.vae, ComponentPlan)
    assert isinstance(diffusers.vae, ComponentPlan)
    assert diffusers.vae.config == canonical.vae.config
    assert set(canonical.vae.ignored) == {"model_ema.decay", "model_ema.num_updates"}
    assert set(diffusers.vae.ignored) == {"model_ema.decay", "model_ema.num_updates"}
    assert set(diffusers.vae.keys) == set(canonical.vae.keys)
    assert diffusers.vae.dtypes == canonical.vae.dtypes
    assert diffusers.vae.transforms
    assert all(
        isinstance(transform, LinearToConv2D) for transform in diffusers.vae.transforms.values()
    )
    assert runtime_component_identity(
        family_id, diffusers.identity_components
    ) == runtime_component_identity(family_id, canonical.identity_components)


@pytest.mark.parametrize(
    ("unet", "expected_family"),
    ((SD15_UNET_CONFIG, "sd15"), (SDXL_UNET_CONFIG, "sdxl"), (SDXL_REFINER_UNET_CONFIG, "sdxl")),
)
def test_sd_taesd_combined_halves_follow_model_family(
    unet: UNetConfig, expected_family: str
) -> None:
    vae_geometry = {
        f"taesd_{role}.{key}": TensorGeometry(shape, FLOAT32)
        for role in ("encoder", "decoder")
        for key, shape in taesd_layout(role).items()
    }
    vae_geometry["vae_scale"] = TensorGeometry((), FLOAT32)
    vae_geometry["vae_shift"] = TensorGeometry((), FLOAT32)
    kwargs: dict[str, WeightSource] = {
        "diffusion": source(unet_geometries(unet), "unet.safetensors"),
        "vae": source(vae_geometry, "taesd.safetensors"),
    }
    if expected_family == "sd15":
        kwargs["clip_l"] = source(clip_l_sd_geometries(), "clip_l.safetensors")
    elif unet == SDXL_UNET_CONFIG:
        kwargs["clip_l"] = source(clip_l_sd_geometries(), "clip_l.safetensors")
        kwargs["clip_g"] = source(clip_g_openclip_geometries(), "clip_g.safetensors")
    else:
        kwargs["clip_g"] = source(clip_g_openclip_geometries(), "clip_g.safetensors")
    plan = plan_sd_assembly(**kwargs)
    assert isinstance(plan.vae, TAESDCodecPlan)
    assert plan.vae.config.family == expected_family
    assert plan.vae.encoder.keys["0.weight"] == "taesd_encoder.0.weight"
    assert plan.vae.scale_keys == ("vae_scale", "vae_shift")


def test_sd_taesd_combined_checkpoint_preserves_scalar_source_keys() -> None:
    checkpoint = sd15_combined_geometries()
    for key in tuple(checkpoint):
        if key.startswith(SD_VAE_PREFIX):
            del checkpoint[key]
    checkpoint.update(
        {
            f"{SD_VAE_PREFIX}taesd_{role}.{key}": TensorGeometry(shape, FLOAT32)
            for role in ("encoder", "decoder")
            for key, shape in taesd_layout(role).items()
        }
    )
    checkpoint[f"{SD_VAE_PREFIX}vae_scale"] = TensorGeometry((), FLOAT32)
    checkpoint[f"{SD_VAE_PREFIX}vae_shift"] = TensorGeometry((), FLOAT32)
    plan = plan_sd_assembly(checkpoint=source(checkpoint, "combined-taesd.safetensors"))
    assert isinstance(plan.vae, TAESDCodecPlan)
    assert plan.vae.scale_keys == (
        f"{SD_VAE_PREFIX}vae_scale",
        f"{SD_VAE_PREFIX}vae_shift",
    )


def test_sd_taesd_missing_half_refuses() -> None:
    only_decoder = {
        f"taesd_decoder.{key}": TensorGeometry(shape, FLOAT32)
        for key, shape in taesd_layout("decoder").items()
    }
    with pytest.raises(AssemblyError, match="missing or invalid TAESD encoder half"):
        plan_sd_assembly(
            diffusion=source(unet_geometries(), "unet.safetensors"),
            clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
            vae=source(only_decoder, "taesd.safetensors"),
        )


def test_sd_taesd_mixed_with_kl_refuses() -> None:
    mixed = kl_geometries()
    mixed.update(
        {
            f"taesd_{role}.{key}": TensorGeometry(shape, FLOAT32)
            for role in ("encoder", "decoder")
            for key, shape in taesd_layout(role).items()
        }
    )
    with pytest.raises(AssemblyError, match="mixed TAESD and non-TAESD layout"):
        plan_sd_assembly(
            diffusion=source(unet_geometries(), "unet.safetensors"),
            clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
            vae=source(mixed, "mixed.safetensors"),
        )


def test_plan_taesd_decoder_accepts_bare_and_prefixed_halves() -> None:
    bare = {key: TensorGeometry(shape, FLOAT32) for key, shape in taesd_layout("decoder").items()}
    plan = plan_taesd_decoder(source(bare, "taesd_decoder.safetensors"), family="sd15")
    assert plan.component == "taesd_decoder"
    assert plan.config.family == "sd15" and plan.config.role == "decoder"
    assert plan.keys["1.weight"] == "1.weight"

    combined = {f"taesd_decoder.{key}": value for key, value in bare.items()}
    combined.update(
        {
            f"taesd_encoder.{key}": TensorGeometry(shape, FLOAT32)
            for key, shape in taesd_layout("encoder").items()
        }
    )
    plan = plan_taesd_decoder(source(combined, "taesd.safetensors"), family="sdxl")
    assert plan.config.family == "sdxl"
    assert plan.keys["1.weight"] == "taesd_decoder.1.weight"
    assert not any(key.startswith("taesd_encoder.") for key in plan.keys.values())


def test_plan_taesd_decoder_rejects_scale_bearing_combined_artifacts() -> None:
    geometry = {
        f"taesd_{role}.{key}": TensorGeometry(shape, FLOAT32)
        for role in ("encoder", "decoder")
        for key, shape in taesd_layout(role).items()
    }
    geometry["vae_scale"] = TensorGeometry((), FLOAT32)
    geometry["vae_shift"] = TensorGeometry((), FLOAT32)
    with pytest.raises(TAESDDetectError, match="decoder artifact"):
        plan_taesd_decoder(source(geometry, "taesd.safetensors"), family="sd15")


def test_plan_taesd_decoder_rejects_colliding_bare_and_prefixed_keys() -> None:
    bare = {key: TensorGeometry(shape, FLOAT32) for key, shape in taesd_layout("decoder").items()}
    both = dict(bare)
    both.update({f"taesd_decoder.{key}": value for key, value in bare.items()})
    with pytest.raises(TAESDDetectError, match="appears twice"):
        plan_taesd_decoder(source(both, "dup.safetensors"), family="sd15")


def test_sd_split_overrides_combined() -> None:
    plan = plan_sd_assembly(
        checkpoint=source(sd15_combined_geometries(), "combined.safetensors"),
        vae=source(kl_geometries(), "override-vae.safetensors"),
    )
    assert isinstance(plan.vae, ComponentPlan)
    assert plan.vae.path == Path("/fake/override-vae.safetensors")
    assert plan.diffusion.path == Path("/fake/combined.safetensors")
    # The combined VAE keys are now unclaimed alongside the scheduler buffers.
    assert any(key.startswith(SD_VAE_PREFIX) for key in plan.unclaimed)


def test_sd_split_openclip_clip_g() -> None:
    plan = plan_sd_assembly(
        diffusion=source(unet_geometries(SDXL_REFINER_UNET_CONFIG), "unet.safetensors"),
        clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.family is SDXL_REFINER
    assert plan.clip_g is not None
    assert plan.clip_g.keys["text_projection.weight"] == "text_projection"
    assert plan.clip_g.ignored == ("logit_scale",)


def test_sd_split_openclip_clip_g_weight_spelling() -> None:
    # The reference conversion also accepts text_projection.weight
    # (already the Linear layout - renamed without a transpose).
    sd = clip_g_openclip_geometries()
    sd["text_projection.weight"] = sd.pop("text_projection")
    plan = plan_sd_assembly(
        diffusion=source(unet_geometries(SDXL_REFINER_UNET_CONFIG), "unet.safetensors"),
        clip_g=source(sd, "clip_g.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.clip_g is not None
    assert plan.clip_g.keys["text_projection.weight"] == "text_projection.weight"
    assert "text_projection.weight" not in plan.clip_g.transforms


def test_sd_plans_are_deterministic() -> None:
    checkpoint = source(sdxl_combined_geometries(), "sdxl.safetensors")
    assert plan_sd_assembly(checkpoint=checkpoint) == plan_sd_assembly(checkpoint=checkpoint)


# -------------------------------------------------------- SD refusals


def test_sd_no_sources_refuses() -> None:
    with pytest.raises(AssemblyError, match="no sources"):
        plan_sd_assembly()


def test_sd15_clip_g_split_refuses() -> None:
    with pytest.raises(AssemblyError, match="no CLIP-G"):
        plan_sd_assembly(
            checkpoint=source(sd15_combined_geometries(), "sd15.safetensors"),
            clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
        )


def test_refiner_clip_l_split_refuses() -> None:
    with pytest.raises(AssemblyError, match="no CLIP-L"):
        plan_sd_assembly(
            checkpoint=source(refiner_combined_geometries(), "refiner.safetensors"),
            clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        )


@pytest.mark.parametrize("zsnr", [False, True])
def test_sdxl_vpred_markers_plan_sampling_and_identity(zsnr: bool) -> None:
    sd = sdxl_combined_geometries()
    sd["v_pred"] = g((), FLOAT32)
    if zsnr:
        sd["ztsnr"] = g((), FLOAT32)
    plan = plan_sd_assembly(checkpoint=source(sd, "sdxl-vpred.safetensors"))
    assert plan.family is SDXL
    assert plan.sampling is not None
    assert plan.sampling.parameterization is Parameterization.V_PREDICTION
    assert plan.sampling.zsnr is zsnr
    assert plan.diffusion.identity_facts == (
        "parameterization=v_prediction",
        f"zsnr={zsnr}",
    )
    assert "v_pred" not in plan.diffusion.keys
    assert "ztsnr" not in plan.diffusion.keys


def test_sdxl_vpred_split_source_reads_top_level_marker() -> None:
    diffusion = unet_geometries(SDXL_UNET_CONFIG)
    diffusion["v_pred"] = g((), FLOAT32)
    plan = plan_sd_assembly(
        diffusion=source(diffusion, "sdxl-vpred.safetensors"),
        clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.sampling is not None
    assert plan.sampling.parameterization is Parameterization.V_PREDICTION
    assert "v_pred" not in plan.diffusion.keys


@pytest.mark.parametrize("with_min", [False, True])
def test_sdxl_edm_vpred_preserves_bounds_and_identity(with_min: bool) -> None:
    sd = sdxl_combined_geometries()
    values = {"edm_vpred.sigma_max": 42.5}
    sd["edm_vpred.sigma_max"] = g((), FLOAT32)
    if with_min:
        values["edm_vpred.sigma_min"] = 0.125
        sd["edm_vpred.sigma_min"] = g((1,), FLOAT32)
    plan = plan_sd_assembly(
        checkpoint=source(sd, "sdxl-edm-vpred.safetensors", scalar_values=values)
    )
    assert plan.family is SDXL
    assert plan.sampling is not None
    assert plan.sampling.parameterization is Parameterization.V_PREDICTION
    assert plan.sampling.space is SamplingSpace.CONTINUOUS_EDM
    assert plan.sampling.sigma_min == (0.125 if with_min else 0.002)
    assert plan.sampling.sigma_max == 42.5
    assert plan.diffusion.identity_facts == (
        "parameterization=v_prediction",
        "sampling_space=continuous_edm",
        f"sigma_min={plan.sampling.sigma_min!r}",
        "sigma_max=42.5",
    )
    assert "edm_vpred.sigma_max" not in plan.diffusion.keys
    assert "edm_vpred.sigma_min" not in plan.diffusion.keys


def test_sdxl_edm_vpred_identity_changes_with_either_bound() -> None:
    def identity(sigma_min: float, sigma_max: float) -> tuple[str, ...]:
        sd = sdxl_combined_geometries()
        sd["edm_vpred.sigma_max"] = g((), FLOAT32)
        sd["edm_vpred.sigma_min"] = g((), FLOAT32)
        plan = plan_sd_assembly(
            checkpoint=source(
                sd,
                scalar_values={
                    "edm_vpred.sigma_min": sigma_min,
                    "edm_vpred.sigma_max": sigma_max,
                },
            )
        )
        return runtime_component_identity(plan.family.id, plan.identity_components)

    base = identity(0.125, 42.5)
    assert identity(0.25, 42.5) != base
    assert identity(0.125, 80.0) != base


def test_only_edm_vpred_uses_the_configuration_scalar_seam() -> None:
    eps = CountingSource(Path("/fake/eps.safetensors"), sdxl_combined_geometries())
    plan_sd_assembly(checkpoint=eps)
    assert eps.scalar_reads == 0

    discrete = sdxl_combined_geometries()
    discrete["v_pred"] = g((), FLOAT32)
    discrete_source = CountingSource(Path("/fake/vpred.safetensors"), discrete)
    plan_sd_assembly(checkpoint=discrete_source)
    assert discrete_source.scalar_reads == 0

    edm = sdxl_combined_geometries()
    edm["edm_vpred.sigma_max"] = g((), FLOAT32)
    edm_source = CountingSource(
        Path("/fake/edm.safetensors"),
        edm,
        scalar_values={"edm_vpred.sigma_max": 42.5},
    )
    plan_sd_assembly(checkpoint=edm_source)
    assert edm_source.scalar_reads == 1


def test_sdxl_edm_vpred_requires_a_configuration_scalar_source() -> None:
    sd = sdxl_combined_geometries()
    sd["edm_vpred.sigma_max"] = g((), FLOAT32)
    source_without_scalars = HeaderOnlySource(Path("/fake/edm.safetensors"), sd)
    with pytest.raises(AssemblyError, match="numeric payload is unavailable"):
        plan_sd_assembly(checkpoint=source_without_scalars)


def test_sdxl_edm_vpred_split_source_reads_bounds() -> None:
    diffusion = unet_geometries(SDXL_UNET_CONFIG)
    diffusion["edm_vpred.sigma_max"] = g((), FLOAT32)
    diffusion["edm_vpred.sigma_min"] = g((), FLOAT32)
    plan = plan_sd_assembly(
        diffusion=source(
            diffusion,
            "sdxl-edm-vpred.safetensors",
            scalar_values={
                "edm_vpred.sigma_max": 42.5,
                "edm_vpred.sigma_min": 0.125,
            },
        ),
        clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.sampling is not None
    assert plan.sampling.space is SamplingSpace.CONTINUOUS_EDM
    assert plan.sampling.sigma_min == 0.125
    assert plan.sampling.sigma_max == 42.5


@pytest.mark.parametrize(
    ("values", "match"),
    (
        ({"edm_vpred.sigma_min": 0.125}, "requires"),
        ({"edm_vpred.sigma_max": float("nan")}, "finite"),
        ({"edm_vpred.sigma_max": float("inf")}, "finite"),
        ({"edm_vpred.sigma_max": -1.0}, "positive"),
        (
            {"edm_vpred.sigma_min": 2.0, "edm_vpred.sigma_max": 1.0},
            "must be <",
        ),
    ),
)
def test_sdxl_edm_vpred_bad_bounds_refuse(values: dict[str, float], match: str) -> None:
    sd = sdxl_combined_geometries()
    for key in values:
        sd[key] = g((), FLOAT32)
    with pytest.raises(AssemblyError, match=match):
        plan_sd_assembly(checkpoint=source(sd, scalar_values=values))


def test_plan_native_identity_configuration_conflict_is_invalid() -> None:
    sd = sdxl_combined_geometries()
    sd["edm_vpred.sigma_min"] = g((), FLOAT32)
    sd["edm_vpred.sigma_max"] = g((), FLOAT32)
    with pytest.raises(NativeRefusalError, match="must be <") as caught:
        plan_native(
            checkpoint=source(
                sd,
                scalar_values={
                    "edm_vpred.sigma_min": 2.0,
                    "edm_vpred.sigma_max": 1.0,
                },
            )
        )
    assert caught.value.category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY


def test_sdxl_ztsnr_without_vpred_remains_canonical_eps() -> None:
    sd = sdxl_combined_geometries()
    sd["ztsnr"] = g((), FLOAT32)
    plan = plan_sd_assembly(checkpoint=source(sd, "sdxl-ztsnr-only.safetensors"))
    assert plan.sampling == SDXL.sampling
    assert plan.diffusion.identity_facts == ()
    assert "ztsnr" not in plan.diffusion.keys


@pytest.mark.parametrize(
    "marker",
    ["v_pred", "ztsnr", "edm_vpred.sigma_max", "edm_vpred.sigma_min"],
)
def test_sdxl_sampling_marker_under_unet_prefix_refuses(marker: str) -> None:
    sd = sdxl_combined_geometries()
    sd[f"model.diffusion_model.{marker}"] = g((), FLOAT32)
    with pytest.raises(AssemblyError, match="must be top-level"):
        plan_sd_assembly(checkpoint=source(sd, "sdxl-misplaced-marker.safetensors"))


@pytest.mark.parametrize("marker", ["edm_mean", "edm_std"])
def test_sdxl_unsupported_sampling_markers_refuse(marker: str) -> None:
    sd = sdxl_combined_geometries()
    sd[marker] = g((), FLOAT32)
    with pytest.raises(AssemblyError, match=marker):
        plan_sd_assembly(checkpoint=source(sd, "sdxl-unsupported.safetensors"))


def test_sd15_inpaint_split_plans_and_ip2p_refuses() -> None:
    from dataclasses import replace as dc_replace

    inpaint = dc_replace(SD15_UNET_CONFIG, in_channels=9)
    plan = plan_sd_assembly(
        diffusion=source(unet_geometries(inpaint), "inpaint.safetensors"),
        clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.family is SD15
    assert plan.diffusion.config.in_channels == 9

    ip2p = dc_replace(SD15_UNET_CONFIG, in_channels=8)
    with pytest.raises(AssemblyError, match="outside the supported"):
        plan_sd_assembly(
            diffusion=source(unet_geometries(ip2p), "ip2p.safetensors"),
            clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
            vae=source(kl_geometries(), "vae.safetensors"),
        )


def test_sdxl_inpaint_split_plans_and_ip2p_refuses() -> None:
    plan = plan_sd_assembly(
        diffusion=source(unet_geometries(SDXL_INPAINT_UNET_CONFIG), "sdxl-inpaint.safetensors"),
        clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
        clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
        vae=source(kl_geometries(), "vae.safetensors"),
    )
    assert plan.family is SDXL
    assert plan.diffusion.config == SDXL_INPAINT_UNET_CONFIG
    assert plan.diffusion.config.adm_in_channels == 2816
    combined = plan_sd_assembly(
        checkpoint=source(
            sdxl_combined_geometries(SDXL_INPAINT_UNET_CONFIG),
            "sdxl-inpaint-combined.safetensors",
        )
    )
    assert combined.family is SDXL
    assert combined.diffusion.config == SDXL_INPAINT_UNET_CONFIG

    from dataclasses import replace as dc_replace

    ip2p = dc_replace(SDXL_UNET_CONFIG, in_channels=8)
    with pytest.raises(AssemblyError, match="outside the supported"):
        plan_sd_assembly(
            diffusion=source(unet_geometries(ip2p), "sdxl-ip2p.safetensors"),
            clip_l=source(clip_l_sd_geometries(), "clip_l.safetensors"),
            clip_g=source(clip_g_openclip_geometries(), "clip_g.safetensors"),
            vae=source(kl_geometries(), "vae.safetensors"),
        )


def test_wrong_clip_width_in_slot_refuses() -> None:
    clip_g_transformers = geometrize(clip_text_layout(CLIP_G_TEXT_CONFIG), FLOAT16)
    with pytest.raises(AssemblyError, match="768-wide"):
        plan_sd_assembly(
            diffusion=source(unet_geometries(), "unet.safetensors"),
            clip_l=source(clip_g_transformers, "clip_g.safetensors"),
            vae=source(kl_geometries(), "vae.safetensors"),
        )


def test_quantized_openclip_refuses() -> None:
    sd = sdxl_combined_geometries()
    quantize_legacy(sd, "transformer.resblocks.0.mlp.c_fc", SDXL_CLIP_G_PREFIX)
    with pytest.raises(AssemblyError, match="quantized OpenCLIP"):
        plan_sd_assembly(checkpoint=source(sd, "sdxl-q.safetensors"))


def test_sd_plan_requires_a_text_encoder() -> None:
    plan = plan_sd_assembly(checkpoint=source(sd15_combined_geometries(), "sd15.safetensors"))
    with pytest.raises(ValueError, match="text encoder"):
        SDAssemblyPlan(
            family=plan.family,
            diffusion=plan.diffusion,
            clip_l=None,
            clip_g=None,
            vae=plan.vae,
        )


def test_sd_plan_sampling_and_identity_facts_cannot_disagree() -> None:
    plan = plan_sd_assembly(checkpoint=source(sdxl_combined_geometries(), "sdxl.safetensors"))
    sampling = SamplingDescriptor(
        Parameterization.V_PREDICTION,
        plan.family.sampling.sigma_min,
        plan.family.sampling.sigma_max,
    )
    with pytest.raises(ValueError, match="sampling descriptor"):
        SDAssemblyPlan(
            family=plan.family,
            diffusion=plan.diffusion,
            clip_l=plan.clip_l,
            clip_g=plan.clip_g,
            vae=plan.vae,
            sampling=sampling,
        )
    vpred_diffusion = replace(
        plan.diffusion,
        identity_facts=("parameterization=v_prediction", "zsnr=False"),
    )
    with pytest.raises(ValueError, match="sampling descriptor"):
        SDAssemblyPlan(
            family=plan.family,
            diffusion=vpred_diffusion,
            clip_l=plan.clip_l,
            clip_g=plan.clip_g,
            vae=plan.vae,
        )


# ------------------------------------------------------ SD real headers


@pytest.mark.skipif(not REAL_SD15.exists(), reason="v1-5-pruned checkpoint absent")
def test_real_sd15_combined() -> None:
    plan = plan_sd_assembly(checkpoint=load_safetensors_header(REAL_SD15))
    assert plan.family is SD15
    assert len(plan.diffusion.keys) == 686
    assert plan.clip_l is not None and plan.clip_g is None
    assert len(plan.clip_l.keys) == 196
    assert plan.clip_l.ignored == (SD15_CLIP_L_PREFIX + "text_model.embeddings.position_ids",)
    assert plan.clip_l.absent == ("text_projection.weight",)
    assert isinstance(plan.vae, ComponentPlan)
    assert len(plan.vae.keys) == 248
    # The LDM wrapper's schedule buffers and EMA bookkeeping.
    assert "alphas_cumprod" in plan.unclaimed
    assert "model_ema.decay" in plan.unclaimed
    assert all(dtype == FLOAT16 for dtype in plan.diffusion.dtypes.values())


@pytest.mark.skipif(not REAL_DREAMSHAPER.exists(), reason="DreamShaper checkpoint absent")
def test_real_dreamshaper_is_plain_sd15() -> None:
    plan = plan_sd_assembly(checkpoint=load_safetensors_header(REAL_DREAMSHAPER))
    assert plan.family is SD15
    assert plan.unclaimed == ()


@pytest.mark.skipif(not REAL_NETAYUME.exists(), reason="NetaYume checkpoint absent")
def test_real_netayume_refuses_as_sd() -> None:
    with pytest.raises(AssemblyError, match="diffusion"):
        plan_sd_assembly(checkpoint=load_safetensors_header(REAL_NETAYUME))
