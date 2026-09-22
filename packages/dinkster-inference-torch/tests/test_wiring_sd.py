"""Stage 5: the SD-era family runtime wiring (test_wiring.py's
SD1/SDXL sibling).

SDRuntime composes only already-pinned pieces (the CLIP tokenizer and
encoders in test_clip_text.py, the golden-pinned UNet in test_unet.py,
SDDenoiser in test_sd_denoise.py, run_denoise, the KL codec), so these
tests pin the COMPOSITION: the per-family CLIP stack (SD1ClipModel /
SDXLClipModel / SDXLRefinerClipModel @ 947c2749), the KSampler recipe
over the shared discrete linear-beta EPS table, the encode_model_conds
ADM defaults (latent-derived pixel sizes, refiner aesthetic polarity),
the guidance refusal, the codec delegation, and the SD-specific
runtime_identity facts (planned transforms rotate, unwired None slots
contribute nothing). Source planning rejects incompatible text slots
before loading a runtime.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
import torch
from clip_fill import fill_value
from dinkster_inference import (
    CLIP_G_PROFILE,
    FLOAT16,
    FLOAT32,
    SD15,
    SD15_CONTROL_RESIDUAL_SITES,
    SDXL,
    SDXL_CONTROL_RESIDUAL_SITES,
    SDXL_REFINER,
    CancellationFlag,
    ClipTextConfig,
    ComponentPlan,
    Conditioning,
    ConditioningRuntime,
    ConstantGainCurve,
    ContinuousEDMSigmas,
    ContributionGain,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    DirectGainTableCurve,
    DiscreteSigmas,
    EffectMaskInput,
    FamilyRuntime,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidancePlanContext,
    GuidancePostCFGDescriptor,
    GuidanceReduceContext,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    InpaintConditioning,
    KLConfig,
    MaskMediaPlacement,
    ModelTokenLayout,
    ModelTokenSegment,
    Parameterization,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PerpNegSamplingGuidance,
    PromptTokenizer,
    Registry,
    RowChunk,
    SamplerDescriptor,
    SamplingDescriptor,
    SamplingGuidance,
    SamplingSegment,
    SamplingSpace,
    SamplingStateEvent,
    SDControlMode,
    TensorGeometry,
    TokenGridTransform,
    UNetConfig,
    WeightEntry,
    build_runtime_identity,
    builtin_sampler_registry,
    builtin_samplers,
    builtin_schedulers,
    load_clip_bpe,
    sampling_execution_context,
    sampling_sigmas,
    use_sampling_environment,
)
from dinkster_inference.controlnet import ControlApplication
from dinkster_inference_torch import (
    SD1_CLIP_L_POLICY,
    SDXL_CLIP_POLICY,
    SDXL_NEGATIVE_AESTHETIC_DEFAULT,
    AssembledSD,
    AutoencoderKL,
    ClipTextEncoder,
    ClipTextModel,
    ControlResourceBindingError,
    DenoiseError,
    EmbeddingLookup,
    GuidanceExecutor,
    GuidanceRegistry,
    SD15ControlNet,
    SD15T2IAdapter,
    SDControlConditioning,
    SDControlResiduals,
    SDDenoiser,
    SDEffectMaskSource,
    SDRuntime,
    SDXLControlLoRA,
    SDXLControlNet,
    SDXLControlNetUnion,
    UNetModel,
    WiringError,
    compose_sdxl_conditioning,
    encode_sdxl_adm,
    encode_sdxl_refiner_adm,
    guidance_transforms,
    latent_process_out,
    prepare_noise,
    run_denoise,
    sd_control_hint_digest,
    sd_effect_mask_source_digest,
    torch_sampler_registry,
    torch_scheduler_registry,
)
from dinkster_inference_torch import sampling_execution as sampling_engine
from dinkster_inference_torch import sampling_execution as sampling_execution_module
from dinkster_inference_torch._conditioning_layout import declare_text_conditioning
from dinkster_inference_torch.attention import builtin_sdpa_kernel
from dinkster_inference_torch.controlnet import (
    _bind_sd15_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_control_lora_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_union_resource,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.guidance import (
    ConditioningEvaluation,
    ConditioningValidationPath,
    GuidedDenoiser,
)
from dinkster_inference_torch.sampling_execution import (
    compile_guidance_plan,
    guided_denoiser,
    run_ksampler_as_custom,
)
from dinkster_inference_torch.schedules import (
    continuous_edm_percent_to_sigma,
    discrete_percent_to_sigma,
)
from dinkster_inference_torch.sd_denoise import SDControlGain
from dinkster_inference_torch.t2i_adapter import (
    _bind_sd15_t2i_adapter_resource,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.wiring import (
    load_runtime,
)
from golden_files import load_platform_golden, platform_digest
from unet_fill import fill_state_dict

CUSTOM_SIGMA_GOLDENS = load_platform_golden(
    Path(__file__).parents[3] / "tests" / "goldens" / "sampling_goldens.json"
)["custom_sigma_queries"]

# Tiny everywhere EXCEPT the CLIP vocabularies: encode_text drives the
# real BPE, whose ids index real-sized embedding tables. The towers
# get DIFFERENT widths so the SDXL feature concat order is observable:
# context_dim = clip_l hidden + clip_g hidden for the base, one
# tower's hidden for SD1/refiner; adm_in_channels = the projected
# CLIP-G pooled width + the reference's Timestep(256) blocks (six for
# the base, five for the refiner).
TINY_CLIP_L = ClipTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    hidden_act="quick_gelu",
    vocab_size=49408,
    eos_token_id=49407,
)
TINY_CLIP_G = ClipTextConfig(
    hidden_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=96,
    hidden_act="gelu",
    vocab_size=49408,
    eos_token_id=49407,
)
TINY_SD1_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(1, 1),
    transformer_depth_output=(1, 1, 1, 1),
    transformer_depth_middle=1,
    context_dim=TINY_CLIP_L.hidden_size,
    use_linear_in_transformer=False,
    num_heads=8,
)
TINY_SD1_INPAINT_UNET = replace(TINY_SD1_UNET, in_channels=9)
TINY_SDXL_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(0, 2),
    transformer_depth_output=(0, 0, 2, 2),
    transformer_depth_middle=2,
    context_dim=TINY_CLIP_L.hidden_size + TINY_CLIP_G.hidden_size,
    use_linear_in_transformer=True,
    adm_in_channels=TINY_CLIP_G.hidden_size + 6 * 256,
    num_head_channels=16,
)
TINY_SDXL_INPAINT_UNET = replace(TINY_SDXL_UNET, in_channels=9)
TINY_REFINER_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(0, 2),
    transformer_depth_output=(0, 0, 2, 2),
    transformer_depth_middle=2,
    context_dim=TINY_CLIP_G.hidden_size,
    use_linear_in_transformer=True,
    adm_in_channels=TINY_CLIP_G.hidden_size + 5 * 256,
    num_head_channels=16,
)
TINY_KL = KLConfig(
    in_channels=3,
    out_channels=3,
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=1,
    z_channels=4,
    embed_dim=4,
)

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


def filled(module: ModuleT) -> ModuleT:
    state = {
        key: fill_value(key, tuple(tensor.shape)) for key, tensor in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)
    return module


def tiny_unet(config: UNetConfig) -> UNetModel:
    model = UNetModel(config)
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


@pytest.fixture(scope="module")
def sd1() -> SDRuntime:
    torch.manual_seed(0)
    assembled = AssembledSD(
        family=SD15,
        diffusion=tiny_unet(TINY_SD1_UNET),
        clip_l=filled(ClipTextModel(TINY_CLIP_L)),
        clip_g=None,
        vae=filled(AutoencoderKL(TINY_KL)),
    )
    return SDRuntime(assembled, runtime_identity="native:test:sd1")


def control_conditioning(
    *,
    child_id: str = "canny",
    strength: float = 1.0,
    window: PercentRange | None = None,
    gain: ContributionGain | None = None,
    model_digest: str = "a" * 64,
    hint_value: float = 0.0,
    previous: SDControlConditioning | None = None,
    effect_masks: tuple[SDEffectMaskSource, ...] = (),
) -> SDControlConditioning:
    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    hint = torch.full((1, 3, 64, 64), hint_value)
    hint_digest = sd_control_hint_digest(hint)
    return SDControlConditioning(
        ControlApplication(
            child_id,
            PayloadReference(hint_digest),
            strength,
            PercentRange(0.0, 1.0) if window is None else window,
            None if previous is None else previous.application,
        ),
        model,
        hint,
        model_digest,
        hint_digest,
        gain,
        previous,
        effect_masks,
    )


def t2i_adapter_conditioning(*, gain: ContributionGain | None = None) -> SDControlConditioning:
    model_digest = "b" * 64
    with torch.device("meta"):
        model = SD15T2IAdapter()
    _bind_sd15_t2i_adapter_resource(model, model_digest)
    hint = torch.zeros(1, 1, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    return SDControlConditioning(
        ControlApplication(
            "canny-adapter",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
        ),
        model,
        hint,
        model_digest,
        hint_digest,
        gain,
    )


def control_lora_conditioning(
    *,
    gain: ContributionGain | None = None,
    effect_masks: tuple[SDEffectMaskSource, ...] = (),
) -> SDControlConditioning:
    model_digest = "c" * 64
    with torch.device("meta"):
        model = SDXLControlLoRA()
    _bind_sdxl_control_lora_resource(model, model_digest)
    hint = torch.zeros(1, 3, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    return SDControlConditioning(
        ControlApplication(
            "canny-control-lora",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
        ),
        model,
        hint,
        model_digest,
        hint_digest,
        gain,
        effect_masks=effect_masks,
    )


def controlnet_union_conditioning(token: str) -> SDControlConditioning:
    model_digest = "d" * 64
    with torch.device("meta"):
        model = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
    _bind_sdxl_controlnet_union_resource(model, model_digest)
    hint = torch.zeros(1, 3, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    mode = SDControlMode("sdxl-controlnet-union", token)  # type: ignore[arg-type]
    return SDControlConditioning(
        ControlApplication(
            "controlnet-union",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
            mode=mode,
        ),
        model,
        hint,
        model_digest,
        hint_digest,
    )


def sdxl_controlnet_conditioning(
    *,
    gain: ContributionGain | None = None,
    effect_masks: tuple[SDEffectMaskSource, ...] = (),
) -> SDControlConditioning:
    model_digest = "e" * 64
    with torch.device("meta"):
        model = SDXLControlNet()
    _bind_sdxl_controlnet_resource(model, model_digest)
    hint = torch.zeros(1, 3, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    return SDControlConditioning(
        ControlApplication(
            "classic-sdxl-controlnet",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
        ),
        model,
        hint,
        model_digest,
        hint_digest,
        gain,
        effect_masks=effect_masks,
    )


def effect_mask_source(mask: torch.Tensor, site: str) -> SDEffectMaskSource:
    digest = sd_effect_mask_source_digest(mask)
    return SDEffectMaskSource(
        EffectMaskInput(
            PayloadDescriptor(PayloadReference(digest), tuple(mask.shape), "float32", "mask"),
            digest,
            ("batch", "height", "width"),
            MaskMediaPlacement.FULL_DOMAIN,
            site,
            "sd15.control-effect-mask.bilinear-align-corners-false.v1",
        ),
        mask,
    )


def test_controlnet_parity_receipt_pins_understood_difference() -> None:
    receipt = json.loads(
        (Path(__file__).parent / "goldens" / "sd15_controlnet_acceptance.json").read_text()
    )
    outlier = receipt["reference"]["isolated_outlier"]
    assert outlier["observed_frequency"] == "1 of 6 identical reference executions"
    assert outlier["stable_sample_max_abs"] == 0.5831151008605957
    assert outlier["stable_sample_mean_abs"] == 0.04189985990524292
    assert outlier["conditioning_bit_exact"] is True
    assert outlier["first_controlnet_input_bit_exact"] is True
    assert outlier["all_first_step_residuals_bit_exact"] is True
    difference = receipt["reference"]["understood_difference"]
    assert difference["bit_exact_inputs"] is True
    assert difference["control_residual_max_abs"] == 0.015625
    assert difference["control_residual_mean_abs"] == 0.0014066696166992188
    assert difference["forced_deterministic_pair"]["bit_exact"] is False


@pytest.fixture
def synthetic_sd15_receipt() -> str:
    import dinkster_inference_torch.distributed as distributed_module

    return distributed_module.guidance_receipt_identity(
        "dinkster.sd15",
        ("topology=guidance", "lane_evaluation=one-batch1-lane-per-rank"),
        torch.float16,
        2,
    )


@pytest.fixture(scope="module")
def sd1_inpaint() -> SDRuntime:
    assembled = AssembledSD(
        family=SD15,
        diffusion=tiny_unet(TINY_SD1_INPAINT_UNET),
        clip_l=filled(ClipTextModel(TINY_CLIP_L)),
        clip_g=None,
        vae=filled(AutoencoderKL(TINY_KL)),
    )
    return SDRuntime(assembled, runtime_identity="native:test:sd1-inpaint")


@pytest.fixture(scope="module")
def sdxl() -> SDRuntime:
    torch.manual_seed(0)
    assembled = AssembledSD(
        family=SDXL,
        diffusion=tiny_unet(TINY_SDXL_UNET),
        clip_l=filled(ClipTextModel(TINY_CLIP_L)),
        clip_g=filled(ClipTextModel(TINY_CLIP_G)),
        vae=filled(AutoencoderKL(TINY_KL)),
    )
    return SDRuntime(assembled, runtime_identity="native:test:sdxl")


@pytest.fixture(scope="module")
def sdxl_inpaint() -> SDRuntime:
    assembled = AssembledSD(
        family=SDXL,
        diffusion=tiny_unet(TINY_SDXL_INPAINT_UNET),
        clip_l=filled(ClipTextModel(TINY_CLIP_L)),
        clip_g=filled(ClipTextModel(TINY_CLIP_G)),
        vae=filled(AutoencoderKL(TINY_KL)),
    )
    return SDRuntime(assembled, runtime_identity="native:test:sdxl-inpaint")


@pytest.fixture(scope="module")
def refiner() -> SDRuntime:
    torch.manual_seed(0)
    assembled = AssembledSD(
        family=SDXL_REFINER,
        diffusion=tiny_unet(TINY_REFINER_UNET),
        clip_l=None,
        clip_g=filled(ClipTextModel(TINY_CLIP_G)),
        vae=filled(AutoencoderKL(TINY_KL)),
    )
    return SDRuntime(assembled, runtime_identity="native:test:refiner")


def test_runtimes_satisfy_the_protocol(sd1: SDRuntime, sdxl: SDRuntime, refiner: SDRuntime) -> None:
    for runtime, family_id in (
        (sd1, "dinkster.sd15"),
        (sdxl, "dinkster.sdxl"),
        (refiner, "dinkster.sdxl_refiner"),
    ):
        seam: FamilyRuntime[torch.Tensor] = runtime
        assert seam.family.id == family_id
        assert runtime.assembled.compute_dtype("vae") is None


def test_sd_runtime_reuses_assembled_attention_status(sd1: SDRuntime) -> None:
    assert sd1.attention_status is sd1.assembled.attention_status


@pytest.mark.parametrize("family", ("sd1", "sdxl", "refiner"))
def test_sd_runtime_prepares_canonical_conditioning(
    family: str, request: pytest.FixtureRequest
) -> None:
    from dinkster_inference_torch import basic_conditioning_to_carrier

    runtime: SDRuntime = request.getfixturevalue(family)
    assert isinstance(runtime, ConditioningRuntime)
    assert runtime.conditioning_identity == runtime.runtime_identity
    encoded = runtime.encode_text("a glass bottle")
    restored = runtime.prepare_single_stream_conditioning(basic_conditioning_to_carrier(encoded))
    assert torch.equal(restored.embeddings, encoded.embeddings)
    assert restored.embeddings.dtype == encoded.embeddings.dtype
    if encoded.pooled is None:
        assert restored.pooled is None
    else:
        assert restored.pooled is not None
        assert torch.equal(restored.pooled, encoded.pooled)


@pytest.mark.parametrize("device", (None, "cpu"))
def test_sd_runtime_uses_resident_execution_device(
    sd1: SDRuntime, monkeypatch: pytest.MonkeyPatch, device: str | None
) -> None:
    from dinkster_inference_torch import basic_conditioning_to_carrier

    encoded = sd1.encode_text("a glass bottle")
    assert next(sd1.assembled.diffusion.parameters()).device.type == "cpu"
    monkeypatch.setitem(
        sd1.assembled.diffusion.__dict__,
        "_dinkster_resident_weights",
        SimpleNamespace(load_device=torch.device("meta")),
    )
    restored = sd1.prepare_single_stream_conditioning(basic_conditioning_to_carrier(encoded))
    assert restored.embeddings.device.type == "meta"
    assert restored.pooled is not None and restored.pooled.device.type == "meta"
    observed: list[object] = []

    def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
        observed.append(kwargs["device"])
        return cast("torch.Tensor", kwargs["latent"])

    monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
    sd1.sample(
        tiny_latent(),
        cond=encoded,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
        compute_dtype=torch.float32,
        device=device,
    )
    assert [torch.device(cast("Any", value)) for value in observed] == [
        torch.device(device or "meta")
    ]


# --- encode_text: the per-family CLIP stack ---------------------------------


def clip_tokenizer() -> PromptTokenizer:
    return PromptTokenizer(encode_word=load_clip_bpe().encode)


class TestEncodeText:
    def test_sd1_is_clip_l_final_hidden_with_raw_pooled(self, sd1: SDRuntime) -> None:
        """SD1ClipModel @ 947c2749: CLIP-L alone, final hidden state,
        raw (unprojected) pooled - ClipTextEncoder's defaults."""
        cond = sd1.encode_text("a photo of a cat")
        assert cond.embeddings.shape == (1, 77, TINY_CLIP_L.hidden_size)
        assert cond.pooled is not None
        assert cond.pooled.shape == (1, TINY_CLIP_L.hidden_size)
        clip_l = sd1.assembled.clip_l
        assert clip_l is not None
        expected = ClipTextEncoder(clip_l, policy=SD1_CLIP_L_POLICY).encode(
            clip_tokenizer().tokenize("a photo of a cat")
        )
        assert torch.equal(cond.embeddings, expected.embeddings)
        assert expected.pooled is not None
        assert torch.equal(cond.pooled, expected.pooled)

    def test_sdxl_concatenates_both_towers(self, sdxl: SDRuntime) -> None:
        """SDXLClipModel.encode_token_weights @ 947c2749: CLIP-L and
        CLIP-G both at the penultimate layer without the final norm,
        feature-concatenated L-then-G, CLIP-G's projected pooled."""
        cond = sdxl.encode_text("a photo of a cat")
        width = TINY_CLIP_L.hidden_size + TINY_CLIP_G.hidden_size
        assert cond.embeddings.shape == (1, 77, width)
        assert cond.pooled is not None
        assert cond.pooled.shape == (1, TINY_CLIP_G.hidden_size)
        clip_l_model = sdxl.assembled.clip_l
        clip_g_model = sdxl.assembled.clip_g
        assert clip_l_model is not None and clip_g_model is not None
        tokens = clip_tokenizer().tokenize("a photo of a cat")
        clip_l = ClipTextEncoder(clip_l_model, policy=SDXL_CLIP_POLICY).encode(tokens)
        clip_g = ClipTextEncoder(
            clip_g_model, profile=CLIP_G_PROFILE, policy=SDXL_CLIP_POLICY
        ).encode(tokens)
        expected = compose_sdxl_conditioning(clip_l, clip_g)
        assert torch.equal(cond.embeddings, expected.embeddings)
        assert expected.pooled is not None
        assert torch.equal(cond.pooled, expected.pooled)

    def test_refiner_is_clip_g_alone(self, refiner: SDRuntime) -> None:
        """SDXLRefinerClipModel @ 947c2749: only the CLIP-G tower."""
        cond = refiner.encode_text("a photo of a cat")
        assert cond.embeddings.shape == (1, 77, TINY_CLIP_G.hidden_size)
        assert cond.pooled is not None
        assert cond.pooled.shape == (1, TINY_CLIP_G.hidden_size)
        clip_g_model = refiner.assembled.clip_g
        assert clip_g_model is not None
        expected = ClipTextEncoder(
            clip_g_model, profile=CLIP_G_PROFILE, policy=SDXL_CLIP_POLICY
        ).encode(clip_tokenizer().tokenize("a photo of a cat"))
        assert torch.equal(cond.embeddings, expected.embeddings)

    def test_sdxl_textual_inversion_uses_both_towers(self, sdxl: SDRuntime) -> None:
        calls: list[tuple[str, str]] = []

        def clip_l_lookup(name: str) -> torch.Tensor | None:
            calls.append(("clip_l", name))
            return torch.ones((2, TINY_CLIP_L.hidden_size)) if name == "pair" else None

        def clip_g_lookup(name: str) -> torch.Tensor | None:
            calls.append(("clip_g", name))
            return torch.ones((2, TINY_CLIP_G.hidden_size)) if name == "pair" else None

        lookups = {
            "clip_l": clip_l_lookup,
            "clip_g": clip_g_lookup,
        }
        runtime = SDRuntime(
            sdxl.assembled,
            runtime_identity="native:test:sdxl-embedding",
            embedding_lookups=lookups,
        )
        cond = runtime.encode_text("embedding:pair")
        plain = runtime.encode_text("pair")
        zero_runtime = SDRuntime(
            sdxl.assembled,
            runtime_identity="native:test:sdxl-embedding-zero",
            embedding_lookups={
                "clip_l": lambda _name: torch.zeros((2, TINY_CLIP_L.hidden_size)),
                "clip_g": lambda _name: torch.zeros((2, TINY_CLIP_G.hidden_size)),
            },
        )
        zero = zero_runtime.encode_text("embedding:pair")
        assert cond.embeddings.shape == (
            1,
            77,
            TINY_CLIP_L.hidden_size + TINY_CLIP_G.hidden_size,
        )
        assert not torch.equal(cond.embeddings, plain.embeddings)
        assert not torch.equal(cond.embeddings, zero.embeddings)
        assert calls.count(("clip_l", "pair")) >= 2
        assert calls.count(("clip_g", "pair")) >= 2

    def test_sd1_textual_inversion_resolves_clip_l(self, sd1: SDRuntime) -> None:
        calls: list[str] = []

        def lookup(name: str) -> torch.Tensor | None:
            calls.append(name)
            return torch.ones((2, TINY_CLIP_L.hidden_size)) if name == "solo" else None

        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:test:sd1-embedding",
            embedding_lookups={"clip_l": lookup},
        )
        embedded = runtime.encode_text("embedding:solo")
        plain = runtime.encode_text("solo")
        zero_runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:test:sd1-embedding-zero",
            embedding_lookups={"clip_l": lambda _name: torch.zeros((2, TINY_CLIP_L.hidden_size))},
        )
        zero = zero_runtime.encode_text("embedding:solo")
        assert not torch.equal(embedded.embeddings, plain.embeddings)
        assert not torch.equal(embedded.embeddings, zero.embeddings)
        assert calls.count("solo") >= 2

    def test_sdxl_textual_inversion_refuses_incompatible_rows(self, sdxl: SDRuntime) -> None:
        lookups: dict[str, EmbeddingLookup] = {
            "clip_l": lambda _name: torch.ones((2, TINY_CLIP_L.hidden_size)),
            "clip_g": lambda _name: torch.ones((1, TINY_CLIP_G.hidden_size)),
        }
        runtime = SDRuntime(
            sdxl.assembled,
            runtime_identity="native:test:sdxl-embedding-mismatch",
            embedding_lookups=lookups,
        )
        with pytest.raises(WiringError, match="incompatible component row counts"):
            runtime.encode_text("embedding:pair")


# --- sample: the KSampler recipe over the discrete EPS table ---------------


def tiny_latent() -> torch.Tensor:
    generator = torch.Generator("cpu")
    generator.manual_seed(99)
    return torch.randn(1, 4, 8, 8, generator=generator)


def guided_sd_denoiser(
    evaluator: SDDenoiser,
    *,
    cond: Conditioning[torch.Tensor],
    cfg: SamplingGuidance[Conditioning[torch.Tensor]] | None,
    sampler: SamplerDescriptor[Any],
    sigmas: Sequence[float],
    seed: int,
    adm_cond: torch.Tensor | None = None,
    adm_uncond: torch.Tensor | None = None,
    batch_evaluation: bool = True,
) -> GuidedDenoiser:
    return guided_denoiser(
        ConditioningEvaluation(
            lambda value, role: evaluator.prepare_conditioning(
                value,
                adm=adm_uncond if role is GuidanceRole.UNCONDITIONAL else adm_cond,
            ),
            evaluator.evaluate_conditioning,
            evaluator.batchable if batch_evaluation else None,
            evaluator.evaluate_conditioning_batch if batch_evaluation else None,
            standard_activation_memory_factor=1.0,
        ),
        input=tiny_latent(),
        executor=None,
        plan=compile_guidance_plan(cond, cfg, sampler, None),
        execution=sampling_execution_context(sigmas, seed),
    )


class TestSample:
    @pytest.mark.parametrize(
        ("fixture_name", "family_id"),
        (
            ("sd1", "dinkster.sd15"),
            ("sdxl", "dinkster.sdxl"),
            ("refiner", "dinkster.sdxl_refiner"),
        ),
    )
    def test_discrete_custom_sigma_queries_match_reference_goldens(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        family_id: str,
    ) -> None:
        runtime = cast("SDRuntime", request.getfixturevalue(fixture_name))
        beta_options = CUSTOM_SIGMA_GOLDENS["beta_options"]
        assert runtime.custom_sampling_beta_sigmas(**beta_options) == tuple(
            CUSTOM_SIGMA_GOLDENS["beta"][family_id]
        )
        assert runtime.custom_sampling_sd_turbo_sigmas(
            **CUSTOM_SIGMA_GOLDENS["sd_turbo_options"]
        ) == tuple(CUSTOM_SIGMA_GOLDENS["sd_turbo"])
        for return_actual_sigma in (False, True):
            for percent in (0.0, 0.37, 1.0):
                key = f"{percent},{return_actual_sigma}"
                assert (
                    runtime.custom_sampling_percent_to_sigma(
                        percent,
                        return_actual_sigma=return_actual_sigma,
                    )
                    == CUSTOM_SIGMA_GOLDENS["percent_to_sigma"][family_id][key]
                )

    def test_continuous_edm_custom_sigma_queries_match_reference_goldens(
        self,
        sdxl: SDRuntime,
    ) -> None:
        sampling = SamplingDescriptor(
            Parameterization.V_PREDICTION,
            0.002,
            120.0,
            space=SamplingSpace.CONTINUOUS_EDM,
        )
        runtime = SDRuntime(
            replace(sdxl.assembled, sampling=sampling),
            runtime_identity="native:test:sdxl-edm-custom-queries",
        )
        beta_options = CUSTOM_SIGMA_GOLDENS["beta_options"]
        beta_indices = torch.tensor(
            (999, 538, 458, 401, 356, 316, 281, 248, 216, 185, 153, 119, 80)
        )
        reference_table = torch.linspace(
            math.log(sampling.sigma_min),
            math.log(sampling.sigma_max),
            1000,
        ).exp()
        assert runtime.custom_sampling_beta_sigmas(**beta_options) == (
            *(float(sigma) for sigma in reference_table[beta_indices]),
            0.0,
        )
        expected_percent = CUSTOM_SIGMA_GOLDENS["percent_to_sigma"]["dinkster.sdxl_continuous_edm"]
        for return_actual_sigma in (False, True):
            for percent in (0.0, 0.37, 1.0):
                key = f"{percent},{return_actual_sigma}"
                assert (
                    runtime.custom_sampling_percent_to_sigma(
                        percent,
                        return_actual_sigma=return_actual_sigma,
                    )
                    == expected_percent[key]
                )
        with pytest.raises(ValueError, match="require a discrete sigma space"):
            runtime.custom_sampling_sd_turbo_sigmas(steps=4, denoise=0.65)

    def test_discrete_percent_query_uses_reference_float32_interpolation(self) -> None:
        space = DiscreteSigmas.linear_beta()
        reference = CUSTOM_SIGMA_GOLDENS["percent_to_sigma"]["dinkster.sd15"]["0.37,False"]
        assert reference == 2.5423052310943604
        assert space.percent_to_sigma(0.37) != reference
        assert discrete_percent_to_sigma(space, 0.37) == reference

    def test_custom_sampling_uses_exact_sigmas_noise_and_denoised_state(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        assert isinstance(sd1, CustomSamplingRuntime)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        sigmas = sd1.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        request = CustomSamplingRequest(sampler, (), sigmas)
        latent = tiny_latent()
        noise = torch.full_like(latent, 0.25)
        denoised = torch.full_like(latent, 0.4)
        output = torch.full_like(latent, 0.6)
        captured: dict[str, object] = {}

        def capture_run(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
            captured.update(kwargs)
            callback = cast("Any", kwargs["on_state"])
            callback(
                SamplingStateEvent(
                    step=0,
                    total=2,
                    sigma=sigmas[0],
                    phase="pre_update",
                    current=latent,
                    denoised=denoised,
                )
            )
            return output

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        result = sd1.sample_custom(
            latent,
            noise=noise,
            cond=sd1.encode_text("a cat"),
            cfg=None,
            request=request,
            seed=9,
        )

        assert captured["noise"] is noise
        assert captured["sigmas"] == sigmas
        assert captured["initial_sigma"] == sigmas[0]
        assert result.output is output
        expected_denoised = latent_process_out(
            denoised,
            SD15.single_stream_latent(),
        )
        assert result.denoised_output is not None
        assert torch.equal(result.denoised_output, expected_denoised)

    def test_custom_sampling_matches_the_standard_sd_path(self, sd1: SDRuntime) -> None:
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:custom-parity",
        )
        latent = tiny_latent()
        cond = custom.encode_text("custom parity")
        seed = 23
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None

        expected = custom.sample(
            latent,
            cond=cond,
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=2,
            seed=seed,
            compute_dtype=torch.float32,
        )
        result = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=None,
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=seed,
        )

        assert torch.equal(result.output, expected)

    def test_ksampler_facade_delegates_sd_extras_to_custom_sampling(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dinkster_inference_torch import sampling_runtime

        captured: dict[str, object] = {}
        output = torch.zeros_like(tiny_latent())

        def capture(owner: object, latent: object, **kwargs: object) -> CustomSamplingResult[Any]:
            captured["owner"] = owner
            captured["latent"] = latent
            captured.update(kwargs)
            return CustomSamplingResult(output, None)

        monkeypatch.setattr(sampling_runtime, "run_ksampler_as_custom", capture)
        latent = tiny_latent()
        control = control_conditioning()
        contributions = cast("Any", (object(),))

        result = sd1.sample(
            latent,
            cond=sd1.encode_text("facade composition"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=29,
            control=control,
            sd15_attention_contributions=contributions,
            compute_dtype=torch.float32,
            device="cpu",
        )

        assert result is output
        assert captured["owner"] is sd1
        assert captured["latent"] is latent
        assert captured["sample_custom_kwargs"] == {
            "control": control,
            "sd15_attention_contributions": contributions,
            "compute_dtype": torch.float32,
            "device": "cpu",
        }

    def test_custom_control_executes_identically_through_facade_and_direct_seam(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[float] = []

        def capture_gain(_evaluator: SDDenoiser, gain: float) -> None:
            selected.append(gain)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            for index in range(3):
                callback(index)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain", capture_gain)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("custom ControlNet parity")
        seed = 31
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        sigmas = sd1.custom_sampling_sigmas("dinkster.normal", 3, 1.0)
        control = control_conditioning(
            gain=ContributionGain(DirectGainTableCurve((1.0, 0.5, 0.0)), 1.0)
        )

        facade = sd1.sample(
            latent,
            cond=cond,
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=3,
            seed=seed,
            control=control,
            compute_dtype=torch.float32,
        )
        facade_gains = tuple(selected)
        selected.clear()
        direct = sd1.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=None,
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=seed,
            control=control,
            compute_dtype=torch.float32,
        )

        assert facade is latent
        assert direct.output is latent
        assert facade_gains == tuple(selected) == (1.0, 0.5, 0.0)

    def test_custom_sampling_perp_neg_zero_scale_matches_plain_cfg(self, sd1: SDRuntime) -> None:
        """neg_scale == 0 zeroes the perpendicular residual, so the guider
        reduces bit-exactly to classifier-free guidance from the empty
        prediction (the reference drops the negative lane the same way)."""
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:perp-neg-zero-parity",
        )
        latent = tiny_latent()
        cond = custom.encode_text("perp neg parity")
        negative = custom.encode_text("blurry")
        empty = custom.encode_text("")
        seed = 31
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        request = CustomSamplingRequest(sampler, (), sigmas)

        expected = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=SamplingGuidance(empty, 3.0),
            request=request,
            seed=seed,
        )
        result = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=PerpNegSamplingGuidance(negative, empty, 3.0, 0.0),
            request=request,
            seed=seed,
        )
        assert torch.equal(result.output, expected.output)

    def test_custom_sampling_perp_neg_changes_the_plain_cfg_output(self, sd1: SDRuntime) -> None:
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:perp-neg-active",
        )
        latent = tiny_latent()
        cond = custom.encode_text("perp neg parity")
        negative = custom.encode_text("blurry")
        empty = custom.encode_text("")
        seed = 31
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        request = CustomSamplingRequest(sampler, (), sigmas)

        plain = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=SamplingGuidance(empty, 3.0),
            request=request,
            seed=seed,
        )
        perp = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=PerpNegSamplingGuidance(negative, empty, 3.0, 1.0),
            request=request,
            seed=seed,
        )
        assert not torch.equal(perp.output, plain.output)

    def test_nag_executes_identically_on_the_sample_and_custom_paths(self, sd1: SDRuntime) -> None:
        """Both SD entry points opt the evaluator into attention-level
        guidance, so a NAG carrier produces bit-identical output on the
        KSampler-shaped sample() path and the decomposed custom path,
        and moves the output relative to plain CFG."""
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:nag-parity",
        )
        latent = tiny_latent()
        cond = custom.encode_text("nag parity")
        negative = custom.encode_text("blurry")
        seed = 37
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        nag = (("dinkster.nag:0", guidance_transforms.nag(5.0, 0.5, 1.5)),)

        expected = custom.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(negative, 3.0, transforms=nag),
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=2,
            seed=seed,
            compute_dtype=torch.float32,
        )
        plain = custom.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(negative, 3.0),
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=2,
            seed=seed,
            compute_dtype=torch.float32,
        )
        assert not torch.equal(expected, plain)
        result = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=SamplingGuidance(negative, 3.0, transforms=nag),
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=seed,
        )
        assert torch.equal(result.output, expected)

    def test_nag_requires_uncond_evaluates_the_negative_lane_at_cfg_one(
        self, sd1: SDRuntime
    ) -> None:
        """NAG's requires_uncond forces the unconditional lane into the
        sample() plan even at cfg 1, mirroring the reference's
        disable_model_cfg1_optimization."""
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:nag-cfg-one",
        )
        latent = tiny_latent()
        cond = custom.encode_text("nag cfg one")
        negative = custom.encode_text("blurry")
        nag = (("dinkster.nag:0", guidance_transforms.nag(5.0, 0.5, 1.5)),)

        def run(cfg: SamplingGuidance[Any] | None) -> torch.Tensor:
            return custom.sample(
                latent,
                cond=cond,
                cfg=cfg,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                seed=41,
                compute_dtype=torch.float32,
            )

        assert not torch.equal(
            run(SamplingGuidance(negative, 1.0, transforms=nag)),
            run(SamplingGuidance(negative, 1.0)),
        )

    @pytest.mark.parametrize(
        ("sampler_id", "scheduler_id", "steps", "denoise", "cfg_scale", "segment"),
        [
            ("dinkster.euler", "dinkster.normal", 3, None, None, None),
            ("dinkster.euler", "dinkster.karras", 4, 0.5, 2.0, None),
            ("dinkster.dpmpp_2m_sde", "dinkster.normal", 3, None, None, None),
            (
                "dinkster.euler",
                "dinkster.normal",
                4,
                None,
                None,
                SamplingSegment(
                    steps=4,
                    start_step=1,
                    end_step=4,
                    add_noise=False,
                    return_with_leftover_noise=False,
                ),
            ),
        ],
    )
    def test_ksampler_as_custom_matches_the_standard_sd_path(
        self,
        sd1: SDRuntime,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None,
        cfg_scale: float | None,
        segment: SamplingSegment | None,
    ) -> None:
        """The sample() KSampler surface is sugar over the custom
        sampling seam: run_ksampler_as_custom composes the identical
        schedule, noise draw, and solver, so outputs are bit-equal."""
        custom = SDRuntime(
            replace(sd1.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:ksampler-sugar-parity",
        )
        latent = tiny_latent()
        cond = custom.encode_text("ksampler sugar parity")
        cfg = (
            None
            if cfg_scale is None
            else SamplingGuidance(custom.encode_text("sugar parity negative"), cfg_scale)
        )
        seed = 29

        expected = custom.sample(
            latent,
            cond=cond,
            cfg=cfg,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            segment=segment,
            compute_dtype=torch.float32,
        )
        result = run_ksampler_as_custom(
            custom,
            latent,
            samplers=cast("Any", custom)._samplers,
            schedulers=cast("Any", custom)._schedulers,
            space=cast("Any", custom)._space,
            flow=False,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            cond=cond,
            cfg=cfg,
            segment=segment,
            error=WiringError,
        )

        assert isinstance(result.output, torch.Tensor)
        assert torch.equal(result.output, expected)

    def test_t2i_adapter_executes_through_control_path_and_caches_features(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def adapter_forward(_model: SD15T2IAdapter, hint: torch.Tensor) -> SDControlResiduals:
            nonlocal calls
            calls += 1
            batch, _, pixel_height, pixel_width = hint.shape
            height, width = pixel_height // 8, pixel_width // 8
            channels = (320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280)
            scales = (1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8)
            down = tuple(
                hint.new_zeros(batch, channel, height // scale, width // scale)
                for channel, scale in zip(channels, scales, strict=True)
            )
            return SDControlResiduals(down, hint.new_zeros(batch, 1280, height // 8, width // 8))

        def unet_forward(
            _model: UNetModel,
            x: torch.Tensor,
            _timesteps: torch.Tensor,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            control: SDControlResiduals | None = None,
            attention_guidance: object | None = None,
        ) -> torch.Tensor:
            assert y is None
            assert context.ndim == 3
            assert control is not None and len(control.down) == 12
            return torch.zeros_like(x)

        monkeypatch.setattr(SD15T2IAdapter, "forward", adapter_forward)
        monkeypatch.setattr(UNetModel, "forward", unet_forward)
        output = sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("adapter control"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            control=t2i_adapter_conditioning(gain=ContributionGain(ConstantGainCurve(1.0), 1.0)),
            compute_dtype=torch.float32,
        )
        assert output.shape == (1, 4, 8, 8)
        assert calls == 1

    def test_cfg_one_fixed_seed_output_digest(self, sd1: SDRuntime) -> None:
        output = sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("cfg one positive"),
            cfg=SamplingGuidance(sd1.encode_text("cfg one negative"), 1.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=17,
            compute_dtype=torch.float32,
        )
        digest = hashlib.sha256(
            bytes(output.contiguous().view(torch.uint8).flatten().tolist())
        ).hexdigest()
        assert digest == platform_digest("sd15_cfg_one")

    def test_declares_cross_attention_layout_for_fused_cfg(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        captured: list[GuidedDenoiser] = []
        original = sampling_execution_module.guided_denoiser

        def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original(*args, **kwargs)
            captured.append(guided)
            return guided

        monkeypatch.setattr(sampling_execution_module, "guided_denoiser", capture)
        cond = sd1.encode_text("layout positive")
        uncond = sd1.encode_text("layout negative")
        sd1.sample(
            tiny_latent(),
            cond=cond,
            cfg=SamplingGuidance(uncond, 2.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=11,
            compute_dtype=torch.float32,
        )

        token_count = cond.embeddings.shape[1]
        expected_layout = ModelTokenLayout(
            (
                ModelTokenSegment(
                    "text_context",
                    "text",
                    "context",
                    0,
                    token_count,
                    (token_count,),
                ),
            ),
            0,
        )
        expected_transform = TokenGridTransform(
            "sd.text-context-repeat.v1",
            "text",
            "text_context",
            (token_count,),
            None,
        )
        compiled = captured[0].conditioning_plan
        assert compiled is not None
        assert len(compiled.calls) == 1
        assert compiled.calls[0].layout_digest == expected_layout.digest
        assert all(
            lane.validation is ConditioningValidationPath.LAYOUT_BACKED for lane in compiled.lanes
        )
        assert all(
            lane.token_transforms == ((expected_transform.transform, expected_transform.digest),)
            for lane in compiled.lanes
        )

    def test_unequal_sd_text_lengths_fuse_against_one_declared_lcm_layout(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        captured: list[GuidedDenoiser] = []
        original = sampling_execution_module.guided_denoiser

        def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original(*args, **kwargs)
            captured.append(guided)
            return guided

        monkeypatch.setattr(sampling_execution_module, "guided_denoiser", capture)
        cond = declare_text_conditioning(
            Conditioning(torch.zeros(1, 2, TINY_CLIP_L.hidden_size)),
            2,
        )
        uncond = declare_text_conditioning(
            Conditioning(torch.zeros(1, 3, TINY_CLIP_L.hidden_size)),
            3,
        )
        sd1.sample(
            tiny_latent(),
            cond=cond,
            cfg=SamplingGuidance(uncond, 2.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=13,
            compute_dtype=torch.float32,
        )

        expected_layout = ModelTokenLayout(
            (
                ModelTokenSegment(
                    "text_context",
                    "text",
                    "context",
                    0,
                    6,
                    (6,),
                ),
            ),
            0,
        )
        expected_sources = (2, 3)
        compiled = captured[0].conditioning_plan
        assert compiled is not None
        assert len(compiled.calls) == 1
        assert compiled.calls[0].layout_digest == expected_layout.digest
        assert tuple(
            TokenGridTransform(
                "sd.text-context-repeat.v1",
                "text",
                "text_context",
                (source,),
                None,
            ).digest
            for source in expected_sources
        ) == tuple(lane.token_transforms[0][1] for lane in compiled.lanes)

    def test_sd15_base_runtime_accepts_sampler_mask(self, sd1: SDRuntime) -> None:
        latent = tiny_latent()
        output = sd1.sample(
            latent,
            cond=sd1.encode_text("a cat"),
            denoise_mask=torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )

        assert output.shape == latent.shape

    def test_sd15_runtime_accepts_sampler_mask_and_inpaint_requires_inpaint_model(
        self,
        sd1: SDRuntime,
        sd1_inpaint: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dinkster_inference_torch.sd_denoise as sd_denoise_module

        latent = tiny_latent()
        cond = sd1_inpaint.encode_text("a cat")
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        masked_image = torch.full_like(latent, 0.25)
        seen: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        original = sd_denoise_module.inpaint_model_input

        def capture(
            noise: torch.Tensor,
            *,
            denoise_mask: torch.Tensor | None,
            masked_image: torch.Tensor | None,
        ) -> torch.Tensor:
            seen.append((denoise_mask, masked_image))
            return original(
                noise,
                denoise_mask=denoise_mask,
                masked_image=masked_image,
            )

        monkeypatch.setattr(sd_denoise_module, "inpaint_model_input", capture)
        output = sd1_inpaint.sample(
            latent,
            cond=cond,
            denoise_mask=mask,
            inpaint=InpaintConditioning(mask=mask, masked_image=masked_image),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert output.shape == latent.shape
        assert output.dtype == torch.float32
        output_digest = hashlib.sha256(
            b"".join(struct.pack("<f", value) for value in output.flatten().tolist())
        ).hexdigest()
        assert output_digest == platform_digest("sd15_inpaint")
        assert seen
        assert seen[0][0] is mask
        seen_masked_image = seen[0][1]
        assert seen_masked_image is not None
        expected_masked = masked_image * SD15.single_stream_latent().scale_factor
        assert torch.equal(seen_masked_image, expected_masked)
        base_cond = sd1.encode_text("a cat")
        base_output = sd1.sample(
            latent,
            cond=base_cond,
            denoise_mask=mask,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )
        assert base_output.shape == latent.shape
        with pytest.raises(WiringError, match="^model does not support inpaint conditioning$"):
            sd1.sample(
                latent,
                cond=base_cond,
                inpaint=InpaintConditioning(mask=mask, masked_image=masked_image),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float32,
            )

    def test_disabled_initial_noise_preserves_ddim_inpaint_noise(
        self,
        sd1_inpaint: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        latent = tiny_latent()
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

        class Captured(RuntimeError):
            pass

        def capture(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
            noise = kwargs["noise"]
            inpaint_noise = kwargs["inpaint_noise"]
            assert isinstance(noise, torch.Tensor)
            assert isinstance(inpaint_noise, torch.Tensor)
            assert torch.equal(noise, torch.zeros_like(latent))
            assert torch.equal(inpaint_noise, prepare_noise(latent, 4))
            raise Captured

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture)
        with pytest.raises(Captured):
            sd1_inpaint.sample(
                latent,
                cond=sd1_inpaint.encode_text("a cat"),
                denoise_mask=mask,
                inpaint=InpaintConditioning(mask=mask, masked_image=torch.full_like(latent, 0.25)),
                sampler_id="dinkster.ddim",
                scheduler_id="dinkster.normal",
                steps=1,
                seed=3,
                segment=SamplingSegment(1, 0, 1, False, False),
                compute_dtype=torch.float32,
            )

    def test_sdxl_inpaint_runtime_preserves_adm_and_model_concat(
        self,
        sdxl: SDRuntime,
        sdxl_inpaint: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dinkster_inference_torch.sd_denoise as sd_denoise_module

        latent = tiny_latent()
        cond = sdxl_inpaint.encode_text("a cat")
        sampler_mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        model_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        masked_image = torch.full_like(latent, 0.25)
        seen_concat: list[torch.Tensor] = []
        seen_adm: list[torch.Tensor] = []
        seen_sampler_masks: list[torch.Tensor | None] = []
        original_concat = sd_denoise_module.inpaint_model_input
        original_forward = sdxl_inpaint.assembled.diffusion.forward
        original_drive = sampling_execution_module.run_denoise

        def capture_concat(
            noise: torch.Tensor,
            *,
            denoise_mask: torch.Tensor | None,
            masked_image: torch.Tensor | None,
        ) -> torch.Tensor:
            assert denoise_mask is model_mask
            value = original_concat(
                noise,
                denoise_mask=denoise_mask,
                masked_image=masked_image,
            )
            seen_concat.append(value)
            assert masked_image is not None
            assert torch.equal(
                masked_image,
                torch.full_like(latent, 0.25) * SDXL.single_stream_latent().scale_factor,
            )
            return value

        def capture_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            y = kwargs.get("y")
            assert isinstance(y, torch.Tensor)
            seen_adm.append(y)
            return original_forward(*args, **kwargs)

        def capture_drive(*args: Any, **kwargs: Any) -> torch.Tensor:
            denoise_mask = kwargs.get("denoise_mask")
            assert denoise_mask is None or isinstance(denoise_mask, torch.Tensor)
            seen_sampler_masks.append(denoise_mask)
            return original_drive(*args, **kwargs)

        monkeypatch.setattr(sd_denoise_module, "inpaint_model_input", capture_concat)
        monkeypatch.setattr(sdxl_inpaint.assembled.diffusion, "forward", capture_forward)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_drive)
        output = sdxl_inpaint.sample(
            latent,
            cond=cond,
            denoise_mask=sampler_mask,
            inpaint=InpaintConditioning(mask=model_mask, masked_image=masked_image),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert output.shape == latent.shape
        assert seen_concat and seen_concat[0].shape == (1, 9, 8, 8)
        assert len(seen_sampler_masks) == 1
        assert seen_sampler_masks[0] is sampler_mask
        assert seen_adm and seen_adm[0].shape == (
            1,
            TINY_SDXL_INPAINT_UNET.adm_in_channels,
        )
        base_cond = sdxl.encode_text("a cat")
        base_output = sdxl.sample(
            latent,
            cond=base_cond,
            denoise_mask=sampler_mask,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float32,
        )
        assert base_output.shape == latent.shape
        with pytest.raises(WiringError, match="does not support inpaint conditioning"):
            sdxl.sample(
                latent,
                cond=base_cond,
                inpaint=InpaintConditioning(mask=model_mask, masked_image=masked_image),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float32,
            )

    @pytest.mark.parametrize(
        "receipt_identity",
        (
            None,
            "distributed:dinkster.sd15:"
            "9d610c932af7e0355b4cb7acf41b300ecd6c3d2ae1d47f2c84fb20595dbf5e14",
        ),
        ids=("no-identity", "withdrawn-d1-identity"),
    )
    def test_sd15_world_size_two_guidance_executes_with_unregistered_receipt(
        self,
        sd1: SDRuntime,
        receipt_identity: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)
        model_runs: list[object] = []

        def capture_run(*_args: object, **_kwargs: object) -> None:
            model_runs.append(1)
            raise RuntimeError("denoising reached")

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=receipt_identity,
        )
        with pytest.raises(RuntimeError, match="denoising reached"):
            runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(sd1.encode_text(""), 7.0),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float16,
                device="cpu",
            )
        assert model_runs == [1]

    def test_synthetic_receipt_admits_facade_and_direct_custom_requests(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)
        denoisers: list[GuidedDenoiser] = []

        def capture_run(denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
            assert isinstance(denoiser, GuidedDenoiser)
            denoisers.append(denoiser)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
        )
        latent = tiny_latent()
        cond = sd1.encode_text("distributed custom parity")
        cfg = SamplingGuidance(sd1.encode_text(""), 7.0)
        seed = 43
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        sigmas = runtime.custom_sampling_sigmas("dinkster.normal", 1, 1.0)

        facade = runtime.sample(
            latent,
            cond=cond,
            cfg=cfg,
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=1,
            seed=seed,
            compute_dtype=torch.float16,
            device="cuda:0",
        )
        direct = runtime.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=cfg,
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=seed,
            compute_dtype=torch.float16,
            device="cuda:0",
        )

        assert facade is latent
        assert direct.output is latent
        assert len(denoisers) == 2
        assert all(
            denoiser._replica_evaluator is not None  # pyright: ignore[reportPrivateUsage]
            for denoiser in denoisers
        )
        assert cfg.uncond is not None
        for guidance, options in (
            (PerpNegSamplingGuidance(cfg.uncond, sd1.encode_text("empty"), 7.0, 1.0), ()),
            (cfg, (("s_churn", 1.0),)),
        ):
            assert (
                runtime.sample_custom(
                    latent,
                    noise=prepare_noise(latent, seed),
                    cond=cond,
                    cfg=guidance,
                    request=CustomSamplingRequest(sampler, options, sigmas),
                    seed=seed,
                    compute_dtype=torch.float16,
                    device="cuda:0",
                ).output
                is latent
            )

    def test_distributed_admission_preserves_extension_behavior(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A receipt does not restrict registered samplers or guidance contributions."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")

        def sample(runtime: SDRuntime, **overrides: Any) -> torch.Tensor:
            kwargs: dict[str, Any] = {
                "cond": cond,
                "cfg": SamplingGuidance(uncond, 7.0),
                "sampler_id": "dinkster.euler",
                "scheduler_id": "dinkster.normal",
                "steps": 1,
                "compute_dtype": torch.float16,
                "device": "cuda:0",
            }
            kwargs.update(overrides)
            return runtime.sample(latent, **kwargs)

        contribution: GuidanceContribution[torch.Tensor] = GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor("ext.scale", lambda context: context.reduced * 1000.0),
            )
        )
        guided = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=GuidanceExecutor(GuidanceRegistry((("ext", contribution),))),
        )
        assert sample(guided) is latent

        inactive = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=GuidanceExecutor(GuidanceRegistry()),
        )
        assert sample(inactive) is latent

        extension_samplers: Registry[Any] = Registry()
        for descriptor in builtin_samplers():
            extension_samplers.register(descriptor)
        euler = next(
            descriptor
            for descriptor in torch_sampler_registry()
            if descriptor.id == "dinkster.euler"
        )
        extension_samplers.register(
            replace(
                euler,
                id="ext.euler_scaled",
                display_name="Ext Euler",
                aliases=(),
            )
        )
        with_extension_samplers = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            sampler_registry=extension_samplers,
        )
        assert sample(with_extension_samplers) is latent
        assert sample(with_extension_samplers, sampler_id="ext.euler_scaled") is latent

        # A scheduler is outside the execution engine: KSampler sugar turns
        # it into the exact sigma table that direct custom sampling accepts.
        schedule_calls: list[int] = []
        builtin_normal = next(
            descriptor for descriptor in builtin_schedulers() if descriptor.id == "dinkster.normal"
        )

        def counting_make_sigmas(steps: int, space: Any) -> tuple[float, ...]:
            schedule_calls.append(steps)
            return builtin_normal.make_sigmas(steps, space)

        custom_schedulers: Registry[Any] = Registry()
        for descriptor in builtin_schedulers():
            custom_schedulers.register(descriptor)
        custom_schedulers.register(
            replace(
                builtin_schedulers()[0],
                id="ext.sched",
                display_name="Ext Schedule",
                make_sigmas=counting_make_sigmas,
                aliases=(),
            )
        )
        with_custom_schedulers = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            scheduler_registry=custom_schedulers,
        )
        assert sample(with_custom_schedulers, scheduler_id="ext.sched") is latent
        assert schedule_calls == [1]

    def test_distributed_admission_forwards_progress_callbacks(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same callback hooks reach the shared engine on either sampling path."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
        )

        def sample(**overrides: Any) -> torch.Tensor:
            kwargs: dict[str, Any] = {
                "cond": cond,
                "cfg": SamplingGuidance(uncond, 7.0),
                "sampler_id": "dinkster.euler",
                "scheduler_id": "dinkster.normal",
                "steps": 1,
                "compute_dtype": torch.float16,
                "device": "cuda:0",
            }
            kwargs.update(overrides)
            return runtime.sample(latent, **kwargs)

        calls: list[object] = []
        assert sample(on_step=calls.append) is latent
        assert sample(on_state=calls.append) is latent
        assert calls == []
        assert sample() is latent

    def test_distributed_admission_preserves_per_run_guidance_transforms(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Per-run guidance contributions are accepted independently of receipts."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
        )
        executed: list[object] = []

        def record(context: Any) -> torch.Tensor:
            executed.append(context)
            return cast("torch.Tensor", context.reduced)

        transforms = (
            (
                "ext",
                GuidanceContribution(post_cfg=(GuidancePostCFGDescriptor("ext.noop", record),)),
            ),
        )

        def sample(cfg: SamplingGuidance[Any]) -> torch.Tensor:
            return runtime.sample(
                latent,
                cond=cond,
                cfg=cfg,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float16,
                device="cuda:0",
            )

        assert sample(SamplingGuidance(uncond, 7.0, transforms)) is latent
        assert executed == []
        assert sample(SamplingGuidance(uncond, 7.0)) is latent

    def test_distributed_admission_preserves_ambient_cancellation(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation is checked during execution, not by an admission type allowlist."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
        )

        def sample() -> torch.Tensor:
            return runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, 7.0),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float16,
                device="cuda:0",
            )

        calls: list[object] = []

        def untrusted() -> bool:
            calls.append(None)
            return False

        with use_sampling_environment((), untrusted):
            assert sample() is latent
        assert calls == []

        class FlagAlike(CancellationFlag):  # pyright: ignore[reportGeneralTypeIssues]
            def __call__(self) -> bool:
                raise AssertionError("subclass cancellation must never execute")

        with use_sampling_environment((), FlagAlike()):
            assert sample() is latent

        with use_sampling_environment((), CancellationFlag()):
            assert sample() is latent
        assert sample() is latent

    def test_distributed_custom_catalog_does_not_mutate_builtin_snapshots(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Builtin snapshots remain canonical while explicit registry overrides execute."""
        import dinkster_inference_torch.distributed as distributed_module
        from dinkster_inference.solvers import DINKSTER_HEUN

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")
        canonical_make = DINKSTER_HEUN.make

        def attacker_make(_options: object) -> object:
            raise AssertionError("attacker solver must never build")

        def attacker_build(**_overrides: object) -> object:
            return attacker_make

        object.__setattr__(DINKSTER_HEUN, "make", attacker_make)
        object.__setattr__(DINKSTER_HEUN, "build", attacker_build)
        try:
            fresh = next(
                descriptor for descriptor in builtin_samplers() if descriptor.id == "dinkster.heun"
            )
            assert fresh is not DINKSTER_HEUN
            assert fresh.make is canonical_make
            assert "build" not in vars(fresh)

            poisoned_registry: Registry[Any] = Registry()
            for descriptor in builtin_samplers():
                poisoned_registry.register(
                    DINKSTER_HEUN if descriptor.id == "dinkster.heun" else descriptor
                )
            runtime = SDRuntime(
                sd1.assembled,
                runtime_identity="native:dinkster.sd15:deployment-specific",
                receipt_identity=synthetic_sd15_receipt,
                sampler_registry=poisoned_registry,
            )
            assert (
                runtime.sample(
                    latent,
                    cond=cond,
                    cfg=SamplingGuidance(uncond, 7.0),
                    sampler_id="dinkster.heun",
                    scheduler_id="dinkster.normal",
                    steps=1,
                    compute_dtype=torch.float16,
                    device="cuda:0",
                )
                is latent
            )
        finally:
            object.__delattr__(DINKSTER_HEUN, "build")
            object.__setattr__(DINKSTER_HEUN, "make", canonical_make)

    def test_distributed_admission_accepts_wrapped_callables_and_executor_subclasses(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Execution uses the configured registry, not equality with a builtin catalog."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")

        def sample(runtime: SDRuntime, **overrides: Any) -> torch.Tensor:
            kwargs: dict[str, Any] = {
                "cond": cond,
                "cfg": SamplingGuidance(uncond, 7.0),
                "sampler_id": "dinkster.euler",
                "scheduler_id": "dinkster.normal",
                "steps": 1,
                "compute_dtype": torch.float16,
                "device": "cuda:0",
            }
            kwargs.update(overrides)
            return runtime.sample(latent, **kwargs)

        class EqualitySpoof:
            """Claims equality with everything while wrapping different code."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return self._inner(*args, **kwargs)

            def __eq__(self, _other: object) -> bool:
                return True

            def __hash__(self) -> int:
                return 0

        builtin_heun = next(
            descriptor
            for descriptor in torch_sampler_registry()
            if descriptor.id == "dinkster.heun"
        )
        spoofed_samplers: Registry[Any] = Registry()
        for descriptor in builtin_samplers():
            spoofed_samplers.register(
                replace(descriptor, make=EqualitySpoof(builtin_heun.make))
                if descriptor.id == "dinkster.heun"
                else descriptor
            )
        assert replace(builtin_heun, make=EqualitySpoof(builtin_heun.make)) == builtin_heun
        spoofed_sampler_runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            sampler_registry=spoofed_samplers,
        )
        assert sample(spoofed_sampler_runtime, sampler_id="dinkster.heun") is latent

        spoofed_schedulers: Registry[Any] = Registry()
        for descriptor in builtin_schedulers():
            spoofed_schedulers.register(
                replace(descriptor, make_sigmas=EqualitySpoof(descriptor.make_sigmas))
                if descriptor.id == "dinkster.normal"
                else descriptor
            )
        spoofed_scheduler_runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            scheduler_registry=spoofed_schedulers,
        )
        assert sample(spoofed_scheduler_runtime) is latent

        class SneakyExecutor(GuidanceExecutor):
            pass

        subclassed_executor = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=SneakyExecutor(GuidanceRegistry()),
        )
        assert sample(subclassed_executor) is latent

        class LyingRegistry(GuidanceRegistry):
            @property
            def active(self) -> bool:
                return False

        contribution: GuidanceContribution[torch.Tensor] = GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor("ext.scale", lambda context: context.reduced * 1000.0),
            )
        )
        lying_registry = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=GuidanceExecutor(LyingRegistry((("ext", contribution),))),
        )
        assert sample(lying_registry) is latent

    def test_inactive_registry_uses_a_fresh_builtin_executor(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The generic inactive-registry safeguard must not reuse caller state.

        An instance-shadowed ``execute`` on a genuine executor survives
        registry inspection, and the caller can mutate that registry after
        plan construction. Plain guidance therefore gets a fresh builtin
        executor independently of receipt observation.
        """
        import dinkster_inference_torch.distributed as distributed_module
        from dinkster_inference_torch.guidance import GuidedDenoiser

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
        captured: list[Any] = []

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            captured.append(args[0])
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)

        smuggled_calls: list[object] = []

        def smuggled(*args: object, **kwargs: object) -> object:
            smuggled_calls.append((args, kwargs))
            raise AssertionError("shadowed execute must never run")

        shadowed = GuidanceExecutor(GuidanceRegistry())
        cast(Any, shadowed).execute = smuggled
        assert type(shadowed) is GuidanceExecutor
        assert not shadowed.registry.active
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=shadowed,
        )
        latent = tiny_latent()
        result = runtime.sample(
            latent,
            cond=sd1.encode_text("a cat"),
            cfg=SamplingGuidance(sd1.encode_text(""), 7.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float16,
            device="cuda:0",
        )
        assert result is latent
        assert smuggled_calls == []
        (denoiser,) = captured
        assert isinstance(denoiser, GuidedDenoiser)
        used = denoiser._executor  # pyright: ignore[reportPrivateUsage]
        assert used is not shadowed
        assert type(used) is GuidanceExecutor
        assert "execute" not in vars(used)
        # A fresh registry: mutating the admitted object's registry after
        # admission cannot activate contributions inside sampling.
        assert used.registry is not shadowed.registry
        assert not used.registry.active

    def test_distributed_admission_executes_registered_descriptor_build(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit registry override executes and propagates its own build failure."""
        import dinkster_inference_torch.distributed as distributed_module

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")

        def sample(runtime: SDRuntime, **overrides: Any) -> torch.Tensor:
            kwargs: dict[str, Any] = {
                "cond": cond,
                "cfg": SamplingGuidance(uncond, 7.0),
                "sampler_id": "dinkster.euler",
                "scheduler_id": "dinkster.normal",
                "steps": 1,
                "compute_dtype": torch.float16,
                "device": "cuda:0",
            }
            kwargs.update(overrides)
            return runtime.sample(latent, **kwargs)

        smuggled_builds: list[object] = []

        def smuggled_build(**overrides: object) -> object:
            smuggled_builds.append(overrides)
            raise AssertionError("shadowed build must never run")

        builtin_heun = next(
            descriptor for descriptor in builtin_samplers() if descriptor.id == "dinkster.heun"
        )
        shadowed_heun = replace(builtin_heun)
        object.__setattr__(shadowed_heun, "build", smuggled_build)
        assert shadowed_heun == builtin_heun
        shadowed_samplers: Registry[Any] = Registry()
        for descriptor in builtin_samplers():
            shadowed_samplers.register(
                shadowed_heun if descriptor.id == "dinkster.heun" else descriptor
            )
        shadowed_sampler_runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            sampler_registry=shadowed_samplers,
        )
        with pytest.raises(AssertionError, match="shadowed build must never run"):
            sample(shadowed_sampler_runtime, sampler_id="dinkster.heun")
        assert len(smuggled_builds) == 1

        # Scheduler construction belongs to KSampler composition, outside
        # the custom execution engine, so inert descriptor state is accepted.
        builtin_normal = next(
            descriptor
            for descriptor in torch_scheduler_registry()
            if descriptor.id == "dinkster.normal"
        )
        shadowed_normal = replace(builtin_normal)
        object.__setattr__(shadowed_normal, "smuggled", object())
        assert shadowed_normal == builtin_normal
        shadowed_schedulers: Registry[Any] = Registry()
        for descriptor in torch_scheduler_registry():
            shadowed_schedulers.register(
                shadowed_normal if descriptor.id == "dinkster.normal" else descriptor
            )
        shadowed_scheduler_runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            scheduler_registry=shadowed_schedulers,
        )
        assert sample(shadowed_scheduler_runtime) is latent

    def test_runtime_copies_caller_scheduler_registry_into_plain_registry(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Caller-supplied scheduler registries are copied at construction:
        a Registry subclass overriding ``get`` never executes during
        sampling, including distributed execution."""
        import dinkster_inference_torch.distributed as distributed_module

        class SneakyRegistry(Registry[Any]):
            def __init__(self) -> None:
                super().__init__()
                self.get_calls: list[str] = []

            def get(self, id_or_alias: str) -> Any:
                self.get_calls.append(id_or_alias)
                return super().get(id_or_alias)

        sneaky = SneakyRegistry()
        for descriptor in torch_scheduler_registry():
            sneaky.register(descriptor)
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            scheduler_registry=sneaky,
        )
        assert type(runtime._schedulers) is Registry  # pyright: ignore[reportPrivateUsage]

        distributed = type("Distributed", (), {"world_size": 2, "mode": "guidance"})()
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: distributed)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: distributed)

        def capability(_device: object) -> tuple[int, int]:
            return (8, 9)

        monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        latent = tiny_latent()
        result = runtime.sample(
            latent,
            cond=sd1.encode_text("a cat"),
            cfg=SamplingGuidance(sd1.encode_text(""), 7.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            compute_dtype=torch.float16,
            device="cuda:0",
        )
        assert result is latent
        assert sneaky.get_calls == []

    def test_shape_and_determinism(self, sd1: SDRuntime) -> None:
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        out = sd1.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert out.shape == latent.shape
        again = sd1.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert torch.equal(out, again)

    def test_ddim_wires_seed_plus_one_inpaint_noise_only(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        calls: list[dict[str, Any]] = []

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            calls.append(kwargs)
            value = kwargs["latent"]
            assert isinstance(value, torch.Tensor)
            return value

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture)
        global_state = torch.random.get_rng_state()
        for sampler_id in ("ddim", "euler"):
            result = sd1.sample(
                latent,
                cond=cond,
                sampler_id=sampler_id,
                scheduler_id="normal",
                steps=2,
                seed=41,
                compute_dtype=torch.float32,
            )
            assert result is latent

        assert torch.equal(torch.random.get_rng_state(), global_state)
        assert torch.equal(calls[0]["noise"], prepare_noise(latent, 41))
        assert torch.equal(calls[0]["inpaint_noise"], prepare_noise(latent, 42))
        assert calls[1]["inpaint_noise"] is None

    def test_sd1_matches_the_documented_recipe(self, sd1: SDRuntime) -> None:
        """sample() is schedule -> prepare_noise -> SDDenoiser ->
        run_denoise over the linear-beta discrete table, no ADM."""
        latent = tiny_latent()
        cond = sd1.encode_text("a cat")
        uncond = sd1.encode_text("")
        actual = sd1.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.euler")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = DiscreteSigmas.linear_beta()
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = SDDenoiser(
            sd1.assembled.diffusion,
            space,
            compute_dtype=torch.float32,
        )
        denoiser = guided_sd_denoiser(
            evaluator,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler=sampler,
            sigmas=sigmas,
            seed=7,
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=SD15,
            seed=7,
            noise_kind=sampler.noise,
        )
        assert torch.equal(actual, expected)

    def test_sdxl_adm_uses_the_encode_model_conds_defaults(self, sdxl: SDRuntime) -> None:
        """encode_model_conds @ 947c2749 fills width/height from the
        latent's pixel dimensions (latent * 8) and leaves crops and
        targets to encode_adm's defaults - both conds get their own
        pooled vector."""
        latent = tiny_latent()
        cond = sdxl.encode_text("a cat")
        uncond = sdxl.encode_text("")
        actual = sdxl.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.euler")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = DiscreteSigmas.linear_beta()
        assert cond.pooled is not None and uncond.pooled is not None
        scale = SDXL.single_stream_latent().spatial_downscale
        width = latent.shape[3] * scale
        height = latent.shape[2] * scale
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = SDDenoiser(
            sdxl.assembled.diffusion,
            space,
            compute_dtype=torch.float32,
        )
        denoiser = guided_sd_denoiser(
            evaluator,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler=sampler,
            sigmas=sigmas,
            seed=7,
            adm_cond=encode_sdxl_adm(cond.pooled, width=width, height=height),
            adm_uncond=encode_sdxl_adm(uncond.pooled, width=width, height=height),
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=SDXL,
            seed=7,
            noise_kind=sampler.noise,
        )
        assert torch.equal(actual, expected)

    def test_sdxl_prepares_adm_once_per_guidance_lane(
        self,
        sdxl: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cond = sdxl.encode_text("a cat")
        uncond = sdxl.encode_text("")
        calls: list[bool] = []
        original = SDRuntime._adm  # pyright: ignore[reportPrivateUsage]

        def counted(
            runtime: SDRuntime,
            conditioning: Conditioning[torch.Tensor] | None,
            latent: torch.Tensor,
            *,
            negative: bool,
        ) -> torch.Tensor | None:
            calls.append(negative)
            return original(runtime, conditioning, latent, negative=negative)

        monkeypatch.setattr(SDRuntime, "_adm", counted)
        sdxl.sample(
            tiny_latent(),
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            seed=7,
            compute_dtype=torch.float32,
        )
        assert calls == [False, True]

    def test_sdxl_prepares_adm_for_each_perp_neg_lane(
        self,
        sdxl: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Perp-neg binds three lanes; only the conditional one prepares
        positive-polarity ADM - the negative and empty lanes both take the
        negative target-size handling."""
        custom = SDRuntime(
            replace(sdxl.assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:perp-neg-adm",
        )
        cond = custom.encode_text("a cat")
        uncond = custom.encode_text("blurry")
        empty = custom.encode_text("")
        calls: list[bool] = []
        original = SDRuntime._adm  # pyright: ignore[reportPrivateUsage]

        def counted(
            runtime: SDRuntime,
            conditioning: Conditioning[torch.Tensor] | None,
            latent: torch.Tensor,
            *,
            negative: bool,
        ) -> torch.Tensor | None:
            calls.append(negative)
            return original(runtime, conditioning, latent, negative=negative)

        monkeypatch.setattr(SDRuntime, "_adm", counted)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        latent = tiny_latent()
        custom.sample_custom(
            latent,
            noise=prepare_noise(latent, 7),
            cond=cond,
            cfg=PerpNegSamplingGuidance(uncond, empty, 3.0, 1.0),
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=7,
        )
        assert calls == [False, True, True]

    def test_sdxl_vprediction_and_zsnr_reach_the_sampling_drive(self, sdxl: SDRuntime) -> None:
        sampling = SamplingDescriptor(
            Parameterization.V_PREDICTION,
            SDXL.sampling.sigma_min,
            SDXL.sampling.sigma_max,
            zsnr=True,
        )
        assembled = AssembledSD(
            family=sdxl.assembled.family,
            diffusion=sdxl.assembled.diffusion,
            clip_l=sdxl.assembled.clip_l,
            clip_g=sdxl.assembled.clip_g,
            vae=sdxl.assembled.vae,
            sampling=sampling,
        )
        runtime = SDRuntime(assembled, runtime_identity="native:test:sdxl-vpred-zsnr")
        latent = tiny_latent()
        cond = runtime.encode_text("a cat")
        actual = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.euler")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = DiscreteSigmas.linear_beta(zsnr=True)
        assert cond.pooled is not None
        scale = SDXL.single_stream_latent().spatial_downscale
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = SDDenoiser(
            assembled.diffusion,
            space,
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        denoiser = guided_sd_denoiser(
            evaluator,
            cond=cond,
            cfg=None,
            sampler=sampler,
            sigmas=sigmas,
            seed=7,
            adm_cond=encode_sdxl_adm(
                cond.pooled,
                width=latent.shape[3] * scale,
                height=latent.shape[2] * scale,
            ),
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=SDXL,
            sampling=sampling,
            seed=7,
            noise_kind=sampler.noise,
        )
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize(
        "sampler_id",
        [
            "dinkster.euler",
            "dinkster.euler_cfg_pp",
            "dinkster.dpmpp_2m_sde",
            "dinkster.sa_solver",
        ],
    )
    def test_sdxl_edm_vprediction_bounds_reach_the_sampling_drive(
        self, sdxl: SDRuntime, sampler_id: str
    ) -> None:
        sampling = SamplingDescriptor(
            Parameterization.V_PREDICTION,
            0.125,
            42.5,
            space=SamplingSpace.CONTINUOUS_EDM,
        )
        assembled = AssembledSD(
            family=sdxl.assembled.family,
            diffusion=sdxl.assembled.diffusion,
            clip_l=sdxl.assembled.clip_l,
            clip_g=sdxl.assembled.clip_g,
            vae=sdxl.assembled.vae,
            sampling=sampling,
        )
        runtime = SDRuntime(assembled, runtime_identity="native:test:sdxl-edm-vpred")
        latent = tiny_latent()
        cond = runtime.encode_text("a cat")
        uncond = runtime.encode_text("")
        actual = runtime.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id=sampler_id,
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get(sampler_id)
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = ContinuousEDMSigmas(min_sigma=0.125, max_sigma=42.5)
        assert cond.pooled is not None and uncond.pooled is not None
        scale = SDXL.single_stream_latent().spatial_downscale
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = SDDenoiser(
            assembled.diffusion,
            space,
            parameterization=Parameterization.V_PREDICTION,
            compute_dtype=torch.float32,
        )
        denoiser = guided_sd_denoiser(
            evaluator,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler=sampler,
            sigmas=sigmas,
            seed=7,
            adm_cond=encode_sdxl_adm(
                cond.pooled,
                width=latent.shape[3] * scale,
                height=latent.shape[2] * scale,
            ),
            adm_uncond=encode_sdxl_adm(
                uncond.pooled,
                width=latent.shape[3] * scale,
                height=latent.shape[2] * scale,
            ),
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=SDXL,
            sampling=sampling,
            seed=7,
            noise_kind=sampler.noise,
            percent_to_sigma=lambda percent: continuous_edm_percent_to_sigma(space, percent),
        )
        assert torch.equal(actual, expected)

    def test_refiner_aesthetic_score_follows_prompt_polarity(self, refiner: SDRuntime) -> None:
        """The reference defaults the refiner's aesthetic score by
        polarity (6.0 positive, 2.5 negative - nodes.py CLIPTextEncode
        has no score input, so KSampler runs hit exactly these)."""
        latent = tiny_latent()
        cond = refiner.encode_text("a cat")
        uncond = refiner.encode_text("")
        actual = refiner.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.euler")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = DiscreteSigmas.linear_beta()
        assert cond.pooled is not None and uncond.pooled is not None
        scale = SDXL_REFINER.single_stream_latent().spatial_downscale
        width = latent.shape[3] * scale
        height = latent.shape[2] * scale
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = SDDenoiser(
            refiner.assembled.diffusion,
            space,
            compute_dtype=torch.float32,
        )
        denoiser = guided_sd_denoiser(
            evaluator,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler=sampler,
            sigmas=sigmas,
            seed=7,
            adm_cond=encode_sdxl_refiner_adm(cond.pooled, width=width, height=height),
            adm_uncond=encode_sdxl_refiner_adm(
                uncond.pooled,
                width=width,
                height=height,
                aesthetic_score=SDXL_NEGATIVE_AESTHETIC_DEFAULT,
            ),
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=SDXL_REFINER,
            seed=7,
            noise_kind=sampler.noise,
        )
        assert torch.equal(actual, expected)

    def test_zero_denoise_returns_latent_untouched(self, sd1: SDRuntime) -> None:
        latent = tiny_latent()
        out = sd1.sample(
            latent,
            cond=sd1.encode_text("a cat"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            denoise=0.0,
        )
        assert torch.equal(out, latent)

    def test_zero_denoise_still_validates_conditioning(self, sd1: SDRuntime) -> None:
        with pytest.raises(DenoiseError, match="embeddings must be"):
            sd1.sample(
                tiny_latent(),
                cond=Conditioning(torch.zeros(1, 4)),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                denoise=0.0,
            )

    def test_guidance_refuses(self, sd1: SDRuntime) -> None:
        with pytest.raises(WiringError, match="distilled-guidance input"):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("a cat"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                guidance=3.5,
            )

    def test_unknown_ids_refuse_by_name(self, sd1: SDRuntime) -> None:
        with pytest.raises(WiringError, match="unknown sampler 'nope'"):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("a cat"),
                sampler_id="nope",
                scheduler_id="dinkster.normal",
                steps=1,
            )
        with pytest.raises(WiringError, match="unknown scheduler 'nope'"):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("a cat"),
                sampler_id="dinkster.euler",
                scheduler_id="nope",
                steps=1,
            )

    def test_zero_control_gain_is_bit_exact_to_no_control(self, sd1: SDRuntime) -> None:
        cond = sd1.encode_text("zero control")
        latent = tiny_latent()
        baseline = sd1.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=239,
            compute_dtype=torch.float32,
        )
        controlled = sd1.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=239,
            control=control_conditioning(gain=ContributionGain(ConstantGainCurve(0.0), 1.0)),
            compute_dtype=torch.float32,
        )
        assert torch.equal(controlled, baseline)

    def test_control_admission_refuses_hint_mutated_after_construction(
        self, sd1: SDRuntime
    ) -> None:
        control = control_conditioning()
        control.hint.fill_(1.0)
        with pytest.raises(ControlResourceBindingError, match="materialized hint tensor"):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("mutated control hint"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                control=control,
                compute_dtype=torch.float32,
            )

    def test_scalar_and_constant_control_schedules_select_identical_rows(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[float] = []
        original_set = SDDenoiser.set_control_gain

        def capture_gain(evaluator: SDDenoiser, gain: float) -> None:
            selected.append(gain)
            original_set(evaluator, gain)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            callback = kwargs["on_step_begin"]
            sigmas = kwargs["sigmas"]
            assert callable(callback) and isinstance(sigmas, Sequence)
            for index in range(len(sigmas) - 1):
                callback(index)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain", capture_gain)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        cond = sd1.encode_text("constant parity")
        sd1.sample(
            tiny_latent(),
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(strength=0.4),
            compute_dtype=torch.float32,
        )
        scalar_rows = tuple(selected)
        selected.clear()
        sd1.sample(
            tiny_latent(),
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(gain=ContributionGain(ConstantGainCurve(0.4), 1.0)),
            compute_dtype=torch.float32,
        )
        assert tuple(selected) == scalar_rows == (0.4, 0.4, 0.4)

    def test_control_window_uses_reference_float32_percent_boundary(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        reference = CUSTOM_SIGMA_GOLDENS["percent_to_sigma"]["dinkster.sd15"]["0.37,False"]
        first_sigma = math.nextafter(reference, math.inf)

        def boundary_sigmas(_steps: int, _space: object) -> tuple[float, ...]:
            return (first_sigma, 0.0)

        schedulers: Registry[Any] = Registry()
        for descriptor in builtin_schedulers():
            schedulers.register(
                replace(descriptor, make_sigmas=boundary_sigmas)
                if descriptor.id == "dinkster.normal"
                else descriptor
            )
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:test:sd15-percent-control-boundary",
            scheduler_registry=schedulers,
        )
        selected: list[float] = []

        def capture_gain(_evaluator: SDDenoiser, gain: float) -> None:
            selected.append(gain)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            callback(0)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain", capture_gain)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        runtime.sample(
            tiny_latent(),
            cond=runtime.encode_text("reference percent boundary"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            control=control_conditioning(window=PercentRange(0.37, 1.0)),
            compute_dtype=torch.float32,
        )
        assert selected == [0.0]

    def test_direct_control_schedule_selects_rows_and_binds_identity(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[float] = []
        evaluator_identities: list[str] = []

        def capture_gain(_evaluator: SDDenoiser, gain: float) -> None:
            selected.append(gain)

        monkeypatch.setattr(SDDenoiser, "set_control_gain", capture_gain)

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            denoiser = args[0]
            assert isinstance(denoiser, GuidedDenoiser)
            conditioning_plan = denoiser.conditioning_plan
            assert conditioning_plan is not None
            evaluator_identities.append(conditioning_plan.calls[0].evaluator_identity)
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            for index in range(3):
                callback(index)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("scheduled control"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(
                gain=ContributionGain(DirectGainTableCurve((1.0, 0.25, 0.0)), 0.5)
            ),
            compute_dtype=torch.float32,
        )
        assert selected == [0.5, 0.125, 0.0]
        prefix = "dinkster.sd.conditioning.v1:intervention-plan="
        assert evaluator_identities[0].startswith(prefix)
        assert len(evaluator_identities[0].removeprefix(prefix)) == 64

        original_identity = evaluator_identities[0]
        evaluator_identities.clear()
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("scheduled control"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(
                gain=ContributionGain(DirectGainTableCurve((1.0, 0.25, 0.0)), 0.5),
                model_digest="b" * 64,
            ),
            compute_dtype=torch.float32,
        )
        model_changed_identity = evaluator_identities[0]
        evaluator_identities.clear()
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("scheduled control"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(
                gain=ContributionGain(DirectGainTableCurve((1.0, 0.25, 0.0)), 0.5),
                hint_value=1.0,
            ),
            compute_dtype=torch.float32,
        )
        assert len({original_identity, model_changed_identity, evaluator_identities[0]}) == 3

    def test_non_euler_fixed_row_sampler_selects_every_control_row(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        selected: list[float] = []

        def capture_gain(_evaluator: SDDenoiser, gain: float) -> None:
            selected.append(gain)

        monkeypatch.setattr(SDDenoiser, "set_control_gain", capture_gain)
        result = sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("non-euler scheduled control"),
            sampler_id="dinkster.dpmpp_2m",
            scheduler_id="dinkster.normal",
            steps=3,
            control=control_conditioning(
                gain=ContributionGain(DirectGainTableCurve((1.0, 0.25, 0.0)), 0.5)
            ),
            compute_dtype=torch.float32,
        )
        assert result.shape == (1, 4, 8, 8)
        assert selected == [0.5, 0.125, 0.0]

    def test_control_chain_selects_independent_rows_oldest_to_newest(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[tuple[float, ...]] = []
        identities: list[str] = []

        def capture_gains(_evaluator: SDDenoiser, gains: tuple[float, ...]) -> None:
            selected.append(gains)

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            denoiser = args[0]
            assert isinstance(denoiser, GuidedDenoiser)
            plan = denoiser.conditioning_plan
            assert plan is not None
            identities.append(plan.calls[0].evaluator_identity)
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            for index in range(3):
                callback(index)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gains", capture_gains)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        oldest = control_conditioning(
            child_id="depth",
            gain=ContributionGain(DirectGainTableCurve((1.0, 0.5, 0.0)), 1.0),
            model_digest="a" * 64,
        )
        newest = control_conditioning(
            child_id="canny",
            gain=ContributionGain(DirectGainTableCurve((0.25, 0.0, 1.0)), 2.0),
            model_digest="b" * 64,
            hint_value=1.0,
            previous=oldest,
        )
        result = sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("ordered control chain"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=newest,
            compute_dtype=torch.float32,
        )
        assert result.shape == (1, 4, 8, 8)
        assert selected == [(1.0, 0.5), (0.5, 0.0), (0.0, 2.0)]

        changed_oldest = control_conditioning(
            child_id="depth",
            gain=oldest.gain,
            model_digest="c" * 64,
        )
        changed_newest = control_conditioning(
            child_id="canny",
            gain=newest.gain,
            model_digest="b" * 64,
            hint_value=1.0,
            previous=changed_oldest,
        )
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("ordered control chain"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=changed_newest,
            compute_dtype=torch.float32,
        )
        assert len(set(identities)) == 2

    def test_control_applies_resolved_site_and_lane_gain_rows(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[SDControlGain] = []

        def capture_rows(_evaluator: SDDenoiser, gains: tuple[SDControlGain, ...]) -> None:
            selected.extend(gains)

        def capture_run(*_args: object, **kwargs: object) -> torch.Tensor:
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            callback(0)
            callback(1)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain_rows", capture_rows)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        site_gains = tuple(
            (
                site_id,
                2.0
                if index == 0
                else 5.0
                if index == len(SD15_CONTROL_RESIDUAL_SITES) - 1
                else 1.0,
            )
            for index, site_id in enumerate(SD15_CONTROL_RESIDUAL_SITES)
        )
        control = control_conditioning(
            gain=ContributionGain(
                DirectGainTableCurve((0.5, 0.25)),
                2.0,
                site_gains,
                (("negative", 3.0), ("positive", 7.0)),
            )
        )
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("site and lane gains"),
            cfg=SamplingGuidance(sd1.encode_text("negative"), 7.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            control=control,
            compute_dtype=torch.float32,
        )

        assert len(selected) == 2
        assert selected[0].lane_ids == ("positive", "negative")
        assert selected[0].site_lane_gains[0] == (14.0, 6.0)
        assert selected[0].site_lane_gains[1] == (7.0, 3.0)
        assert selected[0].site_lane_gains[-1] == (35.0, 15.0)
        assert selected[1].site_lane_gains[0] == (7.0, 3.0)
        assert selected[1].site_lane_gains[1] == (3.5, 1.5)
        assert selected[1].site_lane_gains[-1] == (17.5, 7.5)

    def test_sdxl_control_lora_uses_exact_sites_and_runtime_identity(
        self,
        sdxl: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[SDControlGain] = []
        identities: list[str] = []

        def capture_rows(_evaluator: SDDenoiser, gains: tuple[SDControlGain, ...]) -> None:
            selected.extend(gains)

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            denoiser = args[0]
            assert isinstance(denoiser, GuidedDenoiser)
            assert denoiser.conditioning_plan is not None
            identities.append(denoiser.conditioning_plan.calls[0].evaluator_identity)
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            callback(0)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain_rows", capture_rows)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        site_gains = tuple(
            (site, 2.0 if index == 0 else 1.0)
            for index, site in enumerate(SDXL_CONTROL_RESIDUAL_SITES)
        )
        effect_mask = effect_mask_source(
            torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
            SDXL_CONTROL_RESIDUAL_SITES[0],
        )
        sdxl.sample(
            tiny_latent(),
            cond=sdxl.encode_text("SDXL Control-LoRA"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            control=control_lora_conditioning(
                gain=ContributionGain(
                    ConstantGainCurve(0.5),
                    1.0,
                    site_gains,
                    effect_mask_digests=(effect_mask.declaration.digest,),
                ),
                effect_masks=(effect_mask,),
            ),
            compute_dtype=torch.float32,
        )
        assert len(selected) == 1
        assert len(selected[0].site_lane_gains) == len(SDXL_CONTROL_RESIDUAL_SITES)
        assert selected[0].site_lane_gains[0] == (1.0,)
        assert selected[0].site_lane_gains[-1] == (0.5,)
        assert selected[0].effect_mask_digests == (effect_mask.declaration.digest,)
        assert "intervention-plan=" in identities[0]

    def test_sdxl_controlnet_union_mode_binds_runtime_identity(
        self,
        sdxl: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        identities: list[str] = []

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            denoiser = args[0]
            assert isinstance(denoiser, GuidedDenoiser)
            assert denoiser.conditioning_plan is not None
            identities.append(denoiser.conditioning_plan.calls[0].evaluator_identity)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        for token in ("hed", "pidi"):
            sdxl.sample(
                tiny_latent(),
                cond=sdxl.encode_text("SDXL ControlNet Union"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                control=controlnet_union_conditioning(token),
                compute_dtype=torch.float32,
            )

        assert identities[0] != identities[1]
        assert all(
            identity.startswith("dinkster.sd.conditioning.v1:intervention-plan=")
            for identity in identities
        )

    def test_classic_sdxl_controlnet_uses_exact_sites_and_masks(
        self,
        sdxl: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[SDControlGain] = []

        def capture_rows(_evaluator: SDDenoiser, gains: tuple[SDControlGain, ...]) -> None:
            selected.extend(gains)

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            callback(0)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain_rows", capture_rows)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        effect_mask = effect_mask_source(
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            SDXL_CONTROL_RESIDUAL_SITES[-1],
        )
        sdxl.sample(
            tiny_latent(),
            cond=sdxl.encode_text("classic SDXL ControlNet"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            control=sdxl_controlnet_conditioning(
                gain=ContributionGain(
                    ConstantGainCurve(0.0),
                    1.0,
                    effect_mask_digests=(effect_mask.declaration.digest,),
                ),
                effect_masks=(effect_mask,),
            ),
            compute_dtype=torch.float32,
        )
        assert len(selected[0].site_lane_gains) == len(SDXL_CONTROL_RESIDUAL_SITES)
        assert all(row == (0.0,) for row in selected[0].site_lane_gains)
        assert selected[0].effect_mask_digests == (effect_mask.declaration.digest,)

    def test_control_effect_mask_binds_compiled_field_identity_and_step_rows(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        selected: list[SDControlGain] = []
        identities: list[str] = []

        def capture_rows(_evaluator: SDDenoiser, gains: tuple[SDControlGain, ...]) -> None:
            selected.extend(gains)

        def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
            denoiser = args[0]
            assert isinstance(denoiser, GuidedDenoiser)
            assert denoiser.conditioning_plan is not None
            identities.append(denoiser.conditioning_plan.calls[0].evaluator_identity)
            callback = kwargs["on_step_begin"]
            assert callable(callback)
            callback(0)
            callback(1)
            latent = kwargs["latent"]
            assert isinstance(latent, torch.Tensor)
            return latent

        monkeypatch.setattr(SDDenoiser, "set_control_gain_rows", capture_rows)
        monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_run)
        source = effect_mask_source(
            torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
            SD15_CONTROL_RESIDUAL_SITES[0],
        )
        control = control_conditioning(
            gain=ContributionGain(
                DirectGainTableCurve((1.0, 0.5)),
                1.0,
                effect_mask_digests=(source.declaration.digest,),
            ),
            effect_masks=(source,),
        )

        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("effect mask"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            control=control,
            compute_dtype=torch.float32,
        )

        assert [row.effect_mask_digests for row in selected] == [
            (source.declaration.digest,),
            (source.declaration.digest,),
        ]
        assert "intervention-plan=" in identities[0]

        changed = effect_mask_source(
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            SD15_CONTROL_RESIDUAL_SITES[0],
        )
        changed_control = control_conditioning(
            gain=ContributionGain(
                DirectGainTableCurve((1.0, 0.5)),
                1.0,
                effect_mask_digests=(changed.declaration.digest,),
            ),
            effect_masks=(changed,),
        )
        sd1.sample(
            tiny_latent(),
            cond=sd1.encode_text("effect mask"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            control=changed_control,
            compute_dtype=torch.float32,
        )
        assert len(set(identities)) == 2

    def test_control_effect_mask_refuses_missing_or_extra_source_before_execution(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse_run(*_args: object, **_kwargs: object) -> torch.Tensor:
            pytest.fail("run_denoise must not execute")

        monkeypatch.setattr(
            sampling_engine,
            "run_denoise",
            refuse_run,
        )
        source = effect_mask_source(
            torch.ones(1, 2, 2),
            SD15_CONTROL_RESIDUAL_SITES[0],
        )
        batch_source = effect_mask_source(
            torch.ones(2, 2, 2),
            SD15_CONTROL_RESIDUAL_SITES[0],
        )
        cases = (
            (
                control_conditioning(
                    gain=ContributionGain(
                        ConstantGainCurve(1.0),
                        1.0,
                        effect_mask_digests=(source.declaration.digest,),
                    )
                ),
                "mask_role_mismatch",
            ),
            (
                control_conditioning(
                    gain=ContributionGain(ConstantGainCurve(1.0), 1.0),
                    effect_masks=(source,),
                ),
                "mask_role_mismatch",
            ),
            (
                control_conditioning(
                    gain=ContributionGain(
                        ConstantGainCurve(1.0),
                        1.0,
                        effect_mask_digests=(batch_source.declaration.digest,),
                    ),
                    effect_masks=(batch_source,),
                ),
                "mask_layout_mismatch",
            ),
        )
        for control, message in cases:
            with pytest.raises(WiringError, match=message):
                sd1.sample(
                    tiny_latent(),
                    cond=sd1.encode_text("invalid mask binding"),
                    sampler_id="dinkster.euler",
                    scheduler_id="dinkster.normal",
                    steps=1,
                    control=control,
                    compute_dtype=torch.float32,
                )

    @pytest.mark.parametrize(
        "gain, message",
        [
            (
                ContributionGain(
                    ConstantGainCurve(1.0),
                    1.0,
                    (("unknown.site.v1", 1.0),),
                ),
                "operator_site_mismatch",
            ),
            (
                ContributionGain(
                    ConstantGainCurve(1.0),
                    1.0,
                    lane_gains=(("unknown", 1.0),),
                ),
                "invalid_gain_schedule",
            ),
        ],
    )
    def test_control_refuses_unknown_site_or_lane_before_execution(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
        gain: ContributionGain,
        message: str,
    ) -> None:
        def fail_run(*_args: object, **_kwargs: object) -> torch.Tensor:
            pytest.fail("run_denoise must not execute")

        monkeypatch.setattr(
            sampling_engine,
            "run_denoise",
            fail_run,
        )
        with pytest.raises(WiringError, match=message):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("invalid site or lane"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                control=control_conditioning(gain=gain),
                compute_dtype=torch.float32,
            )

    def test_control_refuses_structured_gain_with_custom_guidance_strategy(
        self,
        sd1: SDRuntime,
        synthetic_sd15_receipt: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        def refuse_plan(
            context: GuidancePlanContext[torch.Tensor],
        ) -> GuidanceEvaluationPlan[torch.Tensor]:
            pytest.fail(f"strategy plan must not execute: {context!r}")

        def refuse_reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
            pytest.fail(f"strategy reducer must not execute: {context!r}")

        strategy = GuidanceStrategyDescriptor("test.strategy", refuse_plan, refuse_reduce)
        contribution: GuidanceContribution[torch.Tensor] = GuidanceContribution(strategy=strategy)
        runtime = SDRuntime(
            sd1.assembled,
            runtime_identity="native:dinkster.sd15:deployment-specific",
            receipt_identity=synthetic_sd15_receipt,
            guidance_executor=GuidanceExecutor(GuidanceRegistry((("test", contribution),))),
        )

        def fail_run(*_args: object, **_kwargs: object) -> torch.Tensor:
            pytest.fail("run_denoise must not execute")

        monkeypatch.setattr(sampling_execution_module, "run_denoise", fail_run)
        with pytest.raises(WiringError, match="unsupported_control_partition"):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("custom guidance strategy"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                control=control_conditioning(
                    gain=ContributionGain(
                        ConstantGainCurve(1.0),
                        1.0,
                        tuple((site, 1.0) for site in SD15_CONTROL_RESIDUAL_SITES),
                    )
                ),
                compute_dtype=torch.float32,
            )

    def test_explicit_control_gain_requires_identity_application(self, sd1: SDRuntime) -> None:
        control = control_conditioning(
            strength=0.5,
            gain=ContributionGain(ConstantGainCurve(1.0), 1.0),
        )
        with pytest.raises(
            WiringError,
            match=r"set the application to identity or fold the intended scaling/window",
        ):
            sd1.sample(
                tiny_latent(),
                cond=sd1.encode_text("unambiguous control gain"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                control=control,
                compute_dtype=torch.float32,
            )

    @pytest.mark.parametrize("sampler_id", ["dinkster.dpm_fast", "dinkster.dpm_adaptive"])
    def test_off_grid_control_sampler_accepts_only_constant_gain(
        self,
        sd1: SDRuntime,
        sampler_id: str,
    ) -> None:
        cond = sd1.encode_text("off-grid control")
        constant = control_conditioning(gain=ContributionGain(ConstantGainCurve(0.0), 1.0))
        result = sd1.sample(
            tiny_latent(),
            cond=cond,
            sampler_id=sampler_id,
            scheduler_id="dinkster.normal",
            steps=3,
            control=constant,
            compute_dtype=torch.float32,
        )
        assert result.shape == (1, 4, 8, 8)

        scheduled = control_conditioning(
            gain=ContributionGain(DirectGainTableCurve((1.0, 0.5, 0.0)), 1.0)
        )
        with pytest.raises(WiringError, match=f"{sampler_id} supports only constant"):
            sd1.sample(
                tiny_latent(),
                cond=cond,
                sampler_id=sampler_id,
                scheduler_id="dinkster.normal",
                steps=3,
                control=scheduled,
                compute_dtype=torch.float32,
            )

        partial_zero = control_conditioning(
            strength=0.0,
            window=PercentRange(0.25, 0.75),
        )
        with pytest.raises(WiringError, match="full application window"):
            sd1.sample(
                tiny_latent(),
                cond=cond,
                sampler_id=sampler_id,
                scheduler_id="dinkster.normal",
                steps=3,
                control=partial_zero,
                compute_dtype=torch.float32,
            )

    def test_distributed_control_matches_single_replica_execution(
        self,
        sd1: SDRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dinkster_inference_torch.distributed as distributed_module

        cond = sd1.encode_text("distributed control")

        def sample() -> torch.Tensor:
            return sd1.sample(
                tiny_latent(),
                cond=cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                control=control_conditioning(gain=ContributionGain(ConstantGainCurve(0.0), 1.0)),
                compute_dtype=torch.float32,
            )

        baseline = sample()
        config = distributed_module.DistributedSamplingConfig(
            0, 2, "guidance", "file:///group", "1" * 32
        )
        monkeypatch.setattr(distributed_module, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(distributed_module, "ensure_process_group", lambda: config)

        def evaluate(evaluator: Any, x: Any, sigma: Any, request: Any) -> Any:
            return evaluator.evaluate(x, sigma, request)

        monkeypatch.setattr(
            distributed_module.DistributedGuidanceEvaluator, "evaluate_request", evaluate
        )
        assert torch.equal(sample(), baseline)


# --- the codec delegation ---------------------------------------------------


class TestCodec:
    def test_encode_decode_roundtrip_shapes(self, sd1: SDRuntime) -> None:
        generator = torch.Generator("cpu")
        generator.manual_seed(11)
        content = torch.rand(1, 3, 16, 16, generator=generator)
        latent = sd1.encode_content(content)
        assert latent.shape == (1, TINY_KL.embed_dim, 8, 8)
        decoded = sd1.decode_latent(latent)
        assert decoded.shape == (1, 3, 16, 16)

    def test_delegates_to_the_codec_plugin(self, sd1: SDRuntime) -> None:
        generator = torch.Generator("cpu")
        generator.manual_seed(12)
        content = torch.rand(1, 3, 16, 16, generator=generator)
        assert torch.equal(sd1.encode_content(content), sd1.codec.encode(content))
        latent = sd1.encode_content(content)
        assert torch.equal(sd1.decode_latent(latent), sd1.codec.decode(latent))


# --- runtime_identity: the SD-specific facts --------------------------------


def component_plan(
    component: str,
    config: object,
    *,
    keys: Mapping[str, str] | None = None,
    transforms: Mapping[str, Any] | None = None,
    absent: tuple[str, ...] = (),
    identity_facts: tuple[str, ...] = (),
) -> ComponentPlan[Any]:
    mapped = dict(keys) if keys is not None else {"w": "w"}
    return ComponentPlan(
        component=component,
        path=Path(f"/fake/{component}.safetensors"),
        config=config,
        keys=mapped,
        dtypes=dict.fromkeys(mapped, FLOAT32),
        quant={},
        absent=absent,
        transforms=transforms if transforms is not None else {},
        identity_facts=identity_facts,
    )


def sd_identity(
    family_id: str = "dinkster.sdxl",
    components: tuple[ComponentPlan[Any] | None, ...] | None = None,
) -> str:
    if components is None:
        components = (
            component_plan("diffusion", TINY_SDXL_UNET),
            component_plan("clip_l", TINY_CLIP_L),
            component_plan("clip_g", TINY_CLIP_G),
            component_plan("vae", TINY_KL),
        )
    return build_runtime_identity(
        family_id,
        components,
        diffusion_dtype=FLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )


class TestSDIdentity:
    def test_family_id_rides_in_the_clear(self) -> None:
        assert sd_identity("dinkster.sd15").startswith("native:dinkster.sd15:")
        assert sd_identity("dinkster.sdxl") != sd_identity("dinkster.sdxl_refiner")

    def test_vprediction_and_zsnr_rotate_without_moving_eps(self) -> None:
        eps = sd_identity()
        vpred_components = (
            component_plan(
                "diffusion",
                TINY_SDXL_UNET,
                identity_facts=("parameterization=v_prediction", "zsnr=False"),
            ),
            component_plan("clip_l", TINY_CLIP_L),
            component_plan("clip_g", TINY_CLIP_G),
            component_plan("vae", TINY_KL),
        )
        zsnr_components = (
            component_plan(
                "diffusion",
                TINY_SDXL_UNET,
                identity_facts=("parameterization=v_prediction", "zsnr=True"),
            ),
            *vpred_components[1:],
        )
        assert sd_identity() == eps
        assert sd_identity(components=vpred_components) != eps
        assert sd_identity(components=zsnr_components) != sd_identity(components=vpred_components)

    def test_sd15_inpaint_config_has_distinct_structural_identity(self) -> None:
        base = sd_identity(
            "dinkster.sd15",
            (component_plan("diffusion", TINY_SD1_UNET), None, None, None),
        )
        assert base == (
            "native:dinkster.sd15:e17354c35fe61670ed9aea25feaf1fa530558cae480eae524c3cf0bfa3f878f0"
        )
        inpaint = sd_identity(
            "dinkster.sd15",
            (component_plan("diffusion", TINY_SD1_INPAINT_UNET), None, None, None),
        )
        assert base != inpaint

    def test_sdxl_inpaint_config_has_distinct_structural_identity(self) -> None:
        base = sd_identity()
        assert base == (
            "native:dinkster.sdxl:e67597853af604cfedae258f3c89f0eed6cafe1b1ef9166ee77d8f6ca34fa3f3"
        )
        inpaint = sd_identity(
            components=(
                component_plan("diffusion", TINY_SDXL_INPAINT_UNET),
                component_plan("clip_l", TINY_CLIP_L),
                component_plan("clip_g", TINY_CLIP_G),
                component_plan("vae", TINY_KL),
            )
        )
        assert base != inpaint

    def test_planned_transforms_rotate(self) -> None:
        """SD CLIP plans derive tensors (the OpenCLIP in_proj split /
        text_projection transpose); a changed derivation is a changed
        execution body, so it must rotate the identity."""
        plain = component_plan("clip_g", TINY_CLIP_G, keys={"q.weight": "in_proj_weight"})
        chunked = component_plan(
            "clip_g",
            TINY_CLIP_G,
            keys={"q.weight": "in_proj_weight"},
            transforms={"q.weight": RowChunk(part=0, parts=3)},
        )
        other_part = component_plan(
            "clip_g",
            TINY_CLIP_G,
            keys={"q.weight": "in_proj_weight"},
            transforms={"q.weight": RowChunk(part=1, parts=3)},
        )
        identities = {sd_identity(components=(plan,)) for plan in (plain, chunked, other_part)}
        assert len(identities) == 3

    def test_defaulted_absent_keys_rotate(self) -> None:
        """A source missing an optional key executes with the filled
        default - a different body than one carrying the tensor."""
        carried = component_plan("clip_l", TINY_CLIP_L)
        defaulted = component_plan("clip_l", TINY_CLIP_L, absent=("text_projection.weight",))
        assert sd_identity(components=(carried,)) != sd_identity(components=(defaulted,))

    def test_runtime_identity_tracks_environment_facts(self) -> None:
        components = (
            component_plan("diffusion", TINY_SD1_UNET),
            component_plan("clip_l", TINY_CLIP_L),
            None,
            component_plan("vae", TINY_KL),
        )

        def identity(**environment: Any) -> str:
            return build_runtime_identity(
                "dinkster.sd15",
                components,
                diffusion_dtype=FLOAT16,
                text_dtype=FLOAT16,
                vae_dtype=FLOAT16,
                fp8_matmul=False,
                **environment,
            )

        baseline = identity()
        environments: list[dict[str, Any]] = [
            {"registry_token": "custom-registries"},
            {"extension_behavior_hash": "0" * 64},
            {"embedding_binding_digest": "1" * 64},
            {"patch_overlay_digests": ("2" * 64,)},
            {"runtime_facts": ("torch=2.13.0+cu130",)},
        ]
        dressed = {identity(**environment) for environment in environments}
        assert baseline not in dressed
        assert len(dressed) == len(environments)
        assert identity() == baseline

    def test_unwired_none_slots_contribute_nothing(self) -> None:
        """The refiner wires no CLIP-L; its None slot must hash
        exactly like a component tuple without the slot (the family
        id already pins the wired slot set)."""
        diffusion = component_plan("diffusion", TINY_REFINER_UNET)
        clip_g = component_plan("clip_g", TINY_CLIP_G)
        vae = component_plan("vae", TINY_KL)
        with_none = sd_identity("dinkster.sdxl_refiner", (diffusion, None, clip_g, vae))
        without_slot = sd_identity("dinkster.sdxl_refiner", (diffusion, clip_g, vae))
        assert with_none == without_slot


# --- the loader slot guards --------------------------------------------------


@dataclass
class FakeSource:
    """In-memory WeightSource (headers only)."""

    path: Path
    geometries: dict[str, TensorGeometry]

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def fake_source() -> FakeSource:
    return FakeSource(
        Path("/fake/split.safetensors"),
        {"w": TensorGeometry((1,), FLOAT32)},
    )


def test_sd_loader_refuses_a_t5_source() -> None:
    with pytest.raises(WiringError, match="no T5 slot"):
        load_runtime(
            checkpoint=fake_source(),
            t5xxl=fake_source(),
        )


def test_flux_loader_refuses_a_clip_g_source() -> None:
    with pytest.raises(WiringError, match="no CLIP-G slot"):
        load_runtime(
            checkpoint=fake_source(),
            clip_g=fake_source(),
        )
