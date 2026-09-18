"""LTX-2 audio-video 19B runtime contracts over the shared sampling engine."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    LTX_SIGMAS,
    LTXAV,
    LTXAV_19B_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_22B_V25_VAE_CONFIG,
    LTXAV_BWE_VOCODER_CONFIG,
    CancellationToken,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    FluxFlowSigmas,
    GuidanceCondition,
    GuidanceEvaluationPlan,
    GuidanceEvaluationRequest,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    LTXGeneratedKeyframes,
    MultiStreamConditioningRuntime,
    MultiStreamLatent,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    ProgressScope,
    SamplingExecutionContext,
    SamplingGuidance,
    SamplingSegment,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
    sampling_sigmas,
)
from dinkster_inference_torch import ltxav_runtime
from dinkster_inference_torch._conditioning_layout import declare_text_conditioning
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.gemma_text import LtxDualTextProjection
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from dinkster_inference_torch.latent_streams import pack_latent_streams, unpack_latent_streams
from dinkster_inference_torch.ltx_connector import LtxTextConnectors
from dinkster_inference_torch.ltxav_runtime import (
    LTXAVAudioCodecRuntime,
    LTXAVDiffusionRuntime,
    LTXAVExecutionOptions,
    LTXAVPreparedConditioning,
    LTXAVRuntimeError,
    LTXAVTextRuntime,
    LTXAVVideoCodecRuntime,
    ltxav_dual_cfg_guidance,
    ltxav_identity_guidance,
    ltxav_modality_guidance,
    ltxav_spatiotemporal_guidance,
)
from dinkster_inference_torch.payloads import tensor_to_payload_binding
from dinkster_inference_torch.sampling_execution import build_sampling_schedule
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry

FUSE_CFG_LANES = ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 2)

IDENTITY_GUIDANCE_GOLDENS = cast(
    "dict[str, Any]",
    json.loads(
        (Path(__file__).parent / "goldens/ltxav_identity_guidance_goldens.json").read_text()
    ),
)


def test_ltxav_diffusion_runtime_declares_audio_cfg_without_text_storage() -> None:
    assert LTXAVDiffusionRuntime.retained_offload_storage_components == frozenset()
    assert LTXAVDiffusionRuntime.supports_audio_cfg is True


class _Diffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=4,
            cross_attention_dim=16,
            attention_head_dim=8,
            num_attention_heads=2,
            audio_in_channels=128,
            audio_cross_attention_dim=16,
            audio_attention_head_dim=8,
            audio_num_attention_heads=2,
            caption_channels=4,
            num_layers=1,
            causal_temporal_positioning=False,
            av_ca_timestep_scale_multiplier=1000.0,
            caption_proj_before_connector=False,
        )
        self.patchify_proj = torch.nn.Linear(4, 4, bias=False)
        self.calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                tuple[torch.Tensor, torch.Tensor | None],
                float,
            ]
        ] = []
        self.denoise_masks: list[torch.Tensor | None] = []
        self.reference_audio: list[torch.Tensor | None] = []
        self.execution_options: list[tuple[frozenset[int], bool, bool]] = []
        self.generated_keyframes: list[LTXGeneratedKeyframes | None] = []

    def preprocess_text_embeds(self, context: torch.Tensor) -> torch.Tensor:
        return context

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        frame_rate: float,
        denoise_mask: torch.Tensor | None = None,
        ref_audio_tokens: torch.Tensor | None = None,
        stg_self_attn_blocks: frozenset[int] = frozenset(),
        a2v_cross_attention: bool = True,
        v2a_cross_attention: bool = True,
        generated_keyframes: LTXGeneratedKeyframes | None = None,
        context_preprocessed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert context_preprocessed is True
        self.denoise_masks.append(None if denoise_mask is None else denoise_mask.detach().clone())
        self.reference_audio.append(
            None if ref_audio_tokens is None else ref_audio_tokens.detach().clone()
        )
        self.execution_options.append(
            (stg_self_attn_blocks, a2v_cross_attention, v2a_cross_attention)
        )
        self.generated_keyframes.append(generated_keyframes)
        self.calls.append(
            (
                video.detach().clone(),
                audio.detach().clone(),
                timesteps.detach().clone(),
                audio_timesteps.detach().clone(),
                (context.detach().clone(), attention_mask),
                frame_rate,
            )
        )
        velocity = context.mean()
        return torch.ones_like(video) * velocity, torch.ones_like(audio) * velocity


class _TextEncoder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        self.texts.append(text)
        return Conditioning(torch.ones((1, 4, 8)), None)


def _runtime(
    diffusion: _Diffusion | None = None, *, dtype: torch.dtype = torch.float32
) -> tuple[LTXAVDiffusionRuntime, _Diffusion]:
    diffusion = _Diffusion() if diffusion is None else diffusion

    def compute_dtype(_component: object) -> torch.dtype:
        return dtype

    assembled = SimpleNamespace(
        family=LTXAV,
        diffusion=diffusion,
        vae=None,
        compute_dtype=compute_dtype,
    )
    runtime = object.__new__(LTXAVDiffusionRuntime)
    raw = cast("Any", runtime)
    raw._assembled = assembled
    raw._runtime_identity = "test:ltxav"
    raw._samplers = torch_sampler_registry()
    raw._schedulers = torch_scheduler_registry()
    return runtime, diffusion


def _text_runtime() -> LTXAVTextRuntime:
    runtime = object.__new__(LTXAVTextRuntime)
    raw = cast("Any", runtime)
    raw._encoder = _TextEncoder()
    raw._text_dim = 8
    raw._text_stream = "gemma3_12b"
    return runtime


def test_text_runtime_composes_19b_and_22b_components_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma = cast("Any", SimpleNamespace(config=SimpleNamespace(architecture="gemma3_ltx_12b")))
    tokenizer = object()
    calls: list[tuple[object, object, object, object]] = []

    class Encoder:
        def __init__(
            self,
            gemma_arg: object,
            projection_arg: object,
            tokenizer_arg: object,
            *,
            connectors: object,
        ) -> None:
            calls.append((gemma_arg, projection_arg, tokenizer_arg, connectors))

    def make_tokenizer(_value: bytes) -> object:
        return tokenizer

    monkeypatch.setattr(ltxav_runtime, "GemmaSentencePieceTokenizer", make_tokenizer)
    monkeypatch.setattr(ltxav_runtime, "LtxGemmaTextEncoder", Encoder)
    projection_19b = object.__new__(torch.nn.Linear)
    projection_19b.out_features = LTXAV_19B_CONFIG.caption_channels
    connectors = object.__new__(LtxTextConnectors)

    runtime_19b = LTXAVTextRuntime(
        gemma,
        projection_19b,
        b"tokenizer",
        connectors=connectors,
    )

    assert cast("Any", runtime_19b)._text_dim == 2 * LTXAV_19B_CONFIG.caption_channels
    assert calls[-1] == (gemma, projection_19b, tokenizer, connectors)

    projection_22b = object.__new__(LtxDualTextProjection)
    runtime_22b = LTXAVTextRuntime(
        gemma,
        projection_22b,
        b"tokenizer",
        connectors=None,
    )

    assert cast("Any", runtime_22b)._text_dim == (
        LTXAV_22B_V23_CONFIG.cross_attention_dim + LTXAV_22B_V23_CONFIG.audio_cross_attention_dim
    )
    assert calls[-1] == (gemma, projection_22b, tokenizer, None)

    with pytest.raises(ValueError, match="19B text requires"):
        LTXAVTextRuntime(gemma, projection_19b, b"tokenizer", connectors=None)
    with pytest.raises(ValueError, match="must not carry 19B connectors"):
        LTXAVTextRuntime(gemma, projection_22b, b"tokenizer", connectors=connectors)
    with pytest.raises(TypeError, match="exact supported projection"):
        LTXAVTextRuntime(gemma, cast("Any", torch.nn.Identity()), b"tokenizer", connectors=None)
    with pytest.raises(ValueError, match="19B text requires"):
        LTXAVTextRuntime(
            gemma,
            projection_19b,
            b"tokenizer",
            connectors=cast("Any", torch.nn.Identity()),
        )


def _video(shape: tuple[int, ...] = (1, 4, 1, 2, 2)) -> torch.Tensor:
    return torch.zeros(shape)


def _audio(shape: tuple[int, ...] = (1, 8, 1, 16)) -> torch.Tensor:
    return torch.zeros(shape)


def _streams(
    video: torch.Tensor | None = None, audio: torch.Tensor | None = None
) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent.from_pairs(
        (
            ("video", _video() if video is None else video),
            ("audio", _audio() if audio is None else audio),
        )
    )


def _prepared(frame_rate: float = 25.0, rows: int = 4) -> LTXAVPreparedConditioning:
    return LTXAVPreparedConditioning(torch.ones((1, rows, 8)), frame_rate)


def _sample(
    runtime: LTXAVDiffusionRuntime,
    latent: object,
    conditioning: object,
    **kwargs: object,
) -> MultiStreamLatent[torch.Tensor]:
    streams = latent if type(latent) is MultiStreamLatent else _streams(cast("Any", latent))
    return runtime.sample_multistream(
        cast("Any", streams),
        conditioning=conditioning,
        sampler_id=cast("str", kwargs.pop("sampler_id", "euler")),
        scheduler_id=cast("str", kwargs.pop("scheduler_id", "simple")),
        steps=cast("int", kwargs.pop("steps", 1)),
        denoise=cast("float", kwargs.pop("denoise", 1.0)),
        seed=cast("int", kwargs.pop("seed", 123)),
        **cast("Any", kwargs),
    )


def test_encode_text_uses_the_ltxav_owned_gemma_encoder() -> None:
    runtime = _text_runtime()

    result = runtime.encode_text("a fox running")

    raw = cast("Any", runtime)
    assert raw._encoder.texts == ["a fox running"]
    assert result.pooled is None
    assert result.embeddings.shape == (1, 4, 8)


def test_carrier_round_trip_preserves_text_and_frame_rate() -> None:
    runtime, _ = _runtime()
    text_runtime = _text_runtime()
    conditioning = Conditioning(torch.ones((1, 4, 8)), None)

    carrier = text_runtime.text_conditioning_carrier(conditioning)
    prepared = runtime.prepare_conditioning(carrier, frame_rate=30.0)

    assert prepared.frame_rate == 30.0
    assert prepared.text.shape == (1, 4, 8)
    assert torch.equal(prepared.text, conditioning.embeddings)


def test_prepare_and_sampling_preserve_generated_keyframe_slots() -> None:
    runtime, diffusion = _runtime()
    text_runtime = _text_runtime()
    carrier = text_runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 4, 8)), None))
    generated = LTXGeneratedKeyframes(4, 2, 1)

    prepared = runtime.prepare_conditioning(carrier, generated_keyframes=generated)
    _sample(runtime, _streams(), prepared)

    assert prepared.generated_keyframes is generated
    assert diffusion.generated_keyframes == [generated]


def test_declared_token_counts_must_cover_every_payload_row() -> None:
    runtime, _ = _runtime()
    text_runtime = _text_runtime()
    conditioning = declare_text_conditioning(Conditioning(torch.ones((1, 4, 8)), None), 3)

    with pytest.raises(LTXAVRuntimeError, match="cover exactly the TEXT payload rows"):
        runtime.prepare_conditioning(text_runtime.text_conditioning_carrier(conditioning))


def test_undeclared_conditioning_attends_to_every_payload_row() -> None:
    runtime, _ = _runtime()
    text_runtime = _text_runtime()

    prepared = runtime.prepare_conditioning(
        text_runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 4, 8)), None))
    )

    assert prepared.text.shape == (1, 4, 8)
    assert prepared.frame_rate == 25.0


def test_pooled_embeddings_are_refused() -> None:
    runtime = _text_runtime()

    with pytest.raises(LTXAVRuntimeError, match="pooled"):
        runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 4, 8)), torch.ones((1, 4))))


def test_prepared_conditioning_refuses_non_tensor_and_integer_text() -> None:
    with pytest.raises(TypeError, match="strided floating"):
        LTXAVPreparedConditioning(cast("Any", "text"))
    with pytest.raises(TypeError, match="strided floating"):
        LTXAVPreparedConditioning(torch.ones((1, 4, 8), dtype=torch.int64))


def test_prepared_conditioning_refuses_untyped_generated_keyframes() -> None:
    with pytest.raises(TypeError, match="exact LTXGeneratedKeyframes"):
        LTXAVPreparedConditioning(
            torch.ones((1, 4, 8)),
            generated_keyframes=cast("Any", {"tokens_per_frame": 4}),
        )


@pytest.mark.parametrize("shape", ((4, 8), (0, 4, 8), (1, 0, 8), (1, 4, 0)))
def test_prepared_conditioning_refuses_degenerate_text_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(LTXAVRuntimeError, match="nonempty rank-3"):
        LTXAVPreparedConditioning(torch.ones(shape))


@pytest.mark.parametrize("frame_rate", (25, True, 0.0, -24.0, math.inf, math.nan))
def test_prepared_conditioning_refuses_non_float_frame_rates(frame_rate: object) -> None:
    with pytest.raises(LTXAVRuntimeError, match="frame rate"):
        LTXAVPreparedConditioning(torch.ones((1, 4, 8)), cast("Any", frame_rate))


def test_sampling_forwards_shared_timesteps_and_the_frame_rate() -> None:
    runtime, diffusion = _runtime()

    _sample(runtime, _streams(), _prepared(30.0))

    assert len(diffusion.calls) == 1
    video, audio, timesteps, audio_timesteps, (context, mask), frame_rate = diffusion.calls[0]
    assert video.shape == (1, 4, 1, 2, 2)
    assert audio.shape == (1, 8, 1, 16)
    assert timesteps.dtype == torch.float32
    assert timesteps.shape == (1,)
    assert torch.equal(timesteps, audio_timesteps)
    assert context.dtype == torch.float32
    assert context.shape == (1, 4, 8)
    assert mask is None
    assert frame_rate == 30.0


def test_cfg_lanes_batch_into_one_model_call() -> None:
    runtime, diffusion = _runtime()
    cfg = SamplingGuidance(cast("Any", _prepared()), 2.0, batching=FUSE_CFG_LANES)

    _sample(runtime, _streams(), _prepared(), cfg=cfg)

    assert len(diffusion.calls) == 1
    video, audio, timesteps, _, (context, _), _ = diffusion.calls[0]
    assert video.shape == (2, 4, 1, 2, 2)
    assert audio.shape == (2, 8, 1, 16)
    assert timesteps.shape == (2,)
    assert context.shape == (2, 4, 8)


def test_cfg_batches_raw_text_lengths_after_diffusion_preprocessing() -> None:
    class FixedContextDiffusion(_Diffusion):
        def __init__(self) -> None:
            super().__init__()
            self.preprocessed_rows: list[int] = []

        def preprocess_text_embeds(self, context: torch.Tensor) -> torch.Tensor:
            self.preprocessed_rows.append(context.shape[1])
            padding = torch.zeros(
                context.shape[0], 6 - context.shape[1], context.shape[2], dtype=context.dtype
            )
            return torch.cat((context, padding), dim=1)

    diffusion = FixedContextDiffusion()
    runtime, _ = _runtime(diffusion)
    positive = LTXAVPreparedConditioning(torch.full((1, 4, 8), 2.0))
    negative = LTXAVPreparedConditioning(torch.full((1, 2, 8), 3.0))

    _sample(
        runtime,
        _streams(),
        positive,
        cfg=SamplingGuidance(cast("Any", negative), 2.0, batching=FUSE_CFG_LANES),
    )

    assert diffusion.preprocessed_rows == [4, 2]
    assert len(diffusion.calls) == 1
    context = diffusion.calls[0][4][0]
    assert context.shape == (2, 6, 8)
    assert torch.equal(context[0, :2], torch.full((2, 8), 3.0))
    assert torch.equal(context[1, :4], torch.full((4, 8), 2.0))


def test_guidance_lanes_must_share_one_frame_rate() -> None:
    runtime, _ = _runtime()
    cfg = SamplingGuidance(cast("Any", _prepared(24.0)), 2.0)

    with pytest.raises(LTXAVRuntimeError, match="share one frame rate"):
        _sample(runtime, _streams(), _prepared(25.0), cfg=cfg)


def test_guidance_lanes_must_share_generated_keyframes() -> None:
    runtime, _ = _runtime()
    positive = replace(_prepared(), generated_keyframes=LTXGeneratedKeyframes(4, 1, 1))
    negative = replace(_prepared(), generated_keyframes=LTXGeneratedKeyframes(4, 2, 1))

    with pytest.raises(LTXAVRuntimeError, match="share generated keyframes"):
        _sample(runtime, _streams(), positive, cfg=SamplingGuidance(cast("Any", negative), 2.0))


def test_sampling_rejects_raw_conditioning_lanes() -> None:
    runtime, _ = _runtime()
    prepared = _prepared()

    with pytest.raises(TypeError, match="exact LTXAVPreparedConditioning"):
        runtime.sample_multistream(
            _streams(),
            conditioning=cast("Any", Conditioning(torch.ones((1, 4, 8)), None)),
            sampler_id="euler",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
        )
    with pytest.raises(TypeError, match="guidance lanes require exact LTXAVPreparedConditioning"):
        _sample(
            runtime,
            _streams(),
            prepared,
            cfg=SamplingGuidance(cast("Any", Conditioning(torch.ones((1, 4, 8)), None)), 2.0),
        )


@pytest.mark.parametrize("shift", (-1.0, 0.0, 3, True, math.inf, math.nan))
def test_sampling_shift_must_be_a_positive_finite_float(shift: object) -> None:
    runtime, _ = _runtime()

    with pytest.raises(LTXAVRuntimeError, match="sampling_shift"):
        _sample(runtime, _streams(), _prepared(), sampling_shift=shift)


def test_explicit_sampling_shift_runs_the_engine() -> None:
    runtime, diffusion = _runtime()

    result = _sample(runtime, _streams(), _prepared(), sampling_shift=3.0)

    assert result.by_role("video").shape == (1, 4, 1, 2, 2)
    assert result.by_role("audio").shape == (1, 8, 1, 16)
    assert len(diffusion.calls) == 1


def test_sampler_denoise_mask_is_forwarded_and_noise_indices_are_refused() -> None:
    runtime, diffusion = _runtime(dtype=torch.bfloat16)

    video_mask = torch.ones((1, 1, 1, 2, 2))
    video_mask[..., 0, 0] = 0.0
    mask = MultiStreamLatent.from_pairs(
        (
            ("video", video_mask),
            ("audio", torch.ones((1, 1, 1, 1))),
        )
    )
    _sample(runtime, _streams(), _prepared(), denoise_mask=mask)
    observed_mask = diffusion.denoise_masks[0]
    assert observed_mask is not None
    assert torch.equal(observed_mask, video_mask)
    assert observed_mask.dtype == torch.bfloat16
    assert diffusion.calls[0][2].dtype == torch.float32
    assert diffusion.calls[0][3].dtype == torch.float32
    assert torch.count_nonzero(diffusion.calls[0][0][..., 0, 0]) == 0
    with pytest.raises(LTXAVRuntimeError, match="noise indices"):
        _sample(runtime, _streams(), _prepared(), noise_inds=(0,))


def test_reference_audio_and_mask_timesteps_reach_the_batched_model() -> None:
    runtime, diffusion = _runtime()
    reference = torch.arange(3 * 128, dtype=torch.float32).reshape(1, 3, 128)
    prepared = LTXAVPreparedConditioning(torch.ones((1, 4, 8)), 25.0, reference)
    streams = _streams(
        torch.zeros((2, 4, 2, 2, 2)),
        torch.zeros((2, 8, 2, 16)),
    )
    mask = MultiStreamLatent.from_pairs(
        (
            (
                "video",
                torch.tensor([1.0, 0.0]).reshape(1, 1, 2, 1, 1).expand(2, 1, 2, 2, 2),
            ),
            ("audio", torch.tensor([0.5, 0.0]).reshape(1, 1, 2, 1).expand(2, 1, 2, 1)),
        )
    )

    _sample(
        runtime,
        streams,
        prepared,
        cfg=SamplingGuidance(cast("Any", prepared), 2.0, batching=FUSE_CFG_LANES),
        denoise_mask=mask,
    )

    assert len(diffusion.calls) == 1
    video_timestep = diffusion.calls[0][2]
    audio_timestep = diffusion.calls[0][3]
    assert video_timestep.shape == (4, 8)
    assert bool((video_timestep[:, :4] > 0.0).all())
    assert torch.count_nonzero(video_timestep[:, 4:]) == 0
    assert audio_timestep.shape == (4, 2)
    assert bool((audio_timestep[:, :1] > 0.0).all())
    assert torch.count_nonzero(audio_timestep[:, 1:]) == 0
    observed_reference = diffusion.reference_audio[0]
    assert observed_reference is not None
    assert observed_reference.shape == (4, 3, 128)
    assert all(torch.equal(row, reference[0]) for row in observed_reference)


def test_identity_guidance_matches_the_no_reference_formula_and_sigma_window() -> None:
    reference = torch.ones((1, 3, 128))
    positive = LTXAVPreparedConditioning(torch.ones((1, 4, 8)), 25.0, reference)
    negative = LTXAVPreparedConditioning(torch.zeros((1, 4, 8)), 25.0, reference)
    contribution = ltxav_identity_guidance(3.0, 1.0, 0.2)
    augment = contribution.plan_augmentations[0].augment
    apply = contribution.post_cfg[0].transform
    plan = GuidanceEvaluationPlan(
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", positive)),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast("Any", negative)),
        ),
        "positive",
        "negative",
    )

    active = augment(
        cast("Any", SimpleNamespace(execution=SimpleNamespace(current_sigma=0.5))), plan
    )
    no_reference = cast("LTXAVPreparedConditioning", active.lanes[-1].conditioning)
    assert active.lanes[-1].role is GuidanceRole.AUXILIARY
    assert no_reference.text is positive.text
    assert no_reference.reference_audio is None
    predictions = GuidancePredictions(
        (
            GuidancePrediction("positive", torch.tensor([11.0]), GuidancePredictionSource.MODEL),
            GuidancePrediction("negative", torch.tensor([2.0]), GuidancePredictionSource.MODEL),
            GuidancePrediction(
                active.lanes[-1].id,
                torch.tensor([5.0]),
                GuidancePredictionSource.MODEL,
            ),
        )
    )
    post_context = SimpleNamespace(
        request=SimpleNamespace(plan=active),
        predictions=predictions,
        reduced=torch.tensor([20.0]),
    )

    assert torch.equal(apply(cast("Any", post_context)), torch.tensor([38.0]))
    for sigma in (1.01, 0.19):
        inactive = augment(
            cast("Any", SimpleNamespace(execution=SimpleNamespace(current_sigma=sigma))), plan
        )
        assert inactive is plan


def test_identity_guidance_refuses_invalid_configuration_and_missing_reference() -> None:
    for values, match in (
        ((-1.0, 1.0, 0.0), "scale must be nonnegative"),
        ((float("nan"), 1.0, 0.0), "scale must be a finite float"),
        ((1.0, float("inf"), 0.0), "sigma_start must be a finite float"),
        ((1.0, 1.0, float("-inf")), "sigma_end must be a finite float"),
    ):
        with pytest.raises(ValueError, match=match):
            ltxav_identity_guidance(*values)

    positive = LTXAVPreparedConditioning(torch.ones((1, 4, 8)))
    contribution = ltxav_identity_guidance(3.0, 1.0, 0.0)
    plan = GuidanceEvaluationPlan(
        (GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", positive)),),
        "positive",
        None,
    )
    context = SimpleNamespace(execution=SimpleNamespace(current_sigma=0.5))
    with pytest.raises(LTXAVRuntimeError, match="requires reference audio"):
        contribution.plan_augmentations[0].augment(cast("Any", context), plan)


@pytest.mark.parametrize("name", sorted(IDENTITY_GUIDANCE_GOLDENS["cases"]))
def test_identity_guidance_matches_executed_comfy_reference(name: str) -> None:
    case = IDENTITY_GUIDANCE_GOLDENS["cases"][name]
    conditional = torch.tensor(case["conditional"], dtype=torch.float32)
    cfg_result = torch.tensor(case["cfg_result"], dtype=torch.float32)
    no_reference = torch.tensor(case["no_reference"], dtype=torch.float32)
    unconditional = conditional * 2.0 - cfg_result
    reference = torch.ones((1, 3, 128))
    positive = LTXAVPreparedConditioning(torch.ones((1, 4, 8)), 25.0, reference)
    negative = LTXAVPreparedConditioning(torch.zeros((1, 4, 8)), 25.0, reference)
    contribution = ltxav_identity_guidance(
        float(case["scale"]),
        float(case["sigma_start"]),
        float(case["sigma_end"]),
    )
    token = CancellationToken(lambda: False)
    execution = SamplingExecutionContext(
        (float(case["sigma"]), 0.0),
        0,
        0,
        float(case["sigma"]),
        1,
        token,
        ProgressScope(token),
        {},
    )
    context = __import__("dinkster_inference").GuidancePlanContext(
        torch.zeros_like(conditional),
        torch.tensor(case["sigma"], dtype=torch.float32),
        2.0,
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", positive)),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast("Any", negative)),
        ),
        False,
        execution,
    )
    evaluated: list[str] = []

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        values = {
            "positive": conditional,
            "negative": unconditional,
            "dinkster.ltxav.identity-no-reference": no_reference,
        }
        evaluated.extend(lane.id for lane in request.plan.lanes)
        return GuidancePredictions(
            tuple(
                GuidancePrediction(lane.id, values[lane.id], GuidancePredictionSource.MODEL)
                for lane in request.plan.lanes
            )
        )

    result = GuidanceExecutor(GuidanceRegistry((("test.ltxav-identity", contribution),))).execute(
        context, evaluate
    )

    assert torch.equal(result.denoised, torch.tensor(case["output"], dtype=torch.float32))
    expected_lanes = ["positive", "negative"]
    if case["calls"]:
        expected_lanes.append("dinkster.ltxav.identity-no-reference")
    assert evaluated == expected_lanes


def test_identity_guidance_runs_the_no_reference_lane_through_custom_sampling() -> None:
    runtime, diffusion = _runtime()
    reference = torch.ones((1, 3, 128))
    positive = LTXAVPreparedConditioning(torch.ones((1, 4, 8)), 25.0, reference)
    negative = LTXAVPreparedConditioning(torch.zeros((1, 4, 8)), 25.0, reference)
    contribution = ltxav_identity_guidance(3.0, 1.0, 0.0)

    _sample(
        runtime,
        _streams(),
        positive,
        cfg=SamplingGuidance(
            cast("Any", negative),
            2.0,
            transforms=(("test.ltxav-id", contribution),),
        ),
    )

    assert len(diffusion.reference_audio) == 2
    assert diffusion.reference_audio[0] is not None
    assert diffusion.reference_audio[1] is None


def test_execution_options_are_strictly_validated() -> None:
    for values, match in (
        ((cast("Any", {1}), True, True), "frozenset"),
        ((cast("Any", frozenset({True})), True, True), "nonnegative integers"),
        ((frozenset({-1}), True, True), "nonnegative integers"),
        ((frozenset(), cast("Any", 1), True), "exact bools"),
        ((frozenset(), True, cast("Any", 0)), "exact bools"),
    ):
        with pytest.raises(TypeError, match=match):
            LTXAVExecutionOptions(*values)

    with pytest.raises(TypeError, match="exact LTXAVExecutionOptions"):
        replace(_prepared(), execution=cast("Any", SimpleNamespace()))


def test_stg_guidance_augments_and_applies_the_reference_formula() -> None:
    positive = _prepared()
    contribution = ltxav_spatiotemporal_guidance(2.5, frozenset({1, 7}), 1.0, 0.2)
    plan = GuidanceEvaluationPlan(
        (GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", positive)),),
        "positive",
        None,
    )
    active = contribution.plan_augmentations[0].augment(
        cast("Any", SimpleNamespace(execution=SimpleNamespace(current_sigma=0.5))), plan
    )
    perturbed = cast("LTXAVPreparedConditioning", active.lanes[-1].conditioning)

    assert perturbed.execution == LTXAVExecutionOptions(frozenset({1, 7}))
    predictions = GuidancePredictions(
        (
            GuidancePrediction("positive", torch.tensor([11.0]), GuidancePredictionSource.MODEL),
            GuidancePrediction(
                active.lanes[-1].id,
                torch.tensor([5.0]),
                GuidancePredictionSource.MODEL,
            ),
        )
    )
    context = SimpleNamespace(
        request=SimpleNamespace(plan=active),
        predictions=predictions,
        reduced=torch.tensor([20.0]),
    )
    assert torch.equal(
        contribution.post_cfg[0].transform(cast("Any", context)), torch.tensor([35.0])
    )

    for sigma in (1.01, 0.19):
        inactive = contribution.plan_augmentations[0].augment(
            cast("Any", SimpleNamespace(execution=SimpleNamespace(current_sigma=sigma))), plan
        )
        assert inactive is plan


def test_modality_guidance_augments_and_applies_the_reference_formula() -> None:
    positive = _prepared()
    contribution = ltxav_modality_guidance(3.0, 1.0, 0.2)
    plan = GuidanceEvaluationPlan(
        (GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", positive)),),
        "positive",
        None,
    )
    active = contribution.plan_augmentations[0].augment(
        cast("Any", SimpleNamespace(execution=SimpleNamespace(current_sigma=0.5))), plan
    )
    decoupled = cast("LTXAVPreparedConditioning", active.lanes[-1].conditioning)

    assert decoupled.execution == LTXAVExecutionOptions(
        a2v_cross_attention=False,
        v2a_cross_attention=False,
    )
    predictions = GuidancePredictions(
        (
            GuidancePrediction("positive", torch.tensor([11.0]), GuidancePredictionSource.MODEL),
            GuidancePrediction(
                active.lanes[-1].id,
                torch.tensor([5.0]),
                GuidancePredictionSource.MODEL,
            ),
        )
    )
    context = SimpleNamespace(
        request=SimpleNamespace(plan=active),
        predictions=predictions,
        reduced=torch.tensor([20.0]),
    )
    assert torch.equal(
        contribution.post_cfg[0].transform(cast("Any", context)), torch.tensor([32.0])
    )


def test_dual_cfg_guidance_applies_independent_packed_stream_scales() -> None:
    contribution = ltxav_dual_cfg_guidance(3.0, 7.0, 4)
    conditions = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", _prepared())),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast("Any", _prepared())),
    )
    plan = cast("Any", contribution.strategy).plan(
        cast("Any", SimpleNamespace(conditions=conditions))
    )
    assert plan == GuidanceEvaluationPlan(
        (
            conditions[0],
            conditions[1],
        ),
        "positive",
        "negative",
    )
    conditional = torch.tensor([[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]])
    unconditional = torch.ones_like(conditional)
    predictions = GuidancePredictions(
        (
            GuidancePrediction("positive", conditional, GuidancePredictionSource.MODEL),
            GuidancePrediction("negative", unconditional, GuidancePredictionSource.MODEL),
        )
    )
    input = torch.full_like(conditional, 9.0)
    context = SimpleNamespace(
        request=SimpleNamespace(plan=plan, input=input),
        predictions=predictions,
        reduced=torch.full_like(conditional, -1.0),
    )

    actual = cast("Any", contribution.strategy).reduce(cast("Any", context))

    conditional_noise = input - conditional
    unconditional_noise = input - unconditional
    guided_noise = unconditional_noise + (conditional_noise - unconditional_noise) * 3.0
    guided_noise[..., 4:] = (
        unconditional_noise[..., 4:]
        + (conditional_noise[..., 4:] - unconditional_noise[..., 4:]) * 7.0
    )
    expected = input - guided_noise
    assert cast("Any", contribution.strategy).requires_uncond is True
    assert torch.equal(actual, expected)


def test_dual_cfg_composes_with_stg_and_repeated_stg_lanes() -> None:
    dual = ltxav_dual_cfg_guidance(3.0, 7.0, 2)
    stg0 = ltxav_spatiotemporal_guidance(
        2.0,
        frozenset({0}),
        1.0,
        0.0,
        lane_id="test.stg:0",
    )
    stg1 = ltxav_spatiotemporal_guidance(
        1.0,
        frozenset({1}),
        1.0,
        0.0,
        lane_id="test.stg:1",
    )
    token = CancellationToken(lambda: False)
    execution = SamplingExecutionContext(
        (0.5, 0.0),
        0,
        0,
        0.5,
        1,
        token,
        ProgressScope(token),
        {},
    )
    context = __import__("dinkster_inference").GuidancePlanContext(
        torch.zeros((1, 1, 4)),
        torch.tensor(0.5),
        7.0,
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", _prepared())),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast("Any", _prepared())),
        ),
        False,
        execution,
    )
    values = {
        "positive": torch.tensor([[[2.0, 3.0, 4.0, 5.0]]]),
        "negative": torch.ones((1, 1, 4)),
        "test.stg:0": torch.zeros((1, 1, 4)),
        "test.stg:1": torch.full((1, 1, 4), -1.0),
    }

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        return GuidancePredictions(
            tuple(
                GuidancePrediction(lane.id, values[lane.id], GuidancePredictionSource.MODEL)
                for lane in request.plan.lanes
            )
        )

    result = GuidanceExecutor(
        GuidanceRegistry(
            (
                ("test.dual", dual),
                ("test.stg-owner:0", stg0),
                ("test.stg-owner:1", stg1),
            )
        )
    ).execute(context, evaluate)

    dual_result = torch.tensor([[[4.0, 7.0, 22.0, 29.0]]])
    assert torch.equal(
        result.denoised,
        dual_result
        + (values["positive"] - values["test.stg:0"]) * 2.0
        + values["positive"]
        - values["test.stg:1"],
    )


@pytest.mark.parametrize(
    ("factory", "values", "match"),
    (
        (ltxav_spatiotemporal_guidance, (-1.0, frozenset({1}), 1.0, 0.0), "nonnegative"),
        (
            ltxav_spatiotemporal_guidance,
            (1.0, cast("Any", frozenset({True})), 1.0, 0.0),
            "nonnegative integers",
        ),
        (ltxav_modality_guidance, (0.9, 1.0, 0.0), "at least 1"),
        (ltxav_modality_guidance, (1.0, math.inf, 0.0), "finite float"),
        (ltxav_dual_cfg_guidance, (math.nan, 1.0, 4), "finite nonnegative"),
        (ltxav_dual_cfg_guidance, (1.0, -1.0, 4), "finite nonnegative"),
        (ltxav_dual_cfg_guidance, (1.0, 1.0, True), "positive exact int"),
    ),
)
def test_ltxav_guidance_refuses_invalid_configuration(
    factory: object,
    values: tuple[object, ...],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        cast("Any", factory)(*values)


@pytest.mark.parametrize(
    ("contribution", "expected"),
    (
        (
            ltxav_spatiotemporal_guidance(2.0, frozenset({0}), 1.0, 0.0),
            (frozenset({0}), True, True),
        ),
        (ltxav_modality_guidance(2.0, 1.0, 0.0), (frozenset(), False, False)),
    ),
)
def test_ltxav_guidance_options_reach_separate_custom_sampling_lanes(
    contribution: object,
    expected: tuple[frozenset[int], bool, bool],
) -> None:
    runtime, diffusion = _runtime()
    positive = _prepared()

    _sample(
        runtime,
        _streams(),
        positive,
        cfg=SamplingGuidance(
            None,
            1.0,
            transforms=(("test.ltxav-guidance", cast("Any", contribution)),),
        ),
    )

    assert diffusion.execution_options == [(frozenset(), True, True), expected]


def test_latent_streams_are_validated_before_sampling() -> None:
    runtime, _ = _runtime()
    prepared = _prepared()

    with pytest.raises(TypeError, match="exact MultiStreamLatent"):
        runtime.sample_multistream(
            cast("Any", _video()),
            conditioning=prepared,
            sampler_id="euler",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
        )
    with pytest.raises(LTXAVRuntimeError, match=r"\('video', 'audio'\)"):
        _sample(runtime, MultiStreamLatent.from_pairs((("video", _video()),)), prepared)
    with pytest.raises(LTXAVRuntimeError, match=r"\('video', 'audio'\)"):
        _sample(
            runtime,
            MultiStreamLatent.from_pairs(
                (("video", _video()), ("audio", _audio()), ("extra", _video()))
            ),
            prepared,
        )
    with pytest.raises(LTXAVRuntimeError, match=r"\[B,4,T,H,W\]"):
        _sample(runtime, _streams(video=_video((1, 5, 1, 2, 2))), prepared)
    with pytest.raises(LTXAVRuntimeError, match=r"\[B,4,T,H,W\]"):
        _sample(runtime, _streams(video=torch.zeros((1, 4, 2, 2))), prepared)
    with pytest.raises(LTXAVRuntimeError, match=r"\[B,8,T,16\]"):
        _sample(runtime, _streams(audio=_audio((1, 7, 1, 16))), prepared)
    with pytest.raises(LTXAVRuntimeError, match=r"\[B,8,T,16\]"):
        _sample(runtime, _streams(audio=_audio((1, 8, 1, 15))), prepared)
    with pytest.raises(LTXAVRuntimeError, match=r"\[B,8,T,16\]"):
        _sample(runtime, _streams(audio=torch.zeros((8, 1, 16))), prepared)
    with pytest.raises(LTXAVRuntimeError, match="share one batch size"):
        _sample(runtime, _streams(audio=_audio((2, 8, 1, 16))), prepared)
    with pytest.raises(TypeError, match="strided floating"):
        _sample(runtime, _streams(video=torch.zeros((1, 4, 1, 2, 2), dtype=torch.int64)), prepared)


def _record_and_binding(
    text: torch.Tensor | None = None,
    *,
    token_count: int | None = 4,
    family_id: str = "dinkster.ltxav",
    streams: tuple[str, ...] = ("gemma3_12b",),
    segment_name: str = "gemma3_12b",
    space: str = "conditioning-text",
) -> tuple[ConditioningRecord, Any]:
    payload = torch.ones((1, 4, 8)) if text is None else text
    binding = tensor_to_payload_binding("ltxav-text", payload, space=space)
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id), binding.shape, binding.dtype, binding.space
    )
    record = ConditioningRecord(
        channels=((ConditioningChannel.TEXT, descriptor),),
        token_layout=TokenLayoutDescriptor(
            family_id,
            1,
            streams,
            (TokenSegmentDescriptor(segment_name, streams[0], 0, token_count),),
        ),
    )
    return record, binding


def test_prepare_conditioning_requires_an_exact_carrier() -> None:
    runtime, _ = _runtime()

    with pytest.raises(TypeError, match="exact ConditioningCarrier"):
        runtime.prepare_conditioning(cast("Any", "carrier"))


def test_prepare_conditioning_requires_exactly_one_record() -> None:
    runtime, _ = _runtime()
    first, first_binding = _record_and_binding()
    second, second_binding = _record_and_binding()
    carrier = make_conditioning_carrier(
        ConditioningSet((first, second)), (first_binding, second_binding)
    )

    with pytest.raises(LTXAVRuntimeError, match="exactly one conditioning record"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_channels_not_consumed_by_ltxav() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    channels = (
        (ConditioningChannel.CONCAT_LATENT, dict(record.channels)[ConditioningChannel.TEXT]),
    )
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, channels=channels),)), (binding,)
    )

    with pytest.raises(LTXAVRuntimeError, match="does not consume.*concat_latent"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_wrong_text_width() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(torch.ones((1, 4, 9)))
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXAVRuntimeError, match=r"\[B,tokens,8\]"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_partial_schedules() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, schedule=PercentRange(0.0, 0.5)),)), (binding,)
    )

    with pytest.raises(LTXAVRuntimeError, match="schedules"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_requires_a_token_layout() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, token_layout=None),)), (binding,)
    )

    with pytest.raises(LTXAVRuntimeError, match="requires a token layout"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_foreign_family_layouts() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(family_id="dinkster.ltxv")
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXAVRuntimeError, match="unsupported"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize(
    "segment_name,streams", (("other", ("gemma3_12b",)), ("t5xxl", ("t5xxl",)))
)
def test_prepare_conditioning_requires_the_exact_gemma_layout(
    segment_name: str, streams: tuple[str, ...]
) -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(segment_name=segment_name, streams=streams)
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXAVRuntimeError, match="exact gemma3_12b token layout"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize(
    ("config", "stream", "expected"),
    (
        (LTXAV_22B_V23_CONFIG, "gemma4_12b", "gemma3_12b"),
        (LTXAV_22B_V25_CONFIG, "gemma3_12b", "gemma4_12b"),
    ),
)
def test_prepare_conditioning_requires_the_gemma_stream_for_the_diffusion_profile(
    config: object,
    stream: str,
    expected: str,
) -> None:
    runtime, diffusion = _runtime()
    cast("Any", diffusion).config = config
    record, binding = _record_and_binding(
        torch.ones((1, 4, 6144)),
        streams=(stream,),
        segment_name=stream,
    )
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXAVRuntimeError, match=f"exact {expected} token layout"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize("token_count", (None, 3, 5))
def test_prepare_conditioning_requires_the_exact_payload_token_count(
    token_count: int | None,
) -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(token_count=token_count)
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXAVRuntimeError, match="cover exactly the TEXT payload rows"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_requires_matching_payload_bindings() -> None:
    runtime, _ = _runtime()
    record, _ = _record_and_binding()
    stray = tensor_to_payload_binding("unrelated", torch.ones((1, 4, 8)), space="conditioning-text")

    with pytest.raises((LTXAVRuntimeError, ValueError)):
        runtime.prepare_conditioning(
            make_conditioning_carrier(ConditioningSet((record,)), (stray,))
        )


def test_prepare_conditioning_refuses_integer_payloads() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(torch.ones((1, 4, 8), dtype=torch.int64))
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(TypeError, match="strided floating"):
        runtime.prepare_conditioning(carrier)


def test_conditioning_identity_pins_the_model_profile() -> None:
    runtime, _ = _runtime()

    assert runtime.conditioning_identity == (
        "dinkster.ltxav.conditioning:v1:dinkster.ltxav:4:16:8:2:128:16:8:2:4:1:False:1000.0"
    )


def test_22b_conditioning_identity_pins_the_dual_projection_profile() -> None:
    runtime, diffusion = _runtime()
    cast("Any", diffusion).config = LTXAV_22B_V23_CONFIG

    assert runtime._text_dim == 6144  # pyright: ignore[reportPrivateUsage]
    assert runtime.conditioning_identity == (
        "dinkster.ltxav.conditioning:v1:dinkster.ltxav:128:4096:128:32:128:2048:64:32:"
        "3840:48:True:1000.0:22b-v2.3"
    )


@pytest.mark.parametrize("use_keyframe_embedding", (False, True))
def test_v25_runtime_accepts_optional_keyframe_embedding_without_changing_conditioning(
    use_keyframe_embedding: bool,
) -> None:
    diffusion = object.__new__(ltxav_runtime.LTXAVModel)
    cast("Any", diffusion).config = replace(
        LTXAV_22B_V25_CONFIG,
        use_keyframes_abs_pos_embedding=use_keyframe_embedding,
    )

    runtime = LTXAVDiffusionRuntime(
        diffusion,
        runtime_identity="v25",
        compute_dtype=torch.bfloat16,
    )

    assert runtime.conditioning_identity.endswith(":22b-v2.5")
    assert runtime.video_vae_config is LTXAV_22B_V25_VAE_CONFIG


def test_runtime_satisfies_the_conditioning_preparation_protocol() -> None:
    runtime, _ = _runtime()

    assert isinstance(runtime, MultiStreamConditioningRuntime)


class _VAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []
        self.decoded: list[torch.Tensor] = []
        self.encode_budgets: list[int | None] = []
        self.decode_budgets: list[int | None] = []

    def encode(self, content: torch.Tensor, *, max_chunk_bytes: int | None = None) -> torch.Tensor:
        self.encoded.append(content.detach().clone())
        self.encode_budgets.append(max_chunk_bytes)
        return content

    def decode(self, latent: torch.Tensor, *, max_chunk_bytes: int | None = None) -> torch.Tensor:
        self.decoded.append(latent.detach().clone())
        self.decode_budgets.append(max_chunk_bytes)
        return latent


class _DiffusionVAE(_VAE):
    def decode(self, latent: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.decoded.append(latent.detach().clone())
        self.decode_budgets.append(cast("int | None", kwargs.get("max_chunk_bytes")))
        return latent


def _codec_runtime() -> tuple[LTXAVVideoCodecRuntime, _VAE]:
    vae = _VAE()
    return LTXAVVideoCodecRuntime(cast("Any", vae), compute_dtype=torch.float32), vae


def test_init_requires_an_exact_ltxav_model_and_nonempty_identity() -> None:
    with pytest.raises(ValueError, match="exact supported model"):
        LTXAVDiffusionRuntime(
            cast("Any", _Diffusion()),
            runtime_identity="test:ltxav",
            compute_dtype=torch.float32,
        )
    nonfinite = _Diffusion()
    cast("Any", nonfinite).config = replace(
        LTXAV_19B_CONFIG,
        av_ca_timestep_scale_multiplier=float("inf"),
    )
    with pytest.raises(ValueError, match="exact supported model"):
        LTXAVDiffusionRuntime(
            cast("Any", nonfinite),
            runtime_identity="test:ltxav",
            compute_dtype=torch.float32,
        )
    diffusion = _Diffusion()
    diffusion.config = cast("Any", LTXAV_19B_CONFIG)
    with pytest.raises(ValueError, match="identity must be nonempty"):
        LTXAVDiffusionRuntime(
            cast("Any", diffusion),
            runtime_identity="",
            compute_dtype=torch.float32,
        )


def test_encode_content_normalizes_into_the_signed_vae_domain() -> None:
    runtime, vae = _codec_runtime()
    content = torch.full((1, 3, 1, 32, 32), 0.5)

    latent = runtime.encode_content(content)

    assert torch.equal(vae.encoded[0], torch.zeros((1, 3, 1, 32, 32)))
    assert latent.dtype == torch.float32
    assert torch.equal(latent, torch.zeros((1, 3, 1, 32, 32)))


def test_encode_content_center_crops_to_the_spatial_downscale_grid() -> None:
    runtime, vae = _codec_runtime()

    runtime.encode_content(torch.zeros((1, 3, 1, 33, 65)))

    assert vae.encoded[0].shape == (1, 3, 1, 32, 64)


def test_decode_latent_denormalizes_and_clamps_the_output() -> None:
    runtime, vae = _codec_runtime()
    high_latent = torch.full((1, 128, 1, 2, 2), 3.0)
    low_latent = torch.full((1, 128, 1, 2, 2), -5.0)

    high = runtime.decode_latent(high_latent)
    low = runtime.decode_latent(low_latent)

    assert len(vae.decoded) == 2
    assert high.data_ptr() == high_latent.data_ptr()
    assert low.data_ptr() == low_latent.data_ptr()
    assert torch.equal(high, torch.ones((1, 128, 1, 2, 2)))
    assert torch.equal(low, torch.zeros((1, 128, 1, 2, 2)))
    assert high.dtype == torch.float32


@pytest.mark.parametrize("vae_type", (_VAE, _DiffusionVAE))
def test_codec_derives_budgets_from_total_memory_for_both_video_vaes(
    monkeypatch: pytest.MonkeyPatch,
    vae_type: type[_VAE],
) -> None:
    vae = vae_type()
    runtime = LTXAVVideoCodecRuntime(cast("Any", vae), compute_dtype=torch.float32)
    total_memory = iter((24 * 1024**3, 15 * 1024**3))
    measured_devices: list[torch.device] = []

    def measure(device: torch.device) -> int:
        measured_devices.append(device)
        return next(total_memory)

    def free_memory(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(free_total=5 * 1024**3)

    monkeypatch.setattr(ltxav_runtime, "get_total_memory", measure, raising=False)
    monkeypatch.setattr(
        ltxav_runtime,
        "get_free_memory",
        free_memory,
        raising=False,
    )
    runtime.encode_content(torch.zeros((1, 3, 1, 32, 32)))
    runtime.decode_latent(torch.zeros((1, 128, 1, 2, 2)))

    assert measured_devices == [torch.device("cpu"), torch.device("cpu")]
    assert vae.encode_budgets == [128 * 1024**2]
    assert vae.decode_budgets == [80 * 1024**2]


class _AudioVAE:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.decoded: list[torch.Tensor] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decoded.append(latent.detach().clone())
        return torch.ones((1, 2, 3, 64))


class _Vocoder:
    def __init__(self) -> None:
        self.config = SimpleNamespace(audio_channels=2, output_sample_rate=24000)
        self.features: list[torch.Tensor] = []

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        self.features.append(features.detach().clone())
        return torch.full((1, 2, 10), 0.5)


class _BWEVocoder(_Vocoder):
    def __init__(self) -> None:
        super().__init__()
        self.config = LTXAV_BWE_VOCODER_CONFIG


def _audio_runtime(audio_vae: _AudioVAE, vocoder: _Vocoder) -> LTXAVAudioCodecRuntime:
    codec = SimpleNamespace(audio_vae=audio_vae, vocoder=vocoder)
    return LTXAVAudioCodecRuntime(cast("Any", codec))


def test_decode_audio_latent_hands_the_mel_output_to_the_vocoder() -> None:
    audio_vae = _AudioVAE()
    vocoder = _Vocoder()
    runtime = _audio_runtime(audio_vae, vocoder)

    preview = runtime.decode_audio_latent(torch.ones((1, 8, 3, 16)))

    assert audio_vae.decoded[0].shape == (1, 8, 3, 16)
    assert vocoder.features[0].shape == (1, 2, 64, 3)
    assert preview.sample_rate == 24000
    assert preview.waveform.dtype == torch.float32
    assert torch.equal(preview.waveform, torch.full((1, 2, 10), 0.5))


def test_decode_audio_latent_uses_the_bwe_base_width_and_output_rate() -> None:
    audio_vae = _AudioVAE()
    vocoder = _BWEVocoder()
    runtime = _audio_runtime(audio_vae, vocoder)

    preview = runtime.decode_audio_latent(torch.ones((1, 8, 3, 16)))

    assert vocoder.features[0].shape == (1, 2, 64, 3)
    assert preview.sample_rate == 48000


class _ArithmeticDiffusion(_Diffusion):
    """Input-dependent fake: the velocities couple latents, timesteps,
    and context so schedule, noise, and CFG-lane parity are all
    visible."""

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        frame_rate: float,
        denoise_mask: torch.Tensor | None = None,
        ref_audio_tokens: torch.Tensor | None = None,
        stg_self_attn_blocks: frozenset[int] = frozenset(),
        a2v_cross_attention: bool = True,
        v2a_cross_attention: bool = True,
        generated_keyframes: LTXGeneratedKeyframes | None = None,
        context_preprocessed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert context_preprocessed is True
        self.denoise_masks.append(None if denoise_mask is None else denoise_mask.detach().clone())
        self.reference_audio.append(
            None if ref_audio_tokens is None else ref_audio_tokens.detach().clone()
        )
        self.execution_options.append(
            (stg_self_attn_blocks, a2v_cross_attention, v2a_cross_attention)
        )
        self.generated_keyframes.append(generated_keyframes)
        self.calls.append(
            (
                video.detach().clone(),
                audio.detach().clone(),
                timesteps.detach().clone(),
                audio_timesteps.detach().clone(),
                (context.detach().clone(), attention_mask),
                frame_rate,
            )
        )
        scale = context.float().mean(dim=(1, 2))
        video_step = (
            timesteps.reshape(video.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1, 1) * 1e-3
        )
        audio_step = (
            audio_timesteps.reshape(audio.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1) * 1e-3
        )
        return (
            video * 0.5 + video_step * 0.25 + scale.reshape(-1, 1, 1, 1, 1),
            audio * 0.5 + audio_step * 0.25 + scale.reshape(-1, 1, 1, 1),
        )


def _random_streams(seed: int = 11) -> MultiStreamLatent[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return MultiStreamLatent.from_pairs(
        (
            ("video", torch.rand((1, 4, 1, 2, 2), generator=generator)),
            ("audio", torch.rand((1, 8, 1, 16), generator=generator)),
        )
    )


def _lane(value: float, frame_rate: float = 25.0) -> LTXAVPreparedConditioning:
    return LTXAVPreparedConditioning(torch.full((1, 4, 8), value), frame_rate)


def _custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def _manual_ksampler_custom(
    runtime: LTXAVDiffusionRuntime,
    streams: MultiStreamLatent[torch.Tensor],
    *,
    cond: LTXAVPreparedConditioning,
    cfg: object = None,
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    seed: int,
    segment: SamplingSegment | None = None,
    sampling_shift: float | None = None,
    denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
) -> MultiStreamLatent[torch.Tensor]:
    """The KSampler sugar written out by hand: the schedule for the
    resolved sampler and scheduler over the (optionally shifted) LTX
    flow space, and the reference initial-noise draw over the packed
    audio-video tensor."""
    sampler = torch_sampler_registry().get(sampler_id)
    scheduler = torch_scheduler_registry().get(scheduler_id)
    assert sampler is not None
    assert scheduler is not None
    space = LTX_SIGMAS if sampling_shift is None else FluxFlowSigmas(shift=sampling_shift)
    schedule = build_sampling_schedule(
        scheduler, space, sampler, steps, denoise=1.0, flow=True, segment=segment
    )
    if segment is not None and not segment.add_noise:
        noise = streams.map(
            lambda stream: torch.zeros_like(stream, dtype=torch.float32, device="cpu")
        )
    else:
        packed, layout = pack_latent_streams(
            MultiStreamLatent.from_pairs(
                (
                    ("video", streams.by_role("video").to(device="cpu", dtype=torch.float32)),
                    ("audio", streams.by_role("audio").to(device="cpu", dtype=torch.float32)),
                )
            )
        )
        noise = unpack_latent_streams(prepare_noise(packed, seed), layout)
    result = runtime.sample_custom(
        streams,
        noise=noise,
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, cond),
        cfg=cast("Any", cfg),
        request=CustomSamplingRequest(sampler, (), schedule.pre_offset),
        seed=seed,
        sampling_shift=sampling_shift,
        denoise_mask=denoise_mask,
    )
    output = result.output
    assert type(output) is MultiStreamLatent
    return output


def test_ltxav_runtime_satisfies_the_custom_sampling_protocol() -> None:
    runtime, _ = _runtime()
    assert isinstance(runtime, CustomSamplingRuntime)


@pytest.mark.parametrize(
    ("sampler_id", "scheduler_id", "steps", "cfg_scale", "segment"),
    [
        ("dinkster.euler", "dinkster.simple", 2, None, None),
        ("dinkster.res_multistep", "dinkster.simple", 2, 2.0, None),
        ("dinkster.dpmpp_sde", "dinkster.simple", 3, None, None),
        (
            "dinkster.euler",
            "dinkster.simple",
            3,
            None,
            SamplingSegment(
                steps=3,
                start_step=1,
                end_step=3,
                add_noise=False,
                return_with_leftover_noise=False,
            ),
        ),
    ],
    ids=("euler", "cfg", "brownian", "segment"),
)
def test_ksampler_surface_is_bit_equal_sugar_over_sample_custom(
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    cfg_scale: float | None,
    segment: SamplingSegment | None,
) -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    positive = _lane(2.0)
    negative = None if cfg_scale is None else _lane(5.0)
    streams = _random_streams()
    identity = runtime.conditioning_identity

    expected = runtime.sample_multistream(
        streams,
        conditioning=positive,
        cfg=SamplingGuidance(cast("Any", negative), 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
    )
    output = _manual_ksampler_custom(
        runtime,
        streams,
        cond=positive,
        cfg=SamplingGuidance(
            None if negative is None else PreparedMultiStreamConditioning(identity, negative),
            1.0 if cfg_scale is None else cfg_scale,
        ),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        seed=185,
        segment=segment,
    )

    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_with_reference_audio_and_denoise_mask() -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    reference = torch.arange(3 * 128, dtype=torch.float32).reshape(1, 3, 128)
    positive = LTXAVPreparedConditioning(torch.full((1, 4, 8), 2.0), 25.0, reference)
    streams = _random_streams()
    mask = MultiStreamLatent.from_pairs(
        (
            ("video", torch.zeros((1, 1, 1, 2, 2))),
            ("audio", torch.ones((1, 1, 1, 1))),
        )
    )

    expected = runtime.sample_multistream(
        streams,
        conditioning=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        denoise_mask=mask,
    )
    output = _manual_ksampler_custom(
        runtime,
        streams,
        cond=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=185,
        denoise_mask=mask,
    )

    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_for_low_precision_latents() -> None:
    """Seeded noise is drawn over a float32 view of the packed streams
    on both surfaces, so half-precision latent streams must not round
    the draw on the decomposed path."""
    runtime, _ = _runtime(_ArithmeticDiffusion())
    streams = _random_streams().map(lambda stream: stream.to(dtype=torch.float16))
    positive = _lane(2.0)

    expected = runtime.sample_multistream(
        streams,
        conditioning=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
    )
    output = _manual_ksampler_custom(
        runtime,
        streams,
        cond=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=185,
    )

    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_for_dual_guidance_lanes() -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    streams = _random_streams()
    positive = _lane(2.0)
    negative = _lane(5.0)
    middle = _lane(3.0)
    identity = runtime.conditioning_identity

    expected = runtime.sample_multistream(
        streams,
        conditioning=positive,
        cfg=DualSamplingGuidance(cast("Any", middle), cast("Any", negative), 3.0, 1.5),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
    )
    output = _manual_ksampler_custom(
        runtime,
        streams,
        cond=positive,
        cfg=DualSamplingGuidance(
            cast("Any", PreparedMultiStreamConditioning(identity, middle)),
            cast("Any", PreparedMultiStreamConditioning(identity, negative)),
            3.0,
            1.5,
        ),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=185,
    )

    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_for_explicit_sampling_shift() -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    streams = _random_streams()
    positive = _lane(2.0)

    expected = runtime.sample_multistream(
        streams,
        conditioning=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=77,
        sampling_shift=5.0,
    )
    output = _manual_ksampler_custom(
        runtime,
        streams,
        cond=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=77,
        sampling_shift=5.0,
    )

    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_custom_sampling_sigma_surfaces_match_the_ltx_flow_space() -> None:
    runtime, _ = _runtime()
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, LTX_SIGMAS, 4, denoise=0.5
    )
    with pytest.raises(LTXAVRuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("test.missing", 4, 1.0)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        LTX_SIGMAS, 4, 0.6, 0.6
    )
    with pytest.raises(ValueError, match="discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(2, 1.0)
    assert runtime.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=False
    ) == custom_percent_to_sigma(
        LTX_SIGMAS, LTX_SIGMAS.percent_to_sigma, 0.5, return_actual_sigma=False
    )
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=True
    ) == custom_percent_to_sigma(
        LTX_SIGMAS, LTX_SIGMAS.percent_to_sigma, 0.3, return_actual_sigma=True
    )


def test_sample_custom_refuses_foreign_inputs_and_unsupported_modes() -> None:
    runtime, fake = _runtime()
    conditioning = _prepared()
    prepared = PreparedMultiStreamConditioning(runtime.conditioning_identity, conditioning)
    latent = _random_streams()
    noise = latent.map(torch.zeros_like)
    request = _custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": prepared, "request": request}
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(TypeError, match="exact MultiStreamLatent"):
        sample(latent=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(LTXAVRuntimeError, match="requires MultiStreamLatent noise"):
        sample(noise=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(LTXAVRuntimeError, match="match the latent stream roles"):
        sample(noise=MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, 1, 2, 2))),)))
    with pytest.raises(TypeError, match="strided floating"):
        sample(
            noise=MultiStreamLatent.from_pairs(
                (
                    ("video", torch.zeros((1, 4, 1, 2, 2), dtype=torch.int64)),
                    ("audio", torch.zeros((1, 8, 1, 16))),
                )
            )
        )
    with pytest.raises(LTXAVRuntimeError, match="match the latent stream shapes"):
        sample(
            noise=MultiStreamLatent.from_pairs(
                (
                    ("video", torch.zeros((1, 4, 1, 2, 2))),
                    ("audio", torch.zeros((1, 8, 1, 15))),
                )
            )
        )
    with pytest.raises(LTXAVRuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(cast("Any", prepared), cast("Any", prepared), 3.0, 1.0))
    with pytest.raises(LTXAVRuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    invalid_mask = latent.replace("video", torch.full_like(latent.by_role("video"), math.nan))
    with pytest.raises(LTXAVRuntimeError, match=r"denoise mask values.*\[0, 1\]"):
        sample(denoise_mask=invalid_mask)
    with pytest.raises(LTXAVRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(LTXAVRuntimeError, match="context windows"):
        sample(context_windows=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(LTXAVRuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(LTXAVRuntimeError, match="brownian sampler needs positive sigmas"):
        sample(request=_custom_request("dinkster.dpmpp_sde", sigmas=(0.0, 0.0)))
    with pytest.raises(LTXAVRuntimeError, match="prepared multi-stream conditioning"):
        sample(cond=cast("Any", conditioning))
    with pytest.raises(LTXAVRuntimeError, match="a different conditioner component"):
        sample(cond=PreparedMultiStreamConditioning("native:other", conditioning))
    with pytest.raises(TypeError, match="exact LTXAVPreparedConditioning"):
        sample(
            cond=PreparedMultiStreamConditioning(
                runtime.conditioning_identity, cast("Any", object())
            )
        )
    with pytest.raises(LTXAVRuntimeError, match="guidance requires prepared"):
        sample(cfg=SamplingGuidance(cast("Any", conditioning), 2.0))
    with pytest.raises(LTXAVRuntimeError, match="different conditioner components"):
        sample(
            cfg=SamplingGuidance(
                cast("Any", PreparedMultiStreamConditioning("native:other", conditioning)), 2.0
            )
        )
    with pytest.raises(TypeError, match="guidance lanes require exact LTXAVPreparedConditioning"):
        sample(
            cfg=SamplingGuidance(
                cast(
                    "Any",
                    PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, cast("Any", object())
                    ),
                ),
                2.0,
            )
        )
    with pytest.raises(LTXAVRuntimeError, match="share one frame rate"):
        sample(
            cfg=SamplingGuidance(
                cast(
                    "Any",
                    PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, _lane(5.0, frame_rate=30.0)
                    ),
                ),
                2.0,
            )
        )
    assert not fake.calls


def test_sample_custom_captures_denoised_output() -> None:
    runtime, _ = _runtime()
    latent = _random_streams()
    result = runtime.sample_custom(
        latent,
        noise=latent.map(torch.zeros_like),
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, _prepared()),
        request=_custom_request(),
        seed=9,
    )
    output = result.output
    assert type(output) is MultiStreamLatent
    assert output.roles == ("video", "audio")
    assert result.denoised_output is not None
    assert result.denoised_output.roles == ("video", "audio")
