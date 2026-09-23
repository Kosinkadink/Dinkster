from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import dinkster_inference_torch.minimax_h3_runtime as h3_runtime_module
import pytest
import torch
from dinkster_inference import (
    MINIMAX_H3,
    MINIMAX_H3_AUDIO_MASK_MAPPING,
    MINIMAX_H3_SIGMAS,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    AdapterPatch,
    AreaDescriptor,
    AreaUnits,
    CancellationFlag,
    Conditioning,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    CustomSamplingRequest,
    CustomSamplingResult,
    DualSamplingGuidance,
    ExecutionObserverAttachment,
    ExecutionSpanEvent,
    GuidanceContractError,
    GuidanceEvaluationRequest,
    GuidancePredictions,
    KeyedContribution,
    LatentStream,
    MaskDescriptor,
    MiniMaxH3AudioContent,
    MiniMaxH3AudioReference,
    MiniMaxH3FL2VARequest,
    MiniMaxH3Keyframe,
    MiniMaxH3KeyframeRole,
    MiniMaxH3REF2VARequest,
    MiniMaxH3Sigmas,
    MiniMaxH3T2VARequest,
    MiniMaxH3Task,
    MiniMaxH3VideoReference,
    MultiStreamLatent,
    Parameterization,
    PatchEntry,
    PatchSet,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PreparedConditioningCarrier,
    PreparedMultiStreamConditioning,
    Registry,
    SamplingCancelled,
    SamplingGuidance,
    SamplingSegment,
    SamplingStateEvent,
    SchedulerDescriptor,
    StepEvent,
    TimelineGuide,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
    offset_first_sigma_for_snr,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    INFERENCE_PATCH_PROVIDERS_SURFACE,
    MASK_PAYLOAD_SPACE,
    AttentionKernel,
    LoRAAdapter,
    MiniMaxH3AttentionKernelFactory,
    MiniMaxH3AudioVaeRuntime,
    MiniMaxH3ConditionerRuntime,
    MiniMaxH3DiTRuntime,
    MiniMaxH3PackedSequenceFacts,
    MiniMaxH3RuntimeError,
    MiniMaxH3VideoVaeRuntime,
    add_minimax_h3_motion_context,
    add_minimax_h3_timeline_guide,
    basic_conditioning_to_carrier,
    empty_minimax_h3_av,
    pack_latent_streams,
    prepare_noise,
    tensor_to_payload_binding,
    unpack_latent_streams,
)
from dinkster_inference_torch import sampling_execution as sampling_execution_module
from dinkster_inference_torch.attention import builtin_sdpa_kernel
from dinkster_inference_torch.brownian import BrownianTreeNoise
from dinkster_inference_torch.denoise import (
    PackedInpaintConfiguration,
    _InpaintDenoiser,  # pyright: ignore[reportPrivateUsage]
    prepare_noise_from_generator,
    run_sampler_engine,
)
from dinkster_inference_torch.distributed import DistributedSamplingConfig
from dinkster_inference_torch.guidance import ConditioningValidationPath, GuidedDenoiser
from dinkster_inference_torch.minimax_h3_conditioning import MiniMaxH3ConditionerInputs
from dinkster_inference_torch.minimax_h3_dit import MiniMaxH3DiTConditioning
from dinkster_inference_torch.patch_providers import PatchProviderSnapshot
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.scheduled_sampling import (
    ScheduledPatchResolution,
    ScheduledPatchResolutionRequest,
    ScheduledPatchResolver,
    ScheduledSamplingOptions,
)
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry


def _h3(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))


class FakeConditioner:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, inputs: MiniMaxH3ConditionerInputs) -> torch.Tensor:
        self.calls += 1
        return torch.zeros((1, inputs.ids.shape[1], 5120), device=inputs.ids.device)


class FakeVideoVAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    @staticmethod
    def encode_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
        batch, _, frames, height, width = input_shape
        temporal = 1 if frames == 1 else 5 * ((frames + 16) // 17) - 3
        return batch, 24, temporal, (height + 15) // 16, (width + 15) // 16

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self.encoded.append(content)
        shape = self.encode_output_shape(tuple(content.shape))
        return torch.zeros(
            shape,
            device=content.device,
            dtype=content.dtype,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + 10


class FakeAudioVAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    @staticmethod
    def encode_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
        batch, stereo, samples = input_shape
        return batch, 32, stereo, (samples + 799) // 800

    def encode(self, waveform: torch.Tensor, *, sample_rate: int) -> torch.Tensor:
        assert sample_rate == 32_000
        self.encoded.append(waveform)
        return torch.zeros(
            self.encode_output_shape(tuple(waveform.shape)),
            device=waveform.device,
            dtype=waveform.dtype,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :2, 0]


class FakeDiT:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[float, object]] = []
        self.sampler_schedules: list[tuple[float, ...] | None] = []
        self.denoise_masks: list[MultiStreamLatent[torch.Tensor] | None] = []
        self.preprocessed_contexts: list[torch.Tensor] = []
        self.attention_kernel = builtin_sdpa_kernel()
        self.video_patch_proj = torch.nn.Linear(1, 1, bias=False)

    def preprocess_text_embeddings(self, context: torch.Tensor) -> torch.Tensor:
        self.preprocessed_contexts.append(context)
        return context

    def __call__(
        self,
        value: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: MiniMaxH3Sigmas,
        sampler_sigmas: tuple[float, ...] | None = None,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
        attention_kernel_factory: MiniMaxH3AttentionKernelFactory | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        del context, conditioning
        self.calls.append((sigma, sigmas))
        self.sampler_schedules.append(sampler_sigmas)
        self.denoise_masks.append(denoise_mask)
        if attention_kernel_factory is not None:
            attention_kernel_factory(MiniMaxH3PackedSequenceFacts(1, ((0, 1, "video"),)))
        return _h3(
            torch.full_like(value.by_role("video"), self.value, dtype=torch.float32),
            torch.full_like(value.by_role("audio"), self.value, dtype=torch.float32),
        )


@dataclass
class RuntimeFixture:
    conditioner_runtime: MiniMaxH3ConditionerRuntime
    fl2va_runtime: MiniMaxH3DiTRuntime
    ref2va_runtime: MiniMaxH3DiTRuntime
    video_runtime: MiniMaxH3VideoVaeRuntime
    audio_runtime: MiniMaxH3AudioVaeRuntime
    fl2va: FakeDiT
    ref2va: FakeDiT
    conditioner: FakeConditioner
    video_vae: FakeVideoVAE
    audio_vae: FakeAudioVAE


@pytest.fixture
def runtime_fixture() -> RuntimeFixture:
    fl2va = FakeDiT(2.0)
    ref2va = FakeDiT(3.0)
    conditioner = FakeConditioner()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    video_runtime = MiniMaxH3VideoVaeRuntime(
        video_vae,  # type: ignore[arg-type]
        runtime_identity="test:h3:video",
        compute_dtype=torch.float16,
    )
    audio_runtime = MiniMaxH3AudioVaeRuntime(
        audio_vae,  # type: ignore[arg-type]
        runtime_identity="test:h3:audio",
        compute_dtype=torch.float32,
    )
    return RuntimeFixture(
        MiniMaxH3ConditionerRuntime(
            conditioner,  # type: ignore[arg-type]
            video_runtime,
            audio_runtime,
            runtime_identity="test:h3:conditioner",
        ),
        MiniMaxH3DiTRuntime(
            fl2va,  # type: ignore[arg-type]
            model_role="fl2va_dit",
            runtime_identity="test:h3:fl2va",
        ),
        MiniMaxH3DiTRuntime(
            ref2va,  # type: ignore[arg-type]
            model_role="ref2va_dit",
            runtime_identity="test:h3:ref2va",
        ),
        video_runtime,
        audio_runtime,
        fl2va,
        ref2va,
        conditioner,
        video_vae,
        audio_vae,
    )


def _target() -> MultiStreamLatent[torch.Tensor]:
    return empty_minimax_h3_av(
        width=32,
        height=32,
        frame_count=5,
        device="cpu",
        dtype=torch.float32,
    )


def test_h3_vae_runtimes_expose_mask_geometry(runtime_fixture: RuntimeFixture) -> None:
    assert runtime_fixture.video_runtime.latent_mask_mapping is MINIMAX_H3_VIDEO_MASK_MAPPING
    assert runtime_fixture.audio_runtime.latent_mask_mapping is MINIMAX_H3_AUDIO_MASK_MAPPING


def test_h3_sampling_uses_reference_float32_video_sigmas(
    runtime_fixture: RuntimeFixture,
) -> None:
    space = runtime_fixture.fl2va_runtime.sampling_sigma_space()
    assert space is MINIMAX_H3_SIGMAS.video
    scheduler = torch_scheduler_registry().get("simple")
    assert scheduler is not None
    sigmas = sampling_sigmas(scheduler, space, 20, denoise=1.0)
    assert sigmas[2] == 0.9908256530761719
    assert sigmas[3] == 0.9855073094367981
    assert sigmas[8] == 0.9473683834075928
    assert sigmas[19] == 0.3870967924594879


def test_h3_token_masks_pool_odd_video_patches_audio_features_and_quantize_up() -> None:
    video_channel = torch.tensor(
        (
            (0.10, 0.20, 0.30, 0.40, 0.41),
            (0.05, 0.15, 0.25, 0.35, 0.45),
            (0.01, 0.02, 0.03, 0.04, 0.05),
        )
    ).reshape(1, 1, 1, 3, 5)
    video = torch.cat((torch.zeros_like(video_channel), video_channel), dim=1)
    audio = torch.tensor(
        (
            ((0.10, 0.20), (0.30, 0.40)),
            ((0.50, 0.05), (0.10, 0.45)),
            ((0.20, 0.35), (0.25, 0.15)),
        )
    ).reshape(1, 3, 2, 2)

    masks = h3_runtime_module._h3_token_grid_masks(  # pyright: ignore[reportPrivateUsage]
        _h3(video, audio), (1, 2, 2)
    )

    expected_video = (
        torch.tensor(
            (
                (52, 52, 103, 103, 116),
                (52, 52, 103, 103, 116),
                (6, 6, 11, 11, 13),
            ),
            dtype=torch.float32,
        ).reshape(1, 1, 1, 3, 5)
        / 256.0
    )
    expected_audio = (
        torch.tensor(((128, 90), (77, 116)), dtype=torch.float32).reshape(1, 1, 2, 2) / 256.0
    )
    torch.testing.assert_close(masks.by_role("video"), expected_video.expand_as(video).contiguous())
    torch.testing.assert_close(masks.by_role("audio"), expected_audio.expand_as(audio).contiguous())


def _h3_inpaint_denoiser(
    inner: object,
    *,
    raw_mask: torch.Tensor,
    token_mask: torch.Tensor,
    latent: torch.Tensor,
    noise: torch.Tensor,
    video_elements: int,
) -> _InpaintDenoiser:
    return _InpaintDenoiser(
        cast("Any", inner),
        mask=raw_mask.to(device=latent.device, dtype=torch.float32),
        latent=latent.to(dtype=torch.float32),
        noise=noise.to(device=latent.device, dtype=torch.float32),
        parameterization=Parameterization.FLOW,
        packed=PackedInpaintConfiguration(
            token_mask,
            video_elements,
            MINIMAX_H3_SIGMAS.video.shift,
            MINIMAX_H3_SIGMAS.audio_shift,
            MINIMAX_H3_SIGMAS.audio_scale,
        ),
    )


def test_h3_inpaint_denoiser_uses_token_input_and_raw_output_masks() -> None:
    class PairDenoiser:
        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
            del sigma
            self.inputs.append(x)
            return x + 2.0

        def call_with_uncond(
            self, x: torch.Tensor, _sigma: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.inputs.append(x)
            return x + 3.0, x + 4.0

    inner = PairDenoiser()
    raw_mask = torch.tensor((0.25, 0.75, 0.20, 0.60)).reshape(1, 1, 4)
    token_mask = torch.tensor((0.50, 1.00, 0.40, 0.80)).reshape(1, 1, 4)
    latent = torch.tensor((10.0, 20.0, 30.0, 40.0)).reshape(1, 1, 4)
    noise = torch.tensor((1.0, 2.0, 3.0, 4.0)).reshape(1, 1, 4)
    wrapped = _h3_inpaint_denoiser(
        inner,
        raw_mask=raw_mask,
        token_mask=token_mask,
        latent=latent,
        noise=noise,
        video_elements=2,
    )
    x = torch.tensor((5.0, 6.0, 7.0, 8.0)).reshape(1, 1, 4)
    sigma = 0.5
    injected = torch.empty_like(x)
    injected[..., :2] = 0.999 * latent[..., :2] + 0.001 * noise[..., :2]
    sigma_video = torch.tensor(sigma, dtype=torch.float32).clamp(min=1e-6)
    base = sigma_video / (
        MINIMAX_H3_SIGMAS.video.shift + sigma_video * (1.0 - MINIMAX_H3_SIGMAS.video.shift)
    )
    sigma_audio = (
        MINIMAX_H3_SIGMAS.audio_shift * base / (1.0 + (MINIMAX_H3_SIGMAS.audio_shift - 1.0) * base)
    )
    audio_factor = (sigma_video / sigma_audio) / MINIMAX_H3_SIGMAS.audio_scale
    injected[..., 2:] = latent[..., 2:] * audio_factor.to(latent.dtype)
    weight = (token_mask - raw_mask) / (1.0 - raw_mask).clamp(min=1e-6)
    weight = torch.where(raw_mask < 1.0, weight.clamp(0.0, 1.0), torch.zeros_like(weight))
    token_source = injected + weight * (x - injected)
    expected_input = x * raw_mask + token_source * (1.0 - raw_mask)

    actual = wrapped(x, sigma)
    combined, uncond = wrapped.call_with_uncond(x, sigma)

    assert len(inner.inputs) == 2
    torch.testing.assert_close(inner.inputs[0], expected_input)
    torch.testing.assert_close(inner.inputs[1], expected_input)
    torch.testing.assert_close(actual, (expected_input + 2.0) * raw_mask + latent * (1 - raw_mask))
    torch.testing.assert_close(
        combined, (expected_input + 3.0) * raw_mask + latent * (1 - raw_mask)
    )
    torch.testing.assert_close(uncond, expected_input + 4.0)


def test_h3_inpaint_denoiser_keeps_fixed_inputs_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PairDenoiser:
        def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
            del sigma
            return x

        def call_with_uncond(
            self, x: torch.Tensor, sigma: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del sigma
            return x, x

    wrapped = _h3_inpaint_denoiser(
        PairDenoiser(),
        raw_mask=torch.full((1, 1, 4), 0.5, dtype=torch.float64),
        token_mask=torch.full((1, 1, 4), 0.75, dtype=torch.float64),
        latent=torch.ones((1, 1, 4), dtype=torch.float32),
        noise=torch.zeros((1, 1, 4), dtype=torch.float64),
        video_elements=2,
    )
    assert all(
        tensor.device == torch.device("cpu") and tensor.dtype == torch.float32
        for tensor in (
            wrapped.mask,
            cast("PackedInpaintConfiguration", wrapped.packed).token_mask,
            wrapped.latent,
            wrapped.noise,
        )
    )

    def refuse_transfer(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("inpaint evaluation transferred a fixed tensor")

    monkeypatch.setattr(torch.Tensor, "to", refuse_transfer)
    x = torch.zeros((1, 1, 4), dtype=torch.float32)
    wrapped(x, 0.5)
    wrapped.call_with_uncond(x, 0.5)


def test_h3_token_input_reference_composition_preserves_binary_edges() -> None:
    raw = torch.tensor((0.0, 0.25, 255 / 256, 1 - 2**-24, 1 - 1e-7, 1.0))
    token = torch.ceil(raw * 256.0) / 256.0
    x = torch.tensor((-2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    injected = torch.tensor((4.0, 3.0, 2.0, 1.0, 0.0, -1.0))
    direct = x * token + injected * (1.0 - token)
    weight = (token - raw) / (1.0 - raw).clamp(min=1e-6)
    weight = torch.where(raw < 1.0, weight.clamp(0.0, 1.0), torch.zeros_like(weight))
    reference_source = injected + weight * (x - injected)
    reference = x * raw + reference_source * (1.0 - raw)

    assert not torch.equal(direct, reference)
    assert torch.equal(reference[raw == 1.0], x[raw == 1.0])
    assert torch.equal(reference[raw == 0.0], injected[raw == 0.0])
    binary = torch.tensor((0.0, 1.0))
    binary_weight = torch.zeros_like(binary)
    binary_source = injected[:2] + binary_weight * (x[:2] - injected[:2])
    binary_reference = x[:2] * binary + binary_source * (1.0 - binary)
    binary_direct = x[:2] * binary + injected[:2] * (1.0 - binary)
    assert torch.equal(binary_direct, binary_reference)


def test_h3_runtime_stays_off_the_conditioning_preparation_protocol() -> None:
    # H3 conditioning is family-specific; the generic carrier-preparing
    # seam must not structurally claim this runtime.
    for runtime_type in (MiniMaxH3ConditionerRuntime, MiniMaxH3DiTRuntime):
        assert not hasattr(runtime_type, "prepare_conditioning")


def test_single_dit_component_refuses_task_role_mismatches(
    runtime_fixture: RuntimeFixture,
) -> None:
    fl2va = MiniMaxH3DiTRuntime(
        runtime_fixture.fl2va,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:fl2va",
    )
    ref2va = MiniMaxH3DiTRuntime(
        runtime_fixture.ref2va,  # type: ignore[arg-type]
        model_role="ref2va_dit",
        runtime_identity="test:ref2va",
    )
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("role admission"),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    arguments: dict[str, Any] = {
        "sampler_id": "res_multistep",
        "scheduler_id": "simple",
        "steps": 1,
        "denoise": 1.0,
        "seed": 123,
    }

    with pytest.raises(MiniMaxH3RuntimeError) as fl2va_failure:
        fl2va.sample_multistream(
            target,
            conditioning=replace(prepared, task=MiniMaxH3Task.REF2VA),
            **arguments,
        )
    assert str(fl2va_failure.value) == ("MiniMax H3 fl2va_dit component cannot sample task REF2VA")

    with pytest.raises(MiniMaxH3RuntimeError) as ref2va_failure:
        ref2va.sample_multistream(target, conditioning=prepared, **arguments)
    assert str(ref2va_failure.value) == ("MiniMax H3 ref2va_dit component cannot sample task T2VA")


def test_single_dit_component_exposes_runtime_identity(
    runtime_fixture: RuntimeFixture,
) -> None:
    identity = "native:dinkster.minimax_h3:" + "5" * 64
    component = MiniMaxH3DiTRuntime(
        runtime_fixture.ref2va,  # type: ignore[arg-type]
        model_role="ref2va_dit",
        runtime_identity=identity,
        receipt_identity="ref2va-receipt",
    )

    assert component.runtime_identity == identity
    assert component.receipt_identity == "ref2va-receipt"


def test_empty_av_snaps_to_exact_video_audio_geometry() -> None:
    value = empty_minimax_h3_av(
        width=64,
        height=32,
        frame_count=6,
        device="cpu",
        dtype=torch.float32,
    )
    assert value.by_role("video").shape == (1, 24, 7, 2, 4)
    assert value.by_role("audio").shape == (1, 32, 2, 37)
    with pytest.raises(ValueError, match="multiples of 32"):
        empty_minimax_h3_av(
            width=48,
            height=32,
            frame_count=5,
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="floating dtype"):
        empty_minimax_h3_av(
            width=32,
            height=32,
            frame_count=5,
            device="cpu",
            dtype=torch.int64,
        )


def test_ordinary_empty_latent_adapts_to_h3_av_geometry(
    runtime_fixture: RuntimeFixture,
) -> None:
    ordinary = torch.zeros((2, 4, 64, 96), dtype=torch.float16)

    value = runtime_fixture.conditioner_runtime.adapt_multistream_latent(
        ordinary,
        source_spatial_downscale=8,
    )

    video = value.by_role("video")
    audio = value.by_role("audio")
    assert video.shape == (2, 24, 1, 32, 48)
    assert audio.shape == (2, 32, 2, 2)
    assert video.dtype == audio.dtype == ordinary.dtype
    assert video.device == audio.device == ordinary.device
    assert torch.count_nonzero(video) == torch.count_nonzero(audio) == 0

    temporal = runtime_fixture.fl2va_runtime.adapt_multistream_latent(
        torch.zeros((1, 4, 6, 8, 12)),
        source_spatial_downscale=16,
        source_temporal_downscale=1,
    )
    assert temporal.by_role("video").shape == (1, 24, 2, 8, 12)
    assert temporal.by_role("audio").shape == (1, 32, 2, 8)


def test_ordinary_latent_adaptation_refuses_nonempty_and_invalid_inputs(
    runtime_fixture: RuntimeFixture,
) -> None:
    adapt = runtime_fixture.conditioner_runtime.adapt_multistream_latent
    with pytest.raises(ValueError, match="all-zero"):
        adapt(torch.ones((1, 4, 8, 8)))
    with pytest.raises(TypeError, match="floating"):
        adapt(torch.zeros((1, 4, 8, 8), dtype=torch.int64))
    with pytest.raises(ValueError, match="positive integer"):
        adapt(
            torch.zeros((1, 4, 8, 8)),
            source_spatial_downscale=0,
        )


def test_t2va_condition_and_sample_use_fl2va_and_paired_pack(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    assert prepared.task.value == "t2va"
    state_events: list[SamplingStateEvent[object]] = []
    span_events: list[ExecutionSpanEvent] = []

    def capture_state(event: SamplingStateEvent[object]) -> None:
        state_events.append(event)
        assert type(event.current) is MultiStreamLatent
        event.current.by_role("video").fill_(99)
        assert type(event.denoised) is MultiStreamLatent
        event.denoised.by_role("audio").fill_(99)

    result = runtime_fixture.fl2va_runtime.sample_multistream(
        target,
        conditioning=prepared,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=1.0,
        seed=123,
        on_step=None,
        on_state=capture_state,
        cancelled=lambda: False,
        observer=ExecutionObserverAttachment(span_events.append, "sample-test"),
        parent_span_id=7,
    )
    packed, layout = pack_latent_streams(target)
    expected = prepare_noise(packed, 123) - 2.0
    video_elements = layout.by_role("video").elements
    expected[..., video_elements:] /= MINIMAX_H3_SIGMAS.audio_scale
    result_packed, result_layout = pack_latent_streams(result)
    assert result_layout == layout
    torch.testing.assert_close(result_packed, expected)
    assert runtime_fixture.fl2va.calls == [(1.0, MINIMAX_H3_SIGMAS)]
    assert runtime_fixture.fl2va.sampler_schedules == [(1.0, 0.0)]
    assert len(runtime_fixture.fl2va.preprocessed_contexts) == 1
    assert runtime_fixture.ref2va.calls == []
    assert len(state_events) == 1
    state = state_events[0]
    assert state.phase == "pre_update"
    assert type(state.current) is MultiStreamLatent
    assert state.current.roles == ("video", "audio")
    assert type(state.denoised) is MultiStreamLatent
    assert [event.operation for event in span_events] == [
        "prepare",
        "prepare",
        "model_evaluation",
        "model_evaluation",
        "finalize",
        "finalize",
    ]
    assert [event.phase for event in span_events] == ["begin", "end"] * 3
    assert all(event.parent_span_id == 7 for event in span_events)


def test_h3_conditioning_carries_and_compiles_the_declared_global_layout(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[GuidedDenoiser] = []
    original = sampling_execution_module.guided_denoiser

    def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
        guided = original(*args, **kwargs)
        captured.append(guided)
        return guided

    monkeypatch.setattr(sampling_execution_module, "guided_denoiser", capture)
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("declared rows"),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    assert prepared.token_layout is not None
    assert (
        prepared.token_layout.partition.sequence_length == prepared.token_layout.layout.valid_rows
    )
    assert prepared.token_layout.partition.shard_count == 1
    assert tuple(segment.identity for segment in prepared.token_layout.layout.segments) == (
        "text",
        "target-audio",
        "target-video",
    )

    runtime_fixture.fl2va_runtime.sample_multistream(
        target,
        conditioning=prepared,
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=0.0,
        seed=17,
    )
    compiled = captured[0].conditioning_plan
    assert compiled is not None
    assert len(compiled.calls) == 1
    assert compiled.calls[0].layout_digest == prepared.token_layout.layout.digest
    assert compiled.lanes[0].validation is ConditioningValidationPath.LAYOUT_BACKED
    assert compiled.lanes[0].token_transforms == tuple(
        (transform.transform, transform.digest) for transform in prepared.token_layout.transforms
    )


def test_h3_timeline_guide_adapter_appends_guides_and_rebuilds_the_layout(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("timeline guides"),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    video_audio = TimelineGuide(
        0,
        1,
        _h3(
            torch.zeros(1, 24, 1, 2, 2),
            torch.zeros(1, 32, 2, 3),
        ),
    )

    first = add_minimax_h3_timeline_guide(prepared, target, video_audio)
    overlapping = TimelineGuide(
        1,
        1,
        MultiStreamLatent.from_pairs((("video", torch.zeros(1, 24, 1, 2, 2)),)),
    )
    with pytest.raises(ValueError, match="guide 2 overlaps guide 1"):
        add_minimax_h3_timeline_guide(first, target, overlapping)
    audio_only = TimelineGuide(
        4,
        1,
        MultiStreamLatent.from_pairs((("audio", torch.zeros(1, 32, 2, 1)),)),
    )
    second = add_minimax_h3_timeline_guide(first, target, audio_only)

    assert prepared.dit.guides == ()
    assert first.dit.guides == (video_audio,)
    assert second.dit.guides == (video_audio, audio_only)
    assert second.dit.frame_count == 5
    assert second.token_layout is not None
    assert tuple(segment.identity for segment in second.token_layout.layout.segments) == (
        "text",
        "guide-1-video",
        "guide-1-audio",
        "guide-2-audio",
        "target-audio",
        "target-video",
    )
    assert second.token_layout.layout.by_identity("guide-1-video").grid == (1, 1, 1)
    assert second.token_layout.layout.by_identity("guide-1-audio").grid == (2, 3)
    assert second.token_layout.layout.by_identity("guide-2-audio").grid == (2, 1)


def test_h3_timeline_guide_adapter_refuses_foreign_target_and_tensor_contracts(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("timeline guide validation"),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    guide = TimelineGuide(
        0,
        1,
        MultiStreamLatent.from_pairs(
            (("video", torch.zeros(1, 24, 1, 2, 2, dtype=torch.float16)),)
        ),
    )

    with pytest.raises(MiniMaxH3RuntimeError, match="tensor contract"):
        add_minimax_h3_timeline_guide(prepared, target, guide)
    foreign = _h3(target.by_role("video"), torch.zeros(1, 32, 2, 9))
    with pytest.raises(MiniMaxH3RuntimeError, match="differs from prepared conditioning"):
        add_minimax_h3_timeline_guide(prepared, foreign, guide)


def test_h3_motion_context_clones_aligned_av_tail_and_preserves_existing_guides(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = empty_minimax_h3_av(
        width=32,
        height=32,
        frame_count=39,
        device="cpu",
        dtype=torch.float32,
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("continue motion"),
        target=target,
        frame_count=39,
        payloads={},
        cancelled=lambda: False,
    )
    end_guide = TimelineGuide(
        38,
        1,
        MultiStreamLatent.from_pairs(
            (("video", torch.zeros(1, 24, 1, 2, 2, dtype=torch.float32)),)
        ),
    )
    prepared = add_minimax_h3_timeline_guide(prepared, target, end_guide)
    previous_video = torch.arange(1 * 24 * 17 * 2 * 2, dtype=torch.float32).reshape(1, 24, 17, 2, 2)
    previous_audio = torch.arange(1 * 32 * 2 * 93, dtype=torch.float32).reshape(1, 32, 2, 93)
    previous = _h3(previous_video, previous_audio)

    conditioned, trim_time = add_minimax_h3_motion_context(prepared, target, previous, 6)

    assert trim_time == 22 / 24
    assert conditioned.dit.guides[0] is end_guide
    context = conditioned.dit.guides[1]
    assert (context.frame_index, context.frame_count) == (0, 22)
    context_video = context.latent.by_role("video")
    context_audio = context.latent.by_role("audio")
    torch.testing.assert_close(context_video, previous_video[:, :, -7:])
    torch.testing.assert_close(context_audio, previous_audio[..., -37:])
    assert context_video.untyped_storage().data_ptr() != previous_video.untyped_storage().data_ptr()
    assert context_audio.untyped_storage().data_ptr() != previous_audio.untyped_storage().data_ptr()
    assert conditioned.token_layout is not None
    assert tuple(segment.identity for segment in conditioned.token_layout.layout.segments) == (
        "text",
        "guide-1-video",
        "guide-2-video",
        "guide-2-audio",
        "target-audio",
        "target-video",
    )


def test_h3_motion_context_refuses_invalid_geometry_and_limits(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = empty_minimax_h3_av(
        width=32,
        height=32,
        frame_count=39,
        device="cpu",
        dtype=torch.float32,
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("continue motion validation"),
        target=target,
        frame_count=39,
        payloads={},
        cancelled=lambda: False,
    )
    previous = empty_minimax_h3_av(
        width=32,
        height=32,
        frame_count=39,
        device="cpu",
        dtype=torch.float32,
    )

    with pytest.raises(TypeError, match="context_length must be an exact integer"):
        add_minimax_h3_motion_context(prepared, target, previous, True)
    malformed_audio = previous.replace("audio", torch.zeros(1, 32, 2, 64))
    with pytest.raises(MiniMaxH3RuntimeError, match="audio length does not match"):
        add_minimax_h3_motion_context(prepared, target, malformed_audio, 22)
    foreign_canvas = _h3(torch.zeros(1, 24, 12, 2, 3), torch.zeros(1, 32, 2, 65))
    with pytest.raises(MiniMaxH3RuntimeError, match="same canvas"):
        add_minimax_h3_motion_context(prepared, target, foreign_canvas, 22)
    with pytest.raises(MiniMaxH3RuntimeError, match="exceeds the previous clip"):
        add_minimax_h3_motion_context(prepared, target, previous, 40)
    foreign_target = empty_minimax_h3_av(
        width=64,
        height=32,
        frame_count=39,
        device="cpu",
        dtype=torch.float32,
    )
    foreign_previous = empty_minimax_h3_av(
        width=64,
        height=32,
        frame_count=39,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(MiniMaxH3RuntimeError, match="differs from prepared conditioning"):
        add_minimax_h3_motion_context(prepared, foreign_target, foreign_previous, 22)

    short_target = _target()
    short_prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("short continuation target"),
        target=short_target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    with pytest.raises(MiniMaxH3RuntimeError, match="does not fit the target"):
        add_minimax_h3_motion_context(short_prepared, short_target, previous, 22)


def test_h3_declared_layout_refuses_materialized_text_mismatch_before_empty_schedule(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest("declared rows"),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    malformed = replace(prepared, context=prepared.context[:, :-1])

    with pytest.raises(GuidanceContractError, match="materialized packed rows"):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=malformed,
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=0.0,
            seed=17,
        )


def test_h3_declared_layout_refuses_wrong_keyframe_grid_before_empty_schedule(
    runtime_fixture: RuntimeFixture,
) -> None:
    image = torch.zeros((1, 32, 64, 3))
    descriptor = PayloadDescriptor(
        PayloadReference("first"), tuple(image.shape), "float32", "worker:minimax-h3"
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3FL2VARequest(
            "declared keyframe",
            (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, descriptor),),
        ),
        target=_target(),
        frame_count=5,
        payloads={"first": image},
        cancelled=lambda: False,
    )
    assert prepared.dit.keyframes[0].video.shape == (1, 24, 1, 2, 4)

    with pytest.raises(GuidanceContractError, match="materialized packed rows or grids"):
        runtime_fixture.fl2va_runtime.sample_multistream(
            _target(),
            conditioning=prepared,
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=0.0,
            seed=17,
        )
    assert runtime_fixture.fl2va.calls == []


def test_h3_declared_layout_refuses_equal_row_different_reference_grid(
    runtime_fixture: RuntimeFixture,
) -> None:
    frame = torch.zeros((1, 32, 64, 3))
    descriptor = PayloadDescriptor(
        PayloadReference("frame"), tuple(frame.shape), "float32", "worker:minimax-h3"
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3REF2VARequest(
            "declared reference",
            (MiniMaxH3VideoReference((descriptor,), (0,), (0.0,)),),
        ),
        target=_target(),
        frame_count=5,
        payloads={"frame": frame},
        cancelled=lambda: False,
    )
    reference = prepared.dit.references[0]
    assert reference.video is not None
    assert prepared.token_layout is not None
    assert prepared.token_layout.layout.by_identity("reference-1-video").grid == (1, 1, 2)
    malformed_reference = replace(
        reference,
        video=torch.zeros((1, 24, 1, 4, 2), dtype=reference.video.dtype),
    )
    malformed = replace(
        prepared,
        dit=replace(prepared.dit, references=(malformed_reference,)),
    )

    with pytest.raises(GuidanceContractError, match="materialized packed rows or grids"):
        runtime_fixture.ref2va_runtime.sample_multistream(
            _target(),
            conditioning=malformed,
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=0.0,
            seed=17,
        )
    assert runtime_fixture.ref2va.calls == []


def test_unset_distributed_mode_preserves_factory_bytes_and_every_guidance_evaluation(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINKSTER_SINGLE_JOB_RANK", raising=False)
    conditioner = runtime_fixture.conditioner_runtime
    runtime = runtime_fixture.fl2va_runtime
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    negative = replace(prepared, context=torch.ones_like(prepared.context))

    def sample(
        factory: MiniMaxH3AttentionKernelFactory | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        return runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(negative, 2.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=lambda: False,
            attention_kernel_factory=factory,
        )

    expected = sample()
    factory_calls = 0

    def factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        nonlocal factory_calls
        factory_calls += 1
        return runtime_fixture.fl2va.attention_kernel

    model_calls_before = len(runtime_fixture.fl2va.calls)
    actual = sample(factory)

    assert factory_calls == len(runtime_fixture.fl2va.calls) - model_calls_before == 2
    assert torch.equal(actual.by_role("video"), expected.by_role("video"))
    assert torch.equal(actual.by_role("audio"), expected.by_role("audio"))


class ContextMeanDiT(torch.nn.Module):
    """Velocity equals the conditioning context's mean, so the cond and
    uncond branches produce distinguishable predictions."""

    def __init__(self, output_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.calls: list[float] = []
        self.conditionings: list[MiniMaxH3DiTConditioning] = []
        self.preprocessed_contexts: list[torch.Tensor] = []
        self.input_dtypes: list[torch.dtype] = []
        self.output_dtype = output_dtype
        self.video_patch_proj = torch.nn.Linear(1, 1, bias=False)
        self.video_patch_proj.weight.data.fill_(1.0)

    def preprocess_text_embeddings(self, context: torch.Tensor) -> torch.Tensor:
        self.preprocessed_contexts.append(context)
        return context

    def __call__(
        self,
        value: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: MiniMaxH3Sigmas,
        sampler_sigmas: tuple[float, ...] | None = None,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        del sigma, sigmas, sampler_sigmas, denoise_mask
        self.conditionings.append(conditioning)
        projection_input = context.mean().reshape(1, 1).to(self.video_patch_proj.weight.dtype)
        velocity = float(self.video_patch_proj(projection_input).detach().item())
        self.calls.append(velocity)
        self.input_dtypes.append(value.by_role("video").dtype)
        return _h3(
            torch.full_like(value.by_role("video"), velocity, dtype=self.output_dtype),
            torch.full_like(value.by_role("audio"), velocity, dtype=self.output_dtype),
        )


def _context_mean_runtime() -> tuple[
    MiniMaxH3DiTRuntime, MiniMaxH3ConditionerRuntime, ContextMeanDiT
]:
    dit = ContextMeanDiT()
    runtime = MiniMaxH3DiTRuntime(
        dit,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
    )
    conditioner = MiniMaxH3ConditionerRuntime(
        FakeConditioner(),  # type: ignore[arg-type]
        runtime_identity="test:h3:conditioner",
    )
    return runtime, conditioner, dit


def _h3_prepared_carrier(
    runtime: MiniMaxH3DiTRuntime,
    entries: tuple[
        tuple[
            h3_runtime_module.MiniMaxH3PreparedConditioning,
            PercentRange,
            torch.Tensor | None,
            AreaDescriptor | None,
            tuple[tuple[str, Any], ...],
        ],
        ...,
    ],
) -> PreparedMultiStreamConditioning:
    records: list[ConditioningRecord] = []
    bindings = []
    payloads: list[object] = []
    for index, (prepared, schedule, mask, area, extension_metadata) in enumerate(entries):
        text = tensor_to_payload_binding(
            f"text-{index}", prepared.context, space="conditioning-text"
        )
        bindings.append(text)
        mask_descriptor = None
        if mask is not None:
            mask_binding = tensor_to_payload_binding(
                f"mask-{index}", mask, space=MASK_PAYLOAD_SPACE
            )
            bindings.append(mask_binding)
            mask_descriptor = MaskDescriptor(PayloadReference(mask_binding.reference_id))
        records.append(
            ConditioningRecord(
                channels=(
                    (
                        ConditioningChannel.TEXT,
                        PayloadDescriptor(
                            PayloadReference(text.reference_id),
                            text.shape,
                            text.dtype,
                            text.space,
                        ),
                    ),
                ),
                area=area,
                mask=mask_descriptor,
                schedule=schedule,
                extension_metadata=extension_metadata,
                token_layout=TokenLayoutDescriptor(
                    MINIMAX_H3.id,
                    1,
                    ("text",),
                    (TokenSegmentDescriptor("text", "text", 0, prepared.context.shape[1]),),
                ),
            )
        )
        payloads.append(prepared)
    carrier = make_conditioning_carrier(ConditioningSet(tuple(records)), tuple(bindings))
    return PreparedMultiStreamConditioning(
        runtime.conditioning_identity,
        PreparedConditioningCarrier(carrier, tuple(payloads)),
    )


_H3_TEST_OVERLAY = "1" * 64
_H3_TEST_STACK = hashlib.sha256(
    json.dumps([_H3_TEST_OVERLAY], separators=(",", ":")).encode()
).hexdigest()


def _h3_patch_metadata() -> tuple[tuple[str, object], ...]:
    effective = hashlib.sha256(
        json.dumps(
            {
                "version": 1,
                "target": MINIMAX_H3.id,
                "text": None,
                "diffusion": _H3_TEST_STACK,
                "transforms": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return (
        ("dinkster.inference/version", 1),
        ("dinkster.inference/target", MINIMAX_H3.id),
        ("dinkster.inference/text-overlay-digests", ()),
        ("dinkster.inference/diffusion-overlay-digests", (_H3_TEST_OVERLAY,)),
        ("dinkster.inference/transform-ids", ()),
        ("dinkster.inference/transform-digests", ()),
        ("dinkster.inference/effective-patch-state", effective),
        ("dinkster.inference/diffusion-overlay-stack-digest", _H3_TEST_STACK),
    )


def _h3_lora_resolver(model: ContextMeanDiT, calls: list[object]) -> ScheduledPatchResolver:
    patch_set = PatchSet(
        {
            "video_patch_proj.weight": (
                PatchEntry(AdapterPatch(LoRAAdapter(torch.ones((1, 1)), torch.ones((1, 1))))),
            )
        },
        structural_digest=_H3_TEST_STACK,
    )
    declaration = KeyedContribution(INFERENCE_PATCH_PROVIDERS_SURFACE, "test.provider")
    snapshot = PatchProviderSnapshot((declaration,))

    def resolve(
        requests: tuple[ScheduledPatchResolutionRequest, ...],
        cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]:
        calls.append((requests, cancel))
        return tuple(
            ScheduledPatchResolution(
                request,
                "worker-generation-1",
                snapshot,
                declaration.id,
                declaration,
                patch_set,
            )
            for request in requests
        )

    return resolve


def test_sampling_casts_latents_to_the_diffusion_compute_dtype() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()

    runtime.sample_multistream(
        target,
        conditioning=_condition_t2va(conditioner, target),
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="euler",
        scheduler_id="simple",
        steps=1,
        denoise=1.0,
        seed=123,
        cancelled=lambda: False,
    )

    assert dit.input_dtypes == [torch.bfloat16]


def test_sampling_upcasts_velocity_before_sigma_multiplication() -> None:
    dit = ContextMeanDiT(torch.bfloat16)
    runtime = MiniMaxH3DiTRuntime(
        dit,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
    )
    conditioner = MiniMaxH3ConditionerRuntime(
        FakeConditioner(),  # type: ignore[arg-type]
        runtime_identity="test:h3:conditioner",
    )
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    cond = PreparedMultiStreamConditioning(
        "test:h3:fl2va", replace(prepared, context=torch.full_like(prepared.context, 2.0))
    )
    events: list[SamplingStateEvent[object]] = []

    runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (0.7, 0.0)),
        on_state=events.append,
    )

    denoised = cast("MultiStreamLatent[torch.Tensor]", events[0].denoised)
    expected = torch.full_like(denoised.by_role("video"), -1.4)
    torch.testing.assert_close(denoised.by_role("video"), expected, rtol=0, atol=0)


def test_h3_prepared_carrier_uses_shared_scheduled_conditioning() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    scheduled = replace(prepared, context=torch.ones_like(prepared.context))
    carrier = basic_conditioning_to_carrier(
        Conditioning(scheduled.context),
        token_layout=TokenLayoutDescriptor(
            MINIMAX_H3.id,
            1,
            ("text",),
            (TokenSegmentDescriptor("text", "text", 0, scheduled.context.shape[1]),),
        ),
    )
    cond = PreparedMultiStreamConditioning(
        runtime.conditioning_identity,
        PreparedConditioningCarrier(carrier, (scheduled,)),
    )

    runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (0.7, 0.0)),
        compute_dtype=torch.float32,
    )

    assert dit.calls == [1.0]
    assert dit.conditionings == [scheduled.dit]
    assert scheduled.task is prepared.task
    assert scheduled.frame_count == prepared.frame_count
    assert scheduled.dit is prepared.dit
    assert scheduled.target_layout is prepared.target_layout
    assert scheduled.token_layout is prepared.token_layout


def test_h3_shared_prompt_schedule_switches_at_step_boundary() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    first = replace(prepared, context=torch.ones_like(prepared.context))
    second = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    cond = _h3_prepared_carrier(
        runtime,
        (
            (first, PercentRange(0.0, 0.5), None, None, ()),
            (second, PercentRange(0.5, 1.0), None, None, ()),
        ),
    )

    runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (1.0, 0.7, 0.0)),
        compute_dtype=torch.float32,
    )

    assert dit.calls == [1.0, 2.0]


def test_h3_shared_conditioning_mask_and_area_weight_video_regions() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    first = replace(prepared, context=torch.ones_like(prepared.context))
    second = replace(prepared, context=torch.full_like(prepared.context, 3.0))
    video = target.by_role("video")
    height, width = video.shape[-2:]
    left = torch.zeros((1, height, width), dtype=torch.float32)
    left[..., : width // 2] = 1.0
    right = AreaDescriptor(
        height,
        width - width // 2,
        0,
        width // 2,
        AreaUnits.LATENT_CELLS,
    )
    cond = _h3_prepared_carrier(
        runtime,
        (
            (first, PercentRange(0.0, 1.0), left, None, ()),
            (second, PercentRange(0.0, 1.0), None, right, ()),
        ),
    )

    result = runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (1.0, 0.0)),
        compute_dtype=torch.float32,
    ).output.by_role("video")

    left_result = result[..., : width // 2]
    right_result = result[..., width // 2 :]
    torch.testing.assert_close(left_result, torch.full_like(left_result, -1.0))
    torch.testing.assert_close(right_result, torch.full_like(right_result, -3.0))


def test_h3_shared_lora_schedule_switches_at_step_boundary() -> None:
    dit = ContextMeanDiT()
    runtime = MiniMaxH3DiTRuntime(
        dit,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
        compute_dtype=torch.float32,
    )
    conditioner = MiniMaxH3ConditionerRuntime(
        FakeConditioner(),  # type: ignore[arg-type]
        runtime_identity="test:h3:conditioner",
    )
    target = _target()
    base = _condition_t2va(conditioner, target)
    prepared = replace(base, context=torch.ones_like(base.context))
    cond = _h3_prepared_carrier(
        runtime,
        (
            (
                prepared,
                PercentRange(0.0, 0.5),
                None,
                None,
                _h3_patch_metadata(),
            ),
            (prepared, PercentRange(0.5, 1.0), None, None, ()),
        ),
    )
    resolver_calls: list[object] = []

    runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (1.0, 0.7, 0.0)),
        compute_dtype=torch.float32,
        scheduled=ScheduledSamplingOptions(resolver=_h3_lora_resolver(dit, resolver_calls)),
    )

    assert len(resolver_calls) == 1
    assert dit.calls == [2.0, 1.0]


def test_h3_shared_lora_mask_confines_patch_to_video_region() -> None:
    dit = ContextMeanDiT()
    runtime = MiniMaxH3DiTRuntime(
        dit,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
        compute_dtype=torch.float32,
    )
    conditioner = MiniMaxH3ConditionerRuntime(
        FakeConditioner(),  # type: ignore[arg-type]
        runtime_identity="test:h3:conditioner",
    )
    target = _target()
    base = _condition_t2va(conditioner, target)
    prepared = replace(base, context=torch.ones_like(base.context))
    video = target.by_role("video")
    height, width = video.shape[-2:]
    left = torch.zeros((1, height, width), dtype=torch.float32)
    left[..., : width // 2] = 1.0
    cond = _h3_prepared_carrier(
        runtime,
        (
            (
                prepared,
                PercentRange(0.0, 1.0),
                left,
                None,
                _h3_patch_metadata(),
            ),
            (prepared, PercentRange(0.0, 1.0), None, None, ()),
        ),
    )
    resolver_calls: list[object] = []

    result = runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", (1.0, 0.0)),
        compute_dtype=torch.float32,
        scheduled=ScheduledSamplingOptions(resolver=_h3_lora_resolver(dit, resolver_calls)),
    ).output.by_role("video")

    assert len(resolver_calls) == 1
    torch.testing.assert_close(
        result[..., : width // 2], torch.full_like(result[..., : width // 2], -1.5)
    )
    torch.testing.assert_close(
        result[..., width // 2 :], torch.full_like(result[..., width // 2 :], -1.0)
    )


def _condition_t2va(
    runtime: MiniMaxH3ConditionerRuntime,
    target: MultiStreamLatent[torch.Tensor],
):  # noqa: ANN202
    return runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )


def _sample_cfg(
    runtime: MiniMaxH3DiTRuntime,
    target: MultiStreamLatent[torch.Tensor],
    conditioning: object,
    uncond: object | None,
    cfg_scale: float,
) -> MultiStreamLatent[torch.Tensor]:
    return runtime.sample_multistream(
        target,
        conditioning=conditioning,
        cfg=SamplingGuidance(uncond, cfg_scale),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=1.0,
        seed=123,
        cancelled=lambda: False,
    )


def test_cfg_combines_cond_and_uncond_predictions() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    result = _sample_cfg(runtime, target, prepared, negative, 2.0)
    assert dit.calls == [0.0, 5.0]
    packed, layout = pack_latent_streams(target)
    combined_velocity = 5.0 + (0.0 - 5.0) * 2.0
    expected = prepare_noise(packed, 123) - combined_velocity
    video_elements = layout.by_role("video").elements
    expected[..., video_elements:] /= MINIMAX_H3_SIGMAS.audio_scale
    result_packed, result_layout = pack_latent_streams(result)
    assert result_layout == layout
    torch.testing.assert_close(result_packed, expected)


def test_runtime_forwards_the_token_mask_to_both_guidance_lanes(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    prepared = _condition_t2va(runtime_fixture.conditioner_runtime, target)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    video_mask = torch.ones_like(target.by_role("video"))
    video_mask[..., 0, 0] = 0.2
    audio_mask = torch.ones_like(target.by_role("audio"))
    audio_mask[:, :, 0, 0] = 0.3
    raw_masks = _h3(video_mask, audio_mask)
    expected = h3_runtime_module._h3_token_grid_masks(  # pyright: ignore[reportPrivateUsage]
        raw_masks, (1, 2, 2)
    )

    runtime_fixture.fl2va_runtime.sample_multistream(
        target,
        conditioning=prepared,
        cfg=SamplingGuidance(negative, 2.0),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=1.0,
        seed=123,
        denoise_mask=raw_masks,
        cancelled=lambda: False,
    )

    assert len(runtime_fixture.fl2va.denoise_masks) == 2
    for actual in runtime_fixture.fl2va.denoise_masks:
        assert type(actual) is MultiStreamLatent
        torch.testing.assert_close(actual.by_role("video"), expected.by_role("video"))
        torch.testing.assert_close(actual.by_role("audio"), expected.by_role("audio"))


def test_h3_fractional_denoise_mask_blends_at_the_declared_video_cells() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    video = target.by_role("video")
    mask = torch.full_like(video, 0.25)
    mask[..., :, 1] = 0.75

    result = runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        cfg=None,
        request=_h3_custom_request("euler", (1.0, 0.0)),
        denoise_mask=mask,
        compute_dtype=torch.float32,
    ).output.by_role("video")

    torch.testing.assert_close(result[..., :, 0], torch.full_like(result[..., :, 0], -0.5))
    torch.testing.assert_close(result[..., :, 1], torch.full_like(result[..., :, 1], -1.5))


@pytest.mark.parametrize("bad_value", (float("nan"), -0.1, 1.1))
def test_runtime_rejects_invalid_h3_mask_values(
    runtime_fixture: RuntimeFixture,
    bad_value: float,
) -> None:
    target = _target()
    prepared = _condition_t2va(runtime_fixture.conditioner_runtime, target)
    mask = torch.ones_like(target.by_role("video"))
    mask[..., 0, 0] = bad_value

    with pytest.raises(MiniMaxH3RuntimeError, match="mask values"):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            denoise_mask=mask,
            cancelled=lambda: False,
        )


def test_cfg_scale_one_ignores_uncond_and_evaluates_once() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    result = _sample_cfg(runtime, target, prepared, negative, 1.0)
    assert dit.calls == [0.0]
    packed, layout = pack_latent_streams(target)
    expected = prepare_noise(packed, 123) - 0.0
    video_elements = layout.by_role("video").elements
    expected[..., video_elements:] /= MINIMAX_H3_SIGMAS.audio_scale
    result_packed, _ = pack_latent_streams(result)
    torch.testing.assert_close(result_packed, expected)


@pytest.mark.parametrize("real_uncond", (False, True))
def test_cfgpp_receives_synthetic_or_real_unconditional_prediction(
    real_uncond: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    negative = (
        replace(prepared, context=torch.full_like(prepared.context, 5.0)) if real_uncond else None
    )
    uncond_records: list[tuple[torch.Tensor, float, torch.Tensor]] = []
    original_call = _InpaintDenoiser.call_with_uncond

    def record_call(
        self: Any,
        x: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = self._input(x, sigma)[0]
        combined, uncond = original_call(self, x, sigma)
        uncond_records.append((model_input, sigma, uncond))
        return combined, uncond

    monkeypatch.setattr(
        _InpaintDenoiser,
        "call_with_uncond",
        record_call,
    )
    result = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=SamplingGuidance(negative, 3.0),
        sampler_id="dinkster.euler_cfg_pp",
        scheduler_id="simple",
        steps=3,
        denoise=0.75,
        seed=123,
        denoise_mask=torch.full_like(target.by_role("video"), 0.5),
        cancelled=lambda: False,
    )
    assert dit.calls == ([2.0, 5.0] * 3 if real_uncond else [2.0] * 3)
    assert len(uncond_records) == 3
    if real_uncond:
        for model_input, sigma, uncond in uncond_records:
            torch.testing.assert_close(uncond, model_input - 5.0 * sigma)
    assert result.roles == target.roles


def test_h3_prepares_each_guidance_lane_once_for_a_multistep_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    preprocessed_dtypes: list[torch.dtype] = []

    def mark_preprocessed(context: torch.Tensor) -> torch.Tensor:
        preprocessed_dtypes.append(context.dtype)
        return context + 7.0

    monkeypatch.setattr(
        dit,
        "preprocess_text_embeddings",
        mark_preprocessed,
    )
    calls = 0
    original = h3_runtime_module.replace

    def counted(instance: object, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if isinstance(instance, MiniMaxH3DiTConditioning):
            calls += 1
        return original(cast(Any, instance), *args, **kwargs)

    monkeypatch.setattr(h3_runtime_module, "replace", counted)
    runtime.sample_multistream(
        target,
        conditioning=prepared,
        cfg=SamplingGuidance(negative, 3.0),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=3,
        denoise=1.0,
        seed=123,
        cancelled=lambda: False,
    )
    assert calls == 2
    assert set(dit.calls) == {7.0, 12.0}
    assert preprocessed_dtypes == [torch.bfloat16, torch.bfloat16]


def test_h3_zero_denoise_validates_guidance_lanes_before_return() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    with pytest.raises(TypeError, match="guidance lanes require"):
        runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(object(), 3.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=3,
            denoise=0.0,
            seed=123,
            cancelled=lambda: False,
        )
    assert dit.calls == []


def test_absent_uncond_with_cfg_above_one_stays_cond_only() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    result = _sample_cfg(runtime, target, positive, None, 3.0)
    assert dit.calls == [2.0]
    packed, layout = pack_latent_streams(target)
    expected = prepare_noise(packed, 123) - 2.0
    video_elements = layout.by_role("video").elements
    expected[..., video_elements:] /= MINIMAX_H3_SIGMAS.audio_scale
    result_packed, result_layout = pack_latent_streams(result)
    assert result_layout == layout
    assert torch.equal(result_packed, expected)


def test_sde_sampler_matches_the_explicit_pre_offset_brownian_reference() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    result = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="dpmpp_sde",
        scheduler_id="simple",
        steps=3,
        denoise=1.0,
        seed=123,
        cancelled=lambda: False,
    )
    sampler = torch_sampler_registry().get("dpmpp_sde")
    scheduler = torch_scheduler_registry().get("simple")
    assert sampler is not None and scheduler is not None
    sampler_latent = _h3(
        target.by_role("video"),
        target.by_role("audio") * MINIMAX_H3_SIGMAS.audio_scale,
    )
    packed, layout = pack_latent_streams(sampler_latent)
    pre_offset = sampling_sigmas(
        scheduler,
        MINIMAX_H3_SIGMAS.video,
        3,
        denoise=1.0,
        discard_penultimate=sampler.discard_penultimate,
    )
    schedule = offset_first_sigma_for_snr(pre_offset, MINIMAX_H3_SIGMAS.video, flow=True)
    assert schedule != pre_offset
    generator = torch.Generator("cpu")
    generator.manual_seed(123)
    stream_noise = sampler_latent.map(
        lambda stream: prepare_noise_from_generator(stream, generator, None)
    )
    noise, _ = pack_latent_streams(stream_noise)
    positive_sigmas = [sigma for sigma in pre_offset if sigma > 0]
    step_noise = BrownianTreeNoise(
        packed.to(dtype=torch.float32),
        min(positive_sigmas),
        max(pre_offset),
        seed=123,
        cpu=True,
    )

    def reference_denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x - torch.full_like(x, 2.0) * sigma

    output = run_sampler_engine(
        reference_denoiser,
        sampler.build(),
        latent=packed,
        noise=noise,
        sigmas=schedule,
        initial_sigma=pre_offset[0],
        parameterization=Parameterization.FLOW,
        sigma_max=MINIMAX_H3_SIGMAS.sigma_max,
        process_in=lambda value: value,
        process_out=lambda value: value,
        seed=123,
        noise_kind=sampler.noise,
        noise_sampler=step_noise,
        percent_to_sigma=MINIMAX_H3_SIGMAS.percent_to_sigma,
        device=packed.device,
    )
    unpacked = unpack_latent_streams(output, layout)
    assert torch.equal(result.by_role("video"), unpacked.by_role("video"))
    assert torch.equal(
        result.by_role("audio"),
        unpacked.by_role("audio") / MINIMAX_H3_SIGMAS.audio_scale,
    )


def test_brownian_sampler_refuses_nonempty_all_zero_custom_schedule() -> None:
    schedulers: Registry[SchedulerDescriptor] = Registry()
    schedulers.register(
        SchedulerDescriptor(
            id="test.all-zero",
            display_name="All zero",
            make_sigmas=lambda _steps, _space: (0.0,),
        )
    )
    runtime, conditioner, dit = _context_mean_runtime()
    runtime = MiniMaxH3DiTRuntime(
        dit,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
        scheduler_registry=schedulers,
    )
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    with pytest.raises(MiniMaxH3RuntimeError, match="H3 brownian sampler needs positive sigmas"):
        runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="dpmpp_sde",
            scheduler_id="test.all-zero",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=lambda: False,
        )


def test_uncond_type_task_and_layout_are_validated() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    with pytest.raises(TypeError, match="guidance lanes require"):
        _sample_cfg(runtime, target, prepared, object(), 2.0)
    with pytest.raises(MiniMaxH3RuntimeError, match="guidance lane targets a different task"):
        _sample_cfg(runtime, target, prepared, replace(prepared, task=MiniMaxH3Task.REF2VA), 2.0)
    other = empty_minimax_h3_av(
        width=64,
        height=32,
        frame_count=5,
        device="cpu",
        dtype=torch.float32,
    )
    _, other_layout = pack_latent_streams(other)
    with pytest.raises(MiniMaxH3RuntimeError, match="guidance lane belongs"):
        _sample_cfg(runtime, target, prepared, replace(prepared, target_layout=other_layout), 2.0)


def test_sequence_mode_preserves_custom_scheduler(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    prepared = _condition_t2va(runtime_fixture.conditioner_runtime, target)
    builtin = torch_scheduler_registry().get("simple")
    assert builtin is not None
    schedulers: Registry[SchedulerDescriptor] = Registry()
    schedulers.register(replace(builtin, display_name="Custom simple"))
    runtime = MiniMaxH3DiTRuntime(
        runtime_fixture.fl2va,  # type: ignore[arg-type]
        model_role="fl2va_dit",
        runtime_identity="test:h3:fl2va",
        scheduler_registry=schedulers,
    )
    config = DistributedSamplingConfig(
        0,
        2,
        "sequence",
        "file:///group",
        "1" * 32,
        sequence_ulysses=2,
        sequence_ring=1,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )

    def process_group() -> DistributedSamplingConfig:
        raise RuntimeError("process group reached")

    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", process_group
    )

    with pytest.raises(RuntimeError, match="process group reached"):
        runtime.sample_multistream(
            target,
            conditioning=prepared,
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=CancellationFlag(),
        )

    assert runtime_fixture.fl2va.calls == []


@pytest.mark.parametrize(
    ("sequence_guidance", "cfg", "message"),
    (
        (2, SamplingGuidance(None, 2.0), "requires conditional and unconditional lanes"),
        (3, SamplingGuidance(None, 1.0), "guidance degree must be 1 or 2"),
    ),
)
def test_sequence_guidance_fails_closed_before_model_evaluation(
    sequence_guidance: int,
    cfg: SamplingGuidance[object],
    message: str,
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    prepared = _condition_t2va(runtime_fixture.conditioner_runtime, target)
    config = DistributedSamplingConfig(
        0,
        sequence_guidance * 2,
        "sequence",
        "file:///group",
        "1" * 32,
        sequence_ulysses=2,
        sequence_ring=1,
        sequence_guidance=sequence_guidance,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", lambda: config
    )

    with pytest.raises(MiniMaxH3RuntimeError, match=message):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=cfg,
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=CancellationFlag(),
        )

    assert runtime_fixture.fl2va.calls == []


@pytest.mark.parametrize("surface", ("ksampler", "custom"))
def test_sequence_mode_continues_without_receipt_to_execution(
    surface: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    if surface == "ksampler":

        class CudaStream:
            layout = torch.strided

            @staticmethod
            def size() -> torch.Size:
                return torch.Size((1, 1))

            def to(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("CUDA latent transfer reached")

        target = cast(
            "MultiStreamLatent[torch.Tensor]",
            MultiStreamLatent.from_pairs((("video", CudaStream()), ("audio", CudaStream()))),
        )

        def validate_target(_target: object) -> None:
            return None

        monkeypatch.setattr(h3_runtime_module, "_validate_av_target", validate_target)
    config = DistributedSamplingConfig(
        0,
        2,
        "sequence",
        "file:///group",
        "1" * 32,
        sequence_ulysses=2,
        sequence_ring=1,
    )

    def process_group() -> DistributedSamplingConfig:
        raise AssertionError("process group reached")

    def cuda_capability(_device: object) -> tuple[int, int]:
        raise AssertionError("CUDA reached")

    monkeypatch.setattr(h3_runtime_module, "distributed_sampling_config", lambda: config)
    monkeypatch.setattr(h3_runtime_module, "ensure_process_group", process_group)
    monkeypatch.setattr(torch.cuda, "get_device_capability", cuda_capability)

    with pytest.raises(
        AssertionError,
        match="CUDA latent transfer reached|process group reached",
    ):
        if surface == "ksampler":
            runtime.sample_multistream(
                target,
                conditioning=prepared,
                sampler_id="res_multistep",
                scheduler_id="simple",
                steps=1,
                denoise=1.0,
                seed=123,
                cancelled=CancellationFlag(),
            )
        else:
            runtime.sample_custom(
                target,
                noise=target.map(torch.zeros_like),
                cond=PreparedMultiStreamConditioning(runtime.runtime_identity, prepared),
                cfg=None,
                request=_h3_custom_request("euler"),
                cancelled=CancellationFlag(),
            )

    assert dit.calls == []


@pytest.mark.parametrize("mode", ["auto", "guidance", "window"])
def test_distributed_h3_single_lane_reaches_replica_evaluation_without_receipt(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    span_events: list[ExecutionSpanEvent] = []

    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: DistributedSamplingConfig(0, 2, mode, "file:///group", "1" * 32),
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group",
        lambda: DistributedSamplingConfig(0, 2, mode, "file:///group", "1" * 32),
    )

    def evaluate(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("distributed evaluation reached")

    monkeypatch.setattr(
        "dinkster_inference_torch.distributed.DistributedGuidanceEvaluator", evaluate
    )
    with pytest.raises(RuntimeError, match="distributed evaluation reached"):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=CancellationFlag(),
            observer=ExecutionObserverAttachment(span_events.append, "mode-test"),
        )

    assert runtime_fixture.fl2va.calls == []
    assert [(event.operation, event.phase) for event in span_events] == [
        ("prepare", "begin"),
        ("prepare", "end"),
    ]


def test_distributed_h3_guidance_admits_unmeasured_ref2va_model(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    prepared = replace(prepared, task=MiniMaxH3Task.REF2VA)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    config = DistributedSamplingConfig(0, 2, "guidance", "file:///group", "1" * 32)
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", lambda: config
    )

    def evaluate(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("distributed evaluation reached")

    monkeypatch.setattr(
        "dinkster_inference_torch.distributed.DistributedGuidanceEvaluator", evaluate
    )
    with pytest.raises(RuntimeError, match="distributed evaluation reached"):
        runtime_fixture.ref2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(negative, 2.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            cancelled=CancellationFlag(),
        )

    assert runtime_fixture.fl2va.calls == []
    assert runtime_fixture.ref2va.calls == []


@pytest.mark.parametrize("mode", ["auto", "guidance", "window"])
@pytest.mark.parametrize("cfg", [1.0, 2.0])
def test_distributed_h3_guidance_uses_the_generic_lane_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    cfg: float,
) -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
    config = DistributedSamplingConfig(0, 2, mode, "file:///group", "1" * 32)

    class FakeReplicas:
        def __init__(
            self,
            evaluate: Callable[
                [torch.Tensor, float, GuidanceEvaluationRequest[torch.Tensor]],
                GuidancePredictions[torch.Tensor],
            ],
        ) -> None:
            self.evaluate = evaluate

        def evaluate_request(
            self,
            x: torch.Tensor,
            sigma: float,
            request: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            return self.evaluate(x, sigma, request)

    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", lambda: config
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.distributed.DistributedGuidanceEvaluator",
        FakeReplicas,
    )

    states: list[object] = []
    result = runtime.sample_multistream(
        target,
        conditioning=prepared,
        cfg=SamplingGuidance(negative, cfg),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=1.0,
        seed=123,
        cancelled=lambda: False,
        on_state=states.append,
    )

    assert states
    assert dit.calls == ([0.0] if cfg == 1.0 else [0.0, 5.0])
    packed, layout = pack_latent_streams(target)
    expected = prepare_noise(packed, 123) - (5.0 + (0.0 - 5.0) * cfg)
    expected[..., layout.by_role("video").elements :] /= MINIMAX_H3_SIGMAS.audio_scale
    result_packed, _ = pack_latent_streams(result)
    torch.testing.assert_close(result_packed, expected)


def test_distributed_h3_guidance_emits_the_non_distributed_step_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(cancelled: Callable[[], bool]) -> tuple[list[tuple[int, int, float]], torch.Tensor]:
        runtime, conditioner, _dit = _context_mean_runtime()
        target = _target()
        prepared = _condition_t2va(conditioner, target)
        negative = replace(prepared, context=torch.full_like(prepared.context, 5.0))
        events: list[tuple[int, int, float]] = []

        def on_step(event: StepEvent) -> None:
            events.append((event.step, event.total, event.sigma))

        result = runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(negative, 2.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=3,
            denoise=1.0,
            seed=123,
            on_step=on_step,
            cancelled=cancelled,
        )
        packed, _ = pack_latent_streams(result)
        return events, packed

    baseline_events, baseline_result = run(lambda: False)
    assert len(baseline_events) == 3

    class LocalReplicas:
        def __init__(
            self,
            evaluate: Callable[
                [torch.Tensor, float, GuidanceEvaluationRequest[torch.Tensor]],
                GuidancePredictions[torch.Tensor],
            ],
        ) -> None:
            self.evaluate = evaluate

        def evaluate_request(
            self,
            x: torch.Tensor,
            sigma: float,
            request: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            return self.evaluate(x, sigma, request)

    config = DistributedSamplingConfig(0, 2, "guidance", "file:///group", "1" * 32)
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", lambda: config
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.distributed.DistributedGuidanceEvaluator",
        LocalReplicas,
    )

    distributed_events, distributed_result = run(CancellationFlag())

    assert distributed_events == baseline_events
    torch.testing.assert_close(distributed_result, baseline_result)


@pytest.mark.parametrize("model_role", ("fl2va_dit", "ref2va_dit"))
def test_sequence_mode_admits_progress_callbacks_and_both_model_roles(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
    model_role: str,
) -> None:
    target = _target()
    prepared = _condition_t2va(runtime_fixture.conditioner_runtime, target)
    runtime = runtime_fixture.fl2va_runtime
    if model_role == "ref2va_dit":
        prepared = replace(prepared, task=MiniMaxH3Task.REF2VA)
        runtime = runtime_fixture.ref2va_runtime
    config = DistributedSamplingConfig(
        0,
        2,
        "sequence",
        "file:///group",
        "1" * 32,
        sequence_ulysses=2,
        sequence_ring=1,
        sequence_guidance=1,
    )
    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.distributed_sampling_config",
        lambda: config,
    )

    def process_group() -> DistributedSamplingConfig:
        raise RuntimeError("process group reached")

    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_runtime.ensure_process_group", process_group
    )

    with pytest.raises(RuntimeError, match="process group reached"):
        runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            on_step=lambda _event: None,
            on_state=lambda _event: None,
            cancelled=lambda: False,
        )

    assert runtime_fixture.fl2va.calls == []
    assert runtime_fixture.ref2va.calls == []


def test_sample_forwards_model_device_to_the_engine(
    runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    runtime_fixture.fl2va.video_patch_proj.to("meta")

    class NoiseReached(RuntimeError):
        pass

    def capture_engine(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
        noise = kwargs["noise"]
        latent = kwargs["latent"]
        assert isinstance(noise, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert noise.device == torch.device("cpu")
        assert noise.dtype == torch.float32
        assert latent.device == torch.device("cpu")
        assert kwargs["device"] == torch.device("meta")
        raise NoiseReached

    monkeypatch.setattr("dinkster_inference_torch.denoise.run_sampler_engine", capture_engine)
    with pytest.raises(NoiseReached):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="res_multistep",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            on_step=None,
            cancelled=lambda: False,
        )


@pytest.mark.parametrize("noise_index", (0, 2))
def test_ddim_inpaint_noise_is_generated_from_the_packed_latent(
    runtime_fixture: RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
    noise_index: int,
) -> None:
    target = _target()
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )

    class NoiseReached(RuntimeError):
        pass

    def capture_engine(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
        actual = kwargs["inpaint_noise"]
        assert isinstance(actual, torch.Tensor)
        scaled = _h3(
            target.by_role("video"),
            target.by_role("audio") * MINIMAX_H3_SIGMAS.audio_scale,
        )
        packed, _ = pack_latent_streams(scaled)
        expected = prepare_noise(packed, 124, (noise_index,))
        assert torch.equal(actual, expected)
        raise NoiseReached

    monkeypatch.setattr("dinkster_inference_torch.denoise.run_sampler_engine", capture_engine)
    with pytest.raises(NoiseReached):
        runtime_fixture.fl2va_runtime.sample_multistream(
            target,
            conditioning=prepared,
            cfg=SamplingGuidance(None, 1.0),
            sampler_id="ddim",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
            denoise_mask=torch.ones_like(target.by_role("video")),
            noise_inds=(noise_index,),
            segment=SamplingSegment(1, 0, 1, False, False),
            cancelled=lambda: False,
        )


def test_partial_denoise_uses_flow_noise_scaling(
    runtime_fixture: RuntimeFixture,
) -> None:
    target = _target()
    target = _h3(target.by_role("video") + 4.0, target.by_role("audio") + 4.0)
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3T2VARequest(""),
        target=target,
        frame_count=5,
        payloads={},
        cancelled=lambda: False,
    )
    result = runtime_fixture.fl2va_runtime.sample_multistream(
        target,
        conditioning=prepared,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=1,
        denoise=0.5,
        seed=123,
        on_step=None,
        cancelled=lambda: False,
    )
    packed, layout = pack_latent_streams(target)
    sigma = MINIMAX_H3_SIGMAS.percent_to_sigma(0.5)
    video_expected = prepare_noise(packed, 123) * sigma + packed * (1.0 - sigma) - 2.0 * sigma
    audio_packed = packed.clone()
    audio_packed[..., layout.by_role("video").elements :] *= MINIMAX_H3_SIGMAS.audio_scale
    audio_expected = (
        prepare_noise(audio_packed, 123) * sigma + audio_packed * (1.0 - sigma) - 2.0 * sigma
    )
    expected = video_expected
    expected[..., layout.by_role("video").elements :] = (
        audio_expected[..., layout.by_role("video").elements :] / MINIMAX_H3_SIGMAS.audio_scale
    )
    result_packed, result_layout = pack_latent_streams(result)
    assert result_layout == layout
    torch.testing.assert_close(result_packed, expected)


def test_fl2va_payload_is_realized_for_conditioner_and_dit_once(
    runtime_fixture: RuntimeFixture,
) -> None:
    image = torch.zeros((1, 32, 32, 3))
    descriptor = PayloadDescriptor(
        PayloadReference("first"), tuple(image.shape), "float32", "worker:minimax-h3"
    )
    request = MiniMaxH3FL2VARequest(
        "move",
        (MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, descriptor),),
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        request,
        target=_target(),
        frame_count=5,
        payloads={"first": image},
        cancelled=lambda: False,
    )
    assert prepared.task.value == "fl2va"
    assert prepared.dit.keyframes[0].resolved_frame_index == 0
    assert prepared.dit.keyframes[0].video.shape == (1, 24, 1, 2, 2)
    assert prepared.token_layout is not None
    assert "keyframe-first" in tuple(
        segment.identity for segment in prepared.token_layout.layout.segments
    )
    assert runtime_fixture.video_vae.encoded[0].shape == (1, 3, 1, 32, 32)
    with pytest.raises(ValueError, match="key set is not exact"):
        runtime_fixture.conditioner_runtime.condition(
            request,
            target=_target(),
            frame_count=5,
            payloads={"first": image, "foreign": image},
            cancelled=lambda: False,
        )


def test_ref2va_audio_payload_preserves_explicit_sample_rate(
    runtime_fixture: RuntimeFixture,
) -> None:
    waveform = torch.zeros((1, 2, 800))
    descriptor = PayloadDescriptor(
        PayloadReference("audio"), tuple(waveform.shape), "float32", "worker:minimax-h3"
    )
    prepared = runtime_fixture.conditioner_runtime.condition(
        MiniMaxH3REF2VARequest("continue", (MiniMaxH3AudioReference(descriptor, 32_000),)),
        target=_target(),
        frame_count=5,
        payloads={"audio": waveform},
        cancelled=lambda: False,
    )
    assert prepared.dit.references[0].audio is not None
    assert prepared.token_layout is not None
    assert "reference-1-audio" in tuple(
        segment.identity for segment in prepared.token_layout.layout.segments
    )


def test_ref2va_encodes_every_full_rate_video_frame(
    runtime_fixture: RuntimeFixture,
) -> None:
    frames = tuple(torch.full((1, 32, 32, 3), float(index)) for index in range(13))
    descriptors = tuple(
        PayloadDescriptor(
            PayloadReference(f"frame-{index}"),
            tuple(frame.shape),
            "float32",
            "worker:minimax-h3",
        )
        for index, frame in enumerate(frames)
    )
    request = MiniMaxH3REF2VARequest(
        "continue",
        (MiniMaxH3VideoReference(descriptors, (0, 12), (0.0, 0.5)),),
    )

    runtime_fixture.conditioner_runtime.condition(
        request,
        target=_target(),
        frame_count=5,
        payloads={f"frame-{index}": frame for index, frame in enumerate(frames)},
        cancelled=lambda: False,
    )

    encoded = runtime_fixture.video_vae.encoded[0]
    assert encoded.shape == (1, 3, 13, 32, 32)
    assert encoded[0, 0, :, 0, 0].tolist() == [float(2 * index - 1) for index in range(13)]


@pytest.mark.parametrize(
    ("descriptor", "payloads", "message"),
    (
        (
            PayloadDescriptor(
                PayloadReference("audio"),
                (1, 2, 800),
                "float32",
                "foreign-space",
            ),
            {"audio": torch.zeros((1, 2, 800))},
            "payload space",
        ),
        (
            PayloadDescriptor(
                PayloadReference("audio"),
                (1, 2, 800),
                "float32",
                "worker:minimax-h3",
            ),
            {},
            "key set is not exact",
        ),
        (
            PayloadDescriptor(
                PayloadReference("audio"),
                (1, 2, 800),
                "float32",
                "worker:minimax-h3",
            ),
            {"audio": torch.zeros((1, 2, 801))},
            "shape differs",
        ),
    ),
)
def test_ref2va_payload_authority_rejects_invalid_descriptors_or_maps(
    runtime_fixture: RuntimeFixture,
    descriptor: PayloadDescriptor,
    payloads: dict[str, torch.Tensor],
    message: str,
) -> None:
    request = MiniMaxH3REF2VARequest("continue", (MiniMaxH3AudioReference(descriptor, 32_000),))
    with pytest.raises(ValueError, match=message):
        runtime_fixture.conditioner_runtime.condition(
            request,
            target=_target(),
            frame_count=5,
            payloads=payloads,
            cancelled=lambda: False,
        )


def test_malformed_audio_refuses_before_conditioner_or_codec_work(
    runtime_fixture: RuntimeFixture,
) -> None:
    waveform = torch.zeros((1, 2, 800), dtype=torch.int64)
    descriptor = PayloadDescriptor(
        PayloadReference("audio"), tuple(waveform.shape), "int64", "worker:minimax-h3"
    )
    request = MiniMaxH3REF2VARequest("continue", (MiniMaxH3AudioReference(descriptor, 32_000),))
    with pytest.raises(TypeError, match="strided floating"):
        runtime_fixture.conditioner_runtime.condition(
            request,
            target=_target(),
            frame_count=5,
            payloads={"audio": waveform},
            cancelled=lambda: False,
        )
    assert runtime_fixture.conditioner.calls == 0
    assert runtime_fixture.video_vae.encoded == []
    assert runtime_fixture.audio_vae.encoded == []


def test_malformed_target_refuses_before_conditioner_or_codec_work(
    runtime_fixture: RuntimeFixture,
) -> None:
    valid = _target()
    target = _h3(valid.by_role("video").to(torch.int64), valid.by_role("audio").to(torch.int64))
    with pytest.raises(TypeError, match="floating tensors"):
        runtime_fixture.conditioner_runtime.condition(
            MiniMaxH3T2VARequest(""),
            target=target,
            frame_count=5,
            payloads={},
            cancelled=lambda: False,
        )
    assert runtime_fixture.conditioner.calls == 0
    assert runtime_fixture.video_vae.encoded == []
    assert runtime_fixture.audio_vae.encoded == []


def test_encode_video_maps_comfy_image_range_to_encoder_input_range(
    runtime_fixture: RuntimeFixture,
) -> None:
    content = torch.full((1, 3, 1, 16, 16), 0.25, dtype=torch.float32)
    runtime_fixture.video_runtime.encode_video(content)
    received = runtime_fixture.video_vae.encoded[-1]
    assert received.dtype is torch.float16
    torch.testing.assert_close(received, torch.full_like(received, -0.5))
    assert content.unique().tolist() == [0.25]


def test_codec_boundaries_and_cancellation(runtime_fixture: RuntimeFixture) -> None:
    video = torch.ones((1, 3, 5, 16, 16), dtype=torch.float32)
    video_latent = runtime_fixture.video_runtime.encode_video(video)
    assert runtime_fixture.video_vae.encoded[-1].dtype is torch.float16
    assert video_latent.dtype is torch.float16
    assert runtime_fixture.video_runtime.decode_video(video_latent.float()).dtype is torch.float16

    waveform = torch.ones((1, 2, 800), dtype=torch.float16)
    latent = runtime_fixture.audio_runtime.encode_audio(MiniMaxH3AudioContent(waveform, 32_000))
    assert runtime_fixture.audio_vae.encoded[-1].dtype is torch.float32
    assert latent.dtype is torch.float32
    decoded = runtime_fixture.audio_runtime.decode_audio(latent.half())
    assert decoded.sample_rate == 32_000
    assert decoded.waveform.shape == (1, 2, 1)
    assert decoded.waveform.dtype is torch.float32
    with pytest.raises(SamplingCancelled):
        runtime_fixture.conditioner_runtime.condition(
            MiniMaxH3T2VARequest("stop"),
            target=_target(),
            frame_count=5,
            payloads={},
            cancelled=lambda: True,
        )


def _h3_custom_request(
    sampler_id: str,
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


@pytest.mark.parametrize(
    ("sampler_id", "scheduler_id", "steps", "cfg_scale", "with_mask", "segment"),
    [
        ("euler", "simple", 2, None, False, None),
        ("res_multistep", "simple", 2, 2.0, False, None),
        ("dpmpp_sde", "simple", 3, None, False, None),
        ("res_multistep", "simple", 2, 2.0, True, None),
        (
            "euler",
            "simple",
            3,
            None,
            False,
            SamplingSegment(
                steps=3,
                start_step=1,
                end_step=3,
                add_noise=False,
                return_with_leftover_noise=False,
            ),
        ),
    ],
    ids=("euler", "cfg", "brownian", "mask", "segment"),
)
def test_ksampler_surface_is_bit_equal_sugar_over_sample_custom(
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    cfg_scale: float | None,
    with_mask: bool,
    segment: SamplingSegment | None,
) -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    negative = (
        None
        if cfg_scale is None
        else replace(prepared, context=torch.full_like(prepared.context, 5.0))
    )
    mask = None
    if with_mask:
        video_mask = torch.ones_like(target.by_role("video"))
        video_mask[..., 0, 0] = 0.2
        audio_mask = torch.ones_like(target.by_role("audio"))
        audio_mask[:, :, 0, 0] = 0.3
        mask = _h3(video_mask, audio_mask)

    expected = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
        denoise_mask=mask,
        cancelled=CancellationFlag(),
    )
    identity = runtime.runtime_identity
    result = run_ksampler_as_custom(
        runtime,
        target,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=MINIMAX_H3_SIGMAS.video,
        flow=True,
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(identity, positive),
        cfg=SamplingGuidance(
            None if negative is None else PreparedMultiStreamConditioning(identity, negative),
            1.0 if cfg_scale is None else cfg_scale,
        ),
        segment=segment,
        denoise_mask=mask,
        error=MiniMaxH3RuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_for_low_precision_latents() -> None:
    """Seeded noise is drawn against float32 views on both surfaces,
    so half-precision latent streams must not round the draw on the
    decomposed path."""
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target().map(lambda stream: stream.to(dtype=torch.float16))
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))

    expected = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="euler",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cancelled=CancellationFlag(),
    )
    result = run_ksampler_as_custom(
        runtime,
        target,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=MINIMAX_H3_SIGMAS.video,
        flow=True,
        sampler_id="euler",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(runtime.runtime_identity, positive),
        cfg=SamplingGuidance(None, 1.0),
        error=MiniMaxH3RuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))


def test_ksampler_sugar_parity_holds_for_cfg_pp_without_negative() -> None:
    """CFG++ samplers consume the guidance scale through a
    synthetic-zero unconditional lane even with no negative payload,
    so an empty negative must keep the scale on both surfaces."""
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    guidance = SamplingGuidance(None, 3.0)

    expected = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=guidance,
        sampler_id="euler_cfg_pp",
        scheduler_id="simple",
        steps=3,
        denoise=0.75,
        seed=123,
        cancelled=CancellationFlag(),
    )
    result = run_ksampler_as_custom(
        runtime,
        target,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=MINIMAX_H3_SIGMAS.video,
        flow=True,
        sampler_id="euler_cfg_pp",
        scheduler_id="simple",
        steps=3,
        denoise=0.75,
        seed=123,
        cond=PreparedMultiStreamConditioning(runtime.runtime_identity, positive),
        cfg=guidance,
        error=MiniMaxH3RuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))
    unguided = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=None,
        sampler_id="euler_cfg_pp",
        scheduler_id="simple",
        steps=3,
        denoise=0.75,
        seed=123,
        cancelled=CancellationFlag(),
    )
    assert not torch.equal(output.by_role("video"), unguided.by_role("video"))


def test_custom_sampling_sigma_methods_match_the_schedule_helpers() -> None:
    runtime, _conditioner, _dit = _context_mean_runtime()
    assert runtime.family is MINIMAX_H3
    scheduler = torch_scheduler_registry().get("simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("simple", 4, 1.0) == sampling_sigmas(
        scheduler, MINIMAX_H3_SIGMAS.video, 4, denoise=1.0
    )
    with pytest.raises(MiniMaxH3RuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("test.missing", 4, 1.0)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        MINIMAX_H3_SIGMAS.video, 4, 0.6, 0.6
    )
    with pytest.raises(ValueError, match="discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(2, 1.0)
    assert runtime.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=False
    ) == custom_percent_to_sigma(
        MINIMAX_H3_SIGMAS.video,
        MINIMAX_H3_SIGMAS.video.percent_to_sigma,
        0.5,
        return_actual_sigma=False,
    )


def test_check_custom_sampling_refuses_inpaint_guidance_and_unknown_samplers() -> None:
    runtime, _conditioner, _dit = _context_mean_runtime()
    request = _h3_custom_request("euler")
    runtime.check_custom_sampling(
        request, has_denoise_mask=True, has_inpaint=False, has_context_windows=False
    )
    with pytest.raises(MiniMaxH3RuntimeError, match="does not support inpaint"):
        runtime.check_custom_sampling(
            request, has_denoise_mask=False, has_inpaint=True, has_context_windows=False
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="does not support context windows"):
        runtime.check_custom_sampling(
            request, has_denoise_mask=False, has_inpaint=False, has_context_windows=True
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="no distilled-guidance input"):
        runtime.check_custom_sampling(
            request,
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
            guidance=3.5,
        )
    descriptor = torch_sampler_registry().get("euler")
    assert descriptor is not None
    unknown = CustomSamplingRequest(replace(descriptor, id="test.missing"), (), (1.0, 0.0))
    with pytest.raises(MiniMaxH3RuntimeError, match="unknown sampler"):
        runtime.check_custom_sampling(
            unknown, has_denoise_mask=False, has_inpaint=False, has_context_windows=False
        )


def test_sample_custom_refuses_malformed_latents_noise_and_conditioning() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    cond = PreparedMultiStreamConditioning("test:h3:fl2va", prepared)
    noise = target.map(torch.zeros_like)
    request = _h3_custom_request("euler")
    packed, _layout = pack_latent_streams(target)

    with pytest.raises(MiniMaxH3RuntimeError, match="requires a MultiStreamLatent latent"):
        runtime.sample_custom(packed, noise=noise, cond=cond, cfg=None, request=request)
    with pytest.raises(MiniMaxH3RuntimeError, match="requires MultiStreamLatent noise"):
        runtime.sample_custom(
            target, noise=cast("Any", packed), cond=cond, cfg=None, request=request
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="dual CFG"):
        runtime.sample_custom(
            target,
            noise=noise,
            cond=cond,
            cfg=DualSamplingGuidance(cond, cond, 2.0, 1.0),
            request=request,
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="requires prepared multi-stream conditioning"):
        runtime.sample_custom(
            target, noise=noise, cond=cast("Any", prepared), cfg=None, request=request
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="different conditioner component"):
        runtime.sample_custom(
            target,
            noise=noise,
            cond=PreparedMultiStreamConditioning("test:h3:other", prepared),
            cfg=None,
            request=request,
        )
    with pytest.raises(TypeError, match="exact MiniMaxH3PreparedConditioning"):
        runtime.sample_custom(
            target,
            noise=noise,
            cond=PreparedMultiStreamConditioning("test:h3:fl2va", object()),
            cfg=None,
            request=request,
        )
    with pytest.raises(
        MiniMaxH3RuntimeError, match="guidance requires prepared multi-stream conditioning"
    ):
        runtime.sample_custom(
            target,
            noise=noise,
            cond=cond,
            cfg=SamplingGuidance(cast("Any", prepared), 2.0),
            request=request,
        )
    with pytest.raises(MiniMaxH3RuntimeError, match="different conditioner components"):
        runtime.sample_custom(
            target,
            noise=noise,
            cond=cond,
            cfg=SamplingGuidance(PreparedMultiStreamConditioning("test:h3:other", prepared), 2.0),
            request=request,
        )
    wrong_noise = _h3(
        torch.zeros_like(target.by_role("video"))[..., :-1],
        torch.zeros_like(target.by_role("audio")),
    )
    with pytest.raises(MiniMaxH3RuntimeError, match="noise topology differs"):
        runtime.sample_custom(target, noise=wrong_noise, cond=cond, cfg=None, request=request)


def test_sample_custom_captures_the_last_denoised_state_with_unscaled_audio() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))
    cond = PreparedMultiStreamConditioning("test:h3:fl2va", positive)
    request = _h3_custom_request("res_multistep", runtime.custom_sampling_sigmas("simple", 2, 1.0))
    noise = target.map(torch.zeros_like)
    events: list[SamplingStateEvent[object]] = []

    result = runtime.sample_custom(
        target,
        noise=noise,
        cond=cond,
        cfg=None,
        request=request,
        seed=7,
        on_state=events.append,
    )

    assert type(result) is CustomSamplingResult
    denoised = result.denoised_output
    assert denoised is not None
    captured = [event.denoised for event in events if type(event.denoised) is MultiStreamLatent]
    assert captured
    last = cast("MultiStreamLatent[torch.Tensor]", captured[-1])
    assert torch.equal(denoised.by_role("video"), last.by_role("video"))
    assert torch.equal(
        denoised.by_role("audio"),
        last.by_role("audio") / MINIMAX_H3_SIGMAS.audio_scale,
    )


def test_ksampler_sugar_matches_custom_state_capture_for_uni_pc() -> None:
    runtime, conditioner, dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    positive = replace(prepared, context=torch.full_like(prepared.context, 2.0))

    expected = runtime.sample_multistream(
        target,
        conditioning=positive,
        cfg=None,
        sampler_id="uni_pc",
        scheduler_id="simple",
        steps=3,
        denoise=1.0,
        seed=185,
        cancelled=CancellationFlag(),
    )
    sugar_calls = len(dit.calls)
    assert sugar_calls > 0
    dit.calls.clear()

    result = run_ksampler_as_custom(
        runtime,
        target,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=MINIMAX_H3_SIGMAS.video,
        flow=True,
        sampler_id="uni_pc",
        scheduler_id="simple",
        steps=3,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(runtime.runtime_identity, positive),
        cfg=None,
        error=MiniMaxH3RuntimeError,
    )
    custom_calls = len(dit.calls)

    assert sugar_calls == custom_calls
    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))
    assert torch.equal(output.by_role("audio"), expected.by_role("audio"))
    assert result.denoised_output is not None


def test_sample_custom_returns_the_input_latent_for_an_empty_schedule() -> None:
    runtime, conditioner, _dit = _context_mean_runtime()
    target = _target()
    prepared = _condition_t2va(conditioner, target)
    cond = PreparedMultiStreamConditioning("test:h3:fl2va", prepared)
    result = runtime.sample_custom(
        target,
        noise=target.map(torch.zeros_like),
        cond=cond,
        cfg=None,
        request=_h3_custom_request("euler", ()),
    )
    assert result.output is target
    assert result.denoised_output is None
