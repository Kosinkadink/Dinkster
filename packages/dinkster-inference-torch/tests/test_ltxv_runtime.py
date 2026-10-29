"""LTX-Video 2B runtime contracts over the shared sampling engine."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    LTX_SIGMAS,
    LTXV,
    LTXV_2B_V09_CONFIG,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ContextFuseMethod,
    ContextWindowSchedule,
    ContextWindowsRuntime,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    FluxFlowSigmas,
    LTXVConfig,
    MultiStreamConditioningRuntime,
    MultiStreamLatent,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    SamplingGuidance,
    SamplingSegment,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
    plan_windows,
    sampling_sigmas,
)
from dinkster_inference_torch import LazyCacheConfig, LTXVModel, ltxv_runtime
from dinkster_inference_torch._conditioning_layout import (
    declare_text_conditioning,
)
from dinkster_inference_torch.context_windows import (
    apply_freenoise,
    windowed_conditioning_evaluation,
)
from dinkster_inference_torch.denoise import prepare_multistream_noise
from dinkster_inference_torch.guidance import ConditioningEvaluation
from dinkster_inference_torch.ltx_component import LTXVTextRuntime
from dinkster_inference_torch.ltx_media import LTXVGuideConditioning
from dinkster_inference_torch.ltxv_runtime import (
    LTXVDiffusionRuntime,
    LTXVPreparedConditioning,
    LTXVRuntimeError,
    LTXVVideoCodecRuntime,
    _crop_spatial_to_multiple,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.payloads import payload_binding_to_tensor, tensor_to_payload_binding
from dinkster_inference_torch.sampling_execution import (
    build_sampling_schedule,
    run_ksampler_as_custom,
)
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

FUSE_CFG_LANES = ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 2)


def test_ltxv_diffusion_runtime_has_no_text_storage() -> None:
    assert LTXVDiffusionRuntime.retained_offload_storage_components == frozenset()


class _Diffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=4,
            cross_attention_dim=16,
            attention_head_dim=8,
            num_attention_heads=2,
            caption_channels=8,
            num_layers=1,
            causal_temporal_positioning=False,
        )
        self.patchify_proj = torch.nn.Linear(4, 4, bias=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        self.denoise_masks: list[torch.Tensor | None] = []
        self.guides: list[object] = []

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        frame_rate: float,
        denoise_mask: torch.Tensor | None = None,
        guides: object = (),
    ) -> torch.Tensor:
        self.denoise_masks.append(None if denoise_mask is None else denoise_mask.detach().clone())
        self.guides.append(guides)
        self.calls.append(
            (
                latent.detach().clone(),
                timesteps.detach().clone(),
                context.detach().clone(),
                attention_mask.detach().clone(),
                frame_rate,
            )
        )
        velocity = context.mean(dim=(1, 2), keepdim=True).reshape(-1, 1, 1, 1, 1)
        return torch.ones_like(latent) * velocity


class _Tokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def tokenize(self, text: str) -> tuple[str, ...]:
        self.texts.append(text)
        return (text,)


class _TextEncoder:
    def __init__(self) -> None:
        self.spans: list[object] = []

    def encode(self, spans: object) -> Conditioning[torch.Tensor]:
        self.spans.append(spans)
        return declare_text_conditioning(Conditioning(torch.ones((1, 4, 8)), None), 3)


def _runtime(
    diffusion: _Diffusion | None = None, *, dtype: torch.dtype = torch.float32
) -> tuple[LTXVDiffusionRuntime, _Diffusion]:
    diffusion = _Diffusion() if diffusion is None else diffusion

    def compute_dtype(_component: object) -> torch.dtype:
        return dtype

    assembled = SimpleNamespace(
        family=LTXV,
        diffusion=diffusion,
        vae=None,
        compute_dtype=compute_dtype,
    )
    runtime = object.__new__(LTXVDiffusionRuntime)
    raw = cast("Any", runtime)
    raw._assembled = assembled
    raw._runtime_identity = "test:ltxv"
    raw._samplers = torch_sampler_registry()
    raw._schedulers = torch_scheduler_registry()
    return runtime, diffusion


def _video(shape: tuple[int, ...] = (1, 4, 1, 2, 2)) -> torch.Tensor:
    return torch.zeros(shape)


def _prepared(tokens: int = 3, frame_rate: float = 25.0, rows: int = 4) -> LTXVPreparedConditioning:
    return LTXVPreparedConditioning(torch.ones((1, rows, 8)), tokens, frame_rate)


def _sample(
    runtime: LTXVDiffusionRuntime,
    latent: object,
    conditioning: object,
    **kwargs: object,
) -> torch.Tensor:
    streams = (
        latent
        if type(latent) is MultiStreamLatent
        else MultiStreamLatent.from_pairs((("video", cast("Any", latent)),))
    )
    result = runtime.sample_multistream(
        cast("Any", streams),
        conditioning=conditioning,
        sampler_id=cast("str", kwargs.pop("sampler_id", "euler")),
        scheduler_id=cast("str", kwargs.pop("scheduler_id", "simple")),
        steps=cast("int", kwargs.pop("steps", 1)),
        denoise=cast("float", kwargs.pop("denoise", 1.0)),
        seed=cast("int", kwargs.pop("seed", 123)),
        **cast("Any", kwargs),
    )
    return result.by_role("video")


def test_encode_text_uses_ltxv_owned_tokenizer_and_encoder() -> None:
    runtime = object.__new__(LTXVTextRuntime)
    raw = cast("Any", runtime)
    raw._tokenizer = _Tokenizer()
    raw._encoder = _TextEncoder()

    result = runtime.encode_text("a fox running")

    assert raw._tokenizer.texts == ["a fox running"]
    assert raw._encoder.spans == [("a fox running",)]
    assert result.pooled is None
    assert result.embeddings.shape == (1, 4, 8)


def test_carrier_round_trip_preserves_the_declared_token_count() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(token_count=3)
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))
    prepared = runtime.prepare_conditioning(carrier, frame_rate=30.0)

    assert prepared.attention_tokens == 3
    assert prepared.frame_rate == 30.0
    assert prepared.text.shape == (1, 4, 8)
    assert torch.equal(prepared.text, payload_binding_to_tensor(binding))


def test_undeclared_conditioning_attends_to_every_payload_row() -> None:
    runtime, _ = _runtime()

    record, binding = _record_and_binding(token_count=4)
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))
    prepared = runtime.prepare_conditioning(carrier)

    assert prepared.attention_tokens == 4
    assert prepared.frame_rate == 25.0


def test_prepared_conditioning_refuses_non_tensor_and_integer_text() -> None:
    with pytest.raises(TypeError, match="strided floating"):
        LTXVPreparedConditioning(cast("Any", "text"), 1)
    with pytest.raises(TypeError, match="strided floating"):
        LTXVPreparedConditioning(torch.ones((1, 4, 8), dtype=torch.int64), 1)


@pytest.mark.parametrize("shape", ((4, 8), (0, 4, 8), (1, 0, 8)))
def test_prepared_conditioning_refuses_degenerate_text_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(LTXVRuntimeError, match="nonempty rank-3"):
        LTXVPreparedConditioning(torch.ones(shape), 1)


@pytest.mark.parametrize("tokens", (0, 5, True, 2.0))
def test_prepared_conditioning_refuses_out_of_range_token_counts(tokens: object) -> None:
    with pytest.raises(LTXVRuntimeError, match="attention token count"):
        LTXVPreparedConditioning(torch.ones((1, 4, 8)), cast("Any", tokens))


@pytest.mark.parametrize("frame_rate", (25, True, 0.0, -24.0, math.inf, math.nan))
def test_prepared_conditioning_refuses_non_float_frame_rates(frame_rate: object) -> None:
    with pytest.raises(LTXVRuntimeError, match="frame rate"):
        LTXVPreparedConditioning(torch.ones((1, 4, 8)), 2, cast("Any", frame_rate))


def test_sampling_masks_the_padded_tail_and_forwards_the_frame_rate() -> None:
    runtime, diffusion = _runtime()

    _sample(runtime, _video(), _prepared(3, 30.0))

    assert len(diffusion.calls) == 1
    latent, timesteps, context, mask, frame_rate = diffusion.calls[0]
    assert torch.equal(mask, torch.tensor([[1, 1, 1, 0]], dtype=torch.long))
    assert frame_rate == 30.0
    assert timesteps.dtype == torch.float32
    assert timesteps.shape == (1,)
    assert context.dtype == torch.float32
    assert context.shape == (1, 4, 8)
    assert latent.shape == (1, 4, 1, 2, 2)


def test_cfg_lanes_batch_into_one_model_call_with_per_lane_masks() -> None:
    runtime, diffusion = _runtime()
    cfg = SamplingGuidance(cast("Any", _prepared(1)), 2.0, batching=FUSE_CFG_LANES)

    _sample(runtime, _video(), _prepared(3), cfg=cfg)

    assert len(diffusion.calls) == 1
    latent, _, context, mask, _ = diffusion.calls[0]
    assert latent.shape[0] == 2
    assert context.shape == (2, 4, 8)
    assert mask.shape == (2, 4)
    assert sorted(mask.sum(dim=1).tolist()) == [1, 3]


def test_guidance_lanes_must_share_one_frame_rate() -> None:
    runtime, _ = _runtime()
    cfg = SamplingGuidance(cast("Any", _prepared(1, 24.0)), 2.0)

    with pytest.raises(LTXVRuntimeError, match="share one frame rate"):
        _sample(runtime, _video(), _prepared(3, 25.0), cfg=cfg)


def test_sampling_rejects_raw_conditioning_lanes() -> None:
    runtime, _ = _runtime()
    prepared = _prepared()

    with pytest.raises(TypeError, match="exact LTXVPreparedConditioning"):
        runtime.sample_multistream(
            MultiStreamLatent.from_pairs((("video", _video()),)),
            conditioning=cast("Any", Conditioning(torch.ones((1, 4, 8)), None)),
            sampler_id="euler",
            scheduler_id="simple",
            steps=1,
            denoise=1.0,
            seed=123,
        )
    with pytest.raises(TypeError, match="guidance lanes require exact LTXVPreparedConditioning"):
        _sample(
            runtime,
            _video(),
            prepared,
            cfg=SamplingGuidance(cast("Any", Conditioning(torch.ones((1, 4, 8)), None)), 2.0),
        )


@pytest.mark.parametrize("shift", (-1.0, 0.0, 3, True, math.inf, math.nan))
def test_sampling_shift_must_be_a_positive_finite_float(shift: object) -> None:
    runtime, _ = _runtime()

    with pytest.raises(LTXVRuntimeError, match="sampling_shift"):
        _sample(runtime, _video(), _prepared(), sampling_shift=shift)


def test_explicit_sampling_shift_runs_the_engine() -> None:
    runtime, diffusion = _runtime()

    result = _sample(runtime, _video(), _prepared(), sampling_shift=3.0)

    assert result.shape == (1, 4, 1, 2, 2)
    assert len(diffusion.calls) == 1


def test_sampler_denoise_mask_is_forwarded_and_noise_indices_are_refused() -> None:
    runtime, diffusion = _runtime(dtype=torch.bfloat16)

    mask = torch.ones((1, 1, 1, 2, 2))
    mask[..., 0, 0] = 0.0
    _sample(runtime, _video(), _prepared(), denoise_mask=mask)
    assert len(diffusion.denoise_masks) == 1
    observed_mask = diffusion.denoise_masks[0]
    assert observed_mask is not None
    assert torch.equal(observed_mask, mask)
    assert observed_mask.dtype == torch.bfloat16
    assert diffusion.calls[0][1].dtype == torch.float32
    assert torch.count_nonzero(diffusion.calls[0][0][..., 0, 0]) == 0
    with pytest.raises(LTXVRuntimeError, match="noise indices"):
        _sample(runtime, _video(), _prepared(), noise_inds=(0,))


def test_guides_and_mask_timesteps_reach_the_batched_model() -> None:
    runtime, diffusion = _runtime()
    keyframes = torch.arange(24, dtype=torch.int64).reshape(1, 3, 4, 2)
    guide = LTXVGuideConditioning(keyframes, (1, 2, 2), 0.75)
    prepared = LTXVPreparedConditioning(
        torch.ones((1, 4, 8)),
        3,
        25.0,
        (guide,),
    )
    latent = torch.zeros((2, 4, 2, 2, 2))
    mask = torch.ones((2, 1, 2, 2, 2))
    mask[:, :, 1] = 0.0

    _sample(
        runtime,
        latent,
        prepared,
        cfg=SamplingGuidance(cast("Any", prepared), 2.0, batching=FUSE_CFG_LANES),
        denoise_mask=mask,
    )

    assert len(diffusion.calls) == 1
    assert diffusion.calls[0][0].shape[0] == 4
    timesteps = diffusion.calls[0][1]
    assert timesteps.shape == (4, 8)
    assert bool((timesteps[:, :4] > 0.0).all())
    assert torch.count_nonzero(timesteps[:, 4:]) == 0
    observed_mask = diffusion.denoise_masks[0]
    assert observed_mask is not None
    assert observed_mask.shape == (4, 1, 2, 2, 2)
    (observed_guide,) = cast("tuple[LTXVGuideConditioning, ...]", diffusion.guides[0])
    assert observed_guide.keyframe_indices.shape == (4, 3, 4, 2)
    assert all(torch.equal(row, keyframes[0]) for row in observed_guide.keyframe_indices)


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
    with pytest.raises(LTXVRuntimeError, match="stream role 'video'"):
        _sample(
            runtime,
            MultiStreamLatent.from_pairs((("video", _video()), ("extra", _video()))),
            prepared,
        )
    with pytest.raises(LTXVRuntimeError, match=r"\[B,4,T,H,W\]"):
        _sample(runtime, _video((1, 5, 1, 2, 2)), prepared)
    with pytest.raises(LTXVRuntimeError, match=r"\[B,4,T,H,W\]"):
        _sample(
            runtime, MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, 2, 2))),)), prepared
        )
    with pytest.raises(TypeError, match="strided floating"):
        _sample(runtime, torch.zeros((1, 4, 1, 2, 2), dtype=torch.int64), prepared)


def _record_and_binding(
    text: torch.Tensor | None = None,
    *,
    token_count: int | None = 3,
    family_id: str = "dinkster.ltxv",
    streams: tuple[str, ...] = ("t5xxl",),
    segment_name: str = "t5xxl",
    space: str = "conditioning-text",
) -> tuple[ConditioningRecord, Any]:
    payload = torch.ones((1, 4, 8)) if text is None else text
    binding = tensor_to_payload_binding("ltxv-text", payload, space=space)
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

    with pytest.raises(LTXVRuntimeError, match="exactly one conditioning record"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_channels_not_consumed_by_ltxv() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    channels = (
        (ConditioningChannel.CONCAT_LATENT, dict(record.channels)[ConditioningChannel.TEXT]),
    )
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, channels=channels),)), (binding,)
    )

    with pytest.raises(LTXVRuntimeError, match="does not consume.*concat_latent"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_wrong_text_width() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(torch.ones((1, 4, 9)))
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXVRuntimeError, match=r"\[B,tokens,8\]"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_partial_schedules() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, schedule=PercentRange(0.0, 0.5)),)), (binding,)
    )

    with pytest.raises(LTXVRuntimeError, match="schedules"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_requires_a_token_layout() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding()
    carrier = make_conditioning_carrier(
        ConditioningSet((replace(record, token_layout=None),)), (binding,)
    )

    with pytest.raises(LTXVRuntimeError, match="requires a token layout"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_refuses_foreign_family_layouts() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(family_id="dinkster.wan21")
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXVRuntimeError, match="unsupported"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_requires_the_exact_t5xxl_layout() -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(segment_name="other")
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXVRuntimeError, match="exact T5-XXL token layout"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize("token_count", (None, 5))
def test_prepare_conditioning_bounds_the_segment_token_count(token_count: int | None) -> None:
    runtime, _ = _runtime()
    record, binding = _record_and_binding(token_count=token_count)
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))

    with pytest.raises(LTXVRuntimeError, match="1 through the TEXT payload rows"):
        runtime.prepare_conditioning(carrier)


def test_prepare_conditioning_requires_matching_payload_bindings() -> None:
    runtime, _ = _runtime()
    record, _ = _record_and_binding()
    stray = tensor_to_payload_binding("unrelated", torch.ones((1, 4, 8)), space="conditioning-text")

    with pytest.raises((LTXVRuntimeError, ValueError)):
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
        "dinkster.ltxv.conditioning:v1:dinkster.ltxv:4:16:8:2:8:1:False"
    )


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


def _codec_runtime() -> tuple[LTXVVideoCodecRuntime, _VAE]:
    vae = _VAE()
    return LTXVVideoCodecRuntime(cast("Any", vae), compute_dtype=torch.float32), vae


def test_init_requires_an_exact_ltxv_model_and_nonempty_identity() -> None:
    with pytest.raises(ValueError, match="exact supported 2B model"):
        LTXVDiffusionRuntime(
            cast("Any", _Diffusion()),
            runtime_identity="test:ltxv",
            compute_dtype=torch.float32,
        )
    diffusion = _Diffusion()
    diffusion.config = cast("Any", LTXV_2B_V09_CONFIG)
    with pytest.raises(ValueError, match="identity must be nonempty"):
        LTXVDiffusionRuntime(
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


def test_codec_derives_encode_and_decode_budgets_from_total_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, vae = _codec_runtime()
    total_memory = iter((24 * 1024**3, 15 * 1024**3))
    measured_devices: list[torch.device] = []

    def measure(device: torch.device) -> int:
        measured_devices.append(device)
        return next(total_memory)

    def free_memory(_device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(free_total=5 * 1024**3)

    monkeypatch.setattr(ltxv_runtime, "get_total_memory", measure, raising=False)
    monkeypatch.setattr(
        ltxv_runtime,
        "get_free_memory",
        free_memory,
        raising=False,
    )
    runtime.encode_content(torch.zeros((1, 3, 1, 32, 32)))
    runtime.decode_latent(torch.zeros((1, 128, 1, 2, 2)))

    assert measured_devices == [torch.device("cpu"), torch.device("cpu")]
    assert vae.encode_budgets == [128 * 1024**2]
    assert vae.decode_budgets == [80 * 1024**2]


def test_crop_centers_on_the_downscale_grid() -> None:
    content = torch.arange(7, dtype=torch.float32).reshape(1, 1, 1, 1, 7).expand(1, 1, 1, 4, 7)

    cropped = _crop_spatial_to_multiple(content, 4)

    assert cropped.shape == (1, 1, 1, 4, 4)
    assert cropped[0, 0, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_crop_refuses_content_below_one_downscale_step() -> None:
    with pytest.raises(ValueError, match="smaller"):
        _crop_spatial_to_multiple(torch.zeros((1, 3, 1, 3, 64)), 4)


class _ArithmeticDiffusion(_Diffusion):
    """Input-dependent fake: the velocity couples latent, timestep, and
    context so schedule, noise, and CFG-lane parity are all visible."""

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        frame_rate: float,
        denoise_mask: torch.Tensor | None = None,
        guides: object = (),
    ) -> torch.Tensor:
        self.denoise_masks.append(None if denoise_mask is None else denoise_mask.detach().clone())
        self.guides.append(guides)
        self.calls.append(
            (
                latent.detach().clone(),
                timesteps.detach().clone(),
                context.detach().clone(),
                attention_mask.detach().clone(),
                frame_rate,
            )
        )
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        step = timesteps.reshape(latent.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1, 1) * 1e-3
        return latent * 0.5 + step * 0.25 + scale


def _random_latent(seed: int = 11) -> MultiStreamLatent[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return MultiStreamLatent.from_pairs(
        (("video", torch.rand((1, 4, 1, 2, 2), generator=generator)),)
    )


def _lane(value: float, tokens: int = 3, frame_rate: float = 25.0) -> LTXVPreparedConditioning:
    return LTXVPreparedConditioning(torch.full((1, 4, 8), value), tokens, frame_rate)


def _custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def _windows_spec(**overrides: Any) -> ContextWindowsSpec:
    values: dict[str, Any] = {
        "schedule": ContextWindowSchedule.STATIC_STANDARD,
        "fuse_method": ContextFuseMethod.PYRAMID,
        "length": 3,
        "overlap": 1,
        "dim": 2,
    }
    values.update(overrides)
    return ContextWindowsSpec(**values)


def _decode_context_output(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def _context_output_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def test_ltxv_context_windows_match_executed_reference() -> None:
    path = Path(__file__).parent / "goldens" / "ltxv_context_windows.json"
    if not path.is_file():
        pytest.skip("LTXV context-window golden has not been minted on its pinned AMD host")
    golden = load_platform_golden(path, allow_portable_fallback=True)
    model_spec = golden["model"]
    config = model_spec["config"]
    model = LTXVModel(
        LTXVConfig(
            in_channels=config["in_channels"],
            cross_attention_dim=config["cross_attention_dim"],
            attention_head_dim=config["attention_head_dim"],
            num_attention_heads=config["num_attention_heads"],
            caption_channels=config["caption_channels"],
            num_layers=config["num_layers"],
            causal_temporal_positioning=config["causal_temporal_positioning"],
        )
    )
    model.load_state_dict(fill_state_dict(model_spec["state_dict"]), strict=True)
    input_spec = golden["input"]
    latent = hashed_input(input_spec["name"], tuple(input_spec["shape"]))
    context = hashed_input(input_spec["context_name"], tuple(input_spec["context_shape"]))
    timestep = torch.tensor(input_spec["timestep"], dtype=torch.float32)
    attention_mask = torch.tensor(input_spec["attention_mask"], dtype=torch.int64)

    evaluation = ConditioningEvaluation(
        lambda value, _role: value,
        lambda window, _sigma, _condition: model(
            window,
            timestep,
            context,
            attention_mask=attention_mask,
            frame_rate=input_spec["frame_rate"],
        ),
    )
    for case in golden["cases"].values():
        spec_values = case["spec"]
        spec = ContextWindowsSpec(
            schedule=ContextWindowSchedule(spec_values["schedule"]),
            fuse_method=ContextFuseMethod(spec_values["fuse_method"]),
            length=spec_values["length"],
            overlap=spec_values["overlap"],
            stride=spec_values["stride"],
            closed_loop=spec_values["closed_loop"],
            dim=spec_values["dim"],
            freenoise=spec_values["freenoise"],
            causal_anchor=spec_values["causal_anchor"],
            cond_retain_indices=tuple(spec_values["cond_retain_indices"]),
            latent_retain_indices=tuple(spec_values["latent_retain_indices"]),
        )
        expected_retain = (0,) if case["retain_first_frame"] else ()
        assert spec.cond_retain_indices == ()
        assert spec.latent_retain_indices == expected_retain
        assert tuple(case["reference_handler"]["cond_retain_indices"]) == expected_retain
        assert tuple(case["reference_handler"]["latent_retain_indices"]) == expected_retain
        assert plan_windows(spec, latent.shape[2], 0) == tuple(
            tuple(window) for window in case["windows"]
        )

        wrapped = windowed_conditioning_evaluation(evaluation, spec, (0.5,))
        with torch.no_grad():
            observed = wrapped.evaluate(latent, 0.5, object())
        expected = _decode_context_output(case["output"])
        assert _context_output_sha256(expected) == case["output_sha256"]
        assert_reference_tensor(observed, expected, rtol=1e-4, atol=1e-5)


def test_ltxv_runtime_satisfies_the_custom_sampling_protocol() -> None:
    runtime, _ = _runtime()
    assert isinstance(runtime, CustomSamplingRuntime)
    assert isinstance(runtime, ContextWindowsRuntime)
    assert runtime.supports_context_windows


def test_context_windows_split_model_applications_and_fuse_to_full_length() -> None:
    video = _video((1, 4, 7, 2, 2))
    conditioning = _prepared()

    baseline_runtime, baseline_diffusion = _runtime()
    baseline = _sample(baseline_runtime, video.clone(), conditioning, steps=1)
    assert [call[0].shape[2] for call in baseline_diffusion.calls] == [7]

    windowed_runtime, windowed_diffusion = _runtime()
    windowed = _sample(
        windowed_runtime,
        video.clone(),
        conditioning,
        steps=1,
        context_windows=_windows_spec(),
    )

    assert [call[0].shape[2] for call in windowed_diffusion.calls] == [3, 3, 3]
    assert torch.equal(windowed, baseline)


def test_context_windows_batch_guidance_lanes_window_per_invocation() -> None:
    runtime, diffusion = _runtime()

    _sample(
        runtime,
        _video((1, 4, 7, 2, 2)),
        _lane(2.0),
        cfg=SamplingGuidance(cast("Any", _lane(0.0)), 2.0, batching=FUSE_CFG_LANES),
        steps=1,
        context_windows=_windows_spec(),
    )

    assert [tuple(call[0].shape[:3]) for call in diffusion.calls] == [(2, 4, 3)] * 3


def test_lazy_cache_skips_complete_windowed_guidance_evaluations() -> None:
    latent = MultiStreamLatent.from_pairs(
        (("video", torch.rand((1, 4, 7, 2, 2), generator=torch.Generator().manual_seed(11))),)
    )
    noise = latent.map(torch.zeros_like)
    positive = _lane(2.0)
    negative = _lane(0.0)
    sigmas = (1.0, 0.8, 0.7, 0.6, 0.0)

    def guidance(
        runtime: LTXVDiffusionRuntime,
    ) -> SamplingGuidance[PreparedMultiStreamConditioning]:
        return SamplingGuidance(
            PreparedMultiStreamConditioning(runtime.conditioning_identity, negative),
            2.0,
            batching=FUSE_CFG_LANES,
        )

    baseline_runtime, baseline_diffusion = _runtime(_ArithmeticDiffusion())
    baseline = baseline_runtime.sample_custom(
        latent,
        noise=noise,
        cond=PreparedMultiStreamConditioning(baseline_runtime.conditioning_identity, positive),
        cfg=guidance(baseline_runtime),
        request=_custom_request(sigmas=sigmas),
        context_windows=_windows_spec(),
    ).output.by_role("video")

    cached_runtime, cached_diffusion = _runtime(_ArithmeticDiffusion())
    cached = cached_runtime.sample_custom(
        latent,
        noise=noise,
        cond=PreparedMultiStreamConditioning(cached_runtime.conditioning_identity, positive),
        cfg=guidance(cached_runtime),
        request=CustomSamplingRequest(
            _custom_request().sampler,
            (),
            sigmas,
            cache=LazyCacheConfig(),
        ),
        context_windows=_windows_spec(),
    ).output.by_role("video")

    assert len(baseline_diffusion.calls) == 12
    assert len(cached_diffusion.calls) == 6
    assert torch.isfinite(cached).all()
    assert not torch.equal(cached, baseline)


def test_context_windows_freenoise_matches_pre_shuffled_custom_noise() -> None:
    seed = 123
    video = _video((1, 4, 7, 2, 2))
    latent = MultiStreamLatent.from_pairs((("video", video),))
    noise_video = torch.arange(video.numel(), dtype=video.dtype).reshape(video.shape)
    noise = MultiStreamLatent.from_pairs((("video", noise_video),))
    shuffled_video = apply_freenoise(noise_video, 2, 3, 1, seed)
    assert not torch.equal(shuffled_video, noise_video)

    freenoise_runtime, _ = _runtime()
    freenoise = freenoise_runtime.sample_custom(
        latent,
        noise=noise,
        cond=PreparedMultiStreamConditioning(freenoise_runtime.conditioning_identity, _prepared()),
        request=_custom_request(),
        seed=seed,
        context_windows=_windows_spec(freenoise=True),
    )

    shuffled_runtime, _ = _runtime()
    shuffled = shuffled_runtime.sample_custom(
        latent,
        noise=MultiStreamLatent.from_pairs((("video", shuffled_video),)),
        cond=PreparedMultiStreamConditioning(shuffled_runtime.conditioning_identity, _prepared()),
        request=_custom_request(),
        seed=seed,
        context_windows=_windows_spec(),
    )

    assert type(freenoise.output) is MultiStreamLatent
    assert type(shuffled.output) is MultiStreamLatent
    assert torch.equal(freenoise.output.by_role("video"), shuffled.output.by_role("video"))


def test_context_windows_latent_retain_pins_full_tensor_frame_zero() -> None:
    runtime, diffusion = _runtime()
    video = (
        torch.arange(1, 8, dtype=torch.float32).reshape(1, 1, 7, 1, 1).expand(1, 4, 7, 2, 2).clone()
    )
    latent = MultiStreamLatent.from_pairs((("video", video),))

    runtime.sample_custom(
        latent,
        noise=latent.map(torch.zeros_like),
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, _prepared()),
        request=_custom_request(sigmas=(0.5, 0.0)),
        context_windows=_windows_spec(latent_retain_indices=(0,)),
    )

    assert len(diffusion.calls) == 3
    retained_frame = diffusion.calls[0][0].select(2, 0)
    assert all(torch.equal(call[0].select(2, 0), retained_frame) for call in diffusion.calls)


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
    negative = None if cfg_scale is None else _lane(5.0, tokens=2)
    latent = _random_latent()
    identity = runtime.conditioning_identity

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=SamplingGuidance(cast("Any", negative), 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=LTX_SIGMAS,
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
        error=LTXVRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))


def test_ksampler_sugar_parity_holds_with_guides_and_denoise_mask() -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    keyframes = torch.arange(24, dtype=torch.int64).reshape(1, 3, 4, 2)
    guide = LTXVGuideConditioning(keyframes, (1, 2, 2), 0.75)
    positive = LTXVPreparedConditioning(
        torch.full((1, 4, 8), 2.0),
        3,
        25.0,
        (guide,),
    )
    latent = MultiStreamLatent.from_pairs(
        (("video", torch.rand((1, 4, 2, 2, 2), generator=torch.Generator().manual_seed(11))),)
    )
    mask = MultiStreamLatent.from_pairs(
        (("video", torch.tensor([1.0, 0.0]).reshape(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2)),)
    )

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        denoise_mask=mask,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=LTX_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        denoise_mask=mask,
        error=LTXVRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))


def test_ksampler_sugar_parity_holds_for_low_precision_latents() -> None:
    """Seeded noise is drawn against float32 views on both surfaces,
    so half-precision latent streams must not round the draw on the
    decomposed path."""
    runtime, _ = _runtime(_ArithmeticDiffusion())
    latent = _random_latent().map(lambda stream: stream.to(dtype=torch.float16))
    positive = _lane(2.0)

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=LTX_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        cfg=SamplingGuidance(None, 1.0),
        error=LTXVRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))


def test_ksampler_sugar_parity_holds_for_dual_guidance_lanes() -> None:
    runtime, _ = _runtime(_ArithmeticDiffusion())
    latent = _random_latent()
    positive = _lane(2.0)
    negative = _lane(5.0, tokens=2)
    middle = _lane(3.0, tokens=4)
    identity = runtime.conditioning_identity

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=DualSamplingGuidance(cast("Any", middle), cast("Any", negative), 3.0, 1.5),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=LTX_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(identity, positive),
        cfg=DualSamplingGuidance(
            cast("Any", PreparedMultiStreamConditioning(identity, middle)),
            cast("Any", PreparedMultiStreamConditioning(identity, negative)),
            3.0,
            1.5,
        ),
        error=LTXVRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))


def test_ksampler_sugar_parity_holds_for_explicit_sampling_shift() -> None:
    """A sampling_shift changes the sigma space, which
    run_ksampler_as_custom cannot forward; the manual composition uses
    the shifted space as the schedule source and passes the shift to
    sample_custom."""
    runtime, _ = _runtime(_ArithmeticDiffusion())
    latent = _random_latent()
    positive = _lane(2.0)
    sampler = torch_sampler_registry().get("dinkster.euler")
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert sampler is not None
    assert scheduler is not None

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=77,
        sampling_shift=5.0,
    )
    schedule = build_sampling_schedule(
        scheduler, FluxFlowSigmas(shift=5.0), sampler, 2, denoise=1.0, flow=True
    )
    result = runtime.sample_custom(
        latent,
        noise=prepare_multistream_noise(latent, 77),
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        request=CustomSamplingRequest(sampler, (), schedule.pre_offset),
        seed=77,
        sampling_shift=5.0,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("video"), expected.by_role("video"))


def test_custom_sampling_sigma_surfaces_match_the_ltx_flow_space() -> None:
    runtime, _ = _runtime()
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, LTX_SIGMAS, 4, denoise=0.5
    )
    with pytest.raises(LTXVRuntimeError, match="unknown scheduler"):
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
    latent = _random_latent()
    noise = latent.map(torch.zeros_like)
    request = _custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": prepared, "request": request}
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(TypeError, match="exact MultiStreamLatent"):
        sample(latent=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(LTXVRuntimeError, match="requires MultiStreamLatent noise"):
        sample(noise=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(LTXVRuntimeError, match="match the latent stream roles"):
        sample(noise=MultiStreamLatent.from_pairs((("audio", torch.zeros((1, 4, 1, 2, 2))),)))
    with pytest.raises(TypeError, match="strided floating"):
        sample(
            noise=MultiStreamLatent.from_pairs(
                (("video", torch.zeros((1, 4, 1, 2, 2), dtype=torch.int64)),)
            )
        )
    with pytest.raises(LTXVRuntimeError, match="match the video latent shape"):
        sample(noise=MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, 1, 2, 3))),)))
    with pytest.raises(LTXVRuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(cast("Any", prepared), cast("Any", prepared), 3.0, 1.0))
    with pytest.raises(LTXVRuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    invalid_mask = latent.replace("video", torch.full_like(latent.by_role("video"), math.nan))
    with pytest.raises(LTXVRuntimeError, match=r"denoise mask values.*\[0, 1\]"):
        sample(denoise_mask=invalid_mask)
    with pytest.raises(LTXVRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(LTXVRuntimeError, match="cond_retain_indices"):
        sample(context_windows=_windows_spec(cond_retain_indices=(0,)))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(LTXVRuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(LTXVRuntimeError, match="brownian sampler needs positive sigmas"):
        sample(request=_custom_request("dinkster.dpmpp_sde", sigmas=(0.0, 0.0)))
    with pytest.raises(LTXVRuntimeError, match="prepared multi-stream conditioning"):
        sample(cond=cast("Any", conditioning))
    with pytest.raises(LTXVRuntimeError, match="a different conditioner component"):
        sample(cond=PreparedMultiStreamConditioning("native:other", conditioning))
    with pytest.raises(TypeError, match="exact LTXVPreparedConditioning"):
        sample(
            cond=PreparedMultiStreamConditioning(
                runtime.conditioning_identity, cast("Any", object())
            )
        )
    with pytest.raises(LTXVRuntimeError, match="guidance requires prepared"):
        sample(cfg=SamplingGuidance(cast("Any", conditioning), 2.0))
    with pytest.raises(LTXVRuntimeError, match="different conditioner components"):
        sample(
            cfg=SamplingGuidance(
                cast("Any", PreparedMultiStreamConditioning("native:other", conditioning)), 2.0
            )
        )
    with pytest.raises(TypeError, match="guidance lanes require exact LTXVPreparedConditioning"):
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
    with pytest.raises(LTXVRuntimeError, match="share one frame rate"):
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
    latent = _random_latent()
    result = runtime.sample_custom(
        latent,
        noise=latent.map(torch.zeros_like),
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, _prepared()),
        request=_custom_request(),
        seed=9,
    )
    output = result.output
    assert type(output) is MultiStreamLatent
    assert output.roles == ("video",)
    assert result.denoised_output is not None
    assert result.denoised_output.roles == ("video",)
