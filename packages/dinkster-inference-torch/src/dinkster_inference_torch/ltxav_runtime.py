"""Executable text-to-audio-video runtime for supported LTX-2 profiles."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from dinkster_inference import (
    LTX_SIGMAS,
    LTXAV,
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_CONFIG,
    LTXAV_19B_VAE_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V23_VAE_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_22B_V25_VAE_CONFIG,
    LTXAV_AUDIO_CHANNELS,
    LTXAV_AUDIO_FREQUENCY_BINS,
    LTXAV_VIDEO_CODEC,
    AudioPreview,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    CustomSamplingResult,
    DualSamplingGuidance,
    FluxFlowSigmas,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidancePlanAugmentationDescriptor,
    GuidancePlanContext,
    GuidancePostCFGContext,
    GuidancePostCFGDescriptor,
    GuidanceReduceContext,
    GuidanceRole,
    GuidanceStrategyDescriptor,
    LatentPackLayout,
    LTXAVConfig,
    LTXGeneratedKeyframes,
    LTXVocoderBWEConfig,
    ModelFamily,
    MultiStreamFamilyRuntime,
    MultiStreamLatent,
    Parameterization,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    Registry,
    SamplerDescriptor,
    SamplingCancelled,
    SamplingGuidance,
    SchedulerDescriptor,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    encode_conditioning_carrier,
    ltx_audio_output_sample_rate,
    make_conditioning_carrier,
)
from dinkster_inference.guidance import GuidancePhaseParticipation

from ._conditioning_layout import DeclaredConditioning, declared_token_count
from .codecs import CodecPlugin
from .denoise import prepare_noise, to_batch
from .gemma_text import GemmaTextModel, LtxDualTextProjection, LtxGemmaTextEncoder
from .gemma_tokenizer import GemmaJsonTokenizer, GemmaSentencePieceTokenizer
from .latent_streams import normalize_latent_mask, pack_latent_streams, unpack_latent_streams
from .ltx_audio_vae import ltx_vocoder_features
from .ltx_connector import LtxTextConnectors
from .ltx_diffusion_vae import LTXDiffusionVideoVAE
from .ltx_media import LTXMediaError, materialize_ltxav_reference_audio
from .ltx_video_vae import LTXVideoVAE, ltxv_vae_max_chunk_bytes
from .ltxav_component import LTXAVAudioCodec
from .ltxav_model import LTXAVModel, pack_av_latents, unpack_av_latents
from .memory import get_total_memory
from .operations import bound_compute_device
from .parameterizations import calculate_denoised, calculate_input
from .payloads import TensorPayloadError, payload_binding_to_tensor, tensor_to_payload_binding
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionInputs,
    SamplingExecutionRegistration,
    SamplingLatentAdapter,
    sampling_execution,
)
from .sampling_runtime import MultiStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)

if TYPE_CHECKING:
    from .checkpoint_runtime import ComponentAssembly
from .solvers import torch_sampler_registry


class LTXAVRuntimeError(ValueError):
    """The requested operation is outside the loaded LTX-2 audio-video profile."""


def _validate_text(value: object, *, text_dim: int) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("LTX-2 audio-video TEXT must be an exact strided floating torch.Tensor")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] != text_dim:
        raise LTXAVRuntimeError(f"LTX-2 audio-video TEXT must be nonempty [B,tokens,{text_dim}]")
    return value


def _validate_frame_rate(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise LTXAVRuntimeError("LTX-2 audio-video frame rate must be a positive finite float")
    return value


@dataclass(frozen=True, slots=True)
class LTXAVExecutionOptions:
    """Per-lane LTX-2 transformer execution controls."""

    stg_self_attn_blocks: frozenset[int] = frozenset()
    a2v_cross_attention: bool = True
    v2a_cross_attention: bool = True

    def __post_init__(self) -> None:
        if type(self.stg_self_attn_blocks) is not frozenset or any(
            type(index) is not int or index < 0 for index in self.stg_self_attn_blocks
        ):
            raise TypeError("LTX-2 STG blocks must be a frozenset of nonnegative integers")
        if type(self.a2v_cross_attention) is not bool or type(self.v2a_cross_attention) is not bool:
            raise TypeError("LTX-2 cross-attention controls must be exact bools")


@dataclass(frozen=True, slots=True)
class LTXAVPreparedConditioning:
    """Materialized LTX-2 audio-video conditioning carried opaquely by the
    multistream seam.

    Every row is attended: the Gemma encoder returns only the attended
    prompt rows (padding is stripped before projection), so there is no
    masked tail and no attention token count."""

    text: torch.Tensor
    frame_rate: float = 25.0
    reference_audio: torch.Tensor | None = None
    execution: LTXAVExecutionOptions = field(default_factory=LTXAVExecutionOptions)
    generated_keyframes: LTXGeneratedKeyframes | None = None

    def __post_init__(self) -> None:
        if (
            type(self.text) is not torch.Tensor
            or not self.text.is_floating_point()
            or self.text.layout != torch.strided
        ):
            raise TypeError("LTX-2 audio-video TEXT must be an exact strided floating torch.Tensor")
        if self.text.ndim != 3 or any(size <= 0 for size in self.text.shape):
            raise LTXAVRuntimeError("LTX-2 audio-video TEXT must be a nonempty rank-3 tensor")
        _validate_frame_rate(self.frame_rate)
        if type(self.execution) is not LTXAVExecutionOptions:
            raise TypeError("LTX-2 execution options must be exact LTXAVExecutionOptions")
        if (
            self.generated_keyframes is not None
            and type(self.generated_keyframes) is not LTXGeneratedKeyframes
        ):
            raise TypeError("LTX-2 generated keyframes must be exact LTXGeneratedKeyframes")
        reference = self.reference_audio
        if reference is not None and (
            type(reference) is not torch.Tensor
            or reference.layout is not torch.strided
            or not reference.is_floating_point()
            or reference.ndim != 3
            or any(size <= 0 for size in reference.shape)
        ):
            raise TypeError(
                "LTX-2 reference audio must be a nonempty floating [B,tokens,width] tensor"
            )


def ltxav_identity_guidance(
    scale: float,
    sigma_start: float,
    sigma_end: float,
    *,
    order: int = 0,
) -> GuidanceContribution[torch.Tensor]:
    """LTXVReferenceAudio ID-LoRA guidance at ComfyUI pin b78cec87."""
    for name, value in (
        ("scale", scale),
        ("sigma_start", sigma_start),
        ("sigma_end", sigma_end),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float")
    if scale < 0.0:
        raise ValueError("scale must be nonnegative")

    no_reference_id = "dinkster.ltxav.identity-no-reference"

    def augment(
        context: GuidancePlanContext[torch.Tensor],
        plan: GuidanceEvaluationPlan[torch.Tensor],
    ) -> GuidanceEvaluationPlan[torch.Tensor]:
        sigma = context.execution.current_sigma
        if scale == 0.0 or sigma > sigma_start or sigma < sigma_end:
            return plan
        primary = next(lane for lane in plan.lanes if lane.id == plan.primary_id)
        source = cast("object", primary.conditioning)
        if type(source) is not LTXAVPreparedConditioning:
            raise TypeError("LTX-2 identity guidance requires exact prepared conditioning")
        if source.reference_audio is None:
            raise LTXAVRuntimeError(
                "LTX-2 identity guidance requires reference audio on positive conditioning"
            )
        lane = GuidanceCondition(
            no_reference_id,
            GuidanceRole.AUXILIARY,
            cast("Conditioning[torch.Tensor]", replace(source, reference_audio=None)),
        )
        return replace(plan, lanes=(*plan.lanes, lane))

    def apply(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        by_id = {prediction.lane_id: prediction.value for prediction in context.predictions.items}
        no_reference = by_id.get(no_reference_id)
        if no_reference is None:
            return context.reduced
        conditional = by_id[context.request.plan.primary_id]
        return context.reduced + (conditional - no_reference) * scale

    metadata = (
        ("config.scale", str(scale)),
        ("config.sigma_end", str(sigma_end)),
        ("config.sigma_start", str(sigma_start)),
    )
    return GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor(
                "dinkster.ltxav.identity-guidance.plan",
                augment,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.ltxav.identity-guidance.apply",
                apply,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
    )


def ltxav_spatiotemporal_guidance(
    scale: float,
    blocks: frozenset[int],
    sigma_start: float,
    sigma_end: float,
    *,
    lane_id: str = "dinkster.ltxav.stg-perturbed",
    order: int = 0,
) -> GuidanceContribution[torch.Tensor]:
    """Guide away from value-passthrough self-attention in selected blocks."""
    for name, value in (
        ("scale", scale),
        ("sigma_start", sigma_start),
        ("sigma_end", sigma_end),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float")
    if scale < 0.0:
        raise ValueError("scale must be nonnegative")
    if type(blocks) is not frozenset or any(
        type(index) is not int or index < 0 for index in blocks
    ):
        raise TypeError("blocks must be a frozenset of nonnegative integers")
    if type(lane_id) is not str or not lane_id:
        raise TypeError("lane_id must be a nonempty exact string")

    def augment(
        context: GuidancePlanContext[torch.Tensor],
        plan: GuidanceEvaluationPlan[torch.Tensor],
    ) -> GuidanceEvaluationPlan[torch.Tensor]:
        sigma = context.execution.current_sigma
        if scale == 0.0 or not blocks or sigma > sigma_start or sigma < sigma_end:
            return plan
        primary = next(lane for lane in plan.lanes if lane.id == plan.primary_id)
        source = cast("object", primary.conditioning)
        if type(source) is not LTXAVPreparedConditioning:
            raise TypeError("LTX-2 STG requires exact prepared conditioning")
        execution = replace(source.execution, stg_self_attn_blocks=blocks)
        lane = GuidanceCondition(
            lane_id,
            GuidanceRole.AUXILIARY,
            cast("Conditioning[torch.Tensor]", replace(source, execution=execution)),
        )
        return replace(plan, lanes=(*plan.lanes, lane))

    def apply(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        predictions = {item.lane_id: item.value for item in context.predictions.items}
        perturbed = predictions.get(lane_id)
        if perturbed is None:
            return context.reduced
        conditional = predictions[context.request.plan.primary_id]
        return context.reduced + (conditional - perturbed) * scale

    metadata = (
        ("config.blocks", ",".join(str(index) for index in sorted(blocks))),
        ("config.scale", str(scale)),
        ("config.sigma_end", str(sigma_end)),
        ("config.sigma_start", str(sigma_start)),
    )
    return GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor(
                "dinkster.ltxav.stg.plan",
                augment,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.ltxav.stg.apply",
                apply,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
    )


def ltxav_modality_guidance(
    scale: float,
    sigma_start: float,
    sigma_end: float,
    *,
    lane_id: str = "dinkster.ltxav.modality-decoupled",
    order: int = 0,
) -> GuidanceContribution[torch.Tensor]:
    """Guide toward the prediction with audio-video coupling enabled."""
    for name, value in (
        ("scale", scale),
        ("sigma_start", sigma_start),
        ("sigma_end", sigma_end),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float")
    if scale < 1.0:
        raise ValueError("modality guidance scale must be at least 1")
    if type(lane_id) is not str or not lane_id:
        raise TypeError("lane_id must be a nonempty exact string")

    def augment(
        context: GuidancePlanContext[torch.Tensor],
        plan: GuidanceEvaluationPlan[torch.Tensor],
    ) -> GuidanceEvaluationPlan[torch.Tensor]:
        sigma = context.execution.current_sigma
        if math.isclose(scale, 1.0) or sigma > sigma_start or sigma < sigma_end:
            return plan
        primary = next(lane for lane in plan.lanes if lane.id == plan.primary_id)
        source = cast("object", primary.conditioning)
        if type(source) is not LTXAVPreparedConditioning:
            raise TypeError("LTX-2 modality guidance requires exact prepared conditioning")
        execution = replace(
            source.execution,
            a2v_cross_attention=False,
            v2a_cross_attention=False,
        )
        lane = GuidanceCondition(
            lane_id,
            GuidanceRole.AUXILIARY,
            cast("Conditioning[torch.Tensor]", replace(source, execution=execution)),
        )
        return replace(plan, lanes=(*plan.lanes, lane))

    def apply(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        predictions = {item.lane_id: item.value for item in context.predictions.items}
        decoupled = predictions.get(lane_id)
        if decoupled is None:
            return context.reduced
        conditional = predictions[context.request.plan.primary_id]
        return context.reduced + (conditional - decoupled) * (scale - 1.0)

    metadata = (
        ("config.scale", str(scale)),
        ("config.sigma_end", str(sigma_end)),
        ("config.sigma_start", str(sigma_start)),
    )
    return GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor(
                "dinkster.ltxav.modality-guidance.plan",
                augment,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.ltxav.modality-guidance.apply",
                apply,
                order=order,
                behavior_metadata=metadata,
            ),
        ),
    )


def ltxav_dual_cfg_guidance(
    video_scale: float,
    audio_scale: float,
    video_elements: int,
) -> GuidanceContribution[torch.Tensor]:
    """Apply independent CFG scales to packed video and audio predictions."""
    for name, value in (("video_scale", video_scale), ("audio_scale", audio_scale)):
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative float")
    if type(video_elements) is not int or video_elements <= 0:
        raise ValueError("video_elements must be a positive exact int")

    def plan(context: GuidancePlanContext[torch.Tensor]) -> GuidanceEvaluationPlan[torch.Tensor]:
        conditions = {lane.id: lane for lane in context.conditions}
        return GuidanceEvaluationPlan(
            (conditions["positive"], conditions["negative"]),
            "positive",
            "negative",
        )

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        predictions = {item.lane_id: item.value for item in context.predictions.items}
        conditional = predictions[context.request.plan.primary_id]
        unconditional_id = context.request.plan.unconditional_id
        if unconditional_id is None or unconditional_id not in predictions:
            raise LTXAVRuntimeError("LTX-2 dual CFG requires an unconditional prediction")
        unconditional = predictions[unconditional_id]
        if conditional.shape[-1] <= video_elements:
            raise LTXAVRuntimeError("LTX-2 dual CFG requires packed video and audio predictions")
        input = context.request.input
        conditional_noise = input - conditional
        unconditional_noise = input - unconditional
        guided_noise = unconditional_noise + (conditional_noise - unconditional_noise) * video_scale
        guided_noise[..., video_elements:] = (
            unconditional_noise[..., video_elements:]
            + (conditional_noise[..., video_elements:] - unconditional_noise[..., video_elements:])
            * audio_scale
        )
        return input - guided_noise

    metadata = (
        ("config.audio_scale", str(audio_scale)),
        ("config.video_elements", video_elements),
        ("config.video_scale", str(video_scale)),
    )
    return GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "dinkster.ltxav.dual-cfg",
            plan,
            reduce,
            participation=GuidancePhaseParticipation.COMPOSE,
            requires_uncond=True,
            behavior_metadata=metadata,
        )
    )


def _conditioning_text(value: object, *, text_dim: int) -> torch.Tensor:
    if type(value) not in (Conditioning, DeclaredConditioning):
        raise TypeError("LTX-2 audio-video guidance lanes require exact Conditioning values")
    conditioning = cast("Conditioning[torch.Tensor]", value)
    if conditioning.pooled is not None:
        raise LTXAVRuntimeError(
            "LTX-2 audio-video text conditioning does not accept pooled embeddings"
        )
    return _validate_text(conditioning.embeddings, text_dim=text_dim)


def _text_carrier(
    value: object, *, text_dim: int, family_id: str, text_stream: str = "gemma3_12b"
) -> ConditioningCarrier:
    text = _conditioning_text(value, text_dim=text_dim)
    conditioning = cast("Conditioning[torch.Tensor]", value)
    declared = declared_token_count(conditioning)
    token_count = int(text.shape[1]) if declared is None else declared
    binding = tensor_to_payload_binding("ltxav-text", text, space="conditioning-text")
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id),
        binding.shape,
        binding.dtype,
        binding.space,
    )
    record = ConditioningRecord(
        channels=((ConditioningChannel.TEXT, descriptor),),
        token_layout=TokenLayoutDescriptor(
            family_id=family_id,
            version=1,
            text_streams=(text_stream,),
            segments=(TokenSegmentDescriptor(text_stream, text_stream, 0, token_count),),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (binding,))


def _materialize_conditioning(
    carrier: object,
    *,
    text_dim: int,
    family_id: str,
    expected_text_stream: str,
) -> torch.Tensor:
    """One carrier -> text embeddings.

    The token layout's single segment must cover exactly the payload
    rows: the Gemma encoder emits only attended rows, so a shorter
    declared prompt length would silently drop conditioning."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("LTX-2 audio-video conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise LTXAVRuntimeError(
            f"LTX-2 audio-video conditioning carrier is invalid: {error}"
        ) from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise LTXAVRuntimeError("LTX-2 audio-video requires exactly one conditioning record")
    record = records[0]
    channels = dict(record.channels)
    unsupported = tuple(
        channel.value for channel in channels if channel is not ConditioningChannel.TEXT
    )
    if unsupported:
        raise LTXAVRuntimeError(
            "LTX-2 audio-video does not consume conditioning channels: " + ", ".join(unsupported)
        )
    descriptor = channels.get(ConditioningChannel.TEXT)
    if descriptor is None:
        raise LTXAVRuntimeError("LTX-2 audio-video conditioning is missing the TEXT channel")
    if len(descriptor.shape) != 3 or descriptor.shape[2] != text_dim:
        raise LTXAVRuntimeError(
            f"LTX-2 audio-video TEXT descriptor must have shape [B,tokens,{text_dim}]"
        )
    if record.area is not None:
        raise LTXAVRuntimeError("LTX-2 audio-video does not consume conditioning areas")
    if record.mask is not None:
        raise LTXAVRuntimeError("LTX-2 audio-video does not consume conditioning masks")
    if record.schedule != PercentRange(0.0, 1.0):
        raise LTXAVRuntimeError("LTX-2 audio-video does not consume conditioning schedules")
    if record.scale_vector is not None:
        raise LTXAVRuntimeError("LTX-2 audio-video does not consume conditioning scale vectors")
    if record.extension_metadata:
        raise LTXAVRuntimeError(
            "LTX-2 audio-video does not consume conditioning extension metadata"
        )
    layout = record.token_layout
    if layout is None:
        raise LTXAVRuntimeError("LTX-2 audio-video conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise LTXAVRuntimeError(f"LTX-2 audio-video token layout is unsupported: {error}") from None
    text_stream = layout.text_streams[0] if len(layout.text_streams) == 1 else None
    if (
        text_stream != expected_text_stream
        or len(layout.segments) != 1
        or layout.segments[0].name != text_stream
        or layout.segments[0].stream != text_stream
        or layout.segments[0].start_token != 0
    ):
        raise LTXAVRuntimeError(
            f"LTX-2 audio-video conditioning requires the exact {expected_text_stream} token layout"
        )
    token_count = layout.segments[0].token_count
    if token_count is None or token_count != descriptor.shape[1]:
        raise LTXAVRuntimeError(
            "LTX-2 audio-video token layout segment must cover exactly the TEXT payload rows"
        )
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise LTXAVRuntimeError(
            "LTX-2 audio-video conditioning carrier has duplicate payload bindings"
        )
    binding = bindings.get(descriptor.reference.id)
    if binding is None:
        raise LTXAVRuntimeError("LTX-2 audio-video TEXT channel has no payload binding")
    if (binding.shape, binding.dtype, binding.space) != (
        descriptor.shape,
        descriptor.dtype,
        descriptor.space,
    ):
        raise LTXAVRuntimeError(
            "LTX-2 audio-video TEXT descriptor does not match its payload binding"
        )
    if binding.space != "conditioning-text":
        raise LTXAVRuntimeError("LTX-2 audio-video TEXT payload has an unsupported tensor space")
    try:
        text = payload_binding_to_tensor(binding)
    except TensorPayloadError as error:
        raise LTXAVRuntimeError(
            f"LTX-2 audio-video TEXT payload could not be decoded: {error}"
        ) from None
    return _validate_text(text, text_dim=text_dim)


def _av_streams(
    value: object, *, video_channels: int
) -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor, torch.Tensor]:
    if type(value) is not MultiStreamLatent:
        raise TypeError("LTX-2 audio-video latent must be an exact MultiStreamLatent")
    streams = cast("MultiStreamLatent[torch.Tensor]", value)
    if streams.roles != ("video", "audio"):
        raise LTXAVRuntimeError(
            "LTX-2 audio-video requires the exact latent stream roles ('video', 'audio')"
        )
    video = streams.by_role("video")
    audio = streams.by_role("audio")
    for latent in (video, audio):
        if (
            type(latent) is not torch.Tensor
            or not latent.is_floating_point()
            or latent.layout != torch.strided
        ):
            raise TypeError(
                "LTX-2 audio-video latents must be exact strided floating torch.Tensor values"
            )
    if (
        video.ndim != 5
        or video.shape[1] != video_channels
        or any(size <= 0 for size in video.shape)
    ):
        raise LTXAVRuntimeError(
            f"LTX-2 audio-video video latent must be nonempty [B,{video_channels},T,H,W]"
        )
    if (
        audio.ndim != 4
        or audio.shape[1] != LTXAV_AUDIO_CHANNELS
        or audio.shape[3] != LTXAV_AUDIO_FREQUENCY_BINS
        or any(size <= 0 for size in audio.shape)
    ):
        raise LTXAVRuntimeError(
            "LTX-2 audio-video audio latent must be nonempty"
            f" [B,{LTXAV_AUDIO_CHANNELS},T,{LTXAV_AUDIO_FREQUENCY_BINS}]"
        )
    if audio.shape[0] != video.shape[0]:
        raise LTXAVRuntimeError("LTX-2 audio-video latent streams must share one batch size")
    return streams, video, audio


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SamplingCancelled("sampling cancelled")


def _sigma_space(sampling_shift: float | None) -> FluxFlowSigmas:
    if sampling_shift is None:
        return LTX_SIGMAS
    if (
        type(sampling_shift) is not float
        or not math.isfinite(sampling_shift)
        or sampling_shift <= 0.0
    ):
        raise LTXAVRuntimeError("sampling_shift must be a positive finite float")
    return FluxFlowSigmas(shift=sampling_shift)


def _crop_spatial_to_multiple(content: torch.Tensor, multiple: int) -> torch.Tensor:
    for dimension in (-2, -1):
        size = content.shape[dimension]
        cropped = (size // multiple) * multiple
        if cropped == 0:
            raise ValueError(
                f"content dimension {dimension} has extent {size}, smaller"
                f" than one {multiple}x downscale step"
            )
        if cropped != size:
            content = content.narrow(dimension, (size % multiple) // 2, cropped)
    return content


@dataclass(frozen=True, slots=True)
class _LTXAVCodec:
    """Adapt the LTX-2 video VAE's chunked entry points to the codec
    encoder/decoder protocols (the VAE streams internal chunks; the
    plugin never tiles it)."""

    vae: LTXVideoVAE | LTXDiffusionVideoVAE

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        max_chunk_bytes = ltxv_vae_max_chunk_bytes(get_total_memory(content.device))
        return self.vae.encode(content, max_chunk_bytes=max_chunk_bytes)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        max_chunk_bytes = ltxv_vae_max_chunk_bytes(get_total_memory(latent.device))
        return self.vae.decode(latent, max_chunk_bytes=max_chunk_bytes)


@dataclass(frozen=True, slots=True)
class _LTXAVModelConditioning:
    text: torch.Tensor
    frame_rate: float
    reference_audio: torch.Tensor | None
    execution: LTXAVExecutionOptions
    generated_keyframes: LTXGeneratedKeyframes | None


def _reference_audio_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return torch.equal(left, right)


@dataclass(frozen=True, slots=True)
class _LTXAVDiffusionAssembly:
    family: ModelFamily
    diffusion: LTXAVModel
    diffusion_dtype: torch.dtype
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.diffusion_dtype if component == "diffusion" else None


@dataclass(frozen=True)
class _LTXAVLatentContext:
    original: MultiStreamLatent[torch.Tensor]
    layout: LatentPackLayout
    stream_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    video_mask: torch.Tensor | None
    audio_mask: torch.Tensor | None


@dataclass(frozen=True)
class _LTXAVLatentAdapter:
    def prepare(
        self,
        runtime: object,
        family: ModelFamily,
        *,
        latent: CustomSamplingLatentValue,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        denoise_mask: CustomSamplingLatentValue | None,
        context: SamplingAdapterContext,
        error: type[Exception],
    ) -> SamplingExecutionInputs:
        del family, error
        owner = cast("LTXAVDiffusionRuntime", runtime)
        if context.options:
            names = ", ".join(sorted(context.options))
            raise LTXAVRuntimeError(f"LTX-2 sampling does not accept adapter options: {names}")
        model = owner.assembled.diffusion
        streams, video, audio = _av_streams(latent, video_channels=model.config.in_channels)
        if type(noise) is not MultiStreamLatent:
            raise LTXAVRuntimeError(
                "LTX-2 audio-video custom sampling requires MultiStreamLatent noise"
            )
        if noise.roles != streams.roles:
            raise LTXAVRuntimeError(
                "LTX-2 audio-video noise streams must match the latent stream roles"
            )
        noise_video = noise.by_role("video")
        noise_audio = noise.by_role("audio")
        for noise_stream, reference in ((noise_video, video), (noise_audio, audio)):
            if (
                type(noise_stream) is not torch.Tensor
                or not noise_stream.is_floating_point()
                or noise_stream.layout != torch.strided
            ):
                raise TypeError(
                    "LTX-2 audio-video noise streams must be exact strided floating"
                    " torch.Tensor values"
                )
            if tuple(noise_stream.shape) != tuple(reference.shape):
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video noise streams must match the latent stream shapes"
                )
        if isinstance(cfg, PerpNegSamplingGuidance):
            raise LTXAVRuntimeError(
                "LTX-2 audio-video custom sampling does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        if type(cond) is not PreparedMultiStreamConditioning:
            raise LTXAVRuntimeError(
                "LTX-2 audio-video custom sampling requires prepared multi-stream conditioning"
            )
        if cond.runtime_identity != owner.conditioning_identity:
            raise LTXAVRuntimeError(
                "LTX-2 audio-video conditioning was prepared by a different conditioner component"
            )
        conditioning = cond.payload
        if type(conditioning) is not LTXAVPreparedConditioning:
            raise TypeError("conditioning must be exact LTXAVPreparedConditioning")
        _validate_text(conditioning.text, text_dim=owner.text_dim)

        def unwrap(value: object) -> LTXAVPreparedConditioning | None:
            if value is None:
                return None
            if type(value) is not PreparedMultiStreamConditioning:
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video custom sampling guidance requires prepared"
                    " multi-stream conditioning"
                )
            if value.runtime_identity != cond.runtime_identity:
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video guidance lanes were prepared by different"
                    " conditioner components"
                )
            payload = value.payload
            if type(payload) is not LTXAVPreparedConditioning:
                raise TypeError(
                    "LTX-2 audio-video guidance lanes require exact LTXAVPreparedConditioning"
                )
            _validate_text(payload.text, text_dim=owner.text_dim)
            return payload

        uncond = unwrap(None if cfg is None else cfg.uncond)
        middle = unwrap(cfg.middle) if isinstance(cfg, DualSamplingGuidance) else None
        if cfg is None:
            guidance_cfg = None
        elif isinstance(cfg, DualSamplingGuidance):
            guidance_cfg = replace(
                cast("DualSamplingGuidance[object]", cfg), uncond=uncond, middle=middle
            )
        else:
            guidance_cfg = replace(cast("SamplingGuidance[object]", cfg), uncond=uncond)
        for lane in (uncond, middle):
            if lane is not None and lane.frame_rate != conditioning.frame_rate:
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video guidance lanes must share one frame rate"
                )
            if lane is not None and not _reference_audio_equal(
                lane.reference_audio, conditioning.reference_audio
            ):
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video guidance lanes must share the same reference audio"
                )
            if lane is not None and lane.generated_keyframes != conditioning.generated_keyframes:
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video guidance lanes must share generated keyframes"
                )
        packed, layout = pack_latent_streams(streams)
        packed_noise, noise_layout = pack_latent_streams(
            MultiStreamLatent.from_pairs((("video", noise_video), ("audio", noise_audio)))
        )
        if noise_layout != layout:
            raise LTXAVRuntimeError("LTX-2 audio-video noise topology differs from the latent")
        video_mask = None
        audio_mask = None
        sampler_mask = None
        if denoise_mask is not None:
            try:
                normalized = normalize_latent_mask(
                    cast("torch.Tensor | MultiStreamLatent[torch.Tensor]", denoise_mask), streams
                )
            except (TypeError, ValueError) as exception:
                raise LTXAVRuntimeError(
                    f"LTX-2 audio-video denoise mask is invalid: {exception}"
                ) from None
            if any(
                not bool(torch.isfinite(mask.payload).all())
                or float(mask.payload.amin()) < 0.0
                or float(mask.payload.amax()) > 1.0
                for mask in normalized.streams
            ):
                raise LTXAVRuntimeError(
                    "LTX-2 audio-video denoise mask values must be finite within [0, 1]"
                )
            sampler_mask = pack_latent_streams(normalized)[0]
            video_mask = normalized.by_role("video")[:, :1]
            audio_mask = normalized.by_role("audio")[:, :1, :, :1]
        return SamplingExecutionInputs(
            packed,
            packed_noise,
            conditioning,
            guidance_cfg,
            sampler_mask,
            _LTXAVLatentContext(
                streams,
                layout,
                (tuple(video.shape), tuple(audio.shape)),
                video_mask,
                audio_mask,
            ),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        context = cast("_LTXAVLatentContext", inputs.latent_context)
        unpacked = unpack_latent_streams(output, context.layout)
        result = context.original.replace("video", unpacked.by_role("video")).replace(
            "audio", unpacked.by_role("audio")
        )
        if denoised is None:
            return CustomSamplingResult(result, None)
        if type(denoised) is not MultiStreamLatent:
            raise TypeError("LTX-2 denoised state must contain a MultiStreamLatent")
        return CustomSamplingResult(
            result,
            context.original.replace("video", denoised.by_role("video")).replace(
                "audio", denoised.by_role("audio")
            ),
        )


class _LTXAVSamplingDenoiser:
    evaluator_identity = "dinkster.ltxav.conditioning.v1"

    def __init__(
        self,
        owner: LTXAVDiffusionRuntime,
        latent_context: _LTXAVLatentContext,
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        space: FluxFlowSigmas,
        cancelled: Callable[[], bool],
    ) -> None:
        self.owner = owner
        self.latent_context = latent_context
        self.device = device
        self.compute_dtype = compute_dtype
        self.space = space
        self.cancelled = cancelled

    def prepare_conditioning(self, value: object, _role: GuidanceRole) -> _LTXAVModelConditioning:
        _check_cancelled(self.cancelled)
        if type(value) is not LTXAVPreparedConditioning:
            raise TypeError("LTX-2 audio-video guidance lanes require exact prepared conditioning")
        model = self.owner.assembled.diffusion
        text = _validate_text(value.text, text_dim=self.owner.text_dim).to(
            device=self.device, dtype=self.compute_dtype
        )
        text = model.preprocess_text_embeds(text)
        reference_audio = value.reference_audio
        if reference_audio is not None:
            if reference_audio.shape[2] != model.config.audio_in_channels:
                raise LTXAVRuntimeError(
                    "LTX-2 reference-audio token width must match the audio model"
                )
            reference_audio = reference_audio.to(device=self.device, dtype=self.compute_dtype)
        return _LTXAVModelConditioning(
            text,
            value.frame_rate,
            reference_audio,
            value.execution,
            value.generated_keyframes,
        )

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: _LTXAVModelConditioning
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    def batchable(self, values: tuple[_LTXAVModelConditioning, ...]) -> bool:
        if not values:
            return False
        first = next(iter(values))
        return all(
            value.text.shape[1:] == first.text.shape[1:]
            and value.frame_rate == first.frame_rate
            and _reference_audio_equal(value.reference_audio, first.reference_audio)
            and value.execution == first.execution
            and value.generated_keyframes == first.generated_keyframes
            for value in values[1:]
        )

    def evaluate_conditioning_batch(
        self, x: torch.Tensor, sigma: float, values: tuple[_LTXAVModelConditioning, ...]
    ) -> tuple[torch.Tensor, ...]:
        _check_cancelled(self.cancelled)
        if not values or not self.batchable(values):
            raise LTXAVRuntimeError("LTX-2 model evaluation requires conditioning")
        first = next(iter(values))
        batch = x.shape[0]
        base = calculate_input(Parameterization.FLOW, sigma, x).to(self.compute_dtype)
        model_input = base if len(values) == 1 else torch.cat((base,) * len(values))
        video_input, audio_input = unpack_av_latents(model_input, self.latent_context.stream_shapes)
        text = torch.cat(tuple(to_batch(value.text, batch) for value in values))
        timestep = self.space.timestep(sigma)
        arguments: dict[str, object] = {}
        if self.latent_context.video_mask is None or self.latent_context.audio_mask is None:
            video_timesteps = torch.full(
                (model_input.shape[0],), timestep, device=self.device, dtype=torch.float32
            )
            audio_timesteps = video_timesteps
        else:
            video_mask = to_batch(
                self.latent_context.video_mask.to(self.device, self.compute_dtype), batch
            )
            audio_mask = to_batch(
                self.latent_context.audio_mask.to(self.device, self.compute_dtype), batch
            )
            video_timesteps = (video_mask.float() * timestep).flatten(1)
            audio_timesteps = (audio_mask.float() * timestep).flatten(1)
            if len(values) > 1:
                video_timesteps = torch.cat((video_timesteps,) * len(values))
                audio_timesteps = torch.cat((audio_timesteps,) * len(values))
                video_mask = torch.cat((video_mask,) * len(values))
            arguments["denoise_mask"] = video_mask
        if first.reference_audio is not None:
            reference = to_batch(first.reference_audio, batch)
            if len(values) > 1:
                reference = torch.cat((reference,) * len(values))
            arguments["ref_audio_tokens"] = reference
        arguments.update(
            stg_self_attn_blocks=first.execution.stg_self_attn_blocks,
            a2v_cross_attention=first.execution.a2v_cross_attention,
            v2a_cross_attention=first.execution.v2a_cross_attention,
            generated_keyframes=first.generated_keyframes,
            context_preprocessed=True,
        )
        video_velocity, audio_velocity = self.owner.assembled.diffusion(
            video_input,
            audio_input,
            video_timesteps,
            audio_timesteps,
            text,
            attention_mask=None,
            frame_rate=first.frame_rate,
            **arguments,
        )
        velocity = pack_av_latents((video_velocity, audio_velocity))[0].float()
        return tuple(
            calculate_denoised(Parameterization.FLOW, sigma, output, x)
            for output in velocity.chunk(len(values))
        )


def _ltxav_denoiser(
    runtime: object, compute_dtype: torch.dtype, context: SamplingAdapterContext
) -> SamplingDenoiserExecution:
    owner = cast("LTXAVDiffusionRuntime", runtime)
    if context.inputs is None or context.device is None:
        raise RuntimeError("LTX-2 sampling context is unresolved")
    latent_context = cast("_LTXAVLatentContext", context.inputs.latent_context)
    evaluator = _LTXAVSamplingDenoiser(
        owner,
        latent_context,
        device=context.device,
        compute_dtype=compute_dtype,
        space=cast("FluxFlowSigmas", owner.sampling_sigma_space()),
        cancelled=context.cancelled,
    )
    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        process_in=lambda value: value,
        process_out=lambda value: value,
        unpack_state=lambda value: unpack_latent_streams(value, latent_context.layout),
        denoise_mask_prepared=True,
        fixed_inpaint_latent=True,
    )


def _ltxav_device(runtime: object) -> torch.device:
    model = cast("LTXAVDiffusionRuntime", runtime).assembled.diffusion
    return bound_compute_device(model.patchify_proj) or model.patchify_proj.weight.device


def _ltxav_compute_dtype(runtime: object) -> torch.dtype:
    return (
        cast("LTXAVDiffusionRuntime", runtime).assembled.compute_dtype("diffusion")
        or torch.bfloat16
    )


class LTXAVDiffusionRuntime(MultiStreamSamplingRuntime):
    """Diffusion-only LTX-2 custom-sampling runtime."""

    retained_offload_storage_components = frozenset()
    sampling_error = LTXAVRuntimeError
    supports_denoised_capture = True
    supports_batch_noise_indices = False
    supports_sampling_shift = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=cast("SamplingLatentAdapter", _LTXAVLatentAdapter()),
        denoiser=_ltxav_denoiser,
        device=_ltxav_device,
        compute_dtype=_ltxav_compute_dtype,
        flow=True,
    )
    supports_audio_cfg = True

    def __init__(
        self,
        diffusion: LTXAVModel,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        config = diffusion.config
        if type(config) is not LTXAVConfig:
            raise ValueError("LTX-2 diffusion runtime requires an exact supported model")
        profile_19b = replace(
            LTXAV_19B_CONFIG,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
        )
        profile_22b = replace(
            LTXAV_22B_V23_CONFIG,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
        )
        profile_22b_v25 = replace(
            LTXAV_22B_V25_CONFIG,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
        )
        if (
            config not in (profile_19b, profile_22b, profile_22b_v25)
            or not math.isfinite(config.av_ca_timestep_scale_multiplier)
            or config.av_ca_timestep_scale_multiplier <= 0.0
        ):
            raise ValueError("LTX-2 diffusion runtime requires an exact supported model")
        if not runtime_identity:
            raise ValueError("LTX-2 diffusion runtime identity must be nonempty")
        self._assembled = _LTXAVDiffusionAssembly(LTXAV, diffusion, compute_dtype)
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._guidance = None

    @property
    def assembled(self) -> _LTXAVDiffusionAssembly:
        return self._assembled

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def video_vae_config(self) -> object:
        config = self.assembled.diffusion.config
        profile_19b = replace(
            LTXAV_19B_CONFIG,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
        )
        if config == profile_19b:
            return LTXAV_19B_VAE_CONFIG
        profile_v25 = replace(
            LTXAV_22B_V25_CONFIG,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
        )
        return LTXAV_22B_V25_VAE_CONFIG if config == profile_v25 else LTXAV_22B_V23_VAE_CONFIG

    @property
    def audio_vae_config(self) -> object:
        return LTXAV_19B_AUDIO_VAE_CONFIG

    @property
    def _text_dim(self) -> int:
        # The model consumes the video and audio text projections
        # concatenated on the feature axis.
        config = self.assembled.diffusion.config
        if config.caption_proj_before_connector:
            return config.cross_attention_dim + config.audio_cross_attention_dim
        return 2 * config.caption_channels

    @property
    def _text_stream(self) -> str:
        config = self.assembled.diffusion.config
        if config.caption_proj_before_connector and not config.ff_bias:
            return "gemma4_12b"
        return "gemma3_12b"

    @property
    def conditioning_identity(self) -> str:
        """Stable compatibility identity for materialized LTX-2 audio-video
        conditioning."""
        config = self.assembled.diffusion.config
        fields = (
            self.family.id,
            config.in_channels,
            config.cross_attention_dim,
            config.attention_head_dim,
            config.num_attention_heads,
            config.audio_in_channels,
            config.audio_cross_attention_dim,
            config.audio_attention_head_dim,
            config.audio_num_attention_heads,
            config.caption_channels,
            config.num_layers,
            config.causal_temporal_positioning,
            config.av_ca_timestep_scale_multiplier,
        )
        identity = "dinkster.ltxav.conditioning:v1:" + ":".join(str(field) for field in fields)
        if config.caption_proj_before_connector:
            suffix = "22b-v2.5" if not config.ff_bias else "22b-v2.3"
            return identity + ":" + suffix
        return identity

    @property
    def text_dim(self) -> int:
        return self._text_dim

    def prepare_conditioning(
        self,
        carrier: ConditioningCarrier,
        *,
        frame_rate: float = 25.0,
        generated_keyframes: LTXGeneratedKeyframes | None = None,
    ) -> LTXAVPreparedConditioning:
        try:
            carrier, reference_audio = materialize_ltxav_reference_audio(
                carrier,
                token_width=self.assembled.diffusion.config.audio_in_channels,
            )
        except LTXMediaError as error:
            raise LTXAVRuntimeError(str(error)) from None
        text = _materialize_conditioning(
            carrier,
            text_dim=self._text_dim,
            family_id=self.family.id,
            expected_text_stream=self._text_stream,
        )
        return LTXAVPreparedConditioning(
            text=text,
            frame_rate=_validate_frame_rate(frame_rate),
            reference_audio=reference_audio,
            generated_keyframes=generated_keyframes,
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FluxFlowSigmas:
        return _sigma_space(sampling_shift)

    def _ksampler_noise(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        seed: int,
        noise_inds: Sequence[int] | None,
        add_noise: bool,
    ) -> MultiStreamLatent[torch.Tensor]:
        model = self.assembled.diffusion
        streams, video, audio = _av_streams(latent, video_channels=model.config.in_channels)
        if not add_noise:
            return super()._ksampler_noise(streams, seed, noise_inds, False)
        # The reference draws once over the packed AV layout, not once per stream.
        packed_f32, noise_layout = pack_latent_streams(
            MultiStreamLatent.from_pairs(
                (
                    ("video", video.to(device="cpu", dtype=torch.float32)),
                    ("audio", audio.to(device="cpu", dtype=torch.float32)),
                )
            )
        )
        return unpack_latent_streams(prepare_noise(packed_f32, seed), noise_layout)

    sample_custom = cast("Any", sampling_execution)  # noqa: F811


class LTXAVTextRuntime:
    """Text-only LTX-2 encoder over independently loaded components."""

    def __init__(
        self,
        gemma: GemmaTextModel,
        projection: object,
        tokenizer_model: bytes,
        *,
        connectors: LtxTextConnectors | None,
    ) -> None:
        if not tokenizer_model:
            raise ValueError("LTX-2 text runtime requires tokenizer model bytes")
        if isinstance(projection, torch.nn.Linear):
            if projection.out_features != LTXAV_19B_CONFIG.caption_channels or not isinstance(
                connectors, LtxTextConnectors
            ):
                raise ValueError("LTX-2 19B text requires the single projection and connectors")
            text_dim = 2 * LTXAV_19B_CONFIG.caption_channels
        elif isinstance(projection, LtxDualTextProjection):
            if connectors is not None:
                raise ValueError("LTX-2.3 22B text projection must not carry 19B connectors")
            text_dim = (
                LTXAV_22B_V23_CONFIG.cross_attention_dim
                + LTXAV_22B_V23_CONFIG.audio_cross_attention_dim
            )
        else:
            raise TypeError("LTX-2 text runtime requires an exact supported projection")
        self._text_dim = text_dim
        self._text_stream = (
            "gemma4_12b" if gemma.config.architecture == "gemma4_ltx_12b" else "gemma3_12b"
        )
        tokenizer = (
            GemmaJsonTokenizer(tokenizer_model)
            if self._text_stream == "gemma4_12b"
            else GemmaSentencePieceTokenizer(tokenizer_model)
        )
        self._encoder = LtxGemmaTextEncoder(
            gemma,
            projection,
            tokenizer,
            connectors=connectors,
        )

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        return self._encoder.encode(text)

    def text_conditioning_carrier(
        self, conditioning: Conditioning[torch.Tensor]
    ) -> ConditioningCarrier:
        return _text_carrier(
            conditioning,
            text_dim=self._text_dim,
            family_id=LTXAV.id,
            text_stream=self._text_stream,
        )


class LTXAVVideoCodecRuntime:
    """Video codec facade over one independently loaded LTX-2 VAE."""

    def __init__(
        self,
        vae: LTXVideoVAE | LTXDiffusionVideoVAE,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        adapter = _LTXAVCodec(vae)
        self.codec = CodecPlugin(
            LTXAV_VIDEO_CODEC,
            adapter,
            adapter,
            content_crop=lambda content: _crop_spatial_to_multiple(
                content, LTXAV_VIDEO_CODEC.latent.spatial_downscale
            ),
            content_in=lambda value: value * 2.0 - 1.0,
            content_out=lambda value: value.float().add_(1.0).div_(2.0).clamp_(0.0, 1.0),
            compute_dtype=compute_dtype,
        )

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content).float()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent).float()


class _LTXAVCheckpointCodec(LTXAVVideoCodecRuntime):
    descriptor = LTXAV_VIDEO_CODEC
    encode = LTXAVVideoCodecRuntime.encode_content
    decode = LTXAVVideoCodecRuntime.decode_latent


def checkpoint_codec(assembled: ComponentAssembly) -> _LTXAVCheckpointCodec | None:
    vae = assembled.components.get("vae")
    return (
        None
        if vae is None
        else _LTXAVCheckpointCodec(
            cast("LTXVideoVAE | LTXDiffusionVideoVAE", vae),
            compute_dtype=assembled.compute_dtype("vae"),
        )
    )


class LTXAVAudioCodecRuntime:
    """Audio codec facade over one independently loaded VAE/vocoder pair."""

    def __init__(self, codec: LTXAVAudioCodec) -> None:
        self.codec = codec

    def decode_audio_latent(self, latent: torch.Tensor) -> AudioPreview[torch.Tensor]:
        audio_vae = self.codec.audio_vae
        vocoder = self.codec.vocoder
        mel = audio_vae.decode(latent.to(dtype=torch.float32))
        if isinstance(vocoder.config, LTXVocoderBWEConfig):
            vocoder_config = vocoder.config.vocoder
            output_sample_rate = vocoder.config.output_sampling_rate
        else:
            vocoder_config = vocoder.config
            output_sample_rate = ltx_audio_output_sample_rate(audio_vae.config, vocoder_config)
        waveform = vocoder(ltx_vocoder_features(mel, vocoder_config.audio_channels)).float()
        return AudioPreview(waveform, output_sample_rate)


_MultiStreamRuntimeCheck: type[MultiStreamFamilyRuntime[torch.Tensor]] = LTXAVDiffusionRuntime
