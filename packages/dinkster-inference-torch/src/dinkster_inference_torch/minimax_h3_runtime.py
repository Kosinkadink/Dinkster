"""Executable per-component audio-video runtimes for MiniMax H3."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    MINIMAX_H3,
    MINIMAX_H3_AUDIO_MASK_MAPPING,
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_SIGMAS,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    MINIMAX_H3_VIDEO_TEMPORAL_MAPPING,
    AudioPreview,
    Conditioning,
    ConditioningCarrier,
    CustomSamplingResult,
    DualSamplingGuidance,
    ExecutionObserverAttachment,
    GuidanceRole,
    LatentMaskMapping,
    LatentPackLayout,
    LatentStream,
    MiniMaxH3AudioContent,
    MiniMaxH3AudioReference,
    MiniMaxH3ConditioningRequest,
    MiniMaxH3Config,
    MiniMaxH3DiTPayloadKind,
    MiniMaxH3FL2VARequest,
    MiniMaxH3GuideTokenGeometry,
    MiniMaxH3ImageReference,
    MiniMaxH3KeyframeRole,
    MiniMaxH3PresentationKind,
    MiniMaxH3REF2VARequest,
    MiniMaxH3ReferenceTokenGeometry,
    MiniMaxH3Sigmas,
    MiniMaxH3SparseAttentionConfig,
    MiniMaxH3Task,
    MiniMaxH3TokenLayoutPlan,
    MiniMaxH3VideoLatentGeometry,
    MiniMaxH3VideoReference,
    ModelFamily,
    ModelTokenLayout,
    MultiStreamLatent,
    PayloadDescriptor,
    PerpNegSamplingGuidance,
    PlacementMap,
    PreparedConditioningCarrier,
    PreparedMultiStreamConditioning,
    Registry,
    SamplerDescriptor,
    SamplingCancelled,
    SamplingGuidance,
    SchedulerDescriptor,
    SequencePartition,
    SequenceShard,
    SigmaSpace,
    TimelineGuide,
    TokenGridTransform,
    TokenLayoutDescriptor,
    TokenLayoutError,
    TokenSegmentDescriptor,
    UspMesh,
    build_canonical_manifest,
    compose_execution,
    execution_span,
    normalize_minimax_h3_conditioning,
    plan_minimax_h3_token_layout,
    plan_sequence_partition,
    prove_manifest_consensus,
)
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32
from dinkster_inference.partition_compatibility import PartitionCompatibility

from .conditioning_adapters import basic_conditioning_to_carrier
from .context_windows import (
    PackedContextWindowAxis,
    PackedContextWindows,
    PackedContextWindowScale,
    PackedContextWindowStream,
)
from .denoise import PackedInpaintConfiguration, prepare_noise
from .distributed import (
    SequenceDigestConsensusTransport,
    distributed_sampling_config,
    ensure_process_group,
    gather_sequence_device_bindings,
    rank_zero_sampling_active,
    sequence_preflight_failed,
)
from .guidance import ConditioningEvaluation
from .latent_streams import normalize_latent_mask, pack_latent_streams, unpack_latent_streams
from .minimax_h3_assembly import AssembledMiniMaxH3Model
from .minimax_h3_attention import (
    MiniMaxH3AttentionKernelFactory,
    MiniMaxH3PackedSegmentKind,
    MiniMaxH3PackedSequenceFacts,
    MiniMaxH3SequenceSharding,
)
from .minimax_h3_audio import MiniMaxH3AudioVAE
from .minimax_h3_conditioner import MiniMaxH3ConditionerModel
from .minimax_h3_conditioning import (
    MiniMaxH3ConditionerInputs,
    MiniMaxH3VisionValue,
    realize_minimax_h3_conditioner_inputs,
)
from .minimax_h3_control import MiniMaxH3FunControlConditioning
from .minimax_h3_dit import (
    MiniMaxH3Attention,
    MiniMaxH3BlockAttention,
    MiniMaxH3BlockAttentionFactory,
    MiniMaxH3DiT,
    MiniMaxH3DiTConditioning,
    MiniMaxH3KeyframeLatent,
    MiniMaxH3ReferenceKind,
    MiniMaxH3ReferenceLatents,
)
from .minimax_h3_sparse_attention import MiniMaxH3SparseAttention
from .minimax_h3_video_vae import MiniMaxH3VideoVAE
from .operations import bound_compute_device
from .regional import MaterializedRegion, full_region_multiplier, materialize_regions
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
    SamplingPipelineHooks,
    sampling_execution,
)
from .sampling_runtime import MultiStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .sequence_parallel_attention import (
    SequenceParallelAttentionKernel,
    gather_sequence_hidden,
)
from .sequence_parallel_plan import (
    build_usp_compiled_plan_slot,
    compile_sequence_parallel_plan,
    exchange_backend_identity,
)
from .solvers import torch_sampler_registry


def _h3_latent(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))


class MiniMaxH3RuntimeError(ValueError):
    """The requested H3 operation violates the executable profile."""


def _h3_token_grid_masks(
    mask: MultiStreamLatent[torch.Tensor],
    patch: tuple[int, int, int],
) -> MultiStreamLatent[torch.Tensor]:
    video = mask.by_role("video")
    video = video.amax(dim=1, keepdim=True).expand_as(video).contiguous()
    _, patch_h, patch_w = patch
    height, width = video.shape[-2:]
    leading_shape = video.shape[:-2]
    padded = torch.nn.functional.pad(
        video.reshape((-1,) + video.shape[-3:]),
        (0, -width % patch_w, 0, -height % patch_h),
        mode="replicate",
    )
    padded = padded.reshape(leading_shape + padded.shape[-2:])
    pooled = padded.reshape(
        *padded.shape[:-2],
        padded.shape[-2] // patch_h,
        patch_h,
        padded.shape[-1] // patch_w,
        patch_w,
    ).amax(dim=(-3, -1))
    video_mask = pooled.repeat_interleave(patch_h, dim=-2).repeat_interleave(patch_w, dim=-1)[
        ..., :height, :width
    ]
    audio = mask.by_role("audio")
    audio_mask = audio.amax(dim=1, keepdim=True).expand_as(audio).contiguous()
    return _h3_latent(
        torch.ceil(video_mask * 256.0) / 256.0,
        torch.ceil(audio_mask * 256.0) / 256.0,
    )


def _validate_h3_mask(mask: MultiStreamLatent[torch.Tensor]) -> None:
    for stream in mask.streams:
        if not bool(torch.isfinite(stream.payload).all()):
            raise MiniMaxH3RuntimeError("H3 denoise mask values must be finite")
        if float(stream.payload.amin()) < 0.0 or float(stream.payload.amax()) > 1.0:
            raise MiniMaxH3RuntimeError("H3 denoise mask values must be within [0, 1]")


def _dit_component_roles(task: MiniMaxH3Task) -> tuple[str, ...]:
    if task is MiniMaxH3Task.T2VA:
        return ("fl2va_dit", "ref2va_dit")
    return ("ref2va_dit",) if task is MiniMaxH3Task.REF2VA else ("fl2va_dit",)


def _h3_attention_backend_identity(attention: MiniMaxH3Attention) -> str:
    evidence = attention.provider_evidence
    identity = f"{evidence.provider}:torch={evidence.torch_version}"
    if evidence.provider_version is not None:
        identity = f"{identity}:dinkster-kitchen={evidence.provider_version}"
    return identity


@dataclass(frozen=True, slots=True)
class MiniMaxH3PreparedConditioning:
    """Conditioner output and exact task-specific DiT payload."""

    task: MiniMaxH3Task
    frame_count: int
    context: torch.Tensor
    dit: MiniMaxH3DiTConditioning
    target_layout: LatentPackLayout
    token_layout: MiniMaxH3TokenLayoutPlan | None = None


def _h3_video_token_grid(
    video: torch.Tensor,
    patch: tuple[int, int, int],
) -> tuple[int, int, int]:
    if video.ndim != 5 or min(video.shape[-3:]) < 1:
        raise TokenLayoutError("H3 materialized video segments must have positive rank-5 geometry")
    temporal, height, width = video.shape[-3:]
    patch_t, patch_h, patch_w = patch
    return (
        -(-temporal // patch_t),
        -(-height // patch_h),
        -(-width // patch_w),
    )


def _materialized_h3_segments(
    value: MultiStreamLatent[torch.Tensor],
    context: torch.Tensor,
    conditioning: MiniMaxH3DiTConditioning,
    patch: tuple[int, int, int],
) -> tuple[tuple[str, MiniMaxH3PackedSegmentKind, tuple[int, ...]], ...]:
    if context.ndim != 3 or context.shape[1] < 1:
        raise TokenLayoutError("H3 materialized text context must have positive rank-3 geometry")
    segments: list[tuple[str, MiniMaxH3PackedSegmentKind, tuple[int, ...]]] = [
        ("text", "text", (context.shape[1],))
    ]
    for keyframe in conditioning.keyframes:
        if keyframe.resolved_frame_index == 0:
            identity = "keyframe-first"
        elif (
            conditioning.frame_count is not None
            and keyframe.resolved_frame_index == conditioning.frame_count - 1
        ):
            identity = "keyframe-last"
        else:
            raise TokenLayoutError("H3 materialized keyframe has an undeclared frame role")
        segments.append((identity, "condition", _h3_video_token_grid(keyframe.video, patch)))
    for ordinal, guide in enumerate(conditioning.guides, start=1):
        if "video" in guide.latent.roles:
            segments.append(
                (
                    f"guide-{ordinal}-video",
                    "condition",
                    _h3_video_token_grid(guide.latent.by_role("video"), patch),
                )
            )
        if "audio" in guide.latent.roles:
            guide_audio = guide.latent.by_role("audio")
            if guide_audio.ndim != 4 or min(guide_audio.shape[-2:]) < 1:
                raise TokenLayoutError(
                    "H3 materialized guide audio must have positive rank-4 geometry"
                )
            segments.append(
                (
                    f"guide-{ordinal}-audio",
                    "condition_audio",
                    tuple(guide_audio.shape[-2:]),
                )
            )
    for ordinal, reference in enumerate(conditioning.references, start=1):
        if reference.audio is not None:
            if reference.audio.ndim != 4 or min(reference.audio.shape[-2:]) < 1:
                raise TokenLayoutError(
                    "H3 materialized reference audio must have positive rank-4 geometry"
                )
            segments.append(
                (
                    f"reference-{ordinal}-audio",
                    "reference_audio",
                    tuple(reference.audio.shape[-2:]),
                )
            )
        if reference.video is not None:
            segments.append(
                (
                    f"reference-{ordinal}-video",
                    "reference_video",
                    _h3_video_token_grid(reference.video, patch),
                )
            )
    video = value.by_role("video")
    audio = value.by_role("audio")
    if audio.ndim != 4 or min(audio.shape[-2:]) < 1:
        raise TokenLayoutError("H3 materialized target audio must have positive rank-4 geometry")
    segments.extend(
        (
            ("target-audio", "audio", tuple(audio.shape[-2:])),
            ("target-video", "video", _h3_video_token_grid(video, patch)),
        )
    )
    return tuple(segments)


def _packed_sequence_facts(
    value: MultiStreamLatent[torch.Tensor],
    context: torch.Tensor,
    conditioning: MiniMaxH3DiTConditioning,
    patch: tuple[int, int, int],
) -> MiniMaxH3PackedSequenceFacts:
    segments: list[tuple[int, int, MiniMaxH3PackedSegmentKind]] = []
    video_grid: tuple[int, int, int] | None = None
    offset = 0
    for _identity, kind, grid in _materialized_h3_segments(value, context, conditioning, patch):
        rows = 1
        for size in grid:
            rows *= size
        segments.append((offset, offset + rows, kind))
        offset += rows
        if kind == "video":
            video_grid = cast("tuple[int, int, int]", grid)
    assert video_grid is not None
    return MiniMaxH3PackedSequenceFacts(offset, tuple(segments), video_grid)


def _validate_h3_model_token_layout(
    value: MultiStreamLatent[torch.Tensor],
    context: torch.Tensor,
    conditioning: MiniMaxH3DiTConditioning,
    patch: tuple[int, int, int],
    layout: ModelTokenLayout,
) -> None:
    declared_segments: list[tuple[str, MiniMaxH3PackedSegmentKind, tuple[int, ...]]] = []
    for segment in layout.segments:
        if segment.identity == "text":
            kind: MiniMaxH3PackedSegmentKind = "text"
        elif segment.role == "condition" and segment.modality == "video":
            kind = "condition"
        elif segment.role == "condition" and segment.modality == "audio":
            kind = "condition_audio"
        elif segment.role == "reference" and segment.modality == "audio":
            kind = "reference_audio"
        elif segment.role == "reference" and segment.modality == "video":
            kind = "reference_video"
        elif segment.role == "target" and segment.modality == "audio":
            kind = "audio"
        elif segment.role == "target" and segment.modality == "video":
            kind = "video"
        else:
            raise TokenLayoutError("H3 layout contains an unknown packed segment role")
        declared_segments.append((segment.identity, kind, segment.grid))
    measured_segments = _materialized_h3_segments(value, context, conditioning, patch)
    if measured_segments != tuple(declared_segments):
        raise TokenLayoutError(
            "H3 materialized packed rows or grids do not match the declared layout"
        )


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SamplingCancelled("sampling cancelled")


def _frame_count_from_video_latent(latent: torch.Tensor) -> int:
    temporal = latent.shape[2]
    if temporal < 1:
        raise MiniMaxH3RuntimeError("H3 video latent must have a positive temporal extent")
    return MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(temporal)


def _adapt_minimax_h3_empty_latent(
    latent: torch.Tensor,
    *,
    source_spatial_downscale: int | None = None,
    source_temporal_downscale: int | None = None,
) -> MultiStreamLatent[torch.Tensor]:
    """Adapt an ordinary all-zero latent to H3's declared AV streams."""

    if type(latent) is not torch.Tensor:
        raise TypeError("H3 latent adaptation requires an exact torch.Tensor")
    if latent.ndim not in (4, 5) or latent.shape[0] < 1 or min(latent.shape[1:]) < 1:
        raise ValueError("H3 latent adaptation requires nonempty rank-4 or rank-5 input")
    if not latent.is_floating_point() or latent.layout != torch.strided:
        raise TypeError("H3 latent adaptation requires a strided floating tensor")
    if bool(torch.count_nonzero(latent)):
        raise ValueError("H3 can adapt only an all-zero ordinary latent")
    for name, value in (
        ("source_spatial_downscale", source_spatial_downscale),
        ("source_temporal_downscale", source_temporal_downscale),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer when provided")

    temporal = 1 if latent.ndim == 4 else latent.shape[2]
    height, width = latent.shape[-2:]
    if source_spatial_downscale is not None:
        height = round(
            height * source_spatial_downscale / MINIMAX_H3_CONFIG.video_spatial_downscale
        )
        width = round(width * source_spatial_downscale / MINIMAX_H3_CONFIG.video_spatial_downscale)
    if source_temporal_downscale is not None:
        temporal = max(1, round(temporal * source_temporal_downscale / 4))
    if min(height, width) < 1:
        raise ValueError("H3 adapted video latent must have positive spatial dimensions")

    frame_count = MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(temporal)
    audio_temporal = round(
        frame_count / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )
    return _h3_latent(
        latent.new_zeros(
            (
                latent.shape[0],
                MINIMAX_H3_CONFIG.video_latent_channels,
                temporal,
                height,
                width,
            )
        ),
        latent.new_zeros(
            (
                latent.shape[0],
                MINIMAX_H3_CONFIG.audio_latent_channels,
                2,
                audio_temporal,
            )
        ),
    )


def empty_minimax_h3_av(
    *,
    width: int,
    height: int,
    frame_count: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> MultiStreamLatent[torch.Tensor]:
    """Allocate the exact H3 generation grid, snapping frames to 17k+5."""

    if any(type(value) is not int for value in (width, height, frame_count)):
        raise TypeError("width, height, and frame_count must be integers")
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 width and height must be positive multiples of 32")
    if frame_count < 1:
        raise ValueError("H3 frame_count must be positive")
    if not dtype.is_floating_point:
        raise TypeError("H3 empty latents require a floating dtype")
    selected_frames = max(5, frame_count)
    selected_frames += (5 - selected_frames) % 17
    video_t = 2 if selected_frames <= 5 else ((selected_frames - 5) // 17) * 5 + 2
    audio_t = round(selected_frames / MINIMAX_H3_CONFIG.video_fps * 40)
    target = torch.device(device)
    return _h3_latent(
        torch.zeros((1, 24, video_t, height // 16, width // 16), device=target, dtype=dtype),
        torch.zeros((1, 32, 2, audio_t), device=target, dtype=dtype),
    )


def _descriptor_tensor(
    descriptor: PayloadDescriptor,
    payloads: Mapping[str, torch.Tensor],
    used: set[str],
) -> torch.Tensor:
    reference = descriptor.reference
    id_ = reference.id
    try:
        tensor = payloads[id_]
    except KeyError as error:
        raise MiniMaxH3RuntimeError(f"payload {id_!r} is missing") from error
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"payload {id_!r} must be an exact torch.Tensor")
    if tuple(tensor.shape) != descriptor.shape:
        raise MiniMaxH3RuntimeError(f"payload {id_!r} shape differs from its descriptor")
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name != descriptor.dtype:
        raise MiniMaxH3RuntimeError(f"payload {id_!r} dtype differs from its descriptor")
    used.add(id_)
    return tensor


def _validate_payload_authority(
    request: MiniMaxH3ConditioningRequest,
    payloads: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    snapshot = dict(payloads)
    if any(type(id_) is not str for id_ in snapshot):
        raise TypeError("H3 payload keys must be exact strings")
    descriptors: list[tuple[PayloadDescriptor, str]] = []
    audio_rates: list[int] = []
    if type(request) is MiniMaxH3FL2VARequest:
        descriptors.extend((keyframe.payload, "image") for keyframe in request.keyframes)
    elif type(request) is MiniMaxH3REF2VARequest:
        for reference in request.references:
            if type(reference) is MiniMaxH3ImageReference:
                descriptors.append((reference.payload, "image"))
            elif type(reference) is MiniMaxH3AudioReference:
                descriptors.append((reference.payload, "audio"))
                audio_rates.append(reference.sample_rate)
            else:
                video = cast(MiniMaxH3VideoReference, reference)
                descriptors.extend((frame, "image") for frame in video.frames)
                if video.audio is not None:
                    descriptors.append((video.audio.payload, "audio"))
                    audio_rates.append(video.audio.sample_rate)
    ids = tuple(descriptor.reference.id for descriptor, _ in descriptors)
    if len(ids) != len(set(ids)):
        raise MiniMaxH3RuntimeError("H3 payload reference IDs must be globally unique")
    if set(snapshot) != set(ids):
        missing = sorted(set(ids) - set(snapshot))
        unused = sorted(set(snapshot) - set(ids))
        detail = f"missing={missing}" if missing else f"unused={unused}"
        raise MiniMaxH3RuntimeError(f"H3 payload key set is not exact: {detail}")
    if any(rate != MINIMAX_H3_CONFIG.audio_sample_rate_hz for rate in audio_rates):
        raise MiniMaxH3RuntimeError("H3 audio references must be 32000 Hz")
    for descriptor, role in descriptors:
        if descriptor.space != "worker:minimax-h3":
            raise MiniMaxH3RuntimeError("H3 payload space must be worker:minimax-h3")
        tensor = snapshot[descriptor.reference.id]
        if type(tensor) is not torch.Tensor:
            raise TypeError("H3 payload values must be exact torch.Tensor values")
        if not tensor.is_floating_point() or tensor.layout != torch.strided:
            raise TypeError("H3 payloads must be strided floating tensors")
        if tuple(tensor.shape) != descriptor.shape:
            raise MiniMaxH3RuntimeError("H3 payload shape differs from its descriptor")
        if str(tensor.dtype).removeprefix("torch.") != descriptor.dtype:
            raise MiniMaxH3RuntimeError("H3 payload dtype differs from its descriptor")
        if role == "image" and (
            tensor.ndim != 4
            or tensor.shape[0] != 1
            or tensor.shape[-1] != 3
            or min(tensor.shape[1:3]) < 2
        ):
            raise MiniMaxH3RuntimeError("H3 image payload must be [1,height,width,3]")
        if role == "audio" and (
            tensor.ndim != 3
            or tensor.shape[0] != 1
            or tensor.shape[1] != MINIMAX_H3_CONFIG.audio_content_channels
            or tensor.shape[2] <= 0
        ):
            raise MiniMaxH3RuntimeError("H3 audio payload must be nonempty [1,2,samples]")
    return snapshot


def _validate_av_target(
    target: MultiStreamLatent[torch.Tensor],
) -> LatentPackLayout:
    if type(target) is not MultiStreamLatent:
        raise TypeError("H3 target must be an exact MultiStreamLatent")
    if target.roles != ("video", "audio"):
        raise MiniMaxH3RuntimeError("H3 target requires exact ordered roles video, audio")
    video, audio = target.by_role("video"), target.by_role("audio")
    if type(video) is not torch.Tensor or type(audio) is not torch.Tensor:
        raise TypeError("H3 target streams must be exact torch.Tensor values")
    if not video.is_floating_point() or not audio.is_floating_point():
        raise TypeError("H3 target streams must be floating tensors")
    if tuple(video.shape[:2]) != (1, 24) or video.ndim != 5:
        raise MiniMaxH3RuntimeError("H3 video target must be [1,24,t,h,w]")
    if tuple(audio.shape[:3]) != (1, 32, 2) or audio.ndim != 4:
        raise MiniMaxH3RuntimeError("H3 audio target must be [1,32,2,t]")
    _, layout = pack_latent_streams(target)
    return layout


def _h3_video_geometry(video: torch.Tensor) -> MiniMaxH3VideoLatentGeometry:
    return MiniMaxH3VideoLatentGeometry(video.shape[2], video.shape[3], video.shape[4])


def _h3_guide_geometry(guide: TimelineGuide[torch.Tensor]) -> MiniMaxH3GuideTokenGeometry:
    video = guide.latent.by_role("video") if "video" in guide.latent.roles else None
    audio = guide.latent.by_role("audio") if "audio" in guide.latent.roles else None
    return MiniMaxH3GuideTokenGeometry(
        guide.frame_index,
        guide.frame_count,
        None if video is None else _h3_video_geometry(video),
        None if audio is None else audio.shape[-1],
    )


def _h3_reference_geometry(
    reference: MiniMaxH3ReferenceLatents,
) -> MiniMaxH3ReferenceTokenGeometry:
    kind = {
        MiniMaxH3ReferenceKind.IMAGE: MiniMaxH3DiTPayloadKind.IMAGE,
        MiniMaxH3ReferenceKind.AUDIO: MiniMaxH3DiTPayloadKind.AUDIO,
        MiniMaxH3ReferenceKind.VIDEO: MiniMaxH3DiTPayloadKind.VIDEO,
    }[reference.kind]
    return MiniMaxH3ReferenceTokenGeometry(
        kind,
        None if reference.video is None else _h3_video_geometry(reference.video),
        None if reference.audio is None else reference.audio.shape[-1],
    )


def _h3_token_layout_for(
    prepared: MiniMaxH3PreparedConditioning,
    target: MultiStreamLatent[torch.Tensor],
    dit: MiniMaxH3DiTConditioning,
) -> MiniMaxH3TokenLayoutPlan:
    if prepared.token_layout is None:
        raise MiniMaxH3RuntimeError("H3 timeline guides require a declared token layout")
    text_grid = prepared.token_layout.layout.by_identity("text").grid
    if len(text_grid) != 1:
        raise TokenLayoutError("H3 text token layout must have one dimension")
    keyframe_roles: list[MiniMaxH3KeyframeRole] = []
    for keyframe in dit.keyframes:
        if keyframe.resolved_frame_index == 0:
            keyframe_roles.append(MiniMaxH3KeyframeRole.FIRST)
        elif keyframe.resolved_frame_index == prepared.frame_count - 1:
            keyframe_roles.append(MiniMaxH3KeyframeRole.LAST)
        else:
            raise TokenLayoutError("H3 keyframe does not resolve to a declared endpoint")
    return plan_minimax_h3_token_layout(
        text_tokens=text_grid[0],
        target_video=_h3_video_geometry(target.by_role("video")),
        target_audio_temporal=target.by_role("audio").shape[-1],
        keyframes=tuple(keyframe_roles),
        guides=tuple(_h3_guide_geometry(guide) for guide in dit.guides),
        references=tuple(_h3_reference_geometry(reference) for reference in dit.references),
    )


def add_minimax_h3_timeline_guide(
    prepared: MiniMaxH3PreparedConditioning,
    target: MultiStreamLatent[torch.Tensor],
    guide: TimelineGuide[torch.Tensor],
) -> MiniMaxH3PreparedConditioning:
    """Validate and append one generic timeline guide to prepared H3 conditioning."""

    if type(prepared) is not MiniMaxH3PreparedConditioning:
        raise TypeError("conditioning must be exact MiniMaxH3PreparedConditioning")
    if type(guide) is not TimelineGuide:
        raise TypeError("guide must be an exact TimelineGuide")
    target_layout = _validate_av_target(target)
    if target_layout != prepared.target_layout:
        raise MiniMaxH3RuntimeError("timeline guide target differs from prepared conditioning")
    if _frame_count_from_video_latent(target.by_role("video")) != prepared.frame_count:
        raise MiniMaxH3RuntimeError("timeline guide target frame count differs from conditioning")
    guide.validate_for_target(prepared.frame_count)
    if guide.latent.roles not in (("video",), ("audio",), ("video", "audio")):
        raise MiniMaxH3RuntimeError("timeline guide requires canonical video and/or audio roles")
    target_video = target.by_role("video")
    target_audio = target.by_role("audio")
    if "video" in guide.latent.roles:
        video = guide.latent.by_role("video")
        if (
            type(video) is not torch.Tensor
            or video.ndim != 5
            or video.shape[:2] != target_video.shape[:2]
            or video.shape[3:] != target_video.shape[3:]
            or video.shape[2] < 1
        ):
            raise MiniMaxH3RuntimeError("timeline guide video must have target-sized H3 geometry")
        if MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(video.shape[2]) != guide.frame_count:
            raise MiniMaxH3RuntimeError(
                "timeline guide video temporal geometry differs from its frame count"
            )
        if (
            not video.is_floating_point()
            or video.layout != torch.strided
            or video.device != target_video.device
        ):
            raise MiniMaxH3RuntimeError(
                "timeline guide video must match the target tensor contract"
            )
    if "audio" in guide.latent.roles:
        audio = guide.latent.by_role("audio")
        if (
            type(audio) is not torch.Tensor
            or audio.ndim != 4
            or audio.shape[:3] != target_audio.shape[:3]
            or audio.shape[-1] < 1
        ):
            raise MiniMaxH3RuntimeError("timeline guide audio must have H3 stereo geometry")
        remaining_audio = int(
            target_audio.shape[-1]
            - MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(guide.frame_index)
        )
        if audio.shape[-1] > remaining_audio:
            raise MiniMaxH3RuntimeError("timeline guide audio exceeds the target timeline")
        if (
            not audio.is_floating_point()
            or audio.layout != torch.strided
            or audio.device != target_audio.device
            or audio.dtype != target_audio.dtype
        ):
            raise MiniMaxH3RuntimeError(
                "timeline guide audio must match the target tensor contract"
            )
    dit = replace(
        prepared.dit,
        frame_count=prepared.frame_count,
        guides=(*prepared.dit.guides, guide),
    )
    return replace(prepared, dit=dit, token_layout=_h3_token_layout_for(prepared, target, dit))


def add_minimax_h3_motion_context(
    prepared: MiniMaxH3PreparedConditioning,
    target: MultiStreamLatent[torch.Tensor],
    previous: MultiStreamLatent[torch.Tensor],
    context_length: int,
) -> tuple[MiniMaxH3PreparedConditioning, float]:
    """Anchor a cloned AV tail from a previous H3 clip at target frame zero."""

    if type(prepared) is not MiniMaxH3PreparedConditioning:
        raise TypeError("conditioning must be exact MiniMaxH3PreparedConditioning")
    if type(context_length) is not int:
        raise TypeError("context_length must be an exact integer")
    target_layout = _validate_av_target(target)
    if target_layout != prepared.target_layout:
        raise MiniMaxH3RuntimeError("motion context target differs from prepared conditioning")
    _validate_av_target(previous)
    target_video = target.by_role("video")
    previous_video = previous.by_role("video")
    previous_audio = previous.by_role("audio")
    target_frames = _frame_count_from_video_latent(target_video)
    if target_frames != prepared.frame_count:
        raise MiniMaxH3RuntimeError("motion context target frame count differs from conditioning")
    previous_frames = _frame_count_from_video_latent(previous_video)
    expected_previous_audio = round(
        previous_frames / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )
    if previous_audio.shape[-1] != expected_previous_audio:
        raise MiniMaxH3RuntimeError("previous H3 audio length does not match its video")
    if previous_video.shape[3:] != target_video.shape[3:]:
        raise MiniMaxH3RuntimeError("previous and target H3 clips must use the same canvas")

    selected_frames = max(5, context_length)
    selected_frames += (5 - selected_frames) % 17
    video_t = 2 if selected_frames <= 5 else ((selected_frames - 5) // 17) * 5 + 2
    audio_t = round(
        selected_frames / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )
    if selected_frames > previous_frames:
        raise MiniMaxH3RuntimeError(
            f"motion context ({selected_frames} frames) exceeds the previous clip's "
            f"{previous_frames} frames"
        )
    if selected_frames > target_frames:
        raise MiniMaxH3RuntimeError(
            f"motion context ({selected_frames} frames) does not fit the target's "
            f"{target_frames} frames"
        )

    guide = TimelineGuide(
        0,
        selected_frames,
        _h3_latent(
            previous_video[:, :, -video_t:].clone(),
            previous_audio[..., -audio_t:].clone(),
        ),
    )
    conditioned = add_minimax_h3_timeline_guide(prepared, target, guide)
    return conditioned, selected_frames / MINIMAX_H3_CONFIG.video_fps


def _video_content(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise MiniMaxH3RuntimeError("H3 video content must be [frames,height,width,3]")
    return tensor.to(device=device).permute(3, 0, 1, 2).unsqueeze(0)


def _move_conditioner_inputs(
    inputs: MiniMaxH3ConditionerInputs, device: torch.device
) -> MiniMaxH3ConditionerInputs:
    return MiniMaxH3ConditionerInputs(
        inputs.ids.to(device),
        inputs.position_ids.to(device),
        inputs.visual_mask.to(device),
        inputs.token_tags.to(device),
        None if inputs.patches is None else inputs.patches.to(device),
        None if inputs.grids is None else inputs.grids.to(device),
        inputs.declared_token_count,
    )


class MiniMaxH3VideoVaeRuntime:
    """Video codec execution for one H3 video VAE component."""

    def __init__(
        self,
        video_vae: MiniMaxH3VideoVAE,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
    ) -> None:
        if not runtime_identity:
            raise ValueError("H3 video VAE runtime identity must be non-empty")
        self._video_vae = video_vae
        self._runtime_identity = runtime_identity
        self._compute_dtype = compute_dtype

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def latent_mask_mapping(self) -> LatentMaskMapping:
        return MINIMAX_H3_VIDEO_MASK_MAPPING

    def encode_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return self._video_vae.encode_output_shape(input_shape)

    def encode_video(self, content: torch.Tensor) -> torch.Tensor:
        # The reference wrapper (comfy/sd.py VAE @ 2a68ce33) feeds this model
        # its default process_input, mapping [0, 1] image content to the
        # [-1, 1] range the encoder expects, before the dtype cast. Decode is
        # asymmetric on purpose: the model finalizes straight to [0, 1].
        return self._video_vae.encode(content.mul(2.0).sub_(1.0).to(dtype=self._compute_dtype))

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        return self._video_vae.decode(latent.to(dtype=self._compute_dtype))

    def preview_visual(self, role: str, latent: torch.Tensor) -> torch.Tensor:
        if role != "video":
            raise ValueError("H3 visual preview role must be 'video'")
        return self.decode_video(latent)


class MiniMaxH3AudioVaeRuntime:
    """Audio codec execution for one H3 audio VAE component."""

    def __init__(
        self,
        audio_vae: MiniMaxH3AudioVAE,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
    ) -> None:
        if not runtime_identity:
            raise ValueError("H3 audio VAE runtime identity must be non-empty")
        self._audio_vae = audio_vae
        self._runtime_identity = runtime_identity
        self._compute_dtype = compute_dtype

    @property
    def config(self) -> MiniMaxH3Config:
        return MINIMAX_H3_CONFIG

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def latent_mask_mapping(self) -> LatentMaskMapping:
        return MINIMAX_H3_AUDIO_MASK_MAPPING

    def encode_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return self._audio_vae.encode_output_shape(input_shape)

    def encode_audio(self, content: MiniMaxH3AudioContent[torch.Tensor]) -> torch.Tensor:
        if type(content) is not MiniMaxH3AudioContent:
            raise TypeError("audio content must be exact MiniMaxH3AudioContent")
        return self._audio_vae.encode(
            content.waveform.to(dtype=self._compute_dtype),
            sample_rate=content.sample_rate,
        )

    def decode_audio(self, latent: torch.Tensor) -> MiniMaxH3AudioContent[torch.Tensor]:
        return MiniMaxH3AudioContent(
            self._audio_vae.decode(latent.to(dtype=self._compute_dtype)),
            self.config.audio_sample_rate_hz,
        )

    def preview_audio(self, role: str, latent: torch.Tensor) -> AudioPreview[torch.Tensor]:
        if role != "audio":
            raise ValueError("H3 audio preview role must be 'audio'")
        audio = self.decode_audio(latent)
        return AudioPreview(audio.waveform, audio.sample_rate)


class MiniMaxH3ConditionerRuntime:
    """Conditioning execution for one H3 conditioner component."""

    def __init__(
        self,
        conditioner: MiniMaxH3ConditionerModel,
        video_vae: MiniMaxH3VideoVaeRuntime | None = None,
        audio_vae: MiniMaxH3AudioVaeRuntime | None = None,
        *,
        runtime_identity: str,
    ) -> None:
        if not runtime_identity:
            raise ValueError("H3 conditioner runtime identity must be non-empty")
        self._conditioner = conditioner
        self._video_vae = video_vae
        self._audio_vae = audio_vae
        self._runtime_identity = runtime_identity

    @property
    def config(self) -> MiniMaxH3Config:
        return MINIMAX_H3_CONFIG

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def video_vae(self) -> MiniMaxH3VideoVaeRuntime | None:
        return self._video_vae

    @property
    def audio_vae(self) -> MiniMaxH3AudioVaeRuntime | None:
        return self._audio_vae

    def adapt_multistream_latent(
        self,
        latent: torch.Tensor,
        *,
        source_spatial_downscale: int | None = None,
        source_temporal_downscale: int | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        return _adapt_minimax_h3_empty_latent(
            latent,
            source_spatial_downscale=source_spatial_downscale,
            source_temporal_downscale=source_temporal_downscale,
        )

    def condition(
        self,
        request: MiniMaxH3ConditioningRequest,
        *,
        target: MultiStreamLatent[torch.Tensor],
        frame_count: int,
        payloads: Mapping[str, torch.Tensor],
        cancelled: Callable[[], bool],
    ) -> MiniMaxH3PreparedConditioning:
        _check_cancelled(cancelled)
        target_layout = _validate_av_target(target)
        if type(frame_count) is not int or frame_count != _frame_count_from_video_latent(
            target.by_role("video")
        ):
            raise MiniMaxH3RuntimeError("frame_count does not match the target video latent")
        expected_audio = round(
            frame_count / self.config.video_fps * self.config.audio_latent_rate_hz
        )
        if target.by_role("audio").shape[-1] != expected_audio:
            raise MiniMaxH3RuntimeError("target audio length does not match frame_count")
        payload_snapshot = _validate_payload_authority(request, payloads)
        plan = normalize_minimax_h3_conditioning(request)
        video_vae = self._video_vae
        audio_vae = self._audio_vae

        def require_video_vae() -> MiniMaxH3VideoVaeRuntime:
            if video_vae is None:
                raise MiniMaxH3RuntimeError(
                    "MiniMax H3 conditioning request requires a video VAE component"
                )
            return video_vae

        def require_audio_vae() -> MiniMaxH3AudioVaeRuntime:
            if audio_vae is None:
                raise MiniMaxH3RuntimeError(
                    "MiniMax H3 conditioning request requires an audio VAE component"
                )
            return audio_vae

        device = target.by_role("video").device
        used: set[str] = set()
        vision: list[MiniMaxH3VisionValue] = []
        for segment in plan.presentation:
            if segment.kind is MiniMaxH3PresentationKind.IMAGE_CONTENT:
                image = _descriptor_tensor(segment.payloads[0], payload_snapshot, used)
                vision.append(MiniMaxH3VisionValue(segment.kind, image.to(device)))
            elif segment.kind is MiniMaxH3PresentationKind.VIDEO_CONTENT:
                frames = tuple(
                    _descriptor_tensor(descriptor, payload_snapshot, used).to(device)
                    for descriptor in segment.payloads
                )
                vision.append(MiniMaxH3VisionValue(segment.kind, torch.cat(frames)))
        inputs = _move_conditioner_inputs(
            realize_minimax_h3_conditioner_inputs(plan, tuple(vision)), device
        )

        def video_geometry(
            descriptors: tuple[PayloadDescriptor, ...],
        ) -> MiniMaxH3VideoLatentGeometry:
            first_shape = descriptors[0].shape
            if any(descriptor.shape[1:3] != first_shape[1:3] for descriptor in descriptors[1:]):
                raise MiniMaxH3RuntimeError("H3 video reference frames must share one size")
            latent_shape = require_video_vae().encode_output_shape(
                (1, 3, len(descriptors), first_shape[1], first_shape[2])
            )
            return MiniMaxH3VideoLatentGeometry(
                latent_shape[2],
                latent_shape[3],
                latent_shape[4],
            )

        def audio_temporal(descriptor: PayloadDescriptor) -> int:
            return require_audio_vae().encode_output_shape(descriptor.shape)[-1]

        declared_references: list[MiniMaxH3ReferenceTokenGeometry] = []
        if type(request) is MiniMaxH3REF2VARequest:
            for reference in request.references:
                if type(reference) is MiniMaxH3ImageReference:
                    declared_references.append(
                        MiniMaxH3ReferenceTokenGeometry(
                            MiniMaxH3DiTPayloadKind.IMAGE,
                            video_geometry((reference.payload,)),
                        )
                    )
                elif type(reference) is MiniMaxH3AudioReference:
                    declared_references.append(
                        MiniMaxH3ReferenceTokenGeometry(
                            MiniMaxH3DiTPayloadKind.AUDIO,
                            audio_temporal=audio_temporal(reference.payload),
                        )
                    )
                else:
                    video_reference = cast(MiniMaxH3VideoReference, reference)
                    declared_references.append(
                        MiniMaxH3ReferenceTokenGeometry(
                            MiniMaxH3DiTPayloadKind.VIDEO,
                            video_geometry(video_reference.frames),
                            (
                                None
                                if video_reference.audio is None
                                else audio_temporal(video_reference.audio.payload)
                            ),
                        )
                    )
        target_video_shape = target_layout.by_role("video").shape
        target_audio_shape = target_layout.by_role("audio").shape
        token_layout = plan_minimax_h3_token_layout(
            text_tokens=inputs.declared_token_count,
            target_video=MiniMaxH3VideoLatentGeometry(
                target_video_shape[2],
                target_video_shape[3],
                target_video_shape[4],
            ),
            target_audio_temporal=target_audio_shape[-1],
            keyframes=(
                tuple(keyframe.role for keyframe in request.keyframes)
                if type(request) is MiniMaxH3FL2VARequest
                else ()
            ),
            references=tuple(declared_references),
        )
        _check_cancelled(cancelled)
        context = self._conditioner.encode(inputs)
        _check_cancelled(cancelled)
        keyframes: list[MiniMaxH3KeyframeLatent] = []
        references: list[MiniMaxH3ReferenceLatents] = []
        if type(request) is MiniMaxH3FL2VARequest:
            for keyframe in request.keyframes:
                content = _descriptor_tensor(keyframe.payload, payload_snapshot, used)
                _check_cancelled(cancelled)
                latent = require_video_vae().encode_video(_video_content(content, device))
                _check_cancelled(cancelled)
                index = 0 if keyframe.role is MiniMaxH3KeyframeRole.FIRST else frame_count - 1
                keyframes.append(MiniMaxH3KeyframeLatent(index, latent))
        elif type(request) is MiniMaxH3REF2VARequest:
            for reference in request.references:
                if type(reference) is MiniMaxH3ImageReference:
                    content = _descriptor_tensor(reference.payload, payload_snapshot, used)
                    _check_cancelled(cancelled)
                    latent = require_video_vae().encode_video(_video_content(content, device))
                    _check_cancelled(cancelled)
                    references.append(
                        MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.IMAGE, latent)
                    )
                elif type(reference) is MiniMaxH3AudioReference:
                    waveform = _descriptor_tensor(reference.payload, payload_snapshot, used).to(
                        device
                    )
                    _check_cancelled(cancelled)
                    latent = (
                        require_audio_vae()
                        .encode_audio(MiniMaxH3AudioContent(waveform, reference.sample_rate))
                        .to(target.by_role("audio"))
                    )
                    _check_cancelled(cancelled)
                    references.append(
                        MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.AUDIO, audio=latent)
                    )
                else:
                    video_reference = cast(MiniMaxH3VideoReference, reference)
                    frames = torch.cat(
                        tuple(
                            _descriptor_tensor(descriptor, payload_snapshot, used)
                            for descriptor in video_reference.frames
                        )
                    )
                    _check_cancelled(cancelled)
                    video = require_video_vae().encode_video(_video_content(frames, device))
                    _check_cancelled(cancelled)
                    audio = None
                    if video_reference.audio is not None:
                        waveform = _descriptor_tensor(
                            video_reference.audio.payload, payload_snapshot, used
                        ).to(device)
                        _check_cancelled(cancelled)
                        audio = (
                            require_audio_vae()
                            .encode_audio(
                                MiniMaxH3AudioContent(waveform, video_reference.audio.sample_rate)
                            )
                            .to(target.by_role("audio"))
                        )
                        _check_cancelled(cancelled)
                    references.append(
                        MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.VIDEO, video, audio)
                    )
        if set(payload_snapshot) != used:
            raise MiniMaxH3RuntimeError(
                f"unreferenced H3 payloads: {', '.join(sorted(set(payload_snapshot) - used))}"
            )
        _check_cancelled(cancelled)
        dit = MiniMaxH3DiTConditioning(
            inputs.token_tags,
            tuple(keyframes),
            tuple(references),
            frame_count if keyframes else None,
        )
        return MiniMaxH3PreparedConditioning(
            plan.task,
            frame_count,
            context,
            dit,
            target_layout,
            token_layout,
        )


@dataclass(frozen=True, slots=True)
class _H3SamplingContext:
    source: MultiStreamLatent[torch.Tensor]
    layout: LatentPackLayout
    conditioning: MiniMaxH3PreparedConditioning
    sigmas: MiniMaxH3Sigmas
    raw_mask: torch.Tensor | None
    token_mask: torch.Tensor | None
    model_mask: MultiStreamLatent[torch.Tensor] | None


_H3EvaluationCondition = tuple[
    str,
    torch.Tensor,
    MiniMaxH3DiTConditioning,
    LatentPackLayout,
    MultiStreamLatent[torch.Tensor] | None,
    MultiStreamLatent[torch.Tensor] | None,
]


def _materialize_h3_conditioning(
    runtime: object,
    carrier: object,
    payloads: tuple[object, ...],
    inputs: SamplingExecutionInputs,
    device: torch.device,
    cancel: Callable[[], bool],
) -> tuple[object, ...]:
    owner = cast("MiniMaxH3DiTRuntime", runtime)
    latent_context = cast("_H3SamplingContext", inputs.latent_context)
    video_shape = latent_context.layout.by_role("video").shape
    if cancel():
        raise SamplingCancelled("MiniMax H3 conditioning materialization was cancelled")
    regions = materialize_regions(
        cast("Any", carrier),
        owner.family.id,
        video_shape[-2],
        video_shape[-1],
        device,
    )
    if len(payloads) != len(regions):
        raise MiniMaxH3RuntimeError("H3 prepared payloads do not match conditioning records")
    mapped: list[MaterializedRegion] = []
    for region, value in zip(regions, payloads, strict=True):
        if type(value) is not MiniMaxH3PreparedConditioning:
            raise TypeError("H3 scheduled payloads require MiniMaxH3PreparedConditioning")
        prepared = value
        if prepared.target_layout != latent_context.layout:
            raise MiniMaxH3RuntimeError("H3 scheduled conditioning belongs to a different target")
        mapped.append(
            replace(
                region,
                family_payload=replace(prepared, context=region.conditioning.embeddings),
            )
        )
    return tuple(mapped)


def _prepare_h3_mask(
    runtime: object,
    inputs: SamplingExecutionInputs,
    denoise_mask: CustomSamplingLatentValue | None,
    _context: SamplingAdapterContext,
) -> SamplingExecutionInputs:
    if denoise_mask is None:
        return inputs
    owner = cast("MiniMaxH3DiTRuntime", runtime)
    latent_context = cast("_H3SamplingContext", inputs.latent_context)
    sampler_latent = unpack_latent_streams(inputs.latent, latent_context.layout)
    raw_masks = normalize_latent_mask(
        cast("torch.Tensor | MultiStreamLatent[torch.Tensor]", denoise_mask),
        sampler_latent,
    )
    _validate_h3_mask(raw_masks)
    token_masks = _h3_token_grid_masks(raw_masks, owner.config.patch)
    packed_raw_mask, raw_layout = pack_latent_streams(raw_masks)
    packed_token_mask, token_layout = pack_latent_streams(token_masks)
    if raw_layout != latent_context.layout or token_layout != latent_context.layout:
        raise MiniMaxH3RuntimeError("H3 denoise mask topology differs from the latent")
    model_mask = (
        token_masks
        if any(float(stream.payload.amin()) < 1.0 - 1e-3 for stream in token_masks.streams)
        else None
    )
    return replace(
        inputs,
        denoise_mask=packed_raw_mask,
        latent_context=replace(
            latent_context,
            raw_mask=packed_raw_mask,
            token_mask=packed_token_mask,
            model_mask=model_mask,
        ),
    )


def _h3_packed_context_windows(
    _runtime: object, inputs: SamplingExecutionInputs
) -> PackedContextWindows:
    layout = cast("_H3SamplingContext", inputs.latent_context).layout
    return PackedContextWindows(
        layout,
        (
            PackedContextWindowAxis(
                2,
                (
                    PackedContextWindowStream("video", 2),
                    PackedContextWindowStream("audio", 3, PackedContextWindowScale.PROPORTIONAL),
                ),
            ),
            PackedContextWindowAxis(3, (PackedContextWindowStream("video", 3),)),
            PackedContextWindowAxis(4, (PackedContextWindowStream("video", 4),)),
        ),
    )


@dataclass(frozen=True, slots=True)
class _H3AttentionBinding:
    kernel_factory: MiniMaxH3AttentionKernelFactory | None
    sparse: MiniMaxH3SparseAttentionConfig | None


def _bind_h3_attention(runtime: object, context: SamplingAdapterContext) -> _H3AttentionBinding:
    del runtime
    factory = context.options.get("attention_kernel_factory")
    sparse = context.options.get("sparse_attention")
    if factory is not None and not callable(factory):
        raise TypeError("attention_kernel_factory must be callable")
    if sparse is not None and type(sparse) is not MiniMaxH3SparseAttentionConfig:
        raise TypeError("sparse_attention must be an exact MiniMaxH3SparseAttentionConfig")
    return _H3AttentionBinding(cast("MiniMaxH3AttentionKernelFactory | None", factory), sparse)


def _encode_h3_conditioning(value: object, reference_prefix: str) -> ConditioningCarrier:
    if type(value) is not MiniMaxH3PreparedConditioning:
        raise TypeError("H3 carrier encoding requires MiniMaxH3PreparedConditioning")
    prepared = value
    return basic_conditioning_to_carrier(
        Conditioning(prepared.context),
        token_layout=TokenLayoutDescriptor(
            MINIMAX_H3.id,
            1,
            ("text",),
            (TokenSegmentDescriptor("text", "text", 0, prepared.context.shape[1]),),
        ),
        reference_prefix=reference_prefix,
    )


@dataclass(frozen=True, slots=True)
class _H3LatentAdapter:
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
        owner = cast("MiniMaxH3DiTRuntime", runtime)
        unknown = set(context.options) - {
            "attention_kernel_factory",
            "sparse_attention",
            "control",
            "noise_inds",
            "scheduler_label",
            "scheduled",
        }
        if unknown:
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 sampling does not accept adapter options: " + ", ".join(sorted(unknown))
            )
        control = context.options.get("control")
        if control is not None and type(control) is not MiniMaxH3FunControlConditioning:
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 control must be exact MiniMaxH3FunControlConditioning"
            )
        if type(latent) is not MultiStreamLatent:
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 custom sampling requires a MultiStreamLatent latent"
            )
        if type(noise) is not MultiStreamLatent:
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 custom sampling requires MultiStreamLatent noise"
            )
        if isinstance(cfg, DualSamplingGuidance):
            raise MiniMaxH3RuntimeError("MiniMax H3 does not support dual CFG guidance")
        if isinstance(cfg, PerpNegSamplingGuidance):
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 custom sampling does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        if type(cond) is not PreparedMultiStreamConditioning:
            raise MiniMaxH3RuntimeError(
                "MiniMax H3 custom sampling requires prepared multi-stream conditioning"
            )
        if cond.runtime_identity != owner.conditioning_identity:
            raise MiniMaxH3RuntimeError(
                "H3 conditioning was prepared by a different conditioner component"
            )
        raw_conditioning = cond.payload
        conditioning_payloads: dict[int, tuple[object, ...]] = {}
        if type(raw_conditioning) is PreparedConditioningCarrier:
            scheduled = raw_conditioning
            if not scheduled.payloads:
                raise MiniMaxH3RuntimeError("H3 scheduled conditioning must not be empty")
            conditioning = scheduled.payloads[0]
            cond_value: object = scheduled.carrier
            conditioning_payloads[id(scheduled.carrier)] = scheduled.payloads
        else:
            conditioning = raw_conditioning
            cond_value = conditioning
        if type(conditioning) is not MiniMaxH3PreparedConditioning:
            raise TypeError("conditioning must be exact MiniMaxH3PreparedConditioning")
        guidance_cfg: SamplingGuidance[object] | None = None
        if cfg is not None:
            uncond_value = cfg.uncond
            if uncond_value is None:
                guidance_cfg = cast("SamplingGuidance[object]", cfg)
            else:
                if type(uncond_value) is not PreparedMultiStreamConditioning:
                    raise MiniMaxH3RuntimeError(
                        "MiniMax H3 custom sampling guidance requires prepared "
                        "multi-stream conditioning"
                    )
                if uncond_value.runtime_identity != cond.runtime_identity:
                    raise MiniMaxH3RuntimeError(
                        "H3 conditional and unconditional lanes were prepared by "
                        "different conditioner components"
                    )
                raw_uncond_payload = uncond_value.payload
                if type(raw_uncond_payload) is PreparedConditioningCarrier:
                    scheduled_uncond = raw_uncond_payload
                    if not scheduled_uncond.payloads:
                        raise MiniMaxH3RuntimeError(
                            "H3 scheduled unconditional conditioning must not be empty"
                        )
                    uncond_payload = scheduled_uncond.payloads[0]
                    guidance_uncond: object = scheduled_uncond.carrier
                    conditioning_payloads[id(scheduled_uncond.carrier)] = scheduled_uncond.payloads
                else:
                    uncond_payload = raw_uncond_payload
                    guidance_uncond = uncond_payload
                if type(uncond_payload) is not MiniMaxH3PreparedConditioning:
                    raise TypeError("H3 guidance lanes require MiniMaxH3PreparedConditioning")
                guidance_cfg = replace(
                    cast("SamplingGuidance[object]", cfg), uncond=guidance_uncond
                )
        allowed_roles = _dit_component_roles(conditioning.task)
        if owner._model_role not in allowed_roles:  # pyright: ignore[reportPrivateUsage]
            raise MiniMaxH3RuntimeError(
                f"MiniMax H3 {owner._model_role} component cannot sample task "  # pyright: ignore[reportPrivateUsage]
                f"{conditioning.task.name}"
            )
        _validate_av_target(latent)
        sigmas = owner._sigmas  # pyright: ignore[reportPrivateUsage]
        sampler_latent = _h3_latent(
            latent.by_role("video").to(dtype=torch.float32),
            latent.by_role("audio").to(dtype=torch.float32) * sigmas.audio_scale,
        )
        packed, layout = pack_latent_streams(sampler_latent)
        packed_noise, noise_layout = pack_latent_streams(noise)
        if noise_layout != layout:
            raise MiniMaxH3RuntimeError("H3 initial noise topology differs from the latent")
        if layout != conditioning.target_layout:
            raise MiniMaxH3RuntimeError("conditioning belongs to a different H3 target")
        latent_context = _H3SamplingContext(
            latent,
            layout,
            conditioning,
            sigmas,
            None,
            None,
            None,
        )
        return SamplingExecutionInputs(
            packed,
            packed_noise,
            cond_value,
            guidance_cfg,
            None,
            latent_context,
            MappingProxyType(conditioning_payloads),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        context = cast("_H3SamplingContext", inputs.latent_context)
        if output is inputs.latent:
            return CustomSamplingResult(context.source, None)
        sigmas = context.sigmas
        unpacked = unpack_latent_streams(output, context.layout)
        output_latent = _h3_latent(
            unpacked.by_role("video"),
            unpacked.by_role("audio") / sigmas.audio_scale,
        )
        denoised_output: MultiStreamLatent[torch.Tensor] | None = None
        if denoised is not None:
            if type(denoised) is not MultiStreamLatent:
                raise TypeError("H3 denoised state must contain a MultiStreamLatent")
            denoised_output = _h3_latent(
                denoised.by_role("video"),
                denoised.by_role("audio") / sigmas.audio_scale,
            )
        return CustomSamplingResult(output_latent, denoised_output)


class MiniMaxH3DiTRuntime(MultiStreamSamplingRuntime):
    """Sampling execution for one role-specific H3 DiT component."""

    sampling_error = MiniMaxH3RuntimeError
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=cast("SamplingLatentAdapter", _H3LatentAdapter()),
        denoiser=lambda runtime, compute_dtype, context: cast(
            "MiniMaxH3DiTRuntime", runtime
        )._sampling_denoiser(compute_dtype, context),
        device=lambda runtime: (
            bound_compute_device(cast("MiniMaxH3DiTRuntime", runtime)._model.video_patch_proj)
            or cast("MiniMaxH3DiTRuntime", runtime)._model.video_patch_proj.weight.device
        ),
        compute_dtype=lambda runtime: cast("MiniMaxH3DiTRuntime", runtime)._compute_dtype,
        flow=True,
        pipeline=SamplingPipelineHooks(
            prepare_mask=_prepare_h3_mask,
            bind_attention=_bind_h3_attention,
            packed_context_windows=_h3_packed_context_windows,
            encode_conditioning=_encode_h3_conditioning,
            materialize_conditioning=_materialize_h3_conditioning,
        ),
    )

    def __init__(
        self,
        model: MiniMaxH3DiT,
        *,
        model_role: str,
        runtime_identity: str,
        receipt_identity: str | None = None,
        conditioning_identity: str | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        assembled: AssembledMiniMaxH3Model | None = None,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        if model_role not in ("fl2va_dit", "ref2va_dit"):
            raise ValueError("H3 DiT component role must be fl2va_dit or ref2va_dit")
        if type(runtime_identity) is not str or not runtime_identity:
            raise ValueError("H3 DiT runtime identity must be non-empty")
        if conditioning_identity is not None and (
            type(conditioning_identity) is not str or not conditioning_identity
        ):
            raise ValueError("H3 DiT conditioning identity must be non-empty")
        if compute_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("H3 DiT compute dtype must be bfloat16 or float32")
        if assembled is None:
            assembled = AssembledMiniMaxH3Model(
                model,
                _component_compute_dtypes=MappingProxyType({"diffusion": compute_dtype}),
            )
        elif assembled.diffusion is not model:
            raise ValueError("H3 DiT runtime assembly must own its model")
        self._model = model
        self.assembled = assembled
        self._model_role = model_role
        self._component_identity = runtime_identity
        self._runtime_identity = runtime_identity
        self._receipt_identity = receipt_identity
        self._conditioning_identity = (
            runtime_identity if conditioning_identity is None else conditioning_identity
        )
        self._compute_dtype = compute_dtype
        self._sigmas = MINIMAX_H3_SIGMAS
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )

    @property
    def config(self) -> MiniMaxH3Config:
        return MINIMAX_H3_CONFIG

    @property
    def family(self) -> ModelFamily:
        return MINIMAX_H3

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def with_conditioner(self, conditioning_identity: str) -> MiniMaxH3DiTRuntime:
        if type(conditioning_identity) is not str or not conditioning_identity:
            raise ValueError("H3 DiT conditioning identity must be non-empty")
        composition = compose_execution(
            MINIMAX_H3_CONFIG.family_id,
            {
                self._model_role: self._component_identity,
                "conditioner": conditioning_identity,
            },
        )
        derived = copy(self)
        derived._runtime_identity = composition.execution_identity
        derived._conditioning_identity = conditioning_identity
        return derived

    @property
    def conditioning_identity(self) -> str:
        return self._conditioning_identity

    @property
    def receipt_identity(self) -> str | None:
        return self._receipt_identity

    def with_sampling_space(self, space: SigmaSpace) -> MiniMaxH3DiTRuntime:
        if type(space) is not MiniMaxH3Sigmas:
            raise MiniMaxH3RuntimeError("MiniMax H3 sampling override requires MiniMaxH3Sigmas")
        derived = copy(self)
        derived._sigmas = space
        return derived

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return self._sigmas.video

    def adapt_multistream_latent(
        self,
        latent: torch.Tensor,
        *,
        source_spatial_downscale: int | None = None,
        source_temporal_downscale: int | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        return _adapt_minimax_h3_empty_latent(
            latent,
            source_spatial_downscale=source_spatial_downscale,
            source_temporal_downscale=source_temporal_downscale,
        )

    def _sampling_denoiser(
        self,
        compute_dtype: torch.dtype,
        context: SamplingAdapterContext,
    ) -> SamplingDenoiserExecution:
        if (
            context.inputs is None
            or context.device is None
            or context.plan is None
            or context.request is None
            or context.sampler is None
            or context.schedule is None
        ):
            raise RuntimeError("MiniMax H3 sampling context is unresolved")
        inputs = context.inputs
        latent_context = cast("_H3SamplingContext", inputs.latent_context)
        conditioning = latent_context.conditioning
        guidance_plan = context.plan
        request = context.request
        seed = context.seed
        cancelled = context.cancelled
        observer = cast("ExecutionObserverAttachment | None", context.observer)
        parent_span_id = context.parent_span_id
        device = torch.device(context.device)
        attention_binding = cast("_H3AttentionBinding", context.attention_binding)
        attention_kernel_factory = attention_binding.kernel_factory
        sparse_attention = (
            None
            if attention_binding.sparse is None
            else MiniMaxH3SparseAttention(attention_binding.sparse, self._sigmas)
        )
        if (
            attention_binding.sparse is not None
            and attention_binding.sparse.selection == "vsa"
            and not self._model.gate_compress
        ):
            logging.warning(
                "VSA: the model has no to_gate_compress layers; "
                "running the fine stage without the coarse branch"
            )
        scheduler_label = request.source_scheduler_id or "custom"
        model_role = self._model_role
        model = self._model
        requested_distributed = distributed_sampling_config()
        requested_sequence = (
            requested_distributed is not None and requested_distributed.mode == "sequence"
        )
        control = cast("MiniMaxH3FunControlConditioning | None", context.options.get("control"))
        if requested_sequence and control is not None:
            raise MiniMaxH3RuntimeError("MiniMax H3 control does not support sequence sharding")
        if requested_sequence and attention_kernel_factory is not None:
            raise MiniMaxH3RuntimeError(
                "single-job sequence mode owns the attention kernel factory"
            )
        if requested_sequence and sparse_attention is not None:
            raise MiniMaxH3RuntimeError(
                "single-job sequence mode does not support BlockSparseAttention"
            )
        sigmas = self._sigmas
        if requested_sequence:
            assert requested_distributed is not None
            if requested_distributed.sequence_guidance not in (1, 2):
                raise MiniMaxH3RuntimeError("single-job H3 sequence guidance degree must be 1 or 2")
            if (
                requested_distributed.sequence_guidance == 2
                and not guidance_plan.needs_unconditional
            ):
                raise MiniMaxH3RuntimeError(
                    "single-job sequence guidance requires conditional and unconditional lanes"
                )
        packed = inputs.latent
        layout = latent_context.layout
        raw_denoise_mask = (
            None
            if latent_context.raw_mask is None
            else unpack_latent_streams(latent_context.raw_mask, layout)
        )
        model_denoise_mask = latent_context.model_mask
        distributed = ensure_process_group()
        from .distributed import synchronized_sampling_call

        synchronized_sampling_call(lambda: _check_cancelled(cancelled), device, "cancellation")
        use_guidance = distributed is not None and distributed.mode != "sequence"
        use_sequence = distributed is not None and distributed.mode == "sequence"
        if use_sequence:
            assert distributed is not None
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            local_device_identity = str(device)
            if visible_devices is not None:
                local_device_identity += f";CUDA_VISIBLE_DEVICES={visible_devices}"
            rank_device_bindings = gather_sequence_device_bindings(
                distributed,
                torch.device(device),
                local_device_identity,
            )
        else:
            rank_device_bindings = ()
        schedule = context.schedule.sigmas

        def device_branch(
            prepared: MiniMaxH3PreparedConditioning,
            lane_identity: str,
        ) -> _H3EvaluationCondition:
            dit = replace(
                prepared.dit,
                text_token_tags=(
                    None
                    if prepared.dit.text_token_tags is None
                    else prepared.dit.text_token_tags.to(device)
                ),
                keyframes=tuple(
                    replace(keyframe, video=keyframe.video.to(device))
                    for keyframe in prepared.dit.keyframes
                ),
                guides=tuple(
                    replace(guide, latent=guide.latent.map(lambda tensor: tensor.to(device)))
                    for guide in prepared.dit.guides
                ),
                references=tuple(
                    replace(
                        reference,
                        video=None if reference.video is None else reference.video.to(device),
                        audio=None if reference.audio is None else reference.audio.to(device),
                    )
                    for reference in prepared.dit.references
                ),
                seed=seed,
            )
            text_context = model.preprocess_text_embeddings(
                prepared.context.to(device=device, dtype=self._compute_dtype)
            )
            return (
                lane_identity,
                text_context,
                dit,
                layout,
                model_denoise_mask,
                raw_denoise_mask,
            )

        def prepare_conditioning(value: object, role: GuidanceRole) -> _H3EvaluationCondition:
            if type(value) is not MiniMaxH3PreparedConditioning:
                raise TypeError("H3 guidance lanes require MiniMaxH3PreparedConditioning")
            if value.task is not conditioning.task:
                raise MiniMaxH3RuntimeError("H3 guidance lane targets a different task")
            if value.target_layout != layout:
                raise MiniMaxH3RuntimeError("H3 guidance lane belongs to a different target")
            lane_identity = "unconditional" if role is GuidanceRole.UNCONDITIONAL else "conditional"
            return device_branch(value, lane_identity)

        distributed = distributed_sampling_config()
        sequence_kernels: dict[
            tuple[MiniMaxH3PackedSequenceFacts, str], SequenceParallelAttentionKernel
        ] = {}

        def evaluate(
            x: torch.Tensor,
            sigma: float,
            conditioning: _H3EvaluationCondition,
        ) -> torch.Tensor:
            (
                lane_identity,
                text_context,
                dit_conditioning,
                active_layout,
                active_model_mask,
                active_raw_mask,
            ) = conditioning
            _check_cancelled(cancelled)
            if sigma <= 0.0:
                raise MiniMaxH3RuntimeError("H3 DiT cannot be evaluated at sigma zero")
            with execution_span(
                observer,
                "sample",
                "model_evaluation",
                parent_span_id=parent_span_id,
                component_role="diffusion",
                device=str(x.device),
            ):
                active_control = (
                    None
                    if control is None
                    else control.patch_for_sigma(
                        sigma,
                        lambda percent: self.custom_sampling_percent_to_sigma(
                            percent,
                            return_actual_sigma=True,
                        ),
                    )
                )
                local_rank_zero = rank_zero_sampling_active()
                selected_block_attention: MiniMaxH3BlockAttentionFactory | None = None
                if sparse_attention is not None:
                    active_sparse_attention = sparse_attention

                    def bind_block_attention(
                        facts: MiniMaxH3PackedSequenceFacts,
                    ) -> MiniMaxH3BlockAttention:
                        return active_sparse_attention.bind(
                            sigma=sigma,
                            lane=lane_identity,
                            facts=facts,
                        )

                    selected_block_attention = bind_block_attention

                if use_sequence and not local_rank_zero:
                    assert distributed is not None
                    preflight_error: BaseException | None = None
                    av: MultiStreamLatent[torch.Tensor] | None = None
                    facts: MiniMaxH3PackedSequenceFacts | None = None
                    compiled_plan = None
                    manifest = None
                    inner_kernel = None
                    sequence_group_ranks: tuple[int, ...] = ()
                    try:
                        av = unpack_latent_streams(x.to(dtype=compute_dtype), active_layout)
                        facts = _packed_sequence_facts(
                            av,
                            text_context,
                            dit_conditioning,
                            model.config.patch,
                        )
                        model._validate_inputs(  # pyright: ignore[reportPrivateUsage]
                            av,
                            sigma,
                            text_context,
                            dit_conditioning,
                            None,
                            active_model_mask,
                        )
                        partition = plan_sequence_partition(
                            facts.sequence_length,
                            distributed.sequence_ulysses * distributed.sequence_ring,
                        )
                        mesh = UspMesh.build(
                            guidance=distributed.sequence_guidance,
                            ulysses=distributed.sequence_ulysses,
                            ring=distributed.sequence_ring,
                        )
                        placement = PlacementMap.identity(mesh.process_mesh)
                        attention = cast(MiniMaxH3Attention, model.blocks[0].attn)
                        provider_identity = _h3_attention_backend_identity(attention)
                        exchange_identity = exchange_backend_identity(mesh)
                        inner_kernel = attention.attention_kernel
                        compiled_plan = compile_sequence_parallel_plan(
                            mesh=mesh,
                            placement=placement,
                            partition=partition,
                            packed_sequence_facts=facts,
                            head_count=model.config.attention_heads,
                            rank_device_bindings=rank_device_bindings,
                            routed_attention_backend_identity=provider_identity,
                            exchange_backend=exchange_identity,
                            attention_kernel=inner_kernel,
                            partition_compatibility=cast(
                                PartitionCompatibility,
                                getattr(inner_kernel, "partition_compatibility", None),
                            ),
                            compute_dtype={
                                torch.bfloat16: BFLOAT16,
                                torch.float16: FLOAT16,
                                torch.float32: FLOAT32,
                            }[self._compute_dtype],
                            device_kind=x.device.type,
                        )
                        slot = build_usp_compiled_plan_slot(compiled_plan)
                        sequence_size = mesh.ulysses * mesh.ring
                        group_start = mesh.coordinates(distributed.rank).guidance * sequence_size
                        sequence_group_ranks = tuple(
                            range(group_start, group_start + sequence_size)
                        )
                        manifest = build_canonical_manifest(
                            runtime_identity=self.runtime_identity,
                            invocation_facts=(
                                f"model_role={model_role}",
                                "lane_assignment="
                                + (
                                    "conditional,unconditional"
                                    if guidance_plan.needs_unconditional
                                    else "conditional"
                                ),
                                f"sampler={request.sampler.id}",
                                f"scheduler={scheduler_label}",
                                f"seed={seed}",
                                "sigma_table=" + ",".join(float(value).hex() for value in schedule),
                                f"packed_sequence={compiled_plan.packing_geometry_digest}",
                            ),
                            slots=(slot,),
                            rank_plan_digests=tuple(
                                (compiled_plan.rank_plan_digests[rank],)
                                for rank in sequence_group_ranks
                            ),
                        )
                    except BaseException as error:
                        preflight_error = error
                    if sequence_preflight_failed(preflight_error is not None, x.device):
                        if preflight_error is not None:
                            raise preflight_error
                        raise MiniMaxH3RuntimeError("peer H3 sequence preflight failed")
                    assert av is not None and facts is not None
                    assert compiled_plan is not None and manifest is not None
                    assert inner_kernel is not None
                    transport = SequenceDigestConsensusTransport(distributed, x.device)
                    if transport.physical_ranks != sequence_group_ranks:
                        raise MiniMaxH3RuntimeError(
                            "compiled sequence group differs from the consensus sideband"
                        )
                    consensus_token = prove_manifest_consensus(
                        manifest,
                        rank=transport.rank,
                        transport=transport,
                    )
                    mesh = compiled_plan.mesh
                    partition = compiled_plan.partition
                    coordinate = mesh.coordinates(distributed.rank)
                    shard_index = coordinate.ulysses * mesh.ring + coordinate.ring
                    shard = partition.shards[shard_index]
                    kernel_key = (facts, consensus_token.manifest_digest)
                    active_kernel = sequence_kernels.get(kernel_key)
                    if active_kernel is None:
                        active_kernel = SequenceParallelAttentionKernel(
                            mesh,
                            inner_kernel,
                            compiled_plan.layouts[distributed.rank],
                            compiled_plan.placement,
                            consensus_token,
                            compiled_plan.routed_attention_backend_identity,
                            compiled_plan.exchange_backend_identity,
                        )
                        sequence_kernels[kernel_key] = active_kernel

                    def sequence_factory(
                        factory_facts: MiniMaxH3PackedSequenceFacts,
                    ) -> SequenceParallelAttentionKernel:
                        if factory_facts != facts:
                            raise MiniMaxH3RuntimeError(
                                "H3 runtime sequence facts disagree with the DiT invocation"
                            )
                        return active_kernel

                    def sequence_gather(
                        hidden: torch.Tensor,
                        gather_partition: SequencePartition,
                        gather_shard: SequenceShard,
                    ) -> torch.Tensor:
                        return gather_sequence_hidden(
                            active_kernel,
                            hidden,
                            gather_partition,
                            gather_shard,
                        )

                    sharding = MiniMaxH3SequenceSharding(
                        facts,
                        partition,
                        shard,
                        sequence_gather,
                    )
                    velocity = model(
                        av,
                        sigma,
                        text_context,
                        conditioning=dit_conditioning,
                        sigmas=sigmas,
                        sampler_sigmas=schedule,
                        control=active_control,
                        denoise_mask=active_model_mask,
                        attention_kernel_factory=sequence_factory,
                        sequence_sharding=sharding,
                    )
                else:
                    av = unpack_latent_streams(x.to(dtype=compute_dtype), active_layout)
                    if attention_kernel_factory is None:
                        velocity = model(
                            av,
                            sigma,
                            text_context,
                            conditioning=dit_conditioning,
                            sigmas=sigmas,
                            sampler_sigmas=schedule,
                            control=active_control,
                            denoise_mask=active_model_mask,
                            **(
                                {}
                                if selected_block_attention is None
                                else {"block_attention_factory": selected_block_attention}
                            ),
                        )
                    else:
                        velocity = model(
                            av,
                            sigma,
                            text_context,
                            conditioning=dit_conditioning,
                            sigmas=sigmas,
                            sampler_sigmas=schedule,
                            control=active_control,
                            denoise_mask=active_model_mask,
                            attention_kernel_factory=attention_kernel_factory,
                            **(
                                {}
                                if selected_block_attention is None
                                else {"block_attention_factory": selected_block_attention}
                            ),
                        )
                if active_raw_mask is not None:
                    # H3 predicts video rows at mask * sigma; scale velocity before
                    # the shared outer conversion applies x0 = x - sigma * velocity.
                    video_velocity = velocity.by_role("video")
                    velocity = _h3_latent(
                        video_velocity
                        * active_raw_mask.by_role("video").to(
                            device=video_velocity.device,
                            dtype=video_velocity.dtype,
                        ),
                        velocity.by_role("audio"),
                    )
                packed_velocity, _ = pack_latent_streams(velocity)
                result = x - packed_velocity.float() * sigma
            _check_cancelled(cancelled)
            return result

        def conditioning_layout(value: object) -> ModelTokenLayout | None:
            if type(value) is not MiniMaxH3PreparedConditioning or value.token_layout is None:
                return None
            return value.token_layout.layout

        def conditioning_transforms(value: object) -> tuple[TokenGridTransform, ...]:
            if type(value) is not MiniMaxH3PreparedConditioning or value.token_layout is None:
                return ()
            return value.token_layout.transforms

        sampler_latent = unpack_latent_streams(packed, layout)
        packed_windows = _h3_packed_context_windows(self, inputs)

        def window_conditioning(
            prepared: _H3EvaluationCondition,
            dim: int,
            indices: tuple[int, ...],
            _shape: object,
        ) -> _H3EvaluationCondition:
            (
                lane,
                text_context,
                dit,
                prepared_layout,
                prepared_mask,
                prepared_raw_mask,
            ) = prepared
            if prepared_layout != layout:
                raise MiniMaxH3RuntimeError("H3 window conditioning must start at the full layout")
            selection = packed_windows.select(packed, dim, indices)
            window_masks: list[MultiStreamLatent[torch.Tensor] | None] = []
            for mask in (prepared_mask, prepared_raw_mask):
                if mask is None:
                    window_masks.append(None)
                    continue
                packed_mask, mask_layout = pack_latent_streams(mask)
                if mask_layout != layout:
                    raise MiniMaxH3RuntimeError("H3 model mask topology differs from the latent")
                selected_mask = packed_windows.select(packed_mask, dim, indices)
                if selected_mask.layout != selection.layout:
                    raise MiniMaxH3RuntimeError("H3 window mask topology differs from the latent")
                window_masks.append(
                    unpack_latent_streams(selected_mask.packed, selected_mask.layout)
                )
            return (
                lane,
                text_context,
                dit,
                selection.layout,
                window_masks[0],
                window_masks[1],
            )

        realization = context.conditioning_realization
        close = None
        if realization is None:
            denoiser_evaluator: object = model
            evaluator = ConditioningEvaluation(
                prepare_conditioning,
                evaluate,
                evaluator_identity=lambda _role: "dinkster.minimax-h3.conditioning.v1",
                standard_activation_memory_factor=self.family.memory_factor,
                layout=conditioning_layout,
                token_transforms=conditioning_transforms,
                window_conditioning=window_conditioning,
                validate_layout=lambda prepared, declared: _validate_h3_model_token_layout(
                    sampler_latent,
                    prepared[1],
                    prepared[2],
                    self.config.patch,
                    declared,
                ),
            )
        else:
            from .scheduled_sampling import FullLatentScheduledConditioningDenoiser

            def project_region(
                region: MaterializedRegion,
                x: torch.Tensor,
                window_dim: int | None,
                window_indices: tuple[int, ...],
            ) -> tuple[object, torch.Tensor]:
                prepared = region.family_payload
                if type(prepared) is not MiniMaxH3PreparedConditioning:
                    raise TypeError("H3 scheduled region requires prepared conditioning")
                video = sampler_latent.by_role("video")
                spatial = full_region_multiplier(
                    region,
                    batch=video.shape[0],
                    channels=video.shape[1],
                    device=video.device,
                )
                video_multiplier = spatial.unsqueeze(2).expand_as(video)
                audio = sampler_latent.by_role("audio")
                audio_multiplier = (
                    torch.zeros_like(audio)
                    if region.area is not None or region.mask is not None
                    else torch.ones_like(audio)
                )
                multiplier, multiplier_layout = pack_latent_streams(
                    _h3_latent(video_multiplier, audio_multiplier)
                )
                if multiplier_layout != layout:
                    raise MiniMaxH3RuntimeError(
                        "H3 regional multiplier topology differs from the latent"
                    )
                if window_dim is not None:
                    selected = packed_windows.select(multiplier, window_dim, window_indices)
                    multiplier = selected.packed
                if multiplier.shape != x.shape:
                    raise MiniMaxH3RuntimeError(
                        "H3 regional multiplier window differs from the latent"
                    )
                return (prepared, window_dim, window_indices), multiplier

            def evaluate_region(
                x: torch.Tensor,
                sigma: float,
                prepared: object,
                role: GuidanceRole,
            ) -> torch.Tensor:
                if (
                    not isinstance(prepared, tuple)
                    or len(prepared) != 3
                    or type(prepared[0]) is not MiniMaxH3PreparedConditioning
                    or prepared[1] is not None
                    and type(prepared[1]) is not int
                    or not isinstance(prepared[2], tuple)
                ):
                    raise TypeError("H3 scheduled window preparation is invalid")
                value, window_dim, window_indices = prepared
                condition = prepare_conditioning(value, role)
                if window_dim is not None:
                    condition = window_conditioning(
                        condition,
                        cast("int", window_dim),
                        cast("tuple[int, ...]", window_indices),
                        tuple(x.shape),
                    )
                return evaluate(x, sigma, condition)

            scheduled_evaluator = FullLatentScheduledConditioningDenoiser(
                space=sigmas.video,
                model=model,
                evaluate=evaluate_region,
                project=project_region,
                patch_sets=cast("Mapping[str, Any]", realization.patch_sets),
                compute_dtype=compute_dtype,
                device=device,
                cancel=cancelled,
            )
            denoiser_evaluator = scheduled_evaluator
            evaluator = ConditioningEvaluation(
                scheduled_evaluator.prepare_conditioning,
                scheduled_evaluator.evaluate_conditioning,
                scheduled_evaluator.batchable,
                scheduled_evaluator.evaluate_conditioning_batch,
                evaluator_identity=lambda _role: scheduled_evaluator.evaluator_identity,
                standard_activation_memory_factor=self.family.memory_factor,
                window_conditioning=scheduled_evaluator.window_conditioning,
            )
            close = scheduled_evaluator.close
        replica_group_size = (
            distributed.sequence_ulysses * distributed.sequence_ring
            if use_sequence and distributed is not None and distributed.sequence_guidance == 2
            else None
        )
        return SamplingDenoiserExecution(
            cast("SamplingDenoiserAdapter", denoiser_evaluator),
            conditioning_evaluation=evaluator,
            replica_group_size=replica_group_size,
            replica_evaluation=use_guidance or replica_group_size is not None,
            distributed_evaluation=(
                distributed is not None and not use_guidance and replica_group_size is None
            ),
            sampling=MINIMAX_H3.sampling,
            percent_to_sigma=sigmas.percent_to_sigma,
            process_in=lambda value: value,
            process_out=lambda value: value,
            unpack_state=lambda value: unpack_latent_streams(value, layout),
            denoise_mask_prepared=True,
            inpaint_noise=(
                prepare_noise(
                    inputs.latent,
                    seed + 1,
                    context.noise_inds,
                )
                if latent_context.raw_mask is not None and context.sampler.random_inpaint_noise
                else inputs.noise
                if latent_context.raw_mask is not None
                else None
            ),
            packed_inpaint=(
                None
                if latent_context.raw_mask is None or latent_context.token_mask is None
                else PackedInpaintConfiguration(
                    latent_context.token_mask,
                    layout.by_role("video").elements,
                    sigmas.video.shift,
                    sigmas.audio_shift,
                    sigmas.audio_scale,
                )
            ),
            close=close,
        )

    sample_custom = sampling_execution


__all__ = [
    "MiniMaxH3AudioVaeRuntime",
    "MiniMaxH3ConditionerRuntime",
    "MiniMaxH3DiTRuntime",
    "MiniMaxH3PreparedConditioning",
    "MiniMaxH3RuntimeError",
    "MiniMaxH3VideoVaeRuntime",
    "add_minimax_h3_motion_context",
    "add_minimax_h3_timeline_guide",
    "empty_minimax_h3_av",
]
