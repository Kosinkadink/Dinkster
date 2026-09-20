"""Executable text-to-video and image-to-video runtime for core Wan profiles."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, cast

import torch
from dinkster_inference import (
    WAN21_ANIMATE2_SETTINGS_KEY,
    WAN21_CAUSAL_INITIAL_LATENT_KEY,
    WAN21_CODEC,
    WAN21_FLOW_RVS_CODEC,
    WAN21_HUMO_17B,
    WAN21_I2V_14B,
    WAN21_SCAIL_REPLACEMENT_KEY,
    WAN21_SIGMAS,
    WAN21_T2V_14B,
    WAN22_CODEC,
    WAN22_DANCER_SETTINGS_KEY,
    WAN22_S2V_14B,
    WAN22_SIGMAS,
    WAN22_WANDANCER_14B,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    DualSamplingGuidance,
    ExecutionObserverAttachment,
    FlowSigmas,
    GuidanceRole,
    ModelFamily,
    ModelTokenLayout,
    MultiStreamFamilyRuntime,
    MultiStreamLatent,
    MultiStreamLatentAdapterRuntime,
    Parameterization,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PreparedMultiStreamConditioning,
    PromptTokenizer,
    Registry,
    SamplerDescriptor,
    SamplerInfo,
    SamplingCancelled,
    SamplingGuidance,
    SamplingStateCallback,
    SamplingStateEvent,
    SchedulerDescriptor,
    SolverStateEvent,
    StepCallback,
    StepEvent,
    TokenLayoutDescriptor,
    TokenLayoutError,
    TokenSegmentDescriptor,
    Wan21Animate2Settings,
    Wan21Config,
    Wan21PoseBlockCacheDevice,
    Wan21PoseBlockCacheSettings,
    Wan21VideoLatentGeometry,
    Wan22DancerSettings,
    decode_wan21_animate2_settings,
    decode_wan22_dancer_settings,
    encode_conditioning_carrier,
    encode_wan21_animate2_settings,
    encode_wan22_dancer_settings,
    make_conditioning_carrier,
    plan_wan21_token_layout,
    sampling_environment_cancellation,
    sampling_execution_context,
)

from ._conditioning_layout import DeclaredConditioning
from .assemble import AssembledWan21
from .codecs import CodecPlugin
from .context_windows import apply_freenoise, windowed_conditioning_evaluation
from .denoise import (
    FluxGuidance,
    prepare_multistream_noise,
    prepare_noise,
    run_sampler_engine,
    to_batch,
)
from .guidance import ConditioningEvaluation
from .operations import bound_compute_device, bound_compute_dtype
from .parameterizations import calculate_denoised, calculate_input
from .payloads import TensorPayloadError, payload_binding_to_tensor, tensor_to_payload_binding
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    brownian_step_noise,
    build_custom_sampling_schedule,
    compile_guidance_plan,
    guided_denoiser,
    narrow_single_stream_custom_sampling,
    resolve_custom_sampling_request,
)
from .sampling_runtime import MultiStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry
from .t5_text import T5TextEncoder
from .umt5_tokenizer import Umt5SentencePieceTokenizer
from .wan21_animate2 import PoseBranchCache, WanAnimate2Model
from .wan21_causal import Wan21CausalModel
from .wan21_humo import Wan21HumoModel
from .wan21_model import Wan21Model
from .wan21_scail import WanScailModel
from .wan21_uni3c import (
    Wan21Uni3CExecution,
    snapshot_wan21_uni3c_execution,
)
from .wan21_vae import LATENTS_MEAN, LATENTS_STD
from .wan22_dancer import Wan22DancerModel
from .wan22_s2v import Wan22S2VModel

if TYPE_CHECKING:
    from .wan21_multitalk import Wan21MultiTalkExecution


class Wan21RuntimeError(ValueError):
    """The requested operation is outside the loaded Wan profile."""


@dataclass(frozen=True, slots=True)
class Wan21InfiniteTalkExecution:
    """One identity-bound InfiniteTalk patch and motion-prefix application."""

    patch: Wan21MultiTalkExecution
    motion_latent: torch.Tensor
    extend: bool
    motion_digest: str

    def __post_init__(self) -> None:
        from .wan21_multitalk import (
            Wan21MultiTalkExecution,
            wan21_multitalk_tensor_digest,
        )

        if type(self.patch) is not Wan21MultiTalkExecution:
            raise TypeError("InfiniteTalk patch must be exact Wan21MultiTalkExecution")
        _validate_video_latent(self.motion_latent)
        if self.motion_latent.shape[0] != 1:
            raise Wan21RuntimeError("InfiniteTalk motion latent must have batch one")
        if type(self.extend) is not bool:
            raise TypeError("InfiniteTalk extend flag must be an exact bool")
        if wan21_multitalk_tensor_digest(self.motion_latent) != self.motion_digest:
            raise Wan21RuntimeError("InfiniteTalk motion latent identity changed")


def _snapshot_infinite_talk_execution(
    execution: Wan21InfiniteTalkExecution,
) -> Wan21InfiniteTalkExecution:
    from .wan21_multitalk import snapshot_wan21_multitalk_execution

    if type(execution) is not Wan21InfiniteTalkExecution:
        raise TypeError("multitalk must be exact Wan21InfiniteTalkExecution")
    return Wan21InfiniteTalkExecution(
        snapshot_wan21_multitalk_execution(execution.patch),
        execution.motion_latent.detach().clone(),
        execution.extend,
        execution.motion_digest,
    )


@dataclass(frozen=True, slots=True)
class Wan21PreparedConditioning:
    """Materialized Wan conditioning carried opaquely by the multistream seam."""

    text: torch.Tensor
    concat_latent: torch.Tensor | None = None
    vision: torch.Tensor | None = None
    vace_frames: tuple[torch.Tensor, ...] = ()
    vace_masks: tuple[torch.Tensor, ...] = ()
    vace_strengths: tuple[float, ...] = ()
    concat_mask_index: int | None = None
    reference_latent: torch.Tensor | None = None
    camera_conditions: torch.Tensor | None = None
    temporal_reference: torch.Tensor | None = None
    context_latents: tuple[torch.Tensor, ...] = ()
    pose_latents: torch.Tensor | None = None
    face_pixel_values: torch.Tensor | None = None
    pose_text: torch.Tensor | None = None
    pose_vision: torch.Tensor | None = None
    pose_schedule: PercentRange | None = None
    animate2_settings: Wan21Animate2Settings | None = None
    scail_reference_latent: torch.Tensor | None = None
    scail_reference_mask: torch.Tensor | None = None
    scail_driving_mask: torch.Tensor | None = None
    scail_replacement: bool = False
    audio_embed: torch.Tensor | None = None
    s2v_reference_latent: torch.Tensor | None = None
    s2v_reference_motion: torch.Tensor | None = None
    s2v_control_video: torch.Tensor | None = None
    humo_audio_embed: torch.Tensor | None = None
    humo_reference_latent: torch.Tensor | None = None
    dancer_audio_embed: torch.Tensor | None = None
    dancer_reference_vision: torch.Tensor | None = None
    dancer_settings: Wan22DancerSettings | None = None

    def __post_init__(self) -> None:
        if (
            type(self.text) is not torch.Tensor
            or not self.text.is_floating_point()
            or self.text.layout != torch.strided
        ):
            raise TypeError("Wan 2.1 TEXT must be an exact strided floating torch.Tensor")
        if self.text.ndim != 3 or any(size <= 0 for size in self.text.shape):
            raise Wan21RuntimeError("Wan 2.1 TEXT must be nonempty [B,tokens,channels]")
        if self.concat_latent is not None:
            _validate_concat_latent(self.concat_latent)
            if self.concat_mask_index is not None and (
                type(self.concat_mask_index) is not int
                or self.concat_mask_index < 0
                or self.concat_mask_index + 4 > self.concat_latent.shape[1]
            ):
                raise Wan21RuntimeError("Wan concat mask index must select four channels")
        elif self.concat_mask_index is not None:
            raise Wan21RuntimeError("Wan concat mask index requires CONCAT_LATENT")
        if self.vision is not None:
            _validate_vision(self.vision)
        if self.reference_latent is not None:
            _validate_reference_latent(self.reference_latent)
        if self.camera_conditions is not None:
            _validate_camera_conditions(self.camera_conditions)
        if self.temporal_reference is not None:
            _validate_video_latent(self.temporal_reference)
        if type(self.context_latents) is not tuple:
            raise TypeError("Wan Bernini context latents must be an exact tuple")
        for latent in self.context_latents:
            _validate_video_latent(latent)
        if self.pose_latents is not None:
            _validate_video_latent(self.pose_latents)
        if self.face_pixel_values is not None:
            _validate_face_pixel_values(self.face_pixel_values)
        if self.pose_text is not None:
            _validate_text(self.pose_text, text_dim=self.text.shape[2])
        if self.pose_vision is not None:
            _validate_vision(self.pose_vision, allowed_rows=(257,))
        if self.pose_schedule is not None and type(self.pose_schedule) is not PercentRange:
            raise TypeError("Wan Animate2 pose schedule must be an exact PercentRange")
        if (
            self.animate2_settings is not None
            and type(self.animate2_settings) is not Wan21Animate2Settings
        ):
            raise TypeError("Wan Animate2 settings must be exact Wan21Animate2Settings")
        if self.scail_reference_latent is not None:
            _validate_video_latent(self.scail_reference_latent)
        if self.scail_reference_mask is not None:
            _validate_video_latent(self.scail_reference_mask, channels=28)
        if self.scail_driving_mask is not None:
            _validate_video_latent(self.scail_driving_mask, channels=28)
        if type(self.scail_replacement) is not bool:
            raise TypeError("Wan SCAIL replacement mode must be an exact bool")
        if self.audio_embed is not None:
            _validate_s2v_audio(self.audio_embed)
        if self.s2v_reference_latent is not None:
            latent = _validate_video_latent(self.s2v_reference_latent)
            if latent.shape[2] != 1:
                raise Wan21RuntimeError("Wan S2V reference latent must contain one frame")
        if self.s2v_reference_motion is not None:
            _validate_video_latent(self.s2v_reference_motion)
        if self.s2v_control_video is not None:
            _validate_video_latent(self.s2v_control_video)
        if self.humo_audio_embed is not None:
            _validate_humo_audio(self.humo_audio_embed)
        if self.humo_reference_latent is not None:
            _validate_video_latent(self.humo_reference_latent)
        if self.dancer_audio_embed is not None:
            _validate_dancer_audio(self.dancer_audio_embed)
        if self.dancer_reference_vision is not None:
            _validate_vision(self.dancer_reference_vision, allowed_rows=(257,))
        if (
            self.dancer_settings is not None
            and type(self.dancer_settings) is not Wan22DancerSettings
        ):
            raise TypeError("WanDancer settings must be exact Wan22DancerSettings")
        if not (
            type(self.vace_frames) is tuple
            and type(self.vace_masks) is tuple
            and type(self.vace_strengths) is tuple
            and len(self.vace_frames) == len(self.vace_masks) == len(self.vace_strengths)
        ):
            raise Wan21RuntimeError("Wan VACE payloads must have matching exact tuples")
        for index, (frames, mask, strength) in enumerate(
            zip(self.vace_frames, self.vace_masks, self.vace_strengths, strict=True)
        ):
            frames = _validate_vace_tensor(frames, channels=32, name=f"VACE_FRAMES[{index}]")
            mask = _validate_vace_tensor(mask, channels=64, name=f"VACE_MASK[{index}]")
            if frames.shape[0] != mask.shape[0] or frames.shape[2:] != mask.shape[2:]:
                raise Wan21RuntimeError("Wan VACE frame and mask geometry must match")
            if type(strength) is not float or not math.isfinite(strength) or strength < 0.0:
                raise Wan21RuntimeError("Wan VACE strengths must be finite non-negative floats")


def _text_only_prepared_conditioning(value: Wan21PreparedConditioning) -> bool:
    """Whether every field other than the text embedding is at its default.

    Structural conditioning (concat latents, references, VACE, camera, audio,
    pose) is frame-aligned to the full video, so a windowed model application
    would pair a sliced latent with unsliced extras."""
    for spec in fields(value):
        if spec.name == "text":
            continue
        current = getattr(value, spec.name)
        if spec.default is None:
            if current is not None:
                return False
        elif current != spec.default:
            return False
    return True


def _validate_video_latent(latent: object, *, channels: int = 16) -> torch.Tensor:
    if (
        type(latent) is not torch.Tensor
        or not latent.is_floating_point()
        or latent.layout != torch.strided
    ):
        raise TypeError("Wan 2.1 latent must be an exact strided floating torch.Tensor")
    if latent.ndim != 5 or latent.shape[1] != channels or any(size <= 0 for size in latent.shape):
        raise Wan21RuntimeError(f"Wan video latent must be nonempty [B,{channels},T,H,W]")
    return latent


def _validate_s2v_audio(value: object) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan S2V audio must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 4
        or value.shape[1:3] != (25, 1024)
        or value.shape[0] <= 0
        or value.shape[3] <= 0
    ):
        raise Wan21RuntimeError("Wan S2V audio must be nonempty [B,25,1024,samples]")
    return value


def _validate_humo_audio(value: object) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan HuMo audio must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 5
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or value.shape[2:] != (8, 5, 1280)
    ):
        raise Wan21RuntimeError("Wan HuMo audio must be nonempty [B,frames,8,5,1280]")
    return value


def _validate_dancer_audio(value: object) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("WanDancer audio must be an exact strided floating torch.Tensor")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] != 35:
        raise Wan21RuntimeError("WanDancer audio must be nonempty [B,time,35]")
    return value


def _validate_text(value: object, *, text_dim: int) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan 2.1 TEXT must be an exact strided floating torch.Tensor")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] != text_dim:
        raise Wan21RuntimeError(f"Wan 2.1 TEXT must be nonempty [B,tokens,{text_dim}]")
    return value


def _validate_concat_latent(value: object, *, channels: int | None = None) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan 2.1 CONCAT_LATENT must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 5
        or (channels is not None and value.shape[1] != channels)
        or any(size <= 0 for size in value.shape)
    ):
        expected = "channels" if channels is None else str(channels)
        raise Wan21RuntimeError(f"Wan CONCAT_LATENT must be nonempty [B,{expected},T,H,W]")
    return value


def _validate_reference_latent(value: object, *, channels: int | None = None) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan REFERENCE_LATENT must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 5
        or (channels is not None and value.shape[1] != channels)
        or any(size <= 0 for size in value.shape)
    ):
        expected = "channels" if channels is None else str(channels)
        raise Wan21RuntimeError(f"Wan REFERENCE_LATENT must be nonempty [B,{expected},T,H,W]")
    return value


def _validate_vision(value: object, *, allowed_rows: tuple[int, ...] = (257, 514)) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan 2.1 VISION_EMBEDDING must be an exact strided floating torch.Tensor")
    expected = " or ".join(f"[B,{rows},1280]" for rows in allowed_rows)
    if (
        value.ndim != 3
        or value.shape[0] <= 0
        or value.shape[1] not in allowed_rows
        or value.shape[2] != 1280
    ):
        raise Wan21RuntimeError(f"Wan 2.1 VISION_EMBEDDING must have shape {expected}")
    return value


def _validate_vace_tensor(value: object, *, channels: int, name: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError(f"Wan {name} must be an exact strided floating torch.Tensor")
    if value.ndim != 5 or value.shape[1] != channels or any(size <= 0 for size in value.shape):
        raise Wan21RuntimeError(f"Wan {name} must be nonempty [B,{channels},T,H,W]")
    return value


def _validate_camera_conditions(value: object, *, channels: int = 24) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan CAMERA_CONDITIONS must be an exact strided floating torch.Tensor")
    if value.ndim != 5 or value.shape[1] != channels or any(size <= 0 for size in value.shape):
        raise Wan21RuntimeError(f"Wan CAMERA_CONDITIONS must be nonempty [B,{channels},T,H,W]")
    return value


def _validate_face_pixel_values(value: object) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("Wan FACE_PIXEL_VALUES must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 5
        or value.shape[1] != 3
        or value.shape[0] <= 0
        or value.shape[2] <= 0
        or value.shape[3:] != (512, 512)
    ):
        raise Wan21RuntimeError("Wan FACE_PIXEL_VALUES must have shape [B,3,T,512,512]")
    return value


def _allowed_vision_rows(config: Wan21Config) -> tuple[int, ...]:
    if config.flf_pos_embed_token_number is None:
        return (257,)
    return (257, config.flf_pos_embed_token_number)


def _concat_mask_index(config: Wan21Config) -> int | None:
    if config.camera_channels is not None:
        return 0
    extra_channels = config.in_channels - config.out_channels
    if extra_channels % config.out_channels == 0:
        return None
    latent_channels = extra_channels - 4
    if latent_channels == config.out_channels:
        return 0
    if latent_channels == config.out_channels * 2:
        return config.out_channels
    raise Wan21RuntimeError("Wan profile has an unsupported concat-channel layout")


def _resize_concat_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    """Match Wan's reference-latent batch selection exactly."""
    source_batch = tensor.shape[0]
    if source_batch == batch:
        return tensor
    if batch <= 1:
        return tensor[:batch]
    output = torch.empty(
        (batch, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if batch < source_batch:
        scale = (source_batch - 1) / (batch - 1)
        for index in range(batch):
            output[index] = tensor[min(round(index * scale), source_batch - 1)]
    else:
        scale = source_batch / batch
        for index in range(batch):
            output[index] = tensor[min(int((index + 0.5) * scale), source_batch - 1)]
    return output


def _normalize_concat_latent(
    value: torch.Tensor,
    *,
    latent_channels: int,
    mask_index: int | None,
    process_in: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if mask_index is None:
        latent_parts = value.split(latent_channels, dim=1)
        return torch.cat(tuple(process_in(part) for part in latent_parts), dim=1)
    before = value[:, :mask_index]
    mask = value[:, mask_index : mask_index + 4]
    after = value[:, mask_index + 4 :]
    normalized = tuple(
        process_in(part)
        for section in (before, after)
        for part in section.split(latent_channels, dim=1)
        if part.shape[1]
    )
    before_count = mask_index // latent_channels
    return torch.cat((*normalized[:before_count], mask, *normalized[before_count:]), dim=1)


_HUMO_ZERO_FIRST = (
    0.8660,
    -0.4326,
    -0.0017,
    -0.4884,
    -0.5283,
    0.9207,
    -0.9896,
    0.4433,
    -0.5543,
    -0.0113,
    0.5753,
    -0.6000,
    -0.8346,
    -0.3497,
    -0.1926,
    -0.6938,
)
_HUMO_ZERO_SECOND = (
    1.0869,
    -1.2370,
    0.0206,
    -0.4357,
    -0.6411,
    2.0307,
    -1.5972,
    1.2659,
    -0.8595,
    -0.4654,
    0.9638,
    -1.6330,
    -1.4310,
    -0.1098,
    -0.3856,
    -1.4583,
)
_HUMO_ZERO_LATER = (
    0.8642,
    -1.8583,
    0.1577,
    0.1350,
    -0.3641,
    2.5863,
    -1.9670,
    1.6065,
    -1.0475,
    -0.8678,
    1.1734,
    -1.8138,
    -1.5933,
    -0.7721,
    -0.3289,
    -1.3745,
)


def _humo_target_concat(video: torch.Tensor) -> torch.Tensor:
    concat = video.new_zeros(video.shape[0], 20, *video.shape[2:])
    concat[:, 4:] = video.new_tensor(_HUMO_ZERO_LATER).view(1, 16, 1, 1, 1)
    concat[:, 4:, :1] = video.new_tensor(_HUMO_ZERO_FIRST).view(1, 16, 1, 1, 1)
    if video.shape[2] > 1:
        concat[:, 4:, 1:2] = video.new_tensor(_HUMO_ZERO_SECOND).view(1, 16, 1, 1, 1)
    return concat


def _conditioning_text(value: object, *, text_dim: int) -> torch.Tensor:
    if type(value) not in (Conditioning, DeclaredConditioning):
        raise TypeError("Wan 2.1 guidance lanes require exact Conditioning values")
    conditioning = cast("Conditioning[torch.Tensor]", value)
    if conditioning.pooled is not None:
        raise Wan21RuntimeError("Wan 2.1 text conditioning does not accept pooled embeddings")
    return _validate_text(conditioning.embeddings, text_dim=text_dim)


def _text_carrier(value: object, *, text_dim: int, family_id: str) -> ConditioningCarrier:
    text = _conditioning_text(value, text_dim=text_dim)
    binding = tensor_to_payload_binding("wan21-text", text, space="conditioning-text")
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id),
        binding.shape,
        binding.dtype,
        binding.space,
    )
    token_count = int(text.shape[1])
    record = ConditioningRecord(
        channels=((ConditioningChannel.TEXT, descriptor),),
        token_layout=TokenLayoutDescriptor(
            family_id=family_id,
            version=1,
            text_streams=("umt5",),
            segments=(TokenSegmentDescriptor("umt5", "umt5", 0, token_count),),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (binding,))


def wan21_text_conditioning_to_carrier(value: object) -> ConditioningCarrier:
    """Bind independently encoded UMT5 conditioning to Wan 2.1."""

    return _text_carrier(value, text_dim=WAN21_T2V_14B.text_dim, family_id="dinkster.wan21")


def _i2v_carrier(
    prepared: Wan21PreparedConditioning,
    concat_latent: torch.Tensor,
    vision: torch.Tensor | None,
    *,
    family_id: str,
) -> ConditioningCarrier:
    text = prepared.text
    bindings = [
        tensor_to_payload_binding("wan21-text", text, space="conditioning-text"),
        tensor_to_payload_binding(
            "wan21-concat-latent", concat_latent, space="conditioning-concat-latent"
        ),
    ]
    channel_names = [ConditioningChannel.TEXT, ConditioningChannel.CONCAT_LATENT]
    if vision is not None:
        bindings.append(
            tensor_to_payload_binding(
                "wan21-vision-embedding", vision, space="conditioning-vision-embedding"
            )
        )
        channel_names.append(ConditioningChannel.VISION_EMBEDDING)
    channels = tuple(
        (
            channel,
            PayloadDescriptor(
                PayloadReference(binding.reference_id),
                binding.shape,
                binding.dtype,
                binding.space,
            ),
        )
        for channel, binding in zip(channel_names, bindings, strict=True)
    )
    token_count = int(text.shape[1])
    record = ConditioningRecord(
        channels=channels,
        token_layout=TokenLayoutDescriptor(
            family_id=family_id,
            version=1,
            text_streams=("umt5",),
            segments=(TokenSegmentDescriptor("umt5", "umt5", 0, token_count),),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def _basic_wan_text_payload(
    carrier: object,
    *,
    name: str,
) -> tuple[PayloadDescriptor, PayloadBinding, TokenLayoutDescriptor]:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError(f"{name} must be an exact ConditioningCarrier")
    typed = carrier
    encode_conditioning_carrier(typed)
    records = typed.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError(f"{name} requires exactly one conditioning record")
    source = records[0]
    if (
        source.area is not None
        or source.mask is not None
        or source.schedule != PercentRange(0.0, 1.0)
        or source.scale_vector is not None
        or source.extension_metadata
    ):
        raise Wan21RuntimeError(f"{name} requires one basic full-schedule record")
    source_channels = dict(source.channels)
    descriptor = source_channels.pop(ConditioningChannel.TEXT, None)
    if descriptor is None or source_channels:
        raise Wan21RuntimeError(f"{name} requires exactly one TEXT channel")
    if len(descriptor.shape) != 3 or any(size <= 0 for size in descriptor.shape):
        raise Wan21RuntimeError(f"{name} TEXT descriptor must be nonempty rank 3")
    layout = TokenLayoutDescriptor(
        family_id=(
            source.token_layout.family_id if source.token_layout is not None else "dinkster.wan21"
        ),
        version=1,
        text_streams=("umt5",),
        segments=(TokenSegmentDescriptor("umt5", "umt5", 0, descriptor.shape[1]),),
    )
    if source.token_layout is not None and source.token_layout != layout:
        raise Wan21RuntimeError(f"{name} text has an incompatible token layout")
    binding = next(
        (item for item in typed.bindings if item.reference_id == descriptor.reference.id),
        None,
    )
    if binding is None:
        raise Wan21RuntimeError(f"{name} TEXT channel has no payload binding")
    return descriptor, binding, layout


def compose_wan21_i2v_conditioning(
    text: ConditioningCarrier,
    *,
    concat_latent: torch.Tensor,
    vision: torch.Tensor | None = None,
) -> ConditioningCarrier:
    """Build one canonical base Wan 2.1 I2V conditioning lane."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="Wan 2.1 I2V text")
    concat = _validate_concat_latent(concat_latent, channels=20)
    values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = [
        (
            ConditioningChannel.CONCAT_LATENT,
            concat,
            "wan21-concat-latent",
            "conditioning-concat-latent",
        )
    ]
    if vision is not None:
        values.append(
            (
                ConditioningChannel.VISION_EMBEDDING,
                _validate_vision(vision, allowed_rows=(257,)),
                "wan21-vision-embedding",
                "conditioning-vision-embedding",
            )
        )
    bindings = [text_binding]
    channels = [(ConditioningChannel.TEXT, descriptor)]
    for channel, value, reference_id, space in values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet((ConditioningRecord(channels=tuple(channels), token_layout=layout),)),
        tuple(bindings),
    )


def compose_wan21_humo_conditioning(
    text: ConditioningCarrier,
    *,
    audio_embed: torch.Tensor,
    reference_latent: torch.Tensor,
) -> ConditioningCarrier:
    """Build one canonical Wan 2.1 HuMo conditioning lane."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="Wan HuMo text")
    values = (
        (
            ConditioningChannel.AUDIO_EMBEDDING,
            _validate_humo_audio(audio_embed),
            "wan21-humo-audio-embedding",
            "conditioning-humo-audio-embedding",
        ),
        (
            ConditioningChannel.REFERENCE_LATENT,
            _validate_video_latent(reference_latent),
            "wan21-humo-reference-latent",
            "conditioning-humo-reference-latent",
        ),
    )
    bindings = [text_binding]
    channels = [(ConditioningChannel.TEXT, descriptor)]
    for channel, value, reference_id, space in values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet((ConditioningRecord(channels=tuple(channels), token_layout=layout),)),
        tuple(bindings),
    )


def compose_wan22_dancer_conditioning(
    text: ConditioningCarrier,
    *,
    concat_latent: torch.Tensor | None = None,
    vision: torch.Tensor | None = None,
    reference_vision: torch.Tensor | None = None,
    audio_embed: torch.Tensor | None = None,
    settings: Wan22DancerSettings | None = None,
) -> ConditioningCarrier:
    """Build one canonical WanDancer conditioning lane."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="WanDancer text")
    if settings is None:
        settings = Wan22DancerSettings()
    elif type(settings) is not Wan22DancerSettings:
        raise TypeError("settings must be exact Wan22DancerSettings")
    values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = []
    if concat_latent is not None:
        values.append(
            (
                ConditioningChannel.CONCAT_LATENT,
                _validate_concat_latent(concat_latent, channels=20),
                "wan22-dancer-concat-latent",
                "conditioning-dancer-concat-latent",
            )
        )
    if vision is not None:
        values.append(
            (
                ConditioningChannel.VISION_EMBEDDING,
                _validate_vision(vision, allowed_rows=(257,)),
                "wan22-dancer-vision-embedding",
                "conditioning-dancer-vision-embedding",
            )
        )
    if reference_vision is not None:
        values.append(
            (
                ConditioningChannel.REFERENCE_VISION_EMBEDDING,
                _validate_vision(reference_vision, allowed_rows=(257,)),
                "wan22-dancer-reference-vision-embedding",
                "conditioning-dancer-reference-vision-embedding",
            )
        )
    if audio_embed is not None:
        values.append(
            (
                ConditioningChannel.AUDIO_EMBEDDING,
                _validate_dancer_audio(audio_embed),
                "wan22-dancer-audio-embedding",
                "conditioning-dancer-audio-embedding",
            )
        )
    bindings = [text_binding]
    channels = [(ConditioningChannel.TEXT, descriptor)]
    for channel, value, reference_id, space in values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet(
            (
                ConditioningRecord(
                    channels=tuple(channels),
                    token_layout=layout,
                    extension_metadata=(
                        (WAN22_DANCER_SETTINGS_KEY, encode_wan22_dancer_settings(settings)),
                    ),
                ),
            )
        ),
        tuple(bindings),
    )


def compose_wan22_s2v_conditioning(
    text: ConditioningCarrier,
    *,
    audio_embed: torch.Tensor | None,
    reference_latent: torch.Tensor | None,
    reference_motion: torch.Tensor | None,
    control_video: torch.Tensor,
) -> ConditioningCarrier:
    """Build one canonical Wan 2.2 S2V conditioning lane."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="Wan S2V text")
    values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = [
        (
            ConditioningChannel.CONTROL_VIDEO,
            _validate_video_latent(control_video),
            "wan22-s2v-control-video",
            "conditioning-control-video",
        )
    ]
    if audio_embed is not None:
        values.append(
            (
                ConditioningChannel.AUDIO_EMBEDDING,
                _validate_s2v_audio(audio_embed),
                "wan22-s2v-audio-embedding",
                "conditioning-audio-embedding",
            )
        )
    if reference_latent is not None:
        reference_latent = _validate_video_latent(reference_latent)
        if reference_latent.shape[2] != 1:
            raise Wan21RuntimeError("Wan S2V reference latent must contain one frame")
        values.append(
            (
                ConditioningChannel.REFERENCE_LATENT,
                reference_latent,
                "wan22-s2v-reference-latent",
                "conditioning-reference-latent",
            )
        )
    if reference_motion is not None:
        values.append(
            (
                ConditioningChannel.REFERENCE_MOTION,
                _validate_video_latent(reference_motion),
                "wan22-s2v-reference-motion",
                "conditioning-reference-motion",
            )
        )
    bindings = [text_binding]
    channels = [(ConditioningChannel.TEXT, descriptor)]
    for channel, value, reference_id, space in values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet((ConditioningRecord(channels=tuple(channels), token_layout=layout),)),
        tuple(bindings),
    )


def compose_wan21_animate2_conditioning(
    text: ConditioningCarrier,
    concat_latent: torch.Tensor,
    *,
    vision: torch.Tensor | None = None,
    pose_text: ConditioningCarrier | None = None,
    pose_vision: torch.Tensor | None = None,
    pose_latents: torch.Tensor | None = None,
    pose_schedule: PercentRange | None = None,
    settings: Wan21Animate2Settings | None = None,
) -> ConditioningCarrier:
    """Build the canonical two-record Wan Animate2 carrier."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="Wan Animate2 text")
    pose_descriptor, pose_source_binding, pose_layout = _basic_wan_text_payload(
        text if pose_text is None else pose_text,
        name="Wan Animate2 pose text",
    )
    if descriptor.shape[2] != pose_descriptor.shape[2]:
        raise Wan21RuntimeError("Wan Animate2 main and pose text widths must match")
    if pose_schedule is None:
        pose_schedule = PercentRange(0.0, 1.0)
    elif type(pose_schedule) is not PercentRange:
        raise TypeError("pose_schedule must be an exact PercentRange")
    if settings is None:
        settings = Wan21Animate2Settings()
    encoded_settings = encode_wan21_animate2_settings(settings)

    bindings = [text_binding]
    main_channels = [(ConditioningChannel.TEXT, descriptor)]
    main_values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = [
        (
            ConditioningChannel.CONCAT_LATENT,
            concat_latent,
            "wan21-concat-latent",
            "conditioning-concat-latent",
        )
    ]
    if vision is not None:
        main_values.append(
            (
                ConditioningChannel.VISION_EMBEDDING,
                vision,
                "wan21-vision-embedding",
                "conditioning-vision-embedding",
            )
        )
    for channel, value, reference_id, space in main_values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        main_channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )

    pose_text_tensor = payload_binding_to_tensor(pose_source_binding)
    pose_binding = tensor_to_payload_binding(
        "wan21-pose-text",
        pose_text_tensor,
        space="conditioning-pose-text",
    )
    bindings.append(pose_binding)
    pose_channels = [
        (
            ConditioningChannel.POSE_TEXT,
            PayloadDescriptor(
                PayloadReference(pose_binding.reference_id),
                pose_binding.shape,
                pose_binding.dtype,
                pose_binding.space,
            ),
        )
    ]
    pose_values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = []
    if pose_vision is not None:
        pose_values.append(
            (
                ConditioningChannel.POSE_VISION_EMBEDDING,
                pose_vision,
                "wan21-pose-vision-embedding",
                "conditioning-pose-vision-embedding",
            )
        )
    if pose_latents is not None:
        pose_values.append(
            (
                ConditioningChannel.POSE_LATENT,
                pose_latents,
                "wan21-pose-latent",
                "conditioning-pose-latent",
            )
        )
    for channel, value, reference_id, space in pose_values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        pose_channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet(
            (
                ConditioningRecord(channels=tuple(main_channels), token_layout=layout),
                ConditioningRecord(
                    channels=tuple(pose_channels),
                    schedule=pose_schedule,
                    token_layout=pose_layout,
                    extension_metadata=((WAN21_ANIMATE2_SETTINGS_KEY, encoded_settings),),
                ),
            )
        ),
        tuple(bindings),
    )


def compose_wan21_animate_conditioning(
    text: ConditioningCarrier,
    concat_latent: torch.Tensor,
    *,
    vision: torch.Tensor | None = None,
    pose_latents: torch.Tensor | None = None,
    face_pixel_values: torch.Tensor | None = None,
) -> ConditioningCarrier:
    """Build the exact Wan Animate carrier from one basic text record."""

    if type(text) is not ConditioningCarrier:
        raise TypeError("Wan Animate text must be an exact ConditioningCarrier")
    encode_conditioning_carrier(text)
    records = text.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError("Wan Animate text requires exactly one conditioning record")
    source = records[0]
    if (
        source.area is not None
        or source.mask is not None
        or source.schedule != PercentRange(0.0, 1.0)
        or source.scale_vector is not None
        or source.extension_metadata
    ):
        raise Wan21RuntimeError("Wan Animate text requires one basic full-schedule record")
    source_channels = dict(source.channels)
    descriptor = source_channels.pop(ConditioningChannel.TEXT, None)
    if descriptor is None or source_channels:
        raise Wan21RuntimeError("Wan Animate text requires exactly one TEXT channel")
    if len(descriptor.shape) != 3 or any(size <= 0 for size in descriptor.shape):
        raise Wan21RuntimeError("Wan Animate TEXT descriptor must be nonempty rank 3")
    expected_layout = TokenLayoutDescriptor(
        family_id=(
            source.token_layout.family_id if source.token_layout is not None else "dinkster.wan21"
        ),
        version=1,
        text_streams=("umt5",),
        segments=(TokenSegmentDescriptor("umt5", "umt5", 0, descriptor.shape[1]),),
    )
    if source.token_layout is not None and source.token_layout != expected_layout:
        raise Wan21RuntimeError("Wan Animate text has an incompatible token layout")
    text_binding = next(
        (binding for binding in text.bindings if binding.reference_id == descriptor.reference.id),
        None,
    )
    if text_binding is None:
        raise Wan21RuntimeError("Wan Animate TEXT channel has no payload binding")

    channel_values: list[tuple[ConditioningChannel, torch.Tensor, str, str]] = [
        (
            ConditioningChannel.CONCAT_LATENT,
            concat_latent,
            "wan21-concat-latent",
            "conditioning-concat-latent",
        )
    ]
    if vision is not None:
        channel_values.append(
            (
                ConditioningChannel.VISION_EMBEDDING,
                vision,
                "wan21-vision-embedding",
                "conditioning-vision-embedding",
            )
        )
    if pose_latents is not None:
        channel_values.append(
            (
                ConditioningChannel.POSE_LATENT,
                pose_latents,
                "wan21-pose-latent",
                "conditioning-pose-latent",
            )
        )
    if face_pixel_values is not None:
        channel_values.append(
            (
                ConditioningChannel.FACE_PIXELS,
                face_pixel_values,
                "wan21-face-pixels",
                "conditioning-face-pixels",
            )
        )
    bindings = [text_binding]
    channels = [(ConditioningChannel.TEXT, descriptor)]
    for channel, value, reference_id, space in channel_values:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )
    return make_conditioning_carrier(
        ConditioningSet(
            (
                ConditioningRecord(
                    channels=tuple(channels),
                    token_layout=expected_layout,
                ),
            )
        ),
        tuple(bindings),
    )


def compose_wan21_scail_conditioning(
    text: ConditioningCarrier,
    reference_latents: tuple[torch.Tensor, ...],
    *,
    vision: torch.Tensor | None = None,
    pose_latents: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    driving_mask: torch.Tensor | None = None,
    pose_schedule: PercentRange | None = None,
    replacement: bool = False,
) -> ConditioningCarrier:
    """Build the canonical SCAIL conditioning records."""

    descriptor, text_binding, layout = _basic_wan_text_payload(text, name="Wan SCAIL text")
    if type(reference_latents) is not tuple:
        raise TypeError("reference_latents must be an exact tuple")
    references = tuple(_validate_video_latent(value) for value in reference_latents)
    if references and any(
        value.shape[0] != references[0].shape[0] or value.shape[3:] != references[0].shape[3:]
        for value in references[1:]
    ):
        raise Wan21RuntimeError("Wan SCAIL reference latents must share batch and spatial shape")
    if type(replacement) is not bool:
        raise TypeError("replacement must be an exact bool")
    if pose_schedule is None:
        pose_schedule = PercentRange(0.0, 1.0)
    elif type(pose_schedule) is not PercentRange:
        raise TypeError("pose_schedule must be an exact PercentRange")
    if vision is not None:
        vision = _validate_vision(vision, allowed_rows=(257,))
    if pose_latents is not None:
        pose_latents = _validate_video_latent(pose_latents)
    if reference_mask is not None:
        reference_mask = _validate_video_latent(reference_mask, channels=28)
    if driving_mask is not None:
        driving_mask = _validate_video_latent(driving_mask, channels=28)
        if pose_latents is None:
            raise Wan21RuntimeError("Wan SCAIL driving mask requires pose latents")
    if reference_mask is not None and not references:
        raise Wan21RuntimeError("Wan SCAIL reference mask requires reference latents")

    bindings = [text_binding]
    main_channels = [(ConditioningChannel.TEXT, descriptor)]

    def add_channel(
        channels: list[tuple[ConditioningChannel, PayloadDescriptor]],
        channel: ConditioningChannel,
        value: torch.Tensor,
        reference_id: str,
        space: str,
    ) -> None:
        binding = tensor_to_payload_binding(reference_id, value, space=space)
        bindings.append(binding)
        channels.append(
            (
                channel,
                PayloadDescriptor(
                    PayloadReference(binding.reference_id),
                    binding.shape,
                    binding.dtype,
                    binding.space,
                ),
            )
        )

    if references:
        add_channel(
            main_channels,
            ConditioningChannel.SCAIL_REFERENCE_LATENT,
            torch.cat((*references[1:], references[0]), dim=2),
            "wan21-scail-reference-latent",
            "conditioning-scail-reference-latent",
        )
    if vision is not None:
        add_channel(
            main_channels,
            ConditioningChannel.VISION_EMBEDDING,
            vision,
            "wan21-vision-embedding",
            "conditioning-vision-embedding",
        )
    if reference_mask is not None:
        add_channel(
            main_channels,
            ConditioningChannel.SCAIL_REFERENCE_MASK,
            reference_mask,
            "wan21-scail-reference-mask",
            "conditioning-scail-reference-mask",
        )

    records = [
        ConditioningRecord(
            channels=tuple(main_channels),
            token_layout=layout,
            extension_metadata=((WAN21_SCAIL_REPLACEMENT_KEY, replacement),),
        )
    ]
    if pose_latents is not None:
        pose_channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = []
        add_channel(
            pose_channels,
            ConditioningChannel.POSE_LATENT,
            pose_latents,
            "wan21-pose-latent",
            "conditioning-pose-latent",
        )
        if driving_mask is not None:
            add_channel(
                pose_channels,
                ConditioningChannel.SCAIL_DRIVING_MASK,
                driving_mask,
                "wan21-scail-driving-mask",
                "conditioning-scail-driving-mask",
            )
        records.append(ConditioningRecord(channels=tuple(pose_channels), schedule=pose_schedule))
    return make_conditioning_carrier(ConditioningSet(tuple(records)), tuple(bindings))


def _materialize_dancer_conditioning(
    carrier: object,
    *,
    text_dim: int,
    family_id: str,
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("WanDancer conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"WanDancer conditioning carrier is invalid: {error}") from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError("WanDancer requires exactly one conditioning record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.schedule != PercentRange(0.0, 1.0)
        or record.scale_vector is not None
    ):
        raise Wan21RuntimeError("WanDancer requires one basic full-schedule record")
    metadata = dict(record.extension_metadata)
    if len(record.extension_metadata) != 1 or set(metadata) != {WAN22_DANCER_SETTINGS_KEY}:
        raise Wan21RuntimeError("WanDancer requires exact settings metadata")
    try:
        settings = decode_wan22_dancer_settings(metadata[WAN22_DANCER_SETTINGS_KEY])
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"WanDancer settings metadata is invalid: {error}") from None
    channels = dict(record.channels)
    if len(channels) != len(record.channels):
        raise Wan21RuntimeError("WanDancer conditioning has duplicate channels")
    supported = {
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
        ConditioningChannel.REFERENCE_VISION_EMBEDDING,
        ConditioningChannel.AUDIO_EMBEDDING,
    }
    unsupported = tuple(channel.value for channel in channels if channel not in supported)
    if unsupported:
        raise Wan21RuntimeError(
            "WanDancer does not consume conditioning channels: " + ", ".join(unsupported)
        )
    descriptor = channels.get(ConditioningChannel.TEXT)
    if descriptor is None:
        raise Wan21RuntimeError("WanDancer conditioning is missing TEXT")
    if len(descriptor.shape) != 3 or descriptor.shape[2] != text_dim:
        raise Wan21RuntimeError(f"WanDancer TEXT descriptor must have shape [B,tokens,{text_dim}]")
    layout = record.token_layout
    if layout is None:
        raise Wan21RuntimeError("WanDancer conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise Wan21RuntimeError(f"WanDancer token layout is unsupported: {error}") from None
    expected_segment = TokenSegmentDescriptor("umt5", "umt5", 0, descriptor.shape[1])
    if layout.text_streams != ("umt5",) or layout.segments != (expected_segment,):
        raise Wan21RuntimeError("WanDancer conditioning requires the exact UMT5 token layout")
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("WanDancer carrier has duplicate payload bindings")
    expected_references = {value.reference.id for value in channels.values()}
    if set(bindings) != expected_references:
        raise Wan21RuntimeError(
            "WanDancer carrier payload bindings must match its channels exactly"
        )

    def materialize(channel: ConditioningChannel, space: str) -> torch.Tensor:
        channel_descriptor = channels[channel]
        binding = bindings[channel_descriptor.reference.id]
        if (binding.shape, binding.dtype, binding.space) != (
            channel_descriptor.shape,
            channel_descriptor.dtype,
            channel_descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"WanDancer {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(f"WanDancer {channel.value} has an unsupported tensor space")
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"WanDancer {channel.value} payload could not be decoded: {error}"
            ) from None

    text = _validate_text(
        materialize(ConditioningChannel.TEXT, "conditioning-text"), text_dim=text_dim
    )
    concat = (
        _validate_concat_latent(
            materialize(
                ConditioningChannel.CONCAT_LATENT,
                "conditioning-dancer-concat-latent",
            ),
            channels=20,
        )
        if ConditioningChannel.CONCAT_LATENT in channels
        else None
    )
    vision = (
        _validate_vision(
            materialize(
                ConditioningChannel.VISION_EMBEDDING,
                "conditioning-dancer-vision-embedding",
            ),
            allowed_rows=(257,),
        )
        if ConditioningChannel.VISION_EMBEDDING in channels
        else None
    )
    reference_vision = (
        _validate_vision(
            materialize(
                ConditioningChannel.REFERENCE_VISION_EMBEDDING,
                "conditioning-dancer-reference-vision-embedding",
            ),
            allowed_rows=(257,),
        )
        if ConditioningChannel.REFERENCE_VISION_EMBEDDING in channels
        else None
    )
    audio = (
        _validate_dancer_audio(
            materialize(
                ConditioningChannel.AUDIO_EMBEDDING,
                "conditioning-dancer-audio-embedding",
            )
        )
        if ConditioningChannel.AUDIO_EMBEDDING in channels
        else None
    )
    return Wan21PreparedConditioning(
        text,
        concat_latent=concat,
        vision=vision,
        concat_mask_index=0 if concat is not None else None,
        dancer_audio_embed=audio,
        dancer_reference_vision=reference_vision,
        dancer_settings=settings,
    )


def _materialize_humo_conditioning(
    carrier: object,
    *,
    text_dim: int,
    family_id: str,
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Wan HuMo conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan HuMo conditioning carrier is invalid: {error}") from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError("Wan HuMo requires exactly one conditioning record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.schedule != PercentRange(0.0, 1.0)
        or record.scale_vector is not None
        or record.extension_metadata
    ):
        raise Wan21RuntimeError("Wan HuMo requires one basic full-schedule record")
    channels = dict(record.channels)
    required = {
        ConditioningChannel.TEXT,
        ConditioningChannel.AUDIO_EMBEDDING,
        ConditioningChannel.REFERENCE_LATENT,
    }
    if set(channels) != required:
        raise Wan21RuntimeError(
            "Wan HuMo requires exactly TEXT, AUDIO_EMBEDDING, and REFERENCE_LATENT"
        )
    text_descriptor = channels[ConditioningChannel.TEXT]
    if len(text_descriptor.shape) != 3 or text_descriptor.shape[2] != text_dim:
        raise Wan21RuntimeError(f"Wan HuMo TEXT descriptor must have shape [B,tokens,{text_dim}]")
    layout = record.token_layout
    if layout is None:
        raise Wan21RuntimeError("Wan HuMo conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise Wan21RuntimeError(f"Wan HuMo token layout is unsupported: {error}") from None
    expected_segment = TokenSegmentDescriptor("umt5", "umt5", 0, text_descriptor.shape[1])
    if layout.text_streams != ("umt5",) or layout.segments != (expected_segment,):
        raise Wan21RuntimeError("Wan HuMo conditioning requires the exact UMT5 token layout")
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("Wan HuMo carrier has duplicate payload bindings")

    def materialize(channel: ConditioningChannel, space: str) -> torch.Tensor:
        channel_descriptor = channels[channel]
        binding = bindings.get(channel_descriptor.reference.id)
        if binding is None:
            raise Wan21RuntimeError(f"Wan HuMo {channel.value} has no payload binding")
        if (binding.shape, binding.dtype, binding.space) != (
            channel_descriptor.shape,
            channel_descriptor.dtype,
            channel_descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"Wan HuMo {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(f"Wan HuMo {channel.value} has an unsupported tensor space")
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"Wan HuMo {channel.value} payload could not be decoded: {error}"
            ) from None

    text = _validate_text(
        materialize(ConditioningChannel.TEXT, "conditioning-text"), text_dim=text_dim
    )
    audio = _validate_humo_audio(
        materialize(
            ConditioningChannel.AUDIO_EMBEDDING,
            "conditioning-humo-audio-embedding",
        )
    )
    reference = _validate_video_latent(
        materialize(
            ConditioningChannel.REFERENCE_LATENT,
            "conditioning-humo-reference-latent",
        )
    )
    return Wan21PreparedConditioning(
        text,
        humo_audio_embed=audio,
        humo_reference_latent=reference,
    )


def _materialize_s2v_conditioning(
    carrier: object,
    *,
    text_dim: int,
    family_id: str,
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Wan S2V conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan S2V conditioning carrier is invalid: {error}") from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError("Wan S2V requires exactly one conditioning record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.schedule != PercentRange(0.0, 1.0)
        or record.scale_vector is not None
        or record.extension_metadata
    ):
        raise Wan21RuntimeError("Wan S2V requires one basic full-schedule record")
    channels = dict(record.channels)
    supported = {
        ConditioningChannel.TEXT,
        ConditioningChannel.AUDIO_EMBEDDING,
        ConditioningChannel.REFERENCE_LATENT,
        ConditioningChannel.REFERENCE_MOTION,
        ConditioningChannel.CONTROL_VIDEO,
    }
    unsupported = tuple(channel.value for channel in channels if channel not in supported)
    if unsupported:
        raise Wan21RuntimeError(
            "Wan S2V does not consume conditioning channels: " + ", ".join(unsupported)
        )
    for required in (ConditioningChannel.TEXT, ConditioningChannel.CONTROL_VIDEO):
        if required not in channels:
            raise Wan21RuntimeError(f"Wan S2V conditioning is missing {required.value}")
    text_descriptor = channels[ConditioningChannel.TEXT]
    if len(text_descriptor.shape) != 3 or text_descriptor.shape[2] != text_dim:
        raise Wan21RuntimeError(f"Wan S2V TEXT descriptor must have shape [B,tokens,{text_dim}]")
    layout = record.token_layout
    if layout is None:
        raise Wan21RuntimeError("Wan S2V conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise Wan21RuntimeError(f"Wan S2V token layout is unsupported: {error}") from None
    expected_segment = TokenSegmentDescriptor("umt5", "umt5", 0, text_descriptor.shape[1])
    if layout.text_streams != ("umt5",) or layout.segments != (expected_segment,):
        raise Wan21RuntimeError("Wan S2V conditioning requires the exact UMT5 token layout")
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("Wan S2V carrier has duplicate payload bindings")

    def materialize(channel: ConditioningChannel, space: str) -> torch.Tensor:
        channel_descriptor = channels[channel]
        binding = bindings.get(channel_descriptor.reference.id)
        if binding is None:
            raise Wan21RuntimeError(f"Wan S2V {channel.value} has no payload binding")
        if (binding.shape, binding.dtype, binding.space) != (
            channel_descriptor.shape,
            channel_descriptor.dtype,
            channel_descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"Wan S2V {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(f"Wan S2V {channel.value} has an unsupported tensor space")
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"Wan S2V {channel.value} payload could not be decoded: {error}"
            ) from None

    text = _validate_text(
        materialize(ConditioningChannel.TEXT, "conditioning-text"), text_dim=text_dim
    )
    audio = (
        _validate_s2v_audio(
            materialize(ConditioningChannel.AUDIO_EMBEDDING, "conditioning-audio-embedding")
        )
        if ConditioningChannel.AUDIO_EMBEDDING in channels
        else None
    )
    reference = (
        _validate_video_latent(
            materialize(ConditioningChannel.REFERENCE_LATENT, "conditioning-reference-latent")
        )
        if ConditioningChannel.REFERENCE_LATENT in channels
        else None
    )
    motion = (
        _validate_video_latent(
            materialize(ConditioningChannel.REFERENCE_MOTION, "conditioning-reference-motion")
        )
        if ConditioningChannel.REFERENCE_MOTION in channels
        else None
    )
    control = _validate_video_latent(
        materialize(ConditioningChannel.CONTROL_VIDEO, "conditioning-control-video")
    )
    return Wan21PreparedConditioning(
        text,
        audio_embed=audio,
        s2v_reference_latent=reference,
        s2v_reference_motion=motion,
        s2v_control_video=control,
    )


def _materialize_conditioning(
    carrier: object,
    *,
    text_dim: int,
    requires_concat: bool,
    accepts_vision: bool,
    accepts_animate: bool = False,
    concat_channels: int = 20,
    concat_mask_index: int | None = 0,
    vision_rows: tuple[int, ...] = (257,),
    family_id: str = "dinkster.wan21",
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Wan 2.1 conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan 2.1 conditioning carrier is invalid: {error}") from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise Wan21RuntimeError("Wan 2.1 requires exactly one conditioning record")
    record = records[0]
    channels = dict(record.channels)
    supported = {ConditioningChannel.TEXT}
    if requires_concat:
        supported.add(ConditioningChannel.CONCAT_LATENT)
    if accepts_vision:
        supported.add(ConditioningChannel.VISION_EMBEDDING)
    if accepts_animate:
        supported.update((ConditioningChannel.POSE_LATENT, ConditioningChannel.FACE_PIXELS))
    unsupported = tuple(channel.value for channel in channels if channel not in supported)
    if unsupported:
        raise Wan21RuntimeError(
            "Wan does not consume conditioning channels: " + ", ".join(unsupported)
        )
    descriptor = channels.get(ConditioningChannel.TEXT)
    if descriptor is None:
        raise Wan21RuntimeError("Wan 2.1 conditioning is missing the TEXT channel")
    required = []
    if requires_concat:
        required.append(ConditioningChannel.CONCAT_LATENT)
    missing = tuple(channel.value for channel in required if channel not in channels)
    if missing:
        raise Wan21RuntimeError("Wan I2V conditioning is missing channels: " + ", ".join(missing))
    if len(descriptor.shape) != 3 or descriptor.shape[2] != text_dim:
        raise Wan21RuntimeError(f"Wan 2.1 TEXT descriptor must have shape [B,tokens,{text_dim}]")
    if record.area is not None:
        raise Wan21RuntimeError("Wan 2.1 does not consume conditioning areas")
    if record.mask is not None:
        raise Wan21RuntimeError("Wan 2.1 does not consume conditioning masks")
    if record.schedule != PercentRange(0.0, 1.0):
        raise Wan21RuntimeError("Wan 2.1 does not consume conditioning schedules")
    if record.scale_vector is not None:
        raise Wan21RuntimeError("Wan 2.1 does not consume conditioning scale vectors")
    if record.extension_metadata:
        raise Wan21RuntimeError("Wan 2.1 does not consume conditioning extension metadata")
    layout = record.token_layout
    if layout is None:
        raise Wan21RuntimeError("Wan 2.1 conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise Wan21RuntimeError(f"Wan 2.1 token layout is unsupported: {error}") from None
    expected_segment = TokenSegmentDescriptor("umt5", "umt5", 0, descriptor.shape[1])
    if layout.text_streams != ("umt5",) or layout.segments != (expected_segment,):
        raise Wan21RuntimeError("Wan 2.1 conditioning requires the exact UMT5 token layout")
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("Wan 2.1 conditioning carrier has duplicate payload bindings")

    def materialize(channel: ConditioningChannel, space: str) -> torch.Tensor:
        channel_descriptor = channels[channel]
        binding = bindings.get(channel_descriptor.reference.id)
        if binding is None:
            raise Wan21RuntimeError(f"Wan 2.1 {channel.value} channel has no payload binding")
        if (binding.shape, binding.dtype, binding.space) != (
            channel_descriptor.shape,
            channel_descriptor.dtype,
            channel_descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"Wan 2.1 {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(
                f"Wan 2.1 {channel.value} payload has an unsupported tensor space"
            )
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"Wan 2.1 {channel.value} payload could not be decoded: {error}"
            ) from None

    text = _validate_text(
        materialize(ConditioningChannel.TEXT, "conditioning-text"), text_dim=text_dim
    )
    if not requires_concat:
        return Wan21PreparedConditioning(text)
    concat_latent = _validate_concat_latent(
        materialize(ConditioningChannel.CONCAT_LATENT, "conditioning-concat-latent"),
        channels=concat_channels,
    )
    vision = (
        _validate_vision(
            materialize(ConditioningChannel.VISION_EMBEDDING, "conditioning-vision-embedding"),
            allowed_rows=vision_rows,
        )
        if accepts_vision and ConditioningChannel.VISION_EMBEDDING in channels
        else None
    )
    pose_latents = (
        _validate_video_latent(
            materialize(ConditioningChannel.POSE_LATENT, "conditioning-pose-latent"),
            channels=16,
        )
        if accepts_animate and ConditioningChannel.POSE_LATENT in channels
        else None
    )
    face_pixel_values = (
        _validate_face_pixel_values(
            materialize(ConditioningChannel.FACE_PIXELS, "conditioning-face-pixels")
        )
        if accepts_animate and ConditioningChannel.FACE_PIXELS in channels
        else None
    )
    return Wan21PreparedConditioning(
        text,
        concat_latent,
        vision,
        concat_mask_index=concat_mask_index,
        pose_latents=pose_latents,
        face_pixel_values=face_pixel_values,
    )


def _materialize_animate2_conditioning(
    carrier: object,
    *,
    text_dim: int,
    concat_channels: int,
    family_id: str,
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Wan Animate2 conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan Animate2 conditioning carrier is invalid: {error}") from None
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("Wan Animate2 carrier has duplicate payload bindings")

    main_records = tuple(
        record
        for record in typed.conditioning.records
        if ConditioningChannel.TEXT in dict(record.channels)
    )
    if len(main_records) != 1:
        raise Wan21RuntimeError("Wan Animate2 requires exactly one main conditioning record")
    main = main_records[0]
    pose_records = tuple(record for record in typed.conditioning.records if record is not main)
    if len(pose_records) > 1:
        raise Wan21RuntimeError("Wan Animate2 accepts at most one pose conditioning record")
    if len(typed.conditioning.records) != 1 + len(pose_records):
        raise Wan21RuntimeError("Wan Animate2 conditioning records are ambiguous")

    def require_plain_record(record: ConditioningRecord, *, pose: bool) -> None:
        name = "pose" if pose else "main"
        if record.area is not None:
            raise Wan21RuntimeError(f"Wan Animate2 {name} record does not consume areas")
        if record.mask is not None:
            raise Wan21RuntimeError(f"Wan Animate2 {name} record does not consume masks")
        if record.scale_vector is not None:
            raise Wan21RuntimeError(f"Wan Animate2 {name} record does not consume scale vectors")
        if pose:
            if type(record.schedule) is not PercentRange:
                raise Wan21RuntimeError("Wan Animate2 pose record requires a percent range")
        elif record.schedule != PercentRange(0.0, 1.0):
            raise Wan21RuntimeError("Wan Animate2 main record must use the full schedule")

    def require_layout(
        record: ConditioningRecord,
        descriptor: PayloadDescriptor,
        *,
        name: str,
    ) -> None:
        layout = record.token_layout
        if layout is None:
            raise Wan21RuntimeError(f"Wan Animate2 {name} record requires a token layout")
        try:
            layout.require_supported(family_id, (1,))
        except ValueError as error:
            raise Wan21RuntimeError(
                f"Wan Animate2 {name} token layout is unsupported: {error}"
            ) from None
        expected = TokenSegmentDescriptor("umt5", "umt5", 0, descriptor.shape[1])
        if layout.text_streams != ("umt5",) or layout.segments != (expected,):
            raise Wan21RuntimeError(
                f"Wan Animate2 {name} record requires the exact UMT5 token layout"
            )

    def materialize(
        channels: Mapping[ConditioningChannel, PayloadDescriptor],
        channel: ConditioningChannel,
        *,
        space: str,
    ) -> torch.Tensor:
        descriptor = channels[channel]
        binding = bindings.get(descriptor.reference.id)
        if binding is None:
            raise Wan21RuntimeError(f"Wan Animate2 {channel.value} channel has no payload binding")
        if (binding.shape, binding.dtype, binding.space) != (
            descriptor.shape,
            descriptor.dtype,
            descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"Wan Animate2 {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(
                f"Wan Animate2 {channel.value} payload has an unsupported tensor space"
            )
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"Wan Animate2 {channel.value} payload could not be decoded: {error}"
            ) from None

    require_plain_record(main, pose=False)
    if main.extension_metadata:
        raise Wan21RuntimeError("Wan Animate2 main record does not consume extension metadata")
    main_channels = dict(main.channels)
    main_supported = {
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
    }
    unsupported = tuple(channel.value for channel in main_channels if channel not in main_supported)
    if unsupported:
        raise Wan21RuntimeError(
            "Wan Animate2 main record does not consume channels: " + ", ".join(unsupported)
        )
    missing = tuple(
        channel.value
        for channel in (ConditioningChannel.TEXT, ConditioningChannel.CONCAT_LATENT)
        if channel not in main_channels
    )
    if missing:
        raise Wan21RuntimeError(
            "Wan Animate2 main record is missing channels: " + ", ".join(missing)
        )
    text_descriptor = main_channels[ConditioningChannel.TEXT]
    require_layout(main, text_descriptor, name="main")
    text = _validate_text(
        materialize(main_channels, ConditioningChannel.TEXT, space="conditioning-text"),
        text_dim=text_dim,
    )
    concat = _validate_concat_latent(
        materialize(
            main_channels,
            ConditioningChannel.CONCAT_LATENT,
            space="conditioning-concat-latent",
        ),
        channels=concat_channels,
    )
    vision = (
        _validate_vision(
            materialize(
                main_channels,
                ConditioningChannel.VISION_EMBEDDING,
                space="conditioning-vision-embedding",
            ),
            allowed_rows=(257,),
        )
        if ConditioningChannel.VISION_EMBEDDING in main_channels
        else None
    )

    if not pose_records:
        return Wan21PreparedConditioning(text, concat, vision, concat_mask_index=0)
    pose = pose_records[0]
    require_plain_record(pose, pose=True)
    pose_channels = dict(pose.channels)
    pose_supported = {
        ConditioningChannel.POSE_TEXT,
        ConditioningChannel.POSE_VISION_EMBEDDING,
        ConditioningChannel.POSE_LATENT,
    }
    unsupported = tuple(channel.value for channel in pose_channels if channel not in pose_supported)
    if unsupported:
        raise Wan21RuntimeError(
            "Wan Animate2 pose record does not consume channels: " + ", ".join(unsupported)
        )
    pose_text_descriptor = pose_channels.get(ConditioningChannel.POSE_TEXT)
    if pose_text_descriptor is None:
        raise Wan21RuntimeError("Wan Animate2 pose record is missing POSE_TEXT")
    require_layout(pose, pose_text_descriptor, name="pose")
    metadata = dict(pose.extension_metadata)
    if tuple(metadata) != (WAN21_ANIMATE2_SETTINGS_KEY,):
        raise Wan21RuntimeError(
            "Wan Animate2 pose record requires exactly the dinkster.wan21/animate2 metadata"
        )
    try:
        settings = decode_wan21_animate2_settings(metadata[WAN21_ANIMATE2_SETTINGS_KEY])
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan Animate2 settings metadata is invalid: {error}") from None
    pose_text = _validate_text(
        materialize(
            pose_channels,
            ConditioningChannel.POSE_TEXT,
            space="conditioning-pose-text",
        ),
        text_dim=text_dim,
    )
    pose_vision = (
        _validate_vision(
            materialize(
                pose_channels,
                ConditioningChannel.POSE_VISION_EMBEDDING,
                space="conditioning-pose-vision-embedding",
            ),
            allowed_rows=(257,),
        )
        if ConditioningChannel.POSE_VISION_EMBEDDING in pose_channels
        else None
    )
    pose_latents = (
        _validate_video_latent(
            materialize(
                pose_channels,
                ConditioningChannel.POSE_LATENT,
                space="conditioning-pose-latent",
            )
        )
        if ConditioningChannel.POSE_LATENT in pose_channels
        else None
    )
    return Wan21PreparedConditioning(
        text,
        concat,
        vision,
        concat_mask_index=0,
        pose_latents=pose_latents,
        pose_text=pose_text,
        pose_vision=pose_vision,
        pose_schedule=cast("PercentRange", pose.schedule),
        animate2_settings=settings,
    )


def _materialize_scail_conditioning(
    carrier: object,
    *,
    text_dim: int,
    accepts_masks: bool,
    family_id: str,
) -> Wan21PreparedConditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Wan SCAIL conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise Wan21RuntimeError(f"Wan SCAIL conditioning carrier is invalid: {error}") from None
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise Wan21RuntimeError("Wan SCAIL carrier has duplicate payload bindings")

    main_records = tuple(
        record
        for record in typed.conditioning.records
        if ConditioningChannel.TEXT in dict(record.channels)
    )
    if len(main_records) != 1:
        raise Wan21RuntimeError("Wan SCAIL requires exactly one main conditioning record")
    main = main_records[0]
    pose_records = tuple(record for record in typed.conditioning.records if record is not main)
    if len(pose_records) > 1 or len(typed.conditioning.records) != 1 + len(pose_records):
        raise Wan21RuntimeError("Wan SCAIL accepts at most one unambiguous pose record")

    def require_plain(record: ConditioningRecord, *, name: str, full_schedule: bool) -> None:
        if record.area is not None or record.mask is not None or record.scale_vector is not None:
            raise Wan21RuntimeError(f"Wan SCAIL {name} record must be plain")
        if full_schedule:
            if record.schedule != PercentRange(0.0, 1.0):
                raise Wan21RuntimeError("Wan SCAIL main record must use the full schedule")
        elif type(record.schedule) is not PercentRange:
            raise Wan21RuntimeError("Wan SCAIL pose record requires a percent range")

    def materialize(
        channels: Mapping[ConditioningChannel, PayloadDescriptor],
        channel: ConditioningChannel,
        *,
        space: str,
    ) -> torch.Tensor:
        descriptor = channels[channel]
        binding = bindings.get(descriptor.reference.id)
        if binding is None:
            raise Wan21RuntimeError(f"Wan SCAIL {channel.value} channel has no payload binding")
        if (binding.shape, binding.dtype, binding.space) != (
            descriptor.shape,
            descriptor.dtype,
            descriptor.space,
        ):
            raise Wan21RuntimeError(
                f"Wan SCAIL {channel.value} descriptor does not match its payload binding"
            )
        if binding.space != space:
            raise Wan21RuntimeError(
                f"Wan SCAIL {channel.value} payload has an unsupported tensor space"
            )
        try:
            return payload_binding_to_tensor(binding)
        except TensorPayloadError as error:
            raise Wan21RuntimeError(
                f"Wan SCAIL {channel.value} payload could not be decoded: {error}"
            ) from None

    require_plain(main, name="main", full_schedule=True)
    main_channels = dict(main.channels)
    supported = {
        ConditioningChannel.TEXT,
        ConditioningChannel.VISION_EMBEDDING,
        ConditioningChannel.SCAIL_REFERENCE_LATENT,
    }
    if accepts_masks:
        supported.add(ConditioningChannel.SCAIL_REFERENCE_MASK)
    unsupported = tuple(channel.value for channel in main_channels if channel not in supported)
    if unsupported:
        raise Wan21RuntimeError(
            "Wan SCAIL main record does not consume channels: " + ", ".join(unsupported)
        )
    if ConditioningChannel.TEXT not in main_channels:
        raise Wan21RuntimeError("Wan SCAIL main record is missing TEXT")
    metadata = dict(main.extension_metadata)
    if tuple(metadata) != (WAN21_SCAIL_REPLACEMENT_KEY,):
        raise Wan21RuntimeError(
            "Wan SCAIL main record requires exactly the replacement-mode metadata"
        )
    replacement = metadata[WAN21_SCAIL_REPLACEMENT_KEY]
    if type(replacement) is not bool:
        raise Wan21RuntimeError("Wan SCAIL replacement-mode metadata must be a bool")
    text_descriptor = main_channels[ConditioningChannel.TEXT]
    layout = main.token_layout
    if layout is None:
        raise Wan21RuntimeError("Wan SCAIL main record requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise Wan21RuntimeError(f"Wan SCAIL token layout is unsupported: {error}") from None
    expected_segment = TokenSegmentDescriptor("umt5", "umt5", 0, text_descriptor.shape[1])
    if layout.text_streams != ("umt5",) or layout.segments != (expected_segment,):
        raise Wan21RuntimeError("Wan SCAIL requires the exact UMT5 token layout")
    text = _validate_text(
        materialize(main_channels, ConditioningChannel.TEXT, space="conditioning-text"),
        text_dim=text_dim,
    )
    reference = (
        _validate_video_latent(
            materialize(
                main_channels,
                ConditioningChannel.SCAIL_REFERENCE_LATENT,
                space="conditioning-scail-reference-latent",
            )
        )
        if ConditioningChannel.SCAIL_REFERENCE_LATENT in main_channels
        else None
    )
    vision = (
        _validate_vision(
            materialize(
                main_channels,
                ConditioningChannel.VISION_EMBEDDING,
                space="conditioning-vision-embedding",
            ),
            allowed_rows=(257,),
        )
        if ConditioningChannel.VISION_EMBEDDING in main_channels
        else None
    )
    reference_mask = (
        _validate_video_latent(
            materialize(
                main_channels,
                ConditioningChannel.SCAIL_REFERENCE_MASK,
                space="conditioning-scail-reference-mask",
            ),
            channels=28,
        )
        if ConditioningChannel.SCAIL_REFERENCE_MASK in main_channels
        else None
    )

    pose_latents: torch.Tensor | None = None
    driving_mask: torch.Tensor | None = None
    pose_schedule: PercentRange | None = None
    if pose_records:
        pose = pose_records[0]
        require_plain(pose, name="pose", full_schedule=False)
        if pose.extension_metadata or pose.token_layout is not None:
            raise Wan21RuntimeError(
                "Wan SCAIL pose record does not consume metadata or token layout"
            )
        pose_channels = dict(pose.channels)
        pose_supported = {ConditioningChannel.POSE_LATENT}
        if accepts_masks:
            pose_supported.add(ConditioningChannel.SCAIL_DRIVING_MASK)
        unsupported = tuple(
            channel.value for channel in pose_channels if channel not in pose_supported
        )
        if unsupported:
            raise Wan21RuntimeError(
                "Wan SCAIL pose record does not consume channels: " + ", ".join(unsupported)
            )
        if ConditioningChannel.POSE_LATENT not in pose_channels:
            raise Wan21RuntimeError("Wan SCAIL pose record is missing POSE_LATENT")
        pose_latents = _validate_video_latent(
            materialize(
                pose_channels,
                ConditioningChannel.POSE_LATENT,
                space="conditioning-pose-latent",
            )
        )
        driving_mask = (
            _validate_video_latent(
                materialize(
                    pose_channels,
                    ConditioningChannel.SCAIL_DRIVING_MASK,
                    space="conditioning-scail-driving-mask",
                ),
                channels=28,
            )
            if ConditioningChannel.SCAIL_DRIVING_MASK in pose_channels
            else None
        )
        pose_schedule = cast("PercentRange", pose.schedule)
    return Wan21PreparedConditioning(
        text=text,
        vision=vision,
        pose_latents=pose_latents,
        pose_schedule=pose_schedule,
        scail_reference_latent=reference,
        scail_reference_mask=reference_mask,
        scail_driving_mask=driving_mask,
        scail_replacement=replacement,
    )


def _video_streams(
    value: object, *, channels: int = 16
) -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    if type(value) is not MultiStreamLatent:
        raise TypeError("Wan 2.1 latent must be an exact MultiStreamLatent")
    streams = cast("MultiStreamLatent[torch.Tensor]", value)
    if streams.roles != ("video",):
        raise Wan21RuntimeError("Wan 2.1 T2V requires the exact latent stream role 'video'")
    return streams, _validate_video_latent(streams.by_role("video"), channels=channels)


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SamplingCancelled("sampling cancelled")


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
class _Wan21ModelConditioning:
    text: torch.Tensor
    concat_latent: torch.Tensor | None
    vision: torch.Tensor | None
    reference_latent: torch.Tensor | None
    vace_context: torch.Tensor | None
    vace_strengths: tuple[float, ...]
    camera_conditions: torch.Tensor | None
    temporal_reference: torch.Tensor | None
    context_latents: tuple[torch.Tensor, ...]
    pose_latents: torch.Tensor | None
    face_pixel_values: torch.Tensor | None
    pose_text: torch.Tensor | None
    pose_vision: torch.Tensor | None
    pose_schedule: PercentRange | None
    animate2_settings: Wan21Animate2Settings | None
    scail_reference_latent: torch.Tensor | None
    scail_reference_mask: torch.Tensor | None
    scail_driving_mask: torch.Tensor | None
    scail_replacement: bool
    audio_embed: torch.Tensor | None
    s2v_reference_latent: torch.Tensor | None
    s2v_reference_motion: torch.Tensor | None
    s2v_control_video: torch.Tensor | None
    humo_audio_embed: torch.Tensor | None
    humo_reference_latent: torch.Tensor | None
    dancer_audio_embed: torch.Tensor | None
    dancer_reference_vision: torch.Tensor | None
    dancer_settings: Wan22DancerSettings | None


class _Wan21LatentNormalizer(torch.nn.Module):
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32), persistent=False
        )

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != len(LATENTS_MEAN):
            raise ValueError("Wan21 normalization requires rank-5 16-channel latents")
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != len(LATENTS_MEAN):
            raise ValueError("Wan21 normalization requires rank-5 16-channel latents")
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return latent * std + mean


@dataclass(frozen=True)
class _Wan21DiffusionAssembly:
    diffusion: Wan21Model | Wan21CausalModel | Wan21HumoModel | Wan22S2VModel | Wan22DancerModel
    family: ModelFamily
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _compute_dtype: torch.dtype | None = field(default=None, repr=False, compare=False)

    def compute_dtype(self, role: str) -> torch.dtype | None:
        if role != "diffusion":
            return None
        if self._compute_dtype is not None:
            return self._compute_dtype
        linear = self.diffusion.time_embedding[0]
        dtype = bound_compute_dtype(linear)
        if dtype is None:
            dtype = getattr(linear, "compute_dtype", None)
        if dtype is None:
            # Initless linears store ordinary weights at their compute dtype.
            weight = getattr(linear, "weight", None)
            dtype = weight.dtype if isinstance(weight, torch.Tensor) else None
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError("Wan diffusion has no bound compute dtype")
        return dtype


def _wan_custom_space(
    assembled: AssembledWan21 | _Wan21DiffusionAssembly,
    sampling_shift: float | None = None,
) -> FlowSigmas:
    model = assembled.diffusion
    if type(model) is Wan21CausalModel:
        if sampling_shift is not None:
            raise Wan21RuntimeError("Wan CausalAR sampling shift is fixed at 5.0")
        return FlowSigmas(shift=5.0)
    if sampling_shift is not None:
        if (
            type(sampling_shift) is not float
            or not math.isfinite(sampling_shift)
            or sampling_shift <= 0.0
        ):
            raise Wan21RuntimeError("sampling_shift must be a positive finite float")
        return FlowSigmas(shift=sampling_shift)
    if model.config.model_variant == "animate2":
        return FlowSigmas(shift=5.0)
    return (
        WAN22_SIGMAS if model.config.out_channels == WAN22_CODEC.latent.channels else WAN21_SIGMAS
    )


class _Wan21CausalDenoiser:
    def __init__(
        self,
        model: Wan21CausalModel,
        text: torch.Tensor,
        *,
        initial_latent: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        self._model = model
        self._text = text
        self._initial_latent = initial_latent
        self._compute_dtype = compute_dtype

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        del x, sigma
        raise Wan21RuntimeError("Wan CausalAR requires the ar_video sampler")

    def sample_autoregressive(
        self,
        x: torch.Tensor,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        num_frame_per_block: int,
        on_step: StepCallback | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self._model.config.out_channels:
            raise Wan21RuntimeError("Wan CausalAR requires rank-5 16-channel video latents")
        if type(num_frame_per_block) is not int or not 1 <= num_frame_per_block <= 64:
            raise Wan21RuntimeError("Wan CausalAR block size must be an integer in [1, 64]")
        if sigmas[-1] != 0.0 or any(
            sigmas[index] <= sigmas[index + 1] for index in range(len(sigmas) - 1)
        ):
            raise Wan21RuntimeError("Wan CausalAR requires strictly decreasing sigmas ending at 0")
        batch, _channels, frames, height, width = x.shape
        rows_per_frame = math.ceil(height / 2) * math.ceil(width / 2)
        text = self._text
        if text.shape[0] not in (1, batch):
            raise Wan21RuntimeError("Wan CausalAR text batch must be one or match the video batch")
        text = text.to(device=x.device, dtype=self._compute_dtype)
        if text.shape[0] == 1 and batch != 1:
            text = text.expand(batch, *text.shape[1:])
        output = torch.zeros_like(x)
        start_frame = 0
        initial = self._initial_latent
        if initial is not None:
            if (
                initial.ndim != 5
                or initial.shape[0] not in (1, batch)
                or initial.shape[1] != x.shape[1]
                or initial.shape[2] > frames
                or initial.shape[3:] != x.shape[3:]
            ):
                raise Wan21RuntimeError("Wan CausalAR initial latent must match the target video")
            initial = initial.to(device=x.device, dtype=self._compute_dtype)
            if initial.shape[0] == 1 and batch != 1:
                initial = initial.expand(batch, *initial.shape[1:])
        caches = self._model.create_caches(
            batch_size=batch,
            max_tokens=frames * rows_per_frame,
            device=x.device,
            dtype=self._compute_dtype,
        )
        if initial is not None:
            start_frame = initial.shape[2]
            output[:, :, :start_frame] = initial
            self._model.forward_block(
                initial,
                torch.zeros((batch,), device=x.device, dtype=torch.float32),
                text,
                time_start=0,
                caches=caches,
            )
        sigma_steps = max(len(sigmas) - 1, 0)
        blocks = math.ceil((frames - start_frame) / num_frame_per_block)
        total_evaluations = blocks * sigma_steps
        evaluation = 0
        for block_index in range(blocks):
            block_frames = min(num_frame_per_block, frames - start_frame)
            end_frame = start_frame + block_frames
            noisy = x[:, :, start_frame:end_frame]
            cache_rows = block_frames * rows_per_frame
            for sigma_index in range(sigma_steps):
                sigma = float(sigmas[sigma_index])
                model_input = calculate_input(Parameterization.FLOW, sigma, noisy).to(
                    dtype=self._compute_dtype
                )
                raw = self._model.forward_block(
                    model_input,
                    torch.full(
                        (batch,),
                        sigma * 1000.0,
                        device=x.device,
                        dtype=torch.float32,
                    ),
                    text,
                    time_start=start_frame,
                    caches=caches,
                )
                denoised = calculate_denoised(Parameterization.FLOW, sigma, raw, noisy).float()
                progress_step = (
                    evaluation * sigma_steps // total_evaluations if total_evaluations else 0
                )
                if info.on_state is not None:
                    state = output.clone()
                    state[:, :, start_frame:end_frame] = noisy
                    denoised_state = output.clone()
                    denoised_state[:, :, start_frame:end_frame] = denoised
                    info.on_state(
                        SolverStateEvent(
                            progress_step,
                            sigma_steps,
                            sigma,
                            "pre_update",
                            state,
                            denoised_state,
                        )
                    )
                if on_step is not None:
                    on_step(StepEvent(progress_step, sigma_steps, sigma))
                sigma_next = float(sigmas[sigma_index + 1])
                if sigma_next == 0.0:
                    noisy = denoised
                else:
                    generator = torch.Generator(device=x.device).manual_seed(
                        info.seed + block_index * 1000 + sigma_index
                    )
                    fresh_noise = torch.randn(
                        denoised.shape,
                        generator=generator,
                        device=x.device,
                        dtype=denoised.dtype,
                    )
                    noisy = denoised * (1.0 - sigma_next) + fresh_noise * sigma_next
                    caches.rewind(cache_rows)
                evaluation += 1
            output[:, :, start_frame:end_frame] = noisy
            caches.rewind(cache_rows)
            self._model.forward_block(
                noisy.to(dtype=self._compute_dtype),
                torch.zeros((batch,), device=x.device, dtype=torch.float32),
                text,
                time_start=start_frame,
                caches=caches,
            )
            start_frame = end_frame
        return output


class Wan21Runtime(MultiStreamSamplingRuntime):
    """Positive/negative UMT5 conditioning, FLOW sampling, and the Wan VAE."""

    retained_offload_storage_components = frozenset({"umt5xxl"})
    sampling_error = Wan21RuntimeError
    supports_sampling_shift = True
    supports_denoised_capture = True
    supports_batch_noise_indices = False

    def __init__(
        self,
        assembled: AssembledWan21,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        pose_cache_settings: Wan21PoseBlockCacheSettings | None = None,
    ) -> None:
        if type(assembled.family.id) is not str or not assembled.family.id.strip():
            raise ValueError("Wan runtime family identity must be a nonempty string")
        if not runtime_identity:
            raise ValueError("Wan runtime identity must be nonempty")
        if pose_cache_settings is not None:
            if type(pose_cache_settings) is not Wan21PoseBlockCacheSettings:
                raise TypeError("pose_cache_settings must be exact Wan21PoseBlockCacheSettings")
            if assembled.diffusion.config.model_variant != "animate2":
                raise ValueError("Wan pose cache settings require an Animate2 profile")
        tokenizer = Umt5SentencePieceTokenizer(assembled.tokenizer_model)
        self._assembled = assembled
        self._runtime_identity = runtime_identity
        flow_rvs = assembled.diffusion.config.model_variant == "flow_rvs"
        wan22_latent = assembled.diffusion.config.out_channels == WAN22_CODEC.latent.channels
        self.codec = CodecPlugin(
            (WAN22_CODEC if wan22_latent else WAN21_FLOW_RVS_CODEC if flow_rvs else WAN21_CODEC),
            assembled.vae,
            assembled.vae,
            content_crop=(
                (
                    lambda content: _crop_spatial_to_multiple(
                        content, WAN22_CODEC.latent.spatial_downscale
                    )
                )
                if wan22_latent
                else None
            ),
            content_in=lambda value: value * 2.0 - 1.0,
            content_out=lambda value: ((value.float() + 1.0) / 2.0).clamp_(0.0, 1.0),
            compute_dtype=assembled.compute_dtype("vae"),
        )
        self._text_encoder = T5TextEncoder(assembled.umt5xxl)
        self._tokenizer = PromptTokenizer(encode_word=tokenizer.encode)
        self._latent_process_in = assembled.vae.process_in
        self._latent_process_out = assembled.vae.process_out
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._pose_cache_settings = pose_cache_settings

    @property
    def assembled(self) -> AssembledWan21:
        return self._assembled

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def supports_denoise_mask(self) -> bool:
        return not isinstance(self.assembled.diffusion, Wan21CausalModel)

    @property
    def conditioning_identity(self) -> str:
        """Stable compatibility identity for materialized Wan conditioning."""
        config = self.assembled.diffusion.config
        fields = (
            self.family.id,
            config.model_type,
            config.model_variant,
            config.in_channels,
            config.out_channels,
            config.hidden_size,
            config.ffn_hidden_size,
            config.num_heads,
            config.num_layers,
            config.text_dim,
            *((config.reference_channels,) if config.reference_channels is not None else ()),
            *(
                ()
                if config.flf_pos_embed_token_number is None
                else (config.flf_pos_embed_token_number,)
            ),
            *((config.vace_layers,) if config.vace_layers is not None else ()),
            *((config.camera_channels,) if config.camera_channels is not None else ()),
        )
        return "dinkster.wan.conditioning:v1:" + ":".join(str(field) for field in fields)

    def encode_text(
        self,
        text: str,
        *,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        spans = self._tokenizer.tokenize(text)
        if min_padding is None and min_length is None:
            return self._text_encoder.encode(spans)
        return self._text_encoder.encode(spans, min_padding=min_padding, min_length=min_length)

    def encode_vision(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
        encoder = self.assembled.clip_vision
        if encoder is None:
            raise Wan21RuntimeError("the loaded Wan 2.1 T2V profile has no CLIP vision encoder")
        return encoder(image, crop=crop).float()

    def text_conditioning_carrier(
        self, conditioning: Conditioning[torch.Tensor]
    ) -> ConditioningCarrier:
        """One encode_text result -> the canonical Wan text carrier."""
        return _text_carrier(
            conditioning,
            text_dim=self.assembled.diffusion.config.text_dim,
            family_id=self.family.id,
        )

    def prepare_text_conditioning(
        self, conditioning: Conditioning[torch.Tensor]
    ) -> Wan21PreparedConditioning:
        text_dim = self.assembled.diffusion.config.text_dim
        return _materialize_conditioning(
            _text_carrier(conditioning, text_dim=text_dim, family_id=self.family.id),
            text_dim=text_dim,
            requires_concat=False,
            accepts_vision=False,
            family_id=self.family.id,
        )

    def prepare_bernini_conditioning(
        self,
        text: Wan21PreparedConditioning,
        context_latents: tuple[torch.Tensor, ...],
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if type(self.assembled.diffusion) is not Wan21Model or config.model_variant != "bernini":
            raise Wan21RuntimeError("Bernini supports only exact native Wan 2.2 14B T2V profiles")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if type(context_latents) is not tuple:
            raise TypeError("context_latents must be an exact tuple")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
            or text.scail_reference_latent is not None
            or text.scail_reference_mask is not None
            or text.scail_driving_mask is not None
            or text.scail_replacement
        ):
            raise Wan21RuntimeError("Wan Bernini text input must contain only TEXT")
        return Wan21PreparedConditioning(
            text.text,
            context_latents=tuple(
                _validate_video_latent(latent, channels=config.out_channels)
                for latent in context_latents
            ),
        )

    def prepare_conditioning(self, carrier: ConditioningCarrier) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if config.model_variant == "wandancer":
            return _materialize_dancer_conditioning(
                carrier,
                text_dim=config.text_dim,
                family_id=self.family.id,
            )
        if config.model_variant == "humo":
            return _materialize_humo_conditioning(
                carrier,
                text_dim=config.text_dim,
                family_id=self.family.id,
            )
        if config.model_variant == "s2v":
            return _materialize_s2v_conditioning(
                carrier,
                text_dim=config.text_dim,
                family_id=self.family.id,
            )
        if config.model_variant in ("scail", "scail2"):
            return _materialize_scail_conditioning(
                carrier,
                text_dim=config.text_dim,
                accepts_masks=config.model_variant == "scail2",
                family_id=self.family.id,
            )
        if config.model_variant == "animate2":
            return _materialize_animate2_conditioning(
                carrier,
                text_dim=config.text_dim,
                concat_channels=config.in_channels - config.out_channels,
                family_id=self.family.id,
            )
        if config.model_variant == "animate":
            return _materialize_conditioning(
                carrier,
                text_dim=config.text_dim,
                requires_concat=True,
                accepts_vision=True,
                accepts_animate=True,
                concat_channels=config.in_channels - config.out_channels,
                concat_mask_index=0,
                vision_rows=(257,),
                family_id=self.family.id,
            )
        if config.camera_channels is not None:
            raise Wan21RuntimeError("Wan camera conditioning uses its typed native adapter")
        requires_concat = config.in_channels > config.out_channels
        return _materialize_conditioning(
            carrier,
            text_dim=config.text_dim,
            requires_concat=requires_concat,
            accepts_vision=config.model_type == "i2v",
            concat_channels=config.in_channels - config.out_channels,
            concat_mask_index=_concat_mask_index(config) if requires_concat else None,
            vision_rows=_allowed_vision_rows(config),
            family_id=self.family.id,
        )

    def prepare_i2v_conditioning(
        self,
        text: Wan21PreparedConditioning,
        concat_latent: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if config.model_variant in ("animate", "animate2", "scail", "scail2", "wandancer"):
            raise Wan21RuntimeError("Wan specialized conditioning uses its typed native adapter")
        if config.camera_channels is not None:
            raise Wan21RuntimeError("Wan camera profiles require camera conditioning")
        if config.in_channels <= config.out_channels:
            raise Wan21RuntimeError("Wan profile does not consume reference-latent conditioning")
        accepts_vision = config.model_type == "i2v"
        if not accepts_vision and vision is not None:
            raise Wan21RuntimeError("Wan 2.2 I2V does not consume vision conditioning")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan 2.1 I2V text input must contain only TEXT")
        concat_channels = config.in_channels - config.out_channels
        return self.prepare_conditioning(
            _i2v_carrier(
                text,
                _validate_concat_latent(concat_latent, channels=concat_channels),
                (
                    None
                    if vision is None
                    else _validate_vision(
                        vision,
                        allowed_rows=_allowed_vision_rows(config),
                    )
                ),
                family_id=self.family.id,
            )
        )

    def prepare_animate_conditioning(
        self,
        text: Wan21PreparedConditioning,
        concat_latent: torch.Tensor,
        *,
        vision: torch.Tensor | None = None,
        pose_latents: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if config.model_variant != "animate":
            raise Wan21RuntimeError("Wan profile does not consume Animate conditioning")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan Animate text input must contain only TEXT")
        return Wan21PreparedConditioning(
            text.text,
            _validate_concat_latent(
                concat_latent,
                channels=config.in_channels - config.out_channels,
            ),
            None if vision is None else _validate_vision(vision, allowed_rows=(257,)),
            concat_mask_index=0,
            pose_latents=(
                None
                if pose_latents is None
                else _validate_video_latent(pose_latents, channels=config.out_channels)
            ),
            face_pixel_values=(
                None
                if face_pixel_values is None
                else _validate_face_pixel_values(face_pixel_values)
            ),
        )

    def prepare_camera_conditioning(
        self,
        text: Wan21PreparedConditioning,
        concat_latent: torch.Tensor | None,
        camera_conditions: torch.Tensor | None,
        vision: torch.Tensor | None = None,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if config.camera_channels is None:
            raise Wan21RuntimeError("Wan profile does not consume camera conditioning")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan camera text input must contain only TEXT")
        requires_vision = config.model_type == "i2v"
        if not requires_vision and vision is not None:
            raise Wan21RuntimeError("Wan 2.2 camera does not consume vision conditioning")
        concat_channels = config.in_channels - config.out_channels
        concat = (
            None
            if concat_latent is None
            else _validate_concat_latent(concat_latent, channels=concat_channels)
        )
        camera = (
            None
            if camera_conditions is None
            else _validate_camera_conditions(
                camera_conditions,
                channels=config.camera_channels,
            )
        )
        if (
            concat is not None
            and camera is not None
            and (
                camera.shape[2] != concat.shape[2]
                or camera.shape[3] != concat.shape[3] * 8
                or camera.shape[4] != concat.shape[4] * 8
            )
        ):
            raise Wan21RuntimeError("Wan camera and reference conditioning geometry must match")
        return Wan21PreparedConditioning(
            text.text,
            concat,
            (
                None
                if vision is None
                else _validate_vision(vision, allowed_rows=_allowed_vision_rows(config))
            ),
            concat_mask_index=(0 if concat is not None and concat_channels == 20 else None),
            camera_conditions=camera,
        )

    def prepare_phantom_conditioning(
        self,
        text: Wan21PreparedConditioning,
        temporal_reference: torch.Tensor | None,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if (
            config.model_type != "t2v"
            or config.model_variant != "base"
            or config.in_channels != config.out_channels
            or config.reference_channels is not None
            or config.vace_layers is not None
            or config.camera_channels is not None
        ):
            raise Wan21RuntimeError("Wan profile does not consume Phantom subject references")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan Phantom text input must contain only TEXT")
        return Wan21PreparedConditioning(
            text.text,
            temporal_reference=(
                None
                if temporal_reference is None
                else _validate_video_latent(temporal_reference, channels=config.out_channels)
            ),
        )

    def prepare_fun_conditioning(
        self,
        text: Wan21PreparedConditioning,
        concat_latent: torch.Tensor,
        *,
        concat_mask_index: int | None,
        vision: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        extra_channels = config.in_channels - config.out_channels
        is_fun = extra_channels in (
            config.out_channels + 4,
            config.out_channels * 2,
            config.out_channels * 2 + 4,
        )
        if not is_fun:
            raise Wan21RuntimeError("Wan profile does not consume Fun conditioning")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.reference_latent is not None
            or text.vace_frames
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan Fun text input must contain only TEXT")
        expected_mask_index = _concat_mask_index(config)
        if concat_mask_index != expected_mask_index:
            raise Wan21RuntimeError(f"Wan Fun concat mask index must be {expected_mask_index}")
        requires_vision = config.model_type == "i2v"
        if requires_vision and vision is None:
            raise Wan21RuntimeError("Wan 2.1 Fun requires vision conditioning")
        if not requires_vision and vision is not None:
            raise Wan21RuntimeError("Wan 2.2 Fun does not consume vision conditioning")
        if reference_latent is not None and config.reference_channels is None:
            raise Wan21RuntimeError("Wan Fun profile does not consume full-reference conditioning")
        return Wan21PreparedConditioning(
            text.text,
            _validate_concat_latent(concat_latent, channels=extra_channels),
            (
                None
                if vision is None
                else _validate_vision(vision, allowed_rows=_allowed_vision_rows(config))
            ),
            concat_mask_index=concat_mask_index,
            reference_latent=(
                None
                if reference_latent is None
                else _validate_reference_latent(
                    reference_latent,
                    channels=config.reference_channels,
                )
            ),
        )

    def prepare_vace_conditioning(
        self,
        text: Wan21PreparedConditioning,
        frames: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
    ) -> Wan21PreparedConditioning:
        config = self.assembled.diffusion.config
        if config.vace_layers is None:
            raise Wan21RuntimeError("Wan profile does not consume VACE conditioning")
        if type(text) is not Wan21PreparedConditioning:
            raise TypeError("text must be exact Wan21PreparedConditioning")
        if (
            text.concat_latent is not None
            or text.vision is not None
            or text.camera_conditions is not None
            or text.temporal_reference is not None
            or text.context_latents
            or text.pose_latents is not None
            or text.face_pixel_values is not None
            or text.pose_text is not None
            or text.pose_vision is not None
            or text.pose_schedule is not None
            or text.animate2_settings is not None
        ):
            raise Wan21RuntimeError("Wan VACE text input must not contain I2V conditioning")
        frames = _validate_vace_tensor(frames, channels=32, name="VACE_FRAMES")
        mask = _validate_vace_tensor(mask, channels=64, name="VACE_MASK")
        if frames.shape[0] != mask.shape[0] or frames.shape[2:] != mask.shape[2:]:
            raise Wan21RuntimeError("Wan VACE frame and mask geometry must match")
        if type(strength) is not float or not math.isfinite(strength) or strength < 0.0:
            raise Wan21RuntimeError("Wan VACE strength must be a finite non-negative float")
        return Wan21PreparedConditioning(
            text.text,
            vace_frames=(*text.vace_frames, frames),
            vace_masks=(*text.vace_masks, mask),
            vace_strengths=(*text.vace_strengths, strength),
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return _wan_custom_space(self.assembled, sampling_shift)

    def _custom_sampling_process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return self._latent_process_in(latent)

    def _custom_sampling_process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return self._latent_process_out(latent)

    def check_custom_sampling(
        self,
        request: CustomSamplingRequest[torch.Tensor],
        *,
        has_denoise_mask: bool,
        has_inpaint: bool,
        has_context_windows: bool,
        guidance: FluxGuidance = None,
    ) -> None:
        MultiStreamSamplingRuntime.check_custom_sampling(
            self,
            request,
            has_denoise_mask=has_denoise_mask,
            has_inpaint=has_inpaint,
            has_context_windows=has_context_windows,
            guidance=guidance,
        )
        model = self.assembled.diffusion
        if type(model) is Wan21CausalModel:
            if request.sampler.id != "dinkster.ar_video":
                raise Wan21RuntimeError("Wan CausalAR requires the ar_video sampler")
            if has_denoise_mask or has_inpaint:
                raise Wan21RuntimeError(
                    "Wan CausalAR does not support masks or inpaint conditioning"
                )
            if has_context_windows:
                raise Wan21RuntimeError("Wan CausalAR does not support context windows")
            if guidance is not None:
                raise Wan21RuntimeError("Wan CausalAR does not support distilled guidance")
        else:
            if request.sampler.id == "dinkster.ar_video":
                raise Wan21RuntimeError("ar_video requires the Wan CausalAR profile")
            if has_context_windows and (
                model.config.model_variant != "base" or model.config.model_type != "t2v"
            ):
                raise Wan21RuntimeError(
                    "Wan context windows support only the base text-to-video profiles"
                )
            if has_context_windows and model.config.vace_layers is not None:
                raise Wan21RuntimeError("Wan context windows do not support VACE models")

    def sample_custom(
        self,
        latent: CustomSamplingLatentValue,
        *,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        request: CustomSamplingRequest[torch.Tensor],
        seed: int = 0,
        guidance: float | None = None,
        denoise_mask: CustomSamplingLatentValue | None = None,
        inpaint: object | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        initial_latent: torch.Tensor | None = None,
        cancelled: Callable[[], bool] | None = None,
        observer: ExecutionObserverAttachment | None = None,
        parent_span_id: int | None = None,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        sampling_shift: float | None = None,
        uni3c: Wan21Uni3CExecution | None = None,
        multitalk: Wan21InfiniteTalkExecution | None = None,
        capture_denoised: bool = True,
    ) -> CustomSamplingResult[torch.Tensor] | CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        model = self.assembled.diffusion
        if type(model) is not Wan21CausalModel:
            if initial_latent is not None:
                raise Wan21RuntimeError("initial_latent is supported only by Wan CausalAR")
            return self._sample_standard_custom(
                latent,
                noise=noise,
                cond=cond,
                cfg=cfg,
                request=request,
                seed=seed,
                guidance=guidance,
                denoise_mask=denoise_mask,
                inpaint=inpaint,
                context_windows=context_windows,
                on_step=on_step,
                on_state=on_state,
                cancelled=cancelled,
                observer=observer,
                parent_span_id=parent_span_id,
                compute_dtype=compute_dtype,
                device=device,
                sampling_shift=sampling_shift,
                uni3c=uni3c,
                multitalk=multitalk,
                capture_denoised=capture_denoised,
            )
        if uni3c is not None or multitalk is not None:
            raise Wan21RuntimeError("Wan CausalAR does not support Uni3C or InfiniteTalk")
        if sampling_shift is not None:
            raise Wan21RuntimeError("Wan CausalAR sampling shift is fixed at 5.0")
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        if cancelled is None:
            cancelled = sampling_environment_cancellation()
        _check_cancelled(cancelled)
        del observer, parent_span_id
        structural = type(latent) is MultiStreamLatent
        if structural:
            streams, video = _video_streams(latent, channels=model.config.out_channels)
            if type(noise) is not MultiStreamLatent or noise.roles != streams.roles:
                raise Wan21RuntimeError(
                    "Wan custom sampling noise must match the latent stream roles"
                )
            noise_video = _validate_video_latent(
                noise.by_role("video"), channels=model.config.out_channels
            )
            if tuple(noise_video.shape) != tuple(video.shape):
                raise Wan21RuntimeError(
                    "Wan custom sampling noise must match the video latent shape"
                )

            def text_lane(value: object | None) -> Conditioning[torch.Tensor] | None:
                if value is None:
                    return None
                if type(value) is not PreparedMultiStreamConditioning:
                    raise Wan21RuntimeError(
                        "Wan custom sampling requires prepared multistream conditioning"
                    )
                if value.runtime_identity != self.conditioning_identity:
                    raise Wan21RuntimeError(
                        "Wan conditioning was prepared by a different conditioner component"
                    )
                if type(value.payload) is not Wan21PreparedConditioning:
                    raise TypeError("conditioning must be exact Wan21PreparedConditioning")
                return Conditioning(value.payload.text)

            prepared_cond = text_lane(cond)
            if prepared_cond is None:
                raise Wan21RuntimeError("Wan CausalAR requires positive conditioning")
            cond = prepared_cond
            if cfg is not None:
                if type(cfg) is not SamplingGuidance:
                    raise Wan21RuntimeError("Wan CausalAR supports only CFG guidance")
                cfg = replace(cfg, uncond=text_lane(cfg.uncond))
            latent, noise = video, noise_video
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=Wan21RuntimeError,
        )
        if latent.ndim != 5 or latent.shape[1] != 16:
            raise Wan21RuntimeError("Wan CausalAR input must be [batch,16,frames,height,width]")
        if cfg is not None and (type(cfg.scale) not in (int, float) or float(cfg.scale) != 1.0):
            raise Wan21RuntimeError("Wan CausalAR requires CFG exactly 1.0")
        if cfg is not None and cfg.transforms:
            raise Wan21RuntimeError("Wan CausalAR does not support guidance transforms")
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=Wan21RuntimeError
        )
        space = FlowSigmas(shift=5.0)
        schedule = build_custom_sampling_schedule(request.sigmas, space, sampler, flow=True)
        assert type(model) is Wan21CausalModel
        text = _validate_text(cond.embeddings, text_dim=model.config.text_dim)
        if text.shape[0] not in (1, latent.shape[0]):
            raise Wan21RuntimeError("Wan CausalAR text batch must be one or match the video batch")
        initial = None
        if initial_latent is not None:
            initial = _validate_video_latent(initial_latent, channels=model.config.out_channels)
            if (
                initial.shape[0] not in (1, latent.shape[0])
                or initial.shape[2] > latent.shape[2]
                or initial.shape[3:] != latent.shape[3:]
            ):
                raise Wan21RuntimeError("Wan CausalAR initial latent must match the target video")
        load_device = (
            bound_compute_device(model.patch_embedding) or model.patch_embedding.weight.device
            if device is None
            else torch.device(device)
        )
        selected_dtype = (
            self.assembled.compute_dtype("diffusion") or torch.bfloat16
            if compute_dtype is None
            else compute_dtype
        )
        prepared_initial = (
            None if initial is None else self._latent_process_in(initial.to(load_device))
        )
        denoiser = _Wan21CausalDenoiser(
            model,
            text,
            initial_latent=prepared_initial,
            compute_dtype=selected_dtype,
        )
        captured: list[torch.Tensor] = []

        def report_step(event: StepEvent) -> None:
            _check_cancelled(cancelled)
            if on_step is not None:
                on_step(event)
            _check_cancelled(cancelled)

        def report_state(event: SamplingStateEvent[object]) -> None:
            _check_cancelled(cancelled)
            if event.denoised is not None:
                if type(event.denoised) is not torch.Tensor:
                    raise TypeError("Wan CausalAR denoised state must contain a torch.Tensor")
                captured[:] = [self._latent_process_out(event.denoised)]
            if on_state is not None:
                on_state(
                    replace(
                        event,
                        current=MultiStreamLatent.from_pairs((("video", event.current),)),
                        denoised=(
                            None
                            if event.denoised is None
                            else MultiStreamLatent.from_pairs((("video", event.denoised),))
                        ),
                    )
                    if structural
                    else event
                )
            _check_cancelled(cancelled)

        output = run_sampler_engine(
            denoiser,
            request.build_solver(),
            latent=latent,
            noise=noise,
            sigmas=schedule.sigmas,
            initial_sigma=schedule.initial_sigma,
            parameterization=Parameterization.FLOW,
            sigma_min=space.sigma_min,
            sigma_max=space.sigma_max,
            process_in=self._latent_process_in,
            process_out=self._latent_process_out,
            seed=seed,
            noise_kind=sampler.noise,
            percent_to_sigma=space.percent_to_sigma,
            device=load_device,
            on_step=report_step,
            on_state=report_state if capture_denoised or on_state is not None else None,
        )
        _check_cancelled(cancelled)
        denoised_output = captured[-1] if captured else (output if capture_denoised else None)
        if structural:
            return CustomSamplingResult(
                MultiStreamLatent.from_pairs((("video", output),)),
                None
                if denoised_output is None
                else MultiStreamLatent.from_pairs((("video", denoised_output),)),
            )
        return CustomSamplingResult(output, denoised_output)

    @property
    def dense_custom_sampling_role(self) -> str | None:
        return "video" if isinstance(self.assembled.diffusion, Wan21CausalModel) else None

    def custom_sampling_latent_kwargs(self, latent: Mapping[object, object]) -> dict[str, object]:
        if not isinstance(self.assembled.diffusion, Wan21CausalModel):
            return {}
        if "batch_index" in latent:
            raise Wan21RuntimeError("Wan CausalAR does not accept per-batch noise indices")
        initial = latent.get(WAN21_CAUSAL_INITIAL_LATENT_KEY)
        if initial is None:
            return {}
        if type(initial) is not torch.Tensor:
            raise TypeError("Wan CausalAR initial latent metadata must be an exact torch.Tensor")
        return {"initial_latent": initial}

    def prepare_custom_sampling_noise(
        self, latent: MultiStreamLatent[torch.Tensor], seed: int, noise_inds: Sequence[int] | None
    ) -> MultiStreamLatent[torch.Tensor]:
        if isinstance(self.assembled.diffusion, Wan21CausalModel):
            if noise_inds is not None:
                raise Wan21RuntimeError("Wan CausalAR does not accept per-batch noise indices")
            return MultiStreamLatent.from_pairs(
                (("video", prepare_noise(latent.by_role("video"), seed)),)
            )
        return prepare_multistream_noise(latent, seed, noise_inds)

    def adapt_multistream_latent(
        self,
        latent: torch.Tensor,
        *,
        source_spatial_downscale: int | None = None,
        source_temporal_downscale: int | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        descriptor = self.family.single_stream_latent()
        if source_spatial_downscale not in (None, descriptor.spatial_downscale):
            raise Wan21RuntimeError("Wan latent spatial scale does not match the loaded model")
        if source_temporal_downscale not in (None, descriptor.temporal_downscale):
            raise Wan21RuntimeError("Wan latent temporal scale does not match the loaded model")
        channels = self.assembled.diffusion.config.out_channels
        return MultiStreamLatent.from_pairs(
            (("video", _validate_video_latent(latent, channels=channels)),)
        )

    def _ksampler_kwargs(
        self, scheduler_id: str, noise_inds: Sequence[int] | None, kwargs: dict[str, object]
    ) -> dict[str, object]:
        if type(self.assembled.diffusion) is Wan21CausalModel:
            raise Wan21RuntimeError("Wan CausalAR requires the ar_video custom sampler")
        return MultiStreamSamplingRuntime._ksampler_kwargs(self, scheduler_id, noise_inds, kwargs)

    def _ksampler_noise(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        seed: int,
        noise_inds: Sequence[int] | None,
        add_noise: bool,
    ) -> MultiStreamLatent[torch.Tensor]:
        _, video = _video_streams(latent, channels=self.assembled.diffusion.config.out_channels)
        noise = prepare_noise(video, seed) if add_noise else torch.zeros_like(video)
        return MultiStreamLatent.from_pairs((("video", noise),))

    def _sample_standard_custom(
        self,
        latent: CustomSamplingLatentValue,
        *,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        request: CustomSamplingRequest[torch.Tensor],
        seed: int,
        guidance: float | None,
        denoise_mask: CustomSamplingLatentValue | None,
        inpaint: object | None,
        context_windows: ContextWindowsSpec | None,
        on_step: StepCallback | None,
        on_state: SamplingStateCallback | None,
        cancelled: Callable[[], bool] | None,
        observer: ExecutionObserverAttachment | None,
        parent_span_id: int | None,
        compute_dtype: torch.dtype | None,
        device: torch.device | str | None,
        sampling_shift: float | None,
        uni3c: Wan21Uni3CExecution | None,
        multitalk: Wan21InfiniteTalkExecution | None,
        capture_denoised: bool,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        del observer, parent_span_id
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=Wan21RuntimeError
        )
        if cancelled is None:
            cancelled = sampling_environment_cancellation()
        _check_cancelled(cancelled)
        model = self.assembled.diffusion
        if type(model) is Wan21CausalModel:
            raise Wan21RuntimeError("Wan CausalAR requires the ar_video custom sampler")
        latent_channels = model.config.out_channels
        streams, video = _video_streams(latent, channels=latent_channels)
        if type(noise) is not MultiStreamLatent or noise.roles != streams.roles:
            raise Wan21RuntimeError("Wan custom sampling noise must match the latent stream roles")
        noise_video = _validate_video_latent(noise.by_role("video"), channels=latent_channels)
        if tuple(noise_video.shape) != tuple(video.shape):
            raise Wan21RuntimeError("Wan custom sampling noise must match the video latent shape")
        if type(cond) is not PreparedMultiStreamConditioning:
            raise Wan21RuntimeError(
                "Wan custom sampling requires prepared multistream conditioning"
            )
        if cond.runtime_identity != self.conditioning_identity:
            raise Wan21RuntimeError(
                "Wan conditioning was prepared by a different conditioner component"
            )
        conditioning = cond.payload
        if type(conditioning) is not Wan21PreparedConditioning:
            raise TypeError("conditioning must be exact Wan21PreparedConditioning")

        def unwrap_lane(value: object | None, what: str) -> Wan21PreparedConditioning | None:
            if value is None:
                return None
            if type(value) is not PreparedMultiStreamConditioning:
                raise Wan21RuntimeError(f"{what} must use prepared multistream conditioning")
            if value.runtime_identity != cond.runtime_identity:
                raise Wan21RuntimeError("Wan guidance lanes were prepared by different components")
            payload = value.payload
            if type(payload) is not Wan21PreparedConditioning:
                raise TypeError(f"{what} must contain exact Wan21PreparedConditioning")
            return payload

        if cfg is None:
            guidance_cfg: (
                SamplingGuidance[Wan21PreparedConditioning]
                | DualSamplingGuidance[Wan21PreparedConditioning]
                | None
            ) = None
        elif isinstance(cfg, DualSamplingGuidance):
            guidance_cfg = cast(
                "DualSamplingGuidance[Wan21PreparedConditioning]",
                replace(
                    cfg,
                    uncond=unwrap_lane(cfg.uncond, "unconditional conditioning"),
                    middle=unwrap_lane(cfg.middle, "middle conditioning"),
                ),
            )
        elif type(cfg) is SamplingGuidance:
            guidance_cfg = cast(
                "SamplingGuidance[Wan21PreparedConditioning]",
                replace(
                    cfg,
                    uncond=unwrap_lane(cfg.uncond, "unconditional conditioning"),
                ),
            )
        else:
            raise Wan21RuntimeError("Wan custom sampling requires basic or dual CFG guidance")
        admitted_uni3c = None
        if uni3c is not None:
            if type(uni3c) is not Wan21Uni3CExecution:
                raise TypeError("uni3c must be exact Wan21Uni3CExecution")
            if type(model) is not Wan21Model or (
                model.config is not WAN21_T2V_14B and model.config is not WAN21_I2V_14B
            ):
                raise Wan21RuntimeError(
                    "Uni3C supports only exact native base Wan 2.1 T2V or I2V 14B"
                )
            if (
                uni3c.render_latent.shape[0] not in (1, video.shape[0])
                or uni3c.render_latent.shape[2:] != video.shape[2:]
            ):
                raise Wan21RuntimeError(
                    "Uni3C render latent batch and geometry must match the target video"
                )
            admitted_uni3c = snapshot_wan21_uni3c_execution(uni3c)
        admitted_multitalk = None
        if multitalk is not None:
            if admitted_uni3c is not None:
                raise Wan21RuntimeError("InfiniteTalk cannot be combined with Uni3C")
            if type(model) is not Wan21Model or model.config is not WAN21_I2V_14B:
                raise Wan21RuntimeError(
                    "InfiniteTalk supports only exact native base Wan 2.1 I2V 14B"
                )
            if video.shape[0] != 1:
                raise Wan21RuntimeError("InfiniteTalk requires a target video batch of one")
            admitted_multitalk = _snapshot_infinite_talk_execution(multitalk)
            motion = admitted_multitalk.motion_latent
            if motion.shape[2] > video.shape[2] or motion.shape[3:] != video.shape[3:]:
                raise Wan21RuntimeError(
                    "InfiniteTalk motion latent must fit and spatially match the target video"
                )
        text_dim = model.config.text_dim

        def validate_prepared(value: Wan21PreparedConditioning, what: str) -> None:
            _validate_text(value.text, text_dim=text_dim)
            if admitted_uni3c is not None and value.temporal_reference is not None:
                raise Wan21RuntimeError("Uni3C cannot be combined with Phantom references")
            humo_fields = (value.humo_audio_embed, value.humo_reference_latent)
            s2v_fields = (
                value.audio_embed,
                value.s2v_reference_latent,
                value.s2v_reference_motion,
                value.s2v_control_video,
            )
            dancer_fields = (
                value.dancer_audio_embed,
                value.dancer_reference_vision,
                value.dancer_settings,
            )
            if model.config.model_variant == "wandancer":
                if type(model) is not Wan22DancerModel or model.config is not WAN22_WANDANCER_14B:
                    raise Wan21RuntimeError(
                        "WanDancer requires the exact native Wan 2.2 14B profile"
                    )
                if (
                    value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.context_latents
                    or value.pose_latents is not None
                    or value.face_pixel_values is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.pose_schedule is not None
                    or value.animate2_settings is not None
                    or value.scail_reference_latent is not None
                    or value.scail_reference_mask is not None
                    or value.scail_driving_mask is not None
                    or value.scail_replacement
                    or any(field is not None for field in humo_fields)
                    or any(field is not None for field in s2v_fields)
                ):
                    raise Wan21RuntimeError(
                        f"WanDancer {what} contains conditioning for another Wan variant"
                    )
                if type(value.dancer_settings) is not Wan22DancerSettings:
                    raise Wan21RuntimeError(f"WanDancer {what} requires exact settings")
                if value.concat_latent is not None:
                    concat = _validate_concat_latent(value.concat_latent, channels=20)
                    if value.concat_mask_index != 0:
                        raise Wan21RuntimeError(f"WanDancer {what} concat mask index must be 0")
                    if concat.shape[2:] != video.shape[2:]:
                        raise Wan21RuntimeError(
                            "WanDancer CONCAT_LATENT temporal and spatial shape must match "
                            "the video latent"
                        )
                elif value.concat_mask_index is not None:
                    raise Wan21RuntimeError(
                        f"WanDancer {what} concat mask index requires CONCAT_LATENT"
                    )
                for name, embedded in (
                    ("vision", value.vision),
                    ("reference vision", value.dancer_reference_vision),
                ):
                    if embedded is None:
                        continue
                    embedded = _validate_vision(embedded, allowed_rows=(257,))
                    if embedded.shape[0] not in (1, video.shape[0]):
                        raise Wan21RuntimeError(
                            f"WanDancer {name} batch must be one or match the target video"
                        )
                if value.dancer_audio_embed is not None:
                    audio = _validate_dancer_audio(value.dancer_audio_embed)
                    if audio.shape[0] not in (1, video.shape[0]):
                        raise Wan21RuntimeError(
                            "WanDancer audio batch must be one or match the target video"
                        )
                return
            if model.config.model_variant == "humo":
                if type(model) is not Wan21HumoModel or model.config is not WAN21_HUMO_17B:
                    raise Wan21RuntimeError(
                        "HuMo requires the exact native Wan 2.1 HuMo 17B profile"
                    )
                if (
                    value.concat_latent is not None
                    or value.vision is not None
                    or value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.context_latents
                    or value.pose_latents is not None
                    or value.face_pixel_values is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.pose_schedule is not None
                    or value.animate2_settings is not None
                    or value.scail_reference_latent is not None
                    or value.scail_reference_mask is not None
                    or value.scail_driving_mask is not None
                    or value.scail_replacement
                    or any(field is not None for field in s2v_fields)
                    or any(field is not None for field in dancer_fields)
                ):
                    raise Wan21RuntimeError(
                        f"Wan HuMo {what} contains conditioning for another Wan variant"
                    )
                if value.humo_audio_embed is None or value.humo_reference_latent is None:
                    raise Wan21RuntimeError(
                        f"Wan HuMo {what} requires AUDIO_EMBEDDING and REFERENCE_LATENT"
                    )
                audio = _validate_humo_audio(value.humo_audio_embed)
                reference = _validate_video_latent(value.humo_reference_latent)
                if audio.shape[0] not in (1, video.shape[0]):
                    raise Wan21RuntimeError(
                        "Wan HuMo audio batch must be one or match the target video"
                    )
                if audio.shape[1] != video.shape[2]:
                    raise Wan21RuntimeError(
                        "Wan HuMo audio must contain one window per target latent frame"
                    )
                if reference.shape[0] not in (1, video.shape[0]):
                    raise Wan21RuntimeError(
                        "Wan HuMo reference batch must be one or match the target video"
                    )
                if reference.shape[-2:] != video.shape[-2:]:
                    raise Wan21RuntimeError(
                        "Wan HuMo reference spatial geometry must match the target video"
                    )
                return
            if any(field is not None for field in humo_fields):
                raise Wan21RuntimeError(f"Wan {what} contains HuMo conditioning")
            if model.config.model_variant == "s2v":
                if type(model) is not Wan22S2VModel or model.config is not WAN22_S2V_14B:
                    raise Wan21RuntimeError("S2V requires the exact native Wan 2.2 14B profile")
                if (
                    value.concat_latent is not None
                    or value.vision is not None
                    or value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.context_latents
                    or value.pose_latents is not None
                    or value.face_pixel_values is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.pose_schedule is not None
                    or value.animate2_settings is not None
                    or value.scail_reference_latent is not None
                    or value.scail_reference_mask is not None
                    or value.scail_driving_mask is not None
                    or value.scail_replacement
                    or any(field is not None for field in dancer_fields)
                ):
                    raise Wan21RuntimeError(
                        f"Wan S2V {what} contains conditioning for another Wan variant"
                    )
                if value.s2v_control_video is None:
                    raise Wan21RuntimeError(f"Wan S2V {what} requires CONTROL_VIDEO")
                if value.audio_embed is not None:
                    audio = _validate_s2v_audio(value.audio_embed)
                    if audio.shape[3] < video.shape[2] * 4:
                        raise Wan21RuntimeError(
                            "Wan S2V audio must provide four samples per target latent frame"
                        )
                for name, latent in (
                    ("reference", value.s2v_reference_latent),
                    ("motion", value.s2v_reference_motion),
                    ("control", value.s2v_control_video),
                ):
                    if latent is None:
                        continue
                    latent = _validate_video_latent(latent, channels=model.config.out_channels)
                    if latent.shape[0] not in (1, video.shape[0]):
                        raise Wan21RuntimeError(
                            f"Wan S2V {name} batch must be one or match the target video"
                        )
                    if latent.shape[-2:] != video.shape[-2:]:
                        raise Wan21RuntimeError(
                            f"Wan S2V {name} spatial geometry must match the target video"
                        )
                reference = value.s2v_reference_latent
                if reference is not None and reference.shape[2] != 1:
                    raise Wan21RuntimeError("Wan S2V reference latent must contain one frame")
                control = value.s2v_control_video
                assert control is not None
                if control.shape[2] != video.shape[2]:
                    raise Wan21RuntimeError(
                        "Wan S2V control temporal geometry must match the target video"
                    )
                return
            if any(field is not None for field in s2v_fields):
                raise Wan21RuntimeError(f"Wan {what} contains S2V conditioning")
            if any(field is not None for field in dancer_fields):
                raise Wan21RuntimeError(f"Wan {what} contains WanDancer conditioning")
            requires_concat = model.config.in_channels > model.config.out_channels
            accepts_vision = model.config.model_type == "i2v"
            requires_vace = model.config.vace_layers is not None
            requires_camera = model.config.camera_channels is not None
            animate = model.config.model_variant == "animate"
            animate2 = model.config.model_variant == "animate2"
            scail = model.config.model_variant in ("scail", "scail2")
            if value.context_latents:
                if (
                    admitted_uni3c is not None
                    or type(model) is not Wan21Model
                    or model.config.model_variant != "bernini"
                ):
                    raise Wan21RuntimeError(
                        "Bernini context latents require exact native Wan 2.2 14B T2V"
                    )
                if (
                    value.concat_latent is not None
                    or value.vision is not None
                    or value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.pose_latents is not None
                    or value.face_pixel_values is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.pose_schedule is not None
                    or value.animate2_settings is not None
                    or value.scail_reference_latent is not None
                    or value.scail_reference_mask is not None
                    or value.scail_driving_mask is not None
                    or value.scail_replacement
                ):
                    raise Wan21RuntimeError(
                        f"Wan Bernini {what} contains conditioning for another Wan variant"
                    )
                for latent in value.context_latents:
                    latent = _validate_video_latent(latent, channels=model.config.out_channels)
                    if latent.shape[0] not in (1, video.shape[0]):
                        raise Wan21RuntimeError(
                            "Wan Bernini context latent batch must be one or match the video batch"
                        )
                return
            if scail:
                if (
                    value.concat_latent is not None
                    or value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.face_pixel_values is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.animate2_settings is not None
                ):
                    raise Wan21RuntimeError(
                        f"Wan SCAIL {what} contains conditioning for another Wan variant"
                    )
                reference = value.scail_reference_latent
                if reference is not None:
                    reference = _validate_video_latent(
                        reference, channels=model.config.out_channels
                    )
                    if reference.shape[-2:] != video.shape[-2:]:
                        raise Wan21RuntimeError(
                            "Wan SCAIL reference spatial shape must match the video latent"
                        )
                if value.vision is not None:
                    _validate_vision(value.vision, allowed_rows=(257,))
                pose = value.pose_latents
                if pose is None:
                    if value.pose_schedule is not None or value.scail_driving_mask is not None:
                        raise Wan21RuntimeError(
                            f"Wan SCAIL {what} pose schedule and driving mask require pose latents"
                        )
                else:
                    pose = _validate_video_latent(pose, channels=model.config.out_channels)
                    if type(value.pose_schedule) is not PercentRange:
                        raise Wan21RuntimeError(
                            f"Wan SCAIL {what} pose latents require a percent schedule"
                        )
                if model.config.model_variant == "scail":
                    if (
                        value.scail_reference_mask is not None
                        or value.scail_driving_mask is not None
                    ):
                        raise Wan21RuntimeError("Wan SCAIL does not consume identity masks")
                else:
                    if value.scail_reference_mask is not None:
                        if reference is None:
                            raise Wan21RuntimeError(
                                f"Wan SCAIL {what} reference mask requires reference latents"
                            )
                        reference_mask = _validate_video_latent(
                            value.scail_reference_mask,
                            channels=28,
                        )
                        if reference_mask.shape[2:] != (
                            reference.shape[2] + video.shape[2],
                            *video.shape[3:],
                        ):
                            raise Wan21RuntimeError(
                                "Wan SCAIL2 reference mask must match reference plus video geometry"
                            )
                    if value.scail_driving_mask is not None:
                        assert pose is not None
                        driving_mask = _validate_video_latent(
                            value.scail_driving_mask,
                            channels=28,
                        )
                        if driving_mask.shape[2:] != pose.shape[2:]:
                            raise Wan21RuntimeError(
                                "Wan SCAIL2 driving mask must match pose geometry"
                            )
                return
            if animate2:
                if (
                    value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.face_pixel_values is not None
                ):
                    raise Wan21RuntimeError(
                        f"Wan Animate2 {what} contains conditioning for another Wan variant"
                    )
                if value.concat_latent is None:
                    raise Wan21RuntimeError(f"Wan Animate2 {what} requires CONCAT_LATENT")
                concat = _validate_concat_latent(
                    value.concat_latent,
                    channels=model.config.in_channels - model.config.out_channels,
                )
                if value.concat_mask_index != 0:
                    raise Wan21RuntimeError(f"Wan Animate2 {what} concat mask index must be 0")
                if concat.shape[2:] != video.shape[2:]:
                    raise Wan21RuntimeError(
                        "Wan Animate2 CONCAT_LATENT temporal and spatial shape must match "
                        "the video latent"
                    )
                if value.vision is not None:
                    _validate_vision(value.vision, allowed_rows=(257,))
                if value.pose_text is not None:
                    _validate_text(value.pose_text, text_dim=text_dim)
                if value.pose_vision is not None:
                    _validate_vision(value.pose_vision, allowed_rows=(257,))
                complete_pose_record = (
                    value.pose_text is not None
                    and value.pose_schedule is not None
                    and value.animate2_settings is not None
                )
                has_pose_field = any(
                    item is not None
                    for item in (
                        value.pose_text,
                        value.pose_vision,
                        value.pose_latents,
                        value.pose_schedule,
                        value.animate2_settings,
                    )
                )
                if has_pose_field and not complete_pose_record:
                    raise Wan21RuntimeError(
                        "Wan Animate2 pose fields require one complete pose record"
                    )
                if value.pose_latents is not None:
                    pose = _validate_video_latent(
                        value.pose_latents,
                        channels=model.config.out_channels,
                    )
                    if pose.shape[-2:] != video.shape[-2:] or pose.shape[2] != video.shape[2] - 1:
                        raise Wan21RuntimeError(
                            "Wan Animate2 pose latent must match video spatial geometry and "
                            "have exactly generation frames minus the reference frame"
                        )
                if (
                    value.pose_schedule is not None
                    and type(value.pose_schedule) is not PercentRange
                ):
                    raise TypeError("Wan Animate2 pose schedule must be an exact PercentRange")
                if value.animate2_settings is not None and (
                    type(value.animate2_settings) is not Wan21Animate2Settings
                ):
                    raise TypeError("Wan Animate2 settings must be exact Wan21Animate2Settings")
                return
            if animate:
                if (
                    value.reference_latent is not None
                    or value.vace_frames
                    or value.vace_masks
                    or value.vace_strengths
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                    or value.pose_text is not None
                    or value.pose_vision is not None
                    or value.pose_schedule is not None
                    or value.animate2_settings is not None
                ):
                    raise Wan21RuntimeError(
                        f"Wan Animate {what} contains conditioning for another Wan variant"
                    )
                if value.concat_latent is None:
                    raise Wan21RuntimeError(f"Wan Animate {what} requires CONCAT_LATENT")
                concat = _validate_concat_latent(
                    value.concat_latent,
                    channels=model.config.in_channels - model.config.out_channels,
                )
                if value.concat_mask_index != 0:
                    raise Wan21RuntimeError(f"Wan Animate {what} concat mask index must be 0")
                if concat.shape[2:] != video.shape[2:]:
                    raise Wan21RuntimeError(
                        "Wan Animate CONCAT_LATENT temporal and spatial shape must match "
                        "the video latent"
                    )
                if value.vision is not None:
                    _validate_vision(value.vision, allowed_rows=(257,))
                if value.pose_latents is not None:
                    pose = _validate_video_latent(
                        value.pose_latents,
                        channels=model.config.out_channels,
                    )
                    if pose.shape[-2:] != video.shape[-2:] or pose.shape[2] >= video.shape[2]:
                        raise Wan21RuntimeError(
                            "Wan Animate pose latent must match video spatial geometry and "
                            "leave the leading reference frame"
                        )
                if value.face_pixel_values is not None:
                    _validate_face_pixel_values(value.face_pixel_values)
                return
            if (
                value.pose_latents is not None
                or value.face_pixel_values is not None
                or value.pose_text is not None
                or value.pose_vision is not None
                or value.pose_schedule is not None
                or value.animate2_settings is not None
            ):
                raise Wan21RuntimeError(f"Wan {what} does not consume Animate conditioning")
            if requires_vace:
                if (
                    value.concat_latent is not None
                    or value.vision is not None
                    or value.reference_latent is not None
                    or value.camera_conditions is not None
                    or value.temporal_reference is not None
                ):
                    raise Wan21RuntimeError(f"Wan VACE {what} must not contain I2V conditioning")
                if not value.vace_frames:
                    raise Wan21RuntimeError(f"Wan VACE {what} requires VACE conditioning")
                for frames, mask in zip(value.vace_frames, value.vace_masks, strict=True):
                    if frames.shape[2:] != video.shape[2:] or mask.shape[2:] != video.shape[2:]:
                        raise Wan21RuntimeError(
                            "Wan VACE temporal and spatial conditioning shape must match "
                            "the video latent"
                        )
                return
            if value.vace_frames or value.vace_masks or value.vace_strengths:
                raise Wan21RuntimeError(
                    f"Wan {model.config.model_type.upper()} {what} has VACE data"
                )
            if requires_camera:
                if value.reference_latent is not None or value.temporal_reference is not None:
                    raise Wan21RuntimeError(
                        f"Wan camera {what} must not contain reference-latent conditioning"
                    )
                concat_channels = model.config.in_channels - model.config.out_channels
                expected_mask_index = (
                    0 if value.concat_latent is not None and concat_channels == 20 else None
                )
                if value.concat_mask_index != expected_mask_index:
                    raise Wan21RuntimeError(
                        f"Wan camera {what} concat mask index must be {expected_mask_index}"
                    )
                if value.concat_latent is not None:
                    concat = _validate_concat_latent(
                        value.concat_latent,
                        channels=concat_channels,
                    )
                    if concat.shape[2:] != video.shape[2:]:
                        raise Wan21RuntimeError(
                            "Wan camera reference temporal and spatial shape must match "
                            "the video latent"
                        )
                if accepts_vision:
                    if value.vision is not None:
                        _validate_vision(value.vision, allowed_rows=(257,))
                elif value.vision is not None:
                    raise Wan21RuntimeError(
                        f"Wan 2.2 camera {what} must not contain VISION_EMBEDDING"
                    )
                if value.camera_conditions is not None:
                    camera_channels = model.config.camera_channels
                    assert camera_channels is not None
                    camera = _validate_camera_conditions(
                        value.camera_conditions,
                        channels=camera_channels,
                    )
                    expected_tail = (
                        video.shape[2],
                        video.shape[3] * 8,
                        video.shape[4] * 8,
                    )
                    if camera.shape[2:] != expected_tail:
                        raise Wan21RuntimeError(
                            "Wan camera trajectory temporal and spatial shape must match the video"
                        )
                return
            if value.camera_conditions is not None:
                raise Wan21RuntimeError(f"Wan {what} does not consume camera conditioning")
            if not requires_concat:
                if (
                    value.concat_latent is not None
                    or value.vision is not None
                    or value.reference_latent is not None
                ):
                    raise Wan21RuntimeError(
                        f"Wan {model.config.model_type.upper()} {what} must contain only TEXT"
                    )
                if value.temporal_reference is not None:
                    if model.config.model_variant != "base":
                        raise Wan21RuntimeError(
                            "Wan profile does not consume Phantom subject references"
                        )
                    reference = _validate_video_latent(
                        value.temporal_reference,
                        channels=model.config.out_channels,
                    )
                    if reference.shape[-2:] != video.shape[-2:]:
                        raise Wan21RuntimeError(
                            "Wan Phantom reference spatial shape must match the video latent"
                        )
                return
            if value.temporal_reference is not None:
                raise Wan21RuntimeError(f"Wan I2V {what} must not contain Phantom references")
            if value.concat_latent is None:
                raise Wan21RuntimeError(f"Wan I2V {what} requires CONCAT_LATENT")
            concat = _validate_concat_latent(
                value.concat_latent,
                channels=model.config.in_channels - model.config.out_channels,
            )
            expected_mask_index = _concat_mask_index(model.config)
            if value.concat_mask_index != expected_mask_index:
                raise Wan21RuntimeError(
                    f"Wan {what} concat mask index must be {expected_mask_index}"
                )
            if accepts_vision:
                if value.vision is not None:
                    _validate_vision(
                        value.vision,
                        allowed_rows=_allowed_vision_rows(model.config),
                    )
            elif value.vision is not None:
                raise Wan21RuntimeError(f"Wan 2.2 I2V {what} must not contain VISION_EMBEDDING")
            if concat.shape[2:] != video.shape[2:]:
                raise Wan21RuntimeError(
                    "Wan 2.1 I2V CONCAT_LATENT temporal and spatial shape must match "
                    "the video latent"
                )
            reference = value.reference_latent
            if reference is not None:
                if model.config.reference_channels is None:
                    raise Wan21RuntimeError(f"Wan {what} does not consume REFERENCE_LATENT")
                reference = _validate_reference_latent(
                    reference,
                    channels=model.config.reference_channels,
                )
                if reference.shape[-2:] != video.shape[-2:]:
                    raise Wan21RuntimeError(
                        "Wan REFERENCE_LATENT spatial shape must match the video latent"
                    )

        validate_prepared(conditioning, "conditioning")
        uncond = None if guidance_cfg is None else guidance_cfg.uncond
        if uncond is not None:
            validate_prepared(uncond, "unconditional conditioning")
        middle = guidance_cfg.middle if isinstance(guidance_cfg, DualSamplingGuidance) else None
        if middle is not None:
            validate_prepared(middle, "middle conditioning")
        bernini_conditioning = any(
            value is not None and bool(value.context_latents)
            for value in (conditioning, uncond, middle)
        )
        if context_windows is not None:
            if model.config.model_variant != "base" or model.config.model_type != "t2v":
                raise Wan21RuntimeError(
                    "Wan context windows support only the base text-to-video profiles"
                )
            if model.config.vace_layers is not None:
                raise Wan21RuntimeError("Wan context windows do not support VACE models")
            if admitted_uni3c is not None or admitted_multitalk is not None:
                raise Wan21RuntimeError(
                    "Wan context windows cannot be combined with Uni3C or InfiniteTalk"
                )
            for lane, what in (
                (conditioning, "conditioning"),
                (uncond, "unconditional conditioning"),
                (middle, "middle conditioning"),
            ):
                if lane is not None and not _text_only_prepared_conditioning(lane):
                    raise Wan21RuntimeError(
                        f"Wan context windows require text-only {what}; frame-aligned or "
                        "structural conditioning cannot be windowed"
                    )
        _check_cancelled(cancelled)
        sigmas = _wan_custom_space(self.assembled, sampling_shift)
        schedule_plan = build_custom_sampling_schedule(
            request.sigmas,
            sigmas,
            sampler,
            flow=True,
        )
        schedule = schedule_plan.sigmas
        guidance_plan = compile_guidance_plan(conditioning, guidance_cfg, sampler, None)
        weight = model.patch_embedding.weight
        model_device = bound_compute_device(model.patch_embedding) or weight.device
        load_device = model_device if device is None else torch.device(device)
        selected_dtype = (
            self.assembled.compute_dtype("diffusion") or torch.bfloat16
            if compute_dtype is None
            else compute_dtype
        )
        uni3c_render = (
            None
            if admitted_uni3c is None
            else admitted_uni3c.render_latent.to(
                device=load_device,
                dtype=selected_dtype,
            )
        )
        if uni3c_render is not None and uni3c_render.shape[0] == 1:
            uni3c_render = uni3c_render.expand(video.shape[0], *uni3c_render.shape[1:])
        pose_cache: PoseBranchCache | None = None
        video = video.to(load_device)
        prepared_multitalk = None
        multitalk_motion = None
        if admitted_multitalk is not None:
            from .wan21_multitalk import (
                Wan21MultiTalkExecution,
                wan21_multitalk_tensor_digest,
            )

            patch = admitted_multitalk.patch
            if type(patch) is not Wan21MultiTalkExecution:
                raise TypeError("InfiniteTalk patch must be exact Wan21MultiTalkExecution")
            audio_context = patch.audio_context.to(
                device=load_device,
                dtype=selected_dtype,
            )
            target_masks = (
                None if patch.target_masks is None else patch.target_masks.to(device=load_device)
            )
            prepared_multitalk = Wan21MultiTalkExecution(
                patch.model,
                audio_context,
                target_masks,
                patch.strength,
                patch.model_digest,
                wan21_multitalk_tensor_digest(audio_context),
                (None if target_masks is None else wan21_multitalk_tensor_digest(target_masks)),
            )
            multitalk_motion = self._latent_process_in(
                admitted_multitalk.motion_latent.to(device=load_device)
            ).to(dtype=selected_dtype)
        humo_target_concat = (
            _humo_target_concat(video.to(dtype=selected_dtype))
            if model.config.model_variant == "humo"
            else None
        )
        prepared_denoise_mask: torch.Tensor | None = None
        frame_denoise_mask: torch.Tensor | None = None
        scail_history: torch.Tensor | None = None
        if denoise_mask is not None:
            if type(denoise_mask) is not torch.Tensor:
                raise TypeError("Wan denoise_mask must be an exact torch.Tensor")
            allowed_channels = (
                (1, 4, video.shape[1])
                if model.config.model_variant == "scail2"
                else (1, video.shape[1])
            )
            if (
                denoise_mask.ndim != 5
                or denoise_mask.shape[0] != video.shape[0]
                or denoise_mask.shape[1] not in allowed_channels
                or denoise_mask.shape[2:] != video.shape[2:]
            ):
                raise Wan21RuntimeError(
                    "Wan denoise_mask must match the video latent with an accepted channel count"
                )
            mask = denoise_mask.to(device=load_device, dtype=torch.float32)
            if not torch.all((mask >= 0.0) & (mask <= 1.0)):
                raise Wan21RuntimeError("Wan denoise_mask values must be in [0, 1]")
            if model.config.model_variant == "scail2":
                collapsed_mask = mask.mean(dim=1, keepdim=True)
                prepared_denoise_mask = (
                    mask
                    if mask.shape[1] == video.shape[1]
                    else mask.expand_as(video)
                    if mask.shape[1] == 1
                    else mask.repeat(1, video.shape[1] // mask.shape[1], 1, 1, 1)
                ).contiguous()
                history_mask = (
                    mask if mask.shape[1] == 4 else collapsed_mask.expand(-1, 4, -1, -1, -1)
                )
                scail_history = 1.0 - history_mask
            else:
                prepared_denoise_mask = (
                    mask.expand_as(video) if mask.shape[1] == 1 else mask
                ).contiguous()
                if model.config.model_type == "ti2v":
                    frame_denoise_mask = mask.mean(dim=(1, 3, 4))
        token_plan = plan_wan21_token_layout(
            Wan21VideoLatentGeometry(video.shape[2], video.shape[3], video.shape[4])
        )
        parameterization = (
            Parameterization.IMAGE_TO_IMAGE_FLOW
            if model.config.model_variant == "flow_rvs"
            else Parameterization.FLOW
        )

        def prepare_conditioning(value: object, _role: GuidanceRole) -> _Wan21ModelConditioning:
            _check_cancelled(cancelled)
            if type(value) is not Wan21PreparedConditioning:
                raise TypeError("Wan 2.1 guidance lanes require exact prepared conditioning")
            text = _validate_text(value.text, text_dim=model.config.text_dim).to(
                device=load_device, dtype=selected_dtype
            )
            concat = value.concat_latent
            vision = value.vision
            reference = value.reference_latent
            temporal_reference = value.temporal_reference
            context_latents: list[torch.Tensor] = []
            for latent in value.context_latents:
                normalized = self.assembled.vae.process_in(
                    _validate_video_latent(
                        latent,
                        channels=model.config.out_channels,
                    ).to(device=load_device)
                ).to(dtype=selected_dtype)
                if normalized.shape[0] == 1 and video.shape[0] != 1:
                    normalized = normalized.expand(video.shape[0], -1, -1, -1, -1)
                context_latents.append(normalized)
            pose_latents = value.pose_latents
            face_pixel_values = value.face_pixel_values
            pose_text = value.pose_text
            pose_vision = value.pose_vision
            scail_reference = value.scail_reference_latent
            scail_reference_mask = value.scail_reference_mask
            scail_driving_mask = value.scail_driving_mask
            audio_embed = value.audio_embed
            s2v_reference = value.s2v_reference_latent
            s2v_motion = value.s2v_reference_motion
            s2v_control = value.s2v_control_video
            humo_audio = value.humo_audio_embed
            humo_reference = value.humo_reference_latent
            dancer_audio = value.dancer_audio_embed
            dancer_reference_vision = value.dancer_reference_vision

            def normalize_s2v_latent(raw: torch.Tensor | None) -> torch.Tensor | None:
                if raw is None:
                    return None
                normalized = self._latent_process_in(
                    _validate_video_latent(raw, channels=model.config.out_channels).to(
                        device=load_device
                    )
                ).to(dtype=selected_dtype)
                if normalized.shape[0] == 1 and video.shape[0] != 1:
                    normalized = normalized.expand(video.shape[0], -1, -1, -1, -1)
                return normalized

            if audio_embed is not None:
                audio_embed = _validate_s2v_audio(audio_embed).to(
                    device=load_device, dtype=selected_dtype
                )
                if audio_embed.shape[0] == 1 and video.shape[0] != 1:
                    audio_embed = audio_embed.expand(video.shape[0], -1, -1, -1)
            s2v_reference = normalize_s2v_latent(s2v_reference)
            s2v_motion = normalize_s2v_latent(s2v_motion)
            s2v_control = normalize_s2v_latent(s2v_control)
            if humo_audio is not None:
                humo_audio = _validate_humo_audio(humo_audio).to(
                    device=load_device,
                    dtype=selected_dtype,
                )
                if humo_audio.shape[0] == 1 and video.shape[0] != 1:
                    humo_audio = humo_audio.expand(video.shape[0], -1, -1, -1, -1)
            if humo_reference is not None:
                normalized_reference = self._latent_process_in(
                    _validate_video_latent(humo_reference).to(device=load_device)
                ).to(dtype=selected_dtype)
                if normalized_reference.shape[0] == 1 and video.shape[0] != 1:
                    normalized_reference = normalized_reference.expand(
                        video.shape[0], -1, -1, -1, -1
                    )
                humo_reference = normalized_reference.new_zeros(
                    normalized_reference.shape[0],
                    36,
                    *normalized_reference.shape[2:],
                )
                humo_reference[:, 16:20] = 1.0
                humo_reference[:, 20:] = normalized_reference
            if dancer_audio is not None:
                dancer_audio = _validate_dancer_audio(dancer_audio).to(
                    device=load_device,
                    dtype=selected_dtype,
                )
                if dancer_audio.shape[0] == 1 and video.shape[0] != 1:
                    dancer_audio = dancer_audio.expand(video.shape[0], -1, -1)
            if dancer_reference_vision is not None:
                dancer_reference_vision = _validate_vision(
                    dancer_reference_vision,
                    allowed_rows=(257,),
                ).to(device=load_device, dtype=selected_dtype)
                if dancer_reference_vision.shape[0] == 1 and video.shape[0] != 1:
                    dancer_reference_vision = dancer_reference_vision.expand(video.shape[0], -1, -1)
            if concat is not None:
                concat = _validate_concat_latent(
                    concat,
                    channels=model.config.in_channels - model.config.out_channels,
                ).to(device=load_device)
                concat = _normalize_concat_latent(
                    concat,
                    latent_channels=model.config.out_channels,
                    mask_index=value.concat_mask_index,
                    process_in=self._latent_process_in,
                ).to(dtype=selected_dtype)
            if vision is not None:
                vision = _validate_vision(vision).to(device=load_device, dtype=selected_dtype)
            if reference is not None:
                reference = self._latent_process_in(
                    _validate_reference_latent(
                        reference,
                        channels=model.config.reference_channels,
                    ).to(device=load_device)
                )[:, :, 0].to(dtype=selected_dtype)
            if temporal_reference is not None:
                temporal_reference = self._latent_process_in(
                    _validate_video_latent(
                        temporal_reference,
                        channels=model.config.out_channels,
                    ).to(device=load_device)
                ).to(dtype=selected_dtype)
            if pose_latents is not None:
                pose_latents = self._latent_process_in(
                    _validate_video_latent(
                        pose_latents,
                        channels=model.config.out_channels,
                    ).to(device=load_device)
                ).to(dtype=selected_dtype)
                if model.config.model_variant in ("scail", "scail2"):
                    pose_latents = torch.cat(
                        (pose_latents, torch.ones_like(pose_latents[:, :4])), dim=1
                    )
            if scail_reference is not None:
                scail_reference = self._latent_process_in(
                    _validate_video_latent(scail_reference).to(device=load_device)
                ).to(dtype=selected_dtype)
                scail_reference = torch.cat(
                    (scail_reference, torch.ones_like(scail_reference[:, :4])), dim=1
                )
            if scail_reference_mask is not None:
                scail_reference_mask = _validate_video_latent(scail_reference_mask, channels=28).to(
                    device=load_device, dtype=selected_dtype
                )
            if scail_driving_mask is not None:
                scail_driving_mask = _validate_video_latent(scail_driving_mask, channels=28).to(
                    device=load_device, dtype=selected_dtype
                )
            if face_pixel_values is not None:
                face_pixel_values = _validate_face_pixel_values(face_pixel_values).to(
                    device=load_device,
                    dtype=selected_dtype,
                )
            if pose_text is not None:
                pose_text = _validate_text(pose_text, text_dim=model.config.text_dim).to(
                    device=load_device,
                    dtype=selected_dtype,
                )
            if pose_vision is not None:
                pose_vision = _validate_vision(pose_vision, allowed_rows=(257,)).to(
                    device=load_device,
                    dtype=selected_dtype,
                )
            frames = value.vace_frames
            masks = value.vace_masks
            strengths = value.vace_strengths
            vace_context = None
            if frames:
                controls: list[torch.Tensor] = []
                for raw_frames, mask in zip(frames, masks, strict=True):
                    raw_frames = raw_frames.to(device=load_device)
                    normalized = torch.cat(
                        tuple(
                            self._latent_process_in(raw_frames[:, offset : offset + 16])
                            for offset in (0, 16)
                        ),
                        dim=1,
                    )
                    controls.append(
                        torch.cat((normalized, mask.to(device=load_device)), dim=1).to(
                            dtype=selected_dtype
                        )
                    )
                vace_context = torch.stack(controls, dim=1)
            camera = (
                None
                if value.camera_conditions is None
                else _validate_camera_conditions(
                    value.camera_conditions,
                    channels=cast("int", model.config.camera_channels),
                ).to(device=load_device, dtype=selected_dtype)
            )
            return _Wan21ModelConditioning(
                text=text,
                concat_latent=concat,
                vision=vision,
                reference_latent=reference,
                vace_context=vace_context,
                vace_strengths=strengths,
                camera_conditions=camera,
                temporal_reference=temporal_reference,
                context_latents=tuple(context_latents),
                pose_latents=pose_latents,
                face_pixel_values=face_pixel_values,
                pose_text=pose_text,
                pose_vision=pose_vision,
                pose_schedule=value.pose_schedule,
                animate2_settings=value.animate2_settings,
                scail_reference_latent=scail_reference,
                scail_reference_mask=scail_reference_mask,
                scail_driving_mask=scail_driving_mask,
                scail_replacement=value.scail_replacement,
                audio_embed=audio_embed,
                s2v_reference_latent=s2v_reference,
                s2v_reference_motion=s2v_motion,
                s2v_control_video=s2v_control,
                humo_audio_embed=humo_audio,
                humo_reference_latent=humo_reference,
                dancer_audio_embed=dancer_audio,
                dancer_reference_vision=dancer_reference_vision,
                dancer_settings=value.dancer_settings,
            )

        def evaluate_batch(
            x: torch.Tensor, sigma: float, contexts: tuple[_Wan21ModelConditioning, ...]
        ) -> tuple[torch.Tensor, ...]:
            _check_cancelled(cancelled)
            batch = x.shape[0]
            base_input = calculate_input(parameterization, sigma, x).to(selected_dtype)
            if multitalk_motion is not None:
                motion = to_batch(multitalk_motion, batch)
                base_input = torch.cat(
                    (motion, base_input[:, :, motion.shape[2] :]),
                    dim=2,
                )
            empty_model_concat = (
                base_input.new_zeros(
                    (
                        batch,
                        model.config.in_channels - model.config.out_channels,
                        *base_input.shape[2:],
                    )
                )
                if (
                    model.config.camera_channels is not None
                    or model.config.model_variant == "wandancer"
                )
                and contexts[0].concat_latent is None
                else None
            )
            model_inputs: list[torch.Tensor] = []
            for value in contexts:
                lane_input = base_input
                if model.config.model_variant in ("scail", "scail2"):
                    history = (
                        lane_input.new_zeros((batch, 4, *lane_input.shape[2:]))
                        if scail_history is None or model.config.model_variant == "scail"
                        else to_batch(scail_history, batch).to(dtype=selected_dtype)
                    )
                    lane_input = torch.cat((lane_input, history), dim=1)
                if humo_target_concat is not None:
                    lane_input = torch.cat(
                        (lane_input, _resize_concat_batch(humo_target_concat, batch)), dim=1
                    )
                if value.concat_latent is not None:
                    lane_input = torch.cat(
                        (lane_input, _resize_concat_batch(value.concat_latent, batch)), dim=1
                    )
                elif empty_model_concat is not None:
                    lane_input = torch.cat((lane_input, empty_model_concat), dim=1)
                model_inputs.append(lane_input)
            model_input = torch.cat(model_inputs)
            active_uni3c = (
                admitted_uni3c
                if admitted_uni3c is not None
                and admitted_uni3c.strength != 0.0
                and admitted_uni3c.window.is_active(sigma, sigmas)
                else None
            )
            uni3c_input = None
            if active_uni3c is not None:
                control_prefix = model_inputs[0][:, :20]
                if control_prefix.shape[1] < 20:
                    control_prefix = torch.cat(
                        (
                            control_prefix,
                            control_prefix.new_zeros(
                                control_prefix.shape[0],
                                20 - control_prefix.shape[1],
                                *control_prefix.shape[2:],
                            ),
                        ),
                        dim=1,
                    )
                assert uni3c_render is not None
                uni3c_input = torch.cat((control_prefix, uni3c_render), dim=1)
            context = torch.cat(tuple(to_batch(value.text, batch) for value in contexts))
            vision = (
                None
                if contexts[0].vision is None
                else torch.cat(
                    tuple(to_batch(cast("torch.Tensor", value.vision), batch) for value in contexts)
                )
            )
            vace_context = (
                None
                if contexts[0].vace_context is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.vace_context), batch)
                        for value in contexts
                    )
                )
            )
            reference = (
                None
                if contexts[0].reference_latent is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.reference_latent), batch)
                        for value in contexts
                    )
                )
            )
            audio_embed = (
                None
                if contexts[0].audio_embed is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.audio_embed), batch)
                        for value in contexts
                    )
                )
            )
            s2v_reference = (
                None
                if contexts[0].s2v_reference_latent is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.s2v_reference_latent), batch)
                        for value in contexts
                    )
                )
            )
            s2v_motion = (
                None
                if contexts[0].s2v_reference_motion is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.s2v_reference_motion), batch)
                        for value in contexts
                    )
                )
            )
            s2v_control = (
                None
                if contexts[0].s2v_control_video is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.s2v_control_video), batch)
                        for value in contexts
                    )
                )
            )
            humo_audio = (
                None
                if contexts[0].humo_audio_embed is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.humo_audio_embed), batch)
                        for value in contexts
                    )
                )
            )
            humo_reference = (
                None
                if contexts[0].humo_reference_latent is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.humo_reference_latent), batch)
                        for value in contexts
                    )
                )
            )
            dancer_audio = (
                None
                if contexts[0].dancer_audio_embed is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.dancer_audio_embed), batch)
                        for value in contexts
                    )
                )
            )
            dancer_reference_vision = (
                None
                if contexts[0].dancer_reference_vision is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.dancer_reference_vision), batch)
                        for value in contexts
                    )
                )
            )
            scail_reference = (
                None
                if contexts[0].scail_reference_latent is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.scail_reference_latent), batch)
                        for value in contexts
                    )
                )
            )
            scail_reference_mask = (
                None
                if contexts[0].scail_reference_mask is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.scail_reference_mask), batch)
                        for value in contexts
                    )
                )
            )
            scail_driving_mask = (
                None
                if contexts[0].scail_driving_mask is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.scail_driving_mask), batch)
                        for value in contexts
                    )
                )
            )
            camera = (
                None
                if contexts[0].camera_conditions is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.camera_conditions), batch)
                        for value in contexts
                    )
                )
            )
            temporal_reference = (
                None
                if contexts[0].temporal_reference is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.temporal_reference), batch)
                        for value in contexts
                    )
                )
            )
            context_latents = tuple(
                contexts[0].context_latents[index]
                if len(contexts) == 1
                else torch.cat(tuple(value.context_latents[index] for value in contexts))
                for index in range(len(contexts[0].context_latents))
            )
            pose_latents = (
                None
                if contexts[0].pose_latents is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.pose_latents), batch)
                        for value in contexts
                    )
                )
            )
            pose_schedule = contexts[0].pose_schedule
            pose_active = pose_latents is not None and (
                model.config.model_variant not in ("animate2", "scail", "scail2")
                or (pose_schedule is not None and pose_schedule.is_active(sigma, sigmas))
            )
            if not pose_active:
                pose_latents = None
                scail_driving_mask = None
            pose_context = (
                None
                if not pose_active or contexts[0].pose_text is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.pose_text), batch) for value in contexts
                    )
                )
            )
            pose_vision = (
                None
                if not pose_active or contexts[0].pose_vision is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.pose_vision), batch)
                        for value in contexts
                    )
                )
            )
            face_pixel_values = (
                None
                if contexts[0].face_pixel_values is None
                else torch.cat(
                    tuple(
                        to_batch(cast("torch.Tensor", value.face_pixel_values), batch)
                        for value in contexts
                    )
                )
            )
            timestep = sigmas.timestep(sigma)
            if frame_denoise_mask is None:
                timesteps = torch.full(
                    (model_input.shape[0],),
                    timestep,
                    device=load_device,
                    dtype=torch.float32,
                )
            else:
                lane_mask = to_batch(frame_denoise_mask, batch)
                timesteps = torch.cat((lane_mask,) * len(contexts)) * timestep
            if vace_context is None:
                if prepared_multitalk is not None:
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        multitalk=prepared_multitalk,
                    ).float()
                elif model.config.model_variant == "wandancer":
                    dancer_model = cast("Wan22DancerModel", model)
                    settings = contexts[0].dancer_settings
                    assert settings is not None
                    velocity = dancer_model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        reference_vision=dancer_reference_vision,
                        audio_embed=dancer_audio,
                        fps=settings.fps,
                        audio_inject_scale=settings.audio_inject_scale,
                    ).float()
                elif model.config.model_variant == "humo":
                    humo_model = cast("Wan21HumoModel", model)
                    assert humo_audio is not None and humo_reference is not None
                    velocity = humo_model(
                        model_input,
                        timesteps,
                        context,
                        audio_embed=humo_audio,
                        reference_latent=humo_reference,
                    ).float()
                elif model.config.model_variant == "s2v":
                    s2v_model = cast("Wan22S2VModel", model)
                    velocity = s2v_model(
                        model_input,
                        timesteps,
                        context,
                        audio_embed=audio_embed,
                        reference_latent=s2v_reference,
                        control_video=s2v_control,
                        reference_motion=s2v_motion,
                    ).float()
                elif model.config.model_variant in ("scail", "scail2"):
                    scail_model = cast("WanScailModel", model)
                    velocity = scail_model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        reference_latent=scail_reference,
                        pose_latents=pose_latents,
                        reference_mask=scail_reference_mask,
                        driving_mask=scail_driving_mask,
                        replacement=contexts[0].scail_replacement,
                    ).float()
                elif model.config.model_variant == "animate2":
                    animate2_model = cast("WanAnimate2Model", model)
                    settings = contexts[0].animate2_settings or Wan21Animate2Settings()
                    velocity = animate2_model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        pose_latents=pose_latents,
                        pose_context=pose_context,
                        pose_vision=pose_vision,
                        pose_strength=settings.pose_strength,
                        reference_strength=settings.reference_strength,
                        pose_cache=pose_cache if pose_active else None,
                    ).float()
                elif model.config.model_variant == "animate":
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        pose_latents=pose_latents,
                        face_pixel_values=face_pixel_values,
                    ).float()
                elif active_uni3c is not None:
                    assert uni3c_input is not None
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        uni3c=active_uni3c,
                        uni3c_input=uni3c_input,
                    ).float()
                elif context_latents:
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        context_latents=context_latents,
                    ).float()
                elif temporal_reference is not None:
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        temporal_reference=temporal_reference,
                    ).float()
                elif camera is not None:
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        camera_conditions=camera,
                    ).float()
                elif reference is None:
                    velocity = model(model_input, timesteps, context, vision).float()
                else:
                    velocity = model(
                        model_input,
                        timesteps,
                        context,
                        vision,
                        reference_latent=reference,
                    ).float()
            else:
                velocity = model(
                    model_input,
                    timesteps,
                    context,
                    vision,
                    vace_context=vace_context,
                    vace_strength=contexts[0].vace_strengths,
                ).float()
            outputs = velocity.chunk(len(contexts))
            return tuple(
                calculate_denoised(parameterization, sigma, output, x) for output in outputs
            )

        def batchable(contexts: tuple[_Wan21ModelConditioning, ...]) -> bool:
            if not contexts:
                return False
            first = contexts[0]
            return all(
                value.text.shape[1:] == first.text.shape[1:]
                and (
                    (value.concat_latent is None and first.concat_latent is None)
                    or (
                        value.concat_latent is not None
                        and first.concat_latent is not None
                        and value.concat_latent.shape[1:] == first.concat_latent.shape[1:]
                    )
                )
                and (
                    (value.vision is None and first.vision is None)
                    or (
                        value.vision is not None
                        and first.vision is not None
                        and value.vision.shape[1:] == first.vision.shape[1:]
                    )
                )
                and (
                    (value.reference_latent is None and first.reference_latent is None)
                    or (
                        value.reference_latent is not None
                        and first.reference_latent is not None
                        and value.reference_latent.shape[1:] == first.reference_latent.shape[1:]
                    )
                )
                and (
                    (value.audio_embed is None and first.audio_embed is None)
                    or (
                        value.audio_embed is not None
                        and first.audio_embed is not None
                        and value.audio_embed.shape[1:] == first.audio_embed.shape[1:]
                    )
                )
                and (
                    (value.s2v_reference_latent is None and first.s2v_reference_latent is None)
                    or (
                        value.s2v_reference_latent is not None
                        and first.s2v_reference_latent is not None
                        and value.s2v_reference_latent.shape[1:]
                        == first.s2v_reference_latent.shape[1:]
                    )
                )
                and (
                    (value.s2v_reference_motion is None and first.s2v_reference_motion is None)
                    or (
                        value.s2v_reference_motion is not None
                        and first.s2v_reference_motion is not None
                        and value.s2v_reference_motion.shape[1:]
                        == first.s2v_reference_motion.shape[1:]
                    )
                )
                and (
                    (value.s2v_control_video is None and first.s2v_control_video is None)
                    or (
                        value.s2v_control_video is not None
                        and first.s2v_control_video is not None
                        and value.s2v_control_video.shape[1:] == first.s2v_control_video.shape[1:]
                    )
                )
                and (
                    (value.humo_audio_embed is None and first.humo_audio_embed is None)
                    or (
                        value.humo_audio_embed is not None
                        and first.humo_audio_embed is not None
                        and value.humo_audio_embed.shape[1:] == first.humo_audio_embed.shape[1:]
                    )
                )
                and (
                    (value.humo_reference_latent is None and first.humo_reference_latent is None)
                    or (
                        value.humo_reference_latent is not None
                        and first.humo_reference_latent is not None
                        and value.humo_reference_latent.shape[1:]
                        == first.humo_reference_latent.shape[1:]
                    )
                )
                and (
                    (value.dancer_audio_embed is None and first.dancer_audio_embed is None)
                    or (
                        value.dancer_audio_embed is not None
                        and first.dancer_audio_embed is not None
                        and value.dancer_audio_embed.shape[1:] == first.dancer_audio_embed.shape[1:]
                    )
                )
                and (
                    (
                        value.dancer_reference_vision is None
                        and first.dancer_reference_vision is None
                    )
                    or (
                        value.dancer_reference_vision is not None
                        and first.dancer_reference_vision is not None
                        and value.dancer_reference_vision.shape[1:]
                        == first.dancer_reference_vision.shape[1:]
                    )
                )
                and value.dancer_settings == first.dancer_settings
                and value.vace_strengths == first.vace_strengths
                and (
                    (value.vace_context is None and first.vace_context is None)
                    or (
                        value.vace_context is not None
                        and first.vace_context is not None
                        and value.vace_context.shape[1:] == first.vace_context.shape[1:]
                    )
                )
                and (
                    (value.camera_conditions is None and first.camera_conditions is None)
                    or (
                        value.camera_conditions is not None
                        and first.camera_conditions is not None
                        and value.camera_conditions.shape[1:] == first.camera_conditions.shape[1:]
                    )
                )
                and (
                    (value.temporal_reference is None and first.temporal_reference is None)
                    or (
                        value.temporal_reference is not None
                        and first.temporal_reference is not None
                        and value.temporal_reference.shape[1:] == first.temporal_reference.shape[1:]
                    )
                )
                and len(value.context_latents) == len(first.context_latents)
                and all(
                    latent.shape[1:] == first.context_latents[index].shape[1:]
                    for index, latent in enumerate(value.context_latents)
                )
                and (
                    (value.pose_latents is None and first.pose_latents is None)
                    or (
                        value.pose_latents is not None
                        and first.pose_latents is not None
                        and value.pose_latents.shape[1:] == first.pose_latents.shape[1:]
                    )
                )
                and (
                    (value.pose_text is None and first.pose_text is None)
                    or (
                        value.pose_text is not None
                        and first.pose_text is not None
                        and value.pose_text.shape[1:] == first.pose_text.shape[1:]
                    )
                )
                and (
                    (value.pose_vision is None and first.pose_vision is None)
                    or (
                        value.pose_vision is not None
                        and first.pose_vision is not None
                        and value.pose_vision.shape[1:] == first.pose_vision.shape[1:]
                    )
                )
                and value.pose_schedule == first.pose_schedule
                and value.animate2_settings == first.animate2_settings
                and value.scail_replacement == first.scail_replacement
                and (
                    (value.scail_reference_latent is None and first.scail_reference_latent is None)
                    or (
                        value.scail_reference_latent is not None
                        and first.scail_reference_latent is not None
                        and value.scail_reference_latent.shape[1:]
                        == first.scail_reference_latent.shape[1:]
                    )
                )
                and (
                    (value.scail_reference_mask is None and first.scail_reference_mask is None)
                    or (
                        value.scail_reference_mask is not None
                        and first.scail_reference_mask is not None
                        and value.scail_reference_mask.shape[1:]
                        == first.scail_reference_mask.shape[1:]
                    )
                )
                and (
                    (value.scail_driving_mask is None and first.scail_driving_mask is None)
                    or (
                        value.scail_driving_mask is not None
                        and first.scail_driving_mask is not None
                        and value.scail_driving_mask.shape[1:] == first.scail_driving_mask.shape[1:]
                    )
                )
                and (
                    (value.face_pixel_values is None and first.face_pixel_values is None)
                    or (
                        value.face_pixel_values is not None
                        and first.face_pixel_values is not None
                        and value.face_pixel_values.shape[1:] == first.face_pixel_values.shape[1:]
                    )
                )
                for value in contexts[1:]
            )

        def validate_layout(_context: _Wan21ModelConditioning, declared: ModelTokenLayout) -> None:
            if declared != token_plan.layout:
                raise TokenLayoutError("Wan 2.1 model rows do not match the target video layout")

        evaluator_identity = (
            "dinkster.wan22.dancer-conditioning.v1"
            if model.config.model_variant == "wandancer"
            else "dinkster.wan21.humo-conditioning.v1"
            if model.config.model_variant == "humo"
            else "dinkster.wan22.s2v-conditioning.v1"
            if model.config.model_variant == "s2v"
            else "dinkster.wan21.bernini-conditioning.v1"
            if bernini_conditioning
            else "dinkster.wan21.scail-conditioning.v1"
            if model.config.model_variant in ("scail", "scail2")
            else "dinkster.wan21.animate2-conditioning.v1"
            if model.config.model_variant == "animate2"
            else "dinkster.wan22.animate-conditioning.v1"
            if model.config.model_variant == "animate"
            else "dinkster.wan21.vace-conditioning.v1"
            if model.config.vace_layers is not None
            else "dinkster.wan21.conditioning.v2"
            if model.config.model_type == "i2v"
            else "dinkster.wan22.conditioning.v1"
            if model.config.in_channels > model.config.out_channels
            else f"{self.family.id}.conditioning.v1"
        )
        if admitted_uni3c is not None:
            evaluator_identity += (
                f":uni3c={admitted_uni3c.model_digest}:{admitted_uni3c.render_digest}:"
                f"{admitted_uni3c.strength.hex()}:"
                f"{admitted_uni3c.window.start_percent.hex()}:"
                f"{admitted_uni3c.window.end_percent.hex()}"
            )
        if admitted_multitalk is not None:
            patch = admitted_multitalk.patch
            evaluator_identity += (
                f":multitalk={patch.model_digest}:{patch.audio_digest}:"
                f"{patch.target_masks_digest}:{patch.strength.hex()}:"
                f"{admitted_multitalk.motion_digest}:{int(admitted_multitalk.extend)}"
            )
        evaluation = ConditioningEvaluation(
            prepare_conditioning,
            lambda x, sigma, context: evaluate_batch(x, sigma, (context,))[0],
            batchable,
            evaluate_batch,
            evaluator_identity=lambda _role: evaluator_identity,
            standard_activation_memory_factor=self.family.memory_factor,
            layout=lambda _value: token_plan.layout,
            token_transforms=lambda _value: token_plan.transforms,
            validate_layout=validate_layout,
        )
        if context_windows is not None:
            evaluation = windowed_conditioning_evaluation(evaluation, context_windows, schedule)

        def report_step(event: StepEvent) -> None:
            _check_cancelled(cancelled)
            if on_step is not None:
                on_step(event)

        denoiser = guided_denoiser(
            evaluation,
            input=video,
            executor=None,
            plan=guidance_plan,
            execution=sampling_execution_context(schedule, seed, report_step, on_state),
        )
        if not schedule_plan.pre_offset:
            if admitted_multitalk is not None and admitted_multitalk.extend:
                motion = admitted_multitalk.motion_latent.to(video)
                output_streams = streams.replace(
                    "video",
                    torch.cat((motion, video[:, :, motion.shape[2] :]), dim=2),
                )
                return CustomSamplingResult(output_streams, None)
            return CustomSamplingResult(streams, None)

        noise_video = noise_video.to(device=load_device, dtype=video.dtype)
        if context_windows is not None and context_windows.freenoise:
            noise_video = apply_freenoise(
                noise_video,
                context_windows.dim,
                context_windows.length,
                context_windows.overlap,
                seed,
            )
        step_noise = brownian_step_noise(
            sampler, schedule_plan, video, seed=seed, device=load_device
        )

        def unpack_video_state(value: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
            return MultiStreamLatent.from_pairs((("video", value),))

        captured_denoised: list[MultiStreamLatent[torch.Tensor]] = []

        def state_progress(event: SamplingStateEvent[object]) -> None:
            _check_cancelled(cancelled)
            if capture_denoised and type(event.denoised) is MultiStreamLatent:
                captured_denoised[:] = [cast("MultiStreamLatent[torch.Tensor]", event.denoised)]
            if on_state is not None:
                on_state(event)
            _check_cancelled(cancelled)

        cache_settings = self._pose_cache_settings
        if cache_settings is not None:
            cache_device = (
                load_device
                if cache_settings.device == Wan21PoseBlockCacheDevice.GPU
                else torch.device("cpu")
            )
            pose_cache = PoseBranchCache(
                cache_device,
                cache_settings.storage.value,
                cache_settings.memory_limit_bytes,
            )
        try:
            output = run_sampler_engine(
                denoiser,
                request.build_solver(),
                latent=video,
                noise=noise_video,
                sigmas=schedule,
                initial_sigma=schedule_plan.initial_sigma,
                parameterization=parameterization,
                sigma_min=sigmas.sigma_min,
                sigma_max=sigmas.sigma_max,
                process_in=self._latent_process_in,
                process_out=self._latent_process_out,
                seed=seed,
                noise_kind=sampler.noise,
                noise_sampler=step_noise,
                percent_to_sigma=sigmas.percent_to_sigma,
                device=load_device,
                on_step=report_step,
                on_state=(state_progress if capture_denoised or on_state is not None else None),
                unpack_state=unpack_video_state,
                denoise_mask=prepared_denoise_mask,
                fixed_inpaint_latent=prepared_denoise_mask is not None
                and (model.config.model_type == "ti2v" or model.config.model_variant == "scail2"),
            )
        finally:
            if pose_cache is not None:
                pose_cache.free()
        _check_cancelled(cancelled)
        if admitted_multitalk is not None and admitted_multitalk.extend:
            motion = admitted_multitalk.motion_latent.to(output)
            output = torch.cat((motion, output[:, :, motion.shape[2] :]), dim=2)
            if captured_denoised:
                denoised = captured_denoised[-1].by_role("video")
                captured_denoised[-1] = streams.replace(
                    "video",
                    torch.cat((motion, denoised[:, :, motion.shape[2] :]), dim=2),
                )
        return CustomSamplingResult(
            streams.replace("video", output),
            captured_denoised[-1] if captured_denoised else None,
        )

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content).float()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent).float()


class Wan21DiffusionRuntime(MultiStreamSamplingRuntime):
    """Diffusion-only Wan 2.1 T2V sampling facade."""

    retained_offload_storage_components = frozenset()
    sampling_error = Wan21RuntimeError
    supports_sampling_shift = True
    supports_denoised_capture = True
    supports_batch_noise_indices = False

    def __init__(
        self,
        diffusion: Wan21Model | Wan21HumoModel | Wan22S2VModel | Wan22DancerModel,
        family: ModelFamily,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype | None = None,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        if not (
            (type(diffusion) is Wan21Model and diffusion.config is WAN21_T2V_14B)
            or (type(diffusion) is Wan21HumoModel and diffusion.config is WAN21_HUMO_17B)
            or (type(diffusion) is Wan22S2VModel and diffusion.config is WAN22_S2V_14B)
            or (type(diffusion) is Wan22DancerModel and diffusion.config is WAN22_WANDANCER_14B)
        ):
            raise ValueError(
                "Wan diffusion runtime requires exact Wan 2.1 T2V, HuMo, Wan 2.2 S2V, "
                "or WanDancer 14B"
            )
        if type(family.id) is not str or not family.id.strip():
            raise ValueError("Wan diffusion runtime family identity must be a nonempty string")
        if not runtime_identity:
            raise ValueError("Wan diffusion runtime identity must be nonempty")
        self._assembled = _Wan21DiffusionAssembly(diffusion, family, _compute_dtype=compute_dtype)
        self._runtime_identity = runtime_identity
        normalizer = _Wan21LatentNormalizer()
        self._normalizer = normalizer
        self._latent_process_in = normalizer.process_in
        self._latent_process_out = normalizer.process_out
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._pose_cache_settings = None

    @property
    def assembled(self) -> _Wan21DiffusionAssembly:
        return self._assembled

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return _wan_custom_space(self.assembled, sampling_shift)

    family = Wan21Runtime.family  # pyright: ignore[reportIncompatibleMethodOverride]
    runtime_identity = Wan21Runtime.runtime_identity
    supports_denoise_mask = Wan21Runtime.supports_denoise_mask
    conditioning_identity = Wan21Runtime.conditioning_identity  # pyright: ignore[reportIncompatibleMethodOverride]
    prepare_text_conditioning = Wan21Runtime.prepare_text_conditioning
    prepare_conditioning = Wan21Runtime.prepare_conditioning
    prepare_i2v_conditioning = Wan21Runtime.prepare_i2v_conditioning
    prepare_animate_conditioning = Wan21Runtime.prepare_animate_conditioning
    prepare_camera_conditioning = Wan21Runtime.prepare_camera_conditioning
    prepare_phantom_conditioning = Wan21Runtime.prepare_phantom_conditioning
    prepare_fun_conditioning = Wan21Runtime.prepare_fun_conditioning
    prepare_vace_conditioning = Wan21Runtime.prepare_vace_conditioning
    check_custom_sampling = Wan21Runtime.check_custom_sampling
    sample_custom = Wan21Runtime.sample_custom
    _sample_standard_custom = Wan21Runtime._sample_standard_custom  # pyright: ignore[reportPrivateUsage]

    @property
    def dense_custom_sampling_role(self) -> str | None:
        return "video" if isinstance(self.assembled.diffusion, Wan21CausalModel) else None

    custom_sampling_latent_kwargs = Wan21Runtime.custom_sampling_latent_kwargs
    prepare_custom_sampling_noise = Wan21Runtime.prepare_custom_sampling_noise
    adapt_multistream_latent = Wan21Runtime.adapt_multistream_latent
    _ksampler_kwargs = Wan21Runtime._ksampler_kwargs  # pyright: ignore[reportPrivateUsage]
    _ksampler_noise = Wan21Runtime._ksampler_noise  # pyright: ignore[reportPrivateUsage]
    _custom_sampling_process_in = Wan21Runtime._custom_sampling_process_in  # pyright: ignore[reportPrivateUsage]
    _custom_sampling_process_out = Wan21Runtime._custom_sampling_process_out  # pyright: ignore[reportPrivateUsage]


class Wan21CausalDiffusionRuntime(MultiStreamSamplingRuntime):
    """Diffusion-only Wan CausalAR custom-sampling facade."""

    retained_offload_storage_components = frozenset()
    sampling_error = Wan21RuntimeError
    supports_sampling_shift = True

    def __init__(
        self,
        diffusion: Wan21CausalModel,
        family: ModelFamily,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype | None = None,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
    ) -> None:
        if type(diffusion) is not Wan21CausalModel:
            raise ValueError("Wan causal diffusion runtime requires an exact CausalAR model")
        if type(family.id) is not str or not family.id.strip():
            raise ValueError(
                "Wan causal diffusion runtime family identity must be a nonempty string"
            )
        if not runtime_identity:
            raise ValueError("Wan causal diffusion runtime identity must be nonempty")
        self._assembled = _Wan21DiffusionAssembly(diffusion, family, _compute_dtype=compute_dtype)
        self._runtime_identity = runtime_identity
        normalizer = _Wan21LatentNormalizer()
        self._normalizer = normalizer
        self._latent_process_in = normalizer.process_in
        self._latent_process_out = normalizer.process_out
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = torch_scheduler_registry()
        self._pose_cache_settings = None

    @property
    def assembled(self) -> _Wan21DiffusionAssembly:
        return self._assembled

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return _wan_custom_space(self.assembled, sampling_shift)

    family = Wan21Runtime.family  # pyright: ignore[reportIncompatibleMethodOverride]
    runtime_identity = Wan21Runtime.runtime_identity
    supports_denoise_mask = Wan21Runtime.supports_denoise_mask
    conditioning_identity = Wan21Runtime.conditioning_identity  # pyright: ignore[reportIncompatibleMethodOverride]
    prepare_text_conditioning = Wan21Runtime.prepare_text_conditioning
    prepare_conditioning = Wan21Runtime.prepare_conditioning
    check_custom_sampling = Wan21Runtime.check_custom_sampling
    sample_custom = Wan21Runtime.sample_custom

    @property
    def dense_custom_sampling_role(self) -> str | None:
        return "video" if isinstance(self.assembled.diffusion, Wan21CausalModel) else None

    custom_sampling_latent_kwargs = Wan21Runtime.custom_sampling_latent_kwargs
    prepare_custom_sampling_noise = Wan21Runtime.prepare_custom_sampling_noise
    adapt_multistream_latent = Wan21Runtime.adapt_multistream_latent
    _ksampler_kwargs = Wan21Runtime._ksampler_kwargs  # pyright: ignore[reportPrivateUsage]
    _custom_sampling_process_in = Wan21Runtime._custom_sampling_process_in  # pyright: ignore[reportPrivateUsage]
    _custom_sampling_process_out = Wan21Runtime._custom_sampling_process_out  # pyright: ignore[reportPrivateUsage]


_MultiStreamRuntimeCheck: type[MultiStreamFamilyRuntime[torch.Tensor]] = Wan21Runtime
_DiffusionMultiStreamRuntimeCheck: type[MultiStreamFamilyRuntime[torch.Tensor]] = (
    Wan21DiffusionRuntime
)
_MultiStreamAdapterCheck: type[MultiStreamLatentAdapterRuntime[torch.Tensor]] = Wan21Runtime
_DiffusionMultiStreamAdapterCheck: type[MultiStreamLatentAdapterRuntime[torch.Tensor]] = (
    Wan21DiffusionRuntime
)


__all__ = [
    "Wan21CausalDiffusionRuntime",
    "Wan21DiffusionRuntime",
    "Wan21InfiniteTalkExecution",
    "Wan21PreparedConditioning",
    "Wan21Runtime",
    "Wan21RuntimeError",
    "compose_wan21_humo_conditioning",
    "compose_wan21_i2v_conditioning",
    "compose_wan22_dancer_conditioning",
    "compose_wan22_s2v_conditioning",
    "wan21_text_conditioning_to_carrier",
]
