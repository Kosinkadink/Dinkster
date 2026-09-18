"""Executable text-to-video runtime for the LTX-Video 2B profiles."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from dinkster_inference import (
    LTX_SIGMAS,
    LTXV,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    LTXV_2B_V095_CONFIG,
    LTXV_2B_V095_VAE_CONFIG,
    LTXV_CODEC,
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
    FluxFlowSigmas,
    GuidanceRole,
    InpaintConditioning,
    LTXVideoVAEConfig,
    ModelFamily,
    MultiStreamFamilyRuntime,
    MultiStreamLatent,
    NoiseKind,
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
    SamplingStateCallback,
    SamplingStateEvent,
    SchedulerDescriptor,
    StepCallback,
    StepEvent,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    calculate_denoised,
    calculate_input,
    encode_conditioning_carrier,
    make_conditioning_carrier,
    sampling_environment_cancellation,
    sampling_execution_context,
)

from ._conditioning_layout import DeclaredConditioning, declared_token_count
from .codecs import CodecPlugin
from .context_windows import apply_freenoise, windowed_conditioning_evaluation
from .denoise import run_sampler_engine, to_batch
from .guidance import ConditioningEvaluation
from .latent_streams import normalize_latent_mask
from .ltx_media import (
    LTXMediaError,
    LTXVGuideConditioning,
    ltxv_guides_equal,
    materialize_ltxv_guides,
)
from .ltx_model import LTXVModel
from .ltx_video_vae import LTXVideoVAE, ltxv_vae_max_chunk_bytes
from .memory import get_total_memory
from .operations import bound_compute_device
from .payloads import TensorPayloadError, payload_binding_to_tensor, tensor_to_payload_binding
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    brownian_step_noise,
    build_custom_sampling_schedule,
    compile_guidance_plan,
    guided_denoiser,
    resolve_custom_sampling_request,
)
from .sampling_runtime import MultiStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry

if TYPE_CHECKING:
    from .checkpoint_runtime import ComponentAssembly


class LTXVRuntimeError(ValueError):
    """The requested operation is outside the loaded LTX-Video profile."""


def _validate_text(value: object, *, text_dim: int) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("LTX-Video TEXT must be an exact strided floating torch.Tensor")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] != text_dim:
        raise LTXVRuntimeError(f"LTX-Video TEXT must be nonempty [B,tokens,{text_dim}]")
    return value


def _validate_frame_rate(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise LTXVRuntimeError("LTX-Video frame rate must be a positive finite float")
    return value


@dataclass(frozen=True, slots=True)
class LTXVPreparedConditioning:
    """Materialized LTX-Video conditioning carried opaquely by the
    multistream seam.

    ``attention_tokens`` is the declared prompt length through EOS:
    the model attends to rows ``[0, attention_tokens)`` and masks the
    padded tail, whose embeddings stay nonzero (classic T5 policy,
    lt.LTXVT5XXL @ b78cec87)."""

    text: torch.Tensor
    attention_tokens: int
    frame_rate: float = 25.0
    guides: tuple[LTXVGuideConditioning, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.text) is not torch.Tensor
            or not self.text.is_floating_point()
            or self.text.layout != torch.strided
        ):
            raise TypeError("LTX-Video TEXT must be an exact strided floating torch.Tensor")
        if self.text.ndim != 3 or any(size <= 0 for size in self.text.shape):
            raise LTXVRuntimeError("LTX-Video TEXT must be a nonempty rank-3 tensor")
        if type(self.attention_tokens) is not int or not (
            1 <= self.attention_tokens <= self.text.shape[1]
        ):
            raise LTXVRuntimeError(
                "LTX-Video attention token count must be an int in [1, text rows]"
            )
        _validate_frame_rate(self.frame_rate)
        if type(self.guides) is not tuple or any(
            type(guide) is not LTXVGuideConditioning for guide in self.guides
        ):
            raise TypeError("LTX-Video guides must be exact LTXVGuideConditioning values")


def _conditioning_text(value: object, *, text_dim: int) -> torch.Tensor:
    if type(value) not in (Conditioning, DeclaredConditioning):
        raise TypeError("LTX-Video guidance lanes require exact Conditioning values")
    conditioning = cast("Conditioning[torch.Tensor]", value)
    if conditioning.pooled is not None:
        raise LTXVRuntimeError("LTX-Video text conditioning does not accept pooled embeddings")
    return _validate_text(conditioning.embeddings, text_dim=text_dim)


def _text_carrier(value: object, *, text_dim: int, family_id: str) -> ConditioningCarrier:
    text = _conditioning_text(value, text_dim=text_dim)
    conditioning = cast("Conditioning[torch.Tensor]", value)
    declared = declared_token_count(conditioning)
    token_count = int(text.shape[1]) if declared is None else declared
    binding = tensor_to_payload_binding("ltxv-text", text, space="conditioning-text")
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
            text_streams=("t5xxl",),
            segments=(TokenSegmentDescriptor("t5xxl", "t5xxl", 0, token_count),),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (binding,))


def ltxv_text_conditioning_to_carrier(
    conditioning: Conditioning[torch.Tensor],
) -> ConditioningCarrier:
    """Convert classic T5 output into canonical LTX-Video conditioning."""
    return _text_carrier(
        conditioning,
        text_dim=LTXV_2B_V09_CONFIG.caption_channels,
        family_id=LTXV.id,
    )


def _materialize_conditioning(
    carrier: object, *, text_dim: int, family_id: str
) -> tuple[torch.Tensor, int]:
    """One carrier -> (text, attention token count).

    The token layout's single segment length may be SHORTER than the
    payload rows: the encoder pads to the profile minimum without
    zeroing, and the segment declares the real prompt length the model
    should attend to."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("LTX-Video conditioning must be an exact ConditioningCarrier")
    typed = carrier
    try:
        encode_conditioning_carrier(typed)
    except (TypeError, ValueError) as error:
        raise LTXVRuntimeError(f"LTX-Video conditioning carrier is invalid: {error}") from None
    records = typed.conditioning.records
    if len(records) != 1:
        raise LTXVRuntimeError("LTX-Video requires exactly one conditioning record")
    record = records[0]
    channels = dict(record.channels)
    unsupported = tuple(
        channel.value for channel in channels if channel is not ConditioningChannel.TEXT
    )
    if unsupported:
        raise LTXVRuntimeError(
            "LTX-Video does not consume conditioning channels: " + ", ".join(unsupported)
        )
    descriptor = channels.get(ConditioningChannel.TEXT)
    if descriptor is None:
        raise LTXVRuntimeError("LTX-Video conditioning is missing the TEXT channel")
    if len(descriptor.shape) != 3 or descriptor.shape[2] != text_dim:
        raise LTXVRuntimeError(f"LTX-Video TEXT descriptor must have shape [B,tokens,{text_dim}]")
    if record.area is not None:
        raise LTXVRuntimeError("LTX-Video does not consume conditioning areas")
    if record.mask is not None:
        raise LTXVRuntimeError("LTX-Video does not consume conditioning masks")
    if record.schedule != PercentRange(0.0, 1.0):
        raise LTXVRuntimeError("LTX-Video does not consume conditioning schedules")
    if record.scale_vector is not None:
        raise LTXVRuntimeError("LTX-Video does not consume conditioning scale vectors")
    if record.extension_metadata:
        raise LTXVRuntimeError("LTX-Video does not consume conditioning extension metadata")
    layout = record.token_layout
    if layout is None:
        raise LTXVRuntimeError("LTX-Video conditioning requires a token layout")
    try:
        layout.require_supported(family_id, (1,))
    except ValueError as error:
        raise LTXVRuntimeError(f"LTX-Video token layout is unsupported: {error}") from None
    if (
        layout.text_streams != ("t5xxl",)
        or len(layout.segments) != 1
        or layout.segments[0].name != "t5xxl"
        or layout.segments[0].stream != "t5xxl"
        or layout.segments[0].start_token != 0
    ):
        raise LTXVRuntimeError("LTX-Video conditioning requires the exact T5-XXL token layout")
    token_count = layout.segments[0].token_count
    if token_count is None or not 1 <= token_count <= descriptor.shape[1]:
        raise LTXVRuntimeError(
            "LTX-Video token layout segment must cover 1 through the TEXT payload rows"
        )
    bindings = {binding.reference_id: binding for binding in typed.bindings}
    if len(bindings) != len(typed.bindings):
        raise LTXVRuntimeError("LTX-Video conditioning carrier has duplicate payload bindings")
    binding = bindings.get(descriptor.reference.id)
    if binding is None:
        raise LTXVRuntimeError("LTX-Video TEXT channel has no payload binding")
    if (binding.shape, binding.dtype, binding.space) != (
        descriptor.shape,
        descriptor.dtype,
        descriptor.space,
    ):
        raise LTXVRuntimeError("LTX-Video TEXT descriptor does not match its payload binding")
    if binding.space != "conditioning-text":
        raise LTXVRuntimeError("LTX-Video TEXT payload has an unsupported tensor space")
    try:
        text = payload_binding_to_tensor(binding)
    except TensorPayloadError as error:
        raise LTXVRuntimeError(f"LTX-Video TEXT payload could not be decoded: {error}") from None
    return _validate_text(text, text_dim=text_dim), int(token_count)


def _video_streams(
    value: object, *, channels: int
) -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    if type(value) is not MultiStreamLatent:
        raise TypeError("LTX-Video latent must be an exact MultiStreamLatent")
    streams = cast("MultiStreamLatent[torch.Tensor]", value)
    if streams.roles != ("video",):
        raise LTXVRuntimeError("LTX-Video requires the exact latent stream role 'video'")
    latent = streams.by_role("video")
    if (
        type(latent) is not torch.Tensor
        or not latent.is_floating_point()
        or latent.layout != torch.strided
    ):
        raise TypeError("LTX-Video latent must be an exact strided floating torch.Tensor")
    if latent.ndim != 5 or latent.shape[1] != channels or any(size <= 0 for size in latent.shape):
        raise LTXVRuntimeError(f"LTX-Video latent must be nonempty [B,{channels},T,H,W]")
    return streams, latent


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
        raise LTXVRuntimeError("sampling_shift must be a positive finite float")
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
class _LTXVCodec:
    """Adapt the LTX-Video VAE's chunked entry points to the codec
    encoder/decoder protocols (the VAE streams internal chunks; the
    plugin never tiles it)."""

    vae: LTXVideoVAE

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        max_chunk_bytes = ltxv_vae_max_chunk_bytes(get_total_memory(content.device))
        return self.vae.encode(content, max_chunk_bytes=max_chunk_bytes)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        max_chunk_bytes = ltxv_vae_max_chunk_bytes(get_total_memory(latent.device))
        return self.vae.decode(latent, max_chunk_bytes=max_chunk_bytes)


def _ltxv_codec_plugin(vae: LTXVideoVAE, compute_dtype: torch.dtype | None) -> CodecPlugin:
    adapter = _LTXVCodec(vae)
    return CodecPlugin(
        LTXV_CODEC,
        adapter,
        adapter,
        content_crop=lambda content: _crop_spatial_to_multiple(
            content, LTXV_CODEC.latent.spatial_downscale
        ),
        content_in=lambda value: value * 2.0 - 1.0,
        content_out=lambda value: value.float().add_(1.0).div_(2.0).clamp_(0.0, 1.0),
        compute_dtype=compute_dtype,
    )


class LTXVVideoCodecRuntime:
    """Video codec facade over one independently loaded LTX-Video VAE."""

    def __init__(self, vae: LTXVideoVAE, *, compute_dtype: torch.dtype | None = None) -> None:
        self.vae = vae
        self.codec = _ltxv_codec_plugin(vae, compute_dtype)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(content).float()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent).float()


class _LTXVCheckpointCodec(LTXVVideoCodecRuntime):
    descriptor = LTXV_CODEC
    encode = LTXVVideoCodecRuntime.encode_content
    decode = LTXVVideoCodecRuntime.decode_latent


def checkpoint_codec(assembled: ComponentAssembly) -> _LTXVCheckpointCodec | None:
    vae = assembled.components.get("vae")
    return (
        None
        if vae is None
        else _LTXVCheckpointCodec(
            cast(LTXVideoVAE, vae), compute_dtype=assembled.compute_dtype("vae")
        )
    )


@dataclass(frozen=True, slots=True)
class _LTXVModelConditioning:
    text: torch.Tensor
    mask: torch.Tensor
    frame_rate: float
    guides: tuple[LTXVGuideConditioning, ...]


def _batch_guides(
    guides: tuple[LTXVGuideConditioning, ...], batch: int, lanes: int
) -> tuple[LTXVGuideConditioning, ...]:
    if lanes <= 0:
        raise ValueError("LTX-Video guide batching requires at least one lane")
    result: list[LTXVGuideConditioning] = []
    for guide in guides:
        keyframes = to_batch(guide.keyframe_indices, batch)
        if lanes > 1:
            keyframes = torch.cat((keyframes,) * lanes)
        result.append(
            LTXVGuideConditioning(
                keyframes,
                guide.latent_shape,
                guide.strength,
                guide.attention_mask,
            )
        )
    return tuple(result)


class LTXVDiffusionRuntime(MultiStreamSamplingRuntime):
    """Diffusion-only classic LTX-Video custom-sampling runtime."""

    retained_offload_storage_components = frozenset()
    sampling_error = LTXVRuntimeError
    supports_denoised_capture = True
    supports_batch_noise_indices = False
    supports_sampling_shift = True

    def __init__(
        self,
        diffusion: LTXVModel,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        vae_configs = {
            LTXV_2B_V09_CONFIG: LTXV_2B_V09_VAE_CONFIG,
            LTXV_2B_V095_CONFIG: LTXV_2B_V095_VAE_CONFIG,
        }
        try:
            vae_config = vae_configs[diffusion.config]
        except (KeyError, TypeError):
            raise ValueError(
                "LTX-Video diffusion runtime requires an exact supported 2B model"
            ) from None
        if not runtime_identity:
            raise ValueError("LTX-Video runtime identity must be nonempty")
        self._assembled = _LTXVDiffusionAssembly(
            LTXV,
            diffusion,
            vae_config,
            compute_dtype,
        )
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry() if sampler_registry is None else sampler_registry
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )

    @property
    def assembled(self) -> _LTXVDiffusionAssembly:
        return self._assembled

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def video_vae_config(self) -> LTXVideoVAEConfig:
        return self.assembled.vae_config

    @property
    def supports_context_windows(self) -> bool:
        return True

    @property
    def conditioning_identity(self) -> str:
        """Stable compatibility identity for materialized LTX-Video conditioning."""
        config = self.assembled.diffusion.config
        fields = (
            self.family.id,
            config.in_channels,
            config.cross_attention_dim,
            config.attention_head_dim,
            config.num_attention_heads,
            config.caption_channels,
            config.num_layers,
            config.causal_temporal_positioning,
        )
        return "dinkster.ltxv.conditioning:v1:" + ":".join(str(field) for field in fields)

    def prepare_conditioning(
        self, carrier: ConditioningCarrier, *, frame_rate: float = 25.0
    ) -> LTXVPreparedConditioning:
        try:
            carrier, guides = materialize_ltxv_guides(carrier)
        except LTXMediaError as error:
            raise LTXVRuntimeError(str(error)) from None
        text, tokens = _materialize_conditioning(
            carrier,
            text_dim=self.assembled.diffusion.config.caption_channels,
            family_id=self.family.id,
        )
        return LTXVPreparedConditioning(
            text,
            tokens,
            _validate_frame_rate(frame_rate),
            guides,
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FluxFlowSigmas:
        return _sigma_space(sampling_shift)

    def sample_custom(
        self,
        latent: CustomSamplingLatentValue,
        *,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue = None,
        request: CustomSamplingRequest[torch.Tensor],
        seed: int = 0,
        guidance: float | None = None,
        denoise_mask: CustomSamplingLatentValue | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        observer: ExecutionObserverAttachment | None = None,
        parent_span_id: int | None = None,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        sampling_shift: float | None = None,
        capture_denoised: bool = True,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        del observer, parent_span_id
        model = self.assembled.diffusion
        text_dim = model.config.caption_channels
        streams, video = _video_streams(latent, channels=model.config.in_channels)
        if type(noise) is not MultiStreamLatent:
            raise LTXVRuntimeError("LTX-Video custom sampling requires MultiStreamLatent noise")
        noise_streams = noise
        if noise_streams.roles != streams.roles:
            raise LTXVRuntimeError("LTX-Video noise streams must match the latent stream roles")
        noise_video = noise_streams.by_role("video")
        if (
            type(noise_video) is not torch.Tensor
            or not noise_video.is_floating_point()
            or noise_video.layout != torch.strided
        ):
            raise TypeError("LTX-Video noise stream must be an exact strided floating torch.Tensor")
        if tuple(noise_video.shape) != tuple(video.shape):
            raise LTXVRuntimeError("LTX-Video noise stream must match the video latent shape")
        if isinstance(cfg, PerpNegSamplingGuidance):
            raise LTXVRuntimeError(
                "LTX-Video custom sampling does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        if type(cond) is not PreparedMultiStreamConditioning:
            raise LTXVRuntimeError(
                "LTX-Video custom sampling requires prepared multi-stream conditioning"
            )
        if cond.runtime_identity != self.conditioning_identity:
            raise LTXVRuntimeError(
                "LTX-Video conditioning was prepared by a different conditioner component"
            )
        conditioning = cond.payload
        if type(conditioning) is not LTXVPreparedConditioning:
            raise TypeError("conditioning must be exact LTXVPreparedConditioning")
        _validate_text(conditioning.text, text_dim=text_dim)

        def unwrap_lane(value: object) -> LTXVPreparedConditioning | None:
            if value is None:
                return None
            if type(value) is not PreparedMultiStreamConditioning:
                raise LTXVRuntimeError(
                    "LTX-Video custom sampling guidance requires prepared multi-stream conditioning"
                )
            if value.runtime_identity != cond.runtime_identity:
                raise LTXVRuntimeError(
                    "LTX-Video guidance lanes were prepared by different conditioner components"
                )
            payload = value.payload
            if type(payload) is not LTXVPreparedConditioning:
                raise TypeError("LTX-Video guidance lanes require exact LTXVPreparedConditioning")
            _validate_text(payload.text, text_dim=text_dim)
            return payload

        guidance_cfg: SamplingGuidance[object] | DualSamplingGuidance[object] | None
        uncond_lane: LTXVPreparedConditioning | None = None
        middle_lane: LTXVPreparedConditioning | None = None
        if cfg is None:
            guidance_cfg = None
        elif isinstance(cfg, DualSamplingGuidance):
            dual = cast("DualSamplingGuidance[object]", cfg)
            uncond_lane = unwrap_lane(dual.uncond)
            middle_lane = unwrap_lane(dual.middle)
            guidance_cfg = replace(dual, uncond=uncond_lane, middle=middle_lane)
        else:
            base = cast("SamplingGuidance[object]", cfg)
            uncond_lane = unwrap_lane(base.uncond)
            guidance_cfg = base if base.uncond is None else replace(base, uncond=uncond_lane)
        # The model takes one scalar frame rate; guidance lanes cannot
        # disagree on it.
        frame_rate = conditioning.frame_rate
        for lane in (uncond_lane, middle_lane):
            if lane is not None and lane.frame_rate != frame_rate:
                raise LTXVRuntimeError("LTX-Video guidance lanes must share one frame rate")
            if lane is not None and not ltxv_guides_equal(lane.guides, conditioning.guides):
                raise LTXVRuntimeError("LTX-Video guidance lanes must share the same guides")
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        if context_windows is not None and context_windows.cond_retain_indices:
            # LTX-Video conditioning is text-only, so there are no
            # frame-aligned conditioning entries for retain indices to pin.
            raise LTXVRuntimeError("LTX-Video context windows do not accept cond_retain_indices")
        if context_windows is not None and conditioning.guides:
            raise LTXVRuntimeError("LTX-Video guide conditioning does not support context windows")
        if context_windows is not None and denoise_mask is not None:
            raise LTXVRuntimeError("LTX-Video denoise masks do not support context windows")
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=LTXVRuntimeError
        )
        if cancelled is None:
            cancelled = sampling_environment_cancellation()
        _check_cancelled(cancelled)
        sigmas = _sigma_space(sampling_shift)
        schedule_plan = build_custom_sampling_schedule(request.sigmas, sigmas, sampler, flow=True)
        schedule = schedule_plan.sigmas
        guidance_plan = compile_guidance_plan(conditioning, guidance_cfg, sampler, None)
        weight = model.patchify_proj.weight
        model_device = bound_compute_device(model.patchify_proj) or weight.device
        load_device = model_device if device is None else torch.device(device)
        selected_dtype = (
            self.assembled.compute_dtype("diffusion") or torch.bfloat16
            if compute_dtype is None
            else compute_dtype
        )
        video = video.to(load_device)
        model_mask: torch.Tensor | None = None
        sampler_mask: torch.Tensor | None = None
        if denoise_mask is not None:
            try:
                normalized = normalize_latent_mask(
                    cast("torch.Tensor | MultiStreamLatent[torch.Tensor]", denoise_mask),
                    streams,
                ).by_role("video")
            except (TypeError, ValueError) as error:
                raise LTXVRuntimeError(f"LTX-Video denoise mask is invalid: {error}") from None
            if (
                not bool(torch.isfinite(normalized).all())
                or float(normalized.amin()) < 0.0
                or float(normalized.amax()) > 1.0
            ):
                raise LTXVRuntimeError("LTX-Video denoise mask values must be finite within [0, 1]")
            sampler_mask = normalized.to(device=load_device)
            model_mask = sampler_mask[:, :1].to(dtype=selected_dtype)
        if conditioning.guides and model_mask is None:
            raise LTXVRuntimeError("LTX-Video guide conditioning requires a denoise mask")

        def prepare_conditioning(value: object, _role: GuidanceRole) -> _LTXVModelConditioning:
            _check_cancelled(cancelled)
            if type(value) is not LTXVPreparedConditioning:
                raise TypeError("LTX-Video guidance lanes require exact prepared conditioning")
            text = _validate_text(value.text, text_dim=text_dim).to(
                device=load_device, dtype=selected_dtype
            )
            # Integer mask: the model turns it additive via
            # (mask - 1) * finfo.max, so 1 = attend, 0 = masked.
            mask = torch.zeros((text.shape[0], text.shape[1]), device=load_device, dtype=torch.long)
            mask[:, : value.attention_tokens] = 1
            guides = tuple(
                LTXVGuideConditioning(
                    guide.keyframe_indices.to(device=load_device),
                    guide.latent_shape,
                    guide.strength,
                    None
                    if guide.attention_mask is None
                    else guide.attention_mask.to(device=load_device, dtype=selected_dtype),
                )
                for guide in value.guides
            )
            return _LTXVModelConditioning(text, mask, value.frame_rate, guides)

        def evaluate_batch(
            x: torch.Tensor,
            sigma: float,
            contexts: tuple[_LTXVModelConditioning, ...],
        ) -> tuple[torch.Tensor, ...]:
            _check_cancelled(cancelled)
            if not contexts:
                raise LTXVRuntimeError("LTX-Video model evaluation requires conditioning")
            first_context = next(iter(contexts))
            batch = x.shape[0]
            lane_frame_rate = first_context.frame_rate
            base_input = calculate_input(Parameterization.FLOW, sigma, x).to(selected_dtype)
            model_input = base_input
            if len(contexts) > 1:
                model_input = torch.cat((base_input,) * len(contexts))
            context = torch.cat(tuple(to_batch(value.text, batch) for value in contexts))
            mask = torch.cat(tuple(to_batch(value.mask, batch) for value in contexts))
            timestep = sigmas.timestep(sigma)
            model_arguments: dict[str, object] = {}
            if model_mask is None:
                timesteps = torch.full(
                    (model_input.shape[0],),
                    timestep,
                    device=load_device,
                    dtype=torch.float32,
                )
            else:
                lane_mask = to_batch(model_mask, batch)
                timesteps = (lane_mask.float() * timestep).flatten(1)
                if len(contexts) > 1:
                    timesteps = torch.cat((timesteps,) * len(contexts))
                model_arguments["denoise_mask"] = (
                    lane_mask if len(contexts) == 1 else torch.cat((lane_mask,) * len(contexts))
                )
            if first_context.guides:
                model_arguments["guides"] = _batch_guides(
                    first_context.guides, batch, len(contexts)
                )
            velocity = model(
                model_input,
                timesteps,
                context,
                attention_mask=mask,
                frame_rate=lane_frame_rate,
                **model_arguments,
            ).float()
            outputs = velocity.chunk(len(contexts))
            return tuple(
                calculate_denoised(Parameterization.FLOW, sigma, output, x) for output in outputs
            )

        def batchable(contexts: tuple[_LTXVModelConditioning, ...]) -> bool:
            if not contexts:
                return False
            first = contexts[0]
            return all(
                value.text.shape[1:] == first.text.shape[1:]
                and value.frame_rate == first.frame_rate
                and ltxv_guides_equal(value.guides, first.guides)
                for value in contexts[1:]
            )

        evaluation = ConditioningEvaluation(
            prepare_conditioning,
            lambda x, sigma, context: evaluate_batch(x, sigma, (context,))[0],
            batchable,
            evaluate_batch,
            evaluator_identity=lambda _role: "dinkster.ltxv.conditioning.v1",
            standard_activation_memory_factor=self.family.memory_factor,
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
            return CustomSamplingResult(streams, None)

        # Rounding a float32 draw to the sampling dtype here is
        # bit-identical to drawing at the stream dtype (torch.randn
        # always draws float32 first), so float32 noise from the
        # composed KSampler path reproduces the reference draw exactly.
        noise_video = noise_video.to(device=load_device, dtype=video.dtype)
        if context_windows is not None and context_windows.freenoise:
            # A pure frame permutation, so applying it after the dtype
            # round is bit-identical to shuffling the float32 draw.
            noise_video = apply_freenoise(
                noise_video,
                context_windows.dim,
                context_windows.length,
                context_windows.overlap,
                seed,
            )
        if sampler.noise in (NoiseKind.BROWNIAN, NoiseKind.BROWNIAN_GPU) and not any(
            value > 0.0 for value in schedule_plan.pre_offset
        ):
            raise LTXVRuntimeError("LTX-Video brownian sampler needs positive sigmas")
        step_noise = brownian_step_noise(
            sampler, schedule_plan, video, seed=seed, device=load_device
        )
        captured_denoised: list[MultiStreamLatent[torch.Tensor]] = []

        def state_progress(event: SamplingStateEvent[object]) -> None:
            _check_cancelled(cancelled)
            if capture_denoised:
                denoised = event.denoised
                if type(denoised) is MultiStreamLatent:
                    captured_denoised[:] = [cast("MultiStreamLatent[torch.Tensor]", denoised)]
            if on_state is not None:
                on_state(event)
            _check_cancelled(cancelled)

        def unpack_video_state(value: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
            return MultiStreamLatent.from_pairs((("video", value),))

        output = run_sampler_engine(
            denoiser,
            request.build_solver(),
            latent=video,
            noise=noise_video,
            sigmas=schedule,
            initial_sigma=schedule_plan.initial_sigma,
            parameterization=Parameterization.FLOW,
            sigma_min=sigmas.sigma_min,
            sigma_max=sigmas.sigma_max,
            # The LTX-Video VAE normalizes latents itself (encode returns
            # normalized means; decode denormalizes), so the sampler-side
            # latent transform is the identity.
            process_in=lambda value: value,
            process_out=lambda value: value,
            seed=seed,
            noise_kind=sampler.noise,
            noise_sampler=step_noise,
            percent_to_sigma=sigmas.percent_to_sigma,
            device=load_device,
            on_step=report_step,
            on_state=state_progress if capture_denoised or on_state is not None else None,
            unpack_state=unpack_video_state,
            denoise_mask=sampler_mask,
            fixed_inpaint_latent=True,
        )
        _check_cancelled(cancelled)
        output_latent = streams.replace("video", output)
        denoised_output: MultiStreamLatent[torch.Tensor] | None = None
        if captured_denoised:
            denoised_output = streams.replace("video", captured_denoised[-1].by_role("video"))
        return CustomSamplingResult(output_latent, denoised_output)


@dataclass(frozen=True, slots=True)
class _LTXVDiffusionAssembly:
    family: ModelFamily
    diffusion: LTXVModel
    vae_config: LTXVideoVAEConfig
    diffusion_dtype: torch.dtype
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.diffusion_dtype if component == "diffusion" else None


_DiffusionMultiStreamRuntimeCheck: type[MultiStreamFamilyRuntime[torch.Tensor]] = (
    LTXVDiffusionRuntime
)
