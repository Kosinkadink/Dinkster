"""Executable text-to-video runtime for the LTX-Video 2B profiles."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

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
    CustomSamplingResult,
    DualSamplingGuidance,
    FluxFlowSigmas,
    GuidanceRole,
    LTXVideoVAEConfig,
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
    make_conditioning_carrier,
)

from ._conditioning_layout import DeclaredConditioning, declared_token_count
from .codecs import CodecPlugin
from .context_windows import apply_freenoise, windowed_conditioning_evaluation
from .denoise import to_batch
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


@dataclass(frozen=True)
class _LTXVLatentContext:
    original: MultiStreamLatent[torch.Tensor]
    model_mask: torch.Tensor | None


@dataclass(frozen=True)
class _LTXVLatentAdapter:
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
        owner = cast("LTXVDiffusionRuntime", runtime)
        if context.options:
            names = ", ".join(sorted(context.options))
            raise LTXVRuntimeError(f"LTX-Video sampling does not accept adapter options: {names}")
        model = owner.assembled.diffusion
        streams, video = _video_streams(latent, channels=model.config.in_channels)
        if type(noise) is not MultiStreamLatent:
            raise LTXVRuntimeError("LTX-Video custom sampling requires MultiStreamLatent noise")
        if noise.roles != streams.roles:
            raise LTXVRuntimeError("LTX-Video noise streams must match the latent stream roles")
        noise_video = noise.by_role("video")
        if (
            type(noise_video) is not torch.Tensor
            or not noise_video.is_floating_point()
            or noise_video.layout != torch.strided
        ):
            raise TypeError("LTX-Video noise stream must be an exact strided floating torch.Tensor")
        if tuple(noise_video.shape) != tuple(video.shape):
            raise LTXVRuntimeError("LTX-Video noise stream must match the video latent shape")
        if context.context_windows is not None and context.context_windows.freenoise:
            noise_video = apply_freenoise(
                noise_video,
                context.context_windows.dim,
                context.context_windows.length,
                context.context_windows.overlap,
                context.seed,
            )
        if isinstance(cfg, PerpNegSamplingGuidance):
            raise LTXVRuntimeError(
                "LTX-Video custom sampling does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        if type(cond) is not PreparedMultiStreamConditioning:
            raise LTXVRuntimeError(
                "LTX-Video custom sampling requires prepared multi-stream conditioning"
            )
        if cond.runtime_identity != owner.conditioning_identity:
            raise LTXVRuntimeError(
                "LTX-Video conditioning was prepared by a different conditioner component"
            )
        conditioning = cond.payload
        if type(conditioning) is not LTXVPreparedConditioning:
            raise TypeError("conditioning must be exact LTXVPreparedConditioning")
        text_dim = model.config.caption_channels
        _validate_text(conditioning.text, text_dim=text_dim)

        def unwrap(value: object) -> LTXVPreparedConditioning | None:
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
                raise LTXVRuntimeError("LTX-Video guidance lanes must share one frame rate")
            if lane is not None and not ltxv_guides_equal(lane.guides, conditioning.guides):
                raise LTXVRuntimeError("LTX-Video guidance lanes must share the same guides")
        if context.context_windows is not None and context.context_windows.cond_retain_indices:
            raise LTXVRuntimeError("LTX-Video context windows do not accept cond_retain_indices")
        if context.context_windows is not None and conditioning.guides:
            raise LTXVRuntimeError("LTX-Video guide conditioning does not support context windows")
        if context.context_windows is not None and denoise_mask is not None:
            raise LTXVRuntimeError("LTX-Video denoise masks do not support context windows")
        sampler_mask = None
        model_mask = None
        if denoise_mask is not None:
            try:
                sampler_mask = normalize_latent_mask(
                    cast("torch.Tensor | MultiStreamLatent[torch.Tensor]", denoise_mask), streams
                ).by_role("video")
            except (TypeError, ValueError) as exception:
                raise LTXVRuntimeError(f"LTX-Video denoise mask is invalid: {exception}") from None
            if (
                not bool(torch.isfinite(sampler_mask).all())
                or float(sampler_mask.amin()) < 0.0
                or float(sampler_mask.amax()) > 1.0
            ):
                raise LTXVRuntimeError("LTX-Video denoise mask values must be finite within [0, 1]")
            model_mask = sampler_mask[:, :1]
        if conditioning.guides and model_mask is None:
            raise LTXVRuntimeError("LTX-Video guide conditioning requires a denoise mask")
        return SamplingExecutionInputs(
            video,
            noise_video,
            conditioning,
            guidance_cfg,
            sampler_mask,
            _LTXVLatentContext(streams, model_mask),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        context = cast("_LTXVLatentContext", inputs.latent_context)
        result = context.original.replace("video", output)
        if denoised is None:
            return CustomSamplingResult(result, None)
        if type(denoised) is not MultiStreamLatent:
            raise TypeError("LTX-Video denoised state must contain a MultiStreamLatent")
        return CustomSamplingResult(
            result, context.original.replace("video", denoised.by_role("video"))
        )


class _LTXVSamplingDenoiser:
    evaluator_identity = "dinkster.ltxv.conditioning.v1"

    def __init__(
        self,
        owner: LTXVDiffusionRuntime,
        model_mask: torch.Tensor | None,
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        space: FluxFlowSigmas,
        cancelled: Callable[[], bool],
    ) -> None:
        self.owner = owner
        self.model_mask = model_mask
        self.device = device
        self.compute_dtype = compute_dtype
        self.space = space
        self.cancelled = cancelled

    def prepare_conditioning(self, value: object, _role: GuidanceRole) -> _LTXVModelConditioning:
        _check_cancelled(self.cancelled)
        if type(value) is not LTXVPreparedConditioning:
            raise TypeError("LTX-Video guidance lanes require exact prepared conditioning")
        text = _validate_text(
            value.text, text_dim=self.owner.assembled.diffusion.config.caption_channels
        ).to(device=self.device, dtype=self.compute_dtype)
        mask = torch.zeros((text.shape[0], text.shape[1]), device=self.device, dtype=torch.long)
        mask[:, : value.attention_tokens] = 1
        guides = tuple(
            LTXVGuideConditioning(
                guide.keyframe_indices.to(device=self.device),
                guide.latent_shape,
                guide.strength,
                None
                if guide.attention_mask is None
                else guide.attention_mask.to(device=self.device, dtype=self.compute_dtype),
            )
            for guide in value.guides
        )
        return _LTXVModelConditioning(text, mask, value.frame_rate, guides)

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: _LTXVModelConditioning
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    def batchable(self, values: tuple[_LTXVModelConditioning, ...]) -> bool:
        if not values:
            return False
        first = values[0]
        return all(
            value.text.shape[1:] == first.text.shape[1:]
            and value.frame_rate == first.frame_rate
            and ltxv_guides_equal(value.guides, first.guides)
            for value in values[1:]
        )

    def evaluate_conditioning_batch(
        self, x: torch.Tensor, sigma: float, values: tuple[_LTXVModelConditioning, ...]
    ) -> tuple[torch.Tensor, ...]:
        _check_cancelled(self.cancelled)
        if not values or not self.batchable(values):
            raise LTXVRuntimeError("LTX-Video model evaluation requires conditioning")
        first = next(iter(values))
        batch = x.shape[0]
        base_input = calculate_input(Parameterization.FLOW, sigma, x).to(self.compute_dtype)
        model_input = base_input if len(values) == 1 else torch.cat((base_input,) * len(values))
        text = torch.cat(tuple(to_batch(value.text, batch) for value in values))
        mask = torch.cat(tuple(to_batch(value.mask, batch) for value in values))
        timestep = self.space.timestep(sigma)
        arguments: dict[str, object] = {}
        if self.model_mask is None:
            timesteps = torch.full(
                (model_input.shape[0],), timestep, device=self.device, dtype=torch.float32
            )
        else:
            lane_mask = to_batch(self.model_mask.to(self.device, self.compute_dtype), batch)
            timesteps = (lane_mask.float() * timestep).flatten(1)
            if len(values) > 1:
                timesteps = torch.cat((timesteps,) * len(values))
            arguments["denoise_mask"] = (
                lane_mask if len(values) == 1 else torch.cat((lane_mask,) * len(values))
            )
        if first.guides:
            arguments["guides"] = _batch_guides(first.guides, batch, len(values))
        velocity = self.owner.assembled.diffusion(
            model_input,
            timesteps,
            text,
            attention_mask=mask,
            frame_rate=first.frame_rate,
            **arguments,
        ).float()
        return tuple(
            calculate_denoised(Parameterization.FLOW, sigma, output, x)
            for output in velocity.chunk(len(values))
        )


def _ltxv_denoiser(
    runtime: object, compute_dtype: torch.dtype, context: SamplingAdapterContext
) -> SamplingDenoiserExecution:
    owner = cast("LTXVDiffusionRuntime", runtime)
    if context.inputs is None or context.device is None or context.schedule is None:
        raise RuntimeError("LTX-Video sampling context is unresolved")
    latent_context = cast("_LTXVLatentContext", context.inputs.latent_context)
    evaluator = _LTXVSamplingDenoiser(
        owner,
        latent_context.model_mask,
        device=context.device,
        compute_dtype=compute_dtype,
        space=cast("FluxFlowSigmas", owner.sampling_sigma_space()),
        cancelled=context.cancelled,
    )
    conditioning_evaluation = None
    if context.context_windows is not None:
        base = ConditioningEvaluation(
            evaluator.prepare_conditioning,
            evaluator.evaluate_conditioning,
            evaluator.batchable,
            evaluator.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: "dinkster.ltxv.conditioning.v1",
            standard_activation_memory_factor=owner.family.memory_factor,
        )
        windowed = windowed_conditioning_evaluation(
            base, context.context_windows, context.schedule.sigmas
        )
        conditioning_evaluation = windowed

    def unpack_video(value: torch.Tensor) -> object:
        return MultiStreamLatent.from_pairs((("video", value),))

    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        conditioning_evaluation=conditioning_evaluation,
        process_in=lambda value: value,
        process_out=lambda value: value,
        unpack_state=unpack_video,
        fixed_inpaint_latent=True,
    )


def _ltxv_device(runtime: object) -> torch.device:
    model = cast("LTXVDiffusionRuntime", runtime).assembled.diffusion
    return bound_compute_device(model.patchify_proj) or model.patchify_proj.weight.device


def _ltxv_compute_dtype(runtime: object) -> torch.dtype:
    return (
        cast("LTXVDiffusionRuntime", runtime).assembled.compute_dtype("diffusion") or torch.bfloat16
    )


class LTXVDiffusionRuntime(MultiStreamSamplingRuntime):
    """Diffusion-only classic LTX-Video custom-sampling runtime."""

    retained_offload_storage_components = frozenset()
    sampling_error = LTXVRuntimeError
    supports_denoised_capture = True
    supports_batch_noise_indices = False
    supports_sampling_shift = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=cast("SamplingLatentAdapter", _LTXVLatentAdapter()),
        denoiser=_ltxv_denoiser,
        device=_ltxv_device,
        compute_dtype=_ltxv_compute_dtype,
        flow=True,
    )

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
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._guidance = None

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

    sample_custom = cast("Any", sampling_execution)  # noqa: F811


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
