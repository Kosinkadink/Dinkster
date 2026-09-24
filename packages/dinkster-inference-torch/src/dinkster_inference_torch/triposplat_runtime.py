"""Executable image-to-splat sampling runtime for the TripoSplat DiT.

The runtime samples the latent and camera streams jointly (the model
forward consumes and denoises both); the octree gaussian decoder and
the Flux2 VAE conditioning encoder stay upstream of sampling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    TRIPOSPLAT,
    TRIPOSPLAT_CONFIG,
    TRIPOSPLAT_SIGMAS,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    CustomSamplingResult,
    DualSamplingGuidance,
    GuidanceRole,
    LatentPackLayout,
    ModelFamily,
    MultiStreamConditioningRuntime,
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
    SigmaSpace,
    TripoSplatConfig,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)

from .denoise import to_batch
from .latent_streams import pack_latent_mask, pack_latent_streams, unpack_latent_streams
from .operations import bound_compute_device
from .parameterizations import calculate_denoised, calculate_input
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
from .sampling_execution import (
    CONTEXT_WINDOWS_UNSUPPORTED,
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
from .triposplat_model import TripoSplatModel


class TripoSplatRuntimeError(ValueError):
    """The requested operation is outside the loaded TripoSplat profile."""


@dataclass(frozen=True)
class TripoSplatConditioning:
    """DINOv3 vision features plus the optional Flux2 reference latent."""

    features: torch.Tensor
    reference_latent: torch.Tensor | None = None


_REFERENCE_LATENT_KEY = "dinkster-model-triposplat/reference-latent"
_FEATURES_SPACE = "triposplat-vision-features"
_REFERENCE_SPACE = "triposplat-reference-latent"


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SamplingCancelled("sampling cancelled")


def _validate_features(value: object, *, config: TripoSplatConfig) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError("TripoSplat features must be an exact strided floating torch.Tensor")
    if (
        value.ndim != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or value.shape[2] != config.cond_channels
    ):
        raise TripoSplatRuntimeError(
            f"TripoSplat features must be nonempty [B,rows,{config.cond_channels}],"
            f" got {tuple(value.shape)}"
        )
    return value


def _validate_reference(
    value: object,
    features: torch.Tensor,
    *,
    config: TripoSplatConfig,
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or not value.is_floating_point()
        or value.layout != torch.strided
    ):
        raise TypeError(
            "TripoSplat reference latent must be an exact strided floating torch.Tensor"
        )
    if (
        value.ndim != 4
        or value.shape[0] != features.shape[0]
        or value.shape[1] != config.cond2_channels
    ):
        raise TripoSplatRuntimeError(
            f"TripoSplat reference latent must be [B,{config.cond2_channels},height,width]"
            f" over the features batch, got {tuple(value.shape)}"
        )
    if value.shape[2] * value.shape[3] > features.shape[1]:
        raise TripoSplatRuntimeError(
            "TripoSplat reference latent tokens must not exceed the features rows"
            f" ({value.shape[2] * value.shape[3]} > {features.shape[1]})"
        )
    return value


def triposplat_conditioning_to_carrier(value: TripoSplatConditioning) -> ConditioningCarrier:
    """Encode exact TripoSplat conditioning without discarding the reference."""

    if type(value) is not TripoSplatConditioning:
        raise TypeError("value must be exact TripoSplatConditioning")
    features = tensor_to_payload_binding("features", value.features, space=_FEATURES_SPACE)
    bindings = [features]
    channels = [
        (
            ConditioningChannel.TEXT,
            PayloadDescriptor(
                PayloadReference(features.reference_id),
                features.shape,
                features.dtype,
                features.space,
            ),
        )
    ]
    metadata: list[tuple[str, PayloadReference | tuple[PayloadReference, ...]]] = []
    if value.reference_latent is not None:
        reference = tensor_to_payload_binding(
            "reference-latent", value.reference_latent, space=_REFERENCE_SPACE
        )
        bindings.append(reference)
        metadata.append((_REFERENCE_LATENT_KEY, PayloadReference(reference.reference_id)))
    record = ConditioningRecord(
        channels=tuple(channels),
        extension_metadata=tuple(metadata),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def materialize_triposplat_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> TripoSplatConditioning:
    """Materialize one canonical TripoSplat carrier on the requested device."""

    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise TripoSplatRuntimeError("TripoSplat conditioning requires one record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.scale_vector is not None
        or record.token_layout is not None
        or type(record.schedule) is not PercentRange
        or (record.schedule.start_percent, record.schedule.end_percent) != (0.0, 1.0)
    ):
        raise TripoSplatRuntimeError("TripoSplat conditioning requires one full unmodified record")
    channels = dict(record.channels)
    features = channels.pop(ConditioningChannel.TEXT, None)
    if features is None or channels:
        raise TripoSplatRuntimeError("TripoSplat conditioning requires exactly the TEXT channel")
    metadata = dict(record.extension_metadata)
    unknown = set(metadata) - {_REFERENCE_LATENT_KEY}
    if unknown:
        raise TripoSplatRuntimeError("TripoSplat conditioning contains unknown extension metadata")
    bindings = {binding.reference_id: binding for binding in carrier.bindings}

    def tensor(reference: PayloadReference, space: str) -> torch.Tensor:
        binding = bindings.get(reference.id)
        if binding is None or binding.space != space:
            raise TripoSplatRuntimeError(f"TripoSplat conditioning requires {space!r} payloads")
        return payload_binding_to_tensor(binding).to(device)

    features_tensor = tensor(features.reference, _FEATURES_SPACE)
    reference_value = metadata.get(_REFERENCE_LATENT_KEY)
    if reference_value is not None and type(reference_value) is not PayloadReference:
        raise TripoSplatRuntimeError("TripoSplat reference latent metadata must be one payload")
    reference_tensor = (
        None if reference_value is None else tensor(reference_value, _REFERENCE_SPACE)
    )
    return TripoSplatConditioning(features_tensor, reference_tensor)


@dataclass(frozen=True, slots=True)
class _TripoSplatModelConditioning:
    features: torch.Tensor
    reference_latent: torch.Tensor | None


@dataclass(frozen=True)
class _TripoSplatDiffusionAssembly:
    diffusion: TripoSplatModel
    compute: torch.dtype
    family: ModelFamily = TRIPOSPLAT
    _storage_dtype_follows_compute: bool = False

    @property
    def components(self) -> Mapping[str, torch.nn.Module]:
        return MappingProxyType({"dit": self.diffusion})

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component in ("dit", "diffusion") else None


@dataclass(frozen=True)
class _TripoSplatLatentContext:
    original: MultiStreamLatent[torch.Tensor]
    layout: LatentPackLayout


@dataclass(frozen=True)
class _TripoSplatLatentAdapter:
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
        owner = cast("TripoSplatDiffusionRuntime", runtime)
        if context.options:
            names = ", ".join(sorted(context.options))
            raise TripoSplatRuntimeError(
                f"TripoSplat sampling does not accept adapter options: {names}"
            )
        if type(latent) is not MultiStreamLatent:
            raise TripoSplatRuntimeError(
                "TripoSplat custom sampling requires a MultiStreamLatent latent"
            )
        if latent.roles != ("latent", "camera"):
            raise TripoSplatRuntimeError(
                "TripoSplat sampling requires exactly the ('latent', 'camera') streams"
            )
        latent_tensor = latent.by_role("latent")
        camera_tensor = latent.by_role("camera")
        config = TRIPOSPLAT_CONFIG
        expected_latent = (latent_tensor.shape[0], config.q_token_length, config.latent_channels)
        if latent_tensor.ndim != 3 or tuple(latent_tensor.shape) != expected_latent:
            raise TripoSplatRuntimeError(
                f"TripoSplat latent stream must have shape {expected_latent},"
                f" got {tuple(latent_tensor.shape)}"
            )
        expected_camera = (latent_tensor.shape[0], 1, config.cam_channels)
        if camera_tensor.ndim != 3 or tuple(camera_tensor.shape) != expected_camera:
            raise TripoSplatRuntimeError(
                f"TripoSplat camera stream must have shape {expected_camera},"
                f" got {tuple(camera_tensor.shape)}"
            )
        if camera_tensor.dtype != latent_tensor.dtype:
            raise TripoSplatRuntimeError("TripoSplat streams must share one dtype")
        if camera_tensor.device != latent_tensor.device:
            raise TripoSplatRuntimeError("TripoSplat streams must share one device")
        packed, layout = pack_latent_streams(latent)
        if type(noise) is not MultiStreamLatent:
            raise TripoSplatRuntimeError(
                "TripoSplat custom sampling requires MultiStreamLatent noise"
            )
        packed_noise, noise_layout = pack_latent_streams(noise)
        if noise_layout != layout:
            raise TripoSplatRuntimeError("TripoSplat noise streams lost the latent layout")
        if type(cond) is not PreparedMultiStreamConditioning:
            raise TripoSplatRuntimeError(
                "TripoSplat custom sampling requires prepared multi-stream conditioning"
            )
        if cond.runtime_identity != owner.conditioning_identity:
            raise TripoSplatRuntimeError(
                "TripoSplat conditioning was prepared by a different conditioner component"
            )
        conditioning = cond.payload
        if type(conditioning) is not TripoSplatConditioning:
            raise TypeError("conditioning must be exact TripoSplatConditioning")
        if isinstance(cfg, DualSamplingGuidance):
            raise TripoSplatRuntimeError("TripoSplat has no dual-guidance recipe")
        if isinstance(cfg, PerpNegSamplingGuidance):
            raise TripoSplatRuntimeError(
                "TripoSplat custom sampling does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        guidance_cfg: SamplingGuidance[object] | None = None
        if cfg is not None:
            uncond = cfg.uncond
            if uncond is None:
                guidance_cfg = cast("SamplingGuidance[object]", cfg)
            else:
                if type(uncond) is not PreparedMultiStreamConditioning:
                    raise TripoSplatRuntimeError(
                        "TripoSplat custom sampling guidance requires prepared"
                        " multi-stream conditioning"
                    )
                if uncond.runtime_identity != cond.runtime_identity:
                    raise TripoSplatRuntimeError(
                        "TripoSplat conditional and unconditional lanes were prepared"
                        " by different conditioner components"
                    )
                if type(uncond.payload) is not TripoSplatConditioning:
                    raise TypeError(
                        "TripoSplat guidance lanes require exact TripoSplatConditioning"
                    )
                guidance_cfg = replace(cast("SamplingGuidance[object]", cfg), uncond=uncond.payload)
        if denoise_mask is not None and not isinstance(
            denoise_mask, (torch.Tensor, MultiStreamLatent)
        ):
            raise TypeError("multi-stream denoise masks must be dense or multi-stream tensors")
        packed_mask = None if denoise_mask is None else pack_latent_mask(denoise_mask, latent)
        return SamplingExecutionInputs(
            packed,
            packed_noise,
            conditioning,
            guidance_cfg,
            packed_mask,
            _TripoSplatLatentContext(latent, layout),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
        context = cast("_TripoSplatLatentContext", inputs.latent_context)

        def restore(value: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
            unpacked = unpack_latent_streams(value, context.layout)
            return context.original.replace("latent", unpacked.by_role("latent")).replace(
                "camera", unpacked.by_role("camera")
            )

        result = restore(output)
        if denoised is None:
            return CustomSamplingResult(result, None)
        if type(denoised) is not MultiStreamLatent:
            raise TypeError("TripoSplat denoised state must contain a MultiStreamLatent")
        return CustomSamplingResult(
            result,
            context.original.replace("latent", denoised.by_role("latent")).replace(
                "camera", denoised.by_role("camera")
            ),
        )


class _TripoSplatSamplingDenoiser:
    evaluator_identity = "dinkster.triposplat.conditioning.v1"

    def __init__(
        self,
        owner: TripoSplatDiffusionRuntime,
        layout: LatentPackLayout,
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        cancelled: Callable[[], bool],
    ) -> None:
        self.owner = owner
        self.layout = layout
        self.device = device
        self.compute_dtype = compute_dtype
        self.cancelled = cancelled

    def prepare_conditioning(
        self, value: object, _role: GuidanceRole
    ) -> _TripoSplatModelConditioning:
        _check_cancelled(self.cancelled)
        if type(value) is not TripoSplatConditioning:
            raise TypeError("TripoSplat guidance lanes require exact TripoSplatConditioning")
        features = _validate_features(value.features, config=TRIPOSPLAT_CONFIG).to(
            device=self.device, dtype=self.compute_dtype
        )
        reference = value.reference_latent
        if reference is not None:
            reference = _validate_reference(reference, value.features, config=TRIPOSPLAT_CONFIG).to(
                device=self.device, dtype=self.compute_dtype
            )
        return _TripoSplatModelConditioning(features, reference)

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: _TripoSplatModelConditioning
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    def batchable(self, conditions: tuple[_TripoSplatModelConditioning, ...]) -> bool:
        if not conditions:
            return False
        first = conditions[0]
        return all(
            value.features.shape[1:] == first.features.shape[1:]
            and (value.reference_latent is None) == (first.reference_latent is None)
            and (
                value.reference_latent is None
                or first.reference_latent is None
                or value.reference_latent.shape[1:] == first.reference_latent.shape[1:]
            )
            for value in conditions[1:]
        )

    def evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[_TripoSplatModelConditioning, ...],
    ) -> tuple[torch.Tensor, ...]:
        _check_cancelled(self.cancelled)
        if not self.batchable(conditions):
            raise TripoSplatRuntimeError("TripoSplat conditioning batch is empty or incompatible")
        batch = x.shape[0]
        count = len(conditions)
        model_input = calculate_input(Parameterization.FLOW, sigma, x).to(self.compute_dtype)
        if count > 1:
            model_input = torch.cat([model_input] * count, dim=0)
        pieces = model_input.split(tuple(stream.elements for stream in self.layout.streams), dim=2)
        latent_in, camera_in = (
            piece.reshape(batch * count, *stream.shape[1:])
            for piece, stream in zip(pieces, self.layout.streams, strict=True)
        )
        timesteps = torch.full(
            (batch * count,),
            TRIPOSPLAT_SIGMAS.timestep(sigma),
            device=self.device,
            dtype=torch.float32,
        )
        features = torch.cat([to_batch(value.features, batch) for value in conditions], dim=0)
        reference = (
            None
            if conditions[0].reference_latent is None
            else torch.cat(
                [
                    to_batch(cast("torch.Tensor", value.reference_latent), batch)
                    for value in conditions
                ],
                dim=0,
            )
        )
        latent_out, camera_out = self.owner.model(
            latent_in,
            camera_in,
            timesteps,
            features,
            reference_latent=reference,
        )
        velocity = pack_latent_streams(
            MultiStreamLatent.from_pairs(
                (("latent", latent_out.float()), ("camera", camera_out.float()))
            )
        )[0]
        flow_input = x if count == 1 else torch.cat([x] * count, dim=0)
        return tuple(
            calculate_denoised(Parameterization.FLOW, sigma, velocity, flow_input).chunk(count)
        )


def _triposplat_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("TripoSplatDiffusionRuntime", runtime)
    if context.inputs is None or context.device is None:
        raise RuntimeError("TripoSplat sampling context is unresolved")
    latent_context = cast("_TripoSplatLatentContext", context.inputs.latent_context)
    evaluator = _TripoSplatSamplingDenoiser(
        owner,
        latent_context.layout,
        device=context.device,
        compute_dtype=compute_dtype,
        cancelled=context.cancelled,
    )
    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        process_in=lambda value: value,
        process_out=lambda value: value,
        unpack_state=lambda value: unpack_latent_streams(value, latent_context.layout),
    )


def _triposplat_device(runtime: object) -> torch.device:
    owner = cast("TripoSplatDiffusionRuntime", runtime)
    return bound_compute_device(owner.model.input_layer) or owner.model.input_layer.weight.device


def _triposplat_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("TripoSplatDiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


class TripoSplatDiffusionRuntime(MultiStreamSamplingRuntime):
    """Joint FLOW sampling of the TripoSplat latent and camera streams."""

    sampling_error = TripoSplatRuntimeError
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=cast("SamplingLatentAdapter", _TripoSplatLatentAdapter()),
        denoiser=_triposplat_denoiser,
        device=_triposplat_device,
        compute_dtype=_triposplat_compute_dtype,
        flow=True,
        capabilities=CONTEXT_WINDOWS_UNSUPPORTED,
    )

    def __init__(
        self,
        model: TripoSplatModel,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        if not runtime_identity:
            raise ValueError("TripoSplat runtime identity must be nonempty")
        self._model = model
        self.assembled = _TripoSplatDiffusionAssembly(model, compute_dtype)
        # The family admits exactly one published architecture, so the
        # runtime binds the canonical config rather than trusting the
        # module attribute.
        self._config: TripoSplatConfig = TRIPOSPLAT_CONFIG
        self._runtime_identity = runtime_identity
        self._compute_dtype = compute_dtype
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._guidance = None

    @property
    def model(self) -> TripoSplatModel:
        return self._model

    @property
    def family(self) -> ModelFamily:
        # The family admits exactly one published architecture, so the
        # canonical catalog entry describes every runtime instance.
        return TRIPOSPLAT

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def conditioning_identity(self) -> str:
        """Stable compatibility identity for materialized TripoSplat conditioning."""
        config = self._config
        fields = (
            config.family_id,
            config.q_token_length,
            config.latent_channels,
            config.cond_channels,
            config.cond2_channels,
            config.cam_channels,
            config.model_channels,
            config.num_blocks,
            config.num_refiner_blocks,
        )
        return "dinkster.triposplat.conditioning:v1:" + ":".join(str(field) for field in fields)

    def prepare_conditioning(self, carrier: ConditioningCarrier) -> TripoSplatConditioning:
        value = materialize_triposplat_conditioning(carrier, device="cpu")
        _validate_features(value.features, config=self._config)
        if value.reference_latent is not None:
            _validate_reference(value.reference_latent, value.features, config=self._config)
        return value

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return TRIPOSPLAT_SIGMAS

    sample_custom = cast("Any", sampling_execution)  # noqa: F811


_MultiStreamRuntimeCheck: type[MultiStreamFamilyRuntime[torch.Tensor]] = TripoSplatDiffusionRuntime
_ConditioningRuntimeCheck: type[MultiStreamConditioningRuntime] = TripoSplatDiffusionRuntime


__all__ = [
    "TripoSplatConditioning",
    "TripoSplatDiffusionRuntime",
    "TripoSplatRuntimeError",
    "materialize_triposplat_conditioning",
    "triposplat_conditioning_to_carrier",
]
