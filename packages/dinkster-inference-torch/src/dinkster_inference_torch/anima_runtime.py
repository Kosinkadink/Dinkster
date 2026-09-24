"""Native text encoding and flow denoising for Anima split components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from dinkster_inference import (
    ANIMA,
    ANIMA_CONFIG,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    CustomSamplingResult,
    FlowSigmas,
    GuidanceRole,
    ModelFamily,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    Registry,
    SamplerDescriptor,
    SamplingGuidance,
    SchedulerDescriptor,
    encode_conditioning_carrier,
    make_conditioning_carrier,
    tokenize_anima_prompt,
)

if TYPE_CHECKING:
    from .checkpoint_runtime import ComponentAssembly

from .anima_model import AnimaModel
from .guidance import (
    ConditioningBatch,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .operations import module_compute_device
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
from .qwen_layer_placement import qwen_layer_placement
from .qwen_text import QwenTextModel
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
    SingleStreamLatentAdapter,
    sampling_execution,
)
from .sampling_runtime import SingleStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry

_CONTEXT_WIDTH = ANIMA_CONFIG.context_width
_MIN_CONTEXT_ROWS = 512
_T5_IDS_KEY = "dinkster-anima/t5xxl-ids"
_T5_WEIGHTS_KEY = "dinkster-anima/t5xxl-weights"


def _module_compute_device(module: torch.nn.Module) -> torch.device:
    if type(module) is QwenTextModel:
        placement = qwen_layer_placement(module)
        if placement is not None:
            return placement.ranges[0].device
    return module_compute_device(module)


def _anima_sigma_space() -> FlowSigmas:
    """The family's reference model-sampling space: discrete flow
    sigmas at the registered Anima shift."""
    return FlowSigmas(shift=ANIMA_CONFIG.sampling_shift)


@dataclass(frozen=True)
class AnimaConditioning(Conditioning[torch.Tensor]):
    """Raw Qwen context plus the T5 token stream consumed by the DiT adapter."""

    t5xxl_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    t5xxl_weights: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        if self.pooled is not None:
            raise AnimaRuntimeError("Anima conditioning does not accept a pooled vector")
        if self.embeddings.ndim != 3 or self.embeddings.shape[0] < 1:
            raise AnimaRuntimeError("Anima raw context must be [batch,tokens,channels]")
        if (
            self.t5xxl_ids.ndim != 2
            or self.t5xxl_ids.shape[0] != self.embeddings.shape[0]
            or self.t5xxl_ids.shape[1] < 1
            or self.t5xxl_ids.dtype != torch.int64
        ):
            raise AnimaRuntimeError("Anima T5 ids must be nonempty int64 [batch,tokens]")
        if (
            self.t5xxl_weights.shape != (*self.t5xxl_ids.shape, 1)
            or not self.t5xxl_weights.is_floating_point()
        ):
            raise AnimaRuntimeError("Anima T5 weights must be floating [batch,tokens,1]")


def anima_conditioning_to_carrier(value: AnimaConditioning) -> ConditioningCarrier:
    """Encode raw split Anima conditioning without baking the DiT adapter."""

    if type(value) is not AnimaConditioning:
        raise TypeError("value must be exact AnimaConditioning")
    text = tensor_to_payload_binding("text", value.embeddings, space="conditioning-text")
    ids = tensor_to_payload_binding("t5xxl-ids", value.t5xxl_ids, space="anima-t5xxl-ids")
    weights = tensor_to_payload_binding(
        "t5xxl-weights", value.t5xxl_weights, space="anima-t5xxl-weights"
    )
    record = ConditioningRecord(
        channels=(
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(
                    PayloadReference(text.reference_id), text.shape, text.dtype, text.space
                ),
            ),
        ),
        extension_metadata=(
            (_T5_IDS_KEY, PayloadReference(ids.reference_id)),
            (_T5_WEIGHTS_KEY, PayloadReference(weights.reference_id)),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (text, ids, weights))


def materialize_anima_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> AnimaConditioning:
    """Materialize one canonical split Anima carrier on the requested device."""

    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise AnimaRuntimeError("Anima conditioning requires one record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.scale_vector is not None
        or record.token_layout is not None
        or type(record.schedule) is not PercentRange
        or (record.schedule.start_percent, record.schedule.end_percent) != (0.0, 1.0)
    ):
        raise AnimaRuntimeError("Anima conditioning requires one full unmodified record")
    channels = dict(record.channels)
    text = channels.pop(ConditioningChannel.TEXT, None)
    if text is None or channels:
        raise AnimaRuntimeError("Anima conditioning requires exactly one TEXT channel")
    metadata = dict(record.extension_metadata)
    if set(metadata) != {_T5_IDS_KEY, _T5_WEIGHTS_KEY}:
        raise AnimaRuntimeError("Anima conditioning requires T5 ids and weights")
    ids_reference = metadata[_T5_IDS_KEY]
    weights_reference = metadata[_T5_WEIGHTS_KEY]
    if (
        type(ids_reference) is not PayloadReference
        or type(weights_reference) is not PayloadReference
    ):
        raise AnimaRuntimeError("Anima T5 metadata must contain payload references")
    bindings = {binding.reference_id: binding for binding in carrier.bindings}

    def tensor(reference: PayloadReference, space: str) -> torch.Tensor:
        binding = bindings.get(reference.id)
        if binding is None or binding.space != space:
            raise AnimaRuntimeError(f"Anima conditioning requires {space!r} payloads")
        return payload_binding_to_tensor(binding).to(device)

    return AnimaConditioning(
        tensor(text.reference, "conditioning-text"),
        None,
        tensor(ids_reference, "anima-t5xxl-ids"),
        tensor(weights_reference, "anima-t5xxl-weights"),
    )


class AnimaTextRuntime:
    """Text-only Anima encoder over an independently resident Qwen component."""

    text_conditioning_carrier = staticmethod(anima_conditioning_to_carrier)

    def __init__(self, text: QwenTextModel) -> None:
        self.text = text

    def encode_text(self, text: str) -> AnimaConditioning:
        tokens = tokenize_anima_prompt(text)
        encoder_device = _module_compute_device(self.text)
        ids = torch.tensor(tokens.qwen_ids, dtype=torch.long, device=encoder_device).unsqueeze(0)
        attention = torch.tensor(
            tokens.qwen_attention_mask, dtype=torch.long, device=encoder_device
        ).unsqueeze(0)
        hidden = self.text(ids, attention).float()
        t5xxl_ids = torch.tensor(tokens.t5xxl_ids, dtype=torch.int64).unsqueeze(0)
        t5xxl_weights = torch.tensor(tokens.t5xxl_weights, dtype=torch.float32).reshape(1, -1, 1)
        return AnimaConditioning(hidden, None, t5xxl_ids, t5xxl_weights)


def checkpoint_text_runtime(assembled: ComponentAssembly) -> AnimaTextRuntime | None:
    text = assembled.components.get("qwen3_06b")
    return None if text is None else AnimaTextRuntime(cast("QwenTextModel", text))


class AnimaRuntimeError(ValueError):
    """An Anima runtime request violates its native contract."""


class AnimaDenoiser:
    """Single-conditioning FLOW evaluator over the native Anima DiT."""

    evaluator_identity = "dinkster.anima.conditioning.v1"

    def __init__(
        self,
        model: AnimaModel,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def prepare_conditioning(
        self,
        value: object,
        role: GuidanceRole = GuidanceRole.CONDITIONAL,
        *,
        lane_id: str | None = None,
    ) -> tuple[torch.Tensor, str]:
        if type(value) is AnimaConditioning:
            raw = value
            device = _module_compute_device(self.model)
            context = self.model.preprocess_text_embeds(
                raw.embeddings.to(device=device, dtype=self.compute_dtype),
                raw.t5xxl_ids.to(device=device),
                raw.t5xxl_weights.to(device=device),
            ).float()
            validate_context = False
        elif isinstance(value, Conditioning):
            if value.pooled is not None:
                raise AnimaRuntimeError("Anima conditioning does not accept a pooled vector")
            context = value.embeddings
            validate_context = True
        else:
            raise AnimaRuntimeError("Anima conditioning must be a Conditioning value")
        if validate_context and (
            context.ndim != 3
            or context.shape[0] < 1
            or context.shape[1] < _MIN_CONTEXT_ROWS
            or context.shape[2] != _CONTEXT_WIDTH
        ):
            raise AnimaRuntimeError(
                f"Anima context must have shape [batch,rows>={_MIN_CONTEXT_ROWS},{_CONTEXT_WIDTH}]"
            )
        if lane_id is None:
            lane_id = "negative" if role is GuidanceRole.UNCONDITIONAL else "positive"
        elif lane_id not in ("positive", "negative"):
            raise AnimaRuntimeError("Anima guidance lane id is not supported")
        return context, lane_id

    @staticmethod
    def batchable(conditions: tuple[tuple[torch.Tensor, str], ...]) -> bool:
        return bool(conditions) and all(
            condition[0].shape[1:] == conditions[0][0].shape[1:] for condition in conditions[1:]
        )

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: tuple[torch.Tensor, str]
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[tuple[torch.Tensor, str], ...],
    ) -> None:
        if not self.batchable(conditions):
            raise AnimaRuntimeError("Anima conditioning batch is empty or incompatible")
        batch = x.shape[0]
        if any(condition[0].shape[0] not in (1, batch) for condition in conditions):
            raise AnimaRuntimeError("Anima conditioning batch must be one or match the latent")

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[tuple[torch.Tensor, str]],
    ) -> torch.Tensor:
        context = torch.cat(
            [
                condition[0]
                .to(device=batch.latent.device, dtype=self.compute_dtype)
                .expand(batch.batch_size, -1, -1)
                for condition in batch.conditions
            ],
            dim=0,
        )
        return self.model(batch.model_input, batch.timestep, context).float()


def _exact_scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


@dataclass(frozen=True)
class _AnimaDiffusionAssembly:
    diffusion: AnimaModel
    family: ModelFamily = ANIMA
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


def _validate_anima_latent(latent: torch.Tensor) -> None:
    channels = ANIMA_CONFIG.latent_channels
    if latent.ndim != 5 or latent.shape[1] != channels or latent.shape[2] != 1:
        raise AnimaRuntimeError(f"Anima input must have shape [batch,{channels},1,height,width]")


@dataclass(frozen=True)
class _AnimaLatentAdapter:
    inner: SingleStreamLatentAdapter = SingleStreamLatentAdapter(_validate_anima_latent)

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
        if context.options:
            names = ", ".join(sorted(context.options))
            raise AnimaRuntimeError(f"Anima sampling does not accept adapter options: {names}")
        inputs = self.inner.prepare(
            runtime,
            family,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            context=context,
            error=error,
        )
        values = [inputs.cond]
        if inputs.cfg is not None:
            values.extend(
                value
                for name in ("uncond", "middle", "empty")
                if (value := getattr(inputs.cfg, name, None)) is not None
            )
        if any(type(value) is not AnimaConditioning for value in values):
            raise AnimaRuntimeError("split Anima sampling requires exact AnimaConditioning")
        return inputs

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[torch.Tensor]:
        if denoised is not None and type(denoised) is not torch.Tensor:
            raise TypeError("Anima denoised state must contain a torch.Tensor")
        return self.inner.finish(inputs, output, denoised)


def _anima_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("AnimaDiffusionRuntime", runtime)
    del context
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            AnimaDenoiser(owner.assembled.diffusion, compute_dtype=compute_dtype),
        )
    )


def _anima_device(runtime: object) -> torch.device:
    owner = cast("AnimaDiffusionRuntime", runtime)
    return _module_compute_device(owner.assembled.diffusion)


def _anima_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("AnimaDiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


def _prepare_anima_guidance(
    runtime: object,
    inputs: SamplingExecutionInputs,
) -> SamplingExecutionInputs:
    owner = cast("AnimaDiffusionRuntime", runtime)
    conditioning = owner._bake_conditioning(  # pyright: ignore[reportPrivateUsage]
        cast("Conditioning[torch.Tensor]", inputs.cond)
    )
    guidance = inputs.cfg
    if guidance is not None:
        guidance = SamplingGuidance(
            (
                None
                if guidance.uncond is None
                else owner._bake_conditioning(  # pyright: ignore[reportPrivateUsage]
                    cast("Conditioning[torch.Tensor]", guidance.uncond)
                )
            ),
            guidance.scale,
            guidance.transforms,
        )
    return replace(inputs, cond=conditioning, cfg=guidance)


class AnimaDiffusionRuntime(SingleStreamSamplingRuntime):
    """Diffusion-only Anima sampling over independently encoded conditioning."""

    streamed_residency_components = frozenset()
    sampling_error = AnimaRuntimeError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=_AnimaLatentAdapter(),
        denoiser=_anima_denoiser,
        device=_anima_device,
        compute_dtype=_anima_compute_dtype,
        flow=True,
        capabilities=CONTEXT_WINDOWS_UNSUPPORTED,
        prepare_guidance=_prepare_anima_guidance,
    )

    def __init__(
        self,
        diffusion: AnimaModel,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = _AnimaDiffusionAssembly(diffusion, compute=compute_dtype)
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _exact_scheduler_registry(scheduler_registry)
        self._guidance = guidance_executor

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _bake_conditioning(self, value: Conditioning[torch.Tensor]) -> Conditioning[torch.Tensor]:
        if type(value) is not AnimaConditioning:
            raise AnimaRuntimeError("split Anima sampling requires exact AnimaConditioning")
        raw = value
        diffusion = self.assembled.diffusion
        context = diffusion.preprocess_text_embeds(
            raw.embeddings.to(
                device=_module_compute_device(diffusion),
                dtype=self.assembled.compute_dtype("diffusion") or torch.bfloat16,
            ),
            raw.t5xxl_ids.to(device=_module_compute_device(diffusion)),
            raw.t5xxl_weights.to(device=_module_compute_device(diffusion)),
        ).float()
        return Conditioning(context, None)

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return _anima_sigma_space()

    sample_custom = sampling_execution

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise AnimaRuntimeError("Anima diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise AnimaRuntimeError("Anima diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise AnimaRuntimeError("Anima diffusion component carries no VAE codec")


__all__ = [
    "AnimaConditioning",
    "AnimaDenoiser",
    "AnimaDiffusionRuntime",
    "AnimaRuntimeError",
    "AnimaTextRuntime",
    "anima_conditioning_to_carrier",
    "materialize_anima_conditioning",
]
