"""Native text encoding and flow denoising for split Ideogram 4 components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    IDEOGRAM4,
    IDEOGRAM4_CONFIG,
    IDEOGRAM4_SIGMAS,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    GuidanceRole,
    ModelFamily,
    Parameterization,
    PayloadDescriptor,
    PayloadReference,
    Registry,
    SamplerDescriptor,
    SchedulerDescriptor,
    SigmaSpace,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)

from .conditioning_adapters import materialize_basic_conditioning
from .guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .ideogram4_conditioner import (
    Ideogram4Conditioning,
    Ideogram4TextEncoder,
)
from .ideogram4_dit import Ideogram4DiT
from .operations import bound_compute_device, module_compute_device
from .parameterizations import calculate_denoised
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
from .qwen_image_text import QwenImageLanguageModel
from .sampling_execution import (
    CONTEXT_WINDOWS_UNSUPPORTED,
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionRegistration,
    SingleStreamLatentAdapter,
    sampling_execution,
)
from .sampling_runtime import SingleStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry

_ATTENTION_MASK_KEY = "dinkster-model-ideogram4/attention-mask"


class Ideogram4RuntimeError(ValueError):
    """An Ideogram 4 runtime request violates its native contract."""


def ideogram4_conditioning_to_carrier(value: Ideogram4Conditioning) -> ConditioningCarrier:
    if type(value) is not Ideogram4Conditioning or value.image_only:
        raise TypeError("Ideogram 4 text conditioning must be an encoded text result")
    text = tensor_to_payload_binding("text", value.embeddings, space="conditioning-text")
    bindings = [text]
    metadata: tuple[tuple[str, PayloadReference], ...] = ()
    if value.attention_mask is not None:
        attention = tensor_to_payload_binding(
            "attention-mask", value.attention_mask, space="ideogram4-attention-mask"
        )
        bindings.append(attention)
        metadata = ((_ATTENTION_MASK_KEY, PayloadReference(attention.reference_id)),)
    record = ConditioningRecord(
        channels=(
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(
                    PayloadReference(text.reference_id), text.shape, text.dtype, text.space
                ),
            ),
        ),
        extension_metadata=metadata,
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def materialize_ideogram4_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> Ideogram4Conditioning:
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("Ideogram 4 conditioning must be an exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise Ideogram4RuntimeError("Ideogram 4 conditioning requires one record")
    record = records[0]
    metadata = dict(record.extension_metadata)
    if set(metadata) - {_ATTENTION_MASK_KEY}:
        raise Ideogram4RuntimeError("Ideogram 4 conditioning contains unknown metadata")
    attention_reference = metadata.get(_ATTENTION_MASK_KEY)
    if attention_reference is not None and type(attention_reference) is not PayloadReference:
        raise Ideogram4RuntimeError("Ideogram 4 attention metadata must be a payload reference")
    referenced = {descriptor.reference.id for _, descriptor in record.channels}
    if record.mask is not None:
        referenced.add(record.mask.payload.id)
    if record.scale_vector is not None:
        referenced.add(record.scale_vector.values.reference.id)
    stripped = make_conditioning_carrier(
        ConditioningSet((replace(record, extension_metadata=()),)),
        tuple(binding for binding in carrier.bindings if binding.reference_id in referenced),
    )
    basic = materialize_basic_conditioning(stripped, device=device)
    attention = None
    if attention_reference is not None:
        bindings = {binding.reference_id: binding for binding in carrier.bindings}
        binding = bindings.get(attention_reference.id)
        if binding is None or binding.space != "ideogram4-attention-mask":
            raise Ideogram4RuntimeError("Ideogram 4 attention mask payload is unavailable")
        attention = payload_binding_to_tensor(binding).to(device)
    return Ideogram4Conditioning(basic.embeddings, basic.pooled, attention)


class _Ideogram4Denoiser:
    evaluator_identity = "dinkster.ideogram4.conditioning.v1"

    def __init__(
        self,
        model: Ideogram4DiT,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def prepare_conditioning(
        self,
        value: object,
        _role: GuidanceRole = GuidanceRole.CONDITIONAL,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not isinstance(value, Conditioning):
            raise Ideogram4RuntimeError("Ideogram 4 conditioning must be a Conditioning value")
        if value.pooled is not None:
            raise Ideogram4RuntimeError("Ideogram 4 conditioning does not accept pooled output")
        if type(value) is Ideogram4Conditioning and value.image_only:
            return None, None
        context = value.embeddings
        if (
            context.ndim != 3
            or context.shape[0] < 1
            or context.shape[1] < 1
            or context.shape[2] != IDEOGRAM4_CONFIG.text_width
        ):
            raise Ideogram4RuntimeError(
                f"Ideogram 4 context must have shape [batch,tokens,{IDEOGRAM4_CONFIG.text_width}]"
            )
        attention = value.attention_mask if type(value) is Ideogram4Conditioning else None
        if attention is not None and (
            attention.shape != context.shape[:2]
            or attention.is_floating_point()
            or attention.is_complex()
            or not bool(torch.all((attention == 0) | (attention == 1)))
        ):
            raise Ideogram4RuntimeError("Ideogram 4 attention mask must be binary [batch,tokens]")
        return context, attention

    def evaluate_conditioning(
        self,
        latent: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> torch.Tensor:
        context, attention = condition
        batch = latent.shape[0]
        if context is not None:
            if context.shape[0] not in (1, batch):
                raise Ideogram4RuntimeError("Ideogram 4 context batch must be one or match latent")
            context = context.to(device=latent.device, dtype=self.compute_dtype).expand(
                batch, -1, -1
            )
            if attention is not None:
                attention = attention.to(device=latent.device).expand(batch, -1)
        model_input = latent.to(dtype=self.compute_dtype)
        timestep = torch.full((batch,), sigma, dtype=torch.float32, device=latent.device)
        output = self.model(model_input, timestep, context, attention).float()
        return calculate_denoised(Parameterization.FLOW, sigma, output, latent)

    @staticmethod
    def batchable(
        conditions: tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...],
    ) -> bool:
        if not conditions:
            return False
        first_context, first_attention = conditions[0]
        return all(
            (context is None) == (first_context is None)
            and (attention is None) == (first_attention is None)
            and (
                context is None
                or first_context is not None
                and context.shape == first_context.shape
            )
            and (
                attention is None
                or first_attention is not None
                and attention.shape == first_attention.shape
            )
            for context, attention in conditions[1:]
        )

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        latent: torch.Tensor,
        conditions: tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...],
    ) -> None:
        if not self.batchable(conditions):
            raise Ideogram4RuntimeError("Ideogram 4 conditioning batch is incompatible")
        batch = latent.shape[0]
        contexts = tuple(context for context, _attention in conditions)
        if any(context is not None and context.shape[0] not in (1, batch) for context in contexts):
            raise Ideogram4RuntimeError(
                "Ideogram 4 conditioning batch must be one or match the latent"
            )

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> torch.Tensor:
        contexts = tuple(context for context, _attention in batch.conditions)
        model_context = (
            None
            if contexts[0] is None
            else torch.cat(
                tuple(
                    cast("torch.Tensor", context)
                    .to(device=batch.latent.device, dtype=self.compute_dtype)
                    .expand(batch.batch_size, -1, -1)
                    for context in contexts
                )
            )
        )
        attentions = tuple(attention for _context, attention in batch.conditions)
        model_attention = (
            None
            if attentions[0] is None
            else torch.cat(
                tuple(
                    cast("torch.Tensor", attention)
                    .to(device=batch.latent.device)
                    .expand(batch.batch_size, -1)
                    for attention in attentions
                )
            )
        )
        return self.model(
            batch.model_input,
            batch.timestep,
            model_context,
            model_attention,
        ).float()


class Ideogram4TextRuntime:
    def __init__(self, text: QwenImageLanguageModel) -> None:
        self._encoder = Ideogram4TextEncoder(text)
        self.text = text

    def encode_text(self, text: str) -> Ideogram4Conditioning:
        return self._encoder.encode(text)


@dataclass(frozen=True)
class _Ideogram4DiffusionAssembly:
    diffusion: Ideogram4DiT
    family: ModelFamily = IDEOGRAM4
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


def _validate_ideogram4_latent(latent: torch.Tensor) -> None:
    channels = IDEOGRAM4.single_stream_latent().channels
    if latent.ndim != 4 or latent.shape[1] != channels:
        raise Ideogram4RuntimeError(
            f"Ideogram 4 latent must have shape [batch,{channels},height,width]"
        )


def _ideogram4_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("Ideogram4DiffusionRuntime", runtime)
    if context.options:
        names = ", ".join(sorted(context.options))
        raise Ideogram4RuntimeError(f"Ideogram 4 sampling does not accept adapter options: {names}")
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            _Ideogram4Denoiser(owner.assembled.diffusion, compute_dtype=compute_dtype),
        )
    )


def _ideogram4_device(runtime: object) -> torch.device:
    owner = cast("Ideogram4DiffusionRuntime", runtime)
    return module_compute_device(owner.assembled.diffusion)


def _ideogram4_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("Ideogram4DiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


class Ideogram4DiffusionRuntime(SingleStreamSamplingRuntime):
    streamed_residency_components = frozenset()
    sampling_error = Ideogram4RuntimeError
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=SingleStreamLatentAdapter(_validate_ideogram4_latent),
        denoiser=_ideogram4_denoiser,
        device=_ideogram4_device,
        compute_dtype=_ideogram4_compute_dtype,
        flow=True,
        capabilities=CONTEXT_WINDOWS_UNSUPPORTED,
    )

    def __init__(
        self,
        diffusion: Ideogram4DiT,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        self.assembled = _Ideogram4DiffusionAssembly(diffusion, compute=compute_dtype)
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        if scheduler_registry is None:
            self._schedulers = torch_scheduler_registry()
        else:
            self._schedulers = Registry[SchedulerDescriptor]()
            for descriptor in scheduler_registry:
                self._schedulers.register(descriptor)
        self._guidance: GuidanceExecutor | None = None

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def conditioning_identity(self) -> str:
        return "dinkster.ideogram4.conditioning:v1"

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Ideogram4Conditioning:
        device = bound_compute_device(self.assembled.diffusion.input_proj)
        return materialize_ideogram4_conditioning(
            carrier,
            device=device or module_compute_device(self.assembled.diffusion),
        )

    @staticmethod
    def image_only_conditioning() -> Ideogram4Conditioning:
        return Ideogram4Conditioning(
            torch.empty((1, 0, IDEOGRAM4_CONFIG.text_width)), None, None, True
        )

    def conditioning_evaluation(
        self,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> ConditioningEvaluation[tuple[torch.Tensor | None, torch.Tensor | None]]:
        dtype = compute_dtype or self.assembled.compute_dtype("diffusion") or torch.bfloat16
        evaluator = _Ideogram4Denoiser(self.assembled.diffusion, compute_dtype=dtype)

        def prepare(
            value: object, _role: GuidanceRole
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            return evaluator.prepare_conditioning(value)

        return ConditioningEvaluation(
            prepare,
            evaluator.evaluate_conditioning,
            None,
            evaluator.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: "dinkster.ideogram4.conditioning.v1",
            standard_activation_memory_factor=self.family.memory_factor,
        )

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return IDEOGRAM4_SIGMAS

    sample_custom = sampling_execution

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise Ideogram4RuntimeError("Ideogram 4 diffusion component has no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise Ideogram4RuntimeError("Ideogram 4 diffusion component has no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise Ideogram4RuntimeError("Ideogram 4 diffusion component has no VAE codec")


__all__ = [
    "Ideogram4DiffusionRuntime",
    "Ideogram4RuntimeError",
    "Ideogram4TextRuntime",
    "ideogram4_conditioning_to_carrier",
    "materialize_ideogram4_conditioning",
]
