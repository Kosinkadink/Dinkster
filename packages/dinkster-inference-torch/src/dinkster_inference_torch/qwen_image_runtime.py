"""Qwen Image conditioning, component runtimes, and custom sampling seam."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    FlowSigmas,
    FluxFlowSigmas,
    InpaintConditioning,
    ModelFamily,
    Parameterization,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    Registry,
    SamplerDescriptor,
    SamplingCancelled,
    SamplingStateCallback,
    SchedulerDescriptor,
    StepCallback,
    encode_conditioning_carrier,
    load_qwen_bpe,
    make_conditioning_carrier,
    require_realized_sampling_step,
    sampling_execution_context,
)
from dinkster_inference.qwen_image_text import format_qwen_image_prompt

from .denoise import prepare_denoise_mask, run_sampler_engine
from .guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .operations import bound_compute_device, module_compute_device
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
from .qwen_image import QwenImage
from .qwen_image_assembly import AssembledQwenImage
from .qwen_image_control import (
    QwenImageControlConditioning,
    QwenImageDiffSynthConditioning,
    QwenImageDiffSynthExecution,
    QwenImageFunControlNet,
    QwenImageInstantXControlNet,
    qwen_image_control_hint_digest,
    snapshot_qwen_image_control_conditioning,
    snapshot_qwen_image_diffsynth_conditioning,
    validate_qwen_image_control_resource,
    validate_qwen_image_diffsynth_resource,
)
from .qwen_image_text import prepare_qwen_image_vision, resize_qwen_image_content
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    brownian_step_noise,
    build_custom_sampling_schedule,
    compile_guidance_plan,
    custom_denoised_callback,
    guided_denoiser,
    narrow_single_stream_custom_sampling,
    resolve_custom_sampling_request,
)
from .sampling_runtime import FlowSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry


class QwenImageRuntimeError(ValueError):
    """The direct Qwen Image invocation violates its narrow runtime contract."""


@dataclass(frozen=True)
class QwenImageConditioning(Conditioning[torch.Tensor]):
    """Qwen Image text context plus variant-owned reference inputs."""

    attention_mask: torch.Tensor | None = None
    reference_latents: tuple[torch.Tensor, ...] = ()
    additional_t_cond: torch.Tensor | None = None


_ATTENTION_MASK_KEY = "dinkster-model-qwen-image/attention-mask"
_REFERENCE_LATENTS_KEY = "dinkster-model-qwen-image/reference-latents"
_ADDITIONAL_T_COND_KEY = "dinkster-model-qwen-image/additional-t-cond"


def qwen_image_conditioning_to_carrier(value: QwenImageConditioning) -> ConditioningCarrier:
    """Encode exact Qwen Image conditioning without discarding variant inputs."""

    if type(value) is not QwenImageConditioning:
        raise TypeError("value must be exact QwenImageConditioning")
    text = tensor_to_payload_binding("text", value.embeddings, space="conditioning-text")
    bindings = [text]
    channels = [
        (
            ConditioningChannel.TEXT,
            PayloadDescriptor(
                PayloadReference(text.reference_id), text.shape, text.dtype, text.space
            ),
        )
    ]
    if value.pooled is not None:
        pooled = tensor_to_payload_binding("pooled", value.pooled, space="conditioning-pooled")
        bindings.append(pooled)
        channels.append(
            (
                ConditioningChannel.POOLED,
                PayloadDescriptor(
                    PayloadReference(pooled.reference_id),
                    pooled.shape,
                    pooled.dtype,
                    pooled.space,
                ),
            )
        )
    metadata: list[tuple[str, PayloadReference | tuple[PayloadReference, ...]]] = []
    if value.attention_mask is not None:
        attention = tensor_to_payload_binding(
            "attention-mask", value.attention_mask, space="qwen-image-attention-mask"
        )
        bindings.append(attention)
        metadata.append((_ATTENTION_MASK_KEY, PayloadReference(attention.reference_id)))
    references = []
    for index, latent in enumerate(value.reference_latents):
        binding = tensor_to_payload_binding(
            f"reference-{index}", latent, space="qwen-image-reference-latent"
        )
        bindings.append(binding)
        references.append(PayloadReference(binding.reference_id))
    if references:
        metadata.append((_REFERENCE_LATENTS_KEY, tuple(references)))
    if value.additional_t_cond is not None:
        temporal = tensor_to_payload_binding(
            "additional-t-cond",
            value.additional_t_cond,
            space="qwen-image-additional-t-cond",
        )
        bindings.append(temporal)
        metadata.append((_ADDITIONAL_T_COND_KEY, PayloadReference(temporal.reference_id)))
    record = ConditioningRecord(
        channels=tuple(channels),
        extension_metadata=tuple(metadata),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def materialize_qwen_image_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> QwenImageConditioning:
    """Materialize one canonical Qwen Image carrier on the requested device."""

    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise QwenImageRuntimeError("Qwen Image conditioning requires one record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.scale_vector is not None
        or record.token_layout is not None
        or type(record.schedule) is not PercentRange
        or (record.schedule.start_percent, record.schedule.end_percent) != (0.0, 1.0)
    ):
        raise QwenImageRuntimeError("Qwen Image conditioning requires one full unmodified record")
    channels = dict(record.channels)
    text = channels.pop(ConditioningChannel.TEXT, None)
    pooled = channels.pop(ConditioningChannel.POOLED, None)
    if text is None or channels:
        raise QwenImageRuntimeError(
            "Qwen Image conditioning requires text and optional pooled channels"
        )
    metadata = dict(record.extension_metadata)
    unknown = set(metadata) - {
        _ATTENTION_MASK_KEY,
        _REFERENCE_LATENTS_KEY,
        _ADDITIONAL_T_COND_KEY,
    }
    if unknown:
        raise QwenImageRuntimeError("Qwen Image conditioning contains unknown extension metadata")
    bindings = {binding.reference_id: binding for binding in carrier.bindings}

    def tensor(reference: PayloadReference, space: str) -> torch.Tensor:
        binding = bindings.get(reference.id)
        if binding is None or binding.space != space:
            raise QwenImageRuntimeError(f"Qwen Image conditioning requires {space!r} payloads")
        return payload_binding_to_tensor(binding).to(device)

    embeddings = tensor(text.reference, "conditioning-text")
    pooled_tensor = None if pooled is None else tensor(pooled.reference, "conditioning-pooled")
    attention_value = metadata.get(_ATTENTION_MASK_KEY)
    if attention_value is not None and type(attention_value) is not PayloadReference:
        raise QwenImageRuntimeError("Qwen Image attention mask metadata must be one payload")
    attention_mask = (
        None if attention_value is None else tensor(attention_value, "qwen-image-attention-mask")
    )
    reference_value = metadata.get(_REFERENCE_LATENTS_KEY, ())
    if not isinstance(reference_value, tuple) or any(
        type(item) is not PayloadReference for item in reference_value
    ):
        raise QwenImageRuntimeError(
            "Qwen Image references metadata must be an ordered payload tuple"
        )
    references = tuple(
        tensor(cast("PayloadReference", item), "qwen-image-reference-latent")
        for item in reference_value
    )
    temporal_value = metadata.get(_ADDITIONAL_T_COND_KEY)
    if temporal_value is not None and type(temporal_value) is not PayloadReference:
        raise QwenImageRuntimeError("Qwen Image temporal metadata must be one payload")
    temporal = (
        None if temporal_value is None else tensor(temporal_value, "qwen-image-additional-t-cond")
    )
    return QwenImageConditioning(embeddings, pooled_tensor, attention_mask, references, temporal)


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    value = cancelled()
    if type(value) is not bool:
        raise QwenImageRuntimeError("cancelled callback must return a bool")
    if value:
        raise SamplingCancelled("sampling cancelled")


def _not_cancelled() -> bool:
    return False


class _QwenImageDenoiser:
    """Single-conditioning lane evaluation; guidance math lives in the executor."""

    def __init__(
        self,
        model: QwenImage,
        cancelled: Callable[[], bool],
        compute_dtype: torch.dtype,
        control: QwenImageControlConditioning | None = None,
        diffsynth: Sequence[QwenImageDiffSynthExecution] = (),
    ) -> None:
        self._model = model
        self._cancelled = cancelled
        self._compute_dtype = compute_dtype
        self._control = (
            None if control is None else snapshot_qwen_image_control_conditioning(control)
        )
        self._control_strength = 0.0
        self._diffsynth = tuple(diffsynth)

    def set_control_strength(self, strength: float) -> None:
        if not math.isfinite(strength) or strength < 0.0:
            raise QwenImageRuntimeError(
                "Qwen Image control strength must be finite and non-negative"
            )
        self._control_strength = strength

    @staticmethod
    def prepare_conditioning(
        conditioning: object,
    ) -> QwenImageConditioning:
        if type(conditioning) is QwenImageConditioning:
            return conditioning
        context, attention_mask = cast("tuple[torch.Tensor, torch.Tensor | None]", conditioning)
        return QwenImageConditioning(context, None, attention_mask)

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        conditioning: QwenImageConditioning,
    ) -> torch.Tensor:
        context = conditioning.embeddings
        attention_mask = conditioning.attention_mask
        reference_latents = conditioning.reference_latents
        additional_t_cond = conditioning.additional_t_cond
        _check_cancelled(self._cancelled)
        model_latent = x.to(dtype=self._compute_dtype)
        timestep = torch.full((x.shape[0],), sigma, dtype=torch.float32, device=x.device)
        model_context = context.to(device=x.device, dtype=self._compute_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device)
        control = self._control
        residuals: tuple[torch.Tensor | None, ...] | None = None
        if control is not None and self._control_strength != 0.0:
            validate_qwen_image_control_resource(control.model, control.model_digest)
            hint = control.hint.to(device=x.device, dtype=self._compute_dtype)
            if control.kind == "fun":
                assert type(control.model) is QwenImageFunControlNet
                generated = control.model(self._model, model_latent, timestep, model_context, hint)
                strength = math.sqrt(self._control_strength)
            else:
                assert type(control.model) is QwenImageInstantXControlNet
                generated = control.model(
                    model_latent, timestep, model_context, hint, attention_mask
                )
                strength = self._control_strength
            residuals = tuple(
                None if residual is None else residual * strength for residual in generated
            )
        for patch in self._diffsynth:
            validate_qwen_image_diffsynth_resource(patch.model, patch.model_digest)
        if residuals is not None and self._diffsynth:
            output = self._model(
                model_latent,
                timestep,
                model_context,
                attention_mask,
                tuple(
                    value.to(device=x.device, dtype=self._compute_dtype)
                    for value in reference_latents
                ),
                (None if additional_t_cond is None else additional_t_cond.to(device=x.device)),
                control_residuals=residuals,
                block_patches=self._diffsynth,
            )
        elif residuals is not None:
            output = self._model(
                model_latent,
                timestep,
                model_context,
                attention_mask,
                tuple(
                    value.to(device=x.device, dtype=self._compute_dtype)
                    for value in reference_latents
                ),
                (None if additional_t_cond is None else additional_t_cond.to(device=x.device)),
                control_residuals=residuals,
            )
        elif self._diffsynth:
            output = self._model(
                model_latent,
                timestep,
                model_context,
                attention_mask,
                tuple(
                    value.to(device=x.device, dtype=self._compute_dtype)
                    for value in reference_latents
                ),
                (None if additional_t_cond is None else additional_t_cond.to(device=x.device)),
                block_patches=self._diffsynth,
            )
        elif reference_latents or additional_t_cond is not None:
            output = self._model(
                model_latent,
                timestep,
                model_context,
                attention_mask,
                tuple(
                    value.to(device=x.device, dtype=self._compute_dtype)
                    for value in reference_latents
                ),
                (None if additional_t_cond is None else additional_t_cond.to(device=x.device)),
            )
        else:
            output = self._model(model_latent, timestep, model_context, attention_mask)
        _check_cancelled(self._cancelled)
        return x - output.to(torch.float32) * sigma

    def batchable(self, conditions: tuple[QwenImageConditioning, ...]) -> bool:
        if not conditions or self._control is not None or self._diffsynth:
            return False
        first = conditions[0]
        return all(
            condition.embeddings.ndim == 3
            and condition.embeddings.shape == first.embeddings.shape
            and condition.embeddings.shape[1] > 0
            and (condition.attention_mask is None) == (first.attention_mask is None)
            and (
                condition.attention_mask is None
                or first.attention_mask is not None
                and condition.attention_mask.shape == condition.embeddings.shape[:2]
                and condition.attention_mask.shape == first.attention_mask.shape
                and not condition.attention_mask.is_complex()
            )
            and not condition.reference_latents
            and condition.additional_t_cond is None
            for condition in conditions
        )

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[QwenImageConditioning, ...],
    ) -> None:
        if not self.batchable(conditions):
            raise QwenImageRuntimeError("Qwen Image conditioning batch is incompatible")
        batch = x.shape[0]
        if any(condition.embeddings.shape[0] not in (1, batch) for condition in conditions):
            raise QwenImageRuntimeError(
                "Qwen Image conditioning batch must be one or match the latent"
            )

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[QwenImageConditioning],
    ) -> torch.Tensor:
        contexts = tuple(
            condition.embeddings.to(device=batch.latent.device, dtype=self._compute_dtype).expand(
                batch.batch_size, -1, -1
            )
            for condition in batch.conditions
        )
        first_mask = batch.conditions[0].attention_mask
        attention_mask = (
            None
            if first_mask is None
            else torch.cat(
                tuple(
                    cast("torch.Tensor", condition.attention_mask)
                    .to(device=batch.latent.device)
                    .expand(batch.batch_size, -1)
                    for condition in batch.conditions
                )
            )
        )
        _check_cancelled(self._cancelled)
        output = self._model(
            batch.model_input,
            batch.timestep,
            torch.cat(contexts),
            attention_mask,
        )
        _check_cancelled(self._cancelled)
        return output

    @staticmethod
    def _conditioning_denoised(
        batch: ConditioningBatch[QwenImageConditioning],
        output: torch.Tensor,
        flow_input: torch.Tensor,
    ) -> torch.Tensor:
        return flow_input - output.to(torch.float32) * batch.sigma


class QwenImageRuntime(FlowSamplingRuntime):
    """Native Qwen Image family runtime over generic assembly plans."""

    sampling_error = QwenImageRuntimeError
    supports_denoised_capture = True

    def __init__(
        self,
        assembled: AssembledQwenImage,
        *,
        runtime_identity: str,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        if assembled.family is None:
            raise QwenImageRuntimeError("Qwen Image runtime requires a declared family")
        self.assembled = assembled
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        if scheduler_registry is None:
            self._schedulers = torch_scheduler_registry()
        else:
            self._schedulers = Registry[SchedulerDescriptor]()
            for descriptor in scheduler_registry:
                self._schedulers.register(descriptor)

    @property
    def family(self) -> ModelFamily:
        assert self.assembled.family is not None
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def conditioning_identity(self) -> str:
        config = self.assembled.diffusion.config
        return (
            "dinkster.qwen-image.conditioning:v1:"
            f"{self.family.id}:{config.patchified_input_channels}:"
            f"{config.hidden_width}:{config.transformer_blocks}:"
            f"{config.default_ref_method}:{config.use_additional_t_cond}"
        )

    def _diffusion_device(self) -> torch.device:
        input_projection = getattr(self.assembled.diffusion, "img_in", None)
        device = (
            bound_compute_device(input_projection)
            if isinstance(input_projection, torch.nn.Module)
            else None
        )
        return device or module_compute_device(self.assembled.diffusion)

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> QwenImageConditioning:
        return materialize_qwen_image_conditioning(carrier, device=self._diffusion_device())

    def encode_text(self, text: str) -> QwenImageConditioning:
        return QwenImageTextRuntime(self.assembled.text).encode_text(text)

    def encode_edit(self, text: str, contents: Sequence[torch.Tensor]) -> QwenImageConditioning:
        images = tuple(contents)
        if not 1 <= len(images) <= 3:
            raise QwenImageRuntimeError("Qwen Image Edit requires one to three reference images")
        context, attention_mask = self.encode_edit_text(text, images, edit_plus=len(images) > 1)
        multiple = 1 if len(images) == 1 else 8
        references = tuple(
            self.encode_content(
                resize_qwen_image_content(image, target_pixels=1024 * 1024, multiple=multiple)
            )
            for image in images
        )
        return QwenImageConditioning(context, None, attention_mask, references)

    def encode_edit_text(
        self,
        text: str,
        contents: Sequence[torch.Tensor],
        *,
        edit_plus: bool = False,
        image_slots: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode an Edit prompt and vision tokens without accessing the codec."""

        return QwenImageTextRuntime(self.assembled.text).encode_edit_text(
            text, contents, edit_plus=edit_plus, image_slots=image_slots
        )

    def _sigma_space(self) -> FlowSigmas | FluxFlowSigmas:
        if self._sampling_space_override is not None:
            return self._sampling_space_override
        return FluxFlowSigmas(shift=self.family.sampling.shift or 1.0, timesteps=10000)

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas | FluxFlowSigmas:
        return self._sigma_space()

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
        control: QwenImageControlConditioning | None = None,
        diffsynth: Sequence[QwenImageDiffSynthConditioning | QwenImageDiffSynthExecution] = (),
        compute_dtype: torch.dtype | None = None,
        capture_denoised: bool = True,
    ) -> CustomSamplingResult[torch.Tensor]:
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=QwenImageRuntimeError,
        )
        if type(cond) is not QwenImageConditioning:
            raise QwenImageRuntimeError("Qwen Image requires exact QwenImageConditioning")
        typed_cond = cond
        config = self.assembled.diffusion.config
        if (
            latent.ndim != 5
            or latent.shape[1] != 16
            or latent.shape[2] < 1
            or (not config.use_additional_t_cond and latent.shape[2] != 1)
        ):
            raise QwenImageRuntimeError(
                "Qwen Image latent must have shape [batch,16,1,height,width], except that "
                "the Layered profile accepts a positive layer extent"
            )
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=QwenImageRuntimeError
        )
        latent = latent.to(self._diffusion_device())
        space = self._sigma_space()
        schedule = build_custom_sampling_schedule(request.sigmas, space, sampler, flow=True)
        step_noise = brownian_step_noise(sampler, schedule, latent, seed=seed)
        if compute_dtype is None:
            compute_dtype = self.assembled.compute_dtype("diffusion") or torch.bfloat16
        admitted_control = None
        control_strengths: tuple[float, ...] | None = None
        if control is not None:
            admitted_control = snapshot_qwen_image_control_conditioning(control)
            if len(schedule.sigmas) > 1:
                start_sigma = space.percent_to_sigma(
                    admitted_control.application.window.start_percent
                )
                end_sigma = space.percent_to_sigma(admitted_control.application.window.end_percent)
                control_strengths = tuple(
                    admitted_control.application.strength
                    if end_sigma <= sigma <= start_sigma
                    else 0.0
                    for sigma in schedule.sigmas[:-1]
                )
                if sampler.id in {"dinkster.dpm_fast", "dinkster.dpm_adaptive"} and any(
                    value != control_strengths[0] for value in control_strengths[1:]
                ):
                    raise QwenImageRuntimeError(
                        f"sampler {sampler.id} requires constant Qwen Image control strength"
                    )
        prepared_diffsynth: list[QwenImageDiffSynthExecution] = []
        diffsynth_identity_lines: list[str] = []
        for index, value in enumerate(diffsynth):
            if type(value) is QwenImageDiffSynthConditioning:
                admitted = snapshot_qwen_image_diffsynth_conditioning(value)
                prepared_diffsynth.append(self._prepare_diffsynth(latent, admitted, compute_dtype))
                diffsynth_identity_lines.append(
                    f"{index}:kind={admitted.kind}:model={admitted.model_digest}:"
                    f"content={admitted.content_digest}:mask={admitted.mask_digest}:"
                    f"strength={admitted.strength.hex()}"
                )
                continue
            if type(value) is not QwenImageDiffSynthExecution:
                raise TypeError(
                    "Qwen Image DiffSynth input must be exact conditioning or execution"
                )
            if value.condition.shape[0] != latent.shape[0]:
                raise QwenImageRuntimeError(
                    "Qwen Image DiffSynth condition batch must match the latent"
                )
            prepared = QwenImageDiffSynthExecution(
                value.model,
                value.condition.detach().clone(),
                value.strength,
                value.model_digest,
            )
            prepared_diffsynth.append(prepared)
            diffsynth_identity_lines.append(
                f"{index}:kind=prepared:model={prepared.model_digest}:"
                f"condition={qwen_image_control_hint_digest(prepared.condition)}:"
                f"strength={prepared.strength.hex()}"
            )
        admitted_diffsynth = tuple(prepared_diffsynth)
        evaluator = _QwenImageDenoiser(
            self.assembled.diffusion,
            _not_cancelled,
            compute_dtype,
            control=admitted_control,
            diffsynth=admitted_diffsynth,
        )
        if control_strengths is not None and sampler.id in {
            "dinkster.dpm_fast",
            "dinkster.dpm_adaptive",
        }:
            evaluator.set_control_strength(control_strengths[0])
        plan = compile_guidance_plan(typed_cond, cfg, sampler, None)
        evaluator_identity = "dinkster.qwen-image.conditioning.v1"
        if admitted_control is not None:
            control_identity = "\n".join(
                (
                    f"kind={admitted_control.kind}",
                    f"model={admitted_control.model_digest}",
                    f"hint={admitted_control.hint_digest}",
                    f"strength={admitted_control.application.strength.hex()}",
                    f"start={admitted_control.application.window.start_percent.hex()}",
                    f"end={admitted_control.application.window.end_percent.hex()}",
                    "",
                )
            )
            evaluator_identity += (
                ":control=" + hashlib.sha256(control_identity.encode()).hexdigest()
            )
        if admitted_diffsynth:
            diffsynth_identity = "\n".join(diffsynth_identity_lines)
            evaluator_identity += (
                ":diffsynth=" + hashlib.sha256(diffsynth_identity.encode()).hexdigest()
            )

        def prepare(value: object, _role: object) -> QwenImageConditioning:
            if type(value) is not QwenImageConditioning:
                raise QwenImageRuntimeError("Qwen Image guidance lane identity changed")
            return evaluator.prepare_conditioning(value)

        report_state: SamplingStateCallback | None
        captured: list[torch.Tensor]
        if capture_denoised:
            report_state, captured = custom_denoised_callback(self.family, on_state)
        else:
            report_state, captured = on_state, []
        denoiser = guided_denoiser(
            ConditioningEvaluation(
                prepare,
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                evaluator_identity=lambda _role: evaluator_identity,
                standard_activation_memory_factor=self.family.memory_factor,
            ),
            input=latent,
            executor=None,
            plan=plan,
            execution=sampling_execution_context(schedule.sigmas, seed, on_step, report_state),
        )

        def apply_control_step(index: int) -> None:
            assert control_strengths is not None
            require_realized_sampling_step(
                index,
                float(schedule.sigmas[index]),
                index / (len(schedule.sigmas) - 2) if len(schedule.sigmas) > 2 else 0.0,
            )
            evaluator.set_control_strength(control_strengths[index])

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
            process_in=lambda value: value,
            process_out=lambda value: value,
            seed=seed,
            noise_kind=sampler.noise,
            noise_sampler=step_noise,
            percent_to_sigma=space.percent_to_sigma,
            on_step=on_step,
            on_step_begin=(
                None
                if control_strengths is None
                or sampler.id in {"dinkster.dpm_fast", "dinkster.dpm_adaptive"}
                else apply_control_step
            ),
            on_state=report_state,
            denoise_mask=prepare_denoise_mask(denoise_mask, latent),
        )
        return CustomSamplingResult(output, captured[-1] if captured else None)

    def _prepare_diffsynth(
        self,
        target: torch.Tensor,
        conditioning: QwenImageDiffSynthConditioning,
        compute_dtype: torch.dtype,
    ) -> QwenImageDiffSynthExecution:
        admitted = conditioning
        if admitted.content.shape[0] != target.shape[0]:
            raise QwenImageRuntimeError("Qwen Image DiffSynth content batch must match the latent")
        content = admitted.content.to(device=target.device)
        spatial_ratio = self.assembled.vae.config.spatial_ratio
        target_size = (target.shape[-2] * spatial_ratio, target.shape[-1] * spatial_ratio)
        if content.shape[-2:] != target_size:
            content = F.interpolate(content, size=target_size, mode="area")
        encoded = self.assembled.vae.encode(content.unsqueeze(2) * 2.0 - 1.0)
        latent = self.assembled.vae.process_in(encoded)
        if admitted.kind == "diffsynth_inpaint":
            if admitted.mask is None:
                mask = torch.ones_like(latent[:, :1])
            else:
                mask = admitted.mask
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask = 1.0 - F.interpolate(
                    mask.to(device=latent.device, dtype=latent.dtype),
                    size=latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask.unsqueeze(2)
            latent = torch.cat((latent, mask), dim=1)
        condition = admitted.model.prepare_condition(latent.to(dtype=compute_dtype))
        return QwenImageDiffSynthExecution(
            admitted.model,
            condition,
            admitted.strength,
            admitted.model_digest,
        )

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return WanVAECodecRuntime(
            self.assembled.vae,
            layered=self.assembled.diffusion.config.use_additional_t_cond,
            compute_dtype=self.assembled.compute_dtype("vae"),
        ).decode_latent(latent)

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        return WanVAECodecRuntime(
            self.assembled.vae, compute_dtype=self.assembled.compute_dtype("vae")
        ).encode_content(content)


class QwenImageTextRuntime:
    """Text-only Qwen Image encoder over an independently resident component."""

    def __init__(self, text: Any) -> None:
        self.text = text

    def _input_device(self) -> torch.device:
        embedding = getattr(getattr(self.text, "model", None), "embed_tokens", None)
        if isinstance(embedding, torch.nn.Module):
            device = bound_compute_device(embedding)
            if device is not None:
                return device
        return module_compute_device(self.text)

    def encode_text(self, text: str) -> QwenImageConditioning:
        prompt = format_qwen_image_prompt(text)
        ids = torch.tensor(
            (load_qwen_bpe().encode(prompt.text),),
            dtype=torch.long,
            device=self._input_device(),
        )
        context, attention_mask = self.text(ids)
        return QwenImageConditioning(context, None, attention_mask)

    def encode_edit_text(
        self,
        text: str,
        contents: Sequence[torch.Tensor],
        *,
        edit_plus: bool = False,
        image_slots: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        images = tuple(contents)
        maximum = 3 if edit_plus else 1
        if len(images) > maximum:
            variant = "Edit Plus" if edit_plus else "Edit"
            raise QwenImageRuntimeError(f"Qwen Image {variant} accepts zero to {maximum} images")
        prompt = format_qwen_image_prompt(
            text, image_count=len(images), edit_plus=edit_plus, image_slots=image_slots
        )
        ids = torch.tensor(
            (load_qwen_bpe().encode(prompt.text),),
            dtype=torch.long,
            device=self._input_device(),
        )
        vision_target = 384 * 384 if edit_plus else 1024 * 1024
        vision_inputs = tuple(
            prepare_qwen_image_vision(image, target_pixels=vision_target) for image in images
        )
        return self.text(
            ids,
            image_patches=tuple(value[0] for value in vision_inputs),
            image_grid_thw=tuple(value[1] for value in vision_inputs),
        )


class WanVAECodecRuntime:
    """Wan VAE codec operations over an independently resident component."""

    def __init__(
        self,
        vae: Any,
        *,
        layered: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.vae = vae
        self.layered = layered
        if compute_dtype is None and callable(method := getattr(vae, "compute_dtype", None)):
            compute_dtype = cast("torch.dtype", method())
        self.compute_dtype = compute_dtype

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        vae_latent = self.vae.process_out(latent)
        if self.compute_dtype is not None:
            vae_latent = vae_latent.to(self.compute_dtype)
        decoded = self.vae.decode(vae_latent)
        if (
            decoded.ndim != 5
            or decoded.shape[2] < 1
            or (not self.layered and decoded.shape[2] != 1)
        ):
            raise QwenImageRuntimeError(
                "Qwen Image decoded content must contain one frame, except that the Layered "
                "profile accepts a positive layer extent"
            )
        normalized = ((decoded.to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
        return normalized if self.layered else normalized[:, :, 0]

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        if content.ndim != 4 or content.shape[1] != 3:
            raise QwenImageRuntimeError("Qwen Image content must have shape [batch,3,height,width]")
        vae_content = content.unsqueeze(2) * 2.0 - 1.0
        if self.compute_dtype is not None:
            vae_content = vae_content.to(self.compute_dtype)
        encoded = self.vae.encode(vae_content)
        return self.vae.process_in(encoded)


@dataclass(frozen=True)
class _QwenImageDiffusionAssembly:
    diffusion: QwenImage
    family: ModelFamily
    vae: Any | None = None
    # Residency enrollment reads these; the component is strict-loaded at
    # its compute dtype already, so storage never follows a separate target.
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, role: str) -> torch.dtype | None:
        return next(self.diffusion.parameters()).dtype if role == "diffusion" else None


class QwenImageDiffusionRuntime(FlowSamplingRuntime):
    """Diffusion-only Qwen Image sampling facade."""

    sampling_error = QwenImageRuntimeError
    supports_denoised_capture = True

    def __init__(
        self,
        diffusion: QwenImage,
        family: ModelFamily,
        *,
        runtime_identity: str,
        vae: Any | None = None,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        self.assembled = _QwenImageDiffusionAssembly(diffusion, family, vae)
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        if scheduler_registry is None:
            self._schedulers = torch_scheduler_registry()
        else:
            self._schedulers = Registry[SchedulerDescriptor]()
            for descriptor in scheduler_registry:
                self._schedulers.register(descriptor)

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    runtime_identity = QwenImageRuntime.runtime_identity
    conditioning_identity = QwenImageRuntime.conditioning_identity
    prepare_single_stream_conditioning = QwenImageRuntime.prepare_single_stream_conditioning
    _diffusion_device = QwenImageRuntime._diffusion_device  # pyright: ignore[reportPrivateUsage]
    _sigma_space = QwenImageRuntime._sigma_space  # pyright: ignore[reportPrivateUsage]
    check_custom_sampling = QwenImageRuntime.check_custom_sampling
    sample_custom = QwenImageRuntime.sample_custom

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas | FluxFlowSigmas:
        return self._sigma_space()

    # The FamilyRuntime members a diffusion-only component cannot serve
    # refuse explicitly; text and VAE ride their own component handles.
    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise QwenImageRuntimeError("Qwen Image diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise QwenImageRuntimeError("Qwen Image diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise QwenImageRuntimeError("Qwen Image diffusion component carries no VAE codec")

    def _prepare_diffsynth(
        self,
        target: torch.Tensor,
        conditioning: QwenImageDiffSynthConditioning,
        compute_dtype: torch.dtype,
    ) -> QwenImageDiffSynthExecution:
        if self.assembled.vae is None:
            raise QwenImageRuntimeError("Qwen Image DiffSynth requires an explicitly supplied VAE")
        return QwenImageRuntime._prepare_diffsynth(self, target, conditioning, compute_dtype)  # pyright: ignore[reportPrivateUsage, reportArgumentType]


__all__ = [
    "QwenImageConditioning",
    "QwenImageDiffusionRuntime",
    "QwenImageRuntime",
    "QwenImageRuntimeError",
    "QwenImageTextRuntime",
    "WanVAECodecRuntime",
    "materialize_qwen_image_conditioning",
    "qwen_image_conditioning_to_carrier",
]
