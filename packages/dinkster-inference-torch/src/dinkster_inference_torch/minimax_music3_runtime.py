"""MiniMax Music 3 text conditioning and FLOW sampling runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    MINIMAX_MUSIC3,
    MINIMAX_MUSIC3_CONFIG,
    Conditioning,
    ConditioningCarrier,
    ConditioningSet,
    ConditionScaleVector,
    FlowSigmas,
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
from tokenizers import Tokenizer

from .conditioning_adapters import basic_conditioning_to_carrier, materialize_basic_conditioning
from .guidance import (
    ConditioningBatch,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .memory import soft_empty_cache
from .minimax_music3_model import MiniMaxMusic3DiT
from .minimax_music3_text import MiniMaxMusic3TextModel, tokenize_music_prompt
from .operations import module_compute_device
from .parameterizations import calculate_denoised
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
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

_CONTEXT_WIDTH = MINIMAX_MUSIC3_CONFIG.context_layers * MINIMAX_MUSIC3_CONFIG.context_width
_CONDITION_SCALE_SPACE = "conditioning-scale"


@dataclass(frozen=True)
class MiniMaxMusic3Conditioning(Conditioning[torch.Tensor]):
    conditioning_scale: torch.Tensor | None = None


def minimax_music3_conditioning_to_carrier(
    value: Conditioning[torch.Tensor],
) -> ConditioningCarrier:
    """Preserve the model's per-batch condition scale in the canonical carrier."""

    carrier = basic_conditioning_to_carrier(value)
    scale = (
        value.conditioning_scale
        if isinstance(value, MiniMaxMusic3Conditioning) and value.conditioning_scale is not None
        else value.embeddings.new_ones((value.embeddings.shape[0],))
    )
    if scale.ndim != 1 or scale.shape[0] not in (1, value.embeddings.shape[0]):
        raise MiniMaxMusic3RuntimeError(
            "MiniMax Music 3 condition scale must be rank one and broadcast over the text batch"
        )
    binding = tensor_to_payload_binding("conditioning_scale", scale, space=_CONDITION_SCALE_SPACE)
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id), binding.shape, binding.dtype, binding.space
    )
    (record,) = carrier.conditioning.records
    return make_conditioning_carrier(
        ConditioningSet((replace(record, scale_vector=ConditionScaleVector(descriptor)),)),
        (*carrier.bindings, binding),
    )


def materialize_minimax_music3_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> MiniMaxMusic3Conditioning:
    """Materialize MiniMax Music 3 text and its per-batch condition scale."""

    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise ValueError(f"MiniMax Music 3 conditioning carries one record, got {len(records)}")
    record = records[0]
    vector = record.scale_vector
    scale_reference = None if vector is None else vector.values.reference.id
    basic = make_conditioning_carrier(
        ConditioningSet((replace(record, scale_vector=None),)),
        tuple(binding for binding in carrier.bindings if binding.reference_id != scale_reference),
    )
    conditioning = materialize_basic_conditioning(basic, device=device)
    if vector is None:
        scale = conditioning.embeddings.new_ones((conditioning.embeddings.shape[0],))
    else:
        descriptor = vector.values
        if descriptor.space != _CONDITION_SCALE_SPACE:
            raise ValueError(
                "MiniMax Music 3 condition scale payload space must be "
                f"{_CONDITION_SCALE_SPACE!r}, got {descriptor.space!r}"
            )
        bindings = {binding.reference_id: binding for binding in carrier.bindings}
        scale = payload_binding_to_tensor(bindings[descriptor.reference.id]).to(device)
    if not scale.is_floating_point() or not bool(torch.isfinite(scale).all().item()):
        raise ValueError("MiniMax Music 3 condition scale must contain finite floating values")
    if scale.ndim != 1 or scale.shape[0] not in (1, conditioning.embeddings.shape[0]):
        raise ValueError(
            "MiniMax Music 3 condition scale must be rank one and broadcast over the text batch"
        )
    return MiniMaxMusic3Conditioning(conditioning.embeddings, None, scale)


def _sigma_space() -> FlowSigmas:
    return FlowSigmas()


def _scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


class MiniMaxMusic3TextRuntime:
    def __init__(
        self,
        text: MiniMaxMusic3TextModel,
        tokenizer: Tokenizer,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        if type(text) is not MiniMaxMusic3TextModel:
            raise TypeError("MiniMax Music 3 text runtime requires its native AR model")
        if type(tokenizer) is not Tokenizer:
            raise TypeError("MiniMax Music 3 text runtime requires its embedded tokenizer")
        self.text = text
        self.tokenizer = tokenizer
        self.compute_dtype = compute_dtype

    def encode_text(
        self,
        caption: str,
        lyrics: str,
        *,
        seed: int,
        max_audio_frames: int,
        cfg_scale: float,
        top_k: int,
    ) -> Conditioning[torch.Tensor]:
        tokens = tokenize_music_prompt(self.tokenizer, caption, lyrics)
        ids = torch.tensor([tokens], dtype=torch.long)
        try:
            hidden = self.text.generate(
                ids,
                seed,
                max_audio_frames,
                compute_dtype=self.compute_dtype,
                cfg_scale=cfg_scale,
                top_k=top_k,
            )
        finally:
            soft_empty_cache(module_compute_device(self.text))
        embeddings = hidden.unsqueeze(0)
        return MiniMaxMusic3Conditioning(
            embeddings,
            None,
            embeddings.new_ones((embeddings.shape[0],)),
        )


class MiniMaxMusic3RuntimeError(ValueError):
    pass


class MiniMaxMusic3Denoiser:
    def __init__(
        self,
        model: MiniMaxMusic3DiT,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype
        self._condition_key: tuple[object, ...] | None = None
        self._prepared_condition: torch.Tensor | None = None
        self._rotary_table: torch.Tensor | None = None

    def prepare_conditioning(
        self, value: object, *, lane_id: str = "positive"
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        if not isinstance(value, Conditioning):
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 conditioning must be a Conditioning value"
            )
        if value.pooled is not None:
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 conditioning does not accept a pooled vector"
            )
        context = value.embeddings
        if (
            context.ndim != 3
            or context.shape[0] < 1
            or context.shape[1] < 1
            or context.shape[2] != _CONTEXT_WIDTH
        ):
            raise MiniMaxMusic3RuntimeError(
                f"MiniMax Music 3 context must be [batch,frames,{_CONTEXT_WIDTH}]"
            )
        if lane_id not in ("positive", "negative"):
            raise MiniMaxMusic3RuntimeError("MiniMax Music 3 guidance lane id is not supported")
        scale = (
            value.conditioning_scale
            if isinstance(value, MiniMaxMusic3Conditioning) and value.conditioning_scale is not None
            else context.new_ones((context.shape[0],))
        )
        if scale.ndim != 1 or scale.shape[0] not in (1, context.shape[0]):
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 condition scale must be rank one and broadcast over the text batch"
            )
        if not scale.is_floating_point() or not bool(torch.isfinite(scale).all().item()):
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 condition scale must contain finite floating values"
            )
        return context, lane_id, scale

    @staticmethod
    def batchable(conditions: tuple[tuple[torch.Tensor, str, torch.Tensor], ...]) -> bool:
        return bool(conditions) and all(
            condition[0].shape[1:] == conditions[0][0].shape[1:] for condition in conditions[1:]
        )

    def evaluate_conditioning(
        self,
        latent: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor, str, torch.Tensor],
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(latent, sigma, (condition,))[0]

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        latent: torch.Tensor,
        conditions: tuple[tuple[torch.Tensor, str, torch.Tensor], ...],
    ) -> None:
        if not self.batchable(conditions):
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 conditioning batch is empty or incompatible"
            )
        batch = latent.shape[0]
        if any(condition[0].shape[0] not in (1, batch) for condition in conditions):
            raise MiniMaxMusic3RuntimeError(
                "MiniMax Music 3 conditioning batch must be one or match the latent"
            )

    @staticmethod
    def _conditioning_timestep(sigma: float) -> float:
        return 1.0 - sigma

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[tuple[torch.Tensor, str, torch.Tensor]],
    ) -> torch.Tensor:
        condition_key = (
            batch.batch_size,
            batch.latent.shape[-1],
            batch.latent.device,
            self.compute_dtype,
            *((id(condition[0]), id(condition[2])) for condition in batch.conditions),
        )
        if condition_key != self._condition_key:
            context = torch.cat(
                [
                    condition[0]
                    .to(device=batch.latent.device, dtype=self.compute_dtype)
                    .expand(batch.batch_size, -1, -1)
                    for condition in batch.conditions
                ],
                dim=0,
            )
            conditioning_scale = torch.cat(
                [
                    condition[2]
                    .to(device=batch.latent.device, dtype=self.compute_dtype)
                    .reshape(-1, 1, 1)
                    .expand(batch.batch_size, -1, -1)
                    for condition in batch.conditions
                ],
                dim=0,
            )
            self._prepared_condition = self.model.prepare_condition(context, conditioning_scale)
            self._rotary_table = self.model.prepare_rotary(batch.model_input)
            self._condition_key = condition_key
        assert self._prepared_condition is not None
        assert self._rotary_table is not None
        return self.model.forward_prepared(
            batch.model_input,
            batch.timestep,
            self._prepared_condition,
            self._rotary_table,
        ).float()

    @staticmethod
    def _conditioning_denoised(
        batch: ConditioningBatch[tuple[torch.Tensor, str, torch.Tensor]],
        output: torch.Tensor,
        flow_input: torch.Tensor,
    ) -> torch.Tensor:
        if batch.lane_count > 1:
            flow_input = flow_input.unflatten(0, (batch.lane_count, batch.batch_size))
            output = output.unflatten(0, (batch.lane_count, batch.batch_size))
        if torch.is_grad_enabled():
            denoised = calculate_denoised(
                Parameterization.FLOW,
                batch.sigma,
                output,
                flow_input,
            )
        else:
            denoised = flow_input - output.mul_(batch.sigma)
        if batch.lane_count > 1:
            denoised = denoised.flatten(0, 1)
        return denoised


@dataclass(frozen=True)
class _MiniMaxMusic3DiffusionAssembly:
    diffusion: MiniMaxMusic3DiT
    family: ModelFamily = MINIMAX_MUSIC3
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


class _MiniMaxMusic3SamplingDenoiser:
    evaluator_identity = "dinkster.minimax_music3.conditioning.v1"

    def __init__(self, model: MiniMaxMusic3DiT, *, compute_dtype: torch.dtype) -> None:
        self.evaluator = MiniMaxMusic3Denoiser(model, compute_dtype=compute_dtype)

    def prepare_conditioning(
        self,
        value: object,
        role: GuidanceRole,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        return self.evaluator.prepare_conditioning(
            value,
            lane_id=("negative" if role is GuidanceRole.UNCONDITIONAL else "positive"),
        )

    def evaluate_conditioning(
        self,
        latent: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor, str, torch.Tensor],
    ) -> torch.Tensor:
        return self.evaluator.evaluate_conditioning(latent, sigma, condition)

    def evaluate_conditioning_batch(
        self,
        latent: torch.Tensor,
        sigma: float,
        conditions: tuple[tuple[torch.Tensor, str, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, ...]:
        return self.evaluator.evaluate_conditioning_batch(latent, sigma, conditions)

    def batchable(
        self,
        conditions: tuple[tuple[torch.Tensor, str, torch.Tensor], ...],
    ) -> bool:
        return self.evaluator.batchable(conditions)


def _validate_minimax_music3_latent(latent: torch.Tensor) -> None:
    channels = MINIMAX_MUSIC3_CONFIG.latent_channels
    if latent.ndim != 3 or latent.shape[1] != channels:
        raise MiniMaxMusic3RuntimeError(
            f"MiniMax Music 3 input must have shape [batch,{channels},frames]"
        )


def _minimax_music3_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("MiniMaxMusic3DiffusionRuntime", runtime)
    if context.options:
        names = ", ".join(sorted(context.options))
        raise MiniMaxMusic3RuntimeError(
            f"MiniMax Music 3 sampling does not accept adapter options: {names}"
        )
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            _MiniMaxMusic3SamplingDenoiser(
                owner.assembled.diffusion,
                compute_dtype=compute_dtype,
            ),
        )
    )


def _minimax_music3_device(runtime: object) -> torch.device:
    owner = cast("MiniMaxMusic3DiffusionRuntime", runtime)
    return module_compute_device(owner.assembled.diffusion)


def _minimax_music3_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("MiniMaxMusic3DiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


class MiniMaxMusic3DiffusionRuntime(SingleStreamSamplingRuntime):
    streamed_residency_components = frozenset()
    sampling_error = MiniMaxMusic3RuntimeError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=SingleStreamLatentAdapter(_validate_minimax_music3_latent),
        denoiser=_minimax_music3_denoiser,
        device=_minimax_music3_device,
        compute_dtype=_minimax_music3_compute_dtype,
        flow=True,
        capabilities=CONTEXT_WINDOWS_UNSUPPORTED,
    )

    def __init__(
        self,
        diffusion: MiniMaxMusic3DiT,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        self.assembled = _MiniMaxMusic3DiffusionAssembly(diffusion, compute=compute_dtype)
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _scheduler_registry(scheduler_registry)
        self._guidance: GuidanceExecutor | None = None

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return _sigma_space()

    sample_custom = sampling_execution

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        del text
        raise MiniMaxMusic3RuntimeError(
            "MiniMax Music 3 diffusion component carries no text encoder"
        )

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        raise MiniMaxMusic3RuntimeError("MiniMax Music 3 diffusion component carries no DAV codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        del content
        raise MiniMaxMusic3RuntimeError("MiniMax Music 3 diffusion component carries no DAV codec")


__all__ = [
    "MiniMaxMusic3Denoiser",
    "MiniMaxMusic3DiffusionRuntime",
    "MiniMaxMusic3RuntimeError",
    "MiniMaxMusic3TextRuntime",
]
