"""Native text encoding, custom sampling, and codecs for Chroma."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    CHROMA,
    CHROMA_RADIANCE,
    T5_XXL_PIXART_PROFILE,
    Conditioning,
    ConditioningCarrier,
    FlowSigmas,
    GuidanceRole,
    ModelFamily,
    PromptTokenizer,
    Registry,
    SamplerDescriptor,
    SchedulerDescriptor,
    load_t5_spm,
)

from .attention import AttentionRole, AttentionStatus
from .chroma import Chroma, ChromaRadiance, ChromaRadianceOptions
from .conditioning_adapters import basic_conditioning_to_carrier, materialize_basic_conditioning
from .denoise import FluxGuidance
from .guidance import (
    ConditioningBatch,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .operations import module_compute_device
from .sampling_execution import (
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
from .t5_text import T5TextEncoder, T5TextModel


class ChromaRuntimeError(ValueError):
    """A Chroma runtime request violates its native contract."""


@dataclass(frozen=True)
class ChromaRadianceOptionWindow:
    """Radiance overrides active over an inclusive sigma window."""

    options: ChromaRadianceOptions
    start_sigma: float = 1.0
    end_sigma: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_sigma <= 1.0 or not 0.0 <= self.end_sigma <= 1.0:
            raise ValueError("Radiance option sigma bounds must each be in [0, 1]")


def _sigma_space(shift: float) -> FlowSigmas:
    if type(shift) is not float or not math.isfinite(shift) or shift <= 0:
        raise ChromaRuntimeError("Chroma sampling shift must be a positive finite float")
    return FlowSigmas(shift=shift, multiplier=1.0)


def _scheduler_registry(
    source: Registry[SchedulerDescriptor] | None,
) -> Registry[SchedulerDescriptor]:
    if source is None:
        return torch_scheduler_registry()
    registry: Registry[SchedulerDescriptor] = Registry()
    for descriptor in source:
        registry.register(descriptor)
    return registry


class ChromaTextRuntime:
    """Text-only PixArt T5 encoder over an independently resident component."""

    def __init__(self, text: T5TextModel) -> None:
        self._encoder = T5TextEncoder(text, profile=T5_XXL_PIXART_PROFILE)
        self._tokenizer = PromptTokenizer(encode_word=load_t5_spm().encode)

    def encode_text(
        self,
        text: str,
        *,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        return self._encoder.encode(
            self._tokenizer.tokenize(text),
            min_padding=min_padding,
            min_length=min_length,
        )

    @staticmethod
    def text_conditioning_carrier(value: Conditioning[torch.Tensor]) -> ConditioningCarrier:
        return basic_conditioning_to_carrier(value)


class ChromaDenoiser:
    """Single-conditioning FLOW evaluator over a Chroma diffusion component."""

    evaluator_identity = "dinkster.chroma.conditioning.v1"

    def __init__(
        self,
        model: Chroma | ChromaRadiance,
        *,
        guidance: float,
        option_windows: tuple[ChromaRadianceOptionWindow, ...] = (),
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.guidance = guidance
        self.option_windows = option_windows
        self.compute_dtype = compute_dtype

    def prepare_conditioning(
        self,
        value: object,
        role: GuidanceRole = GuidanceRole.CONDITIONAL,
        *,
        lane_id: str | None = None,
    ) -> tuple[torch.Tensor, str]:
        if not isinstance(value, Conditioning):
            raise ChromaRuntimeError("Chroma conditioning must be a Conditioning value")
        if value.pooled is not None:
            raise ChromaRuntimeError("Chroma conditioning does not accept a pooled vector")
        context = value.embeddings
        if context.ndim != 3 or context.shape[0] < 1 or context.shape[2] != 4096:
            raise ChromaRuntimeError("Chroma context must have shape [batch,tokens,4096]")
        if lane_id is None:
            lane_id = "negative" if role is GuidanceRole.UNCONDITIONAL else "positive"
        elif lane_id not in ("positive", "negative"):
            raise ChromaRuntimeError("Chroma guidance lane id is not supported")
        return context, lane_id

    @staticmethod
    def batchable(conditions: tuple[tuple[torch.Tensor, str], ...]) -> bool:
        return bool(conditions) and all(
            condition[0].shape[1:] == conditions[0][0].shape[1:] for condition in conditions[1:]
        )

    def _radiance_options(self, sigma: float) -> ChromaRadianceOptions | None:
        for window in self.option_windows:
            if window.end_sigma <= sigma <= window.start_sigma:
                return window.options
        return None

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
            raise ChromaRuntimeError("Chroma conditioning batch is empty or incompatible")
        batch = x.shape[0]
        if any(condition[0].shape[0] not in (1, batch) for condition in conditions):
            raise ChromaRuntimeError("Chroma conditioning batch must be one or match the latent")

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
        guidance = torch.full_like(batch.timestep, self.guidance)
        if isinstance(self.model, ChromaRadiance):
            return self.model(
                batch.model_input,
                batch.timestep,
                context,
                guidance,
                options=self._radiance_options(batch.sigma),
            ).float()
        return self.model(batch.model_input, batch.timestep, context, guidance).float()


@dataclass(frozen=True)
class _ChromaDiffusionAssembly:
    diffusion: Chroma | ChromaRadiance
    family: ModelFamily
    compute: torch.dtype = torch.bfloat16
    attention_status: Mapping[AttentionRole, AttentionStatus] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: dict[str, torch.dtype] = field(
        default_factory=dict, repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


def _validate_chroma_latent(latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise ChromaRuntimeError("Chroma input must have shape [batch,channels,height,width]")


def _chroma_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    owner = cast("ChromaDiffusionRuntime", runtime)
    if context.inputs is None:
        raise AssertionError("Chroma sampling adapter requires resolved execution context")
    channels = owner.family.single_stream_latent().channels
    if context.inputs.latent.shape[1] != channels:
        raise ChromaRuntimeError(f"Chroma input must have shape [batch,{channels},height,width]")
    if context.options:
        names = ", ".join(sorted(context.options))
        raise ChromaRuntimeError(f"Chroma sampling does not accept adapter options: {names}")
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            ChromaDenoiser(
                owner.assembled.diffusion,
                guidance=0.0 if context.guidance is None else cast("float", context.guidance),
                option_windows=owner._option_windows,  # pyright: ignore[reportPrivateUsage]
                compute_dtype=compute_dtype,
            ),
        )
    )


def _chroma_device(runtime: object) -> torch.device:
    owner = cast("ChromaDiffusionRuntime", runtime)
    return module_compute_device(owner.assembled.diffusion)


def _chroma_compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("ChromaDiffusionRuntime", runtime)
    return owner.assembled.compute_dtype("diffusion") or torch.bfloat16


class ChromaDiffusionRuntime(SingleStreamSamplingRuntime):
    """Diffusion-only Chroma custom sampling over independently encoded conditioning."""

    streamed_residency_components = frozenset()
    sampling_error = ChromaRuntimeError
    supports_sampling_shift = True
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=SingleStreamLatentAdapter(_validate_chroma_latent),
        denoiser=_chroma_denoiser,
        device=_chroma_device,
        compute_dtype=_chroma_compute_dtype,
        flow=True,
    )

    def __init__(
        self,
        diffusion: Chroma | ChromaRadiance,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampling_shift: float = 1.0,
        option_windows: tuple[ChromaRadianceOptionWindow, ...] = (),
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
        attention_status: Mapping[AttentionRole, AttentionStatus] | None = None,
    ) -> None:
        family = CHROMA_RADIANCE if isinstance(diffusion, ChromaRadiance) else CHROMA
        _sigma_space(sampling_shift)
        if option_windows and family is not CHROMA_RADIANCE:
            raise ChromaRuntimeError("Radiance options require a Chroma Radiance diffusion model")
        self.assembled = _ChromaDiffusionAssembly(
            diffusion,
            family,
            compute_dtype,
            MappingProxyType(dict(attention_status or {})),
        )
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._sampling_shift = sampling_shift
        self._option_windows = tuple(option_windows)
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = _scheduler_registry(scheduler_registry)
        self._guidance = guidance_executor

    def with_execution_options(
        self,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        sampling_shift: float,
        option_windows: tuple[ChromaRadianceOptionWindow, ...],
        attention_status: Mapping[AttentionRole, AttentionStatus],
    ) -> ChromaDiffusionRuntime:
        return type(self)(
            self.assembled.diffusion,
            runtime_identity=runtime_identity,
            compute_dtype=compute_dtype,
            sampling_shift=sampling_shift,
            option_windows=option_windows,
            sampler_registry=self._samplers,
            scheduler_registry=self._schedulers,
            guidance_executor=self._guidance,
            attention_status=attention_status,
        )

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        return _sigma_space(self._sampling_shift if sampling_shift is None else sampling_shift)

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[torch.Tensor]:
        return materialize_basic_conditioning(
            carrier, device=module_compute_device(self.assembled.diffusion)
        )

    @property
    def supports_distilled_guidance(self) -> bool:
        return True

    def _validate_sampling_guidance(self, guidance: FluxGuidance) -> None:
        super()._validate_sampling_guidance(guidance)
        if guidance is not None and (guidance == "disabled" or not 0.0 <= guidance <= 100.0):
            raise ChromaRuntimeError("Chroma guidance must be in [0, 100]")

    sample_custom = sampling_execution

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise ChromaRuntimeError("Chroma diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise ChromaRuntimeError("Chroma diffusion component carries no codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise ChromaRuntimeError("Chroma diffusion component carries no codec")


__all__ = [
    "ChromaDenoiser",
    "ChromaDiffusionRuntime",
    "ChromaRadianceOptionWindow",
    "ChromaRuntimeError",
    "ChromaTextRuntime",
]
