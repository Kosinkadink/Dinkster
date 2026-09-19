"""Native text encoding and flow denoising for split Krea 2 components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from dinkster_inference import (
    KREA2,
    KREA2_SIGMAS,
    Conditioning,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    GuidanceRole,
    InpaintConditioning,
    ModelFamily,
    Registry,
    SamplerDescriptor,
    SamplingStateCallback,
    SchedulerDescriptor,
    SigmaSpace,
    StepCallback,
    sampling_execution_context,
)

if TYPE_CHECKING:
    from .checkpoint_runtime import ComponentAssembly

from .brownian import BrownianTreeNoise
from .denoise import run_denoise
from .guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
    GuidanceExecutor,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .krea2_conditioner import Krea2TextEncoder
from .krea2_dit import Krea2DiT
from .operations import module_compute_device
from .qwen_image_text import QwenImageLanguageModel
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
from .sampling_runtime import SingleStreamSamplingRuntime
from .schedules import (
    torch_scheduler_registry,
)
from .solvers import torch_sampler_registry


class Krea2RuntimeError(ValueError):
    """A Krea 2 runtime request violates its native contract."""


class _Krea2Denoiser:
    """Single-conditioning FLOW evaluator over the native Krea 2 DiT."""

    def __init__(
        self,
        model: Krea2DiT,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def prepare_conditioning(self, value: object) -> torch.Tensor:
        if not isinstance(value, Conditioning):
            raise Krea2RuntimeError("Krea 2 conditioning must be a Conditioning value")
        if value.pooled is not None:
            raise Krea2RuntimeError("Krea 2 conditioning does not accept a pooled vector")
        context = value.embeddings
        if context.ndim != 3 or context.shape[0] < 1 or context.shape[2] != 30720:
            raise Krea2RuntimeError("Krea 2 context must have shape [batch,tokens,30720]")
        return context

    @staticmethod
    def batchable(conditions: tuple[torch.Tensor, ...]) -> bool:
        return bool(conditions) and all(
            condition.shape[1:] == conditions[0].shape[1:] for condition in conditions[1:]
        )

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[torch.Tensor, ...],
    ) -> None:
        if not self.batchable(conditions):
            raise Krea2RuntimeError("Krea 2 conditioning batch is empty or incompatible")
        batch = x.shape[0]
        if any(condition.shape[0] not in (1, batch) for condition in conditions):
            raise Krea2RuntimeError("Krea 2 conditioning batch must be one or match the latent")

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[torch.Tensor],
    ) -> torch.Tensor:
        context = torch.cat(
            [
                condition.to(device=batch.latent.device, dtype=self.compute_dtype).expand(
                    batch.batch_size, -1, -1
                )
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


class Krea2TextRuntime:
    """Text-only Krea 2 encoder over an independently resident component."""

    def __init__(self, text: QwenImageLanguageModel) -> None:
        self._encoder = Krea2TextEncoder(text)
        self.text = text

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        return self._encoder.encode(text)


def checkpoint_text_runtime(assembled: ComponentAssembly) -> Krea2TextRuntime | None:
    text = assembled.components.get("qwen3vl_4b")
    return None if text is None else Krea2TextRuntime(cast("QwenImageLanguageModel", text))


@dataclass(frozen=True)
class _Krea2DiffusionAssembly:
    diffusion: Krea2DiT
    family: ModelFamily = KREA2
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


class Krea2DiffusionRuntime(SingleStreamSamplingRuntime):
    """Diffusion-only Krea 2 sampling over independently encoded conditioning."""

    streamed_residency_components = frozenset()
    sampling_error = Krea2RuntimeError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True

    def __init__(
        self,
        diffusion: Krea2DiT,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
    ) -> None:
        self.assembled = _Krea2DiffusionAssembly(diffusion, compute=compute_dtype)
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

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return KREA2_SIGMAS

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
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        capture_denoised: bool = True,
    ) -> CustomSamplingResult[torch.Tensor]:
        latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
            self.family.id,
            latent=latent,
            noise=noise,
            cond=cond,
            cfg=cfg,
            denoise_mask=denoise_mask,
            error=Krea2RuntimeError,
        )
        channels = self.family.single_stream_latent().channels
        if latent.ndim != 5 or latent.shape[1] != channels or latent.shape[2] != 1:
            raise Krea2RuntimeError(
                f"Krea 2 latent must have shape [batch,{channels},1,height,width]"
            )
        self.check_custom_sampling(
            request,
            has_denoise_mask=denoise_mask is not None,
            has_inpaint=inpaint is not None,
            has_context_windows=context_windows is not None,
            guidance=guidance,
        )
        sampler, request = resolve_custom_sampling_request(
            self._samplers, request, error=Krea2RuntimeError
        )
        if compute_dtype is None:
            compute_dtype = self.assembled.compute_dtype("diffusion") or torch.bfloat16
        if device is None:
            device = module_compute_device(self.assembled.diffusion)
        space = KREA2_SIGMAS
        schedule = build_custom_sampling_schedule(request.sigmas, space, sampler, flow=True)
        noise_sampler: BrownianTreeNoise | None = brownian_step_noise(
            sampler, schedule, latent, seed=seed, device=device
        )
        plan = compile_guidance_plan(cond, cfg, sampler, self._guidance)
        evaluator = _Krea2Denoiser(self.assembled.diffusion, compute_dtype=compute_dtype)

        def prepare(value: object, _role: GuidanceRole) -> torch.Tensor:
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
                evaluator_identity=lambda _role: "dinkster.krea2.conditioning.v1",
                standard_activation_memory_factor=self.family.memory_factor,
            ),
            input=latent,
            executor=self._guidance,
            plan=plan,
            execution=sampling_execution_context(
                sigmas=schedule.sigmas, seed=seed, on_step=on_step, on_state=report_state
            ),
        )
        output = run_denoise(
            denoiser,
            request.build_solver(),
            latent=latent,
            noise=noise,
            sigmas=schedule.sigmas,
            initial_sigma=schedule.initial_sigma,
            family=self.family,
            seed=seed,
            noise_kind=sampler.noise,
            noise_sampler=noise_sampler,
            percent_to_sigma=space.percent_to_sigma,
            device=device,
            on_step=on_step,
            on_state=report_state,
            denoise_mask=denoise_mask,
        )
        return CustomSamplingResult(output, captured[-1] if captured else None)

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        raise Krea2RuntimeError("Krea 2 diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise Krea2RuntimeError("Krea 2 diffusion component carries no VAE codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        raise Krea2RuntimeError("Krea 2 diffusion component carries no VAE codec")


__all__ = [
    "Krea2DiffusionRuntime",
    "Krea2RuntimeError",
    "Krea2TextRuntime",
]
