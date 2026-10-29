"""Shared sampling surfaces parameterized by the runtime's sigma space."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, Self, cast

import torch
from dinkster_inference import (
    Conditioning,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    FlowSigmas,
    FluxFlowSigmas,
    InpaintConditioning,
    LatentDescriptor,
    ModelFamily,
    MultiStreamLatent,
    MultiStreamLatentDescriptor,
    PreparedMultiStreamConditioning,
    Registry,
    SamplerDescriptor,
    SamplingGuidance,
    SamplingSegment,
    SamplingStateCallback,
    SchedulerDescriptor,
    SigmaSpace,
    SparseLatent,
    StepCallback,
    is_flow_parameterization,
    sampling_sigmas,
)

from . import distributed
from .denoise import (
    FLUX_GUIDANCE_DISABLED,
    FluxGuidance,
    latent_process_in,
    latent_process_out,
    prepare_multistream_noise,
)
from .guidance import GuidanceExecutor, ReplicaEvaluator
from .parameterizations import noise_scaling
from .sampling_execution import (
    CustomSamplingCapabilities,
    CustomSamplingCfgValue,
    SamplingExecutionRegistration,
    SamplingGuidancePlan,
    compile_guidance_plan,
    resolve_custom_sampling_request,
    run_ksampler_as_custom,
)
from .schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    scheduler_on_device,
    sd_turbo_sigmas,
)


@dataclass(frozen=True)
class DistributedGuidanceAdmission:
    """A shared guidance request and plan."""

    request: CustomSamplingRequest[torch.Tensor]
    plan: SamplingGuidancePlan

    def replica_evaluator_factory(self, evaluate: ReplicaEvaluator) -> ReplicaEvaluator:
        return distributed.DistributedGuidanceEvaluator(evaluate).evaluate_request


class SamplingRuntime(ABC):
    """Family runtimes supply their space, not copies of schedule construction."""

    _samplers: Registry[SamplerDescriptor[Any]]
    _schedulers: Registry[SchedulerDescriptor]
    sampling_execution_registration: SamplingExecutionRegistration
    sampling_error: ClassVar[type[Exception]] = ValueError
    supports_sampling_shift: ClassVar[bool] = False
    text_encode_options: ClassVar[frozenset[str]] = frozenset()

    def sampling_runtime(self) -> SamplingRuntime:
        return self

    @property
    @abstractmethod
    def family(self) -> ModelFamily:
        raise NotImplementedError

    @property
    def dense_custom_sampling_role(self) -> str | None:
        return None

    @property
    def receipt_identity(self) -> str | None:
        return None

    def admit_distributed_guidance(
        self,
        request: CustomSamplingRequest[torch.Tensor],
        *,
        cond: object,
        cfg: CustomSamplingCfgValue,
        executor: GuidanceExecutor | None,
    ) -> DistributedGuidanceAdmission | None:
        config = distributed.distributed_sampling_config()
        if config is None:
            return None
        plan = compile_guidance_plan(cond, cfg, request.sampler, executor)
        distributed.ensure_process_group()
        return DistributedGuidanceAdmission(request, plan)

    @property
    def supports_distilled_guidance(self) -> bool:
        return False

    @property
    def supports_inpaint(self) -> bool:
        return False

    @property
    def supports_context_windows(self) -> bool:
        return self.family.engine.supports_context_windows

    def _validate_sampling_guidance(self, guidance: FluxGuidance) -> None:
        if guidance is None or (type(guidance) is str and guidance == FLUX_GUIDANCE_DISABLED):
            return
        if type(guidance) is not float or not math.isfinite(guidance):
            raise self.sampling_error("guidance must be None, 'disabled', or a finite float")
        if not self.supports_distilled_guidance:
            raise self.sampling_error("model has no distilled-guidance input")

    def check_custom_sampling(
        self,
        request: CustomSamplingRequest[torch.Tensor],
        *,
        has_denoise_mask: bool,
        has_inpaint: bool,
        has_context_windows: bool,
        guidance: FluxGuidance = None,
    ) -> None:
        self._validate_sampling_guidance(guidance)
        capabilities: CustomSamplingCapabilities = self.sampling_execution_registration.capabilities
        if has_inpaint and not capabilities.supports_inpaint(self):
            raise self.sampling_error("model does not support inpaint conditioning")
        if has_context_windows and not capabilities.supports_context_windows(self):
            raise self.sampling_error("model does not support context windows")
        sampler, _request = resolve_custom_sampling_request(
            self._samplers, request, error=self.sampling_error
        )
        for restriction in capabilities.restrictions:
            if not restriction.when(self):
                continue
            refused = (
                restriction.required_sampler_id is not None
                and sampler.id != restriction.required_sampler_id
                or restriction.forbidden_sampler_id == sampler.id
                or restriction.refuses_denoise_mask
                and has_denoise_mask
                or restriction.refuses_inpaint
                and has_inpaint
                or restriction.refuses_context_windows
                and has_context_windows
                or restriction.refuses_guidance
                and guidance is not None
            )
            if refused:
                raise self.sampling_error(restriction.message)

    @abstractmethod
    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        """Resolve the family's model-sampling space."""
        raise NotImplementedError

    def sampling_sigma_space(self, sampling_shift: float | None = None) -> SigmaSpace:
        if sampling_shift is not None and not self.supports_sampling_shift:
            raise self.sampling_error("runtime does not support a sampling shift")
        return self._sampling_sigma_space(sampling_shift)

    def _sampling_percent_to_sigma(self, space: SigmaSpace, percent: float) -> float:
        return space.percent_to_sigma(percent)

    def custom_sampling_sigmas(
        self,
        scheduler_id: str,
        steps: int,
        denoise: float,
        *,
        sampling_shift: float | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[float, ...]:
        scheduler = self._schedulers.get(scheduler_id)
        if scheduler is None:
            registered = ", ".join(self._schedulers.ids())
            raise self.sampling_error(
                f"unknown scheduler '{scheduler_id}' (registered: {registered})"
            )
        return sampling_sigmas(
            scheduler_on_device(scheduler, device),
            self.sampling_sigma_space(sampling_shift),
            steps,
            denoise=denoise,
        )

    def custom_sampling_beta_sigmas(
        self,
        steps: int,
        alpha: float,
        beta: float,
        *,
        sampling_shift: float | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[float, ...]:
        return custom_beta_sigmas(
            self.sampling_sigma_space(sampling_shift), steps, alpha, beta, device=device
        )

    def custom_sampling_sd_turbo_sigmas(
        self,
        steps: int,
        denoise: float,
        *,
        sampling_shift: float | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[float, ...]:
        return sd_turbo_sigmas(
            self.sampling_sigma_space(sampling_shift), steps, denoise, device=device
        )

    def custom_sampling_percent_to_sigma(
        self,
        percent: float,
        *,
        return_actual_sigma: bool,
        sampling_shift: float | None = None,
    ) -> float:
        space = self.sampling_sigma_space(sampling_shift)
        return custom_percent_to_sigma(
            space,
            lambda value: self._sampling_percent_to_sigma(space, value),
            percent,
            return_actual_sigma=return_actual_sigma,
        )

    def _custom_sampling_latent_descriptor(self) -> LatentDescriptor:
        latent = self.family.latent
        if type(latent) is not MultiStreamLatentDescriptor:
            return self.family.single_stream_latent()
        for stream_role, descriptor in latent.streams:
            if stream_role == self.dense_custom_sampling_role:
                return descriptor
        raise self.sampling_error("AddNoise requires a dense model latent role")

    def _custom_sampling_process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent_process_in(latent, self._custom_sampling_latent_descriptor())

    def _custom_sampling_process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent_process_out(latent, self._custom_sampling_latent_descriptor())

    def custom_sampling_add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
        *,
        sampling_shift: float | None = None,
    ) -> torch.Tensor:
        self.sampling_sigma_space(sampling_shift)
        normalized = (
            latent
            if not bool(torch.count_nonzero(latent))
            else self._custom_sampling_process_in(latent)
        )
        noisy = noise_scaling(
            self.family.sampling.parameterization,
            sigma,
            noise,
            normalized,
        )
        return torch.nan_to_num(
            self._custom_sampling_process_out(noisy), nan=0.0, posinf=0.0, neginf=0.0
        )


class SingleStreamSamplingRuntime(SamplingRuntime):
    """KSampler composition for runtimes whose custom seam accepts dense tensors."""

    _samplers: Registry[SamplerDescriptor[Any]]
    sampling_compute_dtype: ClassVar[torch.dtype | None] = None
    supports_denoised_capture: ClassVar[bool] = False

    @property
    @abstractmethod
    def family(self) -> ModelFamily:
        raise NotImplementedError

    def sample(
        self,
        latent: torch.Tensor,
        *,
        cond: Conditioning[torch.Tensor],
        cfg: SamplingGuidance[Conditioning[torch.Tensor]] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | Literal["disabled"] | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        sampling_shift: float | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        extra = dict(kwargs)
        schedule_device = cast("torch.device | str | None", extra.pop("schedule_device", None))
        if self.supports_sampling_shift:
            extra["sampling_shift"] = sampling_shift
        result = run_ksampler_as_custom(
            cast("CustomSamplingRuntime[torch.Tensor]", self),
            latent,
            samplers=self._samplers,
            schedulers=self._schedulers,
            space=self.sampling_sigma_space(sampling_shift),
            flow=is_flow_parameterization(self.family.sampling.parameterization),
            device=schedule_device,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            cond=cond,
            cfg=cfg,
            guidance=guidance,
            segment=segment,
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            noise_inds=noise_inds,
            context_windows=cast(
                "ContextWindowsSpec | None", extra.pop("context_windows", context_windows)
            ),
            on_step=on_step,
            on_state=on_state,
            sample_custom_kwargs=extra,
            error=self.sampling_error,
        )
        if type(result.output) is not torch.Tensor:
            raise self.sampling_error("single-stream custom sampling returned a structural latent")
        return result.output


class DenseOrSparseSamplingRuntime(SamplingRuntime):
    """KSampler composition for runtimes that execute dense and sparse stages."""

    _samplers: Registry[SamplerDescriptor[Any]]
    custom_sampling_only: ClassVar[bool] = True

    @property
    @abstractmethod
    def family(self) -> ModelFamily:
        raise NotImplementedError

    def sample(
        self,
        latent: torch.Tensor | SparseLatent[torch.Tensor],
        *,
        cond: object,
        cfg: CustomSamplingCfgValue = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | Literal["disabled"] | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | SparseLatent[torch.Tensor] | None = None,
        inpaint: InpaintConditioning[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        sampling_shift: float | None = None,
        **kwargs: object,
    ) -> torch.Tensor | SparseLatent[torch.Tensor]:
        extra = dict(kwargs)
        schedule_device = cast("torch.device | str | None", extra.pop("schedule_device", None))
        extra.pop("device", None)
        extra.pop("compute_dtype", None)
        extra.setdefault("capture_denoised", False)
        result = run_ksampler_as_custom(
            cast("CustomSamplingRuntime[torch.Tensor]", self),
            latent,
            samplers=self._samplers,
            schedulers=self._schedulers,
            space=self.sampling_sigma_space(sampling_shift),
            flow=is_flow_parameterization(self.family.sampling.parameterization),
            device=schedule_device,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            cond=cast("Any", cond),
            cfg=cfg,
            guidance=guidance,
            segment=segment,
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            noise_inds=noise_inds,
            context_windows=context_windows,
            on_step=on_step,
            on_state=on_state,
            sample_custom_kwargs=extra,
            error=self.sampling_error,
        )
        if type(latent) is torch.Tensor and type(result.output) is not torch.Tensor:
            raise self.sampling_error("dense custom sampling returned a structural latent")
        if type(latent) is SparseLatent and type(result.output) is not SparseLatent:
            raise self.sampling_error("sparse custom sampling returned a non-sparse latent")
        if (
            type(latent) is SparseLatent
            and type(result.output) is SparseLatent
            and not result.output.support.same_support(latent.support)
        ):
            raise self.sampling_error("sparse custom sampling changed the latent support")
        return cast("torch.Tensor | SparseLatent[torch.Tensor]", result.output)


class FlowSamplingRuntime(SingleStreamSamplingRuntime):
    """Explicit space overrides for denoisers that consume sigma as their timestep."""

    _sampling_space_override: FlowSigmas | FluxFlowSigmas | None = None

    def with_sampling_space(self, space: SigmaSpace) -> Self:
        if not isinstance(space, (FlowSigmas, FluxFlowSigmas)):
            raise self.sampling_error("sampling override requires a flow sigma space")
        if isinstance(space, FlowSigmas) and space.multiplier != 1.0:
            raise self.sampling_error("sampling override requires a unit timestep multiplier")
        if not math.isfinite(space.shift):
            raise self.sampling_error("sampling override requires a finite shift")
        try:
            table = space.table
        except (OverflowError, ZeroDivisionError) as error:
            raise self.sampling_error("sampling override produces invalid flow sigmas") from error
        if table is None or not all(math.isfinite(sigma) and 0 < sigma <= 1 for sigma in table):
            raise self.sampling_error("sampling override produces invalid flow sigmas")
        derived = copy(self)
        derived._sampling_space_override = space
        return derived


class MultiStreamSamplingRuntime(SamplingRuntime):
    """KSampler composition over prepared conditioning and ordered latent streams."""

    _samplers: Registry[SamplerDescriptor[Any]]
    supports_denoised_capture: ClassVar[bool] = False
    supports_batch_noise_indices: ClassVar[bool] = True

    @property
    @abstractmethod
    def family(self) -> ModelFamily:
        raise NotImplementedError

    @property
    @abstractmethod
    def conditioning_identity(self) -> str:
        raise NotImplementedError

    def _ksampler_noise(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        seed: int,
        noise_inds: Sequence[int] | None,
        add_noise: bool,
    ) -> MultiStreamLatent[torch.Tensor]:
        if add_noise:
            templates = latent.map(
                lambda stream: torch.empty(
                    stream.size(), dtype=torch.float32, layout=stream.layout, device="cpu"
                )
            )
            return prepare_multistream_noise(templates, seed, noise_inds)
        return latent.map(
            lambda stream: torch.zeros_like(stream, dtype=torch.float32, device="cpu")
        )

    def _ksampler_schedulers(self, scheduler_id: str) -> Registry[SchedulerDescriptor]:
        return self._schedulers

    def sample_multistream(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        *,
        conditioning: object,
        cfg: SamplingGuidance[object] | DualSamplingGuidance[object] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float,
        seed: int,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | MultiStreamLatent[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        sampling_shift: float | None = None,
        **kwargs: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        return self.run_ksampler_as_custom(
            latent,
            conditioning=conditioning,
            cfg=cfg,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            segment=segment,
            denoise_mask=denoise_mask,
            noise_inds=noise_inds,
            context_windows=context_windows,
            on_step=on_step,
            on_state=on_state,
            sampling_shift=sampling_shift,
            **kwargs,
        )

    def run_ksampler_as_custom(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        *,
        conditioning: object,
        cfg: SamplingGuidance[object] | DualSamplingGuidance[object] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float,
        seed: int,
        segment: SamplingSegment | None = None,
        denoise_mask: torch.Tensor | MultiStreamLatent[torch.Tensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
        sampling_shift: float | None = None,
        **kwargs: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        extra = dict(kwargs)
        schedule_device = cast("torch.device | str | None", extra.pop("schedule_device", None))
        if type(latent) is not MultiStreamLatent:
            raise TypeError("latent must be an exact MultiStreamLatent")
        if noise_inds is not None and not self.supports_batch_noise_indices:
            raise self.sampling_error(
                f"{self.family.display_name} does not accept per-batch noise indices"
            )
        identity = self.conditioning_identity

        def wrap_lane(value: object | None) -> PreparedMultiStreamConditioning | None:
            return None if value is None else PreparedMultiStreamConditioning(identity, value)

        prepared_cfg = cfg
        if isinstance(cfg, DualSamplingGuidance):
            prepared_cfg = replace(cfg, uncond=wrap_lane(cfg.uncond), middle=wrap_lane(cfg.middle))
        elif cfg is not None:
            prepared_cfg = replace(cfg, uncond=wrap_lane(cfg.uncond))
        if self.supports_sampling_shift:
            extra["sampling_shift"] = sampling_shift
        result = run_ksampler_as_custom(
            cast("CustomSamplingRuntime[torch.Tensor]", self),
            latent,
            samplers=self._samplers,
            schedulers=self._ksampler_schedulers(scheduler_id),
            space=self.sampling_sigma_space(sampling_shift),
            flow=is_flow_parameterization(self.family.sampling.parameterization),
            device=schedule_device,
            sampler_id=sampler_id,
            scheduler_id=scheduler_id,
            steps=steps,
            denoise=denoise,
            seed=seed,
            cond=PreparedMultiStreamConditioning(identity, conditioning),
            cfg=cast("CustomSamplingCfgValue", prepared_cfg),
            segment=segment,
            denoise_mask=denoise_mask,
            noise_inds=noise_inds,
            context_windows=context_windows,
            on_step=on_step,
            on_state=on_state,
            sample_custom_kwargs=extra,
            noise_factory=lambda add_noise: self._ksampler_noise(
                latent, seed, noise_inds, add_noise
            ),
            error=self.sampling_error,
        )
        if type(result.output) is not MultiStreamLatent:
            raise self.sampling_error("multi-stream custom sampling must return MultiStreamLatent")
        return result.output
