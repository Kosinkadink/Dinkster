"""Generic sampling composition for family runtimes.

:func:`run_sampler_engine` owns solver execution; this module owns
the composition layer directly above it: sampler/scheduler
resolution, the sigma schedule with its SNR offset, pre-offset
brownian step noise, and the guided-denoiser wrapping.

Guidance always executes through :class:`GuidanceExecutor` /
:class:`GuidedDenoiser`, even with no registered contributions: the
executor's builtin plan and reducer replay the plain classifier-free
guidance math bit-exactly (the cfg==1 optimization included), so one
execution path serves both and model code stays blind to CFG and
condition counts. ``force_uncond`` is computed centrally from the
sampler's CFG++ declaration and the registry's requirements - never
by a family runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, overload

import torch
from dinkster_inference import (
    Conditioning,
    ConditioningBatching,
    ConditioningCarrier,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceContribution,
    GuidanceRole,
    InpaintConditioning,
    ModelFamily,
    MultiStreamLatent,
    NoiseKind,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    Registry,
    SamplerDescriptor,
    SamplingDescriptor,
    SamplingExecutionContext,
    SamplingGuidance,
    SamplingSegment,
    SamplingStateCallback,
    SamplingStateEvent,
    SchedulerDescriptor,
    SigmaSpace,
    SparseLatent,
    StepCallback,
    cfg_needs_uncond,
    offset_first_sigma_for_snr,
    sampling_environment_cancellation,
    sampling_execution_context,
    sampling_sigmas,
    use_sampling_environment,
)

from .brownian import BrownianTreeNoise
from .denoise import latent_process_out, prepare_multistream_noise, prepare_noise, run_denoise
from .guidance import (
    ConditioningEvaluation,
    GuidanceExecutor,
    GuidanceRegistry,
    GuidedDenoiser,
    ReplicaEvaluator,
    ReplicaEvaluatorFactory,
    RoutedConditioning,
    dual_cfg_executor,
    merged_guidance_executor,
    routed_conditioning_evaluation,
)
from .guidance_transforms import perp_neg as guidance_transforms_perp_neg
from .schedules import scheduler_on_device
from .sparse import pack_sparse_latent, unpack_sparse_latent


def resolve_sampling(
    samplers: Registry[SamplerDescriptor[Any]],
    schedulers: Registry[SchedulerDescriptor],
    sampler_id: str,
    scheduler_id: str,
    *,
    error: type[Exception],
) -> tuple[SamplerDescriptor[Any], SchedulerDescriptor]:
    """Resolve both descriptors or raise the family's ``error`` type."""
    sampler = samplers.get(sampler_id)
    if sampler is None:
        raise error(f"unknown sampler '{sampler_id}' (registered: {', '.join(samplers.ids())})")
    scheduler = schedulers.get(scheduler_id)
    if scheduler is None:
        raise error(
            f"unknown scheduler '{scheduler_id}' (registered: {', '.join(schedulers.ids())})"
        )
    return sampler, scheduler


def resolve_custom_sampling_request(
    samplers: Registry[SamplerDescriptor[Any]],
    request: CustomSamplingRequest[torch.Tensor],
    *,
    error: type[Exception],
) -> tuple[SamplerDescriptor[Any], CustomSamplingRequest[torch.Tensor]]:
    """Rebind the request onto the registry's descriptor or raise the
    family's ``error`` type for an unknown sampler."""
    if type(request) is not CustomSamplingRequest:
        raise TypeError("custom sampling requires an exact CustomSamplingRequest")
    sampler = samplers.get(request.sampler.id)
    if sampler is None:
        raise error(
            f"unknown sampler '{request.sampler.id}' (registered: {', '.join(samplers.ids())})"
        )
    return sampler, CustomSamplingRequest(
        sampler,
        request.options,
        request.sigmas,
        cache=request.cache,
        timeline=request.timeline,
    )


def custom_denoised_callback(
    family: ModelFamily,
    callback: SamplingStateCallback | None,
) -> tuple[SamplingStateCallback, list[torch.Tensor]]:
    """A state callback that captures the last denoised latent in the
    family's processed-out space while forwarding events to ``callback``."""
    captured: list[torch.Tensor] = []
    latent_descriptor = family.single_stream_latent()

    def report(event: SamplingStateEvent[object]) -> None:
        if event.denoised is not None:
            if type(event.denoised) is not torch.Tensor:
                raise TypeError("custom sampling denoised state must contain a torch.Tensor")
            captured[:] = [latent_process_out(event.denoised, latent_descriptor)]
        if callback is not None:
            callback(event)

    return report, captured


@dataclass(frozen=True)
class SamplingSchedule:
    """One run's sigmas, before and after the SNR offset.

    ``sigmas`` is what the solver walks. ``pre_offset`` supplies the
    initial-state sigma and brownian-tree bounds: the reference builds
    both before offset_first_sigma_for_snr nudges sigmas[0]
    (sample_dpmpp_sde/_2m_sde/_3m_sde @ b78cec87). The two are
    identical unless the offset actually moved the first sigma.
    """

    pre_offset: tuple[float, ...]
    sigmas: tuple[float, ...]

    @property
    def initial_sigma(self) -> float | None:
        return self.pre_offset[0] if self.pre_offset else None


def slice_sampling_schedule(
    sigmas: tuple[float, ...],
    segment: SamplingSegment | None,
    *,
    steps: int,
    denoise: float | None,
) -> tuple[float, ...]:
    """Apply KSamplerAdvanced's end trim, final-zero policy, then start trim."""
    if segment is None:
        return sigmas
    if segment.steps != steps:
        raise ValueError(
            f"sampling segment was built for {segment.steps} steps, but this run uses {steps}"
        )
    if denoise not in (None, 1.0):
        raise ValueError("sampling segments require denoise=1.0 or None")
    sliced = sigmas
    if segment.end_step < len(sliced) - 1:
        sliced = sliced[: segment.end_step + 1]
        if not segment.return_with_leftover_noise:
            sliced = (*sliced[:-1], 0.0)
    if segment.start_step < len(sliced) - 1:
        return sliced[segment.start_step :]
    return ()


def build_sampling_schedule(
    scheduler: SchedulerDescriptor,
    space: SigmaSpace,
    sampler: SamplerDescriptor[Any],
    steps: int,
    *,
    denoise: float | None,
    flow: bool,
    segment: SamplingSegment | None = None,
    device: torch.device | str | None = None,
) -> SamplingSchedule:
    """The KSampler schedule plus the sampler's declared corrections:
    discard-penultimate at scheduling time, the flow SNR offset after.
    ``offset_first_sigma_for_snr`` is the identity for non-flow spaces,
    schedules already below sigma 1, and single-sigma schedules, so no
    extra guards are needed here."""
    if segment is not None:
        if segment.steps != steps:
            raise ValueError(
                f"sampling segment was built for {segment.steps} steps, but this run uses {steps}"
            )
        if denoise not in (None, 1.0):
            raise ValueError("sampling segments require denoise=1.0 or None")
    pre_offset = sampling_sigmas(
        scheduler_on_device(scheduler, device),
        space,
        steps,
        denoise=denoise,
        discard_penultimate=sampler.discard_penultimate,
    )
    pre_offset = slice_sampling_schedule(
        pre_offset,
        segment,
        steps=steps,
        denoise=denoise,
    )
    sigmas = pre_offset
    if sampler.requires_snr_offset and flow and pre_offset:
        sigmas = offset_first_sigma_for_snr(pre_offset, space, flow=True)
    return SamplingSchedule(pre_offset, sigmas)


def build_custom_sampling_schedule(
    sigmas: tuple[float, ...],
    space: SigmaSpace,
    sampler: SamplerDescriptor[Any],
    *,
    flow: bool,
) -> SamplingSchedule:
    """Normalize exact schedule-space sigmas for solver execution."""
    executed = sigmas
    if sampler.requires_snr_offset and flow and sigmas:
        executed = offset_first_sigma_for_snr(sigmas, space, flow=True)
    return SamplingSchedule(sigmas, executed)


def brownian_step_noise(
    sampler: SamplerDescriptor[Any],
    schedule: SamplingSchedule,
    like: torch.Tensor,
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> BrownianTreeNoise | None:
    """The explicit pre-offset brownian tree, only when it differs from
    the engine's own construction.

    ``run_sampler_engine`` already builds the reference tree from the
    executed sigmas; that is bit-identical to a pre-offset tree unless
    the SNR offset moved sigmas[0] (BrownianTreeNoise reads only
    shape/dtype/device from ``like``, and the bounds - min positive
    sigma, max sigma - are equal when the schedules are equal). So
    None means "let the engine build it", and a tree is returned only
    for moved schedules, where the engine can no longer see the
    pre-offset bounds the reference solvers used.
    """
    if sampler.noise not in (NoiseKind.BROWNIAN, NoiseKind.BROWNIAN_GPU):
        return None
    if schedule.sigmas == schedule.pre_offset:
        return None
    positive = [sigma for sigma in schedule.pre_offset if sigma > 0]
    if not positive:
        return None
    target = like.device if device is None else torch.device(device)
    return BrownianTreeNoise(
        like.to(device=target, dtype=torch.float32),
        min(positive),
        max(schedule.pre_offset),
        seed=seed,
        cpu=sampler.noise is NoiseKind.BROWNIAN,
    )


@dataclass(frozen=True)
class SamplingGuidancePlan:
    """Shared guidance inputs and centrally resolved lane demand.

    ``contributions`` are the per-run guidance extension pairs carried
    by this run's :class:`SamplingGuidance`; the executor merges them
    with the load-time registry at denoiser assembly. ``has_strategy``
    reports whether that merged registry declares a strategy, so
    callers never need to consult the load-time registry directly."""

    conditions: tuple[GuidanceCondition[Any], ...]
    cfg_scale: float
    force_uncond: bool
    contributions: tuple[tuple[str, GuidanceContribution[Any]], ...] = ()
    has_strategy: bool = False
    batching: ConditioningBatching = ConditioningBatching()

    @property
    def needs_unconditional(self) -> bool:
        unconditional = next(
            (
                lane.conditioning
                for lane in self.conditions
                if lane.role is GuidanceRole.UNCONDITIONAL
            ),
            None,
        )
        return unconditional is not None and (self.force_uncond or cfg_needs_uncond(self.cfg_scale))

    def with_conditioning(
        self,
        conditional: object,
        unconditional: object | None,
    ) -> SamplingGuidancePlan:
        if tuple(lane.id for lane in self.conditions) != ("positive", "negative"):
            raise ValueError("with_conditioning only supports two-lane guidance")
        if self.needs_unconditional != (unconditional is not None):
            raise ValueError("materialized unconditional lane does not match the guidance plan")
        return SamplingGuidancePlan(
            (
                GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", conditional)),
                GuidanceCondition(
                    "negative", GuidanceRole.UNCONDITIONAL, cast("Any", unconditional)
                ),
            ),
            self.cfg_scale,
            self.force_uncond,
            self.contributions,
            self.has_strategy,
            self.batching,
        )


def compile_guidance_plan(
    cond: object,
    cfg: SamplingGuidance[Any] | DualSamplingGuidance[Any] | PerpNegSamplingGuidance[Any] | None,
    sampler: SamplerDescriptor[Any],
    executor: GuidanceExecutor | None,
) -> SamplingGuidancePlan:
    """Resolve builtin lane demand without exposing CFG policy to a family."""

    guidance = cfg if cfg is not None else SamplingGuidance()
    if isinstance(guidance, PerpNegSamplingGuidance):
        # "dinkster.perp-neg" is a reserved per-run owner id: the carrier
        # injects its combination strategy through the same contribution
        # merge as user transforms, so a user-attached pair reusing the
        # id (or any conflicting strategy, load-time or per-run) refuses
        # loudly through the registry's duplicate-id and single-strategy
        # rules rather than being silently replaced.
        transforms = guidance.transforms + (
            ("dinkster.perp-neg", guidance_transforms_perp_neg(guidance.neg_scale)),
        )
    elif isinstance(guidance, DualSamplingGuidance):
        transforms = dual_cfg_executor(guidance).registry.contributions + guidance.transforms
    else:
        transforms = guidance.transforms
    batching = guidance.batching
    merged = merged_guidance_executor(executor, transforms)

    active = merged is not None and merged.registry.active
    force_uncond = (
        guidance.disable_cfg1_optimization
        or sampler.needs_uncond
        or bool(active and merged is not None and merged.registry.requires_uncond)
    )
    has_strategy = merged is not None and merged.registry.strategy is not None
    if isinstance(guidance, PerpNegSamplingGuidance):
        return SamplingGuidancePlan(
            (
                GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", cond)),
                GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, guidance.uncond),
                GuidanceCondition("empty", GuidanceRole.AUXILIARY, guidance.empty),
            ),
            guidance.scale,
            force_uncond,
            transforms,
            has_strategy,
            batching,
        )
    if isinstance(guidance, DualSamplingGuidance):
        if not active:
            raise ValueError("dual sampling guidance requires its guidance executor")
        return SamplingGuidancePlan(
            (
                GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", cond)),
                GuidanceCondition(
                    "middle",
                    GuidanceRole.UNCONDITIONAL if guidance.nested else GuidanceRole.AUXILIARY,
                    guidance.middle,
                ),
                GuidanceCondition(
                    "negative",
                    GuidanceRole.AUXILIARY if guidance.nested else GuidanceRole.UNCONDITIONAL,
                    guidance.uncond,
                ),
            ),
            guidance.middle_scale,
            force_uncond or guidance.nested,
            transforms,
            has_strategy,
            batching,
        )
    return SamplingGuidancePlan(
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast("Any", cond)),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, guidance.uncond),
        ),
        guidance.scale,
        force_uncond,
        transforms,
        has_strategy,
        batching,
    )


def guided_denoiser(
    conditioning: ConditioningEvaluation[Any],
    *,
    input: torch.Tensor,
    executor: GuidanceExecutor | None,
    plan: SamplingGuidancePlan,
    execution: SamplingExecutionContext,
    replica_evaluator_factory: ReplicaEvaluatorFactory | None = None,
    distributed_evaluation: bool = False,
) -> GuidedDenoiser:
    """Wrap one family denoiser for guided execution.

    ``conditioning`` declares one prepared-condition model evaluation
    and optional compatible batching; lane fan-out, the cfg==1
    optimization, CFG++ force-uncond, and the combination math are
    decided by ``compile_guidance_plan`` and the executor. A supplied ``executor``
    is first merged with the plan's per-run contributions and
    executes only when the merged registry declares contributions;
    None or an inactive registry selects a fresh builtin executor, whose
    execution is bit-identical to plain classifier-free guidance. An
    executor instance is caller-reachable state (its ``execute`` can
    be shadowed), so an executor whose registry declares nothing must
    never run caller code where plain guidance is meant.
    """
    if any(isinstance(lane.conditioning, RoutedConditioning) for lane in plan.conditions):
        conditioning = routed_conditioning_evaluation(conditioning)
    merged = merged_guidance_executor(executor, plan.contributions)
    resolved = (
        merged
        if merged is not None and merged.registry.active
        else GuidanceExecutor(GuidanceRegistry())
    )
    # Attention-kind contributions execute inside the family's model
    # evaluation, so admission is default-deny: a family opts in by
    # declaring evaluate_batch_attention, and anything else refuses
    # before sampling rather than silently sampling without the rewrite.
    if resolved.registry.attention and conditioning.evaluate_batch_attention is None:
        ids = ", ".join(item.value.id for item in resolved.registry.attention)
        raise GuidanceContractError(
            f"attention-kind guidance contributions ({ids}) are not consumed by"
            " this family's model evaluation; the family has not opted in to"
            " attention-level guidance"
        )
    if replica_evaluator_factory is None and not distributed_evaluation:
        from .distributed import DistributedGuidanceEvaluator, distributed_sampling_config

        if distributed_sampling_config() is not None:

            def distributed_replicas(evaluate: ReplicaEvaluator) -> ReplicaEvaluator:
                return DistributedGuidanceEvaluator(evaluate).evaluate_request

            replica_evaluator_factory = distributed_replicas

    return GuidedDenoiser(
        conditioning,
        resolved,
        plan.conditions,
        cfg_scale=plan.cfg_scale,
        force_uncond=plan.force_uncond,
        input=input,
        batching=plan.batching,
        execution=execution,
        replica_evaluator_factory=replica_evaluator_factory,
    )


CustomSamplingLatentValue = (
    torch.Tensor | MultiStreamLatent[torch.Tensor] | SparseLatent[torch.Tensor]
)
"""The seam's latent and noise shape: dense, structural streams, or sparse rows."""

CustomSamplingCondValue = (
    Conditioning[torch.Tensor] | ConditioningCarrier | PreparedMultiStreamConditioning
)
"""The seam's conditioning shape: direct, canonical wire, or family-opaque prepared."""

CustomSamplingCfgValue = (
    SamplingGuidance[Conditioning[torch.Tensor]]
    | SamplingGuidance[ConditioningCarrier]
    | SamplingGuidance[PreparedMultiStreamConditioning]
    | DualSamplingGuidance[Conditioning[torch.Tensor]]
    | DualSamplingGuidance[PreparedMultiStreamConditioning]
    | PerpNegSamplingGuidance[Conditioning[torch.Tensor]]
    | PerpNegSamplingGuidance[PreparedMultiStreamConditioning]
    | None
)
"""The seam's guidance shape; families refuse the kinds they cannot execute."""

SingleStreamCustomSamplingCfg = (
    SamplingGuidance[Conditioning[torch.Tensor]]
    | DualSamplingGuidance[Conditioning[torch.Tensor]]
    | PerpNegSamplingGuidance[Conditioning[torch.Tensor]]
    | None
)
"""The guidance shapes a perp-neg-admitting single-stream family narrows to."""


@dataclass(frozen=True)
class SamplingExecutionInputs:
    latent: torch.Tensor
    noise: torch.Tensor
    cond: object
    cfg: SamplingGuidance[Any] | DualSamplingGuidance[Any] | PerpNegSamplingGuidance[Any] | None
    denoise_mask: torch.Tensor | None
    latent_context: object | None = None


@dataclass(frozen=True)
class SamplingAdapterContext:
    guidance: float | Literal["disabled"] | None
    inpaint: object | None
    context_windows: ContextWindowsSpec | None
    options: Mapping[str, object]
    inputs: SamplingExecutionInputs | None = None
    sampler: SamplerDescriptor[Any] | None = None
    request: CustomSamplingRequest[torch.Tensor] | None = None
    schedule: SamplingSchedule | None = None
    plan: SamplingGuidancePlan | None = None
    seed: int = 0
    device: torch.device | str | None = None
    compute_dtype: torch.dtype | None = None
    cancelled: Callable[[], bool] = lambda: False


class SamplingLatentAdapter(Protocol):
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
    ) -> SamplingExecutionInputs: ...

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[Any]: ...


@dataclass(frozen=True)
class SingleStreamLatentAdapter:
    validate: Callable[[torch.Tensor], None]
    admit_perp_neg: bool = False

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
        del runtime, context
        if self.admit_perp_neg:
            latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
                family.id,
                latent=latent,
                noise=noise,
                cond=cond,
                cfg=cfg,
                denoise_mask=denoise_mask,
                error=error,
                admit_perp_neg=True,
            )
        else:
            latent, noise, cond, cfg, denoise_mask = narrow_single_stream_custom_sampling(
                family.id,
                latent=latent,
                noise=noise,
                cond=cond,
                cfg=cfg,
                denoise_mask=denoise_mask,
                error=error,
            )
        self.validate(latent)
        return SamplingExecutionInputs(
            latent,
            noise,
            cond,
            cfg,
            denoise_mask,
            family.single_stream_latent(),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[torch.Tensor]:
        if denoised is not None and type(denoised) is not torch.Tensor:
            raise TypeError("single-stream denoised state must contain a torch.Tensor")
        return CustomSamplingResult(
            output,
            None
            if denoised is None
            else latent_process_out(denoised, cast("Any", inputs.latent_context)),
        )


class SamplingDenoiserAdapter(Protocol):
    evaluator_identity: str | Callable[[GuidanceRole], str]
    evaluate_conditioning_batch: Callable[
        [torch.Tensor, float, tuple[object, ...]],
        tuple[torch.Tensor, ...],
    ]

    def prepare_conditioning(self, value: object, role: GuidanceRole) -> object: ...

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: object,
    ) -> torch.Tensor: ...

    def batchable(self, conditions: tuple[object, ...]) -> bool: ...


@dataclass(frozen=True)
class SamplingDenoiserExecution:
    evaluator: SamplingDenoiserAdapter
    conditioning_evaluation: ConditioningEvaluation[Any] | None = None
    conditioning_payloads: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    replica_group_size: int | None = None
    distributed_evaluation: bool = False
    solver_options: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    sampling: SamplingDescriptor | None = None
    percent_to_sigma: Callable[[float], float] | None = None
    on_step_begin: Callable[[int], None] | None = None
    process_in: Callable[[torch.Tensor], torch.Tensor] | None = None
    process_out: Callable[[torch.Tensor], torch.Tensor] | None = None
    inpaint_noise: torch.Tensor | None = None
    unpack_state: Callable[[torch.Tensor], object] | None = None
    denoise_mask_prepared: bool = False
    fixed_inpaint_latent: bool = False
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class SamplingExecutionRegistration:
    latent: SamplingLatentAdapter
    denoiser: Callable[[object, torch.dtype, SamplingAdapterContext], SamplingDenoiserExecution]
    device: Callable[[object], torch.device | str | None]
    compute_dtype: Callable[[object], torch.dtype]
    flow: bool
    guidance_executor: Callable[[object], GuidanceExecutor | None] = lambda runtime: getattr(
        runtime, "_guidance", None
    )
    device_from_inputs: (
        Callable[[object, SamplingExecutionInputs], torch.device | str | None] | None
    ) = None
    prepare_guidance: (
        Callable[[object, SamplingExecutionInputs], SamplingExecutionInputs] | None
    ) = None


class DistributedGuidanceAdmission(Protocol):
    request: CustomSamplingRequest[torch.Tensor]
    plan: SamplingGuidancePlan


class SamplingExecutionRuntime(Protocol):
    family: ModelFamily
    sampling_error: type[Exception]
    supports_denoised_capture: bool
    sampling_execution_registration: SamplingExecutionRegistration
    _samplers: Registry[SamplerDescriptor[Any]]
    _guidance: GuidanceExecutor | None

    def check_custom_sampling(
        self,
        request: CustomSamplingRequest[torch.Tensor],
        *,
        has_denoise_mask: bool,
        has_inpaint: bool,
        has_context_windows: bool,
        guidance: float | Literal["disabled"] | None = None,
    ) -> None: ...

    def sampling_sigma_space(self, sampling_shift: float | None = None) -> SigmaSpace: ...

    def admit_distributed_guidance(
        self,
        request: CustomSamplingRequest[torch.Tensor],
        *,
        cond: object,
        cfg: CustomSamplingCfgValue,
        executor: GuidanceExecutor | None,
    ) -> DistributedGuidanceAdmission | None: ...


@overload
def narrow_single_stream_custom_sampling(
    family_id: str,
    *,
    latent: CustomSamplingLatentValue,
    noise: CustomSamplingLatentValue,
    cond: CustomSamplingCondValue,
    cfg: CustomSamplingCfgValue,
    denoise_mask: CustomSamplingLatentValue | None,
    error: type[Exception],
    admit_perp_neg: Literal[False] = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Conditioning[torch.Tensor],
    SamplingGuidance[Conditioning[torch.Tensor]]
    | DualSamplingGuidance[Conditioning[torch.Tensor]]
    | None,
    torch.Tensor | None,
]: ...


@overload
def narrow_single_stream_custom_sampling(
    family_id: str,
    *,
    latent: CustomSamplingLatentValue,
    noise: CustomSamplingLatentValue,
    cond: CustomSamplingCondValue,
    cfg: CustomSamplingCfgValue,
    denoise_mask: CustomSamplingLatentValue | None,
    error: type[Exception],
    admit_perp_neg: Literal[True],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Conditioning[torch.Tensor],
    SingleStreamCustomSamplingCfg,
    torch.Tensor | None,
]: ...


def narrow_single_stream_custom_sampling(
    family_id: str,
    *,
    latent: CustomSamplingLatentValue,
    noise: CustomSamplingLatentValue,
    cond: CustomSamplingCondValue,
    cfg: CustomSamplingCfgValue,
    denoise_mask: CustomSamplingLatentValue | None,
    error: type[Exception],
    admit_perp_neg: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Conditioning[torch.Tensor],
    SingleStreamCustomSamplingCfg,
    torch.Tensor | None,
]:
    """Admit the single-stream shape of the custom sampling seam.

    A family whose latent representation is one tensor calls this
    first: structural latents, prepared multi-stream conditioning
    payloads, and incompatible guidance refuse with the family's
    ``error`` type. Dual guidance is admitted when its middle and
    unconditional lanes contain ``Conditioning`` payloads. The
    returned values carry the narrowed types.
    Perp-neg guidance refuses unless the family opts in with
    ``admit_perp_neg``."""
    if type(latent) in (MultiStreamLatent, SparseLatent):
        raise error(f"family {family_id} custom sampling requires a single-stream tensor latent")
    if type(noise) in (MultiStreamLatent, SparseLatent):
        raise error(f"family {family_id} custom sampling requires single-stream tensor noise")
    if type(cond) is PreparedMultiStreamConditioning:
        raise error(
            f"family {family_id} custom sampling requires Conditioning,"
            " not a prepared multi-stream payload"
        )
    if type(cond) is ConditioningCarrier:
        raise error(f"family {family_id} custom sampling does not accept a ConditioningCarrier")
    if isinstance(cfg, DualSamplingGuidance):
        if not isinstance(cfg.uncond, Conditioning) or not isinstance(cfg.middle, Conditioning):
            raise error(
                f"family {family_id} custom sampling guidance requires a Conditioning payload"
            )
    elif isinstance(cfg, PerpNegSamplingGuidance):
        if not admit_perp_neg:
            raise error(
                f"family {family_id} does not support PerpNegSamplingGuidance"
                " (perp-neg guidance); pass SamplingGuidance"
            )
        if not isinstance(cfg.uncond, Conditioning) or not isinstance(cfg.empty, Conditioning):
            raise error(
                f"family {family_id} custom sampling guidance requires a Conditioning payload"
            )
    elif cfg is not None and cfg.uncond is not None and not isinstance(cfg.uncond, Conditioning):
        raise error(f"family {family_id} custom sampling guidance requires a Conditioning payload")
    if type(denoise_mask) in (MultiStreamLatent, SparseLatent):
        raise error(f"family {family_id} custom sampling requires a single-stream denoise mask")
    return (
        cast("torch.Tensor", latent),
        cast("torch.Tensor", noise),
        cast("Conditioning[torch.Tensor]", cond),
        cast("SingleStreamCustomSamplingCfg", cfg),
        cast("torch.Tensor | None", denoise_mask),
    )


def sampling_execution(
    runtime: object,
    latent: CustomSamplingLatentValue,
    *,
    noise: CustomSamplingLatentValue,
    cond: CustomSamplingCondValue,
    cfg: CustomSamplingCfgValue = None,
    request: CustomSamplingRequest[torch.Tensor],
    seed: int = 0,
    guidance: float | Literal["disabled"] | None = None,
    denoise_mask: CustomSamplingLatentValue | None = None,
    inpaint: object | None = None,
    context_windows: ContextWindowsSpec | None = None,
    on_step: StepCallback | None = None,
    on_state: SamplingStateCallback | None = None,
    sampling_shift: float | None = None,
    compute_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    capture_denoised: bool = True,
    cancelled: Callable[[], bool] | None = None,
    observer: object | None = None,
    parent_span_id: int | None = None,
    **adapter_options: object,
) -> CustomSamplingResult[Any]:
    """Execute one custom-sampling request from registered family adapters."""

    del observer, parent_span_id
    if cancelled is None:
        cancelled = sampling_environment_cancellation()
    owner = cast("SamplingExecutionRuntime", runtime)
    registration = owner.sampling_execution_registration
    adapter_context = SamplingAdapterContext(
        guidance,
        inpaint,
        context_windows,
        MappingProxyType(dict(adapter_options)),
        seed=seed,
        cancelled=cancelled,
    )
    inputs = registration.latent.prepare(
        owner,
        owner.family,
        latent=latent,
        noise=noise,
        cond=cond,
        cfg=cfg,
        denoise_mask=denoise_mask,
        context=adapter_context,
        error=owner.sampling_error,
    )
    owner.check_custom_sampling(
        request,
        has_denoise_mask=inputs.denoise_mask is not None,
        has_inpaint=inpaint is not None,
        has_context_windows=context_windows is not None,
        guidance=guidance,
    )
    sampler, request = resolve_custom_sampling_request(
        owner._samplers,  # pyright: ignore[reportPrivateUsage]
        request,
        error=owner.sampling_error,
    )
    executor = registration.guidance_executor(owner)
    admission = owner.admit_distributed_guidance(
        request,
        cond=inputs.cond,
        cfg=cast("CustomSamplingCfgValue", inputs.cfg),
        executor=executor,
    )
    admitted_plan = None
    if admission is not None:
        request = admission.request
        sampler = request.sampler
        admitted_plan = admission.plan
    space = owner.sampling_sigma_space(sampling_shift)
    schedule = build_custom_sampling_schedule(
        request.sigmas,
        space,
        sampler,
        flow=registration.flow,
    )
    if sampler.noise in (NoiseKind.BROWNIAN, NoiseKind.BROWNIAN_GPU) and not any(
        value > 0.0 for value in schedule.pre_offset
    ):
        raise owner.sampling_error(
            f"{owner.family.display_name} brownian sampler needs positive sigmas"
        )
    if compute_dtype is None:
        compute_dtype = registration.compute_dtype(owner)
    if device is None:
        device = (
            registration.device(owner)
            if registration.device_from_inputs is None
            else registration.device_from_inputs(owner, inputs)
        )
    noise_sampler = brownian_step_noise(
        sampler,
        schedule,
        inputs.latent,
        seed=seed,
        device=device,
    )
    if registration.prepare_guidance is not None:
        inputs = registration.prepare_guidance(owner, inputs)
    plan = (
        compile_guidance_plan(inputs.cond, inputs.cfg, sampler, executor)
        if admitted_plan is None
        else admitted_plan
    )
    adapter_context = replace(
        adapter_context,
        inputs=inputs,
        sampler=sampler,
        request=request,
        schedule=schedule,
        plan=plan,
        seed=seed,
        device=device,
        compute_dtype=compute_dtype,
        cancelled=cancelled,
    )
    denoiser_execution = registration.denoiser(owner, compute_dtype, adapter_context)
    if denoiser_execution.conditioning_payloads:
        known_lanes = {condition.id for condition in plan.conditions}
        unknown_lanes = set(denoiser_execution.conditioning_payloads) - known_lanes
        if unknown_lanes:
            raise owner.sampling_error(
                "conditioning adapter replaced unknown lanes: " + ", ".join(sorted(unknown_lanes))
            )
        plan = replace(
            plan,
            conditions=tuple(
                replace(
                    condition,
                    conditioning=cast(
                        "Any",
                        denoiser_execution.conditioning_payloads.get(
                            condition.id, condition.conditioning
                        ),
                    ),
                )
                for condition in plan.conditions
            ),
        )
    adapter = denoiser_execution.evaluator
    evaluation = denoiser_execution.conditioning_evaluation
    if evaluation is None:
        evaluator_identity = adapter.evaluator_identity
        resolved_evaluator_identity: Callable[[GuidanceRole], str]
        if isinstance(evaluator_identity, str):
            identity = evaluator_identity

            def resolved_evaluator_identity(_role: GuidanceRole, /) -> str:
                return identity

        else:
            resolved_evaluator_identity = evaluator_identity
        evaluation = ConditioningEvaluation(
            adapter.prepare_conditioning,
            adapter.evaluate_conditioning,
            adapter.batchable,
            adapter.evaluate_conditioning_batch,
            evaluator_identity=resolved_evaluator_identity,
            standard_activation_memory_factor=owner.family.memory_factor,
        )
    captured: list[object] = []

    def report_state(event: SamplingStateEvent[object]) -> None:
        if capture_denoised and owner.supports_denoised_capture and event.denoised is not None:
            captured[:] = [event.denoised]
        if on_state is not None:
            on_state(event)

    def report_step(event: Any) -> None:
        if cancelled():
            from dinkster_inference import SamplingCancelled

            raise SamplingCancelled("sampling cancelled")
        if on_step is not None:
            on_step(event)
        if cancelled():
            from dinkster_inference import SamplingCancelled

            raise SamplingCancelled("sampling cancelled")

    replica_evaluator_factory = None
    if denoiser_execution.replica_group_size is not None:
        from .distributed import DistributedGuidanceEvaluator

        group_size = denoiser_execution.replica_group_size

        def distributed_replicas(evaluate: ReplicaEvaluator) -> ReplicaEvaluator:
            return DistributedGuidanceEvaluator(
                evaluate, guidance_group_size=group_size
            ).evaluate_request

        replica_evaluator_factory = distributed_replicas
    denoiser = guided_denoiser(
        evaluation,
        input=inputs.latent,
        executor=executor,
        plan=plan,
        execution=sampling_execution_context(
            sigmas=schedule.sigmas,
            seed=seed,
            on_step=report_step if on_step is not None else None,
            on_state=report_state,
        ),
        replica_evaluator_factory=replica_evaluator_factory,
        distributed_evaluation=denoiser_execution.distributed_evaluation,
    )
    primary: BaseException | None = None
    try:
        with use_sampling_environment((), cancelled):
            output = run_denoise(
                denoiser,
                request.build_solver(**cast("Any", denoiser_execution.solver_options)),
                latent=inputs.latent,
                noise=inputs.noise,
                sigmas=schedule.sigmas,
                initial_sigma=schedule.initial_sigma,
                family=owner.family,
                sampling=denoiser_execution.sampling,
                process_in=denoiser_execution.process_in,
                process_out=denoiser_execution.process_out,
                inpaint_noise=denoiser_execution.inpaint_noise,
                seed=seed,
                noise_kind=sampler.noise,
                noise_sampler=noise_sampler,
                percent_to_sigma=(
                    space.percent_to_sigma
                    if denoiser_execution.percent_to_sigma is None
                    else denoiser_execution.percent_to_sigma
                ),
                device=device,
                on_step=report_step if on_step is not None else None,
                on_step_begin=denoiser_execution.on_step_begin,
                on_state=(
                    report_state
                    if (capture_denoised and owner.supports_denoised_capture)
                    or on_state is not None
                    else None
                ),
                unpack_state=denoiser_execution.unpack_state,
                denoise_mask=inputs.denoise_mask,
                denoise_mask_prepared=denoiser_execution.denoise_mask_prepared,
                fixed_inpaint_latent=denoiser_execution.fixed_inpaint_latent,
            )
    except BaseException as error:
        primary = error
        raise
    finally:
        if denoiser_execution.close is not None:
            try:
                denoiser_execution.close()
            except BaseException:
                if primary is None:
                    raise
    return registration.latent.finish(inputs, output, captured[-1] if captured else None)


def run_ksampler_as_custom(
    runtime: CustomSamplingRuntime[torch.Tensor],
    latent: CustomSamplingLatentValue,
    *,
    samplers: Registry[SamplerDescriptor[Any]],
    schedulers: Registry[SchedulerDescriptor],
    space: SigmaSpace,
    flow: bool,
    device: torch.device | str | None = None,
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    denoise: float | None = None,
    seed: int = 0,
    cond: CustomSamplingCondValue,
    cfg: CustomSamplingCfgValue = None,
    guidance: float | Literal["disabled"] | None = None,
    segment: SamplingSegment | None = None,
    denoise_mask: CustomSamplingLatentValue | None = None,
    inpaint: InpaintConditioning[torch.Tensor] | None = None,
    context_windows: ContextWindowsSpec | None = None,
    noise_inds: Sequence[int] | None = None,
    on_step: StepCallback | None = None,
    on_state: SamplingStateCallback | None = None,
    sample_custom_kwargs: Mapping[str, object] | None = None,
    noise_factory: Callable[[bool], CustomSamplingLatentValue] | None = None,
    error: type[Exception],
) -> (
    CustomSamplingResult[torch.Tensor]
    | CustomSamplingResult[MultiStreamLatent[torch.Tensor]]
    | CustomSamplingResult[SparseLatent[torch.Tensor]]
):
    """One KSampler run expressed as a custom sampling request.

    This is the sugar seam of the one sampling engine: it composes
    exactly what the equivalent node subgraph would - the KSampler
    schedule for the resolved sampler and scheduler (discard-penultimate
    included, pre-offset sigmas so ``sample_custom`` rebuilds the
    identical schedule and brownian bounds), the reference initial-noise
    draw for the seed (zeros when the segment declines noise), and the
    resolved sampler at its default options - then executes the
    family's ``sample_custom``. KSampler-flavored surfaces delegate
    here instead of owning a denoise loop."""
    extra_kwargs = {} if sample_custom_kwargs is None else dict(sample_custom_kwargs)
    reserved_kwargs = {
        "latent",
        "noise",
        "cond",
        "cfg",
        "request",
        "seed",
        "guidance",
        "denoise_mask",
        "inpaint",
        "context_windows",
        "on_step",
        "on_state",
    }
    conflicts = reserved_kwargs.intersection(extra_kwargs)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise error(f"sample_custom_kwargs cannot override KSampler composition: {names}")
    sampler, scheduler = resolve_sampling(
        samplers, schedulers, sampler_id, scheduler_id, error=error
    )
    schedule = build_sampling_schedule(
        scheduler,
        space,
        sampler,
        steps,
        denoise=denoise,
        flow=flow,
        segment=segment,
        device=device,
    )
    request = CustomSamplingRequest(sampler, (), schedule.pre_offset)
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=denoise_mask is not None,
        has_inpaint=inpaint is not None,
        has_context_windows=context_windows is not None,
        guidance=guidance if isinstance(guidance, float) else None,
    )
    add_noise = segment is None or segment.add_noise
    noise: CustomSamplingLatentValue
    if noise_factory is not None:
        noise = noise_factory(add_noise)
    elif type(latent) is SparseLatent:
        support, features = unpack_sparse_latent(latent)
        if not add_noise:
            noise_features = torch.zeros_like(features)
        elif noise_inds is None:
            noise_features = prepare_noise(features, seed)
        else:
            max_points = max(support.batch_counts)
            noise_batches = prepare_noise(
                features.new_empty((support.batch_size, max_points, features.shape[1])),
                seed,
                noise_inds,
            )
            noise_features = torch.cat(
                tuple(
                    noise_batches[batch, :count] for batch, count in enumerate(support.batch_counts)
                )
            )
        noise = pack_sparse_latent(support, noise_features)
    elif type(latent) is MultiStreamLatent:
        noise = (
            prepare_multistream_noise(latent, seed, noise_inds)
            if add_noise
            else latent.map(torch.zeros_like)
        )
    else:
        tensor_latent = cast("torch.Tensor", latent)
        noise = (
            prepare_noise(tensor_latent, seed, noise_inds)
            if add_noise
            else torch.zeros_like(tensor_latent)
        )
    return cast("Any", runtime.sample_custom)(
        latent,
        noise=noise,
        cond=cond,
        cfg=cfg,
        request=request,
        seed=seed,
        guidance=guidance,
        denoise_mask=denoise_mask,
        inpaint=inpaint,
        context_windows=context_windows,
        on_step=on_step,
        on_state=on_state,
        **extra_kwargs,
    )


__all__ = [
    "CustomSamplingCfgValue",
    "CustomSamplingCondValue",
    "CustomSamplingLatentValue",
    "SamplingAdapterContext",
    "SamplingDenoiserAdapter",
    "SamplingExecutionInputs",
    "SamplingExecutionRegistration",
    "SamplingGuidancePlan",
    "SamplingLatentAdapter",
    "SamplingSchedule",
    "SingleStreamLatentAdapter",
    "SingleStreamCustomSamplingCfg",
    "brownian_step_noise",
    "build_custom_sampling_schedule",
    "build_sampling_schedule",
    "compile_guidance_plan",
    "custom_denoised_callback",
    "guided_denoiser",
    "narrow_single_stream_custom_sampling",
    "resolve_custom_sampling_request",
    "resolve_sampling",
    "run_ksampler_as_custom",
    "sampling_execution",
]
