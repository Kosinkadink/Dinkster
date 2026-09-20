"""Scheduled sampling runtime authority and cleanup.

The canonical DMFC carrier is the only schedule input. Patch digests in that
carrier are lookup identities, never authority: an explicit worker-local
resolver must bind every distinct request to one provider declaration,
generation, snapshot, and immutable PatchSet before model or staging work.
"""

# This module intentionally adapts private state from the two concrete runtimes
# without widening the torch-free FamilyRuntime protocol.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from dinkster_inference import (
    ConditioningCarrier,
    CustomSamplingRequest,
    CustomSamplingResult,
    GuidanceRole,
    InpaintConditioning,
    KeyedContribution,
    ModelFamily,
    NoiseKind,
    PatchSet,
    PatchTargetComponent,
    RealizedSamplingTimeline,
    SamplingGuidance,
    SamplingStateCallback,
    StepCallback,
    encode_conditioning_carrier,
    is_flow_parameterization,
    realize_sampling_timeline,
    sampling_execution_context,
    use_sampling_environment,
)

from .denoise import DenoiseError, FluxGuidance, run_denoise
from .guidance import (
    ConditioningBatch,
    ConditioningEvaluation,
)
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .patch_providers import PatchProviderSnapshot
from .regional import (
    MaterializedRegion,
    PreparedGroupedPatches,
    RegionalConditioningError,
    evaluate_grouped_regions,
    flux_grouped_region_evaluator,
    materialize_regions,
    prepare_grouped_patches,
    realize_region_schedules,
    sd_grouped_region_evaluator,
)
from .sampling_execution import (
    SamplingGuidancePlan,
    brownian_step_noise,
    build_custom_sampling_schedule,
    compile_guidance_plan,
    custom_denoised_callback,
    guided_denoiser,
    resolve_custom_sampling_request,
)

if TYPE_CHECKING:
    from .wiring import FluxRuntime, SDRuntime


class ScheduledSamplingError(ValueError):
    """A deterministic scheduled-runtime refusal with a stable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = f"scheduled-sampling:{code}"
        super().__init__(f"{message}: {detail}" if detail else message)


def _refuse(code: str, detail: str = "") -> ScheduledSamplingError:
    return ScheduledSamplingError(code, detail)


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ScheduledPatchResolutionRequest:
    """Exact carrier/runtime/stack identity submitted to patch authority."""

    carrier_sha256: str
    runtime_identity: str
    target: PatchTargetComponent
    overlay_digests: tuple[str, ...]
    stack_digest: str

    def __post_init__(self) -> None:
        if not _is_digest(self.carrier_sha256):
            raise TypeError("carrier_sha256 must be a lowercase sha256 digest")
        if type(self.runtime_identity) is not str or not self.runtime_identity:
            raise TypeError("runtime_identity must be a nonempty string")
        if self.target is not PatchTargetComponent.DIFFUSION:
            raise TypeError("scheduled sampling resolves only the DIFFUSION target")
        if (
            not isinstance(self.overlay_digests, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not self.overlay_digests
            or any(not _is_digest(value) for value in self.overlay_digests)
        ):
            raise TypeError("overlay_digests must be a nonempty digest tuple")
        if not _is_digest(self.stack_digest):
            raise TypeError("stack_digest must be a lowercase sha256 digest")
        expected = hashlib.sha256(
            ("[" + ",".join(f'"{value}"' for value in self.overlay_digests) + "]").encode("ascii")
        ).hexdigest()
        if self.stack_digest != expected:
            raise ValueError("stack_digest does not bind overlay_digests in order")


@dataclass(frozen=True)
class ScheduledPatchResolution:
    """One worker-local provider's authority for a complete patch stack."""

    request: ScheduledPatchResolutionRequest
    generation_key: str
    snapshot: PatchProviderSnapshot
    provider_id: str
    declaration: KeyedContribution
    patch_set: PatchSet[torch.Tensor]

    def __post_init__(self) -> None:
        if type(self.request) is not ScheduledPatchResolutionRequest:
            raise TypeError("request must be ScheduledPatchResolutionRequest")
        if type(self.generation_key) is not str or not self.generation_key:
            raise TypeError("generation_key must be a nonempty string")
        if type(self.snapshot) is not PatchProviderSnapshot:
            raise TypeError("snapshot must be PatchProviderSnapshot")
        if type(self.provider_id) is not str or not self.provider_id:
            raise TypeError("provider_id must be a nonempty string")
        if type(self.declaration) is not KeyedContribution:
            raise TypeError("declaration must be KeyedContribution")
        if self.declaration.id != self.provider_id:
            raise ValueError("provider_id does not match its declaration")
        equal = tuple(
            declaration
            for declaration in self.snapshot.providers
            if declaration.id == self.provider_id
        )
        if equal != (self.declaration,):
            raise ValueError("provider declaration is not present exactly in the snapshot")
        if type(self.patch_set) is not PatchSet:
            raise TypeError("patch_set must be PatchSet")
        if not self.patch_set.structural_digest:
            raise ValueError("PatchSet structural digest must be nonempty")
        if self.patch_set.structural_digest != self.request.stack_digest:
            raise ValueError("PatchSet structural digest does not match the requested stack")


class ScheduledPatchResolver(Protocol):
    """Resolve one complete, immutable request set in one transaction."""

    def __call__(
        self,
        requests: tuple[ScheduledPatchResolutionRequest, ...],
        cancel: Callable[[], bool],
    ) -> tuple[ScheduledPatchResolution, ...]: ...


@dataclass(frozen=True)
class ScheduledSamplingOptions:
    """Family-specific inputs that select scheduled conditioning on the custom seam."""

    resolver: ScheduledPatchResolver | None = None
    cancelled: Callable[[], bool] | None = None


def _check_cancel(cancel: Callable[[], bool]) -> None:
    result = cancel()
    if type(result) is not bool:
        raise _refuse("cancel-callback")
    if result:
        raise _refuse("cancelled")


def _not_cancelled() -> bool:
    return False


def _metadata_request(
    region: MaterializedRegion,
    carrier_sha256: str,
    runtime_identity: str,
) -> ScheduledPatchResolutionRequest | None:
    if region.patch_digest is None:
        return None
    metadata = dict(region.extension_metadata)
    overlays = metadata.get("dinkster.inference/diffusion-overlay-digests")
    stack = metadata.get("dinkster.inference/diffusion-overlay-stack-digest")
    if not isinstance(overlays, (tuple, list)) or any(type(value) is not str for value in overlays):
        raise _refuse("request-overlay-metadata")
    try:
        return ScheduledPatchResolutionRequest(
            carrier_sha256,
            runtime_identity,
            PatchTargetComponent.DIFFUSION,
            cast("tuple[str, ...]", tuple(overlays)),
            cast(str, stack),
        )
    except (TypeError, ValueError) as error:
        raise _refuse("request-identity", str(error)) from None


def _materialize_carrier(
    carrier: ConditioningCarrier,
    family_id: str,
    latent: torch.Tensor,
    device: torch.device,
    runtime_identity: str,
    cancel: Callable[[], bool],
) -> tuple[tuple[MaterializedRegion, ...], tuple[ScheduledPatchResolutionRequest, ...]]:
    _check_cancel(cancel)
    try:
        canonical = encode_conditioning_carrier(carrier)
        regions = materialize_regions(
            carrier,
            family_id,
            int(latent.shape[-2]),
            int(latent.shape[-1]),
            device,
        )
    except RegionalConditioningError as error:
        raise _refuse("materialization", error.code) from None
    except (TypeError, ValueError) as error:
        raise _refuse("carrier", str(error)) from None
    digest = hashlib.sha256(canonical).hexdigest()
    requests = tuple(
        request
        for region in regions
        if (request := _metadata_request(region, digest, runtime_identity)) is not None
    )
    _check_cancel(cancel)
    return regions, requests


def _resolve_patch_sets(
    requests: tuple[ScheduledPatchResolutionRequest, ...],
    resolver: ScheduledPatchResolver | None,
    cancel: Callable[[], bool],
) -> MappingProxyType[str, PatchSet[torch.Tensor]]:
    distinct = tuple(dict.fromkeys(requests))
    if not distinct:
        if resolver is not None and not callable(resolver):
            raise _refuse("resolver-type")
        return MappingProxyType({})
    if resolver is None or not callable(resolver):
        raise _refuse("resolver-required")
    _check_cancel(cancel)
    resolved = resolver(distinct, cancel)
    _check_cancel(cancel)
    if type(resolved) is not tuple:
        raise _refuse("resolution-type")
    if any(type(item) is not ScheduledPatchResolution for item in resolved):
        raise _refuse("resolution-type")
    by_request: dict[int, ScheduledPatchResolution] = {}
    requested_ids = {id(request) for request in distinct}
    for item in resolved:
        request_id = id(item.request)
        if request_id not in requested_ids:
            raise _refuse("resolution-request")
        if request_id in by_request:
            raise _refuse("resolution-duplicate")
        by_request[request_id] = item
    if set(by_request) != requested_ids:
        raise _refuse("resolution-set")
    generation = resolved[0].generation_key
    snapshot = resolved[0].snapshot
    patches: dict[str, PatchSet[torch.Tensor]] = {}
    providers: dict[str, tuple[str, KeyedContribution]] = {}
    patch_digests_by_identity: dict[int, str] = {}
    for request in distinct:
        item = by_request[id(request)]
        if item.request is not request:
            raise _refuse("resolution-request")
        if item.generation_key != generation or not item.generation_key:
            raise _refuse("resolution-generation")
        if item.snapshot != snapshot:
            raise _refuse("resolution-snapshot")
        if item.provider_id != item.declaration.id:
            raise _refuse("resolution-provider")
        if tuple(value for value in item.snapshot.providers if value == item.declaration) != (
            item.declaration,
        ):
            raise _refuse("resolution-declaration")
        if type(item.patch_set) is not PatchSet:
            raise _refuse("resolution-patch-type")
        if item.patch_set.structural_digest != request.stack_digest:
            raise _refuse("resolution-patch-digest")
        previous = patches.get(request.stack_digest)
        if previous is not None and previous is not item.patch_set:
            raise _refuse("resolution-patch-identity")
        previous_provider = providers.get(request.stack_digest)
        authority = (item.provider_id, item.declaration)
        if previous_provider is not None and previous_provider != authority:
            raise _refuse("resolution-provider")
        previous_digest = patch_digests_by_identity.get(id(item.patch_set))
        if previous_digest is not None and previous_digest != request.stack_digest:
            raise _refuse("resolution-patch-identity")
        patches[request.stack_digest] = item.patch_set
        providers[request.stack_digest] = authority
        patch_digests_by_identity[id(item.patch_set)] = request.stack_digest
    return MappingProxyType(patches)


@dataclass(frozen=True)
class _ScheduledConditioning:
    regions: tuple[MaterializedRegion, ...]
    role: GuidanceRole


class _ScheduledDenoiser:
    def __init__(
        self,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
        *,
        family: ModelFamily,
        space: Any,
        model: torch.nn.Module,
        evaluate: Any,
        patch_sets: MappingProxyType[str, PatchSet[torch.Tensor]],
        compute_dtype: torch.dtype,
        device: torch.device,
        cancel: Callable[[], bool],
    ) -> None:
        self._conditional = conditional
        self._unconditional = unconditional
        self._family = family
        self._family_id = family.id
        self._space = space
        self._model = model
        self._evaluate = evaluate
        self._patch_sets = patch_sets
        self._compute_dtype = compute_dtype
        self._device = device
        self._cancel = cancel
        self._prepared: PreparedGroupedPatches | None = None
        self._closed = False
        self.prepare_count = 0
        self.model_calls = 0
        self.staged_bytes = 0

    def _owner(self, x: torch.Tensor) -> PreparedGroupedPatches:
        _check_cancel(self._cancel)
        if self._closed:
            raise _refuse("denoiser-closed")
        if self._prepared is None:
            try:
                self._prepared = prepare_grouped_patches(
                    self._conditional,
                    self._unconditional,
                    x,
                    self._family_id,
                    self._space,
                    self._model,
                    self._patch_sets,
                    self._device,
                    self._compute_dtype,
                    self._cancel,
                )
            except RegionalConditioningError as error:
                raise _refuse("preparation", error.code) from None
            self.prepare_count += 1
            self.staged_bytes = self._prepared.staged_bytes
        return self._prepared

    @staticmethod
    def prepare_conditioning(value: object, role: GuidanceRole) -> _ScheduledConditioning:
        if not isinstance(value, tuple) or any(
            not isinstance(region, MaterializedRegion) for region in value
        ):
            raise _refuse("conditioning-lane")
        return _ScheduledConditioning(cast("tuple[MaterializedRegion, ...]", value), role)

    @staticmethod
    def batchable(conditions: tuple[_ScheduledConditioning, ...]) -> bool:
        return bool(conditions) and len({condition.role for condition in conditions}) == len(
            conditions
        )

    def _evaluate_lanes(
        self,
        x: torch.Tensor,
        sigma: float,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = self._owner(x)
        try:
            result = evaluate_grouped_regions(
                conditional,
                unconditional,
                x,
                float(sigma),
                self._space,
                self._family,
                self._model,
                self._evaluate,
                owner,
                self._cancel,
                compute_dtype=self._compute_dtype,
            )
        except RegionalConditioningError as error:
            raise _refuse("evaluation", error.code) from None
        self.model_calls += result.model_calls
        return result.conditional, result.unconditional

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: _ScheduledConditioning,
    ) -> torch.Tensor:
        if condition.role is GuidanceRole.CONDITIONAL:
            conditional, _ = self._evaluate_lanes(x, sigma, condition.regions, ())
            return conditional
        _, unconditional = self._evaluate_lanes(x, sigma, (), condition.regions)
        return unconditional

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _validate_conditioning_batch(
        self,
        x: torch.Tensor,
        conditions: tuple[_ScheduledConditioning, ...],
    ) -> None:
        if not self.batchable(conditions):
            raise _refuse("conditioning-batch")

    @staticmethod
    def _stack_conditioning_model_input(
        model_input: torch.Tensor,
        conditions: tuple[_ScheduledConditioning, ...],
    ) -> torch.Tensor:
        return model_input

    def _evaluate_conditioning_model(
        self,
        batch: ConditioningBatch[_ScheduledConditioning],
    ) -> torch.Tensor:
        conditions = batch.conditions
        conditional = next(
            (
                condition.regions
                for condition in conditions
                if condition.role is GuidanceRole.CONDITIONAL
            ),
            (),
        )
        unconditional = next(
            (
                condition.regions
                for condition in conditions
                if condition.role is GuidanceRole.UNCONDITIONAL
            ),
            (),
        )
        cond_output, uncond_output = self._evaluate_lanes(
            batch.latent,
            batch.sigma,
            conditional,
            unconditional,
        )
        return torch.cat(
            tuple(
                uncond_output if condition.role is GuidanceRole.UNCONDITIONAL else cond_output
                for condition in conditions
            )
        )

    @staticmethod
    def _conditioning_denoised(
        batch: ConditioningBatch[_ScheduledConditioning],
        output: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._prepared is not None:
            self._prepared.close()
            self._prepared = None


def _progress_callback(
    callback: StepCallback | None, cancel: Callable[[], bool]
) -> StepCallback | None:
    if callback is None:
        return None

    def checked(event: Any) -> None:
        _check_cancel(cancel)
        callback(event)
        _check_cancel(cancel)

    return checked


def _drive(
    denoiser: Any,
    owner: _ScheduledDenoiser,
    solver: Any,
    *,
    latent: torch.Tensor,
    noise: torch.Tensor,
    sigmas: tuple[float, ...],
    initial_sigma: float | None,
    family: Any,
    sampling: Any = None,
    seed: int,
    noise_kind: NoiseKind,
    noise_sampler: Any,
    percent_to_sigma: Callable[[float], float],
    device: torch.device,
    on_step: StepCallback | None,
    on_state: SamplingStateCallback | None,
    cancel: Callable[[], bool],
) -> torch.Tensor:
    primary: BaseException | None = None
    try:
        _check_cancel(cancel)
        with use_sampling_environment((), cancel):
            result = run_denoise(
                denoiser,
                solver,
                latent=latent,
                noise=noise,
                sigmas=sigmas,
                initial_sigma=initial_sigma,
                family=family,
                sampling=sampling,
                seed=seed,
                noise_kind=noise_kind,
                noise_sampler=noise_sampler,
                percent_to_sigma=percent_to_sigma,
                device=device,
                on_step=_progress_callback(on_step, cancel),
                on_state=on_state,
            )
        _check_cancel(cancel)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            owner.close()
        except BaseException:
            if primary is None:
                raise
    return result


def _prepare_carriers(
    runtime: Any,
    latent: torch.Tensor,
    plan: SamplingGuidancePlan,
    *,
    resolver: ScheduledPatchResolver | None,
    device: torch.device,
    cancel: Callable[[], bool],
    timeline: RealizedSamplingTimeline | None,
    space: Any,
) -> tuple[
    tuple[MaterializedRegion, ...],
    tuple[MaterializedRegion, ...],
    MappingProxyType[str, PatchSet[torch.Tensor]],
    SamplingGuidancePlan,
]:
    cond = cast("ConditioningCarrier", plan.conditions[0].conditioning)
    conditional, cond_requests = _materialize_carrier(
        cond, runtime.family.id, latent, device, runtime.runtime_identity, cancel
    )
    unconditional: tuple[MaterializedRegion, ...] = ()
    uncond_requests: tuple[ScheduledPatchResolutionRequest, ...] = ()
    if plan.needs_unconditional:
        uncond = cast("ConditioningCarrier", plan.conditions[1].conditioning)
        unconditional, uncond_requests = _materialize_carrier(
            uncond, runtime.family.id, latent, device, runtime.runtime_identity, cancel
        )
    if timeline is not None:
        conditional = realize_region_schedules(conditional, timeline, space)
        unconditional = realize_region_schedules(unconditional, timeline, space)
    patch_sets = _resolve_patch_sets(cond_requests + uncond_requests, resolver, cancel)
    return (
        conditional,
        unconditional,
        patch_sets,
        plan.with_conditioning(conditional, unconditional or None),
    )


def _narrow_scheduled_values(
    family_id: str,
    *,
    latent: object,
    noise: object,
    cond: object,
    cfg: object,
    denoise_mask: object,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    ConditioningCarrier,
    SamplingGuidance[ConditioningCarrier] | None,
    torch.Tensor | None,
]:
    if not isinstance(latent, torch.Tensor) or not isinstance(noise, torch.Tensor):
        raise _refuse("latent-shape", family_id)
    if type(cond) is not ConditioningCarrier:
        raise _refuse("carrier", "conditional lane must be a ConditioningCarrier")
    if cfg is not None:
        if type(cfg) is not SamplingGuidance:
            raise _refuse("guidance-carrier", "expected SamplingGuidance")
    if denoise_mask is not None and not isinstance(denoise_mask, torch.Tensor):
        raise _refuse("denoise-mask", family_id)
    return (
        latent,
        noise,
        cond,
        cast("SamplingGuidance[ConditioningCarrier] | None", cfg),
        denoise_mask,
    )


def _validate_unconditional_carrier(plan: SamplingGuidancePlan) -> None:
    if not plan.needs_unconditional:
        return
    unconditional = next(
        (
            condition.conditioning
            for condition in plan.conditions
            if condition.role is GuidanceRole.UNCONDITIONAL
        ),
        None,
    )
    if type(cast("object", unconditional)) is not ConditioningCarrier:
        raise _refuse("guidance-carrier", "unconditional lane must be a ConditioningCarrier")


def _resolve_sampler(
    runtime: Any,
    request: CustomSamplingRequest[torch.Tensor],
) -> tuple[Any, CustomSamplingRequest[torch.Tensor]]:
    if type(request) is not CustomSamplingRequest:
        raise TypeError("custom sampling requires an exact CustomSamplingRequest")
    if runtime._samplers.get(request.sampler.id) is None:
        raise _refuse("unknown-sampler", request.sampler.id)
    return resolve_custom_sampling_request(
        runtime._samplers,
        request,
        error=ScheduledSamplingError,
    )


def sample_flux_scheduled_custom(
    runtime: FluxRuntime,
    latent: object,
    *,
    noise: object,
    cond: object,
    cfg: object,
    request: CustomSamplingRequest[torch.Tensor],
    seed: int = 0,
    guidance: FluxGuidance = None,
    denoise_mask: object = None,
    inpaint: InpaintConditioning[torch.Tensor] | None = None,
    context_windows: object = None,
    window_plan: object = None,
    on_step: StepCallback | None = None,
    on_state: SamplingStateCallback | None = None,
    resolver: ScheduledPatchResolver | None = None,
    cancelled: Callable[[], bool] | None = None,
    compute_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    capture_denoised: bool = True,
) -> CustomSamplingResult[torch.Tensor]:
    cancel = _not_cancelled if cancelled is None else cancelled
    if not callable(cancel):
        raise _refuse("cancel-callback")
    if runtime._guidance is not None and runtime._guidance.registry.active:
        raise _refuse("guidance-extensions")
    if type(cfg) is SamplingGuidance and cfg.transforms:
        raise _refuse("guidance-extensions")
    if denoise_mask is not None:
        raise _refuse("denoise-mask", runtime.family.id)
    if inpaint is not None:
        raise _refuse("inpaint", runtime.family.id)
    if context_windows is not None:
        raise _refuse("context-windows", runtime.family.id)
    if window_plan is not None:
        raise _refuse("window-plan", runtime.family.id)
    if guidance == "disabled":
        raise _refuse("distilled-guidance", runtime.family.id)
    if guidance is not None and runtime.assembled.diffusion.guidance_in is None:
        raise DenoiseError(
            "guidance was given but this Flux model has no guidance"
            " embedder (schnell); pass guidance=None"
        )
    latent, noise, cond, cfg, denoise_mask = _narrow_scheduled_values(
        runtime.family.id,
        latent=latent,
        noise=noise,
        cond=cond,
        cfg=cfg,
        denoise_mask=denoise_mask,
    )
    sampler, request = _resolve_sampler(runtime, request)
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=False,
        has_inpaint=False,
        has_context_windows=False,
        guidance=guidance,
    )
    plan = compile_guidance_plan(cond, cfg, sampler, runtime._guidance)
    _validate_unconditional_carrier(plan)
    target = latent.device if device is None else torch.device(device)
    if compute_dtype is None:
        compute_dtype = runtime.assembled.compute_dtype("diffusion") or torch.bfloat16
    space = runtime._space
    schedule = build_custom_sampling_schedule(
        request.sigmas,
        space,
        sampler,
        flow=is_flow_parameterization(runtime.family.sampling.parameterization),
    )
    noise_sampler = brownian_step_noise(sampler, schedule, latent, seed=seed, device=target)
    sigmas = schedule.sigmas
    realized_timeline = (
        None
        if request.timeline is None
        else realize_sampling_timeline(request.timeline, tuple(float(sigma) for sigma in sigmas))
    )
    conditional, unconditional, patch_sets, materialized_plan = _prepare_carriers(
        runtime,
        latent,
        plan,
        resolver=resolver,
        device=target,
        cancel=cancel,
        timeline=realized_timeline,
        space=space,
    )
    denoiser = _ScheduledDenoiser(
        conditional,
        unconditional,
        family=runtime.family,
        space=space,
        model=runtime.assembled.diffusion,
        evaluate=flux_grouped_region_evaluator(
            runtime.assembled.diffusion,
            guidance=guidance,
            compute_dtype=compute_dtype,
        ),
        patch_sets=patch_sets,
        compute_dtype=compute_dtype,
        device=target,
        cancel=cancel,
    )
    guided = guided_denoiser(
        ConditioningEvaluation(
            denoiser.prepare_conditioning,
            denoiser.evaluate_conditioning,
            denoiser.batchable,
            denoiser.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: f"{runtime.family.id}.scheduled-conditioning.v1",
            standard_activation_memory_factor=runtime.family.memory_factor,
        ),
        input=latent,
        executor=None,
        plan=materialized_plan,
        execution=sampling_execution_context(sigmas, seed, on_step),
    )
    report_state: SamplingStateCallback | None
    captured: list[torch.Tensor]
    if capture_denoised:
        report_state, captured = custom_denoised_callback(runtime.family, on_state)
    else:
        report_state, captured = on_state, []
    output = _drive(
        guided,
        denoiser,
        request.build_solver(realized_timeline=realized_timeline),
        latent=latent,
        noise=noise,
        sigmas=sigmas,
        initial_sigma=schedule.initial_sigma,
        family=runtime.family,
        seed=seed,
        noise_kind=sampler.noise,
        noise_sampler=noise_sampler,
        percent_to_sigma=space.percent_to_sigma,
        device=target,
        on_step=on_step,
        on_state=report_state,
        cancel=cancel,
    )
    return CustomSamplingResult(output, captured[-1] if captured else None)


def sample_sd_scheduled_custom(
    runtime: SDRuntime,
    latent: object,
    *,
    noise: object,
    cond: object,
    cfg: object,
    request: CustomSamplingRequest[torch.Tensor],
    seed: int = 0,
    guidance: float | None = None,
    denoise_mask: object = None,
    inpaint: InpaintConditioning[torch.Tensor] | None = None,
    context_windows: object = None,
    control: object = None,
    attention_contributions: object = (),
    on_step: StepCallback | None = None,
    on_state: SamplingStateCallback | None = None,
    resolver: ScheduledPatchResolver | None = None,
    cancelled: Callable[[], bool] | None = None,
    compute_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    capture_denoised: bool = True,
) -> CustomSamplingResult[torch.Tensor]:
    cancel = _not_cancelled if cancelled is None else cancelled
    if not callable(cancel):
        raise _refuse("cancel-callback")
    if control is not None:
        raise _refuse("control", runtime.family.id)
    if type(attention_contributions) is not tuple:
        raise _refuse("attention-contributions", runtime.family.id)
    if attention_contributions:
        raise _refuse("attention-contributions", runtime.family.id)
    if runtime._guidance is not None and runtime._guidance.registry.active:
        raise _refuse("guidance-extensions")
    if type(cfg) is SamplingGuidance and cfg.transforms:
        raise _refuse("guidance-extensions")
    if guidance is not None:
        raise _refuse("distilled-guidance", runtime.family.id)
    if (
        denoise_mask is not None
        or inpaint is not None
        or runtime.assembled.diffusion.config.in_channels == 9
    ):
        raise _refuse("scheduled-sd-inpaint")
    if context_windows is not None:
        raise _refuse("context-windows", runtime.family.id)
    latent, noise, cond, cfg, denoise_mask = _narrow_scheduled_values(
        runtime.family.id,
        latent=latent,
        noise=noise,
        cond=cond,
        cfg=cfg,
        denoise_mask=denoise_mask,
    )
    sampler, request = _resolve_sampler(runtime, request)
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=False,
        has_inpaint=False,
        has_context_windows=False,
        guidance=None,
    )
    plan = compile_guidance_plan(cond, cfg, sampler, runtime._guidance)
    _validate_unconditional_carrier(plan)
    target = latent.device if device is None else torch.device(device)
    if compute_dtype is None:
        compute_dtype = runtime.assembled.compute_dtype("diffusion") or torch.float16
    schedule = build_custom_sampling_schedule(
        request.sigmas,
        runtime._space,
        sampler,
        flow=False,
    )
    sigmas = schedule.sigmas
    realized_timeline = (
        None
        if request.timeline is None
        else realize_sampling_timeline(request.timeline, tuple(float(sigma) for sigma in sigmas))
    )
    conditional, unconditional, patch_sets, materialized_plan = _prepare_carriers(
        runtime,
        latent,
        plan,
        resolver=resolver,
        device=target,
        cancel=cancel,
        timeline=realized_timeline,
        space=runtime._space,
    )
    adm = None
    if runtime.assembled.diffusion.config.adm_in_channels is not None:

        def resolve_adm(region: MaterializedRegion, role: GuidanceRole) -> torch.Tensor | None:
            return runtime._adm(
                region.conditioning,
                latent,
                negative=role is GuidanceRole.UNCONDITIONAL,
            )

        adm = resolve_adm
    denoiser = _ScheduledDenoiser(
        conditional,
        unconditional,
        family=runtime.family,
        space=runtime._space,
        model=runtime.assembled.diffusion,
        evaluate=sd_grouped_region_evaluator(
            runtime.assembled.diffusion,
            runtime._space,
            parameterization=runtime.sampling.parameterization,
            adm=adm,
            compute_dtype=compute_dtype,
        ),
        patch_sets=patch_sets,
        compute_dtype=compute_dtype,
        device=target,
        cancel=cancel,
    )
    guided = guided_denoiser(
        ConditioningEvaluation(
            denoiser.prepare_conditioning,
            denoiser.evaluate_conditioning,
            denoiser.batchable,
            denoiser.evaluate_conditioning_batch,
            evaluator_identity=lambda _role: f"{runtime.family.id}.scheduled-conditioning.v1",
            standard_activation_memory_factor=runtime.family.memory_factor,
        ),
        input=latent,
        executor=None,
        plan=materialized_plan,
        execution=sampling_execution_context(sigmas, seed, on_step),
    )
    report_state: SamplingStateCallback | None
    captured: list[torch.Tensor]
    if capture_denoised:
        report_state, captured = custom_denoised_callback(runtime.family, on_state)
    else:
        report_state, captured = on_state, []
    output = _drive(
        guided,
        denoiser,
        request.build_solver(realized_timeline=realized_timeline),
        latent=latent,
        noise=noise,
        sigmas=sigmas,
        initial_sigma=None,
        family=runtime.family,
        sampling=runtime.sampling,
        seed=seed,
        noise_kind=sampler.noise,
        noise_sampler=None,
        percent_to_sigma=runtime._percent_to_sigma,
        device=target,
        on_step=on_step,
        on_state=report_state,
        cancel=cancel,
    )
    return CustomSamplingResult(output, captured[-1] if captured else None)


__all__ = [
    "ScheduledPatchResolution",
    "ScheduledPatchResolutionRequest",
    "ScheduledPatchResolver",
    "ScheduledSamplingOptions",
    "ScheduledSamplingError",
]
