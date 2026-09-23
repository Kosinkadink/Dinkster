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
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol, cast

import torch
from dinkster_inference import (
    ConditioningCarrier,
    GuidanceRole,
    KeyedContribution,
    ModelFamily,
    PatchSet,
    PatchTargetComponent,
    RealizedSamplingTimeline,
    SamplingGuidance,
    encode_conditioning_carrier,
)

from .guidance import ConditioningBatch
from .guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)
from .patch_providers import PatchProviderSnapshot
from .regional import (
    MaterializedRegion,
    PreparedGroupedPatches,
    RegionalConditioningError,
    evaluate_grouped_regions,
    materialize_regions,
    prepare_grouped_patches,
    realize_region_schedules,
    region_schedule_is_active,
)
from .regional import (
    sd_grouped_region_evaluator as sd_grouped_region_evaluator,
)
from .sampling_execution import SamplingGuidancePlan
from .scaled_patches import PreparedScaledPatches, ScaledPatchError


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
    payloads: tuple[object, ...] = (),
    materialize: (
        Callable[[ConditioningCarrier, tuple[object, ...]], tuple[object, ...]] | None
    ) = None,
) -> tuple[tuple[MaterializedRegion, ...], tuple[ScheduledPatchResolutionRequest, ...]]:
    _check_cancel(cancel)
    try:
        canonical = encode_conditioning_carrier(carrier)
        regions = (
            materialize_regions(
                carrier,
                family_id,
                int(latent.shape[-2]),
                int(latent.shape[-1]),
                device,
            )
            if materialize is None
            else materialize(carrier, payloads)
        )
        if any(not isinstance(region, MaterializedRegion) for region in regions):
            raise TypeError("conditioning materializer must return MaterializedRegion values")
        regions = cast("tuple[MaterializedRegion, ...]", regions)
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


class ScheduledConditioningDenoiser:
    evaluator_identity = "dinkster.scheduled-conditioning.v1"

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
        self._prepared: dict[tuple[object, ...], PreparedGroupedPatches] = {}
        self._window_conditions: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...]], _ScheduledConditioning
        ] = {}
        self._closed = False
        self.prepare_count = 0
        self.model_calls = 0
        self.staged_bytes = 0

    def _owner(
        self,
        x: torch.Tensor,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
    ) -> PreparedGroupedPatches:
        _check_cancel(self._cancel)
        if self._closed:
            raise _refuse("denoiser-closed")
        regions = (*conditional, *unconditional)
        key = (
            id(conditional),
            id(unconditional),
            tuple(x.shape),
            tuple(
                (
                    region.area,
                    region.latent_shape,
                    id(region.mask),
                    region.patch_digest,
                )
                for region in regions
            ),
        )
        prepared = self._prepared.get(key)
        if prepared is None:
            try:
                prepared = prepare_grouped_patches(
                    conditional,
                    unconditional,
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
            self._prepared[key] = prepared
            self.prepare_count += 1
            self.staged_bytes += prepared.staged_bytes
        return prepared

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

    def window_conditioning(
        self,
        condition: _ScheduledConditioning,
        dim: int,
        indices: tuple[int, ...],
        input_shape: Sequence[int],
    ) -> _ScheduledConditioning:
        if len(input_shape) != 4 or dim not in (2, 3):
            raise _refuse("window-layout")
        shape = tuple(input_shape)
        key = (id(condition), dim, indices, shape)
        cached = self._window_conditions.get(key)
        if cached is not None:
            return cached
        latent_shape = (input_shape[2], input_shape[3])
        window_shape = list(latent_shape)
        window_shape[dim - 2] = len(indices)
        mapped: list[MaterializedRegion] = []
        for region in condition.regions:
            if region.latent_shape != latent_shape:
                raise _refuse("window-region-shape")
            mask = region.mask
            if mask is not None:
                index = torch.tensor(indices, dtype=torch.long, device=mask.device)
                mask = mask.index_select(dim - 1, index)
            area = region.area
            if area is None:
                mapped.append(replace(region, mask=mask, latent_shape=tuple(window_shape)))
                continue
            extent = area[dim - 2]
            offset = area[dim]
            positions = tuple(
                position
                for position, source_index in enumerate(indices)
                if offset <= source_index < offset + extent
            )
            if not positions:
                continue
            starts = [positions[0]]
            stops: list[int] = []
            for previous, current in zip(positions, positions[1:], strict=False):
                if current != previous + 1:
                    stops.append(previous + 1)
                    starts.append(current)
            stops.append(positions[-1] + 1)
            for start, stop in zip(starts, stops, strict=True):
                local_area = list(area)
                local_area[dim - 2] = stop - start
                local_area[dim] = start
                mapped.append(
                    replace(
                        region,
                        area=cast("tuple[int, int, int, int]", tuple(local_area)),
                        mask=mask,
                        latent_shape=tuple(window_shape),
                    )
                )
        result = _ScheduledConditioning(tuple(mapped), condition.role)
        self._window_conditions[key] = result
        return result

    def _evaluate_lanes(
        self,
        x: torch.Tensor,
        sigma: float,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = self._owner(x, conditional, unconditional)
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
        for prepared in self._prepared.values():
            prepared.close()
        self._prepared.clear()
        self._window_conditions.clear()


class FullLatentScheduledConditioningDenoiser:
    """Shared scheduled-region evaluation for structural latent layouts."""

    evaluator_identity = "dinkster.scheduled-full-latent.v1"

    def __init__(
        self,
        *,
        space: Any,
        model: torch.nn.Module,
        evaluate: Callable[[torch.Tensor, float, object, GuidanceRole], torch.Tensor],
        project: Callable[[MaterializedRegion, torch.Tensor], tuple[object, torch.Tensor]],
        patch_sets: Mapping[str, PatchSet[torch.Tensor]],
        compute_dtype: torch.dtype,
        device: torch.device,
        cancel: Callable[[], bool],
    ) -> None:
        self._space = space
        self._model = model
        self._evaluate = evaluate
        self._project = project
        self._compute_dtype = compute_dtype
        self._device = device
        self._cancel = cancel
        self._stack = ExitStack()
        self._patches: dict[str, PreparedScaledPatches] = {}
        self._closed = False
        try:
            for digest, patch_set in sorted(patch_sets.items()):
                if patch_set.structural_digest != digest:
                    raise _refuse("patch-mapping")
                self._patches[digest] = self._stack.enter_context(
                    PreparedScaledPatches(
                        model,
                        patch_set,
                        device,
                        compute_dtype,
                        cancel,
                    )
                )
        except ScaledPatchError as error:
            self._stack.close()
            raise _refuse("patch-preparation", error.code) from None
        except BaseException:
            self._stack.close()
            raise

    @staticmethod
    def prepare_conditioning(value: object, role: GuidanceRole) -> _ScheduledConditioning:
        return ScheduledConditioningDenoiser.prepare_conditioning(value, role)

    @staticmethod
    def batchable(conditions: tuple[_ScheduledConditioning, ...]) -> bool:
        del conditions
        return False

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: _ScheduledConditioning,
    ) -> torch.Tensor:
        _check_cancel(self._cancel)
        if self._closed:
            raise _refuse("denoiser-closed")
        output = torch.zeros_like(x)
        count = torch.ones_like(x) * 1e-37
        for region in condition.regions:
            if not region_schedule_is_active(region, sigma, self._space):
                continue
            prepared, multiplier = self._project(region, x)
            if multiplier.shape != x.shape or multiplier.device != x.device:
                raise _refuse("layout-multiplier")
            scale = region.scale_vector
            if scale is None:
                scale = torch.ones(1, dtype=torch.float32, device=x.device)
            if scale.numel() not in (1, x.shape[0]):
                raise _refuse("scale-batch")
            owner = None if region.patch_digest is None else self._patches.get(region.patch_digest)
            if region.patch_digest is not None and owner is None:
                raise _refuse("patch-mapping")
            try:
                activation = nullcontext() if owner is None else owner.activate(scale)
                with activation:
                    _check_cancel(self._cancel)
                    value = self._evaluate(x, sigma, prepared, condition.role)
            except ScaledPatchError as error:
                raise _refuse("patch-activation", error.code) from None
            if value.shape != x.shape or value.dtype != x.dtype or value.device != x.device:
                raise _refuse("callback-contract")
            output.add_(value * multiplier)
            count.add_(multiplier)
        return output / count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stack.close()
        self._patches.clear()


def prepare_scheduled_carriers(
    runtime: Any,
    latent: torch.Tensor,
    plan: SamplingGuidancePlan,
    *,
    resolver: ScheduledPatchResolver | None,
    device: torch.device,
    cancel: Callable[[], bool],
    timeline: RealizedSamplingTimeline | None,
    space: Any,
    payloads: Mapping[int, tuple[object, ...]] = MappingProxyType({}),
    materialize: (
        Callable[[ConditioningCarrier, tuple[object, ...]], tuple[object, ...]] | None
    ) = None,
) -> tuple[
    tuple[MaterializedRegion, ...],
    tuple[MaterializedRegion, ...],
    MappingProxyType[str, PatchSet[torch.Tensor]],
    SamplingGuidancePlan,
]:
    cond = cast("ConditioningCarrier", plan.conditions[0].conditioning)
    conditional, cond_requests = _materialize_carrier(
        cond,
        runtime.family.id,
        latent,
        device,
        runtime.runtime_identity,
        cancel,
        payloads.get(id(cond), ()),
        materialize,
    )
    unconditional: tuple[MaterializedRegion, ...] = ()
    uncond_requests: tuple[ScheduledPatchResolutionRequest, ...] = ()
    if plan.needs_unconditional:
        uncond = cast("ConditioningCarrier", plan.conditions[1].conditioning)
        unconditional, uncond_requests = _materialize_carrier(
            uncond,
            runtime.family.id,
            latent,
            device,
            runtime.runtime_identity,
            cancel,
            payloads.get(id(uncond), ()),
            materialize,
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


def narrow_scheduled_values(
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


def validate_unconditional_carrier(plan: SamplingGuidancePlan) -> None:
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


__all__ = [
    "ScheduledPatchResolution",
    "ScheduledPatchResolutionRequest",
    "ScheduledPatchResolver",
    "ScheduledSamplingOptions",
    "ScheduledSamplingError",
]
