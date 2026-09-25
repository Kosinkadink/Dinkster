"""Shared component execution assembly for native family adapters."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from ..native_arm_core import (
    _NATIVE_PREPARED_CONDITIONING_KEY,
    Any,
    NativeResidencyBusyError,
    NativeRuntimeHandle,
    _active_inference_registries,
    _component_bound_carrier,
    _split_ltx_frame_rate,
    _torch,
    dataclass,
    log,
)
from ..native_arm_runtime import _torch_dtype


@dataclass(frozen=True, slots=True)
class _ResolvedComponentExecution:
    runtime: Any
    positive: object
    negative: object
    conditioning_prepared: bool

    def values(self) -> tuple[Any, object, object]:
        return self.runtime, self.positive, self.negative


def _component_runtime_with_options(
    base_runtime: Any,
    descriptor: Any,
    family_id: str,
    runtime_identity: str,
    compute_dtype: Any,
    sampling_shift: float | None,
    option_windows: tuple[Any, ...],
) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    sampling_runtime = getattr(base_runtime, "component_sampling_runtime", base_runtime)
    runtime_matches = any(
        isinstance(sampling_runtime, execution_symbol(reference))
        for reference in (descriptor.runtime_class, *descriptor.runtime_variants)
    )
    label = descriptor.family.display_name
    if not runtime_matches or base_runtime.runtime_identity != sampling_runtime.runtime_identity:
        raise TypeError(f"model must be a native {label} diffusion component")
    if sampling_runtime.family.id != family_id:
        raise ValueError(
            f"runtime producer family {sampling_runtime.family.id!r} does not match "
            f"reconstruction recipe family {family_id!r}"
        )
    runtime_options = descriptor.execution_options(sampling_runtime, sampling_shift, option_windows)
    assembled = getattr(sampling_runtime, "assembled", None)
    module = sampling_runtime.model if assembled is None else assembled.diffusion
    with_execution_options = getattr(sampling_runtime, "with_execution_options", None)
    if callable(with_execution_options):
        runtime = with_execution_options(
            runtime_identity=runtime_identity,
            compute_dtype=compute_dtype,
            **runtime_options,
        )
    elif descriptor.runtime_with_family:
        runtime = type(sampling_runtime)(
            module,
            sampling_runtime.family,
            runtime_identity=runtime_identity,
            **runtime_options,
        )
    else:
        runtime = type(sampling_runtime)(
            module,
            runtime_identity=runtime_identity,
            compute_dtype=compute_dtype,
            **runtime_options,
        )
    if sampling_runtime is base_runtime:
        return runtime
    replace_runtime = getattr(base_runtime, "with_component_sampling_runtime", None)
    if not callable(replace_runtime):
        raise TypeError(f"native {label} checkpoint cannot replace its sampling runtime")
    return replace_runtime(runtime)


def _resolve_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> _ResolvedComponentExecution | None:
    from dinkster_inference.component_registry import execution_symbol

    recipe = handle.recipe
    descriptor = _active_inference_registries().components.get(recipe.family_id)
    if descriptor is None:
        return None
    execution_options: dict[str, Any] = {}
    if negative_handle is not None:
        execution_options["negative_handle"] = negative_handle
    if image_only_negative:
        execution_options["image_only_negative"] = True
    if descriptor.execution_resolver is not None:
        execution = execution_symbol(descriptor.execution_resolver)(
            handle,
            positive,
            negative,
            inference,
            **execution_options,
        )
        if execution is not None:
            return _ResolvedComponentExecution(*execution, conditioning_prepared=True)
        if execution_options:
            raise TypeError("runtime does not support separate-model or image-only guidance")
        log.warning(
            "component execution resolver declined registered descriptor %s; "
            "using runtime without preparing conditioning",
            descriptor.id,
        )
        return _ResolvedComponentExecution(
            handle.runtime,
            positive,
            negative,
            conditioning_prepared=False,
        )
    if execution_options:
        raise ValueError(
            "component runtime does not implement separate negative-model conditioning"
        )
    if tuple(source.role for source in recipe.sources) != (descriptor.model_role,):
        runtime = handle.runtime
        if descriptor.execution_options is not None:
            sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
            runtime = _component_runtime_with_options(
                runtime,
                descriptor,
                recipe.family_id,
                recipe.runtime_identity,
                sampling_runtime.assembled.compute_dtype("diffusion"),
                sampling_shift,
                option_windows,
            )
        return _ResolvedComponentExecution(
            runtime,
            positive,
            negative,
            conditioning_prepared=False,
        )
    base_runtime = handle.runtime
    runtime_matches = any(
        isinstance(base_runtime, execution_symbol(reference))
        for reference in (descriptor.runtime_class, *descriptor.runtime_variants)
    )
    label = descriptor.family.display_name
    if not runtime_matches or recipe.runtime_identity != base_runtime.runtime_identity:
        raise TypeError(f"model must be a native {label} diffusion component")
    if base_runtime.family.id != recipe.family_id:
        raise ValueError(
            f"runtime producer family {base_runtime.family.id!r} does not match "
            f"reconstruction recipe family {recipe.family_id!r}"
        )
    positive_carrier, positive_binding = _component_bound_carrier(positive, inference)
    if positive_binding is None:
        if descriptor.allow_unbound_conditioning:
            return _ResolvedComponentExecution(
                base_runtime,
                positive,
                negative,
                conditioning_prepared=False,
            )
        raise TypeError(f"positive must be {label} component-bound conditioning")
    conditioning_families = (recipe.family_id, *descriptor.shared_conditioning_families)
    if positive_binding.family_id not in conditioning_families:
        raise ValueError(f"positive {label} conditioning has the wrong component family")
    conditioning_roles = descriptor.conditioning_roles or descriptor.text_encoder_roles
    if positive_binding.role not in conditioning_roles:
        raise ValueError(f"positive {label} conditioning has the wrong component role")
    negative_carrier = None
    if negative not in ([], None):
        negative_carrier, negative_binding = _component_bound_carrier(negative, inference)
        if negative_binding is None:
            raise TypeError(f"negative must be {label} component-bound conditioning or empty")
        if negative_binding.family_id not in conditioning_families:
            raise ValueError(f"negative {label} conditioning has the wrong component family")
        if negative_binding.role != positive_binding.role:
            raise ValueError(f"negative {label} conditioning has the wrong component role")
        if negative_binding != positive_binding:
            raise ValueError(f"{label} conditioning lanes must share one component binding")
    composition = inference.compose_execution(
        recipe.family_id,
        {
            descriptor.model_role: recipe.runtime_identity,
            positive_binding.role: positive_binding.identity,
        },
        shared_component_families=frozenset(descriptor.shared_conditioning_families),
    )
    if descriptor.execution_options is None:
        assembled = getattr(base_runtime, "assembled", None)
        module = base_runtime.model if assembled is None else assembled.diffusion
        if descriptor.runtime_with_family:
            runtime = type(base_runtime)(
                module,
                base_runtime.family,
                runtime_identity=composition.execution_identity,
            )
        else:
            runtime = type(base_runtime)(
                module,
                runtime_identity=composition.execution_identity,
                compute_dtype=_torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
            )
    else:
        runtime = _component_runtime_with_options(
            base_runtime,
            descriptor,
            recipe.family_id,
            composition.execution_identity,
            _torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
            sampling_shift,
            option_windows,
        )

    def prepare(carrier: object) -> object:
        prepare_method = getattr(runtime, descriptor.prepare_conditioning, None)
        prepare_options: dict[str, object] = {}
        if descriptor.frame_rate_conditioning:
            carrier, frame_rate = _split_ltx_frame_rate(carrier, inference)
            if frame_rate is not None:
                prepare_options["frame_rate"] = frame_rate
        conditioning = (
            execution_symbol(descriptor.prepare_conditioning)(carrier, device=handle.load_device)
            if prepare_method is None
            else prepare_method(carrier, **prepare_options)
        )
        if descriptor.conditioning_format == "raw":
            return conditioning
        if descriptor.conditioning_format == "multistream":
            prepared: list[list[Any]] = [
                [
                    inference.PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, conditioning
                    ),
                    {},
                ]
            ]
            return prepared
        return [[conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]

    result: tuple[Any, object, object] = (
        runtime,
        prepare(positive_carrier),
        (
            (None if descriptor.conditioning_format == "raw" else [])
            if negative_carrier is None
            else prepare(negative_carrier)
        ),
    )
    if descriptor.release_conditioning:
        try:
            handle.coordinator.advisory_unload_components(positive_binding.identity)
        except NativeResidencyBusyError:
            pass
    return _ResolvedComponentExecution(*result, conditioning_prepared=True)


def resolve_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object] | None:
    resolved = _resolve_component_execution(
        handle,
        positive,
        negative,
        inference,
        sampling_shift=sampling_shift,
        option_windows=option_windows,
        negative_handle=negative_handle,
        image_only_negative=image_only_negative,
    )
    return None if resolved is None else resolved.values()
