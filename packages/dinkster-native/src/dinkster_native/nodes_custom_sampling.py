"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Sequence

from .families.conditioning import _prepared_multistream_carrier
from .families.minimax_h3 import (
    _adapt_multistream_latent,
    _move_multistream_latent,
    _sampling_memory_requirements,
)
from .native_arm_core import (
    _NATIVE_MASK_BOUNDS_KEY,
    _NATIVE_MASK_KEY,
    Any,
    ExitStack,
    KSampler,
    Mapping,
    NativeRuntimeHandle,
    Node,
    NodeSchema,
    SamplerSelection,
    _condition_entries,
    _CustomGuiderValue,
    _CustomNoiseValue,
    _CustomSamplerValue,
    _CustomSigmasValue,
    _diffusion_unload_roles,
    _DualCFGGuiderValue,
    _DualModelGuiderValue,
    _effective_flux_guidance,
    _inference_registries,
    _LTXAVDualGuiderValue,
    _not_cancelled,
    _PerpNegGuiderValue,
    _split_flux_guidance,
    _torch,
    cast,
    current_execution_context,
    dataclass,
    importlib,
    math,
    multistream_sampling_preview_emitter,
    nullcontext,
    preview_stage,
    replace,
    report_progress,
    sampling_preview_emitter,
)
from .native_arm_latent_utils import _composite_masked_tensor, _plain_latent
from .native_arm_runtime import (
    _application_chain_model,
    _cfg1_optimization_setting,
    _classic_control_context,
    _conditioning,
    _conditioning_classic_control,
    _controlled_conditioning,
    _materialized_application_kwargs,
    _native_model,
    _native_model_sampling_cache,
    _native_model_sampling_space,
    _native_model_sampling_timeline,
    _require_classic_control_keyword,
    _sampling_space_runtime,
    _select_classic_control_binding,
    _uses_native_scheduling,
)
from .native_arm_scheduling import (
    _catalog_id,
    _NativeScheduleState,
    _scheduled_carrier,
)
from .nodes_provider import (
    _bind_sampling_shift,
    _generation_provider_schema,
    _materialize_provider_conditioning,
    _prepare_provider_conditioning,
    _prepare_provider_multistream_conditioning,
    _require_custom_sampling_runtime,
    _runtime_sampling_shift,
)
from .nodes_samplers import (
    _conditioning_batching_value,
    _custom_sampler_value,
)
from .nodes_sampling_runtime import (
    _batch_index_noise_inds,
    _materialize_z_image_control,
    _normalize_empty_latent,
    _resolve_sampling_model,
)


def _custom_sampling_conditioning(
    value: object,
    input_id: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    torch: Any,
) -> tuple[Any, Any]:
    runtime = handle.runtime
    prepared = (
        _prepare_provider_multistream_conditioning(value, input_id, runtime, inference)
        if isinstance(runtime, inference.MultiStreamConditioningRuntime)
        else (
            _prepare_provider_conditioning(value, input_id, runtime, inference)
            if isinstance(runtime, inference.ConditioningRuntime)
            else _materialize_provider_conditioning(value, input_id, handle)
        )
    )
    return _conditioning(prepared, input_id, torch, inference)


def _prepared_multistream_sampling_carrier(
    value: object,
    input_id: str,
    runtime: object,
    inference: Any,
    torch: Any,
) -> Any | None:
    if isinstance(value, inference.ResidentConditioningCarrier):
        return _prepared_multistream_carrier(value, inference, input_id)
    entries = _condition_entries(value, input_id)
    if not entries:
        return None
    if len(entries) == 1 and entries[0][1] == {}:
        return _prepared_multistream_carrier(value, inference, input_id)
    registration = getattr(runtime, "sampling_execution_registration", None)
    pipeline = getattr(registration, "pipeline", None)
    encode = getattr(pipeline, "encode_conditioning", None)
    if not callable(encode):
        raise TypeError(f"{input_id} runtime cannot schedule prepared conditioning")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    runtime_value = cast("Any", runtime)
    records: list[Any] = []
    bindings: list[Any] = []
    payloads: list[object] = []
    for index, entry in enumerate(entries):
        prepared = entry[0]
        metadata = cast("Mapping[object, object]", entry[1])
        if type(prepared) is not inference.PreparedMultiStreamConditioning:
            raise TypeError(f"{input_id} must contain prepared multi-stream conditioning")
        prepared_value = cast("Any", prepared)
        if prepared_value.runtime_identity != runtime_value.conditioning_identity:
            raise ValueError(f"{input_id} conditioning was prepared by a different runtime")
        unsupported = set(metadata) - {
            "start_percent",
            "end_percent",
            "strength",
            _NATIVE_MASK_KEY,
            _NATIVE_MASK_BOUNDS_KEY,
        }
        if unsupported:
            raise ValueError(
                f"{input_id} scheduled prepared metadata is unsupported: "
                + ", ".join(sorted(repr(key) for key in unsupported))
            )
        carrier = cast("Any", encode(prepared_value.payload, f"{input_id}-{index}"))
        if (
            type(carrier) is not inference.ConditioningCarrier
            or len(carrier.conditioning.records) != 1
        ):
            raise TypeError("prepared conditioning encoder must return one canonical record")
        start_value = metadata.get("start_percent", 0.0)
        end_value = metadata.get("end_percent", 1.0)
        strength_value = metadata.get("strength", 1.0)
        if any(type(item) not in (int, float) for item in (start_value, end_value, strength_value)):
            raise TypeError(f"{input_id} schedule bounds and strength must be numeric")
        start = float(cast("int | float", start_value))
        end = float(cast("int | float", end_value))
        strength = float(cast("int | float", strength_value))
        record = carrier.conditioning.records[0]
        mask = metadata.get(_NATIVE_MASK_KEY)
        mask_descriptor = None
        extra_bindings = ()
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{input_id} mask must be a torch.Tensor")
            mask = cast("Any", mask)
            mask_tensor = mask if mask.ndim >= 3 else mask.unsqueeze(0)
            mask_binding = inference_torch.tensor_to_payload_binding(
                f"{input_id}-{index}-mask",
                mask_tensor,
                space=inference_torch.MASK_PAYLOAD_SPACE,
            )
            mask_descriptor = inference.MaskDescriptor(
                inference.PayloadReference(mask_binding.reference_id),
                strength,
                metadata.get(_NATIVE_MASK_BOUNDS_KEY) is True,
            )
            extra_bindings = (mask_binding,)
        area = None
        if mask_descriptor is None and strength != 1.0:
            area = inference.AreaDescriptor(
                1.0,
                1.0,
                0.0,
                0.0,
                inference.AreaUnits.PERCENT,
                strength,
            )
        records.append(
            replace(
                record,
                area=area,
                mask=mask_descriptor,
                schedule=inference.PercentRange(start, end),
            )
        )
        bindings.extend((*carrier.bindings, *extra_bindings))
        payloads.append(prepared_value.payload)
    carrier = inference.make_conditioning_carrier(
        inference.ConditioningSet(tuple(records)), tuple(bindings)
    )
    return inference.PreparedMultiStreamConditioning(
        runtime_value.conditioning_identity,
        inference.PreparedConditioningCarrier(carrier, tuple(payloads)),
    )


def _custom_sampling_has_inpaint(value: object, input_id: str, inference: Any) -> bool:
    if isinstance(value, (inference.ConditioningCarrier, inference.ResidentConditioningCarrier)):
        return False
    return any(
        "concat_mask" in cast("Mapping[object, object]", entry[1])
        or "concat_latent_image" in cast("Mapping[object, object]", entry[1])
        for entry in _condition_entries(value, input_id)
    )


def _execute_generation_custom_sampling(
    *,
    model: object,
    model_negative: object | None = None,
    image_only_negative: bool = False,
    noise: _CustomNoiseValue,
    sampler: object,
    sigmas: _CustomSigmasValue,
    positive: object,
    negative: object | None,
    cfg: float,
    latent_image: object,
    middle: object | None = None,
    middle_scale: float = 1.0,
    nested_guidance: bool = False,
    audio_cfg: float | None = None,
    empty: object | None = None,
    neg_scale: float = 1.0,
    guider_transforms: tuple[tuple[str, object], ...] = (),
    conditioning_batching: object | None = None,
    denoise_mask: object | None = None,
    inpaint: object | None = None,
    negative_inpaint: object | None = None,
    noise_inds: Sequence[int] | None = None,
    context_windows: object | None = None,
) -> tuple[dict[object, object], dict[object, object]]:
    if type(noise) is not _CustomNoiseValue:
        raise TypeError("noise must come from RandomNoise or DisableNoise")
    inference = importlib.import_module("dinkster_inference")
    if inpaint is not None and type(inpaint) is not inference.InpaintConditioning:
        raise TypeError("inpaint must be an exact InpaintConditioning")
    if negative_inpaint is not None:
        if type(negative_inpaint) is not inference.InpaintConditioning:
            raise TypeError("negative_inpaint must be an exact InpaintConditioning")
        if negative is None:
            raise ValueError("negative_inpaint requires negative conditioning")
    if context_windows is not None and type(context_windows) is not inference.ContextWindowsSpec:
        raise TypeError("context_windows must be an exact ContextWindowsSpec")
    if conditioning_batching is None:
        conditioning_batching = inference.ConditioningBatching()
    elif type(conditioning_batching) is not inference.ConditioningBatching:
        raise TypeError("conditioning_batching must be an exact ConditioningBatching")
    if type(sampler) is SamplerSelection:
        context = current_execution_context()
        snapshot_digest = None if context is None else context.extension_snapshot_digest
        if sampler.extension_snapshot_digest != snapshot_digest:
            raise ValueError("sampler extension snapshot changed after KSamplerSelect executed")
        resolved = _custom_sampler_value(sampler.sampler_id, dict(sampler.options))
        if resolved.extension_ids != sampler.extension_ids:
            raise ValueError("sampler extension identities changed after KSamplerSelect executed")
        sampler = resolved
    elif type(sampler) is inference.BuiltinSamplerSelection:
        selection = cast("Any", sampler)
        sampler = _custom_sampler_value(
            selection.sampler_id,
            dict(selection.options),
        )
    if type(sampler) is not _CustomSamplerValue:
        raise TypeError("sampler must come from a Dinkster sampler-selection node")
    if type(sigmas) is not _CustomSigmasValue:
        raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
    classic_control_binding = (
        _conditioning_classic_control(positive, "conditioning")
        if negative is None
        else _select_classic_control_binding(positive, negative)
    )
    if empty is not None:
        empty_control_binding = _conditioning_classic_control(empty, "empty_conditioning")
        if classic_control_binding is None:
            if empty_control_binding is not None:
                raise ValueError(
                    "perp-neg classic ControlNet cannot be attached only to empty conditioning"
                )
        elif classic_control_binding.apply_to_uncond:
            if empty_control_binding not in (None, classic_control_binding):
                raise ValueError(
                    "perp-neg empty classic ControlNet chain must match the positive control"
                )
        elif empty_control_binding != classic_control_binding:
            raise ValueError(
                "perp-neg classic ControlNet must cover empty conditioning or use apply-to-uncond"
            )
    if middle is not None:
        middle_control_binding = _conditioning_classic_control(middle, "cond2")
        if classic_control_binding is None:
            if middle_control_binding is not None:
                raise ValueError("dual CFG classic ControlNet cannot be attached only to cond2")
        elif classic_control_binding.apply_to_uncond:
            if middle_control_binding not in (None, classic_control_binding):
                raise ValueError("dual CFG cond2 classic ControlNet chain must match cond1")
        elif middle_control_binding != classic_control_binding:
            raise ValueError("dual CFG classic ControlNet must cover cond2 or use apply-to-uncond")
    if (controlled := _controlled_conditioning(positive)) is not None:
        positive = controlled.conditioning
    if (controlled := _controlled_conditioning(negative)) is not None:
        negative = controlled.conditioning
    if (controlled := _controlled_conditioning(empty)) is not None:
        empty = controlled.conditioning
    if (controlled := _controlled_conditioning(middle)) is not None:
        middle = controlled.conditioning
    model, applications = _application_chain_model(model, "model")
    (
        handle,
        overlays,
        overlay_resolvers,
        z_image_control,
        sampling_shift,
        model_guidance_transforms,
        overlay_context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    if context_windows is not None and overlay_context_windows is not None:
        raise ValueError(
            "model context windows and explicit context_windows cannot both be provided"
        )
    if context_windows is None:
        context_windows = overlay_context_windows
    model_guidance_transforms, disable_cfg1_optimization = _cfg1_optimization_setting(
        model_guidance_transforms
    )
    guidance_transforms = (*model_guidance_transforms, *guider_transforms)
    negative_handle = None
    if model_negative is not None:
        negative_model, negative_applications = _application_chain_model(
            model_negative, "model_negative"
        )
        if negative_applications:
            raise ValueError("dual-model guidance does not accept negative model applications")
        (
            negative_handle,
            negative_overlays,
            negative_resolvers,
            negative_control,
            negative_sampling_shift,
            negative_transforms,
            negative_context_windows,
            negative_radiance_options,
        ) = _native_model(negative_model, "model_negative")
        if (
            negative_overlays
            or negative_resolvers
            or negative_control is not None
            or negative_sampling_shift is not None
            or _native_model_sampling_space(negative_model) is not None
            or negative_transforms
            or negative_context_windows is not None
            or negative_radiance_options
        ):
            raise ValueError("dual-model guidance does not accept a negative model overlay")
    positive, positive_guidance = _split_flux_guidance(positive)
    if negative is not None:
        negative, negative_guidance = _split_flux_guidance(negative)
    else:
        negative_guidance = None
    guidance = _effective_flux_guidance(positive_guidance, negative_guidance)
    if middle is not None:
        middle, middle_guidance = _split_flux_guidance(middle)
        guidance = _effective_flux_guidance(guidance, middle_guidance)
    if empty is not None:
        if negative is None:
            raise ValueError("perp-neg guidance requires negative conditioning")
        empty, empty_guidance = _split_flux_guidance(empty)
        if empty_guidance is not None:
            raise ValueError("perp-neg empty conditioning does not accept FluxGuidance")
    runtime, positive, negative, _component_execution, prepared_rows = _resolve_sampling_model(
        handle,
        positive,
        negative,
        inference,
        sampling_shift=sampling_shift,
        option_windows=chroma_radiance_options,
        negative_handle=negative_handle,
        image_only_negative=image_only_negative,
    )
    if prepared_rows and negative == []:
        negative = None
    if empty is not None and negative is None:
        raise ValueError("perp-neg guidance requires negative conditioning")
    if empty is not None and prepared_rows:
        empty_runtime, empty, _, _, _ = _resolve_sampling_model(
            handle,
            empty,
            None,
            inference,
            sampling_shift=sampling_shift,
            option_windows=chroma_radiance_options,
        )
        if empty_runtime.runtime_identity != runtime.runtime_identity:
            raise ValueError("empty conditioning must use the same component binding")
    if middle is not None and prepared_rows:
        middle_runtime, middle, _, _, _ = _resolve_sampling_model(
            handle,
            middle,
            None,
            inference,
            sampling_shift=sampling_shift,
            option_windows=chroma_radiance_options,
        )
        if middle_runtime.runtime_identity != runtime.runtime_identity:
            raise ValueError("cond2 must use the same component binding")
    runtime = _sampling_space_runtime(runtime, _native_model_sampling_space(model))
    if not isinstance(runtime, inference.CustomSamplingRuntime):
        raise TypeError(f"model family {runtime.family.id!r} does not support custom sampling")
    if audio_cfg is not None and getattr(runtime, "supports_audio_cfg", None) is not True:
        raise TypeError("LTXV Dual CFG Guider requires LTX-2 audio-video latent sampling")
    runtime_sampling_shift = _runtime_sampling_shift(runtime, sampling_shift)
    bound_sample_custom = _bind_sampling_shift(runtime.sample_custom, runtime_sampling_shift)
    positive_is_carrier = isinstance(positive, inference.ConditioningCarrier)
    negative_is_carrier = negative is not None and isinstance(
        negative, inference.ConditioningCarrier
    )
    if negative is not None and positive_is_carrier != negative_is_carrier:
        raise TypeError("positive and negative must use the same conditioning representation")
    empty_is_carrier = empty is not None and isinstance(empty, inference.ConditioningCarrier)
    if empty is not None and positive_is_carrier != empty_is_carrier:
        raise TypeError("positive and empty must use the same conditioning representation")
    middle_is_carrier = middle is not None and isinstance(middle, inference.ConditioningCarrier)
    if middle is not None and positive_is_carrier != middle_is_carrier:
        raise TypeError("cond1 and cond2 must use the same conditioning representation")
    if classic_control_binding is not None and z_image_control is not None:
        raise ValueError("classic and Z-Image ControlNet cannot be applied together")
    if classic_control_binding is not None:
        _require_classic_control_keyword(bound_sample_custom)
    context = current_execution_context()
    snapshot_digest = None if context is None else context.extension_snapshot_digest
    if sampler.extension_snapshot_digest != snapshot_digest:
        raise ValueError("sampler extension snapshot changed after KSamplerSelect executed")
    request = inference.CustomSamplingRequest(
        sampler.descriptor,
        cast("tuple[tuple[str, Any], ...]", sampler.options),
        sigmas.values,
        cache=cast("Any", _native_model_sampling_cache(model)),
        timeline=cast("Any", _native_model_sampling_timeline(model)),
        source_scheduler_id=sigmas.source_scheduler_id,
    )
    if not isinstance(latent_image, Mapping):
        raise TypeError("latent_image must be a mapping containing 'samples'")
    latent = cast("Mapping[object, object]", latent_image)
    inference_torch = importlib.import_module("dinkster_inference_torch")
    prepare_latent_kwargs = getattr(runtime, "custom_sampling_latent_kwargs", None)
    latent_kwargs: dict[str, object] = (
        {} if prepare_latent_kwargs is None else prepare_latent_kwargs(latent)
    )
    runtime_output_role = getattr(runtime, "dense_custom_sampling_role", None)
    dense_output_role = (
        runtime_output_role
        if type(latent.get("samples")) is not inference.MultiStreamLatent
        else None
    )
    noise_mask = latent.get("noise_mask")
    if denoise_mask is not None:
        if noise_mask is not None:
            raise ValueError(
                "latent_image noise_mask and explicit denoise_mask cannot both be provided"
            )
        noise_mask = denoise_mask
    has_inpaint = (
        inpaint is not None
        or negative_inpaint is not None
        or _custom_sampling_has_inpaint(positive, "positive", inference)
        or (negative is not None and _custom_sampling_has_inpaint(negative, "negative", inference))
        or (
            empty is not None
            and _custom_sampling_has_inpaint(empty, "empty_conditioning", inference)
        )
    )
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=noise_mask is not None,
        has_inpaint=has_inpaint,
        has_context_windows=context_windows is not None,
        guidance=guidance,
    )
    torch = _torch()
    # Single-stream family descriptors may still require structural execution;
    # the runtime-owned adapter declares that boundary for plain workflow latents.
    multistream_family = (
        type(runtime.family.latent) is inference.MultiStreamLatentDescriptor
        or isinstance(runtime, inference.MultiStreamLatentAdapterRuntime)
        or (
            type(latent.get("samples")) is inference.MultiStreamLatent
            and isinstance(runtime, inference.MultiStreamFamilyRuntime)
        )
    )
    if overlays and not multistream_family:
        raise ValueError(
            "custom sampling does not accept model overlays other than guidance transforms"
        )
    sparse_family = type(latent.get("samples")) is inference.SparseLatent
    if multistream_family:
        latent = _adapt_multistream_latent(latent, runtime, torch, inference, "latent_image")
    samples = latent.get("samples")
    if multistream_family:
        if type(samples) is not inference.MultiStreamLatent:
            raise TypeError("latent_image['samples'] must be an exact MultiStreamLatent")
        samples = cast("Any", samples)
    elif sparse_family:
        if type(samples) is not inference.SparseLatent:
            raise TypeError("latent_image['samples'] must be an exact SparseLatent")
        samples = cast("Any", samples)
    else:
        if type(samples) is not torch.Tensor:
            raise TypeError("latent_image['samples'] must be an exact torch.Tensor")
        samples = cast("Any", samples)
        samples = _normalize_empty_latent(samples, runtime, torch)
        descriptor = runtime.family.single_stream_latent()
        expected_rank = descriptor.dimensions + 2
        if samples.ndim != expected_rank:
            raise ValueError(f"custom sampling requires a rank-{expected_rank} latent")
        expected_channels = descriptor.channels
        if samples.shape[1] != expected_channels:
            if bool(torch.count_nonzero(samples)):
                raise ValueError(
                    "custom sampling latent channels must match the model unless the latent "
                    "is empty"
                )
            samples = torch.zeros(
                (samples.shape[0], expected_channels, *samples.shape[2:]),
                dtype=samples.dtype,
                layout=samples.layout,
                device=samples.device,
            )
    derived_noise_inds = _batch_index_noise_inds(latent)
    if noise_inds is not None:
        if isinstance(noise_inds, (str, bytes)):
            raise TypeError("noise_inds must be a sequence of integers")
        if any(type(index) is not int or index < 0 for index in noise_inds):
            raise ValueError("noise_inds values must be nonnegative integers")
        if derived_noise_inds is not None:
            raise ValueError(
                "latent_image batch_index and explicit noise_inds cannot both be provided"
            )
        noise_inds = tuple(noise_inds) or None
    else:
        noise_inds = derived_noise_inds
    if noise_mask is not None and type(noise_mask) is not torch.Tensor:
        if not (multistream_family and type(noise_mask) is inference.MultiStreamLatent):
            raise TypeError("latent_image['noise_mask'] must be an exact torch.Tensor")
    cond_inpaint: Any = None
    uncond_inpaint: Any = None
    empty_cond: Any = None
    empty_inpaint: Any = None
    middle_cond: Any = None
    middle_inpaint: Any = None
    schedule_state: _NativeScheduleState | None = None
    if multistream_family:

        def scheduled_lane(value: object, name: str) -> object:
            assert schedule_state is not None
            rows = (
                _prepare_provider_multistream_conditioning(value, name, runtime, inference)
                if isinstance(value, inference.ConditioningCarrier)
                else value
            )
            return _scheduled_carrier(rows, name, handle, schedule_state)

        scheduled = (
            bool(overlays)
            or _uses_native_scheduling(positive)
            or (negative is not None and _uses_native_scheduling(negative))
            or (empty is not None and _uses_native_scheduling(empty))
            or (middle is not None and _uses_native_scheduling(middle))
        )
        if scheduled:
            schedule_state = _NativeScheduleState(
                handle,
                inference,
                inference_torch,
                ordinary_overlays=overlays,
                ordinary_resolvers=overlay_resolvers,
            )

            cond = scheduled_lane(positive, "positive")
        elif positive_is_carrier:
            cond = _prepare_provider_multistream_conditioning(
                positive, "positive", runtime, inference
            )[0][0]
        else:
            cond = _prepared_multistream_sampling_carrier(
                positive, "positive", runtime, inference, torch
            )
            if cond is None:
                raise TypeError("positive must contain prepared multi-stream conditioning")
        cond_inpaint = None
        # The carrier resolves [] to None: an empty negative means no
        # guidance lane, exactly as on the KSampler path. Malformed
        # entries raise inside the carrier helper.
        if negative is None:
            uncond = None
        elif schedule_state is not None:
            uncond = scheduled_lane(negative, "negative")
        elif negative_is_carrier:
            uncond = _prepare_provider_multistream_conditioning(
                negative, "negative", runtime, inference
            )[0][0]
        else:
            uncond = _prepared_multistream_sampling_carrier(
                negative, "negative", runtime, inference, torch
            )
        uncond_inpaint = None
        if empty is not None:
            if schedule_state is not None:
                empty_cond = scheduled_lane(empty, "empty_conditioning")
            elif isinstance(empty, inference.ConditioningCarrier):
                empty_cond = _prepare_provider_multistream_conditioning(
                    empty, "empty_conditioning", runtime, inference
                )[0][0]
            else:
                empty_cond = _prepared_multistream_sampling_carrier(
                    empty, "empty_conditioning", runtime, inference, torch
                )
                if empty_cond is None:
                    raise TypeError("empty_conditioning must contain multi-stream conditioning")
        if middle is not None:
            if schedule_state is not None:
                middle_cond = scheduled_lane(middle, "cond2")
            elif isinstance(middle, inference.ConditioningCarrier):
                middle_cond = _prepare_provider_multistream_conditioning(
                    middle, "cond2", runtime, inference
                )[0][0]
            else:
                middle_cond = _prepared_multistream_sampling_carrier(
                    middle, "cond2", runtime, inference, torch
                )
                if middle_cond is None:
                    raise TypeError("cond2 must contain multi-stream conditioning")
    else:
        if prepared_rows:
            cond, cond_inpaint = _conditioning(positive, "positive", torch, inference)
        else:
            cond, cond_inpaint = _custom_sampling_conditioning(
                positive, "positive", handle, inference, torch
            )
        if negative is None:
            uncond = None
            uncond_inpaint = None
        elif prepared_rows:
            uncond, uncond_inpaint = _conditioning(negative, "negative", torch, inference)
        else:
            uncond, uncond_inpaint = _custom_sampling_conditioning(
                negative, "negative", handle, inference, torch
            )
        if empty is not None:
            if prepared_rows:
                empty_cond, empty_inpaint = _conditioning(
                    empty, "empty_conditioning", torch, inference
                )
            else:
                empty_cond, empty_inpaint = _custom_sampling_conditioning(
                    empty, "empty_conditioning", handle, inference, torch
                )
        if middle is not None:
            if prepared_rows:
                middle_cond, middle_inpaint = _conditioning(middle, "cond2", torch, inference)
            else:
                middle_cond, middle_inpaint = _custom_sampling_conditioning(
                    middle, "cond2", handle, inference, torch
                )
    if inpaint is not None:
        if cond_inpaint is not None:
            raise ValueError(
                "conditioning concat inpaint and explicit inpaint cannot both be provided"
            )
        cond_inpaint = inpaint
    if negative_inpaint is not None:
        if uncond_inpaint is not None:
            raise ValueError(
                "conditioning concat inpaint and explicit negative_inpaint cannot both be provided"
            )
        uncond_inpaint = negative_inpaint
    if negative is not None and (cond_inpaint is None) != (uncond_inpaint is None):
        raise ValueError("positive and negative inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and uncond_inpaint is not None
        and (
            cond_inpaint.mask is not uncond_inpaint.mask
            or cond_inpaint.masked_image is not uncond_inpaint.masked_image
        )
    ):
        raise ValueError("positive and negative inpaint conditioning must share concat values")
    if empty is not None and (cond_inpaint is None) != (empty_inpaint is None):
        raise ValueError("positive and empty inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and empty_inpaint is not None
        and (
            cond_inpaint.mask is not empty_inpaint.mask
            or cond_inpaint.masked_image is not empty_inpaint.masked_image
        )
    ):
        raise ValueError("positive and empty inpaint conditioning must share concat values")
    if middle is not None and (cond_inpaint is None) != (middle_inpaint is None):
        raise ValueError("cond1 and cond2 inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and middle_inpaint is not None
        and (
            cond_inpaint.mask is not middle_inpaint.mask
            or cond_inpaint.masked_image is not middle_inpaint.masked_image
        )
    ):
        raise ValueError("cond1 and cond2 inpaint conditioning must share concat values")
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=noise_mask is not None,
        has_inpaint=cond_inpaint is not None,
        has_context_windows=context_windows is not None,
        guidance=guidance,
    )
    control_kwargs: dict[str, object] = {}
    if z_image_control is not None:
        control_kwargs["control"] = _materialize_z_image_control(
            handle,
            z_image_control,
            samples,
            torch,
            inference,
            inference_torch,
        )
    seed = 0 if noise.seed is None else noise.seed
    if multistream_family:

        def _zero_stream(stream: Any) -> Any:
            return torch.zeros_like(stream, device="cpu")

        if noise.seed is None:
            generated_noise = samples.map(_zero_stream)
        else:
            prepare_stream_noise = getattr(
                runtime, "prepare_custom_sampling_noise", inference_torch.prepare_multistream_noise
            )
            generated_noise = prepare_stream_noise(samples, seed, noise_inds)
        samples = _move_multistream_latent(samples, handle.load_device)
        if type(noise_mask) is torch.Tensor:
            noise_mask = cast("Any", noise_mask).to(handle.load_device)
        elif type(noise_mask) is inference.MultiStreamLatent:
            noise_mask = _move_multistream_latent(noise_mask, handle.load_device)
        preview = (
            sampling_preview_emitter(handle, stream_role=runtime_output_role)
            if runtime_output_role is not None
            else multistream_sampling_preview_emitter(handle)
        )
    elif sparse_family:
        support, features = inference_torch.unpack_sparse_latent(samples)
        if noise.seed is None:
            noise_features = torch.zeros_like(features, device="cpu")
        elif noise_inds is None:
            noise_features = inference_torch.prepare_noise(features, seed)
        else:
            max_points = max(support.batch_counts)
            noise_batches = inference_torch.prepare_noise(
                features.new_empty((support.batch_size, max_points, features.shape[1])),
                seed,
                noise_inds,
            )
            noise_features = torch.cat(
                tuple(
                    noise_batches[batch, :count] for batch, count in enumerate(support.batch_counts)
                )
            )
        generated_noise = inference_torch.pack_sparse_latent(support, noise_features)
        preview = None
    else:
        generated_noise = (
            torch.zeros(
                samples.shape,
                dtype=samples.dtype,
                layout=samples.layout,
                device="cpu",
            )
            if noise.seed is None
            else inference_torch.prepare_noise(samples, seed, noise_inds)
        )
        preview = sampling_preview_emitter(handle)

    sampling_cfg = max(cfg, middle_scale) if middle_cond is not None else cfg
    if audio_cfg is not None:
        if type(samples) is not inference.MultiStreamLatent:
            raise TypeError("LTXV Dual CFG Guider requires LTX-2 audio-video latent sampling")
        if samples.roles != ("video", "audio"):
            raise TypeError("LTXV Dual CFG Guider requires video and audio latent streams")
        sampling_cfg = max(cfg, audio_cfg)
        if not math.isclose(cfg, audio_cfg):
            video = samples.by_role("video")
            video_elements = math.prod(video.shape[1:])
            guidance_transforms = (
                *guidance_transforms,
                (
                    "dinkster.ltxv_dual_cfg_guider",
                    inference_torch.ltxav_dual_cfg_guidance(
                        float(cfg), float(audio_cfg), video_elements
                    ),
                ),
            )

    def report_step(event: Any) -> None:
        report_progress(event.step + 1, event.total)

    if empty_cond is not None and uncond is None:
        raise ValueError("perp-neg guidance requires negative conditioning")
    if getattr(runtime, "video_vae_config", None) == inference.LTXAV_22B_V25_VAE_CONFIG:
        inference_torch.soft_empty_cache(handle.load_device)
    sampling_memory = _sampling_memory_requirements(runtime, samples)
    stage_negative = negative_handle is not None and (
        not math.isclose(sampling_cfg, 1.0)
        or cast("Any", sampler.descriptor).needs_uncond
        or bool(guidance_transforms)
        or disable_cfg1_optimization
    )
    with (
        _classic_control_context(
            classic_control_binding,
            latent_batch=int(samples.shape[0]) if classic_control_binding is not None else 1,
            torch=torch,
            inference=inference,
            inference_torch=inference_torch,
            base_handle=handle,
        ) as (classic_control, classic_control_handles),
        handle.stage(
            "diffusion",
            memory_required=sampling_memory[0],
            minimum_memory=sampling_memory[1],
            unload_before=_diffusion_unload_roles(handle),
        ),
        (
            negative_handle.stage("diffusion")
            if stage_negative and negative_handle is not None
            else nullcontext()
        ),
        ExitStack() as control_stages,
        (
            z_image_control.handle.stage(observer_stage="sample")
            if z_image_control is not None
            else nullcontext()
        ),
        preview_stage(preview),
        torch.no_grad() if schedule_state is not None else torch.inference_mode(),
        inference.use_sampling_environment(
            sampler.extension_ids, context.cancelled if context is not None else _not_cancelled
        ),
        _materialized_application_kwargs(
            applications,
            handle,
            samples,
            reserved_keys={
                "noise",
                "cond",
                "cfg",
                "request",
                "seed",
                "guidance",
                "denoise_mask",
                "inpaint",
                "noise_inds",
                "on_step",
                "on_state",
                "capture_denoised",
                "compute_dtype",
                "sampling_shift",
                "cancelled",
                "observer",
                "parent_span_id",
                "scheduled",
            },
        ) as application_kwargs,
    ):
        if schedule_state is not None:
            control_stages.callback(schedule_state.close)
        if classic_control is not None:
            control_kwargs["control"] = classic_control
        for control_handle in classic_control_handles:
            control_stages.enter_context(control_handle.stage(observer_stage="sample"))
        result = bound_sample_custom(
            samples,
            noise=generated_noise,
            cond=cond,
            # Every KSampler surface wraps SamplingGuidance unconditionally,
            # and CFG++ samplers consume the scale even without an uncond
            # payload, so the wrapper survives an absent negative here too.
            cfg=(
                inference.PerpNegSamplingGuidance(
                    uncond,
                    empty_cond,
                    sampling_cfg,
                    neg_scale,
                    transforms=guidance_transforms,
                    batching=conditioning_batching,
                    disable_cfg1_optimization=disable_cfg1_optimization,
                )
                if empty_cond is not None
                else (
                    inference.DualSamplingGuidance(
                        middle_cond,
                        uncond,
                        cfg,
                        middle_scale,
                        nested_guidance,
                        transforms=guidance_transforms,
                        batching=conditioning_batching,
                        disable_cfg1_optimization=disable_cfg1_optimization,
                    )
                    if middle_cond is not None
                    else inference.SamplingGuidance(
                        uncond,
                        sampling_cfg,
                        transforms=guidance_transforms,
                        batching=conditioning_batching,
                        disable_cfg1_optimization=disable_cfg1_optimization,
                    )
                )
            ),
            request=request,
            seed=seed,
            guidance=guidance,
            denoise_mask=cast("Any", noise_mask),
            inpaint=cond_inpaint,
            context_windows=context_windows,
            noise_inds=noise_inds,
            on_step=report_step,
            on_state=(preview.on_state if preview is not None else None),
            **(
                {}
                if schedule_state is None
                else {
                    "scheduled": inference_torch.ScheduledSamplingOptions(
                        schedule_state.resolve,
                        context.cancelled if context is not None else _not_cancelled,
                    )
                }
            ),
            **latent_kwargs,
            **control_kwargs,
            **application_kwargs,
        )
        if type(result) is not inference.CustomSamplingResult:
            raise TypeError("custom sampling runtime must return an exact CustomSamplingResult")
        if multistream_family:
            if (
                type(result.output) is not inference.MultiStreamLatent
                or result.output.roles != samples.roles
            ):
                raise TypeError("custom sampling must return the input latent streams")
            if result.denoised_output is not None and (
                type(result.denoised_output) is not inference.MultiStreamLatent
                or result.denoised_output.roles != samples.roles
            ):
                raise TypeError(
                    "custom sampling denoised output must match the input latent streams"
                )
        elif sparse_family:
            if type(result.output) is not inference.SparseLatent:
                raise TypeError("custom sampling must return an exact SparseLatent")
            if not result.output.support.same_support(samples.support):
                raise TypeError("custom sampling must preserve sparse support")
            if result.denoised_output is not None and (
                type(result.denoised_output) is not inference.SparseLatent
                or not result.denoised_output.support.same_support(samples.support)
            ):
                raise TypeError("custom sampling denoised output must preserve sparse support")

        def move_output(value: Any) -> Any:
            if multistream_family:
                if dense_output_role is not None:
                    return value.by_role(dense_output_role).to("cpu")
                return _move_multistream_latent(value, "cpu")
            if sparse_family:
                return value
            return value.to("cpu")

        result = inference.CustomSamplingResult(
            move_output(result.output),
            None if result.denoised_output is None else move_output(result.denoised_output),
        )
    output = dict(latent)
    if dense_output_role is None:
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
    if denoise_mask is not None:
        output["noise_mask"] = denoise_mask
    output["samples"] = result.output
    if result.denoised_output is None:
        denoised_output = dict(output)
    else:
        denoised_output = dict(latent)
        denoised_output["samples"] = result.denoised_output
    return output, denoised_output


class GenerationSamplerCustom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_custom")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: bool,
        noise_seed: int,
        cfg: float,
        positive: object,
        negative: object,
        sampler: object,
        sigmas: object,
        latent_image: object,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
        denoise_mask: object = None,
        inpaint: object = None,
        negative_inpaint: object = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: object = None,
    ) -> Mapping[str, object]:
        if type(add_noise) is not bool:
            raise TypeError("add_noise must be a Boolean")
        if not 0 <= noise_seed <= KSampler.MAX_SEED:
            raise ValueError(f"noise_seed must be in [0, {KSampler.MAX_SEED}], got {noise_seed}")
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        output, denoised_output = _execute_generation_custom_sampling(
            model=model,
            noise=_CustomNoiseValue(noise_seed if add_noise else None),
            sampler=cast("_CustomSamplerValue", sampler),
            sigmas=cast("_CustomSigmasValue", sigmas),
            positive=positive,
            negative=negative,
            cfg=cfg,
            latent_image=latent_image,
            conditioning_batching=_conditioning_batching_value(
                conditioning_batching, max_fused_lanes
            ),
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            negative_inpaint=negative_inpaint,
            noise_inds=noise_inds,
            context_windows=context_windows,
        )
        return cls.outputs(output=output, denoised_output=denoised_output)


class GenerationSamplerCustomAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_custom_advanced")

    @classmethod
    def execute(
        cls,
        *,
        noise: object,
        guider: object,
        sampler: object,
        sigmas: object,
        latent_image: object,
        denoise_mask: object = None,
        inpaint: object = None,
        negative_inpaint: object = None,
        noise_inds: Sequence[int] | None = None,
        context_windows: object = None,
    ) -> Mapping[str, object]:
        if type(guider) is _DualCFGGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.cond1,
                middle=guider.cond2,
                negative=guider.negative,
                cfg=guider.cfg_conds,
                middle_scale=guider.cfg_cond2_negative,
                nested_guidance=guider.nested,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
                denoise_mask=denoise_mask,
                inpaint=inpaint,
                negative_inpaint=negative_inpaint,
                noise_inds=noise_inds,
                context_windows=context_windows,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _DualModelGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                model_negative=guider.model_negative,
                image_only_negative=guider.negative is None,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.cfg,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
                denoise_mask=denoise_mask,
                inpaint=inpaint,
                negative_inpaint=negative_inpaint,
                noise_inds=noise_inds,
                context_windows=context_windows,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _LTXAVDualGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.video_cfg,
                audio_cfg=guider.audio_cfg,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
                denoise_mask=denoise_mask,
                inpaint=inpaint,
                negative_inpaint=negative_inpaint,
                noise_inds=noise_inds,
                context_windows=context_windows,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _PerpNegGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.cfg,
                latent_image=latent_image,
                empty=guider.empty,
                neg_scale=guider.neg_scale,
                conditioning_batching=guider.batching,
                denoise_mask=denoise_mask,
                inpaint=inpaint,
                negative_inpaint=negative_inpaint,
                noise_inds=noise_inds,
                context_windows=context_windows,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is not _CustomGuiderValue:
            raise TypeError("guider must come from a Dinkster guider node")
        typed_guider = guider
        output, denoised_output = _execute_generation_custom_sampling(
            model=typed_guider.model,
            noise=cast("_CustomNoiseValue", noise),
            sampler=cast("_CustomSamplerValue", sampler),
            sigmas=cast("_CustomSigmasValue", sigmas),
            positive=typed_guider.positive,
            negative=typed_guider.negative,
            cfg=typed_guider.cfg,
            latent_image=latent_image,
            guider_transforms=typed_guider.transforms,
            conditioning_batching=typed_guider.batching,
            denoise_mask=denoise_mask,
            inpaint=inpaint,
            negative_inpaint=negative_inpaint,
            noise_inds=noise_inds,
            context_windows=context_windows,
        )
        return cls.outputs(output=output, denoised_output=denoised_output)


@dataclass(frozen=True, slots=True)
class _ImpactRegionalProvider:
    model: object
    positive: object
    negative: object
    cfg: float
    sampler: _CustomSamplerValue
    sigmas: tuple[float, ...]


def _impact_regional_provider(
    value: object,
    *,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    steps: int,
    name: str,
) -> _ImpactRegionalProvider:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an Impact BASIC_PIPE tuple")
    typed_value = cast("tuple[object, ...]", value)
    if len(typed_value) != 5:
        raise TypeError(f"{name} must be an Impact BASIC_PIPE tuple")
    model, _clip, _vae, positive, negative = typed_value
    sampler = _custom_sampler_value(sampler_name)
    runtime, sampling_shift, device = _require_custom_sampling_runtime(
        model, "ImpactRegionalSampler"
    )
    inference = importlib.import_module("dinkster_inference")
    scheduler_id = _catalog_id(_inference_registries(inference).schedulers, scheduler, "scheduler")
    descriptor = cast("Any", sampler.descriptor)
    schedule_steps = steps + 1 if descriptor.discard_penultimate else steps
    build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sigmas, sampling_shift)
    sigmas = build_sigmas(scheduler_id, schedule_steps, 1.0, device=device)
    if descriptor.discard_penultimate:
        sigmas = (*sigmas[:-2], sigmas[-1])
    if len(sigmas) != steps + 1:
        raise ValueError(f"{name} schedule must contain {steps + 1} sigmas, got {len(sigmas)}")
    request = inference.CustomSamplingRequest(descriptor, sampler.options, sigmas)
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=True,
        has_inpaint=False,
        has_context_windows=False,
        guidance=None,
    )
    return _ImpactRegionalProvider(model, positive, negative, cfg, sampler, sigmas)


def _impact_regional_masks(
    mask: object,
    latent: object,
    overlap_factor: int,
) -> tuple[object, object]:
    torch = _torch()
    if type(mask) is not torch.Tensor:
        raise TypeError("mask must be an exact torch.Tensor")
    if not isinstance(latent, Mapping):
        raise TypeError("samples must be a LATENT mapping")
    samples = cast("Mapping[object, object]", latent).get("samples")
    if type(samples) is not torch.Tensor:
        raise TypeError("samples['samples'] must be an exact torch.Tensor")
    samples = cast("Any", samples)
    if samples.ndim != 4:
        raise ValueError("Impact regional sampling requires a rank-4 image latent")
    typed_mask = cast("Any", mask)
    combined_mask = torch.ceil(typed_mask.detach().cpu()).to(torch.int32)
    base_mask = torch.where(combined_mask == 0, 1.0, 0.0)
    source_mask = typed_mask.clone()
    if source_mask.ndim == 4:
        source_mask = source_mask.squeeze(0).squeeze(0)
    elif source_mask.ndim == 3:
        source_mask = source_mask.squeeze(0)
    if source_mask.ndim != 2:
        raise ValueError("Impact regional sampling requires a single 2D mask")
    width = source_mask.shape[1]
    height = source_mask.shape[0]
    resized = torch.nn.functional.interpolate(
        source_mask.reshape((-1, 1, height, width)).to(dtype=torch.float32),
        size=(width, height),
        mode="bilinear",
        align_corners=False,
    )
    region_mask = (
        resized
        if overlap_factor == 0
        else torch.clamp(
            torch.nn.functional.conv2d(
                resized.round(),
                torch.ones(
                    (1, 1, overlap_factor, overlap_factor),
                    device=resized.device,
                    dtype=resized.dtype,
                ),
                padding=math.ceil((overlap_factor - 1) / 2),
            ),
            0.0,
            1.0,
        )
    )
    return (
        base_mask,
        region_mask[:, :, :width, :height].round().squeeze(0).squeeze(0),
    )


def _run_impact_regional_pass(
    provider: _ImpactRegionalProvider,
    latent: Mapping[object, object],
    sigmas: tuple[float, ...],
    *,
    seed: int,
    add_noise: bool,
    mask: object | None,
) -> dict[object, object]:
    current = dict(latent)
    if mask is None:
        current.pop("noise_mask", None)
    else:
        current["noise_mask"] = mask
    output, _denoised = _execute_generation_custom_sampling(
        model=provider.model,
        noise=_CustomNoiseValue(seed if add_noise else None),
        sampler=provider.sampler,
        sigmas=_CustomSigmasValue(sigmas),
        positive=provider.positive,
        negative=provider.negative,
        cfg=provider.cfg,
        latent_image=current,
    )
    output.pop("noise_mask", None)
    return output


class GenerationImpactRegionalSampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.compat.impact_regional_sampler")

    @classmethod
    def execute(
        cls,
        *,
        base_basic_pipe: object,
        region_basic_pipe: object,
        mask: object,
        samples: object,
        seed: int,
        steps: int,
        base_only_steps: int,
        denoise: float,
        overlap_factor: int,
        restore_latent: bool,
        base_cfg: float,
        base_sampler_name: str,
        base_scheduler: str,
        region_cfg: float,
        region_sampler_name: str,
        region_scheduler: str,
    ) -> Mapping[str, object]:
        if not 0 <= seed <= KSampler.MAX_SEED:
            raise ValueError(f"seed must be in [0, {KSampler.MAX_SEED}], got {seed}")
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        if not 0 <= base_only_steps <= KSampler.MAX_STEPS:
            raise ValueError(
                f"base_only_steps must be in [0, {KSampler.MAX_STEPS}], got {base_only_steps}"
            )
        if not 0.0 < denoise <= 1.0:
            raise ValueError(f"denoise must be in (0.0, 1.0], got {denoise}")
        if not 0 <= overlap_factor <= KSampler.MAX_STEPS:
            raise ValueError(
                f"overlap_factor must be in [0, {KSampler.MAX_STEPS}], got {overlap_factor}"
            )
        if type(restore_latent) is not bool:
            raise TypeError("restore_latent must be a Boolean")
        for name, value in (("base_cfg", base_cfg), ("region_cfg", region_cfg)):
            if not 0.0 <= value <= KSampler.MAX_CFG:
                raise ValueError(f"{name} must be in [0.0, {KSampler.MAX_CFG}], got {value}")
        advanced_steps = int(steps / denoise)
        start_at_step = advanced_steps - steps
        base = _impact_regional_provider(
            base_basic_pipe,
            cfg=base_cfg,
            sampler_name=base_sampler_name,
            scheduler=base_scheduler,
            steps=advanced_steps,
            name="base_basic_pipe",
        )
        region = _impact_regional_provider(
            region_basic_pipe,
            cfg=region_cfg,
            sampler_name=region_sampler_name,
            scheduler=region_scheduler,
            steps=advanced_steps,
            name="region_basic_pipe",
        )
        base_mask, region_mask = _impact_regional_masks(mask, samples, overlap_factor)
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a LATENT mapping")
        current = dict(cast("Mapping[object, object]", samples))
        effective_base_only = min(base_only_steps, steps)
        if effective_base_only:
            current = _run_impact_regional_pass(
                base,
                current,
                base.sigmas[start_at_step : start_at_step + effective_base_only + 1],
                seed=seed,
                add_noise=True,
                mask=None,
            )
        add_noise = effective_base_only == 0
        for index in range(start_at_step + effective_base_only, advanced_steps):
            current = _run_impact_regional_pass(
                base,
                current,
                base.sigmas[index : index + 2],
                seed=seed,
                add_noise=add_noise,
                mask=base_mask,
            )
            base_latent = current
            regional = _run_impact_regional_pass(
                region,
                base_latent,
                region.sigmas[index : index + 2],
                seed=seed,
                add_noise=False,
                mask=region_mask,
            )
            if restore_latent:
                torch = _torch()
                output, destination = _plain_latent(base_latent, torch, "base latent")
                _, source = _plain_latent(regional, torch, "regional latent")
                output["samples"] = _composite_masked_tensor(
                    destination.clone(), source, 0, 0, region_mask, 8, False, torch
                )
                current = output
            else:
                current = regional
            add_noise = False
        current.pop("noise_mask", None)
        return cls.outputs(latent=current)
