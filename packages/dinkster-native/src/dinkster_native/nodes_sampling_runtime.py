"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .families.minimax_h3 import (
    _adapt_multistream_latent,
    _move_multistream_latent,
    _prepared_multistream_conditioning,
    _resolve_component_execution,
    _sampling_memory_requirements,
)
from .native_arm_core import (
    Any,
    ExitStack,
    KSampler,
    KSamplerAdvanced,
    Mapping,
    NativeComponentHandle,
    NativeRuntimeHandle,
    Sequence,
    VAEDecode,
    VAEEncode,
    _active_inference_registries,
    _component_bound_carrier,
    _condition_entries,
    _diffusion_unload_roles,
    _effective_flux_guidance,
    _inference_registries,
    _not_cancelled,
    _run_direct_vae,
    _sampler_registry,
    _split_flux_guidance,
    _torch,
    cast,
    current_execution_context,
    current_native_observer,
    importlib,
    math,
    multistream_sampling_preview_emitter,
    native_execution_span,
    preview_stage,
    report_progress,
    sampling_preview_emitter,
)
from .native_arm_runtime import (
    _application_chain_model,
    _application_kwargs,
    _cfg1_optimization_setting,
    _classic_control_context,
    _conditioning,
    _context_windows_sampling_kwargs,
    _controlled_conditioning,
    _ControlledConditioning,
    _materialized_application_kwargs,
    _native_handle,
    _native_model,
    _native_model_sampling_space,
    _require_classic_control_keyword,
    _sampling_space_runtime,
    _scheduled_inpaint,
    _select_classic_control_binding,
    _staged_applications,
    _uses_native_scheduling,
    _ZImageControlBinding,
)
from .native_arm_scheduling import (
    _catalog_id,
    _compute_dtype,
    _NativeScheduleState,
    _scheduled_carrier,
)
from .nodes_provider import (
    _materialize_provider_conditioning,
    _prepare_provider_conditioning,
    _prepare_provider_multistream_conditioning,
    _runtime_sampling_shift,
)

if TYPE_CHECKING:
    from dinkster_inference import SamplingSegment


class _RegisteredComponentCodec:
    def __init__(self, value: object, codec: Any) -> None:
        registry = importlib.import_module("dinkster_native.family_registry")
        self._codec = codec
        self._decode = registry.registered_callable(value, "native_decode")
        self._encode = registry.registered_callable(value, "native_encode")

    @property
    def descriptor(self) -> Any:
        return self._codec.descriptor

    @property
    def resource_identity(self) -> str:
        return cast("str", self._codec.resource_identity)

    @property
    def load_device(self) -> object:
        return self._codec.load_device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._codec, name)

    def require_active(self) -> None:
        self._codec.require_active()

    def stage(self, *args: Any, **kwargs: Any) -> Any:
        return self._codec.stage(*args, **kwargs)

    def decode_latent(self, latent: Any) -> Any:
        return self._decode(self._codec, latent)

    def decode_latent_tiled(
        self,
        latent: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._codec.decode_latent_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._encode(self._codec, content)

    def encode_content_tiled(
        self,
        content: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._codec.encode_content_tiled(content, tile=tile, overlap=overlap)


def _native_component_codec(value: object) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    recipe = getattr(value, "recipe", None)
    family_id = getattr(recipe, "family_id", None)
    descriptor = (
        None if family_id is None else _active_inference_registries().components.get(family_id)
    )
    if descriptor is None or descriptor.codec_adapter is None:
        roles = tuple(source.role for source in getattr(recipe, "sources", ()))
        raise TypeError(
            f"no declared codec adapter; detected family={family_id!r}, roles={roles!r}"
        )
    return _RegisteredComponentCodec(value, execution_symbol(descriptor.codec_adapter)(value))


def _z_image_control_latent(
    handle: NativeRuntimeHandle,
    binding: _ZImageControlBinding,
    samples: Any,
    torch: Any,
    inference_torch: Any,
) -> Any:
    image = cast("Any", binding.image)
    downscale = handle.runtime.assembled.vae.config.spatial_downscale
    target_height = samples.shape[-2] * downscale
    target_width = samples.shape[-1] * downscale
    if image.shape[-2:] != (target_height, target_width):
        old_height, old_width = image.shape[-2:]
        old_aspect = old_width / old_height
        new_aspect = target_width / target_height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        image = image.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        image = torch.nn.functional.interpolate(
            image,
            size=(target_height, target_width),
            mode="area",
        )
    with handle.stage("vae", observer_stage="encode"):
        with torch.inference_mode():
            encoded = handle.runtime.codec.encode(image.to(handle.load_device))
    return inference_torch.latent_process_in(encoded, handle.runtime.family.latent)


def _materialize_z_image_control(
    handle: NativeRuntimeHandle,
    binding: _ZImageControlBinding,
    samples: Any,
    torch: Any,
    inference: Any,
    inference_torch: Any,
) -> Any:
    control_latent = _z_image_control_latent(handle, binding, samples, torch, inference_torch)
    hint_digest = inference_torch.z_image_control_hint_digest(control_latent)
    control_model = cast("Any", binding.handle.module)
    model_digest = control_model.resource_digest
    if model_digest is None:
        raise RuntimeError("Z-Image control patch has no assembly provenance")
    return inference_torch.ZImageControlConditioning(
        inference.ControlApplication(
            "z-image-fun",
            inference.PayloadReference(hint_digest),
            binding.strength,
            inference.PercentRange(0.0, 1.0),
        ),
        control_model,
        control_latent,
        model_digest,
        hint_digest,
    )


def _normalize_empty_latent(samples: Any, runtime: object, torch: Any) -> Any:
    """Match ComfyUI's empty-latent channel and rank normalization."""

    family = getattr(runtime, "family", None)
    if family is None:
        return samples
    single_stream_latent = getattr(family, "single_stream_latent", None)
    if single_stream_latent is None:
        return samples
    try:
        descriptor = single_stream_latent()
    except ValueError:
        return samples
    empty = bool(torch.count_nonzero(samples) == 0)
    if empty and samples.shape[1] != descriptor.channels:
        if samples.shape[1] < 1:
            raise ValueError("empty latent must have at least one channel")
        repeats = [1] * samples.ndim
        repeats[1] = math.ceil(descriptor.channels / samples.shape[1])
        samples = samples.repeat(*repeats).narrow(1, 0, descriptor.channels)
    if descriptor.dimensions == 3 and samples.ndim == 4:
        samples = samples.unsqueeze(2)
    return samples


def _batch_index_noise_inds(latent: Mapping[object, object]) -> tuple[int, ...] | None:
    """Parse a latent's ``batch_index`` into prepare_noise batch indices."""

    batch_index = latent.get("batch_index")
    if batch_index is None:
        return None
    if not isinstance(batch_index, Sequence) or isinstance(batch_index, (str, bytes)):
        raise TypeError("latent_image['batch_index'] must be a sequence of integers")
    raw_noise_inds = cast("Sequence[object]", batch_index)
    if any(type(index) is not int or index < 0 for index in raw_noise_inds):
        raise ValueError("latent_image['batch_index'] values must be nonnegative integers")
    return tuple(cast("int", index) for index in raw_noise_inds)


def _resolve_sampling_model(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object, bool, bool]:
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
    if resolved is None:
        if negative_handle is not None or image_only_negative:
            raise TypeError("runtime does not support separate-model or image-only guidance")
        runtime = handle.runtime
        if isinstance(runtime, inference.MultiStreamConditioningRuntime):
            positive, positive_binding = _component_bound_carrier(positive, inference)
            if negative not in ([], None):
                negative, negative_binding = _component_bound_carrier(negative, inference)
                if positive_binding != negative_binding:
                    raise ValueError("conditioning lanes must share one component binding")
        return runtime, positive, negative, False, False
    runtime, positive, negative = resolved.values()
    descriptor = _active_inference_registries().components.get(handle.recipe.family_id)
    if (
        resolved.conditioning_prepared
        and descriptor is not None
        and descriptor.conditioning_format == "raw"
    ):
        positive = [
            [
                inference.PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
                dict[str, object](),
            ]
        ]
        negative = (
            []
            if negative is None
            else [
                [
                    inference.PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, negative
                    ),
                    dict[str, object](),
                ]
            ]
        )
    return runtime, positive, negative, True, resolved.conditioning_prepared


def _prepare_ksampler_conditioning(
    value: object, name: str, runtime: Any, handle: NativeRuntimeHandle, inference: Any
) -> object:
    if not isinstance(value, inference.ConditioningCarrier):
        return value
    if isinstance(runtime, inference.MultiStreamConditioningRuntime):
        return _prepare_provider_multistream_conditioning(value, name, runtime, inference)
    if isinstance(runtime, inference.ConditioningRuntime):
        return _prepare_provider_conditioning(value, name, runtime, inference)
    if isinstance(runtime, inference.MultiStreamFamilyRuntime):
        raise TypeError(f"{name} runtime cannot prepare canonical conditioning")
    return _materialize_provider_conditioning(value, name, handle)


class NativeKSampler(KSampler):
    """Sample through the runtime's builtin Comfy-compatible catalogs."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        denoise: float,
        segment: SamplingSegment | None = None,
        conditioning_batching: object | None = None,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("seed", seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("denoise", denoise, 0.0, 1.0),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")

        model, applications = _application_chain_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        if conditioning_batching is None:
            conditioning_batching = inference.ConditioningBatching()
        elif type(conditioning_batching) is not inference.ConditioningBatching:
            raise TypeError("conditioning_batching must be an exact ConditioningBatching")
        positive, positive_guidance = _split_flux_guidance(positive)
        negative, negative_guidance = _split_flux_guidance(negative)
        positive_controlled = _controlled_conditioning(positive)
        negative_controlled = _controlled_conditioning(negative)
        if positive_controlled is not None:
            positive = positive_controlled.conditioning
        if negative_controlled is not None:
            negative = negative_controlled.conditioning
        guidance = _effective_flux_guidance(positive_guidance, negative_guidance)
        (
            handle,
            ordinary_overlays,
            ordinary_resolvers,
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        guidance_transforms, disable_cfg1_optimization = _cfg1_optimization_setting(
            guidance_transforms
        )
        runtime, positive, negative, component_execution, conditioning_prepared = (
            _resolve_sampling_model(
                handle,
                positive,
                negative,
                inference,
                sampling_shift=sampling_shift,
                option_windows=chroma_radiance_options,
            )
        )
        component_runtime = runtime if component_execution else None
        if not conditioning_prepared:
            positive = _prepare_ksampler_conditioning(
                positive, "positive", runtime, handle, inference
            )
            negative = _prepare_ksampler_conditioning(
                negative, "negative", runtime, handle, inference
            )

        def restore_control(
            value: object,
            controlled: _ControlledConditioning | None,
            input_id: str,
        ) -> object:
            if controlled is None:
                return value
            result: list[list[object]] = []
            for raw_entry in _condition_entries(value, input_id):
                metadata = dict(cast("Mapping[object, object]", raw_entry[1]))
                metadata["control"] = controlled.binding
                metadata["control_apply_to_uncond"] = controlled.binding.apply_to_uncond
                result.append([raw_entry[0], metadata])
            return result

        positive = restore_control(positive, positive_controlled, "positive")
        negative = restore_control(negative, negative_controlled, "negative")
        latent_mapping = (
            cast("Mapping[object, object]", latent_image)
            if isinstance(latent_image, Mapping)
            else None
        )
        latent_samples = None if latent_mapping is None else latent_mapping.get("samples")
        structural_latent = type(latent_samples) is inference.MultiStreamLatent
        sparse_latent = type(latent_samples) is inference.SparseLatent
        torch = _torch()
        active_runtime = component_runtime if component_runtime is not None else handle.runtime
        runtime_family = getattr(getattr(active_runtime, "family", None), "id", None)
        unload_text_before_diffusion = _diffusion_unload_roles(handle)
        classic_control_binding = _select_classic_control_binding(positive, negative)
        if classic_control_binding is not None:
            if z_image_control is not None:
                raise ValueError("classic and Z-Image ControlNet cannot be applied together")
        plain_latent = not structural_latent
        if latent_mapping is not None and isinstance(
            active_runtime,
            (inference.MultiStreamLatentAdapterRuntime, inference.MultiStreamFamilyRuntime),
        ):
            latent_mapping = _adapt_multistream_latent(
                latent_mapping, active_runtime, torch, inference, "latent_image"
            )
            latent_samples = latent_mapping["samples"]
            structural_latent = type(latent_samples) is inference.MultiStreamLatent
        if (
            latent_mapping is not None
            and structural_latent
            and isinstance(active_runtime, inference.MultiStreamFamilyRuntime)
        ):
            runtime = active_runtime
            if z_image_control is not None:
                raise TypeError("multi-stream sampling does not accept Z-Image ControlNet")
            noise_inds = _batch_index_noise_inds(latent_mapping)
            noise_kwargs = {} if noise_inds is None else {"noise_inds": noise_inds}
            runtime_sampling_shift = _runtime_sampling_shift(runtime, sampling_shift)
            context_windows_kwargs = _context_windows_sampling_kwargs(
                runtime, context_windows, f"{runtime_family!r} multi-stream sampling"
            )
            plain_output = plain_latent and len(cast("Any", latent_samples).roles) == 1
            denoise_mask = latent_mapping.get("noise_mask")
            if denoise_mask is not None:
                if type(denoise_mask) is torch.Tensor:
                    if not plain_output:
                        denoise_mask = cast("Any", denoise_mask).to(handle.load_device)
                elif type(denoise_mask) is inference.MultiStreamLatent:
                    if not plain_output:
                        denoise_mask = _move_multistream_latent(denoise_mask, handle.load_device)
                else:
                    raise TypeError("multi-stream noise_mask must be a tensor or MultiStreamLatent")
            scheduled = (
                bool(ordinary_overlays)
                or _uses_native_scheduling(positive)
                or _uses_native_scheduling(negative)
            )
            schedule_state = None
            inference_torch: Any = None
            uncond_envelope: Any = None
            if scheduled:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                schedule_state = _NativeScheduleState(
                    handle,
                    inference,
                    inference_torch,
                    ordinary_overlays=ordinary_overlays,
                    ordinary_resolvers=ordinary_resolvers,
                )
                try:
                    prepared_envelope = _scheduled_carrier(
                        positive, "positive", handle, schedule_state
                    )
                    prepared = cast("Any", prepared_envelope).payload
                    uncond_envelope = (
                        None
                        if negative == []
                        else _scheduled_carrier(negative, "negative", handle, schedule_state)
                    )
                except BaseException:
                    schedule_state.close()
                    raise
            else:
                prepared = _prepared_multistream_conditioning(
                    positive, inference, "positive", runtime.conditioning_identity
                )
            if prepared is None:
                raise TypeError("positive must contain prepared multi-stream conditioning")
            if negative == []:
                uncond = None
            elif scheduled:
                assert uncond_envelope is not None
                uncond = uncond_envelope.payload
            else:
                uncond = _prepared_multistream_conditioning(
                    negative, inference, "negative", runtime.conditioning_identity
                )
            context = current_execution_context()
            sampler_registry, extension_ids, _ = _sampler_registry(
                inference,
                context.extension_snapshot_digest if context is not None else None,
            )
            sampler_id = _catalog_id(sampler_registry, sampler_name, "sampler")
            scheduler_id = _catalog_id(
                _inference_registries(inference).schedulers, scheduler, "scheduler"
            )
            check_custom_sampling = getattr(runtime, "check_custom_sampling", None)
            if check_custom_sampling is not None:
                check_custom_sampling(
                    inference.CustomSamplingRequest(sampler_registry.get(sampler_id), (), ()),
                    has_denoise_mask=denoise_mask is not None,
                    has_inpaint=False,
                    has_context_windows=context_windows is not None,
                    guidance=guidance,
                )

            def report_multistream_step(event: Any) -> None:
                report_progress(event.step + 1, event.total)

            load_streams = (
                cast("Any", latent_samples)
                if plain_output
                else _move_multistream_latent(cast("Any", latent_samples), handle.load_device)
            )
            preview = (
                sampling_preview_emitter(handle, stream_role=load_streams.roles[0])
                if plain_output
                else multistream_sampling_preview_emitter(handle)
            )
            run_ksampler_as_custom = runtime.run_ksampler_as_custom
            reserved_application_kwargs = {
                "conditioning",
                "cfg",
                "sampler_id",
                "scheduler_id",
                "steps",
                "denoise",
                "seed",
                "segment",
                "denoise_mask",
                "noise_inds",
                "on_step",
                "on_state",
                "cancelled",
                "observer",
                "parent_span_id",
                "sampling_shift",
                "scheduled",
                *context_windows_kwargs,
            }
            with native_execution_span("sample", "sample", device=str(handle.load_device)) as span:
                parent = None if span is None else span.span_id
                with (
                    inference.use_sampling_environment(
                        extension_ids, context.cancelled if context is not None else _not_cancelled
                    ),
                    _staged_applications(applications, handle, stage_runtime=False),
                    ExitStack() as schedule_cleanup,
                ):
                    if schedule_state is not None:
                        schedule_cleanup.callback(schedule_state.close)
                    with torch.no_grad() if schedule_state is not None else torch.inference_mode():
                        application_kwargs = _application_kwargs(
                            applications,
                            handle,
                            load_streams,
                            reserved_keys=reserved_application_kwargs,
                        )
                        with (
                            handle.stage(
                                "diffusion",
                                unload_before=unload_text_before_diffusion,
                                observer_stage="sample",
                                parent_span_id=parent,
                            ),
                            preview_stage(preview),
                        ):
                            result = run_ksampler_as_custom(
                                load_streams,
                                conditioning=prepared,
                                cfg=inference.SamplingGuidance(
                                    uncond,
                                    cfg,
                                    transforms=guidance_transforms,
                                    batching=conditioning_batching,
                                    disable_cfg1_optimization=disable_cfg1_optimization,
                                ),
                                sampler_id=sampler_id,
                                scheduler_id=scheduler_id,
                                steps=steps,
                                denoise=denoise,
                                seed=seed,
                                segment=segment,
                                denoise_mask=denoise_mask,
                                sampling_shift=runtime_sampling_shift,
                                on_step=report_multistream_step,
                                on_state=(preview.on_state if preview is not None else None),
                                cancelled=context.cancelled
                                if context is not None
                                else _not_cancelled,
                                observer=current_native_observer(),
                                parent_span_id=parent,
                                **(
                                    {}
                                    if schedule_state is None
                                    else {
                                        "scheduled": inference_torch.ScheduledSamplingOptions(
                                            schedule_state.resolve,
                                            context.cancelled
                                            if context is not None
                                            else _not_cancelled,
                                        )
                                    }
                                ),
                                **noise_kwargs,
                                **context_windows_kwargs,
                                **application_kwargs,
                            )
            if (
                type(result) is not inference.MultiStreamLatent
                or result.roles != load_streams.roles
            ):
                raise TypeError("multi-stream sampling must return the input latent streams")
            output: dict[object, object] = dict(latent_mapping)
            if isinstance(runtime, inference.MultiStreamLatentAdapterRuntime) and not plain_output:
                output.pop("downscale_ratio_spacial", None)
                output.pop("downscale_ratio_temporal", None)
            output["samples"] = result.by_role(result.roles[0]) if plain_output else result
            return cls.outputs(latent=output)
        if structural_latent:
            raise TypeError("model does not provide multi-stream sampling")
        context = current_execution_context()
        sampler_registry, extension_ids, _ = _sampler_registry(
            inference,
            context.extension_snapshot_digest if context is not None else None,
        )
        scheduled = (
            bool(ordinary_overlays)
            or _uses_native_scheduling(positive)
            or _uses_native_scheduling(negative)
        )
        schedule_state = None
        result: Any = None
        try:
            if scheduled:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                schedule_state = _NativeScheduleState(
                    handle,
                    inference,
                    inference_torch,
                    ordinary_overlays=ordinary_overlays,
                    ordinary_resolvers=ordinary_resolvers,
                )
                cond = _scheduled_carrier(positive, "positive", handle, schedule_state)
                uncond = _scheduled_carrier(negative, "negative", handle, schedule_state)
                cond_inpaint = _scheduled_inpaint(positive, "positive", torch, inference)
                uncond_inpaint = _scheduled_inpaint(negative, "negative", torch, inference)
            else:
                cond, cond_inpaint = _conditioning(positive, "positive", torch, inference)
                uncond, uncond_inpaint = _conditioning(negative, "negative", torch, inference)
            if (cond_inpaint is None) != (uncond_inpaint is None):
                raise ValueError("positive and negative inpaint conditioning must both be present")
            if (
                cond_inpaint is not None
                and uncond_inpaint is not None
                and (
                    cond_inpaint.mask is not uncond_inpaint.mask
                    or cond_inpaint.masked_image is not uncond_inpaint.masked_image
                )
            ):
                raise ValueError(
                    "positive and negative inpaint conditioning must share concat values"
                )
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            latent = cast("Mapping[object, object]", latent_image)
            noise_inds = _batch_index_noise_inds(latent)
            samples_obj = latent.get("samples")
            if sparse_latent:
                samples: Any = samples_obj
            else:
                if not isinstance(samples_obj, torch.Tensor):
                    raise TypeError(
                        "latent_image['samples'] must be a torch.Tensor or SparseLatent"
                    )
                samples = _normalize_empty_latent(cast("Any", samples_obj), active_runtime, torch)
            noise_mask_obj = latent.get("noise_mask")
            if noise_mask_obj is not None and not isinstance(noise_mask_obj, torch.Tensor):
                raise TypeError("latent_image['noise_mask'] must be a torch.Tensor")
            noise_mask = cast("Any", noise_mask_obj)
            sampler_id = _catalog_id(sampler_registry, sampler_name, "sampler")
            scheduler_id = _catalog_id(
                _inference_registries(inference).schedulers, scheduler, "scheduler"
            )
            active_runtime = component_runtime if component_runtime is not None else handle.runtime
            sampling_space = _native_model_sampling_space(model)
            active_runtime = _sampling_space_runtime(active_runtime, sampling_space)
            control_kwargs: dict[str, object] = {}
            classic_control_handles: tuple[NativeComponentHandle, ...] = ()
            if classic_control_binding is not None:
                if schedule_state is not None:
                    _require_classic_control_keyword(active_runtime.sample_scheduled)
                else:
                    for method_name in ("sample", "sample_custom"):
                        receiver = getattr(active_runtime, method_name, None)
                        if receiver is not None:
                            _require_classic_control_keyword(receiver)
            if z_image_control is not None:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                control_kwargs = {
                    "control": _materialize_z_image_control(
                        handle,
                        z_image_control,
                        samples,
                        torch,
                        inference,
                        inference_torch,
                    )
                }
            # sample_scheduled has no on_state seam, and sparse latents have no preview provider.
            preview = (
                sampling_preview_emitter(handle)
                if schedule_state is None and not sparse_latent
                else None
            )
            if context_windows is not None:
                active_runtime.check_custom_sampling(
                    inference.CustomSamplingRequest(sampler_registry.get(sampler_id), (), ()),
                    has_denoise_mask=noise_mask is not None,
                    has_inpaint=cond_inpaint is not None,
                    has_context_windows=True,
                    guidance=guidance,
                )
            sampling_memory = _sampling_memory_requirements(active_runtime, samples)
            with ExitStack() as stages:
                if classic_control_binding is not None:
                    inference_torch = importlib.import_module("dinkster_inference_torch")
                    classic_control, classic_control_handles = stages.enter_context(
                        _classic_control_context(
                            classic_control_binding,
                            latent_batch=int(samples.shape[0]),
                            torch=torch,
                            inference=inference,
                            inference_torch=inference_torch,
                            base_handle=handle,
                        )
                    )
                    control_kwargs["control"] = classic_control
                stages.enter_context(
                    handle.stage(
                        "diffusion",
                        memory_required=sampling_memory[0],
                        minimum_memory=sampling_memory[1],
                        unload_before=unload_text_before_diffusion,
                    )
                )
                if z_image_control is not None:
                    stages.enter_context(z_image_control.handle.stage(observer_stage="sample"))
                for control_handle in classic_control_handles:
                    stages.enter_context(control_handle.stage(observer_stage="sample"))
                stages.enter_context(preview_stage(preview))
                sampling_context = (
                    torch.no_grad() if schedule_state is not None else torch.inference_mode()
                )
                with sampling_context:
                    with (
                        inference.use_sampling_environment(
                            extension_ids,
                            context.cancelled if context is not None else _not_cancelled,
                        ),
                    ):
                        sample: Any = None
                        custom_only = isinstance(
                            active_runtime, inference.CustomSamplingRuntime
                        ) and not callable(getattr(active_runtime, "sample", None))
                        custom_sampling_only = custom_only or bool(
                            getattr(active_runtime, "custom_sampling_only", False)
                        )
                        continue_sampling = not custom_only
                        if custom_sampling_only and (schedule_state is not None or applications):
                            raise ValueError(
                                "custom sampling runtimes do not accept model patches or schedules"
                            )
                        if custom_only:

                            def report_custom_step(event: Any) -> None:
                                report_progress(event.step + 1, event.total)

                            sampling = active_runtime.family.sampling
                            if not inference.is_flow_parameterization(sampling.parameterization):
                                raise ValueError(
                                    "custom-only KSampler runtimes require flow parameterization"
                                )
                            runtime_sampling_shift = _runtime_sampling_shift(
                                active_runtime, sampling_shift
                            )
                            space = sampling_space or inference.FlowSigmas(
                                shift=(sampling.shift if sampling_shift is None else sampling_shift)
                            )
                            custom_result = importlib.import_module(
                                "dinkster_inference_torch.sampling_execution"
                            ).run_ksampler_as_custom(
                                active_runtime,
                                (samples if sparse_latent else samples.to(handle.load_device)),
                                samplers=sampler_registry,
                                schedulers=importlib.import_module(
                                    "dinkster_inference_torch"
                                ).torch_scheduler_registry(),
                                space=space,
                                flow=True,
                                device=handle.load_device,
                                sampler_id=sampler_id,
                                scheduler_id=scheduler_id,
                                steps=steps,
                                denoise=denoise,
                                seed=seed,
                                cond=cond,
                                cfg=inference.SamplingGuidance(
                                    uncond,
                                    cfg,
                                    transforms=guidance_transforms,
                                    batching=conditioning_batching,
                                    disable_cfg1_optimization=disable_cfg1_optimization,
                                ),
                                guidance=guidance,
                                segment=segment,
                                denoise_mask=noise_mask,
                                inpaint=cond_inpaint,
                                noise_inds=noise_inds,
                                context_windows=context_windows,
                                on_step=report_custom_step,
                                on_state=(preview.on_state if preview is not None else None),
                                sample_custom_kwargs={
                                    **control_kwargs,
                                    **(
                                        {}
                                        if runtime_sampling_shift is None
                                        else {"sampling_shift": runtime_sampling_shift}
                                    ),
                                },
                                error=ValueError,
                            )
                            result = custom_result.output
                            continue_sampling = False
                        if continue_sampling:
                            sample = (
                                active_runtime.sample_scheduled
                                if schedule_state is not None
                                else active_runtime.sample
                            )
                        family_conditioning = isinstance(
                            active_runtime, inference.ConditioningRuntime
                        )
                        kwargs: dict[str, object] = {
                            "cond": cond,
                            "cfg": inference.SamplingGuidance(
                                uncond,
                                cfg,
                                transforms=guidance_transforms,
                                batching=conditioning_batching,
                                disable_cfg1_optimization=disable_cfg1_optimization,
                            ),
                            "sampler_id": sampler_id,
                            "scheduler_id": scheduler_id,
                            "steps": steps,
                            "denoise": denoise,
                            "seed": seed,
                            "guidance": guidance,
                            "denoise_mask": noise_mask,
                            "inpaint": cond_inpaint,
                            "segment": segment,
                            "schedule_device": handle.load_device,
                            **control_kwargs,
                        }
                        # Only a non-None value is forwarded: runtimes
                        # without the parameter keep working on the
                        # default path and refuse a real request with a
                        # TypeError instead of dropping it.
                        if noise_inds is not None:
                            kwargs["noise_inds"] = noise_inds
                        if context_windows is not None:
                            kwargs["context_windows"] = context_windows
                        if continue_sampling and (
                            not family_conditioning
                            or getattr(active_runtime, "sampling_compute_dtype", None) is not None
                        ):
                            kwargs["device"] = handle.load_device
                            kwargs["compute_dtype"] = _compute_dtype(handle.runtime)
                        if continue_sampling and sampling_shift is not None:
                            kwargs["sampling_shift"] = _runtime_sampling_shift(
                                active_runtime, sampling_shift
                            )
                        if schedule_state is not None:
                            kwargs["resolver"] = schedule_state.resolve
                        elif preview is not None:
                            kwargs["on_state"] = preview.on_state
                        if continue_sampling:
                            with _materialized_application_kwargs(
                                applications,
                                handle,
                                samples,
                                reserved_keys={
                                    *kwargs,
                                    "cancelled",
                                    "noise_inds",
                                    "observer",
                                    "on_state",
                                    "on_step",
                                    "parent_span_id",
                                    "resolver",
                                },
                            ) as application_kwargs:
                                if (
                                    schedule_state is not None
                                    and "sd15_attention_contributions" in application_kwargs
                                ):
                                    raise ValueError(
                                        "SD1.5 IP-Adapter does not support scheduled prompt or "
                                        "patch sampling"
                                    )
                                kwargs.update(application_kwargs)
                                result = sample(samples, **kwargs)
        finally:
            if schedule_state is not None:
                schedule_state.close()
        output: dict[object, object] = dict(cast("Mapping[object, object]", latent_image))
        output["samples"] = result
        return cls.outputs(latent=output)


class NativeKSamplerAdvanced(KSamplerAdvanced):
    """Run a partial full-schedule range through a native runtime."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        conditioning_batching: object | None = None,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("noise_seed", noise_seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("start_at_step", start_at_step, 0, cls.MAX_STEPS),
            ("end_at_step", end_at_step, 0, cls.MAX_STEPS),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if add_noise not in ("enable", "disable"):
            raise ValueError("add_noise must be 'enable' or 'disable'")
        if return_with_leftover_noise not in ("disable", "enable"):
            raise ValueError("return_with_leftover_noise must be 'disable' or 'enable'")
        effective_end = min(end_at_step, steps)
        if start_at_step >= effective_end:
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            output = dict(cast("Mapping[object, object]", latent_image))
            output.pop("downscale_ratio_spacial", None)
            output.pop("downscale_ratio_temporal", None)
            return cls.outputs(latent=output)
        inference = importlib.import_module("dinkster_inference")
        segment = inference.SamplingSegment(
            steps=steps,
            start_step=start_at_step,
            end_step=effective_end,
            add_noise=add_noise == "enable",
            return_with_leftover_noise=return_with_leftover_noise == "enable",
        )
        result = NativeKSampler.execute(
            model=model,
            seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            segment=segment,
            conditioning_batching=conditioning_batching,
        )
        return cls.outputs(latent=result["latent"])


class NativeVAEDecode(VAEDecode):
    """Decode native NCHW content and publish Comfy's NHWC image form."""

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        handle = _native_handle(vae, "vae")
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent_obj = cast("Mapping[object, object]", samples).get("samples")
        inference = importlib.import_module("dinkster_inference")
        if type(latent_obj) is inference.MultiStreamLatent:
            streams = cast("Any", latent_obj)
            if "video" not in streams.roles:
                raise TypeError("samples['samples'] must contain a video stream")
            latent_obj = streams.by_role("video")
        if not isinstance(latent_obj, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent = cast("Any", latent_obj)
        if len(latent.shape) not in (4, 5):
            raise ValueError(
                "samples['samples'] must be NCHW or NCTHW rank 4/5, "
                f"got shape {tuple(latent.shape)}"
            )
        with handle.stage("vae"):
            with torch.inference_mode():
                load_latent = latent.to(handle.load_device)
                image = _run_direct_vae(
                    handle=handle,
                    value=load_latent,
                    direction="decode",
                    operation=handle.runtime.decode_latent,
                    codec=getattr(handle.runtime, "codec", None),
                )
        if len(image.shape) == 5:
            if image.shape[1] != 3:
                raise ValueError(
                    f"native video VAE decode must return [B,3,T,H,W], got {tuple(image.shape)}"
                )
            image = image.permute(0, 2, 3, 4, 1).flatten(0, 1)
        elif len(image.shape) == 4:
            image = image.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"native VAE decode must return NCHW/NCTHW rank 4/5, got {tuple(image.shape)}"
            )
        return cls.outputs(image=image)


class NativeVAEEncode(VAEEncode):
    """Convert Comfy NHWC images to native NCHW codec content."""

    @classmethod
    def execute(cls, *, pixels: object, vae: object) -> Mapping[str, object]:
        handle = _native_handle(vae, "vae")
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        tensor = cast("Any", pixels)
        if len(tensor.shape) != 4:
            raise ValueError(f"pixels must be NHWC rank 4, got shape {tuple(tensor.shape)}")
        with handle.stage("vae"):
            content = tensor.permute(0, 3, 1, 2).to(handle.load_device)
            if importlib.import_module("dinkster_native.families.wan21").is_runtime_family(
                handle.runtime.family.id
            ):
                content = content.permute(1, 0, 2, 3).unsqueeze(0)
            with torch.inference_mode():
                latent = _run_direct_vae(
                    handle=handle,
                    value=content,
                    direction="encode",
                    operation=handle.runtime.encode_content,
                    codec=getattr(handle.runtime, "codec", None),
                )
        return cls.outputs(latent={"samples": latent})
