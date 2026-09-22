"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import (
    _NATIVE_HOOKS_KEY,
    _NATIVE_MASK_BOUNDS_KEY,
    _NATIVE_MASK_KEY,
    _NATIVE_PROMPT_KEY,
    Any,
    AssetRef,
    Callable,
    ExitStack,
    Mapping,
    NativeComponentHandle,
    NativeRuntimeHandle,
    Path,
    ResidencyRouteFacts,
    _active_inference_registries,
    _condition_entries,
    _freeze_embedding_resource,
    _NativeHooks,
    _NativeLoraHook,
    _not_cancelled,
    _torch,
    cast,
    contextmanager,
    current_execution_context,
    default_native_residency,
    default_pool,
    importlib,
    os,
    replace,
    select_load_device,
)
from .native_arm_runtime import (
    _aimdo_mechanism_factory,
    _aimdo_mode,
    _AimdoComponentFactory,
    _build_runtime_handle,
    _curve_segments,
    _lora_patch_weight_dtype,
    _materialize_patch_sets,
    _native_lora_overlay,
    _overlay_targets,
    _prompt_routes,
    _recipe_source,
    _ScheduledDiffusionDeclaration,
    _ScheduledTextRuntime,
    _source_asset,
    _torch_dtype,
    _weight_source_ref,
)


class _NativeScheduleState:
    def __init__(
        self,
        handle: NativeRuntimeHandle,
        inference: Any,
        inference_torch: Any,
        *,
        ordinary_overlays: tuple[Any, ...] = (),
        ordinary_resolvers: Mapping[str, object] | None = None,
    ) -> None:
        self.handle = handle
        self.inference = inference
        self.inference_torch = inference_torch
        self.executions: list[Any] = []
        self.overlays: dict[str, tuple[Any, ...]] = {}
        self.resolvers: dict[str, Any] = dict(ordinary_resolvers or {})
        self.patch_sets: dict[str, Any] = {}
        self.hooks: tuple[_NativeLoraHook, ...] = ()
        self.ordinary_overlays = ordinary_overlays
        self.declaration = inference.KeyedContribution(
            "inference.patch-providers",
            "dinkster.native.lora",
            behavior_metadata=(("contractVersion", 1),),
        )
        self.snapshot = inference_torch.PatchProviderSnapshot((self.declaration,))

    def builder(
        self,
        _base: object,
        target: object,
        overlays: tuple[Any, ...],
        cancelled: object,
    ) -> object:
        if cast("Any", cancelled)():
            raise RuntimeError("native scheduled variant construction cancelled")
        if target is self.inference.PatchTargetComponent.TEXT:
            resolvers = {
                overlay.source.digest: self.resolvers[overlay.source.digest]
                for overlay in overlays
                if overlay.source.digest in self.resolvers
            }
            return _ScheduledTextRuntime(
                self.handle.clone(cast("Any", overlays), source_resolvers=resolvers)
            )
        return _ScheduledDiffusionDeclaration()

    def register(self, overlays: tuple[Any, ...]) -> None:
        if not overlays:
            return
        digest = self.inference.patch_overlay_stack_digest(overlays)
        assert digest is not None
        existing = self.overlays.get(digest)
        if existing is not None and existing != overlays:
            raise RuntimeError("native scheduled overlay identity collision")
        self.overlays[digest] = overlays
        for overlay in overlays:
            source = overlay.source
            for hook in self.hooks:
                if hook.lora.digest == source.digest and hook.lora.resolver is not None:
                    self.resolvers[source.digest] = hook.lora.resolver

    def add_hooks(self, hooks: _NativeHooks) -> None:
        self.hooks += hooks.loras

    def resolve(self, requests: tuple[Any, ...], cancel: Any) -> tuple[Any, ...]:
        resolved: list[Any] = []
        for request in requests:
            if cancel():
                raise RuntimeError("native scheduled patch resolution cancelled")
            overlays = self.overlays.get(request.stack_digest)
            if overlays is None:
                raise RuntimeError(
                    f"native scheduled patch stack {request.stack_digest} is unknown"
                )
            patch_set = self.patch_sets.get(request.stack_digest)
            if patch_set is None:
                recipe = replace(self.handle.recipe, overlays=overlays)
                patch_sets = _materialize_patch_sets(
                    self.inference,
                    self.inference_torch,
                    recipe,
                    self.resolvers,
                )
                patch_set = patch_sets.get("diffusion")
                if patch_set is None:
                    raise RuntimeError("native scheduled LoRA stack has no diffusion patches")
                self.patch_sets[request.stack_digest] = patch_set
            resolved.append(
                self.inference_torch.ScheduledPatchResolution(
                    request,
                    self.handle.recipe.runtime_identity,
                    self.snapshot,
                    self.declaration.id,
                    self.declaration,
                    patch_set,
                )
            )
        return tuple(resolved)

    def close(self) -> None:
        error: BaseException | None = None
        for execution in self.executions:
            try:
                execution.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        if error is not None:
            raise error


def _scheduled_carrier(
    value: object,
    input_id: str,
    handle: NativeRuntimeHandle,
    state: _NativeScheduleState,
) -> object:
    inference = state.inference
    carriers: list[Any] = []
    for entry_index, raw_entry in enumerate(_condition_entries(value, input_id)):
        metadata = cast("dict[object, object]", raw_entry[1])
        unsupported = set(metadata) - {
            "pooled_output",
            "start_percent",
            "end_percent",
            "strength",
            "concat_mask",
            "concat_latent_image",
            _NATIVE_PROMPT_KEY,
            _NATIVE_HOOKS_KEY,
            _NATIVE_MASK_KEY,
            _NATIVE_MASK_BOUNDS_KEY,
        }
        if unsupported:
            raise ValueError(
                f"{input_id} scheduled conditioning metadata is unsupported on the native arm: "
                + ", ".join(sorted(repr(key) for key in unsupported))
            )
        prompt_data = metadata.get(_NATIVE_PROMPT_KEY)
        if not isinstance(prompt_data, tuple):
            raise ValueError(
                f"{input_id} scheduled conditioning must originate from native CLIP Text Encode"
            )
        prompt_values = cast("tuple[object, ...]", prompt_data)
        if len(prompt_values) != 2 or any(type(item) is not str for item in prompt_values):
            raise ValueError(
                f"{input_id} scheduled conditioning must originate from native CLIP Text Encode"
            )
        text, runtime_identity = cast("tuple[str, str]", prompt_values)
        if runtime_identity != handle.recipe.runtime_identity:
            raise ValueError(f"{input_id} conditioning belongs to a different native runtime")
        start = metadata.get("start_percent", 0.0)
        end = metadata.get("end_percent", 1.0)
        strength = metadata.get("strength", 1.0)
        if any(type(item) not in (int, float) for item in (start, end, strength)):
            raise TypeError(f"{input_id} schedule and strength values must be numbers")
        numeric = cast("tuple[int | float, int | float, int | float]", (start, end, strength))
        start, end, strength = (float(item) for item in numeric)
        schedule = inference.PercentRange(start, end)
        mask = metadata.get(_NATIVE_MASK_KEY)
        mask_binding = None
        mask_descriptor = None
        if mask is not None:
            torch = _torch()
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{input_id} mask must be a torch.Tensor")
            mask_tensor = cast("Any", mask)
            if mask_tensor.ndim < 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_binding = state.inference_torch.tensor_to_payload_binding(
                f"{input_id}-mask-{entry_index}",
                mask_tensor,
                space=state.inference_torch.MASK_PAYLOAD_SPACE,
            )
            mask_descriptor = inference.MaskDescriptor(
                inference.PayloadReference(mask_binding.reference_id),
                strength,
                metadata.get(_NATIVE_MASK_BOUNDS_KEY) is True,
            )
        hooks = metadata.get(_NATIVE_HOOKS_KEY)
        if hooks is not None and not isinstance(hooks, _NativeHooks):
            raise TypeError(f"{input_id} hooks did not come from native hook nodes")
        if hooks is not None:
            state.add_hooks(hooks)
            segments = _curve_segments(hooks)
        else:
            segments = ((0.0, 1.0, ()),)
        for segment_start, segment_end, multipliers in segments:
            effective = inference.intersect_ranges(
                schedule, inference.PercentRange(segment_start, segment_end)
            )
            if effective is inference.EMPTY_RANGE:
                continue
            scheduled_overlays = (
                tuple(
                    _native_lora_overlay(
                        handle,
                        hook.lora,
                        hook.strength_model * multiplier,
                        hook.strength_clip * multiplier,
                    )
                    for hook, multiplier in zip(hooks.loras, multipliers, strict=True)
                    if hook.strength_model * multiplier != 0.0
                    or hook.strength_clip * multiplier != 0.0
                )
                if hooks is not None
                else ()
            )
            overlays = state.ordinary_overlays + scheduled_overlays
            if overlays:
                state.register(overlays)
            text_stacks = (
                (inference.ScheduledPatchStack(effective, overlays),)
                if any(
                    float(overlay.strength_clip) != 0.0 and _overlay_targets(overlay, "text")
                    for overlay in overlays
                )
                else ()
            )
            diffusion_stacks = (
                (inference.ScheduledPatchStack(effective, overlays),)
                if any(
                    float(overlay.strength_model) != 0.0 and _overlay_targets(overlay, "diffusion")
                    for overlay in overlays
                )
                else ()
            )
            request = inference.ScheduledEncodeRequest(
                (
                    inference.ScheduledPrompt(
                        effective,
                        _prompt_routes(inference, handle.runtime.family, text),
                    ),
                ),
                text_patches=text_stacks,
                diffusion_patches=diffusion_stacks,
            )
            execution = inference.ScheduledExecution(inference.ScheduledVariantOwner(state.builder))
            state.executions.append(execution)
            execution_context = current_execution_context()
            carrier: Any = handle.runtime.encode_text_scheduled(
                request,
                execution=execution,
                cancelled=(
                    execution_context.cancelled if execution_context is not None else _not_cancelled
                ),
                type_registry=inference.InferenceTypeRegistry(),
            )
            if mask_descriptor is not None:
                assert mask_binding is not None
                records = tuple(
                    replace(record, mask=mask_descriptor) for record in carrier.conditioning.records
                )
                carrier = inference.make_conditioning_carrier(
                    inference.ConditioningSet(records),
                    (*carrier.bindings, mask_binding),
                )
            elif strength != 1.0:
                area = inference.AreaDescriptor(
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    inference.AreaUnits.PERCENT,
                    strength,
                )
                records = tuple(
                    replace(record, area=area) for record in carrier.conditioning.records
                )
                carrier = inference.make_conditioning_carrier(
                    inference.ConditioningSet(records), carrier.bindings
                )
            carriers.append(carrier)
    records = tuple(record for carrier in carriers for record in carrier.conditioning.records)
    bindings = tuple(binding for carrier in carriers for binding in carrier.bindings)
    return inference.make_conditioning_carrier(inference.ConditioningSet(records), bindings)


def _catalog_id(registry: Any, requested: str, kind: str) -> str:
    descriptor = registry.get(requested)
    if descriptor is None:
        raise ValueError(
            f"unknown {kind} {requested!r} (registered ids: {', '.join(registry.ids())})"
        )
    return cast("str", descriptor.id)


def _compute_dtype(runtime: Any) -> Any:
    dtype = runtime.assembled.compute_dtype("diffusion")
    if dtype is None:
        raise RuntimeError(
            f"native runtime family {runtime.family.id!r} has no assembled diffusion dtype"
        )
    return dtype


def _component_candidate_path(asset: AssetRef) -> Path:
    if asset.resolver is None:
        raise ValueError(f"component asset {asset.digest} has no resolver")
    path = asset.resolver.resolve(asset.digest)
    if path is None:
        raise ValueError(f"component asset {asset.digest} is not materializable")
    return path


def _component_descriptor(family_id: str) -> Any:
    descriptor = _active_inference_registries().components.get(family_id)
    if descriptor is None:
        raise ValueError(f"no component architecture detected for {family_id!r}")
    return descriptor


def _build_component_runtime_handle(
    descriptor: Any,
    asset: AssetRef,
    role: str,
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    as_model: bool = False,
    load_device: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    attention_policy: Any | None = None,
    attention_route_token: Any | None = None,
    artifact_role: str | None = None,
    storage_dtype: object | None = None,
) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    context = current_execution_context()
    if required_recipe is not None:
        attention_policy = required_recipe.knobs.attention_policy
        attention_route_token = required_recipe.knobs.attention_route_token
    elif attention_policy is None:
        attention_policy = "auto" if context is None else context.attention_policy
        attention_route_token = None if context is None else context.attention_route_token
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    load_identity = (
        expected_identity
        if required_recipe is None
        else replace(required_recipe, overlays=()).runtime_identity
    )
    load_kwargs: dict[str, Any] = {}
    attention_backend = descriptor.family.engine.attention_backend(role)
    if attention_backend is not None:
        load_kwargs.update(
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            attention_backend=attention_backend,
        )
    if descriptor.family.engine.quantized_component_load_device:
        load_kwargs["load_device"] = device
    if artifact_role is not None:
        load_kwargs["artifact_role"] = artifact_role
    loaded = execution_symbol(descriptor.loader)(
        _component_candidate_path(asset),
        asset=asset,
        expected_role=role,
        expected_identity=load_identity,
        compute_dtype=_torch_dtype(torch, compute_dtype),
        **load_kwargs,
    )
    if descriptor.tokenizer_attribute is not None and loaded.tokenizer is not None:
        setattr(loaded.module, descriptor.tokenizer_attribute, loaded.tokenizer)
    base_recipe = descriptor.recipe(
        _weight_source_ref(inference, asset),
        loaded,
        compute_dtype,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    label = descriptor.family.display_name
    if base_recipe.runtime_identity != load_identity:
        raise RuntimeError(f"{label} component recipe identity differs from dispatch identity")
    if required_recipe is not None and base_recipe != replace(required_recipe, overlays=()):
        raise RuntimeError(f"rebuilt {label} component recipe differs from retained recipe")
    recipe = base_recipe if required_recipe is None else required_recipe
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError(f"{label} component overlay identity differs from dispatch identity")
    resolvers = dict(source_resolvers or {})
    if asset.resolver is not None:
        resolvers[asset.digest] = asset.resolver
    patch_sets = _materialize_patch_sets(inference, inference_torch, recipe, resolvers)
    if set(patch_sets) - {role}:
        raise RuntimeError(f"{label} component overlays must target only {role}")

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_asset = _source_asset(_recipe_source(next_recipe, role), next_resolvers)
        return _build_component_runtime_handle(
            descriptor,
            next_asset,
            role,
            next_recipe.runtime_identity,
            torch,
            compute_dtype=compute_dtype,
            as_model=as_model,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            artifact_role=artifact_role,
            storage_dtype=storage_dtype,
        )

    if as_model:
        from dinkster_inference.component_registry import build_component_runtime

        runtime = build_component_runtime(
            descriptor, loaded, recipe.runtime_identity, _torch_dtype(torch, compute_dtype)
        )
        handle = _build_runtime_handle(
            runtime,
            torch,
            recipe=recipe,
            load_device=device,
            patch_sets=patch_sets,
            storage_dtype=storage_dtype,
            materializer=materializer,
            source_resolvers=resolvers,
        )
        if descriptor.pool_model:
            pool = default_pool()
            pool.label(handle, asset.name)
            handle.attach_pool(pool)
        return handle

    return _enroll_component_handle(
        loaded.module,
        role,
        recipe=recipe,
        device=device,
        torch=torch,
        materializer=materializer,
        source_resolvers=resolvers,
        label=asset.name,
        patch_set=patch_sets.get(role),
        aimdo_roles=descriptor.aimdo_roles,
        fixed_promotion_roles=descriptor.fixed_promotion_roles,
    )


def _enroll_component_handle(
    module: Any,
    role: str,
    *,
    recipe: Any,
    device: Any,
    torch: Any,
    materializer: Any,
    source_resolvers: Mapping[str, object],
    label: str,
    patch_set: Any = None,
    aimdo_roles: tuple[str, ...] = (),
    fixed_promotion_roles: tuple[str, ...] = (),
    runtime: object | None = None,
) -> NativeComponentHandle:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    enroll_kwargs: dict[str, Any] = (
        {}
        if patch_set is None
        else {"patch_set": patch_set, "patch_weight_dtype": _lora_patch_weight_dtype(device)}
    )
    route_facts_provider: Callable[[], ResidencyRouteFacts] | None = None
    if aimdo_roles:
        mode = _aimdo_mode()
        mechanism_factory, fallback_reason = (
            _aimdo_mechanism_factory(mode, device, torch) if role in aimdo_roles else (None, None)
        )
        if mechanism_factory is None:
            route_facts = ResidencyRouteFacts(
                requested=mode,
                mechanism="eager",
                fallback_reason=fallback_reason,
                resident_components=(role,) if role not in aimdo_roles else (),
                fallback_components=(role,) if fallback_reason is not None else (),
            )

            def eager_route_facts() -> ResidencyRouteFacts:
                return route_facts

            route_facts_provider = eager_route_facts
            coordinator = default_native_residency()
        else:
            component_factory = _AimdoComponentFactory(
                mode,
                mechanism_factory,
                inference_torch.ResidentWeights,
                {id(module): role},
                frozenset(),
                fixed_promotion_components=frozenset(fixed_promotion_roles),
            )
            route_facts_provider = component_factory.route_facts
            coordinator = default_native_residency(free_memory=inference_torch.dynamic_free_memory)
            enroll_kwargs["mechanism_factory"] = component_factory
    else:
        coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
        **enroll_kwargs,
    )
    handle_kwargs: dict[str, Any] = (
        {} if route_facts_provider is None else {"residency_route_facts": route_facts_provider}
    )
    if runtime is not None:
        handle_kwargs["runtime"] = runtime
    handle = NativeComponentHandle(
        module,
        mechanism,
        device,
        resource_identity=recipe.runtime_identity,
        coordinator=coordinator,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=source_resolvers,
        **handle_kwargs,
    )
    pool = default_pool()
    pool.label(handle, label)
    handle.attach_pool(pool)
    return handle


def _ltxav_component_identity(
    inference: Any,
    asset: AssetRef,
    role: str,
    compute_dtype: str,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
) -> tuple[str, Any]:
    path = _component_candidate_path(asset)
    source = inference.load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=path.stat().st_size,
    )
    planned = inference.plan_ltxav_split_component(source, role=role, path=path)
    dtype = {
        "float16": inference.FLOAT16,
        "float32": inference.FLOAT32,
        "bfloat16": inference.BFLOAT16,
    }.get(compute_dtype)
    if dtype is None:
        raise ValueError(f"unsupported LTX-2 component compute dtype {compute_dtype!r}")
    return (
        inference.ltxav_component_runtime_identity(
            planned,
            dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        planned,
    )


class _LTXAVTextHandle:
    family_id = "dinkster.ltxav"
    role = "text"

    def __init__(
        self,
        handles: tuple[NativeComponentHandle, ...],
        runtime_identity: str,
    ) -> None:
        if len(handles) not in (2, 3):
            raise ValueError("LTX-2 text composition requires Gemma and projection components")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        gemma = cast("Any", handles[0].component)
        projection = cast("Any", handles[1].component)
        connectors = None if len(handles) == 2 else cast("Any", handles[2].component)
        if type(gemma) is not inference_torch.LTXAVGemmaComponent:
            raise TypeError("LTX-2 text composition requires the Gemma component")
        self._handles = handles
        self._runtime = inference_torch.LTXAVTextRuntime(
            gemma.model,
            projection,
            gemma.tokenizer_model,
            connectors=connectors,
        )
        self.resource_identity = runtime_identity
        self.load_device = handles[0].load_device

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self._handles[0]

    @property
    def _dinkster_resident_refs(self) -> tuple[NativeComponentHandle, ...]:
        return self._handles[1:]

    def require_active(self) -> None:
        for handle in self._handles:
            handle.require_active()

    @contextmanager
    def stage(self):
        self.require_active()
        with ExitStack() as stages:
            for handle in self._handles:
                stages.enter_context(handle.stage())
            yield

    def encode_text(self, text: str) -> Any:
        return self._runtime.text_conditioning_carrier(self._runtime.encode_text(text))


def _build_ltxav_text_handle(
    text_encoder: AssetRef,
    checkpoint: AssetRef,
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    load_device: Any | None,
) -> _LTXAVTextHandle:
    inference = importlib.import_module("dinkster_inference")
    context = current_execution_context()
    attention_policy = "auto" if context is None else context.attention_policy
    attention_route_token = None if context is None else context.attention_route_token
    text_path = _component_candidate_path(text_encoder)
    text_source = inference.load_safetensors_header(
        text_path,
        asset_digest=text_encoder.digest,
        asset_size=text_path.stat().st_size,
    )
    text_role = inference.identify_ltxav_text_source(text_source)
    if text_role not in ("gemma3_12b", "gemma4_12b"):
        raise RuntimeError("LTX-2 text encoder asset has no supported Gemma text role")
    gemma_identity, _ = _ltxav_component_identity(
        inference,
        text_encoder,
        text_role,
        compute_dtype,
        attention_policy,
        attention_route_token,
    )
    projection_asset = checkpoint
    try:
        projection_identity, projection_plan = _ltxav_component_identity(
            inference,
            projection_asset,
            "text_projection",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
    except inference.LTXAVComponentAssemblyError:
        projection_asset = text_encoder
        projection_identity, projection_plan = _ltxav_component_identity(
            inference,
            projection_asset,
            "text_projection",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
    projection_kind = projection_plan.component.config
    if (text_role == "gemma4_12b") != (projection_kind == "dual_linear_gemma4"):
        raise RuntimeError("LTX-2 Gemma and text projection profiles differ")
    components = {
        text_role: gemma_identity,
        "text_projection": projection_identity,
    }
    connector_identity = None
    if projection_plan.component.config == "single_linear":
        connector_identity, _ = _ltxav_component_identity(
            inference,
            checkpoint,
            "connectors",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
        components["connectors"] = connector_identity
    composition = inference.compose_execution("dinkster.ltxav", components)
    if composition.execution_identity != expected_identity:
        raise RuntimeError("LTX-2 text composition differs from dispatch identity")
    handles: list[NativeComponentHandle] = []
    try:
        handles.append(
            _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                text_encoder,
                text_role,
                gemma_identity,
                torch,
                compute_dtype=compute_dtype,
                load_device=load_device,
            )
        )
        handles.append(
            _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                projection_asset,
                "text_projection",
                projection_identity,
                torch,
                compute_dtype=compute_dtype,
                load_device=load_device,
            )
        )
        if connector_identity is not None:
            handles.append(
                _build_component_runtime_handle(
                    _component_descriptor("dinkster.ltxav"),
                    checkpoint,
                    "connectors",
                    connector_identity,
                    torch,
                    compute_dtype=compute_dtype,
                    load_device=load_device,
                )
            )
    except Exception:
        for handle in reversed(handles):
            handle.terminal_release()
        raise
    return _LTXAVTextHandle(tuple(handles), expected_identity)


def _ltxav_audio_codec_recipe(inference: Any, asset: AssetRef, loaded: Any) -> Any:
    plans = loaded.plan.identity_components
    runtime_facts = tuple(sorted({fact for plan in plans for fact in plan.runtime_facts}))
    return inference.ReconstructionRecipe(
        sources=(inference.WeightSourceBinding("audio_vae", _weight_source_ref(inference, asset)),),
        family_id="dinkster.ltxav",
        component_identity=inference.runtime_component_identity("dinkster.ltxav", plans),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype="unloaded",
            text_dtype="unloaded",
            vae_dtype="float32",
            fp8_matmul=False,
            runtime_facts=runtime_facts,
        ),
    )


def _build_ltxav_audio_codec_handle(
    asset: AssetRef,
    expected_identity: str,
    torch: Any,
    *,
    load_device: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
) -> NativeComponentHandle:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    loaded = inference_torch.load_ltxav_audio_codec(
        _component_candidate_path(asset),
        asset=asset,
        expected_identity=expected_identity,
    )
    recipe = _ltxav_audio_codec_recipe(inference, asset, loaded)
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("LTX-2 audio codec recipe identity differs from dispatch identity")
    if required_recipe is not None and recipe != required_recipe:
        raise RuntimeError("rebuilt LTX-2 audio codec recipe differs from retained recipe")
    resolvers = dict(source_resolvers or {})
    if asset.resolver is not None:
        resolvers[asset.digest] = asset.resolver
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        loaded.module,
        load_device=device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_asset = _source_asset(_recipe_source(next_recipe, "audio_vae"), next_resolvers)
        return _build_ltxav_audio_codec_handle(
            next_asset,
            next_recipe.runtime_identity,
            torch,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
        )

    handle = NativeComponentHandle(
        loaded.module,
        mechanism,
        device,
        resource_identity=expected_identity,
        coordinator=coordinator,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=resolvers,
    )
    pool = default_pool()
    pool.label(handle, asset.name)
    handle.attach_pool(pool)
    return handle


def _component_execution_context(node_type: str) -> Any:
    context = current_execution_context()
    if context is None or context.expected_execution_identity is None:
        raise RuntimeError(f"native {node_type} ran without an expected execution identity")
    return context


def _trellis2_artifact_role_matches(asset: AssetRef, role: str) -> bool:
    inference = importlib.import_module("dinkster_inference")
    path = _component_candidate_path(asset)
    try:
        source = inference.load_safetensors_header(
            path,
            asset_digest=asset.digest,
            asset_size=asset.size,
        )
        inference.plan_trellis2_artifact(source, role=role, path=path)
    except ValueError:
        return False
    return True


def _trellis2_split_model_recipe(
    inference: Any,
    assets: Mapping[str, AssetRef],
    plan: Any,
    compute_dtype: str,
) -> Any:
    components = plan.identity_components
    runtime_facts = tuple(
        sorted({fact for component in components for fact in component.runtime_facts})
    )
    return inference.ReconstructionRecipe(
        sources=tuple(
            inference.WeightSourceBinding(role, _weight_source_ref(inference, assets[role]))
            for role in sorted(assets)
        ),
        family_id="dinkster.trellis2",
        component_identity=inference.runtime_component_identity("dinkster.trellis2", components),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype=compute_dtype,
            text_dtype="unloaded",
            vae_dtype="unloaded",
            fp8_matmul=False,
            runtime_facts=runtime_facts,
        ),
    )


def _build_trellis2_split_model_handle(
    assets: Mapping[str, AssetRef],
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    storage_dtype: object | None = None,
) -> NativeRuntimeHandle:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    role_order = ("shape", "shape-512", "structure", "texture", "texture-512")
    if tuple(sorted(assets)) != role_order:
        raise ValueError(f"TRELLIS.2 split flow roles must be exactly {role_order}")
    loaded = {
        role: inference_torch.load_trellis2_flow_artifact(
            _component_candidate_path(assets[role]),
            asset=assets[role],
            expected_role=role,
            compute_dtype=_torch_dtype(torch, compute_dtype),
        )
        for role in role_order
    }
    plan = inference.Trellis2SplitModelPlan(
        structure=loaded["structure"].plan,
        shape=loaded["shape"].plan,
        shape_512=loaded["shape-512"].plan,
        texture=loaded["texture"].plan,
        texture_512=loaded["texture-512"].plan,
    )
    recipe = _trellis2_split_model_recipe(inference, assets, plan, compute_dtype)
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("TRELLIS.2 split model recipe identity differs from dispatch identity")
    if required_recipe is not None and recipe != required_recipe:
        raise RuntimeError("rebuilt TRELLIS.2 split model recipe differs from retained recipe")
    resolvers = dict(source_resolvers or {})
    for asset in assets.values():
        if asset.resolver is not None:
            resolvers[asset.digest] = asset.resolver
    assembled = inference_torch.AssembledTrellis2(
        inference_torch.Trellis2FlowBundle(
            structure=loaded["structure"].module,
            shape=loaded["shape"].module,
            shape_512=loaded["shape-512"].module,
            texture=loaded["texture"].module,
            texture_512=loaded["texture-512"].module,
        ),
        plan,
        _torch_dtype(torch, compute_dtype),
    )
    runtime = inference_torch.Trellis2DiffusionRuntime(
        assembled,
        runtime_identity=expected_identity,
        compute_dtype=_torch_dtype(torch, compute_dtype),
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_assets = {
            role: _source_asset(_recipe_source(next_recipe, role), next_resolvers)
            for role in role_order
        }
        return _build_trellis2_split_model_handle(
            next_assets,
            next_recipe.runtime_identity,
            torch,
            compute_dtype=compute_dtype,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            storage_dtype=storage_dtype,
        )

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        storage_dtype=storage_dtype,
        materializer=materializer,
        source_resolvers=resolvers,
    )


def _load_detected_component(
    asset: AssetRef,
    kind: str,
    context: Any,
    *,
    load_device: Any | None = None,
    detected: Any | None = None,
    storage_dtype: object | None = None,
) -> Any:
    inference = importlib.import_module("dinkster_inference")
    registry = _active_inference_registries().components
    family_id = context.expected_execution_identity.partition(":")[2].partition(":")[0]
    if detected is None:
        path = _component_candidate_path(asset)
        source = inference.load_safetensors_header(
            path, asset_digest=asset.digest, asset_size=asset.size
        )
        descriptor, role, plan = registry.select(source, path, kind, family_id=family_id)
    else:
        descriptor, role, plan = registry.select_detected(detected, kind, family_id=family_id)
    dtype = {
        "model": context.diffusion_dtype,
        "text": context.text_dtype,
        "codec": context.vae_dtype,
    }[kind]
    if dtype is None:
        dtype = inference.default_vae_dtype(descriptor.id).name if kind == "codec" else "bfloat16"
    storage_kwargs: dict[str, Any] = (
        {} if storage_dtype is None else {"storage_dtype": storage_dtype}
    )
    artifact_role = getattr(plan, "artifact_role", None)
    artifact_kwargs = {} if artifact_role is None else {"artifact_role": artifact_role}
    return _build_component_runtime_handle(
        descriptor,
        asset,
        role,
        context.expected_execution_identity,
        _torch(),
        compute_dtype=dtype,
        as_model=kind == "model",
        load_device=load_device,
        attention_policy=context.attention_policy,
        attention_route_token=context.attention_route_token,
        **storage_kwargs,
        **artifact_kwargs,
    )


def build_text_recipe_handle(
    assets: tuple[AssetRef, ...],
    requested_type: str,
    expected_identity: str,
    *,
    compute_dtype: str,
    load_device: Any | None = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    embedding_resource: Any | None = None,
) -> NativeComponentHandle:
    """Verify ordered sources and reconstruct their explicit text encoding recipe."""
    from dinkster_inference import PatchSet  # noqa: F401
    from dinkster_inference.component_registry import execution_symbol
    from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
    from dinkster_inference.text_recipes import resolve_text_recipe

    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    torch = _torch()
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    if required_recipe is not None:
        attention_policy = required_recipe.knobs.attention_policy
        attention_route_token = required_recipe.knobs.attention_route_token
    with ExitStack() as opened:
        files = tuple(opened.enter_context(asset.open()) for asset in assets)
        sources: list[SafetensorsSource] = []
        for asset, file in zip(assets, files, strict=True):
            if os.fstat(file.fileno()).st_size != asset.size:
                raise ValueError("text component byte size differs from its asset metadata")
            source = load_safetensors_header_from_file(
                file,
                path=_component_candidate_path(asset),
                asset_digest=asset.digest,
                asset_size=asset.size,
            )
            sources.append(replace(source, configuration_file=file))
        registry = _active_inference_registries().components
        detected = tuple(registry.detect(source, source.path) for source in sources)
        binding = resolve_text_recipe(detected, requested_type)
        if any(part.profile is not None for part in binding.components):
            resource = (
                _freeze_embedding_resource(
                    component_roles=tuple(part.role for part in binding.components)
                )
                if embedding_resource is None
                else embedding_resource
            )
        else:
            resource = (None, None)
        embedding_index, embedding_lookups = resource
        base_recipe = binding.recipe(
            tuple(_weight_source_ref(inference, asset) for asset in assets),
            compute_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            embedding_binding_digest=(
                None if embedding_index is None else embedding_index.binding_digest
            ),
        )
        if required_recipe is not None and base_recipe != replace(required_recipe, overlays=()):
            raise RuntimeError("rebuilt text encoding recipe differs from retained recipe")
        recipe = base_recipe if required_recipe is None else required_recipe
        if recipe.runtime_identity != expected_identity:
            raise RuntimeError("text encoding recipe identity differs from dispatch identity")
        load_kwargs: dict[str, object] = {}
        descriptor = registry.get(binding.family_id)
        if descriptor is not None and descriptor.family.engine.attention_backends:
            load_kwargs["attention_backends"] = descriptor.family.engine.attention_backends
        loaded = execution_symbol(binding.loader)(
            binding,
            compute_dtype=_torch_dtype(torch, compute_dtype),
            sources=tuple(sources),
            source_files=files,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            **load_kwargs,
        )
    runtime = execution_symbol(binding.runtime_class)(loaded, embedding_lookups=embedding_lookups)
    resolvers = dict(source_resolvers or {})
    resolvers.update(
        {asset.digest: asset.resolver for asset in assets if asset.resolver is not None}
    )
    patch_sets = cast(
        "dict[str, PatchSet[Any]]",
        _materialize_patch_sets(inference, inference_torch, recipe, resolvers),
    )
    if set(patch_sets) - set(loaded.module):
        raise RuntimeError("text overlays target components absent from the encoding recipe")
    patch_set = (
        inference.PatchSet(
            {
                f"{role}.{key}": tuple(patches.entries(key))
                for role, patches in patch_sets.items()
                for key in patches.keys()
            },
            structural_digest=recipe.patch_stack_digest,
        )
        if patch_sets
        else None
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return build_text_recipe_handle(
            tuple(_source_asset(item.source, next_resolvers) for item in next_recipe.sources),
            binding.id,
            next_recipe.runtime_identity,
            compute_dtype=compute_dtype,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            embedding_resource=resource,
        )

    return _enroll_component_handle(
        loaded.module,
        "text",
        recipe=recipe,
        device=device,
        torch=torch,
        materializer=materializer,
        source_resolvers=resolvers,
        label=" + ".join(asset.name for asset in assets),
        patch_set=patch_set,
        runtime=runtime,
    )
