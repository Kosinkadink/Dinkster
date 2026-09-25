"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .family_registry import load_registered_component
from .native_arm_core import (
    Any,
    ApplyZImageControlPatch,
    AssetRef,
    ControlNetApply,
    ControlNetApplyAdvanced,
    ControlNetLoader,
    LoadCheckpoint,
    LoadClip,
    LoadDiffusionModel,
    LoadDualClip,
    LoadLora,
    LoadLoraModelOnly,
    LoadVae,
    LoadVision,
    LoadZImageControlPatch,
    Mapping,
    NativeComponentHandle,
    NativeRuntimeHandle,
    Node,
    NodeSchema,
    OutputInterface,
    Sequence,
    _active_inference_registries,
    _condition_entries,
    _freeze_embedding_resource,
    _torch,
    cast,
    current_execution_context,
    default_native_residency,
    default_pool,
    hashlib,
    importlib,
    log,
    math,
    native_execution_span,
    os,
    resolve_weight_source,
    select_load_device,
    uses_classic_embedding_bindings,
    validate_lora_execution_mode,
)
from .native_arm_runtime import (
    _build_runtime_handle,
    _canonical_runtime_sources,
    _ClassicControlBinding,
    _ClassicControlEntry,
    _ControlHintSnapshot,
    _load_runtime,
    _materialize_recipe_handle,
    _native_handle,
    _native_lora_overlay,
    _native_model,
    _native_model_sampling_cache,
    _native_model_sampling_space,
    _native_model_sampling_timeline,
    _NativeControlNetResource,
    _NativeModelOverlay,
    _overlay_for_components,
    _overlay_has_offsets,
    _overlay_targets,
    _PixelSpaceCodecHandle,
    _recipe_bundle_name,
    _runtime_recipe,
    _weight_storage_dtype,
    _ZImageControlBinding,
)
from .native_arm_scheduling import (
    _build_component_runtime_handle,
    _component_candidate_path,
    _component_descriptor,
    _component_execution_context,
    _load_detected_component,
    _ltxav_component_identity,
    _trellis2_artifact_role_matches,
    build_text_recipe_handle,
)
from .nodes_provider import _generation_provider_schema

if TYPE_CHECKING:
    from dinkster_inference import ContextWindowsSpec


class NativeLoadClip(LoadClip):
    """Load one official text encoder as an independent component."""

    @classmethod
    def execute(cls, *, text_encoder: object, type: str, device: str) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        from dinkster_inference.sources import load_safetensors_header
        from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe

        context = _component_execution_context("load_clip")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            path = _component_candidate_path(text_encoder)
            source = load_safetensors_header(
                path, asset_digest=text_encoder.digest, asset_size=text_encoder.size
            )
            detected = _active_inference_registries().components.detect(source, path)
            try:
                resolve_text_recipe((detected,), type)
            except UnresolvedTextRecipe:
                fixed = tuple(
                    match for match in detected if not match.descriptor.requires_text_recipe
                )
                if detected and not fixed:
                    raise
                handle = _load_detected_component(
                    text_encoder, "text", context, load_device=load_device, detected=fixed
                )
            else:
                handle = build_text_recipe_handle(
                    (text_encoder,),
                    type,
                    context.expected_execution_identity,
                    compute_dtype=context.text_dtype or "float32",
                    load_device=load_device,
                    attention_policy=context.attention_policy,
                    attention_route_token=context.attention_route_token,
                )
        return cls.outputs(clip=handle)


class NativeLoadDualClip(LoadDualClip):
    """Load two official text encoders as one ordered native recipe."""

    @classmethod
    def execute(
        cls,
        *,
        text_encoder1: object,
        text_encoder2: object,
        type: str,
        device: str,
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder1, AssetRef):
            raise TypeError("text_encoder1 must be an AssetRef")
        if not isinstance(text_encoder2, AssetRef):
            raise TypeError("text_encoder2 must be an AssetRef")
        context = _component_execution_context("load_dual_clip")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            handle = build_text_recipe_handle(
                (text_encoder1, text_encoder2),
                type,
                context.expected_execution_identity,
                compute_dtype=context.text_dtype or "float32",
                load_device=load_device,
                attention_policy=context.attention_policy,
                attention_route_token=context.attention_route_token,
            )
        return cls.outputs(clip=handle)


class NativeLoadVae(LoadVae):
    """Load one official codec as an independent component."""

    @classmethod
    def execute(
        cls, *, vae: object | None = None, pixel_space: bool = False
    ) -> Mapping[str, object]:
        if type(pixel_space) is not bool:
            raise TypeError("pixel_space must be a boolean")
        context = _component_execution_context("load_vae")
        if pixel_space:
            if vae is not None:
                raise ValueError("pixel_space and vae are mutually exclusive")
            inference = importlib.import_module("dinkster_inference")
            compute_dtype = (
                context.vae_dtype or inference.default_vae_dtype(inference.CHROMA_RADIANCE.id).name
            )
            return cls.outputs(
                vae=_PixelSpaceCodecHandle(context.expected_execution_identity, compute_dtype)
            )
        if not isinstance(vae, AssetRef):
            raise TypeError("vae must be an AssetRef")
        with native_execution_span("load", "load"):
            handle = _load_detected_component(vae, "codec", context)
        return cls.outputs(vae=handle)


class NativeLoadVision(LoadVision):
    """Load one admitted vision component under native residency."""

    @classmethod
    def execute(cls, *, vision_encoder: object) -> Mapping[str, object]:
        if not isinstance(vision_encoder, AssetRef):
            raise TypeError("vision_encoder must be an AssetRef")
        context = _component_execution_context("load_vision")
        if not _trellis2_artifact_role_matches(vision_encoder, "vision"):
            raise ValueError("native vision loading requires an admitted TRELLIS.2 vision asset")
        with native_execution_span("load", "load"):
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.trellis2"),
                vision_encoder,
                "vision",
                context.expected_execution_identity,
                _torch(),
                compute_dtype=context.text_dtype or "bfloat16",
            )
        return cls.outputs(vision=handle)


class NativeLoadDiffusionModel(LoadDiffusionModel):
    """Load one admitted diffusion component."""

    @classmethod
    def execute(cls, *, diffusion_model: object, weight_dtype: str) -> Mapping[str, object]:
        if not isinstance(diffusion_model, AssetRef):
            raise TypeError("diffusion_model must be an AssetRef")
        storage_dtype = (
            None if weight_dtype == "default" else _weight_storage_dtype(_torch(), weight_dtype)
        )
        context = current_execution_context()
        if context is None or context.expected_execution_identity is None:
            raise RuntimeError(
                "native load_diffusion_model ran without an expected execution identity"
            )
        with native_execution_span("load", "load"):
            handle = _load_detected_component(
                diffusion_model, "model", context, storage_dtype=storage_dtype
            )
        return cls.outputs(model=handle)


class NativeLoadCheckpoint(LoadCheckpoint):
    """Load a checkpoint through native checkpoint or component plans."""

    @classmethod
    def execute(cls, *, checkpoint: object) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError(f"checkpoint must be an AssetRef, got {type(checkpoint).__name__}")
        path = resolve_weight_source(checkpoint.local_path(), logical_name=checkpoint.name)
        context = current_execution_context()
        if context is None:
            raise RuntimeError(
                "native load_checkpoint ran without an execution context; "
                "the dispatch host must supply the selected body identity"
            )
        expected_identity = context.expected_execution_identity
        if expected_identity is None:
            raise RuntimeError("native load_checkpoint ran without an expected execution identity")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        source = inference.load_safetensors_header(
            path,
            asset_digest=checkpoint.digest,
            asset_size=checkpoint.size,
        )
        plan = inference.plan_native(checkpoint=source, fp8_matmul=context.fp8_matmul)
        embedding_resource = (
            _freeze_embedding_resource() if uses_classic_embedding_bindings(plan) else (None, None)
        )
        embedding_index, _ = embedding_resource
        embedding_binding_digest = (
            None if embedding_index is None else embedding_index.binding_digest
        )
        attention_kwargs: dict[str, object] = {}
        if context.attention_route_token is None:
            runtime = _load_runtime(
                path,
                expected_identity,
                context.fp8_matmul,
                context.extension_snapshot_digest,
                embedding_resource,
                source_assets={"checkpoint": checkpoint},
            )
        else:
            attention_kwargs = {
                "attention_policy": context.attention_policy,
                "attention_route_token": context.attention_route_token,
            }
            runtime = _load_runtime(
                path,
                expected_identity,
                context.fp8_matmul,
                context.extension_snapshot_digest,
                embedding_resource,
                context.attention_policy,
                context.attention_route_token,
                source_assets={"checkpoint": checkpoint},
            )
        recipe = _runtime_recipe(
            inference,
            {"checkpoint": checkpoint},
            {"checkpoint": source},
            fp8_matmul=context.fp8_matmul,
            extension_snapshot_digest=context.extension_snapshot_digest,
            plan=plan,
            embedding_binding_digest=embedding_binding_digest,
            **attention_kwargs,
        )
        if recipe.runtime_identity != expected_identity:
            raise RuntimeError("native checkpoint recipe identity does not match dispatch identity")
        source_resolvers = (
            {} if checkpoint.resolver is None else {checkpoint.digest: checkpoint.resolver}
        )

        def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
            return _materialize_recipe_handle(
                next_recipe, next_resolvers, torch, embedding_resource
            )

        handle = _build_runtime_handle(
            runtime,
            torch,
            recipe=recipe,
            materializer=materializer,
            source_resolvers=source_resolvers,
        )
        pool = default_pool()
        pool.label(handle, checkpoint.name)
        handle.attach_pool(pool)
        return cls.outputs(model=handle, clip=handle, vae=handle)


class NativeLoadModelProfile(Node):
    """Re-probe a stored model profile and project an existing loader result."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_model_profile")

    @classmethod
    def execute(
        cls,
        *,
        checkpoint: object,
        entries: str,
        output_spec: OutputInterface,
    ) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError(f"checkpoint must be an AssetRef, got {type(checkpoint).__name__}")
        inference = importlib.import_module("dinkster_inference")
        path = checkpoint.local_path()
        with checkpoint.open() as verified:
            profile = inference.load_model_output_profile(
                path,
                asset_digest=checkpoint.digest,
                asset_size=os.fstat(verified.fileno()).st_size,
                stored=entries,
                handle=verified,
            )
        loaded = (
            NativeLoadCheckpoint.execute(checkpoint=checkpoint)
            if profile.kind == "checkpoint"
            else NativeLoadDiffusionModel.execute(
                diffusion_model=checkpoint,
                weight_dtype="default",
            )
        )
        return {output.id: loaded[output.id] for output in output_spec.outputs}


def _nvfp4_runtime_status(  # pyright: ignore[reportUnusedFunction]
    handle: NativeRuntimeHandle,
) -> object:
    """Return inference's immutable per-runtime NVFP4 snapshot unchanged."""
    module = importlib.import_module("dinkster_inference_torch._nvfp4_diagnostics")
    return module.nvfp4_runtime_status(handle.runtime)


def _attention_runtime_status(  # pyright: ignore[reportUnusedFunction]
    handle: NativeRuntimeHandle,
) -> object:
    """Return immutable recipe-bound attention routing evidence."""
    handle.require_active()
    inference = importlib.import_module("dinkster_inference")
    return inference.resolve_attention_runtime_status(
        handle.recipe.knobs.attention_policy,
        handle.recipe.knobs.attention_route_token,
    )


def _native_controlnet(value: object) -> _NativeControlNetResource:
    if not isinstance(value, _NativeControlNetResource):
        raise TypeError(
            f"control_net must be a native ControlNet resource, got {type(value).__name__}"
        )
    if value.handle is not None:
        value.handle.require_active()
    return value


def _snapshot_control_hint(
    image: object, torch: Any, inference_torch: Any, channels: int = 3
) -> _ControlHintSnapshot:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"ControlNet image must be a torch.Tensor, got {type(image).__name__}")
    tensor = cast("Any", image)
    allowed_channels = (1, 3) if channels == 1 else (channels,)
    if tensor.ndim != 4 or tensor.shape[-1] not in allowed_channels:
        raise ValueError(
            f"ControlNet image must be [batch x H x W x {channels}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError("ControlNet image batch and spatial dimensions must be positive")
    hint = (
        tensor.detach()
        .movedim(-1, 1)
        .to(device=torch.device("cpu"), dtype=torch.float32)
        .contiguous()
        .clone()
    )
    shape = cast("tuple[int, int, int, int]", tuple(int(dim) for dim in hint.shape))
    return _ControlHintSnapshot(
        shape,
        hint.numpy().tobytes(order="C"),
        inference_torch.sd_control_hint_digest(hint),
    )


def _control_child_id(
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    strength: float,
    start_percent: float,
    end_percent: float,
    previous: Any | None,
) -> str:
    identity = "\n".join(
        (
            resource.resource_digest,
            hint.digest,
            repr(strength),
            repr(start_percent),
            repr(end_percent),
            *(() if resource.mode is None else (repr(resource.mode),)),
            "" if previous is None else previous.child_id,
        )
    )
    return "classic-" + hashlib.sha256(identity.encode("ascii")).hexdigest()[:24]


def _extend_control_binding(
    previous: _ClassicControlBinding | None,
    *,
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    strength: float,
    start_percent: float,
    end_percent: float,
    apply_to_uncond: bool,
    inference: Any,
) -> _ClassicControlBinding:
    pool = default_pool()
    resident_id = "resident:" + pool.rid_for(resource)
    previous_application = None if previous is None else previous.application
    child_id = _control_child_id(
        resource, hint, strength, start_percent, end_percent, previous_application
    )
    application = inference.ControlApplication(
        child_id,
        inference.PayloadReference(hint.digest),
        strength,
        inference.PercentRange(start_percent, end_percent),
        previous_application,
        mode=resource.mode,
    )
    entry = _ClassicControlEntry(child_id, resident_id, resource.resource_digest, hint)
    return _ClassicControlBinding(
        application,
        (*(() if previous is None else previous.entries), entry),
        apply_to_uncond,
    )


def _apply_classic_control(conditioning: object, **kwargs: Any) -> list[list[object]]:
    result: list[list[object]] = []
    for raw_entry in _condition_entries(conditioning, "conditioning"):
        metadata = cast("Mapping[object, object]", raw_entry[1])
        previous = metadata.get("control")
        if previous is not None and not isinstance(previous, _ClassicControlBinding):
            raise TypeError(
                "classic ControlNet cannot chain after a non-native control value; "
                "all ControlNet apply nodes must execute on the native arm"
            )
        binding = _extend_control_binding(previous, **kwargs)
        updated = dict(metadata)
        updated["control"] = binding
        updated["control_apply_to_uncond"] = binding.apply_to_uncond
        result.append([raw_entry[0], updated])
    return result


def _enroll_control_module(
    module: Any, torch: Any, inference_torch: Any, *, discard_on_release: bool = False
) -> NativeComponentHandle:
    load_device = select_load_device(torch)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=load_device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )
    return NativeComponentHandle(
        module,
        mechanism,
        load_device,
        resource_identity=module.resource_digest,
        coordinator=coordinator,
        discard_on_release=discard_on_release,
    )


class NativeControlNetLoader(ControlNetLoader):
    @classmethod
    def execute(cls, *, control_net_name: object) -> Mapping[str, object]:
        if not isinstance(control_net_name, AssetRef):
            raise TypeError(
                f"control_net_name must be an AssetRef, got {type(control_net_name).__name__}"
            )
        from dinkster_inference.component_registry import component_plans, execution_symbol

        path = resolve_weight_source(
            control_net_name.local_path(), logical_name=control_net_name.name
        )
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        source = inference.load_safetensors_header(
            path, asset_digest=control_net_name.digest, asset_size=control_net_name.size
        )
        descriptor, _role, plan = _active_inference_registries().components.select(
            source, path, "controlnet"
        )
        log.info(
            "Control component %s selected by tensor geometry; "
            "target-model compatibility is checked during execution, not by family name",
            descriptor.id,
        )
        torch = _torch()
        load_device = select_load_device(torch)
        controlnet_dtype = (
            torch.float32
            if load_device.type == "cpu"
            else getattr(torch, descriptor.default_diffusion_dtype.name)
        )
        handle = None
        digest = control_net_name.digest.removeprefix("blake3:")
        if not getattr(descriptor, "requires_base", False):
            load_kwargs: dict[str, object] = {}
            attention_backend = descriptor.family.engine.attention_backend("controlnet")
            if attention_backend is not None:
                context = current_execution_context()
                if descriptor.family.engine.attention_requires_route and (
                    context is None or context.attention_route_token is None
                ):
                    raise ValueError(
                        f"control component {descriptor.id} requires an authenticated "
                        "attention route"
                    )
                load_kwargs.update(
                    attention_policy=("auto" if context is None else context.attention_policy),
                    attention_route_token=(
                        None if context is None else context.attention_route_token
                    ),
                    attention_backend=attention_backend,
                )
            module = execution_symbol(descriptor.loader)(plan, controlnet_dtype, **load_kwargs)
            handle = _enroll_control_module(module, torch, inference_torch)
            digest = module.resource_digest
        resource = _NativeControlNetResource(
            handle,
            control_net_name.digest,
            digest,
            getattr(plan, "source_layout", "native"),
            descriptor,
            plan,
            getattr(
                component_plans(plan)[0].config,
                "hint_channels",
                getattr(descriptor, "hint_channels", 3),
            ),
        )
        if handle is not None:
            pool = default_pool()
            pool.label(handle, control_net_name.name)
            handle.attach_pool(pool)
        return cls.outputs(control_net=resource)


class NativeControlNetApply(ControlNetApply):
    @classmethod
    def execute(
        cls,
        *,
        conditioning: object,
        control_net: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        if strength == 0.0:
            return cls.outputs(conditioning=conditioning)
        resource = _native_controlnet(control_net)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        hint = _snapshot_control_hint(image, torch, inference_torch, resource.hint_channels)
        applied = _apply_classic_control(
            conditioning,
            resource=resource,
            hint=hint,
            strength=float(strength),
            start_percent=0.0,
            end_percent=1.0,
            apply_to_uncond=True,
            inference=inference,
        )
        return cls.outputs(conditioning=applied)


class NativeControlNetApplyAdvanced(ControlNetApplyAdvanced):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        control_net: object,
        image: object,
        strength: float,
        start_percent: float,
        end_percent: float,
        vae: object = None,
    ) -> Mapping[str, object]:
        if vae is not None:
            raise ValueError(
                "SD1.5 native ControlNetApplyAdvanced does not support the optional VAE input"
            )
        if strength == 0.0:
            return cls.outputs(positive=positive, negative=negative)
        resource = _native_controlnet(control_net)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        hint = _snapshot_control_hint(image, torch, inference_torch, resource.hint_channels)
        return cls.outputs(
            positive=_apply_classic_control(
                positive,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                apply_to_uncond=False,
                inference=inference,
            ),
            negative=_apply_classic_control(
                negative,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                apply_to_uncond=False,
                inference=inference,
            ),
        )


class NativeLoadZImageControlPatch(LoadZImageControlPatch):
    @classmethod
    def execute(cls, *, model_patch: object) -> Mapping[str, object]:
        if not isinstance(model_patch, AssetRef):
            raise TypeError(f"model_patch must be an AssetRef, got {type(model_patch).__name__}")
        context = current_execution_context()
        if context is None:
            raise RuntimeError("native model patch loading requires an execution context")
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        source = inference.load_safetensors_header(model_patch.local_path())
        attention_kwargs: dict[str, object] = {}
        if context.attention_route_token is not None:
            attention_kwargs = {
                "attention_policy": context.attention_policy,
                "attention_route_token": context.attention_route_token,
            }
        source_keys = source.keys()
        if (
            any(key.endswith("duration_head.attention_pooler.query_tokens") for key in source_keys)
            or "attention_pooler.query_tokens" in source_keys
        ):
            identity, _ = _ltxav_component_identity(
                inference,
                model_patch,
                "duration_head",
                "float32",
            )
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                model_patch,
                "duration_head",
                identity,
                _torch(),
                compute_dtype="float32",
            )
            return cls.outputs(model_patch=handle)
        if "audio_proj.proj1.weight" in source.keys():
            plan = inference.plan_wan21_multitalk(source, asset_digest=model_patch.digest)
            assembled = inference_torch.assemble_wan21_multitalk(plan, **attention_kwargs)
            module = assembled.patch
            resource_identity = inference.extend_runtime_identity(
                inference.build_runtime_identity_from_facts(
                    "dinkster.wan21",
                    inference.runtime_component_identity("dinkster.wan21", (plan.patch,)),
                    diffusion_dtype=inference.BFLOAT16.name,
                    text_dtype="unloaded",
                    vae_dtype="unloaded",
                    fp8_matmul=False,
                    runtime_facts=plan.patch.runtime_facts,
                ),
                (f"resource={assembled.resource_digest}",),
            )
        else:
            plan = inference.plan_z_image_control(source, asset_digest=model_patch.digest)
            assembled = inference_torch.assemble_z_image_control(plan, **attention_kwargs)
            module = assembled.control
            resource_identity = assembled.resource_digest
        torch = _torch()
        load_device = select_load_device(torch)
        coordinator = default_native_residency()
        mechanism = coordinator.enroll_component(
            module,
            load_device=load_device,
            offload_device=torch.device("cpu"),
            enroller=inference_torch.enroll_component,
        )
        handle = NativeComponentHandle(
            module,
            mechanism,
            load_device,
            resource_identity=resource_identity,
            coordinator=coordinator,
        )
        pool = default_pool()
        pool.label(handle, model_patch.name)
        handle.attach_pool(pool)
        return cls.outputs(model_patch=handle)


class NativeApplyZImageControlPatch(ApplyZImageControlPatch):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        model_patch: object,
        vae: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            existing_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        if getattr(handle.runtime, "accepts_z_image_control", False) is not True:
            raise TypeError("Z-Image Fun ControlNet requires a native Z-Image model")
        if _native_handle(vae, "vae") is not handle:
            raise ValueError("model and VAE must come from the same Z-Image runtime")
        if existing_control is not None:
            raise ValueError("a Z-Image model can carry only one Fun ControlNet patch")
        if not isinstance(model_patch, NativeComponentHandle):
            raise TypeError("model_patch must be a native Z-Image control patch")
        model_patch.require_active()
        if getattr(model_patch.module, "accepts_z_image_control_binding", False) is not True:
            raise TypeError("model_patch must be a native Z-Image control patch")
        torch = _torch()
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        tensor = cast("Any", image)
        if tensor.ndim != 4 or tensor.shape[-1] < 3:
            raise ValueError("image must be an NHWC torch.Tensor with at least three channels")
        if not math.isfinite(strength) or not -10.0 <= strength <= 10.0:
            raise ValueError(f"strength must be finite and in [-10.0, 10.0], got {strength}")
        control_image = tensor[..., :3].permute(0, 3, 1, 2).detach().to("cpu").contiguous()
        binding = _ZImageControlBinding(model_patch, control_image, strength)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                binding,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


def load_native_runtime_handle(assets: Mapping[str, object]) -> NativeRuntimeHandle:
    """Load a fixed-role asset bundle through the production native handle path."""
    # Resolve and verify every fixed-role asset before reading headers or
    # constructing runtime state.
    for role, asset in assets.items():
        if not isinstance(asset, AssetRef):
            raise TypeError(f"{role} must be an AssetRef, got {type(asset).__name__}")
    refs = cast("dict[str, AssetRef]", assets)
    paths = {role: refs[role].local_path() for role in assets}
    context = current_execution_context()
    if context is None:
        raise RuntimeError(
            "native runtime loading ran without an execution context; "
            "the host must supply execution knobs"
        )
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    canonical_paths = _canonical_runtime_sources(paths)
    sources = {
        role: inference.load_safetensors_header(path) for role, path in canonical_paths.items()
    }
    plan = inference.plan_native(**sources, fp8_matmul=context.fp8_matmul)
    embedding_resource = (
        _freeze_embedding_resource() if uses_classic_embedding_bindings(plan) else (None, None)
    )
    embedding_index, _ = embedding_resource
    embedding_binding_digest = None if embedding_index is None else embedding_index.binding_digest
    attention_kwargs: dict[str, object] = {}
    if context.attention_route_token is not None:
        attention_kwargs = {
            "attention_policy": context.attention_policy,
            "attention_route_token": context.attention_route_token,
        }
    recipe = _runtime_recipe(
        inference,
        refs,
        sources,
        fp8_matmul=context.fp8_matmul,
        extension_snapshot_digest=context.extension_snapshot_digest,
        plan=plan,
        embedding_binding_digest=embedding_binding_digest,
        **attention_kwargs,
    )
    expected_identity = context.expected_execution_identity or recipe.runtime_identity
    if context.attention_route_token is None:
        runtime = _load_runtime(
            canonical_paths,
            expected_identity,
            context.fp8_matmul,
            context.extension_snapshot_digest,
            embedding_resource,
            source_assets=refs,
        )
    else:
        runtime = _load_runtime(
            canonical_paths,
            expected_identity,
            context.fp8_matmul,
            context.extension_snapshot_digest,
            embedding_resource,
            context.attention_policy,
            context.attention_route_token,
            source_assets=refs,
        )
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("native runtime recipe identity does not match dispatch identity")
    source_resolvers = {
        asset.digest: asset.resolver for asset in refs.values() if asset.resolver is not None
    }

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, embedding_resource)

    handle = _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )
    pool = default_pool()
    label_source = refs["diffusion"] if "diffusion" in refs else refs["checkpoint"]
    pool.label(handle, label_source.name)
    handle.attach_pool(pool)
    return handle


def _lora_execution_mode(handle: NativeRuntimeHandle, requested: str) -> str:
    if requested == "precalculate" or callable(getattr(handle.runtime, "sample_scheduled", None)):
        return requested
    log.warning(
        "native runtime does not support scheduled LoRA patch resolution; "
        "falling back to precalculate execution mode"
    )
    return "precalculate"


def _apply_native_lora_stack(
    model: object,
    clip: object,
    loras: Sequence[tuple[object, float, float]],
    execution_mode: str,
) -> tuple[object, object]:
    validate_lora_execution_mode(execution_mode)
    (
        model_handle,
        existing,
        existing_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    inference = importlib.import_module("dinkster_inference")

    active: list[tuple[AssetRef, float, float]] = []
    for lora, strength_model, strength_clip in loras:
        if not isinstance(lora, AssetRef):
            raise TypeError(f"lora must be an AssetRef, got {type(lora).__name__}")
        if strength_model != 0.0 or strength_clip != 0.0:
            active.append((lora, strength_model, strength_clip))
    family = _active_inference_registries().families.get(model_handle.recipe.family_id)
    if (
        family is not None
        and family.engine.supports(inference.FamilyCapability.SPLIT_TEXT_LORA)
        and isinstance(clip, NativeComponentHandle)
    ):
        text_handle = load_registered_component(clip, "clip")
        if text_handle.recipe is None or (
            text_handle.recipe.family_id != model_handle.recipe.family_id
        ):
            raise ValueError("Flux2 model and clip must belong to the same family")
        return _apply_split_flux2_lora_stack(
            model,
            model_handle,
            text_handle,
            active,
            execution_mode,
            existing=existing,
            existing_resolvers=existing_resolvers,
            z_image_control=z_image_control,
            sampling_shift=sampling_shift,
            guidance_transforms=guidance_transforms,
            context_windows=context_windows,
            chroma_radiance_options=chroma_radiance_options,
        )
    clip_handle = _native_handle(clip, "clip")
    if model_handle is not clip_handle:
        raise ValueError("native Load LoRA requires model and clip from the same runtime handle")
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = model_handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(model_handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            patched_model: object = clone
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                patched_model = _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return patched_model, clone
        return model, clip

    overlays = tuple(
        _native_lora_overlay(model_handle, lora, strength_model, strength_clip)
        for lora, strength_model, strength_clip in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    has_text_patches = any(_overlay_targets(item, "text") for item in combined)
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    execution_mode = _lora_execution_mode(model_handle, execution_mode)
    if execution_mode == "attach" and has_text_patches:
        raise ValueError("LoRA attach mode supports diffusion-only patches")
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if (
        execution_mode == "precalculate"
        or has_text_patches
        or has_offset_patches
        or z_image_control
    ):
        clone = model_handle.clone(combined, source_resolvers=combined_resolvers)
        bundle_name = _recipe_bundle_name(model_handle.recipe)
        default_pool().label(clone, f"{bundle_name} + {active[-1][0].name}")
        patched_model: object = clone
        if (
            z_image_control is not None
            or _native_model_sampling_space(model) is not None
            or sampling_shift is not None
            or guidance_transforms
            or context_windows is not None
            or chroma_radiance_options
            or _native_model_sampling_cache(model) is not None
            or _native_model_sampling_timeline(model) is not None
        ):
            patched_model = _NativeModelOverlay(
                clone,
                (),
                {},
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        return patched_model, clone
    return (
        _NativeModelOverlay(
            model_handle,
            combined,
            combined_resolvers,
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
        ),
        clip_handle,
    )


def _apply_split_flux2_lora_stack(
    model: object,
    model_handle: NativeRuntimeHandle,
    text_handle: NativeComponentHandle,
    active: Sequence[tuple[AssetRef, float, float]],
    execution_mode: str,
    *,
    existing: tuple[Any, ...],
    existing_resolvers: Mapping[str, object],
    z_image_control: _ZImageControlBinding | None,
    sampling_shift: float | None,
    guidance_transforms: tuple[tuple[str, Any], ...],
    context_windows: ContextWindowsSpec | None,
    chroma_radiance_options: tuple[Any, ...],
) -> tuple[object, object]:
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = model_handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(model_handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            patched_model: object = clone
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                patched_model = _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return patched_model, text_handle
        return model, text_handle

    inference = importlib.import_module("dinkster_inference")
    overlays = tuple(
        _native_lora_overlay(
            model_handle,
            lora,
            strength_model,
            strength_clip,
            text_handle=text_handle,
        )
        for lora, strength_model, strength_clip in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    text_recipe = text_handle.recipe
    assert text_recipe is not None
    text_role = text_recipe.sources[0].role
    diffusion_overlays = tuple(
        selected
        for overlay in combined
        if (selected := _overlay_for_components(inference, overlay, {"diffusion"})) is not None
    )
    text_overlays = tuple(
        selected
        for overlay in overlays
        if (selected := _overlay_for_components(inference, overlay, {text_role})) is not None
    )
    has_text_patches = bool(text_overlays)
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    if execution_mode == "attach" and has_text_patches:
        raise ValueError("LoRA attach mode supports diffusion-only patches")
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if execution_mode != "precalculate" and not has_text_patches and not has_offset_patches:
        return (
            _NativeModelOverlay(
                model_handle,
                combined,
                combined_resolvers,
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            ),
            text_handle,
        )

    bundle_name = _recipe_bundle_name(model_handle.recipe)
    overlay_name = active[-1][0].name
    patched_handle = model_handle
    if diffusion_overlays:
        patched_handle = model_handle.clone(
            diffusion_overlays,
            source_resolvers=combined_resolvers,
        )
        default_pool().label(patched_handle, f"{bundle_name} + {overlay_name}")
    patched_text = text_handle
    if text_overlays:
        patched_text = text_handle.clone(
            text_overlays,
            source_resolvers=combined_resolvers,
        )
        default_pool().label(patched_text, f"{bundle_name} text + {overlay_name}")
    patched_model = patched_handle
    if (
        z_image_control is not None
        or _native_model_sampling_space(model) is not None
        or sampling_shift is not None
        or guidance_transforms
        or context_windows is not None
        or chroma_radiance_options
        or _native_model_sampling_cache(model) is not None
        or _native_model_sampling_timeline(model) is not None
    ):
        patched_model = _NativeModelOverlay(
            patched_handle,
            (),
            {},
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
        )
    return patched_model, patched_text


def _apply_native_model_lora_stack(
    model: object,
    loras: Sequence[tuple[object, float]],
    execution_mode: str,
) -> object:
    validate_lora_execution_mode(execution_mode)
    (
        handle,
        existing,
        existing_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    active: list[tuple[AssetRef, float]] = []
    for lora, strength_model in loras:
        if not isinstance(lora, AssetRef):
            raise TypeError(f"lora must be an AssetRef, got {type(lora).__name__}")
        if strength_model != 0.0:
            active.append((lora, strength_model))
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                return _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return clone
        return model

    overlays = tuple(
        _native_lora_overlay(handle, lora, strength_model, 0.0) for lora, strength_model in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    execution_mode = _lora_execution_mode(handle, execution_mode)
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if execution_mode == "precalculate" or has_offset_patches or z_image_control:
        clone = handle.clone(combined, source_resolvers=combined_resolvers)
        bundle_name = _recipe_bundle_name(handle.recipe)
        default_pool().label(clone, f"{bundle_name} + {active[-1][0].name}")
        if (
            z_image_control is not None
            or _native_model_sampling_space(model) is not None
            or sampling_shift is not None
            or guidance_transforms
            or context_windows is not None
            or chroma_radiance_options
            or _native_model_sampling_cache(model) is not None
            or _native_model_sampling_timeline(model) is not None
        ):
            return _NativeModelOverlay(
                clone,
                (),
                {},
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        return clone
    return _NativeModelOverlay(
        handle,
        combined,
        combined_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


class NativeLoadLora(LoadLora):
    """Apply one normalized LoRA through an explicit native execution strategy."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        lora: object,
        strength_model: float,
        strength_clip: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        patched_model, patched_clip = _apply_native_lora_stack(
            model,
            clip,
            ((lora, strength_model, strength_clip),),
            execution_mode,
        )
        return cls.outputs(model=patched_model, clip=patched_clip)


class NativeLoadLoraModelOnly(LoadLoraModelOnly):
    """Apply one diffusion-only LoRA through an explicit native strategy."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        lora: object,
        strength_model: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        return cls.outputs(
            model=_apply_native_model_lora_stack(
                model,
                ((lora, strength_model),),
                execution_mode,
            )
        )
