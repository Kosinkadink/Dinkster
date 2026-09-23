"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedClass=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .native_arm_core import (
    _AIMDO_HEADROOM_TARGET_ENV,
    _DISABLE_CFG1_OPTIMIZATION,
    _NATIVE_HOOKS_KEY,
    _NATIVE_MASK_KEY,
    _NATIVE_PREPARED_CONDITIONING_KEY,
    _NATIVE_PROMPT_KEY,
    _RUNTIME_SOURCE_ROLE_SETS,
    AcceleratorMemoryPolicyError,
    Any,
    AssetRef,
    Callable,
    EmbeddingNameIndex,
    ExitStack,
    Mapping,
    NativeComponentHandle,
    NativeResidencyPool,
    NativeRuntimeHandle,
    ParamSpec,
    Path,
    ResidencyRouteFacts,
    Sequence,
    _active_inference_registries,
    _builtin_inference_registries,
    _condition_entries,
    _embedding_resource,
    _freeze_embedding_resource,
    _GuidedRows,
    _inference_registries,
    _NativeHooks,
    _NativeLoraHook,
    _run_direct_vae,
    _sampler_registry,
    _torch,
    cast,
    contextmanager,
    current_execution_context,
    dataclass,
    default_dependency_residency_pool,
    default_native_residency,
    default_pool,
    hashlib,
    importlib,
    inspect,
    json,
    log,
    math,
    native_memory_policy,
    os,
    platform,
    resolve_weight_source,
    resolver_from_env,
    select_load_device,
    uses_classic_embedding_bindings,
    wraps,
)


def _enroll_control_module(
    module: Any, torch: Any, inference_torch: Any, *, discard_on_release: bool = False
) -> NativeComponentHandle:
    from .nodes_loaders import _enroll_control_module as enroll

    return enroll(
        module,
        torch,
        inference_torch,
        discard_on_release=discard_on_release,
    )


if TYPE_CHECKING:
    from dinkster_inference import ContextWindowsSpec


def _load_runtime(
    path: Path | Mapping[str, Path],
    expected_identity: str,
    fp8_matmul: bool,
    extension_snapshot_digest: str | None = None,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
    source_assets: Mapping[str, AssetRef] | None = None,
) -> Any:
    """Construct the runtime selected by the host's component dtype policy."""
    inference = importlib.import_module("dinkster_inference")
    wiring = importlib.import_module("dinkster_inference_torch.wiring")
    context = current_execution_context()
    dtype_kwargs: dict[str, object] = {}
    if context is not None and context.diffusion_dtype is not None:
        torch = _torch()
        dtype_kwargs = {
            "diffusion_dtype": _torch_dtype(torch, context.diffusion_dtype),
            "text_dtype": _torch_dtype(torch, cast("str", context.text_dtype)),
            "vae_dtype": _torch_dtype(torch, cast("str", context.vae_dtype)),
        }
    resource = _freeze_embedding_resource() if embedding_resource is None else embedding_resource
    embedding_index, embedding_lookups = resource
    embedding_kwargs: dict[str, object] = {}
    if embedding_index is not None and embedding_index.binding_digest is not None:
        assert embedding_lookups is not None
        embedding_kwargs = {
            "embedding_lookups": embedding_lookups,
            "embedding_binding_digest": embedding_index.binding_digest,
        }
    if isinstance(path, Path):
        paths = {"checkpoint": resolve_weight_source(path)}
    else:
        paths = _canonical_runtime_sources(path)
    assets = None if source_assets is None else _canonical_runtime_sources(source_assets)
    if assets is not None and assets.keys() != paths.keys():
        raise ValueError("native runtime assets and paths must have identical roles")
    sources = {
        role: inference.load_safetensors_header(
            source_path,
            **(
                {}
                if assets is None
                else {
                    "asset_digest": assets[role].digest,
                    "asset_size": assets[role].size,
                }
            ),
        )
        for role, source_path in paths.items()
    }
    sampler_registry, _, registry_token = _sampler_registry(inference, extension_snapshot_digest)
    family_registry = _inference_registries(inference, extension_snapshot_digest).families
    attention_kwargs = (
        {}
        if attention_route_token is None
        else {
            "attention_policy": attention_policy,
            "attention_route_token": attention_route_token,
        }
    )
    if registry_token is None:
        return wiring.load_runtime(
            **sources,
            expected_identity=expected_identity,
            storage_dtype_follows_compute=True,
            fp8_matmul=fp8_matmul,
            family_registry=family_registry,
            **dtype_kwargs,
            **attention_kwargs,
            **embedding_kwargs,
        )
    assert extension_snapshot_digest is not None
    guidance_executor = None
    materialize = getattr(inference, "materialize_inference_generation", None)
    generation = None if materialize is None else materialize(extension_snapshot_digest)
    if generation is not None and generation.guidance_contributions:
        inference_torch = importlib.import_module("dinkster_inference_torch")
        registry = inference_torch.GuidanceRegistry(generation.guidance_contributions)
        if registry.active:
            guidance_executor = inference_torch.GuidanceExecutor(registry)
    kwargs = {
        **sources,
        "expected_identity": expected_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": fp8_matmul,
        **dtype_kwargs,
        **attention_kwargs,
        "sampler_registry": sampler_registry,
        "family_registry": family_registry,
        "registry_token": registry_token,
        "extension_behavior_hash": extension_snapshot_digest.removeprefix("sha256:"),
        **embedding_kwargs,
    }
    if guidance_executor is not None:
        kwargs["guidance_executor"] = guidance_executor
    return wiring.load_runtime(**kwargs)


def _torch_dtype(torch: Any, name: str) -> Any:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError:
        raise RuntimeError(f"host selected unsupported runtime dtype {name!r}") from None


def _weight_storage_dtype(torch: Any, name: str) -> Any | None:
    return {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e4m3fn_fast": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(name)


def _lora_patch_weight_dtype(device: object) -> object:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    return inference_torch.lora_compute_dtype(device)


def _require_fp8_matmul_support(torch: Any, device: Any) -> bool:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    probe_error: Exception | None = None
    try:
        supported = bool(inference_torch.supports_fp8_matmul(device))
    except Exception as exc:  # noqa: BLE001 - normalize capability failures
        supported = False
        probe_error = exc
    if supported:
        return True
    capability = "unavailable"
    if getattr(device, "type", None) == "cuda":
        try:
            properties = torch.cuda.get_device_properties(device)
            capability = f"SM {properties.major}.{properties.minor}"
        except Exception:  # noqa: BLE001 - refusal still names unavailable capability
            pass
    message = (
        "fp8-matmul-unsupported: "
        f"device={device}; capability={capability}; "
        f"torch={getattr(torch, '__version__', 'unknown')}"
    )
    if probe_error is not None:
        message += f"; probeError={type(probe_error).__name__}: {probe_error}"
    log.warning("%s; falling back to the default matmul path", message)
    return False


def _disable_runtime_fp8_matmul(runtime: Any) -> None:
    assembled = runtime.assembled
    declared = cast("object", getattr(assembled, "components", None))
    candidates: Sequence[Any] = tuple(
        cast("Mapping[object, Any]", declared).values()
        if isinstance(declared, Mapping)
        else cast("Mapping[str, Any]", vars(assembled)).values()
    )
    components: dict[int, Any] = {
        id(module): module for module in candidates if callable(getattr(module, "modules", None))
    }
    for component in components.values():
        for module in component.modules():
            bind = getattr(module, "bind_fp8_matmul", None)
            if callable(bind):
                bind(False)


def _warn_aimdo_fallback(mode: str, gate: str, detail: str) -> str:
    """Warn about an aimdo gate failure and return it as a route fact reason."""
    if mode == "on":
        log.warning(
            "explicit --aimdo=on requested dynamic VRAM residency, but the %s "
            "gate failed: %s; falling back to eager residency",
            gate,
            detail,
        )
    else:
        log.warning(
            "automatic aimdo dynamic VRAM residency disabled by the %s gate: "
            "%s; falling back to eager residency",
            gate,
            detail,
        )
    return f"{gate} gate failed: {detail}"


def _aimdo_mode() -> str:
    mode = os.environ.get("DINKSTER_AIMDO_ARM", "auto")
    if mode not in ("auto", "on", "off"):
        raise ValueError("DINKSTER_AIMDO_ARM must be 'auto', 'on', or 'off'")
    return mode


def _upstream_default_platform(torch: Any) -> tuple[bool, str]:
    """Mirror ComfyUI main.py's default ModelPatcherDynamic gate.

    NVIDIA CUDA is admitted outside WSL. AMD ROCm requires a parsed runtime
    version of at least 7.14.
    """
    version = getattr(torch, "version", None)
    release = platform.uname().release
    if release.endswith("-Microsoft") or release.endswith("microsoft-standard-WSL2"):
        return False, f"WSL release {release!r} is excluded upstream"
    hip_version = getattr(version, "hip", None)
    if hip_version is not None:
        try:
            parts = str(hip_version).split(".")
            if len(parts) < 2:
                raise ValueError("missing minor version")
            rocm_version = tuple(map(int, parts[:2]))
        except (TypeError, ValueError):
            return (
                False,
                f"ROCm runtime {hip_version!r} is malformed; ROCm 7.14 or later is required",
            )
        if rocm_version >= (7, 14):
            return True, ""
        return False, f"ROCm runtime {hip_version!r} is below required version 7.14"
    if getattr(version, "cuda", None):
        return True, ""
    return False, "no NVIDIA CUDA or AMD ROCm runtime was reported; ROCm 7.14 or later is required"


def _apply_pending_aimdo_headroom() -> None:
    """Publish a headroom frame after activation succeeds."""
    raw = os.environ.get(_AIMDO_HEADROOM_TARGET_ENV)
    if raw is None:
        return
    try:
        target = int(raw)
        if target < 0:
            raise ValueError("negative target")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        applied = inference_torch.set_simple_vram_headroom(target)
    except Exception:  # noqa: BLE001 - headroom control cannot fail a graph
        log.warning("pending aimdo runtime headroom setter raised", exc_info=True)
        return
    if applied is True:
        os.environ.pop(_AIMDO_HEADROOM_TARGET_ENV, None)


def _aimdo_mechanism_factory(mode: str, device: Any, torch: Any) -> tuple[Any | None, str | None]:
    """Resolve AimdoWeights, or (None, reason) when a gate keeps residency eager."""
    if mode not in ("auto", "on"):
        return None, None
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "CUDA availability", repr(exc))
    if not cuda_available or device.type != "cuda":
        return None, _warn_aimdo_fallback(
            mode,
            "CUDA device",
            f"cuda_available={cuda_available}, load_device={device}",
        )
    if mode == "auto":
        try:
            supported, detail = _upstream_default_platform(torch)
        except Exception as exc:
            return None, _warn_aimdo_fallback(mode, "upstream platform", repr(exc))
        if not supported:
            return None, _warn_aimdo_fallback(mode, "upstream platform", detail)
    try:
        importlib.import_module("numpy")
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "NumPy", repr(exc))
    try:
        aimdo = importlib.import_module("dinkster_inference_torch.aimdo_residency")
        mechanism_factory = aimdo.AimdoWeights
        ready = aimdo.ensure_visible_aimdo_devices(native_memory_policy())
    except AcceleratorMemoryPolicyError:
        raise
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "aimdo device activation", repr(exc))
    if ready is not True:
        return None, _warn_aimdo_fallback(
            mode,
            "aimdo device activation",
            f"ensure_visible_aimdo_devices returned {ready!r} for load_device={device}",
        )
    _apply_pending_aimdo_headroom()
    return mechanism_factory, None


class _AimdoComponentFactory:
    """Keep one handle's selected Aimdo route unless its policy requires resident weights."""

    def __init__(
        self,
        mode: str,
        mechanism_factory: Any,
        eager_factory: Any,
        component_by_module: Mapping[int, str],
        resident_components: frozenset[str],
        *,
        fixed_promotion_components: frozenset[str] = frozenset(),
    ) -> None:
        self._mode = mode
        self._mechanism_factory = mechanism_factory
        self._eager_factory = eager_factory
        self._component_by_module = component_by_module
        self._resident_components = resident_components
        self._fixed_promotion_components = fixed_promotion_components
        self._dynamic_components: list[str] = []
        self._resident_served: list[str] = []

    def route_facts(self) -> ResidencyRouteFacts:
        return ResidencyRouteFacts(
            requested=self._mode,
            mechanism="aimdo" if self._dynamic_components else "eager",
            dynamic_components=tuple(self._dynamic_components),
            resident_components=tuple(self._resident_served),
        )

    def __call__(self, weights: Any, **kwargs: Any) -> Any:
        component = self._component_by_module.get(id(getattr(weights, "module", None)), "<unknown>")
        if component in self._resident_components:
            mechanism = self._eager_factory(weights, **kwargs)
            self._resident_served.append(component)
            return mechanism
        try:
            mechanism_kwargs = (
                {
                    **kwargs,
                    "fixed_promotion": True,
                    "promote_non_fp8_raw": True,
                }
                if component in self._fixed_promotion_components
                else kwargs
            )
            mechanism = self._mechanism_factory(weights, **mechanism_kwargs)
        except Exception as exc:
            exc.add_note(
                f"Aimdo construction failed for component {component!r} "
                f"(requested {self._mode}); the selected residency mechanism was not changed"
            )
            log.error(
                "Aimdo construction failed for component %r (requested %s); "
                "failing the load without changing its residency mechanism",
                component,
                self._mode,
                exc_info=True,
            )
            raise
        self._dynamic_components.append(component)
        return mechanism


def _build_runtime_handle(
    runtime: Any,
    torch: Any,
    *,
    recipe: Any,
    load_device: Any | None = None,
    patch_sets: Mapping[str, object] | None = None,
    storage_dtype: object | None = None,
    materializer: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
) -> NativeRuntimeHandle:
    """Enroll one runtime through the production native-handle path."""
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    if recipe.knobs.fp8_matmul and not _require_fp8_matmul_support(torch, device):
        _disable_runtime_fp8_matmul(runtime)
    patch_kwargs: dict[str, Any] = (
        {}
        if not patch_sets
        else {
            "patch_weight_dtype": _lora_patch_weight_dtype(device),
            "patch_key_prefixes": {
                component: prefix for prefix, component in _lora_target_routes(runtime.assembled)
            },
        }
    )
    storage_dtypes = None if storage_dtype is None else {"diffusion": storage_dtype}
    mode = _aimdo_mode()
    mechanism_factory, fallback_reason = _aimdo_mechanism_factory(mode, device, torch)
    if mechanism_factory is None:
        eager_facts = ResidencyRouteFacts(
            requested=mode,
            mechanism="eager",
            fallback_reason=fallback_reason,
        )
        return NativeRuntimeHandle(
            runtime,
            device,
            recipe=recipe,
            patch_sets=patch_sets,
            **patch_kwargs,
            storage_dtypes=storage_dtypes,
            materializer=materializer,
            source_resolvers=source_resolvers,
            coordinator=default_native_residency(),
            residency_route_facts=lambda: eager_facts,
            _torch_module=torch,
        )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    policy = getattr(runtime, "residency_policy", None)
    if policy is not None and not isinstance(policy, inference_torch.NativeResidencyPolicy):
        raise TypeError("runtime residency_policy must be a NativeResidencyPolicy or None")
    assembled = getattr(runtime, "assembled", None)
    classic_components = (
        "diffusion",
        "clip_l",
        "clip_g",
        "t5xxl",
        "umt5xxl",
        "gemma2_2b",
        "text_encoder",
        "text",
        "clip_vision",
        "vae",
    )
    policy_components = () if policy is None else policy.enrollment_components
    declared_components = getattr(assembled, "components", None)
    component_by_module = {
        id(module): component
        for component, module in (
            declared_components.items()
            if declared_components is not None
            else (
                (component, getattr(assembled, component, None))
                for component in dict.fromkeys((*classic_components, *policy_components))
            )
        )
        if module is not None
    }
    component_descriptor = _builtin_inference_registries().components.get(recipe.family_id)
    component_factory = _AimdoComponentFactory(
        mode,
        mechanism_factory,
        inference_torch.ResidentWeights,
        component_by_module,
        frozenset() if policy is None else policy.resident_components,
        fixed_promotion_components=frozenset(
            () if component_descriptor is None else component_descriptor.fixed_promotion_roles
        ),
    )
    return NativeRuntimeHandle(
        runtime,
        device,
        recipe=recipe,
        patch_sets=patch_sets,
        **patch_kwargs,
        storage_dtypes=storage_dtypes,
        materializer=materializer,
        source_resolvers=source_resolvers,
        coordinator=default_native_residency(free_memory=inference_torch.dynamic_free_memory),
        mechanism_factory=component_factory,
        residency_route_facts=component_factory.route_facts,
        _torch_module=torch,
    )


def _weight_source_ref(inference: Any, asset: AssetRef) -> Any:
    return inference.WeightSourceRef(
        digest=asset.digest,
        name=asset.name,
        size=asset.size,
        media_type=asset.media_type,
        virtual_path=asset.virtual_path,
    )


def _canonical_runtime_sources(sources: Mapping[str, Any]) -> dict[str, Any]:
    roles = tuple(sorted(sources))
    if roles not in _RUNTIME_SOURCE_ROLE_SETS:
        expected = " or ".join(str(role_set) for role_set in _RUNTIME_SOURCE_ROLE_SETS)
        raise ValueError(f"native runtime source roles must be exactly {expected}; got {roles}")
    return {role: sources[role] for role in roles}


def _runtime_recipe(
    inference: Any,
    assets: Mapping[str, AssetRef],
    sources: Mapping[str, Any],
    *,
    fp8_matmul: bool,
    extension_snapshot_digest: str | None,
    plan: Any | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
) -> Any:
    canonical_assets = _canonical_runtime_sources(assets)
    canonical_sources = _canonical_runtime_sources(sources)
    if canonical_assets.keys() != canonical_sources.keys():
        raise ValueError("native runtime assets and headers must have identical roles")
    selected_plan = (
        inference.plan_native(**canonical_sources, fp8_matmul=fp8_matmul) if plan is None else plan
    )
    context = current_execution_context()
    selected_dtypes = (
        (
            context.diffusion_dtype,
            context.text_dtype,
            context.vae_dtype,
        )
        if context is not None and context.diffusion_dtype is not None
        else (
            inference.default_diffusion_dtype(selected_plan.family.id).name,
            inference.default_text_dtype(selected_plan.family.id).name,
            inference.default_vae_dtype(selected_plan.family.id).name,
        )
    )
    assert all(isinstance(dtype, str) for dtype in selected_dtypes)
    extension_hash = (
        None
        if extension_snapshot_digest is None
        else extension_snapshot_digest.removeprefix("sha256:")
    )
    recipe = inference.ReconstructionRecipe(
        sources=tuple(
            inference.WeightSourceBinding(role, _weight_source_ref(inference, asset))
            for role, asset in canonical_assets.items()
        ),
        family_id=selected_plan.family.id,
        component_identity=inference.runtime_component_identity(
            selected_plan.family.id, selected_plan.identity_components
        ),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype=selected_dtypes[0],
            text_dtype=selected_dtypes[1],
            vae_dtype=selected_dtypes[2],
            fp8_matmul=fp8_matmul,
            registry_token=extension_snapshot_digest,
            extension_behavior_hash=extension_hash,
            embedding_binding_digest=embedding_binding_digest,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
    )
    return recipe


def _source_asset(source_ref: Any, source_resolvers: Mapping[str, object]) -> AssetRef:
    resolver = source_resolvers.get(source_ref.digest)
    if resolver is None:
        resolver = resolver_from_env()
    if resolver is None:
        raise RuntimeError(
            f"no worker-local asset store can resolve reconstruction source {source_ref.digest}"
        )
    return AssetRef(
        digest=source_ref.digest,
        name=source_ref.name,
        size=source_ref.size,
        media_type=source_ref.media_type,
        virtual_path=source_ref.virtual_path,
        resolver=cast("Any", resolver),
    )


def _recipe_source(recipe: Any, role: str) -> Any:
    for binding in recipe.sources:
        if binding.role == role:
            return binding.source
    raise RuntimeError(f"native reconstruction recipe has no {role!r} source")


def _recipe_bundle_name(recipe: Any) -> str:
    roles = tuple(binding.role for binding in recipe.sources)
    if len(roles) == 1:
        primary_role = roles[0]
    elif roles in _RUNTIME_SOURCE_ROLE_SETS:
        primary_role = "checkpoint" if "checkpoint" in roles else "diffusion"
    else:
        raise RuntimeError(f"native reconstruction recipe has unsupported source roles {roles}")
    return cast("str", _recipe_source(recipe, primary_role).name)


def _materialize_patch_sets(
    inference: Any,
    inference_torch: Any,
    recipe: Any,
    source_resolvers: Mapping[str, object],
) -> dict[str, object]:
    """Materialize builtin overlays into ordered per-component PatchSets.

    ProviderPatchRef materialization remains fail-loud until the first
    installable patch-provider pack triggers production catalog activation,
    as ledgered in ROADMAP. The worker-local provider seam itself is proven
    independently without making callable provider code part of the recipe.
    """
    if not recipe.overlays:
        return {}
    descriptor = _active_inference_registries().components.get(recipe.family_id)
    model_role = "diffusion" if descriptor is None else descriptor.model_role
    grouped: dict[str, dict[str, tuple[object, ...]]] = {}
    for overlay in recipe.overlays:
        asset = _source_asset(overlay.source, source_resolvers)
        path = resolve_weight_source(asset.local_path(), logical_name=asset.name)
        source = inference.load_safetensors_header(path)
        tensors = inference_torch.load_tensors(path, source.keys())
        by_component: dict[str, dict[object, object]] = {}
        for patch in overlay.patches:
            by_component.setdefault(patch.component, {})[patch.target] = patch.decoded
        for component, decoded in by_component.items():
            strength = (
                float(overlay.strength_model)
                if component in ("diffusion", model_role)
                else float(overlay.strength_clip)
            )
            if strength == 0.0:
                continue
            materialized = inference_torch.build_patch_set(
                decoded,
                tensors,
                strength=strength,
            )
            component_entries = grouped.setdefault(component, {})
            for key in materialized.keys():
                component_entries[key] = component_entries.get(key, ()) + tuple(
                    materialized.entries(key)
                )
    stack_digest = recipe.patch_stack_digest
    assert stack_digest is not None
    return {
        component: inference.PatchSet(entries, structural_digest=stack_digest)
        for component, entries in grouped.items()
    }


def _materialize_single_recipe_handle(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
) -> NativeRuntimeHandle:
    """Resolve recipe digests, verify its planned identity, and rebuild."""
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe_binding = getattr(getattr(recipe, "knobs", None), "embedding_binding_digest", None)
    if recipe_binding is not None:
        if embedding_resource is None:
            embedding_resource = _embedding_resource(inference_torch)
        index, _ = embedding_resource
        if index is None or index.binding_digest != recipe_binding:
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
    source_bindings = {binding.role: binding.source for binding in recipe.sources}
    if len(source_bindings) != len(recipe.sources):
        raise RuntimeError("native reconstruction recipe has duplicate source roles")
    canonical_bindings = _canonical_runtime_sources(source_bindings)
    paths: dict[str, Path] = {}
    for role, source_ref in canonical_bindings.items():
        asset = _source_asset(source_ref, source_resolvers)
        path = asset.local_path()
        paths[role] = (
            resolve_weight_source(path, logical_name=asset.name) if role == "checkpoint" else path
        )
    # A split recipe is one authority transaction: every source must resolve
    # and pass digest verification before any header is read.
    sources: dict[str, Any] = {
        role: inference.load_safetensors_header(
            path,
            asset_digest=source_bindings[role].digest,
            asset_size=source_bindings[role].size,
        )
        for role, path in paths.items()
    }
    plan = inference.plan_native(
        **sources,
        fp8_matmul=recipe.knobs.fp8_matmul,
    )
    resource = (
        (_embedding_resource(inference_torch) if embedding_resource is None else embedding_resource)
        if uses_classic_embedding_bindings(plan)
        else (None, None)
    )
    embedding_index, embedding_lookups = resource
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    if worker_binding != recipe_binding:
        raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
    component_identity = inference.runtime_component_identity(
        plan.family.id, plan.identity_components
    )
    if plan.family.id != recipe.family_id or (component_identity != recipe.component_identity):
        raise RuntimeError("resolved sources no longer match the reconstruction recipe")
    sampler_registry, _, registry_token = _sampler_registry(inference, recipe.knobs.registry_token)
    load_kwargs: dict[str, object] = {
        **sources,
        "expected_identity": recipe.runtime_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": recipe.knobs.fp8_matmul,
        "diffusion_dtype": _torch_dtype(torch, recipe.knobs.diffusion_dtype),
        "text_dtype": _torch_dtype(torch, recipe.knobs.text_dtype),
        "vae_dtype": _torch_dtype(torch, recipe.knobs.vae_dtype),
        "patch_overlay_digests": tuple(overlay.structural_digest for overlay in recipe.overlays),
    }
    if recipe.knobs.attention_route_token is not None:
        load_kwargs.update(
            attention_policy=recipe.knobs.attention_policy,
            attention_route_token=recipe.knobs.attention_route_token,
        )
    if worker_binding is not None:
        assert embedding_lookups is not None
        load_kwargs.update(
            embedding_lookups=embedding_lookups,
            embedding_binding_digest=worker_binding,
        )
    if registry_token is not None:
        load_kwargs.update(
            sampler_registry=sampler_registry,
            registry_token=registry_token,
            extension_behavior_hash=recipe.knobs.extension_behavior_hash,
        )
        generation = inference.materialize_inference_generation(registry_token)
        if generation.guidance_contributions:
            guidance_registry = inference_torch.GuidanceRegistry(generation.guidance_contributions)
            if guidance_registry.active:
                load_kwargs["guidance_executor"] = inference_torch.GuidanceExecutor(
                    guidance_registry
                )
    runtime = inference_torch.load_runtime(**load_kwargs)
    patch_sets = _materialize_patch_sets(inference, inference_torch, recipe, source_resolvers)

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, resource)

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        patch_sets=patch_sets,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )


@dataclass(frozen=True)
class _PreparedRecipeMaterialization:
    recipe: Any
    path: tuple[str, ...]
    sources: Mapping[str, Any]
    patch_sets: Mapping[str, object]
    load_extras: Mapping[str, object]
    materialization_key: str
    estimated_cost: int


def _recipe_nodes(recipe: Any) -> dict[tuple[str, ...], Any]:
    nodes: dict[tuple[str, ...], Any] = {}

    def walk(current: Any, path: tuple[str, ...]) -> None:
        nodes[path] = current
        for dependency in current.dependencies:
            walk(dependency.child, path + (dependency.child_id,))

    walk(recipe, ())
    return nodes


def _source_key_facts(source: Any) -> tuple[object, ...]:
    return (
        source.digest,
        source.name,
        source.size,
        source.media_type,
        source.virtual_path,
    )


def _materialization_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return "native-materialization:sha256:" + hashlib.sha256(encoded).hexdigest()


def _prevalidate_recipe_materializations(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None],
) -> tuple[Any, dict[tuple[str, ...], _PreparedRecipeMaterialization]]:
    """Resolve and validate the complete graph before residency can mutate."""
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    dependency_plan = inference.plan_dependencies(recipe)
    recipes = _recipe_nodes(recipe)
    if set(recipes) != {node.path for node in dependency_plan.nodes}:
        raise RuntimeError("native dependency plan does not match its recipe graph")

    embedding_index, _ = embedding_resource
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    nodes_by_path = {node.path: node for node in dependency_plan.nodes}
    for path, current in recipes.items():
        node = nodes_by_path[path]
        if node.runtime_identity != current.runtime_identity:
            raise RuntimeError("native dependency plan identity does not match recipe")
        if path and node.clone_mode != "with-parent":
            raise RuntimeError("native dependency clone mode 'shared' is not supported")
        if path and node.scope in ("contribution", "invocation"):
            raise RuntimeError(f"native dependency scope {node.scope!r} is not supported")
        if (
            current.knobs.embedding_binding_digest is not None
            and current.knobs.embedding_binding_digest != worker_binding
        ):
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
        source_bindings = {binding.role: binding.source for binding in current.sources}
        if len(source_bindings) != len(current.sources):
            raise RuntimeError("native reconstruction recipe has duplicate source roles")
        _canonical_runtime_sources(source_bindings)

    validated: dict[
        tuple[str, ...],
        tuple[
            Mapping[str, Any],
            Mapping[str, object],
            Mapping[str, object],
            int,
        ],
    ] = {}
    for path, current in recipes.items():
        source_bindings = _canonical_runtime_sources(
            {binding.role: binding.source for binding in current.sources}
        )
        paths: dict[str, Path] = {}
        estimated_cost = 0
        for role, source_ref in source_bindings.items():
            asset = _source_asset(source_ref, source_resolvers)
            source_path = asset.local_path()
            paths[role] = (
                resolve_weight_source(source_path, logical_name=asset.name)
                if role == "checkpoint"
                else source_path
            )
            estimated_cost += source_ref.size
        for overlay in current.overlays:
            overlay_asset = _source_asset(overlay.source, source_resolvers)
            overlay_path = resolve_weight_source(
                overlay_asset.local_path(), logical_name=overlay_asset.name
            )
            inference.load_safetensors_header(overlay_path)
            estimated_cost += overlay.source.size
        sources = {
            role: inference.load_safetensors_header(
                source_path,
                asset_digest=source_bindings[role].digest,
                asset_size=source_bindings[role].size,
            )
            for role, source_path in paths.items()
        }
        native_plan = inference.plan_native(
            **sources,
            fp8_matmul=current.knobs.fp8_matmul,
        )
        binding = worker_binding if uses_classic_embedding_bindings(native_plan) else None
        if binding != current.knobs.embedding_binding_digest:
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
        component_identity = inference.runtime_component_identity(
            native_plan.family.id, native_plan.identity_components
        )
        if native_plan.family.id != current.family_id or (
            component_identity != current.component_identity
        ):
            raise RuntimeError("resolved sources no longer match the reconstruction recipe")
        for dtype in (
            current.knobs.diffusion_dtype,
            current.knobs.text_dtype,
            current.knobs.vae_dtype,
        ):
            _torch_dtype(torch, dtype)
        patch_sets = _materialize_patch_sets(inference, inference_torch, current, source_resolvers)
        component_plans = {
            component.component: component
            for component in native_plan.identity_components
            if component is not None
        }
        unknown_components = set(patch_sets) - component_plans.keys()
        if unknown_components:
            raise RuntimeError(
                "native dependency patches name unknown components: "
                + ", ".join(sorted(unknown_components))
            )
        for component, patch_set in patch_sets.items():
            unknown_targets = set(patch_set.keys()) - component_plans[component].keys.keys()
            if unknown_targets:
                raise RuntimeError(
                    f"native dependency patches name unknown {component!r} targets: "
                    + ", ".join(sorted(unknown_targets))
                )
        load_extras: dict[str, object] = {}
        sampler_registry, _, registry_token = _sampler_registry(
            inference, current.knobs.registry_token
        )
        if registry_token is not None:
            load_extras.update(
                sampler_registry=sampler_registry,
                registry_token=registry_token,
                extension_behavior_hash=current.knobs.extension_behavior_hash,
            )
            generation = inference.materialize_inference_generation(registry_token)
            if generation.guidance_contributions:
                guidance_registry = inference_torch.GuidanceRegistry(
                    generation.guidance_contributions
                )
                if guidance_registry.active:
                    load_extras["guidance_executor"] = inference_torch.GuidanceExecutor(
                        guidance_registry
                    )
        validated[path] = (sources, patch_sets, load_extras, estimated_cost)

    prepared: dict[tuple[str, ...], _PreparedRecipeMaterialization] = {}
    for path in dependency_plan.load_order:
        current = recipes[path]
        sources, patch_sets, prepared_load_extras, estimated_cost = validated[path]
        node = nodes_by_path[path]
        child_keys = tuple(
            (dependency.child_id, prepared[path + (dependency.child_id,)].materialization_key)
            for dependency in current.dependencies
        )
        edge_facts = None
        if path:
            edge_facts = (
                path[-1],
                node.residency_group,
                node.scope,
                node.clone_mode,
                node.accounting_owner,
            )
        knob_facts = {
            "diffusion_dtype": current.knobs.diffusion_dtype,
            "text_dtype": current.knobs.text_dtype,
            "vae_dtype": current.knobs.vae_dtype,
            "fp8_matmul": current.knobs.fp8_matmul,
            "registry_token": current.knobs.registry_token,
            "extension_behavior_hash": current.knobs.extension_behavior_hash,
            "embedding_binding_digest": current.knobs.embedding_binding_digest,
        }
        if current.knobs.attention_route_token is not None:
            protocol = importlib.import_module("dinkster_protocol")
            knob_facts.update(
                attention_policy=current.knobs.attention_policy,
                attention_route_token=protocol.attention_route_token_to_wire(
                    current.knobs.attention_route_token
                ),
            )
        payload = {
            "path": path,
            "edge": edge_facts,
            "runtime_identity": current.runtime_identity,
            "family_id": current.family_id,
            "component_identity": current.component_identity,
            "knobs": knob_facts,
            "sources": tuple(
                (binding.role, *_source_key_facts(binding.source)) for binding in current.sources
            ),
            "overlays": tuple(
                (
                    overlay.structural_digest,
                    *_source_key_facts(overlay.source),
                    overlay.dialect,
                    overlay.key_map,
                    overlay.strength_model,
                    overlay.strength_clip,
                )
                for overlay in current.overlays
            ),
            "attachments": tuple(
                (
                    attachment.name,
                    attachment.clone,
                    attachment.device,
                    attachment.rebuild_data,
                    attachment.rebuild_entry_point,
                )
                for attachment in current.attachments
            ),
            "children": child_keys,
        }
        prepared[path] = _PreparedRecipeMaterialization(
            current,
            path,
            sources,
            patch_sets,
            prepared_load_extras,
            _materialization_digest(payload),
            estimated_cost,
        )
    return dependency_plan, prepared


def _dependency_account_identities(
    dependency_plan: Any,
    prepared: Mapping[tuple[str, ...], _PreparedRecipeMaterialization],
) -> dict[tuple[str, ...], str]:
    root_key = prepared[()].materialization_key
    accounts = {(): _materialization_digest((root_key, "root", "parent"))}
    for group in dependency_plan.residency_groups:
        parent_key = prepared[group.declaring_path].materialization_key
        owner = "parent" if group.owner_path == group.declaring_path else group.owner_path[-1]
        account = _materialization_digest((parent_key, "residency-group", group.name, owner))
        for member_path in group.member_paths:
            accounts[member_path] = account
    if set(accounts) != set(prepared):
        raise RuntimeError("native dependency accounting does not cover every node")
    return accounts


def _assemble_prevalidated_handle(
    prepared: _PreparedRecipeMaterialization,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None],
) -> NativeRuntimeHandle:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe = prepared.recipe
    embedding_index, embedding_lookups = (
        (None, None) if recipe.knobs.embedding_binding_digest is None else embedding_resource
    )
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    load_kwargs: dict[str, object] = {
        **prepared.sources,
        "expected_identity": recipe.runtime_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": recipe.knobs.fp8_matmul,
        "diffusion_dtype": _torch_dtype(torch, recipe.knobs.diffusion_dtype),
        "text_dtype": _torch_dtype(torch, recipe.knobs.text_dtype),
        "vae_dtype": _torch_dtype(torch, recipe.knobs.vae_dtype),
        "patch_overlay_digests": tuple(overlay.structural_digest for overlay in recipe.overlays),
    }
    if recipe.knobs.attention_route_token is not None:
        load_kwargs.update(
            attention_policy=recipe.knobs.attention_policy,
            attention_route_token=recipe.knobs.attention_route_token,
        )
    if worker_binding is not None:
        assert embedding_lookups is not None
        load_kwargs.update(
            embedding_lookups=embedding_lookups,
            embedding_binding_digest=worker_binding,
        )
    load_kwargs.update(prepared.load_extras)
    runtime = inference_torch.load_runtime(**load_kwargs)

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, embedding_resource)

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        patch_sets=prepared.patch_sets,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )


def _materialize_recipe_handle(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
) -> NativeRuntimeHandle:
    if not getattr(recipe, "dependencies", ()):
        return _materialize_single_recipe_handle(
            recipe, source_resolvers, torch, embedding_resource
        )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    resource = (
        _embedding_resource(inference_torch) if embedding_resource is None else embedding_resource
    )
    dependency_plan, prepared = _prevalidate_recipe_materializations(
        recipe, source_resolvers, torch, resource
    )
    accounts = _dependency_account_identities(dependency_plan, prepared)
    pool: NativeResidencyPool = default_dependency_residency_pool()

    def assemble_paths(
        paths: tuple[tuple[str, ...], ...],
    ) -> dict[tuple[str, ...], NativeRuntimeHandle]:
        built: dict[tuple[str, ...], NativeRuntimeHandle] = {}
        try:
            for path in paths:
                node = prepared[path]
                token = pool.acquire(
                    node.materialization_key,
                    accounts[path],
                    node.estimated_cost,
                )
                try:
                    handle = _assemble_prevalidated_handle(node, source_resolvers, torch, resource)
                except BaseException:
                    pool.release(token)
                    raise
                handle.bind_residency_token(pool, token)
                built[path] = handle
        except BaseException as exc:
            for path in reversed(tuple(built)):
                try:
                    built[path].terminal_release()
                except BaseException as cleanup:
                    exc.add_note(
                        f"dependency construction rollback for {path!r} also failed: {cleanup!r}"
                    )
            raise
        return built

    initial = dependency_plan.select_active(frozenset()).load_order
    built = assemble_paths(initial)
    root = built.pop(())

    def materialize_conditional(
        active_conditional: frozenset[tuple[str, ...]],
    ) -> Mapping[tuple[str, ...], NativeRuntimeHandle]:
        selected = dependency_plan.select_active(active_conditional)
        existing = root.owned_dependencies
        missing = tuple(path for path in selected.load_order if path and path not in existing)
        return assemble_paths(missing)

    try:
        root.bind_dependencies(dependency_plan, built, materialize_conditional)
    except BaseException as exc:
        for handle in (root, *reversed(tuple(built.values()))):
            if handle.released:
                continue
            try:
                handle.terminal_release()
            except BaseException as cleanup:
                exc.add_note(f"dependency binding rollback also failed: {cleanup!r}")
        raise
    return root


def _logical_lora_key_map(
    inference: Any,
    handle: NativeRuntimeHandle,
    text_handle: NativeComponentHandle | None = None,
) -> dict[str, object]:
    assembled = handle.runtime.assembled
    diffusion_keys = tuple(f"diffusion_model.{key}" for key in assembled.diffusion.state_dict())
    key_map: dict[str, object] = dict(inference.native_unet_key_map(diffusion_keys))
    config = getattr(assembled.diffusion, "config", None)
    if isinstance(config, inference.UNetConfig):
        key_map.update(inference.sd_unet_diffusers_key_map(diffusion_keys, config))
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        key_map.update(inference.flux_linear1_qkv_key_map(diffusion_keys, hidden_size))
    if handle.recipe.family_id == inference.Z_IMAGE_CONFIG.family_id:
        hidden_width = getattr(config, "hidden_width", None)
        if not isinstance(hidden_width, int) or hidden_width <= 0:
            raise RuntimeError("native Z-Image runtime has no valid hidden width")
        key_map.update(inference.z_image_diffusers_key_map(diffusion_keys, hidden_width))
    family = inference.builtin_family_registry().get(handle.recipe.family_id)
    hook = None if family is None else family.engine.feature_hook("lora-key-map")
    if hook is not None:
        module_name, attribute = hook.target.split(":", 1)
        key_map.update(getattr(importlib.import_module(module_name), attribute)(diffusion_keys))
    clip_keys: list[str] = []
    for component, logical_component in (
        ("clip_l", "clip_l"),
        ("clip_g", "clip_g"),
        ("t5xxl", "t5xxl"),
        ("umt5xxl", "t5xxl"),
    ):
        module = getattr(assembled, component, None)
        if module is None:
            continue
        clip_keys.extend(f"{logical_component}.transformer.{key}" for key in module.state_dict())
    if text_handle is not None:
        text_recipe = text_handle.recipe
        if text_recipe is None or len(text_recipe.sources) != 1:
            raise TypeError("split Flux2 text handle has no exact retained component recipe")
        text_role = text_recipe.sources[0].role
        clip_keys.extend(
            f"{text_role}.transformer.model.{key}"
            for key in cast("Any", text_handle.module).state_dict()
        )
    key_map.update(inference.clip_lora_key_map(clip_keys))
    return key_map


def _lora_target_routes(assembled: object) -> tuple[tuple[str, str], ...]:
    routes = [
        (prefix, component)
        for prefix, component in (
            ("diffusion_model.", "diffusion"),
            ("clip_l.transformer.", "clip_l"),
            ("clip_g.transformer.", "clip_g"),
        )
        if getattr(assembled, component, None) is not None
    ]
    if getattr(assembled, "umt5xxl", None) is not None:
        routes.append(("t5xxl.transformer.", "umt5xxl"))
    elif getattr(assembled, "t5xxl", None) is not None:
        routes.append(("t5xxl.transformer.", "t5xxl"))
    declared: Mapping[str, object] | None = getattr(assembled, "components", None)
    if declared is not None:
        names = {id(module): name for name, module in declared.items()}
        return tuple(
            (prefix, names[id(getattr(assembled, component))]) for prefix, component in routes
        )
    return tuple(routes)


def _route_patch_target(
    inference: Any,
    handle: NativeRuntimeHandle,
    target: Any,
    text_role: str | None = None,
) -> tuple[str, Any]:
    for prefix, component in _lora_target_routes(handle.runtime.assembled):
        if target.key.startswith(prefix):
            return component, inference.PatchTarget(
                target.key.removeprefix(prefix), offset=target.offset
            )
    if text_role is not None:
        prefix = f"{text_role}.transformer.model."
        if target.key.startswith(prefix):
            return text_role, inference.PatchTarget(
                target.key.removeprefix(prefix), offset=target.offset
            )
    raise RuntimeError(f"decoded LoRA target {target.key!r} has no native component route")


def _native_lora_overlay(
    handle: NativeRuntimeHandle,
    lora: AssetRef,
    strength_model: float,
    strength_clip: float,
    *,
    text_handle: NativeComponentHandle | None = None,
) -> Any:
    for name, value in (
        ("strength_model", strength_model),
        ("strength_clip", strength_clip),
    ):
        if not -100.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [-100.0, 100.0], got {value}")
    inference = importlib.import_module("dinkster_inference")
    lora_path = resolve_weight_source(lora.local_path(), logical_name=lora.name)
    source = inference.load_safetensors_header(lora_path)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    key_map = _logical_lora_key_map(inference, handle, text_handle)
    decoded = inference.decode_lora(geometries, key_map)
    for diagnostic in decoded.diagnostics:
        log.warning("native LoRA %s: %s", lora.digest, diagnostic)
    if decoded.unmatched:
        log.warning(
            "native LoRA %s has %d unmatched keys: %s",
            lora.digest,
            len(decoded.unmatched),
            ", ".join(decoded.unmatched),
        )
    patches: list[object] = []
    text_role = (
        None
        if text_handle is None or text_handle.recipe is None
        else text_handle.recipe.sources[0].role
    )
    for target, patch in decoded.patches.items():
        component, routed = _route_patch_target(inference, handle, target, text_role)
        patches.append(inference.OverlayPatch(component, routed, patch))
    if not patches:
        raise RuntimeError(f"native LoRA {lora.digest} has no patch keys matching this model")
    return inference.PatchOverlay.from_decoded(
        source=_weight_source_ref(inference, lora),
        dialect=decoded.dialect,
        key_map=f"native.{handle.recipe.family_id}.v1",
        strength_model=strength_model,
        strength_clip=strength_clip,
        patches=tuple(patches),
    )


def _overlay_for_components(inference: Any, overlay: Any, components: set[str]) -> Any | None:
    patches = tuple(patch for patch in overlay.patches if patch.component in components)
    if not patches:
        return None
    return inference.PatchOverlay.from_decoded(
        source=overlay.source,
        dialect=overlay.dialect,
        key_map=overlay.key_map,
        strength_model=float(overlay.strength_model),
        strength_clip=float(overlay.strength_clip),
        patches=patches,
    )


def _native_handle(value: object, input_id: str) -> NativeRuntimeHandle:
    if isinstance(value, _NativeModelOverlay):
        value = value.handle
    if not isinstance(value, NativeRuntimeHandle):
        raise TypeError(
            f"{input_id} must be a native-arm runtime handle, got {type(value).__name__}"
        )
    value.require_active()
    return value


@dataclass(frozen=True)
class _ZImageControlBinding:
    handle: NativeComponentHandle
    image: object
    strength: float


@dataclass(frozen=True, eq=False)
class _NativeControlNetResource:
    handle: NativeComponentHandle | None
    asset_digest: str
    resource_digest: str
    source_layout: str
    descriptor: Any = None
    plan: Any = None
    hint_channels: int = 3
    mode: Any = None

    @property
    def _dinkster_resident_owner(self) -> object:
        return self if self.handle is None else self.handle


@dataclass(frozen=True, slots=True)
class _ControlHintSnapshot:
    shape: tuple[int, int, int, int]
    data: bytes
    digest: str


@dataclass(frozen=True, slots=True)
class _ClassicControlEntry:
    child_id: str
    resident_id: str
    model_digest: str
    hint: _ControlHintSnapshot


@dataclass(frozen=True, slots=True)
class _ClassicControlBinding:
    application: Any
    entries: tuple[_ClassicControlEntry, ...]
    apply_to_uncond: bool


@dataclass(frozen=True, eq=False)
class _ControlledConditioning:
    conditioning: Any
    binding: _ClassicControlBinding
    resources: tuple[_NativeControlNetResource, ...]

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return tuple(
            resource if resource.handle is None else resource.handle for resource in self.resources
        )

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        from dinkster_inference import encode_conditioning_carrier, encode_control_application
        from dinkster_values import stable_hash

        return stable_hash(
            [
                encode_conditioning_carrier(self.conditioning),
                encode_control_application(self.binding.application),
                str(self.binding.apply_to_uncond).encode("ascii"),
                *(resource.resource_digest.encode("ascii") for resource in self.resources),
            ]
        )


def _controlled_conditioning(value: object) -> _ControlledConditioning | None:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, inference.ResidentConditioningCarrier):
        payload = cast("Any", value).payload
        if isinstance(payload, _ControlledConditioning):
            return payload
    return None


def _control_resident_refs(*values: object) -> tuple[object, ...]:
    return tuple(
        resource
        for value in values
        if (controlled := _controlled_conditioning(value)) is not None
        for resource in controlled.resources
    )


@dataclass(frozen=True, eq=False)
class _NativeModelOverlay:
    """Diffusion overlays over one shared resident runtime."""

    handle: NativeRuntimeHandle
    overlays: tuple[Any, ...]
    source_resolvers: Mapping[str, object]
    z_image_control: _ZImageControlBinding | None = None
    sampling_shift: float | None = None
    guidance_transforms: tuple[tuple[str, Any], ...] = ()
    context_windows: ContextWindowsSpec | None = None
    chroma_radiance_options: tuple[Any, ...] = ()
    sampling_cache: Any | None = None
    sampling_timeline: Any | None = None
    sampling_space: Any | None = None

    def __post_init__(self) -> None:
        if self.sampling_space is not None and self.sampling_shift is not None:
            raise ValueError("sampling space and sampling shift are mutually exclusive")
        if self.z_image_control is not None:
            self.z_image_control.handle.register_dependent(self)

    @property
    def _dinkster_resident_owner(self) -> NativeRuntimeHandle:
        return self.handle

    @property
    def _dinkster_application_identity(self) -> str:
        identity = self.handle.recipe.runtime_identity
        if any(
            contribution is _DISABLE_CFG1_OPTIMIZATION
            for _, contribution in self.guidance_transforms
        ):
            inference = importlib.import_module("dinkster_inference")
            return cast(
                "str",
                inference.extend_runtime_identity(
                    identity,
                    ("guidance.disable_cfg1_optimization=true",),
                ),
            )
        return identity

    @property
    def runtime(self) -> Any:
        return self.handle.runtime

    @property
    def recipe(self) -> Any:
        return self.handle.recipe

    @property
    def load_device(self) -> object:
        return self.handle.load_device

    def require_active(self) -> None:
        self.handle.require_active()

    def stage(self, role: str) -> Any:
        return self.handle.stage(role)


class _NativeCodecHandle:
    """Codec-only view of one compat-owned native runtime."""

    def __init__(self, handle: NativeRuntimeHandle) -> None:
        inference = importlib.import_module("dinkster_inference")
        inference.require_inference_runtime_handle(handle, "handle")
        codec = getattr(handle.runtime, "codec", None)
        descriptor = getattr(codec, "descriptor", None)
        if not isinstance(descriptor, inference.CodecDescriptor):
            raise TypeError("native runtime does not expose a CodecDescriptor")
        self._handle = handle
        self._descriptor = descriptor
        self._resource_identity = handle.recipe.runtime_identity

    @property
    def _dinkster_resident_owner(self) -> NativeRuntimeHandle:
        return self._handle

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def load_device(self) -> object:
        return self._handle.load_device

    @property
    def accepts_batched_video(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "accepts_batched_video", False))

    @property
    def accepts_image_batch_latent(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "accepts_image_batch_latent", False))

    @property
    def manages_input_device(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "manages_input_device", False))

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self) -> Any:
        return self._handle.stage("vae")

    def _direct(self, value: Any, direction: str) -> Any:
        codec = self._handle.runtime.codec
        operation = (
            self._handle.runtime.decode_latent
            if direction == "decode"
            else self._handle.runtime.encode_content
        )
        return _run_direct_vae(
            handle=self._handle,
            value=value,
            direction=direction,
            operation=operation,
            codec=codec,
        )

    def decode_latent(self, latent: Any) -> Any:
        return self._direct(latent, "decode")

    def decode_latent_tiled(
        self,
        latent: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._handle.runtime.codec.decode_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._direct(content, "encode")

    def encode_content_tiled(
        self,
        content: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._handle.runtime.codec.encode_tiled(content, tile=tile, overlap=overlap)


class _PixelSpaceCodecHandle:
    """Stateless pixel codec published through the standard codec seam."""

    def __init__(self, resource_identity: str, compute_dtype: str) -> None:
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        torch = _torch()
        self._codec = inference_torch.PixelSpaceCodec(
            compute_dtype=_torch_dtype(torch, compute_dtype)
        )
        self._descriptor = inference.CodecDescriptor(
            id="dinkster.chroma_radiance_pixel_space",
            display_name="Chroma Radiance Pixel Space",
            kind="image",
            latent=inference.CHROMA_RADIANCE.single_stream_latent(),
            supported_dtypes=inference.CHROMA_RADIANCE.supported_dtypes,
            supports_tiling=False,
        )
        self._resource_identity = resource_identity
        self._load_device = torch.device("cpu")

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def load_device(self) -> object:
        return self._load_device

    def require_active(self) -> None:
        return None

    @contextmanager
    def stage(self) -> Any:
        yield

    def decode_latent(self, latent: Any) -> Any:
        return self._codec.decode(latent)

    def encode_content(self, content: Any) -> Any:
        return self._codec.encode(content)


def _native_model(
    value: object, input_id: str
) -> tuple[
    NativeRuntimeHandle,
    tuple[Any, ...],
    Mapping[str, object],
    _ZImageControlBinding | None,
    float | None,
    tuple[tuple[str, Any], ...],
    ContextWindowsSpec | None,
    tuple[Any, ...],
]:
    if isinstance(value, _NativeModelOverlay):
        value.handle.require_active()
        return (
            value.handle,
            value.overlays,
            value.source_resolvers,
            value.z_image_control,
            value.sampling_shift,
            value.guidance_transforms,
            value.context_windows,
            value.chroma_radiance_options,
        )
    return _native_handle(value, input_id), (), {}, None, None, (), None, ()


def _native_model_sampling_cache(value: object) -> object | None:
    return value.sampling_cache if isinstance(value, _NativeModelOverlay) else None


def _native_model_sampling_timeline(value: object) -> object | None:
    return value.sampling_timeline if isinstance(value, _NativeModelOverlay) else None


def _native_model_sampling_space(value: object) -> Any | None:
    return value.sampling_space if isinstance(value, _NativeModelOverlay) else None


def _cfg1_optimization_setting(
    transforms: tuple[tuple[str, Any], ...],
) -> tuple[tuple[tuple[str, Any], ...], bool]:
    ordinary = tuple(
        (owner, contribution)
        for owner, contribution in transforms
        if contribution is not _DISABLE_CFG1_OPTIMIZATION
    )
    return ordinary, len(ordinary) != len(transforms)


def _sampling_space_runtime(runtime: Any, space: Any | None) -> Any:
    if space is None:
        return runtime
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(runtime, inference.SamplingSpaceOverrideRuntime):
        raise TypeError("model runtime does not support a sampling-space override")
    return runtime.with_sampling_space(space)


_ExecuteP = ParamSpec("_ExecuteP")


def _bind_model_sampling_options(
    execute: Callable[_ExecuteP, Mapping[str, object]],
) -> Callable[_ExecuteP, Mapping[str, object]]:
    @wraps(execute)
    def wrapped(*args: _ExecuteP.args, **kwargs: _ExecuteP.kwargs) -> Mapping[str, object]:
        model = cast("Mapping[str, object]", kwargs).get("model")
        inference = importlib.import_module("dinkster_inference")
        if isinstance(model, inference.ApplicationChain):
            model = cast("Any", model).model
        with (
            inference.use_sampling_cache(_native_model_sampling_cache(model)),
            inference.use_sampling_timeline(_native_model_sampling_timeline(model)),
        ):
            return execute(*args, **kwargs)

    return cast("Callable[_ExecuteP, Mapping[str, object]]", wrapped)


def _context_windows_sampling_kwargs(
    runtime: Any, context_windows: ContextWindowsSpec | None, description: str
) -> dict[str, Any]:
    """Admission for KSampler context windows; the runtime revalidates its profile."""
    if context_windows is None:
        return {}
    inference = importlib.import_module("dinkster_inference")
    if not (
        isinstance(runtime, inference.ContextWindowsRuntime) and runtime.supports_context_windows
    ):
        raise ValueError(f"{description} does not support context windows")
    return {"context_windows": context_windows}


def _application_chain_model(value: object, input_id: str) -> tuple[object, tuple[Any, ...]]:
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(value, inference.ApplicationChain):
        return value, ()
    chain = cast("Any", value)
    handle = inference.require_inference_runtime_handle(chain.model, input_id)
    if chain.base_model_identity != handle.recipe.runtime_identity:
        raise ValueError(f"{input_id} application chain base model identity changed")
    family_id = handle.recipe.family_id
    applications = cast("tuple[Any, ...]", chain.applications)
    for index, application in enumerate(applications):
        inference.require_inference_component_handle(
            application.handle,
            f"{input_id} application {index}",
        )
        if application.family_id != family_id:
            raise ValueError(
                f"{input_id} application {index} family_id must match the base model family"
            )
    return chain.model, applications


@contextmanager
def _staged_applications(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    *,
    stage_runtime: bool = True,
) -> Any:
    with ExitStack() as stages:
        for application in applications:
            stages.enter_context(
                application.handle.stage_with(handle, application.role)
                if stage_runtime
                else application.handle.stage()
            )
        yield


def _application_kwargs(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    latent: object,
    *,
    reserved_keys: set[str],
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for index, application in enumerate(applications):
        component = application.handle.component
        materialized = application.materialize_application_kwargs(
            handle.runtime,
            component,
            latent,
        )
        if not isinstance(materialized, Mapping):
            raise TypeError(
                f"model application {index} materialize_application_kwargs must return a mapping"
            )
        for key, item in cast("Mapping[object, object]", materialized).items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"model application {index} kwarg names must be non-empty strings")
            if key in reserved_keys:
                raise ValueError(f"model application {index} kwarg {key!r} collides")
            if key == "sd15_attention_contributions" and key in kwargs:
                previous = kwargs[key]
                if type(previous) is not tuple or type(item) is not tuple:
                    raise TypeError("SD1.5 attention contributions must materialize as tuples")
                kwargs[key] = (*previous, *item)
                continue
            if key in kwargs:
                raise ValueError(f"model application {index} kwarg {key!r} collides")
            kwargs[key] = item
    return kwargs


@contextmanager
def _materialized_application_kwargs(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    latent: object,
    *,
    reserved_keys: set[str],
    stage_runtime: bool = True,
) -> Any:
    with _staged_applications(applications, handle, stage_runtime=stage_runtime):
        yield _application_kwargs(
            applications,
            handle,
            latent,
            reserved_keys=reserved_keys,
        )


def _overlay_targets(overlay: object, component: str) -> bool:
    patches = getattr(overlay, "patches", None)
    if patches is None:
        return True
    if component == "text":
        return any(patch.component != "diffusion" for patch in patches)
    return any(patch.component == component for patch in patches)


def _overlay_has_offsets(overlay: object) -> bool:
    patches = getattr(overlay, "patches", ())
    return any(
        getattr(getattr(patch, "target", None), "offset", None) is not None for patch in patches
    )


def _inpaint_conditioning(
    metadata: Mapping[object, object], input_id: str, torch: Any, inference: Any
) -> Any | None:
    concat_mask = metadata.get("concat_mask")
    concat_latent = metadata.get("concat_latent_image")
    if (concat_mask is None) != (concat_latent is None):
        raise ValueError(
            f"{input_id} inpaint conditioning requires both concat_mask and concat_latent_image"
        )
    if concat_mask is None:
        return None
    if not isinstance(concat_mask, torch.Tensor):
        raise TypeError(f"{input_id} concat_mask must be a torch.Tensor")
    if not isinstance(concat_latent, torch.Tensor):
        raise TypeError(f"{input_id} concat_latent_image must be a torch.Tensor")
    return inference.InpaintConditioning(mask=concat_mask, masked_image=concat_latent)


def _scheduled_inpaint(value: object, input_id: str, torch: Any, inference: Any) -> Any | None:
    resolved = tuple(
        _inpaint_conditioning(cast("Mapping[object, object]", entry[1]), input_id, torch, inference)
        for entry in _condition_entries(value, input_id)
    )
    present = tuple(item for item in resolved if item is not None)
    if not present:
        return None
    first = present[0]
    if len(present) != len(resolved) or any(
        item.mask is not first.mask or item.masked_image is not first.masked_image
        for item in present[1:]
    ):
        raise ValueError(f"{input_id} scheduled inpaint conditioning must share concat values")
    return first


def _conditioning(value: object, input_id: str, torch: Any, inference: Any) -> tuple[Any, Any]:
    """Decode text plus the narrow native SD inpaint conditioning subset."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(
            f"{input_id} conditioning must contain exactly one [tensor, metadata] entry"
        )
    entries = cast("Sequence[object]", value)
    if len(entries) != 1:
        raise ValueError(
            f"{input_id} conditioning must contain exactly one entry, got {len(entries)}"
        )
    entry = entries[0]
    if isinstance(entry, str | bytes) or not isinstance(entry, Sequence):
        raise ValueError(f"{input_id} conditioning entry must be [tensor, metadata]")
    parts = cast("Sequence[object]", entry)
    if len(parts) != 2:
        raise ValueError(
            f"{input_id} conditioning entry must have exactly two elements, got {len(parts)}"
        )
    embeddings, metadata = parts
    if type(embeddings) is inference.PreparedMultiStreamConditioning:
        if type(metadata) is not dict or metadata:
            raise ValueError(
                f"{input_id} prepared multi-stream conditioning requires empty metadata"
            )
        return embeddings, None
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"{input_id} conditioning embeddings must be a torch.Tensor, "
            f"got {type(embeddings).__name__}"
        )
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{input_id} conditioning metadata must be a mapping")
    metadata_map = cast("Mapping[object, object]", metadata)
    unsupported = set(metadata_map) - {
        "pooled_output",
        "concat_mask",
        "concat_latent_image",
        "control",
        "control_apply_to_uncond",
        _NATIVE_PROMPT_KEY,
        _NATIVE_PREPARED_CONDITIONING_KEY,
    }
    if unsupported:
        raise ValueError(
            f"{input_id} conditioning metadata is unsupported on the native arm: "
            + ", ".join(sorted(repr(key) for key in unsupported))
        )
    prepared = metadata_map.get(_NATIVE_PREPARED_CONDITIONING_KEY)
    if prepared is not None:
        if set(metadata_map) - {"control", "control_apply_to_uncond"} != {
            _NATIVE_PREPARED_CONDITIONING_KEY
        }:
            raise ValueError(f"{input_id} prepared conditioning cannot carry legacy metadata")
        if not isinstance(prepared, inference.Conditioning):
            raise TypeError(f"{input_id} prepared conditioning must be a Conditioning value")
        typed_prepared = cast("Any", prepared)
        if typed_prepared.embeddings is not embeddings:
            raise ValueError(f"{input_id} prepared conditioning embeddings are inconsistent")
        return prepared, None
    pooled = metadata_map.get("pooled_output")
    if pooled is not None and not isinstance(pooled, torch.Tensor):
        raise TypeError(
            f"{input_id} pooled_output must be a torch.Tensor when present, "
            f"got {type(pooled).__name__}"
        )
    inpaint = _inpaint_conditioning(metadata_map, input_id, torch, inference)
    return inference.Conditioning(embeddings=embeddings, pooled=pooled), inpaint


def _conditioning_classic_control(value: object, input_id: str) -> _ClassicControlBinding | None:
    if type(value) is _GuidedRows:
        value = value.rows
    controlled = _controlled_conditioning(value)
    if controlled is not None:
        return controlled.binding
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, (inference.ConditioningCarrier, inference.ResidentConditioningCarrier)):
        return None
    entries = _condition_entries(value, input_id)
    if len(entries) != 1:
        if any(
            "control" in cast("Mapping[object, object]", entry[1])
            or "control_apply_to_uncond" in cast("Mapping[object, object]", entry[1])
            for entry in entries
        ):
            raise ValueError(
                f"{input_id} classic ControlNet requires exactly one conditioning entry"
            )
        return None
    metadata = cast("Mapping[object, object]", entries[0][1])
    raw_binding = metadata.get("control")
    raw_apply_to_uncond = metadata.get("control_apply_to_uncond")
    if raw_binding is None:
        if raw_apply_to_uncond is not None:
            raise ValueError(
                f"{input_id} control_apply_to_uncond is present without a control binding"
            )
        return None
    if not isinstance(raw_binding, _ClassicControlBinding):
        raise TypeError(f"{input_id} control metadata is not a native classic ControlNet binding")
    if type(raw_apply_to_uncond) is not bool:
        raise TypeError(f"{input_id} control_apply_to_uncond must be a bool")
    if raw_apply_to_uncond is not raw_binding.apply_to_uncond:
        raise ValueError(f"{input_id} control apply-to-uncond metadata is inconsistent")
    return raw_binding


def _select_classic_control_binding(
    positive: object, negative: object
) -> _ClassicControlBinding | None:
    positive_binding = _conditioning_classic_control(positive, "positive")
    negative_binding = _conditioning_classic_control(negative, "negative")
    if positive_binding is None and negative_binding is None:
        return None
    if positive_binding is None:
        raise ValueError("negative conditioning cannot carry classic ControlNet by itself")
    if negative_binding == positive_binding:
        return positive_binding
    if negative_binding is None and positive_binding.apply_to_uncond:
        return positive_binding
    raise ValueError(
        "positive and negative classic ControlNet chains must match; use "
        "ControlNetApplyAdvanced on both conditioning inputs"
    )


def _require_classic_control_keyword(receiver: Callable[..., Any]) -> None:
    try:
        inspect.signature(receiver).bind_partial(None, control=None)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "classic ControlNet requires the selected sampler to accept the 'control' keyword"
        ) from exc


def _materialize_classic_control(
    binding: _ClassicControlBinding | None,
    *,
    latent_batch: int,
    torch: Any,
    inference: Any,
    inference_torch: Any,
    base_handle: Any = None,
    cleanup: ExitStack | None = None,
) -> tuple[Any | None, tuple[NativeComponentHandle, ...]]:
    from dinkster_inference.component_registry import execution_symbol

    if binding is None:
        return None, ()
    applications: list[Any] = []
    application = binding.application
    while application is not None:
        if type(application) is not inference.ControlApplication:
            raise TypeError("classic ControlNet application metadata is malformed")
        applications.append(application)
        application = application.previous
    applications.reverse()
    if len(applications) != len(binding.entries):
        raise ValueError("classic ControlNet application and resource chains differ in length")
    pool = default_pool()
    resolved = None
    handles: list[NativeComponentHandle] = []
    seen_handles: set[int] = set()
    materialized: dict[str, NativeComponentHandle] = {}
    for application, entry in zip(applications, binding.entries, strict=True):
        if application.child_id != entry.child_id:
            raise ValueError("classic ControlNet application and resource chains are misordered")
        prefix = "resident:"
        if not entry.resident_id.startswith(prefix) or len(entry.resident_id) == len(prefix):
            raise ValueError("classic ControlNet resident identity is malformed")
        resource = pool.get(entry.resident_id[len(prefix) :])
        if not isinstance(resource, _NativeControlNetResource):
            raise TypeError("classic ControlNet resident identity resolved to the wrong value type")
        if resource.resource_digest != entry.model_digest:
            raise ValueError("classic ControlNet assembly provenance changed after apply")
        control_handle = resource.handle or materialized.get(entry.resident_id)
        if control_handle is None:
            if base_handle is None or cleanup is None:
                raise ValueError(
                    "control component requires a sampling model and invocation lifetime"
                )
            base = base_handle.runtime.assembled
            base_source = next(
                source.source
                for source in base_handle.recipe.sources
                if source.role in ("checkpoint", "diffusion")
            )
            module = execution_symbol(resource.descriptor.loader)(
                resource.plan,
                base.compute_dtype("diffusion"),
                base=base,
                base_asset_digest=base_source.digest,
            )
            control_handle = _enroll_control_module(
                module, torch, inference_torch, discard_on_release=True
            )
            cleanup.callback(control_handle.terminal_release)
            materialized[entry.resident_id] = control_handle
        control_handle.require_active()
        shape = entry.hint.shape
        if shape[0] not in (1, latent_batch):
            raise ValueError(
                "ControlNet hint batch must be one or match the latent batch; "
                f"got hint {shape[0]} and latent {latent_batch}"
            )
        expected_bytes = math.prod(shape) * 4
        if len(entry.hint.data) != expected_bytes:
            raise ValueError("classic ControlNet hint snapshot byte length is malformed")
        hint = torch.frombuffer(bytearray(entry.hint.data), dtype=torch.float32).reshape(*shape)
        if inference_torch.sd_control_hint_digest(hint) != entry.hint.digest:
            raise ValueError("classic ControlNet hint snapshot digest changed after apply")
        conditioning_factory = (
            inference_torch.SDControlConditioning
            if resource.descriptor is None
            else execution_symbol(resource.descriptor.runtime_class)
        )
        resolved = conditioning_factory(
            application,
            control_handle.module,
            hint,
            control_handle.resource_identity,
            entry.hint.digest,
            previous=resolved,
        )
        handle_identity = id(control_handle)
        if handle_identity not in seen_handles:
            seen_handles.add(handle_identity)
            handles.append(control_handle)
    return resolved, tuple(handles)


@contextmanager
def _classic_control_context(binding: _ClassicControlBinding | None, **kwargs: Any):
    with ExitStack() as cleanup:
        yield (
            (None, ())
            if binding is None
            else _materialize_classic_control(binding, cleanup=cleanup, **kwargs)
        )


def _uses_native_scheduling(value: object) -> bool:
    for entry in _condition_entries(value, "conditioning"):
        metadata = cast("Mapping[object, object]", entry[1])
        if (
            _NATIVE_HOOKS_KEY in metadata
            or _NATIVE_MASK_KEY in metadata
            or "start_percent" in metadata
            or "end_percent" in metadata
            or metadata.get("strength", 1.0) != 1.0
        ):
            return True
    return False


def _prompt_routes(inference: Any, family: Any, text: str) -> tuple[Any, ...]:
    return tuple(
        inference.ScheduledPromptRoute(inference.EncoderStream.from_encoder_id(encoder_id), text)
        for encoder_id in family.wiring.text_encoders
    )


def _curve_segments(hooks: _NativeHooks) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
    breakpoints = {0.0, 1.0}
    explicit_end = False
    for hook in hooks.loras:
        if hook.keyframes is not None:
            breakpoints.update(percent for percent, _ in hook.keyframes.points)
            explicit_end = explicit_end or any(
                percent == 1.0 for percent, _ in hook.keyframes.points
            )
    ordered = tuple(sorted(breakpoints))

    def multiplier(hook: _NativeLoraHook, percent: float) -> float:
        if hook.keyframes is None or not hook.keyframes.points:
            return 1.0
        result = hook.keyframes.points[0][1]
        for start, strength in hook.keyframes.points:
            if start > percent:
                break
            result = strength
        return result

    segments: list[tuple[float, float, tuple[float, ...]]] = []
    for index, start in enumerate(ordered):
        if index + 1 == len(ordered):
            if start == 1.0 and not explicit_end:
                continue
            end = 1.0
        else:
            boundary = ordered[index + 1]
            end = math.nextafter(boundary, 0.0) if boundary < 1.0 or explicit_end else 1.0
        segments.append((start, end, tuple(multiplier(hook, start) for hook in hooks.loras)))
    return tuple(segments)


class _StagedEncoder:
    def __init__(self, handle: NativeRuntimeHandle, encoder: object) -> None:
        self._handle = handle
        self._encoder = encoder

    def encode(self, *args: object, **kwargs: object) -> object:
        with self._handle.stage("text"):
            with _torch().no_grad():
                return cast("Any", self._encoder).encode(*args, **kwargs)


class _ScheduledTextRuntime:
    _ENCODERS = frozenset(
        {
            "_ovis_encoder",
            "_t5_encoder",
            "_clip_encoder",
            "_clip_l_encoder",
            "_clip_g_encoder",
        }
    )

    def __init__(self, handle: NativeRuntimeHandle) -> None:
        self._handle = handle

    def __getattr__(self, name: str) -> object:
        value = getattr(self._handle.runtime, name)
        if name in self._ENCODERS and value is not None:
            return _StagedEncoder(self._handle, value)
        return value

    def dispose(self) -> None:
        self._handle.terminal_release()


class _ScheduledDiffusionDeclaration:
    def dispose(self) -> None:
        return None
