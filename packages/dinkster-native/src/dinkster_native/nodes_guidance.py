"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .native_arm_conditioning import (
    _conditioning_carrier,
    _rebound_conditioning_carrier,
)
from .native_arm_core import (
    _FLUX2_REFERENCE_LATENTS_KEY,
    Any,
    Mapping,
    Node,
    NodeSchema,
    _torch,
    _with_flux_guidance,
    cast,
    importlib,
    replace,
)
from .native_arm_latent_utils import _check_bounds
from .native_arm_runtime import (
    _native_model,
    _native_model_sampling_cache,
    _native_model_sampling_space,
    _native_model_sampling_timeline,
    _NativeModelOverlay,
)
from .nodes_provider import (
    _bind_sampling_shift,
    _generation_provider_schema,
    _require_custom_sampling_runtime,
)

if TYPE_CHECKING:
    from dinkster_inference import ContextWindowsSpec


class GenerationFluxGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux_guidance")

    @classmethod
    def execute(cls, *, conditioning: object, guidance: float) -> Mapping[str, object]:
        if not 0.0 <= guidance <= 100.0:
            raise ValueError(f"guidance must be in [0.0, 100.0], got {guidance}")
        inference = importlib.import_module("dinkster_inference")
        if type(conditioning) is not inference.ConditioningCarrier:
            raise TypeError("conditioning must come from a Dinkster text-encoding node")
        return cls.outputs(conditioning=_with_flux_guidance(conditioning, float(guidance)))


class GenerationFluxDisableGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux_disable_guidance")

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        inference: Any = importlib.import_module("dinkster_inference")
        if type(conditioning) is not inference.ConditioningCarrier:
            raise TypeError("conditioning must come from a Dinkster text-encoding node")
        return cls.outputs(conditioning=_with_flux_guidance(conditioning, None))


class GenerationReferenceLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.reference_latent")

    @classmethod
    def execute(cls, *, conditioning: object, latent: object = None) -> Mapping[str, object]:
        carrier = _conditioning_carrier(conditioning, "conditioning")
        if latent is None:
            return cls.outputs(conditioning=carrier)
        if not isinstance(latent, Mapping):
            raise TypeError("latent must be a mapping containing 'samples'")
        torch = _torch()
        samples = cast("Mapping[str, object]", latent).get("samples")
        if type(samples) is not torch.Tensor:
            raise TypeError("latent['samples'] must be an exact torch.Tensor")
        inference: Any = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        binding = inference_torch.tensor_to_payload_binding(
            "flux2-reference-latent",
            samples,
            space="flux2-reference-latent",
        )
        reference = inference.PayloadReference(binding.reference_id)
        records: list[Any] = []
        for record in carrier.conditioning.records:
            metadata: dict[str, Any] = dict(record.extension_metadata)
            existing = metadata.get(_FLUX2_REFERENCE_LATENTS_KEY, ())
            if not isinstance(existing, tuple):
                raise TypeError("reference latent metadata must be an ordered payload tuple")
            references = cast("tuple[Any, ...]", existing)
            if any(type(item) is not inference.PayloadReference for item in references):
                raise TypeError("reference latent metadata must be an ordered payload tuple")
            metadata[_FLUX2_REFERENCE_LATENTS_KEY] = (*references, reference)
            records.append(replace(record, extension_metadata=tuple(metadata.items())))
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference,
                records,
                (*carrier.bindings, binding),
            )
        )


def _model_with_guidance_transform(
    model: object, node_type: str, contribution: object
) -> _NativeModelOverlay:
    handle, overlays, resolvers, control, shift, transforms, windows, chroma_options = (
        _native_model(model, "model")
    )
    owner_id = f"{node_type}:{len(transforms)}"
    return _NativeModelOverlay(
        handle,
        overlays,
        resolvers,
        control,
        shift,
        (*transforms, (owner_id, contribution)),
        windows,
        chroma_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


def _guidance_transform_factory(name: str, *args: object, **kwargs: object) -> object:
    module = importlib.import_module("dinkster_inference_torch.guidance_transforms")
    return getattr(module, name)(*args, **kwargs)


def _model_family_is_flow(model: object) -> bool:
    handle = _native_model(model, "model")[0]
    inference = importlib.import_module("dinkster_inference")
    family = next(
        item for item in inference.builtin_families() if item.id == handle.recipe.family_id
    )
    return bool(inference.is_flow_parameterization(family.sampling.parameterization))


class GenerationCfgZeroStar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_zero_star")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("cfg_zero_star")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_zero_star", contribution)
        )


class GenerationCfgNorm(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_norm")

    @classmethod
    def execute(cls, *, model: object, strength: float, pre_cfg: bool) -> Mapping[str, object]:
        _check_bounds(("strength", strength, 0.0, 100.0))
        contribution = _guidance_transform_factory("cfg_norm", float(strength), bool(pre_cfg))
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_norm", contribution)
        )


class GenerationTCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.tcfg")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("tcfg")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.tcfg", contribution)
        )


class GenerationFreSca(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.fresca")

    @classmethod
    def execute(
        cls, *, model: object, scale_low: float, scale_high: float, freq_cutoff: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("scale_low", scale_low, 0.0, 10.0),
            ("scale_high", scale_high, 0.0, 10.0),
            ("freq_cutoff", freq_cutoff, 1, 10_000),
        )
        contribution = _guidance_transform_factory(
            "fresca",
            scale_low=float(scale_low),
            scale_high=float(scale_high),
            freq_cutoff=int(freq_cutoff),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.fresca", contribution)
        )


class GenerationLazyCache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.lazy_cache")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_cache(model) is not None:
            raise ValueError("a model can carry only one sampling cache")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        cache = inference_torch.LazyCacheConfig(
            float(reuse_threshold),
            float(start_percent),
            float(end_percent),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=cache,
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


class GenerationEasyCache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.easy_cache")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_cache(model) is not None:
            raise ValueError("a model can carry only one sampling cache")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        cache = inference_torch.EasyCacheConfig(
            float(reuse_threshold),
            float(start_percent),
            float(end_percent),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=cache,
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


def _sampling_parameter_curve(value: object) -> object:
    inference = importlib.import_module("dinkster_inference")
    raw_points = getattr(value, "points", None)
    interpolation = getattr(value, "interpolation", None)
    if type(raw_points) is not tuple or interpolation not in ("linear", "monotone_cubic"):
        raise TypeError("sol_tau must be a dinkster.curve value")
    try:
        points: list[tuple[float, float]] = []
        for raw_point in cast("tuple[Any, ...]", raw_points):
            point = cast("tuple[Any, ...]", raw_point)
            if type(raw_point) is not tuple or len(point) != 2:
                raise TypeError
            points.append((float(point[0]), float(point[1])))
    except (TypeError, ValueError) as error:
        raise TypeError("sol_tau must be a dinkster.curve value") from error
    return inference.SamplingParameterCurve(tuple(points), interpolation)


class GenerationAttentionSchedule(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.attention_schedule")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        approximate_provider: str,
        start_percent: float,
        end_percent: float,
        conditioning_sink: str = "off",
        sol_tau: object = None,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_timeline(model) is not None:
            raise ValueError("a model can carry only one sampling timeline")
        raw_statuses = getattr(getattr(handle.runtime, "assembled", None), "attention_status", None)
        diffusion_statuses: list[object] = []
        if isinstance(raw_statuses, Mapping):
            for role, status in cast("Mapping[object, object]", raw_statuses).items():
                if role in ("unet", "flux"):
                    diffusion_statuses.append(status)
        if not diffusion_statuses or any(
            getattr(status, "primary", None) != approximate_provider
            for status in diffusion_statuses
        ):
            raise ValueError(
                "attention schedule provider must match the model's selected diffusion provider"
            )
        if any(
            getattr(status, "authenticated", False) is not True for status in diffusion_statuses
        ):
            raise ValueError("attention schedule requires an authenticated provider route")
        inference = importlib.import_module("dinkster_inference")
        sink_modifiers = {
            "off": None,
            "exact_kv": "sol_conditioning_exact_kv",
        }
        if conditioning_sink not in sink_modifiers:
            raise ValueError("attention schedule conditioning sink is unsupported")
        modifier_name = sink_modifiers[conditioning_sink]
        if modifier_name is not None and approximate_provider != "sol":
            raise ValueError("attention schedule conditioning sink requires the Sol provider")
        start = float(start_percent)
        end = float(end_percent)
        curve = None if sol_tau is None else _sampling_parameter_curve(sol_tau)
        schedule = inference.SamplingTimelineSchedule(
            approximate_provider,
            start,
            end,
            curve,
            (
                ()
                if modifier_name is None
                else (inference.AttentionModifierSchedule(modifier_name, start, end),)
            ),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=schedule,
                sampling_space=_native_model_sampling_space(model),
            )
        )


def _model_with_context_windows(model: object, spec: ContextWindowsSpec) -> _NativeModelOverlay:
    (
        handle,
        overlays,
        resolvers,
        control,
        shift,
        transforms,
        existing,
        chroma_options,
    ) = _native_model(model, "model")
    if existing is not None:
        raise ValueError("a model can carry only one context-windows configuration")
    return _NativeModelOverlay(
        handle,
        overlays,
        resolvers,
        control,
        shift,
        transforms,
        spec,
        chroma_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


def _context_windows_spec(
    *,
    length: int,
    overlap: int,
    schedule: str,
    stride: int,
    closed_loop: bool,
    fuse_method: str,
    dim: int,
    freenoise: bool,
    causal_anchor: bool,
    latent_retain_indices: tuple[int, ...],
) -> ContextWindowsSpec:
    inference = importlib.import_module("dinkster_inference")
    try:
        schedule_value = inference.ContextWindowSchedule(schedule)
    except ValueError:
        raise ValueError(f"unknown context schedule {schedule!r}") from None
    try:
        fuse_value = inference.ContextFuseMethod(fuse_method)
    except ValueError:
        raise ValueError(f"unknown fuse method {fuse_method!r}") from None
    if fuse_value is inference.ContextFuseMethod.RELATIVE:
        raise ValueError("relative fusing is not supported")
    return inference.ContextWindowsSpec(
        schedule_value,
        fuse_value,
        length,
        overlap,
        stride=stride,
        closed_loop=closed_loop,
        dim=dim,
        freenoise=freenoise,
        causal_anchor=causal_anchor,
        latent_retain_indices=latent_retain_indices,
    )


class GenerationContextWindowsManual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.context_windows_manual")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        dim: int,
        freenoise: bool,
        causal_window_fix: bool = True,
    ) -> Mapping[str, object]:
        spec = _context_windows_spec(
            length=context_length,
            overlap=context_overlap,
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=dim,
            freenoise=freenoise,
            causal_anchor=causal_window_fix,
            latent_retain_indices=(),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationWanContextWindowsManual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.wan_context_windows_manual")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        freenoise: bool,
    ) -> Mapping[str, object]:
        if type(context_length) is not int or type(context_overlap) is not int:
            raise TypeError("context_length and context_overlap must be ints")
        # WAN's causal VAE packs 4n+1 real frames into n+1 latent frames.
        spec = _context_windows_spec(
            length=max((context_length - 1) // 4 + 1, 1),
            overlap=max(context_overlap // 4, 0),
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=2,
            freenoise=freenoise,
            causal_anchor=True,
            latent_retain_indices=(),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationLTXVContextWindows(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_context_windows")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        freenoise: bool,
        retain_first_frame: bool,
    ) -> Mapping[str, object]:
        if type(context_length) is not int or type(context_overlap) is not int:
            raise TypeError("context_length and context_overlap must be ints")
        spec = _context_windows_spec(
            length=max((context_length - 1) // 8 + 1, 1),
            overlap=max(context_overlap // 8, 0),
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=2,
            freenoise=freenoise,
            causal_anchor=True,
            latent_retain_indices=(0,) if retain_first_frame else (),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationAdaptiveProjectedGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.adaptive_projected_guidance")

    @classmethod
    def execute(
        cls, *, model: object, eta: float, norm_threshold: float, momentum: float
    ) -> Mapping[str, object]:
        _check_bounds(
            ("eta", eta, -10.0, 10.0),
            ("norm_threshold", norm_threshold, 0.0, 50.0),
            ("momentum", momentum, -5.0, 1.0),
        )
        contribution = _guidance_transform_factory(
            "apg", float(eta), float(norm_threshold), float(momentum)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.adaptive_projected_guidance", contribution
            )
        )


class GenerationMahiroGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.mahiro_guidance")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("mahiro")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.mahiro_guidance", contribution)
        )


class GenerationEpsilonScaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.epsilon_scaling")

    @classmethod
    def execute(cls, *, model: object, scaling_factor: float) -> Mapping[str, object]:
        _check_bounds(("scaling_factor", scaling_factor, 0.5, 1.5))
        contribution = _guidance_transform_factory("epsilon_scaling", float(scaling_factor))
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.epsilon_scaling", contribution)
        )


class GenerationCFGOverride(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_override")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        cfg: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("cfg", cfg, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if start_percent > end_percent:
            raise ValueError("start_percent must be less than or equal to end_percent")
        runtime, sampling_shift, _device = _require_custom_sampling_runtime(model, "CFGOverride")
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma, sampling_shift
        )
        sigma_high = percent_to_sigma(start_percent, return_actual_sigma=True)
        sigma_low = percent_to_sigma(end_percent, return_actual_sigma=True)
        contribution = _guidance_transform_factory(
            "cfg_override",
            float(cfg),
            float(sigma_low),
            float(sigma_high),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_override", contribution)
        )


class GenerationRescaleCfg(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.rescale_cfg")

    @classmethod
    def execute(cls, *, model: object, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, 0.0, 1.0))
        contribution = _guidance_transform_factory(
            "rescale_cfg", float(multiplier), flow=_model_family_is_flow(model)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.rescale_cfg", contribution)
        )


class GenerationRenormCfg(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.renorm_cfg")

    @classmethod
    def execute(cls, *, model: object, cfg_trunc: float, renorm_cfg: float) -> Mapping[str, object]:
        _check_bounds(
            ("cfg_trunc", cfg_trunc, 0.0, 100.0),
            ("renorm_cfg", renorm_cfg, 0.0, 100.0),
        )
        contribution = _guidance_transform_factory(
            "renorm_cfg", float(cfg_trunc), float(renorm_cfg)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.renorm_cfg", contribution)
        )


class GenerationTemporalScoreRescaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.temporal_score_rescaling")

    @classmethod
    def execute(cls, *, model: object, tsr_k: float, tsr_sigma: float) -> Mapping[str, object]:
        _check_bounds(
            ("tsr_k", tsr_k, 0.01, 100.0),
            ("tsr_sigma", tsr_sigma, 0.01, 100.0),
        )
        contribution = _guidance_transform_factory(
            "temporal_score_rescaling",
            float(tsr_k),
            float(tsr_sigma),
            flow=_model_family_is_flow(model),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.temporal_score_rescaling", contribution
            )
        )


class GenerationNormalizedAttentionGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.nag")

    @classmethod
    def execute(
        cls, *, model: object, nag_scale: float, nag_alpha: float, nag_tau: float
    ) -> Mapping[str, object]:
        _check_bounds(
            ("nag_scale", nag_scale, 0.0, 50.0),
            ("nag_alpha", nag_alpha, 0.0, 1.0),
            ("nag_tau", nag_tau, 1.0, 10.0),
        )
        contribution = _guidance_transform_factory(
            "nag", float(nag_scale), float(nag_alpha), float(nag_tau)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.nag", contribution)
        )
