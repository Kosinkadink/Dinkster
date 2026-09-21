"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

from .families.minimax_h3 import (
    NativeEmptyLTXAVLatent,
    NativeEmptyLTXVLatent,
)
from .native_arm_conditioning import (
    _averaged_conditioning,
    _combined_conditioning,
    _concatenated_conditioning,
    _conditioning_carrier,
    _conditioning_op_float,
    _rebound_conditioning_carrier,
    _scaled_conditioning,
    _zeroed_conditioning,
)
from .native_arm_core import (
    Any,
    EmptyLatentImage,
    Mapping,
    Node,
    NodeSchema,
    _torch,
    cast,
    importlib,
    math,
    replace,
)
from .native_arm_latent_utils import _check_bounds
from .native_arm_runtime import (
    _application_chain_model,
    _native_model,
    _native_model_sampling_cache,
    _native_model_sampling_space,
    _native_model_sampling_timeline,
    _NativeModelOverlay,
    _sampling_space_runtime,
)
from .nodes_provider import (
    _generation_provider_schema,
    _runtime_sampling_shift,
)

if TYPE_CHECKING:
    from collections.abc import Sequence  # noqa: F401

    from dinkster_inference import FluxFlowSigmas


class GenerationConditioningMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_merge")

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        mode = inputs.get("mode")
        if mode == "combine":
            carriers = [
                _conditioning_carrier(inputs[key], f"conditioning_{index}")
                for index in range(1, 9)
                if (key := f"mode.inputs.conditioning_{index}") in inputs
            ]
            if len(carriers) < 2:
                raise ValueError("combine requires at least two conditioning inputs")
            return cls.outputs(conditioning=_combined_conditioning(inference, carriers))
        if mode in ("average", "concat"):
            to_carrier = _conditioning_carrier(
                inputs.get("mode.conditioning_to"), "conditioning_to"
            )
            from_carrier = _conditioning_carrier(
                inputs.get("mode.conditioning_from"), "conditioning_from"
            )
            if mode == "concat":
                return cls.outputs(
                    conditioning=_concatenated_conditioning(inference, to_carrier, from_carrier)
                )
            strength = _conditioning_op_float(
                inputs.get("mode.conditioning_to_strength"), "conditioning_to_strength", 0.0, 1.0
            )
            return cls.outputs(
                conditioning=_averaged_conditioning(inference, to_carrier, from_carrier, strength)
            )
        raise ValueError(f"unknown conditioning merge mode: {mode!r}")


class GenerationConditioningScale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_scale")

    @classmethod
    def execute(cls, *, conditioning: object, multiplier: float) -> Mapping[str, object]:
        scale = _conditioning_op_float(multiplier, "multiplier", -100.0, 100.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        return cls.outputs(conditioning=_scaled_conditioning(inference, carrier, scale))


class GenerationConditioningSetArea(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_area")

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        carrier = _conditioning_carrier(inputs.get("conditioning"), "conditioning")
        strength = _conditioning_op_float(inputs.get("strength"), "strength", 0.0, 10.0)
        units = inputs.get("units")
        if units == "pixels":

            def latent_cells(name: str, low: int) -> int:
                value = inputs.get(f"units.{name}")
                if type(value) is not int:
                    raise TypeError(f"{name} must be an integer")
                if not low <= value <= 16384:
                    raise ValueError(f"{name} must be in [{low}, 16384], got {value}")
                return value // 8

            area = inference.AreaDescriptor(
                height=latent_cells("height", 64),
                width=latent_cells("width", 64),
                y=latent_cells("y", 0),
                x=latent_cells("x", 0),
                units=inference.AreaUnits.LATENT_CELLS,
                strength=strength,
            )
        elif units == "percent":

            def fraction(name: str) -> float:
                return _conditioning_op_float(inputs.get(f"units.{name}"), name, 0.0, 1.0)

            area = inference.AreaDescriptor(
                height=fraction("height"),
                width=fraction("width"),
                y=fraction("y"),
                x=fraction("x"),
                units=inference.AreaUnits.PERCENT,
                strength=strength,
            )
        elif units == "percent-video":

            def fraction(name: str) -> float:
                return _conditioning_op_float(inputs.get(f"units.{name}"), name, 0.0, 1.0)

            area = inference.AreaDescriptor(
                height=fraction("height"),
                width=fraction("width"),
                y=fraction("y"),
                x=fraction("x"),
                units=inference.AreaUnits.PERCENT,
                strength=strength,
                temporal=fraction("temporal"),
                z=fraction("z"),
            )
        else:
            raise ValueError(f"unknown area units: {units!r}")
        records: list[Any] = []
        for record in carrier.conditioning.records:
            mask = record.mask
            # The pin's ConditioningSetArea also clears set_area_to_bounds.
            if mask is not None:
                mask = replace(mask, set_area_to_bounds=False)
            records.append(replace(record, area=area, mask=mask))
        return cls.outputs(
            conditioning=inference.make_conditioning_carrier(
                inference.ConditioningSet(tuple(records)), carrier.bindings
            )
        )


class GenerationConditioningSetMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_mask")

    @classmethod
    def execute(
        cls, *, conditioning: object, mask: object, strength: float, set_cond_area: str
    ) -> Mapping[str, object]:
        if set_cond_area not in ("default", "mask bounds"):
            raise ValueError(f"unknown set_cond_area: {set_cond_area!r}")
        bounded = _conditioning_op_float(strength, "strength", 0.0, 10.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        torch = _torch()
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        mask_tensor = cast("Any", mask)
        if mask_tensor.ndim < 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        binding = inference_torch.tensor_to_payload_binding(
            "compat-conditioning-set-mask", mask_tensor, space=inference_torch.MASK_PAYLOAD_SPACE
        )
        descriptor = inference.MaskDescriptor(
            inference.PayloadReference(binding.reference_id),
            bounded,
            set_cond_area == "mask bounds",
        )
        records = tuple(replace(record, mask=descriptor) for record in carrier.conditioning.records)
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference, records, (*carrier.bindings, binding)
            )
        )


class GenerationConditioningSetTimestepRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_timestep_range")

    @classmethod
    def execute(cls, *, conditioning: object, start: float, end: float) -> Mapping[str, object]:
        start_percent = _conditioning_op_float(start, "start", 0.0, 1.0)
        end_percent = _conditioning_op_float(end, "end", 0.0, 1.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        # An inverted window can never be active, matching the pin's
        # start/end sigma bounds.
        schedule = (
            inference.PercentRange(start_percent, end_percent)
            if start_percent <= end_percent
            else inference.EMPTY_RANGE
        )
        records = tuple(
            replace(record, schedule=schedule) for record in carrier.conditioning.records
        )
        return cls.outputs(
            conditioning=inference.make_conditioning_carrier(
                inference.ConditioningSet(records), carrier.bindings
            )
        )


class GenerationConditioningZeroOut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_zero_out")

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        return cls.outputs(conditioning=_zeroed_conditioning(inference, carrier))


class GenerationChromaRadianceOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.chroma_radiance_options")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        preserve_wrapper: bool,
        start_sigma: float,
        end_sigma: float,
        nerf_tile_size: int,
        force_sequential_txt_ids: bool,
    ) -> Mapping[str, object]:
        if type(preserve_wrapper) is not bool:
            raise TypeError("preserve_wrapper must be a boolean")
        if type(force_sequential_txt_ids) is not bool:
            raise TypeError("force_sequential_txt_ids must be a boolean")
        _check_bounds(
            ("start_sigma", start_sigma, 0.0, 1.0),
            ("end_sigma", end_sigma, 0.0, 1.0),
        )
        if type(nerf_tile_size) is not int or nerf_tile_size < -1:
            raise ValueError(f"nerf_tile_size must be at least -1, got {nerf_tile_size}")
        if nerf_tile_size < 0 and not force_sequential_txt_ids:
            return cls.outputs(model=model)

        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            context_windows,
            existing_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        if handle.recipe.family_id != inference.CHROMA_RADIANCE.id:
            raise ValueError("Chroma Radiance Options requires a Chroma Radiance model")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        window = inference_torch.ChromaRadianceOptionWindow(
            inference_torch.ChromaRadianceOptions(
                None if nerf_tile_size < 0 else nerf_tile_size,
                force_sequential_txt_ids,
            ),
            float(start_sigma),
            float(end_sigma),
        )
        options = (*existing_options, window) if preserve_wrapper else (window,)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


class GenerationChromaModelSampling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.chroma_model_sampling")

    @classmethod
    def execute(cls, *, model: object, shift: float) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("Chroma Model Sampling does not accept a model application chain")
        (
            handle,
            overlays,
            resolvers,
            control,
            _,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model_value, "model")
        inference = importlib.import_module("dinkster_inference")
        if handle.recipe.family_id not in (inference.CHROMA.id, inference.CHROMA_RADIANCE.id):
            raise ValueError("Chroma Model Sampling requires a Chroma family model")
        if type(shift) is not float or not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("Chroma Model Sampling shift must be a positive finite float")
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                radiance_options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class GenerationModelSamplingSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_sd3")

    @classmethod
    def execute(cls, *, model: object, shift: object) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("Model Sampling SD3 does not accept a model application chain")
        (
            handle,
            overlays,
            resolvers,
            control,
            _,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model_value, "model")
        inference = importlib.import_module("dinkster_inference")
        family = next(
            item for item in inference.builtin_families() if item.id == handle.recipe.family_id
        )
        if not inference.is_flow_parameterization(family.sampling.parameterization):
            raise ValueError("Model Sampling SD3 requires a flow model")
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise ValueError("Model Sampling SD3 shift must be a positive finite float")
        shift = float(shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("Model Sampling SD3 shift must be a positive finite float")
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                radiance_options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class GenerationModelSamplingLTXV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_ltxv")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        latent: object | None = None,
    ) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("ModelSamplingLTXV does not accept a model application chain")
        handle, overlays, resolvers, control, _, transforms, windows, options = _native_model(
            model_value, "model"
        )
        inference = importlib.import_module("dinkster_inference")
        runtime = getattr(handle.runtime, "component_sampling_runtime", handle.runtime)
        if getattr(runtime.family, "sampling", None) != inference.LTX_SAMPLING:
            raise TypeError("ModelSamplingLTXV requires an LTX sampling runtime")

        tokens = inference.LTXV_SHIFT_TOKENS_HIGH
        if latent is not None:
            if not isinstance(latent, Mapping):
                raise TypeError("latent must be a mapping containing 'samples'")
            samples = cast("Mapping[object, object]", latent).get("samples")
            if type(samples) is inference.MultiStreamLatent:
                streams = cast("Any", samples)
                if "video" not in streams.roles:
                    raise TypeError("latent must contain a video stream")
                samples = streams.by_role("video")
            torch = _torch()
            if not isinstance(samples, torch.Tensor):
                raise TypeError("latent samples must be a torch.Tensor or MultiStreamLatent")
            dimensions = tuple(cast("Sequence[int]", cast("Any", samples).shape)[2:])
            if not dimensions or any(type(size) is not int or size < 1 for size in dimensions):
                raise ValueError("latent samples must have positive token dimensions")
            tokens = math.prod(dimensions)

        shift = inference.ltxv_dynamic_shift(
            tokens,
            max_shift=max_shift,
            base_shift=base_shift,
        )
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("ModelSamplingLTXV computed shift must be positive and finite")
        _runtime_sampling_shift(runtime, shift)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class _FluxModelOverlay(Protocol):
    @property
    def runtime(self) -> Any: ...

    @property
    def sampling_space(self) -> FluxFlowSigmas: ...


class _GenerationModelSamplingFluxOutput(TypedDict):
    model: _FluxModelOverlay


class GenerationModelSamplingFlux(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_flux")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        max_shift: float = 1.15,
        base_shift: float = 0.5,
        width: int = 1024,
        height: int = 1024,
    ) -> _GenerationModelSamplingFluxOutput:
        for name, value in (("max_shift", max_shift), ("base_shift", base_shift)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or not 16 <= value <= 16384:
                raise ValueError(f"{name} must be an integer in [16, 16384]")
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("ModelSamplingFlux does not accept a model application chain")
        handle, overlays, resolvers, control, _, transforms, windows, options = _native_model(
            model_value, "model"
        )
        inference = importlib.import_module("dinkster_inference")
        slope = (max_shift - base_shift) / (4096 - 256)
        intercept = base_shift - slope * 256
        shift = (width * height / (8 * 8 * 2 * 2)) * slope + intercept
        space = inference.FluxFlowSigmas(shift=shift)
        _sampling_space_runtime(handle.runtime, space)
        return cast(
            "_GenerationModelSamplingFluxOutput",
            cls.outputs(
                model=_NativeModelOverlay(
                    handle,
                    overlays,
                    resolvers,
                    control,
                    None,
                    transforms,
                    windows,
                    options,
                    sampling_cache=_native_model_sampling_cache(model_value),
                    sampling_timeline=_native_model_sampling_timeline(model_value),
                    sampling_space=space,
                )
            ),
        )


class GenerationEmptyLatentImage(EmptyLatentImage):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_latent_image")


class GenerationEmptySD3LatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_sd3_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}], got {value}")
        torch = _torch()
        samples = torch.zeros((batch_size, 16, height // 8, width // 8), device="cpu")
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 8})


class GenerationEmptyChromaRadianceLatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_chroma_radiance_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}], got {value}")
        torch = _torch()
        samples = torch.zeros((batch_size, 3, height, width), device="cpu")
        return cls.outputs(latent={"samples": samples})


class GenerationEmptyFlux2LatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384  # ComfyUI's MAX_RESOLUTION
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_flux2_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        torch = _torch()
        # intermediate_device() @ b78cec87 is cpu absent ComfyUI's --gpu-only
        # flag, which Dinkster does not wire.
        samples = torch.zeros((batch_size, 128, height // 16, width // 16), device="cpu")
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 16})


class GenerationEmptyLTXAVLatent(NativeEmptyLTXAVLatent):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_ltxav_latent")


class GenerationEmptyLTXVLatent(NativeEmptyLTXVLatent):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_ltxv_latent")
