"""Execution nodes backed directly by the Dinkster-owned ComfyUI fork."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from dinkster_assets import AssetRef
from dinkster_nodes_generation.nodes import (
    ApplyMiniMaxH3FunControlNet,
    BlockSparseAttention,
    CLIPTextEncode,
    EmptyLatentImage,
    EmptyMiniMaxH3AV,
    ExplicitWindowPlan,
    KSampler,
    LoadCheckpoint,
    LoadClip,
    LoadDiffusionModel,
    LoadModelPatch,
    LoadVAE,
    MiniMaxH3CacheDIT,
    MiniMaxH3ImageToVideo,
    MiniMaxH3T2VAConditioning,
    RES4LYFRKBetaSampler,
    SeparateAVLatent,
    SpatialTilePlan,
    TemporalWindowPlan,
    VAEDecode,
    VAEDecodeAudio,
)
from dinkster_schema import Node
from dinkster_workers import current_execution_context


@dataclass(frozen=True, slots=True)
class _ForkConditioning:
    conditioning: object
    owner: object
    fingerprint: str

    @property
    def _dinkster_resident_owner(self) -> object:
        return self.owner

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.fingerprint

    @property
    def _dinkster_input_value(self) -> object:
        return self.conditioning


def _resident_conditioning(conditioning: object, owner: object) -> object:
    import hashlib

    from dinkster_inference_wire import ResidentConditioningCarrier

    fingerprint = "fork-conditioning:" + hashlib.sha256(str(id(owner)).encode()).hexdigest()
    return ResidentConditioningCarrier(_ForkConditioning(conditioning, owner, fingerprint))


def model_for_attention_route(model: object) -> object:
    """Clone a model patcher and attach this invocation's attention route."""
    context = current_execution_context()
    if context is None or context.attention_route_token is None:
        return model
    runtime = cast("Any", context.attention_runtime)
    sampling_model = cast("Any", model).clone()
    transformer_options = sampling_model.model_options["transformer_options"]
    sparse_config = transformer_options.get("dinkster_h3_sparse_attention")
    sampling_model.set_model_optimized_attention(runtime.for_model(context.attention_route_token))
    if sparse_config is not None:
        runtime.configure_sparse_model(sampling_model, sparse_config)
    if runtime.distributed_active():
        from .multigpu import configure_distributed_model

        configure_distributed_model(sampling_model)
    return sampling_model


def _unwrap_conditioning(value: object) -> object:
    payload = getattr(value, "_dinkster_resident_payload", None)
    return payload.conditioning if type(payload) is _ForkConditioning else value


def _fork_samples(samples: object) -> tuple[object, tuple[str, ...] | None]:
    from dinkster_inference_wire import MultiStreamLatent

    if type(samples) is not MultiStreamLatent:
        return samples, None
    streams = cast("MultiStreamLatent[Any]", samples)
    nested = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor(
        tuple(streams.by_role(role) for role in streams.roles)
    )
    return nested, streams.roles


def _dinkster_samples(samples: object, roles: tuple[str, ...] | None) -> object:
    if roles is None:
        return samples
    from dinkster_inference_wire import MultiStreamLatent

    return MultiStreamLatent[Any].from_pairs(
        tuple(zip(roles, cast("Any", samples).unbind(), strict=True))
    )


def _window_layers(plan: object) -> list[dict[str, object]]:
    if plan is None:
        return []
    if not isinstance(plan, Mapping):
        raise TypeError("window plan must be a mapping")
    mapping = cast("Mapping[str, object]", plan)
    if set(mapping) != {"layers"}:
        raise TypeError("window plan must contain only a layers list")
    layers = mapping["layers"]
    if not isinstance(layers, list):
        raise TypeError("window plan layers must be objects")
    values = cast("list[object]", layers)
    if any(not isinstance(layer, dict) for layer in values):
        raise TypeError("window plan layers must be objects")
    return [dict(cast("dict[str, object]", layer)) for layer in values]


def _append_window_layers(plan: object, *layers: dict[str, object]) -> dict[str, object]:
    return {"layers": [*_window_layers(plan), *layers]}


class GenerationTemporalWindowPlan(TemporalWindowPlan):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        plan: object = None,
    ) -> Mapping[str, object]:
        layer: dict[str, object] = {
            "mode": "stock-temporal",
            "context_length": context_length,
            "context_overlap": context_overlap,
            "context_schedule": context_schedule,
            "context_stride": context_stride,
            "closed_loop": closed_loop,
            "fuse_method": fuse_method,
        }
        return cls.outputs(plan=_append_window_layers(plan, layer))


class GenerationSpatialTilePlan(SpatialTilePlan):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        tile_height: int,
        tile_width: int,
        overlap_height: int,
        overlap_width: int,
        fuse_method: str,
        plan: object = None,
    ) -> Mapping[str, object]:
        height: dict[str, object] = {
            "mode": "regular",
            "axis": "height",
            "length": tile_height,
            "overlap": overlap_height,
            "fuse_method": fuse_method,
        }
        width: dict[str, object] = {
            "mode": "regular",
            "axis": "width",
            "length": tile_width,
            "overlap": overlap_width,
            "fuse_method": fuse_method,
        }
        return cls.outputs(plan=_append_window_layers(plan, height, width))


class GenerationExplicitWindowPlan(ExplicitWindowPlan):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        axis: str,
        windows: str,
        wrap: bool,
        fuse_method: str,
        plan: object = None,
    ) -> Mapping[str, object]:
        try:
            index_lists = [
                [int(value.strip()) for value in window.split(",")] for window in windows.split(";")
            ]
        except ValueError:
            raise ValueError("window indices must be comma-separated integers") from None
        if not index_lists or any(not indices for indices in index_lists):
            raise ValueError("every explicit window must contain at least one index")
        layer: dict[str, object] = {
            "mode": "explicit",
            "axis": axis,
            "windows": index_lists,
            "wrap": wrap,
            "fuse_method": fuse_method,
        }
        return cls.outputs(plan=_append_window_layers(plan, layer))


class GenerationRES4LYFRKBetaSampler(RES4LYFRKBetaSampler):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, rk_type: str, eta: float, eta_substep: float
    ) -> Mapping[str, object]:
        from dinkster_inference_wire import BuiltinSamplerSelection

        return cls.outputs(
            sampler=BuiltinSamplerSelection(
                "res4lyf.rk_beta",
                (("rk_type", rk_type), ("eta", eta), ("eta_substep", eta_substep)),
            )
        )


def _weight_profile(window_plan: Any, fuse_method: object, overlap: int) -> object:
    kinds: dict[object, Any] = {
        "flat": window_plan.WindowWeightKind.FLAT,
        "pyramid": window_plan.WindowWeightKind.PYRAMID,
        "overlap-linear": window_plan.WindowWeightKind.OVERLAP_LINEAR,
    }
    try:
        kind = kinds[fuse_method]
    except KeyError:
        raise ValueError(f"unknown window fuse method {fuse_method!r}") from None
    return window_plan.WindowWeightProfile(
        kind,
        overlap if kind is window_plan.WindowWeightKind.OVERLAP_LINEAR else 0,
    )


def _regular_indices(extent: int, length: int, overlap: int) -> tuple[tuple[int, ...], ...]:
    if length < 1 or overlap < 0 or overlap >= length:
        raise ValueError("window length must be positive and overlap must be smaller than length")
    if length >= extent:
        return (tuple(range(extent)),)
    final_start = extent - length
    starts = list(range(0, final_start + 1, length - overlap))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(tuple(range(start, start + length)) for start in starts)


def _latent_media_layouts(latent: object, roles: tuple[str, ...] | None):
    window_plan = cast("Any", importlib.import_module("dinkster_inference.window_plan"))
    window_execution = cast("Any", importlib.import_module("dinkster_inference.window_execution"))
    tensors: tuple[Any, ...] = (
        (cast("Any", latent),) if roles is None else tuple(cast("Any", latent).unbind())
    )
    names = ("latent",) if roles is None else roles
    primary = tensors[0]
    if primary.ndim == 5:
        primary_dimensions = {"temporal": 2, "height": 3, "width": 4}
    elif primary.ndim == 4:
        primary_dimensions = {"height": 2, "width": 3}
    else:
        raise ValueError("window plans require a four- or five-dimensional primary latent")
    extents = {
        axis: int(primary.shape[dimension]) for axis, dimension in primary_dimensions.items()
    }
    return window_plan, window_execution, tensors, names, primary_dimensions, extents


def _compile_window_executor(
    plan: object,
    latent: object,
    roles: tuple[str, ...] | None,
    model_options: dict[str, object],
):
    window_plan, window_execution, tensors, names, primary_dimensions, extents = (
        _latent_media_layouts(latent, roles)
    )
    layers: list[Any] = []
    claimed_axes: set[str] = set()
    wrappable_axes: set[str] = set()
    context_windows = cast("Any", importlib.import_module("dinkster_inference.context_windows"))
    for declaration in _window_layers(plan):
        mode = declaration.get("mode")
        axis = "temporal" if mode == "stock-temporal" else declaration.get("axis")
        if type(axis) is not str or axis not in extents:
            raise ValueError(f"window axis {axis!r} is absent from the primary latent")
        if axis in claimed_axes:
            raise ValueError(f"more than one graph layer claims media axis {axis!r}")
        claimed_axes.add(axis)
        if mode == "stock-temporal":
            if declaration["context_schedule"] != "standard_static":
                raise ValueError("layered temporal plans currently require standard_static")
            temporal = context_windows.TemporalWindowPlan(
                context_windows.get_matching_context_schedule(declaration["context_schedule"]),
                context_windows.get_matching_fuse_method(declaration["fuse_method"]),
                context_length=declaration["context_length"],
                context_overlap=declaration["context_overlap"],
                context_stride=declaration["context_stride"],
                closed_loop=declaration["closed_loop"],
            )
            layer = temporal.layer(extents[axis], model_options)
        else:
            if mode == "regular":
                index_lists = _regular_indices(
                    extents[axis],
                    cast("int", declaration["length"]),
                    cast("int", declaration["overlap"]),
                )
                modular = False
                overlap = cast("int", declaration["overlap"])
            elif mode == "explicit":
                index_lists = tuple(
                    tuple(indices) for indices in cast("list[list[int]]", declaration["windows"])
                )
                modular = declaration.get("wrap") is True
                overlap = 0
                if modular:
                    wrappable_axes.add(axis)
            else:
                raise ValueError(f"unknown window layer mode {mode!r}")
            layer = window_plan.WindowPlanLayer(
                (axis,),
                tuple(
                    window_plan.LayerWindow((window_plan.WindowIndexList(indices, modular),))
                    for indices in index_lists
                ),
                (_weight_profile(window_plan, declaration.get("fuse_method"), overlap),),
                window_plan.MergeDeclaration(),
            )
        layers.append(layer)

    axes = tuple(
        window_plan.MediaAxis(axis, extents[axis], wrappable=axis in wrappable_axes)
        for axis in sorted(claimed_axes)
    )
    kinds: list[Any] = []
    layouts: list[Any] = []
    for name, tensor in zip(names, tensors, strict=True):
        if name == names[0]:
            dimensions = {axis: primary_dimensions[axis] for axis in claimed_axes}
            mappings = tuple(
                window_plan.KindAxisMap(
                    axis,
                    int(tensor.shape[dimensions[axis]]),
                    window_plan.IntegerAffineIndexMap(1),
                )
                for axis in sorted(dimensions)
            )
            invariant = ()
        elif "temporal" in claimed_axes and tensor.ndim == 4:
            dimensions = {"temporal": 3}
            mappings = (
                window_plan.KindAxisMap(
                    "temporal",
                    int(tensor.shape[3]),
                    window_plan.ProportionalRangeIndexMap(),
                ),
            )
            invariant = tuple(sorted(claimed_axes - {"temporal"}))
        else:
            dimensions = {}
            mappings = ()
            invariant = tuple(sorted(claimed_axes))
        kinds.append(window_plan.WindowKind(name, mappings, invariant))
        layouts.append(window_execution.WindowTensorLayout(name, tuple(sorted(dimensions.items()))))
    compiled = window_plan.compile_window_plan(
        axes=axes,
        kinds=tuple(kinds),
        layers=tuple(layers),
    )
    latent_layout = layouts[0] if len(layouts) == 1 else tuple(layouts)
    return window_execution.WindowPlanExecutor(compiled, latent_layout)


class GenerationLoadCheckpoint(LoadCheckpoint):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, checkpoint: object
    ) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError("checkpoint must be an AssetRef")
        model, clip, vae, _ = importlib.import_module(
            "dinkster_inference.sd"
        ).load_checkpoint_guess_config(
            str(checkpoint.local_path()),
            output_vae=True,
            output_clip=True,
            embedding_directory=[],
        )
        return cls.outputs(model=model, clip=clip, vae=vae)


class GenerationLoadModelPatch(LoadModelPatch):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, model_patch: object
    ) -> Mapping[str, object]:
        if not isinstance(model_patch, AssetRef):
            raise TypeError("model_patch must be an AssetRef")
        loaded = importlib.import_module(
            "dinkster_inference.minimax_control"
        ).load_minimax_h3_fun_control_patch(str(model_patch.local_path()))
        return cls.outputs(model_patch=loaded)


class GenerationApplyMiniMaxH3FunControlNet(ApplyMiniMaxH3FunControlNet):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        model: object,
        model_patch: object,
        vae: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        control_video: object = None,
        mask: object = None,
        source_video: object = None,
    ) -> Mapping[str, object]:
        control = None if control_video is None else cast("Any", control_video).movedim(-1, 1)
        source = None if source_video is None else cast("Any", source_video).movedim(-1, 1)
        patched = importlib.import_module(
            "dinkster_inference.minimax_control"
        ).apply_minimax_h3_fun_control(
            model,
            model_patch,
            vae,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            control_video=control,
            mask=mask,
            source_video=source,
        )
        return cls.outputs(model=patched)


class GenerationLoadDiffusionModel(LoadDiffusionModel):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        diffusion_model: object,
        weight_dtype: str,
        gguf_residency: str = "memory",
    ) -> Mapping[str, object]:
        if not isinstance(diffusion_model, AssetRef):
            raise TypeError("diffusion_model must be an AssetRef")
        torch = cast("Any", importlib.import_module("torch"))
        options: dict[str, object] = {"assign_loaded_weights": True}
        if weight_dtype in ("fp8_e4m3fn", "fp8_e4m3fn_fast"):
            options["dtype"] = torch.float8_e4m3fn
            if weight_dtype.endswith("_fast"):
                options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            options["dtype"] = torch.float8_e5m2
        path = diffusion_model.local_path()
        if path.suffix.lower() == ".gguf":
            options["gguf_residency"] = gguf_residency
        model = importlib.import_module("dinkster_inference.sd").load_diffusion_model(
            str(path), model_options=options
        )
        return cls.outputs(model=model)


class NativeLoadClip(LoadClip):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        text_encoder: object,
        type: str,
        text_encoder_2: object = None,
        device: str = "default",
        gguf_residency: str = "memory",
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        if text_encoder_2 is not None and not isinstance(text_encoder_2, AssetRef):
            raise TypeError("text_encoder_2 must be an AssetRef")
        sd = importlib.import_module("dinkster_inference.sd")
        try:
            clip_type = getattr(sd.CLIPType, type.upper())
        except AttributeError:
            raise ValueError(f"unsupported text encoder type: {type}") from None
        options: dict[str, object] = {}
        if device == "cpu":
            torch = cast("Any", importlib.import_module("torch"))
            options["load_device"] = options["offload_device"] = torch.device("cpu")
        paths = [text_encoder.local_path()]
        if text_encoder_2 is not None:
            paths.append(text_encoder_2.local_path())
        if any(path.suffix.lower() == ".gguf" for path in paths):
            options["gguf_residency"] = gguf_residency
        clip = sd.load_clip(
            ckpt_paths=[str(path) for path in paths],
            embedding_directory=[],
            clip_type=clip_type,
            model_options=options,
        )
        return cls.outputs(clip=clip)


class NativeLoadVae(LoadVAE):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, vae: object, pixel_space: bool = False
    ) -> Mapping[str, object]:
        if pixel_space:
            raise ValueError("pixel-space codecs are not supported")
        if not isinstance(vae, AssetRef):
            raise TypeError("vae must be an AssetRef")
        state, metadata = importlib.import_module("dinkster_inference.utils").load_torch_file(
            str(vae.local_path()), return_metadata=True
        )
        loaded = importlib.import_module("dinkster_inference.sd").VAE(sd=state, metadata=metadata)
        loaded.throw_exception_if_invalid()
        return cls.outputs(vae=loaded)


class GenerationClipTextEncode(CLIPTextEncode):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, text: str, clip: object
    ) -> Mapping[str, object]:
        direct = cast("Any", clip)
        conditioning = direct.encode_from_tokens_scheduled(direct.tokenize(text))
        return cls.outputs(conditioning=_resident_conditioning(conditioning, clip))


class GenerationEmptyLatentImage(EmptyLatentImage):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, width: int, height: int, batch_size: int
    ) -> Mapping[str, object]:
        if width % 8 or height % 8:
            raise ValueError("width and height must be divisible by 8")
        torch = cast("Any", importlib.import_module("torch"))
        return cls.outputs(
            latent={"samples": torch.zeros((batch_size, 4, height // 8, width // 8))}
        )


class GenerationKSampler(KSampler):
    MAX_SEED = 0xFFFFFFFFFFFFFFFF
    MAX_STEPS = 10_000
    MAX_CFG = 100.0

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
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
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
        segment: object = None,
        window_plan: object = None,
        sampler: object = None,
    ) -> Mapping[str, object]:
        if conditioning_batching != "auto" or max_fused_lanes != 2:
            raise ValueError("fork sampling only supports default conditioning batching")
        if segment is not None:
            raise ValueError("segmented sampling is not supported")
        if not isinstance(latent_image, Mapping):
            raise TypeError("latent_image must be a mapping")
        torch = cast("Any", importlib.import_module("torch"))
        sample = cast("Any", importlib.import_module("dinkster_inference.sample"))
        model_management = cast(
            "Any", importlib.import_module("dinkster_inference.model_management")
        )
        sampling_model = model_for_attention_route(model)
        source = dict(cast("Mapping[object, object]", latent_image))
        latent, roles = _fork_samples(source["samples"])
        latent = sample.fix_empty_latent_channels(
            sampling_model,
            latent,
            source.get("downscale_ratio_spacial"),
            source.get("downscale_ratio_temporal"),
        )
        if window_plan is not None:
            if sampling_model is model:
                sampling_model = cast("Any", sampling_model).clone()
            model_options = dict(cast("Any", sampling_model).model_options)
            model_options["window_plan"] = _compile_window_executor(
                window_plan,
                latent,
                roles,
                model_options,
            )
            cast("Any", sampling_model).model_options = model_options
        noise = sample.prepare_noise(latent, seed, source.get("batch_index"))
        if sampler is None:
            with torch.inference_mode():
                output = sample.sample(
                    sampling_model,
                    noise,
                    steps,
                    cfg,
                    sampler_name.removeprefix("dinkster."),
                    scheduler.removeprefix("dinkster."),
                    _unwrap_conditioning(positive),
                    _unwrap_conditioning(negative),
                    latent,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False,
                    noise_mask=source.get("noise_mask"),
                    callback=None,
                    disable_pbar=True,
                    seed=seed,
                )
        else:
            from dinkster_inference_wire import BuiltinSamplerSelection

            if type(sampler) is not BuiltinSamplerSelection:
                raise TypeError("sampler must be a built-in sampler selection")
            samplers = cast("Any", importlib.import_module("dinkster_inference.samplers"))
            configured = samplers.KSampler(
                sampling_model,
                steps=steps,
                device=cast("Any", sampling_model).load_device,
                sampler=sampler.sampler_id,
                scheduler=scheduler.removeprefix("dinkster."),
                denoise=denoise,
                model_options=cast("Any", sampling_model).model_options,
            )
            with torch.inference_mode():
                output = samplers.sample(
                    sampling_model,
                    noise,
                    _unwrap_conditioning(positive),
                    _unwrap_conditioning(negative),
                    cfg,
                    cast("Any", sampling_model).load_device,
                    samplers.sampler_object(sampler.sampler_id, dict(sampler.options)),
                    configured.sigmas,
                    cast("Any", sampling_model).model_options,
                    latent_image=latent,
                    denoise_mask=source.get("noise_mask"),
                    callback=None,
                    disable_pbar=True,
                    seed=seed,
                )
            output = output.to(
                device=model_management.intermediate_device(),
                dtype=model_management.intermediate_dtype(),
            )
        source["samples"] = _dinkster_samples(output, roles)
        return cls.outputs(latent=source)


class GenerationVAEDecode(VAEDecode):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, samples: object, vae: object
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples)["samples"]
        from dinkster_inference_wire import MultiStreamLatent

        if type(latent) is MultiStreamLatent:
            latent = cast("MultiStreamLatent[Any]", latent).by_role("video")
        torch = cast("Any", importlib.import_module("torch"))
        with torch.inference_mode():
            image = cast("Any", vae).decode(latent)
            if len(image.shape) == 5:
                image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        return cls.outputs(image=image)


class NativeBlockSparseAttention(BlockSparseAttention):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        model: object,
        selection: str,
        start_percent: float = 0.2,
        end_percent: float = 1.0,
        dense_blocks: str = "",
        min_tokens: int = 12288,
        extra_tokens: int = 256,
        sink_conditioning: str = "exact_kv_and_rows",
        verbose: bool = False,
        **inputs: object,
    ) -> Mapping[str, object]:
        del verbose
        if selection not in ("sol-attn", "sla"):
            raise ValueError("fork sparse attention supports sol-attn and sla selection")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError("sparse attention percentages must be ordered within zero and one")
        if min_tokens < 1:
            raise ValueError("sparse attention min_tokens must be positive")
        if extra_tokens not in (0, 64, 128, 192, 256):
            raise ValueError("sparse attention extra_tokens must be a 64-token increment")
        if sink_conditioning not in ("off", "exact_kv", "exact_kv_and_rows"):
            raise ValueError("sparse attention sink_conditioning is invalid")
        try:
            dense = tuple(int(value.strip()) for value in dense_blocks.split(",") if value.strip())
        except ValueError as exc:
            raise ValueError("sparse attention dense_blocks must contain integers") from exc
        selection_value = inputs.get(
            "selection.tau" if selection == "sol-attn" else "selection.keep_percent",
            1.3 if selection == "sol-attn" else 10.0,
        )
        if not isinstance(selection_value, (int, float)) or isinstance(selection_value, bool):
            raise TypeError("sparse attention selection value must be numeric")
        model_sampling = cast("Any", model).get_model_object("model_sampling")
        patched = cast("Any", model).clone()
        patched.model_options["transformer_options"]["dinkster_h3_sparse_attention"] = {
            "sigma_start": float(model_sampling.percent_to_sigma(start_percent)),
            "sigma_end": float(model_sampling.percent_to_sigma(end_percent)),
            "dense_blocks": dense,
            "min_tokens": int(min_tokens),
            "extra_tokens": int(extra_tokens),
            "sink_conditioning": sink_conditioning,
            "tau": float(selection_value) if selection == "sol-attn" else 0.0,
            "keep_percent": float(selection_value) if selection == "sla" else 0.0,
        }
        return cls.outputs(MODEL=patched)


_H3_CACHE_DIT_POLICIES = {
    "quality": {
        "Fn_compute_blocks": 1,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
    },
    "speed": {
        "Fn_compute_blocks": 1,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
    },
}


def _h3_cache_dit_sampler(executor: object, *args: object, **kwargs: object) -> object:
    extra_args = cast("Mapping[str, Any]", args[2])
    model_options = cast("Mapping[str, Any]", extra_args["model_options"])
    transformer_options = cast("dict[str, Any]", model_options["transformer_options"])
    config = cast("dict[str, Any]", transformer_options["dinkster_h3_cache_dit"])
    runtime = {
        "key": None,
        "key_fields": None,
        "state": None,
        "hits": 0,
        "misses": 0,
        "invalidations": 0,
        "events": [],
    }
    config["runtime"] = runtime
    try:
        return cast("Any", executor)(*args, **kwargs)
    finally:
        sink = config.get("receipt_sink")
        if isinstance(sink, list):
            sink.append(
                {
                    "key": runtime["key_fields"],
                    "hits": runtime["hits"],
                    "misses": runtime["misses"],
                    "invalidations": runtime["invalidations"],
                    "events": list(runtime["events"]),
                }
            )
        runtime["state"] = None
        config.pop("runtime", None)


class NativeMiniMaxH3CacheDIT(MiniMaxH3CacheDIT):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, model: object, policy: str = "quality"
    ) -> Mapping[str, object]:
        try:
            selected = _H3_CACHE_DIT_POLICIES[policy]
        except KeyError as exc:
            raise ValueError("MiniMax H3 Cache-DiT policy must be quality or speed") from exc
        patched = cast("Any", model).clone()
        config = {
            "policy": policy,
            "model_identity": f"{id(patched.model)}:{patched.patches_uuid}",
            **selected,
        }
        patched.model_options["transformer_options"]["dinkster_h3_cache_dit"] = config
        patched.add_wrapper_with_key(
            "sampler_sample", "dinkster_h3_cache_dit", _h3_cache_dit_sampler
        )
        return cls.outputs(MODEL=patched)


def _h3_shape(width: int, height: int, frame_count: int) -> object:
    from dinkster_inference_wire import MultiStreamLatent

    torch = cast("Any", importlib.import_module("torch"))
    while frame_count % 17 != 5:
        frame_count += 1
    video_frames = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_frames = round(frame_count / 24 * 40)
    device = importlib.import_module("dinkster_inference.model_management").intermediate_device()
    return MultiStreamLatent[Any].from_pairs(
        (
            (
                "video",
                torch.zeros((1, 24, video_frames, height // 16, width // 16), device=device),
            ),
            ("audio", torch.zeros((1, 32, 2, audio_frames), device=device)),
        )
    )


class NativeEmptyMiniMaxH3AV(EmptyMiniMaxH3AV):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, width: int, height: int, frame_count: int
    ) -> Mapping[str, object]:
        return cls.outputs(latent={"samples": _h3_shape(width, height, frame_count)})


class NativeMiniMaxH3T2VAConditioning(MiniMaxH3T2VAConditioning):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, clip: object, target: object, prompt: str
    ) -> Mapping[str, object]:
        del target
        direct = cast("Any", clip)
        conditioning = direct.encode_from_tokens_scheduled(direct.tokenize(prompt, images=[]))
        return cls.outputs(conditioning=_resident_conditioning(conditioning, clip))


class NativeMiniMaxH3ImageToVideo(MiniMaxH3ImageToVideo):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        clip: object,
        vae: object,
        prompt: str,
        width: int,
        height: int,
        length: int,
        first_frame: object = None,
        last_frame: object = None,
    ) -> Mapping[str, object]:
        del vae
        if first_frame is not None or last_frame is not None:
            raise ValueError("keyframed H3 conditioning is not supported")
        latent = {"samples": _h3_shape(width, height, length)}
        conditioned = NativeMiniMaxH3T2VAConditioning.execute(
            clip=clip, target=latent, prompt=prompt
        )
        return {"positive": conditioned["conditioning"], "latent": latent}


class NativeSeparateAVLatent(SeparateAVLatent):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, latent: object
    ) -> Mapping[str, object]:
        if not isinstance(latent, Mapping):
            raise TypeError("latent must be a mapping")
        source = dict(cast("Mapping[object, object]", latent))
        streams = cast("Any", source["samples"])
        video = dict(source)
        audio = dict(source)
        video["samples"] = streams.by_role("video")
        audio["samples"] = streams.by_role("audio")
        return cls.outputs(video_latent=video, audio_latent=audio)


class NativeVAEDecodeAudio(VAEDecodeAudio):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, samples: object, vae: object
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples)["samples"]
        from dinkster_inference_wire import MultiStreamLatent

        if type(latent) is MultiStreamLatent:
            latent = cast("MultiStreamLatent[Any]", latent).by_role("audio")
        torch = cast("Any", importlib.import_module("torch"))
        with torch.inference_mode():
            audio = cast("Any", vae).decode(latent).movedim(-1, 1)
            std = torch.std(audio, dim=(1, 2), keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            audio /= std
        sample_rate = getattr(
            vae,
            "audio_sample_rate_output",
            getattr(vae, "audio_sample_rate", 44100),
        )
        return cls.outputs(audio={"waveform": audio, "sample_rate": sample_rate})


FORK_NODES: tuple[type[Node], ...] = (
    GenerationLoadCheckpoint,
    GenerationLoadModelPatch,
    GenerationApplyMiniMaxH3FunControlNet,
    GenerationLoadDiffusionModel,
    NativeLoadClip,
    NativeLoadVae,
    GenerationClipTextEncode,
    GenerationEmptyLatentImage,
    GenerationTemporalWindowPlan,
    GenerationSpatialTilePlan,
    GenerationExplicitWindowPlan,
    GenerationRES4LYFRKBetaSampler,
    GenerationKSampler,
    GenerationVAEDecode,
    NativeBlockSparseAttention,
    NativeMiniMaxH3CacheDIT,
    NativeEmptyMiniMaxH3AV,
    NativeMiniMaxH3T2VAConditioning,
    NativeMiniMaxH3ImageToVideo,
    NativeSeparateAVLatent,
    NativeVAEDecodeAudio,
)
