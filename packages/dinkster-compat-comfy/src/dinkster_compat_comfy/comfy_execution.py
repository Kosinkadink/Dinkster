"""ComfyUI sampling and device eviction for the compatibility execution arm.

Upstream modules are resolved only during execution, never while inspecting
schemas. Native execution does not call this bridge.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_schema import report_progress
from dinkster_workers import current_execution_context

from .preview_emit import comfy_sampling_preview_emitter, preview_stage
from .translate import MULTI_STREAM_ROLES_KEY, from_comfy_multistream, to_comfy_multistream


def _compat_stream_roles(latent: Mapping[object, object]) -> tuple[str, ...]:
    sidecar = latent.get(MULTI_STREAM_ROLES_KEY)
    if sidecar is None:
        return ()
    if not isinstance(sidecar, Mapping):
        raise TypeError("multi-stream LATENT role sidecar must be a mapping")
    sidecar_map = cast("Mapping[object, object]", sidecar)
    if sidecar_map.get("version") != 1:
        raise TypeError("multi-stream LATENT role sidecar must use version 1")
    roles = sidecar_map.get("roles")
    if not isinstance(roles, (list, tuple)):
        raise TypeError("multi-stream LATENT role sidecar must contain roles")
    normalized = tuple(cast("Sequence[object]", roles))
    if not normalized or any(type(role) is not str or not role for role in normalized):
        raise TypeError("multi-stream LATENT roles must be nonempty strings")
    return cast("tuple[str, ...]", normalized)


def _snapshot_compat_state(
    value: object,
    roles: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    unpack_latents: Any,
) -> object:
    torch = importlib.import_module("torch")
    if type(value) is not torch.Tensor:
        raise TypeError("ComfyUI sampler callback state must be an exact torch.Tensor")
    snapshot = cast("Any", value).detach().clone()
    if not roles:
        return snapshot
    payloads = tuple(unpack_latents(snapshot, shapes))
    if len(payloads) != len(roles):
        raise ValueError("ComfyUI callback stream count does not match its role sidecar")
    multi_stream = importlib.import_module("dinkster_inference").MultiStreamLatent
    return multi_stream.from_pairs(zip(roles, payloads, strict=True))


def common_ksampler(
    model: object,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: object,
    negative: object,
    latent: object,
    *,
    denoise: float,
    disable_noise: bool = False,
    start_step: int | None = None,
    last_step: int | None = None,
    force_full_denoise: bool = False,
) -> object:
    compat_latent = cast("Mapping[object, object]", to_comfy_multistream(latent))
    roles = _compat_stream_roles(compat_latent)
    latent_image = compat_latent["samples"]
    sample_module = cast("Any", importlib.import_module("comfy.sample"))
    samplers = cast("Any", importlib.import_module("comfy.samplers"))
    model_management = cast("Any", importlib.import_module("comfy.model_management"))
    utils = cast("Any", importlib.import_module("comfy.utils"))

    latent_image = sample_module.fix_empty_latent_channels(
        model,
        latent_image,
        compat_latent.get("downscale_ratio_spacial"),
        compat_latent.get("downscale_ratio_temporal"),
    )
    stream_shapes: tuple[tuple[int, ...], ...] = ()
    if roles:
        nested_type = importlib.import_module("comfy.nested_tensor").NestedTensor
        if type(latent_image) is not nested_type:
            raise TypeError("multi-stream LATENT samples must be an exact NestedTensor")
        streams = tuple(latent_image.unbind())
        if len(streams) != len(roles):
            raise ValueError("multi-stream LATENT stream count does not match its role sidecar")
        stream_shapes = tuple(tuple(stream.shape) for stream in streams)
    batch_inds = compat_latent.get("batch_index")
    noise = (
        sample_module.prepare_empty_noise(latent_image)
        if disable_noise
        else sample_module.prepare_noise(latent_image, seed, batch_inds)
    )
    noise_mask = compat_latent.get("noise_mask")
    comfy_sampler = samplers.KSampler(
        model,
        steps=steps,
        device=cast("Any", model).load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=cast("Any", model).model_options,
    )
    sigmas = comfy_sampler.sigmas
    if last_step is not None and last_step < len(sigmas) - 1:
        sigmas = sigmas[: last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0
    if start_step is not None:
        if start_step < len(sigmas) - 1:
            sigmas = sigmas[start_step:]
        else:
            sigmas = sigmas[:0]
    delegate = copy.copy(samplers.sampler_object(comfy_sampler.sampler))
    sampler_function = delegate.sampler_function
    preview = comfy_sampling_preview_emitter(model)
    inference = cast("Any", importlib.import_module("dinkster_inference"))
    execution = current_execution_context()
    cancellation = inference.CancellationToken(
        (lambda: False) if execution is None else execution.cancelled
    )

    def report_step(event: Any) -> None:
        report_progress(int(event.step) + 1, int(event.total))

    def rich_sampler_function(
        model_k: object,
        noise_value: object,
        active_sigmas: Any,
        *,
        extra_args: Any,
        callback: object,
        disable: bool,
        **options: Any,
    ) -> object:
        total = len(active_sigmas) - 1

        def report(info: Mapping[str, object]) -> None:
            step = int(cast("Any", info["i"]))
            sigma = info.get("sigma")
            if sigma is None:
                if not 0 <= step < len(active_sigmas):
                    raise ValueError("ComfyUI callback omitted sigma outside its schedule")
                sigma = active_sigmas[step]
            sigma_value = float(cast("Any", sigma))
            current = _snapshot_compat_state(info["x"], roles, stream_shapes, utils.unpack_latents)
            denoised_value = info.get("denoised")
            denoised = (
                None
                if denoised_value is None
                else _snapshot_compat_state(
                    denoised_value, roles, stream_shapes, utils.unpack_latents
                )
            )
            # Adaptive reports attempted updates; UniPC updates after its first callback.
            phase = (
                "post_update"
                if comfy_sampler.sampler == "dpm_adaptive"
                or comfy_sampler.sampler in ("uni_pc", "uni_pc_bh2")
                and step > 0
                else "pre_update"
            )
            state = inference.SamplingStateEvent(
                step,
                total,
                sigma_value,
                phase,
                current,
                denoised,
            )

            def on_state(state_event: object) -> None:
                if preview is not None:
                    preview.on_state(state_event)
                if callback is not None:
                    cast("Any", callback)(info)

            progress = inference.ProgressScope(
                cancellation,
                report_step,
                None if preview is None and callback is None else on_state,
            )
            progress.report(inference.StepEvent(step, total, sigma_value), state)

        return sampler_function(
            model_k,
            noise_value,
            active_sigmas,
            extra_args=extra_args,
            callback=report,
            disable=disable,
            **options,
        )

    delegate.sampler_function = rich_sampler_function
    with preview_stage(preview):
        samples = samplers.sample(
            model,
            noise,
            positive,
            negative,
            cfg,
            cast("Any", model).load_device,
            delegate,
            sigmas,
            cast("Any", model).model_options,
            latent_image=latent_image,
            denoise_mask=noise_mask,
            callback=None,
            disable_pbar=not utils.PROGRESS_BAR_ENABLED,
            seed=seed,
        )
    samples = samples.to(
        device=model_management.intermediate_device(),
        dtype=model_management.intermediate_dtype(),
    )
    output = dict(compat_latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return from_comfy_multistream(output)


def model_unload(obj: object) -> None:
    """Evict one resident's device state through ComfyUI's own manager.

    ComfyUI tracks GPU-loaded models as LoadedModel wrappers around
    ModelPatchers in ``comfy.model_management.current_loaded_models``.
    Unloading through that list keeps its bookkeeping consistent - calling
    ``model_unload`` behind its back is what breaks v1. Defensive by
    construction: no comfy, no patcher shape, or no loaded entry all mean
    "nothing to evict here", never an error.
    """
    try:
        mm = cast("Any", importlib.import_module("comfy.model_management"))
    except Exception:  # noqa: BLE001 - not a comfy child
        return
    patcher = getattr(obj, "patcher", obj)  # CLIP/VAE carry one at .patcher
    loaded = cast("list[Any]", getattr(mm, "current_loaded_models", []))
    for entry in list(loaded):
        model = entry.model  # LoadedModel.model is a weakref-backed property
        if model is not patcher and model is not obj:
            continue
        try:
            entry.model_unload()
            loaded.remove(entry)
        except Exception:
            # ResidentPool must retain loaded accounting when this hook fails.
            raise
