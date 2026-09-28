"""ComfyUI sampling and device eviction for the compatibility execution arm.

Upstream modules are resolved only during execution, never while inspecting
schemas. Native execution does not call this bridge.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

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
    sample_module = cast("Any", importlib.import_module("dinkster_inference.sample"))
    samplers = cast("Any", importlib.import_module("dinkster_inference.samplers"))
    model_management = cast("Any", importlib.import_module("dinkster_inference.model_management"))
    utils = cast("Any", importlib.import_module("dinkster_inference.utils"))

    # A prior VAE residency transition can make the next load take a
    # numerically different path. Normalize to the complete offload state so
    # cold, warm, and fresh-process runs agree.
    model_management.unload_model_and_clones(model)
    latent_image = sample_module.fix_empty_latent_channels(
        model,
        latent_image,
        compat_latent.get("downscale_ratio_spacial"),
        compat_latent.get("downscale_ratio_temporal"),
    )
    if roles:
        nested_type = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor
        if type(latent_image) is not nested_type:
            raise TypeError("multi-stream LATENT samples must be an exact NestedTensor")
        streams = tuple(latent_image.unbind())
        if len(streams) != len(roles):
            raise ValueError("multi-stream LATENT stream count does not match its role sidecar")
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
    samples = samplers.sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        cast("Any", model).load_device,
        samplers.sampler_object(comfy_sampler.sampler),
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
        mm = cast("Any", importlib.import_module("dinkster_inference.model_management"))
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
