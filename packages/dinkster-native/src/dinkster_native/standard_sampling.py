"""Standard ComfyUI sampler execution backed by dinkster_inference."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_nodes_generation.nodes import KSampler, KSamplerAdvanced


def _unwrap_conditioning(value: object) -> object:
    payload = getattr(value, "_dinkster_resident_payload", None)
    return getattr(payload, "conditioning", value)


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


def _sampling_model(model: object) -> object:
    from dinkster_workers import current_execution_context

    context = current_execution_context()
    if context is None or context.attention_route_token is None:
        return model
    runtime = cast("Any", context.attention_runtime)
    sampling_model = cast("Any", model).clone()
    sampling_model.set_model_optimized_attention(runtime.for_model(context.attention_route_token))
    if runtime.distributed_active():
        from .multigpu import configure_distributed_model

        configure_distributed_model(sampling_model)
    return sampling_model


def _sample(
    model: object,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: object,
    negative: object,
    latent_image: object,
    *,
    denoise: float,
    disable_noise: bool,
    start_step: int | None,
    last_step: int | None,
    force_full_denoise: bool,
) -> Mapping[str, object]:
    if not isinstance(latent_image, Mapping):
        raise TypeError("latent_image must be a mapping")

    torch = cast("Any", importlib.import_module("torch"))
    sample = cast("Any", importlib.import_module("dinkster_inference.sample"))
    sampling_model = _sampling_model(model)
    source = dict(cast("Mapping[object, object]", latent_image))
    latent, roles = _fork_samples(source["samples"])
    latent = sample.fix_empty_latent_channels(
        sampling_model,
        latent,
        source.get("downscale_ratio_spacial"),
        source.get("downscale_ratio_temporal"),
    )
    noise = (
        sample.prepare_empty_noise(latent)
        if disable_noise
        else sample.prepare_noise(latent, seed, source.get("batch_index"))
    )
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
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=source.get("noise_mask"),
            callback=None,
            disable_pbar=True,
            seed=seed,
        )
    source.pop("downscale_ratio_spacial", None)
    source.pop("downscale_ratio_temporal", None)
    source["samples"] = _dinkster_samples(output, roles)
    return cast("Mapping[str, object]", source)


class StandardKSampler(KSampler):
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
    ) -> Mapping[str, object]:
        if conditioning_batching != "auto" or max_fused_lanes != 2:
            raise ValueError("standard sampling only supports default conditioning batching")
        return cls.outputs(
            latent=_sample(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
            )
        )


class StandardKSamplerAdvanced(KSamplerAdvanced):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
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
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if conditioning_batching != "auto" or max_fused_lanes != 2:
            raise ValueError("standard sampling only supports default conditioning batching")
        return cls.outputs(
            latent=_sample(
                model,
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=1.0,
                disable_noise=add_noise == "disable",
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=return_with_leftover_noise != "enable",
            )
        )


STANDARD_SAMPLING_NODES = (StandardKSampler, StandardKSamplerAdvanced)
