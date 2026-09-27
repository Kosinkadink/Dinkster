"""ComfyUI-backed sampling execution bodies."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_inference import SamplingSegment
from dinkster_native import native as native_implementation
from dinkster_native.native import SCHEDULED_HOOKS_KEY, ScheduledHooks
from dinkster_native.native_arm import GenerationKSampler as _GenerationKSampler
from dinkster_native.native_arm import GenerationKSamplerAdvanced as _GenerationKSamplerAdvanced
from dinkster_native.native_residency import NativeRuntimeHandle

from . import comfy_execution
from .translate import from_comfy_multistream, to_comfy_multistream


def _comfy_sampling_choice(value: str) -> str:
    for namespace in ("dinkster.", "res4lyf."):
        if value.startswith(namespace):
            return value.removeprefix(namespace)
    return value


def _require_default_conditioning_batching(
    conditioning_batching: object, max_fused_lanes: int
) -> None:
    if conditioning_batching != "auto" or max_fused_lanes != 2:
        raise ValueError(
            "Comfy-backed sampling only supports conditioning_batching='auto' "
            "with max_fused_lanes=2"
        )


def calculate_sigmas(
    model: object, scheduler: str, steps: int, denoise: float
) -> tuple[float, ...]:
    torch = cast("Any", importlib.import_module("torch"))
    samplers = cast("Any", importlib.import_module("dinkster_comfy.samplers"))
    if denoise <= 0.0:
        return ()
    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = samplers.calculate_sigmas(
        cast("Any", model).get_model_object("model_sampling"),
        _comfy_sampling_choice(scheduler),
        total_steps,
    ).cpu()
    sigmas = sigmas[-(steps + 1) :]
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError("dinkster-comfy scheduler must return a torch.Tensor")
    return tuple(float(value) for value in sigmas)


def sample_custom(
    *,
    model: object,
    seed: int | None,
    conditioning: object,
    negative: object | None,
    cfg: float,
    sampler_name: str,
    sigmas: tuple[float, ...],
    latent: object,
) -> tuple[dict[object, object], dict[object, object]]:
    torch = cast("Any", importlib.import_module("torch"))
    sample = cast("Any", importlib.import_module("dinkster_comfy.sample"))
    samplers = cast("Any", importlib.import_module("dinkster_comfy.samplers"))
    model_management = cast("Any", importlib.import_module("dinkster_comfy.model_management"))
    if not isinstance(latent, Mapping):
        raise TypeError("latent_image must be a mapping containing 'samples'")
    compat_value = to_comfy_multistream(cast("Mapping[object, object]", latent))
    if not isinstance(compat_value, Mapping):
        raise TypeError("translated latent must remain a mapping")
    compat_latent = dict(cast("Mapping[object, object]", compat_value))
    latent_image = compat_latent["samples"]
    latent_image = sample.fix_empty_latent_channels(
        model,
        latent_image,
        compat_latent.get("downscale_ratio_spacial"),
        compat_latent.get("downscale_ratio_temporal"),
    )
    compat_latent["samples"] = latent_image
    batch_inds = compat_latent.get("batch_index")
    noise = (
        sample.prepare_empty_noise(latent_image)
        if seed is None
        else sample.prepare_noise(latent_image, seed, batch_inds)
    )
    guider = samplers.CFGGuider(model)
    positive = _compat_conditioning_hooks(conditioning, "positive")
    if negative is None:
        guider.inner_set_conds({"positive": positive})
    else:
        guider.set_conds(positive, _compat_conditioning_hooks(negative, "negative"))
    guider.set_cfg(cfg)
    x0_output: dict[str, object] = {}

    def callback(_step: int, x0: object, _x: object, _total_steps: int) -> None:
        x0_output["x0"] = x0

    samples = guider.sample(
        noise,
        latent_image,
        samplers.sampler_object(_comfy_sampling_choice(sampler_name)),
        torch.tensor(sigmas, dtype=torch.float32),
        denoise_mask=compat_latent.get("noise_mask"),
        callback=callback,
        disable_pbar=True,
        seed=0 if seed is None else seed,
    ).to(model_management.intermediate_device())
    output = dict(compat_latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    denoised = dict(output)
    denoised["samples"] = x0_output.get("x0", samples)
    return (
        cast("dict[object, object]", from_comfy_multistream(output)),
        cast("dict[object, object]", from_comfy_multistream(denoised)),
    )


def _compat_conditioning_hooks(value: object, input_id: str) -> object:
    original: object = value
    if not isinstance(value, list | tuple):
        return original
    result: list[list[object]] = []
    for raw_entry in cast("list[object] | tuple[object, ...]", value):
        if not isinstance(raw_entry, list | tuple):
            return original
        entry: list[object] = list(cast("list[object] | tuple[object, ...]", raw_entry))
        if len(entry) != 2 or not isinstance(entry[1], Mapping):
            return original
        metadata = dict(cast("Mapping[object, object]", entry[1]))
        declarations = metadata.pop(SCHEDULED_HOOKS_KEY, None)
        if declarations is not None:
            if "hooks" in metadata:
                raise ValueError(
                    f"{input_id} conditioning contains both native and compatibility hooks"
                )
            if not isinstance(declarations, ScheduledHooks):
                raise TypeError(f"{input_id} scheduled hooks are malformed")
            comfy_hooks = cast("Any", importlib.import_module("dinkster_comfy.hooks"))
            hooks = comfy_hooks.HookGroup()
            load_lora_file = native_implementation.__dict__["_load_lora_file"]
            for declaration in declarations.loras:
                lora, _metadata = load_lora_file(declaration.lora)
                current = comfy_hooks.create_hook_lora(
                    lora=lora,
                    strength_model=declaration.strength_model,
                    strength_clip=declaration.strength_clip,
                )
                if declaration.keyframes is not None:
                    keyframes = comfy_hooks.HookKeyframeGroup()
                    for start, strength in declaration.keyframes.points:
                        keyframes.add(
                            comfy_hooks.HookKeyframe(
                                strength, start_percent=start, guarantee_steps=0
                            )
                        )
                    current.set_keyframes_on_hooks(keyframes)
                hooks = hooks.clone_and_combine(current)
            metadata["hooks"] = hooks
        entry[1] = metadata
        result.append(entry)
    return result


class KSampler(_GenerationKSampler):
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
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
        segment: SamplingSegment | None = None,
    ) -> Mapping[str, object]:
        if isinstance(model, NativeRuntimeHandle):
            return super().execute(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                conditioning_batching=conditioning_batching,
                max_fused_lanes=max_fused_lanes,
                segment=segment,
            )
        del segment
        _require_default_conditioning_batching(conditioning_batching, max_fused_lanes)
        for name, value, low, high in (
            ("seed", seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("denoise", denoise, 0.0, 1.0),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        sampler_name = _comfy_sampling_choice(sampler_name)
        scheduler = _comfy_sampling_choice(scheduler)
        output = comfy_execution.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            _compat_conditioning_hooks(positive, "positive"),
            _compat_conditioning_hooks(negative, "negative"),
            latent_image,
            denoise=denoise,
        )
        return cls.outputs(latent=output)


class KSamplerAdvanced(_GenerationKSamplerAdvanced):
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
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if isinstance(model, NativeRuntimeHandle):
            return super().execute(
                model=model,
                add_noise=add_noise,
                noise_seed=noise_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                start_at_step=start_at_step,
                end_at_step=end_at_step,
                return_with_leftover_noise=return_with_leftover_noise,
                conditioning_batching=conditioning_batching,
                max_fused_lanes=max_fused_lanes,
            )
        _require_default_conditioning_batching(conditioning_batching, max_fused_lanes)
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
        sampler_name = _comfy_sampling_choice(sampler_name)
        scheduler = _comfy_sampling_choice(scheduler)
        output = comfy_execution.common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            _compat_conditioning_hooks(positive, "positive"),
            _compat_conditioning_hooks(negative, "negative"),
            latent_image,
            denoise=1.0,
            disable_noise=add_noise == "disable",
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=return_with_leftover_noise == "disable",
        )
        return cls.outputs(latent=output)


COMFY_SAMPLING_NODES = (KSampler, KSamplerAdvanced)
