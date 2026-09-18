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

from . import comfy_execution


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
            comfy_hooks = cast("Any", importlib.import_module("comfy.hooks"))
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
