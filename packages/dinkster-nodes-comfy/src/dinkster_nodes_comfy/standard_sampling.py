"""Native ComfyUI standard sampler nodes backed by dinkster_inference."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .sampler_factories import SAMPLER_NAMES
from .sampling_boundary import to_public, to_runtime
from .sigma_schedulers import SCHEDULER_NAMES

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


def _combo(name: str, options: tuple[str, ...]) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _integer(
    name: str, default: int, minimum: int, maximum: int, *, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum),
    )


def _sample(
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: object,
    *,
    denoise: float,
    disable_noise: bool,
    start_step: int | None,
    last_step: int | None,
    force_full_denoise: bool,
) -> Mapping[str, object]:
    source = materialize_latent(latent_image)
    if not isinstance(source, Mapping):
        raise TypeError("latent_image must be a mapping")
    source = cast(Mapping[str, Any], source)
    public_latent = source["samples"]
    latent = to_runtime(public_latent)
    initialize_inference_process()
    sampling = importlib.import_module("dinkster_inference.sample")
    latent = sampling.fix_empty_latent_channels(
        model,
        latent,
        source.get("downscale_ratio_spacial"),
        source.get("downscale_ratio_temporal"),
    )
    noise = (
        sampling.prepare_empty_noise(latent)
        if disable_noise
        else sampling.prepare_noise(latent, seed, source.get("batch_index"))
    )
    result = sampling.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=to_runtime(source.get("noise_mask")),
        callback=None,
        disable_pbar=True,
        seed=seed,
    )
    output = dict(source)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = to_public(result, public_latent)
    return output


class KSampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="KSampler",
            category="model/sampling",
            description=(
                "Uses the provided model, positive and negative conditioning "
                "to denoise the latent image."
            ),
            search_terms=(
                "sampler",
                "sample",
                "generate",
                "denoise",
                "diffuse",
                "txt2img",
                "img2img",
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        control_after_generate="fixed",
                    ),
                ),
                _integer("steps", 20, 1, 10_000),
                InputSpec(
                    "cfg",
                    _FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                _combo("sampler_name", SAMPLER_NAMES),
                _combo("scheduler", SCHEDULER_NAMES),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent_image", _LATENT),
                InputSpec(
                    "denoise", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT, doc="The denoised latent."),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: object,
        denoise: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            LATENT=_sample(
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


class KSamplerAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="KSamplerAdvanced",
            display_name="KSampler (Advanced)",
            category="model/sampling",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "add_noise",
                    _COMBO,
                    advanced=True,
                    widget=ComboWidget((ComboOption("enable"), ComboOption("disable"))),
                ),
                InputSpec(
                    "noise_seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        control_after_generate="fixed",
                    ),
                ),
                _integer("steps", 20, 1, 10_000),
                InputSpec(
                    "cfg",
                    _FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                _combo("sampler_name", SAMPLER_NAMES),
                _combo("scheduler", SCHEDULER_NAMES),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent_image", _LATENT),
                _integer("start_at_step", 0, 0, 10_000, advanced=True),
                _integer("end_at_step", 10_000, 0, 10_000, advanced=True),
                InputSpec(
                    "return_with_leftover_noise",
                    _COMBO,
                    advanced=True,
                    widget=ComboWidget((ComboOption("disable"), ComboOption("enable"))),
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: object,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
    ) -> Mapping[str, object]:
        return cls.outputs(
            LATENT=_sample(
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


STANDARD_SAMPLING_NODES = (KSampler, KSamplerAdvanced)
