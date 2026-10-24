"""Native ComfyUI advanced sampler nodes backed by dinkster_inference."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    SAMPLER_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_SAMPLER = TypeExpr.concrete(SAMPLER_TYPE)
_SAMPLER_CATEGORY = "model/sampling/samplers"
_UPSCALE_METHODS = ("bislerp", "nearest-exact", "bilinear", "area", "bicubic")


def _runtime_samplers() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.samplers")


def sample_lcm_upscale(
    model: Any,
    x: Any,
    sigmas: Any,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool | None = None,
    total_upscale: float = 2.0,
    upscale_method: str = "bislerp",
    upscale_steps: int | None = None,
) -> Any:
    """Run the pinned LCM sampler while applying its ComfyUI scale schedule."""
    del disable
    initialize_inference_process()
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
    extra_args = {} if extra_args is None else extra_args

    with torch.no_grad():
        if upscale_steps is None:
            upscale_steps = max(len(sigmas) // 2 + 1, 2)
        else:
            upscale_steps += 1
            upscale_steps = min(upscale_steps, len(sigmas) + 1)
        upscales = numpy.linspace(1.0, total_upscale, upscale_steps)[1:]

        original_shape = x.size()
        s_in = x.new_ones([x.shape[0]])
        for index in range(len(sigmas) - 1):
            denoised = model(x, sigmas[index] * s_in, **extra_args)
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": index,
                        "sigma": sigmas[index],
                        "sigma_hat": sigmas[index],
                        "denoised": denoised,
                    }
                )
            x = denoised
            if index < len(upscales):
                x = common_upscale(
                    x,
                    round(original_shape[-1] * upscales[index]),
                    round(original_shape[-2] * upscales[index]),
                    upscale_method,
                    "disabled",
                )
            if sigmas[index + 1] > 0:
                x += sigmas[index + 1] * torch.randn_like(x)
    return x


def sample_euler_cfg_pp_alternative(
    model: Any,
    x: Any,
    sigmas: Any,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool | None = None,
) -> Any:
    """Run the alternative ComfyUI Euler CFG++ callback composition."""
    del disable
    initialize_inference_process()
    to_d = importlib.import_module("dinkster_inference.k_diffusion.sampling").to_d
    set_post_cfg = importlib.import_module(
        "dinkster_inference.model_patcher"
    ).set_model_options_post_cfg_function
    torch = importlib.import_module("torch")
    extra_args = {} if extra_args is None else extra_args

    with torch.no_grad():
        uncond_denoised = [0]

        def post_cfg_function(args: Mapping[str, Any]) -> Any:
            uncond_denoised[0] = args["uncond_denoised"]
            return args["denoised"]

        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_post_cfg(
            model_options, post_cfg_function, disable_cfg1_optimization=True
        )
        s_in = x.new_ones([x.shape[0]])
        for index in range(len(sigmas) - 1):
            sigma_hat = sigmas[index]
            denoised = model(x, sigma_hat * s_in, **extra_args)
            derivative = to_d(x - denoised + uncond_denoised[0], sigmas[index], denoised)
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": index,
                        "sigma": sigmas[index],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )
            x = x + derivative * (sigmas[index + 1] - sigma_hat)
    return x


def _sampler_schema(
    node_type: str,
    inputs: tuple[InputSpec, ...],
    *,
    category: str = _SAMPLER_CATEGORY,
    display_name: str = "",
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name or node_type,
        category=category,
        description=description,
        inputs=inputs,
        outputs=(OutputSpec("SAMPLER", _SAMPLER),),
    )


class SamplerLCMUpscale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampler_schema(
            "SamplerLCMUpscale",
            (
                InputSpec(
                    "scale_ratio",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.1, max=20.0, step=0.01),
                ),
                InputSpec(
                    "scale_steps",
                    _INT,
                    default=-1,
                    advanced=True,
                    widget=NumberWidget(min=-1, max=1000, step=1),
                ),
                InputSpec(
                    "upscale_method",
                    _COMBO,
                    default="bislerp",
                    widget=ComboWidget(options=_UPSCALE_METHODS),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, scale_ratio: float, scale_steps: int, upscale_method: str
    ) -> Mapping[str, object]:
        steps = None if scale_steps < 0 else scale_steps
        sampler = _runtime_samplers().KSAMPLER(
            sample_lcm_upscale,
            extra_options={
                "total_upscale": scale_ratio,
                "upscale_steps": steps,
                "upscale_method": upscale_method,
            },
        )
        return cls.outputs(SAMPLER=sampler)


class SamplerLCM(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampler_schema(
            "SamplerLCM",
            (
                InputSpec(
                    "s_noise",
                    _FLOAT,
                    default=1.0,
                    doc="Per-step noise multiplier at the first step (1.0 = match training).",
                    widget=NumberWidget(min=0.0, max=64.0, step=0.01),
                ),
                InputSpec(
                    "s_noise_end",
                    _FLOAT,
                    default=1.0,
                    doc=(
                        "Per-step noise multiplier at the last step. Set equal to s_noise for a "
                        "constant schedule."
                    ),
                    widget=NumberWidget(min=0.0, max=64.0, step=0.01),
                ),
                InputSpec(
                    "noise_clip_std",
                    _FLOAT,
                    default=0.0,
                    doc="Clamp per-step noise to +/- N*std. 0 disables.",
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            description=(
                "LCM sampler with tunable per-step noise. s_noise is a multiplier on the "
                "model's training noise scale"
            ),
        )

    @classmethod
    def execute(
        cls, *, s_noise: float, s_noise_end: float, noise_clip_std: float
    ) -> Mapping[str, object]:
        return cls.outputs(
            SAMPLER=_runtime_samplers().ksampler(
                "lcm",
                {
                    "s_noise": float(s_noise),
                    "s_noise_end": float(s_noise_end),
                    "noise_clip_std": float(noise_clip_std),
                },
            )
        )


class SamplerEulerCFGpp(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampler_schema(
            "SamplerEulerCFGpp",
            (
                InputSpec(
                    "version",
                    _COMBO,
                    default="regular",
                    advanced=True,
                    widget=ComboWidget(options=("regular", "alternative")),
                ),
            ),
            category="experimental",
            display_name="SamplerEulerCFG++",
        )

    @classmethod
    def execute(cls, *, version: str) -> Mapping[str, object]:
        samplers = _runtime_samplers()
        sampler = (
            samplers.KSAMPLER(sample_euler_cfg_pp_alternative)
            if version == "alternative"
            else samplers.ksampler("euler_cfg_pp")
        )
        return cls.outputs(SAMPLER=sampler)


ADVANCED_SAMPLER_NODES = (SamplerLCMUpscale, SamplerLCM, SamplerEulerCFGpp)
