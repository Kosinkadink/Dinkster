"""Native model-sampling and model-patching nodes."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_MAX_RESOLUTION = 16384


def _float(name: str, default: float, *, advanced: bool = False) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=0.0, max=100.0, step=0.01),
    )


def _sigma(name: str, default: float) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=True,
        widget=NumberWidget(min=0.0, max=1000.0, step=0.001),
    )


def _combo(
    name: str, options: tuple[str, ...], default: str, *, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        advanced=advanced,
        widget=ComboWidget(tuple(ComboOption(value) for value in options)),
    )


def _sampling_schema(node_type: str, category: str, inputs: tuple[InputSpec, ...]) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        category=category,
        inputs=(InputSpec("model", _MODEL), *inputs),
        outputs=(OutputSpec("MODEL", _MODEL),),
    )


def _model_config(model: Any) -> Any:
    return model.model.model_config


def _patch_sampling(model: Any, sampling: Any) -> Any:
    result = model.clone()
    result.add_object_patch("model_sampling", sampling)
    return result


class ModelSamplingDiscrete(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingDiscrete",
            "model/patch",
            (
                _combo(
                    "sampling",
                    ("eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"),
                    "eps",
                ),
                InputSpec("zsnr", _BOOLEAN, default=False, advanced=True),
            ),
        )

    @classmethod
    def execute(cls, *, model: Any, sampling: str, zsnr: bool) -> Mapping[str, object]:
        model_sampling = importlib.import_module("dinkster_inference.model_sampling")
        torch = importlib.import_module("torch")
        bases: dict[str, tuple[Any, Any]] = {
            "eps": (model_sampling.ModelSamplingDiscrete, model_sampling.EPS),
            "v_prediction": (model_sampling.ModelSamplingDiscrete, model_sampling.V_PREDICTION),
            "x0": (model_sampling.ModelSamplingDiscrete, model_sampling.X0),
            "img_to_img": (model_sampling.ModelSamplingDiscrete, model_sampling.IMG_TO_IMG),
            "img_to_img_flow": (
                model_sampling.ModelSamplingDiscrete,
                model_sampling.IMG_TO_IMG_FLOW,
            ),
        }
        if sampling == "lcm":

            class LCM(model_sampling.EPS):
                def calculate_denoised(
                    self, sigma: Any, model_output: Any, model_input: Any
                ) -> Any:
                    timestep = self.timestep(sigma).view(
                        sigma.shape[:1] + (1,) * (model_output.ndim - 1)
                    )
                    sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
                    x0 = model_input - model_output * sigma
                    scaled_timestep = timestep * 10.0
                    c_skip = 0.25 / (scaled_timestep**2 + 0.25)
                    c_out = scaled_timestep / (scaled_timestep**2 + 0.25).sqrt()
                    return c_out * x0 + c_skip * model_input

            class Distilled(model_sampling.ModelSamplingDiscrete):
                original_timesteps = 50

                def timestep(self, sigma: Any) -> Any:
                    distances = sigma.log().to(self.log_sigmas.device) - self.log_sigmas[:, None]
                    return (
                        distances.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps
                        + self.skip_steps
                        - 1
                    ).to(sigma.device)

                def sigma(self, timestep: Any) -> Any:
                    t = torch.clamp(
                        (timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1))
                        / self.skip_steps,
                        min=0,
                        max=len(self.sigmas) - 1,
                    )
                    low, high, weight = t.floor().long(), t.ceil().long(), t.frac()
                    return (
                        ((1 - weight) * self.log_sigmas[low] + weight * self.log_sigmas[high])
                        .exp()
                        .to(timestep.device)
                    )

            base, prediction = Distilled, LCM
        else:
            base, prediction = bases[sampling]
        advanced = type("ModelSamplingAdvanced", (base, prediction), {})
        patched: Any = advanced(_model_config(model), zsnr=zsnr)
        if sampling == "lcm":
            original_timesteps = 50
            patched.skip_steps = patched.num_timesteps // original_timesteps
            sigmas = torch.zeros((original_timesteps), dtype=torch.float32)
            for index in range(original_timesteps):
                sigmas[original_timesteps - 1 - index] = patched.sigmas[
                    patched.num_timesteps - 1 - index * patched.skip_steps
                ]
            patched.set_sigmas(sigmas)
        return cls.outputs(MODEL=_patch_sampling(model, patched))


class ModelSamplingContinuousEDM(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingContinuousEDM",
            "model/patch",
            (
                _combo(
                    "sampling",
                    ("v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"),
                    "v_prediction",
                ),
                _sigma("sigma_max", 120.0),
                _sigma("sigma_min", 0.002),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, sampling: str, sigma_max: float, sigma_min: float
    ) -> Mapping[str, object]:
        modules = importlib.import_module("dinkster_inference.model_sampling")
        base = (
            modules.ModelSamplingCosmosRFlow
            if sampling == "cosmos_rflow"
            else modules.ModelSamplingContinuousEDM
        )
        prediction = {
            "eps": modules.EPS,
            "edm": modules.EDM,
            "v_prediction": modules.V_PREDICTION,
            "edm_playground_v2.5": modules.EDM,
            "cosmos_rflow": modules.COSMOS_RFLOW,
        }[sampling]
        advanced = type("ModelSamplingAdvanced", (base, prediction), {})
        result = model.clone()
        patched = advanced(_model_config(model))
        patched.set_parameters(
            sigma_min, sigma_max, 0.5 if sampling in {"edm", "edm_playground_v2.5"} else 1.0
        )
        result.add_object_patch("model_sampling", patched)
        if sampling == "edm_playground_v2.5":
            result.add_object_patch(
                "latent_format",
                importlib.import_module("dinkster_inference.latent_formats").SDXL_Playground_2_5(),
            )
        return cls.outputs(MODEL=result)


class ModelSamplingContinuousV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingContinuousV",
            "model/patch",
            (
                _combo("sampling", ("v_prediction",), "v_prediction"),
                _sigma("sigma_max", 500.0),
                _sigma("sigma_min", 0.03),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, sampling: str, sigma_max: float, sigma_min: float
    ) -> Mapping[str, object]:
        modules = importlib.import_module("dinkster_inference.model_sampling")
        advanced = type(
            "ModelSamplingAdvanced", (modules.ModelSamplingContinuousV, modules.V_PREDICTION), {}
        )
        patched = advanced(_model_config(model))
        patched.set_parameters(sigma_min, sigma_max, 1.0)
        return cls.outputs(MODEL=_patch_sampling(model, patched))


class ModelSamplingStableCascade(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingStableCascade", "model/patch/stable cascade", (_float("shift", 2.0),)
        )

    @classmethod
    def execute(cls, *, model: Any, shift: float) -> Mapping[str, object]:
        modules = importlib.import_module("dinkster_inference.model_sampling")
        advanced = type("ModelSamplingAdvanced", (modules.StableCascadeSampling, modules.EPS), {})
        patched = advanced(_model_config(model))
        patched.set_parameters(shift)
        return cls.outputs(MODEL=_patch_sampling(model, patched))


class ModelSamplingSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingSD3", "model/patch/stable diffusion", (_float("shift", 3.0),)
        )

    @classmethod
    def execute(cls, *, model: Any, shift: float) -> Mapping[str, object]:
        return _flow_patch(model, shift, multiplier=1000)


def _flow_patch(
    model: Any, shift: float, *, multiplier: float, velocity: bool = False
) -> Mapping[str, object]:
    modules = importlib.import_module("dinkster_inference.model_sampling")
    advanced = type(
        "ModelSamplingAdvanced",
        (
            modules.ModelSamplingDiscreteFlow,
            modules.IMG_TO_IMG_VELOCITY if velocity else modules.CONST,
        ),
        {},
    )
    result = model.clone()
    patched = advanced(_model_config(model))
    patched.set_parameters(shift=shift, multiplier=multiplier)
    original = result.get_model_object("model_sampling")
    if hasattr(original, "noise_scale"):
        patched.set_noise_scale(original.noise_scale)
    result.add_object_patch("model_sampling", patched)
    return ModelSamplingSD3.outputs(MODEL=result)


class ModelSamplingAuraFlow(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingAuraFlow",
            "model/patch",
            (
                _float("shift", 1.73),
                _combo("sampling", ("flow", "img_to_img_velocity"), "flow", advanced=True),
            ),
        )

    @classmethod
    def execute(cls, *, model: Any, shift: float, sampling: str = "flow") -> Mapping[str, object]:
        return _flow_patch(model, shift, multiplier=1.0, velocity=sampling == "img_to_img_velocity")


class ModelSamplingFlux(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelSamplingFlux",
            "model/patch/flux",
            (
                _float("max_shift", 1.15, advanced=True),
                _float("base_shift", 0.5, advanced=True),
                InputSpec(
                    "width",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=_MAX_RESOLUTION, step=8),
                ),
                InputSpec(
                    "height",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=16, max=_MAX_RESOLUTION, step=8),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, max_shift: float, base_shift: float, width: int, height: int
    ) -> Mapping[str, object]:
        shift = (
            (width * height / 256) * ((max_shift - base_shift) / 3840)
            + base_shift
            - ((max_shift - base_shift) / 3840) * 256
        )
        modules = importlib.import_module("dinkster_inference.model_sampling")
        advanced = type("ModelSamplingAdvanced", (modules.ModelSamplingFlux, modules.CONST), {})
        patched = advanced(_model_config(model))
        patched.set_parameters(shift=shift)
        return cls.outputs(MODEL=_patch_sampling(model, patched))


class ModelNoiseScale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelNoiseScale",
            "model/patch",
            (
                InputSpec(
                    "noise_scale",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=64.0, step=0.01),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, model: Any, noise_scale: float) -> Mapping[str, object]:
        result = model.clone()
        original = result.get_model_object("model_sampling")
        patched = type(original)(_model_config(result))
        patched.set_parameters(shift=original.shift, multiplier=original.multiplier)
        patched.set_noise_scale(noise_scale)
        result.add_object_patch("model_sampling", patched)
        return cls.outputs(MODEL=result)


class RescaleCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "RescaleCFG",
            "model/patch",
            (
                InputSpec(
                    "multiplier",
                    _FLOAT,
                    default=0.7,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, model: Any, multiplier: float) -> Mapping[str, object]:
        modules = importlib.import_module("dinkster_inference.model_sampling")
        is_flow = isinstance(model.get_model_object("model_sampling"), modules.CONST)
        torch = importlib.import_module("torch")

        def rescale_cfg(args: dict[str, Any]) -> Any:
            x_orig, scale = args["input"], args["cond_scale"]
            if is_flow:
                cond, uncond = args["cond_denoised"], args["uncond_denoised"]
                cfg = uncond + scale * (cond - uncond)
                dims = tuple(range(1, cond.ndim))
                rescaled = cfg * (
                    cond.std(dim=dims, keepdim=True)
                    / cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
                )
                return x_orig - (multiplier * rescaled + (1.0 - multiplier) * cfg)
            cond, uncond, sigma = args["cond"], args["uncond"], args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x = x_orig / (sigma * sigma + 1.0)
            factor = (sigma * sigma + 1.0).sqrt() / sigma
            cond, uncond = (x - (x_orig - cond)) * factor, (x - (x_orig - uncond)) * factor
            cfg = uncond + scale * (cond - uncond)
            rescaled = cfg * (
                torch.std(cond, dim=(1, 2, 3), keepdim=True)
                / torch.std(cfg, dim=(1, 2, 3), keepdim=True)
            )
            return x_orig - (
                x
                - (multiplier * rescaled + (1.0 - multiplier) * cfg)
                * sigma
                / (sigma * sigma + 1.0).sqrt()
            )

        result = model.clone()
        result.set_model_sampler_cfg_function(rescale_cfg)
        return cls.outputs(MODEL=result)


class ModelComputeDtype(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelComputeDtype",
            "advanced/debug",
            (_combo("dtype", ("default", "fp32", "fp16", "bf16"), "default", advanced=True),),
        )

    @classmethod
    def execute(cls, *, model: Any, dtype: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        result = model.clone()
        result.set_model_compute_dtype(
            {"default": None, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
                dtype
            ]
        )
        return cls.outputs(MODEL=result)


class ModelAttentionBackend(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _sampling_schema(
            "ModelAttentionBackend",
            "model/patch",
            (_combo("attention", ("pytorch attention",), "pytorch attention"),),
        )

    @classmethod
    def execute(cls, *, model: Any, attention: str) -> Mapping[str, object]:
        module = importlib.import_module("dinkster_inference.ldm.modules.attention")
        function = module.get_attention_function(
            {"pytorch attention": "pytorch", "comfy kitchen attention": "comfy_kitchen_int8"}.get(
                attention
            ),
            None,
        )
        if function is None:
            logging.warning(
                "Attention backend '%s' is unavailable; using PyTorch attention.", attention
            )
            function = module.get_attention_function("pytorch")
        result = model.clone()
        result.set_model_optimized_attention(function)
        return cls.outputs(MODEL=result)


MODEL_ADVANCED_NODES = (
    ModelSamplingDiscrete,
    ModelSamplingContinuousEDM,
    ModelSamplingContinuousV,
    ModelSamplingStableCascade,
    ModelSamplingSD3,
    ModelSamplingAuraFlow,
    ModelSamplingFlux,
    ModelNoiseScale,
    RescaleCFG,
    ModelComputeDtype,
    ModelAttentionBackend,
)
