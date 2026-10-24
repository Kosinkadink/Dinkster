"""Noise and frequency guidance patches applied by the inference sampler."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)


class EpsilonScaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Epsilon Scaling",
            category="model/patch/unet",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scaling_factor",
                    _FLOAT,
                    default=1.005,
                    advanced=True,
                    widget=NumberWidget(min=0.5, max=1.5, step=0.001, display="number"),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, scaling_factor: float) -> Mapping[str, object]:
        if scaling_factor == 0:
            scaling_factor = 1e-9

        def scale_noise(args: dict[str, Any]) -> Any:
            x = args["input"]
            return x - (x - args["denoised"]) / scaling_factor

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(scale_noise)
        return cls.outputs(MODEL=patched)


class TemporalScoreRescaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TemporalScoreRescaling",
            display_name="TSR - Temporal Score Rescaling",
            category="model/patch/unet",
            description=(
                "[Post-CFG Function]\nTSR - Temporal Score Rescaling (2510.01184)\n\n"
                "Rescaling the model's score or noise to steer the sampling diversity.\n"
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "tsr_k",
                    _FLOAT,
                    default=0.95,
                    advanced=True,
                    doc=(
                        "Controls the rescaling strength.\nLower k produces more detailed results; "
                        "higher k produces smoother results in image generation. "
                        "Setting k = 1 disables rescaling."
                    ),
                    widget=NumberWidget(min=0.01, max=100.0, step=0.001, display="number"),
                ),
                InputSpec(
                    "tsr_sigma",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    doc=(
                        "Controls how early rescaling takes effect.\n"
                        "Larger values take effect earlier."
                    ),
                    widget=NumberWidget(min=0.01, max=100.0, step=0.001, display="number"),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="patched_model"),),
        )

    @classmethod
    def execute(cls, *, model: Any, tsr_k: float, tsr_sigma: float) -> Mapping[str, object]:
        variance = tsr_sigma**2

        def rescale(args: dict[str, Any]) -> Any:
            sigma = args["sigma"]
            denoised = args["denoised"]
            if tsr_k == 1 or sigma == 0:
                return denoised
            torch = importlib.import_module("torch")
            sampling = args["model"].current_patcher.get_model_object("model_sampling")
            initialize_inference_process()
            kernels = importlib.import_module("dinkster_inference.k_diffusion.sampling")
            half_log_snr = kernels.sigma_to_half_log_snr(sigma, sampling)
            snr = (2 * half_log_snr).exp()
            if snr == 0:
                return denoised
            ratio = (snr * variance + 1) / (snr * variance / tsr_k + 1)
            ratio = torch.where(torch.isposinf(snr), tsr_k, ratio)
            alpha = sigma * half_log_snr.exp()
            return torch.lerp(args["input"] / alpha, denoised, ratio)

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(rescale)
        return cls.outputs(MODEL=patched)


def fourier_filter(x: Any, scale_low: float, scale_high: float, freq_cutoff: int) -> Any:
    torch = importlib.import_module("torch")
    fft = torch.fft
    dtype, device = x.dtype, x.device
    frequency = fft.fftshift(fft.fftn(x.to(torch.float32), dim=(-2, -1)), dim=(-2, -1))
    mask = torch.ones(frequency.shape, device=device) * scale_high
    center = mask
    for offset in range(2):
        dim = len(frequency.shape) - 2 + offset
        midpoint = frequency.shape[dim] // 2
        cutoff = min(freq_cutoff, midpoint)
        center = center.narrow(dim, midpoint - cutoff, cutoff * 2)
    center[:] = scale_low
    return fft.ifftn(fft.ifftshift(frequency * mask, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype)


class FreSca(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FreSca",
            display_name="FreSca",
            search_terms=("frequency guidance",),
            category="experimental",
            description="Applies frequency-dependent scaling to the guidance",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scale_low",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    doc="Scaling factor for low-frequency components",
                    widget=NumberWidget(min=0, max=10, step=0.01),
                ),
                InputSpec(
                    "scale_high",
                    _FLOAT,
                    default=1.25,
                    advanced=True,
                    doc="Scaling factor for high-frequency components",
                    widget=NumberWidget(min=0, max=10, step=0.01),
                ),
                InputSpec(
                    "freq_cutoff",
                    TypeExpr.concrete(CORE_INT),
                    default=20,
                    advanced=True,
                    doc="Number of frequency indices around center to consider as low-frequency",
                    widget=NumberWidget(min=1, max=10000, step=1),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, scale_low: float, scale_high: float, freq_cutoff: int
    ) -> Mapping[str, object]:
        def filter_guidance(args: dict[str, Any]) -> Any:
            outputs = args["conds_out"]
            if len(outputs) <= 1 or None in args["conds"][:2]:
                return outputs
            cond, uncond = outputs[:2]
            filtered = fourier_filter(cond - uncond, scale_low, scale_high, freq_cutoff)
            return [filtered + uncond, uncond] + outputs[2:]

        patched = model.clone()
        patched.set_model_sampler_pre_cfg_function(filter_guidance)
        return cls.outputs(MODEL=patched)


NOISE_GUIDANCE_NODES = (EpsilonScaling, TemporalScoreRescaling, FreSca)
