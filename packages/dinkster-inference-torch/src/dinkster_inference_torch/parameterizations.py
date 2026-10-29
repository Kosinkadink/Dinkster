"""Prediction parameterizations: what a diffusion core's output means.

The typed port of ComfyUI's model_sampling *prediction* mixins (EPS /
V_PREDICTION / EDM / CONST / X0, comfy/model_sampling.py @ b78cec87) as
explicit dispatch over the Parameterization enum - the plan rejects
the reference's dynamic multiple inheritance. The *schedule* half
lives in spaces.py.

Sigma enters as a plain float: one noise level per call. The reference
broadcasts a per-batch sigma tensor instead (reshape_sigma); Dinkster's
denoiser assembly owns batching and lifts these per-sigma formulas
over the batch, so the math here stays scalar-in, tensor-out. Tensor
demands are pure arithmetic (ArithTensor) - no torch import.

FLOW is the reference's CONST: x = sigma * noise + (1 - sigma) * latent
flow matching. ``noise_scale`` is CONST's optional attribute of the
same name (default 1.0).
"""

from __future__ import annotations

import math

from dinkster_inference.sampling import Parameterization, TensorT, is_flow_parameterization


def calculate_input(
    parameterization: Parameterization,
    sigma: float,
    noise: TensorT,
    *,
    sigma_data: float = 1.0,
) -> TensorT:
    """Precondition the model input for ``sigma`` (calculate_input
    @ b78cec87)."""
    p = parameterization
    if is_flow_parameterization(p):
        return noise
    # EPS, V_PREDICTION, EDM, X0 share the EPS input scaling.
    return noise * (1.0 / math.sqrt(sigma * sigma + sigma_data * sigma_data))


def calculate_denoised(
    parameterization: Parameterization,
    sigma: float,
    model_output: TensorT,
    model_input: TensorT,
    *,
    sigma_data: float = 1.0,
) -> TensorT:
    """Convert the raw model output into the denoised prediction
    (calculate_denoised @ b78cec87)."""
    p = parameterization
    if p in (Parameterization.X0, Parameterization.IMAGE_TO_IMAGE_FLOW):
        return model_output
    if p in (Parameterization.EPS, Parameterization.FLOW):
        return model_input - model_output * sigma
    variance = sigma * sigma + sigma_data * sigma_data
    skip = sigma_data * sigma_data / variance
    out = sigma * sigma_data / math.sqrt(variance)
    if p is Parameterization.V_PREDICTION:
        return model_input * skip - model_output * out
    if p is Parameterization.EDM:
        return model_input * skip + model_output * out
    raise ValueError(f"unhandled parameterization: {p!r}")


def noise_scaling(
    parameterization: Parameterization,
    sigma: float,
    noise: TensorT,
    latent: TensorT,
    *,
    max_denoise: bool = False,
    noise_scale: float = 1.0,
) -> TensorT:
    """Combine fresh noise with a latent image at ``sigma`` - the
    img2img/inpaint entry point (noise_scaling @ b78cec87)."""
    if parameterization is Parameterization.IMAGE_TO_IMAGE_FLOW:
        return latent
    if parameterization is Parameterization.FLOW:
        return noise * (sigma * noise_scale) + latent * (1.0 - sigma)
    scale = math.sqrt(1.0 + sigma * sigma) if max_denoise else sigma
    return noise * scale + latent


def inverse_noise_scaling(
    parameterization: Parameterization,
    sigma: float,
    latent: TensorT,
) -> TensorT:
    """Undo noise_scaling at the final sigma (inverse_noise_scaling
    @ b78cec87). Identity except for flow models."""
    if parameterization is Parameterization.IMAGE_TO_IMAGE_FLOW:
        return latent * -1.0 + 1.0
    if parameterization is Parameterization.FLOW:
        return latent * (1.0 / (1.0 - sigma))
    return latent


__all__ = [
    "calculate_denoised",
    "calculate_input",
    "inverse_noise_scaling",
    "noise_scaling",
]
