"""Inspire-pack variation noise: seeded batch noise with a blended
variation stream.

Ports prepare_noise, mix_noise, and slerp from ComfyUI-Inspire-Pack
inspire/libs/utils.py @ d23db9aa544de9a6d4c609cb7005fa9e0d42031d,
reproducing the reference's draw order exactly: the variation latent
is drawn FIRST from its own seeded generator, then base noise is
drawn from the base-seed generator, so neither stream perturbs the
other. Every generator here is private (denoise.py discipline); the
reference's global torch.manual_seed calls produce the same draw
sequence.

Scope, fixed by the goldens this module is verified against:

- Rank-4 BCHW latents only; other ranks refuse loudly.
- Draws are float32 then cast to the latent dtype (house style),
  while the reference draws directly at the latent dtype. These are
  bit-identical for float32 latents, which is the verified scope; a
  non-float32 latent caller must first resolve the draw-at-dtype
  difference against the reference and extend the goldens.
- CPU draws only. The reference's noise_device="gpu" path uses the
  global CUDA RNG, which is machine-nondeterministic and fails the
  documented-RNG criterion, so it is not ported.
- prepare_noise + mix_noise semantics only. The reference's
  apply_variation_noise mask blending is a separate surface.

Reference quirks preserved as-is:

- With noise_inds set, batch_seed_mode "incremental" and
  "variation str inc:*" fall through to the comfy path, and the
  comfy noise_inds path never applies variation even at a positive
  strength.
- "variation str inc:*" mixes even at strength 0.0, so slerp's
  trigonometric identity (not an exact passthrough) applies to the
  first batch item.
- The linear mix divides by sqrt((1-s)^2 + s^2) exactly as the
  reference does; reciprocal multiplication is not bit-identical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .denoise import prepare_noise_from_generator

VARIATION_METHODS = ("linear", "slerp")
_INCREMENT_PREFIX = "variation str inc:"


class NoiseVariationError(ValueError):
    """A variation-noise request outside the verified reference scope."""


def _slerp(strength: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Per-row spherical interpolation (reference slerp), including the
    reference's zeroing of non-finite normalized rows: without it,
    degenerate zero-norm rows diverge from the reference bytes."""

    dims = low.shape
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    return (
        (torch.sin((1.0 - strength) * omega) / so).unsqueeze(1) * low
        + (torch.sin(strength * omega) / so).unsqueeze(1) * high
    ).reshape(dims)


def mix_variation_noise(
    from_noise: torch.Tensor,
    to_noise: torch.Tensor,
    strength: float,
    variation_method: str,
) -> torch.Tensor:
    """Reference mix_noise: slerp, or a variance-corrected linear blend."""

    if variation_method not in VARIATION_METHODS:
        raise NoiseVariationError(f"unknown variation method {variation_method!r}")
    if variation_method == "slerp":
        return _slerp(strength, from_noise, to_noise)
    mixed = (1 - strength) * from_noise + strength * to_noise
    mixed /= math.sqrt((1 - strength) ** 2 + strength**2)
    return mixed


def _draw_one_batch(latent: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    return prepare_noise_from_generator(latent[:1], generator)


def _parse_increment_step(batch_seed_mode: str) -> float:
    try:
        return float(batch_seed_mode[len(_INCREMENT_PREFIX) :])
    except ValueError:
        raise NoiseVariationError(
            f"unparseable variation increment in batch seed mode {batch_seed_mode!r}"
        ) from None


def prepare_variation_noise(
    latent: torch.Tensor,
    seed: int,
    *,
    noise_inds: Sequence[int] | None = None,
    batch_seed_mode: str = "comfy",
    variation_seed: int = 0,
    variation_strength: float = 0.0,
    variation_method: str = "linear",
) -> torch.Tensor:
    """Reference prepare_noise on private CPU generators.

    The latent supplies shape, dtype, and layout only; its values
    never enter the result.
    """

    if latent.ndim != 4:
        raise NoiseVariationError("variation noise requires a rank-4 BCHW latent")
    if variation_method not in VARIATION_METHODS:
        raise NoiseVariationError(f"unknown variation method {variation_method!r}")
    incremental_variation = batch_seed_mode.startswith(_INCREMENT_PREFIX)
    if batch_seed_mode not in ("comfy", "incremental") and not incremental_variation:
        raise NoiseVariationError(f"unknown batch seed mode {batch_seed_mode!r}")
    increment_step = _parse_increment_step(batch_seed_mode) if incremental_variation else 0.0

    variation_latent: torch.Tensor | None = None
    if variation_strength > 0 or incremental_variation:
        variation_latent = _draw_one_batch(latent, variation_seed)

    def apply_variation(input_latent: torch.Tensor, strength_up: float = 0.0) -> torch.Tensor:
        if variation_latent is None:
            return input_latent
        variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
        return mix_variation_noise(
            input_latent,
            variation_noise,
            variation_strength + strength_up,
            variation_method,
        )

    if noise_inds is None and batch_seed_mode == "incremental":
        return torch.cat(
            tuple(
                apply_variation(_draw_one_batch(latent, seed + i)) for i in range(latent.shape[0])
            ),
            dim=0,
        )

    if noise_inds is None and incremental_variation:
        return torch.cat(
            tuple(
                apply_variation(_draw_one_batch(latent, seed), increment_step * i)
                for i in range(latent.shape[0])
            ),
            dim=0,
        )

    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    noise = prepare_noise_from_generator(latent, generator, noise_inds)
    if noise_inds is None:
        noise = apply_variation(noise)
    return noise
