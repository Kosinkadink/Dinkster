"""Shared latent parsing for native family adapters."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def _latent_samples(value: object, torch: Any, inference: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    samples = cast("Mapping[object, object]", value).get("samples")
    if type(samples) is torch.Tensor:
        return samples, None
    if type(samples) is inference.MultiStreamLatent:
        return samples, samples
    raise TypeError(f"{name} samples must be a tensor or MultiStreamLatent")
