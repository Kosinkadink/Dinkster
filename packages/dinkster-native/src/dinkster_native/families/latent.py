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


def _move_multistream_latent(value: Any, device: object) -> Any:
    def move(payload: Any) -> Any:
        return payload.to(device)

    return value.map(move)


def _adapt_multistream_latent(
    value: object,
    runtime: object,
    torch: Any,
    inference: Any,
    name: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping) or "samples" not in value:
        raise TypeError(f"{name} must be a LATENT mapping containing 'samples'")
    latent = cast("Mapping[object, object]", value)
    samples = latent["samples"]
    if type(samples) is inference.MultiStreamLatent:
        return latent
    if not isinstance(runtime, inference.MultiStreamLatentAdapterRuntime):
        raise TypeError(f"{name} cannot be adapted to the model's latent streams")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    adapted = cast("Any", runtime).adapt_multistream_latent(
        samples,
        source_spatial_downscale=latent.get("downscale_ratio_spacial"),
        source_temporal_downscale=latent.get("downscale_ratio_temporal"),
    )
    if type(adapted) is not inference.MultiStreamLatent:
        raise TypeError("latent adaptation must return an exact MultiStreamLatent")
    result = dict(latent)
    result["samples"] = adapted
    return result


def _sampling_memory_requirements(runtime: Any, samples: Any) -> tuple[int, int | None]:
    estimate = getattr(runtime, "sampling_memory_requirements", None)
    return (0, None) if estimate is None else estimate(tuple(samples.shape))
