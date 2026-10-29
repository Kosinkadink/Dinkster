"""Shared latent validation and tensor-shaping helpers for native nodes."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import Any, Mapping, cast, importlib


def _check_bounds(*constraints: tuple[str, float, float, float]) -> None:
    for name, value, low, high in constraints:
        if not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


def _check_choice(name: str, value: str, options: tuple[str, ...]) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


def _plain_latent(value: object, torch: Any, name: str) -> tuple[dict[Any, Any], Any]:
    """Unwrap a plain-tensor LATENT mapping, copying its metadata keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    mapping = cast("Mapping[Any, Any]", value)
    samples = mapping.get("samples")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    return dict(mapping), samples


def _reshape_latent_to(target_shape: Any, latent: Any, repeat_batch: bool = True) -> Any:
    """Port of comfy_extras.nodes_latent.reshape_latent_to at b78cec87."""
    utils = importlib.import_module("dinkster_inference_torch.resize")
    if latent.shape[1:] != target_shape[1:]:
        latent = utils.common_upscale(
            latent, target_shape[-1], target_shape[-2], "bilinear", "center"
        )
    if repeat_batch:
        return utils.repeat_to_batch_size(latent, target_shape[0])
    return latent


def _composite_masked_tensor(
    destination: Any,
    source: Any,
    x: int,
    y: int,
    mask: Any,
    multiplier: int,
    resize_source: bool,
    torch: Any,
) -> Any:
    """Port of comfy_extras.nodes_mask.composite at b78cec87; mutates destination."""
    utils = importlib.import_module("dinkster_inference_torch.resize")
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )
    source = utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2])

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = utils.repeat_to_batch_size(mask, source.shape[0])

    # Only the source region overlapping the destination is written, so an
    # offset near the edge never writes out of bounds.
    visible_width, visible_height = (
        destination.shape[-1] - left + min(0, x),
        destination.shape[-2] - top + min(0, y),
    )

    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[..., top:bottom, left:right]

    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination
