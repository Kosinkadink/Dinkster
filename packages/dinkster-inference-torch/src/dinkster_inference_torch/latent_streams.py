"""Torch packing and mask normalization for role-labeled latent streams."""

from __future__ import annotations

import math

import torch
from dinkster_inference import (
    LatentMaskMapping,
    LatentPackLayout,
    LatentPackStreamLayout,
    LatentStream,
    MultiStreamLatent,
)

_REDUCTIONS = frozenset(("max", "min", "mean"))
_TEMPORAL_REDUCTIONS = frozenset((*_REDUCTIONS, "first", "last"))


def pack_latent_streams(
    value: MultiStreamLatent[torch.Tensor],
) -> tuple[torch.Tensor, LatentPackLayout]:
    """Flatten ordered streams into one ``[batch, 1, total]`` tensor."""

    if type(value) is not MultiStreamLatent:
        raise TypeError("value must be an exact MultiStreamLatent")
    tensors = tuple(stream.payload for stream in value.streams)
    if any(type(tensor) is not torch.Tensor for tensor in tensors):
        raise TypeError("latent stream payloads must be exact torch.Tensor values")
    first = tensors[0]
    if not first.is_floating_point():
        raise TypeError("latent streams must use floating-point dtypes")
    if first.layout != torch.strided:
        raise ValueError("latent stream packing requires strided tensors")
    entries: list[LatentPackStreamLayout] = []
    flattened: list[torch.Tensor] = []
    offset = 0
    for stream, tensor in zip(value.streams, tensors, strict=True):
        if not tensor.is_floating_point() or tensor.dtype != first.dtype:
            raise TypeError("latent streams must have the same floating-point dtype")
        if tensor.device != first.device:
            raise ValueError("latent streams must be on the same device")
        if tensor.layout != torch.strided:
            raise ValueError("latent stream packing requires strided tensors")
        shape = tuple(tensor.shape)
        if len(shape) < 2 or any(size <= 0 for size in shape):
            raise ValueError("latent streams require positive batch and content dimensions")
        if shape[0] != first.shape[0]:
            raise ValueError("latent stream batch sizes must match")
        elements = math.prod(shape[1:])
        entries.append(LatentPackStreamLayout(stream.role, shape, elements, offset))
        flattened.append(tensor.reshape(shape[0], 1, elements))
        offset += elements
    layout = LatentPackLayout(tuple(entries))
    return torch.cat(flattened, dim=2), layout


def unpack_latent_streams(
    packed: torch.Tensor,
    layout: LatentPackLayout,
) -> MultiStreamLatent[torch.Tensor]:
    """Restore ordered streams from a validated flattened tensor."""

    if type(layout) is not LatentPackLayout:
        raise TypeError("layout must be an exact LatentPackLayout")
    if type(packed) is not torch.Tensor or packed.layout != torch.strided:
        raise ValueError("latent stream unpacking requires a strided tensor")
    if tuple(packed.shape) != layout.packed_shape:
        raise ValueError(f"packed latent shape must be {layout.packed_shape}")
    pieces = packed.split(tuple(stream.elements for stream in layout.streams), dim=2)
    return MultiStreamLatent(
        tuple(
            LatentStream(stream.role, piece.reshape(stream.shape))
            for stream, piece in zip(layout.streams, pieces, strict=True)
        )
    )


def reshape_latent_mask(mask: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Resize a mask and repeat channels and batch rows to a latent shape."""
    dimensions = len(shape) - 2
    if dimensions == 1:
        mode = "linear"
        if mask.ndim < 3:
            mask = mask.reshape((-1, 1, mask.shape[-1]))
    elif dimensions == 2:
        mode = "bilinear"
        if mask.ndim < 4:
            mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    elif dimensions == 3:
        if mask.ndim < 5:
            mask = mask.reshape((1, 1, -1, mask.shape[-2], mask.shape[-1]))
        mode = "trilinear"
    else:
        raise ValueError("latent masks support one, two, or three content dimensions")
    mask = torch.nn.functional.interpolate(mask, size=shape[2:], mode=mode)
    if mask.shape[1] > shape[1]:
        raise ValueError("latent mask channels must not exceed the target channels")
    if mask.shape[1] < shape[1]:
        mask = mask.repeat((1, math.ceil(shape[1] / mask.shape[1])) + (1,) * dimensions)
        mask = mask[:, : shape[1]]
    if mask.shape[0] != shape[0]:
        mask = mask.repeat((math.ceil(shape[0] / mask.shape[0]),) + (1,) * (mask.ndim - 1))
        mask = mask[: shape[0]]
    return mask


def normalize_latent_mask(
    mask: torch.Tensor | MultiStreamLatent[torch.Tensor],
    latent: MultiStreamLatent[torch.Tensor],
) -> MultiStreamLatent[torch.Tensor]:
    """Apply ComfyUI's per-stream mask defaults, resize, and batch fitting."""

    if type(latent) is not MultiStreamLatent:
        raise TypeError("latent must be an exact MultiStreamLatent")
    if type(mask) is torch.Tensor:
        masks = {latent.roles[0]: mask}
    elif type(mask) is MultiStreamLatent:
        masks = {stream.role: stream.payload for stream in mask.streams}
    else:
        raise TypeError("mask must be a tensor or exact MultiStreamLatent")
    normalized: list[tuple[str, torch.Tensor]] = []
    for stream in latent.streams:
        stream_mask = masks.get(stream.role)
        if stream_mask is None:
            stream_mask = torch.ones_like(stream.payload)
        if type(stream_mask) is not torch.Tensor or not stream_mask.is_floating_point():
            raise TypeError("latent masks must be floating torch.Tensor values")
        stream_mask = reshape_latent_mask(
            stream_mask.to(device=stream.payload.device), tuple(stream.payload.shape)
        ).float()
        normalized.append((stream.role, stream_mask))
    return MultiStreamLatent.from_pairs(normalized)


def _validate_mask_values(mask: torch.Tensor) -> None:
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("latent mask values must be finite")
    if float(mask.amin()) < 0.0 or float(mask.amax()) > 1.0:
        raise ValueError("latent mask values must be within [0, 1]")


def _reduce(mask: torch.Tensor, dimensions: int | tuple[int, ...], mode: str) -> torch.Tensor:
    if mode == "max":
        return mask.amax(dim=dimensions)
    if mode == "min":
        return mask.amin(dim=dimensions)
    if mode == "mean":
        return mask.mean(dim=dimensions)
    raise ValueError(f"unknown mask reduction: {mode}")


def content_mask_to_latent_mask(
    mask: torch.Tensor,
    target: torch.Tensor,
    mapping: LatentMaskMapping,
    *,
    spatial_reduction: str = "max",
    temporal_reduction: str = "max",
) -> torch.Tensor:
    """Reduce a pixel-frame mask onto one declared video latent grid."""

    if (
        type(mask) is not torch.Tensor
        or mask.ndim != 3
        or not mask.is_floating_point()
        or mask.layout != torch.strided
        or min(mask.shape) < 1
    ):
        raise TypeError("content mask must be an exact floating [frames,height,width] tensor")
    if (
        type(target) is not torch.Tensor
        or target.ndim != 5
        or not target.is_floating_point()
        or target.layout != torch.strided
        or min(target.shape) < 1
    ):
        raise TypeError("target must be an exact nonempty floating [B,C,T,H,W] tensor")
    if type(mapping) is not LatentMaskMapping or mapping.spatial_downscale is None:
        raise TypeError("video mask conversion requires an exact spatial LatentMaskMapping")
    if spatial_reduction not in _REDUCTIONS:
        raise ValueError(f"unknown spatial mask reduction: {spatial_reduction}")
    if temporal_reduction not in _TEMPORAL_REDUCTIONS:
        raise ValueError(f"unknown temporal mask reduction: {temporal_reduction}")
    _validate_mask_values(mask)

    latent_frames, latent_height, latent_width = target.shape[2:]
    content_frames = mapping.temporal.content_extent(latent_frames)
    downscale = mapping.spatial_downscale
    expected = (content_frames, latent_height * downscale, latent_width * downscale)
    if tuple(mask.shape) != expected:
        raise ValueError(f"content mask shape must be {expected}, got {tuple(mask.shape)}")
    source = mask.to(device=target.device, dtype=torch.float32)
    spatial = source.reshape(
        content_frames,
        latent_height,
        downscale,
        latent_width,
        downscale,
    )
    spatial = _reduce(spatial, (2, 4), spatial_reduction)
    frames: list[torch.Tensor] = []
    for start, stop in mapping.content_ranges(latent_frames):
        group = spatial[start:stop]
        if temporal_reduction == "first":
            frames.append(group[0])
        elif temporal_reduction == "last":
            frames.append(group[-1])
        else:
            frames.append(_reduce(group, 0, temporal_reduction))
    return torch.stack(frames)


def time_ranges_to_latent_mask(
    target: torch.Tensor,
    mapping: LatentMaskMapping,
    ranges: tuple[tuple[float, float], ...],
    *,
    selected: float = 1.0,
    unselected: float = 0.0,
) -> torch.Tensor:
    """Paint second ranges onto a time-last latent role."""

    if (
        type(target) is not torch.Tensor
        or target.ndim not in (3, 4)
        or not target.is_floating_point()
        or target.layout != torch.strided
        or min(target.shape) < 1
    ):
        raise TypeError("timeline target must be an exact nonempty rank-3/rank-4 tensor")
    if type(mapping) is not LatentMaskMapping or mapping.spatial_downscale is not None:
        raise TypeError("timeline mask conversion requires a time-last LatentMaskMapping")
    if (
        type(selected) not in (int, float)
        or type(unselected) not in (int, float)
        or not math.isfinite(selected)
        or not math.isfinite(unselected)
        or not 0.0 <= selected <= 1.0
        or not 0.0 <= unselected <= 1.0
    ):
        raise ValueError("timeline mask values must be within [0, 1]")
    if type(ranges) is not tuple or any(
        type(item) is not tuple or len(item) != 2 or any(type(value) is not float for value in item)
        for item in ranges
    ):
        raise TypeError("time ranges must be exact (float, float) tuples")

    latent_extent = target.shape[-1]
    content_extent = mapping.temporal.content_extent(latent_extent)
    duration = content_extent / mapping.content_rate_hz
    if any(
        not math.isfinite(start)
        or not math.isfinite(stop)
        or start < 0.0
        or stop <= start
        or stop > duration
        for start, stop in ranges
    ):
        raise ValueError(f"time ranges must be within [0, {duration:g}] seconds")
    indexed = tuple(
        (start * mapping.content_rate_hz, stop * mapping.content_rate_hz) for start, stop in ranges
    )
    timeline = torch.full(
        (latent_extent,),
        unselected,
        dtype=torch.float32,
        device=target.device,
    )
    for index, (start, stop) in enumerate(mapping.content_ranges(latent_extent)):
        if any(
            start < selected_stop and stop > selected_start
            for selected_start, selected_stop in indexed
        ):
            timeline[index] = selected
    height = 1 if target.ndim == 3 else target.shape[-2]
    return timeline.reshape(1, 1, latent_extent).expand(1, height, latent_extent).clone()


def latent_mask_preview(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Collapse batch/channel dimensions to a renderable MASK value."""

    if (
        type(mask) is not torch.Tensor
        or type(target) is not torch.Tensor
        or not mask.is_floating_point()
        or not target.is_floating_point()
        or mask.layout != torch.strided
        or target.layout != torch.strided
    ):
        raise TypeError("latent mask preview requires exact strided floating tensors")
    if tuple(mask.shape) != tuple(target.shape) or target.ndim < 3:
        raise ValueError("normalized latent mask must match its target shape")
    collapsed = mask.amax(dim=(0, 1))
    if collapsed.ndim == 1:
        return collapsed.reshape(1, 1, -1)
    if collapsed.ndim == 2:
        return collapsed.unsqueeze(0)
    if collapsed.ndim == 3:
        return collapsed
    raise ValueError("latent mask preview supports one, two, or three content dimensions")


def latent_mask_to_content_mask(
    mask: torch.Tensor,
    target: torch.Tensor,
    mapping: LatentMaskMapping,
) -> torch.Tensor:
    """Expand one normalized video-role mask back to content geometry."""

    if type(mapping) is not LatentMaskMapping or mapping.spatial_downscale is None:
        raise TypeError("content mask expansion requires a spatial LatentMaskMapping")
    preview = latent_mask_preview(mask, target)
    if target.ndim != 5 or preview.ndim != 3:
        raise ValueError("content mask expansion requires a video latent mask")
    groups = mapping.content_ranges(target.shape[2])
    temporal = torch.cat(
        tuple(
            preview[index : index + 1].repeat(stop - start, 1, 1)
            for index, (start, stop) in enumerate(groups)
        )
    )
    downscale = mapping.spatial_downscale
    return temporal.repeat_interleave(downscale, dim=1).repeat_interleave(downscale, dim=2)


def pack_latent_mask(
    mask: torch.Tensor | MultiStreamLatent[torch.Tensor],
    latent: MultiStreamLatent[torch.Tensor],
) -> torch.Tensor:
    """Normalize Comfy masks per role and pack them to solver shape."""

    packed, _ = pack_latent_streams(normalize_latent_mask(mask, latent))
    return packed


__all__ = [
    "content_mask_to_latent_mask",
    "latent_mask_preview",
    "latent_mask_to_content_mask",
    "normalize_latent_mask",
    "pack_latent_mask",
    "pack_latent_streams",
    "time_ranges_to_latent_mask",
    "unpack_latent_streams",
]
