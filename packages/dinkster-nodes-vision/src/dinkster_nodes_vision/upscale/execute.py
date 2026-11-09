"""Tiled model upscaling over BHWC image batches.

The tiling algorithm reproduces ComfyUI's ``tiled_scale_multidim``
(commit a1079ba1, ``comfy/utils.py``) specialized to two spatial
dimensions: overlapping tiles, a linear feather mask on every tile edge,
and mask-weighted accumulation, so tile seams blend identically to the
reference. Tile size is honored exactly - there is no out-of-memory
retry loop, keeping execution deterministic for a given graph.

Each tile goes through the reference's per-call model contract (spandrel
``ImageModelDescriptor.__call__``): pad the tile right/bottom to satisfy
the model's minimum and multiple-of size requirements (reflect, then
replicate for any remainder), run the forward pass, clamp the result to
[0, 1], and crop the padding back off.
The per-tile clamp happens before feather blending, so tiles that
overshoot [0, 1] blend exactly as they do in the reference; clamping only
the blended image diverges wherever overlapping tiles disagree around the
clamp boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from .loader import LoadedUpscaler, upscaler_for_asset


def tiled_scale_2d(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile: int,
    overlap: int,
    scale: int,
    out_channels: int,
) -> torch.Tensor:
    """Apply ``function`` (a BCHW -> BCHW model at ``scale``) to overlapping
    ``tile`` x ``tile`` crops and feather-blend the results."""
    batch, _, height, width = samples.shape
    output = torch.empty(
        (batch, out_channels, height * scale, width * scale),
        dtype=samples.dtype,
    )
    for index in range(batch):
        sample = samples[index : index + 1]
        if height <= tile and width <= tile:
            output[index : index + 1] = function(sample)
            continue
        out = output[index : index + 1].zero_()
        out_div = torch.zeros((1, 1, height * scale, width * scale), dtype=samples.dtype)
        positions = [
            range(0, size - overlap, tile - overlap) if size > tile else [0]
            for size in (height, width)
        ]
        for it in product(*positions):
            crop = sample
            upscaled: list[int] = []
            for dim, size in enumerate((height, width)):
                position = max(0, min(size - overlap, it[dim]))
                length = min(tile, size - position)
                crop = crop.narrow(dim + 2, position, length)
                upscaled.append(position * scale)
            tile_output = function(crop)
            mask = torch.ones((1, 1, *tile_output.shape[2:]), dtype=samples.dtype)
            feather = scale * overlap
            for dim in (2, 3):
                if feather >= mask.shape[dim]:
                    continue
                for step in range(feather):
                    weight = (step + 1) / feather
                    mask.narrow(dim, step, 1).mul_(weight)
                    mask.narrow(dim, mask.shape[dim] - 1 - step, 1).mul_(weight)
            target = out
            target_div = out_div
            for dim in range(2):
                length = min(tile_output.shape[dim + 2], target.shape[dim + 2] - upscaled[dim])
                target = target.narrow(dim + 2, upscaled[dim], length)
                target_div = target_div.narrow(dim + 2, upscaled[dim], length)
                if length < tile_output.shape[dim + 2]:
                    tile_output = tile_output.narrow(dim + 2, 0, length)
                    mask = mask.narrow(dim + 2, 0, length)
            target.add_(tile_output * mask)
            target_div.add_(mask)
        out.div_(out_div)
    return output


def _padded_extent(size: int, minimum: int, multiple_of: int) -> int:
    """Smallest extent >= ``size`` satisfying the model's size requirements,
    reproducing spandrel SizeRequirements.get_padding."""
    extent = max(minimum, size)
    remainder = extent % multiple_of
    return extent if not remainder else extent + multiple_of - remainder


def _apply_model(model: LoadedUpscaler, tile: torch.Tensor) -> torch.Tensor:
    """Run one BCHW tile through ``model`` under the reference per-call
    contract: pad right/bottom to satisfy the model's minimum and
    multiple-of size requirements, forward, clamp to [0, 1], and crop the
    padding back off."""
    height, width = tile.shape[2], tile.shape[3]
    pad_h = _padded_extent(height, model.minimum, model.multiple_of) - height
    pad_w = _padded_extent(width, model.minimum, model.multiple_of) - width
    if pad_h or pad_w:
        reflect_h = min(pad_h, height - 1)
        reflect_w = min(pad_w, width - 1)
        tile = F.pad(tile, (0, reflect_w, 0, reflect_h), mode="reflect")
        tile = F.pad(tile, (0, pad_w - reflect_w, 0, pad_h - reflect_h), mode="replicate")
    output = model.module(tile).clamp_(0.0, 1.0)
    if pad_h or pad_w:
        output = output[..., : height * model.scale, : width * model.scale]
    return output


def _batch_for_model(image: object, model: LoadedUpscaler) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or min(array.shape[:3]) < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("model upscaling requires finite pixel values")
    batch = torch.from_numpy(np.ascontiguousarray(array)).movedim(-1, -3)
    channels = batch.shape[1]
    if channels == model.in_channels:
        return batch
    if channels == 1 and model.in_channels == 3:
        return batch.repeat(1, 3, 1, 1)
    raise ValueError(f"image has {channels} channels but the model expects {model.in_channels}")


def execute_upscale(
    image: object,
    upscale_model: object,
    *,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    from dinkster_api.v1 import AssetRef

    if not isinstance(upscale_model, AssetRef):
        raise TypeError(f"upscale_model must be an asset, got {type(upscale_model).__name__}")
    if not 0 <= overlap < tile_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < tile_size, got {overlap} and {tile_size}"
        )
    model = upscaler_for_asset(upscale_model)
    batch = _batch_for_model(image, model)
    with torch.no_grad():
        upscaled = tiled_scale_2d(
            batch,
            lambda tile: _apply_model(model, tile),
            tile=tile_size,
            overlap=overlap,
            scale=model.scale,
            out_channels=model.out_channels,
        )
    result = torch.clamp(upscaled.movedim(-3, -1), 0.0, 1.0)
    return cast("np.ndarray", np.ascontiguousarray(result.numpy(), dtype=np.float32))


__all__ = ["execute_upscale", "tiled_scale_2d"]
