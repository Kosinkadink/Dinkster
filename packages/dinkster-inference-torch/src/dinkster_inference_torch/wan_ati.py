"""Wan ATI trajectory preparation and latent motion projection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F


class WanAtiError(ValueError):
    """The requested Wan ATI trajectory is malformed or incompatible."""


_TRACK_LENGTH = 121


def _point(value: object) -> tuple[float, float]:
    if type(value) is not dict:
        raise WanAtiError("Wan ATI track points must be objects with numeric x and y values")
    point = cast("Mapping[object, object]", value)
    x = point.get("x")
    y = point.get("y")
    if type(x) not in (int, float) or type(y) not in (int, float):
        raise WanAtiError("Wan ATI track points must be objects with numeric x and y values")
    x_value = float(cast("int | float", x))
    y_value = float(cast("int | float", y))
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise WanAtiError("Wan ATI track coordinates must be finite")
    return x_value, y_value


def _is_point(value: object) -> bool:
    return type(value) is dict and "x" in value and "y" in value


def _parse_track_batches(tracks: str) -> list[list[list[object]]]:
    if type(tracks) is not str:
        raise TypeError("Wan ATI tracks must be an exact JSON string")
    try:
        parsed: object = json.loads(tracks.replace("'", '"'))
    except json.JSONDecodeError:
        return []
    if type(parsed) is not list:
        raise WanAtiError("Wan ATI tracks JSON must contain a list")
    values = cast("list[object]", parsed)
    if not values:
        return []
    if all(_is_point(value) for value in values):
        return [[values]]
    if all(
        type(value) is list
        and bool(value)
        and all(_is_point(point) for point in cast("list[object]", value))
        for value in values
    ):
        return [[cast("list[object]", value) for value in values]]
    if all(
        type(batch) is list
        and bool(batch)
        and all(
            type(track) is list
            and bool(track)
            and all(_is_point(point) for point in cast("list[object]", track))
            for track in cast("list[object]", batch)
        )
        for batch in values
    ):
        return [
            [cast("list[object]", track) for track in cast("list[object]", batch)]
            for batch in values
        ]
    raise WanAtiError("Wan ATI tracks JSON has an unsupported batch or track structure")


def _pad_track(track: Sequence[object]) -> np.ndarray:
    points = np.asarray([(*_point(point), 1.0) for point in track], dtype=np.float32)
    if len(points) < _TRACK_LENGTH:
        points = np.vstack(
            (
                points,
                np.zeros((_TRACK_LENGTH - len(points), 3), dtype=np.float32),
            )
        )
    else:
        points = points[:_TRACK_LENGTH]
    return points.reshape(_TRACK_LENGTH, 1, 3)


def _process_tracks(
    tracks: np.ndarray,
    *,
    width: int,
    height: int,
    frame_count: int,
) -> torch.Tensor:
    tensor = torch.from_numpy(tracks).float()
    if tensor.shape[1] == _TRACK_LENGTH:
        tensor = tensor.permute(1, 0, 2, 3)
    coordinates, visible = tensor[..., :2], tensor[..., 2:3]
    center = torch.tensor([width, height]).type_as(coordinates) / 2
    coordinates = (coordinates - center) / min(width, height) * 2
    visible = visible * 2 - 1
    timeline = torch.linspace(-1, 1, coordinates.shape[0]).view(-1, 1, 1, 1)
    timeline = timeline.expand(*visible.shape)
    processed = torch.cat((timeline, coordinates, visible), dim=-1).view(_TRACK_LENGTH, -1, 4)
    origin = processed[:1]
    remainder = processed[1:]
    divisor = math.gcd(_TRACK_LENGTH - 1, frame_count)
    repeat = frame_count // divisor
    stride = (_TRACK_LENGTH - 1) // divisor
    remainder = torch.repeat_interleave(remainder, repeat, dim=0)[1::stride]
    return torch.cat((origin, remainder), dim=0)


def _resize_track_batch(tracks: list[torch.Tensor], batch_size: int) -> tuple[torch.Tensor, ...]:
    source_batch = len(tracks)
    if source_batch == batch_size or source_batch == 0:
        return tuple(tracks)
    if batch_size <= 1:
        return tuple(tracks[:batch_size])
    output: list[torch.Tensor] = []
    if batch_size < source_batch:
        scale = (source_batch - 1) / (batch_size - 1)
        for index in range(batch_size):
            output.append(tracks[min(round(index * scale), source_batch - 1)])
    else:
        scale = source_batch / batch_size
        for index in range(batch_size):
            output.append(tracks[min(math.floor((index + 0.5) * scale), source_batch - 1)])
    return tuple(output)


def prepare_wan_ati_tracks(
    tracks: str,
    *,
    width: int,
    height: int,
    length: int,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    """Parse and normalize ComfyUI WanTrackToVideo trajectory JSON."""
    if width <= 0 or height <= 0:
        raise WanAtiError("Wan ATI frame dimensions must be positive")
    if length <= 0:
        raise WanAtiError("Wan ATI video length must be positive")
    if batch_size <= 0:
        raise WanAtiError("Wan ATI batch size must be positive")
    batches = _parse_track_batches(tracks)
    prepared = [
        _process_tracks(
            np.stack(tuple(_pad_track(track) for track in batch), axis=0),
            width=width,
            height=height,
            frame_count=length - 1,
        ).unsqueeze(0)
        for batch in batches
    ]
    return _resize_track_batch(prepared, batch_size)


def _select_vertices(
    target: torch.Tensor,
    indices: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    target = target.expand(
        *tuple(
            [indices.shape[index] if target.shape[index] == 1 else -1 for index in range(dim)]
            + [-1] * (target.ndim - dim)
        )
    )
    padded = indices
    if target.ndim > dim + 1:
        for _ in range(target.ndim - (dim + 1)):
            padded = padded.unsqueeze(-1)
        padded = padded.expand(*([-1] * (dim + 1)), *target.shape[dim + 1 :])
    return torch.gather(target, dim=dim, index=padded)


def _merge_vertices(
    attributes: torch.Tensor,
    weights: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    target_dim = assignments.ndim - 1
    if attributes.ndim == 2:
        shape = [1] * target_dim + list(attributes.shape)
        selected = _select_vertices(attributes.reshape(shape), assignments.long(), dim=target_dim)
    else:
        shape = [attributes.shape[0], *([1] * (target_dim - 1)), *attributes.shape[1:]]
        selected = _select_vertices(attributes.reshape(shape), assignments.long(), dim=target_dim)
    return torch.sum(selected * weights.unsqueeze(-1), dim=-2)


def _patch_motion_single(
    tracks: torch.Tensor,
    video: torch.Tensor,
    *,
    temperature: float,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, temporal, height, width = video.shape
    track_count = tracks.shape[2]
    _, coordinates, visible = torch.split(tracks, [1, 2, 1], dim=-1)
    normalized = coordinates / torch.tensor(
        [width / min(height, width), height / min(height, width)],
        device=coordinates.device,
    )
    normalized = normalized.clamp(-1, 1)
    visible = visible.clamp(0, 1)
    x_axis = torch.linspace(-width / min(height, width), width / min(height, width), width)
    y_axis = torch.linspace(-height / min(height, width), height / min(height, width), height)
    grid = torch.stack(torch.meshgrid(y_axis, x_axis, indexing="ij")[::-1], dim=-1).to(
        coordinates.device
    )
    coordinate_tail = coordinates[:, 1:]
    visible_tail = visible[:, 1:]
    aligned_visibility = visible_tail.view(temporal - 1, 4, *visible_tail.shape[2:]).sum(1)
    aligned_tracks = (coordinate_tail * visible_tail).view(
        temporal - 1, 4, *coordinate_tail.shape[2:]
    ).sum(1) / (aligned_visibility + 1e-5)
    distance = ((aligned_tracks[:, None, None] - grid[None, :, :, None]).pow(2)).sum(-1)
    weight = torch.exp(-distance * temperature) * aligned_visibility.clamp(0, 1).view(
        temporal - 1, 1, 1, track_count
    )
    vertex_weight, vertex_index = torch.topk(
        weight,
        k=min(topk, weight.shape[-1]),
        dim=-1,
    )
    point_feature = F.grid_sample(
        video.permute(1, 0, 2, 3)[:1],
        normalized[:, :1].type(video.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0)
    output_feature = _merge_vertices(point_feature, vertex_weight, vertex_index).permute(3, 0, 1, 2)
    output_weight = vertex_weight.sum(-1)
    mixed = output_feature + video[:, 1:] * (1 - output_weight.clamp(0, 1))
    feature = torch.cat((video[:, :1], mixed), dim=1)
    first_mask = torch.ones(
        (1, height, width),
        dtype=output_weight.dtype,
        device=output_weight.device,
    )
    mask = torch.cat((first_mask, output_weight), dim=0)
    return mask[None].expand(4, -1, -1, -1), feature


def patch_wan_ati_motion(
    tracks: Sequence[torch.Tensor],
    video: torch.Tensor,
    *,
    temperature: float = 220.0,
    topk: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Wan ATI point trajectories through one model-space video latent."""
    if (
        type(video) is not torch.Tensor
        or not video.is_floating_point()
        or video.layout != torch.strided
    ):
        raise TypeError("Wan ATI video must be an exact strided floating torch.Tensor")
    if video.ndim != 5 or video.shape[1] != 16 or any(size <= 0 for size in video.shape):
        raise WanAtiError("Wan ATI video must be nonempty [B,16,T,H,W]")
    if len(tracks) != video.shape[0]:
        raise WanAtiError("Wan ATI trajectory and video batches must match")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise WanAtiError("Wan ATI temperature must be finite and positive")
    if type(topk) is not int or topk <= 0:
        raise WanAtiError("Wan ATI topk must be a positive integer")
    expected_frames = (video.shape[2] - 1) * 4 + 1
    normalized_tracks: list[torch.Tensor] = []
    for value in tracks:
        if (
            type(value) is not torch.Tensor
            or not value.is_floating_point()
            or value.layout != torch.strided
        ):
            raise TypeError("Wan ATI tracks must be exact strided floating torch.Tensor values")
        if (
            value.ndim != 4
            or value.shape[0] != 1
            or value.shape[1] != expected_frames
            or value.shape[2] <= 0
            or value.shape[3] != 4
        ):
            raise WanAtiError(f"Wan ATI tracks must have shape [1,{expected_frames},points,4]")
        normalized_tracks.append(value.to(device=video.device))
    outputs = tuple(
        _patch_motion_single(
            track,
            video[index],
            temperature=float(temperature),
            topk=topk,
        )
        for index, track in enumerate(normalized_tracks)
    )
    return (
        torch.stack(tuple(mask for mask, _feature in outputs), dim=0),
        torch.stack(tuple(feature for _mask, feature in outputs), dim=0),
    )


__all__ = [
    "WanAtiError",
    "patch_wan_ati_motion",
    "prepare_wan_ati_tracks",
]
