"""Geometry helpers required by native MoGe inference."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs]


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return normalized view-plane UV coordinates."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1.0 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def intrinsics_from_focal_center(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
) -> torch.Tensor:
    """Assemble intrinsics from broadcastable focal lengths and centers."""
    fx, fy, cx, cy = [torch.as_tensor(value) for value in (fx, fy, cx, cy)]
    fx, fy, cx, cy = torch.broadcast_tensors(fx, fy, cx, cy)
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)
    return torch.stack(
        [
            torch.stack([fx, zero, cx], dim=-1),
            torch.stack([zero, fy, cy], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Back-project normalized image depth through the camera intrinsics."""
    height, width = depth.shape[-2:]
    device, dtype = depth.device, depth.dtype
    u = (torch.arange(width, dtype=dtype, device=device) + 0.5) / width
    v = (torch.arange(height, dtype=dtype, device=device) + 0.5) / height
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=-1)
    rays = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(intrinsics), pixels)
    return rays * depth.unsqueeze(-1)


def _solve_optimal_shift(
    uv: np.ndarray[Any, Any],
    xyz: np.ndarray[Any, Any],
    focal: float | None = None,
) -> tuple[float, float]:
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        projected = xy / (z + shift)[:, None]
        scale = (
            focal
            if focal is not None
            else float((projected * uv).sum() / np.square(projected).sum())
        )
        return cast("np.ndarray[Any, Any]", (scale * projected - uv).ravel())

    solution = least_squares(residual, x0=0.0, ftol=1e-3, method="lm")
    shift = float(np.asarray(solution["x"]).squeeze())
    if focal is None:
        projected = xy / (z + shift)[:, None]
        focal = float((projected * uv).sum() / np.square(projected).sum())
    return shift, focal


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover focal length and Z shift for a predicted point map."""
    shape = points.shape
    height, width = shape[-3], shape[-2]
    points_batched = points.reshape(-1, height, width, 3)
    mask_batched = None if mask is None else mask.reshape(-1, height, width)
    focal_batched = None if focal is None else focal.reshape(-1)
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_low = F.interpolate(
        points_batched.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_low = (
        F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest")
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_low = None
    if mask_batched is not None:
        mask_low = (
            F.interpolate(
                mask_batched.to(torch.float32).unsqueeze(1),
                downsample_size,
                mode="nearest",
            ).squeeze(1)
            > 0
        )

    uv_numpy = uv_low.detach().cpu().numpy()
    points_numpy = points_low.detach().cpu().numpy()
    mask_numpy = None if mask_low is None else mask_low.detach().cpu().numpy()
    focal_numpy = None if focal_batched is None else focal_batched.detach().cpu().numpy()

    recovered_focal: list[float] = []
    recovered_shift: list[float] = []
    for index in range(points_batched.shape[0]):
        if mask_numpy is None:
            xyz = points_numpy[index].reshape(-1, 3)
            selected_uv = uv_numpy.reshape(-1, 2)
        else:
            selected = mask_numpy[index]
            if selected.sum() < 2:
                recovered_focal.append(1.0)
                recovered_shift.append(0.0)
                continue
            xyz = points_numpy[index][selected]
            selected_uv = uv_numpy[selected]
        if focal_numpy is None:
            shift_value, focal_value = _solve_optimal_shift(selected_uv, xyz)
            recovered_focal.append(focal_value)
        else:
            shift_value, _ = _solve_optimal_shift(selected_uv, xyz, focal=float(focal_numpy[index]))
        recovered_shift.append(shift_value)

    shift_tensor = torch.tensor(recovered_shift, device=points.device, dtype=points.dtype).reshape(
        shape[:-3]
    )
    if focal is None:
        focal_tensor = torch.tensor(
            recovered_focal, device=points.device, dtype=points.dtype
        ).reshape(shape[:-3])
    else:
        focal_tensor = focal.reshape(shape[:-3])
    return focal_tensor, shift_tensor
