"""Tensor resize semantics from ComfyUI 947c2749dd04c51ef0e21b069544d8b0b4f9b411.

Interpolation, center-crop rounding, batch sampling, and Lanczos's uint8
round trip preserve the upstream operations and their order.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int, dim: int = 0) -> torch.Tensor:
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    if tensor.shape[dim] < batch_size:
        repeats = (
            dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (tensor.ndim - 1 - dim)
        )
        return tensor.repeat(repeats).narrow(dim, 0, batch_size)
    return tensor


def resize_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor
    if batch_size <= 1:
        return tensor[:batch_size]
    output = torch.empty([batch_size, *tensor.shape[1:]], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]
    return output


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    def slerp(b1: torch.Tensor, b2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        c = b1.shape[-1]

        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(1) * b1_normalized + (
            torch.sin(r.squeeze(1) * omega) / so
        ).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(
        length_old: int, length_new: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1)
        )
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1)) + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def lanczos(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [
        Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
        for image in samples
    ]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    tensors = []
    for image in images:
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)
        tensors.append(tensor.movedim(-1, 0) if array.ndim == 3 else tensor)
    return torch.stack(tensors).to(samples.device, samples.dtype)


def common_upscale(
    samples: torch.Tensor, width: int, height: int, upscale_method: str, crop: str
) -> torch.Tensor:
    orig_shape: tuple[int, ...] = tuple(samples.shape)
    orig_rank = len(orig_shape)
    if orig_rank > 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1]
        )
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "bislerp":
        out = bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = lanczos(s, width, height)
    else:
        out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

    if orig_rank == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))
