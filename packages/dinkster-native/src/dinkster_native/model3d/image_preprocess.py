"""Image normalization matching ComfyUI clip_model.py at 25dfc16f9ac0."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_preprocess(
    image: torch.Tensor,
    *,
    size: int = 224,
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    crop: bool = True,
) -> torch.Tensor:
    """Resize NHWC images, quantize to the 8-bit grid, and normalize NCHW RGB.

    Preserve the input dtype, including its interpolation support and rounding.
    Extra channels are discarded; values are clamped after resizing, not before.
    """
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if image.shape[2:] != (size, size):
        if crop:
            scale = size / min(image.shape[2], image.shape[3])
            scaled = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scaled = (size, size)
        image = F.interpolate(image, size=scaled, mode="bicubic", antialias=True)
        top = (image.shape[2] - size) // 2
        left = (image.shape[3] - size) // 2
        image = image[:, :, top : top + size, left : left + size]
    image = (255.0 * image).clamp(0, 255).round() / 255.0
    return (image - mean_tensor.view(3, 1, 1)) / std_tensor.view(3, 1, 1)
