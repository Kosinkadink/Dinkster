"""Mask-bounded image compositing, matching ComfyUI 25dfc16f9ac0 nodes_images."""

from __future__ import annotations

import logging
import math

import torch

from .resize import common_upscale


def _crop_image_with_mask(
    item_image: torch.Tensor,
    item_mask: torch.Tensor,
    max_image_size: int,
    pad_factor: float,
    mask_offset: int,
    bg_rgb: tuple[float, float, float],
    aspect_ratio: float,
) -> torch.Tensor:
    img = item_image.permute(2, 0, 1).unsqueeze(0).cpu().float().clamp(0, 1)
    mask = item_mask.unsqueeze(0).unsqueeze(0).cpu().float().clamp(0, 1)
    m2d = mask[0, 0]
    h, w = m2d.shape
    border = torch.cat([m2d[0, :], m2d[-1, :], m2d[:, 0], m2d[:, -1]])
    center = m2d[h // 4 : h - h // 4, w // 4 : w - w // 4]
    if float(border.mean()) > 0.5 and float(center.mean()) < 0.5:
        mask = 1.0 - mask

    if mask_offset > 0:
        r = mask_offset
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=2 * r + 1, stride=1, padding=r)
    elif mask_offset < 0:
        r = -mask_offset
        mask = 1.0 - torch.nn.functional.max_pool2d(
            1.0 - mask, kernel_size=2 * r + 1, stride=1, padding=r
        )
    mask = torch.where(mask < 0.05, torch.zeros_like(mask), mask)

    h, w = img.shape[-2:]
    if max(h, w) > max_image_size:
        scale = max_image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = common_upscale(img, new_w, new_h, "lanczos", "disabled")
        mask = common_upscale(mask, new_w, new_h, "lanczos", "disabled")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        h, w = new_h, new_w

    alpha_u8 = (mask[0, 0].clamp(0, 1) * 255.0).to(torch.uint8)
    fg_pixels = (alpha_u8 > 204).nonzero()
    if fg_pixels.numel() == 0:
        inv_fg = ((255 - alpha_u8) > 204).nonzero()
        if inv_fg.numel() > 0:
            logging.info("Mask bbox empty, using inverted mask.")
            mask = 1.0 - mask
            fg_pixels = inv_fg
    if fg_pixels.numel() > 0:
        y_min, x_min = fg_pixels.min(dim=0).values.tolist()
        y_max, x_max = fg_pixels.max(dim=0).values.tolist()
        center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
        bw, bh = x_max - x_min + 1, y_max - y_min + 1
        if bw / max(bh, 1) >= aspect_ratio:
            crop_w = int(bw * pad_factor)
            crop_h = int(bw / aspect_ratio * pad_factor)
        else:
            crop_h = int(bh * pad_factor)
            crop_w = int(bh * aspect_ratio * pad_factor)
        half_w, half_h = math.ceil(crop_w / 2), math.ceil(crop_h / 2)
        crop_x1, crop_y1 = int(center_x - half_w), int(center_y - half_h)
        crop_x2, crop_y2 = crop_x1 + 2 * half_w, crop_y1 + 2 * half_h
    else:
        logging.warning(
            "Mask for the image is empty; a clean foreground mask is required for best quality."
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w, h

    # Preserve PIL.crop zero-padding semantics before compositing the background.
    pad_l, pad_t = max(0, -crop_x1), max(0, -crop_y1)
    pad_r, pad_b = max(0, crop_x2 - w), max(0, crop_y2 - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = torch.nn.functional.pad(img, (pad_l, pad_r, pad_t, pad_b), value=0.0)
        mask = torch.nn.functional.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=0.0)
        crop_x1 += pad_l
        crop_x2 += pad_l
        crop_y1 += pad_t
        crop_y2 += pad_t
    cropped_img = img[..., crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_mask = mask[..., crop_y1:crop_y2, crop_x1:crop_x2]
    bg = torch.tensor(bg_rgb, dtype=cropped_img.dtype, device=cropped_img.device).view(1, 3, 1, 1)
    return (cropped_img * cropped_mask + bg * (1.0 - cropped_mask)).clamp(0, 1)


def crop_images_to_masks(
    images: torch.Tensor,
    masks: torch.Tensor,
    width: int = 1024,
    height: int = 1024,
    pad_factor: float = 1.0,
    grow_mask: int = 0,
    background: str = "#000000",
) -> torch.Tensor:
    """Return RGB CPU float32 crops with the source's 8-bit Lanczos round trip."""
    color = background.lstrip("#")
    bg_rgb = (
        (int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0)
        if len(color) == 6
        else (0.0, 0.0, 0.0)
    )
    images = images[..., :3]
    batch_size = images.shape[0]
    if masks.shape[0] == 1 and batch_size > 1:
        masks = masks.expand(batch_size, -1, -1)
    elif masks.shape[0] != batch_size:
        raise ValueError(f"Mask batch {masks.shape[0]} does not match image batch {batch_size}")
    if masks.shape[-2:] != images.shape[1:3]:
        masks = common_upscale(
            masks.unsqueeze(1).float(), images.shape[2], images.shape[1], "bilinear", "disabled"
        ).squeeze(1)
    out_images: list[torch.Tensor] = []
    for b in range(batch_size):
        composite = _crop_image_with_mask(
            images[b], masks[b], max(width, height), pad_factor, grow_mask, bg_rgb, width / height
        )
        composite = common_upscale(composite, width, height, "lanczos", "disabled")
        out_images.append(composite.movedim(-3, -1))
    return torch.cat(out_images, dim=0)
