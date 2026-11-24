"""CPU arithmetic shared by image graph nodes and bounded timeline evaluation."""

from __future__ import annotations

import numpy as np
from dinkster_values.colors import color_channels
from PIL import Image, ImageDraw, ImageFont


def parse_color(value: str) -> np.ndarray:
    return np.asarray(color_channels(value), dtype=np.float32)


def source_over(
    source: np.ndarray,
    source_opacity: np.ndarray,
    destination: np.ndarray,
    destination_opacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_premultiplied = source * source_opacity
    destination_premultiplied = destination * destination_opacity
    output_opacity = source_opacity + (1.0 - source_opacity) * destination_opacity
    output = source_premultiplied + (1.0 - source_opacity) * destination_premultiplied
    straight = np.divide(
        output,
        output_opacity,
        out=np.zeros_like(output),
        where=output_opacity > 1e-5,
    )
    return np.clip(straight, 0.0, 1.0), output_opacity


def blend(destination: np.ndarray, source: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return source
    if mode == "multiply":
        return destination * source
    if mode == "screen":
        return 1.0 - (1.0 - destination) * (1.0 - source)
    if mode == "overlay":
        return np.where(
            destination <= 0.5,
            2.0 * destination * source,
            1.0 - 2.0 * (1.0 - destination) * (1.0 - source),
        )
    if mode == "soft_light":
        curve = np.where(
            destination <= 0.25,
            ((16.0 * destination - 12.0) * destination + 4.0) * destination,
            np.sqrt(np.clip(destination, 0.0, None)),
        )
        return np.where(
            source <= 0.5,
            destination - (1.0 - 2.0 * source) * destination * (1.0 - destination),
            destination + (2.0 * source - 1.0) * (curve - destination),
        )
    if mode == "signed_difference":
        return destination - source
    raise ValueError(f"unknown blend mode: {mode}")


def composite_samples(
    destination: np.ndarray, blended: np.ndarray, alpha: float | np.ndarray
) -> np.ndarray:
    return destination * (1.0 - alpha) + blended * alpha


def masked_transition(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = mask[:, :, None]
    if first.shape[-1] != 4 or second.shape[-1] != 4:
        return first * (1.0 - weight) + second * weight
    first_alpha, second_alpha = first[..., 3:], second[..., 3:]
    alpha = first_alpha * (1.0 - weight) + second_alpha * weight
    premultiplied = (
        first[..., :3] * first_alpha * (1.0 - weight) + second[..., :3] * second_alpha * weight
    )
    rgb = np.divide(
        premultiplied,
        alpha,
        out=np.zeros_like(premultiplied),
        where=alpha != 0,
    )
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2), dtype=np.float32)


def overlay_rgba(
    image: np.ndarray, alpha: np.ndarray, rgba: np.ndarray, alpha_mode: str = "source_over"
) -> np.ndarray:
    channels = int(image.shape[3])
    output = np.array(image, copy=True)
    if channels == 1:
        luma = float(rgba[0] * 0.2126 + rgba[1] * 0.7152 + rgba[2] * 0.0722)
        output[:, :, :, 0] = luma * alpha + output[:, :, :, 0] * (1.0 - alpha)
    elif channels == 3:
        output = rgba[:3] * alpha[:, :, :, None] + output * (1.0 - alpha[:, :, :, None])
    elif alpha_mode == "source_over":
        destination_alpha = np.clip(output[:, :, :, 3], 0.0, 1.0)
        result_alpha = alpha + destination_alpha * (1.0 - alpha)
        numerator = rgba[:3] * alpha[:, :, :, None] + output[:, :, :, :3] * destination_alpha[
            :, :, :, None
        ] * (1.0 - alpha[:, :, :, None])
        output[:, :, :, :3] = np.divide(
            numerator,
            result_alpha[:, :, :, None],
            out=np.zeros_like(numerator),
            where=result_alpha[:, :, :, None] > 0,
        )
        output[:, :, :, 3] = result_alpha
    elif alpha_mode == "max":
        output[:, :, :, :3] = rgba[:3] * alpha[:, :, :, None] + output[:, :, :, :3] * (
            1.0 - alpha[:, :, :, None]
        )
        output[:, :, :, 3] = np.maximum(output[:, :, :, 3], alpha)
    return np.ascontiguousarray(output, dtype=np.float32)


def text_coverage(
    width: int, height: int, text: str, x: int, y: int, font_size: int, line_spacing: int
) -> np.ndarray:
    if font_size < 1 or line_spacing < 0:
        raise ValueError("font_size must be positive and line_spacing must be non-negative")
    raster = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(raster)
    draw.multiline_text(
        (x, y), text, fill=255, font=ImageFont.load_default(size=font_size), spacing=line_spacing
    )
    return np.asarray(raster, dtype=np.float32)[None, :, :] / 255.0
